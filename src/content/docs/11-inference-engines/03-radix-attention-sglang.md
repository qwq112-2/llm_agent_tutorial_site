---
title: "11.3 RadixAttention 与 Prefix Cache（SGLang）"
description: "11.2 的 PagedAttention 解决了\"KV cache 在显存里如何放\"的问题——把 cache 切成 16-token block、按需分配、消除碎片；本节解决\"KV cache 在请求之间如何复用\"的问题——production 里成千上万个请求共享同一段 system prompt（\"You are a helpful assistant...\"几百 token）、同一个 ch"
---

> ⏱ 预计阅读 45 分钟 ｜ 难度 ★★★ ｜ 前置：11.2 PagedAttention 与 Continuous Batching

## 一句话本节讲什么

11.2 的 PagedAttention 解决了"KV cache 在显存里如何放"的问题——把 cache 切成 16-token block、按需分配、消除碎片；本节解决"KV cache 在请求之间如何复用"的问题——production 里成千上万个请求共享同一段 system prompt（"You are a helpful assistant..."几百 token）、同一个 chat session 每轮都把历史重发一遍、同一条 agent trajectory 在 best-of-N 采样里被重复多次——朴素实现每个请求都把这些重复部分**重新 prefill 一遍**，纯重复劳动。**Prefix cache** 的 idea 一句话：**已经算过的 KV cache 按 prefix 存起来，命中时直接复用、跳过 prefill**。**RadixAttention**（Zheng 2024, SGLang）把这件事做到极致——用 **radix tree（trie）** 索引 KV cache，新请求来按 token 序列在树上找最长公共前缀、命中部分 zero-cost reuse、剩余从断点起 prefill；LRU 驱逐近期未用的 prefix。配套的 **SGLang DSL** 让 multi-step / branch / agent rollout 场景下的 prefix 复用从"用户手动 cache"变成"框架自动管"，结构化输出（XGrammar 集成）让 agent / tool calling 的 JSON / Schema 严格约束从 nice-to-have 变成 production-ready。本节把 prefix caching 的动机、RadixAttention 的 trie 数据结构、与 vLLM Automatic Prefix Cache 的颗粒度差异、SGLang 的 DSL 与 structured generation、以及"什么场景该用 vLLM、什么场景该用 SGLang"讲透——是 11.2 之后做"agent / multi-turn / 共享前缀"场景部署的关键一环，也是 14.2 / 15.2 agent 系统与 multi-turn RL 训练 rollout 的推理底座。

---

## 1. Mental model（直觉）

### 1.1 Prefix 重复在 production 的四个典型场景

观察一下你身边的 LLM 应用，下面四类负载贡献了**production 里 60%-90% 的 prefill 算力浪费**：

**场景 1：多 user 共享 system prompt**

OpenAI / Claude / 任何商业 chatbot 后端的 system prompt 都不短——常见 1k-10k token，里面塞了 persona 设定、安全约束、tool 描述、few-shot 示例。每个用户的请求都带着这同一段 system prompt 进来——朴素实现每来一个请求就把这段 prefill 一遍：1k system prompt × 1000 个并发 user × 32 层 attention = 巨量重复 matmul。

**场景 2：Multi-turn chat 的 history 增量**

```
turn 1: [system + user_1]                       → assistant_1
turn 2: [system + user_1 + assistant_1 + user_2] → assistant_2
turn 3: [system + user_1 + assistant_1 + user_2 + assistant_2 + user_3] → ...
```

每一轮的 prompt 都包含前面所有轮——朴素实现每轮都把整段 history 重新 prefill。第 N 轮时，前 N-1 轮的 KV 在上一轮就算过了——本可以"在上一轮的 cache 之上只增量算 user_N 这几十个新 token"。

**场景 3：Agent rollout 的 trajectory 共享**

Agent 一条 trajectory 是 `[system + tools_desc + obs_1 + thought_1 + action_1 + obs_2 + ...]` 这种交替结构，常上千 token。**RL rollout** 时为了估计 advantage，要对同一个 prompt 做 group-of-N 采样（GRPO `n=16`）——16 条 trajectory 共享前面 obs_k 之前的全部前缀，朴素实现 16 次重复 prefill。**Best-of-N / MCTS** 同理：从某个状态出发探索多条 branch，每条 branch 的根节点 KV 是同一份。

**场景 4：RAG 的固定模板**

RAG 的 prompt 模板是 `[system + retrieved_docs + question + answer_format]`——同一个文档库的不同问题共享 system + answer_format 部分；同一个 query 的不同 chunk 检索结果共享 question 部分。

**这四个场景的共性**：用户感知不到的"prompt 重复"在底下持续发生——把这些重复部分的 KV cache 算一次、缓存起来、后续请求直接复用，能省掉 production prefill 的大头。

### 1.2 Prefix cache 的核心 idea

一句话：**KV cache 是 prompt 的纯函数——同样的前缀必然产生同样的 KV cache，因此可以缓存复用**。

```
请求 A: [system_prompt(1000 token)] + [user_a(50 token)]
请求 B: [system_prompt(1000 token)] + [user_b(80 token)]

朴素：
  A: prefill 1050 token → KV_A
  B: prefill 1080 token → KV_B    ← 前 1000 个 token 与 A 完全重复算

Prefix cache:
  A: prefill 1050 token → 把前 1000 个 token 的 KV 存到 cache (key=hash(system_prompt))
                          剩下 50 token 算完拼起来
  B: 查 cache(hash(system_prompt)) → 命中！直接借 A 的前 1000 KV
                          只需 prefill 80 token
  → B 节省 1000 token 的 prefill 算力 (~92%)
```

**关键性质**：
- KV cache 是**前缀的确定性函数**（causal mask 下 token i 的 K, V 只依赖于 token 0..i）——同一段 prefix 的 KV 唯一
- 因此 **cache key 可以就是 token 序列本身（或其 hash）**——不同请求只要 prompt 前缀的 token id 序列完全一致，KV 就可以共享
- 复用是 **bit-exact** 的——和 4.7 §2.5 KV cache 的等价性证明一脉相承，**lossless**

### 1.3 RadixAttention 的数据结构：用 trie 管理 prefix

vLLM 的 Automatic Prefix Cache 是 **block 粒度** 的——把 KV cache 按 16-token block 切，每个 block 计算 hash、用 hashtable 索引。这套机制在"多 user 共享 system prompt"这种**前缀完全相同**的场景下足够好用，但在两种场景下颗粒度太粗：

- **branch 探索**：从同一个 prefix 分叉出多条 trajectory（如 best-of-N），不同 branch 共享前缀但分叉点不一定在 16-token 边界上
- **chat session 的增量增长**：每轮新加几个 token、远不到一个 block 大小

**SGLang 团队（Zheng et al. 2024）**的 insight：**用 radix tree（也叫 compressed trie）按 token 粒度索引 KV cache**，每个树节点对应一段 token 序列对应的 KV cache。新请求来时按它的 token 序列在树上从根开始往下匹配，找到的最深节点就是它能复用的最长前缀。

```
            [BOS, "You", "are", "a", "helpful", "assistant", ".", ...] (KV cached)
                            │
                ┌───────────┴───────────┐
        [user, "What", "is", ...]     [user, "Help", "me", ...]
                │                              │
        ┌───────┴────────┐              ┌──────┴──────┐
   [..., "RAG?"]   [..., "agent?"]   [..., "code"] [..., "debug"]
   (req A 复用)    (req B 复用)       (req C 复用)  (req D 复用)
```

**radix tree 的特性**：
- **节点 = 一段 token 序列 + 该段对应的 KV cache 指针**（指向显存里实际存 KV 的 block 池）
- **边 = token 上的字面匹配**——查找时按 token id 走边
- **路径压缩**：连续单分叉的 token 合并成一个节点（这就是 radix tree 与普通 trie 的区别——节省指针开销）
- **新请求 = 在树上找最长公共前缀**：命中节点对应的 KV 直接借用、剩余 token 从此节点为父建新节点
- **LRU 驱逐**：每个节点带"最后使用时间"，free pool 不够时优先驱逐最久未用的叶子节点

对应到 §1.1 的四个场景：
- **多 user 共享 system prompt** → 树根附近一个粗粗的"主干节点"，所有 user 都从它出发分叉
- **multi-turn chat** → 一条沿着 turn 数往下加深的"链状路径"，新 turn 只是在末尾加一个新节点
- **best-of-N 采样** → 同一个 prefix 节点下分叉出 N 条 branch
- **agent trajectory** → 共享 system + tools_desc + 前几步 obs 的 prefix 树

### 1.4 SGLang vs vLLM Prefix Cache 的对比直觉

| 维度 | vLLM Automatic Prefix Cache | SGLang RadixAttention |
|---|---|---|
| 索引数据结构 | hashtable (block hash → block id) | radix tree (token 序列 → KV 节点) |
| 颗粒度 | block (16 token) | token (精确到每个 token) |
| 共享 prefix 完全相同 | ✓ 命中 | ✓ 命中 |
| 共享 prefix 在 block 中间分叉 | ✗ 错失 | ✓ 命中 |
| Branch 场景（best-of-N / MCTS） | 一般 | **优势明显** |
| Chat 增量 | 大部分场景能命中 | **几乎 100% 命中** |
| 实现复杂度 | 中 | 高 |

**直觉总结**：vLLM 的 prefix cache 像"按章节缓存的图书馆"——只能整章借；SGLang 的 RadixAttention 像"按句子缓存"——更精细，但管理开销也高。在"前缀完全规整对齐"的负载下两者收益相近；在"branch / 增量 / agent rollout"等动态分叉场景下 SGLang 显著更优。

### 1.5 SGLang 的另一面：DSL + 结构化输出

SGLang 不只是一个推理 backend——它附带一个 **Python DSL** 用来编排 multi-step / branch / structured prompts。这个 DSL 让"哪些 step 之间能共享 prefix"变成框架可见的信息，自动开启 RadixAttention 复用。

```python
@sgl.function
def multi_step(s, q):
    s += "Step 1: " + sgl.gen("step1", max_tokens=100)
    s += "Step 2: " + sgl.gen("step2", max_tokens=100)
    s += "Final: " + sgl.gen("final", max_tokens=50)
```

每一步 `gen` 之前的字符串就是一段 prefix——SGLang 会把它们注册到 radix tree、step 2 自动复用 step 1 的全部 prefix（包括 step 1 生成的内容）、step 3 同理。如果用户改写成 OpenAI API 的三次独立调用，需要手动把上一步的输出拼回 prompt 重发——SGLang 把这件事自动化了。

另外 SGLang 把 **XGrammar / Outlines** 集成进了 backend——`sgl.gen(regex=...)` 或 `sgl.gen(grammar=...)` 强制输出符合 regex / EBNF / JSON schema，对 agent / tool calling 场景至关重要（13.4 会详讲 function calling 的 schema 约束）。vLLM 也支持 `guided_decoding`，但 SGLang 集成更深、性能更好。

---

## 2. 公式与原理

### 2.1 Prefix cache 的命中收益模型

设一个请求总长 $T$ token，其中前 $T_p$ token 是已被 cache 的 prefix（命中长度），后 $T - T_p$ token 是新内容。

**朴素 prefill 算力**（无 cache）：
$$C_{\text{naive}} = O(L \cdot d \cdot T^2)$$

**带 prefix cache 的 prefill 算力**：
- 命中部分零开销（cache 直接借）
- 新部分长度 $T - T_p$，需要在已有的 $T_p$ KV 之上算 attention
$$C_{\text{cached}} = O(L \cdot d \cdot (T - T_p)(T - T_p + 2 T_p)) = O(L \cdot d \cdot (T^2 - T_p^2))$$

**命中率定义**：$\rho = T_p / T$

**算力节省比**：
$$\text{saving} = \frac{C_{\text{naive}} - C_{\text{cached}}}{C_{\text{naive}}} = \frac{T_p^2}{T^2} = \rho^2$$

**关键观察**：节省比例是命中率的**平方**——$\rho = 0.9$（命中 90%）→ 节省 81% 算力；$\rho = 0.5$ → 节省 25%；$\rho = 0.1$ → 几乎无收益。**这就是为什么 prefix cache 在"system prompt 长 + 共享率高"场景下收益颠覆性，在"prompt 完全不同"场景下几乎无用**——必须先看清场景再决定开不开。

### 2.2 RadixAttention 的 trie 操作形式化

设全局有一棵 radix tree $\mathcal{T}$，每个节点 $v$ 含：
- $\text{tokens}(v)$：该节点对应的 token 序列（一段，长度可变）
- $\text{kv}(v)$：该段 token 在每一层 attention 的 K, V cache（指针指向显存 block 池）
- $\text{parent}(v)$：父节点（拼接形成完整 prefix）
- $\text{children}(v)$：子节点 dict，按 token id 索引
- $\text{last\_used}(v)$：LRU 时间戳
- $\text{ref\_count}(v)$：当前在用此节点的请求数（>0 时不可驱逐）

**核心操作 1：Match**——给定 token 序列 $\mathbf{t} = [t_0, t_1, \dots, t_{T-1}]$，找最长前缀匹配的节点：

```
def match(T, t):
    v = root
    matched = 0
    while matched < len(t):
        c = v.children.get(t[matched])
        if c is None: break
        # 匹配 c.tokens 与 t[matched:] 的最长公共前缀
        prefix_len = lcp(c.tokens, t[matched:])
        matched += prefix_len
        if prefix_len < len(c.tokens):
            break               # c 内部分叉
        v = c                   # 完整匹配 c，继续往下
    return v, matched           # v 上挂的 KV 全部可复用，matched 之后是新 prefill
```

**核心操作 2：Insert**——新请求 prefill 完后，把 $\mathbf{t}$ 的剩余部分挂到树上（必要时分裂节点）：

- 若 match 时是"完整匹配 v"，则在 v 下创建新子节点（边 = 第一个未匹配 token）
- 若 match 时是"v 内部分叉"，则**分裂** v：把 v 拆成 v_prefix（共享部分）+ v_old_suffix（v 原有的剩余部分）+ v_new_suffix（新请求的剩余部分），保持 KV 引用正确

**核心操作 3：Evict**——LRU 驱逐：

```
def evict(T, n_blocks_needed):
    # 候选：所有 ref_count == 0 的叶子节点，按 last_used 升序
    candidates = sorted([v for v in T.leaves() if v.ref_count == 0],
                        key=lambda v: v.last_used)
    freed = 0
    for v in candidates:
        free(v.kv)               # 把 KV 占用的 block 还给 free pool
        v.parent.children.pop(v.first_token)
        freed += len(v.kv_blocks)
        if freed >= n_blocks_needed: break
```

**正确性保证**（必须满足）：
- 复用 KV 时**位置编码必须一致**——RoPE 下 token i 的 K 已经包含了"位置 i"的旋转，复用时 token i 必须仍是位置 i（这一点对 prefix cache 天然成立——只要前缀完全相同、位置自然对齐）
- ref_count 维护要严格——正在被某请求 attention 访问的节点不能驱逐（否则 attention 读到无效显存）
- Copy-on-Write 风格：分叉时已存在的节点不能被新请求修改

### 2.3 vLLM Automatic Prefix Cache 的 hash-based 实现

为了对比，简单看 vLLM 是怎么做的：

- KV cache 已经按 16-token block 切（PagedAttention，11.2）
- 给每个 block 算一个 hash：$\text{hash}(b) = H(\text{prev\_block\_hash}, \text{tokens in this block})$（chained hash 保证 hash 等价于"从 prompt 起到此 block 末的 token 序列"）
- 全局维护一个 `hashtable: block_hash → physical_block_id`
- 新请求 prefill 时按 16-token chunk 算 hash、查 hashtable——命中就借用现有 block、否则新分配
- LRU 驱逐：refcount==0 的 block 按访问时间淘汰

**与 RadixAttention 的差异**：
- 颗粒度：vLLM 16-token block 粒度对齐；RadixAttention 任意 token 边界
- 数据结构：hashtable vs radix tree——hashtable O(1) 查找更快但只支持精确匹配；radix tree O(prefix length) 查找但支持"找最长公共前缀"
- Branch 场景：vLLM 在分叉点不在 block 边界时会错失共享；RadixAttention 总能找到最长公共前缀

**实战收益**：
- "多 user 共享纯前缀"场景两者收益相近（都能命中绝大多数 prefill）
- "best-of-N / MCTS / chat 增量"等分叉场景 SGLang 通常多 10-30% 命中率
- 对 throughput 的影响：负载越偏 agent / branch 场景，SGLang 优势越明显

### 2.4 Cache 命中率监控公式

production 部署 prefix cache 必须监控**命中率**——否则不知道有没有真省算力。两种统计口径：

**Token-level hit rate**（更直接）：
$$\rho_{\text{token}} = \frac{\sum_{\text{req}} T_p^{\text{req}}}{\sum_{\text{req}} T^{\text{req}}}$$

**Compute saving**（直接对应算力）：
$$\text{saving} = \frac{\sum_{\text{req}} (T_p^{\text{req}})^2}{\sum_{\text{req}} (T^{\text{req}})^2}$$

经验值：
- "system prompt 重 + 短 user" 负载：$\rho_{\text{token}} \approx 0.7-0.9$，saving 可达 50-80%
- "agent rollout"：$\rho_{\text{token}} \approx 0.5-0.8$，saving 30-60%
- "chat session 单 user 长对话"：随 turn 累计，第 N 轮 $\rho \approx (N-1)/N$
- "全新 query / 单次 RAG"：$\rho \approx 0.1-0.3$，saving < 10%——开不开都差不多

### 2.5 SGLang DSL 的执行模型

SGLang DSL 的本质是把"一段带 `gen` 占位符的字符串模板"编译成一个**程序图**，每个 `gen` 是一个节点，节点之间的字符串是 prefix。

```python
@sgl.function
def multi_step(s, q):
    s += "Step 1: " + sgl.gen("step1", max_tokens=100)
    s += "Step 2: " + sgl.gen("step2", max_tokens=100)
```

执行流程：
1. SGLang runtime 接到调用 → 把当前 prompt（"Step 1: "）注册到 radix tree
2. 调用 LLM 生成 step1 → 把 step1 的输出 append 到 prompt 后面
3. 看到下一个 `+= "Step 2: "` → prefix 变成 "Step 1: <step1 output>\nStep 2: "
4. 第二步 prefill 时 radix tree 查找 → 命中前面的全部前缀（自动）
5. 同理 step 3

如果用户写成三次独立 OpenAI API 调用，每次都要把前面的 prompt + output 重发一遍——网络开销 + server 端 prefix cache 命中（如果 server 开了）。SGLang DSL 把这件事**在 client 端就编译成一次会话**，传输与 cache 复用更高效。

---

## 3. 最小代码示例

### 3.1 SGLang 启动 server + 基础调用（≤ 25 行）

```bash
# pip install "sglang[all]"
# 启动 server
python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-7B-Instruct \
    --port 30000 \
    --tp 1 \
    --mem-fraction-static 0.85          # 类比 vLLM 的 gpu_memory_utilization
```

```python
import sglang as sgl

# 注册 backend（指向上面启动的 server）
sgl.set_default_backend(sgl.RuntimeEndpoint("http://localhost:30000"))


@sgl.function
def chat(s, q):
    s += sgl.user(q)
    s += sgl.assistant(sgl.gen("answer", max_tokens=200))


state = chat.run(q="什么是 RadixAttention？")
print(state["answer"])
```

**关键点**：
- `launch_server` 起的 server 同样兼容 OpenAI Compatible API（`POST /v1/chat/completions`），可直接被 OpenAI SDK 调用
- `sgl.function` 装饰器把函数体编译成一个程序——`s += ...` 是往 prompt 上 append、`sgl.gen("name")` 是 LLM 生成一段并起别名
- `state["answer"]` 取出生成结果——如果函数里有多个 `gen`，可以分别按 name 取

### 3.2 SGLang DSL multi-step 示例（≤ 25 行）

```python
@sgl.function
def multi_step_qa(s, question):
    s += sgl.system("你是一个分析助手，按 3 步回答用户问题。")
    s += sgl.user(question)
    s += sgl.assistant(
        "我来分 3 步回答。\n"
        "Step 1（拆解问题）：" + sgl.gen("step1", max_tokens=120, stop="\n") + "\n"
        "Step 2（关键分析）：" + sgl.gen("step2", max_tokens=180, stop="\n") + "\n"
        "Step 3（最终结论）：" + sgl.gen("final", max_tokens=80)
    )


state = multi_step_qa.run(question="为什么 prefix cache 在 multi-user system prompt 场景下收益巨大？")

print("STEP 1:", state["step1"])
print("STEP 2:", state["step2"])
print("STEP 3:", state["final"])
```

**关键点**：
- 每一步 `gen` 之前的所有内容（system + user + 前面 step 的输出）自动被 RadixAttention 注册为 prefix
- step 2 prefill 时**完整复用** step 1 之前的 prefix 与 step 1 的输出——零额外 prefill
- 同理 step 3——multi-step prompt 下 prefix cache 命中率几乎 100%
- 对比：用 OpenAI API 写三次 `chat.completions.create` + 手动拼 prompt，需要在 server 端 prefix cache 才能省算力（且取决于 server 实现颗粒度）

### 3.3 结构化 JSON 输出（XGrammar 集成）（≤ 20 行）

```python
import sglang as sgl

# JSON Schema 定义
schema = """{
  "type": "object",
  "properties": {
    "name": {"type": "string"},
    "age": {"type": "integer", "minimum": 0, "maximum": 120},
    "skills": {"type": "array", "items": {"type": "string"}}
  },
  "required": ["name", "age", "skills"]
}"""


@sgl.function
def extract_profile(s, text):
    s += sgl.system("从文本中提取人物信息，输出严格 JSON。")
    s += sgl.user(text)
    s += sgl.assistant(sgl.gen("profile", max_tokens=200, json_schema=schema))


state = extract_profile.run(text="张志伟，27 岁，擅长 LLM 后训练、RL、Python。")
import json
print(json.loads(state["profile"]))   # 保证合法 JSON 且字段齐全
# {'name': '张志伟', 'age': 27, 'skills': ['LLM 后训练', 'RL', 'Python']}
```

**关键点**：
- `json_schema=schema` 触发 XGrammar 在 decode 阶段对 logits 做 mask——只允许"使输出仍能符合 schema 的 token"采样
- 输出**保证合法 JSON**且字段齐全——业务代码可以直接 `json.loads` 解析，不需要 try/except + retry 的脏处理
- 对 agent / tool calling 场景至关重要：function call 的参数必须严格符合 schema（详见 13.4）
- 性能 overhead：XGrammar 在 decode 时按 schema 维护一个 grammar state machine、每步算 token mask——典型场景 latency 增加 5-20%；XGrammar 比 Outlines 快几倍（pre-compile 状态机 + bitmask 优化）

### 3.4 RadixAttention trie 简化模拟（≤ 30 行 Python）

不能跑真模型，但完整演示 radix tree 的"找最长公共前缀 + 节点分裂 + 复用"逻辑：

```python
class RadixNode:
    def __init__(self, tokens=()):
        self.tokens = tuple(tokens)        # 该节点对应的 token 序列段
        self.children = {}                 # token -> RadixNode
        self.kv_cached = False             # 是否真的有 KV 缓存（演示用 bool 代替指针）
        self.last_used = 0


class RadixCache:
    def __init__(self):
        self.root = RadixNode()
        self.t = 0                         # 全局时间戳

    def match_prefix(self, tokens):
        """返回 (匹配节点, 已匹配长度)。"""
        self.t += 1
        node, matched = self.root, 0
        while matched < len(tokens):
            child = node.children.get(tokens[matched])
            if child is None: break
            # 找 child.tokens 与 tokens[matched:] 的最长公共前缀
            lcp = 0
            for a, b in zip(child.tokens, tokens[matched:]):
                if a != b: break
                lcp += 1
            matched += lcp
            child.last_used = self.t       # LRU 更新
            if lcp < len(child.tokens):    # child 内部分叉
                self._split(child, lcp); break
            node = child
        return node, matched

    def insert(self, tokens):
        """命中匹配后把剩余部分挂到树上。"""
        node, matched = self.match_prefix(tokens)
        if matched < len(tokens):          # 有未匹配的尾部 → 创建新节点
            new_node = RadixNode(tokens[matched:])
            new_node.kv_cached = True; new_node.last_used = self.t
            node.children[tokens[matched]] = new_node

    def _split(self, node, at):
        """把 node 在 at 位置分裂成 prefix + suffix。"""
        suffix = RadixNode(node.tokens[at:])
        suffix.children = node.children; suffix.kv_cached = node.kv_cached
        node.tokens = node.tokens[:at]
        node.children = {suffix.tokens[0]: suffix}


# === 用法演示 ===
cache = RadixCache()
sys_prompt = tuple("You are a helpful assistant.".split())   # 简化：用单词当 token
cache.insert(sys_prompt + tuple("What is RAG?".split()))     # 请求 A
cache.insert(sys_prompt + tuple("What is agent?".split()))   # 请求 B → 命中前 6 token

# 新请求 C：完全相同 system prompt + 新问题
node, matched = cache.match_prefix(sys_prompt + tuple("Help debug code.".split()))
print(f"matched {matched} tokens of {len(sys_prompt) + 3}")
# matched 6 tokens of 9  → 复用了 system prompt 全部，仅需 prefill 后 3 token
```

**这 30 行代码包含 RadixAttention 的完整骨架**——真实 SGLang 实现额外加了 ref_count、并发安全、与 PagedAttention block 池的对接、跨 sample 的 KV 共享读、Copy-on-Write 等机制（见 SGLang 源码 `python/sglang/srt/mem_cache/radix_cache.py`）。

### 3.5 Cache 命中率监控

production 部署 SGLang server 后通过 `/metrics` 端点（Prometheus 格式）能拿到：

```
sglang:cache_hit_rate           # 命中率 ρ_token
sglang:running_requests         # 当前在跑的请求数
sglang:waiting_requests         # 排队的请求数
sglang:gen_throughput_token_s   # decode token/s
```

业务代码侧也可以在 client 端用 `sgl.gen` 返回的 `meta_info` 看每条请求的命中字符数：

```python
state = chat.run(q="..."); meta = state.get_meta_info("answer")
# meta 含 cached_tokens / completion_tokens / prompt_tokens 等
```

---

## 4. 工程踩坑与经验

- ❗ **Prefix cache 在 prompt 完全不同的场景几乎无收益，看清场景再用**。§2.1 已推导：算力节省比例是命中率的**平方**——命中率 0.1 时省 1%、命中率 0.3 时省 9%。如果你的负载是"全新 query 为主"（如开放搜索、单轮翻译、无固定 system prompt 的 playground），prefix cache 几乎只增 cache 维护开销而不省算力。**判断方法**：开 server 跑一段时间 → 看命中率 metric。命中率 < 30% 时关掉 prefix cache 反而更稳；> 60% 时是真正赚钱场景。

- ❗ **SGLang 的 DSL 学习曲线略陡，习惯了 OpenAI API 不一定喜欢**。`@sgl.function` + `s += sgl.gen(...)` 这套写法对从 LangChain / OpenAI SDK 转过来的工程师不直观——需要理解"程序图编译"心智模型。**实战建议**：
  - 如果你的应用就是普通 chat / 单轮 RAG → 直接用 SGLang 的 OpenAI Compatible API 端点（`/v1/chat/completions`），享受 RadixAttention 收益、不用学 DSL
  - 如果你的应用是 multi-step agent / branch / 需要细粒度 prefix 控制 → 学 DSL 投资回报高
  - 团队上手周期：单 dev 1-2 天看官方 examples 能写出能跑的程序，深入到 control flow + 自定义 backend 需要 1 周

- ❗ **RadixAttention 的 LRU 驱逐策略对 cache 命中率敏感，long-running 服务要监控命中率**。tree 越大 → 维护成本越高、占的 KV 显存越多；超过阈值后 LRU 驱逐"冷"节点——但有时被驱逐的恰恰是某个间歇被访问的 system prompt（间隔 > LRU 时间窗），下次又要重算。**Production 经验**：
  - 监控 `cache_hit_rate` 趋势——突降通常意味着热点 prefix 被错误驱逐或某类大流量进入冲淡了 cache
  - 调 `--max-prefix-sharing-sequence-length` 与 KV pool 大小平衡 hit rate 与显存
  - 真正不变的"大 system prompt"可考虑 pin 住（SGLang 有 `pin_prefix` API），不让 LRU 驱逐
  - long-running 服务每天看一次命中率分布，命中率衰减时考虑重启 / 调参

- ❗ **vLLM `enable_prefix_caching=True` 在某些 model（如 chat template 不一致）效果不稳，先 benchmark**。vLLM 的 hash-based prefix cache 对"前缀完全一致"敏感——如果你的应用通过 `apply_chat_template` 把 messages 转成 prompt，但不同 sample 的 system prompt 末尾偶尔多/少一个换行符或空格 → token 序列就不一致 → cache miss。**排查清单**：
  - 关掉 prefix cache 跑一遍 baseline，再开 prefix cache 跑 → 看 throughput 到底有没有提升
  - tokenize 几个典型 prompt，对比 token id 序列是否完全一致
  - 检查 `tokenizer.apply_chat_template` 是否给所有 message 加了一致的 BOS / system 模板
  - 不一致时显式传 `--chat-template` 标准化

- ❗ **结构化输出（XGrammar / Outlines）在某些 schema 上 latency overhead 5-20%**。constrained decoding 每步要算"哪些 token 仍能让输出符合 schema"——schema 越复杂（深嵌套 JSON、长 regex、CFG 多分叉）overhead 越大。**实测经验**：
  - 简单 JSON Schema（< 10 字段）overhead < 5%
  - 深嵌套 schema（含 array of object）overhead 10-15%
  - 复杂 EBNF（如 SQL 子集）overhead 15-30%
  - XGrammar 通常比 Outlines 快 2-5×（state machine pre-compile + bitmask 优化）——优先选 XGrammar
  - 上线前用真实 schema 跑 micro-benchmark 测 overhead，不要凭"应该不大"上线
  - 对 latency 敏感场景（如 streaming 用户感知）可考虑"不用 schema 约束 + try/except + retry"做对比

- ❗ **multi-LoRA serving SGLang 与 vLLM 都支持，但 LoRA 与 prefix cache 一起用时要小心 base model 与 LoRA 切换 cache key**。同一个 base model 挂多个 LoRA 时，**同一段 prompt 在不同 LoRA 下产生的 KV 不同**——朴素实现如果只按 prompt token id 做 cache key 而不区分 LoRA id → 跨 LoRA 借用错误的 KV → 输出乱码。**正确**：cache key 要含 `(lora_id, prompt_tokens)`；切换 LoRA 时各自独立的 cache 子树。SGLang 与 vLLM 0.5+ 都已正确处理，但**自定义部署或老版本要 warning**——上线前做"挂 N 个 LoRA、每个发不同 prompt、看输出是否串"的 sanity test。

- ❗ **SGLang 在某些新模型上的支持滞后于 vLLM 几周到几个月**。SGLang 社区比 vLLM 小、新模型适配速度慢——某些刚发布的 LLaMA / Qwen / DeepSeek 新版本 vLLM 支持后 SGLang 还要等几周。**实战策略**：
  - 选型时先去 SGLang `models` 列表确认你的目标模型在
  - 不在的话有 3 条路：(1) 等 SGLang 支持；(2) 自己写 model adapter（参考已有模型的实现，通常几百行）；(3) 用 vLLM
  - 大多数主流模型（LLaMA / Qwen / Mistral / DeepSeek / Gemma）都 first-class 支持，只有奇怪架构 / 多模态新模型才会卡这一步

- ❗ **SGLang 的 server 内存配额参数与 vLLM 不同名，迁移时容易踩坑**。vLLM 用 `gpu_memory_utilization`，SGLang 用 `mem_fraction_static`（语义类似但默认值与计算口径有微妙差异）；vLLM 的 `max_model_len` 在 SGLang 是 `context_length`；vLLM 的 `tensor_parallel_size` 在 SGLang 是 `tp`。**迁移 checklist**：
  - 启动参数对照官方 [Server Arguments](https://docs.sglang.ai/backend/server_arguments.html) 文档逐个 map
  - OOM 时降 `mem_fraction_static`（默认 0.9，降到 0.85）
  - 多卡 hang 时设 `--enable-torch-compile False` 与 `--disable-cuda-graph` 排查
  - 端口默认 30000（vLLM 默认 8000），脚本里要改

- ❗ **Multi-turn chat 要正确传 session_id 才能享受 RadixAttention 增量收益**。SGLang OpenAI 兼容 API 接到一个 chat completion 请求时，server 端按 messages 算 token 序列、查 radix tree——只要前几轮的 messages 一字不差地重新发上来，自然命中。但**如果 client 端每轮重新构造 messages 时给的 turn 顺序、role 字符串、空格缩进有差异 → token 序列变化 → cache miss**。**正确做法**：
  - client 端保留一份 `messages` list、append 新 turn 而不是每次重建
  - 注意 `apply_chat_template` 的渲染稳定性——OS 不同 / `transformers` 版本不同可能导致末尾空格差异
  - 用 SGLang DSL 的 multi-turn 模式（`@sgl.function` 内部 append）能彻底避免这类问题

- ❗ **Prefix cache 命中后 first-token latency 大幅降低，但 decode 速度不变——要正确解读 metrics**。Prefix cache 优化的是 **prefill 阶段**（命中部分跳过），TTFT 显著降低；但 decode 阶段每步还是要从 KV cache 里读出来算 attention，**TBT 不变**。新人看到"开了 prefix cache 但 token/s 没涨多少"会困惑——其实 throughput 涨在"单位时间能服务的请求数"（因为每个请求 prefill 算力少了 → GPU 能腾出来跑更多请求），而不是单条请求的 token/s。**正确监控指标**：QPS、TTFT P50/P99、并发请求数、cache hit rate；只看 token/s 会误判。

---

## 5. 经典 paper

- **Zheng et al., 2024 — SGLang: Efficient Execution of Structured Language Model Programs (arXiv:2312.07104)** — **本节核心必引**。RadixAttention 的提出 paper，SGLang 框架的奠基论文。§3 给 RadixAttention 的 trie 数据结构与 LRU 算法、§4 给 SGLang DSL 的设计理念（compiler-style 处理 LLM 程序）、§5 给 structured output 与 KV cache 复用的协同优化、§6 实测在 multi-step / agent / branch 场景下 throughput 较 vLLM 提升 2-5×。读 §3-4 即可建立 RadixAttention 完整心智模型。
- **Kwon et al., 2023 — Efficient Memory Management for LLM Serving with PagedAttention (arXiv:2309.06180)** — 11.2 已引；本节作为对比组——vLLM 的 hash-based block-level prefix cache 是 RadixAttention 的"前身"，理解两者颗粒度差异需要先理解 PagedAttention 的 block 机制。
- **Dong et al., 2024 — XGrammar: Flexible and Efficient Structured Generation Engine for LLMs (arXiv:2411.15100)** — SGLang 集成的结构化输出引擎。§3 介绍 grammar state machine 的 pre-compile + bitmask 优化（核心 idea：把 CFG 转成可在 GPU 上 bitmask 操作的状态机），让 constrained decoding 的 overhead 从 Outlines 的 30%+ 降到 5%——production 上 grammar 强制约束的可行性来自这一篇。
- **Willard & Louf, 2023 — Efficient Guided Generation for Large Language Models (Outlines) (arXiv:2307.09702)** — constrained decoding / guided generation 的早期代表作。把 CFG / regex 编译成 FSM、每步 decode 用 FSM 状态约束 logits——XGrammar 的"前身"。读它能理解 constrained decoding 的基本原理。
- **SGLang GitHub — [sgl-project/sglang](https://github.com/sgl-project/sglang)** — 直接读源码，重点 `python/sglang/srt/mem_cache/radix_cache.py`（RadixAttention 核心）+ `python/sglang/srt/managers/scheduler.py`（调度器）+ `python/sglang/lang/`（DSL frontend）。本节代码示例的"完整版"都在这里。

---

## 6. 自测与面试题

**Q1（数据结构）**：RadixAttention 的 trie 怎么管理 KV cache？为什么比 hash-based block cache 在 branch 场景更优？

<details>
<summary>Answer sketch</summary>

**RadixAttention 的 trie 管理方式**：
- 全局维护一棵 radix tree（compressed trie），每个节点 = 一段 token 序列 + 对应的 KV cache 指针
- 边按 token id 索引；连续单分叉的 token 合并成一个节点（路径压缩）
- 节点带 `last_used` 时间戳与 `ref_count`
- **新请求来时**：按 token 序列从根开始往下匹配，找到的最深节点即为最长公共前缀——该节点及其祖先的 KV 全部可复用
- **prefill 完后**：把请求的剩余部分作为新子节点挂到树上（必要时分裂已有节点）
- **LRU 驱逐**：refcount==0 的叶子节点按 last_used 升序淘汰

**为什么比 hash-based block cache 在 branch 场景更优**：
- vLLM 的 hash-based prefix cache 是 **block 颗粒度**（16 token 一块），block hash 链式计算——只能匹配"完全相同的整 block 链"
- branch 场景（best-of-N、MCTS、agent rollout 多分叉）的分叉点**很可能不在 block 边界上**——比如两个 branch 共享 prefix 1023 token、第 1024 token 不同；vLLM 在第 64 个 block（token 1024-1039）就开始 miss，错失 1024-1023=1 个 token 之内不能共享但 1023 token 能共享的颗粒度
- RadixAttention 是 **token 颗粒度**——能精确匹配到"共享 1023 token、第 1024 token 分叉"，命中率显著高于 vLLM
- best-of-N=16 的 GRPO rollout 场景下，SGLang 通常多 10-30% 命中率、对应 prefill 算力多省 20-50%

加分：
- 能说出代价——RadixAttention 维护开销（trie 操作、ref_count、并发同步）比 hashtable 高，"前缀完全 block-aligned 的纯共享场景"两者收益相近，SGLang 的"复杂度增加"在这种场景下反而是负担
- 能说 RadixAttention 与 PagedAttention 不冲突——SGLang 底下 KV 物理存储仍然按 block 池管理（参考 PagedAttention 思路），radix tree 只是逻辑索引层
- 能说 Copy-on-Write 用于分叉时的写保护

</details>

**Q2（场景）**：列出 3 个 prefix cache 收益巨大的实际场景 + 1 个收益几乎为 0 的场景，并解释原因。

<details>
<summary>Answer sketch</summary>

**收益巨大的 3 个场景**（必须答出与"前缀重复率高"的关联）：

1. **多 user 共享 system prompt**：商业 chatbot 后端的 system prompt 1k-10k token、所有用户共享——前缀完全相同。命中率 $\rho \approx \frac{T_{\text{sys}}}{T_{\text{sys}} + T_{\text{user}}}$，典型 0.7-0.9。算力节省 $\rho^2 \approx 50\%-80\%$。
2. **Multi-turn chat 的 history 增量**：第 N 轮 prompt = 前 N-1 轮 + 新 user 消息，前 N-1 轮的 KV 在上一轮已算。命中率约 $(N-1)/N$，长对话时趋近 100%。
3. **Agent rollout 的 best-of-N 采样**：GRPO `n=16` 对同一 prompt 采 16 条 trajectory——所有 trajectory 共享 prompt 的 100% 前缀，每个 step 之前的 obs/thought/action 部分逐步分叉但仍大量共享。多 branch 场景 RadixAttention 比 vLLM 优势明显。

**收益几乎为 0 的 1 个场景**：

- **开放搜索 / 一次性翻译 / 无 system prompt 的 playground**——每个请求的 prompt 完全不同（用户搜索 query 各异、翻译输入各异）、共享前缀仅 BOS + 极短模板；命中率 < 0.1，算力节省 < 1%；开 prefix cache 反而引入 cache 维护开销，可能略慢。

加分：
- 能算出"算力节省 = 命中率的平方"（§2.1 公式）
- 能说"RAG with 固定文档库 + 不同 query"是"中等收益"场景——共享 system + answer_format（约 30%）+ 不同 docs/query（约 70%）→ $\rho \approx 0.3$、节省 ~10%
- 能区分 "prefix cache 命中率"（请求级）与 "KV cache 显存占用"（系统级）——它们分别是 prefill 优化与 decode 优化的指标
- 能说"chat session 的命中率随 turn 数线性增长"

</details>

**Q3（选型）**：你做一个 agent 平台（multi-step + tool calling + structured output），vLLM vs SGLang 选哪个？为什么？给出至少 4 个选型理由。

<details>
<summary>Answer sketch</summary>

**结论**：选 **SGLang** 为主，vLLM 为备份/迁移路径。

**4 个选型理由**：

1. **Multi-step agent rollout 的 prefix 复用**：agent 一条 trajectory 是 `system + tools_desc + obs_1 + thought_1 + ...` 长链，每一 step 都重发前面所有内容；RadixAttention 按 token 颗粒度命中（vs vLLM 16-token block 颗粒度可能错失分叉），多 step / 多 branch 场景 throughput 通常 20-50% 优于 vLLM。

2. **Structured output 集成更深**：SGLang 集成 XGrammar，`json_schema=...` / `regex=...` 直接传——tool calling 的参数严格符合 schema、business 代码可直接 `json.loads` 不需要 retry；vLLM 也支持 `guided_decoding` 但集成相对浅、性能略差（XGrammar 比 Outlines 快几倍是公开 benchmark）。

3. **DSL 让 multi-step 编排自动 cache**：`@sgl.function` 把 multi-step / branch 逻辑写成程序图，框架自动把每个 step 之间的 prefix 注册到 radix tree——同样的逻辑用 OpenAI API 写需要每步手动拼 prompt，cache 复用依赖 server 端实现。

4. **Best-of-N / MCTS 等搜索场景**：agent planning 经常对同一状态做多分支探索（ToT、LATS）——RadixAttention 对分叉前缀的精细共享天然契合此类负载。

**vLLM 仍要懂 / 仍可作为备份的理由**：
- 生态最广、文档最全、新模型 first-class 支持最快——选型阶段如果 SGLang 还没适配你的模型、vLLM 是 fallback
- 大规模 batch inference（不强 agent 场景）vLLM 略优——纯 batch 推理走 vLLM 更稳
- Multi-LoRA serving 两者都支持，但 vLLM 的生态与文档更成熟
- production 部署经常**两者都跑**——agent 走 SGLang、纯文本生成走 vLLM，按场景路由

**额外加分**：
- 能说"先小流量 A/B 测命中率与 P99，再决定上量"——选型不能拍脑袋
- 能说 SGLang 的 `mem_fraction_static` 与 vLLM 的 `gpu_memory_utilization` 对应、迁移 checklist
- 能提"Anthropic Prompt Caching / OpenAI Prompt Caching API 商业化卖的就是 prefix cache"——理解这是 production 大规模赚钱的功能、不是学术玩具
- 能区分 11.2 PagedAttention（内存管理层）、11.3 RadixAttention（请求复用层）、5.2 GQA/MLA（架构层）、11.4 量化（精度层）——四层正交可叠加

</details>

---

## 7. 延伸阅读

- [Zheng et al. 2024 — SGLang paper (arXiv:2312.07104)](https://arxiv.org/abs/2312.07104) — RadixAttention 与 SGLang 框架的奠基论文，本节核心引用
- [SGLang GitHub](https://github.com/sgl-project/sglang) — 源码必读：`python/sglang/srt/mem_cache/radix_cache.py` + `python/sglang/srt/managers/scheduler.py` + `python/sglang/lang/`
- [SGLang 官方文档](https://docs.sglang.ai/) — 参数手册、DSL 教程、模型支持列表
- [Dong et al. 2024 — XGrammar paper (arXiv:2411.15100)](https://arxiv.org/abs/2411.15100) — SGLang 集成的结构化输出引擎
- [Willard & Louf 2023 — Outlines paper (arXiv:2307.09702)](https://arxiv.org/abs/2307.09702) — guided generation 的早期代表
- [vLLM Automatic Prefix Caching 文档](https://docs.vllm.ai/en/latest/automatic_prefix_caching/apc.html) — 与 RadixAttention 对比阅读
- [Anthropic Prompt Caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — production 商业化的 prefix cache feature，理解 agent / RAG 场景下用户怎么用 cache
- [LMSYS Blog — SGLang v0.2: Faster Interface and Backend](https://lmsys.org/blog/2024-07-25-sglang-llama3/) — SGLang 团队的官方科普，配合 §1 食用
- 推荐继续读本教程的 **11.4 节《量化：GPTQ / AWQ / GGUF / FP8 / SmoothQuant》**——KV cache 量化与 prefix cache 正交可叠加，production 必组合用
- 推荐继续读本教程的 **11.5 节《投机解码：Speculative / Medusa / EAGLE》**——decode 阶段的另一类加速，与 prefix cache（prefill 优化）正交
- 推荐继续读本教程的 **13.4 节《Function calling 工程》**——结构化输出（XGrammar）在 tool calling 的应用
- 推荐继续读本教程的 **15.2 节《多轮 PPO/GRPO》**——agent rollout 场景下 RadixAttention 是 RL 训练 throughput 的关键基础设施
