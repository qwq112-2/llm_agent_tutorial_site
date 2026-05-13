---
title: "14.5 Memory：scratchpad / vector DB / MemGPT / Generative Agents"
description: "Agent memory 不是单一组件，而是一个 short-term（scratchpad / message history）→ long-term（vector DB / MemGPT / Generative Agents）→ procedural（skill library） 的层级系统——这一节讲清楚各类 memory 的范式、MemGPT 的 OS-style 分页机制、Genera"
---

> ⏱ 预计阅读 35 分钟 ｜ 难度 ★★ ｜ 前置：13.2（embedding / vector DB / retriever）

## 一句话本节讲什么

Agent memory 不是单一组件，而是一个 **short-term（scratchpad / message history）→ long-term（vector DB / MemGPT / Generative Agents）→ procedural（skill library）** 的层级系统——这一节讲清楚各类 memory 的范式、MemGPT 的 OS-style 分页机制、Generative Agents 的 importance + recency + relevance 三因子检索，以及 2025-2026 年向 parameterized memory / KV cache editing 收敛的趋势。

---

## 1. Mental model（直觉）

ChatGPT 用户大概率都遇到过这种事：上周聊了一整下午"我在准备 ACL 投稿、方向是 tool use RL"，下周新开一个对话，模型完全不记得；要么你又复述一遍背景，要么模型瞎猜。这就是 **agent memory 缺位** 最直观的体感。

把视角拉回 LLM 本身：模型参数训完就冻住了，唯一"记得"东西的地方是 **当前 context window 里的 token**。一旦关了对话，所有信息归零。这种"金鱼记忆"对单 turn QA 没问题，但对一个真正"长期陪你"的 agent 完全不够。

Agent memory 要解决的问题谱比 chat 复杂：

- **跨 task**：今天解了一个 SQL bug 的 trick，明天遇到类似的应当复用
- **跨 session**：用户上周说他偏好 Python typed dict 不要 dataclass，三个月后还应记得
- **超长 horizon**：一个 25-agent 的虚拟小镇跑 100 天，每个 agent 累积上万条 observation，怎么"想起来"昨天某个邻居说过的话
- **跨工具结果**：一次 web 搜索返回 20k token，下一步推理只需要其中两段——剩下的扔了浪费、留着塞爆 context

光靠把 context window 加大是不行的——即使 1M context（≈ 几本书），对一个跑一年的 personal assistant 来说也只是几个月对话量；而且 context 越长，**lost in the middle**（13.2 谈过）越严重，前置 1M token 还要算 prefill cost。所以 memory 的核心 design pattern 一句话：

> **Selectively retain + selectively retrieve**——不是什么都记，记下来的也不是每次都全塞 prompt，按需 page in / page out。

四类 memory 的角色分工，可以借 **认知科学** 的分类来类比（Atkinson-Shiffrin / Tulving 的经典模型）：

```
                         ┌─────────────────┐
   当前 task scratchpad ─┤  Short-term     │── 容量小、易失、放在 context window
   message history       │  (Working)      │   实现：list / sliding window / summary
                         └─────────────────┘
                         ┌─────────────────┐
   昨天发生过什么 ──────┤  Episodic       │── 时间序列、事件向量、可检索
   去年说过什么         │  (Long-term)    │   实现：vector DB + metadata
                         └─────────────────┘
                         ┌─────────────────┐
   "Python 是动态语言" ─┤  Semantic       │── 事实 / 知识，与时间无关
   "巴黎是法国首都"     │  (Long-term)    │   实现：RAG（13.2-13.3）
                         └─────────────────┘
                         ┌─────────────────┐
   会怎么写一个二分查找─┤  Procedural     │── 技能 / 可执行 code，做不是说
                         │  (Skill)        │   实现：Voyager skill library
                         └─────────────────┘
                         ┌─────────────────┐
   "我是一个助手、       ┤  Self-knowledge │── 关于自己的元信息
    擅长 X 不擅长 Y"     │                 │   实现：system prompt / persona
                         └─────────────────┘
```

工程上几乎所有"agent memory"产品（Anthropic 的 memory feature、OpenAI 的 ChatGPT memory、mem0 / Letta / LangMem）都是上面这几类的某种组合。下一节把每一类的主流实现拆开讲。

---

## 2. 公式与原理

### 2.1 Short-term memory：scratchpad / message history / sliding window / summary

**Scratchpad** 是 ReAct（14.1）风格 agent 的标准记法：每一步生成的 `Thought / Action / Observation` 直接拼到 prompt 末尾，下一步生成时整个轨迹可见。

```
[system prompt]
[task]
Thought 1: 我先搜一下今天股价
Action 1:  search("AAPL stock price today")
Observation 1: ... (10k token)
Thought 2: 现在我看到价格是 195，下一步算 5 日均线
Action 2:  python("...")
Observation 2: ...
↑ scratchpad 越累越长
```

scratchpad 简单但有两个硬约束：

- **token 预算**：每一步 observation 可能上万 token，10 步就把 context 顶满
- **noise 累积**：早期 observation 多数与最终答案无关，留着稀释 attention

应对策略：

1. **Sliding window**：只保留最近 N 步的 thought/action/observation
2. **Selective retention**：明确标"这一步的 obs 重要 / 不重要"，只保留重要的
3. **Summarization**：把老 turn summarize 成一段 1-2 句话的"context 摘要"
4. **Tool result truncation**：observation 超过 K token 直接截断，原文存到 vector DB 等需要时 retrieve

**Message history** 是 chat 风格的 short-term memory，与 scratchpad 等价，差别只是结构化成 `[{role, content}, ...]` 列表。OpenAI / Anthropic 的 chat completion API 默认就是这种格式。

**Summarization** 的典型做法（LangChain 的 `ConversationSummaryBufferMemory`）：维护一个滑动 buffer 存最近 K 条原文 message，buffer 满了就调用 LLM 把 buffer 头部 summary 化，原文丢弃。**这是踩坑高发区**——多次摘要会出现"消息传递游戏"现象（信息逐次失真），后面踩坑会展开。

### 2.2 Long-term memory：vector DB + metadata

把 message / observation / event embed 后存进 vector DB，需要时按 query 余弦相似度 retrieve top-K——这就是最朴素的 long-term memory，本质上是 **agent 私有的 RAG**（13.2 讲过的 pipeline 直接复用）。

差别在于：

- **document RAG** 检索的是静态知识库（手册、wiki、产品文档）
- **agent memory RAG** 检索的是 **动态生长的对话 / observation 流**——每次交互都在写

写的时机：

- **after tool call**：把 (tool, args, result) 存为一条 memory
- **after user turn**：把 user 说的话存为一条
- **after agent reflection**：把 agent 自己 reflect 出的 lesson 存为一条
- **periodic**：每 N turn 把 short-term scratchpad 整体 summary 后写入

最关键的是 **metadata**：每条 memory 必须带 timestamp / user_id / session_id / source / importance，否则后面没法做时间衰减、user 隔离、aging 策略——这些后面会讲。

朴素的 vector DB memory 已经能 cover 不少场景（mem0 的核心就是这个），但有几个原生缺陷：

- **没有时间衰减**：3 个月前说过的话和昨天说过的话权重一样
- **没有"重要性"区分**：随手一句和"我严重过敏花生"权重一样
- **没有结构化**：episodic 与 semantic 混在一起，"我喜欢 Python" 和 "今天我装了 Python 3.12" 都被当做一条
- **没有 reflection**：raw observation 没法直接得出 high-level 结论（需要二次抽象）

下面 MemGPT 和 Generative Agents 就是分别从 **架构** 和 **检索打分** 两个角度补这些缺陷。

### 2.3 MemGPT：OS-style virtual memory for LLM

Packer et al. 2023 把 OS 的 **virtual memory + paging** 思想搬到 LLM agent 上，核心类比：

| OS | MemGPT |
|---|---|
| RAM（main memory） | LLM 的 context window |
| Disk（external storage） | 外部数据库（vector DB / SQL）|
| Page in / page out | LLM 主动调 tool 加载 / 卸载内容 |
| Page table | recall memory（最近交互的索引） |

MemGPT 把 context window 切成几个固定区域：

```
┌─────────────────────────────────────────┐
│         CONTEXT WINDOW (~8k-32k)         │
├─────────────────────────────────────────┤
│ system prompt + tool definitions         │  ← 静态
├─────────────────────────────────────────┤
│ working context (persona / 用户偏好)     │  ← LLM 可改
├─────────────────────────────────────────┤
│ FIFO message queue (最近对话)            │  ← 满了 evict
├─────────────────────────────────────────┤
│ recall memory hint (元信息)              │  ← 可被 search
└─────────────────────────────────────────┘
        │  page_in / page_out
        ↓
┌─────────────────────────────────────────┐
│   ARCHIVAL MEMORY  (vector DB)           │
│   - 历史所有对话                          │
│   - 用户长期事实                          │
└─────────────────────────────────────────┘
```

LLM 通过 4 个核心 tool 操作 memory：

- `core_memory_append(section, content)`：往 working context 写入
- `core_memory_replace(section, old, new)`：覆盖式编辑（如更新 persona）
- `archival_memory_insert(content)`：往 archival memory 写入
- `archival_memory_search(query)`：从 archival memory retrieve（vector search）

当 FIFO message queue 满了，系统会触发 **memory pressure warning**——LLM 收到一条 system message："你的 message queue 快满了，调用 `core_memory_append` / `archival_memory_insert` 把重要信息保存"，然后 evict 最老的几条 message。LLM 决定哪些保存哪些扔——这是 OS-style memory 的核心：**LLM 自己是 memory manager，不是被动存档**。

这种设计的优点是 **理论上无限对话长度**——只要 archival memory 容量够；缺点见踩坑章节（每次都要 router decide、tool call 增加 cost）。

### 2.4 Generative Agents：memory stream + 三因子打分 + reflection

Park et al. 2023 在斯坦福做了一个 **25 个 LLM agent 的虚拟小镇**（《Generative Agents: Interactive Simulacra of Human Behavior》），每个 agent 累积上万条 observation——这是 memory-heavy agent 的开山之作。它的 memory 设计现在被广泛复用。

**Memory stream** 是核心数据结构：所有 observation / event / dialog 按时间序列存，每条都带 (timestamp, content, last_access_time)。当 agent 要决策时，从 stream 里检索 top-K 相关 memory 注入 prompt——但 retrieval 不只看 embedding 相似度，而是 **三因子加权**：

$$\text{score}(m, q) = \alpha \cdot \text{importance}(m) + \beta \cdot \text{recency}(m) + \gamma \cdot \text{relevance}(m, q)$$

其中：

- **Importance**：LLM 给每条 memory 在写入时打 1-10 分。Prompt 大致是"打分这条 memory 的 poignancy（重要性），1 = 刷牙吃饭这种日常，10 = 分手 / 升学这种人生大事"。Park 2023 的实现把分数归一化到 [0, 1]。
- **Recency**：指数衰减
$$\text{recency}(m) = \exp(-\lambda \cdot \Delta t)$$
其中 $\Delta t$ 是 memory 上次访问到当前的时间间隔（小时为单位），$\lambda$ 是衰减率（Park 2023 取 0.99，半衰期约 100 小时）。注意是 **last access time** 不是 creation time——经常被回想起的记忆更不容易"忘"，符合人类认知。
- **Relevance**：当前 query embedding 与 memory embedding 的余弦相似度。

三个分数都归一化到 [0, 1] 后线性加权，$\alpha = \beta = \gamma = 1$ 是 paper 默认。retrieval 时按 score 排序取 top-K。

**Reflection** 是另一个关键机制：当 agent 累积的 importance 总和超过阈值（约对应 100+ 条普通 observation），就触发一次 reflection——

1. LLM 看最近 100 条 memory，生成 3-5 个"我应当深入思考的问题"（如"我最近为什么频繁去图书馆？"）
2. 对每个问题，retrieve 相关的 memory，再让 LLM 综合出 high-level insight（如"我最近在准备一个研究 project"）
3. 这些 insight 作为新的 memory（标记为 `type=reflection`）写回 stream，importance 高、relevance 大，会被后续 retrieval 优先召回

reflection 的设计解决了 raw observation 没法直接产出抽象 conclusion 的问题——本质是 **memory 的 hierarchical 抽象**，是 Generative Agents 论文最被引用的贡献之一。

### 2.5 Procedural memory：Voyager 的 skill library

Wang et al. 2023 的 Voyager 在 Minecraft 里跑 LLM agent，提出一个不一样的 memory 范式：**memory 不是文字，而是可执行 code**。

流程：

1. agent 在环境里探索，遇到新 task（"造一个铁镐"）
2. LLM 写一段 JavaScript code 调用 Mineflayer API 完成 task
3. 如果 code 执行成功（环境验证），把这段 code + 自然语言描述（"造铁镐"）作为一条 **skill** 存入 library
4. 后续遇类似 task（"造钻石镐"），先 retrieve 相关 skill code 作为 in-context example，再让 LLM 写新 skill

skill 用 embedding 索引（embedding 描述文本），retrieval 与普通 RAG 一样。但 **存的是 code 不是文字**，复用时直接执行——这是 procedural memory 的范式。

这种设计的威力：agent 跑得越久，library 越大，"实力"越强——Voyager 在 Minecraft 里能解锁的物品数量随时间单调上升，而不是 plateau。后来 OpenHands / SWE-agent 这些 coding agent 也部分采用了 skill library 思想。

### 2.6 Reflexion：verbal RL as memory

Shinn et al. 2023 的 Reflexion 提出更轻量的 memory 用法——**失败的经验作为 in-context 提示**：

1. agent 尝试解一个 task，失败（环境给 0/1 reward）
2. LLM 看自己的 trajectory + reward，生成一段 verbal reflection（"我失败是因为没检查 boundary case，下次要先写测试"）
3. 这段 reflection 存到 episodic memory
4. 下一次尝试同 task 时，把 reflection 注入 prompt 作为提示
5. 不需要 gradient 更新，纯 prompt loop

Reflexion 在 HumanEval / AlfWorld 上能把 baseline ReAct 提升 10-30%，是 **不训练参数也能"学习"** 的代表——9.7（self-rewarding）和 14.1（agent 范式）讲过 motivation，本节强调它是 memory 的一种用法。

### 2.7 现代趋势：parameterized memory / KV cache editing

2024-2026 的前沿往两个方向走：

**Parameterized memory**：把 memory 编码进 model 参数本身（continual learning / online learning）。代表工作：

- **MemoryLLM**（Wang 2024）：在 LLM 里加一个 fixed-size "memory pool" parameter block，新信息通过 self-supervised loss 写入
- **Larimar**（IBM 2024）：episodic memory module 挂在 LLM 旁，one-shot edit 修改 fact

优点是检索 = forward pass（无 latency 开销）；缺点是写入慢、容量上限明确、catastrophic forgetting 还没完全解决。

**KV cache editing**：直接编辑 attention KV cache 实现 memory。比如在 prefill 阶段把"用户偏好 Python"prepend 进 prompt 算出 KV cache 后保存，下次 query 直接复用 KV——本质是 **prefix tuning + prefix cache** 的 memory 化（11.3 RadixAttention 是同源思想）。Anthropic 的 prompt caching、SGLang 的 RadixAttention 已经把 prefix cache 做成标准能力，下一步就是让 agent 主动管理"哪些 prefix 长期 cache"。

**Memory as tool**：把 memory 操作封装成 tool（`save_memory(content, importance)` / `recall_memory(query)`）让 agent 自己调——MemGPT 是先驱，现在已成主流。Anthropic 的 memory feature、OpenAI 的 ChatGPT memory 后端实现都是这种 pattern。

### 2.8 方案对比一表

| 方案 | 类型 | 复杂度 | 适用场景 |
|---|---|---|---|
| Scratchpad | short-term working | 低 | 单 task ReAct |
| Message history | short-term | 低 | chat / 短对话 |
| Sliding window | short-term | 低 | 长对话 token 控制 |
| Summarization | short→long bridge | 中 | 长 chat、信息可压缩 |
| Vector DB（mem0 风格）| long-term episodic | 中 | 跨 session 用户记忆 |
| MemGPT | long-term hierarchical | 高 | multi-session 长对话 |
| Generative Agents | episodic + reflection | 高 | sandbox / 仿真 / RPG |
| Voyager skill library | procedural | 高 | open-world / coding agent |
| Reflexion | verbal RL memory | 低 | 任务可重试场景 |
| Parameterized memory | weight-level | 高 | 前沿、未广泛工程化 |
| KV cache editing | activation-level | 中 | prefix-stable 场景、prompt cache |

---

## 3. 最小代码示例

### 3.1 Scratchpad + sliding window

```python
from collections import deque

class Scratchpad:
    """ReAct 风格的 thought/action/obs 累积，带 sliding window 防 context 爆炸"""
    def __init__(self, max_steps=10):
        self.steps = deque(maxlen=max_steps)        # 自动 evict 最老的

    def append(self, thought, action, observation):
        # observation 太长就截断（原文可另存 vector DB 备查）
        obs = observation if len(observation) < 2000 else observation[:2000] + "...[truncated]"
        self.steps.append({"thought": thought, "action": action, "observation": obs})

    def render(self) -> str:
        # 渲染成 prompt 可拼接的 string
        return "\n".join(
            f"Thought {i+1}: {s['thought']}\nAction {i+1}: {s['action']}\n"
            f"Observation {i+1}: {s['observation']}"
            for i, s in enumerate(self.steps)
        )

pad = Scratchpad(max_steps=5)
pad.append("先搜股价", "search('AAPL')", "Price: 195.4 ...")
pad.append("算均线", "python('np.mean(...)')", "192.1")
prompt = "..." + pad.render()       # 拼到 LLM prompt 里
```

`deque(maxlen=N)` 自动丢掉最老的——sliding window 一行实现。observation 截断是 ReAct agent 必做的——tool 返回动辄上万 token，不截断 5 步就把 32k context 撑爆。

### 3.2 Vector DB long-term memory（sentence-transformers + FAISS）

```python
# pip install sentence-transformers faiss-cpu
import faiss, numpy as np, time
from sentence_transformers import SentenceTransformer

class VectorMemory:
    def __init__(self, dim=384, model_name="BAAI/bge-small-zh-v1.5"):
        self.model = SentenceTransformer(model_name)
        self.index = faiss.IndexFlatIP(dim)         # 小数据 brute-force 即可
        self.metas = []                             # (content, timestamp, user_id)

    def add(self, content: str, user_id: str = "default"):
        emb = self.model.encode([content], normalize_embeddings=True).astype(np.float32)
        self.index.add(emb)
        self.metas.append({"content": content, "ts": time.time(), "user": user_id})

    def search(self, query: str, user_id: str, k: int = 5):
        q = self.model.encode([query], normalize_embeddings=True).astype(np.float32)
        scores, idx = self.index.search(q, k * 4)   # 多召回再过滤 user
        out = []
        for s, i in zip(scores[0], idx[0]):
            if i < 0: continue
            if self.metas[i]["user"] != user_id: continue   # ❗ user 隔离
            out.append((self.metas[i]["content"], float(s)))
            if len(out) >= k: break
        return out

mem = VectorMemory()
mem.add("用户偏好 Python typed dict 不要 dataclass", user_id="u123")
print(mem.search("写代码注意什么", user_id="u123"))
```

关键点两个：(1) `normalize_embeddings=True` 后用 inner product 等价于 cosine；(2) **`user_id` 必须强过滤**——多用户系统跨 user 检索是合规事故（见踩坑）。

### 3.3 MemGPT 风格 main / external memory + page in/out（pseudocode）

```python
# 完整 MemGPT 太长，这里给核心 control loop 的伪代码
class MemGPTAgent:
    def __init__(self, llm, archival: VectorMemory, ctx_limit=8000):
        self.llm = llm
        self.archival = archival                    # vector DB
        self.working_ctx = ""                       # persona / 用户偏好（LLM 可改）
        self.fifo = deque()                         # 最近 message
        self.ctx_limit = ctx_limit

    def build_prompt(self, user_msg):
        return f"[SYSTEM]\n[WORKING]\n{self.working_ctx}\n[FIFO]\n" + \
               "\n".join(self.fifo) + f"\n[USER] {user_msg}"

    def step(self, user_msg):
        # 1. memory pressure check
        prompt = self.build_prompt(user_msg)
        if len(prompt) > self.ctx_limit * 0.85:    # 接近满
            warning = "[SYSTEM] memory pressure high, save important content via tools"
            self.fifo.append(warning)
        # 2. LLM 决策（可能调 memory tool）
        out = self.llm(prompt, tools=[
            "core_memory_append", "archival_memory_insert", "archival_memory_search"
        ])
        # 3. 处理 tool call
        for call in out.tool_calls:
            if call.name == "archival_memory_search":
                hits = self.archival.search(call.args["query"], user_id="u")
                self.fifo.append(f"[RECALL] {hits}")          # page in
            elif call.name == "archival_memory_insert":
                self.archival.add(call.args["content"], user_id="u")
            elif call.name == "core_memory_append":
                self.working_ctx += "\n" + call.args["content"]
        # 4. evict 老 message 控制 ctx
        while len(self.build_prompt("")) > self.ctx_limit:
            self.fifo.popleft()                              # page out
        self.fifo.append(f"[ASSISTANT] {out.text}")
        return out.text
```

精髓在第 2-3 步：**LLM 自己调 tool 决定 page in / page out**——不是系统硬规则。这就是 MemGPT 比朴素 RAG memory 强的地方：可以 self-direct。

### 3.4 Generative Agents memory score 计算

```python
import math, time

def memory_score(memory, query_emb, alpha=1.0, beta=1.0, gamma=1.0, decay=0.99):
    """三因子加权：importance + recency + relevance"""
    # importance: LLM 在写入时打的 1-10 分，归一化
    imp = memory["importance"] / 10.0
    # recency: 上次访问到现在的小时数，指数衰减
    hours = (time.time() - memory["last_access"]) / 3600.0
    rec = decay ** hours                          # ≈ exp(-λ·Δt)
    # relevance: query 与 memory embedding 的余弦
    rel = float(memory["emb"] @ query_emb)        # 假设已 normalized
    return alpha * imp + beta * rec + gamma * rel

def retrieve(memories, query_emb, k=5):
    scored = [(m, memory_score(m, query_emb)) for m in memories]
    top = sorted(scored, key=lambda x: -x[1])[:k]
    now = time.time()
    for m, _ in top:
        m["last_access"] = now                    # ❗ 更新 last_access，被回想起=不易忘
    return [m for m, _ in top]
```

注意第 16 行 **更新 last_access**——这是 Generative Agents 的关键细节：经常被回想起的 memory 衰减更慢，模拟人类"用进废退"。漏掉这一行会让 recency 永远基于 creation time，老 memory 必然沉底。

---

## 4. 工程踩坑与经验

- ❗ **Vector DB memory 不加时间衰减 → 老 memory 永远召回**：朴素 mem0 风格只算 cosine similarity，结果 3 个月前的"我喜欢 Python"和昨天的"我换 Rust 了"都被召回，agent 一脸懵。**生产 memory 必须带 recency 因子**（Generative Agents 公式套上去最简单），或者引入"recency boost"——同分时新 memory 优先。
- ❗ **Summarization 多次摘要后信息丢失（"消息传递游戏"）**：LangChain 的 `ConversationSummaryBufferMemory` 默认是 summary of summary——5 轮之后原始细节几乎全没。**必须保留原始 + summary 双存**：summary 进 short-term context 节省 token，原文进 vector DB 当 long-term 备查；要细节时按 query retrieve 原文。
- ❗ **MemGPT 风格 page in/out 增加 LLM call 数**：每个 turn LLM 都要先 decide 要不要 search/insert memory，再做实际响应——一次用户提问可能产生 2-3 次 LLM call，cost 直接翻倍。**轻量场景用普通 vector DB memory + 离线 importance 打分就够**，MemGPT 只在真正多 session 长对话且预算充足时上。Letta（MemGPT 的工程化产品）开始引入 "background memory agent"——后台异步整理 memory，前台主对话不阻塞。
- ❗ **Memory 含 PII / 隐私时必须有 redact 机制**：用户随口说一句"我身份证 110105...."，原封存进 memory，后续 reflection 把这段 summary 进 high-level insight，再 leak 到对话——是真实合规事故。**写入前必须过 PII detector**（Microsoft Presidio / 自训 NER），敏感字段 redact 或 mask；reflection 输出也要二次审计。
- ❗ **Multi-user system 必须 user-isolated memory**：同一个 vector DB 存所有用户的 memory，搜索时只过滤 user_id 在 client 侧——一旦代码 bug 漏过滤，就会跨 user 召回（"用户 A 的偏好"被推荐给"用户 B"）。**强烈建议 namespace 隔离**：vector DB 按 user_id 分 collection / partition（Milvus / Qdrant 都支持），物理隔离 + 逻辑过滤双保险。
- ❗ **Memory 检索的 noise 可能 mislead agent 决策**：vector retrieve 返回的 top-K 不一定真相关——召回了 4 条无关 memory + 1 条相关，agent 可能被 4 条无关的带偏。**reranker 必加**（与 13.2 一样，bge-reranker-v2-m3 / Cohere Rerank）；retrieval 后 LLM 自己再过一遍"判断 memory 是否真相关"也是常见 pattern。
- ❗ **"Aging strategy"：长期不用的 memory 应当 archive 或 forget**：跑 1 年的 personal assistant 累积百万条 memory，绝大多数永远不会再被召回——但每次 search 都要算相似度，cost 与索引大小线性相关。**冷数据下沉**（移到便宜的 disk-based 索引）+ **stale memory 标记淘汰**（最近 6 个月 0 access 的 importance < 3 的 memory 直接删）是必须做的；Generative Agents 的 decay 机制天然就是 soft aging，但还需要 hard delete pipeline。
- ❗ **importance 打分不要每条都让 LLM 打**：Generative Agents 的设计理论优美但每条 memory 都调一次 LLM 打分，写入 cost 高得离谱。**生产做法**：(1) 启发式（消息长度 / 是否含 NER 实体 / 是否含数字）先粗分；(2) batch LLM 打分（10 条一次）；(3) 只对显式 user-facing 事件（emoji react / 用户主动收藏）打分。
- ❗ **session_id 与 user_id 要分开**：用户在 session A 说"我今天饿了"，下个月 session B 突然召回这条提示——很尴尬。**短期 session-scoped memory 与跨 session user-scoped memory 要分开 namespace**，session 结束按规则决定哪些 promote 成 user-level（importance 高、含偏好类陈述、用户显式 confirm 的）。
- ❗ **不要把 memory 当 ground truth**：memory 是 agent 的"记忆"不是"知识库"——LLM 写入时可能记错（user 说"我 25 岁" 被记成 "26"），后续 retrieve 出来当事实用就出错。**memory 召回的内容应作为 hint 不是 fact**——prompt 模板里写"以下是相关历史 memory，可能不准确，必要时和用户 confirm"。

---

## 5. 经典 paper

- **Packer et al., 2023 — MemGPT: Towards LLMs as Operating Systems** — OS-style virtual memory 搬到 LLM agent 的开山之作，定义了 main context / external context 的分层、page in/out 的 tool 接口、memory pressure 触发机制。当前主流 long-term memory 框架（Letta / mem0 部分思路 / OpenAI ChatGPT memory）都受这篇影响。读 §3-4 看 architecture，§5 看 evaluation。
- **Park et al., 2023 — Generative Agents: Interactive Simulacra of Human Behavior** — 必读。25 个 LLM agent 仿真小镇，提出 memory stream + importance/recency/relevance 三因子检索 + reflection 抽象——这套设计现在是所有 episodic memory 系统的事实模板。读 §4.1-4.3 完整 memory architecture，附录 A 有 prompt 模板可直接借鉴。
- **Wang et al., 2023 — Voyager: An Open-Ended Embodied Agent with LLMs** — Minecraft 里的 lifelong learning agent，提出 skill library = procedural memory 的范式，skill 存的是 code 不是文字。读 §3 看 skill library 设计，启发后来 OpenHands / SWE-agent 的 codebase-aware skill 复用。
- **Shinn et al., 2023 — Reflexion: Language Agents with Verbal Reinforcement Learning** — verbal memory 作为"无 gradient 的 RL"——失败 → reflect → 写入 memory → 下次任务读取。AlfWorld / HumanEval 上比 ReAct baseline 提升 10-30%。这是 memory 与 agent 学习的 bridge，与 9.7 self-rewarding 是同源思想的不同实现。
- **加分阅读 — mem0 / Letta / LangMem 工程实现**：mem0（github.com/mem0ai/mem0）和 Letta（MemGPT 的产品化、github.com/letta-ai/letta）是当前最成熟的两个 OSS memory framework，LangMem 是 LangChain 的 memory 模块——读它们的源码看真实生产怎么处理 user 隔离、aging、PII。

---

## 6. 自测与面试题

**Q1（架构）**：列出 agent memory 的 5 大类型，每一类给一个代表性实现 / 论文。

<details>
<summary>Answer sketch</summary>

按 Tulving-style 认知科学分类：

- **Short-term / Working memory**：当前 task 的临时状态。代表实现：ReAct scratchpad（Yao 2022）、message history、sliding window。
- **Long-term episodic memory**：时间序列的 event / observation。代表实现：vector DB + metadata（mem0 / LangMem）、Generative Agents 的 memory stream（Park 2023）。
- **Long-term semantic memory**：与时间无关的事实 / 知识。代表实现：RAG（13.2）、知识图谱、外接 semantic search。
- **Procedural memory**：可执行技能 / 操作流程。代表实现：Voyager skill library（Wang 2023）、agent skill manager。
- **Self-knowledge / Persona memory**：关于 agent 自己的元信息（性格、能力、偏好）。代表实现：MemGPT 的 working context（Packer 2023）、persistent system prompt。

加分：现代 agent 系统通常组合 2-4 类，不会只用一种；MemGPT 同时覆盖 short-term + episodic + self-knowledge，是 hybrid 范式的代表。

</details>

**Q2（公式）**：写出 Generative Agents 的 memory retrieval 综合 score 公式，解释三个因子各自的作用，为什么 recency 用 exponential decay 而不是 linear / step。

<details>
<summary>Answer sketch</summary>

公式：

$$\text{score}(m, q) = \alpha \cdot \text{importance}(m) + \beta \cdot \text{recency}(m) + \gamma \cdot \text{relevance}(m, q)$$

其中：

- **importance**: LLM 在 memory 写入时给 1-10 分（归一化到 [0,1]），衡量"这条 memory 本身的重要性"——1 是日常琐事、10 是人生大事。这是 **静态** 的。
- **recency**: $\text{recency}(m) = \exp(-\lambda \Delta t)$，$\Delta t$ 是 last access time 到当前的小时数，$\lambda$ 控制衰减速度（Park 2023 用 0.99，对应半衰期 ~100 小时）。这是 **动态** 的，随时间变化；用 last_access 而非 creation time，让"经常被想起"的 memory 不被遗忘。
- **relevance**: 当前 query embedding 与 memory embedding 的余弦相似度。
- 三因子归一化后线性加权（Park 2023 默认 $\alpha=\beta=\gamma=1$），retrieve top-K。

为什么 recency 用 exponential decay：

- **生物学动机**：人类遗忘曲线（Ebbinghaus）就是指数形（短期内快速忘、长期缓慢忘），exp decay 是最简单的拟合。
- **数学性质**：单调可微、半衰期可控、永远 > 0（不会硬截断）；linear decay 会有"突然归零"的悬崖（不自然），step function 完全不可微（嵌入打分系统不平滑）。
- **工程**：exp 公式参数只有一个 $\lambda$，半衰期 = $\ln 2 / \lambda$，调起来直观。

加分：实际系统通常会再 normalize 三个分数到同 scale 防止 importance 量纲压过 relevance，或对每个因子做 min-max scaling。

</details>

**Q3（实战）**：你做一个 personal AI assistant，要跨 6 个月记住用户的偏好、日程、人际关系等。memory 系统怎么设计？请覆盖：架构分层、写入策略、检索策略、aging / 隐私、cost 控制。

<details>
<summary>Answer sketch</summary>

参考方案（hybrid memory 架构）：

**架构分层**：

- **Working context（系统 prompt 一部分）**：固定写入用户的 persona / 长期偏好（约 500 token），如"用户是 Python 工程师、偏好 typed dict、喜欢简洁回答"。LLM 可调 `update_persona` tool 编辑。
- **Session memory（短期）**：当前对话 sliding window 保留最近 20 turn，超出后 summarize 进 long-term。
- **Episodic memory（长期，vector DB）**：所有对话 / 事件按 turn 写入，带 metadata（timestamp、user_id、session_id、type、importance）。Milvus / Qdrant，按 user_id 分 collection（物理隔离）。
- **Semantic memory（长期，knowledge graph + vector）**：抽取出的结构化事实——"用户的同事 Alice、就职 Google"——用 LLM 做 entity extraction 后存到 KG（Neo4j）+ embedding 双写。
- **Skill library（可选，procedural）**：用户常用的 workflow（"帮我每周一早上整理 RSS"）存为可复用 prompt template + tool sequence。

**写入策略**：

- Turn 结束后异步 batch 写入（不阻塞主对话）。
- importance 用启发式 + LLM 分级：含 NER 实体 / 偏好陈述 / "记住"等关键词的优先调 LLM 打分；其他用启发式（长度、是否问题）粗分。
- PII detector（Presidio）先过一遍，敏感字段 mask 后再 embed / 存入。

**检索策略**：

- 主对话 query 时：semantic memory（KG 实体匹配）+ episodic memory（vector retrieve top-100）+ working context（直接读）三路并发；reranker（bge-reranker-v2-m3）精排到 top-10。
- 用 Generative Agents 三因子打分（importance + recency + relevance），不只看 cosine。
- prompt 模板明确"以下是相关 memory，可能不准确，必要时与用户 confirm"。

**Aging / 隐私**：

- session_id 与 user_id 隔离 namespace，跨 user retrieve 完全禁止（vector DB partition + code 双重检查）。
- 6 个月内 0 access 且 importance < 3 的 memory 进 cold storage（archive 但不删除，可恢复）；1 年 0 access 且非显式收藏的真删。
- 用户主动 "忘掉这件事" 走 hard delete API，同时清 KG 与 embedding 两侧。
- LLM 生成的 reflection / summary 必须二次 PII 审计后才能写入。

**Cost 控制**：

- 写入侧：异步 + batch（10 条/批）调 LLM 打 importance；不要每 turn 同步调。
- 检索侧：embedding query 缓存（同一 query 在 5 分钟内复用结果）；reranker 只对 top-100 跑（不要 top-1000）。
- 存储侧：embedding 用 Matryoshka 降到 768 维；冷数据进 IVF-PQ 压缩存储。
- LLM 推理：summary / reflection 用便宜模型（Haiku / Qwen-7B），主对话用 Sonnet / GPT-4o。

加分：上线后接入 memory 质量监控——抽样让 LLM-as-judge 评估"这次的 retrieval 是否真有用"，回流改进 importance 打分逻辑。

</details>

---

## 7. 延伸阅读

- [MemGPT / Letta 官方文档](https://docs.letta.com/) — MemGPT 的产品化版本 Letta 的工程文档，最贴近真实生产部署的 memory framework；可以直接读源码看 main context / archival memory 的具体实现。
- [mem0 GitHub](https://github.com/mem0ai/mem0) — 当前最流行的 OSS memory layer，支持 vector + KG 双写、user namespace 隔离、可对接 LangGraph / OpenAI Assistants；轻量场景的首选。
- [Generative Agents 官方代码与论文](https://github.com/joonspk-research/generative_agents) — Park 2023 的官方实现，可以跑起来看 25 agent 小镇的 memory stream + reflection 真实日志，是理解三因子检索 + reflection 最直观的方式。
- [Anthropic — Introducing Memory in Claude](https://www.anthropic.com/news/memory) — Anthropic 关于 Claude memory feature 的官方介绍，能看到工业级 memory 在产品形态上怎么做（用户可见、可编辑、可关闭）。
- [LangMem / LangGraph Memory](https://langchain-ai.github.io/langgraph/concepts/memory/) — LangGraph 框架对 short-term / long-term memory 的官方抽象，对 hybrid 架构的工程化最友好。
- 推荐继续读本教程的 **14.6 节《Multi-agent》**——多 agent 场景下 memory 还要解决"是否共享"、"冲突合并"等新问题；以及 **15.5 节《Agent 鲁棒性》**——memory poisoning（恶意往 memory 里注入误导信息）是 memory-heavy agent 的新攻击面。
