---
title: "3.2 词表构造、特殊 token、OOV"
description: "词表是 LLM 与文字世界的 ABI——这一节讲清楚现代 LLM 怎么决定词表大小、为什么 byte-level BPE 时代 OOV 几乎被消灭、以及 special token / chat template / 工具调用 token 这套\"人为设计的语义协议\"如何深度影响 SFT 与 agent function calling 的成败。"
---

> ⏱ 预计阅读 25 分钟 ｜ 难度 ★ ｜ 前置：3.1（BPE / WordPiece / Unigram / SentencePiece）

## 一句话本节讲什么

词表是 LLM 与文字世界的 ABI——这一节讲清楚现代 LLM 怎么决定词表大小、为什么 byte-level BPE 时代 OOV 几乎被消灭、以及 special token / chat template / 工具调用 token 这套"人为设计的语义协议"如何深度影响 SFT 与 agent function calling 的成败。

---

## 1. Mental model（直觉）

3.1 讲了 BPE 怎么把字符流压成 token 流，但没回答两个工程问题：**词表到底应该多大？这个表里除了"语言 token"还要塞哪些"协议 token"？**

把词表想成一张**编码本**——左列是 token id（整数），右列是它对应的字节序列或语义角色。这张表一旦训完就**和模型 weights 一起冻结**：

- 左列 size = 词表大小 $V$，决定了 embedding 矩阵 $V \times d$ 的参数量
- 右列前几百个 id 通常**留给"特殊 token"**——它们不对应自然语言，而是让模型理解"这是一段对话的开始 / 这是 user 角色 / 这是工具调用的起点"
- 右列剩下的几万到几十万个 id 才是语言本身的 subword 片段

```
token_id │ surface form         │ 角色
─────────┼──────────────────────┼─────────────
   0     │ <|begin_of_text|>    │ 结构性 special
   1     │ <|end_of_text|>      │ 结构性 special
   2     │ <|start_header_id|>  │ chat template
  ...    │ ...                  │ ...
 256     │ "the"                │ 语言 subword
 257     │ "ing"                │ 语言 subword
  ...
128000   │ "🐶" 的 byte-0       │ byte-level 残片
```

理解这张表后，下面三件事会自动连成一条线：

1. **词表大小**是个 trade-off——大覆盖更广、序列更短，但 embedding 矩阵巨大、训练 / 推理都更贵
2. **special token** 是模型与外部协议的握手——chat template 用一组 token 教模型"哪段是 user 说的、哪段该我说"
3. **OOV** 在 byte-level BPE 时代退化成"罕见 token 拆得很碎、效率低"——而不是"模型完全无法表示"

---

## 2. 词表大小的 trade-off 与 embedding 算账

### 2.1 三档典型词表

现代 LLM 的词表大小可粗略分为三档：

| 档位 | $V$ 量级 | 代表模型 | 适用场景 |
|---|---|---|---|
| 小词表 | 30k–50k | GPT-2 (50257)、BERT (30522) | 英文为主 / 早期模型 |
| 中词表 | 100k–130k | GPT-4 / GPT-4o (~100k cl100k_base)、Llama-3 (128k) | 英中混合 + 代码 |
| 大词表 | 150k–256k | DeepSeek-V3 (129k 实激活但训了 200k+ candidate)、Qwen2.5 (151k)、Gemma-2 (256k) | 多语言 + 代码全覆盖 |

**词表大决定一切的不是模型规模而是目标语言数量与领域**：纯英文模型 32k 就够；要覆盖中文、日文、阿拉伯文、code 注释里各种 unicode 符号，没有 100k 以上几乎不可能做到 token 效率不塌。

### 2.2 embedding 矩阵的算账

设 hidden size $d$、词表大小 $V$，则 embedding 参数量为 $V \times d$。把 Llama-3 8B 拿出来算：

$$
V \times d = 128000 \times 4096 \approx 5.24 \times 10^8 \approx 524\text{M}
$$

而模型总参数 8B，也就是说 **embedding 占了约 6%**。换成 Gemma-2 9B 用 256k 词表：

$$
256000 \times 3584 \approx 9.18 \times 10^8 \approx 918\text{M}
$$

接近模型 1/10 的参数都花在 embedding 上。再放大到 70B 模型 + 256k 词表 + $d=8192$：

$$
256000 \times 8192 \approx 2.1 \times 10^9 \approx 2.1\text{B}
$$

**embedding 矩阵单独就 2B 参数**——这就是为什么大模型也不会无脑往 256k 堆词表。

### 2.3 weight tying：input 与 output 共享

经典 trick 来自 Press & Wolf 2017、Inan et al. 2017：**input embedding 矩阵 $E \in \mathbb{R}^{V \times d}$ 与 output projection 矩阵 $W_o \in \mathbb{R}^{d \times V}$ 共享参数**（$W_o = E^\top$），可以省掉一半 embedding 参数。GPT-2 就是 tied 的。

但 **Llama / Qwen / DeepSeek 现在多不 tie**——实测不 tie 的 perplexity 略好，参数代价（再多一份 $V \times d$）在大模型时代相对可接受。Gemma 仍 tie，是因为它走"小模型 + 大词表"路线，省下的参数能让 hidden 多堆一点。

### 2.4 大词表的隐藏成本

- **训练显存**：embedding gradient 也要存，AdamW 的两个 moment buffer 把 embedding 显存占用 ×3
- **softmax 计算**：output 端 $h W_o$ 的输出是 $\mathbb{R}^V$，$V$ 越大每一步推理 logit 计算越贵
- **稀疏梯度优化**：embedding gradient 是稀疏的（每个 batch 只触及少量 token），可以用 sparse optimizer 或 fused embedding bag 显著加速

---

## 3. 特殊 token 体系

special token 是"非语言、纯协议"的 token——它们的作用不是表达内容，而是**告诉模型"现在的语境是什么"**。按用途可分为以下几类。

### 3.1 结构性 token

最古老的一组，几乎所有 LLM 都有：

- `<bos>` / `<s>` / `<|begin_of_text|>`：序列起点，告诉模型"从这里开始"
- `<eos>` / `</s>` / `<|end_of_text|>`：序列终点，**生成时模型采到这个 token 就停**
- `<pad>`：padding 占位，配合 attention mask 让 batch 内不同长度对齐
- `<unk>`：unknown token，**byte-level BPE 时代基本不用**（因为没有 OOV）

### 3.2 chat template token

这一组是后训练时代才出现的**对话协议**——让模型理解 multi-role 对话结构。三种主流风格：

**Llama-3 风格**（精细、role 用 header 包裹）：

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

What is 2+2?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

4<|eot_id|>
```

注意 `<|eot_id|>`（end-of-turn）与 `<|end_of_text|>`（end-of-sequence）是**两个不同 token**——前者是"这一轮发言结束"，后者才是"整段文本结束"。SFT 时 loss 应该让模型学会输出 `<|eot_id|>` 来终止当前 turn。

**Qwen / ChatML 风格**（更紧凑）：

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is 2+2?<|im_end|>
<|im_start|>assistant
4<|im_end|>
```

**Llama-2 / 早期风格**（用文本 marker，不依赖 special token）：

```
<s>[INST] <<SYS>>
You are a helpful assistant.
<</SYS>>

What is 2+2? [/INST] 4 </s>
```

### 3.3 工具调用 token

function calling 的两种工程化路径：

- **新增 special token 路线**（Qwen2.5 / Hermes）：直接在词表里加 `<tool_call>` `</tool_call>` `<tool_response>` `</tool_response>`，模型生成时一旦输出 `<tool_call>` 框架层就知道要解析后续 JSON 并执行工具
- **JSON 字段路线**（OpenAI / Anthropic）：不加新 token，靠 API 层把"是不是 tool call"作为 message 的字段返回，前端拆出来

第一种路线对开源 SFT 更友好（loss mask 容易写）；第二种对闭源模型更灵活（不用动 tokenizer）。

### 3.4 reasoning token

R1 时代的产物——用 `<think>` `</think>` 包裹 long-CoT，让框架层能把"思考过程"与"最终答案"分开渲染。DeepSeek-R1、Qwen QwQ、OpenAI o-series 都用类似设计（虽然 o-series 不开源具体 token）。

### 3.5 mask token

`[MASK]`——BERT/MLM 时代的产物，CLM-only 的现代 LLM 不用。但它仍是面试常问的概念点（理解 MLM 训练目标的关键）。

---

## 4. OOV 的演化：从噩梦到非问题

OOV (out-of-vocabulary) 是经典 NLP 的核心痛点之一。看演化：

- **word-level**（word2vec / GloVe 时代）：词表只有数万常见词，遇到生僻词、专有名词、数字、错拼直接丢给 `<unk>`，**模型完全无法表示**——OOV 是噩梦
- **WordPiece / 字符级 BPE**（BERT / GPT-2 早期）：拆到 subword，最差也能拆到字符，OOV 大幅减少；但还是有"该字符不在词表里"的可能（如某个罕见 Unicode）
- **byte-level BPE**（GPT-2 / GPT-3 / Llama / Qwen）：**词表的 base unit 是 256 个字节而非字符**，UTF-8 编码下任何字符串都能用字节序列表示，**理论上 0 OOV**

具体看 emoji "🐶" 怎么被处理：它的 UTF-8 编码是 `F0 9F 90 B6`（4 个字节）。byte-level BPE 词表里这 256 个字节都有 id，所以 "🐶" 最差被拆成 4 个 byte token；如果 BPE 在训练时见过这个组合，会合并成 1-2 个 token。**永远不会出 `<unk>`**。

但 byte-level BPE 没解决"**罕见 token 拆得很碎、效率低**"的问题——纯英文 BPE 看到中文字符往往要拆 2-3 个 byte token / 字，看到某些罕见 emoji 拆 4 个。这就是为什么纯英文 tokenizer 在中文上的"token 效率"远低于专门训练的中文 tokenizer——一句"我爱人工智能"在 GPT-2 tokenizer 上要 12-15 个 token，在 Qwen tokenizer 上只要 4-5 个。**词表设计直接决定推理成本**（按 token 计费的模型尤其敏感）。

---

## 5. 最小代码示例

### 5.1 给模型加 special token + resize embedding

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")

# 加 5 个新 tool-related special token
new_tokens = ["<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>", "<plan>"]
n_added = tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})

# 关键：必须 resize，否则新 token id 超出 embedding 行数 → out-of-bounds
model.resize_token_embeddings(len(tokenizer))

# trick：手工初始化新 token 的 embedding 为相似现有 token 的均值，加速收敛
with torch.no_grad():
    seed_ids = tokenizer.encode("tool call", add_special_tokens=False)
    seed_emb = model.get_input_embeddings().weight[seed_ids].mean(dim=0)
    for tok in new_tokens:
        tid = tokenizer.convert_tokens_to_ids(tok)
        model.get_input_embeddings().weight[tid] = seed_emb.clone()
```

### 5.2 chat template 实战

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

messages = [
    {"role": "system", "content": "You are a helpful coding assistant."},
    {"role": "user", "content": "用 Python 写 quicksort"},
    {"role": "assistant", "content": "def quicksort(arr): ..."},
    {"role": "user", "content": "再用 Rust 写一遍"},
]

# apply_chat_template 会按模型自带的 jinja 模板拼字符串
prompt_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,  # 末尾追加 "<|im_start|>assistant\n" 引导生成
)
print(prompt_text)
# <|im_start|>system\nYou are ...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n...

# tokenize=True 直接拿 input_ids
input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
print(input_ids.shape, input_ids[0, :15])
```

`add_generation_prompt=True` 是推理时的关键——它会在末尾追加 assistant 角色的 header，让模型"接着"往下生成。SFT 时则用 `False`，因为 assistant 内容已经在 messages 里了。

### 5.3 byte-level BPE 处理中文与 emoji

```python
from transformers import AutoTokenizer

tok_en = AutoTokenizer.from_pretrained("gpt2")            # 英文 BPE
tok_zh = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B") # 中文友好

text = "我爱AI🤖"
print("GPT-2:", tok_en.encode(text), "->", len(tok_en.encode(text)), "tokens")
print("Qwen :", tok_zh.encode(text), "->", len(tok_zh.encode(text)), "tokens")
# GPT-2 大概 9-12 个 token（中文 + emoji 都拆 byte）
# Qwen  大概 4-5 个 token（中文进了词表，emoji 仍可能 2-4 byte）
```

跑一下你会直观感受到"中文友好 tokenizer"省下的成本——按 token 计费的 API 调用，词表选错可以直接让账单翻倍。

---

## 6. 工程踩坑与经验

- ❗ **加 special token 不 resize embedding** → 训练第一步 `IndexError: index out of range`。`add_special_tokens` 后**永远跟一句** `model.resize_token_embeddings(len(tokenizer))`
- ❗ **SFT 用错 chat template** → 推理时模型停不下来，狂输出直到 `max_new_tokens`。原因是模型没学到该 template 的终止符（如 `<|eot_id|>`）。**SFT 与推理必须用同一个 `apply_chat_template`**，不要手拼字符串
- ❗ **混淆 `<|eot_id|>` 与 `<|end_of_text|>`**（Llama-3）：前者是 turn 结束、后者是序列结束。生成时若把 stop token 只设成 `<|end_of_text|>`，模型会把多轮对话连成一段不停吐
- ❗ **`tokenizer.pad_token` 默认 None**（GPT 系）：SFT 时常见做法 `tokenizer.pad_token = tokenizer.eos_token`，但这会让 attention mask 把 pad 也当成 eos，**loss 计算时一定要把 pad 位置的 label 设成 -100**，否则模型会被训成"输出一堆 eos"
- ❗ **多卡训练 ckpt 漏存 tokenizer** → resume 时 vocab 不一致 → embedding 对不上 → loss 直接爆炸或胡言乱语。`save_pretrained` 时模型与 tokenizer **必须存同一目录**
- ❗ **跨模型共享 LoRA 时 tokenizer 不同** → token id 完全错位，LoRA 学到的"在 id 12345 上的偏移"在新模型变成另一个语义。LoRA 只能在同 tokenizer 的 base 上迁移
- ❗ **中文场景用纯英文 tokenizer 微调** → token 效率低 2-3×，相同 sequence length 装的有效内容少 2-3×，长上下文与推理速度同步塌；如果要做中文，从词表层就要选对 base
- ❗ **Llama / Qwen 词表里有 reserved special token 槽位**（如 Llama-3 的 `<|reserved_special_token_0..250|>`）——加自己的 special token 可以**复用这些预留 id**，不需要 resize embedding，避免破坏分布式 ckpt 的 sharding
- ❗ **Agent 场景下罕见 tool name 容易拆 token 错**：如 tool name 取个 `analyze_pareto_frontier_v2` 可能被拆成 6-8 个 token，模型生成时拼错一个字符就调用失败。给 tool 取**短、常见、无下划线断片**的名字（`search_web` 比 `web_search_tool_v2` 鲁棒）

---

## 7. 经典 paper

- **Touvron et al., 2023 — Llama 2 / Llama 3 Tech Report** — chat template 设计的现代范本，Llama-3 把 header / eot / eos 三件套定义清楚，是后续所有开源模型的参考。读 §"Instruction Tuning" / "Tokenizer" 两节即可。
- **Yang et al., 2024 — Qwen2.5 / Qwen Technical Report** — 多语言大词表 + 工具 token 设计的最佳工程参考，特别是 §"Tokenization" 与 §"Tool Use"。Qwen 的 ChatML + tool_call 设计影响了大量国产开源模型。
- **Press & Wolf, 2017 — Using the Output Embedding to Improve Language Models** + **Inan et al., 2017 — Tying Word Vectors and Word Classifiers** — weight tying 的原典，理解为什么早期 LM 默认 tie、现代大 LM 又放弃 tie。两篇加起来 < 20 页，必读。

---

## 8. 自测与面试题

**Q1（概念）**：byte-level BPE 是怎么"消灭" OOV 的？以 emoji "🐶" 为例说明它如何被编码。

<details>
<summary>Answer sketch</summary>

- byte-level BPE 的 base unit 不是字符而是 **256 个 UTF-8 字节**——这 256 个 id 一定在词表里
- 任何 Unicode 字符都能用 UTF-8 字节序列唯一表示，所以**不存在"无法编码"的输入**
- 例："🐶" 的 UTF-8 编码是 `F0 9F 90 B6` 四个字节
  - 最差情况：词表没合并过这个组合 → 拆成 4 个 byte token（如 token id `240, 159, 144, 182`）
  - 优化情况：BPE 训练时见过 → 合并成 1-2 个 token
- 加分：byte-level BPE 没消灭"罕见 token 效率低"的问题——这就是 GPT 类按 token 计费时罕见字符成本翻倍的原因

</details>

**Q2（实战）**：你给一个 base model 加了 5 个新的 tool special token，列出从 `add_token` 到能正常 SFT 的完整步骤。

<details>
<summary>Answer sketch</summary>

完整流程至少 5 步：

1. **加 token**：`tokenizer.add_special_tokens({"additional_special_tokens": [...]})`
2. **resize embedding**：`model.resize_token_embeddings(len(tokenizer))`——否则 forward 时 `IndexError`
3. **（可选）手工初始化新 embedding**：用相似现有 token embedding 的均值替换随机初始化，可显著加速收敛
4. **改 chat template / 数据**：把训练数据的 tool call 段用新 token 包裹（如 `<tool_call>{...}</tool_call>`），并相应修改 `tokenizer.chat_template`（jinja 字符串）
5. **设 loss mask**：assistant 回复（含 tool_call）部分参与 loss，user / system / tool_response 部分 label = -100
6. **存 ckpt 时同时保存 tokenizer**：`tokenizer.save_pretrained(ckpt_dir)` + `model.save_pretrained(ckpt_dir)`，不然 resume 或下游推理对不上 vocab
7. 加分：训练完检查模型在新 token 上的输出概率、确认没把新 token 当成 garbage

</details>

**Q3（trade-off）**：词表 32k vs 256k，从 (embedding 占总参数比例 / 推理速度 / 多语言覆盖 / 训练数据需求) 4 个维度对比。

<details>
<summary>Answer sketch</summary>

按 4 个维度逐条分析：

- **embedding 参数占比**：32k 在 7B 模型上 $32000 \times 4096 \approx 131\text{M}$，约 1.9%；256k 则 ~ 1B 参数，约 14%。大词表在小模型上极不划算
- **推理速度**：每步 forward 末尾的 $hW_o$ 输出 logit 维度是 $V$，softmax + sampling 都随 $V$ 线性变慢；32k 比 256k 快约 8×（仅这一项）。但**序列长度**上 256k 有优势——同样内容 token 数少 2-3×，整体反而可能更快（取决于 prefill / decode 阶段哪个 dominant）
- **多语言覆盖**：32k 几乎只能服务英文 + 简单代码；256k 能塞下中文 / 日文 / 阿拉伯 / 希伯来 / 大量 unicode 符号。要做多语言模型 256k 是底线
- **训练数据需求**：大词表的每个 token 在训练数据中出现次数被稀释，**需要更多训练 token 才能让稀有 vocab 的 embedding 训得好**——256k 词表配 1T token 训练数据可能不够，需要 5T+
- 加分：还有一个隐藏维度——**SFT/RL 阶段的 loss 稳定性**。大词表下 logit 分布更"散"，KL / 熵相关计算的数值范围更大，PPO / GRPO 的 ratio 容易溢出，需要更小心的 logit clip

</details>

---

## 9. 延伸阅读

- [HuggingFace Chat Template 官方文档](https://huggingface.co/docs/transformers/main/chat_templating) — `apply_chat_template` 的所有用法、jinja 语法、自定义 template 的工程参考
- [tiktoken（OpenAI 的 BPE 实现）](https://github.com/openai/tiktoken) — 看 cl100k_base / o200k_base 的 vocab 结构，理解 GPT-4 / GPT-4o 是怎么设计的
- [Qwen 官方 chat template + tool call 文档](https://qwen.readthedocs.io/) — 国产开源生态最规范的 tool token 设计，agent SFT 的 reference
- [Andrej Karpathy — Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE) — 2 小时手撕 byte-level BPE + tiktoken 内部，把 3.1 + 3.2 串起来的最佳视频
- 推荐继续读本教程的 **3.3 节《Batching / packing / 注意力 mask》**——把 token 流变成可训练 batch 的最后一公里
