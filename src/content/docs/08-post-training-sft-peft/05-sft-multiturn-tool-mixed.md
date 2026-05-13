---
title: "8.5 SFT 实战：多轮对话 + tool 调用混合训练"
description: "把 8.1-8.3 学的全拼起来，端到端训一个既能多轮对话又能调工具的 model——核心难点不是训练循环，而是统一不同来源数据的 chat template、给 tool_call / tool observation 段写正确的 loss mask、平衡通用对话与 tool calling 数据比例；这一节是 Module 14 Agent / Module 15 Agent RL 的\"冷启动"
---

> ⏱ 预计阅读 50 分钟 ｜ 难度 ★★★ ｜ 前置：8.1（SFT 数据）、8.2（chat template / loss mask / packing）、8.3（LoRA / QLoRA）、3.2（special token / chat template）

## 一句话本节讲什么

把 8.1-8.3 学的全拼起来，端到端训一个**既能多轮对话又能调工具**的 model——核心难点不是训练循环，而是**统一不同来源数据的 chat template、给 tool_call / tool observation 段写正确的 loss mask、平衡通用对话与 tool calling 数据比例**；这一节是 Module 14 Agent / Module 15 Agent RL 的"冷启动"基线，没这一步直接做 RL 几乎学不会。

---

## 1. Mental model（直觉）

8.1-8.3 三节像在攒乐高积木：8.1 教你怎么准备 SFT 数据、8.2 教你 chat template / loss mask / packing 这些训练侧基本功、8.3 给你 LoRA / QLoRA 把显存压到平民配置。本节把这些拼到一个**完整工程场景**里——训一个能"chat + 调工具"的现代 chat model。

为什么"对话 + 工具"是现代 LLM 训练的事实标准？因为 GPT-4 / Claude 之后的所有商用 chat model **本质上都是 agent**——用户提一个 query，模型可以选择直接答（chat 能力），也可以选择先调一个 tool（搜索 / 计算 / 读文件）拿到 observation 再答（tool use 能力），还可以多轮往复（multi-turn 能力）。这三件事**必须在 SFT 阶段一起训**，单独训某一项都会让模型在融合场景下失败：

- 只训 chat → 模型遇到 "今天北京天气?" 这类需要外部数据的问题硬编一个 "23°C 晴"（幻觉）
- 只训 tool → 模型遇到 "你好" 也尝试调 tool（over-call），自然对话能力退化
- 只训单轮 → 模型在多轮对话里看不见前文上下文（context handoff 失败）

把这三件事放在一张图里理解：

```
              用户视角的"agent 行为"
   ┌─────────────────────────────────────────────┐
   │  user: 北京今天多少度？                     │
   │  asst: <tool_call>{"name":"get_weather",    │  ← model 学会"何时调"
   │         "arguments":{"city":"北京"}}        │     + "调什么 tool"
   │         </tool_call>                        │     + "args 怎么填"
   │  tool: {"temperature":"22°C","cond":"晴"}   │  ← framework 注入
   │  asst: 北京今天 22°C，天气晴朗。            │  ← model 学会"消化 obs"
   │  user: 那上海呢？                           │     + "组织 final answer"
   │  asst: <tool_call>{"name":"get_weather",    │  ← multi-turn 一致性
   │         "arguments":{"city":"上海"}}        │
   │         </tool_call>                        │
   └─────────────────────────────────────────────┘
                     ↑
             SFT 数据要覆盖这整段流程
```

SFT 视角下，这条 trajectory 就是一条 multi-turn 序列，每个 role（`user` / `assistant` / `tool`）有不同的处理：**`user` 段**是输入（mask），**`tool` 段**是 observation（mask，因为是环境给的不是 model 生成的），**`assistant` 段**——包括 `tool_call` 和 final answer——**全部算 loss**。这套 loss schema 和 8.2 讲的"all-turns + completion-only loss"一脉相承，只是把 role 类型从 2 种（user/assistant）扩展到 3 种（user/assistant/tool）。

> 与 13.4 区分：13.4 讲 function calling 的工程接口（JSON schema、并行调用、constrained decoding），是**推理时的 tool 协议**；本节讲怎么把 tool 能力**训进 model 权重**。
> 与 14.3 区分：14.3 讲 Tool Use 训练范式（Toolformer / Gorilla / xLAM 等方法论），本节讲拿到这些数据后**端到端 SFT 一遍的工程实战**。
> 与 15.1 区分：15.1 讲 Agent SFT（FireAct / Agent-FLAN 等"长 trajectory"训练），本节是它的**前置基础**——只覆盖单步 tool call，多步 plan-and-execute 留到 15.1。

---

## 2. 场景设定与数据 mix

### 2.1 目标 model 的能力面

设定一个具体目标：训一个 7-8B 规模的 chat model，能力清单：

1. **通用 chat**：流畅的多轮对话、按指令回答、format 多样（markdown / code / list）
2. **Tool calling**：给定 N 个 tool definition，能正确选择调用哪个、参数 JSON 合法、能消化 tool observation 给出 final answer
3. **多轮一致性**：3-5 轮对话内保持 context、tool call 历史可被复用
4. **Refuse-when-no-tool**：遇到 tool 不能解决的问题（"你今天心情如何？"）不要硬调 tool

base model 选择：**Qwen2.5-7B**（chat template 标准化、tool token 内置）或 **Llama-3.1-8B**。本节代码以 Qwen2.5 为主（国内更常用、tool 设计更工程化）。

### 2.2 数据 mix 配方

参考 Tülu-3 / xLAM / ToolACE 的公开配方，给出一个能直接抄的 mix：

| 类别 | 数据集示例 | 比例 | 作用 |
|---|---|---|---|
| 通用对话 | Magpie-Pro / Tülu-3-SFT-mix / OpenHermes-2.5 | **60%** | 维持基本 chat 流畅性、format 多样 |
| Tool calling | xLAM-function-calling-60k / ToolACE / Hermes-Function-Calling-V1 | **30%** | 单 tool / 并行 tool / sequential tool |
| Reasoning | NuminaMath-CoT / OpenThoughts | **10%** | math / 多步推理（防止 SFT 后 reasoning 退化） |

总量 ~500k-1M 条样本，对 7B 模型 1-2 epoch 足够。

为什么不能 100% 上 tool 数据？**通用 chat 能力会塌**——纯 tool 数据 SFT 出的 model 见到任何 query 都倾向输出 `<tool_call>`，连 "你好" 都试图 search。这是 ToolACE paper 在 ablation §5.3 里专门讨论的现象，社区通用配比是 tool 数据 ≤ 30%-40%。

为什么要混 reasoning 数据？两个原因：(1) tool calling 本质需要"先思考要调哪个 tool、参数怎么填"，这一步是隐式 CoT；(2) 通用 chat 数据 reasoning 密度低，纯 chat + tool SFT 后 model 在 GSM8K / MATH 上分数会跌。

### 2.3 不同 tool 数据集格式速览

主流 tool calling 数据集格式各不相同，**SFT 前必须先 unify 成统一 ChatML 风格**：

**xLAM (Salesforce 2024)**：每条 `(query, tools, answers)` 三元组，answers 是结构化 JSON list：

```json
{
  "query": "Get the weather in Beijing",
  "tools": "[{\"name\": \"get_weather\", \"description\": \"...\", \"parameters\": {...}}]",
  "answers": "[{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Beijing\"}}]"
}
```

**ToolACE (Liu et al. 2024)**：multi-turn 真实对话，每个 assistant turn 可包含 0-N 个 tool_call，已经是 messages list 形式（贴近 OpenAI function calling 风格）。

**Hermes-Function-Calling-V1 (NousResearch)**：直接以 ChatML 序列化，`<tool_call>...</tool_call>` 标签已经嵌好，可以最少改动直接训。

**OpenAI function calling 标注**：messages 里 assistant role 的 `function_call` / `tool_calls` 字段是结构化对象（不在 content 里），需要序列化成字符串才能进 SFT。

**统一目标**：把上面所有格式转成 Qwen ChatML 风格的 messages，每个 message 形如 `{"role": "...", "content": "...", "tools": ...}`，assistant 段里的 tool_call 用 `<tool_call>{"name": ..., "arguments": ...}</tool_call>` 包裹（与 Qwen2.5 官方 chat template 对齐）。

---

## 3. Tool calling 的 token 化协议

### 3.1 Qwen2.5 的 tool calling 格式

Qwen2.5 把 tool 协议直接编进 chat template（jinja 模板会自动渲染），完整一条 trajectory 长这样：

```
<|im_start|>system
You are Qwen, a helpful assistant.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "get_weather", "description": "Get current weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
</tools>

For each function call, return a json object with function name and arguments
within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
<|im_start|>user
北京今天天气?<|im_end|>
<|im_start|>assistant
<tool_call>
{"name": "get_weather", "arguments": {"city": "北京"}}
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
{"temperature": "22°C", "condition": "晴"}
</tool_response><|im_end|>
<|im_start|>assistant
北京今天 22°C，天气晴朗。<|im_end|>
```

几个关键点：

- **tool 定义放在 system prompt 里**，用 `<tools>...</tools>` XML 标签包裹（一条 system 可放多个 tool）
- **assistant 输出 tool_call 也用 XML 标签**：`<tool_call>{"name": ..., "arguments": ...}</tool_call>`，内部是合法 JSON
- **tool observation 的 role 是 `user`**（不是 `tool`），content 用 `<tool_response>...</tool_response>` 包裹——这是 Qwen2.5 的设计选择，比 OpenAI 的独立 `tool` role 更省 token
- **multi-tool 并行调用**：assistant 一段 content 里连续输出多个 `<tool_call>...</tool_call>`，inference 框架解析后并发调用

### 3.2 与其它格式的对比

不同 model 家族的 tool 协议有显著差异：

| Model | tool definition 位置 | tool call 标记 | tool observation role |
|---|---|---|---|
| **Qwen2.5** | system prompt + `<tools>` | `<tool_call>...</tool_call>` | `user` + `<tool_response>` |
| **Llama-3.1** | system prompt（自由格式） | `<\|python_tag\|>{"name":...}` | `ipython` role |
| **Hermes** | system prompt | `<tool_call>...</tool_call>` | `tool` role |
| **OpenAI** | API `tools` 字段（不在 prompt） | message 的 `tool_calls` 字段 | `tool` role |
| **Mistral** | system prompt | `[TOOL_CALLS]...` | `[TOOL_RESULTS]` |

**SFT 实战的铁律**：选定一个目标 model 后，**严格按它的 chat template 训**——把不同来源数据全 unify 到这个 template。混训不同 template（如一半 Qwen 一半 Llama-3.1 格式）会让 model 学糊，**这是新人最常踩的"chat template 错位"坑**（8.2 §10 已经强调过）。

### 3.3 special token 注册（与 3.2 呼应）

Qwen2.5 已经把 `<tool_call>` `</tool_call>` `<tool_response>` `</tool_response>` 内置到 tokenizer 里（保留 special token 槽位预留过），不需要 add token。但如果你换 base model（如 Qwen2-Base 没有这些 token，或者自己加 `<plan>` 等扩展 token），**必须按 3.2 §5.1 那一套先 add_special_tokens + resize_embedding，再开始 SFT**：

```python
new_tokens = ["<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>"]
n_added = tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
if n_added > 0:
    model.resize_token_embeddings(len(tokenizer))
    # 用相似 token embedding 均值 init 新 token，加速收敛
    with torch.no_grad():
        seed_ids = tokenizer.encode("tool call json", add_special_tokens=False)
        seed_emb = model.get_input_embeddings().weight[seed_ids].mean(0)
        for tok in new_tokens:
            tid = tokenizer.convert_tokens_to_ids(tok)
            model.get_input_embeddings().weight[tid] = seed_emb.clone()
            model.get_output_embeddings().weight[tid] = seed_emb.clone()
```

---

## 4. Loss mask schema：哪些算 loss、哪些 mask

这是 agent SFT 与普通 SFT 的**核心差异点**，必须吃透。

### 4.1 三类 token 的 loss 归属

把一条 multi-turn trajectory 分成 5 类 token，对应不同 loss 处理：

| Token 来源 | 例子 | 算 loss？ | 原因 |
|---|---|---|---|
| **System prompt（含 tool 定义）** | `You are Qwen...<tools>...</tools>` | ❌ mask | 是先验设定，不是 model 生成的内容 |
| **User query** | `北京今天天气?` | ❌ mask | 是用户输入 |
| **Assistant tool_call** | `<tool_call>{"name":"get_weather",...}</tool_call>` | ✅ **算 loss** | 教 model 学会"何时调、调什么、参数怎么填" |
| **Tool observation** | `<tool_response>{"temperature":"22°C"}</tool_response>` | ❌ **mask** | 是环境/系统返回的内容，不是 model 生成的 |
| **Assistant final answer** | `北京今天 22°C，天气晴朗。` | ✅ **算 loss** | 教 model "如何消化 obs 给最终答复" |

最容易出错的是第 4 条——**tool observation 段必须 mask**。如果不 mask，model 会被训成"复读 tool obs"：推理时遇到 `<tool_response>{...}</tool_response>` 之后，model 会先把 obs 内容原样吐一遍再生成答复。这是 agent SFT 的高频 bug，xLAM paper §4 专门说过这个 issue。

### 4.2 多轮 trajectory 完整 label 标注

举一个 4 轮的实例（user / asst-tool_call / tool obs / asst-final / user / asst-tool_call / tool obs / asst-final），把 labels 列出来：

```
position │ token segment                            │ label
─────────┼──────────────────────────────────────────┼────────
   0..a  │ <|im_start|>system\n...<tools>...        │ -100
   a..b  │ <|im_start|>user\n北京天气?<|im_end|>    │ -100
   b..c  │ <|im_start|>assistant\n                  │ -100  (header 不算)
   c..d  │ <tool_call>{...}</tool_call><|im_end|>   │ ✅ 算 loss
   d..e  │ <|im_start|>user\n<tool_response>...     │ -100  (含 obs)
   e..f  │ </tool_response><|im_end|>               │ -100
   f..g  │ <|im_start|>assistant\n                  │ -100
   g..h  │ 北京今天 22°C ...<|im_end|>              │ ✅ 算 loss
   h..i  │ <|im_start|>user\n上海呢?<|im_end|>      │ -100
   i..j  │ <|im_start|>assistant\n                  │ -100
   j..k  │ <tool_call>{...}</tool_call><|im_end|>   │ ✅ 算 loss
   ...   │ ...                                       │ ...
```

注意：
- **assistant header** `<|im_start|>assistant\n` 自身 mask（与 8.2 §4.1 同样原则）
- **`<|im_end|>` 必须算 loss**——教 model 学会"何时停"
- **multi-turn all-turns**：每个 assistant 段（tool_call 与 final answer）都算 loss，不只对最后一轮
- 即使 tool_response 在 Qwen2.5 设计里 role 是 `user`，content 部分的 obs JSON 也要 mask 掉

### 4.3 自动化实现：复用 Qwen2.5 chat_template + 手工切 assistant boundary

最干净的实现是在 `apply_chat_template` 后**用 token 序列匹配 assistant boundary**——找到每对 `<|im_start|>assistant\n` 与 `<|im_end|>`，把这之间的 token 标记为 loss 段。完整代码见 §6.2。

注意一个微妙的 case：**Qwen2.5 把 tool observation 也用 `<|im_start|>user` 包裹**，你不能简单地"所有 user role 全 mask"——要用 content 内是否含 `<tool_response>` 区分（其实因为 user 段全 mask、obs 段也该 mask，结果一致；这里强调是为了避免反向逻辑出错）。Llama-3.1 把 obs 放在 `ipython` role 里，更直观但要单独处理。

---

## 5. Tool 数据合成：xLAM / ToolACE 的范式

真实场景下高质量 tool 数据极度稀缺——人工标 1k 条 tool call 成本远高于标 1k 条 chat。**2024 年后主流路线是 LLM 合成 + 校验**，xLAM 和 ToolACE 是两个最有代表性的工作。

### 5.1 Self-Instruct for Tool（xLAM 风格）

xLAM 的数据合成 pipeline 概括成 4 步：

1. **Tool collection**：从 RapidAPI / OpenAPI specs / 自己写的 mock 函数，收集 ~20k 个 tool definition（JSON schema 形式）
2. **Query synthesis**：给 LLM 几个 tool definition + few-shot example，让它生成"用户可能怎么问这些 tool"的 query
3. **Answer synthesis**：让强 LLM（GPT-4o / DeepSeek-V3）按 query + tool list 生成正确的 tool call JSON
4. **Verification**：可执行的 tool 真正调用一次（schema check + 部分 tool 真 call API），过滤生成错误的样本

xLAM 60k 数据集就是这套 pipeline 跑出来的，ablation 显示在 BFCL 上比纯 ToolBench 数据训的高 5-10 个点。

### 5.2 ToolACE：covering 多场景

ToolACE 的核心 insight：tool calling 有几种**正交的能力维度**，必须各自覆盖才能学全：

| 能力 | 例子 | 训练数据要求 |
|---|---|---|
| **单 tool 单调用** | 调一个 weather API | 最基础，~50% 占比 |
| **Parallel call** | 同时查 5 个城市天气 | 一个 assistant turn 内并列多个 `<tool_call>` |
| **Sequential call** | 先 search 再 read 再 summarize | 多个 turn，每 turn 输出一个 call，依赖前 turn 的 obs |
| **Refuse to call** | 用户问"你叫什么"，不该调 search | assistant 直接 chat，不输出 tool_call |
| **Hallucinated tool 防御** | 用户要求调一个不在 tool list 里的工具 | assistant 应当拒绝或解释 tool 不可用 |
| **Param 完整性** | 用户没提供必填 arg，应当反问 | assistant 输出反问而非乱填 args |

每个维度单独看都不复杂，**问题在于全覆盖**——纯单 tool 训的 model 在 parallel call 上得 0 分；纯 happy path 训的 model 见到 hallucinated tool 也乖乖照调。社区共识：tool 数据集设计时**显式按维度配比**（如 parallel 15%、sequential 15%、refuse 10% ……），ToolACE 的 11k 数据就是这么设计的。

### 5.3 Negative example 的重要性

**Refuse-when-no-tool** 这条单独拿出来强调：必须有 negative case。

设想一个失败的 SFT 数据集：里面所有 user query 都是"能用 tool 解决的"，所有 assistant 都以 `<tool_call>` 开头。训完后 model 学到的隐含 pattern 是"只要看到 user query 就 emit `<tool_call>`"——见到 "Tell me a joke" 也尝试调 search_engine。

修复方法：在数据 mix 里强制保证 **15-20% 的样本是"明确不需要 tool"的 chat-only**。这部分可以直接从通用 chat 数据来，但要把 system prompt 里的 tool list 留着（让 model 看到 tool 也学会"不调"），不能直接拿无 tool 的 chat 样本（那训不出"看见 tool 但不调"的能力）。

---

## 6. 完整 SFT script

下面给三段核心代码：(1) 完整 SFT 训练脚本、(2) 数据格式 unify 函数、(3) loss mask 构造函数。组合起来就是一个能跑通的 agent SFT pipeline。

### 6.1 完整 SFT 训练脚本（≤ 80 行）

```python
# agent_sft_train.py
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

MODEL = "Qwen/Qwen2.5-7B"

# 1. tokenizer + pad token
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = "<|endoftext|>"  # Qwen 自带 pad，不要复用 eos

# 2. base model + bf16 + FA2
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",
)

# 3. 多源数据加载 + unify 成统一 messages 格式
chat_ds = load_dataset("Magpie-Align/Magpie-Pro-300K-Filtered", split="train[:60000]")
tool_ds = load_dataset("Salesforce/xlam-function-calling-60k", split="train[:30000]")
math_ds = load_dataset("AI-MO/NuminaMath-CoT", split="train[:10000]")

# unify_to_messages 见 §6.2，把不同源转成 {"messages": [...]} 列
chat_ds = chat_ds.map(unify_to_messages, remove_columns=chat_ds.column_names)
tool_ds = tool_ds.map(unify_to_messages, remove_columns=tool_ds.column_names)
math_ds = math_ds.map(unify_to_messages, remove_columns=math_ds.column_names)

# concat + shuffle（保持 60/30/10 比例）
ds = concatenate_datasets([chat_ds, tool_ds, math_ds]).shuffle(seed=42)

# 4. LoRA config（target 全 7 个 linear）
lora = LoraConfig(
    r=32, lora_alpha=64, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)

# 5. SFT 超参
sft_cfg = SFTConfig(
    output_dir="./qwen-agent-sft",
    num_train_epochs=2,
    learning_rate=1e-4,                  # LoRA 经验值
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    weight_decay=0.0,                    # LoRA 通常不加 wd
    bf16=True,
    gradient_checkpointing=True,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,      # effective bs = 32
    max_seq_length=8192,                 # tool 数据 system prompt 长，留余量
    packing=True,
    completion_only_loss=True,           # 自动 mask non-assistant
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",
    report_to="wandb",
    deepspeed="ds_zero2.json",           # 多卡时叠 ZeRO-2 / 单卡可去掉
)

# 6. Trainer
trainer = SFTTrainer(
    model=model, args=sft_cfg,
    train_dataset=ds.select(range(int(len(ds)*0.98))),
    eval_dataset=ds.select(range(int(len(ds)*0.98), len(ds))),
    peft_config=lora,
    processing_class=tok,
)
trainer.train(resume_from_checkpoint=True)  # 支持自动 resume
trainer.save_model()
```

要点：
- `completion_only_loss=True` + Qwen2.5 chat template 会自动按 `<|im_start|>assistant\n` 边界切 loss mask，handles tool_call 和 final answer 都算 loss、tool obs 不算（因为 obs 在 user role 内）
- `max_seq_length=8192` 给 tool definition 留足空间——5 个 tool 的 schema 拼起来轻松 1500-2500 token
- `packing=True` 在 agent 数据上提升尤其大，因为长度方差极大（短 chat 200 token vs 多轮 multi-tool trajectory 6000 token）
- 大数据 + 多卡训练时 `deepspeed="ds_zero2.json"` 配 ZeRO-2，70B 用 ZeRO-3 / FSDP

### 6.2 Tool calling 数据格式 unify（≤ 30 行）

```python
import json

def unify_to_messages(example):
    """把 xLAM / ToolACE / OpenAI / chat 数据全转成 Qwen2.5 chat messages。"""
    # Case 1: xLAM 风格 (query + tools + answers)
    if "tools" in example and "answers" in example:
        tools = json.loads(example["tools"]) if isinstance(example["tools"], str) else example["tools"]
        answers = json.loads(example["answers"]) if isinstance(example["answers"], str) else example["answers"]
        # tool_call XML 化（Qwen2.5 标准）
        tool_calls_str = "\n".join(
            f"<tool_call>\n{json.dumps(a, ensure_ascii=False)}\n</tool_call>" for a in answers
        )
        return {"messages": [
            {"role": "system", "content": "You are a helpful assistant.", "tools": tools},
            {"role": "user", "content": example["query"]},
            {"role": "assistant", "content": tool_calls_str},
        ]}
    # Case 2: 已经是 messages 格式（ToolACE / Hermes-FC / 通用 chat）
    if "messages" in example:
        msgs = example["messages"]
        # 把 OpenAI tool_calls 字段序列化成 Qwen XML
        for m in msgs:
            if m.get("role") == "assistant" and m.get("tool_calls"):
                xml = "\n".join(f"<tool_call>\n{json.dumps(tc['function'], ensure_ascii=False)}\n</tool_call>"
                                for tc in m["tool_calls"])
                m["content"] = (m.get("content") or "") + xml
                m.pop("tool_calls", None)
        return {"messages": msgs}
    # Case 3: NuminaMath / 单轮 chat (problem + solution)
    if "problem" in example:
        return {"messages": [
            {"role": "user", "content": example["problem"]},
            {"role": "assistant", "content": example["solution"]},
        ]}
    raise ValueError(f"Unknown schema: {example.keys()}")
```

要点：函数内判优先级——先 tool 数据、再 messages 已成型、最后兜底 reasoning。生产环境通常每个数据源**单独写一个 adapter** 而不是这种 if-else 链，这里压缩进 30 行展示统一思路。

### 6.3 手工构造 labels（含 tool_call mask，≤ 25 行）

如果用 `SFTTrainer` 的 `completion_only_loss=True`，labels 由 TRL 自动构造。下面这段是**底层等价实现**，便于理解 / 在自己的训练循环中复用：

```python
def make_agent_labels(messages, tokenizer):
    """构造 input_ids 与 labels：assistant 段（含 tool_call）算 loss，其余 -100。"""
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    full_ids = tokenizer(full_text, add_special_tokens=False).input_ids
    labels = [-100] * len(full_ids)

    # 找所有 <|im_start|>assistant\n ... <|im_end|> 段
    asst_header = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    im_end = tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]

    i = 0
    while i < len(full_ids) - len(asst_header):
        if full_ids[i:i+len(asst_header)] == asst_header:
            # 跳过 header 不算 loss，从 content 起算
            j = i + len(asst_header)
            while j < len(full_ids) and full_ids[j] != im_end:
                labels[j] = full_ids[j]
                j += 1
            if j < len(full_ids):       # 包含 <|im_end|>，教 model 学会停
                labels[j] = full_ids[j]
            i = j + 1
        else:
            i += 1
    return full_ids, labels
```

要点：
- 用 token 序列匹配 `<|im_start|>assistant\n` 找 assistant boundary（比 string 匹配更鲁棒，避免空格 / 编码问题）
- assistant header 自身 mask（与 8.2 §4.1 同样原则）
- `<|im_end|>` 必须算 loss（教 model 学会终止）
- tool_call 内容自然落在 assistant content 里 → 自动算 loss
- tool obs 因为在 `<|im_start|>user` 段里 → 自动 mask（前提是按 Qwen2.5 chat template，把 obs 渲染在 user role）

---

## 7. 评测

SFT 完了之后怎么知道训得好？必须在三个维度上分别 eval：

### 7.1 Chat 流畅性

- **MT-Bench** / **Arena-Hard**：LLM-as-judge 的开放问答评测，验证 model 是否还会"正常说话"。tool SFT 最常见的 regression 就是 chat 能力下降——MT-Bench score 比纯 chat SFT 的 baseline 低 0.3+ 分就说明 tool 数据淹没了 chat 数据，要降 tool 比例
- **AlpacaEval 2.0**：win rate 对比 GPT-4-turbo / Llama-3-70B，判断 chat 输出质量

### 7.2 Tool calling 准确性

- **BFCL (Berkeley Function-Calling Leaderboard)**：分 5 个 sub-track（simple / multiple / parallel / parallel_multiple / relevance），是 tool calling 评测的事实标准。看 overall accuracy + 每个 sub-track 单独分
- **API-Bank** / **ToolBench**：更早期的 tool eval，覆盖真实 RapidAPI 调用
- **Nexus Function Calling**：包含 nested call、long context

BFCL 评测两个核心指标：(1) **AST accuracy**——生成的 tool_call JSON 是否能解析成预期的 AST tree；(2) **Executable accuracy**——真去调用 API，结果是否符合预期。AST 比 Executable 宽松（API 端不稳定也算对）。

### 7.3 自定义 mixed eval

公开 benchmark 不能覆盖所有真实场景。强烈建议**自己写 50-100 条 mixed eval**，覆盖：

- 单轮 chat（"解释一下狭义相对论"）
- 多轮 chat（5 轮以上的连续对话）
- 单 tool call（"北京今天天气?"）
- 并行 tool call（"对比北京、上海、广州的天气"）
- Sequential tool call（"搜一下 Anthropic 最新论文，然后总结第一篇"）
- **Refuse-when-no-tool**（"你叫什么名字?" 不该调 search）
- **Hallucinated tool**（"用 tool xxx 帮我做 y" 但 tool xxx 不存在）

这套 eval 跑起来快，能在每个 SFT epoch 后立刻看到能力 trade-off。

---

## 8. Agent SFT 与 Agent RL 的衔接

### 8.1 SFT 是 Agent RL 的冷启

Module 15 会讲 Agent RL（多轮 PPO / GRPO 在真实环境训 agent），但**几乎所有成功的 agent RL 工作都先做 SFT**。原因：

- **格式约束**：agent RL 的 reward 通常是 trajectory 级别（任务成功率），但 model 必须先会输出**合法的 `<tool_call>` JSON**才能被环境接收。纯 RL from base model 探索空间太大，可能 100k step 都生成不出一个合法 call
- **基础能力**：tool calling 包含很多隐式知识（哪些 tool 适合哪类问题、参数怎么填），SFT 一次性灌进去
- **采样效率**：SFT 后的 model 已经能在 30-50% 的任务上成功，RL 只需在剩下的 50-70% 上做改进；纯 RL 起步成功率可能 < 5%，正样本太稀疏

这个范式被 Agent-FLAN（Chen 2024）、FireAct（Chen 2023）、ToRA（Gou 2023）等多篇 paper 验证。

### 8.2 SFT vs RL 的判断标准

一个常见问题：tool calling 准确率 80% 想提升，是加 SFT 数据还是上 RL？

**选 SFT 的情况**：
- 失败 case 集中在某种**未覆盖的 pattern**（如 model 不会处理 nested JSON 参数）→ 补这种 pattern 的训练数据更直接
- 数据预算充足、可以合成更多高质量 trajectory（用 GPT-4 / DeepSeek-V3 重新生成）
- tool 协议变化（如新加了 5 个 tool）→ 必须 SFT 教 model

**选 RL 的情况**：
- 失败 case 是**系统性的"探索-反馈"问题**（如 model 调用失败后不会重试 / 不会换 tool）→ trajectory-level reward 才能教
- 已经有可执行的 reward signal（task success / unit test pass）
- SFT 数据已经覆盖了所有 pattern 但 model 仍在边界 case 上抖动 → RL 能 fine-tune 决策边界
- 想做 multi-turn planning（不是单步 tool call）→ 必须 trajectory-level RL

实务里通常**先 SFT 到 70-80% → 再 RL fine-tune 到 85-90%+**——单纯堆 SFT 数据从 80% 推到 90% 边际收益急剧下降。这个交接点就是 Module 15 的入口。

---

## 9. 工程踩坑与经验

- ❗ **不同来源数据的 chat template 必须先 unify 再 train**——把 xLAM 的 OpenAI 风格直接和 Hermes 的 ChatML 风格混训会让 model 学糊：assistant turn 一会儿用 `<tool_call>` 一会儿用 OpenAI 的 JSON 字段，推理时 format 不稳定，下游解析直接挂。**铁律**：选定目标 model（如 Qwen2.5）后，所有数据都先转成它的 chat template 再 tokenize。
- ❗ **Tool special token 必须先 add_special_tokens + resize_embedding**——如果 base model（如 Qwen2-Base、Llama-3.1-Base）tokenizer 里没有 `<tool_call>` 等 token，直接训会让这些 token 被拆成 BPE 子片段（如 `<`、`tool`、`_call`、`>`），失去原子语义，model 学得极慢且 inference 时无法 byte-exact 匹配。先 add token + resize embedding（参考 3.2 §5.1）+ 用相似 token 均值 init 新 embedding 加速收敛。
- ❗ **Tool observation 段 loss 必须 mask**——这是 agent SFT #1 高频 bug。如果 obs 段算了 loss，model 学到的 pattern 是"先复读 obs 再答复"，推理时 user 看到 model 把 `<tool_response>{...}` 一字不差先吐出来再回答。Qwen2.5 设计里 obs 在 `user` role 内，自动 mask；但如果你自己写训练循环要特别注意。
- ❗ **通用 chat 与 tool 数据比例失衡**——tool > 50% 时 chat 能力急剧退化（MT-Bench 掉 0.5+ 分）；tool < 20% 时 BFCL 分数上不去。社区甜点是 **chat 60% + tool 30% + reasoning 10%**，这是 Tülu-3 / xLAM SFT recipe 的共识。具体可在自己的 mixed eval 上做小规模 ablation 找最优比例。
- ❗ **多 tool 并行调用的训练数据稀缺，需要 ToolACE/xLAM 风格合成**——开源 tool 数据集（如 ToolBench 早期版本）80%+ 是单 tool 单调用，model 学完不会 parallel call。如果目标场景需要并行（如电商 agent 同时查多个商品），必须显式合成 parallel call 样本，每个 assistant turn 内并列 2-5 个 `<tool_call>`，至少占总 tool 数据 15-20%。
- ❗ **Refuse-when-no-tool negative case 必须有**——纯 happy path 数据训出的 model 见到任何 query 都尝试调 tool。修复：保留 15-20% 的"system prompt 含 tool list 但 user query 不需要 tool / tool 解决不了"的样本，assistant 直接 chat 回答而不输出 `<tool_call>`。Hermes-Function-Calling-V1 数据集里就有专门的 `glaive_negative_samples` split。
- ❗ **SFT 后 generation 时 tool_call JSON 经常不合法**——SFT 的 cross-entropy loss 不能保证生成的 JSON 100% 合法（会有少量 hallucinated tool name、漏闭合括号、参数类型错）。BFCL 上 SFT 完的 model 通常 95%+ JSON 合法，剩下 5% 是真实 production 痛点。**解决：推理时配合 constrained decoding**（XGrammar / Outlines / lm-format-enforcer），强制按 tool schema 生成，合法率推到 100%（细节在 13.4 节）。
- ❗ **Tool observation 不能太长，否则 model 学复读**——如果训练数据里 tool obs 超过 2000 token（如 search 返回的整页 HTML），model 会被训成"obs 越长越想复读"。**解决：obs 在数据准备阶段先 summarize**（用 LLM 摘要到 200-500 token）或截断；推理时 framework 也要做 obs truncation，避免 context bloat。
- ❗ **多轮 history 长 → context 超 max_seq_length → 截断关键信息**——5 轮 multi-tool trajectory 容易超过 8k token（每轮 system + tool def + obs），训练时一截断把 final answer 砍掉，loss 学的是"半句话"。预处理时要按 max_seq_length 过滤（直接丢超长样本）或显式 truncate 历史 obs（保留最近 N 轮完整、远期 obs 用 `[truncated]` 占位）。
- ❗ **"幻觉调用不存在的 tool"——训练时 negative example 不够**——model 偶尔生成 `<tool_call>{"name": "some_tool_not_in_list", "arguments": {}}</tool_call>`。修复：训练数据中加 negative case——system 给 5 个 tool，user 问的事情这 5 个都解决不了，assistant 应输出"抱歉，我没有合适的工具来处理这个请求"而非乱调。这部分占 5-10%。
- ❗ **`completion_only_loss=True` 在 Qwen2.5 上要确认 chat_template 正确解析 assistant boundary**——TRL 内部的 boundary 切分依赖 tokenizer 的 `apply_chat_template` 输出 + 一段固定 marker。Qwen2.5 / Llama-3.1 / ChatGLM 的 marker 不同，旧版 TRL 在某些 chat_template 上会 silently 把 user 也算 loss，跑前务必 dump 一条 batch 的 labels print 出来肉眼确认（参考 8.2 §3.3 的 visual check 函数）。
- ❗ **LoRA rank 在 agent SFT 上要比纯 chat SFT 大**——纯 chat SFT 用 r=8/16 够，agent SFT 至少 r=32，甚至 r=64。原因：tool call 引入了"新输出格式"（`<tool_call>` XML + 严格 JSON），需要更大表达力；LoRA r 太小会让 JSON 错误率高 5-10×。本节代码用 r=32 是经验值。

---

## 10. 经典 paper

- **Liu et al., 2024 — *xLAM: A Family of Large Action Models to Empower AI Agent Systems*** — Salesforce 出品，agent SFT 数据合成 + 训练 recipe 最完整的开源工作。读 §3 数据合成 pipeline + §4 训练 setup + §5 BFCL 评测。Take-away：理解"如何用 LLM 合成 60k 高质量 tool call 数据"，xLAM-7B-fc-r 至今仍是 BFCL 上 7B 规模的 SOTA 之一
- **Liu et al., 2024 — *ToolACE: Winning the Points of LLM Function Calling*** — 多维度 tool 数据合成的范式作。读 §3 multi-agent 合成 framework + §4 evaluation。Take-away：理解"tool calling 不只是单 tool 单调用，必须显式覆盖 parallel / sequential / refuse 等正交维度"，ToolACE-8B 在 BFCL 上能打过 GPT-4
- **Patil et al., 2023 — *Gorilla: Large Language Model Connected with Massive APIs*** — 早期 tool LLM 的代表，提出 retriever-aware fine-tuning（让 model 学会用 retriever 提供的 tool subset 而非 hardcode）。读 §3 数据构造 + §4 评测。Take-away：理解 tool calling 训练的早期范式与"如何处理 1000+ tool 的可扩展性"
- **Qin et al., 2023 — *ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs*** — ToolBench 数据集与 ToolLLaMA 模型，覆盖 RapidAPI 16k+ 真实 API。读 §3 DFSDT (Depth-First Search-based Decision Tree) 数据合成。Take-away：真实 API 环境下的 tool 训练复杂度，是 14.3 节深入展开的基础
- **Yang et al., 2024 — *Qwen2.5 Technical Report*** — Qwen2.5 系列工程实践，§ "Tool Use" 描述了 chat template 标准化、tool special token 设计。Take-away：现代 chat model 的 tool 协议工程标准，本节代码基于这套
- 加分：**[BFCL Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html)** + 数据集文档——评测协议必读，理解每个 sub-track 测的是什么能力维度

---

## 11. 自测与面试题

**Q1（数据 pipeline）**：你要做一个能调 5 个 tool 的 agent SFT，已有 1M 条通用对话 + 100k 条 xLAM tool 数据（OpenAI function calling 格式）。列出从原始数据到能开训的完整 mix + format 步骤。

<details>
<summary>Answer sketch</summary>

完整 7 步：

1. **明确目标 model 与 chat template**：选定 Qwen2.5-7B（或 Llama-3.1-8B），后续所有数据都按它的 chat template 渲染
2. **数据 unify**：把 xLAM 的 OpenAI 风格转成 Qwen2.5 ChatML —— `tool_calls` 字段序列化成 `<tool_call>{...}</tool_call>` XML；通用对话的 messages 直接复用
3. **Tool definition 标准化**：xLAM 自带 tool schema，把 5 个目标 tool 的 JSON schema 整理成统一格式，注入 system prompt 的 `<tools>...</tools>` 段
4. **比例 sampling**：通用对话 600k + tool 100k + 加一部分 reasoning（如 NuminaMath 70k） + refuse-when-no-tool negative case 30k → 总 800k 条
5. **去重 + 质量过滤**：MinHash 去 paraphrase 重复（threshold 0.85）；用 LLM-as-judge 过滤低质量 tool call（如 args 错、tool name 拼错）
6. **覆盖维度检查**：tool 数据按 simple / parallel / sequential / refuse / hallucinated_tool 5 个维度统计占比，缺少的合成补足（用 GPT-4 生成 + schema 校验）
7. **构造 labels**：用 `tokenizer.apply_chat_template` + `completion_only_loss=True` 自动构造 labels；或手工实现（assistant content 含 tool_call 算 loss、user 含 tool obs 全 mask、`<|im_end|>` 算 loss）

加分要点：
- 提到 special token 注册（如果 base 没有 `<tool_call>` 要 add + resize）
- 提到 max_seq_length 设 8k+（因为 5 个 tool definition 容易把 system prompt 撑到 1500+ token）
- 提到训练前 dump 一条 batch print labels 肉眼校验（避免 chat template 解析错位）
- 提到 contamination check（不要把 BFCL eval 数据漏进训练集）

</details>

**Q2（loss mask）**：写出一段含 system / user / assistant-tool_call / tool_obs / assistant-final-answer 的 multi-turn 序列，逐 segment 标出哪些 token 算 loss、哪些 mask。

<details>
<summary>Answer sketch</summary>

```
段             token 内容                                              算 loss?
─────          ──────────────────────────────────────                  ─────────
[1] system     <|im_start|>system\n... <tools>{...}</tools><|im_end|>\n  ❌ -100
[2] user       <|im_start|>user\n北京天气?<|im_end|>\n                   ❌ -100
[3a] header    <|im_start|>assistant\n                                   ❌ -100
[3b] tool_call <tool_call>\n{"name":"get_weather",...}\n</tool_call>     ✅ 算
[3c] eot       <|im_end|>\n                                              ✅ 算
[4a] obs hdr   <|im_start|>user\n<tool_response>\n                       ❌ -100
[4b] obs body  {"temperature":"22°C","cond":"晴"}\n                      ❌ -100
[4c] obs end   </tool_response><|im_end|>\n                              ❌ -100
[5a] header    <|im_start|>assistant\n                                   ❌ -100
[5b] final     北京今天 22°C，天气晴朗。                                 ✅ 算
[5c] eot       <|im_end|>\n                                              ✅ 算
```

要点（写出 4 个即满分）：
- **system + user + tool obs 全 -100**（不是 model 生成的内容）
- **assistant header (`<|im_start|>assistant\n`) 必须 mask**——是 template 给定的"角色提示"，不算 model 生成
- **`<|im_end|>` 必须算 loss**——教 model 学会终止 turn，否则推理时停不下来
- **tool_call 与 final answer 都算 loss**——前者教 "如何调"，后者教 "如何消化 obs 答复"
- **all-turns 多轮**：第二轮 user / assistant 重复同样规则，不只对最后一个 turn 算
- 加分：能指出 Qwen2.5 把 tool obs 包在 `<|im_start|>user` 段里（不是 OpenAI 那种独立 tool role），所以"所有 user 全 mask" 自然把 obs 也 mask 掉；如果按 Llama-3.1 的 `ipython` role 要单独处理这一段

</details>

**Q3（衔接 Agent RL）**：你 SFT 完一个 agent model，BFCL 上 tool calling 准确率 80%，要进一步提升到 90%+。判断：用 RL 还是用更多 SFT 数据？给出判断标准与 trade-off。

<details>
<summary>Answer sketch</summary>

**先做错误归因**（不归因直接选 SFT 或 RL 都是猜）：从 BFCL 失败的 20% case 抽 100 条人工分析，分类失败原因——

**用更多 SFT 数据的情况**：
1. **失败集中在某种未覆盖的 pattern**：如 nested JSON 参数错、某些 sub-track（parallel call / sequential call）显著低于平均 → 补对应 pattern 的 SFT 数据更直接、ROI 高
2. **数据成本可承受**：用 GPT-4 / DeepSeek-V3 合成几万条针对性 trajectory + 跑 schema 校验，几百到几千美元就能拿到
3. **tool 协议有变化**：新增 tool / 改了 schema，必须 SFT 重新教 format
4. **base model 还有空间**：如果 SFT 数据从 100k 扩到 500k 还在涨分（log loss 还在降），先把 SFT scale 拉满再考虑 RL

**用 RL 的情况**：
1. **失败是系统性的"探索-反馈"问题**：如 tool call 失败后 model 不会重试 / 不会换 tool / 不会反问用户 → 这种"决策"问题 SFT 数据很难穷举，trajectory-level reward 才能教
2. **有可执行的 reward signal**：BFCL 的 executable accuracy 可以直接当 reward；或自己构造 sandbox 跑真 tool 拿成功率
3. **SFT 已经饱和**：数据从 500k 加到 1M 不再涨分，loss 早早平台 → 容量瓶颈不在数据量
4. **想做 multi-turn planning 而不是单步 tool call**：trajectory-level 优化才有意义

**Trade-off 边界**：
- RL 比 SFT 复杂 5-10 倍工程量（rollout infra / reward design / KL 约束 / reward hacking 防御），如果 SFT 还能涨分就别轻易上 RL
- RL 容易 reward hacking——只用 schema 合法度做 reward，model 学会输出"格式合法但语义错"的 tool call；reward 设计要 careful（参考 9.6）
- 80% → 90% 是常见的 SFT-to-RL 交接点：再加 SFT 数据边际收益急剧下降，RL fine-tune 几千 step 通常能再涨 5-10 个点
- 实务最优解：**先 SFT 到 80% 再用 RL fine-tune 5-10 个点**——这就是 Agent-FLAN / FireAct / xLAM-fc-r 的标准范式

加分：能指出"判断是否 SFT 饱和"的具体信号——eval loss 平台、不同 lr 下 final score 一致、增加数据带来的提升 < 0.5%/10x data；能提到 RL 阶段仍要混 SFT data 做 reference KL（防止漂移）

</details>

---

## 12. 延伸阅读

- [xLAM GitHub & paper](https://github.com/SalesforceAIResearch/xLAM) — Salesforce 完整 agent SFT 数据生成 + 训练 + eval pipeline，可直接复用做中文版本
- [ToolACE GitHub](https://github.com/Team-ACE/ToolACE) — ToolACE 数据集 + multi-agent 数据合成 framework，覆盖 parallel / sequential / refuse 等正交维度
- [BFCL Leaderboard 与代码](https://gorilla.cs.berkeley.edu/leaderboard.html) — Berkeley Function-Calling Leaderboard，tool calling 评测事实标准；下载 eval set 在自己 SFT 后跑一遍
- [Hermes Function Calling V1 数据集](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1) — 已经按 ChatML + `<tool_call>` 格式准备好的开源 tool 数据，最少改动直接训
- [Qwen Agent 文档](https://qwen.readthedocs.io/en/latest/framework/qwen_agent.html) — Qwen 官方 agent framework，看 production 级 tool 协议设计
- [HuggingFace TRL — SFTTrainer with chat template](https://huggingface.co/docs/trl/sft_trainer#train-on-completions-only) — `completion_only_loss=True` 与 chat template 集成的官方文档
- [XGrammar / Outlines](https://github.com/mlc-ai/xgrammar) — 推理时 constrained decoding，强制按 tool JSON schema 生成，把 SFT 后 5% 的 JSON 合法率 gap 补到 100%
- 推荐继续读本教程的 **Module 9 RLHF**（DPO / PPO / GRPO）——SFT 完了之后怎么用 preference / reward 进一步优化；**14.3 Tool Use 训练**——把本节范式扩展到 retriever-aware tool selection；**15.1 Agent SFT** + **15.2 多轮 PPO/GRPO**——本节是 agent RL 的冷启动基础，Module 15 把 trajectory-level reward 与归因详细展开
