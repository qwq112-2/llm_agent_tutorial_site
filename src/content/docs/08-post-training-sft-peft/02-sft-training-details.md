---
title: "8.2 SFT 训练细节：chat template / loss mask / sample packing"
description: "SFT 表面上只是\"再跑一次 CLM\"，但真正决定模型好坏的是三件事——chat template 必须与 base model 严格对齐、loss mask 让梯度只走 assistant token、sample packing 把训练吞吐推到 padding 的 1.5-3 倍——这三件事任意一项错位，训出来的模型要么停不下来、要么复读用户输入、要么训练 loss 看着漂亮其实在学错信号。"
---

> ⏱ 预计阅读 35 分钟 ｜ 难度 ★★ 🔥 ｜ 前置：3.2（chat template 基础）、3.3（packing / mask 基础）、6.1（CLM loss）

## 一句话本节讲什么

SFT 表面上只是"再跑一次 CLM"，但真正决定模型好坏的是三件事——**chat template 必须与 base model 严格对齐、loss mask 让梯度只走 assistant token、sample packing 把训练吞吐推到 padding 的 1.5-3 倍**——这三件事任意一项错位，训出来的模型要么停不下来、要么复读用户输入、要么训练 loss 看着漂亮其实在学错信号。

---

## 1. Mental model（直觉）

预训练（Module 6）阶段模型在做的事简单粗暴——给它一段从互联网爬来的纯文本，让它**对每个位置都预测下一个 token**。所有 token 都贡献 loss，信号密度 100%，infra 简洁到极致。

SFT（Supervised Fine-Tuning）的任务变了。手里的不再是连续文本，而是**一条条 (instruction, response) 的对话样本**——目标也不再是"什么文本都能续写"，而是"看到 user 的请求，按 assistant 的角色回答"。但训练目标在数学上**仍然是 CLM**——next-token prediction 没变，变的只是三件事：

1. **数据格式**：从纯文本变成结构化的 `[system, user, assistant, user, assistant, ...]` 多轮序列；要把这些 role 拼成一条 token 序列，靠的就是 **chat template**
2. **loss 的归属**：不是所有位置都该算 loss——`system` 是先验设定、`user` 是问题输入，模型不该学着复述它们；只有 `assistant` 段才是模型真正要"学说"的内容；这就是 **loss mask** 要做的事
3. **吞吐工程**：SFT 数据长度方差极大（一条 200 token、一条 4000 token），简单 padding 浪费严重；**sample packing** 把多条拼一条 + block-diagonal mask 是现代 SFT 的工程标配

把这三件事画一张草图：

```
原始数据（多轮对话）
┌─────────────────────────────────────────────────┐
│ system: You are helpful.                        │
│ user:   写个 quicksort                          │
│ assistant: def qs(a): ...                       │
│ user:   再用 Rust                               │
│ assistant: fn qs(a) { ... }                     │
└─────────────────────────────────────────────────┘
              ↓ apply_chat_template
[<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...<|im_end|>\n...]
              ↓ make_labels (loss mask)
input_ids: [ s s s s | u u u u | a a a a | u u | a a a ]   ← 都进 attention
labels   : [-100 ... | -100 ...| a a a a | -100 | a a a ]   ← 只有 assistant 算 loss
              ↓ packing (多条样本拼一条)
packed   : [ sample_1 || sample_2 || sample_3 ]   ← 用 cu_seqlens / block-diag mask 隔开
              ↓ SFTTrainer + flash_attn_2 + bf16
              ↓ 1-3 epoch、cosine lr 5e-6 ~ 5e-5
```

记牢这条管线，剩下所有内容就是把每个箭头里的细节写清楚——以及为什么每一步都有"新手必踩"的坑。

---

## 2. SFT 训练目标的精确定义

### 2.1 仍是 CLM，但带 loss mask

记一条 SFT 样本 tokenize 后是 $\mathbf{x} = (x_1, \dots, x_T)$，每个位置有一个二值 mask $m_t \in \{0, 1\}$（1 = 该位置是 assistant token、属于 loss、参与梯度；0 = 是 system/user/template token、不算 loss）。SFT 的 loss 就是**带 mask 的 CLM**：

$$\mathcal{L}_{\text{SFT}}(\theta) = -\frac{1}{\sum_t m_{t+1}} \sum_{t=1}^{T-1} m_{t+1} \cdot \log P_\theta(x_{t+1} \mid x_{\le t})$$

注意两点：

- **mask 用的是 $m_{t+1}$ 而不是 $m_t$**——因为我们预测的是位置 $t+1$ 的 token，所以要按"被预测位置"是否参与 loss 来加权
- **归一化**用的是有效位置数 $\sum_t m_{t+1}$，不是 $T$——这样不同长度的样本贡献是公平的（不会因为 prompt 长 model 就被惩罚）

工程上**不显式构造 $m$**，而是把不参与 loss 的 label 设成 `-100`：

```python
labels = input_ids.clone()
labels[~assistant_mask] = -100
loss = F.cross_entropy(logits.view(-1, V), labels.view(-1), ignore_index=-100)
```

`ignore_index=-100` 是 PyTorch CE 的默认值，框架自动跳过这些位置。HuggingFace `LlamaForCausalLM`、TRL `SFTTrainer` 内部全是这套约定。

### 2.2 attention 上 user 仍要可见

非常容易混淆——**loss mask 与 attention mask 是两件独立的事**：

- **attention mask**：决定 forward 时位置 $i$ 能 attend 到哪些位置。SFT 里 assistant 段当然要能看到前面的 user 提问，否则没法回答；attention 还是标准的 causal + 可能叠 packing block-diagonal
- **loss mask**：决定 backward 时哪些位置贡献梯度。SFT 里只有 assistant 算 loss，user / system 都设 `-100`

两套 mask 形状也不同：attention mask 是 $(B, T)$ 或 $(B, 1, T, T)$，loss mask 嵌在 $(B, T)$ 的 labels 里（用 `-100` 标记）。3.3 节的"attention mask vs loss mask"那张表如果还印象模糊，回去再看一遍——这是 SFT 工程的根基。

### 2.3 多轮对话的 loss schema

一条多轮对话 `[system, user1, asst1, user2, asst2, user3, asst3]` 上算 loss 有两种主流写法：

- **All-turns**：每个 assistant 段（asst1 / asst2 / asst3）都算 loss，user / system 全 mask。一条样本同时训练"首轮回答"+ "看到自己上一轮回答 + 新 user 后接着回答"+ ...，**数据利用率最高**
- **Last-turn-only**：只对最后一个 assistant 段（asst3）算 loss，前面 asst1 / asst2 都 mask 掉当成 prompt 的一部分。实现简单（一刀切），但同一条样本只学到一次 assistant 输出，**浪费严重**

```
样本：[sys][user1][asst1][user2][asst2][user3][asst3]

All-turns      labels: [-100 -100   asst1   -100   asst2   -100   asst3 ]
Last-turn-only labels: [-100 -100   -100    -100   -100    -100   asst3 ]
```

**现代主流是 all-turns**——一来数据利用率高 2-5×（多轮 dialog 通常 3-5 轮），二来模型显式学会"看着自己上一轮的回复继续接话"，对多轮 handoff 能力关键。Llama-2 / Llama-3、Tülu、OpenHermes 系列 SFT 数据全部 all-turns。LIMA / 早期 Vicuna 用过 last-turn-only，但已被淘汰。

唯一例外：**强 reasoning 数据**（如 long-CoT 训练），有时只对最终答案算 loss、把推理过程当 prompt——这是另一回事，归在 Module 10 讲。

---

## 3. Chat Template：模型理解对话结构的协议

### 3.1 三种主流风格速览

3.2 已经介绍过 special token 体系，这里把"训练角度"再梳理一次。三种现代主流 template，记住它们的 boundary token：

**Llama-3 风格**（header + eot 两件套）：

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are helpful.<|eot_id|><|start_header_id|>user<|end_header_id|>

What is 2+2?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

4<|eot_id|>
```

关键：`<|eot_id|>`（end-of-turn）是每一轮的结束标记，`<|end_of_text|>`（end-of-sequence）才是整段文本的结束——**两个不同 token，SFT 时要让 assistant 学会输出 `<|eot_id|>` 来终止当前 turn**，否则推理时模型会一路吐到 `max_new_tokens`。

**Qwen / ChatML 风格**（紧凑、被广泛复用）：

```
<|im_start|>system
You are helpful.<|im_end|>
<|im_start|>user
What is 2+2?<|im_end|>
<|im_start|>assistant
4<|im_end|>
```

`<|im_start|>` + role 字符串作为开头、`<|im_end|>` 作为结尾。Yi、InternLM、零一万物等大量国产开源模型都借用这套。

**Mistral 风格**（极简，沿用 Llama-2 早期 `[INST]`）：

```
<s>[INST] You are helpful.

What is 2+2? [/INST] 4</s>
```

`[INST]` / `[/INST]` 把 user 内容包起来，assistant 部分裸写不加标记。Mistral-7B-Instruct、Mixtral 用这套；缺点是 system prompt 没有专门标记（要嵌在 `[INST]` 内），不太方便表达复杂角色。

### 3.2 `apply_chat_template` 实战

HuggingFace 的现代标准 API 是 `tokenizer.apply_chat_template(messages, ...)`——它读 tokenizer 自带的 jinja 模板（藏在 `tokenizer_config.json` 的 `chat_template` 字段），按规则把 `messages` 拼成字符串或 token ids。**永远不要自己手拼字符串**——template 字段在版本之间会更新（如 Qwen2 → Qwen2.5 微调过分隔符），手拼版本一旦与 tokenizer 不一致就是潜伏 bug。

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

messages = [
    {"role": "system", "content": "You are a helpful coding assistant."},
    {"role": "user", "content": "用 Python 写 quicksort"},
    {"role": "assistant", "content": "def quicksort(a): ..."},
    {"role": "user", "content": "再用 Rust 写一遍"},
]

# === 训练阶段 ===
# add_generation_prompt=False：messages 已包含完整 assistant 段，直接 tokenize
train_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=False,
    return_tensors="pt",
)

# === 推理阶段 ===
# add_generation_prompt=True：messages 末尾 user 还没回复，
# 模板会追加 "<|im_start|>assistant\n" 引导模型续写
infer_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
)
```

**训练 vs 推理的差异**就在 `add_generation_prompt` 上——这是 SFT 与 inference 对齐时最常见的对齐错误来源。训练时该参数应为 `False`（assistant 内容已经在 messages 里），推理时该为 `True`（让模型从 assistant header 之后开始生成）。

### 3.3 Chat template 对错的肉眼校验

调试 SFT 数据时强烈建议做一次**人肉对照打印**——dump 出 `(input_ids, labels, decoded_text)` 三栏，确认每个 turn 的 boundary 都对齐：

```python
# 调试 helper：让你看清每个 token 是不是该有 loss
for tid, lid in zip(input_ids[0].tolist(), labels[0].tolist()):
    tok = tokenizer.decode([tid])
    has_loss = "✓" if lid != -100 else " "
    print(f"  {has_loss}  id={tid:>6}  text={tok!r}")
```

这种 visual check 能立刻发现"system prompt 末尾被打了 loss 标记"、"`<|eot_id|>` 被错 mask 成 -100"等典型 bug——SFT 启动训练前花 5 分钟跑一遍对照，能省下 5 小时的训完后才发现 ckpt 全废的痛苦。

---

## 4. Loss Mask 的工程实现

### 4.1 手写 mask：从 chat template 找 assistant 边界

最直接的方式是在 token 序列里**找 assistant header 与 eot 之间的位置**：

```python
def make_sft_labels_qwen(messages, tokenizer):
    """
    Qwen / ChatML 模板下，构造 input_ids 与 labels（assistant 段外全 -100）。
    返回 (input_ids: List[int], labels: List[int])
    """
    input_ids, labels = [], []
    # 这两个 token 是 ChatML 的角色 boundary
    im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end   = tokenizer.convert_tokens_to_ids("<|im_end|>")

    for msg in messages:
        # 拼当前 turn 的 token：<|im_start|>{role}\n{content}<|im_end|>\n
        prefix = tokenizer.encode(f"<|im_start|>{msg['role']}\n", add_special_tokens=False)
        body   = tokenizer.encode(msg["content"], add_special_tokens=False)
        suffix = tokenizer.encode(f"<|im_end|>\n", add_special_tokens=False)
        turn_ids = prefix + body + suffix

        if msg["role"] == "assistant":
            # assistant：prefix(role header) 不算 loss，body + <|im_end|> 算
            #   ↑ 一定要包含 <|im_end|>，让模型学会"什么时候停"
            turn_labels = [-100] * len(prefix) + body + suffix
        else:
            # system / user：全 mask
            turn_labels = [-100] * len(turn_ids)

        input_ids.extend(turn_ids)
        labels.extend(turn_labels)
    return input_ids, labels
```

要点：

- **assistant header `<|im_start|>assistant\n` 自身不算 loss**——这部分是 template 给定的"角色提示"，不是模型生成的内容；只有从 content 开始才该有梯度
- **`<|im_end|>` 必须算 loss**——这是模型学"如何终止当前 turn"的唯一信号；漏了它，推理时模型不会主动停
- **多轮**：循环里每个 assistant turn 都按上面规则处理，自然就是 all-turns schema

### 4.2 用 TRL `DataCollatorForCompletionOnlyLM`（最常用快捷方式）

如果 SFT 数据可以表达成 `prompt + response` 形式（典型 Alpaca / Magpie 风格），TRL 提供了开箱即用的 collator：

```python
from trl import DataCollatorForCompletionOnlyLM
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# 关键：response_template 是模型回复**前**的 marker 字符串
# tokenize 后 collator 会找这个序列，之前的 token 全 mask 成 -100
response_template = "<|im_start|>assistant\n"

collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template,
    tokenizer=tokenizer,
    mlm=False,     # 这是 CLM 不是 MLM
)
# 配合 SFTTrainer 即可：trainer = SFTTrainer(..., data_collator=collator)
```

注意 `response_template` 要传**字符串而不是 token ids**——HuggingFace 内部会按字符串 tokenize 并查找；如果字符串编码出来的 token 序列在样本里找不到（如 leading space 或 BOS 不一致），collator 会无声 fallback 到"全部都算 loss"，结果模型把 user 部分都背下来。**务必跑一条样本 print 出 labels 验证**。

### 4.3 多轮场景：`SFTTrainer` 的 `chat_template` 参数

TRL 0.10+ 直接支持把整条 messages 喂进去：

```python
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model="Qwen/Qwen2.5-7B",
    train_dataset=ds,                          # ds 每条形如 {"messages": [...]}
    args=SFTConfig(
        output_dir="./out",
        max_seq_length=4096,
        packing=True,
        completion_only_loss=True,             # 自动只在 assistant 段算 loss
    ),
)
```

`completion_only_loss=True` 让 TRL 自动按 chat template 解析 assistant boundary 并 mask；前提是 dataset 里给的是 `messages` 列且 tokenizer 有合法的 `chat_template`。这是目前最省事的多轮 SFT 写法。

---

## 5. Sample Packing：把吞吐推到极限

### 5.1 为什么 SFT 必须 packing

3.3 已经讲过 packing 的核心思想——多条短样本拼一条长序列、配 block-diagonal mask 隔开、用 FlashAttention 的 `cu_seqlens` 接口避免显式构造 mask 矩阵。**SFT 比 pretrain 更需要 packing**，因为 SFT 数据的长度方差大得离谱：

- 一条简单指令样本：200 token
- 一条多轮代码生成样本：4000 token
- 一条 long-context summarization：32000 token

简单 padding 到 batch 内 max_len，padding ratio 常常 50-70%，算力浪费一大半。开了 packing 后浪费率降到 < 5%，**吞吐直接 +60%-200%**。

### 5.2 一行打开：`SFTConfig(packing=True)`

```python
from trl import SFTConfig
config = SFTConfig(
    output_dir="./out",
    packing=True,
    max_seq_length=4096,
    # 现代 TRL 配合 FA2，自动处理 cu_seqlens / position_ids
    # 老版本需额外传 dataset_kwargs={"add_special_tokens": False}
)
```

TRL 内部用 `ConstantLengthDataset` 把 dataset 里的样本流式拼接到 `max_seq_length`，再配 FlashAttention varlen kernel。开启前需要：

- 模型加载时设 `attn_implementation="flash_attention_2"`
- 安装 `flash-attn`（GPU 必须是 Ampere/Hopper 架构）
- TRL ≥ 0.9（更早版本的 packing 实现有 position_ids bug）

### 5.3 Packing 的隐藏要求：`position_ids` 必须每段重置

这是 packing 与 RoPE 配合的**最关键工程细节**——一条 packed 序列里如果三个原样本拼接，长度 [4, 6, 3]，**三段的 position_ids 必须各自从 0 开始**，不是全局 0..12 递增：

```
正确 (per-sample reset):
  position_ids = [0, 1, 2, 3 | 0, 1, 2, 3, 4, 5 | 0, 1, 2]

错误 (global increment):
  position_ids = [0, 1, 2, 3 | 4, 5, 6, 7, 8, 9 | 10, 11, 12]
```

为什么必须 reset？RoPE 在每个位置注入旋转角度 $\theta_t = t \cdot \omega$，第二个样本如果用全局 position 4-9，相当于让一段独立的对话从 RoPE 的"序列第 4 位"开始——模型在 inference 时见到的永远是从 0 开始的样本，训练 / 推理分布不一致，loss 会偷偷劣化。

TRL 4.40+ 已修复了这个 bug——`packing=True` 时自动重置 position_ids。如果你用更早的版本（或者在 axolotl / LLaMA-Factory 等其他框架里），**务必检查框架是否做了 per-sample position_ids reset**：

```python
# 验证方式：dump 一条 packed batch
batch = next(iter(trainer.get_train_dataloader()))
print("position_ids[0]:", batch["position_ids"][0][:30])
# 看是不是 [0,1,2,...,k1, 0,1,...,k2, 0,1,...]
# 如果是 [0,1,2,3,4,5,...,T-1] 全局递增，框架就没 reset，要手工补
```

---

## 6. 完整 SFT 训练 script（HF TRL + LoRA + packing + bf16）

以下是一个能跑通的最小 SFT 脚本，covers chat template / loss mask / packing / LoRA / bf16 一站式（LoRA 细节在 8.3 详讲，这里只展示集成）：

```python
# sft_train.py
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

MODEL_NAME = "Qwen/Qwen2.5-7B"

# 1. tokenizer + 修补 pad token（Qwen 默认 pad_token 是 None）
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # SFT 常见做法

# 2. base model + bf16 + FlashAttention 2
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

# 3. SFT 数据：每条 {"messages": [{"role": ..., "content": ...}, ...]}
ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:5000]")

# 4. LoRA config（细节在 8.3）
lora = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
)

# 5. SFT 超参（"业界默认值"档）
sft_config = SFTConfig(
    output_dir="./qwen-sft-lora",
    num_train_epochs=2,                    # SFT 不要超 3 epoch
    learning_rate=2e-5,                    # 5e-6 ~ 5e-5 区间
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,                      # SFT warmup 比预训长比例
    weight_decay=0.01,
    bf16=True,
    gradient_checkpointing=True,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,         # effective bs = 32
    max_seq_length=4096,
    packing=True,                          # 关键：开 packing，吞吐 +60%~+200%
    completion_only_loss=True,             # 自动只在 assistant 段算 loss
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",
)

# 6. Trainer + train
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=ds,
    peft_config=lora,
    processing_class=tokenizer,
)
trainer.train()
trainer.save_model()
```

整个 script 不到 60 行，但 chat template、loss mask、packing、LoRA、bf16、cosine warmup 全部就位——这就是现代 SFT 的标准模板。生产环境会再叠 DeepSpeed / FSDP（Module 7 讲）、eval 数据集、wandb 日志等。

---

## 7. SFT 超参经验值（必背）

不同模型 size / 数据规模 / 任务类型差异不大，下面这套是**业界大多数公开 SFT 工作的默认起点**：

| 超参 | 经验值 | 备注 |
|---|---|---|
| **epochs** | 1-3 | SFT 几乎不会超过 3，过拟合风险高 |
| **learning rate** | 5e-6 ~ 5e-5 | 比预训小 10-100×；7B+ 模型偏 1e-5；< 7B 可到 5e-5 |
| **LR schedule** | cosine to 0 | 也有 linear，cosine 略稳 |
| **warmup ratio** | 0.05 ~ 0.10 | 比预训长比例（预训通常 0.01-0.03） |
| **batch (token)** | 100k ~ 1M / step | 远小于预训（M ~ B 级） |
| **weight decay** | 0.01 ~ 0.1 | LoRA 通常 0；full SFT 0.01-0.05 |
| **max_seq_length** | 4k ~ 32k | 长 context 任务（reasoning / multi-turn agent）拉到 32k |
| **dropout** | 0 | 现代 LLM SFT 一般不加 dropout |
| **gradient clip** | 1.0 | 标准值 |
| **optimizer** | AdamW $\beta=(0.9, 0.95)$ | 与预训一致；LoRA 可用 8bit AdamW |

**为什么 SFT 的 lr 必须比预训小 10-100×**——预训阶段模型从随机初始化出发，需要大 lr 快速建立结构；SFT 时模型权重已是高度 informative 的预训表示，大 lr 会"撞翻"这套表示，导致**灾难性遗忘**（catastrophic forgetting）：模型忘掉预训学的通用知识，只记得 SFT 这几万条样本的表面 pattern。一个公认 mode 是用预训末尾 lr 的 1/10 ~ 1/100 作为 SFT 起点。

**为什么 epoch 不要超 3**——SFT 数据集相比预训小 4-6 个量级（百万 vs 万亿 token），模型容量过剩，1-2 epoch 就开始 overfit 训练分布。Llama-2 paper §3.3 明确写"2 epoch is the sweet spot"；过 3 epoch 后 eval loss 上升、生成多样性塌陷。

---

## 8. 训练监控：什么时候该早停

SFT 训练 loss 会一路下降，看着很美——但**真正该看的是 eval loss 与生成质量**。三个核心 monitor：

1. **Eval loss 曲线**：在 holdout SFT 数据上算 loss，2-3 epoch 内通常会平台或上升；上升后立刻 stop，再训只是过拟合
2. **KL(SFT model || base model)**：若发散过快说明模型"漂移"过大，可能损害预训通用能力；超过 threshold（凭经验 0.5-1.0 nat / token）建议降 lr 或停训
3. **小样本生成抽查**：每个 epoch 末用 30-50 条 prompt 让模型生成，**人肉看输出格式、长度、风格**——loss 下降不代表生成更好，有时 loss 降 0.05 但模型开始系统性"复读 user 输入"或"输出过短"，这种问题 loss 看不出，必须看生成

Tülu 3 paper § 4.2 给了一套很具体的"check during SFT"列表，强烈建议参考。

---

## 9. SFT vs Continued Pretraining

最后一个常被混淆的点——SFT 与 continued pretraining 都是"在已有 LLM 上接着训"，但**训练目标和数据形式完全不同**：

| 维度 | SFT | Continued Pretraining |
|---|---|---|
| **数据** | (instruction, response) 对，结构化 | 纯文本，无结构 |
| **loss** | 只在 response 段（带 mask） | 全 token 算（标准 CLM） |
| **目标** | 教模型"按对话格式回答" | 让模型学新领域知识 |
| **数据规模** | 1k ~ 10M 样本 | 1B ~ 100B token |
| **lr** | 5e-6 ~ 5e-5 | 1e-5 ~ 5e-5 |
| **常见场景** | 对齐风格、agent 工具使用 | 中文/医学/法律领域适配 |

**实战经验**：做 domain adaptation（如 medical / legal LLM），**continued pretraining + SFT 两阶段**通常比单纯 SFT 效果好——continued pretraining 让模型先吸收 domain 词汇与知识分布，SFT 再叠"按问答格式说话"的能力。这套做法在 BloombergGPT、MedPaLM、ChatLaw 等垂域模型上反复验证。

---

## 10. 工程踩坑与经验

- ❗ **chat template 错位是 SFT #1 bug**——用错 template（如把 base model 用 Llama-3 template 训，但 tokenizer 实际是 Qwen ChatML）会让模型"看不懂自己的输入格式"，loss 看似正常下降但推理输出全是乱码或停不下来。**铁律**：SFT 数据构造必须用 `tokenizer.apply_chat_template`（与目标模型自带的 jinja 模板一致），不要手拼字符串。
- ❗ **`labels` 没 mask user / system 段** → 模型学着"复读 user 提问再回答"，eval 时一上来先把用户问题原样输出一遍。修复：构造 labels 时按 chat template 的 role boundary 精确切分，user/system/template token 全置 `-100`，只有 assistant content + `<|eot_id|>` / `<|im_end|>` 算 loss。
- ❗ **Sample packing 不加 block-diagonal mask = 跨 sample 信息泄露**。两条独立样本 A、B 拼起来后，A 的 token 会 attend 到 B 的 token，等价于让模型偷看了"另一段对话"。轻则 eval 抖动，重则训出来的模型在长 prompt 下复读训练数据其它 sample 的内容。任何 packing 实现都要写 unit test：两个样本 [A, B] 拼起来 forward vs 各自单独 forward，logits 必须 bit-exactly 相同。
- ❗ **packing 时 `position_ids` 没在每个 sample 内重置**——每段必须从 0 开始，否则 RoPE 位置编码偏移、训练分布与推理分布不一致。TRL 4.40+ 默认修复，但其他框架（早期 axolotl / 自写 collator）要手工检查。dump 一条 batch 看 `position_ids[0]` 是不是 `[0,1,2,...,k1, 0,1,...,k2, ...]` 的分段形式。
- ❗ **SFT epoch > 3 几乎必过拟合**。Llama-2 / Tülu / OpenHermes 都是 2 epoch；超过 3 后 eval loss 上升、生成多样性塌陷。监控 eval loss 拐点早停，不要凭"训练 loss 还在降"硬撑。
- ❗ **SFT lr 不能用预训的 3e-4**——会让模型权重在 SFT 数据上被"冲掉"，灾难性遗忘预训能力；用 5e-6 ~ 5e-5 区间，越大模型用越小的 lr（70B 用 5e-6 ~ 1e-5，7B 用 1e-5 ~ 5e-5）。
- ❗ **Llama-3 的 `<|eot_id|>` 与 `<|end_of_text|>` 是两个不同 token**。SFT 要让 assistant 学会**输出 `<|eot_id|>` 终止当前 turn**；如果训练时漏掉这个 token 的 label（被错置 -100），推理时模型永远不会主动停，只能靠 `max_new_tokens` 强切。生成时 `stop_token_ids` 也要包含 `<|eot_id|>`（很多人只设 `<|end_of_text|>`，结果模型在多轮里把所有 turn 连成一段连续吐）。
- ❗ **multi-turn 数据如果某个 assistant 段被 truncate**（超过 max_seq_length），loss 会对截断处算梯度——训出 model "话说一半"突然停。解决：dataloader 阶段过滤掉超长样本，或者把 truncate 的最后一个 assistant turn 整段 mask 掉、只对完整 turn 算 loss。
- ❗ **`tokenizer.pad_token` 默认 None**（GPT / Qwen / Llama 系），常见做法 `tokenizer.pad_token = tokenizer.eos_token`——但 packing 模式下要小心：如果 pad token = eos token 又没正确 mask，模型会被训成"在 packed 序列结尾连续输出 eos"，推理时一开口就吐 eos 直接停。最稳的做法是用 `<|finetune_right_pad_id|>`（Llama-3.1+ 自带）或加一个独立的 `<|pad|>` 并 resize embedding。
- ❗ **`DataCollatorForCompletionOnlyLM` 的 `response_template` tokenize 不一致会无声降级**——传入字符串 `"<|im_start|>assistant\n"`，collator 内部 tokenize 后在样本里 byte-by-byte 查找；如果样本里实际编码是 `"<|im_start|>assistant"` + 空格 + `"\n"`（多/少一个 whitespace），找不到就 fallback 到"全 token 算 loss"，**没有任何报错**。务必跑一条 batch 把 `labels` 打出来看 -100 是否在 user 部分。

---

## 11. 经典 paper

- **Wei et al., 2022 — Finetuned Language Models Are Zero-Shot Learners (FLAN)** — instruction tuning / SFT 范式的奠基作。Google 把"用 instruction-formatted 数据 SFT 一个预训模型"系统化成方法论，证明 SFT 后 zero-shot 跨任务泛化能力大幅提升。读 §3 数据格式与 §4 实验，理解 SFT 为什么从 NLP trick 变成 LLM 标配。
- **Touvron et al., 2023 — Llama 2** — **SFT 工程细节最详细的公开技术报告**。读 §3.1-3.3 涵盖 chat template 设计（`[INST]` / `[/INST]`）、loss mask、超参（epoch=2、lr=2e-5、cosine、warmup=3%）、多轮训练 schema。如果只读一篇 SFT paper，选这篇。
- **Lambert et al., 2024 — Tülu 3** — 现代 SFT pipeline 的系统化总结。Allen AI 把 SFT 数据混合策略、超参 sweep、eval 流程全部公开，配套 200K+ 高质量 SFT 样本和完整 recipe。读 §4 "SFT" 部分理解 2024+ 的 SFT 最佳实践（all-turns、completion-only loss、packing、cosine warmup 0.1）。
- **Bavarian et al., 2022 — Efficient Training of Language Models to Fill in the Middle (FIM)** — 6.1 已经讲过 FIM 是预训目标，但它对 SFT 有启发：**改数据顺序、不改 loss / 架构**这条思路在多轮 SFT、Tool Use SFT 里反复出现。读 §3 理解"用 special token 标记数据结构"的工程哲学。
- **HuggingFace TRL — `SFTTrainer` 文档** — 必看的实操 reference，`packing` / `completion_only_loss` / `chat_template` 三个开关背后的实现细节都在源码里。

---

## 12. 自测与面试题

**Q1（实战）**：写出 SFT 训练时构造 labels 的核心 5 行（含 `-100` mask 逻辑），数据形式假设是 ChatML 多轮对话。

<details>
<summary>Answer sketch</summary>

核心 5 行（不含函数定义）：

```python
labels = []
for msg in messages:
    turn_ids = tokenizer.encode(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n", add_special_tokens=False)
    body_start = len(tokenizer.encode(f"<|im_start|>{msg['role']}\n", add_special_tokens=False))  # role header 长度
    labels.extend(turn_ids if msg["role"] == "assistant" else [-100] * len(turn_ids))
    if msg["role"] == "assistant":
        labels[-len(turn_ids):body_start - len(turn_ids)] = [-100] * body_start  # role header 那段也 mask 掉
```

要点（写出 3 个即满分）：

- **only assistant turn 算 loss**：user / system 整段 `-100`
- **assistant 内部，role header 部分（`<|im_start|>assistant\n`）也要 `-100`**——只有 content + `<|im_end|>` 算 loss，因为 header 是 template 给定的"角色提示"
- **`<|im_end|>` 必须算 loss**——这是模型学"何时终止"的唯一信号；漏了它推理时停不下来
- **多轮 all-turns**：循环里每个 assistant turn 都按规则处理，不只对最后一个 turn 算
- 加分：能指出实务上更常用 `tokenizer.apply_chat_template` + TRL `DataCollatorForCompletionOnlyLM` 自动处理，手写 mask 主要在调试或定制场景

</details>

**Q2（trade-off）**：sample packing 比朴素 padding throughput 高多少？什么场景下不该用 packing？

<details>
<summary>Answer sketch</summary>

**Throughput 提升量级**：

- 取决于数据长度方差。SFT 数据长度从 200 到 4000 的典型场景，padding ratio 通常 50-70%——**packing 可把吞吐推到 padding 的 1.5-3 倍**
- 长度均匀的数据（如所有样本都接近 max_len），padding 已经几乎没浪费，packing 收益 < 10%
- 极不均匀数据（有几条 32k 与一堆 200），packing 提升能到 4-5×

**不该用 packing 的场景**：

1. **长 sample 主导**：如果数据集大部分样本本身就接近 max_seq_length（如 long-context reasoning），packing 拼不进多少额外样本，但带来 block-diag mask 复杂度，得不偿失
2. **推理 / 生成阶段**：每条样本独立 KV cache，packing 在生成上没意义且会破坏 batch generate 的语义；推理一律用 padding 路线（且左 padding）
3. **样本之间需要严格隔离的 RL 场景**：DPO / GRPO 等 sample-level loss，packing 也能用但 reward 计算要 carefully mask，工程复杂度高
4. **不支持 varlen attention 的旧栈**：如 PyTorch SDPA 不支持 block-diag mask 时，packing 要 fallback 到显式构造 $(T, T)$ mask 矩阵，对 32k 序列长度显存 > 4GB，反而拖慢

加分：能指出 packing + position_ids reset 与 RoPE 配合的工程细节；能区分 packing 与 padding-free training（后者更激进，无 chunk 边界）。

</details>

**Q3（调参）**：你 SFT 一个 7B model，eval loss 在 epoch 1 末就开始上升，可能的 3 个原因？

<details>
<summary>Answer sketch</summary>

至少答到 3 个原因：

1. **lr 太大**——SFT 标准 lr 5e-6 ~ 5e-5，如果错用预训的 3e-4 或 1e-4，模型权重被冲走，1 epoch 内就过拟合训练数据 + 失去预训通用能力。**先把 lr 调成 1e-5 重训**
2. **数据集太小** / 信息密度低——SFT 通常需要至少几千到几万条高质量样本；如果只有几百条或样本质量差（重复、格式不一致），1 epoch 就 overfit。**收集更多数据或用 LoRA 减少有效参数量**
3. **epoch 数据被 trainer 记成 "重复看"**——训练循环把同一份小数据集反复滚多次（如 effective_epoch 算错），eval loss 实际上是 epoch 2-3 的状态。**确认 dataloader 是否有重复采样、检查 wandb 上的 step / sample 计数**
4. **train / eval 分布严重不一致**——eval set 来自不同 domain（如 train 是英文、eval 是中文），1 epoch 后模型偏向 train domain，eval loss 上升不是过拟合而是分布偏移
5. **chat template / loss mask 错配**——eval 时 mask 规则与 train 不同（如 train mask user、eval 全算），eval loss 数值不可比；**对齐 eval 与 train 的 mask 规则**
6. **base model 与 SFT 数据领域差异过大**——如用 code-only base 做 dialog SFT，第一轮就开始遗忘 code 能力，导致 eval（如果包含 code）loss 上升

加分：能指出做 LoRA 而不是 full SFT 时 overfit 风险更低（trainable 参数少）；能指出加 weight decay / 用更小 lr / 早停三种缓解手段的优先级。

</details>

---

## 13. 延伸阅读

- [HuggingFace TRL — SFTTrainer 文档](https://huggingface.co/docs/trl/sft_trainer) — `packing` / `completion_only_loss` / `chat_template` 三个核心开关的官方说明
- [HuggingFace Chat Templating 文档](https://huggingface.co/docs/transformers/main/chat_templating) — `apply_chat_template` 全用法、jinja 模板自定义、训练/推理对齐
- [Tülu 3 GitHub & Tech Report](https://github.com/allenai/open-instruct) — 现代 SFT pipeline 完整 reproduce，配 SFT 数据混合策略与 hyperparam sweep
- [Llama-2 Tech Report § Fine-tuning](https://arxiv.org/abs/2307.09288) — SFT 工程细节最详尽公开来源，chat template + 超参全在 §3
- [TRL `DataCollatorForCompletionOnlyLM` 源码](https://github.com/huggingface/trl/blob/main/trl/trainer/utils.py) — 看现代 collator 怎么找 response template 并构造 loss mask
- 推荐继续读本教程的 **8.3 节《LoRA / QLoRA / DoRA 原理与实现》**——SFT 训练流程定了之后，怎么用 LoRA 把显存压到 1/4
- 推荐继续读本教程的 **9.3 节《PPO 原理与在 LLM 上的形式》**——RLHF 阶段会复用本节的 chat template 与 loss mask 工程，是 SFT 的延伸
