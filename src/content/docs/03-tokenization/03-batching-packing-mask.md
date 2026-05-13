---
title: "3.3 Batching / packing / 注意力 mask"
description: "把变长 token 序列塞进 GPU 的两条主流路线——padding + mask 与 sample packing + block-diagonal mask——讲清原理、代码与坑，给后面 4.1 self-attention 的 causal mask、4.7 KV cache、8.2 SFT loss mask、Module 7 训练 infra 打地基。"
---

> ⏱ 预计阅读 35 分钟 ｜ 难度 ★★ ｜ 前置：3.1 BPE / SentencePiece、3.2 词表与特殊 token、1.5 PyTorch 工作流

## 一句话本节讲什么

把变长 token 序列塞进 GPU 的两条主流路线——**padding + mask** 与 **sample packing + block-diagonal mask**——讲清原理、代码与坑，给后面 4.1 self-attention 的 causal mask、4.7 KV cache、8.2 SFT loss mask、Module 7 训练 infra 打地基。

---

## 1. Mental model（直觉）

LLM 输入天然是**变长**的：用户的一句问话可能 12 个 token，下一条样本可能 1800 个 token。但 GPU 算 attention 时矩阵乘法要求 **batch 里每条序列长度一致**——`(B, T, H)` 这种规整张量才能上 cuBLAS / FlashAttention kernel。怎么把"参差不齐的 token 列表"喂成"齐整的 3D 张量"，这就是本节要回答的全部问题。

历史上有两套答案：

**答案一：padding（早期 BERT / GPT-2 / 几乎所有 HuggingFace 默认行为）**。把 batch 里所有序列右补 `[PAD]` token 到 `max_len`，再用一个 `attention_mask` 告诉 attention "这几个位置是假的、别看"。简单粗暴，但浪费——一个 batch 里若 99 条短样本 + 1 条长样本，99 条都得补到长样本的长度，算力被 pad 吃掉一大半。

**答案二：sample packing（现代 LLaMA / Qwen / DeepSeek 训练标配）**。既然 pad 浪费，就不 pad——把多条短样本**首尾相接**拼成一条长序列，凑到 `max_len` 再切。一条 packed 序列里塞了 7-30 条原样本，几乎零浪费。代价是要小心一件事：**不能让 sample A 的 token attend 到 sample B 的 token**（数据泄露），所以 attention mask 不再是简单下三角，而是**块对角下三角**——每个 block 内部 causal，block 之间互相屏蔽。

ASCII 直觉图——同 4 条样本（长度 3 / 5 / 2 / 4），两种打包方式：

```
方式 A：padding 到 max_len=5
  s1: [t t t P P]                ┐
  s2: [t t t t t]                │ batch shape = (4, 5)
  s3: [t t P P P]                │ 实际 token 14 / 总格子 20
  s4: [t t t t P]                ┘ 浪费 30%

方式 B：packing 到一条长度 14
  packed: [t t t | t t t t t | t t | t t t t]
  block-diagonal mask：4 个 causal block，互相不通气
  batch shape = (1, 14)，零浪费
```

记牢这两幅画面，剩下的内容就是把 mask 怎么写、loss 怎么算、踩坑怎么躲填实。

---

## 2. 公式与原理

### 2.1 Attention mask 的物理意义

回忆 attention 的核心一步（详细公式留到 4.1）：先算原始 score $S = QK^\top / \sqrt{d_k} \in \mathbb{R}^{T \times T}$，再 softmax 归一化，再乘 $V$。**mask 的工作发生在 softmax 之前**——把不该看的位置的 score 置成 $-\infty$（工程上常用 `-1e4` 或 `-1e9`），softmax 之后这些位置的概率就被压到 0：

$$\text{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right) V, \quad M_{ij} \in \{0, -\infty\}$$

mask $M \in \mathbb{R}^{T \times T}$ 只决定"$i$ 能不能看到 $j$"。LLM 实战里有四种典型 mask：

- **Padding mask**：屏蔽 pad 位置。形状 $(B, T)$ 表示"每条样本哪几个位置是真的"，框架内部会扩展成 $(B, 1, T, T)$ 与 score 相加。
- **Causal mask（autoregressive mask）**：让位置 $i$ 只能看 $j \le i$，下三角全 0、上三角全 $-\infty$。这是 decoder-only LLM 的灵魂——保证训练时**每个位置只用过去信息预测下一个 token**，否则就是 cheating。
- **Bidirectional mask（实际上是没 mask）**：BERT / encoder 模型，每个位置都能看全句。
- **Prefix LM mask**：前缀（prompt）双向、后缀（response）causal。T5 / GLM / UL2 用这个范式。

decoder-only LLM 训练时通常**两种 mask 同时存在**——causal mask 强制因果性，padding mask 屏蔽 pad 位置。框架会把它们逻辑 OR 合并成一个最终 mask 矩阵。

### 2.2 Padding mask 的两种格式

HuggingFace 约定 `attention_mask` 是 $(B, T)$，**1 表示真 token、0 表示 pad**：

```
batch = ["猫坐", "猫坐在垫子上"]
input_ids:    [[101, 731, 2, PAD, PAD],
               [101, 731, 5, 8, 9, 102]]      # 第一条 PAD 到 6
attention_mask = [[1, 1, 1, 0, 0, 0],
                  [1, 1, 1, 1, 1, 1]]
```

进入 attention 前要扩展成 $(B, 1, T_q, T_k)$ 加到 score 上。一个 token $i$ 不能 attend 到 pad token $j$ 意味着 mask 矩阵第 $i$ 行的第 $j$ 列要被屏蔽。把 $(B, T)$ 的 1D mask 扩展成 $(B, 1, 1, T)$ 再与 score 广播相加，就能把每一行里 pad 列对应的位置统一压到 $-\infty$。

`torch.nn.functional.scaled_dot_product_attention` 提供了现代标准接口：

```python
out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=True)
```

`is_causal=True` 让框架内部生成下三角 mask（甚至会调 FlashAttention kernel，比手写快很多）；额外的 padding 信息通过 `attn_mask` 传入。

### 2.3 左 padding vs 右 padding

**训练**：通常**右 padding**（pad 在右边）。前向只看真 token + causal mask，pad 位置被 attention 屏蔽 + loss 也被 mask 掉，没影响。

**生成 / 推理**：**必须左 padding**（pad 在左边）。原因：`model.generate()` 是从最后一个 token 开始往后续写、KV cache 也按"最后位置 = 当前 token"组织。如果 pad 在右边，"最后位置"是 pad，generate 把 pad 当成 prompt 末尾继续写，结果一塌糊涂。

```
左 padding（generate 用）：
  [PAD PAD PAD T T T]   ← 最后位置是真 token，generate 接着这里写
右 padding（train 用）：
  [T T T PAD PAD PAD]   ← 最后位置是 pad，generate 会乱
```

设置方式：`tokenizer.padding_side = 'left'`。所有现代 LLM 推理代码（包括 vLLM、SGLang）都默认 / 强制左 padding。

### 2.4 Sample packing 与 block-diagonal mask

设把 $K$ 条样本拼接成一条，长度依次为 $\ell_1, \ell_2, \dots, \ell_K$，总长 $T = \sum_k \ell_k$。block-diagonal mask 定义：

$$M_{ij} = \begin{cases} 0 & \text{若 } i, j \text{ 属于同一条原样本且 } j \le i \\ -\infty & \text{否则} \end{cases}$$

也就是：每条原样本内部是标准 causal 下三角，跨样本一律屏蔽。

**示例**：1 条 packed 序列长度 10，由 $s_1$（4 token）+ $s_2$（6 token）拼成。把 1 标记为"可见"、0 标记为"屏蔽"，mask 矩阵：

```
       j=0 1 2 3 | 4 5 6 7 8 9
i=0  [  1 0 0 0 | 0 0 0 0 0 0 ]
i=1  [  1 1 0 0 | 0 0 0 0 0 0 ]    s1 内部 causal
i=2  [  1 1 1 0 | 0 0 0 0 0 0 ]
i=3  [  1 1 1 1 | 0 0 0 0 0 0 ]
     ────────────┼─────────────
i=4  [  0 0 0 0 | 1 0 0 0 0 0 ]
i=5  [  0 0 0 0 | 1 1 0 0 0 0 ]
i=6  [  0 0 0 0 | 1 1 1 0 0 0 ]    s2 内部 causal
i=7  [  0 0 0 0 | 1 1 1 1 0 0 ]
i=8  [  0 0 0 0 | 1 1 1 1 1 0 ]
i=9  [  0 0 0 0 | 1 1 1 1 1 1 ]
```

显式构造一个 $(T, T)$ 的 mask 矩阵在 $T = 8192$ 时占 $8192^2 \times 1\text{B} = 64\text{ MB}$，多头多层后压力很大。**FlashAttention 2 提供 `flash_attn_varlen_func`**，只需传 `cu_seqlens`（cumulative sequence lengths，累积长度），完全不构造 mask：

```
样本长度 [4, 6, 3]   →   cu_seqlens = [0, 4, 10, 13]
```

kernel 内部按 `cu_seqlens` 分段做 causal attention，每段 $O(\ell_k^2)$ 而非 $O(T^2)$，又省显存又省算力。这是现代 LLM 训练（LLaMA-3 / Qwen / DeepSeek）packing 路线的工程基础。

### 2.5 Loss mask（与 attention mask 区分）

**两件事完全独立**：

- **attention mask**：控制每个位置在 forward 时**能看见谁**
- **loss mask**：控制每个位置在 backward 时**对哪些位置算 loss**

SFT 场景的典型例子——一条 `[user_tokens, assistant_tokens]` 样本，attention 上 assistant 还是要能看到 user（不然没法回答），但 loss 只对 assistant 部分计算（user 部分是 prompt，不该让模型学着复读）：

```
input_ids:  [u u u u | a a a a a]
labels:     [-100 -100 -100 -100 | a a a a a]   # user 部分置 -100
```

PyTorch 的 `F.cross_entropy(..., ignore_index=-100)` 默认会跳过 label = -100 的位置，loss 自然只在 assistant token 上累加。Module 8.2 会把这套展开。

---

## 3. 最小代码示例

### 3.1 手写 padding mask + causal mask + SDPA

```python
import torch
import torch.nn.functional as F

B, T, H = 2, 5, 8
x = torch.randn(B, T, H)
# 假设 batch[0] 真长度 3、batch[1] 真长度 5
seqlen = torch.tensor([3, 5])

# 1. padding mask: (B, T)，1=真 token, 0=pad
pad_mask = torch.arange(T)[None, :] < seqlen[:, None]
# 扩展成 attention 能用的 (B, 1, 1, T)：False 处会被屏蔽
attn_mask = pad_mask[:, None, None, :]   # (B, 1, 1, T)

# 2. causal mask: (T, T) 下三角 True
causal = torch.tril(torch.ones(T, T, dtype=torch.bool))   # (T, T)

# 3. 合并：每个 q 位置 i 可见 = causal(i, j) AND not_pad(j)
final_mask = causal[None, None, :, :] & attn_mask         # (B, 1, T, T)

# 4. 走 SDPA：传布尔 mask（True=可见）
q = k = v = x.unsqueeze(1)                                 # (B, 1=head, T, H)
out = F.scaled_dot_product_attention(q, k, v, attn_mask=final_mask)
print(out.shape)                                           # torch.Size([2, 1, 5, 8])
print("padding mask:\n", pad_mask.int())
print("causal mask:\n", causal.int())
```

要点：`final_mask` 同时屏蔽未来 + pad，是 decoder-only LLM 训练时实际进 attention 的 mask。`F.scaled_dot_product_attention` 在 PyTorch 2.0+ 是现代标准 API，会自动选 FlashAttention / memory-efficient / math 三种 backend 中最合适的一个。

### 3.2 Sample packing + block-diagonal mask + cu_seqlens

```python
import torch

# 两条样本 s1=4 token, s2=6 token，packing 到长度 10
s1 = torch.tensor([11, 12, 13, 14])
s2 = torch.tensor([21, 22, 23, 24, 25, 26])
packed = torch.cat([s1, s2])                  # shape (10,)
seqlens = torch.tensor([4, 6])

# 1. 构造 block-diagonal causal mask (T, T)
T = packed.size(0)
# 每个位置属于哪个 block
block_id = torch.repeat_interleave(torch.arange(len(seqlens)), seqlens)  # [0,0,0,0,1,1,1,1,1,1]
same_block = block_id[:, None] == block_id[None, :]                       # (T, T)
causal     = torch.tril(torch.ones(T, T, dtype=torch.bool))
mask       = same_block & causal                                           # block-diag + causal

print(mask.int())
# 第 4 行起跨 block 的位置被屏蔽，第 4-9 行内部 causal

# 2. cu_seqlens 风格（FlashAttention varlen 接口需要）
# flash_attn_varlen_func(q, k, v, cu_seqlens_q=..., cu_seqlens_k=..., max_seqlen_q=..., max_seqlen_k=..., causal=True)
# 这里 q, k, v 形状是 (total_tokens, n_heads, head_dim)，不带 batch 维
cu_seqlens = torch.cat([torch.zeros(1, dtype=torch.int32),
                        seqlens.cumsum(0).to(torch.int32)])
print("cu_seqlens:", cu_seqlens.tolist())     # [0, 4, 10]
# kernel 内部按 [0:4]、[4:10] 分段做 causal attention，无需构造 (T, T) 矩阵
```

`cu_seqlens` 一定是**长度 K+1、首位为 0、严格递增**的数组。新手最常见的错是漏掉首位 0、或者忘了累加（直接传 `[4, 6]` 而不是 `[0, 4, 10]`），FlashAttention 会段错误或者给出垃圾结果。

### 3.3 Loss mask 示例

```python
import torch
import torch.nn.functional as F

# 词表大小 100，序列长 8，前 4 token 是 user prompt、后 4 是 assistant
V, T = 100, 8
logits = torch.randn(1, T, V)                          # (B, T, V)
input_ids = torch.tensor([[5, 6, 7, 8, 20, 21, 22, 23]])
labels    = input_ids.clone()
labels[:, :4] = -100                                    # mask 掉 user 部分

# CE 只在 assistant 4 个 token 上算
loss = F.cross_entropy(
    logits.view(-1, V),
    labels.view(-1),
    ignore_index=-100,
)
print(loss)   # 只对 4 个有效位置的平均 loss
```

`ignore_index=-100` 是 PyTorch CE 的默认值，所以工程上把不参与 loss 的位置统一置 -100 是约定俗成。HuggingFace 的 `DataCollatorForLanguageModeling` 与 TRL 的 `SFTTrainer` 内部都是这个套路。

---

## 4. 工程踩坑与经验

- ❗ **生成时必须左 padding，否则 `model.generate()` 输出乱码**。原因：`generate` 把序列最后一个位置当成"当前 token"开始续写，KV cache、`position_ids` 都按这个对齐。右 padding 时最后一个位置是 `[PAD]`，模型实际是从 pad 出发续写。**调用 `generate` 前 `tokenizer.padding_side = 'left'` 是铁律**，HF Transformers 官方 warning 也会提醒。
- ❗ **Sample packing 不加 block-diagonal mask 等于数据互相污染**。曾有团队上线 packing 没加 mask，训出的模型在长 prompt 上会复读上一条样本的内容——sample A 的 token 通过 attention "记住"了 sample B 的语义，等价于把训练数据偷偷融合。轻则 eval 抖动，重则 RLHF 阶段 reward hacking 加剧。任何 packing 实现都要在 unit test 里跑一条"两个独立样本拼起来 vs 分开过 forward，logits 必须 bit-exactly 相等"的对比。
- ❗ **HF `attention_mask` 的形状陷阱**。HuggingFace 模型接口约定 `attention_mask` 是 $(B, T)$，但模型内部会通过 `_prepare_4d_causal_attention_mask` 扩展成 $(B, 1, T, T)$ 并把 0 位置加 $-10000$（不是 $-\infty$，避免 fp16 NaN）。如果你自己写 attention 接口接收 $(B, T)$ 的 mask 直接当 $(B, T, T)$ 用，行/列广播全错。读 HF 源码看清调用边界是哪一层。
- ❗ **`labels[i] = -100` 漏掉 user prompt 部分 → 模型学着复读**。SFT 数据构造最常见 bug：忘了 mask 系统 prompt + user 部分，模型把"原样输出 user 输入再回答"当成正确行为，eval 时一上来先复述用户问题，然后才回答。修复方法：构造 labels 时按 chat template 的 `<|im_start|>user ... <|im_end|><|im_start|>assistant` 边界精确切，每个 turn 的 user 段都要 mask。
- ❗ **Chat template 里 `<bos>` / `<eos>` 错位一格 → loss mask 错位 → 模型 EOS 永远不出现**。例：HF 某些 tokenizer 默认会自动加 `<bos>` 但你又手动加了一遍，导致后续所有 label 偏移一位、最关键的 `<eos>` 那一位 label 变成 -100，模型永远学不到"什么时候停"。生成时只能靠 `max_new_tokens` 强切。**调试方法**：dump 出 `(input_ids, labels)` 的对照表，肉眼检查每个 turn 的 boundary 是否对齐。
- ❗ **Left padding + causal mask 在 LLaMA / Qwen / Mistral 上需要专门处理 `position_ids`**。RoPE 的位置编码与序列绝对位置强绑定，左 padding 后真 token 的"绝对位置"已经偏移。HF Transformers 对这些 model 已经在 `prepare_inputs_for_generation` 里重算 `position_ids`（pad 位置 0、真 token 从 0 开始递增），但你自己写 forward 时务必跟上这个约定，否则 batch 推理会偷偷比单条慢且效果劣化。
- ❗ **FlashAttention `cu_seqlens` 必须从 0 开始、int32、长度 K+1**。新手错法：传 `[4, 10]` 而不是 `[0, 4, 10]`；或传 int64 触发 dtype check 报错；或在 GPU 上传 CPU tensor。建议封装一个 helper：`cu_seqlens = F.pad(seqlens.cumsum(0), (1, 0)).to(device, dtype=torch.int32)`，一次写对终身受益。
- ❗ **Padding mask 里 `0` 与 `-inf` / `-1e9` 的混用**。布尔 mask（`True=可见`）、加性 mask（`0=可见、-inf=屏蔽`）两种约定并存。`F.scaled_dot_product_attention` 接受布尔（True 可见）或浮点（0 可见、-inf 屏蔽）；但传错一个像反义词的 mask（True=屏蔽）会把整个 attention 翻转、训出来的模型彻底废掉。**建议**：写一行注释明确约定，并用 `assert mask.dtype == torch.bool` 做 contract check。
- ❗ **Padding 占比是个该监控的指标**。训练日志里加一行 `padding_ratio = (attention_mask == 0).float().mean()`，超过 30% 就考虑切换到 packing。生产经验：dialog 数据 padding ratio 常常 50-70%，packing 一上吞吐直接 +60%。

---

## 5. 经典 paper

- **Vaswani et al., 2017 — Attention Is All You Need** — Transformer 与 attention mask 的源头，§3.2.3 "Attention masks" 一节直接定义了 decoder 的 causal mask。读它确认"为什么 mask 要在 softmax 之前加"——因为 softmax 是非线性的，事后置 0 会破坏归一化。
- **Dao et al., 2022 / 2023 — FlashAttention 1 / 2** — IO-aware attention 的开山，FA2 引入 `flash_attn_varlen_func` 让 sample packing 几乎零额外开销。读 FA2 的 §3.2 "Variable-length sequences" 一节，理解 `cu_seqlens` 接口为什么能避免显式 mask。Module 5.3 会展开 kernel 细节。
- **Touvron et al., 2023 — LLaMA-2** — 公开技术报告里明确写了"all training data was concatenated and chunked"——sample packing 是现代主流 LLM 训练的工程标准。同期 Mistral、Qwen、DeepSeek 的 report 都沿用此惯例。读它确认这不是学术 trick 而是工业常态。
- **HuggingFace TRL — `SFTTrainer` 文档** — 实操参考。`SFTTrainer(packing=True)` 一行开关就启用 sample packing，内部用 `ConstantLengthDataset` 自动拼接 + 配 FA2 varlen。读源码 `trl/trainer/sft_trainer.py` 看现代 SFT 框架怎么把这套流程包好。

---

## 6. 自测与面试题

**Q1（概念）**：padding mask 与 causal mask 的物理意义有何不同？同一个 batch 在 decoder-only LLM 训练时通常需要哪些 mask？

<details>
<summary>Answer sketch</summary>

- **Padding mask**：屏蔽 batch 内 pad 位置，避免 attention 看到无效的 `[PAD]` token。形状 $(B, T)$，跟 batch 内每条样本的真实长度有关。
- **Causal mask**：屏蔽未来位置，让 token $i$ 只能 attend 到 $j \le i$。形状 $(T, T)$ 下三角，与 batch 无关，只跟 decoder-only 的因果建模目标有关。
- **Decoder-only LLM 训练时两者都要**：framework 内部会把它们逻辑 AND 合并成最终 $(B, 1, T, T)$ mask。少 padding mask 模型会从 pad 学到噪声；少 causal mask 模型直接看见未来 → 训练 loss 看似很低但根本没学会预测。
- 加分：sample packing 时再叠一层 block-diagonal，三种 mask 合一。

</details>

**Q2（实战）**：sample packing 时 block-diagonal mask 如何构造？写出 `(B=1, T=10)` 包含 `[s1=4, s2=6]` 两个 sample 的 mask 矩阵（用 0/1 表示，1=可见、0=屏蔽）。

<details>
<summary>Answer sketch</summary>

构造步骤：

1. 计算 `block_id = [0,0,0,0,1,1,1,1,1,1]`（每个 token 属于哪条原样本）
2. `same_block[i,j] = (block_id[i] == block_id[j])`，得到一个块对角 True 矩阵
3. 与下三角 causal mask 做 AND，得到最终 mask

矩阵长这样（**1=可见、0=屏蔽**）：

```
       j=0 1 2 3 | 4 5 6 7 8 9
i=0  [  1 0 0 0 | 0 0 0 0 0 0 ]
i=1  [  1 1 0 0 | 0 0 0 0 0 0 ]
i=2  [  1 1 1 0 | 0 0 0 0 0 0 ]
i=3  [  1 1 1 1 | 0 0 0 0 0 0 ]
i=4  [  0 0 0 0 | 1 0 0 0 0 0 ]
i=5  [  0 0 0 0 | 1 1 0 0 0 0 ]
i=6  [  0 0 0 0 | 1 1 1 0 0 0 ]
i=7  [  0 0 0 0 | 1 1 1 1 0 0 ]
i=8  [  0 0 0 0 | 1 1 1 1 1 0 ]
i=9  [  0 0 0 0 | 1 1 1 1 1 1 ]
```

加分：指出工程上不显式构造这个矩阵，而是传 `cu_seqlens = [0, 4, 10]` 给 `flash_attn_varlen_func`，kernel 内部按段处理，省 $O(T^2)$ 的 mask 显存。

</details>

**Q3（trade-off）**：padding vs packing vs padding-free training 三种数据组织方式有什么取舍？什么场景选哪个？

<details>
<summary>Answer sketch</summary>

| 方案 | 浪费 | 实现复杂度 | 适用场景 |
|---|---|---|---|
| **Padding** | 高（pad 比例 30-70%） | 低，HF 默认 | 推理 / 生成；训练数据长度方差小；快速原型 |
| **Sample packing** | 低（< 5%） | 中，需要 block-diag mask + cu_seqlens | 主流 LLM 训练（pretrain / SFT），数据长度方差大 |
| **Padding-free training** | 零 | 高，要求 model 内部全栈 varlen kernel + position_ids 重写 | 最前沿训练（2024+），HF Trainer `use_padding_free=True`、Liger-Kernel；进一步省 10-30% 算力 |

加分点：

- **生成 / 推理一律用 padding 路线（且左 padding）**——packing 在生成上没意义（每条样本独立 KV cache），且会破坏 batch generate 的语义。
- Packing 适合 SFT / pretrain；DPO / GRPO 这类 sample-level 比较 loss 也能 packing 但要 carefully mask reward。
- Padding-free 是 packing 的"无 chunk 边界"版本——packing 还有切到 `max_len` 的轻微浪费，padding-free 直接每条样本进 varlen attention，理论最优。
- 实际上现代框架（Liger / TRL latest）三者已经统一在同一套 varlen attention 实现下，配置开关而已。

</details>

---

## 7. 延伸阅读

- [PyTorch — `scaled_dot_product_attention` 文档](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) — 现代标准 API 的官方说明，包含 mask 形状约定、自动选 backend 的规则
- [HuggingFace TRL — SFTTrainer Packing 教程](https://huggingface.co/docs/trl/sft_trainer#packing-dataset--constantlengthdataset) — `packing=True` 一行开关背后的实现细节
- [FlashAttention GitHub — `flash_attn_varlen_func`](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_interface.py) — variable-length 接口的源码，看 `cu_seqlens` 怎么传
- [Hugging Face — Padding-Free Training Blog](https://huggingface.co/blog/packing-with-FA2) — FA2 + packing + padding-free 的端到端实战，配 benchmark
- 推荐继续读本教程的 **4.1 节《Self-attention：QKV 与 scaled dot-product》**——把本节的 causal mask 嵌入完整 attention 公式
