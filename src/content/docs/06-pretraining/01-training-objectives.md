---
title: "6.1 训练目标：CLM / MLM / Prefix-LM / FIM"
description: "5.1 已经把\"哪个架构对应哪个训练目标\"讲清楚——本节把每个训练目标的公式 / loss 计算 / mask 实现 / 工程细节一次拆透，特别是为什么现代 LLM 几乎全用 CLM、为什么 code model 必须加 FIM、为什么 MLM 在 LLM scaling 下失宠、以及 T5 的 span corruption 与 UL2 的 mixture-of-denoisers 在哪里。"
---

> ⏱ 预计阅读 35 分钟 ｜ 难度 ★★ ｜ 前置：5.1（三架构对比）、1.4（CE / KL）

## 一句话本节讲什么

5.1 已经把"哪个架构对应哪个训练目标"讲清楚——本节把每个训练目标的**公式 / loss 计算 / mask 实现 / 工程细节**一次拆透，特别是为什么现代 LLM 几乎全用 **CLM**、为什么 code model 必须加 **FIM**、为什么 MLM 在 LLM scaling 下失宠、以及 T5 的 **span corruption** 与 UL2 的 **mixture-of-denoisers** 在哪里。

---

## 1. Mental model（直觉）

### 1.1 训练目标 ≠ 架构

5.1 把"架构 + 训练目标"作为一个 bundle 介绍（BERT = encoder + MLM、GPT = decoder + CLM、T5 = enc-dec + span corruption）——但严格说**训练目标和架构是两件事**，只是历史上某些组合最自然才长在了一起。例如：

- decoder-only 也能做 **prefix-LM**（GLM）、**FIM**（CodeLlama / DeepSeek-Coder）——只改 mask 与样本格式，架构不动
- encoder-decoder 也能做 **CLM**（Bavarian 2022 实验过）——只是没有 span corruption 香
- 同一个 model 可以**同时**训多个目标（UL2、PaLM-2 都是这样）

所以本节的视角是：**给定一个 decoder-only 架构（现代 LLM 主流），你可以选 CLM / Prefix-LM / FIM 中任意一种或混合作为训练目标，每种目标的本质区别在 (1) 哪些 token 算 loss、(2) 算 loss 时能 attend 到哪些 token**。

### 1.2 5 种训练目标的"loss 信号谁贡献"对比图

```
=========  CLM (GPT, LLaMA)  =========     每个位置都预测下一个 token，loss 信号 100%
input:   the  cat  sat  on   the
target:  cat  sat  on   the  mat
loss:     ✓    ✓    ✓    ✓    ✓    ← 全部贡献

=========  MLM (BERT)  ==============      只在 mask 位置算 loss，loss 信号 ~15%
input:   [CLS] the [MASK] sat on the mat [SEP]
                  ↑预测=cat
loss:     ·    ·    ✓    ·   ·  ·   ·   ·   ← 只有 mask 位置

=========  Prefix-LM (GLM)  =========      前缀 bi-attention 但不算 loss，后缀 causal 算 loss
input:   [translate]  [le chat]      ← 前缀（任务指令）  | 后缀（生成内容）
loss:     ·  ·  ·     ✓  ✓  ✓                          后缀 100%

=========  FIM (CodeLlama)  =========      重排为 PRE/SUF/MID 顺序，仍是 CLM
input:   <PRE> def fib(n): <SUF> return result <MID> result = [0, 1] for ... <eos>
loss:     ·    ·  ·  ·     ·    ·  ·  ·       ✓  ✓  ✓  ✓  ✓  ✓  ✓     ← middle 段 100%

=========  Span Corruption (T5)  ====      encoder 看带洞输入，decoder 自回归生成 spans
encoder: Thank you <X> me to <Y> week <Z>
decoder: <X> for inviting <Y> your party last <Z>     ← decoder 全算 loss
```

记住这 5 张草图——本节后面所有数学和代码都是对它们的精确化。

### 1.3 为什么训练目标这么重要

预训练阶段 **>99% 的算力**花在某个固定训练目标上，目标的选择直接决定：

1. **数据效率**：每个 token 能产生多少梯度信号（CLM 100% vs MLM 15%，差 6.7×）
2. **下游任务适配**：CLM → in-context learning / 生成；MLM → 句向量 / 分类；FIM → code completion（在中间填空）
3. **scale 友好性**：训练目标越简单越统一，越容易吃 scaling law 红利——CLM 一统江湖的核心原因之一
4. **infra 复杂度**：MLM / span corruption 要"造" mask，prefix-LM 要 per-sample 切分，FIM 要在 dataloader 阶段重排——pure CLM 是最"傻"也最高效的

---

## 2. 公式与原理

### 2.1 CLM（Causal Language Modeling）

记输入序列 $\mathbf{x} = (x_1, x_2, \dots, x_T)$，每个 $x_t$ 是 vocab 上的 token id。CLM 目标：

$$\mathcal{L}_{\text{CLM}}(\theta) = -\sum_{t=1}^{T-1} \log P_\theta(x_{t+1} \mid x_1, \dots, x_t) = -\sum_{t=1}^{T-1} \log P_\theta(x_{t+1} \mid x_{\le t})$$

由 chain rule：$\log P(\mathbf{x}) = \sum_t \log P(x_{t+1} \mid x_{\le t})$，所以最小化 CLM loss = 最大化序列对数似然 = MLE。这就是为什么 CLM 在概率上是"最干净"的目标——它直接拟合数据的联合分布。

**实现要点**：

- 模型 forward：input `x[:, :T]`，输出 logits `(B, T, V)`
- target 是 input **shift right 一位**：`target[t] = x[t+1]`
- 对 logits `[:, :-1, :]` 与 target `[:, 1:]` 算 cross-entropy
- attention 用 causal mask（下三角），保证位置 $t$ 只 attend 到 $\le t$
- 每个位置都贡献 loss——**信号密度 100%**

注意一个常被混淆的点：**"shift right" 在实现上有两种等价写法**——(1) 在数据预处理阶段直接把 `target = x[1:]` 准备好，模型 forward 一次就出 loss；(2) 模型 forward 用全 `x`，输出 `logits[:, :-1]` 对 `x[:, 1:]` 算 loss（HuggingFace `LlamaForCausalLM` 内部就是这么做的）。两种数值完全等价，挑顺手的实现即可。

### 2.2 MLM（Masked Language Modeling）

随机选 token 位置子集 $\mathcal{M} \subset \{1, \dots, T\}$，$|\mathcal{M}| / T \approx 0.15$，被选中的位置经过 BERT 的 **80/10/10 trick** 替换：

- 80% 概率替换为特殊 token `[MASK]`
- 10% 概率替换为 vocab 中**随机 token**
- 10% 概率**保持原 token 不变**

记替换后序列为 $\tilde{\mathbf{x}}$，目标是预测被 mask 位置的原始 token：

$$\mathcal{L}_{\text{MLM}}(\theta) = -\sum_{t \in \mathcal{M}} \log P_\theta(x_t \mid \tilde{\mathbf{x}})$$

注意 attention 是 **bidirectional**——位置 $t$ 能 attend 到任意位置（包括位置 $t$ 自己）。这是 MLM 与 CLM 在 attention mask 上最本质的区别。

**80/10/10 trick 为什么不可省**：如果 100% 都用 `[MASK]`，模型会学到"看到 `[MASK]` 才需要预测，看到正常 token 就摆烂"——下游 fine-tune 时见不到 `[MASK]`，性能会暴跌。10% 随机替换强迫模型对**任意**位置都保持预测准备（因为随机替换的 token 看起来是正常 token）；10% 保持原 token 强迫模型不能简单地"看到 `[MASK]` 就预测、看到非 `[MASK]` 就 copy"。这个 trick 在 RoBERTa / ELECTRA / DeBERTa 中都被沿用。

**信号密度问题**：每个 sample 只有 ~15% 的 token 算 loss——这是 MLM 在 LLM scaling 下吃亏的根本原因（详见 5.1 §2.2）。曾有研究（如 RoBERTa）尝试 mask ratio 30% 甚至 40%，但 MLM 的本质是"用上下文预测被遮的 token"，mask 太多上下文不够；ratio 15% 是 BERT 经验最优值，沿用至今。

### 2.3 Prefix-LM

把序列分成前缀 $\mathbf{x}_{\text{pre}} = (x_1, \dots, x_p)$ 和后缀 $\mathbf{x}_{\text{suf}} = (x_{p+1}, \dots, x_T)$，训练目标：

- 前缀部分 attention 是 **bidirectional**（任意位置看任意前缀位置）
- 后缀部分 attention 是 **causal**（位置 $t$ 只看 $\le t$，包括所有前缀）
- loss **只在后缀部分**算

$$\mathcal{L}_{\text{prefix-LM}}(\theta) = -\sum_{t=p+1}^{T} \log P_\theta(x_t \mid \mathbf{x}_{\text{pre}}, x_{p+1}, \dots, x_{t-1})$$

**Mask 的精确形状**（前缀长度 $p = 3$、序列长度 $T = 5$）：

```
        key→  pre1 pre2 pre3 suf1 suf2
query↓
pre1         [ 1    1    1    0    0  ]   ← 前缀位置：bi（看所有前缀），但不能看后缀
pre2         [ 1    1    1    0    0  ]
pre3         [ 1    1    1    0    0  ]
suf1         [ 1    1    1    1    0  ]   ← 后缀位置：causal（看所有前缀 + 自己之前的后缀）
suf2         [ 1    1    1    1    1  ]
```

注意前缀 query 的行也只能看到前缀——不能看后缀，否则前缀位置就"偷看"了未来，破坏了 prefix-LM 的因果结构。GLM 论文用的就是这种 mask；UL2 的 X-denoising 也是变体之一。

**与 encoder-decoder 的关系**：可以把 prefix-LM 视作"用一个 stack 实现的 encoder-decoder"——前缀像 encoder（bi-attention），后缀像 decoder（causal + 能看 encoder 的全部）。差别是 encoder-decoder 用两个独立的 stack + cross-attention，prefix-LM 用一个 stack + 混合 mask。

**优点**：prompt 部分可以"细看"（双向编码器风格），生成部分仍是自回归（保留 LM 性质 + KV cache）；Wang 2022 实证 prefix-LM 在 zero-shot 略好于纯 CLM 1-2 个点。

**缺点**：mask 是 per-sample 的（不同 sample 的 prefix 长度不同），训练 infra 比 pure CLM 复杂；FlashAttention 不直接支持 prefix mask（需要 fallback 到 SDPA 或自定义 kernel），训练 throughput 略慢——这也是它没大规模流行的工程原因。

### 2.4 FIM（Fill-in-the-Middle）

CLM 训出来的模型只能从左到右续写——**不能在已有上下文中间填空**。但 code completion 的最常见场景是 IDE 里光标停在函数中间，需要 model 看到 prefix（光标前）+ suffix（光标后），生成 middle（光标位置该填什么）。

Bavarian et al. 2022 的解决方案极其优雅——**不改架构、不改 loss，只改数据顺序**。原序列 $\mathbf{x}$ 随机切成三段 $\mathbf{x} = \text{prefix} \oplus \text{middle} \oplus \text{suffix}$，重排为：

$$\text{PSM 顺序：} \quad \langle\text{PRE}\rangle \; \text{prefix} \; \langle\text{SUF}\rangle \; \text{suffix} \; \langle\text{MID}\rangle \; \text{middle} \; \langle\text{eos}\rangle$$

或者：

$$\text{SPM 顺序：} \quad \langle\text{PRE}\rangle \; \langle\text{SUF}\rangle \; \text{suffix} \; \langle\text{MID}\rangle \; \text{prefix} \; \text{middle} \; \langle\text{eos}\rangle$$

其中 `<PRE>`、`<SUF>`、`<MID>` 是新加的 special token（sentinel）。重排后**仍按 CLM loss 训练**——模型学会"看到 `<MID>` 就开始生成 middle"。

**Inference 阶段**：用户场景是"给 prefix 和 suffix，请填 middle"。把 input 拼成 `<PRE> prefix <SUF> suffix <MID>`，让模型从 `<MID>` 之后续写直到 `<eos>` 即可。

**FIM rate**：训练时**只对部分样本做 FIM 重排**，剩余样本仍是普通 CLM。Bavarian 2022 推荐 50%（一半 FIM、一半 CLM）；DeepSeek-Coder 用 50%，StarCoder 用 50%。比例太低（< 30%）模型 FIM 能力弱；太高（> 70%）普通续写能力会下降。

**PSM vs SPM 的微妙差异**：

- **PSM**（Prefix-Suffix-Middle）：先看 prefix，再看 suffix，最后生成 middle。Bavarian 2022 默认用这个
- **SPM**（Suffix-Prefix-Middle）：先看 suffix，再看 prefix，最后生成 middle。**优点**：在 left-to-right inference 时，已知 prefix 通常是"用户在编辑的代码上文"，把它紧挨在生成位置之前，在 KV cache 复用上更友好（prefix 可以共享 cache，suffix / middle 独立）
- DeepSeek-Coder 默认 SPM，CodeLlama 训练时两种都做（hyperparam ablation 显示差异不大）

**为什么 FIM 是 code model 必加目标**：code completion 的真实场景 90% 是 fill-in-the-middle（在函数中间补一行），不是从头续写。纯 CLM 训练的 model 在 single-line completion 上还能凑合（因为光标后通常是空），但 multi-line completion / 函数体补全场景几乎不可用。FIM 让一个 decoder-only model 同时具备"续写"和"中间填空"两种能力。StarCoder / Code Llama / DeepSeek-Coder / CodeQwen 全都加了 FIM。

### 2.5 Span Corruption（T5）

T5 的训练目标——不是 mask 单个 token，而是 mask **连续 span**。每个 span 用一个 sentinel token 标记：

```
原文:    Thank you for inviting me to your party last week.
              ↓ random select spans, mean length 3, total ratio 15%
encoder: Thank you <X> me to your party <Y> week.
decoder: <X> for inviting <Y> last <Z>
```

形式化：corrupted input 给 encoder，target 是被 mask 的 spans 拼接（每段以对应 sentinel 开头，最后用一个 sentinel 结尾标记结束）。Decoder 自回归生成 target：

$$\mathcal{L}_{\text{span}}(\theta) = -\sum_{t=1}^{|\mathbf{y}|} \log P_\theta(y_t \mid y_{<t}, \tilde{\mathbf{x}})$$

**关键超参**：

- **Mean span length**：典型 3。Raffel 2020 ablation 显示 3-5 都行；< 2 等价于 MLM（退化到单 token），> 5 模型学不会（一个 sentinel 要解码太长一段）
- **Corruption ratio**：典型 15%（与 BERT MLM 一致）。T5 也试过 25% / 50%，效果接近
- **Sentinel 数量**：T5 默认 100 个（`<extra_id_0>` 到 `<extra_id_99>`），同一 sample 用到的 sentinel 上限 = 这个数

**为什么是 encoder-decoder 友好**：input 是带洞的连续序列（适合 encoder bi-attention），target 是 sentinel 装饰的拼接序列（适合 decoder autoregressive）。如果硬塞到 decoder-only 上，等价于 prefix-LM 的一种形式（前缀是 corrupted input + sentinel，后缀是 target）——UL2 的 R-denoising 就是这么做的。

### 2.6 UL2 / Mixture-of-Denoisers

Tay et al. 2022 提出：**一个 model 同时训多个 denoising 目标**，不同目标用不同 mode token 区分（像 prompt prefix）。UL2 的三种 denoising：

| 名称 | 含义 | mean span | corruption ratio | mode token |
|---|---|---|---|---|
| **R-denoising** (Regular) | 短 span，类 T5 | 3 | 15% | `[R]` |
| **S-denoising** (Sequential) | prefix-LM 样式（mask 序列尾部） | n/a | n/a（mask 后半部分） | `[S]` |
| **X-denoising** (eXtreme) | 长 span 或大比例 mask | 32 / 64 | 50% | `[X]` |

训练时**采样不同 denoising 任务混合训**，每个 sample 前面拼对应 mode token（`[R]` / `[S]` / `[X]`），inference 时用对应 mode token 触发。

**实证收益**：UL2-20B 在 zero-shot / few-shot 上同时超越同 size 的 T5（纯 span corruption）和 GPT（纯 CLM）——证明多目标训练**确实**比单目标更通用。PaLM-2 / Gemini 据传都受 UL2 影响（细节未公开）。

**为什么工业上没大规模采用**：

1. UL2 收益是 "few percent"，但实现复杂度（多种数据格式 + mode token + 比例调节）显著上升
2. CLM 已经"够好"且 scaling 到 70B+ 仍稳定，工业上更倾向"简单目标 + 大数据 + 大模型"
3. 多目标的最优配比需要长时间 ablation（X / R / S 各占多少比例），调参成本高
4. CLM + 后续 SFT / RLHF 可以覆盖大部分能力，多目标预训练的边际收益有限

### 2.7 训练目标对比表（必背）

| 目标 | 架构 | 每 step 贡献 loss 比例 | 代表模型 | 现代地位 |
|---|---|---|---|---|
| **CLM** | decoder-only | 100% | GPT 系 / LLaMA / Qwen / DeepSeek | **绝对主流** |
| **MLM** | encoder-only | 15% | BERT / RoBERTa / DeBERTa / ModernBERT | embedding / 分类 |
| **Prefix-LM** | decoder-only（混合 mask） | 后缀 100%、前缀 0% | GLM / UL2 (S-denoising) | 小众但 GLM 仍坚持 |
| **FIM** | decoder-only（CLM 变体） | 100%（middle 段） | CodeLlama / StarCoder / DeepSeek-Coder | **code model 标配辅助目标** |
| **Span corruption** | encoder-decoder | ~15%（取决于 mask） | T5 / FLAN-T5 / mT5 | seq2seq 任务 baseline |
| **Mixture-of-denoisers** | encoder-decoder 或 decoder-only | 多目标混合 | UL2 / PaLM-2（部分） | research 热点，工业未广泛采用 |

### 2.8 现代 LLM 训练目标的趋势

一句话总结当前格局：

- **CLM 主导**——LLaMA / Qwen / DeepSeek / Mistral / GPT-4 / Claude / Gemini 全部 decoder-only + CLM；in-context learning + scale 友好 + infra 简单的"三位一体"让其他目标几乎被淘汰
- **FIM 是 code model 标配辅助目标**——CodeLlama / StarCoder / DeepSeek-Coder / CodeQwen / Qwen2.5-Coder 全用 FIM；50% FIM rate + SPM 顺序是 de facto 标准
- **MLM 退守 embedding / 分类**——BERT 系仍是 MTEB 检索 benchmark 的主力，详见 16.4
- **Span corruption 退守翻译 / 摘要**——FLAN-T5 在 low-resource 翻译仍优于同 size LLM，但新旗舰模型不再用
- **多目标 (UL2) 是研究热点**——PaLM-2 / Gemini 据信受影响，但开源社区基本没人复现，调参 ROI 太低

可以预期**未来 3-5 年 CLM + FIM 仍是 pretraining 默认选择**。"是否要回到多目标"的问题或许在 long-context / multi-modal 场景重新被讨论（比如 vision token 是否用 prefix-LM、长文档是否用 span corruption），但纯文本 LLM 上 CLM 的统治地位短期不会改变。

---

## 3. 最小代码示例

### 3.1 CLM loss 实现（input shift + cross_entropy）

```python
import torch
import torch.nn.functional as F

def clm_loss(logits, input_ids, ignore_index=-100):
    """
    logits:    (B, T, V)  模型 forward 输出（未 softmax）
    input_ids: (B, T)     原始 token ids
    """
    # 关键：shift——logits 第 t 位预测 input_ids 第 t+1 位
    shift_logits = logits[:, :-1, :].contiguous()         # (B, T-1, V)
    shift_labels = input_ids[:, 1:].contiguous()          # (B, T-1)
    # 把最后一个位置的 label 设 ignore（它没有"下一个 token"可预测）
    # ↑ 这里通过 [:, 1:] 已经天然丢掉，不用额外处理
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),     # (B*(T-1), V)
        shift_labels.view(-1),                            # (B*(T-1),)
        ignore_index=ignore_index,                        # padding 位置不算
    )
    return loss
```

注意**所有非 padding 位置都贡献 loss**——这是 CLM 信号密度 100% 的体现。HuggingFace `LlamaForCausalLM.forward` 的内部实现就是这 8 行的等价物。

### 3.2 MLM masking 实现（80/10/10 trick）

```python
import torch

def mlm_mask(input_ids, vocab_size, mask_token_id, pad_token_id,
             mask_prob=0.15):
    """
    input_ids: (B, T)  原始 token ids
    返回 masked_input, labels (loss 只在被 mask 位置算，其他位置 -100)
    """
    labels = input_ids.clone()
    # 1. 随机选 15% 的位置做 mask 候选（排除 padding）
    rand = torch.rand(input_ids.shape, device=input_ids.device)
    mask_candidates = (rand < mask_prob) & (input_ids != pad_token_id)

    # 2. 非 mask 位置的 label 设 -100（CE ignore）
    labels[~mask_candidates] = -100

    # 3. 80/10/10 trick：在被选中的位置上
    rand2 = torch.rand(input_ids.shape, device=input_ids.device)
    masked_input = input_ids.clone()
    # 80% → 替换为 [MASK]
    mask_token = mask_candidates & (rand2 < 0.8)
    masked_input[mask_token] = mask_token_id
    # 10% → 替换为随机 token
    random_token = mask_candidates & (rand2 >= 0.8) & (rand2 < 0.9)
    random_ids = torch.randint(0, vocab_size, input_ids.shape,
                               device=input_ids.device)
    masked_input[random_token] = random_ids[random_token]
    # 剩下 10% → 保持原 token 不变（masked_input 已 = input_ids）

    return masked_input, labels

# loss 计算（forward 之后）
# loss = F.cross_entropy(logits.view(-1, V), labels.view(-1), ignore_index=-100)
# labels 中 ignore_index=-100 的位置不算，自然只有 ~15% 的 token 贡献 loss
```

读这段代码可以更直观地体会"为什么 MLM 信号密度只有 15%"——`labels` 中 85% 的位置都是 `-100`，CE loss 直接跳过。

### 3.3 FIM transform 实现（PSM / SPM 重排）

```python
import torch
import random

def fim_transform(token_ids, pre_id, suf_id, mid_id,
                  fim_rate=0.5, psm_rate=0.5):
    """
    token_ids: list[int]  原始样本（已 tokenize）
    pre_id, suf_id, mid_id: <PRE>, <SUF>, <MID> 三个 sentinel token id
    fim_rate: 多大比例 sample 做 FIM 重排，剩下保持普通 CLM
    psm_rate: 在 FIM sample 中，多大比例用 PSM（剩下用 SPM）
    """
    if random.random() > fim_rate:
        return token_ids                                      # 普通 CLM 样本

    # 随机切两个分点 → prefix / middle / suffix（middle 至少 1 token）
    n = len(token_ids)
    if n < 4:
        return token_ids                                      # 太短不切
    p1, p2 = sorted(random.sample(range(1, n), 2))
    prefix = token_ids[:p1]
    middle = token_ids[p1:p2]
    suffix = token_ids[p2:]

    if random.random() < psm_rate:
        # PSM: <PRE> prefix <SUF> suffix <MID> middle
        return [pre_id] + prefix + [suf_id] + suffix + [mid_id] + middle
    else:
        # SPM: <PRE> <SUF> suffix <MID> prefix middle
        return [pre_id, suf_id] + suffix + [mid_id] + prefix + middle

# 训练时：fim_transform 输出的序列直接用 CLM loss 训
# Inference 时：用户给 prefix + suffix，模型续写 middle
#   PSM 输入: <PRE> prefix <SUF> suffix <MID>      → generate until <eos>
#   SPM 输入: <PRE> <SUF> suffix <MID> prefix      → generate until <eos>
```

5 行核心：随机两切分 + sentinel 拼接重排。`<PRE>` / `<SUF>` / `<MID>` 必须**预先加到 tokenizer special tokens** 并扩展 embedding——否则模型见不到这三个 token，FIM 就无效。

### 3.4 Prefix-LM mask 构造

```python
import torch

def prefix_lm_mask(seq_len, prefix_len):
    """
    返回 attention mask, shape (T, T), 1 表示可见、0 表示禁止
    前 prefix_len 位置 bi-attention（互相可见，不能看后缀）
    后 (T - prefix_len) 位置 causal（看所有前缀 + 自己之前的后缀）
    """
    mask = torch.zeros(seq_len, seq_len)
    # 后缀部分：标准下三角
    causal = torch.tril(torch.ones(seq_len, seq_len))
    mask = causal.clone()
    # 前缀 query（行 0..p-1）：能看所有前缀（列 0..p-1），但不能看后缀
    mask[:prefix_len, :prefix_len] = 1
    mask[:prefix_len, prefix_len:] = 0                    # 显式禁止看后缀
    # 后缀 query：原 causal 已正确（能看所有前缀 + 自己之前的后缀）
    return mask

print(prefix_lm_mask(5, 3).int())
# tensor([[1, 1, 1, 0, 0],
#         [1, 1, 1, 0, 0],
#         [1, 1, 1, 0, 0],
#         [1, 1, 1, 1, 0],
#         [1, 1, 1, 1, 1]])

# attention 应用：scores.masked_fill(mask == 0, float('-inf')).softmax(-1)
```

注意第 5、7 行——前缀 query 必须**显式禁止看后缀**。一个常见 bug 是只对前缀做 `mask[:p, :p] = 1` 而不清零 `mask[:p, p:]`，结果前缀偷看了未来，破坏因果结构（虽然 loss 不在前缀算，但前缀的 hidden state 会污染后缀的 attention）。

---

## 4. 工程踩坑与经验

- ❗ **CLM 的 target = input shift right 一位，最后一个位置可以 ignore_index 掉**。实现上有两种等价写法：(1) `target = input[1:]`、forward 用 `input[:-1]`；(2) forward 用全 input，loss 时取 `logits[:, :-1]` 对 `input[:, 1:]`。HF `LlamaForCausalLM` 是第二种——它接收完整 `input_ids` 作为 labels，内部自动 shift，所以 **labels 应该传完整序列**，不要自己提前 shift（双 shift 是新手 #1 bug，会让 loss 完全错位但数值看起来还像样）。

- ❗ **MLM 训练时一定要有 80/10/10 random replacement，不能全 100% mask**。如果只用 `[MASK]`，模型会学到"看到 `[MASK]` 才工作，看到正常 token 就 copy"——下游 fine-tune（分类 / NER）阶段输入里不会有 `[MASK]`，所有位置都是正常 token，模型直接退化到 random embedding。RoBERTa / DeBERTa / ModernBERT 全部沿用 80/10/10。**自己实现 MLM 时如果省略 random + keep 这两步，结果会非常难看**——见过不止一个开源项目踩这个坑。

- ❗ **FIM 训 code model 时，sentinel token (`<PRE>` / `<SUF>` / `<MID>`) 必须加到 tokenizer special token 并扩展 embedding 矩阵**——否则它们会被切成"<", "PRE", ">"等若干 sub-token，模型完全学不到 FIM 语义。同时**训完后 chat template / IDE 集成层也要支持这些 token**：用户在 IDE 用 fill-in-middle 时，IDE 必须按 `<PRE> prefix <SUF> suffix <MID>` 的格式拼 prompt 给模型，不能直接喂"prefix + suffix"。CodeLlama / StarCoder / DeepSeek-Coder 的 README 里都明确写了这个 prompt 格式，部署集成时务必对齐。

- ❗ **Span corruption 实现时 mean span length 经验是 3-5、ratio 15%**。短了（mean=1）等价于 MLM，浪费 sentinel 开销；长了（mean>10）模型学不会"用一个 sentinel 解码很长一段"，loss 不收敛。Raffel 2020 系统 ablation 过：mean span 3 + ratio 15% 是 sweet spot；T5.1.1 / mT5 都用这个配置。**自己复现 T5 时直接抄这两个数**，不要乱试。

- ❗ **Prefix-LM 的 mask 实现是 per-sample 的（不同 sample 切分点不同），FlashAttention 不直接支持**。FlashAttention 优化的是"全 causal" / "全 bi"两种规则 mask；prefix mask 是混合的，需要 fallback 到普通 SDPA 或自定义 kernel（如 FA varlen + 自定义 mask），训练 throughput 比 pure decoder-only 慢 ~10-20%。这是 GLM 系训练成本相对较高的工程原因之一。**如果你的训练栈不支持 prefix mask，就只能选 pure CLM**，不要硬上 prefix-LM。

- ❗ **FIM 的 SPM (Suffix-Prefix-Middle) 比 PSM 在 inference 时 KV cache 更友好**。SPM 把 prefix（用户实际编辑的代码上文）紧挨在 `<MID>` 之前，inference 时 KV cache 可以这样组织：`<PRE><SUF>` + suffix（这部分用户 IDE 改动概率小）作为 cache prefix；prefix（光标前的代码，用户在打字）作为增量。Suffix 的 cache 命中率高，复用率好。DeepSeek-Coder 默认 SPM 就是为了 production code completion 的 latency。**如果只关心训练阶段，PSM / SPM 性能差异不大**（< 1 个点），生产部署时 SPM 略优。

- ❗ **FIM rate 不要拉到 100%，会损害普通续写能力**。Bavarian 2022 的 ablation 显示 FIM rate=50% 时 FIM 任务和 CLM 任务都达到 sweet spot；rate > 70% 时 left-to-right generation 性能下降明显（因为模型见过太多"乱序"样本，对自然左到右续写的概率分布拟合变差）。**实际工程默认 50%**，不要凭直觉设 80% 或 100%。

- ❗ **FIM rate 改变后 perplexity 不可比**。CLM 训练时的 loss / ppl 反映"对自然顺序文本的拟合度"；加了 FIM 之后 loss 还包含"重排序列的拟合度"，二者数值不可直接对比。评估 FIM 模型的真实续写能力要单独跑一份 pure CLM eval set，不要直接看训练 loss——很多人加 FIM 后看 ppl 升了误以为模型变差，其实是 ppl 的语义变了。

---

## 5. 经典 paper

- **Devlin et al., 2018 — BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding** — MLM 起点。读 §3.1 "Task #1: Masked LM" 的 80/10/10 trick 推导动机，那是 MLM 的核心工程经验。本节 §2.2 的所有结论都建立在这里。
- **Radford et al., 2018/2019 — Improving Language Understanding by Generative Pre-Training (GPT-1) / Language Models are Unsupervised Multitask Learners (GPT-2)** — CLM 路线源头。GPT-2 paper 把"所有 NLP 任务都是 next-token prediction"哲学正式确立。**为什么必读**：理解 CLM 为什么从工程目标变成了 LLM 的"哲学基础"。
- **Raffel et al., 2020 — T5: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer** — span corruption 的提出与系统 ablation。读 §3.3 "Unsupervised Objectives" 和 Table 7 的目标对比——它系统比较了 MLM / prefix-LM / span corruption / 不同 ratio / 不同 mean length 的所有组合，是预训练目标设计的经典实证文献。
- **Bavarian et al., 2022 — Efficient Training of Language Models to Fill in the Middle (FIM)** — FIM 的提出 paper（OpenAI Codex 团队作品）。读 §3 "FIM Training" 的 PSM / SPM 顺序设计 + §4 ablation。**核心结论**："FIM-for-free property"——加 FIM 几乎不损害 CLM 能力，但解锁了 fill-in-middle 能力。这是为什么所有 code model 都加 FIM 的根本依据。
- **Tay et al., 2022 — UL2: Unifying Language Learning Paradigms** — 多目标训练的代表作。读 §3 "Mixture-of-Denoisers" 和 §5 实验。**为什么读**：UL2 是过去几年最有影响力的"混合训练目标"研究，PaLM-2 / Gemini 据信受其影响；理解为什么"多目标比单目标好"的实证逻辑，以及为什么这套方法在工业上没被大规模采用。
- **Du et al., 2022 — GLM: General Language Model Pretraining with Autoregressive Blank Infilling** — Prefix-LM 的现代实现代表（清华 GLM-130B / GLM-4 系列）。读 §2 "GLM Pretraining" 的 mask 设计 + §3 实验。GLM 把 prefix-LM 与 span corruption 合并成一种"autoregressive blank infilling"形式，在中文 NLP 场景影响深远。

---

## 6. 自测与面试题

**Q1（公式）**：写出 CLM 与 MLM 的 loss 公式，说明为什么 CLM 训练效率比 MLM 高。

<details>
<summary>Answer sketch</summary>

公式：

$$\mathcal{L}_{\text{CLM}} = -\sum_{t=1}^{T-1} \log P_\theta(x_{t+1} \mid x_{\le t})$$

$$\mathcal{L}_{\text{MLM}} = -\sum_{t \in \mathcal{M}} \log P_\theta(x_t \mid \tilde{\mathbf{x}}), \quad |\mathcal{M}| / T \approx 0.15$$

效率差异的根源：

- **CLM 每个位置都贡献 loss**（除最后一位），信号密度 ≈ 100%
- **MLM 只在 15% 被 mask 的位置算 loss**，信号密度 ≈ 15%
- 同 compute / 同 data 下，CLM 拿到的梯度信号约是 MLM 的 6.7×
- 这是为什么 LLM scaling 主要走 CLM 路线、MLM 在大模型上没投入

加分点：
- 能指出 chain rule：$\log P(\mathbf{x}) = \sum_t \log P(x_{t+1} | x_{\le t})$，CLM 是干净的 MLE
- 能指出 MLM 信号低不代表 representation 差——MLM 训出的句向量在 NLU benchmark 上仍优于同 size CLM
- 能指出 mask ratio 提到 30% 也救不了 MLM（上下文不够）

</details>

**Q2（实现）**：写出 FIM transform 的核心 5 行 Python：随机切 prefix/middle/suffix 并重排（PSM）。

<details>
<summary>Answer sketch</summary>

核心 5 行（不含 import）：

```python
n = len(token_ids)
p1, p2 = sorted(random.sample(range(1, n), 2))   # 随机两切分点
prefix, middle, suffix = token_ids[:p1], token_ids[p1:p2], token_ids[p2:]
new_seq = [PRE] + prefix + [SUF] + suffix + [MID] + middle  # PSM 顺序
# 训练时直接对 new_seq 做 CLM loss
```

要点：

- **不改 loss、不改架构**——只改数据顺序，仍然是 CLM next-token loss
- **`<PRE>` / `<SUF>` / `<MID>` 必须是 special token**，加到 tokenizer 并扩展 embedding 矩阵
- **训练时只对部分 sample 做 FIM 重排**（FIM rate 50%），其余保持普通 CLM
- **SPM 顺序**：`[PRE] [SUF] suffix [MID] prefix middle`，inference 时 KV cache 复用更友好
- **Inference 时**：把用户的 prefix + suffix 拼成 `[PRE] prefix [SUF] suffix [MID]`，让模型续写到 `<eos>`

加分点：
- 能指出 FIM rate 不要 > 70%，否则普通续写能力下降
- 能指出 Bavarian 2022 的 "FIM-for-free property"
- 能说出 DeepSeek-Coder / CodeLlama / StarCoder 都用 FIM

</details>

**Q3（trade-off）**：为什么现代 LLM 几乎全用 CLM？至少 3 个原因。

<details>
<summary>Answer sketch</summary>

按重要性排序至少答到 3 个：

1. **In-context learning 友好**：CLM 训练时输入序列就是"prompt + answer 拼一起"，inference 时 few-shot examples + new query 也是同一序列，无 train-inference gap；MLM 训出来的模型完全没有"自回归生成"的训练信号，做生成任务必须 hack；encoder-decoder 在"few-shot 上下文该塞 encoder 还是 decoder"上没有自然答案
2. **训练 sample efficiency**：CLM 每个 token 都贡献 loss（信号密度 100%），MLM 只 15%；同 compute 下 CLM 拿到约 6.7× 的梯度信号——LLM scaling 阶段这个差异被放大到不可忽视
3. **任务统一性 / Prompt 范式**：所有 NLP 任务可以转换成 prompt → text generation 一个接口，对应 GPT-2 paper 的 "language models are unsupervised multitask learners" 哲学；MLM 必须每个任务挂一个 task head（分类头、NER 头），与 LLM 的 universal interface 哲学冲突
4. **Infra 简单**：单 stack、KV cache 简单、推理引擎（vLLM / SGLang）原生支持，部署成本低；prefix-LM 与 encoder-decoder 的 infra 复杂度都更高
5. **Wang 2022 实证**：在公平 compute / 参数 / data 下，zero-shot generative 任务上 decoder-only + CLM > encoder-decoder + span corruption > encoder-only + MLM

加分点：
- 能区分"CLM 赢的范围"——是 LLM-as-generator 范式，embedding / 检索任务 encoder-only + MLM 仍占优
- 能指出 code model 在 CLM 基础上加 FIM 是当代标配
- 能指出 UL2 / mixture-of-denoisers 是 research 方向但工业未广泛采用

</details>

---

## 7. 延伸阅读

- [Bavarian et al. 2022 — Efficient Training of Language Models to Fill in the Middle (arXiv)](https://arxiv.org/abs/2207.14255) — FIM 提出 paper，PSM / SPM 设计 + ablation 全在里面
- [Tay et al. 2022 — UL2 (arXiv)](https://arxiv.org/abs/2205.05131) — 混合训练目标的代表，理解 R / S / X denoising 的设计
- [Du et al. 2022 — GLM (arXiv)](https://arxiv.org/abs/2103.10360) — 现代 prefix-LM 的代表实现
- [Raffel et al. 2020 — T5 (arXiv)](https://arxiv.org/abs/1910.10683) — 训练目标系统 ablation 的经典 paper，Table 7 必看
- [HuggingFace `transformers` `LlamaForCausalLM.forward` 源码](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) — 看 CLM loss 的工业级实现，shift labels 的标准写法在那里
- [DeepSeek-Coder paper § FIM training](https://arxiv.org/abs/2401.14196) — 工业 code model 的 FIM 实操细节，SPM 顺序 + 50% rate 的工程选择有详细说明
- [Karpathy nanoGPT — `train.py` 的 loss 计算](https://github.com/karpathy/nanoGPT/blob/master/model.py) — CLM loss 的最简实现，10 行就能看明白
- 推荐继续读本教程的 **6.2 节《数据管线：FineWeb / Dolma / DCLM / 清洗去重配比》**——训练目标定了之后，就该谈"喂什么数据"
- 推荐继续读本教程的 **6.3 节《Scaling Law》**——CLM 在 scaling 上的优势在这一节量化呈现
