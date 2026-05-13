---
title: "5.1 Encoder-only / Decoder-only / Encoder-Decoder 对比"
description: "Transformer 一开始是 encoder-decoder（Vaswani 2017 做翻译），后来分裂成三条路线——encoder-only（BERT 系）做\"理解\"、decoder-only（GPT 系）做\"生成\"、encoder-decoder（T5 系）做\"序列到序列\"；2020 年后 decoder-only 几乎垄断 LLM，但 encoder-only 在 embedding "
---

> ⏱ 预计阅读 35 分钟 ｜ 难度 ★★ ｜ 前置：Module 4 整章（特别是 4.1 self-attention、4.2 multi-head、4.6 完整 decoder-only）

## 一句话本节讲什么

Transformer 一开始是 encoder-decoder（Vaswani 2017 做翻译），后来分裂成三条路线——**encoder-only（BERT 系）做"理解"、decoder-only（GPT 系）做"生成"、encoder-decoder（T5 系）做"序列到序列"**；2020 年后 decoder-only 几乎垄断 LLM，但 encoder-only 在 embedding / 分类场景以 ModernBERT 重新登顶，encoder-decoder 在翻译 / 摘要等纯 seq2seq 任务仍有竞争力——本节把三种架构的注意力 mask、训练目标、典型模型、现代用途讲清，并解释"为什么 decoder-only 赢了"以及"为什么 encoder-only 没死"。

---

## 1. Mental model（直觉）

### 1.1 一张表先建立鸟瞰

| 架构 | 注意力 mask | 训练目标 | 代表模型 | 现代用途 |
|---|---|---|---|---|
| **Encoder-only** | bidirectional（任意位置看任意位置） | MLM（masked language modeling） | BERT, RoBERTa, DeBERTa, ModernBERT | embedding / 分类 / NER / 检索 |
| **Decoder-only** | causal（每个位置只看历史） | CLM（causal LM，next-token 预测） | GPT 系 / LLaMA / Qwen / Mistral / DeepSeek / Claude / Gemini | 所有生成任务，**当代 LLM 主流** |
| **Encoder-Decoder** | encoder bi + decoder causal + cross-attention | span corruption / prefix-LM | T5, FLAN-T5, BART, mBART | 翻译 / 摘要 / 传统 seq2seq |

注意：**这张表的"架构"维度其实是两件事的耦合**——(1) 用几个 stack（一个还是两个）；(2) 每个 stack 用什么 mask。把它们拆开来看，就能理解后面 prefix-LM 这种"杂交"为什么也成立。

### 1.2 三种架构的数据流图（必背 ASCII）

**Encoder-only（BERT）**：

```
input:  [CLS] the cat [MASK] on the mat [SEP]
                        │
            ┌───────────▼───────────┐
            │   N × Encoder Block    │  ← bidirectional self-attention
            │   (每个位置 attend 到所有位置) │
            └───────────┬───────────┘
                        ▼
output: [h_CLS, h_the, h_cat, h_MASK, h_on, h_the, h_mat, h_SEP]
                              │
                              ▼
                       预测 [MASK] 位置 = "sat"
                       （只对 masked 位置算 loss）
```

**Decoder-only（GPT / LLaMA）**：

```
input:  the cat sat on the
              │
   ┌──────────▼──────────┐
   │   N × Decoder Block  │  ← causal self-attention
   │  (位置 t 只 attend 到 0..t) │
   └──────────┬──────────┘
              ▼
output: 每个位置预测下一 token
        the→cat | cat→sat | sat→on | on→the | the→mat
        （所有位置都算 loss）
```

**Encoder-Decoder（T5）**：

```
encoder input:  translate English to French: the cat sat on the mat
                          │
                ┌─────────▼─────────┐
                │   N × Encoder Block │  ← bidirectional
                └─────────┬─────────┘
                          ▼
                  encoded memory: (B, T_src, d)
                          │
                          │  cross-attention (decoder Q × encoder K, V)
                          ▼
decoder input: <bos> le chat
                ┌─────────▼─────────┐
                │   N × Decoder Block │  ← causal self-attn + cross-attn
                └─────────┬─────────┘
                          ▼
              decoder 自回归生成: le chat s'est assis ...
```

**核心区别一句话**：encoder-only 是"读完整句话再输出每个位置的表示"；decoder-only 是"边读边写，下一个 token 只能依赖前面"；encoder-decoder 是"先用 encoder 把 source 读完得到 memory，decoder 再边读 memory 边自回归生成 target"。

### 1.3 为什么"mask 决定一切"

三种架构的本质区别不在层数或参数，而在**self-attention 的 mask 形状**——这决定了"信息能往哪个方向流动"。

```
Encoder-only mask         Decoder-only mask         Prefix-LM mask
(全 1，bidirectional)      (下三角，causal)           (前缀 bi + 后缀 causal)

  1 1 1 1 1                 1 0 0 0 0                 1 1 1 0 0
  1 1 1 1 1                 1 1 0 0 0                 1 1 1 0 0
  1 1 1 1 1                 1 1 1 0 0                 1 1 1 0 0
  1 1 1 1 1                 1 1 1 1 0                 1 1 1 1 0
  1 1 1 1 1                 1 1 1 1 1                 1 1 1 1 1
```

Encoder-decoder 看作**两个 stack 各自有 mask**：encoder 全 1，decoder 是下三角；中间 cross-attention 让 decoder 的每个位置能看到 encoder 的所有位置（即 cross-attn 的 mask 也是全 1）。

记住这张图——后面所有架构选择（包括 GLM / UL2 的 prefix-LM、Chameleon 这种 mixed-modal）都是在 mask 上做文章。

### 1.4 一个易混淆的术语澄清：encoder-only 真的"不能生成"吗？

严格说**能**——你可以让 BERT 在 [MASK] 位置一个一个填，做一种很慢的 iterative generation（这就是 BERT-style "MLM as generator" 的思路，CMLM、Mask-Predict 等 paper）。但**为什么没人这么用**：

1. BERT 没有专门的 lm_head（输出层尺寸是 hidden、不是 vocab），需要额外训练
2. iterative MLM 的 inference 步数远多于 autoregressive，速度奇慢
3. 任意顺序生成的 likelihood 是 ill-defined 的，没有 chain rule 那么干净

所以"encoder-only 不能生成"的更准确说法是**"encoder-only 做生成不实用"**——这是工程现实，不是数学不可能。

---

## 2. 公式与原理

### 2.1 三种训练目标的形式化

记输入序列 $\mathbf{x} = [x_1, x_2, \dots, x_T]$，隐状态 $h_i \in \mathbb{R}^d$。

**MLM（Masked Language Modeling，BERT）**：

随机选 15% 的 token 位置集合 $\mathcal{M} \subset \{1, \dots, T\}$，把它们替换成 `[MASK]`（其中 80% 真换 mask、10% 换随机 token、10% 不变——BERT 原 trick），目标是预测被 mask 的原始 token：

$$\mathcal{L}_{\text{MLM}} = -\sum_{i \in \mathcal{M}} \log p(x_i \mid \mathbf{x}_{\setminus \mathcal{M}})$$

每个 mask 位置可以**双向**看到上下文 $\mathbf{x}_{\setminus \mathcal{M}}$（因为 attention 是 bidirectional），但 loss **只在 15% 的位置上算**。

**CLM（Causal LM，GPT / LLaMA）**：

每个位置预测下一个 token，对**所有**位置算 loss：

$$\mathcal{L}_{\text{CLM}} = -\sum_{t=1}^{T-1} \log p(x_{t+1} \mid x_1, \dots, x_t)$$

由 chain rule，这个目标等价于最大化 $\log p(\mathbf{x}) = \sum_t \log p(x_{t+1} \mid x_{\le t})$——是干净的概率分解。

**Span corruption（T5）**：

随机选若干 span（连续若干 token），用 `<extra_id_0>`、`<extra_id_1>` ... 这样的 sentinel 替换，让 encoder 接收带洞的序列，decoder 自回归生成所有被 mask 的 span（每段以对应 sentinel 开头）：

```
原文:    Thank you for inviting me to your party last week.
encoder: Thank you <X> me to your party <Y> week.
decoder: <X> for inviting <Y> last <Z>
```

形式化：设 corrupted 序列为 $\tilde{\mathbf{x}}$、被 mask 的 spans 拼接 + sentinel 装饰后为 $\mathbf{y}$，目标是

$$\mathcal{L}_{\text{span}} = -\sum_{t=1}^{|\mathbf{y}|} \log p(y_t \mid y_{<t}, \tilde{\mathbf{x}})$$

**Prefix-LM（GLM、UL2 的一种 objective）**：

把序列分成前缀 $\mathbf{x}_{\text{prefix}}$ 与后缀 $\mathbf{x}_{\text{suffix}}$，前缀部分 attention 是 bidirectional，后缀部分是 causal，loss 只在后缀上算：

$$\mathcal{L}_{\text{prefix-LM}} = -\sum_{t \in \text{suffix}} \log p(x_t \mid \mathbf{x}_{\text{prefix}}, x_{<t}^{\text{suffix}})$$

可以视为 "一个 stack 实现的 encoder-decoder"——前缀像 encoder（bi），后缀像 decoder（causal），中间不需要显式 cross-attn，因为整个序列共享一个 attention stack（mask 是混合的）。

### 2.2 训练效率：CLM 为什么 sample-efficient

**关键差异**：MLM 每个 sample 只在 15% 的 token 上算 loss；CLM 在 100% 的 token 上算 loss。

设序列长度 $T$、batch size $B$、训练总 token 数（forward 看到的）为 $N$：

| 目标 | 每 forward 产生的 loss 信号 | 信号密度 |
|---|---|---|
| MLM (15%) | $0.15 \cdot B \cdot T$ | 0.15 |
| CLM (100%) | $B \cdot (T - 1) \approx B \cdot T$ | ~1.0 |
| Span corruption (~15%) | $|\mathbf{y}| \approx 0.15 B T$ + sentinel | ~0.15 |

**含义**：在相同 compute / 相同 data 下，CLM 拿到的"梯度信号"约是 MLM 的 6.7×。这是 decoder-only 在**纯 LM scaling**下吃香的根本原因之一——同样的 GPU 时长，CLM 能把每个 token 的预测都拉来训。

但要打个补丁：MLM 的"信号密度低"是表象，**它在 representation learning 上是高质量的**——每个 mask 位置看到双向上下文，学到的 token 表示更适合下游分类 / 检索任务。这就是为什么 BERT 系在 GLUE / SuperGLUE 等 NLU benchmark 上至今难以被 decoder-only 同 size 模型超越（sentence embedding 任务尤其如此）。

### 2.3 为什么 decoder-only 赢了（核心论点）

这是面试高频题，必须把理由按重要性排序讲清。从 Wang et al. 2022《What LM Architecture and Pretraining Objective Work Best for Zero-Shot Generalization?》的实证 + 工程经验，至少有 4 个原因：

**(1) In-context learning 天然友好**：CLM 训练时输入序列就是"prompt + answer 拼一起"，inference 时 few-shot examples + new query 也是同一个序列，没有 train-inference gap。Encoder-decoder 必须把 prompt 塞进 encoder、answer 塞进 decoder，"few-shot 上下文该放哪个 stack"是个 ad hoc 设计；encoder-only 根本不能生成连续文本。

**(2) 训练效率（§2.2）**：每个 token 都贡献 loss，单位 compute 的信号密度高。

**(3) 任务统一性**：所有 NLP 任务都可以转换成"输入 prompt → 输出 text generation"——分类是输出标签词、QA 是输出答案、翻译是输出译文、code 是输出代码。这与 Module 0 § "5 分钟 demo" 的哲学一致——一个 LLM 一个 API 解决所有问题。Encoder-only 必须每个任务挂一个 task head（分类头、span head、token head）；encoder-decoder 也行但需要 task prefix。

**(4) Wang 2022 的实证**：在公平的 compute / data / 参数预算下，对 zero-shot 任务，decoder-only + CLM > encoder-decoder + span corruption > encoder-only + MLM。**但这个结论有适用范围**：

- zero-shot / few-shot generative 评测下，decoder-only 胜
- 大规模 multitask SFT 后，encoder-decoder + span corruption（adapted FLAN-T5）能反超 decoder-only
- 纯 representation 任务（句子向量、检索），encoder-only 仍是 SOTA（见 §2.4）

所以更准确的说法是**"decoder-only 在 zero-shot generative paradigm 下赢了，并因 scale 友好成为主流"**。

### 2.4 现代 encoder-only 的复兴：为什么 BERT 没死

LLM 时代 BERT 看似过气，但 2023-2024 后 encoder-only 在两个方向重新登顶：

**embedding 模型**：bge / E5 / GTE / NV-Embed / Stella 等当代 SOTA embedding 模型——本质都是 BERT 后裔（或从 LLM "encoderize"）。**为什么**：

- 句子向量需要"看完整句话再压成一个向量"——bidirectional attention 是天然适配
- decoder-only LLM 的 last-token output 不是好句向量（因为 last token 只看到自己之前的，前面的 token 表示无法被聚合）
- LLM2Vec、Echo Embedding 等技巧能把 LLM 转成 embedding model，但仍打不过同 size encoder-only on MTEB（2024-2025 时点）

**ModernBERT (2024, Warner et al.)**：把现代 LLM 的 trick 全套移植回 encoder-only：

- RoPE 取代 absolute position embedding
- FlashAttention（无 causal mask 的全 attention）
- GeGLU 取代 GeLU
- 8192 上下文（原 BERT 仅 512）
- 训练数据量与 LLaMA-1 同级

结果在 GLUE / 检索 / classification 上重新成为 BERT-base / large size 的 SOTA。**意义**：证明了 encoder-only 这条路线没死、只是过去 5 年没人投入新 trick；当 LLM 那套 RoPE / FlashAttn / GeGLU / 大数据训练补齐后，encoder-only 仍然是 representation 任务的最优解。详细见 16.4。

### 2.5 Encoder-Decoder 在哪里仍有用

虽然 LLM 时代 T5 系被 decoder-only 大量替代，但有几个场景仍是 encoder-decoder 占优：

- **传统翻译 / 摘要任务的强 baseline**：mBART、mT5 在 low-resource 语言对上仍优于同 size decoder-only LLM
- **多任务 SFT 训练**：FLAN-T5（11B）在很多 SFT 评测上能与 LLaMA-7B 同台竞争——span corruption + 大量 instruction tuning 是有效组合
- **code generation 的 fill-in-the-middle**：T5-style sentinel + bidirectional encoder 对 "given context before & after, fill middle" 任务天然适配（虽然 decoder-only 也能用 FIM training 做到）

但生成式 LLM 时代（2023+）大势已定：新出的旗舰模型基本都是 decoder-only，encoder-decoder 不再是主流投资方向。

### 2.6 Prefix-LM 的特殊地位

Prefix-LM 是个有意思的"杂交"——只用一个 stack（不分 encoder/decoder），但 mask 是混合的（前缀 bi、后缀 causal）。它的设计动机：

- **优点**：prompt 部分可以双向 attention（像 encoder 一样捕获完整语义），生成部分仍是 causal（保留 LM 性质 + KV cache 友好）。Wang 2022 实证 prefix-LM 在 zero-shot 略好于纯 decoder-only
- **代表模型**：GLM-130B / GLM-4（清华 / 智谱）、UL2（Google）、Mixed-modal models 内部常用 prefix-LM 处理 vision token
- **缺点**：mask 实现复杂（要在 batch 中标记每个 sample 的"prefix 切分点"），训练 infra 与纯 decoder-only 不兼容；KV cache 在 prefix 部分需要"全 prefill"+ 后缀部分增量缓存，工程实现也更繁琐
- **为什么没大流行**：实证收益不大（~1-2 点 zero-shot 提升），但 infra 复杂度大幅增加——大多数团队选择"纯 decoder-only + scale up"路线

GLM-4 至今仍是少数坚持 prefix-LM 的旗舰模型；UL2 的 mixture-of-denoisers（同时用 R-denoising / S-denoising / X-denoising）是更复杂的 hybrid，但也因实现复杂没成为主流。

### 2.7 现代 LLM 谁是哪种（必背速查表）

| 模型族 | 架构 | 典型用途 |
|---|---|---|
| GPT-2 / GPT-3 / GPT-4 / GPT-4o | decoder-only + CLM | LLM 主流 |
| LLaMA-1/2/3 / CodeLlama | decoder-only + CLM | 开源 LLM 主线 |
| Qwen-1.5/2/2.5/3 | decoder-only + CLM | 中文 LLM 标杆 |
| Mistral / Mixtral | decoder-only（Mixtral MoE） | 欧洲开源旗舰 |
| DeepSeek-V2/V3/R1 | decoder-only + MLA + MoE | 推理 / 性能旗舰 |
| Claude 3/3.5/4 / Gemini 1.5/2 | decoder-only（推断，非公开） | 商业旗舰 |
| BERT / RoBERTa / DeBERTa | encoder-only + MLM | 分类 / NER 标杆 |
| ModernBERT (2024) | encoder-only + MLM（现代化） | 重生的 encoder-only |
| bge / E5 / NV-Embed / Stella | encoder-only（基于 BERT 后裔）| embedding SOTA |
| T5 / FLAN-T5 / mT5 | encoder-decoder + span corruption | seq2seq / 翻译 |
| BART / mBART | encoder-decoder（BERT 编 + GPT 解） | 摘要 / 翻译 |
| GLM-4 | prefix-LM | 国产开源 |
| UL2 / mT5-XXL | mixture-of-denoisers / hybrid | research |

面试时被问"XXX 是什么架构"——按这张表答即可。**陷阱题**：DeepSeek-R1 ≠ 一种新架构，它是 DeepSeek-V3 base 上做 RL 得到的"reasoning model"，base 仍然是 decoder-only + MLA + MoE。

---

## 3. 最小代码示例

### 3.1 用 transformers 调三种架构对比（< 25 行）

```python
# 三种架构的 HuggingFace 调用模式对比
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import torch

# === 1. Encoder-only (BERT) - 拿 hidden state 做 embedding 或分类 ===
bert_tok = AutoTokenizer.from_pretrained("bert-base-uncased")
bert = AutoModel.from_pretrained("bert-base-uncased")
inp = bert_tok("The cat sat on the mat.", return_tensors="pt")
out = bert(**inp)                                  # out.last_hidden_state: (1, T, 768)
cls_emb = out.last_hidden_state[:, 0]              # [CLS] token → 句向量（1, 768）
mean_emb = out.last_hidden_state.mean(dim=1)       # mean pool → 句向量（更鲁棒）
print("BERT [CLS] emb shape:", cls_emb.shape)

# === 2. Decoder-only (LLaMA-style) - 自回归生成 ===
gpt_tok = AutoTokenizer.from_pretrained("gpt2")
gpt = AutoModelForCausalLM.from_pretrained("gpt2")
inp = gpt_tok("The capital of France is", return_tensors="pt")
out = gpt.generate(**inp, max_new_tokens=10, do_sample=False)
print("GPT generate:", gpt_tok.decode(out[0]))

# === 3. Encoder-Decoder (T5) - 必须加 task prefix ===
t5_tok = AutoTokenizer.from_pretrained("t5-small")
t5 = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
inp = t5_tok("translate English to French: The cat sat on the mat.",
             return_tensors="pt")                  # ← task prefix 是关键
out = t5.generate(**inp, max_new_tokens=20)
print("T5 translate:", t5_tok.decode(out[0], skip_special_tokens=True))
```

**关键差异**：

- `AutoModel`（不带 head）→ 拿 hidden state，BERT 类典型用法
- `AutoModelForCausalLM` → 带 lm_head，autoregressive `generate` 可用
- `AutoModelForSeq2SeqLM` → encoder-decoder，`generate` 内部会跑 encoder 一次 + decoder 自回归
- T5 的 input **必须**带 task prefix（"translate English to French: ..."、"summarize: ..."），不加几乎不工作——因为预训练时所有任务都是用 prefix 区分的

### 3.2 三种 mask 的对比构造（< 20 行）

```python
import torch

T = 6  # 序列长度

# === 1. Encoder-only mask: 全 1（任意位置 attend 任意位置） ===
encoder_mask = torch.ones(T, T)

# === 2. Decoder-only causal mask: 下三角 ===
causal_mask = torch.tril(torch.ones(T, T))

# === 3. Prefix-LM mask: 前 prefix_len 个位置 bi，剩余 causal ===
prefix_len = 3
prefix_mask = torch.tril(torch.ones(T, T))                 # 先全 causal
prefix_mask[:, :prefix_len] = 1                            # prefix 列全 1（任意 query 都能看到 prefix）

print("Encoder mask (BERT):\n", encoder_mask.int().numpy())
print("\nCausal mask (GPT):\n", causal_mask.int().numpy())
print("\nPrefix-LM mask (GLM, prefix_len=3):\n", prefix_mask.int().numpy())
# 在 attention 里用法：scores.masked_fill(mask == 0, -inf).softmax(-1)
```

**输出**（验证三种架构核心差异在 mask 形状）：

```
Encoder mask (BERT):           Causal mask (GPT):           Prefix-LM (GLM):
[[1 1 1 1 1 1]                 [[1 0 0 0 0 0]               [[1 1 1 0 0 0]
 [1 1 1 1 1 1]                  [1 1 0 0 0 0]                [1 1 1 0 0 0]
 [1 1 1 1 1 1]                  [1 1 1 0 0 0]                [1 1 1 0 0 0]
 [1 1 1 1 1 1]                  [1 1 1 1 0 0]                [1 1 1 1 0 0]
 [1 1 1 1 1 1]                  [1 1 1 1 1 0]                [1 1 1 1 1 0]
 [1 1 1 1 1 1]]                 [1 1 1 1 1 1]]               [1 1 1 1 1 1]]
```

---

## 4. 工程踩坑与经验

- ❗ **用 BERT 做 embedding，不要直接拿 `last_hidden_state` 不 pool**。BERT 输出是 `(B, T, d)`，必须显式 pool 成 `(B, d)`：常用方法是 (1) `[CLS]` token output（`out[:, 0]`，需要在 NSP / sentence-pair 任务上微调过的模型才好）；(2) **mean pooling**（`out.mean(dim=1)`，对未微调的 BERT 更鲁棒，是 SBERT / bge 的默认）；(3) max pooling（少用）。**最常见的初学者错误**是把 `last_hidden_state` 当 `(B, d)` 直接传到下游，shape 对不上；或者想做相似度计算时随便 `out[:, -1]`（拿最后一个 token），那只是 `[SEP]`，几乎没语义。详细见 16.4。

- ❗ **T5 / FLAN-T5 的 input 必须加 task prefix**——"translate English to German: ..."、"summarize: ..."、"question: ... context: ..."。不加 prefix 会得到完全不相关的输出。原因是 T5 预训练时所有 supervised 任务都是用 prefix 区分的，模型把 prefix 当成 task indicator。**坑场景**：用 `t5.generate(input_ids=tokenizer("hello").input_ids)` 期望它"翻译/续写"，结果输出 `<extra_id_0>` 或乱码——它不知道你要它做什么任务。FLAN-T5 因为做过大规模 instruction tuning 容错好一些，但 prefix 仍是最佳实践。

- ❗ **decoder-only LLM 做 sentence embedding 不如 encoder-only，即便有 LLM2Vec 等技巧**。直觉上"LLM 这么大，提个句向量肯定碾压 BERT"——错。LLM 的 last-token output 只看到了完整 prompt，但前面 token 的表示不能聚合（因为 causal mask）；mean pooling 又会被前几个 token 的"未见全文"拖累。LLM2Vec（去掉 causal mask 再 contrastive 微调）、Echo Embedding（把 input 复制两遍，用第二份的 last token）等 hack 能拉近差距，但**同 size 下 encoder-only 在 MTEB 上仍略胜**。production 用 embedding 还是选 bge / E5 / NV-Embed 这类 encoder-only。

- ❗ **prefix-LM 的 mask 实现复杂；GLM 系训练 infra 与纯 decoder-only 不兼容**。每个 batch sample 的 "prefix 长度" 不同，mask 是 per-sample 的；KV cache 在 prefix 阶段是 full bidirectional prefill、suffix 阶段是 incremental causal——两套行为耦合在一个 stack。vLLM / SGLang 等推理引擎对 GLM 的 prefix-LM 支持比 decoder-only 滞后很多。**结论**：选 prefix-LM 之前确认你的训练 + 推理栈支持。

- ❗ **把 BERT-base 当 LLM 用 → 没有 generation 能力**。`AutoModel.from_pretrained("bert-base-uncased")` 加载的是 backbone（无 lm_head + 无 causal mask），调 `.generate()` 会报错或退化。如果真要用 BERT 风格做生成（罕见），需要 `BertLMHeadModel` 配合 `is_decoder=True` 把 self-attention 改成 causal——但效果远不如 GPT。**正确做法**：要 generation 任务直接用 `AutoModelForCausalLM` + decoder-only LLM；BERT 留给 classification / NER / embedding。

- ❗ **`AutoModelForSeq2SeqLM` 的 `generate` 默认行为可能与你预期不符**：T5 的 decoder 起始 token 是 `<pad>`（不是 `<bos>`），如果手动构造 decoder_input_ids 容易错。最安全做法是只传 `input_ids`，让 `generate` 自动处理 decoder start token。另外 `max_length` 与 `max_new_tokens` 在 encoder-decoder 上含义不同——`max_length` 是 decoder 总长度，包含起始 token；建议用 `max_new_tokens` 避免歧义。

- ❗ **encoder-only 模型的 `position_embeddings` 默认 max_length = 512**（BERT / RoBERTa），超过会报 `IndexError: index out of range`。ModernBERT 把这扩到 8192，但老的 bert-base 仍是 512。**坑场景**：拿 BERT 做长文档检索 → 直接喂 1000 token 报错。处理方法：(1) chunking 后分段编码再 pool；(2) 用 ModernBERT / Longformer 等长上下文 encoder；(3) 截断到 512（信息损失但简单）。

---

## 5. 经典 paper

- **Devlin et al., 2018 — BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding** — encoder-only 路线起点。读 §3.1 MLM / NSP 训练目标 + §4 GLUE 实验。它定义了 "pre-train + fine-tune" 范式与 [CLS] / [SEP] token 约定，直到今天 embedding 模型的 token 化都沿用这套。**为什么必读**：理解 encoder-only 的所有现代变体（RoBERTa / DeBERTa / ModernBERT）都从这里出发。
- **Radford et al., 2018/2019 — GPT-1 / GPT-2 (Language Models are Unsupervised Multitask Learners)** — decoder-only 路线起点。GPT-2 paper 提出 "language models are unsupervised multitask learners"，奠定了"所有 NLP 任务都是 next-token prediction"的范式哲学。**为什么必读**：理解为什么 decoder-only 走向 LLM 主流——这篇是 in-context learning + task unification 的源头思想。
- **Raffel et al., 2020 — T5: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer** — encoder-decoder 路线的现代化。提出 span corruption + "everything is text-to-text"（与 GPT-2 哲学殊途同归）。读 §2 architecture comparison + §3.3 unsupervised objectives——它系统比较了 BERT-style MLM、prefix-LM、span corruption 等多种 objective，是本节 §2.1 训练目标对比的实证依据。
- **Wang et al., 2022 — What Language Model Architecture and Pretraining Objective Work Best for Zero-Shot Generalization?** — 三种架构 ablation 的权威 paper。在公平的 compute / data 预算下系统比较 encoder-only / decoder-only / encoder-decoder × MLM / CLM / span corruption。**核心结论**：zero-shot generative 任务下 decoder-only + CLM 胜出，但 multitask SFT 后 encoder-decoder + adapted span 反超。本节 §2.3 "decoder-only 为什么赢" 的硬实证就来自这篇。
- **Warner et al., 2024 — Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference (ModernBERT)** — 现代 encoder-only 复兴的标志性 paper。把 RoPE / FlashAttention / GeGLU / 8192 context / LLM 级数据量全部移植回 BERT 架构，重新成为同 size encoder 的 SOTA。**为什么必读**：证明 encoder-only 路线没死、未来仍有空间——也是 16.4 embedding 模型一节的前置背景。

---

## 6. 自测与面试题

**Q1（架构对比）**：列出 BERT / GPT / T5 三者在 (注意力 mask / 训练目标 / 推理模式) 三个维度的差异。

<details>
<summary>Answer sketch</summary>

| 维度 | BERT (encoder-only) | GPT (decoder-only) | T5 (encoder-decoder) |
|---|---|---|---|
| 注意力 mask | bidirectional（任意位置 attend 任意位置） | causal（下三角，位置 t 只看 0..t） | encoder bi + decoder causal + cross-attention |
| 训练目标 | MLM：随机 mask 15%，预测被 mask 的 token，loss 只在 mask 位置算 | CLM：next-token 预测，所有位置算 loss | span corruption：encoder 看带洞序列，decoder 自回归生成 mask 段（带 sentinel 装饰） |
| 推理模式 | 一次 forward 输出每个位置的表示；做生成需要 iterative MLM（罕见） | 自回归 sampling，逐 token 生成 | encoder 跑一次得到 memory，decoder 自回归 + cross-attention |

加分点：
- 能指出 MLM 信号密度只有 15%、CLM 是 100%——这是 CLM sample-efficient 的根因
- 能指出 BERT 严格说能做 iterative generation 但不实用
- 能指出 T5 的 task prefix 约定与 unified text-to-text 哲学

</details>

**Q2（trade-off）**：为什么 decoder-only 几乎垄断了 LLM？至少 3 个原因。

<details>
<summary>Answer sketch</summary>

按重要性排序至少答到 3 个：

1. **In-context learning 友好**：CLM 训练时输入序列就是 "prompt + answer 拼一起"，inference 时 few-shot examples + new query 也是同一序列，无 train-inference gap；encoder-decoder 在"few-shot 上下文该塞 encoder 还是 decoder"上没有自然答案
2. **训练 sample efficiency**：CLM 每个 token 都贡献 loss（信号密度 100%），MLM 只 15%、span corruption 也只 ~15%；同 compute 下 CLM 拿到约 6.7× 的梯度信号
3. **任务统一性**：所有 NLP 任务可以转换成 prompt → text generation 一个接口，对应 GPT-2 paper 的 "language models are unsupervised multitask learners" 哲学；encoder-only 必须每个任务挂 task head
4. **Wang 2022 实证**：在公平 compute / 参数 / data 下，zero-shot generative 任务上 decoder-only > encoder-decoder > encoder-only
5. **Infra 简单**：单 stack、KV cache 简单、推理引擎（vLLM / SGLang）原生支持，部署成本低；encoder-decoder 与 prefix-LM 的 infra 复杂度都更高

加分：能指出 Wang 2022 的边界——大规模 multitask SFT 后 encoder-decoder 能反超；能指出"赢"的范围是 LLM-as-generator 范式，embedding / 检索任务 encoder-only 仍占优。

</details>

**Q3（前沿）**：encoder-only 在 LLM 时代的最大用途是什么？为什么 ModernBERT 仍有意义？

<details>
<summary>Answer sketch</summary>

**最大用途**：embedding 模型（句子向量 / 检索 / 分类）。bge / E5 / GTE / NV-Embed 等 SOTA embedding 模型至今仍是 encoder-only 架构（BERT / DeBERTa 后裔）。原因：

- 句子向量需要"看完整句话再压成一个向量"——bidirectional attention 天然适配
- decoder-only LLM 的 last-token output 不是好句向量（前面 token 的表示因 causal mask 无法聚合）
- LLM2Vec / Echo Embedding 等 hack 能拉近差距，但同 size 下 encoder-only 在 MTEB 仍略胜

**ModernBERT 的意义**：

1. **证明 encoder-only 没死**——只是过去 5 年没人投入新 trick
2. **现代化**：把 LLM 那套 RoPE / FlashAttention / GeGLU / 8192 context / 大数据量训练全部回填，重新成为 BERT-base / large size 的 SOTA
3. **解锁长文档场景**：原 BERT 只 512 token，ModernBERT 8192，让长文档检索 / RAG embedding 有了 native 支持（不用 chunk）
4. **成本经济性**：分类 / 检索 / NER 这类任务用 ModernBERT-base（149M）就够，不需要 7B+ LLM——production cost 与 latency 优势明显

加分：能指出"未来 embedding 模型大概率仍是 encoder-only 路线，但会越来越多吸收 LLM 的训练 trick"；能引到 16.4 详讲 embedding 模型；能指出 BERT 之所以"过气感"是因为 LLM 时代关注度被吸走，但工业界 NER / 分类 / 检索仍大量在用 BERT 后裔。

</details>

---

## 7. 延伸阅读

- [Devlin et al. 2018 — BERT (arXiv)](https://arxiv.org/abs/1810.04805) — encoder-only 路线起点，MLM 目标的官方说明
- [Radford et al. 2019 — GPT-2 (paper PDF)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — decoder-only 哲学源头："LM are unsupervised multitask learners"
- [Raffel et al. 2020 — T5 (arXiv)](https://arxiv.org/abs/1910.10683) — encoder-decoder 现代化 + span corruption 的系统 ablation
- [Wang et al. 2022 — What LM Architecture and Pretraining Objective Work Best for Zero-Shot Generalization? (arXiv)](https://arxiv.org/abs/2204.05832) — 三种架构 ablation 的权威实证 paper
- [Warner et al. 2024 — ModernBERT (arXiv)](https://arxiv.org/abs/2412.13663) — encoder-only 路线复兴的标志
- [HuggingFace — Model Summary (encoder vs decoder vs encoder-decoder)](https://huggingface.co/learn/llm-course/chapter1/4) — HF 官方对三种架构的入门级说明
- [Sebastian Raschka — Understanding Encoder And Decoder LLMs](https://magazine.sebastianraschka.com/p/understanding-encoder-and-decoder) — blog 长文，配图清晰
- 推荐继续读本教程的 **5.2 节《GQA / MQA / MLA：KV cache 压缩》**——decoder-only 主流之后，attention 层的工业级演化
- 推荐继续读本教程的 **6.1 节《训练目标：CLM / MLM / Prefix-LM / FIM》**——把本节训练目标对比展开到预训练实战层面
- 推荐继续读本教程的 **16.4 节《Embedding：bge / E5 / Instructor / NV-Embed》**——encoder-only 在 LLM 时代的最大现代用途
