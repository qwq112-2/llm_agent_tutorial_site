---
title: "6.5 Long-context：Position Interpolation / NTK / YaRN / LongRoPE"
description: "直接把一个 4k context 训出来的 RoPE 模型拉到 32k / 128k 推理 → perplexity 暴涨、needle test 准确率从 95% 掉到 5%；本节系统讲清四种主流 RoPE 长 context 扩展方案——Position Interpolation（PI，把位置 m 缩到训过的范围）→ NTK-aware（调大 RoPE base 让低频维度变慢）→ YaRN"
---

> ⏱ 预计阅读 55 分钟 ｜ 难度 ★★★ 🔥 必考 ｜ 前置：4.3 RoPE

## 一句话本节讲什么

直接把一个 4k context 训出来的 RoPE 模型拉到 32k / 128k 推理 → perplexity 暴涨、needle test 准确率从 95% 掉到 5%；本节系统讲清四种主流 RoPE 长 context 扩展方案——**Position Interpolation（PI，把位置 m 缩到训过的范围）→ NTK-aware（调大 RoPE base 让低频维度变慢）→ YaRN（NTK-by-parts 分频段处理 + attention temperature，工业 SOTA）→ LongRoPE（用进化算法搜每维 rescale，能扩到 2M+）**——再给完整的 4 阶段 continued-pretraining pipeline、长 context 评测谱（Needle / RULER / LongBench），以及 long-context 与 KV cache、显存的纠缠。这是 2024+ 算法岗 long-context 题型的全部弹药。

---

## 1. Mental model（直觉）

### 1.1 为什么"短训长推"会塌：未训练区间 + 高频混叠

回到 4.3 §2.4：RoPE 把 q, k 的每对维度按位置 $m$ 旋转角度 $m \theta_i$，其中 $\theta_i = 10000^{-2i/d_k}$。训练时模型见过的位置区间是 $m \in [0, L_{\text{train}})$，对每个维度 $i$ 见过的旋转角度集合是

$$\mathcal{A}_i = \{m \theta_i : m = 0, 1, \dots, L_{\text{train}} - 1\}$$

直接拿到 $L_{\text{infer}} = 4 L_{\text{train}}$ 上推理 → 每个维度新增的旋转角 $m \theta_i$（$m \in [L_{\text{train}}, 4 L_{\text{train}})$）**完全没在训练时出现过**。模型学到的"$\cos(m\theta_i)$ 在某区间的分布"突然外推到 4× 范围，attention 行为退化。

更隐蔽地，**高频维度（$i$ 小、$\theta_i \approx 1$）问题更严重**：旋转角 $m \theta_i$ 在 $m = 32k$ 时已经绕了几千圈，相邻 token 的旋转角差异（$\theta_i$）被模型学成"局部信号"——但绕几千圈后这个信号在 32k 范围内已经被周期性重复无数次，相当于发生**频率混叠**（aliasing）。模型分不清"距离 1 的相邻 token"与"距离 1 + 周期 k 的远 token"。

实证后果：

```
LLaMA-2 7B (训练 4k)，直接拉到 32k 推理：
  perplexity:  6.4  →  >1000   （20× 退化）
  needle test 准确率:  95%  →  ~5%
```

直觉：**模型见过 0~4k 范围内的旋转角分布，超出就崩——必须把"位置→旋转角"的映射改造一下，让推理时的旋转角仍落在训练分布里。**

### 1.2 三种"改造"思路（一图流）

```
原始 RoPE 旋转角：   m θ_i,    θ_i = 10000^(-2i/d)
                     │
       ┌─────────────┼─────────────┐
       ▼             ▼             ▼
   ① 缩 m       ② 调 base       ③ 分频段
   (PI)         (NTK-aware)      (YaRN / LongRoPE)

   m → m/s      θ_i 整体变小      高频不缩、低频缩
   全频段缩      ↓                高频局部信号保留
   高频被挤压    高频几乎不变       低频用 PI 处理
   相邻 token   低频显著变慢       中频平滑过渡
   难区分        长程信号清晰       + attention temperature
```

- **思路 ① 缩 m（PI）**：最朴素——既然 32k 没训过、4k 训过，那就把推理时的 32k 位置"压回 4k"，每个位置都按 $m / 4$ 处理。简单粗暴但**所有维度被同等压缩，高频信号被压得相邻 token 几乎没差**。
- **思路 ② 调 base（NTK-aware）**：注意 $\theta_i = b^{-2i/d}$，base $b$ 越大，**低频维度（$i$ 大）旋转得越慢**——也就是低频维度的"等效波长"被拉宽，足以覆盖更长的位置范围；而高频维度（$i$ 小、$\theta_i \approx 1$）几乎不受 base 调大影响。**等价于"只拉伸长程信号、不动局部信号"**——这就是 NTK 比 PI 好的根本原因。
- **思路 ③ 分频段（YaRN / LongRoPE）**：直接告诉算法"高频组完全不动、低频组用 PI 缩、中间用平滑过渡"——把 ① 与 ② 的优点合体。LongRoPE 进一步用 EA 搜每个维度的最优缩放因子，扩到 2M+。

### 1.3 为什么"扩到 1M context"不等于"模型真的能用 1M"

工业现状必须有的清醒：

- **GPT-4o / Gemini 1.5 Pro / Claude 3.5 在 needle-in-a-haystack 上接近 100%**——但这只证明"能从 1M token 里 retrieve 一句明显的密钥"
- **真实 long-context 任务**（多 hop 推理、长代码理解、long-document QA）的有效 context 通常远小于"宣称 context"——一个 1M context 的模型在 32k 之后开始掉点、64k 之后多 hop 几乎崩
- 学术圈管这个叫 "**effective context < advertised context**"——RULER (NVIDIA 2024) 就是为了量化这个 gap 设计的

一句话：**你看到的 "context 1M / 2M" 是营销数字；真实可用 context 看 RULER 而不是 needle test**。

---

## 2. 公式与原理

### 2.1 Position Interpolation（PI，Chen 2023 / Meta）

**核心 idea**：把推理时的位置 $m$ 缩放回训练范围内，

$$m' = \frac{m}{s}, \quad s = \frac{L_{\text{new}}}{L_{\text{train}}}$$

旋转角随之变成

$$m' \theta_i = \frac{m}{s} \theta_i$$

其中 $L_{\text{train}}$ 是训练时 max context（如 4k），$L_{\text{new}}$ 是目标 context（如 32k），$s$ 称为 **scale factor**。$L_{\text{new}} = 32k, L_{\text{train}} = 4k \Rightarrow s = 8$。

**几何含义**：把"32k 个位置"线性压缩到训过的"4k 区间"内——位置 $m = 32000$ 在新模型里相当于位置 $m' = 4000$，attention 模式与训练时位置 4000 一致。

**优点**：
- 公式极简（一行除法）
- **PPL 不会塌**：因为所有旋转角仍落在训练分布里
- 与 RoPE precompute 完美兼容（只在生成 cos/sin 表时把 `t` 除以 $s$ 即可）

**缺点**：
- **高频信息被挤压**：原本相邻 token 的旋转角差是 $\theta_0 = 1$（弧度），缩放后变成 $1/8 \approx 0.125$ 弧度——相邻 token 的方向几乎没区分
- **必须 fine-tune**：直接 zero-shot inference 也掉点（虽然不像不缩那么塌）。Chen 2023 的实验里 fine-tune **几百 step**（约 1B token）就稳了
- 对所有维度同等缩放，不区分"高频应保留 / 低频可压缩"

**Chen 2023 一组关键数据**：LLaMA 7B（训练 2k context）→ PI 扩到 32k，fine-tune 1k step 后 perplexity 与原 2k 几乎一致；naive extrapolation 直接 PPL > 1000。

### 2.2 NTK-aware Scaling（bloc97，2023 reddit blog）

NTK-aware 的提出是个 reddit 草根工作，但被工业界迅速吸收。**核心 idea**：调大 RoPE 的 base $b$，

$$b = 10000 \to b' = b \cdot s^{d/(d-2)}$$

让旋转角变成 $\theta'_i = (b')^{-2i/d}$。

**为什么这样改有效**？回顾 $\theta_i = b^{-2i/d}$：

- 高频维度 $i = 0$：$\theta_0 = b^0 = 1$（与 $b$ 完全无关）
- 低频维度 $i = d/2 - 1$：$\theta_{d/2-1} = b^{-(d-2)/d}$（强依赖 $b$）

调大 $b$ 的效果是 **高频几乎不变、低频显著变慢**——等价于"只压缩长程信号、保留局部信号"。

**经验缩放公式推导**：希望最低频维度 $i = d/2 - 1$ 在新长度 $L_{\text{new}} = sL_{\text{train}}$ 上的旋转角与原模型在 $L_{\text{train}}$ 上的旋转角一致：

$$L_{\text{new}} \cdot (b')^{-(d-2)/d} = L_{\text{train}} \cdot b^{-(d-2)/d}$$

解出

$$b' = b \cdot \left(\frac{L_{\text{new}}}{L_{\text{train}}}\right)^{d/(d-2)} = b \cdot s^{d/(d-2)}$$

举例：$d = 128, s = 4 \Rightarrow b' = 10000 \cdot 4^{128/126} \approx 10000 \cdot 4.09 \approx 40900$。LLaMA-3 long-context 版本的 `rope_theta` 字段就是从 500000 起步——背后就是这个公式。

**优点**：
- 高频被保留 → 相邻 token 仍可区分 → 比 PI 在 zero-shot 直接外推强很多
- 调一行 config 即可

**缺点**：
- **仍然需要少量 fine-tune** 才能到位（zero-shot 比 PI 强但仍不如 fine-tune 后的 PI）
- **中频维度处理不够细**：高频几乎不变、低频显著调整，但中频的"过渡"是隐式发生的，可能不够平滑

### 2.3 YaRN — Yet another RoPE extensioN（Peng 2023）

YaRN 是当前 LLaMA-2/3 / Qwen / Mistral 的工业标准。它做了**两件事**：

#### 2.3.1 NTK-by-parts：分频段处理（核心）

YaRN 不像 NTK-aware 那样"全局调 base"，而是 **明确分频段**：

- **高频维度**（波长 $\lambda_i = 2\pi / \theta_i$ 远小于训练 max_seq）：完全不缩放（保留 local 信息）
- **低频维度**（波长 $\lambda_i$ 远大于训练 max_seq）：用 PI 缩放（缩到训练分布内）
- **中频维度**：用一个 **smooth 过渡函数** $\gamma(\lambda_i)$ 在 0 与 1 之间平滑切换

形式上，对维度 $i$ 定义一个 rescale factor $h(\lambda_i)$：

$$h(\lambda_i) = \begin{cases} 1 & \text{if } \lambda_i \le \alpha \cdot L_{\text{train}} \quad \text{（高频，不缩）} \\ \frac{1}{s} & \text{if } \lambda_i \ge \beta \cdot L_{\text{train}} \quad \text{（低频，PI 缩）} \\ \text{smooth interp} & \text{otherwise} \quad \text{（中频，过渡）} \end{cases}$$

其中 $\alpha = 1, \beta = 32$ 是 paper 推荐值。新旋转角：

$$\theta'_i = \theta_i \cdot h(\lambda_i)$$

具体的 smooth 函数（YaRN paper §3.3）：

$$\gamma(r) = \begin{cases} 0 & r \le \alpha \\ 1 & r \ge \beta \\ \frac{r - \alpha}{\beta - \alpha} & \text{otherwise} \end{cases}$$

其中 $r = L_{\text{train}} / \lambda_i$（ratio，每 max_seq 周期数）。然后

$$h(\lambda_i) = (1 - \gamma) \cdot 1 + \gamma \cdot \frac{1}{s}$$

——线性 blend "不缩"与"PI 缩"。

#### 2.3.2 Attention Temperature（精修）

YaRN 还观察到：scale factor $s$ 越大，attention 分布越"散"（远处 token 太多 → softmax 平滑化）。补救——**在 attention score 上加一个 temperature**：

$$\text{Attention} = \text{softmax}\!\left(\sqrt{\frac{1}{t}} \cdot \frac{QK^\top}{\sqrt{d_k}}\right) V$$

其中 $t = 0.1 \cdot \ln(s) + 1$（YaRN 经验公式）。$\sqrt{1/t} > 1$ 会让 logits 变大、softmax 更 sharp，抵消"远处 token 过多"的稀释效应。

工程实现上**不用真的改 attention kernel**——只要把 q, k 都乘 $\sqrt[4]{1/t}$，等价于 score 乘 $\sqrt{1/t}$。LLaMA / Mistral 的 HF 实现就是这么做的。

#### 2.3.3 YaRN 综合效果

- LLaMA-2 7B：4k → 128k，fine-tune 400 step（约 0.1B token），perplexity 与原 4k 几乎一致
- LLaMA-3 8B 官方 instruct 版的 128k context 用的就是 YaRN
- **比 PI 与 NTK-aware 都强**——成为业界事实标准

### 2.4 LongRoPE（Microsoft 2024）

LongRoPE 把"分频段缩放"推到极致——**用进化算法（EA）直接搜每个维度的 rescale factor**，不用人工分高频 / 中频 / 低频。

**搜索目标**：对每个 RoPE 维度 $i \in [0, d/2)$，找一个 rescale 系数 $\lambda_i \in [1, s_{\max}]$，最小化目标长度上的 perplexity。

**搜索空间**：维度数 $\sim 64$（LLaMA-7B 的 $d_k = 128$），每个维度 rescale factor 离散化成几十个选择 → 总搜索空间 $\sim 10^{100}$，无法暴搜。

**EA 解法**：population-based search，每代用 100 个候选 rescale 向量，根据 PPL 选 top-k 交叉变异，跑几百代收敛。

**两阶段扩展**：
1. **Stage A**（短长度 fine-tune）：用 EA 搜出 $L_{\text{new}}/L_{\text{train}} = 8 \times$ 的 rescale 向量，fine-tune 几百 step
2. **Stage B**（更长扩展）：在 Stage A 基础上再搜更激进的 rescale，直接扩到目标长度（如 2M）

**关键结果**：
- LLaMA-2 7B：4k → **2M context**（500× 扩展），fine-tune 1B token
- Phi-3-mini 用 LongRoPE 扩到 128k，工业部署
- 在 RULER 上比 YaRN 高 5-15 个点（尤其在 multi-hop 任务）

**与 YaRN 区别**：YaRN 是"人工分 3 段、参数少"，LongRoPE 是"算法搜 64 维独立 rescale、参数多"——LongRoPE 表达能力更强但需要 EA 搜索算力。工业实战常先用 YaRN（够用 + 训练快），追求极致 context 时上 LongRoPE。

### 2.5 标准 continued-pretraining pipeline（必背）

把一个 8k context 的 base model 扩展到 128k 的工业流程：

```
Stage 1：base pretraining
   ├─ 数据：FineWeb / Dolma / RefinedWeb，4k-8k 长度的常规网页数据
   ├─ 训练：几 T token（万亿级），常规预训练
   └─ 输出：8k-context base model（如 LLaMA-3 8B base）

         ↓ 切换位置编码（不重新训）

Stage 2：long-context extension（一次性 config 改动）
   ├─ 选 PI / NTK-aware / YaRN / LongRoPE 之一
   ├─ 改 config.rope_scaling = {"type": "yarn", "factor": 16, ...}
   └─ 不训练，只是改 cos/sin 表的生成方式

         ↓ continued pretraining

Stage 3：long-context continued pretraining 🔥
   ├─ 数据：长文档（书、长 GitHub 代码、长论文 + 合成长样本）
   ├─ 序列长度：32k → 64k → 128k 渐进式拉长
   ├─ 训练量：100B-500B token（不是几 T，远小于 Stage 1）
   ├─ 必须用 sequence parallelism / ring attention 才能塞下显存
   └─ 输出：long-context base model（在长 PPL 上稳定）

         ↓ long-context SFT

Stage 4：long-context SFT
   ├─ 数据：LongAlign / ProLong / Long-ShareGPT 等长指令数据
   ├─ 任务：长文档 QA / 多文档总结 / 长代码 review
   ├─ 序列长度：与 Stage 3 一致（32k-128k）
   └─ 输出：long-context instruct model

         ↓ 评测

Stage 5：long-context eval
   ├─ Needle-in-a-Haystack（基线，必过）
   ├─ RULER（综合，工业标准）
   ├─ LongBench / ∞Bench / L-Eval（领域细分）
   └─ 业务侧 RAG / agent 长任务效果
```

**为什么 Stage 3 必须上 100B+ token**：仅 Stage 2 改 config + 短 fine-tune 几百 step 通常 PPL 看着稳，但实际长 context 下游任务全崩——模型没有"长程依赖建模"的训练信号。Chen 2023 的 PI paper 说几百 step 够，是因为他们只测 PPL 不测 needle / RULER。**工业现状（LLaMA-3 / Qwen2.5）：Stage 3 都是 100B-500B token 起步**。

### 2.6 长 context 评测谱

| benchmark | 出处 | 设计 | 结论 |
|---|---|---|---|
| **Needle-in-a-Haystack** | Greg Kamradt 2023 | 在长文档某位置插入"密钥句"，问模型 retrieve | **基线**，必过；过了不代表真能用 |
| **RULER** | NVIDIA 2024 | Needle 升级版：multi-needle / multi-hop / 顺序变化 / aggregation | **工业标准**，量化 effective context |
| **LongBench** | 清华 2023 | 综合多任务（QA / 总结 / 代码 / few-shot），中英双语 | 中文社区主流 |
| **∞Bench** | 2024 | 100k+ 长任务，含数学 / 代码 | 测极长 context |
| **LooGLE** | 2023 | 实时新闻 + 时序推理 | 测时序敏感任务 |
| **L-Eval** | 2023 | 通用长 eval suite | 较早提出 |
| **NIAH-PG19** | 学术变体 | 在 PG19（古典书）上做 needle | 传统语料基线 |

**典型 RULER 结果**（2024 工业模型在宣称 128k context 下）：
- GPT-4o：128k 宣称，RULER effective ~64k
- Claude 3.5 Sonnet：200k 宣称，RULER effective ~128k
- LLaMA-3.1 70B：128k 宣称，RULER effective ~32k（明显低于宣称）
- Qwen2.5 72B：128k 宣称，RULER effective ~64k

工业经验：**effective context 通常是 advertised 的 1/2 到 1/4**——做 RAG / agent 时要按 effective context 设计，而不是按宣称的数字。

### 2.7 Long context 与 KV cache、显存

Long context 是 memory-bound 的极致挑战，与 5.2 GQA / MLA、11.2 PagedAttention、11.4 量化深度纠缠：

- **KV cache 大小爆炸**：LLaMA-3 8B（32 layers, 32 heads, $d_k = 128$，bf16）的单 token KV cache = $32 \times 32 \times 128 \times 2 \times 2 = 524288$ B = 0.5 MB。1M context 单序列 KV = **500 GB**，远超单卡 H100 80GB
- **必须 GQA / MLA**：LLaMA-3 用 GQA（K/V head = 8，1/4 压缩）；DeepSeek-V2/V3 用 MLA（约 $1/15$ 压缩）。1M context 下没有 GQA / MLA 完全跑不动
- **PagedAttention（vLLM）**：把 KV cache 切成 page，按需分配，避免连续显存碎片。长 context 必备
- **KV cache 量化**：FP8 / INT8 KV cache，进一步 2-4× 压缩
- **SGLang RadixAttention**：在长 prefix 重复场景（多用户共享 system prompt、agent 重复 trajectory）显著优化——把相同 prefix 的 KV cache 共享成树（radix tree）

工程结论：**long context 推理性能瓶颈不在 attention compute，而在 KV cache 的内存带宽与容量**。详见 11.2-11.4。

---

## 3. 最小代码示例

### 3.1 PI 缩放：在 RoPE precompute 时把 t 除以 s（11 行）

```python
import torch

def rope_cache_pi(d_k: int, max_seq: int, scale: float = 8.0,
                  base: float = 10000.0):
    """
    Position Interpolation 版本 RoPE cache.
    scale = L_new / L_train，把推理位置 m 压回训练范围。
    """
    inv_freq = 1.0 / (base ** (torch.arange(0, d_k, 2).float() / d_k))   # (d_k/2,)
    t = torch.arange(max_seq).float() / scale                             # 关键：除以 s
    freqs = torch.outer(t, inv_freq)                                      # (max_seq, d_k/2)
    emb = torch.cat([freqs, freqs], dim=-1)                               # (max_seq, d_k)
    return emb.cos(), emb.sin()                                           # (max_seq, d_k) ×2

# 用法：原本训练 L=4k，扩到 L=32k → scale=8
cos, sin = rope_cache_pi(d_k=128, max_seq=32768, scale=8.0)
```

关键就一行：`t = torch.arange(max_seq).float() / scale`。其余与标准 RoPE precompute 一字不差。**这就是 PI 的全部实现**——简单到颠覆很多人的认知。

### 3.2 NTK-aware：调大 base 的具体计算（13 行）

```python
import torch

def rope_cache_ntk(d_k: int, max_seq: int, scale: float = 8.0,
                   base: float = 10000.0):
    """
    NTK-aware 版本 RoPE cache.
    调大 base，让低频维度旋转变慢；高频几乎不变。
    经验公式：b' = b * s^(d/(d-2))
    """
    base_new = base * (scale ** (d_k / (d_k - 2)))                         # 关键：调 base
    inv_freq = 1.0 / (base_new ** (torch.arange(0, d_k, 2).float() / d_k)) # (d_k/2,)
    t = torch.arange(max_seq).float()                                      # 注意 t 不缩
    freqs = torch.outer(t, inv_freq)                                       # (max_seq, d_k/2)
    emb = torch.cat([freqs, freqs], dim=-1)                                # (max_seq, d_k)
    return emb.cos(), emb.sin()

# 用法：scale=8 → base 从 10000 → ~10000 * 8^1.016 ≈ 84000
cos, sin = rope_cache_ntk(d_k=128, max_seq=32768, scale=8.0)
print(10000 * (8 ** (128 / 126)))                                          # 验证 base 调到多少
```

注意：NTK 是调 `base` 不缩 `t`；PI 是缩 `t` 不调 `base`——**两者是正交的两种思路，不要混淆**。LLaMA-3 long-context 版本的 `config.json` 里能看到 `"rope_theta": 500000.0`，背后就是 NTK 思想（虽然现代实际用 YaRN 统一 framework）。

### 3.3 YaRN frequency mask：分频段 rescale（22 行 PyTorch）

```python
import torch
import math

def rope_cache_yarn(d_k: int, max_seq: int, L_train: int = 4096,
                    scale: float = 8.0, base: float = 10000.0,
                    alpha: float = 1.0, beta: float = 32.0):
    """
    YaRN: NTK-by-parts. 高频不缩、低频 PI 缩、中频平滑过渡。
    alpha, beta 是 paper 推荐的频段切分阈值（每 max_seq 旋转圈数）。
    """
    inv_freq = 1.0 / (base ** (torch.arange(0, d_k, 2).float() / d_k))   # (d_k/2,)
    wavelen = 2 * math.pi / inv_freq                                      # 每维波长 (d_k/2,)
    # ratio = L_train / wavelen = 训练长度内绕几圈
    ratio = L_train / wavelen                                             # (d_k/2,)
    # 平滑权重 gamma：高频 (ratio<=alpha) → 0（不缩）；低频 (ratio>=beta) → 1（PI 缩）
    gamma = ((ratio - alpha) / (beta - alpha)).clamp(0.0, 1.0)            # (d_k/2,)
    # h(lambda) = (1-gamma)*1 + gamma*(1/s) → 高频 1、低频 1/s
    h = (1 - gamma) * 1.0 + gamma * (1.0 / scale)                         # (d_k/2,)
    inv_freq_yarn = inv_freq * h                                          # 分频段 rescale
    t = torch.arange(max_seq).float()
    freqs = torch.outer(t, inv_freq_yarn)                                 # (max_seq, d_k/2)
    emb = torch.cat([freqs, freqs], dim=-1)                               # (max_seq, d_k)
    # attention temperature: q/k 乘 sqrt(1/t_attn)，t_attn = 0.1*ln(s)+1
    t_attn = 0.1 * math.log(scale) + 1.0
    mscale = math.sqrt(1.0 / t_attn)                                      # 应用到 cos/sin 即可
    return emb.cos() * mscale, emb.sin() * mscale

cos, sin = rope_cache_yarn(d_k=128, max_seq=131072, L_train=8192, scale=16.0)
```

**逐段解释**：
- `wavelen = 2π / inv_freq`：每个维度的"波长"（旋转一圈所需的位置数）
- `ratio = L_train / wavelen`：在训练长度内这个维度旋转了几圈——高频维度旋转很多圈（ratio 大），低频维度可能不到一圈（ratio 小）
- `gamma`：用 `clamp(0, 1)` 实现 $\gamma$ 函数——高频 ratio ≤ alpha 截断为 0、低频 ratio ≥ beta 截断为 1、中间线性过渡
- `h`：rescale factor，1（保留）与 1/s（PI 缩）的线性 blend
- attention temperature 通过把 cos/sin 整体乘 `sqrt(1/t_attn)` 实现——等价于把 q, k 都乘 `sqrt(sqrt(1/t_attn))`，最终 score 被乘 `sqrt(1/t_attn)`

### 3.4 HuggingFace LLaMA config 的 long-context 配置

实际工业用法**不需要自己写上面这些**——HF transformers 已经把 PI / NTK / YaRN 都内置：

```python
# config.json 修改示例（LLaMA-3 8B 4k → 64k）
{
  "model_type": "llama",
  "max_position_embeddings": 8192,
  "rope_theta": 500000.0,
  "rope_scaling": {
    "type": "yarn",
    "factor": 8.0,
    "original_max_position_embeddings": 8192,
    "beta_fast": 32,
    "beta_slow": 1
  }
}

# 或 Python API
from transformers import AutoConfig, AutoModelForCausalLM
config = AutoConfig.from_pretrained("meta-llama/Meta-Llama-3-8B")
config.rope_scaling = {
    "type": "yarn",
    "factor": 8.0,
    "original_max_position_embeddings": 8192,
}
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B", config=config, torch_dtype="bfloat16"
)
```

`type` 可选：`"linear"`（PI）、`"dynamic"`（NTK-aware）、`"yarn"`、`"longrope"`。**改完 config 必须 continued pretrain**（见 §2.5 Stage 3），不能直接 zero-shot inference。

---

## 4. 工程踩坑与经验

- ❗ **Long-context 扩展必须 continued pretrain 100B+ token 才稳，直接 zero-shot inference 大概率掉点**。Chen 2023 的 PI paper 报告"几百 step fine-tune 即可"是只测 PPL 的乐观结论；工业现实是只 fine-tune 短 step → PPL 看着稳但 needle / RULER 全崩。LLaMA-3 / Qwen2.5 的 long-context 版本都是 100B-500B token 的 long-context continued pretraining 才到位。**面试加分**：能说出"PPL 不掉 ≠ 长程能力到位"。

- ❗ **Continued pretrain 时长 context 数据稀缺，必须合成**。互联网上 32k+ 的"自然文档"极少（书 / 长论文 / 长代码仓库），直接拿来训远不够。LongAlign / ProLong / Long-Data 等工作专门做"合成长样本"——把多个相关短文档拼到一起、给指令要求做"全局总结"，迫使模型用全长 context。**工程坑**：合成数据如果只是简单 concat（无 cross-reference），模型仍然只看局部、长程能力不到位。必须造 multi-hop / 跨段引用类样本。

- ❗ **Needle test 满分 ≠ 实际长 context 能用，必须看 RULER**。Needle 是"在 1M token 里 retrieve 一句明显的密钥句"——这是非常局部的检索任务，模型只要能定位 + 复制就过。但实际 RAG 多 hop / 长代码理解 / long-document QA 需要"跨多个段落综合推理"，effective context 远小于 needle context。**面试常考**："为什么 LLaMA-3 needle 满分但实际 32k 之后掉点"——RULER 测的就是这个 gap。

- ❗ **PI / NTK / YaRN 的 rescale factor `s` 必须与训练时一致，推理时改了会塌**。如果 Stage 3 用 `scale=16` continued pretrain 出来的模型，推理时 config 改成 `scale=8` → 模型见到的旋转角分布完全不同，性能爆炸。**工程规则**：把 `rope_scaling` 字段固化在 model card 与 config.json 里，下游用户严禁修改（除非重训）。HF 的 `config.rope_scaling` 字段是模型属性的一部分，与 weight 绑定。

- ❗ **HuggingFace `config.rope_scaling` 字段在不同 model family 字段名不一致**。LLaMA：`{"type": "yarn", "factor": ..., "original_max_position_embeddings": ...}`；Qwen2：`{"type": "yarn", "factor": ..., "original_max_position_embeddings": ...}`（基本一致）；Mistral 早期：仅支持 `"linear"` 与 `"dynamic"`；Phi-3：用 `"longrope"` 类型，字段是 `short_factor` / `long_factor` 两个数组。**工程坑**：从 LLaMA fork 一份 long-context 配置贴到 Mistral / Phi-3 上 → 启动报 schema 错或静默用错配置。永远查官方 modeling 源码确认字段名。

- ❗ **Long-context 训练显存巨大，必须 sequence parallelism (SP) + ring/striped attention**。32k 序列单卡 H100（80GB）训练时 attention 中间 activation 就 $32000^2 \times h \times \text{bytes}$ → 几十 GB；128k 直接爆。**工程标配**：(1) sequence parallelism（Megatron-LM）按序列维切分 activation，每张卡只持有 1/SP 段；(2) ring attention（Liu 2023）让每张卡环形传 K/V 块，每张卡 KV cache 也是 1/SP；(3) FlashAttention 做 IO-aware reduce。SP=8 是 long-context 训练的常见起点，详见 7.3。

- ❗ **评测要用 RULER 或 LongBench 综合 benchmark，单纯 needle 已不够**。2024+ 工业评测共识：needle pass 是"基线门票"（不过的话 long-context 不可用）；真实评测必须上 RULER（13 个 sub-task：multi-needle / aggregation / multi-hop tracing / variable tracking / common words extraction 等），结合下游业务任务（RAG QA / 长代码补全 / agent 多步执行）。投行 / 互联网公司的 model selection 绝大多数看 RULER 而非 needle。

- ❗ **MLA + RoPE 在 long-context 扩展时与 GQA 不完全一样，DeepSeek 论文有专门讨论**。MLA（Multi-head Latent Attention，5.2 节）把 K/V 压缩成低秩 latent，但 RoPE 必须在压缩前的"原始 K"上施加（rotation 不与低秩压缩交换）。DeepSeek-V2 的解法是 **decoupled RoPE**：把 K 拆成两部分，一部分参与 MLA 压缩、一部分单独走 RoPE 旋转。Long-context 扩展时这两部分需要分别处理 rescale，比标准 RoPE 复杂一层。详见 5.2 与 DeepSeek-V2 paper §2.1.3。

- ❗ **Long-context fine-tune 时长样本要 packing 而不是 padding，否则 90% 算力在算 pad**。32k 长度的 batch 里如果只有 10% 真实样本、90% pad，attention 仍按 32k 算。必须用 sample packing（多条短样本拼到 32k 内、用 attention mask 隔离）。HF `transformers` 的 `DataCollatorForLanguageModeling` 默认不 packing；要么手写 packing collator、要么用 axolotl / OpenRLHF 等支持 packing 的训练框架。

- ❗ **bf16 long-context 下 cos/sin 表必须 fp32**。$m \theta_i$ 在 $m = 100k$、低频 $\theta_i = 10^{-3}$ 时已经是 100，cos/sin 的小数部分在 bf16 下精度不够（bf16 mantissa 只有 7 bit），相邻位置的旋转角会被 round 成同值 → attention 退化。**修复**：cos/sin precompute 永远 fp32 存，apply_rotary 时 cast 到 q/k 的 dtype。HF LLaMA 实现就是这么做。

---

## 5. 经典 paper

- **Chen et al., 2023 — Extending Context Window of Large Language Models via Position Interpolation** — PI 必引。Meta 团队第一篇系统化把 RoPE 长 context 扩展数学化的 paper。读 §3 "Method" 与 §5 实验（LLaMA 7B 2k → 32k，fine-tune 1k step PPL 不掉）就够。**面试常考**：能说出 "PI 的 1 行修改"（除以 s）+ "为什么必须 fine-tune"。
- **Peng et al., 2023 — YaRN: Efficient Context Window Extension of Large Language Models** — YaRN 必引。把 NTK-by-parts + attention temperature 系统化，是 LLaMA-2/3 / Qwen / Mistral 的工业标准。读 §3.3-§3.4 把分频段公式与 temperature 推完，能讲清"为什么比 PI 与 NTK 都强"。
- **Ding et al., 2024 — LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens** — Microsoft，LongRoPE 必引。把 EA 搜索 RoPE rescale 引入长 context 扩展，扩到 2M。读 §3 算法描述与 §4 主实验（LLaMA-2 7B → 2M、Phi-3 → 128k）。
- **bloc97, 2023 — NTK-aware Scaled RoPE allows LLaMA models to have extended context** — reddit r/LocalLLaMA 草根博文，但被工业界广泛引用。30 分钟读完，是理解 "调 base 与 PI 的 trade-off" 的最好入门。链接：https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/
- **加分阅读**：
  - Hsieh et al., 2024 — RULER: What's the Real Context Size of Your Long-Context Language Models — NVIDIA，long-context 评测必读。13 个 sub-task 揭示宣称 vs effective context 的巨大 gap
  - Liu et al., 2023 — Ring Attention with Blockwise Transformers for Near-Infinite Context — long-context 训练 infra 必读
  - Bai et al., 2024 — LongAlign: A Recipe for Long Context Alignment — Stage 4 long-context SFT 数据构造的代表工作

---

## 6. 自测与面试题

**Q1（公式）**：写出 PI 与 NTK-aware 的核心改动公式，说明为什么 NTK 比 PI 好。

<details>
<summary>Answer sketch</summary>

PI（Chen 2023）：把推理时位置 $m$ 缩放回训练范围

$$m' = m / s, \quad s = L_{\text{new}} / L_{\text{train}}$$

旋转角变成 $\frac{m}{s} \theta_i$，**所有维度同等缩放**。

NTK-aware（bloc97 2023）：调大 RoPE 的 base

$$b' = b \cdot s^{d / (d - 2)}, \quad \theta'_i = (b')^{-2i/d}$$

旋转角变成 $m \theta'_i$。

为什么 NTK 比 PI 好：

- $\theta_i = b^{-2i/d}$ 中，**高频维度**（$i$ 小，尤其 $i = 0$）$\theta_0 = 1$ 与 $b$ 无关 → 调大 $b$ 高频几乎不变 → **保留局部信号**
- **低频维度**（$i$ 大）$\theta_{d/2-1}$ 强依赖 $b$ → 调大 $b$ 低频显著变慢 → **拉宽长程波长**
- PI 是"全频段同等缩"，高频被压得相邻 token 难区分（旋转角差从 1 弧度变成 1/s 弧度，s=8 时只剩 0.125 弧度）
- NTK 是"只缩低频、保高频"，相邻 token 仍可清晰区分
- 实证：NTK zero-shot 直接外推就比 PI 强不少；PI 几乎必须 fine-tune

加分：
- 能推 NTK 的经验公式来源：希望 $L_{\text{new}} \cdot (b')^{-(d-2)/d} = L_{\text{train}} \cdot b^{-(d-2)/d}$，解出 $b' = b \cdot s^{d/(d-2)}$
- 能说出 LLaMA-3 long-context 版本 `rope_theta` 字段（500000+）是 NTK 思想的工业落地
- 能指出 NTK 与 PI 是**正交**的两种 idea（NTK 调 base、PI 缩 t），不是互斥；YaRN 把两者合并

</details>

**Q2（pipeline）**：要把一个 8k context model 扩展到 128k，列出完整 4 阶段 pipeline，每阶段说明数据 / 训练量 / 关键技术。

<details>
<summary>Answer sketch</summary>

完整 pipeline（参考 LLaMA-3 / Qwen2.5 工业实践）：

**Stage 1 — Base pretraining**
- 数据：FineWeb / Dolma / RefinedWeb 等大规模 web data，长度 4k-8k
- 训练量：几 T token（万亿级）
- 关键技术：常规预训练，CLM loss + AdamW + bf16
- 输出：8k-context base model

**Stage 2 — Long-context extension（一次性 config 改动）**
- 选 PI / NTK-aware / YaRN / LongRoPE 之一
- 改 `config.rope_scaling`（HF 标准字段）
- 不训练，只是改 cos/sin 表的生成方式

**Stage 3 — Long-context continued pretraining 🔥（最关键）**
- 数据：长文档（书 / 长论文 / 长 GitHub 代码）+ 合成长样本（LongAlign / ProLong）
- 序列长度：32k → 64k → 128k 渐进拉长（避免一次拉到位训练不稳定）
- 训练量：100B-500B token（**远小于 Stage 1 的几 T，但远大于"几百 step fine-tune"**）
- 关键技术：sequence parallelism + ring attention（不上 SP 显存爆）、sample packing（避免 90% 算 pad）
- 输出：long-context base model（PPL 在长 context 上稳定）

**Stage 4 — Long-context SFT**
- 数据：LongAlign / ProLong / Long-ShareGPT 等长指令数据
- 任务：长文档 QA / 多文档总结 / 长代码 review / agent 多步 trajectory
- 序列长度：与 Stage 3 一致
- 输出：long-context instruct model

**Stage 5 — Eval（不算 train 阶段但必做）**
- Needle-in-a-Haystack（基线，必过）
- RULER（综合，工业标准，看 effective context）
- LongBench / ∞Bench / 业务侧任务

加分：
- 能说出 Stage 3 必须 100B+ token 的原因：仅几百 step fine-tune PPL 看着稳但 needle / RULER 全崩
- 能说 long-context 数据稀缺、必须合成（LongAlign / ProLong）
- 能说 sequence parallelism / ring attention 在 7.3 节，是 long-context 训练的硬性 infra 依赖
- 能说 Stage 4 数据要造 multi-hop / 跨段引用样本，不能简单 concat

</details>

**Q3（评测）**：Needle test 满分但实际 multi-hop 任务差，可能的 3 个原因。

<details>
<summary>Answer sketch</summary>

3 个核心原因：

**原因 1：Needle 是"局部检索"任务，multi-hop 是"全局综合推理"，两者考核能力完全不同**
- Needle：在长文档某位置插入一句明显的密钥（如 "the magic number is 42"），问模型 retrieve → 模型只要能"看到 + 复制"就过，不需要任何跨段推理
- Multi-hop：要把分散在多个段落的事实串起来推（"A 在 B 之前；B 在 C 之前；问 A 是否在 C 之前"）→ 需要在长 context 里同时维持多个事实的注意力，并组合
- effective context（能维持多事实关注）远小于 retrieval context（能定位单一密钥）

**原因 2：Stage 3 continued pretraining 数据缺乏 multi-hop 类合成样本**
- 工业 long-context continued pretrain 的数据多是"长文档 next token prediction"，模型学到的是"长程 LM"而非"长程推理"
- 真实 multi-hop 训练信号需要专门的合成数据（多文档 QA、跨段落引用、时序推理任务）
- LongAlign / ProLong 等工作就是为了补这一类训练信号

**原因 3：Attention 在长 context 下的 "lost in the middle" 现象**
- Liu et al. 2023 实证：长 context 中段的 token 被 attention 显著弱化（attention map 偏向开头 + 结尾）
- multi-hop 需要的事实如果分散在中段 → 模型难以同时关注 → 推理失败
- 即使 RoPE / YaRN 数学上能 cover 长 context，attention 的归纳偏置仍倾向"近 + 极远 (anchor)"

加分：
- 能引用 RULER 的 13 sub-task 设计（multi-needle / multi-hop tracing / variable tracking 等都是为了暴露这类 gap）
- 能说出工业经验：effective context 通常是 advertised 的 1/2 到 1/4
- 能说 RAG / agent 设计要按 effective context 而非宣称数字
- 能联想到 Lost in the Middle (Liu 2023) 这篇 paper

</details>

---

## 7. 延伸阅读

- [Chen et al. 2023 — Position Interpolation (arXiv)](https://arxiv.org/abs/2306.15595) — PI 必读
- [Peng et al. 2023 — YaRN (arXiv)](https://arxiv.org/abs/2309.00071) — 工业标准，必读
- [Ding et al. 2024 — LongRoPE (arXiv)](https://arxiv.org/abs/2402.13753) — 2M context 的进化算法解
- [bloc97 — NTK-aware scaling reddit blog](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/) — 草根工作，工业必读
- [Hsieh et al. 2024 — RULER (arXiv)](https://arxiv.org/abs/2404.06654) — long-context 评测必读
- [Liu et al. 2023 — Lost in the Middle (arXiv)](https://arxiv.org/abs/2307.03172) — 解释 needle vs multi-hop gap
- [Bai et al. 2024 — LongAlign (arXiv)](https://arxiv.org/abs/2401.18058) — Stage 4 long-context SFT 数据
- [Liu et al. 2023 — Ring Attention (arXiv)](https://arxiv.org/abs/2310.01889) — long-context 训练 infra
- [Greg Kamradt — Needle in a Haystack repo](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) — needle test 评测代码
- [HuggingFace LLaMA modeling 源码 — `LlamaRotaryEmbedding`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) — 工业 rope_scaling 实现
- 推荐继续读本教程的 **6.6 节《经典开源 LLM 解读：LLaMA / Qwen / DeepSeek 系列》**——具体看每家 long-context 选了什么方案
- 推荐继续读本教程的 **11.2-11.4 节（PagedAttention / RadixAttention / 量化）**——long-context 推理的工程支柱
- 推荐继续读本教程的 **7.3 节《Sequence / Context / Expert Parallelism》**——long-context 训练 infra
