---
title: "4.3 位置编码：绝对 / 相对 / RoPE / ALiBi"
description: "self-attention 把序列当成集合——不加位置信息时颠倒 token 顺序输出完全不变；本节系统讲清四代位置编码的演进：绝对（Sinusoidal / Learned）→ 相对（Shaw / T5 bucket bias）→ RoPE（旋转位置编码，当代 LLM 绝对主流）→ ALiBi（线性距离衰减，外推友好），重点把 RoPE 的\"复数旋转 = 仅依赖相对位置\"的数学性质推一遍、把 "
---

> ⏱ 预计阅读 60 分钟 ｜ 难度 ★★★ 🔥 必考 ｜ 前置：4.1 self-attention

## 一句话本节讲什么

self-attention 把序列当成**集合**——不加位置信息时颠倒 token 顺序输出完全不变；本节系统讲清四代位置编码的演进：**绝对**（Sinusoidal / Learned）→ **相对**（Shaw / T5 bucket bias）→ **RoPE**（旋转位置编码，当代 LLM 绝对主流）→ **ALiBi**（线性距离衰减，外推友好），重点把 RoPE 的"复数旋转 = 仅依赖相对位置"的数学性质推一遍、把 `apply_rotary` 的 PyTorch 标准实现写出来，最后用一段话把 6.5 long-context 扩展（PI / NTK / YaRN / LongRoPE）的"位置编码视角"动机串起来——这是当代 LLM 架构里最高频的设计选择题之一。

---

## 1. Mental model（直觉）

### 1.1 为什么需要位置编码：self-attention 是"集合操作"

回到 4.1 §2.6 性质 1：**self-attention 是置换等变的**。把输入序列 $X = [x_1, x_2, x_3]$ 与置换后的 $X' = [x_3, x_1, x_2]$ 分别喂进同一个 self-attention 层，输出会是同一组向量、只是顺序跟着调换——每个 token 的"内容表示"完全没变。

形式化：对任意置换矩阵 $P$，

$$\text{Attention}(PXW_Q,\, PXW_K,\, PXW_V) = P \cdot \text{Attention}(XW_Q, XW_K, XW_V)$$

物理含义：对 self-attention 而言，"猫吃鱼"和"鱼吃猫"是**完全等价的输入**——它只看到一个 3 元素集合 `{猫, 吃, 鱼}`，没有任何顺序信号。这对图、点云这种天然无序的数据是优点，但对自然语言、代码、音频这种顺序敏感的序列是**灾难**。

vs RNN：RNN 通过 timestep 串行 pass（$h_t = f(h_{t-1}, x_t)$）天然把"我是第几个 token"编码进 hidden state 里，根本不需要额外的位置信息。**self-attention 用并行换来的代价就是——必须显式注入位置信息**。

一句话直觉，记牢：

> **self-attention 把序列当 set；要让它变回 sequence，必须把"我是第几个"的信息塞进 token 表示或 attention 分数里。**

### 1.2 三大流派的演化路径

历史上人类一共试过 4 代方案，每一代都在解决前一代的痛点：

| 代 | 方案 | 代表模型 | 解决了什么 | 引入了什么问题 |
|---|---|---|---|---|
| 1 | **Sinusoidal**（绝对） | 原版 Transformer | 第一个能用的方案；公式可解析 | 模型实证上学不到长程外推 |
| 2 | **Learned**（绝对） | BERT / GPT-2/3 | 训练效果优于 Sinusoidal | **完全不能外推**（超 max_pos 直接随机） |
| 3 | **Relative**（相对） | T5 / Transformer-XL / Shaw 2018 | 天然可外推（依赖距离不依赖绝对位置） | 实现复杂，需要改 attention 公式 |
| 4 | **RoPE**（旋转，混合绝对+相对） | LLaMA / Qwen / DeepSeek / Mistral / Gemma | 兼具绝对（对位置 m 旋转）+ 相对（attention 仅依赖 n−m）；实现简单（只在 q/k 投影后加一次旋转） | 仍需扩展才能 long-context（PI / NTK / YaRN） |
| 4' | **ALiBi**（旋转的对手） | BLOOM / MPT / Falcon-7B | 完全不动 q/k，直接在 score 上加线性距离衰减；外推稳定 | in-distribution 长度上效果不如 RoPE，现已较少用 |

**当代结论**：开源主流（LLaMA-2/3 / Qwen / Mistral / DeepSeek / Gemma）几乎全用 RoPE；ALiBi 仅在早期 BLOOM / MPT 系上见。**面试 90% 会考 RoPE**——能推 + 能写代码。

### 1.3 RoPE 的图像直觉：把每对维度看成 2D 平面上的旋转

RoPE 的核心是个非常优美的几何 trick——

> **把 query / key 的每两维 $(2i, 2i+1)$ 看成 2D 平面上的一个向量；按 token 在序列中的位置 $m$，把这个 2D 向量旋转一个角度 $m\theta_i$**。

不同维度对 $i$ 用不同的"旋转频率" $\theta_i = 10000^{-2i/d}$——低维旋转快（高频），高维旋转慢（低频）。这模仿 sinusoidal PE 的多频率设计。

ASCII 直觉图（以 $d_k = 4$ 为例，每 2 维一组共 2 组）：

```
                  原始 q (位置无关)
       ┌────────────────────────────────┐
       │  q[0:2]            q[2:4]      │
       │   ↑                  ↑          │
       │   │                  │          │
       │   │ 2D 向量           │ 2D 向量  │
       │   └→                 └→         │
       │  (高频组)          (低频组)     │
       └────────────────────────────────┘
                  │ 在位置 m 上旋转
                  ▼
       ┌────────────────────────────────┐
       │  q'[0:2] = R(m θ_0) q[0:2]     │
       │  q'[2:4] = R(m θ_1) q[2:4]     │
       │     旋转角大           旋转角小  │
       └────────────────────────────────┘

每个位置 m 都是一次"分维度群体旋转"，角度只跟 m 与维度 i 有关。
```

**关键性质**（数学在 §2.3 推完，先记结论）：旋转后的 $\langle q'_m, k'_n \rangle$ **只依赖相对位置 $n - m$**——这就是为什么 RoPE 同时是"绝对"（旋转量与位置 $m$ 有关）和"相对"（attention 分数只看距离）。这种"以绝对位置实现相对位置编码"的优雅性，是 RoPE 战胜 Shaw / T5 bucket 的根本原因。

### 1.4 ALiBi 的图像直觉：直接在 score 上"按距离打折"

ALiBi 的哲学完全相反——**不动 q, k，直接在 attention score 上减一个与距离成正比的惩罚项**：

```
原始 score:        q_i · k_j
ALiBi score:      q_i · k_j  -  m_h · |i - j|
                              └───────┬───────┘
                              线性距离惩罚（每 head 一个斜率 m_h）
```

距离越远惩罚越大 → softmax 之后远处 token 的权重指数级衰减。这种 inductive bias 类似"局部 attention"——但又不是硬截断，而是 soft decay。

**为什么 ALiBi 外推友好**：因为它不依赖任何"训过的位置 embedding 或旋转角"，纯粹是 attention 公式里加一项距离函数；推理时序列再长，只要算得出 $|i - j|$ 就能用。这是它在 2021 年红极一时的卖点。

但**为什么 ALiBi 输给了 RoPE**：现代 LLM 的 in-distribution 长度（即训练 max_seq）已经被推到 32k+，ALiBi 的"距离惩罚"会让远处 token 几乎被 mask 掉，丢失长程信息；而 RoPE + 后续的 YaRN / LongRoPE 扩展能在保持长程能力的同时把外推做到百万 token 级。**2024-2026 的主流共识：RoPE + YaRN 体系全面胜出**。

---

## 2. 公式与原理

### 2.1 第一代：Sinusoidal 绝对位置编码（Vaswani 2017）

最朴素的想法——既然 self-attention 没有顺序信号，那就**给每个位置造一个固定向量、加在 token embedding 上**。Vaswani 2017 选择用 sin / cos 的多频率组合：

$$\text{PE}_{(\text{pos}, 2i)} = \sin\!\left(\frac{\text{pos}}{10000^{2i/d}}\right), \quad \text{PE}_{(\text{pos}, 2i+1)} = \cos\!\left(\frac{\text{pos}}{10000^{2i/d}}\right)$$

其中 $\text{pos}$ 是 token 在序列中的位置（$0, 1, \dots, n-1$），$i$ 是维度对的索引（$0, 1, \dots, d/2 - 1$），$d$ 是 hidden dimension。

得到的 $\text{PE} \in \mathbb{R}^{n \times d}$ 直接加到 embedding 上：

$$x'_t = x_t + \text{PE}_t$$

**为什么是 sin/cos 多频率**：每对 $(\sin, \cos)$ 在不同维度对 $i$ 下用不同的"波长"——$i = 0$ 时波长 $2\pi$，$i = d/2 - 1$ 时波长 $2\pi \cdot 10000$。这模仿傅立叶基，让任意 offset $k$ 的 PE 可以表达为 $\text{PE}_{\text{pos}}$ 的**线性变换**（$\sin(a + b) = \sin a \cos b + \cos a \sin b$）——理论上模型能从 $\text{PE}_{\text{pos}}$ 与 $\text{PE}_{\text{pos}+k}$ 之间的线性关系学到"相对距离"。

**实证缺陷**：理论上的"线性可推"性质并没有让模型真正学到长程外推——训过 max_len = 512 的 Transformer 在 1024 长度上效果会显著掉。这是 Sinusoidal 在 LLM 时代被淘汰的根本原因。

### 2.2 第二代：Learned 绝对位置编码（BERT / GPT-2/3）

更暴力的方案——直接学一个 `nn.Embedding(max_pos, d)` 矩阵，每个位置一个向量：

$$\text{PE} \in \mathbb{R}^{\text{max\_pos} \times d}, \quad x'_t = x_t + \text{PE}[t]$$

**优点**：不需要任何先验设计，训练效果通常优于 Sinusoidal（BERT / GPT-2 实证如此）。

**致命缺点**：**完全不能外推**——超出训练 max_pos 的位置直接没有 embedding 可用，强行用随机初始化的位置向量会让模型输出退化为噪声。GPT-3（max_pos = 2048）就是被这个限制死了，要扩 context 必须重新训练。

### 2.3 第三代：相对位置编码（Shaw 2018 / T5 bucket bias）

人类很快意识到——**真正影响 attention 的只是 token 间的相对距离 $i - j$，绝对位置不重要**。Shaw 2018 提出在 attention 里加入相对位置 bias：

$$\text{score}_{ij} = \frac{q_i \cdot k_j}{\sqrt{d_k}} + b_{i-j}$$

$b_{i-j}$ 是一个 learnable 的 scalar，依赖距离 $i - j$ 而不依赖绝对位置。

**T5 bucket bias 变体**（Raffel 2020）：把相对距离映射到固定数量的 bucket（如 32 个），每个 bucket 一个可学习 scalar。距离越大 bucket 跨度越大（log scale），这样既减少参数又支持长距离。

**优点**：
- 天然可外推（依赖距离，距离的语义不会因序列变长而失效）
- 参数量很少（只是 $\mathcal{O}(\text{num\_buckets})$ 个 scalar）

**缺点**：
- 实现复杂——要改 attention 公式，与 SDPA / FlashAttention 的标准接口不直接兼容（FlashAttention 2.0+ 才开始支持任意 bias）
- **不能与 KV cache 完美配合**：T5 的 relative bias 在每层都要算 $i - j$ 的 lookup table，长 context 时这个 lookup 也成 bottleneck

T5 至今仍在用，但 decoder-only LLM 普遍弃用 → 转向 RoPE。

### 2.4 第四代主流：RoPE — Rotary Position Embedding（Su 2021）

RoPE 的核心 insight 极其优雅——**与其在 score 上加 bias，不如在 q/k 投影后给它们"旋转一个与位置成正比的角度"，让内积本身就只依赖相对位置**。

**第 1 步：每两维一组，看成复数 / 2D 平面向量**

把 $q \in \mathbb{R}^{d_k}$ 切成 $d_k / 2$ 个 2D 向量对：

$$q = [\underbrace{q^{(0)}_0, q^{(0)}_1}_{\text{第 0 组}}, \underbrace{q^{(1)}_0, q^{(1)}_1}_{\text{第 1 组}}, \dots, \underbrace{q^{(d_k/2 - 1)}_0, q^{(d_k/2 - 1)}_1}_{\text{第 $d_k/2 - 1$ 组}}]$$

每组 $(q^{(i)}_0, q^{(i)}_1)$ 看成复数 $q^{(i)}_0 + j \cdot q^{(i)}_1 \in \mathbb{C}$，或等价的 2D 平面向量。

**第 2 步：定义旋转角**

第 $i$ 组在位置 $m$ 上的旋转角是 $m \cdot \theta_i$，其中

$$\theta_i = 10000^{-2i/d_k}, \quad i = 0, 1, \dots, d_k/2 - 1$$

低维（$i$ 小）→ $\theta_i$ 大 → 旋转快（高频）；高维（$i$ 大）→ $\theta_i$ 小 → 旋转慢（低频）。这与 sinusoidal 的多频率设计同源。**base = 10000 是 Su 2021 的经验值**——后面 6.5 节会看到 NTK / YaRN 都是在调这个 base。

**第 3 步：旋转矩阵作用**

每组的 2D 旋转矩阵：

$$R_m^{(i)} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix}$$

应用到 q, k：

$$q'_m = R_m \cdot q_m, \quad k'_n = R_n \cdot k_n$$

其中 $R_m$ 是个块对角矩阵——对角线上排着 $d_k / 2$ 个 $2 \times 2$ 旋转块 $R_m^{(0)}, R_m^{(1)}, \dots$。**所以 RoPE 不是把 q 整体旋转，而是分组（每 2 维一组）独立旋转**——这是新手最常踩的实现坑（见 §4 第 1 条）。

**第 4 步（关键性质推导）：内积仅依赖相对位置**

旋转矩阵满足正交性 $R_m^\top R_m = I$，且更关键地：

$$R_m^\top R_n = R_{n - m}$$

（因为 $R_m^\top$ 是逆向旋转 $-m$，$R_n$ 是正向旋转 $n$，复合就是旋转 $n - m$。）

于是：

$$\langle q'_m, k'_n \rangle = (R_m q_m)^\top (R_n k_n) = q_m^\top R_m^\top R_n k_n = q_m^\top R_{n-m} k_n$$

**注意右边只剩 $R_{n-m}$——只依赖相对位置 $n - m$，不依赖 $m$ 和 $n$ 各自的绝对值**。

这就是 RoPE 的核心定理：**虽然旋转操作用了绝对位置 $m$、$n$，但旋转后的 q/k 内积只依赖 $n - m$**。

**复指数视角（更优雅）**：把每组 2D 向量看成复数 $q^{(i)}_m \in \mathbb{C}$，旋转 = 乘 $e^{im\theta_i}$。则

$$\overline{q^{(i)}_m e^{im\theta_i}} \cdot k^{(i)}_n e^{in\theta_i} = \overline{q^{(i)}_m} \cdot k^{(i)}_n \cdot e^{i(n-m)\theta_i}$$

（$\overline{\cdot}$ 是复共轭，对应"$q$ 取转置"。）取实部就是 attention score 在该组上的贡献——同样**只依赖 $n - m$**。复指数视角让公式从一堆 sin/cos 简化成一行复数乘法，是 RoFormer paper §3 的标准推法。

**第 5 步：好处总结**

- **同时是绝对 + 相对**：旋转量 $m\theta_i$ 用到了绝对位置 $m$（这让模型能区分位置 0 与位置 100），但 attention 仅依赖距离 $n - m$（这给了相对位置编码的好性质）
- **外推能力比纯绝对位置好**（虽然仍有限，需要 PI / YaRN 进一步扩展）
- **实现极简**：只在 q, k 投影后加一次旋转（不像 T5 要改 attention 公式或加 bias 表），与 SDPA / FlashAttention 完美兼容
- **不增加参数量**：cos / sin 表是 precompute 的常量，不是 learnable 参数

LLaMA / LLaMA-2/3 / Qwen / Qwen2/2.5 / DeepSeek-V2/V3 / Mistral / Gemma / Phi / GLM-4 全部用 RoPE（或其变体）。**面试时被问"现代 LLM 用什么位置编码"答 RoPE 几乎万无一失**。

### 2.5 RoPE 的高效实现技巧（必懂）

直接按 §2.4 写一个 $d_k \times d_k$ 的块对角旋转矩阵再 matmul 是巨大浪费——实际上每组 $2 \times 2$ 旋转 $\begin{pmatrix} \cos & -\sin \\ \sin & \cos \end{pmatrix} \begin{pmatrix} q_0 \\ q_1 \end{pmatrix} = \begin{pmatrix} q_0 \cos - q_1 \sin \\ q_0 \sin + q_1 \cos \end{pmatrix}$ 可以写成两个张量的 elementwise 乘加：

$$q' = q \odot \cos + \text{rotate\_half}(q) \odot \sin$$

其中 `rotate_half` 把后半段挪到前半段并取负、前半段挪到后半段（这是 HuggingFace 实际实现的"分半对偶"约定，对应"重新排布维度对"）：

```python
def rotate_half(x):
    # x: (..., d_k)，把后半段取负挪到前面
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)
```

> 注：HuggingFace LLaMA 实现把"每 2 维一组"重排成"前 $d_k/2$ 维与后 $d_k/2$ 维一一配对"——这与 §2.4 的"相邻 2 维一组"在数学上等价（都是 $d_k/2$ 个独立 2D 旋转），只是维度排列方式不同。**实战中保持训练与推理用同一个约定即可**。

cos / sin 表 precompute（只在初始化或第一次 forward 时算一次）：

```python
# theta_i = 10000^{-2i/d_k}, i = 0,1,...,d_k/2-1
inv_freq = 1.0 / (10000 ** (torch.arange(0, d_k, 2).float() / d_k))   # (d_k/2,)
t = torch.arange(max_seq).float()                                      # (max_seq,)
freqs = torch.outer(t, inv_freq)                                       # (max_seq, d_k/2)
cos = freqs.cos()[..., None, :].repeat(1, 1, 2).flatten(-2)            # (max_seq, d_k)
sin = freqs.sin()[..., None, :].repeat(1, 1, 2).flatten(-2)            # (max_seq, d_k)
```

之后 forward 时按 token 位置切片即可（详见 §3.2 完整代码）。

### 2.6 ALiBi — Attention with Linear Bias（Press 2021）

ALiBi 的实现比 RoPE 还简单——**完全不动 q, k**，只在 softmax 前的 score 上加一个线性距离惩罚：

$$\text{score}_{ij} = \frac{q_i \cdot k_j}{\sqrt{d_k}} - m_h \cdot |i - j|$$

causal LM 时 $j \le i$，$|i - j| = i - j \ge 0$，惩罚单调。

**关键：每个 head 一个不同的斜率 $m_h$**，且这些斜率是 geometric sequence（公比 $2^{-8/h}$ 之类）：

$$m_h = 2^{-8h/H}, \quad h = 1, 2, \dots, H$$

例如 $H = 8$ 时 $m \in \{2^{-1}, 2^{-2}, 2^{-3}, \dots, 2^{-8}\} = \{0.5, 0.25, 0.125, \dots, 0.0039\}$。**斜率大的 head 关注近处（局部），斜率小的 head 关注远处（全局）**——这种"head 异质性"是 ALiBi 的精髓。注意 $H = 8$ 与 $H = 32$ 的 slope 完全不同，必须按头数重新算（见 §4 第 5 条）。

**优点**：
- 训练时 max_seq = 1024，推理时直接外推到 16k+ 也能稳定工作（不需要 fine-tune），这是 RoPE 直接外推不具备的
- 实现 5 行代码，无需 cos/sin 表

**缺点**（也是它输给 RoPE 的原因）：
- in-distribution 长度（即训练 max_seq 内）效果不如 RoPE，尤其在 1k-32k 这个主流区间
- 距离惩罚是硬性的"线性衰减"，长程依赖（如全文档 retrieval）会被压制
- 不像 RoPE 有 YaRN / LongRoPE 等优雅扩展路径

**用户**：BLOOM（176B，2022）、MPT（MosaicML，2023）、Falcon-7B 早期版本——都是 2022-2023 年的产物，2024 年后新出的开源 LLM 几乎清一色 RoPE。

### 2.7 现代 LLM 位置编码对比表

| 模型 | 位置编码 | max context（公开版本） | 备注 |
|---|---|---|---|
| 原版 Transformer | Sinusoidal | 512 | 无外推能力 |
| GPT-2 | Learned | 1024 | 不能外推 |
| GPT-3 | Learned | 2048 | 同上，被 max_pos 锁死 |
| BERT / RoBERTa | Learned | 512 | 同上 |
| T5 | Relative bucket bias | 512（可外推） | bucket = 32 |
| Transformer-XL | Relative + segment recurrence | 1k+ | 早期长 context 探索 |
| **LLaMA / LLaMA-2** | **RoPE** | 2k / 4k | 业界标杆 |
| **LLaMA-3 / 3.1 / 3.2** | **RoPE + scaled** | 8k / 128k | 用 NTK-aware 扩展 |
| **Qwen / Qwen2/2.5** | **RoPE** | 32k / 128k | YaRN 扩展 |
| **DeepSeek-V2/V3** | **RoPE + decoupled** | 128k | RoPE 与 NoPE 混合 |
| **Mistral / Mixtral** | **RoPE** | 32k | + sliding window |
| **Gemma / Gemma-2** | **RoPE** | 8k / 128k | 标准 RoPE |
| BLOOM | ALiBi | 2048 | 较早设计 |
| MPT | ALiBi | 8k | + FlashAttention |
| Falcon-7B | ALiBi | 2k | Falcon-40B 后改 RoPE |
| Phi-3 | RoPE 变体 | 4k / 128k | RoPE-derivative |
| GLM-4 | RoPE 变体 | 128k | 2D RoPE |

**一眼总结**：2024 年后新出的开源 LLM **100% 用 RoPE 或其变体**。ALiBi 只活在 2022-2023 的"老古董"上。

### 2.8 与 6.5 的衔接：long-context 扩展全是 RoPE 之上的"修补"

直接在远超训练 max_seq 的长度上推理 RoPE 模型 → 性能塌陷（perplexity 暴涨）。原因：旋转角 $m \theta_i$ 在 $m$ 很大、$\theta_i$ 很小的低频组上会进入"训练时从未见过的角度区间"，模型无法 generalize。

业界三种主流"修补"方案，本节只点名（细节在 6.5 详讲）：

- **Position Interpolation (PI, Chen 2023)**：把 $m\theta_i$ 改成 $\frac{m}{s}\theta_i$（$s$ = 缩放因子），相当于把"位置 0 \~ $sL$"线性压到训过的"0 \~ $L$"区间内。简单粗暴，需要短 fine-tune
- **NTK-aware scaling**：调大 base $10000 \to b \cdot 10000$（$b > 1$）→ 高频组（$\theta_i$ 大）几乎不变，低频组（$\theta_i$ 小）的"等效波长"被拉宽。无需 fine-tune 即可外推 2-4×
- **YaRN / LongRoPE**：分频段不同处理（高频 NTK、中频 PI、低频原 RoPE），是当前 SOTA。LongRoPE 通过演化算法搜出每个频段的最优缩放，把 LLaMA-2 7B 直接外推到 2M token

一句话：**所有现代 long-context 扩展都建立在 RoPE 的"旋转角度可调"之上**——这是 ALiBi 没有的优雅性，也是 RoPE 战胜 ALiBi 的最后一击。详见 6.5。

---

## 3. 最小代码示例

### 3.1 手撕 Sinusoidal PE（13 行，历史对照）

```python
import torch
import math

def sinusoidal_pe(max_len: int, d: int) -> torch.Tensor:
    """返回 (max_len, d) 的 sinusoidal 位置编码表（Vaswani 2017）。"""
    pe = torch.zeros(max_len, d)                                      # (L, d)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (L, 1)
    # 分频率：div_term[i] = 1 / 10000^(2i/d)
    div_term = torch.exp(torch.arange(0, d, 2).float() * -(math.log(10000.0) / d))  # (d/2,)
    pe[:, 0::2] = torch.sin(position * div_term)                      # 偶数维 sin
    pe[:, 1::2] = torch.cos(position * div_term)                      # 奇数维 cos
    return pe                                                         # (L, d)

# 使用：x 是 token embedding (B, T, d)
pe = sinusoidal_pe(max_len=1024, d=512)
x = x + pe[:x.size(1)]    # 加在 embedding 上即可
```

历史对照参考。注意 `x = x + pe[:T]`——用切片避免越界。

### 3.2 手撕 RoPE（28 行，HF LLaMA 风格）

```python
import torch
import torch.nn.functional as F

def precompute_rope_cache(d_k: int, max_seq: int, base: float = 10000.0):
    """precompute (cos, sin) 表，shape 各 (max_seq, d_k)。"""
    # theta_i = base^(-2i/d_k), i = 0,1,...,d_k/2-1
    inv_freq = 1.0 / (base ** (torch.arange(0, d_k, 2).float() / d_k))   # (d_k/2,)
    t = torch.arange(max_seq).float()                                    # (max_seq,)
    freqs = torch.outer(t, inv_freq)                                     # (max_seq, d_k/2)
    # 复制一份拼到后面（HF 约定：前 d_k/2 与后 d_k/2 配对）
    emb = torch.cat([freqs, freqs], dim=-1)                              # (max_seq, d_k)
    return emb.cos(), emb.sin()                                          # (max_seq, d_k) ×2

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """把后半段取负挪到前面，前半段挪到后面。"""
    x1, x2 = x.chunk(2, dim=-1)                                          # 各 (..., d_k/2)
    return torch.cat([-x2, x1], dim=-1)                                  # (..., d_k)

def apply_rotary(q: torch.Tensor, k: torch.Tensor,
                 cos: torch.Tensor, sin: torch.Tensor):
    """
    q, k: (B, h, T, d_k)
    cos, sin: (T, d_k)，按 token 位置切好的
    """
    # 关键 4 行：HF LLaMA 标准实现
    q_rot = q * cos + rotate_half(q) * sin                               # (B, h, T, d_k)
    k_rot = k * cos + rotate_half(k) * sin                               # (B, h, T, d_k)
    return q_rot, k_rot

# === 用法演示 ===
B, h, T, d_k = 2, 8, 16, 64
q = torch.randn(B, h, T, d_k)
k = torch.randn(B, h, T, d_k)

cos_full, sin_full = precompute_rope_cache(d_k, max_seq=1024)
cos, sin = cos_full[:T], sin_full[:T]                                    # 切片到当前 seq

q_rot, k_rot = apply_rotary(q, k, cos, sin)
# 之后正常做 attention：scores = q_rot @ k_rot.transpose(-2,-1) / d_k**0.5
scores = (q_rot @ k_rot.transpose(-2, -1)) / (d_k ** 0.5)                # (B, h, T, T)
```

**关键点逐行**：
- `precompute_rope_cache`：cos / sin 表是常量，**只算一次**——典型 HF 实现把它注册成 `register_buffer` 跟随 `.to(device)` 一起移动
- `rotate_half`：HF 约定"前半负后半"——这与 RoFormer 原 paper 的"相邻 2 维"约定数学等价但维度排布不同
- `apply_rotary` 的核心 4 行：`q * cos + rotate_half(q) * sin`——这就是 §2.4 那个 $2 \times 2$ 旋转矩阵作用的等价 elementwise 形式
- **shape 对齐**：cos/sin 是 `(T, d_k)`，q/k 是 `(B, h, T, d_k)`——PyTorch broadcast 会自动把 cos/sin 广播到 batch 与 head 维（前面补 1 即可：`cos[None, None, :, :]`，HF 实际写法略有不同但语义一致）
- **核心优势**：整个 RoPE 实现没碰 attention 公式——把 `apply_rotary` 加在 q/k 投影之后、attention 计算之前即可，与 SDPA / FlashAttention 完美兼容

### 3.3 验证 RoPE 内积只依赖相对位置（10 行 sanity test）

```python
torch.manual_seed(0)
d_k, max_seq = 64, 128
cos, sin = precompute_rope_cache(d_k, max_seq)

# 同一个 q, k（无 batch 无 head）；放在两组不同绝对位置但相同相对距离
q_raw = torch.randn(1, 1, 1, d_k)
k_raw = torch.randn(1, 1, 1, d_k)

def score_at(m, n):
    q_m, _ = apply_rotary(q_raw, k_raw, cos[m:m+1], sin[m:m+1])
    _, k_n = apply_rotary(q_raw, k_raw, cos[n:n+1], sin[n:n+1])
    return (q_m * k_n).sum().item()                                   # 内积

# 距离均为 5，但绝对位置不同
print(score_at(3, 8), score_at(20, 25), score_at(50, 55))             # 应当数值很接近
```

跑出来三个数应当近似相等（浮点 round-off 内）——这就是 RoPE 核心定理的实证。

### 3.4 ALiBi bias 矩阵构造（13 行）

```python
import torch
import math

def alibi_slopes(num_heads: int) -> torch.Tensor:
    """每 head 的斜率 m_h，geometric sequence。"""
    # 标准设计：n = 2^(8/h)，slope = 1/n, 1/n^2, ..., 1/n^h
    n = 2 ** (8 / num_heads)
    return torch.tensor([1.0 / (n ** (i + 1)) for i in range(num_heads)])  # (h,)

def alibi_bias(num_heads: int, max_seq: int) -> torch.Tensor:
    """构造 (h, T, T) 的 ALiBi bias，causal 时上三角不需要可省略。"""
    slopes = alibi_slopes(num_heads)                                    # (h,)
    pos = torch.arange(max_seq)
    rel = pos[None, :] - pos[:, None]                                   # (T, T)，j - i
    # rel < 0 是上三角（未来位置），causal 会被 mask 成 -inf；这里只关心 i >= j
    bias = -slopes[:, None, None] * rel.abs()[None, :, :]               # (h, T, T)
    return bias                                                         # 加到 score 上即可

# 用法：score = q @ k.T / sqrt(d_k) + alibi_bias(h, T)，再 softmax
bias = alibi_bias(num_heads=8, max_seq=512)                             # (8, 512, 512)
print(bias.shape, alibi_slopes(8))
```

注意 `alibi_slopes(8)` 与 `alibi_slopes(32)` 完全不同——这是 §4 第 5 条要警惕的工程坑。

---

## 4. 工程踩坑与经验

- ❗ **RoPE 实现最经典 bug：把 q/k 整体旋转而不是按 head 内 d_k 维分组**。RoPE 的旋转是"每 2 维一组的独立 2D 旋转"，必须在 head 维内进行（即 $(B, h, T, d_k)$ 张量的最后一维 $d_k$ 上分组）。新手常见错误：把 $d = h \cdot d_k$ 当成单个向量整体旋转、或者把 sin/cos 对齐到错误的维度。表现是模型 loss 完全不下降、attention map 退化成无序噪声。**修复**：永远在 q/k 投影 + reshape 到 $(B, h, T, d_k)$ 之后再调用 `apply_rotary`。

- ❗ **RoPE 的 cos/sin precompute 表必须覆盖最大 seqlen，超出 → IndexError**。HF LLaMA 默认 precompute 到 `max_position_embeddings`（如 4096）；如果 inference 时序列拉到 8192 直接 index out of bound。**生产规则**：precompute 到 `max_seqlen × 4` 留余量，或者在 `apply_rotary` 里做"按需扩展"——发现传入的 `T` 超出当前 cache 长度时动态再 precompute 一段。LLaMA-2 7B → 32k context 的扩展时这个细节经常掉坑。

- ❗ **RoPE 的 base = 10000 是经验值；为 long-context 必须调（NTK / YaRN）**。在 8k+ 长度上直接用 default base 推理 RoPE 模型 → perplexity 暴涨 2-10×。NTK-aware 的做法是把 base 调大（如 $10000 \to 50000$），让低频组的"等效波长"覆盖更长的位置范围。LLaMA-3 / Qwen2 的 long-context 版本都在 config 里能看到 `rope_theta` 这个字段被设成 500000+。**面试常考**："你怎么把一个 4k context 的 RoPE 模型扩展到 32k"——答案就是 NTK-aware / YaRN。

- ❗ **Position id 在 packing / KV cache resume / left padding 时必须特别处理**。RoPE 的旋转角依赖 token 的 position id，所以下面三种场景必须显式传 position id 而不是用 `arange(T)`：（1）Sample packing（3.3 节）把多条样本拼到一个序列里，每条要从 0 重新开始；（2）KV cache resume 时新 token 的 position id 应当是 `cache_len + new_offset`；（3）Left padding 时左边 pad 不算位置，第一个真实 token 的 position id 应当从 0 开始。这些场景搞错就是 RoPE 旋转角错了 → attention 全乱 → 生成胡言乱语。HF generate 内部维护了 `position_ids` 张量，自己手写 KV cache 时必须复刻这个机制（参见 4.7 节）。

- ❗ **ALiBi 的 head slope 是 geometric sequence，不同 head 数要重新算**。$H = 8$ 时 slope $\in \{2^{-1}, 2^{-2}, \dots, 2^{-8}\}$；$H = 32$ 时 slope $\in \{2^{-0.25}, 2^{-0.5}, \dots, 2^{-8}\}$（公式 $m_h = 2^{-8h/H}$）。新手常见错误：把 $H = 8$ 的 slope 表硬塞给 $H = 32$ 模型——结果 32 个 head 只有 8 种斜率重复 4 次，"head 异质性"完全失效。**修复**：永远调用 `alibi_slopes(num_heads)` 函数动态生成，不要 hardcode。

- ❗ **Sinusoidal PE 与 token embedding 相加时尺度不一致 → 必须把 embedding 乘 $\sqrt{d}$**。Vaswani 2017 §3.4 显式提到："we multiply those weights by $\sqrt{d_{\text{model}}}$"——因为 nn.Embedding 默认初始化方差是 $1/d$，而 sinusoidal PE 的方差是 $1/2$（sin/cos 各贡献 $1/2$）；不平衡尺度会让位置信号要么被 token 信号淹没要么反过来。**修复**：`x = self.embed(tokens) * math.sqrt(self.d_model) + pe`。**注意**：用 RoPE / ALiBi 的现代 LLM 不需要这一步——RoPE 不动 embedding 加性而是改 attention，scale 不存在；学 nanoGPT 实现时这点容易混淆。

- ❗ **不同位置编码方案的 model 不能相互 LoRA / merge / 嫁接**。RoPE 模型（LLaMA）的 q/k 投影后会被旋转，sinusoidal PE 模型的 q/k 是"原样"——两者 weight space 的语义完全不同。强行把 LLaMA 训出来的 LoRA adapter 套到 GPT-2（learned PE）上 → 必然崩盘。同理 LLaMA 与 BLOOM（ALiBi）之间的权重也没法直接 merge。**实战规则**：LoRA / 蒸馏 / 权重合并必须在**位置编码相同**的模型族内进行（同为 LLaMA 系、同为 Qwen 系等）。

- ❗ **RoPE 与 KV cache 配合时，cos/sin 的 indexing 必须用"全局位置"不是"cache slot index"**。decoding 时把新 token 的 q/k 按位置 $m = \text{past\_len}$ 旋转、把 cache 里所有历史 k 按各自的原位置旋转。常见 bug：对新 token 用 $m = 0$（"它是 cache 的第 0 个 slot"），旋转角错 → attention 全乱。**正确实现**：cache 里存"已旋转的 k"，新 token 的 k 按当前 past_len 旋转后 append 进去；q 永远只算一个新 token、按当前位置旋转一次。LLaMA 的 HF 实现 `apply_rotary_pos_emb` 直接接收 `position_ids` 参数避免这个坑。

- ❗ **bf16 / fp16 下 RoPE 的 cos/sin 表用 fp32 精度存**。cos/sin 在 $m\theta_i$ 很大时（如 $m = 100k$、$\theta_i = 10^{-3}$，乘积 100）周期性会让 fp16 精度丢失成 0/1 离散值，导致旋转角误差累积。**最佳实践**：cos/sin precompute 永远用 fp32，apply_rotary 时再 cast 回 q/k 的 dtype。HF LLaMA 实现就是这么做的，源码里能看到 `cos = cos.to(q.dtype)` 这种显式 cast。

---

## 5. 经典 paper

- **Vaswani et al., 2017 — Attention Is All You Need** — sinusoidal PE 的原典，§3.5 "Positional Encoding" 直接给出公式与"PE 是相对位置的线性函数"的设计动机。本节 §2.1 与它一致。这篇 11 页 paper 在 4.1 已经必读过一遍，看 PE 这节要重点关注 footnote 5（学过的 learned PE 与 sinusoidal 实证差不多，但选了 sinusoidal 因为可外推）——实际后来发现两者都不能外推，PE 设计被推翻是后面相对/旋转 PE 的契机。
- **Shaw, Uszkoreit & Vaswani, 2018 — Self-Attention with Relative Position Representations** — 相对位置编码的起点。提出在 attention 公式里给 K 与 V 都加上一个位置依赖 bias，是 T5 bucket bias 与 Transformer-XL 的思想源头。读它能体会"为什么相对位置才是更本质的设计选择"——本节 §2.3 的精神祖师爷。
- **Su et al., 2021 — RoFormer: Enhanced Transformer with Rotary Position Embedding** — RoPE 必引。论文用复指数视角推出 §2.4 第 4 步那个核心定理 $\langle q'_m, k'_n \rangle$ 仅依赖 $n - m$，简洁优雅。读 §3 "Method" 全部 + §3.4 的复指数推导（30 分钟搞定），是面试时被问"推一遍 RoPE"的标准答案。
- **Press, Smith & Lewis, 2021 — Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation** — ALiBi 必引。卖点很直接：在 1024 长度训练，能直接外推到 16k+ 不掉点。读它体会"位置编码不一定要嵌在 q/k 里，加在 score 上也行"——是一种与 RoPE 完全正交的设计哲学。虽然 ALiBi 已经被 RoPE 时代抛在身后，但它的"外推哲学"启发了后续 NoPE（无位置编码）等方向。
- **加分阅读**：
  - Chen et al., 2023 — Extending Context Window of LLMs via Position Interpolation（PI）：第一个把 RoPE long-context 扩展系统化的 paper，6.5 详讲
  - bloc97 / NTK-aware scaling（reddit 2023 的一篇博文，后被 paper 引用）：把 PI 改成调 base，无需 fine-tune
  - Peng et al., 2023 — YaRN: Efficient Context Window Extension of Large Language Models：当前长 context RoPE 扩展的 SOTA 之一，6.5 详讲

---

## 6. 自测与面试题

**Q1（推导）**：解释为什么 RoPE 的内积 $\langle R_m q, R_n k \rangle$ 仅依赖相对位置 $n - m$。用旋转矩阵性质 $R_m^\top R_n = R_{n-m}$ 推一遍。

<details>
<summary>Answer sketch</summary>

要点：

- RoPE 把 q, k 按各自位置 $m$、$n$ 用旋转矩阵 $R_m$、$R_n$ 旋转：$q'_m = R_m q$，$k'_n = R_n k$
- 关键性质：旋转矩阵的转置等于逆向旋转（$R_m^\top = R_{-m} = R_m^{-1}$，因为旋转矩阵正交），且旋转复合等于角度相加：$R_m^\top R_n = R_{-m} R_n = R_{n-m}$
- 由此推：

$$\langle q'_m, k'_n \rangle = (R_m q)^\top (R_n k) = q^\top R_m^\top R_n k = q^\top R_{n-m} k$$

- 右边只剩 $R_{n-m}$，**只依赖相对位置 $n - m$**，与 $m$、$n$ 各自的绝对值无关
- 这意味着任意两个 token 在序列中的"绝对位置"不影响它们的 attention score；只有它们之间的"距离"影响

加分：
- 能用复指数视角更优雅地推一遍：把每组 2D 看成复数，$q^*_m k_n e^{i(n-m)\theta}$ 取实部就是 score
- 能指出"虽然 attention 只依赖相对位置，但旋转操作仍用了绝对位置 $m$"——这就是 RoPE "兼具绝对 + 相对" 的精髓
- 能解释为什么这比 Shaw 2018 的"加 bias"方案更优：RoPE 不改 attention 公式，与 SDPA / FlashAttention 直接兼容

</details>

**Q2（实现）**：写出 `apply_rotary(q, k, cos, sin)` 的核心 4 行 PyTorch（用 `rotate_half` 技巧）。说明 q, k, cos, sin 各自的 shape 与 broadcast 规则。

<details>
<summary>Answer sketch</summary>

核心 4 行：

```python
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

def apply_rotary(q, k, cos, sin):
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot
```

shape 与 broadcast：
- q, k：`(B, h, T, d_k)`（标准 multi-head 形状）
- cos, sin：`(T, d_k)`（从 precompute 表按 token 位置切片得到）
- broadcast：cos/sin 在 batch 与 head 维上自动 broadcast；HF 实际实现常显式写 `cos[None, None, :, :]` 让形状对齐为 `(1, 1, T, d_k)` 便于阅读

为什么是 `q * cos + rotate_half(q) * sin`：
- 数学上对应每对 2 维 $(q_0, q_1)$ 做 2D 旋转 $\begin{pmatrix}\cos & -\sin \\ \sin & \cos\end{pmatrix} \begin{pmatrix}q_0 \\ q_1\end{pmatrix}$
- HF 约定把"前 d_k/2 维"与"后 d_k/2 维"配对（不是相邻 2 维），所以 `rotate_half` 把后半负挪到前面
- 这与 RoFormer paper 的"相邻 2 维"约定数学等价、维度排布不同——只要训练与推理用同一个约定即可

加分：
- 能指出 cos/sin 必须用 fp32 精度 precompute，再 cast 回 q/k dtype（避免 long context 下精度丢失）
- 能解释为什么这个 elementwise 写法比"显式构造 d_k × d_k 的块对角旋转矩阵再 matmul"快几个数量级（避免 $O(d_k^2)$ matmul）
- 能说出 HF LLaMA 源码里 `apply_rotary_pos_emb` 还接收 `position_ids` 参数，用于 KV cache resume 时正确切片 cos/sin

</details>

**Q3（trade-off）**：RoPE vs ALiBi 在 (in-distribution 长度效果 / 外推能力 / 实现复杂度 / long-context 扩展友好度) 4 个维度对比；为什么现代 LLM 几乎全选 RoPE？

<details>
<summary>Answer sketch</summary>

四维对比表：

| 维度 | RoPE | ALiBi |
|---|---|---|
| in-distribution 长度效果（训练 max_seq 内） | **更好**（attention 表达力强） | 略差（线性距离惩罚太硬） |
| 外推能力（直接超出训练长度） | 较弱（需要 PI / NTK / YaRN） | **强**（直接外推到 16k+ 不掉点） |
| 实现复杂度 | 中（需要 precompute cos/sin 表 + apply_rotary） | **简单**（5 行加一个 bias 矩阵） |
| long-context 扩展友好度 | **极好**（PI / NTK / YaRN / LongRoPE 完整生态） | 差（没有类似优雅的扩展路径） |

为什么现代 LLM 几乎全选 RoPE：

1. **2024+ 主流场景是"in-distribution 长 context"而非"短训长推"**：训练时直接用 32k+ 长度（数据足够），ALiBi 的"外推友好"卖点失去价值，反而它在 in-distribution 上的表达力劣势被放大
2. **YaRN / LongRoPE 解决了 RoPE 的外推问题**：曾经 ALiBi 唯一的优势（无需 fine-tune 就能外推）已经被 RoPE + YaRN 在 fine-tune 后超越；LongRoPE 甚至能把 LLaMA-2 7B 外推到 2M token
3. **RoPE 与 SDPA / FlashAttention / KV cache 完美兼容**：旋转加在 q/k 投影之后、attention 计算之前，对底层 kernel 完全透明；ALiBi 需要 attention kernel 支持任意 bias（FlashAttention 2.0 才有）
4. **RoPE 的 inductive bias 更软**：旋转角度的频率谱给了模型"区分长短距离"的能力（高频组捕捉局部、低频组捕捉全局）；ALiBi 的线性距离惩罚是硬性"远 = 弱"，对长程 retrieval 类任务（needle in haystack）不友好
5. **历史路径依赖**：LLaMA / Qwen / DeepSeek / Mistral / Gemma 全用 RoPE → 整个生态（LoRA / 量化 / 推理框架）都围绕 RoPE 优化 → 新模型继续选 RoPE 是最低成本路径

加分：
- 能说出 ALiBi 仍有"早期 BLOOM / MPT / Falcon-7B"的代表用户，2022-2023 年是它的高光期
- 能指出 RoPE 的 base = 10000 是 hyperparameter，long-context 扩展时调到 500000+
- 能引用具体数字：YaRN 把 LLaMA-2 7B 从 4k 扩展到 128k 后 perplexity 几乎不掉、LongRoPE 扩到 2M

</details>

---

## 7. 延伸阅读

- [Su et al. 2021 — RoFormer (arXiv)](https://arxiv.org/abs/2104.09864) — RoPE 原 paper，§3 完整推导值得抄一遍
- [Press et al. 2021 — Train Short, Test Long: ALiBi (arXiv)](https://arxiv.org/abs/2108.12409) — ALiBi 原 paper，正反两面都该读
- [Shaw et al. 2018 — Relative Position Representations (arXiv)](https://arxiv.org/abs/1803.02155) — 相对位置编码起点
- [HuggingFace LLaMA `modeling_llama.py` 中的 `apply_rotary_pos_emb`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) — 工业级 RoPE 实现，本节 §3.2 就是它的精炼版
- [Chen et al. 2023 — Position Interpolation (arXiv)](https://arxiv.org/abs/2306.15595) — RoPE long-context 扩展的第一篇 paper，6.5 节详讲
- [Peng et al. 2023 — YaRN (arXiv)](https://arxiv.org/abs/2309.00071) — 当前 RoPE 长 context 扩展 SOTA 之一，6.5 节详讲
- [EleutherAI Blog — RoPE 可视化](https://blog.eleuther.ai/rotary-embeddings/) — 全网最好的 RoPE 直觉文章，配 §1 mental model 食用
- [Lilian Weng — The Transformer Family v2.0 §位置编码部分](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/) — 位置编码的全谱综述
- 推荐继续读本教程的 **4.4 节《LayerNorm / RMSNorm 与 Pre-LN vs Post-LN》**——继续走 Transformer 内部组件链；之后 4.6 完整 decoder-only 实现会用到本节的 RoPE
- 推荐继续读本教程的 **6.5 节《Long-context：Position Interpolation / NTK / YaRN / LongRoPE》**——本节最后一段铺垫的细节全部展开
