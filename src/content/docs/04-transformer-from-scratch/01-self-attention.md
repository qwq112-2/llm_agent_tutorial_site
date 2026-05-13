---
title: "4.1 Self-attention：QKV 与 scaled dot-product"
description: "把 Bahdanau cross-attention 的\"decoder 看 encoder\"推广成\"序列内每个 token 用自己的 query 在所有 token 的 key 上检索、再按相关度从所有 token 的 value 加权汇总\"——这就是 self-attention，公式 $\\\\text{Attention}=\\\\text{softmax}V$。本节把 QKV 的物理意义、scale"
---

> ⏱ 预计阅读 45 分钟 ｜ 难度 ★★ 🔥 必考 ｜ 前置：2.4 Bahdanau attention、3.3 batching 与 mask

## 一句话本节讲什么

把 Bahdanau cross-attention 的"decoder 看 encoder"推广成"序列内每个 token 用自己的 query 在所有 token 的 key 上检索、再按相关度从所有 token 的 value 加权汇总"——这就是 self-attention，公式 $\text{Attention}(Q,K,V)=\text{softmax}(QK^\top/\sqrt{d_k})V$。本节把 QKV 的物理意义、scaled dot-product 的逐步推导、$\sqrt{d_k}$ 缩放的方差解释、causal mask、复杂度 trade-off、置换不变性、以及一份手撕 single-head 的最小 PyTorch 实现讲透——这是后面 80 节（multi-head、位置编码、KV cache、PPO、推理引擎、Agent RL）的共同基石。

---

## 1. Mental model（直觉）

### 1.1 从 cross-attention 到 self-attention 的飞跃

2.4 节末尾埋下的种子：Bahdanau attention 是 **cross-attention**——decoder 在生成每一个 target token 时去看 encoder 的所有 hidden states，按相关度加权汇总成 context vector。Vaswani 2017 把这个机制做了一次"内化"：

> 如果序列**内部**每一个 token 都能用相同的机制看其他 token（包括自己），那就完全不需要 RNN 来串行聚合上下文了。

这就是 self-attention：**同一个序列既扮演 query 来源，又扮演 key 与 value 来源**。每个位置不再被动接受 RNN 一格一格传过来的信息，而是主动"广播一个查询、检索全序列"。

一句话直觉，记牢这一句：

> **每个 token 用自己的 query，在所有 token 的 key 上做相似度检索，按相关度从所有 token 的 value 里加权汇总，得到自己的新表示。**

ASCII 直觉图——一句 5 token 的句子 `[The cat sat on mat]`，看 `sat` 这个位置在做什么：

```
                    Q (sat 的 query)
                          │
         ┌──────┬─────────┼─────────┬──────┐
         ▼      ▼         ▼         ▼      ▼
       K(The) K(cat)   K(sat)    K(on)  K(mat)        ← 所有 token 的 key
         │      │         │         │      │
       打分    打分      打分      打分    打分          ← Q · K  得 5 个相似度
         │      │         │         │      │
         └──── softmax 归一成权重 α (和=1) ────┘
                          │
         ┌──────┬─────────┼─────────┬──────┐
         ▼      ▼         ▼         ▼      ▼
       V(The) V(cat)   V(sat)    V(on)  V(mat)        ← 所有 token 的 value
         │      │         │         │      │
         └──── 加权求和 = sat 的新表示 ─────┘
```

所有 5 个位置**并行**做同样的事——每个位置都广播一个 query、收一份 5 个 key 的相似度、用这 5 个权重去聚合 5 个 value。一次 forward 全做完，是矩阵乘法而非 for 循环。这与 RNN 必须 timestep 串行形成根本对比，也是 Transformer 训练吞吐能拉到 RNN 几十上百倍的根因。

### 1.2 图书馆类比

把 self-attention 想成一次图书馆检索：

- **Query（Q）**：你心里想找的关键词。比如"如何手推 BP"。
- **Key（K）**：书架上每本书贴的标签。比如《深度学习》、《菜谱》、《历史》。
- **Value（V）**：每本书真正的内容。如果某本书被高度匹配，它的内容会被打开来读。

你拿着 Q 与每本书的 K 做相似度匹配，软地（soft）按相似度权重把所有书的 V 加权读一遍。**Q/K/V 三件套是同一份输入 $X$ 经过三个不同的线性投影得到的**——同样的 token，从不同视角分别变成"我想找什么"、"我能提供什么"、"如果被选中我贡献什么"。这种"同源三投影"是 self-attention 与 Bahdanau cross-attention 在公式形式上最显著的差异。

### 1.3 为什么 self-attention 比 RNN 强（30 秒版）

| 维度 | RNN | Self-attention |
|---|---|---|
| **并行性** | 时间维必须串行（$h_t$ 依赖 $h_{t-1}$） | 一次 matmul 全序列并行 |
| **长程依赖** | 需要走 $\|i-j\|$ 步，梯度衰减 | 任意两 token **一跳直连** |
| **GPU 友好度** | 难（kernel 串行、batch 利用率低） | 极高（全是大 matmul） |
| **复杂度** | $O(nd^2)$ 时间 | $O(n^2 d)$ 时间，$O(n^2)$ 空间 |

代价是 $O(n^2)$ 的二次复杂度——长序列时这会反咬一口，催生了 FlashAttention、KV cache 压缩（GQA / MLA）、SSM（Mamba）等所有"如何与 $O(n^2)$ 抗争"的研究方向。详见本节 §4 与后续 5.x、7.6。

---

## 2. 公式与原理

### 2.1 输入与三投影

设输入序列已经经过 embedding，得到

$$X \in \mathbb{R}^{n \times d}$$

其中 $n$ 是序列长度（token 数）、$d$ 是 hidden dimension（典型 768 / 4096 / 8192）。**先忽略 batch 维**，最后再讨论广播。

定义三个可学习投影矩阵：

$$W_Q \in \mathbb{R}^{d \times d_k}, \quad W_K \in \mathbb{R}^{d \times d_k}, \quad W_V \in \mathbb{R}^{d \times d_v}$$

通常单头时取 $d_k = d_v = d$；进入 multi-head 后每个 head 取 $d_k = d_v = d / h$（$h$ 为头数）——这一点 4.2 详讲，本节按 $d_k = d_v$ 处理即可。

三投影：

$$Q = X W_Q \in \mathbb{R}^{n \times d_k}, \quad K = X W_K \in \mathbb{R}^{n \times d_k}, \quad V = X W_V \in \mathbb{R}^{n \times d_v}$$

物理意义复述一遍：

- $Q$ 第 $i$ 行 $q_i$：第 $i$ 个 token 想找什么（query 向量）
- $K$ 第 $j$ 行 $k_j$：第 $j$ 个 token 能提供什么（key 向量）
- $V$ 第 $j$ 行 $v_j$：第 $j$ 个 token 如果被选中贡献什么（value 向量）

**$W_Q, W_K, W_V$ 是模型参数，由训练学出来**——这是 self-attention 与传统 attention 的关键差异：相似度不是写死的（如余弦相似度），而是通过两个学出来的投影后再做内积，模型自己决定"哪种投影下两个 token 才算相似"。

### 2.2 Scaled Dot-Product Attention 五步推导

**第 1 步：相似度矩阵**

$$S = QK^\top \in \mathbb{R}^{n \times n}, \quad S_{ij} = q_i \cdot k_j = \sum_{p=1}^{d_k} Q_{ip} K_{jp}$$

$S$ 的第 $i$ 行就是"第 $i$ 个 token 对所有 $n$ 个 token（包括自己）的原始相似度"。一次 matmul 把 $n^2$ 个相似度全算完。

**第 2 步：缩放**

$$S' = \frac{S}{\sqrt{d_k}}$$

为什么除以 $\sqrt{d_k}$ 不是 $d_k$ 不是 $1$？这是 §2.3 的整段推导。

**第 3 步：mask（可选）**

把不允许看的位置加 $-\infty$（工程实现常用 `-1e9` 或 `-65504` 在 fp16 下；**bf16 / fp32 直接用 `float('-inf')` 更安全**）：

$$S''_{ij} = S'_{ij} + M_{ij}, \quad M_{ij} \in \{0, -\infty\}$$

decoder-only LLM 训练时 $M$ 是上三角（位置 $i$ 不能看 $j > i$），加 padding mask 时 pad 位置整列再屏蔽。详见 3.3 与本节 §2.4。

**第 4 步：按行 softmax**

$$A = \text{softmax}(S'') \in \mathbb{R}^{n \times n}, \quad A_{ij} = \frac{\exp(S''_{ij})}{\sum_{j'=1}^{n} \exp(S''_{ij'})}$$

$A$ 的每一行是一个概率分布——第 $i$ 行就是"第 $i$ 个 token 对所有 $n$ 个 token 的注意力权重，和为 1"。$-\infty$ 的位置经 softmax 后变成严格 0。

**第 5 步：加权 V**

$$\text{Attention}(Q, K, V) = A V \in \mathbb{R}^{n \times d_v}$$

输出第 $i$ 行 = $\sum_j A_{ij} v_j$，是"按权重 $A_{ij}$ 把所有 $n$ 个 value 向量加权求和的结果"——也就是第 $i$ 个 token 的新表示。

完整公式（这是本节最该背的一行）：

$$\boxed{\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V}$$

**形状全程总结**：$X: (n, d) \to Q,K: (n, d_k), V: (n, d_v) \to S: (n, n) \to A: (n, n) \to \text{out}: (n, d_v)$。带 batch 时全部前面加 $B$，加 multi-head 后再加 $h$，但核心五步不变。

### 2.3 $\sqrt{d_k}$ 的来源（必考点 / Vaswani 2017 §3.2.1）

为什么除以 $\sqrt{d_k}$ 这件事，几乎每场 LLM 算法面试都会问。完整推导一次走透。

**前提假设**：$q, k$ 是两个 $d_k$ 维向量，每个分量 $q_i, k_i$ 都是均值 0、方差 1 的独立随机变量（这是 Xavier / Kaiming 初始化下、加 LayerNorm 之后的合理近似）。

**计算内积的均值与方差**：

$$q \cdot k = \sum_{i=1}^{d_k} q_i k_i$$

由独立性、$\mathbb{E}[q_i] = \mathbb{E}[k_i] = 0$：

$$\mathbb{E}[q_i k_i] = \mathbb{E}[q_i]\,\mathbb{E}[k_i] = 0 \;\Rightarrow\; \mathbb{E}[q \cdot k] = 0$$

方差用独立性 $\text{Var}(X+Y) = \text{Var}(X) + \text{Var}(Y)$：

$$\text{Var}(q_i k_i) = \mathbb{E}[q_i^2 k_i^2] - 0 = \mathbb{E}[q_i^2]\,\mathbb{E}[k_i^2] = 1 \cdot 1 = 1$$

$$\text{Var}(q \cdot k) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i) = d_k$$

**关键结论**：内积 $q \cdot k$ 的方差是 $d_k$，**不是 1**。

**为什么这是问题**：softmax 对输入的 magnitude 极其敏感。考虑两个 score $a$ 和 $a + \Delta$，softmax 之后比例是 $e^\Delta$；当 $a$ 自身 magnitude 已经很大时（比如 $d_k = 64$，方差 64，标准差 8，单个 score 容易到 $\pm 20$），softmax 输出会**急剧饱和**——最大的那个接近 1，其他接近 0。

数值例子：$d_k = 1$ 时方差 1，score 大致在 $[-3, 3]$ 之间，softmax 输出像 $[0.4, 0.3, 0.2, 0.1]$，分布平滑；$d_k = 512$ 时方差 512、std ≈ 23，score 容易拉到 $[-70, 70]$，softmax 输出几乎是 one-hot，类似 $[1.0, 10^{-30}, 10^{-30}, \dots]$。

softmax 在饱和区的梯度近似 $p_i (1 - p_i) \approx 0$（当 $p_i \to 1$ 或 $p_i \to 0$）——这就是 **vanishing gradient**。整个 attention 层学不动，loss 不下降。

**修复**：除以 $\sqrt{d_k}$ 把方差拉回 1：

$$\text{Var}\!\left(\frac{q \cdot k}{\sqrt{d_k}}\right) = \frac{1}{d_k} \cdot d_k = 1$$

scaled 之后无论 $d_k$ 多大，单个 score 的标准差都稳定在 1 附近，softmax 不会进入饱和区，梯度健康。这就是 $\sqrt{d_k}$ 的全部来历——一个**非常朴素的方差归一化**，不是某种深奥的数值技巧。

**为什么是 $\sqrt{d_k}$ 不是 $d_k$**：标准差 = $\sqrt{\text{方差}}$，归一化方差到 1 应当除以标准差。如果除以 $d_k$，方差会被压成 $1/d_k$，softmax 会过于"扁平"（所有位置概率几乎相等），attention 退化成"对所有位置取平均"，模型表达能力反而下降。

**Vaswani 原文的实证验证**：paper §3.2.1 footnote 4 直接做了 $d_k = 64$ 时 with-scaling vs without-scaling 的对照实验——without-scaling 的训练曲线明显更慢更差。后来 Karpathy 在 nanoGPT 注释里也复现过这一点。

### 2.4 Causal mask（与 3.3 呼应）

decoder-only LLM 训练时一个核心约束：位置 $t$ 只能用 $\le t$ 的位置预测下一个 token，否则等于训练时偷看未来 = cheating（也叫 label leakage）。

实现：在 softmax 之前把 score 矩阵 $S'$ 的**严格上三角**（$j > i$ 部分）加 $-\infty$：

```
   j=0  1  2  3  4
i=0 [ 0 -∞ -∞ -∞ -∞ ]    位置 0 只能看自己
i=1 [ 0  0 -∞ -∞ -∞ ]    位置 1 看 0,1
i=2 [ 0  0  0 -∞ -∞ ]    位置 2 看 0,1,2
i=3 [ 0  0  0  0 -∞ ]    位置 3 看 0,1,2,3
i=4 [ 0  0  0  0  0 ]    位置 4 看全
```

PyTorch 推荐写法：

```python
causal = torch.tril(torch.ones(n, n, dtype=torch.bool))    # 下三角 True
S = S.masked_fill(~causal, float('-inf'))                  # 上三角填 -inf
```

或者直接用 `F.scaled_dot_product_attention(q, k, v, is_causal=True)`，PyTorch 2.0+ 内部用 FlashAttention backend、不会显式构造 $(n, n)$ mask 矩阵。两种方式都要熟练。3.3 节已经把 padding mask + causal mask 合并的工程细节讲完了，本节不重复，但要记得：**实战训练时 final_mask = causal AND padding_mask**。

### 2.5 复杂度分析（关键 trade-off）

self-attention 的代价是它的"全局可见性"换来的——任意两 token 直连，意味着要算 $n \times n$ 个相似度。

**时间复杂度**：

- 三投影 $XW_Q, XW_K, XW_V$：$O(n d^2)$（其实是 $O(n d \cdot d_k)$，单头时 $d_k = d$）
- $QK^\top$：$O(n^2 d_k)$
- $A V$：$O(n^2 d_v)$
- **总计**：$O(n d^2 + n^2 d)$

当 $n \gg d$（长序列）时主导项是 $O(n^2 d)$；当 $n \ll d$（短序列、宽 hidden）时主导项是 $O(n d^2)$。LLM 训练里长 context 越来越流行，主导项基本都是 $n^2$ 这一项。

**空间复杂度**：

- attention matrix $A \in \mathbb{R}^{n \times n}$：$O(n^2)$
- 多头时每头一份，总 $O(h \cdot n^2)$

具体感受一下：

| n | A 矩阵 size（bf16，单头） | 32 head 总共 |
|---|---|---|
| 2k | 8 MB | 256 MB |
| 8k | 128 MB | 4 GB |
| 32k | 2 GB | 64 GB |
| 128k | 32 GB | 1 TB（爆显存） |

**128k context 的 attention matrix 单卡装不下**——这就是 FlashAttention（5.3 / 7.6）的根本动机：不显式存这个 $n \times n$ 矩阵，而是分块在 SRAM 里 online softmax，省掉 $O(n^2)$ 的 HBM 访问。也是 GQA / MQA / MLA（5.2）压缩 KV cache、SSM（5.5）走 $O(n)$ 路线的动机。

**vs RNN**：

- RNN：时间 $O(n d^2)$（每步 $O(d^2)$、共 $n$ 步），但**无法并行**——必须等 $h_{t-1}$ 算完才能算 $h_t$。
- self-attention：时间 $O(n^2 d + n d^2)$，看似更高，但**完全并行**——一次 matmul 把所有 $n$ 个位置算完。GPU 友好度天差地别。

直观的"挂钟时间"对比：训练一个 124M 参数的模型，序列长度 1024、batch 32，self-attention 在 A100 上大约 100 ms / step；同尺寸 LSTM 即使理论 FLOPs 更低，挂钟时间也要 800-1500 ms，因为 LSTM 的 timestep 串行让 GPU 利用率只有 20-30%，而 attention 利用率能到 80%+。

### 2.6 关键性质

**性质 1：置换不变性（permutation equivariance）**

如果不加位置编码，self-attention 对**输入 token 顺序完全不敏感**——把 `[The cat sat]` 和 `[sat cat The]` 喂进去，输出会是完全一样的多重集（multiset），只是顺序对应着调换。

形式化：对任意置换矩阵 $P$，

$$\text{Attention}(PQ, PK, PV) = P \cdot \text{Attention}(Q, K, V)$$

这是 self-attention 的"特征"也是"bug"——好处是它天然适合处理无序集合（图、点云）；坏处是它对自然语言这种序列敏感的输入需要**显式注入位置信息**。这就是 4.3 位置编码的存在意义：把绝对 / 相对位置信息加到 token 表示里，破坏置换不变性。

**性质 2：全局可见性 + 一跳直连**

每个 token 一次 forward 就能看到所有其他 token，任意两 token 之间的"信息传递路径长度"是 **1**（vs RNN 是 $|i - j|$、CNN 是 $\lceil |i-j| / k \rceil$）。这是 self-attention 解决长程依赖的根本机制。

**性质 3：与 CNN / RNN 的对比**

| 维度 | RNN | CNN | Self-attention |
|---|---|---|---|
| 感受野 | 理论无限、实际短 | local（kernel size $k$） | **global**（一跳） |
| 顺序敏感 | 是（隐式） | 是（隐式） | **否**（需显式位置编码） |
| 并行 | 沿时间不行 | 完全并行 | **完全并行** |
| 长程依赖 | 弱（梯度衰减） | 弱（要堆很多层） | **强** |
| 计算量 | $O(nd^2)$ | $O(nkd^2)$ | $O(n^2 d + nd^2)$ |
| GPU 利用率 | 低 | 高 | **高** |

这张表是 Vaswani 2017 §4 "Why Self-Attention" 的核心论点——self-attention 在 4 个维度里 3 个完胜 RNN/CNN，唯一代价是 $n^2$ 复杂度，而这个代价被工程优化（FlashAttention）和架构改进（GQA / SSM）压到了可接受范围。

### 2.7 Multi-head 一句话预告

单头 self-attention 的容量有限——一组 $W_Q, W_K, W_V$ 只能学一种"相似度模式"。Vaswani 的解法是把 $d$ 维分成 $h$ 个 $d/h$ 维子空间，**并行**做 $h$ 个独立 attention（每个有自己的一组 $W_Q^{(i)}, W_K^{(i)}, W_V^{(i)}$），最后把 $h$ 个 $(n, d/h)$ 输出拼起来过一个 $W_O$。这样每个 head 可以专注学一种 attention pattern（句法、语义、共指、长程……），整体表达力大幅提升。详见 4.2，本节按 $h = 1$ 即可。

---

## 3. 最小代码示例

### 3.1 手撕 single-head causal self-attention（28 行）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        self.d_k = d_model
        self.W_q = nn.Linear(d_model, d_model, bias=False)   # 三个投影矩阵
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        # 预先注册下三角 causal mask 到 buffer，避免每次 forward 重建
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_len, max_len, dtype=torch.bool)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d)
        B, T, _ = x.shape
        Q = self.W_q(x)                                        # (B, T, d_k)
        K = self.W_k(x)                                        # (B, T, d_k)
        V = self.W_v(x)                                        # (B, T, d_k)

        # Step 1: 相似度 + Step 2: 缩放
        scores = Q @ K.transpose(-2, -1) / (self.d_k ** 0.5)   # (B, T, T)
        # Step 3: causal mask
        mask = self.causal_mask[:T, :T]                        # (T, T) bool
        scores = scores.masked_fill(~mask, float("-inf"))      # 上三角填 -inf
        # Step 4: softmax 按行归一
        attn = F.softmax(scores, dim=-1)                       # (B, T, T)
        # Step 5: 加权 V
        out = attn @ V                                         # (B, T, d_k)
        return out
```

每一步都对应 §2.2 的五步公式。**关键点**：

- `register_buffer` 把 causal mask 存成模型 buffer，会跟着 `model.to(device)` 移动到 GPU、不参与梯度更新；每次 forward 切片 `[:T, :T]` 即可，不重新构造。
- `Q @ K.transpose(-2, -1)` 的 `transpose(-2, -1)` 把 K 最后两维 swap——对 `(B, T, d_k)` 张量等价于把 K 转置成 `(B, d_k, T)`，matmul 出来 `(B, T, T)`。
- `float("-inf")` 在 fp32 / bf16 下都安全；fp16 训练时建议改用 `torch.finfo(scores.dtype).min`（约 $-6.5 \times 10^4$）避免溢出 NaN。
- 这里 $d_v = d_k = d_{\text{model}}$，没有 output projection $W_O$——multi-head 才需要 $W_O$ 把多个 head 拼起来再投影回 $d$ 维。

### 3.2 用 PyTorch 2.0 现代 API（自动选 FlashAttention backend）

```python
import torch
import torch.nn.functional as F

class SingleHeadSelfAttentionSDPA(torch.nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.W_q = torch.nn.Linear(d_model, d_model, bias=False)
        self.W_k = torch.nn.Linear(d_model, d_model, bias=False)
        self.W_v = torch.nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SDPA 期望形状 (B, n_head, T, d_k)，单头时 n_head=1
        B, T, _ = x.shape
        Q = self.W_q(x).unsqueeze(1)                           # (B, 1, T, d_k)
        K = self.W_k(x).unsqueeze(1)
        V = self.W_v(x).unsqueeze(1)
        out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)  # (B, 1, T, d_k)
        return out.squeeze(1)
```

`F.scaled_dot_product_attention` 是 PyTorch 2.0 引入的标准接口，内部根据硬件、序列长度、dtype 自动选三种 backend 之一：FlashAttention（最快）、memory-efficient attention、math（朴素实现）。生产代码写这一行就够，**不要自己手写 attention 上线**——FlashAttention backend 比手写快 2-4×、显存省一半。

### 3.3 数值一致性验证

```python
torch.manual_seed(0)
B, T, d = 2, 16, 32
x = torch.randn(B, T, d)

m1 = SingleHeadSelfAttention(d_model=d, max_len=T)
m2 = SingleHeadSelfAttentionSDPA(d_model=d)
# 共享权重确保对比的是实现而非参数差异
m2.W_q.weight.data = m1.W_q.weight.data.clone()
m2.W_k.weight.data = m1.W_k.weight.data.clone()
m2.W_v.weight.data = m1.W_v.weight.data.clone()

with torch.no_grad():
    out1 = m1(x)
    out2 = m2(x)

print((out1 - out2).abs().max())     # 期望 < 1e-5（fp32 数值噪声范围）
assert torch.allclose(out1, out2, atol=1e-5), "两种实现数值不一致！"
print("PASS：手撕版本与 SDPA 版本数值一致")
```

跑一次确认两种实现 bit-level 等价（fp32 下 max abs error < 1e-5 都属于浮点 round-off）。**这种 cross-check 是手撕 attention 的 sanity test 标配**——4.6 完整 decoder-only 实现时还会反复用。

### 3.4 工业级参考

Karpathy 的 nanoGPT [`model.py`](https://github.com/karpathy/nanoGPT/blob/master/model.py) 里的 `CausalSelfAttention` 类是本节代码的工业版（multi-head + dropout + 可选 SDPA / 手写切换），强烈建议读一遍——50 行不到，是"工业级精炼"的最佳教材。本节代码可以理解为它的单头简化版。

---

## 4. 工程踩坑与经验

- ❗ **忘了除以 $\sqrt{d_k}$ → softmax 梯度消失，loss 不下降**。新手手撕 attention 最经典 bug：照着公式抄了 $QK^\top V$，漏了 $\sqrt{d_k}$。表现是训练前几个 step loss 几乎不动、attention map 退化成 one-hot（每个 token 只 attend 自己）。修复：永远写成 `scores = (Q @ K.transpose(-2,-1)) * (self.d_k ** -0.5)`，把 scaling 与 matmul 写在同一行。
- ❗ **mask 用 `-1e9` 在 fp16 下不严格 = 0**。fp16 数值范围约 $\pm 6.5 \times 10^4$，`-1e9` 在 fp16 下会被 clamp 成 `-inf`（运气好）或 `-65504`（运气坏，softmax 后仍有 $10^{-30}$ 量级残留，多层累加可能引爆 NaN）。**正确做法**：用 `mask.bool()` + `masked_fill(~mask, float('-inf'))`；或直接用 `F.scaled_dot_product_attention(..., is_causal=True)` 让框架处理；或在 fp16 下用 `torch.finfo(scores.dtype).min`。
- ❗ **`softmax(...) @ V` 中间 attention matrix 占 $O(n^2)$ 显存**。$n = 8192$、bf16、32 head 时仅 attention 矩阵就要 $32 \times 8192^2 \times 2 / 10^9 \approx 4.3$ GB / layer——一个 32 层模型光 attention matrix 就 130 GB，单卡装不下。FlashAttention 通过分块 + online softmax 完全不实例化这个矩阵，把显存压到 $O(n)$。**所以"长 context 训练" = "必须用 FlashAttention"**，不是建议是必需。
- ❗ **self-attention 是 $O(n^2)$，training step 时间随 seqlen 二次增长，不要为了"长一点反正没事"开 4096**。常见误区："反正机器有显存，把 seqlen 从 1024 开到 4096 应该没事"——错。step time 会变成 $\approx 16$ 倍（$4^2$），同样 token 数训完要的钟点反而**更长**。除非数据里真有需要 4096 上下文的样本（比如 multi-turn agent trajectory），否则**长 seqlen 是亏的**。生产规则：先看数据 token 长度分布（p50 / p95 / p99），choose seqlen ≈ p95。
- ❗ **$Q/K/V$ 的 $d_k$ 维度通常等于 $d / h$（$h$ 为头数），不是 $d$ 本身**。多头时常见 bug 是漏掉 reshape：`Q = self.W_q(x).view(B, T, h, d_k).transpose(1, 2)`，少 transpose 一步 head 与 sequence 维度搞反，attention 算出来形状对但语义全乱、loss 不收敛。建议每一步打 shape 注释 `# (B, h, T, d_k)` 防止维度漂移。
- ❗ **不加位置编码的 self-attention 退化成"词袋"——颠倒 token 顺序输出不变**。性质 1 的实战后果：纯 self-attention 是置换等变的，"猫吃鱼"和"鱼吃猫"输出会完全一样（同一组向量、不同顺序）——对自然语言这显然是灾难。**任何 production self-attention 都必须搭配位置编码**（绝对 PE / RoPE / ALiBi 之一），4.3 详讲。如果你的 attention 没接位置编码、模型完全学不会语序，先怀疑这个。
- ❗ **训练时 `attention_mask` 经过 broadcast 后是 `(B, 1, T, T)`，自己写要小心维度**。HuggingFace 的 `attention_mask` 输入是 $(B, T)$，但模型内部要扩展到 $(B, 1, T, T)$ 才能与 score 相加（多头时 1 这一维会自动 broadcast 到 $h$）。直接把 $(B, T)$ 当 $(B, T, T)$ 用是新手最经典的 broadcast bug——score 的 row / column 全错。建议封一个 `def expand_mask(mask_2d: BoolTensor) -> BoolTensor` helper 统一处理。
- ❗ **dropout 应该加在 softmax 之后、$V$ 加权之前**（attention dropout）。Vaswani 2017 原文这么做，nanoGPT / HF 实现都跟随。新手有时把 dropout 加在 score 之前（错，会改变 softmax 概率分布），或加在最后输出（错，效果差）。**正确位置**：`attn = F.dropout(F.softmax(scores, dim=-1), p=0.1) @ V`。LLaMA / Qwen 等现代 LLM 通常 attention dropout 直接关掉（设 0），但你要知道 default 应该加在哪。

---

## 5. 经典 paper

- **Vaswani et al., 2017 — Attention Is All You Need** — Transformer 原典，本节就是它的复述与扩展。必读 §3.2 "Scaled Dot-Product Attention"（公式与 $\sqrt{d_k}$ scaling 的 footnote）和 §4 "Why Self-Attention"（与 RNN / CNN 的复杂度对比表）。这两节内容直接对应本节 §2.2-§2.3 与 §2.6。整篇 paper 只有 11 页，回头每年都该重读一遍——细节读透了能避免 80% 的 attention 实现 bug。
- **Bahdanau, Cho & Bengio, 2014 — Neural Machine Translation by Jointly Learning to Align and Translate** — attention 的精神祖师爷（cross-attention 起点）。本节没有展开（2.4 节专讲），但读它能体会"为什么 self-attention 是 cross-attention 的合理推广"——Vaswani 2017 §3.2 的引用第一篇就是它。
- **Karpathy — nanoGPT** — 不是 paper，但价值堪比 paper。`model.py` 里 50 行的 `CausalSelfAttention` 是工业级 single-head / multi-head 实现的最佳教材，注释里直接讨论了 SDPA backend 选择、Flash 启用条件、causal mask 的 buffer 实现。本节 §3 的代码风格直接受其影响。

---

## 6. 自测与面试题

**Q1（公式）**：写出 scaled dot-product attention 的完整公式，并解释为什么要除以 $\sqrt{d_k}$（推导 $q \cdot k$ 的方差为什么是 $d_k$）。

<details>
<summary>Answer sketch</summary>

完整公式：

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

其中 $Q, K \in \mathbb{R}^{n \times d_k}$、$V \in \mathbb{R}^{n \times d_v}$。

为什么除以 $\sqrt{d_k}$ 的推导（要点）：

- 假设 $q, k$ 各分量 $q_i, k_i$ 是均值 0、方差 1 的独立随机变量
- $\mathbb{E}[q_i k_i] = \mathbb{E}[q_i]\mathbb{E}[k_i] = 0$，所以 $\mathbb{E}[q \cdot k] = 0$
- $\text{Var}(q_i k_i) = \mathbb{E}[q_i^2]\mathbb{E}[k_i^2] = 1$
- 由独立性，$\text{Var}(q \cdot k) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i) = d_k$
- $d_k$ 大时 → score magnitude 大 → softmax 进入饱和区（max → 1，其他 → 0）→ 梯度 $p(1-p) \to 0$ → 训练学不动
- 除以 $\sqrt{d_k}$（标准差）让 score 方差归一到 1，softmax 在合理梯度区间
- 不除 $d_k$ 是因为那样会把 score 压得过扁，softmax 退化成均匀分布，attention 失去区分能力

加分：能引用 Vaswani 2017 §3.2.1 footnote 4 的实证；能说出 fp16 下 mask 用 `-1e9` 不安全应该用 `float('-inf')` 或 `torch.finfo(dtype).min`。

</details>

**Q2（实现）**：用 PyTorch 写一个 single-head causal self-attention（≤ 15 行核心代码），输入 `(B, T, d)`，输出 `(B, T, d)`，要求实现 scaled dot-product + causal mask + softmax + V 加权，并且与 `F.scaled_dot_product_attention(is_causal=True)` 数值一致。

<details>
<summary>Answer sketch</summary>

核心 15 行：

```python
class CausalSelfAttn(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.W_q = nn.Linear(d, d, bias=False)
        self.W_k = nn.Linear(d, d, bias=False)
        self.W_v = nn.Linear(d, d, bias=False)
        self.d = d

    def forward(self, x):                                   # x: (B, T, d)
        B, T, d = x.shape
        Q, K, V = self.W_q(x), self.W_k(x), self.W_v(x)     # 各 (B, T, d)
        scores = Q @ K.transpose(-2, -1) / (d ** 0.5)       # (B, T, T)
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
        scores = scores.masked_fill(~mask, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        return attn @ V                                     # (B, T, d)
```

考核要点：

- 三个独立投影 $W_Q, W_K, W_V$
- $\sqrt{d_k}$ scaling 不能漏（最常考）
- causal mask 用下三角 + `masked_fill(~mask, -inf)` 而不是 `-1e9`
- softmax 在 mask 之后、加权 V 之前
- 输出形状 `(B, T, d)` 与输入一致（单头时 $d_v = d$）

数值一致性验证：

```python
ref = F.scaled_dot_product_attention(Q.unsqueeze(1), K.unsqueeze(1), V.unsqueeze(1),
                                     is_causal=True).squeeze(1)
assert torch.allclose(out, ref, atol=1e-5)
```

加分：能解释为什么 SDPA 期望 4D 形状 `(B, h, T, d_k)`；能说出 `register_buffer` 比每次构造 mask 更优；能指出 fp16 下应该 `torch.finfo(scores.dtype).min` 替代 `-inf`。

</details>

**Q3（trade-off）**：self-attention 与 RNN 在（并行性 / 计算复杂度 / 长程依赖）三个维度对比；如果 $n = 128\text{k}$，self-attention 会带来什么问题，业界有哪些应对方案？

<details>
<summary>Answer sketch</summary>

三维对比表：

| 维度 | RNN | Self-attention |
|---|---|---|
| 并行性 | **沿时间无法并行**（$h_t$ 依赖 $h_{t-1}$） | **完全并行**（一次 matmul） |
| 计算复杂度 | $O(n d^2)$ 时间，$O(n)$ 空间 | $O(n^2 d + n d^2)$ 时间，$O(n^2)$ 空间 |
| 长程依赖 | 弱（梯度衰减、走 $\|i-j\|$ 步） | 强（任意两 token 一跳直连） |
| GPU 利用率 | 低（串行 kernel） | 高（大 matmul） |

$n = 128\text{k}$ 的问题：

- **显存爆炸**：attention matrix $128\text{k} \times 128\text{k} \times \text{bf16} = 32$ GB / head，32 head 总 1 TB，单卡完全装不下
- **计算量爆炸**：$O(n^2 d)$ 训练 step 时间 $\propto 128^2 = 16384$ 倍 of $n=1\text{k}$
- **数值精度**：bf16 / fp16 下 softmax 累加 128k 项，数值不稳定

业界应对方案（多答更好）：

- **FlashAttention 1/2/3**（5.3 / 7.6）：IO-aware kernel，分块 + online softmax，不实例化 $n \times n$ 矩阵，显存 $O(n)$，2-4× 加速
- **PagedAttention / Continuous Batching**（11.2）：vLLM 推理时 KV cache 分页管理，避免长 context 内存碎片
- **GQA / MQA / MLA**（5.2）：压缩 KV cache 头数，KV 显存 / 通信量降 4-8×（推理友好）
- **Sliding window attention**（Mistral / Longformer）：每个 token 只看附近 $w$ 个 token，复杂度 $O(nw)$；牺牲全局可见性换扩展性
- **SSM / Mamba**（5.5）：放弃 attention，用 state space model，复杂度 $O(n)$，但表达力 trade-off 仍在研究中
- **Position Interpolation / NTK / YaRN / LongRoPE**（6.5）：长 context 的位置编码外推，不直接解决 $n^2$ 问题但让模型能用上长 context
- **Ring Attention / Sequence Parallelism**（7.3）：把序列维切到多卡，每卡只算一段，通过 ring 通信汇总
- **稀疏 attention**（BigBird / LongNet / Sparse Transformer）：用 local + global + random pattern 把稠密 $n^2$ 压成稀疏 $O(n \log n)$ 或 $O(n)$

加分：能区分"训练 long context"与"推理 long context"两个不同问题——训练靠 FlashAttention + sequence parallelism，推理靠 PagedAttention + KV cache 压缩 + 量化。

</details>

---

## 7. 延伸阅读

- [Vaswani et al. 2017 — Attention Is All You Need (arXiv)](https://arxiv.org/abs/1706.03762) — 原 paper，11 页全部精读
- [Karpathy — nanoGPT `model.py`](https://github.com/karpathy/nanoGPT/blob/master/model.py) — 工业级 `CausalSelfAttention` 50 行实现，注释里讨论 SDPA 与 Flash backend 选择
- [Karpathy — Let's build GPT (YouTube)](https://www.youtube.com/watch?v=kCc8FmEb1nY) — 2 小时手撕 GPT-2，self-attention 部分的讲解是全网最直观的之一
- [Jay Alammar — The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — 全网最好的 Transformer 可视化教程，Q/K/V 流动的动画图配本节 §1 mental model 食用
- [Lilian Weng — The Transformer Family](https://lilianweng.github.io/posts/2020-04-07-the-transformer-family/) — 从 vanilla self-attention 到各种变体（sparse、linear、long-context）的全谱综述
- [PyTorch — `scaled_dot_product_attention` 文档](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) — 现代标准 API 的 backend 选择规则与 mask 约定
- 推荐继续读本教程的 **4.2 节《Multi-head attention 与 head 的物理意义》**——把单头的 $W_Q, W_K, W_V$ 切成 $h$ 份并行做，理解为什么"多头"能让模型并行学不同的 attention pattern
