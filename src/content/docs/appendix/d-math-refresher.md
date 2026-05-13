---
title: "附录 D：数学速查（线代 / 概率 / 信息论 / 优化）"
description: "LLM/Agent 算法工程师真正会用到的那点数学的 cheat sheet——不教你怎么推 SVD，只告诉你\"看到 $D_{KL}$ 该想到什么、写 `matmul` 时维度怎么对、Adam 那个 update 在算什么\"。"
---

> ⏱ 预计阅读 25 分钟 ｜ 难度 ★ ｜ 前置：无

## 一句话本节讲什么

LLM/Agent 算法工程师真正会用到的那点数学的 cheat sheet——不教你怎么推 SVD，只告诉你"看到 $D_{KL}$ 该想到什么、写 `matmul` 时维度怎么对、Adam 那个 update 在算什么"。

---

## 0. 怎么用这份速查

- 如果你**完全没学过**这些概念：把它当地图，用到哪个再去查 Goodfellow《Deep Learning》对应章节
- 如果你**学过但忘了**：直接当 cheat sheet 翻
- 如果你**正在面试前夕**：重点看每个概念后面"在 LLM 中的用例"那行，这是面试官真正想听的

不追求严谨证明，追求**"看到这个符号脑子里立刻浮现哪段代码 / 哪个 paper"**。

---

## 1. 线性代数（Linear Algebra）

LLM 的所有计算最后都落到张量的乘加。线代不是要你手算特征值，是要你**对维度有直觉**。

### 1.1 向量、矩阵、张量

| 对象 | 维度记法 | PyTorch shape | 在 LLM 中 |
|---|---|---|---|
| 标量 | $a \in \mathbb{R}$ | `()` | loss 值 |
| 向量 | $\mathbf{v} \in \mathbb{R}^d$ | `(d,)` | 单个 token embedding |
| 矩阵 | $A \in \mathbb{R}^{m \times n}$ | `(m, n)` | Linear layer 权重 |
| 张量 | $T \in \mathbb{R}^{b \times n \times d}$ | `(B, N, D)` | 一个 batch 的隐状态 |

LLM 中最常见的 4D 张量：`(batch, head, seq_len, head_dim)`，这是 multi-head attention 里 Q/K/V 的形状。

### 1.2 矩阵乘法（最常用）

$C = AB$，要求 $A \in \mathbb{R}^{m \times k}$、$B \in \mathbb{R}^{k \times n}$，结果 $C \in \mathbb{R}^{m \times n}$。**中间维度必须相等**。

LLM 中：
- `nn.Linear(d_in, d_out)` 内部就是 $XW^T + b$，$W \in \mathbb{R}^{d_{out} \times d_{in}}$
- Attention 的 $QK^T$：$Q \in \mathbb{R}^{n \times d_k}$，$K^T \in \mathbb{R}^{d_k \times n}$，结果是 $n \times n$ 的注意力分数

### 1.3 转置 / 逆 / 迹

- 转置 $A^T$：行列互换，shape $(m,n) \to (n,m)$
- 逆 $A^{-1}$：满足 $AA^{-1} = I$；**深度学习里几乎不直接求逆**（数值不稳定 + $O(n^3)$）
- 迹 $\text{tr}(A) = \sum_i A_{ii}$：对角线之和；性质 $\text{tr}(AB) = \text{tr}(BA)$

### 1.4 三种"乘法"别搞混

| 名字 | 符号 | 形状要求 | 例子 |
|---|---|---|---|
| 内积（dot product） | $\mathbf{u} \cdot \mathbf{v}$ | 同维向量 → 标量 | attention score 的核心 |
| Hadamard 积（elementwise） | $A \odot B$ | 同 shape → 同 shape | gating（如 SwiGLU） |
| 矩阵乘 | $AB$ | 中间维相等 | Linear layer |

PyTorch 里：`a @ b` = matmul，`a * b` = Hadamard，`(a*b).sum()` = inner product。

### 1.5 范数（Norm）

- L1 范数：$\|\mathbf{x}\|_1 = \sum_i |x_i|$，鼓励稀疏（Lasso）
- L2 范数：$\|\mathbf{x}\|_2 = \sqrt{\sum_i x_i^2}$，最常用，weight decay 用的就是它
- Frobenius 范数（矩阵的 L2）：$\|A\|_F = \sqrt{\sum_{ij} A_{ij}^2}$

LLM 中：grad clipping 用的就是梯度向量的 L2 范数；embedding 归一化（如 Sentence-BERT）也是 L2。

### 1.6 特征值 / 特征向量 / spectral radius

满足 $A\mathbf{v} = \lambda \mathbf{v}$ 的 $\lambda$ 叫特征值，$\mathbf{v}$ 叫特征向量。最大的 $|\lambda|$ 叫 **spectral radius**。

LLM 中的相关性：RNN 的隐状态递推 $h_t = W h_{t-1}$，如果 $W$ 的 spectral radius < 1 → 梯度消失；> 1 → 梯度爆炸。这是 LSTM / GRU 设计 gating 的根本动机，也是 Transformer 用 residual 而非纯递推的原因。

### 1.7 SVD 与矩阵的秩

任何 $A \in \mathbb{R}^{m \times n}$ 可分解为 $A = U\Sigma V^T$，其中 $\Sigma$ 是对角阵，对角线是奇异值。**rank**(秩) = 非零奇异值个数。

LLM 中：**LoRA** 假设权重更新 $\Delta W$ 是低秩的，于是用 $\Delta W = BA$，$B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times d}$，$r \ll d$，参数从 $d^2$ 降到 $2dr$。这就是 SVD 思想的工程化。

---

## 2. 微积分 / 自动微分

### 2.1 导数与梯度

- 一元导数 $\frac{df}{dx}$：函数在该点的瞬时变化率
- 偏导 $\frac{\partial f}{\partial x_i}$：固定其他变量，对 $x_i$ 求导
- 梯度 $\nabla f = \left[\frac{\partial f}{\partial x_1}, \dots, \frac{\partial f}{\partial x_n}\right]^T$：所有偏导拼成的向量，指向函数增长最快方向
- Jacobian $J_{ij} = \frac{\partial f_i}{\partial x_j}$：向量函数 → 向量函数的"梯度推广"

### 2.2 链式法则（反向传播的核心）

复合函数 $y = f(g(x))$ 的导数：

$$\frac{dy}{dx} = \frac{df}{dg} \cdot \frac{dg}{dx}$$

**反向传播 = 链式法则在计算图上的递推**。autograd（PyTorch / JAX）就是把每个 op 的局部导数事先实现好，前向时记录计算图，反向时按链式法则乘起来。

### 2.3 梯度下降

参数更新：

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta L(\theta_t)$$

$\eta$ 是学习率。**Adam** 加了一阶矩 $m$（动量）和二阶矩 $v$（自适应学习率）：

$$\theta_{t+1} = \theta_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

详细见 1.2 节。

---

## 3. 概率论

### 3.1 概率分布

- **离散分布**：categorical distribution，给每个 token 分配概率，所有概率和为 1。LLM 的 next-token 分布就是 categorical
- **连续分布**：normal distribution $\mathcal{N}(\mu, \sigma^2)$，权重初始化常用

### 3.2 条件概率与 Bayes 公式

$$P(A|B) = \frac{P(A \cap B)}{P(B)}, \quad P(A|B) = \frac{P(B|A)P(A)}{P(B)}$$

LLM 的 causal language modeling 本质就是建模 $P(x_t | x_{<t})$。

### 3.3 期望、方差、协方差

- 期望 $\mathbb{E}[X] = \sum_x x P(x)$（离散）或 $\int x p(x) dx$（连续）
- 方差 $\text{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2]$
- 协方差 $\text{Cov}(X, Y) = \mathbb{E}[(X-\mathbb{E}X)(Y-\mathbb{E}Y)]$

LLM 中：BatchNorm / LayerNorm 算的就是均值和方差；$QK^T$ 除以 $\sqrt{d_k}$ 是为了控制方差到 1（见 4.1 节）。

### 3.4 极大似然估计（MLE）

给定数据 $D = \{x_1, \dots, x_N\}$ 和参数 $\theta$，找让数据概率最大的 $\theta$：

$$\theta^* = \arg\max_\theta \prod_i P(x_i | \theta) = \arg\max_\theta \sum_i \log P(x_i | \theta)$$

**LLM 预训练就是 MLE**：最大化训练集 $\sum_t \log P(x_t | x_{<t}; \theta)$，等价于最小化 cross-entropy loss。

### 3.5 采样（Sampling）

从 next-token 分布 $p$ 里采一个 token：
- **Greedy / argmax**：取概率最大的（确定性）
- **Temperature** $T$：$p_i' \propto \exp(\log p_i / T)$。$T \to 0$ → argmax；$T \to \infty$ → uniform
- **Top-k**：只在概率前 k 个里采
- **Top-p (nucleus)**：取累积概率达 p 的最小集合，从里面采

详细见 11 章推理引擎。

---

## 4. 信息论

LLM 的 loss、RLHF 的 KL 约束、知识蒸馏的目标——全都建立在这四个概念上。

### 4.1 熵（Entropy）

$$H(p) = -\sum_x p(x) \log p(x)$$

直觉：分布有多"不确定"。均匀分布 entropy 最大，one-hot entropy = 0。

### 4.2 交叉熵（Cross-entropy）

$$H(p, q) = -\sum_x p(x) \log q(x)$$

直觉：用分布 $q$ 编码来自 $p$ 的样本，平均需要多少 bits。**LLM 训练 loss 就是 CE**：$p$ 是真实 next-token 的 one-hot，$q$ 是模型预测分布，结果简化为 $-\log q(x_{true})$。

### 4.3 KL Divergence

$$D_{KL}(p \| q) = \sum_x p(x) \log \frac{p(x)}{q(x)} = H(p, q) - H(p)$$

直觉：分布 $q$ 离分布 $p$ 有多远。**关键性质**：
- $D_{KL} \geq 0$，等于 0 当且仅当 $p = q$
- **不对称**：$D_{KL}(p \| q) \neq D_{KL}(q \| p)$

LLM 中：**RLHF 的 KL 约束** $D_{KL}(\pi_\theta \| \pi_{\text{ref}})$ 防止 policy 偏离 SFT model 太远（详见 9.3）；知识蒸馏用 KL 让 student 学 teacher 分布。

**Forward vs Reverse KL**：
- Forward KL $D_{KL}(p \| q)$：$q$ 必须覆盖 $p$ 的所有 mode（mass-covering，"求大同"）
- Reverse KL $D_{KL}(q \| p)$：$q$ 只需 cover 一个 mode（mode-seeking，"挑一个"）
- RLHF 用 reverse KL，导致 mode-collapse 倾向（output 单一化）

### 4.4 互信息（Mutual Information）

$$I(X; Y) = D_{KL}(p(x,y) \| p(x)p(y)) = H(X) - H(X|Y)$$

直觉：知道 $Y$ 后对 $X$ 的不确定性减少多少。表征学习里常用（InfoNCE、CLIP 的对比学习）。

---

## 5. 优化基础

### 5.1 凸 vs 非凸

凸函数：任意两点连线在函数图像上方（或之上）。凸优化有全局最优解。**深度学习是非凸优化**——只能找局部最优，靠初始化 + SGD 的随机性碰运气。

### 5.2 主流优化器对比

| 优化器 | 更新规则关键 | 何时用 |
|---|---|---|
| SGD | $\theta \leftarrow \theta - \eta g$ | 简单任务、CV 经典 |
| Momentum | 加一阶矩 $m$（指数滑动平均） | 加速收敛、跨过 saddle point |
| Adam | 一阶矩 $m$ + 二阶矩 $v$ | LLM 默认，收敛稳定 |
| AdamW | Adam + 解耦的 weight decay | **LLM 的事实标准** |

详细见 1.2 节。

### 5.3 学习率 schedule

LLM 训练的标准做法：
- **Warmup**：前 ~1% steps 线性升 lr 到 peak（避免初始梯度爆炸）
- **Cosine decay**：之后按 $\eta_t = \eta_{\min} + 0.5(\eta_{\max} - \eta_{\min})(1 + \cos(\pi t/T))$ 衰减

### 5.4 梯度裁剪（Grad Clip）与 Weight Decay

- Grad clip：当 $\|g\|_2 > c$ 时把 $g$ 缩放到 $c$，防止梯度爆炸（一般 $c=1.0$）
- Weight decay：在 loss 里加 $\lambda \|\theta\|_2^2$ 等价于每次更新对 $\theta$ 乘个 $(1-\eta\lambda)$，防过拟合

---

## 6. 数学符号速查表

| 符号 | 含义 | 例子 |
|---|---|---|
| $\mathbb{R}^n$ | n 维实数向量空间 | $\mathbf{v} \in \mathbb{R}^{768}$ |
| $\mathbb{E}[X]$ | 期望 | $\mathbb{E}_{x \sim p}[f(x)]$ |
| $\mathbb{E}_{x \sim p}[\cdot]$ | 在分布 $p$ 下对 $x$ 取期望 | RLHF 目标的常见写法 |
| $\nabla_\theta f$ | $f$ 对 $\theta$ 的梯度 | $\nabla_\theta L$ 反向传播算的 |
| $\| \cdot \|_2$ | L2 范数 | grad clip 看的就是这个 |
| $\sigma(x)$ | sigmoid，$\frac{1}{1+e^{-x}}$ | DPO loss 里出现 |
| $\odot$ | element-wise（Hadamard）积 | SwiGLU 的门控 |
| $\otimes$ | tensor / outer product | 论文里很少用，留意 |
| $\propto$ | 正比于 | $p(x) \propto e^{-E(x)}$ |
| $\sim$ | 服从分布 | $x \sim \mathcal{N}(0, 1)$ |
| $\arg\min / \arg\max$ | 取最优参数 | $\theta^* = \arg\min_\theta L$ |
| $\circ$ | 函数复合 | $f \circ g$ 即 $f(g(\cdot))$ |
| $\mathbb{1}[\cdot]$ | 指示函数（条件成立 = 1） | RLVR reward 常见 |
| $\hat{x}$ | $x$ 的估计 / 归一化 | LayerNorm 的输出 |

---

## 7. 关键公式速查

| 名字 | 公式 | 在哪用 |
|---|---|---|
| Softmax | $\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$ | 输出层、attention score |
| Sigmoid | $\sigma(x) = \frac{1}{1+e^{-x}}$ | 二分类、DPO |
| GELU | $\text{GELU}(x) = x \cdot \Phi(x)$，$\Phi$ 是标准正态 CDF | Transformer FFN |
| LayerNorm | $\hat{x} = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$ | 每个 sub-layer 后 |
| Cross-entropy | $L = -\sum_x p(x) \log q(x)$ | LM loss |
| KL divergence | $D_{KL}(p\|q) = \sum_x p(x) \log\frac{p(x)}{q(x)}$ | RLHF KL 约束 |
| Adam update | $\theta \leftarrow \theta - \eta \cdot \hat{m}/(\sqrt{\hat{v}} + \epsilon)$ | LLM 默认优化器 |
| Self-attention | $\text{Attn}(Q,K,V) = \text{softmax}(QK^T/\sqrt{d_k})V$ | Transformer 核心 |

---

## 8. 最小代码示例（PyTorch）

把上面提到的几个核心 op 串起来跑一遍：

```python
import torch
import torch.nn.functional as F

torch.manual_seed(0)
B, N, D, V = 2, 4, 8, 100  # batch, seq_len, hidden, vocab

# 1. matmul：注意 shape
x = torch.randn(B, N, D)            # (B, N, D)
W = torch.randn(D, V)               # (D, V)
logits = x @ W                      # (B, N, V)  # 中间维 D 对齐

# 2. softmax → probs
probs = F.softmax(logits, dim=-1)   # 沿 vocab 维归一化

# 3. cross-entropy（数值稳定写法：直接用 logits）
targets = torch.randint(0, V, (B, N))
ce = F.cross_entropy(logits.view(-1, V), targets.view(-1))
print(f"CE loss: {ce.item():.4f}")

# 4. KL divergence: D_KL(p || q)
p_logits = torch.randn(B, V)
q_logits = torch.randn(B, V)
log_p = F.log_softmax(p_logits, dim=-1)
log_q = F.log_softmax(q_logits, dim=-1)
# F.kl_div 期望: input=log_q, target=p (probs)，算的是 D_KL(p||q)
kl = F.kl_div(log_q, log_p, reduction='batchmean', log_target=True)
print(f"KL(p||q): {kl.item():.4f}")

# 5. sampling with temperature + top-k
T = 0.8
last_logits = logits[:, -1, :] / T              # (B, V)
topk = 5
vals, idx = last_logits.topk(topk, dim=-1)      # (B, topk)
sample_probs = F.softmax(vals, dim=-1)
chosen = torch.multinomial(sample_probs, 1)     # (B, 1)
next_token = idx.gather(-1, chosen)             # (B, 1)
print(f"Sampled token ids: {next_token.squeeze(-1).tolist()}")
```

跑这段代码能验证你对：matmul shape / softmax / CE 数值稳定写法 / KL 方向 / temperature + top-k 采样的理解。

---

## 9. 工程踩坑与经验

- ❗ **Numerical stability**：算 `log(softmax(x))` 永远用 `F.log_softmax(x, dim=-1)`，**不要** `softmax` 完再 `log`。后者在 logit 很小时会得到 $\log 0 = -\infty$。同理 cross-entropy 用 `F.cross_entropy(logits, ...)` 直接喂 logits，PyTorch 内部用 LogSumExp trick 做了稳定化。
- ❗ **KL divergence 不对称**：`F.kl_div(input, target)` 的 `input` 必须是 **log-probabilities**，且算的是 $D_{KL}(\text{target} \| \text{exp(input)})$。方向搞反在 RLHF 里可能让 reverse / forward KL 颠倒，调一晚上找不到 bug。
- ❗ **Sampling 时 temperature → 0 不要直接除以 0**：理论上 $T=0$ 等价 argmax，实现上用 `if T < eps: argmax else: multinomial(softmax(logits/T))`。`multinomial` 收到 NaN / 全 0 概率会报错或采到 invalid index。
- ❗ **链式法则的 outer / inner**：$\frac{d}{dx} f(g(x)) = f'(g(x)) \cdot g'(x)$，**outer 在 inner 上的导**乘以 inner 的导。手推 attention backward / RoPE backward 时容易把顺序写反。
- ❗ **矩阵维度匹配是 #1 bug 来源**：每写一行 `matmul` / `einsum` / `reshape` 都在注释里标 shape。`# (B, H, N, D) @ (B, H, D, N) -> (B, H, N, N)` 这种注释能省 90% 的调试时间。
- ❗ **浮点比较不要用 `==`**：fp16/bf16 下 $1.0 \neq 1.0$ 是常态。用 `torch.allclose(a, b, atol=1e-6, rtol=1e-5)`，并且写单测时 atol 要根据 dtype 调（bf16 给到 1e-2 才合理）。
- ❗ **Adam 的 `epsilon` 不要乱改**：默认 `1e-8` 是分母防 0 用的，但在 bf16 下 `1e-8` 会被 round 到 0，常见做法是 LLM 训练改 `1e-6` 或 `1e-5`（详见 1.2）。

---

## 10. 经典 paper / 教材

- **Goodfellow, Bengio, Courville 2016 — Deep Learning** — 经典 DL 教材，第 2-4 章覆盖本附录所有线代 / 概率 / 优化基础。看不下去全本可以只看这三章。
- **Murphy 2012 — Machine Learning: A Probabilistic Perspective** — 概率视角的 ML 圣经，特别推荐第 2 章（probability）和第 8 章（logistic regression / MLE）。
- **Deisenroth, Faisal, Ong 2020 — Mathematics for Machine Learning** — [免费 PDF](https://mml-book.github.io/)，针对 ML 重新组织的数学教材，比纯数学课本 friendly 得多。

加分资源：
- **3Blue1Brown — Essence of Linear Algebra / Calculus / Neural Networks** — YouTube 系列，建立**几何直觉**最快的方式。看完一遍以后看 attention / RoPE 公式都不一样。

---

## 11. 自测与面试题

**Q1（公式题）：** 写出 KL divergence 与 cross-entropy 的关系，并解释为什么训练 LM 时优化 cross-entropy 与最小化 KL 等价。

<details>
<summary>Answer sketch</summary>

- 定义：$H(p) = -\sum p \log p$，$H(p, q) = -\sum p \log q$
- 关系：$D_{KL}(p \| q) = H(p, q) - H(p)$
- LM 训练时，$p$ 是真实 next-token 分布（数据给定，是常量），$H(p)$ 不依赖 $\theta$
- 所以 $\min_\theta H(p, q_\theta) \Leftrightarrow \min_\theta D_{KL}(p \| q_\theta)$
- 直觉：CE 多了一个常数偏移 $H(p)$，梯度完全相同

</details>

**Q2（理解题）：** 为什么 LLM 训练用 cross-entropy 而不是 MSE？从概率角度解释。

<details>
<summary>Answer sketch</summary>

- LLM 的输出是离散分布（token 上的 categorical），不是连续值。
- 从 MLE 角度：最大化 $\log P(x_t | x_{<t}; \theta)$ 等价于最小化负对数似然 = cross-entropy（categorical 分布的 NLL）
- MSE 对应的概率假设是高斯分布（连续、对称），用在 token 上不合适
- 工程后果：MSE 在 logit 空间梯度会饱和（$\sigma'$ 很小），CE + softmax 的 grad 是 $q - p$，简洁且不饱和
- 加分点：cross-entropy 对错误预测的 penalty 是 $-\log q$，错得越离谱 loss 越大，符合 information-theoretic 直觉

</details>

**Q3（实现题）：** 用 PyTorch 写一个 numerically stable 的 LogSumExp（≤ 5 行）。

<details>
<summary>Answer sketch</summary>

```python
def logsumexp(x, dim=-1):
    m = x.max(dim=dim, keepdim=True).values
    return (x - m).exp().sum(dim=dim).log() + m.squeeze(dim)
```

- 关键 trick：$\log\sum_i e^{x_i} = m + \log\sum_i e^{x_i - m}$，$m = \max_i x_i$
- 减去 max 后所有 $e^{x_i - m} \leq 1$，避免 overflow
- 至少一项 $e^{x_i - m} = 1$，避免 underflow → log(0)
- PyTorch 直接有 `torch.logsumexp(x, dim)`，但面试要求自己实现

</details>

---

## 12. 延伸阅读

- [3Blue1Brown — Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra) — 看完线代的几何直觉就上来了
- [Mathematics for Machine Learning (free PDF)](https://mml-book.github.io/) — ML 视角的数学教材，第 5-7 章（向量微积分 / 概率 / 优化）特别推荐
- [Distill — Why Momentum Really Works](https://distill.pub/2017/momentum/) — 交互式可视化解释 momentum，比公式直观 10 倍
- [CS229 Linear Algebra Review (Stanford)](http://cs229.stanford.edu/section/cs229-linalg.pdf) — 短小精悍的线代 review notes
- 看完本附录可继续读 1.1（神经网络与反向传播）→ 1.2（优化器）→ 1.4（损失函数），这三节是数学知识在 LLM 中的第一次落地
