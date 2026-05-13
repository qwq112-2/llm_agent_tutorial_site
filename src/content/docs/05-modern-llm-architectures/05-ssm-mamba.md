---
title: "5.5 SSM 路线：Mamba / Mamba-2 与 Transformer 对比"
description: "State Space Model（SSM）是\"现代化的 RNN\"——用线性递推 + 硬件友好的 parallel scan，把 RNN 的 $O$ 推理优势保留下来、把\"无法并行训练\"这个原罪砸掉；从 S4 → Mamba → Mamba-2 ，selective mechanism 与 SSM-attention duality 让它成为 Transformer 之外最值得了解的 LLM 架构"
---

> ⏱ 预计阅读 50 分钟 ｜ 难度 ★★★ ｜ 前置：2.3 RNN/LSTM/GRU、4.1 Self-attention

## 一句话本节讲什么

State Space Model（SSM）是"现代化的 RNN"——用线性递推 + 硬件友好的 parallel scan，把 RNN 的 $O(n)$ 推理优势保留下来、把"无法并行训练"这个原罪砸掉；从 S4 (2021) → Mamba (2023) → Mamba-2 (2024)，selective mechanism 与 SSM-attention duality 让它成为 Transformer 之外最值得了解的 LLM 架构线，但因 in-context learning 弱、生态不成熟、最大公开尺寸仅 2.8B，2024-2026 仍是 Transformer 主导 + Mamba/Hybrid (Jamba) 在长 context 与流式场景做补位。

---

## 1. Mental model（直觉）

### 1.1 一句话定位：把 RNN 的"串行"砸掉，把"$O(n)$ 推理"留下

回到 2.3 节的结论——RNN 的两个原罪是 **(1) 训练无法并行；(2) 长程依赖弱**。Transformer 用 self-attention 一次 matmul 全序列并行（4.1 §2.5），代价是推理时 KV cache 随 $T$ 线性增长、$O(T^2)$ 训练复杂度。两条路线各拿走一头：

```
                训练并行?       推理 per-token?      长 context 友好?
   RNN/LSTM       ✗               O(d²)                  弱
   Transformer    ✓               O(T·d)（KV cache）     一般
   Mamba (SSM)    ✓               O(d²)                  强
```

Mamba 的"野心"就是这张表的最后一行——**训练时和 Transformer 一样并行，推理时和 RNN 一样 $O(1)$ state、$O(d^2)$ per token**。怎么做到？答案是把 RNN 的非线性递推 $h_t = \tanh(W_h h_{t-1} + W_x x_t)$ 换成**线性**递推 $x_t = \bar A x_{t-1} + \bar B u_t$——线性递推有一个魔法性质：可以写成"前缀积 + 前缀和"的形式，用 **parallel scan** 在 $O(\log n)$ 并行 depth 里算完 $n$ 步。这是整条 SSM 路线的核心技术筹码。

### 1.2 SSM 是什么——从控制论搬过来的老朋友

State Space Model 是控制论 / 信号处理里 60 年代就成熟的工具。把任意线性时不变系统写成两条方程：

```
        u(t) ──► [ ẋ = A x + B u ]──► x(t) ──► [ y = C x + D u ]──► y(t)
                  状态演化（隐式记忆）              状态读出 + 跳连
```

- **$u(t)$**：input（一维或多维信号）
- **$x(t)$**：hidden state（系统的"记忆"）
- **$y(t)$**：output
- **$A, B, C, D$**：四个矩阵，分别管"state 怎么演化"、"input 怎么注入 state"、"state 怎么读出"、"input 怎么直接跳连到 output"

NLP 里我们处理的是离散 token 序列，所以要把上面的连续 ODE 离散化成 $x_t = \bar A x_{t-1} + \bar B u_t$，$y_t = C x_t + D u_t$——形式与 RNN 的 $h_t = f(h_{t-1}, x_t)$ 几乎一致，差别只在"$f$ 是线性"。一旦是线性，整条序列就可以**并行展开**：

$$x_T = \bar A^T x_0 + \sum_{t=1}^{T} \bar A^{T-t} \bar B u_t$$

这就是 RNN 形式与卷积形式的双面性——training 用"卷积/scan"形式拿并行，inference 用"递推"形式拿 $O(1)$ state。**Mamba 论文标题里的 "Linear-Time Sequence Modeling" 指的就是这件事。**

### 1.3 S4 → Mamba → Mamba-2 的演化轴

```
   2021              2023                       2024
   ─────             ─────                      ─────
   S4                Mamba                      Mamba-2
   (Gu et al.)       (Gu & Dao)                 (Dao & Gu)
   │                 │                          │
   │ HiPPO 初始化    │ + Selective Mechanism    │ + State Space Duality
   │ 长程记忆        │   (A, B, C 依赖输入)     │   (SSM ≡ attention 的揭示)
   │ LRA SOTA        │ + 硬件友好 scan kernel   │ + matrix-form recurrence
   │ 但 LM 不行      │ → 1.4B 接近 LLaMA        │ → 训练 2-8× 提速
                       Selective State Space
```

记忆口诀：

- **S4** 解决"SSM 在长序列任务上能不能赢" → 用 HiPPO 矩阵把 $A$ 初始化成"理论上能记住任意长历史"的形态。
- **Mamba** 解决"SSM 能不能做 language modeling" → 让 $A, B, C$ **依赖于输入** $u_t$（input-dependent / selective），让模型自己决定每步记什么忘什么；再配 hardware-aware kernel 让真实训练吞吐和 Transformer 比肩。
- **Mamba-2** 解决"SSM 怎么进一步加速 + 与 attention 在数学上是什么关系" → 揭示 SSM 与 attention 的**对偶**（State Space Duality, SSD），用 multi-head 风格 + matrix-form recurrence 让训练再提速。

下文 §2 把这三步逐个写公式。

### 1.4 Mamba 的优势场景与短板（30 秒版）

**优势**：

- 推理 per-token 是 $O(d^2)$ 的固定成本，**不随 context 长度增长**——超长 context（>1M token）推理是 attention 的硬伤，Mamba 天然占优
- state 是固定 size（如 16），不像 KV cache 线性涨——边缘设备 / 流式生成友好

**短板**：

- **In-context learning 弱**：固定 size 的 state 容纳不下"任意 few-shot example"的所有细节，针对"需要精确 retrieve"的任务（needle-in-haystack）远差于 attention
- **生态不成熟**：vLLM / FlashAttention / LoRA / DPO 工具链都建立在 attention 上，Mamba 多数要自定义 CUDA kernel
- **未充分 scale**：公开最大 Mamba-2 是 2.8B，超过这个尺寸的 scaling 行为没充分验证；同期 Transformer 已经到了 R1 671B / LLaMA-3 405B

所以 2024-2026 的现实工程态度是：**Transformer 是主流，Mamba 在"长 context 推理 + 流式 + 边缘"做补位，Hybrid (Jamba) 是折中方案**。

---

## 2. 公式与原理

### 2.1 连续 SSM → 离散 SSM

经典控制论的连续状态空间模型：

$$\dot x(t) = A x(t) + B u(t), \qquad y(t) = C x(t) + D u(t)$$

其中 $u(t) \in \mathbb{R}$（标量 input，多维时按维度独立），$x(t) \in \mathbb{R}^N$ 是 hidden state（$N$ 是 SSM state size，Mamba 默认 $N = 16$），$y(t) \in \mathbb{R}$。维度：$A \in \mathbb{R}^{N \times N}$、$B \in \mathbb{R}^{N \times 1}$、$C \in \mathbb{R}^{1 \times N}$、$D \in \mathbb{R}$。

NLP 里的输入是离散 token 序列，所以要离散化成时间步 $t = 1, 2, \dots, T$。常用 **Zero-Order Hold (ZOH)**：假设 input 在采样间隔 $\Delta$ 内保持常数，则可解出闭式：

$$\bar A = \exp(\Delta A), \qquad \bar B = (\Delta A)^{-1} (\exp(\Delta A) - I) \cdot \Delta B$$

离散后的递推：

$$\boxed{x_t = \bar A x_{t-1} + \bar B u_t, \qquad y_t = C x_t + D u_t}$$

这就是离散 SSM 的核心。**$D u_t$ 是 skip connection，工程上常省略或单独实现；本节后面公式默认省略 $D$。**

形状提醒：每一步 $u_t \in \mathbb{R}$、$x_t \in \mathbb{R}^N$、$y_t \in \mathbb{R}$。多维 input（如 $d$ 维 embedding）的处理是**按维度独立**跑 $d$ 个 SSM（每个 channel 一组 $A, B, C$），所以 SSM 总参数量是 $d \cdot O(N^2)$，比 attention 的 $O(d^2)$ 小很多。

### 2.2 两种等价形态：递推 vs 卷积/scan

把上面的递推展开：

$$x_t = \bar B u_t + \bar A \bar B u_{t-1} + \bar A^2 \bar B u_{t-2} + \dots + \bar A^{t-1} \bar B u_1$$

代入 $y_t = C x_t$：

$$y_t = \sum_{k=0}^{t-1} C \bar A^k \bar B \cdot u_{t-k}$$

这是一个**卷积**：$y = K * u$，其中卷积核是

$$\bar K = (C \bar B,\ C \bar A \bar B,\ C \bar A^2 \bar B,\ \dots,\ C \bar A^{T-1} \bar B) \in \mathbb{R}^T$$

**这就是 SSM 的双面性**：

| 形态 | 用在哪 | 复杂度 |
|---|---|---|
| 递推形态 $x_t = \bar A x_{t-1} + \bar B u_t$ | **推理**：每步 $O(N^2)$，state $O(N)$ | 串行但极快 |
| 卷积形态 $y = \bar K * u$ | **训练**：FFT 卷积 $O(T \log T)$ | 全序列并行 |
| Parallel scan | **训练**：$O(T)$ work，$O(\log T)$ depth | 在 GPU 上比 FFT 还快 |

S4 用的是 FFT 卷积形态，Mamba 用的是 parallel scan。**Parallel scan** 是 GPU 友好的"并行前缀和"算法（Blelloch 1990），把"$x_t$ 依赖 $x_{t-1}$"的串行依赖改造成树状归并：第一轮把相邻两步合并成"等价的两步累计", 第二轮把"两步累计"两两合并成"四步累计"……$\log_2 T$ 轮做完。每一轮 GPU 完全并行。

### 2.3 S4：HiPPO 初始化让 SSM 在长序列上开窍

S4（Gu et al. 2021）的核心贡献不是 SSM 本身（控制论 60 年代就有），而是**让 SSM 在 NLP 任务上能 work**——靠的是 $A$ 矩阵的特殊初始化：**HiPPO**（High-order Polynomial Projection Operators）。

直觉：随机初始化的 $A$ 在做指数 $\bar A^t = \exp(t A)$ 时，要么衰减到 0（state 全忘）要么爆炸到 $\infty$（state 不稳）。HiPPO 把 $A$ 初始化成一个特定结构的矩阵——具体是 Legendre 多项式的某种基底——让 $x_t$ 等价于"对历史 input 序列做某组正交多项式的拟合系数"。这相当于 state 自动维护一个"压缩版历史摘要"，理论上能保留任意长程信息。

S4 在 **Long Range Arena (LRA)** benchmark 上把多项 Transformer 远远甩开（PathX 任务上 Transformer 几乎随机 50%，S4 拿到 96%），但**在 language modeling 上不如同 size Transformer**——原因下一节讲。

### 2.4 Mamba：Selective Mechanism 让 SSM 能做 LM

S4 不能做 LM 的根本原因：**$A, B, C$ 是固定参数，与输入 $u_t$ 无关**。这意味着 SSM 对每个 token 都"一视同仁"地推进 state，不能根据"当前 token 是什么"动态决定"是该记还是该忘"。语言里有大量需要"选择性记忆"的场景——比如读到 "Mary said: ..." 之后要把 "Mary" 这个名字记进 state，读到一段废话时要选择性忘掉。固定 $A, B, C$ 做不到。

Mamba（Gu & Dao 2023）的核心改动：**让 $\bar B, C, \Delta$ 依赖输入 $u_t$**，称为 **Selective Mechanism / Selective SSM (S6)**：

$$B_t = \mathrm{Linear}_B(u_t), \quad C_t = \mathrm{Linear}_C(u_t), \quad \Delta_t = \mathrm{softplus}(\mathrm{Linear}_\Delta(u_t))$$

然后离散化也依赖输入：$\bar A_t = \exp(\Delta_t A)$，$\bar B_t = \Delta_t \cdot B_t$（论文用了简化的 Euler 离散，不是 ZOH）。$A$ 矩阵本身仍是 trainable parameter，但**不依赖输入**——保持结构稳定。

新的递推：

$$\boxed{x_t = \bar A_t x_{t-1} + \bar B_t u_t, \qquad y_t = C_t x_t}$$

直观解读：

- $\Delta_t$ 大 → $\bar A_t = \exp(\Delta_t A)$ 接近 0（如果 $A$ 是稳定的），意味着"忘掉旧 state、当前 input 占主导"——读到重要 token 时 $\Delta_t$ 大
- $\Delta_t$ 小 → $\bar A_t$ 接近单位阵，"保留旧 state、忽略当前 input"——读到废话时 $\Delta_t$ 小
- $C_t$ 依赖输入 → 模型可以根据当前 token 动态决定"读出 state 的哪一部分"

**代价**：input-dependent 之后**不能再用 FFT 卷积形态**了——$\bar K$ 不再是固定的，而是每步都变。所以 Mamba 必须用 parallel scan，且 scan 要在 SRAM 里完成、不能把 $T \times d \times N$ 的中间 state 写到 HBM。这就是论文 §3.3 的 **Hardware-Aware Algorithm**：

- 把 $A, B, C, \Delta$ 在 HBM 里
- Parallel scan 中间的 state $x_t$ 全部留在 SRAM（每个 SM 的 shared memory），算完直接给下一阶段用
- 类似 FlashAttention 的 IO-aware 思路——**减少 HBM 读写而非减少 FLOPs**

最终训练吞吐：Mamba-1.4B 与同 size Transformer 在 A100 上吞吐相当；推理时长 context 下显著更快（Mamba 推理 per-token 复杂度与 $T$ 无关）。

### 2.5 Mamba-2：State Space Duality 与 multi-head 风格

Mamba-2（Dao & Gu 2024）的两个核心贡献：

**(1) State Space Duality (SSD)** —— 揭示一类结构化 SSM 与一类 masked attention **数学等价**：

$$y = (\bar L \odot Q K^\top) v$$

其中 $Q, K$ 来自 SSM 的 $C, B$，$\bar L$ 是一个由 $\bar A_t$ 决定的"下三角衰减矩阵"——它的 $(i, j)$ 元素是 $\prod_{t=j+1}^{i} \bar A_t$（当 $j \le i$，否则 0）。

直觉：把 SSM 的"sequential scan"形态写成 matrix form 后，它**就是一个带衰减 mask 的 linear attention**。这把 SSM 与 attention 之间的桥梁正式架起来——很多原本只能用 attention 的 trick（multi-head、tensor parallel、grouped）现在都能用到 SSM 上。

**(2) Multi-head 风格的 SSM head + matrix-form recurrence**：

- 把 SSM 的 channel 分成 $h$ 个 head，每个 head 共享一组 $A$ 但有独立的 $B, C$（类似 GQA 共享 KV、独立 Q）
- 用 matrix multiplication 替代部分 scan，吃满 GPU 的 tensor core（scan 用 CUDA core，效率不如 tensor core）
- 实测训练速度是 Mamba-1 的 2-8×

**性能**：Mamba-2 2.7B 在 Pile / SlimPajama 上与同 size Transformer 持平或略好；在 needle-in-haystack 等"精确检索"任务上仍然落后。

### 2.6 Mamba vs Transformer 全维度对比

这张表是本节最该背的——任何 Mamba 相关面试都会问这几条：

| 维度 | Transformer | Mamba (SSM) |
|---|---|---|
| **推理时间 (per token)** | $O(T \cdot d)$（要算当前 token 与所有 KV 的 attention） | $O(d^2 + d N)$（state 固定，与 $T$ **无关**） |
| **推理显存** | KV cache 随 $T$ 线性增长 | SSM state 固定 size $d \cdot N$ |
| **训练时间 (per step)** | $O(T^2 \cdot d)$（attention） | $O(T \cdot d \cdot N)$（parallel scan，work 线性） |
| **训练并行 depth** | $O(\log T)$（matmul） | $O(\log T)$（parallel scan） |
| **In-context learning** | 强（attention 直接 retrieve 任意位置） | 弱（state 容量有限，固定 size） |
| **长 context (>1M)** | 复杂（YaRN/LongRoPE 扩展、KV cache 爆炸） | 天然友好 |
| **needle-in-haystack** | 强 | 弱（state 太小） |
| **软件生态** | 成熟（HF / vLLM / FlashAttention / FSDP / TRL） | 不成熟（自定义 CUDA kernel、SFT/DPO 工具链有限） |
| **大模型规模** | LLaMA-3 405B、R1 671B | Mamba-2 公开最大 2.8B |
| **工程复杂度** | 标准（PyTorch + HF transformers） | 高（依赖 `mamba-ssm` + `causal-conv1d` CUDA kernel） |

注意几个**容易被混淆的点**：

- "Mamba 训练 $O(T \log T)$" 是不严谨的说法——work 是 $O(T)$，**并行 depth** 是 $O(\log T)$；不像 attention 是 $O(T^2)$ work。但因为 attention 的 $O(T^2)$ 是 well-parallelized matmul、Mamba 的 scan 用 CUDA core，**短到中等序列下 attention 实际更快**——Mamba 的优势要在 $T \ge 8\text{k}$ 才显著。
- "推理 $O(1)$" 是指**与 $T$ 无关**，不是指 0 成本——每步仍然是 $O(d^2 + dN)$ 的固定成本。

### 2.7 Hybrid 路线：Jamba 把两者混用

纯 Mamba 在精确检索类任务上的硬伤短期内难以彻底解决，但完全用 attention 又拿不到 Mamba 的长 context 优势。**Hybrid 架构**就是折中：在 Transformer block 之间穿插 Mamba block。

代表作 **Jamba** (AI21 2024, 52B / 12B active 的 MoE+Mamba hybrid)：

```
   ┌───────────────────────────────────┐
   │  Transformer block (with attn)    │   ← 1 个，处理需要 retrieve 的位置
   ├───────────────────────────────────┤
   │  Mamba block                      │   ┐
   │  Mamba block                      │   │ 7 个，处理大部分 token
   │  Mamba block                      │   │
   │  ...                              │   ┘
   ├───────────────────────────────────┤
   │  Transformer block (with attn)    │
   ├───────────────────────────────────┤
   │  Mamba block × 7                  │
   └───────────────────────────────────┘
   每 8 个 block 里 1 个是 attention，7 个是 Mamba
```

实证结果：

- 比纯 Mamba 在 needle-in-haystack / few-shot 任务上明显更好
- 比纯 Transformer 推理快、长 context 显存省（因为 attention 层只占 1/8）
- 但仍然受制于那 1/8 的 attention 层——KV cache 还是要存

类似思路的还有 **Zamba**、**Samba**、**Hymba** 等。**Hybrid 是 2024-2026 SSM 路线在工业上最现实的形态。**

### 2.8 RWKV 与 RetNet——同一阵营的另外两条线

Mamba 不是唯一的"非 attention LLM"路线，2023 年同期还有两个值得知道的方向：

- **RWKV** (Peng 2023)：把 attention 改写成 receptance-weighted RNN 形式，递推可以并行训练。已经做到 14B+，社区活跃。优势：训练效率高、推理 $O(1)$ state；劣势：与 Mamba 类似，长程精确检索弱。
- **RetNet** (Sun 2023)：Microsoft 提出的"Retentive Network"，用一个带衰减的 retention 替代 attention，三种等价形态（recurrent / parallel / chunkwise）覆盖训练 + 推理。

它们与 Mamba 同属"非 attention LLM"阵营，思路都是"用某种线性/half-linear 形式既保 RNN 的 $O(1)$ 推理、又保 Transformer 的并行训练"。**面试时如果问"Transformer 之外的 LLM 架构"，能把 SSM (Mamba) / RWKV / RetNet 三条线说出来就够了。**

---

## 3. 最小代码示例

### 3.1 朴素 discrete SSM forward（不带 selective、不带 scan）

仅用 PyTorch 原生算子展示**离散 SSM 的数学模型**——真正的 Mamba 训练需要自定义 CUDA scan kernel，Python for loop 会比 attention 慢 100×，**这段代码只用来理解公式，不要用来训练**。

```python
import torch
import torch.nn as nn

class NaiveSSM(nn.Module):
    """最朴素的 discrete SSM，单 channel + 固定 A,B,C,D（非 selective）。
    数学上对应 §2.1 公式：x_t = A_bar x_{t-1} + B_bar u_t,  y_t = C x_t + D u_t
    """
    def __init__(self, state_size: int = 16):
        super().__init__()
        self.N = state_size                            # SSM state 维度 N
        # A, B, C, D 都是 trainable parameter
        self.A = nn.Parameter(torch.randn(state_size, state_size) * 0.01)
        self.B = nn.Parameter(torch.randn(state_size, 1))
        self.C = nn.Parameter(torch.randn(1, state_size))
        self.D = nn.Parameter(torch.zeros(1))          # skip connection
        self.log_delta = nn.Parameter(torch.zeros(1))  # 离散化步长 Delta（log 参数化保正）

    def discretize(self):
        """Zero-Order Hold 离散化：A_bar = exp(Delta * A), B_bar = (Delta * A)^{-1} (A_bar - I) * Delta * B"""
        delta = torch.exp(self.log_delta)
        A_bar = torch.matrix_exp(delta * self.A)       # (N, N)
        # 简化版用 Euler：A_bar ≈ I + delta * A,  B_bar ≈ delta * B（与 Mamba 论文 §2 一致）
        B_bar = delta * self.B                         # (N, 1)
        return A_bar, B_bar

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        # u: (B, T) 单 channel input；多 channel 时按维度独立跑 d 个 SSM
        B_size, T = u.shape
        A_bar, B_bar = self.discretize()
        x = torch.zeros(B_size, self.N, device=u.device)   # 初始 state
        ys = []
        for t in range(T):                              # ← for loop，仅演示用！
            u_t = u[:, t:t+1]                           # (B, 1)
            x = x @ A_bar.T + u_t @ B_bar.T             # (B, N), 公式 x_t = A_bar x_{t-1} + B_bar u_t
            y_t = x @ self.C.T + u_t * self.D           # (B, 1), 公式 y_t = C x_t + D u_t
            ys.append(y_t)
        return torch.stack(ys, dim=1).squeeze(-1)       # (B, T)

# 使用示例
model = NaiveSSM(state_size=16)
u = torch.randn(2, 128)                                  # batch=2, T=128
y = model(u)
print(y.shape)                                           # torch.Size([2, 128])
```

关键点：

- **for loop 是为了让公式可读**——真正的 Mamba 用 `mamba-ssm` 包里的 `selective_scan_fn`，CUDA kernel 在 SRAM 里做 parallel scan，不实例化中间 state 到 HBM
- $A$ 在 Mamba 里用**对角 + 复数初始化**（结构化 SSM, S4D）而不是稠密随机，保证数值稳定与计算高效；这里为了简化用稠密随机
- 这段代码**没有 selective mechanism**——$A, B, C$ 与 $u$ 无关。Selective 版本要把 `B, C, log_delta` 改成由 `nn.Linear(d_model, ·)(u_t)` 动态生成（见 Mamba 论文 Algorithm 2）

### 3.2 用 `mamba-ssm` 库调一个真 Mamba block

`mamba-ssm` 是 Tri Dao 维护的官方实现，封装了 selective scan CUDA kernel。安装：

```bash
pip install mamba-ssm causal-conv1d
```

最小调用：

```python
import torch
from mamba_ssm import Mamba

# Mamba block: 输入输出形状一致 (B, T, d_model)
mamba_block = Mamba(
    d_model=512,        # hidden dim
    d_state=16,         # SSM state size N（默认 16，调大改善记忆但提升计算）
    d_conv=4,           # 前置 1D conv 的 kernel size
    expand=2,           # block 内部把 d_model 扩展到 expand*d_model
).cuda()

x = torch.randn(2, 1024, 512).cuda()    # batch=2, T=1024, d=512
y = mamba_block(x)                       # (2, 1024, 512)
print(y.shape)                           # torch.Size([2, 1024, 512])
```

注意：

- **必须装 `causal-conv1d`**——Mamba block 内部有一个前置的 causal 1D conv，加速版要 CUDA kernel 支持。装不上 / 不在 GPU 上时会 fallback 到极慢的 PyTorch 实现
- 接口与 `nn.MultiheadAttention` 几乎一样（输入输出 `(B, T, d_model)`），可以直接替换 Transformer 里的 attention block 做 Hybrid 实验
- HuggingFace `transformers` 也已经原生支持 Mamba（`MambaForCausalLM`、`MambaConfig`），可以像加载 LLaMA 一样加载预训练 Mamba 检查点

---

## 4. 工程踩坑与经验

- ❗ **Mamba 的"$O(T \log T)$ 训练 / $O(T)$ 推理"依赖 parallel scan 的高效 CUDA kernel；纯 Python for loop 会比 attention 慢 100× 以上**。新手照着公式手写 for loop scan 然后报告"Mamba 训练比 Transformer 慢 10 倍"——是的，没有 `mamba-ssm` 的自定义 kernel，Mamba 在 PyTorch 上是个玩具。**实战必须用 `mamba_ssm.ops.selective_scan_interface.selective_scan_fn`**，不要自己写 scan。
- ❗ **Mamba 的 in-context learning 显著弱于同 size Transformer**——few-shot prompting 效果不如 LLaMA-1.4B。固定 size 的 SSM state（默认 $N = 16$，每 channel）容纳不下 5-shot example 的所有细节。如果你的应用重度依赖 in-context learning（few-shot CoT、新格式适应），不要选纯 Mamba；用 Hybrid 或回到 Transformer。
- ❗ **Needle-in-haystack 任务 Mamba 显著差于 Transformer**。把一句"密码是 7392"埋在 50k token 废话中，attention 能直接 retrieve 出来，Mamba state 早就被冲掉了。这不是 bug，是 SSM 的**结构性限制**——固定 state size 没法做"任意位置精确检索"。Hybrid 加少量 attention 层是当前最现实的解。
- ❗ **想 fine-tune Mamba 的 SFT/DPO 工具链不成熟**。`trl` 截至 2026 仍以 attention 为一等公民，DPO / GRPO 的 reference model log-prob 计算、KV cache 复用、LoRA 注入都是按 attention 设计的；用到 Mamba 上要么 patch 框架要么忍受性能。`peft` 对 Mamba 的 LoRA 支持也较新、不稳定。**生产环境跑 Mamba SFT，准备好踩坑预算。**
- ❗ **Hybrid Mamba-Transformer 的 attention layer 仍受 KV cache 限制；不能"完全摆脱" long context 推理瓶颈**。Jamba 每 8 个 block 1 个 attention，KV cache 是纯 Transformer 的 1/8，但仍随 $T$ 线性涨——128k context 下还是不小。彻底摆脱 KV cache 只有**纯 SSM** 一条路，但代价就是上面提到的 in-context / retrieval 弱化。
- ❗ **Mamba 论文公开的最大尺寸是 2.8B（Mamba-2-2.7B），超过这个尺寸的 scaling law 行为没充分验证**。同期 Transformer 已到 405B / 671B 量级。"Mamba 在小尺寸接近 Transformer" ≠ "Mamba 在大尺寸还能接近 Transformer"——这是开放问题，不要按小模型经验外推。Hybrid 路线（Jamba）至少有 52B 公开数据点，更值得参考。
- ❗ **SSM state size $N$ 调大不是免费的**——Mamba 默认 $N = 16$ 看似很小，但 selective SSM 的 scan 复杂度是 $O(T \cdot d \cdot N)$，state 计算的中间 tensor 是 `(B, T, d, N)`。$N$ 翻倍内存与计算都翻倍。调到 $N = 64$ 已经会显著拖慢训练，要权衡。
- ❗ **`d_conv` 这个前置 1D conv 不是装饰**——Mamba block 在 SSM 之前有一个 kernel size = 4 的 causal conv1d，作用类似 short-range bias。去掉它效果会明显下降（论文 ablation 有），但要记得它是 causal conv（左侧 padding），不是普通 conv。
- ❗ **离散化方式（ZOH vs Euler vs Bilinear）会影响数值稳定性**。Mamba 论文实际上用的是 Euler 简化（$\bar A \approx I + \Delta A$）而不是严格 ZOH（$\bar A = \exp(\Delta A)$），因为 $\exp$ 矩阵在 fp16 下数值不稳。自己实现时不要照抄数学课本的 ZOH 公式。

---

## 5. 经典 paper

- **Gu, Goel & Ré, 2021 — Efficiently Modeling Long Sequences with Structured State Spaces (S4)** — SSM 在 NLP 任务上的开山作。读它能搞清楚为什么 HiPPO 初始化能解决长程记忆、为什么离散 SSM 等价于卷积、以及 LRA benchmark 上 SSM 完胜 Transformer 的实证根据。本节 §2.3 就是它的浓缩。
- **Gu & Dao, 2023 — Mamba: Linear-Time Sequence Modeling with Selective State Spaces** — Mamba 原典，必读。核心读 §3.2 selective mechanism（让 $B, C, \Delta$ 依赖 input 是 SSM 突破 LM 任务的钥匙）和 §3.3 hardware-aware algorithm（parallel scan + SRAM-resident state，类似 FlashAttention 的 IO-aware 思路）。
- **Dao & Gu, 2024 — Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality (Mamba-2)** — 揭示 SSM 与 attention 的数学对偶。这是 SSM 路线最具理论深度的论文，理解 SSD 之后会发现"SSM vs attention" 的二元对立其实是同一连续谱上的两个极端。论文同时给出了训练 2-8× 提速的工程方案。
- **Lieber et al., 2024 — Jamba: A Hybrid Transformer-Mamba Language Model (AI21)** — Hybrid 范式的代表作，52B / 12B active MoE+Mamba。读它能看清"工业上怎么把 Mamba 用起来"——纯 Mamba 落地难，Hybrid 是当前最务实的路径。
- 加分：**Peng et al., 2023 — RWKV** 与 **Sun et al., 2023 — RetNet**——同期非 attention LLM 的另外两条线，与 Mamba 同属一个阵营，能体会"如何兼顾 RNN 的 $O(1)$ 推理与 Transformer 的并行训练"这个公共目标的多种解法。

---

## 6. 自测与面试题

**Q1（数学）**：写出 discrete SSM 的递推公式，并解释 Mamba 的 selective mechanism 与 S4 的核心区别是什么。

<details>
<summary>Answer sketch</summary>

discrete SSM 的递推（必背）：

$$x_t = \bar A x_{t-1} + \bar B u_t, \qquad y_t = C x_t + D u_t$$

其中 $u_t \in \mathbb{R}$、$x_t \in \mathbb{R}^N$、$y_t \in \mathbb{R}$；$\bar A, \bar B$ 由连续参数 $A, B$ 经离散化（ZOH 或 Euler）得到，离散化步长是 $\Delta$。

Selective mechanism vs S4 的核心区别：

- **S4**：$A, B, C, \Delta$ 都是**与输入无关**的固定 trainable parameter；好处是可以写成卷积形式 $y = \bar K * u$ 用 FFT 训练，坏处是模型对每个 token "一视同仁"，不能根据当前 token 决定记/忘
- **Mamba (selective SSM, S6)**：让 $\bar B, C, \Delta$ **依赖输入**，即 $B_t = \mathrm{Linear}_B(u_t)$ 等；$A$ 仍然不依赖输入。这样模型能根据当前 token 动态决定"$\Delta_t$ 大 → 忘旧记新 / $\Delta_t$ 小 → 保旧忽略新"
- 代价：input-dependent 之后**不能再用 FFT 卷积**，必须用 parallel scan，且 scan 要在 SRAM 里完成（hardware-aware kernel）

加分：能指出这是 SSM 突破 language modeling 任务的关键改动，因为语言天然需要"选择性记忆"（如人名、关键事实要记，废话要忘）。

</details>

**Q2（trade-off）**：Mamba vs Transformer 在 (推理时间 / 训练吞吐 / in-context learning / 长 context) 4 个维度对比，每个维度说清谁强谁弱、为什么。

<details>
<summary>Answer sketch</summary>

| 维度 | Transformer | Mamba | 谁赢 |
|---|---|---|---|
| 推理时间 (per token) | $O(T \cdot d)$（要算与所有 KV 的 attention） | $O(d^2 + dN)$（state 固定，与 $T$ 无关） | **Mamba**（尤其长 $T$） |
| 训练吞吐 | $O(T^2 d)$ work，全 matmul，吃满 tensor core | $O(T d N)$ work，scan 用 CUDA core 效率略低 | **短/中序列 Transformer**；$T \ge 8\text{k}$ Mamba 反超 |
| In-context learning | 强（attention 直接 retrieve 任意位置 example） | 弱（固定 state size 容纳不下 5-shot 全部细节） | **Transformer** |
| 长 context (>1M) | 复杂（YaRN/LongRoPE 扩展、KV cache 显存爆炸） | 天然友好（state 固定 size、与 $T$ 无关） | **Mamba** |

要点解释：

- 推理优势：Mamba 的 state 是固定大小（$d \cdot N$），不像 KV cache 线性涨——所以 1M context 推理时显存差距能上百倍
- 训练 trade-off：Mamba 的 work 是 $O(T)$ 而 attention 是 $O(T^2)$，看似 Mamba 必胜；但 attention 的 $O(T^2)$ 是 well-parallelized matmul（吃 tensor core），Mamba 的 scan 用 CUDA core 效率不如，**短序列下 attention 实际更快**
- ICL 弱化的根因：固定 size state 是"压缩历史摘要"，无法做"任意位置精确检索"；attention 的 $n \times n$ 相似度矩阵天然支持任意检索
- 长 context：是 Mamba 最大卖点；同时也是 attention 的硬伤（KV cache 爆炸、$O(T^2)$ 计算）

加分：能提到具体数字（Mamba state size 默认 $N = 16$；Mamba-2 公开最大 2.8B；Mamba 优势在 $T \ge 8\text{k}$ 才显著）；能提到 Hybrid（Jamba）作为折中方案。

</details>

**Q3（前沿）**：为什么 2024-2026 主流 LLM 仍然是 Transformer 而不是 Mamba？至少给 3 个原因，并说出 Mamba 哪些场景仍有不可替代性。

<details>
<summary>Answer sketch</summary>

Transformer 主导的原因（至少 3 条）：

1. **生态完整 / 工具链成熟**：vLLM / FlashAttention / FSDP / TRL / PEFT / DeepSpeed 等几乎所有 LLM 基础设施都是基于 attention 设计的；Mamba 的 SFT / DPO / LoRA 工具链截至 2026 仍不成熟
2. **In-context learning 与 needle-in-haystack 弱**：Mamba 的固定 state 容纳不了"任意位置精确 retrieve"，对需要 few-shot 适应、长文档检索的实际应用是硬伤
3. **未充分 scale 验证**：公开最大 Mamba-2 是 2.8B，与 Transformer 405B / 671B 相差两个数量级，scaling law 行为不明
4. **训练 wall-clock 优势不明显**：Mamba 的 $O(T)$ work 优势要到 $T \ge 8\text{k}$ 才能盖过 attention 的 tensor core 优势；多数预训练数据 seqlen 还在 2k-8k
5. **工程复杂度高**：依赖自定义 CUDA kernel（`mamba-ssm` + `causal-conv1d`），对 ROCm / CPU / 移动端等非 NVIDIA-CUDA 环境支持差

Mamba 不可替代的场景：

- **超长 context (>1M token) 推理**：state 固定，KV cache-free
- **流式 / 实时生成**：每 token $O(1)$ 状态更新，天然增量
- **边缘设备 / 移动端 LLM**：无 KV cache 显存优势在 7B 以下小模型 + 长上下文场景显著
- **长序列时序信号**：DNA / 音频 / 时序传感器等天然超长且 attention 处理不了的领域（DNA Mamba / Caduceus 在 genomics 已是 SOTA）

务实结论（加分）：**Hybrid 路线（Jamba / Zamba / Samba）是 2024-2026 工业上最现实的形态**——保留 attention 的 retrieval 能力 + Mamba 的长 context 推理优势；纯 Mamba 在通用 LLM 战场短期内不会颠覆 Transformer。

差答："Mamba 性能不如 Transformer 所以没人用"——没看到 Mamba 在长 context 与流式上的真实优势；或者"Mamba 一定会取代 Transformer"——没看到 ICL / 生态 / scale 的硬约束。

</details>

---

## 7. 延伸阅读

- [Mamba 原 paper (arXiv 2312.00752)](https://arxiv.org/abs/2312.00752) — Gu & Dao 2023，核心读 §3.2-§3.3
- [Mamba-2 原 paper (arXiv 2405.21060)](https://arxiv.org/abs/2405.21060) — Dao & Gu 2024，State Space Duality 与 SSD 算法
- [`state-spaces/mamba` 官方实现](https://github.com/state-spaces/mamba) — Tri Dao 维护的 reference implementation，含 selective scan CUDA kernel + 预训练 1.4B / 2.8B 检查点
- [Sasha Rush — The Annotated S4 (blog)](https://srush.github.io/annotated-s4/) — S4 的逐行注释博客，讲清 SSM ↔ 卷积的双面性，是理解 SSM 公式最直观的资源
- [Tri Dao — Mamba 系列讲座 (YouTube)](https://www.youtube.com/results?search_query=tri+dao+mamba) — 作者本人讲 selective mechanism 与 hardware-aware kernel 的设计动机
- [AI21 — Jamba 技术博客](https://www.ai21.com/blog/announcing-jamba) — Hybrid 架构的工程动机与实测数据
- [HuggingFace — `MambaForCausalLM` 文档](https://huggingface.co/docs/transformers/model_doc/mamba) — HF 官方 Mamba 接口，可以像加载 LLaMA 一样加载预训练 Mamba
- 推荐继续读本教程的 **Module 6《预训练》**——回到 Transformer 主线，从 6.1 训练目标开始系统讲 pretrain；Mamba 路线在工业上仍是 niche，主线必须先打牢 Transformer 训练栈
