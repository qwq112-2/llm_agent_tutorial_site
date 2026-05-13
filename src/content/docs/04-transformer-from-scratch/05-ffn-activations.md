---
title: "4.5 FFN 与激活：ReLU / GELU / SwiGLU"
description: "讲清 Transformer block 里\"另一半\"——FFN 子层——的位置、参数量、激活函数从 ReLU → GELU → SwiGLU 的演化路径，以及现代 LLM（LLaMA / Qwen / Mistral）为什么把 SwiGLU 当标配，并把背后那条容易被忽略的工程细节\"$d_{ff}$ 不再是 $4d$\"讲透。"
---

> ⏱ 预计阅读 30 分钟 ｜ 难度 ★★ ｜ 前置：1.1 反向传播、4.1 self-attention（理解 Transformer block 结构）

## 一句话本节讲什么

讲清 Transformer block 里"另一半"——**FFN 子层**——的位置、参数量、激活函数从 ReLU → GELU → SwiGLU 的演化路径，以及现代 LLM（LLaMA / Qwen / Mistral）为什么把 SwiGLU 当标配，并把背后那条容易被忽略的工程细节"$d_{ff}$ 不再是 $4d$"讲透。

---

## 1. Mental model（直觉）

一个标准 Transformer block 由两个子层串联：**self-attention 子层 + FFN 子层**，每个子层各自带 residual + norm（Pre-LN 或 Post-LN，详见 4.4）。两者的分工非常清楚：

- **Self-attention** 负责 **token 间通信**——让每个位置能根据上下文聚合其他位置的信息（4.1 已讲）
- **FFN** 负责 **token 内加工**——对每个位置**独立**做一次非线性变换，token 之间不互相看

也就是说，FFN 是 **token-wise** 的（position-wise feed-forward network 是它的全名）。如果把 attention 看成"横向"操作，FFN 就是"纵向"操作：

```
        token_0  token_1  token_2  ...  token_T
attention   ↔──────↔──────↔             ←── 横向通信
FFN         ↓      ↓      ↓     ↓       ←── 纵向加工
```

直觉上，attention 决定**关注谁**，FFN 决定**怎么处理**关注后的信息——一种"先收集证据、再独立思考"的分工。

更不直觉的是参数分配：很多人以为大模型主要是 attention 在吃参数，**实际上 FFN 才是参数大户**。以 LLaMA-2 7B 为例，attention 部分（QKV + output projection）合计约 1B 参数，FFN 部分（W1 + W2 + W3）约 5B 参数，**FFN 占总参数 60-70%、占总计算量也大致同比例**。后面 5.4 MoE 之所以专门把 FFN sparse 化，8.3 LoRA 之所以经常加在 FFN 矩阵上、Module 11 推理量化之所以重点优化 FFN，都是因为这一节要讲的"FFN 是参数与算力的主战场"。

第二层直觉：**为什么要先升维再降维？** 经典 FFN 的形态是 $d \to d_{ff} \to d$，中间维度 $d_{ff}$ 通常是 $4d$（原版 Transformer）甚至 $\frac{8}{3}d$（LLaMA）。可以这么理解：单个 token 在 attention 输出后已经是一个 $d$ 维向量，但这个向量"挤"在低维空间里很多模式互相干扰；把它升到一个更宽的高维空间，每个隐藏单元可以"专门负责"一个模式（类比 sparse coding），处理完再投回低维空间。"宽 + 浅"（一层 FFN，宽 4 倍）在实证上比"窄 + 深"（多层 FFN，每层等宽）更划算。

---

## 2. 公式与原理

### 2.1 经典 ReLU FFN（Vaswani 2017）

原版 Transformer 的 FFN 公式非常朴素：

$$\text{FFN}(x) = W_2 \cdot \text{ReLU}(W_1 x + b_1) + b_2$$

其中：

- $x \in \mathbb{R}^{d}$ 是单个 token 的隐藏向量（实际 batch 是 $(B, T, d)$，FFN 在最后一维上独立作用）
- $W_1 \in \mathbb{R}^{d_{ff} \times d}$，$b_1 \in \mathbb{R}^{d_{ff}}$ —— 升维
- $W_2 \in \mathbb{R}^{d \times d_{ff}}$，$b_2 \in \mathbb{R}^{d}$ —— 降维
- $d_{ff} = 4d$ —— 这是 Vaswani 等人的默认配方（base 模型 $d=512, d_{ff}=2048$；large 模型 $d=1024, d_{ff}=4096$）

参数量：$2 d \cdot d_{ff} = 8 d^2$（忽略 bias）。这个 $8d^2$ 是后面所有架构对比的基准。

### 2.2 激活函数演化

**ReLU**：$\text{ReLU}(x) = \max(0, x)$。简单、计算便宜（一次比较 + 一次乘法），但有 **dead neuron** 问题——一旦某个隐藏单元的预激活长期为负，梯度恒为 0，这个单元永远学不到东西。Transformer 时代早期影响不大（参数多、冗余足），但更精细的下游任务上会损失一点点表达力。

**GELU（Gaussian Error Linear Unit）**：$\text{GELU}(x) = x \cdot \Phi(x)$，其中 $\Phi$ 是标准正态分布的 CDF。直觉是"按概率门控"——输入 $x$ 大概率（$\Phi(x)$ 大）就放行，否则压抑。常见的 tanh 近似形式：

$$\text{GELU}(x) \approx 0.5 x \left(1 + \tanh\!\left(\sqrt{2/\pi} \cdot (x + 0.044715 x^3)\right)\right)$$

GELU 是 **smooth 版的 ReLU**：在 0 附近平滑可微、在负数区允许少量梯度通过、在正数区接近线性。**BERT、GPT-2、GPT-3 全用 GELU**，比 ReLU 在常见 NLP benchmark 上能涨 0.5-1 个百分点。

**SiLU / Swish**：$\text{Swish}(x) = x \cdot \sigma(x)$，其中 $\sigma$ 是 sigmoid。曲线与 GELU 几乎重合（差异 < 1%），但只需要一次 sigmoid，比 GELU 的 erf / tanh 近似更便宜。PyTorch 里就是 `F.silu(x)`，silu 是 Swish 的另一个名字。

**SwiGLU（Shazeer 2020 提出，LLaMA 起广泛采用）**：在 Swish 基础上引入 **GLU（Gated Linear Unit）** 思想。原始 GLU 用 sigmoid 门：$\text{GLU}(x) = \sigma(W_1 x) \odot (W_2 x)$。SwiGLU 把门换成 Swish：

$$\text{SwiGLU}(x) = \text{Swish}(W_1 x) \odot (W_2 x)$$

注意右边出现了**两个独立的线性投影** $W_1, W_2$，一路过 Swish 充当门控，另一路保持线性，逐元素相乘——本质是"用一路非线性门 × 一路线性表示"，让网络学会"哪些维度该传、哪些该屏蔽"。完整 FFN 还要再投回 $d$ 维：

$$\text{FFN}_{\text{SwiGLU}}(x) = W_3 \big( \text{Swish}(W_1 x) \odot (W_2 x) \big)$$

其中 $W_1, W_2 \in \mathbb{R}^{d_{ff} \times d}$，$W_3 \in \mathbb{R}^{d \times d_{ff}}$。**多了一个矩阵**。Shazeer 在原 paper 里实证：在等参数预算下，SwiGLU 比 ReLU/GELU FFN 在 GLUE / SuperGLUE / WMT 上稳定涨 1-2 个百分点。

### 2.3 SwiGLU 的参数量调整（容易踩的坑）

经典 ReLU FFN 有 2 个矩阵，SwiGLU 有 3 个矩阵。如果保持 $d_{ff} = 4d$ 不变，SwiGLU 的参数量会从 $8d^2$ 涨到 $12d^2$——多了 50%。直接对比就不公平了（"涨点"可能是参数变多带来的）。

为了**等参数对比**，需要把 $d_{ff}$ 缩小：

$$3 d \cdot d_{ff}^{\text{SwiGLU}} = 8 d^2 \implies d_{ff}^{\text{SwiGLU}} = \frac{8}{3} d \approx 2.67 d$$

这就是 **LLaMA / LLaMA-2 / LLaMA-3 / Qwen / Mistral** 的工程公约：**$d_{ff}$ 取约 $\frac{8}{3} d$，再 round 到 256（或 128 / 64）的倍数**（kernel / TensorCore 友好，详见 4.6 与 Module 7）。

几个具体例子：

| 模型 | $d$（hidden size） | 朴素 $\frac{8}{3} d$ | 实际 $d_{ff}$ | 解释 |
|---|---|---|---|---|
| LLaMA-2 7B | 4096 | 10922.67 | **11008** | 11008 = 43 × 256 |
| LLaMA-2 13B | 5120 | 13653.33 | **13824** | 13824 = 54 × 256 |
| LLaMA-2 70B | 8192 | 21845.33 | **28672** | 70B 的 GQA 结构故意把 FFN 放更大 |
| LLaMA-3 8B | 4096 | 10922.67 | **14336** | LLaMA-3 全系列把 $d_{ff}$ 做大，约 $3.5 d$ |
| LLaMA-3 70B | 8192 | 21845.33 | **28672** | 与 LLaMA-2 70B 持平 |
| Qwen-2.5 7B | 3584 | 9557.33 | **18944** | 阿里把 FFN 做得更激进，约 $5.3 d$ |
| Mistral 7B | 4096 | 10922.67 | **14336** | 与 LLaMA-3 8B 一致 |

可以看到 LLaMA-2 严格遵守 $\frac{8}{3} d$ 公约，而 LLaMA-3 / Qwen-2.5 等新一代为了在固定参数预算下提升能力，**把 $d_{ff}$ 进一步放大**（间接说明 FFN 的"参数 → 性能"边际收益还没饱和）。但**核心规律没变**：现代 LLM 的 $d_{ff}$ 都是按"实际工程考量"设的，**不再是简单的 $4d$**。

---

## 3. 最小代码示例

### 3.1 手写 LLaMA-style SwiGLU FFN

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LLaMA_FFN(nn.Module):
    """LLaMA-style SwiGLU FFN. 注意：bias=False 是现代 LLM 的惯例。"""
    def __init__(self, d: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d, d_ff, bias=False)   # gate projection
        self.w2 = nn.Linear(d, d_ff, bias=False)   # up projection
        self.w3 = nn.Linear(d_ff, d, bias=False)   # down projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d) -> (B, T, d)
        return self.w3(F.silu(self.w1(x)) * self.w2(x))
```

关键点：

- `F.silu` 即 Swish，PyTorch 原生支持
- `*` 是逐元素乘（element-wise），实现 SwiGLU 的"门 × 线性"
- 三个线性层全部 `bias=False`——LLaMA / Qwen / Mistral 的标准做法（去 bias 不影响效果，反而省一点点参数和访存）
- `w1`（gate）与 `w2`（up）通常在工程上会合成一个矩阵 `gate_up_proj` 一次算完，再切成两半，减少一次 GEMM 调用（vLLM / TGI 都这么做）

### 3.2 ReLU vs GELU vs SwiGLU 的输出对比

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ReLU_FFN(nn.Module):
    def __init__(self, d, d_ff):
        super().__init__()
        self.w1, self.w2 = nn.Linear(d, d_ff), nn.Linear(d_ff, d)
    def forward(self, x): return self.w2(F.relu(self.w1(x)))

class GELU_FFN(nn.Module):
    def __init__(self, d, d_ff):
        super().__init__()
        self.w1, self.w2 = nn.Linear(d, d_ff), nn.Linear(d_ff, d)
    def forward(self, x): return self.w2(F.gelu(self.w1(x), approximate='tanh'))

torch.manual_seed(0)
d, d_ff_classic, d_ff_swiglu = 512, 2048, int(2048 * 2 / 3)  # 等参数对比
x = torch.randn(2, 4, d)
print("ReLU  out norm:", ReLU_FFN(d, d_ff_classic)(x).norm().item())
print("GELU  out norm:", GELU_FFN(d, d_ff_classic)(x).norm().item())
print("SwiGLU out norm:", LLaMA_FFN(d, d_ff_swiglu)(x).norm().item())

# 顺便看看三种激活在 [-3, 3] 上的曲线
xs = torch.linspace(-3, 3, 7)
print("x       :", xs.tolist())
print("ReLU(x) :", F.relu(xs).tolist())
print("GELU(x) :", F.gelu(xs, approximate='tanh').tolist())
print("SiLU(x) :", F.silu(xs).tolist())
```

跑出来会看到 ReLU 在 $x < 0$ 时硬截断为 0，GELU/SiLU 在负数区有"轻微放行"（如 $x=-1$ 时 SiLU $\approx -0.27$），整体曲线更平滑。这种平滑性使梯度更稳定，是 GELU/SiLU 优于 ReLU 的核心原因。

### 3.3 LLaMA-2 7B 的 FFN 参数量计算

```python
d, d_ff = 4096, 11008
n_layers = 32  # LLaMA-2 7B 一共 32 层

# 单层 FFN 三矩阵无 bias
ffn_per_layer = 3 * d * d_ff
total_ffn = n_layers * ffn_per_layer
print(f"单层 FFN 参数: {ffn_per_layer / 1e6:.2f}M")
print(f"全模型 FFN 参数: {total_ffn / 1e9:.2f}B")
# 单层 FFN 参数: 135.27M
# 全模型 FFN 参数: 4.33B  -> 占 7B 模型的 ~62%
```

输出印证了"FFN 占 60-70% 总参数"的说法。同样的算法可以用来快速估算任何 LLM 的 FFN 参数预算。

---

## 4. 工程踩坑与经验

- ❗ **$d_{ff}$ 不是 $4d$**——实现 LLaMA-style SwiGLU 时，要按 $\frac{8}{3} d$ 算再 round 到 256（或更小幂次）的倍数，写成 $4d$ 会导致与开源 ckpt 形状不兼容、加载报错。LLaMA-2 7B 是 11008 不是 16384，新人最容易在这一步翻车
- ❗ **现代 LLM FFN 不要 bias**——LLaMA / Qwen / Mistral 的 FFN 三个矩阵都 `bias=False`。bias 几乎不涨点，去掉省一点参数和一次访存。如果你 fine-tune 时手滑加了 bias，会导致 state_dict key 对不上、weight tying / merge 时也会出错
- ❗ **SwiGLU 比 ReLU FFN 慢 30-50%**——3 矩阵 vs 2 矩阵，FLOPs 多 50%；但 quality / param 提升明显，整体是值得的 trade-off。如果在极度 latency-sensitive 场景（端侧推理、speculative decoding 的 draft 模型）可以考虑回退到 GELU/ReLU FFN，但主流 7B+ 模型已经没人这么省了
- ❗ **`F.gelu` 有两种实现**：默认 `approximate='none'` 用 erf 精确计算，`approximate='tanh'` 用 tanh 近似（$0.5 x (1 + \tanh(\sqrt{2/\pi}(x + 0.044715 x^3)))$）。BERT / GPT-2 训练时用的是 tanh 近似，与开源 ckpt 对齐时**必须**核对——用错近似会让 logits 有 1e-3 量级的差异，对评测和强化学习的梯度非常敏感
- ❗ **FP16 下 SiLU 的 sigmoid 可能下溢**——当输入很负（如 $x < -10$），$\sigma(x)$ 会被 fp16 round 到 0，整段梯度消失。bf16 因为指数位更宽（8 位）而稳得多，这是 bf16 取代 fp16 的核心原因之一。fp8 训练时 activation 还需要专门的 per-tensor / per-block scaling，否则 SwiGLU 的中间张量很容易越界（DeepSeek-V3 论文有详细处理方法）
- ❗ **FFN 占 70% 参数 + 70% 计算 → 推理优化的主战场**——量化（GPTQ/AWQ/FP8，详见 11.4）首先量化 FFN 矩阵；MoE（5.4）把 FFN 替换为多个 expert；LoRA（8.3）默认加在 FFN 的 W1/W2/W3 上。如果你做推理性能 / 显存优化，永远先看 FFN
- ❗ **SwiGLU 中间 activation 显存压力大**：训练时 $W_1 x$ 和 $W_2 x$ 要保留到反向以算梯度，shape 是 $(B, T, d_{ff})$ —— $d_{ff}$ 比 $d$ 大 2.67-5 倍，FFN 是 activation memory 大户。activation recomputation（7.5）默认会重算 FFN 中间量

---

## 5. 经典 paper

- **Vaswani et al., 2017 — Attention is All You Need** — 原版 Transformer，§3.3 Position-wise Feed-Forward Networks 给出了 ReLU FFN 与 $d_{ff} = 4d$ 的初始配方，是后续所有 FFN 变体的对照基准
- **Hendrycks & Gimpel, 2016 — Gaussian Error Linear Units (GELU)** — 首次提出 GELU 激活，给出 $x \cdot \Phi(x)$ 的概率门控解释与 tanh 近似式，BERT / GPT-2/3 时代的事实标准激活
- **Shazeer, 2020 — GLU Variants Improve Transformer** — 仅 5 页的"experimental note"，系统对比 ReLU / GELU / Swish / GLU / GeGLU / SwiGLU，结论是 SwiGLU 全胜。LLaMA / Qwen / Mistral 现代架构选 SwiGLU 的直接源头
- **Touvron et al., 2023 — LLaMA: Open and Efficient Foundation Language Models** — 首个大规模开源采用 SwiGLU 的 LLM，§2.1 明确提到 $d_{ff} = \frac{2}{3} \cdot 4d$ 的等参数 trade-off，把 SwiGLU 从论文实验推上工业事实标准

---

## 6. 自测与面试题

**Q1（公式）：** 写出 SwiGLU 的完整 FFN 公式（含 $W_1, W_2, W_3$ 的维度），说明它与原版 ReLU FFN 在"等参数量"约束下，$d_{ff}$ 应当怎么选。

<details>
<summary>Answer sketch</summary>

要点：

- 完整公式：$\text{FFN}_{\text{SwiGLU}}(x) = W_3 \big( \text{Swish}(W_1 x) \odot (W_2 x) \big)$
- 维度：$W_1, W_2 \in \mathbb{R}^{d_{ff} \times d}$（升维 + 门控），$W_3 \in \mathbb{R}^{d \times d_{ff}}$（降维）
- 参数量：$3 d \cdot d_{ff}$（无 bias）
- 等参数对比：原版 ReLU FFN 参数量 $2 d \cdot 4d = 8 d^2$，要让 SwiGLU 参数等于 $8 d^2$，需 $d_{ff} = \frac{8}{3} d \approx 2.67 d$
- LLaMA 工程公约：再 round 到 256 的倍数，例 LLaMA-2 7B（$d=4096$）取 11008
- 加分：解释 SwiGLU 的"gated"思想——一路 Swish 门 × 一路线性，让网络学会"哪些维度该传、哪些该屏蔽"

</details>

**Q2（实现）：** 用 PyTorch 写一个 LLaMA 风格的 FFN（≤ 8 行核心代码），假设 $d=4096$，给出 $d_{ff}$ 的合理值并解释。

<details>
<summary>Answer sketch</summary>

核心代码（≤ 8 行）：

```python
class LLaMA_FFN(nn.Module):
    def __init__(self, d, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d, d_ff, bias=False)   # gate
        self.w2 = nn.Linear(d, d_ff, bias=False)   # up
        self.w3 = nn.Linear(d_ff, d, bias=False)   # down
    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))
```

$d_{ff}$ 选取：

- 朴素计算：$\frac{8}{3} \times 4096 \approx 10922.67$
- round 到 256 的倍数：11008（与 LLaMA-2 7B 完全一致）
- 关键三点：(1) 三个矩阵全部 `bias=False`；(2) `silu` 等价于 swish；(3) `*` 是逐元素乘，不是矩阵乘
- 加分：提到 vLLM / TGI 实现里会把 w1 和 w2 合并成一个 `gate_up_proj` 矩阵一次 GEMM 算出，再 split，节省 kernel launch

</details>

**Q3（trade-off）：** SwiGLU 比 ReLU FFN 慢 30%+、参数也稍多，为什么 LLaMA / Qwen / Mistral 还是全用 SwiGLU？至少给 2 个理由。

<details>
<summary>Answer sketch</summary>

至少要点到：

- **质量提升明显**：Shazeer 2020 实验表明在等参数预算下，SwiGLU 在 GLUE / SuperGLUE / 翻译任务上比 ReLU/GELU FFN 涨 1-2 个百分点。LLM 规模下这种相对优势仍保持，预训练 loss 也更低
- **预训练阶段算力是 sunk cost**：30% 的 FFN 速度损失分摊到全模型只占总训练时间 ~15%，相对最终模型质量提升完全划算
- **gated 思想本身具备表达力优势**：Swish 门让 FFN 具备"选择性传递"的能力，类似 LSTM 的 gate，在长序列、复杂语义上更不容易丢信息
- **生态/checkpoint 兼容性**：开源大模型几乎全用 SwiGLU，新模型采用 SwiGLU 可以直接复用 LLaMA factory / vLLM / TGI 等工具链的优化 kernel
- **加分**：边界讨论——在端侧极度 latency-sensitive、或 speculative decoding 的 draft 模型上，可以考虑回退到 GELU/ReLU FFN 换速度

</details>

---

## 7. 延伸阅读

- [The Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/) — 经典图解 Transformer 全貌，包括 FFN 在 block 中的位置
- [GLU Variants Improve Transformer (Shazeer 2020 arXiv)](https://arxiv.org/abs/2002.05202) — 5 页的"experimental note"，通读半小时，是 SwiGLU 选型的直接依据
- [LLaMA paper (Touvron et al. 2023)](https://arxiv.org/abs/2302.13971) — §2.1 的几行字解释了 SwiGLU + $\frac{8}{3} d$ 公约的工程取舍
- [PyTorch nn.functional.silu / gelu 源码与文档](https://pytorch.org/docs/stable/generated/torch.nn.functional.silu.html) — 核对激活函数的精确实现，特别是 GELU 的 erf vs tanh 近似
- 推荐继续读本教程的 **4.6 完整 decoder-only 实现**——把 4.1-4.5 的所有积木拼成一个能跑能训的 nanoGPT；以及 **5.4 MoE**——SwiGLU FFN 的下一步，把单 FFN 替换成多 expert + router，详细讲负载均衡与路由
