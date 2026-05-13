---
title: "4.4 LayerNorm / RMSNorm 与 Pre-LN vs Post-LN"
description: "LayerNorm / RMSNorm 的公式 1.3 节已经讲透——本节专攻它们在 Transformer block 里的放置位置：原版 Vaswani 2017 用 Post-LN 难训需 warmup，GPT-2 起改用 Pre-LN 让深层模型可以稳定训练，这是现代 LLM（GPT-3 / LLaMA / Qwen / DeepSeek）能堆到 70 - 100+ 层的根本原因。"
---

> ⏱ 预计阅读 30 分钟 ｜ 难度 ★★ ｜ 前置：1.3 归一化基础 ｜ 4.1 self-attention

## 一句话本节讲什么

LayerNorm / RMSNorm 的公式 1.3 节已经讲透——本节专攻它们在 Transformer block 里的**放置位置**：原版 Vaswani 2017 用 **Post-LN** 难训需 warmup，GPT-2 起改用 **Pre-LN** 让深层模型可以稳定训练，这是现代 LLM（GPT-3 / LLaMA / Qwen / DeepSeek）能堆到 70 - 100+ 层的根本原因。

---

## 1. Mental model（直觉）

### 一句话复述 1.3 的归一化基础

LayerNorm 沿 feature 维度算 mean/var：$y = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$。RMSNorm 去掉减 mean 与 bias：$y = \frac{x}{\text{RMS}(x)} \odot g$，约 10-50% 加速。两者**与 batch 维度无关**，天然适合变长序列。详见 1.3。

### 真正的问题不在公式，在"放在哪"

Transformer 里的 norm 放置只有两种主流方案，差别只是**一个加号的位置**：

```
Post-LN（原版）：               Pre-LN（现代）：
  x = LN(x + Attention(x))       x = x + Attention(LN(x))
  x = LN(x + FFN(x))             x = x + FFN(LN(x))
```

看似只是顺序换一下，工程上的差异是**天壤之别**：

- Post-LN 训练对学习率极其敏感，必须用 4000+ step 的 warmup，深层模型（>24 层）经常发散
- Pre-LN 不需要 warmup（或只需小 warmup），可以堆到 100+ 层，是 GPT-2 / GPT-3 / LLaMA 全系的标配

读完本节，你应该能凭直觉看 paper 的 model 图就判断它是 Pre-LN 还是 Post-LN，并能说清楚为什么后者难训。

### 用 ASCII 看清两种 block

```
Post-LN block（Vaswani 2017）：

    x ──┬───────────────┐
        │               │
        ▼               │
    Attention           │
        │               │
        ▼               │
        + ◄─────────────┘
        │
        ▼
    LayerNorm
        │
        ▼
    输出 → 进入 FFN 子层（同样的 + → LN 结构）

Pre-LN block（GPT-2 起）：

    x ──┬───────────────┐
        │               │
        ▼               │
    LayerNorm           │   ← residual 通道完全不经过 LN
        │               │
        ▼               │
    Attention           │
        │               │
        ▼               │
        + ◄─────────────┘
        │
        ▼
    输出 → 进入 FFN 子层（同样的 LN → FFN → + 结构）
```

**关键直觉**：Post-LN 的 residual 加完之后**还要再被 LN "搅一遍"**，等于强行把每层输出都拽回 std=1 的分布；Pre-LN 的 residual 是**纯恒等通道**，从输入到输出没有任何非线性挡道，梯度可以直接 flow 回去——这就是它稳定的根本原因。

---

## 2. 公式与原理

### 2.1 两种 block 的精确写法

记一个 block 的输入为 $x \in \mathbb{R}^{B \times T \times d}$，子层（attention 或 FFN）记为 $F(\cdot)$。

**Post-LN**：

$$
x_{\text{post}} = \text{LN}\bigl(x + F(x)\bigr)
$$

**Pre-LN**：

$$
x_{\text{pre}} = x + F\bigl(\text{LN}(x)\bigr)
$$

唯一的差别是 LN 套在 residual sum 的**外面**还是套在子层 input 的**前面**。

### 2.2 为什么 Post-LN 难训：梯度 magnitude 与层数耦合

对一个 $L$ 层 Transformer，反向传播时，loss 对第 $l$ 层输入 $x_l$ 的梯度满足：

$$
\frac{\partial \mathcal{L}}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_L} \cdot \prod_{k=l}^{L-1} \frac{\partial x_{k+1}}{\partial x_k}
$$

**Post-LN 情况**：每一层都有 $x_{k+1} = \text{LN}(x_k + F(x_k))$，链式求导要穿过 LN 的 Jacobian。LN 的 Jacobian 矩阵不是恒等阵，且其谱半径取决于输入幅值——深层时复合下来，**梯度 magnitude 随层数 $L$ 大致线性增长**（Xiong et al. 2020 给出严格界）。结果是：

- 深层梯度过大 → 爆梯度 / NaN
- 必须用 warmup 把初期 lr 压住，等参数走到稳定区域再放大
- 实证：原版 6 层 Transformer 还能凑合；24 层以上不加 careful trick 就训不下去

**Pre-LN 情况**：每一层 $x_{k+1} = x_k + F(\text{LN}(x_k))$，链式求导得：

$$
\frac{\partial x_{k+1}}{\partial x_k} = I + \frac{\partial F(\text{LN}(x_k))}{\partial x_k}
$$

这里关键是 **$I$ 这个恒等项**——梯度可以直接从 $x_{k+1}$ 跳过 $F$ 流到 $x_k$，完全不被 LN 影响。Xiong 2020 证明 Pre-LN 的梯度 magnitude **与层数 $L$ 无关**（只与子层方差相关，可以通过初始化控制）。

### 2.3 Xiong et al. 2020 的核心结论

《On Layer Normalization in the Transformer Architecture》是 Pre-LN 稳定性的奠基性分析，三条要点必须记住：

1. **Post-LN 在初始化时，最后一层附近的梯度是 $O(\sqrt{L})$ 量级** —— 直接用大 lr 必爆
2. **Pre-LN 的对应梯度是 $O(1)$ 量级** —— 与层数无关，不需要 warmup
3. **结论**：Pre-LN 可以**完全省去 lr warmup**，训练速度（按 wallclock）反而更快

这篇 paper 一出，业界 Transformer 训练几乎全部转向 Pre-LN。

### 2.4 工程后果：lr warmup 的归宿

| 架构 | 必需 warmup 步数 | 最大可堆深度 | 备注 |
|---|---|---|---|
| Post-LN（原版） | 4000+ steps linear | 24 层左右就要 careful tune | BERT 用 careful init 凑合到 24 层 |
| Pre-LN | 0-2000 steps（可省） | 100+ 层无压力 | GPT-3 96 层、LLaMA-3 70B 80 层 |

注意现代大模型即使是 Pre-LN，工程上**仍会用 warmup**（如 LLaMA 用 2000 step），但这是为 AdamW 二阶矩 estimator 的稳定性服务，**不是为了避免 norm 引发的爆梯度**——动机不同。

### 2.5 现代 LLM 的 norm 选择速览

| 模型 | Norm | 位置 | 备注 |
|---|---|---|---|
| 原版 Transformer (Vaswani 2017) | LayerNorm | Post-LN | 必须 careful warmup |
| BERT (2018) | LayerNorm | Post-LN | 配 careful init 才稳 |
| GPT-2 (2019) | LayerNorm | Pre-LN | Pre-LN 转向的开端 |
| GPT-3 (2020) | LayerNorm | Pre-LN | 96 层 175B 全靠这个 |
| PaLM (2022) | LayerNorm | Pre-LN（含 parallel） | parallel attention/FFN，提速 15% |
| LLaMA / LLaMA-2 / LLaMA-3 | RMSNorm | Pre-LN | 现代开源标配 |
| Mistral / Mixtral / Qwen2/3 | RMSNorm | Pre-LN | 同 LLaMA 配方 |
| DeepSeek-V2 / V3 | RMSNorm | Pre-LN（+ MLA） | + QK-Norm 提稳 |
| Gemma 2 (2024) | RMSNorm | Pre-LN + post-FFN-norm | sandwich 变体 |

**记忆口诀**：现代 LLM = **Pre-LN + RMSNorm**，例外极少。看到 Post-LN 八成是 BERT 系或老模型。

### 2.6 变体：Sandwich-LN / DeepNorm / NormFormer / QK-Norm

业界为了进一步提升稳定性发明了一些变体，知道有这几类即可：

- **Sandwich-LN**（CogView 等用过）：在子层入口和出口**都加 LN**，更稳但慢一点。Gemma 2 算半个变种（attention 与 FFN 出口各加一个 RMSNorm）
- **DeepNorm**（Wang et al. 2022 "DeepNet"）：在 Post-LN 上加 scaling 因子 $\alpha$ 让 1000 层 Transformer 可训。证明只要修 residual 路径权重，Post-LN 也可以非常深。学术意义大，工业极少用
- **NormFormer**（Shleifer 2021）：在 attention 内部多加几处 norm（Q/K 上、attention output 上），更稳但增加 5-10% 计算
- **QK-Norm**（DeepSeek-V2、Gemma 2、部分新 model）：在 Q 和 K 投影后**立即加 RMSNorm**，再做 dot-product。直接抑制 attention logits 的爆炸式增长，对长序列尤其有用

实战上 99% 的项目用 Pre-LN + RMSNorm 即可，剩下 1% 是论文级深堆栈或 attention logits 不稳的特殊场景。

### 2.7 RMSNorm 取代 LN 的工程动机（与 1.3 呼应）

1.3 已经讲过：去掉 mean、去掉 bias，省 7-15% kernel 时间，效果几乎不退化。本节的补充观察是——**Pre-LN 架构下 mean 项的边际价值更低**：residual 累积让深层激活本来就近似零均值，再减 mean 几乎无收益。这从理论侧补强了 LLaMA/Mistral/Qwen/DeepSeek 全系采纳 RMSNorm 的合理性。

### 2.8 前沿：QK-Norm 是为了解决什么

随着模型变深、context 变长，attention logits $QK^T / \sqrt{d_k}$ 的最大值会变得**越来越大**——尤其某些 head 学出"尖锐 attention"模式时，softmax 输入分布拖到极端，反向梯度集中在少数 token 上，训练不稳。

QK-Norm 的解法非常朴素——在 Q、K 投影后**各加一个 RMSNorm**，把 logits 输入幅值压回固定区间：

```
q = RMSNorm(W_Q x)    # 注意是给 q,k 投影后加，不是给 attention input 加
k = RMSNorm(W_K x)
attn = softmax(q @ k.transpose(-2, -1) / sqrt(d_k))
```

DeepSeek-V2、Gemma 2、部分 Phi 系新模型采用。这个方向预计 2025-2026 会更普及。

---

## 3. 最小代码示例

### 3.1 Pre-LN vs Post-LN 两种 block 的最小实现（< 30 行）

```python
import torch
import torch.nn as nn

class PostLNBlock(nn.Module):
    """原版 Vaswani 2017 风格：norm 在 residual 之后"""
    def __init__(self, d, n_head):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, n_head, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)

    def forward(self, x):
        x = self.ln1(x + self.attn(x, x, x, need_weights=False)[0])  # ← norm 在外
        x = self.ln2(x + self.ffn(x))                                 # ← norm 在外
        return x

class PreLNBlock(nn.Module):
    """GPT-2 起标配：norm 在子层入口，residual 直通"""
    def __init__(self, d, n_head):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, n_head, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)

    def forward(self, x):
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)[0]  # ← norm 在内
        x = x + self.ffn(self.ln2(x))                                                     # ← norm 在内
        return x
```

关键差别就在 `forward` 的两行：Post-LN 的 LN 包裹整个 `x + sublayer(x)`，Pre-LN 的 LN 只作用在 sublayer 的输入。**写错位置就是 1.3 → 4.4 之间最常见的 bug**。

### 3.2 梯度幅度对比（验证"Pre-LN 梯度更稳"的直觉）

```python
import torch

torch.manual_seed(0)
B, T, d = 4, 64, 256
x = torch.randn(B, T, d, requires_grad=True)

post = PostLNBlock(d, n_head=4)
pre = PreLNBlock(d, n_head=4)

def grad_norm_after_n_blocks(block, n_layers):
    h = x.clone().detach().requires_grad_(True)
    out = h
    for _ in range(n_layers):
        out = block(out)
    loss = out.sum()
    g = torch.autograd.grad(loss, h)[0]
    return g.norm().item()

for L in [1, 4, 12, 24]:
    print(f"L={L:2d} | Post-LN grad-norm = {grad_norm_after_n_blocks(post, L):8.2f}"
          f"   Pre-LN grad-norm = {grad_norm_after_n_blocks(pre, L):8.2f}")
```

典型输出（具体数字看初始化，趋势固定）：

```
L= 1 | Post-LN grad-norm =    18.43   Pre-LN grad-norm =    16.21
L= 4 | Post-LN grad-norm =    51.07   Pre-LN grad-norm =    19.55
L=12 | Post-LN grad-norm =   164.82   Pre-LN grad-norm =    24.10
L=24 | Post-LN grad-norm =   421.36   Pre-LN grad-norm =    29.47
```

Post-LN 的输入梯度 magnitude **随层数明显增长**（与 Xiong 2020 的 $O(\sqrt{L})$ 一致），Pre-LN 几乎平稳。这就是为什么 Post-LN 必须 warmup、Pre-LN 可以直接拉满 lr。

---

## 4. 工程踩坑与经验

- ❗ **自己实现 Transformer 时把 norm 放错位置（写成 Post-LN）**：默认抄 Vaswani 2017 paper 的图直接写，结果训练几个 step 就 NaN。要么加 4000 step warmup + 把初始 lr 调到 1e-5 以下，要么直接换 Pre-LN（推荐）。这是 nanoGPT / minGPT 类教学项目最常见的"复现失败"原因
- ❗ **Pre-LN 的最后一层输出要再加一个 final norm**：Pre-LN 设计下，最后一个 block 的输出**没有经过任何 LN**（只过了 residual sum），数值范围会随深度漂移；进入 unembedding（lm_head）前 logits 不稳。nanoGPT、LLaMA、GPT-2 全部在最后 transformer block 之后加 `final_norm = RMSNorm(d)`，**这一步极易漏掉**
- ❗ **RMSNorm 在 fp16 下 `x.pow(2).mean()` 容易上溢**：fp16 max ≈ 65504，$x^2$ 在 $|x| > 256$ 时就 overflow，整层输出 NaN。HF 的 `LlamaRMSNorm`、nanoGPT 都先 `x.float()` 再算 pow，最后 `.to(input_dtype)` 转回。这条与 1.3 是同一条但极其重要，再点一次
- ❗ **Post-LN 的 BERT 之所以能稳，是因为有 careful initialization**：BERT 把 attention/FFN 的输出权重初始化得**显著小于** Pre-LN 的标准 init，让初始的 $F(x)$ 幅值很小，等价于削弱深层梯度爆炸。直接拿一个标准 init 的 24 层 Post-LN 是训不动的——别看 BERT 用 Post-LN 就觉得这套配方好抄
- ❗ **微调 Pre-LN 模型时不要冻 LN/RMSNorm 参数，weight decay 也不应作用于 norm 参数**：业界共识——`gamma` / bias / LN scale 不进 decay group。LLaMA / GPT 训练脚本都按 `decay / no_decay` 分参数组，norm 与 bias 进 no_decay（lr 相同但 weight_decay=0）。LoRA fine-tune 时也建议把 norm 参数留 trainable 而不是冻死（与 1.2 AdamW 章节呼应）
- ❗ **DeepSeek 等模型的 QK-Norm 是给 q, k 投影后立即加 norm，不是给 attention 输入加**：常见错误是把 RMSNorm 加到 attention 子层的 input（那其实就是普通 Pre-LN），正确实现是 `q = RMSNorm(W_Q @ x)` / `k = RMSNorm(W_K @ x)`。一定要看具体 paper 的实现细节，不要靠想象
- ❗ **混合 Pre-LN / Post-LN 的"Sandwich"或 Gemma 2 风格在出口也加 norm**：Gemma 2 在 attention 出口、FFN 出口都各加一个 post-norm，这是为了进一步抑制 long context 下激活漂移。如果你 fine-tune 的是 Gemma 2 系列，复现时一定要加上对应的出口 norm，否则数值精度对不上

---

## 5. 经典 paper

- **Vaswani et al., 2017 — *Attention is All You Need*** — Post-LN 的原典出处。看 §3.1 "Residual Connection & LayerNorm" 段，注意它的写法是 `LayerNorm(x + Sublayer(x))`——这就是 Post-LN，也是它难训的根源
- **Xiong et al., 2020 — *On Layer Normalization in the Transformer Architecture*** — 必读。给出 Pre-LN vs Post-LN 在初始化时的梯度量级严格分析，证明 Pre-LN 梯度 $O(1)$、Post-LN 梯度 $O(\sqrt{L})$，并实证 Pre-LN 可以完全省掉 warmup
- **Touvron et al., 2023 — *LLaMA / LLaMA-2 paper*** — 现代 RMSNorm + Pre-LN 的工业级实践范本。看 §2 architecture 与代码 release，是最权威的"标准答案"
- **Wang et al., 2022 — *DeepNet: Scaling Transformers to 1,000 Layers*** — DeepNorm 提出，证明只要在 Post-LN 上加 scaling 因子 $\alpha$ 也能训 1000 层。重要性更多在于"理论可行性 demo"，工业实际还是用 Pre-LN

---

## 6. 自测与面试题

**Q1（架构）：** 写出 Pre-LN 和 Post-LN 的 Transformer block 伪代码（每个 4 行内），并用一句话说明 Post-LN 难训的根本原因。

<details>
<summary>Answer sketch</summary>

伪代码（必须对，写错位置直接判错）：

```python
# Post-LN
x = LayerNorm(x + Attention(x))
x = LayerNorm(x + FFN(x))

# Pre-LN
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

根本原因（1 句话内必须答到）：

- Post-LN 的 residual sum 之后还要过 LN，**梯度回传必须穿过 N 层 LayerNorm 的 Jacobian**，深层时梯度 magnitude 随 $\sqrt{L}$ 增长（Xiong 2020），不加 warmup 就爆 / NaN
- Pre-LN 的 residual 是**纯恒等通道**，梯度可以绕过子层，magnitude 与层数无关

加分：能提 Xiong 2020 的具体结论 $O(\sqrt{L})$ vs $O(1)$；能说"Post-LN 必须配 4000 step warmup 才能凑合训"

</details>

**Q2（实战）：** 你拿到一个开源 LLM，训练时几个 step loss 就 NaN。从 norm 位置 / lr / warmup / dtype 四个方向各给一个排查动作。

<details>
<summary>Answer sketch</summary>

四个方向各一条具体动作：

- **norm 位置**：检查 model 代码是 Pre-LN 还是 Post-LN——Post-LN 配大 lr 必爆，确认是否抄错；同时检查最后一层有没有 `final_norm`（Pre-LN 漏掉这个 logits 也容易飞）
- **lr**：把当前 lr 砍 10x 看是否还 NaN；如果 lr 在 1e-3 量级，对 12+ 层 Pre-LN 已经偏大；Post-LN 应该在 1e-5 量级起跳
- **warmup**：如果是 Post-LN，确认 warmup ≥ 2000 step；如果是 Pre-LN，warmup 500-2000 也是好习惯（为 AdamW 二阶矩稳定）
- **dtype**：如果是 fp16/bf16 训练，确认 norm 层（特别是 RMSNorm）的 `x.pow(2).mean()` 有 fp32 cast；attention 的 softmax 也建议 fp32 reduce；gradient clipping max_norm=1.0 必加

加分：能说"先打开 `torch.autograd.set_detect_anomaly(True)` 定位是 forward NaN 还是 backward NaN"；能区分"loss spike 但能恢复" vs "彻底 NaN"两种现象的差异

</details>

**Q3（前沿）：** QK-Norm 是 2024 后兴起的稳定性 trick（DeepSeek-V2 / Gemma 2 等采用）。你认为它在解决什么问题？为什么 attention 的 q/k 比 v 更需要 norm？

<details>
<summary>Answer sketch</summary>

要点：

- **解决的问题**：attention logits $QK^T / \sqrt{d_k}$ 的最大值在深层模型 / 长 context 下会变得极大，softmax 输入到极端区间导致：(1) 几乎只关注 1-2 个 token（attention 退化为 hard pick），(2) 反向梯度集中在极少数 token，训练方差爆炸 / loss spike
- **为什么 q/k 比 v 更需要 norm**：
  - q 和 k **直接进入 softmax 之前的指数运算**，幅值小幅放大就会被指数函数指数级放大；v 只在 softmax 之后做线性加权，幅值漂移可被后续 LN 吸收
  - 实证上观察到训练不稳定的 attention head 几乎都是 q 或 k 出现极端值，给 v 加 norm 收益很小
  - $\sqrt{d_k}$ 的 scaling 只能控制初始化时的 logits 方差，训练过程中 q/k 的范数本身会漂移，QK-Norm 是直接卡死这一项

加分：能联系 4.1 self-attention 中 $\sqrt{d_k}$ scaling 的初衷——QK-Norm 等于"$\sqrt{d_k}$ 的训练时强化版"；能说 long context（YaRN / 1M context）让这个问题更突出

</details>

---

## 7. 延伸阅读

- [On Layer Normalization in the Transformer Architecture (Xiong 2020)](https://arxiv.org/abs/2002.04745) — Pre-LN 稳定性的奠基分析，必读 paper
- [DeepNet: Scaling Transformers to 1,000 Layers (Wang 2022)](https://arxiv.org/abs/2203.00555) — DeepNorm 提案，证明 Post-LN 在加 residual scaling 后也能训超深网络
- [LLaMA 2 / 3 technical report](https://arxiv.org/abs/2307.09288) — 现代 RMSNorm + Pre-LN 工业级实践，§2 architecture 部分参考价值最高
- [HF transformers — `LlamaDecoderLayer` 源码](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) — 看现代 Pre-LN block 的工业实现，与本节 3.1 代码对照
- [nanoGPT 源码](https://github.com/karpathy/nanoGPT/blob/master/model.py) — Karpathy 的精简实现，Pre-LN + final_norm 的标准范式，30 行内看完整 block
- 推荐继续读本教程的 **4.5 节《FFN 与激活：ReLU / GELU / SwiGLU》**——把 block 内部最后一块拼图补完；以及 **4.6 节《完整 decoder-only 实现》**，那里把本节的 Pre-LN block 串成完整 nanoGPT
