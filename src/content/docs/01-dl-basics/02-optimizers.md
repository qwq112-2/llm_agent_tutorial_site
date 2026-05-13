---
title: "1.2 优化器：SGD / Momentum / Adam / AdamW + LR schedule"
description: "从最朴素的 $\\\\theta \\\\leftarrow \\\\theta - \\\\eta g$ 推到 LLM 训练标配 AdamW + warmup + cosine decay——讲清每一步为什么这么改、改完解决了什么、引入了什么新代价；这是算法岗面试 90% 会问到的\"基础但易翻车\"高频题。"
---

> ⏱ 预计阅读 35 分钟 ｜ 难度 ★★ ｜ 前置：1.1 神经网络与反向传播

## 一句话本节讲什么

从最朴素的 $\theta \leftarrow \theta - \eta g$ 推到 LLM 训练标配 **AdamW + warmup + cosine decay**——讲清每一步为什么这么改、改完解决了什么、引入了什么新代价；这是算法岗面试 90% 会问到的"基础但易翻车"高频题。

---

## 1. Mental model（直觉）

把训练 loss landscape 想象成一片高维丘陵地形，参数 $\theta$ 是一个小球，优化器决定**它每一步朝哪走、走多远**。

不同优化器是不同的"驾驶策略"：

```
SGD          : 每一步只看脚下那块地的最陡下坡方向，刚正不阿但容易在峡谷里左右乱撞
Momentum     : 在 SGD 上加一个"惯性"，下坡时越滚越快，对小坑视而不见
Adam         : 给每个参数单独配一个"自适应油门"——梯度大的方向自动减速、稀疏的方向加速
AdamW        : Adam 上把 weight decay 拆出来单独算，修一个隐藏多年的 bug
```

四代优化器解决了三个具体痛点：

1. **方向震荡**（SGD → Momentum）：高维 loss 表面常常一个方向窄、另一个方向宽（"病态曲率 / ill-conditioning"），SGD 在窄方向反复横跳，Momentum 把多步梯度加权平均，**让横跳互相抵消、纵向累加**。
2. **不同参数尺度差异大**（Momentum → Adam）：Embedding 层的稀疏更新、LayerNorm 的 scale 参数、attention 矩阵——它们的梯度幅度可能差几个量级。Adam 给每个参数维护一个**梯度二阶矩 EMA**，相当于"按梯度噪声反向缩放学习率"。
3. **Adam 把 weight decay 也缩放了**（Adam → AdamW）：Adam 的 weight decay 写在 loss 里就会被 $\sqrt{\hat v_t}$ 一起除掉，导致**梯度大的参数 decay 反而少**，与"weight decay 是无差别正则"的初衷相反。AdamW 把 decay 直接写到参数更新式里，绕过自适应缩放。

LR schedule 解决的是**另一维度的问题**：训练初期参数随机，梯度 / 二阶矩估计都不准，必须 warmup（小步试探）；训练后期接近最优解，要 decay（小步精细）。LLM 上几乎是"必备组合"——没 warmup 大概率前几百步 loss 飞，没 decay 后期收敛慢。

> 一句话总结全节：**LLM 训练 = AdamW(β1=0.9, β2=0.95, wd=0.1) + linear warmup + cosine decay**，剩下都是细节。理解为什么是这套配置，就理解了 90% 的预训练超参。

---

## 2. 公式与原理

记号约定：$\theta_t \in \mathbb{R}^d$ 是第 $t$ 步的参数向量，$g_t = \nabla_\theta \mathcal{L}(\theta_{t-1})$ 是这一步的梯度，$\eta$ 是学习率（标量）。

### 2.1 SGD：朴素梯度下降

最简单的更新式：

$$
\theta_t = \theta_{t-1} - \eta \cdot g_t
$$

直觉就是"沿着 loss 下降最快的方向走一步"。问题在于 $g_t$ 是从一个 mini-batch 估计来的、噪声很大；而且 loss 表面常常是**长条形山谷**（窄方向曲率大、宽方向曲率小），SGD 在窄方向上 $|g_t|$ 大、被反复推回中央，在宽方向上 $|g_t|$ 小、走得很慢——表现为"在山谷里 Z 字震荡"。

### 2.2 Momentum：给 SGD 加惯性

把过去的梯度做指数加权平均，作为"速度" $v_t$：

$$
v_t = \mu v_{t-1} + g_t, \qquad \theta_t = \theta_{t-1} - \eta \cdot v_t
$$

其中 $\mu \in [0, 1)$ 是动量系数，常取 0.9。展开 $v_t$：

$$
v_t = g_t + \mu g_{t-1} + \mu^2 g_{t-2} + \cdots
$$

可以看到 $v_t$ 是**所有历史梯度的几何加权和**。在 Z 字震荡的窄方向上，相邻几步 $g_t$ 反号，加起来互相抵消；在持续下坡的宽方向上同号累加、越滚越快。这是 momentum "**抑制震荡 + 加速收敛**" 的物理来源。

> **Nesterov 变体**：把 $g_t$ 算在 $\theta_{t-1} - \eta \mu v_{t-1}$ 这个"前瞻位置"上而不是当前位置，相当于"先按惯性走一步、再看那里的坡度"，二阶项理论收敛更快但 LLM 上几乎不用——经验表明对 transformer 增益微弱、调试难度上升。

### 2.3 Adam：自适应学习率

Momentum 用了一阶矩 $m_t$（梯度的 EMA）；Adam 同时维护**二阶矩** $v_t$（梯度平方的 EMA）：

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
$$

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \quad (\text{逐元素平方})
$$

直接用 $m_t / \sqrt{v_t}$ 更新有个问题：初始化 $m_0 = v_0 = 0$，前几步 $m_t, v_t$ 都被严重低估。Kingma 的修正是**偏差修正**（bias correction）：

$$
\hat m_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat v_t = \frac{v_t}{1 - \beta_2^t}
$$

直觉上：$\beta_1^t \to 0$ 之后修正项 $\to 1$，几乎无影响；但在 $t = 1$ 时 $\hat m_1 = g_1$、$\hat v_1 = g_1^2$，相当于"第一步等价于一次正常的 RMSProp 步长"。

最终更新式：

$$
\theta_t = \theta_{t-1} - \eta \cdot \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}
$$

$\epsilon$（典型 $10^{-8}$）是数值稳定项，防止 $\sqrt{\hat v_t} = 0$ 时除零，并且**限制更新幅度上界约为 $\eta / \epsilon$**——某种意义上是隐式的梯度裁剪。

**为什么 Adam 在 LLM 上是默认**：transformer 不同位置参数的梯度幅度差异巨大（embedding 稀疏、LayerNorm scale 接近 1、QKV 矩阵更新密集），SGD/Momentum 只有一个全局 lr，调起来必然顾此失彼；Adam 给每个参数维一个有效 lr $\eta / (\sqrt{\hat v_{t,i}} + \epsilon)$，**让大梯度方向自动收敛、小梯度方向不被压制**。代价是显存：$m, v$ 都要 fp32 保存，每个参数 2 倍额外开销（详见第 4 章踩坑）。

### 2.4 AdamW：解耦 weight decay

L2 正则的标准做法是在 loss 里加 $\frac{\lambda}{2} \|\theta\|^2$，对 $\theta_i$ 的梯度变成 $g_i + \lambda \theta_i$。在 SGD 里这恰好等价于"每步乘 $(1 - \eta \lambda)$"，所以 L2 与 weight decay 在 SGD 下是**同一个东西**。

但在 Adam 里，加进 loss 的 $\lambda \theta$ 项会跟 $g_t$ 一起过 EMA、一起被 $\sqrt{\hat v_t}$ 缩放：

$$
\theta_t \leftarrow \theta_{t-1} - \eta \cdot \frac{\hat m_t \;+\; \lambda \theta_{t-1} \cdot (\text{被 EMA 平滑})}{\sqrt{\hat v_t} + \epsilon}
$$

后果是**梯度大的参数（$\sqrt{\hat v_t}$ 大）被 decay 得少，梯度小的参数被 decay 得多**——和"无差别正则"的初衷完全相反。Loshchilov & Hutter 2017 把它修了：weight decay 不进 loss、不进 EMA，**直接乘到参数上**：

$$
\theta_t = \theta_{t-1} - \eta \cdot \left( \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon} + \lambda \theta_{t-1} \right)
$$

这就是 **AdamW**——名字里的 W 就是 "decoupled Weight decay"。LLM 训练几乎全部用 AdamW，wd 一般取 0.1（远大于 CV 的 0.01，原因见第 4 章）。

### 2.5 Lion：sign-based 新变体

Chen et al. 2023 用符号搜索发现的 optimizer：

$$
u_t = \text{sign}(\beta_1 m_{t-1} + (1 - \beta_1) g_t), \quad \theta_t = \theta_{t-1} - \eta(u_t + \lambda \theta_{t-1})
$$

$$
m_t = \beta_2 m_{t-1} + (1 - \beta_2) g_t
$$

只用 $m$、不用 $v$，**显存比 AdamW 省一半**。Trade-off：lr 要比 AdamW 调小 3-10 倍（因为 sign 出来的更新模长固定为 $\sqrt{d}$），对 batch size 敏感、对 LayerNorm 等小参数效果不如 AdamW 稳。**面试角度**：知道它存在、知道它"省一半 optimizer state"、知道 Google PaLM-2 / 部分开源模型用它就够了，工业界主流仍是 AdamW。

### 2.6 LR schedule

**Warmup**：从 0 或一个很小的 lr 线性涨到 $\eta_{\max}$，典型长度 500-2000 step（占总训练 0.1%-1%）：

$$
\eta_t = \eta_{\max} \cdot \frac{t}{T_\text{warmup}}, \quad t \le T_\text{warmup}
$$

**Cosine decay**：从 $\eta_{\max}$ 余弦下降到 $\eta_{\min}$（通常 $\eta_{\min} = 0.1 \eta_{\max}$ 或 0）：

$$
\eta_t = \eta_{\min} + \tfrac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\pi \cdot \frac{t - T_\text{warmup}}{T - T_\text{warmup}}\right)\right)
$$

cosine 的好处是**前期 decay 慢（保持探索）、后期 decay 快（精细收敛）**，loss curve 通常比 linear decay 更平滑。

简短对比：

| Schedule | 形状 | 适用 |
|---|---|---|
| **constant** | 一条直线 | 短期 finetune / 超参搜索 baseline |
| **linear decay** | 线性下降到 0 | RoBERTa / 某些 SFT 设置 |
| **cosine decay** | 余弦曲线 | LLM 预训练标配（GPT / LLaMA / Qwen）|
| **one-cycle**（Smith 2018） | warmup + cosine 一个周期 + 末段更小 | CV 短训练 |
| **WSD**（warmup-stable-decay）| warmup + 长 stable + 短 decay | DeepSeek-V3 / MiniCPM 用，可中途切 ckpt 继续训 |

---

## 3. 最小代码示例

### 3.1 手写 Adam（25 行内）

```python
import torch

class MyAdam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.params = list(params)
        self.lr, self.b1, self.b2, self.eps = lr, betas[0], betas[1], eps
        self.t = 0
        self.m = [torch.zeros_like(p) for p in self.params]   # 一阶矩
        self.v = [torch.zeros_like(p) for p in self.params]   # 二阶矩

    @torch.no_grad()
    def step(self):
        self.t += 1
        bc1 = 1 - self.b1 ** self.t                           # 偏差修正分母
        bc2 = 1 - self.b2 ** self.t
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None: continue
            g = p.grad
            m.mul_(self.b1).add_(g, alpha=1 - self.b1)        # m_t = β1·m + (1-β1)·g
            v.mul_(self.b2).addcmul_(g, g, value=1 - self.b2) # v_t = β2·v + (1-β2)·g²
            m_hat = m / bc1
            v_hat = v / bc2
            p.addcdiv_(m_hat, v_hat.sqrt().add_(self.eps), value=-self.lr)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None: p.grad.zero_()
```

关键点：`addcmul_(g, g, value=1-β2)` 是 in-place 的 $v + (1-\beta_2) g^2$；`addcdiv_(m_hat, denom, value=-lr)` 是 in-place 的 $\theta - \eta \hat m / \text{denom}$。这两个 fused 操作能避免分配中间 tensor，是 PyTorch 优化器源码常见模式。

**对应的 PyTorch 调用**（LLaMA-style 推荐配置）：

```python
import torch
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,                    # peak lr，7B 模型典型范围 1e-4 ~ 6e-4
    betas=(0.9, 0.95),          # β2=0.95 而不是 0.999，原因见第 4 章
    eps=1e-8,
    weight_decay=0.1,           # LLM 标配 0.1，远大于 CV 的 0.01
    fused=True,                 # CUDA 上 fused kernel，速度快 30%+
)
```

### 3.2 Linear warmup + cosine decay（LambdaLR）

```python
import math
from torch.optim.lr_scheduler import LambdaLR

def get_warmup_cosine_schedule(optimizer, num_warmup_steps, num_total_steps, min_lr_ratio=0.1):
    def lr_lambda(step):
        # Phase 1: linear warmup 从 0 涨到 1.0（即涨到 optimizer 的 base lr）
        if step < num_warmup_steps:
            return step / max(1, num_warmup_steps)
        # Phase 2: cosine decay 从 1.0 到 min_lr_ratio
        progress = (step - num_warmup_steps) / max(1, num_total_steps - num_warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    return LambdaLR(optimizer, lr_lambda)

# 典型用法：100k step 训练，前 2k step warmup
scheduler = get_warmup_cosine_schedule(optimizer, num_warmup_steps=2000, num_total_steps=100_000)

# 训练循环里每 step 调一次
for step in range(100_000):
    loss = model(batch).loss; loss.backward()
    optimizer.step(); scheduler.step(); optimizer.zero_grad()
```

这个实现等价于 HuggingFace 的 `get_cosine_schedule_with_warmup`。`LambdaLR` 接受一个返回**乘子**的函数，PyTorch 会自动用 `optimizer.param_groups[i]['lr'] = base_lr * lr_lambda(step)` 设置。

---

## 4. 工程踩坑与经验

- ❗ **Adam 不加 warmup 大概率前几百步 loss 飞**——尤其 bf16 下。原因有两个：（1）随机初始化的参数在第一步梯度方差极大，$\hat v_t$ 估计严重偏低、 $\hat m_t / \sqrt{\hat v_t}$ 数值不稳；（2）bf16 mantissa 只有 7 位，初期 loss 范围大、量化误差被放大。LLM 训练里 warmup 不是可选项是必备项，2k step linear warmup 是 7B 量级的安全默认。
- ❗ **AdamW 的 `weight_decay` 在 LLM 上常用 0.1，远大于 CV 的 0.01**。直觉解释：（1）LLM 参数量大、容量过剩、需要更强正则；（2）LLM 训练 epoch 通常 < 1（数据 > 参数），过拟合方式不同于 CV 的"反复看同一张图"；（3）大 wd 让权重朝 0 收缩、隐式控制 effective rank、提升泛化。LLaMA / Qwen / DeepSeek 都用 0.1，新手一般不要乱改。
- ❗ **β2 在 LLM 上常调到 0.95（LLaMA），而不是默认 0.999**。0.999 对应"过去约 1000 步的二阶矩 EMA"，长 EMA 在 LLM 上反而**滞后**——梯度噪声大、loss 突变多，二阶矩跟不上参数实际状态变化，引发训练不稳。0.95 对应约 20 步窗口，更敏感、能更快响应梯度尺度变化。同理 β1 在 large batch 训练里有时降到 0.9 以下（如 0.85）。
- ❗ **Adam 的显存开销：每个参数额外 2 倍**。fp32 保存 $m, v$，加上参数本身和梯度，共 $4 \times \text{numel}$ bytes（参数）+ $4 \times \text{numel}$（梯度）+ $4 \times \text{numel}$（m）+ $4 \times \text{numel}$（v） = **每参数 16 bytes**。70B 模型仅 optimizer state 就要 $70\text{B} \times 8 = 560\text{GB}$，加参数 + 梯度共 1120 GB——远超单卡 80GB H100。这就是 Module 7 ZeRO 的核心动机：把 m/v 切到不同 rank 上、需要时通信。
- ❗ **resume 训练时 optimizer state 必须一起 ckpt**，否则 m/v 重置等于把训了 N 步的模型当成"刚初始化的 + 当前 weights"——前几百 step 等同热启动重新调 v_t，loss 会飙一下、训练曲线断崖。HuggingFace `Trainer` 默认会保存 `optimizer.pt`，自定义训练循环要记得 `torch.save({'model': ..., 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'step': step}, path)`。
- ❗ **LR schedule 的总 step 数要包含 grad accumulation**。如果 batch_size=64、micro_batch=8、grad_accum=8，那么 100k optimizer step 对应 100k × 8 = 800k forward-backward。`scheduler.step()` 应在每次 `optimizer.step()` 后调用、而不是每次 forward 后——HF Trainer 处理好了，自定义循环最常踩这个坑。
- ❗ **Embedding / LayerNorm / bias 不应该 decay**——这是个隐性约定。AdamW 默认对所有参数 decay，但 LayerNorm 的 scale 应该接近 1、bias 没有"过拟合"概念、embedding decay 会损害 token 表达。实操上要把模型参数分两组传给 optimizer：

```python
decay_params, nodecay_params = [], []
for n, p in model.named_parameters():
    if p.dim() < 2 or 'norm' in n.lower() or 'bias' in n: nodecay_params.append(p)
    else: decay_params.append(p)
optimizer = torch.optim.AdamW([
    {'params': decay_params,   'weight_decay': 0.1},
    {'params': nodecay_params, 'weight_decay': 0.0},
], lr=3e-4, betas=(0.9, 0.95))
```

---

## 5. 经典 paper

- **Kingma & Ba, 2014 — Adam: A Method for Stochastic Optimization** — Adam 的原典。读 §2 推导一阶矩 / 二阶矩 / 偏差修正三件套（本节公式与该 §2.1 完全一致），读 §6.4 看作者建议的默认 $\beta_1=0.9, \beta_2=0.999, \epsilon=10^{-8}$ 来自哪些实验——**默认值是 CV 时代的 legacy，不是真理**。
- **Loshchilov & Hutter, 2017 — Decoupled Weight Decay Regularization** — AdamW 的提出。读 §2 看为什么 L2 正则与 weight decay 在 Adam 里不等价（这是面试高频题），读 §3 看 ImageNet 上的 ablation——为后续 LLM 训练为什么用 AdamW 而不是 Adam 提供了实证根基。
- **Chen et al., 2023 — Symbolic Discovery of Optimization Algorithms** — Lion 的提出。不需要细看，只需知道：用进化搜索 + program search 在搜索空间里挖到的；只用一阶矩、用 sign 函数；显存比 AdamW 省一半；lr 要小 3-10 倍。**面试中"听说过 Lion 吗"答出这几点就过关**。

---

## 6. 自测与面试题

**Q1（公式题）**：写出 Adam 的完整更新式（含偏差修正），并解释 $\epsilon$ 的两个作用。

<details>
<summary>Answer sketch</summary>

完整更新式分四步：

1. $m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$
2. $v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$
3. $\hat m_t = m_t / (1 - \beta_1^t)$，$\hat v_t = v_t / (1 - \beta_2^t)$
4. $\theta_t = \theta_{t-1} - \eta \cdot \hat m_t / (\sqrt{\hat v_t} + \epsilon)$

$\epsilon$ 的作用：

- **数值稳定**：防止 $\sqrt{\hat v_t} \to 0$ 时除零（特别是稀疏梯度的早期 step）
- **隐式更新幅度上界**：当 $\hat v_t \to 0$ 时分母为 $\epsilon$，单步更新最大约为 $\eta / \epsilon \cdot \hat m_t$，相当于不会任意变大，对训练初期稳定性有帮助
- 加分点：$\epsilon$ 的位置有两种实现，"在开方外"（PyTorch 默认 $\sqrt{v} + \epsilon$）和"在开方内"（$\sqrt{v + \epsilon}$，TF 早期），前者主流

</details>

**Q2（trade-off 题）**：为什么 LLM 训练几乎必然要 warmup？给至少 2 个理由。

<details>
<summary>Answer sketch</summary>

至少要点到：

- **二阶矩估计不稳**：Adam 在 $t$ 较小时 $\hat v_t$ 接近 $g_1^2$，从单 sample 估的二阶矩方差极大，直接用 peak lr 会让 $\hat m_t / \sqrt{\hat v_t}$ 数值不稳——warmup 给 $\hat v_t$ 时间在多步 EMA 下收敛到合理估计
- **初始参数随机**：随机初始化的参数与目标分布相距很远，第一步梯度量级大、方向噪声大；小 lr 让前几百步只做"温和的方向修正"，等参数到了 loss 表面合理区域再加速
- **bf16 / fp16 数值边界**：训练初期 loss 范围大（10+），bf16 mantissa 7 位，warmup 让 loss 先快速下降到 1-3 区间再加速、减少量化误差爆炸的风险
- **大 batch / 高并行下尤其严重**：batch 越大梯度估计越准但 effective lr 也按 sqrt(B) 涨，没 warmup 直接发散
- 反例：constant lr 在 finetune（lr 1e-5、几个 epoch）上有时不需要 warmup，因为模型已经在合理区域

</details>

**Q3（实战题）**：你在调一个 7B 模型，loss 在第 200 step 后突然 NaN，你会按什么顺序排查？lr / β / weight decay / warmup / 数据 / dtype。

<details>
<summary>Answer sketch</summary>

排查顺序遵循"先排能 1 分钟验证的、最常见的"：

1. **dtype + loss scale**：第一嫌疑。bf16 训练通常稳，fp16 训练必须配 GradScaler；先看是 fp16 没开 loss scale，还是 bf16 下某层（如 softmax / log_softmax）数值溢出。改成 fp32 跑 100 step 看是否复现，能立刻定位
2. **数据**：loss NaN 80% 是某条样本里有奇异 token / 全 padding / label 全 -100（loss = 0/0）。把 step 200 那个 batch 单独跑出来 print 一下 input_ids / labels / loss，看是不是某条样本独自把 loss 拉飞
3. **lr 太大 / warmup 不够**：peak lr 6e-4 对 7B 已偏激进，如果只 warmup 200 step 就到 peak，可能在 step 200 刚到 peak 时炸。把 warmup 调到 2000 step、peak lr 砍半再试
4. **β2 设置**：如果 β2 仍是默认 0.999，在 LLM 上长 EMA 会让 $\hat v_t$ 跟不上梯度突变，二阶矩一漂浮就 NaN。改 0.95 试
5. **weight decay**：wd 0.1 + 没排除 norm/bias，可能把 LayerNorm scale decay 到接近 0 触发数值问题。检查 param_group 配置
6. **gradient clipping**：上面都不行就直接上 clip_grad_norm_(1.0)，是治标但能立即止血
7. **检查 attention mask / RoPE**：Module 4 范围。某些 padding 处理 bug 会导致 attention 在 -inf 上 softmax，输出 NaN

加分：说明会做的 diagnostic——save NaN 前的 ckpt + 梯度直方图 + 找出哪一层先出现 NaN（hook 每层 forward 输出查 `torch.isnan`）

</details>

---

## 7. 延伸阅读

- [PyTorch torch.optim 源码](https://github.com/pytorch/pytorch/tree/main/torch/optim) — AdamW / SGD 的官方实现，看 `_single_tensor_adamw` 和 `_multi_tensor_adamw` 对比 fused / non-fused 路径
- [HuggingFace Optimization 文档](https://huggingface.co/docs/transformers/main_classes/optimizer_schedules) — 整理了所有 schedule 的可视化曲线，`get_cosine_schedule_with_warmup` 等函数即开即用
- [Sebastian Raschka — Practical Tips for Finetuning LLMs Using LoRA](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms) — LLM finetune 的 lr / wd / batch 实战调参经验
- [LLaMA paper 附录 A.5 / Qwen tech report](https://arxiv.org/abs/2302.13971) — 几个旗舰开源 LLM 公开的优化器超参，对照看会发现"约定俗成"的范围其实很窄
- 推荐继续读本教程的 **1.3 节《正则化与归一化：Dropout / BN / LN / RMSNorm》**——和本节 weight decay 是同一组"防过拟合"工具的两面
