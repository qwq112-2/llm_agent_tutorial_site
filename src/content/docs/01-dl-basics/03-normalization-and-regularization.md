---
title: "1.3 正则化与归一化：Dropout / BN / LN / RMSNorm"
description: "正则化（Dropout / Weight Decay）压制过拟合，归一化（BN / LN / RMSNorm）稳定训练并加速收敛——本节把四种 Norm 与三种正则化讲透，并解释为什么现代 LLM 几乎全部使用 RMSNorm + 极小 dropout + AdamW 这套组合。"
---

> ⏱ 预计阅读 35 分钟 ｜ 难度 ★★ 🔥 ｜ 前置：1.1 反向传播 ｜ 算法岗面试高频

## 一句话本节讲什么

正则化（Dropout / Weight Decay）压制过拟合，归一化（BN / LN / RMSNorm）稳定训练并加速收敛——本节把四种 Norm 与三种正则化讲透，并解释为什么现代 LLM 几乎全部使用 RMSNorm + 极小 dropout + AdamW 这套组合。

---

## 1. Mental model（直觉）

### 正则化与归一化是两个不同动机

读者最容易混的就是这两个词，先掰开：

- **正则化（regularization）** 的目标是**降低 generalization gap**——训练 loss 与 eval loss 的差距。手段是给模型施加"约束"或"噪声"：Dropout 给激活加噪、Weight Decay 给参数加 L2 惩罚、Label Smoothing 给标签加噪、Early Stopping 干脆不让训完。
- **归一化（normalization）** 的目标是**让训练稳定 / 收敛更快**——通过把中间激活的分布"拉回"某个固定区间（mean=0、std=1），避免每一层都需要重新适应前一层的分布漂移（"internal covariate shift"，BN 论文的原始动机；后来被证明这个解释不完全对，但稳定训练的事实是确凿的）。

两者在工程上经常一起出现，但解决的是不同问题。Dropout 调大了模型可能欠拟合；LayerNorm 没加好，模型可能根本训不下去（loss NaN）。

### 四种 Norm 的一张图

```
input shape: (B, T, C)   # batch, seq_len, channel/feature

BatchNorm:    沿 (B, T) 算 mean/var, 每个 channel 独立
              ┌────────────┐
              │   ........ │  ← 跨 batch、跨 token，对每个 feature 单独算
              └────────────┘

LayerNorm:    沿 C 算 mean/var, 每个 (B, T) 位置独立
              ┌────────────┐
              │   ▒▒▒▒▒▒▒▒ │  ← 一个 token 内部所有 feature 一起算
              └────────────┘

RMSNorm:      沿 C 算 RMS（不减 mean），每个 (B, T) 位置独立
              同 LayerNorm 的轴, 但只 scale 不 center
```

**关键 take-away**：BN 的统计量依赖 batch 维度（因此 batch 小 / batch 内分布异质就崩），LN/RMSNorm 完全在单个样本内部完成（因此天然适合变长序列、流式推理、小 batch）。这是 LLM 全面抛弃 BN 的根本原因。

### 为什么 LLM 用"小 dropout + 大 weight decay"

经典 CV 模型（ResNet）常用 dropout=0.5、weight_decay=1e-4 的组合。LLM 的配方完全不同：

- **Dropout 普遍 0.0-0.1**：现代 LLM 训练 token 数远超参数量（Chinchilla / over-training），数据本身已是"最强正则"，再加大 dropout 反而让信号被噪声 dominate。LLaMA-2/3、Qwen2/3、DeepSeek-V3 的预训练都是 dropout=0。
- **Weight decay 通常 0.1**：注意这是 AdamW 的 decoupled weight decay（见 1.2 节），数值上比 SGD 时代大得多，但因为 AdamW 把它从梯度里解耦了，本质是直接乘性收缩参数。

---

## 2. 公式与原理

### 2.1 Dropout

设第 $l$ 层激活 $h \in \mathbb{R}^d$，dropout 概率 $p \in [0, 1)$。**训练时**采样掩码 $m \in \{0, 1\}^d$，$m_i \sim \text{Bernoulli}(1-p)$ 独立同分布，输出：

$$
\tilde h = \frac{m \odot h}{1 - p}
$$

**推理时**直接 $\tilde h = h$，不做任何 mask。

这种实现叫 **inverted dropout**——把"补偿因子" $\frac{1}{1-p}$ 放在训练时，而不是放在推理时。这样推理代码和无 dropout 的网络完全一致，部署友好。早期论文里写的是"训练时不除、推理时乘 $1-p$"，工程上几乎没人这么做。

**为什么除以 $1-p$**：训练时的期望激活 $\mathbb{E}[\tilde h] = \frac{1}{1-p}(1-p) h = h$，与推理时一致。如果不除，训练 / 推理的激活幅值不匹配，模型在推理阶段会系统性偏弱。

**Hinton 的 mental model**：dropout 训练相当于在指数级多个共享参数的子网络上做 ensemble，推理时是这些子网络的近似几何平均。这个解释直观但不严格，工程上把它当 "noise injection" 理解就够。

### 2.2 BatchNorm

对一个 mini-batch $B = \{x^{(1)}, \dots, x^{(m)}\}$，每个 $x \in \mathbb{R}^C$（CNN 中通常按 channel 算，把空间维度也并进 batch 一起统计）：

$$
\mu_B = \frac{1}{m} \sum_{i=1}^m x^{(i)}, \quad \sigma_B^2 = \frac{1}{m} \sum_{i=1}^m (x^{(i)} - \mu_B)^2
$$

$$
\hat x = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \quad y = \gamma \odot \hat x + \beta
$$

其中 $\gamma, \beta \in \mathbb{R}^C$ 是可学习的 affine 参数（防止"normalize 之后表达力受限"），$\epsilon$（如 $1\text{e-}5$）防除零。

**训练 vs 推理**：训练时用 batch 统计量 $\mu_B, \sigma_B^2$；推理时用训练阶段维护的滑动平均 `running_mean`、`running_var`：

$$
\text{running\_mean} \leftarrow (1 - \alpha)\, \text{running\_mean} + \alpha\, \mu_B
$$

momentum $\alpha$ 通常 0.1。**这里就是工程灾难高发区**——`model.eval()` 必须在推理前显式调用，否则 BN 仍用当前 batch 的统计（甚至 batch=1 时方差是 0）。

**为什么 LLM 不用 BN**：

1. **变长序列**：BN 对 batch 维度求统计，把不同长度序列 pad 在一起算，pad 区域会污染统计量。
2. **小 batch 不稳**：LLM 训练 per-GPU batch 经常只有 1-4，batch 统计量噪声极大。
3. **多 GPU 同步成本**：BN 需要 cross-GPU 同步统计量（`SyncBatchNorm`），通信开销大。
4. **训推不一致**：LLM 经常一边训一边在线评估，running stats 还没收敛时 eval 就跑偏。

LN / RMSNorm 完全没有这些问题。

### 2.3 LayerNorm

对单个样本的 feature 向量 $x \in \mathbb{R}^d$：

$$
\mu = \frac{1}{d} \sum_{i=1}^d x_i, \quad \sigma^2 = \frac{1}{d} \sum_{i=1}^d (x_i - \mu)^2
$$

$$
\hat x = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}, \quad y = \gamma \odot \hat x + \beta
$$

形式与 BN 几乎一样，**唯一差别在求 $\mu, \sigma^2$ 的轴**：LN 沿 feature 维度算，每个 token / 每个样本独立。这意味着 batch size 是 1 还是 1024 完全不影响 LN 的输出，也无需 running stats。

在 Transformer 里，输入是 $(B, T, C)$，PyTorch 的 `nn.LayerNorm(C)` 会对最后一个维度做归一化——每个 (batch, time) 位置独立 normalize 自己的 channel 向量。

### 2.4 RMSNorm

Zhang & Sennrich 2019 的核心观察是：**LN 中"减 mean"这一步，对最终性能的贡献几乎可以忽略**。真正起作用的是把激活的"幅值"约束住。于是把 LN 简化成只 rescale：

$$
\text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}, \quad y = \frac{x}{\text{RMS}(x)} \odot g
$$

其中 $g \in \mathbb{R}^d$ 是可学习的 scale 参数（与 LN 的 $\gamma$ 同形），**不再有 bias $\beta$**。

**省了什么**：

- 不算 mean → 少一次 reduce
- 不减 mean → 少一次 elementwise sub
- 不学 bias → 少 $d$ 个参数

实测在 GPU 上比 LN 快 **10-50%**（取决于 hidden size 和 sequence length），且效果在 LLM 预训练上与 LN 几乎无差距甚至略好。这就是 LLaMA / Mistral / Qwen / DeepSeek 全系列把 LN 换成 RMSNorm 的原因。

### 2.5 Weight Decay 与 L2 正则化

经典 L2 正则化是在 loss 上加 $\frac{\lambda}{2} \|\theta\|^2$；weight decay 是在参数更新时直接乘 $(1 - \eta\lambda)$ 收缩参数。在 vanilla SGD 下两者数学等价。

但在 **Adam 下两者不等价**——Adam 用二阶矩 $\sqrt{v_t}$ 自适应缩放梯度，会同时缩放 L2 项的梯度，导致大梯度参数的 L2 惩罚被"稀释"。AdamW（Loshchilov & Hutter 2019）的修正是把 weight decay 从梯度中**解耦**，直接对参数做 $\theta \leftarrow \theta - \eta\lambda\theta$。详见 1.2 节。

**记忆点**：用 Adam 系优化器，永远写 `optim.AdamW(...)` 而不是 `optim.Adam(..., weight_decay=...)`。

### 2.6 完整正则化谱系（一段带过）

LLM 主要靠 weight decay + 极小 dropout，但完整的"正则化工具箱"还包含：

- **Data augmentation**：CV 标配（crop / flip / mixup），NLP 较少（back-translation / token mask 算半个）。
- **Label smoothing**：把 one-hot 标签 $[0, 0, 1, 0]$ 改成 $[0.025, 0.025, 0.925, 0.025]$，缓解模型对标签过自信。Inception-v3 / Transformer 原版都用 0.1。
- **Early stopping**：在 eval loss 拐点停止训练。LLM 训练通常按 token 预算训完不停，所以反而少用。
- **Gradient clipping**：严格说是稳定训练手段而非正则化，但常一起出现。LLM 标配 max_norm=1.0。

---

## 3. 最小代码示例

### 3.1 手写 RMSNorm（< 15 行）

```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(d))   # 可学习 scale，初始化为 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 关键：先转 fp32 算 RMS，再转回原 dtype，避免 fp16 上溢
        orig_dtype = x.dtype
        x_f = x.float()
        rms = x_f.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x_f * rms).to(orig_dtype) * self.g
```

关键点：

- `mean(dim=-1, keepdim=True)` 在最后一维（feature 维）做 reduce，broadcast 友好
- `rsqrt()` 比 `1/sqrt()` 快且数值更稳
- 先 cast 到 fp32 再计算 `pow(2)`——bf16/fp16 下 $x^2$ 容易超出 dtype 范围（fp16 max ≈ 65504），HF transformers 与 nanoGPT 都这么做

### 3.2 LayerNorm vs RMSNorm 对比（验证 + 计时）

```python
import time
import torch
import torch.nn as nn

torch.manual_seed(0)
B, T, C = 8, 1024, 4096
x = torch.randn(B, T, C, device="cuda", dtype=torch.bfloat16)

ln = nn.LayerNorm(C).cuda().to(torch.bfloat16)
rms = RMSNorm(C).cuda().to(torch.bfloat16)

# 1. 数值对比：随机 x 上两者输出的相对差异
y_ln, y_rms = ln(x), rms(x)
print(f"LN  mean={y_ln.float().mean():+.4f}  std={y_ln.float().std():.4f}")
print(f"RMS mean={y_rms.float().mean():+.4f} std={y_rms.float().std():.4f}")

# 2. 计时（warmup 后跑 100 次取均值）
def bench(fn, n=100):
    for _ in range(10): fn(x); torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n): fn(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000  # ms

print(f"LN  forward: {bench(ln):.3f} ms")
print(f"RMS forward: {bench(rms):.3f} ms")
```

典型 A100 上结果（数量级仅供参考）：

```
LN  mean=+0.0000  std=1.0000
RMS mean=+0.0001  std=0.9998
LN  forward: 0.42 ms
RMS forward: 0.31 ms   # ~25% 加速
```

LN 严格把 mean 拉到 0；RMSNorm 不约束 mean，但 std 也接近 1（因为输入本身近似零均值）。这就是 RMSNorm 论文的核心论点：**当 mean 项不重要时，去掉它换来效率几乎免费**。

### 3.3 Dropout 在 train / eval 模式下的差异

```python
import torch
import torch.nn as nn

torch.manual_seed(42)
drop = nn.Dropout(p=0.5)
x = torch.ones(8)

# train mode：随机 mask 50% 元素，剩下的乘 1/(1-0.5) = 2
drop.train()
print("train:", drop(x))   # 例如: tensor([2., 0., 2., 2., 0., 0., 2., 2.])

# eval mode：恒等映射，没有 mask 也不缩放
drop.eval()
print("eval :", drop(x))   # tensor([1., 1., 1., 1., 1., 1., 1., 1.])
```

**关键点**：是否 dropout 由 `model.train()` / `model.eval()` 控制，**不是**由有无 `optimizer.step()` 决定。验证集 / 推理时忘记切到 eval 模式，是初学者最常见 bug 之一（loss 看起来"震荡得很")。同样的开关也控制 BN 是否使用 running stats，所以 `model.eval()` 是 PyTorch 推理的强制动作。

---

## 4. 工程踩坑与经验

- ❗ **BN 忘了切 `model.eval()` → 推理性能塌**：训练阶段 BN 用 batch 统计，推理时如果还在 train mode，就会用当前 batch（可能只有 1 个样本）的 mean/var，方差直接为 0，输出 NaN 或全 0。所有 PyTorch 推理脚本第一行就该是 `model.eval()` + `torch.no_grad()`（或 `torch.inference_mode()`）。
- ❗ **多卡训练 BN 必须用 `SyncBatchNorm`**：DDP 下每张卡独立统计自己的 batch，等价 batch_size 除以卡数，统计噪声暴涨。`nn.SyncBatchNorm.convert_sync_batchnorm(model)` 一行解决，但通信开销大，这也是为什么大模型早就抛弃 BN。
- ❗ **从 ImageNet ckpt fine-tune 时 `running_mean/var` 不匹配新数据**：从自然图像迁移到医学影像 / 卫星图，输入分布完全不同，但 BN 的 running stats 还指向 ImageNet 的旧分布。常见做法：fine-tune 前用新数据跑几个 epoch 只更新 running stats（model.train() 但梯度 detach），或干脆 reset 后重新累计。
- ❗ **RMSNorm 在 fp16/bf16 下 `x.pow(2)` 容易溢出**：fp16 max ≈ 65504，$x^2$ 在 $|x| > 256$ 时就 overflow，整层输出 NaN。HF transformers 的 `LlamaRMSNorm`、nanoGPT 的实现都先 `x.float()` 再算 pow，最后 `.to(input_dtype)` 转回。这是 LLM 训练 NaN 排查清单上的高优先项。
- ❗ **Dropout p 不是越大越好**：经典 ResNet 的 0.5 是因为模型容量远大于数据；LLM 训练数据 token 数 / 参数数 ≥ 20（Chinchilla），数据本身就是强正则，再加 0.3+ dropout 会让模型欠拟合。LLaMA / Qwen / DeepSeek 预训练 dropout=0；SFT / DPO 等小数据后训练阶段才上 0.05-0.1。
- ❗ **LayerNorm 的 `eps` 别用默认 1e-5 训 LLM**：原版 1e-5 在 fp16 下偶尔触发数值问题，HF 实现 `LlamaRMSNorm` 用 1e-6，Megatron 用 1e-5 但开 fp32 reduce。眼前 loss spike 排查不到原因时，把所有 norm 的 eps 调到 1e-6 + 强制 fp32 reduce 是常见 mitigation。
- ❗ **AdamW 的 weight_decay 不要施加到 LN/RMSNorm 的 $\gamma$ 与 bias 上**：业界共识——norm 层的 scale / shift 参数不应被衰减，否则会破坏归一化效果。GPT-2、LLaMA 训练脚本都会把参数按 "decay / no_decay" 分组，norm 与 bias 进 no_decay 组（lr 相同但 weight_decay=0）。

---

## 5. 经典 paper

- **Hinton et al., 2012/2014 — *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*** — Dropout 的奠基论文。要 take away：(1) ensemble of subnetworks 这一 mental model；(2) inverted dropout 的实现细节；(3) 与 model averaging 的等价近似论证。
- **Ioffe & Szegedy, 2015 — *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift*** — BN 原典，提出 internal covariate shift 概念（后来被 Santurkar 2018 推翻，但 BN 的有效性是确凿的）。读了能理解为什么 CNN 时代 BN 是 default，以及为什么 sequence model 不能直接套。
- **Ba, Kiros & Hinton, 2016 — *Layer Normalization*** — LN 原典，明确提出"沿 feature 维度归一化、与 batch 解耦"的动机，最初是为 RNN 设计的，后来被 Vaswani 2017 沿用到 Transformer。
- **Zhang & Sennrich, 2019 — *Root Mean Square Layer Normalization*** — RMSNorm 原典。核心实验：在多个 NLP 任务上对比 LN vs RMSNorm，证明去掉 mean 项对效果几乎无影响、却能省 7-64% 的运行时间。这一发现直接被 LLaMA 全系列采纳。

---

## 6. 自测与面试题

**Q1（概念）：** 为什么现代 LLM 几乎全用 RMSNorm 而不是 LayerNorm？至少答出 2 点。

<details>
<summary>Answer sketch</summary>

至少包含：

- **效率**：少一次 mean reduce、少一次 sub、少一组 bias 参数，A100 上典型加速 10-50%。在 LLaMA 70B 这种规模上累积起来非常可观
- **效果不退化**：Zhang & Sennrich 2019 的实验显示 RMSNorm 在多任务上与 LN 几乎打平甚至略好，说明 LN 中 "减 mean" 这一步对最终性能贡献极小
- **代码更简洁**：少一个可学习参数 $\beta$，分布式训练 / 量化 / kernel fusion 都更容易处理
- 加分：Pre-LN Transformer 的输入本来就近似零均值（残差累积 + 层数深），mean 项的边际价值更低
- 加分：LLaMA / Mistral / Qwen / DeepSeek 全系采纳，已成事实标准

</details>

**Q2（实现）：** 写出 RMSNorm 的核心 4 行 PyTorch 代码（不算 `__init__`）。

<details>
<summary>Answer sketch</summary>

```python
def forward(self, x):
    orig_dtype = x.dtype
    x = x.float()                                          # 1. fp32 防溢出
    x = x * x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()  # 2. 除以 RMS
    return x.to(orig_dtype) * self.g                       # 3. 转回 dtype 后乘 scale
```

要点：

- `pow(2).mean(-1)` 是 RMS 的核心
- `rsqrt()` 比 `1/sqrt()` 数值更稳
- fp32 cast 是工程必须，否则 bf16/fp16 下 $x^2$ 易溢出
- 没有 bias，只有 scale `g`

漏掉 fp32 cast 不算错但是踩坑预警；漏掉 `keepdim=True` 会 broadcast shape 错；写成 LN 的"先减 mean 再除 std"算审题失败

</details>

**Q3（实战）：** 你 fine-tune 一个用 BN 的视觉 backbone，发现 eval loss 远高于 train loss，且 eval 输出分布异常。给出 3 个可能原因。

<details>
<summary>Answer sketch</summary>

至少包含其中 3 个：

- **忘了 `model.eval()`**：BN 仍在用当前 eval batch 的统计量，eval batch 小（甚至 1）时方差极小或为 0，输出畸形。最经典 bug
- **`running_mean / running_var` 与新数据分布不匹配**：从 ImageNet 预训练迁移到医学 / 卫星 / 工业视觉时，旧 stats 完全失效。修复：fine-tune 前用新数据跑几个前向只更新 stats，或 reset
- **多卡 DDP 下没用 `SyncBatchNorm`**：每张卡独立 BN，等效 batch 太小，stats 噪声大；训练时凑合，eval 时（单卡或不同 batch 划分）分布漂移暴露
- **BN 的 momentum 设得太大或太小**：默认 0.1 对 small batch fine-tune 不够 smooth，stats 跟着 loss 抖动；改 0.01 或冻结 BN
- **fine-tune 时 BN 应不应该冻结**：很多论文建议冻结 BN 到 eval 模式（`model.eval()` 但其他层 train），避免新数据少时 stats 被污染
- 进阶答：考虑直接把 BN 换成 GN/LN，或迁移到 transformer backbone 顺手就把这个问题绕开

加分：能说出 "如果 train loss 也异常 → 数据 / lr 问题；只有 eval 异常 → 大概率是 BN 或 dropout 的 train/eval mode 切换问题" 这种 debug 思路

</details>

---

## 7. 延伸阅读

- [Sebastian Raschka — Understanding LLM Architectures (RMSNorm 章节)](https://magazine.sebastianraschka.com/p/understanding-large-language-models) — 用清晰的 PyTorch 对照展示 LN→RMSNorm 的演化
- [PyTorch 官方文档 — `nn.LayerNorm` / `nn.BatchNorm2d` / `nn.Dropout`](https://pytorch.org/docs/stable/nn.html#normalization-layers) — 把每个参数（特别是 `eps`、`momentum`、`affine`、`track_running_stats`）的实际语义弄清楚
- [HF transformers — `LlamaRMSNorm` 源码](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) — 工业级 RMSNorm 实现（包含 fp32 upcast 与 fused kernel 路径）
- [Santurkar et al., 2018 — *How Does Batch Normalization Help Optimization?*](https://arxiv.org/abs/1805.11604) — 推翻 BN 原论文 "internal covariate shift" 解释，提出真正起作用的是 loss landscape 的平滑化。读完会对 BN 有更现代的理解
- 推荐继续读本教程的 **1.4 节《损失函数：CE / KL / 对比学习》**——把"输出层 → loss → 反向"链路补完；以及 **4.4 节《LayerNorm/RMSNorm 与 Pre-LN vs Post-LN》**，那里会展开 norm 在 Transformer block 的放置位置如何决定训练稳定性
