---
title: "1.4 损失函数：CE / KL / 对比学习"
description: "把 LLM 全栈最常用的三类 loss 一次讲清——CE 是预训练 / SFT 的主菜，KL 是 RLHF / distillation 的约束工具，InfoNCE 是 embedding / 多模态对齐的标准武器——以及这三者在数学上其实共享同一个信息论根源。"
---

> ⏱ 预计阅读 25 分钟 ｜ 难度 ★ ｜ 前置：1.1 反向传播

## 一句话本节讲什么

把 LLM 全栈最常用的三类 loss 一次讲清——**CE 是预训练 / SFT 的主菜，KL 是 RLHF / distillation 的约束工具，InfoNCE 是 embedding / 多模态对齐的标准武器**——以及这三者在数学上其实共享同一个信息论根源。

---

## 1. Mental model（直觉）

LLM 训练里你只会反复见到三类损失，它们解决的问题完全不一样，但都从 "log + 概率" 里长出来：

- **CE（Cross-Entropy）**：模型给出一个分布 $q$，真实标签是 $p$（通常是 one-hot），loss 衡量 "用 $q$ 编码真实分布 $p$ 平均要多少 bit"。bit 越少越好——也就是模型把概率质量越往真实标签上压越好。LLM 预训练 / SFT 的 next-token prediction，本质就是在每个位置上做一次 K=vocab_size 的 CE。

- **KL（Kullback-Leibler Divergence）**：两个分布之间的 "偏离量"，衡量 "用 $q$ 编码 $p$ 比用 $p$ 自己编码多浪费多少 bit"。在 LLM 里出现的两个最经典场景：(1) RLHF / RL 后训练时，把 $D_{KL}(\pi_\theta \| \pi_{\text{ref}})$ 加到 loss 里，**防止 policy 漂离 reference model 太远**（避免 reward hacking 和能力坍缩）；(2) Knowledge distillation 时，让 student 的输出分布对齐 teacher 的输出分布。

- **InfoNCE（对比学习）**：手头没有"标签是哪个类"，但有"哪两个样本是一对（正样本）"。loss 把正样本拉近、把 batch 内其他样本（负样本）推远——形式上长得就像 K=batch_size 的 CE。CLIP、SimCSE、bge / E5 / NV-Embed 这些 embedding 模型背后都是这个家伙。

一张直觉图，串起这三类 loss：

```
     真实分布 p              KL 衡量 "p 与 q 的距离"
        ↓                    CE 衡量 "用 q 编码 p 的代价"
        ↓                    InfoNCE 是 batch 内 K-way CE
   ┌────┴─────┐
   │  H(p)    │  ← 真实分布的固有熵（与模型无关，常数）
   ├──────────┤
   │  KL(p‖q) │  ← 模型可优化的部分
   └──────────┘
        ↓
     CE = H(p) + KL(p‖q)
```

记住这一行：**当 $p$ 是 one-hot（监督学习）时，$H(p) = 0$，所以 CE 与 KL 数值上完全等价、梯度也等价**——这就解释了为什么大多数分类问题随便叫 "CE loss" 或 "KL loss" 都对。

---

## 2. 公式与原理

### 2.1 从信息论到 CE

定义 $p, q$ 为同一个离散事件集合 $\mathcal{Y} = \{1, 2, \dots, K\}$ 上的两个概率分布，$p_y, q_y \in [0, 1]$，$\sum_y p_y = \sum_y q_y = 1$。

- **Entropy**（熵）：$H(p) = -\sum_y p_y \log p_y$，描述 $p$ 自身的不确定性。
- **Cross-Entropy**（交叉熵）：$H(p, q) = -\sum_y p_y \log q_y$，描述用 $q$ 编码 $p$ 的平均代价。
- **KL Divergence**：$D_{KL}(p \| q) = \sum_y p_y \log \frac{p_y}{q_y} = H(p, q) - H(p)$。

监督分类的关键观察：当 $p$ 是 one-hot（即只有真实类 $y$ 上 $p_y = 1$，其他位置为 0），

$$H(p, q) = -\log q_y, \quad H(p) = 0, \quad D_{KL}(p \| q) = -\log q_y$$

所以 CE loss 写成

$$L_{\text{CE}}(x, y) = -\log q_y = -\log \frac{\exp(z_y)}{\sum_k \exp(z_k)}$$

其中 $z \in \mathbb{R}^K$ 是 logits，$q = \text{softmax}(z)$。这就是 PyTorch `F.cross_entropy(logits, labels)` 一行的展开。

**与 MLE 的等价**：在 i.i.d. 假设下最小化 CE = 最大化对数似然。这是 Goodfellow Ch.5 的核心定理之一。LLM 预训练的 next-token prediction 在每个位置 $t$ 上算 $-\log P_\theta(x_t | x_{<t})$，sum 起来就是序列的负对数似然，等价于对全语料做 MLE。

**Label smoothing**：把 one-hot 改成

$$\tilde{p}_y = (1 - \alpha) \cdot \mathbb{1}[y = y_{\text{true}}] + \frac{\alpha}{K}$$

引入小常数 $\alpha$（典型 0.1）把概率质量散一点到非真实类。直观上避免模型对正确类输出无穷大 logit（导致 over-confidence、校准变差），来自 Szegedy 2015 Inception-v3 的工程经验。

### 2.2 KL 的不对称性

$D_{KL}(p \| q) \neq D_{KL}(q \| p)$，这一点在 LLM 里很关键。

- **Forward KL**：$D_{KL}(p \| q) = \sum_y p_y \log \frac{p_y}{q_y}$。当 $p_y > 0$ 而 $q_y \to 0$，loss 爆炸——这迫使 $q$ "覆盖" $p$ 的所有支集，行为上 **zero-avoiding（mass-covering）**。Distillation 通常用这个方向：student 必须覆盖 teacher 所有可能输出。
- **Reverse KL**：$D_{KL}(q \| p) = \sum_y q_y \log \frac{q_y}{p_y}$。当 $p_y \to 0$ 而 $q_y > 0$ 时爆炸——这迫使 $q$ 只在 $p$ 的高密度区放质量，行为上 **mode-seeking**。RLHF 的 PPO loss 项 $\beta \cdot D_{KL}(\pi_\theta \| \pi_{\text{ref}})$ 是 reverse KL，让 policy 倾向 stick 在 ref 的几个模式附近，而不是去探索 ref 没怎么覆盖的区域。

**RLHF 中的角色**：完整 PPO 目标含约束项

$$\mathcal{L}_{\text{PPO}}(\theta) = \mathbb{E} \left[ \min(r_t \hat{A}_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) \hat{A}_t) \right] - \beta \cdot D_{KL}(\pi_\theta \| \pi_{\text{ref}})$$

$\beta$ 是 KL 系数，控制 policy 偏离 ref 的力度，**典型 0.01-0.1**。$\beta$ 太小 → policy 自由漂走 → reward hacking + 通用能力坍缩；$\beta$ 太大 → policy 被钉在 ref 上 → 学不到新行为。Module 9.3 / 9.6 会详细展开。

**Distillation 中的角色**：温度软化的 KL，

$$\mathcal{L}_{\text{distill}} = T^2 \cdot D_{KL}\left(\text{softmax}(z_t / T) \| \text{softmax}(z_s / T)\right)$$

teacher logits $z_t$ 和 student logits $z_s$ 都过温度 $T > 1$ 软化，$T^2$ 用来补偿梯度被 $1/T^2$ 缩放的事实（Hinton 2015）。

### 2.3 InfoNCE：当只有"哪两个是一对"

设 query $q$、正样本 $k^+$、$N-1$ 个负样本 $\{k_i^-\}_{i=1}^{N-1}$。InfoNCE loss 写成

$$\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\text{sim}(q, k^+) / \tau)}{\exp(\text{sim}(q, k^+) / \tau) + \sum_i \exp(\text{sim}(q, k_i^-) / \tau)}$$

其中 $\text{sim}$ 通常是 cosine similarity（先 L2 normalize 再点积），$\tau > 0$ 是 temperature。

形式上这就是一个 K-way classification 的 CE loss，"类别" 是 N 个候选 key，"正确类" 是 $k^+$。van den Oord 2018 证明它是 mutual information 的下界，所以叫 InfoNCE。

**Temperature τ 的作用**：

- $\tau$ 小（如 0.05）：softmax 被锐化，模型更看重 hardest negative，收敛快但容易过拟合 hard negative。
- $\tau$ 大（如 0.5）：softmax 被平滑，所有负样本都贡献梯度，对 hard negative 没那么敏感，收敛慢但更稳。
- CLIP 用 **learnable τ**，初始化 $1/\tau \approx 14.3$（即 $\tau \approx 0.07$），训练中可学；SimCSE 固定 $\tau = 0.05$；MoCo $\tau = 0.07$。

**典型应用**：

- **CLIP**：image-text 配对，batch 内 N 张图、N 段文本，构造 $N \times N$ 相似度矩阵，对角线是正样本，行/列两个方向分别做 InfoNCE，loss 取均值。
- **SimCSE**：同一句话过 encoder 两次（dropout 不同）作为正样本对，batch 内其他句子作负样本。
- **bge / E5 / NV-Embed**：LLM-era embedding，hard negative 由 BM25 / ANN 检索挖出来，配合 in-batch negative + cross-batch gather。

### 2.4 回归类损失（一段带过）

回归任务（output 是连续值）常用 MSE：$\mathcal{L} = \frac{1}{N} \sum (y - \hat{y})^2$；对 outlier 鲁棒一点的版本是 Huber loss（小残差用 L2、大残差用 L1）。在 LLM 后训练里用得相对少，但 reward model **如果输出 scalar reward** 可以用 MSE 拟合人工标的连续偏好分（实际工业 RM 几乎都用 Bradley-Terry 的 pairwise sigmoid loss，把"A 优于 B" 转为二分类，详见 9.2）。

---

## 3. 最小代码示例

### 3.1 CE（带 mask 与 label smoothing）

```python
import torch
import torch.nn.functional as F

# 模拟一个 batch：B=2, T=4, V=10
B, T, V = 2, 4, 10
logits = torch.randn(B, T, V)        # 模型 raw 输出，未 softmax
labels = torch.tensor([[3, 5, 7, -100],   # -100 = padding，不算 loss
                       [1, 2, -100, -100]])

# 关键：F.cross_entropy 期望 (N, C) 的 logits，所以 reshape
loss_mean = F.cross_entropy(
    logits.reshape(-1, V),           # (B*T, V)
    labels.reshape(-1),              # (B*T,)
    ignore_index=-100,               # mask 掉 padding 位置
    label_smoothing=0.1,             # SFT 常用 0.0~0.1
    reduction='mean',                # 对非 ignore 位置取均值
)

# reduction='none' 拿到每个 token 的 loss，常用于自定义 weighted loss
loss_per_token = F.cross_entropy(
    logits.reshape(-1, V), labels.reshape(-1),
    ignore_index=-100, reduction='none',
)  # shape: (B*T,)，被 ignore 的位置值为 0
```

关键点：

- `F.cross_entropy` 内部已经做了 `log_softmax + nll_loss`，**外面千万不要再套一层 softmax**（新手 #1 bug）。
- `ignore_index=-100` 是 HuggingFace 全家桶的默认 padding 标签，预处理脚本会把 padding token 对应的 label 设为 -100。
- `reduction='mean'` 对**非 ignore** 位置取平均；`'sum'` 求和（多卡训练时如果想做 token-level mean 通常先 sum 再除全局有效 token 数）；`'none'` 不规约，自己后处理。

### 3.2 InfoNCE 手撕（in-batch negative 版）

```python
import torch
import torch.nn.functional as F

def info_nce(query, key, temperature=0.07):
    """
    query, key: (B, D)，已经过 encoder 输出的 embedding
    第 i 个 query 的正样本就是第 i 个 key，其余 (B-1) 个 key 为负样本
    """
    q = F.normalize(query, dim=-1)         # L2 normalize → cosine sim
    k = F.normalize(key, dim=-1)
    logits = q @ k.t() / temperature       # (B, B) 相似度矩阵
    labels = torch.arange(q.size(0), device=q.device)  # 对角线为正
    return F.cross_entropy(logits, labels)

# 测试
B, D = 8, 128
q = torch.randn(B, D)
k = torch.randn(B, D)
loss = info_nce(q, k)                      # 标量
```

13 行就是 SimCSE / CLIP 的核心。CLIP 是双向版（再算一次 `info_nce(k, q)` 取均值），bge / E5 在此基础上加了显式 hard negative（拼到 logits 里）和 cross-device gather（见踩坑第 5 条）。

### 3.3 KL（distillation 与 RLHF 通用）

```python
import torch
import torch.nn.functional as F

# student / teacher logits，shape (B, V)
s_logits = torch.randn(8, 1000, requires_grad=True)
t_logits = torch.randn(8, 1000)

# F.kl_div 的 API 坑：第 1 个参数必须是 log-probs，第 2 个必须是 probs
kl = F.kl_div(
    F.log_softmax(s_logits, dim=-1),    # 注意是 log_softmax
    F.softmax(t_logits, dim=-1),        # 注意是 softmax
    reduction='batchmean',              # 推荐：先求和再除 batch
    log_target=False,                   # True 时第 2 个也是 log-probs
)
# kl 计算的是 D_KL(target || input) = D_KL(teacher || student)
```

PyTorch `F.kl_div` 的两个 API 设计陷阱：

1. **第一个参数是 log-probs，不是 logits 也不是 probs**。如果传 logits 会算出错的数；如果传 probs 会得到 NaN（log(0) = -inf 然后参与运算）。
2. **`reduction='mean'` 是按 element 求平均（除以 B*V）**，不是按 batch；要想按 batch 取平均必须用 `reduction='batchmean'`，这是 KL 的"标准"定义。
3. 计算的方向是 $D_{KL}(\text{target} \| \text{input})$。如果你想算 reverse 方向（如 RLHF 里 $D_{KL}(\pi_\theta \| \pi_{\text{ref}})$），让 $\pi_\theta$ 当 input、$\pi_{\text{ref}}$ 当 target——记住"input 在前是 log，target 在后是 prob"。

---

## 4. 工程踩坑与经验

- ❗ **`F.cross_entropy` 已经包含 `log_softmax`，不要再外面套一层 `softmax`**——这是新手 #1 bug，套了之后 loss 看起来还像样但数值完全错（梯度也错）。如果你已经手工算了 log_softmax，请用 `F.nll_loss(log_probs, labels)`，二者只差 log_softmax 这一步。
- ❗ **`F.kl_div(input, target)` 的 input 必须先 `log_softmax`，target 必须先 `softmax`**——API 不对称（不像 CE 全自动），这是踩坑率第二高的点。同时 `reduction` 默认 `'mean'` 会按 element 平均，要 KL 的标准定义请显式传 `reduction='batchmean'`。
- ❗ **Label smoothing 在 LLM 预训练 / DPO 中通常不用**，因为预训练阶段保留 model 输出 confidence 的真实形态对下游 RM / DPO 至关重要；在 SFT / 多分类任务里 $\alpha=0.1$ 是常见默认。RLHF 之后再加 label smoothing 几乎没人做。
- ❗ **InfoNCE 的 temperature 是关键超参**，对结果影响远大于 lr / batch。CLIP 用 learnable $\tau$（初始 0.07）+ clip 防止过小，SimCSE 固定 0.05，bge 固定 0.02——盲目套别人的 $\tau$ 经常掉 2-5 个点。先 sweep 再训。
- ❗ **多 GPU 训对比学习时，需要 `all_gather` 全局 negatives**——单卡 batch 256 → 8 卡 in-batch negative 仍然只有 256 个会浪费数据。正确做法：每张卡 forward 自己 batch，然后 `dist.all_gather(embeddings)` 拼成 256×8=2048 的全局 batch 算 InfoNCE。注意 gather 出的 tensor 默认不带梯度，需要把"自己这份"插回原 tensor 保留梯度（`local_idx` 这一段是开源 contrastive 训练库的标配 trick）。
- ❗ **`reduction='mean'` 在变长序列上是 micro-average**——`F.cross_entropy(..., ignore_index=-100, reduction='mean')` 是对所有非 -100 token 取均值，长序列权重大、短序列权重小。如果想做 sequence-level mean（每条样本贡献相等），需要先 `reduction='none'` 拿 per-token loss，再按样本 mask 自己求均值。HF Trainer 默认是 micro-average，多数人不知道。
- ❗ **CE loss 的 NaN 大概率是 logits 出现 inf**（fp16 下尤其常见，比如 attention logits 没做 mask 处理）——先 `torch.isfinite(logits).all()` 排查，再排查 label 是否越界（label >= vocab_size 时 CE 会直接挂或者出 NaN）。

---

## 5. 经典 paper

- **Goodfellow, Bengio & Courville, 2016 — Deep Learning, Ch.5**（[deeplearningbook.org](https://www.deeplearningbook.org/)）— 关于 entropy / cross-entropy / MLE / KL 的最佳一站式综述。读 Ch.5.5 (MLE) 与 Ch.3.13 (Information Theory) 这两节，本节所有数学的推导背景都在那里。Murphy《PRML》Ch.2-3 是更厚但更严谨的替代。
- **van den Oord, Li & Vinyals, 2018 — Representation Learning with Contrastive Predictive Coding (CPC)** — InfoNCE 的提出者，证明它是 mutual information 的下界。读 §3-4 即可，§5 之后是 CPC 的具体应用，可跳。
- **Radford et al., 2021 — Learning Transferable Visual Models From Natural Language Supervision (CLIP)** — 对比学习用到 image-text 上的代表作，把 InfoNCE 双向 loss + learnable temperature 这套范式确立下来。本教程 Module 16.1 / 16.4 都会回到 CLIP 的 idea。
- **Szegedy et al., 2015 — Rethinking the Inception Architecture for Computer Vision** — label smoothing 的工程引用源（§7），即使你不做 CV，引这篇论 label smoothing 是学界惯例。

---

## 6. 自测与面试题

**Q1（概念）**：CE 和 KL 的关系是什么？什么时候它们的梯度等价？

<details>
<summary>Answer sketch</summary>

- 关系式：$H(p, q) = H(p) + D_{KL}(p \| q)$，CE = entropy + KL。
- 当 $p$ 与模型参数 $\theta$ 无关时（监督学习里 $p$ 是固定的真实标签分布），$H(p)$ 是常数，$\nabla_\theta H(p, q) = \nabla_\theta D_{KL}(p \| q)$，所以梯度等价、最优解一样。
- 极端情况：$p$ 是 one-hot 时 $H(p) = 0$，连数值都相等，CE = $-\log q_y$ = $D_{KL}(p \| q)$。
- 反例：在 distillation / RLHF 里 $p$（teacher / ref 分布）也会变化时，CE 和 KL 行为不同——一般用 KL 而不用 CE，因为只想约束分布距离不想被 teacher 自己的熵搅进 loss。

</details>

**Q2（API）**：写出 PyTorch 计算 $D_{KL}(\pi_{\text{student}} \| \pi_{\text{teacher}})$ 的正确代码，注意 log_softmax / softmax 顺序。

<details>
<summary>Answer sketch</summary>

要点：

- `F.kl_div` 的签名是 `kl_div(input, target)`，计算 $D_{KL}(\text{target} \| \text{input})$（注意方向反了）。
- 想算 $D_{KL}(\text{student} \| \text{teacher})$ 就要让 **student 当 target、teacher 当 input**（容易写反！）。但更常见的 distillation 写法是想算 $D_{KL}(\text{teacher} \| \text{student})$（forward KL），即 teacher 当 target、student 当 input。
- input 必须先 `log_softmax`，target 必须先 `softmax`。
- 必须传 `reduction='batchmean'`。

参考代码（forward KL，distillation 标准方向）：

```python
loss = F.kl_div(
    F.log_softmax(student_logits, dim=-1),
    F.softmax(teacher_logits, dim=-1),
    reduction='batchmean',
)
```

如果题目要求 reverse 方向，互换 student / teacher 在调用里的位置即可。

</details>

**Q3（trade-off）**：在 RLHF 的 PPO loss 里加约束项 $\beta \cdot D_{KL}(\pi_\theta \| \pi_{\text{ref}})$，$\beta$ 设大 / 设小分别有什么后果？为什么 RLHF 一定要这一项？

<details>
<summary>Answer sketch</summary>

- **$\beta$ 太小 / 不加**：policy 自由漂离 ref → 容易 reward hacking（找到 reward model 的偏好漏洞但实际生成质量崩）+ 通用能力坍缩（pretrain / SFT 学到的多样性丢失，输出风格变窄）。极端情况 KL 直接发散，训练不收敛。
- **$\beta$ 太大**：policy 被钉在 ref 附近 → 学不到 RM 想引导的新行为，reward 提升缓慢甚至原地踏步。极端情况 KL 项主导 loss，PPO 退化成"纯模仿 ref"。
- **典型经验值**：0.01~0.1，不同实现差异大；DeepSeek、Llama 论文都给过 ablation。还可用 adaptive KL controller（动态调 $\beta$ 维持目标 KL 在 [3, 10] nats / sequence 范围）。
- **本质原因**：RM 是用有限人工标注训出来的近似器，远离训练分布的样本上 RM 的判别不可信；KL 约束等价于"只在 ref 附近的可信邻域里搜索 reward 更高的策略"。这是后训练 RL 与游戏 RL 最大的不同——LLM 不能纵容 policy 自由探索。
- 加分：DPO 的 $\beta$ 参数有类似作用但形式上是闭式解里的温度，与 PPO 的 $\beta$ 不能直接互换数值；GRPO 在 DeepSeek 实现里把 KL 改成了 unbiased estimator 形式。Module 9.3 / 9.4 / 9.5 详讲。

</details>

---

## 7. 延伸阅读

- [Lilian Weng — Contrastive Representation Learning](https://lilianweng.github.io/posts/2021-05-31-contrastive/) — InfoNCE / NCE / Triplet / SupCon 一篇通讲，配大量直觉图，本节对比学习部分的最佳延伸。
- [Sebastian Raschka — A Visual Introduction to KL Divergence](https://magazine.sebastianraschka.com/) — KL 的可视化讲解，特别适合搞清楚 forward vs reverse 的几何意义。
- [HuggingFace TRL `PPOTrainer` 源码](https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py) — 看 `compute_rewards` 和 KL 项的实现，是把本节 KL 公式落到 RLHF 工程上的最直接代码参考。
- [PyTorch `F.cross_entropy` / `F.kl_div` 官方文档](https://pytorch.org/docs/stable/nn.functional.html) — API 细节，特别是 `ignore_index` / `label_smoothing` / `reduction` 的精确语义，建议读原文一次免得踩坑。
- 推荐继续读本教程的 **1.5 节《PyTorch 工作流与显存 / dtype 心智模型》**——把 loss 与 fp16 / bf16 / loss scaling 串起来。
