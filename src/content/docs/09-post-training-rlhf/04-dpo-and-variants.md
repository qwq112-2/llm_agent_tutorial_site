---
title: "9.4 DPO 闭式解推导 + 变体（IPO / KTO / SimPO / ORPO / RLOO）"
description: "DPO 把 RLHF 三段式（SFT → RM → PPO）的后两段合并成\"一个 supervised loss\"：通过对 KL 约束下 RL 最优解的闭式反演，把 reward 用 policy / reference 的对数比表示出来，再代入 Bradley-Terry preference model，得到一个不需要 RM、不需要 sampling、像 SFT 一样训的偏好优化算法——本节把"
---

> ⏱ 预计阅读 45 分钟 ｜ 难度 ★★★ 🔥 ｜ 前置：9.2 Reward Model、9.3 PPO、1.4 KL

## 一句话本节讲什么

DPO 把 RLHF 三段式（SFT → RM → PPO）的后两段合并成"一个 supervised loss"：通过对 KL 约束下 RL 最优解的闭式反演，把 reward 用 policy / reference 的对数比表示出来，再代入 Bradley-Terry preference model，得到一个不需要 RM、不需要 sampling、像 SFT 一样训的偏好优化算法——本节把这条推导链每一步说清楚，再把 IPO / KTO / SimPO / ORPO / RLOO 这五个 2023-2024 年的主流变体放进同一张对比表里讲透。

---

## 1. Mental model（直觉）

9.3 节的 PPO 跑起来要装四个模型（policy + reference + RM + critic）、每步要 generate、要 reward forward、要 KL 估计——8B 模型一张 80G 卡都吃力。RLHF 的"重"很大程度上来自这套架子。

DPO（Direct Preference Optimization, Rafailov 2023）的洞察非常优雅：

> RLHF 的优化目标 $\max_\pi \mathbb{E}[r] - \beta D_{KL}(\pi \| \pi_{\text{ref}})$ 是有闭式最优解的。这个闭式解可以**反过来**把 reward 写成 policy 与 reference 的对数比。把这个表达式塞回 Bradley-Terry preference model，reward 的"绝对值"项（含归一化常数 $Z(x)$）在 chosen-rejected 差里直接抵消掉了——剩下的就是一个完全用 policy log-probability 表达的 supervised loss。

换句话说：**RLHF 用 RM 做了一次"偏好 → reward 标量"的中间编码，然后再让 policy 去拟合这个 reward；DPO 直接跳过中间编码，把 policy 写进 preference 似然里直接训**。中间编码丢了，但目标函数等价（同一个 KL-constrained reward maximization 的解析解）。

直觉上 DPO 在做的事是：

```
        +─────── 增大 log π(y_w | x) / π_ref(y_w | x)  ──+
        │                                               │
偏好对  │                                               │ 等价于
(x, y_w, y_l)  ────→  loss = -log σ(β · [Δw - Δl])  ←──┤  "implicit reward"
        │                                               │  r̂(x, y) = β log π/π_ref
        +─────── 减小 log π(y_l | x) / π_ref(y_l | x)  ──+
```

代价是：
1. **Offline**——只能用静态偏好对，不能在线探索（policy 不更新分布上的"新数据"）；
2. **没有 reward function**——你失去了"给任意 (x, y) 打分"的能力（详见 9.2 §6 Q3）；
3. **效果上限略低于 PPO**——在大规模 / 强 RM / 强探索场景下，PPO 仍然更强（LLaMA-3、Tülu 3 路线最终都还是接了一段 PPO refine）。

但 DPO 的工程性价比极高——一个 PPO 工程要几人周搭，DPO 一晚上写完一个 trainer 就能跑。它是 2024 年开源对齐的"事实标准"。本节后半部分的 5 个变体则各自针对 DPO 的某个痛点：

- **IPO** 修复 DPO 在 deterministic preference 下的过拟合（log-sigmoid 永远不饱和）；
- **KTO** 把 pairwise 数据松弛到 unary（这条好/这条不好），数据更易收集；
- **SimPO** 干掉 reference model（省一半显存）+ length normalize；
- **ORPO** 把 SFT 和 preference 揉成一个阶段（不需要 SFT pre-stage）；
- **RLOO** 走另一条路：留在 on-policy，但用 leave-one-out baseline 去掉 critic，向 GRPO 心智模型靠拢。

记住一句：**DPO 是"把 RLHF 的最优解直接当 supervised target"，变体都在动它的某一条假设**。

---

## 2. 公式与原理

### 2.1 从 RLHF 目标到 DPO 闭式解（核心 4 步）

**Step 1 — 写出 KL-约束的 RLHF 优化目标**

给定 prompt 分布 $\rho(x)$、reference policy $\pi_{\text{ref}}$、reward function $r(x, y)$、KL 系数 $\beta > 0$，RLHF 的优化目标是

$$\max_{\pi}\; \mathbb{E}_{x \sim \rho,\, y \sim \pi(\cdot|x)}\bigl[r(x, y)\bigr]\; -\; \beta\, D_{KL}\bigl(\pi(\cdot|x)\,\|\,\pi_{\text{ref}}(\cdot|x)\bigr)$$

其中 $\pi(\cdot|x), \pi_{\text{ref}}(\cdot|x)$ 都是定义在 response 空间 $\mathcal{Y}$ 上的概率分布。9.3 节讲过 PPO 是这个目标的 stochastic gradient + clipping 近似。

**Step 2 — 求闭式最优 policy**

固定 $x$，把目标写成

$$\max_{\pi(\cdot|x)} \sum_y \pi(y|x)\,r(x, y)\; -\; \beta \sum_y \pi(y|x) \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}$$

约束 $\sum_y \pi(y|x) = 1$，$\pi(y|x) \geq 0$。这是带归一化约束的凸优化（KL 是凸的、reward 项是线性的），用 Lagrange 乘子或直接套"max-entropy regularized RL"的标准结论得到

$$\boxed{\; \pi^*(y|x)\;=\;\frac{1}{Z(x)}\,\pi_{\text{ref}}(y|x)\,\exp\!\left(\frac{1}{\beta}\,r(x, y)\right)\;}$$

其中 $Z(x) = \sum_y \pi_{\text{ref}}(y|x)\exp(r(x, y)/\beta)$ 是把 $\pi^*$ 归一化为概率分布的 partition function。

直觉上这是 reference policy "被 reward 加权"的版本——reward 大的 $y$ 概率被指数级放大，$\beta$ 越小放大越激进；$\beta \to \infty$ 时 $\pi^* \to \pi_{\text{ref}}$，$\beta \to 0$ 时 $\pi^*$ 退化到 reward 的 argmax。

**Step 3 — 反演 reward**

对 Step 2 的闭式解两边取对数、整理：

$$\log \pi^*(y|x) = \log \pi_{\text{ref}}(y|x) + \frac{1}{\beta} r(x, y) - \log Z(x)$$

解出 reward：

$$r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)$$

**这一步是 DPO 的核心 trick**：reward function 可以**完全用 (最优 policy, reference policy) 的对数比**表达。注意 $\beta \log Z(x)$ 这一项只依赖 $x$ 不依赖 $y$，是 prompt-specific 常数。

**Step 4 — 代入 Bradley-Terry，常数项抵消**

回顾 9.2 §2.1，Bradley-Terry preference model 给"$y_w$ 比 $y_l$ 好"的概率：

$$P(y_w \succ y_l \mid x) = \sigma\bigl(r(x, y_w) - r(x, y_l)\bigr)$$

把 Step 3 的 reward 表达式代入差 $r(x, y_w) - r(x, y_l)$：

$$r(x, y_w) - r(x, y_l) = \beta \log \frac{\pi^*(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi^*(y_l|x)}{\pi_{\text{ref}}(y_l|x)} + \underbrace{\beta\log Z(x) - \beta \log Z(x)}_{=\,0}$$

**$\beta \log Z(x)$ 这个棘手的常数（要知道它得对整个 response 空间求和，根本算不动）在 chosen-rejected 差里完美抵消**。这就是 DPO 能落地的关键——否则我们得估 $Z(x)$，那就和 PPO 一样得 sampling，DPO 也就没有意义了。

把 $\pi^*$ 替换成参数化的 $\pi_\theta$（即"假设当前 policy 就是最优 policy"），对偏好数据 $\mathcal{D} = \{(x, y_w, y_l)\}$ 取负对数似然：

$$\boxed{\;\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\!\left[\log \sigma\!\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]\;}$$

记 **implicit reward** $\hat{r}_\theta(x, y) := \beta \log (\pi_\theta(y|x) / \pi_{\text{ref}}(y|x))$，则 DPO loss 的形式与 9.2 节 RM loss $-\log \sigma(r_w - r_l)$ **完全相同**——只是把"显式 RM 的 reward"换成了"policy ratio 的 log"。这就是"DPO 把 RM 隐含进 loss 里"的精确含义。

### 2.2 梯度直觉

对 $\theta$ 求导（记 $\Delta = \hat{r}_\theta(x, y_w) - \hat{r}_\theta(x, y_l)$）：

$$\nabla_\theta \mathcal{L}_{\text{DPO}} = -\beta\,\bigl(1 - \sigma(\Delta)\bigr)\,\Bigl[\nabla_\theta \log \pi_\theta(y_w|x) - \nabla_\theta \log \pi_\theta(y_l|x)\Bigr]$$

读这个梯度：
- **方向**：增大 $\log \pi_\theta(y_w|x)$，减小 $\log \pi_\theta(y_l|x)$——直觉上"奖励 chosen、惩罚 rejected"；
- **强度** $1 - \sigma(\Delta)$：当 chosen 已经远高于 rejected（$\Delta \gg 0$）时趋零，即 RM 训练时的同款 margin 行为——容易样本不再贡献梯度；
- **$\beta$ 的角色**：直接缩放梯度强度，但更重要的是它定义了 implicit reward 的尺度——$\beta$ 大则 policy 更难偏离 reference（每一点 ratio 差别都被放大成大 reward）；$\beta$ 小则 policy 可以偏离更远（同样的 ratio 差只对应小 reward，需要更剧烈的 policy 改动才能拉开 $\Delta$）。

### 2.3 DPO 与 PPO 的对比

| 维度 | PPO | DPO |
|------|-----|-----|
| Reward Model | **显式**，独立训 | **隐含**于 loss（$\hat{r} = \beta \log \pi/\pi_{\text{ref}}$）|
| 数据流向 | online（policy 实时 generate）| offline（静态 preference dataset）|
| 模型数量 | 4（policy + ref + RM + critic）| 2（policy + ref）|
| 显存 | 重（≥ 2.5× SFT）| 轻（约 2× SFT，chosen / rejected 双 forward）|
| 实现复杂度 | 高（rollout / GAE / KL 估计 / clipping）| 低（一个 supervised loss）|
| 训练稳定性 | 难调（reward hacking / KL collapse / value 崩）| 稳定（像 SFT 一样）|
| 探索能力 | 强（on-policy）| 无（受限于 dataset 覆盖）|
| 效果上限 | 高（在大资源 + 强 RM 下）| 略低（offline 限制）|
| 适用场景 | 大资源、需要持续探索 | 数据已有、资源紧张、快速迭代 |

### 2.4 $\beta$ 的物理意义与调参

$\beta$ 在 DPO 里有双重身份——它既是 KL 约束强度（来自 RLHF 目标），又是 implicit reward 的尺度（$\hat{r} = \beta \log \pi/\pi_{\text{ref}}$）。

- $\beta \to 0$：完全跟随 preference，policy 可以无视 reference 大幅漂移——容易 overfit 到训练 pair，对 OOD prompt 输出崩坏；
- $\beta \to \infty$：基本不动 reference model（KL 约束极强）——learning signal 被压扁，几乎学不到东西；
- 经验范围：**$\beta \in [0.05, 0.5]$**，开源经典配置 $\beta = 0.1$。LLaMA-3 / Tülu 3 / Zephyr 都在这个区间。

调参规则：训练数据偏好信号弱（标注噪声大、chosen / rejected 差距小）→ $\beta$ 小一些（如 0.05）让模型更"听话"；偏好信号强且想保留 base capability → $\beta$ 大一些（如 0.3）。

### 2.5 五个变体的核心公式

为不重复，每个变体只给最关键的 loss 形式与"和 DPO 比改了什么"。

**IPO（Identity Preference Optimization, Azar et al. 2023）**

DPO 用 log-sigmoid loss，问题是当数据是 deterministic preference（即 $P(y_w \succ y_l) = 1$，标注完全一致）时，loss $-\log \sigma(\beta \Delta)$ 对 $\Delta$ 永远不饱和——梯度推 $\Delta \to \infty$，policy 在 chosen 上的概率推到极致、rejected 推到 0，过拟合严重。IPO 把 sigmoid 换成 squared loss：

$$\mathcal{L}_{\text{IPO}}(\theta) = \mathbb{E}\!\left[\Bigl(\hat{r}_\theta(x, y_w) - \hat{r}_\theta(x, y_l) - \frac{1}{2\tau}\Bigr)^2\right]$$

即"让 chosen-rejected 的 implicit reward 差**等于** $1/(2\tau)$"，而不是"越大越好"。$\tau$ 是 IPO 的 KL-like 系数，与 DPO 的 $\beta$ 反比对应（$\tau$ 大 → 目标 margin 小 → 更接近 reference）。

**KTO（Kahneman-Tversky Optimization, Ethayarajh et al. 2024）**

DPO 要 pairwise 数据 $(x, y_w, y_l)$，但很多场景人能给的反馈是 unary 的——"这条 response 是好/是坏"（点赞 / 点踩、举报、用户重写）。KTO 基于 prospect theory（人对损失比对收益更敏感）设计 loss：

$$\mathcal{L}_{\text{KTO}}(\theta) = \mathbb{E}\bigl[\lambda_y\,(1 - \sigma(\beta \cdot (\hat{r}_\theta(x, y) - z_0)))\bigr]$$

其中 $z_0 = \beta \cdot \mathbb{E}_{x', y'}[\text{KL}(\pi_\theta(\cdot|x') \| \pi_{\text{ref}}(\cdot|x'))]$ 作为 reference baseline，$\lambda_y$ 区分"desirable / undesirable" 样本（损失敏感系数不同）。直观：好样本要 $\hat{r}$ 高于 baseline，坏样本要低于 baseline。**好处是数据收集成本极低**——直接从产品日志的赞踩信号训。

**SimPO（Meng et al. 2024）**

DPO 必须装 reference model 推理 $\pi_{\text{ref}}$（显存 2× / forward 2×）。SimPO 干脆**去掉 reference**，用 length-normalized average log probability 当 reward：

$$\hat{r}_{\text{SimPO}}(x, y) = \frac{\beta}{|y|} \log \pi_\theta(y|x) = \frac{\beta}{|y|} \sum_{t=1}^{|y|} \log \pi_\theta(y_t | x, y_{<t})$$

loss 形式仍是 $-\log \sigma(\hat{r}(x, y_w) - \hat{r}(x, y_l) - \gamma)$，其中 $\gamma$ 是额外的 target reward margin（$\hat{r}_w - \hat{r}_l$ 至少要大过 $\gamma$ 才停学）。

两个改动：(1) **去 ref**——单模型训练，省一半显存 & 一次 forward；(2) **length norm**——把每步平均 log-prob 当 reward，结构上抑制 length hacking。Meng et al. 实证 SimPO 在多个 benchmark 比 DPO 涨 1-3 个点。代价是没有 reference 这个"锚"，需要 SFT base 已经较强，否则 policy 可能漂坏（见 §4 踩坑）。

**ORPO（Odds Ratio Preference Optimization, Hong et al. 2024）**

主流 alignment pipeline 是 SFT → DPO 两阶段。ORPO 的卖点是"一阶段干完"：在 SFT loss 之外加一项 odds ratio penalty，把 chosen 推上去、rejected 拉下来，**不需要 reference model 也不需要 SFT pre-stage**：

$$\mathcal{L}_{\text{ORPO}} = \mathcal{L}_{\text{SFT}}(y_w) + \lambda \cdot \mathcal{L}_{\text{OR}}$$

其中 $\mathcal{L}_{\text{OR}} = -\log \sigma\bigl(\log \tfrac{\text{odds}_\theta(y_w|x)}{\text{odds}_\theta(y_l|x)}\bigr)$，odds $:= p / (1-p)$ 用 chosen / rejected 的序列概率算。$\lambda$ 是 odds ratio loss 的权重。适合从 base model 直接做 alignment，不想训 SFT pre-stage 的场景（如 small data 实验、快速 prototype）。

**RLOO（REINFORCE Leave-One-Out, Ahmadian et al. 2024）**

前面四个都是 offline preference optimization。RLOO 走另一条路——回到 on-policy RL，但简化 PPO：扔掉 critic，用 batch 内"其它样本的平均 reward"作 baseline。

对每个 prompt $x$ 采样 $K$ 个 response $y_1, \dots, y_K$（用一个 RM 打分），第 $i$ 个的 advantage：

$$A_i = R(x, y_i) - \frac{1}{K-1} \sum_{j \neq i} R(x, y_j)$$

REINFORCE 梯度：$\nabla \mathcal{L} = -\mathbb{E}\bigl[A_i \cdot \nabla \log \pi_\theta(y_i | x)\bigr]$，再加 KL penalty。**和 GRPO 心智模型几乎一致**（都用 group 内 baseline 估 advantage、都不要 critic），但 RLOO 用 leave-one-out 而 GRPO 用 mean-and-std 标准化（详见 9.5）。Ahmadian et al. 实证 RLOO 在 LLM 上往往打平甚至超过 PPO，且代码简洁很多。

### 2.6 变体对比表

| 方法 | online/offline | 需要 ref model | 数据格式 | 关键超参 | 核心改进 |
|------|---------------|---------------|---------|---------|---------|
| **DPO** | offline | 是 | pairwise | $\beta$ | 把 RM 隐含进 loss，supervised 形式 |
| **IPO** | offline | 是 | pairwise | $\tau$ | log-sigmoid → squared，防 deterministic preference 过拟合 |
| **KTO** | offline | 是 | unary（good / bad）| $\beta, \lambda_+, \lambda_-$ | 单边 label，prospect-theory 损失敏感 |
| **SimPO** | offline | **否** | pairwise | $\beta, \gamma$ | 去 ref + length-normalized reward |
| **ORPO** | offline | **否** | pairwise（含 SFT label）| $\lambda$ | SFT + preference 一阶段，无需 SFT pre-stage |
| **RLOO** | online | 是（含 RM）| 在线 generate + RM | $K, \beta_{KL}$ | leave-one-out baseline 替代 critic |

---

## 3. 最小代码示例

### 3.1 DPO loss 完整实现（手撕 PyTorch）

```python
import torch
import torch.nn.functional as F

def get_seq_logprob(model, input_ids, labels, attn_mask):
    """计算 sequence-level log-probability: sum_t log π(y_t | x, y_<t)
    labels: 与 input_ids 对齐，prompt 部分用 -100 mask 掉（不计入 logp）。"""
    logits = model(input_ids, attention_mask=attn_mask).logits   # (B, T, V)
    # shift: 用 t 时刻的 logits 预测 t+1 时刻的 token
    shift_logits = logits[:, :-1, :].contiguous()                # (B, T-1, V)
    shift_labels = labels[:, 1:].contiguous()                    # (B, T-1)
    logp = F.log_softmax(shift_logits, dim=-1)                   # (B, T-1, V)
    loss_mask = (shift_labels != -100).float()                   # (B, T-1)
    safe_labels = shift_labels.masked_fill(shift_labels == -100, 0)
    per_tok_logp = torch.gather(logp, -1, safe_labels.unsqueeze(-1)).squeeze(-1)  # (B, T-1)
    return (per_tok_logp * loss_mask).sum(dim=-1)                # (B,) sequence logp


def dpo_loss(model, ref_model, batch, beta=0.1):
    # 一次 batch 含 chosen / rejected 两份输入
    chosen_logp     = get_seq_logprob(model,     batch["chosen_ids"],     batch["chosen_labels"],     batch["chosen_mask"])
    rejected_logp   = get_seq_logprob(model,     batch["rejected_ids"],   batch["rejected_labels"],   batch["rejected_mask"])
    with torch.no_grad():    # ref model 不更新，省显存
        chosen_ref_logp   = get_seq_logprob(ref_model, batch["chosen_ids"],   batch["chosen_labels"],   batch["chosen_mask"])
        rejected_ref_logp = get_seq_logprob(ref_model, batch["rejected_ids"], batch["rejected_labels"], batch["rejected_mask"])

    # implicit reward 的差 = β · (logπ - logπ_ref)_w − β · (logπ - logπ_ref)_l
    chosen_reward   = beta * (chosen_logp   - chosen_ref_logp)
    rejected_reward = beta * (rejected_logp - rejected_ref_logp)
    loss = -F.logsigmoid(chosen_reward - rejected_reward).mean()

    # 监控量：implicit reward margin / accuracy
    margin = (chosen_reward - rejected_reward).mean().item()
    acc    = (chosen_reward > rejected_reward).float().mean().item()
    return loss, {"margin": margin, "acc": acc}
```

关键细节：
- **`labels` 对 prompt 部分必须 mask 成 -100**——只算 response token 的 log-prob，不然 prompt token 也进 reward 估计，loss 完全错；
- **ref model `torch.no_grad()`**——节约 activation 显存（ref model 只 forward）；工业实现里 ref model 通常用 fp16 / 共享 base 权重 + 不同 LoRA adapter；
- **`F.logsigmoid` 比 `log(sigmoid(x))` 数值稳定**（同 9.2 RM）；
- 监控 `margin` 与 `acc`：margin 应稳步增长（chosen 比 rejected 的 implicit reward 差越拉越大），acc 应 > 50% 并爬升到 70-80%。

### 3.2 SimPO loss 实现（去 ref + length norm）

```python
import torch
import torch.nn.functional as F

def simpo_loss(model, batch, beta=2.0, gamma=1.0):
    """SimPO: 不需要 ref model；reward = (β / |y|) · sum_t log π(y_t|x, y_<t)"""
    chosen_logp   = get_seq_logprob(model, batch["chosen_ids"],   batch["chosen_labels"],   batch["chosen_mask"])
    rejected_logp = get_seq_logprob(model, batch["rejected_ids"], batch["rejected_labels"], batch["rejected_mask"])

    # length-normalize: 用 response token 数量
    chosen_len   = (batch["chosen_labels"][:, 1:]   != -100).sum(dim=-1).clamp(min=1)
    rejected_len = (batch["rejected_labels"][:, 1:] != -100).sum(dim=-1).clamp(min=1)

    chosen_reward   = beta * chosen_logp   / chosen_len
    rejected_reward = beta * rejected_logp / rejected_len

    # 比 DPO 多一个 target margin γ：要求差距至少 γ
    loss = -F.logsigmoid(chosen_reward - rejected_reward - gamma).mean()
    return loss
```

对比 DPO 的两个区别一目了然：(1) 没有 ref_model 调用；(2) `/ chosen_len` 做 length normalization。$\beta$ 在 SimPO 里通常**比 DPO 大一个量级**（DPO 用 0.1，SimPO 论文用 2.0~2.5），因为去掉 ref 后的 reward 数值范围本身小。$\gamma$ 是 target margin，实践 0.5-1.5。

### 3.3 TRL DPOTrainer 完整配置（生产推荐）

```python
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer
from peft import LoraConfig

base = "meta-llama/Llama-3.1-8B-Instruct"   # 必须用 SFT 后的模型，不能 base
tok  = AutoTokenizer.from_pretrained(base)
tok.pad_token = tok.eos_token

policy = AutoModelForCausalLM.from_pretrained(base, torch_dtype="bfloat16")
ref    = AutoModelForCausalLM.from_pretrained(base, torch_dtype="bfloat16")
ref.requires_grad_(False)

ds = load_dataset("trl-lib/ultrafeedback_binarized", split="train")   # {prompt, chosen, rejected}

cfg = DPOConfig(
    output_dir="./dpo-llama3-8b",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-7,        # DPO lr 通常更小，5e-7 ~ 1e-6
    num_train_epochs=1,        # 1-3 epoch；超过容易过拟
    bf16=True,
    max_length=2048,
    max_prompt_length=1024,
    beta=0.1,                  # 经典默认
    loss_type="sigmoid",       # "sigmoid"=DPO, "ipo"=IPO, "kto_pair"=KTO 风格 等
    report_to="wandb",
)

peft_cfg = LoraConfig(r=64, lora_alpha=128, target_modules="all-linear")

trainer = DPOTrainer(
    model=policy, ref_model=ref, args=cfg, tokenizer=tok,
    train_dataset=ds, peft_config=peft_cfg,
)
trainer.train()
```

要点：
- **`learning_rate=5e-7`**——DPO 的 lr 比 SFT 还低 1-2 个数量级。SFT 常用 2e-5，DPO 工业实操 5e-7 ~ 1e-6（全参）/ 1e-5 ~ 5e-5（LoRA）；
- **`loss_type`** 在 TRL 里是切换变体的开关：`"sigmoid"`（DPO）/ `"ipo"`（IPO）/ `"kto_pair"`（KTO 的 pair 版本）/ `"hinge"` 等；要 SimPO 用 `CPOTrainer` 或 [SimPO 官方仓库](https://github.com/princeton-nlp/SimPO)；
- **`ref_model=None`** 时 TRL 自动用 LoRA disable adapter 的 base 当 ref（省一半显存）——LoRA 训 DPO 必用此 trick；
- **数据格式**：HF datasets 必须有 `prompt` / `chosen` / `rejected` 三列；TRL 自动套 chat template。

### 3.4 Iterative DPO 伪代码

DPO 是 offline 的根本痛点是 policy 探索不到训练 pair 之外的 response。**Iterative DPO** 用当前 policy 不断 generate 新 pair，逐步把 DPO 推向 online——LLaMA-3 / Tülu 3 / SPPO 都用了类似做法。

```python
policy = load_sft_model()
ref    = load_sft_model()    # 永远是 SFT model，每轮不变
rm     = load_reward_model() # 用 RM 评判新 pair
prompts = load_unlabeled_prompts(N=50_000)

for round in range(NUM_ROUNDS):     # 通常 2-4 轮
    new_pairs = []
    for x in prompts:
        # 1) 当前 policy 采 K 个 response（带 diversity）
        responses = policy.generate(x, K=8, temperature=0.9, top_p=0.95)
        # 2) RM 打分，取最高 / 最低当 chosen / rejected
        rewards = [rm.score(x, y) for y in responses]
        best, worst = argmax(rewards), argmin(rewards)
        if rewards[best] - rewards[worst] > MARGIN_THRESHOLD:    # 过滤无差异 pair
            new_pairs.append({"prompt": x, "chosen": responses[best], "rejected": responses[worst]})

    # 3) 用新 pair + 旧 pair（可选）训一轮 DPO，ref 仍是最初 SFT model
    policy = dpo_train(policy, ref, dataset=new_pairs, beta=0.1, epochs=1)
```

注意三个工程细节：(1) **ref 永远是最初的 SFT model**，不能换成上一轮的 policy（会造成 policy 离最初 anchor 越来越远，不稳定）；(2) **过滤低 margin pair**——RM 给 chosen / rejected 分差不到阈值（如 1.0）的 pair 丢掉，避免噪声；(3) **diversity sampling**——`temperature ≥ 0.9` + `top_p` 保证候选有差异，否则 pair 信息量不足。

---

## 4. 工程踩坑与经验

- ❗ **DPO 的 ref model 必须严格等于 policy 的 SFT base**——不一致 → policy 在 ratio 项 $\log(\pi/\pi_{\text{ref}})$ 上一开始就有大偏差，loss 错把"模型差异"当 reward 信号，训完模型崩。LoRA 训时 ref 用"disable adapter 的 base"是最省内存的标准做法。
- ❗ **DPO $\beta$ 调小（< 0.05）→ implicit reward 过拟合**：policy 把 chosen 概率推到 ~1、rejected 推到 ~0，对 OOD prompt 输出严重 collapse 到训练数据风格，常见症状是模板化、重复、拒答率飙升。把 $\beta$ 拉到 0.1-0.3 通常能救回。
- ❗ **Length hacking**：DPO chosen / rejected 长度差异大时（chosen 平均长 30%+），model 会学到"长 = chosen"——训完输出冗长啰嗦。**SimPO 的 length normalization 是最干净的解**；DPO 党的对策是数据预处理时按长度分桶 / 主动构造短 chosen vs 长 rejected 反向 pair。
- ❗ **Pairwise data 质量是 DPO 的天花板**——质量差的偏好对（标注员盲选 / RM 自动打分质量低）训出来 model 也差。建议优先用 UltraFeedback / HelpSteer2 / Skywork-Reward-Preference-80K 这些经过验证的数据集，少自己拼脏数据。
- ❗ **KTO 的 $\beta$ 与 DPO 不直接可比**——KTO 的 reward 计算引入了 KL baseline $z_0$，与 DPO 的 implicit reward 数值范围不同，**换 KTO 后 $\beta$ 必须重新 sweep**。论文实践常见 $\beta \in [0.1, 1.0]$，与 DPO 的 [0.05, 0.5] 错位。
- ❗ **SimPO 没 ref model → 更依赖强 SFT base**：reference model 是 DPO 的"锚"，SimPO 没了这个锚，policy 在错的方向上漂没有约束。如果 SFT base 本身只跑了几千条数据 / 弱 instruction following，SimPO 直接训出来的模型可能 instruction 都不会 follow。**经验上 SimPO 在已经 well-tuned 的 SFT model（如 Llama-3-Instruct）上才显出优势**。
- ❗ **DPO 训练时 chosen / rejected 同 batch 双 forward → 显存约 2× SFT**——很多人只算 policy 一次 forward 的显存，忘了 ref model 也要双 forward（虽然 no_grad 但 hidden state 仍要存）。8B 模型全参 DPO 在 80G H100 上 batch_size 通常 1-2，要配 gradient_accumulation。LoRA + 8bit ref 是单卡跑 DPO 的标配。
- ❗ **Iterative DPO 容易"自激"**——policy 学自己生成的偏好，错的偏好被反复强化，最后输出彻底偏离人类偏好。三道防线：(1) RM 必须高质量（不然 self-reward 失效）；(2) generation 用高 temperature + top_p 保 diversity；(3) 每轮控制更新幅度（lr 衰减 / KL 监控 / 强制 keep 原始 SFT pair 在数据里）。
- ❗ **DPO/SimPO 的 evals 不能只看 training reward / margin**：implicit reward 上升 ≠ 人类偏好上升（[Park 2024] 等多篇 paper 论证 DPO reward 与 Arena-Hard / MT-Bench / AlpacaEval-LC 评分相关性弱）。**真实评测必须用人类对齐 benchmark：Arena-Hard、MT-Bench、AlpacaEval-LC、IFEval**，至少跑两个再说"DPO 训得好"。
- ❗ **TRL `DPOTrainer` 的 `loss_type="ipo"` 时 `beta` 字段是 IPO 的 $\tau$**——别看到字段叫 beta 就用 DPO 的 0.1，IPO 的 $\tau$ 一般 0.01-0.1（与 DPO 反比）。改 loss_type 必须重新 sweep 关键超参。

---

## 5. 经典 paper

- **Rafailov et al., 2023 — Direct Preference Optimization: Your Language Model is Secretly a Reward Model** — DPO 原论文。本节 §2.1 推导每一步在它的 §4 都有详尽对应；§5 的实证部分论证 DPO 与 PPO 在 IMDB sentiment / TL;DR summarization / Anthropic HH 上效果相当但训练复杂度低一个量级。**算法岗必精读**。
- **Azar et al., 2023 — A General Theoretical Paradigm to Understand Learning from Human Preferences (IPO)** — 把 RLHF / DPO 放进统一 $\Psi$PO framework，指出 DPO 在 deterministic preference 下的过拟合 failure mode 并给出 IPO 解。读 §3-5 能从根本上理解"为什么 DPO 不是终点"。
- **Ethayarajh et al., 2024 — KTO: Model Alignment as Prospect Theoretic Optimization** — KTO 原论文。把 Kahneman-Tversky prospect theory 套到 alignment 上，论证 unary feedback 也够用——对生产场景（用户赞踩日志）特别有价值。
- **Meng et al., 2024 — SimPO: Simple Preference Optimization with a Reference-Free Reward** — SimPO 原论文，工程性极强：一个 length-normalized reward + target margin 就稳定打过 DPO 1-3 个点。Princeton NLP 的实现非常清晰，**强烈建议跟 repo 跑一遍**。
- **Hong et al., 2024 — ORPO: Monolithic Preference Optimization without Reference Model** — ORPO 原论文。用 odds ratio 把 SFT 和 preference 一阶段结合，对小数据 / 快 prototype 场景实用。
- **Ahmadian et al., 2024 — Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs (RLOO)** — Cohere 的反思之作，论证 PPO 的复杂度对 LLM RLHF 是 overkill，去 critic 的 leave-one-out REINFORCE 在 LLM 上简洁且常更优。是理解 GRPO（9.5）的最佳前序。
- **Tang et al., 2024 — Generalized Preference Optimization: A Unified Approach to Offline Alignment** — 把 DPO / IPO / SLiC 等用一个广义 loss family 统一。读完能在脑子里建立"所有 offline preference loss 都是 $f(\Delta)$ 的某个选择"的统一图景。

---

## 6. 自测与面试题

**Q1（推导）**：写出从 RLHF 优化目标 $\max_\pi \mathbb{E}[r] - \beta D_{KL}(\pi \| \pi_{\text{ref}})$ 推到 DPO loss 的关键 4 步，特别说明 $\beta \log Z(x)$ 为什么能消掉。

<details>
<summary>Answer sketch</summary>

- **Step 1**：写出 KL-约束的 RLHF 目标 $\max_\pi \mathbb{E}_{x, y \sim \pi}[r(x, y)] - \beta D_{KL}(\pi \| \pi_{\text{ref}})$。
- **Step 2 — 闭式最优解**：固定 $x$ 看作分布优化，加 $\sum \pi = 1$ 约束用 Lagrange，得 $\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp(r(x, y) / \beta)$，其中 $Z(x) = \sum_y \pi_{\text{ref}}(y|x) \exp(r(x, y)/\beta)$。本质是 max-entropy regularized RL 的标准结论。
- **Step 3 — 反演 reward**：对 Step 2 闭式解取 log 解出 $r(x, y) = \beta \log (\pi^*(y|x) / \pi_{\text{ref}}(y|x)) + \beta \log Z(x)$。这是 DPO 的核心 trick——reward 完全用 (policy, ref) 的 log-ratio 表达。
- **Step 4 — 代入 BT 抵消 $Z$**：BT preference $P(y_w \succ y_l) = \sigma(r(x, y_w) - r(x, y_l))$；把 Step 3 reward 代入差，**$\beta \log Z(x)$ 只依赖 $x$ 不依赖 $y$，在 $r(x, y_w) - r(x, y_l)$ 中两项完全相同直接相减为 0**。这一抵消让 $Z(x)$（需要对整个 response 空间求和、根本算不动）从最终 loss 里彻底消失。把 $\pi^*$ 替换为参数化 $\pi_\theta$、对偏好数据取 NLL，得 $\mathcal{L}_{\text{DPO}} = -\mathbb{E}[\log \sigma(\beta \log \pi_\theta(y_w|x) / \pi_{\text{ref}}(y_w|x) - \beta \log \pi_\theta(y_l|x) / \pi_{\text{ref}}(y_l|x))]$。
- **加分**：强调 $\beta \log Z(x)$ 抵消是 DPO "可落地" 的根本原因——否则要估 $Z$ 就得 sampling，DPO 也就和 PPO 没区别。

</details>

**Q2（变体）**：DPO 与 SimPO 的主要差异是什么？SimPO 为什么能去掉 reference model？请同时说明这个去 ref 的代价。

<details>
<summary>Answer sketch</summary>

- **形式上的两点差异**：
  1. SimPO 不需要 ref model：implicit reward 直接定义为 $\hat{r}(x, y) = \frac{\beta}{|y|} \log \pi_\theta(y|x)$，去掉了 DPO 的 $\log(\pi_\theta / \pi_{\text{ref}})$ 形式中的 ref 项；
  2. SimPO 显式 length-normalize（除以 $|y|$），并加了 target margin $\gamma$。
- **为什么能去 ref**：DPO 用 ref 是因为它的 implicit reward 来自 KL-约束 RL 的闭式解，必然带 $\pi_{\text{ref}}$；SimPO **放弃了"reward 来自 RLHF 闭式解"这个数学根据**，直接定义 reward = average log-prob，loss 形式仍是 BT pairwise，但不再是 RLHF 的等价物。所以"去 ref"是工程简化、不是数学等价——SimPO 是一个新算法，不是 DPO 的简化版。
- **去 ref 的代价**：
  1. 失去 anchor——ref model 在 DPO 里像 KL 约束的"锚"，约束 policy 不要漂太远；SimPO 没了这个锚，**对 SFT base 质量要求更高**（base 弱时容易漂坏）；
  2. 失去理论保证——DPO 训练目标可证明等价于 RLHF 最优解（在 BT 假设下）；SimPO 只能 empirically 论证 work；
  3. 一些 OOD prompt 上鲁棒性可能不如 DPO（无 ref 提供的"回到熟悉分布"的信号）。
- **优势**：(1) 显存省一半（一个 model 而非两个）；(2) forward 减少一半；(3) length-normalize 缓解 length hacking；(4) 实证多数 benchmark 涨 1-3 点。
- **加分**：SimPO 的 $\beta$ 与 DPO 不可直接比较（DPO 0.1 ↔ SimPO 2.0 量级），换算法必须重新 sweep。

</details>

**Q3（实战选型）**：你手头有 10 万条 pairwise 偏好数据 + 1 张 24GB 显存的卡（如 4090 / A10），目标是把一个 7B Instruct 模型对齐得更好。请分别列出用 DPO / SimPO / KTO 三个方案的具体配置（模型、训练方式、关键超参 lr / β / batch / epoch），并说明你最终会选哪个、为什么。

<details>
<summary>Answer sketch</summary>

**显存估计先做**：7B 模型 bf16 全参 ≈ 14GB 仅 weights；DPO 还要装 ref（再 14GB），总共远超 24GB——必须用 LoRA + ref 共享 base。

- **DPO 配置**：
  - 模型：Llama-3.1-8B-Instruct + LoRA（r=32~64, alpha=64~128, target_modules=all-linear）
  - ref：用 LoRA disable adapter 的 base 做 ref（TRL `DPOTrainer` ref_model=None 自动）
  - 关键超参：lr=5e-6（LoRA 比全参高一档）, β=0.1, per_device_batch=1, grad_accum=16~32（有效 batch 16-32）, epoch=1
  - max_length=2048, bf16, 加 gradient checkpointing
  - 监控：implicit reward margin、pairwise acc、Arena-Hard / AlpacaEval-LC 离线 eval

- **SimPO 配置**：
  - 模型：Llama-3.1-8B-Instruct + LoRA（同上 r/alpha）
  - **不要 ref model**——显存直接省一半，可以把 batch 加大或 LoRA r 调高
  - 关键超参：lr=1e-5（LoRA SimPO 实测可比 DPO 略高）, β=2.0~2.5（注意与 DPO 不可比！）, γ=0.5~1.5, per_device_batch=2~4, grad_accum=8~16, epoch=1
  - 适合用 Princeton NLP 官方 repo（`SimPOTrainer`）或 TRL `CPOTrainer` + simpo loss

- **KTO 配置**：
  - 前提：你需要把 pairwise 数据展开成 unary（chosen → desirable, rejected → undesirable），10 万 pair → 20 万 unary 样本
  - 模型：Llama-3.1-8B-Instruct + LoRA（同上）
  - 关键超参：lr=5e-6, β=0.5（与 DPO 不可比，KTO 论文常用 [0.1, 1.0]）, λ_+ ≈ 1.0, λ_- ≈ 1.0（如果 desirable / undesirable 数量不等要调）, per_device_batch=2, grad_accum=8, epoch=1
  - 注意：KTO 的强项是真正只有 unary label 的场景（如 thumbs up/down 日志），如果你已经有 pairwise，KTO 通常不如 DPO/SimPO

- **最终选型推荐**：
  - **首选 SimPO**——24GB 单卡场景显存最关键，SimPO 去掉 ref 后能用更大 batch / 更高 LoRA r，训练速度快 1.5-2×；只要 base 是 well-tuned 的 Instruct model（条件满足），实证效果普遍更好。
  - **保守稳健选 DPO**——理论清晰、TRL 工具链最成熟、社区经验最多，遇到问题最容易 debug；如果 base 是自己 SFT 的可能弱，DPO 的 ref anchor 更安全。
  - **不选 KTO**——已经有 pairwise 数据时 KTO 没有优势，把 pairwise 展开成 unary 反而丢了 chosen/rejected 之间的相对信息。
  - **加分讨论**：可以两个都跑（资源允许）然后用 LLM-as-Judge / Arena-Hard 离线对比；或者跑 SimPO 拿到 baseline 后再做一轮 iterative DPO（用当前 model + RM 生成新 pair）进一步推。

</details>

---

## 7. 延伸阅读

- [TRL DPOTrainer 官方文档](https://huggingface.co/docs/trl/main/en/dpo_trainer) — §3.3 配置的完整 API、所有 `loss_type`（DPO/IPO/KTO 等）的官方对照表，是工业落地最权威的入口。
- [Princeton NLP / SimPO 仓库](https://github.com/princeton-nlp/SimPO) — SimPO 官方实现，包含完整训练 / eval 脚本、Llama-3 / Mistral / Gemma 多 base 的配方，跑一遍就能拿到论文复现结果。
- [Hugging Face Alignment Handbook](https://github.com/huggingface/alignment-handbook) — Zephyr 训练流程的官方 cookbook，DPO 完整 pipeline 的事实参考实现，从数据 → SFT → DPO → eval 一条龙。
- [Lambert 2024 — RLHF Book Ch.7 "Direct Alignment Algorithms"](https://rlhfbook.com/c/12-direct-alignment.html) — Nathan Lambert 写的开源教材里讲 DPO 与变体的章节，对推导和 design choice 的讨论比原论文更教学化。
- [Tülu 3 技术报告](https://arxiv.org/abs/2411.15124) — AllenAI 2024 年的开源对齐 SOTA pipeline：SFT → DPO → RLVR (PPO)，详细记录每段的数据、超参、消融，是"现代 alignment pipeline"的实操圣经。
- 推荐继续读本教程的 **9.5 节《GRPO：去 critic 的 group-relative advantage》** —— RLOO 的"群体内 baseline"思想在 GRPO 里发扬光大，DeepSeek-R1 把它推到了 SOTA；以及 **9.6 节《工程踩坑：reward hacking / RM 漂移 / KL 坍塌 / length bias》** —— DPO/SimPO 的 length hacking、margin 过拟合等问题在那里集中处理。
