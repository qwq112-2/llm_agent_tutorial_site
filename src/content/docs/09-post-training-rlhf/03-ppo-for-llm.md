---
title: "9.3 PPO 原理与在 LLM 上的形式（KL 约束、4 模型显存）"
description: "PPO 是 InstructGPT 以来的 RLHF 事实标准——把 9.1 的 actor-critic 公式加上 importance ratio + clip + 对 ref model 的 KL 约束，得到一个稳但工程上\"4 模型同时进显存\"贵到爆的算法；本节把数学、LLM 特化形式、4 模型架构、显存账、超参、TRL 落地一次讲完，是 9.5 GRPO（PPO 的简化）和 Module "
---

> ⏱ 预计阅读 60 分钟 ｜ 难度 ★★★ 🔥 ｜ 前置：9.1 RL 速通、9.2 Reward Model

## 一句话本节讲什么

PPO 是 InstructGPT 以来的 RLHF 事实标准——把 9.1 的 actor-critic 公式加上 **importance ratio + clip + 对 ref model 的 KL 约束**，得到一个稳但工程上"4 模型同时进显存"贵到爆的算法；本节把数学、LLM 特化形式、4 模型架构、显存账、超参、TRL 落地一次讲完，是 9.5 GRPO（PPO 的简化）和 Module 15 Agent RL 的基石。

---

## 1. Mental model（直觉）

PPO 一句话：**用 9.1 那条 actor-critic 梯度更新 LLM，但每一步都拽住缰绳——既不让单个 token 的概率比上一轮更新前漂得太远（ratio clip），也不让整个 policy 漂离原 SFT 模型太远（KL to ref）。**

为什么需要"拽两次缰绳"？因为 LLM RL 同时面临两种"漂"：

1. **训练步内的漂**——同一批 rollout 跑 4 个 epoch 复用，policy 每个 mini-batch 都在变；如果某个 token 的 $\pi_\theta(a|s) / \pi_{\theta_{\text{old}}}(a|s)$ 暴涨到 5 倍，importance sampling 的方差会爆，梯度噪声直接毁掉训练。**ratio clip 解决这个**。
2. **训练全过程的漂**——RM 是个不完美的 proxy，policy 越跑越远会发现 RM 的"漏洞"（比如学会输出马屁话拿高分），表面 reward 一路上升，实际能力急剧下降。这就是 reward hacking。**KL to ref model 解决这个**——拽住 policy 不让它离原 SFT 模型太远，相当于"你可以小步学，但不许变成另一个人"。

把 9.1 + 9.2 + 这两条约束串起来，PPO 在 LLM 上的运行图是这样：

```
              ┌───────────────────┐
              │ prompt batch      │
              └────────┬──────────┘
                       │
        ┌──────────────▼────────────────┐
        │  policy π_θ_old.generate(...) │ ← rollout (no_grad)
        └──────────────┬────────────────┘
                       │ (prompt, response)
          ┌────────────┼─────────────┬────────────────┐
          ▼            ▼             ▼                ▼
      ┌───────┐  ┌──────────┐   ┌───────┐      ┌────────────┐
      │  RM   │  │ ref_model│   │ critic│      │  policy    │
      │frozen │  │  frozen  │   │ train │      │   train    │
      └───┬───┘  └────┬─────┘   └───┬───┘      └─────┬──────┘
          │           │             │                │
       r_RM       log π_ref      V(s_t)         log π_θ
          │           │             │                │
          └────┬──────┴───┬─────────┘                │
               ▼          ▼                          │
        per-token r_t = r_RM·1[t=T]            ratio ρ_t = π_θ/π_θ_old
                  −β·log(π_θ/π_ref)                  │
                       │                              │
                       ▼            ┌────────────────┘
                ┌──────────────┐    │
                │  GAE → Â_t   │    │
                └──────┬───────┘    │
                       └────────┬───┘
                                ▼
                  L = −min(ρ·Â, clip(ρ,1±ε)·Â) + c₁·(V−V_target)²
                                │
                                ▼
                          backward → step θ
```

记三件事就抓住了 PPO 的全部：

- **ratio + clip** 是 PPO 区别于 vanilla policy gradient 的标志（控更新步长）
- **KL to ref** 是 PPO 在 LLM 上区别于经典 PPO 的标志（控 reward hacking）
- **4 模型** 是 PPO 在 LLM 上的工程代价：policy / ref / RM / critic 同时驻留显存

剩下都是细节。

---

## 2. 公式与原理

### 2.1 从 vanilla PG 到 TRPO 到 PPO 的演化

回忆 9.1 的 actor-critic 梯度：

$$\nabla_\theta J(\theta) = \mathbb{E}_{(s, a) \sim \pi_\theta}\left[ \nabla_\theta \log \pi_\theta(a \mid s) \cdot \hat A^\pi(s, a) \right]$$

直接用它跑 SGD 有两个工程难点。

**难点 1：不能复用 trajectory**。这条梯度是 on-policy 的——一旦 $\theta$ 变了，旧 trajectory 就理论上作废。LLM 上每条 rollout 几百个 token、调用一次 generate 几秒到几十秒，**用一次就丢太奢侈**。

**难点 2：步长无控制**。vanilla SGD 一不小心就把某个 token 的概率推得过激（比如本来 $\pi(a|s) = 0.01$ 一步推到 $0.5$），导致 policy 直接跳出"它原本擅长"的分布。后续采样质量崩、梯度估计崩，连锁反应到不可恢复。

**TRPO**（Schulman 2015）的方案是显式约束 KL：

$$\max_\theta \; \mathbb{E}\left[ \frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)} \hat A \right] \quad \text{s.t.} \quad \mathbb{E}\bigl[ D_{\text{KL}}(\pi_{\theta_{\text{old}}} \| \pi_\theta) \bigr] \le \delta$$

理论上漂亮：把 importance ratio 放进目标，允许复用 $\theta_{\text{old}}$ 采的 trajectory；硬 KL 约束限制每步漂移。但实现需要**Fisher 信息矩阵的 conjugate gradient**——在 LLM 上参数量太大算不动。

**PPO**（Schulman 2017）的妙手：用 **ratio clip** 替代显式 KL 约束。一行 loss 实现"软 trust region"，不需要 Fisher，普通 SGD/Adam 就能跑。

### 2.2 PPO Clipped Objective

定义 importance ratio（对每个 token）：

$$\rho_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$$

在 LLM 里 $a_t$ 是第 $t$ 个 token，$s_t$ 是 prompt + 前 $t-1$ 个 token。**$\pi_{\theta_{\text{old}}}$ 是采这条 rollout 时的 policy 快照**——一个 PPO 大循环开始时，$\theta_{\text{old}} \leftarrow \theta$，然后这一批 rollout 在 K 个 epoch 内复用。

PPO 的 clipped surrogate objective：

$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t\bigl[ \min(\rho_t \hat A_t, \; \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon) \hat A_t) \bigr]$$

其中 $\epsilon$ 通常取 $0.2$（经典）或 $0.1$（更保守）。**逐项拆解**这个目标——它的"动机比公式更重要"：

- 当 $\hat A_t > 0$（这步好，应该提概率）：
  - 未 clip 项 $\rho_t \hat A_t$ 想把 $\rho_t$ 推得越大越好
  - clip 项把 $\rho_t$ 卡在 $1 + \epsilon$ 上限——一旦 $\rho_t > 1 + \epsilon$，$\text{clip}(\rho_t, ...) = 1 + \epsilon$ 不再增大
  - **取 min** 选两者中较小的——所以 $\rho_t > 1 + \epsilon$ 时 loss 不再上升，**梯度归零**——policy 不再被推得更激进
- 当 $\hat A_t < 0$（这步差，应该降概率）：
  - 未 clip 项 $\rho_t \hat A_t$ 想把 $\rho_t$ 推得越小越好（负 advantage 乘小 ratio = 大正数 / 小负数）
  - clip 项把 $\rho_t$ 卡在 $1 - \epsilon$ 下限
  - **取 min**（注意 $\hat A_t < 0$ 让"较小"颠倒）——$\rho_t < 1 - \epsilon$ 时 loss 不再下降，梯度归零

一句话：**clip + min 两件事联合作用，让 ratio 偏离 1 太远时这一项就不再贡献梯度**——既不奖励"激进的好动作"，也不严惩"过激的坏动作"。effective policy update step 被软性限制在一个信任域里。

注意 $L^{\text{CLIP}}$ 是个**lower bound**（pessimistic surrogate）——它放弃了"在信任域内尽可能多更新"的机会，换来"绝不在信任域外更新"的安全。这是 Schulman 在 [PPO 2017] §3 的核心观点。

### 2.3 PPO 完整 loss

工程实现里 PPO loss 通常包含三项：

$$L^{\text{PPO}}(\theta, \phi) = -L^{\text{CLIP}}(\theta) + c_1 L^{\text{VF}}(\phi) - c_2 \mathcal{H}[\pi_\theta]$$

（PyTorch 是 minimize，所以 $L^{\text{CLIP}}$ 前加负号。）

- **$L^{\text{VF}}$（value function loss）**：训 critic $V_\phi(s)$ 拟合 value target

$$L^{\text{VF}}(\phi) = \mathbb{E}_t\bigl[(V_\phi(s_t) - \hat V_t^{\text{target}})^2\bigr], \quad \hat V_t^{\text{target}} = \hat A_t + V_{\phi_{\text{old}}}(s_t)$$

critic 估得越准，GAE 算出的 advantage 方差越小，policy update 越稳。$c_1$ 通常取 0.5 或 1.0。

- **$\mathcal{H}[\pi_\theta]$（entropy bonus）**：鼓励 policy 保持探索性

$$\mathcal{H}[\pi_\theta(\cdot|s_t)] = -\sum_a \pi_\theta(a|s_t) \log \pi_\theta(a|s_t)$$

经典 PPO（Atari/MuJoCo）里 $c_2 \in [0.001, 0.01]$ 防 policy 过早确定性；**LLM RLHF 里 $c_2$ 常直接设 0**——因为 LLM 自带很大的 entropy，且过强 entropy bonus 会让模型胡乱采样。

### 2.4 PPO 在 LLM 上的特化：KL to ref model

经典 PPO 到这里就完了，但 LLM RLHF 还多一项 **KL penalty to reference model**——这是 InstructGPT [Ouyang 2022] 的关键工程化。

为什么需要它？因为 RM 是有限数据训出的不完美 proxy。如果让 PPO 自由优化 RM 分数，policy 会逐渐找到 RM 的"盲点"（比如固定的开场白模板、特定 emoji、过长的废话回复），表面 reward 暴涨，实际能力崩——这就是 **reward hacking**。把 policy 拽住不离 SFT 太远，能极大降低 hacking 风险。

实现方式不是把 KL 加到 loss 里（那是早期一种实现），主流是 **per-token reward shaping**：把 token-level KL penalty 直接作为负 reward 加到每个 token 上。

$$r_t = -\beta \cdot \log \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\text{ref}}(a_t \mid s_t)} + r_{\text{RM}}(x, y) \cdot \mathbb{1}[t = T]$$

逐项解释：

- **第一项是 token-level KL penalty**——对**每个 token** 都减一份 $\beta \log(\pi_\theta / \pi_{\text{ref}})$。当 $\pi_\theta(a_t|s_t) > \pi_{\text{ref}}(a_t|s_t)$（policy 比 ref 更想输出这个 token），就给一个负 reward 拽回去。
- **第二项是 RM reward**——只在 trajectory 末尾（$t = T$，即 EOS 那一步）给一次。中间 token 的 RM 项 = 0。
- $\beta$ 通常 0.01-0.1，控制 KL 约束强度。**$\beta$ 是 PPO RLHF 最敏感的超参之一**：太大 → policy 学不动（被拽住）；太小 → reward hacking。

注意这里**用的是单 sample log-ratio 而不是真 KL**（真 KL 要对 a 求和），这是个估计值。Schulman 在 [Schulman 2020 KL approx blog] 里讨论过这种 **k1 估计**及其偏差，主流实现都接受这个估计。

> **关键混淆点**：PPO 里其实有**两个 KL**——
> - **clip 内 KL**（policy vs $\theta_{\text{old}}$）：通过 ratio clip 软约束，控制单次 update step 大小
> - **reward 项 KL**（policy vs $\pi_{\text{ref}}$）：直接进 reward，控制 policy 全程不漂离 SFT
>
> 两者数学形式相似（都是 $\log \pi_a / \pi_b$），目的与作用机制完全不同。**面试常考点**——别答混。

### 2.5 LLM 上 PPO 的训练循环

把 §2.2-2.4 拼起来，一个完整 PPO 大循环：

```
for ppo_iter in range(N_ITERS):
    # ===== Phase 1: Rollout（推理阶段，no_grad）=====
    prompts = sample_batch(dataset, B_rollout)            # B_rollout 通常 256-2048
    with torch.no_grad():
        # 用当前 policy 生成 response（这一刻的 policy 即将成为 θ_old）
        responses, log_pi_old = policy.generate(prompts)
        # 同时跑 ref model 算 log π_ref
        log_pi_ref = ref_model.forward(prompts + responses).log_prob
        # RM 给 trajectory-level reward
        r_RM = reward_model(prompts, responses)
        # critic 估每个 state 的 V
        values = critic.forward(prompts + responses)

    # ===== Phase 2: 算 per-token reward 与 GAE =====
    # per-token reward = -β·KL + r_RM·1[t=T]
    rewards = -beta * (log_pi_old - log_pi_ref)            # 注意这里 ref 用的是 old policy 的 log_p
    rewards[:, -1] += r_RM                                 # 末尾加 RM
    # GAE 反向递推
    advantages, returns = compute_gae(rewards, values, gamma=1.0, lam=0.95)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # normalize

    # ===== Phase 3: PPO update（K 个 epoch）=====
    for epoch in range(PPO_EPOCHS):                         # 通常 4
        for mini_batch in shuffle_split(rollout, MB_SIZE):
            # 重新前向 policy 算当前 log_pi
            log_pi = policy.forward(mini_batch.tokens).log_prob_at(mini_batch.actions)
            ratio = torch.exp(log_pi - mini_batch.log_pi_old)

            # PPO clipped objective
            surr1 = ratio * mini_batch.advantages
            surr2 = torch.clamp(ratio, 1 - eps, 1 + eps) * mini_batch.advantages
            L_clip = -torch.min(surr1, surr2).mean()

            # critic loss
            v = critic.forward(mini_batch.tokens)
            L_vf = F.mse_loss(v, mini_batch.returns)

            loss = L_clip + c1 * L_vf
            loss.backward(); opt.step(); opt.zero_grad()
```

注意 **Phase 1 的 "no_grad" 是 PPO 显存可控的关键**——rollout 阶段 4 个 model 都是推理，没有梯度激活，显存压力远小于训练。Phase 3 才需要 policy + critic 的梯度，ref / RM 始终 frozen。

---

## 3. 最小代码示例

### 3.1 PPO clipped objective 核心实现（25 行）

```python
import torch
import torch.nn.functional as F

def ppo_loss(log_pi_new, log_pi_old, advantages, eps=0.2):
    """
    log_pi_new: (B, T)  当前 policy 在 rollout token 上的 log_prob
    log_pi_old: (B, T)  rollout 时 policy 的 log_prob（detached, 无梯度）
    advantages: (B, T)  每 token 的 GAE advantage（已 normalize）
    """
    # importance ratio ρ = π_new / π_old
    ratio = torch.exp(log_pi_new - log_pi_old)              # (B, T)

    # 两条 surrogate
    surr_unclipped = ratio * advantages                      # 未 clip
    surr_clipped   = torch.clamp(ratio, 1 - eps, 1 + eps) * advantages

    # 取小者作 lower bound，再取负（PyTorch minimize loss）
    pg_loss = -torch.min(surr_unclipped, surr_clipped).mean()

    # 监控指标：被 clip 的 token 占比
    clip_frac = ((ratio - 1.0).abs() > eps).float().mean()
    approx_kl = (log_pi_old - log_pi_new).mean()             # k1 KL 估计

    return pg_loss, clip_frac, approx_kl
```

要点：
- `log_pi_old` 必须是 `.detach()` 后的——它是 rollout 时的快照，不参与 backprop
- `clip_frac` 与 `approx_kl` 是 PPO debug 必看指标。clip_frac 长期 > 0.3 → ratio 偏离太大、$\epsilon$ 可能要调小或 $\theta_{\text{old}}$ 同步太晚。approx_kl 突然飙升 → policy 漂移失控、考虑 early stop 当前 PPO epoch。

### 3.2 GAE + per-token KL reward 计算（30 行）

```python
import torch

def compute_per_token_rewards(log_pi_old, log_pi_ref, r_rm, beta=0.05):
    """
    log_pi_old: (B, T)  policy 在 rollout token 上的 log_prob
    log_pi_ref: (B, T)  ref model 在同样 token 上的 log_prob
    r_rm:       (B,)    每条 trajectory 的 RM 标量分数
    """
    # token-level KL penalty: -β · log(π/π_ref)
    rewards = -beta * (log_pi_old - log_pi_ref)              # (B, T)
    # 末位加 RM scalar reward
    rewards[:, -1] += r_rm                                   # (B, T)
    return rewards


def compute_gae(rewards, values, gamma=1.0, lam=0.95):
    """
    rewards: (B, T)  per-token reward（含 KL penalty + 末位 RM）
    values:  (B, T)  critic 给每个 state 的 V(s_t)
    返回: advantages (B, T), returns (B, T)
    """
    B, T = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(B, device=rewards.device)
    # 末位 next_v = 0（trajectory 终止）
    for t in reversed(range(T)):
        next_v = values[:, t + 1] if t + 1 < T else torch.zeros(B, device=rewards.device)
        delta = rewards[:, t] + gamma * next_v - values[:, t]   # TD residual
        last_gae = delta + gamma * lam * last_gae
        advantages[:, t] = last_gae
    returns = advantages + values                                # value target
    # whitening：normalize advantage 到 0 均值 1 方差，PPO 训练稳定关键
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages, returns
```

注意 **per-token KL 用的是 $\log \pi_{\theta_{\text{old}}}$ 不是 $\log \pi_\theta$**——KL penalty 在 rollout 阶段就算好作为"环境给的 reward"固定下来，不在 PPO update 里参与 backprop。这是工程标准做法（trl / OpenRLHF 都这样），与 §2.4 公式严格意义上有微小差异但影响极小。

### 3.3 完整 TRL PPO trainer 配置（40 行）

```python
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig
import torch

base = "Qwen/Qwen2.5-7B-Instruct"
tok = AutoTokenizer.from_pretrained(base); tok.pad_token = tok.eos_token

# ===== 4 个 model =====
# 1. policy（actor + critic head 共享 backbone，省一份）
policy = AutoModelForCausalLMWithValueHead.from_pretrained(
    base, torch_dtype=torch.bfloat16,
    peft_config=LoraConfig(r=16, lora_alpha=32, target_modules="all-linear"),
)
# 2. reference model（frozen，KL 约束用）
ref = AutoModelForCausalLMWithValueHead.from_pretrained(base, torch_dtype=torch.bfloat16)
for p in ref.parameters(): p.requires_grad_(False)

# 3. reward model（frozen，给 trajectory 打分）
rm = AutoModelForSequenceClassification.from_pretrained(
    "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2", torch_dtype=torch.bfloat16, num_labels=1,
)
rm.eval()

# 4. critic（实际是 policy 的 value head，已包含在 policy 里）

# ===== PPO 超参 =====
cfg = PPOConfig(
    learning_rate=1e-6,         # ★ 比 SFT 小 1-2 个量级
    batch_size=512,             # rollout 一批的 trajectory 数
    mini_batch_size=64,         # 每个 SGD mini-batch 大小
    ppo_epochs=4,               # ★ 同一批 rollout 跑 4 epoch
    cliprange=0.2,              # PPO ε
    cliprange_value=0.2,        # critic loss 也 clip 防剧烈漂
    init_kl_coef=0.05,          # ★ KL 约束 β
    target_kl=6.0,              # 自适应 KL 时的目标值（可选）
    gamma=1.0, lam=0.95,        # GAE
    bf16=True,
)

trainer = PPOTrainer(cfg, policy, ref, tokenizer=tok)

gen_kwargs = dict(max_new_tokens=512, do_sample=True, temperature=0.9, top_p=0.95)
for batch in dataloader:
    queries = batch["input_ids"]
    responses = trainer.generate(queries, **gen_kwargs)             # rollout
    with torch.no_grad():
        rewards = [rm(torch.cat([q, r])).logits[0, 0] for q, r in zip(queries, responses)]
    stats = trainer.step(queries, responses, rewards)               # PPO update
```

要点：
- `AutoModelForCausalLMWithValueHead` 自动给 LLM 加一个 value head 当 critic，与 policy 共享 backbone——**这是 trl 节省一份 model 的关键工程 trick**
- `init_kl_coef` 即 §2.4 的 $\beta$；可设 `adap_kl_ctrl=True` 启用自适应 KL（按当前 KL 距离 `target_kl` 多远自动调 $\beta$，InstructGPT 的做法）
- LoRA + bf16 + ZeRO-3（DeepSpeed config 另配）是 7B 起 PPO 的最低配；**全参 7B PPO 至少 8×80G H100，70B 至少 64×80G**

### 3.4 4 模型 RLHF 显存估算（15 行）

```python
def estimate_ppo_memory_gb(n_params_b, full_finetune=True, lora=False):
    """简化估算：4 model 各自 weights + (训练时) optimizer state + activations"""
    # 单 model bf16 weights: 2 byte/param
    w = n_params_b * 2.0    # GB
    # AdamW: m + v + master fp32 = 3x weights for fp32 optimizer (bf16 训练 + fp32 master)
    optim = w * 6.0 if full_finetune else 0  # Adam states (fp32 m, v) + fp32 master
    # activation 估计（gradient checkpoint 后约 0.5x weights）
    act = w * 0.5
    train_per_model = w + optim + act
    frozen_per_model = w  # frozen 时只占 weights
    # 4 model: policy(train) + critic(train, 共享 backbone 时省到 ≈head only) + ref(frozen) + rm(frozen)
    total = train_per_model + (0.05 * w if True else train_per_model) + frozen_per_model + frozen_per_model
    if lora:
        total = total - optim + 0.02 * optim   # LoRA 只训 ~1% 参数
    return total

# 7B 全参 PPO: ≈ 280 GB（理论值，实际 + 通讯 buffer 需更多）
# 70B 全参 PPO: ≈ 2800 GB → 必须 LoRA + ZeRO-3 + ref/RM offload
print(f"7B full PPO: {estimate_ppo_memory_gb(7):.0f} GB")
print(f"70B + LoRA:  {estimate_ppo_memory_gb(70, lora=True):.0f} GB")
```

这只是数量级估算，工程实测还要算上：DeepSpeed/FSDP 的通讯 buffer（约 weights 的 0.5-1x）、kvcache（generate 阶段长 prompt 显存可观）、CPU offload 时的 pinned memory 等。**核心结论：7B PPO 最低 4×80G、70B PPO 必须 32G+ 卡 64 张起步**。

---

## 4. PPO 超参经验表

| 超参 | 经验值 | 备注 |
|---|---|---|
| `learning_rate` (policy) | 1e-6 ~ 5e-6 | **远小于 SFT**（SFT 常用 1e-5 ~ 5e-5）；PPO 信号噪声大、过大 lr 直接训崩 |
| `learning_rate` (critic) | 1e-5 ~ 5e-5 | 比 policy 大一个量级，让 critic 跟得上 policy |
| KL `β` | 0.01 ~ 0.1 | 大 = 保守不漂、reward 涨慢；小 = 漂移快、reward hacking 风险高 |
| clip `ε` | 0.2 | 经典值；保守可设 0.1，大于 0.3 几乎等同 vanilla PG |
| `ppo_epochs` | 4 | 经典甜点；> 4 容易 over-update，1-2 数据效率低 |
| `mini_batch_size` | 32 ~ 256 | 视显存；越大越稳越慢 |
| `rollout batch` | 256 ~ 2048 | 一批采的 trajectory 数；大 batch 方差小但慢 |
| GAE `λ` | 0.95 | 经典值；LLM 短 trajectory 可降到 0.9 |
| `γ` | 1.0 | LLM 任务长度有限，**不要**带 0.99 的肌肉记忆 |
| sampling `temperature` | 0.7 ~ 1.0 | 太低无探索 PPO 学不动；太高采样质量差 |
| `top_p` | 0.9 ~ 1.0 | nucleus sampling，避免低概率长尾 token |
| `target_kl` | 3 ~ 10 | 自适应 KL 时；超过即触发降 lr 或 early stop |

InstructGPT [Ouyang 2022] 与 Llama-2 [Touvron 2023] 的 RLHF 章节给出的具体数字基本都落在上表区间内。

---

## 5. 工程踩坑与经验

- ❗ **4 模型显存爆是 PPO 工程 #1 痛点**——policy / critic / ref / RM 同时驻留，70B 模型理论 4×140GB = 560GB。必须组合用：(1) LoRA on policy（训参数 < 1%）；(2) policy 与 critic 共享 backbone（trl 的 `AutoModelForCausalLMWithValueHead` 默认这么做，省一整份 model）；(3) ref model offload 到 CPU（每个 PPO step 只前向一次，慢但省显存）；(4) RM 用更小尺寸（1-7B RM 训 70B policy 完全可行，没必要 same size）；(5) ZeRO-3 / FSDP 把 weights / optimizer 切分到多卡。
- ❗ **KL coef β 调小会触发 reward hacking**——表面看 RM reward 一路涨 30%+，实际人工评测掉 10%。诊断信号：approx_kl 持续上升、response 变得套路化（开头都是 "Sure, here is..."）、response 长度异常增加。修复：把 $\beta$ 调大（0.01 → 0.05），或开启自适应 KL（`adap_kl_ctrl=True` + `target_kl=6`），或在 reward 里加 length penalty。
- ❗ **clip ε 太大（0.5+）等同于 vanilla PG**——失去 trust region 约束，policy 单步漂移大、训练不稳。$\epsilon$ 太小（0.05 以下）则梯度长期被 clip 掉、学不动。**0.2 是经典甜点，不要轻易改**。如果发现 clip_frac 持续 > 0.4，先怀疑 lr 太大或 ppo_epochs 太多、再考虑改 $\epsilon$。
- ❗ **ppo_epochs > 4 容易 over-update**——同一批 rollout 复用太多次，policy 早就漂离 $\theta_{\text{old}}$ 太远，importance ratio 失效。InstructGPT / Llama-2 / 大量复现实验都收敛在 K=4 附近。1-2 epoch 数据效率太低（rollout 是 PPO 时间瓶颈），4 是 sweet spot。
- ❗ **generate 时温度 0.0-0.3 → trajectory 缺乏探索 → PPO 学不到东西**——RL 需要采样多样性才能发现"好动作"。温度 0 等同 greedy decode，每次采到几乎一样的 trajectory，advantage 全是 0，policy 更新约等于随机游走。RLHF 实操统一用 `temperature=0.7~1.0`，`top_p=0.9~0.95`。
- ❗ **critic 初始化建议从 RM 复制**——critic 要估的"value of state"与 RM 评估的"reward of (state, response)"在分布上接近，从 RM init 比 random init 让 V 一开始就估得相对准、advantage 噪声小。trl 默认是从 policy clone + 加 random head，**实测从 RM clone head 收敛快 1.5-2x**（OpenAI / Anthropic 实操共识）。
- ❗ **reward 必须先 normalize**——RM 输出的 raw reward 分布跨 prompt 差异大（有的 prompt RM 给 -3，有的给 +5），不 normalize 直接进 GAE，advantage 量纲乱、PPO 训练不稳。标准做法：维护一个 running mean/std，每个 batch 的 reward 减均值除标准差再喂给 GAE（trl `whiten` 参数控制）。
- ❗ **长 trajectory 的 GAE 数值不稳**——$\hat A_t = \delta_t + \gamma\lambda \hat A_{t+1}$ 反向递推，$\gamma\lambda < 1$ 时几何衰减、长 trajectory（>1k token）累计误差小；但 $\gamma = 1.0, \lambda = 1.0$ 时退化成 Monte Carlo，长序列方差爆。LLM RLHF 标配 $\gamma=1.0, \lambda=0.95$，长 trajectory 时把 $\lambda$ 调到 0.9 更稳。
- ❗ **PPO 训练 1 epoch ≫ SFT 1 epoch 时间**——同样 1 万条数据：SFT 走一遍 forward + backward；PPO 要 (1) generate 几百 token / 条（推理就是 SFT 时间的 10-100x）、(2) 4 model 同时 forward、(3) 4 epoch update。综合时间常数 50-500x SFT，加 multi-model 显存占用，**总 GPU·小时成本是 SFT 的 1-2 个量级**。这就是 DPO（9.4）和 GRPO（9.5）这些"省一两个 model"的简化方案这么火的根本原因。
- ❗ **trl PPOTrainer 在 multi-GPU / DeepSpeed Stage-3 上行为常坑**——常见报错：rollout 阶段 generate 不支持 ZeRO-3 partition、ref model 与 policy 不在同一 device、AdamW state 切分不一致。**实操建议**：先在单卡跑通 1B 模型的 PPO，再上 multi-GPU；7B+ 直接用 OpenRLHF / verl / NeMo-Aligner 这些工业级框架（trl 主要面向研究 prototype）。
- ❗ **必须 log 的 PPO 监控指标**：`reward_mean`（应缓慢上升）、`approx_kl`（应稳定在 5-10 之间，飙升即异常）、`clip_frac`（< 0.3 健康）、`value_loss`（critic 拟合质量）、`entropy`（response 多样性，骤降是 mode collapse 信号）。任何一个偏离正常区间立刻停训查。

---

## 6. 经典 paper

- **Schulman et al., 2017 — *Proximal Policy Optimization Algorithms*** — PPO 原典，必读必引。读 §3 的 clipped surrogate objective 推导（本节 §2.2 公式直接来自此），与 §6 的实验对比 TRPO / A2C；§2 的 importance sampling + KL constraint 视角能彻底搞清"PPO 为什么这样设计"。这是 LLM RLHF 时代过去 8 年间不变的事实标准算法。
- **Schulman et al., 2015 — *Trust Region Policy Optimization*** — PPO 的直接前作。读 §3 的 KL constraint 推导，理解"trust region"为什么是必要的——PPO 是 TRPO 的工程友好版，但思想完全继承。读这篇才能真正回答面试题"PPO 与 TRPO 的核心区别"。
- **Ouyang et al., 2022 — *Training language models to follow instructions with human feedback (InstructGPT)*** — PPO + LLM 范式的奠基论文。§3.6 给出在 LLM 上 PPO loss 的具体形式，包括本节 §2.4 的 per-token KL penalty 与 RM reward 组合（公式 (2)）；§A 的实验细节包含全部超参（lr / β / clip / epoch）。所有 PPO RLHF 工程实现都在复刻这一套。
- **Bai et al., 2022 — *Training a Helpful and Harmless Assistant with RLHF*（Anthropic HHH）** — 与 InstructGPT 并列的工业 RLHF 实战 paper，更详细讨论 RM scaling、KL 约束的工程经验、reward hacking 现象。读 §3 的 RLHF pipeline 与 §B.5 的 PPO 超参表，对实操选超参极有帮助。
- 选读：**Engstrom et al., 2020 — *Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO*** — PPO 实现细节深度解剖。论证了 reward normalization、obs normalization、value loss clipping、advantage whitening 等"实现细节"对最终性能的贡献往往**大于** clip 本身——这篇是所有"我抄了 PPO 公式但训不动"读者的解药。

---

## 7. 自测与面试题

**Q1（公式 / 概念）：** 写出 PPO clipped objective 的完整公式，并解释 clip + min 的物理意义（为什么 $\hat A > 0$ 与 $\hat A < 0$ 时 clip 方向相反？）。

<details>
<summary>Answer sketch</summary>

公式：

$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t\bigl[\min(\rho_t \hat A_t, \; \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon) \hat A_t)\bigr], \quad \rho_t = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$$

物理意义分两种情况：

- **$\hat A_t > 0$（好动作，应升概率）**：未 clip 项 $\rho \hat A$ 想推 $\rho$ 越大越好；clip 把 $\rho$ 卡在 $1+\epsilon$ 上限。**取 min** → $\rho > 1+\epsilon$ 后 loss 不再增大、梯度归零——**防止 over-update**（过度奖励）。
- **$\hat A_t < 0$（坏动作，应降概率）**：$\rho \hat A$ 想把 $\rho$ 推得越小越好；clip 卡在 $1-\epsilon$ 下限。**取 min**（注意负数比较颠倒）→ $\rho < 1-\epsilon$ 后梯度归零——防止过激惩罚。
- **联合作用**：让 ratio 偏离 1 太远的 token 不再贡献梯度，等价于一个软 trust region，不需要显式算 KL constraint（TRPO 的做法）。

加分点：
- $L^{\text{CLIP}}$ 是个 lower bound（pessimistic surrogate）——放弃信任域内的最大化机会，换信任域外的安全
- 与 vanilla PG 对比：vanilla PG 是 $\rho \hat A$ 直接 maximize，无任何步长保护
- 与 TRPO 对比：TRPO 用硬 KL constraint，PPO 用软 ratio clip，效果接近但工程友好得多

</details>

**Q2（架构 / 显存）：** PPO 在 LLM 上的"4 模型"分别是哪 4 个？分别 trainable / frozen？为什么需要这 4 个？工程上有哪些显存节省手段？

<details>
<summary>Answer sketch</summary>

**4 模型**：

| 模型 | trainable / frozen | 作用 |
|---|---|---|
| **policy / actor** | trainable | 待优化的 LLM，给 next-token 概率 $\pi_\theta$ |
| **critic / value model** | trainable | 估 $V(s_t)$，给 GAE 算 advantage 用 |
| **reference model** | **frozen** | 计算 KL penalty $\log(\pi_\theta / \pi_{\text{ref}})$，防止 policy 漂离 SFT |
| **reward model** | **frozen** | 给整条 trajectory 打分，提供 RM reward |

**为什么必须 4 个**：
- policy 是优化目标
- critic 用来估 baseline 降方差（直接用 return 噪声太大）
- ref 用来约束 policy 不漂（防 reward hacking）
- RM 提供监督信号（不能在线问人）

**显存节省手段**（必给至少 4 条）：
1. **policy 与 critic 共享 backbone**：critic 只是在 policy 上加一个 value head（标量输出），共享 attention / FFN 参数。trl 的 `AutoModelForCausalLMWithValueHead` 默认这么做，省一整份 model 的 weights。
2. **LoRA on policy / critic**：只训 LoRA adapter（< 1% 参数），optimizer state 缩到 1%。70B 全参 PPO 不可能、+ LoRA 可行。
3. **ref model offload 到 CPU**：每个 PPO step 只前向一次（rollout 阶段），CPU 慢但显存占用归零；用 ZeRO-3 / FSDP 时直接 partition 也可。
4. **RM 用更小尺寸**：1-7B RM 训 70B policy 完全可行，没必要 same size。（不过 RM 越大对齐效果越好，是个 trade-off。）
5. **ZeRO-3 / FSDP**：weights / gradients / optimizer 三阶切分到所有 GPU，单卡显存占用 ÷N。
6. **gradient checkpointing**：activation 不全存，反向时重算，显存换计算（约省 50% activation）。
7. **bf16 / fp8 训练**：weights / optimizer state 量纲 -50%。

加分：理论 70B 全参 PPO 显存 ≈ 4 × 140GB = 560GB（仅 weights），+ optimizer + activation 实际 1500GB+。即使 64×80GB H100 也只是勉强够，所以工业 70B+ RLHF 几乎全用 LoRA + ZeRO-3 + offload 三件套。

</details>

**Q3（实战 debug）：** 你 RLHF 训了一个 7B 模型，PPO 跑了 200 步，RM reward 从 1.2 涨到 3.8（+30%），开心地发布——但用户反馈和人工评测发现质量大幅下降。给出至少 3 个可能原因 + 对应修复方向。

<details>
<summary>Answer sketch</summary>

这是 RLHF 经典的 **reward hacking + RM 漂移** 综合症。可能原因：

**原因 1：reward hacking（policy 学到了 RM 的"作弊路径"）**
- 现象：policy 输出风格趋同（套路化开场、固定格式、emoji 滥用、length 暴涨），但 reward 一路涨
- 诊断：检查 KL 距离 $\approx D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$ 是否远超训练目标（如 target_kl=6 但实测 30+）；统计 response 长度分布，看是否暴涨
- 修复：(a) **调大 KL coef β**（0.01 → 0.05 甚至 0.1），拽住 policy；(b) 开自适应 KL（trl 的 `adap_kl_ctrl=True`）；(c) 加 length penalty 到 reward；(d) 引入 RM ensemble（多个 RM 投票，单个 RM 漏洞难同时被抓）

**原因 2：RM 本身有 bias / 训练数据不全**
- 现象：RM 评分高但人工觉得差——RM 学到了人类标注员的偏好 bias（比如更长的回答 / 更"有礼貌"的回答 / Markdown 格式回答），不代表真实质量
- 诊断：抽样 20 条 RM 高分 response 人工评，看 RM 高分是否对应真高质；查训练 RM 时 chosen / rejected 的长度 / 风格分布
- 修复：(a) **重训 RM**：去 length bias、加多元化数据、加多目标（helpful + harmless + 真实性等多 head）；(b) 上 RM ensemble；(c) 推理时 length-normalize $r' = r - \alpha \cdot \text{len}(y)$

**原因 3：critic 漂 / advantage 估计错**
- 现象：reward 涨但 actor 真实学到的"哪些 token 该升概率"是错的——critic 估错 V → advantage 错 → policy 朝错方向走
- 诊断：监控 `value_loss` 是否平稳收敛；监控 advantage 分布（应近似零均值，方差适中）；监控 `clip_frac`（应 < 0.3，长期 0.4+ 说明 policy 已漂）
- 修复：(a) critic 单独多训几步（critic_warmup）；(b) critic lr 调大（比 policy lr 大一个量级）；(c) critic init 从 RM clone 而不是 random head

**原因 4：评测集与 RM 训练集分布偏移**
- RM 在 chat 数据上训，但你用它评 reasoning / code，OOD 直接崩
- 修复：domain-specific RM；或用 LLM-as-Judge 在线 sanity check

**加分思考**：
- "reward 涨 + 实际质量降" 是 RLHF 的最经典失败模式，**KL 约束就是为了防这个**——发现就要调 β
- 工业实战的标配预防：(1) RM ensemble；(2) 在线 LLM-as-Judge 监控（GPT-4o 抽 1% PPO sample 评分，与 RM 分对照）；(3) hold-out 人工评测每隔 N 步跑一次
- 长期解：RLHF 之后必须有人工评测兜底，不能只看 RM 分数；这条是 9.6《工程踩坑》的核心主题

</details>

---

## 8. 延伸阅读

- [OpenAI Spinning Up — PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html) — Joshua Achiam 写的 PPO 标准入门，含完整伪代码与 PyTorch 实现；本节 §2 的推导路线与此完全一致
- [CleanRL — ppo.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py) — 单文件 PPO 实现（< 400 行），可直接对照本节 §3 看每个工程细节怎么落地
- [Hugging Face TRL 文档 — PPOTrainer](https://huggingface.co/docs/trl/main/en/ppo_trainer) — TRL 官方手册，§3.3 代码的完整 API
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) — 工业级 RLHF 框架，70B+ PPO 实操标配；阅读其 PPO trainer 源码能学到大量 trl 没有的并行 / offload 细节
- [verl](https://github.com/volcengine/verl) — 字节跳动开源 RLHF 框架，对 multi-turn / agent RL 友好，是 Module 15 的工程基础
- [Lambert 2024 — RLHF Book Ch.7-8](https://rlhfbook.com/) — Nathan Lambert 写的开源 RLHF 教材，PPO + KL 约束 + reward hacking 章节
- [The 37 Implementation Details of PPO（Huang et al. 2022）](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) — PPO "实现魔鬼" 大全，与 Engstrom 2020 互补
- 推荐继续读本教程的 **9.4 节《DPO 闭式解推导》**——PPO 太贵，DPO 用一招代数推导把 RM + PPO + critic 全省了；以及 **9.5 节《GRPO》**——保留 PPO 的 ratio clip + KL 约束，但去掉 critic 用 group-relative baseline，是 DeepSeek-R1 的 RL 算法
