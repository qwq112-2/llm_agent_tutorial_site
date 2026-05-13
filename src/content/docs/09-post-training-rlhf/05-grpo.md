---
title: "9.5 GRPO：去 critic 的 group-relative advantage"
description: "GRPO 是 DeepSeekMath（Shao 2024）提出、被 DeepSeek-R1 一战封神的 RLHF 算法——把 9.3 PPO 的 critic 整个砍掉，对每个 prompt 采 G 条 response 用组内均值-标准差归一化当 advantage，省一份 model + 训练更稳，是 2024-2026 LLM 后训练与 Agent RL 的事实新基线。"
---

> ⏱ 预计阅读 55 分钟 ｜ 难度 ★★★ 🔥 ｜ 前置：9.3 PPO

## 一句话本节讲什么

GRPO 是 DeepSeekMath（Shao 2024）提出、被 DeepSeek-R1 一战封神的 RLHF 算法——把 9.3 PPO 的 critic 整个砍掉，对每个 prompt 采 G 条 response 用**组内均值-标准差归一化**当 advantage，省一份 model + 训练更稳，是 2024-2026 LLM 后训练与 Agent RL 的事实新基线。

---

## 1. Mental model（直觉）

记住 9.3 PPO 的核心痛点：**4 个 model 同时进显存**——policy / critic / ref / RM——70B 模型仅 weights 就 560GB，工程上极其昂贵。其中 critic 还是个"二等公民"：它存在的唯一目的是**估 baseline 降方差**（让 advantage 不要全是裸 return），但它本身要训、要 forward + backward、要算 GAE，又难调又容易漂——很多 RLHF 失败案例的根因就是 "critic 估不准 → advantage 错 → policy 学崩"。

GRPO 的灵感简单到让人怀疑：**critic 不就是想给 advantage 一个 baseline 吗？那不如对同一个 prompt 多采几条 response，用这几条的均值当 baseline。**

```
PPO 视角（per-token，用 critic 估 V）：
   Â_t = r_t + γV(s_{t+1}) - V(s_t)   （需要训 critic）

GRPO 视角（per-trajectory，用组均值估 baseline）：
   对 prompt x，采 G 条 response y_1..y_G
   每条拿 reward r_1..r_G（用 RM 或 verifier）
   Â_i = (r_i - mean(r)) / std(r)      （免费的 baseline，不需要 critic！）
   每条 trajectory 内所有 token 共享这个 Â_i
```

类比一下：你考完试想知道自己考得好不好。

- **PPO 的做法**：训一个"考试预测模型"（critic），告诉你"这种题型的平均分应该是 70 分"，然后看你考了 85 分 → 比预期高 15 分 → 算"做对了"。
- **GRPO 的做法**：让全班 8 个同学做同一套题，算这 8 个人的均值 75 标准差 5，你考 85 → $(85-75)/5 = +2$ 标准差 → 显著好。**根本不需要单独训那个"预测模型"**。

这一招省掉的不只是 critic 的显存（policy 一份的量级），更省掉了"critic 训不准毁全局"的整个失败模式。代价是**每个 prompt 要多采 G - 1 条 response**，generate 时间变 G 倍——但 LLM RLHF 里 generate 本来就是瓶颈、且去掉 critic 后单步 step 显著更稳，综合下来 GRPO 的总成本反而比 PPO 低。

GRPO 的另一个隐藏属性：它对 **verifiable reward**（如数学题答案对错、code unit test 通过/失败）的 0/1 二值奖励格外友好——同 prompt 8 条 response 里有的对有的错，组内归一化后正负 advantage 自然分明，policy 信号清晰。这正是 DeepSeek-R1 用 GRPO 纯 RL 训出 long-CoT reasoning 的关键，下一节 10.3 详讲。

GRPO 与 PPO 的 mental model 差异图：

```
                PPO 的 4 模型（每 prompt 采 1 条 trajectory）
   ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐
   │ policy │  │ critic │  │  ref   │  │   RM   │
   │ train  │  │ train  │  │ frozen │  │ frozen │
   └────────┘  └────────┘  └────────┘  └────────┘
   advantage 来自 critic（GAE per-token）

                GRPO 的 3 模型（每 prompt 采 G 条）
   ┌────────┐                ┌────────┐  ┌────────┐
   │ policy │                │  ref   │  │   RM   │
   │ train  │                │ frozen │  │ frozen │
   └────────┘                └────────┘  └────────┘
   advantage 来自 G 条 reward 的组内归一化（trajectory-level）
```

---

## 2. 公式与原理

### 2.1 从 PPO 到 GRPO 的一行修改

回忆 9.3 PPO 的 token-level objective（已加 KL 约束的简化版）：

$$L^{\text{PPO}}(\theta) = \mathbb{E}_t\bigl[\min(\rho_t \hat A_t, \; \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon) \hat A_t)\bigr] - \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$$

其中 $\hat A_t$ 是 GAE 估出的 token-level advantage，**依赖 critic** $V_\phi$。

GRPO 的整套替换只动一处：把 $\hat A_t$ 换成"组内归一化的 trajectory-level reward"。其余 ratio clip + KL constraint 框架完全照搬 PPO。

### 2.2 GRPO 的 advantage 构造

设 prompt $x$，policy $\pi_\theta$ 采样 $G$ 条 response $\{y_1, y_2, \dots, y_G\}$（典型 $G = 8 \sim 64$）。对每条 response 用 RM 或 verifier 打分得到 trajectory-level reward $r_i \in \mathbb{R}$。

**组内 baseline 与归一化**：

$$\bar r = \frac{1}{G}\sum_{i=1}^G r_i, \qquad \sigma_r = \sqrt{\frac{1}{G}\sum_{i=1}^G (r_i - \bar r)^2}$$

**GRPO advantage**（trajectory-level，再广播到每个 token）：

$$\hat A_i = \frac{r_i - \bar r}{\sigma_r + \epsilon_{\text{stab}}}, \qquad \hat A_{i, t} = \hat A_i \;\; \forall t \in \{1, \dots, |y_i|\}$$

其中 $\epsilon_{\text{stab}} \approx 10^{-8}$ 是数值稳定项。**第 $i$ 条 response 内所有 token 共享同一个 $\hat A_i$**——这是 GRPO 与 PPO 最显著的算法差异：PPO 的 advantage 是 per-token 的（GAE 给出每 token 不同的 $\hat A_t$），GRPO 是 per-trajectory 的（粗粒度，但代价是 critic 全免）。

为什么这个 baseline "合法"？回到 9.1 的 policy gradient 理论：在 $\nabla \log \pi \cdot A$ 的梯度估计里，**任何与 action 无关的函数 $b(s)$ 都是合法 baseline**——它会让方差降低但不引入偏差。组均值 $\bar r$ 是对 prompt $x$ 给定下"平均 reward 期望"的 Monte Carlo 估计，正是一个合法 baseline。除以 $\sigma_r$ 进一步把 advantage scale 标准化，与 reward 量纲解耦，让 PPO clip 的 $\epsilon$ 在不同任务上行为一致。

### 2.3 GRPO 完整目标

把 §2.2 的 advantage 代回 PPO 的 ratio clip 框架。设 $\rho_{i,t}(\theta) = \pi_\theta(y_{i,t} \mid x, y_{i,<t}) / \pi_{\theta_{\text{old}}}(y_{i,t} \mid x, y_{i,<t})$，

$$
\boxed{
L_{\text{GRPO}}(\theta) = \mathbb{E}_{x, \{y_i\}_{i=1}^G}\left[ \frac{1}{G}\sum_{i=1}^G \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \min(\rho_{i,t}\hat A_i, \; \text{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon)\hat A_i) \right] - \beta\, D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})
}
$$

逐项拆解：

- **外层 $\frac{1}{G}\sum_i$**：对组内 $G$ 条 response 求平均
- **内层 $\frac{1}{|y_i|}\sum_t$**：对单条 response 内 token 求平均（DeepSeekMath 原版用 trajectory-level 平均，DAPO / Dr.GRPO 改为 token-level，§4 详谈）
- **min + clip**：与 PPO 完全一致（防止 ratio 偏离 1 太远）
- **$\hat A_i$**：从 §2.2 的组内归一化得到（**注意所有 token 共用同一个值**）
- **$\beta D_{\text{KL}}$**：DeepSeekMath 原版用的是 **k3 估计**的无偏 KL（见下方）

### 2.4 KL 项的实现：k1 vs k3 估计

PPO 把 KL 作为 per-token reward shaping（9.3 §2.4），GRPO 原版则把 KL 直接作为 loss 的一项 $-\beta D_{\text{KL}}$。DeepSeekMath 论文 §4.1.3 给出了一个非常实用的 **k3 无偏 KL 估计**：

$$D_{\text{KL}}\bigl(\pi_\theta \| \pi_{\text{ref}}\bigr) \approx \frac{\pi_{\text{ref}}(y_t \mid x, y_{<t})}{\pi_\theta(y_t \mid x, y_{<t})} - \log \frac{\pi_{\text{ref}}(y_t \mid x, y_{<t})}{\pi_\theta(y_t \mid x, y_{<t})} - 1$$

设 $u = \pi_{\text{ref}} / \pi_\theta$，则 k3 估计 $= u - \log u - 1 \ge 0$（恒非负，无偏）。对比 PPO 常用的 **k1 估计** $\log(\pi_\theta / \pi_{\text{ref}})$（有偏但简单），k3 在样本效率上更好，是 GRPO 的标志性细节之一。

### 2.5 GRPO vs PPO 对比表

| 维度 | PPO | GRPO |
|---|---|---|
| 模型数 | **4**（policy + ref + RM + critic） | **3**（policy + ref + RM；**无 critic**） |
| advantage 估计 | GAE on critic（per-token） | 组内归一化（per-trajectory） |
| Trajectory / batch | 1 ~ N（每 prompt 1 条） | $G \times N$（每 prompt $G$ 条，$G$ 典型 8-64） |
| Generate 成本 | 1× | **$G\times$**（GRPO 时间瓶颈） |
| 显存峰值 | 重（4 model + GAE buffer） | 轻（无 critic + 无 GAE buffer） |
| 适用场景 | 通用 RLHF | reasoning / math / verifiable reward 尤佳 |
| 训练稳定性 | 中（critic 漂是常见失败模式） | 较稳（无 critic 漂移问题） |
| 工程复杂度 | 高（4 model + GAE + KL shaping） | 低（3 model + 一行 group normalize） |
| KL 实现 | per-token reward shaping（k1） | loss 项（k3 无偏估计） |
| 代表论文 | InstructGPT 2022 | DeepSeekMath 2024 / DeepSeek-R1 2025 |

**核心 trade-off**：GRPO 用 G 倍 generate 时间换掉了 critic 的训练成本与显存。在 LLM RLHF 场景里，generate 成本可摊（rollout 阶段 no_grad）、critic 成本不可摊（必须 backward），所以 GRPO 总账比 PPO 划算——尤其当 reward 来自 verifier（不需要 RM forward）、或 LoRA only 训 policy 的时候。

---

## 3. 最小代码示例

### 3.1 Group baseline + advantage normalization（10 行）

```python
import torch

def group_advantage(rewards, eps=1e-8):
    """
    rewards: (B, G)  每个 prompt 采 G 条 response 对应的 reward
    返回:    (B, G)  每个 trajectory 的 GRPO advantage
    """
    mean = rewards.mean(dim=1, keepdim=True)               # (B, 1)
    std = rewards.std(dim=1, keepdim=True) + eps           # (B, 1)
    advantages = (rewards - mean) / std                    # (B, G)
    return advantages
```

要点：std 加 `eps` 防 reward 全相同时除零；组内归一化天然零均值单位方差，无需额外 whitening。

### 3.2 手写 GRPO 完整 step（45 行）

```python
import torch
import torch.nn.functional as F

def grpo_step(policy, ref_model, prompts, reward_fn, optimizer,
              G=8, beta=0.04, eps_clip=0.2, gen_kwargs=None):
    """
    policy:     待训 LLM（policy = π_θ）
    ref_model:  frozen 参考模型（π_ref）
    prompts:    list[str]，长度 B
    reward_fn:  (prompt, response) -> scalar reward（来自 RM 或 verifier）
    """
    B = len(prompts)

    # ===== Phase 1: rollout（每 prompt 采 G 条 response）=====
    with torch.no_grad():
        # generate (B, G) 条 response；典型实现：把 prompt 复制 G 份后 batch generate
        responses = policy.generate(prompts, num_return_sequences=G, **gen_kwargs)  # (B*G,)
        # 计算旧 policy logp 与 ref logp（per-token）
        log_pi_old = policy.compute_logp(prompts, responses)         # (B*G, T)
        log_pi_ref = ref_model.compute_logp(prompts, responses)      # (B*G, T)
        # reward
        rewards = torch.tensor([reward_fn(p, r) for p, r in zip(prompts*G, responses)])  # (B*G,)
        rewards = rewards.view(B, G)                                  # (B, G)

    # ===== Phase 2: 组内归一化 advantage =====
    advantages = group_advantage(rewards)                            # (B, G)
    advantages = advantages.view(B*G, 1)                             # 广播到 token 维

    # ===== Phase 3: PPO-clipped objective + KL =====
    log_pi_new = policy.compute_logp(prompts*G, responses)           # (B*G, T)，带梯度
    ratio = torch.exp(log_pi_new - log_pi_old)                       # (B*G, T)

    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantages
    pg_loss = -torch.min(surr1, surr2).mean()                        # 标量

    # k3 无偏 KL: u - log u - 1, u = π_ref/π_new
    log_u = log_pi_ref - log_pi_new
    kl = (torch.exp(log_u) - log_u - 1).mean()

    loss = pg_loss + beta * kl
    optimizer.zero_grad(); loss.backward(); optimizer.step()

    return {"loss": loss.item(), "pg_loss": pg_loss.item(),
            "kl": kl.item(), "reward_mean": rewards.mean().item()}
```

要点：
- `log_pi_old` / `log_pi_ref` 在 Phase 1 内 `no_grad` 下计算并 `detach`，不进 backprop
- advantage 是 `(B*G, 1)`，自动广播到 token 维，确保**单条 response 内每个 token 共享同一 $\hat A_i$**
- KL 用 k3 估计 `exp(u) - u - 1`，恒非负、无偏
- 相比 9.3 的 PPO 实现：**没有 critic.forward()、没有 GAE 反向递推、没有 value loss**——少了整整一支训练支路

### 3.3 TRL GRPOTrainer 配置（25 行）

```python
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig

base = "Qwen/Qwen2.5-Math-7B"
tok = AutoTokenizer.from_pretrained(base); tok.pad_token = tok.eos_token
policy = AutoModelForCausalLM.from_pretrained(base, torch_dtype="bfloat16")

# verifier-style reward：直接用函数（无需 RM model）
def reward_math(prompts, completions, answer, **kw):
    return [1.0 if extract_answer(c) == a else -1.0
            for c, a in zip(completions, answer)]

cfg = GRPOConfig(
    output_dir="ckpt/grpo",
    learning_rate=1e-6,         # ★ 与 PPO 同量级
    per_device_train_batch_size=8,
    num_generations=16,         # ★ G = 16，组内 sample 数
    max_prompt_length=512,
    max_completion_length=2048, # 长 CoT 必须开大
    beta=0.04,                  # ★ KL 系数（注意：比 PPO 小）
    epsilon=0.2,                # ratio clip
    temperature=0.9,            # ★ 必须 > 0.5 保证组内 diversity
    bf16=True,
    gradient_checkpointing=True,
    peft_config=LoraConfig(r=16, lora_alpha=32, target_modules="all-linear"),
)

trainer = GRPOTrainer(
    model=policy, processing_class=tok,
    reward_funcs=[reward_math], args=cfg, train_dataset=ds,
)
trainer.train()
```

要点：
- `num_generations` 即 $G$，TRL 默认 8，DeepSeek 原论文用 64，工业实操 16 是常见甜点
- `reward_funcs` 接受 list[Callable]：可以同时挂多个 reward（数学正确 + 格式正确 + 长度 penalty），TRL 会加权求和
- `beta=0.04` 显著小于 PPO 的 0.05-0.1——GRPO 没 critic 漂移，KL 约束可松一些
- LoRA + bf16 + gradient checkpointing 是 7B GRPO 的最低配；全参 7B GRPO 需要 4×80G H100 起步

### 3.4 dynamic sampling（DAPO 风格的"去掉 zero-advantage group"）

```python
def filter_zero_advantage_groups(rewards, min_std=1e-3):
    """
    rewards: (B, G)  组内 reward
    返回:    bool mask (B,)，True 表示该 prompt 的 group 应保留
    动机：std≈0 的 group 意味着 G 条 response 全对或全错，
          归一化后 advantage 全 0，policy 学不到东西，浪费一次 update。
    """
    return rewards.std(dim=1) > min_std
```

DAPO（Yu 2024）观测到 RLVR 训练后期大量 prompt 的 G 条 response 全对（或全错），advantage 全 0 直接浪费算力，过滤掉这些 group 显著提升训练效率。

---

## 4. GRPO 的现代变体

GRPO 自 2024 年初提出后短短 18 个月内催生了一整个变体家族。下面是面试与实操中最高频的几个：

### 4.1 DAPO（ByteDance 2024，verl 同款）

针对 GRPO 在长 CoT RLVR 场景的三个具体问题给出工程化修复：

- **Dynamic sampling**：剔除 advantage 全 0 的 group（reward 全相同），节省无效计算
- **Token-level loss**：原版 GRPO 是 trajectory-level 平均（每条 response 内 token 先平均再跨条平均），导致**长 trajectory 的单 token 信号被稀释**——长答案里每个 token 的梯度比短答案小。DAPO 改为先把所有 token 拍平再求平均，长短公平
- **Clip-higher**：上 clip 边界放宽到 $1 + \epsilon_{\text{high}}$（如 0.28），下边界保持 $1 - \epsilon_{\text{low}}$（如 0.2）。直觉：当某 token 之前概率很低（比如 reasoning 里突然蹦出的关键 token），ratio 自然容易超过 $1 + \epsilon$，对称 clip 会扼杀这种"低概率好动作"的探索

DAPO 在 AIME 等数学基准上比原版 GRPO 显著更稳，是 verl 框架的默认配置。

### 4.2 Dr.GRPO（Liu et al. 2024）

诊断 GRPO 有两类系统性 bias 并修复：

- **Length bias**：trajectory-level 平均会让长答案的每 token 梯度被除以更大的 $|y_i|$，导致 policy "偏好"短答案——这与 reasoning 任务"思考越长越好"的目标冲突。Dr.GRPO 把 $\frac{1}{|y_i|}$ 拿掉
- **Std normalization bias**：除以 $\sigma_r$ 让组内 reward 接近的样本（即任务"easy"或"hard"两端）advantage 反而被放大，引入难度偏好。Dr.GRPO 去掉除以 std 这一步，只保留减均值

修复后 GRPO 在 reasoning 任务上长度更可控、不再"投机取巧地短答案"。

### 4.3 GSPO（Qwen3 2025）

提出 **sequence-level importance ratio** 替代 token-level：

$$\rho_i = \exp\left(\frac{1}{|y_i|}\sum_t \bigl(\log \pi_\theta(y_{i,t}) - \log \pi_{\theta_{\text{old}}}(y_{i,t})\bigr)\right)$$

整条 response 共用一个 ratio，再做 PPO clip。动机：token-level ratio 在长序列上方差极大（少量 token 的 ratio 异常会污染整体梯度），sequence-level 更稳。Qwen3 系列后训练用的就是这个。

### 4.4 RLOO / REINFORCE++（Ahmadian 2024）

**Leave-one-out baseline**：对第 $i$ 条 response，baseline 用其余 $G-1$ 条的均值

$$\hat A_i^{\text{RLOO}} = r_i - \frac{1}{G-1}\sum_{j \ne i} r_j$$

理论上 RLOO 是组均值 baseline 的无偏改进（去掉了"自己也参与求均值"的细微偏差）。Ahmadian et al. 2024 发现 RLOO 在 RLHF 上比 PPO 更稳且更省算力。GRPO 与 RLOO 在数学上非常接近，可视为同一思想（去 critic + 组内 baseline）的两种工程实现。

### 4.5 共同思想总结

这些变体的 pattern 都是 **"在 GRPO 骨架上修 bias / 调 baseline / 改 loss aggregation"**：

| 变体 | 修了什么 | 关键改动 |
|---|---|---|
| DAPO | 训练效率 + clip 不对称 | dynamic sampling, token-level loss, clip-higher |
| Dr.GRPO | length / difficulty bias | 去 $\frac{1}{|y_i|}$ + 去 $\sigma_r$ |
| GSPO | token-level 方差 | sequence-level ratio |
| RLOO | baseline 偏差 | leave-one-out 均值 |

面试时常被问"GRPO 之后还有什么"——记住这 4 个 + 它们各自修的痛点即可。

---

## 5. 工程踩坑与经验

- ❗ **Group size G < 4 → baseline 噪声大、advantage 不准**——$G=2$ 时归一化基本就是"两条互比"，正负方向乱跳；实测 $G \ge 8$ 才稳，DeepSeek-R1 用到 $G=64$。但 $G$ 太大（>64）只是降方差边际收益递减、generate 成本却线性增长，**$G=8 \sim 16$ 是工业实操甜点**。
- ❗ **Temperature < 0.5 → group 内 response 几乎相同 → advantage 全 0 → policy 不学**——GRPO 的所有信号都来自组内 reward 差异，没有差异就没有梯度。务必 `temperature \in [0.6, 1.0]`，配 `top_p=0.9~0.95`。极端情况下温度 0 + greedy decode 时 GRPO 训练曲线会**完全水平**——经典 day-1 bug。
- ❗ **同 prompt 内 G 条 response reward 全相同（全对/全错）→ advantage 全 0**——RLVR 中后期非常常见（容易题全 pass、超难题全 fail）。原版 GRPO 会照常 update（贡献 0 梯度浪费算力），DAPO 的 dynamic sampling 在 batch 内剔除这类 prompt 重新补采，**实测 RLVR 训练到 50% steps 时约 30-50% prompt 落入此类**，不修复纯属浪费算力。
- ❗ **GRPO 总训练时间不一定比 PPO 短**——单步 step 更便宜（无 critic backward），但每 prompt 要 generate G 倍、rollout 阶段时间是 PPO 的 G 倍。**G=16 时 GRPO wall-clock ≈ PPO 的 1.5-2.5x**（PPO 也要 ppo_epochs=4 多次 update，部分摊销了）。GRPO 真正的优势在**显存**和**训练稳定性**，不是 wall-clock。
- ❗ **KL coef β 显著小于 PPO（0.01-0.05）**——经验值。原因：(1) 没 critic 错估带来的连锁漂移；(2) advantage 已被 std 归一化、量纲稳定，policy 自然不会大步漂；(3) GRPO 论文用 0.04，DAPO 甚至用 0.001。**β > 0.1 通常会让 GRPO 学不动**——KL penalty 完全压住组内 reward 信号。
- ❗ **长 trajectory + 大 G → 显存爆**——8K context 的 long-CoT × G=16 = 128K token 同时进 backward，连 80GB H100 都吃不消。标配组合：(1) LoRA only 训 policy；(2) gradient checkpointing；(3) generate 阶段用 vLLM 离线（与训练 step 解耦）；(4) DeepSpeed ZeRO-3 切 weights/optimizer。
- ❗ **Token-level vs trajectory-level loss aggregation 影响巨大**——DeepSeekMath 原版是 $\frac{1}{|y_i|}$ trajectory-level 平均，长答案的单 token 梯度被稀释。改 token-level（拍平所有 token 后平均）后 reasoning 任务长度提升、AIME 准确率显著涨 5-10 pp（DAPO 实测）。**新写 GRPO 默认就用 token-level，原版 trajectory-level 是 deprecated**。
- ❗ **Verifier reward (binary 0/1) 比 RM 更适合 GRPO**——RM 输出连续分数会让 advantage 受 RM 系统偏差（如 length bias）放大；verifier 给 0/1 的 ground truth 信号，组内归一化后 advantage 必为 $\pm \frac{\sqrt{G-1}}{\text{some scale}}$ 的清晰二极分布，policy 学得最快。这就是 GRPO + RLVR 在 reasoning 上特别成功的原因。
- ❗ **TRL 的 GRPOTrainer 默认参数不一定与 DeepSeek 官方实现一致**——TRL 默认 `loss_type="grpo"` 是 trajectory-level，要复现 DeepSeek-R1 需手动设 `loss_type="dr_grpo"` 或 `loss_type="dapo"`。版本 0.13+ 才完整支持，老版本 KL 实现是 k1 不是 k3。**复现实验前务必 check TRL 版本与 loss_type**。
- ❗ **Generate 与训练分离能省大量时间**——TRL 默认 generate 在训练进程内（HF generate），慢且占显存。生产实践把 generate 切到 vLLM / SGLang 服务（同卡或异步），训练进程只做 backward，rollout 时间 ↓ 5-10x。verl / OpenRLHF 都内置了这种 disaggregated rollout 架构。
- ❗ **GRPO 对 reward 噪声比 PPO 更敏感**——只采 G 条 response，组内 reward 噪声直接进 advantage；PPO 有 critic 当 low-pass filter 平滑过。诊断：reward 标准差在不同 batch 间波动大（>50%）即为高噪声 → 加大 G 或用 RM ensemble。
- ❗ **每个 prompt 的 G 条 response 应**真正独立采样**——常见 bug：用 `num_return_sequences=G` 但 KV cache 复用导致前缀同步、response 出现"开头一致后面分叉"，破坏组内多样性。修复：用 `do_sample=True` + 不同 random seed 保证每条独立。

---

## 6. 经典 paper

- **Shao et al., 2024 — *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*** — GRPO 提出原典，必读必引。读 §4.1 "Group Relative Policy Optimization" 完整推导（本节 §2 的公式直接来自此），§4.1.3 给出 k3 KL 估计的工程理由，§4.2 实验对比 PPO 显示同等算力下 GRPO 在 GSM8K / MATH 上更优。这是过去两年所有 GRPO 工作的公共起点。
- **DeepSeek-AI, 2025 — *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*** — GRPO + RLVR 的工业胜利。读 §2 "Approach"——R1-Zero 直接用 GRPO + 数学/code verifier 从 base model 训出 long-CoT reasoning，**完全跳过 SFT cold start**——这是 RLVR 范式的奠基实验，10.3 详讲。本节读它能理解 GRPO 在工业 reasoning 上为什么这么火。
- **Yu et al., 2024 — *DAPO: An Open-Source LLM Reinforcement Learning System at Scale*** — ByteDance verl 框架的算法核心。读 §3 的 4 个改进（dynamic sampling、token-level loss、clip-higher、overlong filtering），是 GRPO "下一代"事实标准；同时是 verl 框架的默认配置——读这篇能直接对接到工业级 Agent RL 实操。
- 选读：**Liu et al., 2024 — *Understanding R1-Zero-Like Training: A Critical Perspective (Dr.GRPO)*** — 系统性诊断了 GRPO 的 length bias 与 difficulty bias，给出去 $\frac{1}{|y|}$ 与去 std 的最简修复。读 §3 的 bias 分析推导，对面试"GRPO 还有什么问题"加分明显。
- 选读：**Ahmadian et al., 2024 — *Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs (RLOO)*** — 与 GRPO 思想殊途同归（去 critic + 组内 baseline）。读 §3 的 leave-one-out baseline 推导，能理解 GRPO 不是凭空出现而是有完整的"去 critic"思想脉络。

---

## 7. 自测与面试题

**Q1（公式 / 概念）：** 写出 GRPO 的 advantage 计算公式（含 group baseline 与 std normalization），并解释为什么 GRPO 不需要 critic。

<details>
<summary>Answer sketch</summary>

公式：对 prompt $x$ 采 $G$ 条 response $\{y_1, \dots, y_G\}$，每条得 reward $r_i$：

$$\hat A_i = \frac{r_i - \bar r}{\sigma_r + \epsilon}, \quad \bar r = \frac{1}{G}\sum r_i, \quad \sigma_r = \text{std}(r)$$

第 $i$ 条 response 内所有 token 共享 $\hat A_i$（trajectory-level）。

为什么不需要 critic：
- **Critic 在 PPO 里的唯一作用**是给 advantage 一个 baseline 来降方差（$\hat A_t = r_t + \gamma V(s_{t+1}) - V(s_t)$）
- **Policy gradient 理论**告诉我们：任何与 action 无关的函数 $b(s)$ 都是合法 baseline，不引入偏差只降方差
- 组内均值 $\bar r$ 就是对 prompt $x$ 给定下"平均 reward 期望"的 Monte Carlo 估计，**天然合法 baseline**
- 除以 $\sigma_r$ 进一步把 advantage 标准化到单位方差，与 reward 量纲解耦

加分：
- 代价：每 prompt 要采 G 条 response（generate 时间 ×G）
- 好处：省 critic 的 weights / optimizer / GAE 计算，且消除 "critic 估不准" 的失败模式
- 这是 PPO 4 模型 → GRPO 3 模型的关键瘦身

</details>

**Q2（trade-off）：** GRPO 比 PPO 显存省，但每 step 慢 G 倍（generate 要采 G 条），为什么仍成为 2024-2026 的主流？至少给 3 个理由。

<details>
<summary>Answer sketch</summary>

**理由 1：去 critic 不只省显存，还消除一类失败模式**
- PPO 中 critic 漂移是常见失败原因（critic 估错 V → advantage 错 → policy 学崩）
- GRPO 用 Monte Carlo group baseline 代替 critic，advantage 估计是无偏的、不会漂
- 训练稳定性显著高于 PPO，调参更轻松

**理由 2：generate 成本可摊，critic 成本不可摊**
- Generate 阶段是 no_grad 推理，可用 vLLM / SGLang 加速、可异步、可量化（fp8/int4）
- Critic 训练必须 forward + backward + optimizer step，无法摊销
- 综合算力账：generate ×G 增加的成本通常小于 critic 训练 + 4 model 显存的成本

**理由 3：与 verifiable reward 完美契合**
- RLVR 范式（数学 / code verifier 给 0/1 reward）是 2024-2026 reasoning 模型主流
- 0/1 reward 在组内归一化后 advantage 信号清晰（正负二极分布）
- DeepSeek-R1 / Qwen2.5-Math / o1 复现都用 GRPO + verifier，效果远超 PPO + RM

**理由 4（加分）：工程复杂度低、生态完善**
- 实现简单（少一个 model、少一支 GAE 反向递推）
- TRL / verl / OpenRLHF / NeMo-Aligner 都原生支持
- 衍生变体丰富（DAPO / Dr.GRPO / GSPO / RLOO）满足不同场景

**理由 5（加分）：Agent RL 的 trajectory-level reward 天然友好**
- 多轮 agent 任务里 reward 通常只在 episode 末尾给（任务成功/失败）
- PPO 的 GAE per-token 在稀疏 reward 上估计极差
- GRPO 的 trajectory-level advantage 与稀疏 reward 完全匹配（Module 15 主题）

</details>

**Q3（前沿）：** DAPO 在 GRPO 基础上改了哪些？为什么这些改动能让训练更稳？

<details>
<summary>Answer sketch</summary>

DAPO 的 4 个核心改动：

**1. Dynamic sampling（动态过滤无信号 group）**
- 问题：RLVR 中后期，大量 prompt 的 G 条 response 全对或全错 → reward std ≈ 0 → advantage 全 0 → 浪费一次 update
- 修复：batch 内剔除这类 prompt 并重新补采有差异的 prompt
- 收益：后期训练有效梯度比例从 ~50% 提升到接近 100%，wall-clock 训练效率显著提升

**2. Token-level loss aggregation（取消长度归一化）**
- 问题：原版 GRPO 是 $\frac{1}{|y_i|} \sum_t \dots$ 先对单条 response 内 token 平均，导致**长 trajectory 的单 token 梯度被稀释**——长答案里关键 token 的信号被均化掉
- 修复：去掉 $\frac{1}{|y_i|}$，改为先把所有 token 拍平再做整体平均
- 收益：长 CoT 答案的关键推理 token 获得正常权重，AIME 等长推理任务准确率涨 5-10 pp

**3. Clip-higher（不对称 clip 边界）**
- 问题：原版 PPO clip 上下对称 $[1-\epsilon, 1+\epsilon]$。但 reasoning 中常出现"之前低概率的关键 token 突然冒出"——ratio 自然容易 > $1+\epsilon$，被 clip 杀掉，扼杀探索
- 修复：上界放宽到 $1 + \epsilon_{\text{high}}$（如 0.28），下界保持 $1 - \epsilon_{\text{low}}$（如 0.2）
- 收益：保留 "low-prob → high-prob" 的关键 token 梯度，鼓励 policy 对正面 reward 的"激进 update"

**4. Overlong filtering（超长样本过滤）**
- 问题：超长（接近 max_completion_length）的 response 往往是 policy 卡住胡说，reward 一般很差但梯度噪声大
- 修复：训练时丢弃超长 sample 或对其 reward 做特殊处理
- 收益：避免噪声 sample 主导梯度

**为什么这些改动让训练更稳**：
- 共同主线是"**让有效梯度信号占比最大化、让噪声梯度最小化**"
- Dynamic sampling 解决"零信号样本"
- Token-level loss 解决"长样本信号被稀释"
- Clip-higher 解决"好信号被对称 clip 误杀"
- Overlong filtering 解决"噪声样本污染"

加分：
- DAPO 在 AIME 2024 上把 Qwen2.5-32B base 训到了 50 分（与 R1-Zero 相当），完全开源，是 GRPO 工业落地的 reference
- verl 框架的默认配置就是 DAPO；2026 年新做 RLVR / Agent RL 几乎都从 DAPO 起步而非原版 GRPO

</details>

---

## 8. 延伸阅读

- [Hugging Face TRL 文档 — GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer) — TRL 官方手册，§3.3 代码的完整 API；含 DAPO / Dr.GRPO loss_type 切换说明
- [verl（ByteDance）](https://github.com/volcengine/verl) — DAPO 同款工业级 RLHF / Agent RL 框架，源码是 GRPO 工程化最完整的开源实现
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) — 早期 RLHF 框架，GRPO + RLOO 都支持，与 vLLM rollout 解耦做得很好
- [DeepSeek-R1 官方仓库与技术报告](https://github.com/deepseek-ai/DeepSeek-R1) — R1-Zero / R1 训练流程的官方说明
- [Lambert 2024 — RLHF Book Ch.11 GRPO](https://rlhfbook.com/) — Nathan Lambert 的开源 RLHF 教材 GRPO 章节，与本节互补
- [Will Brown — GRPO from scratch in 80 lines](https://github.com/willccbb/verifiers) — 教学级 GRPO 实现，配合本节 §3.2 看
- [unsloth GRPO Notebook](https://docs.unsloth.ai/basics/reasoning-grpo) — 单卡 24GB GPU 跑通 GRPO 的最小实操，零基础友好
- 推荐继续读本教程的 **9.6 节《工程踩坑：reward hacking / RM 漂移 / KL 坍塌》**——GRPO 与 PPO 共有的 RLHF 工程地雷集大成；以及 **10.3 节《RLVR 与 DeepSeek-R1》**——GRPO + verifier 的范式详解；**Module 15** 把 GRPO 推广到多轮 Agent RL
