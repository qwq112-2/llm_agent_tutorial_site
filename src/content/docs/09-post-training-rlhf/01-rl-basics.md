---
title: "9.1 RL 速通：policy / value / advantage / return / GAE"
description: "把读 PPO / GRPO / RLVR 之前必须的 RL 心智模型一次性说清——policy / value / advantage / return / GAE / on-policy 这几个词、为什么 LLM RLHF 里只用其中很窄的一个子集；本节是 9.3-9.5 的入场券，不学 RL 教材整本，只学\"够用且对得上 LLM\"。"
---

> ⏱ 预计阅读 45 分钟 ｜ 难度 ★★★ ｜ 前置：1.1 反向传播、1.2 优化器

## 一句话本节讲什么

把读 PPO / GRPO / RLVR 之前必须的 RL 心智模型一次性说清——**policy / value / advantage / return / GAE / on-policy** 这几个词、为什么 LLM RLHF 里只用其中很窄的一个子集；本节是 9.3-9.5 的入场券，不学 RL 教材整本，只学"够用且对得上 LLM"。

---

## 1. Mental model（直觉）

**RL 与监督学习的根本差异是反馈形式**。监督学习给的是 ground-truth label（"这道题答案是 42"），RL 只给一个 reward 标量（"你这次回答得 7 分"），不告诉你正确答案是什么、不告诉你哪一步走错了——你得**自己从分数倒推哪些动作值得加强**。

最小心智模型只有 5 个词：

```
                   ┌──────────────┐
       state s ───►│              │──── reward r ─────┐
                   │  environment │                   │
       action a ◄──│              │                   │
                   └──────────────┘                   │
              ▲                                       │
              │                                       ▼
       ┌──────┴──────┐                       update policy
       │  agent (π)  │◄──────────────────────────────────
       └─────────────┘
```

- **agent**：会做决策的实体
- **environment**：与 agent 交互的"世界"
- **state $s$**：环境给 agent 的观测
- **action $a$**：agent 选择的动作
- **reward $r$**：环境对 action 的标量反馈
- **trajectory** $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \dots)$：一整条交互序列

把这 5 个词翻译到 LLM 上——这是本节最关键的一次"映射"，**记不住其它都白搭**：

| 经典 RL | LLM RLHF |
|---|---|
| agent | 你的 LLM（policy 网络） |
| environment | "采样器 + reward model" 的组合 |
| state $s_t$ | 当前已生成的 token 序列（prompt + 已采的 $t$ 个 token） |
| action $a_t$ | 下一个 token（从 vocab 里选一个） |
| reward $r_t$ | 通常 $t < T$ 时为 0，只在末尾 $r_T = $ RM 打分 |
| trajectory $\tau$ | 一整条 generation（prompt + 完整回复） |

所以 LLM RL 是一种**特殊形态**的 RL：

- action 空间 = 词表大小 50k+，不是 Atari 几个按钮
- trajectory = 几百到几千 token，不是几十步
- reward = 末尾给一次的 trajectory-level 标量，不是每步给
- state 没有显式的状态空间，就是"截至此刻的 token 序列"

记住这四点的"特殊性"，9.3 PPO 为什么要 clip ratio、9.5 GRPO 为什么去 critic、10.3 RLVR 为什么 reward 必须 verifiable，全部都能从这里推。

剩下三个高频概念也一句话先过：

- **policy** = LLM 本身（softmax over vocab）
- **value** = "从这个 state 出发预计能拿到多少 reward"
- **advantage** = "这一步比平均水平好多少" → 用它而不是 reward 本身去做 policy gradient

---

## 2. 公式与原理

### 2.1 policy / value / Q / advantage 四件套

记号：状态空间 $\mathcal{S}$，动作空间 $\mathcal{A}$，参数 $\theta \in \mathbb{R}^d$，折扣因子 $\gamma \in [0, 1]$。

**Policy** 是一个条件概率分布：

$$
\pi_\theta(a \mid s) : \mathcal{S} \to \Delta(\mathcal{A})
$$

LLM 的 policy 就是模型的 next-token 分布——给定已生成的 token 序列 $s_t$，对词表做 softmax 得到 $\pi_\theta(a_t \mid s_t)$。**LLM 就是 policy，没有"额外的 policy 头"**。

**Return**（累计 reward）从 $t$ 时刻往后算：

$$
G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k} = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots
$$

$\gamma$ 是**折扣因子**（discount factor），经典 RL 里取 0.99 之类——意思是"远期 reward 比近期 reward 不值钱"。**LLM RLHF 里 $\gamma = 1$**（任务长度有限，最终 reward 一次性给，没必要再 discount），后续 PPO / GRPO 公式里看到 $\gamma$ 一律默认 1。

**State value** 是 policy $\pi$ 下从 $s$ 出发的期望 return：

$$
V^\pi(s) = \mathbb{E}_{\tau \sim \pi}[G_t \mid s_t = s]
$$

**Action value**（也叫 Q value）是"在 $s$ 选 $a$、之后按 $\pi$ 走"的期望 return：

$$
Q^\pi(s, a) = \mathbb{E}[r_t + \gamma V^\pi(s_{t+1}) \mid s_t = s, a_t = a]
$$

**Advantage** 把两者一减：

$$
A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)
$$

直觉：**advantage 衡量"在 state $s$ 上，选 action $a$ 比该 state 上的平均决策好多少"**。
- $A > 0$：这一步比平均好 → policy 应当**增加**这一步的概率
- $A < 0$：这一步比平均差 → policy 应当**减少**这一步的概率

为什么不直接用 return $G_t$ 而要算 advantage？因为 return 里包含**与当前动作无关的环境运气成分**——比如本来就是个高 value 的 state，无论选什么 action 都会得高 return，policy 没必要为此"邀功"；advantage 把 baseline $V(s)$ 减掉就把这部分运气剥离了，**只留下"该动作的相对贡献"**，方差更小。

### 2.2 policy gradient 定理

目标是最大化期望 return：

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)], \qquad R(\tau) = \sum_{t=0}^{T} r_t
$$

直接求导有个问题：期望是对 $\tau$ 取的，而 $\tau$ 的分布**依赖 $\theta$**——不能把 $\nabla_\theta$ 直接搬进期望。**Log-derivative trick**（也叫 score function estimator）解决这个：

$$
\nabla_\theta p(\tau; \theta) = p(\tau; \theta) \nabla_\theta \log p(\tau; \theta)
$$

把它代回 $J(\theta) = \int p(\tau; \theta) R(\tau) d\tau$，得到**REINFORCE 梯度**（Williams 1992）：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[ R(\tau) \cdot \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \right]
$$

直觉解读：**整条 trajectory 的 reward 越高，把 trajectory 上每一步的 logp 推得越高**——朴素到不能再朴素。

实操上最大的问题是 **方差极大**：$R(\tau)$ 在不同 $\tau$ 上波动很大，用蒙特卡罗估期望时方差直接炸。两个标准变种来降方差：

**变种 1：reward-to-go**。第 $t$ 步的动作只能影响 $t$ 之后的 reward，与 $t$ 之前的 reward 无关，所以梯度可以收紧成：

$$
\nabla_\theta J = \mathbb{E}\left[ \sum_{t} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot G_t \right]
$$

**变种 2：加 baseline**。任何**与 $a_t$ 无关**的函数 $b(s_t)$ 都能从期望里减掉而不改变梯度的期望（因为 $\mathbb{E}_a[\nabla \log \pi(a|s)] = 0$）：

$$
\nabla_\theta J = \mathbb{E}\left[ \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot (G_t - b(s_t)) \right]
$$

减 baseline 不影响期望但**减小方差**。最自然的 baseline 选择就是 $V^\pi(s_t)$，于是 $G_t - V(s_t) \approx Q(s_t, a_t) - V(s_t) = A(s_t, a_t)$，得到**Actor-Critic** 风格的梯度：

$$
\boxed{\;\nabla_\theta J = \mathbb{E}\left[ \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot A^\pi(s_t, a_t) \right]\;}
$$

这是 PPO / GRPO / 几乎所有现代 LLM RL 算法的**共同祖梯度**。剩下的差别就是"怎么估 $A$"和"怎么约束更新幅度"。

### 2.3 GAE：Generalized Advantage Estimator

怎么估 $A(s_t, a_t)$？两个极端：

- **Monte Carlo**：用整条 trajectory 的实际 return 减 $V$，$A_t \approx G_t - V(s_t)$。**无偏**但**高方差**（$G_t$ 包含未来所有 step 的随机噪声）。
- **TD(0)**：用一步 bootstrapping，$A_t \approx r_t + \gamma V(s_{t+1}) - V(s_t)$。**低方差**但**高 bias**（依赖估的不准的 $V$）。

GAE（Schulman 2015）是这两端的**几何加权混合**。先定义 **TD residual**：

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

它本身就是单步 advantage 估计。GAE 把 $k$-step TD residual 按指数加权累加：

$$
\hat A_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
$$

参数 $\lambda \in [0, 1]$ 控制 bias-variance trade-off：

- $\lambda = 0$：$\hat A_t = \delta_t$，纯 TD(0)，**高 bias 低 variance**
- $\lambda = 1$：$\hat A_t = \sum_l \gamma^l \delta_{t+l} = G_t - V(s_t)$，纯 Monte Carlo，**低 bias 高 variance**
- $\lambda = 0.95$（PPO 默认）：经验最优

实现上不是真的算无穷和，而是从 trajectory 末尾**反向递推**：

$$
\hat A_t = \delta_t + \gamma \lambda \, \hat A_{t+1}, \qquad \hat A_T = \delta_T
$$

一遍 $O(T)$ 扫完。同时还能顺手算出 value 的训练目标（"value target"）：

$$
\hat V_t^{\text{target}} = \hat A_t + V(s_t)
$$

它就是 critic 网络要回归的 label——critic loss 用 $(V_\phi(s_t) - \hat V_t^{\text{target}})^2$ MSE 训。

### 2.4 on-policy vs off-policy

按"训练用的 trajectory 是不是当前 policy 采的"分两类：

- **on-policy**：必须用**当前** $\pi_\theta$ 采的 trajectory 来更新 $\theta$。一旦 $\theta$ 变了，旧 trajectory 就理论上"作废"。代表：REINFORCE、A2C、TRPO、**PPO**、**GRPO**。
- **off-policy**：可以用任何（甚至来自其它 policy 或历史 buffer 的）trajectory 更新。代表：Q-learning、DQN、SAC。

LLM RLHF 几乎全 on-policy。原因：（1）LLM 模型大、采样 trajectory 慢，但 update 也很贵，on-policy 不算亏；（2）off-policy 需要 importance sampling 修正分布偏差，LLM 这种长序列高维 action 上 IS ratio 方差爆炸，工程上不好做。

但**严格 on-policy 数据效率太低**——一个 batch trajectory 用一次就丢，太浪费。PPO 的关键 trick 就是**用 importance ratio 在"几乎 on-policy"的范围内多复用 K 个 epoch**：

$$
\rho_t = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_\text{old}}(a_t \mid s_t)}
$$

只要 $\rho_t$ 离 1 不太远（PPO 用 clip 强行约束），就当作 on-policy 处理。**这是"LLM RLHF 一个 batch 用 4 个 epoch"背后的合法性来源**——不是凑数，是 PPO 算法定义里就这样设计的。具体在 9.3 详讲。

### 2.5 几个常见混淆点的辨析

- **reward vs advantage**：reward 是环境给的瞬时反馈；advantage 是用 baseline 修正过的、用来更新 policy 的"信号"。policy gradient 里乘的是 advantage，**不是 reward**。
- **return vs reward**：return 是 reward 沿时间累加（$G_t = \sum_k \gamma^k r_{t+k}$）；reward 是单步标量。
- **value vs Q value**：$V(s)$ 不固定 action（对 action 取期望），$Q(s, a)$ 固定具体 $a$。$A = Q - V$。
- **$\gamma$ vs $\lambda$**：$\gamma$ 是 reward discount（"未来 reward 折现"），定义在 return 公式里；$\lambda$ 是 GAE 的 bias-variance 旋钮，**只在算 advantage 时用**。两者数学上独立。

---

## 3. 最小代码示例

### 3.1 REINFORCE on CartPole（35 行内）

完整 policy gradient 循环：取样 trajectory → 算 return → backward。

```python
import gym, torch, torch.nn as nn
import torch.nn.functional as F

env = gym.make("CartPole-v1")                                   # state 4 维, action 2 个
policy = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 2))
opt = torch.optim.Adam(policy.parameters(), lr=1e-2)

def sample_trajectory():
    s, _ = env.reset(); log_probs, rewards = [], []
    for _ in range(500):
        logits = policy(torch.as_tensor(s, dtype=torch.float32))
        dist   = torch.distributions.Categorical(logits=logits)
        a      = dist.sample()                                  # 按 π 采样动作
        log_probs.append(dist.log_prob(a))                      # 记录 log π(a|s)
        s, r, term, trunc, _ = env.step(a.item())
        rewards.append(r)
        if term or trunc: break
    return torch.stack(log_probs), rewards

def compute_returns(rewards, gamma=0.99):
    G, returns = 0.0, []
    for r in reversed(rewards):                                 # 从末尾倒着累加
        G = r + gamma * G
        returns.insert(0, G)
    return torch.tensor(returns, dtype=torch.float32)

for epi in range(500):
    log_probs, rewards = sample_trajectory()
    returns = compute_returns(rewards)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)   # baseline + normalize
    loss = -(log_probs * returns).sum()                              # REINFORCE 梯度
    opt.zero_grad(); loss.backward(); opt.step()
    if epi % 50 == 0: print(f"ep {epi}  return={sum(rewards):.0f}")
```

关键三处：
- `dist.log_prob(a)` 等于 $\log \pi_\theta(a \mid s)$，policy gradient 公式里要对它求导
- `returns.mean()` 当作 baseline——这是 GRPO 用 group mean 当 baseline 的祖先思想
- loss 前面的负号——因为 PyTorch 默认**最小化** loss，而 RL 是**最大化** $J(\theta)$

### 3.2 GAE 计算函数（20 行内）

```python
import torch

def compute_gae(rewards, values, gamma=1.0, lam=0.95, last_value=0.0):
    """
    rewards: list[float] 长度 T
    values:  list[float] 长度 T，对应每个 state 的 V(s_t)
    last_value: V(s_T)，bootstrap 用；trajectory 终止时为 0
    返回: advantages 与 returns 各长度 T
    """
    T = len(rewards)
    advantages = [0.0] * T
    gae = 0.0
    for t in reversed(range(T)):                                 # 从末尾反向递推
        next_v = values[t + 1] if t + 1 < T else last_value
        delta  = rewards[t] + gamma * next_v - values[t]         # TD residual
        gae    = delta + gamma * lam * gae                       # GAE 递推
        advantages[t] = gae
    returns = [a + v for a, v in zip(advantages, values)]        # value target = adv + V
    return torch.tensor(advantages), torch.tensor(returns)

# 在 LLM RLHF 里：rewards 通常只有最后一位非零（trajectory-level reward）
# 例：T = 50 token，reward = [0]*49 + [3.5]，gamma=1.0，GAE 把 3.5 摊回前面每个 token
```

注意 **LLM 视角的特殊性**：rewards 几乎全是 0，只有末尾一个非零（来自 reward model）。GAE 递推会自动把这个 trajectory-level reward "spread" 到前面每一步——每个 token 拿到的 advantage 是一个被 $\gamma\lambda$ 衰减后的版本。这正是 PPO 在 LLM 上能 work 的关键工程细节。

### 3.3 LLM 视角的 policy 与 log_prob（15 行内）

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
input_ids = tok("1+1=", return_tensors="pt").input_ids                   # [1, T]

logits = model(input_ids).logits                                         # [1, T, V]  V=vocab 大小
log_probs = torch.log_softmax(logits, dim=-1)                            # 对每个位置算 log π(·|s_t)
# 对应位置的 log π(a_t|s_t)：a_t 就是下一个 token，索引取出来即可
next_token_log_prob = log_probs[0, -1]                                   # [V]，所有候选 token 的 log_prob
sampled = torch.distributions.Categorical(logits=logits[0, -1]).sample() # 按 π 采一个 token
# 假设这条 generation 末尾从 reward model 拿到 r=2.5，那么 advantage 就用它（减 baseline）
# policy gradient = - log π(sampled | s_t) * advantage  ——形式与 CartPole 一模一样
```

**两个极简但本质的事实**：
1. LLM 的 forward 输出 logits → softmax → 就是 policy 分布，"action a" = "下一个 token id"
2. RL 要的 $\log \pi(a \mid s)$ 就是 `log_softmax(logits)` 在对应 token 上取值——跟语言建模 loss 用的是**同一个量**。这就是 RLHF 跟 SFT 共用同一份 forward 代码、只换 loss 形式的根本原因。

---

## 4. 工程踩坑与经验

- ❗ **LLM RL 的 reward 是 trajectory-level，不是 step-level**——RM 对一整条 generation 打一个分，不是给每个 token 打分。用 GAE 时通常做法：把 reward vector 设成 `[0, 0, ..., 0, r_T]`，然后让 GAE 自动 spread 到每个 token；千万别错误地把 RM 分数广播到每一位。process reward model（PRM，10.2 讲）才是每步打分，那是另一个故事。
- ❗ **baseline 用 critic 估 $V(s)$ 时，critic 训不准就会让 advantage 错**——这是个连锁反应：critic 估错 → advantage 错 → policy 朝错方向走 → 采到的 trajectory 分布变 → critic 更难学。LLM 上 critic 一般是从 policy clone 一份再加 value head（参数量翻倍、显存压力翻倍）。**GRPO 的核心动机就是去掉 critic**——用同一个 prompt 的 group 内 reward 均值当 baseline，绕过 critic 漂移的问题。
- ❗ **GAE 的 $\lambda = 0.95$ 是经验值，对 LLM 短 trajectory 可以更小**——0.95 是 Schulman 在 MuJoCo 几百步连续控制任务上调出来的；LLM 上 trajectory 几百到几千 token、reward 又只有末尾一次，$\lambda$ 调到 0.9 甚至 0.5 在某些设定下更稳。OpenRLHF / verl 等框架默认仍是 0.95，但调参时不要把它当神圣常数。
- ❗ **on-policy RL 数据效率低，每条 trajectory 用 1 次很浪费**——PPO 的 trick 是在同一批 rollout 上跑 K 个 epoch、每 epoch 内分 mini-batch，用 importance ratio + clip 保持"近似 on-policy"。LLM RLHF 实战 K 通常 1-4：太大 ratio 偏离会触发 clip 失效（梯度变 0），太小数据浪费。**ratio 必须实时 log**——监控分布是否压在 $[1-\epsilon, 1+\epsilon]$ 区间内是 PPO debug 的第一指标。
- ❗ **不要混淆 reward 与 advantage**——reward 是环境/RM 给的原始信号，advantage 是用 baseline 修正过的更新信号。policy gradient 公式里乘的是 **advantage**，不是 reward；如果代码里直接用 reward 做 policy gradient 会有方差爆炸 + 数值漂移。GRPO 的 group-relative advantage 本质也是一种 baseline（"减掉这个 prompt 上的 group mean reward"）。
- ❗ **经典 RL 的 $\gamma$ 在 LLM RLHF 里通常 = 1**——不要从 RL 教材带来 $\gamma = 0.99$ 的肌肉记忆。LLM 任务长度有限（几百到几千 token）、reward 一次性给在末尾，没必要再 discount。$\gamma = 0.99$ 用在 LLM 上反而会让 trajectory 早期 token 拿到的 advantage 被严重衰减、信号变弱。
- ❗ **经典 RL 的"state space"在 LLM 是 token 序列，不是显式 vector**——别被 Sutton 教材里"$s_t \in \mathbb{R}^n$"卡住，LLM 里 $s_t$ 就是"截至此刻已生成的 prompt + tokens"，是个**变长字符串**。所以 LLM RL 没有显式的 state encoder——transformer forward 本身就是 state encoder。这也是为什么 critic 通常直接复用 LLM backbone + 一个标量 head，没有独立的 value 网络架构。
- ❗ **policy gradient 是 maximize，PyTorch 是 minimize loss**——所有 RL 代码都要在 loss 前面加负号 `loss = -(log_probs * advantages).mean()`，这是新手第一天最容易忘的。如果训完发现 policy 越训越差，先检查这个负号。

---

## 5. 经典 paper

- **Sutton & Barto, 2018 — *Reinforcement Learning: An Introduction*（书，2nd ed.）** — RL 圣经，不是 paper 是教材。本节所有概念的"权威定义"都在第 3-13 章。读 ch 3（MDP）、ch 6（TD learning）、ch 13（policy gradient methods）三章足够覆盖 LLM RL 需要的所有 RL 基础；ch 13 的 REINFORCE + Actor-Critic 推导与本节 §2.2 完全对应。
- **Schulman, Moritz, Levine, Jordan, Abbeel, 2015 — *High-Dimensional Continuous Control Using Generalized Advantage Estimation*** — GAE 原典，本节 §2.3 公式直接引自该 §3。读 §2-§3 推导 GAE 与 $\lambda$ 的 bias-variance 解释；这是后续 PPO / GRPO / RLHF 工程实现里 advantage 计算的事实标准，不读这篇就读不懂任何 RL 框架的源码。
- **Mnih et al., 2016 — *Asynchronous Methods for Deep Reinforcement Learning*（A2C/A3C）** — 把 Actor-Critic 工业化的关键 paper。读它能搞清"为什么需要 critic、critic 怎么和 actor 一起训、为什么 advantage 比 return 好"——A2C 是 PPO 的直接前身，公式上 PPO 就是 A2C 加了 ratio clip。
- 选读：**Williams, 1992 — *Simple statistical gradient-following algorithms for connectionist reinforcement learning*** — REINFORCE 原典（早 Sutton 教材 26 年）。3 页讲清 log-derivative trick 与 score function estimator——本节 §2.2 推导的源头。

---

## 6. 自测与面试题

**Q1（概念）：** 用 1-2 句话解释什么是 advantage？为什么用它代替 return 做 policy gradient？

<details>
<summary>Answer sketch</summary>

要点：

- **定义**：$A(s, a) = Q(s, a) - V(s)$，"在 state $s$ 选 action $a$ 比该 state 上的平均决策好多少"
- **为什么不用 return**：return $G_t$ 包含**与当前动作无关的环境运气成分**——本来就是高 value 的 state，无论选什么 action 都拿高 return，policy 不该因此邀功；减 baseline $V(s)$ 就把这部分剥离了
- **数学上**：减任何不依赖 $a$ 的 baseline 都不改变梯度的期望（因为 $\mathbb{E}_a[\nabla \log \pi(a|s)] = 0$），但**显著减小方差** → 训练更稳、收敛更快
- 加分：advantage 也可以理解为"对 action 的反事实评价"——比"如果用 $\pi$ 在 $s$ 上随便选"好多少
- 加分：GRPO 的 group-relative advantage 本质也是个 baseline——用同 prompt 的 group mean reward 减掉

</details>

**Q2（公式）：** 写出 GAE 的递推公式，并解释 $\lambda \to 0$ 与 $\lambda \to 1$ 两个极端的物理意义。

<details>
<summary>Answer sketch</summary>

公式：

- TD residual: $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$
- GAE 定义: $\hat A_t^{\text{GAE}} = \sum_{l=0}^\infty (\gamma\lambda)^l \delta_{t+l}$
- 实现用反向递推: $\hat A_t = \delta_t + \gamma\lambda \hat A_{t+1}$，$\hat A_T = \delta_T$

两个极端：

- $\lambda = 0$：$\hat A_t = \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$，**纯单步 TD**——只用一步真实 reward + bootstrap 的 $V$；**低 variance 高 bias**（依赖 $V$ 估的不准）
- $\lambda = 1$：$\hat A_t = \sum_l \gamma^l \delta_{t+l} = G_t - V(s_t)$（$\delta$ 望远镜消项展开），**纯 Monte Carlo**——用整条 trajectory 的实际 return 减 baseline；**低 bias 高 variance**
- 中间值（如 0.95）：bias 与 variance 的折中，PPO 实战默认
- 加分：在 LLM RLHF 里 reward 几乎全是 0、只有末尾非 0，这时 GAE 实际上把末尾 reward 用 $(\gamma\lambda)^l$ 衰减地分摊到每个 token

</details>

**Q3（LLM 映射）：** 在 LLM RLHF 里，state / action / reward 分别对应什么？以一道 GSM8K 数学题"Janet 的鸭子每天下 16 个蛋……"为例，具体说明。

<details>
<summary>Answer sketch</summary>

抽象映射：

- **state $s_t$**：当前已生成的 token 序列 = prompt + 已采样的 $t$ 个 token
- **action $a_t$**：下一个 token，从 vocab（约 50k 大小）里选一个
- **reward $r_t$**：通常 $t < T$ 时为 0，$t = T$ 时为 reward model（或 verifier）打的标量分

GSM8K 具体例子（假设 prompt = 题面，response 是模型生成的 CoT + 答案）：

- $s_0$ = "Janet 的鸭子每天下 16 个蛋…总共能赚多少？\n"（prompt 本身）
- $a_0$ = 模型采的第一个 token，比如 "Let"
- $s_1$ = prompt + "Let"
- $a_1$ = 下一个 token，比如 "'s"
- ...一路采到 $a_T$ = "<eos>"（或回答完毕）
- 整条 trajectory $\tau$ = prompt + 完整 CoT + 答案
- **reward $r_T$**：① 若用 RLHF + RM：RM 对整条 response 打个偏好分（如 2.3）；② 若用 RLVR（10.3）：把模型答的最终数字与 ground truth 18 对比，对了 $r = 1$ 错了 $r = 0$；③ 中间 $r_0, ..., r_{T-1}$ 通通 0

加分要点：

- LLM 的 action space 是 50k+ 词表，远大于经典 RL 的几个动作；reward 极稀疏（一条 trajectory 一个标量）；trajectory 长（几百-几千 token）。这"超大 action space + sparse reward + long trajectory"组合是 LLM RL 比经典 RL 更难的根本原因
- 进阶：PRM（process reward model，10.2）会给每个 reasoning step 打分，相当于把稀疏 reward 变 dense；这是 process supervision 的基本动机

</details>

---

## 7. 延伸阅读

- [Spinning Up in Deep RL（OpenAI）](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) — Joshua Achiam 写的 RL 入门，把 policy gradient / GAE / PPO 推导讲得最清楚的免费资源；本节 §2 思路与该教程的 "Intro to Policy Optimization" 完全一致
- [CleanRL](https://github.com/vwxyzjn/cleanrl) — 单文件 RL 实现，REINFORCE / A2C / PPO / GAE 各一个 .py、可直接 diff 学习；本节 §3 风格借鉴自此
- [Hugging Face Deep RL Course](https://huggingface.co/learn/deep-rl-course) — 互动式 RL 课程，从 Q-learning 一路讲到 PPO，配 Gym 环境实验
- [Lilian Weng — Policy Gradient Algorithms（博客）](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/) — REINFORCE / A2C / TRPO / PPO / SAC / IMPALA 的算法谱系图，公式整齐对照
- [Schulman 博士论文 — Optimizing Expectations](https://www.cs.berkeley.edu/~jschulman/papers/thesis.pdf) — GAE 与 PPO 的提出者博士论文，第 3 章是 GAE 完整推导
- 推荐继续读本教程的 **9.2 节《Reward Model：Bradley-Terry + 训练实操》**——本节最后那个"reward $r_T$ 从哪来"的问号，下一节回答
