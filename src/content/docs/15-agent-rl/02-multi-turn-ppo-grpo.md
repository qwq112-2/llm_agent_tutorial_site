---
title: "15.2 多轮 PPO/GRPO：trajectory-level reward 与归因"
description: "把 9.5 的 single-turn GRPO 推广到 multi-turn agent：trajectory 由 `assistant ↔ tool` 多轮交替构成，reward 在 trajectory 末尾给一次，关键工程点是 loss mask（tool observation 不算 loss）+ trajectory-level advantage 在 assistant token "
---

> ⏱ 预计阅读 65 分钟 ｜ 难度 ★★★ 🔥 ｜ 前置：9.5 GRPO、14.3 Tool Use 训练

## 一句话本节讲什么

把 9.5 的 single-turn GRPO 推广到 multi-turn agent：trajectory 由 `assistant ↔ tool` 多轮交替构成，reward 在 trajectory 末尾给一次，**关键工程点是 loss mask（tool observation 不算 loss）+ trajectory-level advantage 在 assistant token 上 broadcast + generate/train 解耦的 rollout infra**——这是 Search-R1 / SWE-RL / Agent-R1 这类 2025 年新一代 agent 模型的算法底座。

---

## 1. Mental model（直觉）

9.5 的 GRPO 处理的是"single-turn"场景：给 prompt $x$，policy 一次 generate response $y$，对 $y$ 整体打一个 reward $r$，组内归一化做 advantage——整个 trajectory 就是一段连续 token，没有"环境插话"这一说。

multi-turn agent RL 的世界长得完全不同。一条 trajectory 是这样的：

```
[system prompt + user query]                        ← 输入 (mask 掉)
  ├── [assistant turn 1: thought + <tool_call>]     ← model 生成 (要算 loss)
  ├── [tool observation 1: search 结果]             ← 环境注入 (mask 掉!)
  ├── [assistant turn 2: thought + <tool_call>]     ← model 生成 (要算 loss)
  ├── [tool observation 2: page content]            ← 环境注入 (mask 掉!)
  ├── ...
  └── [assistant turn N: final answer]              ← model 生成 (要算 loss)
                                                       ↓
                                            verifier(final answer) → r ∈ {0, 1}
```

工程上多了 4 件 single-turn RL 不必处理的麻烦事：

1. **Loss mask 必须精确到 segment**——tool observation 那一段 token 是环境/外部 API 写入的，**不是 policy 生成的**。如果在那一段也算 policy gradient loss，model 会学着"复读 observation"，因为这些 token 在 trajectory 里 ground truth 的位置上，model 会被训练去预测它们。这是 multi-turn agent RL 的 day-1 bug，做错一次整次实验全废。
2. **Reward 极度稀疏**：trajectory 长度可能 8k-32k token，reward 只在末尾给一个 0/1。如果按 9.1 GAE 那套 per-token 估计，critic 估不准的程度比 single-turn PPO 高一个量级——所以 multi-turn 场景下 GRPO 的"trajectory-level advantage"反而比 PPO 更合适：直接把 $\hat A$ broadcast 到所有 assistant token，简单、稳定、可解释。
3. **Trajectory 长度方差极大**：有的 task 2 turn 解决（`search → answer`），有的要 15 turn（反复尝试、错误恢复）。同一个 batch 里短的 1k token、长的 30k token，padding 浪费严重，packing 成为必需。
4. **Tool execution 不可控**：tool 调用要走真实环境（HTTP API、code sandbox、browser），可能慢（秒级到分钟级）、可能失败（网络抖、API 限流）、可能 hang。**generate 与 tool execution 必须 async + sandboxed + timeout**，否则训练循环被一个超时 tool 卡住。

mental model 一句话：**multi-turn agent RL = single-turn GRPO + loss mask + 复杂 rollout infra**。算法骨架完全没变，但工程难度上了一个台阶——这也是为什么 verl / OpenRLHF 这种"有 multi-turn rollout 引擎"的框架在 2025 年成为主流，而不是早期 TRL。

```
single-turn GRPO（9.5）              multi-turn agent GRPO（本节）
─────────────────────                ─────────────────────────────
prompt → generate y → reward(y) → A  prompt → multi-turn rollout
                                       ├── assistant ↔ tool ×N
                                       └── final → verifier → r
batch 内 trajectory 都一样长          batch 内长度 1k-30k 都有
所有 token 都算 loss                  必须 segment 级 loss mask
generate 是同步推理                   generate 必须 async + sandbox
```

记住这个对比表，下面所有内容都是对这 4 条工程难点的具体应对。

---

## 2. 公式与原理

### 2.1 多轮 trajectory 的形式化

设一条 multi-turn agent trajectory $\tau$ 由 $T$ 个 token 组成，每个 token $\tau_t$ 都属于以下 3 类 segment 之一：

- $S_{\text{prompt}}$：system prompt + user query + tool definitions（输入，不算 loss、不属 policy）
- $S_{\text{asst}}$：assistant 生成的内容（thought / tool_call / final answer，**算 loss、属 policy**）
- $S_{\text{obs}}$：tool execution 返回的 observation（环境注入，**不算 loss、不属 policy**）

定义 segment mask $m_t \in \{0, 1\}$：

$$m_t = \begin{cases} 1 & \tau_t \in S_{\text{asst}} \\ 0 & \tau_t \in S_{\text{prompt}} \cup S_{\text{obs}} \end{cases}$$

policy 在 trajectory 上的对数似然只对 $m_t = 1$ 的 token 求和：

$$\log \pi_\theta(\tau) = \sum_{t=1}^{T} m_t \cdot \log \pi_\theta(\tau_t \mid \tau_{<t})$$

注意 prefix $\tau_{<t}$ 包含**所有**之前的 token（含 prompt 和 observation）——model 必须看到 observation 才能生成下一轮 assistant token，但生成 observation 本身**不应该被 train**。这一点在 §3.1 的 mask 实现里要再次强调。

### 2.2 Trajectory-level reward 与 broadcast

在 multi-turn agent RL 主流做法里，reward 只在 trajectory 末尾给一次：

$$R(\tau) = \text{verifier}(\text{final\_answer}(\tau)) \in \mathbb{R}$$

最常见是 0/1 outcome reward（task 成功 = 1、失败 = 0），也可以是 partial credit（unit test 通过比例）或 composite（success + format + length penalty）。

**关键问题：怎么把这一个标量 $R$ 归因到 trajectory 内每个 assistant token 的 advantage？** 三种主流方案：

**方案 A — Constant broadcast（最简单、最主流）**：trajectory 内所有 assistant token 共享同一个 advantage：

$$\hat A_t = \hat A(\tau), \quad \forall t \text{ s.t. } m_t = 1$$

其中 $\hat A(\tau)$ 由组内归一化得到（同 9.5）。这是 GRPO 在 multi-turn 上的"零修改"扩展，也是 Search-R1 / SWE-RL / Agent-R1 默认做法。

**方案 B — Discount γ broadcast**：仿 GAE 思路，把 $R$ 按时间折扣回每 token：

$$\hat A_t = \gamma^{T-t} \cdot R, \quad \forall t \text{ s.t. } m_t = 1$$

直觉是"越接近 final answer 的 token credit 越多"。实操中 LLM agent RL 几乎不用——一是 reward sparse + trajectory 长，$\gamma^{T-t}$ 在前期 token 上几乎为 0，等于这些 token 不学；二是没有理论根据说明 final answer 之前的 token 应该被折扣（你 search 的关键 query 也很重要）。

**方案 C — PRM step reward**：用 Process Reward Model（10.2）对每一 turn 单独打分，把 reward 拆到 step 级别。理论上更 dense，但 PRM 训练数据稀缺、PRM 自身漂移风险大。Verifiable reward 时代（2025+）几乎被弃用。

**结论**：multi-turn LLM agent RL 默认用方案 A——**trajectory-level reward + 组内归一化 + 在所有 assistant token 上 broadcast 同一个 $\hat A$**。简单、稳定、与 9.5 GRPO 同构。下面所有公式都按方案 A 写。

### 2.3 Multi-turn GRPO 完整目标

设 prompt（task）$x$，policy 采样 $G$ 条 trajectory $\{\tau_1, \dots, \tau_G\}$（每条是完整的 multi-turn rollout），每条 trajectory 末尾打 reward $r_i = R(\tau_i)$。组内归一化得 trajectory-level advantage：

$$\hat A_i = \frac{r_i - \bar r}{\sigma_r + \epsilon_{\text{stab}}}, \qquad \bar r = \frac{1}{G}\sum_i r_i, \quad \sigma_r = \text{std}(r)$$

把 $\hat A_i$ broadcast 到 trajectory $\tau_i$ 内所有 assistant token（即 $m_{i,t} = 1$ 的 token）。设 $\rho_{i,t}(\theta) = \pi_\theta(\tau_{i,t} \mid \tau_{i,<t}) / \pi_{\theta_{\text{old}}}(\tau_{i,t} \mid \tau_{i,<t})$，

$$
\boxed{
L_{\text{multi-GRPO}}(\theta) = \mathbb{E}\left[ \frac{1}{G}\sum_{i=1}^G \frac{1}{\sum_t m_{i,t}} \sum_{t=1}^{T_i} m_{i,t} \cdot \min(\rho_{i,t}\hat A_i, \; \text{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon)\hat A_i) \right] - \beta \cdot D_{\text{KL}}^{\text{masked}}
}
$$

与 9.5 §2.3 的式子比较，**只多了一个 $m_{i,t}$ 因子**——这一个 mask 把 tool observation segment 完全屏蔽出 loss 与梯度。aggregation 上分母用 $\sum_t m_{i,t}$（assistant token 数）而非 $|y_i|$（trajectory 总长），同样是为了不被 observation 长度污染。

KL 项也要 masked——只在 assistant token 上计算 $D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$，observation token 的 KL 没有意义（model 不输出这些 token）：

$$D_{\text{KL}}^{\text{masked}} = \frac{1}{\sum_t m_{i,t}} \sum_{t} m_{i,t} \cdot \left( \frac{\pi_{\text{ref}}}{\pi_\theta} - \log \frac{\pi_{\text{ref}}}{\pi_\theta} - 1 \right)$$

（k3 估计同 9.5）

### 2.4 与 single-turn GRPO 的差异表

| 维度 | Single-turn GRPO（9.5） | Multi-turn GRPO（本节） |
|---|---|---|
| Trajectory 结构 | 一段连续生成 | `assistant ↔ tool` 交替 |
| Trajectory 长度 | 1-4k token，方差小 | 1-30k token，方差大 |
| Reward 给法 | 末尾一次 | 末尾一次（更 sparse） |
| Loss mask | 全 1（response 全算） | segment 级（assistant=1, obs=0） |
| Advantage 形式 | trajectory-level broadcast | trajectory-level broadcast（同） |
| Generate 成本 | 一次 forward generate | 多 turn × tool exec 时间 |
| Rollout infra | 内嵌训练进程也能跑 | 必须 async + sandbox |
| 显存 | 中（context 短） | 高（context 长 + G 倍） |
| 典型 G | 8-64 | 4-16（generate 太贵） |
| 典型 KL coef β | 0.04 | **0.001-0.01**（更小） |
| 代表工作 | DeepSeekMath, R1 | Search-R1, SWE-RL, Agent-R1 |

注意 KL 系数行：multi-turn 的 trajectory 内信号本来就 noisy（reward sparse + trajectory 长 + observation 不可控），KL 必须放小才能让 reward 信号穿透；放大反而 policy 学不动。

---

## 3. 最小代码示例

### 3.1 Loss mask 构造（assistant / tool obs 分段）

```python
# build_loss_mask.py
import torch

def build_assistant_mask(token_ids, tokenizer,
                         asst_open="<|im_start|>assistant",
                         asst_close="<|im_end|>",
                         tool_open="<|im_start|>tool",
                         tool_close="<|im_end|>"):
    """
    扫描 trajectory token_ids，标出 assistant segment 为 1，其余为 0。
    适配 ChatML 风格协议（Qwen / xLAM）；其它协议改 marker 即可。
    """
    text = tokenizer.decode(token_ids)
    mask = torch.zeros(len(token_ids), dtype=torch.long)
    in_asst = False
    # 重新逐 token decode 来定位边界（生产里更常见做法是 generate 时同步记录 segment 偏移）
    pos = 0
    for i, tid in enumerate(token_ids):
        piece = tokenizer.decode([tid])
        pos += len(piece)
        if asst_open in text[max(0, pos-len(asst_open)-5):pos]:
            in_asst = True
        if in_asst:
            mask[i] = 1
        if asst_close in text[max(0, pos-len(asst_close)-5):pos] and in_asst:
            in_asst = False
    return mask  # (T,) 0/1 mask
```

要点：生产实现**不应**靠 decode 后字符串扫描——慢且易错。verl / OpenRLHF 的做法是 rollout 时**同步**记录每个 segment 的 `(start_token_idx, end_token_idx, segment_type)` 元数据，训练时直接按 metadata 写 mask。上面这个版本仅供示意 mask 含义。

### 3.2 Trajectory-level advantage broadcast（10 行）

```python
# trajectory_advantage.py
import torch

def trajectory_advantage(rewards, masks, eps=1e-8):
    """
    rewards: (B, G)        每 task 采 G 条 trajectory 的标量 reward
    masks:   (B, G, T_max) assistant segment mask（padding 部分也是 0）
    返回:    (B, G, T_max) 每 token 的 advantage（assistant token 共享 trajectory advantage，
                          observation / padding 处为 0）
    """
    mean = rewards.mean(dim=1, keepdim=True)              # (B, 1)
    std = rewards.std(dim=1, keepdim=True) + eps          # (B, 1)
    traj_adv = (rewards - mean) / std                     # (B, G)
    # broadcast 到 token 维：(B, G, 1) * (B, G, T_max) → (B, G, T_max)
    return traj_adv.unsqueeze(-1) * masks
```

要点：mask 同时承担两件事——(1) 把 advantage 从 trajectory 级 broadcast 到 token 级；(2) 把 observation/padding token 的 advantage 直接置 0，等价于这些 token 不进 PG loss。**一行乘法搞定 9.5 GRPO 在 multi-turn 场景的核心扩展**。

### 3.3 Multi-turn GRPO 完整 step 骨架（45 行）

```python
# multi_turn_grpo_step.py
import torch
import torch.nn.functional as F

def multi_turn_grpo_step(policy, ref_model, tasks, rollout_fn, reward_fn, optimizer,
                         G=8, beta=0.005, eps_clip=0.2):
    """
    policy:     待训 LLM (π_θ)
    ref_model:  frozen 参考模型 (π_ref)
    tasks:      list[dict]，每个含 user query + tool list + ground truth
    rollout_fn: (policy, task) -> (token_ids, segment_mask) 的 multi-turn rollout
                                  含 assistant ↔ tool ↔ assistant 交替，sandbox 执行
    reward_fn:  (task, trajectory) -> scalar (e.g. 0/1 verifier)
    """
    B = len(tasks)

    # ===== Phase 1: rollout（每 task 采 G 条完整 multi-turn trajectory）=====
    trajectories, masks, rewards = [], [], []
    for task in tasks:
        for g in range(G):
            with torch.no_grad():
                # rollout_fn 内部跑 assistant.generate → execute_tool（sandbox）→ next assistant.generate
                tok_ids, seg_mask = rollout_fn(policy, task)         # (T,), (T,)
                r = reward_fn(task, tok_ids)                         # scalar
            trajectories.append(tok_ids); masks.append(seg_mask); rewards.append(r)

    tok_ids   = pad_stack(trajectories)                              # (B*G, T_max)
    masks     = pad_stack(masks)                                     # (B*G, T_max)
    rewards_t = torch.tensor(rewards).view(B, G)                     # (B, G)

    # ===== Phase 2: trajectory-level group advantage =====
    advantages = trajectory_advantage(rewards_t, masks.view(B, G, -1))  # (B, G, T_max)
    advantages = advantages.view(B*G, -1)                            # (B*G, T_max)

    # ===== Phase 3: log-probs（old / ref / new）=====
    with torch.no_grad():
        log_pi_old = compute_log_probs(policy,    tok_ids)           # (B*G, T_max)
        log_pi_ref = compute_log_probs(ref_model, tok_ids)
    log_pi_new = compute_log_probs(policy, tok_ids)                  # 带梯度

    # ===== Phase 4: PPO clipped objective + masked KL =====
    ratio = torch.exp(log_pi_new - log_pi_old)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantages
    pg_loss_per_tok = -torch.min(surr1, surr2)                       # (B*G, T_max)
    # 关键：mask + 按 assistant token 数归一化（不是按 trajectory 总长）
    pg_loss = (pg_loss_per_tok * masks).sum() / masks.sum().clamp(min=1)

    log_u = log_pi_ref - log_pi_new
    kl_per_tok = torch.exp(log_u) - log_u - 1                        # k3, (B*G, T_max)
    kl = (kl_per_tok * masks).sum() / masks.sum().clamp(min=1)

    loss = pg_loss + beta * kl
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    return {"loss": loss.item(), "pg": pg_loss.item(), "kl": kl.item(),
            "reward_mean": rewards_t.mean().item()}
```

要点对比 9.5 §3.2 的 single-turn GRPO：
- **`rollout_fn` 替代了 `policy.generate`**——multi-turn 情况下 generate 是个循环（generate → parse tool_call → execute_tool → 把 obs 拼回 → 再 generate），且每个 trajectory 的 tool exec 必须 sandbox + timeout
- **`segment_mask` 是新增第一公民**——advantage broadcast、PG loss aggregation、KL aggregation 三处都要乘 mask
- **`pg_loss.sum() / masks.sum()` 而非 `.mean()`**——`.mean()` 会把 padding 与 observation 的 0 也算进分母，等于把 loss 稀释到 trajectory 总长（呼应 DAPO 的 token-level loss 思想）
- **β 默认 0.005**（远小于 9.5 的 0.04）——multi-turn trajectory 内信号已 noisy，KL 必须松

### 3.4 Async generate-train 解耦（伪代码）

```python
# async_rollout_train_loop.py
# 简化的 generate / train 解耦架构 —— 真实实现见 verl / OpenRLHF
import asyncio

class AsyncRolloutTrainLoop:
    def __init__(self, vllm_actor, train_engine, rollout_queue):
        self.vllm = vllm_actor       # vLLM serve（独立进程 / 独立 GPU）
        self.trainer = train_engine  # PyTorch / DeepSpeed 训练引擎
        self.queue = rollout_queue   # 缓冲已完成 trajectory

    async def rollout_worker(self, tasks):
        """异步采 trajectory，完一条入队一条。tool exec 走 sandbox + timeout。"""
        for task in tasks:
            traj = await self.vllm.multi_turn_generate(
                task, max_turns=10, tool_executor=sandbox_exec, timeout=30
            )
            await self.queue.put(traj)

    async def train_worker(self):
        """从 queue 取够一个 batch (B*G 条) 就 step 一次。"""
        while True:
            batch = await self.queue.get_batch(size=64)
            self.trainer.grpo_step(batch)
            # 周期性 sync policy 权重到 vllm（每 K step 一次）
            if self.trainer.global_step % 10 == 0:
                await self.vllm.update_weights(self.trainer.policy.state_dict())
```

要点：**generate 是 multi-turn agent RL 的瓶颈**（每条 trajectory 几十 tool call 全是网络 + LLM forward）。把 generate 推到 vLLM 独立 serve、train 在另一组 GPU 异步消费 queue，rollout 时间可以与 backward 时间并行。verl / OpenRLHF / TRL 0.13+ 都内置了类似架构。

---

## 4. Tool 输出 / Observation 工程

multi-turn agent RL 的另一个独立工程模块——观测处理。整个 trajectory 的可控性都被它影响。

### 4.1 Observation 截断与摘要

tool 返回内容长度**完全不可控**：

- `web_search` 返回 10 条结果的摘要 → 1k token
- `read_url` 返回整页 HTML → 50k token，远超 context
- `execute_python` 返回大 dataframe 的 print → 任意长

如果不处理，一条 trajectory 在第 3 turn 就把 8k context 撑爆，后续 turn 直接 truncate；32k context 的模型也撑不住几个 `read_url`。三种策略：

- **硬截断**（最简单）：`obs = obs[:max_obs_tokens]`，如 max_obs_tokens=2048。简单粗暴但可能丢关键信息
- **摘要化**：用一个小 model（甚至 policy 自己）总结长 obs 成 N token 摘要再注入。质量更好但增加 forward 次数
- **结构化抽取**：tool 设计阶段就只返回结构化数据（如 `{"title": ..., "snippet": ...}` × top-5）而非 raw HTML

工业实操：**对 web / file / db 这类天然长 obs 的 tool，截断 + 结构化抽取双保险**；摘要化主要用在 long-context agent 场景。

### 4.2 Sandbox 与 timeout

tool execution 必须在隔离环境跑，否则训练循环会被各种意外 hang：

- `execute_python` → 死循环 / 无限内存分配 → docker / firejail / nsjail 隔离 + 30s timeout + memory limit
- `web_search` / `read_url` → 网络抖、API 限流 → HTTP timeout + retry with backoff + circuit breaker
- `bash` shell → `rm -rf /` 之类操作 → 容器隔离 + 只读 mount + 限制 syscall
- `db_query` → 全表扫描慢查询 → query timeout + read-only role

任何 timeout 或异常**不能让 rollout worker 挂掉**——优雅降级为 `obs = "[error: timeout]"` 让 model 在 trajectory 内继续，由 RL 学"遇到 error 怎么 recover"。

### 4.3 Mock environment for training

真实 web / API 环境对训练有几个问题：(1) 慢（秒级）(2) 不可复现（结果随时间变）(3) 收费 / 限流 (4) 安全风险。训练阶段建议 mock：

- **API 缓存**：把训练集 task 涉及的 tool call 预先全部 cache 一遍，训练时直接命中缓存
- **Mock tool**：写一个 mock 替代真实 tool，返回预定义的合理输出（用于打 RL pipeline 通）
- **离线 sandbox**：SWE-RL 把 GitHub repo + 测试用例打包成 docker image，训练时离线跑 unit test 拿 reward，比上 GitHub 跑稳定 100×

evaluation 时再切回真实 environment 看 model 在 unseen 环境的泛化能力。

---

## 5. Reward 设计

### 5.1 Outcome reward（最常用）

trajectory 末尾 verifier 给 0/1，最干净：

- **数学**：答案是否等于 ground truth（与 10.3 RLVR 一致）
- **Code**：unit test 通过 → 1 / 失败 → 0（HumanEval / SWE-bench 风格）
- **Web agent**：任务完成度（如"找到正确的航班并加入购物车" → 1）
- **Search agent**：答案与 ground truth 是否 string match 或 LLM judge 通过

优点：信号干净、verifier 不漂移、与 GRPO 组内归一化天然契合。缺点：信号 sparse（trajectory 长但只 1 个 reward 数字）。

### 5.2 Composite reward

实务里很少用纯 0/1，会组合多个信号防止 reward hacking：

$$R(\tau) = R_{\text{success}}(\tau) + \alpha_{\text{fmt}} \cdot R_{\text{format}}(\tau) - \alpha_{\text{len}} \cdot \text{len}(\tau) - \alpha_{\text{tool}} \cdot N_{\text{tool}}(\tau)$$

- $R_{\text{format}}$：trajectory 是否符合 `<thought>...</thought><tool_call>...</tool_call>` 格式（避免输出乱七八糟）
- $-\alpha_{\text{len}} \cdot \text{len}$：长度 penalty，防 length hacking（§7 详谈）
- $-\alpha_{\text{tool}} \cdot N_{\text{tool}}$：tool call 次数 penalty，防 tool spam

权重要谨慎调——format / length 项的权重显著小于 success（如 $\alpha_{\text{fmt}} = 0.1$、$\alpha_{\text{len}} = 0.001$），否则 model 会优先优化 format 而不学 task。

### 5.3 Step reward（PRM）

每 turn 单独打分，更 dense，但 PRM 数据稀缺、PRM 自身漂移风险大。当前 multi-turn agent RL 主流不用，仅在数学 reasoning 长链场景偶尔出现（10.2 PRM）。

### 5.4 Verifier-based reward（与 10.3 衔接）

凡是有"程序可判断对错"的任务都首选 verifier reward——math equality check、unit test runner、SQL 执行结果对比、web 页面状态机检查。verifier 是规则代码不会被 hacked，是 RLVR 范式（10.3）成功的根本。Search-R1 / SWE-RL / WebGPT 全靠 verifier reward。

---

## 6. Multi-turn RL 框架对比

写一个生产级 multi-turn agent RL pipeline 不要从零写——选对框架省 60% 工作量。

| 框架 | 出品方 | Multi-turn 支持 | Async rollout | DAPO | LoRA | 特点 |
|---|---|---|---|---|---|---|
| **verl** | ByteDance | ★★★（一等公民） | ★★★（vLLM ray） | ✓ | ✓ | DAPO 同款，事实工业标准 |
| **OpenRLHF** | OpenLLMAI | ★★ | ★★★（Ray + vLLM） | ✓ | ✓ | 早期支持，文档好 |
| **TRL (HF)** | HuggingFace | ★（GRPOTrainer 部分支持） | ✓（v0.13+） | 部分 | ✓ | 入门好但 multi-turn 弱 |
| **NeMo-Aligner** | NVIDIA | ★★ | ✓ | - | ✓ | 大尺度但学习曲线陡 |
| **TorchTune** | Meta | ★（实验性） | - | - | ✓ | 还在快速迭代 |
| **AReal** | THU | ★★★ | ✓ | - | ✓ | 学术圈轻量 |

**怎么选**：
- 工业 / 生产 / 想跑 DAPO → **verl**（默认推荐）
- 学术 paper / 简单 prototype → **OpenRLHF** 或 **TRL**
- 大规模（千卡级）→ NeMo-Aligner
- 单卡 / colab demo → TRL + unsloth

**verl + vLLM** 是 2025 年 multi-turn agent RL 工程黄金组合，Search-R1 / SWE-RL / 各种 R1-style agent 复现几乎都用这个 stack。

---

## 7. 现代 Agent RL 实例

5 个有代表性的"GRPO + multi-turn + verifiable reward"工作（详细在 15.4 节，本节简介定位）：

- **Search-R1（Hou et al., 2025）**：教 model 学会用 search engine 做开放域 QA。trajectory = `query → search → read → think → answer`，verifier 对比答案与 ground truth。GRPO + outcome reward，base model 是 Qwen2.5-7B-Base，训完后在 NQ / TriviaQA 上显著超过 SFT-only baseline
- **ReSearch / ReTool（2025）**：R1 范式 + RAG / tool。在数学推理基础上融合工具调用（计算器、Python interpreter），扩展 long-CoT 到工具增强 reasoning
- **Agent-R1（清华，2025）**：通用 multi-turn agent RL 框架，基于 verl + GRPO，提供 search / RAG / web 多种环境的开箱即用配置。社区落地率最高的开源 agent RL repo 之一
- **SWE-RL（Pan et al., Meta 2025）**：在 SWE-bench 上训 agent。trajectory = `read_file → edit → run_tests`，verifier = 测试通过率。是"RL 训 software engineering agent"的代表作，证明 GRPO 在 code agent 领域有效
- **WebGPT 风格 web RL**：早期 OpenAI 工作，trajectory = browser action sequence，reward = 答案质量（人工或 LLM judge）。现代复现如 AgentTrek / OpenAgents 多用 GRPO

共同 pattern：**GRPO 算法 + outcome verifier reward + multi-turn rollout + base 模型从 SFT 起步**（很少 R1-Zero 那样直接从 base RL）。下节 15.3 给真实环境 case study，15.4 详讲算法连接。

---

## 8. 工程踩坑与经验

- ❗ **Loss mask 算错（最常见、最致命）**——把 tool observation 段也算进 loss，model 会被训成"复读 observation"——前一 turn search 出什么结果，下一 turn 就抄一遍。表现是训练 loss 看起来一直在降但 eval reward 上不去。**测试方法**：训完一个小 step，让 model generate 一条 trajectory，看是否在 assistant turn 输出 search 结果原文。修复：rollout 时同步记录 segment 边界（不要靠事后字符串扫描），mask 一定要单测过
- ❗ **Long trajectory 显存爆**——8k context × G=16 trajectory 同时 backward = 128k token，连 80GB H100 都吃不消。标配组合：(1) **LoRA only**（不全参），(2) **gradient checkpointing**，(3) **vLLM 离线 generate**（与训练 step 解耦），(4) **DeepSpeed ZeRO-3** 切 weights/optimizer，(5) **packing**（不同长度 trajectory packed 到一起，复用 attention mask）
- ❗ **Tool exec 不 sandbox + timeout 训练直接 hang**——一条 trajectory 调了个慢 SQL 卡 10 分钟，整个 batch 都等它，rollout worker 从 100 cells/s 掉到 1。**铁律**：每个 tool exec 必须有 hard timeout（30-60s）+ 异常优雅降级为 `[error: timeout]`，让 model 在 trajectory 内 recover。verl 的 sandbox 模块用 nsjail + cgroup 限制 CPU/RAM/syscall
- ❗ **Reward 0/1 过 sparse → group_size 必须 ≥ 8 才有 variance**——RLVR outcome reward 只给 0/1。如果 G=4 且任务难度均衡，组内大概率 4 条全 0 或 4 条全 1，advantage 全 0 没梯度。$G \ge 8$ 才能稳定有混合（"3 条对 5 条错"），$G = 16$ 是 multi-turn 的工业甜点（再大 generate 太贵）。配 DAPO 的 dynamic sampling 进一步剔除全 0 advantage 的 group
- ❗ **Async generate 与 sync train 解耦是工程关键**——sync 模式下"generate → train → generate → train"串行，generate 在 multi-turn 下吃掉 80%+ 时间，GPU 利用率 < 30%。async 模式下 generate workers 持续往 queue 灌 trajectory、train workers 持续消费，GPU 利用率上 70%+，端到端时间砍半以上。verl / OpenRLHF 内置，TRL 0.13+ 才支持
- ❗ **Observation 不可控长度必须 truncate / summarize**——`read_url` 返回 50k token 直接撑爆 context，连后续 turn 都 generate 不了。统一在 tool wrapper 层做 `obs = obs[:max_obs_tokens]`，或对 web/HTML 类 tool 强制做结构化抽取（只返 `{title, snippet}`）。max_obs_tokens 默认 2048 是保守值
- ❗ **Multi-turn RL 训练 token 数远大于 SFT，cost 高 10-100×**——SFT 一条数据 1k token，multi-turn RL 一条 trajectory 8k-30k token × G 倍 generate × ppo_epochs。同样训 10k step，RL 烧掉的算力是 SFT 的 50 倍以上。预算评估必须算清楚——别看着 GRPO 算法简单就低估了实际 cost
- ❗ **vLLM + verl 是现代 agent RL 工程主流组合**——vLLM 提供高吞吐 generate 服务（continuous batching + PagedAttention），verl 提供 multi-turn rollout 调度 + sandbox + DAPO 算法 + Ray 异步 actor。这个 stack 是 ByteDance / 阿里 / Search-R1 / SWE-RL 的标配。复现 R1-style agent 论文几乎都在这上面
- ❗ **KL coef 在 multi-turn 上要更小（0.001-0.01）**——9.5 single-turn GRPO 用 0.04，multi-turn 必须再小一个数量级。原因：(1) trajectory 长 + assistant token 多 → KL 累积大，β 不变会压垮 reward 信号；(2) 多轮 reward 本身就 noisy，policy 需要更多自由度探索；(3) DAPO 论文建议甚至到 0.001。β > 0.05 在 multi-turn 上几乎一定学不动
- ❗ **不同 trajectory 长度差异大时 padding 浪费严重**——一个 batch 内最短 1k 最长 28k，按 max len padding 等于浪费 96% 计算。**packing trick**（与 8.2 SFT 同源）：把多条短 trajectory 拼到一起 + block-diagonal attention mask + position id 重置，吞吐能提 3-5×。verl 的 packed batch 模式默认开启
- ❗ **policy 权重要定期 sync 到 vLLM rollout server**——async 架构下 vLLM 用的 weights 总是落后训练 K 步，相当于 off-policy。K 太大（>100 step）advantage 估计漂移、PPO clip 经常 saturate。工业实操每 5-10 step sync 一次，sync 用 NCCL broadcast 而非保存 ckpt 再 load（后者慢 100×）
- ❗ **同 task 的 G 条 trajectory 必须真正独立采样**——常见 bug：rollout 时 G 条共享同一个 random seed → 前缀完全一致，只是末尾分叉，组内多样性退化、advantage 失效。修复：每条 trajectory 独立 sampling seed + temperature ≥ 0.7 + top_p ≤ 0.95
- ❗ **真实环境训练前必须先用 mock 环境跑通 pipeline**——直接上 真实 web / API 训练，pipeline bug + 网络问题 + reward 算错混在一起根本调不出来。先用 mock tool（返回固定输出）把 GRPO step / mask / advantage / KL 全跑通，再切 cached real environment，最后切 live environment。SWE-RL 的 docker sandbox 就是这个路径

---

## 9. 经典 paper

- **Shao et al., 2024 — *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*** — GRPO 算法原典（与 9.5 同源）。本节算法骨架（组内归一化 advantage + PPO clip + k3 KL）完全继承 §4.1，只是 trajectory 从 single-turn response 推广到 multi-turn agent。读 §4.1 + §4.1.3 KL 的工程理由，是理解 multi-turn GRPO 的前提
- **DeepSeek-AI, 2025 — *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*** — RLVR + GRPO 工业胜利。R1-Zero 用 verifier reward + GRPO 从 base model 训出 long-CoT，验证了"trajectory-level reward + 组内归一化"的可扩展性。本节 multi-turn 版本就是 R1 思想加 tool use 维度。读 §2 Approach 看 reward 设计与 KL 处理
- **Hou et al., 2025 — *Search-R1: Training LLMs to Reason and Leverage Search Engines via Reinforcement Learning*** — 第一篇明确把 R1 范式扩展到 multi-turn search agent 的论文。trajectory = `search ↔ retrieve ↔ answer`，loss mask 严格区分 model token 与 search result token，GRPO + outcome verifier reward。读 §3 的 mask 实现 + §4 的 reward 设计——是本节 §3 代码的现实参考
- **Yu et al., 2024 — *DAPO: An Open-Source LLM Reinforcement Learning System at Scale*** — verl 框架的算法核心（同 9.5）。本节的 token-level loss aggregation、dynamic sampling、clip-higher、async rollout 全是 DAPO 的工程建议。读 §3 的 4 个改进 + §4 的 verl 架构图，能直接对接到工业级 multi-turn agent RL
- 选读：**Pan et al., 2025 — *SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution*** — Meta 把 GRPO 用到 SWE-bench 上。trajectory = `read → edit → run_tests`，verifier = test pass rate，全程 docker sandbox。读 §3 的 sandbox 设计 + §4 的 reward shaping，是 multi-turn agent RL 在 code 领域的代表作
- 选读：**Wang et al., 2025 — *Agent-R1*** — 清华出品的通用 agent RL 框架（基于 verl）。读其 GitHub README 与 config 设计，是开源 agent RL 落地率最高的 repo 之一

---

## 10. 自测与面试题

**Q1（公式）：** trajectory-level reward 怎么 broadcast 到每 token 的 advantage？写出两种主流方案，并说明 LLM agent RL 默认选哪种、为什么。

<details>
<summary>Answer sketch</summary>

**方案 A — Constant broadcast**（主流）：
$$\hat A_t = \hat A(\tau) = \frac{R(\tau) - \bar R}{\sigma_R + \epsilon}, \quad \forall t \text{ s.t. } m_t = 1$$

trajectory 内所有 assistant token 共享同一个组内归一化 advantage，observation token 经 mask 置 0。

**方案 B — Discount γ**：
$$\hat A_t = \gamma^{T-t} \cdot R, \quad \forall t \text{ s.t. } m_t = 1$$

仿 GAE 思路，越接近 final answer 的 token credit 越多。

**LLM agent RL 默认 A，理由**：
- 简单——零修改沿用 9.5 GRPO 骨架，仅多一个 mask 因子
- 稳定——trajectory-level Monte Carlo baseline 是无偏的，没有 critic 漂移
- 与 sparse outcome reward 完全契合——sparse reward 下 GAE per-token 估计极差，constant broadcast 最稳
- 长 trajectory 上 $\gamma^{T-t}$ 会让前期 token 的 advantage 趋零（等于不学），违反"前期决策也重要"的事实

加分：
- 第三种方案 PRM step reward 在 RLVR 时代基本被弃用（PRM 数据稀缺、PRM 自身漂移）
- DAPO / Search-R1 / SWE-RL 全用方案 A，是 2025 年事实标准
- mask 把 observation token 的 advantage 置 0 等价于"这些 token 不进 loss"，是 multi-turn 与 single-turn GRPO 唯一算法差异

</details>

**Q2（实战）：** 你训一个 search agent，base 选 Qwen2.5-7B-Instruct。列出从 environment + reward 到 GRPO update 的完整 pipeline。

<details>
<summary>Answer sketch</summary>

**0. Base & SFT 起步**
- base = Qwen2.5-7B-Instruct（或先做 search agent SFT 起步，呼应 15.1）
- ref_model = base 的 frozen 副本（用于 KL）

**1. Environment**
- Tool 集合：`web_search(query) → top-5 snippets`、`read_url(url) → page content[:2048]`、`finish(answer)`
- Sandbox：tool exec 走独立 worker，HTTP timeout 30s，超时返回 `[error: timeout]`
- Mock 缓存：训练集所有 query 预先 cache 一份 search 结果，训练阶段命中缓存（速度 10×、可复现）

**2. Reward**
- Verifier：`R(τ) = 1 if string_match(answer, ground_truth) else 0`（NQ / TriviaQA 风格）
- 加 format penalty：`-0.1 if no <thought>...</thought>`
- 加 length penalty：`-0.001 * len(τ)` 防 length hacking
- 加 tool spam penalty：`-0.05 * max(0, n_tool - 5)` 鼓励高效

**3. Rollout**
- 每 task 采 G=8 条 trajectory，max_turns=10，max_obs=2048
- vLLM serve policy（独立 GPU）+ async rollout workers
- 每条 trajectory 同步记录 segment metadata：`[(start, end, "asst" or "obs"), ...]`

**4. GRPO update**
- 组内归一化：`A_i = (r_i - mean(r)) / (std(r) + eps)`
- DAPO 优化：dynamic sampling 剔除 std=0 的 group + token-level loss aggregation + clip-higher (ε_low=0.2, ε_high=0.28)
- Loss mask：assistant=1, obs/system/padding=0；PG loss 与 KL 都按 mask 加权
- KL：k3 估计，β=0.005（multi-turn 必须比 single-turn 小）
- LoRA r=32 + bf16 + grad checkpointing + ZeRO-3

**5. 训练循环（async）**
- rollout workers（vLLM）持续灌 queue
- train workers 凑齐 batch（B=4 task × G=8 = 32 trajectory）就 step
- 每 5-10 step NCCL broadcast policy 权重到 vLLM

**6. Eval**
- 切到 live environment（真实 search engine），eval set 跑 pass@1
- 关键指标：success rate / avg trajectory len / avg tool calls / KL divergence
- ablation：去掉 dynamic sampling / clip-higher / token-level loss 各跑一组

**7. 工程基建**
- Framework：verl + vLLM（默认推荐）或 OpenRLHF
- Logging：W&B 跟踪 reward / KL / pg_loss / mask ratio / generation len
- Checkpoint：每 50 step 存一次（rollback 用）

加分要点：
- 提到 mask 单测（造一条假 trajectory 手算 mask 与代码对齐）是 day-1 必做
- 提到 reward hacking 监控（avg trajectory len 突涨 / tool call 数突涨即 hacking 信号）
- 提到 generate 与 train 解耦、policy 权重 NCCL sync 是工程关键
- 提到先 mock environment 跑通 pipeline，再切 cached real，最后 live

</details>

**Q3（hacking）：** multi-turn agent RL 中 length hacking / tool spam 怎么发生？怎么缓解？

<details>
<summary>Answer sketch</summary>

**Length hacking 怎么发生**
- outcome reward = 0/1，trajectory 越长 → 探索空间越大 → "蒙对"概率提高
- policy 学到的隐含策略：**多写 thought、多调几个 tool、绕一圈再答** → 平均 reward 升高
- 表现：训练曲线上 reward 涨但 trajectory 平均长度从 1.5k 涨到 8k；eval 速度砍半，部署成本暴增

**Tool spam 怎么发生**
- 同样源于 0/1 reward + 探索容错
- policy 学到："不确定就多调几个 tool 兜底" → search 调 5 次、read 10 个 url、最后随便选
- 表现：avg tool calls 从 3 涨到 15，token 消耗暴增，real environment 评测时被 API rate limit 卡

**Reflection hacking（额外考点）**
- 写很多 "let me think again..." "double check..." 但内容空洞
- 长但没信息密度，对 final answer 几乎无贡献
- 表现：trajectory 中 thought 占比从 30% 涨到 70%

**缓解方法**
- **Length penalty**：reward 加 `-α_len * len(τ)`，α_len 取 0.0005-0.005（小于 success 主项 1-2 个数量级）
- **Tool call cost**：reward 加 `-α_tool * max(0, n_tool - threshold)`（如 n_tool > 5 才扣分），鼓励高效
- **Format reward**：用规则 / LLM judge 检查 thought 是否言之有物（呼应 Constitutional AI / LLM-judge）
- **Max budget 强约束**：max_turns / max_tokens 硬上限，超限直接终止 trajectory 并 reward=0
- **Dr.GRPO 的 length de-bias**：去掉 $\frac{1}{|y_i|}$ 让长 trajectory 不被偏好
- **Token-level loss aggregation**（DAPO）：长短 trajectory 公平，避免长 trajectory 单 token 信号被稀释推动 policy 偏好长输出
- **Process Reward**（PRM 10.2）：每 turn 单独评分，让无意义 turn 不拿 reward（成本高、多用于数学）
- **监控指标**：训练日志一定要 track avg_traj_len / avg_n_tool / avg_thought_ratio，三者突涨即 hacking 早期信号

加分要点：
- 与 9.6 reward hacking 章节呼应——agent RL 的 hacking 模式是 single-turn RLHF length bias 在多轮维度的放大版
- 提到 verifier-only reward 比 RM-based reward 抗 hacking 更强（verifier 是规则、不会被骗）
- 提到 Search-R1 / SWE-RL 等工作都报告了 length / tool spam 现象，必须显式 penalty
- 极端情况：reward 设计错了（如 partial credit 给"调过 search 就 +0.2"）会直接训出"逢任务必先 search 5 次"的 policy

</details>

---

## 11. 延伸阅读

- [verl GitHub（ByteDance）](https://github.com/volcengine/verl) — multi-turn agent RL 工业级框架，DAPO + GRPO 默认配置，本节的 multi-turn rollout / sandbox / async 都是 verl 的实现思想
- [OpenRLHF GitHub](https://github.com/OpenRLHF/OpenRLHF) — 早期支持 multi-turn 与 vLLM 解耦的开源 RLHF 框架，文档质量高
- [Search-R1 GitHub & paper](https://github.com/PeterGriffinJin/Search-R1) — multi-turn search agent 的 GRPO 复现，loss mask 与 reward 设计可直接参考
- [SWE-RL paper (Meta)](https://arxiv.org/abs/2502.18449) — code agent 的 multi-turn RL 代表，sandbox 与 verifier reward 实现细节
- [Agent-R1 GitHub（清华）](https://github.com/0russwest0/Agent-R1) — 通用 agent RL 框架，覆盖 search / RAG / web 多种环境的开箱即用配置
- [TRL GRPOTrainer 文档](https://huggingface.co/docs/trl/main/en/grpo_trainer) — TRL 0.13+ 对 multi-turn 的部分支持，入门可用
- [Lambert RLHF Book](https://rlhfbook.com/) — Nathan Lambert 的开源 RLHF 教材，有 multi-turn 与 agent RL 章节
- 推荐继续读本教程的 **9.6 节《工程踩坑：reward hacking / RM 漂移 / KL 坍塌》**——本节 hacking 模式的 single-turn 起源；**15.3《真实环境 RL：SWE-Gym / WebArena / OSWorld / WebGPT》**——真实环境 case study；**15.4《Reasoning + Agent：Search-R1 / ReSearch / ReTool / Agent-R1》**——本节算法到具体工作的完整连接；**15.5《Agent 鲁棒性》**——multi-turn 中的 tool failure / observation perturbation
