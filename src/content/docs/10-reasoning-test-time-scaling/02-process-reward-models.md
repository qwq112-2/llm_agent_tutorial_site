---
title: "10.2 PRM 与 Process Reward（Lightman \"Let's Verify\"）"
description: "ORM 只看最终答案对错给一个标量，PRM 给推理的每一步打分——本节讲清楚 PRM 的训练目标、PRM800K / Math-Shepherd 两条数据路线、如何用 PRM 做 BoN / beam search / RL，以及为什么 DeepSeek-R1 最后选择\"verifiable outcome reward\"而不是 PRM。"
---

> ⏱ 预计阅读 35 分钟 ｜ 难度 ★★★ ｜ 前置：9.2 Reward Model；10.1 CoT / SC / BoN

## 一句话本节讲什么

**ORM 只看最终答案对错给一个标量，PRM 给推理的每一步打分**——本节讲清楚 PRM 的训练目标、PRM800K / Math-Shepherd 两条数据路线、如何用 PRM 做 BoN / beam search / RL，以及为什么 DeepSeek-R1 最后选择"verifiable outcome reward"而不是 PRM。

---

## 1. Mental model（直觉）

回想 9.2：RM 接收 `(prompt, full_response)` 输出一个标量。PPO 在每条 trajectory 末尾收到这一个 reward——长度可能上千 token，但梯度信号只有一根。这种"末端奖励"的形式叫做 **outcome reward**。

> **ORM (Outcome Reward Model)**：(question, full solution) → 一个 reward
> **PRM (Process Reward Model)**：(question, partial solution up to step $t$) → 每一步一个 reward

为什么数学推理 / 长 CoT 要 PRM 不要 ORM？想象一道 5 步推理题，模型在第 3 步算错了符号、第 4 步基于错误结果继续推、第 5 步给出错误答案。

- ORM 只知道"最终答案错"——把 5 步全部当成"坏样本"惩罚，包括完全正确的第 1、2 步。
- PRM 能精确定位"第 3 步是错点"——前两步给正分，第 3 步开始给负分。

这就是 dense reward 的优势：**信号更精细、信用分配（credit assignment）更准**。

```
question: 求 lim_{x→0} (sin x - x) / x^3

ORM:                                                   ← reward
   step1: 用 L'Hopital  →  step2: 错求导  →  step3: 算成 1/3   ❌
                                                       (final reward = -1)

PRM:        +0.9          -0.7         -0.9
   step1 ───────→  step2 ───────→  step3 ───────→  最终错答案
   正确             首次出错        继续传播错
```

但这件事的代价是数据：**ORM 只要标"最终答案对错"（1 bit），PRM 要标"每一步对错"（K bit per sample）**。Lightman 在 OpenAI 拉了一个团队标了 80 万 step（PRM800K），后来 Wang 提出 Math-Shepherd 用 MCTS rollout 自动估 step-level label——但自动 label 噪声 10-20%，不如人工。

PRM 还有一个微妙问题：**它本身是 learned model，policy 学久了会 reward hacking** ——产生"看起来每步都很合理但答案错"的诡异 trajectory。这是 DeepSeek-R1 团队最终弃用 PRM、转向 verifier (math 答案精确匹配 / code unit test) 的核心理由。换句话说：

> 在 verifiable 的 task 上，**rule-based outcome > learned process**；在 non-verifiable 的 task（开放对话、长文写作）上，PRM 仍然是为数不多的 dense supervision 选项。

记一句话：**PRM 是 ORM 的"显微镜版"，提供更密的信号；但显微镜会引入更多噪声、更易被骗、且制造成本极高**。

---

## 2. 公式与原理

### 2.1 形式化定义

设 question $x$ 配一个分步推理 trajectory $\tau = (s_1, s_2, \dots, s_T)$，其中 $s_t$ 是第 $t$ 步推理（一行公式 / 一段自然语言论证）。最终答案 $a$ 从最后一步抽出。

- **ORM**：$r_\phi^{\text{ORM}}(x, \tau) \in \mathbb{R}$，只输出一个标量；训练标签 $y \in \{0, 1\}$ 表示最终答案是否正确。等价于 9.2 的 RM 但去掉 pairwise（因为 outcome label 是 absolute 的）。
- **PRM**：$r_\phi^{\text{PRM}}(x, s_{1:t}) \in [0, 1]$，对每个前缀 $s_{1:t}$ 输出第 $t$ 步的"正确概率"。训练标签 $y_t \in \{0, 1\}$（甚至 $\{-1, 0, 1\}$，对应错 / 中性 / 对）逐步标注。

PRM 的标量训练 loss（per sample，T 步）是 binary cross-entropy 加和：

$$\mathcal{L}_{\text{PRM}}(\phi) = -\sum_{t=1}^{T} \Bigl[y_t \log r_\phi(x, s_{1:t}) + (1 - y_t) \log\bigl(1 - r_\phi(x, s_{1:t})\bigr)\Bigr]$$

实现上用一个 special separator token（如 `\n\n` 或自定义 `<step>`）切出每个 $t$ 的位置，把该位置的 hidden state 过 reward head 即可——一次 forward 输出 T 个标量。

### 2.2 PRM 数据收集：人工 vs 自动

#### (a) 人工标注：PRM800K（Lightman 2023）

OpenAI 团队雇约 80 个标注员/月，对 MATH 数据集的 12k 题、每题约 6 个 GPT-4 generation，**逐步标注每一步的 correct / neutral / incorrect**，最终得到约 80 万 step 标签。这是迄今最大的人工 PRM 数据。

成本估算：80 人 × 月薪 ~$3000-5000 × ~3 个月 = **\$1M-2M 量级**。这是大厂级工程。

#### (b) 自动化：Math-Shepherd（Wang 2024）

人工不可 scale，Math-Shepherd 提出用 **MCTS-style rollout** 估 step-level 价值：

> 给定前缀 $s_{1:t}$，从该前缀往后 sample N 条 completion（用 base model continue 到答案），统计这 N 条的最终答案准确率 $\hat{p}_t$。把 $\hat{p}_t$ 当作第 $t$ 步的 "value"——前缀价值越高，意味着这一步是"在通往正确答案的路径上"。

这是 RL 里 Monte Carlo value estimation 的直接应用，把 PRM 训练和 RL value learning 统一了起来。OmegaPRM（Wang 2024）进一步用更高效的 MCTS 变体（带 PUCT 选择）降低 rollout 数。

**自动化 label 的噪声来源**：

- 一个错的中间步骤后续仍可能"歪打正着"答对（rollout label = 高，但 step 实际错的）
- 一个对的中间步骤后续可能因为 sampling temperature 走偏（rollout label = 低，但 step 是对的）

实证：自动 label 与人工 label 的 step-level agreement 大约 80-90%——够用但不完美。

### 2.3 PRM 在 BoN 中的用法（aggregation 函数）

最直接的用法：generate $N$ 个候选 trajectory，对每个用 PRM 算一个总分，选最高。问题是 PRM 输出的是**每步**分数 $\{r_1, \dots, r_T\}$，要怎么聚合成一个标量？常见三种：

| Aggregator | 公式 | 直觉 | Lightman 实证 |
|------------|------|------|--------------|
| **Last-step** | $r_T$ | 只看最后一步（≈ ORM） | baseline |
| **Min** | $\min_t r_t$ | "短板效应"：任一步坏就否决 | **最佳** |
| **Product** | $\prod_t r_t$ | 假设步间独立 | 与 min 接近 |
| **Mean** | $\frac{1}{T}\sum_t r_t$ | 平均 | 略差，长 trace 被稀释 |

Lightman §4.3 报告 **min 效果最好**——直觉是数学推理只要有一步错整条就废，min 直接对应"trajectory 是否全程正确"的概率。

PRM-guided BoN 在 MATH 上比 ORM-guided BoN 一般高 5-15 个百分点（Lightman §5），且 N 越大差距越大（PRM 在 N=64, 256 仍持续提升，ORM 在 N=16 后趋于饱和）。

### 2.4 PRM 在 search / RL 中的用法

把 PRM 当 value function 接到搜索算法里：

- **PRM-guided beam search**：每一步保留 PRM 分数最高的 top-k beam，扩展下一步。
- **PRM-guided MCTS**：树搜索的 value backup 用 PRM 估而不是随机 rollout，节省大量 LLM call。
- **Process-supervised PPO/GRPO**：PPO 的 per-token reward 用 PRM 在 step 边界处给值，其他 token 给 0。理论上 reward 更 dense → advantage estimation 更准 → 收敛更快。

**但实战中 process-supervised RL 反而少见**，原因见 §2.5。

### 2.5 为什么 DeepSeek-R1 弃用 PRM（必考）

DeepSeek-R1 报告 §2.2.3 明确说"不用 PRM"，给出三个理由：

1. **细粒度的 step 定义本身困难**：什么算"一步"？line break？句号？数学论证里一行 LaTeX？自动化 split 在长 CoT 上极易失误（split 错 → label 错 → PRM 学到的是 noise）。
2. **PRM 自动 label 不可靠**：Math-Shepherd 风格的 rollout label 噪声 10-20%，scale up 后误差会放大。
3. **Reward hacking**：RL 训得越久，policy 越擅长 produce "PRM 喜欢但答案错"的 trajectory。这是 learned reward function 的通病（9.6 详谈）。

DeepSeek-R1 转而用 **rule-based verifier**：

- 数学题：从输出抽 `\boxed{...}` 与 ground truth 字符串精确比对 → reward ∈ {0, 1}
- 编程题：跑 unit test → reward = 通过比例

这种 reward 叫 **RLVR (RL with Verifiable Reward)**，10.3 详讲。它的优势是：

- 100% 可信（rule-based，不存在 model drift）
- 0 标注成本（自动判分）
- 完全 robust to hacking（policy 没法骗一个精确字符串匹配）

代价：只在**有 ground truth**的任务上能用。开放对话 / 长文创作 / 主观偏好仍需 RM 或 PRM。

> **决策树**：
> - 任务有可验证答案（math/code/形式化推理）→ 用 verifier (RLVR)
> - 任务无可验证答案，但需要 dense supervision → PRM
> - 任务无可验证答案，sparse outcome 即可 → ORM (RLHF)

### 2.6 PRM 与 Verifier 的本质区别

| | PRM | Verifier |
|--|-----|----------|
| 实现 | learned neural model | rule / executor |
| 输出 | soft score $\in [0, 1]$ | hard label $\in \{0, 1\}$ |
| 粒度 | per step | per outcome |
| 训练成本 | 极高（80 万 step 标注） | 0（写规则） |
| Hacking 风险 | 高 | 低 |
| 适用任务 | 任意（包括开放任务） | 仅 verifiable task |
| 代表工作 | Lightman 2023, Math-Shepherd | GSM8K verifier (Cobbe 2021), DeepSeek-R1 |

很多文献把"verifier"当 PRM / ORM 的统称，本节统一遵循 DeepSeek 的用法：**verifier = rule-based hard judge**，与 learned reward model 严格区分。

---

## 3. 最小代码示例

以下示例假设已有 SFT model + tokenizer，重点演示 PRM 的核心训练 loss、PRM-guided BoN、PRM 推理打分。

### 3.1 PRM 训练核心 loss

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PRM(nn.Module):
    def __init__(self, base_model, hidden_size, step_token_id):
        super().__init__()
        self.backbone = base_model            # SFT-init 的 transformer
        self.reward_head = nn.Linear(hidden_size, 1, bias=False)
        self.step_token_id = step_token_id    # 用作 step 边界的特殊 token id

    def forward(self, input_ids, attention_mask):
        h = self.backbone(input_ids, attention_mask=attention_mask).last_hidden_state  # (B, T, d)
        # 找出每个 step 边界 token 的位置
        is_boundary = (input_ids == self.step_token_id)                                # (B, T) bool
        step_logits = self.reward_head(h).squeeze(-1)                                  # (B, T)
        return step_logits, is_boundary

def prm_loss(model, batch):
    # batch: input_ids, attention_mask, step_labels (B, T) ∈ {-100, 0, 1}, -100 是非 step 位置
    step_logits, _ = model(batch["input_ids"], batch["attention_mask"])
    labels = batch["step_labels"]                                                      # (B, T)
    mask = labels != -100                                                              # 只在 step 边界算 loss
    loss = F.binary_cross_entropy_with_logits(
        step_logits[mask], labels[mask].float(), reduction="mean"
    )
    return loss
```

关键点：

- 用一个 **step 边界 token**（如 `\n\n` 或自定义 `<step>`）切分推理步骤；boundary 位置的 hidden state 喂给 reward head，得到该步的 logit。
- `step_labels` 是稀疏的——非 step 位置填 `-100` 跳过；step 位置填 0/1 表示该步对错。
- 一次 forward 算所有 step 的 loss——比"每 step 一次 forward" 高效 T 倍。

### 3.2 PRM 推理：给 trajectory 每步打分

```python
@torch.no_grad()
def score_trajectory(prm, tok, question: str, steps: list[str]) -> list[float]:
    """对一条 trajectory 的每一步返回 PRM 概率分。"""
    # 把 question + steps 用 step 边界拼起来，例如：
    #   "Q: ...\n\nStep1: ...\n\nStep2: ...\n\n"
    # 每个 \n\n 之前是一个 step，prm.step_token_id 对应 \n\n 的 token id（视 tokenizer 而定）
    text = f"Q: {question}\n\n" + "\n\n".join(steps) + "\n\n"
    enc = tok(text, return_tensors="pt").to(prm.backbone.device)
    step_logits, is_boundary = prm(enc.input_ids, enc.attention_mask)
    # 取所有 boundary 位置的概率（最后一个 boundary 是 trailing 的，丢掉）
    probs = torch.sigmoid(step_logits[is_boundary]).cpu().tolist()
    return probs[: len(steps)]
```

### 3.3 PRM-guided Best-of-N

```python
def prm_guided_bon(policy, prm, tok, question: str, N: int = 16,
                   aggregator: str = "min") -> str:
    # 1. policy 采样 N 条候选推理 trajectory
    prompt = f"Q: {question}\n\nA: Let's solve this step by step.\n\n"
    inputs = tok(prompt, return_tensors="pt").to(policy.device)
    out = policy.generate(**inputs, max_new_tokens=512, do_sample=True,
                          temperature=0.8, num_return_sequences=N)
    candidates = [tok.decode(o[inputs.input_ids.size(1):], skip_special_tokens=True)
                  for o in out]

    # 2. 对每个 candidate 切 step（按 \n\n / "Step k:" / "##" 等切法，视 prompt 而定）
    def split_steps(text: str) -> list[str]:
        return [s.strip() for s in text.split("\n\n") if s.strip()]

    # 3. 用 PRM 给每个 candidate 打分
    scored = []
    for c in candidates:
        step_scores = score_trajectory(prm, tok, question, split_steps(c))
        if not step_scores:
            scored.append((c, -float("inf"))); continue
        if aggregator == "min":
            agg = min(step_scores)
        elif aggregator == "prod":
            agg = float(torch.tensor(step_scores).log().sum().exp())
        elif aggregator == "mean":
            agg = sum(step_scores) / len(step_scores)
        else:  # last
            agg = step_scores[-1]
        scored.append((c, agg))

    # 4. 选最高分 candidate
    return max(scored, key=lambda x: x[1])[0]
```

关键点：

- `aggregator="min"` 是 Lightman 推荐，直接对应"trajectory 全程正确概率"。
- step 切分的鲁棒性是工程难点——prompt 里强制 `Step 1:` / `Step 2:` 编号会让切分稳定很多。
- N 通常取 16-256；MATH 上 N=64 是个 sweet spot。

---

## 4. 工程踩坑与经验

- ❗ **PRM 训练数据标注成本极高**——PRM800K 是 OpenAI 80 人/月、3 个月、~\$1M-2M 标出来的，初创公司或学校不可能复制。中小团队的可行路线是：(1) 直接用开源 PRM（Math-Shepherd / Skywork-PRM / DeepSeek-Math PRM）做 BoN；(2) 用 Math-Shepherd 风格自动 label 标自有数据，承受 10-20% 噪声。
- ❗ **自动 label (Math-Shepherd) 的噪声有结构性偏差**：一个错的中间步骤后面"歪打正着"答对（label 反而高），一个对的步骤后面 sample 走偏（label 反而低）。要靠加大 rollout 数 N 和 ensemble 缓解，但 N 翻倍 → cost 翻倍。OmegaPRM 用 PUCT-MCTS 选择性扩展节点把 cost 降一半。
- ❗ **PRM 在 RL 训练中极易 reward hacking**：policy 学几千步后会发现"产生看起来逻辑严密、PRM 给高分、但最终答案错"的 trajectory（俗称 "PRM 漂亮的废话"）。一旦发现 RL 中 PRM reward 在涨但 ground-truth accuracy 不涨甚至下跌，立刻停训。常见缓解：(1) PRM + outcome verifier 双 reward 互相约束；(2) 周期性重训 PRM；(3) RM ensemble 取最小（pessimistic reward）。
- ❗ **长 reasoning trace 上 reward 累计有 numerical 问题**：用 product aggregation 时，1000 步每步 0.99 → 总分 $0.99^{1000} \approx 4 \times 10^{-5}$，浮点下溢。一定用 log-prob 累加（`sum(log p_t)`）再 exp，或干脆改用 min。同理 RL 里 step reward 求和时要做 normalization，否则长 trajectory 完全主导 batch 梯度。
- ❗ **outcome reward (verifier) 在 verifiable task 上更 robust + scalable**：DeepSeek-R1 (R1-Zero) 完全用 rule-based verifier 训出来，没有 PRM。verifiable task（math 答案对错 / code unit test 通过率）上 verifier 是 strict dominant 选项——0 标注成本、0 hacking 风险、infinite 可 scale。**只有 non-verifiable task（开放对话、长文创作）上 PRM 才不可替代**。
- ❗ **PRM 的 step boundary 检测是工程难点**：模型实际生成的 trace 不一定按你期望的格式 split（缺 `\n\n`、step 编号错乱、混入代码块）。三个对策：(1) prompt 强制 `Step 1:` / `Step 2:` 编号 + few-shot 示例；(2) SFT 阶段就把 step 格式教进 model；(3) 用 LLM-as-splitter 后处理。前两个更可靠。
- ❗ **PRM 不能跨域使用**：math PRM 不能直接套到 code reasoning 上——step 的语义完全不同（数学每步是一个公式推导，代码每步是函数调用 / 状态变化）。要么按域单独训，要么用 multi-domain PRM 显式条件化。
- ❗ **不要把 PRM 当成 "可解释的 RM"**：PRM 给每步分数看着可解释，但 attention / hidden state 内部仍是黑盒；高分 step 不代表 model "理解"了那一步对，只代表"和训练数据里高分 step 像"。把 PRM 输出作为 debug 信号要谨慎（容易归因错）。

---

## 5. 经典 paper

- **Lightman et al., 2023 — Let's Verify Step by Step** — PRM 的奠基论文，发布 PRM800K 数据集 + 实验证明 process supervision > outcome supervision (在 MATH 上 PRM-guided BoN 比 ORM-guided BoN 高 5-10 个百分点)。读 §3 (data collection) + §4.3 (aggregation 选择) + §5 (实验结果)，本节 §2.1, §2.3 直接基于此。
- **Wang et al., 2024 — Math-Shepherd: Verify and Reinforce LLMs Step-by-step Without Human Annotations** — 自动化 PRM label 的代表作，把 MCTS rollout 当 step value 估计。读 §3 (label 生成) + §4 (与 PRM800K 对比)，理解为什么自动 label 80-90% agree 于人工但仍可用。
- **Cobbe et al., 2021 — Training Verifiers to Solve Math Word Problems** — GSM8K 数据集 + 早期 outcome verifier 思路。这是 RM/PRM/verifier 三者的共同源头：文章里的 "verifier" 是一个 binary classifier 给最终答案打分，正是后来 ORM 的雏形。读它能看清楚 PRM 是怎么从这条 outcome 路线长出来的。
- **DeepSeek-AI, 2024 — DeepSeek-Math: Pushing the Limits of Mathematical Reasoning in Open Language Models** — 开源 Math 模型 + 开源 PRM，工业级 PRM 训练实操参考。报告里同时给了 PRM 的训练细节和 GRPO + RM 的组合方案。
- **DeepSeek-AI, 2025 — DeepSeek-R1** — §2.2.3 明确解释为什么不用 PRM (step 难定义 / 自动 label 不准 / hacking)，转用 rule-based verifier。本节 §2.5 的论点全部出自此处。10.3 会展开 R1 的全流程。
- **Wang et al., 2024 — OmegaPRM: Improved Math Reasoning via Automated Process Supervision** — Math-Shepherd 的高效版，用 PUCT-MCTS 而非朴素 rollout，把 PRM label 生成 cost 降低一半以上。要自己 scale 自动 PRM 数据必读。

---

## 6. 自测与面试题

**Q1（对比）**：从 (训练成本 / signal density / reward hacking 风险 / scalability) 4 个维度对比 ORM 与 PRM，并给出"什么场景选哪个"的决策建议。

<details>
<summary>Answer sketch</summary>

| 维度 | ORM | PRM |
|------|-----|-----|
| **训练成本** | 低（per-trajectory label，1 bit）| 高（per-step label，K bit，PRM800K 量级 \$1M+）|
| **Signal density** | 稀疏（trajectory 末尾一根 reward）| 密集（每步一根 reward）|
| **Reward hacking 风险** | 中（policy 可学冗余啰嗦 / 风格 hack）| **高**（更易"逻辑漂亮但答案错"，因为信号更细）|
| **Scalability** | 高（标注简单 + 可大规模 pairwise）| 低（人工 step 标注难 scale，自动 label 噪声 10-20%）|

决策建议：

- **verifiable task（math/code）+ 大规模 RL** → 直接 verifier (rule-based)，不用 ORM 也不用 PRM。DeepSeek-R1 验证。
- **verifiable task + 小规模 / 推理时 BoN** → PRM-guided BoN（min aggregation），比 ORM-guided 涨 5-15%，Lightman 实证。
- **non-verifiable task（开放对话 / 长文）+ 需要 RL** → ORM (RLHF 经典路线)；如果有充足预算且需要 dense 信号，再考虑 PRM。
- **non-verifiable task + 推理时质量提升** → ORM-guided BoN 即可，PRM 投入产出比不划算。

加分点：现代实践常用 **PRM + verifier 双 reward**——verifier 防止 hacking 兜底，PRM 提供 dense 信号加速收敛。

</details>

**Q2（实战）**：你要做一个 math reasoning 模型，已有 SFT-base + 一个开源 PRM (Skywork-PRM)，写出 PRM-guided BoN 的完整流程，并指出至少 2 个工程实现的细节坑。

<details>
<summary>Answer sketch</summary>

完整流程（5 步）：

1. **Prompt 构造**：用强制分步格式的 prompt，例如 `"Solve step by step. Use 'Step k:' for each step.\n\nQuestion: {q}\n\nStep 1:"`，目的是让 model 输出可被稳定 split 的 trace。
2. **多次 sample**：用 SFT model 以 `temperature=0.7~1.0, top_p=0.9~0.95` sample N 条候选 trajectory（N=16/32/64，根据预算）。
3. **Step 切分**：对每条 trajectory 按 `Step \d+:` regex 或 `\n\n` 切成 step list。**坑 1**：长 trace 容易丢失 `Step k:` 格式或 step 编号错乱，要做容错（fallback 到 `\n\n` 切 + 取奇数行等）。
4. **PRM 打分**：把 (question, step_list) 喂给 PRM，得到每步 score。**aggregation 用 min**（Lightman 实证最佳）；备选 product (取 log 累加防数值下溢) 或 last。
5. **选最优**：取 aggregated score 最高的 candidate 作为最终答案。可选 ensemble：把 PRM-min top-3 candidates 再做 majority vote on extracted answer，进一步提升。

工程坑：

- **坑 1（已点出）**：step 切分鲁棒性。最稳的做法是 SFT 阶段就把 step 格式教进 model（数据里就用 `Step 1:` 编号），inference 时切分基本不出错。
- **坑 2**：PRM 的 chat template / step separator 必须和 PRM 训练时一致——Skywork-PRM 训练用 `\n\n` 当边界、Math-Shepherd 用 `Step k:`，搞错直接掉 10-20% accuracy。每个开源 PRM 的 README 必须细看。
- **坑 3**：N 的甜点不是越大越好。当 PRM 本身有 5-10% noise 时，N=128 后 PRM 的 max-score candidate 可能就是 PRM 偏好的 hacking sample。Lightman §5 的曲线显示 PRM-guided BoN 在 N=256 才趋于饱和，但 noisy PRM 上 N=64 就够。
- **坑 4**：把 final answer 抽出来后做 majority vote，往往比单纯取 PRM top-1 更稳——融合 PRM 的 dense 信号 + Self-Consistency 的 ensemble 信号。
- **坑 5**：PRM 推理 cost 不可忽略——N 个 candidate × T 步 × 一次 forward。实际部署常用小一号的 PRM (1B-7B) 给大 policy (70B+) 评分。

</details>

**Q3（前沿）**：DeepSeek-R1 在 §2.2.3 明确说不用 PRM，给出至少 3 个理由；并讨论 PRM 与 verifier 各自适合什么场景，未来 process reward 还有研究价值吗？

<details>
<summary>Answer sketch</summary>

DeepSeek-R1 弃用 PRM 的 3 个理由（报告原文）：

1. **细粒度 step 定义本身困难**：长 CoT 里"什么算一步"没有清晰定义；自动 split 在长 trace 上极易失误，错的 split 喂给 PRM 训练就是 noise。
2. **自动 label 不可靠**：Math-Shepherd 风格的 rollout label 噪声 10-20%；scale up 后误差放大，PRM 学到的就是 noisy proxy。
3. **Reward hacking**：RL 训得越久，policy 越擅长"骗 PRM"——产生 PRM 高分但答案错的 trajectory；这是所有 learned reward function 的通病。

各自适合的场景：

- **Verifier 适合**：verifiable task（math 精确匹配 / code unit test / 形式化推理）+ 大规模 RL 训练（reward 100% 可信、0 标注成本、0 hacking）。这是 RLVR 路线，DeepSeek-R1 / Kimi-1.5 验证。
- **PRM 适合**：(a) non-verifiable task 需要 dense supervision（开放对话、长文创作）；(b) 推理时 BoN / search（即使有 verifier，PRM 仍能在中间步骤剪枝）；(c) 多轮 agent 的 step-level credit assignment（每个 tool call 这一步好不好），延伸到 Module 15 的 multi-turn agent RL。

PRM 未来的研究价值（讨论）：

- **核心价值仍在 non-verifiable domain**：长文创作、开放对话、agent multi-step、creative reasoning 这些任务没有 ground truth verifier，只能靠 learned model；PRM 仍是 dense supervision 的为数不多选项。
- **改进方向**：(1) 更鲁棒的 step 定义（从 token-level CoT 到 segment-level CoT）；(2) 更准的自动 label（OmegaPRM / 蒸馏更强 model 当 oracle）；(3) PRM + verifier hybrid（verifier 兜底防 hacking，PRM 给中间步骤梯度）；(4) PRM 用作 search heuristic 而不是 RL reward，绕开 hacking 问题。
- **PRM 已死说？** 在 verifiable domain 几乎被 verifier 取代；但在 verifiable + non-verifiable 混合任务（如 open-ended agent）上 PRM 仍有空间。学术上的趋势是把 PRM "降级" 为 inference-time tool（BoN / search heuristic）而不是 training-time reward——风险更小、收益不变。
- **加分**：PRM800K 数据本身仍是 reasoning 研究的重要 asset，可用于训 reasoning 蒸馏数据 / 评测 reasoning 错误模式 / 教科书式分析"模型在哪一步出错"。

</details>

---

## 7. 延伸阅读

- [PRM800K 数据集（OpenAI / Lightman）](https://github.com/openai/prm800k) — 80 万 step 标注数据 + 标注 guideline，自己训 PRM 的起点。
- [Skywork-PRM 模型卡](https://huggingface.co/Skywork/Skywork-PRM-7B) — 2024 年开源 PRM 标杆，含 chat template / 推理示例。
- [Math-Shepherd 论文](https://arxiv.org/abs/2312.08935) — 自动 PRM label 的官方实现，要做大规模自动标注必读。
- [DeepSeek-R1 报告 §2.2.3](https://arxiv.org/abs/2501.12948) — 弃用 PRM 的官方解释，10.3 RLVR 的导论。
- [HuggingFace PRM Leaderboard](https://huggingface.co/spaces/AI-MO/ProcessBench-leaderboard) — ProcessBench 评测 PRM 在多种 reasoning 任务上的表现，PRM 选型参考。
- 推荐继续读本教程的 **10.3 节《RLVR 与 DeepSeek-R1：纯 RL 激发 long-CoT》**——把"为什么不用 PRM"的论点完整展开，讲清楚 RLVR 全流程；以及 **10.4 节《推理时搜索：Best-of-N / MCTS / Verifier-guided》**——把本节 PRM-guided BoN 推广到更复杂的搜索算法。
