---
title: "10.3 RLVR 与 DeepSeek-R1：纯 RL 激发 long-CoT"
description: "RLVR（RL with Verifiable Rewards）= 把 9.5 GRPO 的 \"RM 给 reward\" 换成 \"verifier（数学答案对错 / unit test 通过 / 格式匹配）给 0/1 reward\" 的 RL 范式；DeepSeek-R1（2025.01）用 RLVR + GRPO 证明了纯 RL 可以从 base model 自发涌现 long-CoT、self"
---

> ⏱ 预计阅读 60 分钟 ｜ 难度 ★★★ 🔥 ｜ 前置：9.5 GRPO

## 一句话本节讲什么

RLVR（RL with Verifiable Rewards）= 把 9.5 GRPO 的 "RM 给 reward" 换成 "verifier（数学答案对错 / unit test 通过 / 格式匹配）给 0/1 reward" 的 RL 范式；DeepSeek-R1（2025.01）用 RLVR + GRPO 证明了**纯 RL 可以从 base model 自发涌现 long-CoT、self-check、reflection**——著名的 "Aha moment"——并由此引爆 2025 年整个 reasoning 模型浪潮（QwQ / Qwen3-thinking / GLM-Z1 / Kimi K2 等几乎全部跟进）。本节讲清 RLVR 的定义、R1 的 4-5 阶段 pipeline、reward 设计、与传统 RLHF 的本质差异，以及小规模复现路径。

---

## 1. Mental model（直觉）

### 1.1 RLVR 是 RLHF 的"去 RM 版"

回想 9.2-9.3 的传统 RLHF 路径：

```
preference 数据 → 训 RM → 用 RM 给 policy 打 reward → PPO/GRPO 更新 policy
                  ↑ 这一步成本高、且 RM 会被 hack
```

RM 训练需要海量 preference 标注（OpenAI InstructGPT 用了 ~33K pair，Anthropic HH 100K+），而且训出来的 RM 本身**只是对人类偏好的近似**，policy 一旦发现 RM 的弱点（输出特别长、放更多 emoji、加保险话术）就会**reward hacking**——一边 RM 分数蹭蹭涨，一边人工评测反而退步。这是 RLHF 头号工程地雷（详见 9.6）。

RLVR 的破局思路简单粗暴：**对一类任务，"对错"是客观可判定的——根本不需要 RM**。

```
verifiable task 的 reward：
   Math:    答案 == 标准答案？      → 0/1
   Code:    所有 unit test 通过？   → 0/1
   Format:  输出符合 schema？       → 0/1
```

数学题答案 42 就是 42，code 通过 unit test 就是通过，没法 hack——你不能对 sympy 说"求你给我的答案打 1 分"。这是 RLVR 比 RM-based RLHF 强的根本原因：**reward signal 是 ground truth 的、不可被 policy 攻击**。

代价是显而易见的：**RLVR 仅适用于 verifiable task**。chat helpfulness、creative writing、emotional support 这种"好不好"取决于人类感受的任务，verifier 无能为力，仍要回到 RM 范式。所以 R1 的完整 pipeline 是 RLVR + 传统 RLHF 的混合（§2 详解）。

类比：训运动员。

- **传统 RLHF**：请一群裁判看比赛打分（RM）→ 运动员可能学会"讨好裁判"而不是真正变强
- **RLVR**：直接看终点线秒表（verifier）→ 跑得快就是快，无法作弊
- **现实**：体操、跳水还是要裁判（unverifiable），田径、游泳就用秒表（verifiable）

### 1.2 R1 之前 vs R1 之后的世界观

**2024.09 之前**（pre-o1 时代）：
- 大家相信 reasoning 能力主要来自 **SFT + CoT prompting**
- LLM 输出长度典型 < 1K token
- "推理时多花 token = 涨点" 不是主流认知

**2024.09 OpenAI o1 发布**：
- 首次展示"训练时让 model 学会长 reasoning + 推理时让它写 5K-50K token 的思考"能涨点
- 但闭源、技术细节 0 公开，社区只有 PRM（10.2）+ MCTS 的猜测

**2025.01 DeepSeek-R1 发布**（开源 + 完整技术报告）：
- 公开证明：**仅用 GRPO + verifier reward 从 base model 训**就能涌现 long-CoT、self-reflection
- R1-Zero（无 SFT cold start 版）训练曲线里 model 自发出现"等等，让我重新检查这一步" — Aha moment
- AIME / MATH 等 benchmark 上 R1 与 o1 同档；671B MoE 总成本据传 < $6M
- **整个开源社区在两周内集体 pivot 到 RLVR + GRPO**——QwQ、Qwen3-thinking、GLM-Z1、Doubao-Reasoning、Kimi K2 都是此后产物

R1 的历史地位类比：**R1 之于 reasoning，像 ChatGPT 之于 chat、像 BERT 之于 NLP**——它不是第一个有 reasoning 能力的 model，而是**第一个把范式定型并完整开源**的 model。

### 1.3 RLVR 的 mental model 图

```
              传统 RLHF (PPO + RM)
   ┌─────────┐    ┌─────────┐    ┌─────────┐
   │ policy  │───→│   RM    │───→│ reward  │
   └─────────┘    │(learned)│    └─────────┘
                  └─────────┘
                  ↑ 标注成本高、可被 hack

              RLVR (GRPO + verifier)
   ┌─────────┐    ┌──────────┐    ┌─────────┐
   │ policy  │───→│ verifier │───→│ 0/1     │
   └─────────┘    │(rule)    │    │ reward  │
                  └──────────┘    └─────────┘
                  ↑ 零标注、不可 hack
```

记住一个 takeaway：**RM 是对人类偏好的有损近似，verifier 是 ground truth 本身**。这是 RLVR 范式的根。

---

## 2. 公式与原理

### 2.1 RLVR 的 reward 设计

RLVR 的 RL 算法部分与 9.5 GRPO 完全一样（PPO 也行，但 GRPO 因为不需要 critic + 对 0/1 reward 友好成为事实标准）。**RLVR 的全部独特性都在 reward function 上**。

形式化：给定 prompt $x$、policy 采样的 response $y$、ground-truth $y^*$，verifier 是一个**确定性函数** $V: (y, y^*) \to \{0, 1\}$（或离散小集合）：

$$r(x, y, y^*) = V(y, y^*)$$

最常见的两类 verifier：

**Math verifier**：从 response 中提取最终答案 `extract(y)`，与 $y^*$ 数值比较。

$$r_{\text{math}}(y, y^*) = \mathbb{1}[\,\text{equiv}(\text{extract}(y), y^*)\,]$$

`equiv` 是数学等价判定：处理分数 / 浮点 / 单位 / $\pi$ 表达式 / `\frac{1}{2} == 0.5` 等等。**纯字符串匹配会漏判一大堆**——这是工程踩坑头号雷（§4 详谈）。

**Code verifier**：把 response 中的 code 块抽出来，跑所有 unit test。

$$r_{\text{code}}(y, T) = \frac{|\{t \in T : \text{exec}(y, t) = \text{pass}\}|}{|T|}$$

可以是离散 0/1（全过才得分）或连续比例（pass rate），后者梯度信号更密。

### 2.2 R1 的复合 reward：accuracy + format

R1 实际用的是 **accuracy + format** 两项加权和。Format reward 强制 model 把思考与答案分别包裹在 `<think>...</think>` 和 `<answer>...</answer>` 里，这是后续 inference 时分离 thinking trace 与最终回答的关键。

$$r_{\text{R1}}(y, y^*) = \lambda_{\text{acc}} \cdot \mathbb{1}[\text{ans correct}] + \lambda_{\text{fmt}} \cdot \mathbb{1}[\text{format valid}]$$

DeepSeek 原论文的具体权重是 $\lambda_{\text{acc}} = 1$，$\lambda_{\text{fmt}} = 1$（同等权重的二维 0/1 → 总 reward $\in \{0, 1, 2\}$）。也有实现把 format 作为 hard gate（format 错就 reward = 0，format 对再判 accuracy）——两种都 work。

带入 9.5 GRPO 的组内归一化得 advantage：

$$\hat A_i = \frac{r_i - \bar r}{\sigma_r + \epsilon}, \quad r_i = r_{\text{R1}}(y_i, y^*)$$

由于 $r_i \in \{0, 1, 2\}$ 是低基数离散值，组内 $G = 16 \sim 64$ 条 response 的 reward 分布通常是**清晰的多极分布**（一堆 0、一堆 1、一堆 2），归一化后 advantage 信号极强——这正是 GRPO + RLVR 在 reasoning 上特别高效的原因。

### 2.3 R1 的完整训练 pipeline

DeepSeek-R1 论文给出两套 model：**R1-Zero**（demonstrate 纯 RL 可行）与 **R1**（实用版，多阶段）。

#### R1-Zero：纯 RL，无 SFT cold start

```
DeepSeek-V3-base
       │
       │  GRPO + verifier reward (math accuracy + format)
       │  无任何 SFT
       ↓
   R1-Zero
   ✓ AIME / MATH 涨到 o1-preview 水平
   ✓ 自发涌现 long-CoT、self-check、reflection (Aha moment)
   ✗ 输出可读性差（中英混杂、format 乱、有时 token 怪异）
```

R1-Zero 的科学价值 >> 实用价值：它证明了 **base model 内部本身就藏着 reasoning 能力，RL 只是把它"激发"出来**——不需要 SFT 教它怎么思考，给对的 reward 它自己会学。这与"必须 SFT 才会 CoT"的传统认知冲突，是 R1 论文最重要的 finding。

#### R1：4 阶段实用版

R1-Zero 输出可读性差，无法直接产品化。R1 在前后加 SFT 与第二轮 RL 修正：

```
DeepSeek-V3-base
       │
       │  Stage 1：SFT cold start
       │  数据：~thousand 条高质量 long-CoT（人工 + R1-Zero 输出筛选）
       │  目的：教 <think>/<answer> format + 中文流畅 + 基本 reasoning 模板
       ↓
   Cold-start SFT model
       │
       │  Stage 2：Reasoning RL
       │  GRPO + verifier reward (math + code + 部分 logic)
       │  + language consistency reward（避免中英混杂）
       │  目的：刷高 reasoning 能力
       ↓
   Reasoning RL model
       │
       │  Stage 3：Rejection sampling SFT
       │  从 Stage 2 model 大量采样、用 verifier 过滤正确的，加上通用 SFT 数据
       │  规模：~600K reasoning + ~200K general 数据
       │  目的：把 RL 学到的 reasoning 能力"稳定化"，并补回通用能力
       ↓
   General SFT model
       │
       │  Stage 4：General RLHF
       │  PPO 或 GRPO，reward 来自 RM（helpful + safe）+ verifier（reasoning task 仍用）
       │  目的：对齐人类偏好（chat、refusal 等 unverifiable task）
       ↓
   DeepSeek-R1 (final)
```

四阶段口诀：**Cold-start SFT → Reasoning RL → Reject-sample SFT → General RLHF**。

注意 Stage 3 与 Stage 4 的 RM 出现表明：**完整 R1 不是纯 RLVR**——helpful / harmless 这种 unverifiable 维度仍要 RM。RLVR 与传统 RLHF 在工业 pipeline 里是**互补**而非替代关系。

#### R1-Distill：把 R1 蒸馏到小 model

```
R1（teacher）─ generate ─→ ~800K reasoning 样本（含 <think>...</think>）
                              │
                              │  纯 SFT（无 RL）
                              ↓
       Qwen-7B / Qwen-32B / Llama-8B / Llama-70B base
                              │
                              ↓
                         R1-Distill series
```

发现：**对小 model（7B-70B）来说，"用 R1 输出做 SFT" 比 "自己跑 RLVR" 效果更好且更省**。这是因为小 model 自己 RL 探索能力弱，直接学 R1 已搜索好的 trace 更高效。所以**绝大部分开源 reasoning 7B-32B 都走 distill 路线**而非自己 RL。

### 2.4 Aha Moment：纯 RL 涌现 reflection

R1 论文里最 dramatic 的图（论文 Fig. 3）：训练中后期，R1-Zero 的输出里**自发出现**类似下面的语句：

```
... 设 x = 5，代入得 25 + 10 + 1 = 36 ≠ 35。
等等，让我重新检查这一步。哦，我把 9 看成 10 了，应该是 25 + 9 + 1 = 35。✓
```

关键词："等等让我重新检查"、"哦，我之前算错了"、"换一种方法试试"——这是**self-reflection** 与 **error correction** 的 emergent behavior。**没有人在 SFT 数据里教过这种话**，是 GRPO 在 verifier reward 驱动下让 policy **自己**发现"先尝试一次、错了回头检查"比"一次性给答案"reward 更高，于是自发 adopt 这种策略。

paper 里同时观测到：
- 平均输出长度从训练初期 ~200 token 一路增长到后期 ~5000-10000 token
- 单条 response 内出现多次 "wait / let me check / actually" 的频率与 accuracy 正相关

**Aha moment 的本质**：reasoning 是 base model 已具备的"潜在能力"，RL 在 verifier 反馈下重新分配了 prior（让"先想清楚再答"的策略概率上升）。它不是无中生有的新能力，而是**潜能的激发**。

后续工作（TinyZero、Logic-RL）在远小得多的 model（甚至 1.5B-3B）上复现了同样的 Aha moment，证明这不是 671B 大模型的专利——**verifier + GRPO + 足够 RL step** 是 emergence 的充分条件。

### 2.5 RLVR vs 传统 RLHF 对比表

| 维度 | 传统 RLHF（PPO + RM） | RLVR（GRPO + verifier） |
|---|---|---|
| **reward 来源** | learned RM | rule-based verifier（sympy / pytest / regex） |
| **数据成本** | 高（preference 人工标注 ~10K-100K pair） | 低（任务已有 ground truth，无需额外标注） |
| **reward hack** | 严重（length / sycophancy / formatting bias） | 几乎无（数学对就是对） |
| **reward 信号密度** | 中（连续分数，但有 RM 噪声） | 高（0/1 ground truth） |
| **适用任务** | 通用（chat、helpful、harmless） | verifiable（math / code / format / logic） |
| **算法主流** | PPO | GRPO |
| **可解释性** | 低（RM 是黑盒） | 高（reward = "答案对错"） |
| **scaling 上限** | 受 RM 大小 / 标注质量限制 | 几乎无上限（verifier 是确定性程序） |
| **典型代表** | InstructGPT、Claude-2、LLaMA-2-Chat | DeepSeek-R1、QwQ、Qwen3-thinking、GLM-Z1 |
| **二者结合** | R1 Stage 4 用 RM 做 helpful；RLVR 适合的 reasoning 仍用 verifier | — |

**核心 takeaway**：RLVR 不是要取代传统 RLHF，而是**在 verifiable task 上提供一条更便宜更稳更不可 hack 的路**。完整工业级 reasoning model 的 pipeline 通常是 "RLVR for reasoning + RLHF for chat" 的混合。

---

## 3. 最小代码示例

### 3.1 Math verifier（22 行）

```python
import re
from sympy import simplify, sympify, Rational

def extract_answer(response: str) -> str | None:
    """从 R1 风格输出中抽取 <answer>...</answer> 内的数值答案"""
    m = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if not m:
        return None
    # 兼容 \boxed{...}、纯数字、分数等多种写法
    ans = m.group(1).strip()
    boxed = re.search(r"\\boxed\{([^}]+)\}", ans)
    return boxed.group(1).strip() if boxed else ans

def math_verify(response: str, gt: str) -> float:
    """返回 1.0 (正确) / 0.0 (错误)。处理分数、浮点、符号等价"""
    pred = extract_answer(response)
    if pred is None:
        return 0.0
    try:
        # 用 sympy 做符号等价判定（处理 1/2 == 0.5、sqrt(4) == 2 等）
        return 1.0 if simplify(sympify(pred) - sympify(gt)) == 0 else 0.0
    except Exception:
        # 兜底：字符串比较（去空格、统一小写）
        return 1.0 if pred.replace(" ", "").lower() == gt.replace(" ", "").lower() else 0.0
```

要点：
- **必须用 sympy / math_verify 等库**做等价判定，纯字符串匹配会漏判 `1/2 vs 0.5`、`\frac{1}{2}` vs `0.5`、`sqrt(2)` vs `2^{1/2}` 等等
- 抽取层要兼容 `\boxed{...}`、纯数字、`\text{...}` 等 LaTeX 包装
- try/except 兜底到字符串匹配——某些 expression sympify 会失败（如 "Yes/No" 类答案）

工业实现可直接用开源库 [Hendrycks `math_verify`](https://github.com/huggingface/Math-Verify) 或 [DeepSeek 官方 `math_eval` 脚本](https://github.com/deepseek-ai/DeepSeek-Math)，覆盖更全面。

### 3.2 Format reward（15 行）

```python
import re

THINK_PATTERN = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)

def format_reward(response: str) -> float:
    """检查 R1 风格 <think>...</think><answer>...</answer> format"""
    if not THINK_PATTERN.search(response):
        return 0.0
    # 进一步检查 <think> 与 <answer> 各只出现一次（防 model 重复嵌套）
    if response.count("<think>") != 1 or response.count("<answer>") != 1:
        return 0.5  # 部分给分，鼓励渐进学习
    return 1.0

def r1_reward(response: str, gt: str,
              w_acc: float = 1.0, w_fmt: float = 1.0) -> float:
    """R1 复合 reward = accuracy + format 加权和"""
    return w_acc * math_verify(response, gt) + w_fmt * format_reward(response)
```

要点：
- Format reward 给"部分分"（0.5）有助于训练初期 model 渐进学到 schema——纯 0/1 时早期梯度太稀疏
- 实战中 `w_fmt` 在训练初期可设大（如 2.0）让 model 先学 format，后期降到 0.5 或更低让 accuracy 主导
- DeepSeek 原论文是简单等权 1:1，足够 work

### 3.3 R1-style GRPO 训练循环骨架（35 行）

```python
import torch
from torch.optim import AdamW

def r1_grpo_train_step(policy, ref_model, prompts, ground_truths,
                       optimizer, G=16, beta=0.04, eps_clip=0.2,
                       w_acc=1.0, w_fmt=1.0, gen_kwargs=None):
    """R1 风格的单步 GRPO 训练（基于 9.5 §3.2 的 grpo_step 简化版）"""
    B = len(prompts)
    gen_kwargs = gen_kwargs or {"temperature": 0.9, "top_p": 0.95, "max_new_tokens": 4096}

    # ===== Phase 1: rollout (G 条 / prompt) =====
    with torch.no_grad():
        responses = policy.generate(prompts, num_return_sequences=G, **gen_kwargs)  # (B*G,)
        log_pi_old = policy.compute_logp(prompts, responses)        # (B*G, T)
        log_pi_ref = ref_model.compute_logp(prompts, responses)     # (B*G, T)

        # ===== verifier reward (核心：accuracy + format) =====
        rewards = torch.tensor([
            r1_reward(r, gt, w_acc, w_fmt)
            for r, gt in zip(responses, [gt for gt in ground_truths for _ in range(G)])
        ]).view(B, G)

    # ===== Phase 2: 组内归一化 advantage (9.5 §2.2) =====
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True) + 1e-8
    advantages = ((rewards - mean) / std).view(B*G, 1)

    # ===== Phase 3: GRPO loss =====
    log_pi_new = policy.compute_logp(prompts*G, responses)          # (B*G, T) with grad
    ratio = torch.exp(log_pi_new - log_pi_old)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantages
    pg_loss = -torch.min(surr1, surr2).mean()
    log_u = log_pi_ref - log_pi_new
    kl = (torch.exp(log_u) - log_u - 1).mean()                      # k3 KL
    loss = pg_loss + beta * kl

    optimizer.zero_grad(); loss.backward(); optimizer.step()
    return {"loss": loss.item(), "reward_mean": rewards.mean().item(),
            "reward_std": rewards.std(dim=1).mean().item()}
```

要点：
- **核心改动相对 9.5 GRPO 仅一行**——把 `reward_fn = RM(...)` 换成 `r1_reward(...)`（verifier）。算法部分照搬 GRPO
- `temperature=0.9` 必须够高，否则 G 条 response 趋同、advantage 全 0（9.5 经典踩坑复读）
- `max_new_tokens=4096` 起步，long-CoT 后期可能 8K-16K，要相应预留显存
- **生产实战要把 generate 切到 vLLM**——纯 HF generate 跑 G=16、4096 长度极慢

### 3.4 用 TRL 一站式 RLVR（20 行）

```python
from trl import GRPOConfig, GRPOTrainer
from datasets import load_dataset

# verifier-style reward function（不需要训 RM）
def reward_acc(prompts, completions, answer, **kw):
    return [math_verify(c, a) for c, a in zip(completions, answer)]
def reward_fmt(prompts, completions, **kw):
    return [format_reward(c) for c in completions]

ds = load_dataset("HuggingFaceH4/MATH-500", split="train")  # 含 answer 列
cfg = GRPOConfig(
    output_dir="ckpt/r1_repro",
    learning_rate=1e-6,
    num_generations=16,                  # G
    max_completion_length=4096,
    temperature=0.9,
    beta=0.04,                           # KL 系数
    bf16=True, gradient_checkpointing=True,
    loss_type="dapo",                    # 用 DAPO 改进版 (token-level + clip-higher)
)
trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-Math-7B",
    reward_funcs=[reward_acc, reward_fmt],   # 多 reward 自动加权求和
    args=cfg, train_dataset=ds,
)
trainer.train()
```

要点：
- TRL 原生支持多个 reward function（list 形式），等权求和——可分别给 accuracy / format / length penalty 等
- `loss_type="dapo"` 启用 9.5 §4.1 介绍的 DAPO 改进（token-level loss + clip-higher），比原版 GRPO 在 long-CoT 上更稳
- 7B + LoRA + GRPO + G=16 在 4×80G H100 上一天能跑完一个 epoch（数据 1-3K prompt），是单组复现 R1-Zero "aha moment" 的最低配

---

## 4. 工程踩坑与经验

- ❗ **R1-Zero 输出可读性差是 by design 的**——纯 RL 没有 SFT 教 format / 教语言一致性，输出会出现中英混杂（中文 prompt 突然冒英文 reasoning）、format 乱、甚至偶尔的乱码 token。**实战部署的 reasoning model 必须加 cold-start SFT**（R1 Stage 1）。如果你只想科学复现 "aha moment" 可以跳过 cold start，但产品化必须有 SFT。
- ❗ **Math verifier 的鲁棒性是 RLVR 头号工程难点**——纯字符串匹配会让 `\frac{1}{2}` vs `0.5`、`2π` vs `6.2832`、`x = 5` vs `5` 全部判错，导致 reward 噪声爆表、policy 学的是 "猜对 verifier 偏好的写法" 而不是真正解题。**务必上 sympy 或 math_verify**，处理符号等价、单位、tolerance、boxed 包装。**verifier 漏洞 = reward hacking 入口**。
- ❗ **Length penalty 必须加，否则 long-CoT 越来越长（length hacking）**——RLVR 训练过程中 model 会发现 "想越久越可能对" 是 verifier 反馈给的 implicit signal，输出长度从 200 token 涨到 50K+ token，推理 cost 爆炸。常见做法：(1) 设 `max_completion_length` hard cap；(2) 加 length penalty reward `r_len = -α * len(response) / max_len`；(3) 用 DAPO 的 overlong filtering 直接丢超长样本。R1 原论文用方案 (1) + (3)。
- ❗ **verifier reward 在 unverifiable task 上完全不适用**——chat helpfulness、creative writing、emotional support 没有"对错"，给 0/1 reward 等于随机。这类任务必须回到 RM 范式。**R1 的 Stage 4 General RLHF 就是为此而存在**。如果你的训练数据混了大量 unverifiable task，要么剔除要么挂 RM——不要让 verifier 在它无能的地方乱给 0。
- ❗ **R1 / R1-Distill 部署时推理 cost 是普通 chat model 的 5-50 倍**——单次 response 平均 5K-30K token（thinking + answer），TTFT 与总 latency 显著增加。生产部署要做 capacity estimation：用户量 × 平均 thinking token × per-token cost = 真实成本。Qwen3 / GLM-Z1 等给用户提供 `thinking on/off` toggle 就是为了让用户在 latency / quality 之间显式选择。
- ❗ **R1-Distill 容易让小 model "学坏"——只学到 reasoning 模式，通用能力反而下降**——大量 long-CoT SFT 数据会让 7B 小 model 在 chitchat / 简单问答上也开始喷长 thinking，体验糟。修复：(1) 蒸馏数据混入大量 short-response 通用数据（典型 reasoning : general = 1 : 3）；(2) 训练时区分"需要 thinking 的 prompt" 与"不需要的"，在不需要的 prompt 上用 short response 训练。Qwen3 的 thinking-mode toggle 就是这个思路的延伸。
- ❗ **Cold-start 数据 quality 影响巨大**——R1 Stage 1 只用了 ~thousand 条人工筛选的 long-CoT 数据，但每条都极高质量（reasoning 步骤清晰、format 规范、语言一致）。Open-R1 / TinyZero 复现时如果用的 cold-start 数据脏（mixed CoT 风格、错误推理过程被当作正例），后续 RL 阶段往错误方向 collapse。**宁可少而精**，cold-start 几百条高质量 >> 几万条脏数据。
- ❗ **`<think>` 标签必须 SFT 教，光 RL 学不全**——format reward 能让 model 学会大致包裹格式，但细节（标签是否单独成行、内部缩进、`<think>` 与 `<answer>` 之间是否有特定分隔符）光靠 reward shaping 学不准。Cold-start SFT 几百条样本就能教明白，**不要试图纯 RL 学 format**——这是 R1 Stage 1 存在的核心理由之一。
- ❗ **GRPO + verifier 的组内 reward 全 0 / 全 1 现象比 RM-based RLHF 更严重**——简单 prompt G 条全对（reward 全 1）、超难 prompt G 条全错（reward 全 0），advantage 都被归一化为 0，这一 update 完全浪费。**必上 DAPO 的 dynamic sampling**（9.5 §4.1）剔除零方差 group。RLVR 训练到 50% step 时此类 group 占比可能高达 40-60%。
- ❗ **R1 训练 cold-start SFT 的 long-CoT 数据极其稀缺**——市面上没有现成"教 model 怎么 reflection"的 dataset。常见获取路径：(1) 从 o1 / R1 等已有 reasoning model 蒸馏；(2) 从 R1-Zero 自己输出里挑可读的（但需大量人工筛）；(3) 用专门的 long-CoT 数据集如 NuminaMath-CoT、OpenThoughts。**自己从零造 cold-start 数据是 R1 复现最大的工程成本**，开源 cold-start 数据是开源社区的稀缺资源。
- ❗ **language consistency reward 别忘了加**——R1 论文 Stage 2 单独有一项 "language consistency reward"：检测 response 中目标语言（中文 / 英文）的 token 占比，低于阈值（如 80%）就扣分。否则 RLVR 会让中文 prompt 输出夹大量英文 reasoning（base model 数学知识更多在英文语料上）。看似小细节，对中文用户体验影响巨大。
- ❗ **千万别在 R1-Zero 阶段做 reward shaping 太花哨**——见过有人加 5 项 reward（accuracy + format + length + coherence + step-count），权重难调、相互打架，最后 model 学的是"满足复杂 reward 函数的几何"而不是"做对题"。**R1 原版只 2 项（acc + format）已经够 work**。Occam's razor。

---

## 5. 经典 paper

- **DeepSeek-AI, 2025 — *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*** — 本节核心必读。读 §2 的训练 pipeline（R1-Zero vs R1 的 4 阶段差异）、§3 的 R1-Zero 训练曲线（看 Aha moment 与 length 增长）、§4 的 distillation 实验。这是开源社区 reasoning model 的圣经，没看过这篇就没法跟人聊 RLVR。Tech report 写得相当详细，也有大量 ablation。
- **Shao et al., 2024 — *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*** — GRPO 提出原典（9.5 已介绍），同时也是 R1 算法基础。读 §4 的 GRPO + math verifier 实验，是 R1 的"算法前传"。R1 的算法部分本质上就是 DeepSeekMath 的 GRPO + verifier 范式 scale 到 671B MoE。
- **OpenAI, 2024 — *Learning to Reason with LLMs (o1 系统卡)*** — RLVR 范式的"引爆点"。文档很短（OpenAI 不公开技术细节），但读它能 calibrate 整个 reasoning 范式的起源——R1 之所以震撼是因为它**复刻并超越了 o1 同时完整开源**。读时关注 "test-time compute scaling" 这个概念，与 §3.x 的 inference cost 直接相关。
- 选读：**Pan et al., 2025 — *TinyZero: From R1-Zero to a 30 USD Reproduction*** — Berkeley 用 3B Qwen 在 ~$30 算力下复现 R1-Zero "aha moment"，证明 emergence 不是大模型专利。读它的 §4 训练曲线，对理解"什么是真正的最低复现门槛"有帮助。是教学最友好的复现工作。
- 选读：**HuggingFace, 2025 — *Open-R1: A Fully Open Reproduction of DeepSeek-R1*** — HF 主导的 R1 完整开源复现项目，包含 cold-start 数据集（OpenR1-Math-220K）、训练 recipe、评测 pipeline。如果你要真正动手复现，这是 reference。GitHub `huggingface/open-r1`。
- 选读：**Hou et al., 2025 — *Search-R1: Training LLMs to Reason and Leverage Search Engines via RL*** — 把 RLVR 推广到 agent 场景（search tool）的代表作。本节聚焦 reasoning，但 R1 范式向 agent 的迁移正是 Module 15 主题——读这篇能 preview。

---

## 6. 自测与面试题

**Q1（pipeline）：** 写出 DeepSeek-R1 的完整 4 阶段训练 pipeline，每阶段说明输入数据、训练算法、目的。同时说明 R1-Zero 与 R1 的关系。

<details>
<summary>Answer sketch</summary>

**R1 完整 pipeline（4 阶段）**：

| 阶段 | 输入 | 算法 | reward / loss | 目的 |
|---|---|---|---|---|
| Stage 1: Cold-start SFT | DeepSeek-V3-base + ~thousand long-CoT 样本 | SFT (next-token) | CE loss | 教 `<think>/<answer>` format、语言一致性、基本 reasoning 模板 |
| Stage 2: Reasoning RL | Cold-start SFT model + math/code/logic prompts | **GRPO + verifier** | accuracy + format + language consistency | 刷高 reasoning 能力（核心 RLVR 阶段） |
| Stage 3: Rejection sampling SFT | Stage 2 model 大量采样 + verifier 过滤 + 通用 SFT 数据（~600K reason + ~200K general） | SFT | CE loss | 把 RL 学到的 reasoning trace "稳定化"，并补回通用对话能力 |
| Stage 4: General RLHF | General SFT model + 通用 prompts | PPO/GRPO + **RM**（helpful/safe）+ verifier（reasoning task） | RM score + verifier | 对齐人类偏好、refusal、harmless（unverifiable task） |

**R1-Zero vs R1 关系**：
- **R1-Zero** = 只走"V3-base + GRPO + verifier"，跳过 Stage 1/3/4。是**科学 demonstration**，证明纯 RL 可以涌现 long-CoT。但输出可读性差（中英混杂、format 乱），不能产品化
- **R1** = R1-Zero 的实用化升级，前后加 SFT + Stage 4 RLHF 解决可读性、通用能力、对齐三个问题
- 二者**算法核心一样**（GRPO + verifier），区别在于"是否搭配 SFT"

加分：
- Stage 3 用 rejection sampling 是因为 Stage 2 RL 输出有正有错，必须用 verifier 过滤后再 SFT
- Stage 4 是 RLVR 与传统 RLHF 的混合——reasoning task 仍用 verifier，chat task 用 RM
- R1-Distill（Qwen-7/32B、Llama-8/70B）= 用 R1 输出蒸馏到小 model，是另一条独立产线

</details>

**Q2（reward）：** 为什么 RLVR 用 verifier 而不是 RM？至少给 3 个理由。verifier 不适用于哪些任务？这种局限如何在 R1 完整 pipeline 中被处理？

<details>
<summary>Answer sketch</summary>

**Verifier 优于 RM 的 3+ 理由**：

1. **不可 hack**：数学答案对就是对，code 通过 unit test 就是通过，policy 没法"游戏" verifier。RM 反过来是头号 reward hacking 源头（length bias、sycophancy、formatting hack）
2. **零标注成本**：verifier 是确定性程序（sympy / pytest / regex），不需要 preference 标注。RM 训练需要 10K-100K 量级 preference pair，标注成本极高
3. **scale 上限高**：verifier 的"质量"由代码 quality 决定，能跑就 100% 准确（不像 RM 还要担心 OOD、ensemble）。任务越多 verifier 加越多就行
4. **加分理由**：
   - reward 信号是 ground truth 的 0/1，组内归一化后 advantage 信号清晰（GRPO 友好）
   - 可解释性高（"为什么这个 sample 高 reward" 一眼可见）
   - 不用维护额外 RM model（省一份显存 + 训练流水）

**Verifier 不适用的任务**：
- **chat helpfulness**：什么叫"有帮助"无客观定义
- **creative writing**：诗歌 / 故事好坏取决于人类感受
- **emotional support / safety refusal**：没有"对错"
- **指令遵循的语气、风格、长度偏好**：主观维度
- 共同特征：**reward 信号必须依赖人类感受才能给**

**R1 中如何处理这种局限**：
- **R1 用 RLVR + RLHF 混合 pipeline**：Stage 2 reasoning task 用 verifier（数学 / code），Stage 4 通用 task 用 RM（helpful / safe）
- 单 verifier 是 reasoning model 的核心引擎，但**完整产品化的 reasoning model 必然要混 RM**——它们是互补关系不是替代关系
- 这也解释了为什么 R1 之后 helpful chat 能力还能保持——Stage 4 没把 reasoning RL 学的能力洗掉

</details>

**Q3（前沿）：** "Aha moment" 现象的本质是什么？纯 RL 真的能"涌现 reasoning"吗？这种涌现对 small model（< 7B）也成立吗？

<details>
<summary>Answer sketch</summary>

**Aha moment 现象描述**：
- R1-Zero 训练中后期（典型 ~50% step 之后），model 输出里**自发**出现 "等等让我重新检查这步"、"哦我之前算错了"、"换种方法试试" 等 self-reflection 与 error correction 语句
- 平均输出长度从 ~200 token 涨到 5K-10K token，且长度增长与 accuracy 正相关
- 没有任何 SFT 数据教过这种话，是 RL 在 verifier reward 下自发学到的策略

**本质（深入解释）**：

1. **不是"无中生有"，是 prior 重新分配**：base model 在预训练中已经见过大量 reasoning 文本（教科书、论文、Stack Exchange 上的解题过程），"先想清楚再答" / "检查中间步骤" 这种 token sequence 在 prior 里已经存在但**概率较低**。RL 通过 verifier reward 让这种 sequence 的概率上升，相当于"激活潜在能力"。

2. **GRPO + 0/1 reward 是 catalyst**：
   - 同 prompt G 条 response 里，"短而错" reward = 0，"长但反复检查后对" reward = 1
   - 组内归一化后，"长而对"的 advantage 显著为正，policy gradient 强力推它
   - 几百轮后 policy 学到 "先尝试 → 检查 → 修正" 比 "一次给答案"reward 期望更高，于是 adopt 这种策略

3. **不是 magic，是 emergent strategy**：
   - 类似 multi-armed bandit 中 agent 自发学会"先 explore 再 exploit"
   - 不是 model 突然"理解"了 reasoning 是什么，而是在 reward landscape 上找到了一条更优的策略

**纯 RL 真的能涌现 reasoning 吗？争议点**：
- **支持**："纯 RL（无 cold start）" 在 R1-Zero、TinyZero、Logic-RL 多次复现，确实从 base model 涌现 reflection
- **质疑**：base model 已是预训练好的 LLM，"纯 RL" 严格说是"以预训练为底的 RL"——不是从 random init 学到 reasoning。**真正的"无 prior 涌现 reasoning"目前没人做到**
- **务实的中间态**：reasoning 能力是**预训练 + RL 共同的产物**，预训练提供 capability ceiling，RL 提供 elicitation。R1 的贡献是**证明 RL elicitation 可以非常 dramatic**——比之前的 SFT-only 强得多

**Small model 是否成立**：
- **TinyZero**（Berkeley 2025）：3B Qwen 在 ~$30 算力下复现 aha moment ✓
- **Logic-RL**（清华 2025）：7B 模型在纯 logic puzzle 上学到 reflection ✓
- **SimpleRL-Zoo**：1.5B-7B 系列均能复现，但**< 1.5B 时显著退化**——base model 太弱，prior 里缺乏足够的 reasoning sequence，RL 无 "可激活" 内容
- **结论**：emergence 不是大模型专利，但有 base model size lower bound（经验 ~1.5B），低于此 RL 学不出来——印证了"RL 是激活预训练潜能"的解释

加分：
- "Aha moment" 是个 narrative-friendly 名字，但严谨地说是 **policy 收敛到一个 reflection-friendly mode** 的相变现象
- 与 Wei 2022 "emergent abilities of LLMs" 的争论同源——是真涌现还是 metric artifact，尚未完全定论
- 工业实践**不依赖** aha moment 的"灵性"解释，把它当作可复现的 training dynamics 即可

</details>

---

## 7. 延伸阅读

- [DeepSeek-R1 GitHub & 技术报告](https://github.com/deepseek-ai/DeepSeek-R1) — 模型权重 + tech report，必看的 reasoning model 圣经
- [HuggingFace Open-R1](https://github.com/huggingface/open-r1) — 全开源 R1 复现项目，含 cold-start 数据集（OpenR1-Math-220K）、训练脚本、评测
- [TinyZero](https://github.com/Jiayi-Pan/TinyZero) — Berkeley $30 复现 R1-Zero "aha moment"，3B 小 model 教学最友好
- [Logic-RL](https://github.com/Unakar/Logic-RL) — 清华纯 logic puzzle 复现 R1-Zero，验证 emergence 不依赖大模型
- [SimpleRL-Zoo](https://github.com/hkust-nlp/simpleRL-reason) — HKUST 多 size / 多任务 R1 复现 zoo
- [Math-Verify](https://github.com/huggingface/Math-Verify) — HF 维护的 robust math verifier 库，处理 sympy / boxed / fraction 等等价
- [Will Brown — Verifiers](https://github.com/willccbb/verifiers) — 教学级 RLVR + GRPO 实现，配合本节 §3.3 看
- [unsloth GRPO Notebook](https://docs.unsloth.ai/basics/reasoning-grpo) — 单卡 24GB 跑通 R1-style RLVR 的最小实操
- [Nathan Lambert — Interconnects: R1 Analysis](https://www.interconnects.ai/p/deepseek-r1-recipe-for-o1) — R1 paper 最权威的英文解读，配合原 paper 看
- 推荐继续读本教程的 **9.6 节《工程踩坑：reward hacking / RM 漂移 / KL 坍塌》**——RLVR 部分规避了 reward hacking，但 length / language hacking 仍存在，9.6 系统讲；**10.4 节《推理时搜索：Best-of-N / MCTS / Verifier-guided》**——R1 的"训练时学 long-CoT"与"推理时多采样"是 reasoning 能力的两条独立轴；**Module 15** 把 RLVR + GRPO 推广到多轮 Agent RL（Search-R1 / ReSearch / ReTool / Agent-R1），是 R1 范式的下一站
