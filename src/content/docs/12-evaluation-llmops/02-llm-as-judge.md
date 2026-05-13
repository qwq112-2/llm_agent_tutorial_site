---
title: "12.2 LLM-as-Judge / Pairwise / Reward 评测"
description: "LLM-as-Judge 是把\"给 打分 / 比大小\"这件事委托给一个强 LLM（GPT-4 / Claude / R1）来做的方法——pairwise（A vs B 选优）/ single-grading（1-10 打分）/ reference-based（参考答案对比） 三种主流形式，撑起了 MT-Bench、Arena-Hard、AlpacaEval 2.0 这些当代 chat 评测，也撑起"
---

> ⏱ 预计阅读 30 分钟 ｜ 难度 ★★ ｜ 前置：9.2 Reward Model、12.1 通用评测

## 一句话本节讲什么

LLM-as-Judge 是把"给 (prompt, response) 打分 / 比大小"这件事委托给一个强 LLM（GPT-4 / Claude / R1）来做的方法——**pairwise（A vs B 选优）/ single-grading（1-10 打分）/ reference-based（参考答案对比）** 三种主流形式，撑起了 MT-Bench、Arena-Hard、AlpacaEval 2.0 这些当代 chat 评测，也撑起了 9.2 RM 训练的一大半数据来源；本节把三种 judge 形式、judge 的可靠性数据、5 大 bias 与缓解、Reward Bench 这种"评测 RM 的 meta-benchmark"，以及"训了一个新模型如何用 judge 严肃证明它比 baseline 好"这条工程闭环讲清楚。

安全 / 红队评测在 12.3 单独讲，本节聚焦 **quality judge**（"答得好不好"），不涉及 "答得安不安全"。

---

## 1. Mental model（直觉）

### 1.1 为什么要 LLM-as-Judge

12.1 的 5 大维度里，**MMLU / GSM8K / HumanEval / IFEval 都有「机器可验证的正确答案」**——选 ABCD、跑 unit test、regex 数 bullet——所以可以 auto evaluate。但 chat 类任务（"帮我润色这段邮件"、"解释一下 Bradley-Terry 模型"、"写一个程序员笑话"）**没有 ground truth**——只有"相对而言哪个更好"。

历史上这种任务只能靠人评：让两个标注员看 A 和 B，选谁更好，最后用 Bradley-Terry / Elo 拟合。Chatbot Arena 至今仍是这个范式的金标。问题是：

- **慢**：1 万对比较要几天到几周
- **贵**：一对 0.5-2 USD（外包标注）
- **不可重跑**：今天标完，明天换个 baseline 又要重标一轮

LLM-as-Judge 的核心 trick 是：**用一个强 LLM（GPT-4 / Claude 3.5 / DeepSeek-V3）当"代理裁判"**。一对的成本 0.001-0.01 USD，几小时能跑 10k 对，且**与人评一致率 80-85%**——比"两个人之间"的一致率（75-80%）还高一点。这个数据出自 [Zheng 2023]，是 LLM-as-Judge 落地的关键证据。

工业流水线长这样：

```
              prompt 集合 (80 / 500 / 805 题)
                       │
         ┌─────────────┴─────────────┐
         ▼                           ▼
   model_A.generate            model_B (baseline).generate
         │                           │
         └────────── (response_A, response_B) ─────────┐
                                                       │
                                                       ▼
                                          ┌────────────────────────┐
                                          │  Judge LLM (GPT-4o)    │
                                          │  prompt: "A vs B 哪个   │
                                          │   更好？randomize order"│
                                          └────────────────────────┘
                                                       │
                                                       ▼
                                              {A wins / tie / B wins}
                                                       │
                                                       ▼
                                       汇总 N 题 → win rate + 95% CI
```

读完本节应当能：(a) 写出可用的 pairwise judge prompt（含 position randomize）；(b) 看到 "A 比 B 高 8%" 这种声明能问 "样本多少？CI 多宽？哪种 judge？swap 了 order 没？length 控制了吗？"；(c) 知道什么时候用 RM 而不是 LLM-as-Judge（成本 100×、qps 50×）。

### 1.2 judge 与 RM 是什么关系

9.2 训的 Reward Model 本质上是**特化、轻量、低延迟版的 LLM-as-Judge**：

| 维度 | LLM-as-Judge | Reward Model |
|---|---|---|
| 形式 | prompt-based 强 LLM | finetuned 小模型 + scalar head |
| 输入 | (prompt, A, B) 或 (prompt, response) | (prompt, response) |
| 输出 | 文本：选 A / 选 B / 1-10 分 | scalar reward |
| 一致率 | 80-85%（GPT-4 vs human） | 65-75%（开源 8B RM vs human） |
| 成本/样本 | 0.001-0.01 USD | 0.00001 USD（自托管） |
| QPS | 受 API 限制 | 可达 1000+ |
| 适用 | 离线评测、judge 训练数据生成 | RLHF 在线训练、Best-of-N 实时打分 |

主流 SFT-RM-DPO pipeline 的 RM 训练数据，**很大比例是用 GPT-4 做 LLM-as-Judge 标的**（UltraFeedback、Magpie-Pro-DPO 等都是这个套路）。所以 LLM-as-Judge 既是评测工具，也是 RM 训练的数据源——本节讲的 bias 和缓解，每一条都同时影响"评测靠不靠谱"和"训出来的 RM 漂不漂"。

---

## 2. 三种 judge 形式与原理

### 2.1 Pairwise（工业最主流）

输入 $(x, y_A, y_B)$，judge 输出 $\{A, B, \text{tie}\}$。汇总 $N$ 题后报 **win rate**：

$$\text{WR}(\text{model A vs B}) = \frac{\#\{A \text{ wins}\} + 0.5 \cdot \#\{\text{tie}\}}{N}$$

为什么主流：

- **noise 小**：judge 不需要给出绝对分数，只需要一个二选一/三选一判断，inter-judge agreement 显著高于 1-10 打分
- **不需要 absolute scale**：judge 之间标定"7 分到底是几分"非常难，但"哪个更好"跨 judge 一致性很高
- **直接对应业务关心的指标**："新模型比老模型好的概率"

代价是：一次只能比 2 个，**要排 K 个模型需要 $O(K^2)$ 比较**（或用 Bradley-Terry / Elo 把 pairwise 压成 ranking）。Arena-Hard / AlpacaEval 2.0 / Chatbot Arena 都是 pairwise。

### 2.2 Single-grading

输入 $(x, y)$，judge 输出 1-10 分（或 1-5）。汇总 $N$ 题取均值。

优点：单条样本就能跑（不需要 baseline），适合"模型独立体检"。MT-Bench 是代表（80 题，每题 1-10 分，最后报 8.2 / 7.5 这种总分）。

缺点：**绝对分数稳定性差**——同一个 response，让 GPT-4 在不同时间打分，方差能到 1.5 分以上。所以 MT-Bench 总分 8.2 vs 8.0 几乎不能下结论"A 比 B 强"，需要做显著性检验。现代实践逐渐转向 pairwise，single-grading 主要做诊断（看模型在哪一类 prompt 上掉分）。

### 2.3 Reference-based

输入 $(x, y, y^*)$，$y^*$ 是参考答案（人工写的或 strong model 的输出），judge 比较 $y$ 与 $y^*$ 的相对质量，给分或选优。

代表：**AlpacaEval 2.0**（与 GPT-4-1106 baseline 比胜率）、**Arena-Hard**（与 Llama-3.1-405B baseline 比胜率）。reference-based 本质上是 pairwise 的特例（baseline = 一个固定 model），但因为 baseline 永远固定，**结果是稳定的相对量**，跨实验可比。

实战上 reference-based pairwise 是 SOTA：既继承了 pairwise 的低 noise，又通过固定 baseline 拿到了"绝对意义上的 ranking"，所以 Arena-Hard 与 Chatbot Arena Elo 的相关性能到 0.95+。

### 2.4 主流 benchmark 对照

| Benchmark | 形式 | 题量 | Judge | Baseline | 关键设计 |
|---|---|---|---|---|---|
| **MT-Bench** [Zheng 2023] | single-grading + pairwise | 80（多轮） | GPT-4 | 无 | 8 类覆盖；现代基线 |
| **Arena-Hard** [Li 2024] | pairwise | 500 | GPT-4-Turbo | Llama-3.1-405B | 从 Chatbot Arena 难题筛；与人评 Elo 相关 0.95+ |
| **AlpacaEval 2.0** [Dubois 2024] | pairwise + LC-WR | 805 | GPT-4-1106 | GPT-4-1106 | length-controlled 校正长度 bias |
| **Chatbot Arena** [LMSYS] | pairwise（人评） | $\infty$ | 人类 | 全部模型互比 | Elo / BT 拟合；金标但慢 |
| **WildBench** | both grading + pairwise | 1024 | GPT-4o | 多 baseline | 真实用户日志，难度更高 |

### 2.5 Length-Controlled Win Rate（LC-WR）

[Dubois 2024] 观察到一个核心问题：**长 response 系统性赢**——judge 看到长输出会觉得"更全面更用心"。如果 model A 平均生成 800 token、model B 平均 400 token，A 的 raw win rate 可能比真实质量高 5-10%。

LC-WR 的做法是用 logistic regression 把 win rate 与 length 解耦：

$$\text{logit}(P(A \succ B \mid x)) = \alpha + \beta_q \cdot \text{quality} + \beta_l \cdot (\text{len}_A - \text{len}_B)$$

回归出 $\beta_l$ 后，把它的 length 贡献从 win rate 里减掉，得到 "length-controlled win rate"。AlpacaEval 2.0 leaderboard 上的 LC-WR 与 Chatbot Arena Elo 的相关性显著高于 raw WR（0.98 vs 0.94），说明 length bias 真的在 raw WR 里污染了 5%+。

---

## 3. 最小代码示例

### 3.1 Pairwise judge（含 position randomize）

```python
import json, random
from openai import OpenAI
client = OpenAI()

PAIRWISE_PROMPT = """[Instruction]
You are an impartial judge. Compare two responses to the same user question
and decide which one is better. Consider helpfulness, accuracy, depth, and
safety. Do NOT prefer longer or more verbose responses; reward conciseness
when appropriate. Ignore which response is labeled A or B.

[Question]
{question}

[Response A]
{a}

[Response B]
{b}

Output strictly one of: "A", "B", "tie". No other text."""

def pairwise_judge(question, resp_1, resp_2, judge="gpt-4o-2024-11-20"):
    swap = random.random() < 0.5                       # ❶ 随机 swap，消 position bias
    a, b = (resp_2, resp_1) if swap else (resp_1, resp_2)
    msg = PAIRWISE_PROMPT.format(question=question, a=a, b=b)
    out = client.chat.completions.create(
        model=judge, temperature=0.0,
        messages=[{"role": "user", "content": msg}],
    ).choices[0].message.content.strip().upper()
    if out not in {"A", "B", "TIE"}: return "tie"
    if swap:                                           # ❷ swap 回原顺序
        out = {"A": "B", "B": "A", "TIE": "TIE"}[out]
    return {"A": "model_1", "B": "model_2", "TIE": "tie"}[out]
```

关键点：(1) `temperature=0.0` 让 judge 决策可复现；(2) 每条样本独立 random swap，**消 position bias**——不 swap 的话 A 位胜率系统性高 5-15%，这是 LLM-as-Judge 第一杀手 bias；(3) prompt 显式写 "Do NOT prefer longer responses" 能把 length bias 砍掉一部分（但不能完全消，仍要靠 LC-WR）；(4) judge 偶尔会输出额外文字（"A is better because..."），用 strict format check + 兜底 tie。

### 3.2 MT-Bench 风格 single-grading

```python
SINGLE_PROMPT = """[Instruction]
You are an impartial judge. Rate the assistant's response on a scale of 1-10:
- 10: outstanding (helpful, accurate, well-structured, no fluff)
-  7: solid (correct + helpful but minor issues)
-  4: weak (partially correct or off-topic)
-  1: harmful / completely wrong / refuses without reason
Avoid length / position / verbosity bias.

[Question]
{question}

[Reference answer (for grounding, may be empty)]
{reference}

[Assistant Response]
{response}

Output strictly: "Rating: <number>"
"""

import re
def single_grade(question, response, reference="", judge="gpt-4o-2024-11-20"):
    msg = SINGLE_PROMPT.format(question=question, reference=reference, response=response)
    out = client.chat.completions.create(
        model=judge, temperature=0.0,
        messages=[{"role": "user", "content": msg}],
    ).choices[0].message.content
    m = re.search(r"Rating:\s*(\d+(?:\.\d+)?)", out)
    return float(m.group(1)) if m else None
```

**reference answer** 是关键加分项——MT-Bench reasoning / math / coding 这三类题都给参考答案，让 judge 不需要"自己解一遍题"就能判定。没 reference 时 GPT-4 在 math 类问题上判错率显著上升（[Zheng 2023] 报 14% → 5%）。

### 3.3 完整 win rate + 95% CI

```python
import math
from collections import Counter

def winrate_with_ci(judgements, alpha=0.05):
    """
    judgements: list of "model_1" / "model_2" / "tie"
    returns: (wr_for_model_1, ci_low, ci_high)
    """
    n = len(judgements)
    cnt = Counter(judgements)
    # tie 算 0.5 win，是 Chatbot Arena / Arena-Hard 的标准做法
    wins = cnt["model_1"] + 0.5 * cnt["tie"]
    p = wins / n
    # 用 Wilson score interval 而不是 normal approx，小样本/极端 p 都更准
    z = 1.96 if alpha == 0.05 else 2.576
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return p, center - half, center + half

# 例：跑了 200 道 prompt
judgements = ["model_1"] * 110 + ["model_2"] * 80 + ["tie"] * 10
wr, lo, hi = winrate_with_ci(judgements)
print(f"Win rate: {wr:.3f}, 95% CI: [{lo:.3f}, {hi:.3f}]")
# Win rate: 0.575, 95% CI: [0.506, 0.642]
# CI 跨过 0.5 → 还不能说显著好于 baseline
```

**Wilson score interval** 比 normal approximation 在小样本（n < 200）下显著更准，是 LMSYS / OpenAI 评测脚本的默认。`tie` 算 0.5 是约定俗成，**配对实验（比较两个模型）** 还可以用 McNemar / paired bootstrap，进一步收紧 CI。

### 3.4 Self-preference bias 检测

```python
def self_preference_test(prompts, model_x_outputs, model_y_outputs,
                          judges=("gpt-4o", "claude-3-5-sonnet", "deepseek-chat")):
    """
    用多个 judge 评同一对 (X, Y)，看不同 judge 给出的 win rate 差多少。
    如果某个 judge 是 X 的 family（比如 GPT-4 评 GPT-4 输出），
    它给 X 的 win rate 通常比其他 judge 高 3-10%。
    """
    results = {}
    for j in judges:
        verdicts = [pairwise_judge(p, x, y, judge=j)
                    for p, x, y in zip(prompts, model_x_outputs, model_y_outputs)]
        wr, _, _ = winrate_with_ci(verdicts)
        results[j] = wr
    spread = max(results.values()) - min(results.values())
    print(f"WR by judge: {results}")
    print(f"Spread: {spread:.3f}  (>0.05 提示 self-preference bias 显著)")
    return results
```

实战发现：评 GPT-4 输出时 GPT-4 当 judge 给的 WR 经常比 Claude judge 高 5-8%。**生产严肃评测必须 cross-judge**（至少 2 家 family 的 judge 取均值），避免循环裁判。

---

## 4. 工程踩坑与经验

- ❗ **必须 randomize position（A/B 对调），否则 win rate 偏 5-15%**——[Wang 2023] 在多个 judge 上系统验证：即便 GPT-4 这种最强 judge，A 位仍系统性受偏好。Arena-Hard 的标准做法是 **每对 judge 两次（A-B 一次、B-A 一次）取均值**，比单次随机更彻底。Position bias 是 LLM-as-Judge 第一杀手，遗漏直接让结论失真。
- ❗ **judge 用 GPT-4 评 GPT-4 输出 → self-preference bias**——judge 偏向自己 family 的输出，GPT-4 评 GPT-4 / Claude 评 Claude 都有 3-8% 系统偏置。生产严肃评测必须 cross-judge：GPT-4o + Claude 3.5 Sonnet + DeepSeek-V3 多 judge 取均值，或者**至少不要让 judge 与被评模型同 family**。
- ❗ **Length 不控制时 verbose model 系统性赢**——同一个观点写 800 字与 200 字，judge 偏好长的；这让"加 markdown / bullet 装饰" 这种廉价 trick 直接赢 5-10%。缓解：(a) prompt 里写 "do NOT prefer longer"；(b) 用 LC-WR；(c) 数据预筛掉长度差极端的对；(d) **训练阶段** 要做 length normalize（10.x reasoning 章节会展开）。
- ❗ **judge sample 数 < 100 → CI 太宽，难做 statistically significant 比较**——按 Wilson interval，n=80（MT-Bench 题量）时 95% CI 半宽通常 ±10%，意味着 win rate 差 < 10% 都不显著；n=500（Arena-Hard）半宽 ±4%；n=2000+ 才能稳测 ±2% 的差异。生产线性能比较至少 500 题起步。
- ❗ **judge prompt 措辞影响巨大**——"more helpful" vs "more factually accurate" vs "more concise" 评出来的 winner 可能完全不同。所以**比较 A vs B 时 judge prompt 必须固定**，且要在 prompt 里写明评测维度优先级（一般顺序：accuracy > helpfulness > clarity > brevity > style）。换 prompt 重跑要重新发布数字。
- ❗ **中文场景用英文 judge 准确率下降**——GPT-4 / Claude 在英文 chat 上一致率 80%+，在中文 chat 上掉到 70-75%。需要：(a) judge prompt 用中文写；(b) 用对中文友好的 judge（DeepSeek-V3 / Qwen-Max / Doubao-Pro）；(c) 双语 judge 取均值。
- ❗ **Reasoning task 用 GPT-4o 当 judge 不如 R1 / o1**——评 math / coding / 复杂 reasoning 时，judge 自己得能"解一遍题"才能判对错。chat-only 的 GPT-4o 评 reasoning 输出会被表面流畅性骗，**reasoning model judge（DeepSeek-R1 / o1-mini）** 的 reasoning task 一致率更高（[Lambert 2024] RewardBench reasoning 子集证据明显）。
- ❗ **judge 对 verbosity / 装饰性 markdown 加分**——同一个内容，全文加上 `**bold**` / `### 标题` / `1. 2. 3.` 列表，judge 的 win rate 能涨 3-5%。这是与 length bias 不同的独立 bias，缓解：prompt 里写 "ignore formatting unless required"，或者**评测前对所有响应做 markdown strip 归一化**。
- ❗ **judge 输出格式偶尔会跑偏**——即使要求 "Output A or B"，GPT-4 偶尔会输出 "A is better because..."，要做 strict regex + 兜底 tie，不要 silently 当 A 处理。统计 fallback rate（一般 < 1%），>5% 说明 prompt 设计有问题。
- ❗ **不要用 7B 当 judge**——[Zheng 2023] 报 7B-13B judge 与人评一致率只 65-70%，13B-30B 也只 70-75%，**70B+ 才到 80%**。如果必须自托管 judge，最低门槛是 Llama-3.1-70B-Instruct / Qwen2.5-72B 这种规模。Open-source judge 第一选择是 Skywork-Critic-Llama-3.1-70B 这种为 judge 任务专门 finetune 的模型。

---

## 5. 经典 paper

- **Zheng et al., 2023 — Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena** — LLM-as-Judge 的奠基论文。提出 MT-Bench / Chatbot Arena 两个评测体系，**第一次系统给出 GPT-4 与人评一致率 80%+ 的实证**（与人-人之间一致率 75-80% 同档），并讨论了 position / verbosity / self-enhancement 三大 bias 与缓解。本节 §1.1 / §2.1 / §4 全部以这篇为底座，强烈精读。
- **Wang et al., 2023 — Large Language Models are not Fair Evaluators** — 系统性、大规模实测 LLM-as-Judge 的 bias，给出 position bias 的量化数据（同一对 swap 顺序 win rate 能差 ±10-20%）以及缓解方案（calibration + multiple evidence calibration）。读这一篇能从"知道有 bias"升级到"知道每种 bias 严重到什么数值量级"。
- **Li et al., 2024 — From Crowdsourced Data to High-Quality Benchmarks: Arena-Hard Auto** — Arena-Hard 论文。从 Chatbot Arena 真实日志中筛 500 道高 disagreement、高难度的题做 auto benchmark，**与 Chatbot Arena Elo 相关性 0.95+**，是当下 chat 评测最重要的工程产出之一。读它能学会"如何把人评数据反过来教自动 judge"。
- **Dubois et al., 2024 — Length-Controlled AlpacaEval** — 提出 LC-WR（用 logistic regression 把 length 与 quality 解耦），AlpacaEval 2.0 的核心。本节 §2.5 公式来自这里，是 length bias 的事实标准缓解。
- **Lambert et al., 2024 — RewardBench: Evaluating Reward Models for Language Modeling** — meta-benchmark：评测 RM / judge 本身。数据是若干 (prompt, chosen, rejected) 已知偏好对，看 RM/judge 选 chosen 的准确率，分 chat / chat-hard / safety / reasoning 4 个 split。读完能选 RM、能 sanity check 自己训的 judge 对不对。

---

## 6. 自测与面试题

**Q1（设计）：** 给定一个 prompt 和两个模型 response，写一个 pairwise judge prompt，要求消 position bias 与 length bias、并显式给出评测维度。

<details>
<summary>Answer sketch</summary>

完整 prompt 至少包含：

- **角色**：明确说"impartial judge"，不要带 emotional / persona 描述
- **任务**：compare A vs B, output one of {A, B, tie}
- **评测维度**：列优先级，例如 accuracy > helpfulness > clarity > brevity（具体看场景）
- **bias 警告**：
  - "Do NOT favor longer responses; reward conciseness when appropriate"
  - "Do NOT favor more decorated formatting (bullets, bold, headers)"
  - "Ignore which is labeled A or B; judge solely on content"
- **输出约束**：strict format `"A" / "B" / "tie"`，方便 regex 解析
- **加分**：reference answer（可选）做 grounding；one-shot 示例进一步对齐 judge 风格

**配套工程**：
- 调用层 random swap（§3.1 ❶）+ swap 回（❷）；或 swap 双方各跑一次取一致结论
- temperature=0、judge 模型固定版本号（gpt-4o-2024-11-20 而不是 gpt-4o，避免 silently 升级）
- fallback tie 处理 + 统计 fallback rate

</details>

**Q2（bias）：** 列出 LLM-as-Judge 的 5 种典型 bias，每种给 1 个缓解方法。

<details>
<summary>Answer sketch</summary>

| Bias | 表现 | 缓解 |
|---|---|---|
| **Position bias** | A 位胜率系统性偏高 5-15% | random swap 或 swap 双方各跑一次取均值 |
| **Length bias** | 长 response 系统性赢 | LC-WR（length-controlled）/ prompt 里写 "do NOT favor longer" / 数据筛掉长度差极端对 |
| **Verbosity / formatting bias** | markdown 装饰 / bullet 加分 | prompt 写 "ignore formatting unless required" / 评测前 strip markdown |
| **Self-preference bias** | judge 偏向同 family 输出（GPT-4 评 GPT-4 偏高 3-8%） | cross-judge：GPT-4 + Claude + DeepSeek 取均值，至少避免同 family |
| **Sycophancy bias** | 附和 prompt 里既有立场的回答更易得分 | judge prompt 强调 "evaluate independently of any opinions in the question" |

**加分**：能补 (a) **reasoning bias**——chat-only judge 评 reasoning task 不准，缓解用 reasoning model judge（R1 / o1）；(b) **first-token bias**——judge 倾向于复述自己第一个 token，所以输出格式应当迫使 judge 先 reason 再决策（CoT-prompt judge）。

</details>

**Q3（实战）：** 你训了一个新 7B chat 模型，要"严肃证明它比上一版 baseline 好"。列出完整评测设计：数据集、judge 选择、样本数、显著性检验、报告口径。

<details>
<summary>Answer sketch</summary>

参考 Arena-Hard / AlpacaEval 现代实践，完整设计如下：

**1. 数据集组合（覆盖 5 维度，且 LLM-as-Judge 适用层）**
- **Arena-Hard 500 题**：通用 chat 难题，pairwise vs Llama-3.1-405B baseline；与 Chatbot Arena Elo 相关性高
- **AlpacaEval 2.0 805 题**：通用 instruction，LC-WR 校正长度
- **MT-Bench 80 题**：诊断用，看在哪一类（writing / math / coding / ...）掉分
- **WildBench 1024 题**（如果想覆盖真实分布）
- 中文场景额外加 **AlignBench / SuperCLUE-OPEN**（中文 LLM-as-Judge）

**2. Judge 选择（cross-judge 必做）**
- 至少 2 个 judge：GPT-4o + Claude 3.5 Sonnet（避免单一 family bias）
- reasoning 类题加 DeepSeek-R1 / o1-mini judge
- 中文题加 DeepSeek-V3 / Qwen-Max judge
- 报告**取多 judge 均值** + spread（spread > 5% 提示有 bias，需进一步分析）

**3. 样本数与 CI**
- Arena-Hard 500 题：Wilson 95% CI 半宽 ±4%，能稳测 ±5% 以上的差异
- 如果差异预期 < 5%，扩展到 2000+ 题（自筛 prompt 或重复评测取均值）
- 报告必须给 Wilson / bootstrap 95% CI，不只是 point estimate

**4. 控制 bias**
- **position randomize**：每对 swap 双方各跑一次取均值
- **length control**：用 LC-WR；同时报 raw WR（透明）
- **format normalize**：评测前 strip markdown / 标准化 bullet 格式
- judge 模型固定版本号（gpt-4o-2024-11-20 而非 gpt-4o）+ temperature=0

**5. 显著性检验**
- **paired bootstrap**（更严格于 Wilson）：对同一批 prompt resample，看 wr_diff 的 CI 是否过 0
- **McNemar test**：配对二项检验，对 (A wins, B wins, both right, both wrong) 四格表
- 报告 **p-value + effect size**（Cohen's h），只报 p < 0.05 不够

**6. RM / 离线 sanity check**
- 训练完 baseline RM（可用开源 Skywork-Reward / ArmoRM）
- 离线给两模型每 prompt 算 reward，看 reward 分布是否与 judge win rate 一致
- 不一致需要排查（RM 漂 or judge 偏）

**7. 报告口径**
- "Model X 在 Arena-Hard（500 题）上对 Llama-3.1-405B 的 LC-WR 为 62.3%（95% CI [58.1%, 66.4%]，GPT-4o + Claude judge 均值，position swap 完整）；对上一版 baseline 的 paired LC-WR 为 54.1%（CI [50.6%, 57.5%]，p < 0.01 by McNemar）"
- 同时附 MT-Bench 8 类细分（看是否有维度回退）
- **绝不能**：单 judge、单 run、不报 CI、不 swap order、用饱和 benchmark 报新数字

**8. 加分**
- 上线前再做一轮 **Chatbot Arena pre-launch**（小流量人评）拿真 Elo 校验
- 持续监控：模型上线后用 production logs 抽样跑 LLM-as-Judge 做 regression
- 关键 release 用 **不同 judge 复跑**作为 sanity check

</details>

---

## 7. 延伸阅读

- [LMSYS FastChat / MT-Bench 仓库](https://github.com/lm-sys/FastChat) — MT-Bench 与 Chatbot Arena 的官方实现，包含 judge prompt 模板、双 judge 配置、Elo 计算脚本，工程上"怎么落地"的事实标准
- [Arena-Hard-Auto 仓库](https://github.com/lm-sys/arena-hard-auto) — Arena-Hard 的运行脚本与 leaderboard，含 swap-order pair-judge 与 BT 拟合代码
- [AlpacaEval 仓库](https://github.com/tatsu-lab/alpaca_eval) — AlpacaEval 2.0 与 LC-WR 实现，含 length-controlled 回归代码
- [RewardBench Leaderboard](https://huggingface.co/spaces/allenai/reward-bench) — RM / judge 的 meta-benchmark 排行榜，选 RM 必看
- [Lambert 2024 — RLHF Book Ch.7 "Evaluation"](https://rlhfbook.com/c/07-evaluation.html) — Nathan Lambert 在写的开源 RLHF 教材中评测章节，把 judge / RM / human eval 的关系梳理得最干净
- 推荐继续读本教程的 **12.3 安全评测与红队**（HarmBench / GCG / jailbreak 谱系，与本节互补的另一类 judge：safety judge）、**9.2 RM 训练实操**（本节 judge 的反向应用：用 LLM-as-Judge 标的偏好对当 RM 训练数据）、**10.2 PRM 与 Process Reward**（judge 在 reasoning 中间步骤的应用）
