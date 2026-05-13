---
title: "10.1 CoT / Self-Consistency / ToT / GoT / Reflexion"
description: "在不改模型权重的前提下，通过 更多 inference token / 多次 sample / 显式搜索 / 失败反思 来换 accuracy 的一整套 inference-time reasoning 方法谱：CoT、Self-Consistency、ToT、GoT、Reflexion、Self-Refine、Best-of-N。"
---

> ⏱ 预计阅读 25 分钟 ｜ 难度 ★★ ｜ 前置：无（建议熟悉 chat-completion API）

## 一句话本节讲什么

在不改模型权重的前提下，通过 **更多 inference token / 多次 sample / 显式搜索 / 失败反思** 来换 accuracy 的一整套 inference-time reasoning 方法谱：CoT、Self-Consistency、ToT、GoT、Reflexion、Self-Refine、Best-of-N。

---

## 1. Mental model（直觉）

LLM 一次 forward 的算力是固定的。一个 70B model 在 GSM8K 上一次 sample 拿 50%，**不代表它的能力上限是 50%**——可能它会做、但 forward 一次没跑出来。

**第二维度的 scaling**：模型大小是第一维，**test-time compute 是第二维**。多花 5×、10×、100× 的推理 token，可能换来 10-30% 的 accuracy。这是 OpenAI o1 / DeepSeek-R1 时代之前就被反复验证过的规律——它们只不过把这件事 **从 prompt-time trick 内化到了 model 自己**。

不同方法本质是在回答 "**这些额外 compute 怎么花**" 这一问题：

```
                         单条 path                  多条 path
                    ┌──────────────────┐      ┌──────────────────┐
单步 generate       │   Vanilla        │      │  Best-of-N       │
                    │   (baseline)     │      │  Self-Consistency│
                    └──────────────────┘      └──────────────────┘
显式 reasoning      ┌──────────────────┐      ┌──────────────────┐
step                │   CoT            │      │  ToT / GoT       │
                    │   Plan-and-Solve │      │  (search)        │
                    └──────────────────┘      └──────────────────┘
带 feedback loop    ┌──────────────────┐      ┌──────────────────┐
                    │   Self-Refine    │      │  Reflexion       │
                    │   (self-critic)  │      │  (env verifier)  │
                    └──────────────────┘      └──────────────────┘
```

记住一条经验线：**同样多 compute，结构化方法 > 朴素采样**。但结构化方法的实现复杂度也指数级上升（ToT > SC > CoT），工程上要看 ROI。

---

## 2. 公式与原理

inference-time reasoning 没有训练 loss 可推，但有几条统计观察值得形式化。

**Self-Consistency 的 majority vote**。设 model 对问题 $x$ 的答案后验为 $p(y \mid x)$，sample $N$ 次得到 $\{y_1, \dots, y_N\}$。majority 估计：

$$\hat{y} = \arg\max_{y} \sum_{i=1}^{N} \mathbb{1}[y_i = y]$$

如果正确答案 $y^*$ 的概率 $p^* > 0.5$，由弱大数定律 $\hat{y} \to y^*$ 几乎必然。**关键**：CoT 让 $p(y^* \mid x)$ 从 0.3 升到 0.55，再叠 SC 把"超过 50% 的优势"放大成接近 100%。这就是 SC 加速远超 BoN 的根因——它利用了 reasoning path 的 **错误彼此独立、正确彼此一致** 这一不对称。

**Best-of-N 的期望提升**。设 reward model 打分 $r(y) \in [0,1]$，N 次 sample 后取 $\hat{y} = \arg\max_i r(y_i)$。当 RM 校准良好时，

$$\mathbb{E}[r(\hat{y})] = \mathbb{E}\left[\max_{i \le N} r(y_i)\right] \approx 1 - (1 - \bar{r})^N$$

边际收益按 $N$ 对数式衰减——这也是 BoN 一般取 $N \in \{8, 16, 32\}$ 不再加大的原因。

**ToT 的搜索复杂度**。设每个 thought 平均 expand $b$ 个子节点，深度 $d$，则节点总数 $O(b^d)$；每个节点要 model 自评一次 + LLM call 至少 2 次（generate + evaluate），实际 cost $\propto 2 \cdot b^d$。所以 ToT 普遍只在 $d \le 5, b \le 5$ 的小问题上跑（24 game、creative writing），不能直接 scale 到长 trajectory。

---

## 3. 最小代码示例

下面四段代码用 OpenAI / 兼容接口（chat-completion）作为 LLM 抽象，假设有一个 `llm(prompt, T=0.0, n=1)` helper 返回 `List[str]`。

### 3.1 Zero-shot CoT prompt

```python
ZSCOT = "Q: {question}\nA: Let's think step by step."

def cot_answer(q: str) -> str:
    out = llm(ZSCOT.format(question=q), T=0.0, n=1)[0]
    return extract_final_answer(out)  # 从 reasoning 末尾抽 "The answer is X"
```

一行 magic phrase（Kojima 2022）就能在 GSM8K 上让 PaLM-540B 从 17% → 41%。**这也是历史上 ROI 最高的 prompt trick**。

### 3.2 Self-Consistency

```python
from collections import Counter

CSCOT_FEW_SHOT = """Q: ... A: Let's think step by step. ... The answer is 18.
Q: {question}
A: Let's think step by step."""

def self_consistency(q: str, n: int = 20, T: float = 0.7) -> str:
    # 关键 1: temperature > 0.5，否则 n 次 sample 全一样
    samples = llm(CSCOT_FEW_SHOT.format(question=q), T=T, n=n)
    # 关键 2: 从每条 reasoning 中提取最终答案 (regex / "The answer is" 等)
    answers = [extract_final_answer(s) for s in samples]
    answers = [a for a in answers if a is not None]
    if not answers:
        return None
    # 关键 3: majority vote (生成式答案需先 cluster / canonicalize)
    return Counter(answers).most_common(1)[0][0]
```

`n=20, T=0.7` 是 Wang 2022 的常用配置。注意 `extract_final_answer` 在数学题上用 regex `r"answer is (-?\d+\.?\d*)"`，open-ended 任务要做 cluster。

### 3.3 简化 ToT（BFS + 自评 + prune）

```python
def tot_solve(problem: str, depth: int = 3, beam: int = 3, branch: int = 5) -> str:
    # 每个 node = 当前已生成的部分 reasoning（一个字符串）
    frontier = [""]
    for d in range(depth):
        candidates = []
        for path in frontier:
            # 1) expand: 让 model 生成 branch 个下一步 thought
            prompt = f"Problem: {problem}\nPartial reasoning so far:\n{path}\nGive ONE next step:"
            new_steps = llm(prompt, T=0.7, n=branch)
            for step in new_steps:
                candidates.append(path + "\n" + step)
        # 2) evaluate: 让 model 给每个 candidate 打 1-10 分（self-critique）
        scores = []
        for c in candidates:
            judge = f"Rate this reasoning from 1-10 on whether it leads to a correct solution:\n{c}\nScore:"
            try:
                scores.append(int(llm(judge, T=0.0)[0].strip().split()[0]))
            except Exception:
                scores.append(0)
        # 3) prune: 只保留 top-beam 条 path
        ranked = sorted(zip(scores, candidates), reverse=True)
        frontier = [c for _, c in ranked[:beam]]
    # 最后一轮的最佳 path → 让 model 写出 final answer
    final = llm(f"Reasoning:\n{frontier[0]}\nFinal answer only:", T=0.0)[0]
    return final
```

这是 ToT 的"教学版本"。Yao 2023 原版还区分 `propose / sample / vote` 等子策略，调参极敏感。

### 3.4 Reflexion loop

```python
def reflexion(task: str, verifier, max_trials: int = 4) -> str:
    memory = []  # 历次 reflection 的 lessons
    for trial in range(max_trials):
        hint = "\n".join(f"- {m}" for m in memory) if memory else "(none)"
        prompt = f"Task: {task}\nLessons from past attempts:\n{hint}\nSolve it:"
        attempt = llm(prompt, T=0.7, n=1)[0]
        ok, signal = verifier(attempt)         # 关键: 必须有 verifier (env / unit test)
        if ok:
            return attempt
        # 失败 → 让 model 自己反思 "为什么错"，写入 memory
        reflect = llm(
            f"Task: {task}\nYour attempt:\n{attempt}\nVerifier feedback: {signal}\n"
            f"In one sentence, what lesson should you remember next time?",
            T=0.7,
        )[0]
        memory.append(reflect.strip())
    return attempt  # 最后一次（即使失败）
```

`verifier` 是 Reflexion 区别于 Self-Refine 的关键——可以是 unit test、game env reward、math grader。**没有外部 signal，Reflexion 退化成自言自语**。

---

## 4. 工程踩坑与经验

- ❗ **CoT 在 < 7B model 上几乎无用**（Wei 2022 称之为 emergent ability）。在 7B-13B 上效果一般，常用的甜点是 **8B+ instruct model**。如果你给 1.3B model 加 CoT 没效果，是正常的，不是 prompt 写错。
- ❗ **Self-Consistency 的 temperature 必须 > 0.5**。`T=0` 时 `n=20` 全是一样的 sample，等于白烧 token。常用 0.7-1.0；过高（>1.2）会导致 reasoning 跑飞，反而正确 path 比例下降。
- ❗ **ToT 调参极烦**：`depth × branch × beam` 三个 hyperparameter 加 evaluation prompt template，每个都对结果敏感。生产环境基本只在 game / planning 这类难 baseline、小 state space 的任务上才值得上 ToT。
- ❗ **Reflexion 在没 verifier 的任务上没法跑**。HumanEval / GSM8K / Game-of-24 这些有 ground-truth checker 的任务才能用；写作、对话这类 open-ended 任务上 Reflexion 退化成 Self-Refine。
- ❗ **BoN 与 RM bias 强相关**。RM 的偏好漏洞会被 BoN 系统性放大，你最后选出的"最高分"答案可能是 reward hacking 的结果（如 RM 偏长答案 → BoN 选 N 中最长的）。N 越大放大越严重。
- ❗ **R1 / o1 之后，CoT prompting 不再需要**。reasoning model 自带 long-CoT，再加 "Let's think step by step" 反而干扰其内置 thinking template。但 **BoN / Self-Consistency / ToT search 在 reasoning model 上仍有效**——R1 + BoN 在 AIME 上仍能涨几个点。
- ❗ **Self-Consistency 的 vote 在生成式答案上难做**。multi-choice 直接 vote letter；math 题 vote 数字；但开放式 QA 你怎么 cluster "巴黎" 和 "Paris, France"？需要 LLM 二次 normalize 或 embedding cluster，工程量陡增。
- ❗ **CoT 会"诱导幻觉"**：让 7B model 多说话反而更容易自圆其说错误结论。短任务（情感分类、NER）上 CoT 经常掉点，**别无脑加**。

---

## 5. 经典 paper

- **Wei et al., 2022 — Chain-of-Thought Prompting Elicits Reasoning in Large Language Models** — CoT 的奠基论文，证明大模型加 step-by-step reasoning example 在数学/常识 benchmark 上质变；**emergent ability** 概念也来自这里。是后续所有 reasoning 工作的起点。
- **Wang et al., 2022 — Self-Consistency Improves Chain-of-Thought Reasoning** — 把"多 sample + majority vote"做成 CoT 的标配 add-on，工程上几乎零成本接入，**至今仍是最 cost-effective 的 inference-time scaling**。
- **Yao et al., 2023 — Tree of Thoughts: Deliberate Problem Solving with Large Language Models** — 把 reasoning 显式建成 tree + 自评 + 搜索的范式开端，启发了后续 LATS / MCTS-LLM 等大量工作。Module 14.4 Planning 一章会回到 ToT。
- **Shinn et al., 2023 — Reflexion: Language Agents with Verbal Reinforcement Learning** — "verbal RL" 概念的提出，agent 失败 → 自然语言反思 → 写入 memory；后续 agent self-improvement 的鼻祖之一。
- 加分阅读：**Madaan 2023 — Self-Refine**（无 verifier 的 critique-refine 循环）、**Besta 2023 — Graph of Thoughts**（ToT 推广到 DAG）、**Kojima 2022 — Large Language Models are Zero-Shot Reasoners**（"Let's think step by step" 的出处）。

---

## 6. 自测与面试题

**Q1：** 把 CoT、Self-Consistency、ToT、Reflexion 四种方法按 **inference cost、适用任务、是否需要外部 verifier** 三个维度对比。

<details>
<summary>Answer sketch</summary>

应当包含一张表 + 关键观察：

| 方法 | inference cost | 适用任务 | 是否需要 verifier |
|---|---|---|---|
| CoT (zero/few-shot) | 1× | 通用 reasoning（math / commonsense） | 无 |
| Self-Consistency | 5-20× | multi-choice / math（答案可 cluster） | 无（majority vote） |
| ToT | 10-100× | 难搜索任务（Game-of-24 / planning / creative writing） | model 自评（也可外部） |
| Reflexion | 5-50×（含 retry） | 可 verify 的任务（HumanEval / 数学 / game env） | **必须有外部 verifier** |

关键观察：
- 朴素多 sample（SC / BoN）vs 显式搜索（ToT）：cost 量级和实现复杂度都拉开
- Reflexion 与其他三者最大的差异是 **依赖外部 signal**，否则退化成 Self-Refine

</details>

**Q2：** 你的 7B model 在 GSM8K accuracy 50%。在不重训权重的前提下，列出 3 种 inference-time scaling 方法以及预期收益与代价。

<details>
<summary>Answer sketch</summary>

应该覆盖至少 3 个层次的方法 + 量化代价：

1. **Self-Consistency**（最 ROI）
   - 配置：`n=20, T=0.7` + few-shot CoT prompt
   - 预期：50% → 60-65%（10-15 个点，符合 Wang 2022 在 GSM8K 上的报告）
   - 代价：~20× inference token

2. **Best-of-N + RM**
   - 需要一个 reward model（PRM 更好，见 10.2）
   - 预期：N=16 时 50% → 58-62%
   - 代价：~16× sample + RM forward；当心 RM hacking

3. **Reflexion**（需要 verifier）
   - GSM8K 有 ground-truth answer，可作为 binary verifier
   - 预期：max 4 轮约 50% → 60%
   - 代价：失败样本平均 3-4× 调用

加分点：提到 **R1-distill-7B** 这类 reasoning model 是更根本的解决方案（直接训练 long-CoT），而不是堆 inference scaffold。

</details>

**Q3：** R1 / o1 等 reasoning model 出现之后，本节讲的这些 prompt-time 方法还有意义吗？哪些 obsoleted、哪些仍重要？

<details>
<summary>Answer sketch</summary>

有意义，但用法变了。需要区分讨论：

- **被内化（基本 obsoleted）**
  - **CoT prompting**：reasoning model 一次 generation 自带 5k-50k token 的 thinking，再加 "Let's think step by step" 多余甚至有害（破坏其 special token 模板）
  - **Self-Refine 的简单 critique 循环**：reasoning model 在 thinking 阶段已经在自我 critique
  - **Plan-and-Solve**：内化为 thinking 内的 high-level plan

- **仍然有效**
  - **Self-Consistency / BoN**：reasoning model 也有 sampling variance，R1 + BoN（或 SC）在 AIME 等高难 benchmark 上仍能再涨几个点
  - **Reflexion**（在 agent 场景下）：跨 trajectory 的 lesson learning 不是单次 long-CoT 能替代的，agent 多 episode 失败重试仍需要 explicit memory
  - **ToT-like search**：对于 search 空间巨大的任务（如 LATS for web agent），单条 long-CoT 仍然不够，需要并行展开 + reward-guided search

- **趋势**：reasoning model（一次 generation 内的隐式 search）+ 外部 search scaffold（多次 generation 之间的显式 search）是互补的，而不是替代关系。10.4 推理时搜索会展开。

</details>

---

## 7. 延伸阅读

- [Wei et al. 2022 CoT 原 paper (arXiv 2201.11903)](https://arxiv.org/abs/2201.11903) — 必读，emergent ability 的实证起源
- [Wang et al. 2022 Self-Consistency (arXiv 2203.11171)](https://arxiv.org/abs/2203.11171) — 工程上必接入的"零成本"加分项
- [Yao et al. 2023 Tree of Thoughts (arXiv 2305.10601)](https://arxiv.org/abs/2305.10601) + [官方 repo](https://github.com/princeton-nlp/tree-of-thought-llm) — 跑一下 Game-of-24 demo，对 ToT 实现复杂度有体感
- [Shinn et al. 2023 Reflexion (arXiv 2303.11366)](https://arxiv.org/abs/2303.11366) — "verbal RL" 的 mental model 对后续 agent self-improve 的工作（ReAct → Reflexion → LATS → Agent-R1）影响深远
- [Lilian Weng — LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) — 把 CoT / ToT / Reflexion 放进 agent 全景的优秀综述
- 推荐继续读：本教程 **10.2** PRM（让 verifier 从二值变成 step-level 信号）、**10.4** 推理时搜索（BoN / MCTS / verifier-guided 的系统化）、**Module 14.2** ReAct + Reflection agent（Reflexion 在 agent 上的工程落地）
