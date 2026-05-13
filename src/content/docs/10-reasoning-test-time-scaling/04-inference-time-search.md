---
title: "10.4 推理时搜索：Best-of-N / MCTS / Verifier-guided"
description: "把 Best-of-N、Self-Consistency、ToT、MCTS、LATS、Beam search 统一在 \"花更多 inference compute 换 accuracy\" 的 N-path trade-off 视角下，讲清楚每种方法的 search 形式、cost 量级、对 verifier / PRM 的依赖，以及在 R1 时代它们的边界还剩多少。"
---

> ⏱ 预计阅读 35 分钟 ｜ 难度 ★★★ ｜ 前置：10.1 CoT/SC/ToT；10.2 PRM；建议先看 10.3 R1

## 一句话本节讲什么

把 Best-of-N、Self-Consistency、ToT、MCTS、LATS、Beam search 统一在 **"花更多 inference compute 换 accuracy"** 的 N-path trade-off 视角下，讲清楚每种方法的 search 形式、cost 量级、对 verifier / PRM 的依赖，以及在 R1 时代它们的边界还剩多少。

---

## 1. Mental model（直觉）

10.1 末尾留了一句话：**模型大小是第一维 scaling、test-time compute 是第二维 scaling**。第二维有一个统一的 "N" 参数——一次推理产生多少条 path。

```
                    quality
                       ▲
                       │             ┌─ MCTS / LATS  (verifier-guided search)
                       │            ╱
                       │       ┌── BoN / SC          (flat sampling)
                       │      ╱
                       │  ┌── Greedy                  (N = 1)
                       └──┴────────────────────────────► cost (N)
                         1   8   32   128   1k   10k
```

N=1 是 greedy，N>1 时**怎么花这些 path** 决定了你属于哪一种方法：

- **Sample + select**：N 条独立 sample，用 verifier / RM / PRM 选 1 条 → **Best-of-N**
- **Sample + vote**：N 条独立 sample，按答案 majority vote → **Self-Consistency**
- **Search**：tree / graph 显式展开，节点间共享前缀，用 value 引导 expand → **ToT / MCTS / LATS**

三者的 cost 量级与 verifier 依赖差别巨大。一句话区分：

> **flat = 简单粗暴、log 收敛；tree search = 复杂精细、高 cost 但能 reach 更难的解**。

为什么 tree search 能 reach flat sample 摸不到的解？因为**树结构允许在中间节点回退**——flat sample 一旦走错第 2 步，后面 998 个 token 全部浪费；tree search 在第 2 步就被低 value child 剪掉，重新从 sibling 出发。形式化地讲，flat sample 的有效搜索空间是 $|y|$ 条独立 trajectory，tree search 则是 $K^d$ 条共享前缀的 trajectory——同样 LLM 调用次数，tree 覆盖的"独立解"更多，但代价是 evaluator 必须能在 partial 状态上给 signal（这就是 PRM 在 search 里的核心价值）。

第二条心智线索：**search 必须有 leaf evaluation**。BoN 靠 RM、SC 靠 vote、MCTS 靠 PRM/verifier，**没有 leaf 信号 search 退化为随机采样**。这条线把 10.2 的 PRM 和本节直接连起来——PRM 不是 RL 的工具，更天然是 search 的 value function。

第三条线索：R1 / o1 出现后，"long-CoT 内化了 search-like reasoning"。**外部 search 还有意义吗？** 有，但 margin 缩小了。R1 + BoN 在 AIME 上仍涨 2-5 个点，但增益远低于 base model 时代的 10-20 点。这是 §2.5 要展开的 trade-off。

---

## 2. 公式与原理

### 2.1 N-path 统一视角

设 policy $\pi$ 对 question $x$ 输出 trajectory $y$ 的分布 $\pi(y \mid x)$。任一 inference-time 方法可写成

$$\hat{y} = \mathcal{A}\bigl(\{y_i\}_{i=1}^N,\, V\bigr),\quad y_i \sim \pi(\cdot \mid x)$$

其中 $\{y_i\}$ 是 N 条候选（独立 sample 或 search 节点），$V$ 是某种 evaluator（RM / PRM / verifier / 自评 / vote），$\mathcal{A}$ 是聚合算子。N=1, $\mathcal{A}=\text{id}$ 即 greedy。

| 方法 | 候选生成方式 | $V$（leaf evaluator） | $\mathcal{A}$（聚合） |
|---|---|---|---|
| Greedy | $\arg\max_y \pi(y\mid x)$ (1 条) | 无 | id |
| Best-of-N | i.i.d. sample N 条 | RM / PRM / verifier | $\arg\max_i V(y_i)$ |
| Self-Consistency | i.i.d. sample N 条 | 无（用答案 cluster） | majority vote on $\text{ans}(y_i)$ |
| ToT | tree expand + 自评 prune | LLM self-rate / PRM | tree 顶端 max |
| MCTS | UCB-guided expand + rollout | PRM / verifier | 根节点 visit-weighted child |
| LATS | MCTS + reflection memory | env reward + value | 同 MCTS |
| Beam search | token-level top-K 扩展 | 累计 logprob | 末端 top-1 |

记一句话：**N 维 trade-off 的核心问题是 candidate generation 与 leaf evaluation 这两件事各自怎么做、怎么搭**。

### 2.2 Best-of-N (BoN) 的 scaling law

设单 sample 准确率 $p_0 \in (0,1)$。**理想 verifier**（输出 0/1 与 ground truth 一致）下，N 次 sample 至少有一条对的概率：

$$p_N = 1 - (1 - p_0)^N$$

$p_0 = 0.5$ 时 $p_N = 1 - 0.5^N$，N=8 即达 99.6%——这是 BoN 的"理论上限"。

但实际 verifier 不完美——RM / PRM 有 bias 和 noise。Cobbe 2021（GSM8K verifier）的实证规律是：

$$\text{accuracy}(N) \approx a + b \cdot \log N$$

经验上 N=8 → 16 涨 3-5 个点，N=16 → 32 涨 1-3 个点，N=64 之后基本饱和。这是 BoN **log-收敛** 的根本原因：

- 一方面是数学上 $1 - (1-p_0)^N$ 在 $p_0$ 中等水平时迅速饱和；
- 另一方面是 imperfect verifier 在 N 增大时更容易选到 "RM hacking" sample（高 RM 分但答错），margin 抵消。

工程上的甜点：**N ∈ {8, 16, 32}**，超过 64 通常不值得，除非 verifier 是 rule-based（math 字符串匹配 / code unit test），那时可以放心拉到 N=256 甚至更大（Lightman 2023）。

PRM-aggregated BoN 见 10.2 §2.3：把 PRM 的 step-level 分用 min / product / mean 聚合成 trajectory 分，min 经验最优。**PRM-BoN 比 ORM-BoN 在 MATH 上高 5-15 个点，且饱和更晚**。

### 2.3 Self-Consistency vs BoN

二者都是 flat N-sample，差别只在 $V$：

- BoN：$V$ 是 learned RM / PRM / verifier
- SC：$V$ 是 vote 算子（不需要 model）

工程含义：

- SC **不需要 RM**，零额外成本，最 cost-effective（10.1 §3.2）
- 但 SC 只对 **discrete answer**（multi-choice / 数学数字 / 代码精确匹配）work——open-ended 答案没法 vote
- BoN 通用，但需要训 / 用一个 RM

一个被忽视的事实：**SC 等价于"用 majority over generations 当 verifier 的 BoN"**——它隐含假设错误答案彼此独立、正确答案彼此一致。这个假设在 short-form numerical answer 上 hold，在 long-form 上不 hold。

为什么 SC 在数学题这类任务上有时甚至打赢 BoN？因为 majority vote 是**无 bias 的 evaluator**——只要正确答案的概率 $p^* > 0.5$，N 大时必然收敛到正确答案；而 learned RM 有 bias（偏长 / 偏 markdown / 偏特定风格），N 大反而放大 bias。结论：**有 ground truth + 答案离散可 cluster 的任务，SC 是 BoN 的强基线甚至上限**。这就是为什么很多 paper 报告 BoN 时必须同时报告 SC——如果你的 BoN 打不过 SC，说明你的 RM/verifier 在这个任务上还不如纯 vote 有用。

### 2.4 MCTS for LLM reasoning

经典 MCTS 四步循环：

```
Root: question x
              ┌─ select  (UCB1: 选 child argmax_c [Q(c) + c_puct · √(ln N(parent)/N(c))])
              │
         ┌────┴────┐
       child1   child2          ← expansion (LLM propose K next thoughts)
       /   \      |
     ...   ...   leaf
                  │
                  └─ simulate (rollout to terminal, get verifier reward)
                  │
                  └─ backup    (沿 path 更新 Q, N)
```

把这套搬到 LLM reasoning 上：

- **node** = "已生成的部分 reasoning 前缀"（一段 thought / 一个 step）
- **edge** = "LLM 在该前缀下生成的下一段 thought"
- **expand**：让 LLM 在该 node 上 sample $K$ 个 next-step candidate（如 K=5），每个成为 child
- **simulate**：从 leaf 让 LLM rollout 到 terminal answer，用 verifier (math / code) 或 PRM 给 reward
- **backup**：把 reward 沿 path 回传，更新每个 node 的 $Q, N$

**UCB1 / PUCT 选择规则**（c_puct 是探索系数）：

$$\text{UCB}(s, a) = Q(s, a) + c_{\text{puct}} \cdot \sqrt{\frac{\ln N(s)}{N(s, a)}}$$

或 AlphaZero 风格的 PUCT：

$$\text{PUCT}(s, a) = Q(s, a) + c_{\text{puct}} \cdot P(a \mid s) \cdot \frac{\sqrt{N(s)}}{1 + N(s, a)}$$

其中 $P(a \mid s)$ 是 prior（可由 LLM logit 给出）。c_puct 经典默认 $\sqrt{2} \approx 1.41$，但 LLM reasoning 上常用 1.0-2.0 之间手调——**对结果极其敏感**（§4 踩坑）。

代表工作：

- **rStar** (Microsoft 2024)：MCTS + 多种 reasoning action（propose subgoal / restate problem / self-rephrase），互相验证 trajectory
- **MCTSr** (Zhang 2024)：MCTS + self-refine + math verifier，7B model 在 MATH 上接近 GPT-4
- **AlphaProof / AlphaGeometry** (DeepMind 2024)：MCTS + LLM proposer + 形式化 verifier，IMO 2024 银牌

实证：math/code 上 MCTS 比 BoN 涨 5-15 个点，**但 cost 高 1-2 个数量级**（每个 root 50-500 个 LLM call）。

把 MCTS 在 LLM 上 work 的关键拆成三件事：

1. **action space 设计**：不是简单的"continue 一段"，而是定义一组高层 action（rStar 用 propose-subgoal / restate-problem / self-rephrase / generate-next-step / verify 五种 action），让搜索更有结构。
2. **value function 来源**：rollout to terminal + verifier (慢但准) vs PRM 直接打分 (快但 noisy) vs 二者混合 (推荐)。混合策略：浅层 node 用 PRM 快速剪枝、深层 node 才 rollout。
3. **prior 的注入**：PUCT 里的 $P(a \mid s)$ 用 LLM 自己的 logit 给——意味着 search 不会盲目均匀展开，而是优先扩 LLM 觉得有 promise 的方向。这就是 AlphaZero "self-play 的 LLM 版"。

### 2.5 ToT / GoT 与 MCTS 的关系

ToT 是 MCTS 的简化版：

| 维度 | ToT (Yao 2023) | MCTS |
|---|---|---|
| selection | BFS / DFS (无 UCB) | UCB / PUCT |
| evaluation | LLM self-rate | PRM / verifier rollout |
| backup | 无 (一次性 rank) | 有 (每次 simulate 后回传) |
| value learning | 无 | 有 (Q 在多次 visit 中收敛) |
| 适用范围 | depth ≤ 5, branch ≤ 5 | depth ≤ 20, branch ≤ 10 |

ToT 的主要工程优势是**实现极简、无需 verifier**（自评即可），但搜索效率低、cost 大。MCTS 工程复杂得多，但配合 PRM / verifier 可以在更深的 reasoning tree 上有效。

GoT (Besta 2023) 把 ToT 推广到 DAG（允许 thought merge / split），表达力更强但调参更难，工业落地少。

### 2.6 LATS (Language Agent Tree Search)

Zhou 2023 把 MCTS + Reflexion 统一在 agent 任务上：

- node = agent 状态（包括 history / observation）
- edge = action（tool call / reasoning step）
- value = (env feedback) + (LLM self-value)
- 失败 path 的 reflection 写入 memory，影响后续 selection 的 prior

**适用**：复杂 web 任务（WebShop / HotpotQA）、multi-step planning。**不适用**：纯 reasoning 任务（math/code），因为 ToT/MCTS 更轻。

LATS 与 MCTS 的核心差别有三处：(1) **value 来源混合**——env 反馈 (硬信号) + LLM 自评 (软信号)，加权融合；(2) **失败 path 不丢弃**——而是触发 reflection，把"为什么这条 path 失败"写进 memory，影响后续 sibling node 的 prior；(3) **action 是 agent 级别**——一个 action 可能是"调用 search API"或"点击页面元素"，不是 reasoning step。这三点让 LATS 在 agent 任务上比朴素 MCTS 强，但实现复杂度也相应提高一档。LATS 详细展开在 **Module 14.4 Planning** 一章，本节只点到为止。

### 2.7 Beam search vs thought-level search

Beam search 是 **token-level** search：

- 每步保留 K 个 partial 序列（按累计 logprob 排序）
- 每个 partial 序列扩展 vocab_size 个 next token
- 总 cost 约 K × seq_len 次 LLM forward

它本质上是在 token grid 上搜，**不是在 thought grid 上搜**。区别：

- ToT / MCTS：每个 node 是一段 thought（百到千 token），节点数 10-1000，靠 LLM 自评 / PRM
- Beam search：每个 node 是一个 token，节点数 K × seq_len，靠 token logprob

LLM 时代 sampling（top-p / top-k）几乎完全替代了 beam search，原因有三：

1. **Beam search 偏短偏 generic**——logprob 累乘倾向选高频短句，diverse 差
2. **重复问题严重**——经典的 "the the the" 现象，需 length penalty / repetition penalty 缓解
3. **Beam K=4 收益远小于 sample K=4**——sample 利用了 model uncertainty，beam 没有

PRM 时代 beam search 短暂复活了一下：**PRM-guided beam search**（Lightman 2023）把 beam 的 logprob 改成 PRM 的 step score，效果接近 BoN。但仍然是 thought-level，不是 token-level。和 2.4 章节的 sampling-replace-beam 思路呼应。

注意区分三个层次的 search 粒度：

- **token-level**（beam search）：每个 node 是一个 token，节点数 K × seq_len，用 logprob 排序——已被 sampling 替代
- **step / thought-level**（ToT / MCTS / PRM-guided beam）：每个 node 是一段 thought（几十到几百 token），节点数 10-1000，用 PRM / verifier / 自评打分——本节核心
- **trajectory-level**（BoN / SC）：每个 node 是一整条 reasoning（千到万 token），节点数 N，用 RM / verifier / vote 打分——工业首选

粒度越细，搜索空间越大、cost 越高、对 evaluator 的精度要求也越高。step-level 是 sweet spot——既能利用前缀共享、又有可靠的 PRM 信号；token-level 太细评估难做、trajectory-level 太粗信息损失大。

### 2.8 推理时 vs 训练时的 trade-off

记一个三角：

```
                         total compute budget
                              /  |  \
                             /   |   \
                       train compute   inference compute
                       (scaling law)   (BoN/MCTS/SC)
                              \   |   /
                               internalize?
                                  │
                              R1 / o1 路线
                            (long-CoT 内化 search)
```

- **更多训练 compute** → scaling law (6.3)，最确定的提升路径，但 cost 是 one-time 大笔投入
- **更多推理 compute** → BoN / SC / MCTS，不动 weight 立刻能用，但每次 inference 都要付 N×
- **R1 / o1**：把推理 compute 内化到 model 自己的 long-CoT 里——一次 generation 长达 5k-50k token，等于在 token 流内部做了 implicit search

三者对应的"compute 转化为 quality"的效率不同。一个粗糙但实用的对比：

| 路径 | 边际 cost | 边际 quality | 何时上 |
|---|---|---|---|
| 训练 scaling | 高 (重训需百万美金级) | 稳定提升，跨任务通用 | 有 budget + 数据 + infra |
| 推理 BoN/SC | 中 (N× per query) | log 收敛，task 依赖 | 模型已 ship、临时提升单类任务 |
| Long-CoT 内化 (R1 RL) | 高一次 + 中 per query | 跨任务大幅提升 | 有 RL pipeline + verifier |

工业界 2024 - 2025 的共识在向"内化优先 + 必要时叠加 BoN"收敛——R1 / o1 类 model 跑 default 就拿到强 baseline，剩余的难题再用 N=8-16 的 BoN 兜底。

这三条路径**并不互斥**。现代实践常见组合：

- **R1-base + BoN**：R1 输出 + N 条 sample + verifier 选最优，AIME 上仍涨 2-5 点
- **R1-base + SC**：更便宜的 ensemble，对 numerical answer 任务首选
- **R1-base + MCTS**：cost 极高，主要在 math olympiad 这类 R1 单 sample 也只有 30-40% 通过率的任务上才值得

边界：R1 类 model 已经把"自我反思 / 自我验证 / backtracking"内化进 chain，**外部 search 的 marginal 增益缩水**。base model 时代 BoN N=64 涨 15 点，R1 时代 N=64 可能只涨 3 点。

更深一层的洞察：long-CoT 内化的是**单条 trajectory 上的 implicit search**——model 在生成时的"等等让我重新想想"、"假设 x 成立"、"换个思路"这些 token，本质上是 trajectory 内部的 backtracking 和 branching；但它**不能内化跨 trajectory 的 ensemble**——多 sample 的 majority vote / verifier-select 仍是单条 generation 做不到的。所以 R1 + Self-Consistency / R1 + BoN 在高难任务上仍 work，但 R1 + ToT 几乎没增益（因为 R1 的 long-CoT 已经覆盖了 ToT 的探索空间）。一个经验性 rule of thumb：**reasoning model 时代，flat ensemble 仍 work，tree search 让位给 model 内化**。

---

## 3. 最小代码示例

下面三段代码假设有 `llm(prompt, T, n)` 返回 `List[str]` 和（可选）`verifier(question, answer) → 0/1`、`reward_model(question, response) → float`。

### 3.1 Best-of-N + RM 选最优

```python
def best_of_n(question: str, reward_model, N: int = 16, T: float = 0.8) -> str:
    # 1. policy 独立 sample N 条候选
    prompt = f"Q: {question}\nA: Let's solve it step by step.\n"
    candidates = llm(prompt, T=T, n=N)             # T 不可为 0，否则 N 条全一样
    # 2. RM 对每条打分（也可换 PRM 用 min-aggregation，见 10.2 §3.3）
    scores = [reward_model(question, c) for c in candidates]
    # 3. 选最高分
    best_idx = max(range(N), key=lambda i: scores[i])
    return candidates[best_idx]
```

**关键**：N 取 16-32 是 sweet spot；reward_model 可换 verifier（math/code）或 PRM。换成 verifier 时 score 是 0/1，要 tie-break 用 logprob 或随机。

### 3.2 Self-Consistency for math

```python
import re
from collections import Counter

ANS_RE = re.compile(r"(?:answer is|=)\s*\\?boxed?\{?\s*(-?\d+(?:\.\d+)?)\s*\}?", re.IGNORECASE)

def extract_answer(text: str):
    """从 reasoning 末尾抽数字答案；找不到返回 None。"""
    matches = ANS_RE.findall(text)
    return matches[-1] if matches else None      # 取最后一次匹配（通常是 final answer）

def self_consistency_math(question: str, N: int = 20, T: float = 0.7) -> str:
    prompt = f"Q: {question}\nA: Let's think step by step.\n"
    samples = llm(prompt, T=T, n=N)               # T > 0.5 才有 diversity
    answers = [extract_answer(s) for s in samples]
    answers = [a for a in answers if a is not None]
    if not answers:
        return None
    return Counter(answers).most_common(1)[0][0]  # majority vote
```

**关键**：`T=0.7` 与 `N=20` 是 Wang 2022 经典配置；answer extraction 在 math 上靠 regex，open-ended 任务需 LLM 二次 normalize。

### 3.3 简化 MCTS for LLM reasoning

```python
import math, random
from dataclasses import dataclass, field

@dataclass
class Node:
    state: str                                    # 已生成的 partial reasoning
    parent: "Node" = None
    children: list = field(default_factory=list)
    visits: int = 0
    value_sum: float = 0.0
    is_terminal: bool = False

    def Q(self): return self.value_sum / max(self.visits, 1)
    def is_leaf(self): return not self.children

def ucb(child, parent, c_puct=1.41):
    if child.visits == 0: return float("inf")
    return child.Q() + c_puct * math.sqrt(math.log(parent.visits) / child.visits)

def select(node):
    while not node.is_leaf() and not node.is_terminal:
        node = max(node.children, key=lambda c: ucb(c, node))
    return node

def expand(node, K=4):
    if node.is_terminal: return node
    next_steps = llm(f"Continue ONE step:\n{node.state}\n", T=0.8, n=K)
    for step in next_steps:
        child_state = node.state + "\n" + step
        terminal = "answer is" in step.lower() or len(child_state) > 4000
        node.children.append(Node(state=child_state, parent=node, is_terminal=terminal))
    return random.choice(node.children)

def simulate(node, verifier, question):
    """从 node 滚到 terminal，verifier 给 0/1。"""
    rollout = llm(f"Finish the solution:\n{node.state}\nFinal answer:", T=0.7, n=1)[0]
    return verifier(question, rollout)            # PRM 也可在此处替换

def backup(node, reward):
    while node is not None:
        node.visits += 1
        node.value_sum += reward
        node = node.parent

def mcts_solve(question: str, verifier, n_iter: int = 50) -> str:
    root = Node(state=f"Q: {question}\nLet's solve it step by step.\n")
    for _ in range(n_iter):
        leaf = select(root)
        child = expand(leaf) if leaf.visits > 0 else leaf   # 第一次访问先 simulate 再 expand
        reward = simulate(child, verifier, question)
        backup(child, reward)
    # 选 root 下 visit 最多的 child（更鲁棒于 noisy Q）
    best = max(root.children, key=lambda c: c.visits)
    return best.state
```

**关键**：
- `c_puct=1.41` 是 AlphaZero 默认值，调参极敏感（§4）
- `simulate` 用 verifier 给 0/1 reward；用 PRM 时改成 PRM(node.state) 即可
- 最后选 `visits` 最多的而非 `Q` 最大的，是 AlphaZero 经验——更鲁棒于 imperfect value
- 实际工程要加 `max_depth` / token budget cap 防爆炸

---

## 4. 工程踩坑与经验

- ❗ **BoN 的 N 增大边际收益快速递减**（log 收敛）。N=8→16 涨 3-5 点，N=16→32 涨 1-3 点，N>64 通常不值得——除非 verifier 是 rule-based (math 字符串匹配 / code unit test)，那时可以拉到 256+。learned RM 上 N>64 反而可能掉点（更易选到 RM hacking sample）。
- ❗ **MCTS 的 c_puct 调参极敏感**，default $\sqrt{2} \approx 1.41$ 是经典值，但 LLM reasoning 上常需手调到 1.0-2.5。c_puct 太小 → exploit 主导，陷在初始好 child 出不来；太大 → 大量 wasted expansion。先在 small benchmark 上扫 5 个值。
- ❗ **Search 的 reward 必须 normalize**，否则 UCB 偏向高 reward 分支不可控。常见做法：把 PRM/verifier reward 缩到 [0,1]，或用 z-score normalize per-batch。GRPO 的 group-mean normalize 是一个好参考。
- ❗ **R1 出现后基座 reasoning 已强，BoN/MCTS 收益降低**。R1 类 long-CoT model 内化了部分 search-like reasoning，外部 BoN 在 AIME 上从 base model 时代涨 10-15 点缩到 2-5 点。如果你的 model 已是 R1-distill，先评测单 sample baseline，再决定是否上 search——不要无脑加。
- ❗ **Search cost 高时要有 budget cap**。MCTS 一个 question 50-500 LLM call 是常态，无 cap 容易爆 token / 超时。建议总 token < 100k 或总 LLM call < 500，超过就 early-terminate 取当前最佳。MCTS 还要加 max_depth（避免无限 rollout）和 max_nodes（避免节点爆炸）。
- ❗ **不可 verifiable 任务 (chat helpfulness / 写作) 上 BoN/MCTS 弱**——RM 本身的 bias 会被 search 系统性放大。RM 偏长答案 → BoN 选最长；RM 偏 markdown → BoN 选格式漂亮但内容空的。verifiable task (math/code) 上 search 强、unverifiable 上 search 弱，是一条硬规律。
- ❗ **Step 切分鲁棒性是 MCTS / PRM 的共同痛点**——什么算"一步"？模型实际生成的 trace 不一定有清晰边界。三个对策：(1) prompt 强制 `Step k:` 编号；(2) SFT 阶段就把 step 格式教进 model；(3) 用 `\n\n` / 句号兜底切分。最稳是 (2)。
- ❗ **Beam search 在 LLM reasoning 上几乎已死**。token-level beam K=4 收益远小于 sample K=4，且容易陷入"重复 / generic short answer"。新工程别再用 beam search 当 baseline，用 nucleus sampling + BoN 即可。PRM-guided beam search 是个例外（thought-level beam），但已被 MCTS 覆盖。
- ❗ **MCTS 的 simulate 步骤是 cost 大头**。每个 leaf 都要 rollout 到 terminal，单次 rollout 几千 token。两种节省：(a) 用 PRM 直接给 leaf 估 value，跳过 rollout；(b) 限制 rollout 长度（max 200 token）+ early-stop。AlphaProof / rStar 都用 (a)。
- ❗ **N 大时部署需配合 PagedAttention / RadixAttention**。BoN N=64 / MCTS 100 节点意味着同一 prompt 下大量并行 generation，朴素实现会因为 KV cache 重复计算导致 cost 翻倍。vLLM 的 PagedAttention（11.2）和 SGLang 的 RadixAttention（11.3）通过共享前缀 KV cache 把 BoN/SC 的实际 cost 降到接近 1× single-sample——不上这层基础设施，inference search 的账完全不划算。
- ❗ **"voting + selection" 双重融合往往比单方法更稳**。PRM-BoN 选 top-K 后再对 extracted answer 做 majority vote (BoN ∩ SC)，AIME 上能比单纯 BoN top-1 再涨 1-3 点。本质是用 SC 的低 variance 来缓冲 PRM 的 hacking——简单技巧但工程产出比极高。

---

## 5. 经典 paper

- **Cobbe et al., 2021 — Training Verifiers to Solve Math Word Problems** — GSM8K 数据集 + BoN-with-verifier 的奠基论文，**BoN 的 log scaling law 实证起源**。读 §5 的 verifier-guided BoN 曲线，是后续所有 inference-time scaling 的 baseline。
- **Wang et al., 2022 — Self-Consistency Improves Chain-of-Thought Reasoning** — 用 majority vote 替代 RM 做 BoN 的极简路线，**至今仍是最 cost-effective 的 inference scaling**（无 RM 训练成本）。读 §3 的 vote 算法 + §5 的 N 曲线。
- **Yao et al., 2023 — Tree of Thoughts: Deliberate Problem Solving with LLMs** — 把 reasoning 显式建成 tree + 自评 + 搜索，启发了后续 MCTS-LLM 系列。本节 §2.5 ToT vs MCTS 对比直接基于此。
- **Zhou et al., 2023 — Language Agent Tree Search (LATS)** — MCTS + Reflexion + value function 在 agent 上的统一框架。本节只点到，详细展开在 Module 14.4 Planning。
- **Qi et al., 2024 — rStar: Mutual Reasoning Makes Smaller LLMs Stronger Problem-Solvers** — Microsoft 2024 把 MCTS + multi-action（subgoal / rephrase / restate）用在 7B model 上，达到接近 GPT-4 Turbo 的 math 表现。读 §3 的 action space 设计是 MCTS-LLM 工程化代表。
- 加分阅读：**DeepMind 2024 — AlphaProof / AlphaGeometry 2** (IMO 2024 银牌，MCTS + LLM + 形式化 verifier)、**OpenAI 2024 — o1 system card** (推测内化了 search-like reasoning，但官方未明说算法)、**Zhang et al., 2024 — MCTSr: Math + MCTS + self-refine**。

---

## 6. 自测与面试题

**Q1（统一视角）**：用 N 维 trade-off 串起 Greedy / BoN / Self-Consistency / ToT / MCTS 5 种方法，给出每种方法的 (candidate generation / leaf evaluation / aggregation / cost / 适用任务) 对比表。

<details>
<summary>Answer sketch</summary>

| 方法 | candidate generation | leaf evaluation $V$ | aggregation $\mathcal{A}$ | cost | 适用任务 |
|---|---|---|---|---|---|
| **Greedy** | 1 条 (argmax) | 无 | id | 1× | baseline，所有任务 |
| **Best-of-N** | i.i.d. sample N 条 (T>0) | RM / PRM / verifier | $\arg\max_i V(y_i)$ | N× | 通用，math/code 最强（有 verifier） |
| **Self-Consistency** | i.i.d. sample N 条 (T>0) | 无 (用答案 cluster) | majority vote | N× | discrete answer (multi-choice / 数学数字) |
| **ToT** | tree expand (BFS/DFS), 每节点 K 个 thought | LLM self-rate | tree top-1 | $K^d$ × ≈ 10-100× | short reasoning (Game-of-24, planning) |
| **MCTS** | UCB-guided expand + rollout | PRM / verifier | visit-weighted root child | 50-500× | math / code / formal proof |

关键观察：
- N=1 → N>1 是第一个跃迁（greedy → flat sample）
- flat → tree 是第二个跃迁（独立 sample → 共享前缀的搜索）
- $V$ 的"信号强度"决定 search 上限：vote (弱) < learned RM (中) < PRM (中强) < verifier (强)
- cost 的 1, N, $K^d$ 三个量级要心里有数

加分点：指出**没有 leaf evaluator，search 退化为随机采样**；以及 **learned $V$（RM/PRM）的 hacking 风险随 N 增大放大**——这是 BoN log 饱和的工程根因。

</details>

**Q2（实战）**：你正在做一个 math reasoning model，candidate 方案有 (a) base model + BoN N=32，(b) base model + MCTS（每题 200 LLM call），(c) DeepSeek-R1-distill + BoN N=8。三种方案的预期成本与质量怎么排？怎么选？

<details>
<summary>Answer sketch</summary>

成本（按单题 LLM token / call 量级）：

| 方案 | LLM 调用次数 | token 量级 | RM 依赖 |
|---|---|---|---|
| (a) base + BoN N=32 | 32 sample + 32 RM forward | ~32 × 500 token | 必须有 RM 或 verifier |
| (b) base + MCTS 200 call | 200 LLM call (含 expand + rollout) | ~200 × 300 token | 必须有 PRM / verifier |
| (c) R1-distill + BoN N=8 | 8 sample + 8 RM forward | ~8 × 5000 token (R1 long-CoT) | 同 (a) |

质量（GSM8K / MATH 上的经验）：

- (a) base + BoN N=32：base GSM8K acc 50% → 60-65%（涨 10-15 点）
- (b) base + MCTS 200 call：50% → 65-70%（涨 15-20 点，MCTS > BoN 一档）
- (c) R1-distill + BoN N=8：R1-distill 单 sample 已 80%+，BoN 涨 2-4 点 → 82-85%

**怎么选**：

- 如果可以用 R1-distill：**首选 (c)**——绝对 accuracy 最高，单 token cost 虽高但调用次数少
- 如果只能用 base model 且 verifier 鲁棒（math 字符串匹配）：**(b) MCTS** 上限更高，但 200 call 不一定能负担
- 工程权衡：**(a) BoN 是第一选择**——实现 30 行、调参少、cost 可控；MCTS 只在 (a) 不够 + 单题预算足够时才上
- 加分点：(c) 还可以叠 (a) → R1-distill + BoN，AIME 上仍能再涨 2-5 点

</details>

**Q3（前沿）**：DeepSeek-R1 / OpenAI o1 已经把 search-like reasoning 内化到 long-CoT 里，外部 search（BoN / MCTS）是否还必要？哪些场景下仍 useful？

<details>
<summary>Answer sketch</summary>

仍必要，但 margin 缩水。需要分情形：

**外部 search 仍有效的场景**：

1. **极难 task 单 sample 不够**：AIME / IMO / Frontier-Math 等 reasoning model 单 sample 通过率 30-50% 的任务，BoN N=8-16 仍能涨 5-10 点，verifier 信号清晰
2. **降本场景**：用小 reasoning model + BoN，可能比大 reasoning model 单 sample 便宜——模型 size 与 N 的 trade-off 要算账
3. **verifier 强但 model 弱**：code 生成 + unit test verifier 时，BoN 几乎"免费"涨点（unit test 全自动）
4. **agent / multi-step 任务**：长 trajectory 上的 backtracking / re-plan，long-CoT 内化效果有限，仍需 LATS-style 外部 search（详 14.4）
5. **降低 variance**：reasoning model 一次 generation 的 variance 仍很大，SC / BoN 是简单可靠的方差减少手段

**外部 search 退化的场景**：

1. **easy task** (GSM8K)：R1-distill 单 sample 已 90%+，BoN 涨 < 1 点，cost 不值
2. **non-verifiable task**（聊天 / 写作）：RM hacking 严重，BoN 反而降低质量
3. **预算极紧**：reasoning model 本身 long-CoT 已 5k-50k token，再 ×N 直接爆 budget

**趋势观察**：

- "**reasoning model 内化的 search**" + "**外部显式 search**" 是互补关系，不是替代关系
- 计算预算分配从 "all in train" → "train + test-time" → "train + (long-CoT internalized + external search)" 三层叠加
- 学术上 PRM 也从 "RL 训练 reward" 降级为 "inference-time search heuristic"——绕开 hacking 风险，保留 dense 信号收益（10.2 §2.5 趋势）

加分点：提到 OpenAI o3 / DeepSeek-V3-R1 等可能在内部已经做了 RL + search hybrid，外部 search API 看起来"消失"是因为被打包进 model behavior，**算力账依然存在**——只是 buyer 从 dev 变成了 model provider。

</details>

---

## 7. 延伸阅读

- [Cobbe et al. 2021 — Training Verifiers (arXiv 2110.14168)](https://arxiv.org/abs/2110.14168) — BoN 的 log scaling law 实证起源，所有 inference-time scaling 工作的 baseline
- [Wang et al. 2022 — Self-Consistency (arXiv 2203.11171)](https://arxiv.org/abs/2203.11171) — 工程上零成本的 BoN 替代品
- [Yao et al. 2023 — Tree of Thoughts (arXiv 2305.10601)](https://arxiv.org/abs/2305.10601) + [官方 repo](https://github.com/princeton-nlp/tree-of-thought-llm) — ToT 原 paper + Game-of-24 实现，跑一遍能感知 search 复杂度
- [Zhou et al. 2023 — LATS (arXiv 2310.04406)](https://arxiv.org/abs/2310.04406) — MCTS + reflection + value 的 agent 统一框架，14.4 详谈
- [Qi et al. 2024 — rStar (arXiv 2408.06195)](https://arxiv.org/abs/2408.06195) — Microsoft MCTS-LLM 在 small model 上的代表工作，工程细节丰富
- [DeepMind 2024 — AlphaProof / AlphaGeometry 2 blog](https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/) — MCTS + LLM + 形式化 verifier 在 IMO 上达到银牌
- [OpenAI o1 system card](https://openai.com/index/openai-o1-system-card/) — 推测内化了 search-like reasoning，官方未明说算法但 hint 处处可见
- 推荐继续读：**Module 11** 推理引擎（BoN/MCTS 的高 N 部署需要 11.2 PagedAttention / 11.3 RadixAttention 才能 cost-effective）；**Module 14.4 Planning**（LATS 在 agent 上的完整展开）；**Module 15.4 Reasoning + Agent**（Search-R1 / ReSearch / ReTool 把 search 训进 model 的前沿路线）
