---
title: "14.4 Planning：ToT / LATS / Plan-and-Solve / Code-as-Policy"
description: "把 agent 决定 \"下一步做什么\" 的几大流派——Plan-and-Solve（显式 plan）/ LLM+P（形式化 PDDL）/ ToT 与 LATS（树搜索）/ Code-as-Policy（plan as code）/ Hierarchical（多层 planner）——放进同一张坐标系里讲清楚：planning 与 reasoning 的边界、何时该上 search、为什么 reas"
---

> ⏱ 预计阅读 35 分钟 ｜ 难度 ★★★ ｜ 前置：10.1 CoT/SC/ToT、10.4 推理时搜索；14.1 Agent 范式、14.2 ReAct

## 一句话本节讲什么

把 agent 决定 "下一步做什么" 的几大流派——**Plan-and-Solve（显式 plan）/ LLM+P（形式化 PDDL）/ ToT 与 LATS（树搜索）/ Code-as-Policy（plan as code）/ Hierarchical（多层 planner）**——放进同一张坐标系里讲清楚：planning 与 reasoning 的边界、何时该上 search、为什么 reasoning agent 时代 LATS 仍是 hard task 的 SOTA。

---

## 1. Mental model（直觉）

### 1.1 Planning 不等于 Reasoning

10.1 / 10.4 讲的 reasoning 全部发生在 **token 空间内部**：模型一次 forward 生成 long-CoT、跑 self-consistency、做 BoN，最终输出一个 answer。**没有外部世界、没有 tool、没有 observation**。

Agent planning 不一样——它要决定的是 **action 序列**，每个 action 可能：

- 调用 tool（搜索、代码执行、写文件）
- 等待 environment 返回 observation（搜索结果、代码报错、浏览器截图）
- 根据 observation 决定下一个 action

```
   reasoning (10.x):
       ┌───────────────────────────────────────┐
       │  one forward                           │
       │  long-CoT: think... think... answer    │  ← 完全 in-token
       └───────────────────────────────────────┘

   agent planning (14.4):
       plan ──► action_1 ──► obs_1 ──► action_2 ──► obs_2 ──► ... ──► done
                  │            ▲          │
                  └──tool──────┘          └──可能 replan──
                  (外部世界)                (observation 改变 plan)
```

一句话：**reasoning 是脑内思考，planning 是 action sequencing**。reasoning 的 unit 是 thought / step；planning 的 unit 是 action / sub-task。两者可以叠加（agent 内部用 reasoning 决定 next action），但分析时要分开看。

### 1.2 三个层次的 planning

按 plan 表达粒度从粗到细，agent planning 有三层：

```
strategic   ┌────────────────────────────────────────┐
 (高层)     │ "解决用户问题需要：检索资料 → 生成总结" │  ← 任务分解、sub-goal
            └────────────────────────────────────────┘
                              │
tactical    ┌────────────────────────────────────────┐
 (中层)     │ "检索用 search tool，总结用 LLM"        │  ← 选 tool / 选 sub-agent
            └────────────────────────────────────────┘
                              │
operational ┌────────────────────────────────────────┐
 (底层)     │ search("LATS paper 2023")              │  ← 具体 action
            └────────────────────────────────────────┘
```

简单 agent（ReAct）三层混在一起塞进一次 LLM call。复杂 agent（HuggingGPT / MetaGPT）显式分层：先 strategic plan、再 tactical plan、最后 operational execution。Plan-and-Solve 大致对应"显式 strategic plan + 隐式 tactical/operational"。

### 1.3 search vs internalization：reasoning agent 时代的边界

10.4 §2.8 留了一句：**reasoning model 内化了 trajectory 内部的 implicit search，但跨 trajectory 的 ensemble 仍需外部 search**。这条线在 agent planning 上同样成立：

- **简单 agent task**（单 tool 调用 / 浅层 reasoning）：reasoning model + ReAct 一次 long-CoT 解决，外层 LATS 多余
- **难 agent task**（SWE-agent 跨文件改动 / 长 web 任务 / 多 sub-task 协作）：单 trajectory 一旦走错就需要回退，**LATS-style 树搜索仍是 SOTA**

把这两条心智先记住——它会贯穿后面 7 种方法的取舍。

---

## 2. 公式与原理：6 种 planning 流派

下面把"agent planning"这个广义 umbrella 拆成 6 种代表方法，每种讲清 **plan 形式 / 是否 search / 适用任务**。

### 2.1 Plan-and-Solve（Wang 2023）

最朴素的 planning baseline：把"先想清楚、再动手"用 prompt 显式化。

**两段 prompt**：

1. **Plan stage**：`"Let's first understand the problem and devise a plan to solve it. Then, let's carry out the plan."`
2. **Solve stage**：在同一 LLM call 里、plan 之后接着写 step-by-step 执行

实证：相对 vanilla CoT 在 GSM8K / SVAMP / AQuA 等数学任务上涨 5-10%。**它的价值不在算法，而在于建立了"plan 与 execution 应该 explicit 分离"的 mental discipline**——哪怕只是 prompt-level 的分离。

适用：通用 baseline，几行 prompt 改造，**任何 agent 都该先试 plan-and-solve 再决定要不要上 search**。

### 2.2 LLM+P（Liu 2023）：把 plan 翻成 PDDL

经典 AI planning 的语言是 **PDDL（Planning Domain Definition Language）**——一种把世界状态、action 前置/后置条件、目标都形式化的 DSL，配合 fast-downward 等求解器可以**保证最优解**。

LLM+P 的思路：

```
natural language task ──► [LLM as translator] ──► PDDL problem file
                                                        │
                                                        ▼
                                          ┌──────────────────────┐
                                          │  classical planner   │  ← 完备 + 最优
                                          │  (FastDownward)       │
                                          └──────────────────────┘
                                                        │
                                                        ▼
                                              optimal action plan
```

**优势**：

- 经典 planner 在可形式化的 domain 上 **完备 + 最优**（LLM 自己 reasoning 做不到）
- 速度快、可解释（每个 action 有 precondition）

**致命限制**：

- 只适合**可形式化**的 task：blocks-world / 机器人导航 / 简单 puzzle
- 现实 web task / SWE task / 对话 task **写不出 PDDL**（state space 是无限自然语言）
- LLM → PDDL 翻译本身有 hallucination 风险，写错一个 predicate 全盘失败

工业落地少。但在 robotics / formal planning benchmark 上仍是强 baseline，提示我们：**LLM 不擅长 search，符号系统擅长——能形式化的就别让 LLM 自己 reason**。

### 2.3 Tree of Thoughts (ToT) for agent

ToT 已在 10.1 / 10.4 详讲。在 agent context 下复用时几个关键差异：

- **node 是 partial plan**（"已经 decide 的 action 序列前缀"），不是单纯的 reasoning step
- **expand**：让 LLM 在当前 partial plan 下提出 K 种 next action 候选
- **evaluate**：LLM 自评每个 partial plan 的 promise（也可以查 tool / 看 observation）
- **prune**：BFS / DFS 保留 top-beam 条 plan path

**适用范围**：short reasoning + small action space（Game-of-24、24-step 短规划）。**不适用**：长 trajectory web agent（state space 爆炸、自评 noisy）。

ToT 在 agent 上是个**过渡范式**——简洁、不需要 verifier，但搜索深度受限。深 search 要上 LATS。

### 2.4 LATS（Zhou 2023）：MCTS + reflection + value function

**Language Agent Tree Search** 是 agent 树搜索的代表。把 10.4 §2.6 的简介展开：

四阶段循环（沿用 MCTS 抽象，但 node / value 改成 agent 语义）：

```
Root: agent state s_0 (initial observation)

  ┌─────────── 1. Select ─────────────────┐
  │ 从 root 沿 UCB 走到 leaf:              │
  │   a* = argmax_a [Q(s,a) + c·U(s,a)]    │
  │ U(s,a) = √(ln N(s) / N(s,a))           │
  └────────────────────────────────────────┘
                     │
  ┌─────────── 2. Expand ─────────────────┐
  │ LLM propose K 个 next action 候选       │
  │ 每个 child = (action, expected_obs)     │
  └────────────────────────────────────────┘
                     │
  ┌─────────── 3. Simulate ───────────────┐
  │ 实际执行 action → 拿真 observation      │
  │ 沿当前 plan rollout 到 terminal:        │
  │   reward = env_reward + λ·LLM_value     │
  └────────────────────────────────────────┘
                     │
  ┌─────────── 4. Backprop + Reflect ─────┐
  │ 沿 path 更新 Q, N                       │
  │ 若 reward 低: trigger reflection        │
  │   → "为什么这条 path 失败" 写入 memory   │
  │   → 影响后续 sibling 的 prior           │
  └────────────────────────────────────────┘
```

LATS 与朴素 MCTS for LLM 的三处关键差别：

1. **value 来源混合**：纯 env reward 太稀疏（很多 web task 只有 final success/fail），LLM self-value 又太 noisy。LATS 用 **加权融合**：$V(s) = \alpha \cdot V_{\text{env}}(s) + (1-\alpha) \cdot V_{\text{LLM}}(s)$，浅层用 LLM value 快速剪枝、深层 rollout 拿 env reward。
2. **失败 path 触发 reflection**：参考 10.1 Reflexion，失败 trajectory 不丢弃，让 LLM 写一句"lesson"存进 memory，下次同类 sibling 被 select 时把 lesson 注入 prompt 提高/降低 prior。
3. **action 是 agent 级别**：一个 action 是"调用 search API"或"点击按钮"，不是 reasoning step。这意味着 simulate 阶段每个 rollout 都要真的跑环境，**cost 比纯 reasoning 的 MCTS 还高一档**。

**实证（Zhou 2023）**：

| Benchmark | baseline (ReAct) | LATS | 提升 |
|---|---|---|---|
| HotpotQA (QA) | 32% | 47% | +15 |
| WebShop | 28% | 38% | +10 |
| HumanEval (code) | 67% | 92% | +25 |

代价是 **10-50× LLM call**（MCTS iter × expand K × rollout）。

LATS 的核心定位：**当单 trajectory 不可靠（agent 容易跑偏、tool 可能失败、需要 backtrack）时，LATS 用 search + reflection 把 ReAct 的 robustness 拉到生产可用线**。

### 2.5 Code-as-Policy（Liang 2022）

把 plan 直接表达成 **可执行 Python 代码**：

```
task: "把所有蓝色方块叠到红色方块上"
        │
        ▼
LLM 生成:
    blue_blocks = detect_blocks(color="blue")
    red_block = detect_blocks(color="red")[0]
    for b in blue_blocks:
        pick_and_place(b, red_block)
        red_block = b  # 更新栈顶
        │
        ▼
    exec() ──► 机器人 / 模拟器执行
```

为什么代码比 free-form text plan 强？

- **天然支持控制流**：if / for / while / try-except 直接表达条件分支与循环，free-form text plan 做不到
- **可调用 function**：把 perception / action 包装成 API，LLM 写 plan 等于在调用一个 robot DSL
- **可验证**：syntax error、type check、unit test 给 strong feedback signal
- **可组合**：sub-routine 可被复用，形成可累积的 skill 库（Voyager 在 Minecraft 中正是用这一点做 lifelong learning）

**著名后续**：

- **Voyager**（Wang 2023）：Minecraft agent，把学到的 Python skill 存进 skill library，跨任务复用
- **OpenAI Code Interpreter / Advanced Data Analysis**：本质上是 Code-as-Policy 在 data analysis domain 的产品化
- **CodeAct / CodeAgent**：把 ReAct 的 "Action: tool_name(args)" 直接换成 "Action: ```python\n...\n```"，已是现代 agent framework 的主流 action 表达

风险：`exec(untrusted_code)` 是经典安全洞，必须 sandbox（Docker / restricted python / E2B / Modal）。**生产环境 Code-as-Policy 必须有 sandbox**，§4 会展开。

### 2.6 HuggingGPT 风格 Planner-Executor

复杂任务（research / software dev）一个 LLM 一次 generation 解决不了。HuggingGPT（Shen 2023）把 LLM 当 **high-level planner**：

```
user task: "把这张照片里的人换个表情，再用语音解说"
        │
        ▼
┌─────────────────────────────────────────┐
│  Planner LLM (e.g. GPT-4)               │
│  分解出 task graph:                      │
│    1. face detection (HF model A)        │
│    2. expression edit (HF model B)       │
│    3. caption generation (HF model C)    │
│    4. text-to-speech (HF model D)        │
└─────────────────────────────────────────┘
        │
        ▼
   逐个调用 sub-model / sub-agent，把 outputs 组合
```

一般化模式：**planner 输出一个 task graph (DAG)**，每个 node 是一个 sub-task + 该 sub-task 用什么 tool/model 执行；executor 按拓扑序调度。

适用：**多 model / 多 tool 调度** 的场景；**复杂多步任务**（写完整论文 / 端到端 software 项目）。

局限：planner 一次性出 plan，**中途不 replan**；如果 sub-task 失败就崩。所以现代 agent 多用"planner 输出 plan → executor 执行 → 失败时 trigger 局部 replan"的混合形态。

### 2.7 Hierarchical Planning（MetaGPT 风格）

**多 agent + 角色化 planning**：

```
   Product Manager (LLM, 高层 strategic plan)
            │  output: PRD doc
            ▼
   Architect      (LLM, 中层 tactical: 模块分解 / API 设计)
            │  output: tech spec
            ▼
   Engineer × N   (LLM, operational: 写代码)
            │  output: code
            ▼
   QA Engineer    (LLM, verification)
```

每层 agent 的输出是下层的输入，**plan 在层间传递时被逐步细化**。优势：每个 agent 的 prompt 更聚焦、context 更短；劣势：层间 communication overhead 大、容易过度工程。

适用：**复杂 software 任务、跨 role 协作**（MetaGPT 的"小型软件公司"是教科书 demo）。
不适用：**简单任务**——单 agent 足够时硬上 hierarchical 是过度设计。

Multi-agent 框架的完整对比留到 14.6。

### 2.8 Reasoning Agent：plan 与 reasoning 边界模糊

2025 趋势：**Search-R1 / ReSearch / ReTool / Agent-R1** 这类 R1-derived agent 把 plan + tool call + reflection 全部塞进**一次 long-CoT**：

```
<think>
要回答这个问题，我先搜索 X...
[调用 search] → 拿到 obs_1
看起来还需要进一步查 Y...
[调用 search] → 拿到 obs_2
综合两次结果，答案是 Z
</think>

<answer>Z</answer>
```

整个 plan-tool-reasoning loop 内化进 model 自己的 generation。结果：

- **简单 / 中等任务上彻底覆盖** Plan-and-Solve / ReAct 的位置
- 但**复杂 search 任务（深度 backtrack / 大状态空间）仍打不过外部 LATS**——因为单 trajectory 还是单 trajectory，ensemble 与树搜索做不了

15.4 会展开 reasoning agent 的训练。本节只点出：**reasoning agent 的兴起让 "plan 与 reasoning 的边界" 越来越模糊**——但**外部 search 在 hard task 上仍是 SOTA**。

### 2.9 一张方法对比表（必看）

| 方法 | plan 形式 | 是否 search | 是否需要 verifier | cost | 适用场景 |
|---|---|---|---|---|---|
| **Plan-and-Solve** | text plan | 无 | 无 | 1× | 通用 baseline，先试这个 |
| **LLM+P** | PDDL 形式化 | 经典 planner | 无（planner 自带） | 中 | 可形式化 task（机器人 / blocks-world） |
| **ToT** | text tree | 有（BFS/DFS + 自评） | 弱（LLM self-rate） | 10-100× | short reasoning、small action space |
| **LATS** | text tree + value | 有（MCTS + reflect） | 强（env reward） | 50-500× | hard agent task（web / SWE-agent） |
| **Code-as-Policy** | Python code | 无（exec 即执行） | 强（unit test / runtime） | 1× | 机器人 / data analysis / 可代码化任务 |
| **HuggingGPT** | task graph (DAG) | 无 | 弱 | 中 | 多 model / 多 tool 调度 |
| **Hierarchical** | 多层 plan | 可选 | 中（每层可独立 verify） | 高（多 agent） | 复杂 software / 跨 role 协作 |
| **Reasoning Agent** | long-CoT 内嵌 | 内化 | 训练时需要 | 中（long generation） | 现代 SOTA，简单/中等 task 首选 |

记一句话：**Plan-and-Solve 是 baseline、LATS 是 hard task 上限、Code-as-Policy 是工程最爱、Reasoning Agent 是当下默认起点**。

---

## 3. 最小代码示例

下面三段代码都假设有 `llm(prompt, T=0.0, n=1) -> List[str]` 抽象。

### 3.1 Plan-and-Solve prompt 模板

```python
PLAN_AND_SOLVE_PROMPT = """Q: {question}

A: Let's first understand the problem, extract relevant variables and their corresponding numerals,
and devise a complete plan. Then, let's carry out the plan, calculate intermediate variables
(pay attention to correct numerical calculation and commonsense), solve the problem step by step,
and show the answer.

Plan:
"""

def plan_and_solve(question: str) -> str:
    out = llm(PLAN_AND_SOLVE_PROMPT.format(question=question), T=0.0, n=1)[0]
    # 模型会先输出 "Plan: 1. ... 2. ..." 然后接 "Solution: ..."
    return out
```

**关键**：原 paper（Wang 2023）的 PS+ prompt 不止"先 plan 再 solve"，还显式要求 **extract variables / pay attention to commonsense**——这两句对数学题鲁棒性贡献巨大，**别把 prompt 简化掉**。GSM8K 上 PS+ 比 vanilla CoT 涨 5-10 点几乎全靠这两句的 anchoring 效果。

### 3.2 简化 LATS 骨架（MCTS for agent action）

```python
import math, random
from dataclasses import dataclass, field

@dataclass
class AgentNode:
    state: str                       # history: 已执行 actions + observations
    action: str = ""                 # 到达本 node 的 action（root 为空）
    parent: "AgentNode" = None
    children: list = field(default_factory=list)
    visits: int = 0
    value_sum: float = 0.0
    reflections: list = field(default_factory=list)  # 失败 lesson
    is_terminal: bool = False

    def Q(self): return self.value_sum / max(self.visits, 1)

def ucb(child, parent, c=1.41):
    if child.visits == 0: return float("inf")
    return child.Q() + c * math.sqrt(math.log(parent.visits) / child.visits)

def select(node):
    while node.children and not node.is_terminal:
        node = max(node.children, key=lambda c: ucb(c, node))
    return node

def expand(node, env, K=3):
    """LLM propose K 个 next action 候选；这里只 stub 一个 ReAct 风格 prompt。"""
    hints = "\n".join(f"- {r}" for r in collect_reflections(node))   # 把祖先 reflection 注入 prior
    prompt = f"History:\n{node.state}\nLessons:\n{hints}\nPropose ONE next action:"
    candidates = llm(prompt, T=0.8, n=K)
    for act in candidates:
        obs = env.step(act)                                          # 实际执行 action 拿 observation
        new_state = node.state + f"\nAction: {act}\nObservation: {obs}"
        terminal = env.done() or "Final Answer" in act
        node.children.append(AgentNode(state=new_state, action=act, parent=node, is_terminal=terminal))
    return random.choice(node.children)

def simulate(node, env):
    """rollout 到 terminal；reward 混合 env signal + LLM self-value。"""
    if node.is_terminal:
        return env.reward()
    rollout = llm(f"Continue to finish:\n{node.state}", T=0.7, n=1)[0]
    env_r = env.reward()                                             # 0/1 任务成功
    llm_v = float(llm(f"Score this trajectory 0-1:\n{node.state}\n{rollout}", T=0.0)[0])
    return 0.5 * env_r + 0.5 * llm_v

def backprop(node, reward, threshold=0.3):
    while node is not None:
        node.visits += 1
        node.value_sum += reward
        if reward < threshold:                                        # 失败 → 触发 reflection
            lesson = llm(f"This path failed:\n{node.state}\nWhat lesson?", T=0.7)[0]
            node.reflections.append(lesson.strip())
        node = node.parent

def lats(question, env, n_iter=30):
    root = AgentNode(state=f"Goal: {question}")
    for _ in range(n_iter):
        leaf = select(root)
        child = expand(leaf, env) if leaf.visits > 0 else leaf
        reward = simulate(child, env)
        backprop(child, reward)
    return max(root.children, key=lambda c: c.visits).action          # AlphaZero 经验: visit > Q
```

**关键**：
- `expand` 要 **真的执行 action 拿 observation**——这是 LATS 与纯 reasoning MCTS 的核心差别（cost 大头）
- `backprop` 在低 reward 时触发 reflection，把 lesson 写入 node；后续 `select` 走到此 node 时 lesson 通过 `expand` 注入 prompt 影响 prior（参考 Reflexion）
- 最后选 `visits` 最多的 child（AlphaZero 经验，鲁棒于 noisy Q）
- 实际工程要加 budget cap（max LLM call / max depth），否则 web agent 一题烧几百刀

### 3.3 Code-as-Policy 示例

```python
SYSTEM = """You are a robot controller. Output ONLY Python code that uses these primitives:
- detect(color: str) -> List[Block]
- pick_and_place(src: Block, dst: Block) -> None
- get_top(stack: Block) -> Block
"""

USER = "Stack all blue blocks on top of the red block."

def code_as_policy(task: str, env):
    code = llm(f"{SYSTEM}\n\nTask: {task}\nCode:", T=0.0, n=1)[0]
    code = strip_markdown_fence(code)                # 去掉 ```python ... ```
    # CRITICAL: 必须 sandbox，下面只是教学示意
    safe_globals = {
        "detect": env.detect,
        "pick_and_place": env.pick_and_place,
        "get_top": env.get_top,
    }
    try:
        exec(code, safe_globals, {})                 # 生产环境替换为 docker / E2B / restricted python
    except Exception as e:
        # 把 error 喂回 LLM 触发 self-repair（一次试错通常够）
        retry = llm(f"{SYSTEM}\nTask: {task}\nFailed code:\n{code}\nError: {e}\nFix:", T=0.0)[0]
        exec(strip_markdown_fence(retry), safe_globals, {})
```

**关键**：
- `safe_globals` 把可调 primitive 显式 whitelist，比 `exec(code)` 暴露整个 Python 安全得多——但远远不够，生产必须 sandbox（**§4 详谈**）
- error → 喂回 LLM → retry 是 Code-as-Policy 必带 pattern，单次成功率往往 60-70%，retry 一次能拉到 85%+
- `T=0.0` 因为代码执行不容许 syntax random；要 diversity 时改成 BoN（采 N 条 code 跑 unit test 选 pass 的）

---

## 4. 工程踩坑与经验

- ❗ **Plan 粒度的甜点**：plan 太详细（每步精确到 tool 参数）→ token 浪费 + execution 阶段 LLM 不灵活，一个 obs 异常就崩；plan 太粗（"先搜索再总结"）→ execution 容易跑偏。经验上 strategic level（3-7 步、每步一句话）最稳，**operational 级别留给 execution LLM 现场决定**。Plan-and-Solve 原 paper 也有意只让 plan 停在 outline 级别。
- ❗ **LATS / MCTS 在 agent 任务上 cost 极高**——每 node 一次 LLM call + 真 env step（web agent 单 step 几秒），一个 root 50-500 LLM call 是常态。**只在 hard task 上才值得**：单 sample 通过率 <40% 且有清晰 verifier 的任务（SWE-Bench / HumanEval+ / WebShop hard split）。简单任务 LATS 的边际收益 < cost，老老实实用 reasoning agent + ReAct。
- ❗ **Code-as-Policy 的 `exec` 安全风险被严重低估**。LLM 完全可能输出 `import os; os.system("rm -rf /")` 或者外联恶意 URL。生产环境必须用 **sandbox**：Docker container（资源限制 + network policy）/ E2B / Modal sandbox / WebContainer / restricted python（移除 `__builtins__` 危险项）。**永远不要在生产宿主机直接 exec untrusted code**——这条比 prompt injection 更容易被忽视、危害更大。
- ❗ **Plan 不要 hard-code 太多 if-else**——见过工程师把 planner 输出强行 schema 化（"必须输出 JSON 结构 {step: ..., tool: ..., params: ...}"），结果 LLM 灵活度被掐死、面对意外 observation 没法 replan。Schema 是好的，但 schema 内要留 **`thoughts: str`** 这类自由表达字段让 LLM 解释自己的决策，**别只留参数槽**。
- ❗ **Reasoning agent 出现后，简单 task 不需要外层 LATS**——R1-distill / Qwen3-Reasoner / DeepSeek-R1 单 sample 在 GSM8K / 简单 web task 上已经 90%+，外层套 LATS 涨不到 1 点反而 cost ×50。**先评测 reasoning model + ReAct baseline**，再决定要不要上 search。这条 rule of thumb 在 2025 之后特别重要。
- ❗ **Hierarchical Planning 容易过度工程**——MetaGPT 的"小公司"demo 很炫，但 5 个 agent 串起来 token 翻 10 倍、串行延迟翻 5 倍、还容易在 agent 间 communication 上丢信息（"PM 写的 PRD 工程师误读"）。**simple task 用 single agent 就行**，hierarchical 留给真正多 role 协作的复杂场景（写一个完整 web 项目、写一篇 research paper）。
- ❗ **LATS 的 reflection 不是免费午餐**——每次失败都生成一句 lesson 写入 memory，次数多了 prompt 会被 lessons 撑爆，且 lessons 之间可能互相矛盾。需要 dedup（embedding 相似度去重）+ retention policy（只保留 top-K 最 frequent / 最 informative）。生产 LATS 的 reflection 池一般限 20-50 条上限。
- ❗ **PDDL（LLM+P 路线）写错一个 predicate 全盘失败**——不像 LLM 自由文本能模糊容错，formal planner 对 input 极挑剔。LLM 把 task 翻成 PDDL 时，常见 bug 是漏 precondition / 写错 type / 弄混 object id。生产建议：**PDDL parsing 失败时不要重试 LLM，回退到 free-form Plan-and-Solve**——可形式化的好处和 LLM 不可靠性的 trade-off 不一定划算。

---

## 5. 经典 paper

- **Wang et al., 2023 — Plan-and-Solve Prompting** (ACL 2023) — 用两段 prompt 把 plan 与 solve 显式分开，GSM8K 涨 5-10%。**意义不在算法而在 mental discipline**：任何 agent 都该先试 plan-and-solve baseline 再决定上更复杂的 search/hierarchical。
- **Liu et al., 2023 — LLM+P: Empowering LLMs with Optimal Planning Proficiency** — 把 LLM 当 NL→PDDL 翻译器，用 classical planner 求最优解。读了能知道**符号 planner 在可形式化 domain 上仍是 LLM 短板**——别让 LLM 做它不擅长的 search。
- **Yao et al., 2023 — Tree of Thoughts** (NeurIPS 2023) — 已在 10.1 详读；本节看 §3 "Game of 24" 与 "Creative Writing" 两个 case，体会 ToT 如何把 reasoning 显式建成 tree，是 LATS 的直接前身。
- **Zhou et al., 2023 — Language Agent Tree Search (LATS)** (ICML 2024) — agent 树搜索的代表作，把 MCTS + Reflexion + value function 统一在 agent 框架下。读 §3 算法描述 + §4 在 HotpotQA / WebShop / HumanEval 三个 task 上的对比，理解为什么 search + reflection 比纯 ReAct 强一档。
- **Liang et al., 2022 — Code as Policies: Language Model Programs for Embodied Control** (ICRA 2023) — Google Brain 把 LLM 输出代码作为 robot plan 的奠基论文。读 §3 的 prompt 设计 + §4 的 robot demo，理解"为什么 code 比 free-form text plan 强"——是 Voyager / CodeAct / Code Interpreter 的共同祖先。
- 加分阅读：**Wang et al., 2023 — Voyager** (Minecraft lifelong learning agent + skill library)、**Hou et al., 2025 — Search-R1** (RL 把 search 内化进 reasoning model 的代表)、**Wang et al., 2024 — A Survey on LLM-Based Agent Planning** (综述)。

---

## 6. 自测与面试题

**Q1（对比）**：Plan-and-Solve / ToT / LATS / Code-as-Policy 各自适合什么任务？给出选型 decision tree。

<details>
<summary>Answer sketch</summary>

应当包含一张 decision tree + 每种方法的 sweet spot：

```
任务类型？
├── 数学/常识 reasoning（无 tool / 单次 generation）
│   └── Plan-and-Solve（最简单、最通用 baseline）
│       └── 难任务可叠 SC / BoN
│
├── short search task（Game-of-24 / 24-step planning，small action space）
│   └── ToT（深度 ≤ 5、branch ≤ 5，无需外部 verifier）
│
├── 长 trajectory agent task（web / SWE / 多步 tool）
│   └── 有 verifier 且单 sample 通过率 < 40%
│       └── LATS（MCTS + reflection，cost 高但是 hard task SOTA）
│   └── reasoning model + ReAct 单 sample 已够好
│       └── 不上 LATS，省 cost
│
└── 任务可代码化（机器人 / data analysis / 可 unit test）
    └── Code-as-Policy（控制流强、可验证、可累积 skill）
        └── 必须配 sandbox
```

key takeaway：**先试 Plan-and-Solve / Reasoning Agent baseline，跑不通再考虑 search；可代码化的任务优先 Code-as-Policy**。

</details>

**Q2（实战）**：你做一个 SWE agent（修 GitHub issue、跑测试、提 PR），用 LATS-style search 还是 reasoning model + ReAct？给出判断标准。

<details>
<summary>Answer sketch</summary>

应当给出**多维度判断**而非二元答案：

| 判断标准 | 倾向 ReAct + reasoning model | 倾向 LATS |
|---|---|---|
| **任务难度** | bug fix / 单文件改动 | 跨文件重构 / 难 issue |
| **单 sample 通过率** | > 50%（reasoning model 在 SWE-Bench Verified 上 baseline） | < 30% |
| **verifier 可靠性** | unit test 全 pass 即认为成功（清晰） | 同左 |
| **预算** | per-issue < $1 | per-issue 可接受 $5-50 |
| **延迟** | 用户在线等待（< 1 min） | 异步任务（小时级 ok） |
| **rollback 成本** | 改动可 revert（git） | 改动可 revert |

**一般推荐**：先用 **reasoning model（Claude / o1 / DeepSeek-R1）+ ReAct + 强 verifier（pytest）**，在 SWE-Bench-Verified 这类 easy/medium 上能拿到 60-70%。剩余 30-40% 的难 issue 才考虑 LATS——把 search budget 集中在 hard task 上，**不要无差别上 search**。

加分点：提到现代 SWE agent（OpenHands / Aider / Devin）实际用的是 **reasoning model + iterative ReAct + selective rollback**，而不是完整 MCTS——cost 与 LATS 的 marginal 提升不划算。LATS 在学术 benchmark 仍 SOTA，但工业落地多是 ReAct + 局部 retry 这种 "lite search"。

</details>

**Q3（前沿）**：reasoning model（R1 / o1 / Claude）已经在 long-CoT 内嵌 plan + tool call + reflection。LATS 在 2025 之后还有意义吗？哪些任务仍需要外部 search？

<details>
<summary>Answer sketch</summary>

仍有意义，但 niche 缩小。需要分情形讨论：

**LATS 仍然必要的场景**：

1. **长 trajectory + 大 action space**：web agent 跑几十步、SWE agent 跨 repo 改十几个文件——reasoning model 单 trajectory 一旦走错没法回退，LATS 的 backtracking 仍是 SOTA
2. **真实环境带 stochasticity / failure**：tool 可能挂、API 可能超时、UI 可能变——外部 LATS 的 multi-trajectory 能 robust 应对，单 long-CoT 一条 path 跑挂就完
3. **极难 verifiable task**：math olympiad / formal proof / hard code generation，reasoning model 单 sample 30-40%，LATS + verifier 仍能涨 10-20 点（参考 AlphaProof 的 IMO 银牌）
4. **需要并行探索多种 strategy 的任务**：例如做 hyperparameter search、做 solution diversity——LATS 的 tree expansion 天然支持，long-CoT 单线程做不到

**LATS 退化的场景**：

1. **简单 / 中等任务**：reasoning model + ReAct 单 sample 90%+，外层 LATS 涨 < 1 点 cost ×50
2. **non-verifiable task**（聊天 / 写作）：MCTS 没有 reward signal 退化为随机 expand
3. **延迟敏感**（用户在线等）：LATS 的 50-500 LLM call 不可接受

**趋势观察**：

- "**reasoning agent 内化的 implicit search**" + "**外部 LATS-style explicit search**" 是互补关系，不是替代
- 工业界 2025 共识：**reasoning model + ReAct 当 default、hard task fallback 到 LATS**
- 学术上 LATS 仍是 hard agent benchmark 的 SOTA holder（WebArena hard split / SWE-Bench Verified hard split）
- 长期看，**search 可能继续被内化**——15.4 的 Search-R1 / ReSearch / Agent-R1 就是把 search 通过 RL 训进 model 自己——但内化"跨 trajectory ensemble"目前仍是开放问题

加分点：提到外部 search 的真实成本不是 token 而是**延迟**——LATS 串行 50 步 web action 需几分钟到几小时，对在线产品不可用，但对离线 agent（research / coding）完全可接受。**应用场景的延迟容忍度**而非纯 quality 指标，常常是 LATS 取舍的真正决定性因素。

</details>

---

## 7. 延伸阅读

- [Wang et al. 2023 — Plan-and-Solve (arXiv 2305.04091)](https://arxiv.org/abs/2305.04091) — 最简 planning baseline 的原 paper，5 分钟读完即可上手
- [Liu et al. 2023 — LLM+P (arXiv 2304.11477)](https://arxiv.org/abs/2304.11477) — LLM 与 classical planner 的 hybrid 范式
- [Yao et al. 2023 — Tree of Thoughts (arXiv 2305.10601)](https://arxiv.org/abs/2305.10601) + [官方 repo](https://github.com/princeton-nlp/tree-of-thought-llm) — Game-of-24 demo 是体感 ToT 的最快路径
- [Zhou et al. 2023 — LATS (arXiv 2310.04406)](https://arxiv.org/abs/2310.04406) + [官方 repo](https://github.com/lapisrocks/LanguageAgentTreeSearch) — agent 树搜索的奠基作
- [Liang et al. 2022 — Code as Policies (arXiv 2209.07753)](https://arxiv.org/abs/2209.07753) + [项目主页](https://code-as-policies.github.io/) — 含 robot demo 视频，直观理解 code-as-plan
- [Wang et al. 2023 — Voyager (arXiv 2305.16291)](https://arxiv.org/abs/2305.16291) — Code-as-Policy 在 Minecraft lifelong learning 上的代表落地
- [Shen et al. 2023 — HuggingGPT (arXiv 2303.17580)](https://arxiv.org/abs/2303.17580) — planner-executor 范式 + 多 model 调度
- [Hong et al. 2023 — MetaGPT (arXiv 2308.00352)](https://arxiv.org/abs/2308.00352) — Hierarchical multi-agent planning 的代表
- [Wang et al. 2024 — A Survey on LLM-Based Agent Planning (arXiv 2402.02716)](https://arxiv.org/abs/2402.02716) — agent planning 流派综述，建议作为本节后的一站式回顾
- 推荐继续读：本教程 **14.5** Memory（LATS 的 reflection 池本质是一种 memory）、**14.6** Multi-agent（Hierarchical 的完整展开）、**15.4** Reasoning + Agent（Search-R1 / ReSearch / Agent-R1 把 search 通过 RL 内化的前沿路线）
