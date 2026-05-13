---
title: "14.1 Agent 范式概览：从 ReAct 到 Reasoning Agent"
description: "LLM Agent 的本质是 \"LLM 当 controller、tool 当 capability、loop 把两者串起来\"——这一节梳理 2022-2026 从 ReAct 到 reasoning agent 的范式演进、核心组件抽象、主流框架对比，并为后续 14.2-14.7 的具体实现搭好坐标系。"
---

> ⏱ 预计阅读 35 分钟 ｜ 难度 ★★ ｜ 前置：13.1（Prompt 工程）、13.4（Function calling）

## 一句话本节讲什么

LLM Agent 的本质是 **"LLM 当 controller、tool 当 capability、loop 把两者串起来"**——这一节梳理 2022-2026 从 ReAct 到 reasoning agent 的范式演进、核心组件抽象、主流框架对比，并为后续 14.2-14.7 的具体实现搭好坐标系。

---

## 1. Mental model（直觉）

### 1.1 LLM 与 Agent 的本质区别

把"LLM 调用"摆开成两种模式：

```
模式 A：Chat LLM（一次 forward）
  user prompt ──► [LLM] ──► text answer
                  一次结束

模式 B：Agent（loop + 外部世界）
              ┌────────────────────────┐
              │                         ▼
  user goal ─►│      ┌──────────┐    ┌──────────┐
              │      │   LLM    │ ──►│  Action  │
              │      │ (Brain)  │    │ (tool)   │
              │      └─────▲────┘    └─────┬────┘
              │            │               │
              │            └─Observation◄──┘
              └─── 直到任务完成 / step 上限 ──┘
```

关键差异有三：

1. **多次调用**：Chat 是单次 forward；Agent 是一个 **loop**，每次 forward 输出一个 action（thought + tool call），执行后把 observation 喂回 LLM 继续 reason
2. **接外部世界**：Chat 完全在 token 空间内完成；Agent 通过 tool（API、shell、browser、screen）**改变和感知外部状态**
3. **任务驱动**：Chat 的"完成"由 user 判断；Agent 的"完成"由 LLM 自己判断（输出 final answer / 触发 stop）

一句话总结：**LLM 是一个会说话的大脑；Agent 是这个大脑加上手脚和环境的反馈回路**。

### 1.2 Agent = LLM + tool + memory + planner + loop

工业上把 agent 拆成 5 个组件：

| 组件 | 角色 | 类比 |
|---|---|---|
| **LLM** | controller / brain | 决策中枢 |
| **Tool** | capability | 手脚（搜索、代码执行、API） |
| **Memory** | state | 记忆（短期 scratchpad / 长期 vector DB） |
| **Planner** | task decomposition | 大脑的"计划区"（可选，简单任务无需） |
| **Loop / Orchestrator** | control flow | 心跳（while not done） |

简单 agent 只需要 LLM + Tool + Loop（即 ReAct）；复杂 agent 才需要 planner 和 long-term memory。后续节会逐个展开（memory 在 14.5、planner 在 14.4）。

### 1.3 三要素：Observation、Action、Reasoning

ReAct 论文提出的 OAR 抽象成了所有 agent 范式的最小共识：

- **Observation**：环境给 agent 的输入（tool 返回值、网页内容、screenshot、user reply）
- **Action**：agent 输出的、会改变外部世界的指令（call API、点击按钮、写文件）
- **Reasoning（Thought）**：agent 的内部思考——分析当前 observation、决定下一步 action

Chat LLM 只有 reasoning，没有 action / observation；Agent 把这三者串成 loop。

---

## 2. 范式演进时间线（2022-2026）

时间线按"有什么新范式被立起来"排，不按 paper 引用数。

### 2.1 2022 H1：CoT — 让 LLM"显式思考"

**Wei et al. 2022 《Chain-of-Thought Prompting》** 证明：在 prompt 里加 reasoning 示例，能让 LLM 解决之前做不了的多步推理（GSM8K 从 17%→78%）。

这一阶段 LLM 还只是"会说话的大脑"——没有 tool、没有 loop。但 **CoT 是 agent 的前作**：它建立了"LLM 可以输出结构化思考过程"这个心智模型，为后面的 ReAct 把 thought 提到 first-class 位置铺路。

### 2.2 2022.10 ReAct — Agent loop 范式诞生

**Yao et al. 2022 《ReAct: Synergizing Reasoning and Acting》** 是 agent 时代的真正起点。它把 LLM 输出从纯 text 改成 **`Thought → Action → Observation` 交错结构**，由外部 driver 解析、执行、回填。这是第一个让 LLM 跳出"text-only"的形式化范式，几乎所有现代 agent 框架都是它的变体。

### 2.3 2023 Q1：自我改进 + tool 学习

三件事在同一季度发生：

- **Toolformer (Schick 2023)**：self-supervised 标注——让 base LLM 自己尝试在 token 流里插 API call，**API 让 loss 下降的就保留**，把 tool use 烧进 weights，第一次证明"模型可以自己学会调 tool"
- **Reflexion (Shinn 2023)**：失败后用自然语言写一份 reflection 存进 memory，下一轮重试。这是 verbal RL 的雏形——不更新 weights，但用 in-context memory 做"轻量学习"
- **HuggingGPT (Shen 2023)**：LLM 当 controller，调度 HuggingFace 上的多个 expert model（vision / speech / NLP）。"LLM 是 task-routing 大脑"的范式被点亮

### 2.4 2023 Q2：自主 agent 工程化爆发

**AutoGPT / BabyAGI / AgentGPT** 在 GitHub 一夜爆火，做的事其实就是"ReAct + 长任务循环"。工程粗糙、效果不稳，但点燃了开发者社区，让"autonomous agent"从论文概念变成大众词汇。

教训：当时这一波 agent 普遍**完成不了真实任务**，原因是底层 LLM（GPT-3.5）reasoning 能力不够。这是"必要条件 = 强 base model"的最早佐证。

### 2.5 2023 Q3-Q4：Multi-agent 抽象

**CAMEL（双 agent role-play）→ MetaGPT（SOP 标准化）→ ChatDev（waterfall）→ AutoGen（conversable agent）** 一连串 multi-agent 框架出现。核心思路：把任务分解给"扮演不同角色"的 agent，模仿人类团队协作。

学术上漂亮，但工业上后续证明 **multi-agent 不一定优于强 single agent + good tool**——这一点 2025 后被反复实证（见第 6 节"现状评估"）。

### 2.6 2024：Framework 收敛

**LangGraph（state machine + graph）/ OpenAI Assistants API（thread + run）/ Anthropic Tool Use API**——三家把碎片化的 agent 实现收敛到了几个标准抽象上。从这一年起，做 agent 不再是"自己造轮子"，而是"在哪个 framework 上拼组件"。

### 2.7 2024.10：Computer Use — agent 走出 API

**Anthropic Computer Use (Claude 3.5 Sonnet)** 让 agent 直接看 screenshot、控制鼠标键盘——agent 第一次可以操作"任何"软件，不再受限于"有 API 的 service"。OpenAI Operator (2025)、UI-TARS 跟进，**GUI agent** 成为新方向。

### 2.8 2025.01：R1 浪潮 — Reasoning + Agent 融合

**DeepSeek-R1 (2025)** 证明：纯 RL + verifiable reward 能让 base model 自己长出 long-CoT 推理能力。这个范式被立刻搬到 agent：

- **Search-R1 / ReSearch**：multi-turn search agent，用 RL 训"什么时候 search、用什么 query"
- **ReTool**：把 long-CoT 的"思考节奏"和"调 tool 的节奏"统一在一个 RL loop 里
- **Agent-R1 / Qwen3-Agent / GLM-Z1-Agent**：业界把 R1 范式做成 agentic 产品

这催生了 **reasoning agent**：长 CoT thinking + 多轮 tool call **被同一套 RL 训到 weights 里**。这与传统 ReAct（prompt-based、无 RL）有本质差异（详见第 6 节）。

### 2.9 2025-2026：Reasoning Agent 成形

到 2026 年，主流 frontier 模型（Claude 4.x、o3、GPT-5、Qwen3、GLM-Z1）都同时具备 **long thinking + agentic tool use** 能力。Agent 已经从"prompt + framework"变成"模型本身就是 agent"——这是当前最新的范式形态。

---

## 3. ReAct 范式详解

### 3.1 核心 4 步循环

ReAct 的 prompt 形式（伪代码）：

```
你可以用以下 tool: search(query), calculate(expr)
按以下格式回答：
Thought: <你的分析>
Action: <tool_name>(<args>)
Observation: <由系统填入>
... (循环) ...
Thought: I now know the final answer.
Final Answer: <答案>
```

完整循环：

1. **Thought**：LLM 基于当前上下文输出一段思考——分析 observation、决定下一步要做什么
2. **Action**：紧接着输出一条 tool call（如 `search("ReAct paper authors")`）
3. **External execution**：driver（agent runtime）解析 action、执行 tool、得到结果
4. **Observation**：把 tool 返回值作为 `Observation: ...` 拼到上下文末尾，回到第 1 步

直到 LLM 输出 `Final Answer:` 或达到 max step。

### 3.2 为什么 ReAct 是 agent 的"原型"

ReAct 干了三件关键的事，奠定了所有后续范式的基线：

1. **把 thought 变成 first-class 输出**——之前 LLM 只输出 text，ReAct 让它输出 `Thought + Action` 这种**结构化的内部状态 + 外部行动**
2. **把 LLM 与外部世界解耦**——LLM 只负责生成 action 描述，**不负责执行**；execution 由 trusted runtime 做。这是 sandbox / safety / determinism 的基础
3. **建立 observation 回路**——observation 是 LLM 之前没见过的 token，**它强制 LLM 接受外部反馈**，避免 hallucinate（"假装查了 wiki 然后编答案"）

后续的 Reflexion = ReAct + 失败后 reflection；ToolLLaMA = ReAct trajectory 拿来做 SFT；Search-R1 = ReAct 拿来做 RL。**全是 ReAct 的延伸**。

### 3.3 ReAct vs Plan-and-Execute

主要的对立范式是 **Plan-and-Execute**（HuggingGPT、LLM+P、Plan-and-Solve）：

| 维度 | ReAct | Plan-and-Execute |
|---|---|---|
| 决策时机 | 边走边想（reactive） | 先全 plan 再执行（proactive） |
| Adaptivity | 高（每步根据 observation 调整） | 低（plan 写定后难修正） |
| Token 消耗 | 中（thought 反复出现） | 低（plan 只写一次） |
| 适用任务 | open-ended、需 exploration | well-defined、步骤可枚举 |
| 工程难度 | 低（一个 loop 即可） | 中（需 plan parser + executor） |

**现代 trend 是 hybrid**：top-level 用 plan 给 high-level subgoal，每个 subgoal 内部用 ReAct 探索。LATS、Code-as-Policy 都属于这种混合范式（在 14.4 详讲）。

---

## 4. 现代 Agent 主流类型

按"用在哪"分，2026 年主流 agent 大致 7 类：

| 类型 | 代表 | 适用场景 |
|---|---|---|
| **Chat agent** | OpenAI Assistants、Claude Tool Use | 通用对话 + 简单 tool |
| **Coding agent** | Devin、OpenHands、Claude Code、Cursor Agent | 真实 repo 的代码任务（SWE-bench） |
| **Research agent** | OpenAI Deep Research、Perplexity、Gemini Deep Research | 多 source 搜索 + 综合写报告 |
| **Browser agent** | WebVoyager、UI-TARS、Browser-Use | 网页交互（点击、填表、爬数据） |
| **Computer Use agent** | Anthropic CU、OpenAI Operator | OS 级 GUI（截图 + 鼠键控制） |
| **Multi-agent** | MetaGPT、AutoGen、CrewAI | 复杂任务分解 / 角色协作 |
| **Reasoning agent** | Search-R1、ReTool、Agent-R1、Qwen3-Agent | R1-derived，long-CoT + 多轮 tool |

类型之间不互斥——Claude Code 既是 coding agent 也是 reasoning agent；Devin 内部用了 multi-agent 架构。但每类有自己的**主战场 benchmark**：

- Coding agent → **SWE-bench Verified**
- Research agent → **GAIA**
- Browser agent → **WebArena / Mind2Web**
- Computer Use agent → **OSWorld**
- General agent → **AgentBench / τ-bench**

记住这 5 个 benchmark 名，是 2026 年讨论 agent 的最低准入。

---

## 5. Agent Framework 对比

主流 framework 选型对比（按"接近底层 / 自由度"排序）：

| Framework | 风格 | 强项 | 弱点 |
|---|---|---|---|
| **直接写 LLM call** | 自己写 loop | 完全控制、零依赖 | 重复造轮子 |
| **smolagents (HF)** | 极简 + code-as-action | 教学友好、~1000 行核心代码 | 生态小 |
| **LangChain (LCEL)** | composable chain | 生态最广、connector 多 | 抽象重、debug 难 |
| **LangGraph** | state machine + graph | 复杂 workflow、可视化、checkpoint | 学习曲线陡 |
| **OpenAI Assistants API** | 托管 + thread 抽象 | 简单部署、内置 RAG / code interp | vendor lock-in、灵活度低 |
| **Anthropic Tool Use / Computer Use** | tool + screenshot | GUI agent 唯一原生方案 | 仅 Claude |
| **AutoGen (Microsoft)** | conversable agent + group chat | multi-agent 协作直观 | 单 agent 任务过度设计 |
| **CrewAI** | role-based crew | 易上手、模板化 | 灵活度低 |

**选型经验**：

- 学习 / 原型：**直接写 loop** 或 **smolagents**——理解底层最重要
- 生产 single agent：**LangGraph** 或 **OpenAI Assistants**——前者灵活、后者省事
- 生产 multi-agent：**AutoGen** 或 **LangGraph**——AutoGen 表达更自然
- GUI / Computer Use：**Anthropic Computer Use API**——目前唯一成熟方案
- 不要：**LangChain (legacy chain)**——已被 LangGraph / LCEL 取代，新项目别用

---

## 6. 2026 年 Agent 现状的关键判断

四个 take-away，每一个都对设计决策有直接影响：

### 6.1 Single agent + good tool 通常 ≥ multi-agent

2024-2025 多个 benchmark（SWE-bench、GAIA、τ-bench）的实证结论：**强 single agent 配好 tool**，往往打过精心设计的 multi-agent 系统。原因：

- Multi-agent 引入额外的 communication overhead 与 error propagation
- 强 frontier model 单 agent 的 reasoning ability 已经足够做 task decomposition
- Multi-agent 的"角色化"在评测里更多是 demo 价值，不是性能价值

**工程建议**：默认 single agent + 丰富 tool；只在任务有明确**并行 / 异构专精**需求时才上 multi-agent（例：一个 agent 写代码、一个 agent 跑 test、一个 agent 做 code review，且这三件事可流水线）。

### 6.2 Computer Use 仍未稳定

虽然 demo 惊艳，但 2026 年 OSWorld 上 SOTA 成功率仍 < 50%。主要痛点：

- Screenshot 解析对 vision model 要求高（小按钮、密集文本）
- GUI 状态难表达（每个 tick 状态都在变）
- Action 粒度难选（pixel-level 太累、widget-level 又依赖 accessibility tree）

不要急于把 Computer Use 推进生产。它现在的位置类似 2023 年的 AutoGPT——demo 可看，产品化早。

### 6.3 MCP 协议 — Tool 接口标准化

**MCP (Model Context Protocol, Anthropic 2024)** 是 agent 时代的"USB"——一个开源协议，让 tool 提供方和 agent 框架之间用统一格式握手。2026 年已成事实标准：

- Claude Code、Cursor、OpenHands 等主流 coding agent 都支持
- 各类工具（GitHub、Linear、Slack、Postgres、文件系统）都开始提供 MCP server
- Agent 不再绑死在某个 framework 的 tool registry 上

学 agent 的工程师必须知道 MCP——这是接 tool 的"标准插孔"。

### 6.4 Reasoning agent 是当前前沿

R1 之后，**长 CoT thinking + 多轮 tool call 被同一个 RL loop 训到 weights** 已成新范式。它与传统 ReAct 的差异：

| 维度 | 传统 ReAct agent | Reasoning agent (R1-derived) |
|---|---|---|
| Thinking 长度 | 短（每步 50-200 token） | 长（每步 500-5000 token，整轨迹可达 50k+） |
| Tool 触发时机 | prompt 决定 | weights 决定（model 自己学会何时调） |
| Reasoning 来源 | prompt 引导 | RLVR 烧进 weights |
| 训练方式 | 通常无训练（纯 prompt）或 SFT | RL（多轮 GRPO / PPO） |
| Token 成本 | 中 | 高（5-10×） |
| Robustness | 中（依赖 prompt 设计） | 高（observation perturbation 下更稳） |
| 部署需求 | 任意 LLM API | 需要 long-context 推理引擎 + 大 KV cache |

Reasoning agent 是 2025-2026 的主线，但部署门槛高（详见第 7 节踩坑）。

---

## 7. 最小代码示例

### 7.1 最简 ReAct loop（不用 framework）

下面 30 行实现一个完整的 ReAct agent，用 OpenAI API + 一个假的 search tool 演示循环结构。

```python
import json, re
from openai import OpenAI

client = OpenAI()
TOOLS = {"search": lambda q: f"[search result for '{q}': ReAct was proposed by Yao et al. in 2022]"}

SYSTEM = """You are a ReAct agent. Reply ONE step at a time in this exact format:
Thought: <your reasoning>
Action: <tool_name>(<args>)
or, when done:
Thought: I have the final answer.
Final Answer: <answer>"""

def react(question: str, max_steps: int = 10):
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": question}]
    for step in range(max_steps):
        out = client.chat.completions.create(model="gpt-4o-mini", messages=msgs,
                                             stop=["Observation:"]).choices[0].message.content
        print(f"--- Step {step} ---\n{out}")
        if "Final Answer:" in out:
            return out.split("Final Answer:")[-1].strip()
        m = re.search(r"Action:\s*(\w+)\((.*?)\)", out)
        if not m: raise ValueError(f"No action in: {out}")
        tool, arg = m.group(1), m.group(2).strip("\"' ")
        obs = TOOLS[tool](arg)
        msgs.append({"role": "assistant", "content": out})
        msgs.append({"role": "user", "content": f"Observation: {obs}"})
    raise RuntimeError("Max steps exceeded")

print(react("Who proposed the ReAct paradigm and when?"))
```

关键点：
- `stop=["Observation:"]` 阻止 LLM 自己幻觉 observation——必须由 driver 填
- `max_steps` 是**死循环防御的硬约束**，必须设
- 一次 LLM 输出包含 `Thought + Action`，driver 解析 action、执行 tool、把 observation 拼回去——这就是 ReAct 的完整骨架

### 7.2 OpenAI Assistants API 最小用法

```python
from openai import OpenAI
client = OpenAI()

assistant = client.beta.assistants.create(
    name="Math Tutor",
    instructions="You are a math tutor. Use code interpreter to solve problems.",
    model="gpt-4o",
    tools=[{"type": "code_interpreter"}],
)
thread = client.beta.threads.create()
client.beta.threads.messages.create(
    thread_id=thread.id, role="user",
    content="Solve x^2 - 5x + 6 = 0 step by step using Python.",
)
run = client.beta.threads.runs.create_and_poll(
    thread_id=thread.id, assistant_id=assistant.id,
)
msgs = client.beta.threads.messages.list(thread_id=thread.id)
print(msgs.data[0].content[0].text.value)
```

Assistants API 把 `thread`、`run`、`tool execution`、`memory` 全托管，不用自己写 loop——代价是 vendor lock-in 与黑盒。

### 7.3 smolagents 一行调 agent

```python
from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel

agent = CodeAgent(
    tools=[DuckDuckGoSearchTool()],
    model=HfApiModel(model_id="meta-llama/Llama-3.3-70B-Instruct"),
)
result = agent.run("How many ICML papers had 'agent' in the title in 2024?")
print(result)
```

smolagents 的特色是 **code-as-action**——agent 直接生成 Python code 调 tool（而不是 JSON tool call），适合教学与原型。`CodeAgent` 内部就是一个 ReAct loop + Python sandbox。

---

## 8. 工程踩坑与经验

- ❗ **ReAct loop 必须设 max step**（典型 10-20）。LLM 可能陷入 "Thought: I should search again..." 的死循环——见过 GPT-3.5 在简单问题上跑 50 步只为反复 query 同一关键词。**没有 max step 的 agent 是定时炸弹**，会烧光 budget
- ❗ **Tool description 直接决定 tool 调用准确率**——把 tool 当 API 文档写：name 短、description 一句话讲"what it does"+"when to use"+"when NOT to use"，参数描述含示例。tool 多于 10 个时考虑 tool retrieval（用 embedding 选 top-k 给 LLM）
- ❗ **Multi-agent 不是默认更好**——除非任务能明确并行 / 专精，single agent + good tool 通常更稳。引入 multi-agent 前先量化：是否真有 sub-task 可独立执行？communication 成本是否 < 收益？这是个 hard problem，别为了"看起来高级"上 multi-agent
- ❗ **Computer Use 当前不稳**，screenshot 解析失败、GUI 状态漂移、action 粒度难调，OSWorld 成功率 < 50%。可以做 demo / 内部工具，**不要急于产品化** to 2C
- ❗ **Agent 的 token consumption 远超 chat**——一次 GAIA 任务 50k-500k token 是常态，SWE-bench 单 task 可达 1M+。**cost 必须算清**：每步 prefill 都重算（除非用 prompt cache），long thinking model 还要再 ×3-10
- ❗ **Reasoning agent (R1-derived) trajectory 极长**——单条可达 50k+ token，KV cache 压力巨大。部署需 vLLM + GQA + 量化（INT4 / FP8）+ chunked prefill，否则 GPU 显存炸
- ❗ **Agent benchmark 的"成功率"不可直接比**——同一 SWE-bench Verified，不同 setup（max step、tool set、retry 策略、prompt）成功率可差 15-30%。看 paper 一定要看 evaluation 章节的细节
- ❗ **不要在 agent 里做 stateful 副作用 without idempotency**——LLM 可能因为 retry / 误判重复执行同一个 action，导致重复发邮件、重复下单。生产环境的 tool 必须设计为**幂等**，或 driver 层做去重
- ❗ **Tool failure 必须有明确 error 反馈**——很多 framework 把 tool exception silently swallow，agent 只看到空 observation 就开始幻觉。把 exception 包装成结构化 `Observation: ERROR(<msg>)` 喂回去，让 LLM 显式处理
- ❗ **Observation 注入位置**：放在 user role 而不是 assistant role——这样模型把 observation 当"环境反馈"而不是"自己的输出"，避免 next-step 把 observation 内容当作自己的 prior reasoning 重复

---

## 9. 经典 paper

- **Yao et al., 2022 — ReAct: Synergizing Reasoning and Acting in Language Models** — 必读。Agent 时代的真正开山作，第一次形式化 `Thought → Action → Observation` loop。读 §2 的方法定义 + §3 的 HotpotQA / ALFWorld 实验，理解为什么"reasoning 与 action 交错"比"先 plan 再做"更鲁棒
- **Wei et al., 2022 — Chain-of-Thought Prompting Elicits Reasoning in LLMs** — Agent 的前作。CoT 把 reasoning 当 first-class output 这个心智模型，是 ReAct 的直接基础
- **Wang et al., 2024 — A Survey on Large Language Model based Autonomous Agents** — 综述。系统梳理 perception / brain / action / memory 四组件，及多 agent 协作模式。读完一遍能建立完整的 agent landscape mental map
- 加分：**Anthropic 2024 — Computer Use Blog & Tool Use Documentation** — 工程视角看 tool / computer use 的 API 设计；**Hong et al., 2023 — MetaGPT** + **Wu et al., 2023 — AutoGen** + **Shen et al., 2023 — HuggingGPT** 是 multi-agent / LLM-as-controller 三个代表

---

## 10. 自测与面试题

**Q1（演化）**：列出 2022-2026 Agent 范式 5 个关键节点 + 各自的范式贡献。

<details>
<summary>Answer sketch</summary>

要覆盖的关键节点（5 选其中典型 5 个）：

1. **2022 H1 — CoT (Wei 2022)**：让 LLM 显式输出 reasoning step，是 agent 的"前作"。贡献：建立"reasoning 可作为 first-class 输出"的心智模型
2. **2022.10 — ReAct (Yao 2022)**：Agent loop 范式正式诞生。贡献：形式化 `Thought → Action → Observation` 交错结构，把 LLM 与外部世界解耦
3. **2023 Q1 — Toolformer / Reflexion / HuggingGPT**：tool learning + verbal RL + LLM-as-controller 三件事同期发生。贡献：分别证明"模型可自学 tool"、"verbal reflection 可做轻量学习"、"LLM 可调度异构 model"
4. **2023 Q3-Q4 — Multi-agent (CAMEL / AutoGen / MetaGPT)**：协作范式抽象化。贡献：role-play / SOP / conversable agent 等模式被工业化
5. **2024.10 — Computer Use (Anthropic)**：Agent 走出 API。贡献：第一次让 agent 用 screenshot + GUI 操作任何软件
6. **2025.01 — DeepSeek-R1 + 衍生 Reasoning Agent (Search-R1 / ReTool / Agent-R1)**：Long-CoT + 多轮 tool call 被 RL 烧进 weights。贡献：reasoning agent 这个新形态成形

加分：能区分"纯 prompt 驱动的 agent"（ReAct / AutoGPT）vs "训练驱动的 agent"（Toolformer / R1-derived），并能说明 framework 收敛（2024 LangGraph / Assistants）也是一个范式节点

</details>

**Q2（架构）**：写出 ReAct 的核心 4 步循环；它与 Plan-and-Execute 的差异？

<details>
<summary>Answer sketch</summary>

**ReAct 4 步循环**：
1. **Thought**：LLM 基于当前上下文（含历次 observation）输出推理步骤——分析、决策下一步做什么
2. **Action**：紧接着输出一条 tool call（如 `search("...")`），由 driver 解析
3. **External execution**：runtime 执行 tool，得到结果
4. **Observation**：把结果拼回上下文（通常以 `Observation: ...` 形式 + user role）

直到模型输出 `Final Answer:` 或达到 max step 上限

**与 Plan-and-Execute 的核心差异**：

| 维度 | ReAct | Plan-and-Execute |
|---|---|---|
| 决策时机 | 边走边想（reactive） | 先全 plan，再按 plan 执行 |
| Adaptivity | 高，能根据 observation 实时调整 | 低，plan 写死后难修正 |
| Token 成本 | 中 | 低（plan 只算一次） |
| 适用任务 | open-ended、需 exploration（QA、搜索） | well-defined、步骤可枚举（pipeline 调度） |

**现代 trend 是 hybrid**：top-level plan 给出 high-level subgoal，每个 subgoal 内部用 ReAct 处理细节。LATS、Code-as-Policy 都是这种混合范式。

加分：能指出 ReAct 的 observation 必须由 driver 填（不能让 LLM 自己幻觉），以及 max step 是必须的死循环防御

</details>

**Q3（前沿）**：R1 之后，reasoning agent 与传统 ReAct agent 有哪些核心差异？

<details>
<summary>Answer sketch</summary>

至少要覆盖以下几个维度的对比：

**1. Reasoning 来源**：
- 传统 ReAct：reasoning 由 prompt 引导（"Thought:"），思考长度短（每步 50-200 token）
- Reasoning agent：long-CoT thinking 被 RLVR 烧进 weights，思考长度长（每步 500-5000 token，整轨迹可达 50k+）

**2. Tool 触发时机**：
- 传统 ReAct：何时调 tool 由 prompt 引导，模型遵循 prompt 模板
- Reasoning agent：模型自己学会"何时该 search、何时该 calc、何时停"，权重内化

**3. 训练方式**：
- 传统 ReAct：通常 zero-shot prompt，或 SFT trajectory（FireAct / Agent-FLAN）
- Reasoning agent：多轮 RL（GRPO / PPO），trajectory-level reward + 可能含 process reward

**4. Robustness**：
- 传统 ReAct：依赖 prompt 设计，observation 扰动 / tool failure 下容易崩
- Reasoning agent：RL 训练中见过失败 trajectory，recovery 与 retry 行为更鲁棒

**5. Token 成本与部署**：
- 传统 ReAct：任意 LLM API 都能跑
- Reasoning agent：trajectory 极长（50k+），KV cache 压力巨大，需 vLLM + GQA / MLA + 量化 + chunked prefill；token 成本 5-10×

**6. 评测表现**：
- 在 SWE-bench Verified、τ-bench、GAIA 等 hard benchmark 上，reasoning agent 显著优于纯 prompt ReAct
- 但在简单 chat + 1-2 个 tool 的场景，强 frontier 模型的 ReAct 已足够好

加分：能指出"reasoning agent 不是 ReAct 的替代，而是 ReAct 的'权重化'"——`Thought → Action → Observation` 的范式没变，变的是 thought 的长度与权重内化程度。也可以指出 multi-agent vs single reasoning agent 之争——业界正在向 single reasoning agent + good tool 收敛

</details>

---

## 11. 延伸阅读

- [Lilian Weng — LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) — OpenAI 前研究员的系统综述，至今仍是 agent 概念地图最好的入门
- [Anthropic — Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents) — Anthropic 工程团队 2024 的 agent design pattern 总结，强调 simplicity over complexity，必读
- [Anthropic — Computer Use & Tool Use Docs](https://docs.anthropic.com/en/docs/build-with-claude/computer-use) — 唯一原生 GUI agent 文档，工程细节丰富
- [Model Context Protocol (MCP)](https://modelcontextprotocol.io) — 2026 年事实标准的 tool 接口协议
- [smolagents](https://github.com/huggingface/smolagents) — HF 的极简 agent 框架，~1000 行核心代码，是读源码理解 agent 内部最好的入口
- [LangGraph](https://langchain-ai.github.io/langgraph/) — 复杂 workflow agent 的工业首选
- [Awesome LLM-Powered Agent](https://github.com/hyp1231/awesome-llm-powered-agent) — Agent 论文与工具的 awesome list，更新活跃
- 推荐继续读本教程的 **14.2 节《实现一个最小 ReAct + Reflection agent》**——把本节的 ReAct 概念真正落到 200 行可跑代码上
