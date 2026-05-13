---
title: "14.7 Agent Framework 对比：LangGraph / OpenAI Assistants / Anthropic Tool Use"
description: "主流 agent framework 不是\"哪个更强\"的对比，而是设计哲学的分叉——OpenAI Assistants 走 vendor 托管、Anthropic Tool Use 走原生协议、LangGraph 走显式 state machine、smolagents 走 code-as-action、AutoGen / CrewAI / MetaGPT 走 multi-agent；本节给一棵选"
---

> ⏱ 预计阅读 35 分钟 ｜ 难度 ★★ ｜ 前置：14.2（手撕 ReAct）

## 一句话本节讲什么

主流 agent framework 不是"哪个更强"的对比，而是**设计哲学的分叉**——OpenAI Assistants 走 vendor 托管、Anthropic Tool Use 走原生协议、LangGraph 走显式 state machine、smolagents 走 code-as-action、AutoGen / CrewAI / MetaGPT 走 multi-agent；本节给一棵选型决策树 + 4 段最小代码 demo + Anthropic 2024《Building Effective Agents》提倡的"先 workflow 再 agent"原则，让你在原型 / 生产 / 教学三种场景下知道用哪套。

---

## 1. Mental model（直觉）

### 1.1 Framework 谱系图

把 2024-2026 主流 framework 按"抽象层次 × 控制粒度"摊在一张图上：

```
              抽象层次 高
                  ↑
   托管 API ──────┼──────── 多 agent 编排
   (Assistants)   │       (AutoGen / CrewAI / MetaGPT)
                  │
   ──────────────┼──────────────  控制粒度
   左：黑盒易用   │   右：白盒可控
                  │
   原生协议 ──────┼──────── State machine
   (Anthropic    │       (LangGraph)
    Tool Use)    │
                  │
                  │   极简代码 / 自己写
                  │   (smolagents / 14.2 手撕)
                  ↓
              抽象层次 低
```

四个象限的工程含义：

- **左上（托管，黑盒）**：OpenAI Assistants 这类 API 把 thread / message / run / tool 全托管在云端，调一次 API 就能跑——上手 5 分钟，但 vendor lock 重、debug 黑盒
- **右上（编排，黑盒）**：AutoGen / CrewAI / MetaGPT 把 multi-agent 通讯抽象成 role / task / dialogue，适合复杂协作但黑盒多
- **右下（白盒，可控）**：LangGraph 的 state machine + node + edge 是显式的，你能精确画出 agent 在状态空间走的轨迹
- **左下（极简，白盒）**：smolagents / 14.2 手撕 < 200 行，没有抽象就没有黑盒——但要自己重造轮子

### 1.2 一句话区分 5 大主流

读者不需要全记，记住"它解决了什么独特问题"即可：

- **OpenAI Assistants**：把 agent state（thread + message + run）**托管在云端**，省你自己存 history 的活
- **Anthropic Tool Use**：原生 `tool_use` block + Computer Use（screenshot / bash / 鼠标），是**协议层**而非 framework 层
- **LangChain**：thin wrapper around LLM API，把 prompt / chain / tool / memory 标准化成**LCEL 表达式**
- **LangGraph**：在 LangChain 之上把 control flow 抽成 **state graph**，适合 stateful 多步 workflow + human-in-loop + checkpoint
- **smolagents**：HuggingFace 出的极简框架（~1000 行核心），主打 **code-as-action**——LLM 输出 Python code 而非 JSON

### 1.3 一句话区分 multi-agent 三家

- **AutoGen**（Microsoft）：通用 multi-agent **conversation framework**，agent 之间像聊天一样异步通讯
- **CrewAI**：role-based，每个 agent 有 `role / goal / backstory`，**易上手**做"团队分工"型任务
- **MetaGPT**：把"软件公司"SOP 硬编码进 framework——product manager / architect / engineer / QA 流水线产出 codebase

### 1.4 framework 用得早不如用得对

新人最常踩的坑：**一上来就上 LangGraph / AutoGen**。但 80% 的真实需求其实是"用一个 LLM + 几个 tool 跑通一个流程"——这种场景手写 < 200 行（14.2）反而更可控、更省调试时间。Anthropic 2024 的《Building Effective Agents》白皮书直接劝退："先尝试 prompt + tool，不行再 workflow，最后才考虑 agent"。

---

## 2. 三大主流详解：Assistants / Tool Use / LangGraph

### 2.1 OpenAI Assistants API：托管 thread + run

核心模型是 4 个云端对象：

```
Assistant（持久化 agent 配置：name / instructions / tools / model）
   │
   ├─ Thread（一段对话的 ID，自动累积 message）
   │     └─ Message（user / assistant / tool）
   │
   └─ Run（一次"启动 assistant 跑这个 thread"的执行实例）
         └─ Run Step（每一 step 的 tool call / message create）
```

**优势**：

- **不用自己存 history**——thread ID 即 state，下次接着传同一个 thread_id 即可
- **Tool 内置常用项**：File Search（自动 RAG over uploaded files）、Code Interpreter（云端 Python sandbox）、Function calling
- **5 分钟跑通**：注册 → create assistant → create thread → run → poll status → 拿结果

**劣势**：

- **Vendor lock**：换模型（如换 Claude / Qwen）等于全部重写
- **黑盒 debug**：thread 里的 retrieval / chunking / reranker 你看不到
- **Token cost 不可控**：thread 长了之后每 turn 都把整个 history 喂进去，不能像本地一样自己 truncate
- **Polling 模式**：要循环 poll `run.status`，不像本地 streaming 那么直接

**适用场景**：内部 demo / hackathon / 不在乎 vendor lock 的快速 prototype。**生产几乎不用**——大厂业务里几乎没人把 agent state 丢给 OpenAI 托管。

### 2.2 Anthropic Tool Use + Computer Use：协议层

Anthropic 走的不是"framework"路线，而是**把 agent 能力做进 API 协议**：

- **Tool Use**：response 直接出 `tool_use` content block（带 `id / name / input`），你执行后用 `tool_result` block 回传——比 OpenAI function calling 更结构化
- **Computer Use**（2024.10 发布，2026 年仍 beta）：4 个原生 tool——`computer_20250124`（screenshot + 鼠标 + 键盘）、`text_editor_20250429`（文件编辑）、`bash_20250124`（命令行）、`web_search_20250305`——让 LLM 直接操作 OS / browser

**Computer Use 的工程含义**：以前做 GUI agent 要自己接 Playwright / pyautogui / OCR 一堆，现在 Anthropic 把它做成标准 tool——你只负责跑一个 VM 容器，LLM 自己看 screenshot 决定下一步点哪。

**适用场景**：

- 需要 GUI / browser / OS 操作的 agent（Claude Code、Manus、各类 computer-use demo）
- 想要"模型层 + 协议层"打通、不绕一层 framework 的极简栈
- 注意 Computer Use 仍 alpha——准确率和延迟未到 production，**仅做 demo 与研究**

### 2.3 LangChain → LangGraph 演进

**LangChain（2022-2023）**：早期把 prompt / chain / agent / memory / retriever 全部抽象，结果**抽象太多 + 接口频繁 breaking change**——2023 年圈内 meme："debug LangChain 比 debug 自己写还难"。

**LCEL（LangChain Expression Language，2023）**是 LangChain 团队的"自救"——用 `|` 管道运算符把组件拼起来：

```python
chain = prompt | llm | output_parser
result = chain.invoke({"question": "..."})
```

**LangGraph（2024-，现在的主线）**：直接放弃"什么都抽象"的路线，回归到**显式 state machine**——你要画出节点（node）和边（edge），state 在节点间流动。

```
       ┌──────┐
   ───▶│ plan │──┐
       └──────┘  │
                 ▼
            ┌────────┐  no
   ┌────────│ act    │◀──── decide_next
   │        └────────┘  yes
   ▼
┌──────┐  ┌────────┐
│ tool │──│ check  │── done ──▶ END
└──────┘  └────────┘
```

**LangGraph 的杀手特性**：

- **Checkpoint / 持久化**：每个 node 执行后自动 snapshot state，崩了能从断点恢复
- **Human-in-the-loop**：在某 edge 上插 `interrupt_before=["dangerous_action"]`，等人审批后再继续
- **Time travel**：能回到任何历史 state 重跑
- **Subgraph**：复杂 agent 可以嵌套 subgraph

**适用场景**：

- **stateful 多步 workflow**（不是简单 ReAct loop）：比如 "调研 → 写初稿 → 审稿 → 改稿 → 发布"
- 需要 **human approval** 的 agent（金融 / 客服 / 合规）
- 需要 **resume from checkpoint** 的长任务（小时级 / 天级）

**学习曲线提醒**：LangGraph 的概念（State / Node / Edge / Conditional / Send / Channel）一上手挺重，**不要在简单需求上用**——简单 ReAct 用 14.2 那 30 行 function calling 即可。

### 2.4 smolagents：极简 + code-as-action

HuggingFace 在 2024 年底推出的 ~1000 行框架，主打两件事：

1. **极简**：核心 abstraction 只有 `Tool / Agent / Memory`，没有 LCEL / Graph 那种重抽象
2. **Code Agent**：让 LLM 输出 **Python code** 作为 action，而非 JSON tool call

`CodeAgent` 让 LLM 直接生成 `result = search("...");  print(result)` 这种 Python，在沙箱里执行，把 stdout 喂回去。**这是 14.4 § "Code-as-Policy" 的最直接实现**——比 JSON function calling 表达力高（能写 loop / conditional / 数据 transform），但需要 sandbox 隔离。

**适用场景**：教学、需要 code-as-action 表达力的任务、不想被 framework 绑架的极简栈。

---

## 3. Multi-agent 三家对比：AutoGen / CrewAI / MetaGPT

### 3.1 设计哲学

- **AutoGen**：通用 conversation framework，agent 之间是平等的"对话方"。核心抽象是 `ConversableAgent`——可以 1v1、可以 group chat、可以 nested chat。**最灵活但要自己设计协作逻辑**
- **CrewAI**：role-based。每个 agent 有 `role / goal / backstory`，组成 `Crew` 执行 `Task`。**最易上手**——3 行代码定义一个研究员 + 一个写手就能跑
- **MetaGPT**：硬编码"软件公司 SOP"——product manager → architect → engineer → QA。**最 opinionated**，但仅适合软件开发流水线

### 3.2 选型经验

- 复杂自定义协作 → **AutoGen**
- 易上手的 role-based 团队 → **CrewAI**
- 软件开发流水线 → **MetaGPT**
- **超过 80% 的所谓 "multi-agent" 任务其实 single agent + tool 就够了**——14.6 已经详讲过 multi-agent 的过度复杂性陷阱

---

## 4. Framework 选型决策树

```
你的需求是什么？
│
├─ 5 分钟跑通 demo + 用 OpenAI 模型
│       └─ OpenAI Assistants API
│
├─ 需要 GUI / browser / OS 操作
│       └─ Anthropic Computer Use（注意仍 beta）
│
├─ 简单单 agent + tool（< 5 个 tool、< 10 step）
│       └─ 14.2 手撕 < 50 行 / smolagents（生产推荐手撕 + native function calling）
│
├─ 复杂 stateful workflow + human approval + checkpoint
│       └─ LangGraph
│
├─ Multi-agent 协作
│       ├─ 通用 → AutoGen
│       ├─ Role-based 易上手 → CrewAI
│       └─ 软件开发 → MetaGPT
│
├─ 需要 code-as-action
│       └─ smolagents CodeAgent
│
├─ 评测 / 安全 / red team
│       └─ Inspect AI（UK AISI 出品，专攻 LLM evaluation）
│
└─ 大规模生产 + 极致控制
        └─ 自己写 thin driver（呼应 14.2）+ 多 framework 混合 wrap
```

**实战经验**：很少有公司"全栈 LangChain"或"全栈 LangGraph"。常见组合是 **手写 driver + OpenAI/Anthropic native function calling + 一两个 framework 工具（trace 用 LangSmith / 评测用 Inspect AI）**。

---

## 5. 最小代码示例：4 段 demo

### 5.1 OpenAI Assistants（30 行）

```python
"""OpenAI Assistants API: 托管式 agent 最小 demo."""
import time
from openai import OpenAI

client = OpenAI()

# 1. 创建 assistant（一次性，可复用）
assistant = client.beta.assistants.create(
    name="Math Tutor",
    instructions="You are a math tutor. Use code interpreter for calculations.",
    model="gpt-4o-mini",
    tools=[{"type": "code_interpreter"}],            # 内置 sandbox
)

# 2. 创建 thread（每个用户会话一个）
thread = client.beta.threads.create()

# 3. 发消息
client.beta.threads.messages.create(
    thread_id=thread.id, role="user",
    content="求解 x^2 - 5x + 6 = 0",
)

# 4. 启动 run + 轮询直到完成
run = client.beta.threads.runs.create(thread_id=thread.id, assistant_id=assistant.id)
while run.status not in ("completed", "failed", "cancelled"):
    time.sleep(1)
    run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)

# 5. 取最新回复
msgs = client.beta.threads.messages.list(thread_id=thread.id, order="desc", limit=1)
print(msgs.data[0].content[0].text.value)
```

亮点是**完全不用自己存 history**——thread.id 就是 state；缺点是 polling 模式 + thread token 一去不回头，长 thread 极贵。

### 5.2 LangGraph state machine 简单 agent（35 行）

```python
"""LangGraph: state machine + tool node 最小 ReAct."""
from typing import Annotated, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

@tool
def get_weather(city: str) -> str:
    """Get current weather of a city."""
    return f"{city}: 22°C cloudy"

# 1. 定义 state schema（messages 用 add_messages reducer 自动累积）
class State(TypedDict):
    messages: Annotated[list, add_messages]

llm = ChatOpenAI(model="gpt-4o-mini").bind_tools([get_weather])

# 2. 定义 node：调用 LLM
def call_llm(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

# 3. 定义 conditional edge：有 tool_call 走 tool 节点，否则结束
def route(state: State):
    return "tools" if state["messages"][-1].tool_calls else END

# 4. 拼图
graph = StateGraph(State)
graph.add_node("agent", call_llm)
graph.add_node("tools", ToolNode([get_weather]))
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", route, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")                 # tool 完了再回 agent

app = graph.compile()
result = app.invoke({"messages": [("user", "北京天气如何？")]})
print(result["messages"][-1].content)
```

**关键点**：`StateGraph` 显式声明 state schema → 加 node → 加 edge（普通 / conditional）→ compile。`ToolNode` 是 prebuilt 的 helper。

### 5.3 smolagents code agent（< 15 行）

```python
"""smolagents CodeAgent: LLM 输出 Python code 作 action."""
from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel

# 一行 model + 一行 tool list + 一行 agent
agent = CodeAgent(
    tools=[DuckDuckGoSearchTool()],
    model=HfApiModel(model_id="meta-llama/Llama-3.3-70B-Instruct"),
)

# run() 内部把任务、tool descriptions 丢给 LLM
# LLM 输出形如：results = duckduckgo_search("...");  print(results[:3])
# smolagents 在沙箱里 exec，把 stdout 拼回 next turn
result = agent.run("2024 年 ACL best paper 是什么？")
print(result)
```

**核心理念**：JSON tool call 表达力受限，让 LLM 直接写 Python code 既能调 tool、又能写 loop / conditional / 后处理。代价是 sandbox 必须严格隔离。

### 5.4 CrewAI 多 role demo（< 25 行）

```python
"""CrewAI: 2 个 role 协作完成研究 + 写作。"""
from crewai import Agent, Task, Crew, Process

researcher = Agent(
    role="Senior AI Researcher",
    goal="Find latest agent RL papers",
    backstory="Reads arxiv every day, expert at distillation.",
    verbose=True,
)
writer = Agent(
    role="Tech Writer",
    goal="Write a 200-word summary for a non-expert.",
    backstory="Translates dense ML papers into clear prose.",
    verbose=True,
)
research_task = Task(
    description="Find 3 most-cited 2025 agent RL papers and list their core contributions.",
    expected_output="A bulleted list of 3 papers.",
    agent=researcher,
)
write_task = Task(
    description="Based on the research, write a 200-word summary.",
    expected_output="200-word plain-English summary.",
    agent=writer,
    context=[research_task],                      # 依赖前一 task 的输出
)
crew = Crew(agents=[researcher, writer], tasks=[research_task, write_task], process=Process.sequential)
print(crew.kickoff())
```

**关键点**：`Agent` = role + goal + backstory；`Task` 用 `context` 串成 DAG；`Crew` 用 `Process.sequential` 或 `Process.hierarchical` 调度。**适合"分工明确"的工作流，但不要硬塞简单需求**。

---

## 6. Anthropic 2024《Building Effective Agents》核心原则

这篇 2024 年 12 月发布的工程白皮书是 2025-2026 年 agent 设计的事实标杆，必读。核心 3 条：

### 6.1 不要过度工程

> "When building applications with LLMs, we recommend finding the simplest solution possible, and only increasing complexity when needed."

实操含义：**先 prompt → 不行再 prompt + tool → 不行再 workflow → 最后才 full agent**。每加一层复杂度都要有明确收益证据。

### 6.2 Workflow vs Agent 的明确区分

- **Workflow**：固定 path，LLM 在预定义流程里执行（chain / route / parallelize / orchestrator-worker / evaluator-optimizer）
- **Agent**：动态决策，LLM 自己决定下一步做什么、用什么 tool、何时停止

**90% 的业务需求其实是 workflow，不是 agent**——agent 的"动态自主"是 bug 来源也是 cost 来源。能用 workflow 解决就别上 agent。

### 6.3 五大 composable patterns

文章提出 5 个可组合的 pattern，每个都是"加 1 层复杂度"的最小增量：

1. **Prompt chaining**：A → B → C 串行（如 outline → draft → polish）
2. **Routing**：classify input → route to specialized prompt（如客服 query → billing/tech/sales）
3. **Parallelization**：N 个 LLM call 并发 → aggregate（如 N 个 reviewer 投票）
4. **Orchestrator-worker**：1 个 orchestrator LLM 拆 task → 派给 N 个 worker LLM
5. **Evaluator-optimizer**：generator + evaluator 循环（如写作 → 评分 → 改写）

**这 5 个 pattern 覆盖 80% 的"agent-like"业务需求**。真正需要"自主决策 + 多步 tool use"的 full agent 反而是少数。

---

## 7. Trace / Debug 工具

Agent 系统的可观测性比普通 LLM call 重要 10 倍——一次 run 几十 step 的 trace 没工具看根本调不了。

| 工具 | 公司 | 强项 |
|---|---|---|
| **LangSmith** | LangChain | LangChain / LangGraph 生态原生集成最深 |
| **Langfuse** | 开源 | 自托管首选，OpenTelemetry 兼容 |
| **Phoenix** | Arize | 评测 + trace 一体，OSS |
| **Helicone** | YC | 简单 proxy 模式，5 分钟接入 |
| **OpenTelemetry GenAI semantic convention** | CNCF | 标准协议（不是工具），未来跨厂商互通 |

**实战经验**：Trace 全开 cost 不低（每 turn 多发一份数据 + 存储费用），生产环境通常 sample 1-10%，全量 trace 只在 debug 模式开。这条与 12.4 LLMOps 衔接。

---

## 8. 工程踩坑与经验

- ❗ **LangChain abstraction 过多 → debug 极难**：早期 LangChain 经常出现"我都不知道这个 prompt 最终是什么样子"的情况。debug 时要么 set `langchain.debug = True` 把所有中间 prompt 打出来，要么直接 trace 走 LangSmith。**这也是 2024 后大家更倾向 LangGraph 的原因——后者把 control flow 显式化了**
- ❗ **OpenAI Assistants thread 长了 token cost 飞涨**：thread 是云端托管，每 run 都把整个 history 喂进去——10 turn 后单次 run 可能烧 50k+ token。要么定期 clean thread / archive 老 message，要么干脆放弃 Assistants 用本地 chat completion 自己控 history
- ❗ **LangGraph 学习曲线陡，不要一上来就用**：State / Node / Edge / Conditional / Send / Channel / Checkpoint 一堆概念，简单 ReAct 上 LangGraph 是杀鸡用牛刀。判断标准——**只有当你需要 "checkpoint resume / human-in-loop / 复杂 conditional flow" 时才上**，否则手撕 30 行
- ❗ **Multi-agent framework 经常过度复杂**：CrewAI / AutoGen 看起来很酷，但 80% 的"multi-agent"任务其实 single agent + tool 就够了。multi-agent 的额外 overhead（更多 LLM call、更难 debug、role 之间通讯歧义）经常吃掉 framework 收益。**先尝试 single agent，确认不够再上 multi-agent**
- ❗ **Framework 升级频繁，必须 pin version**：LangChain / LangGraph / CrewAI 至少每月一次 minor breaking change（API rename、参数顺序变、deprecation）。`requirements.txt` 必须 `==` 锁定版本，升级前在 staging 跑全套回归测试
- ❗ **Anthropic Computer Use 仍 alpha**：2024.10 发布到 2026 年仍标 beta——准确率 OSWorld 上 ~22%（Anthropic 自己 paper 数据），延迟高、screenshot 经常误点。**仅 demo / research，不可上生产**。生产 GUI agent 仍要自己接 Playwright + 视觉模型，或观望 UI-TARS 这类专门训过的开源 GUI agent
- ❗ **Trace 全开 cost 高 → 生产 sample 1-10%**：每 step 一条 trace 数据上报 + 存储不便宜。production 配置：dev / staging 100%、production 1%、出错 trace 100%、按 user_id 抽样保证 trace 完整性
- ❗ **OpenAI Assistants 的 vendor lock 比想象的深**：thread / file / vector store 全在云端，迁移到 Anthropic / Qwen 时所有 state 要重新 import。如果有迁移可能性，从一开始就用 chat completion + 本地 history 自己管，**不要图方便上 Assistants**
- ❗ **Framework 的 trace 不一定准**：LangChain trace 可能漏掉 raw prompt / 漏掉 retry attempt。生产关键链路除了 framework trace，还要自己在 driver 层加一层 prompt + response 全量日志（可异步上报），别信 framework 一面之词
- ❗ **CodeAgent / Code Interpreter 的沙箱必须严格**：smolagents CodeAgent 是 LLM 写 Python code 直接执行——没有沙箱（Docker / gVisor / E2B）就是直接 RCE。OpenAI Code Interpreter 已自带云端沙箱，自建用 E2B / Modal / Daytona

---

## 9. 经典 paper 与资源

- **Anthropic 2024 — Building Effective Agents**（必引）— 2024 年最有影响的 agent 工程白皮书。提出 workflow vs agent 区分、5 大 composable pattern（chaining / routing / parallelization / orchestrator-worker / evaluator-optimizer）、"先简单再复杂"原则。**现代 agent 设计的事实标杆**，读完直接形成"何时该上 framework、何时该手写"的工程直觉
- **Wu et al., 2023 — AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation** — Microsoft 的 multi-agent framework 论文。读 §2 的 ConversableAgent 抽象 + §3 的 group chat 模式，理解通用 multi-agent conversation 的设计哲学
- **Wang et al., 2023 — Voyager: An Open-Ended Embodied Agent with LLMs** — code-as-action 的早期代表（虽是 Minecraft 场景），smolagents CodeAgent 思想的源头之一。理解为什么 "LLM 写 code 当 action" 比 "LLM 出 JSON tool call" 表达力高得多
- 加分：**smolagents 源码** (github.com/huggingface/smolagents) — ~1000 行核心代码，把"最小 agent framework"拆得清清楚楚，对照 14.2 的 150 行手撕版理解 framework 化的代价；**Inspect AI** (github.com/UKGovernmentBEIS/inspect_ai) — UK AISI 出品的 agent 评测框架，model card / safety eval 必备

---

## 10. 自测与面试题

**Q1（选型）**：列出 5 个 agent framework 的核心定位 + 适用场景。

<details>
<summary>Answer sketch</summary>

要覆盖每个 framework 的 "**独特价值**" + "**适用场景**"：

1. **OpenAI Assistants API** — 托管式 thread + run + 内置 tool（File Search / Code Interpreter / Function calling）。适合 5 分钟 demo / hackathon / 不在乎 vendor lock 的内部工具
2. **Anthropic Tool Use + Computer Use** — 协议层（不是 framework），原生 `tool_use` block + Computer Use（screenshot / bash / edit）让 LLM 操作 OS。适合 GUI agent / browser agent，注意 Computer Use 仍 beta
3. **LangGraph** — 显式 state machine（State / Node / Edge），支持 checkpoint / human-in-loop / time travel。适合 stateful 多步 workflow、需要 human approval 的任务、长任务 resume
4. **smolagents** — HuggingFace 极简框架，核心 ~1000 行，主打 CodeAgent（LLM 输出 Python code 作 action）。适合教学、code-as-action 任务、不想被重 framework 绑架的极简栈
5. **CrewAI** — role-based multi-agent，每个 agent 有 role / goal / backstory。适合"分工明确"的团队协作型任务（research + write、PM + dev）。**记得提示**：80% 任务其实 single agent 就够，不要硬塞

加分：能补上 AutoGen（通用 multi-agent conversation）/ MetaGPT（软件开发 SOP）/ Inspect AI（评测）/ LangChain（早期 thin wrapper，现已被 LangGraph 取代主线地位）。

加分：能说出"很少有公司全栈用一个 framework，常见是手写 driver + native function calling + 一两个 framework 工具组合"。

</details>

**Q2（vs）**：手写 < 200 行 ReAct（14.2 风格）vs 用 LangChain / LangGraph，何时选哪个？

<details>
<summary>Answer sketch</summary>

**判断维度**：

| 维度 | 手写 | Framework |
|---|---|---|
| 透明度 / 可控性 | 高（每行你都看得懂） | 低（abstraction 层多） |
| 上手 / 开发速度 | 中（要造轮子） | 高（开箱即用 tool / memory / retriever） |
| Debug 难度 | 低（直接 print） | 高（要会 trace / debug 模式 / 看源码） |
| 生态 / 集成 | 自己写 connector | 数百 integration 现成 |
| 升级风险 | 自己控 | breaking change 频繁 |
| 教学价值 | 高 | 低 |

**选手写**的场景：
- **简单 single agent + ≤ 5 tool**（80% 业务）——手写更省调试时间
- **生产关键路径**——透明可控比"快"更重要，出 bug 能直接看 stack
- **学习 / 面试**——理解原理比会调 API 重要

**选 framework**的场景：
- **复杂 stateful workflow**（多 step / 分支 / 需要 checkpoint） → LangGraph
- **需要 human-in-loop / approval** → LangGraph
- **快速集成现成 tool**（搜索 / DB / RAG / vector store / 数十种 LLM provider） → LangChain / LangGraph
- **multi-agent 协作**（且单 agent 真的不够） → AutoGen / CrewAI

**实战 best practice**：原型用 framework 验证可行性，**生产 fork 出来精简 / 重写**关键路径——这是大厂 agent 团队的常见做法。Anthropic 2024 文章也强调 "use framework as starting point but be willing to drop down to direct API calls"。

加分：能区分 "用 framework 的 trace / observability 工具" vs "用 framework 的 abstraction"——前者（LangSmith / Langfuse）几乎总是值得，后者（chain / agent abstraction）经常是负担。

</details>

**Q3（前沿）**：Anthropic《Building Effective Agents》提倡"不要过度工程"，如何在 simple workflow 与 full agent 之间做选择？

<details>
<summary>Answer sketch</summary>

**Anthropic 的核心原则**：先简单再复杂，每加一层复杂度都要有明确收益证据。

**Workflow vs Agent 的本质区别**：
- **Workflow**：path 固定，LLM 在预定义流程的某些 node 上做局部决策（如 classify、extract、summarize）。**确定性高、token 可控、debug 容易**
- **Agent**：path 动态，LLM 自己决定下一步做什么、用什么 tool、何时停止。**灵活性高，但不确定性、token cost、debug 难度都高**

**决策步骤**（按 Anthropic 推荐顺序）：

1. **先尝试单次 prompt + tool**：能不能一次 LLM call + 几个 tool 解决？能就这样
2. **不行就上 workflow（5 大 pattern）**：
   - 串行流水线 → **prompt chaining**（A → B → C）
   - 分类后路由 → **routing**（classify → branch）
   - 并发处理 → **parallelization**（N 个 LLM 投票 / N 个 task 并发）
   - 任务拆分 → **orchestrator-worker**（1 个 orchestrator 拆 → N 个 worker 执行）
   - 迭代优化 → **evaluator-optimizer**（generator + evaluator 循环）
3. **workflow 都不够才上 full agent**：当且仅当任务路径**真的事先无法预知**时（开放式 research、用户意图模糊、需要长期 trial-and-error）

**"必须 full agent"的 signal**：
- 任务步数事先未知（5 步还是 50 步？取决于环境反馈）
- 需要 dynamic tool selection（什么时候搜索 / 什么时候计算 / 什么时候 ask user 取决于具体内容）
- 长 horizon trial-and-error（如 SWE-bench、coding agent）

**"workflow 就够"的 signal**：
- 流程能画出固定流程图
- 每 step 的输入输出 schema 能枚举
- 业务能容忍"流程外的请求"被拒绝

**反例（典型过度工程）**：
- "用户问问题 → agent 决定调 search 还是直接答" — 这是 routing workflow，不是 agent
- "翻译 + 校对 + 润色" — 这是 prompt chaining，不需要 agent loop
- "客服分类 → billing/tech/sales 不同 prompt" — 这是 routing，不需要 agent

加分：能提到 cost / latency / failure mode 的工程对比——agent 的平均 token 是 workflow 的 5-10 倍，failure mode 也更难 reproduce。能 workflow 就别 agent 是工程层面的省钱省心原则。

加分：能联系到 14.2 的"thin driver" 思路——即使是 full agent，也应该尽量把不变的 path 固化，只在真正需要"动态决策"的少数 node 让 LLM 自由发挥，介于 workflow 和 full agent 之间的 hybrid 形态在生产上最常见。

</details>

---

## 11. 延伸阅读

- [Anthropic — Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents) — 必读。2024 年最被引用的 agent 工程文章，5 大 composable pattern + workflow/agent 区分原则
- [LangGraph 官方文档](https://langchain-ai.github.io/langgraph/) — State / Node / Edge / Checkpoint / Human-in-loop 完整 tutorial，看官方示例理解为什么 LangGraph 是 LangChain 团队的"agent OS"方向
- [OpenAI Assistants API Overview](https://platform.openai.com/docs/assistants/overview) — Assistant / Thread / Message / Run 四对象模型 + 内置 tool（Code Interpreter / File Search / Function calling）使用指南
- [Anthropic Computer Use](https://docs.anthropic.com/en/docs/build-with-claude/computer-use) — Computer Use 协议 + reference implementation，跑一个 Docker 容器就能体验 Claude 操作 GUI
- [smolagents 源码](https://github.com/huggingface/smolagents) — 极简 agent framework，~1000 行核心代码，对照 14.2 的 150 行手撕版理解 framework 化的代价
- [Inspect AI](https://inspect.ai-safety-institute.org.uk/) — UK AISI 出品的 LLM 评测框架，做 agent safety eval / capability eval 时优先用
- [LangSmith](https://smith.langchain.com/) / [Langfuse](https://langfuse.com/) / [Phoenix](https://phoenix.arize.com/) — agent trace 工具三选一，呼应 12.4 LLMOps
- [AutoGen 官方文档](https://microsoft.github.io/autogen/) / [CrewAI 文档](https://docs.crewai.com/) / [MetaGPT 论文](https://arxiv.org/abs/2308.00352) — multi-agent 三家深入读
- 推荐继续读本教程的 **Module 15 Agent RL**——本节聚焦"用现成 framework 拼 agent"，Module 15 聚焦"怎么把 agent 能力训进模型"（Agent SFT、多轮 PPO/GRPO、真实环境 RL、Agent 鲁棒性等），是 2025-2026 前沿方向
