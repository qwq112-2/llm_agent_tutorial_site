---
title: "14.6 Multi-agent：CAMEL / AutoGen / MetaGPT / orchestrator-worker"
description: "Multi-agent 把\"一个 agent 干完所有事\"拆成多个 agent 各司其职 + 互相通信——这一节梳理 5 大协作范式（role-play / conversable / SOP / waterfall / orchestrator-worker / debate）、四个代表框架（CAMEL / AutoGen / MetaGPT / ChatDev），并诚实交代一个反直觉的结论：m"
---

> ⏱ 预计阅读 40 分钟 ｜ 难度 ★★★ ｜ 前置：14.1（Agent 范式概览）

## 一句话本节讲什么

Multi-agent 把"一个 agent 干完所有事"拆成**多个 agent 各司其职 + 互相通信**——这一节梳理 5 大协作范式（role-play / conversable / SOP / waterfall / orchestrator-worker / debate）、四个代表框架（CAMEL / AutoGen / MetaGPT / ChatDev），并诚实交代一个反直觉的结论：**multi-agent 不一定打得过 single agent + 好 tool**。

---

## 1. Mental model（直觉）

### 1.1 为什么要"多个 agent"

Single agent 已经能 ReAct + 调 tool（14.1、14.2），那为什么还要 multi-agent？四个动机：

1. **任务分解**：复杂任务拆成可独立执行的子任务，每个 agent 专精一类（搜索 / 写代码 / review）
2. **角色化**：模仿人类协作——CEO 拍板、PM 写需求、Engineer 实现、QA 测试，每个 role 有自己的 prompt + history
3. **可控性 + 可解释**：每个 agent 独立 context，**便于调试和审计**——而不是把所有东西塞进一个 50k token 的 system prompt
4. **专精模型混搭**：不同 agent 可以用不同 model（写代码用 Claude，搜索用 GPT，review 用便宜模型），成本和能力 trade-off

### 1.2 三种协作拓扑

把 N 个 agent 摆开，通信结构主要有三种：

```
(a) Pair / Role-play (CAMEL)
   ┌────────┐         ┌──────────┐
   │  User  │ ◄────► │ Assistant │
   │ Agent  │         │  Agent   │
   └────────┘         └──────────┘
         双向对话，无外部协调者

(b) Group chat / Free-form (AutoGen)
              ┌─────────┐
       ┌─────►│ Agent A │◄─────┐
       │      └─────────┘      │
       │            ▲          │
       │            │          │
   ┌───┴───┐  ┌─────┴────┐  ┌──┴────┐
   │ Agent │◄─┤ Manager  ├─►│ Agent │
   │   B   │  │ (router) │  │   C   │
   └───────┘  └──────────┘  └───────┘
        所有 agent 共享一个频道，manager 决定发言顺序

(c) Orchestrator-Worker (现代主流)
                ┌──────────────┐
                │ Orchestrator │ ◄── 唯一决策者
                │  (planner)   │
                └──────┬───────┘
              ┌────────┼────────┐
              ▼        ▼        ▼
         ┌────────┐┌────────┐┌────────┐
         │Worker 1││Worker 2││Worker 3│
         │ search ││  code  ││  write │
         └────────┘└────────┘└────────┘
        worker 不互相通信，只回 orchestrator
```

简单 task 用 (a)；探索性 task 用 (b)；生产 agent 几乎都用 (c)——orchestrator-worker 是 2024 年后**事实主流**。

### 1.3 一个反直觉的事实

学完 multi-agent，第一直觉是"分工合作 = 更强"。但 2024-2026 多个公开 benchmark（SWE-bench、GAIA、τ-bench）反复验证：**强 single agent + 丰富 tool，常常打得过精心设计的 multi-agent 系统**。原因留到第 6 节展开。先把这个结论记住——它会救你不少 over-engineering。

---

## 2. Multi-agent 的 5 大协作范式

按"通信结构 + 决策方式"切，主流 5 种范式：

| 范式 | 代表 | 通信结构 | 适用任务 |
|---|---|---|---|
| **Role-play** | CAMEL | 双 agent 对话 | 数据合成、对话仿真 |
| **Conversable / Group chat** | AutoGen | N agent 自由通信 | 探索性、人机混合 |
| **SOP-driven** | MetaGPT | 标准化 artifact 传递 | 流程化任务（软件开发） |
| **Waterfall** | ChatDev | 阶段性流水 | 顺序明确的工程项目 |
| **Orchestrator-Worker** | OpenAI Swarm / Anthropic / HuggingGPT | 一对多、星型 | 生产级通用 agent |
| **Debate / Critic** | Du 2023 | N agent 辩论 + judge | reasoning / factuality |

注意：这 6 种不是互斥分类，而是"原型"——真实系统经常混合。例如 Devin 内部既有 orchestrator-worker，也有 critic 做 self-review。

---

## 3. 四个代表框架

### 3.1 CAMEL（2023, Li et al.）—— Role-play 起点

**核心机制**：两个 agent，一个扮演 **user role**（提需求、纠偏），一个扮演 **assistant role**（执行、回答）。给定一个**高层 task**（例 "Develop a stock trading bot"），用 task specifier 自动展开成具体 instruction，然后两个 agent 自动对话直到任务完成。

**关键贡献**：

- 第一个把"role-playing prompt"工程化的 framework
- 提出 **inception prompting**：用元 prompt 让两个 agent 自动 stay in role（防止 role flip / "我也是 AI"穿帮）
- 副产品：**大规模合成对话数据**——CAMEL 后来主要被引用为 synthetic data 生成方案

**适用场景**：

- 合成 role-play / instruction-following 数据（SFT 训练用）
- 仿真用户行为（评测、压测 agent）
- **不适合**生产任务——双 agent 容易"越聊越远"，缺乏 hard constraint

### 3.2 AutoGen（2023, Wu et al., Microsoft）—— Conversable agent 标准

AutoGen 把所有 agent 抽象成 **ConversableAgent**：每个 agent 有 `send` / `receive` / `generate_reply` 三个方法，agent 之间通过**消息**通信。框架本身不规定通信拓扑，你可以：

- **Two-agent**：assistant + user_proxy（user_proxy 可执行 code、做 human-in-the-loop）
- **Group chat**：多 agent 共享一个频道，由 `GroupChatManager` 决定下一个发言者（round-robin / LLM-routed / custom）
- **Nested chat**：一个 agent 内部触发另一组 agent 的子对话

**关键贡献**：

- **统一抽象**：把 single-agent / multi-agent / human-in-loop 收敛到同一组接口
- **Code execution as first-class**：`UserProxyAgent` 默认能执行 LLM 生成的 code，是 code-as-action 的早期工业实现
- **Group chat manager**：把"谁该发言"显式化，比 free-form 更可控

AutoGen 是 2024 年事实上的 multi-agent 框架龙头。后续 AutoGen 0.4+ 重构了底层（actor model + async），但核心抽象未变。

### 3.3 MetaGPT（2023, Hong et al.）—— SOP-driven 软件公司

MetaGPT 的核心 insight：**人类组织的高效，来源于 SOP（Standard Operating Procedure）+ 标准化 artifact**，而不是自由聊天。它把一个 agent 团队组织成一个**虚拟软件公司**：

```
ProductManager  ──► PRD.md
       │
       ▼
   Architect    ──► design.md + system diagram
       │
       ▼
ProjectManager  ──► task_list.json
       │
       ▼
    Engineer    ──► source code (multiple files)
       │
       ▼
       QA       ──► test code + bug report
```

每个角色（role）有：
- **固定的 LLM prompt**（"你是 Architect，输出 design doc"）
- **明确的输入 artifact**（上一阶段的 doc）
- **明确的输出 artifact**（标准化结构）
- **publish-subscribe 消息总线**（Role 关注哪些 artifact 出现就被触发）

**关键贡献**：

- 把 **SOP** 作为 multi-agent 协作的核心约束——比 free-form group chat 稳定得多
- **结构化 artifact** 比自然语言消息更适合 LLM 生成（PRD / design 都有 schema）
- 在 HumanEval / MBPP 等代码任务上证明 SOP-driven 显著优于 free-form multi-agent

**适用场景**：流程清晰、artifact 标准化的任务（软件开发、报告生成、合规审核）。**不适合**探索性任务。

### 3.4 ChatDev（2023, Qian et al.）—— Waterfall 软件开发

ChatDev 比 MetaGPT 更"瀑布"——明确把软件开发分成 **design → coding → testing → documenting** 四个阶段，每个阶段是一组 agent 的**双人对话**（CEO ↔ CTO、Programmer ↔ Reviewer 等）。比 MetaGPT 更轻量（更少 role、更线性流程），但思想一脉相承：**用 SOP 约束 multi-agent 行为**。

历史定位：ChatDev 与 MetaGPT 是 2023 年下半年 multi-agent SOP 路线的两个代表，理论价值大于工程价值——真正的生产级 coding agent（Devin / SWE-agent / Claude Code）后来都走了 orchestrator-worker 路线。

### 3.5 Orchestrator-Worker（现代主流，2024+）

**核心结构**：

- 一个**强 LLM 当 orchestrator**：负责 plan、决定下一步、决定调哪个 worker、整合 worker 结果
- N 个 **worker** 各司其职：search worker、code worker、write worker、verify worker……
- worker **不互相通信**——只接收 orchestrator 的指令，把结果回给 orchestrator

```
        ┌───────────────────────────────────┐
        │            Orchestrator            │
        │  (plan + dispatch + synthesize)   │
        └────┬───────────┬──────────┬───────┘
   step 1 ──►│           │          │◄── final answer
             ▼           ▼          ▼
         ┌──────┐    ┌──────┐   ┌──────┐
         │search│    │ code │   │write │
         │worker│    │worker│   │worker│
         └──────┘    └──────┘   └──────┘
              step result 回流给 orchestrator
```

代表实现：

- **OpenAI Swarm**（2024 实验框架）+ Agents SDK（2025 GA）
- **Anthropic Building Effective Agents** 蓝图：明确推荐 orchestrator-worker 作为 default pattern
- **HuggingGPT (Shen 2023)**：早期 orchestrator-worker 雏形，LLM 调度 HuggingFace 上的 expert model
- **Devin / OpenDevin / SWE-agent**：内部都是 orchestrator-worker 拓扑

为什么这一范式胜出？

1. **可控**：所有决策走 orchestrator，单一 source of truth，便于 debug 与监控
2. **可观测**：worker 接口标准化，每次调用可记录、可重放
3. **可扩展**：加新能力 = 加新 worker，不用动 orchestrator 主逻辑
4. **避免 chaos**：free-form group chat 容易出现 agent 互相"礼貌等对方"或"互相反对"，orchestrator-worker 没有这个问题

**典型工程模板**：orchestrator 用强 model（Claude Sonnet / GPT-5）做 reasoning，worker 用便宜 model 或 specialized fine-tuned 小模型——成本最优。

### 3.6 Debate / Critic 范式

**Du et al. 2023 《Improving Factuality and Reasoning in LMs through Multiagent Debate》**：N 个 agent 各自给出答案，互相看对方答案后**重新作答**，迭代 K 轮，最后投票或由 judge 选最优答案。

实证结果：在 reasoning 任务（GSM8K、MMLU 子集）上，debate 比 single CoT + self-consistency 提升 2-5%。本质是 **LLM-as-judge 的 multi-agent 化** + **diverse generation 提高 cover rate**。

但 debate 有两个臭名昭著的问题（详见踩坑章节）：

- **Echo chamber**：agent 倾向于附和别人，越辩越同意，失去 diversity
- **Token 成本爆炸**：N agent × K 轮，token 是 single agent 的 N×K 倍

**适用场景**：高 stake 的 factuality / reasoning 任务，且预算允许。**不适合**生产高频路径。

---

## 4. 最小代码示例

### 4.1 CAMEL：双 agent role-play

```python
from openai import OpenAI

client = OpenAI()
TASK = "Design a 3-day Tokyo trip itinerary for a family with kids."

USER_SYS = f"You are a CURIOUS USER planning: '{TASK}'. Ask the assistant ONE concrete question per turn. After 3 turns, say 'TASK_DONE'."
ASSIST_SYS = f"You are an EXPERT TRAVEL AGENT helping with: '{TASK}'. Answer ONE question per turn, concise."

def chat(sys, history):
    return client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": sys}, *history],
    ).choices[0].message.content

user_hist, assist_hist = [], []
msg = "Let's start. What's the first thing I should know?"
for turn in range(4):
    print(f"\n--- Turn {turn} | USER ---\n{msg}")
    if "TASK_DONE" in msg: break
    assist_hist.append({"role": "user", "content": msg})
    reply = chat(ASSIST_SYS, assist_hist)
    assist_hist.append({"role": "assistant", "content": reply})
    print(f"\n--- Turn {turn} | ASSISTANT ---\n{reply}")
    user_hist.append({"role": "user", "content": reply})
    msg = chat(USER_SYS, user_hist)
    user_hist.append({"role": "assistant", "content": msg})
```

关键点：两个 agent 的 history 是**独立维护**的——assistant 看不到 user 内部的"思考"，反之亦然。这是 role-play 的基本要求，否则 role 会塌陷。

### 4.2 AutoGen group chat：3 agent 协作

```python
# pip install pyautogen
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

cfg = {"config_list": [{"model": "gpt-4o-mini", "api_key": "<your-key>"}]}

planner = AssistantAgent(name="Planner", llm_config=cfg,
    system_message="You break tasks into steps. After plan accepted, say 'PLAN_DONE'.")
coder = AssistantAgent(name="Coder", llm_config=cfg,
    system_message="You write Python code based on plan steps. Output ONLY one ```python block.")
critic = AssistantAgent(name="Critic", llm_config=cfg,
    system_message="You review code for bugs/edge cases. If OK, reply 'APPROVED'; else point out issues.")
user = UserProxyAgent(name="User", human_input_mode="NEVER", code_execution_config=False,
    is_termination_msg=lambda m: "APPROVED" in m.get("content", ""))

groupchat = GroupChat(agents=[user, planner, coder, critic], messages=[], max_round=8)
manager = GroupChatManager(groupchat=groupchat, llm_config=cfg)
user.initiate_chat(manager, message="Write a Python function to compute n-th Fibonacci with memoization.")
```

`GroupChatManager` 内部用一个 LLM call 决定**下一个发言者**（基于 system prompt 和当前对话）。`is_termination_msg` 是循环退出条件——multi-agent 必须有，否则 max_round 用完才能停。

### 4.3 Orchestrator-Worker：planner 决定 worker

```python
from openai import OpenAI
import json

client = OpenAI()

WORKERS = {
    "search": "You search the web. Given a query, return 1-3 relevant facts.",
    "calc":   "You do arithmetic. Given an expression, return the numeric result.",
    "write":  "You compose final answer. Given findings, return a short paragraph.",
}

def call(sys, msg, model="gpt-4o-mini"):
    return client.chat.completions.create(
        model=model, messages=[{"role": "system", "content": sys}, {"role": "user", "content": msg}],
    ).choices[0].message.content

ORCH_SYS = """You are an orchestrator. Available workers: search, calc, write.
Reply STRICT JSON: {"worker": "...", "input": "...", "done": false}
When ready to output answer, set "worker": "write", "done": true."""

def orchestrate(task, max_steps=6):
    findings = []
    for step in range(max_steps):
        ctx = f"Task: {task}\nFindings so far: {findings}\nWhat next?"
        plan = json.loads(call(ORCH_SYS, ctx, model="gpt-4o"))  # strong model for orchestrator
        worker, inp, done = plan["worker"], plan["input"], plan["done"]
        result = call(WORKERS[worker], inp)  # cheap model for worker
        print(f"[{step}] orch->{worker}({inp[:40]}...) => {result[:60]}...")
        findings.append({worker: result})
        if done: return result
    return "MAX_STEPS_EXCEEDED"

print(orchestrate("How many seconds are in a leap year (366 days)?"))
```

关键点：**orchestrator 用强 model**（gpt-4o）保证决策准确，**worker 用便宜 model**（gpt-4o-mini）控成本——这是 orchestrator-worker 范式的"标配 trick"，省钱又不掉点。

### 4.4 Multi-agent debate：3 agent 辩论 + judge

```python
from openai import OpenAI

client = OpenAI()
QUESTION = "If a train travels 60 km/h for 2.5 hours, how far does it go? Show reasoning."

def ask(sys, msg):
    return client.chat.completions.create(model="gpt-4o-mini",
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": msg}],
    ).choices[0].message.content

# Round 1: 3 agents independently answer
roles = ["You are a careful mathematician.",
         "You are a physics teacher who double-checks units.",
         "You are a skeptic who questions assumptions."]
answers = [ask(r, QUESTION) for r in roles]

# Round 2: each agent sees others' answers and revises
revised = []
for i, role in enumerate(roles):
    others = "\n\n".join(f"Agent {j}: {a}" for j, a in enumerate(answers) if j != i)
    revised.append(ask(role, f"{QUESTION}\n\nOther agents said:\n{others}\n\nRevise your answer."))

# Judge picks best
judge = ask("You are an impartial judge. Pick the most correct answer; quote it.",
            f"Question: {QUESTION}\n\n" + "\n\n".join(f"A{i}: {a}" for i, a in enumerate(revised)))
print("FINAL:", judge)
```

关键点：每个 agent 用**不同 role prompt** 强行制造 diversity——否则 3 个 agent 会输出几乎相同的答案，debate 退化为 self-consistency。

---

## 5. 工程踩坑与经验

- ❗ **Multi-agent 的 token cost 是 single agent 的 N 倍甚至 N² 倍**——一次 group chat 每轮所有 agent 都要看完整 history，N agent × K 轮 ≈ O(N²K) token。生产前**必算成本**，并对比 single agent 基线。否则上线第二天财务找你
- ❗ **Agent 间 message protocol 必须明确**——是 free-text？JSON？特定 schema？没有约定就 chaos：agent A 输出 markdown，agent B 期待 JSON，下游 parse 全炸。MetaGPT 的成功一半归功于**强制 artifact schema**
- ❗ **Debate 容易变 echo chamber**——agent 倾向于附和别人。对策：(1) 用强对立的 role prompt（"you are skeptic"）；(2) 控制信息流（agent A 看不到 B 的 reasoning，只看结论）；(3) 加 **"you must disagree at least once"** 这种 hard constraint
- ❗ **Orchestrator 太弱 → worker 选错 → 全盘崩**——orchestrator 是单点决策，弱模型当 orchestrator 等于"指挥官 IQ 不够"。规则：**orchestrator 用 frontier model（Claude Sonnet+/GPT-5+），worker 可用便宜模型**。绝对不要为省钱让小模型当 orchestrator
- ❗ **Multi-agent benchmark 不成熟**——SWE-bench / GAIA 等主流 benchmark 都是 task-level，不专门评 multi-agent。MAgIC、MultiAgentBench 是新尝试但 leaderboard 稀疏，paper 里"我们的 multi-agent 涨了 X%"经常 cherry-pick，看时要谨慎
- ❗ **简单任务不要上 multi-agent**——SWE-bench 上 Anthropic Claude Code 用 single agent 长期占榜首；OpenAI o3 在 GAIA 上 single agent 也极强。**先跑 single agent baseline，确认天花板再决定要不要 multi**。多数情况下 multi-agent 是 over-engineering
- ❗ **Agent 间 history 共享策略要选好**——三种典型：(1) **全共享**（每个 agent 看完整历史）：context 爆炸；(2) **摘要共享**（manager 出摘要）：信息丢失；(3) **私有 + handoff**（agent 切换时只交接相关字段）：最稳定但 protocol 设计累。生产推荐 (3)
- ❗ **Worker 失败必须有显式 error handling**——orchestrator-worker 中 worker 失败若被 silently swallow，orchestrator 会基于"空结果"继续 plan，全错。统一约定：worker 返回 `{ok: bool, result?, error?}`，orchestrator 必须显式处理 `ok=False`
- ❗ **不同 agent 用不同 model 时注意 tokenizer / format 差异**——Claude 的 JSON mode、OpenAI 的 tool call、Gemini 的 schema，输出格式不通用。混合调度时在 driver 层做 format 转换，否则下游 agent parse 失败
- ❗ **Multi-agent 的"成功率"难归因**——任务失败时是 orchestrator 错？某个 worker 错？还是 protocol 错？必须**每个 agent 单独埋点**（input、output、latency、cost），否则线上故障无法定位

---

## 6. 关键判断：Multi-agent 真的更强吗？

这一节回答开头那个反直觉结论。

### 6.1 实证结果：常常**不更强**

2024-2026 多个公开 benchmark 的实证：

- **SWE-bench Verified**：Anthropic Claude Code（**single agent**）长期占据榜首（截至 2026 中），打过多个精心设计的 multi-agent 系统
- **GAIA**：OpenAI Deep Research（single reasoning agent + tool）显著超过 multi-agent baseline
- **τ-bench**：single agent + 强 tool use 训练，效果优于 customer-service-multi-agent
- **HumanEval / MBPP**：MetaGPT 当年 paper 数字漂亮，但用 GPT-4o single + good prompt 后差距大幅缩小

为什么？三个原因：

1. **Frontier model 的 reasoning ability 已经够强**——能自己做 task decomposition、self-correction，不需要外部 role 拆分
2. **Multi-agent 引入额外失败模式**：communication 损耗、role 误解、消息序列化反序列化错误、agent 间互相误导
3. **Orchestration overhead 大**：每次"切换 agent"都要重新 prefill context，成本和延迟双倍

### 6.2 那 Multi-agent 的价值在哪

不是说 multi-agent 没用，而是它的价值不在"性能上限"，在以下几方面：

1. **可控性 + 可解释**：每 agent 独立 context，便于 debug、audit、单点替换
2. **专精化**：worker 可以 fine-tune 成 domain expert，比通用大模型更准
3. **成本优化**：orchestrator 用大模型、worker 用小模型，整体 cost 降一个量级
4. **并行化**：真正可并行的 sub-task（多个 search、多个 file 编辑）走 multi-worker 加速
5. **safety / sandbox**：高风险 action（写文件、调外部 API）由 dedicated worker 执行，便于权限隔离
6. **synthetic data**：CAMEL 风格的双 agent 对话，是合成 SFT / RLHF 数据的重要 pipeline

### 6.3 实战决策树

```
任务是否能明确拆成可并行 / 异构专精的 sub-task？
├─ 否 ──► single agent + 丰富 tool
└─ 是 ──► 子任务之间是否有强顺序依赖？
         ├─ 否 ──► orchestrator-worker（并行 worker）
         └─ 是 ──► 流程是否标准化（artifact schema 清晰）？
                  ├─ 是 ──► SOP-driven（MetaGPT 风格）
                  └─ 否 ──► orchestrator-worker（顺序调度）
```

99% 的生产 agent 落在 "single + tool" 或 "orchestrator-worker" 两个分支。CAMEL / debate / group chat 主要用于研究、数据合成、特定 reasoning 任务。

### 6.4 2025-2026 趋势

1. **Reasoning model 内化 multi-agent**：o3 / Claude 4.x / GPT-5 / Qwen3 / GLM-Z1 一次 forward 的 long-CoT 已能完成简单"多角色思考"——不需要外部 multi-agent 框架就能 mimic critic + planner 行为
2. **MCP（Model Context Protocol）** 成为 tool 标准接口，**A2A（Agent-to-Agent）** 协议在路上——agent 间通信开始有 W3C 级标准
3. **Coding agent 全面收敛到 orchestrator-worker**：Devin / OpenDevin / SWE-agent / Claude Code 内部架构高度相似——main loop（orchestrator）+ specialized sub-agent（file editor / runner / planner）
4. **Multi-agent benchmark 起步**：MAgIC、MultiAgentBench、AgentVerse 子集开始评测协作能力，但远未到 SWE-bench 那种成熟度

---

## 7. 经典 paper

- **Li et al., 2023 — CAMEL: Communicative Agents for "Mind" Exploration of Large Language Model Society** — 必读。Role-play multi-agent 的开山作。读 §3 的 inception prompting 设计，理解为什么"角色固定"是 role-play 不塌陷的关键。也是合成对话数据的经典 pipeline 起点
- **Wu et al., 2023 — AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation** — 必读。把 multi-agent 收敛到 ConversableAgent 抽象，2024 年事实标准。读 §3 的 framework 设计 + §4 的 group chat / nested chat 案例，理解通用 multi-agent framework 的 API 形态
- **Hong et al., 2023 — MetaGPT: Meta Programming for Multi-Agent Collaborative Framework** — 必读。SOP-driven 路线代表，用"软件公司"类比把 multi-agent 工程化。读 §3 的 role + SOP + 共享消息池设计，体会"为什么 free-form group chat 不如 SOP 稳定"
- 加分：**Qian et al., 2023 — ChatDev**（waterfall 软件开发的轻量版）；**Du et al., 2023 — Improving Factuality and Reasoning in LMs through Multiagent Debate**（debate 范式）；**Anthropic 2024 — Building Effective Agents**（工程视角，明确推荐 orchestrator-worker 作为 default pattern，强调 simplicity over complexity）

---

## 8. 自测与面试题

**Q1（范式）**：列出 multi-agent 的 5 大协作范式，每个给一个代表框架。

<details>
<summary>Answer sketch</summary>

至少答出 5 个：

1. **Role-play**：双 agent 扮演角色对话——代表 **CAMEL**（Li 2023），副产品是合成对话数据
2. **Conversable / Group chat**：N agent 自由通信，由 manager 决定顺序——代表 **AutoGen**（Wu 2023, Microsoft），2024 事实标准
3. **SOP-driven**：标准化 artifact + 固定 role + publish-subscribe 消息总线——代表 **MetaGPT**（Hong 2023），模拟软件公司
4. **Waterfall**：明确阶段性流水（design → coding → testing → doc）——代表 **ChatDev**（Qian 2023），MetaGPT 的轻量版
5. **Orchestrator-Worker**：一个 strong planner + N specialized executor，星型拓扑——代表 **OpenAI Swarm / Anthropic Building Effective Agents / HuggingGPT**，2024+ 现代主流
6. **Debate / Critic**：N agent 各自答 → 互相看答案 → 重新答 → judge 选优——代表 **Du 2023 Multi-agent Debate**，提升 factuality

加分：能指出"orchestrator-worker 是生产首选，其余偏研究 / 数据合成 / 特定任务"，并能讲出每种范式的通信结构（pair / star / chain / mesh）

</details>

**Q2（实战）**：你要做一个软件开发 agent，会选 MetaGPT 还是 orchestrator-worker？理由？

<details>
<summary>Answer sketch</summary>

**结论**：生产首选 **orchestrator-worker**（除非项目高度流程化）。

**理由**（至少 4 点）：

1. **Adaptivity**：真实代码任务有大量探索（read repo、grep、debug），SOP 写不死。MetaGPT 的固定 role + artifact schema 适合"从零写一个新项目"，但对"在已有 100k 行 repo 改 bug"力不从心。SWE-bench 的真实分布更接近后者
2. **可扩展**：orchestrator-worker 加新能力 = 加新 worker（file edit / shell / search / lint），不动主逻辑；MetaGPT 加新 role 要改 SOP 流程
3. **业界先例**：Devin / OpenDevin / SWE-agent / Claude Code 全是 orchestrator-worker 拓扑，SWE-bench Verified 上 Claude Code（single agent）长期占榜首，证明这一架构的天花板高
4. **Token 成本**：MetaGPT 每个 role 都要看完整 PRD + design，context 重复；orchestrator-worker 中 worker 只看任务相关 input，节省 token
5. **可观测**：所有决策走 orchestrator，单一 source of truth，便于 debug 与监控

**MetaGPT 适合的场景**：从零开始写 well-defined 小项目（demo / 内部工具）、教学示例、流程标准化的报告生成。**不适合**真实 repo 维护、debug 任务、long-tail SWE 工作。

加分：能指出"实际生产中往往混合"——orchestrator-worker 主架构 + 特定环节（如 code review）借用 critic / debate；并能提到 MCP 让 worker 接入标准化 tool 是趋势

</details>

**Q3（trade-off）**：实证上 multi-agent 不一定打过 single agent + good tool，那它的价值在哪？

<details>
<summary>Answer sketch</summary>

先**承认这一事实**——SWE-bench / GAIA / τ-bench 等多 benchmark 上，强 single agent + 丰富 tool 常常打过 multi-agent。

**Multi-agent 的真实价值**（不在性能上限，在以下方面）：

1. **可控性 + 可解释**：每 agent 独立 context、独立 prompt，便于 debug、audit、灰度替换。一个 agent 出 bug 只换它，不动整体
2. **专精化**：worker 可 fine-tune 成 domain expert（如专门 SQL agent、专门 LaTeX agent），比通用大模型更准
3. **成本优化**：orchestrator 用 frontier model（强 reasoning），worker 用小模型 / specialized model，整体 cost 可降 5-10×
4. **真正可并行的任务加速**：多个独立 search / 多个 file 同时编辑，multi-worker 直接缩 wall-clock time
5. **Safety / sandbox**：高风险 action（写文件、调金融 API）由 dedicated worker 执行，便于权限隔离与审计
6. **Synthetic data 生成**：CAMEL 风格双 agent 对话是合成 SFT / RLHF 数据的重要 pipeline——这是 multi-agent 最稳定的工业应用
7. **Human-in-the-loop**：AutoGen 的 UserProxyAgent 让人类自然加入 multi-agent 流程，single agent 难以同等优雅地表达

**反过来 single + tool 的优势**：

- 上限高（frontier reasoning model 自己能做 task decomposition）
- 部署简单（一个 model + 一组 tool）
- 成本低（无 inter-agent communication overhead）
- 失败模式少（无 protocol error / role 误解）

**实战原则**：默认 single + 丰富 tool；只在任务有**明确并行 / 异构专精 / 流程标准化** 需求时才上 multi-agent，且优先选 orchestrator-worker 拓扑。

加分：能指出 2025-2026 趋势——reasoning model 内化"多角色思考"，从外部 multi-agent 框架向"模型本身就是 multi-role thinker"演进，且 MCP / A2A 等 agent 协议正在标准化跨 agent 通信

</details>

---

## 9. 延伸阅读

- [Anthropic — Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents) — 2024 年最有影响力的 agent design 文章，明确推荐 orchestrator-worker 为 default，反复强调 simplicity over complexity，必读
- [AutoGen GitHub & Docs](https://microsoft.github.io/autogen/) — 主流 multi-agent 框架的官方文档与样例，0.4+ 版本重构后的 actor model 值得读
- [MetaGPT GitHub](https://github.com/geekan/MetaGPT) — SOP-driven 路线代表，看 `metagpt/roles/` 下的 role 定义如何用 prompt + artifact schema 实现"虚拟软件公司"
- [CAMEL GitHub](https://github.com/camel-ai/camel) — Role-play multi-agent 与合成数据 pipeline 的开源实现，CAMEL 团队后续做了 OASIS（agent society 仿真）值得跟进
- [OpenAI Swarm / Agents SDK](https://github.com/openai/swarm) — OpenAI 的轻量级 orchestrator-worker 实验框架，看 `handoff` 抽象如何工程化 agent 切换
- [Du et al. 2023 — Multi-agent Debate paper](https://arxiv.org/abs/2305.14325) — Debate 范式的奠基论文
- [MAgIC Benchmark](https://github.com/cathyxl/MAgIC) — 早期 multi-agent collaboration benchmark，了解评测方法
- 推荐继续读本教程的 **14.7 节《Agent Framework 对比：LangGraph / OpenAI Assistants / Anthropic》**——本节关注协作范式与代表框架，14.7 关注当下主流 single-agent / multi-agent 通用 framework 的工程对比与选型
