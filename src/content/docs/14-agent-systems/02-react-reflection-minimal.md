---
title: "14.2 实现一个最小 ReAct + Reflection agent"
description: "把 14.1 的 ReAct 心智模型落到 < 200 行可跑代码——手撕 prompt template / parser / tool registry / loop controller / reflection memory 五个组件，理解为什么\"用现成 LLM 跑 agent\"的瓶颈不在 LLM 而在外层这套确定性管道，并把同一个 demo 用 OpenAI function calli"
---

> ⏱ 预计阅读 40 分钟 ｜ 难度 ★★ ｜ 前置：14.1（Agent 范式概览）、13.4（Function calling）｜🔥 必考

## 一句话本节讲什么

把 14.1 的 ReAct 心智模型落到 **< 200 行可跑代码**——手撕 prompt template / parser / tool registry / loop controller / reflection memory 五个组件，理解为什么"用现成 LLM 跑 agent"的瓶颈不在 LLM 而在外层这套**确定性管道**，并把同一个 demo 用 OpenAI function calling 重写成 30 行的"现代版"做对比。

---

## 1. Mental model（直觉）

### 1.1 Agent runtime 的三层视角

读 framework 源码（smolagents / LangGraph / OpenAI Assistants）会发现，所有 ReAct agent 的实现都可以拆成三层：

```
┌──────────────────────────────────────────────┐
│  Layer 3: 策略层（Reflection / Memory）       │
│    - 失败后写反思、下次重试时注入 hint        │
│    - 这一层是"软"的，决定 agent 学习能力      │
├──────────────────────────────────────────────┤
│  Layer 2: 控制层（Loop / Parser / Executor）  │
│    - while step < max_step:                  │
│        out = llm(prompt + history)           │
│        thought, action = parse(out)          │
│        obs = execute(action)                 │
│        history.append(...)                   │
│    - 这一层是"硬"的，决定 agent 是否能跑通    │
├──────────────────────────────────────────────┤
│  Layer 1: 接口层（Tool registry / Schema）    │
│    - tool name → callable + 参数描述         │
│    - 这一层是"接口契约"，决定能力边界          │
└──────────────────────────────────────────────┘
```

读者写第一个 agent 最容易踩的坑是**把 LLM 当万能的 controller**——以为只要 prompt 给得好就能跑，结果发现 90% 的 bug 都出在 Layer 2：parser 抓不到 `Action:`、tool 抛 exception 没 capture、history 越来越长撑爆 context。**Agent 工程的核心难度不在 prompt，而在外层的确定性管道**。

### 1.2 ReAct loop 的 5 行伪代码

把 14.1 §3.1 的 4 步循环写成可执行的伪代码：

```python
history = []
for step in range(max_step):
    prompt = build_prompt(system, tools, instruction, history, reflections)
    output = llm.generate(prompt, stop=["Observation:"])
    thought, action = parse(output)
    if action.type == "final_answer":
        return action.value
    obs = execute(action)                 # 含 try/except，error 也是 obs
    history.append((thought, action, obs))
raise MaxStepExceeded                      # 触发 reflection
```

记住这 8 行——后面的 150 行都是把这个骨架填成可跑的实物。

### 1.3 Reflection 的位置：失败 → 反思 → 重试

ReAct 本身没有"学习"——同一个任务跑 100 次还是同样的错。Reflexion (Shinn 2023) 加的就是这一层：

```
trial 1: ReAct loop → 失败 → LLM 反思失败原因 → reflections.append(reflection)
trial 2: ReAct loop（prompt 里多一段 reflections） → 成功 / 再失败再反思
trial 3: ...
```

它是 **verbal RL**：不更新 weights，但用自然语言写下"上次错在哪、下次应该怎么做"，作为 in-context hint 影响下一轮。本质是用 LLM 的 in-context learning 模拟梯度下降。

---

## 2. 五个组件逐个拆解

### 2.1 Tool Registry：name → callable + schema

最简版就是一个 dict，每个 entry 含 `func / description / params`：

```python
TOOLS = {
    "get_weather": {
        "func": lambda city: {"city": city, "temp_c": 22, "condition": "cloudy"},
        "description": "Get current weather of a city. Use when the user asks about temperature/rain.",
        "params": {"city": "English city name, e.g. 'Beijing'"},
    },
    "calculator": {
        "func": lambda expression: eval(expression),  # demo only, 生产用 ast
        "description": "Evaluate a math expression. Use for arithmetic.",
        "params": {"expression": "Python-style expression, e.g. '22 - 18'"},
    },
}
```

工程上要补三件事：

- **签名校验**：调用前检查 `args.keys() == required_params`，缺参数直接返回 error observation 而不是让 Python 抛 TypeError
- **白名单 enforce**：parser 抓到的 tool name 必须在 registry 里，否则返回 `ERROR: tool 'xxx' not found, available: [...]`
- **timeout / retry**：tool 是 HTTP call 时务必加 timeout，transient error 重试 1-2 次，永久 error 直接回传给 LLM

### 2.2 Prompt Template：tool 描述 + ReAct 格式 + history

完整 prompt 结构：

```
<system 段>
You are a ReAct agent. You have access to tools.

Available tools:
- get_weather(city: str): Get current weather of a city.
- calculator(expression: str): Evaluate a math expression.

Use this exact format:
Thought: <your reasoning>
Action: <tool_name>
Action Input: <JSON of args>
Observation: <result, filled by system>
... (repeat as needed)
Thought: I have the final answer.
Final Answer: <your answer>

<reflections 段（可选）>
Reflections from previous attempts:
- [trial 1] I forgot to convert city names to English; next time always use English.

<user 段>
Question: 北京今天比上海温度高几度？

<history 段>
Thought: I should check Beijing first.
Action: get_weather
Action Input: {"city": "Beijing"}
Observation: {"city":"Beijing","temp_c":22,"condition":"cloudy"}
```

注意：

- **`Action` 与 `Action Input` 拆两行**比合成一行（`Action: get_weather({"city":"Beijing"})`）更稳——前者是 OpenAI Function calling 的标准结构，后者是 LangChain 早期 hacky 形式，正则容易抓错
- **history 在 prompt 里逐 turn 拼接**，新一 turn 的 observation 拼在最后，LLM 的下一次输出从 `Thought:` 开始
- **`stop=["Observation:"]` 必须设**——不设的话 LLM 会自己幻觉一个 observation 接着 reason，等于"假装查了 wiki 然后编答案"

### 2.3 Parser：从纯文本提取结构

最小 regex parser：

```python
import re, json

def parse(output: str):
    if "Final Answer:" in output:
        return {"type": "final_answer", "value": output.split("Final Answer:")[-1].strip()}
    m_action = re.search(r"Action:\s*(\w+)", output)
    m_input = re.search(r"Action Input:\s*(\{.*?\})", output, re.DOTALL)
    if not m_action or not m_input:
        return {"type": "error", "msg": f"Cannot parse action from: {output[-200:]}"}
    try:
        args = json.loads(m_input.group(1))
    except json.JSONDecodeError as e:
        return {"type": "error", "msg": f"Invalid JSON in Action Input: {e}"}
    return {"type": "tool_call", "name": m_action.group(1), "args": args}
```

工程经验：

- **正则一定要 `re.DOTALL`**——LLM 的 JSON 经常跨行，单行匹配抓不到嵌套
- **parse 失败也要返回结构化 error**，让 loop 把 error 拼回 prompt 让 LLM 自己重试，而不是 raise 出去崩
- **理论 parse 准确率上限 99%**，剩下 1% 用 constrained decoding（13.4 §5）或者直接换成 native function calling（见本节 §6）拉到 100%

### 2.4 Executor：执行 + capture exception

执行 tool 时**最关键的一条原则**：把 exception 包装成 observation 喂回 LLM，**不要** raise。

```python
def execute(call: dict) -> str:
    name = call["name"]
    if name not in TOOLS:
        return f"ERROR: Tool '{name}' not found. Available: {list(TOOLS)}"
    try:
        result = TOOLS[name]["func"](**call["args"])
        return json.dumps(result, ensure_ascii=False)
    except TypeError as e:                          # 参数对不上 / 缺字段
        return f"ERROR: Argument mismatch for '{name}': {e}"
    except Exception as e:                          # tool 内部失败
        return f"ERROR: Tool '{name}' failed: {type(e).__name__}: {e}"
```

为什么不能 raise——LLM 完全不知道你的 Python stack trace，只看得见 prompt 里的 observation。把 error 当 observation 是让 LLM "看见失败 → 自我修正"的唯一通道。这条原则在所有 agent framework（LangGraph / smolagents）的 executor 实现里都一致。

### 2.5 Loop Controller：max_step + 死循环 detect + 终止

最小 loop：

```python
def react_loop(question, max_step=10, reflections=None):
    history = []
    for step in range(max_step):
        prompt = build_prompt(question, history, reflections)
        output = llm_call(prompt, stop=["Observation:"])
        parsed = parse(output)
        if parsed["type"] == "final_answer":
            return {"answer": parsed["value"], "trace": history, "success": True}
        if parsed["type"] == "error":
            obs = f"PARSE_ERROR: {parsed['msg']}. Please retry with correct format."
        else:
            obs = execute(parsed)
        # 死循环 detect：连续 2 次相同 (action, args) 直接 break
        if len(history) >= 2 and history[-1][:2] == (parsed.get("name"), str(parsed.get("args"))):
            return {"answer": None, "trace": history, "success": False, "reason": "loop_detected"}
        history.append((parsed.get("name"), str(parsed.get("args")), obs))
    return {"answer": None, "trace": history, "success": False, "reason": "max_step_exceeded"}
```

三个必备 guard：

- **`max_step` 上限**（典型 10-20）：触发即终止，标记 fail，传给 reflection
- **死循环 detect**：连续 2 次相同 `(action, args)` 一定是卡住了，立刻 break；更激进的可以做 last-K 滑窗判断
- **token budget**：超过 prompt 长度上限或预算（如 50k token）也要触发终止

### 2.6 Reflection：失败后写一段 in-context hint

Reflexion 的核心实现就 20 行：

```python
REFLECT_PROMPT = """You just failed at this task:
Question: {question}
Trace: {trace}
Failure reason: {reason}

Write a short reflection (1-2 sentences) on what went wrong and what to try differently next time.
Focus on: which tool to use, how to format arguments, what to avoid.
Reflection:"""

def reflect(question, result):
    if result["success"]:
        return None
    msg = REFLECT_PROMPT.format(
        question=question,
        trace=result["trace"][-3:],     # 只给最后 3 步，省 token
        reason=result["reason"],
    )
    return llm_call(msg, stop=None).strip()

def react_with_reflection(question, max_trials=3, max_step=10):
    reflections = []
    for trial in range(max_trials):
        result = react_loop(question, max_step=max_step, reflections=reflections)
        if result["success"]:
            return result
        reflection = reflect(question, result)
        if reflection:
            reflections.append(f"[trial {trial+1}] {reflection}")
        # 控制 reflection 数量，太多会 confuse model
        reflections = reflections[-2:]
    return result
```

注意：

- **每次 trial 是独立的 ReAct loop**——history 清空，但 reflections 保留并注入新 prompt
- **保留 top-2 最新 reflection** 而不是全部——> 3 条经验上反而 confuse model
- **reflection 内容是自然语言**——不是 token-level gradient、不是 weight update，纯 in-context

---

## 3. 完整可跑代码（< 150 行）

下面是 self-contained 的完整实现，把上面 5 个组件拼起来。复制到本地，把 `OPENAI_API_KEY` 设好就能跑。

```python
"""
Minimal ReAct + Reflection agent in < 150 lines.
Run: pip install openai && OPENAI_API_KEY=sk-... python react_min.py
"""
import json, re, os
from openai import OpenAI

client = OpenAI()
MODEL = "gpt-4o-mini"

# ---------- 1. Tool Registry ----------
TOOLS = {
    "get_weather": {
        "func": lambda city: {"city": city, "temp_c": {"Beijing": 22, "Shanghai": 18, "London": 12}.get(city, 20)},
        "description": "Get current weather of a city. Returns temperature in Celsius.",
        "params": {"city": "English city name, e.g. 'Beijing'."},
    },
    "calculator": {
        "func": lambda expression: eval(expression, {"__builtins__": {}}, {}),
        "description": "Evaluate a math expression. Use for arithmetic only.",
        "params": {"expression": "Python-style math, e.g. '22 - 18' or '3 * (4+5)'."},
    },
}

# ---------- 2. Prompt Template ----------
def build_prompt(question, history, reflections):
    tool_desc = "\n".join(
        f"- {name}({', '.join(t['params'])}): {t['description']}\n  args: {t['params']}"
        for name, t in TOOLS.items()
    )
    refl = "\nReflections from past attempts:\n" + "\n".join(reflections) if reflections else ""
    hist = ""
    for name, args, obs in history:
        hist += f"\nAction: {name}\nAction Input: {args}\nObservation: {obs}\n"
    return f"""You are a ReAct agent.

Available tools:
{tool_desc}

Use this EXACT format (one step at a time):
Thought: <reasoning>
Action: <tool_name>
Action Input: <JSON dict of args>

When done:
Thought: I have the final answer.
Final Answer: <answer>
{refl}

Question: {question}{hist}
"""

# ---------- 3. Parser ----------
def parse(output):
    if "Final Answer:" in output:
        return {"type": "final_answer", "value": output.split("Final Answer:")[-1].strip()}
    m_a = re.search(r"Action:\s*(\w+)", output)
    m_i = re.search(r"Action Input:\s*(\{.*?\})", output, re.DOTALL)
    if not m_a or not m_i:
        return {"type": "error", "msg": f"Cannot parse from: {output[-200:]}"}
    try:
        args = json.loads(m_i.group(1))
    except json.JSONDecodeError as e:
        return {"type": "error", "msg": f"Bad JSON: {e}"}
    return {"type": "tool_call", "name": m_a.group(1), "args": args}

# ---------- 4. Executor ----------
def execute(call):
    name = call["name"]
    if name not in TOOLS:
        return f"ERROR: tool '{name}' not found. Available: {list(TOOLS)}"
    try:
        return json.dumps(TOOLS[name]["func"](**call["args"]), ensure_ascii=False)
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"

# ---------- 5. LLM Call ----------
def llm_call(prompt, stop=None):
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        stop=stop,
        temperature=0,
    )
    return resp.choices[0].message.content

# ---------- 6. Loop Controller ----------
def react_loop(question, max_step=10, reflections=None):
    history = []
    last_call = None
    for step in range(max_step):
        prompt = build_prompt(question, history, reflections or [])
        out = llm_call(prompt, stop=["Observation:"])
        print(f"\n--- Step {step+1} ---\n{out}")
        parsed = parse(out)
        if parsed["type"] == "final_answer":
            return {"answer": parsed["value"], "trace": history, "success": True}
        if parsed["type"] == "error":
            obs = f"PARSE_ERROR: {parsed['msg']}. Reformat strictly."
            history.append(("__parse_error__", "", obs))
            continue
        cur_call = (parsed["name"], json.dumps(parsed["args"], sort_keys=True))
        if cur_call == last_call:
            return {"answer": None, "trace": history, "success": False, "reason": "loop_detected"}
        last_call = cur_call
        obs = execute(parsed)
        print(f"Observation: {obs}")
        history.append((parsed["name"], json.dumps(parsed["args"]), obs))
    return {"answer": None, "trace": history, "success": False, "reason": "max_step_exceeded"}

# ---------- 7. Reflection ----------
REFLECT = """Task FAILED:
Question: {q}
Last steps: {trace}
Reason: {reason}
Write a 1-sentence reflection on what to do differently next time. Reflection:"""

def reflect(question, result):
    msg = REFLECT.format(q=question, trace=result["trace"][-3:], reason=result["reason"])
    return llm_call(msg).strip()

def react_with_reflection(question, max_trials=3, max_step=10):
    reflections = []
    for trial in range(max_trials):
        print(f"\n========== TRIAL {trial+1} ==========")
        result = react_loop(question, max_step=max_step, reflections=reflections)
        if result["success"]:
            return result
        r = reflect(question, result)
        print(f"\n[Reflection] {r}")
        reflections.append(f"[trial {trial+1}] {r}")
        reflections = reflections[-2:]                  # 只保留最近 2 条
    return result

# ---------- 8. Demo ----------
if __name__ == "__main__":
    q = "北京今天比伦敦温度高几度？"
    res = react_with_reflection(q)
    print(f"\n=== FINAL ===\n{res}")
```

跑通后典型输出（删节）：

```
========== TRIAL 1 ==========
--- Step 1 ---
Thought: I need to check the temperature in both Beijing and London.
Action: get_weather
Action Input: {"city": "Beijing"}
Observation: {"city": "Beijing", "temp_c": 22}

--- Step 2 ---
Thought: Now London.
Action: get_weather
Action Input: {"city": "London"}
Observation: {"city": "London", "temp_c": 12}

--- Step 3 ---
Thought: Beijing is 22 - 12 = 10 degrees warmer.
Action: calculator
Action Input: {"expression": "22 - 12"}
Observation: 10

--- Step 4 ---
Thought: I have the final answer.
Final Answer: 北京比伦敦高 10°C。
```

整个 demo < 150 行（去掉注释和空行约 100 行），含完整 5 组件 + reflection，可直接 copy 上手 debug。

---

## 4. 用 OpenAI Function Calling 重写为现代版

text parser 路线最大的问题是**format drift**——LLM 偶尔忘记空一行、用 markdown 包 JSON、把 `Action Input:` 写成 `Input:`，parser 就 fail。OpenAI / Anthropic 的 native function calling（13.4 §2）把 schema 做成强类型协议，从根本上消灭这类 fail。

```python
"""ReAct via OpenAI native function calling - <30 lines core."""
import json
from openai import OpenAI

client = OpenAI()
TOOLS = {
    "get_weather": (lambda city: {"city": city, "temp_c": {"Beijing": 22, "London": 12}.get(city, 20)},
                    {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}),
    "calculator": (lambda expression: eval(expression, {"__builtins__": {}}, {}),
                   {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}),
}
schemas = [{"type": "function", "function": {"name": n, "description": n, "parameters": p}}
           for n, (_, p) in TOOLS.items()]

def react_fc(question, max_step=10):
    msgs = [{"role": "user", "content": question}]
    for _ in range(max_step):
        resp = client.chat.completions.create(model="gpt-4o-mini", messages=msgs, tools=schemas)
        m = resp.choices[0].message
        msgs.append(m)
        if not m.tool_calls:
            return m.content                                            # final answer
        for tc in m.tool_calls:                                         # 支持 parallel
            fn, _ = TOOLS[tc.function.name]
            try:
                result = fn(**json.loads(tc.function.arguments))
            except Exception as e:
                result = f"ERROR: {e}"
            msgs.append({"role": "tool", "tool_call_id": tc.id,
                         "content": json.dumps(result, ensure_ascii=False)})
    return None

print(react_fc("北京比伦敦温度高几度？"))
```

对比 §3 的 150 行版本：

| 维度 | Text-parser ReAct | Function calling ReAct |
|---|---|---|
| 代码量 | ~150 行 | ~30 行 |
| Parser 鲁棒性 | 95-99%（regex 易 fail） | 100%（SDK 已 parse） |
| Parallel tool | 手动支持，麻烦 | 默认开 |
| Streaming | 要自己拼 chunk | SDK helper |
| 教学价值 | 高（看清每个组件） | 低（黑盒） |
| 生产推荐 | ❌ | ✅ |

**结论**：理解原理写一遍 §3 的版本，生产代码用 §4 的 function calling，**两者不冲突**。Reflection 这一层可以叠在 function calling 上——把 trial 失败后的反思加在下一轮的 system prompt 里即可，本质相同。

---

## 5. 现代 ReAct 的几个改进方向

### 5.1 Function calling 替代 text parser

如 §4 所述，这是 production 的事实标准。仅当 (a) 用的 model 不支持 native function calling（如某些早期 OSS chat model），或 (b) 想要 code-as-action（smolagents 风格），才退回到 text parser。

### 5.2 Streaming：边 reasoning 边返回

13.4 §4 讲过 streaming 下 `tool_call.arguments` 是 chunk 拼接的。Agent loop 里加 streaming 主要为两个目的：

- **UX**：用户能看见 thought 的 typing 效果，比"转圈 30s 才出结果"友好
- **Early stop**：长 thought 的中段如果检测到敏感内容 / 跑偏，可以提前 abort 省 token

实现上把 `react_loop` 里的 `llm_call` 换成 `stream=True`，逐 chunk yield 给 UI，最后拼完整 message 再 `parse + execute`。

### 5.3 Parallel tool calls

用户问"北京和上海哪个暖和"，理想是同 turn 出两个 `get_weather` call、并行执行、一起拼回。13.4 §3 已详讲。Reflection 与 parallel 不冲突——失败后照常反思即可。

### 5.4 Reasoning model 直接 reason

R1 / o1 / o3 这类 reasoning model（10.3 详讲）一次 forward 就出 long-CoT，**不需要外部 ReAct loop 拆 turn**。典型 pattern 变成：

```
user → reasoning_model.generate(thinking=long, tools=[...]) → 一次性产出 trajectory
```

模型自己在 long-thought 内部完成"分析 → 决定调 tool → 等结果 → 综合"的全过程，外部 driver 只负责执行 tool 返回结果。**对这类模型套传统 ReAct prompt 反而拖累——会强行打断它的 long-CoT 节奏**，让它在每个 short turn 之间切换，损失 reasoning 连贯性。

工程判断：

- 用 Claude 3.5 Sonnet / GPT-4o / Qwen2.5 这类**通用 chat model** → 用传统 ReAct + function calling
- 用 R1 / o3 / Claude Opus 4 with extended thinking → **去掉 ReAct 框架**，让模型自己 reason，driver 只负责 tool execution

---

## 6. 工程踩坑与经验

- ❗ **Text parser 迟早 fail**——LLM 的 format drift 是必然的（同一 prompt 跑 1000 次总有 1-10 次格式漂移）。production 要么用 native function calling，要么用 constrained decoding（13.4 §5）兜底。**别迷信 prompt 工程能把 parse 准确率拉到 100%**
- ❗ **`max_step` 必须设**（典型 10-20）——LLM 陷入"Thought: I should search again..."死循环的概率比想象中高，尤其在简单问题上反复怀疑自己。没有 max_step 的 agent 是定时炸弹，会烧光 budget
- ❗ **Tool error 要 capture 后 observation 化**——`raise` 出去 LLM 完全看不到，只剩用户看见 500。把所有 exception 包成 `ERROR: ...` 字符串喂回去，让 LLM 显式处理是 agent recovery 的核心通道
- ❗ **Reflection 数量要限制**（top-2 最相关）——> 3 条 reflection 在 prompt 里反而 confuse model（"上次说不要 X，上上次说要 Y，到底听哪个"）。Reflexion 原论文也建议 sliding window
- ❗ **History 越长越贵**——每一 turn 都把所有 past observation 拼进 prompt，token 累积是 $O(n^2)$ 级别。超过 5-10 turn 必须 truncate（保留 first + last K）或 summarize（让 LLM 把老 turn 总结一句）
- ❗ **Reasoning model 不要套 ReAct prompt**——R1 / o1 的 long-CoT 自带 "reasoning + acting" 节奏，外面套 "Thought: / Action:" 模板会打断它的 long-thought 连贯性，反而掉点。R1-class 模型用 native tool use 接口即可
- ❗ **Tool description 是 prompt 的一部分**——花 30 分钟优化一个 tool 的 description（加正例 / 负例 / 参数语义示例）经常比换 model 还有用。BFCL 上同 model 同 task 仅 description 不同，准确率差 10%+（呼应 13.4 §2.2）
- ❗ **Final answer 形态不固定时用 schema 强制**——业务下游要 `{"price": 100, "currency": "USD"}` 这种结构化结果，**不要**靠 prompt 里写 "请按这个格式输出 JSON"，要么用 OpenAI 的 `response_format=Pydantic`，要么用 vLLM `guided_decoding`。**自由文本输出 → 业务解析**这条路在 production 几乎一定会爆
- ❗ **死循环 detect 不能只看完全相同**——LLM 会用细微变体绕开（`{"city":"Beijing"}` vs `{"city": "Beijing"}` vs `{"city":"beijing"}`）。比较时要 normalize（lowercase + strip whitespace）；更狠的可以做 embedding 相似度判断
- ❗ **`stop=["Observation:"]` 必须设**——否则 LLM 会自己幻觉 observation 接着 reason，等于"假装查了 wiki 然后编答案"。这是 ReAct 实现里最常被新人忘掉的细节
- ❗ **Reflection 的 prompt 要明确"做错了什么、下次怎么做"**——只让 LLM "反思" 它会写很多哲学化的废话（"我应该更仔细思考"）。要明确要 actionable insight：tool 选错 / 参数填错 / 没考虑某 case
- ❗ **测试 agent 不要只看 success rate**——还要看 token cost、step count、tool failure rate。一个 80% success 但平均 30 step / 每次 50k token 的 agent 不可上线，远不如 70% success / 8 step / 8k token 的版本

---

## 7. 经典 paper

- **Yao et al., 2022 — ReAct: Synergizing Reasoning and Acting in Language Models** — 必读。本节代码就是这篇 §3.1 的直接实现。重点看 §2 的 OAR 形式化、§4 的 prompt 设计示例、§5 在 HotpotQA / ALFWorld 的对比实验——理解为什么"边 reason 边 act"比"先 plan 再做"更鲁棒
- **Shinn et al., 2023 — Reflexion: Language Agents with Verbal Reinforcement Learning** — Reflection 机制的原典。读 §3 的 reflection prompt 设计与 §4 的 actor / evaluator / self-reflection 三模块拆解，理解"verbal RL"为什么能在不更新 weights 的情况下提升 task success rate（HumanEval pass@1 从 67% 提到 91%）
- **Anthropic 2024 — Building Effective Agents** — 工程视角的 agent design pattern 总结。强调 simplicity over complexity——文中的 augmented LLM / prompt chaining / routing / parallelization / orchestrator-worker / evaluator-optimizer 几个 pattern 直接对应本节的 ReAct + Reflection
- 加分：**smolagents 源码** (github.com/huggingface/smolagents) — 把"最小 agent"做到 ~1000 行核心代码的开源实现，跟本节 150 行版本相互印证；**LangChain ReAct agent docs**（python.langchain.com/docs/modules/agents/agent_types/react）

---

## 8. 自测与面试题

**Q1（实现）**：写出最小 ReAct loop 的 5 步伪代码（不用调任何库）。

<details>
<summary>Answer sketch</summary>

要覆盖 5 步：

```python
history = []
for step in range(max_step):
    # 1. 构造 prompt：含 system / tool descriptions / history / 当前 question
    prompt = build_prompt(system, tools, instruction, history)
    # 2. LLM 生成 thought + action（必带 stop=["Observation:"]）
    output = llm.generate(prompt, stop=["Observation:"])
    # 3. parse 出 action（thought 顺带）
    thought, action = parse(output)
    # 4. 终止判定
    if action.type == "final_answer":
        return action.value
    # 5. 执行 tool（含 try/except 把 error 转 observation），写回 history
    obs = execute(action)
    history.append((thought, action, obs))
raise MaxStepExceeded
```

评分点：
- 5 步齐全（build prompt / generate / parse / final answer 判定 / execute + append）
- 必须有 `max_step` 上限
- 必须有 `stop=["Observation:"]`
- exception 转 observation 不能 raise
- 加分：能补上死循环 detect（连续相同 action 就 break）

</details>

**Q2（debug）**：你的 ReAct agent 经常 hallucinate 不存在的 tool 名称（如 prompt 里只有 `search` 和 `calculator`，model 却调 `web_search`），给出 3 个不同方向的修复方案。

<details>
<summary>Answer sketch</summary>

三个方向（侵入性递增）：

**1. Prompt 层：tool description 写更清楚 + few-shot**
- 在 system prompt 末尾加一句 `"You MUST only use tools from the above list. Do NOT invent new tool names."`
- 给 1-2 个 few-shot 示例，演示完整 thought → action → observation
- 代价：几乎零，应该先做。**典型涨点 5-10%**

**2. Parser / Executor 层：白名单 enforce + error 反馈**
- parse 出 tool name 后立刻 check 是否在 registry 里
- 不在就返回结构化 observation：`ERROR: Tool 'web_search' not found. Available tools: [search, calculator]. Did you mean 'search'?`
- LLM 看到这条 observation 通常下一步就改对
- 代价：几行代码。**production 必备**

**3. 协议层：换 native function calling**
- OpenAI / Anthropic / Qwen 的 function calling 通过 SDK schema 把 `name` 字段约束到 enum——model 在生成 `tool_calls.function.name` 时**只能从 schema 里的 name 中选**（许多实现底层用 constrained decoding 强制）
- 代价：要重写 agent loop，但能从根上消灭 hallucinated tool

加分：还可以提到**tool retrieval**——tool 多于 20 个时，先用 embedding 检索 top-K 个候选 tool 再给 LLM，hallucinate 概率随候选数下降而下降（13.4 §8）。也可以提到**用 reflection 机制**：上次调错 tool 后写一条反思 "Available tools are X, Y, do not invent names"，下次注入 prompt。

</details>

**Q3（前沿）**：Reasoning model（DeepSeek-R1 / o3 / Claude Opus 4 with extended thinking）出现后，外部 ReAct loop 还有意义吗？

<details>
<summary>Answer sketch</summary>

**结论**：意义减弱但没消失，分场景看。

**Reasoning model 取代 ReAct loop 的部分**：
- 单 turn 内的 long-CoT 已经隐式完成 "Thought → Action 决策"——不需要外部 driver 强行拆 turn
- R1-class 模型在 thinking 内部已经会自己推理 "我现在该不该调 tool / 调什么 tool / 上次结果怎么解读"，外面套 ReAct prompt 反而打断这个连贯节奏
- 实测：在 SWE-bench / GAIA 上，Claude Opus 4 with extended thinking 用 native tool use（无 ReAct prompt）经常比套 ReAct loop 更高分

**但 ReAct loop 仍然有意义的部分**：
- **Tool execution 必须外部化**——LLM 永远不能真的调 API、不能真的写文件，driver 仍要做这件事
- **Multi-turn 状态管理**——stateful tool（browser、shell、DB）的状态要外部维护，loop controller 不可省
- **Safety / sandbox / audit**——所有 action 要有外部审计、白名单、权限检查，这层不能内化进 weights
- **Reflection / cross-trial memory**——一次任务失败后跨 trial 的反思记忆，仍需外部 store
- **简单 / 弱 model**：用 7B 开源 model 时它没有 long-CoT 能力，传统 ReAct + 强 prompt 是唯一现实选择

**新形态**：把 ReAct loop **简化为 thin driver**——不再 prompt LLM "请按 Thought / Action 格式"，而是让 reasoning model 自由发挥，driver 只做 tool dispatch + history append + max_step guard。这是 Claude Code、Cursor Agent 等现代产品的实际架构。

加分：能区分 "ReAct prompt template" 与 "ReAct architectural pattern" 这两件事——前者（强 format 约束的 prompt）正在被 reasoning model 取代，后者（loop + tool dispatch + observation feedback 的整体架构）仍是所有 agent 的骨架。

</details>

---

## 9. 延伸阅读

- [ReAct 论文官方 repo](https://github.com/ysymyth/ReAct) — 含 HotpotQA / ALFWorld 上的 prompt 与评测代码，对照本节的 150 行实现差异
- [Reflexion 论文 repo](https://github.com/noahshinn/reflexion) — Reflection 机制的官方实现，含 actor / evaluator / self-reflection 三模块代码
- [Anthropic — Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents) — 2024 年最被引用的 agent design pattern 文章，强调"simplicity over complexity"，本节就是它推崇的"自己写 < 200 行 loop"的实物
- [smolagents 源码](https://github.com/huggingface/smolagents) — HuggingFace 极简 agent 框架，~1000 行核心代码，是本节 150 行版本的"工业化"版本，对比阅读收益高
- [LangChain ReAct agent](https://python.langchain.com/docs/modules/agents/agent_types/react) — 工业级 ReAct agent 实现，代码繁但接口标准，看 production 形态
- [OpenAI Cookbook — How to call functions with chat models](https://cookbook.openai.com/examples/how_to_call_functions_with_chat_models) — function calling 完整闭环示例，对应本节 §4 现代版
- 推荐继续读本教程的 **14.3 节《Tool Use 训练》**——本节假设"用现成 LLM 跑 agent"，14.3 讲怎么把 tool use 能力训进 model（Toolformer / Gorilla / xLAM / ToolLLaMA 谱系）；以及 **14.4 节《Planning》**——把单一 ReAct loop 升级成 plan + execute 混合范式
