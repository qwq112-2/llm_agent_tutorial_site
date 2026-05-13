---
title: "13.4 Function calling 工程：JSON schema / 并行调用 / constrained decoding"
description: "Function calling 是把 LLM 当 controller、把外部代码当 capability 的标准化协议——这一节讲清楚 inference 时的工程层（JSON schema 设计 / OpenAI / Anthropic / vLLM 三家格式差异 / parallel & streaming / constrained decoding 与 XGrammar / 主流可靠性"
---

> ⏱ 预计阅读 35 分钟 ｜ 难度 ★★ ｜ 前置：3.2（特殊 token / chat template）｜🔥 必考

## 一句话本节讲什么

Function calling 是把 LLM 当 controller、把外部代码当 capability 的标准化协议——这一节讲清楚 inference 时的工程层（JSON schema 设计 / OpenAI / Anthropic / vLLM 三家格式差异 / parallel & streaming / constrained decoding 与 XGrammar / 主流可靠性踩坑），区别于 8.5 与 14.3 的"训练层"内容，回答"已经有一个支持 tool 的 model 了，怎么把它接进生产系统"。

---

## 1. Mental model（直觉）

把 function calling 想成一种**受约束的文本生成 + 外置执行 loop**：

```
┌──────────┐  prompt + tool schemas       ┌─────────────┐
│  Caller  │ ───────────────────────────► │     LLM     │
│ (你的    │                               │             │
│  程序)   │ ◄─── tool_call (JSON) ─────── │  生成器     │
│          │                               └─────────────┘
│  执行    │  city="Beijing"
│  函数    │ ───────────────────────────► get_weather()
│          │                               returns 25°C
│          │ ───────────────────────────► ┌─────────────┐
│          │     prompt + tool result     │     LLM     │
│          │                               │             │
│          │ ◄─── final answer ─────────── │  生成器     │
└──────────┘                               └─────────────┘
```

LLM **不亲自执行**任何代码——它只输出一段结构化 JSON，告诉 caller "请帮我用这些参数调这个函数"，caller 执行完再把结果（observation）拼回 prompt 让 LLM 继续。这是与 ReAct agent loop（14.2 详讲）同源的心智模型，只不过现代 LLM 把"思考要调哪个工具"这件事从 free-form text 升级成了**强类型 JSON 协议**。

四个角色要分清：

- **Tool**：真正的外部能力，由 caller 在自己进程里执行（HTTP API、数据库、本地函数都行）
- **Tool Schema**：用 JSON Schema 描述 tool 的接口（名字 / 参数 / 类型 / 描述），随 prompt 喂给 LLM
- **Tool Call**：LLM 的输出，结构化的 `{name, arguments}`
- **Tool Response / Observation**：caller 执行后的结果，以特定 role 拼回 message history

理解这个分工后，三件事就顺了：

1. **本节聚焦 inference 工程**——不讨论"怎么训出会调用 tool 的模型"（那是 8.5、14.3、15.x）
2. **schema 设计 = prompt 工程的一部分**——description 写得好坏直接影响调用准确率
3. **格式差异是兼容性问题**——OpenAI 的 `tool_calls` 字段、Anthropic 的 `tool_use` block、vLLM 的 `<tool_call>` 标签互不通用，工程上要写**统一抽象层**

LLM 在生成层面其实没有"function call"这种"原生指令"——它**永远只在生成 token 流**。所谓的 tool_call 是 (a) 训练时把"决定调工具+输出 JSON"这件事 SFT/RLHF 进了模型 (b) 推理框架按照特定的 chat template / special token（见 3.2 节）把生成出的字符串解析回结构化字段 (c) SDK 层再把它包装成对象。理解这三步分工，就能搞清"为什么换个 inference engine 同一个 model 表现差异这么大"——多半是 chat template 或 parser 没对上。

> 本节的"训练层"内容（如何 SFT 出 tool-use 能力、如何用 RL 训多轮调用）请看 **8.5 多轮+tool 混合 SFT** 与 **14.3 Toolformer / Gorilla / xLAM**。本节假设你拿到的是**已经支持 tool calling 的 model**（任何 GPT-4o / Claude 3.5 / Qwen2.5-Instruct / Llama-3.1-Instruct 都行），讨论"怎么用好它"。

---

## 2. JSON Schema：tool 的"prompt 接口"

### 2.1 标准格式

OpenAI 风格的 tool schema 是工业事实标准（Anthropic / Qwen / Llama / Mistral / DeepSeek 都兼容这套结构，字段名稍有差异）：

```json
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "Get current weather for a city. Use this when the user asks about temperature, rain, snow, or 'how's the weather'.",
    "parameters": {
      "type": "object",
      "properties": {
        "city": {
          "type": "string",
          "description": "City name in English, e.g. 'Beijing', 'New York'."
        },
        "unit": {
          "type": "string",
          "enum": ["celsius", "fahrenheit"],
          "description": "Temperature unit. Default celsius."
        }
      },
      "required": ["city"]
    }
  }
}
```

四个字段决定一切：

- **`name`**：tool 的标识符。LLM 会**逐字符**生成这个字符串，所以名字越短、越像常见英文单词、越无下划线断片越鲁棒（`search_web` > `web_search_tool_v2_internal`）
- **`description`**：tool 的"自然语言文档"。**这是 prompt 的一部分**——写得好直接涨调用准确率 10%+。要写清楚 (1) 这个 tool 能做什么 (2) 什么时候**应该**调它 (3) 什么时候**不应该**调它（避免 LLM 在不需要时硬调）
- **`parameters`**：JSON Schema，严格的 `{type, properties, required}` 结构。`enum` 用来限定枚举值，`required` 标必填字段
- 各 property 的 **`description`**：每个参数也要单独描述。**最容易被忽略也最重要**——如果 `city` 没写"用英文"，模型可能传 "北京"，下游 API 报错

### 2.2 description 的写法直接决定准确率

社区与 BFCL 的实测经验：同一 model 同一 task，仅修改 tool description（其他都不动），调用准确率波动 5-15%。三条 checklist：

1. **写正例**："Use this when the user asks for stock price."
2. **写负例**："Do NOT use this for cryptocurrency prices, use `get_crypto_price` instead."
3. **写参数语义**：city 是英文还是中文？日期格式是 ISO 8601 还是 yyyy-MM-dd？枚举值大小写敏感吗？

把 tool description 当成"内部 API 文档"来写，不是"给同事的 wiki"——LLM 是你最严格也最笨的 reader。

一个真实对比：同一个 model 上，把 description 从 `"Search the web."`（5 个 token）改成 `"Search the public web for recent news, factual answers, and entity lookups. Use this when the user asks about events after the model's training cutoff (e.g. 'what happened today'), or when explicitly asked to 'search'. Do NOT use this for code questions — use 'run_python' instead."`（约 60 token），BFCL irrelevance 子集准确率从 71% 涨到 84%——多花的 55 个 prompt token 在 production 用量下几乎可忽略，accuracy 涨幅却显著。**Tool description 是性价比最高的优化点之一**。

另一个常见错误是把 schema 写得太"宽容"——比如 `params` 写成 `{"type": "object"}`（不指定 properties），LLM 会自由生成任何字段、下游 parser 一定崩。**永远精确写 properties + required + types**，宁可重复写 description 也不要省 schema。

### 2.3 OpenAI / Anthropic / vLLM 的格式差异

三家的接口大同小异，但工程上**互不兼容**，必须写一层 adapter：

| 维度 | OpenAI | Anthropic | vLLM (OSS Qwen / Llama) |
|---|---|---|---|
| 请求字段 | `tools=[{type:"function", function:{...}}]` | `tools=[{name, description, input_schema}]` | `tools=[...]` (兼容 OpenAI) |
| 响应表示 | `message.tool_calls=[{id, type, function:{name, arguments}}]` | `content=[{type:"tool_use", id, name, input}]` | 同 OpenAI（vLLM 解析 chat template 内 `<tool_call>...</tool_call>` 还原） |
| arguments 类型 | **string**（要 `json.loads`） | **dict**（已 parse） | string |
| tool result 回传 role | `"tool"` + `tool_call_id` | `user` role + `content=[{type:"tool_result", tool_use_id, content}]` | `"tool"` + `tool_call_id` |
| parallel 默认 | 默认开 (`parallel_tool_calls=True`) | 默认开 | 取决于 model |
| streaming arguments | 增量 string，要 buffer | 增量字符串，SDK 有 `accumulate()` helper | 增量 string |

**实战 take away**：如果你的产品要同时接 OpenAI / Claude / 自家 OSS 部署的 Qwen，**不要直接把 SDK 返回的对象往业务层传**——写一个 `UnifiedToolCall(id, name, arguments_dict)` dataclass，每家在 adapter 层归一化后再吐给业务。下游就一份代码即可。

---

## 3. Parallel Tool Calls

LLM 在一次 turn 里同时输出多个 tool call，由 caller **并行**执行后一次性把所有 result 喂回。典型场景：用户问"明天北京和上海哪个暖和"——LLM 应该同时调 `get_weather(Beijing)` 和 `get_weather(Shanghai)`，而不是先后两轮。

```python
# 一次 response 里包含两个 tool_call
response.choices[0].message.tool_calls
# [
#   ToolCall(id="call_1", function={"name":"get_weather", "arguments":'{"city":"Beijing"}'}),
#   ToolCall(id="call_2", function={"name":"get_weather", "arguments":'{"city":"Shanghai"}'}),
# ]
```

工程要点：

- **OpenAI / Anthropic / Qwen2.5+ 默认开**（`parallel_tool_calls=True`）；要关闭就显式设 `False`
- **不是所有 model 都支持稳定的 parallel**——LLaMA-3-8B-instruct、早期 Mistral 经常 parallel 出错（要么只出一个、要么参数串行污染）。生产系统在小模型上**默认 sequential 更安全**
- caller 端**真的要并行执行**才有意义——用 `asyncio.gather` / `concurrent.futures` 同时发出两个 HTTP 请求，再按 `tool_call_id` 拼回响应
- result 回传时**每个 tool_call_id 都要有对应的 tool message**，缺一个 OpenAI 会报 `400`：

```python
messages.append(response.choices[0].message)  # 包含全部 tool_calls
for tc in response.choices[0].message.tool_calls:
    result = execute(tc.function.name, json.loads(tc.function.arguments))
    messages.append({
        "role": "tool",
        "tool_call_id": tc.id,
        "content": json.dumps(result),
    })
```

---

## 4. Streaming + Tool Calls

streaming 模式下 tool_call 的 `arguments` 字段是**逐 token 增量返回**的（毕竟它是模型生成的字符串）。客户端必须 **buffer 完整 JSON 后才能 `json.loads`**，否则中间任何一帧 parse 都会报 `Unterminated string`。

```python
# OpenAI streaming: arguments 是 chunk 拼起来的
buffer = {}  # tool_call_id -> {"name": ..., "arguments": ""}
for chunk in stream:
    delta = chunk.choices[0].delta
    if delta.tool_calls:
        for tc in delta.tool_calls:
            idx = tc.index
            if idx not in buffer:
                buffer[idx] = {"name": tc.function.name or "", "arguments": ""}
            if tc.function.arguments:
                buffer[idx]["arguments"] += tc.function.arguments
# stream 结束后再 parse
for idx, tc in buffer.items():
    args = json.loads(tc["arguments"])
```

OpenAI / Anthropic SDK 都提供 `stream.get_final_message()` / `stream.until_done()` 这类 helper——production 用 SDK 别手撕。但**理解底层是 chunk 拼接**很重要，因为 (1) UI 上要做"打字机"动画时要决定哪些 chunk 渲染、哪些隐藏 (2) 调试 tool_call_id 串台时要看原始流。

---

## 5. Constrained Decoding：从概率上消灭非法 JSON

prompt 加 `"output JSON"` 的成功率上限大概 95-99%——剩下的 1-5% 总会出现 `{"city": "Beijing"`（缺右括号）、`{"city: Beijing}`（缺引号）这类 syntax 错误。constrained decoding 用**生成时屏蔽非法 token** 的方式把成功率推到 100%。

### 5.1 原理

每一步生成 logit 时，按当前已生成前缀计算"哪些 token 在语法上合法"，把非法 token 的 logit 设为 $-\infty$ 再 softmax。例如已经输出 `{"city":` 时，下一个合法 token 必须是 string 起始的 `"`，所以把所有非 `"` 开头的 token mask 掉。

三种主流实现：

- **JSONFormer** (Jain 2023)：早期工作，对每种 JSON 结构（object / array / string / number）写硬编码 finite-state machine。简单 schema 够用，复杂 nested 难写
- **Outlines** (Willard 2023)：把 regex 或 Pydantic model 编译成 FSM，逐 token 查表。Hugging Face 生态友好，对开发者最熟
- **XGrammar** (Dong et al., 2024)：基于上下文无关文法（CFG），编译期把 grammar 转成压缩 mask 矩阵 + token-level lookup，**比 Outlines 快 5-10×**，长 schema 上几乎零 overhead。已被 vLLM (`guided_decoding_backend="xgrammar"`) 与 SGLang 设为默认

### 5.2 集成方式

vLLM 的 guided JSON：

```python
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

schema = {
    "type": "object",
    "properties": {
        "city": {"type": "string"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
    },
    "required": ["city"],
}
sp = SamplingParams(
    temperature=0.7,
    max_tokens=128,
    guided_decoding=GuidedDecodingParams(json=schema, backend="xgrammar"),
)
out = llm.generate(["调用 get_weather"], sp)
print(out[0].outputs[0].text)  # 100% 是合法 JSON
```

Outlines 的 Pydantic 风格：

```python
import outlines
from pydantic import BaseModel
from typing import Literal

class GetWeather(BaseModel):
    city: str
    unit: Literal["celsius", "fahrenheit"]

model = outlines.models.transformers("Qwen/Qwen2.5-7B-Instruct")
gen = outlines.generate.json(model, GetWeather)
out = gen("Tell me weather in Beijing in JSON")
print(out)  # GetWeather(city="Beijing", unit="celsius")
```

### 5.3 代价：5-20% latency overhead

constrained decoding 不是免费的：

- 每步要计算"当前合法 token 集合"——对深度 nested schema 或大正则表达式可能 $O(V)$ 量级
- XGrammar 优化到接近常数时间，但仍有 5-10% 增量；纯 Outlines / JSONFormer 复杂 schema 上能掉 20-30%
- KV cache 不变（只 mask 不影响 attention），所以 throughput 影响小于 latency

**什么时候开**：production 与外部系统对接（数据库 / 第三方 API）、对 syntax 100% 合法刚需的场景。**什么时候关**：探索性聊天、free-form 输出、内部工具能容忍 retry 的场景。

### 5.4 一个直观的 FSM 例子

理解 constrained decoding 最直接的方式是手画一个 schema 的状态机。考虑 schema `{"city": "<string>"}`：

```
state 0 (起点) ──"{"──► state 1
state 1       ──"\""──► state 2
state 2       ──"city"──► state 3
state 3       ──"\""──► state 4
state 4       ──":"──► state 5
state 5       ──"\""──► state 6
state 6       ──任意非引号字符──► state 6 (循环)
state 6       ──"\""──► state 7
state 7       ──"}"──► ACCEPT
```

在 state 1 时只有 `"` 是合法 next token，模型在 vocab 上的所有非 `"`-起始 token logit 都被 mask 成 $-\infty$；在 state 6 的"字符串内容"循环里几乎所有 token 都合法，mask 几乎不起作用——所以**constrained decoding 对生成质量的影响非常局部**，主要在结构边界处发挥作用，"中间内容"完全由 model 自己决定（这就是为什么它不会让 model "变蠢"）。

XGrammar 与 Outlines 的核心工程优化是把这种 FSM **编译成 token-level 的位图查表**而不是字符级的状态推进——一个 token 通常对应多个字符（如 `","city":"`），FSM 直接做 token transition 而非字符 transition，省掉每步 100-1000× 的状态计算。

---

## 6. 最小代码示例

### 6.1 OpenAI 完整 function calling 闭环

```python
import json
from openai import OpenAI

client = OpenAI()
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "English city name"}},
            "required": ["city"],
        },
    },
}]

def get_weather(city: str) -> str:
    return json.dumps({"city": city, "temp_c": 25, "condition": "sunny"})

messages = [{"role": "user", "content": "What's the weather in Beijing?"}]
resp = client.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=tools)
msg = resp.choices[0].message
messages.append(msg)  # 包含 tool_calls 的 assistant message

for tc in msg.tool_calls:                                # 可能 parallel 多个
    args = json.loads(tc.function.arguments)
    result = get_weather(**args)
    messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

final = client.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=tools)
print(final.choices[0].message.content)  # "It's sunny and 25°C in Beijing today."
```

整个 loop 三件事：(1) 第一次 call 拿 tool_call (2) 本地执行得 result (3) 把 result 拼回 messages 再 call 一次拿最终答案。Multi-turn agent loop 就是把这个 pattern 套在 while 里。

### 6.2 Pydantic schema + structured output

```python
from openai import OpenAI
from pydantic import BaseModel
from typing import Literal

class WeatherQuery(BaseModel):
    city: str
    unit: Literal["celsius", "fahrenheit"] = "celsius"

client = OpenAI()
resp = client.beta.chat.completions.parse(  # 注意是 beta.parse
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Extract weather query from: 'How hot is NYC in F?'"}],
    response_format=WeatherQuery,
)
query: WeatherQuery = resp.choices[0].message.parsed  # 已 instantiate Pydantic 对象
print(query.city, query.unit)  # "NYC" "fahrenheit"
```

`response_format=Pydantic` 是 OpenAI 2024 推出的"strict structured output"——后端用 constrained decoding 保证返回的 JSON 100% 满足 Pydantic schema，比手动 parse 健壮得多。Anthropic 的 `tool_use` 在 `disable_parallel_tool_use=True` 下功能等价。

### 6.3 XGrammar 直调（不依赖 vLLM）

```python
import xgrammar as xgr
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer)
compiler = xgr.GrammarCompiler(tokenizer_info)

schema = '{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}'
matcher = xgr.GrammarMatcher(compiler.compile_json_schema(schema))

# 在自定义 generate loop 里：
# 1. logits = model.forward(input_ids)
# 2. mask = torch.zeros(vocab_size); matcher.fill_next_token_bitmask(mask)
# 3. logits[mask == 0] = -inf
# 4. token = sample(logits); matcher.accept_token(token)
```

XGrammar 把 grammar→token mask 编译一次后逐步前进，FSM 加速到 sub-microsecond。生产里很少这样裸调，理解原理用。

---

## 7. MCP（Model Context Protocol）：next-gen tool 协议

Anthropic 在 2024 年提出的开放协议——把 LLM ↔ tool server 之间的接口标准化，类似"LLM 世界的 Language Server Protocol"。

核心思想：tool 不再嵌在每次 prompt 里随调随传，而是由独立的 **MCP Server**（自己进程、自己生命周期）通过标准 JSON-RPC 接口暴露 capability，**任何 MCP-compatible 的 client（Claude Desktop、Cursor、VSCode、Cline 等）都能直接接入**。

与 OpenAI function calling 的对比：

| 维度 | OpenAI function calling | MCP |
|---|---|---|
| 协议 | 厂商私有 | 开放、跨厂商 |
| Tool 部署 | 嵌在 caller 进程 | 独立 server |
| 发现 tool | 写死在 prompt | server 自描述 |
| 权限 | caller 自管 | 协议层有 capability negotiation |
| 复用 | 每个应用自己实现 | 写一次 server，N 个 client 用 |

MCP 还非常新（2024-2025 期），但增长速度极快。技术心智上不是替换 function calling，而是**升一层抽象**——underlying 仍然是 LLM 输出 tool_call JSON、外部执行回传，只不过把"tool 在哪、怎么调用、有什么权限"这些工程问题标准化了。生产系统接 OSS tool 生态时优先看 MCP server 列表（modelcontextprotocol.io 的 servers 仓库），自己实现的 tool 也建议封 MCP server，**未来 3 年可能成为事实标准**。

---

## 8. 工程踩坑与经验

- ❗ **Tool description 是 prompt 的一部分**——花 30 分钟改一个 tool 的 description（加正例 / 负例 / 参数语义）经常比 fine-tune 一周更划算。BFCL 上同 model 同 task 仅 description 不同，准确率能差 10%+
- ❗ **同一 model 的 OpenAI 兼容 API 与 native API 输出 tool_call 格式可能不同**——很多 OSS 部署（如 Together / Fireworks 上的 Qwen）对外暴露 OpenAI 兼容协议，但底层 chat template 用的是 `<tool_call>` 标签，转换层 bug 时会丢字段或格式错。**先 native test，再走兼容层**
- ❗ **Streaming 时 `tool_call.arguments` 是增量 string**——直接 `json.loads(chunk)` 100% 报错。必须等同一 `tool_call_id` 的所有 chunk 拼完再 parse，OpenAI / Anthropic SDK 的 helper 已经处理好，手撕时千万别忘
- ❗ **Parallel tool calls 在小 model / 老 model 上不稳定**——LLaMA-3-8B / Mistral-7B / 早期开源 7B 模型经常并行出错（漏一个、串台、参数污染）。Build-up 阶段用 `parallel_tool_calls=False` 强制 sequential 更安全，待 model upgrade 后再开
- ❗ **Constrained decoding 在长 / 深 nested schema 上性能下降明显**——XGrammar 对 flat schema 几乎零开销，但对 5-6 层嵌套 + 多个 oneOf 的 schema 可能掉 20%+ throughput。schema 设计上**扁平 + enum 限定 > 深嵌套**
- ❗ **Tool 选错时 LLM 自己也不知道**——它会自信地填好参数 return tool_call。caller 必须做 sanity check：tool name 是否在白名单内、required 字段是否齐、enum 是否合法。错了要么返回 error 让 LLM re-plan，要么外层 retry
- ❗ **Multi-turn function calling 时 tool message 的 role 不能写错**——OpenAI 要 `"role": "tool"` + `"tool_call_id": "..."`，Anthropic 要 `"role": "user"` + `content=[{type:"tool_result", ...}]`。写成 `"role": "function"`（OpenAI 旧 API）现在直接 `400`
- ❗ **Function calling SFT 数据格式与推理时格式必须一致**（呼应 8.5）——训练数据用 `<tool_call>...JSON...</tool_call>` 包裹，推理时框架（vLLM / SGLang）必须用同样的 chat template parse 出 tool_call 字段。错位 = 训了等于没训
- ❗ **候选 tool 数 > 20 时调用准确率断崖下跌**——LLM 在 attention 上要"读完所有 tool description 再决策"，candidate 越多越容易混淆。**上层先做 tool routing**（用 embedding 检索 / 关键词预筛 top-k 个候选 tool 再给 LLM 看）是 production 必备
- ❗ **Tool 的 timeout 与 retry 是 caller 的责任**——LLM 不知道你的 API 多久 timeout，要么 caller 端加 timeout + retry，要么把超时结果作为 observation 返回让 LLM 决定 re-plan。**别让 LLM 等 30s 才得到一个失败 observation**
- ❗ **不要把"内部工具"和"用户工具"放同一份 tools 列表**——LLM 容易把内部 admin 工具（如 `delete_all_users`）暴露给用户。production 必须按 caller 身份动态过滤 tool 列表，**最小权限原则在 LLM 时代同样适用**
- ❗ **Tool 返回值过长会污染 prompt 进而拉慢后续 turn**——例如 search tool 返回 50 条结果（每条 500 token）会瞬间塞 25k token 进 history。**caller 端要做 response truncation / summarization**，否则后续每轮 KV cache 都背着这堆垃圾，TTFT 越来越长

---

## 9. Benchmark 速查

- **BFCL（Berkeley Function Calling Leaderboard, Yan et al. 2024）**：最现代的 function calling 评测，覆盖 simple / multiple / parallel / parallel_multiple / irrelevance（不该调时不调） / executable（真正能跑通）等多档。看 model 选型先看 BFCL 排行
- **API-Bank**（Li et al. 2023）：多轮 + 真实 API 调用，更接近 agent 场景
- **NexusRaven V2**：Nexusflow 出品，function calling 训练数据 + 评测一体化
- **ToolBench / ToolLLM**：5000+ 真实 RapidAPI 工具，多步 tool 调用评测

> Tool calling 与 Agent / RAG 的边界：function calling 是 agent 的 **capability 层**（让 LLM 能动手）；RAG 可以包装成一个 `search_documents` tool；现代 agent ≈ function calling + planner（Module 14.4） + memory（14.5）。本节把"调单 tool / 多 tool"的工程基础打牢，14 章直接在此之上构建 multi-step loop。

---

## 10. 经典 paper

- **OpenAI Function Calling 文档**（platform.openai.com/docs/guides/function-calling） — 工业事实标准的官方文档。`tools` / `tool_calls` / `parallel_tool_calls` / `strict` 字段定义、Pydantic 风格的 structured output、streaming 行为都在这里。**所有做 LLM 应用的工程师必读**，因为其他厂商的 API 都在向它对齐
- **Dong et al., 2024 — XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models** — constrained decoding 的 SOTA 引擎。读 §3 的 grammar→token mask 编译思路、§5 的 benchmark（vs Outlines / JSONFormer 的 5-10× 提速），是理解 vLLM / SGLang 默认 backend 行为的钥匙
- **Yan et al., 2024 — Berkeley Function Calling Leaderboard (BFCL) v1/v2/v3** — function calling 评测的现代基准。读 v3 的 multi-turn / multi-step 评估方法、irrelevance 指标的设计——比 paper 更重要的是 **leaderboard 数据**会持续更新，选 OSS model 先看 BFCL
- 加分阅读：**Willard & Louf, 2023 — Efficient Guided Generation for LLMs (Outlines)** 的 §4 FSM 构造；**MCP 官方文档**（modelcontextprotocol.io）

---

## 11. 自测与面试题

**Q1（schema）**：写一个 `get_user_info(user_id, fields)` 的完整 OpenAI tool schema，要求：`user_id` 是 string 必填，`fields` 是字符串数组、可选、枚举值仅限 `name / email / phone / address` 四种。

<details>
<summary>Answer sketch</summary>

```json
{
  "type": "function",
  "function": {
    "name": "get_user_info",
    "description": "Fetch profile fields of a user by user_id. Use this when the user asks 'who is X' or 'show me X's profile'. Do NOT use for billing/payment info — use get_billing instead.",
    "parameters": {
      "type": "object",
      "properties": {
        "user_id": {
          "type": "string",
          "description": "Internal user ID, e.g. 'usr_abc123'."
        },
        "fields": {
          "type": "array",
          "items": {
            "type": "string",
            "enum": ["name", "email", "phone", "address"]
          },
          "description": "Which fields to return. If omitted, returns all 4 fields."
        }
      },
      "required": ["user_id"]
    }
  }
}
```

评分点：
- `description` 写了正例 + 负例（不要用于 billing）
- 每个 property 都有 description
- `fields` 用 array + items.enum 而不是 string + enum（数组场景）
- `required` 只含 `user_id`，`fields` 是 optional（在 description 里说明默认行为）
- 加分：补一句 "Returns 200 + JSON on success, 404 if user not found"，让 LLM 知道 error 形态

</details>

**Q2（实战）**：你 model 经常输出 `{"city": "Beijing"`（缺右括号）这类截断 JSON，给出 3 个不同方向的解决方案，并讨论各自代价。

<details>
<summary>Answer sketch</summary>

三个方向（按工程"侵入性"排序）：

1. **`max_tokens` / 截断检查**：先排查是不是 `max_tokens` 设小了导致 hard cut。response 里看 `finish_reason="length"` 就是这个原因。**几乎零代价**，先排除这个再说
2. **Constrained decoding（XGrammar / Outlines / vLLM guided_decoding）**：从生成层保证 100% syntax 合法。代价：5-20% latency overhead；需要 inference engine 支持（vLLM / SGLang 内置；商业 API 用 OpenAI 的 `strict=True` 或 `response_format=Pydantic`）
3. **客户端容错 parser + retry**：用 `json-repair` / `partial-json-parser` 这类库尝试修复，失败了在 prompt 末尾加 "Your last output was malformed JSON, please retry with valid JSON" 重试一次。代价：retry 翻倍 token 费用、retry 仍可能失败（统计可达 99%+ 但不到 100%）

trade-off：production 关键路径（金额 / 数据库写入）用方案 2 求确定性；探索性 / 内部工具用方案 3 省成本；方案 1 永远先做。

加分：还可以用 **prompt 工程**——在 system 里加 "Output ONLY valid JSON, nothing else" + few-shot 示范一个标准输出，能从 95% 提到 99%；但永远到不了 100%，所以关键路径还是要 constrained decoding 兜底。

</details>

**Q3（trade-off）**：constrained decoding 100% 保证 syntax 合法但有 5-20% latency overhead，从 (产品场景 / schema 复杂度 / 模型可靠性 / 成本) 4 个维度讨论什么时候开、什么时候关。

<details>
<summary>Answer sketch</summary>

按维度分析：

- **产品场景**：写数据库 / 调外部支付 / 发通知这类**有副作用**的 tool，**必开**——一次 syntax 错误的代价（用户投诉 / 数据脏掉）远大于 20% latency。聊天 / 摘要 / 内部 demo **可关**
- **schema 复杂度**：flat schema（< 5 fields，无嵌套），constrained decoding 的 overhead < 5%，**几乎免费 → 开**。深嵌套 / 多 oneOf / 长正则的 schema overhead 可能 30%+，要权衡甚至重构 schema 而不是关
- **模型可靠性**：用 GPT-4o / Claude 3.5 这类强 model，自然 syntax 合法率已经 > 99%，**可以关 + 客户端容错 retry 兜底**；用 7B 开源 model，自然合法率 90-95%，**强烈建议开**——retry 成本 + 用户体验损失大于 latency
- **成本**：constrained decoding **不增加 token 计费**（只 mask logit 不增加生成长度），但增加 GPU latency → 间接增加成本。如果 throughput 是瓶颈（per-GPU QPS 受限）影响明显，如果是 latency-bound（用户等单条）影响小

加分：还有"**hybrid** 策略"——默认关，仅在客户端 parse 失败 retry 时把 constrained decoding 打开，叫做 "fallback constrained mode"。production 推荐方案，平均成本最低。

</details>

---

## 12. 延伸阅读

- [OpenAI Function Calling Guide](https://platform.openai.com/docs/guides/function-calling) — 工业事实标准的官方文档，含 `strict` mode / Pydantic structured output / streaming 行为
- [Anthropic Tool Use Documentation](https://docs.anthropic.com/en/docs/build-with-claude/tool-use) — Claude 的 tool_use block 协议、`disable_parallel_tool_use` 行为、与 OpenAI 差异点对照
- [XGrammar GitHub](https://github.com/mlc-ai/xgrammar) — constrained decoding 引擎，含 vLLM / SGLang / 裸 PyTorch 集成示例
- [Outlines GitHub](https://github.com/outlines-dev/outlines) — Pydantic / regex / JSON schema 全套结构化生成
- [Berkeley Function Calling Leaderboard (BFCL)](https://gorilla.cs.berkeley.edu/leaderboard.html) — 选 model 必看的实时 leaderboard
- [Model Context Protocol](https://modelcontextprotocol.io/) — Anthropic 主导的开放 tool 协议，含 servers 列表与 SDK
- 推荐继续读本教程的 **14.2 节《最小 ReAct + Reflection agent》**——把单次 tool call 升级成多步 agent loop；以及 **8.5 节《SFT 实战：多轮对话 + tool 混合训练》**理解 function calling 的训练侧
