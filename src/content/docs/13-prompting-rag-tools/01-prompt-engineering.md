---
title: "13.1 Prompt 工程：system / few-shot / CoT / role / 结构化输出"
description: "把 prompt 当成一种zero-cost、可量化、可 A/B 的 inference-time intervention——这一节讲清楚 system / few-shot / CoT / role / 结构化输出五件套的算法工程师视角设计原则、各自的失效模式与现代 reasoning model 时代哪些技术过时、哪些反而更重要。"
---

> ⏱ 预计阅读 30 分钟 ｜ 难度 ★★ ｜ 前置：3.2（chat template / 特殊 token）

## 一句话本节讲什么

把 prompt 当成一种**zero-cost、可量化、可 A/B 的 inference-time intervention**——这一节讲清楚 system / few-shot / CoT / role / 结构化输出五件套的算法工程师视角设计原则、各自的失效模式与现代 reasoning model 时代哪些技术过时、哪些反而更重要。

---

## 1. Mental model（直觉）

很多人把 prompt 工程理解为"拼咒语"——这是 LLM **应用工程师**的视角。算法工程师视角不一样：**prompt 是 weights 之外的另一种"训练"**，只不过它发生在 inference time，不更新参数。

把整个建模链条按"成本 / 灵活度 / 可解释性"摆开：

```
              cost              flexibility      interpretability
Pretrain   ████████████████        ★              ★
SFT        ████████                ★★             ★★
RLHF       ██████████              ★★             ★
Prompt     ░  (几乎为 0)           ★★★★★          ★★★★★
```

prompt 工程的独特位置：**几乎零成本、改一行立刻生效、改了什么完全可读**。代价是它**不改 weights**——能力上限被预训练 + post-training 决定。所以 prompt 工程的工作不是"教模型一个新能力"，而是**把模型已经有的能力高效激发出来**：

- 模型预训练时见过百万个"role-play 专家答题"的样本 → 用 role prompting 激发
- 模型见过百万个"先推理再答案"的样本 → 用 CoT 激发
- 模型见过 JSON / API call 这类格式 → 用 structured output 激发

理解了这点，几个常见困惑都能化解：

1. **为什么"You are a helpful assistant" 真的有效**——不是模型"被催眠"，而是 RLHF 把这个 system prompt 当锚点训过
2. **为什么大模型对 few-shot 依赖减小**——SFT/RLHF 做完，模型已经把"按指令做事"内化成默认行为
3. **为什么 reasoning model 不需要再加 "Let's think step by step"**——CoT 已经被 RLVR / long-CoT SFT 烧进 weights

本节聚焦**纯 inference 时的 prompt 设计**，不涉及 prompt-tuning（PEFT 方法，那是 8.4 的内容）。

---

## 2. Prompt 的五件套与设计原则

一个工业级的 prompt 通常包含以下五个组件，按重要性排列：

| 组件 | 作用 | 典型位置 |
|---|---|---|
| **system prompt** | 定义模型角色 / 任务 / 约束 / 输出格式 | message 序列开头 |
| **role / persona** | 让模型进入特定知识空间 | system 内 or user 开头 |
| **few-shot examples** | in-context learning 示范输入-输出 | user 之前或 system 末尾 |
| **chain-of-thought** | 显式引导推理 | system 指令 + few-shot 示例 |
| **structured output** | JSON / XML / 表格等格式约束 | system + 末尾 reminder |

### 2.1 system prompt：协议 + 角色 + 约束

system prompt 在 chat template 下是个独立 role（见 3.2 节），它与 user 消息的本质区别是：**模型在 RLHF 阶段被训练为"高度服从 system 指令"**——这是 OpenAI / Anthropic 的 instruction hierarchy 设计。所以 system 是写 hard constraints 的最佳位置：

```
你是 X 领域专家。请：
1. 用中文回答
2. 输出格式严格符合下面 JSON schema：{...}
3. 不知道的事直接说"我不知道"，不要编造
```

**长度警告**：system prompt 越长，TTFT（time-to-first-token）和 cost 越高。1k token 的 system prompt 在每个请求都要重算 prefill——这就是为什么 Claude / OpenAI 都提供 **prompt caching** 把长 system prompt 的 KV cache 复用（见 11.3 RadixAttention）。

### 2.2 few-shot examples：in-context learning 设计

few-shot 是 GPT-3 [Brown et al. 2020] 论文的核心发现——**模型可以从几个示例里学会一个新任务，不需要 finetune**。设计要点：

- **example 数量**：1-5 个最常用，超过 5 收益急剧递减；CoT example 1-3 个最常用（例子太长占 context）
- **example 选择**：
  - **多样性**：覆盖任务的不同 sub-pattern（适合开放任务）
  - **相似性**：用 kNN retrieval 找与 query 最像的 example（适合 close-ended task，如分类）
- **顺序**：order matters——经验法则是 **easy 在前、hard 在后**（last example 对 model 影响最大，叫 recency bias）
- **format consistency**：所有 example 用**完全相同**的 input / output 格式（包括标点、换行、空格），不一致会让模型 confused

> ❗ 关键：模型越大，对 few-shot 依赖越小。GPT-4 / Claude 3.5 这种级别，zero-shot 已经能解决大多数任务，few-shot 只在**输出格式严格要求**或**边缘 case 多**时才必要。

### 2.3 role / persona prompting

"You are an expert in X..." 这种 role 设定经过多个研究验证（Salewski et al. 2023 等），平均能提升 **5-10% accuracy**，特别是在 domain-specific 任务上。直觉解释：role 让模型从预训练分布中**条件采样到"专家语料"对应的子空间**。

但 persona 是把双刃剑：

- ✅ "You are a senior Python engineer" → 写代码风格更规范
- ❌ "You are a doctor" → 在医学问答上**反而更倾向给具体诊断**（hallucination 风险上升）

工程上的折中：用 role 限定**风格 / 格式 / 视角**，避免用 role 鼓励模型"假装有它没有的能力"。

---

## 3. Chain-of-Thought：从 prompt trick 到内化能力

### 3.1 三种 CoT 范式

**Zero-shot CoT** [Kojima et al. 2022]：在 prompt 末尾加一句魔法咒语：

```
Q: ...
A: Let's think step by step.
```

简单粗暴，对 GPT-3 / PaLM 时代的 base model 提升 GSM8K 准确率 ~17% → ~78%（论文数据）。

**Few-shot CoT** [Wei et al. 2022]：在 example 里**显式展示推理过程**：

```
Q: Roger has 5 tennis balls. He buys 2 more cans of 3 balls each. How many balls does he have now?
A: Roger started with 5 balls. 2 cans of 3 balls each is 6 balls. 5 + 6 = 11. The answer is 11.

Q: <new question>
A:
```

效果比 zero-shot CoT 更稳定，但 example 占 context 多。

**Self-Consistency** [Wang et al. 2022]：CoT 的概率扩展——**采样 N 条不同的推理链，对最终答案做 majority vote**。直觉：正确答案有多条路径能到达，错误答案则各走各的；多次 sample 让正确答案"汇聚"。GSM8K 上 CoT + Self-Consistency 比单 CoT 再涨 10%+。

数学形式：设模型对推理链 $r$ 与答案 $a$ 的联合分布为 $p(r, a \mid x)$，则

$$
\hat{a} = \arg\max_{a} \sum_{r} \mathbb{1}[\text{extract}(r) = a] \cdot p(r, a \mid x)
$$

实现上用 $T > 0$ 多次采样 $N$ 条 trajectory，对答案投票即可。

### 3.2 CoT 在 reasoning model 时代的退场

DeepSeek-R1 / OpenAI o-series / Qwen QwQ 这一代 reasoning model，**CoT 已经被 RLVR（10.3 节）烧进 weights**。它们看到一个数学题会自动产生几千 token 的 long-CoT，**不需要也不应该再 prompt 它"think step by step"**：

- ❌ 加 "Let's think step by step" → 模型可能把这句话当 user 真实需求，**多输出一段冗余 reasoning**，浪费 token 和钱
- ❌ 给 few-shot CoT example → 模型可能模仿 example 的 reasoning 长度，**比自己自由 reasoning 更短，反而损失准确率**
- ✅ 直接问问题，让 model 自己 reasoning

这是一个清晰的**技术迁移**：曾经是 prompt trick 的东西，被收入 weights 后 prompt 上反而要去掉。

### 3.3 CoT 的失效模式

CoT 不是免费午餐。在小模型（< 7B）上 CoT 可能产生 **fake reasoning / rationalization**：模型先猜了一个错答案，然后**编造出看似合理但与答案无关的推理过程**。这种现象在 [Lanham et al. 2023] 的 faithfulness 研究里被系统记录——CoT 的可读性掩盖了它**未必是模型真正的决策路径**。

工程上的应对：**不要把 CoT 当 ground truth 解释**，仍要用最终答案的 metric 评估；reasoning model 时代用 PRM（process reward model，10.2 节）打分中间步骤来缓解这个问题。

---

## 4. 结构化输出：四条路线

让 LLM 输出可被下游程序 parse 的结构化数据，是 agent / pipeline 系统的刚需。四种路线对比：

| 路线 | 实现 | syntax 保证 | 灵活度 | 适用 |
|---|---|---|---|---|
| **1. Prompt 要求** | "Output strict JSON: {...}" | ❌ ~5-10% syntax 错 | 最高 | quick prototype |
| **2. Function calling** | OpenAI / Anthropic API 原生 | ✅（API 端校验） | 中 | 闭源 API |
| **3. Constrained decoding** | XGrammar / Outlines / lm-format-enforcer | ✅ 100% syntax | 中 | self-host 模型 |
| **4. Retry + parser** | 失败重试 + 部分 parse | 视次数而定 | 高 | 兜底方案 |

### 4.1 路线 1：纯 prompt（最弱保证）

```
请按以下 JSON 格式回复，不要包含其他文字：
{"intent": "...", "slots": {...}}
```

简单但**不保证 syntax**——常见错误：多余的 markdown 代码块包裹、漏逗号、字符串里没转义引号、把 JSON 当成 Python dict 输出（False / True / None）。

### 4.2 路线 2：Function calling

OpenAI / Anthropic / Gemini 都在 API 层提供 `tools` / `tool_choice` 参数，模型生成时**API server 会确保符合 JSON schema**。背后实现既可能是 prompt + 后处理（早期），也可能是 constrained decoding（现在）。这是闭源场景的标准方案，详细工程在 13.4 讲。

### 4.3 路线 3：Constrained decoding（self-host 标配）

原理：在每一步 decode 时，**用 schema 编译出"当前位置允许的 token 集合"**，把 logit 上其他 token 全部 mask 成 $-\infty$，softmax 后只能 sample 到合法 token。

主流实现：
- **Outlines**：FSM-based，支持 regex / JSON schema / Pydantic
- **XGrammar**（陈天奇组）：上下文无关文法 + GPU-friendly mask 计算，速度快到对推理 throughput 几乎无损
- **lm-format-enforcer**：vLLM / SGLang 默认集成

这是开源 self-host 场景的最佳方案——**100% syntax 合法**，且 throughput 接近 unconstrained。

### 4.4 路线 4：Retry + parser（兜底）

无论用哪种方案，生产环境**永远要加一层 parser + retry**。常见实现：

```
for attempt in range(3):
    output = call_llm(prompt)
    try:
        result = json.loads(output)
        validate(result, schema)
        return result
    except (JSONDecodeError, ValidationError) as e:
        prompt += f"\n\nPrevious output had error: {e}. Please fix and retry."
```

---

## 5. 最小代码示例

### 5.1 构造 system + few-shot + CoT 的完整 prompt

```python
from openai import OpenAI

client = OpenAI()

SYSTEM = """你是一个数学解题助手。请按以下格式回答：
1. 先用 "Step 1:", "Step 2:" 列出推理
2. 最后一行写 "Answer: <数字>"
"""

FEW_SHOT = [
    {"role": "user", "content": "Q: Roger has 5 balls. He buys 2 cans of 3. How many?"},
    {"role": "assistant", "content": "Step 1: Start with 5 balls.\nStep 2: 2 cans × 3 = 6 balls.\nStep 3: 5 + 6 = 11.\nAnswer: 11"},
    {"role": "user", "content": "Q: A baker had 24 cookies. He sold 1/3. How many left?"},
    {"role": "assistant", "content": "Step 1: 24 cookies total.\nStep 2: Sold 24 × 1/3 = 8.\nStep 3: 24 - 8 = 16.\nAnswer: 16"},
]

query = {"role": "user", "content": "Q: A train leaves at 9am at 60 mph. Another at 10am at 80 mph from same place. When do they meet?"}

resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "system", "content": SYSTEM}, *FEW_SHOT, query],
    temperature=0.0,
)
print(resp.choices[0].message.content)
```

关键点：system 写格式约束，few-shot 用**完全一致**的 "Step k:" / "Answer:" 风格；最后 query 用相同的 "Q: ..." 前缀，让 in-context learning 生效。

### 5.2 OpenAI function calling 结构化输出

```python
from openai import OpenAI
client = OpenAI()

tools = [{
    "type": "function",
    "function": {
        "name": "extract_user_intent",
        "parameters": {
            "type": "object",
            "properties": {
                "intent": {"type": "string", "enum": ["book", "cancel", "query"]},
                "date": {"type": "string", "description": "YYYY-MM-DD"},
                "city": {"type": "string"},
            },
            "required": ["intent"],
        },
    },
}]

resp = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "帮我订 6 月 12 号到上海的票"}],
    tools=tools,
    tool_choice={"type": "function", "function": {"name": "extract_user_intent"}},
)
print(resp.choices[0].message.tool_calls[0].function.arguments)
# {"intent": "book", "date": "2026-06-12", "city": "上海"}
```

`tool_choice` 强制模型必须调用指定函数，是结构化输出最稳定的写法（不然模型可能选择 free-text 回复）。

### 5.3 Outlines constrained generation（self-host）

```python
import outlines
from pydantic import BaseModel
from typing import Literal

class Intent(BaseModel):
    intent: Literal["book", "cancel", "query"]
    date: str  # YYYY-MM-DD
    city: str

model = outlines.models.transformers("Qwen/Qwen2.5-7B-Instruct")
generator = outlines.generate.json(model, Intent)

result = generator("帮我订 6 月 12 号到上海的票，请用 JSON 输出 intent/date/city")
print(result)
# Intent(intent='book', date='2026-06-12', city='上海')
```

Outlines 在 decode 阶段把不符合 schema 的 token logit 置 $-\infty$，**100% 保证 syntax 合法**——不会出现漏引号、漏逗号、enum 取错值的问题。

### 5.4 Self-Consistency 多 sample + vote

```python
from openai import OpenAI
from collections import Counter
import re

client = OpenAI()

def extract_answer(text: str) -> str:
    m = re.search(r"Answer:\s*([\-\d\.]+)", text)
    return m.group(1) if m else None

def self_consistency(prompt: str, n: int = 8, model: str = "gpt-4o-mini"):
    answers = []
    for _ in range(n):
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,  # 必须 > 0 才能多样
        )
        a = extract_answer(resp.choices[0].message.content)
        if a is not None:
            answers.append(a)
    if not answers:
        return None
    return Counter(answers).most_common(1)[0][0]

print(self_consistency("Q: ... Let's think step by step. End with 'Answer: <number>'"))
```

关键：`temperature > 0` 才能采样到不同推理路径；$n=5\sim20$ 是经验甜点（GSM8K 类任务 $n=20$ 后收益饱和）。代价是 token 成本 $\times n$。

---

## 6. 工程踩坑与经验

- ❗ **Few-shot example 数量超过 5 收益递减**——尤其 CoT example 长，3 个左右最常用。example 比 query 长 10 倍以上时，要考虑改 SFT 而不是堆 example
- ❗ **Long context 时 important info 放在开头或结尾**——[Liu et al. 2023] "Lost in the Middle" 实证：长 context 中间位置的信息 retrieval 准确率比首尾低 20-30%。RAG 把最相关 chunk 放最后是工业标准
- ❗ **JSON 输出不加 constrained decoding 时，syntax 错误率 5-10%**（GPT-4 级别约 1-3%，开源 7B 模型可达 15%+）。生产环境必须配 schema validator + retry，或直接上 function calling / Outlines
- ❗ **CoT 在小模型（< 7B）上可能 fake reasoning**：模型先猜答案再编理由（rationalization），CoT 看似合理但与决策无关。评估别只看 reasoning 流畅度，要看 final answer accuracy
- ❗ **Reasoning model（R1 / o1 / QwQ）prompt 时不要再加 "let's think step by step"**——模型会把这当成 user 真实需求多输出冗余 reasoning，token 翻倍。直接问问题
- ❗ **System prompt > 2k token 显著增加 TTFT 与 cost**——每次请求都要重算 prefill。用 prompt caching（Anthropic / OpenAI 都有原生 API；vLLM / SGLang 用 RadixAttention 自动缓存）能省 90%+ 重复 cost
- ❗ **Prompt 优化必须 quantify**：测 100+ sample 的 metric，不要凭单 case 拍脑袋"这个 prompt 比那个好"。单 case 的差异基本都在 model variance 内，统计上无意义
- ❗ **A/B 两个 prompt 至少跑 100+ sample**，并固定 seed / temperature；理想情况算 paired bootstrap 置信区间，避免被 1-2 个 outlier 误导
- ❗ **Few-shot example 顺序影响显著**：尤其是 last example 的 recency bias。如果发现模型输出风格"漂"向某个 example，把它移到中间
- ❗ **Format drift**：长 generation 中模型可能慢慢偏离指定格式（如 markdown 渐变成 plain text）。在 system 末尾再 reminder 一次格式 + 用 stop token 强制截断，能显著缓解
- ❗ **Tool / Agent 场景 prompt 里别用太多 markdown 列表**——模型可能把这种结构当成 output template 模仿，导致 tool call 输出多余的 markdown，function calling parser 失败

---

## 7. 经典 paper

- **Brown et al., 2020 — Language Models are Few-Shot Learners (GPT-3)** — in-context learning 的奠基论文，第一次系统证明大模型可以从 prompt 里几个示例学会新任务，不需要 finetune。读 §3 的 zero/one/few-shot 对比 + §4 的 task suite，理解 few-shot 为什么 work。
- **Wei et al., 2022 — Chain-of-Thought Prompting Elicits Reasoning in Large Language Models** — Few-shot CoT 的开创论文，证明显式推理示例能让 LLM 解决之前做不了的多步推理。这篇 + Kojima 2022 是 reasoning prompt 方向的双子星。
- **Kojima et al., 2022 — Large Language Models are Zero-Shot Reasoners** — Zero-shot CoT 的 "Let's think step by step" 来源，证明 zero-shot 也能激活推理能力。读完会理解为什么这一句话成了 prompt 工程史上最著名的咒语。
- 加分：**Wang et al., 2022 — Self-Consistency Improves Chain of Thought Reasoning** — CoT 的 majority vote 扩展，也是 test-time scaling 的早期形态（10.4 节会展开）；**Khattab et al., 2023 — DSPy** 把 prompt 当 program 编译，是 prompt 优化方向最有影响力的工作。

---

## 8. 自测与面试题

**Q1（设计）**：写一个完整的 system + few-shot + CoT prompt 用于解 GSM8K 数学题，要求包含 2 个 example，并指定输出末尾必须有 "Answer: <number>"。

<details>
<summary>Answer sketch</summary>

完整 prompt 至少包含三部分：

**System**：
```
You are a math problem solver. For each question:
1. Write step-by-step reasoning prefixed with "Step k:"
2. End with a single line "Answer: <number>" (no units, no extra text)
```

**Few-shot example 1**（简单算术）：
```
Q: Tom has 3 apples. He buys 5 more. How many apples?
A: Step 1: Tom starts with 3 apples.
   Step 2: He buys 5 more, so total = 3 + 5 = 8.
   Answer: 8
```

**Few-shot example 2**（多步、含分数）：
```
Q: A class has 30 students. 2/5 are girls. How many boys?
A: Step 1: Total students = 30.
   Step 2: Girls = 30 × 2/5 = 12.
   Step 3: Boys = 30 - 12 = 18.
   Answer: 18
```

**关键设计点**：
- 两个 example 难度递增（Q1 单步算术、Q2 多步含分数），覆盖 GSM8K 的常见 pattern
- 格式严格一致（同样的 "Step k:" 与 "Answer:" 风格）
- system 显式约束 "no units, no extra text" 以方便下游正则提取
- 加分：把 query 也用 "Q: ... A:" 包裹保持一致；用 temperature=0 + Self-Consistency 进一步提升准确率

</details>

**Q2（trade-off）**：让 LLM 输出 JSON 的 4 种方法（prompt 要求 / function calling / constrained decoding / retry+parser）各自的优缺点与适用场景？

<details>
<summary>Answer sketch</summary>

按 4 个方法逐条分析（核心维度：syntax 保证 / 灵活度 / 工程成本 / 适用场景）：

| 方法 | 优点 | 缺点 | 适用 |
|---|---|---|---|
| **Prompt 要求** | 0 工程成本、改 prompt 即可 | 5-10% syntax 错率，开源小模型更高 | quick prototype、内部工具 |
| **Function calling** | API 原生支持、稳定、含 schema 校验 | 仅闭源 API；schema 表达力有限 | 闭源生产、agent 工具调用 |
| **Constrained decoding** | 100% syntax 合法、对 throughput 影响小 | 需 self-host、复杂 schema 编译可能慢；可能让 model 输出"凑合法的废话" | 开源 self-host、严格格式场景 |
| **Retry + parser** | 任何方案都能叠加、兜底鲁棒 | 增加 latency 与 cost；多次失败会 fall back 到错误数据 | 生产环境必备的最后一道防线 |

**最佳实践组合**：
- **闭源生产**：function calling + retry parser
- **开源 self-host**：constrained decoding（Outlines / XGrammar）+ retry parser
- **快速 demo**：prompt + retry parser

加分：constrained decoding 有个 subtle 失效——**model 被强制只能输出合法 token 时可能产生"语义不对但 syntax 对"的输出**（如全选 enum 的第一个值）。所以即使用了它，仍要在 evaluation 阶段验证 semantic 正确性。

</details>

**Q3（前沿）**：reasoning model（R1 / o1 / QwQ）出现后，prompt 工程哪些技术变得不再必要？哪些反而更重要？

<details>
<summary>Answer sketch</summary>

**变得不再必要**：

- **Zero-shot CoT 咒语**："Let's think step by step" 在 R1/o1 上反而会**触发额外冗余 reasoning**，浪费 token 和钱。reasoning model 的 long-CoT 已经被 RLVR 烧进 weights
- **Few-shot CoT example**：模型可能模仿 example 的 reasoning 长度，限制自己自由 reasoning 的 budget；很多 reasoning model 官方文档明确建议**不要给 CoT example**
- **复杂的 reasoning 引导话术**："think carefully" / "consider step by step" / "verify your answer" 等都被内化
- **Self-Consistency 的部分价值**：reasoning model 内部已经在做"多路径推理 + 自我验证"，外层再投票收益变小（但仍有效，只是 cost-benefit 不如以前划算）

**反而更重要**：

- **清晰简洁的 task description**：reasoning model 对 system prompt 中"任务定义模糊"特别敏感——它会用 long-CoT 反复猜测意图。prompt 要 specific
- **结构化输出约束**：reasoning model 的 output 通常很长（thinking + answer），下游解析压力变大，function calling / constrained decoding 更必要
- **答案位置与格式约束**：明确告诉模型"最终答案放在 \boxed{} 里" / "用 JSON 末尾包裹"，避免 long output 中答案难以提取
- **token budget 控制**：reasoning model 经常会"想太多" → 显式给 max_tokens for thinking 或者用 "respond concisely" 引导（OpenAI 已开放 reasoning_effort 参数）
- **避免 prompt injection 反思**：reasoning model 容易把 user prompt 当 system 真实意图重新解读，prompt injection 攻击面更大；system prompt 里加 instruction hierarchy 防御更重要

加分：reasoning model 时代的 prompt 工程从"教 model 怎么 think"变成"告诉 model 你要什么"——**指令清晰度比推理引导更关键**。这与 SFT/RLHF 的发展规律一致：能力被烧进 weights，prompt 退化为 task spec。

</details>

---

## 9. 延伸阅读

- [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering) — 官方 prompt 设计 best practice，覆盖 GPT-4o / o1 等模型差异
- [Anthropic Prompt Engineering Docs](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview) — Claude 的 prompt 模式与 XML tag 风格的工业级参考
- [DSPy 官方文档](https://dspy.ai) — 把 prompt 当 program 编译的现代框架，prompt 自动优化方向必看
- [Outlines](https://github.com/dottxt-ai/outlines) — constrained decoding 的工业级开源实现
- [XGrammar](https://github.com/mlc-ai/xgrammar) — 陈天奇组的高性能 grammar-constrained generation
- [Lilian Weng — Prompt Engineering Blog](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/) — OpenAI 前研究员的系统综述，含大量 paper 索引
- 推荐继续读本教程的 **13.2 节《RAG 基础》**——把外部知识注入 prompt 的最重要方法，与本节的 in-context learning 一脉相承
