---
title: "15.1 Agent SFT：FireAct / Agent-FLAN / AgentTuning / ToRA"
description: "Agent SFT 是 Module 15 整个 Agent RL 章节的入口——它在 8.5 / 14.3 的\"单步 tool call SFT\"之上，加一个新维度：完整 trajectory 监督（query → thought → action → obs → thought → ... → final answer），让 model 学会\"多步规划 + 错误恢复 + 何时停\"这套 plan"
---

> ⏱ 预计阅读 50 分钟 ｜ 难度 ★★★ ｜ 前置：14.3（Tool Use 训练范式）、8.5（多轮 + tool 混合 SFT）

## 一句话本节讲什么

Agent SFT 是 Module 15 整个 Agent RL 章节的**入口**——它在 8.5 / 14.3 的"单步 tool call SFT"之上，加一个新维度：**完整 trajectory 监督**（query → thought → action → obs → thought → ... → final answer），让 model 学会"多步规划 + 错误恢复 + 何时停"这套 plan-and-execute 行为；FireAct / Agent-FLAN / AgentTuning / ToRA 是 2023-2024 这一范式的四种代表风格，本节梳理它们的设计差异、给出 trajectory 数据 unify + loss mask 的工程实现，并把"什么时候 SFT 够、什么时候必须 RL"这条与 15.2 衔接的边界画清楚。

---

## 1. Mental model（直觉）

8.5 训了"看到 user query 输出一个 `<tool_call>` + 拿到 obs 给 final answer"——**只覆盖单步 plan-and-act**。14.3 把"tool 数据本身怎么造"讲清楚了，但落到训练范式仍偏单 turn。本节回答上一层问题：**当任务需要 5-20 步的多次 tool 调用（边走边想、走错回退）时，model 是怎么 SFT 训出来的？**

把 Agent SFT 想成"trajectory 模仿学习"——给一段 expert 的完整做题过程（含每一步的思考、每一次工具调用、每一次工具返回、最终答案），让 model 端到端地把这条 trajectory 整段学下来：

```
   普通 SFT (chat / 8.5 single-step)
   ────────────────────────────────
   (query, response) 二元组
   loss 只对 response 段
   监督粒度：1 个 turn

   Agent SFT (FireAct / Agent-FLAN / ToRA / AgentTuning)
   ────────────────────────────────
   trajectory:
       query
       thought_1 → action_1 → obs_1
       thought_2 → action_2 → obs_2
       ...
       thought_T → final_answer
   loss 对所有 thought + action 段（每个 assistant turn 都监督）
   监督粒度：T 个 turn
```

四个代表方法的核心 framing：

- **FireAct (2023.10, Princeton)**：第一篇正经做 ReAct trajectory SFT。核心 insight：**用 GPT-4 生成 ReAct / CoT / Reflexion 三种风格的 trajectory 混合训**，比单一风格 SFT 涨 5-10%——多 prompting style 是隐式数据增广
- **Agent-FLAN (2024, Shanghai AI Lab)**：把 agent 能力解耦成 **reasoning / format / tool-use 三任务**，各自 SFT 后 mix。**强制要求 negative example**（"不该调 tool 时怎么办"），是工程范本
- **AgentTuning (2023.10, THU)**：**6 个 agent task 数据合成 SFT**（ALFWorld / WebShop / Mind2Web / DataBase / OS / KG），证明跨 domain agent 数据混训能让 LLaMA-2 在某些 agent 任务上接近 GPT-3.5
- **ToRA (2023.09, MSRA + THU)**：**Tool-integrated Reasoning** 的开端——math reasoning 与 Python interpreter 交错。核心贡献是**output-space shaping**：让 model 学会"何时纯 reasoning、何时 emit code、何时停"这套混合输出协议

读者看完本节后，遇到任何"我要 SFT 一个能多步调 tool 的 agent" 应该立刻能回答：(1) trajectory 数据从哪来、什么格式、(2) loss mask 怎么写（哪段算、哪段 mask）、(3) 4 种数据 mix 风格选哪种、(4) 什么时候 SFT 够、什么时候上 RL。

> 与 8.5 区分：8.5 是 single-step tool call SFT，本节是 multi-step trajectory SFT。
> 与 14.3 区分：14.3 讲 tool 数据"是怎么造的"（Toolformer / Gorilla / xLAM 数据合成范式），本节讲拿到 multi-step trajectory 后**怎么端到端训**。
> 与 15.2 区分：15.2 用 PPO/GRPO 在 trajectory 级别做 RL，本节是它的**冷启**——SFT 让 model 先会 format + 基础 plan，RL 再 fine-tune 决策边界。

---

## 2. Agent SFT 与普通 SFT 的核心差异

### 2.1 数据形态：trajectory vs (query, response)

普通 chat SFT 的训练样本是 `(query, response)` 二元组；agent SFT 的训练样本是一条 **trajectory**：

$$\tau = \big(q,\ (t_1, a_1, o_1),\ (t_2, a_2, o_2),\ \ldots,\ (t_T, a_T, o_T),\ y\big)$$

其中 $q$ 是初始 query、$t_i$ 是第 $i$ 步的 thought（reasoning trace）、$a_i$ 是 action（tool call 或 final answer）、$o_i$ 是 observation（环境/工具返回）、$y$ 是最终答案。$T$ 是步数，典型值 3-15，长 trajectory 可达 30+。

把 trajectory 摊平成 token 序列，长度通常是 5k-50k token——这是 agent SFT 与普通 SFT 第一个工程差异：**单条样本极长**，对 sequence packing 与显存策略压力大。

### 2.2 监督粒度：每个 thought / action 都算 loss

对 $\tau$ 而言，**$t_i$ 与 $a_i$ 都是 model 应当生成的内容**，所以所有 $t_i, a_i, y$ 段都进 loss——loss 是 $T$ 个 assistant turn 的 token-level cross-entropy 之和：

$$\mathcal{L}(\tau) = -\sum_{i=1}^{T} \sum_{x \in t_i \cup a_i} \log p_\theta(x \mid \text{prefix})\ -\ \sum_{x \in y} \log p_\theta(x \mid \text{prefix})$$

而 $q, o_1, \ldots, o_T$ 段全部 mask（$\text{label} = -100$）——这些是用户输入与环境返回，不是 model 生成的内容。这套 loss schema 是 8.5 §4 的"all-turns + completion-only" 在 multi-step 下的自然推广。

### 2.3 与 14.3 tool use SFT 的差别

- **14.3 单步 tool call 训练强调"调对 tool"**：(query, tool_call, obs, answer) 4 段，loss 在 tool_call + answer 上
- **Agent SFT 强调"全程 trajectory"**：可能 5-20 步循环，每步都有 thought + action，loss 在所有 thought + action + final answer 上
- 工程区别最大的是 **trajectory 长度**——14.3 数据 typical 1k-3k token，agent SFT 数据 typical 5k-50k token，packing / FA / context window 都更紧张

### 2.4 与 RLHF 的差别

普通 RLHF（Module 9）的 reward 在**最后一步**给（reward model 看完整 response 打一个分）；agent SFT 也只在最后一步给监督信号（final answer 是否对），但**所有中间 thought / action 也算 loss**——这是 imitation learning 的特性，比 RL 的 sparse reward 训起来快得多。这也是为什么 SFT 要先于 RL：把 trajectory 模式先 imitation 学进权重，RL 再去 fine-tune 决策。

---

## 3. 四种代表方法详解

### 3.1 FireAct (Chen et al., 2023.10)

FireAct 是第一篇正经做 ReAct trajectory SFT 的工作（在 HotpotQA / Bamboogle 等 multi-hop QA 上）。两个核心 insight：

**Insight 1：用 GPT-4 生成 trajectory 比人写好得多。**FireAct 用 GPT-4 + ReAct prompt 在 HotpotQA 上批量生成 trajectory，过滤掉 final answer 错的，剩下的就是 SFT 数据。这套"strong LLM 蒸馏 trajectory"成为后续所有 agent SFT 工作的标配。

**Insight 2：多种 prompting style 混合 SFT 比单一 style 涨 5-10%。**FireAct 同时用 **CoT**（纯思考无工具）、**ReAct**（thought + action 交错）、**Reflexion**（错了之后反思重试）三种 prompt 让 GPT-4 生成不同 style 的 trajectory，再 mix SFT。结果：

| Setup | EM on HotpotQA |
|---|---|
| ReAct only SFT | ~31% |
| CoT only SFT | ~27% |
| Reflexion only SFT | ~30% |
| **三者 mix SFT** | **~37%** |

为什么 mix 涨这么多？**多 style 数据是隐式数据增广**——同一个问题用不同 reasoning template 解一遍，model 学到的不是"模仿 ReAct format"，而是"在 X 类问题上选择合适的 reasoning style"，泛化更好。这条经验后来在 Agent-FLAN / xLAM 等工作上反复被验证。

FireAct 的局限：只验证了 multi-hop QA 这一类任务，工具种类少（基本只有 search），trajectory 长度短（typically 3-5 步）。但它确立了"trajectory SFT + GPT-4 蒸馏 + 多 style 混合"的范式，是后续工作的祖师爷。

### 3.2 Agent-FLAN (Chen et al., 2024)

Agent-FLAN（Shanghai AI Lab）的命名借了 FLAN（Finetuned Language Net）的思路，把 agent 能力**显式拆解成正交任务**分别训。三个核心拆解：

| 子任务 | 训练目标 | 数据来源 |
|---|---|---|
| **Reasoning** | 学 ReAct / CoT 的思考链路 | HotpotQA / GSM8K trajectory |
| **Format** | 学 `<tool_call>` JSON / 输出协议 | 合成的 format 专项数据 |
| **Tool use** | 学 tool selection + 参数填写 | ToolBench / API-Bank trajectory |

Agent-FLAN 发现**三个任务的训练数据如果合在一起 mix SFT**，效果不如**先各自 SFT 一遍再 mix**——原因：reasoning 与 format 在样本上下文里是 entangled 的，model 容易把"思考"学成"模仿 format"。拆开训能让 model 在每个能力维度上都打牢，再混合时不互相干扰。

**核心贡献：negative example 必须含。**Agent-FLAN 在数据中显式加入 **negative trajectory**——主要两类：

1. **Refuse-when-no-tool**：system 给 tool list 但 user query 不需要 tool，assistant 应直接 chat 回答（与 8.5 §5.3 / 14.3 §9 呼应）
2. **Hallucinated tool 防御**：user 要求调一个不在 list 的 tool，assistant 应解释 tool 不可用而非乱调

Agent-FLAN ablation 显示：去掉 negative example 后 model 在 held-out task 上的 over-call 率从 5% 暴增到 30%——见到任何 query 都尝试 emit `<tool_call>`。这是工程范本：**任何 agent SFT 数据集 negative case 占比应在 10-20%**。

### 3.3 AgentTuning (Zeng et al., 2023.10, THU)

AgentTuning 是**第一个跨 domain 大规模 agent SFT** 的工作。核心问题：单个 agent task（如只 ALFWorld）SFT 出的 model，换到另一个 task（如 WebShop）上几乎是零分；能不能用多任务混训让 model 学到 "通用 agent 能力"？

AgentTuning 的 **AgentInstruct 数据集**包含 6 个 agent task（每个 GPT-4 蒸馏 200-1000 条 trajectory）：

| Task | 类型 | trajectory 长度 |
|---|---|---|
| **ALFWorld** | 文字版 embodied household task | 短-中（5-15 步） |
| **WebShop** | 模拟在线购物 | 中（10-20 步） |
| **Mind2Web** | 真实网页操作 | 短（3-8 步） |
| **DataBase** | SQL query 数据库交互 | 短（2-5 步） |
| **Operating System** | shell command 执行 | 中（5-10 步） |
| **Knowledge Graph** | KG query 与遍历 | 中（5-15 步） |

把这 6 个 task 的 trajectory + 通用 chat 数据按 **agent : chat ≈ 1 : 4** 混合 SFT LLaMA-2-7B/13B/70B。结果在 6 个 held-in task 上有大幅提升，而通用 chat 能力（MMLU、MT-Bench）几乎没有 regression——证明了**"少量高质量 agent 数据 + 大量通用 chat 数据"是平衡 agent 能力与通用能力的可行配比**。

更让社区震动的是**held-out 泛化**：在训练完全没见过的 agent task（如 SciWorld）上，AgentTuning-LLaMA-2-70B 直接 zero-shot 接近 GPT-3.5。这给了一个强信号：agent SFT 的"通用能力"是可迁移的，不需要为每个新 task 都重新训。

AgentTuning 的局限：trajectory 都是 happy path（GPT-4 当 teacher，错的过滤掉），缺少 error recovery 与 negative case。这部分 gap 由 Agent-FLAN 补上。

### 3.4 ToRA (Gou et al., 2023.09)

ToRA（**To**ol-integrated **R**easoning **A**gent）专攻 **math reasoning + Python interpreter** 的整合。背景：纯 CoT 解 GSM8K / MATH 的 model 在符号计算（大数除法、积分）上经常算错；如果让 model 学会"思考到一半时 emit Python code 让 interpreter 算"，准确率能上一大截。

ToRA 数据格式（math problem 的 trajectory）：

```
Problem: 求 ∫_0^π sin(x)*cos(x) dx 的值。

<reason>
利用恒等式 sin(x)cos(x) = sin(2x)/2，原积分变为：
(1/2) ∫_0^π sin(2x) dx
我手算容易出错，调 Python 验证：
</reason>
<code>
from sympy import integrate, sin, symbols, pi
x = symbols('x')
result = integrate(sin(2*x)/2, (x, 0, pi))
print(result)
</code>
<output>0</output>
<reason>
积分为 0。这符合 sin(2x) 在 [0, π] 上正负面积相消的几何直觉。
</reason>
<answer>0</answer>
```

ToRA 的核心贡献是 **output-space shaping**——让 model 学会用三种 tag 之一（`<reason>` / `<code>` / `<answer>`）作为下一步的输出，等价于在每个 token 决策点上做"三分类 + 内容生成"。这是 2023 年最早把 "reasoning 与工具调用交错"做成 first-class 训练范式的工作之一。

ToRA 数据合成 4 步：
1. 收集 GSM8K / MATH / TabMWP / HMWP 等 math 数据集 ~16k 题
2. 对每题用 GPT-4 + ReAct prompt 生成 reason+code+output trajectory
3. 用 Python 真执行 `<code>` 段，把 `<output>` 替换成真实输出
4. 检查 `<answer>` 是否与 ground truth 一致，对的留下做 SFT 数据

ToRA-Code-7B 在 MATH 上从 LLaMA-2-Code 的 ~18% 涨到 ~45%，在 GSM8K 上从 ~30% 涨到 ~73%——奠定了 "tool-integrated reasoning" 范式，后续 OpenMathInstruct / MathCoder / DeepSeek-Math 都继承了这个思路。

### 3.5 四种方法对比速查

| 方法 | 核心 contribution | 适用场景 | 数据 scale | 局限 |
|---|---|---|---|---|
| **FireAct** | 多 prompt style 混合 SFT | multi-hop QA | 几千-万级 | 工具种类少、trajectory 短 |
| **Agent-FLAN** | reasoning/format/tool 拆解 + negative | 通用 tool agent | 几万级 | 需要细粒度数据标签 |
| **AgentTuning** | 跨 6 task agent 混训 | 多 domain 通用 agent | AgentInstruct 1.8k 高质量 | 缺 error recovery |
| **ToRA** | reasoning + Python interpreter 整合 | math reasoning | 16k math trajectory | 仅 math 域 |

**面试时要能立刻说出哪个方法对应哪个 framing**——这是 agent RL 章节的入门基本功。

---

## 4. Trajectory 数据格式标准化

### 4.1 统一到 ChatML 风格

不论 FireAct / AgentTuning / ToRA / xLAM 的原始 schema 是什么，落到训练前都要 unify 到一个 chat template。Qwen2.5 风格示例：

```
<|im_start|>system
You are a helpful agent. You may call tools listed below.
<tools>...</tools>
<|im_end|>
<|im_start|>user
查询 2024 年北京 GDP 并与上海对比。
<|im_end|>
<|im_start|>assistant
<think>
用户要 2024 年北京和上海的 GDP 数据。先查北京：
</think>
<tool_call>{"name": "search", "arguments": {"query": "北京 2024 GDP"}}</tool_call>
<|im_end|>
<|im_start|>user
<tool_response>北京 2024 年 GDP 约 5.0 万亿元。</tool_response>
<|im_end|>
<|im_start|>assistant
<think>拿到北京数据，再查上海：</think>
<tool_call>{"name": "search", "arguments": {"query": "上海 2024 GDP"}}</tool_call>
<|im_end|>
<|im_start|>user
<tool_response>上海 2024 年 GDP 约 5.4 万亿元。</tool_response>
<|im_end|>
<|im_start|>assistant
<think>已拿到两组数据，可以对比给答案。</think>
2024 年上海 GDP（约 5.4 万亿元）略高于北京（约 5.0 万亿元）……
<|im_end|>
```

注意 **`<think>` 段也算 loss**——这是 ReAct / Reflexion 风格 trajectory 的关键，model 学的不只是"调什么 tool"，更是"为什么调"。

> 不同 base model 的 thought 标签不同：Qwen2.5 用 `<think>...</think>`、Llama-3.1 用 `## Thought:`、Hermes 用 `<reasoning>...</reasoning>`。**目标 model 选定后所有 trajectory 数据都要按它的 thought 标签 unify**，否则 model 学糊。

### 4.2 多 turn loss mask 规则（与 8.5 §4 衔接）

把 trajectory 摊成 token 序列后，每个 token 按 5 类划分 loss：

| Token 类型 | 例子 | 算 loss？ |
|---|---|---|
| System prompt（含 tool def） | `<|im_start|>system\n...<tools>...` | ❌ -100 |
| User query | `<|im_start|>user\n查询 ...` | ❌ -100 |
| Assistant header | `<|im_start|>assistant\n` | ❌ -100 |
| **Assistant thought + tool_call + final answer** | `<think>...</think><tool_call>...</tool_call>` | ✅ **算 loss** |
| `<|im_end|>` after assistant | | ✅ 算 loss（教 model 学会停） |
| Tool observation（在 user role 内） | `<tool_response>...</tool_response>` | ❌ -100 |

**最容易出错的是把 `<think>` 段 mask 掉**——一些 ReAct agent SFT 实现里把 `<think>` 当成"reasoning trace 不重要"忽略，结果 model 推理时根本不会输出 thought，直接 emit tool_call，准确率掉 5-10%。**`<think>` 段必须算 loss**，否则 ReAct 模式根本训不进去。

---

## 5. Trajectory 来源：4 种合成路径

agent SFT 数据极度稀缺（人工写一条 multi-step trajectory 成本极高，市场价 5-20 USD/条）。社区主流是 4 种自动合成路径：

### 5.1 Strong LLM 蒸馏

最常见做法。用 GPT-4 / Claude / DeepSeek-V3 在目标环境里跑 ReAct prompt，把成功的 trajectory 留下做 SFT 数据。FireAct / AgentTuning / ToRA 全用这种方法。

- 优点：质量高（teacher model 强），快
- 缺点：每条 trajectory 几次 LLM call，cost 0.1-2 USD/条；用 GPT-4 做 commercial agent 有 ToS 风险

### 5.2 Self-Instruct for Agent

让 base model 自身在环境里探索，用 verifier filter 出成功的 trajectory。代表：xLAM 后期、Agent-R 等。

- 优点：无外部依赖、cost 低
- 缺点：base 弱时探索成功率太低，需要先 GPT-4 cold start

### 5.3 Real environment trace

从已部署的 agent（如 production 客服 agent）收集真实 trajectory，人工标注成功/失败。代表：Sierra τ-bench 部分数据。

- 优点：与真实分布一致
- 缺点：production 才有，研究复现门槛高

### 5.4 MCTS / search-based

用 MCTS（Monte Carlo Tree Search）/ DFSDT 在解空间搜索，把搜到的高 reward 路径作为 trajectory。代表：14.3 讲过的 ToolLLaMA DFSDT、AgentQ、ReST-MCTS\*。

- 优点：能在 base 能力之上构造比 base 更好的 trajectory（"改进"，不只是"模仿"）
- 缺点：每条 trajectory 搜索 cost 4-10× 普通蒸馏

**实务最优组合**：先用 strong LLM 蒸馏拿 1k-10k 高质量 cold start → 再用 self-instruct 或 MCTS 扩到 50k-100k → SFT 完上 RL。

---

## 6. 现代 Agent SFT 数据集速查

工程上能直接拉来训的开源 agent trajectory 数据集（按主流程度排）：

| 数据集 | 来源 | 规模 | 特点 |
|---|---|---|---|
| **AgentInstruct** | THU AgentTuning | 1.8k 高质量 trajectory | 跨 6 task，质量优先于量 |
| **xLAM-DPO** | Salesforce | 60k+ | 含 preference 对，可 SFT 也可 DPO |
| **ToolBench-trajectory** | OpenBMB | 12k+ | DFSDT 生成的 multi-step trajectory |
| **APIGen** | Salesforce 2024 | 60k | 函数调用为主，单 turn 居多 |
| **Hermes-FC-V1** | NousResearch | 10k+ | ChatML 已成型，含 negative |
| **ToolACE** | Liu et al. 2024 | 11k | 显式按维度配比合成 |
| **OpenAct / WebArena trace** | 各家收集 | 几百-千级 | Web agent 专项 |

**怎么选**：
- 通用 multi-step agent → AgentInstruct + xLAM-DPO 主力
- math reasoning → ToRA 数据 + OpenMathInstruct
- tool-heavy → ToolBench-trajectory + ToolACE
- web agent → Mind2Web + WebArena trace

---

## 7. 代码示例

下面三段代码涵盖 agent SFT 工程的三个最关键环节。

### 7.1 Agent trajectory 数据 unify 到 ChatML

```python
# unify_trajectory.py
import json

def unify_agent_trajectory(example, target="qwen"):
    """把 FireAct / AgentTuning / ToRA 等 trajectory 统一成 Qwen ChatML messages。"""
    # 期望 input：example 含 'query', 'steps' (list of dict), 'final_answer'
    # 每个 step: {"thought": str, "tool_call": {...}, "observation": str}
    msgs = [{"role": "system", "content": example.get("system", "You are a helpful agent."),
             "tools": example.get("tools", [])}]
    msgs.append({"role": "user", "content": example["query"]})
    for step in example["steps"]:
        # assistant turn = thought + tool_call
        asst_content = f"<think>\n{step['thought']}\n</think>\n"
        if step.get("tool_call"):
            asst_content += f"<tool_call>{json.dumps(step['tool_call'], ensure_ascii=False)}</tool_call>"
        msgs.append({"role": "assistant", "content": asst_content})
        # 每个 tool_call 对应一个 obs（在 Qwen 里 obs 用 user role 包裹）
        if step.get("observation") is not None:
            msgs.append({"role": "user",
                         "content": f"<tool_response>\n{step['observation']}\n</tool_response>"})
    # 最终 answer turn（也是 assistant，含 think + answer）
    msgs.append({"role": "assistant",
                 "content": f"<think>\n{example.get('final_thought','')}\n</think>\n{example['final_answer']}"})
    return {"messages": msgs}
```

要点：每个 step 拆成两个 message——assistant（thought + tool_call）+ user（tool obs）。final answer 单独一个 assistant message。生产里通常每个数据源单独写一个 adapter（FireAct / AgentInstruct / ToolBench 字段名都不同），这里压缩到 20 行展示主干。

### 7.2 多 turn agent loss mask 实现

```python
# agent_loss_mask.py
def build_agent_labels(messages, tokenizer, max_len=8192):
    """所有 assistant 段（含 <think> + <tool_call> + final answer）算 loss，其余 -100。"""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    ids = tokenizer(text, add_special_tokens=False).input_ids[:max_len]
    labels = [-100] * len(ids)
    # 用 token 序列匹配 assistant boundary
    asst_hdr = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    im_end = tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]
    i, n_hdr = 0, len(asst_hdr)
    while i <= len(ids) - n_hdr:
        if ids[i:i+n_hdr] == asst_hdr:
            j = i + n_hdr                         # assistant 内容起点
            while j < len(ids) and ids[j] != im_end:
                labels[j] = ids[j]                # thought + tool_call + answer 全算
                j += 1
            if j < len(ids):
                labels[j] = ids[j]                # <|im_end|> 算 loss，教停
            i = j + 1
        else:
            i += 1
    return ids, labels
```

要点：与 8.5 §6.3 同套思路扩展到 multi-step——一条 trajectory 有 T 个 assistant turn 都被这个循环正确捕获。**`<think>` 段在 assistant content 内，自动算 loss**——这就是 ReAct trajectory SFT 比单步 tool SFT 涨分的关键。

### 7.3 TRL SFTTrainer agent 配置

```python
# agent_sft_train.py
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

MODEL = "Qwen/Qwen2.5-7B"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",
)
ds = load_dataset("THUDM/AgentInstruct", split="train").map(unify_agent_trajectory)

lora = LoraConfig(
    r=64, lora_alpha=128, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
)
sft_cfg = SFTConfig(
    output_dir="./agent-sft",
    num_train_epochs=3,
    learning_rate=5e-5,                    # agent 数据短小精， lr 略小避免过拟合
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    bf16=True,
    gradient_checkpointing=True,           # 长 trajectory 必开
    per_device_train_batch_size=1,         # 单 trajectory 长，batch 不能大
    gradient_accumulation_steps=32,        # effective bs=32
    max_seq_length=16384,                  # agent trajectory 比 chat 长 5-10×
    packing=True,                          # 长度方差大时尤其重要
    completion_only_loss=True,
    save_strategy="epoch",
    logging_steps=10,
)
trainer = SFTTrainer(model=model, args=sft_cfg, train_dataset=ds,
                     peft_config=lora, processing_class=tok)
trainer.train()
```

要点：与 8.5 §6.1 的关键差别——`max_seq_length=16384`（trajectory 长）、`r=64`（agent SFT 比纯 chat 需要更大表达力）、`per_device_train_batch_size=1`（单样本就把显存吃满）、`num_train_epochs=3`（agent 数据通常较少，多刷几轮）。

---

## 8. Agent SFT 与 Agent RL 的衔接（Module 15 主线）

### 8.1 SFT 是 Agent RL 的冷启

DeepSeek-R1（Module 10.3）给出的范式 — **少量 SFT cold start + 大量 RLVR** — 在 agent 场景同样成立：

- 直接从 base 上 RL，几乎不收敛——多步 trajectory 的探索空间是 $|V|^L$（vocab × trajectory 长度），随机探索撞中正样本概率近似 0
- SFT 让 model 先学会基本格式（`<think>` / `<tool_call>` 协议）+ 基础 plan template，RL 阶段从 30-50% 起步，比从 0% 起步效率高 10-100×
- 工程典型配比：**SFT ~10k-100k trajectory + RL 几千 step**

### 8.2 SFT 够还是要 RL？

决策树（呼应 8.5 §8.2 / 14.3 §11 Q3，但 agent 维度更具体）：

**SFT 还能涨分的情况，先继续 SFT**：
- BFCL / 自定义 eval 失败集中在某种**未见过的 pattern**（如新 tool、参数嵌套）→ 补针对性 trajectory
- 增加 10× SFT 数据仍能涨 1-2 分 → 数据 scale 还没饱和
- trajectory 长度短（< 5 步），SFT 已能覆盖大部分决策点

**必须上 Agent RL 的情况**：
- 失败是**系统性的"探索-反馈"问题**：model 调失败后不会重试 / 不会换 tool / 不会反问用户 → trajectory-level reward 才能教
- 有可执行 reward signal（task success / unit test pass / exec accuracy）
- multi-turn planning（trajectory > 10 步），SFT 数据无法穷举所有决策路径
- SFT 已饱和（数据 ×10 不再涨分）

**80% → 90% 是常见的 SFT-to-RL 交接点**——这是 Agent-R1 / Search-R1 / ReSearch 等工作的反复验证（Module 15.4 详讲）。实务最优：**先 SFT 到 70-80% → 再 RL fine-tune 5-15 个点**。

---

## 9. 工程踩坑与经验

- ❗ **Trajectory 数据 quality 决定 model 上限，差数据训不出 good agent**——agent SFT 与普通 chat SFT 的关键差别在于：差 trajectory 不是"答得不好"，而是"教 model 错误的决策模式"。一条 GPT-4 生成的 trajectory 中间 tool call 错了但 final answer 蒙对，被留下做 SFT 数据，model 学的是"调错 tool 也能糊弄过去"。**铁律**：合成 trajectory 必须有 verifier 严格 filter（final answer 与 GT 一致 + 中间 tool call schema 合法 + obs 真实可重现），宁可 1k 高质量也不要 100k 含噪
- ❗ **Loss mask 错位（user / obs 段算 loss）→ model 学复读 user**——这是 agent SFT #1 高频 bug。如果不小心把 tool obs 段（`<tool_response>{...}`）算了 loss，model 推理时遇到 `<tool_response>` 后会把 obs 内容原样吐一遍再回答，产生荒谬输出。**修复**：训练前 dump 一个 batch print labels 肉眼校验，确认 user / obs 段全是 -100、assistant + `<think>` + tool_call + final 段全是 token id（参考 8.2 §3.3 的 visual check 函数）
- ❗ **Trajectory 长度需要 truncate 时优先保留 assistant 段，丢早期 obs**——15 步 trajectory 经常超 max_seq_length，简单从尾截断会把 final answer 砍掉，loss 学一半。正确做法：保留 system + user + 所有 assistant turn + 最近 N 轮完整 obs，远期 obs 用 `[truncated]` 占位字符串。这样 model 还是能学到"完整决策链"，只是丢了一些早期环境细节
- ❗ **Negative trajectory（失败的）也要有，但比例 < 30%**——只训 happy path（GPT-4 一次走通）的 model 在真实环境遇到 tool 报错就僵死。需要 mix 进 **error recovery trajectory**——前一步 obs 报错（如 API 超时 / 参数非法），model 学会 retry / 换 tool / 反问用户。Agent-FLAN 的经验是 negative + recovery trajectory 占 15-25%，超过 30% 反而损害 happy path 的纯净度
- ❗ **Agent SFT 后通用 chat 能力可能下降，要 mix 通用数据**——纯 trajectory SFT 出的 model 在 MT-Bench / Arena-Hard 上能掉 0.5-1.0 分（chat 流畅度退化、任何 query 都倾向 emit `<think>`）。社区甜点是 **agent trajectory : chat data = 1 : 4**（与 AgentTuning §3.3 同源），别贪心把 agent 比例推到 50%+。也可以用"先 chat SFT 打底再 agent SFT"的两阶段策略
- ❗ **Strong LLM 蒸馏成本高（每 trajectory 几 USD），用开源 model 替代**——GPT-4 蒸馏 1 万条 multi-step trajectory 成本 ~3000-10000 USD，且有 ToS 风险（OpenAI 禁止用 GPT-4 输出训 competing model）。**替代方案**：DeepSeek-V3 / Qwen2.5-72B / Llama-3.1-405B 做 teacher，schema 一样、能力相当、license 干净。xLAM 后续版本和 AgentTuning 复现都验证了这条路径
- ❗ **`<think>` 段必须算 loss，否则 ReAct 模式根本训不进去**——一些早期实现里把 `<think>` 当成"中间产物不重要"特意 mask 掉，结果 SFT 完 model 推理时跳过 thought 直接 emit `<tool_call>`，准确率掉 5-10%。**`<think>` 是 model 应当生成的内容**，必须算 loss。这条与 14.3 §3.3 "tool description 写法影响 SFT" 是同一类的 mistake——loss 段错位是 SFT 工程最隐蔽也最常见的问题
- ❗ **SFT-only agent 在 OOD task 弱，需 RL 提升 → Module 15.2**——纯 SFT 出的 agent 在训练分布内表现好，换到没见过的 task / tool / domain 上准确率掉一半。这是 imitation learning 的固有局限——model 学的是"模仿 expert trajectory"，没有"探索更好策略"的能力。Agent RL 在 trajectory-level reward 下能让 model 跳出 expert 分布，找到更优解（这正是 Search-R1 / ReSearch / Agent-R1 等 RL agent 比 SFT-only agent 强的关键，Module 15.4 详讲）
- ❗ **Trajectory 数据合成时要平衡 step 长度分布**——只用 short trajectory（3-5 步）训出的 model 遇到 10+ 步 task 容易 early stop（生成几步就 emit final answer）；只用 long trajectory 训的 model 遇到 1-2 步能解的 simple task 也强行多调几次 tool。**修复**：合成数据时显式按 step 数 bucket（3-5 / 6-10 / 11-20 / 20+），每 bucket 占 20-30%

---

## 10. 经典 paper

- **Chen et al., 2023 — *FireAct: Toward Language Agent Fine-tuning*** — agent trajectory SFT 的奠基作。读 §3 数据合成（GPT-4 蒸馏 ReAct + CoT + Reflexion）+ §4 ablation（mix vs single style）。Take-away：理解"多 prompting style 混合 SFT 是隐式数据增广"，agent SFT 范式的祖师爷
- **Chen et al., 2024 — *Agent-FLAN: Designing Data and Methods of Effective Agent Tuning for Large Language Models*** — 工程范本。读 §3 reasoning/format/tool 三任务拆解 + §4 negative example 设计。Take-away：理解"agent 能力可以解耦成正交任务分别训"，以及为什么 negative case 必须占 10-20%
- **Zeng et al., 2023 — *AgentTuning: Enabling Generalized Agent Abilities for LLMs*** — 跨 domain agent SFT 的代表作。读 §3 AgentInstruct 数据集（6 个 task） + §4 跨 task 泛化实验。Take-away：理解 "agent : chat ≈ 1 : 4 混训"配方，少量高质量 agent 数据 + 大量通用 chat 能 hold 住通用能力 + 显著提升 agent 能力
- **Gou et al., 2023 — *ToRA: A Tool-Integrated Reasoning Agent for Mathematical Problem Solving*** — math reasoning + tool 整合的早期范式。读 §3 output-space shaping（reason/code/answer 三 tag）+ §4 训练 setup。Take-away：理解"reasoning 与 tool 调用交错"作为 first-class 训练范式，是后续 Search-R1 / ReTool 的思想源头
- 加分阅读：**Liu et al., 2024 — *xLAM: A Family of Large Action Models*** — agent SFT 工业化的最完整 recipe；**Liu et al., 2024 — *APIGen*** — Salesforce 60k 函数调用数据合成 pipeline

---

## 11. 自测与面试题

**Q1（数据）**：Agent SFT 数据 vs 通用 chat SFT 的核心差异？给 3 个差别。

<details>
<summary>Answer sketch</summary>

至少答出 3 个：

- **数据形态**：通用 SFT 是 `(query, response)` 二元组（1 turn）；agent SFT 是 trajectory `(q, t_1, a_1, o_1, ..., t_T, a_T, y)`，T 个 turn（typical 3-15）
- **监督粒度**：通用 SFT loss 只在 response 段（1 个 assistant turn）；agent SFT loss 在所有 thought + tool_call + final answer 段（T 个 assistant turn 都监督）
- **样本长度**：通用 chat 典型 200-2000 token；agent trajectory 典型 5k-50k token，对 packing / context window / 显存压力大
- **mask 复杂度**：通用 SFT 只 mask user 段；agent SFT 要 mask user + tool obs（在 Qwen 里 obs 也在 user role 内）+ assistant header
- **`<think>` 段是 first-class 监督内容**：agent SFT 必须把 `<think>` 算 loss（教 model 学会 reasoning before action），通用 chat 通常没这个段
- **error recovery / negative case 是数据组成的一部分**：通用 chat 数据基本都是"正确回答"，agent SFT 必须 mix 进 10-20% 的"tool 失败后重试 / 拒绝调 tool"的样本
- **数据合成成本高**：通用 chat 每条几 cent；agent trajectory 每条 0.1-2 USD（GPT-4 蒸馏几次 LLM call）

加分要点：
- 能联系 8.5（单步 tool SFT）和本节（multi-step trajectory SFT）的递进关系
- 能指出 trajectory SFT 是 imitation learning 的经典形式，与 Module 15.2 RL 的探索-反馈范式互补

</details>

**Q2（实战）**：你训一个 ReAct agent，列出 SFT 数据合成 + 训练完整流程。

<details>
<summary>Answer sketch</summary>

完整 8 步 pipeline：

**数据合成阶段**

1. **环境与 tool 定义**：明确 agent 要在什么环境（如 search engine + Python interpreter）、tool 列表（5-20 个）、tool schema 写清
2. **Cold-start trajectory 蒸馏**：从一个 task pool（如 HotpotQA / GSM8K / 自己业务任务）选 1-5k 题，用 GPT-4 / DeepSeek-V3 + ReAct prompt 跑 trajectory；每题尝试 1-3 次取最佳
3. **Verifier filter**：对每条 trajectory 校验：(a) final answer 与 GT 一致 (b) 每个中间 tool_call schema 合法、tool name 在白名单 (c) tool obs 真实可重现（execute 一次对比）；任一 fail 丢弃
4. **多 style + negative 增强**：参照 FireAct 用 CoT / ReAct / Reflexion 三种 prompt 各跑一遍，再加 10-20% negative case（refuse / retry / hallucinated tool 防御）

**训练阶段**

5. **数据 unify**：所有 trajectory 转成目标 model 的 ChatML（如 Qwen2.5），thought 用 `<think>...</think>`，tool call 用 `<tool_call>...</tool_call>`；按 8.5 / 本节 §4 schema
6. **Mix 通用数据防 chat 退化**：trajectory : 通用 chat : reasoning ≈ 1 : 4 : 1（参考 AgentTuning），总量 50k-200k
7. **超参**：base 选 Qwen2.5-7B；LoRA r=64 lora_alpha=128（agent 比 chat 需要更大 r）；max_seq_length=16k；effective bs=32；lr=5e-5 cosine + warmup 5%；2-3 epoch；bf16 + FA2 + grad checkpoint；`completion_only_loss=True`
8. **多维评测**：(a) BFCL（tool calling 准确率）(b) 自定义 multi-step task eval（happy path + recovery + refuse 各 30 条）(c) MT-Bench 看 chat 是否 regression (d) held-out task 测泛化

加分要点：
- 提到 训练前 dump batch print labels 肉眼校验 loss mask 正确（`<think>` / tool_call / final 算、user / obs / header mask）
- 提到 bucket trajectory by step length 让 model 见到长短分布平衡
- 提到 contamination check（不要把 eval 数据漏进训练）
- 提到 Stage 4 决策——eval 完看是否 80%+ 且失败集中在决策类问题，再决定上 Module 15.2 RL

</details>

**Q3（衔接）**：Agent SFT 后再做 Agent RL 的判断标准？什么时候 SFT 够？

<details>
<summary>Answer sketch</summary>

**先做错误归因再选 SFT/RL**——抽 100 条失败 case 人工分类，根据失败模式决定。

**SFT 够（不需要 RL）的情况**：

1. **失败集中在未覆盖的 pattern**：某类 tool / 某种 schema / 特定子任务显著低 → 补对应 SFT 数据更直接、ROI 高
2. **base 还有 SFT 空间**：数据从 50k 扩到 500k 还在涨分（loss 还在降）→ 先把 SFT scale 拉满
3. **trajectory 短（≤ 5 步）**：决策路径少，SFT 能覆盖大部分；RL 边际收益小
4. **Tool 协议变化（新增 / 改 schema）**：必须 SFT 教 format
5. **业务对决策容错高**：调错 tool 影响小（如内部 demo / 探索性场景）

**必须上 RL 的情况**：

1. **系统性"探索-反馈"问题**：tool 调失败后 model 不会 retry / 不会换 tool / 不会反问 → trajectory-level reward 才能教，SFT 数据无法穷举所有失败路径
2. **有可执行 reward signal**：task success / unit test pass / executable accuracy，reward 干净时 RL 收益最大
3. **SFT 已饱和**：数据 ×10 不再涨分，loss 早平台
4. **multi-turn planning（trajectory > 10 步）**：单步 SFT 学不出"先 plan 再 execute"的全局策略
5. **OOD 泛化弱**：SFT 在训练分布内强、换 task / tool 直接掉 30%+ → RL 探索能跳出 expert 分布

**Trade-off 边界**：

- RL 比 SFT 复杂 5-10 倍工程量（rollout infra、reward design、KL 约束、reward hacking 防御）；SFT 还能涨分别轻易上 RL
- RL 容易 reward hacking——只用 schema 合法度做 reward，model 学会输出"格式合法但语义错"的 tool call；多目标 reward（pass + format + step penalty）才稳
- **80% → 90% 是常见 SFT-to-RL 交接点**——再加 SFT 数据边际收益急剧下降，RL fine-tune 几千 step 通常能再涨 5-15 个点
- **DeepSeek-R1 范式**：少量高质量 SFT cold start (~10k) + 大量 RLVR (~几千 step) 是当前最高效路径，Search-R1 / Agent-R1 / ReSearch 都是这个 recipe（Module 15.4 详讲）

加分要点：
- 能指出"判断 SFT 饱和"的具体信号——eval loss 平台、不同 lr final score 一致、增加 10× data 提升 < 0.5%
- 能提到 RL 阶段仍要混 SFT data 做 reference KL（防漂移），呼应 9.6
- 能指出 trajectory-level reward 是 Module 15.2 的核心，归因到每一步 thought / action 是难点

</details>

---

## 12. 延伸阅读

- [FireAct GitHub](https://github.com/anchen1011/FireAct) — Princeton 出品，ReAct trajectory SFT 第一篇正经实现，data 与 training script 完整开源
- [Agent-FLAN GitHub](https://github.com/InternLM/Agent-FLAN) — Shanghai AI Lab 出品，包含 negative example 设计与 reasoning/format/tool 三任务拆解的实现
- [AgentTuning GitHub & AgentInstruct](https://github.com/THUDM/AgentTuning) — THU 出品，AgentInstruct 1.8k 高质量 trajectory + 跨 6 task 训练 recipe
- [ToRA GitHub](https://github.com/microsoft/ToRA) — MSRA + THU，math reasoning + Python interpreter 整合的标杆，含 16k 数据 + 训练 + eval
- [xLAM GitHub & paper](https://github.com/SalesforceAIResearch/xLAM) — Salesforce 出品 agent SFT 工业方案，全 size 系列 model + 60k 数据
- [APIGen 数据集 + paper](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) — Salesforce 函数调用数据合成 pipeline 工业范本
- [AgentBench](https://github.com/THUDM/AgentBench) — THU 推出的 agent eval benchmark，与 AgentTuning 配套
- 推荐继续读本教程的 **15.2 多轮 PPO/GRPO** —— trajectory-level reward 与归因（本节是冷启动基础）；**15.4 Reasoning + Agent** —— Search-R1 / ReSearch / Agent-R1 等 SFT + RLVR 的成熟范式；**Module 9 RLHF** —— PPO / GRPO 数学基础；**14.3 Tool Use 训练** —— tool 数据本身怎么造的演化历程
