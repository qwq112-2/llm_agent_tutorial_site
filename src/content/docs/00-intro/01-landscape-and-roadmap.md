---
title: "0.1 LLM/Agent 算法工程师在做什么 + 5 分钟跑通 demo + 学习地图"
description: "回答三件事：LLM/Agent 算法工程师每天到底在做什么、当下（2026）这个领域走到了哪一步、剩下 86 节按什么顺序读——读完这一节，你应该知道自己未来要不要继续读下去，以及怎么读。"
---

> ⏱ 预计阅读 25 分钟 ｜ 难度 ★ ｜ 前置：无（整本书的入口）

## 一句话本节讲什么

回答三件事：**LLM/Agent 算法工程师每天到底在做什么、当下（2026）这个领域走到了哪一步、剩下 86 节按什么顺序读**——读完这一节，你应该知道自己未来要不要继续读下去，以及怎么读。

---

## 1. 三类工程师的边界

业界常把 "做大模型的" 笼统叫做 "LLM 工程师"，但岗位实际分化成三档，技能栈、招聘 JD、考核维度都不同。先把这三类分清，后面 86 节才知道哪些必学、哪些可以跳。

- **LLM 算法工程师**：负责模型本身——预训练、SFT、RLHF/GRPO、长上下文、推理加速、scaling law 实验。产出是 weights 和 loss curve。
- **Agent 算法工程师**：负责让模型 "会用工具、会多轮交互、会自己纠错"——tool use 训练、多轮 RL、planning、memory、agent benchmark。产出是 trajectory 与任务成功率。
- **LLM 应用工程师**：负责把现成模型 / API 拼成产品——prompt 工程、RAG 管线、function calling、LangChain/LangGraph 业务编排、可观测性。产出是上线服务和业务指标。

| 维度 | LLM 算法工程师 | Agent 算法工程师 | LLM 应用工程师 |
|---|---|---|---|
| 日常做什么 | 跑预训练 / 后训练实验、调架构、调 infra、看 loss/eval | 设计 agent loop、训 tool use、做多轮 RL、盯 benchmark | 写 prompt、搭 RAG、调 function calling、做评测兜底 |
| 必备技能 | PyTorch + 分布式 + CUDA 心智 + RL 基础 + paper sense | 上面 + agent framework + RL 多轮归因 + env 工程 | Python + LLM API + RAG/向量库 + 业务理解 |
| 招聘 JD 关键词 | pretrain / post-training / RLHF / GRPO / FlashAttention / Megatron / vLLM / scaling law | agent / tool use / multi-turn RL / function calling / SWE-bench / planning | RAG / LangChain / prompt / vector DB / function calling / 业务落地 |
| 能去的公司 | OpenAI / Anthropic / DeepSeek / 字节 Seed / 阿里通义 / 智谱 / Moonshot / 各大厂基模团队 | 同上 + Cognition / Adept / 各大厂 agent 团队 / 创业公司 | 几乎所有有 AI 业务的公司 |
| 入门难度 | 高（infra + RL + 数学密集） | 高（多模型协同 + RL 多轮难题） | 中（API 熟练 + 工程感） |

本教程**目标培养前两者**——LLM 算法工程师与 Agent 算法工程师。但 Module 13（Prompt / RAG / Tool）是三类都要懂的最小公分母，应用工程师可以只读 Module 0、3、12、13、14。

一句话总结边界：**应用工程师调 prompt，算法工程师改 weights，agent 工程师设计 loop**。

---

## 2. 当下处在什么阶段

LLM 与 Agent 是同一条主线上的两个阶段，简短时间线：

- **2017** Transformer 发布，self-attention 取代 RNN
- **2020** GPT-3 (175B) 展示 in-context learning，"大模型" 概念成立
- **2022 Q4** ChatGPT 上线 + ReAct paper，分别奠定 LLM 产品形态与 Agent 范式
- **2023** LLaMA 开源，整个开源生态起步；DPO 提出，后训练简化；AutoGPT 引爆 agent 热潮
- **2024** Mixtral / DeepSeek-MoE 让 MoE 进入主流；FlashAttention-3、vLLM、SGLang 把推理推到新台阶；Anthropic Computer Use 打开 GUI agent
- **2025 Q1** DeepSeek-R1 + GRPO 证明纯 RL + verifiable reward 可以激发 long-CoT，"reasoning model" 与 RLVR 成为新范式
- **2025-2026** Reasoning agent（Search-R1 / ReSearch / ReTool / Agent-R1）把 R1 范式迁移到 agent；native multimodal（GPT-4o、Gemini 2.x）+ 1M+ context + computer use 三栖

**当下（2026 视角）的前沿地图**——七个仍在剧烈演化的方向：

1. **Reasoning models**：long-CoT + RLVR，成为继 pretrain scaling 之后的第二条 scaling 轴（test-time compute）
2. **Agent RL**：把 GRPO/PPO 推到多轮 trajectory，关键难点是 advantage 归因与 observation mask
3. **Native multimodal**：early-fusion token 化（Chameleon、GPT-4o），取代 "vision encoder + projector + LLM" 的拼装路线
4. **Computer use & GUI agent**：Anthropic Computer Use、OpenAI Operator、UI-TARS，把 agent 从 API 世界推到屏幕世界
5. **Long-context**：1M+ token 已是商业模型标配，YaRN / LongRoPE / Ring Attention 的训练 + 推理双侧优化
6. **MoE 极致化**：DeepSeek-V3 fp8 训练、fine-grained expert + shared expert，参数量与激活量解耦
7. **Multi-turn agent 鲁棒性**：observation perturbation、tool failure、recovery——下一波 agent 论文的核心战场

读完这本书，你应当能在每个方向上至少**复述 1-2 篇代表 paper 的 motivation 与 trade-off**，并能在面试里说清楚 "为什么是这么做、它解决了什么、它引入了什么代价"。

---

## 3. 5 分钟跑通 demo

不要先看 80 节理论再写代码。先跑一个最小 agent，建立 "原来 LLM 能调用工具" 的肌肉记忆，再回头看每一层是怎么实现的。

**准备**（30 秒）：

```bash
pip install openai
export OPENAI_API_KEY=sk-xxx   # 或 ANTHROPIC_API_KEY
```

**代码**（< 30 行，OpenAI Python SDK v1.x 风格）：

```python
import json
from openai import OpenAI

client = OpenAI()  # 自动读 OPENAI_API_KEY

# 1. 定义一个假的 tool（真实场景会调天气 API）
def get_weather(city: str) -> str:
    return json.dumps({"city": city, "temp_c": 14, "condition": "cloudy"})

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "parameters": {"type": "object",
                       "properties": {"city": {"type": "string"}},
                       "required": ["city"]},
    },
}]

messages = [{"role": "user", "content": "今天伦敦天气怎么样？"}]

# 2. 第一次 call：模型决定要不要用 tool
resp = client.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=tools)
msg = resp.choices[0].message
messages.append(msg)

# 3. 如果模型选择了 tool call，我们执行并把 observation 喂回去
for call in msg.tool_calls or []:
    args = json.loads(call.function.arguments)
    observation = get_weather(**args)
    messages.append({"role": "tool", "tool_call_id": call.id, "content": observation})

# 4. 第二次 call：模型基于 observation 给最终答案
final = client.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=tools)
print(final.choices[0].message.content)
```

> Anthropic 等价写法：把 `from openai import OpenAI` 换成 `from anthropic import Anthropic`，`client.messages.create(model="claude-sonnet-4-5", ...)`，tool schema 字段名稍有差异（`input_schema` vs `parameters`），其余 loop 结构完全一致。

这 28 行代码已经体现了一个 agent 的全部核心要素：

- **Tool definition**（第 9-17 行）：用 JSON schema 把工具能力描述给模型——这是 function calling 的契约
- **Model choosing tool**（第 22 行）：模型根据 user query 自主决定调不调、调哪个、参数填什么——这一步是 LLM 的 "决策"
- **Observation feedback**（第 27-30 行）：执行 tool 拿到 observation，以 `role="tool"` 的消息塞回 message history——这是 ReAct loop 的 "感知"
- **Final answer**（第 33-34 行）：模型基于 observation 生成自然语言答案——这是 "整合"

把 `get_weather` 换成真实 API、把单步 loop 改成 `while` 循环允许多步、把 `messages` 加上长期 memory，就是 Module 14 要讲的完整 agent。把模型从 API 换成自己训的、tool selection 从 prompt 改成 RL 训出来的，就是 Module 15 要讲的 agent RL。

---

## 4. 学习地图

整本教程 16 个 module、87 节，按 **"DL 基础 → Transformer → 预训练 → Infra → 后训练 → 推理部署 → 应用 → Agent → 多模态"** 的线性顺序组织。**不为赶 agent 热点把 RLHF 提前**——agent RL 必须建立在 RLHF 心智模型上。

### 16 个 module 拓扑

```
                    module 0   引言与学习地图（你在这里）
                          │
       ┌──────────────────┼──────────────────┐
       │                  │                  │
   module 1            module 2          module 3
   DL 基础速通          NLP 任务全景       Tokenization
       │                  │                  │
       └──────────────────┼──────────────────┘
                          ▼
                   module 4  Transformer from scratch     ★ 必经之路
                   module 5  现代 LLM 架构变体（GQA/MoE/Mamba）
                   module 6  预训练（Scaling Law / Long-context）
                   module 7  训练 Infra（FSDP / TP / fp8）  ★ 独立成块
                          │
                          ▼
                   module 8  后训练 I：SFT 与 PEFT
                   module 9  后训练 II：RLHF / DPO / GRPO     ★ 后训练核心
                   module 10 Reasoning 与 Test-time scaling
                          │
                          ▼
                   module 11 推理引擎（vLLM / 量化 / 投机解码）
                   module 12 评测与 LLMOps
                          │
                          ▼
                   module 13 Prompt / RAG / Tool（应用三件套）
                   module 14 Agent 系统                      ★ Agent 核心
                   module 15 Agent RL 与多轮鲁棒性             ★ 前沿
                   module 16 多模态 / Embedding / Computer Use
                          │
                          ▼
                   附录 A/B/C  Capstone 项目
                   附录 D     数学速查
```

**几个 module 的关键作用要先记住**：

- **Module 4** 是分水岭——手撕 nanoGPT 之前都算 "学语法"，之后才能读懂任何现代 LLM paper
- **Module 7** 是"独立成块"的——只有自己训过 70B+ 模型才能体会 ZeRO / TP / PP 的痛点，没机会跑也要先懂心智模型
- **Module 9** 是 LLM 算法岗面试 90% 会考的——RLHF 的四模型显存、PPO 的 KL 项、DPO 的闭式解、GRPO 的 group baseline，每一个都是高频题
- **Module 14-15** 是 agent 算法岗的核心战场——从 ReAct 到多轮 GRPO，覆盖学术与工业全谱

### 三种推荐学习路径

| 路径 | 适用人群 | 顺序 | 预计耗时 |
|---|---|---|---|
| **A. 0 基础完整路径** | 在校生 / 转行工程师 / 想系统打底 | 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11 → 12 → 13 → 14 → 15 → 16 | 3-6 个月（每天 2h） |
| **B. 已会 DL，专攻 LLM 后训练** | 有 PyTorch + DL 经验，瞄准基模 / 后训练岗 | 0 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11（→ 12 选读） | 6-10 周 |
| **C. 已会 LLM 基础，专攻 Agent** | 已懂 Transformer + 后训练，瞄准 agent 岗 | 0 → 9（速读，建立 RL 心智） → 10 → 13 → 14 → 15（→ 16.5 GUI agent） | 4-6 周 |

校招党额外建议：路径 A 走完之后，把所有标 🔥 的节再过一遍——那是面试官最爱问的高频点，分布在 Module 1.2 / 1.3 / 3.1 / 4 整章 / 5.2 / 5.3 / 5.4 / 6.2 / 6.3 / 6.5 / 7.1 / 7.2 / 7.5 / 8.1 / 8.2 / 8.3 / 9 全章 / 10.3 / 11.2 / 11.4 / 11.5 / 12.1 / 13.2 / 13.4 / 14.2 / 14.3 / 15.2 / 15.4 / 16.1 / 16.4。

---

## 5. 怎么用这本书

不是 "从头读到尾" 那么简单。每节 1500-7000 字，照搬阅读会很疲劳。下面 5 条是经过 calibration 的方法：

- **看不懂的节先标 ★ 跳过，全书读完一遍后回头啃**——LLM 知识网状高度耦合（如 RoPE 涉及 4.1 self-attention + 6.5 long-context + 11.2 vLLM 的 KV cache），第二遍很多疑惑会自然消解
- **每节末尾的"自测与面试题"必做**——不是为了答对，是 calibration：能复述出 answer sketch 的要点，才算真懂；做不出立刻回到对应小节
- **必读 paper 至少读 abstract + introduction + conclusion**——80% 的 idea 在这三段里。不要被 method section 的公式吓住，paper 是为了懂 motivation 与 trade-off，不是抄公式
- **配合至少一个开源仓库跟读**——推荐三选一：[nanoGPT](https://github.com/karpathy/nanoGPT)（懂 Module 4-6）、[Hugging Face transformers](https://github.com/huggingface/transformers)（懂 Module 5、8）、[vLLM](https://github.com/vllm-project/vllm)（懂 Module 7、11）。代码 > 文字
- **校招准备者优先看带 🔥 标记的节**——这些是面试 90% 命中点；时间不够可以 70% 时间投 🔥 节、30% 时间投其他

每节大概 30 分钟到 2 小时不等。**不要妄想一次读完一个 module**——每天稳定 1-2 节，三个月走完比突击三周记得牢得多。

---

## 6. 心法

工程师视角（不是学生视角）的几条建议，每一条都是面试官/导师级前辈反复强调过的：

- **不会跑代码 ≠ 懂算法**。每节给的最小代码至少跑一遍，把 print 加在中间看 shape，看 attention weights 分布，看 loss 怎么变。LLM 是高度经验科学，纸面理解极易出错
- **看 paper 不是看英文，是看 idea**。先用 1-2 句话回答 "这篇 paper 解决了什么 / 它的核心 trick 是什么 / 它的代价是什么"，再去看公式。看不懂公式时回头看 motivation，常常豁然开朗
- **面试官最爱问 trade-off，不是 fact**。每个技术问自己三遍："它解决了什么 / 它引入了什么代价 / 什么场景下不该用它"。例：FlashAttention 解决 memory，但短序列下 kernel launch overhead 反而更慢；MoE 减激活，但 routing 不均会浪费算力；GRPO 去 critic 省显存，但 reward sparse 时方差大
- **不要追新追到忘了基础**。R1 / o3 / Computer Use 当然要看，但 Transformer self-attention、KV cache、RLHF 三模型显存这些 5 年没变的 fundamentals 才是面试拍板的东西。新东西看 motivation 就够，老东西要会推导
- **算法工程师 = 做实验的工程师**。学会写 ablation 表、画 loss curve、控制变量、算 FLOPs/displacement。光会写训练循环不算 algo eng

---

## 7. 全局推荐资源

不是 paper 而是 "整本读完都用得上" 的三个核心资源：

- **Stanford CS336 — Language Modeling from Scratch**（Tatsunori Hashimoto / Percy Liang 2024-2025）— 最系统的 LLM from-scratch 课程，5 个 assignment 覆盖 tokenization → architecture → training → systems → alignment。本教程的 Module 4-7 与 CS336 高度重合，建议配套食用：[lectures](https://stanford-cs336.github.io/) + [assignment repo](https://github.com/stanford-cs336)
- **Andrej Karpathy — Neural Networks: Zero to Hero**（YouTube + GitHub）— 最佳的 from-scratch 实现入门，从 micrograd 一路到 GPT-2。Module 1、4 学完之后回头看一遍，会有 "原来是这样" 的顿悟：[YouTube playlist](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
- **Hugging Face LLM Course**（HF team，2024 持续更新）— 最好的开源生态实操教程，把 transformers / datasets / peft / trl / accelerate 串起来。Module 8、9、11 学完后用它做端到端实操：[huggingface.co/learn/llm-course](https://huggingface.co/learn/llm-course)

---

## 8. 自测题

**Q1（概念题）**：LLM 算法工程师和 LLM 应用工程师的核心技能差异在哪 3 个层面？

<details>
<summary>Answer sketch</summary>

至少要点到：

- **介入深度**：算法工程师改 weights / loss / 架构；应用工程师改 prompt / 业务逻辑，不动模型
- **必备 stack**：算法工程师需要 PyTorch + 分布式（FSDP/TP/PP）+ CUDA 心智 + RL 数学；应用工程师需要 LLM API + RAG + LangChain/LangGraph + 业务理解
- **产出与考核**：算法工程师交付 weights、训练 / eval 报告、scaling 曲线；应用工程师交付上线服务、业务指标（CTR / 满意度 / 转化率）
- 加分：Agent 算法工程师是中间态——既要懂算法（多轮 RL / tool use 训练），也要懂应用（agent loop / framework）

</details>

**Q2（前沿感知）**：举出 3 个 2024-2026 年改变 LLM/Agent 范式的关键节点，分别带来了什么变化？

<details>
<summary>Answer sketch</summary>

任选 3 个，每个说清 "之前 → 之后" 的范式变化：

- **DeepSeek-R1 + GRPO（2024.02 paper / 2025.01 R1）**：之前 reasoning 靠 prompt（CoT / Self-Consistency / ToT）；之后纯 RL + verifiable reward 可激发 long-CoT，test-time compute 成为 scaling 第二轴
- **Anthropic Computer Use（2024.10）+ OpenAI Operator（2025）**：把 agent 从 API 世界推到屏幕世界，催生 GUI agent / OSWorld / UI-TARS 一整条赛道
- **DeepSeek-V3 fp8 训练（2024.12）**：把 fp8 端到端训练验证可行，训练吞吐再提 1.5-2×；MoE 参数量与激活量解耦成为常态
- **Mixtral / DeepSeek-MoE（2024）**：MoE 从研究热词变成开源主流，fine-grained expert + shared expert 架构定型
- **Native multimodal（GPT-4o 2024 / Chameleon / Gemini 2.x）**：取代 "vision encoder + projector + LLM" 的拼装路线，走 early-fusion token 化
- **Search-R1 / ReSearch / ReTool / Agent-R1（2025）**：把 R1 范式迁移到 agent，开启 reasoning agent + tool use 的多轮 RL 浪潮

</details>

**Q3（学习方法 / 开放题）**：你打算怎么用这本书？请写出 3 条具体的学习习惯。

<details>
<summary>Answer sketch（hint，不是标准答案）</summary>

好的回答会包含：

- **节奏**：每天 / 每周读多少节，估算 3-6 个月走完路径 A
- **可操作的 calibration 机制**：每节自测题先做、不会就回头读、配 Anki / 笔记 / blog 写出来
- **代码动手承诺**：至少跟读一个开源 repo（nanoGPT / transformers / vLLM 选一），每节代码至少跑一遍并记一条 "我没想到的" observation
- **paper 阅读节奏**：每节必读 paper 至少读 abs+intro+conclusion，每周精读 1 篇
- **目标导向**：如果是校招就重点刷 🔥 节 + Module 9 + Module 14；如果是研究就额外读 Module 5 / 10 / 15 的 SOTA paper

差的回答："认真读完每一节"——没有可执行性。

</details>

---

## 9. 延伸阅读

- [Sebastian Raschka — Build a Large Language Model (From Scratch)](https://github.com/rasbt/LLMs-from-scratch) — 配套 GitHub repo，Karpathy 风格的精简代码，本书 Module 1-9 的最佳辅助
- [Lilian Weng — Blog (lilianweng.github.io)](https://lilianweng.github.io/) — Agent / RLHF / Hallucination / Diffusion 的扫盲长文，每篇都是 1 万字综述级
- [Hugging Face Daily Papers](https://huggingface.co/papers) — 每天精选 LLM/Agent paper，社区投票排序，跟踪 SOTA 必备
- [smol-course](https://github.com/huggingface/smol-course) — HF 官方的 4 周后训练实战课程，SFT/DPO/Eval/Agent 各一周，对应本书 Module 8、9、12、14
- 推荐继续读本教程的 **1.1 节《神经网络与反向传播》**——从 mathematical bedrock 开始
