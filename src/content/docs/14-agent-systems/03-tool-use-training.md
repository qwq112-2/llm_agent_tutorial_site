---
title: "14.3 Tool Use 训练：Toolformer / Gorilla / ToolLLaMA / xLAM"
description: "Tool Use 训练讲的是怎么把\"调工具\"这件事训进 model 权重——从 2023 Toolformer 用 self-supervised 让 model 自己标 API 调用、到 Gorilla 的 retrieval-augmented fine-tuning、到 ToolLLaMA 在 16k+ RapidAPI 上做 DFSDT 多步 trajectory、再到 2024 Tool"
---

> ⏱ 预计阅读 60 分钟 ｜ 难度 ★★★ ｜ 前置：8.5（多轮 + tool 混合 SFT）、13.4（function calling 工程）｜ 🔥 必考

## 一句话本节讲什么

Tool Use 训练讲的是**怎么把"调工具"这件事训进 model 权重**——从 2023 Toolformer 用 self-supervised 让 model 自己标 API 调用、到 Gorilla 的 retrieval-augmented fine-tuning、到 ToolLLaMA 在 16k+ RapidAPI 上做 DFSDT 多步 trajectory、再到 2024 ToolACE / xLAM 把数据合成做成"工业 pipeline"——本节梳理 4 个时代的方法论、给出现代 SFT pipeline 的标准答案，并把"什么时候 SFT 够、什么时候必须 RL"这条与 Module 15 衔接的边界讲清楚。

---

## 1. Mental model（直觉）

13.4 讲了 **inference 时**怎么把 tool call 这套协议跑通（JSON schema、constrained decoding、parallel call），8.5 讲了**给定 tool 数据**怎么端到端 SFT 一遍——本节回答上一层问题：**这些 tool 数据是怎么来的？训练范式怎么演化的？现代 pipeline 长什么样？**

把 tool use 训练的演化想成"data difficulty 的逐级提升"：

```
2023 Toolformer  ──► single API, in-text     ─┐
                    self-supervised 自标       │
                                               │
2023 Gorilla     ──► single API, retrieval    ─┤  Single-turn
                    retriever-aware FT         │  Single-tool 时代
                                               │
2023 ToolLLaMA   ──► multi API, multi-step    ─┤
                    DFSDT trajectory          ─┘

2024 ToolACE     ──► dimension-coverage      ─┐  Multi-turn
                    parallel+sequential+      │  Multi-tool 时代
                    refuse 全维度合成         │  (data-centric)
                                               │
2024 xLAM        ──► industrial pipeline    ──┤
                    60k high-quality data    │
                    1B-70B 全 size           │
                                              │
2025+ Agent RL   ──► SFT 冷启 + RLVR/GRPO  ──┘  Trajectory
                    multi-turn agent RL          级 RL 时代
                    （Module 15 详讲）
```

四个时代的核心 framing：

- **第 1 代（Toolformer 2023.03）**：把 tool call 嵌入"语言模型继续预测下一个 token"的语境里——给一段连续文本，模型在最有用的位置插一个 `<API>...</API>` 标签，得到 API 结果后能让后续 token 的 perplexity 下降。**self-supervised** 是关键词
- **第 2 代（Gorilla 2023.05）**：从"零样本无 retrieve"到"训练时把 retrieved API doc 喂给 model"。tool 数从十几个跳到 1600 个，retriever 成为 first-class citizen。**retrieval-augmented fine-tuning (RAT)**
- **第 3 代（ToolLLaMA 2023.07）**：把 single-turn 升级到 multi-step trajectory；tool 数推到 16k 真实 RapidAPI；用 DFSDT（决策树式搜索）让 ChatGPT 当 teacher 生成数据。**multi-step + scale-up**
- **第 4 代（ToolACE / xLAM 2024）**：data-centric 的工业化——不再追求"更大 tool 池"，而是显式覆盖正交的能力维度（parallel / sequential / refuse / nested arg / negative case）。xLAM 把这套 pipeline 做成全 size 系列模型，长期占据 BFCL 7B 头部
- **第 5 代（2025+ Agent RL）**：SFT 把 format 与基础能力打底，RL 用 verifier reward 在 trajectory 级别精调多步决策——这部分本节会勾勒，详细展开在 Module 15

读者读完本节后，看到任何"我要训一个能调 N 个 tool 的 agent"的需求，应该能立刻回答出：(1) 数据从哪来、怎么合成 (2) 用什么 schema (3) 训练 pipeline 是什么 (4) 用什么 benchmark 评测 (5) 什么时候需要上 RL。

> 与 8.5 区分：8.5 讲拿到数据后**端到端 SFT 一遍的工程实战**，本节讲**这些数据本身是怎么造出来的、训练范式有哪些**。
> 与 13.4 区分：13.4 讲 **inference 时**的 tool 协议，本节讲**training 时**怎么把能力训进权重。
> 与 Module 15 区分：本节止于"SFT 把 format 打底"，trajectory-level RL 的细节（reward design、KL 约束、归因）留给 15.1 / 15.2。

---

## 2. 第 1 代：Toolformer（Schick et al., 2023）

### 2.1 核心 idea：让 model 自己标 API 调用位置

Toolformer 是第一个**完全 self-supervised** 的 tool learning 方法——不依赖人工标注、不依赖 GPT-4 蒸馏，只用 model 自己 + 一段无标注文本就能产出训练数据。

核心 insight：**如果在文本中合适位置插入一次 API call 与结果，能让后续 token 的 perplexity 下降，那这个位置就是"该调 tool"的位置**。这把"何时调 tool"这件事从主观判断变成了可计算的 likelihood ratio。

### 2.2 数据合成 4 步流程

给定一段无标注文本 $x = (x_1, x_2, \ldots, x_n)$ 与一组 API（如 calculator / QA system / translator / Wikipedia search）：

**Step 1：候选位置采样**。用 few-shot prompt 让 base LM 在 $x$ 中每个位置 $i$ 生成最多 $k$ 个候选 API call $c_i^{(1)}, c_i^{(2)}, \ldots$，每个 call 形如 `[Calc(7*8)]`（tool name + args）。

**Step 2：执行 API**。对每个候选 $c_i^{(j)}$ 执行 API 拿 result $r_i^{(j)}$，组合成完整 call 序列 $\text{API}(c_i^{(j)}) \to r_i^{(j)}$。

**Step 3：过滤——计算"插入 API 是否有用"**。这是核心步骤。定义两种 prefix：

- 不插 API：$x_{1:i-1}$
- 插 API + result：$x_{1:i-1} \cdot \text{API}(c_i^{(j)}) \to r_i^{(j)}$

然后看后续 token $x_i, x_{i+1}, \ldots, x_{i+L}$ 在两种 prefix 下的加权 cross-entropy loss：

$$L_i^{+} = \sum_{t=i}^{i+L} w_{t-i} \cdot \ell(x_t \mid x_{1:i-1}, \text{API}(c_i^{(j)}) \to r_i^{(j)})$$

$$L_i^{-} = \min\left(\sum_{t=i}^{i+L} w_{t-i} \cdot \ell(x_t \mid x_{1:i-1}), \sum_{t=i}^{i+L} w_{t-i} \cdot \ell(x_t \mid x_{1:i-1}, \text{API}(c_i^{(j)}) \to \varepsilon)\right)$$

其中 $\ell$ 是 negative log-likelihood，$w_{t-i}$ 是位置距离权重（越远权重越小），$\varepsilon$ 表示空 result。**只保留 $L_i^{-} - L_i^{+} \geq \tau$ 的 call**——即"插了 API 后 perplexity 下降至少 $\tau$"的 call 才认为有用。$\tau$ 通常取 0.5-1.0。

**Step 4：构造 SFT 数据**。把过滤剩下的 $(x, \text{API call}, \text{result})$ 拼成训练样本，把 API call 直接嵌入文本作为新的 SFT 数据，再 fine-tune base LM。fine-tune 后的 model 推理时就会**主动**在合适位置 emit API call。

### 2.3 评价：开创性但 scale 有限

Toolformer 证明了 **self-supervised 学 tool use 是可行的**，是这条路线的奠基工作。但实际局限：

- **只覆盖 single-call**，不支持多步 / 多 tool 协同
- **只覆盖了 in-text API**（calculator / QA / translator / wiki / calendar 5 个），不覆盖现代 RESTful API 的复杂 schema
- **依赖 base LM 自身能力**——base LM 太弱时 Step 1 的候选 call 质量就低，整个 loop 退化

这些局限直接催生了 Gorilla / ToolLLaMA 的下一代方法。

---

## 3. 第 2 代：Gorilla（Patil et al., 2023）

### 3.1 任务设定：1600+ ML API

Gorilla 关注的是一类很窄但很实用的 tool——**ML model API**：HuggingFace Hub / TensorFlow Hub / PyTorch Hub 上的 1600+ 个公开 model 调用接口。给一句话需求（如"我要做一个图像分类模型"），让 LLM 输出对应的 model name + 调用代码片段。

为什么这个 setting 有意思？因为 ML API 是**频繁更新**的——HuggingFace 每天有新 model 上线，老 model 可能被弃用。如果 model 是"硬记"了某些 API 进权重，三个月后就过时了。Gorilla 给出的方案是：让 model **学会"看 doc 调 API"**，新 API 的 doc 可以在 inference 时塞进 context。

### 3.2 RAT：Retrieval-Augmented Fine-Tuning

核心方法叫 **Retrieval-Augmented Fine-Tuning (RAT)**——**训练时**就把 retrieved API doc 加入 input：

```
[Training input]
User query: "I want to translate English to Chinese."
Retrieved API doc: 
  <<<api>>>: Helsinki-NLP/opus-mt-en-zh
  <<<api_provider>>>: HuggingFace
  <<<explanation>>>: This model translates English to Chinese...
  <<<code>>>: from transformers import pipeline; ...

[Training target]
Use the model `Helsinki-NLP/opus-mt-en-zh`. Code:
  from transformers import pipeline
  translator = pipeline('translation_en_to_zh', model='Helsinki-NLP/opus-mt-en-zh')
  result = translator("Hello world")
```

训练时 retrieved doc 是 **gold doc**（真正与 query 匹配的那篇）；inference 时用一个独立 retriever（BM25 / DPR / GPT-Embedding 都可）取 top-k 个候选 doc 喂 model。

为什么 RAT 比"先 SFT 再加 retriever"好？因为它让 model **学会从 doc 里抽 API 名 + 参数 schema** 这个具体动作，而不是把 API 名硬记进权重。inference 时即使 retrieved doc 是新发布的 API，model 也能正确生成调用代码（**zero-shot 到新 API**）。

### 3.3 三种推理模式

Gorilla 评测时区分三种 retrieval 模式，对应不同 production setup：

| 模式 | retriever 行为 | 准确率 | 适用场景 |
|---|---|---|---|
| **Zero-shot** | 不用 retriever，纯靠权重内的 API 知识 | 最低 | API 池稳定 + 训练已覆盖 |
| **BM25 retrieval** | 经典 sparse retriever | 中 | 简单 tool 数 < 1000 |
| **GPT-Embedding retrieval** | dense retriever，semantic match | 最高 | 现代 production 标配 |

实测：retriever 加上后准确率从 ~30% 跳到 ~60%+。**这个数字暗示了一个工程铁律**：tool 数 > 50 时 retriever 必备，仅靠把所有 tool def 塞进 context window 既贵（prompt 长）又不准（attention 在 N 个 tool 上分散）。

### 3.4 Gorilla 的遗产

Gorilla 之后，"retriever-aware tool LLM"成为标配：

- **ToolLLaMA** 沿用了这个思路，加了 multi-step
- **xLAM-fc-r** 后缀的 `r` 就是 "retriever-aware"
- 现代企业 agent（候选 tool 数百到上千）几乎全用 RAT 风格部署

---

## 4. 第 3 代：ToolLLaMA / ToolBench（Qin et al., 2023）

### 4.1 数据规模：16k+ 真实 RapidAPI

ToolBench 是第一个**大规模真实 tool**数据集——从 RapidAPI 收集了 16,464 个真实 REST API（覆盖 49 个 category，3,451 个 tool），每个都有真实可调用的 endpoint + OpenAPI spec。这个 scale 让 tool learning 从"5-10 个玩具 tool"跨入"上千 tool 的真实场景"。

数据集组织成三种任务类型：

- **I1 (single-tool)**：一个 task 内只允许调一个 tool
- **I2 (intra-category multi-tool)**：同 category 内多个 tool 协同
- **I3 (inter-category multi-tool)**：跨 category 多个 tool 协同（最难）

### 4.2 DFSDT：Depth-First Search-based Decision Tree

ToolLLaMA 的关键贡献是 **DFSDT**——一种用 ChatGPT 当 teacher 生成 multi-step trajectory 的方法。传统做法（CoT-style 让 ChatGPT 一次性想清楚）在 multi-tool 场景下成功率低（30-40%），DFSDT 把它做成**搜索式 reasoning**：

```
        Root: user query
        │
        ├── Try call_A(args1) ──► obs_1 ──► not enough info
        │   │
        │   ├── Try call_B(...) ──► obs_2 ──► success ✓ (reach answer)
        │   │
        │   └── Try call_C(...) ──► obs_3 ──► error ✗ (backtrack)
        │
        ├── Try call_A(args2) ──► obs_4 ──► ...
        │
        └── Try call_D(...) ──► ...
```

每个节点是一次 tool call + observation。如果当前路径走不通（obs 显示 error 或不足够信息），**回溯**到上一节点试别的 call；走通了就 commit 这条 trajectory 作为一条 SFT 样本。

DFSDT 把任务成功率从 ~40%（CoT baseline）拉到 ~70%+。代价是 token 消耗 4-10×（要展开整棵决策树），但 **teacher cost 不重要——data 是一次性的固定投入**，下游 SFT 反复用。

### 4.3 训练与评测

ToolLLaMA = LLaMA-7B 在 ToolBench 上 SFT。评测有两条 track：

- **Pass Rate**：trajectory 是否走完
- **Win Rate**：与 ChatGPT baseline 的人工 / GPT-4 pairwise 对比

ToolLLaMA-7B 在 I1 上 win rate ~60%（vs ChatGPT），在 I3 上 ~50%，证明了 **7B 开源 model 经过 ToolBench SFT 能在 tool use 上接近 GPT-3.5**。这个数字在 2023 年是震撼的，是开源 tool LLM 真正出圈的起点。

### 4.4 后续：StableToolBench

ToolBench 一个广为人知的问题——**真实 RapidAPI 不稳定**：评测同一 model 不同时间跑分波动 5-10%，因为 API 随时挂或返回值变。**StableToolBench**（Guo et al., 2024）的工作是把 16k API 在固定时间快照下，所有 API 调用都通过 cache 或 mock，让评测可复现。production 评测建议用 StableToolBench 而不是裸 ToolBench。

---

## 5. 第 4 代：ToolACE / xLAM 2024

### 5.1 ToolACE：Accuracy + Complexity + Diversity

ToolACE（Liu et al., 2024）的命名就是它的方法论——**A**ccuracy + **C**omplexity + **E**ngagement。核心 insight：tool 数据的"质"比"量"更重要，要显式按正交维度合成。

ToolACE 的 multi-agent 数据合成 pipeline：

```
┌──────────────┐    pool of 26k tools    ┌──────────────┐
│ Tool         │ ───────────────────────►│ User Agent   │
│ Self-Evolution│                         │ (生成 query) │
└──────────────┘                          └──────┬───────┘
                                                 │
                  ┌──────────────────────────────┤
                  │                              │
                  ▼                              ▼
           ┌─────────────┐               ┌─────────────┐
           │ Assistant   │               │ Validator   │
           │ Agent       │ ──tool_call──►│ Agent       │
           │ (生成 call) │ ◄──feedback── │ (4 类校验)  │
           └─────────────┘               └─────────────┘
```

四类校验包括：(1) tool name 在白名单 (2) 参数 schema 合法 (3) 必填字段齐全 (4) 调用语义合理。任何一项 fail 都把样本踢掉。

ToolACE 显式覆盖的能力维度（与 8.5 §5.2 表格呼应）：

| 维度 | 占比 |
|---|---|
| Single-call | ~50% |
| Parallel call | ~15% |
| Sequential / dependent call | ~15% |
| Multi-step (5+ turn) | ~10% |
| Refuse-when-no-tool | ~10% |

最终 ToolACE-8B 在 BFCL 上能打过 GPT-4，证明了**高质量数据合成 + 8B 开源 model = SOTA function calling**。

### 5.2 xLAM：Salesforce 的工业 pipeline

xLAM（Liu et al., 2024）来自 Salesforce AI Research，是**最完整的开源 agent SFT 工业方案**。区别于 ToolACE 偏研究范式，xLAM 直接交付了：

- **xLAM-function-calling-60k**：60k 高质量 tool call 数据集，开源
- **xLAM-1B / 7B / 8x7B / 70B-fc-r**：从 1B 到 70B 全 size 函数调用模型
- 完整 training recipe + ablation 报告

xLAM 数据合成的 4 步（与 8.5 §5.1 同源）：

1. **Tool collection**：~20k tool definition（OpenAPI / HuggingFace / 自写 mock 函数）
2. **Query synthesis**：给 LLM 几个 tool def + few-shot，让它生成"用户可能怎么问"的 query
3. **Answer synthesis**：用 GPT-4o / DeepSeek-V3 按 (query, tools) 生成正确 tool call
4. **Verification**：可执行的 tool 真调一次（schema check + API check），过滤错误样本

**xLAM-fc-r** 后缀的 `r` 表示"retriever-aware"——训练时 system prompt 里放的 tool 数是动态的（5-30 个），让 model 习惯不同 retrieval 配置。这一点直接继承自 Gorilla 的 RAT 思路。

xLAM 在 BFCL 上的表现：

| Model | BFCL Overall |
|---|---|
| xLAM-7B-fc-r | ~80%（接近 GPT-4） |
| xLAM-8x22B-r | ~85%（超过 GPT-4） |
| xLAM-1B-fc-r | ~70%（小模型 SOTA） |

8x22B 超过 GPT-4 的事实在 2024 年下半年震动社区——证明了**开源 + 高质量 SFT 数据足以在 tool calling 这一垂类做到顶尖**，不需要 RL 加持。

---

## 6. 现代 Tool Use 训练 pipeline（标准答案）

把上面 4 代方法融合，2025 年面试或上岗写一份"标准 tool agent 训练 pipeline"，必须按下面 5 个 stage 组织：

```
Stage 1: Base Model
   │  Qwen2.5-7B-Base / Llama-3.1-8B-Base
   ▼
Stage 2: General SFT
   │  Magpie-Pro-300k / Tülu-3-SFT / OpenHermes-2.5
   │  (chat 流畅度 + format 多样性 + reasoning 基础)
   ▼
Stage 3: Tool Calling SFT
   │  xLAM-60k / ToolACE / Hermes-FC-V1
   │  + 自合成的领域 tool 数据
   │  (chat 60% + tool 30% + reasoning 10% mix)
   ▼
Stage 4: (Optional) Agent RL  ──► Module 15 详讲
   │  Verifier reward / Tool execution reward
   │  GRPO / multi-turn PPO
   │  (从 80% 推到 90%+)
   ▼
Stage 5: Evaluation
   │  BFCL v3 (overall + 5 sub-track)
   │  ToolBench / StableToolBench
   │  τ-bench (端到端 task)
   │  + 自定义 mixed eval (50-100 条覆盖 happy path / refuse / parallel)
   ▼
Production 部署
```

**几个关键决策点**：

- **Stage 2 vs Stage 3 是否合并**？小数据量（< 100k tool 数据）可以合并成"通用 + tool 一起 mix SFT"；数据量大或 base 较弱时分两 stage 更稳——先打通用 chat 底，再灌 tool 能力
- **是否 Stage 4 RL**？看 Stage 3 后 BFCL 分数。> 80% 且失败集中在"决策类问题"（多步规划、错误恢复）才考虑 RL；< 80% 还在 format / 单步准确率上挣扎，先回头加 SFT 数据更划算
- **Stage 5 必须有自定义 eval**——开源 benchmark 覆盖不了你的领域 tool 与业务场景

---

## 7. Benchmark 速查表

tool calling 这个垂类的 benchmark 演化得也很快，列出主流的 8 个：

| Benchmark | 形式 | 规模 | 特点 | 现状 |
|---|---|---|---|---|
| **API-Bank** (Li 2023) | 多轮 + 真实 API | 73 API | 早期标杆，覆盖 search / calc / 日程 | 已被 BFCL 取代主导地位 |
| **ToolBench** (Qin 2023) | 真实 RapidAPI | 16k+ API | 大规模、I1/I2/I3 三档 | 数据稀缺时仍用 |
| **StableToolBench** (Guo 2024) | 改进 ToolBench | 同上 | API 调用 cache，可复现 | 推荐替代 ToolBench |
| **BFCL v1-v3** (Yan 2024) | 函数调用 | 多类 | 事实标准，按 sub-track 细分 | **首选必看** |
| **T-Eval** (Chen 2024) | process 级 | - | 把 tool use 拆成 6 个能力维度细评 | 学术评测常用 |
| **NexusRaven V2** | parallel + nested | - | Nexusflow 出品，专测复杂 call | training 数据 + eval 一体 |
| **ToolSandbox** (Apple 2024) | stateful + 用户交互 | - | 模拟真实 user-agent 多轮 | 交互式 eval 标杆 |
| **τ-bench** (Yao 2024) | agent 真实 task | 客服 / 航空 | end-to-end，用户 simulator | **最贴近 production** |

**怎么选**：

- 选 model / 写 paper 必看 **BFCL**（业内对齐唯一标准）
- 测多步 agent 能力上 **τ-bench**（更接近 production）
- 测 process 细粒度上 **T-Eval**
- 复现性优先用 **StableToolBench**（不要用裸 ToolBench）

### BFCL 提交 format 简短示例

```python
# BFCL 评测的输出格式（AST track）
[
    {
        "id": "simple_001",
        "result": [{
            "get_weather": {"city": "Beijing", "unit": "celsius"}
        }]
    },
    {
        "id": "parallel_002",
        "result": [
            {"get_weather": {"city": "Beijing"}},
            {"get_weather": {"city": "Shanghai"}}
        ]
    }
]
```

每条记录的 `result` 是一个 list，每个 element 是 `{tool_name: {arg_name: value}}` 的 dict。BFCL 评测脚本会做 AST tree 比对（容忍空格、引号、key 顺序差异）+ 可执行的 actual call 比对。

---

## 8. 代码示例

下面 4 段代码分别对应 4 个时代的核心数据合成范式。

### 8.1 Toolformer 风格 self-supervised 数据生成

```python
# toolformer_data_synthesis.py
import math, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def perplexity(model, tok, text):
    """计算 text 的 perplexity（数值越低 = model 越自信）。"""
    ids = tok(text, return_tensors="pt").input_ids
    with torch.no_grad():
        out = model(ids, labels=ids)
    return math.exp(out.loss.item())

def try_insert_api(model, tok, text, position, api_call_str, api_result, tau=0.5):
    """在 text 的 position 处试着插入 API call，看后续 perplexity 是否下降 tau。"""
    prefix = text[:position]                       # 之前的内容
    suffix = text[position:]                       # 之后的内容（待预测）
    # 不插 API 的 baseline perplexity
    ppl_no = perplexity(model, tok, prefix + suffix)
    # 插了 API + result 之后再接 suffix 的 perplexity
    inserted = f"{prefix} [{api_call_str} -> {api_result}] {suffix}"
    ppl_yes = perplexity(model, tok, inserted)
    # 只保留"插了之后明显更自信"的 case
    return ppl_no - ppl_yes >= tau, ppl_no, ppl_yes

# 示例：在 "7 * 8 等于" 后插入 calculator call
text = "请计算：7 * 8 等于 56。"
# 假设 base LM 在位置 8（"等于"之后）想插一个 Calc(7*8) -> 56
keep, ppl_no, ppl_yes = try_insert_api(
    model, tok, text, position=8,
    api_call_str="Calc(7*8)", api_result="56", tau=0.5,
)
if keep:
    # 保留这条 (position, api_call, result) 作为 SFT 数据
    print(f"keep: ppl drop {ppl_no:.2f} -> {ppl_yes:.2f}")
```

要点：核心是 perplexity 比较——base LM 自己是 "judge"，self-supervised 不需要任何人工标注。生产里需要扩展：(1) 同一位置 sample 多个候选 call 取最佳 (2) 用加权 cross-entropy 而非整段 perplexity (3) 距离 i 越远的 token 权重越小（避免远端噪音）。

### 8.2 Gorilla retrieval-augmented prompt 构造

```python
# gorilla_rat_prompt.py
def build_rat_input(query, retrieved_docs, k=3):
    """RAT 风格：把 retrieved API doc 拼进 prompt，让 model 学'看 doc 调 API'。"""
    docs_str = "\n\n".join([
        f"<<<api>>>: {d['name']}\n"
        f"<<<provider>>>: {d['provider']}\n"
        f"<<<description>>>: {d['description']}\n"
        f"<<<code>>>: {d['code_template']}"
        for d in retrieved_docs[:k]
    ])
    prompt = (
        f"You are a helpful assistant that selects the right API.\n\n"
        f"### Available APIs (top-{k} retrieved):\n{docs_str}\n\n"
        f"### User query:\n{query}\n\n"
        f"### Response (use the API + give code):\n"
    )
    return prompt

# 训练时 retrieved_docs[0] 是 gold API；inference 时由 retriever 给出
example_query = "I want to translate English to Chinese."
example_docs = [{
    "name": "Helsinki-NLP/opus-mt-en-zh", "provider": "HuggingFace",
    "description": "English to Chinese translation model.",
    "code_template": "from transformers import pipeline\np = pipeline('translation_en_to_zh', model='{name}')",
}]
print(build_rat_input(example_query, example_docs))
```

要点：训练时 retrieved doc 是 gold（确保 supervised signal 干净）；inference 时换成真 retriever 输出。这样训出的 model 即使遇到没见过的新 API，只要 retriever 能取到 doc，就能 zero-shot 调对（因为它学的是"看 doc 抽 API name + args"这个**动作**而不是"硬记 API 名"）。

### 8.3 ToolACE 风格 data synthesis（多 agent 合成 + 校验）

```python
# toolace_data_synthesis.py
import json
from openai import OpenAI

client = OpenAI()

def synth_tool_sample(tool_def, mode="single"):
    """用 LLM 合成 (tool_def, query, expected_call) 三元组。"""
    # Step 1: User Agent 生成 user query（按 mode 决定难度）
    user_prompt = (
        f"Given this tool:\n{json.dumps(tool_def, indent=2)}\n"
        f"Generate a realistic user query that requires this tool. "
        f"Mode: {mode} (single/parallel/sequential/refuse).\n"
        f"For 'refuse' mode, generate a query that this tool CANNOT solve."
    )
    query = client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": user_prompt}],
    ).choices[0].message.content

    # Step 2: Assistant Agent 生成 expected tool call（refuse 模式直接给 reason）
    if mode == "refuse":
        expected = {"call": None, "reason": "tool not applicable"}
    else:
        asst_prompt = (
            f"Tool:\n{json.dumps(tool_def)}\nQuery: {query}\n"
            f"Output ONLY valid JSON of the tool call: {{\"name\":..., \"arguments\":{{...}}}}"
        )
        call_str = client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": asst_prompt}],
            response_format={"type": "json_object"},
        ).choices[0].message.content
        expected = {"call": json.loads(call_str)}

    # Step 3: Validator 校验 schema（简化版：检查 name + required args）
    if expected["call"] is not None:
        required = tool_def["function"]["parameters"].get("required", [])
        args = expected["call"].get("arguments", {})
        if not all(r in args for r in required):
            return None  # required 字段缺失 → 丢弃
        if expected["call"]["name"] != tool_def["function"]["name"]:
            return None  # tool name 错 → 丢弃
    return {"tool": tool_def, "query": query, "expected": expected, "mode": mode}
```

要点：multi-agent 框架的核心是**生成与校验解耦**——User Agent 生成需求、Assistant Agent 生成 call、Validator 严格 check。任何一步 fail 都丢样本，**只保留 100% schema-clean 的样本进 SFT**。生产里 Validator 还要包括"真去调一次 tool 看返回是否合法"，把假 call 也过滤掉。

### 8.4 BFCL 提交 format

```python
# bfcl_submission.py
import json

def to_bfcl_format(model_output, item_id):
    """把 model 输出的 tool_calls 转成 BFCL 评测可接受的格式。"""
    # model_output 形如：[{"name":"get_weather","arguments":{"city":"Beijing"}}]
    return {
        "id": item_id,
        "result": [{call["name"]: call["arguments"]} for call in model_output],
    }

# 单 call 示例
sample_1 = to_bfcl_format(
    [{"name": "get_weather", "arguments": {"city": "Beijing", "unit": "celsius"}}],
    item_id="simple_001",
)
# parallel 示例：一个 turn 多个 call
sample_2 = to_bfcl_format(
    [{"name": "get_weather", "arguments": {"city": "Beijing"}},
     {"name": "get_weather", "arguments": {"city": "Shanghai"}}],
    item_id="parallel_002",
)
print(json.dumps([sample_1, sample_2], ensure_ascii=False, indent=2))
```

---

## 9. 工程踩坑与经验

- ❗ **数据合成质量决定 model tool use 上限**——bad data 训不出 good tool use。tool name 错 / 参数 schema 错 / args 不在 enum 内的样本，training 时不会自动报错，但会污染 model 的 tool call 分布。**铁律**：合成 pipeline 的 Validator 必须严格——schema check + required field check + （可选）真调用一次，任何一项 fail 都丢。宁可 60k 高质量数据，不要 500k 含噪数据
- ❗ **Tool 数量 > 50 时 retrieval-augmented 必备**——把 100 个 tool 的 schema 全塞进 system prompt，单个 prompt 轻松 5k+ token，不仅烧钱（每 turn 都重新算 attention 在所有 tool def 上）还会让 model 的 tool selection 准确率显著下跌（attention 在 100 个候选上分散）。Gorilla / xLAM-fc-r 的 retrieval 思路是 production 必备：用 embedding retriever 先取 top-5 ~ top-10 个候选 tool，再把这小集合喂给 LLM 决策
- ❗ **Negative example（"不调 tool"）必须有，否则 model 见啥都调**——如果训练数据里所有 query 都需要调 tool、所有 assistant 都以 `<tool_call>` 开头，model 学到的隐含模式是"看到 query → emit tool call"，连 "你好" 也调 search。修复：保留 10-15% "system 给 tool list 但 query 不需要 / tool 解决不了"的样本，assistant 直接 chat 回答而不输出 `<tool_call>`。BFCL 的 `irrelevance` sub-track 专门测这个能力
- ❗ **Multi-step trajectory 训练比 single-turn 重要**——真实 task 几乎都是 multi-step（先 search 拿候选、再 read 看详情、再 summarize 答复）。开源 tool 数据集（如 ToolBench 早期版本）80%+ 是 single-turn single-tool，model 学完在多步任务上得分接近 0。**至少要 30%+ 的训练数据是 multi-turn 的**，且要包含错误恢复（前一步 obs 报错后换工具或换参数重试）
- ❗ **Tool generalization 是研究热点——训过的 tool 与新 tool 性能差异大**——同一 model 在"训练时见过的 tool"上准确率 90%，换一组从没见过的 tool（schema 完全新）可能掉到 60%。这是**过拟合 tool name** 的典型症状。缓解方法：(1) RAT 风格，让 model 学"看 doc 调"而非"硬记 API" (2) 数据合成时大量轮换 tool name 与 args，让 model 接触 schema 多样性 (3) 评测时必须用 held-out tool 集合，不能只在训练分布上看分
- ❗ **Function call 格式必须与下游推理框架一致**——OpenAI 用 message 的 `tool_calls` 字段、Anthropic 用 `content=[{type:"tool_use",...}]` block、Qwen 用 `<tool_call>...</tool_call>` XML、Llama-3.1 用 `<|python_tag|>{"name":...}` ——**SFT 时按哪个格式训，inference 时框架就必须按同一个格式 parse**。错位 = 训了等于没训（呼应 8.5 §3.2 与 13.4 §2.3）
- ❗ **Tool calling SFT 后通用 chat 能力可能下降，要 mix 通用数据**——纯 tool 数据 SFT 出的 model 在 MT-Bench / Arena-Hard 上能掉 0.3-0.5 分（chat 流畅度 / format 多样性退化）。社区甜点是 **chat 60% + tool 30% + reasoning 10%** 的 mix（与 8.5 §2.2 一致），别贪心把 tool 比例推到 50%+
- ❗ **用 GPT-4 蒸馏 tool calling 数据有 ToS 风险**——OpenAI 的 ToS 明确禁止"用 GPT-4 输出训练 competing model"。学术 paper 用 GPT-4 蒸馏没人查，商用上线可能踩雷。建议用 **DeepSeek-V3 / Qwen2.5-72B / Llama-3.1-405B** 这类开源模型做 teacher，schema 一样、能力相当、license 干净。xLAM 后续版本就专门做了 DeepSeek teacher 的 ablation 证明效果不输 GPT-4
- ❗ **Tool description 写法直接影响 SFT 效果**（呼应 13.4 §2.2）——tool description 是训练数据的一部分，质量直接决定 model 学什么。description 写 "Search the web."（5 token） vs 写 "Search the public web for recent news. Use this when the user asks about events after training cutoff. Do NOT use for code questions."（60 token），SFT 后 model 在 irrelevance 维度差 10%+。合成 tool def 时不要省 description
- ❗ **多 tool 并行调用的训练数据稀缺，一定要显式合成**——开源 tool 数据集 80%+ 是 single call，model 学完不会 parallel call（"对比北京、上海、广州的天气"会被串成 3 轮单 call）。如果目标场景需要并行，必须显式合成 parallel 样本（一个 assistant turn 内并列 2-5 个 `<tool_call>`），至少占总 tool 数据 15-20%。ToolACE 的"维度配比"思想就是为此

---

## 10. 经典 paper

- **Schick et al., 2023 — *Toolformer: Language Models Can Teach Themselves to Use Tools*** — tool learning 的奠基作。读 §2 的 self-supervised data 合成 4 步 + §3 的 perplexity-based filter 设计。Take-away：理解"用 LM 自己当 judge"的 self-supervised 范式，是后续所有自动化数据合成的祖师爷
- **Patil et al., 2023 — *Gorilla: Large Language Model Connected with Massive APIs*** — retrieval-augmented tool LLM 的代表。读 §3 的 RAT 训练流程 + §5 的 retrieval mode 对比（zero-shot / BM25 / GPT-Embedding）。Take-away：理解"训练时 gold doc + inference 时 retrieved doc"的范式，以及为什么 tool 数 > 50 时 retriever 必备
- **Qin et al., 2023 — *ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs*** — ToolBench 数据集 + ToolLLaMA 模型，把 tool LLM 推到真实 RapidAPI 16k+ scale。读 §3 DFSDT 的搜索式 trajectory 生成 + §4 的 I1/I2/I3 任务分类。Take-away：理解 multi-step trajectory 数据怎么自动合成、为什么 DFSDT 比 CoT teacher 高 30%
- **Liu et al., 2024 — *ToolACE: Winning the Points of LLM Function Calling*** — 多维度 tool 数据合成的范式作。读 §3 multi-agent 合成 framework + §4 的 11 类能力维度配比。Take-away：理解"tool calling 不只是单 tool 单调用，必须显式按正交维度配比合成"，ToolACE-8B 在 BFCL 上能打过 GPT-4
- **Liu et al., 2024 — *xLAM: A Family of Large Action Models to Empower AI Agent Systems*** — Salesforce 出品，最完整的开源工业 agent SFT 方案。读 §3 数据合成 pipeline + §4 训练 setup + §5 BFCL 评测。Take-away：理解全 size（1B/7B/8x7B/70B）系列模型的 training recipe，xLAM-7B-fc-r 至今仍是 7B 规模 BFCL SOTA 之一
- 加分阅读：**Yan et al., 2024 — Berkeley Function Calling Leaderboard (BFCL) v1/v2/v3** 评测协议；**Yao et al., 2024 — τ-bench** 的 end-to-end agent 评测；**[BFCL 官方 leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html)** 持续更新

---

## 11. 自测与面试题

**Q1（演化时间线）**：Toolformer / Gorilla / ToolLLaMA / ToolACE 各自的核心创新是什么？为什么要演化出下一代？

<details>
<summary>Answer sketch</summary>

按"解决了上一代什么问题"的视角串起来：

- **Toolformer (2023.03)**：奠基作，**self-supervised 用 perplexity 比较自动标 API call 位置**——不需要人工标注，base LM 自己当 judge。**局限**：只支持 single-call、tool 数十几个、不支持复杂 schema → 催生 Gorilla
- **Gorilla (2023.05)**：扩到 1600+ HuggingFace/TF Hub/PyTorch Hub ML API；**核心创新 RAT (Retrieval-Augmented Fine-Tuning)**——训练时把 retrieved API doc 加进 input，让 model 学"看 doc 调 API"而非硬记 API name；**zero-shot 到新 API**。**局限**：仍是 single-step、API 类型窄（ML API） → 催生 ToolLLaMA
- **ToolLLaMA / ToolBench (2023.07)**：把 tool 数推到 **16k+ 真实 RapidAPI**，覆盖 49 category；**核心创新 DFSDT (Depth-First Search-based Decision Tree)**——用 ChatGPT 当 teacher 做搜索式 multi-step trajectory 生成，比 CoT baseline 高 30%；分 I1/I2/I3 任务难度。**局限**：数据 happy path 居多、parallel/refuse 维度不足、API 不稳定（已被 StableToolBench 修复） → 催生 ToolACE
- **ToolACE / xLAM (2024)**：data-centric 工业化——**核心创新是显式按正交能力维度配比合成**（single 50% + parallel 15% + sequential 15% + multi-step 10% + refuse 10%）；ToolACE 用 multi-agent 合成 + Validator 严格校验；xLAM 把这套做成全 size（1B/7B/8x22B/70B-fc-r）发布。**结果**：ToolACE-8B / xLAM-8x22B 在 BFCL 上能打过 GPT-4

加分要点：
- 能指出 4 代的"驱动力"——data scale ↑、能力维度 ↑、retriever-aware ↑
- 能指出 2025+ 第 5 代是 SFT + Agent RL（与 Module 15 衔接）
- 能联系 13.4 / 8.5：13.4 是 inference 协议、8.5 是端到端 SFT 工程、本节是数据范式演化
- 能指出 4 代都没有解决的核心问题：tool generalization（训过 tool 与新 tool 性能 gap），这是研究热点

</details>

**Q2（实战 pipeline）**：你要训一个能调 100 个企业内部 API 的 agent，base 选 Qwen2.5-7B，列出从 0 到 production 的完整数据 + 训练 pipeline。

<details>
<summary>Answer sketch</summary>

完整 9 步 pipeline：

**数据准备阶段**

1. **Tool definition 整理**：把 100 个 API 写成 OpenAPI / JSON schema，每个 tool 包含 name + 详细 description（正例 / 负例 / 参数语义都要写，呼应 13.4 §2.2）+ parameter schema（type / enum / required 都精确）
2. **Tool retriever 准备**：tool 数 100 > 50，必须 retrieval-augmented（呼应 Gorilla / xLAM-fc-r 思路）。用 bge / E5 / OpenAI embedding 给每个 tool description embed，建一个小 vector DB；inference 时按 user query 取 top-5 ~ top-10 个候选 tool 喂 LLM
3. **数据合成**：用 ToolACE / xLAM 风格 multi-agent 合成 pipeline——
   - Tool Self-Evolution：基于 100 个 tool 生成 query（按 mode：single 50% / parallel 15% / sequential 15% / multi-step 10% / refuse 10%）
   - Assistant Agent（用 DeepSeek-V3 而非 GPT-4，避免 ToS 风险）生成 expected tool call
   - Validator 严格 check：name 在白名单、required 字段齐、enum 合法、能否真调通过
   - 总量目标 30k-50k 高质量样本

**训练阶段**

4. **Stage 2 通用 SFT 基线**：Magpie-Pro 60k + Tülu-3-SFT 30k + NuminaMath 10k 先训通用 chat + reasoning 能力（如果 base 是 Qwen2.5-7B-Instruct 则可跳过这步，已经是 chat model）
5. **Stage 3 tool calling SFT**：将上一步 mix 与 tool 数据按 chat 60% + tool 30% + reasoning 10% 比例混合，按 Qwen2.5 chat template + `<tool_call>` 协议统一格式（呼应 8.5 §3.1 / §6.1）
6. **训练超参**：LoRA r=32（agent SFT 比纯 chat SFT 需要更大 r，呼应 8.5 §9）、lr=1e-4 / cosine decay / warmup 5%、bf16 + FA2 + grad checkpointing、max_seq_len=8k（tool def 长）、effective bs=32、2 epoch
7. **Loss mask**：tool_call 段算 loss、final answer 段算 loss、tool obs 段全 mask、`<|im_end|>` 算 loss（详见 8.5 §4）

**评测 + 上线**

8. **多维评测**：(1) BFCL v3 看 overall + 5 个 sub-track 单独分；(2) 自定义 mixed eval 50-100 条覆盖 happy path / refuse / parallel / hallucinated tool；(3) MT-Bench 看 chat 是否 regression；(4) 用 held-out tool 集（训练没见过的 5-10 个）测 generalization
9. **判断是否 Stage 4 RL**：BFCL > 80% 且失败集中在"决策类问题"（多步规划 / 错误恢复）则上 Module 15 的 Agent RL；如失败集中在 single call 准确率 / format 错则回头加 SFT 数据

加分要点：
- 提到 retriever-aware training（训练时 system prompt 里 tool 数动态变化 5-30 个）让 model 鲁棒
- 提到 contamination check（不要把 BFCL eval 数据漏进训练）
- 提到 inference 时配 constrained decoding 把 JSON 合法率推到 100%（13.4 §5）
- 提到 tool obs 长度截断 / summarize（避免 context bloat 让后续 turn TTFT 高）
- 提到不同来源数据先 unify 到统一 chat template 再训（铁律，8.5 §9 反复强调）

</details>

**Q3（前沿衔接）**：SFT + Agent RL 的衔接（Module 15 预告）：什么时候 SFT 够、什么时候必须 RL？给出判断标准。

<details>
<summary>Answer sketch</summary>

**先错误归因**（不归因直接选 SFT/RL 都是猜）——抽 100 条 BFCL 失败 case 人工分类失败原因，再决定。

**SFT 够（不需要 RL）的情况**：

1. **失败集中在"未覆盖的 pattern"**：如 nested JSON 参数错、某个 sub-track（parallel call / sequential call / refuse）显著低 → 补对应 pattern 的 SFT 数据更直接
2. **base 还有空间**：SFT 数据从 100k 扩到 500k 还在涨分（loss 还在降）→ 先把 SFT scale 拉满
3. **tool 协议变化**：新增 tool / 改了 schema → 必须 SFT 教 model 新格式
4. **业务对决策容错高**：调错 tool 也无大碍（如内部 demo / 探索性场景） → SFT 到 80% 已足够，多余精力不如做 retriever 优化

**必须上 RL 的情况**：

1. **失败是系统性的"探索-反馈"问题**：tool 调失败后 model 不会重试 / 不会换 tool / 不会反问用户 → 这种"决策类 multi-step"问题 SFT 数据很难穷举所有 trajectory，**trajectory-level reward 才能教**
2. **有可执行 reward signal**：BFCL 的 executable accuracy / 自己 sandbox 跑真 tool 拿成功率 / unit test pass → reward signal 干净时 RL 收益最大
3. **SFT 已饱和**：数据从 500k 加到 1M 不再涨分、loss 早已平台 → 容量瓶颈不在数据量，在"决策边界"
4. **multi-turn planning 而非 single-step tool call**：trajectory-level 优化才有意义，单步 SFT 学不出"先 plan 再 execute"

**Trade-off 边界**：

- RL 比 SFT 复杂 5-10 倍工程量（rollout infra / reward design / KL 约束 / reward hacking 防御，参考 9.6）。SFT 还能涨分就别轻易上 RL
- RL 容易 reward hacking——只用 schema 合法度做 reward，model 学会输出"格式合法但语义错"的 tool call。reward 设计要 careful（pass + format + 步数惩罚多目标融合）
- **80% → 90% 是常见 SFT-to-RL 交接点**——再加 SFT 数据边际收益急剧下降，RL fine-tune 几千 step 通常能再涨 5-10 个点
- 实务最优解：**先 SFT 到 80% → 再 RL fine-tune 5-10 个点**——这是 Agent-FLAN / FireAct / xLAM-fc-r 后期版本的标准范式（Module 15.1 详讲）

加分要点：
- 能指出"判断 SFT 饱和"的具体信号——eval loss 平台、不同 lr 下 final score 一致、增加 10× data 提升 < 0.5%
- 能提到 RL 阶段仍要混 SFT data 做 reference KL（防漂移）
- 能指出 2025 年 R1-style RLVR (verifier reward) 与 GRPO 是 agent RL 的事实标准（Module 9.5 / 10.3 / 15.2 衔接）
- 能提到 MCP（Model Context Protocol）作为 2025+ 的标准化 tool 接口（13.4 §7），未来 tool 训练数据可能直接基于 MCP server 的 capability metadata

</details>

---

## 12. 延伸阅读

- [Toolformer 复现 GitHub](https://github.com/lucidrains/toolformer-pytorch) — Toolformer 的开源 PyTorch 复现，理解 perplexity-based filter 的实现细节
- [Gorilla GitHub](https://github.com/ShishirPatil/gorilla) — Berkeley Gorilla 的代码 + 数据 + retriever，含 RAT 训练 recipe
- [ToolBench / ToolLLaMA GitHub](https://github.com/OpenBMB/ToolBench) — ToolBench 16k API 数据集 + DFSDT 实现 + 评测脚本
- [StableToolBench](https://github.com/THUNLP-MT/StableToolBench) — 解决 ToolBench API 不稳定问题，复现性优先
- [ToolACE Hugging Face dataset](https://huggingface.co/Team-ACE) — ToolACE 11k 多维度 tool 数据集，可直接 fine-tune
- [xLAM GitHub & paper](https://github.com/SalesforceAIResearch/xLAM) — Salesforce 完整 agent SFT pipeline + 数据 + 全 size 模型
- [BFCL Leaderboard 与代码](https://gorilla.cs.berkeley.edu/leaderboard.html) — Berkeley Function-Calling Leaderboard，评测事实标准
- [τ-bench GitHub](https://github.com/sierra-research/tau-bench) — Yao 等 2024 提出的真实场景 agent eval（客服 / 航空），最贴近 production
- [Hermes Function Calling V1](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1) — NousResearch 出品，已按 ChatML + `<tool_call>` 格式准备好的开源 tool 数据
- 推荐继续读本教程的 **8.5 节《SFT 实战：多轮+tool 混合训练》** —— 拿到本节范式产生的数据后端到端 SFT 工程；**13.4 节《Function calling 工程》** —— inference 时的 tool 协议；**Module 15.1《Agent SFT》**与 **15.2《多轮 PPO/GRPO》** —— SFT 之后用 trajectory-level RL 把 tool use 推到 90%+ 的下一阶段
