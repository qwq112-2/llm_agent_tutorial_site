---
title: "15.4 Reasoning + Agent：Search-R1 / ReSearch / ReTool / Agent-R1"
description: "把 10.3 的 R1 范式（GRPO + verifier reward 涌现 long-CoT）从 single-turn math/code 迁移到 multi-turn agent 任务（search / tool / browser），就得到 2025 年最热的 Reasoning Agent 路线——Search-R1 / ReSearch / ReTool / Agent-R1 / "
---

> ⏱ 预计阅读 65 分钟 ｜ 难度 ★★★ 🔥 ｜ 前置：10.3 RLVR/DeepSeek-R1、15.2 多轮 PPO/GRPO

## 一句话本节讲什么

把 10.3 的 R1 范式（GRPO + verifier reward 涌现 long-CoT）从 single-turn math/code 迁移到 multi-turn agent 任务（search / tool / browser），就得到 2025 年最热的 **Reasoning Agent** 路线——Search-R1 / ReSearch / ReTool / Agent-R1 / SWE-RL 是这条线的代表作；它们的算法骨架完全是 15.2 的 multi-turn GRPO，独特性在于"task 成功"本身可被 verifier 客观判定（answer match / unit test / page state），从而把 R1 的"不可 hack 的 reward signal"红利完整搬到 agent 场景。本节讲清"R1 → Agent"的迁移动机、4 个代表工作的差异、现代 reasoning agent 的 6-stage 训练 pipeline、composite reward 设计、以及 2025-2026 的开放问题。

---

## 1. Mental model（直觉）

### 1.1 R1 范式的 takeaway 与"自然下一站"

10.3 的核心 finding：**verifier-based reward + GRPO** 在 verifiable task 上能让 base model 自发涌现 long-CoT、self-reflection、error correction（Aha moment）——而且**完全不需要 RM**，比传统 RLHF 便宜、稳、不可 hack。

但 10.3 的 R1 跑的是 **single-turn 数学题/code 题**：给一个 prompt，输出一段 thinking + answer，verifier 判定对错。这一范式的**自然下一问**是：

> agent 任务（search / tool / web / SWE）也常有 verifiable outcome（task 成功 / unit test 通过 / 答案命中），能不能把 R1 + GRPO 直接搬过去？

答案是肯定的——而且实操上**算法骨架几乎不需要改**：把 single-turn trajectory 推广成 multi-turn `assistant ↔ tool` trajectory（15.2 已搭好）、把 verifier 从 sympy 换成"task 成功判定 / unit test runner / page state 检查"，就得到了现代 Reasoning Agent 训练范式。

把这两章合起来看：

```
10.3 R1            +   15.2 multi-turn GRPO     =   15.4 Reasoning Agent
─────────              ──────────────────────       ──────────────────────
GRPO + verifier        trajectory + loss mask      Search-R1 / ReSearch /
single-turn CoT        async rollout + sandbox     ReTool / Agent-R1 / SWE-RL
math / code            assistant ↔ tool ↔ ...      multi-turn long-CoT + tool
"Aha moment"           outcome reward broadcast    "Aha + tool composition"
```

记住一句话：**Reasoning Agent = R1 范式 × multi-turn agent 工程**。本节讲的 4 个代表工作都是这条公式不同 axis 的实例化。

为什么这个迁移不是 trivial：很多人以为"既然算法骨架不变，那把 R1 跑在 multi-turn agent 上就是工程的事"——这是低估了。multi-turn 引入了三个 R1 时代不存在的难题：(a) **trajectory 比 R1 长 5-10 倍**（10k → 30k token），显存与 KL 都要重新调；(b) **reward 比 R1 更 sparse**（agent task 需要多步正确才成功，R1 只要最终答案对），advantage variance 大；(c) **environment 不可控**（tool API 限流 / 失败 / 慢），train pipeline 必须 async + sandbox。这三个是 §4 工程踩坑反复出现的根因。

### 1.2 为什么 2025 年突然涌现这一波？

3 个 enabler 同时到位：

1. **R1 (2025.01)** 把"verifier + GRPO 涌现 long-CoT"范式公开 → 立刻有人尝试搬到 agent
2. **verl / OpenRLHF (2024-2025)** 提供生产级 multi-turn rollout 引擎（async + sandbox + DAPO），把 15.2 的工程门槛从月级降到周级
3. **Qwen2.5 / Qwen3 / Llama-3 base** 提供能扛 32k+ context、tool format 友好、math/code 基本功扎实的 base model

任何一个 enabler 缺位（比如 2024 年没有 verl，多轮 rollout 都得自己写）这一波就起不来。**Search-R1（2025.03）→ ReSearch（2025.04）→ ReTool（2025.04）→ Agent-R1（2025.05）→ SWE-RL（2025.02）**——半年里集中爆发，全部踩在这三个基础上。

值得注意的是，这一波 reasoning agent 与 14.x 章讲的 ReAct / Toolformer / xLAM 等"agent SFT 路线"是**互补而非替代**关系：SFT 教 model **怎么用 tool**（schema、模板、基本流程），RL 教 model **什么时候用 tool、用几次、怎么从 tool 失败中 recover**。SFT 决定下限、RL 决定上限——所以 §2.3 的 6-stage pipeline 把 agent SFT cold start 放在 RL 之前，这是工业界共识。

### 1.3 Mental model 图：从 R1 到 Reasoning Agent

```
   R1 (10.3)                              Reasoning Agent (本节)
   ─────────                              ────────────────────────
   prompt                                 user query + tool defs
     │                                      │
     ▼                                      ▼
   <think>...</think>                   ┌──────────────────────┐
   <answer>...</answer>                 │ <think>...</think>   │
     │                                  │ <tool_call>...</tool>│ ──→ search engine
     ▼                                  │ <observation>...</obs│ ←── results
   verifier(answer)                     │ <think>...</think>   │
     │                                  │ <tool_call>...</tool>│ ──→ python sandbox
     ▼                                  │ <observation>...</obs│ ←── stdout
   r ∈ {0, 1}                           │ <think>...</think>   │
                                        │ <answer>...</answer> │
                                        └──────────────────────┘
                                              │
                                              ▼
                                          verifier(final answer / task state)
                                              │
                                              ▼
                                            r ∈ {0, 1}
```

差别全在中间的 trajectory 形态：R1 是"一段长 thinking + 答案"，Reasoning Agent 是"think → call tool → observe → think → call tool → ... → answer" 的循环。reward 给法、GRPO 算法、verifier 思想完全一样。

---

## 2. 公式与原理

### 2.1 算法骨架：直接复用 15.2 multi-turn GRPO

设 task $x$，policy 采样 $G$ 条完整 multi-turn trajectory $\{\tau_1, \dots, \tau_G\}$，每条 trajectory 末尾打 verifier reward $r_i = R(\tau_i)$，组内归一化得 trajectory-level advantage：

$$\hat A_i = \frac{r_i - \bar r}{\sigma_r + \epsilon}, \quad \bar r = \frac{1}{G}\sum_i r_i$$

把 $\hat A_i$ broadcast 到 trajectory $\tau_i$ 的所有 assistant token（observation token 经 segment mask $m_{i,t}$ 屏蔽）：

$$L = -\mathbb{E}\!\left[ \frac{1}{G}\sum_{i=1}^{G} \frac{1}{\sum_t m_{i,t}} \sum_{t=1}^{T_i} m_{i,t} \cdot \min\!\big(\rho_{i,t} \hat A_i,\ \text{clip}(\rho_{i,t}, 1{-}\epsilon, 1{+}\epsilon)\hat A_i\big) \right] + \beta \cdot D_{\text{KL}}^{\text{masked}}$$

其中 $\rho_{i,t} = \pi_\theta / \pi_{\theta_{\text{old}}}$，$D_{\text{KL}}^{\text{masked}}$ 仅在 assistant token 上计算（k3 估计，同 9.5）。

**这就是公式全部** ——本节 4 个代表工作（Search-R1 / ReSearch / ReTool / Agent-R1）的算法主体都是这个式子，差异不在算法，在 **environment（什么 tool）+ reward（怎么 verify）+ task 数据（什么 task）** 三处。

### 2.2 Reasoning Agent 的 reward 设计：Composite reward

R1 用的是 `accuracy + format` 两项加权和。Reasoning Agent 上多 1-2 项变成 4 项 composite：

$$R(\tau) = \underbrace{R_{\text{acc}}(\tau)}_{\text{outcome}} + \lambda_{\text{fmt}} \cdot R_{\text{fmt}}(\tau) - \lambda_{\text{len}} \cdot \text{len}(\tau) - \lambda_{\text{tool}} \cdot N_{\text{tool}}(\tau)$$

各项分别为什么需要：

| 项 | 形式 | 防什么 hacking |
|---|---|---|
| **Accuracy / Outcome** | $\mathbb{1}[\text{verifier pass}]$ | 主信号，必给 |
| **Format** | $\mathbb{1}[\text{合法 tool\_call schema}]$ | 防 model 输出 garbage tool call、parse fail 失败累积 |
| **Length penalty** | $\propto \text{len}(\tau)$ | 防 long-CoT 失控（10k → 100k token），cost 爆炸 |
| **Tool call cost** | $\propto N_{\text{tool}}(\tau)$ | 防 tool spam（不确定就疯狂 search 兜底） |

**权重原则**：accuracy 永远主导（系数 = 1.0），format 给小正向（0.1-0.3），length / tool call 给小负向（0.0005-0.005）。**任何一项扣分系数大于 accuracy 主项，policy 都会优先优化那一项而不是真正解题**——R1 论文也强调过这点。

很多论文（Search-R1、ReSearch）实测发现：**format reward 在训练初期权重大一点（0.5-1.0）让 model 先学会 schema，后期降到 0.1**，否则纯 0/1 outcome 信号让 model 早期连 tool call 都调不对。

### 2.3 现代 Reasoning Agent 训练 6-stage pipeline

工业级（不是论文 demo 级）训练一个 production reasoning agent 几乎都走这 6 个 stage。串起来 = **R1 的 4 阶段 pipeline + 多轮 agent SFT 与 capability refresh**。

```
┌────────────────────────────────────────────────────────────────────┐
│ Stage 1: Base Model（如 Qwen3-7B / Qwen2.5-7B / Llama3-8B）        │
│   预训练好的 base，不要直接拿 instruct 版                            │
└────────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────────────┐
│ Stage 2: 通用 SFT（Magpie / Tülu-3 / OpenHermes）                   │
│   先把 base 训成"会聊天 + 会跟随指令"——避免后面 agent SFT 把通用    │
│   能力洗光。可跳过（如果 base = instruct 版），但生产线推荐做       │
└────────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────────────┐
│ Stage 3: Agent SFT cold start（含 long-CoT + tool format）         │
│   数据：ToolBench / xLAM / Magpie-tool / R1 蒸馏出的 agent CoT     │
│   目的：教 ChatML / <think>/<tool_call>/<observation> 格式 +       │
│         基本 reasoning + tool 模板。规模 10k-100k 条               │
└────────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────────────┐
│ Stage 4: Multi-turn GRPO + verifier 【核心】                       │
│   环境：search engine / python sandbox / web browser / SWE repo    │
│   reward：composite（acc + fmt - len - tool）                      │
│   算法：multi-turn GRPO（15.2 §3.3）+ DAPO 优化                    │
│   规模：1-10k task，每 task G=8 trajectory，几千-几万 step         │
└────────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────────────┐
│ Stage 5: 通用 Chat refine（防能力退化）                            │
│   再走一遍通用 SFT + 轻量 RLHF（RM 给 reward），把可能被 agent RL  │
│   洗掉的 chat / refusal / safety 能力补回来。呼应 R1 Stage 4       │
└────────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────────────┐
│ Stage 6: 评测 BFCL / SWE-bench / GAIA / WebArena / TAU-bench       │
│   分别评 tool calling / 代码 agent / 通用 agent / web agent / 多轮 │
│   不是单个 benchmark 高就完事，要看 cross-benchmark 一致性          │
└────────────────────────────────────────────────────────────────────┘
```

口诀：**Base → 通用 SFT → Agent SFT → Agent RL → Chat refine → Eval**。Stage 1+2 可合并（用 instruct 版直接进 Stage 3），Stage 5 在科研复现里常省略，但生产线必做。

### 2.4 4 个代表工作对比

| 工作 | 出品方 | 核心环境 | Reward verifier | Base model | 创新点 |
|---|---|---|---|---|---|
| **Search-R1** | UIUC（Hou et al., 2025） | Wikipedia search | exact match(answer, gold) | Qwen2.5-7B-Base | 第一个把 R1 范式公开搬到 multi-turn search agent |
| **ReSearch** | 中科院（Chen et al., 2025） | 多 hop search + reasoning | F1 / EM on multi-hop QA | Qwen2.5-7B-Base | 强调 multi-hop reasoning + search 协同，HotpotQA SOTA |
| **ReTool** | 蚂蚁（Feng et al., 2025） | calculator + python interpreter | math answer match | Qwen2.5-Math-7B | 通用 tool reasoning RL，不限 search |
| **Agent-R1** | 清华（Wang et al., 2025） | search + RAG + web 多环境 | task-specific verifier | 多 base 可选 | 通用 agent RL **框架**，跨 task 跨 domain 训练 |
| 选读：**SWE-RL** | Meta（Pan et al., 2025） | SWE repo + pytest | unit test pass rate | Llama3-70B | code agent multi-turn RL，SWE-bench |

读法：**Search-R1 是范式开端**（最值得读源码），ReSearch 是"难任务上 fine-grained 优化"代表，ReTool 是"超出 search 的通用 tool"代表，Agent-R1 是"框架级别开箱即用"代表，SWE-RL 是"工业级 code agent"代表。

#### Search-R1 详例

最简单也最 instructive。trajectory 长这样：

```
<|im_start|>user
Who composed the soundtrack for the 1972 film "The Godfather"?
<|im_end|>
<|im_start|>assistant
<think>I need to search for the composer of The Godfather (1972).</think>
<search>The Godfather 1972 film composer soundtrack</search>
<|im_end|>
<|im_start|>tool
[SEARCH RESULTS]
1. "The Godfather (1972) - Soundtrack by Nino Rota..."
2. "Nino Rota composed the iconic theme..."
3. ...
<|im_end|>
<|im_start|>assistant
<think>Based on the search, Nino Rota composed the soundtrack.</think>
<answer>Nino Rota</answer>
<|im_end|>
```

- **数据**：HotpotQA / NQ / TriviaQA 的 (question, gold_answer) pair
- **Tool**：单个 `search(query) → top-K snippets`（K=5 / 10）；retriever 是 BM25 over Wikipedia 或 dense retriever（E5）
- **Reward**：`R = exact_match(answer, gold)`（也有用 cover_em：gold 出现在 answer 里）
- **GRPO**：`G=8`、`β=0.001`（multi-turn 必须小）、`max_turns=4`
- **效果**：Qwen-7B + Search-R1 在 NQ / HotpotQA / TriviaQA 平均 EM 显著超过 SFT-only baseline，与 GPT-4 retrieval-augmented 接近

最简洁的 reasoning agent 实例——**1 个 tool、1 个 verifier、1 个 reward**。其他工作都是它的扩展。

#### ReSearch 与 Search-R1 的细节差异

ReSearch（中科院 2025.04）几乎与 Search-R1 同时期发布，重点不同：
- 数据集偏 multi-hop（HotpotQA、Musique、2WikiMultiHopQA）——一道题需要多次 search 串起来
- Reward 用 F1（部分匹配也得分）而非 pure EM，对 long answer 更友好
- 更细致的 reasoning template 设计（强制每次 search 后写一段 reflection）
- 在 multi-hop QA 上比 Search-R1 高几个点，是该方向 SOTA

#### ReTool：从 search 到通用 tool

ReTool（蚂蚁 2025.04）把 Search-R1 的"单 search tool"扩到 calculator / python interpreter：
- 数据：MATH / AIME 数学题
- Tool：`python(code) → stdout`（沙箱跑），用于精确算术 / 验证中间步骤
- Reward：math answer verifier（同 R1）
- 创新：reasoning 中插入 tool call 让 model "**用程序检验自己**"，类似人类做数学题时拿计算器复核

效果：Qwen2.5-Math-7B + ReTool 在 AIME 上比纯 R1 高几个点——证明**让 reasoning model 学会工具是有意义的**，不是所有事都靠 long-CoT 算。

ReTool 的洞察很重要：**纯 long-CoT 在精确算术上有 fundamental limit**——你让 model 心算 `7 位数 × 7 位数` 注定不稳，但让它写一行 `print(1234567 * 7654321)` 就 100% 准。Reasoning + tool 不是炫技，是把 LLM 不擅长的事 offload 给确定性工具，把 long-CoT 留给真正需要 reasoning 的部分。这与人类做数学题"心算简单的、计算器算复杂的"是同一逻辑。

#### Agent-R1：框架级整合

Agent-R1（清华 2025.05，基于 verl）不是单篇 paper，而是开源 repo。提供 search / RAG / web / code 多种环境的开箱即用配置，跨 task 训练（一份 model 同时 train search + math + code）。

强调点：
- **Cross-domain generalization**——单 model 多任务训，泛化到 unseen task
- **配置驱动**——改 YAML 就能切环境 / reward / base model，不用改代码
- 落地率高，社区 fork / star 数仅次于 verl 自身

是当前**最适合自己复现"R1 风格 agent"的起点**。

#### 4 个工作的"共同公式"

抽象出来都是同一个 template：

> **Reasoning Agent = Base + Agent SFT cold start + multi-turn GRPO（环境 E、verifier V、composite reward R）**

变化的只有三件事：
- 环境 $E$：search engine（Search-R1）/ python sandbox（ReTool）/ docker repo（SWE-RL）/ web browser（GAIA-style）
- Verifier $V$：answer match / unit test / page state machine
- Reward weights：accuracy / format / length / tool_call cost 各项系数

任何"R1 + 新 tool" 类的论文你都可以套这个模板理解——本质都在做同一件事，差别在 plug 进什么 environment + verifier。**搞清楚这个 template 比记住每篇论文的细节更重要**。

---

## 3. 最小代码示例

### 3.1 Search-R1 trajectory 格式（22 行）

```python
# search_r1_trajectory_template.py
SYSTEM = """You are a helpful assistant. Use <think> for reasoning,
<search> to query a search engine, and <answer> to give the final answer."""

# 示例 trajectory，rollout 时按下面 schema 拼装
TRAJECTORY_TEMPLATE = """\
<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
<think>{thought_1}</think>
<search>{query_1}</search><|im_end|>
<|im_start|>tool
{search_results_1}<|im_end|>
<|im_start|>assistant
<think>{thought_2}</think>
<answer>{final_answer}</answer><|im_end|>
"""

# 关键点：
# 1. <think>/<search>/<answer> 是 R1 风格 schema，format reward 检查这三个 tag
# 2. <|im_start|>tool 段是环境注入的 search results，loss mask = 0
# 3. 每个 assistant turn 内都先 think 再决策（search 或 answer），强制 reasoning
```

### 3.2 Reasoning Agent reward function（24 行）

```python
# reasoning_agent_reward.py
import re

def search_r1_reward(trajectory: str, gold_answer: str,
                     w_acc: float = 1.0, w_fmt: float = 0.2,
                     w_len: float = 0.0005, w_tool: float = 0.05,
                     n_tool_threshold: int = 5) -> float:
    """
    Reasoning agent composite reward = accuracy + format - length - tool_spam.
    """
    # 1) Accuracy: exact match between extracted answer and gold
    m = re.search(r"<answer>(.*?)</answer>", trajectory, re.DOTALL)
    pred = m.group(1).strip() if m else ""
    r_acc = 1.0 if pred.lower() == gold_answer.lower() else 0.0

    # 2) Format: <think> 与 <search>/<answer> 必须正确出现
    has_think = bool(re.search(r"<think>.*?</think>", trajectory, re.DOTALL))
    has_answer = bool(re.search(r"<answer>.*?</answer>", trajectory, re.DOTALL))
    r_fmt = 1.0 if (has_think and has_answer) else 0.0

    # 3) Length penalty: trajectory 越长越扣分（防 length hacking）
    r_len = len(trajectory.split())  # token 估计

    # 4) Tool call cost: 超过阈值的 tool call 扣分（防 tool spam）
    n_tool = len(re.findall(r"<search>", trajectory))
    r_tool = max(0, n_tool - n_tool_threshold)

    return w_acc * r_acc + w_fmt * r_fmt - w_len * r_len - w_tool * r_tool
```

要点：4 项加权和，accuracy 是主项（系数 1.0），其余项的系数都比 accuracy 小至少 1 个数量级——否则 policy 会优先 hack 那项。生产里 length / tool 系数往往要根据训练曲线再 tune（见 §4 踩坑）。

### 3.3 Reasoning Agent GRPO 训练循环骨架（38 行）

```python
# reasoning_agent_grpo_loop.py
import torch
from search_r1_reward import search_r1_reward

def reasoning_agent_grpo_step(policy, ref_model, tasks, rollout_fn,
                              optimizer, G=8, beta=0.001, eps_clip=0.2):
    """
    Reasoning Agent multi-turn GRPO step (= 15.2 §3.3 + verifier reward).
    tasks: list[{"question": str, "gold_answer": str}]
    rollout_fn: (policy, question) -> (token_ids, segment_mask, decoded_text)
                内部跑 assistant ↔ search ↔ assistant 多轮，sandbox 执行
    """
    B = len(tasks)
    trajectories, masks, rewards = [], [], []

    # ===== Phase 1: rollout (G 条 / task) =====
    for task in tasks:
        for _ in range(G):
            with torch.no_grad():
                tok_ids, seg_mask, text = rollout_fn(policy, task["question"])
                # ★ verifier reward —— 把 R1 的 math_verify 换成 search_r1_reward ★
                r = search_r1_reward(text, task["gold_answer"])
            trajectories.append(tok_ids); masks.append(seg_mask); rewards.append(r)

    tok_ids = pad_stack(trajectories)                    # (B*G, T_max)
    masks   = pad_stack(masks)                           # (B*G, T_max)
    rewards_t = torch.tensor(rewards).view(B, G)         # (B, G)

    # ===== Phase 2: trajectory-level group advantage（15.2 §3.2）=====
    mean = rewards_t.mean(dim=1, keepdim=True)
    std  = rewards_t.std(dim=1, keepdim=True) + 1e-8
    advantages = ((rewards_t - mean) / std).unsqueeze(-1) * masks.view(B, G, -1)
    advantages = advantages.view(B*G, -1)

    # ===== Phase 3: GRPO loss with masked KL（15.2 §3.3）=====
    log_pi_old = compute_log_probs(policy, tok_ids, no_grad=True)
    log_pi_ref = compute_log_probs(ref_model, tok_ids, no_grad=True)
    log_pi_new = compute_log_probs(policy, tok_ids)
    ratio = torch.exp(log_pi_new - log_pi_old)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantages
    pg_loss = -((torch.min(surr1, surr2) * masks).sum() / masks.sum().clamp(min=1))
    log_u = log_pi_ref - log_pi_new
    kl = (((torch.exp(log_u) - log_u - 1) * masks).sum() / masks.sum().clamp(min=1))

    loss = pg_loss + beta * kl
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    return {"loss": loss.item(), "reward_mean": rewards_t.mean().item()}
```

要点：
- **核心改动相对 15.2 §3.3 仅 1 行**——把 `reward_fn` 换成 reasoning agent 的 composite verifier reward。算法部分照搬
- `β = 0.001` 比 single-turn 小 40 倍——multi-turn + sparse outcome reward 必须如此
- `G = 8` 是 long trajectory 上的工业甜点；想加 dynamic sampling（DAPO）剔除 std=0 的 group 进一步提升
- 配套生产 stack：**rollout 用 vLLM async serve、训练用 verl + DeepSpeed ZeRO-3 + LoRA**——8B 模型在 8×80G H100 上一天能跑完一个 epoch（1-3k task）

### 3.4 一键复现：用 verl 配置 Search-R1 风格训练

```yaml
# verl_search_r1.yaml（伪示意，真实配置见 verl/Agent-R1 repo）
algorithm:
  name: grpo
  group_size: 8
  kl_coef: 0.001
  clip_range: 0.2
  loss_type: dapo               # token-level + clip-higher

policy:
  model: Qwen/Qwen2.5-7B-Instruct
  lora: { r: 32, alpha: 64 }
  bf16: true

rollout:
  engine: vllm
  max_turns: 4
  max_obs_tokens: 2048
  temperature: 0.9
  tools:
    - name: search
      backend: bm25_wikipedia    # 或 dense_e5
      top_k: 5

reward:
  funcs:
    - { type: exact_match, weight: 1.0 }
    - { type: format_check, weight: 0.2 }
    - { type: length_penalty, weight: 0.0005 }
    - { type: tool_spam_penalty, weight: 0.05, threshold: 5 }

dataset:
  train: hotpotqa+nq+triviaqa
  eval:  hotpotqa_dev,nq_dev
```

verl / Agent-R1 都支持类似 YAML 配置，**改环境（search → python）+ reward + dataset 三处就能切到 ReTool / Agent-R1 等其他工作**。

---

## 4. 工程踩坑与经验

- ❗ **Reasoning agent trajectory 极长（10k-100k token），显存压力远大于 single-turn RLVR**——8 条 G × 30k token = 240k token 同时 backward。**必备组合**：(1) **LoRA only**（不全参，r=32-64），(2) **gradient checkpointing**（显存换 30% 算力），(3) **DeepSpeed ZeRO-3** 切 weights/optimizer，(4) **GQA / MQA 友好的 base**（Qwen2.5+ 自带），(5) **vLLM async generate** 与训练 GPU 隔离。少任何一个 8B 模型都跑不动 8×H100。
- ❗ **Search engine API rate limit 在训练时直接卡死 rollout**——SerpAPI 免费档每分钟几十 query，G=8 × max_turns=4 × batch=32 = 1024 query / step，秒级就炸 rate limit。**铁律**：(1) 训练前先把训练集所有 query 离线 cache 一份（response 全存 disk），训练时全程命中缓存；(2) 自部署 Wikipedia BM25 / 自部署 ES + dense retriever，不依赖外部 API；(3) 真正用 live API 仅在 eval 阶段。Search-R1 论文用的就是离线 Wikipedia BM25 retriever，不是 Google search。
- ❗ **Search results 必须可控长度截断（top-K 截断 + 单条 snippet 截断）**——`read_url` 一个长 page 直接 50k token 撑爆 context，一条 trajectory 在第 2 turn 就 OOM。统一在 tool wrapper 层做：`return [snippet[:512] for snippet in results[:top_k]]`，top_k=5、单 snippet ≤ 512 token 是保守值。**单 step OOM 比训得不好难调 10 倍**。
- ❗ **Verifier reward 必须 robust，否则 reward 噪声 = 训练崩**——math 用 sympy 不能纯字符串（`1/2 vs 0.5`）；search agent 用 cover_em / F1 而非 strict EM（`Nino Rota` vs `the composer Nino Rota` 应该都算对，strict EM 会判错）；code 用 unit test runner 但要 sandbox + timeout。Search-R1 论文实测 cover_em > strict EM 几个点，不是模型变强而是 reward 噪声变小。**verifier 漏洞 = reward hacking 入口**，与 10.3 一致。
- ❗ **Length penalty 调不好两头都崩**——系数太小（< 1e-4）：long-CoT 失控，trajectory 从 5k 涨到 50k，cost 爆炸；系数太大（> 1e-2）：think 段被压到 1-2 句话不够 reasoning，accuracy 反而下降。经验值 `1e-4 ~ 1e-3` 之间，**先用 1e-3 跑、看 avg_traj_len 曲线再 tune**。同样的事也发生在 tool call cost——threshold 设 5 是中位经验，但任务难就要放宽到 8-10。
- ❗ **Cold start 数据 quality 是 6-stage pipeline 的瓶颈**——Stage 3 agent SFT 数据脏（错误 tool call schema、reasoning 断在中途、observation 顺序乱），Stage 4 RL 直接学坏。常见来源：(1) **R1 / o1 / Claude 蒸馏 long-CoT + tool call 数据**（最高质量但 cost 高），(2) ToolBench / xLAM / Magpie-tool 等开源 dataset（量大但 noise 高，必须人工 / LLM judge 二次清洗），(3) 自己跑 SFT base model 输出 + verifier 过滤正确的（rejection sampling）。**宁可 5k 高质量，不要 100k 脏数据**——R1 Stage 1 也只用了 thousand 条。
- ❗ **GRPO group size 在 long trajectory 上必须减小**——9.5 single-turn 推荐 G=16-64，multi-turn agent 因为 generate 太贵（每 trajectory 几十 tool call + 长 generate），G=4-8 才是工业甜点。G 太小（< 4）组内 variance 不够、advantage 失效；G 太大（> 16）generate cost 直接乘倍，单 step 几小时跑不完。**配 DAPO dynamic sampling 进一步剔除 std=0 group**，避免浪费。
- ❗ **Train env 与 eval env 一致性极重要**——常见雷：train 用 Wikipedia BM25 检索（cache 好的），eval 用 live Google search → 检索结果分布完全不同，model 在 train env 学到的"search query 风格" 在 eval 完全不 work，performance gap 50%+。**两种解法**：(1) train / eval 同 env（都 Wikipedia 或都 Google），(2) train 阶段引入 multi-env mixing 强迫 model 学习 retriever-agnostic 风格。Search-R1 论文 train + eval 都用 Wikipedia + BM25，是 controlled setup；现实生产 deploy 时通常要在自己的 retriever 上重新训一遍。
- ❗ **Tool exec 不 sandbox + timeout 训练直接 hang**（与 15.2 同源）——code agent / python interpreter 跑死循环 / 无限内存分配 → docker / firejail / nsjail 隔离 + 30s timeout + memory limit。任何 timeout 优雅降级为 `obs = "[error: timeout]"` 让 model 在 trajectory 内 recover、由 RL 学"遇到 error 怎么 recover"。SWE-RL 的 docker sandbox 是工业标杆。
- ❗ **多种 reward 项加权调参很烦，建议训练分阶段调权重**——training 早期：format 权重大（0.5-1.0），先让 model 学会 schema；training 中期：accuracy 主导（1.0），format 降到 0.2；training 后期：加 length / tool penalty 收紧。也可走 **curriculum reward**——前 N step 关掉 length penalty 让 model 自由探索，后 N step 加上让它收敛。一上来 4 项 reward 同时打满经常 model 跑偏。
- ❗ **Eval benchmark 选什么直接决定你 train 什么**——只 eval HotpotQA → 你的 model 只擅长 multi-hop QA、SWE 上跌；只 eval BFCL → 你 train 的是 single-turn function call，multi-turn 不行。**生产线必须 cross-benchmark eval**：BFCL（function call）+ TAU-bench（multi-turn dialog with tool）+ GAIA（通用 agent）+ SWE-bench（code agent）+ WebArena（web）。看 cross-benchmark 一致性，单 benchmark 高分容易 overfit。
- ❗ **Long-CoT eval 时务必报告 thinking tokens 与 latency**——一个 reasoning agent 在 SWE-bench 上 50% pass rate 但平均消耗 30k thinking token / 任务 vs 另一个 45% pass rate 但 5k token / 任务——后者工业落地价值高得多。**只报 accuracy 不报 cost 是 reasoning model 论文最常见的误导**。Qwen3 / GLM-Z1 给用户提供 thinking on/off toggle 就是这个权衡的产品化体现。

---

## 5. 经典 paper

- **DeepSeek-AI, 2025 — *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*** — 范式起点。本节所有工作都是 R1 思想的 multi-turn agent 推广。读 §2 的训练 pipeline、§3 的 R1-Zero training dynamics（Aha moment）、§4 的 distillation——这是理解"为什么 R1 能搬到 agent"的根。
- **Hou et al., 2025 — *Search-R1: Training LLMs to Reason and Leverage Search Engines via Reinforcement Learning*** — **本节必读**。第一篇明确把 R1 范式扩展到 multi-turn search agent 的论文。读 §3 的 trajectory 格式、loss mask 实现、§4 reward 设计、§5 实验——是本节代码 §3 与 §4 工程踩坑的直接 reference。GitHub `PeterGriffinJin/Search-R1`。
- **Chen et al., 2025 — *ReSearch: Learning to Reason with Search for LLMs via Reinforcement Learning*** — 中科院出品，multi-hop QA SOTA。读 §3 的 reflection-after-search template 设计 + §4 与 Search-R1 的对比实验。是 reasoning + search 协同的代表作。
- **Wang et al., 2025 — *Agent-R1*** — 清华出品的开源通用 agent RL 框架（基于 verl）。不是论文，是 reference repo——`0russwest0/Agent-R1`，配置驱动可跨 task 跨 domain，是当前最适合自己复现 R1-style agent 的起点。
- 选读：**Feng et al., 2025 — *ReTool: Reinforcement Learning for Strategic Tool Use in LLMs*** — 蚂蚁出品，把 RLVR 从 search 推广到 calculator / python interpreter。读 §3 的 tool-augmented reasoning template，是"reasoning + tool composition"代表。
- 选读：**Pan et al., 2025 — *SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution*** — Meta，SWE-bench 上的 agent multi-turn RL 代表。读 §3 sandbox 设计 + §4 reward shaping，是 code agent RL 工业落地参考。
- 选读：**Yu et al., 2024 — *DAPO: An Open-Source LLM Reinforcement Learning System at Scale*** — verl 框架算法核心。本节训练实操中的 token-level loss、dynamic sampling、clip-higher 全是 DAPO 建议，是 production reasoning agent RL 的工程支撑。

---

## 6. 自测与面试题

**Q1（pipeline）：** 写出现代 Reasoning Agent 的完整 6-stage 训练 pipeline，每 stage 说明输入数据、训练算法、目的。同时说明哪些 stage 在科研复现里可省略、哪些在生产线必做。

<details>
<summary>Answer sketch</summary>

**完整 6-stage pipeline**：

| Stage | 输入 | 算法 | 目的 |
|---|---|---|---|
| 1. Base Model | Qwen3-7B / Llama3-8B 等预训练 base | — | 起点。**不**用 instruct 版方便完全控制后续阶段 |
| 2. 通用 SFT | Magpie / Tülu-3 / OpenHermes (~100k-1M) | SFT (CE loss) | 让 base 会 chat / 跟随指令，避免后面 agent SFT 把通用能力洗光 |
| 3. Agent SFT cold start | ToolBench / xLAM / R1 蒸馏 agent CoT (~10k-100k) | SFT | 教 ChatML、`<think>/<tool_call>/<observation>` schema、基本 reasoning + tool 模板 |
| 4. **Multi-turn GRPO + verifier** | 1-10k task + verifier (search/code/web) | **multi-turn GRPO**（15.2 §3.3）+ DAPO 优化 | **核心 RL 阶段**——在 verifiable agent task 上学 long-CoT + tool composition |
| 5. 通用 Chat refine | 通用 SFT 数据 + 轻量 RLHF (RM 给 reward) | SFT + RLHF | 把可能被 agent RL 洗掉的 chat / refusal / safety 能力补回来 |
| 6. Eval | BFCL / TAU-bench / GAIA / SWE-bench / WebArena | — | cross-benchmark 一致性看，不只看单个 |

**科研可省略**：
- Stage 2 通用 SFT（直接用 instruct 版替代）
- Stage 5 chat refine（科研只关心 task 性能，不管退化）

**生产线必做**：
- 全部 6 stage 都做。Stage 5 不做的话 chat 体验会显著退化（user 问"你好"也开始疯狂 long-CoT）

**加分要点**：
- Stage 3 cold start 数据 quality 影响极大，**宁可少而精**（呼应 R1 Stage 1）
- Stage 4 用 multi-turn GRPO（15.2 §3.3）+ verifier reward（10.3 思想），是 R1 + 多轮工程的合体
- Stage 5 是 R1 Stage 4 General RLHF 在 agent 场景的等价物——RLVR 不能解决 chat/safety，必须挂 RM
- Stage 6 单 benchmark 高分容易 overfit，cross-benchmark 一致性才是 production 信号

</details>

**Q2（reward）：** Reasoning Agent 的 reward 通常是 `accuracy + format - length penalty - tool call cost` 4 项加权和。为什么需要每一项？如果只用 accuracy 一项会怎样？权重怎么调？

<details>
<summary>Answer sketch</summary>

**4 项分别防什么**：

| 项 | 形式 | 防什么 hacking | 不加会怎样 |
|---|---|---|---|
| **Accuracy** | $\mathbb{1}[\text{verifier pass}]$ | — | 没有主信号、训不动（必给） |
| **Format** | $\mathbb{1}[\text{合法 schema}]$ | parse 失败累积 | tool_call 输出 garbage、搜索结果都拿不到、accuracy 不涨 |
| **Length penalty** | $-\lambda_{\text{len}} \cdot \text{len}(\tau)$ | long-CoT 失控（length hacking） | trajectory 从 5k 涨到 50k+ token，cost 爆炸 |
| **Tool call cost** | $-\lambda_{\text{tool}} \cdot \max(0, N - \text{thres})$ | tool spam | "不确定就疯狂 search 兜底"，10+ tool call / task |

**只用 accuracy 一项会怎样**：
1. **Schema 失败累积** — 早期 model 不会按 `<tool_call>` 格式输出，verifier 直接判错，reward 全 0、advantage 全 0、训不动
2. **Length / Tool spam hacking** — model 学到"trajectory 长一点 / search 多调几次就 lucky 蒙对的概率高" → 平均 reward 涨但 cost 爆炸
3. **Reflection hacking** — 写很多"let me think again"但内容空洞，长却没信息密度

**权重原则**：
- **Accuracy 永远主导**（系数 = 1.0）
- **Format 给小正向**（0.1-0.3）——早期可大（0.5-1.0）让 model 先学 schema，后期降到 0.1
- **Length / Tool 给小负向**（0.0005-0.005，比 accuracy 小至少 1 个数量级）
- **任何 penalty 项系数 ≥ accuracy 主项 → policy 优先优化 penalty 而不是真解题**

**调参 trick**：
- **Curriculum reward**：前 N step 只开 accuracy + format，后 N step 加 length / tool 收紧
- **训练日志监控** `avg_traj_len` / `avg_n_tool` / `avg_thought_ratio`——突涨即 hacking 早期信号，及时调权重

加分要点：
- Search-R1 / SWE-RL 等论文都报告了 length / tool spam 现象，必须显式 penalty
- 与 9.6 reward hacking 章节呼应——agent RL 的 hacking 是 single-turn RLHF length bias 在多轮的放大版
- verifier-only reward 比 RM-based 抗 hacking 更强，但 length / tool hacking 仍存在（verifier 不管"用多少 token / 多少 tool 拿到结果"）

</details>

**Q3（前沿）：** 从 Search-R1 / ReSearch / ReTool / Agent-R1 中挑 2 个对比，说出共性与差异。它们与 SWE-RL（code agent）有什么关系？

<details>
<summary>Answer sketch</summary>

**共性（4 个工作 + SWE-RL）**：
1. **算法骨架完全一样**：multi-turn GRPO + trajectory-level outcome reward + 组内归一化 advantage + segment loss mask
2. **Reward 都是 verifier-based**：不用 RM，用 rule-based verifier（answer match / unit test / page state）
3. **Base 都从 SFT cold start 开始**（很少 R1-Zero 那样直接 base RL）——因为 agent format/schema 太复杂，纯 RL 学不来
4. **生产 stack 都是 vLLM + verl/OpenRLHF + LoRA + ZeRO-3**

**举例：Search-R1 vs ReTool 对比**：

| 维度 | Search-R1 | ReTool |
|---|---|---|
| **环境** | search engine（Wikipedia BM25） | python interpreter（sandbox） |
| **任务** | open-domain QA（NQ / HotpotQA） | math reasoning（MATH / AIME） |
| **Base** | Qwen2.5-7B-Base | Qwen2.5-Math-7B（已 math 加强） |
| **Tool 数** | 1（search） | 1（python） |
| **Verifier** | exact_match / cover_em vs gold | math answer equality（sympy） |
| **创新** | 第一个 R1 → multi-turn search 的明确扩展 | 让 reasoning model 学会**用程序检验自己** |
| **典型 trajectory 长** | 1-3k token | 3-10k token（含 python output） |

**核心差异**：
- Search-R1 解决的是"**信息检索**"问题——model 不知道答案，需要外部 search 拿信息
- ReTool 解决的是"**计算精确性**"问题——model 大概知道思路但 long arithmetic 容易错，用 python 复核

**举例：Agent-R1 与其他 3 个的关系**：
- Agent-R1 不是单一 paper，是 framework——把 Search-R1 / ReTool 等的 environment + reward 抽象成 YAML 配置
- 一份 model 同时 train search + math + code 多任务，强调 cross-domain generalization
- 是上面 3 个的"工程通用化"版本

**与 SWE-RL 的关系**：
- SWE-RL = 把 ReTool 的"sandbox tool 反馈" 推到极致——不是单 python 调用，而是完整 git repo + 多文件 edit + pytest 全套
- Trajectory：`read_file → edit_file → run_tests → 看哪个 fail → re-edit → re-run`，可能 20+ turn
- Verifier：unit test pass rate（partial credit）
- 是这条范式在**最复杂 agent task** 上的代表——证明 R1 + GRPO 能 scale 到 SWE-bench 这种 production-grade 难度

**共同 Pattern 总结**：
**算法不变**（multi-turn GRPO + verifier reward + trajectory-level advantage broadcast），**变的是 environment（什么 tool）+ reward verifier（怎么判 task 成功）+ task 数据**——不同工作就是这 3 维上的不同 instantiation。

加分要点：
- 提到 **5 个工作的共同 limitation**：cross-tool composition（A、B 单独 train 完，A+B 组合用是否 work？）尚未充分验证
- 提到 ReSearch 与 Search-R1 同时期（2025.04），都是 multi-hop 强化方向，差异在 reward F1 vs EM、reflection template 设计
- 提到这条范式的下一步：**multi-agent RL**（一个 main + 多 sub-agent）+ **long-horizon RL**（trajectory > 100 turn 的 OS-level 任务）

</details>

---

## 7. 2025-2026 现状与开放问题（补充）

### 7.1 现状

- **Open-R1 / TinyZero** 已成功复现 R1-Zero 风格 emergence，门槛降到单 8B 模型 + ~$30 算力，证明 reasoning agent 不是大厂专利
- **Reasoning agent 是 2025-2026 工业落地热点**：Perplexity（search agent）、OpenAI Deep Research（multi-hop search agent）、Cursor / Claude Code（code agent）、Anthropic Computer Use 全部走"reasoning + tool" 路线，背后训练范式都是本节这一支
- **MCP（Model Context Protocol）+ reasoning agent** 是部署趋势：MCP 提供标准 tool 接口，reasoning agent 提供 RL 训出的"会用 tool 的脑子"，两者解耦后 agent 可以"plug and play"任意 tool——也意味着**未来 reasoning agent 的训练 task 与部署 task 可以是不同 tool**，对 cross-tool generalization 提出更高要求
- **国内业务基模团队**（小红书 / 阿里 / 字节 / 腾讯等）几乎全部跟进——production-grade 业务 agent（搜索 / 客服 / 审核）的训练范式从"SFT-only"过渡到"SFT + multi-turn GRPO + verifier"。在中文场景下，verifier 设计本身又多一层难度（answer match 必须考虑同义词 / 简繁 / 句式变化），往往要用 LLM-as-judge 兜底
- **Benchmark 端**：BFCL（function call 准确性）、TAU-bench（multi-turn dialog with tool）、SWE-bench-Verified、GAIA、WebArena 几个 benchmark 共同定义了"reasoning agent 是否好"的标尺；2026 年新出现的 OSWorld、AgentBench 进一步覆盖 OS-level / 长程任务

### 7.2 开放问题（必须知道）

1. **Reward verifier 在 unverifiable task 上不适用**（与 R1 同问题）：chat helpfulness / creative agent / 情感支持仍需 RM。**当前生产线方案**：reasoning task 用 verifier，chat task 用 RM，混合训练（呼应 R1 Stage 4）
2. **Long trajectory 显存爆炸**：64k+ token trajectory × G=8 + ZeRO-3 + grad checkpoint 在 8×80G H100 上仍勉强。**研究方向**：sequence parallelism（7.3）+ 更高效 packing + selective activation recomputation
3. **Cross-domain generalization 弱**：Search-R1 在 search 上训完搬到 SWE 任务表现差。**根因**：每个任务需要不同的 tool 风格 / reasoning pattern，单 task RL 容易 overfit。**研究方向**：Agent-R1 的多 task 混训、meta-RL across tools
4. **Multi-tool composition 是黑盒**：训过 search tool、训过 calculator tool，能 zero-shot composition 用 search + calculator 解题吗？**实验结果**：部分 work 报告 emergent composition，部分报告完全不 transfer。**当前共识**：composition 能力随 base model size 增强（70B+ 比 7B 强很多）
5. **Thinking efficiency**：long-CoT 越来越长（5k → 50k token），cost 越来越高。**研究方向**：(1) thinking budget control（用户给 budget、model 在 budget 内 reasoning），(2) **adaptive thinking**（简单题短想、难题长想，Qwen3 thinking toggle 雏形），(3) thinking 蒸馏（teacher 50k token 蒸馏到 student 5k token）
6. **Verifier 设计本身的代价**：写一个 robust math verifier 已经很难（10.3 §4 详谈），写 SWE / web / GAIA 的 verifier 需要的工程量更大。**研究方向**：LLM-as-judge 半自动 verifier（但又引入 RM 漂移问题）、process reward model 复活
7. **Sim2Real gap**：train 在 mock environment（cached search / docker sandbox），eval 在 live environment（Google search / GitHub），分布差距巨大。**研究方向**：mixed env training、env randomization、online RL with safety guards
8. **Agent 鲁棒性**（15.5 详谈）：observation perturbation / tool failure / API error 下 agent 是否能 recover，是当前 reasoning agent 在生产部署时的最大短板

---

## 8. 延伸阅读

- [Search-R1 GitHub & paper](https://github.com/PeterGriffinJin/Search-R1) — multi-turn search agent 的 GRPO 复现，loss mask 与 reward 设计可直接参考
- [Agent-R1 GitHub（清华）](https://github.com/0russwest0/Agent-R1) — 开源通用 agent RL 框架，配置驱动，落地最容易
- [verl GitHub（ByteDance）](https://github.com/volcengine/verl) — 工业级 multi-turn agent RL 框架，DAPO + GRPO 默认配置
- [OpenRLHF GitHub](https://github.com/OpenRLHF/OpenRLHF) — 早期支持 multi-turn 与 vLLM 解耦
- [ReTool paper (arXiv)](https://arxiv.org/abs/2504.11536) — 蚂蚁 tool-augmented reasoning RL
- [SWE-RL paper (arXiv)](https://arxiv.org/abs/2502.18449) — Meta，code agent multi-turn RL
- [HuggingFace Open-R1](https://github.com/huggingface/open-r1) — R1 完整开源复现项目，配 cold-start 数据集（OpenR1-Math-220K）
- [ReSearch paper](https://arxiv.org/abs/2503.19470) — 中科院 multi-hop search RL
- [Nathan Lambert RLHF Book](https://rlhfbook.com/) — 含 multi-turn / agent RL 章节
- [TAU-bench (Anthropic)](https://github.com/sierra-research/tau-bench) — 评估 multi-turn agent 工具使用能力，2025 年新晋 benchmark
- 推荐继续读本教程的 **15.5《Agent 鲁棒性：observation perturbation / tool failure / recovery》**——本节的"Sim2Real gap"和"鲁棒性"问题在 15.5 系统讲；**11.5《投机解码》**——reasoning agent 长输出的 inference 加速方案；**16.5《Computer Use & GUI Agent》**——把本节范式推到 GUI / OS-level 操作的另一前沿
