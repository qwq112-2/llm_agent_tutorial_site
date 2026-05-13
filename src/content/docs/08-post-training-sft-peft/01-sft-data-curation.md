---
title: "8.1 SFT 数据构造：Self-Instruct / Magpie / OpenHermes / Evol-Instruct"
description: "预训练用的是 web 上抓来的万亿 token 自学语言，SFT 完全是另一个游戏——量级降到几千到几百万条 `` 对，quality 比 quantity 重要一个数量级。本节梳理 SFT 数据的 5 大构造路线（人工 / Self-Instruct / Magpie / Evol-Instruct / 蒸馏）、主流开源数据集速览、curation pipeline 与多任务配比经验，帮你理解为"
---

> ⏱ 预计阅读 35 分钟 ｜ 难度 ★★ 🔥 ｜ 前置：6.2 预训数据 pipeline

## 一句话本节讲什么

预训练用的是 web 上抓来的万亿 token 自学语言，**SFT 完全是另一个游戏**——量级降到几千到几百万条 `(instruction, response)` 对，**quality 比 quantity 重要一个数量级**。本节梳理 SFT 数据的 5 大构造路线（人工 / Self-Instruct / Magpie / Evol-Instruct / 蒸馏）、主流开源数据集速览、curation pipeline 与多任务配比经验，帮你理解为什么 LIMA 1000 条数据能训出近 GPT-3.5 水平、Magpie 又如何把"现代规模化 SFT 数据"成本压到几乎为零。

---

## 1. Mental model（直觉）

把预训和 SFT 想成两段截然不同的"教育阶段"。

**预训**像让一个孩子从婴儿开始读 25 亿本书——他不知道在干嘛，只是在每个 token 上猜下一个，靠 trillion 级别的语料自学了语法、世界知识、推理 pattern。读完后他**懂很多东西，但不会"按指令回答问题"**——你问他"用 3 句话总结一下相对论"，他可能会接着写"……是爱因斯坦在 1915 年提出的，本论文将进一步……"——他在做"续写"，不是"回答"。

**SFT** 是教他"接到指令应该怎么回答"——准备一批高质量的 `(instruction, response)` 对，用监督学习的方式让他的输出对齐到 response 的格式与风格。这一步**不是教他新知识**（trillion-scale 的预训之后，再加几十万条 SFT 不可能教会他什么本质新东西），而是**教他"使用"已学的知识**——按指令格式输出、保持 multi-turn 一致性、在应该拒绝时拒绝、按要求生成 code/table/markdown。

```
                  预训 (pretrain)              SFT
            ┌────────────────────────┐  ┌──────────────────────┐
数据规模    │  10^12 - 10^13 token   │  │  10^3 - 10^7 sample  │
任务形式    │  next-token prediction │  │  (instr, resp) 对    │
监督信号    │  自监督（无标签）      │  │  监督（response 是答案）│
目标        │  学语言 + 学世界知识   │  │  学"按指令输出"      │
quality 重要│  ★★ (重要)             │  │  ★★★★★ (极致重要)   │
quantity 重要│ ★★★★★ (越多越好)       │  │  ★★ (够用即可)       │
            └────────────────────────┘  └──────────────────────┘
```

为什么 SFT 阶段 quality 比 quantity 重要这么多？两个原因：

1. **预训阶段已经"喂饱了知识"**——SFT 不需要再用海量数据让模型学语法。每条 SFT 样本要做的事很窄："教模型在这种 instruction 下怎么响应"。一条高质量的 multi-turn 数据胜过 100 条堆砌的低质量单轮。
2. **SFT 是 imitation learning**——模型会**精确模仿**你给它的 response 格式、风格、错误。喂垃圾就模仿垃圾，喂啰嗦就学会啰嗦，喂"我不知道"就学到处装傻。这是 LIMA 论文（Zhou 2023）那条著名结论的根源——**1000 条精挑数据 + LLaMA-65B 训完 ≈ GPT-3.5 水平**，"less is more for alignment"。

那 SFT 数据从哪来？**5 大主流路线**：人工标注（OpenAssistant、LIMA 这种金标）、Self-Instruct（GPT-4 自动扩写，Alpaca 范式）、Magpie（让 base model 用空 prompt template "自吐" instruction）、Evol-Instruct（WizardLM 的"让指令变难变广"）、蒸馏（用 GPT-4 / Claude 生 response）。每条路线都有代表数据集，下文会一一拆开。

> 与 6.2 区分：6.2 讲的是 PB 级 web 数据怎么清成 trillion token；本节讲的是几千到几百万条 instruction-response 对怎么构造、过滤、配比。**MinHash 去重、LLM-as-judge 这些工具两边都用，但量级差 5-7 个数量级、目标完全不同**——预训追"广度覆盖"，SFT 追"高质量响应模仿"。
> 与 8.5 区分：本节只讲"数据从哪来、怎么过滤"，**SFT 训练细节（chat template / loss mask / sample packing）是 8.2，多轮 + tool 混合训练实战是 8.5**。

---

## 2. 公式与原理

### 2.1 SFT 的形式化

SFT 用的目标和预训完全一样——对每条 `(instruction, response)` 对，把它们拼成单条序列 $x = [\text{instr}, \text{resp}]$，最大化 response 部分的 token likelihood：

$$
\mathcal{L}_{\text{SFT}}(\theta) = - \sum_{t \in \text{resp}} \log p_\theta(x_t \mid x_{<t})
$$

注意求和**只在 response 的 token 上**——instruction 部分不算 loss（这就是 8.2 要细讲的 loss mask）。其余完全是普通的 next-token prediction。

唯一"数据侧"的事：每条样本 $(x_i, y_i)$ 怎么构造、怎么挑、怎么混。下面 5 大来源就是回答这个问题。

### 2.2 SFT 数据 5 大构造方法

#### (1) 人工标注

最贵也最干净。代表数据集 **OpenAssistant (OASST，LAION 2023)**——35 个国家志愿者写了 161k 条消息、组织成 33k 多轮对话树，**完全人工 prompt + 人工回答 + 人工偏好排序**。是公认的开源 SFT 金标，特别是多轮对话部分。

人工的优点：质量天花板高、不会有 LLM 自吐的"模式化"（机器写的 response 经常有共同的句式套路）。缺点：贵、慢、规模有限，万级以上几乎不可能纯人工。

**LIMA (Zhou 2023, Meta)** 是人工路线的极致代表——只精挑 1000 条 sample（来自 Stack Exchange / WikiHow / 人工写），训 LLaMA-65B，结果在人类偏好评测上**接近 GPT-3.5**。论文标题就叫 *Less Is More for Alignment*，是 SFT 阶段"quality > quantity"最有力的实证。

#### (2) Self-Instruct (Wang 2022)

第一篇把"用 LLM 自动造 SFT 数据"系统化的工作。整体流程：

1. **手写 175 个 seed instruction**（覆盖各种任务类型）
2. **用 GPT-3 / GPT-4 生成新 instruction**——few-shot 给它 8 条 seed，让它接着写新的
3. **过滤**——去重（ROUGE-L 相似度 > 0.7 丢）、过滤含禁用词的、过滤太短/太长的
4. **用 LLM 生成 response**——对每条新 instruction，再让 LLM 写答案

**Alpaca (Stanford 2023)** 就是 Self-Instruct 的直接产物——用 175 个 seed + GPT-3.5（text-davinci-003）扩出 52k 条 (instruction, response)，训 LLaMA-7B，成本只有 $600 但效果接近原版 GPT-3.5。

缺点：**质量参差**——GPT-3.5 生成的 instruction 有相当比例重复、模糊或不实际；response 也带浓重 GPT-3.5 风格的 bias（凡事都列 5 条、都说"作为 AI 模型"）。早期 Vicuna / Dolly / Koala 等都是 Self-Instruct 范式。

#### (3) Magpie (Xu et al. 2024)

2024 年的新范式，**从根本上改变了"开源 SFT 数据如何便宜地规模化"这件事**。

**核心 idea**：不需要 seed instruction，直接让 base model（不是 chat model！）"自吐" instruction。怎么做？利用 chat-tuned 模型已经训过的 prompt template——给 LLaMA-3-Instruct 输入这串前缀：

```
<|begin_of_text|><|start_header_id|>user<|end_header_id|>

```

然后 sample。模型已经学过 user 这一段后面"应该"是什么——它会自动 sample 出一条 user instruction（因为它的训练数据里 user header 后面就是 user 写的话）。再用同一个模型续接 assistant header，让它自己回答自己的问题。

**整个流程 100% 模型生成、零 seed、零人工 prompt engineering**。Magpie-Pro（用 Llama-3-70B-Instruct 生成的版本）一次性产出 100 万-300 万条数据，质量 ablation 显示**用 Magpie 1M 数据 SFT 出来的 LLaMA-3-8B 在 AlpacaEval 上能打过用 1M Self-Instruct 数据训的同 size 模型**。

为什么 Magpie 质量更好？两个机制：

- 它从 chat model 的 **真实 prompt 分布**采样（chat model 的 instruction 分布更贴近真实用户行为，而 Self-Instruct 是 GPT-3.5 在 8-shot 引导下"想象"出的 instruction，分布偏窄）
- diversity 高——纯 sample 不受 seed 锚定，话题分布更自然散开

#### (4) Evol-Instruct (WizardLM, Xu 2023)

WizardLM 团队的核心方法。idea：拿现有 instruction，**用 LLM 让它"进化"——变更难、变更广**。两条 evolution 路线：

**In-depth evolving**：让 instruction 变更难。具体 5 种 operator：

- **Add constraints**：加约束（"用 100 字以内回答"、"只用 Python 标准库"）
- **Deepen**：让问题更深入（"解释原理" → "解释原理并对比 3 种替代方案"）
- **Concretize**：把抽象问题具体化（"写个排序算法" → "写一个对包含 100 万条记录的 CSV 文件做归并排序的 Python 实现"）
- **Increase reasoning steps**：增加推理步骤
- **Complicate input**：让输入变复杂（输入数据更长、更多边界 case）

**In-breadth evolving**：让 instruction 变更广。给一条 instruction，让 LLM 生成"不同 domain 但相似难度"的新 instruction（数学题 → 物理题 → 化学题 → ……）。

**WizardLM、WizardCoder、WizardMath** 系列都基于 Evol-Instruct——70k 条 evolved 数据训出的 LLaMA 在 MT-Bench 上**比同 size Self-Instruct 数据训的高 5-10 个百分点**，是 2023 年开源 SFT 数据的事实最强方案。

代价：每条数据需要多次 LLM call（先 evolve 几轮、再生成 response），**API 成本远高于 Self-Instruct**。

#### (5) 蒸馏 (Distillation)

用强 model（GPT-4 / Claude-3.5 / DeepSeek-V3）生成 response 蒸馏给弱 model。这是"现代开源 chat model"最广泛的数据来源之一。代表数据集：

- **Vicuna (LMSYS 2023)**——70k 条从 ShareGPT 收集的真实用户与 ChatGPT 的对话（用户主动分享，不是 OpenAI 漏出的），多轮对话能力特别好
- **OpenHermes-2.5 (NousResearch)**——综合 1M 条多源 + GPT-4 蒸馏，是 2024 年开源界最常用的 SFT 综合数据集之一
- **Tülu (AI2)**——综合多源（FLAN、WizardLM、ShareGPT、code、math …）+ 蒸馏，配套 Llama 系列开源 chat model
- **OpenOrca / SlimOrca**——基于微软 Orca 范式，用 GPT-4 给每个回答加 reasoning trace 蒸馏

蒸馏的硬伤：**法律灰色地带**——OpenAI ToS 禁止用其 API output 训"竞争 model"。学术研究广泛使用，但商业用必须慎重——可换成用 Qwen / DeepSeek / Mistral 等开源 model 蒸馏（明确允许商用的 license）。

### 2.3 现代主流开源 SFT 数据集速览

| 数据集 | 来源 | 规模 | 特点 |
|---|---|---|---|
| **OpenAssistant (OASST1/2)** | 人工 | 33k 对话 / 161k 消息 | 多轮对话金标，license 完全 open |
| **LIMA** | 精选人工 | 1k | "less is more"实证标杆 |
| **Alpaca** | Self-Instruct | 52k | 早期范本，质量已被现代数据集超越 |
| **WizardLM** | Evol-Instruct | 70k-250k | 复杂指令能力强 |
| **WizardMath / WizardCoder** | Evol-Instruct + domain | 数百 k | math / code 专用 |
| **Vicuna (ShareGPT)** | 蒸馏 | 70k 多轮 | 多轮对话好 |
| **OpenHermes-2.5** | 多源 + 蒸馏 | 1M | 综合 SOTA，最常用之一 |
| **Tülu-3 SFT mix** | AI2 综合 | 940k | 综合 + 含 reasoning，工程范本 |
| **Magpie / Magpie-Pro** | base 自吐 | 1M-3M | 现代规模化 cheap pipeline |
| **Infinity-Instruct** | 多源 + curation | 7M | 大规模综合 |
| **NuminaMath-CoT** | math 专用 | 860k | math reasoning 标杆 |
| **Code-Feedback** | code 专用 | ~70k | code SFT 含错误反馈 |
| **OpenOrca / SlimOrca** | GPT-4 蒸馏 + reasoning | 4M / 500k | 含 reasoning trace |
| **DeepSeek-R1-Distill data** | R1 蒸馏 | 800k | long-CoT 蒸馏，2025 新潮 |

中文 SFT 数据集相对落后一代，常见的有：**BELLE**（链家，几百万条 Self-Instruct 中文）、**MOSS**、**FireFly**、**Wizard-LM-Chinese**、**COIG / COIG-CQIA**（智源）、**Chinese-Magpie**。商用项目通常自己再人工标注或蒸馏一批补强。

### 2.4 SFT 数据 curation pipeline

拿到一批原始数据后的标准 6 步：

```
原始数据 (1M+)
    │
    ├─[1] 去重 (MinHash + LSH，threshold ~0.8)
    │        └→ 砍掉 10-30%
    │
    ├─[2] 质量过滤 (LLM-as-judge 打分 + 阈值)
    │        └→ 砍掉 30-60%
    │
    ├─[3] Diversity sampling (embedding 聚类后按 cluster 平衡采样)
    │        └→ 进一步精挑
    │
    ├─[4] Length / format filter (过短 < 20 / 过长 > 8k 丢)
    │
    ├─[5] Safety filter (含暴力 / 隐私 / 仇恨内容丢)
    │
    └─[6] Domain rebalance (按目标 use case 调比例)
        ↓
    最终 SFT 数据 (10k - 1M)
```

**(1) 去重**——和 6.2 预训去重同源技术，MinHash + LSH（参见 6.2）。SFT 阶段尤其重要，因为 Self-Instruct 类数据有大量"措辞不同但语义相同"的样本，不去重会让模型学到该 pattern 的过度模仿。

**(2) 质量过滤**——主流是 **LLM-as-judge**：用 GPT-4 / Llama-3-70B 给每条数据打 1-5 分，留 ≥ 4 分的。打分维度通常 4 个：instruction 清晰度 / response 正确性 / response 完整性 / format 合规性。

**(3) Diversity sampling**——用 embedding 模型（如 bge-large）算每条 instruction 的向量，K-means 聚成几百个 cluster，按 cluster 平衡采样而不是 uniform random，能避免某些常见话题（"写一首诗"、"解释 X 概念"）占比过大。

**(4) Length / format filter**——过短（< 20 token instruction 多半是低信息量）、过长（> 8k token 训练时 OOM 风险）、格式异常（带乱码、控制字符）都丢。

**(5) Safety filter**——对开源发布尤其重要。可用 Llama-Guard、专门的 safety classifier 过一遍。

**(6) Domain rebalance**——根据目标 model 的 use case 调比例。一个常见配方（通用 chat model）：

| 类别 | 比例 |
|---|---|
| 通用对话 / instruction following | 50% |
| Code | 20% |
| Math / reasoning | 15% |
| 多语言 (含中文) | 10% |
| Safety / refusal | 5% |

**单一任务过 dominant 会让 model 通用能力下降**——纯 math SFT 会让 model 闲聊变 disfluent；纯 code SFT 让 model 解释自然语言概念时变啰嗦。这套 50/20/15/10/5 配比和 Tülu-3、OpenHermes-2.5 等公开配方大致一致。

### 2.5 SFT 数据 quality 实战经验

几条社区共识级的经验值：

- **质量 > 数量**：LIMA 1k > Alpaca 52k 是已被反复验证的事实
- **多轮 > 单轮**：现代 chat model 必须有相当比例 multi-turn 数据（≥ 30%），否则上下文跟随能力差
- **思维链 (CoT) 必须保留**：reasoning 任务的 response 要展示推理过程而不只是答案——直接给答案训出来的 model 推理能力急剧下降，这是 R1 之后被反复证实的（R1-Distill 之所以有效，本质就是用 long-CoT response 替换了短答案）
- **拒绝示例 (refusal)**：训 model 学会拒绝（不知道 / 不应该回答 / 涉及隐私），但比例要严格控制 < 5%——多了会学成"凡事都拒"
- **format diversity**：让 model 在训练中见到多样格式（plain text / markdown / code block / table / latex / JSON），泛化好
- **不要 contamination**：和 6.2 一样，training data 不能含测试 benchmark 的题，否则评测分虚高

### 2.6 新趋势：reasoning 数据 / R1 distillation

2025 年的新挑战：**long-CoT 数据稀缺且珍贵**。R1 / o1 / Qwen3-thinking 等 reasoning model 出现后，"在 response 里展示完整推理过程"成了 SFT 数据的新标配，但绝大部分历史 SFT 数据集都只有"短答案"——重新生成 long-CoT 版本成本高。

主流做法：

- 用 R1 / DeepSeek-R1 / o1 给现有 SFT 题目重新生成 long-CoT 答案，蒸馏给小 model（**DeepSeek-R1-Distill-Qwen / Llama 系列就是这么训的**，800k 条 R1 生成 trace 蒸到 7B-70B）
- **Open-R1 项目（HuggingFace）**：用开源方法复现 R1 数据流程
- **NuminaMath-CoT**：math 专用 long-CoT 数据
- **Skywork-OR1 / OpenThoughts**：纯 reasoning 蒸馏数据

这条路在 2025 年是开源界最热的方向之一，会在 10.3 节（RLVR / DeepSeek-R1）展开讲。

---

## 3. 最小代码示例

### 3.1 Magpie 风格生成 instruction

```python
# Magpie：给 chat model 输入空的 user header，让它自己 sample 出 instruction
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "meta-llama/Llama-3-8B-Instruct"
tok = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")

# 关键：只给到 user header，让模型续 user 内容
prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
inputs = tok(prompt, return_tensors="pt").to(model.device)

# 高温度 sample 保证 diversity
out = model.generate(**inputs, max_new_tokens=256, do_sample=True,
                     temperature=1.0, top_p=0.95,
                     eos_token_id=tok.convert_tokens_to_ids("<|eot_id|>"))
instruction = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print("Magpie instruction:", instruction)

# 拿到 instruction 后，再拼上 assistant header 让同一模型自己回答
full = prompt + instruction + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
inputs = tok(full, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.7)
response = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
```

关键点：**第一步只给 user header，不给任何 seed**——模型在自己的训练分布下 sample 出"用户最常问的"那类 instruction，diversity 远好于 8-shot Self-Instruct。生产环境 batch_size 拉到 64+ 可以一晚上生成几十万条。

### 3.2 Evol-Instruct in-depth deepening 模板

```python
# WizardLM 原 paper 的 In-depth deepening prompt（简化版）
EVOL_DEEPEN = """I want you act as a Prompt Rewriter.
Your objective is to rewrite a given prompt into a more complex version
to make it harder for AI models to handle.

You SHOULD:
- Add 1-2 more constraints/requirements to the original prompt
- The rewritten prompt MUST be reasonable and answerable by humans
- The rewritten prompt should NOT exceed the original by more than 20 words

#The Given Prompt#:
{instruction}

#Rewritten Prompt#:
"""

from openai import OpenAI
client = OpenAI()

def evolve_deepen(instruction: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": EVOL_DEEPEN.format(instruction=instruction)}],
        temperature=0.7,
    )
    return resp.choices[0].message.content.strip()

orig = "Write a Python function to sort a list."
evolved = evolve_deepen(orig)
print("Evolved:", evolved)
# 期望输出类似："Write a Python function to sort a list of dictionaries by 
# multiple keys, with descending option per key, using only standard library."
```

实战 tip：完整 Evol-Instruct 是 5 个 operator（add constraints / deepen / concretize / increase reasoning / complicate input）+ in-breadth 共 6 个 prompt 模板，每条数据可 evolve 1-3 轮。每轮后用一个 "elimination" prompt 检查 evolved instruction 是否合理（不合理就回退到上一轮）。

### 3.3 MinHash dedupe SFT 数据

```python
# pip install datasketch
from datasketch import MinHash, MinHashLSH

def sft_minhash(item, k=128, ngram=5):
    """对 (instruction, response) 对算 MinHash，instruction 部分加权"""
    text = item["instruction"] + " " + item["response"]
    m = MinHash(num_perm=k)
    tokens = text.split()
    for i in range(len(tokens) - ngram + 1):
        m.update(" ".join(tokens[i:i+ngram]).encode())
    return m

# 假设 dataset 是 [{"instruction": ..., "response": ...}, ...]
dataset = [
    {"instruction": "Write a poem about autumn", "response": "Leaves fall gently..."},
    {"instruction": "Compose a poem about autumn", "response": "Leaves fall gently..."},  # 近重复
    {"instruction": "Explain quantum entanglement", "response": "Quantum entanglement is..."},
]

lsh = MinHashLSH(threshold=0.8, num_perm=128)
keep_ids, sigs = [], {}
for i, item in enumerate(dataset):
    sig = sft_minhash(item)
    if not lsh.query(sig):           # 没有近似重复 → 留
        lsh.insert(f"id_{i}", sig)
        sigs[f"id_{i}"] = sig
        keep_ids.append(i)
print(f"原始 {len(dataset)} 条，去重后保留 {len(keep_ids)} 条")
```

工程上 SFT dedupe 的 threshold 通常比预训稍宽（0.8-0.85 都常用）——SFT 阶段稍许重复对模型影响不大，但完全的 paraphrase（"写一首关于秋天的诗" vs "创作一首秋天的诗"）必须去掉。

### 3.4 LLM-as-judge 质量打分

```python
JUDGE_PROMPT = """You are an expert data quality judge.

Rate the following (instruction, response) pair on 4 dimensions, each 1-5:
- instruction_clarity: Is the instruction unambiguous?
- response_correctness: Is the response factually correct?
- response_completeness: Does the response fully address the instruction?
- format_quality: Is the format appropriate (markdown/code/list etc.)?

Output ONLY a JSON object: {{"clarity":X, "correctness":X, "completeness":X, "format":X}}

Instruction: {instruction}
Response: {response}
"""

import json
from openai import OpenAI
client = OpenAI()

def quality_score(item) -> float:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(**item)}],
        temperature=0.0, response_format={"type": "json_object"},
    )
    scores = json.loads(resp.choices[0].message.content)
    return sum(scores.values()) / len(scores)   # 平均分

item = {"instruction": "What is the boiling point of water?", "response": "100°C at 1 atm."}
print("score:", quality_score(item))   # 约 4.5+，留下
# 实操：score >= 4.0 留，3.0-4.0 边缘人工抽检，< 3.0 丢
```

实战 trade-off：GPT-4 全量打分对 1M 数据要 $1k+ 成本，主流做法是先用 GPT-4 给小样本（10k）打分，再 fine-tune 一个小 BERT 当"分数预测器"对全量打分（和 6.2 FineWeb-Edu 同样的两阶段思路）。

---

## 4. 工程踩坑与经验

- ❗ **Self-Instruct 数据未做 dedupe → 训出来 model 重复回答模式**——Self-Instruct 的常见失败 mode：GPT-3.5 在 8-shot 引导下会反复生成"Write a poem about X"、"Explain Y in simple terms"这种 template-like instruction，不去重直接训会让 model 学到这个 pattern，之后用户问什么问题它都倾向用模板化套路回答。MinHash threshold 0.8 是社区经验值。
- ❗ **数据 quality 比 quantity 重要**——LIMA 1k > Alpaca 52k 是事实，但**这条经验不能极端化**：低于 1k 数据 SFT 的 model 经常 format 不稳定（chat template 学不全）。社区推荐 sweet spot 是 **10k-100k 高 quality > 1M 普通 quality**。如果数据来源混杂，先用 LLM-as-judge 过滤一遍再训，几乎一定比直接全量训好。
- ❗ **多轮对话 response 必须保持 role 一致性**——多轮数据里 assistant 不要突然冒出"User:"接着写——这种是 raw ShareGPT 数据常见 bug（用户对话被截断时模型续写了下一轮 user message）。训前必须用规则过滤：每条 assistant response 不能含 `User:` / `Human:` / chat template 的 user 标记。Vicuna 早期版本就因为没过滤这个，model 会自言自语扮演 user。
- ❗ **用 GPT-4 蒸馏数据有 ToS 风险**——OpenAI 的 [Terms of Service](https://openai.com/policies/terms-of-use) 第 2(c) 明确禁止用其 output 训"竞争模型"。学术发表通常没事，但商业部署有法律风险。**替代方案**：用 Qwen-2.5-72B、DeepSeek-V3、Mistral-Large 这类开源 weight 的 strong model 蒸馏，license 明确允许商用。质量差距比想象的小。
- ❗ **Long-CoT 数据稀缺，盲目用 short answer 数据训会丢失推理能力**——这是 R1 后的新挑战。如果你的 base model 是带 thinking 能力的（Qwen3-thinking、R1-Distill 系列），SFT 数据里的 response 必须保留 long reasoning trace；用普通 ShortGPT 风格的 SFT 数据训，会**直接让模型忘记 think**——R1-Distill-Qwen-7B 团队公开报告过这个 trap，纯 short answer SFT 后 AIME 分数掉 20+ 点。
- ❗ **Code SFT 数据要含 stack trace / error message**——不只是 happy path 的 "写一个函数"。Code-Feedback、OpenCodeInterpreter 等数据集就是专门补这个空缺：让 response 包含"先写一版 → 报错 → 看 traceback → 改一版"的完整 debug 流程。这样训出来的 code model 才知道遇到 error 怎么处理，否则一遇到 traceback 就开始幻觉解决方案。
- ❗ **Math 数据要保留中间推理 + 答案可验证**——单纯给最终数字答案训，model 学不到推理路径。**NuminaMath / GSM8K 风格**：response 必须含 step-by-step + 最终答案 boxed 起来（例如 `\boxed{42}`），方便后续 RL 阶段做 rule-based reward（参考 9.5、10.3）。
- ❗ **Refusal 数据不能太多（< 5%）**——训 model 学会拒绝是必要的（涉及暴力 / 个人隐私 / 越狱），但比例超过 5% model 会**学到"凡事都拒"**的 over-refusal pattern：问"Python 怎么读文件" model 也警告"涉及文件系统访问安全风险"。Llama-2-Chat 早期发布版本因为 refusal 太多被社区调侃 "Llama-NoFun-2"，后续版本就大幅降比例。
- ❗ **Format diversity 重要**——纯 markdown 训出的 model 在 plain text 场景下效果差（强行加 `**` `##` 让输出变难看），纯 code 训出的 model 解释自然语言变啰嗦。SFT 数据集要混合 markdown / plain / code / table / JSON / latex 等多种 format，让模型学会**根据 instruction 推断合适的 format**。
- ❗ **Tokenizer 不一致会"看不见"重复**——用 LLaMA tokenizer 的 dedupe 结果直接用到 Qwen 训练，"重复"判断可能失效（因为 token 边界不同）。SFT dedupe 推荐基于 word-level n-gram，而不是 token-level，结果跨 model 通用。

---

## 5. 经典 paper

- **Wang et al., 2022 — Self-Instruct: Aligning Language Models with Self-Generated Instructions** — 用 LLM 自动造 SFT 数据的开山作。读 §3 数据生成 pipeline + §4 过滤策略。Take-away：理解整个"LLM 自生成 instruction → 过滤 → 训练"范式，Alpaca / Vicuna / 大量 2023 早期 chat model 都直接用这套。
- **Zhou et al., 2023 — LIMA: Less Is More for Alignment** — Meta 出品，1000 条精选 SFT 数据训 LLaMA-65B 接近 GPT-3.5 水平。读 §2 数据来源（Stack Exchange + WikiHow + 人工）+ §5 ablation（quality vs quantity）。Take-away：建立"SFT 阶段 quality > quantity"的 mental model，理解为什么 LLM 不需要海量 SFT 数据就能 align。
- **Xu et al., 2023 — WizardLM: Empowering Large Language Models to Follow Complex Instructions (Evol-Instruct)** — Evol-Instruct 方法论。读 §3 in-depth + in-breadth evolving 的 5 个 operator + prompt 模板。Take-away：理解"让指令变难变广"作为 SFT 数据增强的范式，是 2023 年开源界 SFT 数据 SOTA。
- **Xu et al., 2024 — Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing** — 现代规模化 SFT 数据生成 SOTA。读 §3 让 base/chat model 用空 user header 自吐 instruction 的 trick + §4 与 Self-Instruct 的对比 ablation。Take-away：理解"Magpie 范式如何把开源 SFT 数据成本压到几乎为零"，2024-2025 多个开源 chat model（包括 Llama-3 后续工作）数据流程参考。
- **Lambert et al., 2024 — Tülu 3: Pushing Frontiers in Open Language Model Post-Training** — AI2 出品，**完整 SFT + DPO + RLVR 工程范本**。读 §3 SFT 数据 mixing（940k 条 + domain rebalance）+ §6 整体 recipe。Take-away：现代开源 post-training 完整工程实例，从数据 mix、训练超参到评测一条龙公开，是工业落地最完整的开源参考。
- 加分阅读：**DeepSeek-AI, 2025 — DeepSeek-R1**——R1-Distill 数据的来源说明（800k long-CoT trace 蒸馏），是 reasoning 数据的代表工作。

---

## 6. 自测与面试题

**Q1（方法概念）**：列出 SFT 数据 5 大来源（人工 / Self-Instruct / Magpie / Evol-Instruct / 蒸馏），各给 1 个代表数据集，并说明每个方法的核心 idea 与一个最大的局限。

<details>
<summary>Answer sketch</summary>

| 方法 | 代表数据集 | 核心 idea | 局限 |
|---|---|---|---|
| **人工标注** | OpenAssistant (33k) / LIMA (1k) | 人工写 prompt + 人工写 response | 贵、慢、规模有限（万级以上不可行） |
| **Self-Instruct** | Alpaca (52k) | 175 个 seed → GPT-3.5/4 用 8-shot 扩出新 instruction → LLM 生 response | 质量参差、重复多、有 GPT-3.5 风格 bias（凡事列 5 条） |
| **Magpie** | Magpie-Pro (1M-3M) | 给 chat model 输入空 user header → sample → 自吐 instruction 与 response（零 seed） | 依赖 base/chat model 已经训得好；diversity 来自 sample 温度 |
| **Evol-Instruct** | WizardLM (70k) | in-depth (add constraints / deepen / concretize / increase reasoning) + in-breadth (跨 domain) 让 instruction 变难变广 | API 成本高（多轮 LLM call）；evolved instruction 可能 unrealistic |
| **蒸馏** | Vicuna / OpenHermes-2.5 / Tülu-3 | 用 GPT-4 / Claude / 开源 strong model 生 response 蒸馏给弱 model | 法律灰色（OpenAI ToS 风险），商业用要换开源 model 蒸馏 |

加分要点：

- 提到现代主流是 **多源 mix**（OpenHermes-2.5、Tülu-3 都是综合多来源 + curation）
- 提到 **Magpie 是 2024 后规模化 SFT 数据成本压到极低的关键**（不需要 seed、不需要昂贵 API、几乎完全本地推理）
- 提到 **R1-Distill 是 2025 新趋势**（用 R1 / o1 生成 long-CoT trace 蒸馏）

</details>

**Q2（实战 pipeline）**：你要做一个中文 chat model 的 SFT，目标 1M 条高质量数据。列出从 0 到 1M 完整 pipeline（来源选择 → 生成/收集 → 过滤 → mixing → 最终 dataset）。

<details>
<summary>Answer sketch</summary>

**Step 1 — 多源采集（先汇集 5M+ 候选）**：

- **通用对话**：用 Magpie 范式让 Qwen-2.5-72B-Instruct 自吐中文 instruction-response（500k-1M）
- **复杂指令**：用 Evol-Instruct 把 Magpie 数据再 evolve 一轮（150k）
- **Code**：从 The Stack / OpenCodeInterpreter 中文化 + Code-Feedback 风格补 debug 流程（150k）
- **Math**：从 NuminaMath-CoT 翻译成中文 + 中文数学题 (Math23K) 重新生成 long-CoT response（120k）
- **Multi-turn 对话**：用 ShareGPT 中文子集 + COIG-CQIA + 自己 batch 生成（100k）
- **Refusal / Safety**：人工写 + 红队数据（30k）
- **多语言（中英混合）**：保留 5-10% 英文数据维持英文能力（100k）
- **Reasoning long-CoT**：用 DeepSeek-R1 / QwQ 重新生成 thinking trace（80k）

**Step 2 — 去重**：MinHash + LSH（threshold 0.8，K=128）整体 dedupe，约砍 20% → 4M

**Step 3 — 质量过滤**：

- 用 GPT-4o / Qwen-2.5-72B 当 LLM-as-judge 给所有数据打 1-5 分（4 维度：clarity / correctness / completeness / format）
- 留 ≥ 4 分的（约 50%-60% 留下）→ 2M-2.4M

**Step 4 — Diversity sampling**：

- 用 bge-large-zh 算 instruction 向量
- K-means 聚 500 个 cluster，按 cluster 平衡采样（避免常见话题如"写诗"占比过大）
- 采样到 1.5M

**Step 5 — Length / format / safety filter**：

- 过短 (< 20 token) / 过长 (> 8k) 丢
- 格式异常（乱码、控制字符、含训练 contamination 标记的）丢
- Llama-Guard-Chinese 过 safety → 1.2M

**Step 6 — Domain rebalance**：按目标比例混合（通用 50% / code 20% / math 15% / 多语言 10% / safety 5%）→ 最终 1M

**Step 7 — Contamination check**：把 C-Eval / CMMLU / GSM8K / HumanEval / IFEval 的题目转成 13-gram，substring match 命中的删掉

**Step 8 — 抽样人工 review**：随机抽 500 条人工标 quality，作为 holdout 监控指标

加分要点：

- 提到 **保留 long-CoT 数据**（避免 base model 是 thinking 类时丢失推理能力）
- 提到 **format diversity**（混 markdown / plain / code / table）
- 提到 **license / ToS 合规性**（用 Qwen / DeepSeek 蒸馏不用 GPT-4 蒸馏，规避商业风险）
- 提到 **训练前 holdout 一份高质量 eval set 作为快速 sanity check**

</details>

**Q3（trade-off / 边界）**：LIMA 实证 1000 条数据训出 strong model，但工业 SFT 普遍用 100k+ 甚至 1M+ 数据，为什么不直接学 LIMA 用 1k？至少 3 个原因 + 这条 trade-off 的边界。

<details>
<summary>Answer sketch</summary>

**为什么工业不直接用 1k**：

**原因 1：LIMA 在窄分布上 strong，但 coverage 不够**——1000 条数据无法覆盖现代 chat model 需要的所有 use case（多轮对话、code、math、多语言、tool calling、safety、不同 format）。每个 domain 都需要一定量的代表样本，1k 平均下来每个 domain 只剩几十条，模型见不够多 pattern 学不到稳定能力。LIMA 论文的 evaluation 也主要在 open-ended 单轮对话，没覆盖现代 chat model 的全部能力面。

**原因 2：LIMA 1k 是"极致精挑"**——人工从 Stack Exchange / WikiHow 几十万条候选里精筛、再人工修改的 1000 条，**每条 quality 接近天花板**。工业不可能投入这么大人力去精挑 1000 条，反而是"中等 quality 的 100k"在工程效率上更划算（自动化 pipeline 一晚上跑完）。

**原因 3：LIMA 测试场景是单轮 open-ended**——多轮对话、function calling、long-context、reasoning 这些现代必备能力，1k 数据训不出来。Multi-turn 一致性、tool 使用 schema 学习这些都需要看到大量样例才学得稳定。

**原因 4：现代 SFT 还要兼顾 RL stage 准备**——SFT 后接 DPO / GRPO（参考 Module 9），SFT 阶段的 model 需要在各种 domain 都有"基础能力"才能在 RL stage 进一步优化。1k 数据训出来的 model 在很多 domain 几乎是 zero-shot 水平，RL 阶段没东西可优化。

**原因 5：format diversity & robustness 需要量**——不同 format（markdown / JSON / table / latex）、不同长度、不同语言混在一起需要大量样本让 model"见多识广"，1k 远远不够。

**这条 trade-off 的边界**：

- **base model 强弱**：base 越强（更大、更新、pretrain 更好），SFT 数据越少越够——这也是 LIMA 用 LLaMA-65B 才成立，换 LLaMA-7B 同样 1k 数据效果会差很多
- **目标 use case 窄度**：如果只做单一 domain（比如纯 medical chat），1k-10k 精挑就够；做通用 chat 必须 100k+
- **是否后续接 RL**：纯 SFT 终点输出的 chat model 需要更多 SFT 数据；后接 DPO/RL 的 model SFT 阶段可以更精简
- **量是有上限的**：100k → 1M 的提升远比 10k → 100k 小，**1M 之后边际收益快速下降**——盲目堆量不如花同样精力提 quality

总结心智模型：**LIMA 证明的是 SFT 不需要数百万低 quality 数据；它没有证明 SFT 只需要 1k**。现代工业实践的甜点在 100k-1M 高 quality + 严格 curation。

</details>

---

## 7. 延伸阅读

- [HuggingFace Open SFT datasets list](https://huggingface.co/datasets?task_categories=task_categories:text-generation&sort=trending) — HuggingFace 上 trending SFT 数据集实时榜单，看现在大家在用什么。
- [Magpie GitHub](https://github.com/magpie-align/magpie) — Magpie 官方仓库，含完整数据生成脚本和过滤工具，可直接复用做中文版本。
- [WizardLM Evol-Instruct GitHub](https://github.com/nlpxucan/WizardLM) — Evol-Instruct 完整 prompt 模板（5 个 in-depth + 1 个 in-breadth）。
- [Tülu 3 报告](https://allenai.org/papers/tulu-3-report.pdf) — AI2 完整 post-training recipe，工业落地最完整的开源参考。
- [LIMA paper](https://arxiv.org/abs/2305.11206) — Meta "less is more" 原文，1000 条数据的来源筛选过程值得细看。
- [OpenHermes-2.5 dataset](https://huggingface.co/datasets/teknium/OpenHermes-2.5) — 2024 年开源界用得最广的综合 SFT 数据集，看它的 source mix 学习配比。
- [Datatrove + text-dedup](https://github.com/huggingface/datatrove) — HuggingFace 的数据清洗框架，SFT 阶段 dedupe 和 filter 都能直接用（与 6.2 同一套工具）。
- 推荐继续读本教程的 **8.2 节《SFT 训练细节》**——本节讲数据从哪来，8.2 讲拿到数据后 chat template 怎么拼、loss 怎么 mask、sample packing 怎么做；以及 **Module 9 偏好数据**——SFT 完了之后怎么用 preference 数据继续做 DPO / GRPO。
