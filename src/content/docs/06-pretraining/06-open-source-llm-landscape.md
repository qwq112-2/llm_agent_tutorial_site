---
title: "6.6 经典开源 LLM 解读：LLaMA / Qwen / DeepSeek 系列"
description: "把 2023-2026 年开源 LLM 的演化时间线、三大主流系列（LLaMA / Qwen / DeepSeek）每代的架构选择与训练 trick、以及怎么从 model card / paper 里 5 分钟读出\"它做了什么、做对了什么\"——这些过去四年最有价值的工程信号一次性梳理清楚。读完这节，你看到 `config.json` 里的 `num_attention_heads` / `num"
---

> ⏱ 预计阅读 45 分钟 ｜ 难度 ★★ ｜ 前置：Module 4（Transformer）、Module 5（GQA / MoE / 现代架构变体）

## 一句话本节讲什么

把 2023-2026 年开源 LLM 的演化时间线、三大主流系列（LLaMA / Qwen / DeepSeek）每代的架构选择与训练 trick、以及怎么从 model card / paper 里 5 分钟读出"它做了什么、做对了什么"——这些过去四年最有价值的工程信号一次性梳理清楚。读完这节，你看到 `config.json` 里的 `num_attention_heads` / `num_key_value_heads` / `intermediate_size` / `vocab_size` 几个字段就该知道这是"LLaMA-3 8B"、"Qwen2.5 7B" 还是 "DeepSeek-V3"，并且能讲出它们在 attention（MHA / GQA / MLA）、norm（RMSNorm + Pre-LN）、激活（SwiGLU）、位置编码（RoPE base 选择）、tokenizer（SentencePiece vs tiktoken-like）、over-train ratio、license 等维度上的关键差异。这是后训练 / 推理工程师面试的"地图题"，也是接手任何开源 base 做 SFT / RLHF 时的第一步功课。

---

## 1. Mental model（直觉）

### 1.1 一张图看懂：开源 LLM 时代地图（2022 → 2026）

```
2022 之前：BERT / GPT-2 / GPT-3 时代
            └─ 开源派靠 EleutherAI（GPT-J 6B、GPT-NeoX 20B）撑场
            └─ GPT-3 175B 不开源，与开源差距巨大

2023 Q1：LLaMA-1（Meta，6.7B / 13B / 65B）
            └─ research-only license，但 leak 出来引爆生态
            └─ 标准化"现代 LLM"组件：RoPE + SwiGLU + RMSNorm + Pre-LN
            └─ 衍生出 Alpaca / Vicuna / WizardLM 等指令微调狂潮

2023 Q3：LLaMA-2（7B / 13B / 70B）
            └─ 商用 license，第一个真正能商用的开源 LLM
            └─ 70B 引入 GQA-8（5.2 节讲过的 KV cache 压缩）
            └─ Qwen 1.0（阿里，0.5B-72B）同期发布，中文场景崛起

2023 Q4：Mistral 7B / Mixtral 8x7B（法国 Mistral）
            └─ 7B 引入 sliding window attention + GQA-8
            └─ Mixtral 是开源 MoE 标杆（5.4 节讲过）

2024 Q1-Q2：Qwen 1.5 / Qwen 2 / DeepSeek-V2
            └─ Qwen 切到 tiktoken-like BPE 词表 152k
            └─ DeepSeek-V2 引入 MLA + DeepSeek-MoE，236B / 21B 激活

2024 Q2-Q3：LLaMA-3 / LLaMA-3.1（8B / 70B / 405B）
            └─ 词表 32k → 128k；15T token over-train（远超 Chinchilla 比例）
            └─ 405B 是开源 dense 第一次冲击闭源旗舰

2024 Q4：DeepSeek-V3（671B / 37B 激活）
            └─ FP8 端到端训练 + MTP 辅助 + 14.8T token
            └─ 训练只花 ~5.5M USD（对比 GPT-4 估算 ~100M USD）

2025 Q1：DeepSeek-R1（基于 V3 base，纯 RL 激发 long-CoT）
            └─ "reasoning model" 范式开创者，详见 10.3
            └─ R1-Distill 系列把能力蒸馏到 1.5B-70B 小 model

2025 Q2-Q4：Qwen3 / Kimi-K2 / GLM-Z1 / Llama 4
            └─ reasoning 浪潮，几乎每个系列都出 thinking 模式
            └─ MoE 进一步主流化（Qwen3-MoE / Llama 4 Scout/Maverick）

2026：Hybrid 架构（Mamba × Transformer）/ native multimodal 走向开源
            └─ 长 context 128k+ 标配；闭源（GPT-4 / Claude 4.x / Gemini 2.x）
               仍领先 6-12 月，但开源差距持续缩小
```

记住三条主轴线：

- **Meta LLaMA 系**——西方开源标杆，把"现代 LLM 标准组件"打包给社区，每代节奏稳健
- **阿里 Qwen 系**——中文 / 多语场景最强，对齐 / instruction-following 业界口碑好
- **DeepSeek 系**——最具技术亮点的"工程派"，把 MLA / DeepSeek-MoE / FP8 / RLVR 一个接一个推到极致

Mistral / Gemma / Phi 是重要的"配角"，分别代表"欧洲精简派"、"Google 开源旗舰"、"小模型 + 高质量数据"三条小众路线。

### 1.2 为什么需要"读 model card 的眼力"

接手任何开源 base 做 SFT / RLHF / 部署时，第一件事都是看它的 `config.json` 与 paper / model card——里面藏着所有重要工程信号：

- **能不能商用**？看 license（Apache 2.0 / MIT 是真开源；LLaMA-2 license 商用要 < 7 亿月活；Mistral medium 是研究 only）
- **架构能不能直接 SFT**？看 attention 类型（MHA / GQA / MLA），MLA 在 HuggingFace `transformers` 里训练支持很弱
- **chat template 是什么**？SFT 的 loss mask 必须 align，错了直接训出垃圾
- **tokenizer 是什么**？SentencePiece vs tiktoken-like 的处理 path 完全不同
- **训练 token 数是多少**？决定它"还能不能再 SFT"——over-trained 模型（LLaMA-3 用 75× Chinchilla）的 SFT 数据需求与 LLaMA-2（28× Chinchilla）不一样
- **context length**？决定能装多长的 prompt + history

这一节会把这些"读图技巧"系统化讲清楚。**面试时如果一个候选人说不出"LLaMA-3 8B 与 Qwen2.5 7B 的关键差异"，他大概率没真正用过开源 LLM**。

### 1.3 现代 LLM 的"标配组件"——LLaMA-1 留下的遗产

LLaMA-1（2023.02）最大的贡献不是 65B 这个数字，而是**把"现代 LLM 的标准组件"打包定型**——之后几乎所有开源 decoder-only 模型都沿用这套组合：

| 组件 | LLaMA-1 选择 | 替代方案 | 现状 |
|---|---|---|---|
| 位置编码 | **RoPE**（4.3 节） | 绝对 / ALiBi / 相对 | RoPE 已成事实标配 |
| 激活 | **SwiGLU**（4.5 节） | ReLU / GELU | SwiGLU 主流，少数用 GeGLU |
| Norm | **RMSNorm**（1.3 节） | LayerNorm | RMSNorm 主流 |
| Norm 位置 | **Pre-LN**（4.4 节） | Post-LN / Sandwich | Pre-LN 主流 |
| Attention | MHA | GQA / MLA | LLaMA-2 70B 起切 GQA |
| Bias | **去掉所有 Linear 的 bias** | 带 bias | 主流去 bias |

**任何 2023 年后出现的"现代 LLM"几乎都满足这套配置**——Qwen / Mistral / Gemma / Phi / DeepSeek 全部以这套为 baseline，差异只在 attention（MHA/GQA/MLA）、tokenizer、训练数据规模、对齐 recipe 这几个维度。

所以"读懂一个新开源 LLM"的成本其实很低——大部分组件都是已知的，你只需要看它在哪几个维度做了不同的选择。

---

## 2. 三大主流系列详解

### 2.1 LLaMA 系列（Meta，2023-）：西方开源标杆

LLaMA 系列的演化是**最稳健**的——每代都在前代基础上做"渐进式升级"，没有激进的架构换代。

#### LLaMA-1（2023.02）：标准化"现代 LLM 组件"

- **配置**：6.7B / 13B / 32.5B / 65B 四档
- **架构**：MHA + RoPE + SwiGLU + RMSNorm + Pre-LN（首次把这套组合定型）
- **数据**：1T-1.4T token，CommonCrawl + C4 + GitHub + Wikipedia + 论文 + StackExchange
- **License**：research-only（这就是它后来 leak 引爆社区的伏笔）
- **历史意义**：Alpaca / Vicuna / WizardLM 等所有 2023 中文社区的 SFT 模型几乎都建在 LLaMA-1 上

#### LLaMA-2（2023.07）：商用 license + GQA 首秀

- **配置**：7B / 13B / 70B
- **架构升级**：70B 引入 **GQA-8**（$h_q = 64, h_{kv} = 8$，5.2 节）
- **数据**：2T token（比 LLaMA-1 翻倍）；context 4k
- **License**：Llama 2 Community License（< 7 亿月活可商用，第一个真正商用友好的开源 LLM）
- **chat 版本**：LLaMA-2-Chat 是第一个完整走 SFT + RM + RLHF 的开源对齐 model（详见 Module 9）
- **工程意义**：Mistral / Qwen / Gemma 等所有后续模型的 GQA 配置都参考 LLaMA-2 70B 的设计

#### LLaMA-3（2024.04）：词表升级 + over-trained 时代

- **配置**：8B / 70B（后续 3.1 加 405B）
- **关键变化**：
  - 词表从 32k（SentencePiece）扩到 **128k**（tiktoken-style BPE，与 GPT-4 同款）——中文 / 代码效率大幅提升
  - **GQA-8 全系**（包括 8B，不再只是 70B 才用）
  - 训练数据 **15T token**（LLaMA-2 的 7.5×），属于典型"over-trained"配置（详见 6.3 Scaling Law）
  - 新 chat template `<|begin_of_text|><|start_header_id|>user<|end_header_id|>...`
  - context 8k

#### LLaMA-3.1 / 3.2 / 3.3（2024.07+）：long context + multimodal + 405B

- **3.1**：context 8k → **128k**（用 RoPE base scaling，详见 6.5）；新增 **405B 旗舰**
- **3.2**：multimodal（11B / 90B vision 版本）+ 1B / 3B 端侧版本
- **3.3**：纯文本 70B refresh，能力对齐 405B 但便宜

#### LLaMA 系列关键架构表（必背）

| 模型 | params | n_layer | d (hidden) | h_q | h_kv (g) | d_ff | training tokens |
|---|---|---|---|---|---|---|---|
| LLaMA-1 7B | 7B | 32 | 4096 | 32 | 32 (MHA) | 11008 | 1T |
| LLaMA-2 7B | 7B | 32 | 4096 | 32 | 32 (MHA) | 11008 | 2T |
| LLaMA-2 70B | 70B | 80 | 8192 | 64 | 8 (GQA) | 28672 | 2T |
| LLaMA-3 8B | 8B | 32 | 4096 | 32 | 8 (GQA) | 14336 | 15T |
| LLaMA-3 70B | 70B | 80 | 8192 | 64 | 8 (GQA) | 28672 | 15T |
| LLaMA-3 405B | 405B | 126 | 16384 | 128 | 8 (GQA) | 53248 | 15.6T |

**眯眼能看出的三件事**：(1) 7B / 70B 的 hidden / layer 配置三代几乎不变，主要差异在数据量与 GQA；(2) LLaMA-3 8B 用 15T token 是 LLaMA-2 7B 的 7.5×——典型 over-train；(3) 405B 用 8 个 KV head 与 70B 一致，attention 显存与 70B 相当（5.2 节算账）。

### 2.2 Qwen 系列（阿里，2023-）：中文 / 多语场景最强

Qwen 系列的演化关键词是**中文优先 + 对齐口碑**。架构上跟随 LLaMA，但 tokenizer / chat template / 训练数据有自己的特色。

#### Qwen 1.0（2023.08）

- **配置**：0.5B / 1.8B / 4B / 7B / 14B / 32B / 72B（粒度比 LLaMA 细很多）
- **架构**：基本 LLaMA 风（RoPE + SwiGLU + RMSNorm），但带 attention bias（少数现代 LLM 仍带 bias 的）
- **特色**：中英双语对齐重视，词表 152k tiktoken-like BPE（不是 SentencePiece）

#### Qwen 1.5 / Qwen 2（2024）

- **架构**：MHA → GQA 全系切换；Qwen2 7B 用 GQA-4（$h_q = 28, h_{kv} = 4$）
- **特色**：context 32k → 128k；引入 sliding window 部分变体
- **chat template**：`<|im_start|>user\n...\n<|im_end|>\n<|im_start|>assistant\n...`（ChatML 风格）

#### Qwen 2.5（2024.09）：code & math 强

- **配置**：0.5B / 1.5B / 3B / 7B / 14B / 32B / 72B + Qwen2.5-Coder + Qwen2.5-Math 专项
- **数据**：18T token（比 LLaMA-3 的 15T 还多），code & math 占比高
- **特色**：在中文、代码、数学三个维度上 7B / 72B 几乎都是同档最强开源 base

#### Qwen 3（2025+）：reasoning + thinking mode

- **特色**：
  - 引入 **thinking mode 切换**——同一 model 支持"立即回答"与"先 think 再答"两种模式
  - 同期推出 Qwen3-Agent / Qwen3-VL / Qwen3-Coder 等 task-specific 变体
  - MoE 版本（Qwen3-MoE）跟进 DeepSeek 路线

**Qwen 系列读 config 时的关键信号**：(1) tokenizer 是 `Qwen2Tokenizer`（tiktoken-like），不是 LLaMA 的 `LlamaTokenizer`（SentencePiece），处理 path 不同；(2) chat template 是 ChatML 风格 `<|im_start|>` / `<|im_end|>`；(3) Qwen2 系开始 attention 不带 bias，但早期 Qwen 1.0 带 bias——load checkpoint 时要注意 config 字段 `attention_bias`。

### 2.3 DeepSeek 系列（深度求索，2023-）：技术亮点最密集

DeepSeek 是 2024 年开源 LLM 里**最具工程亮点**的系列——几乎每代都有一个全新的架构创新，把 MoE / MLA / FP8 / RLVR 这些方向各推到一个新的高度。

#### DeepSeek-V1（2023.11）：标准 LLaMA-clone

- 67B dense，基本是 LLaMA-2 70B 的复刻，没有特别的架构创新
- 主要价值是建立训练 / 数据 pipeline，为后续 V2 / V3 铺路

#### DeepSeek-MoE（2024.01）：fine-grained + shared expert

- **关键 trick**：**fine-grained expert**（更多更小，64 个 expert）+ **shared expert**（1-2 个始终激活）
- 详见 5.4 节——DeepSeek-MoE 是开源 MoE 路线的关键创新，影响了后续所有 fine-grained MoE 模型

#### DeepSeek-V2（2024.05）：MLA 首秀

- **配置**：236B 总参 / 21B 激活
- **关键创新**：
  - **MLA**（Multi-head Latent Attention，5.2 节详讲）——把 KV cache 压到 ~1152 B / token / layer，比 GQA-8 还小 2×
  - **DeepSeek-MoE**（256 expert + 2 shared）
- **意义**：第一次让 236B 量级的 MoE 模型在 H800 上能 batch 服务多 user，推理便宜 + 性能强

#### DeepSeek-Coder-V2（2024.06）：code-focused + FIM

- 基于 V2 架构 fine-tune 到代码任务，增加 **FIM**（Fill-in-the-Middle，6.1 节）训练目标
- 是 2024 年开源 code model 的最强档

#### DeepSeek-V3（2024.12）：FP8 端到端 + MTP

- **配置**：671B 总参 / 37B 激活，14.8T token
- **关键创新**：
  - **FP8 端到端训练**（详见 7.4 混合精度）——比 bf16 节省 ~50% 显存与带宽
  - **MTP**（Multi-Token Prediction）训练辅助——让 model 同时预测下 1、下 2、下 3 个 token，提升训练效率
  - **bias-based balancing**（无 aux loss MoE，5.4 节）
- **传奇数字**：训练只花 ~5.5M USD compute（GPT-4 估算 ~100M USD）——靠 FP8 + MoE + MLA + 自研 EP kernel（DeepEP）的全栈优化才达到

#### DeepSeek-R1（2025.01）：reasoning model 浪潮开创者

- 基于 V3 base，**纯 RL（GRPO + verifiable reward）激发 long-CoT**——不依赖 SFT 也能训出 reasoning 能力
- 配套 **R1-Distill 系列**（1.5B / 7B / 14B / 32B / 70B），把 reasoning 能力蒸馏到小 model
- 是 "reasoning model" 范式的开创者，技术细节详见 **10.3 RLVR 与 DeepSeek-R1**——本节只用作"reasoning 浪潮"的标志事件提一句

### 2.4 Mistral / Gemma / Phi 速览

| 系列 | 出品方 | 特色 | 代表 model |
|---|---|---|---|
| Mistral 7B | 法国 Mistral AI | sliding window attention + GQA-8 | Mistral 7B v0.1 / v0.3 |
| Mixtral | Mistral AI | 8 expert + top-2 MoE，开源 MoE 标杆 | Mixtral 8x7B / 8x22B |
| Mistral Large/Medium | Mistral AI | 闭源旗舰（API only） | Mistral Large 2 |
| Gemma 2 / 3 | Google | soft-cap attention + 256k 词表 | Gemma 2 9B/27B、Gemma 3 multimodal |
| Phi 2/3/4 | Microsoft | "textbook is all you need"，data quality 极致 | Phi-3 mini 3.8B、Phi-4 14B |

**记忆抓手**：
- Mistral = "欧洲精简派"，模型小但每个组件都精细打磨
- Gemma = "Google 开源代表"，词表大（256k）+ soft-cap attention 是特色
- Phi = "小模型 + 高质量数据"路线，benchmark 好看但 generalize 到非 textbook 任务有局限

### 2.5 怎么 5 分钟读完一个开源 LLM 的 model card / paper

读到一个新 LLM，按以下顺序快速 scan：

1. **Params 总数 / 激活数**（MoE 必看激活）—— 决定显存、speed
2. **Training tokens**—— 决定 over-train 程度（< 20× Chinchilla 是 under-train，> 50× 是 over-train）
3. **Tokenizer 类型与词表大小**—— SentencePiece (LLaMA) / tiktoken-like (Qwen, GPT-4) / 自研
4. **Context length**—— 8k / 32k / 128k / 1M
5. **Attention 类型**—— MHA / GQA (g 多少) / MLA
6. **Norm / 激活 / 位置编码**—— 一般都是 RMSNorm + Pre-LN + SwiGLU + RoPE，但 RoPE base 值（10000 / 500000 / 1000000）告诉你 long context 处理方式
7. **Chat template**—— SFT / 推理 align 的关键
8. **License**—— 商用 / 研究 / 数据可用性
9. **Ablation 表**—— 比主结果更有信息量，告诉你哪些 trick 真有用
10. **训练 infra**—— 用了多少卡 × 多少天，用了什么 parallelism（DDP / TP / PP / EP / SP）

**最关键的两条**：(1) **看 ablation 比看主结果更有信息量**——主结果是 "我的 model 多强"，ablation 才是 "哪些 trick 真有用"，对工程师价值大得多；(2) **看 license**——LLaMA-2 license 商用要 < 7 亿月活，DeepSeek-V3 是 MIT，Qwen 系是 Tongyi Qianwen License，差异很大。

### 2.6 2026 现状速览（必知）

- **Reasoning model 已是开源主流**——R1 / QwQ / Qwen3 / GLM-Z1 / Llama 4 Reasoning 等都有 thinking 模式
- **MoE 普遍化**——DeepSeek / Mixtral / Qwen-MoE / GLM-MoE / Llama 4 全在 MoE 路线
- **长 context 128k+ 标配**，部分模型 1M（Gemini / Qwen2.5-1M）
- **native multimodal 是新前沿**——Gemma 3 vision、Qwen-VL、InternVL、Llama 4 multimodal
- **闭源仍领先 6-12 月**（GPT-4 / Claude 4.x / Gemini 2.x），但**开源差距持续缩小**——DeepSeek-V3 已在多数 benchmark 接近 GPT-4，R1 已接近 o1

---

## 3. 最小代码示例

### 3.1 从 HuggingFace 加载并对比三个 model 的 config

```python
# pip install transformers
from transformers import AutoConfig

models = {
    "LLaMA-2 7B":   "meta-llama/Llama-2-7b-hf",
    "LLaMA-3 8B":   "meta-llama/Meta-Llama-3-8B",
    "Qwen2.5 7B":   "Qwen/Qwen2.5-7B",
    "DeepSeek-V3":  "deepseek-ai/DeepSeek-V3",  # 671B MoE，仅看 config
}

print(f"{'Model':14s} {'vocab':>7s} {'L':>3s} {'d':>5s} {'h_q':>4s} {'h_kv':>4s} {'d_ff':>6s} {'attn'}")
for name, repo in models.items():
    c = AutoConfig.from_pretrained(repo, trust_remote_code=True)
    n_kv = getattr(c, "num_key_value_heads", c.num_attention_heads)
    attn = "MLA" if "deepseek" in repo.lower() and hasattr(c, "kv_lora_rank") else \
           ("GQA" if n_kv < c.num_attention_heads else "MHA")
    d_ff = getattr(c, "intermediate_size", getattr(c, "moe_intermediate_size", -1))
    print(f"{name:14s} {c.vocab_size:>7d} {c.num_hidden_layers:>3d} {c.hidden_size:>5d} "
          f"{c.num_attention_heads:>4d} {n_kv:>4d} {d_ff:>6d} {attn}")

# 期望输出（数字以官方 config 为准）：
# Model          vocab   L     d  h_q h_kv   d_ff attn
# LLaMA-2 7B     32000  32  4096   32   32  11008 MHA
# LLaMA-3 8B    128256  32  4096   32    8  14336 GQA
# Qwen2.5 7B    152064  28  3584   28    4  18944 GQA
# DeepSeek-V3   129280  61  7168  128  128   2048 MLA   # MoE expert 维度
```

**这段代码是面试常用的"读 config 题"答案**——能从 `vocab_size` / `num_key_value_heads` / `kv_lora_rank` 等字段一眼看出模型属于哪个系列与代际。注意：DeepSeek-V3 是 MoE，`intermediate_size` 是单个 expert 的 FFN 维（而不是 dense FFN），需要结合 `n_routed_experts` / `num_experts_per_tok` 才能算出激活参数（详见 5.4 节）。

### 3.2 用 transformers 跑同一 prompt 对比 LLaMA-3 vs Qwen2.5

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

prompt = "用一句话解释什么是 attention 机制。"
for name, repo in [("LLaMA-3 8B-Instruct", "meta-llama/Meta-Llama-3-8B-Instruct"),
                   ("Qwen2.5 7B-Instruct", "Qwen/Qwen2.5-7B-Instruct")]:
    tok = AutoTokenizer.from_pretrained(repo)
    mdl = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.bfloat16, device_map="auto")

    # 关键：用各自的 chat template，不要手拼
    msgs = [{"role": "user", "content": prompt}]
    inputs = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(mdl.device)
    out = mdl.generate(inputs, max_new_tokens=80, do_sample=False)
    print(f"\n=== {name} ===")
    print(tok.decode(out[0][inputs.shape[1]:], skip_special_tokens=True))
```

**关键 1 行**：`tok.apply_chat_template(msgs, add_generation_prompt=True)`——这是 HuggingFace 自 2024 中起的标准做法，自动处理每个模型的不同 chat template（LLaMA-3 的 `<|start_header_id|>` vs Qwen 的 `<|im_start|>`）。手拼 chat template 是 SFT / 推理出错的头号坑（详见 §4 第 1 条）。

---

## 4. 工程踩坑与经验

- ❗ **不同 LLM 的 chat template 完全不同，SFT / 推理时务必 align**。LLaMA-2 用 `[INST] ... [/INST]`、LLaMA-3 用 `<|start_header_id|>user<|end_header_id|>...<|eot_id|>`、Qwen 用 ChatML `<|im_start|>user\n...\n<|im_end|>`、DeepSeek 用 `<｜User｜>...<｜Assistant｜>`（注意是全角 ｜ 不是 |）。**用错 chat template 会让 SFT 直接训出垃圾 model**——loss mask 错位、推理时 stop token 不匹配、对齐能力全部失效。HuggingFace 2024 中起统一用 `tokenizer.apply_chat_template(messages, ...)`，**永远不要自己手拼**。同时检查 `tokenizer.chat_template` 字段是否存在——少数早期 ckpt 没有 template 字段，要从 model card 抄过来。

- ❗ **LLaMA-3 把词表从 32k 升到 128k，与 LLaMA-2 不兼容**。embedding 层从 $32000 \times 4096$ 变成 $128256 \times 4096$，无法 hot-swap；任何在 LLaMA-2 上训好的 LoRA / adapter / 词表扩展工具都要重做。**实战影响**：LLaMA-2 时代的中文社区 SFT 工具（如各种"中文 LLaMA 词表扩展"）在 LLaMA-3 上完全失效——LLaMA-3 的 128k 词表对中文已经友好得多，不需要再扩。同样地，LLaMA-3 的 `<|begin_of_text|>` / `<|end_of_text|>` / `<|eot_id|>` 等 special token id 与 LLaMA-2 全部不同，写 stop criteria 时要按 model 重设。

- ❗ **Qwen 系 tokenizer 是 tiktoken-like，不是 SentencePiece——code path 不一样**。LLaMA / Mistral / Gemma 用 `LlamaTokenizer` / `SentencePiece`（基于 BPE 但有 SP 的 byte fallback），Qwen 用 `Qwen2Tokenizer`（tiktoken-style 纯 BPE，与 GPT-4 / LLaMA-3 同款）。**实战影响**：(1) Qwen tokenizer 必须 `trust_remote_code=True` 才能 load（早期版本）；(2) sentence-piece 工具链（`sentencepiece` 库的 `SentencePieceProcessor`）对 Qwen 不工作；(3) 词表 merge / 扩展工具要分开维护。Qwen2.5+ 的 tokenizer 已被 transformers 主线支持，但仍要注意 `tokenizer_config.json` 里的 `tokenizer_class` 字段。

- ❗ **DeepSeek-V3 用 MLA + FP8 训练，自己复现需要专门的 kernel；HF transformers 不支持 MLA training**。MLA 的 RoPE 解耦设计（5.2 §2.5）需要专用 kernel；FP8 端到端训练需要 NVIDIA H100/H200 的 Transformer Engine + 自定义 scale 管理。**transformers 库的 `DeepseekV2Model` / `DeepseekV3Model` 只支持 inference 与 LoRA fine-tune，不支持 full pretraining**——如果你要从 V3 base 做 continual pretraining，必须用 DeepSeek 官方的 `DeepEP` + `Megatron-DeepSpeed` fork 或类似 framework。SFT / RLHF 用 LoRA 或 QLoRA 通常没问题，但要小心 LoRA rank 设大了会撞 MLA latent 的 bottleneck。

- ❗ **R1 类 reasoning model 在 generation 时会输出大段 `<think>...</think>`，下游应用要解析或丢弃**。DeepSeek-R1 / QwQ / Qwen3 thinking mode / GLM-Z1 都会在 final answer 前先输出一段 reasoning（`<think>` 标签包裹），通常占总 token 的 70%+。**直接把这段塞给下游 user 会让产品显得"啰嗦"**——production 应用通常 split 出 `<think>` 与 final answer，think 部分只用于 telemetry / debug。同时计费场景注意：reasoning token 也算 output token，cost 比 non-reasoning model 高 3-5×。Qwen3 等支持 thinking 模式切换的 model，在不需要 reasoning 时务必把 `enable_thinking=False`。

- ❗ **MoE model（Mixtral / DeepSeek）的 fine-tune 与 dense 大不同——RM / policy / ref 都要 expert，显存爆炸**。完整 PPO 显存里有 4 个模型副本（policy / ref / RM / critic），dense 模型已经吃紧，MoE 直接 4× 总参——Mixtral 8x22B (141B) PPO 需要 ~600 GB 显存，至少 8×H100 才能跑；DeepSeek-V3 (671B) PPO 实际不可行，业界基本只在 V3 上做 LoRA-based DPO 或 GRPO（详见 9.3-9.5）。**实战教训**：MoE base 上做 RLHF 时，能用 DPO / RLOO / GRPO（去 critic、去 RM 的变体）就别用 PPO；LoRA + bf16 是最低预算的 minimum viable setup。

- ❗ **Phi 系小 model 在 textbook quality data 上表现好，但 generalize 到非 textbook 任务有局限**。Phi-3 mini 3.8B 在 MMLU / GSM8K 等 academic benchmark 接近 LLaMA-3 8B，但在真实多轮对话、agent / tool use、中文场景上明显弱——因为训练数据 heavy 偏向 textbook / synthetic GPT-4 改写。**选型建议**：研究 / demo / 小数据 SFT 选 Phi 没问题；production 应用要做 instruction-following / tool use / 中文，宁可选 Qwen2.5 3B 或 LLaMA-3.2 3B。

- ❗ **License 雷区**：LLaMA-2 license 商用要求 < 7 亿月活、必须显示 "Built with Meta Llama"；Qwen 系是 Tongyi Qianwen License（与 Apache 2.0 类似但有限制）；DeepSeek-V3 / R1 是 MIT（最宽松）；Mistral Large / Medium 是闭源 API；Gemma 是 Gemma Terms of Use（限制 use case）；Phi 系是 MIT。**做 production 选型时 license 要在第一步就确认**——用错 license 后期合规整改成本极高。商用最安全的开源选择：DeepSeek 系（MIT）、Mistral 7B（Apache 2.0）、Phi（MIT）。

---

## 5. 经典 paper

- **Touvron et al., 2023 — LLaMA: Open and Efficient Foundation Language Models** — LLaMA-1 原 paper。读它能看到"现代 LLM 标准组件"（RoPE + SwiGLU + RMSNorm + Pre-LN）的完整定型，以及"用 1.4T 公开数据 train 65B 能打过 GPT-3 175B"的 Chinchilla 时代精神。这是开源 LLM 时代的起点 paper。
- **Touvron et al., 2023 — Llama 2: Open Foundation and Fine-Tuned Chat Models** — LLaMA-2 paper，§2.2 详述 GQA-8 的工程动机与配置（与 Ainslie 2023 GQA paper 配合读），§3 完整 SFT + RM + RLHF pipeline 是开源对齐范式的标杆。
- **Dubey et al., 2024 — The Llama 3 Herd of Models** — LLaMA-3 全 family 综述，~92 页非常厚。重点读 §2（数据 pipeline）、§3（pretrain scale）、§4（post-train）、§5（multimodal）。这是 2024 年最值得精读的 LLM tech report 之一，把"15T token over-train + 405B dense"路线讲透。
- **Yang et al., 2024 — Qwen2 / Qwen2.5 Technical Report** — Qwen 系最新 tech report，重点看数据配比、code/math 数据 ablation、tokenizer 设计。Qwen 系的工程细节往往比 LLaMA 公开得更细，对中文场景工程师价值高。
- **DeepSeek-AI, 2024 — DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model** — MLA + DeepSeek-MoE 的奠基 paper，§2.1 是 MLA 完整描述，§2.2 是 fine-grained + shared expert。读这一篇 = 学懂 DeepSeek 系全部架构创新的一半。
- **DeepSeek-AI, 2024 — DeepSeek-V3 Technical Report** — **2024 最值得精读的 LLM tech report，没有之一**。FP8 端到端训练、MTP、bias-based balancing、DeepEP 通信优化全栈细节，671B / 37B 在 ~5.5M USD 训出 GPT-4 级能力。任何 LLM infra 工程师都应该精读一遍。
- **DeepSeek-AI, 2025 — DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning** — reasoning model 浪潮的标志 paper，纯 RL（GRPO + verifiable reward）激发 long-CoT。**详细技术细节本节不展开，留给 10.3 节精讲**——本节只把它作为"reasoning 范式开创"的标志事件提一句。
- **Jiang et al., 2024 — Mixtral of Experts** — Mixtral 8x7B 工程细节最清楚的 paper，是开源 MoE 的标准对照。配合 5.4 节 MoE 食用。
- **加分**：Gemma 2 / Gemma 3 paper（看 soft-cap attention 与 256k 词表设计）；Mistral 7B paper（看 sliding window attention 工程）。

---

## 6. 自测与面试题

**Q1（架构对比 / 算账）**：列出 **LLaMA-3 8B / Qwen2.5 7B / DeepSeek-V3** 三个 model 在以下 6 个维度的关键差异：(1) 注意力类型 (2) Norm 类型与位置 (3) 激活函数 (4) 位置编码 (5) 词表大小 (6) over-train ratio（用 Chinchilla 比例 20× 作 baseline）。

<details>
<summary>Answer sketch</summary>

至少要点到：

| 维度 | LLaMA-3 8B | Qwen2.5 7B | DeepSeek-V3 |
|---|---|---|---|
| 注意力 | **GQA-8** ($h_q=32, h_{kv}=8$) | **GQA-4** ($h_q=28, h_{kv}=4$) | **MLA**（latent dim 512 + rope dim 64）|
| Norm | RMSNorm + Pre-LN | RMSNorm + Pre-LN | RMSNorm + Pre-LN |
| 激活 | SwiGLU | SwiGLU | SwiGLU |
| 位置编码 | RoPE (base=500000) | RoPE (base=1000000) | RoPE 解耦版（4.3 + 5.2）|
| 词表 | 128k（tiktoken-like） | **152k**（tiktoken-like，最大）| 129k（tiktoken-like）|
| 训练 token | 15T | 18T | 14.8T |
| Chinchilla 倍数 | 15T / (20 × 8B) = **94×** | 18T / (20 × 7B) = **129×** | 14.8T / (20 × 37B 激活) ≈ **20×**（按激活算）|

**关键观察**：
- 三者 Norm / 激活 / 位置编码都是 LLaMA-1 那套"现代 LLM 标配"——演化的差异主要在 attention 与 tokenizer
- DeepSeek-V3 是 MoE，要按激活参数算 Chinchilla 比例（37B 而非 671B），结果反而是三者中"最保守"
- Qwen2.5 7B 用 18T token over-train 到 129× Chinchilla，是过度训练程度最深的——这是它在中小尺寸 benchmark 表现强的原因之一
- 词表 Qwen 152k 最大，对多语言（中 / 日 / 韩）友好

加分：能算出三者的 KV cache 大小差异（LLaMA-3 8B GQA-8 = 2048 B/token/layer；Qwen2.5 7B GQA-4 = 1024 B/token/layer；DeepSeek-V3 MLA ≈ 1152 B/token/layer），并指出 Qwen2.5 7B 的 KV cache 单 token 最小、最适合长 context 推理。

</details>

**Q2（趋势 / 历史观）**：从 LLaMA-1（2023.02）到 DeepSeek-V3（2024.12），开源 LLM 在架构与训练上的关键技术演化是什么？至少列 4 条。

<details>
<summary>Answer sketch</summary>

至少要点到 4 条（每条配一句"为什么"）：

1. **Attention：MHA → GQA → MLA**——KV cache 压缩需求驱动；70B+ 模型纯 MHA 推理经济上不可行；MLA 是 long context 极致路线
2. **Dense → MoE**——总参数与激活参数解耦（5.4 节）；让 671B model 推理像 37B 一样快；2024 后开源 frontier 模型几乎清一色 MoE
3. **训练数据：1T → 15T+，从 Chinchilla 比例（20×）演化到 over-train（75-130×）**——推理成本远大于训练成本时，多 train 划算（详见 6.3）；LLaMA-3 与 Qwen2.5 是典型代表
4. **词表：32k → 128k+**（tiktoken-like 取代 SentencePiece）——多语言 / 代码效率提升；与 GPT-4 同款 tokenization 让对比公平
5. **Context：4k → 128k → 1M**——RoPE base scaling / NTK / YaRN / LongRoPE 等长 context 技术成熟（6.5 节）
6. **训练精度：fp32 → bf16 → fp8**（DeepSeek-V3 端到端 fp8）——节省显存与带宽，进一步压低训练成本（详见 7.4）
7. **对齐范式：SFT → SFT + RM + PPO → DPO → GRPO**（详见 Module 9）
8. **Reasoning model 浪潮**：纯 RL 激发 long-CoT 成为新范式（10.3）

加分：能指出每个演化背后的"工程驱动力"（推理成本、long context 需求、对齐稳定性等），而不只是罗列技术名词；能讲"开源差距从落后 GPT-4 18 个月缩小到 6-12 个月"的整体节奏。

</details>

**Q3（实战选型 / 工程判断）**：你在做一个**中文场景的 SFT** 项目（数据 ~50k 条多轮对话，目标是 chatbot），要从 **LLaMA-3 8B / Qwen2.5 7B / DeepSeek-V2-Lite (MoE)** 三个 base 中选一个，理由？

<details>
<summary>Answer sketch</summary>

**首选 Qwen2.5 7B**，主要理由（至少 3 条）：

1. **中文质量天花板更高**：Qwen 系训练数据中文占比高、tokenizer 词表 152k 对中文 token 化效率好（中文字 / 词的平均 token 数比 LLaMA-3 的 128k 词表更少），decode 速度与上下文利用率都好。LLaMA-3 在中文场景虽然词表 128k 比 LLaMA-2 32k 进步巨大，但中文知识深度仍比 Qwen 差一档
2. **chat template 与对齐更成熟**：Qwen2.5-Instruct 的 baseline chat 能力已经很强，SFT 起步分高；中文 system prompt / 多轮 context 处理已经"开箱即用"
3. **生态与文档对齐**：阿里 / DataWhale / ModelScope 等中文社区有大量 Qwen SFT recipe，遇到 issue 容易找到答案；Qwen 官方 ms-swift / Qwen-Agent 等工具链对中文场景调试友好
4. **size 与 GQA-4 推理友好**：7B + GQA-4 (KV cache 1024 B/token/layer) 是同档最便宜的推理配置，production 部署成本低

**为什么不选其他两个**：

- **LLaMA-3 8B**：中文知识相对弱、词表对中文不如 Qwen 紧凑；如果 SFT 数据已经覆盖了中文 domain，可以用，但 cold start 弱于 Qwen。**适合场景**：英文为主 / 多语场景；想要 Meta 品牌 / 严格 license 合规
- **DeepSeek-V2-Lite (16B/2.4B 激活 MoE)**：虽然技术亮点多（MLA + DeepSeek-MoE），但**MoE base 做 SFT 的踩坑远多于 dense**——LoRA 加在哪 expert 没标准方案、HF transformers 对 MLA training 支持弱、推理引擎适配（vLLM / SGLang）需要新版本。**50k 条对话数据上 MoE 优势体现不出来**，工程复杂度反而拉高。**适合场景**：数据量很大（M+ 条）+ infra 成熟时

**反例 / 加分**：
- 如果数据是 code / math 为主：可以考虑 Qwen2.5-Coder / Qwen2.5-Math 专项 base，比通用 7B 更好
- 如果要做 reasoning agent：可以考虑从 R1-Distill-Qwen-7B 起步（已有 long-CoT 能力）
- 如果 license 极敏感：DeepSeek（MIT）> Qwen（Tongyi Qianwen License）> LLaMA-3（Meta License）
- 能指出"先用 50 条数据快速 SFT 三个 base 跑 small-scale ablation 看 baseline，再做最终选型"——这是真实工程师的做法，而不是纸上谈兵

</details>

---

## 7. 延伸阅读

- [Llama 2 paper (arXiv:2307.09288)](https://arxiv.org/abs/2307.09288) — LLaMA-2 完整 tech report，§2 architecture + §3 RLHF 必读
- [Llama 3 Herd of Models (arXiv:2407.21783)](https://arxiv.org/abs/2407.21783) — LLaMA-3 family 92 页综述，2024 年开源 LLM 工程范本
- [Qwen2.5 Technical Report (arXiv:2412.15115)](https://arxiv.org/abs/2412.15115) — Qwen2.5 数据 / 训练 / 评测细节，中文场景工程师必读
- [DeepSeek-V2 paper (arXiv:2405.04434)](https://arxiv.org/abs/2405.04434) — MLA + DeepSeek-MoE 奠基 paper
- [DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) — **2024 最值得精读的 LLM tech report**，FP8 + MLA + MoE 全栈细节
- [DeepSeek-R1 paper (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948) — reasoning model 浪潮起点；技术细节看 10.3
- [Mixtral of Experts paper (arXiv:2401.04088)](https://arxiv.org/abs/2401.04088) — 开源 MoE 标杆 tech report
- [Mistral 7B paper (arXiv:2310.06825)](https://arxiv.org/abs/2310.06825) — sliding window attention + GQA-8
- [Gemma 2 paper (arXiv:2408.00118)](https://arxiv.org/abs/2408.00118) — Google 开源旗舰，soft-cap attention 设计
- [Phi-3 Technical Report (arXiv:2404.14219)](https://arxiv.org/abs/2404.14219) — small model + textbook data 路线代表
- [HuggingFace Open LLM Leaderboard](https://huggingface.co/open-llm-leaderboard) — 开源 LLM benchmark 排行榜，看 frontier 进展
- [LMSYS Chatbot Arena](https://lmarena.ai/) — 真人盲评的 LLM 排行榜，比 academic benchmark 更接近用户体验
- 推荐继续读本教程的 **5.2 节 GQA / MLA**（KV cache 压缩详解，本节多次引用）
- 推荐继续读本教程的 **5.4 节 MoE**（DeepSeek-MoE / Mixtral 详解，本节 MoE 部分基础）
- 推荐继续读本教程的 **6.3 节 Scaling Law**（理解为什么 LLaMA-3 / Qwen2.5 都 over-train）
- 推荐继续读本教程的 **10.3 节 RLVR 与 DeepSeek-R1**（R1 训练细节，本节只点到为止）
