---
title: "学习地图"
---

> 这是本教程的**spec**与**学习地图**。每一节的标题、依赖关系、覆盖知识点都在这里。

## 设计原则

1. **理论实现为主线、API 应用为辅** — 主框架按 from-scratch 路线（Karpathy / Raschka / CS336），FSDL 风格的"先 ship 再深入"做成 Module 0 的 5 分钟 demo
2. **infra 必须独立成块** — Module 7 用 6 节专门讲并行 / 混合精度 / kernel
3. **pretrain → SFT → RLHF 严格线性顺序** — 不为赶 Agent 热点把 RLHF 提前；Agent RL 必须建立在 RLHF 心智模型上
4. **应用层后置** — Prompt / RAG / Agent 在 Module 13-15
5. **每节统一模板**：`一句话 → mental model → 公式 → 代码 → 工程经验 → 必读 paper → 面试题`，固定结构降低读者认知负担
6. **不写**：业务落地 case study、Diffusion / 图像生成

## 节标准模板

```markdown
# X.Y 节标题

> ⏱ 预计阅读 N 分钟 ｜ 难度 ★★☆ ｜ 前置：X.Z

## 一句话本节讲什么
## 1. Mental model（直觉）
## 2. 公式与原理
## 3. 最小代码示例（PyTorch / HF / vLLM 任一）
## 4. 工程踩坑与经验
## 5. 经典 paper（2-3 篇，注明价值）
## 6. 自测与面试题（3 题，附答案 sketch）
## 7. 延伸阅读
```

## 难度与必考标记

- ★ 入门 / ★★ 进阶 / ★★★ 高阶
- 🔥 = 算法岗强必考点（面试高频出现）

---

## Module 0 — 引言与学习地图（1 节）

| # | 标题 | 难度 |
|---|------|------|
| 0.1 | LLM/Agent 算法工程师在做什么 + 5 分钟跑通 demo + 学习地图 | ★ |

## Module 1 — DL 基础速通（5 节）

| # | 标题 | 难度 |
|---|------|------|
| 1.1 | 神经网络与反向传播（手推 + autograd 心智模型） | ★ |
| 1.2 | 优化器：SGD / Momentum / Adam / AdamW + LR schedule 🔥 | ★★ |
| 1.3 | 正则化与归一化：Dropout / BN / LN / RMSNorm 🔥 | ★★ |
| 1.4 | 损失函数：CE / KL / 对比学习 | ★ |
| 1.5 | PyTorch 工作流与显存 / dtype 心智模型 | ★★ |

## Module 2 — NLP 任务全景与传统语言模型（4 节）

| # | 标题 | 难度 |
|---|------|------|
| 2.1 | NLP 任务全景：分类 / 序列标注 / Seq2Seq / QA | ★ |
| 2.2 | 词向量：word2vec / GloVe / fastText | ★ |
| 2.3 | RNN / LSTM / GRU | ★★ |
| 2.4 | Seq2Seq + Bahdanau Attention（动机：为什么需要 Transformer） | ★★ |

## Module 3 — Tokenization 与数据管线（3 节）

| # | 标题 | 难度 |
|---|------|------|
| 3.1 | BPE / WordPiece / Unigram / SentencePiece 🔥 | ★★ |
| 3.2 | 词表构造、特殊 token、OOV | ★ |
| 3.3 | Batching / packing / 注意力 mask | ★★ |

## Module 4 — Transformer 架构 from scratch（7 节）🔥 整章必考

| # | 标题 | 难度 |
|---|------|------|
| 4.1 | Self-attention：QKV 与 scaled dot-product 🔥 | ★★ |
| 4.2 | Multi-head attention 与 head 的物理意义 🔥 | ★★ |
| 4.3 | 位置编码：绝对 / 相对 / RoPE / ALiBi 🔥 | ★★★ |
| 4.4 | LayerNorm / RMSNorm 与 Pre-LN vs Post-LN | ★★ |
| 4.5 | FFN 与激活：ReLU / GELU / SwiGLU | ★★ |
| 4.6 | 完整 decoder-only 实现（手撕 nanoGPT） 🔥 | ★★ |
| 4.7 | KV Cache 原理与实现 🔥 | ★★★ |

## Module 5 — 现代 LLM 架构变体（5 节）

| # | 标题 | 难度 |
|---|------|------|
| 5.1 | Encoder-only / Decoder-only / Encoder-Decoder 对比 | ★★ |
| 5.2 | GQA / MQA / MLA：KV cache 压缩 🔥 | ★★★ |
| 5.3 | FlashAttention 1/2/3：IO-aware kernel 🔥 | ★★★ |
| 5.4 | MoE：Mixtral / DeepSeek-MoE 路由与负载均衡 🔥 | ★★★ |
| 5.5 | SSM 路线：Mamba / Mamba-2 与 Transformer 对比 | ★★★ |

## Module 6 — 预训练（6 节）

| # | 标题 | 难度 |
|---|------|------|
| 6.1 | 训练目标：CLM / MLM / Prefix-LM / FIM | ★★ |
| 6.2 | 数据管线：FineWeb / Dolma / DCLM / 清洗去重配比 🔥 | ★★ |
| 6.3 | Scaling Law：Kaplan / Chinchilla / DeepSeek + over-training 反思 🔥 | ★★★ |
| 6.4 | 训练曲线诊断与超参（lr / batch / warmup） | ★★ |
| 6.5 | Long-context：Position Interpolation / NTK / YaRN / LongRoPE 🔥 | ★★★ |
| 6.6 | 经典开源 LLM 解读：LLaMA / Qwen / DeepSeek 系列 | ★★ |

## Module 7 — 训练 Infra（6 节）🔥 独立 module

| # | 标题 | 难度 |
|---|------|------|
| 7.1 | 数据并行：DDP / ZeRO 1/2/3 / FSDP 🔥 | ★★★ |
| 7.2 | 模型并行：TP（Megatron）/ PP（1F1B、Zero-Bubble） 🔥 | ★★★ |
| 7.3 | Sequence / Context / Expert Parallelism | ★★★ |
| 7.4 | 混合精度：fp16 / bf16 / fp8 + Loss Scaling | ★★ |
| 7.5 | 显存优化：Activation Recomputation / Selective / Offload 🔥 | ★★★ |
| 7.6 | Triton / CUDA kernel 入门 + FlashAttention 走读 | ★★★ |

## Module 8 — 后训练 I：SFT 与 PEFT（5 节）

| # | 标题 | 难度 |
|---|------|------|
| 8.1 | SFT 数据构造：Self-Instruct / Magpie / OpenHermes / Evol-Instruct 🔥 | ★★ |
| 8.2 | SFT 训练细节：chat template / loss mask / sample packing 🔥 | ★★ |
| 8.3 | LoRA / QLoRA / DoRA 原理与实现 🔥 | ★★★ |
| 8.4 | Adapter / Prefix-tuning / P-tuning v2 | ★★ |
| 8.5 | SFT 实战：多轮对话 + tool 混合训练 | ★★★ |

## Module 9 — 后训练 II：偏好优化与 RLHF（7 节）🔥 全章必考

| # | 标题 | 难度 |
|---|------|------|
| 9.1 | RL 速通：policy / value / advantage / return / GAE | ★★★ |
| 9.2 | Reward Model：Bradley-Terry + 训练实操 🔥 | ★★ |
| 9.3 | PPO 原理与在 LLM 上的形式（KL 约束、4 模型显存） 🔥 | ★★★ |
| 9.4 | DPO 闭式解推导 + 变体（IPO / KTO / SimPO / ORPO / RLOO） 🔥 | ★★★ |
| 9.5 | GRPO：去 critic 的 group-relative advantage 🔥 | ★★★ |
| 9.6 | 工程踩坑：reward hacking / RM 漂移 / KL 坍塌 / length bias 🔥 | ★★★ |
| 9.7 | RLAIF / Constitutional AI / Self-Rewarding LM | ★★ |

## Module 10 — Reasoning 与 Test-time Scaling（4 节）

| # | 标题 | 难度 |
|---|------|------|
| 10.1 | CoT / Self-Consistency / ToT / GoT / Reflexion | ★★ |
| 10.2 | PRM 与 Process Reward（Lightman "Let's Verify"） | ★★★ |
| 10.3 | RLVR 与 DeepSeek-R1：纯 RL 激发 long-CoT 🔥 | ★★★ |
| 10.4 | 推理时搜索：Best-of-N / MCTS / Verifier-guided | ★★★ |

## Module 11 — 推理引擎与部署（5 节）

| # | 标题 | 难度 |
|---|------|------|
| 11.1 | 推理性能指标：TTFT / TBT / throughput / 容量 | ★★ |
| 11.2 | PagedAttention 与 Continuous Batching（vLLM） 🔥 | ★★★ |
| 11.3 | RadixAttention 与 Prefix Cache（SGLang） | ★★★ |
| 11.4 | 量化：GPTQ / AWQ / GGUF / FP8 / SmoothQuant 🔥 | ★★★ |
| 11.5 | 投机解码：Speculative / Medusa / EAGLE 🔥 | ★★★ |

## Module 12 — 评测、可观测性、LLMOps（4 节）

| # | 标题 | 难度 |
|---|------|------|
| 12.1 | 通用评测：MMLU / GSM8K / HumanEval / IFEval / Arena 🔥 | ★★ |
| 12.2 | LLM-as-Judge / Pairwise / Reward 评测 | ★★ |
| 12.3 | 安全评测与红队：HarmBench / GCG / jailbreak 谱系 | ★★★ |
| 12.4 | 在线监控、回归测试、prompt / 数据版本管理 | ★★ |

## Module 13 — Prompting / RAG / 工具增强（4 节）

| # | 标题 | 难度 |
|---|------|------|
| 13.1 | Prompt 工程：system / few-shot / CoT / role / 结构化输出 | ★★ |
| 13.2 | RAG 基础：embedding / retriever / reranker / chunking 🔥 | ★★ |
| 13.3 | 进阶 RAG：HyDE / RAG-Fusion / Self-RAG / GraphRAG | ★★★ |
| 13.4 | Function calling 工程：JSON schema / 并行调用 / constrained decoding 🔥 | ★★ |

## Module 14 — Agent 系统（7 节）🔥 Agent 章节核心

| # | 标题 | 难度 |
|---|------|------|
| 14.1 | Agent 范式概览：从 ReAct 到 Reasoning Agent | ★★ |
| 14.2 | 实现一个最小 ReAct + Reflection agent 🔥 | ★★ |
| 14.3 | Tool Use 训练：Toolformer / Gorilla / ToolLLaMA / xLAM 🔥 | ★★★ |
| 14.4 | Planning：ToT / LATS / Plan-and-Solve / Code-as-Policy | ★★★ |
| 14.5 | Memory：scratchpad / vector DB / MemGPT / Generative Agents | ★★ |
| 14.6 | Multi-agent：CAMEL / AutoGen / MetaGPT / orchestrator-worker | ★★★ |
| 14.7 | Agent Framework 对比：LangGraph / OpenAI Assistants / Anthropic | ★★ |

## Module 15 — Agent RL 与多轮鲁棒性（5 节）🔥 前沿

| # | 标题 | 难度 |
|---|------|------|
| 15.1 | Agent SFT：FireAct / Agent-FLAN / AgentTuning / ToRA | ★★★ |
| 15.2 | 多轮 PPO/GRPO：trajectory-level reward 与归因 🔥 | ★★★ |
| 15.3 | 真实环境 RL：SWE-Gym / WebArena / OSWorld / WebGPT | ★★★ |
| 15.4 | Reasoning + Agent：Search-R1 / ReSearch / ReTool / Agent-R1 🔥 | ★★★ |
| 15.5 | Agent 鲁棒性：observation perturbation / tool failure / recovery | ★★★ |

## Module 16 — 多模态、Embedding、Computer Use（5 节）

| # | 标题 | 难度 |
|---|------|------|
| 16.1 | VLM：CLIP / LLaVA / Qwen-VL / InternVL（vision encoder + projector + LLM） 🔥 | ★★★ |
| 16.2 | Native multimodal：Chameleon / GPT-4o / Gemini | ★★★ |
| 16.3 | 语音：Whisper / VALL-E / CosyVoice / 全双工 Moshi | ★★★ |
| 16.4 | Embedding：bge / E5 / Instructor / NV-Embed 🔥 | ★★ |
| 16.5 | Computer Use & GUI Agent：OSWorld / Anthropic / SeeClick / UI-TARS | ★★★ |

## 附录

| # | 标题 |
|---|------|
| A | Capstone 1：复现 GPT-2 124M（数据 → 训完 → 评测） |
| B | Capstone 2：SFT + LoRA + 一个 RAG demo |
| C | Capstone 3：Agent end-to-end（search + code interpreter + memory） |
| D | 数学速查（备用，给真·零基础读者） |

---

## 节数统计

- 主体：1 + 5 + 4 + 3 + 7 + 5 + 6 + 6 + 5 + 7 + 4 + 5 + 4 + 4 + 7 + 5 + 5 = **83 节**
- 附录：4 篇
- **合计：87 篇**
