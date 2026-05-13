---
title: "面试题索引"
---

> 87 节 × 3 题 = **261 道自测/面试题**，按知识点反向索引。校招 / 跳槽前刷一遍。
>
> **使用方式**：先扫"高频 Top 30"自测；再用"反向索引（按题型）"做专项；最后用"完整索引（按 Module）"扫盲。完整 answer sketch 在原节 §6。

---

## 0. 速查：高频面试 Top 30

精选 30 道最高频题，按 5 大方向分类。每题给出 (题干 1 句) + (节号)。

### 0.1 架构 / Transformer / Attention（6 题）

- **A1**：写出 scaled dot-product attention 公式 + 为什么除 $\sqrt{d_k}$（推导 $q \cdot k$ 方差为 $d_k$） — 节 4.1
- **A2**：写出 multi-head attention 公式 + 为什么 $d_k = d/h$ — 节 4.2
- **A3**：RoPE 内积仅依赖相对位置 $n-m$ 的推导（$R_m^\top R_n = R_{n-m}$） — 节 4.3
- **A4**：Pre-LN vs Post-LN 伪代码 + Post-LN 难训的根本原因 — 节 4.4
- **A5**：写出 SwiGLU FFN 公式 + 等参数下 $d_{ff}$ 怎么选 — 节 4.5
- **A6**：朴素 generate vs KV cache 复杂度差异（每 step / 总 generate） — 节 4.7

### 0.2 后训练 / RLHF / DPO / GRPO（8 题）

- **B1**：写 GAE 递推公式 + $\lambda \to 0/1$ 的物理意义 — 节 9.1
- **B2**：写 Bradley-Terry RM loss + 为什么用 sigmoid — 节 9.2
- **B3**：写 PPO clipped objective + 解释 clip + min 的物理意义（$\hat A$ 正负方向相反） — 节 9.3
- **B4**：PPO "4 模型"是哪 4 个（trainable / frozen） + 显存优化 — 节 9.3
- **B5**：从 RLHF $\max_\pi \mathbb{E}[r] - \beta D_{KL}$ 推 DPO loss 的关键 4 步（$\beta \log Z(x)$ 怎么消） — 节 9.4
- **B6**：写 GRPO advantage 公式（group baseline + std normalization） + 为什么不需 critic — 节 9.5
- **B7**：5 种 RLHF reward hacking + 各 1 个缓解方法 — 节 9.6
- **B8**：写出 R1 完整 4 阶段训练 pipeline + R1-Zero 与 R1 的关系 — 节 10.3

### 0.3 Infra / Parallelism / 量化（6 题）

- **C1**：70B 在 16×H100 上 DDP / ZeRO-2 / ZeRO-3 各占多少显存（weight + grad + optim） — 节 7.1
- **C2**：Megatron 风格 FFN 的 TP 切法（W₁ 列切 + W₂ 行切 + 1 次 all-reduce） — 节 7.2
- **C3**：fp16 vs bf16（动态范围 / 精度 / scaler / LLM 用谁） — 节 7.4
- **C4**：Korthikanti 公式算 LLaMA-2 13B activation 显存 + selective recompute — 节 7.5
- **C5**：PagedAttention 借鉴 OS 哪个机制 + 为什么显存利用率从 30% → 95% — 节 11.2
- **C6**：W4A16 / W8A8 / FP8 量化方案 + 各自硬件场景 — 节 11.4

### 0.4 Agent / 多轮 RL / Tool Use（5 题）

- **D1**：ReAct 核心 4 步循环 + 与 Plan-and-Execute 的差异 — 节 14.1
- **D2**：最小 ReAct loop 5 步伪代码（不用任何库） — 节 14.2
- **D3**：trajectory-level reward 如何 broadcast 到每 token 的 advantage（2 种主流方案） — 节 15.2
- **D4**：现代 Reasoning Agent 的 6-stage 训练 pipeline + 哪些科研可省、生产必做 — 节 15.4
- **D5**：multi-turn agent RL 中 length hacking / tool spam 怎么发生 + 缓解 — 节 15.2

### 0.5 评测 / 工程 / 推理优化 / RAG（5 题）

- **E1**：LLM 推理 prefill compute-bound vs decode memory-bound 的两句话解释 — 节 11.1
- **E2**：算 H100 跑 LLaMA-3 70B TP=8 的单 stream decode TPS 上限 — 节 11.1
- **E3**：标准 RAG 5 个阶段 + 每阶段主流工具 — 节 13.2
- **E4**：1M context LLM 已普及，为什么还要做 RAG（4 个维度） — 节 13.2
- **E5**：LLM-as-Judge 5 种典型 bias + 各 1 个缓解 — 节 12.2

---

## 1. 完整索引（按 Module 分组）

> 每节 3 题，1 句话题干。完整 answer sketch 在原节 §6 自测与面试题。

### Module 0 — Intro

- **0.1 Q1**：LLM 算法 vs 应用工程师的 3 层核心技能差异
- **0.1 Q2**：2024-2026 改变 LLM/Agent 范式的 3 个关键节点
- **0.1 Q3**：你打算怎么用本书（3 条具体学习习惯）

### Module 1 — DL Basics

- **1.1 Q1**：为什么 PyTorch `tensor.grad` 默认是累加而不是覆盖
- **1.1 Q2**：手推 $y = \sigma(Wx+b)$ 的 $\partial L / \partial W$ 形状与表达式
- **1.1 Q3**：PPO 多 epoch 重复 backward，用 `retain_graph=True` 还是别的方案
- **1.2 Q1**：写 Adam 完整更新式（含偏差修正）+ $\epsilon$ 的两个作用
- **1.2 Q2**：为什么 LLM 训练几乎必然要 warmup（≥ 2 个理由）
- **1.2 Q3**：7B step 200 后 NaN 的排查顺序（lr / β / wd / warmup / data / dtype）
- **1.3 Q1**：为什么现代 LLM 用 RMSNorm 而非 LayerNorm（≥ 2 点）
- **1.3 Q2**：写 RMSNorm 核心 4 行 PyTorch
- **1.3 Q3**：BN backbone fine-tune 后 eval 远高于 train，3 个原因
- **1.4 Q1**：CE 与 KL 的关系 + 何时梯度等价
- **1.4 Q2**：写 PyTorch 算 $D_{KL}(\pi_s \| \pi_t)$（注意 log_softmax / softmax 顺序）
- **1.4 Q3**：PPO loss 里 $\beta \cdot D_{KL}$ 设大/设小的后果 + 为什么必须有
- **1.5 Q1**：bf16 + AdamW 训 7B（仅 weight + grad + optim）显存计算
- **1.5 Q2**：fp16 vs bf16，为什么 LLM 一律用 bf16（≥ 2 点）
- **1.5 Q3**：step 1000 NaN，从 dtype / lr / data / scaler 各给一个排查动作

### Module 2 — NLP Landscape

- **2.1 Q1**：BERT 接 task head vs GPT-3 直接 prompt：两种范式优缺点
- **2.1 Q2**：金融欺诈新闻判别，3 种解法（fine-tune / prompt / agent）的 cost / latency / acc / 可解释性对比
- **2.1 Q3**：为什么生成式 QA 不能用 EM / F1（≥ 2 个原因 + 替代方案）
- **2.2 Q1**：one-hot vs distributed representation 本质差异 + 为什么后者支持向量算术
- **2.2 Q2**：写 Skip-gram + Negative Sampling 的 loss 公式 + 比 full softmax 快多少
- **2.2 Q3**：现代 LLM 不用预训 word2vec / GloVe / fastText 的 ≥ 2 条原因
- **2.3 Q1**：写 LSTM 4 个门公式 + 一句话解释 cell state 为何缓解梯度消失
- **2.3 Q2**：LSTM vs GRU vs Transformer 的工程选型（≥ 3 个判断标准）
- **2.3 Q3**：Mamba / SSM 是不是 RNN 的回归
- **2.4 Q1**：写 Bahdanau attention 三步公式 + 解释信息瓶颈
- **2.4 Q2**：从 Bahdanau additive → Luong multiplicative → self-attention 每步变了什么
- **2.4 Q3**：beam search 在 LLM 时代被 sampling 取代的 ≥ 2 个原因

### Module 3 — Tokenization

- **3.1 Q1**：写 BPE 训练算法 5 步伪代码 + 用 corpus 演示一轮合并
- **3.1 Q2**：BPE / WordPiece / Unigram 4 维度对比
- **3.1 Q3**：用 LLaMA tokenizer 训中文 LLM 推理慢效果差，诊断 + 解决
- **3.2 Q1**：byte-level BPE 怎么消灭 OOV（以 emoji "🐶" 为例）
- **3.2 Q2**：base model 加 5 个 tool special token 到能 SFT 的完整步骤
- **3.2 Q3**：词表 32k vs 256k 的 4 维度对比
- **3.3 Q1**：padding mask vs causal mask 物理意义；decoder-only LLM 训练需哪些 mask
- **3.3 Q2**：sample packing 的 block-diagonal mask 构造（B=1, T=10, [s1=4, s2=6]）
- **3.3 Q3**：padding vs packing vs padding-free training 的取舍

### Module 4 — Transformer from Scratch

- **4.1 Q1**：写 scaled dot-product attention 完整公式 + 推导 $\sqrt{d_k}$
- **4.1 Q2**：用 PyTorch 写 single-head causal self-attention（≤ 15 行）
- **4.1 Q3**：self-attention vs RNN 三维对比；$n=128k$ 时的应对方案
- **4.2 Q1**：写 multi-head attention 公式 + 为什么 $d_k = d/h$（参数等价性）
- **4.2 Q2**：用 3 个 `nn.Linear(d,d)` 实现 multi-head Q/K/V + reshape/permute
- **4.2 Q3**：head 数 32 vs 8 优缺点；KV cache 爆的应对方案
- **4.3 Q1**：RoPE 内积 $\langle R_m q, R_n k \rangle$ 仅依赖 $n-m$ 的推导
- **4.3 Q2**：写 `apply_rotary(q,k,cos,sin)` 核心 4 行（rotate_half 技巧）
- **4.3 Q3**：RoPE vs ALiBi 4 维度对比；为什么现代 LLM 几乎全选 RoPE
- **4.4 Q1**：Pre-LN / Post-LN 伪代码 + Post-LN 难训根本原因
- **4.4 Q2**：开源 LLM 训练几 step 就 NaN，从 norm/lr/warmup/dtype 各给排查
- **4.4 Q3**：QK-Norm 解决什么问题；为什么 q/k 比 v 更需要 norm
- **4.5 Q1**：写 SwiGLU FFN 完整公式（含 $W_1, W_2, W_3$ 维度）+ 等参数下 $d_{ff}$
- **4.5 Q2**：用 PyTorch 写 LLaMA 风格 FFN（≤ 8 行），$d=4096$ 的 $d_{ff}$ 取值
- **4.5 Q3**：SwiGLU 慢且大，LLaMA / Qwen / Mistral 仍全用的 ≥ 2 理由
- **4.6 Q1**：画 decoder-only 完整数据流（input tokens → logits）+ LLaMA-style 配方
- **4.6 Q2**：用 $P \approx Vd + 12Ld^2$ 估算 LLaMA-2 13B 参数 + 误差从何来
- **4.6 Q3**：用 ≤ 30 行 PyTorch 写 LLaMA-style Block（Pre-RMSNorm + SDPA + SwiGLU + 2 residual）
- **4.7 Q1**：朴素 generate vs KV cache 复杂度（每 step / 总 generate）
- **4.7 Q2**：算 LLaMA-3 70B 在 T=32k 下 MHA / GQA-8 KV cache 显存
- **4.7 Q3**：70B batch=2 OOM，3 个优化方向（按性价比排序）+ trade-off

### Module 5 — Modern LLM Architectures

- **5.1 Q1**：BERT / GPT / T5 在 (mask / 训练目标 / 推理模式) 三维差异
- **5.1 Q2**：为什么 decoder-only 几乎垄断 LLM（≥ 3 个原因）
- **5.1 Q3**：encoder-only 在 LLM 时代最大用途；ModernBERT 为何仍有意义
- **5.2 Q1**：写 GQA vs MHA 参数对比 + 算 LLaMA-3 70B GQA KV cache
- **5.2 Q2**：用 PyTorch 写最小 GQA forward（≤ 15 行，含 K/V g→h_q expand）
- **5.2 Q3**：MLA 比 GQA 工程复杂得多，DeepSeek 为何坚持用（≥ 2 理由）
- **5.3 Q1**：为什么标准 attention 是 memory-bound；FlashAttention 怎么解决
- **5.3 Q2**：写 online softmax 关键 rescaling 公式（块 1 → 块 2 更新）
- **5.3 Q3**：开 FlashAttention 但 step time 与 baseline 一样慢，排查清单
- **5.4 Q1**：算 Mixtral 8x22B 总参数 vs 激活参数
- **5.4 Q2**：用 PyTorch 写 MoE router + top-K + GShard aux loss（5-8 行）
- **5.4 Q3**：DeepSeek-MoE 256 小 expert vs Mixtral 8 大 expert；DeepSeek-V3 为何用 bias-based balancing
- **5.5 Q1**：写 discrete SSM 递推公式 + Mamba 的 selective vs S4
- **5.5 Q2**：Mamba vs Transformer 在 4 维度对比
- **5.5 Q3**：为什么 2024-2026 主流仍是 Transformer（≥ 3 原因）+ Mamba 不可替代场景

### Module 6 — Pretraining

- **6.1 Q1**：写 CLM 与 MLM loss 公式 + 为什么 CLM 训练效率高
- **6.1 Q2**：写 FIM transform 的核心 5 行 Python（PSM 重排）
- **6.1 Q3**：为什么现代 LLM 几乎全用 CLM（≥ 3 原因）
- **6.2 Q1**：列预训数据 pipeline 5 阶段 + 每阶段关键工具
- **6.2 Q2**：MinHash + LSH 怎么找近似重复文档（5 步）+ 为什么 unbiased
- **6.2 Q3**：DCLM-baseline 3.8T 比 RedPajama 30T 高 5-8 分，"少而精 > 多而杂" 为什么成立
- **6.3 Q1**：写 Chinchilla compute-optimal 配比 + 算 LLaMA-2 7B 最优 token 数
- **6.3 Q2**：为什么 LLaMA-3 8B 用 15T（远超 Chinchilla），从 deploy 视角解释
- **6.3 Q3**：emergent ability 真实 vs metric 错觉
- **6.4 Q1**：7B step 3000 NaN 的排查顺序（lr / data / dtype / scheduler / scaler）
- **6.4 Q2**：LLaMA-2 70B lr 比 7B 小一半，为什么（≥ 2 理由）
- **6.4 Q3**：接手 30 天预训任务，判断"训练健康度"看哪 5 个指标
- **6.5 Q1**：写 PI 与 NTK-aware 核心改动公式 + 为什么 NTK 比 PI 好
- **6.5 Q2**：8k → 128k context 扩展的完整 4 阶段 pipeline
- **6.5 Q3**：Needle test 满分但 multi-hop 任务差，3 个原因
- **6.6 Q1**：LLaMA-3 8B / Qwen2.5 7B / DeepSeek-V3 在 6 维度差异
- **6.6 Q2**：从 LLaMA-1 → DeepSeek-V3 的关键技术演化（≥ 4 条）
- **6.6 Q3**：中文 SFT 项目从 LLaMA-3 8B / Qwen2.5 7B / DeepSeek-V2-Lite 选 base 的理由

### Module 7 — Training Infra

- **7.1 Q1**：70B 在 16×H100 上 DDP / ZeRO-2 / ZeRO-3 各自单卡 weight+grad+optim
- **7.1 Q2**：ZeRO-3 / FSDP 相对 DDP 通信开销多多少 + 何时反而不值得用
- **7.1 Q3**：1×H100 80G 上 7B SFT 装不下，≥ 3 个可行解决方向
- **7.2 Q1**：Megatron FFN 的 TP 切法 + 为什么只需 1 次 all-reduce
- **7.2 Q2**：算 GPipe / 1F1B / Interleaved($v=4$) / Zero-Bubble 在 $p=8, m=16$ 的 bubble ratio
- **7.2 Q3**：LLaMA-3 405B 在 4096 GPU 上的 (DP, TP, PP) 配置 + mesh 拓扑
- **7.3 Q1**：SP / CP / EP 各切什么维度；为什么 SP 必须与 TP 共用 group
- **7.3 Q2**：Ring Attention 通信量；与 TP attention 谁更适合 long-context
- **7.3 Q3**：1024 GPU 训 DeepSeek-V3-style MoE 的 5D 配置
- **7.4 Q1**：fp16 vs bf16 在 4 维度对比
- **7.4 Q2**：fp8 训练为什么用 E4M3 + E5M2 两种格式
- **7.4 Q3**：13B + bf16 + ZeRO-2 仍 OOM，≥ 3 个 dtype 优化方向
- **7.5 Q1**：用 Korthikanti 公式算 LLaMA-2 13B activation 显存
- **7.5 Q2**：Selective Recomputation 比 Full 好在哪；Megatron 为何默认 selective
- **7.5 Q3**：8×A100 80G fine-tune 70B，叠加 ≥ 4 种显存优化技术
- **7.6 Q1**：Triton vs CUDA 关系；为什么 LLM 工程师更应该学 Triton
- **7.6 Q2**：写 Triton vector add kernel 核心 5 行（pid / mask / load+store）
- **7.6 Q3**：FlashAttention v2 把 outer loop 从 K/V 换成 Q，为什么更快

### Module 8 — Post-Training: SFT / PEFT

- **8.1 Q1**：列 SFT 数据 5 大来源（人工 / Self-Instruct / Magpie / Evol / 蒸馏）+ 各 1 代表数据集 + 局限
- **8.1 Q2**：中文 chat model SFT 1M 数据完整 pipeline
- **8.1 Q3**：LIMA 1k 训出 strong model，工业为何用 100k+（≥ 3 原因 + 边界）
- **8.2 Q1**：写 SFT 训练构造 labels 的核心 5 行（含 -100 mask、ChatML 多轮）
- **8.2 Q2**：sample packing vs padding 的 throughput 收益；何时不该用
- **8.2 Q3**：7B SFT eval loss 在 epoch 1 末就上升的 3 个原因
- **8.3 Q1**：写 LoRA 公式 $W' = W + (\alpha/r) BA$ + 为什么 $\alpha$ 除 $r$
- **8.3 Q2**：24GB 4090 上 SFT LLaMA-3 8B 的 QLoRA 完整配置
- **8.3 Q3**：QLoRA 比 LoRA 显存省 4× 但慢 30%，为什么慢；何时不该用
- **8.4 Q1**：Adapter / Prefix-tuning / LoRA 的核心差异；为何 LoRA 成事实标准
- **8.4 Q2**：写 Houlsby Adapter 核心 forward 公式 + 5 行 PyTorch
- **8.4 Q3**：什么场景选 Adapter / Prefix-tuning 而非 LoRA
- **8.5 Q1**：1M 通用 + 100k xLAM 数据，到能开训的 mix + format 步骤
- **8.5 Q2**：multi-turn (system / user / assistant-tool / tool_obs / final-answer) 序列的 loss mask 标注
- **8.5 Q3**：BFCL tool calling 80% → 90%+，用 RL 还是更多 SFT（判断标准）

### Module 9 — Post-Training: RLHF

- **9.1 Q1**：1-2 句解释 advantage + 为什么用它代替 return
- **9.1 Q2**：写 GAE 递推公式 + $\lambda \to 0/1$ 的物理意义
- **9.1 Q3**：LLM RLHF 的 state / action / reward 映射（GSM8K Janet 鸭子题为例）
- **9.2 Q1**：写 Bradley-Terry RM loss + 为什么用 sigmoid
- **9.2 Q2**：RM val acc 72% 但 PPO 后 policy 越来越长（chosen 永远比 rejected 长），≥ 3 个解决方向
- **9.2 Q3**：DPO 不需 RM，为什么大公司还训 RM（≥ 3 个不可替代场景）
- **9.3 Q1**：写 PPO clipped objective 完整公式 + 解释 clip + min（$\hat A$ 正负方向）
- **9.3 Q2**：PPO 的 4 模型（trainable / frozen + 为什么 + 显存节省）
- **9.3 Q3**：7B RLHF 200 步 RM reward +30% 但人工评测大幅下降，≥ 3 个原因 + 修复
- **9.4 Q1**：从 $\max \mathbb{E}[r] - \beta D_{KL}$ 推 DPO loss 关键 4 步（$\beta \log Z(x)$ 怎么消）
- **9.4 Q2**：DPO vs SimPO 主要差异；SimPO 去 ref 的代价
- **9.4 Q3**：10 万 pairwise + 24GB 卡上 DPO / SimPO / KTO 三方案的具体配置
- **9.5 Q1**：写 GRPO advantage 公式（group baseline + std normalization）+ 为什么不需 critic
- **9.5 Q2**：GRPO 比 PPO 慢 G 倍仍成主流的 ≥ 3 理由
- **9.5 Q3**：DAPO 在 GRPO 上改了什么；为什么更稳
- **9.6 Q1**：5 种 RLHF reward hacking + 各 1 缓解
- **9.6 Q2**：RM reward +30% 但 MT-Bench -5 分的 3 个排查 + 修复 step
- **9.6 Q3**：GRPO + verifier 训 reasoning model 时 length hacking 怎么发生 + 缓解
- **9.7 Q1**：RLHF / RLAIF / Constitutional AI / Self-Rewarding LM 的 preference 信号来源差异
- **9.7 Q2**：5 万美金 GPT-4 RLAIF 100k 标注的成本计算 + ≥ 3 个风险
- **9.7 Q3**：Self-Rewarding LM vs RLVR 的 ≥ 2 个判断标准

### Module 10 — Reasoning / Test-Time Scaling

- **10.1 Q1**：CoT / Self-Consistency / ToT / Reflexion 在 (cost / 任务 / verifier) 三维对比
- **10.1 Q2**：7B GSM8K 50%，不重训权重的 3 种 inference-time scaling 方法
- **10.1 Q3**：R1 / o1 出现后，prompt-time 方法哪些 obsoleted、哪些仍重要
- **10.2 Q1**：ORM vs PRM 4 维度对比 + 决策建议
- **10.2 Q2**：用 Skywork-PRM 做 PRM-guided BoN 的完整流程 + ≥ 2 个工程坑
- **10.2 Q3**：DeepSeek-R1 不用 PRM 的 ≥ 3 理由；PRM 与 verifier 适用场景
- **10.3 Q1**：写 R1 完整 4 阶段训练 pipeline + R1-Zero 与 R1 关系
- **10.3 Q2**：RLVR 用 verifier 而非 RM 的 ≥ 3 理由 + verifier 不适用场景
- **10.3 Q3**："Aha moment" 的本质；纯 RL 涌现 reasoning 对 < 7B 模型成立吗
- **10.4 Q1**：用 N 维 trade-off 串 Greedy / BoN / Self-Consistency / ToT / MCTS 5 种
- **10.4 Q2**：math reasoning 中 BoN N=32 / MCTS / R1-distill BoN N=8 的选型
- **10.4 Q3**：R1 / o1 已内化 search-like reasoning，外部 search 是否仍必要

### Module 11 — Inference Engines

- **11.1 Q1**：用两句话解释 prefill compute-bound / decode memory-bound
- **11.1 Q2**：算 H100 跑 LLaMA-3 70B TP=8 的单 stream decode TPS 上限
- **11.1 Q3**：70B + 8×H100 + GQA-8 服务 P99 TTFT 1.5s（SLO 500ms），3 个优化方向
- **11.2 Q1**：PagedAttention 借鉴 OS 哪个机制；为什么显存利用率 30% → 95%
- **11.2 Q2**：算 LLaMA-3 70B + PagedAttention + max_model_len=4096 在 H100 80G 能服务多少 sample
- **11.2 Q3**：vLLM P99 偶发 5+ 秒，从 (chunked prefill / 量化 / TP / batch) 各给优化
- **11.3 Q1**：RadixAttention 的 trie 怎么管 KV cache；为何 branch 场景比 hash 优
- **11.3 Q2**：3 个 prefix cache 收益巨大场景 + 1 个收益 0 场景
- **11.3 Q3**：agent 平台 vLLM vs SGLang 选哪个（≥ 4 个理由）
- **11.4 Q1**：W4A16 / W8A8 / FP8 三种量化方案 + 各自硬件场景
- **11.4 Q2**：LLaMA-3 70B 部署到 4×A100 80G 的量化 + 推理框架完整选型
- **11.4 Q3**：AWQ vs GPTQ 都是 W4A16，AWQ "activation-aware" 在哪；实战差异
- **11.5 Q1**：写 spec decoding rejection sampling 算法 + 证明输出分布 lossless
- **11.5 Q2**：Spec Decoding / Medusa / EAGLE 核心差异；EAGLE-3 为何能 3-5×
- **11.5 Q3**：LLaMA-3 70B 单 stream chat 的完整 spec decoding 配置（draft / num_spec / 何时关 / 监控）

### Module 12 — Evaluation / LLMOps

- **12.1 Q1**：现代 LLM 评测 5 大维度 + 各 1 代表 benchmark + 评测形式
- **12.1 Q2**：怀疑某模型 MMLU 89 分有水分，3 种 verify 方法
- **12.1 Q3**：通用 7B chat 助手必跑 ≥ 4 个 benchmark + 评测什么
- **12.2 Q1**：写 pairwise judge prompt（消 position bias + length bias + 显式维度）
- **12.2 Q2**：LLM-as-Judge 的 5 种典型 bias + 各 1 缓解
- **12.2 Q3**：新 7B chat 模型 vs baseline 的完整评测设计（数据 / judge / 样本 / 显著性 / 报告）
- **12.3 Q1**：5 类 LLM safety 风险 + 各 1 benchmark；为何只看 jailbench 不够
- **12.3 Q2**：GCG 的优化目标；为什么 suffix 能 transfer；perplexity filter 能完全防住吗
- **12.3 Q3**：公众 chat LLM 的 input / training / output 三层防御 + Agent 还要加什么
- **12.4 Q1**：LLM 客服系统 ≥ 8 个必须监控指标 + 告警阈值依据
- **12.4 Q2**：prompt v1 → v2 升级的完整 A/B 实验设计 + 决策标准
- **12.4 Q3**：OpenAI API down 1 小时，graceful degrade 怎么做

### Module 13 — Prompting / RAG / Tools

- **13.1 Q1**：写完整 system + few-shot + CoT prompt 解 GSM8K（含 2 example、固定输出格式）
- **13.1 Q2**：让 LLM 输 JSON 的 4 种方法（prompt / function call / constrained / retry）优缺点
- **13.1 Q3**：reasoning model 出现后，prompt 工程哪些不再必要 / 哪些更重要
- **13.2 Q1**：标准 RAG 5 个阶段 + 每阶段主流工具
- **13.2 Q2**：1M context 已普及，为什么还要做 RAG（≥ 4 维度）
- **13.2 Q3**：100 万工单 + 中英文混合 + P99 < 2s + QPS 500 的企业客服 RAG 完整选型
- **13.3 Q1**：HyDE / RAG-Fusion / Self-RAG / GraphRAG 各自痛点 + 决策树
- **13.3 Q2**：50 万 page 法律咨询 RAG（high factuality + 跨 doc）完整方案
- **13.3 Q3**：Agentic RAG vs Search-R1 的差异 + 选型场景
- **13.4 Q1**：写 `get_user_info(user_id, fields)` 的完整 OpenAI tool schema
- **13.4 Q2**：model 经常输 `{"city": "Beijing"`（截断 JSON）的 3 个解决方向
- **13.4 Q3**：constrained decoding 100% 合法但 5-20% latency 开销，何时开 / 关

### Module 14 — Agent Systems

- **14.1 Q1**：列 2022-2026 Agent 范式 5 个关键节点 + 范式贡献
- **14.1 Q2**：写 ReAct 核心 4 步循环 + 与 Plan-and-Execute 差异
- **14.1 Q3**：R1 后 reasoning agent vs 传统 ReAct 的核心差异
- **14.2 Q1**：写最小 ReAct loop 5 步伪代码（不用任何库）
- **14.2 Q2**：ReAct agent 经常 hallucinate 不存在 tool 名（如 web_search），3 个修复方向
- **14.2 Q3**：Reasoning model 出现后，外部 ReAct loop 还有意义吗
- **14.3 Q1**：Toolformer / Gorilla / ToolLLaMA / ToolACE 核心创新 + 演化原因
- **14.3 Q2**：100 个内部 API agent，base Qwen2.5-7B，从 0 到 production 的完整 pipeline
- **14.3 Q3**：什么时候 SFT 够、什么时候必须 RL（判断标准）
- **14.4 Q1**：Plan-and-Solve / ToT / LATS / Code-as-Policy 适用任务 + 决策树
- **14.4 Q2**：SWE agent (修 issue + 跑测试 + 提 PR) 用 LATS 还是 reasoning + ReAct
- **14.4 Q3**：reasoning model 已内嵌 plan + tool + reflection，LATS 在 2025 还有意义吗
- **14.5 Q1**：列 agent memory 5 大类型 + 各 1 代表实现 / 论文
- **14.5 Q2**：写 Generative Agents memory retrieval 综合 score 公式 + 三因子作用
- **14.5 Q3**：personal AI assistant 跨 6 个月记忆，memory 系统设计（架构 / 写入 / 检索 / aging / 隐私 / cost）
- **14.6 Q1**：multi-agent 5 大协作范式 + 各 1 代表框架
- **14.6 Q2**：软件开发 agent 选 MetaGPT 还是 orchestrator-worker
- **14.6 Q3**：multi-agent 不一定打过 single + good tool，价值在哪
- **14.7 Q1**：5 个 agent framework 核心定位 + 适用场景
- **14.7 Q2**：手写 < 200 行 ReAct vs LangChain / LangGraph，何时选哪个
- **14.7 Q3**：Anthropic《Building Effective Agents》"不要过度工程"，simple workflow vs full agent 的选择

### Module 15 — Agent RL

- **15.1 Q1**：Agent SFT 数据 vs 通用 chat SFT 的 3 个核心差别
- **15.1 Q2**：训 ReAct agent 的 SFT 数据合成 + 训练完整流程
- **15.1 Q3**：Agent SFT 后做 Agent RL 的判断标准
- **15.2 Q1**：trajectory-level reward 怎么 broadcast 到每 token advantage（2 种主流方案）
- **15.2 Q2**：训 search agent，base Qwen2.5-7B-Instruct，从 env + reward 到 GRPO update 的完整 pipeline
- **15.2 Q3**：multi-turn agent RL 中 length hacking / tool spam 怎么发生 + 缓解
- **15.3 Q1**：WebArena / SWE-Gym / OSWorld 三个真实 env 的核心差异
- **15.3 Q2**：真实 env 上做 RL 的 5 个工程挑战 + 各 1 解决方向
- **15.3 Q3**：SWE-RL 的核心创新；为什么真实 env RL 是 SWE agent 关键
- **15.4 Q1**：写现代 Reasoning Agent 的 6-stage 训练 pipeline + 哪些科研可省 / 生产必做
- **15.4 Q2**：Reasoning Agent reward 4 项（acc + format - length - tool cost）每项作用 + 只用 acc 会怎样
- **15.4 Q3**：Search-R1 / ReSearch / ReTool / Agent-R1 中挑 2 个对比 + 与 SWE-RL 关系
- **15.5 Q1**：列 agent 真实部署 5 种 robustness 失效场景 + 各 1 缓解方法
- **15.5 Q2**：怎么用 adversarial training 提升 robustness（data aug + reward）
- **15.5 Q3**：multi-turn agent RL 中 noisy obs 的 advantage 怎么处理（2025-2026 热点）

### Module 16 — Multimodal Extensions

- **16.1 Q1**：写 LLaVA 三件套架构（vision encoder + projector + LLM）+ 两阶段训练（frozen / trainable）
- **16.1 Q2**：1024×1024 / patch 14 图有多少 vision token；3 种 mitigation + 代表 VLM
- **16.1 Q3**：图表理解 VLM 在 Qwen2.5-VL-7B / InternVL2-8B / LLaVA-OneVision-7B 三选一
- **16.2 Q1**：LLaVA 拼接范式 vs Chameleon native 范式的 3 维度差异
- **16.2 Q2**：native multimodal 优势明显，为何 2025-2026 主流仍是接接器（≥ 4 原因）
- **16.2 Q3**：Gemini 2.x 的 native 具体如何 native；为什么大厂都走 native
- **16.3 Q1**：VALL-E LLM 范式做 TTS 的关键 idea；为何要 Encodec；AR + NAR 双阶段动机
- **16.3 Q2**：Moshi / GPT-4o realtime vs 传统 ASR+LLM+TTS 三维度对比
- **16.3 Q3**：中文实时语音助手（手机 App）的 ASR / TTS / LLM 选型 + latency 优化
- **16.4 Q1**：写 InfoNCE loss 公式 + 为什么 hard negative mining 是关键
- **16.4 Q2**：bge-m3 (568M, 1024 dim) vs E5-mistral-7b (7B, 4096 dim) 的 ≥ 4 维度选型
- **16.4 Q3**：500 万 chunk 中英混合企业 RAG 的 embedding 选型 + retrieval pipeline + benchmark
- **16.5 Q1**：GUI agent 的 3-4 种输入表示 + 各自 trade-off + 选型场景
- **16.5 Q2**：OS 助手 agent 从 base VLM 到 production 的完整 pipeline（grounding + action + RL）
- **16.5 Q3**：Reasoning model + GUI agent 的融合点；2026 年可能突破

### Appendix

- **附A Q1**：Module 4.6 mini-LLaMA → GPT-2 124M 的改动清单 + "为什么"
- **附A Q2**：1×A100 80G 训 GPT-2 124M 的 throughput + 墙钟时间
- **附A Q3**：复现完 GPT-2 124M 后到 chat-able 的完整 SFT pipeline
- **附B Q1**：24 GB GPU 上 SFT + LoRA + RAG 完整 8-step pipeline
- **附B Q2**：QLoRA r=64 vs r=128 vs full SFT trade-off + 三种预算选型
- **附B Q3**：客服 agent 答产品手册：(a) SFT (b) long-context (c) RAG 三选一
- **附C Q1**：mini Deep Research agent 的 planner-worker-synthesizer 数据流图
- **附C Q2**：sub-task 10 step 没解决（max_step 触发），3 个 debug 方向
- **附C Q3**：mini Deep Research → production 级（OpenAI Deep Research 量级）的 3 大方向
- **附D Q1**：写 KL 与 cross-entropy 关系 + 为何训 LM 优化 CE = 最小化 KL
- **附D Q2**：为什么 LLM 训练用 cross-entropy 而非 MSE（概率角度）
- **附D Q3**：用 PyTorch 写 numerically stable LogSumExp（≤ 5 行）

---

## 2. 反向索引（按知识点 / 题型）

### 2.1 公式推导题（Math / Derivation）

- **scaled dot-product attention 公式 + $\sqrt{d_k}$ 推导**：4.1
- **multi-head attention 公式**：4.2
- **RoPE 内积仅依赖相对位置的推导**：4.3
- **SwiGLU FFN 完整公式**：4.5
- **LSTM 4 门 + Bahdanau attention 三步**：2.3 / 2.4
- **online softmax (FlashAttention) rescaling**：5.3
- **discrete SSM 递推 (Mamba)**：5.5
- **Adam 完整更新式 + 偏差修正**：1.2
- **CE / KL 关系 + LogSumExp**：1.4 / 附D
- **Skip-gram + Negative Sampling loss**：2.2
- **CLM vs MLM loss**：6.1
- **Chinchilla compute-optimal**：6.3
- **PI / NTK-aware long-context**：6.5
- **Bradley-Terry RM loss**：9.2
- **PPO clipped objective**：9.3
- **DPO 推导（含 $\beta \log Z(x)$ 消去）**：9.4
- **GRPO advantage（group baseline + std）**：9.5
- **GAE 递推**：9.1
- **trajectory reward → token advantage broadcast**：15.2
- **Generative Agents memory retrieval score**：14.5
- **InfoNCE loss + hard negative**：16.4
- **spec decoding rejection sampling lossless 证明**：11.5

### 2.2 代码实现题（Implementation）

- **single-head causal self-attention（≤ 15 行）**：4.1
- **multi-head Q/K/V projection + reshape/permute**：4.2
- **RoPE `apply_rotary` (rotate_half) 4 行**：4.3
- **LLaMA-style FFN (SwiGLU) ≤ 8 行**：4.5
- **LLaMA-style Block ≤ 30 行（Pre-RMSNorm + SDPA + SwiGLU + 2 residual）**：4.6
- **RMSNorm 4 行**：1.3
- **PyTorch 算 KL（log_softmax / softmax 顺序）**：1.4
- **GQA forward ≤ 15 行（K/V g→h_q expand）**：5.2
- **MoE router + top-K + GShard aux loss**：5.4
- **FIM transform PSM 重排 5 行**：6.1
- **SFT labels + -100 mask（ChatML 多轮）**：8.2
- **Houlsby Adapter 5 行**：8.4
- **Triton vector add kernel 5 行（pid / mask / load+store）**：7.6
- **OpenAI tool schema (`get_user_info`)**：13.4
- **最小 ReAct loop 5 步伪代码**：14.2
- **MinHash + LSH 5 步**：6.2
- **BPE 训练 5 步伪代码 + 演示**：3.1
- **block-diagonal mask（sample packing）**：3.3
- **pairwise judge prompt（消 bias）**：12.2
- **system + few-shot + CoT prompt（GSM8K）**：13.1
- **完整 LLM data flow 画图（decoder-only）**：4.6
- **stable LogSumExp 5 行**：附D

### 2.3 显存 / 算账 题（Capacity / Calculation）

- **bf16 + AdamW 训 7B 显存（weight + grad + optim）**：1.5
- **70B 在 16×H100 上 DDP / ZeRO-2 / ZeRO-3 单卡显存**：7.1
- **LLaMA-2 13B 估算（$P \approx Vd + 12Ld^2$）**：4.6
- **LLaMA-3 70B 在 T=32k 下 MHA / GQA-8 KV cache**：4.7
- **LLaMA-3 70B GQA KV cache（80 层 / 64 头 / g=8）**：5.2
- **Mixtral 8x22B 总参数 vs 激活参数**：5.4
- **Korthikanti 公式 LLaMA-2 13B activation 显存**：7.5
- **GPipe / 1F1B / Interleaved / Zero-Bubble bubble ratio**：7.2
- **Chinchilla 算 LLaMA-2 7B 最优 token 数**：6.3
- **H100 跑 LLaMA-3 70B TP=8 decode TPS 上限**：11.1
- **PagedAttention LLaMA-3 70B 在 H100 80G 同时 sample 数**：11.2
- **1024×1024 / patch 14 图的 vision token 数**：16.1
- **1×A100 80G 训 GPT-2 124M throughput / 墙钟**：附A
- **LLaMA-2 70B vs 7B lr 配比**：6.4

### 2.4 概念辨析题（"X vs Y" / "为什么 X 而不是 Y"）

- **fp16 vs bf16**：1.5 / 7.4
- **LSTM vs GRU vs Transformer**：2.3
- **BPE vs WordPiece vs Unigram**：3.1
- **词表 32k vs 256k**：3.2
- **padding vs packing vs padding-free**：3.3
- **head 数 32 vs 8**：4.2
- **RoPE vs ALiBi**：4.3
- **Pre-LN vs Post-LN**：4.4
- **MHA vs GQA vs MLA**：5.2
- **DeepSeek-MoE 256 小 vs Mixtral 8 大 expert**：5.4
- **Mamba vs Transformer**：5.5
- **encoder-only / decoder-only / encoder-decoder**：5.1
- **DDP vs ZeRO-2 vs ZeRO-3 / FSDP**：7.1
- **SP vs CP vs EP**：7.3
- **Selective vs Full Recompute**：7.5
- **Triton vs CUDA**：7.6
- **LoRA vs QLoRA vs full SFT**：8.3 / 附B
- **Adapter vs Prefix-tuning vs LoRA**：8.4
- **DPO vs SimPO vs KTO**：9.4
- **PPO vs GRPO**：9.5
- **RLHF vs RLAIF vs CAI vs Self-Rewarding**：9.7
- **ORM vs PRM**：10.2
- **Self-Rewarding LM vs RLVR**：9.7 / 10.3
- **CoT / Self-Consistency / ToT / Reflexion**：10.1
- **vLLM vs SGLang**：11.3
- **W4A16 / W8A8 / FP8（AWQ vs GPTQ）**：11.4
- **Spec Decoding vs Medusa vs EAGLE**：11.5
- **HyDE / RAG-Fusion / Self-RAG / GraphRAG**：13.3
- **constrained decoding：开 vs 关**：13.4
- **ReAct vs Plan-and-Execute / Reasoning agent**：14.1 / 14.2
- **Plan-and-Solve / ToT / LATS / Code-as-Policy**：14.4
- **Toolformer / Gorilla / ToolLLaMA / ToolACE**：14.3
- **手写 ReAct vs LangChain/LangGraph**：14.7
- **MetaGPT vs orchestrator-worker**：14.6
- **WebArena / SWE-Gym / OSWorld**：15.3
- **bge-m3 vs E5-mistral-7b**：16.4
- **LLaVA 拼接 vs Chameleon native**：16.2
- **Moshi vs ASR+LLM+TTS pipeline**：16.3
- **Agentic RAG vs Search-R1**：13.3
- **prompt-time scaling vs reasoning model**：10.1 / 10.4 / 13.1 / 14.2 / 14.4

### 2.5 实战 trade-off / 选型题

- **LLM 数据组织**：3.3, 8.2
- **24 GB GPU 上 SFT 7B**：7.1, 8.3, 附B
- **70B 部署到 4×A100**：11.4
- **70B + 8×H100 + GQA-8 服务 SLO**：11.1
- **真实 env RL 工程挑战**：15.3
- **企业客服 RAG（100 万工单）**：13.2
- **法律 RAG（50 万 page，high factuality）**：13.3
- **中文 SFT 模型选 base**：6.6
- **图表理解 VLM 选型**：16.1
- **中文实时语音助手**：16.3
- **企业知识库 embedding 选型**：16.4
- **agent platform 框架选型**：14.7
- **multi-agent 设计选型**：14.6
- **客服 agent：SFT vs long-context vs RAG**：附B
- **agent SFT vs RL 衔接**：8.5, 14.3, 15.1
- **JSON 输出**：13.1, 13.4

### 2.6 调参 / Debug / 故障排查题

- **loss NaN 排查**（顺序 lr/data/dtype/scheduler/scaler）：1.2, 1.5, 4.4, 6.4
- **eval loss 早升**：1.3, 8.2
- **RM reward 涨但 eval 降 / hacking**：9.2, 9.3, 9.6
- **PPO policy 越来越长**：9.2
- **length hacking / tool spam**：9.6, 15.2
- **FlashAttention 开了不加速**：5.3
- **vLLM P99 偶发飙高**：11.2
- **TTFT 飘到 1.5s**：11.1
- **ReAct hallucinate 不存在 tool**：14.2
- **截断 JSON `{"city": "Beijing"`**：13.4
- **agent sub-task 触发 max_step**：附C
- **30 天预训健康度判断（5 指标）**：6.4
- **LLM 客服系统监控（≥ 8 指标）**：12.4
- **OpenAI API down graceful degrade**：12.4

### 2.7 前沿 / 系统设计题（2024-2026）

- **R1 / RLVR / DeepSeek-R1 pipeline**：10.3, 15.4
- **DeepSeek-V3 (MoE 256 expert / aux-free balancing / MLA)**：5.2, 5.4, 6.6
- **DAPO / GRPO 演化**：9.5
- **QK-Norm 稳定性 trick**：4.4
- **FlashAttention v2 outer-loop 换向**：7.6
- **Reasoning Agent / Search-R1 / ReSearch / ReTool**：15.4
- **SWE-RL 核心创新**：15.3
- **noisy obs advantage（用户当前研究方向）**：15.5
- **emergent ability 真伪**：6.3
- **Aha moment 涌现**：10.3
- **Self-Rewarding LM**：9.7
- **Mamba / SSM 是否回归**：2.3, 5.5
- **ModernBERT 在 LLM 时代意义**：5.1
- **Anthropic Building Effective Agents（不过度工程）**：14.7
- **EAGLE-3 3-5× 加速**：11.5
- **GUI agent + reasoning 融合**：16.5
- **Gemini 2.x native multimodal**：16.2
- **agent robustness（对应用户 ACL 2026 paper）**：15.5

---

## 3. 校招准备路线

### 3.1 时间紧（< 1 月，30 题为主）

| Day | 内容 | 章节 |
|---|---|---|
| Day 1-3 | Transformer 必考（self-attn / MHA / RoPE / SwiGLU / KV cache / Pre-LN） | Module 4 全章 |
| Day 4 | 现代架构（GQA / MLA / FlashAttention / MoE） | Module 5 |
| Day 5 | RLHF（GAE / RM / PPO / DPO / GRPO / hacking） | Module 9 |
| Day 6 | Infra（FSDP / TP / 量化 / vLLM / spec decoding） | 7.1, 7.2, 11.2, 11.4, 11.5 |
| Day 7 | Reasoning / Agent（R1 pipeline / ReAct / Agent RL） | 10.3, 14.2, 15.4 |

每日刷"反向索引"对应章节的 Q1+Q2，盲做 → 对照 answer sketch。

### 3.2 完整准备（3 月）

按教程顺序走完 Module 0 → 16 + 附录，每节 Q1/Q2/Q3 都做。
- 每周完整覆盖 1-2 个 module。
- 每 module 结束后做该 module 全部 Q（≈ 15 题）的"模拟面试"——口述 5 分钟讲清。
- Module 9 / 10 / 15 应该刷 2 遍。

### 3.3 跳槽准备（公司核心方向）

- **后训练岗**（小红书 LLM 后训练 / DeepSeek / Qwen / 字节 Doubao）：Module 8 + 9 + 10 + 15 全章 + 2.4 节 RLHF reward hacking 系列
- **Infra 岗**（NVIDIA / 蚂蚁 / 字节 Veomni）：Module 7 全章 + 11.2 / 11.4 / 11.5 + 7.6 Triton
- **Agent 岗**（Anthropic / Manus / 阿里 AgentScope）：Module 14 + 15 全章 + 13.2 RAG + 13.4 function calling + 16.5 GUI agent
- **多模态岗**（智谱 / 通义 VL / Hailuo）：Module 16 全章 + 4 (Transformer 基础) + 5.2 GQA / 5.3 FlashAttention

---

## 4. 自测方法建议

1. **盲做** Q1/Q2/Q3（计时：每题最多 5 分钟，公式题写在白板上）
2. **对照** 原节 §6 的 answer sketch；卡住的标记 `★`
3. **缺则** 回到对应节正文 + 补 README.md / ROADMAP.md 的脉络
4. **复盘**：每周末统计 `★` 分布，集中攻克高频缺漏的知识点
5. **口述训练**：把 Top 30 中的每道题用 5 分钟向自己（或朋友 / 镜子）讲清楚——能讲清才算真懂
6. **Mock interview**：找同方向 senior，把"反向索引按知识点"做成题库随机抽题

---

## 附：与本书其他索引文件的关系

- **README.md** — 教程入口、学习路线图
- **ROADMAP.md** — 87 节正文目录、章节依赖关系
- **面试题索引.md（本文件）** — 261 道自测题反向索引
- 完整 answer sketch 在每节正文 §6 自测与面试题（折叠 `<details>` 内）
