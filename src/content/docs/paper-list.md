---
title: "论文清单"
---

> 87 节内容引用的全部经典 paper 整理，按主题与节双索引。涵盖 ~120 篇核心 paper，覆盖架构 / 预训练 / 后训练 / Infra / Agent / 多模态 / RAG / 评测 8 大方向，时间跨度 1986-2026。

> 阅读约定：粗体 = 必读必引；普通 = 推荐精读；选读 = 进阶/补充。每篇标注 [出现节列表]。

---

## 速览：~50 篇核心必读

精选最值得反复重读、面试 / 复现 / 工程必备的 paper。按 6 大主题分组。

### 1. 架构 (~7 篇)

- **Vaswani et al., 2017 — *Attention is All You Need*** — Transformer 原典，全书出现频次最高 [节 2.4, 3.3, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 5.1]
- **Su et al., 2021 — *RoFormer (RoPE)*** — 旋转位置编码，现代 LLM 标配 [4.3, 5.5]
- **Shazeer, 2020 — *GLU Variants Improve Transformer (SwiGLU)*** — FFN 激活的事实标准 [4.5]
- **Zhang & Sennrich, 2019 — *Root Mean Square Layer Normalization (RMSNorm)*** — 现代 LLM Norm 事实标准 [1.3, 4.4]
- **Xiong et al., 2020 — *On Layer Normalization in the Transformer Architecture*** — Pre-LN vs Post-LN 严格分析，奠定现代设计 [4.4]
- **Ainslie et al., 2023 — *GQA*** — KV cache 压缩中庸之道，LLaMA-2/3 标配 [5.2]
- **Dai et al., 2024 / DeepSeek-AI 2024 — *DeepSeek-V2 (MLA)*** — KV cache 压缩激进路线 [5.2, 6.6]

### 2. 预训练 & Scaling (~7 篇)

- **Devlin et al., 2018 — *BERT*** — encoder-only 起点，MLM 范式 [2.1, 5.1, 6.1]
- **Radford et al., 2018/2019 — *GPT-1 / GPT-2*** — decoder-only LM 路线起点 [5.1, 6.1, 13.1, 附录 A]
- **Brown et al., 2020 — *GPT-3 (Few-Shot Learners)*** — in-context learning 奠基 [2.1, 4.6, 13.1, 附录 A]
- **Kaplan et al., 2020 — *Scaling Laws for Neural Language Models*** — power law 原典 [6.3, 6.4]
- **Hoffmann et al., 2022 — *Chinchilla*** — compute-optimal $D \approx 20N$ [6.3, 6.4, 附录 A]
- **Penedo et al., 2024 — *FineWeb*** — 现代开源 corpus 事实参考 [6.2]
- **Touvron et al., 2023 — *LLaMA / LLaMA-2*** — 现代工业 LLM 范本（架构 + RLHF） [3.3, 4.4, 4.6, 6.4, 6.6, 8.2, 9.6, 附录 B]

### 3. 后训练 & RLHF (~10 篇)

- **Wei et al., 2022 — *FLAN (Finetuned LMs are Zero-Shot Learners)*** — instruction tuning 奠基 [8.2]
- **Hu et al., 2021 — *LoRA*** — PEFT 事实标准 [8.3, 附录 B]
- **Dettmers et al., 2023 — *QLoRA*** — 4-bit + LoRA 单卡训 65B [7.5, 8.3, 附录 B]
- **Christiano et al., 2017 — *Deep RL from Human Preferences*** — RLHF 鼻祖 [9.2]
- **Ouyang et al., 2022 — *InstructGPT*** — SFT→RM→PPO 三段式 [9.2, 9.3]
- **Schulman et al., 2017 — *PPO*** — RLHF 时代不变的算法基础 [9.3]
- **Rafailov et al., 2023 — *DPO (Your LM is Secretly a Reward Model)*** — 简化 RLHF [9.4]
- **Shao et al., 2024 — *DeepSeekMath (GRPO)*** — 当代 reasoning RL 算法基础 [9.5, 10.3, 15.2]
- **DeepSeek-AI, 2025 — *DeepSeek-R1*** — RLVR + long-CoT 范式胜利 [6.6, 9.5, 10.2, 10.3, 15.2, 15.4]
- **Bai et al., 2022 — *Constitutional AI / RLHF (HH)*** — Anthropic 系 alignment 完整方法 [9.2, 9.3, 9.6, 9.7]

### 4. Infra & 推理 (~7 篇)

- **Rajbhandari et al., 2020 — *ZeRO*** — DeepSpeed 三阶段分片，FSDP 直系前作 [7.1]
- **Shoeybi et al., 2019 — *Megatron-LM*** — Tensor Parallel 范式奠基 [7.2]
- **Korthikanti et al., 2022 — *Reducing Activation Recomputation (Megatron-SP)*** — Sequence Parallel + selective recompute [7.3, 7.5]
- **Micikevicius et al., 2017 — *Mixed Precision Training*** — fp16/AMP 开山 [1.5, 7.4]
- **DeepSeek-AI, 2024 — *DeepSeek-V3 Technical Report*** — fp8 + MTP + bias balancing 集大成 [1.5, 5.4, 6.6, 7.3, 7.4]
- **Dao et al., 2022/2023/2024 — *FlashAttention 1/2/3*** — IO-aware attention，硬件协同设计 [3.3, 5.3, 7.6]
- **Kwon et al., 2023 — *PagedAttention / vLLM*** — production 推理事实标准 [4.7, 11.2, 11.3]

### 5. Agent (~10 篇)

- **Yao et al., 2022 — *ReAct*** — Agent 时代真正开山作 [14.1, 14.2, 附录 C]
- **Wei et al., 2022 — *Chain-of-Thought Prompting*** — reasoning prompt 鼻祖 [10.1, 13.1, 14.1]
- **Shinn et al., 2023 — *Reflexion (Verbal RL)*** — 失败反思机制 [10.1, 14.2, 14.5, 附录 C]
- **Yao et al., 2023 — *Tree of Thoughts*** — search-based reasoning 起点 [10.1, 10.4, 14.4]
- **Schick et al., 2023 — *Toolformer*** — tool learning 奠基 [14.3]
- **Liu et al., 2024 — *xLAM*** — agent SFT 工业最完整 recipe [8.5, 14.3, 15.1]
- **Park et al., 2023 — *Generative Agents*** — episodic memory 事实模板 [14.5]
- **Wu et al., 2023 — *AutoGen*** — multi-agent framework 事实标准 [14.1, 14.6, 14.7]
- **Anthropic 2024 — *Building Effective Agents*** — agent 工程白皮书 [14.1, 14.2, 14.6, 14.7, 附录 C]
- **Hou et al., 2025 — *Search-R1*** — R1 范式向 multi-turn agent 扩展第一例 [10.3, 13.3, 14.4, 15.2, 15.4]

### 6. 多模态 & RAG (~7 篇)

- **Radford et al., 2021 — *CLIP*** — 对比学习对齐 image-text 奠基 [1.4, 16.1]
- **Liu et al., 2023 — *LLaVA / LLaVA-1.5*** — 接接器范式开山 [16.1]
- **Bai et al., 2024 — *Qwen2-VL*** — 动态分辨率 + 3D RoPE 工程标杆 [16.1]
- **Radford et al., 2022 — *Whisper*** — ASR 的"BERT 时刻" [16.3]
- **Lewis et al., 2020 — *RAG (Retrieval-Augmented Generation)*** — RAG 概念原典 [13.2, 附录 B]
- **Karpukhin et al., 2020 — *DPR (Dense Passage Retrieval)*** — dense retrieval 范式 [13.2, 16.4]
- **Xiao et al., 2024 — *BGE M3-Embedding*** — 中英多语 embedding 事实标准 [13.2, 16.4, 附录 B]

### 评测 & Safety 加分 (~3 篇)

- **Zheng et al., 2023 — *MT-Bench / Chatbot Arena (LLM-as-Judge)*** — judge 一致性奠基 [9.7, 12.1, 12.2]
- **Hendrycks et al., 2020 — *MMLU*** — 通用能力评测奠基 [12.1]
- **Zou et al., 2023 — *GCG (Universal Adversarial Attacks)*** — jailbreak 分水岭 [12.3]

---

## 完整索引（按 Module 分组）

### Module 1 — DL Basics

- (1.1) Rumelhart, Hinton & Williams 1986 — *Learning representations by back-propagating errors* (Nature) — 反向传播原典
- (1.1) Baydin et al. 2017 — *Automatic Differentiation in Machine Learning: a Survey* (JMLR) — autograd 最佳综述
- (1.1) Griewank 2000 — *Evaluating Derivatives* (book) — AD 圣经（选读）
- (1.2) Kingma & Ba 2014 — *Adam: A Method for Stochastic Optimization* — Adam 原典
- (1.2) Loshchilov & Hutter 2017 — *Decoupled Weight Decay Regularization (AdamW)* — AdamW 提出
- (1.2) Chen et al. 2023 — *Symbolic Discovery of Optimization Algorithms (Lion)* — Lion 提出
- (1.3) Hinton et al. 2012/2014 — *Dropout* — Dropout 奠基
- (1.3) Ioffe & Szegedy 2015 — *Batch Normalization* — BN 原典
- (1.3) Ba, Kiros & Hinton 2016 — *Layer Normalization* — LN 原典
- (1.3) Zhang & Sennrich 2019 — *RMSNorm* — RMSNorm 原典
- (1.4) Goodfellow, Bengio, Courville 2016 — *Deep Learning Ch.5* (book) — entropy/CE/KL/MLE 综述
- (1.4) van den Oord et al. 2018 — *Representation Learning with CPC (InfoNCE)* — InfoNCE 提出者
- (1.4) Radford et al. 2021 — *CLIP* — InfoNCE on image-text
- (1.4) Szegedy et al. 2015 — *Rethinking the Inception Architecture* — label smoothing 起源
- (1.5) Micikevicius et al. 2017 — *Mixed Precision Training* — fp16/AMP 开山
- (1.5) NVIDIA 2022 — *Hopper Architecture Whitepaper* — H100 + fp8 hardware spec
- (1.5) DeepSeek-AI 2024 — *DeepSeek-V3 Technical Report* — 端到端 fp8 训练范本

### Module 2 — NLP Landscape

- (2.1) Devlin et al. 2018 — *BERT* — encoder-only 起点
- (2.1) Raffel et al. 2020 — *T5* — text-to-text 统一
- (2.1) Brown et al. 2020 — *GPT-3* — in-context learning
- (2.1) Wang et al. 2018 — *GLUE* — NLP benchmark 平台
- (2.2) Mikolov et al. 2013 — *word2vec (Skip-gram + NEG)* — word embedding 原典
- (2.2) Pennington, Socher & Manning 2014 — *GloVe* — 全局共现 embedding
- (2.2) Bojanowski et al. 2017 — *fastText* — subword embedding
- (2.2) Peters et al. 2018 — *ELMo* — contextual embedding 起点
- (2.3) Hochreiter & Schmidhuber 1997 — *LSTM* — LSTM 原典
- (2.3) Cho et al. 2014 — *GRU + Encoder-Decoder* — seq2seq 原典
- (2.3) Pascanu, Mikolov & Bengio 2013 — *On the difficulty of training RNNs* — 梯度消失/爆炸 + clip
- (2.3) Karpathy 2015 — *The Unreasonable Effectiveness of RNNs* (blog) — char-RNN 时代标志
- (2.4) Sutskever, Vinyals & Le 2014 — *Sequence to Sequence Learning with Neural Networks* — vanilla seq2seq
- (2.4) Bahdanau, Cho & Bengio 2014 — *NMT by Jointly Learning to Align and Translate* — attention 起点
- (2.4) Luong, Pham & Manning 2015 — *Effective Approaches to Attention-based NMT* — global/local attention
- (2.4) Vaswani et al. 2017 — *Attention Is All You Need* — Transformer 终点站

### Module 3 — Tokenization

- (3.1) Sennrich, Haddow & Birch 2016 — *Neural MT of Rare Words with Subword Units (BPE)* — 现代 tokenizer 事实标准
- (3.1) Kudo & Richardson 2018 — *SentencePiece* — 多语 tokenizer 工具
- (3.1) Kudo 2018 — *Subword Regularization (Unigram LM)* — T5/Gemma 用
- (3.1) Schuster & Nakajima 2012 — *Japanese and Korean Voice Search* — WordPiece 起源
- (3.3) Vaswani et al. 2017 — *Attention Is All You Need* — causal mask 源头
- (3.3) Dao et al. 2022/2023 — *FlashAttention 1/2 (varlen)* — sample packing 零开销
- (3.3) Touvron et al. 2023 — *LLaMA-2* — sample packing 工业惯例

### Module 4 — Transformer from Scratch

- (4.1) Vaswani et al. 2017 — *Attention Is All You Need* — scaled dot-product attention 原典
- (4.1) Bahdanau et al. 2014 — *NMT* — cross-attention 精神祖师爷
- (4.1) Karpathy — *nanoGPT* (code) — 工业级 SDPA 实现教材
- (4.2) Vaswani et al. 2017 — *Attention Is All You Need* — multi-head attention 原典
- (4.2) Clark et al. 2019 — *What Does BERT Look At?* — head 可解释性
- (4.2) Michel et al. 2019 — *Are Sixteen Heads Really Better than One?* — head pruning 开山
- (4.3) Vaswani et al. 2017 — *Attention Is All You Need* — sinusoidal PE
- (4.3) Shaw, Uszkoreit & Vaswani 2018 — *Self-Attention with Relative PE* — 相对位置起点
- (4.3) Su et al. 2021 — *RoFormer (RoPE)* — RoPE 必引
- (4.3) Press, Smith & Lewis 2021 — *ALiBi (Train Short Test Long)* — 外推式位置编码
- (4.3) Chen et al. 2023 — *Position Interpolation (PI)* — RoPE long-context 系统化
- (4.3) bloc97 2023 — *NTK-aware Scaled RoPE* (blog)
- (4.4) Vaswani et al. 2017 — *Attention is All You Need* — Post-LN 出处
- (4.4) Xiong et al. 2020 — *On Layer Normalization in the Transformer Architecture* — Pre-LN vs Post-LN 严格分析
- (4.4) Touvron et al. 2023 — *LLaMA / LLaMA-2* — 现代 RMSNorm + Pre-LN 范本
- (4.4) Wang et al. 2022 — *DeepNet (Scaling Transformers to 1000 Layers)* — DeepNorm
- (4.5) Vaswani et al. 2017 — *Attention is All You Need* — ReLU FFN 配方原典
- (4.5) Hendrycks & Gimpel 2016 — *GELU* — GELU 提出
- (4.5) Shazeer 2020 — *GLU Variants Improve Transformer* — SwiGLU 全胜
- (4.5) Touvron et al. 2023 — *LLaMA* — SwiGLU 工业事实标准
- (4.6) Vaswani et al. 2017 — *Attention Is All You Need* — Transformer 原典
- (4.6) Touvron et al. 2023 — *LLaMA & LLaMA-2* — 现代 decoder-only 工程标杆
- (4.6) Karpathy — *nanoGPT* (code) — 教学价值堪比 paper
- (4.6) Brown et al. 2020 — *GPT-3* — decoder-only scale-up 范例
- (4.7) Vaswani et al. 2017 — *Attention Is All You Need* — KV cache 隐含来源
- (4.7) Pope et al. 2022 — *Efficiently Scaling Transformer Inference* — production 推理 scaling 开山
- (4.7) Kwon et al. 2023 — *PagedAttention / vLLM* — KV cache 碎片解决
- (4.7) Karpathy — *nanoGPT / llama2.c* (code) — 极简 KV cache 实现

### Module 5 — Modern LLM Architectures

- (5.1) Devlin et al. 2018 — *BERT* — encoder-only
- (5.1) Radford et al. 2018/2019 — *GPT-1 / GPT-2* — decoder-only
- (5.1) Raffel et al. 2020 — *T5* — encoder-decoder 现代化
- (5.1) Wang et al. 2022 — *What LM Architecture & Pretraining Objective Work Best for Zero-Shot?* — 三架构 ablation 权威
- (5.1) Warner et al. 2024 — *ModernBERT* — encoder-only 复兴
- (5.2) Shazeer 2019 — *Fast Transformer Decoding (MQA)* — MQA 必读
- (5.2) Ainslie et al. 2023 — *GQA* — GQA 必读
- (5.2) DeepSeek-AI 2024 — *DeepSeek-V2 (MLA)* — MLA 必读
- (5.2) Touvron et al. 2023 — *LLaMA-2* — GQA 工业落地
- (5.3) Dao et al. 2022 — *FlashAttention 1* — IO-aware attention 开山
- (5.3) Dao 2023 — *FlashAttention 2* — 工程优化合集
- (5.3) Shah et al. 2024 — *FlashAttention 3* — Hopper async + fp8
- (5.3) Milakov & Gimelshein 2018 — *Online normalizer calculation for softmax* — online softmax 数学源头
- (5.4) Shazeer et al. 2017 — *Outrageously Large NN (Sparsely-Gated MoE)* — MoE 现代起点
- (5.4) Lepikhin et al. 2020 — *GShard* — MoE on Transformer + aux loss 协议
- (5.4) Fedus et al. 2022 — *Switch Transformer* — K=1 极致路由 1.6T
- (5.4) Jiang et al. 2024 — *Mixtral of Experts* — 现代开源 MoE 标杆
- (5.4) Dai et al. 2024 — *DeepSeek-MoE* — fine-grained + shared experts
- (5.4) DeepSeek-AI 2024 — *DeepSeek-V3 Tech Report* — 2024 必精读 LLM tech report
- (5.5) Gu, Goel & Ré 2021 — *S4 (Efficiently Modeling Long Sequences with Structured SSM)* — SSM in NLP 开山
- (5.5) Gu & Dao 2023 — *Mamba* — selective SSM
- (5.5) Dao & Gu 2024 — *Mamba-2 (Transformers are SSMs)* — SSM-attention 对偶
- (5.5) Lieber et al. 2024 — *Jamba* — Hybrid Transformer-Mamba
- (5.5) Peng et al. 2023 — *RWKV* / Sun et al. 2023 — *RetNet* (加分)

### Module 6 — Pretraining

- (6.1) Devlin et al. 2018 — *BERT (MLM)* — MLM 起点
- (6.1) Radford et al. 2018/2019 — *GPT-1 / GPT-2 (CLM)* — CLM 路线源头
- (6.1) Raffel et al. 2020 — *T5 (span corruption)* — 系统 ablation
- (6.1) Bavarian et al. 2022 — *FIM (Efficient Training of LMs to Fill in the Middle)* — 所有 code model 加 FIM 的根据
- (6.1) Tay et al. 2022 — *UL2 (Mixture-of-Denoisers)* — 多目标训练
- (6.1) Du et al. 2022 — *GLM (Autoregressive Blank Infilling)* — Prefix-LM 现代代表
- (6.2) Gao et al. 2020 — *The Pile* — 早期组合 corpus 范本
- (6.2) Lee et al. 2021 — *Deduplicating Training Data Makes LMs Better* — 去重必读
- (6.2) Penedo et al. 2024 — *FineWeb* — 现代 SOTA 公开 corpus
- (6.2) Li et al. 2024 — *DataComp-LM (DCLM)* — model-based filter SOTA
- (6.2) Soldaini et al. 2024 — *Dolma* — 完全 open corpus
- (6.2) Rae et al. 2021 — *Gopher* MassiveText filter (加分)
- (6.2) Xie et al. 2023 — *DoReMi* (加分) — domain mixing
- (6.3) Kaplan et al. 2020 — *Scaling Laws for Neural Language Models* — power law 原典
- (6.3) Hoffmann et al. 2022 — *Chinchilla (Training Compute-Optimal LLMs)* — 必读必背
- (6.3) Shao et al. 2024 — *DeepSeekMath* (§6 scaling section) — Chinchilla 修正
- (6.3) Wei et al. 2022 — *Emergent Abilities of LLMs* — emergence 提出
- (6.3) Schaeffer et al. 2023 — *Are Emergent Abilities a Mirage?* (NeurIPS 2023 best) — emergence 反驳
- (6.3) Sardana et al. 2024 — *Beyond Chinchilla-Optimal* (加分) — over-training scaling law
- (6.4) McCandlish et al. 2018 — *An Empirical Model of Large-Batch Training* — critical batch size
- (6.4) Smith et al. 2017 — *Don't Decay the LR, Increase the Batch Size* — batch / lr 对偶
- (6.4) Touvron et al. 2023 — *LLaMA-2* — 现代 LLM 标准超参清单
- (6.4) Kaplan et al. 2020 / Hoffmann et al. 2022 — *Scaling Laws / Chinchilla* — lr-N 关系
- (6.5) Chen et al. 2023 — *Position Interpolation (PI)* — Meta long-context 第一系统化
- (6.5) Peng et al. 2023 — *YaRN* — LLaMA-2/3/Qwen/Mistral 工业标准
- (6.5) Ding et al. 2024 — *LongRoPE* — 扩到 2M
- (6.5) bloc97 2023 — *NTK-aware Scaled RoPE* (blog)
- (6.5) Hsieh et al. 2024 — *RULER* (加分) — long-context 真实评测
- (6.5) Liu et al. 2023 — *Ring Attention* (加分) — long-context 训练 infra
- (6.6) Touvron et al. 2023 — *LLaMA* — 开源 LLM 时代起点
- (6.6) Touvron et al. 2023 — *Llama 2* — GQA-8 + RLHF 工程范本
- (6.6) Dubey et al. 2024 — *Llama 3 Herd* — 405B + 15T over-training
- (6.6) Yang et al. 2024 — *Qwen2 / Qwen2.5 Technical Report* — 中文 LLM 工程细节
- (6.6) DeepSeek-AI 2024 — *DeepSeek-V2* — MLA + DeepSeek-MoE 奠基
- (6.6) DeepSeek-AI 2024 — *DeepSeek-V3 Tech Report* — 2024 必精读
- (6.6) DeepSeek-AI 2025 — *DeepSeek-R1* — reasoning model 浪潮标志

### Module 7 — Training Infra

- (7.1) Rajbhandari et al. 2020 — *ZeRO* — DeepSpeed 三阶段奠基
- (7.1) Zhao et al. 2023 — *PyTorch FSDP* — FSDP 工程化
- (7.1) Goyal et al. 2017 — *Accurate, Large Minibatch SGD* — DDP linear scaling rule
- (7.1) Ren et al. 2021 — *ZeRO-Infinity* (加分) — CPU/NVMe offload
- (7.2) Shoeybi et al. 2019 — *Megatron-LM* — TP 奠基
- (7.2) Huang et al. 2019 — *GPipe* — PP 起点
- (7.2) Narayanan et al. 2021 — *Efficient Large-Scale LM Training (Megatron 3D parallel)* — 3D parallel 系统化
- (7.2) Qi et al. 2024 — *Zero Bubble Pipeline Parallelism* — PP 调度 SOTA
- (7.2) Korthikanti et al. 2022 — *Megatron-SP* (加分)
- (7.3) Korthikanti et al. 2022 — *Reducing Activation Recomputation (Megatron-SP)* — SP + selective recompute
- (7.3) Liu et al. 2023 — *Ring Attention with Blockwise Transformers* — Ring Attention 起源
- (7.3) Brandon et al. 2023 — *Striped Attention* — Ring Attention causal 优化
- (7.3) Lepikhin et al. 2020 — *GShard* / Fedus et al. 2022 — *Switch Transformer* — EP 奠基
- (7.3) DeepSeek-AI 2024 — *DeepSeek-V3 Tech Report* — DeepEP + bias balancing 工业实现
- (7.3) Jacobs et al. 2023 — *DeepSpeed Ulysses* (加分)
- (7.4) Micikevicius et al. 2017 — *Mixed Precision Training* — fp16/AMP 开山
- (7.4) Micikevicius et al. 2022 — *FP8 Formats for Deep Learning* — fp8 标准
- (7.4) DeepSeek-AI 2024 — *DeepSeek-V3 Tech Report* — 端到端 fp8 训练
- (7.5) Chen et al. 2016 — *Training Deep Nets with Sublinear Memory Cost* — gradient checkpointing 开山
- (7.5) Korthikanti et al. 2022 — *Reducing Activation Recomputation* — selective recompute
- (7.5) Rajbhandari et al. 2021 — *ZeRO-Infinity* — CPU/NVMe offload
- (7.5) Dettmers et al. 2023 — *QLoRA* (加分)
- (7.6) Tillet, Kung, Cox 2019 — *Triton: An Intermediate Language and Compiler* — Triton 原典
- (7.6) Dao et al. 2022 — *FlashAttention 1*
- (7.6) Dao 2023 — *FlashAttention 2*

### Module 8 — Post-training SFT/PEFT

- (8.1) Wang et al. 2022 — *Self-Instruct* — LLM 自动造 SFT 数据开山
- (8.1) Zhou et al. 2023 — *LIMA (Less Is More for Alignment)* — quality > quantity
- (8.1) Xu et al. 2023 — *WizardLM (Evol-Instruct)* — instruction 进化
- (8.1) Xu et al. 2024 — *Magpie* — 现代规模化 SFT 数据生成 SOTA
- (8.1) Lambert et al. 2024 — *Tülu 3* — 完整 SFT+DPO+RLVR 工程范本
- (8.1) DeepSeek-AI 2025 — *DeepSeek-R1* (加分) — long-CoT 蒸馏数据
- (8.2) Wei et al. 2022 — *FLAN* — instruction tuning 奠基
- (8.2) Touvron et al. 2023 — *Llama 2* — SFT 工程细节最详细公开报告
- (8.2) Lambert et al. 2024 — *Tülu 3* — 现代 SFT pipeline 总结
- (8.2) Bavarian et al. 2022 — *FIM* — special token 标记数据结构
- (8.2) HuggingFace TRL — *SFTTrainer* (docs)
- (8.3) Hu et al. 2021 — *LoRA* — LoRA 原典
- (8.3) Dettmers et al. 2023 — *QLoRA* — NF4 + double quant + paged optim
- (8.3) Liu et al. 2024 — *DoRA (Weight-Decomposed LoRA)* — magnitude/direction 分解
- (8.3) Aghajanyan et al. 2020 — *Intrinsic Dimensionality* — LoRA 理论先声
- (8.4) Houlsby et al. 2019 — *Parameter-Efficient Transfer Learning for NLP (Adapter)* — Adapter 起点
- (8.4) Li & Liang 2021 — *Prefix-Tuning* — Prefix-tuning 原典
- (8.4) Liu et al. 2022 — *Few-Shot PEFT (IA³)* — IA³ 提出
- (8.4) Lester et al. 2021 — *Prompt Tuning (Power of Scale)* (选读)
- (8.4) Liu et al. 2022 — *P-Tuning v2* (选读)
- (8.4) Zhao et al. 2024 — *GaLore (Gradient Low-Rank Projection)* (选读)
- (8.5) Liu et al. 2024 — *xLAM* — agent SFT 数据 + 训练 recipe 最完整
- (8.5) Liu et al. 2024 — *ToolACE* — multi-agent 合成 framework
- (8.5) Patil et al. 2023 — *Gorilla* — early tool LLM
- (8.5) Qin et al. 2023 — *ToolLLM* — 16k+ 真实 API
- (8.5) Yang et al. 2024 — *Qwen2.5 Tech Report* — chat template + tool special token
- (8.5) BFCL Leaderboard (加分)

### Module 9 — Post-training RLHF

- (9.1) Sutton & Barto 2018 — *Reinforcement Learning: An Introduction* (book) — RL 圣经
- (9.1) Schulman et al. 2015 — *Generalized Advantage Estimation (GAE)* — GAE 原典
- (9.1) Mnih et al. 2016 — *Asynchronous Methods for Deep RL (A2C/A3C)* — Actor-Critic 工业化
- (9.1) Williams 1992 — *REINFORCE* (选读) — REINFORCE 原典
- (9.2) Christiano et al. 2017 — *Deep RL from Human Preferences* — RLHF 奠基
- (9.2) Ouyang et al. 2022 — *InstructGPT* — SFT→RM→PPO 三段式
- (9.2) Bai et al. 2022 — *Training a Helpful and Harmless Assistant with RLHF* — Anthropic 版 RLHF
- (9.2) Cui et al. 2024 — *UltraFeedback* — 当代最常用开源偏好数据集
- (9.2) Liu et al. 2024 — *Skywork-Reward* + Wang et al. 2024 — *ArmoRM* — 开源 RM 标杆
- (9.3) Schulman et al. 2017 — *PPO (Proximal Policy Optimization Algorithms)* — PPO 原典
- (9.3) Schulman et al. 2015 — *TRPO (Trust Region Policy Optimization)* — PPO 直接前作
- (9.3) Ouyang et al. 2022 — *InstructGPT* — PPO + LLM 范式
- (9.3) Bai et al. 2022 — *RLHF (Anthropic HHH)* — 工业 RLHF 实战
- (9.3) Engstrom et al. 2020 — *Implementation Matters in Deep Policy Gradients* (选读) — PPO 实现细节解剖
- (9.4) Rafailov et al. 2023 — *DPO (Your LM is Secretly a Reward Model)* — DPO 原论文
- (9.4) Azar et al. 2023 — *IPO (General Theoretical Paradigm)* — DPO 过拟合 + IPO
- (9.4) Ethayarajh et al. 2024 — *KTO (Prospect Theoretic Optimization)* — unary feedback
- (9.4) Meng et al. 2024 — *SimPO* — reference-free reward
- (9.4) Hong et al. 2024 — *ORPO* — odds ratio + SFT 一体
- (9.4) Ahmadian et al. 2024 — *RLOO (Back to Basics)* — REINFORCE 重生
- (9.4) Tang et al. 2024 — *Generalized Preference Optimization* — 统一 framework
- (9.5) Shao et al. 2024 — *DeepSeekMath (GRPO)* — GRPO 原典
- (9.5) DeepSeek-AI 2025 — *DeepSeek-R1* — GRPO + RLVR 工业胜利
- (9.5) Yu et al. 2024 — *DAPO* — verl 框架算法核心
- (9.5) Liu et al. 2024 — *Dr.GRPO (Understanding R1-Zero-Like Training)* (选读) — GRPO bias 修复
- (9.5) Ahmadian et al. 2024 — *RLOO* (选读)
- (9.6) Skalse et al. 2022 — *Defining and Characterizing Reward Hacking* — reward hacking 数学定义
- (9.6) Singhal et al. 2023 — *Length Correlations in RLHF* — length bias 奠基
- (9.6) Coste et al. 2023 — *Reward Model Ensembles Help Mitigate Overoptimization* — RM ensemble
- (9.6) Sharma et al. 2023 — *Towards Understanding Sycophancy in LMs* — sycophancy 成因
- (9.6) Touvron et al. 2023 — *LLaMA 2* — 工业 RLHF v1-v5 迭代
- (9.6) Gao et al. 2023 — *Scaling Laws for RM Overoptimization* (选读)
- (9.6) Bai et al. 2022 — *RLHF HHH* (选读)
- (9.7) Bai et al. 2022 — *Constitutional AI (CAI)* — RLAIF + CAI 奠基
- (9.7) Lee et al. 2023 — *RLAIF (Google)* — RLAIF vs RLHF 实证
- (9.7) Yuan et al. 2024 — *Self-Rewarding LMs (Meta)* — self-reward 范式
- (9.7) Cui et al. 2024 — *UltraFeedback* — 当代开源 RLAIF 偏好集
- (9.7) Zheng et al. 2023 — *MT-Bench / Chatbot Arena (LLM-as-Judge)* — judge 可靠性

### Module 10 — Reasoning & Test-time Scaling

- (10.1) Wei et al. 2022 — *Chain-of-Thought Prompting (CoT)* — CoT 奠基
- (10.1) Wang et al. 2022 — *Self-Consistency* — majority vote
- (10.1) Yao et al. 2023 — *Tree of Thoughts (ToT)* — search-based reasoning
- (10.1) Shinn et al. 2023 — *Reflexion* — verbal RL 鼻祖
- (10.1) Madaan et al. 2023 — *Self-Refine* (加分)
- (10.1) Besta et al. 2023 — *Graph of Thoughts* (加分)
- (10.1) Kojima et al. 2022 — *Zero-Shot Reasoners ("Let's think step by step")* (加分)
- (10.2) Lightman et al. 2023 — *Let's Verify Step by Step (PRM800K)* — PRM 奠基
- (10.2) Wang et al. 2024 — *Math-Shepherd* — 自动化 PRM label
- (10.2) Cobbe et al. 2021 — *Training Verifiers (GSM8K)* — outcome verifier 起源
- (10.2) DeepSeek-AI 2024 — *DeepSeek-Math* — 开源 Math + 开源 PRM
- (10.2) DeepSeek-AI 2025 — *DeepSeek-R1* — 为何不用 PRM 转用 rule verifier
- (10.2) Wang et al. 2024 — *OmegaPRM* — Math-Shepherd 高效版
- (10.3) DeepSeek-AI 2025 — *DeepSeek-R1* — RLVR 范式核心
- (10.3) Shao et al. 2024 — *DeepSeekMath (GRPO)* — R1 算法基础
- (10.3) OpenAI 2024 — *Learning to Reason with LLMs (o1 system card)* — RLVR 引爆点
- (10.3) Pan et al. 2025 — *TinyZero* (选读) — $30 复现 R1-Zero
- (10.3) HuggingFace 2025 — *Open-R1* (选读) — HF 完整复现
- (10.3) Hou et al. 2025 — *Search-R1* (选读) — RLVR + search
- (10.4) Cobbe et al. 2021 — *Training Verifiers (GSM8K)* — BoN-with-verifier 奠基
- (10.4) Wang et al. 2022 — *Self-Consistency* — majority vote BoN
- (10.4) Yao et al. 2023 — *Tree of Thoughts*
- (10.4) Zhou et al. 2023 — *LATS (Language Agent Tree Search)* — MCTS + Reflexion + value
- (10.4) Qi et al. 2024 — *rStar (Mutual Reasoning)* — MCTS + 7B model
- (10.4) DeepMind 2024 — *AlphaProof / AlphaGeometry 2* (加分)
- (10.4) OpenAI 2024 — *o1 system card* (加分)
- (10.4) Zhang et al. 2024 — *MCTSr* (加分)

### Module 11 — Inference Engines

- (11.1) Pope et al. 2022 — *Efficiently Scaling Transformer Inference* — production 推理开山
- (11.1) Patel et al. 2023 — *Splitwise (Phase Splitting)* — prefill / decode 解耦
- (11.1) Agrawal et al. 2024 — *SARATHI (Chunked Prefill)* — TTFT/TBT 平衡
- (11.2) Kwon et al. 2023 — *PagedAttention / vLLM* — 核心必引
- (11.2) Yu et al. 2022 — *Orca (OSDI'22)* — Continuous Batching 鼻祖
- (11.2) Patel et al. 2023 — *Splitwise* — prefill/decode 分离
- (11.2) vLLM GitHub — `vllm-project/vllm`
- (11.3) Zheng et al. 2024 — *SGLang (RadixAttention)* — 核心必引
- (11.3) Kwon et al. 2023 — *PagedAttention*
- (11.3) Dong et al. 2024 — *XGrammar* — structured generation engine
- (11.3) Willard & Louf 2023 — *Outlines (Efficient Guided Generation)* — constrained decoding 早期
- (11.3) SGLang GitHub
- (11.4) Frantar et al. 2022 — *GPTQ* — PTQ 开山
- (11.4) Lin et al. 2023 — *AWQ (Activation-aware Weight Quantization)* — 工业最爱
- (11.4) Xiao et al. 2022 — *SmoothQuant* — W8A8 突破
- (11.4) Dettmers et al. 2022 — *LLM.int8()* (延伸)
- (11.4) Micikevicius et al. 2022 — *FP8 Formats* (延伸)
- (11.4) Tseng et al. 2024 — *QuaRot* (延伸) — W4A4 SOTA
- (11.5) Leviathan et al. 2023 — *Speculative Decoding* — 必读奠基
- (11.5) Chen et al. 2023 — *Accelerating LLM Decoding with Speculative Sampling (DeepMind)* — 同期独立提出
- (11.5) Cai et al. 2024 — *Medusa* — 多 head + tree attention
- (11.5) Li et al. 2024 — *EAGLE / EAGLE-2 / EAGLE-3* — 现 SOTA
- (11.5) Stern, Shazeer, Uszkoreit 2018 — *Blockwise Parallel Decoding* (加分)
- (11.5) Saxena 2023 — *Prompt Lookup Decoding (PLD)* — 极简零训练

### Module 12 — Evaluation & LLMOps

- (12.1) Hendrycks et al. 2020 — *MMLU* — 通用能力评测奠基
- (12.1) Cobbe et al. 2021 — *GSM8K (Training Verifiers)* — math + CoT + verifier
- (12.1) Chen et al. 2021 — *HumanEval (Evaluating LLMs Trained on Code)* — pass@k 标准
- (12.1) Zheng et al. 2023 — *MT-Bench / Chatbot Arena* — judge 系统讨论
- (12.1) Zhou et al. 2023 — *IFEval* — instruction-following
- (12.1) Wang et al. 2024 — *MMLU-Pro* (加分)
- (12.1) Jain et al. 2024 — *LiveCodeBench* (加分)
- (12.1) Jimenez et al. 2023 — *SWE-bench* (加分)
- (12.2) Zheng et al. 2023 — *MT-Bench / Chatbot Arena* — LLM-as-Judge 奠基
- (12.2) Wang et al. 2023 — *LLMs are not Fair Evaluators* — bias 量化
- (12.2) Li et al. 2024 — *Arena-Hard Auto* — 自动 benchmark
- (12.2) Dubois et al. 2024 — *Length-Controlled AlpacaEval* — LC-WR
- (12.2) Lambert et al. 2024 — *RewardBench* — meta-benchmark
- (12.3) Mazeika et al. 2024 — *HarmBench* — safety eval 事实标准
- (12.3) Zou et al. 2023 — *GCG (Universal Adversarial Attacks)* — jailbreak 分水岭
- (12.3) Anil et al. 2024 — *Many-shot Jailbreaking* — 长 context 攻击
- (12.3) Inan et al. 2023 — *Llama Guard* (加分)
- (12.3) Sharma et al. 2025 — *Constitutional Classifier* (加分)
- (12.3) Röttger et al. 2024 — *XSTest* (加分) — over-refusal
- (12.4) Sculley et al. 2015 — *Hidden Technical Debt in ML Systems* — ML 工程经典
- (12.4) OpenTelemetry GenAI Semantic Conventions
- (12.4) Langfuse / LangSmith Cookbook (docs)

### Module 13 — Prompting / RAG / Tools

- (13.1) Brown et al. 2020 — *GPT-3 (Few-Shot Learners)* — in-context learning 奠基
- (13.1) Wei et al. 2022 — *Chain-of-Thought Prompting* — Few-shot CoT 开创
- (13.1) Kojima et al. 2022 — *Zero-Shot Reasoners* — "Let's think step by step"
- (13.1) Wang et al. 2022 — *Self-Consistency* (加分)
- (13.1) Khattab et al. 2023 — *DSPy* (加分)
- (13.2) Lewis et al. 2020 — *RAG (Retrieval-Augmented Generation)* — RAG 概念原典
- (13.2) Karpukhin et al. 2020 — *DPR (Dense Passage Retrieval)* — dense retrieval 范式
- (13.2) Xiao et al. 2024 — *BGE M3-Embedding* — 中英多语事实标准
- (13.2) Khattab & Zaharia 2020 — *ColBERT* — late-interaction
- (13.2) Es et al. 2023 — *RAGAS* — RAG 评测事实标准
- (13.3) Gao et al. 2022 — *HyDE (Hypothetical Document Embeddings)* — HyDE 原典
- (13.3) Asai et al. 2023 — *Self-RAG* — reflection token + 自适应 retrieve
- (13.3) Yan et al. 2024 — *CRAG (Corrective RAG)* — retrieval evaluator + web 兜底
- (13.3) Edge et al. 2024 — *GraphRAG (Microsoft)* — entity-relation graph + community
- (13.3) Hou et al. 2025 — *Search-R1* — RLVR + search
- (13.3) Raudaschl 2023 — *RAG-Fusion* (blog)
- (13.4) OpenAI Function Calling 文档
- (13.4) Dong et al. 2024 — *XGrammar* — structured generation SOTA
- (13.4) Yan et al. 2024 — *BFCL v1/v2/v3* — function calling 基准
- (13.4) Willard & Louf 2023 — *Outlines* (加分)
- (13.4) MCP 官方文档 (加分)

### Module 14 — Agent Systems

- (14.1) Yao et al. 2022 — *ReAct* — Agent 时代真正开山
- (14.1) Wei et al. 2022 — *Chain-of-Thought Prompting* — Agent 前作
- (14.1) Wang et al. 2024 — *Survey on LLM-based Autonomous Agents*
- (14.1) Anthropic 2024 — *Computer Use Blog & Tool Use Documentation* (加分)
- (14.1) Hong et al. 2023 — *MetaGPT* + Wu et al. 2023 — *AutoGen* + Shen et al. 2023 — *HuggingGPT* (加分)
- (14.2) Yao et al. 2022 — *ReAct*
- (14.2) Shinn et al. 2023 — *Reflexion*
- (14.2) Anthropic 2024 — *Building Effective Agents*
- (14.2) smolagents (code, 加分)
- (14.3) Schick et al. 2023 — *Toolformer* — tool learning 奠基
- (14.3) Patil et al. 2023 — *Gorilla* — retrieval-augmented tool LLM
- (14.3) Qin et al. 2023 — *ToolLLM* — 16k+ 真实 API + DFSDT
- (14.3) Liu et al. 2024 — *ToolACE* — multi-agent 合成
- (14.3) Liu et al. 2024 — *xLAM* — 工业最完整 recipe
- (14.3) Yan et al. 2024 — *BFCL v1/v2/v3* (加分)
- (14.3) Yao et al. 2024 — *τ-bench* (加分)
- (14.4) Wang et al. 2023 — *Plan-and-Solve Prompting* — plan/solve 显式分开
- (14.4) Liu et al. 2023 — *LLM+P (NL→PDDL)* — symbolic planner
- (14.4) Yao et al. 2023 — *Tree of Thoughts*
- (14.4) Zhou et al. 2023 — *LATS (Language Agent Tree Search)* — MCTS + Reflexion + value
- (14.4) Liang et al. 2022 — *Code as Policies (Google Brain)* — code-as-action 祖先
- (14.4) Wang et al. 2023 — *Voyager* (加分) — Minecraft skill library
- (14.4) Hou et al. 2025 — *Search-R1* (加分)
- (14.5) Packer et al. 2023 — *MemGPT (LLMs as OS)* — virtual memory 开山
- (14.5) Park et al. 2023 — *Generative Agents* — episodic memory 模板
- (14.5) Wang et al. 2023 — *Voyager* — skill library = procedural memory
- (14.5) Shinn et al. 2023 — *Reflexion* — verbal memory
- (14.5) mem0 / Letta / LangMem (code, 加分)
- (14.6) Li et al. 2023 — *CAMEL* — role-play multi-agent 开山
- (14.6) Wu et al. 2023 — *AutoGen* — ConversableAgent 抽象
- (14.6) Hong et al. 2023 — *MetaGPT* — SOP-driven 路线
- (14.6) Qian et al. 2023 — *ChatDev* (加分)
- (14.6) Du et al. 2023 — *Multiagent Debate* (加分)
- (14.6) Anthropic 2024 — *Building Effective Agents* (加分)
- (14.7) Anthropic 2024 — *Building Effective Agents* — agent 工程白皮书
- (14.7) Wu et al. 2023 — *AutoGen*
- (14.7) Wang et al. 2023 — *Voyager* — code-as-action 早期代表
- (14.7) smolagents (code, 加分)
- (14.7) Inspect AI (code, 加分) — UK AISI agent 评测框架

### Module 15 — Agent RL

- (15.1) Chen et al. 2023 — *FireAct* — agent trajectory SFT 奠基
- (15.1) Chen et al. 2024 — *Agent-FLAN* — 三任务拆解 + negative example
- (15.1) Zeng et al. 2023 — *AgentTuning (AgentInstruct)* — 跨 domain SFT
- (15.1) Gou et al. 2023 — *ToRA (Tool-Integrated Reasoning Agent)* — reason+tool 交错
- (15.1) Liu et al. 2024 — *xLAM* (加分)
- (15.1) Liu et al. 2024 — *APIGen* (加分) — Salesforce 60k 函数调用合成
- (15.2) Shao et al. 2024 — *DeepSeekMath (GRPO)* — 算法骨架
- (15.2) DeepSeek-AI 2025 — *DeepSeek-R1* — RLVR + GRPO 工业胜利
- (15.2) Hou et al. 2025 — *Search-R1* — multi-turn search agent 第一例
- (15.2) Yu et al. 2024 — *DAPO* — verl 算法核心
- (15.2) Pan et al. 2025 — *SWE-RL* (选读) — Meta SWE-bench RL
- (15.2) Wang et al. 2025 — *Agent-R1* (选读) — 清华开源框架
- (15.3) Nakano et al. 2021 — *WebGPT* — 真实 env RL 鼻祖
- (15.3) Zhou et al. 2023 — *WebArena* — Web agent 黄金 benchmark
- (15.3) Xie et al. 2024 — *OSWorld* — OS-level computer use benchmark
- (15.3) Pan et al. 2025 — *SWE-RL* — 真实 env RL 工业 scale
- (15.3) Jimenez et al. 2023 — *SWE-bench* (加分)
- (15.3) Deng et al. 2023 — *Mind2Web* (加分)
- (15.3) Anthropic 2024 — *Computer Use blog* (加分)
- (15.4) DeepSeek-AI 2025 — *DeepSeek-R1* — 范式起点
- (15.4) Hou et al. 2025 — *Search-R1* — 必读
- (15.4) Chen et al. 2025 — *ReSearch (Reason with Search via RL)* — multi-hop QA SOTA
- (15.4) Wang et al. 2025 — *Agent-R1* — 通用 agent RL 框架
- (15.4) Feng et al. 2025 — *ReTool (Strategic Tool Use via RL)* (选读) — 蚂蚁
- (15.4) Pan et al. 2025 — *SWE-RL* (选读)
- (15.4) Yu et al. 2024 — *DAPO* (选读)
- (15.5) Yao et al. 2024 — *τ-bench* — Sierra 真实 agent 鲁棒性
- (15.5) Guo et al. 2024 — *StableToolBench* — 评测可复现
- (15.5) Liu et al. 2024 — *Towards Robust Tool Use (Adversarial Eval)* — perturbation taxonomy
- (15.5) Wu et al. 2024 — *Adversarial Attacks on LLM Agents* — attack vectors
- (15.5) Pack-Coupling / Fission-GRPO 类 (2025-2026, 选读) — multi-turn RL noisy turn 处理前沿

### Module 16 — Multimodal Extensions

- (16.1) Radford et al. 2021 — *CLIP* — image-text 对比学习奠基
- (16.1) Liu et al. 2023 — *LLaVA / LLaVA-1.5* — 接接器范式开山
- (16.1) Bai et al. 2024 — *Qwen2-VL* — 动态分辨率 + 3D RoPE
- (16.1) Chen et al. 2024 — *InternVL* — scale up vision tower
- (16.1) Li et al. 2023 — *BLIP-2 (Q-Former)* — 历史范式
- (16.1) Zhai et al. 2023 — *SigLIP (Sigmoid Loss)* — 现代 VLM vision tower
- (16.2) Chameleon Team (Meta) 2024 — *Chameleon (Mixed-Modal Early-Fusion)* — native multimodal 开源代表
- (16.2) van den Oord et al. 2017 — *VQ-VAE* — 离散化奠基
- (16.2) Gemini Team 2024 — *Gemini 1.5* — native + 长 context
- (16.2) Esser et al. 2020 — *VQGAN* — 现代 image tokenizer
- (16.2) OpenAI 2024 — *GPT-4o System Card* (加分)
- (16.2) Lu et al. 2024 — *Janus / Janus-Pro* (加分) — 解耦 vision encoder
- (16.2) Xie et al. 2024 — *Show-o* (加分) — AR + discrete diffusion 统一
- (16.3) Radford et al. 2022 — *Whisper* — ASR 的 BERT 时刻
- (16.3) Wang et al. 2023 — *VALL-E (Neural Codec LM TTS)* — TTS 的 GPT 时刻
- (16.3) Défossez et al. 2024 — *Moshi* — 全双工 speech LLM
- (16.3) Défossez et al. 2022 — *Encodec (High Fidelity Neural Audio Compression)* — audio token 标准
- (16.4) Reimers & Gurevych 2019 — *Sentence-BERT* — sentence embedding 开山
- (16.4) Karpukhin et al. 2020 — *DPR* — dense retrieval 落地
- (16.4) Xiao et al. 2024 — *BGE M3-Embedding*
- (16.4) Wang et al. 2024 — *E5-mistral (Improving Text Embeddings with LLMs)* — LLM as embedding backbone
- (16.4) Lee et al. 2024 — *NV-Embed* — latent attention pooling MTEB SOTA
- (16.4) Muennighoff et al. 2022 — *MTEB* — embedding 选型必读
- (16.4) Su et al. 2023 — *Instructor (One Embedder, Any Task)* — instruction-aware
- (16.5) Anthropic 2024 — *Computer Use blog & docs* — 商业化 desktop agent
- (16.5) Xie et al. 2024 — *OSWorld* — OS-level GUI benchmark
- (16.5) Cheng et al. 2024 — *SeeClick (GUI Grounding)* — grounding 必须显式预训练
- (16.5) Qin et al. 2025 — *UI-TARS (字节)* — 当前开源 GUI agent SOTA
- (16.5) Yang et al. 2023 — *Set-of-Mark Prompting (SoM)* — visual grounding 转换
- (16.5) Wang et al. 2024 — *Mobile-Agent* (加分)
- (16.5) Rawles et al. 2024 — *AndroidWorld* (加分)
- (16.5) Lu et al. 2024 — *OmniParser* (加分)
- (16.5) Hong et al. 2024 — *CogAgent* (加分)

### Appendix

- (A) Radford et al. 2019 — *GPT-2 (Unsupervised Multitask Learners)* — capstone baseline
- (A) Brown et al. 2020 — *GPT-3 (Few-Shot Learners)* — 训练超参标准答案
- (A) Hoffmann et al. 2022 — *Chinchilla* — over-training 解释
- (A) Karpathy — *build-nanogpt + YouTube Let's reproduce GPT-2* — 复现实战参考
- (B) Hu et al. 2021 — *LoRA* — capstone Step 3 配置依据
- (B) Dettmers et al. 2023 — *QLoRA* — 4090 跑通根本原因
- (B) Lewis et al. 2020 — *RAG* — capstone Step 5-6 范式
- (B) Xiao et al. 2024 — *BGE M3-Embedding* — capstone embedding
- (B) Touvron et al. 2023 — *Llama 2* / Meta 2024 — *Llama 3* (选读)
- (B) HuggingFace TRL / PEFT / vLLM Multi-LoRA / RAGAS docs (工程必读)
- (C) Yao et al. 2022 — *ReAct* — worker_loop 范式
- (C) Shinn et al. 2023 — *Reflexion* — 反思机制简化版
- (C) Mialon et al. 2023 — *GAIA Benchmark* — agent 评测事实标准
- (C) Anthropic 2024 — *Building Effective Agents* — orchestrator-worker pattern
- (C) OpenAI 2025 — *Introducing Deep Research* (商业产品参考)
- (C) Perplexity API documentation (商业产品参考)
- (D) Goodfellow, Bengio, Courville 2016 — *Deep Learning* (book, ch.2-4)
- (D) Murphy 2012 — *Machine Learning: A Probabilistic Perspective* (book)
- (D) Deisenroth, Faisal, Ong 2020 — *Mathematics for Machine Learning* (free book)
- (D) 3Blue1Brown — *Essence of Linear Algebra / Calculus / NN* (YouTube)

---

## 完整索引（按主题精细分组）

### 1.1 Architecture - Transformer 原典与变体

- Vaswani et al. 2017 — *Attention is All You Need*
- Xiong et al. 2020 — *On Layer Normalization in Transformer*
- Wang et al. 2022 — *DeepNet (Scaling to 1000 Layers)*
- Touvron et al. 2023 — *LLaMA / LLaMA-2*
- Brown et al. 2020 — *GPT-3*
- Karpathy — *nanoGPT / build-nanogpt*

### 1.2 Architecture - Attention Variants (FlashAttention / MLA / GQA)

- Dao et al. 2022 — *FlashAttention 1*
- Dao 2023 — *FlashAttention 2*
- Shah et al. 2024 — *FlashAttention 3*
- Milakov & Gimelshein 2018 — *Online Softmax*
- Shazeer 2019 — *MQA (Fast Transformer Decoding)*
- Ainslie et al. 2023 — *GQA*
- DeepSeek-AI 2024 — *DeepSeek-V2 (MLA)*
- Clark et al. 2019 — *What Does BERT Look At?*
- Michel et al. 2019 — *Are Sixteen Heads Really Better than One?*

### 1.3 Architecture - Position Encoding / Norm / FFN

- Shaw, Uszkoreit & Vaswani 2018 — *Self-Attention with Relative PE*
- Su et al. 2021 — *RoFormer (RoPE)*
- Press, Smith & Lewis 2021 — *ALiBi*
- Chen et al. 2023 — *Position Interpolation*
- Peng et al. 2023 — *YaRN*
- Ding et al. 2024 — *LongRoPE*
- Ba, Kiros & Hinton 2016 — *Layer Normalization*
- Zhang & Sennrich 2019 — *RMSNorm*
- Hendrycks & Gimpel 2016 — *GELU*
- Shazeer 2020 — *GLU Variants (SwiGLU)*

### 1.4 Architecture - MoE / SSM / 替代方案

- Shazeer et al. 2017 — *Sparsely-Gated MoE*
- Lepikhin et al. 2020 — *GShard*
- Fedus et al. 2022 — *Switch Transformer*
- Jiang et al. 2024 — *Mixtral of Experts*
- Dai et al. 2024 — *DeepSeek-MoE*
- Gu, Goel & Ré 2021 — *S4*
- Gu & Dao 2023 — *Mamba*
- Dao & Gu 2024 — *Mamba-2*
- Lieber et al. 2024 — *Jamba*
- Peng et al. 2023 — *RWKV* / Sun et al. 2023 — *RetNet*

### 2.1 Pretraining - Data

- Gao et al. 2020 — *The Pile*
- Lee et al. 2021 — *Deduplicating Training Data Makes LMs Better*
- Penedo et al. 2024 — *FineWeb*
- Li et al. 2024 — *DataComp-LM*
- Soldaini et al. 2024 — *Dolma*
- Rae et al. 2021 — *Gopher MassiveText filter*
- Xie et al. 2023 — *DoReMi*

### 2.2 Pretraining - Scaling Laws / Objectives

- Kaplan et al. 2020 — *Scaling Laws for Neural LM*
- Hoffmann et al. 2022 — *Chinchilla*
- Sardana et al. 2024 — *Beyond Chinchilla-Optimal*
- Wei et al. 2022 — *Emergent Abilities*
- Schaeffer et al. 2023 — *Are Emergent Abilities a Mirage?*
- McCandlish et al. 2018 — *Empirical Model of Large-Batch Training*
- Smith et al. 2017 — *Don't Decay the LR, Increase the Batch Size*
- Devlin et al. 2018 — *BERT (MLM)*
- Radford et al. 2018/2019 — *GPT-1 / GPT-2 (CLM)*
- Raffel et al. 2020 — *T5 (span corruption)*
- Bavarian et al. 2022 — *FIM*
- Tay et al. 2022 — *UL2*
- Du et al. 2022 — *GLM*
- Wang et al. 2022 — *What LM Architecture & Pretraining Objective?*

### 2.3 Pretraining - LLM Tech Reports

- Touvron et al. 2023 — *LLaMA / LLaMA-2*
- Dubey et al. 2024 — *Llama 3 Herd*
- Yang et al. 2024 — *Qwen2 / Qwen2.5*
- DeepSeek-AI 2024 — *DeepSeek-V2 / V3*
- Warner et al. 2024 — *ModernBERT*

### 3.1 Post-training - SFT / Data

- Wang et al. 2022 — *Self-Instruct*
- Zhou et al. 2023 — *LIMA*
- Xu et al. 2023 — *WizardLM (Evol-Instruct)*
- Xu et al. 2024 — *Magpie*
- Lambert et al. 2024 — *Tülu 3*
- Wei et al. 2022 — *FLAN*

### 3.2 Post-training - PEFT (LoRA / QLoRA / Adapter / Prefix)

- Hu et al. 2021 — *LoRA*
- Dettmers et al. 2023 — *QLoRA*
- Liu et al. 2024 — *DoRA*
- Aghajanyan et al. 2020 — *Intrinsic Dimensionality*
- Houlsby et al. 2019 — *Adapter*
- Li & Liang 2021 — *Prefix-Tuning*
- Liu et al. 2022 — *IA³ / P-Tuning v2*
- Lester et al. 2021 — *Prompt Tuning*
- Zhao et al. 2024 — *GaLore*

### 3.3 Post-training - RLHF / DPO / GRPO / 偏好

- Christiano et al. 2017 — *Deep RL from Human Preferences*
- Ouyang et al. 2022 — *InstructGPT*
- Bai et al. 2022 — *RLHF (HH)*
- Bai et al. 2022 — *Constitutional AI*
- Lee et al. 2023 — *RLAIF (Google)*
- Yuan et al. 2024 — *Self-Rewarding LMs*
- Cui et al. 2024 — *UltraFeedback*
- Liu et al. 2024 — *Skywork-Reward* / Wang et al. 2024 — *ArmoRM*
- Schulman et al. 2015 — *TRPO*
- Schulman et al. 2017 — *PPO*
- Engstrom et al. 2020 — *Implementation Matters*
- Sutton & Barto 2018 — *RL Introduction*
- Schulman et al. 2015 — *GAE*
- Mnih et al. 2016 — *A2C/A3C*
- Williams 1992 — *REINFORCE*
- Rafailov et al. 2023 — *DPO*
- Azar et al. 2023 — *IPO*
- Ethayarajh et al. 2024 — *KTO*
- Meng et al. 2024 — *SimPO*
- Hong et al. 2024 — *ORPO*
- Ahmadian et al. 2024 — *RLOO*
- Tang et al. 2024 — *Generalized Preference Optimization*
- Shao et al. 2024 — *DeepSeekMath (GRPO)*
- Yu et al. 2024 — *DAPO*
- Liu et al. 2024 — *Dr.GRPO*
- Skalse et al. 2022 — *Defining Reward Hacking*
- Singhal et al. 2023 — *Length Correlations in RLHF*
- Coste et al. 2023 — *RM Ensembles Mitigate Overoptimization*
- Sharma et al. 2023 — *Sycophancy*
- Gao et al. 2023 — *Scaling Laws for RM Overoptimization*

### 3.4 Reasoning - CoT / PRM / RLVR / Search

- Wei et al. 2022 — *CoT Prompting*
- Wang et al. 2022 — *Self-Consistency*
- Yao et al. 2023 — *Tree of Thoughts*
- Shinn et al. 2023 — *Reflexion*
- Madaan et al. 2023 — *Self-Refine*
- Kojima et al. 2022 — *Zero-Shot Reasoners*
- Lightman et al. 2023 — *Let's Verify Step by Step (PRM800K)*
- Wang et al. 2024 — *Math-Shepherd*
- Wang et al. 2024 — *OmegaPRM*
- Cobbe et al. 2021 — *GSM8K*
- DeepSeek-AI 2024 — *DeepSeek-Math*
- DeepSeek-AI 2025 — *DeepSeek-R1*
- OpenAI 2024 — *o1 system card*
- Pan et al. 2025 — *TinyZero*
- HuggingFace 2025 — *Open-R1*
- Zhou et al. 2023 — *LATS*
- Qi et al. 2024 — *rStar*

### 4.1 Infra - Parallel (DP / TP / PP / SP / EP / CP)

- Rajbhandari et al. 2020 — *ZeRO*
- Zhao et al. 2023 — *PyTorch FSDP*
- Goyal et al. 2017 — *Accurate Large Minibatch SGD*
- Ren et al. 2021 — *ZeRO-Infinity*
- Shoeybi et al. 2019 — *Megatron-LM*
- Huang et al. 2019 — *GPipe*
- Narayanan et al. 2021 — *Megatron 3D Parallel*
- Qi et al. 2024 — *Zero Bubble Pipeline Parallelism*
- Korthikanti et al. 2022 — *Megatron-SP*
- Liu et al. 2023 — *Ring Attention*
- Brandon et al. 2023 — *Striped Attention*
- Jacobs et al. 2023 — *DeepSpeed Ulysses*

### 4.2 Infra - Mixed Precision / Memory / Kernel

- Micikevicius et al. 2017 — *Mixed Precision Training*
- Micikevicius et al. 2022 — *FP8 Formats*
- DeepSeek-AI 2024 — *DeepSeek-V3 Tech Report*
- Chen et al. 2016 — *Sublinear Memory Cost*
- Tillet, Kung, Cox 2019 — *Triton*

### 4.3 Inference - Engines / KV / Speculative / Quantization

- Pope et al. 2022 — *Efficiently Scaling Transformer Inference*
- Patel et al. 2023 — *Splitwise*
- Agrawal et al. 2024 — *SARATHI (Chunked Prefill)*
- Kwon et al. 2023 — *PagedAttention / vLLM*
- Yu et al. 2022 — *Orca (Continuous Batching)*
- Zheng et al. 2024 — *SGLang (RadixAttention)*
- Dong et al. 2024 — *XGrammar*
- Willard & Louf 2023 — *Outlines*
- Frantar et al. 2022 — *GPTQ*
- Lin et al. 2023 — *AWQ*
- Xiao et al. 2022 — *SmoothQuant*
- Dettmers et al. 2022 — *LLM.int8()*
- Tseng et al. 2024 — *QuaRot*
- Leviathan et al. 2023 — *Speculative Decoding*
- Chen et al. 2023 — *Speculative Sampling (DeepMind)*
- Cai et al. 2024 — *Medusa*
- Li et al. 2024 — *EAGLE / EAGLE-2 / EAGLE-3*
- Stern, Shazeer, Uszkoreit 2018 — *Blockwise Parallel Decoding*
- Saxena 2023 — *Prompt Lookup Decoding*

### 5.1 Agent - Paradigms / Frameworks

- Yao et al. 2022 — *ReAct*
- Wei et al. 2022 — *CoT*
- Shinn et al. 2023 — *Reflexion*
- Wang et al. 2024 — *Survey on LLM-based Agents*
- Anthropic 2024 — *Building Effective Agents*
- Wu et al. 2023 — *AutoGen*
- Hong et al. 2023 — *MetaGPT*
- Shen et al. 2023 — *HuggingGPT*
- Li et al. 2023 — *CAMEL*
- Qian et al. 2023 — *ChatDev*
- Du et al. 2023 — *Multiagent Debate*

### 5.2 Agent - Tool Use / Training

- Schick et al. 2023 — *Toolformer*
- Patil et al. 2023 — *Gorilla*
- Qin et al. 2023 — *ToolLLM (DFSDT)*
- Liu et al. 2024 — *ToolACE*
- Liu et al. 2024 — *xLAM*
- Liu et al. 2024 — *APIGen*
- Yang et al. 2024 — *Qwen2.5 Tech Report*
- Yan et al. 2024 — *BFCL v1/v2/v3*
- OpenAI Function Calling 文档
- Dong et al. 2024 — *XGrammar*

### 5.3 Agent - Planning / Search / Memory

- Wang et al. 2023 — *Plan-and-Solve*
- Liu et al. 2023 — *LLM+P*
- Yao et al. 2023 — *Tree of Thoughts*
- Zhou et al. 2023 — *LATS*
- Liang et al. 2022 — *Code as Policies*
- Wang et al. 2023 — *Voyager*
- Packer et al. 2023 — *MemGPT*
- Park et al. 2023 — *Generative Agents*

### 5.4 Agent - SFT (Trajectory / Tool / Multi-task)

- Chen et al. 2023 — *FireAct*
- Chen et al. 2024 — *Agent-FLAN*
- Zeng et al. 2023 — *AgentTuning*
- Gou et al. 2023 — *ToRA*

### 5.5 Agent - RL (Multi-turn / RLVR / Real-env / Reasoning)

- Shao et al. 2024 — *DeepSeekMath (GRPO)*
- DeepSeek-AI 2025 — *DeepSeek-R1*
- Hou et al. 2025 — *Search-R1*
- Yu et al. 2024 — *DAPO*
- Pan et al. 2025 — *SWE-RL*
- Wang et al. 2025 — *Agent-R1*
- Chen et al. 2025 — *ReSearch*
- Feng et al. 2025 — *ReTool*
- Nakano et al. 2021 — *WebGPT*
- Zhou et al. 2023 — *WebArena*
- Xie et al. 2024 — *OSWorld*
- Jimenez et al. 2023 — *SWE-bench*
- Deng et al. 2023 — *Mind2Web*

### 5.6 Agent - Robustness / Safety / Eval

- Yao et al. 2024 — *τ-bench*
- Guo et al. 2024 — *StableToolBench*
- Liu et al. 2024 — *Towards Robust Tool Use (Adversarial Eval)*
- Wu et al. 2024 — *Adversarial Attacks on LLM Agents*
- Mialon et al. 2023 — *GAIA*
- Pack-Coupling / Fission-GRPO 类 (2025-2026)

### 6.1 Multimodal - VLM / Native

- Radford et al. 2021 — *CLIP*
- Liu et al. 2023 — *LLaVA / LLaVA-1.5*
- Bai et al. 2024 — *Qwen2-VL*
- Chen et al. 2024 — *InternVL*
- Li et al. 2023 — *BLIP-2*
- Zhai et al. 2023 — *SigLIP*
- Chameleon Team 2024 — *Chameleon*
- van den Oord et al. 2017 — *VQ-VAE*
- Esser et al. 2020 — *VQGAN*
- Gemini Team 2024 — *Gemini 1.5*
- OpenAI 2024 — *GPT-4o System Card*
- Lu et al. 2024 — *Janus / Janus-Pro*
- Xie et al. 2024 — *Show-o*

### 6.2 Multimodal - Audio / Speech

- Radford et al. 2022 — *Whisper*
- Wang et al. 2023 — *VALL-E*
- Défossez et al. 2024 — *Moshi*
- Défossez et al. 2022 — *Encodec*

### 6.3 Multimodal - GUI / Computer Use

- Anthropic 2024 — *Computer Use*
- Xie et al. 2024 — *OSWorld*
- Cheng et al. 2024 — *SeeClick*
- Qin et al. 2025 — *UI-TARS*
- Yang et al. 2023 — *Set-of-Mark Prompting*
- Wang et al. 2024 — *Mobile-Agent*
- Rawles et al. 2024 — *AndroidWorld*
- Lu et al. 2024 — *OmniParser*
- Hong et al. 2024 — *CogAgent*

### 6.4 RAG / Embedding

- Lewis et al. 2020 — *RAG*
- Karpukhin et al. 2020 — *DPR*
- Khattab & Zaharia 2020 — *ColBERT*
- Reimers & Gurevych 2019 — *Sentence-BERT*
- Xiao et al. 2024 — *BGE M3-Embedding*
- Wang et al. 2024 — *E5-mistral*
- Lee et al. 2024 — *NV-Embed*
- Su et al. 2023 — *Instructor*
- Muennighoff et al. 2022 — *MTEB*
- Es et al. 2023 — *RAGAS*
- Gao et al. 2022 — *HyDE*
- Asai et al. 2023 — *Self-RAG*
- Yan et al. 2024 — *CRAG*
- Edge et al. 2024 — *GraphRAG*
- Hou et al. 2025 — *Search-R1*

### 7.1 Evaluation - Capability / Code / Math / IF

- Hendrycks et al. 2020 — *MMLU*
- Wang et al. 2024 — *MMLU-Pro*
- Cobbe et al. 2021 — *GSM8K*
- Chen et al. 2021 — *HumanEval*
- Jain et al. 2024 — *LiveCodeBench*
- Jimenez et al. 2023 — *SWE-bench*
- Zhou et al. 2023 — *IFEval*
- Mialon et al. 2023 — *GAIA*

### 7.2 Evaluation - LLM-as-Judge / Arena / RM

- Zheng et al. 2023 — *MT-Bench / Chatbot Arena*
- Wang et al. 2023 — *LLMs are not Fair Evaluators*
- Li et al. 2024 — *Arena-Hard Auto*
- Dubois et al. 2024 — *Length-Controlled AlpacaEval*
- Lambert et al. 2024 — *RewardBench*

### 7.3 Safety / Red Teaming / Robustness

- Mazeika et al. 2024 — *HarmBench*
- Zou et al. 2023 — *GCG*
- Anil et al. 2024 — *Many-shot Jailbreaking*
- Inan et al. 2023 — *Llama Guard*
- Sharma et al. 2025 — *Constitutional Classifier*
- Röttger et al. 2024 — *XSTest*

### 7.4 LLMOps / Engineering

- Sculley et al. 2015 — *Hidden Technical Debt in ML Systems*
- OpenTelemetry GenAI Semantic Conventions
- Langfuse / LangSmith Cookbook

### 8. NLP 经典 / DL 基础

- Rumelhart, Hinton & Williams 1986 — *Backprop (Nature)*
- Baydin et al. 2017 — *AD Survey (JMLR)*
- Griewank 2000 — *Evaluating Derivatives*
- Kingma & Ba 2014 — *Adam*
- Loshchilov & Hutter 2017 — *AdamW*
- Chen et al. 2023 — *Lion*
- Hinton et al. 2012/2014 — *Dropout*
- Ioffe & Szegedy 2015 — *Batch Normalization*
- Goodfellow, Bengio, Courville 2016 — *Deep Learning* (book)
- Murphy 2012 — *ML: A Probabilistic Perspective* (book)
- Deisenroth, Faisal, Ong 2020 — *Math for ML*
- van den Oord et al. 2018 — *CPC (InfoNCE)*
- Szegedy et al. 2015 — *Inception (label smoothing 起源)*
- Hochreiter & Schmidhuber 1997 — *LSTM*
- Cho et al. 2014 — *GRU + Encoder-Decoder*
- Pascanu, Mikolov & Bengio 2013 — *Difficulty of training RNNs*
- Karpathy 2015 — *Unreasonable Effectiveness of RNNs*
- Sutskever, Vinyals & Le 2014 — *Seq2Seq*
- Bahdanau, Cho & Bengio 2014 — *NMT (Attention)*
- Luong, Pham & Manning 2015 — *Effective Attention NMT*
- Mikolov et al. 2013 — *word2vec*
- Pennington, Socher & Manning 2014 — *GloVe*
- Bojanowski et al. 2017 — *fastText*
- Peters et al. 2018 — *ELMo*
- Wang et al. 2018 — *GLUE*
- Sennrich, Haddow & Birch 2016 — *BPE*
- Kudo & Richardson 2018 — *SentencePiece*
- Kudo 2018 — *Subword Regularization (Unigram LM)*
- Schuster & Nakajima 2012 — *WordPiece*

---

## 阅读建议

按本教程的学习路径，paper 推荐阅读顺序。

### 初学者（精读 abstract + intro + conclusion，~15 篇）

1. **Vaswani 2017 — Transformer** — 一切的起点
2. **Devlin 2018 — BERT** — encoder-only
3. **Radford 2019 — GPT-2** — decoder-only LM 哲学
4. **Brown 2020 — GPT-3** — in-context learning
5. **Touvron 2023 — LLaMA-2** — 现代工业 LLM 范本（架构 + RLHF + SFT 一站式）
6. **Hoffmann 2022 — Chinchilla** — scaling law 必背
7. **Hu 2021 — LoRA** — PEFT 起手式
8. **Ouyang 2022 — InstructGPT** — RLHF 三段式
9. **Rafailov 2023 — DPO** — 简化 RLHF
10. **Wei 2022 — CoT** — reasoning 鼻祖
11. **Yao 2022 — ReAct** — Agent 开山
12. **Lewis 2020 — RAG** — RAG 概念原典
13. **Anthropic 2024 — Building Effective Agents** — agent 工程白皮书
14. **Kwon 2023 — vLLM/PagedAttention** — production 推理基础
15. **Karpathy — nanoGPT + build-nanogpt** — 不是 paper 但价值堪比

### 进阶（精读全文 + 跑代码，~25 篇）

1. **Su 2021 — RoFormer (RoPE)** — 公式推一遍
2. **Shazeer 2020 — GLU Variants (SwiGLU)** — 5 页 ablation
3. **Ainslie 2023 — GQA** — KV cache 压缩
4. **DeepSeek-AI 2024 — DeepSeek-V2 (MLA) / V3 Tech Report** — 2024 必精读
5. **Dao 2022/2023 — FlashAttention 1/2** — IO-aware 算法-硬件协同
6. **Rajbhandari 2020 — ZeRO** — 分布式训练范式
7. **Shoeybi 2019 — Megatron-LM** — TP 范式
8. **Korthikanti 2022 — Megatron-SP** — Sequence Parallel + selective recompute
9. **Micikevicius 2017 — Mixed Precision** — fp16/bf16/fp8 体系
10. **Schulman 2017 — PPO** — RLHF 算法基础
11. **Shao 2024 — DeepSeekMath (GRPO)** — 当代 RL 算法骨架
12. **DeepSeek-AI 2025 — DeepSeek-R1** — RLVR + long-CoT 圣经
13. **Yu 2024 — DAPO** — verl 算法核心
14. **Lightman 2023 — Let's Verify Step by Step (PRM800K)**
15. **Hou 2025 — Search-R1** — agent RL 落地参考
16. **Liu 2024 — xLAM** — agent SFT 工业最完整 recipe
17. **Park 2023 — Generative Agents** — episodic memory 模板
18. **Packer 2023 — MemGPT** — long-term memory 系统
19. **Bai 2024 — Qwen2-VL** — 现代 VLM 范本
20. **Chameleon 2024** — native multimodal 开源
21. **Radford 2022 — Whisper** — ASR 现代化
22. **Xiao 2024 — BGE M3-Embedding** — 多语 RAG 主力
23. **Asai 2023 — Self-RAG** + **Yan 2024 — CRAG** — 自适应 RAG
24. **Zheng 2024 — SGLang (RadixAttention)** — agent 时代推理引擎
25. **Leviathan 2023 — Speculative Decoding** + **EAGLE** — 推理加速 SOTA

### 前沿（追踪 2024-2026 最新，~15 篇）

1. **DeepSeek-AI 2024 — DeepSeek-V3 Technical Report** (fp8 + MoE infra)
2. **DeepSeek-AI 2025 — DeepSeek-R1** (RLVR 圣经)
3. **Yu 2024 — DAPO** (verl + dynamic sampling + clip-higher)
4. **Pan 2025 — TinyZero** (R1-Zero $30 复现)
5. **Hou 2025 — Search-R1** (multi-turn agent RL 第一例)
6. **Pan 2025 — SWE-RL** (Meta SWE-bench RL 工业 scale)
7. **Chen 2025 — ReSearch** (multi-hop QA SOTA)
8. **Feng 2025 — ReTool** (tool composition RL)
9. **Yao 2024 — τ-bench** (agent consistency 评测标杆)
10. **Lambert 2024 — Tülu 3** (SFT+DPO+RLVR 完整开源)
11. **Penedo 2024 — FineWeb** (现代开源 corpus)
12. **Sardana 2024 — Beyond Chinchilla-Optimal** (over-training scaling)
13. **Qin 2025 — UI-TARS (字节)** (开源 GUI agent SOTA)
14. **Bai 2024 — Qwen2-VL / Qwen2.5-VL** (动态分辨率)
15. **Edge 2024 — GraphRAG (Microsoft)** + **Hou 2025 — Search-R1** (RAG 下一代)

### 面试高频（必须能讲清楚 motivation + trade-off + 公式）

- **Vaswani 2017** (attention 公式 / multi-head 设计 / mask 时机)
- **Su 2021 RoPE** (复指数推导 + 为什么不外推 + PI/YaRN 怎么修)
- **Ainslie 2023 GQA** + **DeepSeek MLA** (KV cache 压缩 trade-off)
- **Hoffmann 2022 Chinchilla** ($D \approx 20N$ + IsoFLOP 三角验证)
- **Hu 2021 LoRA** + **Dettmers QLoRA** (低秩假设 + NF4 + paged optim)
- **Rafailov 2023 DPO** (从 PPO 推导 closed-form RM)
- **Schulman 2017 PPO** (importance sampling + clip + KL constraint)
- **Shao 2024 GRPO** (组内归一化 + k3 KL + 去 critic)
- **DeepSeek-R1** (RLVR + GRPO + 4 阶段 pipeline + Aha moment)
- **Yao 2022 ReAct** (Thought-Action-Observation loop)
- **Kwon 2023 PagedAttention** (KV cache 碎片 + block table)
- **Dao FlashAttention** (online softmax + IO-aware)
- **Rajbhandari ZeRO** (三阶段分片 + 通信代价)
