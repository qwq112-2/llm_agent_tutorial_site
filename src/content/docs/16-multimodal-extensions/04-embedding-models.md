---
title: "16.4 Embedding：bge / E5 / Instructor / NV-Embed"
description: "把 13.2 里\"调一个 embedding model 当 retriever\"这件事拆开——讲清现代 sentence/passage embedding 的 训练范式（contrastive + hard negative + InfoNCE）、两条技术路线（encoder-only 的 bge / jina vs LLM-based 的 E5-mistral / NV-Embed）、Ins"
---

> ⏱ 预计阅读 30 分钟 ｜ 难度 ★★ ｜ 前置：13.2（RAG pipeline 与 embedding 在其中的位置）、1.4（InfoNCE）、2.2（word2vec 时代的 static embedding）

## 一句话本节讲什么

把 13.2 里"调一个 embedding model 当 retriever"这件事拆开——讲清现代 sentence/passage embedding 的 **训练范式（contrastive + hard negative + InfoNCE）**、**两条技术路线（encoder-only 的 bge / jina vs LLM-based 的 E5-mistral / NV-Embed）**、**Instructor 范式带来的 prompt-aware embedding**、以及 **MTEB 选型与中英文实战决策**——读完知道为什么 bge-m3 是中英 RAG 默认、为什么 NV-Embed-v2 在 MTEB 长期 SOTA 但工业里很少用。

---

## 1. Mental model（直觉）

13.2 节把 embedding model 当作 RAG 流水线里一个 **黑盒函数** $f_\theta: \text{text} \to \mathbb{R}^d$ 用——只关心输入文本输出向量。本节把这个黑盒拆开。

回到 2.2 节末尾留的伏笔：word2vec 时代的 embedding 是 **word-level static** 的，每个词查表得一个固定向量。BERT/GPT 之后的 LLM 把它升级到 **token-level contextual**——同一 token 在不同句子里得到不同的 hidden state。但 RAG 要的是 **sentence/passage-level** 的固定向量：

- 整段 chunk 进去，出来 **一个** 向量（不是一串 hidden state）
- 同一段文本（不依赖 query）总是同一个向量（要能离线缓存到 vector DB）
- 与 query embedding 的 cosine 相似度要直接反映语义相关性

把 BERT 这种 contextual encoder 改造成一个 sentence embedding model 的最朴素思路是 **Sentence-BERT**（Reimers 2019）：

```
       原版 BERT 用法                       Sentence-BERT 用法（双塔）
                                       
   [CLS] sent A [SEP] sent B [SEP]      query → BERT → mean-pool → e_q
              ↓                          doc   → BERT → mean-pool → e_d
        cross-attention                          ↓
              ↓                              cosine(e_q, e_d)
        相似度 0~1
        
    query × N_doc 次 forward            离线：N_doc 次 forward 缓存
    每次 query 都要 N_doc forward        在线：1 次 query forward + ANN 查询
       O(N) 时延                            O(log N) 时延
```

但光把 BERT 拿来 mean-pool 一下不够——BERT 预训练目标（MLM）只让相邻 token 的表示接近,不保证 **整句语义相似的两段文本**embedding 靠近。要让 cosine 真正反映语义,必须 **再训一遍**——给定一堆 `(query, relevant doc, irrelevant doc)` 三元组,用 **对比学习** 显式拉近正样本对、推远负样本对。这条路从 2019 Sentence-BERT 走到 2024 NV-Embed,核心 loss 始终是 1.4 节那个 InfoNCE,变的只是 **backbone 越用越大**（300M BERT → 7B Mistral）、**训练数据越来越精**（早期手标 NLI → 现代 GPT-4 合成 + 多阶段 fine-tune）、**negative mining 越来越狠**（in-batch → mined hard negatives → cross-encoder 蒸馏）。

直觉上,embedding model 就是 "**用对比学习把 LLM 的 hidden space 重新整形,让 cosine 距离 = 语义距离**" 的产物。整形 ≠ 重训——多数现代 embedding 都是在已有 LLM checkpoint 上做 lightweight fine-tune,而不是从头预训练。

---

## 2. 公式与原理

### 2.1 训练目标：InfoNCE 与 hard negative

给定 batch 里的 query-positive 对 $(q_i, d_i^+)$,以及 $N-1$ 个负样本 $\{d_{i,j}^-\}_{j=1}^{N-1}$,InfoNCE loss（与 1.4 节同公式）：

$$\mathcal{L}_{\text{InfoNCE}} = -\frac{1}{B} \sum_{i=1}^{B} \log \frac{\exp(\text{sim}(q_i, d_i^+) / \tau)}{\exp(\text{sim}(q_i, d_i^+) / \tau) + \sum_{j=1}^{N-1} \exp(\text{sim}(q_i, d_{i,j}^-) / \tau)}$$

其中 $\text{sim}(\cdot, \cdot)$ 通常是 cosine（先 L2 normalize 再做内积）,$\tau$ 是 temperature(典型 0.01-0.05)、$B$ 是 batch size。

直觉是 **softmax 分类**:把 "$d_i^+$ 是 $q_i$ 的相关文档" 当成正确类别,"$d_{i,j}^-$ 是相关文档" 当成错误类别——loss 让 $q_i$ 与 $d_i^+$ 的相似度在 normalize 后接近 1。

**温度 $\tau$ 的作用**:$\tau$ 越小,loss 对相似度差异越敏感——同样把正样本相似度从 0.7 推到 0.8,$\tau=0.01$ 时 loss 从 ~70 降到 ~10,$\tau=1$ 时 loss 几乎不变。**$\tau$ 太大模型学不动,$\tau$ 太小训练不稳**——bge / E5 系经验值都在 $0.01\sim 0.05$。

#### 负样本从哪来

InfoNCE 的性能 95% 由负样本质量决定:

- **In-batch negatives**(最便宜):batch size $B$ 时,每个 query 用其他 $B-1$ 个样本的 positive doc 当负样本——零成本拿到 $B-1$ 个 negative。但这些 negative 多数 **太容易**(从全语料里随机抽,与 query 主题完全无关),loss 早早降到底但下游检索依然弱。
- **Hard negatives**(关键):用一个 retriever(可以是上一版 embedding model、BM25、甚至随机初始化的 base model) 给 $q_i$ 召回 top-K 候选,挑出 **跟 $q_i$ 看着相关但实际不是答案** 的那些当 hard negative。这些是模型最容易混淆的负样本,加进 batch 后 loss 立刻变难,模型才会学到细粒度的相关性区分。
- **False negative 过滤**:hard negative 里混入"其实是正样本但没标注"的 doc 会把模型往反方向拉。常用做法是 **跳过 top-N**(比如召回 top-200,只取 rank 11-200 当 hard negative,top-10 大概率混入正样本就跳过)、或用 cross-encoder 打分剔除高分项。

实际工业训练 batch 长这样:**1 query + 1 positive + 8~16 mined hard negatives**,再叠 in-batch 负样本——一个 batch 实际有效负样本可达数百。

### 2.2 Pooling 策略:[CLS] / mean / last token

把 transformer 输出的一串 hidden state $\{h_1, h_2, \dots, h_L\} \in \mathbb{R}^{L \times d}$ 压成一个 fixed-size vector 有几种主流做法:

| 策略 | 公式 | 适用 backbone |
|---|---|---|
| **[CLS] pooling** | $e = h_{\text{[CLS]}}$ | BERT 系(encoder-only) |
| **Mean pooling** | $e = \frac{1}{L} \sum_{l=1}^L h_l$(去 padding) | bge / E5 encoder-only 默认 |
| **Last token pooling** | $e = h_L$ | LLM-based(decoder-only) |
| **Weighted mean** | $e = \sum_l w_l h_l$,$w_l$ 学习或位置加权 | 少数 SOTA 模型 |

**为什么 LLM-based embedding 取 last token?** decoder-only LLM 走 causal mask,只有最后一个 token 能看到全部上文——把 last token hidden state 当 sentence representation 是最自然的选择。E5-mistral / NV-Embed / gte-Qwen 全是这套。代价是 padding 必须左侧对齐(右 pad 会让 last token 是 `<pad>`,直接废掉)。

**为什么不用 [CLS]?** BERT 的 [CLS] 是为 next-sentence prediction 设计的,不天然学到"整句语义";Sentence-BERT 论文实测 mean pooling 比 [CLS] 在 STS 任务上高 4-7 个点。bge 系列默认就用 mean pool。

### 2.3 现代训练 pipeline:四阶段范式

bge / E5-mistral / NV-Embed 这些 SOTA 模型都遵循 **多阶段递进训练**,直接一阶段端到端跑不出 SOTA:

| 阶段 | 数据 | 目标 | 典型规模 |
|---|---|---|---|
| **Stage 1: Backbone 预训练** | 通用语料(C4 / Wikipedia) | MLM 或 next-token | 复用 BERT / Mistral 现成 checkpoint,不必自训 |
| **Stage 2: Weakly-supervised contrastive** | 大规模弱监督 pair 数据(网页 title-body、Reddit post-comment、问答论坛 QA 对) | InfoNCE,in-batch 负样本 | 数十亿 pair,batch 几千~几万 |
| **Stage 3: Supervised contrastive fine-tune** | 高质量标注 retrieval 数据(MS-MARCO / NQ / HotpotQA / TriviaQA) | InfoNCE + mined hard negatives | 百万级 query,batch 数百~数千 |
| **Stage 4: Instruction / multi-task tuning** | 多任务带 instruction 的数据(Instructor / E5 mistral 的合成数据) | 同 InfoNCE,但带 task prefix | 百万级,覆盖 retrieval / classification / clustering / STS 等 |

E5-mistral 把 stage 4 玩到了极致:**用 GPT-4 合成 150 个任务 × 数千 query 的 instruction-aware 数据**,直接绕过传统 retrieval 数据集瓶颈——这是它当年(2024 初)在 MTEB 屠榜的关键。NV-Embed 在 stage 3-4 之间加了一个 **two-stage fine-tune**:先纯 retrieval、再 multi-task 混训,避免后续任务把 retrieval 性能搞坏。

### 2.4 Encoder-only 路线:bge 谱系

BAAI(智源)的 bge(BAAI General Embedding) 是开源 encoder-only embedding 的事实标准:

- **bge-large-en-v1.5**(2023):BERT-large 改造,335M 参数,1024 维,英文 MTEB ~64,长期英文开源 baseline
- **bge-large-zh-v1.5**(2023):中文版,326M,中文 C-MTEB baseline
- **bge-m3**(Chen 2024):**multi-lingual + multi-functionality**,XLM-RoBERTa large backbone,568M,**同时输出三种表示**:
  - **Dense**:1024 维向量,标准 cosine 检索
  - **Sparse(lexical)**:词级别权重(类似 SPLADE),做 BM25-like 检索
  - **Multi-vector(ColBERT-like)**:每个 token 一个向量,做 late interaction 精排
- 训练时三种 head 联合优化,推理时按需取——一个模型替代 dense / sparse / ColBERT 三套流水线

**bge-m3 的工程价值** > MTEB 分数:很多场景 dense + BM25 + reranker 三件套(13.2)用 bge-m3 一个模型全 cover,运维成本砍三分之二。是中英 / 多语 RAG 现在的默认起点。

### 2.5 LLM-based 路线:E5-mistral / NV-Embed / gte-Qwen

2024 年开始大家发现:**用 7B LLM 当 encoder + lightweight contrastive fine-tune,效果远超 300M 的 BERT-style encoder**——直接继承了 LLM 在大规模 next-token 预训练里学到的语义知识。

代表作:

- **E5-mistral-7b-instruct**(Wang 2024,微软):Mistral-7B + LoRA + GPT-4 合成 instruction 数据,4096 维,2024 初 MTEB SOTA(67+)。开源、最早把 "LLM as embedder" 跑通的工作。
- **NV-Embed-v2**(Lee 2024,NVIDIA):Mistral-7B 改造,加 **latent attention pooling**(可学习的 query token 对所有 hidden state 做 cross-attention 聚合,比 last-token pooling 更强),加 **two-stage fine-tune**——MTEB 长期排在 70+ 顶部。开源但训练数据复杂。
- **gte-Qwen2-7B-instruct**(阿里):Qwen2-7B 改造,3584 维,中英双语强,MTEB 70+,中文场景比 NV-Embed 更稳。
- **stella / SFR-Embedding**:其他几个长期混在 MTEB top-10 的 LLM-based 模型,基本同套配方。

**优劣** 一句话:质量上 LLM-based > encoder-only 3-8 个 MTEB 点,但 **推理慢 5-10×、显存占用高 10-20×、向量维度大 4×**。下游 vector DB 存储 / 检索 / 在线 latency 全部跟着膨胀,中小规模业务很少能用得起。

### 2.6 Instructor 范式:instruction-aware embedding

Su et al. 2023 的 **Instructor** 引入了"同一 embedding model 在不同任务上输入不同 instruction prefix,得到不同向量"的范式:

```
普通 embedding:  encode("The capital of France is Paris.")  → fixed vector

Instructor:      encode("Represent the document for retrieval: " + 
                        "The capital of France is Paris.")  → vector A
                 encode("Represent the document for clustering by topic: " +
                        "The capital of France is Paris.")  → vector B
                 # A 和 B 来自同一模型同一段文本,但因 instruction 不同而几何位置不同
```

**为什么有效**:同一段文本在不同任务里"该被强调的语义维度"不一样——retrieval 看具体事实(capital, France, Paris),clustering 看主题(地理 vs 体育),classification 看情感/类别。Instructor 在训练时把 task instruction 和文本一起 tokenize,模型学会根据 instruction 调整 pooling 出来的向量方向。

bge / E5 系把 instruction 简化成 **固定的两个 prefix**:

- query 端:`"query: <q>"` 或 `"Represent this sentence for searching relevant passages: <q>"`
- passage 端:`"passage: <d>"` 或不加(bge-m3 的 passage 端不加)

**漏写 prefix 是新手 #1 错误**,直接掉 5-10 个点 NDCG——而且这种 bug 不报错,只让你怀疑人生。

### 2.7 MTEB:embedding 选型的 leaderboard

**MTEB**(Massive Text Embedding Benchmark, Muennighoff 2022)是当下 embedding model 选型的事实标准:

- **覆盖 56 个数据集 × 8 类任务**:Retrieval / Reranking / Classification / Clustering / Pair Classification / STS / Summarization / Bitext Mining
- **C-MTEB**:中文版 35 个任务
- **MTEB-fr / MTEB-pl** 等多语种衍生
- **leaderboard**:HuggingFace Spaces 实时更新

但 MTEB 高分 ≠ 你的业务好(踩坑会再讲一遍):

- MTEB 的 retrieval 任务多数是**通用语义检索**(MS-MARCO / Natural Questions),你的业务可能是**代码检索**或**金融术语检索**——两者所需的几何完全不同
- LLM-based 模型在 MTEB 上 SOTA,但 latency 是 7-10× 你能扛吗?
- **必须自己用业务数据 benchmark**——拿 100 条真实 query + ground truth doc 测 NDCG@10,比看 MTEB 排行靠谱 100 倍

### 2.8 选型对比表

实战常见模型横向对比(2025 年视角):

| Model | 类型 | size | dim | 中英 | MTEB | 备注 |
|---|---|---|---|---|---|---|
| `bge-large-en-v1.5` | encoder-only | 335M | 1024 | 英 | ~64 | 英文经典 baseline |
| `bge-large-zh-v1.5` | encoder-only | 326M | 1024 | 中 | C-MTEB ~64 | 中文经典 baseline |
| **`bge-m3`** | encoder-only multi-func | 568M | 1024 | 双语 | ~60+ | dense+sparse+ColBERT,中英 RAG 默认 |
| `jina-embeddings-v3` | encoder-only | 570M | 1024(可降 256) | 多语 | ~60+ | 8k context、task-specific LoRA |
| `mGTE-base`(阿里) | encoder-only | 305M | 768 | 多语 | ~60 | 中文 / 多语备选 |
| `E5-mistral-7b` | LLM-based | 7B | 4096 | 双语 | ~67+ | 第一个 LLM-based 屠榜模型 |
| **`NV-Embed-v2`** | LLM-based | 7B | 4096 | 英 | ~72+ | MTEB 长期英文 SOTA |
| `gte-Qwen2-7B` | LLM-based | 7B | 3584 | 双语 | ~70+ | 中英双语 SOTA |
| `text-embedding-3-large`(OpenAI) | API | - | 3072(Matryoshka 可降 256) | 多语 | ~64 | 商用 baseline、最易接入 |
| `voyage-3-large` | API(Voyage AI) | - | 1024 | 多语 | ~70 | RAG 专用商用 SOTA |

口诀:**中英 RAG 起点 bge-m3 → 不够再上 gte-Qwen2-7B 或 text-embedding-3-large → 极致质量上 NV-Embed-v2 但准备好掏 GPU**。

---

## 3. 最小代码示例

### 3.1 bge-m3 调用:dense + sparse + ColBERT 三种输出

```python
# pip install -U FlagEmbedding
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

queries = ["人工智能后训练 RLHF 的关键是什么?"]
docs = [
    "RLHF 的核心是 reward model 与 PPO,本节略。",
    "炒鸡蛋的关键是火候和翻面频率。",
    "DPO 通过闭式解避免显式 reward model。",
]

# 一次 forward 同时产出三种 representation
out_q = model.encode(queries, return_dense=True, return_sparse=True, return_colbert_vecs=True)
out_d = model.encode(docs,    return_dense=True, return_sparse=True, return_colbert_vecs=True)

# 1. Dense:cosine 相似度(标准 RAG 用法)
import numpy as np
dense_sim = out_q["dense_vecs"] @ out_d["dense_vecs"].T   # (1, 3)

# 2. Sparse(lexical):词级权重做 dot product
sparse_sim = model.compute_lexical_matching_score(out_q["lexical_weights"][0], out_d["lexical_weights"])

# 3. ColBERT(multi-vector):每个 query token 取与 doc tokens 的 max similarity 求和
colbert_sim = [model.colbert_score(out_q["colbert_vecs"][0], dv) for dv in out_d["colbert_vecs"]]

# 三路融合通常按 0.4 dense + 0.2 sparse + 0.4 colbert 加权,或走 RRF
print("dense:", dense_sim, "\nsparse:", sparse_sim, "\ncolbert:", colbert_sim)
```

bge-m3 的工程价值在这里一目了然:**一次 forward、三种检索能力**——以前 dense(bge-large) + sparse(SPLADE / BM25) + multi-vector(ColBERT) 要部署三个模型,现在一个搞定。

### 3.2 InfoNCE training core(PyTorch)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class EmbeddingModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder  # 任意 transformer,如 AutoModel.from_pretrained("BAAI/bge-base-en-v1.5")

    def encode(self, input_ids, attention_mask):
        h = self.encoder(input_ids, attention_mask=attention_mask).last_hidden_state  # (B, L, d)
        # mean pooling(屏蔽 padding)
        mask = attention_mask.unsqueeze(-1).float()
        emb = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)                    # (B, d)
        return F.normalize(emb, p=2, dim=-1)  # L2 normalize → 后面内积 = cosine

def info_nce_loss(q_emb, pos_emb, neg_emb, tau=0.02):
    """
    q_emb  : (B, d)         query embeddings
    pos_emb: (B, d)         相应 positive doc embeddings
    neg_emb: (B, K, d)      每个 query 的 K 个 hard negatives
    """
    B, K, d = neg_emb.shape
    # in-batch negatives:每个 query 把其他 query 的 positive 也当负样本
    all_d = torch.cat([pos_emb.unsqueeze(1), neg_emb], dim=1)              # (B, 1+K, d)
    # 跨 batch flatten 让每个 query 看到 B*(1+K) 个候选
    flat_d = all_d.reshape(B * (1 + K), d)                                  # (B*(1+K), d)

    logits = q_emb @ flat_d.T / tau                                         # (B, B*(1+K))
    # 第 i 个 query 的正确答案在 flat_d 里 index = i*(1+K)(每段 1+K 的第 0 个)
    labels = torch.arange(B, device=q_emb.device) * (1 + K)
    return F.cross_entropy(logits, labels)
```

关键 3 行就是 InfoNCE:`logits = q @ d.T / tau` + `cross_entropy(logits, label_of_pos)`。**温度 $\tau$ 直接除在 logits 上**,等价于 $\exp(\cdot/\tau)$ 后再 softmax。`F.normalize` 让内积 = cosine,这是工程实现的标准做法。

### 3.3 Hard negative mining 简化版

```python
import torch, faiss, numpy as np

@torch.no_grad()
def mine_hard_negatives(model, queries, docs, num_negs=8, skip_top=10):
    """
    用当前 model 给每个 query 召回 top-(skip_top + num_negs) 候选,
    跳过最 top 几个(可能混入正样本),取剩下的当 hard negative。
    
    返回 negatives: list of list of doc_id, len=len(queries)
    """
    # 1. encode all docs,建临时索引
    doc_emb = model.encode(docs, normalize_embeddings=True).astype(np.float32)
    index = faiss.IndexFlatIP(doc_emb.shape[1])
    index.add(doc_emb)

    # 2. encode queries,召回 top-K
    q_emb = model.encode(queries, normalize_embeddings=True).astype(np.float32)
    K = skip_top + num_negs
    _, topk_idx = index.search(q_emb, K)  # (Q, K)

    # 3. 跳过 top-skip_top 取 num_negs 个
    return topk_idx[:, skip_top:].tolist()  # 每行长度 num_negs

# 训练循环里每 N 个 epoch 重新 mine 一次:
# 模型变强 → 它召回的 hard negative 也跟着变难 → 课程式越训越强
```

**`skip_top=10` 是 false negative 的简易过滤**——top-10 大概率混入真正的 positive(尤其是当训练数据是从 MS-MARCO 来的、本身标注就稀疏);从 rank 11 开始取相对安全。生产系统会再叠一道 cross-encoder 过滤(用 reranker 给候选打分,score 太高的直接丢弃)。

---

## 4. 工程踩坑与经验

- ❗ **bge / E5 系 query 与 passage 必须加不同 prefix**:bge 中文系列 query 端要 prepend `"为这个句子生成表示以用于检索相关文章:"`(老版本)或新版自动处理;bge-m3 query 端不加但 LongDoc 任务要加;E5 系列 query 端 `"query: "` / passage 端 `"passage: "`。**漏加直接掉 5-10 个点 NDCG**——这种 bug 不报错,只让你怀疑模型不行。每次接新 model 第一件事 cd 到 `model_card.md` 看 prefix 约定,不要凭直觉。
- ❗ **LLM-based embedding 比 encoder-only 慢 5-10×**:7B 模型 forward 一段 512 token chunk 在 A10 上 ~50 ms,bge-large(335M) 只要 ~6 ms。离线建库 100 万 chunk:bge 要 1.5 小时,E5-mistral 要 14 小时——上线前算清楚 throughput 预算,别被 MTEB 排行忽悠到选不起的模型。
- ❗ **Embedding 维度大(4096)→ vector DB 显存压力大**:NV-Embed / E5-mistral 输出 4096 维,1000 万 chunk 就是 160 GB float32 / 80 GB float16——单机 GPU 显存装不下,只能上 disk-based ANN(性能掉一档)或分布式 vector DB(运维成本飙升)。**实战:用 Matryoshka embedding 压到 1024 维**(OpenAI 3-large 与 NV-Embed-v2 都原生支持训练时多维度联合优化的 Matryoshka 切片),质量几乎无损但存储与计算开销 1/4。
- ❗ **Hard negative mining 必须过滤 false negative**:用上一版 model 召回的 top-10 里大概率混了 unlabeled positive(尤其当训练数据是 MS-MARCO 这种稀疏标注的),把它们当 negative 训会 **直接把模型往反方向拉**。简单做法是 skip top-10 / top-20 取 rank 11+ 当 hard negative;生产做法是用一个独立的 cross-encoder reranker 过一遍,score > 阈值的直接丢。
- ❗ **多语言 query 用单语 embedding → 跨语言性能塌**:用 `bge-large-en` 处理中文 query 不会报错,但中文 query → 中文 doc 的 retrieval 几乎随机。**中英混合或多语场景必须用 multilingual model**:bge-m3、jina-v3、multilingual-e5、gte-Qwen 这一类。它们在训练时就把所有语言投到同一向量空间,中英文跨语言 cosine 才有意义。
- ❗ **Cosine similarity 阈值不可移植到不同 model**:每个 embedding model 的 "好相似" 绝对值不同——bge-m3 上 cosine 0.6 可能就算很相关,NV-Embed 上要 0.85 才算。换 model 要 **重新校准阈值**,不要把 prod 里 "score > 0.7 才返回" 这种规则盲移植。校准方法:抽 100 条已知相关 / 不相关的 query-doc pair,看 score 分布的最佳分隔点。
- ❗ **MTEB SOTA ≠ 你的下游任务好**:MTEB 56 任务覆盖广但每类只有几个数据集,你的业务(代码检索 / 法律检索 / 多轮 QA)可能跟 MTEB 训练集分布完全不同。**铁律:自己做 mini-benchmark**——抽 50-100 条真实 query + ground truth doc,跑 NDCG@10 / Recall@5 比 MTEB 排行可信 100 倍。
- ❗ **last token pooling 必须 left padding**:LLM-based embedding 取最后一个非 padding token 的 hidden state,如果用 right padding(BERT 时代默认)last token 就是 `<pad>`,直接废掉。HuggingFace 默认 right pad,接 LLM-based embedding 时记得 `tokenizer.padding_side = "left"`。
- ❗ **batch size 直接决定 in-batch negative 数量**:contrastive 训练的 batch size 比一般 SFT 重要得多——batch 32 时 in-batch negative 只有 31 个,batch 1024 时有 1023 个,后者训出的 embedding 质量明显高。bge / E5 训练时 batch 都开到 数千甚至上万(配合 gradient checkpointing 和小 dim 模型),小公司复现时 GPU 不够会显著掉点。

---

## 5. 经典 paper

- **Reimers & Gurevych, 2019 — Sentence-BERT** — 把 BERT 改造成 siamese / triplet 网络做 sentence embedding 的开山之作,定义了 mean pooling + cosine + STS 评测的标准流程。读 §3-4 理解为什么"BERT 直接拿来 mean-pool"不够好,必须用 NLI / STS 数据再 fine-tune——这是后续所有 sentence/passage embedding 工作的祖师爷。
- **Karpukhin et al., 2020 — Dense Passage Retrieval(DPR)** — dense retrieval 在开放域 QA 上的落地之作,证明 BERT 双塔 + in-batch negative 的对比训练就能吊打 BM25。读 §3 理解 in-batch negative 的具体实现 + 为什么需要 hard negative mining,这是 bge / E5 / NV-Embed 的共同模板。
- **Xiao et al., 2024 — BGE M3-Embedding** — 中英 / 多语开源 embedding 的事实标准,**一个模型同时输出 dense / sparse / multi-vector 三种 representation**,工程价值远超 MTEB 分数。读 §3 看多功能联合训练的 loss 设计——下游做中文 / 多语 RAG 几乎绕不开。
- **Wang et al., 2024 — Improving Text Embeddings with Large Language Models(E5-mistral)** — 第一篇把 LLM(Mistral-7B)当 embedding backbone 跑通的工作,**用 GPT-4 合成 150 个任务的 instruction-aware 数据**直接屠 MTEB。读 §3 看合成数据的 prompt 模板 + §4 的 training recipe,这套做法被后续 NV-Embed / gte-Qwen 全部继承。
- **Lee et al., 2024 — NV-Embed** — NVIDIA 把 LLM-based embedding 推到 MTEB 长期 SOTA(72+),关键创新是 **latent attention pooling**(代替 last-token pooling)和 **two-stage fine-tune**(retrieval 先、multi-task 后,避免 task interference)。读 §3.2 的 pooling 设计——是当前 LLM-based embedding 最 elegant 的一个 trick。
- **Muennighoff et al., 2022 — MTEB: Massive Text Embedding Benchmark** — embedding 选型必读,定义了 8 类任务 56 个数据集的标准化评测协议。不需要细读 paper,看一遍 leaderboard + paper §3 的 task definition 就够——主要是理解 "高 MTEB ≠ 好下游"。
- **Su et al., 2023 — One Embedder, Any Task(Instructor)** — instruction-aware embedding 范式提出,**同一模型 + 不同 instruction prefix → 不同向量**。这套思想被 bge / E5 简化成固定 query/passage prefix 沿用至今。读 §3 的 instruction 设计 + §5 的 ablation,理解为什么 prefix 这么重要。

---

## 6. 自测与面试题

**Q1(公式)**:写出 InfoNCE loss 的公式,并解释为什么 hard negative mining 是 embedding 训练的关键。

<details>
<summary>Answer sketch</summary>

公式:

$$\mathcal{L}_{\text{InfoNCE}} = -\frac{1}{B} \sum_{i=1}^{B} \log \frac{\exp(\text{sim}(q_i, d_i^+) / \tau)}{\exp(\text{sim}(q_i, d_i^+) / \tau) + \sum_{j} \exp(\text{sim}(q_i, d_{i,j}^-) / \tau)}$$

变量:$q_i$ query embedding(先 L2 normalize 到单位向量),$d_i^+$ 对应 positive doc,$d_{i,j}^-$ 第 $j$ 个负样本,$\tau$ 温度(典型 0.01-0.05),$B$ batch size。直觉是 K+1 路 softmax 分类——把 positive 当正确类、negatives 当错误类,推 $q$ 与 $d^+$ 的相似度接近 1。

为什么 hard negative 关键:

- **In-batch negative 太容易**:从全语料随机抽,与 query 主题完全无关——loss 早早降到底但模型只学到"区分主题",学不到"区分细微相关性差异"。下游 retrieval 上稍微难一点的 query 就召不出来。
- **Hard negative 制造区分难度**:用上一版 model 召回的 top-K 里取"看着相关但实际不是答案"的当 negative,逼模型学到细粒度匹配——就像 CV 里 hard example mining,把 model 的弱点暴露出来训。
- **量化效果**:工业经验,加 mined hard negative 后 retrieval NDCG@10 提升 5-15 个点,远超模型规模升级的收益。
- 加分:false negative(unlabeled positive 误当 negative)是常见陷阱,简单做法是 skip top-10 / top-20 跳过最高分,生产做法是用 cross-encoder 过滤。
- 加分:温度 $\tau$ 决定 loss 对相似度差异的敏感度——$\tau$ 太大模型学不动,太小训练不稳,$0.01\sim 0.05$ 是 bge / E5 的经验值。

</details>

**Q2(trade-off)**:你要给团队选 RAG 用的 embedding model,候选是 `bge-m3`(568M, encoder-only, 1024 dim)和 `E5-mistral-7b`(7B, LLM-based, 4096 dim)。从至少 4 个维度做 trade-off 分析,给出选型决策框架。

<details>
<summary>Answer sketch</summary>

四个维度对比:

- **质量**:E5-mistral 在 MTEB 上 ~67+,bge-m3 ~60——E5 强 5-7 个点,主要在 retrieval 难任务上拉开差距。但你的业务任务可能与 MTEB 分布不同,**必须自己 benchmark**。
- **延迟 / 吞吐**:bge-m3 在 A10 forward 一段 512 token chunk ~10 ms,E5-mistral ~50-80 ms——慢 5-8×。在线 query encoding 影响 P99,离线建库影响时间预算(100 万 chunk:bge 1.5 小时 vs E5 14 小时)。
- **存储 / 显存**:bge-m3 输出 1024 维,E5-mistral 4096 维——同样 1000 万 chunk,前者占 40 GB float32 / 20 GB float16,后者 160 GB / 80 GB。vector DB 显存装不下就要走 disk-based ANN(性能掉一档)或分布式(运维成本飙升)。
- **多功能**:bge-m3 一个模型同时输出 dense + sparse + ColBERT,可以在一处替代 dense + BM25 + reranker 三件套。E5-mistral 只输出 dense,要做 hybrid 必须再跑独立的 BM25 / SPLADE。
- **多语言**:bge-m3 是 multilingual XLM-R 训出来,中英表现接近;E5-mistral 训练数据以英文为主,中文表现明显弱(虽然支持但远不如中英混合训练)。
- **部署成本**:bge-m3 单 A10 / L4 就能跑得很爽,E5-mistral 必须 A100 / H100,GPU 成本 5-10×。

决策框架:

- **默认 bge-m3**——如果 SLA 不要求 P99 < 100ms / 不需要 MTEB 顶级质量 / 中文场景:bge-m3 几乎无脑选。
- **升 E5-mistral 的条件**:1) 业务对 retrieval 质量极端敏感(法律 / 医疗); 2) 已经验证 bge-m3 在自己业务的 NDCG 不够; 3) GPU 预算充足、可接受 5-10× cost。
- **加分**:也可考虑 `gte-Qwen2-7B-instruct`(中英比 E5-mistral 强)或 OpenAI `text-embedding-3-large`(API,免运维,Matryoshka 可降维)。

</details>

**Q3(实战)**:让你为一个 **中英文混合的企业知识库 RAG** 做 embedding 选型,数据规模 500 万 chunk,要求 P99 query latency < 200 ms,业务方接受 GPU 成本但不接受 OpenAI 等出海 API。请给出 1) embedding model 选型 + 维度;2) 是否做 dimension reduction;3) 完整 retrieval pipeline 设计;4) 验证选型的 mini-benchmark 流程。

<details>
<summary>Answer sketch</summary>

**1) Embedding model 选型**:

- 主选 **bge-m3**(568M, 1024 dim, multilingual, encoder-only):
  - 中英表现接近、双语 RAG 默认起点
  - 一个模型同时输出 dense + sparse + ColBERT,省半条流水线
  - 单 A10 / L4 forward ~10 ms,P99 时延预算友好
- 备选 **gte-Qwen2-7B-instruct**(7B, 3584 dim):中英更强,但 P99 < 200 ms 在 7B 模型上紧张,需 H100;只在 bge-m3 业务测试 NDCG@10 不达标时再升

**2) Dimension reduction**:

- bge-m3 原生 1024 维,对 500 万 chunk 体量不需要降维(500 万 × 1024 × float16 = 10 GB,单卡 GPU 装得下)
- 如果上 gte-Qwen 7B(3584 维 → 35 GB),必须做 Matryoshka 切片或 PCA 降到 1024——但 gte-Qwen 不天然支持 Matryoshka,需要离线 PCA / OPQ 后再入索引

**3) Retrieval pipeline**:

```
query
  ↓
bge-m3.encode(query) → dense_q + sparse_q
  ├─ Milvus(HNSW, M=32, ef=128) dense 召回 top-100
  ├─ bge-m3 sparse / Elasticsearch BM25 召回 top-100
  └─ RRF 融合(k=60)→ top-100
  ↓
bge-reranker-v2-m3(cross-encoder)精排 top-100 → top-10
  ↓
LLM(Qwen2.5-32B / DeepSeek-V3 等)+ context prompt 生成
```

延迟预算分配(P99 < 200 ms):

- query encoding:bge-m3 单卡 ~15 ms
- HNSW dense 检索:Milvus 100 万级 ~30 ms,500 万级分片后 ~50 ms
- BM25 检索:Elasticsearch ~30 ms
- RRF 融合 + 取候选:< 5 ms
- Reranker(top-100):bge-reranker-v2-m3 在 A10 ~80 ms(batch 化)
- 合计:~180 ms,刚好 fit

**4) Mini-benchmark 流程**:

- 找业务方拿 50-100 条真实 query + 标注的 ground-truth 相关 doc
- 评测指标:NDCG@10 / Recall@5 / MRR(retrieval 端);RAGAS faithfulness + context relevance(端到端)
- Baseline:bge-m3 dense-only,bge-m3 dense+sparse RRF,bge-m3 dense+sparse+reranker
- 加分对比:bge-large-zh-v1.5(纯中文 baseline,看多语模型有没有反而掉点)、gte-Qwen2-7B(看升级 7B 能涨多少)
- **结论以业务 NDCG 为准,不要看 MTEB 排行**

加分:

- 上线后挂 LLM-as-judge 抽样监控 faithfulness,持续观察是否需要换模型
- 数据更新走 Milvus upsert + 软删除,不要全量重建
- 中英 query 都要 sample 测试,确保跨语言 retrieval 没有塌(e.g. 中文 query 召回英文 doc 的能力)

</details>

---

## 7. 延伸阅读

- [MTEB Leaderboard(HuggingFace)](https://huggingface.co/spaces/mteb/leaderboard) — embedding 选型必看,按任务和语言筛实时跟 SOTA;C-MTEB 中文版同站点。
- [BGE GitHub(FlagEmbedding)](https://github.com/FlagOpen/FlagEmbedding) — bge 全系列(embedding + reranker + LLM-based)的官方实现 + 训练 / fine-tune 教程,中文社区最完整。
- [E5 / Multilingual-E5 paper code](https://github.com/microsoft/unilm/tree/master/e5) — 微软 E5 全系列开源,含训练数据合成 prompt(GPT-4 合成 150 任务的核心 trick)。
- [NVIDIA NV-Embed model card](https://huggingface.co/nvidia/NV-Embed-v2) — latent attention pooling 与 two-stage fine-tune 的 reference 实现。
- [Sentence-Transformers documentation](https://www.sbert.net/) — Reimers 维护的库,封装了几乎所有主流 embedding model 的 encode / cosine 接口,工业默认。
- [Pinecone — Embedding Model Evaluation Guide](https://www.pinecone.io/learn/series/rag/embedding-models-rundown/) — 各家商用 / 开源 embedding 的实测对比,讲选型决策很直接。
- 推荐继续读本教程的 **13.3 节《进阶 RAG》** —— 把本节训出 / 选好的 embedding 接入 HyDE / RAG-Fusion / GraphRAG 等高级 retrieval 范式。
