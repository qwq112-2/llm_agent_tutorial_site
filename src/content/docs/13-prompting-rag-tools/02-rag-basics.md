---
title: "13.2 RAG 基础：embedding / retriever / reranker / chunking"
description: "RAG 不是一个模型，而是一条 chunk → embed → retrieve → rerank → generate 的工程流水线——这一节讲清楚每一环的主流选型（bge-m3 / Milvus / bge-reranker / RRF）、参数甜点（chunk 500-1000 token、top-100 召回 + top-10 rerank）、以及在 1M context model 时代为"
---

> ⏱ 预计阅读 35 分钟 ｜ 难度 ★★ ｜ 前置：2.2（distributed representation 与 embedding 心智模型）

## 一句话本节讲什么

RAG 不是一个模型，而是一条 **chunk → embed → retrieve → rerank → generate** 的工程流水线——这一节讲清楚每一环的主流选型（bge-m3 / Milvus / bge-reranker / RRF）、参数甜点（chunk 500-1000 token、top-100 召回 + top-10 rerank）、以及在 1M context model 时代为什么 RAG 仍然是工业首选。

---

## 1. Mental model（直觉）

LLM 训完之后参数就冻住了。它的世界观停在某个 cutoff date（GPT-4 是 2023-04，Claude 4.7 是 2025-04），不知道你的公司内网文档，不知道今天的财报，更不知道半小时前刚发布的产品手册。强行问会发生三件事：

- **答不出来**："抱歉，我无法访问最新信息"——好歹诚实
- **幻觉**：编一个看起来很合理但完全错的答案——最危险
- **过时**：用旧版本 API 写代码，出现"GPT-3.5 时代真理但 Claude 4.7 已废弃"的回答

RAG（**R**etrieval **A**ugmented **G**eneration）的思路一句话：**先去外部知识库捞相关材料，把材料塞进 prompt 当 context，让 LLM 看着材料说话**。

```
              用户 query
                 │
        ┌────────┴─────────┐
        │ 1. embed(query)  │     ← 同一个 embedding model
        └────────┬─────────┘
                 ↓ 向量
        ┌──────────────────┐
        │ 2. ANN 在 Vector │     ← Milvus / Qdrant / FAISS
        │    DB 里检索 top-K│       HNSW 索引
        └────────┬─────────┘
                 ↓ K 个候选 chunk
        ┌──────────────────┐
        │ 3. Reranker 精排 │     ← bge-reranker (cross-encoder)
        │    保留 top-N     │       通常 K=100 → N=10
        └────────┬─────────┘
                 ↓ N 个高质量 chunk
        ┌──────────────────┐
        │ 4. LLM with      │
        │    context prompt│     ← "根据下面材料回答..."
        └────────┬─────────┘
                 ↓
              答案 + 引用
```

离线侧（建库）走的是另一条路：

```
原始文档 → chunking → 每个 chunk embed → 写入 Vector DB（带 metadata）
```

四个根本好处摆在桌面：

- **时效性**：LLM 不动、知识库一直在长，新文档进来 reindex 就行
- **私域**：公司内网 / 个人笔记 / 客服历史，不需要把数据塞进 LLM 训练集
- **减幻觉**：模型有材料可引，比"凭参数记忆"靠谱得多
- **可溯源**：每个答案都能指回具体哪段哪个文档——合规、可审计

但 RAG 也不是免费的。它把 LLM 调用变成了一次 **检索 + 生成的复合系统**——任何一环出问题（chunk 切坏、embedding 不匹配、reranker 漏掉关键 chunk、prompt 模板让 LLM 忽略 context）整个 pipeline 就塌了。下面把每一环拆开讲。

---

## 2. 公式与原理

### 2.1 Chunking：把文档切成可检索的最小语义单元

文档动辄几十万 token，不可能整篇做 embedding（embedding model 上下文通常 512-8192 token）也不可能整篇塞 prompt。必须先切成 **chunk**——理想的 chunk 满足三个条件：

1. **语义自洽**：单独读这一段能理解大意（不被句子切断、不被表格切断）
2. **大小适中**：太小信息不够，太大稀释 query 相关性
3. **覆盖完整**：相邻 chunk 间有少量 overlap 防止边界信息丢失

主流策略：

| 策略 | 描述 | 适用 |
|---|---|---|
| **Fixed-size** | 按 token 数硬切（每 500-1000 一段） | baseline、structureless 文本 |
| **Sentence-based** | 按句子边界切，凑够 N 句 | 通用文本 |
| **Recursive character splitter** | 按层级分隔符（`\n\n` → `\n` → `。` → ` `）递归切，尽量不破坏自然结构 | LangChain 默认、工业常用 |
| **Document-aware** | 按 markdown 标题 / HTML 标签 / PDF 章节切 | 结构化文档（手册、wiki）|
| **Semantic chunking** | 相邻句 embedding 相似度低于阈值就切开 | 主题切换密的文本，新但贵 |
| **Sliding window** | chunk 间留 50-100 token overlap | 配合任何上面的策略，几乎必加 |

经验值不用想：**chunk size 500-1000 token + overlap 50-100 token + recursive splitter**——这是 90% 业务的合理起点。再细的优化按下游 retrieval 指标（NDCG@10）调。

### 2.2 Embedding model：把 chunk 与 query 投到同一向量空间

embedding model 的目标：让 **语义相似的文本** 在 $\mathbb{R}^d$ 里 **余弦相似度高**，让无关文本相似度低。给定文本 $x$，embedding model $f_\theta$ 输出 $\mathbf{e}_x = f_\theta(x) \in \mathbb{R}^d$，相似度

$$\text{sim}(x, y) = \frac{\mathbf{e}_x \cdot \mathbf{e}_y}{\|\mathbf{e}_x\| \cdot \|\mathbf{e}_y\|}$$

训练范式是 **对比学习（InfoNCE）**——拉近正样本对（query, relevant doc），推远负样本对（query, irrelevant doc）。这条路 2.2 节讲过 motivation（word2vec 是 word-level 同款思想），现代 embedding model 把它做到了 sentence/passage level，详细机制在 16.4 节。

主流选型表（2025 年实战推荐）：

| Model | size | dim | 中英 | 备注 |
|---|---|---|---|---|
| OpenAI `text-embedding-3-large` | API | 3072（可降到 256） | 强 | 闭源 baseline、Matryoshka 维度可裁 |
| **BAAI `bge-m3`** | 568M | 1024 | 双语强 | 同时输出 dense + sparse + ColBERT 三种表示 |
| `bge-large-en-v1.5` | 335M | 1024 | 英文 | 经典英文开源 baseline |
| `bge-large-zh-v1.5` | 326M | 1024 | 中文 | 中文 baseline |
| `E5-mistral-7b-instruct` | 7B | 4096 | 双语 | LLM-based、强但慢 |
| `gte-Qwen2-7B-instruct` | 7B | 3584 | 双语 | LLM-based、Qwen2 base |
| `NV-Embed-v2` | 7B | 4096 | 英文 | MTEB 长期 SOTA |
| `jina-embeddings-v3` | 570M | 1024 | 多语 | 多语言均衡，task-specific LoRA |

工程选型口诀：

- **中英混合 / 多语言 → bge-m3**（一个模型搞定 dense + sparse，省一半流水线）
- **纯英文 / 追求质量 → NV-Embed-v2 或 OpenAI 3-large**
- **中文为主 / 私有化部署 → bge-large-zh-v1.5**（小、快、效果稳）
- **追求速度 / 边缘部署 → bge-small / gte-small**（几十 MB）

注意几乎所有 embedding model 对 **query 与 document 用不同 prefix**——bge 系列要写 `"query: <q>"` / `"passage: <d>"`，E5 系列同理。**漏写直接掉 5-10 个点 NDCG**（详见踩坑）。

### 2.3 Vector DB 与 ANN 索引：从精确到近似

Vector DB 干的事：存几百万到几十亿条 $d$ 维向量，给一个 query 向量返回最相似的 top-K。

最朴素是 **brute-force**——对所有 $N$ 个向量算余弦相似度排序，$O(N \cdot d)$。$N=10^4$ 还行，$N=10^7$ 单 query 就要几百毫秒——**必须用 ANN（Approximate Nearest Neighbor）索引**。

主流索引算法：

- **HNSW（Hierarchical Navigable Small World）**：多层图结构，从上到下贪心搜索。**最主流、recall 高、构建慢、内存大**。关键参数：`M`（每个点的邻居数，典型 16-64）、`ef_construction`（建图时的搜索宽度，典型 200）、`ef_search`（查询时搜索宽度，典型 50-500，越大越准越慢）。
- **IVF（Inverted File）**：先 K-means 聚成 `nlist` 簇，查询时只在最近的 `nprobe` 簇里搜。**内存省、速度快但 recall 略差**，常和 PQ 组合成 IVF-PQ。
- **ScaNN（Google）**：anisotropic vector quantization，速度快质量高，但生态不如 HNSW 普及。

主流 Vector DB / 库：

| 类型 | 代表 | 特点 |
|---|---|---|
| **专业 Vector DB** | Milvus / Qdrant / Weaviate / Vespa | 分布式、HNSW + 滤波 + 多副本、生产首选 |
| **轻量级** | Chroma / FAISS（library，不是 service） | 单机 < 千万级、原型 / 本地 demo 首选 |
| **PG 扩展** | pgvector | 复用 PostgreSQL 运维栈、< 百万级数据合适 |
| **托管** | Pinecone / Vertex AI Matching Engine | 不想自己运维就用、贵 |

容量与延迟参考：百万级单机 HNSW，P99 < 50ms 没问题；过了亿级要分片 + 异构存储（Milvus / Vespa 路线）。

### 2.4 Reranker：cross-encoder 的精排环节

embedding 检索是 **bi-encoder** 范式——query 和 doc **分别** 过 encoder 得到向量，再算内积。优点是 doc embedding 可以离线算好缓存，查询时只 encode query 一次。**缺点是 query 与 doc 之间没有交互**——细粒度匹配（"今年" 指代哪一年、否定词、数字精确匹配）经常错。

reranker 走 **cross-encoder** 范式：把 (query, doc) 拼成一个序列 `[CLS] query [SEP] doc [SEP]` 送进 encoder，输出一个标量 relevance score。query 与 doc 在每一层都做 cross-attention，匹配精度高一个数量级。

代价是慢——每个候选 doc 都要单独 forward 一次，无法预计算。所以经典 pipeline 是 **召回 + 精排两段式**：

```
embedding retrieval (top-100)  ──→  reranker 重排  ──→  top-10 给 LLM
   bi-encoder                       cross-encoder
   毫秒级                            十几毫秒 × 100 = 秒级
   高召回低精度                       高精度
```

主流 reranker：

- **BAAI `bge-reranker-v2-m3`** / `bge-reranker-large` — 开源首选、中英都强
- **Cohere Rerank v3** — API、英文 SOTA、商用首选
- **`ColBERTv2`** — late-interaction 路线（每个 token 独立 embedding，最大相似度求和），介于 bi 和 cross 之间

实测 reranker 在 NDCG@10 / MRR 上能比纯 dense retrieval 提升 **5-15%**，几乎是 RAG pipeline 性价比最高的一环。

### 2.5 Retrieval 策略：dense / sparse / hybrid

- **Dense retrieval**：embedding 内积 / 余弦——擅长语义、对同义词鲁棒、对 OOV 词友好
- **Sparse retrieval（BM25）**：基于词频 / IDF 的经典 lexical match——擅长关键词精确匹配（人名、产品型号、代码标识符）、对低频术语强
- **Hybrid retrieval**：两路结果融合，**Reciprocal Rank Fusion（RRF）** 是最常见的融合方法

RRF 的公式简单到一行：每个文档 $d$ 的最终分数

$$\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + \text{rank}_r(d)}$$

其中 $R$ 是所有检索器（这里 $|R|=2$，dense + sparse），$\text{rank}_r(d)$ 是 $d$ 在第 $r$ 个检索器结果里的排名（1-indexed），$k$ 是平滑常数（典型 60）。两条召回都靠前的文档自然分数高——**不需要额外训练融合权重**。

- **Multi-vector retrieval**：ColBERTv2 给每个 token 单独 embedding，相似度用 MaxSim 求和。精度比 dense 高、内存比 dense 大几十倍——大库慎用。

工业铁律：**dense + BM25 + reranker 三件套**——三者覆盖语义召回、词面召回、精排，几乎 cover 所有 retrieval 失效模式。

### 2.6 RAG vs Long-context LLM：边界在哪

2024 之后 1M context 模型开始普及（Gemini 1.5、Claude 4.7、Qwen3 等都支持），自然问题：**为什么不直接把所有文档塞进 context？**

| 维度 | RAG | Long-context LLM |
|---|---|---|
| **成本** | context 短，每次 query 几千 token | 100k+ token，每 query 几美元 |
| **延迟** | 短 prefill + 检索几十 ms | prefill 100k 要数秒（即使有 prefix cache）|
| **可扩展性** | 百万级文档 / TB 级语料 | 上限是模型 context window |
| **可更新性** | 增改 doc 直接 reindex | 重新塞 prompt |
| **跨文档推理** | 弱（chunk 是孤立的）| 强（同一 attention pass）|
| **lost in the middle** | 不严重（只塞 top-N）| 严重（中间信息易被忽略）|

现代实践：**RAG 仍是主流**——长 context 不是用来"装一切"，而是 **作为 RAG 的 context window 内 grounding**——retrieval 把范围缩到几十 K 高相关 token，再用 long-context LLM 做高质量推理。Anthropic 在 contextual retrieval 文章里把这叫做"RAG vs no-RAG"边界——只要候选材料超过几万 token 就还是 RAG 划算。

### 2.7 简单 RAG 的局限（13.3 的预告）

- **lost in the middle**：Liu et al. 2023 证明 LLM 对 prompt 中间位置的信息明显弱于头尾，长 context 中间的关键 chunk 易被忽略
- **multi-hop reasoning**：跨多个文档的链式推理（"A 公司 CEO 的母校是哪"）单轮 retrieval 召不全
- **fragmented chunks**：chunk 切断了语义，关键信息可能分散在多个 chunk
- **no context awareness**：retrieve 只看 query 不看对话历史，多轮场景下 query reformulation 必须做

解决这些就是 **进阶 RAG**——HyDE / RAG-Fusion / Self-RAG / GraphRAG，留到 13.3 详讲。

### 2.8 评测：retrieval 指标 vs 端到端指标

- **Retrieval 端**：Recall@K（top-K 里有没有 ground truth）、NDCG@K（位置加权的相关性）、MRR（第一个相关结果的倒数排名）、Hit@K
- **End-to-end**：**RAGAS**（faithfulness 答案是否忠实于 context、answer relevance 答案是否回答了 query、context relevance 检索的 context 是否相关）、人工评测
- **embedding model leaderboard**：**MTEB**（Massive Text Embedding Benchmark）涵盖 56 个数据集 8 类任务，是选 embedding model 的事实标准

---

## 3. 最小代码示例

### 3.1 完整 RAG pipeline（chunk + embed + retrieve + LLM）

```python
# pip install sentence-transformers faiss-cpu
import faiss, numpy as np
from sentence_transformers import SentenceTransformer

# 1. 准备文档与 chunking（这里用最朴素的固定切分演示）
docs = [open(f).read() for f in ["doc1.txt", "doc2.txt", "doc3.txt"]]

def chunk_fixed(text, size=500, overlap=50):
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i : i + size]))
        i += size - overlap                         # sliding window
    return chunks

chunks = [c for d in docs for c in chunk_fixed(d)]  # 全部 chunk 摊平

# 2. embedding（bge-m3 自带 query/passage 区分）
model = SentenceTransformer("BAAI/bge-m3")
chunk_emb = model.encode(chunks, normalize_embeddings=True)  # (N, 1024)

# 3. 建 FAISS HNSW 索引
index = faiss.IndexHNSWFlat(1024, 32)               # M=32
index.hnsw.efConstruction = 200
index.add(chunk_emb.astype(np.float32))

# 4. retrieve
def retrieve(query, k=5):
    q_emb = model.encode([query], normalize_embeddings=True).astype(np.float32)
    scores, idx = index.search(q_emb, k)
    return [(chunks[i], scores[0][j]) for j, i in enumerate(idx[0])]

# 5. 喂给 LLM
context = "\n\n".join(c for c, _ in retrieve("公司 2024 年第三季度营收？"))
prompt = f"根据下面材料回答问题，只用材料里的信息：\n\n{context}\n\n问题：公司 2024 年第三季度营收？"
# llm.generate(prompt)   # 接你的 LLM 客户端
```

50 行内一条端到端 RAG 跑起来。生产里把 FAISS 换成 Milvus、把 chunking 换成 LangChain 的 `RecursiveCharacterTextSplitter`、把 LLM 接成 vLLM 或 Anthropic API 即可。

### 3.2 Hybrid search（dense + BM25 + RRF）

```python
# pip install rank_bm25
from rank_bm25 import BM25Okapi

# 假设 chunks / chunk_emb / index 已建好（见 3.1）
bm25 = BM25Okapi([c.split() for c in chunks])

def hybrid_search(query, k=10, k_rrf=60):
    # dense 召回
    q_emb = model.encode([query], normalize_embeddings=True).astype(np.float32)
    _, dense_idx = index.search(q_emb, k * 5)       # 多召回几路给 RRF 用
    dense_rank = {i: r + 1 for r, i in enumerate(dense_idx[0])}
    # sparse 召回
    bm25_scores = bm25.get_scores(query.split())
    bm25_idx = np.argsort(-bm25_scores)[: k * 5]
    bm25_rank = {i: r + 1 for r, i in enumerate(bm25_idx)}
    # RRF 融合
    all_ids = set(dense_rank) | set(bm25_rank)
    rrf = {i: 1 / (k_rrf + dense_rank.get(i, 1e9)) +
              1 / (k_rrf + bm25_rank.get(i, 1e9)) for i in all_ids}
    return [chunks[i] for i, _ in sorted(rrf.items(), key=lambda x: -x[1])[:k]]
```

`k_rrf=60` 是 Cormack 2009 的原始推荐，几乎不用调。RRF 的好处是 **不需要标注数据训练融合权重**——两路检索器都靠前的文档自然胜出。

### 3.3 Reranker pipeline

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)

def retrieve_then_rerank(query, k_recall=100, k_final=10):
    candidates = hybrid_search(query, k=k_recall)             # 100 路召回
    pairs = [(query, c) for c in candidates]
    scores = reranker.predict(pairs)                          # cross-encoder 打分
    ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
    return [c for c, _ in ranked[:k_final]]                   # top-10
```

15 行实现 RAG 里 ROI 最高的环节——上面的 `bge-reranker-v2-m3` 是当前开源中英 reranker 的 SOTA。

### 3.4 Chunking 策略对比（fixed vs recursive）

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text = open("doc1.md").read()

# 方案 A：fixed-size（按字符数硬切，不看分隔符）
fixed_chunks = [text[i : i + 500] for i in range(0, len(text), 450)]   # overlap 50

# 方案 B：recursive（按 markdown 自然分隔符递归切）
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50,
    separators=["\n## ", "\n### ", "\n\n", "\n", "。", " ", ""],
)
recursive_chunks = splitter.split_text(text)

print(f"fixed: {len(fixed_chunks)} 段、平均 {sum(len(c) for c in fixed_chunks)//len(fixed_chunks)} 字")
print(f"recursive: {len(recursive_chunks)} 段、平均 {sum(len(c) for c in recursive_chunks)//len(recursive_chunks)} 字")
# recursive 的 chunk 数会少一些但每段更"完整"——边界落在自然分隔处
```

跑一次会立刻看到差别：fixed 经常把 markdown 标题、表格、代码块从中间切断；recursive 会优先在 `\n## `、`\n\n` 这种自然边界切，retrieval 质量明显更稳。

---

## 4. 工程踩坑与经验

- ❗ **Chunk size 是 RAG 的第一超参**：太小（< 100 token）→ 上下文残缺，retrieve 拿到一堆"半句话"；太大（> 2000）→ 一段里塞了多个主题，与 query 的相似度被稀释，rank 下沉。**500-1000 token + 50-100 overlap** 是 90% 业务的甜点；技术文档 / 代码可往 1500 走，FAQ / 短文本可往 300 走。
- ❗ **Embedding model 必须配套 query / document 不同 prefix**：bge 系列要 `"query: <q>"` / `"passage: <d>"`，E5 同理，instructor 系列要再加 task instruction prefix。**忘写 prefix 通常掉 5-10 个点 NDCG**——而且这种 bug 不会报错，只会让你以为 embedding model 不行。`SentenceTransformer.encode` 在新版会自动处理 bge 的 prefix，但一定 check 文档。
- ❗ **多语言场景必须用多语 embedding**：中英混合数据用 `bge-large-en` 之类单语模型，中文 chunk 与中文 query 之间相似度还行，但 **中文 chunk vs 英文 query** 的检索几乎完全塌掉。**用 bge-m3 / multilingual-e5 / jina-v3 这类多语模型**，单一向量空间覆盖所有语种。
- ❗ **FAISS 的 IndexFlat 不是 ANN 是 brute-force**：很多 demo 用 `IndexFlatIP` 跑通就上生产，结果 1M 数据后 P99 飙到 1s+。**> 100 万向量必须换 `IndexHNSWFlat` 或专业 vector DB**（Milvus / Qdrant）；> 1 亿要走分布式分片。
- ❗ **Hybrid search 的融合公式优先选 RRF**：早期教材会教"dense_score × 0.6 + bm25_score × 0.4"——但 dense 余弦相似度（[-1, 1]）和 BM25 分数（无界正数）量纲完全不同，加权要做归一化，超参难调。**RRF 只看 rank 不看 score，免标定免归一**，几乎是工业默认。
- ❗ **Reranker 是 latency 大头**：cross-encoder 每个 candidate 一次 forward，top-100 在 GPU 上也要 50-200 ms。**典型配置 retrieve top-100 + rerank top-10**——超过 100 边际收益递减，不到 50 又会漏掉好结果。bge-reranker-v2-m3 在 A10 / L4 上跑 100 候选大概 100 ms 量级，是 P99 latency 的主要预算项。
- ❗ **Document update 要走增量索引**：vector DB 全量重建百万级要几十分钟、千万级要小时级——任何工业 RAG 必须支持 **upsert（按文档 ID 更新）+ 软删除**。HNSW 原生不支持高效删除，必须用 Milvus / Qdrant 这种封装好的；纯 FAISS 只能定期重建。
- ❗ **Embedding 维度高显存与计算开销大**：4096 维向量比 1024 维多 4× 存储、4× 距离计算开销。**用 Matryoshka embedding（OpenAI 3-large / NV-Embed-v2）可以无痛降维到 1024 / 768 / 256**，质量几乎无损。百万级以上一定降维。
- ❗ **prompt 模板里要明确 grounding 指令**：常见错误是把 context 塞进去就完事，结果 LLM 还是混用 context 与参数知识。**必须写"只根据下面材料回答，材料没有就说不知道；引用时标 \[doc-id\]"**——否则幻觉与无引用问题立刻冒头。
- ❗ **不要把 RAG 评测只盯 retrieval 指标**：Recall@10 = 100% 的 pipeline 端到端答案准确率可能只有 60%——因为 LLM 没用对 context、或者 context 太多被 lost in the middle。**RAGAS 三件套（faithfulness / answer relevance / context relevance）+ 人工评 50 条**，比 NDCG 更接近用户体感。

---

## 5. 经典 paper

- **Lewis et al., 2020 — Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks** — RAG 概念的原典，把 retriever（DPR）和 generator（BART）端到端联合训练，定义了 RAG-Sequence / RAG-Token 两种生成模式。现代工业 RAG 多数 **不再 joint train**（retriever 和 LLM 解耦更工程友好），但这篇定下的 "retrieval as prompt augmentation" 心智模型是后续所有工作的起点。读 §2-3 即可。
- **Karpukhin et al., 2020 — Dense Passage Retrieval (DPR)** — dense retrieval 范式落地之作，证明用 BERT 双塔 + 对比学习训出的 embedding 在开放域 QA 上吊打 BM25。读 §3-4 理解 in-batch negatives 的训练方式，这是后续 bge / E5 / NV-Embed 的共同模板。
- **Xiao et al., 2024 — BGE M3-Embedding** — 当前中英 / 多语开源 embedding 的事实标准，**一个模型同时输出 dense / sparse / multi-vector 三种表示**，省一半流水线。读 §3 看多功能训练目标的设计；下游做中文 / 多语 RAG 几乎绕不开这个模型。
- **Khattab & Zaharia, 2020 — ColBERT** — late-interaction 范式提出，每个 token 独立 embedding + MaxSim 相似度，介于 bi-encoder（单向量）和 cross-encoder（全交互）之间——精度接近 cross-encoder 但可预计算 doc embedding。bge-m3 的 multi-vector 输出就是 ColBERT 路线。
- **Es et al., 2023 — RAGAS: Automated Evaluation of Retrieval Augmented Generation** — RAG 端到端评测的事实标准框架，定义 faithfulness / answer relevance / context relevance 三个 LLM-judged 指标。读完能直接搭一套自动评测——做 RAG 的人必备。

---

## 6. 自测与面试题

**Q1（pipeline）**：标准 RAG 的 5 个阶段是哪些？每个阶段一句话说明做什么、用什么主流工具。

<details>
<summary>Answer sketch</summary>

- **Chunking**：把原始文档切成 500-1000 token 的 chunk，用 LangChain `RecursiveCharacterTextSplitter` 之类按层级分隔符递归切，加 50-100 token overlap。
- **Embedding**：用 bi-encoder（bge-m3 / E5 / OpenAI 3-large）把 chunk 编码成 $d$ 维稠密向量，离线写入 vector DB；query 时同一模型编码 query。**注意 query / passage prefix**。
- **Retrieve**：vector DB（Milvus / Qdrant / FAISS）用 HNSW 索引返回与 query embedding 最相似的 top-K（典型 K=100），可叠加 BM25 走 hybrid + RRF。
- **Rerank**：cross-encoder（bge-reranker-v2-m3 / Cohere Rerank）对 top-K 重新精排，保留 top-N（典型 N=10）。
- **Generate**：把 top-N chunk 塞进 prompt 模板（带 grounding 指令），LLM 生成最终答案，附引用。

加分：retrieve 和 rerank 的 trade-off 是 K 越大召回越全但 rerank 越慢，100→10 是经验甜点。

</details>

**Q2（trade-off）**：现在 1M context LLM 已经普及，"为什么还要做 RAG，直接把所有文档塞 prompt 不行吗？" 从至少 4 个维度对比。

<details>
<summary>Answer sketch</summary>

至少答 4 个维度：

- **成本**：100k token context 单次 query 几美元，RAG 检索后只塞 5-10k 高相关 context，成本低 1-2 个数量级。
- **延迟**：100k prefill 即使有 prefix cache 也要数秒；RAG 检索几十 ms + 短 prefill 几百 ms，TTFT 低 5-10×。
- **可扩展性**：context window 上限（1M token ≈ 几本书），RAG 可承载 TB 级语料 / 百万文档，只受 vector DB 容量限制。
- **可更新性**：RAG 增改文档直接 upsert vector DB（秒级），long-context 要重新组装 prompt 而且没办法做大规模"知识库"管理。
- **lost in the middle**：长 context 中间位置信息易被 LLM 忽略（Liu 2023），RAG 只塞 top-N 高相关 chunk 反而 grounding 更稳。
- **跨文档推理**：long-context 一个 attention pass 内可以跨文档关联，RAG 这点弱（chunk 是孤立的，需要 multi-hop / GraphRAG 等进阶手段补救）——这是 long-context 的独家优势。

结论：现代实践是 RAG + long-context 互补——retrieve 把候选缩到几十 K，再让 long-context LLM 在里面做高质量推理。

</details>

**Q3（实战）**：让你做一个企业客服 RAG，要求：100 万条历史工单 + 10 万页产品文档、中英文混合、P99 latency < 2s、QPS 500、答案必须可溯源。请列出技术选型（embedding / vector DB / reranker / LLM / chunking 策略）以及关键容量估算。

<details>
<summary>Answer sketch</summary>

参考方案：

- **Chunking**：LangChain `RecursiveCharacterTextSplitter`，chunk 500 token + overlap 80；工单按"问题 + 解决方案"对作为 atomic chunk（业务结构化数据应优先用 document-aware 切，不要硬切）。预估 chunk 总数 ~5M。
- **Embedding**：**bge-m3**（568M、双语、dense 1024 维）——一个模型搞定中英、可输出 sparse 顺手做 hybrid。Matryoshka 切到 768 维省 25% 存储。离线建库估算：5M chunk × 768 × 4B = 15 GB embedding。
- **Vector DB**：**Milvus 集群**（3 副本 + HNSW，M=32, ef=100），> 100 万必须分布式 + 增量索引。15 GB embedding 单分片 GPU 显存放得下，retrieve P99 < 30 ms。
- **Hybrid**：dense（bge-m3 dense 输出）+ BM25（Elasticsearch 或 Milvus 自带 sparse），RRF 融合，召回 top-100。
- **Reranker**：**bge-reranker-v2-m3**，rerank 100 → 10，A10 GPU 估 ~80 ms。
- **LLM**：Qwen2.5-32B / DeepSeek-V3 / Claude Haiku 4.7（视成本和私有化要求选）。Prompt 模板必须明确 "只根据材料、引用 [doc-id]、材料没有就说不知道"。
- **延迟预算分配**（P99 < 2s）：embedding query 20ms + retrieve 50ms + rerank 100ms + LLM TTFT 300ms + 生成 1.5s ≈ 1.97s，刚好 fit。
- **可溯源**：每个 chunk 写入 metadata（doc_id、url、版本、更新时间），LLM 引用 [doc-id] 后端反查直接返回原文链接。
- **QPS 500**：embedding/rerank 是瓶颈——bge-m3 在 A10 单卡 ~200 QPS，需 3 卡；reranker 100 candidate × forward 是大头，需要 batch 化 + 多卡。Vector DB Milvus 集群水平扩展即可。
- **加分**：增量更新走 upsert + 软删除；冷启动用日志离线评测 NDCG@10 + RAGAS 三件套；上线后接入 LLM-as-judge 抽样监控 faithfulness。

</details>

---

## 7. 延伸阅读

- [Anthropic — Introducing Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) — 2024 年 9 月 Anthropic 提出在 chunk 前 prepend 一段 context summary，把 retrieval failure rate 降 49%。是现代 RAG 最重要的简单 trick 之一。
- [LangChain — RAG From Scratch (YouTube)](https://github.com/langchain-ai/rag-from-scratch) — Harrison Chase 亲自录的 14 集视频 + notebook，从基础 RAG 到 query transformation / routing / advanced retrieval 全覆盖。
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) — embedding model 选型必看，按任务（retrieval / reranking / STS）和语言筛，实时跟踪 SOTA。
- [Pinecone — Vector Database Learning Hub](https://www.pinecone.io/learn/) — vector DB / ANN 索引算法（HNSW / IVF / PQ）的最易读入门，讲机制和图解都很清晰。
- [Hamel Husain — Field Notes on RAG](https://hamel.dev/blog/posts/evals/) — RAG 上线评测的实战经验，强调先搭 eval 再迭代，避免拍脑袋调参。
- 推荐继续读本教程的 **13.3 节《进阶 RAG》**——把 lost in the middle / multi-hop / 对话历史等本节没解的问题用 HyDE / Self-RAG / GraphRAG 补齐。
