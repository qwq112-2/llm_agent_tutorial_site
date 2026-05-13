---
title: "13.3 进阶 RAG：HyDE / RAG-Fusion / Self-RAG / GraphRAG"
description: "vanilla RAG 是\"一次 embed → 一次 retrieve → 一次 generate\"的直线 pipeline，每个环节都有明确的失败模式——这一节把社区 2022-2026 年提出的进阶手段按\"治哪个病\"整理：HyDE 治短 query 与长 doc 的 embedding 不匹配，RAG-Fusion 治单 query 召回不全，Self-RAG / CRAG 治\"该不该信 r"
---

> ⏱ 预计阅读 45 分钟 ｜ 难度 ★★★ ｜ 前置：13.2

## 一句话本节讲什么

vanilla RAG 是"一次 embed → 一次 retrieve → 一次 generate"的直线 pipeline，每个环节都有明确的失败模式——这一节把社区 2022-2026 年提出的进阶手段按"治哪个病"整理：HyDE 治短 query 与长 doc 的 embedding 不匹配，RAG-Fusion 治单 query 召回不全，Self-RAG / CRAG 治"该不该信 retrieve 出来的东西"，GraphRAG 治跨 chunk 的 corpus-level 理解，Agentic RAG / R1 + RAG 把 RAG 当 tool 让 reasoning model 自己调。

---

## 1. Mental model（直觉）

vanilla RAG 在 13.2 已经讲清——`chunk → embed → retrieve → rerank → generate` 一条直线。这条直线在简单 fact 查询上效果很好，但只要 query 稍微复杂一点就会塌：

- 用户问 "RoPE 是怎么解决长文本外推的"——query 短、术语密；doc 是某篇博客的长段铺垫——**embedding 不在一个语义层**
- 用户问 "对比 LLaMA / Qwen / DeepSeek 的训练数据策略"——这是个 **whole-corpus 综合题**，不是某一个 chunk 能答的
- 用户问 "公司 CEO 的母校"——需要 **先查 CEO 是谁，再查 TA 母校**——单轮 retrieve 出不来
- retrieve 召回了一堆 doc，里面有的相关有的不相关——**LLM 不知道该信哪个**

进阶 RAG 的所有方法本质都在**给 vanilla RAG 加 feedback loop**：要么在 retrieve 之前重写 query，要么在 retrieve 之后让 LLM 自我判断，要么把 retrieve 拆成多轮，要么换一种完全不同的索引结构。可以画成这样一张谱系图：

```
                  vanilla RAG (13.2)
                        │
   ┌────────────────────┼────────────────────┬──────────────────┐
   │                    │                    │                  │
"query 不好"        "召回不够"          "retrieve 不可信"      "chunk 视角太局部"
   │                    │                    │                  │
HyDE                Multi-query           Self-RAG            GraphRAG
(LLM 生成假答)      RAG-Fusion+RRF        CRAG                Community summary
                                                              ↓
                                                       LightRAG / LazyGraphRAG
                        │
                        ↓
                   Agentic RAG
            (LLM 自己决定 retrieve 几次)
                        │
                        ↓
                R1 + RAG / Search-R1
            (reasoning model 把 search 当 tool)
```

一个反复出现的设计 pattern：**用更多 LLM call 换更高 retrieval 质量**。HyDE 是 "1 次额外 LLM call 生成 hypothetical doc"；RAG-Fusion 是 "1 次 LLM 重写 + N 次 retrieve"；Self-RAG / CRAG 是 "retrieve 后再让 LLM 评一次"；Agentic RAG 是 "LLM 反复调 retrieve 直到满意"。

工程上的核心问题不是"哪个方法最强"——是 **哪个方法的 cost / benefit 在你的场景下值得**。下文把每个方法 + 适用场景 + 成本一起讲清楚。

---

## 2. 公式与原理

### 2.1 HyDE：让 LLM 先"猜一个答案"再去 retrieve

vanilla RAG 把 **query 的 embedding** 拿去搜 doc。问题是：query 通常很短（"RoPE 怎么外推"，10 个 token），doc 通常很长（一段几百字的解释）——同一个 embedding 空间里它们的语义密度差很多，直接算余弦距离往往**不如让 LLM 先生成一个"假答案"再用假答案去搜**。

HyDE（**Hy**pothetical **D**ocument **E**mbeddings，Gao et al. 2022）的核心思路一行公式：

$$\text{retrieve}(q) \;\to\; \text{retrieve}(\text{LLM}(q))$$

具体步骤：

1. 给 LLM 一个 prompt："请回答下面这个问题：{q}"，让它生成一段 hypothetical answer $\hat{a}$（哪怕完全是幻觉也没关系）
2. 把 $\hat{a}$ 而不是 $q$ 送进 embedding model
3. 用 $\text{embed}(\hat{a})$ 去 vector DB retrieve

为什么 work：**embedding model 训练时见到的就是"长 doc 与长 doc"或"短 query 与长 doc"的对比对**——前一种语义匹配通常更准。让 LLM 把短 query "翻译" 成 doc-shape 的伪答案，相当于把检索任务从 **跨形态匹配（query↔doc）** 降级成 **同形态匹配（doc↔doc）**。

适用：

- query 非常短（< 20 token）、术语稀疏、doc 长且密
- zero-shot QA / 开放域搜索
- LLM 对该领域有一定先验（即使生成的假答案有错，方向大致对就行）

不适用：

- query 已经很长 / 信息密（比如几十词的法律条文式 query）
- LLM 对该领域完全没先验（生成的 hypothetical doc 全错，把 retrieve 带偏）
- 高 throughput 场景（每个 query 多 1 次 LLM call，TTFT 多几百 ms）

### 2.2 RAG-Fusion：multi-query + RRF

单个 query 的 embedding 是一个点，retrieve 拿到的是这个点周围的 top-K——**对同义词、改写表达、隐含意图覆盖不全**。RAG-Fusion（Raudaschl 2023）的思路：让 LLM 把原 query 改写成 N 个变体（typically 3-5），每个变体各自 retrieve，再用 **Reciprocal Rank Fusion（RRF）** 合并。

公式（与 13.2 hybrid search 同款）：

$$\text{score}_{\text{RRF}}(d) = \sum_{q \in Q} \frac{1}{k + \text{rank}_q(d)}$$

- $Q$ 是 LLM 生成的所有 query 变体（含原 query）
- $\text{rank}_q(d)$ 是 doc $d$ 在 query $q$ 的 retrieve 结果里的排名（1-indexed）
- $k$ 是平滑常数，原始论文取 60，几乎不用调
- 多个 query 都把 $d$ 排在前面 → $d$ 的 RRF score 自然高

Multi-query 的 LLM prompt 大致：

```
请把下面的查询改写成 4 个不同表述但语义等价的版本，每行一个：
原查询：{q}
```

适用：

- query 含糊 / 多义 / 用户用语与文档术语不一致
- 用户表述非专业（"那个能算梯度的库"→ 改写成 "PyTorch autograd 用法"）
- recall 重要 > precision（先把候选捞全，再让 reranker 精排）

调参经验：

- query 数 $|Q|=3$-$5$ 是甜点；超过 5 边际收益快速递减、且 LLM 容易造重复 query
- 必须把 **原 query 也加入** $Q$——LLM 改写有时候会偏离原意
- 配合 reranker 使用：multi-query → RRF top-100 → cross-encoder rerank top-10

### 2.3 Self-RAG：让 LLM 自己决定要不要 retrieve

vanilla RAG 是"无脑 retrieve"——不管 query 是 "1+1=?" 还是 "公司 2024 Q3 营收"都先 retrieve 一遍，浪费计算 + 引入噪声。Self-RAG（Asai et al. 2023）训练 LLM **输出特殊 token 来自我控制 retrieve 行为和评估**：

| Token | 含义 |
|---|---|
| `[Retrieve]` / `[NoRetrieve]` | 当前 step 是否需要去 retrieve |
| `[Relevant]` / `[Irrelevant]` | retrieve 回来的 chunk 与 query 是否相关 |
| `[Supported]` / `[NotSupported]` / `[PartiallySupported]` | 当前生成内容是否被 chunk 支持 |
| `[Useful: 5]` / `[Useful: 1]` | 整体回答的 utility 评分 |

decoding 时 model 自己决定何时插这些 token：query 来了先输出 `[Retrieve]` / `[NoRetrieve]`；如果 retrieve 了，对每个候选 chunk 输出 `[Relevant]` / `[Irrelevant]`；生成每一段答案后输出 `[Supported]` 等 critique token。

实现关键点（**纯 prompt 仿不出真 Self-RAG**）：

- 必须 SFT 训练阶段把这些 reflection token 加入词表 + 构造带 critique 标注的训练数据
- 论文里 critique 数据是用 GPT-4 自动标注的（self-distill）
- inference 时可以用 **tree-decoding**：让模型生成多个候选回答，按 critique token 打分选最优

实证收益：

- 减少 hallucination（`[Supported]` 起到自我约束作用）
- 提升 factuality（在 PubMedQA / PopQA 等 benchmark 上明显涨点）
- 自适应 retrieve 减少不必要的 retrieval call（适合 hybrid query 场景）

### 2.4 CRAG：retrieve 完之后做一次 quality check

CRAG（**C**orrective **R**AG，Yan et al. 2024）的思路与 Self-RAG 互补——**不修改 LLM 训练，而是在 retrieve 后插一个 retrieval evaluator**，把 retrieve 结果分成三类：

```
        retrieve (top-K)
              │
              ↓
     ┌─────────────────┐
     │ retrieval       │  ← 一个小 model（fine-tuned T5 / 强 LLM）
     │ evaluator       │
     └────────┬────────┘
              │
       ┌──────┼──────┐
       ↓      ↓      ↓
   correct  ambig.  incorrect
       │      │      │
       │      │      └──→ 弃 retrieve、走 web search
       │      └──────────→ retrieve + web search 拼接
       └─────────────────→ retrieve 内容直接用
```

具体流程：

1. retrieve 拿到 top-K candidate
2. evaluator 给每个 candidate 打个相关性分（0-1），整体聚合判定为 correct / ambiguous / incorrect
3. **correct**：直接把 retrieve 内容做 knowledge refinement（去掉无关句子、压缩）后送 LLM
4. **incorrect**：完全弃 retrieve，转去 web search（Google / Bing API）补充
5. **ambiguous**：retrieve + web search 都用，互补 grounding

evaluator 的实现选择：

- **训一个 T5-base** 当评估器（论文方案，便宜但要标注数据）
- **直接用 GPT-4 / Claude 当评估器**（贵但准、零样本可用，工业捷径）

CRAG 与 Self-RAG 的关键差别：

| 维度 | Self-RAG | CRAG |
|---|---|---|
| 修改 LLM | 必须重训 + 加 reflection token | 不改 LLM |
| 退路 | 没找到就生成时标 `[NotSupported]` | 直接走 web search 兜底 |
| 部署难度 | 高（要训 model） | 低（plug-in evaluator 就行） |
| 适用 | 私有化、定制 LLM 场景 | 业务侧"挂在现有 RAG 上"场景 |

### 2.5 GraphRAG：从 chunk 视角到 entity-graph 视角

前面所有方法都在 **chunk 这个粒度** 上玩花样——但有一类问题用 chunk 根本回答不了：

> "请总结这 10000 份 customer feedback 的主要主题"
> "这本书里 A 角色和 B 角色的关系是怎么演变的"

这种 **whole-corpus understanding** 问题，retrieve 出 5 个 chunk 根本不够看——你需要的是 **整个语料的结构化表示**。GraphRAG（Edge et al. 2024，Microsoft）的做法：

**离线建图**：

1. 把所有文档分 chunk
2. 用 LLM 做 **entity / relation extraction**——从每个 chunk 抽出 (entity, type, description) 和 (entity₁, entity₂, relation)
3. 合并所有 chunk 的 entity（用 entity name 做 dedup），建一个全局 knowledge graph
4. 在 graph 上跑 **community detection**（典型 Leiden 算法），把节点聚成 hierarchical communities
5. 对每个 community，用 LLM 生成 **community summary**（这个社群是讲什么的、关键 entity 是哪些）

**Query 时**：

- **Local query**（"X entity 是什么"）：先 entity match 到节点，沿邻居拉相关 chunk
- **Global query**（"全语料主要主题"）：先用 query 匹配 community summary，再把相关 community 内的 chunk 聚合，map-reduce 式生成最终答案

公式上没什么数学，全是 graph 工程。但有个关键 trade-off：

| 维度 | vanilla RAG | GraphRAG |
|---|---|---|
| 建库 cost | 低（embed 一次） | **高**（entity extraction 每 chunk 1 次 LLM call，百万 doc 几百 USD）|
| Query cost | 低（向量 retrieve） | 中（community match + map-reduce） |
| Specific fact | 强 | 弱（graph 抽象掉了细节）|
| Whole-corpus 综合 | 弱 | **强** |
| 索引可解释性 | 弱（只是向量）| 强（人能看懂的 entity graph）|

### 2.6 LightRAG / LazyGraphRAG：GraphRAG 的工程化简化

GraphRAG 最大问题是 **建图 cost 高**——每个 chunk 一次 LLM call，加上 community summary 还要一遍 LLM。LightRAG / LazyGraphRAG 的思路：

- **延迟 community summary**：建图时只抽 entity / relation，**不预生成 summary**；query 时再 on-the-fly 对相关 community 做 summary
- **复用 entity index**：把 entity 当成 retrieval index 的一部分，沿用现成的 vector retrieve infra
- **混合检索**：dense retrieve 先召 chunk，再沿 entity edge 扩展邻居 chunk，形成局部 subgraph

工程上：建图 cost 降一个数量级，但 query 时延略涨，整体性价比更适合中等规模（几万-几十万 doc）的场景。

### 2.7 Agentic RAG：把 RAG 当 tool 让 agent 自己调

前面所有方法的 retrieve 时机都是固定的（HyDE 是 query 一来就 retrieve、Self-RAG 是 model 输出 `[Retrieve]` 时 retrieve）。Agentic RAG 把 retrieve **完全交给 agent loop**：

```
agent.run(query):
    while not satisfied and step < max_step:
        action = LLM.plan(query, history)
        if action.type == "retrieve":
            chunks = retrieve(action.query)
            history.append(chunks)
        elif action.type == "answer":
            return action.text
        else:
            ...  # 其他 tool
```

agent 可能 retrieve 多次（先粗后细）、也可能一次都不 retrieve（用参数知识就够）、还可以**改写 query 再 retrieve**。Perplexity / OpenAI Deep Research / Anthropic 的 web research feature 都用这套范式。

适用：

- 复杂 multi-step 问题（"总结领域 X 的近 3 年进展并对比")
- 需要跨数据源（向量库 + web + 数据库 SQL + code 执行）
- 对延迟不敏感（可以接受 5-30s）的 background research 任务

陷阱（详见踩坑）：

- 死循环：retrieve 完觉得不够，再 retrieve，再不够，再 retrieve... 必加 `max_step`
- token 爆炸：每次 retrieve 都把 chunk 塞回 history，几轮就到 100k
- agent 框架与具体实现见 Module 14

### 2.8 R1 + RAG / Search-R1：reasoning model 调 search

2025 年最火的范式——把 RAG 与 reasoning model（DeepSeek-R1 / o1 系列）结合。Search-R1（Hou et al. 2025）的做法：用 **RLVR** 训练 reasoning model 在 long CoT 中**自然地调用 search engine**：

```
<think>
我需要查一下 X 的最新数据。
<search>X 2025 年最新数据</search>
<result>...</result>
看起来 ... 我再查一下 Y 的对比。
<search>Y 与 X 对比</search>
<result>...</result>
所以 ...
</think>
<answer>...</answer>
```

模型在 RL 训练时根据**最终答案对错**回传 reward——learn to retrieve 的时机、query 的写法、以及如何在 CoT 里利用 retrieve 结果。这是 Module 15.4 的内容，本节只做衔接。

### 2.9 方法对比一览（必背表）

| 方法 | 适合 query | 增加 cost | 是否要训 | scaling |
|---|---|---|---|---|
| vanilla RAG (13.2) | specific fact | baseline | 否 | 百万 doc |
| HyDE | 短 query / 长 doc | +1 LLM call | 否 | 同 vanilla |
| Multi-query / RAG-Fusion | 含糊 / 多义 query | +1 LLM call + N retrieve | 否 | 同 vanilla |
| Self-RAG | 高 factuality 要求 | +若干 critique LLM 计算 | **是** | 同 vanilla |
| CRAG | 不确定 retrieve 质量 | +1 evaluator call (+ web search) | 评估器可训可零样本 | 同 vanilla |
| GraphRAG | whole-corpus 理解 | 建图 \$\$\$，query 中 | 否 | 万级 doc |
| LightRAG | 中等规模 corpus 综合 | 建图 \$，query 中 | 否 | 几十万 doc |
| Long-context (1M) | 跨多 doc 推理 | 每 query 极高 | 否 | 万级 doc |
| Agentic RAG | 复杂 multi-step | 多 LLM + tool | 否（用 prompt agent） | 通用 |
| R1 + RAG / Search-R1 | reasoning + 检索 | 极高（long CoT + retrieve）| **是**（RL 训）| 通用 |

---

## 3. 最小代码示例

### 3.1 HyDE 实现

```python
# pip install sentence-transformers faiss-cpu openai
from sentence_transformers import SentenceTransformer
import numpy as np

embed_model = SentenceTransformer("BAAI/bge-m3")
# index, chunks 见 13.2 §3.1 假设已建好

def llm_generate(prompt: str) -> str:
    # 接你自己的 LLM client，这里用伪代码
    return client.chat(prompt, max_tokens=200)

def hyde_retrieve(query: str, k: int = 10):
    # 1. 让 LLM 先生成一段 hypothetical answer（不要求事实正确）
    hyde_prompt = f"请用一段 100 字以内的话回答下面问题，可以编造细节：\n{query}"
    hypothetical_doc = llm_generate(hyde_prompt)
    # 2. 用 hypothetical doc 的 embedding 去 retrieve（不用 query embedding）
    h_emb = embed_model.encode([hypothetical_doc],
                               normalize_embeddings=True).astype(np.float32)
    _, idx = index.search(h_emb, k)
    return [chunks[i] for i in idx[0]]
```

关键点：用的是 `embed(hypothetical_doc)` 而不是 `embed(query)`——doc-shape 的伪答案与真 doc 在同一语义层面，余弦更"对齐"。HyDE 在 query 已经够长够清晰时通常没收益，**评测时一定 A/B 对比 vanilla RAG**。

### 3.2 RAG-Fusion + RRF

```python
def multi_query_rewrite(query: str, n: int = 4) -> list[str]:
    prompt = (f"请把下面查询改写成 {n} 个不同表述但语义等价的版本，"
              f"每行一个，不要编号：\n{query}")
    rewrites = llm_generate(prompt).strip().split("\n")
    return [query] + [r.strip() for r in rewrites if r.strip()]   # 原 query 也加入

def rag_fusion_retrieve(query: str, k: int = 10, k_rrf: int = 60):
    queries = multi_query_rewrite(query)
    rank_lists = []
    for q in queries:
        q_emb = embed_model.encode([q],
                                   normalize_embeddings=True).astype(np.float32)
        _, idx = index.search(q_emb, k * 5)            # 每路多召一些给 RRF 用
        rank_lists.append({int(i): r + 1 for r, i in enumerate(idx[0])})
    # RRF 融合
    all_ids = set().union(*[r.keys() for r in rank_lists])
    rrf = {i: sum(1 / (k_rrf + r.get(i, 1e9)) for r in rank_lists)
           for i in all_ids}
    top = sorted(rrf.items(), key=lambda x: -x[1])[:k]
    return [chunks[i] for i, _ in top]
```

`k_rrf=60` 用 Cormack 2009 的原始推荐；`n=4` query 变体（含原 query 共 5 个）是经验甜点。**不要超过 5 个变体**——LLM 容易开始造重复 query，retrieval 噪声反而上升。

### 3.3 Self-RAG 简化 prompt 流程（注意：真 Self-RAG 必须训）

```python
# 注意：纯 prompt 仿不出真 Self-RAG（reflection token 必须 SFT 训）
# 这里给一个"prompt 化简版"，捕获 Self-RAG 的核心思想：retrieve 自适应 + 自我评估
def self_rag_simplified(query: str, retrieve_fn) -> str:
    # Step 1: 决定要不要 retrieve
    decide_prompt = (f"问题：{query}\n"
                     f"这个问题需要外部知识吗？只回答 yes 或 no。")
    if llm_generate(decide_prompt).strip().lower().startswith("n"):
        return llm_generate(query)                    # 不 retrieve 直接答

    # Step 2: retrieve + 对每个 chunk 打 relevance
    chunks = retrieve_fn(query, k=5)
    relevant = []
    for c in chunks:
        rel = llm_generate(f"chunk:\n{c}\n\n问题:{query}\n这个 chunk 相关吗(yes/no):")
        if rel.strip().lower().startswith("y"):
            relevant.append(c)

    # Step 3: 用相关 chunk 生成答案 + self-critique
    ctx = "\n\n".join(relevant) if relevant else "(无相关材料)"
    ans = llm_generate(f"根据材料回答:\n{ctx}\n\n问题:{query}")
    critique = llm_generate(f"答案:{ans}\n材料:{ctx}\n答案是否被材料支持(yes/no):")
    return ans if critique.strip().lower().startswith("y") else f"{ans}\n[未被材料完全支持]"
```

工业里跑 prompt 化简版要做 4 次 LLM call——慢但门槛低。要追求论文里的效果必须按 Asai 2023 §4 的方法 SFT + critique data distillation。

### 3.4 GraphRAG 简化 entity extraction + community summary（伪代码）

```python
# 简化的 GraphRAG 离线建图流程（伪代码 — 工业实现见 Microsoft graphrag repo）
import networkx as nx
from collections import defaultdict

EXTRACT_PROMPT = """从下面文本抽出 entity 和 relation。格式：
ENTITIES: name1|type1|description1; name2|type2|description2
RELATIONS: src1|tgt1|relation1; src2|tgt2|relation2

文本：{chunk}"""

def build_graph(chunks: list[str]) -> nx.Graph:
    G = nx.Graph()
    for c in chunks:
        out = llm_generate(EXTRACT_PROMPT.format(chunk=c))
        ents, rels = parse_entities_relations(out)        # 解析成结构化
        for name, typ, desc in ents:
            if name in G.nodes:
                G.nodes[name]["desc"] += " " + desc       # 合并多 chunk 的描述
            else:
                G.add_node(name, type=typ, desc=desc)
        for src, tgt, rel in rels:
            G.add_edge(src, tgt, rel=rel)
    return G

def detect_communities(G: nx.Graph) -> dict:
    # Leiden / Louvain 算 hierarchical communities
    from networkx.algorithms.community import louvain_communities
    return louvain_communities(G, resolution=1.0)

def summarize_communities(G, communities) -> dict:
    summaries = {}
    for cid, members in enumerate(communities):
        sub = G.subgraph(members)
        info = "\n".join(f"{n}: {sub.nodes[n]['desc']}" for n in sub.nodes)
        summaries[cid] = llm_generate(f"用 100 字总结这个社群:\n{info}")
    return summaries
# query 时：先匹配 community summary（粗粒度），再 drill-down 到 community 内 chunk（细粒度）
```

工业 GraphRAG 还有 multi-level community（hierarchical Leiden）、map-reduce 答案生成等细节，参考 Microsoft 开源的 `graphrag` repo。这里 30 行抓住核心：**entity extraction → graph 合并 → community detection → community summary**。

---

## 4. 工程踩坑与经验

- ❗ **HyDE 在 query 已经长 + 含义清晰时没收益（甚至变差）**：HyDE 的前提是 "query 太短不足以匹配长 doc"。一旦 query 本身就有 50+ token、术语充分（比如长 form 的 prompt-style 问题），LLM 生成的 hypothetical doc 反而引入 hallucination 和噪声，retrieve 质量掉点。**上线前一定 A/B 测一下 vanilla vs HyDE**，别信 paper 一句话就 enable。
- ❗ **Multi-query 的 query 数 > 5 边际收益快速递减**：3-5 个 query 是甜点，6 个开始 LLM 造重复，10 个之后 retrieval 噪声 > 多样性收益。RRF 也救不了——它只看 rank，不区分"多个 query 都召回的真好结果"和"一个垃圾 query 反复召回的垃圾"。
- ❗ **Self-RAG 不能纯 prompt 实现**：reflection token（`[Retrieve]` / `[Supported]` 等）必须加入词表 + SFT 训练，否则 model 输出的不是真 token 而是字符串，无法 driving decoding。**网上很多 "Self-RAG with prompt only" 的教程只是模仿了思想，达不到 paper 的效果**。要么按 Asai 2023 §4 训练，要么用 CRAG（不需要训 LLM）。
- ❗ **GraphRAG 建图 cost 高得惊人**：每 chunk 1 次 LLM call 抽 entity，再加 community summary——百万 doc 用 GPT-4o 抽一次大概 几百 USD 起步，用 GPT-4 上千 USD。建议 **先用 LLaMA-3-8B / Qwen-2.5-7B 等开源 model 在私有 GPU 上抽**（一台 A100 几小时），关键 entity 再用 GPT-4 校正。
- ❗ **CRAG 的 retrieval evaluator 不能省**：偷懒用 BM25 score / 余弦距离做 evaluator 几乎等于没有——这俩在 13.2 已经是 retrieve 的依据，再用一次没新信息。要么训一个 T5-base 评估器（论文方案），要么直接调 GPT-4 / Claude（贵但准）。**别用 dense retrieval 的 cosine 当 evaluator**——这是常见的工业 anti-pattern。
- ❗ **不要为复杂而复杂——80% 场景 vanilla RAG + reranker 已够**：一上来就 GraphRAG + Self-RAG + Agentic RAG 堆栈是新手最常见的过度工程。工业实战中 80% 的 RAG 应用 **vanilla RAG + bge-reranker + 好的 chunking 策略** 就够了。先 baseline 到 RAGAS faithfulness > 0.85 再考虑加 fancy 方法，每加一层都要跑 A/B 验证，没收益就回退。
- ❗ **Agentic RAG 必加 max_step + token budget**：agent loop 容易死循环——retrieve 完觉得"还不够全面"就再 retrieve，再不够就改写 query 再 retrieve，到 step 20 还在 retrieve。**至少加 `max_step=5-10`、每步 token budget、early stop（answer confidence > threshold 就停）**。Perplexity 等生产系统的 agent loop 通常硬限制在 3-5 轮。
- ❗ **R1 + RAG 时 long CoT 与 RAG context 共存，token 爆炸要 budget**：reasoning model 一段 CoT 几千 token，RAG context 又几千 token，几轮 retrieve 之后 context 过 50k 是常态。**必须做 history compression**——历史 retrieve 结果只保留 summary 不保留原文、CoT 跨步骤截断。Search-R1 论文里专门用 "context masking" 技巧避免 token 爆炸。
- ❗ **GraphRAG 的 entity dedup 是大坑**：LLM 抽 entity 时同一实体可能被写成 "Apple" / "Apple Inc." / "苹果公司" / "苹果"——直接按 name 合并会丢掉关系，全分开又会让 graph 碎片化。工业里需要 **entity resolution / linking** 模块（embedding 聚类 + LLM 校验），这部分 paper 里轻描淡写但工程量很大。
- ❗ **HyDE / RAG-Fusion / Self-RAG 加在一起 ≠ 效果叠加**：很多人觉得"既然每个都涨点那都加上一定更好"——错。多个方法在 retrieval 维度上互相干扰：HyDE 的 hypothetical doc + multi-query 的多 query embedding 一起跑，candidate 池被严重 over-sample，reranker 反而被噪声压垮。**逐个 ablation 加，每加一个看 NDCG@10 / RAGAS 是否真的涨**，不涨就回退。

---

## 5. 经典 paper

- **Gao et al., 2022 — Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE)** — HyDE 原典，提出"用 LLM 生成 hypothetical doc 再 retrieve"的范式。读 §3 看 BM25 / dense / HyDE 在 zero-shot 跨语言 retrieval 上的对比；论文最大贡献是证明 LLM-generated hypothetical doc 即使有错也能显著提升 dense retrieval recall。
- **Asai et al., 2023 — Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection** — Self-RAG 原典，定义 4 类 reflection token、给出 critique data 自动构造方案、tree-decoding 推理算法。读 §3-4 理解 reflection token 设计与训练数据生成 pipeline；这篇之后"自适应 retrieve" 才成为标准能力。
- **Yan et al., 2024 — Corrective Retrieval Augmented Generation (CRAG)** — CRAG 原典，提出"retrieval evaluator + web search 兜底"的轻量 plug-in 方案。读 §3 看三档评估（correct / ambiguous / incorrect）的细节、知识精炼算法（decompose-then-recompose）；与 Self-RAG 互补——一个改 LLM 一个不改。
- **Edge et al., 2024 — From Local to Global: A Graph RAG Approach to Query-Focused Summarization** — Microsoft GraphRAG 原典，把 entity-relation graph + community detection + map-reduce summary 完整工程化。读 §2-3 看建图 pipeline + community summarization + global query 流程；理解 chunk-RAG 解决不了的 whole-corpus 问题怎么破。
- **Hou et al., 2025 — Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning** — Search-R1 用 RLVR 训 reasoning model 在 CoT 中自然调 search 的代表作。读 §3-4 理解 RL reward 设计、context masking 防 token 爆炸；与 Module 15.4 衔接，是 R1 + RAG 范式的代表。
- **Raudaschl, 2023 — Forget RAG, the Future is RAG-Fusion** — RAG-Fusion 的原始博客（不是 peer-reviewed paper，但社区影响力大），把 multi-query + RRF 的方案 popular 化。读完能直接搭一套 RAG-Fusion，工程价值高。

---

## 6. 自测与面试题

**Q1（对比）**：HyDE / RAG-Fusion / Self-RAG / GraphRAG 各自解决 vanilla RAG 的什么痛点？给一个判断"什么场景用哪个"的决策树。

<details>
<summary>Answer sketch</summary>

四个方法对应四个不同 limitation：

- **HyDE** → 解决 **短 query 与长 doc 的 embedding 不对齐**（query embedding 与 doc embedding 在语义空间不在同一密度层）。先让 LLM 生成 hypothetical doc，用 doc-shape 伪答案的 embedding 去 retrieve，做"同形态匹配"。
- **RAG-Fusion** → 解决 **单 query 召回不全 / query 含糊或多义**。LLM 改写出 3-5 个 query 变体，每个 retrieve，再用 RRF 融合 ranking。
- **Self-RAG** → 解决 **vanilla RAG 无脑 retrieve + 生成时不知道该不该信**。SFT 训练 LLM 输出 reflection token（[Retrieve] / [Supported] 等）做自适应 retrieve + 自我评估。
- **GraphRAG** → 解决 **chunk 视角太局部、whole-corpus 综合答不出**。离线把 doc 抽成 entity-relation graph + community summary，query 时按 community 粒度先 match 再 drill down。

决策树（简化版）：

- query 是 specific fact 查询？→ vanilla RAG (+ reranker)，不需要进阶
- query 短且术语稀疏 / doc 长且密？→ 加 HyDE
- query 含糊 / 用户表达与文档术语不一致？→ 加 RAG-Fusion
- 高 factuality 要求 + 私有化部署可以训 LLM？→ Self-RAG
- 高 factuality 要求 + 不想训 LLM？→ CRAG（plug-in evaluator + web search 兜底）
- 问的是 whole-corpus 综合（"主要主题" / "关系演变"）？→ GraphRAG / LightRAG
- 复杂 multi-step 不确定步数？→ Agentic RAG（注意 max_step）
- 任务是 reasoning + retrieval 复合 + 可以训？→ R1 + RAG / Search-R1

加分点：可以补充"不要为复杂而复杂——80% 场景 vanilla RAG + reranker 就够"。

</details>

**Q2（实战）**：你做一个法律咨询 RAG，文档 50 万 page、要求 high factuality + 跨多 doc 综合（"找出所有提到 X 法条的合同并总结争议点"）。给完整方案，包括 chunking / embedding / vector DB / 进阶 RAG 选型 / 评测。

<details>
<summary>Answer sketch</summary>

参考方案：

- **chunking**：按法律文档结构切（条款 / 章节为 atomic unit），用 document-aware splitter，chunk 600 token + overlap 100，保留 metadata（法条编号、合同 ID、签订日期）。50 万 page ≈ 5-10M chunks。
- **embedding**：bge-m3（中英双语 + dense + sparse 同时输出，适合中英混合的法律文档）；离线 indexing 用 GPU 集群跑，约 100-500 GB embedding 数据。
- **vector DB**：Milvus 集群（HNSW + IVF 分层 + sparse index），亿级别向量必须分布式 + 多副本。
- **hybrid retrieval**：dense (bge-m3 dense) + BM25（法条编号、当事人姓名等专有词必须 lexical match）+ RRF 融合。
- **进阶 RAG 选型**：
  - **HyDE**：法律 query 通常较长且术语充分（"X 法第 Y 条第 Z 款"），HyDE 收益不大，不加。
  - **RAG-Fusion**：用户问询表达多样，加 RAG-Fusion（n=3）能覆盖更多召回，加。
  - **GraphRAG**：跨多 doc 综合是 GraphRAG 的强项——抽 entity（法条编号、当事人、争议点、合同类型）建 graph，按法条做 community，回答 "找出所有提到 X 法条的合同" 类问题。建议 **离线 GraphRAG 索引 + 在线 vanilla RAG**：先 graph 选定相关 community 的 doc 子集，再在子集内做 vanilla RAG retrieve，避免每次 query 都跑全 graph。
  - **CRAG**：法律对 factuality 极高要求，加 retrieval evaluator + 权威法规库 fallback（不用 web search 用法规专库），incorrect 时切换权威库。
  - **不加 Self-RAG**：训 LLM 成本高、法律 LLM 私有化部署人力少。
- **reranker**：bge-reranker-v2-m3 + 法律领域 fine-tune（用历史咨询的 (query, doc) pairs SFT）。
- **LLM**：私有化部署 Qwen2.5-72B 或 DeepSeek-V3，法律领域 SFT。Prompt 模板严格要求 **只引用 [doc-id] / [法条编号]，没找到就说不知道**。
- **评测**：
  - retrieval 端：NDCG@10、Recall@100
  - end-to-end：RAGAS 三件套 + 法律专家人工评 200 条 + 引用准确率（引到的法条是否真的存在 + 内容是否对应）
  - 在线监控：用户反馈 + LLM-as-judge 抽样 5%
- **额外考虑**：法律文档版本管理（法条修订、合同 amendment）必须走 vector DB 软删除 + 版本号 metadata，retrieve 时按时间过滤。

加分点：提到 GraphRAG 离线 + vanilla RAG 在线的混合架构、CRAG 的兜底库换成法规专库、引用准确率作为独立评测指标。

</details>

**Q3（前沿）**：Agentic RAG 与 R1 + RAG（如 Search-R1）的差异是什么？什么场景下用哪个？

<details>
<summary>Answer sketch</summary>

核心差异在 **retrieval 决策的承载者** 与 **训练范式**：

- **Agentic RAG**：retrieval 是 **外层 agent loop** 决定的——LLM 是 reasoning + tool call 的"executor"，但调用 search 的逻辑写在 agent framework 的 prompt / control flow 里（"如果 model 输出 `<tool_call>` 就调，否则继续"）。可以 **零训 / 纯 prompt** 实现（基于通用 instruct LLM 的 ReAct / function calling 能力）。代表：LangGraph + Anthropic Claude tool use、AutoGen。
- **R1 + RAG / Search-R1**：retrieval 是 **LLM 在 CoT 内部** 自然 emit 的——`<search>...</search>` 是 model 在 reasoning 流中决定的，不是外层逻辑控制的。**必须 RL 训练**（RLVR：根据最终答案对错回传 reward，learn to retrieve 的时机和 query 写法）。代表：Search-R1、ReSearch、ReTool、Agent-R1。

| 维度 | Agentic RAG | R1 + RAG |
|---|---|---|
| retrieve 决策位置 | 外层 agent loop | LLM CoT 内部 |
| 是否要训 LLM | 否（prompt 可用） | **是**（RLVR） |
| 灵活性 | 高（任意 tool 编排） | 受训练分布限制 |
| reasoning 与 retrieval 衔接 | 弱（retrieve 完才接着 reason） | 强（在 CoT 中自然交错） |
| 部署门槛 | 低 | 高（要 RL infra） |

场景选择：

- **复杂 multi-step research / 多种 tool 编排（search + code + DB + ...）** → Agentic RAG（灵活、零训上线）
- **task 收敛在 reasoning + search 复合 + 有训练资源** → R1 + RAG，效果上限更高（reasoning 与 retrieval 在 CoT 内部深度耦合，避免外层 agent 的 "retrieve 完忘了为什么要 retrieve"）
- **快速 prototype / 业务方做 demo** → Agentic RAG
- **大规模深度研究产品（Perplexity Pro / OpenAI Deep Research）** → 两者结合：底层 reasoning model 用 R1 范式训过，外层再套 agent framework 编排多 tool

加分点：提到 R1 + RAG 训练时 long CoT 与 retrieve context 容易 token 爆炸，需要 context masking / history compression；Agentic RAG 容易死循环，需要 max_step。

</details>

---

## 7. 延伸阅读

- [Microsoft GraphRAG repo](https://github.com/microsoft/graphrag) — Edge 2024 论文的官方开源实现，能直接跑 entity extraction + community detection + global/local query 全 pipeline。
- [LangChain — Advanced RAG cookbook](https://python.langchain.com/docs/concepts/rag) — HyDE / multi-query / parent-document retrieval / contextual compression 等 advanced retrieval 方法的工程实现合集。
- [LlamaIndex — Advanced retrieval strategies](https://docs.llamaindex.ai/en/stable/optimizing/advanced_retrieval/advanced_retrieval/) — 与 LangChain 互补的另一个 RAG framework，对 GraphRAG / CRAG / Self-RAG 都有官方 demo。
- [Anthropic — Building effective agents](https://www.anthropic.com/research/building-effective-agents) — Agentic RAG 的工程指导，强调 "augmented LLM as building block"、何时 simple workflow 何时 agent。
- [HKUDS LightRAG repo](https://github.com/HKUDS/LightRAG) — GraphRAG 的工程化轻量替代，建图 cost 低、效果接近，几十万 doc 量级首选。
- 推荐继续读本教程的 **13.4《Function calling 工程》**——进阶 RAG 与 function calling 是 Agent（Module 14）的两大基础能力。
