---
title: "6.2 数据管线：FineWeb / Dolma / DCLM / 清洗去重配比"
description: "把 LLM 预训数据从 The Pile（2020）一路梳理到 FineWeb-Edu / DCLM（2024）的脉络，并把现代数据 pipeline 的五个阶段——URL/语言过滤 → 正文提取 → 去重 → 质量过滤 → domain mixing——拆开讲清算法、工具与踩坑，让你理解为什么 2024 年之后整个领域的共识从\"更多数据\"翻转成了\"更好的数据 > 更多的数据\"。"
---

> ⏱ 预计阅读 45 分钟 ｜ 难度 ★★★ ｜ 前置：3.1 BPE/SentencePiece（数据进 tokenizer 之前的所有事）

## 一句话本节讲什么

把 LLM 预训数据从 The Pile（2020）一路梳理到 FineWeb-Edu / DCLM（2024）的脉络，并把现代数据 pipeline 的五个阶段——**URL/语言过滤 → 正文提取 → 去重 → 质量过滤 → domain mixing**——拆开讲清算法、工具与踩坑，让你理解为什么 2024 年之后整个领域的共识从"更多数据"翻转成了"**更好的数据 > 更多的数据**"。

---

## 1. Mental model（直觉）

预训练这件事，**90% 的人花 90% 的精力调架构、调超参，但真正决定模型上限的是数据**。同一个 7B 架构，数据从 RedPajama 换成 DCLM，MMLU 能差 5-8 个点；同一份 1B 模型用 FineWeb-Edu 训 350B token，能打过用普通 web 数据训 1T token 的同 size 模型——这是 2024 年之后整个行业反复验证过的事实。

那预训练数据长什么样？想象一下：

```
                    ┌────────────────────────────────────┐
互联网原始 dump  →  │  Common Crawl (PB 级，≈ 2500 亿网页) │
                    └──────────────┬─────────────────────┘
                                   │
            ┌──────────────────────▼──────────────────────┐
            │  阶段 1: URL 过滤 + 语言识别                 │  ← 砍掉 75% 网页
            └──────────────────────┬──────────────────────┘
            ┌──────────────────────▼──────────────────────┐
            │  阶段 2: 从 WARC 用 trafilatura 提正文       │  ← 砍掉 nav/footer/ads
            └──────────────────────┬──────────────────────┘
            ┌──────────────────────▼──────────────────────┐
            │  阶段 3: MinHash + LSH 近似去重              │  ← 砍掉 50-80% 文档
            └──────────────────────┬──────────────────────┘
            ┌──────────────────────▼──────────────────────┐
            │  阶段 4: 质量过滤（规则 + 分类器 + LLM-judge）│ ← 砍掉 70-90% 剩余
            └──────────────────────┬──────────────────────┘
            ┌──────────────────────▼──────────────────────┐
            │  阶段 5: 多 source 按比例 mixing             │  ← + code/math/wiki
            └──────────────────────┬──────────────────────┘
                                   ▼
                        15T → 1T-3T 高质量 token
```

整个过程像炼油：**输入 PB 级原油，输出 TB 级高纯度燃料**——每一步都是大幅减量、定向提纯。FineWeb 团队公开过一张转化率：从一个 Common Crawl snapshot（约 90TB 压缩 HTML）到最终训练用的英文文本，**只有约 1-2% 留下来**，剩下全砍。

为什么这么狠？因为 LLM 是"压缩器"——它把训练 corpus 压缩进参数里。喂垃圾就压缩垃圾：模板 spam、SEO 农场、机器翻译低质英文、被 LLM 已经污染过的 web 文本，每一份都在浪费 FLOPs、占据本可以学有用知识的参数容量。

本节剩下三个核心问题：(1) 现代主流开源 corpus 是哪些、按什么思路演进的；(2) pipeline 五阶段每一步用什么算法、什么工具；(3) 为什么 quality 比 quantity 重要——以及它的边界在哪。

> 与 8.1 SFT 数据构造的区分：本节是"无监督 web pretrain 数据"——量是 trillion 级 token、目标是覆盖广度；8.1 是"指令对话数据"——量是几万到几百万 sample、目标是格式与对齐。技术工具有重叠（去重、质量分类器都用），但**数据规模差 5-7 个数量级**，方法论完全不同。

---

## 2. 公式与原理

### 2.1 预训 corpus 演化时间线

| 年份 | 数据集 | 规模 | 出品 | 关键贡献 |
|---|---|---|---|---|
| 2019 | **OpenWebText** | 40GB | 社区 | 第一个公开复现 GPT-2 的 WebText |
| 2019 | **C4** | ~750GB | Google (T5) | Common Crawl 简单清洗，长期 baseline |
| 2020 | **The Pile** | 825GB | EleutherAI | 22 个子集（books、GitHub、ArXiv、StackExchange...）的"组合 corpus"范式 |
| 2023 | **RedPajama-1T** | 1.2T | Together | 复现 LLaMA-1 配方的开源版 |
| 2023 | **RedPajama-v2** | 30T | Together | 量极大但只做去重 + 规则，无强 quality filter |
| 2023 | **SlimPajama** | 627B | Cerebras | RedPajama 严格去重版本，质量显著好于原版 |
| 2023 | **Dolma** | 3T | AI2 | 完全 open（含 license metadata），OLMo 用 |
| 2024 | **FineWeb** | 15T | HuggingFace | 现代清洗 + fastText 质量分类器，新 baseline |
| 2024 | **FineWeb-Edu** | 1.3T | HuggingFace | LLM-as-judge 选教育内容，量小但 quality 极高 |
| 2024 | **DCLM** | 3.8T | Apple/UWashington | model-based filter，"currently SOTA" |
| 2024 | **The Stack v2** | 900B | BigCode | 主流开源 code corpus，含 license 信息 |

**中文预训 corpus**：WuDao Corpora（智源，3TB）、Skywork Open Web（昆仑万维，150B token）、ChineseFineWeb / CCI3（智源在 FineWeb 流程上的中文版）、MAP-CC（多源中文，800B+），整体**比英文落后约一代**——FineWeb 级的中文 corpus 直到 2024 年下半年才出现。

几条规律值得记：

- **The Pile 树立"组合 corpus"范式**——预训不只是 web，要混进 books、code、math、science——后来所有团队都沿用
- **RedPajama → SlimPajama → FineWeb 是同一根线的迭代**：一开始追量，发现"去重就是 free lunch"，再发现"质量分类器更狠"
- **FineWeb-Edu 与 DCLM 是两条不同的"质量优先"路线**：前者用 LLM-as-judge 直接挑高分文档，后者用 model-based perplexity / classifier 大规模过滤
- **Dolma 的独特价值是 100% open**——含每篇文档的来源 URL、license、采集时间，对学术研究、可复现至关重要（OLMo 是配套模型）

### 2.2 数据 pipeline 五阶段

#### 阶段 1：URL / 语言过滤

**URL blocklist**：第一道粗筛——成人、暴力、赌博、已知 spam 域名直接整站丢弃。社区事实标准是 [UT1 blocklist](https://dsi.ut-capitole.fr/blacklists/index_en.php)（图卢兹大学维护），FineWeb / DCLM 都用。

**语言识别**：用统计模型对每篇文档判断主语言。主流工具：

- **fastText langid**（Bojanowski 2016）—— Facebook 出品，176 种语言，速度快（~1ms/doc），是 FineWeb 用的方案
- **pyCLD3** —— Google Compact Language Detector，160 种语言
- **GlotLID** —— 2023 新出，覆盖 1665 种语言，对低资源语言更好

实际经验：fastText 在欧洲语言上 > 99% 准确，但**对中日韩混合文本、code switching 文本会出错**——比如 GitHub README 经常英中混合，可能被判成纯中文丢弃。FineWeb 只保留英文（confidence > 0.65），结果约 25% 的 Common Crawl 网页通过这一关。

#### 阶段 2：正文提取（boilerplate removal）

互联网网页平均 70-80% 的字节是 HTML 标签、CSS、导航栏、页脚、广告、cookie 弹窗——只有 20-30% 是真正的"正文"。这一步要**从 raw HTML 提出干净的正文**。

**Common Crawl 提供三种格式**：

- **WARC**：原始 HTTP 响应（含完整 HTML），最大但最完整
- **WAT**：metadata（URL、Content-Type、links），不含正文
- **WET**：CC 自己提取过的纯文本，**质量很差**——正则切的，nav 和 ads 经常没切干净

**现代做法（FineWeb / DCLM）：从 WARC 自己用 trafilatura 重新提取**——WET 已经被业界公认"不能用"，所有严肃 corpus 都从 WARC 重提。

**提取工具**：

- **trafilatura**（Barbaresi 2021）—— Python 写的事实标准，准确率最高、速度合理，FineWeb 用它
- **readability-lxml** —— Mozilla 算法的 Python 移植，比较老但稳定
- **Justext** —— 基于 boilerplate 启发式
- 自写 **BeautifulSoup** —— 通用但不够智能

trafilatura 内部的核心逻辑是 DOM tree 启发式 + ML 分类器：识别 `<article>` `<main>` `<p>` 这类正文容器，过滤 `<nav>` `<aside>` `<footer>`，再对剩余文本做密度判断（链接密度低 = 正文，链接密度高 = 列表/导航）。

#### 阶段 3：去重 (Deduplication)

**Lee 2021 的核心发现**：**去重大幅提升训练效果**——dedupe 后的 model 在同样 step 数下 perplexity 显著更低、下游任务更强、且**生成时 verbatim memorization 大幅减少**（避免输出训练集原文，对隐私和 license 都重要）。dedupe 是预训数据 pipeline 公认的"free lunch"——成本低、收益大、几乎没坏处。

**两类去重**：

**(a) Exact dedupe**：精确匹配——用 hash（MD5 / SHA256）算每个文档（或段落）的指纹，相同 hash 的丢一份。能抓"完全 copy-paste 的转载"，但抓不住"换了几个字的近似重复"。

**(b) Near-duplicate (fuzzy) dedupe**：抓"语义/字面 90% 相似"的文档。**MinHash + LSH 是工业标准**，原理：

1. **n-gram 化**：把每个文档切成 n-gram（典型 5-gram）的 set $S = \{g_1, g_2, \dots\}$
2. **MinHash signature**：取 $K$ 个 hash 函数 $h_1, \dots, h_K$（典型 $K = 128$），signature 第 $k$ 维是 $\min_{g \in S} h_k(g)$。**关键性质**：两个 set 的 MinHash signature 在第 $k$ 维相等的**期望概率 = Jaccard 相似度** $J(S_1, S_2) = |S_1 \cap S_2| / |S_1 \cup S_2|$。signature 长度 $K$ 决定估计精度
3. **LSH (Locality Sensitive Hashing)**：把 $K$ 维 signature 切成 $b$ 个 band、每 band 长 $r$（满足 $b \cdot r = K$），同一 band 完全相同的两个文档进同一 bucket 当 candidate
4. **Verify**：候选对计算精确 Jaccard，> threshold（典型 0.8）判为 dupe，丢一份

LSH 的 $(b, r)$ 选择决定 recall/precision trade-off：候选对 collide 的概率近似 $1 - (1 - s^r)^b$，其中 $s$ 是真实 Jaccard——这是个 S 形曲线，**$(b, r)$ 调拐点位置**。FineWeb 用 $K=112$、$(b, r) = (14, 8)$，DCLM 与 SlimPajama 用 $(K=128, b=8, r=16)$ 这种偏 recall 的配置。

**SimHash** 是另一种 fuzzy 去重方案（Charikar 2002，Google 早期用），把文档投影成 64-bit 指纹、按 Hamming 距离判重——比 MinHash 快但 recall 略低，**在 LLM 数据领域 MinHash 更主流**。

**去重的两种粒度**：

- **document-level**：整个文档比对——快但只抓"整篇重复"
- **paragraph-level**：每个段落单独比对——能抓"局部抄袭"，更细
- **substring dedupe** (Lee 2021)：用 suffix array 找跨文档共享的长子串（例如 50+ token 完全相同的片段直接删），是最严格的一档

工程上 FineWeb 做 document + paragraph 两层 MinHash，DCLM 在 FineWeb 基础上又加了一轮跨 snapshot 去重——**Common Crawl 不同月的 snapshot 之间重复极多**，跨 snapshot 不去重等于训了好多遍同样的内容。

#### 阶段 4：质量过滤

这是 2024 年之后竞争最激烈的环节。从粗到精四档：

**(a) 规则过滤（Gopher rules，Rae 2021）**——DeepMind 在 Gopher paper 里公开的一组启发式：

- 文档长度 50-100,000 字符
- 平均词长 3-10
- 含 stopword 占比 > 0.06（"the / and / of"等占比太低多半是机器生成或无意义文本）
- 字母占比 > 0.7（过多数字/符号丢）
- 过多重复 line / 重复 n-gram 丢

这套规则简单粗暴但**至今仍是所有现代 pipeline 的基础**——FineWeb / DCLM / Dolma 都先跑一遍 Gopher 规则。

**(b) fastText 质量分类器**——用"高质量 reference 语料"（典型 Wikipedia / 书籍）作正例、随机 web 文本作负例训二分类，对每篇文档输出 quality score。FineWeb v1 用的就是这套，门槛通常取 score > 0.65。

**(c) Model-based filter（DCLM 范式）**——用一个小 LM（比如已经训好的 1B 模型）对每个文档算 perplexity，**低 perplexity 留下、高 perplexity 丢**。直觉：训好的 LM 的 perplexity 反映"这文档有多接近自然、连贯的文本"——被广告关键词堆砌的 spam、机翻劣质内容 perplexity 会显著偏高。DCLM 用这套筛出 3.8T token，对应模型在 MMLU 上比同 size 用 RedPajama 训的模型高 5-8 个点。

**(d) LLM-as-judge filter（FineWeb-Edu）**——直接用大 LLM（FineWeb-Edu 用 Llama-3-70B-Instruct）对每篇文档打 0-5 分"教育性"，留 ≥ 3 分的。结果 1.3T token 的 FineWeb-Edu，**同样训 350B token，下游 benchmark 几乎全面打过 15T 全量 FineWeb**——这是"quality > quantity"最有力的实证。

代价是 **LLM-as-judge 推理成本极高**——给 15T token 全部打分需要数百万 GPU 小时，所以实操是分两阶段：先用 LLM 给小样本打分，再训一个小 BERT 当"分数预测器"对全量打分。

**(e) Domain mixing 时的隐式质量**——后面阶段 5 的 mixing 比例本身也是质量过滤的延伸：把 Wikipedia / books / arXiv 的权重提高，等于变相"加权高质量 source"。

#### 阶段 5：Domain mixing

预训语料不只是 web——通常还要混 code、math、books、wiki、multilingual、science papers。混合比例对模型能力影响很大：

- code 占比上去，HumanEval 涨但常识掉
- math 占比上去，GSM8K / MATH 涨但闲聊 disfluent
- multilingual 占比上去，跨语言能力涨但单语言 perplexity 略降

主流配方（典型 7B-70B 量级）：

```yaml
web:           0.65
code:          0.10
books:         0.10
math:          0.05
wiki:          0.05
multilingual:  0.05
```

**怎么选最优 mixing weight**？两条研究路线：

- **DoReMi**（Xie 2023）—— Distributionally Robust Optimization：先训一个 reference model，再训一个"domain proxy" model 找让 reference 提升最大的权重组合。**实证比简单平均涨 1-2 个 MMLU 点**
- **RegMix**（Liu 2024）—— 用回归预测：训若干小 model 在不同 mixing 下的 loss，拟合 (mixing → loss) 的回归模型，用回归预测最优 mixing
- **Data Mixing Laws**（Ye 2024）—— scaling law 风格：拟合 loss 关于 (data scale × mixing) 的联合 law，外推到大模型

工程上，多数团队没条件跑 DoReMi——直接抄 LLaMA / Qwen 的公开配方，再小幅调即可。

### 2.3 三组现代 corpus 对比

| Dataset | Size | Filter 类型 | 特点 | 典型 user |
|---|---|---|---|---|
| **C4** | 750GB | 简单规则 | 老 baseline | T5、早期 LLaMA |
| **The Pile** | 825GB | 子集 curate + 规则 | 22 source 组合范式 | GPT-Neo、GPT-J、Pythia |
| **RedPajama-v2** | 30T | 弱（去重 + 规则） | 量大但 quality 一般 | 部分研究复现实验 |
| **SlimPajama** | 627B | RedPajama 严格去重 | 中等 quality、中等量 | TinyLlama、研究界 baseline |
| **Dolma** | 3T | open + Gopher 规则 | 完全 open（license） | OLMo |
| **FineWeb** | 15T | fastText + 严格 dedup | 量大且 quality 较好 | 多数 2024 后开源 LLM 复现 |
| **FineWeb-Edu** | 1.3T | LLM-as-judge | quality 最强但 size 小 | SmolLM 系列 |
| **DCLM-baseline** | 3.8T | model-based filter | 当前 SOTA | DCLM-7B、Apple 研究 |
| **The Stack v2** | 900B | license filter + 去重 | 主流 code source | StarCoder2 |

### 2.4 一段话：data quality 影响有多大

DCLM paper 的关键消融：**同 1B 和 7B 模型、同 token 预算（260B / 2.5T），训 DCLM-baseline 的模型在 MMLU 上比训 RedPajama 的模型高 5-8 个点**——这是同尺度下数据 quality 能带来的最大效应之一，**远大于多数架构改动**。FineWeb-Edu 类似实证：1.3T 的 Edu 子集训 350B token 的小模型在 MMLU/HellaSwag 上能压制 15T 全量 FineWeb。

但**别过度神化**——这套结论的边界：

- pretrain 仍然需要 trillion 级 token——SFT 几万 sample 就够，但 pretrain 不行（参考 6.3 scaling law）
- "quality > quantity" 比较的是同 source 类型——不是说 1B token 高质量能取代 1T 低质量，是说"过滤掉 70% 的垃圾后剩下 30% 训得更好"
- quality classifier 本身有 bias——fastText quality 分类器训出来的高分文档偏 Wikipedia 风格，模型学了"维基化"语言，遇到 informal 文本可能掉点

---

## 3. 最小代码示例

### 3.1 trafilatura 提取正文

```python
# pip install trafilatura
import trafilatura

html = trafilatura.fetch_url("https://en.wikipedia.org/wiki/Large_language_model")
text = trafilatura.extract(
    html,
    include_comments=False,    # 评论一般是低质 spam
    include_tables=True,       # 表格里常有事实信息，建议保留
    deduplicate=True,          # 文档内段落去重
    favor_precision=True,      # precision 优先（少噪音）/ recall 优先（少漏字）
)
print(text[:500])
```

实战 tip：`favor_precision=True` 对 LLM pretrain 更安全（宁丢勿错）；`include_tables=True` 对维基/科普类页面尤其重要——FineWeb 默认开。

### 3.2 MinHash + LSH 近似去重 demo

```python
# pip install datasketch
from datasketch import MinHash, MinHashLSH

def mh(text, k=128, ngram=5):
    """把文本变成 MinHash signature"""
    m = MinHash(num_perm=k)
    tokens = text.split()
    for i in range(len(tokens) - ngram + 1):
        m.update(" ".join(tokens[i:i+ngram]).encode())
    return m

doc1 = "the quick brown fox jumps over the lazy dog in the morning sun"
doc2 = "the quick brown fox jumps over the lazy cat in the morning sun"   # 1 词不同
doc3 = "completely unrelated content about quantum computing and qubits"

m1, m2, m3 = mh(doc1), mh(doc2), mh(doc3)
print("Jaccard(1,2) ≈", m1.jaccard(m2))   # ~0.7-0.9，候选 dupe
print("Jaccard(1,3) ≈", m1.jaccard(m3))   # ~0.0

# LSH：批量找候选重复对
lsh = MinHashLSH(threshold=0.8, num_perm=128)
lsh.insert("doc1", m1); lsh.insert("doc2", m2); lsh.insert("doc3", m3)
print("doc1 的 candidate dupes:", lsh.query(m1))   # ['doc1', 'doc2']
```

`datasketch` 是工业用得最多的 Python MinHash/LSH 库；HuggingFace 出的 [`text-dedup`](https://github.com/ChenghaoMou/text-dedup) 是基于它的 distributed dedupe pipeline，FineWeb 跑 15T token 用的就是 `text-dedup` 的 Spark 版本。

### 3.3 fastText quality classifier 用法

```python
# pip install fasttext
import fasttext

# load 预训好的 quality classifier（FineWeb / DCLM 风格的二分类器）
model = fasttext.load_model("quality_classifier.bin")

def quality_score(text):
    text = text.replace("\n", " ")
    labels, probs = model.predict(text, k=2)   # 返回 top-2 label 和概率
    # 假设标签是 __label__hq / __label__lq
    return dict(zip(labels, probs)).get("__label__hq", 0.0)

text = "Photosynthesis is the process by which green plants use sunlight..."
print("quality:", quality_score(text))    # ~0.85，保留
text2 = "BUY VIAGRA NOW!!! click click click discount discount"
print("quality:", quality_score(text2))   # ~0.05，丢
```

DCLM 公开了他们的 fastText classifier 权重（[dclm-baseline-1.0 仓库](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0)），可以直接 load 复用——**自己训一个 classifier 也只需要几小时**：用 OpenWebMath / Wikipedia 当正例、随机 Common Crawl 当负例，fastText supervised mode 跑一次即可。

### 3.4 Domain mixing 配置示例

```yaml
# 一个简化的 7B 量级 LLM 配方（参考 LLaMA / Qwen 公开配比）
sources:
  fineweb_edu:        # 高质量 web
    weight: 0.55
    epochs: 1.5       # 允许过 1.5 个 epoch（高质量重复也 OK）
  the_stack_v2:       # code
    weight: 0.10
    license_filter: ["MIT", "Apache-2.0", "BSD-3"]   # 排除 GPL
  dolma_books:        # books
    weight: 0.10
  proof_pile_2:       # math
    weight: 0.05
  wikipedia_en+zh:    # wiki，多语言
    weight: 0.05
    epochs: 3         # wiki 高质量，多过几遍
  multilingual_cc:    # 多语言 web
    weight: 0.05
  arxiv:              # 科学
    weight: 0.05
  stackexchange:      # QA 风格
    weight: 0.05

total_tokens: 2.0T
context_length: 8192
```

注意 `epochs` 这一项——现代经验是**高质量 source 可以多过几遍（3-5 epoch）**，纯 web 一般不过 epoch（避免 over-memorization）。这个配方与 LLaMA-3、Qwen-2.5 公开的策略一致。

---

## 4. 工程踩坑与经验

- ❗ **Common Crawl 的 WET 不要用**——CC 自己提取的 WET 文件正则切得很糙，nav/footer/cookie 弹窗经常残留。FineWeb / DCLM 全部从 WARC 用 trafilatura 重提，这一步看似"重做轮子"实则是 quality 提升的关键来源之一。代价是 WARC 比 WET 大 5-10×，存储与下载成本高。
- ❗ **MinHash 的 `(K, b, r)` 是个艺术**——$K=128$ 是常见基线，$(b, r) = (8, 16)$ 偏 recall（容易把"勉强相似"也判 dupe）、$(14, 8)$ 偏 precision（更严格）。FineWeb 用 (b=14, r=8)、SlimPajama 用 (b=8, r=16)；threshold 通常取 Jaccard 0.7-0.85。**调错 $(b, r)$ 会出现两种灾难**：(a) 全去重 → 数据少 50% 还训不好；(b) 几乎不去重 → 模型 verbatim memorization 严重。
- ❗ **去重过度会丢合法重复内容**——新闻通稿、ToS 法律模板、API 文档、教科书引用句这些"高重复但合法"的内容，去重过狠会把它们清光，反而让模型学不到"标准用语"。Threshold 0.8 是社区经验拐点，**不要追求 0.95+ 这种极致严格**。SlimPajama 团队就发现某些 source 用 0.7 反而比 0.85 训出来的 model 强（保留更多多样性）。
- ❗ **多语言模型必须做 language ID + balance**——直接喂 Common Crawl 做多语训练，**英文会 dominate 60%+ token**，小语种被边缘化。Qwen / DeepSeek / Gemma 都做"语言重采样"：低资源语言 upsample 2-5 倍，让每种语言至少占 1-3% token 才能学到基本能力。语言 ID 用 fastText 时 confidence threshold 不要太低（< 0.5 会有 code-switch 文本误判，特别坑中文 LLM）。
- ❗ **fastText quality classifier 的"reference bias"**——用 Wikipedia 当正例训出来的 classifier 会偏好"维基百科风格"——formal、第三人称、带引用的文本，结果模型在 informal 对话、口语场景下表现差。**FineWeb-Edu 的 LLM-as-judge 是更"开放"的 quality 定义**（教育性 ≠ 维基风格），所以 quality 上限更高。给自家训 classifier 时，正例集合要多样化（Wikipedia + 教科书 + 科普文 + StackExchange answer 等），不要只用 Wikipedia。
- ❗ **DCLM 风格的 model-based filter 训成本不低**——用 1B perplexity model 给 15T token 打分，需要数千 GPU 小时；用 fastText classifier 几乎免费。**实操 trade-off**：小团队用 fastText（FineWeb 路线），有算力的用 model-based（DCLM 路线）。两者带来的 MMLU 差距约 2-4 点，看资源是否值得。
- ❗ **代码数据必须 license filter**——GitHub 公开代码 ≠ 可商用代码。GPL/AGPL 训出来的模型生成代码可能继承 copyleft 义务，企业用引发法律风险。**The Stack v2 的核心价值就是带详细 license metadata**——只留 MIT/Apache-2.0/BSD/MPL 等 permissive license。Codex / Copilot 当年的诉讼就源于这个问题。
- ❗ **test set contamination 是大问题**——MMLU、GSM8K、HumanEval 等 benchmark 的题目和答案大量出现在 web 上（教程、论坛、stack overflow），不做 contamination check 会让 benchmark 分虚高。**标准做法**：把 benchmark 的每道题转成 13-gram（或更长），在 pretrain corpus 里 substring match，命中的 doc 删掉。Llama-3、Qwen-2.5、DeepSeek-V3 都报告做过这一步——**有些早期模型 MMLU 高分有部分来自 contamination，对比新旧 model benchmark 时要警惕**。
- ❗ **流式 dedupe vs 全量 dedupe 的工程取舍**——全量 MinHash + LSH 跑 15T token 需要 PB 级中间存储（每个 doc 128 维 signature × 4 byte ≈ 0.5 KB，15T token 按 500 token/doc 算 = 30B doc = 15TB signature，再做 all-pair candidate 匹配会再放大）。实操是**按 snapshot 分块 + 跨块跨索引**，FineWeb 用 Spark/Slurm 集群分布式跑，单机 single-thread MinHash 在 1B token 以上就吃不消。

---

## 5. 经典 paper

- **Gao et al., 2020 — The Pile: An 800GB Dataset of Diverse Text for Language Modeling** — 早期"组合 corpus"范本。读 §2 数据集构成与 §5 各 source 的 quality 分析。Take-away：理解为什么预训不只用 web，而是 web + books + code + arXiv + StackExchange + … 多 source 组合，奠定后来所有 corpus 的范式。
- **Lee et al., 2021 — Deduplicating Training Data Makes Language Models Better** — 去重论文必读。读 §3 ExactSubstr / NearDup 算法 + §5 实证。Take-away：dedupe 是"free lunch"——同样 step 数训练 perplexity 显著降，且大幅减少 verbatim memorization（隐私 + license）。这是 SlimPajama / FineWeb / DCLM 的共同思想基础。
- **Penedo et al., 2024 — The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale** — 现代 SOTA 公开 corpus 论文。读全文（HuggingFace blog 也有可读版本），重点 §3 pipeline 五阶段细节、§5 FineWeb-Edu 的 LLM-as-judge 流程、§6 ablation。Take-away：现代 LLM 数据 pipeline 的事实参考实现，2024 年之后所有团队的对照基线。
- **Li et al., 2024 — DataComp-LM: In Search of the Next Generation of Training Sets for Language Models** — DCLM benchmark + model-based filter SOTA。读 §3 model-based filter 设计、§4 filter 的 MMLU 消融。Take-away：把"data 也是个可以被 benchmark 优化的对象"这件事系统化，证明 model-based filter 比 fastText 在 MMLU 上稳定高 2-4 点。
- **Soldaini et al., 2024 — Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research** — 完全 open（含 license metadata）。读 §3 pipeline、§4 toolkit。Take-away：理解"完全可复现 corpus"在学术研究中的价值，OLMo 用它训出可完全审计的 7B 模型。
- 加分阅读：**Rae et al., 2021 — Gopher** §A.1.1 的"MassiveText quality filter rules"——所有现代 pipeline 第一关都跑这套规则；**Xie et al., 2023 — DoReMi** —— domain mixing 的代表方法。

---

## 6. 自测与面试题

**Q1（pipeline）**：列出 LLM 预训数据 pipeline 的 5 个阶段，每阶段一句话说明做什么 + 关键工具或算法。

<details>
<summary>Answer sketch</summary>

1. **URL / 语言过滤**：用 UT1 blocklist 砍掉黑名单域名，用 fastText langid 留目标语言（FineWeb 砍掉约 75% Common Crawl）
2. **正文提取（boilerplate removal）**：从 Common Crawl WARC（不是 WET）用 **trafilatura** 提取正文，去掉 nav / footer / ads
3. **去重**：先 exact dedupe（hash），再 fuzzy dedupe（**MinHash + LSH**，n-gram → MinHash signature → LSH bucket → Jaccard verify > 0.8 视为 dupe），可加 substring dedupe；通常砍掉 50-80%
4. **质量过滤**：从粗到精四档——**Gopher 规则**（长度/词长/stopword 占比/字母占比）→ **fastText 分类器**（高质量 reference 训二分类）→ **model-based filter**（小 LM 算 perplexity，DCLM 范式）→ **LLM-as-judge**（大 LLM 打教育性分，FineWeb-Edu 范式）
5. **Domain mixing**：web / books / code / math / wiki / multilingual 按比例混合，配方可由 **DoReMi / RegMix** 学出，或抄 LLaMA/Qwen 公开配比

加分要点：

- 提到 FineWeb 公开统计：从一个 CC snapshot 到最终训练数据**只剩约 1-2%**——5 阶段都是大幅减量
- 提到 contamination check 是阶段 4 之后的额外一步——n-gram 匹配 benchmark 答案删掉
- 提到 5 阶段对应工具链：`trafilatura`（提取）+ `datasketch` / `text-dedup`（去重）+ `fasttext`（分类）+ `nemo-curator` / `datatrove`（端到端 framework）

</details>

**Q2（去重）**：MinHash + LSH 怎么找文档近似重复？写出关键 5 步 + 解释为什么能 unbiased estimate Jaccard。

<details>
<summary>Answer sketch</summary>

5 步：

1. **n-gram 化**：每个文档转成 n-gram set（典型 5-gram word level），$S = \{g_1, g_2, \dots\}$
2. **MinHash signature**：选 $K$ 个独立 hash 函数 $h_1, \dots, h_K$（$K=128$），signature 第 $k$ 维为 $\text{sig}_k = \min_{g \in S} h_k(g)$。结果 signature 是个 $K$ 维整数向量
3. **LSH 切 band**：把 signature 切成 $b$ 个 band（每个 band 长 $r$，$b \cdot r = K$），同 band 完全相同的文档进同一 bucket 当 candidate
4. **Candidate 收集**：所有同 bucket 文档对都是 candidate
5. **Jaccard verify**：对每个 candidate 对算精确 Jaccard 或 MinHash 估计 Jaccard，> threshold（如 0.8）判 dupe，丢一份

**为什么 unbiased**：MinHash 的关键性质——对随机 hash 函数 $h$，$P[\min_{g \in S_1} h(g) = \min_{g \in S_2} h(g)] = J(S_1, S_2) = |S_1 \cap S_2| / |S_1 \cup S_2|$。直觉：如果在 $S_1 \cup S_2$ 全部元素上算 hash，最小值要么来自 intersection（此时两边相等）、要么来自 symmetric difference（不等），前者概率正好是 Jaccard。$K$ 个独立 hash 函数取平均 → 标准差 $1/\sqrt{K}$，$K=128$ 时估计误差约 9%。

加分要点：

- $(b, r)$ 调拐点位置——同 bucket 的概率近似 $1 - (1 - s^r)^b$，是 S 形曲线
- 提到 **Lee 2021** 的核心结论：dedupe 是 "free lunch"——perplexity 降、memorization 减少
- 提到三种粒度：document / paragraph / substring
- 提到 SimHash 是另一种 fuzzy 去重方案但 LLM 数据领域 MinHash 更主流（recall 更好）

</details>

**Q3（quality vs quantity）**：DCLM-baseline 3.8T 和 FineWeb-Edu 1.3T 都比 RedPajama-v2 30T 训出来的同 size 模型在 MMLU 上高 5-8 个点。"少而精 > 多而杂"为什么成立？至少 2 个机制解释 + 这条结论的边界。

<details>
<summary>Answer sketch</summary>

**机制 1：参数容量被无效内容浪费**——LLM 是有限容量的"压缩器"，模型参数总量决定能记多少 pattern。喂 spam/SEO 农场/机翻劣质文本，等于让 transformer 把容量分给"广告关键词堆砌的 statistical pattern"，挤掉了真正有用的知识。同样 7B 参数，喂 quality 数据能记更多有用 pattern → MMLU 更强。

**机制 2：FLOPs 预算固定下的有效 token 比例**——总训练 FLOPs 固定（例如 1B 模型 26B FLOPs/token × 3T token），如果其中 70% token 是低 quality（重复/spam/无意义），等于实际只在 30% 有效 token 上训——且模型还得"学会忽略噪声"，反而消耗额外容量。质量过滤把 30T 砍到 3T 但都是有效 token，FLOPs 利用率高。

**机制 3：减少 noise 让 loss 信号更清晰**——低质量文本带来的 loss 噪声会让训练 trajectory 更乱、收敛更慢；clean 数据下同样 step 数能达到更低 loss、对应下游更强。

**机制 4（含 FineWeb-Edu 特例）**：高 quality 数据更接近"模型评测时面对的内容分布"——MMLU 题目本身是教育性 question，预训接触更多教育性内容自然提升 MMLU 应试能力（这是它有效的部分原因，也部分意味着评测被 dataset distribution 隐式优化）。

**边界条件**：

- pretrain 仍需 trillion 级 token——SFT 几万 sample 就够，但 pretrain 数据 < 100B 一般训不出可用 LLM。FineWeb-Edu 1.3T 是"少"——但仍是 trillion 级，不是几亿
- **同 source 类型的对比才成立**——不是说"1B 高质量 = 1T 低质量"，而是"过滤掉 70% 垃圾后剩下的 30% 训得更好"
- **quality classifier 引入 bias**——fastText quality 分类器训出来的"高分文档"偏 Wikipedia 风格，模型可能在 informal 文本下掉点；LLM-as-judge 的 bias 来自打分模型本身的偏好
- **过度依赖 quality filter 导致多样性下降**——SlimPajama 团队发现某些 source 用 0.7 threshold 比 0.85 训出来的 model 更强（多样性 > 极致清洁）
- **scaling regime 影响**：在 over-trained 区（参考 6.3 scaling law），quality 收益更大；在 under-trained 区，多一些 token 可能更划算

加分要点：

- DCLM 论文用 **DataComp-LM benchmark** 把"data 是个可以被优化的对象"系统化——和 ML 模型一样可以做消融、对比、leaderboard
- 提到 contamination：高 MMLU 分中可能有一部分来自 benchmark 答案在 pretrain 数据里
- 提到 6.3 scaling law 的 over-training 视角——固定 FLOPs 下选择 (model size, data size, data quality) 三元 trade-off

</details>

---

## 7. 延伸阅读

- [HuggingFace FineWeb blog post](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1) — FineWeb 的 pipeline 全公开，每一步都有 ablation，是现代数据清洗最完整的参考实现。
- [DataComp-LM (DCLM) GitHub](https://github.com/mlfoundations/dclm) — DCLM 的完整 benchmark 框架与 baseline 数据，附 model-based filter 训练脚本。
- [HuggingFace `datatrove`](https://github.com/huggingface/datatrove) — FineWeb 团队同款数据清洗 framework，包装了 trafilatura/MinHash/Gopher rules 的 pipeline DSL，单机到 Slurm 集群都能跑。
- [`text-dedup` GitHub](https://github.com/ChenghaoMou/text-dedup) — 工业级 MinHash + LSH dedupe 实现，支持 PySpark 分布式，FineWeb 用的就是它的变体。
- [Common Crawl 官方文档](https://commoncrawl.org/the-data) — 理解 WARC / WAT / WET 三种格式与如何下载，所有现代英文 pretrain corpus 的共同上游。
- [The Pile paper + corpus](https://pile.eleuther.ai/) — EleutherAI 的"组合 corpus"开山之作。
- [NVIDIA NeMo Curator](https://github.com/NVIDIA/NeMo-Curator) — NVIDIA 出的 GPU 加速数据清洗 framework，对超大规模团队（万亿 token+）更高效。
- 推荐继续读本教程的 **6.3 节《Scaling Law》**——本节回答"数据要怎么清"，6.3 回答"数据要多少 token、模型要多大才匹配"，两节合起来构成 pretrain 数据决策的完整闭环；以及 **8.1 节《SFT 数据构造》**——同根技术（去重、quality 分类）但完全不同 scale 与目标。
