---
title: "2.2 词向量：word2vec / GloVe / fastText"
description: "讲清 embedding 这件事是怎么开始的——从 one-hot 走到 distributed representation，再用 word2vec / GloVe / fastText 三个经典方法把 \"让相似词在向量空间靠近\" 的不同 motivation 摆出来——并用一段话告诉你：现代 LLM 的 token embedding 其实是同一件事的\"端到端版本\"，但本质从 static 变"
---

> ⏱ 预计阅读 25 分钟 ｜ 难度 ★ ｜ 前置：无

## 一句话本节讲什么

讲清 **embedding 这件事是怎么开始的**——从 one-hot 走到 distributed representation，再用 word2vec / GloVe / fastText 三个经典方法把 "让相似词在向量空间靠近" 的不同 motivation 摆出来——并用一段话告诉你：现代 LLM 的 token embedding 其实是同一件事的"端到端版本"，但本质从 **static** 变成了 **contextual**，这是 BERT/GPT 范式与 word2vec 范式的根本分界。

---

## 1. Mental model（直觉）

NLP 第一个绕不开的问题：**计算机怎么表示一个词**？

最朴素的方案是 **one-hot**——给词表 $\mathcal{V}$（大小 $V$，通常 5 万到 50 万）里第 $i$ 个词分配一个 $V$ 维向量，第 $i$ 维是 1 其余是 0：

```
"king":  [0, 0, 0, ..., 1, 0, ..., 0]   # 第 4217 维 = 1
"queen": [0, 0, 0, ..., 0, 0, ..., 1]   # 第 8841 维 = 1
"car":   [1, 0, 0, ..., 0, 0, ..., 0]
```

两个问题立刻暴露：

- **维度爆炸**：词表 5 万就要 5 万维稀疏向量，10 万就 10 万维。下游任何模型都要把它压扁。
- **语义全失**：任意两个不同词的 one-hot **内积恒为 0**——"king" 和 "queen" 的"距离"和 "king" 和 "car" 的"距离"完全一样。模型完全靠下游任务自己把语义"再学回来"。

**Distributed representation** 的 idea 来自 Hinton 1986：用一个 **低维稠密向量**（典型 100-300 维）表示一个词，让"语义"分散在每一维上——就像人脑的概念不是某一个神经元，而是一片激活模式。理想性质：

- "king" 和 "queen" 的余弦相似度高（同为皇室）
- "king" 和 "man"、"queen" 和 "woman" 的差向量近似平行（捕捉了 "性别" 这一 latent 维度）
- 由此涌现出经典演示：**`vec("king") - vec("man") + vec("woman") ≈ vec("queen")`**

这个 "向量加减能做语义类比" 的现象不是设计目标，是 **从大量文本上做无监督训练后自然涌现的**——这一点是 word2vec 在 2013 年震撼整个 NLP 的核心。

一图概括三种范式：

```
           one-hot                    word2vec / GloVe / fastText             LLM token embedding
        (V 维稀疏)                          (d 维稠密 static)                    (d 维稠密 contextual)
   king  [0,...,1,...,0]                king  [0.21, -0.4, ..., 0.07]      king  → hidden state 因上下文而异
   queen [0,...,0,...,1]                queen [0.19, -0.3, ..., 0.11]            "river bank" 的 bank
   bank  [...,1,...,0,0]                bank  [0.08,  0.6, ..., -0.2]            "Bank of America" 的 bank
                                                ↑                                       ↑
                                        每个词永远同一向量                       同一 token 不同上下文不同向量
                                          (无法区分一词多义)                          (BERT/GPT 范式核心)
```

剩下三节（word2vec / GloVe / fastText）回答的都是同一个问题：**怎么从一堆文本无监督地学出"中间这一列"**。最后一节回答："为什么现代 LLM 不再用左侧三个，而走到了右侧"。

---

## 2. 公式与原理

### 2.1 word2vec：CBOW 与 Skip-gram

word2vec 是 Mikolov 2013 提出的两个超轻量的"浅层网络"，本质就是 **一个 lookup 表 + 一个 softmax 分类器**。两种结构：

- **CBOW（Continuous Bag-of-Words）**：用窗口里的上下文词预测中心词。窗口大小 $c$，对中心词 $w_t$ 建模

$$P(w_t \mid w_{t-c}, \dots, w_{t-1}, w_{t+1}, \dots, w_{t+c})$$

- **Skip-gram**：反过来，用中心词预测窗口里每一个上下文词

$$\mathcal{L}_{\text{Skip-gram}} = -\frac{1}{T} \sum_{t=1}^{T} \sum_{\substack{-c \le j \le c \\ j \neq 0}} \log P(w_{t+j} \mid w_t)$$

CBOW 训得快但对低频词友好度差；Skip-gram 慢但对低频词更友好——**实践中 Skip-gram 更常用**（也是 gensim、原版 word2vec.c 的默认推荐）。

每个词在网络里有 **两个向量**：作为中心词时叫 **input vector** $v_w \in \mathbb{R}^d$，作为上下文时叫 **output vector** $v'_w \in \mathbb{R}^d$。最朴素的 softmax 形式：

$$P(w_o \mid w_c) = \frac{\exp(v'^\top_{w_o} v_{w_c})}{\sum_{w \in \mathcal{V}} \exp(v'^\top_w v_{w_c})}$$

分母要遍历整个词表 $V$（5 万到几十万）—— **每一步训练都做 $O(V)$ 计算根本扛不住**。这就引出 word2vec 论文最关键的工程贡献：把 softmax 近似掉。

#### Negative Sampling（NEG，最常用）

把 "在 V 上选正确的上下文词" 改成 "区分正样本对 vs $k$ 个采样负样本对"。对一个观测到的中心-上下文对 $(w_c, w_o)$，loss 写成

$$\mathcal{L}_{\text{NEG}} = -\log \sigma(v'^\top_{w_o} v_{w_c}) - \sum_{i=1}^{k} \mathbb{E}_{w_n \sim P_n(w)} \left[ \log \sigma(-v'^\top_{w_n} v_{w_c}) \right]$$

其中 $\sigma(x) = 1/(1+e^{-x})$，$P_n$ 是负采样分布。直觉就是 **K+1 路二分类**——把正样本对 push 到 1、$k$ 个采样的负样本对 push 到 0，每步只算 $O(k)$ 个内积。

负采样分布 $P_n$ 不是均匀的，也不是直接按词频，而是 **unigram 分布的 3/4 次方**：

$$P_n(w) \propto U(w)^{3/4}$$

这个 0.75 次方是经验拍出来的——比起均匀分布给低频词更多机会，比起原始 unigram 又对高频词降权。Mikolov 论文里 ablation 说 $0.75$ 显著好于 $0.5$ 和 $1.0$。

$k$（每个正样本配多少负样本）的经验值：**小数据集 5-20，大数据集 2-5**。

#### Hierarchical Softmax（备选）

把 V 个词组织成 Huffman 二叉树，每个词对应一条根到叶的路径长 $\log_2 V$，每一步是一个二分类。这样把 $O(V)$ 降到 $O(\log V)$。**实际工程里不如 NEG 流行**——NEG 实现更简单、超参更少、效果在大多数任务上不亚于 H-Softmax。

#### Subsampling of frequent words

像 "the" / "of" / "a" 这类高频功能词出现次数远多于内容词，但携带语义少。Mikolov 论文给了一个简单的概率丢弃公式：以概率

$$P(\text{drop } w_i) = 1 - \sqrt{\frac{t}{f(w_i)}}$$

把高频词从训练样本里随机丢掉（$t$ 典型 $10^{-5}$，$f$ 是词频）。这样既加快训练 2-10×，也间接给低频词更多曝光，最终 embedding 质量反而提升。

### 2.2 GloVe：从全局共现矩阵直接 factorize

word2vec 是 **local context、stochastic** 的——每次只看一个窗口。Pennington 2014 的 GloVe 反过来：先扫一遍全语料，统计 **共现矩阵** $X \in \mathbb{R}^{V \times V}$，$X_{ij}$ = 词 $j$ 出现在词 $i$ 上下文窗口里的总次数。然后做加权最小二乘 factorize：

$$\mathcal{L}_{\text{GloVe}} = \sum_{i,j=1}^{V} f(X_{ij}) \left( v_i^\top v_j + b_i + b_j - \log X_{ij} \right)^2$$

其中 $v_i, v_j \in \mathbb{R}^d$ 是要学的词向量、$b_i, b_j \in \mathbb{R}$ 是偏置；$f$ 是 weighting function 防止低频共现项把 loss 主导：

$$f(x) = \begin{cases} (x / x_{\max})^\alpha & x < x_{\max} \\ 1 & x \ge x_{\max} \end{cases}$$

典型 $x_{\max}=100, \alpha = 3/4$（同样是这个 0.75 次方，背后逻辑类似 NEG）。

**与 word2vec 的对比**：

| 维度 | word2vec | GloVe |
|---|---|---|
| 信息源 | 局部窗口、stochastic | 全局共现矩阵、deterministic |
| 训练步骤 | 一遍遍扫语料、随机 SGD | 先扫一遍生成 $X$，再做矩阵 factorize |
| 对罕见词 | 较友好（Skip-gram） | 靠 $f(X_{ij})$ 显式 down-weight |
| 向量类比表现 | 强 | 同样强（Pennington 称略优） |
| 工程友好度 | 流式、内存友好 | 需要先存 $X$（$V \times V$ 稀疏） |

GloVe 在 2014-2017 间是学术 baseline 的常客，工业上 word2vec 实现更多。**两者得到的 embedding 在下游任务上差距很小**（多数 paper 报告 1-2 个点之内），所以选哪个更多看工程便利。

### 2.3 fastText：把单词进一步拆成 character n-gram

word2vec / GloVe 都是 **word-level**：每个 word 一个独立 embedding。问题：

- 词表外（OOV）词没有 embedding（推理时遇到 "covid" / 拼写错的词只能 unk）
- 形态丰富的语言（俄语、芬兰语、土耳其语、阿拉伯语）一个词根有上百种屈折形态，每种形态都要一个独立向量浪费数据
- 同一个词根的形态变体（"play" / "playing" / "played" / "player"）的 embedding 完全独立，模型必须从数据里"重新学一遍它们其实相关"

Bojanowski 2017 的 fastText 把每个词进一步拆成 **character n-gram**（典型 $n \in \{3, 4, 5, 6\}$），词的 embedding 是它所有 n-gram embedding 的 sum：

$$v_w = \sum_{g \in \mathcal{G}_w} v_g$$

其中 $\mathcal{G}_w$ 是词 $w$ 的所有 character n-gram 集合（含特殊边界符 `<` `>`，避免 "her" 词与 "where" 里的 "her" 子串混淆）。例如 `where` 在 $n=3$ 时拆为 `<wh, whe, her, ere, re>`，加上整词 `<where>` 本身。

由此带来的好处：

- **OOV 处理**：从未见过的词只要 character n-gram 见过，就能合成出一个 embedding
- **形态共享**：`playing` / `played` 共享多个 n-gram，自动捕捉词根
- **训练目标本身仍是 Skip-gram + NEG**——fastText 只是把 input vector 的来源从"一个 word lookup"换成"多个 n-gram lookup 求和"

这个 **subword 思想** 与下一章 3.1 BPE 是同源的——只不过 fastText 是固定的 character n-gram，BPE 是统计驱动的合并算法。从 fastText 到 BPE 这条路通向了现代 LLM tokenizer。

### 2.4 现代 LLM 的 token embedding：从 static 到 contextual

到这里要说一件 contradicting 的事：**现代 LLM 不用预训练的 word2vec / GloVe / fastText 作为输入 embedding**。取而代之的是这样几行：

```python
import torch.nn as nn
embed = nn.Embedding(50000, 768)  # vocab=50k, hidden=768
# embed.weight 形状 (50000, 768)，就是 LLM 的 token embedding
# 训练开始时随机初始化，训练结束时与整个模型一起 end-to-end 学完
```

几个关键差异：

- **不预训练，直接 end-to-end 学**：LLM 预训练目标（next-token prediction）本身就在学一个比 word2vec **更优** 的 embedding——把预训练 word2vec 塞进去当初始化反而比随机初始化差（见踩坑第 5 条），因为 word2vec 的"几何"和 LLM 内部所需的"几何"不一样。
- **token 不是 word**：LLM 走 BPE / WordPiece 等 subword tokenizer（详见 3.1），词表里的 "token" 多数是 subword 片段（`▁play`、`ing`、`tion`、`，` 等），OOV 几乎不存在。
- **Weight tying（embedding-projection 共享）**：LLM 输出端把 hidden state 投到 vocab 上算 softmax，这个 projection 矩阵 $W_{\text{out}} \in \mathbb{R}^{V \times d}$ 与 input embedding $W_{\text{in}}$ **共享同一组参数**（Press & Wolf 2017）。一是省一半 embedding 参数（GPT-2 vocab=5w、d=1600 时省 8000 万），二是 input/output 表示在同一空间，理论与经验都更好。
- **本质区别：static → contextual**：
  - **static**（word2vec / GloVe / fastText）：每个词永远一个固定向量，"bank" 在 "river bank" 和 "Bank of America" 里得到 **完全相同** 的向量，下游任务必须自己去从上下文区分一词多义。
  - **contextual**（LLM 的 hidden state）：`nn.Embedding` 给出的是初始向量，但经过几十层 self-attention + FFN 之后，第 $L$ 层第 $t$ 个位置的 hidden state $h_t^{(L)} \in \mathbb{R}^d$ 是 "token + 全部上下文" 的函数——同一 token 在不同句子里得到 **不同** 的表示，自动解决一词多义。

contextual embedding 这条路最早由 **ELMo**（Peters 2018）走通——用双向 LSTM 在大语料上预训练，下游任务把 LSTM 的 hidden state 当 embedding 用，而不是查一个固定表。CoVe（McCann 2017）也属于同期。**ELMo 是 BERT/GPT 范式的直接前身**——只不过 BERT 把 LSTM 换成 Transformer，把"先训 embedding 再喂下游"改成"整个模型端到端 fine-tune"，效果一举把 NLP 推到新台阶。本节不展开 ELMo / BERT 内部，留给 Module 4-5。

时间线压缩：

```
1986  Hinton — distributed representation 概念
2003  Bengio NNLM — 神经网络 LM 把 embedding 作为副产品学出来
2013  Mikolov word2vec — 把 embedding 训练独立、轻量、可规模化
2014  Pennington GloVe — 全局共现 factorize
2017  Bojanowski fastText — 引入 subword
2018  Peters ELMo — contextual embedding 起点（BiLSTM）
2018+ BERT / GPT — Transformer 取代 LSTM，端到端 contextual 范式确立
现代  LLM 的 nn.Embedding + 几十层 self-attention，把 word2vec 时代的所有问题端到端解决
2024+ 现代 embedding 模型（bge / E5 / NV-Embed，详见 Module 16.4）反过来用 LLM 当 encoder + 对比学习训 sentence/passage embedding
```

word2vec 没死——它只是 **"被吸收到 LLM 的第一层和最后一层里" + "在 sentence/passage 级别被对比学习重新发明了一遍"**。

---

## 3. 最小代码示例

### 3.1 手撕 Skip-gram + Negative Sampling forward

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SkipGramNEG(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.in_emb = nn.Embedding(vocab_size, dim)   # 中心词向量 v_c
        self.out_emb = nn.Embedding(vocab_size, dim)  # 上下文词向量 v'_o

    def forward(self, center, pos_ctx, neg_ctx):
        """
        center : (B,)              中心词 id
        pos_ctx: (B,)              正样本上下文词 id
        neg_ctx: (B, k)            采样的 k 个负样本上下文词 id
        """
        v_c = self.in_emb(center)              # (B, d)
        v_pos = self.out_emb(pos_ctx)          # (B, d)
        v_neg = self.out_emb(neg_ctx)          # (B, k, d)

        # 正样本：log σ(v'_o · v_c)
        pos_score = (v_c * v_pos).sum(dim=-1)              # (B,)
        pos_loss = F.logsigmoid(pos_score)                 # (B,)

        # 负样本：log σ(-v'_n · v_c)
        neg_score = torch.bmm(v_neg, v_c.unsqueeze(-1)).squeeze(-1)  # (B, k)
        neg_loss = F.logsigmoid(-neg_score).sum(dim=-1)              # (B,)

        return -(pos_loss + neg_loss).mean()  # NEG loss
```

关键 5 行就是 NEG loss 的实现——`pos_score` 是正样本对内积，`neg_score` 是 $k$ 个负样本对内积，`-logsigmoid(pos) - sum(logsigmoid(-neg))` 就是公式 §2.1 的 NEG。两套 embedding（`in_emb` / `out_emb`）一直保持独立，训练完后 **通常只用 `in_emb` 作为最终词向量**——也有人取两者平均，差距微小。

### 3.2 用 gensim 加载预训练 word2vec / GloVe 演示类比

```python
# pip install gensim
from gensim.models import KeyedVectors
import gensim.downloader as api

# 加载 Google News 预训练 word2vec（300 维，约 1.6 GB，首次自动下载）
wv = api.load("word2vec-google-news-300")

# 1. 最相似词
print(wv.most_similar("king", topn=5))
# [('kings', 0.71), ('queen', 0.65), ('monarch', 0.64), ...]

# 2. 经典类比：king - man + woman ≈ ?
print(wv.most_similar(positive=["king", "woman"], negative=["man"], topn=3))
# [('queen', 0.71), ('monarch', 0.62), ('princess', 0.59)]

# 3. 余弦相似度
print(wv.similarity("paris", "france"))   # 0.66
print(wv.similarity("paris", "banana"))   # 0.06

# GloVe 用法完全一致：api.load("glove-wiki-gigaword-300")
```

这就是 word2vec / GloVe 最经典的演示——`king - man + woman ≈ queen` 不是手工设计的，是从 Google News 上无监督训练出来后自然涌现的几何性质。

### 3.3 LLM 的 token embedding 长什么样

```python
import torch.nn as nn
embed = nn.Embedding(50000, 768)  # GPT-2 small 量级：vocab=50k, d=768，embed.weight 就是它的 token embedding
```

---

## 4. 工程踩坑与经验

- ❗ **window size**：word2vec 的 window 越小越偏 **句法**（小 window = 更近距离，捕捉局部语法关系），越大越偏 **语义**（大 window = 主题相关性）。**典型默认 5（句法/通用）vs 10（语义/topic 偏向）**，下游任务必须 sweep。CBOW 默认 5、Skip-gram 默认 10 是 gensim 与原版 word2vec.c 的工业惯例。
- ❗ **Negative samples $k$ 的取值**：Mikolov 原文经验—— **小数据集（百万词级）取 $k=5\sim20$，大数据集（亿词以上）取 $k=2\sim5$** 即可。$k$ 越大越准但越慢。盲目取大 $k$ 在大语料上既慢又收益小，是新手常踩的浪费。
- ❗ **中文 word2vec 的切词器选择决定一切**：jieba / pkuseg / spaCy 中文 / THULAC / HanLP 切出的"词"边界不同——同样一句"人工智能产业发展"，jieba 切成 `人工智能 / 产业 / 发展`，pkuseg 可能切成 `人工 / 智能 / 产业 / 发展`，结果是**两套切词器训出的 embedding 完全不能混用**（连词表都对不齐）。**现代 LLM 走 subword（BPE）完全绕开了这个问题**——这也是 NLP 历史上中文与英文最大的工程鸿沟之一。
- ❗ **传统 word2vec 不能区分一词多义**：`bank`（银行）和 `bank`（河岸）共享同一个向量，无论上下文如何。fastText 把词拆成 n-gram 也没解决这个问题（仍是 word level）。**真正解决一词多义的是 contextual embedding**（ELMo / BERT / LLM hidden state），同一 token 在不同上下文得到不同表示。如果你的下游任务（如 WSD、NER）严重依赖一词多义，请直接上预训练 LLM，不要套 word2vec。
- ❗ **把 word2vec 当 LLM 输入 embedding 会比 random init 还差**——这是新手常犯的 #1 错误。LLM 预训练目标（next-token CE loss）本身就在学一个比 word2vec 几何上更优的 embedding（contextual、与 hidden space 对齐、和 output projection tied），强行用 word2vec 做初始化反而把模型限制在了一个次优的"word-level static 几何"里，下游 loss 收敛更慢、终点更高。**LLM 的 embedding 该让模型自己端到端学**——这件事不要省。
- ❗ **subsampling threshold $t$ 的取值**：典型 $t=10^{-5}$，太小会丢太多高频词（包括有意义的实词）训练数据反而下降，太大不起作用。如果你的语料里 stop words 已经预清洗过，subsampling 收益就小很多。
- ❗ **GloVe 的共现矩阵内存爆炸**：$V=10$ 万时稠密 $X$ 是 $10^{10}$ 个 float，必须用稀疏存储（CSR / hash table）。Pennington 原版 C 实现用 OpenMP + 自定义 hash，工业上用 GloVe 训自己语料的人极少（多数直接下载 Stanford 发布的预训练）。

---

## 5. 经典 paper

- **Mikolov et al., 2013 — Distributed Representations of Words and Phrases and their Compositionality** — word2vec 的原典（Skip-gram + Negative Sampling + Subsampling 这套范式确立）。读 §2-3 即可，§4 是短语 embedding 扩展。注意还有姊妹篇 *Efficient Estimation of Word Representations in Vector Space*（更早，提出 CBOW/Skip-gram 架构），两篇要一起看才完整。Take-away：**为什么 NEG 比 hierarchical softmax 流行**、**为什么 subsampling + 0.75 次方负采样**。
- **Pennington, Socher & Manning, 2014 — GloVe: Global Vectors for Word Representation** — 全局共现 factorize 路线代表作。读 §3 公式推导（从 ratio of co-occurrence 推到 weighted least squares 那一段非常漂亮）。Take-away：local context（word2vec）vs global statistics（GloVe）是同一目标的两种实现，理解二者差异比记公式重要。
- **Bojanowski et al., 2017 — Enriching Word Vectors with Subword Information** — fastText 原典。读 §3 把 character n-gram 引入 Skip-gram 的 trick，这是 subword 思想第一次清晰落地，**与下一章 3.1 BPE 同根**。Take-away：处理 OOV、形态丰富语言、和"为什么 LLM 走 subword tokenizer"。
- **Peters et al., 2018 — Deep Contextualized Word Representations (ELMo)** — contextual embedding 的范式起点。本节不展开（属于 Module 4-5），但要知道这篇是 word2vec 时代到 BERT 时代的"过桥论文"——读 abstract + introduction 理解 "为什么从 static 走向 contextual" 就够。

---

## 6. 自测与面试题

**Q1（概念）**：one-hot encoding 和 distributed representation 的本质区别在哪？为什么后者支持 "king - man + woman ≈ queen" 这种向量算术？

<details>
<summary>Answer sketch</summary>

要点：

- **维度与稀疏性**：one-hot 是 $V$ 维稀疏（只有 1 维是 1，其余全 0）；distributed 是 $d$ 维稠密（典型 100-300）每一维都是实数。
- **几何性质**：one-hot 任意两个不同词的内积 = 0，余弦相似度恒为 0——**几何上完全不携带语义信息**。distributed 的相似词向量靠近（cos 相似度高），不同 latent 维度可对应不同语义/语法属性。
- **类比为什么涌现**：训练目标（Skip-gram 的"用中心词预测上下文"）让出现在相似上下文里的词得到相似的向量。而 "king" 与 "queen" 共享 royal 上下文（court / throne / kingdom），"man" 与 "woman" 共享 gender 上下文（he/she、father/mother）—— **"性别" 这一 latent 维度在向量空间被自动表示成一条近似平行的方向**，所以 `king - man` 抹去 male 信息再 `+ woman` 加上 female 信息就指向 queen 附近。
- 加分：这种类比能力在 word2vec 之前完全没有，是 distributed representation + 大语料无监督训练的涌现性质，2013 年震撼整个 NLP 领域。

</details>

**Q2（数学/代码）**：写出 Skip-gram + Negative Sampling 的 loss 公式，并解释为什么这样写比 full softmax 快多少。

<details>
<summary>Answer sketch</summary>

公式：

$$\mathcal{L}_{\text{NEG}} = -\log \sigma(v'^\top_{w_o} v_{w_c}) - \sum_{i=1}^{k} \log \sigma(-v'^\top_{w_{n_i}} v_{w_c})$$

其中 $\sigma$ 是 sigmoid，$w_c$ 是中心词、$w_o$ 是观测到的正样本上下文、$w_{n_i}$ 是从 $P_n(w) \propto U(w)^{3/4}$ 采样的负样本。

为什么快：

- **Full softmax**：每一步要算 $\sum_{w \in \mathcal{V}} \exp(v'^\top_w v_{w_c})$，$O(V \cdot d)$ 计算，$V=5\text{w}$ 时每步 $\sim 5\text{w} \cdot 300 = 1.5 \times 10^7$ 次乘加。
- **NEG**：每步只算 1 个正样本对 + $k$ 个负样本对的内积，$O((k+1) \cdot d)$，$k=10, d=300$ 时每步 $\sim 3300$ 次乘加—— **快约 $V/(k+1) \approx 5000$ 倍**。
- 本质：把 V 路 softmax 近似成 K+1 路二分类（正 vs 负），用采样近似 normalize 项的和。
- 加分：NEG 不是 NCE 的精确版本（去掉了 normalize 常数），但实际 word embedding 任务下与 NCE 效果相当；hierarchical softmax 是另一种 $O(\log V)$ 加速方案，但工程实现复杂度高于 NEG，主流仍是 NEG。
- 加分：负采样分布 $\propto U(w)^{0.75}$ 是 Mikolov 经验拍出来的，相比均匀分布给低频词更多机会、相比 unigram 又对高频词降权。

</details>

**Q3（trade-off / 延伸）**：现代 LLM 不再使用预训练 word2vec / GloVe / fastText 作为 input embedding，至少给出 2 条原因。

<details>
<summary>Answer sketch</summary>

至少答出 2 条；好的回答覆盖 3-4 条：

- **end-to-end 学的几何更适配模型**：LLM 的 next-token CE loss 本身就在学一个 contextual、与下游 hidden space 对齐的 embedding；word2vec 几何（基于"在窗口内共现的词靠近"）对 LLM 的 transformer hidden space 不是最优——实测 word2vec 初始化反而比 random init 收敛慢、终点高（见踩坑第 5 条）。
- **static vs contextual 的根本差异**：word2vec 给一个词永远同一向量，无法区分 "bank" 银行 vs 河岸；LLM 的 hidden state 是 token + 全部上下文的函数，自动解决一词多义。LLM 的 nn.Embedding 只是一个起点，contextualization 由后续几十层 self-attention 完成。
- **token 不是 word**：LLM 走 BPE / WordPiece / Unigram subword tokenization，词表里多数是 subword 片段（`▁play`、`ing`、`tion`），与 word-level 的 word2vec 词表对不上；OOV 几乎不存在，也就不需要 fastText 的 subword n-gram 求和了。
- **Weight tying 节省参数**：LLM 把 input embedding 与 output projection 共享同一组参数（Press & Wolf 2017），既省一半 embedding 参数也让 input/output 在同一空间——这件事跟"用预训练 word2vec 初始化"不兼容（output projection 在哪？）。
- **多语言/多模态扩展**：LLM 一个词表覆盖所有语言（甚至代码），多语言 word2vec 要么需要对齐，要么需要分语种独立训。
- 加分：现代 embedding 模型（bge / E5 / NV-Embed，Module 16.4）做了一次"反向回归"——用预训练 LLM 当 encoder + 对比学习训 sentence/passage embedding，比 word-level word2vec 强 1-2 个数量级，但 motivation 与 InfoNCE（1.4 节）一脉相承。

</details>

---

## 7. 延伸阅读

- [Chris McCormick — Word2Vec Tutorial Part 1 / Part 2 (Negative Sampling)](https://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/) — Skip-gram + NEG 最清晰的可视化讲解，配大量直觉图，本节数学的最佳辅助。
- [Stanford CS224N — Lecture 1 & 2 on Word Vectors](https://web.stanford.edu/class/cs224n/) — Manning 亲自讲 word2vec / GloVe / 类比涌现的机制，slide + video 都开放，30 分钟看完比读 paper 快。
- [gensim word2vec API doc](https://radimrehurek.com/gensim/models/word2vec.html) — 工业最常用的 word2vec 实现，要训自己语料的 embedding 直接用它；`gensim.downloader` 可一行加载所有公开的 pretrained 词向量。
- [Sebastian Ruder — On word embeddings (3 篇 blog 系列)](https://www.ruder.io/word-embeddings-1/) — word2vec / GloVe / fastText / contextual 的历史脉络，讲 motivation 比 paper 清晰。
- 推荐继续读本教程的 **2.3 节《RNN / LSTM / GRU》**——把"序列建模"补齐，为 2.4 Bahdanau attention 和 4 章 Transformer 铺路。
