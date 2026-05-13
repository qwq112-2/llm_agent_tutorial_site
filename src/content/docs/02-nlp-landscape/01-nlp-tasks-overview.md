---
title: "2.1 NLP 任务全景：分类 / 序列标注 / Seq2Seq / QA"
description: "给读者一张\"问题地图\"——LLM 时代之前 NLP 在解哪 5 大类任务、它们的输入 / 输出 shape 与评测指标各是什么、以及为什么 GPT-3 之后这 5 类都被 prompt-based generation 统一成了同一个范式。这张地图建立后，后续每个新模型都能放回到\"它解决了哪类问题\"的格子里。"
---

> ⏱ 预计阅读 25 分钟 ｜ 难度 ★ ｜ 前置：无（Module 2 入口节）

## 一句话本节讲什么

给读者一张"问题地图"——LLM 时代之前 NLP 在解哪 5 大类任务、它们的输入 / 输出 shape 与评测指标各是什么、以及为什么 GPT-3 之后这 5 类都被 prompt-based generation 统一成了同一个范式。**这张地图建立后，后续每个新模型都能放回到"它解决了哪类问题"的格子里**。

---

## 1. Mental model（直觉）

NLP 不是一个任务，是一族任务。把它们按 **"输入是什么 shape、输出是什么 shape"** 这两个轴排开，就能看出底层只有 5 种基本形态——所有现代 NLP / LLM benchmark 都是它们的组合。

```
                  输出 shape →
                  ┌─────────┬─────────┬─────────┬─────────┐
                  │ 1 label │ N label │  span   │  text   │
                  │ (sent)  │ (token) │ (a,b)   │ (free)  │
   ┌──────────────┼─────────┼─────────┼─────────┼─────────┤
 输│ 1 sentence   │  分类    │ 序列标注 │    -    │ 单文本  │
 入│              │ (IMDB)  │ (NER)   │         │ 摘要    │
 s ├──────────────┼─────────┼─────────┼─────────┼─────────┤
 h │ 2 sentences  │ NLI/STS │    -    │ 抽取 QA │ 生成 QA │
 a │              │ (MNLI)  │         │ (SQuAD) │ (ELI5)  │
 p ├──────────────┼─────────┼─────────┼─────────┼─────────┤
 e │ src 序列 →   │    -    │    -    │    -    │ Seq2Seq │
 ↓ │ tgt 序列     │         │         │         │ (WMT)   │
   └──────────────┴─────────┴─────────┴─────────┴─────────┘
```

**BERT 时代（2018-2020）的做法是**：每个 cell 都接一个 task-specific head——分类用 `[CLS] → Linear → softmax`、序列标注用 `每个 token → Linear → BIO label`、抽取 QA 用 `每个 token → 2 个 Linear 预测 start/end`、Seq2Seq 用 encoder-decoder。**5 类任务就是 5 套训练 / 推理 pipeline**。

**GPT-3 之后（2020+）的做法是**：把所有 cell 都塞进同一个 cell——`text → text`。分类变成"判断这条评论是正面还是负面：xxx → 答案：正面"；NER 变成"提取这段话中的人名：xxx → 答案：[张三, 李四]"；NLI 变成"判断 s1 是否蕴含 s2"。任务的"格式"从 model architecture 退化为 prompt template，**一个模型搞定所有 cell**。

理解这张表是理解 Module 4 之后所有内容的前提：每读到一个新模型 / benchmark / 训练目标，先问自己——它在哪个 cell 里？解决的是哪一类输入 / 输出 shape？

---

## 2. 公式与原理

NLP 任务的"原理"不是公式，而是 **(input, output, loss) 三元组**。下面把 5 大任务族按这个结构系统过一遍，再讲范式统一。

### 2.1 文本分类（Text Classification）

- **输入 / 输出**：单条文本 $x \in \mathcal{V}^*$（$\mathcal{V}$ 是词表），输出标签 $y \in \{1, \dots, K\}$。
- **代表数据集**：IMDB / SST-2（情感二分类）、AG News（主题 4 分类）、Jigsaw Toxic（多标签有害内容检测）。
- **典型 head**：encoder 输出取 `[CLS]` 或 mean pooling 得到 $h \in \mathbb{R}^d$，过 $W \in \mathbb{R}^{d \times K}$，softmax + CE loss（即 1.4 节讲的 $-\log q_y$）。
- **评测**：accuracy（类平衡时）/ F1（类不平衡时，macro vs micro 要分清）/ AUC（二分类按 score 排序的鲁棒指标）。
- **难点**：(1) **长文本截断**——BERT 默认 512 token，长文档要么截尾要么 sliding window 取均值；(2) **类别不平衡**——正例 1% 时 accuracy 99% 全猜负也能拿到，必须看 minority-class F1；(3) **多标签 vs 多分类**——多标签用 sigmoid + per-class BCE，多分类用 softmax + CE，新手经常搞混。

### 2.2 序列标注（Sequence Labeling / Token Classification）

- **输入 / 输出**：序列 $x = (x_1, \dots, x_n)$，输出 per-token 标签 $y = (y_1, \dots, y_n)$，每个 $y_i \in \{1, \dots, K\}$。
- **代表数据集**：CoNLL-2003（英文 NER：PER/ORG/LOC/MISC）、OntoNotes 5.0（更细粒度 NER 18 类）、Universal Dependencies（POS 标注）、CoNLL-2000（chunking）。
- **典型 head**：每个 token 的 hidden state $h_i \in \mathbb{R}^d$ 过共享 $W \in \mathbb{R}^{d \times K}$，per-token softmax + CE。早期还会接 CRF 层做全局解码，BERT 时代之后大多直接 argmax 也够用。
- **BIO / BIOES schema**（必须懂）：原始 NER 标签只有"实体类型"，但实体可能跨多个 token（如"小红 / 书"），需要标注每个 token 在实体中的位置：

  - **BIO** 三标：`B-` (Begin) / `I-` (Inside) / `O` (Outside)。如"小红书 总部 在 上海" → `B-ORG I-ORG I-ORG O O B-LOC`。
  - **BIOES** 五标：增加 `E-` (End) / `S-` (Single)。同例 → `B-ORG I-ORG E-ORG O O S-LOC`。
  - **为什么用 BIOES**：BIO 在两个相邻同类实体（如"张三 李四"两个人名连写）边界处会被合并成一个实体；BIOES 显式标 End / Single 解决了这个边界歧义，工业 NER 默认推荐。

- **评测**：**entity-level F1**，不是 token-level F1。一个实体所有 token 都标对才算对——只标对 "B-PER" 但漏了 "I-PER" 算预测错一个完整实体。`seqeval` 是事实标准库。
- **难点**：嵌套实体（"清华大学计算机系"既是 ORG 又包含子 ORG）、不连续实体（医疗领域常见）、跨句指代——这些是 NER paper 持续刷的 SOTA 战场。

### 2.3 Seq2Seq（生成）

- **输入 / 输出**：source 序列 $x = (x_1, \dots, x_n)$，target 序列 $y = (y_1, \dots, y_m)$，长度 $m$ 与 $n$ 不必相等。
- **代表数据集**：WMT（机器翻译，2014-2024 年度大赛）、CNN/DailyMail（抽取式新闻摘要 ~50 词）、XSum（极短摘要 ~25 词，更难）、改写 / 风格迁移 / 数据到文本（Data-to-Text）。
- **典型 head**：encoder-decoder（早期 RNN，后来 Transformer 原版）或 decoder-only autoregressive，per-token CE loss：

  $$\mathcal{L} = -\sum_{t=1}^{m} \log P_\theta(y_t \mid x, y_{<t})$$

- **评测**：BLEU（n-gram precision，机器翻译标配）/ ROUGE-1/2/L（n-gram recall，摘要标配）/ METEOR（带同义词召回）/ chrF（character-level F-score，多语言友好）。LLM 时代这些 n-gram 指标越来越被 BLEURT、COMET、GPT-4-as-judge 取代——**因为 LLM 输出的合理改写常常 BLEU 低但人眼看更好**。
- **难点**：
  - **曝光偏差（exposure bias）**：训练时 decoder 输入是 ground-truth $y_{<t}$（teacher forcing），推理时输入是模型自己生成的 $\hat{y}_{<t}$，分布不一致——错误会沿着 timestep 累积放大。
  - **解码策略**：贪心 / beam search（翻译用，beam_size=4-5）/ sampling（top-k / top-p / temperature，对话生成用）。每种都有 trade-off：beam 倾向短而保守、sampling 倾向多样但不稳。
  - **长度偏置**：beam search 不加 length penalty 会偏好短句，translate output 莫名其妙缺一半。

### 2.4 问答 QA

QA 拆两类，差异**比想象中大**：

**(a) 抽取式 QA（Extractive QA）**

- **输入 / 输出**：(question $q$, context $c$) → answer 是 $c$ 中的一段连续 span $(a_{\text{start}}, a_{\text{end}})$。
- **代表数据集**：SQuAD 1.1（answer 必在 context 中）、SQuAD 2.0（增加"无答案"判断）、Natural Questions（开放域，context 是整个 Wikipedia）。
- **典型 head**：把 $(q, c)$ 拼成一个序列输入 encoder，每个 token 接 2 个 Linear 头分别预测它是 start / end 的概率，取 $\arg\max_{a, b: a \le b}$ 联合分数。
- **评测**：**EM (Exact Match)** + **token-level F1**（注意是 token 级，不是字符级，也不是字面相等：预测 "Albert Einstein" 与 ground-truth "Einstein, Albert" 算 F1≈1.0）。

**(b) 生成式 QA（Generative / Abstractive QA）**

- **输入 / 输出**：(question $q$, [可选 context $c$]) → answer 是自由文本，不必出现在任何输入里。
- **代表数据集**：ELI5（"像我 5 岁一样解释"长答案）、TriviaQA（开放域知识问答）、NarrativeQA（基于故事书的推理问答）。
- **典型 head**：seq2seq / decoder-only autoregressive 生成，与 §2.3 共享框架。
- **评测**：ROUGE-L（与摘要类似），但越来越多用 **LLM-as-judge**（GPT-4 评分 1-5）或 **fact-checking**（用 retrieval 比对答案与 ground-truth knowledge）——因为同一个事实有无数种表达方式，n-gram 指标抓不住。
- **抽取 vs 生成的工程后果**：抽取式输出可控（一定来自 context，不会幻觉），但表达不灵活；生成式表达自然，但幻觉无法 100% 排除——这就是 RAG 体系（Module 13.2）想吃两头甜的源头。

### 2.5 NLI / 文本相似度

- **输入 / 输出**：(sentence 1 $s_1$, sentence 2 $s_2$) → label 或 score。
- **NLI（Natural Language Inference）**：3 分类 entailment / contradiction / neutral。代表数据集 SNLI（图像 caption 衍生）、MNLI（多领域）、RTE（GLUE 子集）、ANLI（对抗版本）。
- **STS（Semantic Textual Similarity）**：连续相似度回归，0-5 分。代表数据集 STS-B（GLUE 子集）。
- **典型 head**：BERT 把 $s_1$ 与 $s_2$ 拼成 `[CLS] s1 [SEP] s2 [SEP]`，`[CLS]` 过 MLP——分类用 softmax+CE，回归用 MSE。Sentence-BERT（Reimers 2019）改成 dual encoder + cosine 相似度，是后来 embedding model 的雏形。
- **评测**：NLI 用 accuracy（类基本平衡）；STS-B 用 Pearson / Spearman 相关系数（衡量预测分数与人工分数的单调一致性）。
- **难点**：详见 §4 的 annotation artifact 踩坑——这类任务的 spurious correlation 是 NLP 数据质量研究的经典案例。

### 2.6 LLM 时代的范式统一

T5（Raffel 2020）提出 **"all tasks are text-to-text"**：任意任务都用一个固定格式的 input prompt + 自由文本 output 表示。GPT-3（Brown 2020）更进一步——**连 fine-tune 都不需要**，把 task 描述写进 prompt 用 in-context learning 就行。

举几个范式平移的例子：

| 任务 | BERT 时代 | LLM 时代 prompt |
|---|---|---|
| 情感分类 | `[CLS] → Linear(2) → softmax` | "判断这条评论的情感（正面 / 负面）：xxx → 答案：" |
| NER | per-token BIO head | "提取这段话中的人名地名：xxx → 答案：{persons: [...], locations: [...]}" |
| NLI | `[CLS] s1 [SEP] s2 → Linear(3)` | "判断前提是否蕴含假设。前提：s1 假设：s2 → 答案（entailment / contradiction / neutral）：" |
| 抽取 QA | start/end head on context | "根据 context 回答 question。Context: c Question: q → 答案：" |
| 翻译 | encoder-decoder | "把以下中文翻译成英文：xxx → 答案：" |

**优点**：multi-task 一套 model、零样本可迁移、天然支持新任务（只要能写出 prompt）。这也是 LLM 评测从"per-task fine-tune accuracy"转向"prompt-based zero/few-shot"的根本原因（详见 Module 12.1）。

**代价**：**输出格式不可控**——模型可能回答"我觉得是正面，因为..."而不是只输出"正面"。工程上要么靠 constrained decoding（Outlines / XGrammar，Module 13.4 详讲）、要么靠 JSON schema + 后处理 / 重试。这是 prompt-based NLP 与 head-based NLP 最大的工程差异。

### 2.7 任务族与 LLM 训练阶段的对应

| 训练阶段 | 涉及哪类任务 | 数据形态 |
|---|---|---|
| **预训练（CLM / MLM）** | 不直接对应任何一类，是 next-token 自监督 | 海量裸文本（FineWeb / Dolma），无标签 |
| **SFT** | 上述全部 5 类 + 多轮对话 + 工具调用 | (input, output) 监督对，混合所有任务 |
| **RLHF / DPO / GRPO** | 与具体任务正交，是偏好对齐 | (chosen, rejected) 偏好对 |
| **Agent / Tool Use** | QA + Seq2Seq + 工具结果反馈的组合 | trajectory：(user → tool call → observation → answer) |

**经典 NLP pipeline 时代的痕迹**：2010 年代 NLP 系统是 tokenization → POS tagging → parsing → NER → SRL → coreference → ... 串起来的"管线"，每一环一个独立模型。预训练 + 端到端 fine-tune 把这条管线碾平。但有趣的是，到 RAG / agent 时代，**NER / 实体链接 / parsing 又以 tool 形式回归**——LLM 自己做不准的精细任务，调外部专用模型完成。**模块化没死，只是从架构层迁到了 tool 层**。

---

## 3. 最小代码示例

不要试图实现这些任务，本节是地图不是教程。用 HuggingFace `pipeline` 一次跑通 5 类任务的代表，让读者直观感受"原来这些任务都能 1 行调起来"。

```python
from transformers import pipeline

# 1) 文本分类（情感）
clf = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
print(clf("I love this movie"))
# [{'label': 'POSITIVE', 'score': 0.999}]

# 2) 序列标注（NER），aggregation_strategy='simple' 把 BIO 标签合成 entity span
ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
print(ner("Barack Obama was born in Hawaii."))
# [{'entity_group': 'PER', 'word': 'Barack Obama', ...},
#  {'entity_group': 'LOC', 'word': 'Hawaii', ...}]

# 3) Seq2Seq（中英翻译）
translator = pipeline("translation_zh_to_en", model="Helsinki-NLP/opus-mt-zh-en")
print(translator("今天天气很好"))
# [{'translation_text': 'The weather is fine today.'}]

# 4) 抽取式 QA（从 context 中抽 span）
qa = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
print(qa(question="Where was Obama born?", context="Barack Obama was born in Hawaii."))
# {'answer': 'Hawaii', 'start': 25, 'end': 31, 'score': 0.97}

# 5) NLI / 句子相似度（用 sentence-transformers 算 cosine）
from sentence_transformers import SentenceTransformer, util
emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
v = emb.encode(["a man is eating food", "a person is having a meal"])
print(util.cos_sim(v[0], v[1]).item())   # ≈ 0.85
```

关键点：

- **每行一个 task，背后是一个完整的 fine-tuned 模型**（HuggingFace Hub 上有上万个针对单任务的 BERT / DistilBERT 变体）。这正是 BERT 时代"每任务一个模型"的真实体现。
- **任务名（`sentiment-analysis` / `ner` / `translation_zh_to_en` / `question-answering`）就是任务族的标签**，与本节 §2 的分类一一对应。
- 跑一遍可以肉眼对比"任务格式 vs LLM prompt"——后者只需要把上述 5 个调用都换成 `chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}])`，prompt 里描述任务即可。

---

## 4. 工程踩坑与经验

- ❗ **NER 用 BIO 时相邻同类实体会被合并**——"张三李四"两个人名连写会被预测成一个 PER。BIOES 五标（B/I/O/E/S）显式标 End / Single 解决边界歧义，工业 NER 默认推荐 BIOES。如果坚持用 BIO，至少要做"连续 I- 但前面不是 B-"这种异常情况的后处理。
- ❗ **NER 评测必须用 entity-level F1，不是 token-level F1**——`seqeval` 是事实标准。token-level F1 会把"标对 90% token 但实体边界全错"判成高分，工业上线后会被 PM 当场打脸。
- ❗ **Seq2Seq 的 BLEU 高 ≠ 翻译质量好**——"the the the the"也可能 BLEU 不低（n-gram 命中），而合理改写常常 BLEU 低但人眼看更好。LLM 时代用 BLEURT / COMET / GPT-4-as-judge 替代，或至少加 chrF 这类 character-level 指标做交叉验证。Module 12.2 详讲。
- ❗ **抽取式 QA 的 SQuAD F1 是 token-level、不是字面相等**——预测 "Albert Einstein" 与 ground-truth "Einstein, Albert" 算 F1 ≈ 1.0；预测 "the Einstein" 算 F1 ≈ 0.67（多了一个 token "the"，少了一个 token "Albert"）。新手用字面 == 评估会得到远低于报道的分数。
- ❗ **NLI 数据有大量 annotation artifact**——SNLI / MNLI 里 contradiction 类的 hypothesis 包含否定词（not / no / never）的频率远高于 entailment 类，模型只看 hypothesis（不看 premise）也能拿到 60%+ accuracy。这是 spurious correlation 经典案例，引出 ANLI / HANS 等"对抗版本" benchmark。做 NLI 评测一定要 cross-validate 多个数据集。
- ❗ **prompt-based 把所有任务统一后，输出格式不稳定**——模型可能输出"我觉得这条是正面，因为..."而不是只输出"正面"。生产环境必须用 constrained decoding（Outlines / XGrammar，Module 13.4）或 JSON schema + 后处理 + 重试 + fallback，否则下游解析会随机崩。
- ❗ **不要用 `BertTokenizer.encode()` 在 NER 上直接 align 标签**——sub-word 切分会把 "playing" 切成 `play ##ing`，labels 必须按 word piece 重新对齐（推荐 `tokenizer(..., is_split_into_words=True)` + `word_ids()` 自动对齐，HuggingFace 教程标准范式）。手工对齐 bug 多到劝退。

---

## 5. 经典 paper

- **Devlin et al., 2018 — BERT: Pre-training of Deep Bidirectional Transformers** — 确立 "pretrain + 任务 head fine-tune" 的标准范式，本节 §2.1-2.5 的每类任务在 BERT 论文 §4 都给了 head 设计与 GLUE / SQuAD 实验。读完才能理解为什么 LLM 时代之前每类任务都是一套独立 pipeline。
- **Raffel et al., 2020 — T5: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer** — "把所有任务都当 text-to-text"的范式重塑，是 LLM 时代任务统一的先声。读 §2-3 的统一 input format 设计，本节 §2.6 的范式平移表就是 T5 思想的简化版。
- **Brown et al., 2020 — GPT-3: Language Models are Few-Shot Learners** — in-context learning 让任务定义从 fine-tune head 进一步退化为 prompt。读 §3 的 zero-shot / one-shot / few-shot 实验，理解为什么 GPT-3 之后 NLP benchmark 大规模转向 prompt-based 评测。
- **Wang et al., 2018 — GLUE: A Multi-Task Benchmark and Analysis Platform** — 把 9 个 NLP 任务（分类 / NLI / 相似度）打包成统一 benchmark，本节 §2.1 / §2.5 的几乎所有数据集都在 GLUE 里。后续 SuperGLUE（Wang 2019）补了更难的任务，是 BERT-era benchmark 的事实标准。

---

## 6. 自测与面试题

**Q1（概念）**：为什么 BERT 时代每个任务都要接一个 task-specific head，而 GPT-3 之后大家直接用 prompt？这两种范式各有什么优缺点？

<details>
<summary>Answer sketch</summary>

至少要点到：

- **BERT 时代的 head 范式**：encoder 输出 hidden state，每类任务接不同的 head（分类 → `[CLS]` + Linear；NER → per-token Linear；抽取 QA → start/end Linear；Seq2Seq → encoder-decoder）。每个任务单独 fine-tune，训练目标明确、输出格式可控、监督信号强。
- **GPT-3 时代的 prompt 范式**：把任务描述写进 prompt，模型一律输出自由文本；所有任务共享一个 weights，靠 in-context learning 完成。
- **head 范式优点**：监督信号强、输出格式可控、单任务 SOTA 容易达到。**缺点**：每个任务一套 weights、不能跨任务迁移、加新任务要重训。
- **prompt 范式优点**：multi-task 一套 model、零样本可迁移到新任务、天然支持指令风格交互。**缺点**：输出格式不稳定（要 constrained decoding / 后处理）、单任务精度可能低于 fine-tuned 专用模型、prompt 设计本身成为新工程问题。
- 加分点：T5 是过渡形态——所有任务都 text-to-text 但仍 fine-tune；GPT-3 进一步去掉 fine-tune；今天的实际工程是混合（prompt + few-shot + 必要时 SFT/LoRA）。

</details>

**Q2（实战）**：你拿到一个新任务"判断这条新闻是否包含金融欺诈线索"，请描述 3 种可能的解法（fine-tune classifier / prompt + LLM / agent + 工具），分别比较 cost / latency / accuracy / interpretability。

<details>
<summary>Answer sketch</summary>

三种解法的对比要点：

| 维度 | (a) Fine-tune classifier | (b) Prompt + LLM | (c) Agent + 工具 |
|---|---|---|---|
| **方案** | 标几千条数据，BERT/DeBERTa 二分类 fine-tune | 写 prompt + few-shot，调 GPT-4o / Claude | LLM 调用 NER 抽实体 + 检索内部黑名单 + 信用风控规则引擎 |
| **训练 cost** | 中（标数据 + GPU 训几小时） | 低（只需写 prompt） | 高（要搭工具链 + 写 agent loop） |
| **推理 cost** | 极低（小模型） | 中-高（API 按 token 计费） | 高（多步调用累加） |
| **latency** | 几十 ms | 1-5 秒 | 5-30 秒 |
| **accuracy** | 数据足够时最高，少样本时差 | 数据少时最优，长尾欺诈类型靠 prompt 难覆盖 | 准确率 + 可解释性兼顾，但工具失败会级联 |
| **interpretability** | 弱（黑盒分类 score） | 中（可让 LLM 输出理由，但不一定真实） | 强（每步证据可追溯） |

加分点：

- 实务上常组合使用——先 fine-tune 小模型粗筛大批量，疑似的进入 LLM 二审，高风险的进入 agent + 人工 review。
- 监管 / 合规场景必须选 (c)，因为 audit trail 是硬要求。
- 数据漂移快的场景（新欺诈手法每周变）选 (b)，因为 prompt 改一下就能上线，fine-tune 重训太慢。

</details>

**Q3（评测）**：为什么生成式 QA 不能用 EM / F1？至少 2 个原因 + 替代方案。

<details>
<summary>Answer sketch</summary>

至少 2 个原因：

- **同一事实有无数种表达**：问 "Where was Obama born?"，正确答案 "Hawaii" / "He was born in Hawaii" / "In Honolulu, Hawaii" / "美国夏威夷" 都对，但 token-level F1 只能匹配字面 token，会大量误判。EM 更是只接受完全一致的字符串。
- **生成答案常带解释 / 修饰**：LLM 倾向输出 "Obama was born in Hawaii in 1961." 而不是只输出 "Hawaii"——多出的 token 会拉低 F1 score 但答案完全正确。
- **多语言 / 跨形式回答**：问中文给英文回答（或反之）、用代码块回答数值题——n-gram 指标完全失效。
- **抽取式 QA 不存在这个问题**：因为 ground-truth answer 一定是 context 中的连续 span，预测也是 span，所以 token-level F1 才合理。

替代方案：

- **LLM-as-judge**：用 GPT-4 / Claude 给生成答案打 1-5 分，prompt 里给出 ground-truth + criteria。Module 12.2 详讲。
- **Semantic similarity**：用 embedding 模型算 cosine，BERTScore / BLEURT 是这条路线的代表。
- **Fact-checking**：把生成答案分解成 atomic facts，用 retrieval 比对 ground-truth knowledge base（FActScore 风格）。
- **Multi-reference + best-of**：收集多个人工写的等价答案，取 max F1。
- 加分：生成式 QA 在 RAG / agent 体系下还要看 grounding（答案是否来自检索结果），这是另一维度的评测。

</details>

---

## 7. 延伸阅读

- [HuggingFace NLP Course — Chapter 7: Main NLP Tasks](https://huggingface.co/learn/nlp-course/chapter7) — 5 大任务族每类一节，配 transformers 代码端到端跑通，本节"任务地图"的最佳实操配套。
- [Papers with Code — NLP Benchmark](https://paperswithcode.com/area/natural-language-processing) — 每个任务的 SOTA 排行榜与对应 paper，跟踪范式演进必备。
- [seqeval 库文档](https://github.com/chakki-works/seqeval) — entity-level F1 的事实标准实现，BIO / BIOES / IOB1 都支持，做 NER 必装。
- [Lin 2024 — A Survey on LLM-as-a-Judge](https://arxiv.org/abs/2411.15594) — 系统综述 LLM 评测从 n-gram 指标到 LLM-judge 的演进，对应本节 §4 中"BLEU 为什么不重要了"的延伸。
- 推荐继续读本教程的 **2.2 节《词向量：word2vec / GloVe / fastText》**——把"任务地图"落到第一代分布式表示的具体技术上。
