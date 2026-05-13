---
title: "3.1 BPE / WordPiece / Unigram / SentencePiece"
description: "讲清现代 LLM 为什么不用 word-level 也不用 char-level，而是统一走 subword tokenization——并把 4 种主流算法（BPE / WordPiece / Unigram / SentencePiece）的 idea、合并准则、训练算法、谁在用摆出来，让你看到 GPT、LLaMA、Qwen、BERT、Gemma 的 tokenizer 选型不是随机的，背后是"
---

> ⏱ 预计阅读 35 分钟 ｜ 难度 ★★ ｜ 前置：无（2.2 fastText 的 subword 思想是同源铺垫）

## 一句话本节讲什么

讲清现代 LLM 为什么不用 word-level 也不用 char-level，而是统一走 **subword tokenization**——并把 4 种主流算法（**BPE / WordPiece / Unigram / SentencePiece**）的 idea、合并准则、训练算法、谁在用摆出来，让你看到 GPT、LLaMA、Qwen、BERT、Gemma 的 tokenizer 选型不是随机的，背后是几条清晰的 trade-off 主线。

---

## 1. Mental model（直觉）

文本进入模型之前要先变成整数 id 序列，这一步就是 **tokenization**。表面上是个无聊的工程问题，实际上是 LLM 全栈里第一个绕不开的设计选择——它直接决定 **词表大小**、**embedding 矩阵规模**、**序列长度**、**多语言/代码友好度**、**OOV 行为**，甚至 **API 计费**（OpenAI 按 token 收钱）。

历史上有三种切法：

```
   word-level                char-level                    subword (BPE/WPC/Unigram)
"unbelievable" → 1 token    "unbelievable" → 12 tokens    "unbelievable" → ["un", "believ", "able"]
                                                                              3 tokens
词表 100w+ (爆炸)           词表 256 (字节)                词表 3w-15w (可控)
OOV 严重                    几乎无 OOV                     最差能拆到 byte，无 OOV
形态丰富语言灾难             序列长 5-10×，长程依赖更难      序列长度合理，多语言通吃
```

- **Word-level**：自然但词表爆炸——英文光屈折形态（play/plays/playing/played/player...）就把词表撑到百万级；俄语、芬兰语、土耳其语等形态丰富语言更夸张，一个词根有上百种屈折。OOV（out-of-vocabulary）严重——线上一旦遇到没见过的词只能扔 `<unk>`，对低频实体名、新词、拼写错误完全无能为力。
- **Char-level**：词表很小（英文 ASCII 256，加 Unicode 不过几万），几乎不存在 OOV——但代价是**序列变长 5-10 倍**：同一句话原本 200 token，char-level 要 1500-2000 token。Self-attention 是 $O(n^2)$，序列变长 5 倍计算和显存就翻 25 倍，长程依赖也更难学。
- **Subword 是甜点**：词表 3 万到 15 万可控、序列长度温和、几乎无 OOV（最差也能拆到单字节）。**所有现代 LLM 都走这条路**——分歧只在算法细节上。

subword 的核心 idea 用一句话概括：**让高频整词保持一个 token，让低频/未见过的词拆成可见过的 subword 片段拼起来**。`tokenization` → `[token, ization]`，`unbelievable` → `[un, believ, able]`，`covid-19` → `[c, ovid, -, 19]`。这样既享受了 word-level 的"信息密度高、序列短"，又享受了 char-level 的"无 OOV、词表小"。

剩下 4 节回答的是同一个问题：**怎么从一堆文本无监督地学出"哪些 subword 该进词表、怎么切"**。BPE 用频率贪心、WordPiece 用似然贪心、Unigram 用概率模型 + EM 反向 prune，SentencePiece 是把这些算法工具化、language-agnostic 化的工具。

---

## 2. 公式与原理

### 2.1 BPE（Byte-Pair Encoding）

BPE 起源于 1994 年的数据压缩算法（Gage），Sennrich 2016 把它引入 NMT 解决 rare word 问题，从此成为 LLM tokenizer 的事实标准（GPT 系全家、LLaMA 系全家）。

**算法步骤**（训练）：

1. **初始化**：把每个 word 拆成 character 序列，词表 $\mathcal{V}_0$ = 所有出现过的 character
2. **统计**：在当前 corpus 上数所有相邻 token 对 $(x, y)$ 的频次
3. **合并**：选频次最高的 pair $(x^*, y^*)$，把它合并成新 token $x^* y^*$ 加入词表，corpus 中所有出现的 $(x^*, y^*)$ 替换成 $x^*y^*$
4. **重复**：第 2-3 步迭代直到达到目标词表大小 $V$（典型 30k-150k）

每次合并产生一条 **merge rule**——训练完得到一组按顺序的 merge rules，推理时严格按这个顺序对输入文本反复 apply 直到无可合并。

时间复杂度大约 $O(N \cdot V)$（$N$ 是 corpus token 数、$V$ 是目标词表大小），现代实现（HuggingFace `tokenizers` Rust + tiktoken C++）在亿 token 级语料上几十分钟到几小时即可训完。

**Byte-level BPE**（GPT-2 起用）：原版 BPE 在 character level 工作，遇到从未见过的 Unicode 字符（emoji、生僻汉字）仍然会 OOV。Radford 2019 把初始 alphabet 改成 **256 个 UTF-8 字节**——任何文本都能被 UTF-8 编码成字节序列，而 256 个字节是有限可枚举的，**这就把 OOV 彻底从 LLM 输入端消灭了**。代价是中文、emoji 这种"一个字符 = 多个 byte"的 token 在序列里更长，需要靠 BPE 的合并机制把常见字符合回去。

**现代变体——tiktoken**：OpenAI 自家的 byte-level BPE 实现（C++ 加速，比 HF tokenizers Python 实现快 3-5 倍），定义了几个标准词表：

- `r50k_base`：GPT-3 / Codex，词表 50,257
- `cl100k_base`：GPT-3.5 / GPT-4，词表 100,277
- `o200k_base`：GPT-4o / o1，词表 199,997（中文/多语言友好度大幅提升）

**OpenAI 计费严格按这几个词表算 token 数**——估错 token 直接影响成本。

### 2.2 WordPiece

Schuster & Nakajima 2012 在 Google 的语音搜索系统中提出，BERT（Devlin 2018）让它出名。算法骨架和 BPE 几乎一样，**唯一差异在合并准则**：

- BPE 合并最频繁的 pair：$\arg\max_{(x,y)} \text{count}(x, y)$
- WordPiece 合并使语料 likelihood 增益最大的 pair：

$$\arg\max_{(x, y)} \frac{\text{count}(x, y)}{\text{count}(x) \cdot \text{count}(y)}$$

直觉：BPE 看绝对共现频次，WordPiece 看 **共现的"超出独立假设"程度**——如果 $(x, y)$ 经常出现仅仅因为 $x$ 和 $y$ 都很高频（比如 `the` 后接 `e`），不该合并；只有它们一起出现的概率显著高于独立时才合并。形式上等价于在一个 unigram language model 下，合并这一对能让训练 corpus 的 log-likelihood 增益最大：

$$\Delta \mathcal{L} = \log \frac{P(xy)}{P(x) P(y)}$$

工程上 WordPiece 训练略慢于 BPE（要算分数而非数频次）、效果与 BPE 相近（多数下游任务差距 < 1 个点）。BERT 系（含 RoBERTa、DistilBERT、ELECTRA）几乎全用 WordPiece，**典型标识是带 `##` 前缀**：`playing` → `play`, `##ing`，`##` 表示 "接前一个 token 不带空格"。

### 2.3 Unigram LM

Kudo 2018 提出，与 BPE/WordPiece 的"自底向上贪心合并"思路相反，Unigram 是**自顶向下 prune**：

1. **初始化大词表**：用启发式（高频字符串 + 后缀数组）准备一个超大候选词表（典型 100w-300w）
2. **赋概率**：每个 token $w$ 有概率 $P(w)$，对一个 word/sentence $\mathbf{x}$ 的切分 $\mathbf{s} = (s_1, \dots, s_m)$ 概率为
   $$P(\mathbf{s}) = \prod_{i=1}^{m} P(s_i)$$
   一个 word 通常有多种切分方式 $\mathcal{S}(\mathbf{x})$（lattice），可计算 $P(\mathbf{x}) = \sum_{\mathbf{s} \in \mathcal{S}(\mathbf{x})} P(\mathbf{s})$
3. **EM 估计**：用 EM 算法在固定词表下估计 $P(w)$（E 步算每种切分的后验，M 步更新 $P(w)$ = 期望出现次数 / 总数）
4. **Prune**：算每个 token 的"删除影响"（删了它训练 likelihood 下降多少），删掉影响最小的那一批
5. **重复 3-4** 直到词表降到目标大小 $V$

推理时 Unigram 用 Viterbi 找 **概率最大的切分** $\arg\max_\mathbf{s} P(\mathbf{s})$。

Unigram 最独特的能力是 **subword regularization**——既然一个 word 有多种合法切分，训练时可以**随机采样不同切分**当数据增强，让模型对 tokenization 噪声更鲁棒（论文显示在低资源 NMT 上能涨 1-2 个 BLEU）。BPE 也有对应的 BPE-dropout 实现类似效果，但 Unigram 是天然支持的。

Unigram 适合 **形态丰富、多语言** 场景，是 SentencePiece 的默认算法，T5、ALBERT、Gemma 在用。

### 2.4 SentencePiece

Kudo & Richardson 2018 提出，**严格说 SentencePiece 不是新算法，是个工具**——它做了三件事让 tokenization 真正 language-agnostic：

- **不预 tokenize**：直接对原始 raw text 训练，不需要 jieba 切中文 / Moses 切英文 / MeCab 切日文。**这一步在中文 LLM 上意义巨大**——把切词器选择这个历史遗留问题彻底废掉。
- **空格也是 token**：把空格替换成可见字符 `▁`（U+2581）当普通字符处理。`Hello world` → `▁Hello ▁world`，这样 detokenize 时只要把 `▁` 换回空格就 lossless，无论是不是用空格分词的语言都能统一处理。
- **多算法支持**：内部支持 `--model_type=bpe` 和 `--model_type=unigram` 两种模式，工程切换只改一个参数。

**SentencePiece 是当代多语言 LLM 的事实标配**——LLaMA 1/2、Mistral、Qwen 早期版本（v1）、Gemma、T5、ALBERT 都用它。LLaMA 系用的是 SentencePiece **BPE 模式**，Gemma/T5 用的是 **Unigram 模式**。

### 2.5 现代 LLM tokenizer 选型对比

| 模型 | 算法 | 词表大小 | 实现 |
|---|---|---|---|
| GPT-2 | byte-level BPE | 50,257 | tiktoken (`r50k_base`) |
| GPT-3.5 / GPT-4 | byte-level BPE | 100,277 | tiktoken (`cl100k_base`) |
| GPT-4o / o1 | byte-level BPE | 199,997 | tiktoken (`o200k_base`) |
| Llama 2 | SentencePiece BPE | 32,000 | sentencepiece |
| Llama 3 | tiktoken-style byte BPE | 128,256 | tiktoken-compatible |
| Qwen 2.5 | byte-level BPE (tiktoken-style) | ~152,064 | 自研（兼容 HF） |
| DeepSeek-V3 / R1 | byte-level BPE | ~129,280 | 自研 |
| BERT | WordPiece | 30,522 | bert-tokenizer |
| T5 | SentencePiece Unigram | 32,128 | sentencepiece |
| Gemma 2 | SentencePiece Unigram | 256,128 | sentencepiece |
| Mistral / Mixtral | SentencePiece BPE | 32,000 | sentencepiece |

几条规律：

- **GPT 系坚持 byte-level BPE**——OpenAI 一以贯之，`o200k_base` 通过扩词表大幅改善多语言效率
- **早期开源 LLM（LLaMA 1/2/Mistral）用 SentencePiece BPE 32k**——简单、多语言通吃、训练成本低
- **新一代开源 LLM（LLaMA 3、Qwen 2.5、DeepSeek、Gemma 2）词表都扩到了 100k-256k**——主要为多语言/代码效率，但带来 embedding 矩阵膨胀
- **BERT 系用 WordPiece** 是历史遗留——新模型很少再选 WordPiece

### 2.6 Trade-off 与影响

- **词表大** → embedding 矩阵 $W \in \mathbb{R}^{V \times d}$ 更大（70B 模型 $V=128k, d=8192$ 时光 input embedding 就 1B 参数，加上 tied output 不变），但每个 token 信息密度更高、平均序列更短、推理更快
- **词表小** → embedding 省，但 tokenization 低效——同样的中文段落可能多出 30-50% token
- **多语言 LLM 词表往往更大** —— 要覆盖各语言的常见 subword（Gemma 2 用 256k 是为了 100+ 语言）
- **tokenizer 不一致 → embedding 完全不兼容**——这就是 RAG 系统里 embedding 模型（如 bge-m3）和生成 LLM（如 Qwen-72B）的 token id 完全对不上，必须分别 tokenize

---

## 3. 最小代码示例

### 3.1 手撕 BPE 训练（≤ 35 行）

```python
from collections import Counter, defaultdict

def train_bpe(corpus, num_merges):
    """
    corpus: list[str]，每个 str 是一个 word（已按空格 / 任意 pre-tokenizer 切过）
    return: list[(str, str)]，按顺序的 merge rules
    """
    # 1. 初始化：每个 word 拆成 char 序列，结尾加 </w> 标记词边界
    word_freqs = Counter(corpus)
    splits = {w: list(w) + ["</w>"] for w in word_freqs}

    merges = []
    for _ in range(num_merges):
        # 2. 数所有相邻 pair 的频次（按 word 频次加权）
        pair_freqs = defaultdict(int)
        for w, freq in word_freqs.items():
            symbols = splits[w]
            for i in range(len(symbols) - 1):
                pair_freqs[(symbols[i], symbols[i + 1])] += freq

        if not pair_freqs:
            break
        # 3. 选最频繁的 pair 合并
        best = max(pair_freqs, key=pair_freqs.get)
        merges.append(best)

        # 4. 在所有 word 的 split 里 apply 这个 merge
        for w in splits:
            symbols, new_symbols, i = splits[w], [], 0
            while i < len(symbols):
                if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == best:
                    new_symbols.append(symbols[i] + symbols[i + 1])
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            splits[w] = new_symbols
    return merges

# Demo
corpus = ["low"] * 5 + ["lower"] * 2 + ["newest"] * 6 + ["widest"] * 3
print(train_bpe(corpus, num_merges=10))
# [('e', 's'), ('es', 't'), ('est', '</w>'), ('l', 'o'), ('lo', 'w'), ...]
```

关键点：第 11 行按 word 频次对 pair 加权（不是只看一个 word 内的 pair 数）；第 16 行 `max` 选最频繁的；第 21-26 行在每个 word 的 split 上 apply merge——这就是 BPE 的全部，30 行写完。**这段代码可以直接跑**，作为面试白板题非常常见。

### 3.2 用 tiktoken 演示 GPT-4 tokenizer 中英文 token 数对比

```python
# pip install tiktoken
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")  # GPT-3.5 / GPT-4 用的词表
print(enc.encode("hello world"))
# [15339, 1917]                        → 2 tokens
print(enc.encode("你好世界"))
# [57668, 53901, 3574, 244, 98220]     → 5 tokens（4 个汉字 = 5 token，平均 1.25 token/字）

# o200k_base 对中文友好得多（GPT-4o）
enc2 = tiktoken.get_encoding("o200k_base")
print(enc2.encode("你好世界"))
# [177519, 99489]                      → 2 tokens（4 个汉字仅 2 token）

# 估算 1000 个中文字符的成本
text = "人工智能" * 250   # 1000 字
print(f"cl100k: {len(enc.encode(text))} tokens")    # ~1500 tokens
print(f"o200k:  {len(enc2.encode(text))} tokens")   # ~600 tokens
```

中文在 `cl100k_base` 上每字 1.5-3 token，到 `o200k_base` 改善到 0.5-1 token——**OpenAI 计费按这步算**，估算成本必须用对应词表。

### 3.3 用 transformers 加载 LLaMA tokenizer

```python
# pip install transformers sentencepiece
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
print(tok.encode("Hello world"))
# [128000, 9906, 1917]   # 128000 是 BOS=<|begin_of_text|>

# Special token & chat template
messages = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "今天天气如何？"},
]
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(prompt)
# <|begin_of_text|><|start_header_id|>system<|end_header_id|>
# You are helpful.<|eot_id|><|start_header_id|>user<|end_header_id|>
# 今天天气如何？<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```

`apply_chat_template` 把 messages 列表渲染成模型预期的格式，含特殊 token（`<|begin_of_text|>`、`<|eot_id|>` 等）和 role 标记——**SFT 与 inference 必须用同一个 chat template，否则模型输出全乱**（详见 8.2）。

### 3.4 用 sentencepiece 训自己的 tokenizer

```python
# pip install sentencepiece
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="my_corpus.txt",        # 一行一个 sentence 的纯文本
    model_prefix="my_tokenizer",  # 输出 my_tokenizer.model + my_tokenizer.vocab
    vocab_size=32000,
    model_type="bpe",             # 或 "unigram"
    character_coverage=0.9995,    # 中文/日文建议 0.9995；英文用 1.0
)

sp = spm.SentencePieceProcessor(model_file="my_tokenizer.model")
print(sp.encode("hello 世界", out_type=str))
# ['▁hello', '▁', '世', '界']
```

10 行内训完——SentencePiece 直接吃 raw text，不需要预切词。

---

## 4. 工程踩坑与经验

- ❗ **中文/日文每字约 2-3 token（GPT-4 cl100k_base）**——1000 中文字大约 2000-3000 token，不要按"1 token = 1 word"估算 prompt 成本，会差 2-3 倍。GPT-4o 的 o200k_base 把中文压到约 0.5-1 token/字，改善很大但仍不能按 word 算。
- ❗ **tokenizer 训练时必须包含目标语言/领域语料**——拿英文 tokenizer 跑中文/代码 token 效率差 3-5 倍。这是中文 LLM 团队普遍要做"扩中文词表"或自训 tokenizer 的根本原因。同理：医疗/法律/金融垂域模型如果直接用通用 LLaMA tokenizer，专业术语会被切得支离破碎。
- ❗ **加 special token 时不要忘了 `model.resize_token_embeddings(len(tokenizer))`**——`tokenizer.add_special_tokens({...})` 只改了 tokenizer，embedding 矩阵 $W \in \mathbb{R}^{V \times d}$ 没扩，新 token id 一查就越界报 IndexError 或随机噪声 embedding（更阴险，模型不报错但效果稀烂）。SFT 和加 reasoning thinking token 时是高发场景。
- ❗ **LLaMA / SentencePiece 的"前缀空格"问题**：`tokenizer.encode("Hello")` 与 `tokenizer.encode(" Hello")` 通常不一样——SentencePiece 默认会在文本首加 `▁`，但如果你手动拼 prompt 时已经有空格，可能产生 `▁` `▁Hello` 这种双空格 token，模型从未见过这个组合，效果异常下降。`add_special_tokens=True/False` 和 `add_prefix_space` 都要调清楚，或者用 `apply_chat_template` 让框架处理。
- ❗ **tiktoken 的 `cl100k_base` 是 GPT-3.5/GPT-4 用的，不能算 token 时省掉这步**——OpenAI 计费严格按这个词表算，用 `len(text.split())` 或 `len(text) / 4` 这种近似估算可以差 30-100%，正式预算和限流都按 tiktoken 实测。GPT-4o 必须换 `o200k_base`。
- ❗ **自训 tokenizer 不要用过大词表（200k+）**——对 70B 量级模型，$W_{\text{embed}} = V \cdot d = 200k \cdot 8192 \approx 1.6\text{B}$ 参数，加上 tied output 仍是这个数（不 tied 就翻倍），占总参数 2-3%、占 fp16 显存 3.2GB——而且 vocab logits 的 softmax 是推理热点，词表大幅影响首 token 延迟。**词表大小要按"覆盖率收益曲线"切，到 90-95 分位停**，不是越大越好。
- ❗ **byte-level BPE 解码遇到非法 UTF-8 序列时会报错**——流式生成时如果一个汉字（3 byte UTF-8）只生成了 2 byte 就 stop，`tokenizer.decode` 会抛异常或显示乱码 `�`。生产环境要用 `errors="replace"` 兜底，或在 streaming layer 做 byte buffer 等下一个 token 凑齐 UTF-8 再 decode。
- ❗ **不同 tokenizer 的 BOS/EOS/PAD 行为完全不同**——LLaMA 默认有 BOS 无 EOS（chat template 里手动加 `<|eot_id|>`），Qwen 有 EOS 无 BOS，BERT 用 `[CLS]` `[SEP]`。SFT 数据 pack 时要 align 到本模型的 special token 集合，不要拷贝其他模型的 template。

---

## 5. 经典 paper

- **Sennrich, Haddow & Birch, 2016 — Neural Machine Translation of Rare Words with Subword Units** — 把 BPE 从 1994 年的数据压缩算法引入 NLP，奠定现代 LLM tokenizer 的事实标准。读 §3 算法描述（5 行伪代码就讲清了），重点理解 **"为什么 BPE 能把 OOV 问题转化成 subword 序列问题"**。Take-away：所有 GPT 系都用这套，必读。
- **Kudo & Richardson, 2018 — SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing** — SentencePiece 工具论文，工程价值大于理论价值。读 §3 即可——**为什么把空格也作为 token、为什么不预 tokenize**。Take-away：现代多语言 LLM 工具链选型的"事实标准"，理解它能让你看懂 LLaMA / Mistral / Gemma / T5 的 tokenizer 文件。
- **Kudo, 2018 — Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates** — Unigram LM 提出 + subword sampling 正则化。读 §3-4 unigram model 与 EM 训练流程。Take-away：唯一一种支持"概率切分采样"的 subword 算法，T5/Gemma/ALBERT 选它的根本原因。
- **Schuster & Nakajima, 2012 — Japanese and Korean Voice Search** — WordPiece 的起源（谷歌内部用了 6 年才发 paper），BERT 让它出名。读 §2.3 即可——**WordPiece 的 likelihood-based 合并准则与 BPE 频率准则的区别**。Take-away：理解 BERT 系的 `##` 前缀从哪来。

加分阅读：[OpenAI tiktoken GitHub](https://github.com/openai/tiktoken)——OpenAI 自家 byte-level BPE C++ 实现，附详细的 vocab 文件与 cookbook，工业最快的 BPE encoder。

---

## 6. 自测与面试题

**Q1（算法）**：写出 BPE 训练算法的伪代码（5 步左右），并用 corpus `["low"×5, "lower"×2, "newest"×6, "widest"×3]` 演示一轮合并过程。

<details>
<summary>Answer sketch</summary>

伪代码 5 步：

1. **Init**：每个 word 拆成 char 序列（可加词尾 `</w>` 标记），词表 = 所有 char
2. **Count**：对所有 word（按 word 频次加权）数所有相邻 pair 的频次
3. **Pick**：选频次最高的 pair $(x^*, y^*)$
4. **Merge**：把这个 pair 加入词表，corpus 中所有 $(x^*, y^*)$ 替换成 $x^*y^*$
5. **Repeat 2-4** 直到达到目标词表大小 $V$

第一轮演示：

- 初始 split：`l o w </w>` × 5、`l o w e r </w>` × 2、`n e w e s t </w>` × 6、`w i d e s t </w>` × 3
- 数 pair（按 word 频次加权）：`(e, s)` 出现在 newest×6 和 widest×3 共 9 次、`(s, t)` 同样 9 次、`(l, o)` 7 次、`(o, w)` 7 次……
- 选 `(e, s)`（或 `(s, t)`，并列时实现定）合并 → 新 token `es`（或 `st`）
- 更新 split：`n e w es t </w>` × 6、`w i d es t </w>` × 3
- 加分：解释为什么按 word 频次加权（一个高频 word 内的 pair 应被多算）；提到 byte-level BPE 把初始 alphabet 换成 256 个 byte 解决 Unicode OOV。

</details>

**Q2（trade-off）**：BPE / WordPiece / Unigram 三种算法从 (合并准则 / 训练速度 / 多切分支持 / 现代 LLM 谁在用) 4 个维度对比。

<details>
<summary>Answer sketch</summary>

| 维度 | BPE | WordPiece | Unigram |
|---|---|---|---|
| 合并准则 | 最频繁 pair $\arg\max \text{count}(x,y)$ | likelihood 增益最大 $\arg\max \frac{P(xy)}{P(x)P(y)}$ | 反向：先大词表 + EM 估概率 + prune 影响最小的 |
| 训练速度 | 最快（贪心数频次） | 略慢（要算 likelihood 分数） | 最慢（EM 多轮 + prune 多轮） |
| 多切分支持 | 原版无（BPE-dropout 是后加 trick） | 无 | 天然支持（Viterbi 找最大概率切分；训练时可采样） |
| 谁在用 | GPT-2/3/4/4o 全家、LLaMA 全家、Mistral、Qwen、DeepSeek（开源 LLM 主流） | BERT 全家、ELECTRA、DistilBERT | T5、ALBERT、Gemma、XLNet |

加分要点：

- BPE 与 WordPiece 算法骨架完全一样，**只在合并准则一行不同**
- WordPiece 的 likelihood 准则等价于 unigram LM 下的 mutual information 最大化
- Unigram 是"自顶向下 prune"，与 BPE/WordPiece"自底向上合并"思路完全相反
- subword regularization 是 Unigram 独有的优势——低资源 NMT/翻译任务能涨 1-2 BLEU
- 现代 LLM 选 BPE 多于 WordPiece，主要因为 byte-level BPE 在多语言/代码场景对 OOV 更鲁棒

</details>

**Q3（实战诊断）**：你训了一个中文 LLM，但用了 LLaMA 1 / 2 原版 tokenizer（SentencePiece BPE 32k，主要英文语料训）。线上发现推理特别慢、效果也比同 size 模型差很多。诊断思路 + 解决方案？

<details>
<summary>Answer sketch</summary>

**诊断**：

- **直接验证 token 效率**：拿一段 1000 字中文跑 `tokenizer.encode`，LLaMA 原版 tokenizer 在中文上往往每字 2-3 token（甚至更多，因为很多汉字会被切成多个 byte 再合并），同样的中文段落 token 数是中文友好 tokenizer 的 2-3 倍
- **推理慢的原因**：序列长度直接乘 2-3 → self-attention $O(n^2)$ 计算量乘 4-9 倍、KV cache 显存翻 2-3 倍 → TTFT、TBT、throughput 都崩
- **效果差的原因**：词表里几乎没有中文 subword，模型必须从 byte 级别学中文语义，等于把 transformer 内部很多容量浪费在"先拼字"这一步上；同时由于 token 太碎，long-range pattern 难学
- **附加症状**：输出可能出现 byte 级乱码（流式 decode 不全）

**解决方案**（按工程成本递增）：

- **方案 A：扩词表（vocabulary extension）**——在 LLaMA 原 tokenizer 基础上，用 SentencePiece 在中文语料上训一个 ~20k token 的中文 BPE，merge 进原词表（去重后总共 ~50k）。然后 `model.resize_token_embeddings`，新增的中文 token embedding 用旧的 byte 序列 embedding 平均初始化（一种常见 warm start）。**这是 Chinese-LLaMA / Chinese-Alpaca / Atom-7B 等中文 LLaMA 衍生工作的标准做法**，需要继续 pretrain 几十亿到几千亿 token 让新 token 收敛。
- **方案 B：换 tokenizer 重训**——直接用中文 + 英文 + 代码混合语料训一个新的 SentencePiece tokenizer（vocab 100k-150k），再从头预训。成本最大但效果最好，**Qwen / DeepSeek / Yi 都走这条路**。
- **方案 C（治标不治本）**：用现成的中文友好开源 LLM（Qwen / DeepSeek / Yi）的 tokenizer，不要硬上 LLaMA 1/2。
- **方案 D**：升级到 LLaMA 3，它已经把词表扩到 128k 且对中文友好得多。

加分要点：

- 提到扩词表后**新 embedding 初始化策略**（mean of constituent byte embeddings vs random）对收敛速度的影响
- 提到扩词表后**输出 lm_head 也要 resize**（如果 weight tied 会自动同步，否则要手动）
- 提到这个问题不只在中文出现——日文、韩文、泰文、阿拉伯文都有，凡是 LLaMA 原 tokenizer 训练语料覆盖不足的语言都要扩
- 提到 OpenAI 在 GPT-4 → GPT-4o 时把 `cl100k_base` 升级到 `o200k_base` 也是同一动机的商业版本

</details>

---

## 7. 延伸阅读

- [HuggingFace NLP Course — Tokenizers chapter](https://huggingface.co/learn/nlp-course/chapter6) — 最系统的 4 种 subword 算法实操教程，每个算法都有从零实现 + HF 库调用版本，配套 Colab，本节代码扩展首选。
- [OpenAI tiktoken GitHub](https://github.com/openai/tiktoken) — OpenAI 官方 byte-level BPE C++ 实现，含 `cl100k_base` / `o200k_base` 的 vocab 文件、详细 cookbook。算 GPT 系 token 数必备。
- [SentencePiece GitHub](https://github.com/google/sentencepiece) — Google 官方 SentencePiece 实现，README 与 wiki 把 BPE 模式和 Unigram 模式的训练参数讲得最清楚。
- [Andrej Karpathy — Let's build the GPT Tokenizer (YouTube, 2024)](https://www.youtube.com/watch?v=zduSFxRajkE) — Karpathy 2 小时手撕 GPT tokenizer 的视频，配 [minbpe GitHub](https://github.com/karpathy/minbpe)，把 byte-level BPE 的细节讲到极致。
- [Chinese-LLaMA-Alpaca repo](https://github.com/ymcui/Chinese-LLaMA-Alpaca) — 中文社区扩 LLaMA 词表的代表工作，可作为本节 Q3 实战的参考实现。
- 推荐继续读本教程的 **3.2 节《词表构造、特殊 token、OOV》** —— 进一步讲怎么设计 special token 集合（BOS/EOS/PAD/chat role/tool/thinking）、词表大小如何按覆盖率切、OOV 与 byte fallback 的兜底策略。
