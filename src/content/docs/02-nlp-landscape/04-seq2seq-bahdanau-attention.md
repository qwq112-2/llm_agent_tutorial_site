---
title: "2.4 Seq2Seq + Bahdanau Attention（动机：为什么需要 Transformer）"
description: "Sutskever 2014 的 vanilla seq2seq 把整句源语言压成一个固定长度向量，长句一压就丢；Bahdanau 2014 让 decoder 在每一步重新看一遍 encoder 的所有 hidden states 并按相关度加权——这就是\"attention\"的诞生，也是 Transformer 之所以叫 Attention is All You Need 的真正起点。本节讲清"
---

> ⏱ 预计阅读 30 分钟 ｜ 难度 ★★ ｜ 前置：2.3 RNN/LSTM/GRU

## 一句话本节讲什么

Sutskever 2014 的 vanilla seq2seq 把整句源语言压成**一个固定长度向量**，长句一压就丢；Bahdanau 2014 让 decoder 在每一步**重新看一遍** encoder 的所有 hidden states 并按相关度加权——这就是"attention"的诞生，也是 Transformer 之所以叫 *Attention is All You Need* 的真正起点。本节讲清楚 vanilla seq2seq 的瓶颈、Bahdanau / Luong 的公式与对比，并把"attention 机制如何一步步脱离 RNN 长成 self-attention"的演化逻辑铺到 4.1 self-attention 的门口。

---

## 1. Mental model（直觉）

### 1.1 vanilla seq2seq：把整句话塞进一个瓶子

机器翻译的形式化：给定源语言序列 $x = (x_1, \dots, x_T)$，生成目标语言序列 $y = (y_1, \dots, y_{T'})$。Sutskever, Vinyals & Le 2014（NIPS）和 Cho et al. 2014（EMNLP）几乎同时提出了**encoder-decoder** 范式：

```
   source: "I love NLP"                          target: "我  爱  NLP"
                                                          ▲   ▲   ▲
   ┌─────┐  ┌─────┐  ┌─────┐                    ┌─────┐  ┌─────┐  ┌─────┐
   │ RNN │─►│ RNN │─►│ RNN │═══ h_T ═══════════►│ RNN │─►│ RNN │─►│ RNN │
   └─────┘  └─────┘  └─────┘  (context vector)  └─────┘  └─────┘  └─────┘
      ▲        ▲        ▲                          ▲        ▲        ▲
      I       love     NLP                       <bos>      我       爱
                              encoder ───┘└─── decoder
```

- **Encoder**（一般是 LSTM/GRU）按时间吃完整个 source，把最后一步的 hidden state $h_T$ 当作整个 source 句子的"摘要"——这就是 **context vector**。
- **Decoder**（也是 LSTM/GRU）以 $h_T$ 为初始状态，autoregressively 生成 target：每步输入上一时刻预测的 token，输出下一时刻的 token 概率分布。
- **训练**：teacher forcing——decoder 每一步的输入用 ground-truth $y_{t-1}$（不是模型自己预测的），loss 是 per-token cross-entropy $\mathcal{L} = -\sum_t \log P_\theta(y_t \mid x, y_{<t})$。
- **推理**：greedy（每步取 argmax）或 beam search（保留 top-$k$ 路径）。

这个架构 2014 年是石破天惊的——之前主流机器翻译还是 IBM model + phrase-based SMT 那一套统计 + 词对齐 + 短语表的复杂 pipeline，seq2seq 用一个端到端的神经网络打掉了大半道工序。

### 1.2 致命瓶颈：信息瓶颈（information bottleneck）

仔细盯着 $h_T$ 看一眼。它是一个固定维度（典型 256 或 1024）的向量，却要承载**整句源语言**的所有信息——句法、语义、词序、长程依赖、专有名词。一句 5 词的句子塞进去尚可；一句 50 词的法语长句塞进同一个 256 维向量，**信息一定有损**。

Cho 自己在 2014 年同期的另一篇论文（*On the Properties of Neural Machine Translation*）里就报告了这个现象：vanilla seq2seq 的 BLEU 随着源句长度增加先上升后**急剧下降**——20 词以内表现尚可，50 词以上几乎崩盘。

直觉上的对比：人类翻译长句时**不会**把整句话先全记在脑子里再开始说，而是边读边译——译到当前位置时**回头看**与之相关的几个源词。这个"回头看 + 按相关度加权"的动作就是 attention。

### 1.3 Bahdanau 的关键 idea：每一步重新看一遍

Bahdanau, Cho & Bengio 2014（ICLR 2015）的题目就是 *Neural Machine Translation by Jointly Learning to Align and Translate*——"jointly learning to align" 是核心。让 decoder 在生成每一个 target token $y_t$ 时：

1. **重新看一遍** encoder 所有 timestep 的 hidden states $\{h_1, \dots, h_T\}$；
2. 计算 decoder 当前状态 $s_{t-1}$ 与每一个 $h_i$ 的"对齐分数" $e_{t,i}$；
3. softmax 出权重 $\alpha_{t,i}$，加权求和得到一个 **timestep-specific context vector** $c_t$；
4. 用 $c_t$ 而不是固定的 $h_T$ 来更新 decoder 状态、预测下一个 token。

```
                                  α_{t,1} α_{t,2} α_{t,3}  α_{t,T}
                                    │       │       │  ...   │
   encoder hidden states:         h_1     h_2     h_3       h_T
                                    │       │       │        │
                                    └───────┴───────┴────────┘
                                              │
                                            sum (weighted by α_{t,i})
                                              │
                                              ▼
                              context vector c_t  ───►  decoder step t
                                                          │
                                                          ▼  predict y_t
```

注意两件事：

- **每一个 decoder timestep 都重新算一次** $\alpha_{t,i}$ 和 $c_t$——不再有"那一个" context vector，而是 $T'$ 个 context vectors。
- **$\alpha_{t,i}$ 天然可解释**：它是 decoder 在生成第 $t$ 个 target 词时对第 $i$ 个 source 词的注意力权重，可视化出来就是经典的 NMT attention map（详见 §4 工程踩坑里的 caveat）。

这一招直接把 NMT 的 BLEU 在长句上拉回来——**Bahdanau 2014 成为 attention 这个机制在深度学习里的诞生时刻**。后续的 Luong attention、self-attention、Transformer，全部是在这个核心 idea 上的演化。

---

## 2. 公式与原理

### 2.1 vanilla seq2seq 的形式化

Encoder（GRU 为例）按 timestep 推进：

$$
h_i = \mathrm{GRU}_{\text{enc}}(x_i, h_{i-1}), \quad i = 1, \dots, T
$$

context vector 取最后一步：$c = h_T \in \mathbb{R}^{d_h}$。

Decoder 以 $c$ 为初始状态：$s_0 = c$，然后

$$
s_t = \mathrm{GRU}_{\text{dec}}(y_{t-1}, s_{t-1}), \quad P(y_t \mid x, y_{<t}) = \mathrm{softmax}(W_o s_t + b_o)
$$

训练 loss 为 per-token cross-entropy，推理时 $y_{t-1}$ 用模型自己预测的 token（greedy / beam）。

**信息瓶颈的本质**——所有 source 信息必须通过宽度为 $d_h$ 的"管道" $c = h_T$ 才能到达 decoder。$d_h$ 是常数，source 长度可以很大，故对长句必然有损；而且 $h_T$ 是 RNN 末端 hidden，它本身就受梯度消失影响，对前几个 token 的记忆已经稀薄。

### 2.2 Bahdanau attention 的三步公式

Bahdanau 在 vanilla seq2seq 上做三处改造：

**(1) Encoder 改为 BiRNN**——每个位置的 hidden state 同时编码左右上下文：

$$
h_i = [\overrightarrow h_i \,;\, \overleftarrow h_i] \in \mathbb{R}^{2 d_h}
$$

下文为简洁记 $h_i \in \mathbb{R}^{d_h}$（即把 $2 d_h$ 重命名为 $d_h$）。

**(2) Attention 三步公式**——decoder 在 timestep $t$ 准备生成 $y_t$ 时（已经有 $s_{t-1}$）：

$$
\boxed{
\begin{aligned}
\text{score:}    \quad & e_{t,i} = v_a^\top \tanh(W_a s_{t-1} + U_a h_i)  \\
\text{weights:}  \quad & \alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_{j=1}^{T} \exp(e_{t,j})}  \\
\text{context:}  \quad & c_t = \sum_{i=1}^{T} \alpha_{t,i} \, h_i
\end{aligned}
}
$$

变量与维度：

- $s_{t-1} \in \mathbb{R}^{d_s}$：decoder 上一步 hidden state；
- $h_i \in \mathbb{R}^{d_h}$：encoder 第 $i$ 步 hidden state；
- $W_a \in \mathbb{R}^{d_a \times d_s}$、$U_a \in \mathbb{R}^{d_a \times d_h}$：把两边映射到统一对齐空间 $\mathbb{R}^{d_a}$；
- $v_a \in \mathbb{R}^{d_a}$：把对齐空间打成一个标量分数；
- $\alpha_{t,i} \in [0, 1]$，且 $\sum_i \alpha_{t,i} = 1$；
- $c_t \in \mathbb{R}^{d_h}$：timestep $t$ 专属的 context vector。

`score` 这一步叫 **additive (concat) attention**——把两边相加后过 $\tanh$ 再点积一个向量打分。这是一个**单隐层 MLP**（$d_a$ 通常取 $d_h$ 量级），所以 Bahdanau attention 又被称作 "MLP attention"。

**(3) Decoder 状态更新**——把 $c_t$ 拼到 decoder 输入里：

$$
s_t = \mathrm{GRU}_{\text{dec}}([y_{t-1} \,;\, c_t], s_{t-1}), \quad P(y_t) = \mathrm{softmax}(W_o[s_t \,;\, c_t \,;\, y_{t-1}] + b_o)
$$

注意 $c_t$ **既参与 hidden state 更新、又参与最终输出**——这是 Bahdanau 原 paper 的实现细节，工程上简化只用其一也能 work。

### 2.3 Luong attention：简化 + multiplicative

Luong, Pham & Manning 2015（EMNLP）一年后提出的 *Effective Approaches to Attention-based NMT* 做了三件事：

**(1) Score 函数三选一**——给出三种打分方式：

$$
\text{score}(s_t, h_i) = \begin{cases}
s_t^\top h_i & \text{(dot)}  \\
s_t^\top W_a h_i & \text{(general)}  \\
v_a^\top \tanh(W_a [s_t \,;\, h_i]) & \text{(concat / Bahdanau-like)}
\end{cases}
$$

**dot / general 是 multiplicative attention**——核心是一次内积，没有 $\tanh$ 与中间 MLP，**算得快、参数少**。这是后来 Transformer 选 dot product 的源头。

**(2) Global vs local attention**——global 看所有 encoder positions（与 Bahdanau 一致），local 只看以预测对齐位置 $p_t$ 为中心的小窗口（缓解长 source 的计算开销）。LLM 时代 local attention 几乎被遗忘，但思路在 sliding window attention（Mistral 7B / Longformer）里复活。

**(3) Decoder 用当前步而不是上一步状态**——Bahdanau 用 $s_{t-1}$ 算 attention 再算 $s_t$；Luong 用 $s_t$ 算 attention 再算"attentional hidden state" $\tilde s_t = \tanh(W_c [c_t \,;\, s_t])$ 用于预测。这是工程上的小调整，效果差异不大。

**Bahdanau vs Luong 对照表**：

| 维度 | Bahdanau (2014) | Luong (2015) |
|---|---|---|
| score 函数 | additive (MLP) | dot / general / concat（三选一） |
| 主流选择 | additive 唯一 | dot / general（更快） |
| 计算开销 | $O(T \cdot d_a)$ MLP | $O(T \cdot d)$ matmul |
| Decoder 状态 | 用 $s_{t-1}$ 算 attention | 用 $s_t$ 算 attention |
| Attention 范围 | global only | global / local |
| 与 Transformer 关系 | additive 路线终结 | **multiplicative 路线，直通 self-attention** |

**关键洞察**——Vaswani 2017 的 scaled dot-product attention 公式 $\mathrm{softmax}(QK^\top / \sqrt{d_k}) V$ 本质就是 Luong 的 dot score + softmax + value 加权，再加一个 $\sqrt{d_k}$ scaling 修复方差爆炸。**Luong 是 attention 从"RNN 辅助插件"转向"独立可扩展机制"的中间站**。

### 2.4 Beam search 简短一段

Seq2seq 推理需要从 $P(y_t \mid x, y_{<t})$ 这个概率分布里采出一条具体的序列。两种主流策略：

- **Greedy**：每步取 $\arg\max$。简单但贪心导致全局次优——前几步选错就再也回不来。
- **Beam search**：每步保留累计 log-prob 最高的 $k$ 条候选（$k$ = beam size，NMT 常用 4-10），最终选 score 最高的一条。可以加 **length penalty** $\frac{\sum \log P}{\mathrm{len}(y)^\alpha}$（$\alpha \approx 0.6$）防止偏好短句。

NMT 时代 beam search 是黄金标准，BLEU 从 greedy 的 25 能涨到 beam=5 的 28。但 **LLM 时代 beam search 逐渐被 sampling（temperature / top-$k$ / top-$p$ / nucleus）取代**，原因详见 §4 工程踩坑。

### 2.5 Attention 解决了什么、留下了什么

**解决了**：

- **信息瓶颈**——decoder 不再依赖单一 $h_T$，每步动态聚焦，长句 BLEU 显著回升；
- **长程依赖弱**——$\alpha_{t,i}$ 给 decoder step $t$ 与 encoder step $i$ 一条**直接的路径**，不必经过 $T-i$ 步 RNN 串行衰减；
- **可解释性**——$\alpha_{t,i}$ 矩阵可视化即词对齐图，机器翻译领域第一次有了"看模型在看哪"的工具。

**没解决**（这是 Transformer 的入口）：

- **Encoder 仍是 RNN**——$h_i$ 还得按 timestep 串行计算，**整体训练仍然无法并行**；
- **encoder 内部 token 之间还是靠 RNN 聚合信息**——长 source 句子中 encoder 自己的远距离表达就已经衰减了，再优秀的 attention 也补不回 encoder 的损失；
- **score 是 MLP**（additive），每个 $(t, i)$ 都要算一次 forward，比 dot product 慢得多。

### 2.6 从 Bahdanau 到 Self-attention 的演化逻辑

这是**全节最该记住的承接**。Bahdanau attention 是 **cross-attention**——decoder 看 encoder。关键飞跃问一句：

> 如果 encoder 内部各 token 之间也用 attention 互相看，是不是就完全不需要 RNN 了？

答案是**是**。这就是 self-attention 的诞生：

- **Bahdanau (2014)**：cross-attention，additive score，attention 作为 RNN 的辅助插件；
- **Luong (2015)**：cross-attention，multiplicative score（dot / general），attention 计算更高效；
- **Vaswani (2017)**：把 attention 用到 encoder / decoder **内部**（self-attention），加 $\sqrt{d_k}$ scaling 修复 dot product 在高维下的方差问题，加 multi-head 让模型学多种 attention pattern——**RNN 彻底被取代，整张图变成一次大 matmul，可以 GPU 全并行**。

最终公式 $\mathrm{Attention}(Q, K, V) = \mathrm{softmax}(QK^\top / \sqrt{d_k}) V$ 三个组件都能直接对回 Bahdanau：

- $Q$（query）= decoder state $s_{t-1}$ 的一般化；
- $K$（key）= encoder state $h_i$ 的一般化；
- $V$（value）= 用于加权求和的"内容"，Bahdanau 里 $K$ 和 $V$ 共用 $h_i$，Transformer 里把它们解耦成两个不同投影。

一句话浓缩 **Attention is All You Need**：把 attention 这个本来用来辅助 RNN 的机制，单独抽出来就足够了——RNN 这一层多余。

承接到 Module 4：4.1 self-attention 会从 $QKV$ 投影开始详讲 scaled dot-product，请把本节 2.3-2.6 的演化链条记牢，4.1 的公式不会让你陌生。

---

## 3. 最小代码示例

手写 Bahdanau attention + GRU encoder/decoder 的最小翻译模型骨架。**目标是让 forward 走通、shape 一目了然**，不追求训完真翻译（数据集太大）。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- A. Bahdanau (additive) attention：核心 10 行 ---
class BahdanauAttention(nn.Module):
    def __init__(self, d_s, d_h, d_a):
        super().__init__()
        self.W_a = nn.Linear(d_s, d_a, bias=False)   # decoder state → align space
        self.U_a = nn.Linear(d_h, d_a, bias=False)   # encoder state → align space
        self.v_a = nn.Linear(d_a, 1, bias=False)     # align space → scalar score

    def forward(self, s_prev, enc_h, mask=None):
        # s_prev: (B, d_s)        decoder 上一步 hidden
        # enc_h:  (B, T, d_h)     encoder 所有步 hidden
        # mask:   (B, T) bool     True = 有效 token, False = padding
        s_exp = self.W_a(s_prev).unsqueeze(1)        # (B, 1, d_a)
        h_exp = self.U_a(enc_h)                      # (B, T, d_a)
        e = self.v_a(torch.tanh(s_exp + h_exp)).squeeze(-1)  # (B, T)
        if mask is not None:
            e = e.masked_fill(~mask, -1e9)           # padding 处不参与 softmax
        alpha = F.softmax(e, dim=-1)                 # (B, T) 注意力权重
        c_t = (alpha.unsqueeze(-1) * enc_h).sum(dim=1)  # (B, d_h) context vector
        return c_t, alpha

# --- B. Encoder + Decoder 骨架 ---
class Encoder(nn.Module):
    def __init__(self, vocab, emb=128, hid=256):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb)
        self.gru = nn.GRU(emb, hid, batch_first=True, bidirectional=True)

    def forward(self, src):                          # src: (B, T_src)
        x = self.emb(src)                            # (B, T_src, emb)
        out, h_n = self.gru(x)                       # out: (B, T_src, 2*hid)
        return out                                   # 只用 out，下游 attention 看所有步

class Decoder(nn.Module):
    def __init__(self, vocab, emb=128, hid=256, d_a=256):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb)
        self.attn = BahdanauAttention(d_s=hid, d_h=2*hid, d_a=d_a)
        self.gru = nn.GRU(emb + 2*hid, hid, batch_first=True)   # 输入拼了 c_t
        self.out = nn.Linear(hid + 2*hid, vocab)

    def forward_step(self, y_prev, s_prev, enc_h, mask):
        # y_prev: (B,)  上一步 token；s_prev: (B, hid)；enc_h: (B, T_src, 2*hid)
        c_t, alpha = self.attn(s_prev, enc_h, mask)             # (B, 2*hid)
        y_emb = self.emb(y_prev).unsqueeze(1)                   # (B, 1, emb)
        gru_in = torch.cat([y_emb, c_t.unsqueeze(1)], dim=-1)   # (B, 1, emb + 2*hid)
        _, s_t = self.gru(gru_in, s_prev.unsqueeze(0))          # s_t: (1, B, hid)
        s_t = s_t.squeeze(0)                                    # (B, hid)
        logits = self.out(torch.cat([s_t, c_t], dim=-1))        # (B, vocab)
        return logits, s_t, alpha
```

走通示例：

```python
B, T_src, V = 4, 12, 5000
enc = Encoder(V); dec = Decoder(V)
src = torch.randint(0, V, (B, T_src))
mask = torch.ones(B, T_src, dtype=torch.bool)
enc_h = enc(src)                                  # (B, 12, 512)
s = torch.zeros(B, 256)                           # decoder 初始 hidden（实际应从 enc 投影）
y_prev = torch.zeros(B, dtype=torch.long)         # <bos>
logits, s, alpha = dec.forward_step(y_prev, s, enc_h, mask)
print(logits.shape, alpha.shape)  # torch.Size([4, 5000]) torch.Size([4, 12])
```

关键点：

- `BahdanauAttention.forward` 三行就是 §2.2 的三步公式 score → softmax → context，**与公式一一对应**——读者可以对照公式逐行核对。
- `mask.masked_fill(~mask, -1e9)` 是处理 padding 的标准做法——padding 位置设极小数让 softmax 输出 ≈ 0。3.3 节 batching/masking 会反复用这个技巧。
- Decoder 一次只走一步（`forward_step`），训练时外层套一个 `for t in range(T_tgt): logits[t] = dec.forward_step(y[t-1], ...)`，推理时 `y_prev` 用模型自己的预测。这种 step-by-step 写法和 Transformer 的并行 decoder forward 形成鲜明对比——后者一次 matmul 就把所有 timestep 算完，**这就是 self-attention 在效率上的根本优势**。
- 真实 NMT 还要：`<bos>` / `<eos>` token、teacher forcing 切换、beam search 解码、length penalty——这些扩展不影响理解 attention 的核心。

---

## 4. 工程踩坑与经验

- ❗ **Bahdanau attention 在长序列上仍然慢——score 函数是 MLP**：每个 $(t, i)$ 对都要过一次 $\tanh + W_a + U_a + v_a$，$T \cdot T'$ 个组合下计算量大。Luong 改 dot product 直接退化成一次 matmul，Transformer 用 $QK^\top$ 把所有 $(t, i)$ 一次性算完——**这是后来 attention 加速的关键起点**。如果你在 2014 年代码里仍想用 additive，序列长度尽量 < 100；超长就上 multiplicative。
- ❗ **teacher forcing 训练 vs autoregressive 推理的 distribution shift（exposure bias）**：训练时 decoder 输入是 ground-truth $y_{t-1}$，推理时是模型自己预测的 $\hat{y}_{t-1}$——一旦中间预测错一个，后续 decoder 状态就进入"训练时从未见过"的分布，错误指数累积。**这是 RL / scheduled sampling / RLHF 的根本动机之一**：用 model-generated trajectory 做训练，让训练分布逼近推理分布。Module 9 RLHF 与 Module 15 Agent RL 的多轮 rollout 思路都源于这里。
- ❗ **beam search 在低质量 model 上容易输出 "the the the the"（degeneration）**：当 model 概率分布过于"sharp 但错"时，beam search 反复挑高概率 token 形成重复——经典的 Holtzman et al. 2019 *The Curious Case of Neural Text Degeneration* 给出了完整分析。LLM 时代用 nucleus sampling（top-$p = 0.9$）+ temperature 0.7-1.0 + repetition penalty 解决，beam 几乎只在受限领域翻译里残留。
- ❗ **attention map 可视化要小心 over-interpretation**：$\alpha_{t,i}$ 看起来像"模型在 step $t$ 注意 source 第 $i$ 个词"，但 Jain & Wallace 2019 *Attention is not Explanation* 与 Wiegreffe & Pinter 2019 *Attention is not not Explanation* 的论战说明了一个关键事实——**attention 是相关性而非因果解释**。同一个预测可以用很多种 attention 分布产生，attention map 漂亮不等于模型逻辑正确，反过来也成立。可视化做 sanity check 可以，作为模型可解释性证据要谨慎。
- ❗ **用 attention 不代表自动解决长程依赖**：Bahdanau 时代的 encoder 还是 RNN，长 source 句子里 $h_i$（$i$ 较小时）已经在 BiRNN 里被远端冲淡了，再好的 attention 也只能加权"已经损失的信息"。**真正解长程依赖要等 self-attention 把 RNN 拿掉**——encoder 内部任意两个 token 直接一跳 attention 连接。
- ❗ **encoder hidden 的初始化与 decoder 不一定共享维度**：Bahdanau 用 BiRNN 后 $h_i \in \mathbb{R}^{2 d_h}$，decoder 是单向 RNN $s_t \in \mathbb{R}^{d_h}$——维度不一致是常见 bug 来源。标准做法：用一个 `nn.Linear(2*hid, hid) + tanh` 把 encoder 最后一步反向 hidden（$\overleftarrow h_1$，包含整句正向信息）投影成 decoder 的 $s_0$。代码里 `s = torch.zeros(B, hid)` 是为了演示，实际不能这么干。

---

## 5. 经典 paper

- **Sutskever, Vinyals & Le, 2014 — Sequence to Sequence Learning with Neural Networks** — vanilla seq2seq 原典，4 层 LSTM encoder-decoder + 反转 source 顺序的 trick，把 WMT'14 英法翻译做到当时最佳。读它能体会"端到端神经网络第一次打掉传统统计 MT pipeline"的范式冲击；本节 §1.1 / §2.1 的架构图与公式直接对应这篇 §3。
- **Bahdanau, Cho & Bengio, 2014 — Neural Machine Translation by Jointly Learning to Align and Translate** — attention 起点，"jointly learning to align and translate" 的副标题就是核心思想。本节 §2.2 的三步公式与 §1.3 的 mental model 直接来自这篇 §3.1，是 Transformer 真正的精神祖师爷。强烈建议精读 §3 + 长句 BLEU 实验图。
- **Luong, Pham & Manning, 2015 — Effective Approaches to Attention-based Neural Machine Translation** — global / local attention + dot / general / concat 三种 score 函数对照实验。本节 §2.3 的对照表来自这篇 §3。最关键的是它证明了 multiplicative attention 与 additive 效果相当但**显著更快**——这是 Transformer 选 dot product 的实证基础。
- **Vaswani et al., 2017 — Attention Is All You Need** — 终点站。这里只提一句：本节末尾的 $\mathrm{Attention}(Q,K,V) = \mathrm{softmax}(QK^\top/\sqrt{d_k})V$ 公式与 multi-head 设计在 4.1 详讲，本节只用它来说明"attention 机制能完全取代 RNN"这件事。

---

## 6. 自测与面试题

**Q1（公式）**：写出 Bahdanau attention 的完整三步公式（score / softmax / context），并解释 vanilla seq2seq 的"信息瓶颈"具体指什么。

<details>
<summary>Answer sketch</summary>

三步公式（变量维度首次出现要标）：

- score：$e_{t,i} = v_a^\top \tanh(W_a s_{t-1} + U_a h_i)$，$W_a \in \mathbb{R}^{d_a \times d_s}$、$U_a \in \mathbb{R}^{d_a \times d_h}$、$v_a \in \mathbb{R}^{d_a}$；
- softmax：$\alpha_{t,i} = \exp(e_{t,i}) / \sum_j \exp(e_{t,j})$，归一化到 $\sum_i \alpha_{t,i} = 1$；
- context：$c_t = \sum_i \alpha_{t,i} h_i \in \mathbb{R}^{d_h}$。

信息瓶颈的本质：

- vanilla seq2seq 把整个 source 句子压缩成 encoder 最后一步的 hidden state $h_T$（**单一固定维度 $d_h$ 向量**）；
- $d_h$ 是常数（典型 256/1024），source 长度可以任意大——长句必然有损；
- 加上 RNN 自身的梯度消失，$h_T$ 对 source 前几个 token 的记忆已经稀薄；
- 实证：vanilla seq2seq 的 BLEU 在 50 词以上长句上急剧下降，Bahdanau attention 把这条曲线拉回来。

加分：能用一句话比喻——"逼一个人把整本小说一口气背下来再复述，不如让他边看边译"。

</details>

**Q2（演化）**：从 Bahdanau additive attention → Luong multiplicative → Transformer self-attention，每一步主要变了什么？为什么这些变化都是合理的？

<details>
<summary>Answer sketch</summary>

三步演化要点：

**Bahdanau → Luong**：

- score 从 additive $v_a^\top \tanh(W_a s + U_a h)$ 变成 dot $s^\top h$ / general $s^\top W h$；
- 为什么合理：(1) MLP 计算量大，dot product 一次 matmul 把所有 $(t, i)$ 算完，**速度大幅提升**；(2) 实验上效果与 additive 持平甚至略好；(3) 与 GPU 高度并行的 matmul 基础设施天然契合。

**Luong → Transformer self-attention**：

- 把 attention 用到 encoder / decoder **内部**（不只是 cross），$Q/K/V$ 都来自同一序列的不同投影；
- 加 $\sqrt{d_k}$ scaling：$d_k$ 大时 $QK^\top$ 方差大，softmax 饱和，scaling 把方差拉回 1；
- 加 multi-head：让模型并行学多种 attention pattern；
- 为什么合理：(1) encoder 内部 attention 取代 RNN，**全图可并行 matmul**，训练吞吐质变；(2) 长程依赖任意两 token 直接一跳，不再受 RNN 串行衰减；(3) self-attention 本质是"全连接但权重 input-dependent"的灵活机制，表达能力强。

加分：能指出 Bahdanau 的 cross-attention 在 Transformer encoder-decoder 里仍然存在（4.6 节会用到），并不是被替代而是被推广——self-attention 是兄弟而非 successor。

</details>

**Q3（trade-off）**：beam search 在 NMT 时代是黄金标准，为什么 LLM 时代改用 sampling？至少 2 个原因。

<details>
<summary>Answer sketch</summary>

至少 2 个原因（多答更好）：

- **输出空间维度差异**：NMT 的 target 通常是相对受限的翻译（同一句话合理翻译数量有限，beam=5 已经能扫到大半），beam search 找最高 likelihood 序列效果好；LLM 输出空间是**所有可能的自然语言**，多样性本身有价值（创意写作、对话、reasoning），beam search 倾向"safe but boring" 的高 likelihood 输出，不符合 LLM 用户期望。
- **Likelihood ≠ quality 的失配（degeneration）**：Holtzman 2019 证明，"the the the the" 这种重复退化在高 likelihood 区域反而集中——beam search 主动找高概率序列就主动撞上 degeneration。Sampling（temperature / top-$k$ / top-$p$）通过引入随机性绕开这块"高概率 trap"。
- **RLHF 后的 LLM 概率分布已被对齐过**：模型自己输出的 token 分布已经在偏好上 calibrated，sampling 一条就够好；beam 反而可能放大 RM 不喜欢的某种系统性 bias。
- **Test-time compute 的灵活性**：sampling 天然支持 self-consistency / best-of-N / majority vote 这类 test-time scaling 技术（Module 10 详讲），beam search 给出的是 deterministic top-$k$，多样性受限。
- **工程角度**：beam search 在 batched 推理与 KV cache 管理上比 sampling 复杂得多——每条 beam 都要复制 KV cache 状态，vLLM 等推理引擎对 sampling 的优化更彻底。
- 加分：能补一句"特定场景 beam 没死"——constrained decoding 任务（结构化输出、形式语言生成、code completion）beam 仍有用，因为这些任务的搜索空间可枚举且 likelihood 与 quality 高度一致。

</details>

---

## 7. 延伸阅读

- [Jay Alammar — Visualizing A Neural Machine Translation Model](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/) — 全网最好的 seq2seq + attention 可视化教程，配 GIF 动画讲清每一步 attention 权重如何流动，本节 §1 mental model 受其影响。
- [PyTorch Tutorial — NLP From Scratch: Translation with a Sequence to Sequence Network and Attention](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html) — 官方 seq2seq + Bahdanau attention 完整可跑实现（French → English），能把本节 §3 代码骨架补成端到端训练。
- [Lilian Weng — Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/) — 从 Bahdanau / Luong 到 self-attention / Transformer 的全谱综述，本节 §2.6 演化链条的扩展版。
- [Holtzman et al. 2019 — The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) — beam search 退化问题的系统分析 + nucleus sampling 提出，§4 工程踩坑里 degeneration 与 sampling 选择的依据。
- 推荐继续读本教程的 **4.1 节《Self-attention：QKV 与 scaled dot-product》**——本节 §2.6 埋下的种子在那一节正式开花，从 Luong 的 dot product 一路推到现代 attention 公式。
