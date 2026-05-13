---
title: "附录 A：Capstone 1 — 复现 GPT-2 124M（数据 → 训练 → 评测）"
description: "把 Module 4-7 学的所有零件——decoder-only 架构（4.6）、tokenizer（3.1）、数据 pipeline（6.2）、训练 infra（7.1 DDP / 7.4 bf16）——拼成一个 end-to-end 工程：在 1 张 A100 / 1 台 8×H100 上从零复现 OpenAI 2019 年的 GPT-2 124M，目标 val loss ≈ 3.0、Hel"
---

> ⏱ 预计阅读 90 分钟 + 数天到两周训练时间 ｜ 难度 ★★★ ｜ 前置：Module 4-7 全部学完

## 一句话本节讲什么

把 Module 4-7 学的所有零件——**decoder-only 架构（4.6）、tokenizer（3.1）、数据 pipeline（6.2）、训练 infra（7.1 DDP / 7.4 bf16）**——拼成一个 end-to-end 工程：在 1 张 A100 / 1 台 8×H100 上从零复现 OpenAI 2019 年的 GPT-2 124M，目标 val loss ≈ 3.0、HellaSwag ≈ 30%（接近原版 31.6%）；本节高度参考 Andrej Karpathy 的 [`build-nanogpt`](https://github.com/karpathy/build-nanogpt) + 配套 YouTube《Let's reproduce GPT-2 (124M)》。

---

## 1. Mental model（直觉）

### 1.1 为什么要复现 GPT-2 124M

**不是"造轮子"——是把前 7 个 Module 的概念全部接上电、跑通一遍**。这一节之前你已经学过：

- 怎么写一个 decoder-only Transformer（4.6）
- 怎么处理预训数据（3.1 tokenizer + 6.2 pipeline）
- 怎么开 bf16 + DDP（7.1 / 7.4）
- 怎么算参数量、显存、FLOPs（1.5 / 6.3）

但**没有任何一个章节让你把这些组合起来在真实数据上从 random init 训到 LLM**。Capstone 1 就是这个组合。完成后你会拿到三样东西：

1. **一个能跑的 124M GPT-2 ckpt**——loss 曲线、HellaSwag 分数都能与 OpenAI 2019 年原版对齐
2. **一份从下载数据到上传 ckpt 的完整脚本**——以后训任何 size 的 LLM 改几行 config 即可
3. **真实的"训了几百小时 GPU"踩坑账**——bf16 第一次 forward 慢、grad accumulation 边界条件、HellaSwag 评测要算 conditional likelihood、wandb 日志怎么配——这些细节书里学不到，只有真训过才知道

为什么选 GPT-2 124M 而不是 1.5B 或 LLaMA-7B？三个原因：

- **算力门槛低**：8×A100 两天能跑完，单 A100 两周也能跑完（LLaMA-7B 同等规模训练要几十张 H100 一个月）
- **有公开 baseline**：OpenAI paper、HellaSwag、val loss 都有具体数字可对照
- **架构经典**：GPT-2 配方是后来所有 decoder-only LLM 的祖本，掌握 124M 等于掌握"怎么把任何 dense decoder LLM 训起来"

### 1.2 GPT-2 124M vs 4.6 的 mini-LLaMA：差在哪

4.6 节训了一个几 MB 的 mini-LLaMA 在 tinyShakespeare 上 overfit。从 mini-LLaMA 到 GPT-2 124M，**架构上只改 4 处、规模上放大 100×、数据上换成 web-scale**：

| 维度 | mini-LLaMA（4.6） | GPT-2 124M | 备注 |
|---|---|---|---|
| Norm | RMSNorm | **LayerNorm**（含 mean + bias） | GPT-2 是 2019 年模型，那时还没 RMSNorm |
| 激活 | SwiGLU（3 矩阵） | **GELU**（2 矩阵 FFN） | 2019 年 SwiGLU 还没出 |
| 位置编码 | RoPE | **学习式绝对 PE**（`nn.Embedding(1024, 768)`） | GPT-2 用 learned absolute PE，不是 RoPE |
| FFN 维度 | $\frac{8}{3} d \approx 2048$ | **$4d = 3072$** | GPT-2 标准 FFN 配比 |
| $d$ / $L$ / $h$ | 128 / 4 / 4 | **768 / 12 / 12** | scale 100× |
| context | 64 | **1024** | 16× |
| vocab | 256 | **50257**（GPT-2 BPE） | 真实 BPE，不是 byte-level |
| 数据 | 1MB tinyShakespeare | **10B token FineWeb-Edu** | 10000× |
| 训练量 | 200 step | **19073 step**（10B / 524k token-per-step） | 100× |
| GPU | CPU 也行 | **8×A100 ~2 天 / 1×A100 ~2 周** | 4-5 个量级 |

**用一行话总结**：架构基本不变，只是把 norm 换 LN、激活换 GELU、位置编码换 learned absolute——剩下全靠 scale。

### 1.3 完整 pipeline 鸟瞰

```
┌─────────────────────────────────────────────────────────────┐
│ 阶段 0: 准备 (1 小时)                                        │
│   - 装 PyTorch 2.x + tiktoken + datasets + wandb            │
│   - 申请 HF token、wandb token                              │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 阶段 1: 数据 (4-12 小时, IO 主导)                            │
│   - 下载 FineWeb-Edu sample-10BT (~24 GB parquet)            │
│   - tiktoken `gpt2` 编码 → 100M token 一 shard               │
│   - 输出 ~100 个 .bin shard（每个 ~200 MB）                  │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 阶段 2: 训练 (8×A100 ~50 小时)                                │
│   - GPT-2 124M from scratch                                  │
│   - bf16 + DDP + grad accum (524k token / step)              │
│   - lr 6e-4, warmup 715, cosine 19073                        │
│   - 每 1000 步: val loss + HellaSwag + sample text + ckpt   │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 阶段 3: 评测 + 上传 (1-2 小时)                               │
│   - HellaSwag 全集评测 (~10k 题)                             │
│   - val loss / perplexity 对比                               │
│   - 推上 HF Hub                                              │
└─────────────────────────────────────────────────────────────┘
```

总计 ~70 小时人 + GPU time（8×A100）。这是一个**周末项目级**的 LLM 复现——不是"训不起 LLM 的科研团队的特权"。

---

## 2. 公式与原理

### 2.1 训练量计算（Chinchilla 视角）

GPT-2 124M 原版用 ~40 GB 文本训练（约 10B token）。从 Chinchilla scaling law 视角看：

$$D_{\text{Chinchilla}} \approx 20 \cdot N$$

124M × 20 = **2.5B token** 是 Chinchilla optimal——所以 GPT-2 124M 训 10B token 是**轻度 over-trained**（4× Chinchilla）。这正是 LLaMA / Qwen 的小模型也常用的策略：小模型多训点 token，推理 cost 更划算（参考 6.3 over-training 讨论）。

### 2.2 batch size 与 token-per-step

GPT-3 paper Table 2.1 标 GPT-2 small (124M) 用 **batch size 0.5M token**——具体怎么拆：

$$\text{tokens per step} = B_{\text{micro}} \times \text{seqlen} \times N_{\text{GPU}} \times N_{\text{accum}}$$

8×A100 80GB 配方：

| 项 | 值 | 说明 |
|---|---|---|
| `B_micro` (per GPU) | 16 | bf16 下 124M + seqlen=1024 + adam state，A100 80GB 还能撑 |
| `seqlen` | 1024 | GPT-2 原版 context |
| `N_GPU` | 8 | 单机 8 卡 |
| `N_accum` | 4 | grad accumulation |
| **token / step** | **524288 = 0.5M** | 16 × 1024 × 8 × 4 |

总训练 step：

$$N_{\text{step}} = \frac{10^{10}}{524288} \approx 19073$$

**单卡复现配方**（A100 80GB / H100，无 DDP）：

| 项 | 值 |
|---|---|
| `B_micro` | 16 |
| `seqlen` | 1024 |
| `N_GPU` | 1 |
| `N_accum` | **32**（保持 token / step = 524k） |
| 训练时间 | ~14 天 |

token per step 必须保持 524k——这是 GPT-3 paper 标定的 batch size，改小会影响收敛。

### 2.3 学习率配方

GPT-3 paper Table 2.1：GPT-2 small 用 **lr = 6e-4 + cosine to 10% + warmup 375M token**。换算到 step：

- warmup token = 375M → warmup step = $375 \times 10^6 / 524288 \approx 715$
- decay 到 final lr 的 10%（即 6e-5）

```
lr ↑
   │     ┌──────╲
   │    ╱        ╲___________
   │   ╱                     ╲
   │  ╱                       ╲___
   │ ╱                            ╲___
   │╱                                 ╲
6e-5──────────────────────────────────────
   │
   │   715           ~19073
   └─────────────────────────────────→ step
       warmup    cosine decay
```

`lr` 公式（与 1.2 节 LR schedule 一致）：

$$\eta(t) = \begin{cases}
\eta_{\max} \cdot \frac{t+1}{T_{\text{warm}}} & \text{if } t < T_{\text{warm}} \\
\eta_{\min} + 0.5 (\eta_{\max} - \eta_{\min}) \left(1 + \cos\left(\pi \frac{t - T_{\text{warm}}}{T_{\text{total}} - T_{\text{warm}}}\right)\right) & \text{if } T_{\text{warm}} \le t < T_{\text{total}} \\
\eta_{\min} & \text{otherwise}
\end{cases}$$

其中 $\eta_{\max} = 6 \times 10^{-4}$、$\eta_{\min} = 6 \times 10^{-5}$、$T_{\text{warm}} = 715$、$T_{\text{total}} = 19073$。

其他超参（与 GPT-3 paper Table 2.1 + nanoGPT 对齐）：

- AdamW，$\beta_1 = 0.9$、$\beta_2 = 0.95$、$\epsilon = 10^{-8}$、weight decay $= 0.1$
- gradient clip max norm = 1.0
- weight decay 只作用在 2D 参数（weight matrix），1D 参数（bias / norm gain）不衰减
- 初始化：Linear / Embedding 用 $\mathcal{N}(0, 0.02)$，残差路径上的 proj 层 std 缩 $\frac{1}{\sqrt{2L}}$（GPT-2 原版 trick，控制 residual stream 数值不爆炸）

### 2.4 throughput 估算（与硬件实测对账）

每 step 的 FLOPs（与 6.3 scaling law 公式一致）：

$$F_{\text{step}} \approx 6 \cdot N \cdot D_{\text{tok-per-step}} = 6 \times 1.24 \times 10^8 \times 5.24 \times 10^5 \approx 3.9 \times 10^{14} \text{ FLOPs}$$

A100 bf16 峰值 ≈ 312 TFLOP/s（实测 MFU ≈ 50% → ~160 TFLOP/s），单卡每步耗时：

$$t_{\text{step}}^{1\text{GPU}} = \frac{3.9 \times 10^{14}}{1.6 \times 10^{14}} \approx 2.4 \text{ s}$$

8 卡 DDP 理想 8× 加速（实测 ~6×，DDP 有通信开销）：

$$t_{\text{step}}^{8\text{GPU}} \approx 0.4 \text{ s}$$

总训练时间：

- 单 A100：$2.4 \times 19073 \approx 12.7$ 小时？这是**理论极限**——实际 MFU 只 35-45%、加上 dataloader 等开销，实测约 **2 周**
- 8×A100：$0.4 \times 19073 \approx 2.1$ 小时？**同样理论极限**——实际 MFU + DDP overhead + eval / ckpt 时间，约 **48 小时**
- 8×H100：H100 bf16 峰值 ~990 TFLOP/s（A100 的 3×），约 **20-24 小时**

这是 Karpathy 在 build-nanogpt 视频里报的实测时间——他用的就是 8×H100 跑了一晚上。

### 2.5 HellaSwag 评测原理（必须懂）

HellaSwag 是 commonsense reasoning benchmark，每题 1 个 context + 4 个 candidate ending，模型选最合理的一个。**评测方式不是 generation，是 conditional likelihood**：

对题目 $(c, [e_1, e_2, e_3, e_4])$，每个 candidate 的得分是：

$$\text{score}(e_i) = \frac{1}{|e_i|} \sum_{t \in e_i} \log p(t \mid c, e_i^{<t})$$

即 **candidate token 的平均 log-likelihood**（context token 不算 loss）。模型选 $\arg\max_i \text{score}(e_i)$，与标签对比算 accuracy。

为什么要平均（除以 candidate 长度）？因为 4 个 candidate 长度不一，**没归一化的 sum log-prob 会偏向短 candidate**（短 sequence sum 更小负数，prob 更大）。HellaSwag 官方评测脚本默认按 token 数归一化。

GPT-2 124M 原版 HellaSwag = **31.6%**（OpenAI 2019 paper）。本 capstone 训完后通常能跑到 **29-32%** 之间，是合理复现区间。注意 4-way random baseline = 25%，所以 30% 不是"什么都没学会"——是"略好于瞎猜"，但这就是 124M 在常识推理上的真实能力上限。

---

## 3. 最小代码示例

### 3.1 完整 GPT-2 124M Model（< 150 行 self-contained）

把 4.6 的 mini-LLaMA 改造成 GPT-2 风格——核心改动：RMSNorm → LayerNorm、SwiGLU → GELU、RoPE → learned absolute PE。

```python
# gpt2.py — GPT-2 124M 完整实现，self-contained
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class GPT2Config:
    block_size: int = 1024          # context length
    vocab_size: int = 50304         # GPT-2 BPE 是 50257，pad 到 64 倍数 50304 加速
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768               # d_model = 12 * 64


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPT2Config):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        # 一次性产 Q, K, V（合三 Linear，省 launch overhead）
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.c_proj.GPT2_RESIDUAL_PROJ = 1   # 标记，初始化时缩 std
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)                                              # (B, T, 3C)
        q, k, v = qkv.split(self.n_embd, dim=2)
        # reshape 到 (B, h, T, d_k)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # PyTorch 2.x SDPA 自动选 FlashAttention，处理 causal mask
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: GPT2Config):
        super().__init__()
        # GPT-2 标准 FFN: 2 矩阵, d_ff = 4d, GELU
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.c_proj.GPT2_RESIDUAL_PROJ = 1
        # GPT-2 用 'tanh' 近似 GELU（与 OpenAI 原版对齐，比 exact GELU 略快）
        self.act = nn.GELU(approximate='tanh')

    def forward(self, x):
        return self.c_proj(self.act(self.c_fc(x)))


class Block(nn.Module):
    """Pre-LN block: x = x + Attn(LN(x)); x = x + MLP(LN(x))"""
    def __init__(self, cfg: GPT2Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2(nn.Module):
    def __init__(self, cfg: GPT2Config):
        super().__init__()
        self.cfg = cfg
        # 词嵌入 + 学习式位置嵌入（GPT-2 风格，不是 RoPE）
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)              # final LayerNorm（Pre-LN 必加）
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # Weight tying: lm_head 与 wte 共享权重（节省 V·d 参数）
        self.lm_head.weight = self.wte.weight
        # 初始化
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            std = 0.02
            # GPT-2 trick: residual path 上的 proj 缩 std 1/sqrt(2L)
            if hasattr(m, 'GPT2_RESIDUAL_PROJ'):
                std *= (2 * self.cfg.n_layer) ** -0.5
            nn.init.normal_(m.weight, mean=0.0, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)                  # (B, T, d)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                           # (B, T, V)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1),
                ignore_index=-1,
            )
        return logits, loss

    def configure_optimizers(self, weight_decay, lr, betas, device):
        """权重衰减只作用在 2D 参数（weight matrix），1D 参数（bias / norm）不衰减。"""
        decay_params = [p for p in self.parameters() if p.dim() >= 2 and p.requires_grad]
        nodecay_params = [p for p in self.parameters() if p.dim() < 2 and p.requires_grad]
        groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        # PyTorch 2.x 的 fused AdamW，CUDA 上加速 ~2×
        use_fused = (device == 'cuda')
        return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=1e-8, fused=use_fused)
```

总行数约 110 行（含空行注释），与 4.6 mini-LLaMA 高度对应——只改了 `RMSNorm → LayerNorm`、`SwiGLU → MLP+GELU`、`RoPE → wpe`、加了 `GPT2_RESIDUAL_PROJ` 初始化 trick + `configure_optimizers` 区分 weight decay。

**vocab_size 从 50257 pad 到 50304** 是 nanoGPT 的实战 trick：50304 = 64 × 786 是 64 倍数，CUDA tensor core 在最后一维是 64 倍数时显著加速；50257 是质数，会让 GEMM kernel 走不到 fast path。代价是多 47 个无用 vocab，浪费 0.04% 参数——绝对划算。

### 3.2 数据准备脚本（< 50 行）

下载 FineWeb-Edu sample-10BT、用 tiktoken 编码、写成 .bin shard。

```python
# fineweb.py — 下载 + tokenize + 写 shard
import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

DATA_CACHE_DIR = "edu_fineweb10B"
SHARD_SIZE = int(1e8)               # 100M token / shard, 共 ~100 shard
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# 加载 FineWeb-Edu 10BT subset (~24 GB parquet, HF Hub 自动 download + cache)
ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train")

enc = tiktoken.get_encoding("gpt2")
EOT = enc._special_tokens['<|endoftext|>']                              # 50256

def tokenize(doc):
    """单文档 → uint16 token array（前缀 EOT 作为 doc 分隔）"""
    tokens = [EOT]
    tokens.extend(enc.encode_ordinary(doc["text"]))
    arr = np.array(tokens, dtype=np.uint16)                             # vocab < 65536, uint16 够用
    assert (arr < 2**16).all(), "vocab > uint16, 改 uint32"
    return arr

def write_shard(filename, arr):
    np.save(filename, arr)                                              # 实际用 .npy 更安全；想用 .bin 直接 arr.tofile

# 多进程 tokenize（单进程是数据准备瓶颈！）
nproc = max(1, os.cpu_count() // 2)
with mp.Pool(nproc) as pool:
    shard_index = 0
    buf = np.empty(SHARD_SIZE, dtype=np.uint16)
    buf_count = 0
    progress = tqdm(total=SHARD_SIZE, unit="tok", desc=f"Shard {shard_index}")
    for tokens in pool.imap(tokenize, ds, chunksize=16):
        if buf_count + len(tokens) < SHARD_SIZE:
            buf[buf_count:buf_count + len(tokens)] = tokens
            buf_count += len(tokens)
            progress.update(len(tokens))
        else:
            split = "val" if shard_index == 0 else "train"
            fname = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
            # 把当前文档剩下的部分塞进 buf 最后空隙
            remainder = SHARD_SIZE - buf_count
            buf[buf_count:] = tokens[:remainder]
            write_shard(fname, buf)
            shard_index += 1
            progress = tqdm(total=SHARD_SIZE, unit="tok", desc=f"Shard {shard_index}")
            # 把溢出的塞进新 buf
            buf[0:len(tokens) - remainder] = tokens[remainder:]
            buf_count = len(tokens) - remainder

    if buf_count > 0:
        fname = os.path.join(DATA_CACHE_DIR, f"edufineweb_train_{shard_index:06d}")
        write_shard(fname, buf[:buf_count])
```

跑一次约 4-12 小时（取决于 IO 与 CPU 数），输出 100 个 shard，shard 0 是 val（约 100M token），其余 99 个是 train（共约 9.9B token）。

**为什么用 multiprocessing？** tiktoken 是 Rust 写的纯 CPU bound op，单进程 ~50k token/s——10B token 单线程要 60 小时。32 核机器开 16 进程后 ~800k token/s，4 小时跑完。

**为什么用 uint16？** GPT-2 vocab = 50257 < $2^{16} = 65536$，存 uint16 比 int32 省 50% 磁盘 + IO。整个 10B token 占 20 GB（uint16）vs 40 GB（int32），训练时 dataloader IO 也减半。

### 3.3 训练 script（< 100 行）

完整 DDP + bf16 + grad accum + monitor + ckpt：

```python
# train.py — GPT-2 124M 训练 script，支持单卡 / DDP
# 启动方式:
#   单卡: python train.py
#   8 卡: torchrun --standalone --nproc_per_node=8 train.py
import os, math, time, glob
import numpy as np
import torch
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from gpt2 import GPT2, GPT2Config

# === DDP setup ===
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend='nccl')
    rank, local_rank, world = int(os.environ['RANK']), int(os.environ['LOCAL_RANK']), int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(local_rank); device = f'cuda:{local_rank}'
    master = (rank == 0)
else:
    rank, world, device, master = 0, 1, 'cuda', True

# === 超参（GPT-3 paper Table 2.1） ===
total_batch = 524288                               # 0.5M token / step
B, T = 16, 1024                                    # micro batch
assert total_batch % (B * T * world) == 0
accum = total_batch // (B * T * world)             # 单卡 32, 8 卡 4
max_lr, min_lr, warmup, max_steps = 6e-4, 6e-5, 715, 19073

# === Data loader（按 step 顺序读 shard） ===
class ShardLoader:
    def __init__(self, B, T, rank, world, split):
        self.B, self.T, self.rank, self.world = B, T, rank, world
        self.shards = sorted(glob.glob(f"edu_fineweb10B/edufineweb_{split}_*.npy"))
        assert len(self.shards), f"no {split} shard"
        self.reset()
    def reset(self):
        self.cur_shard = 0
        self.tokens = torch.from_numpy(np.load(self.shards[0]).astype(np.int32))
        self.pos = self.B * self.T * self.rank
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.pos : self.pos + B*T + 1]
        x, y = buf[:-1].view(B, T), buf[1:].view(B, T)
        self.pos += B * T * self.world
        if self.pos + B*T*self.world + 1 >= len(self.tokens):
            self.cur_shard = (self.cur_shard + 1) % len(self.shards)
            self.tokens = torch.from_numpy(np.load(self.shards[self.cur_shard]).astype(np.int32))
            self.pos = B * T * self.rank
        return x, y

train_loader = ShardLoader(B, T, rank, world, "train")
val_loader = ShardLoader(B, T, rank, world, "val")

# === Model ===
torch.set_float32_matmul_precision('high')         # TF32 matmul，A100 上 ~2× 加速
model = GPT2(GPT2Config()).to(device)
model = torch.compile(model)                       # PyTorch 2.x compile，吞吐 +30-50%
if ddp: model = DDP(model, device_ids=[local_rank])
raw_model = model.module if ddp else model
optim = raw_model.configure_optimizers(0.1, max_lr, (0.9, 0.95), 'cuda')

def get_lr(it):
    if it < warmup: return max_lr * (it + 1) / warmup
    if it >= max_steps: return min_lr
    decay = (it - warmup) / (max_steps - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * decay))

# === Train loop ===
for step in range(max_steps):
    t0 = time.time()
    # 周期性 eval
    if step % 250 == 0:
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for _ in range(20):
                x, y = val_loader.next_batch(); x, y = x.to(device), y.to(device)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    _, loss = model(x, y)
                val_loss += loss.item() / 20
        if ddp:
            val_t = torch.tensor(val_loss, device=device); dist.all_reduce(val_t, op=dist.ReduceOp.AVG)
            val_loss = val_t.item()
        if master: print(f"step {step:5d} | val_loss {val_loss:.4f}")
        model.train()

    # 训练 step（grad accumulation）
    optim.zero_grad(set_to_none=True)
    loss_acc = 0.0
    for micro in range(accum):
        x, y = train_loader.next_batch(); x, y = x.to(device), y.to(device)
        if ddp: model.require_backward_grad_sync = (micro == accum - 1)   # 只在最后一个 micro 同步
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            _, loss = model(x, y)
        loss = loss / accum                                                # 等价于 mean over accum
        loss_acc += loss.detach()
        loss.backward()
    if ddp: dist.all_reduce(loss_acc, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)
    for g in optim.param_groups: g['lr'] = lr
    optim.step()
    torch.cuda.synchronize()
    dt = time.time() - t0
    tok_per_sec = total_batch / dt
    if master and step % 10 == 0:
        print(f"step {step:5d} | loss {loss_acc.item():.4f} | lr {lr:.2e} | norm {norm:.2f} | "
              f"dt {dt*1000:.0f}ms | tok/s {tok_per_sec:.0f}")

    # 保存 ckpt
    if master and step > 0 and (step % 5000 == 0 or step == max_steps - 1):
        ckpt = {'model': raw_model.state_dict(), 'optim': optim.state_dict(),
                'step': step, 'config': raw_model.cfg}
        torch.save(ckpt, f"ckpt_{step:06d}.pt")

if ddp: destroy_process_group()
```

启动命令：

```bash
# 单卡（A100 80GB / H100，约 14 天）
python train.py

# 单机 8 卡 DDP（约 48 小时）
torchrun --standalone --nproc_per_node=8 train.py
```

### 3.4 HellaSwag 评测脚本（< 50 行）

每 1000 步评测一次（也可独立跑）。核心是 conditional likelihood——4 个 candidate 算 log p(ending | context)，归一化后取 argmax。

```python
# eval_hellaswag.py — HellaSwag 4-way conditional likelihood eval
import torch
import torch.nn.functional as F
import tiktoken
from datasets import load_dataset

enc = tiktoken.get_encoding("gpt2")
ds = load_dataset("Rowan/hellaswag", split="validation")    # ~10k examples

@torch.no_grad()
def eval_hellaswag(model, device, max_examples=None):
    model.eval()
    correct, total = 0, 0
    for i, ex in enumerate(ds):
        if max_examples and i >= max_examples: break
        ctx = ex["ctx"]                                       # context string
        endings = ex["endings"]                               # 4 candidate strings
        label = int(ex["label"])                              # 0/1/2/3

        ctx_tokens = enc.encode_ordinary(ctx)
        scores = []
        for end in endings:
            end_tokens = enc.encode_ordinary(" " + end)       # 加空格匹配 BPE 边界
            full = torch.tensor([ctx_tokens + end_tokens], device=device)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits, _ = model(full)
            # 只算 ending token 的 NLL
            shift_logits = logits[0, len(ctx_tokens)-1 : -1, :]    # 预测 ending 的 logits
            shift_targets = torch.tensor(end_tokens, device=device)
            nll = F.cross_entropy(shift_logits, shift_targets, reduction='sum')
            avg_nll = nll.item() / len(end_tokens)            # 长度归一化（关键！）
            scores.append(-avg_nll)                           # 高分 = 高 likelihood
        pred = max(range(4), key=lambda j: scores[j])
        correct += int(pred == label); total += 1
    acc = correct / total
    print(f"HellaSwag acc: {acc:.4f} ({correct}/{total})")
    return acc

# 用法：
# from gpt2 import GPT2, GPT2Config
# m = GPT2(GPT2Config()).cuda()
# m.load_state_dict(torch.load("ckpt_019072.pt")['model'])
# eval_hellaswag(m, 'cuda')
```

跑一次全集 10k 题约 5-15 分钟（取决于 batch size），训练中可以只跑 1000 题快速估算（差 ±1.5%）。

### 3.5 训练曲线对照（预期值）

训完后 wandb 曲线大致：

```
val_loss
  ▲
5.5├──╲
    │  ╲
    │   ╲___
4.0│       ╲___
    │           ╲_____
    │                 ╲_____
3.0│                       ╲___________   ← 终点 ~3.0
    │
    └────┬────┬────┬────┬────┬────┬──→ step
         1k   3k   6k  10k  15k  19k

HellaSwag acc
  ▲
0.30│                                  ╱─── 终点 ~30%
    │                            ____╱
0.27│                       ____╱
    │                  ____╱
0.25│  ─random baseline─
    └────┬────┬────┬────┬────┬────┬──→ step
         1k   3k   6k  10k  15k  19k
```

如果你的 val_loss 卡在 4.0+ 不下、或 HellaSwag 一直 25% 不动——回去查 §4 踩坑列表。

---

## 4. 工程踩坑与经验

- ❗ **FineWeb-Edu 下载慢——用 hf-hub 加速器**。HF Hub 默认通过 `huggingface_hub.snapshot_download`，国内 / 网络抖动情况下 24 GB 数据可能要十几小时。**对策**：(1) 设 `HF_ENDPOINT=https://hf-mirror.com`（国内镜像）；(2) 装 [`hfd`](https://github.com/padeoe/hf-mirror-site) 或用 `aria2c -x 16` 并发下载；(3) 用 `cache_dir` 指向大盘并 resume。FineWeb-Edu 10BT 是 24 GB parquet，不要保存在 100GB 系统盘。

- ❗ **tiktoken tokenize 必须多进程，否则 10B token 单进程要 60 小时**。tiktoken 是 Rust 写的纯 CPU bound op，单进程 50-80k token/s，10B / 60k ≈ 46 小时。**对策**：用 `multiprocessing.Pool(os.cpu_count() // 2)` + `imap(chunksize=16)`，32 核机器吞吐能到 800k token/s，4 小时跑完。注意 chunksize 太小（=1）会让 worker 间通信 overhead 主导，吞吐更差；太大（>64）会让 progress bar 卡顿。

- ❗ **DDP launch 必须用 `torchrun --nproc_per_node=8`，不是 `python -m torch.distributed.launch` 也不是 `python train.py`**。`torch.distributed.launch` 已经 deprecated；直接 `python train.py` 会跳过 DDP setup（`RANK` 环境变量没设），8 张卡的训练只用 1 张。**症状**：你以为跑 8 卡，结果 throughput 只有单卡的水平。**check**：训练 log 里第一行打印 `world={world_size}` 来确认。

- ❗ **bf16 第一次 forward 比之后慢 5-10×（CUDA kernel JIT 编译 + autocast cache 预热）**。新手 看到第 0 步耗时 30s 以为 bug——其实第 100 步开始稳定在 400ms，第 0-10 步是 warm up。**对策**：训练 step 计时从 step 10 之后才打 throughput；step 0 单独标 "warm up"，不要进 wandb。`torch.compile()` 第一步还会再多 30-60s 的 trace + compile 时间，是正常的。

- ❗ **grad accumulation 的 loss 必须 `loss / accum` 再 backward**。否则等价于把 N 个 micro batch 的 loss 加起来再求梯度——梯度大 N 倍，effective lr 也大 N 倍，训练 diverge。**正确**：每个 micro `loss = loss / accum; loss.backward()`，累计后再 `optimizer.step()`。等价于 mean over accum micros。本节 §3.3 已写。

- ❗ **DDP + grad accumulation 时，前 N-1 个 micro batch 不要 all-reduce**。默认 DDP 每次 backward 自动 all-reduce 梯度，但前 N-1 个 micro 梯度还要继续累加，提前 all-reduce 是浪费——而且 reduce 的是不完整梯度，没意义。**对策**：用 `model.require_backward_grad_sync = (micro == accum - 1)`（本节 §3.3 已加）或包 `model.no_sync()` context manager。漏这一步通信 overhead 让 8 卡 DDP 几乎没加速。

- ❗ **HellaSwag 评测必须做 length normalization——按 token 数除一下**。4 个 candidate ending 长度不一，直接 sum log-prob 会偏向短 candidate（短 sequence sum 更小负数）。**正确**：`avg_nll = sum_nll / len(end_tokens)`，再选 max。本节 §3.4 已写。漏这一步 HellaSwag acc 会被报得偏低 5-8%。

- ❗ **HellaSwag 的 BPE 边界要加 leading space**：`enc.encode(" " + ending)` 而不是 `enc.encode(ending)`。GPT-2 BPE 把空格当 token 一部分（"hello" 与 " hello" 是不同 token），不加空格 ending 与 context 接缝处会出现 BPE 不连续，conditional likelihood 失真。OpenAI 官方 HellaSwag 评测、lm-eval-harness、Karpathy build-nanogpt 全部加 leading space，这是公认惯例。

- ❗ **ckpt 必须保存 model + optim + scheduler state，单存 model 无法 resume**。AdamW 的 m / v 状态承载了 "之前训了多少步、梯度累积到什么样的二阶动量"——单 load model state，optim state 重置成 0，第一步等价于 cold start，loss 立刻飙升然后慢慢回落。**正确**：`{'model': ..., 'optim': ..., 'step': ..., 'rng': torch.get_rng_state()}`，resume 时全部 load。本节 §3.3 已包含 model + optim + step。

- ❗ **dataloader IO 是 throughput 的隐形天花板——一定开 `num_workers > 0` + `pin_memory=True`**。本节 §3.3 用的是简化的同步 ShardLoader（直接 `np.load`），单 shard 200MB 加载需 0.5-2s，每 ~3000 步换一次 shard，会有周期性卡顿。**改进**：用 `torch.utils.data.DataLoader(num_workers=4, pin_memory=True, prefetch_factor=2)`，IO 与 compute overlap，throughput 提 10-30%。Karpathy 的 build-nanogpt repo 给了一个 `DistributedSampler` 的工业版。

- ❗ **loss 在 step 1000-3000 突然 spike 是常见现象，多数自愈**。LLM 训练前期不稳定，会偶尔出现 loss 跳升（5.0 → 7.0），grad_norm 也会 spike 到 5.0+。原因可能是某个 batch 的极端 token 分布、bf16 数值噪声、或 attention 暂时 collapse。**对策**：(1) 加 grad clip max_norm=1.0（必加）；(2) 监测但不要恐慌——如果 50-100 步内自己 recover 到 trend line 就没事；(3) 如果 loss 一直不回来，回退最近 ckpt + 跳过那段数据 + 减小 lr 一档继续。

- ❗ **wandb / tensorboard 必须开，否则训完两天不知道发生了什么**。LLM 训练 19000 step 跑两天，没监控你只能盯 stdout——一旦中途崩了或者 loss 早就 diverge 了你都不知道。**最低标准**：log loss / lr / grad_norm / tokens_per_sec / val_loss / HellaSwag_acc 共 6 项；高级一点加 attention entropy / activation norm / weight L2。本节 §3.3 没集成 wandb（保持代码简洁），实战务必加上 `wandb.init(project="reproduce-gpt2", name=run_name)` + `wandb.log(...)`。

- ❗ **LayerNorm bias 和 weight decay 不要混**。本节 `configure_optimizers` 区分了 2D（decay）/ 1D（no decay）参数——LayerNorm gain、bias 都是 1D，不应该 weight decay。GPT-3 paper、nanoGPT 都这么做。如果你统一对所有参数 weight decay，LayerNorm gain 会被慢慢拉向 0，模型不收敛或 loss 偏高 ~0.05。

- ❗ **`torch.compile()` 第一步会卡 30-60 秒**：PyTorch 2.x compile 是 trace + 编译 fused kernel，第一次 forward / backward 各要编译一次。新手以为代码挂了。**对策**：(1) 看 stderr 有没有 `Inductor compilation` 这种 log；(2) 给 `torch.compile(model, mode='reduce-overhead')` 试试更激进的编译模式；(3) 第一次跑加上 `TORCH_LOGS=+dynamo` 看编译过程。compile 后单步加速 30-50%，绝对值得等。

- ❗ **NCCL 启动错误的常见原因**：(1) 没设 `NCCL_DEBUG=INFO` 看不到具体错；(2) 多机时 `MASTER_ADDR` / `MASTER_PORT` 没对齐；(3) 防火墙挡了 NCCL 端口；(4) 多机网络是 IPoIB 但环境变量没开 `NCCL_IB_DISABLE=0`；(5) docker 容器没 share IPC namespace（需要 `--ipc=host`）。**调试 first principle**：单机 8 卡先跑通，再上多机。

---

## 5. 经典 paper

- **Radford et al., 2019 — Language Models are Unsupervised Multitask Learners (GPT-2)** — 原始 paper，定义了 124M / 355M / 774M / 1.5B 四档模型规格。Table 4 给出 GPT-2 124M 的 HellaSwag = 31.6%、LAMBADA ppl 35.13、WikiText 29.41，是本 capstone 的 baseline。读 §2 model + §3 zero-shot 实验即可。

- **Brown et al., 2020 — Language Models are Few-Shot Learners (GPT-3)** — 训练超参的标准答案。Table 2.1 标了 GPT-3 small (125M) 的 lr=6e-4、batch=0.5M token、warmup=375M token——本 capstone 的训练配方完全照抄。读 §2 Model + Table 2.1 即可。GPT-3 paper 也是后来 scaling law 讨论的起点（虽然 Chinchilla 才完整提出）。

- **Hoffmann et al., 2022 — Training Compute-Optimal Large Language Models (Chinchilla)** — 解释为什么 124M 训 10B token 是 4× over-training（Chinchilla optimal 只需 2.5B token），以及为什么小模型多训 token 仍然合理。读 §3 + Table 3 的 scaling law 公式。

- **Andrej Karpathy — build-nanogpt GitHub + YouTube《Let's reproduce GPT-2 (124M)》** — 不是 paper 但价值堪比。本节代码风格、超参、踩坑列表高度参考这两份材料。视频 4 小时讲解从数据下载到 8×H100 训练的全流程，是 LLM 复现教程里**最系统、最坦诚**的——Karpathy 把每个工程决策的"为什么"都讲了。强烈建议第一次复现时一边看视频一边照着抄代码。

- **Penedo et al., 2024 — The FineWeb Datasets** — FineWeb-Edu 数据集论文，本 capstone 用的就是它的 sample-10BT 子集。读 §3 + §5（FineWeb-Edu 的 LLM-as-judge 流程）。Take-away：理解为什么用 FineWeb-Edu 训出来的 GPT-2 124M 比用 OpenAI 原版 WebText 训的还略强（FineWeb-Edu quality 更高，详见 6.2）。

---

## 6. 自测与面试题

**Q1（架构）**：从 Module 4.6 的 mini-LLaMA 到 GPT-2 124M，需要改哪些组件？给出改动清单 + 每处改动的"为什么"。

<details>
<summary>Answer sketch</summary>

至少 4 处架构改动 + 5 处规模改动：

**架构改动**：

| 组件 | mini-LLaMA | GPT-2 124M | 为什么 |
|---|---|---|---|
| Norm | RMSNorm（无 bias） | **LayerNorm**（含 mean + bias） | GPT-2 是 2019 年模型，RMSNorm 还没普及 |
| 激活 / FFN | SwiGLU（3 矩阵，d_ff = 8/3 d） | **GELU**（2 矩阵，d_ff = 4d） | GPT-2 用 OpenAI 自己 fork 的 GELU tanh 近似 |
| 位置编码 | RoPE | **学习式绝对 PE**（`nn.Embedding(1024, 768)`） | GPT-2 用 learned absolute PE |
| 残差初始化 | 标准 N(0, 0.02) | **额外缩 1/sqrt(2L)** | GPT-2 paper §2 提到的 trick，控制 residual stream 数值不爆炸 |

**规模改动**（参数 + 数据 + step）：

| 维度 | mini-LLaMA | GPT-2 124M | 倍数 |
|---|---|---|---|
| d_model | 128 | 768 | 6× |
| n_layer | 4 | 12 | 3× |
| n_head | 4 | 12 | 3× |
| context | 64 | 1024 | 16× |
| vocab | 256 | 50257（pad 50304） | 200× |
| 总参数 | 0.8M | 124M | 150× |
| 数据 | 1MB | 10B token | 10000× |
| step | 200 | 19073 | 100× |

加分：

- 提到 vocab pad 到 50304 是为了让 GEMM 走 fast path（CUDA tensor core 在 64 倍数下加速）
- 提到 weight tying 不变（GPT-2 也 tie）
- 提到 LayerNorm bias、Linear bias 在 GPT-2 都开启（bias=True），与 LLaMA 全 False 不同
- 提到 GELU 用 `approximate='tanh'` 与 OpenAI 原版对齐（exact GELU 略慢、数值微差）

</details>

**Q2（计算）**：1×A100 80GB 训 GPT-2 124M 需要多少时间？算 throughput（FLOPs / token / step）+ 实际墙钟时间。

<details>
<summary>Answer sketch</summary>

**Step 1：算 FLOPs / step**

每 token 6N FLOPs（N=124M），token / step = 524288：

$$F_{\text{step}} = 6 \times 1.24 \times 10^8 \times 5.24 \times 10^5 \approx 3.9 \times 10^{14} \text{ FLOPs}$$

**Step 2：A100 bf16 实测吞吐**

A100 bf16 峰值 312 TFLOP/s，实际 MFU（Model FLOPs Utilization）约 35-45%（dataloader / DDP / Python overhead 拖累）。取 40%：

$$\text{effective} = 312 \times 0.4 \approx 125 \text{ TFLOP/s} = 1.25 \times 10^{14} \text{ FLOP/s}$$

**Step 3：每步耗时**

$$t_{\text{step}} = \frac{3.9 \times 10^{14}}{1.25 \times 10^{14}} \approx 3.1 \text{ s}$$

**Step 4：总时间**

$$T = 3.1 \times 19073 \approx 59000 \text{ s} \approx 16 \text{ h}$$

**等等——这是理论值，实际报告的是 ~14 天，差 20×**。差在哪？

- ckpt 保存（每 5000 步 ~30s，可忽略）
- val eval（每 250 步 ~5s，累计 10 分钟可忽略）
- HellaSwag eval（每 1000 步 ~10 分钟，累计 3 小时）
- bf16 / kernel warm up（前几百步慢 2-3×）
- **dataloader IO 抖动**（最大头）：单 A100 没 8 卡那种"cuda compute 时 cpu 偷偷 prefetch"的优势，IO 容易卡 GPU
- Python / autograd overhead 在小 model 上占比高
- `torch.compile()` 编译时间（30-60s 一次）

去掉这些 overhead，纯 GPU 算的话 ~16h；加上所有 overhead，单 A100 实测约 **3-5 天**（不是 14 天）——**14 天是没开 `torch.compile()` + 没开 SDPA flash backend + dataloader 同步阻塞的"naive"实现**。Karpathy 视频里报的就是优化后单 A100 ~24 小时（24 GB H100）级别。

加分：

- 解释 MFU = $\frac{\text{achieved FLOPs}}{\text{peak FLOPs}}$，是衡量训练效率的关键指标，35-50% 是 dense decoder LLM 的合理范围
- 算 8×A100 的理想时间：理论 $16/8 = 2$ h，实测 ~48 h（DDP all-reduce + IO + 各种 overhead）
- 算 8×H100 的理想时间：H100 ~3× A100，理论 $48/3 = 16$ h，实测 20-24 h
- 提到 A100 SXM4 vs PCIe 差 20-30%（NVLink 带宽 vs PCIe）

</details>

**Q3（延伸）**：复现完 GPT-2 124M 后怎么把它进化成 chat-able model？列出从 base → chat 的完整 SFT pipeline。

<details>
<summary>Answer sketch</summary>

base → chat 至少 5 步（细节见 Module 8 + 附录 B Capstone 2）：

**Step 1：选 SFT 数据集**——典型选项 OpenHermes-2.5、UltraChat、ShareGPT、OpenAssistant。GPT-2 124M 太小，不要用 1M+ 量级，3-10 万条多轮对话足够。

**Step 2：定 chat template**——例如 ChatML：
```
<|im_start|>system\n{system}\n<|im_end|>
<|im_start|>user\n{user}\n<|im_end|>
<|im_start|>assistant\n{assistant}\n<|im_end|>
```
要把 `<|im_start|>` / `<|im_end|>` 加进 tokenizer special token，GPT-2 BPE 默认没有，要扩词表（vocab_size 50257 → 50260）。扩词表后 wte / lm_head 末几行用 mean init。

**Step 3：Loss masking**——只对 assistant token 算 loss，user / system token 设 `targets = -100`（`F.cross_entropy(ignore_index=-100)` 自动跳过）。如果对全部 token 算 loss，模型会学"模仿用户提问"——这是常见的 SFT 错误。详见 8.2 chat template。

**Step 4：训练**——SFT 比 pretrain 短得多：
- lr 显著降低：2e-5 ~ 5e-5（base 的 1/10）
- epoch 1-3 即可（SFT 数据少，多 epoch 容易过拟合）
- batch size 32-128（按 token 算几十万即可）
- bf16 + AdamW 同 pretrain
- 8×A100 几小时跑完

**Step 5：评测**——SFT 后 perplexity 不再有意义，要测：
- MT-Bench / IFEval（指令遵循）
- 多轮对话定性测试（生成是否连贯、不复读、终止符正常）
- HellaSwag / MMLU（确认没掉太多 base 能力）

GPT-2 124M 这个 size SFT 后**仍然只是 toy chat**——能回答简单 hi/who are you，复杂任务全 fail。这是参数量天花板（124M），不是 SFT 没做对。要真 chat-able 至少 1B+。

加分：

- 提到 SFT 后还可以接 LoRA fine-tune（Module 8.3 + 附录 B）做轻量化领域适配
- 提到 SFT 后接 DPO / GRPO 做 preference alignment（Module 9）
- 提到 SFT 后接 RAG（Module 13.2）补外部知识
- 提到 chat template 不要"现训现造"——直接抄主流 LLaMA-3 / Qwen-2 / ChatML 三家任一，避免 BPE 边界 + special token 边界踩坑
- 提到 sample packing（8.2）：SFT 数据通常每条几百 token，直接 batch 浪费 padding；packing 把多条短样本拼到 4k 上下文里，吞吐提 2-4×

</details>

---

## 7. 延伸阅读

- [Karpathy — build-nanogpt GitHub](https://github.com/karpathy/build-nanogpt) — 本 capstone 的祖本，含完整 train.py / fineweb.py / hellaswag.py，是最系统的 GPT-2 124M 复现公开实现
- [Karpathy — Let's reproduce GPT-2 (124M) YouTube](https://www.youtube.com/watch?v=l8pRSuU81PU) — 4 小时配套视频教程，每个工程决策都讲为什么
- [Karpathy — nanoGPT GitHub](https://github.com/karpathy/nanoGPT) — build-nanogpt 的精简版前身，model.py < 300 行，是 GPT-2 124M / 355M 的标准实现
- [HuggingFace FineWeb-Edu sample-10BT](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) — 本 capstone 用的数据集，10B token 子集 ~24 GB
- [GPT-2 paper (Radford 2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — 原始 paper，HellaSwag 31.6% baseline 的出处
- [GPT-3 paper (Brown 2020)](https://arxiv.org/abs/2005.14165) — Table 2.1 的训练超参标准答案
- [HellaSwag 数据集](https://huggingface.co/datasets/Rowan/hellaswag) — 评测用，10k 验证集
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — 工业级 benchmark 评测框架，HellaSwag / MMLU / GSM8K 一键跑，比自写 eval 脚本更可靠
- [wandb LLM training quickstart](https://docs.wandb.ai/guides/integrations/huggingface) — 训练监控配置，**强烈建议训练前装上**
- 推荐继续读本教程的 **附录 B：Capstone 2 — SFT + LoRA + RAG demo**——把本 capstone 训出的 base model 进化成 chat-able + 带外部知识的小型应用
- 推荐继续读本教程的 **附录 C：Capstone 3 — Agent end-to-end**——把 LLM 接上 tool / search / memory，构建完整 agent
- 推荐回头复习 **Module 4.6 完整 decoder-only 实现**——本节代码 §3.1 与 4.6 §3.1 对照阅读，能直观看到"现代 LLM"（LLaMA 风格）vs "经典 GPT-2"风格的具体差异
