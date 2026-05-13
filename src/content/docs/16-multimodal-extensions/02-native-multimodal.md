---
title: "16.2 Native multimodal：Chameleon / GPT-4o / Gemini"
description: "把\"vision encoder + projector + LLM\"的接接器范式与\"image / audio 像 text 一样 tokenize 后丢进同一个 Transformer\"的 native multimodal 范式对照讲清——以 Chameleon（开源、Mixed-modal early-fusion）为可复现的代表，以 GPT-4o（omni 模态）与 Gemini 1.5"
---

> ⏱ 预计阅读 40 分钟 ｜ 难度 ★★★ ｜ 前置：16.1 VLM 接接器范式

## 一句话本节讲什么

把"vision encoder + projector + LLM"的**接接器范式**与"image / audio 像 text 一样 tokenize 后丢进同一个 Transformer"的 **native multimodal 范式**对照讲清——以 Chameleon（开源、Mixed-modal early-fusion）为可复现的代表，以 GPT-4o（omni 模态）与 Gemini 1.5/2.x（1M+ context、native from scratch）为闭源前沿，配 VQ-VAE / Encodec 这两套把 image / audio 离散化为 token id 的核心技术，一并讲明白 native 范式的优劣、训练 modality 冲突等工程要点，建立 16.3 语音、16.5 GUI Agent 的多模态心智模型。

---

## 1. Mental model（直觉）

### 1.1 从"拼接"到"原生"——一句话区分两套范式

16.1 讲的 LLaVA / Qwen-VL / InternVL 都属于**接接器（late fusion）范式**：image 走 vision encoder + projector 产 vision token，text 走 BPE tokenizer + embedding 产 text token，两路 token 在 LLM 入口处 concat。LLM 是**复用预训好的纯文本 decoder-only**，"看图能力"主要靠 projector 与 visual SFT 后注入。

native multimodal 走的是另一条路：**image / audio 与 text 在 tokenizer 层就被统一**。一张图片先用 VQ-VAE 量化成 1024 个 codebook id，与 BPE token id 共享同一个词表；一段语音先用 Encodec 量化成 50 Hz 的 codebook id 序列，也进同一个词表。从 pretrain 阶段开始，模型就用单一的 next-token prediction 同时预测下一个 text token、下一个 image token、下一个 audio token——**没有"vision encoder"这个独立模块**，所有 modality 的"看 / 听 / 说"都在同一个 Transformer 里完成。

```
=== 接接器范式 (16.1: LLaVA / Qwen-VL) ===

[image] ──► CLIP-ViT ──► projector ─┐
                                    ├──► concat ──► LLM ──► text
[text]  ──► BPE tokenizer ──────────┘                       (单输出 modality)

=== Native multimodal 范式 (16.2: Chameleon / GPT-4o / Gemini) ===

[image] ──► VQ-VAE encoder ──┐
                             │
[audio] ──► Encodec ─────────┼──► 统一 token 序列 ──► Transformer ──► token
                             │      (text/image/audio                   │
[text]  ──► BPE ─────────────┘       共享词表 / 共享位置)                 │
                                                                        ▼
                                                  ┌──────┬──────┬──────┐
                                                  │ text │image │audio │
                                                  │  id  │ id   │  id  │ ← 输出可同时含三种
                                                  └──────┴──────┴──────┘
                                                       │      │      │
                                                       │      ▼      ▼
                                                       │  VQ-VAE  Encodec
                                                       │  decoder decoder
                                                       │      │      │
                                                       text  image  audio
```

记住一句话：**接接器是"翻译"——把 image hidden 翻译成 LLM 听得懂的 token embedding；native 是"统一"——image / audio / text 在 tokenizer 层就讲同一种语言**。

### 1.2 早期 fusion vs 后期 fusion

学术上常用 early-fusion / late-fusion 描述这两种范式：

- **late-fusion**（16.1 接接器）：vision 与 text 各自 encode，到很后期才融合（仅在 LLM 输入层 concat）。LLM 内部 self-attention 才发生跨 modality 交互
- **early-fusion**（16.2 native）：image / audio 在 tokenizer 阶段就被映射到统一的离散 token id 空间，从 embedding layer 起就跨 modality 共享

early-fusion 的最大优势：**模型可以在每一层、每一个 head 上做 cross-modal interaction**——image patch 的"语义"与文本 token 的"语义"在同一空间共同演化。late-fusion 由于 vision feature 是 frozen vision encoder 一次性产出的，不会再随 LLM 上下文动态更新。

### 1.3 为什么"统一 tokenize"才能"统一生成"

接接器范式有一个隐含限制：**只能 text 单向输出**——LLM 只会预测下一个 text token，没法生成 image 或 audio。原因是 vision feature 是 vision encoder 输出的 continuous hidden vector，LLM 词表里没有"image token id"可供预测。

native 范式天然解决这个问题：image / audio 已经被量化成 codebook id，与 text token 一样可以被 next-token prediction 预测。预测出来的 image / audio token 序列再用对应的 decoder（VQ-VAE / Encodec decoder）解回 pixel / waveform。**这是 GPT-4o 能"原生"输出语音、Gemini 2.x 能"原生"生成图片的根本机制**。

### 1.4 三家代表谁走得最纯

| 模型 | 输入 modality | 输出 modality | 训练方式 | 开源？ |
|---|---|---|---|---|
| **Chameleon** (Meta, 2024) | text + image | text + image | early-fusion，从 0 训 | 是（7B / 34B 权重） |
| **GPT-4o** (OpenAI, 2024.05) | text + image + audio + (video) | text + image + audio | early-fusion，从 0 训 | 否 |
| **Gemini 1.5 / 2.x** (Google, 2024-) | text + image + video + audio | text + image + audio（推测） | early-fusion，从 0 训 | 否（部分技术报告公开） |

Chameleon 是目前**唯一开源**的真正 native multimodal model——它的论文与权重让学界第一次能复现 early-fusion 范式的训练困难、modality 冲突、loss 平衡等核心问题。GPT-4o / Gemini 走得更远（加 audio + video，端到端低延迟），但闭源细节有限。

---

## 2. 公式与原理

### 2.1 image tokenization：VQ-VAE 与 VQGAN

native multimodal 的基石是把连续 image 离散化为 token id。主流方法是 **VQ-VAE**（van den Oord et al., 2017）及其改进 **VQGAN**（Esser et al., 2020）。

**核心思路**：训一个 encoder $E$ 把图片压缩到 latent $z_e \in \mathbb{R}^{h \times w \times d}$（典型 $h = w = 32$、$d = 256$），然后用一个**learnable codebook** $\mathcal{C} = \{e_k\}_{k=1}^{K}$（$K$ = 词表大小，典型 $8192$）做最近邻量化：

$$z_q(x)_{ij} = e_{k^*}, \quad k^* = \arg\min_k \|z_e(x)_{ij} - e_k\|_2$$

每个空间位置 $(i, j)$ 都被替换成它在 codebook 中最近的那个 entry——一张图就被表示成 $h \cdot w$ 个 token id（典型 1024 个），词表大小 $K$。然后 decoder $D$ 从 $z_q$ 重建图片：

$$\hat{x} = D(z_q)$$

训练 loss 三项：

$$\mathcal{L}_{\text{VQVAE}} = \underbrace{\|x - \hat{x}\|_2^2}_{\text{reconstruction}} + \underbrace{\|\text{sg}(z_e) - z_q\|_2^2}_{\text{codebook loss}} + \beta \underbrace{\|z_e - \text{sg}(z_q)\|_2^2}_{\text{commitment loss}}$$

其中 $\text{sg}(\cdot)$ 是 stop-gradient（截断梯度），$\beta \approx 0.25$。codebook loss 让 codebook entry 向 encoder 输出靠拢，commitment loss 反过来让 encoder 输出向 codebook entry 靠拢；reconstruction loss 用 **straight-through estimator** 把梯度从 $z_q$ 直接绕过量化操作传回 $z_e$。

**VQGAN 的关键升级**：把 reconstruction loss 从 L2 改成 perceptual loss + adversarial loss（加判别器 $D$），让重建图清晰锐利（L2 容易模糊）。VQGAN 的 codebook 也更易学好——是现代 native multimodal（包括 Chameleon）image tokenizer 的标配。

**Magvit-v2 / Movq**：现代 video / image tokenizer，引入 lookup-free quantization、超大 codebook（262k）等改进，是 2024+ 大厂自研 tokenizer 的方向（Gemini 2.x、Veo 等）。

**关键 trade-off**——codebook 大小 $K$ 与 spatial size $h \cdot w$：

- $K$ 大（如 16384）→ 重建好但 LLM 词表暴涨、训练数据稀疏（每个 token id 出现次数少）
- $K$ 小（如 1024）→ 重建模糊
- $h \cdot w$ 大（如 $64 \times 64 = 4096$）→ 细节保留但 LLM 序列长
- $h \cdot w$ 小（如 $16 \times 16 = 256$）→ context 友好但生成糊

Chameleon 的 image tokenizer 用 $K = 8192$、$h = w = 32$（一张 256² 图 → 1024 token）。

### 2.2 Chameleon：mixed-modal early-fusion

**核心架构**：标准 decoder-only Transformer，唯一不同是 **vocab 是 text + image 的并集**。

$$|\mathcal{V}| = |\mathcal{V}_{\text{text}}| + |\mathcal{V}_{\text{image}}| \approx 65536 + 8192 \approx 73728$$

训练数据是 text + image **混合 token 序列**——例如：

```
<bos> A photo of a cat: <image_start> tok_imgA_001 tok_imgA_002 ... tok_imgA_1024 <image_end> sitting on a chair. <eos>
        ↑ text token                   ↑ 1024 个 image VQ token                       ↑ text token
```

模型学的是单一 next-token prediction loss：

$$\mathcal{L}_{\text{Chameleon}} = -\sum_{t=1}^{T} \log p(x_t | x_{<t})$$

唯一与纯 LLM 不同的是：$x_t$ 既可以是 text token id，也可以是 image token id（同一个词表）。**输出端**：当 LLM 预测出 `<image_start>` 后，下游可以选择继续 sample 出 1024 个 image token，再用 VQGAN decoder 解回 image。

**训练挑战 #1：modality token 数严重不平衡**——一张 256² 图就 1024 个 image token，相当于 ~250 词的 paragraph；如果训练数据 1:1 image-text 配比，但每个 image-text pair 里 image token 比 caption text token 多 10-20×，那 image token 的 loss 会**主导 gradient**，模型变成"看图生成图"专家而忽视 text 能力。

**Chameleon 的解决方案**——**modality-aware loss balancing + QK-norm**：

- **QK-norm**（Henry et al. 2020 的 query-key normalization）：训练大 vocab 模型时 attention logits 数值不稳，引入 layer-wise QK-norm 让 logits 落在合理范围
- **dropout-after-norm**：进一步稳定训练
- **batch 内 modality 比例严格控制**：text-only / image-text mix / image-only 三类数据按比例采样，而非"自然"语料分布

**训练挑战 #2：modality conflict（loss landscape 完全不同）**——image token 与 text token 的 loss 数量级差很多。image token 的 codebook 是 8k 而 text 是 65k，image token 的预测概率分布更"集中"（同一图区域内 token 分布有限），导致 cross-entropy loss 数值天然小于 text。如果不做 normalization，optimizer 会在 image-text 间反复"摆动"，loss landscape 高度非凸。

Chameleon 7B / 34B **公开权重**，是学界研究 early-fusion 训练动力学的唯一参照系。可在 HuggingFace `facebook/chameleon-7b` / `facebook/chameleon-30b` 直接加载（注意：原始 release 删除了 image generation head，仅保留 text-out 模式；研究界有 reverse-engineer 重新加上 image-out 的工作）。

### 2.3 GPT-4o：从底层重训的 omni 模型

OpenAI 2024.05 发布的 GPT-4o（"o" = omni）是第一个真正端到端处理 text + image + audio + video 的产品级模型。**核心特征**（来自 OpenAI 系统卡与官方 blog）：

1. **从底层重训**——不是在 GPT-4 上接 vision / audio adapter，是从 0 设计一个新模型，pretrain 阶段就把所有 modality token 混在一起训
2. **端到端低延迟**——传统语音助手是 ASR → LLM → TTS 三段 pipeline（延迟 2-5 秒），GPT-4o 全程在一个模型内（声波 token → 直接生成回应 token），**~320ms 平均响应延迟**接近人类对话
3. **跨 modality 表达保留**——传统 ASR 把"愤怒地说出 hello"压缩成纯文本 "hello"，丢失了情绪、语速、停顿；GPT-4o 直接处理 audio token，能区分语气、笑声、唱歌
4. **能输出 audio**——next-token prediction 可以预测下一个 audio token，再用 audio decoder 合成波形

OpenAI **没公开 tokenizer / 架构细节**——可推测：image 走类 VQGAN tokenizer（codebook 几千-万级别）；audio 走类 Encodec / SoundStream neural codec（codebook 1024-2048、帧率 12.5-50 Hz）；text 仍用 cl100k / o200k tiktoken；总词表估计在 20 万级别。

### 2.4 Gemini 1.5 / 2.x：native + 长 context + MoE（推测）

Google 的 Gemini 系列从 1.0 起就声称 "natively multimodal from the ground up"——即设计阶段就是 multimodal，不是先训 LLM 再加 modality。

**Gemini 1.5 Pro**（2024.02）的关键特性：

- **1M token context**（实验最高 10M）——能在单个 prompt 内放入数小时视频、上千页 PDF、整个代码库
- 支持 text + image + video + audio 输入
- **MoE 架构（推测）**——技术报告未明说，但从 inference 速度（远快于同等 FLOP 的 dense 模型）与 sparse activation 描述推测是 sparse MoE

**Gemini 2.x**（2024.12-）进一步：

- 加入 **image generation native output**（与 GPT-4o 类似的 next-image-token prediction）
- audio dialog 端到端（"Live API"）
- 推理强化版本（Gemini 2.0 Flash Thinking、2.5 Pro）

Gemini 团队公开的**关键技术 ingredient**：

- video tokenization：用类 Magvit / Magvit-v2 的 video tokenizer，把视频帧序列映射成时空 token
- **位置编码扩展**：跨 modality 共享位置编码（类似 Qwen2-VL 的 3D RoPE 思路）
- 长 context 训练：YaRN / 类似 RoPE 扩展技巧（详见 6.5）

**总结**：Gemini 的 "native" 体现在**统一 tokenize + 从 0 联合 pretrain + 长 context + 任意 modality 输入输出**，是 native 范式当前的工程上限。

### 2.5 Audio tokenization：Encodec / SoundStream

audio modality 的 native 化与 image 同思路：用 **neural codec** 把 audio waveform 量化成 discrete token id。

- **SoundStream** (Zeghidour et al. 2021, Google)：第一个端到端 neural audio codec，VQ-VAE 思路 + Residual Vector Quantization (RVQ)
- **Encodec** (Défossez et al. 2022, Meta)：SoundStream 的改进，更高质量、更易训
- **RVQ**（残差向量量化）：单层 VQ codebook 容量有限（典型 1024），用多层级联——第一层量化 audio frame → 残差再丢给第二层量化 → ... → $L$ 层（典型 8）。每帧得到 $L$ 个 token id，等价于 $1024^L$ 大小的"超 codebook"

典型 audio token 率：**Encodec 24 kHz @ 8 codebook = 75 Hz × 8 = 600 token/sec**。一段 10 秒语音就 6000 token，长对话 context 压力大。GPT-4o / Moshi / VALL-E 等都采用此类 neural codec 做底层 audio tokenizer。详见 16.3。

### 2.6 native 范式的优势与劣势

**优势**：

1. **真正的 cross-modal reasoning**——image / audio token 与 text token 在 attention 里逐层深度融合，模型能学到"看到这张图后语调应该惊讶"这种跨 modality 因果
2. **生成式 multimodal**——能输出 image / audio，不仅仅是输入。这是 GPT-4o 语音对话、Gemini 2.x 图片生成的根本
3. **scaling friendly**——与 LLM scaling law 一致：更大的模型 + 更多的多模态 token，性能持续上涨；不像接接器范式被 vision encoder 上限卡住
4. **简洁优雅**——架构上是统一的 Transformer，没有 vision encoder / projector / Q-Former 等异质组件

**劣势**：

1. **训练成本巨大**——必须从 0 重训，几亿 GPU-hour 起步；不能复用社区已有的 LLM 权重
2. **数据 hungry**——需要大规模 paired text + image + audio + video 数据，且 modality 混合配比要细心调
3. **modality conflict**——image token 数远多于 text token、loss scale 不同，optimizer 容易被某个 modality 主导，需要 modality-aware norm / loss balance
4. **开源极少**——Chameleon 是目前唯一公开的 7B+ early-fusion 权重
5. **推理慢**——image / audio 输出需要先生成 1024+ 个 token 再过 VQGAN / Encodec decoder 解码，比纯 LLM text 输出慢得多
6. **safety 难做**——能生成 image / audio 意味着可生成 deepfake、有害图像、伪造语音；safety 训练比纯 text 模型复杂得多

### 2.7 范式对比表（必背）

| 方法 | 范式 | 训练成本 | 输入 | 输出 | 是否复用预训 LLM | 代表年代 |
|---|---|---|---|---|---|---|
| **LLaVA-1.5** | encoder + projector (late-fusion) | 低（百卡天） | text + image | text-only | 是 | 2023 |
| **Qwen2.5-VL** | encoder + projector + 动态分辨率 | 中（千卡天） | text + image + video | text-only | 是 | 2024 |
| **InternVL2** | encoder（大）+ pixel shuffle | 中-高 | text + image + video | text-only | 是 | 2024 |
| **Chameleon** | early-fusion VQ token | 高（万卡天） | text + image | text + image | 否（从 0 训） | 2024 |
| **GPT-4o** | early-fusion native omni | 极高 | text + image + audio + video | text + image + audio | 否 | 2024 |
| **Gemini 1.5/2.x** | early-fusion native + 长 context | 极高 | text + image + video + audio | text + image + audio（推测） | 否 | 2024-2025 |

### 2.8 unified understanding + generation 的开源探索

native 范式的"双向"（既理解 image 又生成 image）在开源界有几条探索路线：

- **Show-o**（Xie et al. 2024, ByteDance / NUS）：用一个 decoder-only Transformer 同时做 image understanding（图 → text）与 image generation（text → image），结合 autoregressive next-token prediction（for text）与 discrete diffusion（for image），是 unified model 的代表
- **Janus / Janus-Pro**（Lu et al. 2024-2025, DeepSeek）：用**两套 vision encoder**——理解任务用 SigLIP，生成任务用 VQ tokenizer，二者共享同一个 LLM。比 Chameleon 训练更稳，开源（1B / 7B）
- **SEED-X**（Tencent / NUS 2024）：unified MLLM 用 SEED tokenizer 把 image 离散化，支持图文交替输出
- **Emu3**（BAAI 2024）：纯 next-token prediction 在 image / video / text 上 scale

这些工作证明 **native 范式开源化是可能的**，但目前性能与 GPT-4o / Gemini 仍有较大差距。2025-2026 是开源 native multimodal 加速追赶的年份。

### 2.9 2025-2026 现状速写

- **接接器范式仍是开源主流**——LLaVA / Qwen-VL / InternVL 系覆盖绝大多数开源 VLM 应用（OCR / chart / GUI / RAG），原因是复用 LLM + 工程成熟
- **native 是大公司方向**——OpenAI / Google / Meta / Anthropic（多模态 Claude 内部实现细节未公开）走 native，因为他们能承担从 0 重训的成本，且 native 是触达"真正 omni" / "image generation"的唯一路径
- **趋势：output multimodal 必然普及**——2024 GPT-4o 的"语音对话"、Gemini 2.x 的"原生图片生成"已经把用户预期拉高，2025-2026 开源 VLM 也在补"图片输出"能力
- **中文 native multimodal 公开的极少**——Tencent / 阿里 / 字节有部分内部模型（如 Hunyuan-Multimodal、Qwen-Omni、Show-o），完全公开权重的稀缺

---

## 3. 最小代码示例

### 3.1 简化 VQ-VAE image tokenizer（< 25 行 PyTorch）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MiniVQVAE(nn.Module):
    def __init__(self, in_channels=3, hidden=256, K=512):
        super().__init__()
        # encoder: 把 image 下采样 8 倍 → (B, hidden, H/8, W/8)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 4, 2, 1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 4, 2, 1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 4, 2, 1))
        # codebook：K 个 hidden 维向量
        self.codebook = nn.Embedding(K, hidden)
        self.codebook.weight.data.uniform_(-1.0/K, 1.0/K)
        # decoder：对称上采样
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden, hidden, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose2d(hidden, hidden, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose2d(hidden, in_channels, 4, 2, 1))

    def forward(self, x):
        z_e = self.encoder(x)                                  # (B, D, h, w)
        b, d, h, w = z_e.shape
        flat = z_e.permute(0, 2, 3, 1).reshape(-1, d)          # (B*h*w, D)
        # 最近邻量化：找每个 z_e 在 codebook 里最近的 entry
        dist = (flat.pow(2).sum(1, keepdim=True)
                - 2 * flat @ self.codebook.weight.t()
                + self.codebook.weight.pow(2).sum(1))
        ids = dist.argmin(1)                                   # (B*h*w,) ← 这就是 image token id
        z_q = self.codebook(ids).view(b, h, w, d).permute(0, 3, 1, 2)
        # straight-through：让梯度绕过量化直接传回 encoder
        z_q_st = z_e + (z_q - z_e).detach()
        x_rec = self.decoder(z_q_st)
        loss = F.mse_loss(x_rec, x) + F.mse_loss(z_q, z_e.detach()) \
               + 0.25 * F.mse_loss(z_e, z_q.detach())
        return x_rec, ids.view(b, h, w), loss
```

**关键点**：

- `ids` 就是 image 被离散化后的 token id 序列——256×256 输入 → 32×32 = 1024 个 token id
- straight-through estimator (`z_e + (z_q - z_e).detach()`) 让 reconstruction loss 的梯度跳过 argmin（不可导）直接传回 encoder
- 三项 loss：reconstruction + codebook + commitment（`β = 0.25`）
- 真实 VQGAN 还要加 perceptual loss + GAN discriminator 让重建清晰

### 3.2 Chameleon-style 混合 token 序列送入 Transformer（< 25 行）

```python
import torch
import torch.nn as nn

# === 词表设计 ===
VOCAB_TEXT = 32000      # text BPE
VOCAB_IMAGE = 8192      # image VQ codebook
VOCAB_TOTAL = VOCAB_TEXT + VOCAB_IMAGE + 4   # +4 个特殊 token
TOK_BOS, TOK_EOS, TOK_IMG_START, TOK_IMG_END = (
    VOCAB_TEXT + VOCAB_IMAGE + i for i in range(4))

class ChameleonStyle(nn.Module):
    def __init__(self, d_model=512, n_layer=6):
        super().__init__()
        # 唯一关键：embedding 是 text + image + special 共享一张表
        self.embed = nn.Embedding(VOCAB_TOTAL, d_model)
        layer = nn.TransformerEncoderLayer(d_model, 8, batch_first=True)
        self.blocks = nn.TransformerEncoder(layer, n_layer)
        self.head = nn.Linear(d_model, VOCAB_TOTAL)            # 同样输出整词表

    def forward(self, ids):
        # ids 是 text + image 混合 token 序列，例如：
        # [BOS, t_1, t_2, IMG_START, i_1, ..., i_1024, IMG_END, t_3, EOS]
        x = self.embed(ids)
        mask = nn.Transformer.generate_square_subsequent_mask(ids.size(1)).to(ids.device)
        h = self.blocks(x, mask=mask, is_causal=True)
        return self.head(h)                                    # (B, T, VOCAB_TOTAL)

# 构造一个 mixed-modal sample
text_pre = torch.tensor([TOK_BOS, 100, 250, 999])              # text token id ∈ [0, 32000)
image_ids = torch.randint(VOCAB_TEXT, VOCAB_TEXT + VOCAB_IMAGE, (1024,))
text_post = torch.tensor([300, 88, TOK_EOS])
seq = torch.cat([text_pre,
                 torch.tensor([TOK_IMG_START]), image_ids, torch.tensor([TOK_IMG_END]),
                 text_post]).unsqueeze(0)                      # (1, T)

logits = ChameleonStyle()(seq)                                 # (1, T, VOCAB_TOTAL)
# next-token prediction loss 对 text 和 image token 一视同仁
```

**关键点**：

- `embed` 与 `head` 都用 `VOCAB_TOTAL`——image / text 共享一张词表是 Chameleon 的本质
- image token id 通过 offset（`+ VOCAB_TEXT`）放进同一空间，模型从 id 数值就能区分 modality
- 训练 loss 仍是 next-token prediction，没有 modality-specific head
- 真实 Chameleon 还有 QK-norm、modality-aware loss balance 等稳定训练 trick，本示例省略

### 3.3 HuggingFace Chameleon 推理 demo（< 20 行）

```python
# pip install transformers pillow
from transformers import ChameleonProcessor, ChameleonForConditionalGeneration
import torch
from PIL import Image

model_id = "facebook/chameleon-7b"

processor = ChameleonProcessor.from_pretrained(model_id)
model = ChameleonForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto")

image = Image.open("dog.jpg")
prompt = "<image>What breed is this dog and what is it doing?"

inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda", torch.bfloat16)

# Chameleon 公开权重的 image 输出 head 已被 Meta 移除（safety 原因），仅 text-out
out_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
print(processor.decode(out_ids[0], skip_special_tokens=True))
```

**关键点**：

- Chameleon 的 `<image>` placeholder 由 `ChameleonProcessor` 自动展开成 1024 个 image VQ token id 插入到 prompt 中
- Meta 公开权重时**主动移除了 image generation head**（防 deepfake），所以官方 demo 只能 text-out。学界有 reverse-engineer 项目恢复 image-out（如 Anole）
- `device_map="auto"` 允许 7B 模型在单张 24G GPU 上 bf16 推理
- 如果要试 native multimodal 的"输出 image"，开源选项目前是 Janus-Pro / Show-o / Anole

---

## 4. 工程踩坑与经验

- ❗ **Chameleon-like 训练必须做 modality-aware loss balance，否则 image token 主导 gradient**。一张 256² 图就 1024 个 image token，相当于一段 ~250 词的 paragraph；如果 image-text 数据 1:1 配比，每个 sample 的 image token 数远多于 text token，loss gradient 自然被 image 主导——模型变成"看图生成图"专家，text 能力退化。**对策**：(1) batch 内严格控制 text-only / image-text / image-only 三类数据比例；(2) loss 按 modality 单独 normalize 后再相加；(3) Chameleon 用 QK-norm + dropout-after-norm 进一步稳定数值。盲目"自然"配比训出来的 native model 几乎必崩。
- ❗ **Native multimodal 几乎不可能在小团队复现**。Chameleon 7B 训练用 5T+ token，估计千卡月级别；GPT-4o / Gemini 万卡月级别。学术界几乎无法从 0 复现，只能 (1) 用 Chameleon-7B 公开权重 fine-tune；(2) 走"unified understanding + generation"开源路线（Janus / Show-o）；(3) 在小数据集上跑教学 demo（如 256² 图 + 简短 caption）。**如果你的需求是落地一个能看图的产品，无脑选 16.1 的接接器范式（LLaVA / Qwen-VL）**——native 是研究方向，不是工程首选。
- ❗ **Image VQ tokenizer 的 codebook size 是 trade-off，没有最优解**。codebook $K$ 大（如 16384）→ 重建图清晰、单 token 信息密度高，但 LLM 词表暴涨、低频 codebook entry 长尾（很多 entry 在数据里出现极少，gradient sparse）；$K$ 小（如 1024）→ codebook 利用率高但重建糊。**实践 sweet spot**：$K \in [4096, 16384]$，spatial size $32 \times 32$ 或 $64 \times 64$。Chameleon 用 $K = 8192$、$32 \times 32$。VQGAN 训练时还要监控 codebook usage（理想 80%+ entry 被频繁用到，<50% 说明 codebook 在 collapse）。
- ❗ **中文 native multimodal model 几乎没有公开权重**。除部分 Tencent / 阿里 / 字节内部探索（Hunyuan-Multimodal、Qwen-Omni 公测、Show-o），完全开源的中文 native multimodal 极少。**如果要做中文 native 应用**，大概率走"基于 Chameleon-7B / Janus-Pro / Show-o + 中文数据 continue-pretrain + 中文 visual SFT"路线，工程门槛极高且不一定 work（中文 image-text 数据稀缺）。中文场景仍建议优先 Qwen2.5-VL / InternVL2 这类接接器范式 SOTA。
- ❗ **Native model 的 image / audio 输出推理慢得多**。LLM 输出一段 100 字的 text 大约 100-200 个 token；输出一张 256² 图要 1024 个 image VQ token + 一次 VQGAN decoder forward；输出 10 秒语音要 6000 个 audio codec token + Encodec decoder forward。**端到端延迟上**："看图回答 text" ≈ 纯 LLM；"看图生成图" 比纯 LLM 慢 3-5×；"实时语音对话"必须靠流式生成 + 优化的 codec decoder（GPT-4o / Moshi 都做了大量工程）。如果你部署 native 模型对外服务，必须给"输出 modality"单独的 latency budget。
- ❗ **Output multimodal 的安全风险远超 text-only**。能生成 image 意味着能生成 deepfake / NSFW / 暴力图；能生成 audio 意味着能伪造任意人语音。**Chameleon 公开权重时主动移除了 image generation head**就是出于这个考虑。GPT-4o 的语音输出有专门的 voice clone 防御（限定几个预置 voice、拒绝模仿真人）。任何上线 native multimodal 的产品都必须额外加：(1) image / audio safety classifier 后置过滤；(2) 训练阶段 RLHF / DPO 引导拒绝有害生成；(3) watermark（如 Google SynthID）标记 AI-generated 内容；(4) 拒绝 "as a celebrity say X" 类 prompt。这是 LLM safety 之外的全新攻击面。
- ❗ **HuggingFace Chameleon 加载需要对齐 chat template 与 image placeholder**。`ChameleonProcessor` 会把 prompt 里的 `<image>` 自动展开成 1024 个 image token——但这个 placeholder 一定要写在 prompt 字符串里、不能漏；image 必须 PIL.Image 类型且为 RGB；多图场景每个 `<image>` 对应一张图，顺序严格匹配。手拼 prompt（不用 `processor.apply_chat_template`）几乎必踩坑。
- ❗ **Codebook collapse 是 VQ-VAE 训练的常见失败模式**。训不好的 VQ-VAE 会出现"99% 的输入都被映射到 codebook 里同一个 entry"——重建效果极差且 LLM 学不到任何 image 信息。**对策**：(1) commitment loss 系数 $\beta$ 调大（0.25-1.0）让 encoder 更"努力"用满 codebook；(2) EMA-update codebook（van den Oord 2017 §3）替代 gradient-update，更稳；(3) dead code revival——周期性把没被使用的 entry 重置为最近一批 encoder 输出的随机样本；(4) 监控 codebook usage entropy，<50% utilization 立即 stop & debug。
- ❗ **Audio token 帧率与 LLM context 严重 mismatch**。Encodec 24kHz / 8 codebook → 600 token/sec，10 分钟语音就 36 万 token，远超主流 LLM context window。**GPT-4o / Moshi 的工程对策**：(1) 降低 codebook 数（用 4-codebook RVQ 而非 8）；(2) 降低帧率（用 12.5 Hz 而非 50 Hz neural codec）；(3) 流式生成（user 一边说一边模型一边处理，不等完整 sample）。直接用现成 Encodec 训长语音对话几乎必爆 context。
- ❗ **联合 pretrain 的数据配比是黑魔法**。多 modality 预训需要的 data mixture（text-only : image-text : audio-text : video-text : interleaved）没有公开 best practice——Chameleon 论文给的配比是该工作团队的探索结果，不一定迁移到你的数据上。**业界共识**：text-only 至少占 40-60%（防纯文本能力退化）、image-text 30-40%、其他 modality 视场景。任何 modality 配比变化都需要 re-pretrain 验证，成本极高。这也是 native multimodal "实验代价远超普通 LLM" 的根本原因之一。

---

## 5. 经典 paper

- **Chameleon Team (Meta), 2024 — Chameleon: Mixed-Modal Early-Fusion Foundation Models** — 必读的 native multimodal 开源代表作。读 §2 Method（架构 + tokenizer）+ §3 Pre-training（数据混合、QK-norm 等稳定训练 trick）足以——理解了 Chameleon 的混合 token + 单一 next-token prediction loss，就掌握了 early-fusion 的核心。本节 §2.2 + §3.2 直接源于这篇。
- **van den Oord et al., 2017 — Neural Discrete Representation Learning (VQ-VAE)** — 必读的"image / audio 离散化"奠基作。读 §2 VQ-VAE 公式 + §3 实验即可——理解 codebook + straight-through + 三项 loss 后就能看懂所有 image / audio token 化的论文。本节 §2.1 + §3.1 直接复述这篇。
- **Gemini Team (Google), 2024 — Gemini 1.5: Unlocking Multimodal Understanding Across Millions of Tokens of Context** — 必读的 native + 长 context 代表作。读 §2 Architecture + §3 Long Context 章节——理解 1M context 的工程实现 + native multimodal 的 Google 路线。Gemini 2.x 技术报告也建议浏览，看 image-out / audio-out 的演进。
- **Esser et al., 2020 — Taming Transformers for High-Resolution Image Synthesis (VQGAN)** — 必读的"现代 image tokenizer"代表作。读 §3.1 VQGAN（perceptual + adversarial loss）即可——理解 VQ-VAE 到 VQGAN 的关键改进。Chameleon / Janus / Show-o 的 image tokenizer 都基于此。
- **OpenAI, 2024 — GPT-4o System Card** — 加分必读。虽然技术细节不公开，但 system card 透露了 omni 模型的 latency 数据、safety 措施、能力 benchmark，能感受 native multimodal 的产品级表现。
- **Lu et al., 2024 — Janus / Janus-Pro: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation** — 加分必读。开源 unified multimodal 的代表，"两套 vision encoder（理解用 SigLIP、生成用 VQ）+ 共享 LLM"的设计在 Chameleon 之外提供了另一条路。
- **Xie et al., 2024 — Show-o: One Single Transformer to Unify Multimodal Understanding and Generation** — 加分必读。把 autoregressive（for text）与 discrete diffusion（for image）放进同一个 Transformer，是 unified model 的另一条思路。

---

## 6. 自测与面试题

**Q1（对比）**：LLaVA 拼接范式与 Chameleon native 范式的 3 个核心差异是什么？请从 (1) tokenizer / 词表、(2) 训练成本与数据、(3) 输出能力三个维度展开。

<details>
<summary>Answer sketch</summary>

**3 个核心差异**：

| 维度 | LLaVA（拼接 / late-fusion） | Chameleon（native / early-fusion） |
|---|---|---|
| **(1) tokenizer / 词表** | 两套独立：text 用 BPE（~32k）、image 走 CLIP-ViT 产 continuous hidden + projector 转到 LLM token space（不在词表里）。**vision 与 text 词表分离** | **统一词表**：text BPE (32k) + image VQ codebook (8k) + special token，共 ~73k；image 被 VQ-VAE 量化成离散 id 与 text id 共享 embedding |
| **(2) 训练成本与数据** | 复用预训好的 LLM + CLIP（两个昂贵预训模型免费用），只训 projector + visual SFT，几百卡天即可 | **从 0 重训**——必须重新跑 multi-modal pretrain，万卡月级别。需要大规模 paired text + image 数据，配比要严格控制（防 image token 主导 gradient） |
| **(3) 输出能力** | **只能 text-out**——LLM 只预测 text token id，没法生成 image | **可同时 text + image-out**——next-token prediction 可预测 image VQ token，再用 VQGAN decoder 解回图片 |

**额外可加分点**：

- early-fusion 的 cross-modal interaction 在每一层 attention 里都发生，late-fusion 仅在 LLM 入口处 concat 后才发生
- LLaVA 的 vision encoder（CLIP）通常 freeze，Chameleon 整个 Transformer 联合训
- LLaVA 的 vision token 不参与 loss（labels mask 为 -100），Chameleon 的 image token 与 text token 一视同仁参与 next-token prediction loss
- LLaVA 易 fine-tune（社区 friendly），Chameleon 的 fine-tune 需要重训 image VQ tokenizer 或至少 align 到既有 codebook

</details>

**Q2（trade-off）**：Native multimodal 既然有"真正 cross-modal reasoning"和"生成式 multimodal"两大优势，为什么 2025-2026 开源主流仍是 16.1 的接接器范式？至少给 4 条原因。

<details>
<summary>Answer sketch</summary>

**4 条主要原因**（覆盖工程、数据、生态、安全）：

1. **训练成本不可承受**——native 必须从 0 重训，几亿 GPU-hour 起步，开源社区与中小团队完全负担不起。接接器范式可以复用 Llama / Qwen / DeepSeek 等公开权重，几百卡天就能跑通一个能 demo 的 VLM。**这是最根本的原因**——成本差 100×+
2. **数据稀缺**——native 需要大规模 paired text + image + audio + video 数据，且配比要严格控制；高质量的 multi-modal interleaved 数据（图文交替 web 文档）是 OpenAI / Google 的内部资产。开源社区的 image-text 数据（LAION / DataComp / Capfusion）做接接器够用，但跑 native 远不够
3. **modality 配比黑魔法**——任何配比变化都要 re-pretrain 验证，实验成本极高。接接器范式只调 visual SFT 数据，几小时一轮，迭代极快
4. **生态成熟度**——LLaVA / Qwen-VL 系有完整的开源 fine-tune pipeline、HF integration、社区数据集、quantization / vLLM 支持；Chameleon 等 native 模型生态尚浅，工程链路不齐
5. **safety 与责任**——能生成 image / audio 的模型有 deepfake / NSFW / 伪造语音风险，Meta 公开 Chameleon 时主动移除 image generation head 就是这个原因。开源社区难以承担相应的责任与监管
6. **接接器对 99% 应用够用**——OCR、chart 理解、document QA、GUI Agent、RAG、多图理解等绝大多数 VLM 应用只需要"看图 + 输出 text"，接接器完全 cover。"原生输出 image / audio" 是真正用户少数（创意工具、AI 艺术、对话语音助手）的需求

**结论**：native 是研究 / 大厂方向；接接器是工程 / 落地方向。两者长期共存，并非互相替代。

</details>

**Q3（前沿）**：Gemini 2.x 的 "native" 具体如何 native？为什么 2024-2026 大厂（OpenAI / Google / Meta）都在走 native 路线？

<details>
<summary>Answer sketch</summary>

**Gemini 2.x 的 "native" 体现在 4 个层面**：

1. **统一 tokenize + 共享词表**——image / video / audio / text 在 tokenizer 阶段就被映射到统一的离散 token id 空间（image / video 走类 Magvit-v2 时空 tokenizer、audio 走 neural codec、text 走 SentencePiece），共享一张大词表
2. **从 0 联合 pretrain**——不是先训 LLM 再加 modality adapter，pretrain 阶段就用混合 modality 数据训 next-token prediction
3. **任意 modality 输入输出**——输入 text + image + video + audio 任意组合，输出 text + image + audio（2.x 的关键升级是 image generation native 化）
4. **跨 modality 共享位置编码 + 长 context**——统一 RoPE / 长 context 训练（YaRN / 类似技巧）让 1M+ token 内的多 modality 序列保持一致的位置语义

**为什么大厂都走 native**（4 条战略原因）：

1. **触达"原生 omni"的唯一路径**——GPT-4o 的端到端低延迟语音对话、Gemini 2.x 的原生图片生成只能靠 native 实现。接接器范式做不到（vision encoder + LLM 之间不可能做出 320ms 端到端 voice）
2. **scaling law 红利**——native 模型的 scaling 与纯 LLM 一致：更大的模型 + 更多的多 modality token，benchmark 持续涨。接接器范式被 vision encoder 上限卡住，scaling 主要靠 LLM 部分
3. **训练数据规模化**——大厂有 web 级 multi-modal 数据（YouTube、Google Photos、Web 上所有图文混排页面），native 范式能完整吸收；接接器只能用 paired image-text，浪费大量"interleaved" 数据
4. **战略护城河**——native 重训成本极高（几亿 GPU-hour），是大厂相对开源社区与小团队的天然壁垒。这与"大厂走 RLHF 闭环、开源走 SFT" 是同样的护城河逻辑
5. **多模态产品形态需求**——从语音助手（OpenAI Voice、Google Assistant）到视频理解（YouTube 分析）到 image generation 工具，产品迫使技术栈走向 native

**加分**：能讨论"native 与接接器并非二选一"——很多大厂内部同时维护两条线（接接器做研究 / 快速迭代，native 做产品 / 长期投入），且开源社区在不断追赶 native（Janus-Pro、Show-o、Anole 等开源 unified model）。

</details>

---

## 7. 延伸阅读

- [Chameleon paper (arXiv 2405.09818)](https://arxiv.org/abs/2405.09818) — Meta Chameleon 原论文，必读 §2-§3 架构与训练
- [Chameleon HuggingFace 模型卡](https://huggingface.co/facebook/chameleon-7b) — 7B 权重与官方 inference demo
- [Anole GitHub (GAIR-NLP/anole)](https://github.com/GAIR-NLP/anole) — 重新加上 Chameleon 被移除的 image generation head 的开源工作，让你能真正试 native image-out
- [Janus-Pro GitHub (deepseek-ai/Janus)](https://github.com/deepseek-ai/Janus) — DeepSeek 开源 unified understanding + generation 模型，1B / 7B 权重
- [Show-o GitHub (showlab/Show-o)](https://github.com/showlab/Show-o) — autoregressive + discrete diffusion 混合的 unified model
- [Gemini 1.5 技术报告 (arXiv 2403.05530)](https://arxiv.org/abs/2403.05530) — Google 关于 1M context + native multimodal 的官方技术细节
- [GPT-4o 介绍 (OpenAI Blog)](https://openai.com/index/hello-gpt-4o/) — GPT-4o 官方介绍 + 系统卡链接，了解 omni 模型产品级表现
- [VQ-VAE paper (arXiv 1711.00937)](https://arxiv.org/abs/1711.00937) — van den Oord 原论文，必读 §2 量化 + straight-through
- [VQGAN paper (arXiv 2012.09841)](https://arxiv.org/abs/2012.09841) — Esser et al. VQGAN，理解现代 image tokenizer
- [Magvit-v2 paper (arXiv 2310.05737)](https://arxiv.org/abs/2310.05737) — Google 现代 video tokenizer，Gemini 2.x 推测使用
- 推荐继续读本教程的 **16.3 节《语音：Whisper / VALL-E / CosyVoice / 全双工 Moshi》**——audio 端的 native multimodal 详细讲法
- 推荐继续读本教程的 **16.1 节《VLM：CLIP / LLaVA / Qwen-VL / InternVL》**——本节多次对比的接接器范式基础
