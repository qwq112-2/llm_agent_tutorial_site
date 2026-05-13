---
title: "16.1 VLM：CLIP / LLaVA / Qwen-VL / InternVL（vision encoder + projector + LLM）"
description: "把\"接接器\"范式的 VLM 一次讲清——vision encoder（CLIP / SigLIP）+ projector（MLP / Q-Former / pixel shuffle）+ LLM——以 CLIP 为对比学习起点、LLaVA 为简洁三件套范例、Qwen2.5-VL 与 InternVL2 为现代 SOTA，串起从 2021 到 2025 的开源 VLM 演化主线。本节同时把\"visi"
---

> ⏱ 预计阅读 50 分钟 ｜ 难度 ★★★ 🔥 必考 ｜ 前置：Module 4（Transformer）、1.4 §2.3 InfoNCE

## 一句话本节讲什么

把"接接器"范式的 VLM 一次讲清——**vision encoder（CLIP / SigLIP）+ projector（MLP / Q-Former / pixel shuffle）+ LLM**——以 CLIP 为对比学习起点、LLaVA 为简洁三件套范例、Qwen2.5-VL 与 InternVL2 为现代 SOTA，串起从 2021 到 2025 的开源 VLM 演化主线。本节同时把"vision token 数 vs 分辨率的 trade-off"、"projector 预训 → vision-LLM 联合 SFT 两阶段训练流程"、"动态分辨率 / pixel shuffle 等现代 mitigation"这三件工程必考点讲透——是 native multimodal（16.2）出场前必须先建立的 mental model，也是 multimodal embedding（16.4）和 GUI Agent（16.5）的共同骨架。

---

## 1. Mental model（直觉）

### 1.1 把"看图说话"拆成三件套

LLM 是文本世界的专家——给它 token id 它能预测下一个 token id。但 LLM 不会"看图"：图片是 RGB pixel array，不是 token id。怎么办？最直接的思路就是**把图片也变成 token id**，再丢进 LLM。

但这里有个 mental gap：图像 pixel 与文本 token 完全不同源——一张 224×224 图有 150528 个 float，一个 BPE token 是一个 int。强行把 pixel 当成 token 直接 embed，模型学不到任何东西。**关键在中间需要一个"翻译层"**：先用一个专门看图的网络把图片压缩成 hidden vector，再把 hidden vector 投影到 LLM 能听懂的"伪 token embedding"——这就是 LLaVA 提出的三件套范式：

```
图像 (3, H, W)
   │
   ▼
┌─────────────────┐
│ Vision Encoder  │   ← CLIP-ViT / SigLIP，输出 (N_vis, d_vis) hidden
│  (frozen 或微调)  │      N_vis = (H/P)·(W/P)，P = patch size（典型 14）
└────────┬────────┘
         │ (N_vis, d_vis)
         ▼
┌─────────────────┐
│   Projector     │   ← MLP / Q-Former / pixel shuffle
│  (核心可训部件)   │      把 d_vis → d_llm，可能也压缩 N_vis
└────────┬────────┘
         │ (N_vis', d_llm)  ← 与 LLM token embedding 同空间
         ▼
   ┌──────────────────────────────────────┐
   │ [vision tokens] [text tokens]        │   ← concat 后送 LLM
   └──────────────────────────────────────┘
                    │
                    ▼
              ┌──────────┐
              │   LLM    │   ← Vicuna / LLaMA / Qwen，标准 decoder-only
              │ (decoder)│
              └─────┬────┘
                    │
                    ▼
              text tokens (response)
```

**记住这张图**——所有"接接器"范式 VLM（LLaVA / Qwen-VL / InternVL / MiniCPM-V / Yi-VL / Idefics）都是这张图的变种。区别只在三处：(1) vision encoder 选哪个（CLIP / SigLIP / EVA / 自训 InternViT）、(2) projector 怎么设计（简单 MLP / Q-Former / pixel shuffle / patch merger）、(3) 训练数据与策略（什么时候 freeze 谁、什么时候 unfreeze）。

### 1.2 为什么不是 end-to-end 从 pixel 训起

理论上你可以从零开始训一个 multi-modal Transformer，让它直接吃 pixel 与文本 token——但实际上没人这么做，**因为预训好的 vision encoder 与预训好的 LLM 都太贵了**。

- 训一个 CLIP-ViT-L 要 4 亿 image-text pair × 数千 GPU days
- 训一个 7B LLM 要 2T+ token × 数万 GPU days
- 如果重新 end-to-end 训，等于把这两笔钱都重投一遍

LLaVA 的核心 insight 是：**把两个预训好的模型用一个轻量 projector 拼起来，只用 100k 量级的 visual instruction 数据就能让它具备视觉对话能力**——558k caption pair 训 projector 几小时 + 150k visual instruction 训 LLM 几天，开源界一台 8×A100 就能跑通。这就是 LLaVA 引爆开源 VLM 的根本原因——**门槛被降到了"组装"级别**。

### 1.3 为什么 vision encoder 要"对比学习"出来（CLIP 的角色）

简单 ImageNet 预训出来的 ResNet 或 ViT 也能产 image hidden，但它们的表征是"分类导向"的——对"猫 / 狗 / 飞机"这种 1000 类敏感，对 "一个穿红裙的女人在沙滩上" 这种自由文本描述基本无感。**VLM 需要的不是"分类好的 vision encoder"，而是"能跟自由文本对齐的 vision encoder"**。

CLIP 的对比学习恰好是这种对齐的标准答案：在 4 亿 (image, caption) pair 上让"image embedding"与"caption embedding"在共享空间里对齐——训完后 image hidden 天然就跟自然语言"听得懂"。所以现代 VLM 的 vision tower 几乎全部是 CLIP-family（CLIP-ViT-L / SigLIP / EVA-CLIP / DFN-CLIP）。**用纯 ImageNet 预训的 ViT 当 vision tower 性能会塌掉 10+ 点**——这是 LLaVA 早期 ablation 的明确结论。

### 1.4 与 native multimodal 的根本区别（16.2 衔接）

本节的"接接器"范式有一个共同特征：**vision 与 text 是分开 tokenize、分开 encode 的**——image 走 vision encoder + projector 产 vision token，text 走 BPE tokenizer + embedding 产 text token，两者在 LLM 入口处 concat 后才汇合。这种"late fusion"易实现、能复用预训好的 LLM。

**Native multimodal**（Chameleon / GPT-4o / Gemini）走的是另一条路：**image / audio 与 text 共享 tokenizer**——image 被 VQ-VAE 量化成离散 token id（如 codebook 8192），与 text token id 在同一个词表里，从 pretrain 阶段就一起训 next-token prediction。这种"early fusion"潜力更大但工程门槛高得多——必须从 0 训。详见 16.2。

记住一句话：**本节学的是怎么"组装"一个 VLM；16.2 学的是怎么"原生"训一个 VLM**。前者是 2023-2025 开源主流，后者是 2024+ 闭源前沿。

---

## 2. 公式与原理

### 2.1 CLIP：对比学习对齐 image / text

CLIP 的核心是两个 encoder：image encoder $f_I: \mathbb{R}^{3 \times H \times W} \to \mathbb{R}^d$、text encoder $f_T: \text{Token}^L \to \mathbb{R}^d$，输出共享 $d$ 维 embedding 空间。

batch 内有 $N$ 个 (image, text) pair，构造 $N \times N$ 相似度矩阵 $S$：

$$S_{ij} = \frac{f_I(x_i^{\text{img}}) \cdot f_T(x_j^{\text{txt}})}{\tau \cdot \|f_I(x_i^{\text{img}})\| \cdot \|f_T(x_j^{\text{txt}})\|} \in \mathbb{R}^{N \times N}$$

其中 $\tau$ 是 learnable temperature（CLIP 初始 $1/\tau \approx 14.3$）。对角线 $S_{ii}$ 是正样本（image $i$ 与 text $i$ 是真实配对），off-diagonal 是 batch-internal negative。

双向 InfoNCE loss（image-to-text + text-to-image，平均）：

$$\mathcal{L}_{\text{CLIP}} = \frac{1}{2N}\sum_{i=1}^{N}\bigl[-\log \frac{e^{S_{ii}}}{\sum_j e^{S_{ij}}} - \log \frac{e^{S_{ii}}}{\sum_j e^{S_{ji}}}\bigr]$$

这就是 1.4 §2.3 InfoNCE 公式直接套用。**关键点**：

- batch size 必须够大（CLIP 用 32k）——negative 越多对比越严格
- text encoder 在 CLIP 时代是从 0 训的小 Transformer；现代 VLM 直接复用 CLIP-ViT 当 vision encoder，不再用 CLIP 自带的 text encoder
- training 完后 image / text 落入同一 embedding 空间——支持 zero-shot classification（拿 1000 类 label 文本 embedding 与 image embedding cosine 比，取最大）

### 2.2 SigLIP：sigmoid 替代 softmax

CLIP 的 softmax InfoNCE 在大 batch 下数值不稳（$N \times N$ 相似度矩阵 32k×32k = 1B 个 entry，softmax 需要 row-wise log-sum-exp）。Zhai 2023 提出 **SigLIP**：把 row-wise softmax 改成 element-wise sigmoid——每个 (image, text) pair 独立做二分类（"是不是配对"），不再 cross-pair normalize：

$$\mathcal{L}_{\text{SigLIP}} = -\frac{1}{N^2}\sum_{i,j}\log\sigma\bigl(z_{ij}\cdot S_{ij} + b\bigr), \quad z_{ij} = \begin{cases}+1 & i = j\\ -1 & i \neq j\end{cases}$$

其中 $b$ 是 learnable bias（修正大量负样本带来的偏置）。**好处**：

- 无 row-wise normalize → 数值更稳，能 scale 到更大 batch
- 可以做 sub-batch 计算（每个设备只看自己的 row × 全局的 col），节省通信
- 实证 SigLIP 比 CLIP 略好且更易 scale → 现代 VLM 的 vision tower 主流是 SigLIP

### 2.3 LLaVA：projector 的最简形式

LLaVA-1.0 的 projector 就是**一个 nn.Linear**：

$$h^{\text{LLM}}_{i} = W \cdot h^{\text{vis}}_{i} + b, \quad W \in \mathbb{R}^{d_{\text{LLM}} \times d_{\text{vis}}}$$

LLaVA-1.5 升级为 **2 层 MLP**（带 GELU）：

$$h^{\text{LLM}}_{i} = W_2 \cdot \text{GELU}(W_1 \cdot h^{\text{vis}}_{i})$$

参数量极小（典型 $d_{\text{vis}} = 1024$、$d_{\text{LLM}} = 4096$，2 层 MLP 仅 $\sim 25 \text{M}$ 参数，相比 7B LLM 不到 0.4%）。但实证它就是够用——比复杂的 Q-Former 性能基本持平。

**两阶段训练流程（必背）**：

| 阶段 | 数据 | 可训参数 | 目标 |
|---|---|---|---|
| Stage 1 (Pre-training / Alignment) | 558k caption pair | **只 projector** | 把 vision feature 投到 LLM token space |
| Stage 2 (Visual Instruction Tuning) | 150k visual instruction | **projector + LLM**（vision encoder freeze） | 教 LLM 怎么用 vision 信息回答问题 |

为什么要分两阶段？

- Stage 1：vision encoder 与 LLM 都 freeze，只训 projector。projector 是一个新初始化的小网络——直接和 LLM 一起 fine-tune 会让 LLM 被未对齐的 vision feature 污染（几个 step 就崩）。先单独训 projector 把 vision-LLM 接口"焊牢"
- Stage 2：projector 已经学会了"翻译"，可以解锁 LLM 一起 SFT，让 LLM 学到"什么时候该看图、看图后怎么用"。vision encoder 通常仍 freeze（避免破坏 CLIP 已学好的 vision 表征）

**LLaVA-1.6（NeXT）/ LLaVA-OneVision 的进化**：用 **AnyRes**（把高分辨率图分成多个 patch 块，每块独立过 vision encoder 后拼接）支持高分辨率；OneVision 进一步用 SigLIP 替代 CLIP-L、增加 video 支持。

### 2.4 BLIP-2：Q-Former 抽取 vision feature

BLIP-2 不直接把 vision encoder 所有 patch 都丢给 LLM，而是引入 **Q-Former**——一组 learnable query token（典型 32 个），通过 cross-attention 从 vision feature 里"提取"信息：

$$Q_{\text{out}} = \text{CrossAttn}(Q_{\text{learn}}, K = h^{\text{vis}}, V = h^{\text{vis}})$$

输出 32 个 query token 投到 LLM token space。**好处**：

- 固定数量的 vision token（32 个），与图像分辨率无关 → context 压力小
- query token 可以学到"哪些 vision 区域重要"

**坏处**：

- Q-Former 自身要训（BLIP-2 用 image-text matching + ITC + ITG 三阶段预训）
- 信息瓶颈——32 个 token 装不下高分辨率图的细节（OCR / chart 任务表现差）

**结论**：现代主流不再用 Q-Former，因为简洁的 MLP projector + 高分辨率 vision encoder 综合更优。BLIP-2 是早期探索的代表，了解即可。

### 2.5 Vision token 数 vs 分辨率的 trade-off（必考）

vision token 数 = $\lceil H/P \rceil \cdot \lceil W/P \rceil$，$P$ 是 patch size（CLIP-ViT 通常 14）。

| 分辨率 | patch size | vision token 数 |
|---|---|---|
| 224×224 | 14 | 256 |
| 336×336 | 14 | 576 |
| 448×448 | 14 | 1024 |
| 1024×1024 | 14 | 5184 |
| 2048×2048 | 14 | 21316 |

**问题**：高分辨率对 OCR / chart / detection 类任务至关重要，但 vision token 占用 LLM context window——5k token 已经吃掉一半 8k context。如果还有几张图、还要回答长问题，context 直接爆。

**业界 mitigation 方法**（必背）：

1. **Pixel shuffle**（InternVL）：把 $2 \times 2$ 邻域的 vision token 沿 channel 维拼接 → token 数 $\div 4$，channel 数 $\times 4$。空间信息保留在 channel 里，LLM 仍能看到。InternVL2 标配
2. **Q-Former / Resampler**（BLIP-2 / Idefics）：用 learnable query 抽固定数量 token，与分辨率无关。信息有损但量可控
3. **Dynamic resolution / Patch merger**（Qwen2-VL）：按原图实际宽高动态切 patch，不强制 resize 到固定分辨率；相邻 token 用一个 patch merger MLP 合并 4 个 → 1 个
4. **AnyRes / Tile**（LLaVA-NeXT）：高分辨率图切成多个 tile（如 4×4），每个 tile 独立过 vision encoder 后拼接；可保留细节但 token 数仍随分辨率增长
5. **Token pruning / merging**（学术为主）：基于 attention score 动态删掉不重要的 vision token

实践上 **pixel shuffle + AnyRes** 是 2024-2025 主流组合（InternVL2、LLaVA-OneVision），**dynamic resolution** 是 Qwen 系独特路线。

### 2.6 Qwen-VL 系：动态分辨率 + 3D RoPE

**Qwen-VL（v1, 2023）**：CLIP-ViT-bigG（1.9B）+ position-aware vision-language adapter（cross-attention + 3 层 MLP）+ Qwen-7B。

**Qwen2-VL（2024）**：核心两大升级——

1. **Native Resolution（动态分辨率）**：图像不强制 resize 到 224 / 336，按原图 pixel 数动态切 patch（受 max_pixels 约束，典型 1280×28×28）。一张 4K 图与一张 thumb 走同一个流程，patch 数自动适配
2. **3D RoPE**：把传统 1D RoPE 扩展为 (temporal, height, width) 三轴——image 是 (1, h, w)、video 是 (T, h, w)。相同的 RoPE 公式在三个轴上独立旋转，让模型能同时建模空间与时间关系

**Qwen2.5-VL（2024 末）**：进一步加强 OCR / chart / video understanding，是 2024-2025 开源 VLM SOTA 之一。多模态 SFT 数据规模到 1B+ pair，支持长视频（最长 1 小时）。

### 2.7 InternVL 系：scale up vision tower

LLaVA / Qwen-VL 的 vision encoder 都是几百 M 参数的 CLIP-ViT-L。**InternVL 的不同思路**：把 vision encoder 也 scale 到 LLM 量级——**InternViT-6B**（vs CLIP-L 304M，大 20×）。

核心 insight：传统 VLM 的 vision encoder 与 LLM 参数严重不平衡（300M vs 7B），vision 端是瓶颈。把 vision encoder scale 起来后能学到更强的视觉表征，下游 VLM 任务整体提升。

InternVL2 / InternVL3 的关键 ingredient：

- **InternViT-6B**：自训的大 vision encoder，超过 CLIP-L
- **Pixel shuffle**：$2 \times 2$ 邻域合并把 vision token 数 $\div 4$
- **Multi-stage training**：先 ViT-LLM contrastive 对齐，再 SFT 加 instruction tuning，最后 RLHF 调 hallucination

InternVL2/3 在 MMMU、MathVista、ChartQA 等多个 VLM benchmark 上长期 leader，是开源 VLM 的另一条 SOTA 路线（与 Qwen2.5-VL 并列）。

### 2.8 方案对比表

| 方法 | vision encoder | projector | 是否动态分辨率 | vision token 数（典型） | 代表 |
|---|---|---|---|---|---|
| LLaVA-1.5 | CLIP ViT-L | 2-layer MLP | 否（336²） | 576 | LLaVA-1.5 |
| LLaVA-1.6 (NeXT) | CLIP ViT-L | 2-layer MLP | AnyRes（4 tile） | 2304 | LLaVA-NeXT |
| LLaVA-OneVision | SigLIP | MLP | 是 | 动态 | LLaVA-OV |
| BLIP-2 | EVA-CLIP-g | Q-Former (32 query) | 否 | 32 | BLIP-2 |
| MiniCPM-V | SigLIP | Resampler (64-96 query) | 是（slice） | 64-256 | MiniCPM-V 2.6 |
| Qwen2-VL / 2.5-VL | DFN ViT (675M) | Patch merger (2×2) | 是（native） | 动态 | Qwen2.5-VL |
| InternVL2 / 3 | InternViT-6B | Pixel shuffle + MLP | 部分（tile） | 256-1024 | InternVL2/3 |

---

## 3. 最小代码示例

### 3.1 CLIP zero-shot classification（OpenCLIP，< 25 行）

```python
# pip install open_clip_torch torch torchvision
import torch
import open_clip
from PIL import Image

# 加载 OpenCLIP 预训权重（也可换 SigLIP-Base）
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-L-14", pretrained="openai")          # CLIP-ViT-L/14, 4 亿对预训
tokenizer = open_clip.get_tokenizer("ViT-L-14")
model.eval()

image = preprocess(Image.open("cat.jpg")).unsqueeze(0)   # (1, 3, 224, 224)
labels = ["a photo of a cat", "a photo of a dog", "a photo of a car"]
text = tokenizer(labels)                                  # (3, 77) token ids

with torch.no_grad():
    img_feat = model.encode_image(image)                  # (1, 768)
    txt_feat = model.encode_text(text)                    # (3, 768)
    img_feat /= img_feat.norm(dim=-1, keepdim=True)       # L2 normalize
    txt_feat /= txt_feat.norm(dim=-1, keepdim=True)
    probs = (100.0 * img_feat @ txt_feat.T).softmax(-1)   # (1, 3)

print({l: round(p.item(), 3) for l, p in zip(labels, probs[0])})
# 期望: {"a photo of a cat": 0.98, "a photo of a dog": 0.01, "a photo of a car": 0.01}
```

**关键点**：

- `encode_image` / `encode_text` 都已包含 final projection 到共享 embedding 空间
- 必须 L2 normalize 后再点积——CLIP 训练时就用 cosine similarity
- $\times 100$ 是 $1/\tau$ 的近似（CLIP learnable τ ≈ 0.01），让 softmax 不至于太平
- zero-shot：候选 label 没在训练里见过，靠的是 image / text embedding 在共享空间的对齐

### 3.2 简化 LLaVA 架构（< 35 行 PyTorch）

```python
import torch
import torch.nn as nn
from transformers import CLIPVisionModel, AutoModelForCausalLM, AutoTokenizer

class MiniLlava(nn.Module):
    def __init__(self, vision_name="openai/clip-vit-large-patch14-336",
                 llm_name="lmsys/vicuna-7b-v1.5"):
        super().__init__()
        # 1. Vision encoder（frozen）
        self.vision = CLIPVisionModel.from_pretrained(vision_name)
        for p in self.vision.parameters():
            p.requires_grad = False                              # Stage 1 freeze
        d_vis = self.vision.config.hidden_size                   # 1024 for ViT-L

        # 2. LLM
        self.llm = AutoModelForCausalLM.from_pretrained(llm_name,
            torch_dtype=torch.bfloat16)
        d_llm = self.llm.config.hidden_size                      # 4096 for Vicuna-7B

        # 3. Projector：2 层 MLP（LLaVA-1.5 配方），唯一 Stage 1 可训部件
        self.projector = nn.Sequential(
            nn.Linear(d_vis, d_llm),
            nn.GELU(),
            nn.Linear(d_llm, d_llm),
        )

    def forward(self, pixel_values, input_ids, labels=None):
        # 1. 图 → vision feature → 投到 LLM token 空间
        vis_out = self.vision(pixel_values).last_hidden_state    # (B, N_vis, d_vis)
        vis_out = vis_out[:, 1:]                                  # 去掉 [CLS]
        vis_emb = self.projector(vis_out)                         # (B, N_vis, d_llm)

        # 2. text id → LLM embed
        txt_emb = self.llm.get_input_embeddings()(input_ids)      # (B, T_txt, d_llm)

        # 3. concat 成 (B, N_vis + T_txt, d_llm) 后送 LLM
        inputs = torch.cat([vis_emb, txt_emb], dim=1)
        return self.llm(inputs_embeds=inputs, labels=labels)
```

**关键点**：

- vision encoder `requires_grad = False`——Stage 1 + Stage 2 都 freeze 是 LLaVA 默认；只 unfreeze projector（Stage 1）或 projector + LLM（Stage 2）
- `vis_out[:, 1:]`：CLIP-ViT 第 0 个 token 是 [CLS]，VLM 通常用 patch tokens（不要 [CLS]）
- vision token 与 text token concat 后整体送 LLM——注意 vision token 在前、text 在后是 LLaVA 约定，但有的实现是 `<image>` placeholder + text 按位置插入
- 真实 LLaVA 还要处理 image placeholder token、attention mask、labels 的 -100 mask（vision token 不算 loss），以上是简化版

### 3.3 HuggingFace 推理：LLaVA / Qwen2.5-VL（< 25 行）

```python
# pip install transformers pillow
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch
from PIL import Image

# === 选 1：LLaVA-1.5 ===
# model_id = "llava-hf/llava-1.5-7b-hf"

# === 选 2：Qwen2.5-VL（推荐 2024+ 任务）===
model_id = "Qwen/Qwen2.5-VL-7B-Instruct"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForVision2Seq.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto")

image = Image.open("chart.png")
messages = [{"role": "user", "content": [
    {"type": "image", "image": image},
    {"type": "text", "text": "What's the main trend in this chart?"}]}]

# 关键：每个 VLM 的 chat template / image token 处理都不一样，必须用对应 processor
text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda")

out_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
print(processor.batch_decode(out_ids, skip_special_tokens=True)[0])
```

**关键点**：

- 不同 VLM 的 input format 完全不同（LLaVA 用 `<image>` placeholder、Qwen-VL 用 `<|vision_start|>...<|vision_end|>`、InternVL 用自己的 `<img></img>`）—— 必须用对应 `processor`，**不要手拼 prompt string**
- `apply_chat_template` 自动加 system / user / image token 等标记
- bf16 + `device_map="auto"` 可以让 7B VLM 在单张 24G GPU 上跑

---

## 4. 工程踩坑与经验

- ❗ **Vision encoder 选错 → 性能塌**。VLM 的"上限"严重受 vision encoder 限制——用 CLIP-ViT-B（base, 86M）替代 CLIP-ViT-L/14 (304M) 直接掉 5+ 点；用纯 ImageNet 预训的 ViT 替换 CLIP（无文本对齐）能掉 10+ 点。**结论**：现代 VLM 的 vision tower 必须是 CLIP-family（CLIP / SigLIP / EVA-CLIP / DFN-CLIP），且至少 ViT-L 量级；2024+ 主流是 SigLIP-SO400M（400M 参数、native 支持高分辨率）。
- ❗ **Projector 是 alignment 关键，但简单 MLP 够用**。LLaVA 早期就比较过 Linear / 2-MLP / Q-Former / Resampler——**2 层 MLP 与 Q-Former 性能基本持平，前者参数少 10×、训练快 3×**。Q-Former / Perceiver Resampler 的复杂度只在你需要"固定 vision token 数"时才有意义（极长 context、多张图）。一般场景**直接 2 层 MLP**——简单粗暴够用。BLIP-2 的复杂 Q-Former 是历史遗产。
- ❗ **高分辨率 image 的 vision token 数巨大，必须做 mitigation**。1024×1024 / patch 14 = 5184 token，2048 直接 21316 token——一张高分图就吃光 LLM context。**必须用** dynamic resolution（Qwen2-VL）/ pixel shuffle（InternVL）/ AnyRes（LLaVA-NeXT）/ Resampler（MiniCPM-V）之一。如果你 fine-tune VLM 不调这些机制，OCR / chart / document 类任务会卡死或 OOM。开源默认 config 里 `max_pixels` / `min_pixels` / `use_pixel_shuffle` 是关键超参。
- ❗ **Visual SFT 数据 quality 决定 hallucination 率（VLM hallucination 比 text 严重）**。LLM 文本幻觉已经够头疼，VLM 因为 vision-text alignment 不完美会**编造图里没有的对象**——典型如"图里有几个人"答错、看不见的细节硬编。原因：早期 visual SFT 数据（如 LLaVA-Instruct）是 GPT-4 看 caption 编出来的（GPT-4 当时也看不见图），数据本身就有 hallucination。**对策**：用更高质量的人工标注数据（ShareGPT-4V、Cambrian-1 数据集）、加 visual RLHF（POVID、RLHF-V）、加 OCR / detection 类数据增强 grounding。VLM 的 hallucination 是面试与工业落地都 care 的硬伤。
- ❗ **Vision encoder 是否 freeze 影响很大**。早期 stage 必须 freeze——projector 还没对齐就 unfreeze vision encoder 会让 CLIP 学好的视觉表征被冲掉。**典型策略**：Stage 1（projector pretrain）freeze vision；Stage 2（visual SFT）通常仍 freeze vision，只解锁 projector + LLM；**Stage 3（fine-grained / domain SFT）才考虑 unfreeze vision encoder**——且只 unfreeze 最后 2-4 层（保护底层 generic feature）。盲目"全 unfreeze 一起训"是新手最常见的 setup error，几个 step loss 就崩。
- ❗ **Multi-image / video 输入要按特殊顺序 token 化，与 RoPE 对齐**。Qwen2-VL 的 3D RoPE 期望 image 是 `(1, h, w)`、video 是 `(T, h, w)`，patch token 要按 `(t, h, w)` flatten 顺序送入 LLM——**flatten 顺序错（如把 w 放最外层）会让 RoPE 旋转角度对不上，模型直接退化**。多图场景还要在不同图之间插入 image separator token（不同 VLM 用法不一样：LLaVA 用 `\n`、Qwen-VL 用 `<|vision_pad|>`）。一定要看官方 processor 源码确认序列化格式。
- ❗ **HF transformers 不同 VLM 的 input format 完全不同**。LLaVA 是 `<image>\n{user_text}`、Qwen2-VL 是 `<|vision_start|><|image_pad|><|vision_end|>{user_text}`、InternVL 是 `<img><IMG_CONTEXT></img>{user_text}`、MiniCPM-V 是 `(<image>./</image>){user_text}`。**手拼 prompt 几乎必踩坑**——image placeholder 写错就导致 vision token 没被正确替换，模型当成纯文本输出（看起来在"回答"但完全没看图）。**永远用 `processor.apply_chat_template`**，不要 hardcode prompt string。
- ❗ **Vision token 不参与 loss 计算（labels 要 mask）**。SFT 时 labels 通常是 `input_ids` 的 shift；但 vision token 位置不应算 cross-entropy（它们是 image embedding，不是 text id）。LLaVA / Qwen-VL 的训练代码都会把 vision token 对应位置的 label 设为 `-100`（`F.cross_entropy(ignore_index=-100)` 跳过）。**漏 mask 会让 loss 试图"预测 vision token 内容"——这件事根本无意义，loss 数值乱跳、训练不收敛**。
- ❗ **bf16 / fp16 下 vision encoder 可能数值不稳**。CLIP-ViT 在大 batch + fp16 下偶发 NaN（attention logits 爆掉）。建议 vision encoder 跑 bf16（动态范围大于 fp16），LLM 也跑 bf16，projector 的 final linear cast 到 fp32 算 loss。Qwen-VL 官方推理 demo 用 bf16 + flash-attn-2 能稳定跑大部分场景。

---

## 5. 经典 paper

- **Radford et al., 2021 — Learning Transferable Visual Models From Natural Language Supervision (CLIP)** — 必读的"对比学习对齐 image-text"奠基作。读 §2 Approach + §3.1 Zero-shot Transfer 足以——理解了 CLIP 的双 encoder + InfoNCE 范式，就能看懂所有现代 VLM 的 vision tower 选择动机。本节 §2.1 公式直接复述这篇 §2.5。
- **Liu et al., 2023 — Visual Instruction Tuning (LLaVA) / Improved Baselines with Visual Instruction Tuning (LLaVA-1.5)** — 必读的"接接器范式"开山作。两阶段训练流程（projector pretrain → visual SFT）+ 简洁 MLP projector + GPT-4 生成 visual instruction 的范式被后续所有开源 VLM 沿用。本节 §2.3 + §3.2 的代码骨架直接来自这两篇 paper 的实现。
- **Bai et al., 2024 — Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution** — 必读的"动态分辨率 + 3D RoPE"代表作。读 §2 Methodology 一节即可——理解 native resolution 怎么实现、3D RoPE 怎么扩展，就掌握了 2024+ 最重要的 VLM 工程升级。Qwen2.5-VL 是其后续，整体架构相同。
- **Chen et al., 2024 — InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks** — 必读的"scale up vision tower"代表作。InternViT-6B 论证了"VLM 的 vision encoder 也要大"，与 LLaVA / Qwen 系的"小 vision + 大 LLM"路线形成鲜明对比。读 §3 Architecture 与 §4 Training 即可。
- **Li et al., 2023 — BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models** — Q-Former 范式的代表作。即便现代主流不再用 Q-Former，理解它对"为什么后来选了 MLP projector"很重要。读 §3 Method 即可。
- **Zhai et al., 2023 — Sigmoid Loss for Language Image Pre-training (SigLIP)** — 加分必读。理解 sigmoid 替代 softmax 后 contrastive 训练为什么更稳、更易 scale——是现代 VLM vision tower 选 SigLIP 的根本原因。

---

## 6. 自测与面试题

**Q1（架构）**：写出 LLaVA 三件套架构（vision encoder + projector + LLM），并描述两阶段训练流程。每个阶段哪些参数 frozen / 哪些 trainable？为什么要分两阶段？

<details>
<summary>Answer sketch</summary>

架构（必须画出三件套数据流）：

```
image (3,H,W) → Vision Encoder (CLIP-ViT-L, frozen) → vision feature (N_vis, d_vis=1024)
                                                              │
                                                              ▼
                                                       Projector (2-layer MLP)
                                                              │
                                                              ▼
                                                       vision embedding (N_vis, d_llm=4096)
                                                              │
                                                       concat with
                                                       text embedding (T_txt, d_llm)
                                                              │
                                                              ▼
                                                       LLM (Vicuna/LLaMA-2)
                                                              │
                                                              ▼
                                                       text response
```

两阶段训练（必背表）：

| 阶段 | 数据 | Trainable | Frozen | 目的 |
|---|---|---|---|---|
| Stage 1 (Pre-training / Alignment) | 558k caption pair | **只 projector** | vision + LLM | 把 vision feature 投到 LLM token space |
| Stage 2 (Visual Instruction Tuning) | 150k visual instruction (LLaVA-Instruct) | **projector + LLM** | vision encoder | 教 LLM 怎么用 vision 信息回答 |

为什么分两阶段？

- Stage 1：projector 是新初始化的小网络，输出与 LLM 期望的 token embedding 完全不对齐——直接和 LLM 一起 fine-tune 会让 LLM 被未对齐的 vision feature 污染，几个 step 就崩。先单独训 projector 把"翻译接口"焊牢，loss 收敛后再进 Stage 2。
- Stage 2：projector 已对齐，可以解锁 LLM 一起 SFT。vision encoder 仍 freeze——CLIP 学好的视觉表征是宝贵预训资产，过早 unfreeze 容易冲掉。

加分：

- 能说出 LLaVA-1.0 用单 Linear、LLaVA-1.5 升级到 2 层 MLP（GELU）
- 能说出 LLaVA-1.6 (NeXT) 加了 AnyRes 支持高分辨率
- 能说出现代变体（Stage 3 fine-grained）才会 unfreeze vision encoder 的最后几层

</details>

**Q2（trade-off）**：高分辨率图能提升 OCR / chart / document 任务，但 vision token 数会爆——一张 1024×1024 / patch 14 图有多少 token？说出业界 3 种 mitigation 方法及代表 VLM。

<details>
<summary>Answer sketch</summary>

- token 数计算：$\lceil 1024/14 \rceil^2 = 74^2 = 5476$（或按 73 算 = 5329，patch=14 时 1024 不整除取上下界都对）
- 一张 1024×1024 图就吃 ~5k token，几张图 + 长 prompt 直接超过 8k context

**3 种 mitigation 方法**（业界主流）：

| 方法 | 原理 | 代表 VLM |
|---|---|---|
| **Pixel shuffle** | 把 $2 \times 2$ 邻域的 vision token 沿 channel 维拼接 → token 数 ÷ 4，channel × 4，空间信息保留在 channel | InternVL2 / 3 |
| **Dynamic resolution / Patch merger** | 不强制 resize，按原图实际宽高动态切 patch；相邻 token 用 patch merger MLP 合并 4→1 | Qwen2-VL / 2.5-VL |
| **Q-Former / Resampler** | learnable query 通过 cross-attention 抽取固定数量 token，与分辨率无关 | BLIP-2 / MiniCPM-V / Idefics |
| **AnyRes / Tile** | 高分图切成多个 tile，每 tile 独立过 vision encoder 再拼接；保留细节但 token 仍随 tile 数增长 | LLaVA-NeXT / OneVision |

加分：

- 能说出"pixel shuffle 比 Q-Former 好"——pixel shuffle 是无损的（信息搬到 channel），Q-Former 是有损（信息瓶颈）
- 能说出 Qwen2-VL 的 max_pixels 参数控制单张图最大 token 数（默认 1280×28×28 ≈ 1280 patch）
- 能讨论 token pruning / merging（学术）vs 工程主流的差异

</details>

**Q3（实战）**：你要 fine-tune 一个 VLM 做"图表理解（chart / table 问答）"，base model 在 Qwen2.5-VL-7B / InternVL2-8B / LLaVA-OneVision-7B 三选一，给出选择与理由。

<details>
<summary>Answer sketch</summary>

**优先选 Qwen2.5-VL-7B 或 InternVL2-8B**（任选其一都合理，需说出权衡）。理由要从以下几个维度展开：

- **分辨率支持**：图表理解的关键是清晰看到 axis label / legend / 数字——必须高分辨率。
  - Qwen2.5-VL：native dynamic resolution，按原图自适应，理论支持任意大小（最大 max_pixels 控制）。**最适合 chart**——不会 resize 失真
  - InternVL2：tile + pixel shuffle，可支持高分辨率 + 控制 token 数
  - LLaVA-OneVision：AnyRes 4-9 tile，分辨率支持次于前两个

- **OCR / document benchmark**：
  - Qwen2.5-VL 在 ChartQA / DocVQA / OCRBench 上是 2024 末开源 SOTA（特别是 OCR 强化数据训练）
  - InternVL2 在 ChartQA / MathVista 上也是 leader，整体均衡
  - LLaVA-OneVision 通用更平衡，OCR 不是其强项

- **Fine-tune 友好度**：
  - LLaVA-OneVision / InternVL 开源 training code 完整，HF Trainer 友好
  - Qwen2.5-VL 官方有 ms-swift / Qwen-VL-Finetune，但 dynamic resolution 让 batching 复杂（每张图 token 数不一样，padding 策略要设计）

- **基础 LLM 质量**：Qwen2.5-7B 与 InternLM2.5-7B 都强于 Vicuna 系（LLaVA 默认）。

**结论建议**：
- 如果数据量大、有工程能力调 batching → **Qwen2.5-VL-7B**（OCR / chart 第一梯队 + native resolution 最适配）
- 如果想训稳、用现成 fine-tune pipeline → **InternVL2-8B**（pixel shuffle 控制 token 数稳定，HF 集成好）
- 不建议选 LLaVA-OneVision——通用强但 chart-specific 弱于前两者

加分：

- 能讨论"是否需要再加 SFT 数据扩 chart-specific instruction"（如 ChartQA-Plus、PlotQA）
- 能提到 LoRA fine-tune VLM 时只 LoRA LLM 部分（不 LoRA vision encoder 与 projector）
- 能提到评估用 ChartQA-Pro / OCRBench / DocVQA 而不是 MMMU（后者是通用 multimodal 推理）
- 能提到加 visual RLHF / DPO 减少 hallucination（chart 数字读错是高频幻觉）

</details>

---

## 7. 延伸阅读

- [Liu et al. — LLaVA paper (arXiv 2304.08485)](https://arxiv.org/abs/2304.08485) — LLaVA 原论文，必读 §3 Visual Instruction Tuning
- [LLaVA GitHub (haotian-liu/LLaVA)](https://github.com/haotian-liu/LLaVA) — LLaVA 1.5/1.6 完整训练 + 推理代码，本节 §3.2 简化版的 production 参照
- [Qwen2.5-VL 技术报告](https://qwenlm.github.io/blog/qwen2.5-vl/) — Qwen2.5-VL 详细架构与 benchmark，了解 native resolution + 3D RoPE
- [InternVL GitHub (OpenGVLab/InternVL)](https://github.com/OpenGVLab/InternVL) — InternVL2/3 完整代码 + pixel shuffle 实现细节
- [OpenCLIP GitHub (mlfoundations/open_clip)](https://github.com/mlfoundations/open_clip) — CLIP / SigLIP 开源实现，本节 §3.1 代码所用
- [HuggingFace Transformers — VLM 模型清单](https://huggingface.co/docs/transformers/model_doc/llava) — LLaVA / Qwen2-VL / InternVL2 / Idefics 在 HF 的 API 文档与 processor 用法
- [Cambrian-1 paper (NYU)](https://arxiv.org/abs/2406.16860) — 系统比较 20+ vision encoder 在 VLM 中的效果，强烈推荐研究 vision tower 选型必读
- 推荐继续读本教程的 **16.2 节《Native multimodal：Chameleon / GPT-4o / Gemini》**——early-fusion VLM 与本节"接接器"范式的根本对比
- 推荐继续读本教程的 **16.4 节《Embedding：bge / E5 / Instructor / NV-Embed》**——CLIP 的对比学习思路在文本 / 多模态 embedding 上的延续
- 推荐继续读本教程的 **16.5 节《Computer Use & GUI Agent》**——VLM 在 GUI 场景的真实落地（OSWorld / SeeClick / UI-TARS）
