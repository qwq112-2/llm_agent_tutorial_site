---
title: "8.3 LoRA / QLoRA / DoRA 原理与实现"
description: "把 fine-tune 时的 $\\\\Delta W$ 假设成低秩矩阵 $BA$，只训练这两小块、把 base model 冻死——这一个朴素假设撑起了 LoRA / QLoRA / DoRA 三件套，让单卡 24GB GPU 能 fine-tune 70B 模型，是 2026 年所有 SFT / DPO / GRPO 实战的事实标配。"
---

> ⏱ 预计阅读 50 分钟 ｜ 难度 ★★★ ｜ 前置：1.1 反向传播、4.5 FFN（理解 LoRA 加在哪几个 linear 上）

## 一句话本节讲什么

把 fine-tune 时的 $\Delta W$ 假设成低秩矩阵 $BA$，只训练这两小块、把 base model 冻死——这一个朴素假设撑起了 LoRA / QLoRA / DoRA 三件套，让单卡 24GB GPU 能 fine-tune 70B 模型，是 2026 年所有 SFT / DPO / GRPO 实战的事实标配。

---

## 1. Mental model（直觉）

Full fine-tune 一个 LLaMA-3 70B 模型需要多少显存？

- 模型权重 fp16：140 GB
- 梯度 fp16：140 GB
- AdamW 一阶矩 fp32：280 GB
- AdamW 二阶矩 fp32：280 GB
- 加 forward activation、optimizer overhead——**总共 ~1 TB**

8 张 H100 80GB 也只有 640 GB，**普通团队根本玩不起**。问题来了：fine-tune 阶段我们真的需要更新所有 700 亿参数吗？

[Aghajanyan 2020] 的回答是 **no**——language model 在 fine-tune 时的"内禀维度（intrinsic dimension）"远小于参数数量，常常只需更新一个几百到几千维子空间就能匹配 full fine-tune 的效果。Hu et al. 2021 的 LoRA 把这个观察落地成一行假设：

> **Fine-tune 引入的权重变化 $\Delta W$ 是低秩的，可以分解为两个小矩阵 $\Delta W = BA$。**

冻住原 $W$，只训这俩小矩阵——参数量从 $d^2$ 降到 $2rd$（$r \ll d$），训练显存从"权重 + 梯度 + optimizer state"全套大缩水到只剩 LoRA 那一小撮。把心智模型画出来：

```
        ┌──────────────────┐
   x ──►│   W (frozen)     │──► Wx ───┐
        │   d × d          │          │
        └──────────────────┘          ▼
        ┌─────┐    ┌─────┐                     ┌────┐
   x ──►│ A   │──► │ B   │──► (α/r)·BAx ──►(+) │ y  │
        │r×d  │    │d×r  │                     └────┘
        └─────┘    └─────┘
       trainable  trainable
       (kaiming)  (zeros)
```

三条心智锚点，本节后面所有内容都围绕它们：

1. **LoRA 是"加法旁路"**：$y = Wx + (\alpha/r) \cdot BAx$，base 这条主路径完全不动，旁路 $BA$ 起初为 0（B 用 0 初始化），训完合并回 W 即可，**推理时零 overhead**
2. **QLoRA 是"把主路径压扁"**：base $W$ 用 NF4 4-bit 量化（省 4 倍显存），旁路 LoRA 仍用 bf16 全精度训练。base 不更新所以量化无所谓
3. **DoRA 是"把 W 拆成长度 + 方向"**：方向部分仍用 LoRA 调整，单独再训一个 magnitude 向量——参数几乎一样，但学习容量更接近 full fine-tune

工程上，PEFT（Parameter-Efficient Fine-Tuning）家族不止 LoRA，还有 Adapter / Prefix-tuning / P-tuning v2（详见 8.4）。但**LoRA 是事实标准**：HuggingFace `peft` 库、TRL、DeepSpeed、unsloth、Axolotl，所有训练框架都把 LoRA 当一等公民；vLLM、SGLang、TGI 都支持 multi-LoRA serving；DPO / GRPO / RLHF 实战中也几乎都先在 LoRA 上做。本节聚焦 LoRA + QLoRA + DoRA 三个最重要的成员。

---

## 2. 公式与原理

### 2.1 LoRA：低秩分解

设原 linear 层权重 $W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$（PyTorch 约定 `nn.Linear` 的 `weight` 是 `(out, in)`），fine-tune 后变为 $W' = W + \Delta W$。LoRA 假设 $\Delta W$ 低秩，引入两个小矩阵：

$$
\Delta W = B A, \quad A \in \mathbb{R}^{r \times d_\text{in}}, \; B \in \mathbb{R}^{d_\text{out} \times r}
$$

其中 $r \ll \min(d_\text{in}, d_\text{out})$ 是**秩（rank）**，是 LoRA 唯一的核心超参。前向传播：

$$
y = W x + \Delta W x = W x + \frac{\alpha}{r} \cdot B A x
$$

训练时 $W$ freeze，只训 $A, B$。可训练参数从 $d_\text{out} \cdot d_\text{in}$ 降到 $r \cdot (d_\text{in} + d_\text{out})$。以 LLaMA-2 7B（$d=4096$）为例：

- 单个 attention $W_q$（4096 × 4096）原本 16.78M 参数
- LoRA $r=16$：$16 \times (4096+4096) = 131072$ 参数 = 0.13M，**省 128 倍**
- 全模型 7 个 linear（q、k、v、o、gate、up、down）× 32 层全加 LoRA $r=16$：约 4M 参数（占 7B 的 **0.06%**）
- 同设置 $r=64$：约 17M 参数（**0.25%**）

### 2.2 缩放因子 $\alpha$ 的物理含义

公式里的 $\alpha / r$ 这个系数容易被新人误以为是"学习率的替代品"，其实是 **rank-aware scaling**——为了让更换 $r$ 时**不需要重新调学习率**。

直观推导：$BAx$ 输出方差大致正比于 $r$（Kaiming-A、零-B 初始化下，$A$ 每一行方差固定，B 慢慢学起来；学到的 $\Delta W$ 实际有效"贡献"随 $r$ 增长）。乘以 $1/r$ 就把 $r$ 维度上的标度归一化掉。**实务上**：

- Hu 原 paper 推荐 $\alpha = r$，比值 $\alpha/r = 1$
- HuggingFace PEFT 默认 $\alpha = 2r$（比值 = 2），更激进、常用配置
- 经验等价关系：**$\alpha=32, r=8$ 与 $\alpha=64, r=16$ 等价**（都是 $\alpha/r = 4$ 的有效 lr scaling）；这是 $\alpha$ 解耦的本意

后续 [Kalajdzievski 2023 — rsLoRA] 指出：当 $r$ 很大时 $\alpha/r$ 这个 scaling 导致梯度幅度偏小，建议改用 $\alpha / \sqrt{r}$（rank-stabilized LoRA，rsLoRA）。HuggingFace PEFT 通过 `use_rslora=True` 启用。

### 2.3 初始化：A Kaiming，B 零

至关重要的细节：**$B$ 必须用 0 初始化**，$A$ 用 Kaiming uniform（PyTorch `Linear` 默认）。

为什么？看 $\Delta W = BA$：

- 若 $B = 0$，则 $\Delta W = 0$，LoRA 起步与 base model 完全等价 ——"零起点" 安全启动
- 训练第一步反向时，$\frac{\partial L}{\partial A} = B^\top \frac{\partial L}{\partial(BA \cdot x)} \cdot x^\top$，但 $B = 0$ 让这一项为 0；而 $\frac{\partial L}{\partial B} = \frac{\partial L}{\partial(BA \cdot x)} \cdot (Ax)^\top$ 不为 0（因为 $A \neq 0$）—— **B 先动起来**，然后 A 也开始接收非零梯度

如果反过来——A=0、B Kaiming——会发生什么？$\Delta W = B \cdot 0 = 0$ 仍然成立、起步依然安全，但 $\frac{\partial L}{\partial B} = 0 \cdot \cdots = 0$，而 $\frac{\partial L}{\partial A}$ 不为零——只有 A 动 B 不动，对称性破缺；反过来更稳定一些。**HuggingFace PEFT 用的就是 A Kaiming + B 零**。

如果两个都 Kaiming——$\Delta W \neq 0$，base model 起步就被推飞，loss spike 几乎必然发生。**LoRA 实现的 #1 经典 bug** 就是写反这两个初始化。

### 2.4 QLoRA：把 base 量化到 4-bit

[Dettmers 2023 — QLoRA] 把 LoRA 的"省显存"再推进一大步：**把 base model 用 NF4 量化到 4-bit，LoRA 仍 bf16 训练**。

显存账本（7B 模型）：

| 组件 | LoRA (bf16 base) | QLoRA (NF4 base) |
|---|---|---|
| Base weight | 14 GB (bf16) | **3.5 GB** (NF4) |
| LoRA weight (r=16) | 0.04 GB | 0.04 GB |
| LoRA gradient | 0.04 GB | 0.04 GB |
| AdamW state（仅 LoRA） | 0.16 GB | 0.16 GB |
| Activation | ~3 GB | ~3 GB |
| **总计** | ~17 GB | **~7 GB** |

相同道理，70B QLoRA 把 base 从 140GB 压到 35GB——**单张 H100 80GB 就能 SFT 70B**，是 2023 之后开源 fine-tune 民主化的最大推手。QLoRA 的三个关键 trick：

**(1) NF4（NormalFloat 4-bit）量化**

普通 4-bit 量化把权重均匀切 16 个 bin（INT4），但**LLM 权重经验上服从近似正态分布**——均匀切 bin 在密度高的 0 附近浪费精度。NF4 基于"权重 ~ $\mathcal{N}(0, \sigma^2)$"假设，让 16 个量化点在标准正态分布的等概率分位数上：

$$
q_i = \Phi^{-1}\!\left(\frac{i}{17}\right), \quad i = 1, \dots, 15, \text{ 加 0 共 16 个值}
$$

直观效果：0 附近的 bin 密、远离 0 的 bin 疏；同样 4 bit 比 INT4 量化误差小 ~30%。NF4 是**理论最优**的"假设权重正态分布下的"4-bit 量化。

**(2) Double Quantization（双重量化）**

NF4 量化时每 64 个权重共用一个 fp32 的 scale 因子（quantization constant），平均每参数引入 $32/64 = 0.5$ bits 额外开销。Double Quant 把这堆 scale 因子**自身再量化一次**（用 fp8），把 0.5 bits 压到 ~0.13 bits，全模型省 0.4 bits/param——**7B 模型省 350 MB 显存**，看似小，但配合 4-bit 已经是边际优化。

**(3) Paged Optimizers**

CUDA 11+ 提供 unified memory：optimizer state 放在 GPU 内存里，但 OOM 时**自动 page-out 到 CPU 内存**，下次访问时再 page-in。bitsandbytes 的 `PagedAdamW` 把 AdamW 的 m / v 状态包装成 paged tensor，遇到长序列 spike 时不会直接 OOM 而是平滑降速。**实战中训长序列时几乎必开**。

### 2.5 DoRA：拆出 magnitude / direction

[Liu 2024 — DoRA] 观察到：full fine-tune 学到的 $\Delta W$ 在"长度（magnitude）"和"方向（direction）"上的变化模式与 LoRA 学到的有显著差异——LoRA 倾向于 magnitude 与 direction 同步变化，full fine-tune 则常常只动方向不动长度（或反之）。这种"耦合"限制了 LoRA 的学习容量。

DoRA 的解法：**把 $W$ 显式分解成 magnitude vector $m$ + direction matrix $V$**：

$$
W = m \cdot \frac{V}{\|V\|_c}
$$

其中 $\|V\|_c$ 是按列（column）的 L2 范数，$m \in \mathbb{R}^{d_\text{out}}$ 是每列的标量长度，$V / \|V\|_c$ 是单位方向矩阵。fine-tune 时：

- **方向 $V$** 用 LoRA 形式更新：$V' = V + BA$
- **长度 $m$** 单独训练为可学习向量

合起来：

$$
W' = m \cdot \frac{V + BA}{\|V + BA\|_c}
$$

参数量比 LoRA 多一个 $d_\text{out}$ 维向量（几乎可以忽略），实证在多个 SFT / commonsense reasoning benchmark 上比 LoRA **稳定涨 1-3 个百分点**，甚至接近 full fine-tune。HuggingFace PEFT 通过 `use_dora=True` 启用，工程改动几乎为零。

代价：训练显存略增（多存 $m$ 与归一化中间量）、计算多 ~10%；分布式训练时 column-norm 是个 reduction 操作要小心。

### 2.6 其他变体一句话扫过

- **rsLoRA** [Kalajdzievski 2023]：把 $\alpha/r$ 改成 $\alpha/\sqrt{r}$，让大 $r$ 时梯度尺度更稳；HuggingFace 一行 `use_rslora=True`
- **VeRA** [Kopiczko 2024]：所有层共享同一个随机冻结的 A、B，每层只训 scaling 向量；参数量再降 10 倍；适合 multi-task / 个性化场景
- **GaLore** [Zhao 2024]：把**梯度**投影到 low-rank 子空间（不是把 weight 分解），可以做 full-parameter 更新但显存接近 LoRA；预训练阶段也能用
- **AdaLoRA / IncreLoRA**：自适应分配 rank，重要层 r 大、不重要层 r 小；Pareto 更优但工程复杂

主流仍是 **LoRA + QLoRA + DoRA**，其它变体面试可点名。

---

## 3. 最小代码示例

### 3.1 手撕 LoRALinear（≤ 30 行）

```python
import math
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    """把现有 nn.Linear 包成 LoRA 版：base frozen + 旁路 BA。"""

    def __init__(self, base_layer: nn.Linear, r: int = 16,
                 alpha: int = 32, dropout: float = 0.05):
        super().__init__()
        self.base = base_layer                              # 原 W 作为 frozen 主路径
        for p in self.base.parameters():
            p.requires_grad = False                         # 关键：base 不更新

        in_d = base_layer.in_features
        out_d = base_layer.out_features
        # PyTorch nn.Linear.weight: (out, in)，A 和 B 形状要匹配
        self.lora_A = nn.Parameter(torch.zeros(r, in_d))    # (r, in)
        self.lora_B = nn.Parameter(torch.zeros(out_d, r))   # (out, r)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B 保持 0 初始化 —— 让 ΔW 起步为 0，base 等价启动
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # base path：(B, T, in) -> (B, T, out)
        out = self.base(x)
        # LoRA path：x → dropout → A^T → B^T → 缩放
        lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return out + lora_out * self.scaling
```

关键点：

- `lora_A` 用 `kaiming_uniform_(a=sqrt(5))`，与 PyTorch `nn.Linear.weight` 默认初始化对齐
- `lora_B` 保留 `torch.zeros`——**写反就是经典 bug**
- `scaling = alpha / r`，前向乘进去；改 $r$ 不需要重调 lr
- dropout 加在输入侧（不是输出侧）—— PEFT 库的实现规范
- 训练时用 `model.named_parameters()` 过滤 `requires_grad=True` 的参数喂给 optimizer，自动只训 LoRA 部分

### 3.2 HuggingFace PEFT 完整 SFT 配置（≤ 30 行）

```python
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

base = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B", torch_dtype="bfloat16", device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

lora_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    bias="none", task_type=TaskType.CAUSAL_LM,
    # 现代主流：q/k/v/o + gate/up/down 全加，效果接近 full FT
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    # use_rslora=True,   # 需要 rsLoRA 时打开
    # use_dora=True,     # 需要 DoRA 时打开
)
model = get_peft_model(base, lora_cfg)
model.print_trainable_parameters()
# trainable params: 20,971,520 || all params: 8,051,232,768 || trainable%: 0.26

trainer = SFTTrainer(
    model=model, tokenizer=tokenizer,
    train_dataset=..., args=SFTConfig(
        output_dir="out", per_device_train_batch_size=2,
        gradient_accumulation_steps=8, learning_rate=2e-4,
        num_train_epochs=3, bf16=True, logging_steps=10,
    ),
)
trainer.train()
model.save_pretrained("out/lora_adapter")        # 只存 LoRA，~80 MB
```

关键点：

- `target_modules` 列名对应 LLaMA 系架构 `q_proj` 等；不同模型 module 名不同（Mistral 同 LLaMA、Qwen 是 `c_attn` / `c_proj` / `w1/w2/c_proj` —— 必查 `print(model)`）
- `bias="none"` 表示不训 bias；可选 `"lora_only"`（只训 LoRA 引入的 bias）或 `"all"`
- `lr=2e-4` 是 LoRA 经验起点，比 full fine-tune 的 1e-5 ~ 5e-5 大 4-20 倍——因为只训少量参数，可以放大 lr
- `save_pretrained` 只存 LoRA adapter（A、B、config），**不存 base**，文件几十 MB

### 3.3 QLoRA 加载 + 训练（≤ 25 行）

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",                  # NormalFloat 4-bit
    bnb_4bit_use_double_quant=True,             # double quant 再省 0.4 bits
    bnb_4bit_compute_dtype=torch.bfloat16,      # ❗ 必须 bf16，fp16 会 NaN
)

base = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    quantization_config=bnb_cfg, device_map="auto",
)
# 把 layer norm / lm_head 提到 fp32、关闭 cache、enable grad checkpoint
base = prepare_model_for_kbit_training(base)

lora_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(base, lora_cfg)
# 后续 SFTTrainer/Trainer 训练同 3.2，用 paged_adamw_8bit 优化器进一步省显存
```

显存：8B QLoRA + ckpt + batch=2 + seqlen=2048 在 24 GB RTX 4090 跑通；70B QLoRA 在单 80 GB H100 也跑得动。这是 QLoRA 的杀手级用例。

### 3.4 推理时 merge_and_unload（≤ 15 行）

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B", torch_dtype="bfloat16", device_map="auto",
)
model = PeftModel.from_pretrained(base, "out/lora_adapter")  # base + LoRA

# 关键一步：把 LoRA 合并回 base，得到一个普通 nn.Linear 模型
merged = model.merge_and_unload()
merged.save_pretrained("out/merged_model")
# 此后推理与普通 model 完全一样：W' = W + (alpha/r)·BA 已物化到 W
```

`merge_and_unload` 把 $W' = W + (\alpha/r) \cdot BA$ 显式算出来覆盖回 base.weight，模型变回纯 `nn.Linear` 结构，**推理时 LoRA 旁路完全消失，零 overhead**。如果要继续保留 multi-LoRA serving 能力，则**不要 merge**，保持 PEFT model 形态。

### 3.5 vLLM Multi-LoRA serving（≤ 15 行）

```python
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# base 只 load 一次，多个 LoRA 共享
llm = LLM(
    model="meta-llama/Meta-Llama-3-8B",
    enable_lora=True, max_loras=4, max_lora_rank=16,
)
sql_lora = LoRARequest("sql_adapter", 1, "out/lora_sql/")
math_lora = LoRARequest("math_adapter", 2, "out/lora_math/")

sampling = SamplingParams(temperature=0.0, max_tokens=128)
out_sql  = llm.generate(["SELECT * FROM ..."], sampling, lora_request=sql_lora)
out_math = llm.generate(["x^2 - 5x + 6 = 0"],  sampling, lora_request=math_lora)
# 同 batch 不同 sample 路由到不同 LoRA，base 显存只占一份
```

要点：所有 LoRA **必须同 rank**（`max_lora_rank=16` 表示上限），否则 batch 内 attention 不能合并 GEMM。多租户 SaaS 场景的标配。

---

## 4. 工程踩坑与经验

- ❗ **B 必须 0 初始化、A Kaiming，写反会让 base model 起步就漂**——这是 LoRA 实现的 #1 经典 bug。两个都 Kaiming 时 $\Delta W \neq 0$，loss 第一步就 spike；A=0、B Kaiming 也能起步等价但训练动力学不对称。HuggingFace PEFT 的标准做法：A Kaiming uniform + B zeros，照抄就行
- ❗ **`alpha / r` 是有效 lr scaling，不是独立超参**——`alpha=32, r=8` 与 `alpha=64, r=16` 在前向数值上等价（都是比值=4）。改 r 时如果想保持"相对学习强度"不变，等比例改 alpha；想增强 LoRA 表达力则只增 r 不动 alpha。新人常常分别调 alpha 和 r，等价于在调同一个量两次
- ❗ **`target_modules` 只加 q,v 与加全部 7 个 linear 效果差距很大**——LoRA 原 paper 只加 `q_proj, v_proj`（参数最省），但近 2 年实证：**加全部 q/k/v/o + gate/up/down 几乎与 full fine-tune 等效**，多花的 0.1% 参数完全值得。除非你在做极端低预算实验，**默认全加 7 个**
- ❗ **QLoRA 的 NF4 需要 bitsandbytes 库 + 兼容 GPU**——bitsandbytes 在 V100 / T4 / 部分 AMD 卡上不支持或退化；A100 / H100 / RTX 30/40/50 系都 OK。Windows 上 `bitsandbytes` 安装坑多，建议 WSL / Linux
- ❗ **QLoRA 的 `bnb_4bit_compute_dtype` 必须 bf16，fp16 容易 NaN**——NF4 反量化后矩阵乘的中间量动态范围大，fp16（指数 5 位）很容易上/下溢；bf16（指数 8 位）才稳。如果 GPU 不支持 bf16（V100、消费级 GTX），不要用 QLoRA，回到 LoRA + bf16/fp16 base
- ❗ **LoRA 训完保存只有几十 MB，但 load 时必须先 load base 再 load LoRA**——`PeftModel.from_pretrained(base, lora_path)` 这种顺序，反过来不行。新人最常见报错"找不到 base weight"就是直接 `from_pretrained(lora_path)`
- ❗ **LoRA + ZeRO-3 / FSDP 时，注意 LoRA 与 base 的 group 配置**——FSDP 默认按整个 nn.Module 分 shard，base 与 LoRA 在同一 module 时一起被 shard；想让 LoRA 留在 GPU 不被 shard 需要配 `auto_wrap_policy` 把 base 与 LoRA 拆开。DeepSpeed ZeRO-3 类似，需要在 `parameter_offload` 配置里指定 LoRA 不 offload。`accelerate` + `peft` 集成默认已经处理好，但自己手撕 FSDP 容易踩
- ❗ **LoRA rank 选择遵循 task 复杂度**——通用 chat / 风格对齐 r=8 ~ 16 够；code / math / 长 reasoning r=32 ~ 64；复杂多任务（GRPO 多 reward 头）r=64 ~ 128；超过 128 边际收益急剧下降，且参数量已经接近 full FT 一部分，性价比低
- ❗ **Multi-LoRA serving (vLLM `enable_lora`) 要求 batch 内不同 LoRA 的 rank 一致**，否则 batched GEMM 不能合并；如果不同 task 必须不同 r，serving 框架会 fall back 到串行执行，吞吐大跌。设计阶段就把 r 统一成一个值（如 16 或 32）
- ❗ **DoRA 比 LoRA 显存略多（多存 magnitude + 列范数中间量）、训练慢 5-10%**，但开 `use_dora=True` 在大多 SFT benchmark 上能涨 1-3 个点。代价小、收益稳，2026 年默认开就行
- ❗ **`merge_and_unload` 后不要再保存为 LoRA adapter**——merge 之后 LoRA 那两个矩阵已经被吃进 base.weight，再调 `save_pretrained` 会得到一个 base 大小（GB 级）的 ckpt 而不是几十 MB 的 adapter。要么 merge 走推理路径、要么不 merge 走 PEFT 路径，不要混
- ❗ **LoRA fine-tune 时 lr 比 full FT 大一个数量级**——LoRA 经验值 1e-4 ~ 5e-4，full FT 经验 1e-5 ~ 5e-5。原因：训练参数少、梯度更稀疏、再加 $\alpha/r$ scaling 把更新幅度放大；用 full FT 的小 lr 训 LoRA 会非常慢甚至不收敛

---

## 5. 经典 paper

- **Hu et al., 2021 — *LoRA: Low-Rank Adaptation of Large Language Models*** — LoRA 原典，提出 $\Delta W = BA$ 的低秩假设、$\alpha/r$ scaling、零初始化等所有核心设计。读它能搞清楚"为什么这么设计"——本节 §2.1-2.3 的所有公式直接来自这篇 §4
- **Dettmers et al., 2023 — *QLoRA: Efficient Finetuning of Quantized LLMs*** — NF4 + double quant + paged optimizer 三件套的完整推导，§3 严格证明 NF4 是"假设权重正态分布下的"信息论最优 4-bit 量化，是 4-bit 训练的必读
- **Liu et al., 2024 — *DoRA: Weight-Decomposed Low-Rank Adaptation*** — magnitude/direction 分解的提出，§3 实证 LoRA vs full FT 的"耦合度"差异，给出 DoRA 的解法。读完能理解"为什么 LoRA 与 full FT 仍有 gap"以及 DoRA 怎么补
- 选读：**Aghajanyan et al., 2020 — *Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning*** — LoRA 的理论先声，给出"language model fine-tune 的内禀维度只有几百到几千"的实证，是 LoRA "$\Delta W$ 低秩"假设的根因解释

---

## 6. 自测与面试题

**Q1（公式）：** 写出 LoRA 的核心公式 $W' = W + (\alpha/r) \cdot BA$，并解释为什么 $\alpha$ 要除 $r$。

<details>
<summary>Answer sketch</summary>

要点：

- 公式：$y = Wx + (\alpha/r) \cdot BAx$，其中 $A \in \mathbb{R}^{r \times d_\text{in}}$、$B \in \mathbb{R}^{d_\text{out} \times r}$、$r \ll d$
- 训练参数：$r(d_\text{in} + d_\text{out})$，远小于 $d_\text{in} d_\text{out}$
- $\alpha/r$ 的作用：rank-aware scaling——让**更换 $r$ 时不需要重新调学习率**
- 直观推导：$BA$ 输出方差大致正比于 $r$（B 慢慢学起来，每多一个 rank 就多一份贡献），除以 $r$ 把这个标度归一化掉
- 等价关系：$\alpha=32, r=8$ 与 $\alpha=64, r=16$ 在前向数值与等效 lr 上等价（比值都是 4）
- 加分：提一句 rsLoRA 改成 $\alpha/\sqrt{r}$，让大 $r$ 时梯度幅度更稳；HuggingFace PEFT 通过 `use_rslora=True` 启用
- 加分：解释初始化——B 用 0 让 $\Delta W$ 起步为 0，A 用 Kaiming 保证后续梯度非零

</details>

**Q2（实战）：** 你要在 24GB RTX 4090 上 SFT LLaMA-3 8B，列出 QLoRA 的完整配置：rank、target_modules、quantization、显存优化叠加。

<details>
<summary>Answer sketch</summary>

完整配置点 checklist：

- **base 量化**：`BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)`
- **prepare_model_for_kbit_training(model)** —— 把 LayerNorm 提 fp32、关闭 KV cache、开 gradient checkpoint
- **LoRA 配置**：r=16，lora_alpha=32，lora_dropout=0.05，bias="none"
- **target_modules**：q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj —— 全加 7 个 linear
- **训练超参**：bf16=True、per_device_batch_size=2、gradient_accumulation_steps=16（等效 batch=32）、lr=2e-4、warmup_ratio=0.03
- **优化器**：`optim="paged_adamw_8bit"` —— bitsandbytes 8-bit Adam + paged，省显存又防 OOM
- **gradient_checkpointing=True** —— 重算 activation 换显存
- **max_seq_length**：先试 2048，能跑再加；4090 24GB 上 8B QLoRA + checkpoint 应该能上 4096
- **加分**：提到用 unsloth / Liger Kernel 替代 HuggingFace 默认 attention 实现，可再省 30-50% 显存 + 加速 2x

显存账本（粗算）：
- NF4 base: 4 GB；LoRA adapter + grad + 8-bit optim: <1 GB；activation w/ checkpoint: ~10 GB；overhead: ~3 GB；**合计 ~18 GB**，留 6 GB buffer 给长序列 spike

</details>

**Q3（trade-off）：** QLoRA 比 LoRA 显存省 ~4 倍，但训练慢 30% 左右，为什么慢？什么时候**不该**用 QLoRA？

<details>
<summary>Answer sketch</summary>

慢的原因（多线索都要点到）：

- **每次前向都要 dequantize**：base 权重存的是 NF4 4-bit 紧凑格式，每次 GEMM 前要 dequant 成 bf16 再做矩阵乘，相当于多了一次内存读 + 反量化 kernel；训练时每个 forward + backward 都触发一次（推理时还有 KV cache）
- **bitsandbytes 的 4-bit kernel 比 cuBLAS 的 bf16 GEMM 慢**——cuBLAS / cuDNN 极致优化，bitsandbytes 4-bit kernel 还在追赶；近 1-2 年差距已经缩小（unsloth + 自研 kernel 已基本无 gap）
- **double quant 解码也有开销**：每次访问 scale 因子要先反量化它自身，又一次小 kernel
- **paged optimizer 在显存够的情况下也会引入额外 host-device 传输**—— OOM 才会 trigger swap，平时无影响

什么时候**不该**用 QLoRA：

- **显存充足的场景**：8 张 H100 训 7B，显存随便堆，用 LoRA / 甚至 full FT 更快、效果更稳
- **对训练吞吐极敏感的大规模 RL**：PPO / GRPO 多次 rollout + update，QLoRA 的 30% 速度损失会被放大
- **数值精度极敏感的任务**：4-bit 量化引入的小误差，在 reward model 训练或长 CoT reasoning 上偶尔会让效果略低于 LoRA（差异通常 <1%，但严苛 benchmark 上能体现）
- **GPU 不支持 bf16 / 4-bit 的旧机器**：V100、T4 等

加分：边界讨论——**LoRA 与 QLoRA 不是竞争关系而是补全显存预算**：能 LoRA 就 LoRA，显存不够再上 QLoRA；2026 年的实战默认是 7B-13B 用 LoRA、30B-70B 用 QLoRA

</details>

---

## 7. 延伸阅读

- [HuggingFace PEFT 文档](https://huggingface.co/docs/peft) — LoRA / DoRA / rsLoRA / VeRA 等所有变体的官方实现与示例，本节 §3 代码的来源
- [QLoRA 官方 repo (artidoro/qlora)](https://github.com/artidoro/qlora) — Dettmers 团队的训练脚本，包含 NF4 / paged optimizer 的端到端实战
- [bitsandbytes 文档](https://huggingface.co/docs/bitsandbytes) — NF4 / 8-bit Adam / paged optimizer 的工程实现与 GPU 兼容性说明
- [Unsloth (unslothai/unsloth)](https://github.com/unslothai/unsloth) — 把 LoRA / QLoRA 训练速度做到 HF 的 2-5 倍、显存再省 30%；端侧 / 单卡 SFT 必看
- [vLLM Multi-LoRA Serving 文档](https://docs.vllm.ai/en/latest/models/lora.html) — 工业 multi-LoRA 部署，§3.5 代码的官方扩展版
- [Sebastian Raschka — Practical Tips for Finetuning LLMs Using LoRA](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms) — 实战调参经验长文，rank/alpha/target_modules 的实证 ablation
- 推荐继续读本教程的 **8.4 Adapter / Prefix-tuning / P-tuning v2**——LoRA 之外的其它 PEFT 路线；**8.5 SFT 实战**——把本节的 LoRA 配置端到端跑通；**Module 9 RLHF**——DPO / PPO / GRPO 几乎都基于 LoRA fine-tune
