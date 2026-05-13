---
title: "7.5 显存优化：Activation Recomputation / Selective / Offload"
description: "7.1 的 FSDP 把 weight + grad + optim 这三件套的显存按卡数分掉，但还有一个被忽视的大头：activation 显存——它和 $B \\\\cdot T \\\\cdot L \\\\cdot d$ 强相关，7B 模型上轻松超过 weight 本身。本节给出 activation 的算账公式，再讲三件用通信换显存、用计算换显存、用带宽换显存的标准武器：gradient checkpoi"
---

> ⏱ 预计阅读 45 分钟 ｜ 难度 ★★★ 🔥 ｜ 前置：7.1 ZeRO/FSDP

## 一句话本节讲什么

7.1 的 FSDP 把 weight + grad + optim 这三件套的显存按卡数分掉，但**还有一个被忽视的大头：activation 显存**——它和 $B \cdot T \cdot L \cdot d$ 强相关，7B 模型上轻松超过 weight 本身。本节给出 activation 的算账公式，再讲三件用通信换显存、用计算换显存、用带宽换显存的标准武器：**gradient checkpointing**（Chen 2016）、**selective recomputation**（Korthikanti 2022）、**CPU/NVMe offload**（ZeRO-Infinity）。

---

## 1. Mental model（直觉）

接 7.1 的画面：FSDP 把 70B 在 8×H100 上的"三件套"压到 105 GB，但实际跑起来还是 OOM。问题出在**第四件套**：**activation**——forward 算出来的所有中间 tensor，backward 要它们做链式求导，所以默认全部缓存在显存里。

直觉上 activation 显存与"模型一遍 forward 经过的所有中间 tensor 总和"成正比：

- 每过一层 transformer block，attention 内部至少要存 $Q / K / V$、注意力权重 $\text{softmax}(QK^T)$、attn output；FFN 至少要存中间 hidden、激活函数前后；外加 LayerNorm 统计量
- 这些 tensor 的形状都是 $[B, T, d]$ 或 $[B, h, T, T]$ 量级
- 层数 $L$ 一叠加 → activation 显存 $\propto B \cdot T \cdot L \cdot d$（线性于层数 $L$）

7B (LLaMA-2，$L=32$、$d=4096$、$B=4$、$T=4096$、bf16) 的 activation 估算 ≈ **70 GB**——比 weight 的 14 GB 还大。这就是"activation 是大模型训练的第二杀手，仅次于 optimizer state"的来源。**注意 activation 显存对 batch 与序列长度是严格线性、对层数也线性、对 hidden size 也线性**——也就是说它只能靠"砍掉某段 activation"或者"挪到别的存储介质"两个方法降低，没法像 optim state 一样靠 FSDP 把它沿参数维度切成 $1/N$（activation 不在参数维度上）。

要砍 activation 显存，本质上有三条路：

```text
                 显存换计算         显存换带宽          换都不换
                 ─────────         ─────────          ──────
      forward    存 1/√L 的 ckpt   存到 CPU            老老实实存
      backward   再算一次 forward   传回来用            直接读
                 +33% compute      +PCIe 同步等待      0 开销
                 ✅ Chen 2016       ✅ Activation       ❌ 显存爆
                 gradient ckpt     offload
```

第一条是**gradient checkpointing**：forward 时只在少数"checkpoint 层"存 activation，其他层的 activation 扔掉；backward 用到时**重新算一遍 forward** 把它造出来。砍显存约 $\sqrt{L}$ 倍，代价是一次额外 forward（约 +33% 训练时间）。

但全砍太血亏——有些 op 算起来贵但 activation 小（attention matmul），有些 op 算起来便宜但 activation 大（softmax 输出、SwiGLU 中间值）。**Selective recomputation**（Megatron）只 ckpt 后者，把 +33% 压到 +5-10%，显存还省得差不多——这是当代精细化版本，Megatron 默认开启。

第二条是**activation offload**：forward 算完一层，把 activation 通过 PCIe 搬到 CPU 内存；backward 前 prefetch 回 GPU。显存彻底不占，代价是 PCIe 带宽（~32 GB/s）vs HBM（~3 TB/s）的差距，必须有 prefetch overlap 才能掩盖。

第三条更狠——**ZeRO-Infinity**：weight、grad、optim、activation 全都 offload，CPU 内存满了往 NVMe 写。100B+ 模型在单 8 卡机器上跑成为可能，代价是慢 30-100%。

这三条路并不互斥，**实际生产几乎都是组合使用**：FSDP 砍 weight/grad/optim、selective recompute 砍 activation、offload 兜底——这正是 7B 在 4090 / 70B 在 8 卡 H100 / 405B 在 1024 卡集群上的标准配方。它们的共同哲学是"**没有免费的午餐，只有合适的换钱方式**"：要么用计算（FLOPs 翻 1.05-1.33 倍）换显存，要么用通信带宽（PCIe / NVMe 同步）换显存，要么用通信延迟（all-gather）换显存。

记住一个口诀：**weight/grad/optim 用 FSDP（7.1）；activation 用 selective recomputation；显存还不够时上 offload；终极兜底是 QLoRA**。本节后面是这个口诀的展开和算账。

---

## 2. 公式与原理

### 2.1 Activation 显存的算账（Korthikanti 2022 公式）

每个 transformer block 的 activation 显存来源（按 fp16/bf16 = 2 B/element 计）：

| 来源 | 形状 | size (bytes) |
|---|---|---|
| attention 输入（LN 输出） | $[B, T, d]$ | $2BTd$ |
| Q / K / V projection 输入 | $[B, T, d]$ | $2BTd$ |
| Q, K, V 张量 | 各 $[B, T, d]$ | $6BTd$ |
| $QK^T / \sqrt{d_k}$ 的 attention scores | $[B, h, T, T]$ | $2 B h T^2$ |
| softmax 输出 | $[B, h, T, T]$ | $2 B h T^2$ |
| dropout mask | $[B, h, T, T]$ | $B h T^2$（mask 用 1 byte） |
| attention $\times V$ 输出 | $[B, T, d]$ | $2BTd$ |
| output projection 输入 | $[B, T, d]$ | $2BTd$ |
| FFN LN 输入 | $[B, T, d]$ | $2BTd$ |
| FFN 中间 hidden | $[B, T, 4d]$ | $8BTd$ |
| FFN 激活后 | $[B, T, 4d]$ | $8BTd$ |
| FFN dropout mask | $[B, T, d]$ | $BTd$ |

把 attention 部分（与 $T^2$ 相关）和 FFN 部分（与 $T$ 线性）相加，每层总和约：

$$M_{\text{act}}^{(\text{layer})} \approx s \cdot B T d \left(34 + \frac{5 h T}{d}\right) \text{ bytes}$$

其中 $s = 1$（fp16/bf16）；常用近似形式（在 $hT/d \approx 1$ 量级时）：

$$\boxed{M_{\text{act}} \approx 34 \cdot B \cdot T \cdot L \cdot d \text{ bytes}}$$

这是 Korthikanti 2022 给的"naïve activation memory" 公式（含 attention + FFN + norm 全部中间值，未开任何优化）。**LLaMA-2 7B 实例**（$L=32$、$d=4096$、$B=4$、$T=4096$、bf16）：

$$34 \times 4 \times 4096 \times 32 \times 4096 \approx 73 \text{ GB}$$

——比 7B weight (bf16) 的 14 GB 大 5 倍。这就是为什么训练 7B 时光算 weight 觉得"应该够"，实跑就 OOM。

> 注 1：上式默认无 FlashAttention。FlashAttention 不显式存 $[B,h,T,T]$ 的 attention scores（用 IO-aware tiling 在 SRAM 内算），把 $5BhT^2$ 这一项干掉，公式变成 $\approx 12 BTLd$ 量级——这是 5.3 节的主题。
> 注 2：$T^2$ 项当 $T$ 很长时会反超 FFN 项。$T=8K$ 之后 attention scores 占主导，长 context 训练时 FlashAttention 不只是快、还是显存救命稻草。

### 2.2 Gradient Checkpointing：显存换计算

**核心思想**（Chen 2016）：forward 只在少数"checkpoint 节点"保留 activation，节点之间的中间 activation 全扔。backward 时用到某段时，**从最近的 checkpoint 重新 forward 一遍**把中间 activation 造出来，再做反向。

为什么这能省显存？因为反向求导只在"用到"中间 activation 的瞬间需要它——之前可以不存、之后可以扔。极端情况下你甚至可以只存输入 $x$，每次 backward 时把整个 forward 重算一遍——显存降到 $O(1)$，但计算翻 $L$ 倍，得不偿失。Chen 2016 给出的最优 trade-off 是：**段长选 $\sqrt{L}$，显存与计算都 $\sqrt{L}$ 量级**。

设把 $L$ 层模型沿深度分成 $\sqrt{L}$ 个段，每段 $\sqrt{L}$ 层。每段只在段头存一个 activation。Forward 显存：

$$M_{\text{ckpt}} = \underbrace{O(\sqrt{L})}_{\text{段头}} + \underbrace{O(\sqrt{L})}_{\text{当前段内临时}} = O(\sqrt{L})$$

砍掉一个 $\sqrt{L}$ 因子。代价是 backward 时每段要 recompute 一次 forward——总 compute 从 "1 forward + 2 backward" 变成 "2 forward + 2 backward"，**约 +33% 训练时间**。

实际工程的近似总结：

| 配置 | activation 显存 | 训练时间 |
|---|---|---|
| 不开 ckpt | 100% | 100% |
| Full ckpt（每 layer 一个 ckpt 节点） | ~10-30% | ~133% |
| 段长 $\sqrt{L}$ 最优 ckpt | ~$1/\sqrt{L} \approx$ 15% | ~133% |

**注意**：HuggingFace `gradient_checkpointing_enable()` 默认是"每个 transformer layer 一个 ckpt 节点"——对 32 层的 7B，segment 数就是 32，远没到 $\sqrt{32} \approx 5.6$ 的最优。但工程上选择"按 layer ckpt"是因为它最简单、与 FSDP unit 对齐、显存收益已经极大。

### 2.3 Selective Activation Recomputation（Korthikanti 2022）

观察 §2.1 的清单：activation 显存的大头集中在两类 op：

- **softmax 输出 / dropout mask**：$5 B h T^2$ 量级，是 $T$ 长时的主项
- **FFN 中间 hidden（4× 维度）**：$16 B T d$，FFN 部分主项

而它们的**重算成本极低**——softmax 和 SwiGLU/GELU 都是 element-wise + 一个轻 matmul。相比之下：

- Q / K / V projection、attention output projection、FFN 两个 Linear：**重算成本高**（matmul，FLOPs 大），但 activation 占用相对小

**Selective recomputation 策略**：只 ckpt 那些"高显存 / 低计算"的 op，保留"低显存 / 高计算"的 op。Megatron 的具体做法：

```text
保留的 activation:    Q/K/V proj 输入、attention output proj 输入、
                     FFN 两个 Linear 输入  ← 重算贵，但显存小
丢弃 + 重算的 activation: softmax 输出、attention 输出 ×V 中间、
                     FFN 中间 hidden、dropout mask
                     ← 显存大、重算便宜
```

实证数据（Megatron-LM, GPT-22B, 22B params）：

| 策略 | activation 显存 | 训练 throughput |
|---|---|---|
| 无 ckpt | 100% | 100% |
| Full ckpt | ~30% | ~70-75% (-25-30%) |
| **Selective** | **~50%** | **~95% (-5%)** |

**Selective 用 1/3 的 compute 代价拿到 Full ckpt 80% 的显存收益**——这是当代 Megatron-LM、最新版 DeepSpeed、FSDP2 的默认推荐。

为什么 selective 这么划算？本质是**显存与计算量在不同 op 上严重不对称**：

- 一个 $[B, h, T, T]$ 的 softmax 输出，只需要做一次 element-wise exp + 归一化（FLOPs $\propto BhT^2$，但每个 element 只算一次），重算成本极低；存它却要 $5 BhT^2$ bytes
- 一个 $d \times 4d$ 的 FFN Linear matmul，FLOPs 是 $8 BTd^2$（重算贵），但其输出只是 $[B, T, 4d]$（存便宜）

把"重算/存"比值低的 op 都列出来，按比值排序，从低往高加进 recompute 列表——这就是 selective 的工程做法。Megatron 把这个选择硬编码进 transformer block 的 forward 里，省去了用户调参。

> 边界：**FlashAttention 已经把 attention scores 不存了**——所以 selective 在 attention 这边的收益变小，但在 SwiGLU + RMSNorm 的 FFN 中间 activation 上仍然非常大。当代 LLaMA / Qwen 架构上，selective 的主要收益其实来自 FFN 而非 attention。

### 2.4 Activation Offload (CPU)

把 activation 在 forward 后**异步搬到 CPU 内存**，backward 用前 prefetch 回 GPU：

```text
forward layer i :  在 GPU 算完 → async 拷贝 activation 到 CPU → free GPU 显存
forward layer i+1: 同样
...
backward layer L-1: prefetch (L-1 的 activation) → 算
backward layer L-2: prefetch (L-2 的 activation) → 算
...
```

显存：activation 几乎清零，代价是 **PCIe 带宽** 与同步开销。

**带宽现实**：

- HBM3 (H100) ：~3 TB/s
- PCIe 5.0 x16 ：~64 GB/s（H100/H200）
- PCIe 4.0 x16 ：~32 GB/s（A100、4090）
- PCIe 3.0 x16 ：~16 GB/s

7B 一层 activation 约 2 GB → 在 PCIe 4.0 上传一次约 60 ms。一层 forward 算上 attention + FFN 大概几 ms 到几十 ms 量级——**计算掩盖不住传输**，offload 反而变慢。所以 activation offload 适合：**模型很大 / batch 很大、单层计算时间长 ≥ 传输时间**的场景。粗略判别准则：单层 FLOPs ÷ GPU 算力 ≥ 单层 activation 字节数 ÷ PCIe 带宽，offload 才划算。70B 一层在 H100 上 forward 要 50-100 ms，远大于 PCIe 4.0 传输 4-8 GB activation 的 130-250 ms？——并不一定划算，要看具体配置；只能 profile 后决定。

PyTorch FSDP `cpu_offload` 主要 offload 的是 **weight + grad** 而非 activation；activation offload 在 PyTorch 主线还没 stable，DeepSpeed `activation_checkpointing.cpu_checkpointing` 是更成熟的实现。

### 2.5 ZeRO-Infinity (Ren 2021)：CPU + NVMe 极致 offload

把所有可 offload 的对象按"用得多 → 用得少"分级放置：

```text
GPU HBM         ：当前 layer 的 weight / activation（用就拿）
CPU DRAM        ：optimizer state、暂时不用的 weight、ckpt activation
NVMe SSD        ：极端情况下 optimizer state 也放这里
```

实战上，DeepSpeed `nvme_offload` 让 100B+ 模型在单 8×A100 80G 节点上跑通成为可能——但训练速度通常慢 50-100%。生产大集群训练用 FSDP 多机即可，ZeRO-Infinity 主要面向"卡少 + 模型大"的资源紧张场景。一句话总结 offload 的适用边界：**"宁可慢但要能跑"** 优先 offload；**"贵但要快"** 优先多卡 FSDP，不碰 offload。

### 2.6 多种显存优化策略对比表

| 策略 | 节省显存对象 | 节省比例 | 计算 / 通信代价 | 实现复杂度 |
|---|---|---|---|---|
| Gradient Checkpointing (full) | activation | ~70-80% | +33% compute | 低（一行 enable） |
| **Selective Recomputation** | activation | ~50% | +5-10% compute | 中（需要选 op） |
| Activation Offload (CPU) | activation | ~70% | PCIe 同步等待 | 中 |
| FSDP（7.1） | weight + grad + optim | $1/N$ | +50% 通信 | 中 |
| ZeRO-Infinity (NVMe offload) | weight + optim | 极大 | 慢 30-100% | 高 |
| LoRA / QLoRA（8.3） | optim 几乎全省 | optim → ~0 | 略损质量 | 低 |
| 混合精度 / fp8（7.4） | weight + grad + 计算 | ~50% | 工程复杂 | 高 |

口诀："**FSDP 砍三件套，selective recomp 砍 activation，offload 兜底，QLoRA 极致省**"——这四件武器是当代 LLM 训练显存优化的全部工具。

### 2.7 实战："7B SFT in 24GB GPU" 的组合方案

单张 RTX 4090（24 GB） 能不能 SFT 7B？答案是能，靠**叠加 4 件武器**：

| 优化 | 7B 显存压力 |
|---|---|
| 原始 bf16 weight | 14 GB |
| → **QLoRA 4-bit weight** | 3.5 GB |
| 原始 AdamW optim | 56 GB |
| → 只训 LoRA adapter (rank=64, ~0.06% 参数) → optim 1 GB | |
| → **Adam-8bit** 把这 1 GB 再砍 4 倍 → 0.25 GB | |
| 原始 activation (B=1, T=2048) | ~5 GB |
| → **gradient checkpointing** 砍 70% → 1.5 GB | |
| **总计 GPU 占用** | **~10-12 GB** ✅ |

剩 12 GB 留给推理临时 buffer、kv-cache、PyTorch overhead，刚好 fit 24 GB。**这是 unsloth / axolotl / llama-factory 在 4090 上 SFT 7B 的标准配方**。注意这个组合是 4 件武器叠乘，少任何一件都会爆——QLoRA 是关键的"weight 减半再减半"，Adam-8bit 是关键的"optim state 又压 4 倍"，gradient checkpointing 是关键的"activation 砍 70%"，LoRA adapter 是关键的"被训参数从 7B 砍到 4M"。任何单一武器都救不了 24 GB，必须全开。

### 2.8 Liger Kernel / Fused op 的显存收益

最近一类隐形的 activation 优化是 **fused kernel**：把 cross-entropy、SwiGLU、RMSNorm 这种"算少存多"的 op 在 kernel 内部融合，**不存中间 activation 直接出最终结果**。

代表实现：

- **Liger Kernel**（LinkedIn 2024）：fused cross-entropy / SwiGLU / RMSNorm，**比 PyTorch 默认省 ~30% activation 显存**，吞吐 +20%
- **Apex** 老版本的 `FusedRMSNorm` / `FusedAdam`
- HuggingFace `transformers` 4.40+ 默认集成 Liger 或 fused-LCE

与 gradient checkpointing 协同——fused kernel 先把单层 activation 砍 30%，再做 ckpt 砍 70%，叠乘后 activation 显存只剩 20%。fused kernel 的另一个好处是吞吐提升不收税：相比 gradient checkpointing 的 +33% 训练时间代价，fused kernel 是**纯赚**——既省显存又快。所以新项目的标准做法是"先用 Liger / 内置 fused kernel 占优，剩余显存压力再用 ckpt 兜底"，而不是直接堆 ckpt。

---

## 3. 最小代码示例

### 3.1 Gradient Checkpointing：手动 + HuggingFace 一键

```python
# 方式 1：手动包 checkpoint（自定义模型）
import torch, torch.nn as nn
from torch.utils.checkpoint import checkpoint

class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, 8, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))

    def _forward(self, x):
        x = x + self.attn(x, x, x, need_weights=False)[0]
        return x + self.ffn(x)

    def forward(self, x):
        # use_reentrant=False 是 PyTorch 2.0+ 推荐的新 API
        return checkpoint(self._forward, x, use_reentrant=False)

# 方式 2：HuggingFace 一键（已有 HF 模型时）
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf')
model.gradient_checkpointing_enable()             # ← 一行开启
model.config.use_cache = False                    # ← 必须！否则 ckpt + cache 冲突
```

注意 `use_cache=False`：训练时 KV cache 没用，留着会与 ckpt 冲突报 warning。

### 3.2 FSDP + Activation Checkpointing 组合

```python
import functools
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
    CheckpointImpl,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

# 步骤 1：先把 model 包成 FSDP（参考 7.1 的 3.2 节）
model = FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD, ...)

# 步骤 2：在 FSDP unit 上叠加 activation checkpointing
non_reentrant_wrapper = functools.partial(
    checkpoint_wrapper,
    checkpoint_impl=CheckpointImpl.NO_REENTRANT,        # 必选 NO_REENTRANT
)
apply_activation_checkpointing(
    model,
    checkpoint_wrapper_fn=non_reentrant_wrapper,
    check_fn=lambda m: isinstance(m, LlamaDecoderLayer),  # 每个 decoder layer 加 ckpt
)
# 之后训练循环与单卡完全一样
```

关键点：

1. `apply_activation_checkpointing` **必须在 FSDP wrap 之后**调用，顺序反了会报错
2. `check_fn` 决定哪些 module 被 ckpt——和 FSDP `auto_wrap_policy` 通常用同一组 transformer layer class 对齐，否则 FSDP unit 与 ckpt 边界错位会触发额外 all-gather

### 3.3 Activation 显存估算函数

```python
def estimate_activation_gb(B: int, T: int, L: int, d: int,
                           dtype_bytes: int = 2,
                           use_flash: bool = True) -> float:
    """估算训练时 activation 显存（GB），不含 weight/grad/optim。
       公式：~34*B*T*L*d (Korthikanti 2022)，FlashAttn 减去 ~5*B*h*T^2*L 项。
    """
    base = 34 * B * T * L * d * dtype_bytes
    if not use_flash:
        h = max(d // 128, 1)               # 假设 head_dim=128
        base += 5 * B * h * T * T * L * dtype_bytes
    return base / 1e9

# >>> estimate_activation_gb(B=4, T=4096, L=32, d=4096)        # LLaMA-2 7B, FlashAttn
# 73.0
# >>> estimate_activation_gb(B=2, T=4096, L=40, d=5120)        # LLaMA-2 13B, FlashAttn
# 57.1
# >>> estimate_activation_gb(B=4, T=4096, L=32, d=4096) * 0.3  # +full ckpt
# 21.9
```

实际工程结果再 ×1.2-1.5 留 PyTorch / NCCL workspace 余量。

### 3.4 DeepSpeed-Infinity Offload Config

```json
// ds_config.json：单卡训 7B 全参 + 全 offload，启动：deepspeed train.py --deepspeed ds_config.json
{
  "train_micro_batch_size_per_gpu": 1,
  "gradient_accumulation_steps": 16,
  "bf16": { "enabled": true },

  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "nvme",                       // optim state 卸到 NVMe
      "nvme_path": "/local_nvme/zero_offload",
      "pin_memory": true,
      "buffer_count": 4
    },
    "offload_param": {
      "device": "cpu",                        // weight 卸到 CPU
      "pin_memory": true
    },
    "overlap_comm": true,
    "contiguous_gradients": true
  },

  "activation_checkpointing": {               // activation 也 ckpt
    "partition_activations": true,
    "cpu_checkpointing": true,                // 进一步把 ckpt activation 也卸到 CPU
    "contiguous_memory_optimization": true
  }
}
```

四档 offload（GPU → CPU → NVMe + activation ckpt + activation offload）全开，是 DeepSpeed-Infinity 的极致用法。

---

## 4. 工程踩坑与经验

- ❗ **`use_reentrant=True` 是 PyTorch 旧 API**，2.0+ 推 `use_reentrant=False`；旧版本在 FSDP 下会有 grad 不更新或 grad 错误的 bug。HuggingFace 4.36+ 默认已切到 `False`，但你要从 PyTorch checkpoint API 直接调时**手动加这个参数**。
- ❗ **gradient checkpointing 与 dropout 结合时必须 deterministic**。ckpt 会重算 forward，如果 dropout 两次的 mask 不一样，第二次 forward 算出的 activation 与第一次保存的不一致，反向算出来的 grad 就错了。PyTorch `torch.utils.checkpoint` 默认会 save 并 restore RNG state，但**自定义 ckpt 实现要手动 `preserve_rng_state=True`**——这是隐性 bug 之王。
- ❗ **selective recomputation 配置不当会反而变慢**。如果错把 attention 的 QKV projection 也 ckpt 了（这是计算密集 op），重算成本高、节省显存少，等于花 +20% compute 换 5% 显存。Megatron 默认配置只 ckpt softmax 输出 + FFN 中间 hidden + dropout mask，**改之前先 profile**。
- ❗ **activation offload 在 PCIe 3.0 系统上反而比不 offload 慢**。RTX 3090 / V100 这类 PCIe 3.0 x16 = 16 GB/s 的带宽，传 1 层 activation (2 GB) 要 125 ms，远超单层 forward 时间。**先查机器 PCIe 版本再决定开 offload**：`lspci -vv | grep -i "lnkcap\|lnksta"`。
- ❗ **DeepSpeed offload 与 FSDP 不能直接混用**。两个框架都管 weight 的分片 / load，混用会冲突死锁。**单一 codebase 选一个**：用 HF Trainer 的 `--deepspeed` 走 DeepSpeed offload，或用 `--fsdp` 走 FSDP（FSDP 自带 `cpu_offload=True` 是 weight offload）。
- ❗ **HF Trainer 的 `gradient_checkpointing` 与 `gradient_accumulation_steps` 都开时**，accumulation 中每个 micro-step 都会 ckpt+recompute → 计算量翻倍而非每个 macro-step 翻倍。这是预期行为但很多人误以为开 ckpt 之后 grad accumulation 几乎免费——实际 wall-clock 真的会 +33% × 每个 micro-step。
- ❗ **微调时 ckpt 在 LoRA layer 上无意义**。LoRA 只训 adapter，base model frozen 已经不存 grad，但 forward 仍会算 base model 的 activation——**正确做法是 ckpt freeze 的 backbone 层**而不是 LoRA 层。HF PEFT 的 `model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})` 默认对整个 backbone 生效，包括 LoRA 注入的层；通常没问题但要意识到这一点。
- ❗ **selective recomputation 在 SwiGLU + RMSNorm 上节省最大**。SwiGLU 中间 hidden 是 $[B, T, 4d]$（甚至 $\frac{8d}{3}$），是 activation 大头；RMSNorm 也存 LN 输入。FlashAttention 已经把 attention 的 $T^2$ 项干掉，**LLaMA 系列上 selective 主要收益来自 FFN 而不是 attention**——这点和 GPT-2/22B 的 Korthikanti 原 paper 时代不一样。
- ❗ **FSDP 的 `activation_checkpointing=True` 老 API 已 deprecated**。新代码用 `apply_activation_checkpointing` 在 FSDP wrap 后单独调用，不要写在 FSDP 构造函数里。
- ❗ **ckpt 与 `torch.compile` 兼容性**：PyTorch < 2.3 时 `checkpoint(use_reentrant=False)` 经常 graph break，2.3+ 才稳定。如果 compile 后看到训练 throughput 没提升甚至变慢，先关 ckpt 看 baseline、再加回来对比。
- ❗ **不要忘了"保存与重算的边界 op 必须可微 + 无副作用"**。如果 ckpt 段内含有随机数生成、计数器更新、in-place 操作或者外部调用（如打 log、写文件），第二次 forward 会重复执行这些副作用——典型坑是 `BatchNorm` 在 ckpt 段内运行时会两次更新 running mean / var，统计量被污染。LayerNorm / RMSNorm 没这个问题（无状态），是 transformer 训练的福音。
- ❗ **混合精度 + ckpt 时检查 autocast 范围**。`torch.cuda.amp.autocast` 是 thread-local context，ckpt 重算 forward 时如果不在 autocast 范围内执行，重算用 fp32 而原 forward 用 bf16，数值精度不一致→ grad 错。PyTorch 2.0+ ckpt 默认会保存 autocast state，但自定义实现要手动管理。

---

## 5. 经典 paper

- **Chen et al., 2016 — Training Deep Nets with Sublinear Memory Cost** — gradient checkpointing 的开山之作。证明用 $\sqrt{L}$ 个 ckpt 节点可把 activation 显存从 $O(L)$ 降到 $O(\sqrt{L})$，代价是 1 次额外 forward。读 §3 算法 1，搞清楚 segment 选择的最优解为什么是 $\sqrt{L}$。本节 §2.2 就是它。
- **Korthikanti et al., 2022 — Reducing Activation Recomputation in Large Transformer Models** — Megatron 团队 2022 的 selective recomputation paper。给出本节 §2.1 的 activation 显存 $34 BTd$ 公式（论文 Table 1），并提出 selective 策略——只 ckpt 高显存低算的 op，把 +33% compute 压到 +5%。读 §3 + §4，把 selective 和 sequence parallel（7.3）的来龙去脉搞通。
- **Rajbhandari et al., 2021 — ZeRO-Infinity: Breaking the GPU Memory Wall** — DeepSpeed 团队的 CPU/NVMe offload 系统化论文。讲清"如何把 weight / optim / activation 分级放置在 GPU/CPU/NVMe，并通过 prefetch + bandwidth-centric partitioning 维持训练 throughput"。读 §3 系统设计 + §6 实验对比，理解 offload 的工程边界。
- **加分：Dettmers et al., 2023 — QLoRA: Efficient Finetuning of Quantized LLMs** — 4-bit NF4 weight + LoRA + paged optim，在单 48GB GPU 上 SFT 65B。本节 §2.7 "7B SFT in 24GB" 就是 QLoRA + ckpt + Adam-8bit 组合的具体落地。8.3 LoRA 节会详细讲。

---

## 6. 自测与面试题

**Q1（算账）**：用 Korthikanti 公式（$M_{\text{act}} \approx 34 BTLd$ bytes）计算 LLaMA-2 13B（$L=40$、$d=5120$、$B=2$、$T=4096$、bf16）训练时的 activation 显存。如果开 full gradient checkpointing 能节省到多少？开 selective recomputation 呢？

<details>
<summary>Answer sketch</summary>

1. naïve activation 显存：

$$34 \times 2 \times 4096 \times 40 \times 5120 \approx 57.1 \text{ GB}$$

2. **Full gradient checkpointing**：经验上砍到 ~20-30% → **~12-17 GB**（如果按"每 layer 一个 ckpt 节点"则砍 ~70%；按 $\sqrt{L} \approx 6$ 段最优 ckpt 砍更多但 PyTorch 默认按 layer 即可）

3. **Selective recomputation**：约砍到 50% → **~28 GB**

4. 对比 weight：13B bf16 weight = 26 GB —— activation 与 weight 同量级，所以 13B 训练上 selective + FlashAttention 是默认必开

加分：

- 注意公式假设无 FlashAttention。如果用 FlashAttention，attention $T^2$ 项消去，公式接近 $\approx 12 BTLd$
- 公式里的 34 是 fp16/bf16 假设；fp32 ×2，fp8 ×0.5
- 真实训练显存 = activation + weight + grad + optim + 临时 buffer，要全算

</details>

**Q2（trade-off）**：Selective Recomputation 比 Full Checkpointing 好在哪？为什么 Megatron 默认开 selective 而不是 full？

<details>
<summary>Answer sketch</summary>

**Full ckpt** 对每个 layer 的所有 op 做"forward 不存 → backward 重算"，砍显存最多但代价均匀地加在所有 op 上，**+33% compute**。

**Selective** 只对"高显存 / 低计算"的 op recompute，"低显存 / 高计算"的 op 仍保留 activation——这正好利用了 transformer 内部 op 的不均匀性：

- softmax 输出、SwiGLU 中间 hidden、dropout mask：activation 大、计算便宜 → **应该 recompute**
- Q/K/V projection、attention output proj、FFN 两个 Linear：activation 小、计算贵（matmul） → **应该保留**

实证（Korthikanti 2022）：

| 策略 | activation 显存 | 训练时间 |
|---|---|---|
| Full ckpt | ~30% | +33% |
| Selective | ~50% | **+5-10%** |

Selective 用 1/3 的 compute 代价拿到 Full 80% 的显存收益——Pareto 改进。所以 Megatron 默认 selective，HuggingFace 也在向这个方向靠（开 ckpt 时优先 selective、退化才走 full）。

加分：当代 LLaMA 架构上 attention 已经被 FlashAttention 干掉 $T^2$ 项，selective 收益主要来自 FFN 中的 SwiGLU 中间 hidden ($[B, T, 4d]$ 或 $[B, T, 8d/3]$)；纯 transformer 时代 selective 在 attention 上收益更大。

</details>

**Q3（实战）**：你要在 8×A100 80G 上 fine-tune 70B 模型，给出**至少 4 种**显存优化技术叠加，并解释组合能否 fit 80G。

<details>
<summary>Answer sketch</summary>

70B 不开任何优化的 weight + grad + optim ≈ 840 GB（bf16 + AdamW），单卡 80G 完全装不下；activation 还要再 50-100 GB（B=1, T=4096）。必须叠加：

1. **FSDP (ZeRO-3)**：weight + grad + optim 砍 1/8 → 单卡 105 GB（仍超）
2. **Selective activation recomputation**：activation 从 ~80 GB（70B, B=1, T=4K）砍到 ~40 GB
3. **CPU offload optim state**（FSDP `cpu_offload=True` 或 DeepSpeed ZeRO-3 + offload_optimizer）：把 70 GB optim 卸到 CPU → 单卡 weight+grad ≈ 35 GB
4. **micro-batch=1 + gradient accumulation**：减小瞬时 activation，靠 accum 模拟大 batch
5. **bf16/fp8 混合精度**（已默认，7.4 详讲）
6. **加分：FSDP × TP 混合**（7.2 节）：8 卡内 TP=2，跨节点 FSDP=4，weight 进一步切；这是工业 70B 训练的真正配方
7. **加分：FlashAttention**（5.3 节）：消去 attention $T^2$ 项 activation
8. **加分：sequence parallel**（7.3 节）：把 LayerNorm / dropout 这类剩余 activation 沿序列维切给多卡

组合成本估算（FSDP + selective recomp + offload optim + micro-batch=1 + FlashAttn）：

- weight (FSDP) = 17.5 GB
- grad (FSDP) = 17.5 GB
- optim (CPU offload) = 0 GB on GPU
- activation (selective + FlashAttn) = ~20-30 GB
- 临时 + workspace = ~10 GB
- **总计 ~65-75 GB**，刚好 fit 80G ✅

差答案：只列 LoRA——这是 fine-tune 但不算 full-param SFT；要明确"如果题目要求全参 SFT 必须 FSDP+ckpt+offload，如果允许 PEFT 才上 LoRA"。

</details>

---

## 7. 延伸阅读

- [PyTorch — torch.utils.checkpoint 文档](https://pytorch.org/docs/stable/checkpoint.html) — `use_reentrant` 两个分支的详细差异，调试 ckpt + dropout 的 RNG 问题必读
- [PyTorch FSDP — Activation Checkpointing](https://pytorch.org/tutorials/intermediate/FSDP_advanced_tutorial.html#applying-activation-checkpointing) — `apply_activation_checkpointing` 的官方用例
- [Megatron-LM — Selective Activation Recomputation](https://github.com/NVIDIA/Megatron-LM/blob/main/docs/source/api-guide/transformer.md) — Megatron 配置 selective 的实战 flag 说明
- [DeepSpeed — Activation Checkpointing](https://www.deepspeed.ai/tutorials/activation-checkpointing/) — DeepSpeed 的 partition_activations / cpu_checkpointing 用法
- [HuggingFace — Performance and Scalability](https://huggingface.co/docs/transformers/performance) — Trainer 端开 ckpt + 各种省显存技巧的 catalog
- [Liger Kernel](https://github.com/linkedin/Liger-Kernel) — 当代 fused kernel 实战库，开就省 ~30% activation
- [Stas Bekman — ML Engineering Book / Memory Optimization](https://github.com/stas00/ml-engineering/tree/master/training/performance) — 业界最完整的显存调优手册
- 推荐继续读本教程的 **5.3 FlashAttention**（消去 $T^2$ 项 activation 的 IO-aware kernel）、**8.3 LoRA / QLoRA**（PEFT 的极致省显存）、**7.6 Triton kernel 入门**（手写 fused kernel 进一步省 activation）
