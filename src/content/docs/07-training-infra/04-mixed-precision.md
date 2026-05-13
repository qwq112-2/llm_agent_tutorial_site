---
title: "7.4 混合精度：fp16 / bf16 / fp8 + Loss Scaling"
description: "混合精度训练 = forward / backward 用低精度（bf16 / fp8）省显存与算力，optimizer step 用 fp32 master weight 保数值稳定；fp16 时代靠 Loss Scaling 续命，bf16 时代直接 autocast 一把梭，fp8 时代靠 E4M3 + E5M2 + per-tile scaling 把 671B 模型端到端训出来。"
---

> ⏱ 预计阅读 30 分钟 ｜ 难度 ★★ ｜ 前置：1.5 PyTorch 工作流与显存 / dtype 心智模型

## 一句话本节讲什么

混合精度训练 = **forward / backward 用低精度（bf16 / fp8）省显存与算力，optimizer step 用 fp32 master weight 保数值稳定**；fp16 时代靠 Loss Scaling 续命，bf16 时代直接 autocast 一把梭，fp8 时代靠 E4M3 + E5M2 + per-tile scaling 把 671B 模型端到端训出来。

---

## 1. Mental model（直觉）

1.5 节已经把 dtype 全谱系铺过：fp32 / fp16 / bf16 / fp8 字节数与精度的对照表你应该已经记熟。本节要把这张表"装进训练循环"——回答一个具体问题：**一个 forward + backward + step 的过程，每个张量到底用什么精度？**

混合精度训练的核心 mental model 是 **"三层结构"**：

```
                ┌─── master weight (fp32) ─── 永久保存的"真"权重
                │           │
                │           ▼ cast
                │    ┌─ working weight (bf16/fp8) ──┐
                │    │                              │
                │    ▼                              ▼
            optimizer ◄── grad (fp32) ◄── grad (bf16/fp8) ◄── backward
            step                ▲                     ▲
                │                                     │
                │            forward / backward 在低精度
                └────────── 写回 master weight ───────┘
```

为什么必须有"master weight in fp32"这一层？因为 **bf16 的 7 位尾数精度太低**——一个 1.0 的权重加上 lr=1e-4 × grad=1e-4 = 1e-8 的更新，bf16 根本表达不出来这个差异（最近的可表示数仍是 1.0），更新被吃掉了。fp32 master weight 是一道"数值精度的保险"，让微小的累积更新不丢失。

第二个直觉：**bf16 在 LLM 训练里完胜 fp16，关键不是精度而是动态范围**。fp16 max 只有 65504，LLM 中梯度一爆就 inf；bf16 与 fp32 同 8 位指数（max ≈ $3 \times 10^{38}$），几乎不会上溢。fp16 时代发明的 Loss Scaling 是为了"垫起小梯度免被截断成 0"——bf16 来了之后，整套 scaling 机制变成历史包袱。**自 2022 年以后，LLaMA / Qwen / DeepSeek 没有一家用 fp16 跑 LLM 预训练**。

第三个直觉：**fp8 不是"再缩一半"那么简单**。fp8 只有 4 位指数 + 3 位尾数（E4M3），一个张量里数值跨度稍大就量化失真，所以 fp8 训练必须配 **per-tensor / per-tile scaling**——每个张量动态记一个 scale，把数值压到 fp8 能表示的"窗口"里。DeepSeek-V3 把这套搞到 671B 端到端训出来，是 2024 年 LLM infra 最硬核的工程之一。

---

## 2. 公式与原理

### 2.1 dtype 速查（与 1.5 节呼应，本节关注训练侧含义）

| dtype | 字节 | 指数 | 尾数 | max | min normal | 是否需 Loss Scaling | LLM 训练角色 |
|---|---|---|---|---|---|---|---|
| fp32 | 4 | 8 | 23 | $3.4 \times 10^{38}$ | $1.2 \times 10^{-38}$ | — | master weight、optim state |
| fp16 | 2 | 5 | 10 | $65504$ | $6 \times 10^{-5}$ | **必需** | V100 时代，已退役 |
| bf16 | 2 | 8 | 7 | $3.4 \times 10^{38}$ | $1.2 \times 10^{-38}$ | 不需要 | **现役 LLM 训练标配** |
| fp8 E4M3 | 1 | 4 | 3 | $448$ | $1.95 \times 10^{-3}$ | 配 per-tensor scaling | H100 forward / activation |
| fp8 E5M2 | 1 | 5 | 2 | $57344$ | $6 \times 10^{-5}$ | 配 per-tensor scaling | H100 backward / gradient |

**关键差异**：

- fp16 vs bf16：**fp16 范围窄（5 位指数）但精度高（10 位尾数），bf16 范围与 fp32 相同（8 位指数）但精度低（7 位尾数）**。LLM 训练对范围敏感（梯度跨好几个数量级），对精度不敏感（accumulation 已被 fp32 master weight 兜底），所以 bf16 赢。
- fp8 E4M3 vs E5M2：**E4M3 范围 ±448，精度更好；E5M2 范围 ±57344，精度更差**。Forward 中的激活值经过 LayerNorm 后通常落在 ±10 区间，用 E4M3 精度更高；backward 中的梯度跨度大（不同层差几个量级），用 E5M2 范围更安全。

### 2.2 混合精度的三层结构（必背）

完整流程，以 bf16 mixed precision + AdamW 为例：

1. **存**：master weight $W_{\text{fp32}}$ 永久驻 GPU
2. **cast**：每步 forward 前把 $W_{\text{fp32}} \to W_{\text{bf16}}$（autocast 自动做）
3. **forward**：$y = f(W_{\text{bf16}}, x_{\text{bf16}})$，激活全 bf16，省显存约一半
4. **backward**：$g_{\text{bf16}} = \partial L / \partial W_{\text{bf16}}$，梯度也是 bf16
5. **upcast**：optim step 前把 $g_{\text{bf16}} \to g_{\text{fp32}}$
6. **step**：$W_{\text{fp32}} \leftarrow W_{\text{fp32}} - \eta \cdot \text{Adam}(g_{\text{fp32}}, m_{\text{fp32}}, v_{\text{fp32}})$
7. **回到第 2 步**

这套流程的显存账（7B model）：

$$M = \underbrace{P \cdot 2}_{W_{\text{bf16}}} + \underbrace{P \cdot 2}_{g_{\text{bf16}}} + \underbrace{P \cdot 4}_{W_{\text{fp32}} \text{ master}} + \underbrace{2 \cdot P \cdot 4}_{m, v \text{ fp32}} = 16P = 112 \text{ GB}$$

注意比 1.5 节算的 84 GB 多了 28 GB——那是**没考虑 master weight**的简化版。Megatron / DeepSpeed 的 ZeRO 论文里通常把这 16P 拆成 "2P (bf16 weight) + 2P (bf16 grad) + 12P (fp32 weight + m + v)"，这就是 ZeRO-1/2/3 各 shard 哪几项的分母。

### 2.3 fp16 + Loss Scaling 的数学

fp16 的 normal range 下界是 $2^{-14} \approx 6 \times 10^{-5}$（subnormal 可到 $2^{-24}$ 但精度极差）。LLM 训练中的梯度经常落在 $10^{-7}$ 量级，**直接被截断成 0**——这就是 fp16 的"小梯度下溢"问题。

**Loss Scaling 的解法**：把 loss 乘一个大数 $S$（典型 $S = 2^{16}$），由链式法则梯度也跟着乘 $S$：

$$L' = S \cdot L \quad \Rightarrow \quad \frac{\partial L'}{\partial W} = S \cdot \frac{\partial L}{\partial W}$$

放大后的梯度落在 $10^{-7} \cdot 2^{16} \approx 6.5 \times 10^{-3}$，远高于 fp16 下界，安全表示。Optimizer step 之前再除回 $S$，恢复真梯度。

**动态调整 $S$**（PyTorch `GradScaler` 做的事）：

- 监测每次 backward 后的梯度是否含 inf / NaN
- 如果有 → 说明 $S$ 太大造成上溢 → **降 $S$（典型乘 0.5）+ 跳过本次 step**
- 如果连续 N 步（典型 N=2000）都没有 inf → **升 $S$（典型乘 2）**
- 起始 $S = 2^{16}$，平稳后通常稳定在 $2^{15}$ ~ $2^{17}$

### 2.4 bf16 完胜 fp16 的工程原因

| 维度 | fp16 | bf16 | 谁赢 |
|---|---|---|---|
| 动态范围 | $\pm 65504$ | $\pm 3.4 \times 10^{38}$ | **bf16** |
| 精度 | 10 位尾数（~3-4 位十进制） | 7 位尾数（~2-3 位十进制） | fp16 |
| 是否需 GradScaler | **必需** | 不需要 | bf16 |
| 训练 NaN 概率 | 高（梯度爆炸即上溢） | 极低 | bf16 |
| LLM 训练实证 | 早期 GPT-3、OPT 用过 | LLaMA / Qwen / DeepSeek 全系 | bf16 |
| 硬件支持 | V100 起 | A100 / H100 / TPU / MI300 | 平 |

bf16 唯一的弱点是**精度低**——比如计算 LayerNorm 内部的 $\sum x^2$ 时，bf16 的 7 位尾数容易丢失数值精度。解决方案是 **autocast 白名单**：matmul / conv 这种"计算密集 + 数值平滑"的 op 用 bf16，sum / softmax / norm 这种"精度敏感"的 op 自动 fallback fp32。这套规则 PyTorch 写死在内核里，平时不用管。

### 2.5 fp8 训练（H100 + Transformer Engine + DeepSeek-V3）

fp8 训练的核心挑战：**8 位的动态范围太窄**，必须在每个张量上配一个 **scale factor** 把数值"压"进表示范围。

**两种 fp8 格式的分工**：

- **E4M3（4 exp + 3 mantissa）**：max = 448，精度较高 → 用于 **forward 中的 activation 和 weight**（数值经过 LayerNorm 后通常在 ±10 内）
- **E5M2（5 exp + 2 mantissa）**：max = 57344，精度较低 → 用于 **backward 中的 gradient**（跨层梯度量级差很大，需要更宽的范围）

**Per-tensor scaling 的工作方式**（NVIDIA Transformer Engine 默认方案）：

每个张量 $X$ 关联一个 scale factor $s$。存储时存 $X_{\text{fp8}} = \text{cast\_to\_fp8}(X / s)$，使用时还原 $X \approx s \cdot X_{\text{fp8}}$。$s$ 的选取目标是让 $X / s$ 的最大绝对值刚好接近 fp8 的 max（448 或 57344），充分利用表示范围。

$s$ 是动态更新的——记录最近几步的 $\max |X|$，按一个 **amax history** 滚动调整。Transformer Engine 内部叫 `scaling_recipe`。

**DeepSeek-V3 的关键改进（fine-grained scaling）**：

per-tensor scaling 的弱点：一个张量内若有 outlier（比如 attention 的某个 head 比其他大 100 倍），$s$ 被 outlier 拉大，正常元素全被压成 fp8 的极小值，量化误差爆炸。

DeepSeek-V3 的解法：**tile-wise scaling**（也叫 block-wise）。把张量切成 128×128 的 tile，每个 tile 一个独立 scale。Outlier 被局限在自己的 tile 里，不影响其他 tile 的精度。代价是 scale factor 的存储与计算开销变大，但精度大幅提升。

**fp8 训练里"必须保留高精度"的位置**：

- **master weight**：fp32（数值稳定的最后防线）
- **optimizer state（m, v）**：fp32
- **gradient reduce / all-reduce**：bf16 或 fp32（fp8 累加误差大）
- **LayerNorm / RMSNorm 的中间累加**：fp32（小数值平方求和容易上溢或丢精度）
- **embedding / output projection**：通常 bf16（精度敏感）

DeepSeek-V3 的实际配方是 **"大部分 GEMM 用 fp8 + 关键 op 保留 bf16/fp32"** 的混合，吞吐相比 bf16 baseline 提升约 **1.5-2×**，是首个开源端到端 fp8 训练成功的 671B 级 MoE。

---

## 3. 最小代码示例

### 3.1 完整 bf16 AMP 训练循环（推荐写法）

```python
import torch
import torch.nn as nn
from torch.amp import autocast

def train_bf16(model, loader, criterion, optimizer, scheduler, device='cuda'):
    model.train()
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        # autocast 包整个 forward + loss，不能只包一半
        with autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(x)                # 内部 op 自动选 bf16
            loss = criterion(logits, y)      # loss 计算也在 autocast 区域内

        # bf16 不需要 GradScaler，直接 backward / step
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # grad clip 直接做
        optimizer.step()
        scheduler.step()
```

**要点**：

- `autocast` 必须包住 forward + loss 的全部计算，否则部分 tensor 是 fp32 部分是 bf16，会触发隐式 cast 报错或精度丢失。
- bf16 训练完全**不需要 `GradScaler`**——写它不报错（PyTorch 对 bf16 把 scaler 做成 no-op），但会让读代码的人误以为需要 scale。
- `clip_grad_norm_` 在 bf16 下可以直接调用，不用 unscale。

### 3.2 fp16 + GradScaler 训练循环（V100 等老卡才用）

```python
import torch
from torch.amp import autocast, GradScaler

def train_fp16(model, loader, criterion, optimizer, scheduler, device='cuda'):
    model.train()
    scaler = GradScaler()  # 默认 init_scale=2**16，自动动态调整

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type='cuda', dtype=torch.float16):
            logits = model(x)
            loss = criterion(logits, y)

        # 关键：scale loss → backward → unscale → clip → step → update
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)              # 必须先 unscale 才能正确 clip
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)                  # 内部检测 inf：若有则跳过 step
        scaler.update()                         # 根据是否出现 inf 调整 scale
        scheduler.step()
```

**要点**：

- 顺序严格：`scale(loss).backward()` → `unscale_()` → `clip_grad` → `scaler.step()` → `scaler.update()`，错一个都会导致 scale 不正确。
- `scaler.step()` 内部会检查梯度是否含 inf，**含 inf 就跳过 step**——这一步参数没更新但 lr scheduler 还会走（这是有意为之，避免 schedule 错位）。
- `scaler.state_dict()` 也要 save / load，否则 resume 后 scale 从 init 重来会发散一波。

### 3.3 NVIDIA Transformer Engine fp8（H100 才能跑）

```python
import torch
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import DelayedScaling, Format

# 用 te.Linear 替代 nn.Linear，te.LayerNorm 替代 nn.LayerNorm
model = torch.nn.Sequential(
    te.LayerNorm(1024),
    te.Linear(1024, 4096),     # 内部走 fp8 GEMM
    torch.nn.GELU(),
    te.Linear(4096, 1024),
).cuda()

# fp8 scaling recipe：HYBRID = forward E4M3 + backward E5M2
fp8_recipe = DelayedScaling(
    fp8_format=Format.HYBRID,
    amax_history_len=16,        # 用最近 16 步的 max abs 估 scale
    amax_compute_algo='max',
)

for x, y in loader:
    optimizer.zero_grad()
    # fp8_autocast 上下文里，te 层走 fp8；非 te 层（如 GELU）保持 bf16
    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        out = model(x.cuda().bfloat16())
        loss = criterion(out, y.cuda())
    loss.backward()
    optimizer.step()
```

**要点**：

- `te.Linear` / `te.LayerNorm` 等是 Transformer Engine 提供的 fp8-aware 层，内部用 fp8 GEMM + 自动 scale 管理；普通 `nn.Linear` 进 fp8 区域不会被自动转换。
- `Format.HYBRID` = forward E4M3 + backward E5M2，是官方推荐配方；纯 E4M3 或纯 E5M2 也可选但通常较差。
- `fp8_autocast` 必须包整个 model forward；非 te 层会自动 fallback bf16 / fp32。

---

## 4. 工程踩坑与经验

- ❗ **bf16 训练写 `GradScaler` 是误导**。PyTorch 对 bf16 的 GradScaler 做了 no-op 兜底（不会报错），但读代码的人看到 scaler 会以为需要 scale。**bf16 直接 `loss.backward()` + `optimizer.step()` 即可**，scaler 只在 fp16 路径写。

- ❗ **fp16 训练时 LayerNorm / softmax 的中间累加必须 cast 到 fp32**。LayerNorm 算 $\sum x^2 / N$ 时，hidden=4096 的张量平方求和很容易超过 fp16 的 65504。HuggingFace Transformers / nanoGPT 都在 LayerNorm forward 里手动 `x.float()` 做累加再 cast 回去，否则 fp16 训练会无声地数值漂移。bf16 没这个问题（动态范围够），但精度敏感的 op autocast 也会自动 fallback fp32。

- ❗ **autocast 必须包整个 forward + loss，不能只包一半**。常见错误：把 `autocast` 只包到 model forward，loss 在 autocast 外算 → 部分 op 在 fp32 部分在 bf16，autograd 在反向传播时遇到 dtype 不匹配会触发隐式 cast 或 silent 数值问题。**正确做法**：从 `model(x)` 到 `criterion(logits, y)` 全部进 autocast。

- ❗ **fp16 训练中 `scaler.unscale_(optimizer)` 必须在 grad clip 之前调用**。否则你 clip 的是放大 $S$ 倍后的梯度，clip threshold 1.0 实际等价于 $1.0 / S$，clip 不到位 → 梯度爆炸或 NaN。bf16 不需要 unscale，可以直接 clip。

- ❗ **fp8 训练当前只在 H100 / H200 / MI300X 等 Hopper+ GPU 上有意义**。A100 没有 fp8 tensor core（虽然能软模拟但比 bf16 还慢），跑 fp8 等于自找麻烦。Transformer Engine 在 A100 上 import 不报错但实际走的是 bf16 fallback。

- ❗ **DeepSeek-V3 fp8 训练用的是自家魔改的 fine-grained scaling，HF / Megatron 公版 Transformer Engine 不能直接复现 1.5-2× 加速**。开源生态里 NVIDIA Transformer Engine + Megatron-LM 已经能跑 fp8，但对吞吐与稳定性的优化远不如 DeepSeek 内部版。复现 fp8 训练效果一般要自己写 tile-wise scaling kernel。

- ❗ **混合精度与 ZeRO / FSDP 的搭配**：optimizer state 必须保持 fp32 master weight，否则训练塌。FSDP 的 `MixedPrecision` 配置三个 dtype：`param_dtype=bf16`（forward 时 cast）、`reduce_dtype=fp32`（all-reduce 用 fp32 防累积误差）、`buffer_dtype=bf16`。把 `reduce_dtype` 改 bf16 可以再省通信带宽 2×，但大模型上风险高（梯度 all-reduce 累加误差），DeepSeek-V3 也没敢这么做。

- ❗ **fp16 / bf16 推理时 batch=1 与 batch=N 输出可能略不一样**。原因：matmul 内部 reduce 顺序不同（不同 batch 走不同 kernel），fp16/bf16 累加非结合律 → bit-level 输出不一致。这**不是 bug**，是浮点数本身的特性，单元测试要用 `torch.allclose(rtol=1e-3, atol=1e-3)` 而非 `torch.equal`。

- ❗ **AMP 训练时 `loss.item()` 与 `grad_norm` 打日志要谨慎**。每次 `.item()` 强制 device → host 同步，打断 GPU 流水线。建议每 N 步（如 N=10）才打一次日志，平时只 `loss.detach()` 在 GPU 上累加。

- ❗ **从 fp16 checkpoint 加载到 bf16 训练**会丢精度但通常没事；反过来 bf16 ckpt → fp16 训练**很可能立刻 NaN**（master weight 在 bf16 时已经累积了一些 fp16 表达不了的更新，cast 回 fp16 时数值溢出）。迁移精度方向时优先 fp16 → bf16，不要反向。

---

## 5. 经典 paper

- **Micikevicius et al., 2017 — Mixed Precision Training** — fp16 / AMP 的开山 paper，首次提出 master weight in fp32、loss scaling、动态 scale factor 三件核心技术。本节 §2.3 fp16 + GradScaler 的所有"为什么"都来自这篇 §3。读完能彻底理解为什么 fp16 路径必须 scale 而 bf16 不需要。
- **Micikevicius et al., 2022 — FP8 Formats for Deep Learning** — NVIDIA + Intel + ARM 联合提出的 fp8 标准 paper，定义了 E4M3 与 E5M2 两种格式、per-tensor scaling 协议、训练与推理的不同精度需求。Hopper 架构 fp8 tensor core 的 spec 与这篇完全对齐，是理解 §2.5 的硬件基础。
- **DeepSeek-AI, 2024.12 — DeepSeek-V3 Technical Report** — 首个开源端到端 fp8 训练成功的 671B MoE，§3.3 "FP8 Training" 一节详细讲了 fine-grained per-tile scaling、E4M3/E5M2 分工、哪些 op 强制 bf16 / fp32 的工程细节。读它能知道"工业级 fp8 训练"和"toy fp8 demo"差在哪。

---

## 6. 自测与面试题

**Q1（dtype 对比）**：fp16 vs bf16，分别给出（动态范围 / 精度 / 是否需要 GradScaler / LLM 训练用谁）4 个维度的对比。

<details>
<summary>Answer sketch</summary>

| 维度 | fp16 | bf16 |
|---|---|---|
| 动态范围 | $\pm 65504$（5 位指数） | $\pm 3.4 \times 10^{38}$（8 位指数，与 fp32 同） |
| 精度 | 10 位尾数（~3-4 位十进制） | 7 位尾数（~2-3 位十进制） |
| 是否需 GradScaler | **必需**（小梯度下溢、大梯度上溢都会发生） | 不需要（动态范围够，autocast 一把梭） |
| LLM 训练用谁 | 已退役（早期 GPT-3 用过） | **现役标配**（LLaMA / Qwen / DeepSeek 全系） |

加分：

- 指出 bf16 胜出的关键是**动态范围**而不是精度——bf16 精度其实**更低**（7 位 < 10 位）。
- 指出"训练对范围敏感、对精度不敏感"是因为 fp32 master weight 兜底了精度。
- 指出 fp16 时代留下的 LayerNorm 内部 `x.float()` 累加 trick 在 bf16 下大多不需要（autocast 自动 fallback）。

</details>

**Q2（fp8 格式）**：fp8 训练为什么用 E4M3 + E5M2 两种格式？分别用在什么阶段？

<details>
<summary>Answer sketch</summary>

要点：

- **E4M3（4 exp + 3 mantissa, max=448）**：精度高、范围窄。用于 **forward 中的 activation 与 weight**——这些张量经过 LayerNorm 后数值通常在 ±10 范围，E4M3 的精度优势能直接转化为模型质量。
- **E5M2（5 exp + 2 mantissa, max=57344）**：精度低、范围广。用于 **backward 中的 gradient**——梯度跨多层量级差很大（比如靠近 output 的层梯度大、靠近 input 的层梯度小），需要更宽的动态范围避免上下溢。
- 这种"forward E4M3 + backward E5M2"的组合在 NVIDIA Transformer Engine 里叫 `Format.HYBRID`，是默认推荐配方，DeepSeek-V3 也用这个组合。
- 加分：指出 fp8 必须配 **per-tensor / per-tile scaling factor**，否则 8 位根本承载不了 LLM 训练里的数值跨度；DeepSeek-V3 的 fine-grained tile-wise scaling 是其训练成功的关键工程改进。

</details>

**Q3（实战）**：你训 13B 模型，bf16 + ZeRO-2 仍然 OOM，列出至少 3 个 dtype / 精度方向的优化（不算 model 切分）。

<details>
<summary>Answer sketch</summary>

至少 3 条 dtype / 精度向的方案：

1. **降到 fp8**（如果在 H100 上）：把 GEMM 从 bf16 → fp8（E4M3 forward + E5M2 backward），weight + activation 显存再省一半，吞吐提升 1.5-2×。需要换用 NVIDIA Transformer Engine 的 `te.Linear` / `te.LayerNorm`。代价：optimizer state 仍然 fp32，所以总显存只省了"weight + grad"那部分，不是全部减半。
2. **gradient 用 bf16，但 optimizer state 改 fp16 / bf16**（牺牲精度换显存）：默认 AdamW 的 m / v 是 fp32（每参数 8 bytes），改成 bf16 m / v 可以省一半 optim state 显存（每参数 4 bytes）。**风险**：长训中 m / v 累积精度损失，可能影响收敛——bnb 的 8-bit AdamW 是更激进的方案（每参数 2 bytes），实际可用但需要测过 loss curve 确认无明显退化。
3. **CPU offload optimizer state**（DeepSpeed / FSDP 都支持）：把 fp32 master weight + m + v 全部 offload 到 CPU RAM，GPU 只存 bf16 working weight + grad。每步 step 时 GPU → CPU → GPU 数据搬运，吞吐降 30-50%，但显存能再省 12P bytes（7B 模型省 84 GB），是 OOM 死线时的杀手锏。
4. **activation recomputation**（Module 7.5 详讲）：严格说不算 dtype，但与精度无关地能砍 activation 显存 50-80%。可以与 fp8 / bf16 / offload 叠加。

加分：指出"换 dtype 不是免费午餐"——fp8 需要硬件、optim state 降精度需要测稳定性、offload 牺牲吞吐，工程要权衡训练时间 vs 显存预算。

</details>

---

## 7. 延伸阅读

- [PyTorch AMP Examples](https://pytorch.org/docs/stable/notes/amp_examples.html) — 官方 autocast + GradScaler 文档，包含 grad accumulation、grad penalty、多 GPU 等边角 case 的标准模板
- [NVIDIA Transformer Engine 文档](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html) — fp8 训练的官方实现，包含 `DelayedScaling` recipe、与 Megatron-LM 集成、与 FSDP 协同的工程指南
- [DeepSeek-V3 Technical Report (arxiv 2412.19437)](https://arxiv.org/abs/2412.19437) — fp8 端到端训练的工业级实证，§3.3 fp8 训练细节是目前公开材料里最翔实的
- [Mixed Precision Training: Theory and Practice (NVIDIA Blog)](https://developer.nvidia.com/blog/mixed-precision-training-deep-neural-networks/) — 把 Micikevicius 2017 的核心思想用工程语言重写，配 PyTorch 代码，比原 paper 易读
- 推荐继续读本教程的 **7.5 显存优化：Activation Recomputation / Selective / Offload**——本节解决了 weight / grad / optim 的精度账，下一节解决 activation 这块"剩下的大头"
