---
title: "1.5 PyTorch 工作流与显存 / dtype 心智模型"
description: "把 PyTorch 训练循环的 5 步模板写到肌肉记忆里，建立\"参数量 → 显存\"的心算公式，弄清 fp32 / bf16 / fp8 各自的取舍——这是后面 Module 7（FSDP / TP / fp8）、Module 8（SFT 实操）、Module 11（推理量化）所有训练 / 推理工程的共同底座。"
---

> ⏱ 预计阅读 35 分钟 ｜ 难度 ★★ ｜ 前置：1.1 反向传播、1.2 优化器

## 一句话本节讲什么

把 PyTorch 训练循环的 5 步模板写到肌肉记忆里，建立**"参数量 → 显存"的心算公式**，弄清 fp32 / bf16 / fp8 各自的取舍——这是后面 Module 7（FSDP / TP / fp8）、Module 8（SFT 实操）、Module 11（推理量化）所有训练 / 推理工程的共同底座。

---

## 1. Mental model（直觉）

写训练代码之前，先建立两个画面感。

**画面一：训练循环就是一个固定 5 步的 ritual**。任何 PyTorch 训练，无论是 MLP、CNN、Transformer 还是 7B LLM，骨架都是同一个：清梯度 → forward → backward → optimizer step →（可选）scheduler step。**这 5 步顺序不能错、不能漏**。新手最常见的 bug 是把 `optimizer.zero_grad()` 写在 `loss.backward()` 之后（梯度被清零，模型不学），或者忘了 `optimizer.step()`（loss 一直不降）。

**画面二：GPU 显存是一块"被四类张量瓜分"的内存**。打开 `nvidia-smi` 看到 78 GB 占用，那 78 GB 是被四类东西吃掉的——**weight、gradient、optimizer state、activation**。前三项与 batch 无关、只跟参数量与 dtype 挂钩；activation 跟 `batch × seqlen × hidden × layers` 强相关。这就是为什么 7B 模型 fp32 训练**光参数那三项就要 112 GB**——A100 80G 一张卡装不下，必须 mixed precision + ZeRO 才能 fit，这是 Module 7 全章存在的原因。

ASCII 图示——80 GB GPU 的显存预算：

```
A100 80G ─┬─ Weight        7B × 2B = 14 GB   ┐
          ├─ Gradient      7B × 2B = 14 GB   │ "三件套" 共 70 GB
          ├─ Optim (Adam) 2× 7B × 4B = 56 GB ┘ （bf16 weight + fp32 optim 配方）
          ├─ Activation   随 batch × seqlen 浮动 ── 推理时 = 0；训练时常占 10-30 GB
          ├─ Workspace    cuBLAS / NCCL 缓冲，约 1-3 GB
          └─ ★ 剩下的就是 OOM 边界
```

把这两个画面记牢，本节剩下的内容只是把它们填实。

---

## 2. 公式与原理

### 2.1 训练循环模板（5 步）

```python
model.train()
for batch in dataloader:
    optimizer.zero_grad(set_to_none=True)   # 1. 清梯度（用 set_to_none 更快）
    output = model(batch.input)             # 2. forward
    loss = criterion(output, batch.label)
    loss.backward()                         # 3. backward（autograd 自动算梯度）
    optimizer.step()                        # 4. 用梯度更新参数
    scheduler.step()                        # 5. lr schedule（可选）
```

eval 循环必须**两个开关一起拉**：

```python
model.eval()                  # 切换 BN / Dropout 行为（推理模式）
with torch.no_grad():         # 关闭 autograd（不建图、不存 activation）
    for batch in val_loader:
        out = model(batch.input)
```

`model.eval()` 与 `torch.no_grad()` 是正交的两件事——前者改变层的**前向行为**（Dropout 不丢弃、BatchNorm 用 running stats），后者关闭**反向准备**（不存中间 activation、不建计算图）。少做任何一个都会出错：只 `model.eval()` 不 `no_grad()`，显存照样炸（activation 全留着）；只 `no_grad()` 不 `model.eval()`，Dropout 还在随机丢，eval 指标抖动。

### 2.2 显存四件套公式

设参数量为 $P$（如 7B = $7 \times 10^9$），dtype 字节数为 $b$（fp32: 4，bf16: 2，fp8: 1）。

- **Weight**：$M_w = P \cdot b$
- **Gradient**：$M_g = P \cdot b$（与 weight 同 dtype）
- **Optimizer state**：取决于优化器
  - SGD：$0$（无 state）
  - SGD + momentum：$P \cdot 4$（一份 momentum，fp32）
  - Adam / AdamW：$2 \cdot P \cdot 4 = 8P$（一阶 momentum $m$ + 二阶 $v$，**通常都用 fp32**，即使 weight 是 bf16）
- **Activation**：$M_a \propto B \cdot L \cdot H \cdot N_{\text{layer}}$，其中 $B$ = batch、$L$ = seqlen、$H$ = hidden、$N_{\text{layer}}$ = 层数。Transformer 训练时 activation 通常**与三件套同量级甚至更大**（Module 7.5 讲 activation recomputation 就是为了砍这一项）

**算一笔账：7B fp32 全精度训练用 AdamW**：

$$M_{\text{weight}} + M_{\text{grad}} + M_{\text{optim}} = 7\text{B} \cdot (4 + 4 + 8) = 112 \text{ GB}$$

A100 80G 单卡装不下、H100 80G 也装不下——必须 8 卡 ZeRO-3 才能勉强 fit。

**换 bf16 mixed precision**（weight + grad bf16，optim state fp32）：

$$M = 7\text{B} \cdot (2 + 2 + 8) = 84 \text{ GB}$$

仍然爆 80G 单卡，但 8 卡 ZeRO-2 就能很轻松，对比 fp32 省了 28 GB / model。

**再换 ZeRO-3 + bf16 + 8 卡**（参数 / 梯度 / 优化器都 shard 到 8 卡）：

$$M_{\text{per GPU}} \approx \frac{84}{8} = 10.5 \text{ GB}$$

再叠加 activation 也能塞进 40G 卡——这就是当前主流 7B 训练的最小配方。

> ⚠️ 上面计算**忽略了 activation、workspace、checkpoint buffer、KV cache（推理才有）**等。实际工程估算建议在算出三件套后再额外预留 30-50% 给 activation。

### 2.3 dtype 全谱系

| dtype | 字节 | 指数位 | 尾数位 | 动态范围 | 精度（约） | 用途 |
|---|---|---|---|---|---|---|
| **fp32** | 4 | 8 | 23 | $\pm 10^{38}$ | 7 位十进制 | 传统标准，optimizer state |
| **fp16** | 2 | 5 | 10 | $\pm 65504$ | 3-4 位 | **已被 bf16 取代**，需 GradScaler |
| **bf16** | 2 | 8 | 7 | $\pm 10^{38}$ | 2-3 位 | **LLM 训练标配**，与 fp32 同 range |
| **fp8 E4M3** | 1 | 4 | 3 | $\pm 448$ | ~1 位 | H100 forward；权重/激活量化 |
| **fp8 E5M2** | 1 | 5 | 2 | $\pm 57344$ | ~1 位 | H100 backward（梯度 range 大）|
| **int8 / int4** | 1 / 0.5 | — | — | 整数 | — | 推理量化（GPTQ / AWQ）|

**关键直觉**：

- **指数位决定能表示多大 / 多小**（动态范围）
- **尾数位决定相邻两个数差多少**（精度 / 分辨率）
- **fp16 vs bf16 的根本差异在指数位**：fp16 只有 5 位指数，超过 65504 就上溢成 inf，LLM 中常见的较大梯度值会爆炸。bf16 有 8 位指数（与 fp32 同），动态范围完全够用，代价是尾数只有 7 位精度差一点——但训练对动态范围的敏感度远高于对精度的敏感度，所以 **bf16 几乎全面胜出**

**fp8 训练的玩法**（H100 + Transformer Engine）：forward 用 E4M3（精度优先，动态范围够），backward 用 E5M2（动态范围优先，因为梯度的量级跨度更大），optimizer state 仍然 fp32。DeepSeek-V3 是首个端到端 fp8 训练验证的开源 LLM，吞吐对比 bf16 可提升 1.5-2×，详见 Module 7.4。

### 2.4 Mixed precision 的工程模式

PyTorch 提供 `torch.autocast` + `torch.cuda.amp.GradScaler` 两件套：

```python
from torch.amp import autocast, GradScaler

scaler = GradScaler()  # bf16 训练时不需要，fp16 才需要

for batch in dataloader:
    optimizer.zero_grad(set_to_none=True)
    with autocast(device_type='cuda', dtype=torch.bfloat16):
        output = model(batch.input)         # 内部 op 自动选低精度
        loss = criterion(output, batch.label)
    # bf16 不需要 scaler，直接：
    loss.backward()
    optimizer.step()
```

**为什么 bf16 不需要 GradScaler**：fp16 的 max value 是 65504，梯度容易上溢成 inf；GradScaler 的做法是**先把 loss 放大 $S$ 倍**（典型 $S = 2^{16}$）再 backward，让小梯度不会下溢成 0，最后 unscale。bf16 动态范围与 fp32 相同，根本不会上溢，scaler 多此一举。

**autocast 内部规则**：matmul / conv 等"计算密集"op 用低精度（快、省显存），sum / softmax / norm 等"精度敏感"op 自动 fallback 到 fp32。这套白名单写在 PyTorch 源码 `torch/_amp/`，知道有这件事就够了，平时不用手动管。

---

## 3. 最小代码示例

完整的小 trainer 模板，带 mixed precision、grad accumulation、checkpoint save/load、scheduler，60 行内自包含：

```python
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader

def train_one_epoch(model, dataloader, criterion, optimizer, scheduler,
                    device, accum_steps=4, ckpt_path='ckpt.pt'):
    model.train()
    scaler = GradScaler(enabled=False)  # bf16 用不到；fp16 改 enabled=True
    optimizer.zero_grad(set_to_none=True)

    for step, (x, y) in enumerate(dataloader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        with autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(x)
            loss = criterion(logits, y) / accum_steps   # 关键：除以 accum_steps

        loss.backward()                                  # 梯度累加（不 step）

        if (step + 1) % accum_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        if (step + 1) % 1000 == 0:
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),       # 必须存，否则 lr 重启
                'scaler': scaler.state_dict(),
                'step': step,
            }, ckpt_path)

def load_ckpt(model, optimizer, scheduler, scaler, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler.load_state_dict(ckpt['scheduler'])
    scaler.load_state_dict(ckpt['scaler'])
    return ckpt['step']

# DataLoader 推荐参数（训练侧）：
# DataLoader(dataset, batch_size=32, num_workers=8, pin_memory=True,
#            persistent_workers=True, prefetch_factor=2, shuffle=True)
```

配套的"显存占用估算"工具——5 行，输入参数量 + dtype，输出 weight / grad / optim 三项 GB：

```python
def estimate_mem_gb(num_params: float, dtype_bytes: int = 2,
                   optim: str = 'adamw') -> dict:
    """num_params 单位是 B（如 7e9 写成 7e9 或 7_000_000_000）"""
    w = num_params * dtype_bytes
    g = num_params * dtype_bytes
    o = num_params * 8 if optim == 'adamw' else num_params * 4 if optim == 'sgd_momentum' else 0
    return {k: v / 1e9 for k, v in dict(weight=w, grad=g, optim=o, total=w+g+o).items()}

# >>> estimate_mem_gb(7e9, dtype_bytes=2, optim='adamw')
# {'weight': 14.0, 'grad': 14.0, 'optim': 56.0, 'total': 84.0}
```

**关键行解读**：

- 第 12 行 `loss / accum_steps` —— grad accumulation 的灵魂。如果不除，等价于把有效 batch 放大了 `accum_steps` 倍而 lr 没缩，梯度爆炸。
- 第 14-17 行 `if (step+1) % accum_steps == 0` —— 只有累计够 N 步才 `step + zero_grad`，中间 N-1 步只 backward 累加。
- 第 22 行 `'scheduler': scheduler.state_dict()` —— **resume 时最容易漏的一项**。少了它，restart 后 lr 会从 warmup 起点重新走一遍 cosine schedule，等价于把当前学习率拉回 peak，loss 会突然抖一下。
- 第 9 行 `non_blocking=True` —— 配合 DataLoader 的 `pin_memory=True`，让 CPU→GPU copy 与 GPU 计算重叠，吞吐 +20-30%。

---

## 4. 工程踩坑与经验

- ❗ **`model.eval()` ≠ `torch.no_grad()`**。前者切 BN / Dropout / 部分自定义层的 forward 行为，后者关 autograd 不存 activation。eval 时**两个都要**：少 `eval()` 会让 Dropout 随机丢、BN 用 batch stats 而不是 running stats（小 batch 直接崩）；少 `no_grad()` 显存涨 5-10 倍因为 activation 全留着。
- ❗ **fp16 上溢 NaN，bf16 几乎不会**。fp16 的 max 是 65504，LLM 训练中梯度乘 lr 后可能 > 65504 → inf → 整个 update 变 NaN。bf16 与 fp32 同样的 8 位指数（max ≈ $3 \times 10^{38}$），不会上溢。**结论：自 2022 年后所有 LLM 训练默认 bf16，不再用 fp16**，除非你是 V100 这种不支持 bf16 的老卡。
- ❗ **`num_workers > 0` 在 macOS / Windows 必须 `if __name__ == '__main__':` 包住**。这两个平台的 multiprocessing 用 spawn 而非 fork，子进程会重新 import 主脚本，没保护就无限递归 spawn → RuntimeError。Linux 用 fork 没事，但跨平台代码统一加保护是好习惯。
- ❗ **`pin_memory=True` 提速 30-50% 但吃 RAM**。pinned memory 是不能被换出 swap 的物理内存，每个 worker 都会预分配一份；多卡训练 8 个 DataLoader × 8 workers × 大 batch 可能让宿主机 RAM 爆掉。监控 `free -h`，OOM 先关 pin_memory。
- ❗ **`optimizer.zero_grad(set_to_none=True)` 比默认快**。默认 `zero_grad()` 是把每个 grad tensor `fill_(0)`（一次写操作 + 内存读写），`set_to_none=True` 直接把 `param.grad` 引用置 None（下次 backward 时再分配）。后者**更快、显存峰值更低**，PyTorch 1.7+ 推荐默认开。
- ❗ **checkpoint resume 必须 load scheduler.state_dict()**。否则 cosine / warmup schedule 重新从 step 0 算，lr 突然跳回 peak，loss 立刻抖动甚至发散。同理 `GradScaler` 也要 save / load。
- ❗ **grad accumulation 的 loss 必须除以 accumulation_steps**。直觉：累加 N 次梯度 = 一次大 batch 的梯度和（不是均值），如果 loss 不归一化，相当于 lr 隐式放大了 N 倍，发散是迟早的。BCE / CE 默认是 mean reduction，所以累加后要除回去。
- ❗ **DDP / FSDP 训练别直接 `loss.item()`**。`.item()` 会强制 device → host 同步，等所有 GPU 算完才返回，把流水线打断。改用 `loss.detach()` 在 GPU 上累计，每 N 步再 `all_reduce + .item()` 拉到 host 打日志。
- ❗ **`torch.compile` 与 autocast 顺序**：先 `model = torch.compile(model)` 再用 `autocast` 包 forward，编译图会把混合精度规则一起编译进去；反过来可能编译失败或丢失优化。

---

## 5. 经典 paper

- **Micikevicius et al., 2017 — Mixed Precision Training** — fp16 / AMP 的开山 paper，提出 loss scaling、master weight in fp32、动态 scale factor 三件核心技术。读它能彻底理解为什么 fp16 需要 GradScaler 而 bf16 不需要——本节 §2.4 的所有"为什么"都来自这篇。
- **NVIDIA Hopper Architecture Whitepaper（2022）** — H100 与 Transformer Engine 的硬件 spec，定义了 fp8 的 E4M3 / E5M2 两种格式与 per-tensor scaling 机制。Module 7.4 fp8 训练的硬件假设全在这里。读 §3.5 "FP8 Tensor Cores" 一节即可。
- **DeepSeek-V3 Technical Report（2024.12）** — 首个开源端到端 fp8 训练的 671B MoE，工程细节极其翔实：fine-grained per-tile scaling、accumulate in fp32、E4M3 forward + E5M2 backward 的实战验证。读 §3.3 "FP8 Training" 一节，对应本节 §2.3。

---

## 6. 自测与面试题

**Q1（计算）**：用 bf16 + AdamW 训一个 7B 模型，光算 weight + gradient + optimizer state（不考虑 activation），需要多少 GB 显存？写出公式与计算过程。

<details>
<summary>Answer sketch</summary>

公式：

$$M = P \cdot b_{\text{weight}} + P \cdot b_{\text{grad}} + 2 \cdot P \cdot b_{\text{optim}}$$

代入 $P = 7 \times 10^9$，bf16 weight + bf16 grad + fp32 optim：

$$M = 7 \times 10^9 \times (2 + 2 + 8) \text{ B} = 84 \times 10^9 \text{ B} = 84 \text{ GB}$$

加分点：

- 指出 AdamW optimizer state = $m + v$ 两份，**即使 weight 是 bf16，optim state 通常仍用 fp32** 防止累积误差。
- 指出这是 A100 80G **单卡装不下**的关键原因，必须 ZeRO-2 / 3 切分。
- 指出还没算 activation——实际工程要再预留 20-40 GB 给 activation（取决于 batch × seqlen），所以一卡更不可能。

</details>

**Q2（trade-off）**：fp16 vs bf16，为什么现代 LLM 训练一律用 bf16？至少答出 2 点。

<details>
<summary>Answer sketch</summary>

至少要点到：

- **动态范围**：bf16 与 fp32 同样 8 位指数（max ≈ $3 \times 10^{38}$），fp16 只有 5 位指数（max = 65504）。LLM 训练中梯度值容易超过 65504 → fp16 上溢 inf → NaN，bf16 无此问题。
- **不需要 GradScaler**：fp16 必须用 loss scaling 避免下溢，工程额外复杂度（save/load scaler state、调 scale factor、处理 inf 跳过 step）。bf16 直接 autocast 一把梭。
- **数值稳定**：训练 LLM 几个月不希望中途 NaN restart，bf16 的稳定性显著优于 fp16。
- **硬件支持**：A100 / H100 / TPU 都原生支持 bf16，吞吐与 fp16 持平。

差的回答：只说 "bf16 精度更高"——其实尾数 7 位 < fp16 的 10 位，bf16 精度更**低**。胜出点是动态范围，不是精度。

</details>

**Q3（实战）**：训练到 step 1000，loss 突然 NaN。请从 dtype / lr / data / scaler 四个方向各给一个具体排查动作。

<details>
<summary>Answer sketch</summary>

每个方向至少给一个可执行动作：

- **dtype**：检查是不是用 fp16 + 没开 GradScaler，或 GradScaler 的 scale 太大上溢；快速验证：换 bf16 重训 step 0-1500，看 NaN 还在不在。如果换 bf16 就好了，定位 fp16 上溢。
- **lr**：打印这一步前 N 个 step 的 lr 与 grad norm，看 grad norm 是不是早就在涨（gradient explosion 前兆）；加 `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)` 限幅；或把 peak_lr 砍半重训。
- **data**：检查 step 1000 附近的 batch 是否有异常 sample（label 全 0、input 含 inf / nan、token 长度异常长）；加 `assert not torch.isnan(x).any()` 在 dataloader 出口；定位到具体 sample 后人工查 raw data。
- **scaler**：如果用了 GradScaler，看 scaler state 是否在 step 1000 前频繁 `update`（说明 inf 不断出现），考虑降低初始 scale factor 或换 bf16 摆脱 scaler。
- 加分：说出"NaN 一旦出现会污染整个模型"——不能继续训，必须 rollback 到上一个 checkpoint + 修复后重启。

</details>

---

## 7. 延伸阅读

- [PyTorch Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html) — 官方权威，pin_memory / num_workers / channels_last / cudnn benchmark 等技巧的标准答案
- [PyTorch AMP Examples](https://pytorch.org/docs/stable/notes/amp_examples.html) — autocast + GradScaler 的官方使用模板，包含 grad accumulation、grad penalty 等边角 case
- [HuggingFace Transformers — Trainer source](https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py) — 工业级 trainer 实现的最佳参考，本节代码示例的"完整版"。重点看 `_inner_training_loop` 与 `training_step`
- [Eleuther AI — Transformer Math 101](https://blog.eleuther.ai/transformer-math/) — 显存 / FLOPs / 通信开销的公式速查，Module 7 之前必读
- 推荐继续读本教程的 **Module 7.4《混合精度：fp16 / bf16 / fp8 + Loss Scaling》**——把本节的 dtype 心智模型展开到分布式训练 infra 层
