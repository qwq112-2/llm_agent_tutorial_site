---
title: "6.4 训练曲线诊断与超参（lr / batch / warmup）"
description: "预训练跑起来之后，工程师 80% 的时间不是写代码，而是盯着 wandb 上的几条曲线——本节把那几条曲线（loss / grad_norm / lr / throughput）的\"健康姿态\"、常见异常（spike / divergence / plateau）的诊断流程，以及 LLaMA-2 / DeepSeek 量级的现代超参经验值（lr 3e-4、β2 0.95、wd 0.1、batch 4"
---

> ⏱ 预计阅读 35 分钟 ｜ 难度 ★★ ｜ 前置：1.2（优化器与 LR schedule）、6.1（CLM loss）

## 一句话本节讲什么

预训练跑起来之后，工程师 80% 的时间不是写代码，而是**盯着 wandb 上的几条曲线**——本节把那几条曲线（loss / grad_norm / lr / throughput）的"健康姿态"、常见异常（spike / divergence / plateau）的诊断流程，以及 LLaMA-2 / DeepSeek 量级的现代超参经验值（lr 3e-4、β2 0.95、wd 0.1、batch 4M token、warmup 2k step）一次说清楚。

---

## 1. Mental model（直觉）

### 1.1 训 LLM = 开飞机，不是开车

训一个 nano 模型时，你 `python train.py` 就能跑完，loss 不收敛了重训一遍即可。但训一个 7B / 70B model 是另一种工程：

- 一次实验**几天到几个月**，单次 GPU 时长成本五位数美金起步
- 训到 **step 30k 突然 loss 飞**，rollback 损失的不是几分钟而是几天
- 100 张 GPU 中**任意一张挂掉**，都可能让整个 batch 数据丢失甚至权重损坏
- 训练曲线**不能回退重跑**——上千万美元的算力没法说"再来一次"

所以预训练工程的核心动作不是"调参跑出最好结果"，而是**"实时监控 + 异常预警 + 快速止损"**。1.2 节讲了优化器和 LR schedule 的"是什么"和"怎么调"，本节专门讲"训练已经在跑、你坐在 wandb 前面应该看什么"。

### 1.2 训练曲线的"四个仪表盘"心智模型

把 wandb dashboard 想象成飞机驾驶舱，至少 4 个仪表必须始终在视线内：

```
仪表 1: train/eval loss     ← 主要业务指标，loss 是不是在按 scaling law 预期下降
仪表 2: gradient norm       ← 健康预警，spike 通常领先 loss 异常 5-50 step
仪表 3: learning rate       ← 状态指示，确认 schedule 在按预期跑（warmup 完没？）
仪表 4: throughput (tok/s)  ← infra 健康，掉 5% 就要查（gradient sync 卡 / OOM 重试）
```

这四个仪表是"必看"，下文 §2 会展开。还有几个"建议看"的（activation norm、per-layer grad、memory util），不在主屏但 alert 要配。

### 1.3 训练异常的"五种长相"

经验上 LLM 训练异常分五大类，每一类都有典型的 ASCII 长相和对应处理：

```
loss
 │
 │   ╱╲                      Spike：突然涨一下又回落
 │  ╱  ╲___________          → 80% 是 bad batch / lr 偏大，自愈或 skip
 │
 │  ╱╲                       Divergence：飞掉 NaN
 │ ╱  ╲___ /‾‾NaN            → lr 过大 / no warmup / fp16 失稳
 │
 │ ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾           Plateau：长时间不降
 │                            → lr 过小 / batch 过大 / data 重复
 │
 │ ‾‾‾─────                  Slow：在降但很慢
 │                            → optim state 没 resume / data 质量差
 │
 │  ╲╱‾╲╱‾╲╱                 Epoch jump：epoch 边界跳变
 │                            → data shuffling 不一致 / epoch order matters
```

这五种长相是预训练工程师的"常用语"。看到 wandb 异常截图，一眼能说出"这是 spike 还是 divergence"，是这个岗位的入门要求。

### 1.4 一句话总结全节

**LLM 训练监控的核心三件事：(1) 看 loss + grad_norm 的相对变化（spike 前 grad 总会先涨）、(2) 知道哪些超参是"动了八成会出事"的（lr / wd / β2 / warmup），(3) 准备好 skip-on-spike 与 ckpt rollback 的自动化。** 剩下的都是细节。

---

## 2. 公式与原理

预训练阶段几乎不引入新公式（loss 的 CLM 形式 6.1 已给、optimizer 公式 1.2 已推导）。本节的"公式与原理"集中在**怎么把这些量监控起来**，以及它们的"健康范围"是什么。

### 2.1 训练 monitor 的标准指标

每个指标的"是什么 / 看什么 / 健康范围"如下：

| 指标 | 计算 | 看什么 | 健康范围（7B 量级） |
|---|---|---|---|
| **train loss** | per-step CE loss | 是否平滑下降 | 1k step: ~5 → 100k step: ~2.5 |
| **eval loss / ppl** | 在 holdout 1B token 上算 | 与 train 的 gap 是否合理 | eval/train ratio 1.0–1.1 |
| **gradient norm** | $\|\nabla\theta\|_2$（global） | spike 预警 | 0.1–1.0；> 5 警觉，> 20 危险 |
| **learning rate** | scheduler 当前值 | 确认 warmup / decay 阶段对 | 按 schedule 预期 |
| **throughput** | tokens/s/GPU | infra 健康 | LLaMA-2 7B：~3500 tok/s/A100 |
| **GPU memory util** | nvidia-smi | OOM 预警 | 70–90%（留 buffer 防 spike） |
| **activation norm** | 某几层 hidden 的 $\|h\|$ | 早期发现 norm 飘逸 | 数量级稳定（< 10× 漂移） |
| **per-layer grad** | 每层参数的 grad norm | 定位"哪一层先炸" | 各层量级接近、不应有几个数量级差异 |

其中 **train loss + grad_norm + lr + throughput** 是必上 dashboard 的"四仪表"；后面几个是 alert / 调试时再细看。

### 2.2 Gradient norm 的关键性

**`grad_norm`（梯度全局范数）是 LLM 训练最有诊断价值的单一指标**——比 loss 灵敏 10 倍、比 activation 容易计算。它的定义：

$$\|g\|_2 = \sqrt{\sum_{i} g_i^2}$$

其中 $g_i$ 是参数 $i$ 在本 step 的梯度。PyTorch 的 `torch.nn.utils.clip_grad_norm_` 在裁剪前就会算出这个值，把它返回出来 log 即可（不需要额外计算开销）。

**为什么 grad_norm 比 loss 灵敏**：

1. **Loss 是 forward 量，grad 是 backward 量**——梯度反映"参数空间发生了什么"，loss 反映"输出空间发生了什么"。bad batch 通常先在梯度上表现出来。
2. **Spike 前 grad 几乎总会先涨**——经验观察：loss spike 前 5–50 step，grad_norm 已经开始异常上升（从 0.5 涨到 5）。这个 5–50 step 的窗口是"自动化 skip-on-spike"的核心依据。
3. **Grad norm 突然变 0** 也是异常（可能 fp16 下溢、某层 frozen 了），同样要 alert。

健康的 grad_norm 曲线在 LLM 训练中应该是**带噪声但稳定的水平线**，典型值 0.3–1.0，偶有 spike 到 2-3 但马上回落。如果你看到 grad_norm 曲线在持续上升（趋势上升而不是 spike），几乎可以确定训练快要崩。

### 2.3 健康 loss 曲线的"三阶段"形态

成熟的 7B / 70B 量级 LLM 预训练，loss 曲线应该呈现明显的三阶段：

```
loss
 8 │\
 6 │ \\___              ← 早期 (0 - 1k step): warmup + 初期快速下降
 4 │     \\____         ← 中期 (10k - 100k step): 平滑指数下降
 3 │          \____     ← 后期 (100k+): 趋于平稳，scaling law 收益递减
 2 │              ‾‾‾‾
 0 └─────────────────── step
```

**早期 (0–1k step)**：loss 从 ~10（接近 random，$-\log(1/\text{vocab})$ ≈ 10.5）快速降到 5 左右。看的是 **warmup 是否平滑**——这个阶段最容易出第一类异常（warmup 不够 / lr 飞 / dtype 不对）。健康表现：曲线连续无尖刺，与 warmup 步数线性关系明显。

**中期 (10k–100k step)**：loss 平滑指数下降。看 **eval/train ratio**（应在 1.0–1.1，> 1.2 警觉过拟合，< 0.95 警觉 data leak）。这个阶段大部分时间没有突发情况，主要是看 throughput 别掉、grad_norm 别趋势上升。

**后期 (100k+ step)**：loss 接近 1.5–2.5（取决于 model size），下降速度肉眼可见变慢——这是 scaling law 的收益递减，**不是异常**。这个阶段更要关注 throughput / 资源利用率，因为 loss 已经不会大幅下降了。

### 2.4 五种异常的诊断与处理

#### Spike（尖刺）

```
loss
 4 │
 3 │     ╱╲
 2 │    ╱  ╲___
 1 │___/        ___
   └─────────────── step
```

**长相**：loss 突然涨一下（从 2 涨到 4），5–20 step 内自己回落。

**原因排序**（按出现频率）：

1. **Bad batch**：数据里夹了一条异常样本（极长重复 token、奇异字符、高 entropy 噪声、tokenizer 切错的 binary 数据）—— 60% 的 spike 是这个
2. **lr 偏大**：peak lr 调得激进（如 7B 用 1e-3），偶尔遇到大梯度方向就推过头
3. **fp16 上溢**：bf16 下少见、fp16 下需要看 GradScaler 是否在动作

**处理顺序**：

1. **先观察 5–10 step**，很多 spike 自愈，不要立即 rollback（rollback 也是几小时损失）
2. 看 grad_norm 曲线——spike 前 grad_norm 是否先涨？涨多少？
3. 如果 5 step 没回落、仍在上升 → 触发 **skip-on-spike** 机制（见 §2.5）
4. 如果 spike 反复（一天 spike 5 次以上）→ 降 lr 或检查数据清洗

#### Divergence（爆炸）

```
loss
 │       ╱
 │      ╱
 │     ╱
 │____╱____ NaN
 └─────────── step
```

**长相**：loss 直线飞起再变 NaN，无法自愈。

**原因排序**：

1. **lr 过大 + 没 warmup**：第一名原因。直接用 peak lr 训 7B，前 200 step 必飞
2. **dtype 选错**：fp16 + softmax/log_softmax 数值溢出
3. **Data corruption**：数据加载 bug，如 label 错位、padding 全部 -100 导致 0/0
4. **Resume 没带 optimizer state**：等同热启动，详见 §4 踩坑

**处理流程**：

1. **rollback 到最近健康 ckpt**（典型保留过去 3–5 个，每 1k step 一存）
2. 降 lr 50% + 增加 warmup + 改 bf16（如果在 fp16）
3. **必须找到根本原因再继续**——不能只是"重启就好"，下次还会炸

#### Plateau（平台）

```
loss
 4 │
 3 │\
 2 │ \____________________
 1 │
   └──────────────────────── step
```

**长相**：loss 长期（10k+ step）不下降。

**原因排序**：

1. **lr 过小**：optimizer 在"原地踏步"
2. **batch size 过大 + lr 没 scaling**：critical batch size 之外，effective lr 不够
3. **数据重复 / 已学完**：1B token 数据集训第三个 epoch，model 已经背了
4. **Optim state 损坏**：resume 时 m / v EMA 没正确恢复

**处理**：先验证不是 data 的问题（换 holdout 看 loss 是否一致），再考虑动 lr。

#### Slow convergence（慢）

类似 plateau 但 loss 仍在缓慢下降。常见于：(1) resume 没带 scheduler state（lr 又从 0 重新 warmup）、(2) data quality 差（FineWeb-Edu 比 CommonCrawl raw 收敛快 2-3 倍）、(3) batch 不够大（大 batch 训练更稳定，详见 §2.6）。

#### Epoch jump（epoch 边界跳变）

```
loss
 │\
 │ ╲__       ╲__       ╲__
 │    ╲     ╱   ╲     ╱   ╲
 │     ╲___╱     ╲___╱     ╲___
 └────────────────────────────── step
       epoch1     epoch2    epoch3
```

**长相**：每个 epoch 边界 loss 明显上跳后又下降。

**原因**：data shuffling 不一致（每 epoch 用不同 seed shuffle，但 hard examples 的位置变化导致 loss 局部偏移）；或者 data 顺序对 loss 有影响（curriculum 设计问题）。

**LLM 预训练的特殊性**：现代 LLM 几乎都是 **<1 epoch 训练**（1T-15T token、10-100B model），不存在传统意义的 epoch jump。这种异常更多在 SFT 阶段出现。

### 2.5 Skip-on-spike：工业级 spike 处理流程

DeepMind / Anthropic / Meta 的预训练 pipeline 都有自动化 skip-on-spike 机制，标准流程：

```python
# 伪代码
running_loss = ExponentialMovingAverage(alpha=0.99)
for step, batch in enumerate(loader):
    loss, grad_norm = train_step(batch)
    
    # 检测：当前 loss 比 EMA 高 N 倍，或 grad_norm 比基线高 K 倍
    if loss > 5 * running_loss.value or grad_norm > 20 * baseline_grad:
        # 跳过这个 batch：撤销 optimizer.step、不更新参数
        log_warning(f"step {step}: spike detected, loss={loss}, gn={grad_norm}")
        skipped_batches.append((step, batch_id))
        continue  # 不算这一步
    
    optimizer.step()
    running_loss.update(loss)
```

**实战参数**：通常 "loss > EMA × 5" 或 "grad_norm > 历史 99 分位数 × 3" 触发 skip。skip 数量本身也是要 monitor 的指标——一天 skip > 50 次说明 data 有系统性问题，要查清洗 pipeline，不能继续训。

### 2.6 Batch size 的特殊性与 critical batch size

LLM 用**极大 batch**（1M–4M token / step）训练——LLaMA-2 用 4M token（512 sequence × 8192 length）；DeepSeek-V3 用 ~15M token。这与 CV 的几百 batch size 完全不同。

**为什么大 batch**：

1. **梯度估计更准**：batch 内 token 多，梯度方差小，optimizer 更稳定
2. **infra 利用率高**：大 batch 摊薄 communication overhead，TP/PP/DP 协同效率好
3. **配合 AdamW**：自适应 optimizer 在大 batch 下不容易"一步过冲"

但**大 batch 不是越大越好**——McCandlish et al. 2018 提出 **critical batch size** 概念：超过这个临界值，batch 再增大边际收益急剧下降（每 token 的 loss 改善 ≈ 0）。critical batch size 与训练阶段相关：训练初期临界值小（~256 sample），后期变大（~4096 sample），所以现代 LLM 才能用 4M+ batch。

**Linear scaling rule 的边界**：经典 "batch ↑ × 2 → lr ↑ × 2" 规则（Goyal 2017）只对**小 batch 区间**成立。大 batch 后 lr 与 batch 的关系是 sub-linear 甚至 saturation：LLaMA-2 7B 的 batch 4M、lr 3e-4 是经验调出来的，不是按 linear scaling 推的。

### 2.7 现代 LLM 超参经验值表

| 超参 | 经验值 | 备注 |
|---|---|---|
| **lr peak** | 1e-4 ~ 6e-4 | 与 model size 反相关；7B: 3e-4，70B: 1.5e-4 |
| **warmup steps** | 2000 step 或 0.5–1% total | LLaMA: 2000 step linear |
| **LR schedule** | cosine to 10% peak | 主流；WSD 是新替代 |
| **batch size (token)** | 1M–4M token / step | LLaMA-2 用 4M；DeepSeek-V3 ~15M |
| **β1, β2** | 0.9, **0.95** | β2 显著小于默认 0.999，原因见 1.2 §4 |
| **weight decay** | 0.1 | 远大于 CV 的 0.01 |
| **grad clip** | 1.0（global norm） | reasoning RL 训练可能要 0.5-0.7 |
| **dropout** | 0（pretrain） / 0.1（fine-tune） | LLM pretrain 通常 0 |
| **dtype** | bf16（compute）/ fp32（params, optim） | mixed precision 标配 |
| **eps (Adam)** | 1e-8 | LLaMA-2 用 1e-5 增加稳定性 |

**这张表是面试高频考点**——背下来，能 cover 大部分 LLM 预训练超参问题。

### 2.8 训练 monitor 的工具栈

| 工具 | 类型 | 适用 |
|---|---|---|
| **Weights & Biases (wandb)** | 商业 SaaS | 业界主流，公司大模型团队几乎全用 |
| **TensorBoard** | 开源 | 经典选择，本地 / 简单部署 |
| **MLflow** | 开源 / 企业级 | 实验管理 + ckpt 版本，企业内部多 |
| **Aim** | 开源 | Meta 推出，自托管友好 |
| **Comet ML** | 商业 SaaS | wandb 的竞品 |

**关键 metric 要 log 的**：loss / grad_norm / lr / throughput (tok/s) / token_consumed (累积) / GPU util / GPU mem / batch loss histogram / per-layer grad histogram。

**最重要的工程纪律**：metrics 必须**同时**写到本地 disk（json log 或 csv）和远程 dashboard。OOM / 网络中断 / dashboard 服务挂了的时候，本地 log 是唯一能事后追溯的依据。Anthropic / OpenAI 内部据说都自研了带本地落盘的 logger，开源项目可以用 `wandb.init(mode='offline')` 实现类似效果。

---

## 3. 最小代码示例

### 3.1 完整训练 monitor 代码骨架

```python
import torch, wandb, time
from torch.nn.utils import clip_grad_norm_

def train_loop(model, optimizer, scheduler, loader, total_steps,
               ckpt_every=1000, log_every=10):
    wandb.init(project="pretrain-7b", config={"lr": 3e-4, "wd": 0.1})
    ema_loss = None                                  # spike 检测的 EMA baseline
    skipped = 0
    t_last = time.time()
    
    for step, batch in enumerate(loader):
        # forward + backward
        loss = model(**batch).loss
        loss.backward()
        # 关键：clip_grad_norm_ 返回 clip 前的 grad norm，直接 log
        grad_norm = clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # skip-on-spike：5x EMA 触发
        if ema_loss is not None and loss.item() > 5 * ema_loss:
            skipped += 1
            optimizer.zero_grad()
            wandb.log({"spike/skipped": skipped, "spike/loss": loss.item()}, step=step)
            continue
        
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        # 更新 EMA + log
        ema_loss = loss.item() if ema_loss is None else 0.99 * ema_loss + 0.01 * loss.item()
        if step % log_every == 0:
            tps = (log_every * batch["input_ids"].numel()) / (time.time() - t_last)
            wandb.log({
                "train/loss": loss.item(),
                "train/grad_norm": grad_norm.item(),
                "train/lr": scheduler.get_last_lr()[0],
                "train/tokens_per_sec": tps,
                "train/skipped_total": skipped,
            }, step=step)
            t_last = time.time()
        
        # ckpt：必须包含 optimizer + scheduler state
        if step > 0 and step % ckpt_every == 0:
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": step,
                "ema_loss": ema_loss,
            }, f"ckpt_step{step}.pt")
        
        if step >= total_steps: break
```

四个关键点：(1) `clip_grad_norm_` 顺手返回 grad norm，零开销 log；(2) skip-on-spike 用 EMA 比较；(3) 每 N step log 一次 throughput；(4) ckpt 必须存 optim + scheduler state。这 50 行是 LLM 预训练循环的"骨架"，工业级训练栈（如 Megatron-LM）的核心循环也就是这个结构。

### 3.2 LR schedule 可视化

```python
import math, matplotlib.pyplot as plt

def lr_at(step, peak=3e-4, warmup=2000, total=100_000, min_ratio=0.1):
    if step < warmup:
        return peak * step / warmup
    progress = (step - warmup) / (total - warmup)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return peak * (min_ratio + (1 - min_ratio) * cosine)

steps = range(0, 100_000, 100)
lrs = [lr_at(s) for s in steps]
plt.plot(steps, lrs); plt.xlabel("step"); plt.ylabel("lr")
plt.axvline(2000, ls='--', alpha=0.3, label='warmup end')
plt.legend(); plt.title("warmup 2k + cosine to 0.1× peak")
plt.savefig("lr_schedule.png")
```

跑一下这个，会看到一个清晰的"线性涨到 3e-4 → 余弦下降到 3e-5"的曲线。**训练前先跑这个 sanity check** 是个好习惯——能及时发现 schedule 写错了（如 total step 数算错）。

### 3.3 训练健康 check 函数

```python
def health_check(loss, grad_norm, lr, tps, expected_lr):
    """返回 None 或 alert 字符串。集成到训练循环，每 1000 step 调一次。"""
    issues = []
    if loss != loss:  # NaN
        issues.append(f"loss is NaN")
    if grad_norm > 20:
        issues.append(f"grad_norm too high: {grad_norm:.2f}")
    if grad_norm < 1e-5:
        issues.append(f"grad_norm vanishingly small: {grad_norm:.2e}")
    if abs(lr - expected_lr) / max(expected_lr, 1e-9) > 0.05:
        issues.append(f"lr drift: actual={lr}, expected={expected_lr}")
    if tps < 1000:  # 7B on A100 应该 > 3000
        issues.append(f"throughput suspiciously low: {tps:.0f} tok/s")
    return "; ".join(issues) if issues else None
```

把这个挂到训练循环里，触发 alert 时 push 到 Slack / 邮件。集群通宵跑训练时**必须**有 alert，不然第二天早上发现 loss NaN 已经损失 8 小时算力。

---

## 4. 工程踩坑与经验

- ❗ **Resume 训练必须 load `optimizer.state_dict()` + `scheduler.state_dict()`**，否则等于"用当前 weights 开始一次热启动"——optimizer 的 m/v 一阶二阶矩重置、scheduler 的 step counter 重置，前 1k step 的 loss 会明显飙高。HuggingFace `Trainer` 默认会保存 `optimizer.pt` 和 `scheduler.pt`，自定义训练循环这是**新手 #1 bug**。判断有没有踩坑很简单：resume 后看 loss 第一个 step 是否与中断前最后一个 step 接近，若飙了 0.5+ 就是没正确 resume。

- ❗ **Loss spike 不一定立即 rollback——很多 spike 5–10 step 内会自愈**。新手见到 spike 第一反应"赶快 rollback 重训"，但 rollback 的成本（损失几小时算力 + 重启 infra）通常比"等 5 step 看一下"更大。工业实战的标准流程：spike 来了先看 grad_norm 趋势，5 step 内回落继续训；不回落或反复 spike 才 rollback。Anthropic / DeepMind 都有自动化 skip-on-spike pipeline，把 skip 当成正常事件处理而不是灾难。

- ❗ **bf16 训练几乎不需要 GradScaler，但要小心 reduce 操作的 fp32 累加**。bf16 mantissa 7 位，单个数值的精度低，但**动态范围与 fp32 一致**（指数 8 位），所以不会上溢——这是 bf16 优于 fp16 的核心。但 all_reduce / reduce_scatter 在多 GPU 下做求和时，bf16 累加误差会被放大（多卡求和 N 次，误差累积到 $\sqrt{N}$ 量级）。解决方法：通信时 cast 到 fp32 再累加再 cast 回 bf16，PyTorch FSDP / DeepSpeed 都默认这么做（`reduce_dtype=fp32`）。**自己写分布式训练时务必显式指定 reduce dtype**。

- ❗ **`grad_clip = 1.0` 是经验值，对 R1 类 reasoning RL 训练可能要调到 0.5–0.7**。pretrain 阶段梯度噪声主要来自数据，1.0 够用；但 RL 训练（GRPO / RLOO）的 reward signal 噪声大、advantage 方差高，1.0 可能不够紧——DeepSeek-R1 train log 里能看到 grad_clip 经常被触发。具体值需要 ablation，但**"看到 RL 训练就先把 grad_clip 砍半"** 是有用的启发式。

- ❗ **Warmup step 太少（< 500）容易前期 loss 飞，太多（> 5000）浪费**。常见错误：用一个 100 step 的 warmup 训 100k step——前几个 step lr 变化太陡，AdamW 的 m/v 估计跟不上，loss 必飞；另一种错误是用 10k step warmup 训 100k step，相当于 10% 的训练时间都在 sub-optimal lr 跑——浪费算力。LLaMA / Qwen / DeepSeek 的经验：**2000 step 是 7B-70B 的安全默认**，total step 越大 warmup 占比越小（如 1M step 训练用 2k warmup 即可，占比 0.2%）。

- ❗ **Linear scaling lr "rule of thumb" 只对小 batch 适用，大 batch 后非线性**。Goyal et al. 2017 的 "batch × 2 → lr × 2" 在 batch < 8K（CV 量级）很准；但 LLM batch 4M 已经在 critical batch size 之外，lr 与 batch 关系趋于 saturation。**实际工程做法**：固定 batch（4M token）+ 调 lr，不要按 linear scaling 推算。如果 infra 限制必须改 batch 大小（如 batch 加倍），lr 只调 1.3–1.5 倍而不是 2 倍。

- ❗ **训练 monitor 必须 log 到本地 disk + 远程 dashboard，OOM 时要能事后追溯**。wandb 默认走网络 push metric，dashboard 服务挂了或网络瘸了会丢 log。Anthropic / OpenAI 据说都自研了带本地落盘的 logger，开源方案是 `wandb.init(mode='offline')` 或同时写 csv/json 到本地。OOM 复盘时如果只有云端 dashboard 没本地 log，常常找不到崩溃前最后几个 step 的状态——**这是真实公司里训练事故的常见痛点**。

- ❗ **Throughput (tokens/s/GPU) 是 LLM 训练 infra 健康最敏感的指标，掉 5% 就要查**。loss / grad_norm 反映"训练算法健康"，throughput 反映"训练 infra 健康"。常见 throughput 下降原因：(1) gradient sync 卡顿（某张 GPU 慢、网络抖动）、(2) data loader 跟不上（pre-fetch 不够、shard 分配不均）、(3) OOM 触发 retry（PyTorch 默认会 retry 一次再 throw）、(4) NCCL 超时重连。**throughput alert 的阈值要设到 5%**，10% 就晚了——一晚上 GPU 时长几十万美金，5% 损失也是几千刀。

- ❗ **FSDP / DeepSpeed ZeRO 下 grad_norm 的计算要全局聚合**。单卡只能看到自己 shard 的梯度，要算 global grad_norm 必须 all_reduce。`clip_grad_norm_` 在 FSDP / DeepSpeed 内部会处理这个，但**自己手算 grad_norm 时要记得做 all_reduce**——常见 bug 是只 log 单卡的 partial grad_norm，曲线看起来只有真实值的 1/N（N = world_size）。

- ❗ **Eval loss 必须用 fixed holdout，而且要至少 1B token 才稳定**。新手用 100M token holdout，eval loss 噪声 ±0.05；用 1B token，噪声 ±0.005，才能看出小幅 plateau / 退化。HuggingFace 的 `evaluation_strategy="steps"` 默认是用 train data 的一个子集做 eval——绝对不要这么干，必须严格隔离 train / eval。

---

## 5. 经典 paper

- **McCandlish et al., 2018 — An Empirical Model of Large-Batch Training** — critical batch size 概念的提出。读 §2 推导 critical batch size = "梯度噪声尺度"，读 §3 看 batch size 与训练效率的实证曲线。**为什么读**：理解 LLM 为什么用 4M token batch、为什么继续加大 batch 边际收益消失，这篇是源头。

- **Smith et al., 2017 — Don't Decay the Learning Rate, Increase the Batch Size** — 提出"训练后期与其降 lr 不如加 batch"的视角。读 §2-3 论证 batch size 与 lr 的对偶关系。**为什么读**：理解为什么 LLM 训练倾向"大 batch + cosine decay lr"而不是"小 batch + step decay lr"，并理解 Linear scaling rule 的边界。

- **Touvron et al., 2023 — LLaMA-2: Open Foundation and Fine-Tuned Chat Models** — 现代 LLM 标准超参的事实参照。读 §2.2 "Training Details"——AdamW(0.9, 0.95, ε=1e-5) + cosine to 10% + 2000 step warmup + grad clip 1.0 + wd 0.1 + batch 4M token。**这一节几乎所有"经验值"都来自这里或同期的 Qwen / Mistral**。读这篇等于背下了现代 LLM 预训练超参清单。

- **Kaplan et al., 2020 / Hoffmann et al., 2022 — Scaling Laws / Chinchilla** — 与 6.3 衔接，scaling law 也是超参经验的来源（如 lr 与 model size 反相关的关系）。读 Kaplan 2020 §6 "Empirical Trends with Compute" 看 lr 与 N 的拟合曲线。**为什么读**：理解为什么 7B 用 3e-4、70B 用 1.5e-4 不是拍脑袋，背后有 scaling law 支撑。

---

## 6. 自测与面试题

**Q1（实战）**：你训 7B model，loss 在 step 3000 突然 NaN，按什么顺序排查（lr / data / dtype / scheduler / scaler）？

<details>
<summary>Answer sketch</summary>

排查顺序遵循"先排 1 分钟能验证的、最常见的"：

1. **dtype + loss scale**：第一嫌疑。先确认是 bf16 还是 fp16；fp16 必须配 GradScaler，没配就直接 NaN。bf16 下检查是不是某层（softmax / log_softmax / cross_entropy）触发了边界值。改成 fp32 跑同一 step 是否复现，能立刻定位
2. **数据**：80% 的 NaN 是某条样本里有奇异 token / 全 padding / labels 全 -100（loss 0/0）。把 step 3000 那个 batch 单独 print 出来，看 input_ids / labels 有没有异常
3. **lr 是否到 peak + warmup 是否够**：step 3000 刚好可能是 warmup 结束附近。如果 warmup 只 500 step、peak lr 6e-4，刚到 peak 时 loss 飞很常见；扩大 warmup 到 2k step + 砍半 peak lr
4. **grad_norm 历史**：拉出 NaN 前 100 step 的 grad_norm 曲线，是否在持续上升？如果 spike 前 grad 已经从 1 涨到 10，说明数据 + lr 联合作用，不是单一原因
5. **β2 是否设默认 0.999**：LLM 应该用 0.95，不少新手照搬 PyTorch 默认值，长 EMA 跟不上梯度变化导致 NaN
6. **Resume 状态**：是不是从 step 2500 ckpt resume 但没带 optimizer state？等同热启动
7. **Gradient clipping**：上面都不对就先上 grad_clip=0.5 止血，但要继续找 root cause

加分：会主动 save NaN 前的 ckpt + grad histogram + 做"哪一层先 NaN"的 hook 分析

</details>

**Q2（超参）**：LLaMA-2 70B 的 lr 比 7B 小一半（1.5e-4 vs 3e-4），为什么？至少 2 个理由。

<details>
<summary>Answer sketch</summary>

至少答到 2 个：

1. **Scaling law 的实证**：Kaplan 2020 / 后续研究显示 optimal lr 与 model size N 大致呈反相关 $\eta^* \propto N^{-\alpha}$（α ≈ 0.5）。70B 比 7B 大 10 倍，理论 lr 应小 ~3 倍，实际取 0.5 倍是经验调出来的折中
2. **大模型梯度方差更大**：参数量大 → 每个参数的更新影响整体行为 → optimizer 一步过冲的风险更高 → 需要更小的 lr 来稳定训练
3. **大模型激活值数值范围更大**：70B 的 hidden state norm 比 7B 大，反向梯度量级也更大，相同 lr 下"等效步长"更大
4. **大 batch 也对应更小相对 lr**：70B 训练通常用更大 batch（infra 配套），critical batch size 之外 lr 不能按 linear scaling 涨
5. **稳定性 trade-off**：训练成本一次几百万美金，70B 不能像 7B 那样"飞了重训"，所以 lr 选保守值

加分：能引用 Kaplan 2020 的具体公式或 LLaMA-2 paper §2.2 Table 5 的对应数值

</details>

**Q3（monitor）**：你接手一个跑了 30 天的预训任务，需要快速判断"训练健康度"，应该看哪 5 个指标？

<details>
<summary>Answer sketch</summary>

按重要性排：

1. **eval loss 趋势**：是不是仍在下降？最近 1 周下降了多少？如果 plateau > 5 天 + 已经接近 scaling law 预测值，说明在收益递减区间，可以考虑停止
2. **train/eval gap (eval/train ratio)**：1.0–1.1 健康；> 1.2 警觉过拟合（数据多样性不够 / epoch 太多）；< 0.95 警觉 data leak
3. **grad_norm 趋势**：是不是稳定在 0.3–1.0 区间？有没有趋势上升（潜在不稳）？spike 频率是多少？
4. **throughput (tokens/s/GPU)**：与训练初期相比降了多少？掉超过 5% 要查 infra
5. **skip-on-spike 累积数**：如果一周 skip 几百次，说明数据有系统性问题；< 50 次/周算正常

加分：

- 看 lr 是否还在按 schedule 走（确认 scheduler 没出 bug）
- 看 GPU memory util 是否稳定（OOM 风险）
- 看 ckpt frequency 与 disk 占用
- 看最近 10 个 ckpt 的 eval loss，确认没有"看起来健康但实际在退化"
- 拉一个最新 ckpt 跑下游小 benchmark（如 HellaSwag），物理验证模型还活着

</details>

---

## 7. 延伸阅读

- [Wandb 官方 LLM training tutorial](https://wandb.ai/site/articles/intro-to-mlops-machine-learning-experiment-tracking) — 业界主流 monitor 工具的官方上手教程，含 LLM 场景的 dashboard 模板
- [LLaMA-2 paper §2.2 Training Details (arXiv)](https://arxiv.org/abs/2307.09288) — 现代 LLM 超参事实参照
- [DeepSeek-V3 Technical Report (arXiv)](https://arxiv.org/abs/2412.19437) — 万亿参数训练的 monitor 与 stability 实战，含 spike 处理细节
- [McCandlish et al. 2018 — Empirical Model of Large-Batch Training (arXiv)](https://arxiv.org/abs/1812.06162) — critical batch size 的源头 paper
- [Smith 2017 — Don't Decay the LR, Increase the Batch Size (arXiv)](https://arxiv.org/abs/1711.00489) — 大 batch 训练的经典视角
- [Megatron-LM 训练循环源码](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/training.py) — 工业级训练 monitor 实现的参考
- [HuggingFace `Trainer` 源码](https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py) — 看主流框架怎么处理 ckpt / scheduler / spike
- 推荐继续读本教程的 **6.5 节《Long-context: Position Interpolation / NTK / YaRN / LongRoPE》**——pretrain 完成后扩长上下文的训练技巧，超参经验有相通也有不同
- 推荐继续读本教程的 **7.1 节《数据并行：DDP / ZeRO 1/2/3 / FSDP》**——本节提到的"FSDP 下 grad_norm 全局聚合"在那里有 infra 细节
