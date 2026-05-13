---
title: "7.1 数据并行：DDP / ZeRO 1/2/3 / FSDP"
description: "单卡装不下大模型 → 复制一份模型并行处理不同 batch（DDP）→ 显存仍然不够 → 把 optimizer / gradient / weight 逐级 shard 到多卡（ZeRO-1 / 2 / 3）→ PyTorch 把 ZeRO-3 重新实现成原生的 FSDP——这是当前所有 7B 以上 LLM 训练（含 SFT、RLHF、Continual Pretrain）的事实底座。"
---

> ⏱ 预计阅读 50 分钟 ｜ 难度 ★★★ 🔥 ｜ 前置：1.5 PyTorch 工作流与显存 / dtype

## 一句话本节讲什么

单卡装不下大模型 → 复制一份模型并行处理不同 batch（**DDP**）→ 显存仍然不够 → 把 optimizer / gradient / weight 逐级 shard 到多卡（**ZeRO-1 / 2 / 3**）→ PyTorch 把 ZeRO-3 重新实现成原生的 **FSDP**——这是当前所有 7B 以上 LLM 训练（含 SFT、RLHF、Continual Pretrain）的事实底座。

---

## 1. Mental model（直觉）

接 1.5 节的"显存四件套"画面：7B 模型在 bf16 + AdamW 下，光 weight + grad + optim 就要 84 GB，A100 80G 单卡装不下；70B 直接 840 GB——一张卡的内存是个**硬墙**。要继续训，唯一出路是把负载摊到多卡。

分布式训练有**两条路线**：

- **数据并行（Data Parallel, DP）**：每张卡保留**一份完整模型**，把不同 batch 喂给不同卡，每步同步梯度。简单粗暴，扩展性极好，但每张卡都要装得下整个模型——这是 DDP 的天花板。
- **模型并行（Model Parallel, MP）**：把**模型本身切成片**分散到多卡。每张卡只装一部分参数，但前向反向时要在卡之间传 activation 或 logits。复杂，通信开销大，是 7.2（TP）和 7.3（PP / SP）的主题。

本节聚焦 DP 路线。但纯 DDP 有一个根本浪费：**8 卡 DDP 上，每张卡都存了同一份 weight、同一份 grad、同一份 optimizer state**，等于把 1 张卡的内存压力 × 8 复制了 8 份。如果允许卡之间通信换显存——只让卡 0 存 1/8 的 optimizer、卡 1 存另一段、用时再 all-gather 拼回来——单卡显存就能线性下降。这个"用通信换显存"的思想就是 **ZeRO**（Zero Redundancy Optimizer）。

ZeRO 分三阶段，**贪心程度递增**：

```
DDP        :  [W][G][O]   [W][G][O]   [W][G][O]   [W][G][O]   ← 每卡一套完整三件套
ZeRO-1     :  [W][G][O₁]  [W][G][O₂]  [W][G][O₃]  [W][G][O₄]  ← optim 分片
ZeRO-2     :  [W][G₁][O₁] [W][G₂][O₂] [W][G₃][O₃] [W][G₄][O₄] ← grad 也分片
ZeRO-3/FSDP:  [W₁][G₁][O₁][W₂][G₂][O₂][W₃][G₃][O₃][W₄][G₄][O₄] ← weight 也分片
              ↑用时 all-gather 拼回完整 W，算完释放
```

每多分一项，单卡显存就少一份；代价是每步通信次数和通信量都上升。**FSDP**（Fully Sharded Data Parallel）是 PyTorch 原生重新实现的 ZeRO-3，没有新机制，只是把 ZeRO-3 做进了 PyTorch 主线，工程上和 HF Trainer / vLLM / `torch.compile` 集成更顺——这是 PyTorch 2.x 后的事实标准。

记住一个口诀：**DDP 浪费显存，ZeRO 用通信换；分到 ZeRO-3 / FSDP，每张卡只存 1/N 的 weight，前向反向时按需拼回**。其余细节都是这个口诀的展开。

---

## 2. 公式与原理

### 2.1 DDP：标准数据并行

设 $N$ 张 GPU，每张卡 rank 为 $r \in \{0, 1, \dots, N-1\}$，模型参数集合为 $\theta$。DDP 的工作方式：

1. **初始化**：每张卡 load 完整 $\theta$（broadcast 自 rank 0 保证一致）
2. **数据切分**：全局 batch $B$ 切成 $N$ 份，每卡拿 $B / N$ 个 sample（`DistributedSampler` 干这个）
3. **forward / backward**：每张卡独立算自己的 loss $\mathcal{L}_r$ 和梯度 $g_r$
4. **梯度同步**：执行 **all-reduce**：$g \leftarrow \frac{1}{N} \sum_{r=0}^{N-1} g_r$，所有卡得到完全一致的全局平均梯度
5. **optimizer step**：每张卡独立用 $g$ 更新自己的 $\theta$（结果一致，因为初始 $\theta$ 一致、梯度一致、优化器超参一致）

**通信量**：每步 1 次 all-reduce，传输量 $\approx 2 \cdot |\theta|$（ring all-reduce 实现，每个参数被发送和接收各一次，总量 $2(N-1)/N \cdot |\theta| \approx 2 |\theta|$ for large $N$）。

**单卡显存**：$M_{\text{DDP}} = M_w + M_g + M_o$（与单卡训练完全一样）。设 bf16 weight + bf16 grad + fp32 AdamW，$|\theta| = P$ 个参数：

$$M_{\text{DDP}} = 2P + 2P + 8P = 12P \text{ bytes}$$

### 2.2 ZeRO-1：optimizer state 分片

观察：optimizer state（Adam 的 $m, v$ 各一份 fp32）占 $8P$，是显存大头。但**它只在 optimizer step 时被读写**，forward / backward 用不到——为什么每张卡都存一份？

ZeRO-1 把 optimizer state 沿参数维度等分给 $N$ 张卡：每张卡只存 $1/N$ 的 $(m, v)$，对应 $1/N$ 的参数。

执行流程：

1. forward / backward 与 DDP 一样（weight / grad 仍每卡一份）
2. **梯度 reduce-scatter**：替代 all-reduce。每张卡只收到自己负责那 $1/N$ 参数对应的全局平均梯度（通信量减半）
3. **optimizer step**：每张卡只更新自己负责的 $1/N$ 参数（在自己存的 $(m, v)$ 上做）
4. **weight all-gather**：把更新后的 $1/N$ 参数广播给所有卡，拼回完整 weight 用于下一步 forward

**单卡显存**：

$$M_{\text{ZeRO-1}} = 2P + 2P + \frac{8P}{N}$$

**通信量**：reduce-scatter（$\approx P$）+ all-gather（$\approx P$）= $2P$，与 DDP 的 all-reduce 同量级。结论：**ZeRO-1 显存大降、通信不变**——免费午餐。

### 2.3 ZeRO-2：再分梯度

观察：既然每张卡只更新 $1/N$ 参数，那它**只需要那 $1/N$ 参数对应的梯度**，剩下的梯度算完发出去就可以扔了。

ZeRO-2：backward 过程中，每个 layer 算完 grad 后立刻做一次 reduce-scatter——卡 $r$ 只保留自己那一段的 reduced grad，其他段的 grad 立即释放。

**单卡显存**：

$$M_{\text{ZeRO-2}} = 2P + \frac{2P}{N} + \frac{8P}{N}$$

**通信量**：与 ZeRO-1 同（reduce-scatter + all-gather = $2P$），只是 reduce-scatter 被打散到 backward 的每一层（更利于和反向计算 overlap）。

### 2.4 ZeRO-3 / FSDP：再分 weight

最后一步：**weight 也分**。每张卡只持久存 $1/N$ 的 weight，前向 / 反向用到某个 layer 时**临时 all-gather 拼出完整 weight，算完立刻释放**。

执行流程（forward）：

```
for layer in model:
    full_weight = all_gather(layer.local_shard)   # 通信，临时拿全 weight
    activation = layer.forward(activation, full_weight)
    free(full_weight)                              # 立刻释放，回到 1/N 状态
```

backward 时同样：每个 layer 用到时再 all-gather weight，算完释放；同时算完 grad 立刻 reduce-scatter 出去。

**单卡显存**：

$$M_{\text{ZeRO-3}} = \frac{2P}{N} + \frac{2P}{N} + \frac{8P}{N} = \frac{12P}{N}$$

**通信量**：每个 layer forward + backward 各做一次 all-gather + 一次 reduce-scatter，总量 $\approx 3P$（forward 1 次 all-gather + backward 1 次 all-gather + 1 次 reduce-scatter）。**通信开销约为 DDP 的 1.5 倍**（DDP 是 $2P$，ZeRO-3 是 $3P$，文献常说"约 50% 增加"；如果 forward 后不释放、为 backward 缓存，则只多 $P$，通信加 50%；如果释放则多 $2P$，加 100%——具体看实现配置）。

PyTorch **FSDP** 是 ZeRO-3 在 PyTorch 主线的重新实现：把若干 module 包成一个 **FSDP Unit**，每个 unit 内的所有参数 flatten 成一个大 buffer 整体分片。**FSDP2**（PyTorch 2.4+）改成 **per-parameter sharding**——粒度更细、不再需要 flatten、和 `torch.compile` / activation checkpointing 兼容更好，并新增 **HSDP**（Hybrid）：node 内做 ZeRO-3，node 间做 DDP，避免跨 node all-gather 的高延迟。

### 2.5 显存对比表（必背）

设 $P$ 个参数，bf16 weight + bf16 grad + fp32 AdamW（$m + v$）。$N$ 张卡。

| 方案 | 单卡显存 / param | 70B on 8×H100 80G | 通信量 / step |
|---|---|---|---|
| DDP | $2 + 2 + 8 = 12$ B | $840$ GB ❌ | $\approx 2P$ |
| ZeRO-1 | $2 + 2 + 8/N$ | $70 \times (2 + 2 + 1) = 350$ GB ❌ | $\approx 2P$ |
| ZeRO-2 | $2 + 2/N + 8/N$ | $70 \times (2 + 0.25 + 1) = 227$ GB ❌ | $\approx 2P$ |
| ZeRO-3 / FSDP | $12/N$ | $70 \times 1.5 = 105$ GB ⚠️ | $\approx 3P$ |

70B 在 8×H100 80G 上，即使 ZeRO-3 也仍超出 80G——必须**叠加 activation checkpointing**（7.5 节）+ 减小 micro-batch 才 fit；实际工业训练 70B 通常 16 卡 FSDP 起步，或 FSDP × TP 混合（7.2 节）。

### 2.6 通信代价与 overlap

裸 ZeRO-3 通信比 DDP 多 50-100%，听起来很慢。现代实现的救赎是 **compute / communication overlap**：

- **prefetch**：算 layer $i$ 的 forward 时，**异步**发起 layer $i+1$ 的 weight all-gather，下一层算的时候 weight 已经到位
- **backward overlap**：算 layer $i$ 的反向时，异步 reduce-scatter layer $i+1$ 的 grad
- 通信完全隐藏在计算后面，wall-clock 接近 DDP

但 overlap 的前提是**计算时间 > 通信时间**——如果模型小（layer 算得快）或带宽差（多 node 跨 IB），通信掩盖不住，wall-clock 直接退化。这就是 **HSDP 存在的原因**：跨 node 通信慢，索性 node 间不分 weight、只 DDP；node 内 NVLink 带宽高，做 ZeRO-3 没问题。

### 2.7 CPU / NVMe offload

DeepSpeed-Infinity (Ren 2021) 在 ZeRO-3 之上再走一步：**把 optimizer state（甚至 weight、grad）offload 到 CPU 内存或 NVMe**，需要时再传回 GPU。

- **CPU offload**：optim state 放主机 RAM。需要时通过 PCIe 传回，PCIe 带宽 ~32 GB/s 远低于 HBM 的 ~3 TB/s，训练慢 30-100%
- **NVMe offload**：极端情况下连 RAM 都不够，把 state 放 SSD。带宽再降一档，慢 2-5 倍

适用场景：**显存极度紧张、不在乎慢一倍**——典型如个人 RTX 4090 训 7B SFT、单 H100 训 70B LoRA 等。生产大集群训练几乎不用 offload。

---

## 3. 最小代码示例

### 3.1 DDP 启动模板

```python
# train_ddp.py，启动：torchrun --nproc_per_node=4 train_ddp.py
import os, torch, torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

def main():
    rank = int(os.environ['LOCAL_RANK'])             # torchrun 注入
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(backend='nccl')

    model = MyModel().cuda()
    model = DDP(model, device_ids=[rank])            # 包一层就行

    sampler = DistributedSampler(train_dataset, shuffle=True)
    loader = DataLoader(train_dataset, batch_size=32, sampler=sampler,
                       pin_memory=True, num_workers=4)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)                     # 必须！否则每 epoch shuffle 一致
        for x, y in loader:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            optim.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()                          # DDP 自动 all-reduce grad
            optim.step()
    torch.distributed.destroy_process_group()

if __name__ == '__main__':
    main()
```

### 3.2 FSDP 包装

```python
# train_fsdp.py，启动：torchrun --nproc_per_node=8 train_fsdp.py
import functools, torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

torch.distributed.init_process_group(backend='nccl')
rank = int(os.environ['LOCAL_RANK']); torch.cuda.set_device(rank)

model = LlamaForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf')

# 关键 1：auto wrap policy —— 每个 LlamaDecoderLayer 独立 wrap 成 FSDP unit
auto_wrap = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={LlamaDecoderLayer},
)

# 关键 2：bf16 mixed precision
mp = MixedPrecision(param_dtype=torch.bfloat16,
                    reduce_dtype=torch.bfloat16,
                    buffer_dtype=torch.bfloat16)

model = FSDP(model,
             sharding_strategy=ShardingStrategy.FULL_SHARD,    # = ZeRO-3；HYBRID_SHARD 即 HSDP
             auto_wrap_policy=auto_wrap,
             mixed_precision=mp,
             device_id=rank,
             use_orig_params=True)                              # FSDP2 风格，配合 torch.compile

optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
# 之后训练循环与单卡一致
```

`ShardingStrategy` 的取值：
- `FULL_SHARD` = ZeRO-3（最省显存，通信最多）
- `SHARD_GRAD_OP` = ZeRO-2
- `HYBRID_SHARD` = HSDP（node 内 ZeRO-3，node 间 DDP）
- `NO_SHARD` = 退化为 DDP

### 3.3 DeepSpeed ZeRO-3 + offload 配置

```json
// ds_config.json，启动：deepspeed train.py --deepspeed ds_config.json
{
  "train_batch_size": 256,
  "train_micro_batch_size_per_gpu": 4,
  "gradient_accumulation_steps": 8,
  "bf16": { "enabled": true },
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": { "device": "cpu", "pin_memory": true },
    "offload_param":     { "device": "cpu", "pin_memory": true },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": 5e8,
    "stage3_prefetch_bucket_size": 5e8,
    "stage3_param_persistence_threshold": 1e6
  },
  "gradient_clipping": 1.0,
  "steps_per_print": 50
}
```

关键字段：

- `stage: 3` 启用 ZeRO-3；`stage: 1 / 2` 对应 ZeRO-1 / 2
- `offload_optimizer.device: cpu` 把 optim state 卸到主机内存（必须配 `pin_memory: true` 否则巨慢）
- `overlap_comm: true` 是通信掩盖计算的开关，**生产必开**
- `stage3_prefetch_bucket_size` 控制 prefetch 多少参数，太小掩盖不住通信，太大瞬时显存 spike

### 3.4 单卡显存估算工具

```python
def estimate_mem_per_gpu(num_params: float, world_size: int,
                         strategy: str) -> dict:
    """bf16 weight + bf16 grad + fp32 AdamW，返回 GB"""
    P, N = num_params, world_size
    if strategy == 'ddp':
        w, g, o = 2*P, 2*P, 8*P
    elif strategy == 'zero1':
        w, g, o = 2*P, 2*P, 8*P/N
    elif strategy == 'zero2':
        w, g, o = 2*P, 2*P/N, 8*P/N
    elif strategy in ('zero3', 'fsdp'):
        w, g, o = 2*P/N, 2*P/N, 8*P/N
    else:
        raise ValueError(strategy)
    return {'weight_GB': w/1e9, 'grad_GB': g/1e9,
            'optim_GB': o/1e9, 'total_GB': (w+g+o)/1e9}

# >>> estimate_mem_per_gpu(70e9, 8, 'fsdp')
# {'weight_GB': 17.5, 'grad_GB': 17.5, 'optim_GB': 70.0, 'total_GB': 105.0}
# >>> estimate_mem_per_gpu(7e9, 8, 'fsdp')
# {'weight_GB': 1.75, 'grad_GB': 1.75, 'optim_GB': 7.0, 'total_GB': 10.5}
```

注意：**这只算三件套**，没算 activation / workspace / FSDP all-gather 时的瞬时 weight buffer。实际工程在结果上再 ×1.3-1.5 留余量。

### 3.5 FSDP vs DeepSpeed 实战选择

| 维度 | FSDP（PyTorch 原生） | DeepSpeed |
|---|---|---|
| 与 PyTorch 集成 | 原生，跟 `torch.compile`、HF Trainer 顺 | 自成一套 engine，需 wrap |
| feature 完整度 | 跟齐 ZeRO-3，但 offload / MoE 弱 | ZeRO + offload + MoE + Curriculum 全 |
| 多 node | HSDP 友好 | 老 codebase 沉淀多 |
| 推理对接 | 直接 `model.state_dict()` 给 vLLM | 需要 `consolidated` 转换 |
| 新项目推荐 | ✅ 优先 FSDP / FSDP2 | 老代码 / 需 offload 时用 |

---

## 4. 工程踩坑与经验

- ❗ **多 node 启动 DDP / FSDP 必须正确配 `MASTER_ADDR / MASTER_PORT / WORLD_SIZE / RANK`**。slurm 上常见踩坑：`MASTER_ADDR` 写成了 head node 的 hostname 但子节点解析不到、`MASTER_PORT` 与系统服务冲突、`SLURM_PROCID` 没映射成 `RANK`。建议固定模板：用 `srun` + `torchrun --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT` 做 rendezvous，比手写 env 稳。
- ❗ **FSDP 的 auto wrap policy 不当 = 没用 FSDP**。如果 wrap 粒度太细（每个 Linear 单独 wrap），单个 unit 参数量太小，all-gather 通信频繁但每次量很小、kernel launch overhead 吃光收益；如果太粗（整个模型 wrap 成 1 个 unit），FSDP 退化成 DDP，不分片。**最佳实践**：transformer 模型用 `transformer_auto_wrap_policy` 按 decoder layer wrap，每个 unit 是一个 `LlamaDecoderLayer` / `Qwen2DecoderLayer`。
- ❗ **ZeRO-3 / FSDP 的 forward 时 all-gather 全 weight，瞬时显存 spike**。如果一个 layer 很大（如 70B 模型的 attention block），all-gather 拼出的完整 weight 加上 activation 可能瞬间超过 80G OOM。**对策**：(a) 配合 activation checkpointing（7.5 节）减少 activation 占用；(b) 减小 micro-batch；(c) `stage3_param_persistence_threshold` 控制小参数不分片，减少 gather 开销。
- ❗ **FSDP 与 gradient accumulation 一起用要 `model.no_sync()`**。默认每次 backward 都做 reduce-scatter 同步梯度——但 grad accumulation 期望中间 N-1 步只在本地累加梯度、最后一次再同步，否则中间步的 reduce-scatter 全是浪费、更糟的是 grad accumulation 语义错乱。正确写法：
  ```python
  for i, (x, y) in enumerate(loader):
      ctx = model.no_sync() if (i+1) % accum != 0 else nullcontext()
      with ctx:
          loss = criterion(model(x), y) / accum
          loss.backward()
      if (i+1) % accum == 0:
          optim.step(); optim.zero_grad()
  ```
- ❗ **DeepSpeed ZeRO-3 + HF Trainer 时 optimizer 由 DeepSpeed 接管**。HF Trainer 会检测到 `deepspeed=` 参数并自动用 DeepSpeed 的 fused AdamW，不要再自己创建 `torch.optim.AdamW` 传进去——会被静默忽略，但 lr / weight_decay 等参数可能没传进 DeepSpeed config，调出来发现 lr 是 ds 默认值不是你设的。**所有超参写在 ds_config.json 里**，或用 `auto` 让 Trainer 注入。
- ❗ **FSDP wrap 的 modules 必须在同一个 process group 内**。如果某些 module 用了不同的 PG（例如部分模型走 TP、部分走 FSDP），跨 PG 的 collective 会死锁或报 `NCCL warn` 但不报错——表现为训练 hang 在 backward，无 traceback。Debug 时用 `TORCH_DISTRIBUTED_DEBUG=DETAIL` + `NCCL_DEBUG=INFO` 才能定位。
- ❗ **多机 NCCL 死锁排查**：FSDP / ZeRO-3 collective 极多，跨机经常 hang。第一步打 `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ALL`，看是否在 `ncclCommInitRank` 卡住（网络问题）还是某个 `ncclAllGather` 超时（rank 不一致）。第二步 `torch.distributed.barrier()` 在关键步骤前后加，定位是哪个 collective hang。第三步检查所有 rank 是否走相同的 control flow（条件分支不一致是 hang 第一杀手）。
- ❗ **RLHF / DPO 训练时多个模型同时存在**，policy + reference + reward + value（PPO）四个模型挤在一张卡，FSDP 必须仔细规划：reference / RM 不需要 grad / optim，可以 freeze + bf16；policy 才走完整 FSDP；甚至 reference 可以 offload 到 CPU 按需 load。9.3 节会详细讲 PPO 的 4 模型显存配方。
- ❗ **FSDP save / load checkpoint 有三种模式**：`FULL_STATE_DICT`（rank 0 收齐，与单卡兼容但慢且易 OOM）、`SHARDED_STATE_DICT`（每 rank 存自己那段，快但格式不通用）、`LOCAL_STATE_DICT`（已废弃）。**生产用 `SHARDED_STATE_DICT` save**（快），需要导给 vLLM / HF 推理时用 `dist_cp` 工具或脚本 consolidate 成 full。
- ❗ **`torch.compile` + FSDP 在 PyTorch 2.0/2.1 不稳**，2.3+ 才稳定，FSDP2 (`use_orig_params=True`) 才彻底兼容。老版本组合常见 graph break、shape mismatch。

---

## 5. 经典 paper

- **Rajbhandari et al., 2020 — ZeRO: Memory Optimizations Toward Training Trillion Parameter Models** — DeepSpeed 的开山 paper，本节 ZeRO-1 / 2 / 3 三阶段的理论与公式都来自这里。读 §4 "ZeRO-DP" 的三阶段表，即可掌握 §2 全部内容。一作 Samyam Rajbhandari 后来做 DeepSpeed-Infinity / MoE 也是同思路延展。
- **Zhao et al., 2023 — PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel** — FSDP 工程化论文，讲清 PyTorch 团队为什么要重新实现 ZeRO-3（与 PyTorch 生态深度集成、auto-wrap policy、unit 划分、HSDP 设计）。读完会理解为什么新项目都用 FSDP 而不是 DeepSpeed。
- **Goyal et al., 2017 — Accurate, Large Minibatch SGD** — DDP 的经典 scaling 论文，提出 linear scaling rule（batch ×K → lr ×K）和 warmup 策略。LLM 训练 lr scheduler 设计仍在沿用。读 §2 即可，看 large-batch DDP 训练为什么需要 warmup。
- **加分阅读：Ren et al., 2021 — ZeRO-Infinity** — CPU / NVMe offload 的工程极致，单 V100 训 1T 参数模型的 demo。本节 §2.7 offload 部分的来源。

---

## 6. 自测与面试题

**Q1（显存计算）**：70B 参数模型在 16 卡 H100 上，分别用 DDP / ZeRO-2 / ZeRO-3（= FSDP）训练，bf16 weight + bf16 grad + fp32 AdamW，单卡 weight + grad + optim 三项各占多少 GB？

<details>
<summary>Answer sketch</summary>

公式：bf16 = 2 B/param，fp32 AdamW state = 8 B/param（$m$ + $v$ 各 4 B）。

| 方案 | 单卡公式 | 单卡 GB（70B / 16 卡）|
|---|---|---|
| DDP | $2P + 2P + 8P = 12P$ | $70 \times 12 = 840$ GB ❌ |
| ZeRO-2 | $2P + 2P/N + 8P/N$ | $140 + 8.75 + 35 = 183.75$ GB ❌ |
| ZeRO-3 / FSDP | $12P/N$ | $70 \times 12 / 16 = 52.5$ GB ✅ |

加分点：

- 指出 70B DDP / ZeRO-2 在 16 卡 H100 80G 仍装不下（DDP 死，ZeRO-2 也死），必须 ZeRO-3 / FSDP 才 fit
- 指出 ZeRO-3 的 52.5 GB 还**没算 activation**——叠加 activation checkpointing（7.5）和小 micro-batch 才能稳定训
- 真实工业训 70B 通常 FSDP × TP 混合（8 卡 TP 内 + 跨节点 FSDP），不只用纯 FSDP

</details>

**Q2（trade-off）**：ZeRO-3 / FSDP 相对 DDP 的通信开销大约多多少？什么场景下 ZeRO-3 反而不值得用？

<details>
<summary>Answer sketch</summary>

**通信对比**：

- DDP：每 step 1 次 all-reduce，量 $\approx 2P$
- ZeRO-3 / FSDP：每 step 2 次 all-gather（forward + backward）+ 1 次 reduce-scatter，量 $\approx 3P$
- **大约多 50%-100%**（取决于是否在 forward 后缓存 weight 到 backward）

**ZeRO-3 不值得用的场景**：

1. **模型本来就装得下**：例如 1B-3B 模型在 H100 上 DDP 完全 fit，没必要承担额外通信
2. **跨 node 通信带宽差**：multi-node 没 IB / RDMA，all-gather 在 backward 时无法被计算掩盖，wall-clock 退化严重；这种情况用 **HSDP**（node 内 FSDP，node 间 DDP）折中
3. **layer 太小算得太快**：例如小模型每 layer 计算 < 1ms，通信掩盖不住，bubble 暴露
4. **inference / eval**：无 backward 不需要 grad / optim，DDP 即可

差答案：只说"通信多就不用"——要给出具体场景与数量级。

</details>

**Q3（实战）**：你在公司 1 张 H100 80G 上要做 7B 模型 SFT，full param finetune 装不下（bf16 + AdamW 要 84 GB）。给出**至少 3 个**可行解决方向，说明各自的代价。

<details>
<summary>Answer sketch</summary>

至少给出 3 个方向并说出代价：

1. **借多卡走 FSDP**：换 8×H100 / 8×A100 用 FSDP，单卡降到 ~10.5 GB，最干净的解。代价：要协调多卡资源，单机 8 卡未必有
2. **CPU offload optim state**：单卡用 DeepSpeed ZeRO-3 + `offload_optimizer: cpu`，84 GB 中 56 GB 的 optim state 卸到主机内存，单卡显存压力降到 ~30 GB。代价：训练慢 30-100%，需要主机有 ≥100 GB RAM
3. **LoRA / QLoRA 改 PEFT**：只训 LoRA adapter（rank=64 通常 < 1% 参数），optim state 只占被训参数的，整体显存 < 20 GB；QLoRA 把 base model 量化到 4-bit 再加 LoRA，base 只 ~4 GB。代价：可能略低于全参 SFT 的最终质量，但 7B SFT 上差距很小（详见 8.3 LoRA 章节）
4. **Activation checkpointing**：开 `gradient_checkpointing=True`，反向时重算 activation 而不存，砍 60-70% activation 显存（7.5 详细讲）。代价：训练慢 20-30%
5. **减小 batch / 短 seqlen + grad accumulation**：micro-batch=1、seqlen=2048（不 4K/8K），用 grad accum 模拟大 batch。代价：单步慢
6. **bf16 → fp8**：用 Transformer Engine fp8 训练，weight + grad 砍半到 1 byte/param，单 H100 fp8 三件套 56 GB（仍紧）。代价：fp8 训练稳定性还在演化、需 Hopper 卡

加分：会算"组合方案"——例如 QLoRA + activation checkpointing 单 H100 单卡 7B SFT 极舒服，是当前最经济方案。

</details>

---

## 7. 延伸阅读

- [PyTorch FSDP Tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html) — 官方教程，从 wrap policy 到 mixed precision 到 checkpoint 全覆盖
- [PyTorch FSDP2 Migration](https://pytorch.org/docs/stable/distributed.fsdp.fully_shard.html) — FSDP2（`fully_shard` API）的迁移指南，PyTorch 2.4+ 新代码直接看这个
- [DeepSpeed ZeRO Tutorial](https://www.deepspeed.ai/tutorials/zero/) — DeepSpeed 官方 ZeRO 用法 + config 字段大全
- [HuggingFace — PyTorch Fully Sharded Data Parallel](https://huggingface.co/docs/accelerate/usage_guides/fsdp) — Accelerate / Trainer 上用 FSDP 的标准接入
- [Stas Bekman — ML Engineering Book / Parallelism](https://github.com/stas00/ml-engineering/tree/master/training/model-parallelism) — 业界最完整的并行训练实战手册，工程细节远超论文
- [DeepSpeed-Infinity Tutorial](https://www.deepspeed.ai/tutorials/zero-offload/) — CPU / NVMe offload 实操
- 推荐继续读本教程的 **7.2 模型并行（TP / PP）**——FSDP 触顶后，70B+ 必须叠加张量 / 流水并行；以及 **7.5 显存优化**——activation checkpointing 与 selective recomputation 是 FSDP 的标配伙伴
