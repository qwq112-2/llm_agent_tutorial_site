---
title: "7.2 模型并行：TP（Megatron）/ PP（1F1B / Zero-Bubble）"
description: "7.1 的 ZeRO/FSDP 把\"同一份模型\"的三件套切给多卡，但 forward 时仍要 all-gather 出完整 weight —— 当单 layer 都装不下（70B+ 的 attention / FFN）时这条路也死了；本节讲真正把 weight matrix 切成块的 Tensor Parallelism 与按 layer 切到多卡的 Pipeline Parallelism ，两"
---

> ⏱ 预计阅读 55 分钟 ｜ 难度 ★★★ 🔥 ｜ 前置：7.1 数据并行：DDP / ZeRO / FSDP

## 一句话本节讲什么

7.1 的 ZeRO/FSDP 把"同一份模型"的三件套切给多卡，但 forward 时仍要 all-gather 出**完整 weight** —— 当**单 layer 都装不下**（70B+ 的 attention / FFN）时这条路也死了；本节讲**真正把 weight matrix 切成块**的 **Tensor Parallelism (TP, Megatron)** 与**按 layer 切到多卡**的 **Pipeline Parallelism (PP, GPipe / 1F1B / Zero-Bubble)**，两者与 DP/FSDP 组合成现代 70B-405B 训练的事实标准 **3D / 5D parallel**。

---

## 1. Mental model（直觉）

接 7.1 的 ZeRO-3 / FSDP：单卡显存看似线性下降到 $12P/N$，但有一个**隐藏假设** —— forward 时一定要把某一个 layer 的完整 weight all-gather 回某张卡上。换句话说，**单张卡的显存必须装得下「单个 layer 的完整 weight + 该 layer 的 activation + 临时 buffer」**。当模型大到 70B、405B，**单层 FFN 的 hidden dim = 28672**，按 SwiGLU 算 $W_1 \in \mathbb{R}^{8192 \times 28672 \times 3}$ 单个 weight 矩阵就 1.4 GB，加上 activation 和 buffer 直接破 80 GB。**ZeRO 救不了，单层都装不下**。

唯一出路是把 weight matrix **本身切成块**：一张卡上不再持有完整的 $W$，而只持有 $W$ 的一个**列分片**或**行分片**，前向时这张卡只算它那一块乘出来的部分结果，再通过通信把多张卡的部分结果拼合（all-reduce / all-gather）。这就是 **Model Parallel (MP)**。

DP 和 MP 的本质区别一句话：

```
DDP / ZeRO / FSDP : 切「数据」（每卡看不同 batch）
TP / PP           : 切「计算」（每卡只算 weight 的一块）
```

MP 内部又分两条主路：

- **Tensor Parallel (TP)** — **沿 weight matrix 内部维度切**。把 $W \in \mathbb{R}^{d \times d'}$ 沿列（或行）等分到 $k$ 张卡，每卡持有 $W$ 的 $1/k$ 列。**同一 layer 的计算在 k 卡间分摊**，每层结束 all-reduce 一次拼回完整输出。**通信极密**（每 layer 1-2 次 all-reduce），要求 GPU 间是 **NVLink** 这种高带宽互联，TP size 通常 ≤ 8（单 node 内）。
- **Pipeline Parallel (PP)** — **按 layer 切**。layer 1-10 装在 GPU 0、layer 11-20 装在 GPU 1、...，micro-batch 像流水线一样在 GPU 间穿过。**通信稀疏**（每 stage 之间只传一次 activation），通信量远小于 TP，**可以跨 node 走 IB**。代价是流水线启动 / 收尾会有 **bubble**（部分 GPU 空闲），调度策略（GPipe → 1F1B → Interleaved 1F1B → Zero-Bubble）的演进就是在不断**砍 bubble**。

一张 ASCII 图说清两者形状：

```
TP（沿 weight 切，4 卡协作算同一 layer）：
 GPU0 持 W[:,  0:1024]  ──┐
 GPU1 持 W[:,1024:2048] ──┼─→  X · W[:, ?]  →  all-reduce  →  完整输出 Y
 GPU2 持 W[:,2048:3072] ──┤        部分和                        （此 layer 完成，下一 layer 同样切）
 GPU3 持 W[:,3072:4096] ──┘

PP（按 layer 切，4 卡每个负责模型的 1/4）：
 GPU0: layer 1-10  ──activation──→ GPU1: layer 11-20 ──→ GPU2: layer 21-30 ──→ GPU3: layer 31-40
                  ↑                                                                ↑
              传 activation 一次                                            最后一卡算 loss & 启动反向
```

记住一个口诀：**TP 通信密但稀疏切粒度（同 node 内 NVLink）；PP 通信稀疏但 bubble 难调（跨 node IB 也能用）；3D parallel = DP × TP × PP，TP 装 node 内、PP 跨 node、DP 跨集群——这就是 LLaMA-3 405B 怎么训出来的**。

---

## 2. 公式与原理

### 2.1 Tensor Parallel（Megatron 范式）

Shoeybi 2019 的 Megatron-LM 给了一个**只用 1 次 all-reduce 完成整个 FFN** 的精巧切法。考虑一个标准 Transformer FFN（这里用 ReLU/GeLU 简单形式，SwiGLU 同理）：

$$Y = \text{GeLU}(X W_1) W_2$$

其中 $X \in \mathbb{R}^{b \times s \times d}$（batch × seq × hidden），$W_1 \in \mathbb{R}^{d \times 4d}$，$W_2 \in \mathbb{R}^{4d \times d}$。要把它切到 $k$ 张卡。

**关键观察**：如果 $W_1$ **按列切**、$W_2$ **按行切**，中间的 GeLU 是 element-wise 的，不会跨卡耦合。

**$W_1$ 按列切**（Column Parallel）：

$$W_1 = [W_1^{(1)}, W_1^{(2)}, \dots, W_1^{(k)}], \quad W_1^{(i)} \in \mathbb{R}^{d \times 4d/k}$$

每卡 $i$ 算：

$$Z^{(i)} = \text{GeLU}(X W_1^{(i)}) \in \mathbb{R}^{b \times s \times 4d/k}$$

注意 $X$ 在每张卡上**都是完整的**（来自上一层的输出），但 $Z^{(i)}$ 只是 $4d/k$ 维度的部分输出。GeLU 是逐元素的 → 不跨卡。

**$W_2$ 按行切**（Row Parallel）：

$$W_2 = \begin{pmatrix} W_2^{(1)} \\ W_2^{(2)} \\ \vdots \\ W_2^{(k)} \end{pmatrix}, \quad W_2^{(i)} \in \mathbb{R}^{4d/k \times d}$$

每卡 $i$ 算 $Z^{(i)} W_2^{(i)} \in \mathbb{R}^{b \times s \times d}$ —— **每张卡都得到一个 $d$ 维的部分和**。最终：

$$Y = \sum_{i=1}^{k} Z^{(i)} W_2^{(i)}$$

**这一步靠 all-reduce 完成**。整个 FFN 只需要 **1 次 all-reduce**（forward），backward 同理 1 次。**关键是中间 GeLU 没有引入额外通信**——这就是为什么"列切 + 行切"组合优于其他切法。

**Attention 切法**：天然按 head 切。$h$ 个 head 平均分到 $k$ 卡，每卡负责 $h/k$ 个 head 的 $W_Q, W_K, W_V, W_O$（同样是 $W_{QKV}$ 列切、$W_O$ 行切的组合）。Attention block 也是 **1 次 all-reduce**。

**Megatron TP 的 layer 内通信账**：每 transformer layer = 1 attention + 1 FFN = **2 次 all-reduce**（forward）+ **2 次 all-reduce**（backward）= 4 次/layer。

**通信量**：每次 all-reduce 的量 $\approx 2 \cdot b \cdot s \cdot d$ bytes（bf16，环 all-reduce 的近似）。一个 70B 模型 80 layer × 4 collective × $2bsd$ bytes = 海量。**这就是 TP 必须走 NVLink 的根本原因** —— 单 H100 NVLink 带宽 900 GB/s，跨 node IB 只有 50-100 GB/s，差一个数量级，TP 跨 node 直接性能塌方。

**Megatron 还有个微妙优化**：forward 的 row-parallel 是 `g`（all-reduce），backward 是 `f`（identity）；column-parallel 反过来。`f` 和 `g` 是一对共轭的通信原语，这样 backward 也只需要 1 次 collective。原 paper §3 有详细推导。

### 2.2 Pipeline Parallel（GPipe / 1F1B / Zero-Bubble）

把 $L$ 层模型切成 $p$ 段（stage），每段装在 1 张 GPU 上。**每个 stage 持有连续的 $L/p$ 层**。前向时第 $i$ 个 stage 算完后把 activation 发给第 $i+1$ 个 stage；反向时 grad 反着传回。

如果一个 batch 一次送进流水线，stage 0 算完后 stage 1 才能开工 —— **stage 0 等 stage 1 算完它的 backward 前一直空闲**，这就是 **bubble**。

#### GPipe（Huang 2019）

朴素思路：把一个大 batch 切成 $m$ 个 **micro-batch**，依次送进流水线；所有 forward 完成后再统一做 backward。

```
GPipe 时序图（p=4 stage, m=4 micro-batch，F = forward, B = backward）：
                       time →
GPU 0:  F1  F2  F3  F4  -   -   -   -   B4  B3  B2  B1
GPU 1:  -   F1  F2  F3  F4  -   -   B4  B3  B2  B1  -
GPU 2:  -   -   F1  F2  F3  F4  B4  B3  B2  B1  -   -
GPU 3:  -   -   -   F1  F2  F3  F4  B4  B3  B2  B1  -
                       ↑           ↑
                     bubble       bubble
```

bubble ratio 公式：

$$\text{bubble}_{\text{GPipe}} = \frac{p - 1}{m}$$

例如 $p=8, m=8$，bubble = 87.5%（壮观地浪费）。要把 bubble 压到 < 25%，需要 $m \geq 4p$。

**致命缺点**：所有 micro-batch 的 activation 必须**全部存着**等 backward 用，activation 显存峰值 = $m \cdot M_{\text{act per micro}}$，$m$ 一大就 OOM。

#### 1F1B（One-Forward-One-Backward, PipeDream）

观察：forward 和 backward 可以**交错**。每个 stage 一旦做完一个 micro-batch 的 forward，**立刻**开始下一个 micro-batch 的 forward；**当对应 backward 信号传回来时立刻插队做 backward**——这样最后一个 stage 一直在 1F1B 切换，不留空隙。

```
1F1B 时序图（p=4, m=8）：
                              time →
GPU 0:  F1 F2 F3 F4 B1 F5 B2 F6 B3 F7 B4 F8 B5 B6 B7 B8
GPU 1:  -  F1 F2 F3 F4 B1 F5 B2 F6 B3 F7 B4 F8 B5 B6 B7 B8
GPU 2:  -  -  F1 F2 F3 F4 B1 F5 B2 F6 B3 F7 B4 F8 B5 B6 B7 B8
GPU 3:  -  -  -  F1 B1 F2 B2 F3 B3 F4 B4 F5 B5 F6 B6 F7 B7 F8 B8
                       ↑ 最后 stage 直接 1F1B，无空隙
```

**bubble 公式不变**（仍是 $(p-1)/m$，因为启动/收尾仍要 $p-1$ 步），但**关键收益是 activation 显存**：每个 stage 同时只持有 $\leq p$ 个 in-flight micro-batch 的 activation（不是 $m$ 个），**显存与 m 解耦**。这让 $m$ 可以放大到 $\geq 4p$ 而不 OOM。

#### Interleaved 1F1B（Megatron-LM）

把每个 stage 进一步**切成 $v$ 个小 chunk**（chunk 之间是不连续的 layer）。例如 stage 0 持有 layer {1,2, 17,18}，stage 1 持有 {3,4, 19,20}，... 这样每个 micro-batch 在每个 stage 上 forward $v$ 次（每次一个 chunk），bubble 进一步缩小：

$$\text{bubble}_{\text{interleaved}} = \frac{1}{v} \cdot \frac{p - 1}{m}$$

代价：通信次数 ×$v$（每个 chunk 之间都要传 activation），通信量增加。Megatron 默认 $v=2$ 或 $4$。

#### Zero-Bubble PP（Qi 2024）

更激进：把每个 layer 的 backward 拆成两步 —— **B**（算输入 grad，传给上一 stage）和 **W**（算 weight grad，纯本地）。**W 没有跨 stage 依赖**，可以任意调度。精心安排顺序，让 W 填满 bubble：

$$\text{bubble}_{\text{ZB-V}} \approx 0$$

ZB1P / ZB2P / ZB-V 是三种调度变体，复杂度递增、bubble 递减。**ZB-V 实测在 p=8, m=24 上 bubble < 1%**，但实现极其复杂，目前主要在 Megatron-Core 和阿里通义内部使用。

#### bubble ratio 速算表（必背）

| 调度 | bubble 公式 | $p=8, m=8$ | $p=8, m=32$ |
|---|---|---|---|
| GPipe | $(p-1)/m$ | 87.5% | 21.9% |
| 1F1B | $(p-1)/m$ | 87.5% | 21.9% |
| Interleaved-2 | $(p-1)/(2m)$ | 43.7% | 10.9% |
| Interleaved-4 | $(p-1)/(4m)$ | 21.9% | 5.5% |
| Zero-Bubble | $\approx 0$ | $\sim 0$ | $\sim 0$ |

工程经验：**$m \geq 4p$ 是 PP 起步线**，否则 bubble > 25% 不划算。

### 2.3 TP vs PP 对比（必给表）

| 维度 | TP | PP |
|---|---|---|
| 切分方式 | weight matrix 内部切（按列/行） | 按 layer 段切 |
| 通信原语 | all-reduce（密集） | send/recv（点对点） |
| 通信量 | 大，每 layer 2 次 all-reduce | 小，每 stage 1 次 activation |
| 通信频率 | 极高（每 layer） | 低（每 stage 边界） |
| 必需互联 | NVLink (intra-node) | InfiniBand (inter-node) 也 OK |
| 典型 size | TP = 4 / 8（≤ 单 node GPU 数） | PP = 4 - 32（跨 node） |
| 显存收益 | weight / grad / activation 都 ÷k | 只 weight / grad ÷p（activation 看调度） |
| 实现复杂度 | 中（手写 collective） | 高（bubble 调度 + send/recv 死锁） |
| 与 dropout | 需手动 sync random state | 干净 |
| 与 ZeRO 兼容 | TP+ZeRO-1 / 2 OK；TP+ZeRO-3 复杂 | PP+ZeRO-1 OK；PP+ZeRO-3 极复杂 |
| 调试难度 | 中 | 高（多 stage hang 难定位） |

### 2.4 3D Parallel = DP × TP × PP

现代大规模训练的事实标准。以 **LLaMA-3 405B on 16k H100** 为例（Meta 报告）：

- **TP = 8**：装在 1 个 node 内 8 卡，靠 NVLink 高带宽跑 all-reduce
- **PP = 16**：把 405B 的 126 层切 16 段，每段 ~8 层，跨 node 走 IB 传 activation
- **DP = 128**（FSDP / context parallel 形式）：跨集群复制
- $8 \times 16 \times 128 = 16384$ GPU，每张卡只持有 $\frac{1}{8 \times 16} = 1/128$ 的 weight

mesh 拓扑安排原则：

```
通信带宽:  NVLink (intra-node) >> IB (inter-node) >> Ethernet (inter-cluster)
对应映射:  TP   < node           PP < cluster        DP < global
          通信最密  ↑           中            ↑  通信最稀疏
```

**核心思想：通信越密的并行维度放在带宽越高的互联上**。这是 3D parallel 配置的第一原则。

如果再叠加 **ZeRO-1**（只分 optimizer state，不影响 forward 通信模式），这个组合常被叫做 **4D parallel**（DP + TP + PP + ZeRO-1）。Megatron-Core / DeepSpeed-Megatron 都支持。

### 2.5 5D parallel 预告（与 7.3 衔接）

完整工业训练栈正在演进到 **5D parallel**：

- **SP (Sequence Parallelism)**：把 sequence 维度（而不是 hidden 维度）切到多卡，配合 TP 摊薄 LayerNorm / Dropout 的显存。Megatron-SP（Korthikanti 2022）首提
- **CP (Context Parallelism)**：long-context（>32k）专用，把 attention 矩阵的 sequence 维度切到多卡，配合 ring attention 通信。LLaMA-3 长上下文训练的关键
- **EP (Expert Parallelism)**：MoE 专家分到不同卡，DeepSeek-MoE / Mixtral 必备

5D = DP + TP + PP + SP + EP。再加 FSDP/ZeRO-1 就是 **6D**。本节只讲 TP/PP，SP/CP/EP 在 **7.3** 详解。

---

## 3. 最小代码示例

### 3.1 Tensor Parallel：手写 ColumnParallelLinear / RowParallelLinear

```python
# tp_ffn.py，启动：torchrun --nproc_per_node=4 tp_ffn.py
import os, torch, torch.nn as nn, torch.distributed as dist

class ColumnParallelLinear(nn.Module):
    """W 按列切：每卡持 [d, d'/k]，输出是部分列"""
    def __init__(self, d_in, d_out, tp_world):
        super().__init__()
        assert d_out % tp_world == 0, "d_out must divide tp_world"
        self.weight = nn.Parameter(torch.empty(d_in, d_out // tp_world))
        nn.init.xavier_uniform_(self.weight)
    def forward(self, x):                            # x: [b, s, d_in]，每卡相同
        return x @ self.weight                       # → [b, s, d_out/k]，每卡持部分列

class RowParallelLinear(nn.Module):
    """W 按行切：每卡持 [d/k, d_out]，输入是部分列，输出 all-reduce 拼合"""
    def __init__(self, d_in, d_out, tp_world):
        super().__init__()
        assert d_in % tp_world == 0
        self.weight = nn.Parameter(torch.empty(d_in // tp_world, d_out))
        nn.init.xavier_uniform_(self.weight)
    def forward(self, x):                            # x: [b, s, d_in/k]
        out = x @ self.weight                        # → [b, s, d_out] 部分和
        dist.all_reduce(out, op=dist.ReduceOp.SUM)   # 关键：跨 tp 卡求和
        return out                                   # 完整 [b, s, d_out]

class MegatronFFN(nn.Module):
    def __init__(self, d, tp_world):
        super().__init__()
        self.fc1 = ColumnParallelLinear(d, 4 * d, tp_world)   # W1 列切
        self.fc2 = RowParallelLinear(4 * d, d, tp_world)      # W2 行切
    def forward(self, x):
        return self.fc2(torch.nn.functional.gelu(self.fc1(x)))  # 整个 FFN 1 次 all-reduce

if __name__ == '__main__':
    rank = int(os.environ['LOCAL_RANK']); torch.cuda.set_device(rank)
    dist.init_process_group(backend='nccl')
    ffn = MegatronFFN(d=4096, tp_world=4).cuda()
    x = torch.randn(2, 128, 4096, device='cuda')
    print(rank, ffn(x).shape)                        # 每卡输出形状一致：[2, 128, 4096]
```

**关键点**：`fc1` 列切 → 中间 GeLU 在每卡的部分列上独立做 → `fc2` 行切 + all-reduce 拼回完整输出。**整个 FFN 仅 1 次跨卡通信**——这就是 Megatron 的核心精妙之处。生产代码（Megatron-LM、Transformer Engine）还会加上 `f` / `g` 通信原语让 backward 也只需 1 次 all-reduce，并 fuse all-reduce 到 GEMM kernel 里减少 overhead。

### 3.2 Pipeline Parallel：用 PyTorch 2.x 原生 `torch.distributed.pipelining`

```python
# pp_demo.py，启动：torchrun --nproc_per_node=4 pp_demo.py
import os, torch, torch.nn as nn, torch.distributed as dist
from torch.distributed.pipelining import pipeline, ScheduleGPipe, Schedule1F1B, SplitPoint

class TinyTransformer(nn.Module):
    def __init__(self, n_layer=8, d=512):
        super().__init__()
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d, 8, batch_first=True)
                                     for _ in range(n_layer)])
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

if __name__ == '__main__':
    rank = int(os.environ['LOCAL_RANK']); torch.cuda.set_device(rank)
    dist.init_process_group(backend='nccl')
    pp_world = dist.get_world_size()                                # = 4

    model = TinyTransformer(n_layer=8, d=512)
    example = torch.randn(8, 32, 512)                                # global batch=8

    # 关键 1：声明 split point，把 8 层切 4 段（每段 2 层）
    pipe = pipeline(model, mb_args=(example,),
                    split_spec={'layers.2': SplitPoint.BEGINNING,
                                'layers.4': SplitPoint.BEGINNING,
                                'layers.6': SplitPoint.BEGINNING})
    stage = pipe.build_stage(rank, device=f'cuda:{rank}')

    # 关键 2：选调度（GPipe 朴素 / 1F1B 显存友好），n_microbatches=4
    schedule = Schedule1F1B(stage, n_microbatches=4,
                            loss_fn=lambda y, t: (y - t).pow(2).mean())
    target = torch.randn(8, 32, 512, device=f'cuda:{rank}')

    # 关键 3：last stage 提供 target，schedule 内部自动 1F1B 跑完前/反向
    if rank == pp_world - 1:
        loss = schedule.step(target=target)
    else:
        schedule.step()
```

**关键点**：`pipeline()` 把模型按 `split_spec` 切 stage，`Schedule1F1B` 自动管理 micro-batch 调度、send/recv 和 1F1B 交错。手写 1F1B 调度器要 200+ 行（PipeDream 原 paper 附录），用 `torch.distributed.pipelining` 是 PyTorch 2.4+ 的官方推荐方式（替代旧的 `torch.distributed.pipeline.sync`）。Megatron-Core 有更成熟的 Interleaved 1F1B / Zero-Bubble 实现，但需要更深的代码改造。

### 3.3 Bubble ratio 速算函数

```python
def bubble_ratio(p: int, m: int, schedule: str = '1f1b', v: int = 1) -> float:
    """
    p: pipeline stage 数
    m: micro-batch 数
    schedule: 'gpipe' / '1f1b' / 'interleaved' / 'zero_bubble'
    v: interleaved 的 chunk 数（>1）
    返回 bubble 占总时间的比例
    """
    if schedule in ('gpipe', '1f1b'):  return (p - 1) / m
    if schedule == 'interleaved':      return (p - 1) / (v * m)
    if schedule == 'zero_bubble':      return 0.0
    raise ValueError(schedule)

# >>> bubble_ratio(8, 8, 'gpipe')           # 0.875  极差
# >>> bubble_ratio(8, 32, '1f1b')           # 0.219  尚可
# >>> bubble_ratio(8, 32, 'interleaved', v=4)  # 0.055  好
# >>> bubble_ratio(8, 16, 'zero_bubble')    # 0.0    理论最优
```

### 3.4 3D Parallel 的 mesh 配置示例

```python
# 用 PyTorch DeviceMesh 描述 3D 拓扑
from torch.distributed.device_mesh import init_device_mesh

# 假设 64 GPU，配置 (DP=4, PP=4, TP=4)
mesh = init_device_mesh('cuda', (4, 4, 4), mesh_dim_names=('dp', 'pp', 'tp'))

dp_group = mesh['dp'].get_group()        # 跨 cluster 同步 grad
pp_group = mesh['pp'].get_group()        # 跨 stage 传 activation
tp_group = mesh['tp'].get_group()        # 同 layer 内 all-reduce
# Megatron-Core / TorchTitan 内部都用这种 mesh 抽象组织 collective
```

---

## 4. 工程踩坑与经验

- ❗ **TP size 必须 ≤ 单 node GPU 数**。TP 每 layer 4 次 all-reduce、通信量 $\propto bsd$ 极大，必须走 NVLink（H100 单 node 8 卡 NVLink 900 GB/s）。一旦 TP 跨 node 走 IB（50-100 GB/s），通信暴涨 10 倍，wall-clock 直接崩盘。LLaMA-3 405B 用 TP=8 不是巧合，是 **NVIDIA HGX 单 node 8 卡** 的物理上限。InfiniBand SHArP / NVLink Switch 系统能扩到 16-72 卡 NVLink，但仍是稀有硬件。
- ❗ **TP 切 attention head 时，head 数必须能整除 TP size**。LLaMA-2-70B 有 64 head，TP=8 OK（每卡 8 head），TP=16 也 OK（4 head），但 TP=12 不行。GQA 模型还要注意 **KV head 数也要能整除 TP size**（LLaMA-3 8B 有 8 KV head，TP=8 时每卡 1 KV head，TP=16 直接死）。设计模型时如果想留灵活性，head 数选 64/128 这种 2 的高次幂友好。
- ❗ **PP 的 micro-batch 数 m 必须 ≥ 4p**，否则 bubble > 25% 不划算。但 m 太大显存又不够（即使 1F1B 也要存 $\leq p$ 个 in-flight activation）。**实战经验**：先估 $m = \max(4p, \lceil B / (b_\mu) \rceil)$，其中 $B$ 是全局 batch、$b_\mu$ 是单 micro-batch 大小；如果 $m < 4p$ 就被迫减小 $b_\mu$ 来增 $m$。
- ❗ **PP + ZeRO-3 / FSDP 兼容性差**。PP 的 send/recv 假设每个 stage 持有完整 weight，而 ZeRO-3 把 weight 也分片，两者**调度顺序冲突**：PP 想"现在 forward 这个 layer"，ZeRO-3 说"等我先 all-gather weight"——容易死锁。**生产推荐组合**：PP + **ZeRO-1**（只分 optim state），不要 ZeRO-3。Megatron-LM 默认就是 PP + ZeRO-1 + activation checkpointing，DeepSpeed-Megatron 同。如果一定要 PP + ZeRO-3，DeepSpeed 的 `PipeDream-2BW` 实现做了特殊的 weight stash + double buffering。
- ❗ **TP 内部的 dropout / layernorm random state 必须显式 sync seed**。TP 把同一 layer 的计算切到多卡，但 dropout mask 是基于 random number 生成的——如果每张卡用各自的 random state，会生成**不同的 dropout mask**，结果各卡的"部分输出"不可加和（all-reduce 结果错误）。Megatron 的解决：用 `model_parallel_cuda_manual_seed(seed)` 给所有 TP rank 设同一 seed；如果是 sequence parallel 下要更复杂的 seed 协调。漏掉这一步训出来的模型 loss 看起来正常但隐式偏移。
- ❗ **Megatron-LM 的 TP 实现假设 layer 输入是 contiguous tensor**（`[b, s, d]` 标准 3D）。如果你的自定义 layer 输出是 list / dict / 非 contiguous（如经过 transpose 没 `.contiguous()`），Megatron 的 `reduce-scatter` / `all-gather` 会报 stride mismatch 或静默错误。**对策**：自定义 layer 强制 `.contiguous()` 之后再返回；用 Transformer Engine 的 fused ops 通常能避开。
- ❗ **PP 的点对点通信极易死锁，奇数 rank 尤甚**。`dist.send(tensor, dst=next_rank)` + `dist.recv(tensor, src=prev_rank)` 如果两侧 rank 启动顺序错了、或者 tensor shape 不一致、或者用了不同的 stream，**hang 没有 traceback**。Debug 三件套：(a) `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=COLL` 看是哪步 collective 卡住；(b) 用 `dist.batch_isend_irecv()` 替代裸 send/recv 让 NCCL 自己排序；(c) 关键步骤前后加 `dist.barrier(group=pp_group)`。Megatron 内部的 `p2p_communication.py` 是值得读的范本。
- ❗ **5D parallel 的 mesh 配置高度依赖 cluster topology**。同一份代码在 H100 NVL72（72 卡 NVLink 域）vs 8×H100 PCIe（无 NVLink）上的最佳 (TP, PP, DP) 完全不同。**经验配置**：
  - $\leq 8$B：单卡或 DDP/FSDP 即可，无需 TP/PP
  - $13$B–$70$B：FSDP（单 node）或 TP=4/8 + FSDP
  - $70$B–$200$B：TP=8 + PP=2-4 + FSDP
  - $400$B+：TP=8 + PP=8-16 + FSDP/DP
  - MoE 模型（DeepSeek-V3 671B）：再叠加 EP=64-256（7.3 节）
- ❗ **PP checkpoint 保存与加载是工程噩梦**。每个 stage 只持有自己那段 layer 的 state_dict，save 出来是 N 份分片 + 一份 mapping；load 时需要按相同 PP 配置重启动，不能改 PP size。**实战做法**：训练用 sharded checkpoint（快），需要 inference 时用脚本 consolidate 成 full HF format（Megatron-Core 提供 `tools/checkpoint/util.py`）。改 PP size 必须先 consolidate 到 full、再用新 PP 切分重新加载。
- ❗ **TP / PP 的实现选型**：从头新项目首选 **Megatron-Core**（NVIDIA 官方，活跃维护），其次 **TorchTitan**（PyTorch 团队的 reference 实现，代码干净易读），再次 **DeepSpeed-Megatron**（DeepSpeed 为主时）。HF Transformers 的 `accelerate` / `Trainer` 对 TP 的支持仍然弱，PP 几乎没有——HF + 大模型训练通常仍然落到 Megatron-Core 后端。

---

## 5. 经典 paper

- **Shoeybi et al., 2019 — Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism** — TP 范式的奠基作。读 §3 的 column-parallel + row-parallel 组合推导，理解为什么"列切 + 行切"能让整个 FFN 只需 1 次 all-reduce；§4 讲 attention 切 head 与 embedding 切的细节，本节 §2.1 全部基于此。
- **Huang et al., 2019 — GPipe: Easy Scaling with Micro-Batch Pipeline Parallelism** — PP 起点。读 §3 看 micro-batch pipeline 的基本思路与 bubble 公式 $(p-1)/m$；理解 GPipe 的核心约束（必须存所有 micro-batch 的 activation）就能理解 1F1B 为什么是改进。本节 §2.2 起点。
- **Narayanan et al., 2021 — Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM** — 3D parallel 的系统化论文。读 §4 看 Interleaved 1F1B 调度图，§6 看真实 175B 模型在 1024 GPU 上的 (TP=8, PP=32, DP=4) 配置，是工业级配置的范本。本节 §2.4 / §2.5 数据来源。
- **Qi et al., 2024 — Zero Bubble Pipeline Parallelism** — 把 backward 拆成 B + W 两步、调度 W 填 bubble 的精巧设计。读 §3 的 ZB-V 调度图，理解"weight grad 没有跨 stage 依赖"为什么是 zero-bubble 的钥匙。是 PP 调度的当前 SOTA。
- **加分阅读：Korthikanti et al., 2022 — Reducing Activation Recomputation in Large Transformer Models** — Megatron-SP（Sequence Parallel）+ selective activation recomputation 的提出。本节 §2.5 SP 预告与 7.5 节 activation checkpointing 都基于此。

---

## 6. 自测与面试题

**Q1（架构）**：写出 Megatron 风格 FFN 的 TP 切法（$W_1$ 按列切 + $W_2$ 按行切 + all-reduce），并解释为什么这种切法只需要 1 次 all-reduce。

<details>
<summary>Answer sketch</summary>

切法：

- $W_1 \in \mathbb{R}^{d \times 4d}$ **按列切** 成 $k$ 份：$W_1 = [W_1^{(1)}, \dots, W_1^{(k)}]$，卡 $i$ 持 $W_1^{(i)} \in \mathbb{R}^{d \times 4d/k}$
- 每卡算 $Z^{(i)} = \text{GeLU}(X W_1^{(i)})$，输入 $X$ 在每卡相同，输出是 $4d/k$ 维的部分列
- $W_2 \in \mathbb{R}^{4d \times d}$ **按行切**：卡 $i$ 持 $W_2^{(i)} \in \mathbb{R}^{4d/k \times d}$
- 每卡算 $Z^{(i)} W_2^{(i)} \in \mathbb{R}^{b \times s \times d}$，每卡得到完整 $d$ 维输出的**部分和**
- 最后 `all_reduce(sum)` 把 $k$ 份部分和加起来 = 完整 $Y$

为什么只需 1 次 all-reduce：

- 中间 GeLU 是 **element-wise** 的，作用在每个 $4d/k$ 维元素上独立进行——不需要跨卡的元素信息
- 如果 $W_1$ 也按行切（这样输出是部分和），那么 GeLU 必须在 all-reduce 后才能算（因为 $\text{GeLU}(a+b) \neq \text{GeLU}(a) + \text{GeLU}(b)$）——会多 1 次 all-reduce
- "列切 + 行切"组合让两次 GEMM 中间没有跨卡耦合，只在最后输出处通信一次

加分：指出 backward 也只需 1 次 all-reduce（用 `f`/`g` 共轭通信原语，详见 Megatron paper §3）；attention 切法同理（$W_{QKV}$ 列切 + $W_O$ 行切）。

</details>

**Q2（计算）**：算 GPipe / 1F1B / Interleaved($v=4$) / Zero-Bubble 在 $p=8$, $m=16$ 时的 bubble ratio。哪个能保持 bubble < 10%？

<details>
<summary>Answer sketch</summary>

代入公式：

- GPipe: $(p-1)/m = 7/16 = 43.75\%$
- 1F1B: $(p-1)/m = 7/16 = 43.75\%$（注：1F1B 与 GPipe 的 bubble 公式相同，差异在 activation 显存而非 bubble 大小）
- Interleaved($v=4$): $(p-1)/(v m) = 7/64 = 10.94\%$
- Zero-Bubble: $\approx 0\%$

**bubble < 10% 的方案**：Zero-Bubble（理论 0），Interleaved($v=4$) 在 10.94% 边缘——要达到严格 < 10% 需 $v \geq 8$ 或加大 $m$（如 $m = 32$ 时 Interleaved-4 = 5.5%）。

加分：

- 指出 1F1B 与 GPipe bubble 公式相同，但 **1F1B 的 activation 显存与 m 解耦**（只存 $\leq p$ 个 in-flight），这才是 1F1B 真正的工程价值
- 指出 Interleaved 的代价：通信次数 $\times v$（每个 chunk 边界都要传 activation），$v$ 太大通信反而成瓶颈
- Zero-Bubble 的工程难度极高，目前 Megatron-Core 才有完整实现，多数生产仍用 Interleaved-2/4

</details>

**Q3（实战）**：训 LLaMA-3 405B 在 4096 GPU（512 个 H100 8 卡 node，node 内 NVLink，node 间 IB 400 Gbps）上，给出一个合理的 (DP, TP, PP) 配置。说明你的 mesh 拓扑映射。

<details>
<summary>Answer sketch</summary>

参考配置：**TP=8, PP=16, DP=32**（$8 \times 16 \times 32 = 4096$）。

mesh 拓扑映射（核心原则：通信越密的并行维度放在带宽越高的互联上）：

- **TP=8 装在 1 个 node 内**：8 卡 NVLink 900 GB/s，承担每 layer 4 次 all-reduce 的密集通信。**TP 不能跨 node**——这是物理硬约束
- **PP=16 跨 node**：405B 共 126 层，切 16 段每段 ~8 层，跨 node 传 activation。每个 PP stage 占 8 卡（1 整 node），16 个 PP stage = 16 个 node。PP 通信稀疏，IB 400 Gbps 够用
- **DP=32 跨集群**：32 个 PP 副本，每个副本占 16 个 node，总共 $32 \times 16 = 512$ node = 4096 GPU。DP 通信最稀疏（每 step 1 次 all-reduce gradient），跨集群 IB 即可

**micro-batch 选择**：$m \geq 4p = 64$。设单 micro-batch 长 8192 token、$b_\mu = 1$ → 全局 token = $32 \times 64 \times 8192 \approx 16.7M$ token/step，符合 405B 训练的 batch size 量级（~16M token）。

**叠加优化**：

- **ZeRO-1** 分 optim state 到 DP 维度（节省 weight $\times 4$ 的 fp32 master），不影响 forward/backward 通信模式
- **Selective activation checkpointing**（7.5 节）平衡显存与重算
- **CP / SP**（7.3 节）如果 sequence > 32k 还要叠加

加分：

- 指出 **TP 必须 ≤ 8** 是 NVLink 单 node 物理约束
- 指出 PP / DP 比例可以微调：PP 大显存压力小但 bubble 多，DP 大 batch size 大但训练效率打折
- 提到实际 LLaMA-3 405B 的 (TP=8, PP=16, DP=N) 配置见 Meta tech report
- 指出**绝不能** TP=16（跨 node）或 PP=2（bubble 过小但模型放不下）

</details>

---

## 7. 延伸阅读

- [Megatron-LM GitHub (NVIDIA)](https://github.com/NVIDIA/Megatron-LM) — TP/PP 经典开源实现，`megatron/core/tensor_parallel/` 与 `megatron/core/pipeline_parallel/` 是必读
- [Megatron-Core 文档](https://docs.nvidia.com/megatron-core/) — Megatron-LM 重构后的库形态，配置项和 mesh 抽象更现代
- [TorchTitan: PyTorch native large-model training](https://github.com/pytorch/torchtitan) — PyTorch 团队的 reference 大模型训练 codebase，3D parallel + FSDP2 + `torch.compile` 完整集成，代码量适中适合学习
- [PyTorch Pipelining Tutorial](https://pytorch.org/tutorials/intermediate/pipelining_tutorial.html) — `torch.distributed.pipelining` 官方入门
- [HuggingFace Model Parallelism guide](https://huggingface.co/docs/transformers/perf_train_gpu_many) — 各种并行方案的对比与 HF 接入方式
- [Stas Bekman — ML Engineering / model-parallelism](https://github.com/stas00/ml-engineering/tree/master/training/model-parallelism) — 业界最完整的模型并行实战手册
- [BLOOM 175B Training Chronicles](https://github.com/bigscience-workshop/bigscience/blob/master/train/tr11-176B-ml/chronicles.md) — 真实 175B 模型 (TP=4, PP=12, DP=8) 训练日志，所有踩坑都在里面
- 推荐继续读本教程的 **7.3 SP / CP / EP**（5D parallel 完整版）；**7.5 显存优化**（activation checkpointing 与 PP/TP 是绝配伙伴）；**7.6 Triton / CUDA kernel**（理解 Megatron 内部 fused all-reduce + GEMM 是怎么做到 overlap 的）
