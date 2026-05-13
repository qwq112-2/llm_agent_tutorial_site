---
title: "7.3 Sequence / Context / Expert Parallelism"
description: "7.1 切了 weight/grad/optim、7.2 切了同 layer 内的计算，但还有三条没切：activation 显存随 sequence 二次涨（attention matrix）、long-context 单卡装不下完整 attention、MoE 的 N 个 expert 也得分卡。本节讲三个补充维度：SP（Sequence Parallel，沿 seq 维切 LayerNorm"
---

> ⏱ 预计阅读 60 分钟 ｜ 难度 ★★★ ｜ 前置：7.2 模型并行（TP/PP）；5.4 MoE

## 一句话本节讲什么

7.1 (DDP/ZeRO) 切了 weight/grad/optim、7.2 (TP/PP) 切了同 layer 内的计算，但还有三条没切：**activation 显存随 sequence 二次涨**（attention matrix）、**long-context 单卡装不下完整 attention**、**MoE 的 N 个 expert 也得分卡**。本节讲三个补充维度：**SP**（Sequence Parallel，沿 seq 维切 LayerNorm/Dropout 摊薄 activation）、**CP**（Context Parallel，长 context 训练用的 Ring/Striped Attention）、**EP**（Expert Parallel，MoE 把 expert 分卡 + all-to-all 路由 token），三者与 DP/TP/PP 拼成现代大规模训练的 **5D parallel = DP × TP × PP × SP × EP**。

---

## 1. Mental model（直觉）

7.2 把模型并行的两条主路（TP 切 weight 内部、PP 按 layer 切段）讲透了，并指出 3D parallel = DP × TP × PP 是 LLaMA-3 405B 那一档训练的事实标准。但**真上规模就会发现，这三个维度还没把所有"显存大头"都摊薄**：

- **activation 显存**沿 sequence 维度 $T$ 增长——attention 矩阵 $\mathbb{R}^{T \times T}$ 是 $O(T^2)$，LayerNorm/Dropout 的输入 buffer 是 $O(BTd)$。TP 把 $d$ 切到 $k$ 卡，但 $T$ 没动；当 $T = 128k, 1M$ 时，单卡装不下完整 KV 与 attention scores
- **MoE 模型的 N 个 expert**全部塞到一张卡里参数量爆炸（DeepSeek-V3 总参 671B），TP/PP 不天然处理"expert 这个维度"
- **long-context 训练**（LLaMA-3 的 128k 阶段、Gemini 的 1M 阶段）单 sample 的 KV 都超过单卡显存

三个补丁分别对应一个新维度：

```
SP (Sequence Parallel)  : 把 sequence 维度切到 TP 组内的多卡，
                          专攻 LayerNorm / Dropout 等 token-wise op 的 activation
                          —— 与 TP 共用 process group，配合 TP 摊薄 activation
CP (Context Parallel)   : 把 sequence 切到多卡做 attention（Ring / Striped Attention）
                          —— long-context 训练专用，每卡只持 1/N 的 Q/K/V
EP (Expert Parallel)    : 把 N 个 expert 分到 M 张卡（每卡 N/M 个 expert）
                          —— MoE 专用，token 通过 all-to-all 路由到 expert 所在 GPU
```

记住一个口诀：**TP 切 hidden 维 d，SP 切 sequence 维 T 的 token-wise 部分，CP 切 sequence 维 T 的 attention 部分，EP 切 expert 维 N**。这四个加上 PP（按 layer 段切）和 DP（按 batch 切），就组成了现代 5D 训练栈的全部维度。

一张图看清五个维度切的"是什么"：

```
原始张量空间:  [Batch B, Sequence T, Hidden d] × Layer L × Expert N（MoE）
                  ↓        ↓          ↓            ↓          ↓
切给:           DP        SP/CP       TP           PP         EP
                                   (含 attention head)

通信原语:    all-reduce  all-gather   all-reduce  send/recv  all-to-all
              (grad)    + reduce-     (每 layer)   (activ.)   (token 路由)
                        scatter
```

第二个心智抓手是**通信带宽与并行维度的映射**（把 7.2 的原则扩展）：

- **TP / SP** 通信最密（每 layer 多次 collective）→ 必须 intra-node NVLink，size ≤ 8
- **EP** 的 all-to-all 极重 → 优先 intra-node，cross-node 也行但要好的 IB
- **CP** 通信中等（Ring 每步传 K/V tile）→ 同样建议 intra-node 或低延迟 IB
- **PP** 通信稀疏（每 stage 一次 activation）→ 跨 node IB 即可
- **DP** 通信最稀疏（每 step 一次 grad all-reduce）→ 跨 cluster

5D parallel 的工程难点不在"哪个原理最难"，而在**怎么把这五个维度组合到一张物理拓扑上不互相打架**。下面三节分别拆 SP / CP / EP 的原理与公式，§2.4 整合 5D mesh 配置。

---

## 2. 公式与原理

### 2.1 Sequence Parallel（Megatron-SP，Korthikanti 2022）

**问题起点**：TP 把 attention 与 FFN 内部的 weight matrix 切到 $k$ 卡，但 LayerNorm / Dropout / residual 这些**逐 token 操作不天然适合 TP**——它们需要完整的 hidden 维度才能算（LayerNorm 要在 $d$ 维上算 mean/var）。Megatron 原版的处理是**这些 op 在每张卡上独立做**，意味着每卡都要存一份完整的 $\mathbb{R}^{B \times T \times d}$ activation buffer，**TP 没省到这部分**。

**SP 的核心思想**：在 LayerNorm / Dropout 这些 token-wise op 处，**沿 sequence 维度切**：每卡只算 $T/k$ 个 token 的 LayerNorm。这样在 TP region 之间的 SP region 里，activation 显存从 $BTd$ 降到 $BTd/k$。

数据流详解（一个 transformer layer 内）：

```
                  shape per GPU
SP region    : (B, T/k, d)          ← 每卡只持 1/k 的 token
   │
   │  LayerNorm（每卡只算自己那 T/k 个 token，不需通信）
   ↓
   │  all-gather along seq dim     ← TP→SP 边界的关键通信
   ↓
TP region    : (B, T,   d)          ← 完整 sequence，进入 TP attention/FFN
   │
   │  TP attention 或 FFN（每 layer 1 次 all-reduce，沿用 7.2 §2.1）
   ↓
   │  reduce-scatter along seq dim ← TP→SP 边界，把 all-reduce 拆成 reduce-scatter
   ↓
SP region    : (B, T/k, d)
```

**最妙的优化**：原本 TP region 出口要做 1 次 all-reduce 拼回完整输出（$BTd$ 通信量），SP 把它**等价改写为 reduce-scatter**（$BTd/k$ 通信量到每卡 → 总量同 all-reduce，但单卡 buffer 只 $BTd/k$）；下次进 TP 时再 all-gather 回来（同样 $BTd/k$ per rank）。**通信总量没变，但 activation 单卡显存降到 $1/k$**。

**通信账（每 layer）**：

| | 原 TP（无 SP） | TP + SP |
|---|---|---|
| TP 入口 | identity（$X$ 已完整）| all-gather along T，$BTd/k$ per rank |
| TP 出口 | all-reduce，$BTd$ per rank | reduce-scatter along T，$BTd/k$ per rank |
| 总通信量 | $2 BTd$（all-reduce 等价 2 倍传输）| $2 BTd$（all-gather + reduce-scatter 总量同）|
| LN/Dropout activation per GPU | $BTd$ | $BTd / k$ |

**关键结论**：SP 是**免费午餐** —— 通信总量没增加（all-reduce ≡ reduce-scatter + all-gather 在带宽上等价），但 LN/Dropout 区间的 activation 显存降到 $1/k$。原 paper 实测 LLaMA-style 模型上 activation 节省 ~30%。

**与 TP 共用 process group**：SP 不需要新建独立的通信组，直接复用 TP group。这是工程上必须强调的点——从代码看 SP 就是几个 `all_gather` / `reduce_scatter` 操作的插入，不增加分布式复杂度。

**DeepSpeed Ulysses 是 SP 的另一变体**（Jacobs 2023）：把 attention head 维度跨 sequence chunk 重分布（all-to-all 把 sequence 切换成 head 切），用于 long-context 训练。本质上是"SP 跨 head 维度做"的一个变种，原理与 Megatron-SP 不同但目标类似。

### 2.2 Context Parallel（Ring / Striped Attention）

**问题起点**：训 128k+ context 时，单卡装不下完整 attention matrix（$T^2$ 暴涨）。**SP 解决不了**——SP 只切 token-wise op，attention 内部仍要算 $\mathbb{R}^{T \times T}$ 的 score。需要把 sequence 切到多卡后**让多卡协同算 attention**。

**Ring Attention 核心思想**（Liu 2023）：把 sequence 沿 T 维切到 N 卡，每卡持有 $1/N$ 的 Q、K、V。每卡用自己的 Q 与"环上传过来的"K/V tile 算 partial attention，**像 FlashAttention 的 online softmax 一样累加 partial output**，转一圈后每卡得到自己那 $T/N$ 个 token 的完整 attention 输出。

数据流：

```
N=4 卡的 Ring Attention：

GPU0 持 Q0, K0, V0   ┐
GPU1 持 Q1, K1, V1   │
GPU2 持 Q2, K2, V2   │   每卡 Qi 是固定的（不动），K, V 在环上传
GPU3 持 Q3, K3, V3   ┘

Step 0:  GPU0 算 Q0 @ K0  → partial out_0
         GPU1 算 Q1 @ K1  → partial out_1
         (同时，每卡把自己的 K, V 发给下一卡)
Step 1:  GPU0 算 Q0 @ K3  (从 GPU3 收来的)  → 累加到 out_0
         GPU1 算 Q1 @ K0  → 累加到 out_1
Step 2:  ...
Step 3:  ...

转完 N=4 步后，每卡得到自己 Qi 与全部 N 块 K/V 的 attention 结果（即完整 attention output 的 1/N）
```

**算法骨架（与 FlashAttention 同源）**：每步算到的 partial attention 用 online softmax 公式累加（维护当前的 running max $m$ 和 normalizer $\ell$，新数据来了重新 rescale 旧 output 再加）。这里复用 5.3 节 FlashAttention 的 tile-wise softmax 推导，区别只是 tile 不在单卡 SRAM 之间流动，而在多卡 NVLink/IB 之间流动。

**通信量**：每步传一个 K/V tile（每个 $\mathbb{R}^{B \times T/N \times d}$），共 $N-1$ 步。总通信量 $\approx 2 \cdot B \cdot T \cdot d$（每卡传出和接收的 K/V 之和），**与 sequence 长度线性而非二次**——这是 Ring Attention 相对单卡 attention 的关键优势。

**与 TP attention 的对比**：

| | TP attention | Ring Attention (CP) |
|---|---|---|
| 切的维度 | head 维（$h$ 切到 $k$ 卡）| sequence 维（$T$ 切到 $N$ 卡）|
| 单卡 KV 大小 | $T$ 完整 × $h/k$ head | $T/N$ × 全部 head |
| 通信原语 | all-reduce | ring send/recv |
| 通信量（每 layer）| $O(BTd)$，1 次 all-reduce | $O(BTd)$，$N-1$ 次 send/recv tile |
| 适合场景 | 通用（< 32k context）| long-context（≥ 32k）|
| 与 long-context 配合 | head 不能再切（head 数有限）| 可任意扩 N |

**Striped Attention（Brandon 2023）**：Ring Attention 的 **load imbalance** 问题——causal attention 下，序列前半的 token 看的 K/V 少（因果 mask），序列后半看得多，**直接按连续段切会让后段 GPU 算量远大于前段**。Striped Attention 把 K/V 按 stride 切（GPU 0 持 token {0, N, 2N, ...}，GPU 1 持 {1, N+1, 2N+1, ...}），每卡 token 在 sequence 里均匀分布，causal mask 下算量自动均衡。NVIDIA Transformer Engine 的 CP 实现默认就是 striped 风格。

**Tree Attention / Hierarchical CP**：DeepSeek-V3、LLaMA-3 长上下文阶段用的工程优化，把 ring 拓扑改成 tree（intra-node ring + inter-node ring 两层），利用 NVLink/IB 的带宽差异，进一步减少跨 node 的 K/V 传输量。本质是 §2.4 的 mesh-aware CP。

**RoPE 与 CP 的关键 gotcha**：CP 切 sequence 后，每张卡的 token 在原始 sequence 里的 position 必须用 **global offset** 起算（卡 $r$ 的本地 token $t$ 的 global position 是 $r \cdot T/N + t$，striped 模式则是 $r + t \cdot N$）。如果偷懒用 local position，RoPE 编码错位、attention 数值就全错。这是 CP 实现里最常见的一类 silent bug（loss 看起来正常，但 long-context eval 全 0）。

### 2.3 Expert Parallel（MoE 专用）

**问题起点**：5.4 节讲过 MoE 把 dense FFN 替换成 N 个 expert FFN + router，token 通过 top-K 路由到不同 expert。N 通常 8-256，单卡放不下全部 expert（DeepSeek-V3 共 256 expert × 每个 expert 几亿参数 = 数百 GB）。**EP 的核心**：把 N 个 expert 分到 M 张 GPU 上，每卡持 $N/M$ 个 expert。

但 MoE 的特殊性是**每个 token 路由到的 expert 与卡之间没有静态对应**——batch 内 token 1 可能去 expert 3（GPU 0），token 2 可能去 expert 17（GPU 2）。要让每个 token 跑到它的 expert 所在的 GPU 上，必须做**all-to-all 通信**：

```
EP=4 卡，每卡持 N/4 个 expert：

Step 1 (dispatch all-to-all):
  GPU 0 收到 batch 中所有路由到 expert 0~N/4-1 的 token
  GPU 1 收到 ...                          expert N/4~2N/4-1
  GPU 2 收到 ...                          expert 2N/4~3N/4-1
  GPU 3 收到 ...                          expert 3N/4~N-1

Step 2 (local expert compute):
  每卡用本地 expert 处理收到的 token

Step 3 (combine all-to-all):
  把每卡的 expert 输出按原 token 的来源卡发回
  每卡组装回原始 (B, T, d) shape
```

**all-to-all 是 MoE 训练的主要 overhead**：

- 通信原语 `all_to_all` 让每个 rank 把 $N$ 块数据发给所有其他 rank（$N$ 是 EP world size）
- 通信量 $\approx 2 \cdot B \cdot T \cdot K \cdot d$（dispatch + combine），$K$ 是 top-K
- 与 attention/FFN compute 严重 contention，常占 MoE 训练 wall-clock 的 30-50%

**DeepEP（DeepSeek 2024）**：DeepSeek 自研的 MoE all-to-all kernel，开源后填补了开源 EP infra 的空白。优化点包括：

- **NVSHMEM-based async**：用 NVIDIA NVSHMEM 替代 NCCL 的同步语义，dispatch 与 expert compute overlap
- **Intra-node NVLink + inter-node IB 分开调度**：intra-node all-to-all 走 NVLink、inter-node 走 IB，避开 NCCL 默认的"跨拓扑统一处理"开销
- **Low-latency 模式（用于推理）**：把通信延迟压到几 μs 级别

实测 DeepEP 比 NCCL 默认 all-to-all 快 2-3×，是 DeepSeek-V3 训练能跑得动的关键 infra。

**EP 与 TP 的嵌套**（DeepSeek-V3 范式）：

```
DeepSeek-V3 架构：
  EP=64    （256 个 expert 分给 64 卡，每卡 4 expert）
  内嵌 TP=4 （每个 expert FFN 内部再做 TP=4 切，进一步省单 expert 显存）

总 expert 维并行度 = EP × TP = 256
```

EP 与 TP 嵌套是 fine-grained MoE 才需要的——expert 数多 + 单 expert 容量小 + 总参极大，单纯 EP 不够。Mixtral 8x7B 这种 coarse MoE 一般 EP=8 就够用，不需要嵌套。

**EP 与 PP 的协同**：EP 通常**装在同一个 PP stage 内**——每个 stage 持自己那段 layer 的 expert，跨 stage 不共享 expert（MoE layer 在每个 stage 内部独立切 EP）。这样 PP 的 send/recv 只传 sequence activation、不传 expert 状态，不互相打架。

### 2.4 5D Parallel = DP × TP × PP × SP × EP

把所有维度整合到一张 mesh 上，配置原则（按通信密度从高到低）：

```
intra-node   ←──────────────────────→   inter-cluster
NVLink 900GB/s         IB 50-400 GB/s        Eth/cross-region

[TP / SP] [EP]                  [PP]                [DP]
 ↑        ↑                       ↑                  ↑
每 layer   token all-to-all      stage activation   step gradient
4 collective (重)                  (中)              (轻)
```

**规模配置参考**：

| 模型 | 典型 (DP, TP, PP, SP, EP) | 备注 |
|---|---|---|
| LLaMA-3 70B（dense, 1024 GPU）| (DP=16, TP=8, PP=8, SP=8) | TP/SP 共用 group，无 EP |
| LLaMA-3 405B（dense, 16384 GPU）| (DP=128, TP=8, PP=16, SP=8) | TP=8 装单 node |
| DeepSeek-V3 671B（MoE, ~2048 GPU）| (DP=4, TP=4, PP=16, EP=64) | TP×EP 嵌套，EP 跨 node |
| 长 context 训练（LLaMA-3 128k 阶段）| (DP=*, TP=8, PP=*, CP=8) | CP 是 long-context 专用，dense 模型也用 |

**配置约束（必须遵守）**：

- $\text{TP} \times \text{SP}$ ≤ 单 node GPU 数（NVLink 物理上限）
- 满足 $L \mod \text{PP} = 0$（layer 数能被 PP 整除）
- $h \mod \text{TP} = 0$（attention head 数能被 TP 整除）
- $T \mod \text{CP} = 0$ 且 $T \mod \text{SP} = 0$
- $N_\text{expert} \mod \text{EP} = 0$（expert 数能被 EP 整除）
- $\text{DP} \times \text{TP} \times \text{PP} = \text{world\_size}$（dense）
- $\text{DP} \times \text{TP} \times \text{PP} \times \text{EP} \geq \text{world\_size}$（MoE，EP 与 TP 共享卡时算量按 max 取）

**5D activation 单卡 estimate**：

$$M_\text{act per GPU} \approx \frac{B \cdot T \cdot d \cdot L_\text{stage}}{\text{DP} \cdot \text{TP} \cdot \text{SP\_factor}}$$

其中 $L_\text{stage} = L / \text{PP}$ 是单 stage 的 layer 数，SP_factor 是 SP 把 token-wise op 摊薄的比例（约等于 TP）。MoE 模型再除一个 EP 因子（每 expert 一份独立 activation）。

**DeviceMesh API**：PyTorch 2.4+ 的 `init_device_mesh` 是表达多维并行的官方 API，统一替代手动 `init_process_group` + dim-by-dim 建 group 的老写法。所有 collective 都用 mesh sub-dim 拿 group，可读性远高于之前的 rank 算术。

### 2.5 五个维度的 activation 节省总账

| 维度 | activation 切的维度 | 节省比例 | 通信原语 |
|---|---|---|---|
| DP | 不切（每卡 batch / DP）| $1/\text{DP}$（按 batch）| all-reduce grad |
| TP | hidden d 维 | $1/\text{TP}$（attention/FFN 内部）| all-reduce / reduce-scatter |
| PP | layer 段 | 每 stage 仅 $1/\text{PP}$ 的 activation | send/recv |
| SP | sequence T 维（token-wise op）| $1/\text{TP}$（与 TP 共用）| all-gather + reduce-scatter |
| CP | sequence T 维（attention）| $1/\text{CP}$（attention 内部）| ring send/recv K/V |
| EP | expert N 维 | $1/\text{EP}$（每卡只持自己 expert 的 activation）| all-to-all |

**5D parallel 的 activation 单卡综合**：

$$M_\text{act} \approx \frac{B \cdot T \cdot d \cdot L}{\text{DP} \cdot \text{TP} \cdot \text{PP} \cdot \text{SP\_factor} \cdot \text{CP}} + \frac{T^2 \cdot h}{\text{CP}^2 \cdot \text{TP}}$$

第一项是 LayerNorm/FFN 的 $O(BTd)$ buffer，第二项是 attention 的 $O(T^2)$ score matrix（CP 切了 T，所以 $T^2/CP^2$）。MoE 模型再叠 EP 因子。这就是为什么 long-context + MoE 训练必须 5D parallel 全开——任何一维不切都会某处 OOM。

---

## 3. 最小代码示例

### 3.1 SP 示意：LayerNorm 沿 sequence 切 + all-gather/reduce-scatter

```python
# sp_layernorm_demo.py，启动：torchrun --nproc_per_node=4 sp_layernorm_demo.py
import os, torch, torch.nn as nn, torch.distributed as dist

class SPLayerNorm(nn.Module):
    """每卡持 (B, T/k, d)，本地算 LayerNorm（不需通信，因为 d 维完整）"""
    def __init__(self, d):
        super().__init__()
        self.ln = nn.LayerNorm(d)
    def forward(self, x):                                # x: (B, T/k, d)
        return self.ln(x)                                # LN 沿 d 维独立，无跨卡

def sp_to_tp(x_sp, tp_group):
    """SP region (B, T/k, d) → TP region (B, T, d)：sequence 维 all-gather"""
    k = dist.get_world_size(tp_group)
    out = [torch.empty_like(x_sp) for _ in range(k)]
    dist.all_gather(out, x_sp.contiguous(), group=tp_group)
    return torch.cat(out, dim=1)                         # (B, T, d)

def tp_to_sp(x_tp, tp_group):
    """TP region (B, T, d) → SP region (B, T/k, d)：reduce-scatter（替代 all-reduce）"""
    k = dist.get_world_size(tp_group)
    chunks = list(x_tp.chunk(k, dim=1))                  # k 个 (B, T/k, d)
    out = torch.empty_like(chunks[0])
    dist.reduce_scatter(out, [c.contiguous() for c in chunks], group=tp_group)
    return out

if __name__ == '__main__':
    rank = int(os.environ['LOCAL_RANK']); torch.cuda.set_device(rank)
    dist.init_process_group(backend='nccl')
    tp = dist.group.WORLD                                # demo 简化：world = TP
    B, T, d = 2, 128, 512
    x_sp = torch.randn(B, T // 4, d, device='cuda')      # 每卡只持 1/4 sequence
    ln = SPLayerNorm(d).cuda()
    h_sp = ln(x_sp)                                      # 本地 LN
    h_tp = sp_to_tp(h_sp, tp)                            # → (B, T, d) 进 TP attention
    # ... 假设 TP attention 算完 h_tp 仍是 (B, T, d) ...
    h_sp2 = tp_to_sp(h_tp, tp)                           # reduce-scatter 回 SP region
    print(rank, h_sp.shape, h_tp.shape, h_sp2.shape)
```

关键点：`SPLayerNorm` 在 SP region 里**本地独立算**（不需通信，因为 $d$ 维完整），TP/SP 边界用 `all_gather` / `reduce_scatter` 转换。生产实现（Megatron-Core `tensor_parallel.mappings`）会把这些 collective fuse 到 GEMM kernel 里减 overhead，但语义就是这段 demo 的形状。

### 3.2 Ring Attention 算法骨架（online softmax 累加）

```python
# ring_attention_skeleton.py（教学版骨架，省略 mask、bf16 等细节）
import torch, torch.distributed as dist

def ring_attention(Q_local, K_local, V_local, cp_group):
    """
    Q/K/V_local: (B, T/N, h, d_head)  每卡持 1/N 的 sequence
    返回 O_local: (B, T/N, h, d_head)  每卡持完整 attention 的 1/N 输出
    """
    N = dist.get_world_size(cp_group)
    rank = dist.get_rank(cp_group)
    next_r = (rank + 1) % N
    prev_r = (rank - 1) % N

    # online softmax 状态：running max m 与 normalizer ℓ
    O = torch.zeros_like(Q_local)
    m = torch.full(Q_local.shape[:-1] + (1,), float('-inf'), device=Q_local.device)
    ell = torch.zeros_like(m)

    K_buf, V_buf = K_local.clone(), V_local.clone()       # 当前持有的 K/V tile
    K_recv = torch.empty_like(K_buf)
    V_recv = torch.empty_like(V_buf)

    for step in range(N):
        # 启动下一步的 K/V 传输（异步，与本地 attention compute overlap）
        if step < N - 1:
            req_k = dist.isend(K_buf, next_r, group=cp_group)
            req_v = dist.isend(V_buf, next_r, group=cp_group)
            req_kr = dist.irecv(K_recv, prev_r, group=cp_group)
            req_vr = dist.irecv(V_recv, prev_r, group=cp_group)

        # 用 Q_local 与当前 K_buf, V_buf 算 partial attention
        S = torch.einsum('bthd,bshd->bhts', Q_local, K_buf) / (Q_local.size(-1) ** 0.5)
        m_new = torch.maximum(m, S.max(dim=-1, keepdim=True).values.transpose(1, 2))
        P = (S - m_new.transpose(1, 2)).exp()                                  # rescale
        ell_new = ell * (m - m_new).exp() + P.sum(dim=-1, keepdim=True).transpose(1, 2)
        O = O * (m - m_new).exp() + torch.einsum('bhts,bshd->bthd', P, V_buf)
        m, ell = m_new, ell_new

        if step < N - 1:
            req_k.wait(); req_v.wait(); req_kr.wait(); req_vr.wait()
            K_buf, V_buf = K_recv.clone(), V_recv.clone()                       # 环上下一块

    return O / ell.transpose(1, 2)                                              # 最终 normalize
```

关键点：(1) `isend/irecv` 异步传 K/V tile，与本地 GEMM overlap；(2) **online softmax** 维护 running max $m$ 和 normalizer $\ell$，每步来新数据时重新 rescale 旧 output（与 5.3 FlashAttention 同公式）；(3) 转完 $N$ 步后每卡得到自己 $T/N$ 个 token 的完整 attention。生产实现（Megatron-LM CP / TransformerEngine CP）还会做 striped 切分、causal mask 早跳、tree-topology 等优化，但核心 loop 是这段骨架。

### 3.3 EP all-to-all 示意：MoE token 重排

```python
# ep_all_to_all_demo.py（演示 dispatch + combine 的 all-to-all）
import os, torch, torch.distributed as dist

def moe_dispatch(tokens, expert_ids, ep_group):
    """
    tokens:     (T, d)         本卡的 T 个 token
    expert_ids: (T,) long      每个 token 路由到的 expert id（0 ~ N-1）
    ep_group:   EP process group，size = M（卡数）
    返回每卡收到的所有去本地 expert 的 token，连同来源信息
    """
    M = dist.get_world_size(ep_group)
    n_per = (expert_ids.max() + 1) // M                   # 每卡 expert 数 N/M
    target_rank = expert_ids // n_per                     # token 该去哪张卡

    # 按 target_rank 排序 → 连续段（all-to-all 要求每段连续）
    order = target_rank.argsort()
    tokens_sorted = tokens[order]
    target_sorted = target_rank[order]

    # 算 send/recv 的 split sizes（每张卡发给 rank j 多少 token）
    send_counts = torch.bincount(target_sorted, minlength=M).tolist()
    recv_counts = [0] * M
    dist.all_to_all_single(
        torch.tensor(recv_counts, device=tokens.device),
        torch.tensor(send_counts, device=tokens.device),
        group=ep_group,
    )                                                      # 简化：用 all_to_all_single 交换 counts

    # 真正的 token all-to-all
    recv_buf = torch.empty(sum(recv_counts), tokens.size(-1), device=tokens.device)
    dist.all_to_all_single(recv_buf, tokens_sorted,
                           output_split_sizes=recv_counts,
                           input_split_sizes=send_counts,
                           group=ep_group)
    return recv_buf, order, send_counts, recv_counts      # 后两个用于 combine 反向

# combine 是把 expert 输出按 (send_counts, recv_counts) 反向再 all_to_all_single 一次
```

关键点：(1) MoE 的 all-to-all 是 **uneven**（每卡发给不同卡的 token 数不同，靠 `input/output_split_sizes` 描述）；(2) NCCL 的 `all_to_all_single` 已支持 uneven split，但效率不如 DeepEP；(3) 实际 MoE layer 还要先做 token 排序把同 expert 的连续放，再做 all-to-all，最后 combine 时反向重排。这段 demo 省掉了 K=2 时一个 token 复制 K 份的细节。

### 3.4 5D Parallel 的 DeviceMesh 配置（PyTorch 2.4+）

```python
# 5d_mesh_demo.py（PyTorch 2.4+ 推荐写法）
from torch.distributed.device_mesh import init_device_mesh

# 假设 2048 GPU 训 DeepSeek-V3-style MoE 模型
# 配置: DP=4, TP=4, PP=8, EP=16 (TP 和 SP 共享，CP 此处不开)
mesh = init_device_mesh(
    'cuda',
    mesh_shape=(4, 8, 16, 4),                           # (DP, PP, EP, TP)
    mesh_dim_names=('dp', 'pp', 'ep', 'tp'),
)

dp_group = mesh['dp'].get_group()                       # 跨 cluster grad sync
pp_group = mesh['pp'].get_group()                       # PP send/recv
ep_group = mesh['ep'].get_group()                       # MoE all-to-all
tp_group = mesh['tp'].get_group()                       # TP all-reduce + SP all-gather/RS

# Sequence parallel 共用 tp_group（不建独立 group）
sp_group = tp_group

# Long-context 时再加 CP：mesh_shape 末尾加一维即可，dim_names 加 'cp'
# 训 RoPE 时每张卡的 token global offset 用 mesh['cp'].get_local_rank() 算
```

关键点：(1) `DeviceMesh` 把多维并行表达成一个 N-D 张量，每个 dim 对应一种并行；(2) 各维度的 process group 直接 `mesh[dim_name].get_group()` 拿，不用手算 rank；(3) **SP 不建独立 group**，复用 TP group；(4) PyTorch 2.4 之前用的是手写 `init_process_group + dim_by_dim_split_groups`，老代码常见但已 deprecated。

---

## 4. 工程踩坑与经验

- ❗ **SP 与 TP 必须共享同一 process group**——SP 的 all-gather/reduce-scatter 与 TP 的 all-reduce 是"等价改写"关系（all-reduce ≡ reduce-scatter + all-gather），共用 group 才能让 fused kernel（TransformerEngine 把 GEMM + collective 融成一个 kernel）正确工作。新人常见错误是为 SP 单独 `init_process_group` 建一组，结果通信路径变长且 fusion 失效，wall-clock 反而变慢
- ❗ **Ring Attention 的 RoPE position 必须用 global offset**——CP=N 时，rank $r$ 持有的本地 token $t$ 在原始 sequence 里的位置是 $r \cdot T/N + t$（连续切）或 $r + t \cdot N$（striped 切）。如果偷懒用 `position = local_t` 算 RoPE，**attention 数值在每张卡都错位且不可恢复**。这是 CP 实现里的头号 silent bug——loss 看着正常，长 context eval 全 0。Debug 第一步：打印每张卡的 `cos/sin` 第一项与单卡参考实现对比
- ❗ **CP 与 DP 切的都是 batch/seq，不要混淆**——DP 切 batch（每卡看不同 sample），CP 切 sequence（每卡看同一 sample 的不同段）。long-context 训练时 batch size 通常 = 1（单 sample 已经 100k+ token），CP 维度 = 8/16，DP 维度只能从 sample 数算（数据集没那么多 100k+ sample）。**有的代码把 CP 的 sequence chunk 错当 DP batch 处理，loss 变成"每段独立训"完全错**——务必在数据 loader 层就把 sample 按 CP rank 切片，而不是先 batch 再切
- ❗ **EP 的 all-to-all 是 MoE 训练的最大瓶颈，NCCL 默认实现远不够快**——NCCL 把 all-to-all 当通用 collective 处理，没利用 NVLink/IB 拓扑差异，跨 node 的 all-to-all 延迟特别糟。生产 MoE 训练几乎必须上 **DeepEP**（DeepSeek 开源）或 **Tutel**（Microsoft）或 **Megablocks**（Databricks），各自实测比 NCCL 快 2-3×。不熟悉这些 kernel 库前不要轻易上 256-expert MoE
- ❗ **MoE 的 RLHF 显存极重，policy / ref / reward 全部 expert 化**——9.3 节 PPO 的 4 模型 + EP=64 = 256 个 expert 的实例同时在显存里。一个折衷：**只 EP policy，ref / reward 用 dense 等效模型（distilled）**，能省 2-3 倍显存。这是 2025-2026 年开源社区还在摸索的方向，DeepSeek-R1 的 RL pipeline 没完全公开
- ❗ **DeviceMesh API 在 PyTorch 2.4 才稳定**——2.0/2.1 上 API 有 breaking change，2.2/2.3 部分功能有 bug。如果集群 PyTorch 版本是 2.1，用旧的 `init_process_group` + 手动建多个 sub-group（用 `new_group` 按 rank 列表切）写法。Megatron-Core 在 2.1 之前的版本里就是手动建 group + rank 算术，2.4 之后才迁到 DeviceMesh
- ❗ **Long-context 训练时 CP 的 batch size 通常 = 1**，每个 step 只算一个 sample（已经被 CP 切到多卡），与 DP "batch 跨卡复制"思路完全不同。这导致**梯度累加 step 数极大**（要凑 1M token global batch 需要数百 step），debug 周期变长。建议先用短 context（4k）跑通 5D 流水再切到 long-context
- ❗ **EP 内每个 expert 内部还可以 TP（嵌套）**，DeepSeek-V3 是 TP=4 内套 EP=64。嵌套时 expert FFN 内部的 weight 也按 §7.2 §2.1 的 column/row 切，所以一个 expert 的 forward = TP all-reduce + EP all-to-all 双重通信。**不要试图在嵌套层之间共享 process group**——TP group 只在同一个 expert 的内部 4 卡之间，EP group 跨所有 64 卡，两者必须独立
- ❗ **PP + CP 协同时，每个 stage 的 layer 数与 CP 切片必须对齐**——如果 stage 0 持 layer 1-10、stage 1 持 layer 11-20，每个 stage 内部都要做 CP attention。**activation 在 PP 边界传输时 shape 是 (B, T/CP, d)**（CP-切的）而不是 (B, T, d)。这要求 send/recv 的 buffer shape 提前算好，否则 NCCL hang。Megatron-LM 在 PP+CP 协同时手动做了 send/recv shape 协议，Open-Sora / Megatron-LM 0.7+ 才稳定支持
- ❗ **SP 与 activation checkpointing 协同时，rematerialization 也要按 SP shape**——7.5 节会讲 activation checkpointing。如果 SP region 的 LayerNorm 输出被 checkpoint 掉、backward 时 recompute，recompute 的 forward 必须仍走 (B, T/k, d) shape，不能不小心走成完整 (B, T, d) ——否则显存反而暴涨。Megatron-Core 的 selective checkpointing 已正确处理，但自己实现 checkpoint wrapper 时极易踩

---

## 5. 经典 paper

- **Korthikanti et al., 2022 — Reducing Activation Recomputation in Large Transformer Models** — Megatron-SP 的提出。读 §4 的 sequence parallelism + selective activation recomputation 组合，理解为什么"all-reduce ≡ reduce-scatter + all-gather"的等价改写能把 LN/Dropout activation 砍到 $1/k$ 而不增通信量。本节 §2.1 全部基于此
- **Liu et al., 2023 — Ring Attention with Blockwise Transformers for Near-Infinite Context** — Ring Attention 起源。读 §3 看 ring 拓扑下 K/V tile 流转 + online softmax 累加的算法，理解 Ring Attention 与 FlashAttention 的同源关系（都是 tile-wise online softmax，区别在 tile 流的位置：单卡 SRAM vs 多卡 NVLink）
- **Brandon et al., 2023 — Striped Attention** — Ring Attention 的 load imbalance 优化，causal mask 下 striped 切比连续切平均算量提升 ~2×。读这篇能理解为什么 NVIDIA TransformerEngine 的 CP 默认是 striped 而非朴素 ring
- **Lepikhin et al., 2020 — GShard** & **Fedus et al., 2022 — Switch Transformer** — EP 的奠基。GShard 第一次把 MoE + all-to-all 路由系统化（含 capacity factor / token dropping 的工程协议），Switch 把 EP scaling 推到 1.6T 总参。本节 §2.3 的 EP 抽象沿用 GShard 的术语
- **DeepSeek-AI, 2024 — DeepSeek-V3 Technical Report** — DeepEP + bias balancing + EP×TP 嵌套的工业级实现。读 §3 的 MoE infra 与 §4 的训练系统，理解 256-expert / 671B 模型在 2048 GPU 上怎么跑下来。**2024 年 MoE infra 最值得精读的 tech report**
- **加分阅读：Jacobs et al., 2023 — DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models** — SP 的另一变体，把 attention 在 head 与 sequence 维度间 all-to-all 互换。是 Megatron-SP 之外的 long-context 训练路线，理解 SP 的"另一种切法"

---

## 6. 自测与面试题

**Q1（架构）：** SP / CP / EP 三者各自切的是什么维度？分别解决什么 bottleneck？为什么 SP 与 TP 必须共用 process group 而 CP / EP 不需要？

<details>
<summary>Answer sketch</summary>

切的维度与解决的 bottleneck：

- **SP**（Sequence Parallel）：在 LayerNorm / Dropout / residual 等 token-wise op 处沿 **sequence 维 T** 切。解决的 bottleneck 是**这些 op 的 activation 单卡 buffer 太大**（TP 切了 $d$ 但没切 $T$，LN 输入 $\mathbb{R}^{B \times T \times d}$ 在每张卡都要存完整一份）。SP 把它降到 $1/k$
- **CP**（Context Parallel）：沿 **sequence 维 T** 切，但作用范围是 **attention 内部**。解决的 bottleneck 是 long-context 训练时 attention matrix $O(T^2)$ 单卡装不下。配合 Ring/Striped Attention 在多卡间流转 K/V tile，每卡只持 $1/N$ 的 sequence
- **EP**（Expert Parallel）：沿 **expert 维 N** 切。解决的 bottleneck 是 MoE 模型 N 个 expert 的总参数量太大单卡装不下（DeepSeek-V3 256 expert 共数百 GB），通过 all-to-all 把 token 路由到 expert 所在 GPU

为什么 SP 与 TP 共用 process group：

- SP 的 all-gather + reduce-scatter 是 **TP 原 all-reduce 的等价改写**（all-reduce 通信量 = 2BTd ≡ reduce-scatter (BTd/k) + all-gather (BTd/k) 的总量），通信路径完全沿用 TP，必须同 group 才能让 fused GEMM+collective kernel 正确工作
- CP 切 attention 内部的 sequence，与 TP 切 head 是**正交维度**，必须独立 group
- EP 切 expert 维度，与 TP/SP/CP 都正交，独立 group

加分：指出 SP 不增加额外通信量（all-reduce ≡ RS + AG），是"免费"摊薄 LN/Dropout activation 的优化

</details>

**Q2（计算 / 对比）：** Ring Attention 的通信量是多少？与 TP attention 比谁更适合 long-context 训练？给出一个量化对比。

<details>
<summary>Answer sketch</summary>

Ring Attention 通信量：

- 每步传一个 K/V tile，shape $(B, T/N, h, d_{head})$，bf16 占 $2 \cdot B \cdot T/N \cdot h \cdot d_{head}$ bytes（K + V 各一份所以 ×2，再 ×2 因为 bf16）
- 共 $N-1$ 步（环转一圈），总通信量 $\approx 2 \cdot 2 \cdot B \cdot T \cdot h \cdot d_{head}$ = $O(BTd)$（$h \cdot d_{head} = d$）
- **关键性质**：通信量与 $T$ **线性**而非 $T^2$，这是 Ring Attention 相对单卡 attention（不可行，$O(T^2)$ 显存）和 TP attention（head 数限制）的根本优势

对比 TP attention：

| | TP attention (TP=N) | Ring Attention (CP=N) |
|---|---|---|
| 切的维度 | head（$h$ 切到 N 卡）| sequence（$T$ 切到 N 卡）|
| 单卡 KV 存量 | $T \cdot h/N \cdot d_{head}$ | $T/N \cdot h \cdot d_{head}$ |
| 单层通信量 | 1 次 all-reduce ≈ $2BTd$ | $N-1$ 次 send/recv ≈ $2BTd$ |
| 上限 | $N \leq h$（head 数硬约束，LLaMA-3 8B 是 32 head 所以 TP ≤ 32，但 NVLink 物理限制 TP ≤ 8）| $N$ 任意（可扩到几十几百）|
| 适合场景 | 通用 < 32k context，TP 已经把单 layer 显存切够了 | long-context ≥ 64k，单卡 attention matrix 装不下 |

**long-context 训练**：CP 完胜。原因是 (1) TP 受 head 数限制扩不到太大；(2) 长 context 下 attention $O(T^2)$ 显存只有切 $T$ 才能解；(3) Ring Attention 的通信量 $O(BTd)$ 与 sequence 长度线性，scaling 友好

加分：指出实际 long-context 训练里两者**叠加用**——TP 切 head（intra-node），CP 切 sequence（intra-node 或跨 node），各管一半

</details>

**Q3（5D 配置）：** 给定 1024 GPU（128 个 H100 8 卡 node，node 内 NVLink，node 间 IB 400 Gbps）训练 DeepSeek-V3-style MoE 模型（256 expert / top-8 / 61 层 / hidden 7168）。给出一个合理的 (DP, TP, PP, EP) 配置，并说明 mesh 拓扑映射。

<details>
<summary>Answer sketch</summary>

参考配置：**TP=4, EP=16, PP=8, DP=2**（$4 \times 16 \times 8 \times 2 = 1024$）

mesh 拓扑映射（核心原则：通信越密放在带宽越高的互联上）：

- **TP=4 装在 1 个 node 内的 4 卡**：每个 expert 内部 FFN 做 TP=4 切，每 layer 4 次 all-reduce 走 NVLink。**TP 不能跨 node**（同 7.2 §4 硬约束）
- **EP=16 跨 node**：256 expert 分给 16 卡，每卡 16 个 expert。EP 的 all-to-all 跨 node 通过 IB 400 Gbps，需要上 DeepEP 优化才能扛住通信开销。**注意 EP 与 TP 嵌套**：每张卡同时是 TP=4 group 内的一员、EP=16 group 内的一员，TP×EP = 64 卡 = 8 个 node 物理映射
- **PP=8 跨 node**：61 层切 8 段，每段 ~8 层，跨 node 传 activation。PP send/recv 稀疏，IB 够用。每个 PP stage 占 64 卡 = 8 个 node
- **DP=2 跨集群**：$1024 / (4 \times 16 \times 8) = 2$，再切两份做 DP，每份 512 卡。DP 通信最稀疏

micro-batch：$m \geq 4 \times \text{PP} = 32$。设单 micro-batch $b_\mu = 1$、seq $T = 4096$ → global token = $2 \times 32 \times 4096 \approx 0.26M$ token/step（V3 实际训练用的更大，要用更高 $b_\mu$ 或更长 seq 凑到 ~10M token/step）

叠加优化：

- **DeepEP**：EP=16 跨 node 的 all-to-all 必须用 DeepEP（NCCL 默认会塌）
- **bias-based balancing**（5.4 §2.3）：避免 aux loss 污染主 loss
- **fp8 训练**：与 V3 原 paper 一致，省 50% 显存与带宽
- **selective activation checkpointing**（7.5 节）：attention 不 checkpoint，FFN/expert 输出 checkpoint
- **Sequence Parallel**：与 TP=4 配套开（共用 group），免费摊薄 LN/Dropout activation

加分：

- 指出 **TP=4 而非 TP=8** 是因为 V3 expert 较小（fine-grained MoE），TP=8 会让单 expert 内部 GEMM underutilize
- 指出 EP=16 跨 node 是 TP×EP 嵌套必经之路（EP 不可能装在单 node 内，单 node 8 卡放不下 256 expert）
- 提到这个配置近似 DeepSeek 自己公开的 V3 训练配置（TP=4, EP 几十到 256 视集群规模），不能简单照搬 LLaMA-3 dense 模型的 TP=8 + PP=16

</details>

---

## 7. 延伸阅读

- [Megatron-LM Sequence Parallel 文档](https://github.com/NVIDIA/Megatron-LM/blob/main/docs/source/api-guide/tensor_parallel.rst) — Megatron-Core 的 SP 实现细节，配 `tensor_parallel/mappings.py` 源码读
- [NVIDIA Transformer Engine — Context Parallel](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/context_parallel.html) — TE 的 CP 实现（含 Ring + Striped + Tree 多种拓扑）
- [Ring Attention 官方实现 (Liu et al.)](https://github.com/lhao499/RingAttention) — 论文配套代码，含 JAX/PyTorch 双版本
- [DeepEP (DeepSeek)](https://github.com/deepseek-ai/DeepEP) — 必看，工业级 EP all-to-all kernel，含 NVSHMEM-based 异步 dispatch 实现
- [Tutel (Microsoft)](https://github.com/microsoft/tutel) — 另一套 MoE EP 框架，NCCL-based 但做了大量调度优化
- [Megablocks (Databricks)](https://github.com/databricks/megablocks) — block-sparse MoE GEMM kernel，与 EP 通信库正交（管 expert 内部计算高效化）
- [TorchTitan 5D Parallel 示例](https://github.com/pytorch/torchtitan) — PyTorch 团队的 reference 实现，DeviceMesh + DP+TP+PP+SP 完整配置代码，建议作为新项目的起点
- [DeepSeek-V3 Tech Report](https://arxiv.org/abs/2412.19437) — 必读 §3 (MoE) + §4 (Infra)，看 EP×TP 嵌套与 DeepEP 的真实配置
- 推荐继续读本教程的 **7.4 混合精度**（fp8 训练与 5D parallel 协同）、**7.5 显存优化**（activation checkpointing 与 SP/CP/EP 协同）、**7.6 Triton/CUDA kernel**（理解 DeepEP / TE 内部通信 kernel 怎么做的）
