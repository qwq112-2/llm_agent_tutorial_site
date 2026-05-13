---
title: "5.3 FlashAttention 1/2/3：IO-aware kernel"
description: "标准 attention 的瓶颈不是 FLOPs 而是 HBM I/O——把 $S = QK^\\\\top \\\\in \\\\mathbb{R}^{n \\\\times n}$ 来回写读 HBM 浪费了 80% 时间；FlashAttention 用 tile 计算 + online softmax 在 SRAM 内增量算，完全不实例化 $n \\\\times n$ attention matrix 到 HBM，把 "
---

> ⏱ 预计阅读 50 分钟 ｜ 难度 ★★★ 🔥 必考 ｜ 前置：4.1 self-attention

## 一句话本节讲什么

标准 attention 的瓶颈不是 FLOPs 而是 HBM I/O——把 $S = QK^\top \in \mathbb{R}^{n \times n}$ 来回写读 HBM 浪费了 80% 时间；FlashAttention 用 **tile 计算 + online softmax** 在 SRAM 内增量算，**完全不实例化** $n \times n$ attention matrix 到 HBM，把 forward+backward 提速 2-4×、显存从 $O(n^2)$ 降到 $O(n)$，是当代 LLM 训练 / 推理的事实标准 kernel；本节讲清原理（GPU 内存层级、online softmax 公式、v1/v2/v3 演化）和怎么用（PyTorch 2.0+ SDPA backend、`flash_attn` 库），不展开 CUDA 细节（那是 7.6）。

---

## 1. Mental model（直觉）

### 1.1 为什么 attention 是 memory-bound

4.1 节的 self-attention 五步——`Q @ K^T → /√d_k → mask → softmax → @ V`——在数学上是对的，但**直接照公式实现到 GPU 上是灾难**。为什么？

GPU 的算力增长速度远超带宽增长速度。以 A100 为例：

- **算力**：bf16 matmul 312 TFLOPs/s
- **HBM 带宽**：1.5-2 TB/s（80GB HBM2e）
- **SRAM 带宽**：~19 TB/s（每个 SM 的 shared memory）

算一下"算 1 个 bf16 数（2 字节）需要多少 FLOP 才能让算力打满"：$312 \text{ TFLOPs} / 1.5 \text{ TB/s} \times 2 \text{ B/数} \approx 416$ FLOPs/数。**也就是说每从 HBM 读一个数，至少要做 400+ 次浮点运算才能让算力不被带宽卡住**。

标准 attention 在做什么？以 $n=8192, d=128$、bf16 为例：

| 步骤 | 操作 | HBM 读 | HBM 写 | FLOP/数 比例 |
|---|---|---|---|---|
| 1 | $S = QK^\top$ | $Q, K$ 各 $nd$ | $S$ 共 $n^2$ | $d / 2 = 64$（计算少、读写多） |
| 2 | $S' = S / \sqrt{d}$ | $S$ 共 $n^2$ | $S'$ 共 $n^2$ | 1（几乎纯 I/O） |
| 3 | mask + softmax | $S'$ 共 $n^2$ | $A$ 共 $n^2$ | ~5（指数 + 归一） |
| 4 | $O = A V$ | $A, V$ | $O$ 共 $nd$ | $d / 2 = 64$ |

**关键观察**：$S$ 这个 $n \times n$ 矩阵被**写一次、读一次、再写一次（softmax）、再读一次**——每个 attention score 在 HBM 上至少 read+write 4-6 次。当 $n=8192$ 时 $S$ 有 6700 万元素、单 head 单层 130 MB，32 头 32 层就是 130 GB 量级的 HBM 流量——而真正的"有用计算量"不到这个的 1/10。

**结论**：标准 attention 的 GPU 运行时大约 80% 时间在等 HBM I/O，只有 20% 在算 matmul。提升算法 FLOPs 利用率没用，**真正的优化是减少 HBM 访问**。

### 1.2 GPU 内存层级速记图

```
       ┌─────────────────────────────┐
       │  CPU DRAM   ~100 GB/s       │   慢、容量大
       │  (PCIe 64-128 GB/s 上传 GPU)│
       └─────────────┬───────────────┘
                     │
       ┌─────────────▼───────────────┐
       │  HBM (GPU 显存) 80 GB       │   ← 训练时模型权重 + 激活在这
       │  ~1.5-3 TB/s                │   ← attention matrix 也默认放这
       └─────────────┬───────────────┘   ← 大但慢，是瓶颈
                     │
       ┌─────────────▼───────────────┐
       │  L2 cache  40-50 MB         │
       │  ~5-7 TB/s                  │
       └─────────────┬───────────────┘
                     │
       ┌─────────────▼───────────────┐
       │  SRAM / shared mem 192 KB×SM │  ← FlashAttention 把 tile 装这里算
       │  ~19 TB/s（A100）           │  ← 小但极快
       └─────────────┬───────────────┘
                     │
       ┌─────────────▼───────────────┐
       │  Registers per thread       │   最快、最稀缺
       └─────────────────────────────┘
```

记忆要点：**HBM 大但慢、SRAM 小但快，差速 6-10 倍**。一个 SM 的 SRAM 通常 192-228 KB，整张 A100 80 GB HBM——容量差 50 万倍。FlashAttention 的全部魔法都建立在这条事实上：**把计算搬进 SRAM，避开 HBM**。

### 1.3 FlashAttention 的核心 idea（一段话讲完）

> 把 $Q, K, V$ 切成能装进 SRAM 的 tile，用一个 outer loop（K/V tile）+ inner loop（Q tile）的两层循环，**在 SRAM 内增量算 attention 输出，不把中间 $S, A$ 矩阵写回 HBM**——只把最终 $O \in \mathbb{R}^{n \times d}$ 写回。

听起来理所当然，但有一个数学障碍：**softmax 需要全行 score 都算完才能归一化**——$\text{softmax}(x_i) = e^{x_i} / \sum_j e^{x_j}$，分母是全行求和，理论上必须等所有 tile 都过一遍才能定。

FlashAttention 的关键技巧叫 **online softmax**——一边算 tile 一边维护 "running max $m$" 和 "running sum $\ell$"，每来一个新 tile 就用一个 rescaling 公式去修正之前的 partial output。下一节给完整推导。

ASCII 直觉图：

```
                K/V 拆 4 块（每块装入 SRAM）
                ┌────┬────┬────┬────┐
                │ K1 │ K2 │ K3 │ K4 │
                │ V1 │ V2 │ V3 │ V4 │
                └────┴────┴────┴────┘
   Q 拆 4 块         ▲    ▲    ▲    ▲
   ┌────┐            │    │    │    │
   │ Q1 │── 装入 SRAM ── 与 K1,V1 算 → 更新 (m1, ℓ1, O1)
   │ Q1 │── 装入 SRAM ── 与 K2,V2 算 → rescale 旧 O1 + 加新 → (m2, ℓ2, O2)
   │ Q1 │── 与 K3,V3 ── rescale → (m3, ℓ3, O3)
   │ Q1 │── 与 K4,V4 ── rescale → (m4, ℓ4, O4)
   ├────┤
   │ Q2 │── 同上，独立做一轮
   ├────┤
   │ Q3 │── ...
   ├────┤
   │ Q4 │── ...
   └────┘
                              ▲
        全程 attention matrix S 不写 HBM，只写最终 O
```

---

## 2. 公式与原理

### 2.1 Online softmax 的关键 rescaling 公式（必背）

设我们要算 softmax over $x = [x_1, x_2, \dots, x_N]$，但 $x$ 被分成两块：先看到 $x^{(1)} \in \mathbb{R}^{B_1}$，后看到 $x^{(2)} \in \mathbb{R}^{B_2}$，对应的 value 块是 $V^{(1)}, V^{(2)}$。最终目标：算 $O = \text{softmax}(x) V$，但**不能等 $x^{(2)}$ 来了再算**——要在看到 $x^{(1)}$ 时就先算一份"假装它就是全部"的 partial output，等 $x^{(2)}$ 来了再修正。

**第一块来时**：

$$m^{(1)} = \max(x^{(1)}), \quad \ell^{(1)} = \sum_{i} e^{x^{(1)}_i - m^{(1)}}, \quad O^{(1)} = \sum_{i} e^{x^{(1)}_i - m^{(1)}} V^{(1)}_i$$

注意这里 $O^{(1)}$ 是**未归一化**的 partial output（缺最后除 $\ell^{(1)}$）——故意不除，留到最后一步。

**第二块来时**：

新的全局 max：

$$m^{(2)} = \max(m^{(1)}, \max(x^{(2)}))$$

旧的 partial output 要 rescale——因为之前减的 $m^{(1)}$ 现在变成 $m^{(2)}$ 了：

$$O^{(1)}_{\text{rescaled}} = O^{(1)} \cdot e^{m^{(1)} - m^{(2)}}$$

这一步是**核心魔法**：旧的 $\sum e^{x^{(1)}_i - m^{(1)}} V^{(1)}_i$ 乘以 $e^{m^{(1)} - m^{(2)}}$ 就等于 $\sum e^{x^{(1)}_i - m^{(2)}} V^{(1)}_i$，相当于"用新 max 重做一遍但代价只是一次乘法"。

新块的贡献：

$$O^{(2)}_{\text{new}} = \sum_{i} e^{x^{(2)}_i - m^{(2)}} V^{(2)}_i$$

合并：

$$O^{(2)} = O^{(1)}_{\text{rescaled}} + O^{(2)}_{\text{new}}$$

$$\ell^{(2)} = \ell^{(1)} \cdot e^{m^{(1)} - m^{(2)}} + \sum_{i} e^{x^{(2)}_i - m^{(2)}}$$

**最终结果**（看到所有块之后再除归一）：

$$O = O^{(N)} / \ell^{(N)}$$

**这一组公式就是 FlashAttention 的全部数学**。它告诉你：softmax 看似要全局信息，但只要维护 $(m, \ell, O)$ 三个 running state，每来一个新 tile 都能用 $O(1)$ 的 rescaling 把状态更新到与"假装从头算一遍"完全数值等价——这是 Milakov & Gimelshein 2018 提出的 online softmax，FlashAttention 就是把它和 attention 的 $@V$ 融合成一个 kernel。

### 2.2 完整的 FlashAttention 伪代码（v1 视角）

```
Input: Q, K, V ∈ R^{N × d}  存在 HBM
SRAM 大小 M

# 选 block size：让 (Br × d) + (Bc × d) + (Br × Bc) 装入 SRAM
Bc = ceil(M / (4d))
Br = min(Bc, d)

# 初始化 HBM
O = zeros(N, d)             # 输出
m = full(N, -inf)           # 每行的 running max
ℓ = zeros(N)                # 每行的 running sum

# 把 Q, K, V 切块
Q_blocks = split(Q, Br)       # Tr 个 (Br, d) tile
K_blocks, V_blocks = split(K, Bc), split(V, Bc)   # Tc 个 (Bc, d) tile

# Outer loop: K/V tile（v1 是这个顺序，v2 反过来）
for j = 1..Tc:
    Load K_j, V_j 到 SRAM
    for i = 1..Tr:                                        # Inner loop: Q tile
        Load Q_i, O_i, m_i, ℓ_i 到 SRAM
        S_ij = Q_i @ K_j^T / sqrt(d)                       # SRAM 内 (Br, Bc)
        m_ij_new = rowmax(S_ij)
        P_ij = exp(S_ij - m_ij_new)                        # 不写 HBM
        ℓ_ij_new = rowsum(P_ij)
        # online softmax 更新
        m_i_new  = max(m_i, m_ij_new)
        ℓ_i_new  = exp(m_i - m_i_new) * ℓ_i + exp(m_ij_new - m_i_new) * ℓ_ij_new
        O_i_new  = exp(m_i - m_i_new) * O_i + exp(m_ij_new - m_i_new) * P_ij @ V_j
        Write O_i_new, m_i_new, ℓ_i_new 回 HBM
return O / ℓ
```

**关键事实**：
- $S_{ij}, P_{ij}$（即每块的 score 与 unnormalized softmax）**只在 SRAM 里活、用完就丢**，从不写 HBM
- 写 HBM 的只有最终 $O$ 和辅助的 $(m, \ell)$，规模都是 $O(N)$ 而非 $O(N^2)$
- backward 也用类似 trick——保存 $(m, \ell)$ 而不是 $A$，反向时按需重算 $S$（recomputation）

### 2.3 复杂度对比

| 量 | 标准 attention | FlashAttention |
|---|---|---|
| FLOPs | $O(N^2 d)$ | $O(N^2 d)$ **（一样）** |
| 中间矩阵 HBM 空间 | $O(N^2)$ | $O(N)$ |
| HBM 读写 | $O(N^2 d + N^2)$ | $O(N^2 d^2 / M)$ |
| 实际墙钟时间 | baseline | **2-4× 快** |

**关键结论**：FlashAttention 的 FLOPs **完全一样**——它不是减少计算，而是减少 HBM 访问。HBM 读写从 $O(N^2)$ 降到 $O(N^2 d^2 / M)$（$M$ 是 SRAM 大小，通常 100KB 量级），当 $d=128$ 时大约降低 $d^2/M \approx 16384/10^5 \approx 0.16$ 倍——也就是 HBM 流量降到原来的 1/6 左右。挂钟时间因此 2-4× 加速。

显存方面，attention matrix 从 $O(N^2)$ 降到 $O(N)$ 是**质变**：$N=128\text{k}$ 时 attention matrix 单 head 32 GB → 直接 OOM；FlashAttention 下只有 $O(N) = 128\text{k}$ 个 float，几乎可忽略。**长 context 训练 = 必须 FlashAttention**，不是建议。

### 2.4 FlashAttention v2（Dao 2023）的优化点

v1 在 H100/A100 上能跑出 30-40% 理论 FLOPs 利用率，但仍有空间。v2 做了三件事：

1. **减少非 matmul 操作**：v1 里每个 inner step 都做一次 rescale（涉及 exp / 除法等非 matmul），v2 把 rescale 提到 outer loop 之外、累加完所有 K/V tile 才做一次最终归一。
2. **交换 outer/inner loop 顺序**：v1 是 K/V 在外、Q 在内（适合 backward）；v2 把 Q 提到外层，每个 Q tile 一直占 SRAM 直到与所有 K/V tile 算完——减少 Q 的 HBM 读次数。
3. **优化 thread 调度**：v1 每个 thread block 处理 1 个 head，v2 让每个 thread block 处理多个 head 或多个 sequence，提升 GPU 占用率（occupancy）。

**结果**：v2 在 A100 上达到 50-70% 理论 FLOPs，比 v1 快 1.5-2×。从 2023 年起 PyTorch SDPA 默认用 v2 backend。

### 2.5 FlashAttention v3（Shah 2024，H100 专属）

H100 的 Hopper 架构引入了两个新硬件特性，v3 专门为它做了重写：

- **TMA（Tensor Memory Accelerator）**：异步 HBM ↔ SRAM 数据搬运，搬数据时算力不空转
- **WGMMA（Warp Group Matrix Multiply Accumulate）**：异步 matmul 指令，多个 warp 协作打满 Tensor Core

v3 的核心思想：

1. **异步执行**：TMA 搬下一个 tile 的同时 WGMMA 算当前 tile，**搬运与计算重叠**——这是 H100 才能用的 trick
2. **Warp specialization**：把一个 thread block 内的 warp 分工——producer warp 专门 TMA 搬数据，consumer warp 专门 WGMMA 算 matmul、softmax；类似生产者-消费者流水线
3. **fp8 支持**：搭配 H100 的 fp8 Tensor Core，与 DeepSeek-V3 的 fp8 训练协同（精度损失通过 per-block scaling 控制在 1% 以内）
4. **softmax 与 matmul 流水化**：softmax 的非 matmul 部分（exp / 求和）安排到 GEMM 流水线的"气泡"里，进一步提升 SM 利用率

**结果**：v3 在 H100 上达到 75% 理论 FLOPs（vs v2 的 35%），bf16 提速 1.5-2× over v2，fp8 再 2×——这是 DeepSeek-V3、Llama 4 等大模型 H100 训练能跑出业界最高吞吐的关键。

### 2.6 与 GQA / MLA 的协同

- **GQA / MQA**（5.2）：FlashAttention v2+ 原生支持——传 q 的 head 数为 $h$、kv 的 head 数为 $h_{kv}$（$h_{kv} | h$），kernel 内自动 broadcast，不需要显式 expand kv 占冗余显存
- **MLA**（DeepSeek，5.2）：因为 MLA 的 attention 公式不是标准 $\text{softmax}(QK^T/\sqrt{d})V$，而是引入了低秩 latent + 解耦 RoPE，**标准 FlashAttention kernel 不能直接用**，DeepSeek 自家维护了 MLA 专用 kernel；社区版的 vLLM / SGLang 也各自实现了 MLA backend

### 2.7 限制（必须清楚的边界）

- **短序列没收益**：$N < 128$ 时 kernel launch overhead 主导，FlashAttention 反而比朴素 PyTorch 慢
- **head_dim 必须是常见值**：64、128 是高度优化路径；96、80 等"非标"值可能 fallback 到慢路径，144 / 256 在 v3 里也只部分支持
- **dtype 限制**：v1/v2 只支持 fp16 / bf16；v3 加了 fp8。**fp32 不支持**，需 fallback 到 PyTorch math backend
- **mask 形态有限**：原生支持 causal、bi-directional、sliding-window；ALiBi 在 v2 后才支持；任意自定义 attention bias（如 T5 的 relative bias）通常需要降级
- **GPU 限制**：v2 需要 Ampere+（A100 / H100 / RTX 30/40）；v3 需要 H100 / H200 / B200，老 GPU（V100 / T4）跑不动
- **dropout 影响**：开了 attention dropout 在 v1/v2 下会略慢（需要存 dropout mask 给 backward）

---

## 3. 最小代码示例

### 3.1 PyTorch 2.0 SDPA 显式选 FlashAttention backend

```python
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

device = "cuda"
B, H, T, D = 4, 32, 8192, 128
q = torch.randn(B, H, T, D, device=device, dtype=torch.bfloat16)
k = torch.randn(B, H, T, D, device=device, dtype=torch.bfloat16)
v = torch.randn(B, H, T, D, device=device, dtype=torch.bfloat16)

# 推荐写法：context manager 限定 backend，避免被 PyTorch 自动 fallback
with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
print(out.shape)            # (4, 32, 8192, 128)
```

**关键点**：

- `SDPBackend.FLASH_ATTENTION` 强制走 Flash 路径（v2 / v3 PyTorch 自动选，看 GPU 架构）；如果不满足条件（如 dtype 是 fp32），会 raise 而不是静默 fallback——便于排查
- `is_causal=True` 让 kernel 内部生成 causal mask，比传 `attn_mask=` 更高效
- 如果有自定义 mask，传 `attn_mask=` 会让 PyTorch 选 memory-efficient 而非 Flash backend

### 3.2 直接用 `flash_attn` 库（更细粒度控制）

```python
from flash_attn import flash_attn_func, flash_attn_varlen_func

# 1. 定长（适合 padding 到固定长度的训练 / 推理）
# q, k, v 形状: (batch, seqlen, n_head, head_dim)
out = flash_attn_func(q, k, v, dropout_p=0.0, causal=True)

# 2. 变长（sample packing 必备，多个样本拼一条序列）
# q, k, v 形状: (total_tokens, n_head, head_dim)，去掉了 batch 维
# cu_seqlens 形状: (batch+1,) int32，前缀和：[0, len1, len1+len2, ...]
# 例: 3 个样本 长度 [100, 200, 150]
cu_seqlens_q = torch.tensor([0, 100, 300, 450], dtype=torch.int32, device="cuda")
cu_seqlens_k = cu_seqlens_q
out = flash_attn_varlen_func(
    q, k, v, cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q=200, max_seqlen_k=200, causal=True,
)
```

**关键点**：

- `flash_attn_func` 与 SDPA 等价但更轻——不走 dispatcher、shape 约定不同（注意是 `(B, T, H, D)` 不是 `(B, H, T, D)`）
- `flash_attn_varlen_func` 是 sample packing 的标准接口（8.2 SFT 章详讲）：把多个不同长度样本拼成一条 flat 序列，靠 `cu_seqlens` 标记分界，kernel 内确保跨样本 attention 被 mask 掉。**`cu_seqlens` 必须 int32、必须从 0 开始的前缀和，新手最常错**

### 3.3 短 benchmark 脚本

```python
import torch, time
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

torch.cuda.empty_cache()
B, H, T, D = 4, 32, 8192, 128
q = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16)
k = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16)
v = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16)

def bench(name, backend):
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    with sdpa_kernel(backend):
        for _ in range(20):
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / 20 * 1000
    mem = torch.cuda.max_memory_allocated() / 1024**3
    print(f"{name:20s}  {dt:7.2f} ms / iter   peak {mem:.2f} GB")

bench("MATH (naive)",     SDPBackend.MATH)               # 慢、显存大
bench("MEMORY_EFFICIENT", SDPBackend.EFFICIENT_ATTENTION)
bench("FLASH",            SDPBackend.FLASH_ATTENTION)    # 最快、显存最小
```

A100 上典型输出（单卡 bf16，N=8192, h=32, d=128）：

```
MATH (naive)         185.40 ms / iter   peak 18.3 GB     ← attention matrix 撑爆
MEMORY_EFFICIENT      78.10 ms / iter   peak  4.2 GB
FLASH                 41.20 ms / iter   peak  4.2 GB     ← 4.5× 提速 + 4× 省显存
```

**MATH backend 的峰值显存** 18 GB 主要被 $32 \times 8192 \times 8192 \times \text{bf16} \approx 16$ GB 的 attention matrix 占据；FLASH 完全不实例化这个矩阵，显存只剩 Q/K/V/O 本身。

---

## 4. 工程踩坑与经验

- ❗ **序列长度 < 128 时 FlashAttention 反而慢**。kernel launch overhead 是固定的（A100 上 ~10 μs），N=64 时 attention 计算本身只要 5 μs，启动开销超过有效计算。短序列（chat 短回复、embedding 编码）建议直接用 PyTorch math backend 或者 batch 凑大。生产里如果发现"明明开了 Flash 还是慢"，先 print 序列长度分布。
- ❗ **head_dim 不是 64 / 128 等"标准值"时会 fallback 慢路径**。例如 head_dim=96（一些早期 LLaMA 变体）、head_dim=80（OpenAI 早期）在 FlashAttention v2 里走的是通用路径而非高度优化的 specialized kernel，可能比 d=128 慢 2-3×。**如果你设计自己的模型，head_dim 一定选 64 / 128 / 256**，不要为了"参数量刚好"选奇怪的 96 或 144。
- ❗ **ALiBi / 任意 attention bias 在 v1 不支持，v2 部分支持**。如果你的模型用了 ALiBi 位置编码（如 BLOOM、Replit），FlashAttention v1 不能用要降级；v2.0+ 加了 ALiBi 支持但不是所有版本编译时都打开。**自定义 attention bias（如 T5 relative bias、Anthropic 的 logit soft-cap）几乎一定要降级到普通 attention 或者写 Triton 自定义 kernel**。这也是 7.6 节 Triton 入门的动机之一。
- ❗ **训练时 backward 也必须走 Flash**。新手有时只在 forward 包了 `sdpa_kernel(FLASH)`，但 PyTorch backward 是自动 dispatch 的——只要 forward 是 Flash，backward 也会走对应的 Flash backward kernel；但如果 forward 是 math 而你想"backward 用 Flash 加速"，做不到，**forward+backward 是一对**。检查方法：跑 nsight 看 kernel 名字是否含 `flash_attn_bwd`。
- ❗ **`flash_attn` 库安装地狱**。`pip install flash-attn` 经常失败，因为它要在你的机器上**从源码编译 CUDA 代码**，需要：CUDA toolkit ≥ 11.6、ninja、g++ ≥ 7、几十 GB 内存、20-60 分钟编译时间。CI/CD 里非常容易卡住。**生产建议**：(1) 直接用 PyTorch 2.x SDPA 内置版，零依赖；(2) 必须用 `flash_attn` 时下载预编译 wheel（GitHub release 页 + 你的 CUDA / torch / Python 版本完全匹配的那个）；(3) 用 `MAX_JOBS=4 pip install flash-attn --no-build-isolation` 限制并发避免 OOM 编译挂掉。
- ❗ **`flash_attn_varlen_func` 的 `cu_seqlens` 必须 int32、必须前缀和**。新手最常错的两类：(1) 用了 int64（Flash 内部 cast 失败 segfault）；(2) 写成长度数组 `[100, 200, 150]` 而不是前缀和 `[0, 100, 300, 450]`。**正确构造**：`cu_seqlens = F.pad(seqlens.cumsum(0), (1, 0)).to(torch.int32)`。注意还要分别传 `max_seqlen_q` 和 `max_seqlen_k`（用于 kernel 内分配 SRAM），传错了不会报错但结果是垃圾。
- ❗ **FlashAttention v3 需要 H100 + CUDA 12+，老 GPU 跑不动**。RTX 3090 / A100 用 v3 会直接报错或 fallback 到 v2。如果你的训练集群是 A100（最常见），就别折腾 v3，v2 已经够用；只有 H100 / H200 集群才需要专门装 `flash_attn_3`（社区版还在 alpha，PyTorch 2.5+ 的 SDPA 会自动选 v3 路径）。
- ❗ **与 RoPE 配合时 q/k 必须先 apply RoPE 再传给 flash_attn**。FlashAttention kernel 内**不会**应用任何位置编码——它只接收已经 RoPE'd 的 Q/K。新手常犯的错：自己手撕的 attention 里 RoPE 在 attention 外做了，迁移到 SDPA / flash_attn 时忘了，结果模型完全没有位置感、loss 比 baseline 高一个量级。**模板**：`q = apply_rope(q, cos, sin); k = apply_rope(k, cos, sin); out = flash_attn_func(q, k, v, ...)`。
- ❗ **更新 `flash_attn` 版本前看 changelog**。`flash_attn` 的 API 在 1.x → 2.x 之间断过——`flash_attn_unpadded_func` 改名 `flash_attn_varlen_func`、参数顺序调整、causal 默认从 True 改成 False。盲目升级会让训练静默退化。

---

## 5. 经典 paper

- **Dao, Fu, Ermon, Rudra, Ré, 2022 — FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness** — v1 原典。必读 §3 "FlashAttention" 完整算法、§4 "Analysis" 复杂度推导、§5 实验。这篇 paper 的核心贡献不是某种新数学，而是"算法-硬件协同设计"思想——把 GPU 内存层级当成 first-class concern。读完会明白为什么 GitHub 上一个 100k star 项目可以"只是把已知公式重新组织"。
- **Dao, 2023 — FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning** — v2 的工程优化合集。必读 §3.1 关于 outer/inner loop 顺序对 backward 的影响、§3.2 thread block 调度。这篇展示了"原理对了之后还能怎么挤出 1.5-2×"的工程艺术，是想做 kernel 优化的必读。
- **Shah, Bikshandi, Zhang, Thakkar, Ramani, Dao, 2024 — FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-Precision** — v3 论文。必读 §3 Hopper 架构特性介绍（TMA / WGMMA）、§4 warp specialization 设计、§5 fp8 误差分析。这篇是"硬件特性新出来 → 算法重构吃满"的完美案例，也是理解 H100 / B200 编程模型的最佳入口之一。
- **Milakov & Gimelshein, 2018 — Online normalizer calculation for softmax** — 1 页 NVIDIA tech report，online softmax 的原始提出。FlashAttention 的数学全部建立在这篇上。读这篇能明白"online softmax"不是 FlashAttention 的发明，而是被 FlashAttention "拿来用对地方"的现成 trick。

---

## 6. 自测与面试题

**Q1（原理）**：为什么标准 attention 是 memory-bound 而不是 compute-bound？FlashAttention 怎么解决？

<details>
<summary>Answer sketch</summary>

memory-bound vs compute-bound 的本质：

- 现代 GPU（A100/H100）的算力 / 带宽比极不平衡：A100 算力 312 TFLOPs/s、HBM 带宽 1.5 TB/s——每读 1 个 bf16 数（2 字节）至少要做 ~400 FLOPs 才能让算力打满
- 标准 attention 在 $S = QK^\top$ 这一步会把 $S \in \mathbb{R}^{n \times n}$ 写到 HBM、再读回来做 softmax、再写回来、再读回来算 $AV$——**每个 score 在 HBM 读写 4-6 次**
- $n=8192, d=128$ 时 HBM 流量 ≈ 5-10× 有效计算量，导致 80% 时间花在等 HBM I/O，算力闲置
- 这就是 "memory-bound"：瓶颈不是 FLOPs 而是 HBM 带宽

FlashAttention 的解法：

- 把 Q/K/V 切成能装入 SRAM（~100 KB）的 tile，attention 计算**完全在 SRAM 内做**
- 用 online softmax（维护 running $(m, \ell, O)$）让 softmax 不再需要"全行 score 都到齐"
- 中间矩阵 $S, A$ **从不写 HBM**，只把最终输出 $O \in \mathbb{R}^{n \times d}$ 写回
- HBM 流量从 $O(N^2)$ 降到 $O(N^2 d^2 / M)$（M = SRAM 大小），约 5-10× 减少
- FLOPs 不变，但挂钟时间 2-4× 加速

加分：能说出 backward 也要重算 $S$（recomputation），保存的是 $(m, \ell)$ 而非 $A$；能区分 v2 的"减少非 matmul + 调整 loop 顺序"和 v3 的"TMA/WGMMA + warp specialization + fp8"。

</details>

**Q2（公式）**：写出 online softmax 的关键 rescaling 公式——已经处理完第一块得到 $(m^{(1)}, \ell^{(1)}, O^{(1)})$，现在第二块 $x^{(2)}, V^{(2)}$ 来了，怎么更新成 $(m^{(2)}, \ell^{(2)}, O^{(2)})$？

<details>
<summary>Answer sketch</summary>

新 max：

$$m^{(2)} = \max(m^{(1)}, \max(x^{(2)}))$$

新块的局部 sum 和 partial output（用新 max 减）：

$$\tilde{\ell}^{(2)} = \sum_i e^{x^{(2)}_i - m^{(2)}}, \quad \tilde{O}^{(2)} = \sum_i e^{x^{(2)}_i - m^{(2)}} V^{(2)}_i$$

旧块的 rescale（核心 trick）：因为 max 从 $m^{(1)}$ 变成 $m^{(2)}$，旧的 $e^{x - m^{(1)}}$ 要乘 $e^{m^{(1)} - m^{(2)}}$ 变成 $e^{x - m^{(2)}}$：

$$\ell^{(2)} = \ell^{(1)} \cdot e^{m^{(1)} - m^{(2)}} + \tilde{\ell}^{(2)}$$

$$O^{(2)} = O^{(1)} \cdot e^{m^{(1)} - m^{(2)}} + \tilde{O}^{(2)}$$

最终（看完所有块）再除归一：

$$O_{\text{final}} = O^{(N)} / \ell^{(N)}$$

要点：

- 关键洞察是 **$e^{x - m^{(1)}} = e^{x - m^{(2)}} \cdot e^{m^{(2)} - m^{(1)}}$**——max 改了，乘一次指数就修正
- 因为 $m^{(2)} \ge m^{(1)}$，$e^{m^{(1)} - m^{(2)}} \le 1$，所以 rescale 是缩小（数值稳定）
- 整个过程数值上**与"等所有块到齐再算 softmax"完全等价**，不是近似
- backward 时只需保存 $(m, \ell)$（$O(N)$ 空间）而非 $A$（$O(N^2)$ 空间），按需重算 $S$

加分：能说出这个 trick 来自 Milakov & Gimelshein 2018，FlashAttention 的贡献是把它和 attention 的 $@V$ 融合成单一 kernel；能解释为什么不直接除归一而是延后到最后——因为延后后所有 tile 的累加都在同一尺度上，避免反复除法的数值误差。

</details>

**Q3（实战 / 排查）**：你在训练一个 LLaMA 风格的 7B 模型，按理论应该有 FlashAttention 加速，但实测 step time 和 baseline 一样慢。请给出系统的排查清单。

<details>
<summary>Answer sketch</summary>

按"最常见错"到"罕见错"排序：

1. **dtype 检查**：FlashAttention v1/v2 只支持 fp16 / bf16；如果训练在 fp32 下（如忘了开 mixed precision），会静默 fallback 到 math backend。
   - `print(q.dtype)` 应该是 `torch.bfloat16` 或 `torch.float16`

2. **head_dim 检查**：head_dim 必须是 64 / 128 等"标准值"。
   - 计算：`head_dim = hidden_size // n_head`，如果是 96、80、144，可能走 fallback
   - 7B LLaMA 默认 d=4096, h=32, head_dim=128 ✓

3. **序列长度检查**：N < 128 时 Flash 反而慢。
   - 看实际训练 batch 的 seqlen 分布；packing 后应该接近 max_seq_len

4. **SDPA backend 是否真启用**：
   - 用 `torch.backends.cuda.flash_sdp_enabled()` 检查
   - 用 `with sdpa_kernel(SDPBackend.FLASH_ATTENTION):` 强制启用，看是否报错（报错说明环境根本不满足）
   - 终极方案：`nsight-sys` 或 `torch.profiler` 看 kernel 名字是否含 `flash_attn`

5. **flash_attn 库是否装好**：
   - 如果用 HF transformers，配置是否设了 `attn_implementation="flash_attention_2"`
   - `import flash_attn; print(flash_attn.__version__)` 看是否 import 失败
   - 编译 wheel 是否对应你的 CUDA / torch / Python 版本

6. **mask 类型检查**：
   - 自定义 attention bias、ALiBi（v2 前）、padding mask 传成 `attn_mask=` 时 PyTorch 会选 memory-efficient 而非 Flash
   - 改成 `is_causal=True` + 用 sample packing 的 `cu_seqlens` 而非 padding，能确保走 Flash

7. **GPU 架构匹配**：v2 需要 Ampere+，v3 需要 Hopper。在 V100 / T4 / P100 上 v2 直接不可用。

8. **RoPE / 位置编码位置**：q/k 是否在传给 flash_attn 前已经 apply 了 RoPE。

9. **batch / head 维度顺序**：SDPA 要 `(B, H, T, D)`，`flash_attn_func` 要 `(B, T, H, D)`——传错形状会报错或更慢。

10. **dropout**：开了 attention dropout 会略慢（保存 mask 给 backward），可酌情关掉。

加分：能说"先用 `torch.profiler` 出火焰图看哪个 kernel 占了大头"——这是工业级排查方法，比靠经验猜更靠谱。

</details>

---

## 7. 延伸阅读

- [FlashAttention 官方仓库 — Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) — Tri Dao 维护的官方实现，README 列了所有版本的 GPU / dtype / head_dim 支持矩阵；issue 区是排查问题的最好资源
- [Tri Dao 个人主页](https://tridao.me/) — 三篇 FlashAttention 论文的作者，主页有 talk slides 与最新进展
- [PyTorch — `scaled_dot_product_attention` 文档](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) — 官方 backend 选择规则、`SDPBackend` 枚举、is_causal vs attn_mask 的 trade-off
- [PyTorch — `torch.nn.attention.sdpa_kernel` context manager](https://pytorch.org/docs/stable/generated/torch.nn.attention.sdpa_kernel.html) — 如何强制选定 backend（生产代码必须知道）
- [Horace He — Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html) — memory-bound vs compute-bound 的经典科普长文，本节 §1.1 的灵感来源
- [Aleksa Gordić — FlashAttention paper walkthrough (YouTube)](https://www.youtube.com/watch?v=gMOAud7hZg4) — 1 小时把 FlashAttention v1 论文从头讲到尾，可视化做得极好
- 推荐继续读本教程的 **5.4 节《MoE：Mixtral / DeepSeek-MoE 路由与负载均衡》**——把 attention 优化告一段落，进入 FFN 侧的 MoE 革命
- 想真正理解 kernel 实现细节，等读到 **7.6 节《Triton / CUDA kernel 入门 + FlashAttention 走读》**——本节只讲"用"和"原理"，那一节才走读 Triton 实现
