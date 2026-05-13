---
title: "7.6 Triton / CUDA kernel 入门 + FlashAttention 走读"
description: "PyTorch 的\"标准 op 组合\"会启动一堆小 kernel、每次都跑一趟 HBM，吃光带宽；自定义 kernel 用 fusion 把多步合成一个 kernel、把数据按 tile 关在 SRAM 里算完再写出去——这是 LLM 训练里所有 hot path（attention / norm / activation / cross_entropy）都做了的事；本节先讲清\"为什么要写 ker"
---

> ⏱ 预计阅读 60 分钟 ｜ 难度 ★★★ ｜ 前置：5.3 FlashAttention（必读，本节是它的"内部视角"）

## 一句话本节讲什么

PyTorch 的"标准 op 组合"会启动一堆小 kernel、每次都跑一趟 HBM，吃光带宽；自定义 kernel 用 **fusion** 把多步合成一个 kernel、把数据按 tile 关在 SRAM 里算完再写出去——这是 LLM 训练里所有 hot path（attention / norm / activation / cross_entropy）都做了的事；本节先讲清"为什么要写 kernel"和"Triton vs CUDA"，再用 vector add + fused softmax 两个最小例子带过 Triton 的核心抽象（program / load / store / mask），最后用伪代码走读 FlashAttention 的内核结构（tile + online softmax 在代码层怎么落地），并给出现代 LLM kernel 生态地图与"算法工程师该不该自己写 kernel"的边界回答。

---

## 1. Mental model（直觉）

### 1.1 PyTorch 的"组合 op"为什么慢

写 `out = x + y * z` 时，PyTorch 默认行为是：

```
kernel 1: tmp = y * z         读 y, z 写 tmp（HBM ↔ HBM）
kernel 2: out = x + tmp       读 x, tmp 写 out（HBM ↔ HBM）
```

每个 op 是一次 **kernel launch**——CPU 提交一个任务给 GPU、GPU 调度一个 thread block 去跑。两个问题：

1. **HBM I/O 浪费**：`tmp` 在 HBM 上写一次又读一次，但它的物理意义只是"中间结果"——本来根本不需要落到 HBM
2. **kernel launch overhead**：每次 launch 在 A100 上 5-10 μs，对小 tensor 来说计算本身可能只要 2 μs，开销比计算还大

5.3 节算过：A100 的算力 / 带宽比是"每读 1 个 bf16 数要做 ~400 FLOPs 才能不被带宽卡住"。逐 op 跑的代码大部分时候 FLOPs / 数远小于这个门槛，于是 GPU 算力闲着、带宽吃满。

**自定义 kernel 的解法叫 fusion**：把上面两步合成一个 kernel：

```
fused_kernel: out = x + y * z   一次性读 x, y, z 算完写 out
```

`tmp` 这个中间值留在 register / SRAM 里，从不去 HBM。**1 次 launch，3 读 1 写**，I/O 减少一半、launch overhead 减少一半。在 LLM 这种"大 tensor + 一连串 elementwise / reduction op"的场景下，fusion 经常带来 2-5× 加速。

### 1.2 LLM 训练里哪些地方需要自定义 kernel

LLM 的 forward + backward 看起来花样很多，但 90% 时间集中在少数几个 hot path：

| Hot path | 标准 PyTorch 拆几个 kernel | 已有 fused 实现 |
|---|---|---|
| attention（QKV + softmax + AV） | 5-10 个 | **FlashAttention** |
| RMSNorm（mean / var / scale） | 3-5 个 | Liger / Apex / Triton |
| SwiGLU（silu + gate * up） | 3 个 | Liger |
| cross_entropy（log_softmax + nll） | 4-6 个，且要实例化 logits | **Liger fused CE**（省 vocab 维显存 60%） |
| RoPE（cos/sin × q/k 拆维拼回） | 4-8 个 | flash_attn / Liger |
| AdamW（指数滑动 + 更新） | 6-8 个 | Apex fused / `torch._foreach` |
| MoE all-to-all | 通信 + scatter / gather 一堆 | DeepEP（DeepSeek） |

**90% 的训练时间在 5% 的 op 上**——只要把这 5% 的 op 用 fused kernel 替掉，整个训练就能快一个数量级。这就是为什么 LLM infra 要专门有"kernel 工程"这条线。

### 1.3 CUDA vs Triton：两条路

**CUDA**（NVIDIA, 2007）：

- C++ 风格 DSL，最贴近硬件
- 程序员要手动管：线程块 / warp / shared memory 分配、内存合并访问（coalescing）、bank conflict、register pressure
- 性能上限最高（cuBLAS / cuDNN / TensorRT 都是 CUDA 写的）
- 开发慢——一个 fused softmax 写一周不稀奇，调性能再两周
- 学习曲线陡，工业界懂的人少

**Triton**（OpenAI, 2019）：

- Python-style DSL，函数加 `@triton.jit` 装饰器后被编译到 PTX
- **block 级编程**：你写"一个 thread block 处理一个 tile"，**不直接管 thread**
- 编译器自动处理：内存合并、shared memory 分配、bank conflict 规避、向量化
- 性能：成熟的 Triton kernel 通常达到手写 CUDA 的 **80-95%**
- 开发速度：5-10× 于 CUDA，一个工程师一天能写出可用的 kernel

**结论**：

- 极致性能（cuBLAS / cuDNN / TensorRT）：NVIDIA 自己人用 CUDA + PTX inline，普通人摸不到也不需要碰
- 学术 prototype + 工业 fused kernel：**Triton**，是 LLM 工程师的事实首选
- 现实中，FlashAttention v1 是 CUDA、v2 仍是 CUDA + 部分 Triton 参考实现、社区版的 attention 变体（sliding window、ALiBi、custom mask）几乎全是 Triton

**心智模型一句话**：CUDA 是汇编、Triton 是 C；除非你在 NVIDIA 内部或写 cuBLAS-killer，否则学 Triton 就够。

### 1.4 Triton 的核心抽象（必须先建立）

Triton 把"程序"切成两层：

```
        ┌─────────────────────────────────────────┐
        │  Grid（一堆 program）                   │
        │  ┌────┐ ┌────┐ ┌────┐ ┌────┐           │
        │  │ P0 │ │ P1 │ │ P2 │ │ P3 │ ...       │
        │  └────┘ └────┘ └────┘ └────┘           │
        │   每个 program = 一个 thread block      │
        │   每个 program 处理 1 个 tile           │
        └─────────────────────────────────────────┘
                      │
                每个 program 内部
                      ▼
        ┌─────────────────────────────────────────┐
        │  block-level operations                 │
        │  tl.load / tl.store: 与 HBM 交互        │
        │  tl.sum / tl.max / tl.exp: block 内并行 │
        │  + - * / : elementwise on tile          │
        │  mask: 处理边界（最后一个 tile 不满）    │
        └─────────────────────────────────────────┘
```

关键名词：

- **Program** = 一个 thread block，处理 1 个 tile（比如 1024 个元素或 1 行 softmax）
- **`tl.program_id(axis=0)`** = 当前 program 在 grid 里的索引（自己是第几个）
- **`tl.arange(0, BLOCK_SIZE)`** = block 内的位置向量 `[0, 1, ..., BLOCK_SIZE-1]`
- **`tl.load(ptr, mask=...)`** = 从 HBM 读一个 tile 的数据（mask 用来处理"最后一块不满"的边界）
- **`tl.store(ptr, value, mask=...)`** = 把 tile 的结果写回 HBM
- **Block-level op**：`tl.sum(x, axis=0)` 在 tile 内做 reduction，编译器自动调度 thread 协作

你**不写循环**遍历 tile 内每个元素——你写"把这一整个 tile 当成一个向量来操作"，编译器把它展开成多个 thread 并行算。

---

## 2. 公式与原理

本节没有"公式"——Triton kernel 的"原理"就是**手把手把 tensor 切 tile，把循环结构和访问模式表达清楚**。三个最重要的设计决策：

1. **怎么切 tile**：BLOCK_SIZE 选多大？要让 (输入 tile + 中间结果 + 输出 tile) 装得进 SRAM（~100-200 KB / SM）。
   - elementwise op：常用 1024 / 2048
   - softmax / norm（按行 reduction）：BLOCK_SIZE = 下一个 ≥ N 的 2 的幂，N = vocab/hidden_dim
   - matmul / attention：BR × BC，常用 64×64、128×64、128×128

2. **grid 怎么算**：grid = ceil(总元素数 / BLOCK_SIZE)。Triton 用 `triton.cdiv(n, BLOCK_SIZE)`（ceiling division）。

3. **mask 怎么处理边界**：当 N 不能被 BLOCK_SIZE 整除时，最后一个 program 处理的 tile 是不满的——`offsets < N` 这个 mask 告诉 `tl.load / tl.store` "超出的位置不要读 / 不要写"。

下文用两个最小例子把这些落到代码上。

---

## 3. 最小代码示例

### 3.1 第一个 Triton kernel：vector add（必看，每行带注释）

```python
import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr, y_ptr, out_ptr,             # 三个 tensor 的 HBM 起始地址
    n,                                 # 元素总数
    BLOCK_SIZE: tl.constexpr,          # 编译期常量（生成不同 BLOCK_SIZE 的特化版本）
):
    pid = tl.program_id(axis=0)        # 当前 program 是第几个（0, 1, 2, ...）
    block_start = pid * BLOCK_SIZE     # 这个 program 负责的 tile 起点
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # tile 内每个元素的全局下标
    mask = offsets < n                 # 最后一个 tile 可能不满，超出的位置 mask 掉
    x = tl.load(x_ptr + offsets, mask=mask)  # 一次读 BLOCK_SIZE 个 x
    y = tl.load(y_ptr + offsets, mask=mask)  # 一次读 BLOCK_SIZE 个 y
    tl.store(out_ptr + offsets, x + y, mask=mask)  # 写回（mask 外的位置不写）


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and y.is_cuda and x.shape == y.shape
    out = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)  # ceil(n/BLOCK_SIZE) 个 program
    add_kernel[grid](x, y, out, n, BLOCK_SIZE=1024)
    return out


if __name__ == "__main__":
    a = torch.randn(1_000_003, device="cuda")  # 故意非整除，验证 mask
    b = torch.randn(1_000_003, device="cuda")
    out = add(a, b)
    torch.testing.assert_close(out, a + b)
    print("ok")
```

**关键解读**：

- `@triton.jit`：标记这是 Triton kernel，第一次 call 会被 JIT 编译到 PTX，缓存供后续 call 复用
- `BLOCK_SIZE: tl.constexpr`：编译期常量。改 BLOCK_SIZE 会生成新的 specialized binary，但同 BLOCK_SIZE 的多次调用走同一个 binary
- `add_kernel[grid](...)`：方括号传 grid（program 数量）、圆括号传 kernel 参数。这是 Triton 的"meta-programming"语法
- 整个 kernel **没有显式循环**——`tl.load / tl.store` 一次操作 BLOCK_SIZE 个元素，编译器把它编译成多个 thread 并行

`add` 这个例子的性能其实比不上 PyTorch 的 `+`（PyTorch 的 elementwise 已经被高度优化），意义在于建立 Triton 的"五行模板"：**program_id → offsets → mask → load → compute & store**。所有 elementwise / 归约 kernel 都是这个骨架。

### 3.2 第二个 Triton kernel：fused softmax（一行 softmax 一个 program）

标准 PyTorch 的 `softmax(x, dim=-1)` 在底层至少 3 个 kernel：

1. `max = x.max(dim=-1)` —— 读 x 写 max
2. `e = (x - max).exp()` —— 读 x, max 写 e
3. `out = e / e.sum(dim=-1)` —— 读 e 写 out

每个中间 tensor 都过一遍 HBM。fused 版本只用 1 个 kernel：

```python
import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    out_ptr, x_ptr, n_cols,
    stride_row,                        # 一行有多少个元素的字节步长
    BLOCK_SIZE: tl.constexpr,          # ≥ n_cols 的下一个 2 的幂
):
    row = tl.program_id(0)             # 一个 program 处理一行
    row_start = x_ptr + row * stride_row
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    # 1) load 整行进 SRAM（mask 外用 -inf 占位，不影响 max / exp）
    x = tl.load(row_start + cols, mask=mask, other=-float("inf"))
    # 2) 数值稳定 softmax：减 row max
    x = x - tl.max(x, axis=0)
    # 3) exp + 归一（全部在 SRAM 里）
    num = tl.exp(x)
    out = num / tl.sum(num, axis=0)
    # 4) 一次写回 HBM
    tl.store(out_ptr + row * stride_row + cols, out, mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.ndim == 2
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)   # Triton 要求 2 的幂
    out = torch.empty_like(x)
    softmax_kernel[(n_rows,)](out, x, n_cols, x.stride(0), BLOCK_SIZE=BLOCK_SIZE)
    return out
```

**关键解读**：

- 一个 program 处理**一整行**，整行装进 SRAM 后做 max / exp / sum / div 全部在内部完成——**中间值从不写回 HBM**
- `tl.load(..., other=-float("inf"))`：mask 外的位置用 `-inf` 填，让 `tl.max` 不会被未初始化数据污染
- `tl.max(x, axis=0)` / `tl.sum`：block 级并行 reduction，编译器自动用 warp shuffle 加速
- 在 hidden_dim ≤ 16384 时（绝大多数 LLM 的 vocab / FFN dim 都满足），整行装得进 SRAM，性能 **2-3× 于 PyTorch**

**核心教训**：fusion 的本质是"中间结果留在 SRAM，不去 HBM"。能做到这一点的算子都值得 fuse——softmax / norm / SwiGLU / RoPE 全都符合。

### 3.3 FlashAttention 内核结构走读（伪代码，体现 tile + online softmax）

5.3 节给了 FlashAttention 的算法伪代码，本节给"接近真实 Triton kernel 写法"的伪代码，让你看到 5.3 的算法怎么落到 Triton 抽象上：

```python
@triton.jit
def flash_attn_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr,            # tensors（L 存 logsumexp 给 backward）
    sm_scale,                                      # 1/sqrt(d_k)
    N, d,                                          # seq_len, head_dim
    BR: tl.constexpr, BC: tl.constexpr,            # tile 大小：Q tile 高 BR、K/V tile 高 BC
):
    # ---- v2 顺序：Q 在外、K/V 在内（每个 program 处理一个 Q tile）----
    q_block_id = tl.program_id(0)                  # 第几个 Q tile
    head_id    = tl.program_id(1)                  # 第几个 head（grid 第二维）

    # 1) 把这个 Q tile 一次性 load 到 SRAM——之后整个 inner loop 不再读
    q_offsets = q_block_id * BR + tl.arange(0, BR)
    Q = tl.load(Q_ptr + q_offsets[:, None] * d + tl.arange(0, d)[None, :])

    # 2) 初始化 running state（在 SRAM / register 里，从不写 HBM）
    m = tl.full((BR,), -float("inf"), dtype=tl.float32)   # running max
    l = tl.zeros((BR,), dtype=tl.float32)                  # running sum
    O = tl.zeros((BR, d), dtype=tl.float32)                # running output

    # 3) Inner loop：流式遍历所有 K/V tile
    for kv_block_id in range(0, tl.cdiv(N, BC)):
        kv_offsets = kv_block_id * BC + tl.arange(0, BC)
        K = tl.load(K_ptr + kv_offsets[:, None] * d + tl.arange(0, d)[None, :])
        V = tl.load(V_ptr + kv_offsets[:, None] * d + tl.arange(0, d)[None, :])

        # 3a) tile 内算 attention score：matmul 在 SRAM 完成
        S = tl.dot(Q, tl.trans(K)) * sm_scale       # (BR, BC)
        # 3b) causal mask（本 tile 部分）—— 略
        # 3c) online softmax 更新（5.3 节公式 §2.1）
        m_new = tl.maximum(m, tl.max(S, axis=1))
        alpha = tl.exp(m - m_new)                    # 旧 state 的 rescale 因子
        P = tl.exp(S - m_new[:, None])               # 当前 tile 的 unnormalized softmax
        l = l * alpha + tl.sum(P, axis=1)
        O = O * alpha[:, None] + tl.dot(P, V)        # accumulate
        m = m_new

    # 4) 最终归一 + 写回 HBM（只写最终 O 和 logsumexp，不写 attention matrix）
    O = O / l[:, None]
    tl.store(O_ptr + q_offsets[:, None] * d + tl.arange(0, d)[None, :], O)
    tl.store(L_ptr + q_offsets, m + tl.log(l))      # backward 重算用
```

**对照 5.3 节看的几个关键点**：

- **outer = Q tile**（v2 选择）：每个 program 一直占着一个 Q tile，inner loop 流式扫所有 K/V——Q 只读一次，K/V 各读一遍，总 HBM 读量 $O(N^2 d / B_R)$
- **`tl.dot(Q, tl.trans(K))`**：Triton 的 matmul，编译器自动用 Tensor Core
- **`alpha = tl.exp(m - m_new)`** + `O = O * alpha[:,None] + tl.dot(P, V)`：online softmax 的 rescale + accumulate，**S 矩阵从不写 HBM**
- **写回的只有 O 和 L**：L = log(sum) + max，logsumexp 这一个标量给 backward 用，attention matrix 不存——backward 时按需重算 S

**真实 FlashAttention 比这复杂在哪**：

- causal mask 要做 tile-level skip（K tile 完全在 Q tile 之后就不用算）
- BR / BC 要 autotune（不同 GPU、不同 d、不同 N 最优值不同）
- backward 比 forward 复杂 3-5 倍：要分别对 Q / K / V 算 grad，要重新 load 一遍 Q/K/V 和 O/L 重算 S，且要做 atomic add（多个 Q tile 的 grad 都加到同一份 K/V grad 上）
- 寄存器压力管理：BR=128, BC=128, d=128 时 Q+K+V+S+O 在一个 thread block 的 register 里逼近上限，要小心 spill 到 local memory

**这就是为什么自己写 attention 变体的 backward 是 PhD 工作量**——5.3 节最后那句话现在有体感了。

### 3.4 找 hot path：torch.profiler 用法（必会）

写 kernel 前先 profile，找真正的 bottleneck。错误顺序：拍脑袋觉得"这块慢"就去写 Triton，结果优化的 op 只占 2% wall clock。

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    for _ in range(5):                              # warmup
        out = model(x)
    with record_function("hot_step"):               # 给自己关心的 region 命名
        for _ in range(20):
            out = model(x)
            loss = out.mean(); loss.backward()
    torch.cuda.synchronize()

# 按 GPU 时间排序，看哪个 kernel 吃得最多
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
prof.export_chrome_trace("trace.json")              # 拖到 chrome://tracing 看火焰图
```

**经验排序**（按命中率）：

1. attention（如果没开 Flash）—— 用 `sdpa_kernel(FLASH_ATTENTION)` 解决
2. cross_entropy（vocab 大时实例化 logits 撑爆）—— 用 Liger fused CE
3. RMSNorm / SwiGLU—— 用 Liger / Apex fused
4. AdamW—— 用 `torch.optim.AdamW(fused=True)` 或 `torch._foreach_*`
5. RoPE—— 用 flash_attn / Liger 的 fused 版

profile 显示这五项已经是 fused 后还嫌慢，再考虑写自己的 Triton kernel。

---

## 4. 工程踩坑与经验

- ❗ **Triton kernel 第一次 call 要编译，慢；warmup 后才能 benchmark**。`@triton.jit` 是 JIT，第一次会花几百毫秒到几秒编译 PTX 并缓存。新手 benchmark 时把第一次 call 算进去，得出"Triton 比 PyTorch 慢 100×"的离谱结论。**正确做法**：先跑 5-10 次 warmup，再用 `triton.testing.do_bench` 或 `torch.cuda.synchronize` + `time.perf_counter` 量稳态时间。

- ❗ **`BLOCK_SIZE` 必须是 2 的幂，且要与 GPU SM / shared memory 大小匹配**。Triton 内部用了大量"tile size = power of 2"的优化路径，传 1000、3000 这种非 2 幂会报错或走慢路径。常用值：elementwise 1024/2048、softmax/norm 用 `triton.next_power_of_2(n_cols)`、matmul 64/128/256。**经验**：BR×BC×4字节 + 输入 tile + 输出 tile 不要超过 128 KB（A100 单 SM 的 SRAM 上限），否则 occupancy 暴跌。

- ❗ **mask 错位 → silent corruption（不报错但结果错）**。最常见的两种：(1) 忘了 mask，最后一个不满的 tile 读了"越界垃圾"参与计算；(2) mask 写错方向（`offsets > n` 而不是 `offsets < n`），等价于全部 mask 掉，输出全 0 但不报错。**强制做法**：写完 kernel 第一件事是用一个故意非整除的 size（如 1_000_003）跑 `torch.testing.assert_close(triton_out, pytorch_out)`，对比通过才算写完。

- ❗ **自己写的 Triton kernel 性能很难超过现成的（FlashAttention / Liger / cuBLAS）**。这些 kernel 是顶级工程师调了几个月、跑了几千组 autotune 参数搜出来的。**造轮子前先 profile 找真 bottleneck，再确认这个 op 没有现成 fused 实现**。常见误区：花一周写了个 fused RMSNorm 比 Liger 慢 30%，时间还不如花在改进数据 pipeline 上。

- ❗ **Triton kernel 不是 PyTorch op，不参与 autograd**。Triton kernel 默认只算 forward，**没有 backward**。要参与训练必须用 `torch.autograd.Function` 包一层、自己实现 backward kernel。所以"用 Triton 写一个 fused softmax 替换 PyTorch 的 softmax"如果只写了 forward，训练阶段会直接报"no grad fn"。生产里的 fused kernel 都是 forward + backward 一对。

- ❗ **多 GPU 场景下 Triton kernel 不会自动 sync / 不会自动 dispatch 到正确 device**。`add_kernel[grid](x, y, out, ...)` 默认在 `x.device` 上跑，但 NCCL collective、stream sync、graph capture 等高阶用法（DDP / FSDP 内部用的）需要 kernel 实现注意 `torch.cuda.current_stream()` / `current_device()`。**经验**：多卡训练时 Triton kernel 出现"非确定性结果"，先怀疑 stream 没对齐，用 `torch.cuda.synchronize()` 在前后强制同步定位。

- ❗ **FlashAttention 的 backward kernel 比 forward 复杂得多**。forward 一个 program 一行 Q tile 单调流式累加 K/V；backward 要对 Q、K、V 三个 tensor 算 grad，K/V 的 grad 是"每个 Q tile 都贡献一份"，必须用 atomic add 或者两个 pass（pass 1 算 dQ、pass 2 算 dK/dV）才能并发安全。**结论**：不要试图自己写 attention 变体的 backward，难度是 forward 的 3-5 倍；要做变体优先复用 flash_attn 的 `flash_attn_with_kvcache`、`flash_attn_varlen_func` 等 API 拼。

- ❗ **Triton 不支持所有 PyTorch op，部分要 fallback**。例如 `tl.complex`（复数）、稀疏运算、bf32 / tf32 显式控制、自定义 dtype 都没原生支持；某些 reduction（如 cumsum 在老版本不稳）会报错。遇到这些必须 fallback 到 PyTorch 在 kernel 外做，或者升级到最新 Triton（每个 minor 版本都在加 op）。**版本管理**：固定 Triton 版本到 lockfile，不要让 CI 每次 install latest——Triton 0.x → 2.x → 3.x 的 API 经常变。

- ❗ **Triton autotune 的 cache 失效会导致首跑变慢**。`@triton.autotune` 会跑多组配置选最快的，结果存到磁盘 cache。代码改 BLOCK_SIZE / num_warps / num_stages 任何一个，cache 失效要重跑——这一次 call 可能花几分钟。生产部署时要确保 autotune cache 跟着模型一起发布，否则第一次推理慢得离谱。

- ❗ **Triton kernel 不是任何场景都比 PyTorch 快**。小 tensor（< 几 KB）launch overhead 主导，PyTorch 的 fused elementwise（如 `torch._foreach_add`）反而更快；非 hot path（每个 step 只跑 1 次的 op）即使快 10× 也只省毫秒不值得。**判断标准**：profile 显示这个 op > 5% wall clock 才值得 fuse。

---

## 5. 经典 paper

- **Tillet, Kung, Cox, 2019 — Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations** — Triton 原典（OpenAI 当时还在 OpenAI Research）。必读 §3 "Triton-IR" 关于 tile 抽象的设计、§4 编译器 pass。这篇展示了为什么 "block-level programming" 是比 thread-level 更适合 DL kernel 的抽象——因为 DL op 的并行天然是 block 粒度（batch / head / row）。
- **Dao, Fu, Ermon, Rudra, Ré, 2022 — FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness** — 5.3 节深入读过；本节再读一遍专门看 §3.1 algorithm 1 和官方仓库 `flash_attn/flash_attn_triton.py` 的对应——你会看到本节 §3.3 伪代码与真实 Triton 实现一一对应，加上 causal mask、autotune、backward 的工业级处理。
- **Dao, 2023 — FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning** — 必读 §3.1 关于"为什么 v2 把 outer loop 从 K/V 换成 Q"的工程论证——这是 §3.3 走读里"v2 顺序"那一句的依据，也是 Q3 自测题的考点。

---

## 6. 自测与面试题

**Q1（概念）**：Triton 与 CUDA 的关系是什么？为什么 LLM 工程师更应该学 Triton 而不是 CUDA？

<details>
<summary>Answer sketch</summary>

关系：

- CUDA 是 NVIDIA 的 C++ 风格底层并行编程语言（thread-level），程序员手动管 thread / warp / shared memory / register
- Triton 是 OpenAI 2019 提出的 Python-style DSL（block-level），`@triton.jit` 编译到 PTX，与 CUDA 编译产物在同一层
- Triton 编译器自动处理：内存合并访问、shared memory 分配、bank conflict、向量化、thread block 调度
- 性能：成熟 Triton 通常达到等价 CUDA 的 80-95%

为什么 LLM 工程师选 Triton：

- 开发速度 5-10× 于 CUDA：写 fused softmax / RMSNorm 一天写完，CUDA 要一周
- 抽象层级匹配 LLM 需求：DL op 的并行天然是 block 粒度（一行 softmax / 一个 attention tile），与 Triton 的 block 模型完美对齐
- 工业界 fused kernel（FlashAttention 部分参考实现、Liger Kernel、Mamba 实现等）都是 Triton——要看懂别人代码、要 fork 改一改，必须懂 Triton
- 极致性能（cuBLAS / cuDNN / TensorRT）由 NVIDIA 维护，普通工程师碰不到也不需要碰

学 CUDA 的场景：在 NVIDIA / AMD 内部写库、做学术 GPU paper、调极致性能（最后 5%）。绝大多数 LLM 算法工程师 / infra 工程师只需要学 Triton。

加分：能说"Triton 是 PTX 之上的高层 DSL，CUDA 也是编译到 PTX；两者是同级别的输入语言而不是上下层关系"。

</details>

**Q2（实现）**：写出 Triton vector add kernel 的核心 5 行：program_id 算 offset、mask 处理边界、load / store 加 mask。

<details>
<summary>Answer sketch</summary>

```python
pid = tl.program_id(axis=0)
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
mask = offsets < n
x = tl.load(x_ptr + offsets, mask=mask)
y = tl.load(y_ptr + offsets, mask=mask)
tl.store(out_ptr + offsets, x + y, mask=mask)
```

要点：

- **`pid = tl.program_id(0)`**：当前 program 在 grid 的位置；grid 第一维是 program 数 = `cdiv(n, BLOCK_SIZE)`
- **`offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)`**：tile 内每个元素对应的全局下标（一个长度 BLOCK_SIZE 的向量）
- **`mask = offsets < n`**：最后一个 program 处理的 tile 可能不满，超出的位置 mask 掉，避免越界读写
- **`tl.load(ptr + offsets, mask=mask)`**：一次读 BLOCK_SIZE 个元素，mask 外位置返回 0（或 `other=` 指定的值）
- **`tl.store(ptr + offsets, value, mask=mask)`**：一次写 BLOCK_SIZE 个元素，mask 外位置不写

加分：能说出"BLOCK_SIZE 必须 2 的幂、grid 用 `lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)` 让 BLOCK_SIZE 可调"；能解释为什么没有显式循环——因为 `tl.load / tl.arange` 是 block-level op，编译器自动展开成多 thread 并行。

</details>

**Q3（前沿）**：FlashAttention v2 把 outer loop 从 K/V 换成 Q（每个 thread block 处理一个 Q tile，inner loop 流式扫所有 K/V），为什么这样比 v1 快？

<details>
<summary>Answer sketch</summary>

v1 的 outer loop 是 K/V tile、inner loop 是 Q tile。意味着：每个 K/V tile 对所有 Q tile 算一遍，每个 Q tile 在不同 K/V tile 之间被反复 load / 反复维护 running state $(m, \ell, O)$ 写回 HBM。

v2 反过来：outer = Q tile，每个 thread block 一直占着一个 Q tile 直到与所有 K/V tile 算完。带来的好处：

1. **Q 只读一次**：Q tile 装进 SRAM 后整个 inner loop 不再读 HBM（v1 里 Q 被读 Tc 次）。HBM 读量从 $O(N^2 d / B_C)$ 降到 $O(N^2 d / B_R)$ + Q 的一次性读
2. **running state 不写 HBM**：$(m, \ell, O)$ 全程留在 register / SRAM，inner loop 完成后才一次性写回最终 O 和 L。v1 里每个 inner step 都要把 $(m, \ell, O)$ 写 HBM 给下个 K/V iteration 用
3. **rescale 次数减少**：每个 Q tile 只在最后归一化一次（除以 $\ell$），v1 里每个 inner step 都要 rescale O。非 matmul 操作（exp / 除法）减少 → Tensor Core 利用率提高
4. **更适合并行**：每个 Q tile 完全独立，可以分给不同 thread block / 不同 SM 并发；v1 的 K/V outer loop 之间共享 running state，并行度低

**结论**：v2 的核心收益是"减少 HBM 读 + 减少非 matmul 操作"，A100 上从 30% 理论 FLOPs 提升到 50-70%，对长 context 提速 1.5-2×。

为什么 v1 当初选 K/V 在外？v1 把 K/V 放外是因为 backward 的天然结构（dK/dV 是"每个 Q tile 都贡献一份"，K/V 在外便于 atomic accumulate）。v2 的洞察是：forward 和 backward 的最优 loop 顺序不同，应该分开优化——forward 用 Q-outer，backward 仍 K/V-outer。

加分：能提"v3 在 H100 上更进一步用 warp specialization（producer warp 搬数据 / consumer warp 算 matmul），把 TMA 与 WGMMA 流水化"，这是硬件特性驱动的下一层优化（5.3 §2.5）。

</details>

---

## 7. 延伸阅读

- [Triton 官方教程](https://triton-lang.org/main/getting-started/tutorials/) — vector add → softmax → matmul → fused attention 一路升级，本节 §3 的两个例子来自这里。读完前 4 个 tutorial 基本可以独立写工业级 fused kernel
- [Dao-AILab/flash-attention 官方仓库](https://github.com/Dao-AILab/flash-attention) — `flash_attn/flash_attn_triton.py` 是 FlashAttention 的 Triton 参考实现（C++ / CUDA 版在 `csrc/`），对照本节 §3.3 伪代码看，理解工业级实现的复杂度
- [linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel) — HuggingFace Trainer 已默认开启的 fused kernel 集合（RMSNorm / SwiGLU / RoPE / cross_entropy / fused_linear_cross_entropy），全部 Triton。是"算法工程师不写 kernel 但要会用"的标杆——README 给了每个 kernel 的提速 / 省显存数字，照着开就有 10-30% 训练加速
- [DeepSeek-AI/DeepEP](https://github.com/deepseek-ai/DeepEP) — DeepSeek 开源的 MoE all-to-all 通信 kernel（Hopper TMA + 节点内 NVLink / 节点间 IB），是 5.4 MoE 章的 infra 配套，也是国内 LLM kernel 工程的代表作
- [Horace He — Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html) — memory-bound vs compute-bound 的经典科普，对"为什么要 fuse"建立第一性认知
- [Sasha Rush — GPU Puzzles](https://github.com/srush/GPU-Puzzles) — 14 道交互式题目从 vector add 到 matmul，用 Numba 风格但思路与 Triton 完全一致，是建立 kernel 直觉的最快路径
- 推荐继续读本教程的 **8.1 节《SFT 数据构造》**——训练 infra 章告一段落，进入后训练数据工程
- 想看 kernel 在真实 LLM 训练里怎么集成，回头读 **5.3 节《FlashAttention 1/2/3》**——本节是"内核视角"，5.3 是"用法视角"，两者互补
