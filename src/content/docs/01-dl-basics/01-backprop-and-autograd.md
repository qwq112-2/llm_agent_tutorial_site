---
title: "1.1 神经网络与反向传播（手推 + autograd 心智模型）"
description: "把神经网络看成一张 DAG，前向是拓扑序求值、反向是链式法则反向走一遍——理解这一点，autograd、梯度累加、`retain_graph`、梯度爆炸/消失就都不再神秘，后面 Module 4 的 Transformer 训练、Module 8 的 LoRA、Module 9 的 PPO 全建立在它之上。"
---

> ⏱ 预计阅读 35 分钟 ｜ 难度 ★ ｜ 前置：无

## 一句话本节讲什么

把神经网络看成一张 DAG，前向是拓扑序求值、反向是链式法则反向走一遍——理解这一点，autograd、梯度累加、`retain_graph`、梯度爆炸/消失就都不再神秘，后面 Module 4 的 Transformer 训练、Module 8 的 LoRA、Module 9 的 PPO 全建立在它之上。

---

## 1. Mental model（直觉）

神经网络训练的全部数学，本质上只有两件事：**前向算 loss**、**反向算梯度**。

把网络看成一张**有向无环图（DAG）**：

- **节点 = 张量**（输入 / 权重 / 中间激活 / 输出 / loss 标量）
- **边 = 算子**（matmul / add / ReLU / softmax / ...）

**前向传播**就是对这张图做拓扑排序，从输入和参数出发，按依赖顺序逐节点求值，最后算出一个标量 loss $L$。**反向传播**就是从 $L$ 出发，反向沿着同一张图走一遍，每经过一个算子就用**链式法则**把上游传下来的梯度乘上该算子的局部 Jacobian，得到对每个节点的梯度。

```
       前向 (forward)  ───────────────────────►
   x ──► [W1, +b1] ──► z1 ──► ReLU ──► h1 ──► [W2, +b2] ──► y_hat ──► MSE ──► L
                                                                              │
   ◄──────────────────────────── 反向 (backward) ──────────────────────────────┘
   每条边反向走时：上游 grad × 本算子局部 Jacobian
```

记住三个心智锚点，本节后面所有内容都围绕它们：

1. **图是动态的**——PyTorch 是 define-by-run，每次 forward 都重新构图（这是为什么控制流、动态 shape、`if/while` 都能直接用 Python 写）
2. **梯度是"沿图反向流动的链式法则"**——不是某种黑魔法，每一步都能在草稿纸上手推
3. **autograd 帮你做的只有两件事**：① 在 forward 时记录每个 tensor 的"父节点 + 算子"，② backward 时按反图走一遍。其它你以为它做的（调度、显存、混合精度）都是别的子系统

工程上，所有"我的 loss 为什么不下降"的 debug 90% 落到这张图上：要么图断了（`detach` / `requires_grad=False` / inplace 把链路打掉），要么图错了（`zero_grad` 忘了调，梯度变 2 倍 3 倍累加），要么图爆了（连乘后数值溢出）。

---

## 2. 公式与原理

### 2.1 链式法则与 Jacobian 形状

设标量 loss $L = f(g(h(x)))$，链式法则给出：

$$
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial f} \cdot \frac{\partial f}{\partial g} \cdot \frac{\partial g}{\partial h} \cdot \frac{\partial h}{\partial x}
$$

在神经网络里，每个中间变量都是张量。设输入 $x \in \mathbb{R}^n$、中间 $u \in \mathbb{R}^m$、loss $L \in \mathbb{R}$，那么：

- $\frac{\partial L}{\partial u} \in \mathbb{R}^m$（标量对向量的梯度，与变量同形）
- $\frac{\partial u}{\partial x} \in \mathbb{R}^{m \times n}$（向量对向量的 Jacobian）
- 链式法则给出 $\frac{\partial L}{\partial x} = \left(\frac{\partial u}{\partial x}\right)^\top \frac{\partial L}{\partial u} \in \mathbb{R}^n$

**关键观察**：autograd 实际上**从不显式构造 Jacobian 矩阵**——它只算 **vector-Jacobian product (VJP)**：给定上游梯度 $\bar{u} = \frac{\partial L}{\partial u}$，每个算子提供一个函数 `vjp(u_bar) -> x_bar`，避免存储 $m \times n$ 的大矩阵。这是 reverse-mode AD 比 forward-mode AD 在深度学习里更经济的根本原因（输出维度是 1，输入维度是百万级）。

### 2.2 手推 2 层 MLP

模型：$\hat{y} = W_2 \, \mathrm{ReLU}(W_1 x + b_1) + b_2$，loss 用 MSE。

明确所有维度（B 是 batch size）：

- $x \in \mathbb{R}^{B \times d_\text{in}}$
- $W_1 \in \mathbb{R}^{d_\text{in} \times d_h}$，$b_1 \in \mathbb{R}^{d_h}$
- $z_1 = x W_1 + b_1 \in \mathbb{R}^{B \times d_h}$
- $h_1 = \mathrm{ReLU}(z_1) \in \mathbb{R}^{B \times d_h}$
- $W_2 \in \mathbb{R}^{d_h \times d_\text{out}}$，$b_2 \in \mathbb{R}^{d_\text{out}}$
- $\hat{y} = h_1 W_2 + b_2 \in \mathbb{R}^{B \times d_\text{out}}$
- $L = \frac{1}{B} \sum_{i,j} (\hat{y}_{ij} - y_{ij})^2 \in \mathbb{R}$

**Forward** 一遍把 $z_1, h_1, \hat{y}$ 都缓存下来——这就是为什么训练比推理多占 2-5 倍显存（要存 activation 给 backward 用，Module 7.5 的 activation recomputation 就是来省这部分的）。

**Backward** 一步步推（每一步都给出形状）：

第一步：$\bar{\hat{y}} = \frac{\partial L}{\partial \hat{y}} = \frac{2}{B}(\hat{y} - y) \in \mathbb{R}^{B \times d_\text{out}}$

第二步：$\hat{y} = h_1 W_2 + b_2$，对 $W_2$、$b_2$、$h_1$ 求梯度——

$$
\frac{\partial L}{\partial W_2} = h_1^\top \, \bar{\hat{y}} \in \mathbb{R}^{d_h \times d_\text{out}}
$$

$$
\frac{\partial L}{\partial b_2} = \sum_{i=1}^{B} \bar{\hat{y}}_{i,:} \in \mathbb{R}^{d_\text{out}}
$$

$$
\bar{h_1} = \bar{\hat{y}} \, W_2^\top \in \mathbb{R}^{B \times d_h}
$$

> 形状检查小技巧：$W_2$ 形状是 $d_h \times d_\text{out}$，所以 $\frac{\partial L}{\partial W_2}$ 也必须是 $d_h \times d_\text{out}$；左乘的因子必须给出 $d_h$ 维（$h_1^\top$），右乘的必须给出 $d_\text{out}$ 维（$\bar{\hat{y}}$）。**形状对了，公式基本就对了**——这是手推矩阵微分的最快验算方法。

第三步：穿过 ReLU。$h_1 = \max(0, z_1)$ 的导数是 $\mathbb{1}[z_1 > 0]$，所以：

$$
\bar{z_1} = \bar{h_1} \odot \mathbb{1}[z_1 > 0] \in \mathbb{R}^{B \times d_h}
$$

（$\odot$ 是 element-wise 乘，$\mathbb{1}[\cdot]$ 是指示函数，$z_1 = 0$ 处 PyTorch 约定导数为 0）

第四步：$z_1 = x W_1 + b_1$，对称地：

$$
\frac{\partial L}{\partial W_1} = x^\top \, \bar{z_1} \in \mathbb{R}^{d_\text{in} \times d_h}
$$

$$
\frac{\partial L}{\partial b_1} = \sum_{i=1}^{B} \bar{z_1}_{i,:} \in \mathbb{R}^{d_h}
$$

至此 4 个参数梯度全部得到。完整流程在草稿纸上不超过 10 行，是后面所有"理解 LoRA 的 A B 矩阵梯度"、"理解 PPO 里 logp 对 logits 的梯度"的基础——一定要亲手推一遍。

### 2.3 数值梯度 vs 解析梯度（gradcheck）

手推完之后，怎么知道自己没推错？用**中心差分**做数值验证：

$$
\frac{\partial L}{\partial \theta_i} \approx \frac{L(\theta + \epsilon e_i) - L(\theta - \epsilon e_i)}{2\epsilon}
$$

其中 $e_i$ 是第 $i$ 个标准基向量、$\epsilon$ 通常取 $10^{-5}$ 到 $10^{-7}$。中心差分比单边差分 $\frac{f(x+\epsilon)-f(x)}{\epsilon}$ 精度高一阶（$O(\epsilon^2)$ vs $O(\epsilon)$）。

判定标准用**相对误差**而不是绝对误差：

$$
\text{rel\_err} = \frac{|g_\text{numeric} - g_\text{analytic}|}{\max(|g_\text{numeric}|, |g_\text{analytic}|, 10^{-8})}
$$

经验阈值：

- $< 10^{-7}$：基本就是对的（fp64 下）
- $10^{-7} \sim 10^{-4}$：可能有数值精度问题，可接受
- $> 10^{-2}$：几乎肯定推错了

PyTorch 提供 `torch.autograd.gradcheck`，写自定义 `Function` 时必跑，是手撕算子的"单元测试"。

### 2.4 梯度消失 / 爆炸的根因

把链式法则写开：

$$
\frac{\partial L}{\partial x_0} = \prod_{l=1}^{L} J_l, \quad J_l = \frac{\partial x_l}{\partial x_{l-1}}
$$

如果每层 Jacobian 的"奇异值平均"小于 1，连乘后梯度按指数衰减到 0（**梯度消失**）；大于 1 则按指数膨胀到 $\infty$（**梯度爆炸**）。这是为什么：

- **初始化**要让前向 / 反向激活方差守恒（Xavier、Kaiming——Module 1.3 详讲）
- **归一化**（LayerNorm / RMSNorm）把每层激活拉回稳定尺度（Module 1.3）
- **残差连接**让梯度多一条 $\frac{\partial}{\partial x_l}(x_l + f(x_l)) = I + \frac{\partial f}{\partial x_l}$ 的"高速公路"，把连乘拆成连加
- **Pre-LN** 比 Post-LN 在深 Transformer 上更稳（Module 4.4）

本节只点出根因——**梯度的稳定性是连乘问题，不是单层问题**——后续 1.3、Module 4 会逐个展开它的对策。

---

## 3. 最小代码示例

### 3.1 micrograd 风格的 `Value` class

下面 38 行实现一个标量级 autograd 引擎，完整覆盖 forward + backward：

```python
class Value:
    """Scalar autograd, micrograd-style."""
    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)         # 父节点（who created me）
        self._backward = lambda: None       # 该节点对父节点的梯度回流函数
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
        def _backward():
            self.grad  += 1.0 * out.grad    # d(out)/d(self) = 1
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        def _backward():
            self.grad  += other.data * out.grad   # d(out)/d(self) = other
            other.grad += self.data  * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), "relu")
        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo, visited = [], set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for p in v._prev: build(p)
                topo.append(v)
        build(self)
        self.grad = 1.0                     # dL/dL = 1
        for v in reversed(topo):            # 反向拓扑序
            v._backward()


# 跑一下：f = (a*b + c).relu()
a, b, c = Value(2.0), Value(-3.0), Value(10.0)
f = (a * b + c).relu()                      # = relu(2*-3 + 10) = relu(4) = 4.0
f.backward()
print(f.data, a.grad, b.grad, c.grad)       # 4.0  -3.0  2.0  1.0
```

**为什么这么写、关键在哪几行**：

- 每个算子（`__add__` / `__mul__` / `relu`）做两件事：① 算 forward 数值；② **闭包**捕获父节点引用，定义"上游 grad 来了之后怎么把梯度加到父节点"
- `out._backward` 用闭包而不是类方法——这就是 PyTorch `grad_fn` 的极简版
- `backward()` 里先做**拓扑排序**（保证一个节点的所有下游都先算完），然后从 $\frac{\partial L}{\partial L} = 1$ 出发反向走
- 注意是 `+=` 而不是 `=`——同一个变量可能被多次使用（`f = a*a` 时 `a` 出现两次），梯度必须**累加**。这就是 PyTorch 默认梯度累加的根本原因

把 `Value` 换成 `Tensor`、把闭包换成 C++ `Node`、把拓扑排序换成 engine、把 VJP 表手写改成派发——这就是 PyTorch autograd 的 1:1 升级版。

### 3.2 PyTorch 等价对照

```python
import torch
a = torch.tensor(2.0,  requires_grad=True)
b = torch.tensor(-3.0, requires_grad=True)
c = torch.tensor(10.0, requires_grad=True)

f = (a * b + c).relu()      # 同一张图自动构建
f.backward()                # 自动反向

print(f.item(), a.grad.item(), b.grad.item(), c.grad.item())
# 4.0  -3.0  2.0  1.0   ←  与 micrograd 完全一致
```

`requires_grad=True` 打开追踪后，PyTorch 在每次 forward 时把 op 节点挂到张量的 `.grad_fn` 上，`.backward()` 触发同样的反向拓扑遍历——只不过工程上做了 C++ engine、CUDA stream、显存复用等优化。**心智模型完全一样**。

---

## 4. 工程踩坑与经验

- ❗ **`loss.backward()` 后忘记 `optimizer.zero_grad()` 是新手 #1 bug**。PyTorch 默认梯度累加（见 §3.1 的 `+=` 解释），不清零等于把上一 step 的梯度叠到这一 step 上，等效 lr 翻倍 → loss 直接发散。标准模板：`optimizer.zero_grad(); loss.backward(); optimizer.step()`。新版本 API 推荐 `optimizer.zero_grad(set_to_none=True)`，省内存且让未参与计算的参数 grad 保持 `None` 而不是 0
- ❗ **`retain_graph=True` 的真实使用场景**——计算图在 `.backward()` 一次后默认被释放（省显存），如果后面还要再 backward 同一张图就会报 `Trying to backward through the graph a second time`。必须用 `retain_graph=True` 的情况：① 高阶导（`grad` 后再 `backward`，如 GAN gradient penalty、Hessian-vector product）；② 同一 forward 拆出多个 loss 分别 backward（如 multi-task 想分开看每个 loss 的 grad norm）；③ RL/RLHF 里需要从同一 forward 同时算 policy loss 与 value loss，且不愿意把 forward 跑两遍。**注意**：`retain_graph=True` 不会清梯度，必须配合 `zero_grad`，否则梯度会双倍累加
- ❗ **`tensor.detach()` vs `tensor.data` 的差异**——两者都返回"脱离计算图"的张量，但 `.detach()` **共享存储但记录版本号**，inplace 修改会被 autograd 检测到并报错；`.data` **共享存储且不检查**，inplace 修改会**静默破坏**计算图，给出错误梯度且不报错。结论：**永远用 `.detach()`，不要碰 `.data`**。`.data` 是历史包袱，PyTorch 1.x 之后官方建议不再使用
- ❗ **inplace op 破坏计算图**——`x += 1`、`x.relu_()`、`x.copy_(...)` 这种带下划线后缀或复合赋值的 op 会原地修改张量，如果 autograd 之后还需要这个张量做 backward（很多算子的 backward 需要读 forward 的 input/output），就会报 `one of the variables needed for gradient computation has been modified by an inplace operation`。debug 时把所有 `+=` 改成 `x = x + 1`、`relu_()` 改成 `relu()` 试试就知道
- ❗ **`torch.no_grad()` / `.detach()` / `inference_mode()` 三者区别**——`no_grad()` 是上下文管理器，禁用 autograd 追踪（forward 不建图，省显存），用于 eval / generate；`.detach()` 是单点切断（这一个张量脱图，其它仍追踪）；`inference_mode()` 比 `no_grad` 更激进——除了不建图，还禁用 version counter 等所有 autograd 元数据，纯推理快 5-10%，但产出的张量**不能再放回**追踪图（否则报错），所以训练循环里千万别用
- ❗ **梯度爆炸用 `clip_grad_norm_` 不是 `clip_grad_value_`**——前者按 L2 范数裁剪整个参数组（保持方向），后者按值裁剪每个分量（破坏方向）。LLM 训练里 grad clip threshold 常取 1.0；grad norm 突然飙高是 loss spike 的早期信号，要监控 `grad_norm` 而不仅仅 loss
- ❗ **AMP / fp16 训练里 `loss.backward()` 之前要 `scaler.scale(loss).backward()`**——fp16 的最小正数约 $6 \times 10^{-5}$，比这小的梯度会 underflow 成 0。GradScaler 把 loss 放大 $2^k$ 倍再 backward，`step` 前 unscale 回来，是 fp16 训练的标配（Module 7.4 详讲）

---

## 5. 经典 paper

- **Rumelhart, Hinton & Williams, 1986 — *Learning representations by back-propagating errors*（Nature）** — 反向传播的原典 paper，3 页讲清楚链式法则在多层网络上的应用。今天看公式的形式可能不同，但 mental model 和本节 §2.2 的 2 层 MLP 推导**一字不差**。读它能让你直观感受到"40 年前奠基的东西"——所有现代 LLM 训练都站在这 3 页纸上
- **Baydin, Pearlmutter, Radul & Siskind, 2017 — *Automatic Differentiation in Machine Learning: a Survey*（JMLR）** — autograd 系统的最佳综述，把 forward-mode / reverse-mode、tape-based / source-transformation、symbolic / numerical / automatic 区分讲得最清楚。读完能彻底搞懂 PyTorch (reverse-mode tape) 和 JAX (source-transformation + functional) 的设计差异
- 选读：**Griewank, 2000 — *Evaluating Derivatives*（书）** — AD 圣经，第 3-4 章把 reverse-mode 的复杂度 $O(\text{forward cost})$ 严格证明出来。如果将来要写训练框架 / 优化器，这本是案头书

---

## 6. 自测与面试题

**Q1（概念）：** 为什么 PyTorch 默认 `tensor.grad` 是**累加**的，而不是覆盖的？

<details>
<summary>Answer sketch</summary>

- 同一个张量在计算图里**可能被多次使用**（如 weight tying、share embedding、Siamese 网络），每次出现都会有一条独立的梯度回流路径，必须累加才正确
- 见 §3.1 micrograd 实现的 `+=`：拓扑排序保证父节点最后被处理，但同一个父节点的多个下游会**多次调用** `_backward`，梯度必须累加而不是覆盖
- 累加还带来一个工程便利——**gradient accumulation** 训练（小 GPU 模拟大 batch）：跑 K 个 micro-batch 的 forward+backward 不调 `step`，让梯度自然累加，第 K 步再 `step + zero_grad`，等效 batch_size × K
- 代价就是新手必须显式 `zero_grad`，忘了就是 §4 的 #1 bug

</details>

**Q2（手推）：** 给定 $y = \sigma(W x + b)$，其中 $\sigma$ 是 sigmoid、$x \in \mathbb{R}^{B \times d_\text{in}}$、$W \in \mathbb{R}^{d_\text{in} \times d_\text{out}}$、$y \in \mathbb{R}^{B \times d_\text{out}}$、loss 已知 $\bar{y} = \frac{\partial L}{\partial y}$。写出 $\frac{\partial L}{\partial W}$ 的形状与表达式。

<details>
<summary>Answer sketch</summary>

要点：

- 先穿过 sigmoid：sigmoid 的导数是 $\sigma'(z) = \sigma(z)(1 - \sigma(z)) = y \odot (1 - y)$（element-wise）。设 $z = Wx + b$，则 $\bar{z} = \bar{y} \odot y \odot (1 - y) \in \mathbb{R}^{B \times d_\text{out}}$
- 再穿过 affine：$z = xW + b$，所以 $\frac{\partial L}{\partial W} = x^\top \bar{z} \in \mathbb{R}^{d_\text{in} \times d_\text{out}}$
- 形状检查：$x^\top$ 是 $d_\text{in} \times B$，$\bar{z}$ 是 $B \times d_\text{out}$，乘起来 $d_\text{in} \times d_\text{out}$，与 $W$ 同形 ✓
- 加分点：注意 batch 维度被 $x^\top \bar{z}$ 这一步"内积掉了"，**不需要再除以 B**——因为 $\bar{y}$ 里已经吸收了 loss 对 batch 的归一化（如 mean reduction 的 $1/B$ 已经在 $\bar{y}$ 里了）
- 进一步加分：sigmoid + BCE 在数值上的稳定写法（`logsigmoid` / `BCEWithLogitsLoss`）——不直接算 sigmoid 再算 BCE，否则大正/负 logits 会 overflow

</details>

**Q3（延伸）：** 在 RLHF 的 PPO 里，policy update 一个 epoch 内可能多次对同一批 rollout 算 loss + backward（mini-batch + 多 PPO epoch），应该用 `retain_graph=True` 还是别的方案？为什么？

<details>
<summary>Answer sketch</summary>

要点：

- **不应该用 `retain_graph=True`**。`retain_graph` 是为"同一张前向图被多次反向"设计的；PPO 的多次 backward 是**对不同 mini-batch、不同 forward 的**多次 backward——每次 mini-batch 都重新跑 policy forward 算 logp、再算 ratio 和 surrogate loss、再 backward
- 真正的做法：① rollout 阶段用 `with torch.no_grad()` 跑 actor 收集 trajectory，把 `(state, action, old_logp, advantage)` 存下来——**old_logp 必须 detach 存数值**，不带梯度；② update 阶段对每个 mini-batch **重新做一次 actor forward** 算 `new_logp`，与存好的 `old_logp` 计算 ratio = `exp(new_logp - old_logp)`，算 PPO surrogate loss 然后 backward。新 forward 新图，每次 backward 后图自然释放
- 用 `retain_graph=True` 会带来什么后果：① 显存爆炸（每个 epoch 的图都不释放）；② 而且仍然不对——因为 PPO 的 ratio 需要**新旧 policy 在同一 (s, a) 上的 logp 之比**，旧 logp 必须是数值常量，retain 同一张图算两次 backward 算的是"同一 policy 对同一 input 的两次梯度"，梯度信号错了
- 同样道理适用 GRPO（Module 9.5）、DPO（9.4）——所有 off-policy / 多 epoch 的 RLHF 算法都是"存 detach 的 old logp + 新 forward 算 new logp"模式
- **`retain_graph` 真正该用的场景**是 single forward 多 backward 且共享一张图的，例如 GAN gradient penalty 里要算 $\nabla_x D(x)$ 的 norm 再 backward 一次，那是同一张 D 图

</details>

---

## 7. 延伸阅读

- [Karpathy — micrograd (GitHub)](https://github.com/karpathy/micrograd) — 100 行实现 autograd 引擎，本节 §3.1 代码的祖本。配套 YouTube 视频《The spelled-out intro to neural networks and backpropagation》是入门 backprop 的最佳一小时
- [PyTorch Autograd 官方教程](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html) — 官方对 `requires_grad` / `grad_fn` / `backward` / `no_grad` 的最简洁说明，配合本节工程踩坑读
- [PyTorch Autograd Mechanics](https://pytorch.org/docs/stable/notes/autograd.html) — autograd 的"机制说明书"，详细讲计算图构建、版本号、inplace 检测、保存的张量、双 backward。官方文档里最值得通读的一篇
- [CS231n — Backpropagation, Intuitions](https://cs231n.github.io/optimization-2/) — Stanford 公开课讲义，把"梯度沿图反向流动"的直觉建立得最好；推荐读完后做附带的 vectorized gradient 习题
- [colah — Calculus on Computational Graphs: Backpropagation](https://colah.github.io/posts/2015-08-Backprop/) — 用最少公式把 forward-mode 与 reverse-mode AD 的区别讲清楚，理解"为什么深度学习用 reverse-mode"的最佳一文
- 推荐继续读本教程的 **1.2 节《优化器：SGD / Momentum / Adam / AdamW + LR schedule》**——梯度算出来之后该怎么用它更新参数
