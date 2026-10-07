# sparse-linear-solver-go

一个零依赖的 Go 稀疏线性代数库：多种稀疏存储格式及相互转换、矩阵-向量乘法、
带预条件子的共轭梯度（CG）迭代求解器。所有非法输入都以**可区分的错误**被拒绝，
不会静默丢元素、补零或 panic；不收敛时返回错误和最后一个迭代点，不会给出
"看着合理但错误"的答案。

## 目录结构

```
sparse/          库代码
  matrix.go      Matrix 公共接口
  errors.go      哨兵错误与 StructureError（带出错位置）
  numeric.go     内部数值辅助函数
  coo.go         COO（坐标/三元组）格式
  csr.go         CSR（行压缩）格式
  csc.go         CSC（列压缩）格式
  precond.go     Preconditioner 接口 + Identity / Jacobi / Chain
  cg.go          预条件共轭梯度求解器
cmd/demo/        确定性可复现示例
*_test.go        全部测试（正常路径 + 边界场景）
```

## 支持的矩阵格式

| 格式 | 说明 | 构造器 |
|------|------|--------|
| COO | 坐标/三元组 `(row, col, value)`，允许录入时乱序 | `NewCOO` |
| CSR | 行压缩，行内列索引严格递增 | `NewCSR` |
| CSC | 列压缩，列内行索引严格递增 | `NewCSC` |

三种格式都实现 `Matrix` 接口：

```go
type Matrix interface {
    Dims() (rows, cols int)
    NNZ() int                       // 结构化存储的元素个数
    MatVec(x []float64) ([]float64, error) // y = A*x
}
```

格式之间可两两互转，转换保持元素语义一致（COO 是枢纽表示）：

```go
coo.ToCSR(); coo.ToCSC()
csr.ToCOO(); csr.ToCSC()
csc.ToCOO(); csc.ToCSR()
```

### 构造期的非法结构校验

所有构造器都会拷贝入参切片，并在返回前校验。错误用哨兵值区分，
可通过 `errors.Is` 判定；同时可 `errors.As` 到 `*sparse.StructureError`
拿到出错元素下标与行列坐标：

| 哨兵错误 | 触发场景 |
|----------|----------|
| `ErrInvalidStructure` | 维度非正、切片为 nil/长度不符、rowPtr/colPtr 非单调或首尾不符、压缩格式内索引未排序 |
| `ErrIndexOutOfRange` | 行/列索引越界 |
| `ErrDuplicateEntry` | COO 中同一 `(row,col)` 重复出现；CSR/CSC 中同一行/列内重复索引 |
| `ErrNonFiniteValue` | 元素值为 NaN 或 ±Inf |
| `ErrDimensionMismatch` | MatVec / 求解器中向量长度与矩阵维度不符、矩阵非方阵等 |

重复元素一律拒绝（不做静默求和）；零值可以作为结构化元素显式存储（NNZ 照常计数）。

## 求解器：预条件共轭梯度

`SolveCG(a Matrix, b []float64, opts CGOptions) (*CGResult, error)` 求解 `A*x = b`，
要求 `A` 为对称正定（SPD）方阵。

### 收敛判定

- 收敛条件：相对残差 `||r_k||_2 / ||b||_2 <= tol`，其中 `r_k` 由 CG 递推更新。
- 默认容差：`max(1e-10, 10 * 2^-52 * n)`（只依赖维度，确定且可复现）；
  也可通过 `CGOptions.Tolerance` 显式指定。
- 最大迭代数：`CGOptions.MaxIterations <= 0` 时默认为 `n`（精确运算下 CG 的
  理论上限）。达到上限仍未收敛时返回包装了 `ErrNotConverged` 的错误，
  同时 `CGResult`（非 nil）携带最后一个迭代点、迭代数与相对残差。
- `b = 0` 时归一化分母按 1 处理，SPD 下零解在第 0 次迭代判定收敛。

`CGResult` 字段：`X, Iterations, ResidualNorm, RelResidual, Converged`。

### 失败语义（均可用 errors.Is 区分）

| 错误 | 含义 |
|------|------|
| `ErrDimensionMismatch` | A 非方阵；b / 初值长度不对；预条件子输出长度不对 |
| `ErrNonFiniteValue` | b、初值、任何迭代量（x/r/z/Ap）或标量（rho、p'Ap）出现 NaN/Inf |
| `ErrSingularMatrix` | 出现零分母（`rho=0` 或 `p'Ap=0`）：奇异矩阵或预条件子使残差归零 |
| `ErrBreakdown` | 出现负的 `p'Ap` 或负 `rho`：A（或预条件系统）非正定；步长非法 |
| `ErrNotConverged` | 迭代数用尽但未达到容差，结果只作为"最后迭代点"报告 |

每次迭代都对 `x、r、z、Ap` 做有限性检查；MatVec 结果溢出为 Inf 也会被
识别为 `ErrNonFiniteValue`。求解器不会修改调用方传入的 b 或 InitialGuess。

### 可复现性

- 默认初值为**零向量**（不是随机初值）；整个算法没有任何随机源。
- 相同 `(A, b, 预条件子, 选项)` 多次运行得到逐位相同的迭代序列、收敛判定与结果
  （测试中以 `-count=2` 和循环重复求解校验）。
- 预条件子被视为纯算子，不允许携带跨调用的可变状态语义；`Apply` 重复调用
  同一输入必须得到同一输出（`ChainPreconditioner` 有确定性测试）。

## 预条件子（可插拔）

```go
type Preconditioner interface {
    Apply(r []float64) ([]float64, error) // 返回 z ≈ M^{-1} r
}
```

内置实现：

- `IdentityPreconditioner`：M = I（默认，即普通 CG）。
- `JacobiPreconditioner`：M = diag(A)。要求矩阵为方阵且所有对角元非零、有限；
  零对角返回 `ErrSingularMatrix`。对 COO/CSR/CSC 直接取对角，其它 `Matrix`
  实现通过单位向量 `A*e_j` 探测取对角。
- `ChainPreconditioner`：按固定顺序组合多个预条件子
  `z = p_k(...p_1(r))`；空链或 nil 成员在构造期被拒绝。

自定义预条件子只需实现接口并在 `CGOptions.Preconditioner` 中注入；求解器会
校验其输出长度与有限性。

## 本地验证方法

需要 Go 1.25+（开发环境为 Go 1.26）。

```bash
# 1. 构建
go build ./...

# 2. 静态检查
go vet ./...

# 3. 全部测试（-v 可看到每个场景的 scenario 日志与用例结果；-race 检测数据竞争）
go test -race -count=1 -cover ./...
go test -v ./... | grep scenario

# 4. 可复现示例：同一 Laplacian 分别用 identity / Jacobi / chain 求解，
#    并演示不定矩阵被明确拒绝
go run ./cmd/demo
```

### 可复现的示例结论

`go run ./cmd/demo`（6×6 一维 Dirichlet Laplacian，精确解 `x*=(1..6)`，容差 1e-12）：

```
system: 1-D Laplacian 6x6, nnz=16; exact solution x*=[1..6]
  identity (no preconditioning) converged=true  iters=6  relres=8.310e-17 x=[ 1 2 3 4 5 6]
  Jacobi                       converged=true  iters=6  relres=8.310e-17 x=[ 1 2 3 4 5 6]
  chain(identity, Jacobi)      converged=true  iters=6  relres=8.310e-17 x=[ 1 2 3 4 5 6]
negative-definite direction rejected as expected: ... ErrBreakdown ...
```

测试覆盖的关键场景：

- 正常求解：COO/CSR/CSC 三种格式结果逐位一致；Jacobi、chain(id, Jacobi) 与
  identity 在同一输入下收敛判定一致；重复运行结果逐位相同。
- 非法结构：越界索引、重复元素、长度不符、指针非单调、NaN/Inf 元素各自对应
  不同的哨兵错误，且 `StructureError` 给出出错元素位置。
- 维度问题：非方阵、b/x0/预条件子向量长度不符。
- 奇异/非 SPD：零矩阵、零对角奇异阵、不定对角阵、零对角不定阵分别触发
  `ErrSingularMatrix` / `ErrBreakdown`。
- 不收敛：人为压低迭代上限，返回 `ErrNotConverged` 且结果标记为非收敛，
  并给出最终相对残差。
- 病态矩阵：cond≈1e6 时无预条件 CG 仅能达到有限精度（日志中同时给出
  `cond*relres` 理论误差界）；cond=1e12 的对角阵在 Jacobi 预条件下一步收敛；
  极端病态且无法保证正确时宁可报错也不返回未标记的错解。
- 数值异常：乘积溢出产生 Inf、输入 NaN 均被 `ErrNonFiniteValue` 拦截。
