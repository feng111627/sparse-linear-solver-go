# sparse-linear-solver-go

一个纯 Go 实现的稀疏线性方程组求解库，支持多种稀疏矩阵存储格式、矩阵向量乘法、
可插拔预条件子的共轭梯度（CG）迭代求解器。

## 支持的矩阵格式

| 格式 | 类型 | 说明 |
|------|------|------|
| COO  | `sparse.COO` | 坐标格式，逐元素存储 `(row, col, value)` |
| CSR  | `sparse.CSR` | 压缩行存储（`RowPtr`/`ColIdx`/`Values`） |
| CSC  | `sparse.CSC` | 压缩列存储（`ColPtr`/`RowIdx`/`Values`） |

三种格式之间可以相互转换（`ToCOO`/`ToCSR`/`ToCSC`），转换保证矩阵内容不变。

### 结构校验

所有构造函数（`NewCOO`/`NewCSR`/`NewCSC`）都会严格校验输入，非法结构一律返回
可区分的错误类型，绝不静默丢元素、补零或崩溃：

- `*sparse.ErrDuplicateEntry`：同一 `(row, col)` 位置出现重复元素；
- `*sparse.ErrIndexOutOfRange`：行/列索引越界；
- `*sparse.ErrDimensionMismatch`：矩阵与向量（或矩阵与矩阵）维度不匹配；
- `*sparse.ErrInvalidStructure`：指针数组长度错误、非单调、NaN/Inf 元素值等。

## 矩阵向量乘法

`sparse.MatVec(m, x)` 对三种格式均可计算 `y = A*x`，维度不匹配时返回
`*sparse.ErrDimensionMismatch`。

## 求解器与收敛判定

`solver.SolveCG(a, b, opts)` 使用（可预条件的）共轭梯度法求解 `A*x = b`，
要求 `A` 为对称正定（SPD）矩阵。

- **收敛判定**：相对残差 `||r_k|| / ||b|| <= Tolerance`；
  当 `b` 为零向量时退化为绝对残差 `||r_k||`（此时直接返回零解）。
- **可复现性**：初始向量固定取零向量，不使用任何随机数；同一输入多次求解
  得到位级一致的结果，切换预条件子不改变解的语义。
- **确定性错误**：
  - `*solver.ErrNotSPD`：迭代中检测到矩阵非对称正定（`rho <= 0` 或 `pᵀAp <= 0`），
    奇异矩阵与不定矩阵都会走这条路径；
  - `*solver.ErrNotConverged`：达到 `MaxIter` 仍未满足收敛阈值；
  - `*solver.ErrNumerical`：右端项或迭代过程中出现 NaN/Inf；
  - `*sparse.ErrDimensionMismatch`：矩阵非方阵或 `b` 维度不符。

### 预条件子

预条件子实现统一接口：

```go
type Preconditioner interface {
    Apply(r []float64) ([]float64, error) // 求解 M*z = r
}
```

内置实现：

- `solver.Identity{}`：不做预处理（等价于 `Options.Preconditioner = nil`）；
- `solver.NewJacobi(a)`：Jacobi 预条件子 `M = diag(A)`，对角线含零时构造即报错。

通过 `Options.Preconditioner` 接入求解流程，可自由替换或扩展。

## 本地验证

```bash
go build ./...   # 构建
go vet ./...     # 静态检查
go test ./...    # 运行全部测试
go test ./... -v # 查看每个用例覆盖的场景与结果
```

测试覆盖：正常求解路径、三种格式的构造与互转、重复元素/越界索引/维度不匹配、
奇异矩阵、不定矩阵、病态矩阵不收敛、NaN 输入、零右端项、零对角预条件子，
以及多次运行与多种预条件子下结果的一致性。
