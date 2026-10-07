// Package solver 提供基于稀疏矩阵的迭代求解器与预条件子。
package solver

import (
	"fmt"
	"math"

	"github.com/feng111627/sparse-linear-solver-go/sparse"
)

// ErrNotConverged 表示迭代在最大步数内未收敛。
type ErrNotConverged struct {
	Iterations int
	Residual   float64
}

func (e *ErrNotConverged) Error() string {
	return fmt.Sprintf("solver: not converged after %d iterations (residual=%g)", e.Iterations, e.Residual)
}

// ErrNotSPD 表示矩阵不满足共轭梯度法要求的对称正定性。
type ErrNotSPD struct {
	Iteration int
}

func (e *ErrNotSPD) Error() string {
	return fmt.Sprintf("solver: matrix is not symmetric positive definite (detected at iteration %d)", e.Iteration)
}

// ErrNumerical 表示迭代过程中出现 NaN/Inf 等数值异常。
type ErrNumerical struct {
	Iteration int
	Detail    string
}

func (e *ErrNumerical) Error() string {
	return fmt.Sprintf("solver: numerical error at iteration %d: %s", e.Iteration, e.Detail)
}

// Options 控制求解过程。
type Options struct {
	Tolerance float64 // 相对残差收敛阈值 ||r||/||b|| <= Tolerance
	MaxIter   int     // 最大迭代步数
	// Preconditioner 为 nil 时表示不使用预条件子。
	Preconditioner Preconditioner
}

// Result 是求解结果。
type Result struct {
	X          []float64
	Iterations int
	Residual   float64 // 最终相对残差 ||r||/||b||
	Converged  bool
}

func dot(a, b []float64) float64 {
	s := 0.0
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

func norm(a []float64) float64 { return math.Sqrt(dot(a, a)) }

func finite(v []float64) bool {
	for _, x := range v {
		if math.IsNaN(x) || math.IsInf(x, 0) {
			return false
		}
	}
	return true
}

// SolveCG 用（可预条件的）共轭梯度法求解 A*x=b，A 必须为对称正定矩阵。
//
// 为保证可复现性，初始向量固定取零向量，不使用任何随机数。
// 收敛判定：||r_k|| / ||b|| <= Tolerance（b 为零向量时改用绝对残差 ||r||）。
func SolveCG(a *sparse.CSR, b []float64, opts Options) (*Result, error) {
	n := a.NumRows()
	if a.NumCols() != n {
		return nil, &sparse.ErrDimensionMismatch{
			What: "SolveCG requires square matrix",
			RowsA: a.NumRows(), ColsA: a.NumCols(),
		}
	}
	if len(b) != n {
		return nil, &sparse.ErrDimensionMismatch{
			What:  "SolveCG",
			RowsA: n, ColsA: n,
			RowsB: len(b), ColsB: 1,
		}
	}
	if !finite(b) {
		return nil, &ErrNumerical{Iteration: 0, Detail: "right-hand side contains NaN/Inf"}
	}
	tol := opts.Tolerance
	if tol <= 0 {
		tol = 1e-10
	}
	maxIter := opts.MaxIter
	if maxIter <= 0 {
		maxIter = 2 * n
	}
	pre := opts.Preconditioner
	if pre == nil {
		pre = Identity{}
	}

	// 固定零初始向量：x0 = 0, r0 = b。
	x := make([]float64, n)
	r := make([]float64, n)
	copy(r, b)

	bNorm := norm(b)
	// 收敛判定的归一化基准：b 为零向量时退化为绝对残差。
	scale := bNorm
	if scale == 0 {
		scale = 1.0
	}

	// 初始残差已满足收敛条件（如 b 为零向量）时直接返回零解。
	if norm(r)/scale <= tol {
		return &Result{X: x, Iterations: 0, Residual: norm(r) / scale, Converged: true}, nil
	}

	z, err := pre.Apply(r)
	if err != nil {
		return nil, err
	}
	if len(z) != n {
		return nil, &sparse.ErrDimensionMismatch{
			What:  "preconditioner output",
			RowsA: n, ColsA: n,
			RowsB: len(z), ColsB: 1,
		}
	}
	p := make([]float64, n)
	copy(p, z)
	rho := dot(r, z)
	if rho <= 0 {
		return nil, &ErrNotSPD{Iteration: 0}
	}

	for k := 0; k < maxIter; k++ {
		if !finite(r) || !finite(p) {
			return nil, &ErrNumerical{Iteration: k, Detail: "NaN/Inf in iterate"}
		}
		ap, err := sparse.MatVec(a, p)
		if err != nil {
			return nil, err
		}
		pap := dot(p, ap)
		if pap <= 0 {
			return nil, &ErrNotSPD{Iteration: k}
		}
		alpha := rho / pap
		for i := 0; i < n; i++ {
			x[i] += alpha * p[i]
			r[i] -= alpha * ap[i]
		}
		relRes := norm(r) / scale
		if relRes <= tol {
			return &Result{X: x, Iterations: k + 1, Residual: relRes, Converged: true}, nil
		}
		z, err = pre.Apply(r)
		if err != nil {
			return nil, err
		}
		rhoNew := dot(r, z)
		if rhoNew <= 0 {
			return nil, &ErrNotSPD{Iteration: k}
		}
		beta := rhoNew / rho
		for i := 0; i < n; i++ {
			p[i] = z[i] + beta*p[i]
		}
		rho = rhoNew
	}
	return nil, &ErrNotConverged{Iterations: maxIter, Residual: norm(r) / scale}
}
