package solver

import (
	"fmt"

	"github.com/feng111627/sparse-linear-solver-go/sparse"
)

// Preconditioner 表示一个预条件子 M≈A，Apply 求解 M*z = r。
type Preconditioner interface {
	// Apply 求解 M*z = r，返回 z。输入维度必须等于矩阵阶数。
	Apply(r []float64) ([]float64, error)
}

// Identity 预条件子：不做任何处理（等价于无预条件）。
type Identity struct{}

// Apply 实现 Preconditioner。
func (Identity) Apply(r []float64) ([]float64, error) {
	z := make([]float64, len(r))
	copy(z, r)
	return z, nil
}

// Jacobi 预条件子：M = diag(A)。
type Jacobi struct {
	n       int
	invDiag []float64
}

// NewJacobi 从 CSR 矩阵构造 Jacobi 预条件子；矩阵必须为方阵且对角线全非零。
func NewJacobi(a *sparse.CSR) (*Jacobi, error) {
	if a.NumRows() != a.NumCols() {
		return nil, &sparse.ErrDimensionMismatch{
			What: "jacobi preconditioner requires square matrix",
			RowsA: a.NumRows(), ColsA: a.NumCols(),
		}
	}
	inv := make([]float64, a.NumRows())
	for i := 0; i < a.NumRows(); i++ {
		diag := 0.0
		found := false
		for k := a.RowPtr[i]; k < a.RowPtr[i+1]; k++ {
			if a.ColIdx[k] == i {
				diag = a.Values[k]
				found = true
				break
			}
		}
		if !found || diag == 0 {
			return nil, fmt.Errorf("solver: jacobi preconditioner: zero diagonal at row %d", i)
		}
		inv[i] = 1.0 / diag
	}
	return &Jacobi{n: a.NumRows(), invDiag: inv}, nil
}

// Apply 实现 Preconditioner。
func (j *Jacobi) Apply(r []float64) ([]float64, error) {
	if len(r) != j.n {
		return nil, &sparse.ErrDimensionMismatch{
			What:  "jacobi apply",
			RowsA: j.n, ColsA: j.n,
			RowsB: len(r), ColsB: 1,
		}
	}
	z := make([]float64, len(r))
	for i := range r {
		z[i] = j.invDiag[i] * r[i]
	}
	return z, nil
}
