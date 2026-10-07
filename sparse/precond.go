package sparse

import (
	"fmt"
	"math"
)

// Preconditioner applies the inverse preconditioning operator M^{-1} to a
// residual vector r, returning z with M*z ~ r.
//
// Implementations must be stateless across SolveCG calls with respect to the
// input: applying the same preconditioner to the same vector repeatedly must
// return the same result.
type Preconditioner interface {
	Apply(r []float64) ([]float64, error)
}

// IdentityPreconditioner leaves the residual unchanged (a defensive copy).
type IdentityPreconditioner struct{}

// NewIdentityPreconditioner returns the trivial preconditioner M = I.
func NewIdentityPreconditioner() *IdentityPreconditioner {
	return &IdentityPreconditioner{}
}

// Apply returns a copy of r after validating finiteness.
func (p *IdentityPreconditioner) Apply(r []float64) ([]float64, error) {
	const op = "IdentityPreconditioner.Apply"
	if err := checkFinite(r, op, "r"); err != nil {
		return nil, err
	}
	z := append([]float64(nil), r...)
	return z, nil
}

// JacobiPreconditioner applies the inverse of the matrix diagonal.
type JacobiPreconditioner struct {
	invDiag []float64
}

// NewJacobiPreconditioner builds M = diag(A). The matrix must be square and
// every diagonal entry must be non-zero and finite, otherwise a sentinel
// error (ErrDimensionMismatch / ErrSingularMatrix / ErrNonFiniteValue) is
// returned. Zero off-diagonal entries are irrelevant.
func NewJacobiPreconditioner(m Matrix) (*JacobiPreconditioner, error) {
	const op = "NewJacobiPreconditioner"
	rows, cols := m.Dims()
	if rows != cols {
		return nil, dimensionError(op, "matrix must be square, got %dx%d", rows, cols)
	}
	diag, err := extractDiagonal(m)
	if err != nil {
		return nil, err
	}
	inv := make([]float64, rows)
	for i, d := range diag {
		if math.IsNaN(d) || math.IsInf(d, 0) {
			return nil, fmt.Errorf("%w: %s: diagonal entry %d is NaN or Inf",
				ErrNonFiniteValue, op, i)
		}
		if d == 0 {
			return nil, fmt.Errorf("%w: %s: zero diagonal entry at (%d,%d)",
				ErrSingularMatrix, op, i, i)
		}
		inv[i] = 1 / d
	}
	return &JacobiPreconditioner{invDiag: inv}, nil
}

// Apply computes z[i] = r[i] / A[i,i].
func (p *JacobiPreconditioner) Apply(r []float64) ([]float64, error) {
	const op = "JacobiPreconditioner.Apply"
	if len(r) != len(p.invDiag) {
		return nil, dimensionError(op, "r has length %d, preconditioner dimension is %d",
			len(r), len(p.invDiag))
	}
	if err := checkFinite(r, op, "r"); err != nil {
		return nil, err
	}
	z := make([]float64, len(r))
	for i := range r {
		z[i] = p.invDiag[i] * r[i]
	}
	if err := checkFinite(z, op, "z"); err != nil {
		return nil, err
	}
	return z, nil
}

// ChainPreconditioner applies a fixed sequence of preconditioners in order:
// z = p_k(...p_2(p_1(r))). It is deterministic: no randomness is involved.
type ChainPreconditioner struct {
	ps []Preconditioner
}

// NewChainPreconditioner composes the given preconditioners. An empty list is
// rejected.
func NewChainPreconditioner(ps ...Preconditioner) (*ChainPreconditioner, error) {
	const op = "NewChainPreconditioner"
	if len(ps) == 0 {
		return nil, structureError(op, ErrInvalidStructure,
			"preconditioner chain must contain at least one member")
	}
	for i, p := range ps {
		if p == nil {
			return nil, structureError(op, ErrInvalidStructure,
				fmt.Sprintf("preconditioner at position %d is nil", i))
		}
	}
	cp := make([]Preconditioner, len(ps))
	copy(cp, ps)
	return &ChainPreconditioner{ps: cp}, nil
}

// Apply runs each member preconditioner on the previous result.
func (p *ChainPreconditioner) Apply(r []float64) ([]float64, error) {
	z := r
	for i, q := range p.ps {
		next, err := q.Apply(z)
		if err != nil {
			return nil, fmt.Errorf("ChainPreconditioner.Apply: member %d: %w", i, err)
		}
		z = next
	}
	return append([]float64(nil), z...), nil
}

// extractDiagonal returns the dense diagonal of a square matrix.
func extractDiagonal(m Matrix) ([]float64, error) {
	switch t := m.(type) {
	case *COO:
		n := t.rows
		d := make([]float64, n)
		for k := range t.data {
			if t.row[k] == t.col[k] {
				d[t.row[k]] = t.data[k]
			}
		}
		return d, nil
	case *CSR:
		n := t.rows
		d := make([]float64, n)
		for i := 0; i < n; i++ {
			for k := t.rowPtr[i]; k < t.rowPtr[i+1]; k++ {
				if t.colInd[k] == i {
					d[i] = t.data[k]
					break
				}
			}
		}
		return d, nil
	case *CSC:
		n := t.cols
		d := make([]float64, n)
		for j := 0; j < n; j++ {
			for k := t.colPtr[j]; k < t.colPtr[j+1]; k++ {
				if t.rowInd[k] == j {
					d[j] = t.data[k]
					break
				}
			}
		}
		return d, nil
	default:
		// Generic fallback: column j of A is A*e_j, whose j-th entry is A[j,j].
		rows, cols := m.Dims()
		d := make([]float64, rows)
		e := make([]float64, cols)
		for j := 0; j < cols; j++ {
			if j > 0 {
				e[j-1] = 0
			}
			e[j] = 1
			col, err := m.MatVec(e)
			if err != nil {
				return nil, err
			}
			d[j] = col[j]
		}
		return d, nil
	}
}

// checkFinite returns an ErrNonFiniteValue error if v contains NaN/Inf.
func checkFinite(v []float64, op, name string) error {
	for i, x := range v {
		if math.IsNaN(x) || math.IsInf(x, 0) {
			return fmtNonFinite(op, name, i)
		}
	}
	return nil
}
