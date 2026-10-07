package solver

import (
	"errors"
	"math"
	"testing"

	"github.com/feng111627/sparse-linear-solver-go/sparse"
)

// 构造 n 阶 SPD 三对角矩阵 [2 -1; -1 2 -1; ...]。
func tridiagSPD(t *testing.T, n int) *sparse.CSR {
	t.Helper()
	var ri, ci []int
	var v []float64
	for i := 0; i < n; i++ {
		if i > 0 {
			ri = append(ri, i)
			ci = append(ci, i-1)
			v = append(v, -1)
		}
		ri = append(ri, i)
		ci = append(ci, i)
		v = append(v, 2)
		if i < n-1 {
			ri = append(ri, i)
			ci = append(ci, i+1)
			v = append(v, -1)
		}
	}
	coo, err := sparse.NewCOO(n, n, ri, ci, v)
	if err != nil {
		t.Fatalf("setup: %v", err)
	}
	csr, err := coo.ToCSR()
	if err != nil {
		t.Fatalf("setup: %v", err)
	}
	return csr
}

func relErr(x, want []float64) float64 {
	num, den := 0.0, 0.0
	for i := range x {
		d := x[i] - want[i]
		num += d * d
		den += want[i] * want[i]
	}
	return math.Sqrt(num / den)
}

func TestSolveCG_Basic(t *testing.T) {
	n := 20
	a := tridiagSPD(t, n)
	// 精确解取全 1 向量，b = A*1。
	want := make([]float64, n)
	for i := range want {
		want[i] = 1
	}
	b, err := sparse.MatVec(a, want)
	if err != nil {
		t.Fatalf("setup: %v", err)
	}
	res, err := SolveCG(a, b, Options{Tolerance: 1e-12, MaxIter: 500})
	if err != nil {
		t.Fatalf("solve: %v", err)
	}
	if !res.Converged {
		t.Fatal("expected convergence")
	}
	if e := relErr(res.X, want); e > 1e-8 {
		t.Fatalf("solution error too large: %g", e)
	}
	t.Logf("converged in %d iterations, relres=%g, solution error=%g",
		res.Iterations, res.Residual, relErr(res.X, want))
}

func TestSolveCG_Preconditioners_Consistent(t *testing.T) {
	n := 30
	a := tridiagSPD(t, n)
	want := make([]float64, n)
	for i := range want {
		want[i] = float64(i%3) + 0.5
	}
	b, err := sparse.MatVec(a, want)
	if err != nil {
		t.Fatalf("setup: %v", err)
	}
	jac, err := NewJacobi(a)
	if err != nil {
		t.Fatalf("jacobi: %v", err)
	}
	opts := Options{Tolerance: 1e-12, MaxIter: 500}
	rNone, err := SolveCG(a, b, opts)
	if err != nil {
		t.Fatalf("no precond: %v", err)
	}
	opts.Preconditioner = Identity{}
	rIdent, err := SolveCG(a, b, opts)
	if err != nil {
		t.Fatalf("identity precond: %v", err)
	}
	opts.Preconditioner = jac
	rJac, err := SolveCG(a, b, opts)
	if err != nil {
		t.Fatalf("jacobi precond: %v", err)
	}
	// 同一输入下，不同预条件子都必须收敛到同一个解。
	for name, r := range map[string]*Result{"identity": rIdent, "jacobi": rJac} {
		if !r.Converged {
			t.Fatalf("%s: not converged", name)
		}
		if e := relErr(r.X, want); e > 1e-8 {
			t.Fatalf("%s: solution error %g", name, e)
		}
	}
	t.Logf("no-precond: %d iters; identity: %d iters; jacobi: %d iters; all converge to same solution",
		rNone.Iterations, rIdent.Iterations, rJac.Iterations)
}

func TestSolveCG_Reproducible(t *testing.T) {
	n := 15
	a := tridiagSPD(t, n)
	b := make([]float64, n)
	for i := range b {
		b[i] = float64(i) + 1
	}
	r1, err := SolveCG(a, b, Options{Tolerance: 1e-12, MaxIter: 500})
	if err != nil {
		t.Fatalf("run1: %v", err)
	}
	r2, err := SolveCG(a, b, Options{Tolerance: 1e-12, MaxIter: 500})
	if err != nil {
		t.Fatalf("run2: %v", err)
	}
	if r1.Iterations != r2.Iterations || r1.Residual != r2.Residual {
		t.Fatal("two runs differ in iterations/residual")
	}
	for i := range r1.X {
		if r1.X[i] != r2.X[i] {
			t.Fatal("two runs differ in solution")
		}
	}
	t.Logf("two identical runs give bit-identical results (iters=%d)", r1.Iterations)
}

func TestSolveCG_DimensionMismatch(t *testing.T) {
	a := tridiagSPD(t, 4)
	_, err := SolveCG(a, []float64{1, 2, 3}, Options{})
	var dm *sparse.ErrDimensionMismatch
	if !errors.As(err, &dm) {
		t.Fatalf("expected ErrDimensionMismatch, got %v", err)
	}
	t.Logf("dimension mismatch rejected: %v", err)
}

func TestSolveCG_Singular(t *testing.T) {
	// 奇异矩阵：全零行使矩阵半正定，CG 应报 ErrNotSPD 而不是返回错误解。
	coo, err := sparse.NewCOO(2, 2, []int{0}, []int{0}, []float64{1})
	if err != nil {
		t.Fatalf("setup: %v", err)
	}
	a, err := coo.ToCSR()
	if err != nil {
		t.Fatalf("setup: %v", err)
	}
	_, err = SolveCG(a, []float64{1, 1}, Options{Tolerance: 1e-12, MaxIter: 100})
	var nspd *ErrNotSPD
	if !errors.As(err, &nspd) {
		t.Fatalf("expected ErrNotSPD, got %v", err)
	}
	t.Logf("singular matrix rejected: %v", err)
}

func TestSolveCG_Indefinite(t *testing.T) {
	// 不定矩阵 diag(1, -1)：不是 SPD，必须报错。
	coo, err := sparse.NewCOO(2, 2, []int{0, 1}, []int{0, 1}, []float64{1, -1})
	if err != nil {
		t.Fatalf("setup: %v", err)
	}
	a, err := coo.ToCSR()
	if err != nil {
		t.Fatalf("setup: %v", err)
	}
	_, err = SolveCG(a, []float64{1, 1}, Options{Tolerance: 1e-12, MaxIter: 100})
	var nspd *ErrNotSPD
	if !errors.As(err, &nspd) {
		t.Fatalf("expected ErrNotSPD, got %v", err)
	}
	t.Logf("indefinite matrix rejected: %v", err)
}

func TestSolveCG_NotConverged(t *testing.T) {
	// 病态矩阵：对角元跨度 1..1e8，条件数约 1e8；MaxIter 很小必然不收敛。
	n := 50
	var ri, ci []int
	var v []float64
	for i := 0; i < n; i++ {
		ri = append(ri, i)
		ci = append(ci, i)
		v = append(v, math.Pow(10, 8*float64(i)/float64(n-1)))
	}
	coo, err := sparse.NewCOO(n, n, ri, ci, v)
	if err != nil {
		t.Fatalf("setup: %v", err)
	}
	a, err := coo.ToCSR()
	if err != nil {
		t.Fatalf("setup: %v", err)
	}
	b := make([]float64, n)
	for i := range b {
		b[i] = 1
	}
	_, err = SolveCG(a, b, Options{Tolerance: 1e-14, MaxIter: 3})
	var nc *ErrNotConverged
	if !errors.As(err, &nc) {
		t.Fatalf("expected ErrNotConverged, got %v", err)
	}
	t.Logf("ill-conditioned matrix with tiny MaxIter rejected: %v", err)
}

func TestSolveCG_NaNRHS(t *testing.T) {
	a := tridiagSPD(t, 3)
	_, err := SolveCG(a, []float64{1, math.NaN(), 1}, Options{})
	var ne *ErrNumerical
	if !errors.As(err, &ne) {
		t.Fatalf("expected ErrNumerical, got %v", err)
	}
	t.Logf("NaN right-hand side rejected: %v", err)
}

func TestSolveCG_ZeroRHS(t *testing.T) {
	a := tridiagSPD(t, 5)
	res, err := SolveCG(a, make([]float64, 5), Options{Tolerance: 1e-12})
	if err != nil {
		t.Fatalf("solve: %v", err)
	}
	for i, x := range res.X {
		if x != 0 {
			t.Fatalf("x[%d]=%v, want 0", i, x)
		}
	}
	t.Logf("zero RHS converges immediately to zero solution (iters=%d)", res.Iterations)
}

func TestJacobi_ZeroDiagonal(t *testing.T) {
	coo, err := sparse.NewCOO(2, 2, []int{0, 1}, []int{1, 1}, []float64{1, 1})
	if err != nil {
		t.Fatalf("setup: %v", err)
	}
	a, err := coo.ToCSR()
	if err != nil {
		t.Fatalf("setup: %v", err)
	}
	_, err = NewJacobi(a)
	if err == nil {
		t.Fatal("expected error for zero diagonal")
	}
	t.Logf("zero diagonal rejected: %v", err)
}
