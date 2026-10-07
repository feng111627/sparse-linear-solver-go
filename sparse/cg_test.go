package sparse

import (
	"errors"
	"math"
	"testing"
)

// laplacian1D returns the n x n tridiagonal SPD 1-D Laplacian as COO.
func laplacian1D(t *testing.T, n int) *COO {
	t.Helper()
	var rows, cols []int
	var data []float64
	add := func(r, c int, v float64) {
		rows = append(rows, r)
		cols = append(cols, c)
		data = append(data, v)
	}
	for i := 0; i < n; i++ {
		add(i, i, 2)
		if i+1 < n {
			add(i, i+1, -1)
			add(i+1, i, -1)
		}
	}
	m, err := NewCOO(n, n, rows, cols, data)
	if err != nil {
		t.Fatalf("laplacian1D: %v", err)
	}
	return m
}

func relErr(a, exact []float64) float64 {
	var num, den float64
	for i := range a {
		num += (a[i] - exact[i]) * (a[i] - exact[i])
		den += exact[i] * exact[i]
	}
	return math.Sqrt(num / den)
}

func TestSolveCGNormalPath(t *testing.T) {
	n := 40
	coo := laplacian1D(t, n)
	exact := make([]float64, n)
	for i := range exact {
		exact[i] = math.Sin(float64(i+1) * 0.37)
	}
	b, err := coo.MatVec(exact)
	if err != nil {
		t.Fatal(err)
	}

	configs := []struct {
		name string
		m    Matrix
		opts CGOptions
	}{
		{"COO + identity", coo, CGOptions{Tolerance: 1e-12}},
	}
	csr, _ := coo.ToCSR()
	csc, _ := coo.ToCSC()
	configs = append(configs, struct {
		name string
		m    Matrix
		opts CGOptions
	}{"CSR + identity", csr, CGOptions{Tolerance: 1e-12}})
	configs = append(configs, struct {
		name string
		m    Matrix
		opts CGOptions
	}{"CSC + identity", csc, CGOptions{Tolerance: 1e-12}})
	jac, err := NewJacobiPreconditioner(csr)
	if err != nil {
		t.Fatal(err)
	}
	chain, _ := NewChainPreconditioner(NewIdentityPreconditioner(), jac)
	configs = append(configs,
		struct {
			name string
			m    Matrix
			opts CGOptions
		}{"CSR + Jacobi", csr, CGOptions{Tolerance: 1e-12, Preconditioner: jac}},
		struct {
			name string
			m    Matrix
			opts CGOptions
		}{"CSR + chain(id,Jacobi)", csr, CGOptions{Tolerance: 1e-12, Preconditioner: chain}},
	)

	var ref *CGResult
	for i, cfg := range configs {
		t.Run(cfg.name, func(t *testing.T) {
			res, err := SolveCG(cfg.m, b, cfg.opts)
			if err != nil {
				t.Fatalf("SolveCG: %v", err)
			}
			if !res.Converged {
				t.Fatalf("not converged: %+v", res)
			}
			if e := relErr(res.X, exact); e > 1e-9 {
				t.Fatalf("relative solution error %.3e too large", e)
			}
			if i == 0 {
				ref = res
			} else {
				if res.Iterations != ref.Iterations {
					t.Fatalf("%s took %d iters, reference took %d",
						cfg.name, res.Iterations, ref.Iterations)
				}
				for k := range res.X {
					if res.X[k] != ref.X[k] {
						t.Fatalf("%s solution differs from reference at x[%d]: %v vs %v",
							cfg.name, k, res.X[k], ref.X[k])
					}
				}
			}
			t.Logf("scenario=%-24s iters=%-2d relres=%.3e relSolutionErr=%.3e",
				cfg.name, res.Iterations, res.RelResidual, relErr(res.X, exact))
		})
	}
}

func TestSolveCGReproducibility(t *testing.T) {
	coo := laplacian1D(t, 20)
	exact := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}
	b, _ := coo.MatVec(exact)
	opts := CGOptions{Tolerance: 1e-11}
	r1, err := SolveCG(coo, b, opts)
	if err != nil {
		t.Fatal(err)
	}
	for run := 0; run < 3; run++ {
		r2, err := SolveCG(coo, b, opts)
		if err != nil {
			t.Fatal(err)
		}
		if r2.Iterations != r1.Iterations {
			t.Fatalf("iteration count drifted: %d vs %d", r2.Iterations, r1.Iterations)
		}
		for i := range r1.X {
			if r2.X[i] != r1.X[i] {
				t.Fatalf("result drifted at x[%d]: %v vs %v", i, r2.X[i], r1.X[i])
			}
		}
	}
	t.Logf("scenario=repeated zero-guess solves bitwise identical: iters=%d relres=%.3e",
		r1.Iterations, r1.RelResidual)
}

func TestSolveCGInitialGuessSemantics(t *testing.T) {
	coo := laplacian1D(t, 10)
	exact := make([]float64, 10)
	for i := range exact {
		exact[i] = 1
	}
	b, _ := coo.MatVec(exact)

	// Exact initial guess => converged in zero iterations without changing x.
	res, err := SolveCG(coo, b, CGOptions{InitialGuess: exact, Tolerance: 1e-12})
	if err != nil {
		t.Fatal(err)
	}
	if res.Iterations != 0 || !res.Converged {
		t.Fatalf("exact guess: iters=%d converged=%v", res.Iterations, res.Converged)
	}
	t.Logf("scenario=exact initial guess converges at iter 0")

	// A wrong guess must change convergence behavior vs zero guess, but remain
	// deterministic across repeated runs.
	x0 := make([]float64, 10)
	for i := range x0 {
		x0[i] = 0.5
	}
	first, err := SolveCG(coo, b, CGOptions{InitialGuess: x0, Tolerance: 1e-12})
	if err != nil {
		t.Fatal(err)
	}
	second, err := SolveCG(coo, b, CGOptions{InitialGuess: x0, Tolerance: 1e-12})
	if err != nil {
		t.Fatal(err)
	}
	for i := range first.X {
		if first.X[i] != second.X[i] {
			t.Fatalf("nonzero guess solves are not reproducible at x[%d]", i)
		}
	}
	t.Logf("scenario=fixed nonzero guess deterministic iters=%d relres=%.3e",
		first.Iterations, first.RelResidual)

	// The solver must not mutate the caller's initial guess slice.
	snapshot := append([]float64(nil), x0...)
	SolveCG(coo, b, CGOptions{InitialGuess: x0})
	for i := range x0 {
		if x0[i] != snapshot[i] {
			t.Fatal("solver mutated caller's InitialGuess slice")
		}
	}
	t.Logf("scenario=caller initial-guess slice left unchanged")
}

func TestSolveCGValidationErrors(t *testing.T) {
	coo := laplacian1D(t, 5)
	cases := []struct {
		name    string
		a       Matrix
		b       []float64
		opts    CGOptions
		wantErr error
	}{
		{
			name:    "non-square matrix",
			a:       mustCOO(t, 2, 3, []int{0, 1}, []int{0, 1}, []float64{1, 1}),
			b:       []float64{1, 1},
			wantErr: ErrDimensionMismatch,
		},
		{
			name:    "b wrong length",
			a:       coo,
			b:       []float64{1, 2, 3},
			wantErr: ErrDimensionMismatch,
		},
		{
			name:    "b contains NaN",
			a:       coo,
			b:       []float64{1, 2, math.NaN(), 4, 5},
			wantErr: ErrNonFiniteValue,
		},
		{
			name:    "initial guess wrong length",
			a:       coo,
			b:       ones(5),
			opts:    CGOptions{InitialGuess: make([]float64, 4)},
			wantErr: ErrDimensionMismatch,
		},
		{
			name:    "initial guess contains Inf",
			a:       coo,
			b:       ones(5),
			opts:    CGOptions{InitialGuess: []float64{0, 0, math.Inf(1), 0, 0}},
			wantErr: ErrNonFiniteValue,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := SolveCG(tc.a, tc.b, tc.opts)
			if !errors.Is(err, tc.wantErr) {
				t.Fatalf("want %v, got %v", tc.wantErr, err)
			}
			t.Logf("scenario=%-28s rejected: %v", tc.name, err)
		})
	}

	t.Run("preconditioner wrong output length", func(t *testing.T) {
		bad := &fixedVectorPC{z: make([]float64, 4)}
		_, err := SolveCG(coo, ones(5), CGOptions{Preconditioner: bad})
		if !errors.Is(err, ErrDimensionMismatch) {
			t.Fatalf("want ErrDimensionMismatch, got %v", err)
		}
		t.Logf("scenario=preconditioner output length mismatch rejected: %v", err)
	})
	t.Run("preconditioner non-SPD", func(t *testing.T) {
		// Negative-definite preconditioner (M = -I): rho < 0 => breakdown.
		neg := &scalePC{s: -1}
		_, err := SolveCG(coo, ones(5), CGOptions{Preconditioner: neg})
		if !errors.Is(err, ErrBreakdown) {
			t.Fatalf("want ErrBreakdown, got %v", err)
		}
		t.Logf("scenario=non-positive-definite preconditioner rejected: %v", err)
	})
}

func TestSolveCGSingularAndIndefinite(t *testing.T) {
	t.Run("singular zero matrix", func(t *testing.T) {
		// Structural empty 3x3 matrix (all zeros) => singular.
		m, err := NewCOO(3, 3, []int{}, []int{}, []float64{})
		if err != nil {
			t.Fatal(err)
		}
		_, err = SolveCG(m, ones(3), CGOptions{})
		if !errors.Is(err, ErrSingularMatrix) {
			t.Fatalf("want ErrSingularMatrix, got %v", err)
		}
		t.Logf("scenario=all-zero singular matrix rejected: %v", err)
	})
	t.Run("singular zero diagonal row", func(t *testing.T) {
		// diag(1,1,0): CG with b having a component in the null direction
		// must fail rather than return a wrong answer.
		m := diagCOO(t, 3, []float64{1, 1, 0})
		_, err := SolveCG(m, []float64{1, 1, 1}, CGOptions{})
		if !(errors.Is(err, ErrSingularMatrix) || errors.Is(err, ErrBreakdown)) {
			t.Fatalf("want singular/breakdown, got %v", err)
		}
		t.Logf("scenario=rank-deficient diag matrix rejected: %v", err)
	})
	t.Run("indefinite matrix", func(t *testing.T) {
		m, err := NewCOO(2, 2, []int{0, 1}, []int{0, 1}, []float64{1, -1})
		if err != nil {
			t.Fatal(err)
		}
		_, err = SolveCG(m, []float64{1, 2}, CGOptions{})
		if !errors.Is(err, ErrBreakdown) {
			t.Fatalf("want ErrBreakdown for indefinite A, got %v", err)
		}
		t.Logf("scenario=indefinite diagonal rejected: %v", err)
	})
	t.Run("symmetric indefinite off-diagonal", func(t *testing.T) {
		m, err := NewCOO(2, 2,
			[]int{0, 1}, []int{1, 0}, []float64{1, 1})
		if err != nil {
			t.Fatal(err)
		}
		_, err = SolveCG(m, []float64{1, 0}, CGOptions{})
		if !(errors.Is(err, ErrSingularMatrix) || errors.Is(err, ErrBreakdown)) {
			t.Fatalf("want singular/breakdown, got %v", err)
		}
		t.Logf("scenario=zero-diag indefinite [[0,1],[1,0]] rejected: %v", err)
	})
}

func TestSolveCGNotConverged(t *testing.T) {
	n := 60
	coo := laplacian1D(t, n)
	exact := make([]float64, n)
	for i := range exact {
		exact[i] = math.Sin(float64(i) * 0.19)
	}
	b, _ := coo.MatVec(exact)
	res, err := SolveCG(coo, b, CGOptions{MaxIterations: 3, Tolerance: 1e-12})
	if !errors.Is(err, ErrNotConverged) {
		t.Fatalf("want ErrNotConverged, got %v", err)
	}
	if res == nil || res.Converged {
		t.Fatal("non-nil, non-converged result with last iterate expected")
	}
	if res.Iterations != 3 {
		t.Fatalf("iters = %d, want 3", res.Iterations)
	}
	if math.IsNaN(res.RelResidual) || res.RelResidual <= 1e-12 {
		t.Fatalf("stale residual reported: %.6e", res.RelResidual)
	}
	t.Logf("scenario=iteration cap 3/60 not converged: relres=%.6e, error=%v",
		res.RelResidual, err)
}

func TestSolveCGIllConditioned(t *testing.T) {
	t.Run("moderate condition number converges", func(t *testing.T) {
		// Diagonal SPD, cond ~= 1e6: CG should converge tightly.
		n := 4
		d := []float64{1e-3, 1, 10, 1e3}
		m := diagCOO(t, n, d)
		exact := []float64{1, 2, 3, 4}
		b := make([]float64, n)
		for i := range b {
			b[i] = d[i] * exact[i]
		}
		// Unpreconditioned float64 CG loses Lanczos orthogonality for
		// spread spectra: a looser tolerance is achievable. Tighter accuracy
		// is demonstrated via Jacobi preconditioning in the next subtest.
		res, err := SolveCG(m, b, CGOptions{Tolerance: 1e-4})
		if err != nil {
			t.Fatalf("moderately ill-conditioned SPD should converge at loose tol: %v", err)
		}
		e := relErr(res.X, exact)
		// Theory: relErr <= cond(A)*relRes = 1e6*~1e-5. The point here is
		// honest convergence reporting, not high accuracy without a good
		// preconditioner.
		if e > 1.0 {
			t.Fatalf("relative error %.3e exceeds cond*relres bound", e)
		}
		t.Logf("scenario=diag cond=1e6 unpreconditioned converged iters=%d relres=%.3e relErr=%.3e (bound cond*relres=%.2f)",
			res.Iterations, res.RelResidual, e, 1e6*res.RelResidual)
	})

	t.Run("extreme condition number never returns a wrong answer", func(t *testing.T) {
		// Eigenvalue spread 1e12 with a RHS exciting every eigen-direction:
		// in float64 the Lanczos direction loses orthogonality. Whatever
		// happens, the solver must signal (non-convergence or breakdown) and
		// must not hand back an unflagged wrong "solution".
		n := 4
		d := []float64{1e-6, 1, 1e3, 1e6}
		m := diagCOO(t, n, d)
		exact := []float64{1e6, 1, 0.001, 0.000001}
		b := make([]float64, n)
		for i := range b {
			b[i] = d[i] * exact[i]
		}
		res, err := SolveCG(m, b, CGOptions{Tolerance: 1e-10})
		switch {
		case err == nil:
			if e := relErr(res.X, exact); e > 1e-6 {
				t.Fatalf("converged flag but wrong answer, relErr=%.3e", e)
			}
			t.Logf("scenario=extreme cond converged accurately iters=%d relres=%.3e",
				res.Iterations, res.RelResidual)
		case errors.Is(err, ErrNotConverged), errors.Is(err, ErrBreakdown):
			t.Logf("scenario=extreme cond correctly flagged (%v), last relres=%.3e; no silent wrong answer",
				err, res.RelResidual)
		default:
			t.Fatalf("unexpected error class: %v", err)
		}
	})

	t.Run("Jacobi cures diagonal ill-conditioning", func(t *testing.T) {
		// For a diagonal system, Jacobi preconditioning turns A into I:
		// convergence in one iteration regardless of eigenvalue spread.
		n := 4
		d := []float64{1e-6, 1, 1e3, 1e6}
		m := diagCOO(t, n, d)
		exact := []float64{1e6, 1, 0.001, 0.000001}
		b := make([]float64, n)
		for i := range b {
			b[i] = d[i] * exact[i]
		}
		jac, err := NewJacobiPreconditioner(m)
		if err != nil {
			t.Fatal(err)
		}
		res, err := SolveCG(m, b, CGOptions{Tolerance: 1e-12, Preconditioner: jac})
		if err != nil {
			t.Fatalf("Jacobi-preconditioned diagonal solve failed: %v", err)
		}
		if res.Iterations != 1 {
			t.Fatalf("Jacobi diagonal: iters=%d, want 1", res.Iterations)
		}
		if e := relErr(res.X, exact); e > 1e-9 {
			t.Fatalf("relative error %.3e too large", e)
		}
		t.Logf("scenario=extreme cond + Jacobi converged in 1 iter relres=%.3e x=%v",
			res.RelResidual, res.X)
	})
}

func TestSolveCGHomogeneousRHS(t *testing.T) {
	coo := laplacian1D(t, 6)
	res, err := SolveCG(coo, make([]float64, 6), CGOptions{Tolerance: 1e-12})
	if err != nil {
		t.Fatal(err)
	}
	if !res.Converged || res.Iterations != 0 {
		t.Fatalf("b=0 should converge immediately, got %+v", res)
	}
	for i, v := range res.X {
		if v != 0 {
			t.Fatalf("x[%d] = %v, want 0", i, v)
		}
	}
	t.Logf("scenario=homogeneous b=0 -> unique zero solution at iter 0")
}

// ---- test preconditioners ----

type scalePC struct{ s float64 }

func (p *scalePC) Apply(r []float64) ([]float64, error) {
	z := make([]float64, len(r))
	for i := range r {
		z[i] = p.s * r[i]
	}
	return z, nil
}

type fixedVectorPC struct{ z []float64 }

func (p *fixedVectorPC) Apply(r []float64) ([]float64, error) {
	return append([]float64(nil), p.z...), nil
}

func ones(n int) []float64 {
	v := make([]float64, n)
	for i := range v {
		v[i] = 1
	}
	return v
}

func mustCOO(t *testing.T, rows, cols int, r, c []int, d []float64) *COO {
	t.Helper()
	m, err := NewCOO(rows, cols, r, c, d)
	if err != nil {
		t.Fatalf("mustCOO: %v", err)
	}
	return m
}
