package sparse

import (
	"fmt"
	"math"
)

// machineEpsilon is the gap between 1 and the next larger float64.
const machineEpsilon = 2.220446049250313e-16 // 2^-52

// Default relative residual tolerance when none is supplied:
// max(1e-10, 10*eps*n), a scale-aware threshold tied only to the dimension.
func defaultTolerance(n int) float64 {
	t := 10 * float64(n) * machineEpsilon
	if t < 1e-10 {
		t = 1e-10
	}
	return t
}

// CGOptions controls the conjugate gradient solver.
type CGOptions struct {
	// MaxIterations caps the number of CG iterations. If <= 0 it defaults to
	// the matrix dimension (theoretical convergence bound for exact SPD A).
	MaxIterations int
	// Tolerance is the relative residual threshold ||r|| / ||b||. If <= 0 the
	// scale-aware default tolerance is used.
	Tolerance float64
	// Preconditioner is optional; nil means the identity preconditioner.
	Preconditioner Preconditioner
	// InitialGuess is optional; nil means the zero vector. A non-nil vector is
	// copied; it must have the right length and contain only finite values.
	InitialGuess []float64
}

// CGResult reports the outcome of a solve. On error it still carries the best
// iterate reached, so callers can inspect progress even for non-convergence.
type CGResult struct {
	X            []float64
	Iterations   int
	ResidualNorm float64
	// RelResidual is ||A*x - b|| / ||b|| at the returned iterate.
	RelResidual float64
	Converged   bool
}

// SolveCG solves A*x = b for a symmetric positive definite matrix A using the
// (optionally preconditioned) conjugate gradient method.
//
// Determinism: with the default zero initial guess there is no randomness; the
// same (A, b, preconditioner, options) always yields the same iterates and
// convergence decision. Preconditioners are treated as pure operators.
//
// Error semantics (use errors.Is):
//   - ErrDimensionMismatch: non-square A, wrong b/x0/preconditioner lengths.
//   - ErrNonFiniteValue: NaN/Inf in b, x0 or any iterate/residual.
//   - ErrSingularMatrix: a CG denominator vanishes (breakdown), e.g. singular A.
//   - ErrBreakdown: a non-positive denominator reveals non-SPD (indefinite) A.
//   - ErrNotConverged: MaxIterations reached before the residual tolerance.
//
// When ErrNotConverged is returned, the result pointer is non-nil and holds
// the last iterate; other errors may return a nil result.
func SolveCG(a Matrix, b []float64, opts CGOptions) (*CGResult, error) {
	const op = "SolveCG"
	rows, cols := a.Dims()
	if rows != cols {
		return nil, dimensionError(op, "CG requires a square matrix, got %dx%d", rows, cols)
	}
	n := rows
	if b == nil {
		return nil, structureError(op, ErrInvalidStructure, "right-hand side b must not be nil")
	}
	if len(b) != n {
		return nil, dimensionError(op, "b has length %d, matrix dimension is %d", len(b), n)
	}
	if err := checkFinite(b, op, "b"); err != nil {
		return nil, err
	}
	x := make([]float64, n)
	if opts.InitialGuess != nil {
		if len(opts.InitialGuess) != n {
			return nil, dimensionError(op, "initial guess has length %d, expected %d",
				len(opts.InitialGuess), n)
		}
		if err := checkFinite(opts.InitialGuess, op, "x0"); err != nil {
			return nil, err
		}
		copy(x, opts.InitialGuess)
	}
	maxIter := opts.MaxIterations
	if maxIter <= 0 {
		maxIter = n
	}
	tol := opts.Tolerance
	if tol <= 0 || math.IsNaN(tol) || math.IsInf(tol, 0) {
		tol = defaultTolerance(n)
	}
	pc := Preconditioner(opts.Preconditioner)
	if pc == nil {
		pc = NewIdentityPreconditioner()
	}

	bnorm := math.Sqrt(dot(b, b))
	if bnorm == 0 {
		// Homogeneous system: x = 0 is the unique SPD solution. If the caller
		// supplied a nonzero guess the residual still determines convergence.
		bnorm = 1
	}

	// r = b - A*x
	ax, err := a.MatVec(x)
	if err != nil {
		return nil, fmt.Errorf("%s: initial matvec: %w", op, err)
	}
	r := make([]float64, n)
	for i := 0; i < n; i++ {
		r[i] = b[i] - ax[i]
	}
	if err := checkFinite(r, op, "r0"); err != nil {
		return nil, err
	}

	result := func(iter int, res []float64, converged bool) (*CGResult, error) {
		rn := norm2(res)
		return &CGResult{
			X:            x,
			Iterations:   iter,
			ResidualNorm: rn,
			RelResidual:  rn / bnorm,
			Converged:    converged,
		}, nil
	}

	if rn := norm2(r); rn/bnorm <= tol {
		return result(0, r, true)
	}

	z, err := pc.Apply(r)
	if err != nil {
		return nil, fmt.Errorf("%s: preconditioner: %w", op, err)
	}
	if len(z) != n {
		return nil, dimensionError(op, "preconditioner returned vector of length %d, expected %d",
			len(z), n)
	}
	if err := checkFinite(z, op, "z"); err != nil {
		return nil, err
	}
	p := make([]float64, n)
	copy(p, z)
	rho := dot(r, z)
	if err := checkScalar(rho, op, "rho0"); err != nil {
		return nil, err
	}
	if rho == 0 {
		// M^{-1} r == 0 with r != 0 means the preconditioner annihilates the
		// residual: it cannot be used with this matrix/vector.
		return nil, fmt.Errorf("%w: %s: preconditioner produced zero direction for nonzero residual",
			ErrSingularMatrix, op)
	}
	if rho < 0 {
		return nil, fmt.Errorf("%w: %s: preconditioned residual inner product is negative (matrix/preconditioner not positive definite)",
			ErrBreakdown, op)
	}

	var ap []float64
	for iter := 1; iter <= maxIter; iter++ {
		ap, err = a.MatVec(p)
		if err != nil {
			return nil, fmt.Errorf("%s: matvec at iteration %d: %w", op, iter, err)
		}
		if err := checkFinite(ap, op, "Ap"); err != nil {
			return nil, err
		}
		pAp := dot(p, ap)
		if err := checkScalar(pAp, op, "p'Ap"); err != nil {
			return nil, err
		}
		if pAp <= 0 {
			// Negative curvature => A is not positive definite along p;
			// zero curvature => singular/breakdown. No "plausible" answer is
			// returned.
			if pAp == 0 {
				return nil, fmt.Errorf("%w: %s: zero p'Ap at iteration %d (singular matrix or breakdown)",
					ErrSingularMatrix, op, iter)
			}
			return nil, fmt.Errorf("%w: %s: negative p'Ap=%g at iteration %d (matrix not symmetric positive definite)",
				ErrBreakdown, op, pAp, iter)
		}
		alpha := rho / pAp
		if math.IsNaN(alpha) || math.IsInf(alpha, 0) || alpha <= 0 {
			return nil, fmt.Errorf("%w: %s: invalid step length alpha=%g at iteration %d",
				ErrBreakdown, op, alpha, iter)
		}
		for i := 0; i < n; i++ {
			x[i] += alpha * p[i]
			r[i] -= alpha * ap[i]
		}
		if err := checkFinite(x, op, "x"); err != nil {
			return nil, err
		}
		if err := checkFinite(r, op, "r"); err != nil {
			return nil, err
		}
		if rn := norm2(r); rn/bnorm <= tol {
			return result(iter, r, true)
		}
		z, err = pc.Apply(r)
		if err != nil {
			return nil, fmt.Errorf("%s: preconditioner at iteration %d: %w", op, iter, err)
		}
		if len(z) != n {
			return nil, dimensionError(op, "preconditioner returned vector of length %d at iteration %d",
				len(z), iter)
		}
		if err := checkFinite(z, op, "z"); err != nil {
			return nil, err
		}
		rhoNew := dot(r, z)
		if err := checkScalar(rhoNew, op, "rho"); err != nil {
			return nil, err
		}
		if rhoNew <= 0 {
			if rhoNew == 0 {
				return nil, fmt.Errorf("%w: %s: zero rho at iteration %d (breakdown)",
					ErrSingularMatrix, op, iter)
			}
			return nil, fmt.Errorf("%w: %s: negative rho=%g at iteration %d (matrix/preconditioner not positive definite)",
				ErrBreakdown, op, rhoNew, iter)
		}
		beta := rhoNew / rho
		for i := 0; i < n; i++ {
			p[i] = z[i] + beta*p[i]
		}
		rho = rhoNew
	}

	res, err := result(maxIter, r, false)
	if err != nil {
		return nil, err
	}
	return res, fmt.Errorf("%w: after %d iterations, relative residual %.6e exceeds tolerance %.3e",
		ErrNotConverged, maxIter, res.RelResidual, tol)
}

func dot(a, b []float64) float64 {
	s := 0.0
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

func norm2(a []float64) float64 {
	return math.Sqrt(dot(a, a))
}

func checkScalar(v float64, op, name string) error {
	if math.IsNaN(v) || math.IsInf(v, 0) {
		return fmt.Errorf("%w: %s: scalar %s is NaN or Inf", ErrNonFiniteValue, op, name)
	}
	return nil
}
