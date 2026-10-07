// Command demo solves a small sparse linear system deterministically and
// prints the result for every supported preconditioner configuration.
//
// Run with: go run ./cmd/demo
package main

import (
	"errors"
	"fmt"
	"os"

	"github.com/feng111627/sparse-linear-solver-go/sparse"
)

func main() {
	// 1-D Dirichlet Laplacian on n = 6 interior points: tridiagonal, SPD.
	const n = 6
	var rows, cols []int
	var data []float64
	for i := 0; i < n; i++ {
		rows = append(rows, i)
		cols = append(cols, i)
		data = append(data, 2.0)
		if i+1 < n {
			rows = append(rows, i, i+1)
			cols = append(cols, i+1, i)
			data = append(data, -1.0, -1.0)
		}
	}
	coo, err := sparse.NewCOO(n, n, rows, cols, data)
	must("build COO", err)
	csr, err := coo.ToCSR()
	must("convert COO->CSR", err)
	csc, err := coo.ToCSC()
	must("convert COO->CSC", err)

	// b = A * [1..n] so the exact solution is known: x* = (1,2,...,n).
	exact := make([]float64, n)
	b := make([]float64, n)
	for i := range exact {
		exact[i] = float64(i + 1)
	}
	for _, mx := range []sparse.Matrix{coo, csr, csc} {
		y, err := mx.MatVec(exact)
		must("matvec", err)
		copy(b, y)
	}
	fmt.Printf("system: 1-D Laplacian %dx%d, nnz=%d; exact solution x*=[1..%d]\n",
		n, n, csr.NNZ(), n)

	jac, err := sparse.NewJacobiPreconditioner(csr)
	must("build Jacobi", err)
	chain, err := sparse.NewChainPreconditioner(
		sparse.NewIdentityPreconditioner(), jac)
	must("build chain", err)

	cases := []struct {
		name string
		pc   sparse.Preconditioner
	}{
		{"identity (no preconditioning)", nil},
		{"Jacobi", jac},
		{"chain(identity, Jacobi)", chain},
	}
	for _, tc := range cases {
		res, err := sparse.SolveCG(csr, b, sparse.CGOptions{
			Preconditioner: tc.pc,
			Tolerance:      1e-12,
		})
		if err != nil {
			fmt.Printf("  %-28s ERROR: %v\n", tc.name, err)
			os.Exit(1)
		}
		fmt.Printf("  %-28s converged=%-5v iters=%-2d relres=%.3e x=% .6f\n",
			tc.name, res.Converged, res.Iterations, res.RelResidual, res.X)
	}

	// Deterministic failure demonstration: an indefinite diagonal matrix
	// (one negative pivot) must be rejected with ErrBreakdown, never with a
	// silently wrong answer.
	bad, err := sparse.NewCOO(2, 2,
		[]int{0, 1}, []int{0, 1}, []float64{1, -1})
	must("build indefinite", err)
	_, err = sparse.SolveCG(bad, []float64{1, 2}, sparse.CGOptions{})
	switch {
	case errors.Is(err, sparse.ErrBreakdown):
		fmt.Printf("negative-definite direction rejected as expected: %v\n", err)
	case err != nil:
		fmt.Printf("rejected with error: %v\n", err)
	default:
		fmt.Println("UNEXPECTED: indefinite matrix was accepted")
		os.Exit(1)
	}
}

func must(stage string, err error) {
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s failed: %v\n", stage, err)
		os.Exit(1)
	}
}
