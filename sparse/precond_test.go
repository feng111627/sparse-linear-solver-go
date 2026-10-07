package sparse

import (
	"errors"
	"math"
	"testing"
)

func diagCOO(t *testing.T, n int, diag []float64) *COO {
	t.Helper()
	r := make([]int, n)
	c := make([]int, n)
	for i := 0; i < n; i++ {
		r[i], c[i] = i, i
	}
	m, err := NewCOO(n, n, r, c, diag)
	if err != nil {
		t.Fatalf("diagCOO: %v", err)
	}
	return m
}

func TestJacobiPreconditioner(t *testing.T) {
	m := sampleCOO(t) // diag = 2, 3, 4
	j, err := NewJacobiPreconditioner(m)
	if err != nil {
		t.Fatalf("NewJacobi: %v", err)
	}
	z, err := j.Apply([]float64{2, 6, 4})
	if err != nil {
		t.Fatalf("Apply: %v", err)
	}
	want := []float64{1, 2, 1}
	for i := range want {
		if z[i] != want[i] {
			t.Fatalf("z = %v, want %v", z, want)
		}
	}
	t.Logf("scenario=Jacobi apply diag(2,3,4)^-1*[2 6 4] = %v", z)

	t.Run("works across formats", func(t *testing.T) {
		csc, _ := m.ToCSC()
		j2, err := NewJacobiPreconditioner(csc)
		if err != nil {
			t.Fatalf("Jacobi on CSC: %v", err)
		}
		z2, err := j2.Apply([]float64{2, 6, 4})
		if err != nil {
			t.Fatal(err)
		}
		for i := range z {
			if z2[i] != z[i] {
				t.Fatalf("CSC-based Jacobi %v != COO-based %v", z2, z)
			}
		}
		t.Logf("scenario=Jacobi built from CSC agrees with COO: %v", z2)
	})

	t.Run("dimension mismatch on apply", func(t *testing.T) {
		_, err := j.Apply([]float64{1, 2})
		if !errors.Is(err, ErrDimensionMismatch) {
			t.Fatalf("want ErrDimensionMismatch, got %v", err)
		}
		t.Logf("scenario=Jacobi apply wrong length rejected: %v", err)
	})
}

func TestJacobiPreconditionerRejects(t *testing.T) {
	t.Run("zero diagonal is singular", func(t *testing.T) {
		m := diagCOO(t, 3, []float64{2, 0, 4})
		_, err := NewJacobiPreconditioner(m)
		if !errors.Is(err, ErrSingularMatrix) {
			t.Fatalf("want ErrSingularMatrix, got %v", err)
		}
		t.Logf("scenario=zero diagonal rejected: %v", err)
	})
	t.Run("non-square matrix rejected", func(t *testing.T) {
		// 2x3 matrix with nonzero entries on the "diagonal".
		m, err := NewCOO(2, 3, []int{0, 1}, []int{0, 1}, []float64{2, 3})
		if err != nil {
			t.Fatal(err)
		}
		_, err = NewJacobiPreconditioner(m)
		if !errors.Is(err, ErrDimensionMismatch) {
			t.Fatalf("want ErrDimensionMismatch, got %v", err)
		}
		t.Logf("scenario=non-square Jacobi rejected: %v", err)
	})
	t.Run("NaN r rejected", func(t *testing.T) {
		j, _ := NewJacobiPreconditioner(diagCOO(t, 2, []float64{1, 1}))
		_, err := j.Apply([]float64{1, math.NaN()})
		if !errors.Is(err, ErrNonFiniteValue) {
			t.Fatalf("want ErrNonFiniteValue, got %v", err)
		}
		t.Logf("scenario=NaN residual rejected: %v", err)
	})
}

func TestIdentityPreconditioner(t *testing.T) {
	p := NewIdentityPreconditioner()
	in := []float64{3, -1, 2}
	z, err := p.Apply(in)
	if err != nil {
		t.Fatal(err)
	}
	in[0] = 999 // mutating the input must not affect the returned copy
	if z[0] != 3 {
		t.Fatalf("Apply must copy input, got z=%v", z)
	}
	t.Logf("scenario=identity returns independent copy: %v", z)
}

func TestChainPreconditioner(t *testing.T) {
	m := sampleCOO(t)
	j, err := NewJacobiPreconditioner(m)
	if err != nil {
		t.Fatal(err)
	}
	id := NewIdentityPreconditioner()

	t.Run("empty chain rejected", func(t *testing.T) {
		_, err := NewChainPreconditioner()
		if !errors.Is(err, ErrInvalidStructure) {
			t.Fatalf("want ErrInvalidStructure, got %v", err)
		}
		t.Logf("scenario=empty chain rejected: %v", err)
	})
	t.Run("nil member rejected", func(t *testing.T) {
		_, err := NewChainPreconditioner(id, nil)
		if !errors.Is(err, ErrInvalidStructure) {
			t.Fatalf("want ErrInvalidStructure, got %v", err)
		}
		t.Logf("scenario=nil chain member rejected: %v", err)
	})
	t.Run("chain(id, Jacobi) equals Jacobi and is deterministic", func(t *testing.T) {
		ch, err := NewChainPreconditioner(id, j)
		if err != nil {
			t.Fatal(err)
		}
		r := []float64{4, -3, 8}
		var first []float64
		for k := 0; k < 5; k++ {
			z, err := ch.Apply(r)
			if err != nil {
				t.Fatal(err)
			}
			if first == nil {
				first = z
			} else {
				for i := range first {
					if z[i] != first[i] {
						t.Fatalf("chain apply not deterministic at call %d: %v vs %v", k, z, first)
					}
				}
			}
		}
		want, _ := j.Apply(r)
		for i := range want {
			if first[i] != want[i] {
				t.Fatalf("chain(id,J) = %v, want J = %v", first, want)
			}
		}
		t.Logf("scenario=chain(id, Jacobi) repeated 5x identical = %v", first)
	})
}
