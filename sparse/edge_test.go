package sparse

import (
	"errors"
	"math"
	"testing"
)

func TestAccessorsReturnCopies(t *testing.T) {
	coo := sampleCOO(t)
	csr, _ := coo.ToCSR()
	csc, _ := coo.ToCSC()

	ri, ci, da := coo.RowIndices(), coo.ColIndices(), coo.Data()
	ri[0] = 99
	if coo.RowIndices()[0] == 99 {
		t.Fatal("RowIndices must return a copy")
	}
	ci[0] = 99
	if coo.ColIndices()[0] == 99 {
		t.Fatal("ColIndices must return a copy")
	}
	da[0] = 99
	if coo.Data()[0] == 99 {
		t.Fatal("Data must return a copy")
	}

	rp, cind, cd := csr.RowPointers(), csr.ColIndices(), csr.Data()
	rp[0] = 99
	cind[0] = 99
	cd[0] = 99
	if csr.RowPointers()[0] == 99 || csr.ColIndices()[0] == 99 || csr.Data()[0] == 99 {
		t.Fatal("CSR accessors must return copies")
	}

	cp, rid, cscd := csc.ColPointers(), csc.RowIndices(), csc.Data()
	cp[0] = 99
	rid[0] = 99
	cscd[0] = 99
	if csc.ColPointers()[0] == 99 || csc.RowIndices()[0] == 99 || csc.Data()[0] == 99 {
		t.Fatal("CSC accessors must return copies")
	}
	t.Log("scenario=all accessors return defensive copies")
}

func TestCSCValidationBranches(t *testing.T) {
	cases := []struct {
		name    string
		colPtr  []int
		rowInd  []int
		data    []float64
		wantErr error
	}{
		{"bad colPtr length", []int{0, 1}, []int{0}, []float64{1}, ErrInvalidStructure},
		{"colPtr end != nnz", []int{0, 0, 0}, []int{0}, []float64{1}, ErrInvalidStructure},
		{"rowInd out of range", []int{0, 1, 1}, []int{9}, []float64{1}, ErrIndexOutOfRange},
		{"NaN value", []int{0, 1, 1}, []int{0}, []float64{math.NaN()}, ErrNonFiniteValue},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := NewCSC(2, 2, tc.colPtr, tc.rowInd, tc.data)
			if !errors.Is(err, tc.wantErr) {
				t.Fatalf("want %v, got %v", tc.wantErr, err)
			}
			t.Logf("scenario=%-22s rejected: %v", tc.name, err)
		})
	}
}

func TestMatVecOverflowDetected(t *testing.T) {
	// Finite matrix and finite x whose product overflows: must be reported as
	// a non-finite result, not silently returned as +-Inf.
	m := diagCOO(t, 1, []float64{1e300})
	for _, mx := range []Matrix{m} {
		_, err := mx.MatVec([]float64{1e300})
		if !errors.Is(err, ErrNonFiniteValue) {
			t.Fatalf("want ErrNonFiniteValue on overflow, got %v", err)
		}
		t.Logf("scenario=finite inputs with overflowing product rejected: %v", err)
	}
}

// wrapperMatrix hides the concrete *COO/*CSR/*CSC types so Jacobi construction
// exercises the generic extractDiagonal fallback (A*e_j column probing).
type wrapperMatrix struct{ m Matrix }

func (w *wrapperMatrix) Dims() (int, int)                      { return w.m.Dims() }
func (w *wrapperMatrix) NNZ() int                              { return w.m.NNZ() }
func (w *wrapperMatrix) MatVec(x []float64) ([]float64, error) { return w.m.MatVec(x) }

func TestJacobiGenericMatrixFallback(t *testing.T) {
	base := sampleCOO(t)
	j1, err := NewJacobiPreconditioner(base)
	if err != nil {
		t.Fatal(err)
	}
	j2, err := NewJacobiPreconditioner(&wrapperMatrix{m: base})
	if err != nil {
		t.Fatalf("generic Matrix Jacobi: %v", err)
	}
	r := []float64{4, -9, 8}
	z1, _ := j1.Apply(r)
	z2, _ := j2.Apply(r)
	for i := range z1 {
		if z1[i] != z2[i] {
			t.Fatalf("generic fallback %v != concrete %v", z2, z1)
		}
	}
	t.Logf("scenario=Jacobi over generic Matrix interface agrees: %v", z2)
}

func TestSolveCGNilRHS(t *testing.T) {
	coo := laplacian1D(t, 3)
	_, err := SolveCG(coo, nil, CGOptions{})
	if !errors.Is(err, ErrInvalidStructure) {
		t.Fatalf("want ErrInvalidStructure for nil b, got %v", err)
	}
	t.Logf("scenario=nil right-hand side rejected: %v", err)
}

func TestStructureErrorUnwrap(t *testing.T) {
	_, err := NewCOO(1, 1, []int{2}, []int{0}, []float64{1})
	var se *StructureError
	if !errors.As(err, &se) {
		t.Fatalf("expected *StructureError, got %T: %v", err, err)
	}
	if !errors.Is(err, ErrIndexOutOfRange) || se.Index != 0 || se.Row != 2 {
		t.Fatalf("unexpected structure error detail: %+v", se)
	}
	t.Logf("scenario=distinguishable error detail available: op=%s kind=%s index=%d row=%d col=%d",
		se.Op, se.Kind, se.Index, se.Row, se.Col)
}
