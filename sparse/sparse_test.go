package sparse

import (
	"errors"
	"math"
	"testing"
)

func TestNewCOO_Valid(t *testing.T) {
	m, err := NewCOO(2, 3, []int{0, 1}, []int{1, 2}, []float64{1.5, -2})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if m.NumRows() != 2 || m.NumCols() != 3 || m.NNZ() != 2 {
		t.Fatalf("bad dims: %dx%d nnz=%d", m.NumRows(), m.NumCols(), m.NNZ())
	}
	t.Log("valid COO accepted")
}

func TestNewCOO_Duplicate(t *testing.T) {
	_, err := NewCOO(2, 2, []int{0, 0}, []int{1, 1}, []float64{1, 2})
	var dup *ErrDuplicateEntry
	if !errors.As(err, &dup) {
		t.Fatalf("expected ErrDuplicateEntry, got %v", err)
	}
	t.Logf("duplicate rejected: %v", err)
}

func TestNewCOO_IndexOutOfRange(t *testing.T) {
	_, err := NewCOO(2, 2, []int{0, 5}, []int{0, 0}, []float64{1, 2})
	var oor *ErrIndexOutOfRange
	if !errors.As(err, &oor) {
		t.Fatalf("expected ErrIndexOutOfRange, got %v", err)
	}
	t.Logf("out-of-range rejected: %v", err)
}

func TestNewCOO_LengthMismatch(t *testing.T) {
	_, err := NewCOO(2, 2, []int{0}, []int{0, 1}, []float64{1})
	var inv *ErrInvalidStructure
	if !errors.As(err, &inv) {
		t.Fatalf("expected ErrInvalidStructure, got %v", err)
	}
	t.Logf("length mismatch rejected: %v", err)
}

func TestNewCOO_NaNValue(t *testing.T) {
	_, err := NewCOO(1, 1, []int{0}, []int{0}, []float64{math.NaN()})
	var inv *ErrInvalidStructure
	if !errors.As(err, &inv) {
		t.Fatalf("expected ErrInvalidStructure, got %v", err)
	}
	t.Logf("NaN value rejected: %v", err)
}

func TestNewCSR_InvalidRowPtr(t *testing.T) {
	_, err := NewCSR(2, 2, []int{0, 1}, []int{0}, []float64{1})
	var inv *ErrInvalidStructure
	if !errors.As(err, &inv) {
		t.Fatalf("expected ErrInvalidStructure, got %v", err)
	}
	t.Logf("bad rowPtr rejected: %v", err)
}

func TestNewCSR_DuplicateInRow(t *testing.T) {
	_, err := NewCSR(1, 3, []int{0, 2}, []int{1, 1}, []float64{1, 2})
	var dup *ErrDuplicateEntry
	if !errors.As(err, &dup) {
		t.Fatalf("expected ErrDuplicateEntry, got %v", err)
	}
	t.Logf("CSR duplicate rejected: %v", err)
}

func TestNewCSC_IndexOutOfRange(t *testing.T) {
	_, err := NewCSC(2, 2, []int{0, 1, 1}, []int{3}, []float64{1})
	var oor *ErrIndexOutOfRange
	if !errors.As(err, &oor) {
		t.Fatalf("expected ErrIndexOutOfRange, got %v", err)
	}
	t.Logf("CSC out-of-range rejected: %v", err)
}

// 3x3 矩阵:
// [4 1 0]
// [1 3 0]
// [0 0 2]
func sampleCOO(t *testing.T) *COO {
	t.Helper()
	m, err := NewCOO(3, 3,
		[]int{2, 0, 1, 0, 1},
		[]int{2, 1, 1, 0, 0},
		[]float64{2, 1, 3, 4, 1})
	if err != nil {
		t.Fatalf("setup: %v", err)
	}
	return m
}

func denseOf(t *testing.T, m Matrix) [][]float64 {
	t.Helper()
	d := make([][]float64, m.NumRows())
	for i := range d {
		d[i] = make([]float64, m.NumCols())
	}
	for j := 0; j < m.NumCols(); j++ {
		e := make([]float64, m.NumCols())
		e[j] = 1
		col, err := MatVec(m, e)
		if err != nil {
			t.Fatalf("matvec: %v", err)
		}
		for i := range d {
			d[i][j] = col[i]
		}
	}
	return d
}

func equalDense(a, b [][]float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if len(a[i]) != len(b[i]) {
			return false
		}
		for j := range a[i] {
			if a[i][j] != b[i][j] {
				return false
			}
		}
	}
	return true
}

func TestConversions_RoundTrip(t *testing.T) {
	coo := sampleCOO(t)
	ref := denseOf(t, coo)

	csr, err := coo.ToCSR()
	if err != nil {
		t.Fatalf("ToCSR: %v", err)
	}
	if !equalDense(ref, denseOf(t, csr)) {
		t.Fatal("COO->CSR changed matrix content")
	}

	csc, err := coo.ToCSC()
	if err != nil {
		t.Fatalf("ToCSC: %v", err)
	}
	if !equalDense(ref, denseOf(t, csc)) {
		t.Fatal("COO->CSC changed matrix content")
	}

	csr2, err := csc.ToCSR()
	if err != nil {
		t.Fatalf("CSC->CSR: %v", err)
	}
	if !equalDense(ref, denseOf(t, csr2)) {
		t.Fatal("CSC->CSR changed matrix content")
	}

	csc2, err := csr.ToCSC()
	if err != nil {
		t.Fatalf("CSR->CSC: %v", err)
	}
	if !equalDense(ref, denseOf(t, csc2)) {
		t.Fatal("CSR->CSC changed matrix content")
	}

	coo2 := csr.ToCOO()
	if !equalDense(ref, denseOf(t, coo2)) {
		t.Fatal("CSR->COO changed matrix content")
	}
	t.Log("all format conversions preserve matrix content")
}

func TestMatVec_CSR(t *testing.T) {
	csr, err := sampleCOO(t).ToCSR()
	if err != nil {
		t.Fatalf("setup: %v", err)
	}
	y, err := MatVec(csr, []float64{1, 1, 1})
	if err != nil {
		t.Fatalf("matvec: %v", err)
	}
	want := []float64{5, 4, 2}
	for i := range want {
		if y[i] != want[i] {
			t.Fatalf("y[%d]=%v want %v", i, y[i], want[i])
		}
	}
	t.Logf("CSR matvec correct: %v", y)
}

func TestMatVec_DimensionMismatch(t *testing.T) {
	csr, err := sampleCOO(t).ToCSR()
	if err != nil {
		t.Fatalf("setup: %v", err)
	}
	_, err = MatVec(csr, []float64{1, 2})
	var dm *ErrDimensionMismatch
	if !errors.As(err, &dm) {
		t.Fatalf("expected ErrDimensionMismatch, got %v", err)
	}
	t.Logf("matvec dimension mismatch rejected: %v", err)
}
