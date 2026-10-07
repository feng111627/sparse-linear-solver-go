package sparse

import (
	"errors"
	"math"
	"testing"
)

// dense converts any sparse matrix to a dense [rows][cols] slice for golden
// comparisons.
func toDense(t *testing.T, m Matrix) [][]float64 {
	t.Helper()
	r, c := m.Dims()
	d := make([][]float64, r)
	for i := range d {
		d[i] = make([]float64, c)
	}
	switch a := m.(type) {
	case *COO:
		for k := range a.data {
			d[a.row[k]][a.col[k]] = a.data[k]
		}
	case *CSR:
		for i := 0; i < r; i++ {
			for k := a.rowPtr[i]; k < a.rowPtr[i+1]; k++ {
				d[i][a.colInd[k]] = a.data[k]
			}
		}
	case *CSC:
		for j := 0; j < c; j++ {
			for k := a.colPtr[j]; k < a.colPtr[j+1]; k++ {
				d[a.rowInd[k]][j] = a.data[k]
			}
		}
	default:
		t.Fatalf("unsupported matrix type %T", a)
	}
	return d
}

func denseEqual(a, b [][]float64) bool {
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

// sampleCOO is a 3x3 matrix with entries deliberately out of order and
// including a zero structural entry:
//
//	[ 2  0 -1]
//	[ 0  3  0]
//	[-1  0  4]
func sampleCOO(t *testing.T) *COO {
	t.Helper()
	m, err := NewCOO(3, 3,
		[]int{2, 0, 1, 0, 2},
		[]int{0, 2, 1, 0, 2},
		[]float64{-1, -1, 3, 2, 4})
	if err != nil {
		t.Fatalf("sampleCOO: %v", err)
	}
	return m
}

var goldenDense = [][]float64{
	{2, 0, -1},
	{0, 3, 0},
	{-1, 0, 4},
}

func TestNewCOORejectsInvalidStructures(t *testing.T) {
	cases := []struct {
		name    string
		rows    int
		cols    int
		rowIdx  []int
		colIdx  []int
		data    []float64
		wantErr error
	}{
		{"zero dimension", 0, 3, nil, nil, nil, ErrInvalidStructure},
		{"negative dimension", 2, -1, []int{}, []int{}, []float64{}, ErrInvalidStructure},
		{"length mismatch", 2, 2, []int{0, 1}, []int{0}, []float64{1, 2}, ErrInvalidStructure},
		{"nil slices", 2, 2, nil, nil, nil, ErrInvalidStructure},
		{"row index out of range", 2, 2, []int{0, 2}, []int{0, 1}, []float64{1, 2}, ErrIndexOutOfRange},
		{"negative row index", 2, 2, []int{-1, 1}, []int{0, 1}, []float64{1, 2}, ErrIndexOutOfRange},
		{"col index out of range", 2, 2, []int{0, 1}, []int{0, 5}, []float64{1, 2}, ErrIndexOutOfRange},
		{"duplicate coordinate", 2, 2, []int{0, 0}, []int{1, 1}, []float64{1, 2}, ErrDuplicateEntry},
		{"NaN value", 2, 2, []int{0, 1}, []int{0, 1}, []float64{1, math.NaN()}, ErrNonFiniteValue},
		{"+Inf value", 2, 2, []int{0, 1}, []int{0, 1}, []float64{1, math.Inf(1)}, ErrNonFiniteValue},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := NewCOO(tc.rows, tc.cols, tc.rowIdx, tc.colIdx, tc.data)
			if !errors.Is(err, tc.wantErr) {
				t.Fatalf("want error matching %v, got %v", tc.wantErr, err)
			}
			var se *StructureError
			if errors.As(err, &se) {
				t.Logf("scenario=%-24s rejected op=%s index=%d row=%d col=%d: %v",
					tc.name, se.Op, se.Index, se.Row, se.Col, err)
			} else {
				t.Logf("scenario=%-24s rejected: %v", tc.name, err)
			}
		})
	}
}

func TestNewCSRRejectsInvalidStructures(t *testing.T) {
	cases := []struct {
		name    string
		rows    int
		cols    int
		rowPtr  []int
		colInd  []int
		data    []float64
		wantErr error
	}{
		{"bad rowPtr length", 2, 2, []int{0, 1}, []int{0}, []float64{1}, ErrInvalidStructure},
		{"rowPtr not starting zero", 2, 2, []int{1, 1, 2}, []int{0, 1}, []float64{1, 2}, ErrInvalidStructure},
		{"rowPtr end != nnz", 2, 2, []int{0, 0, 0}, []int{0}, []float64{1}, ErrInvalidStructure},
		{"rowPtr decreasing", 2, 2, []int{0, 2, 1}, []int{0, 1}, []float64{1, 2}, ErrInvalidStructure},
		{"colInd out of range", 2, 2, []int{0, 1, 2}, []int{0, 9}, []float64{1, 2}, ErrIndexOutOfRange},
		{"unsorted columns", 2, 2, []int{0, 2, 2}, []int{1, 0}, []float64{1, 2}, ErrInvalidStructure},
		{"duplicate column in row", 2, 2, []int{0, 2, 2}, []int{0, 0}, []float64{1, 2}, ErrDuplicateEntry},
		{"NaN value", 2, 2, []int{0, 1, 2}, []int{0, 1}, []float64{math.NaN(), 1}, ErrNonFiniteValue},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := NewCSR(tc.rows, tc.cols, tc.rowPtr, tc.colInd, tc.data)
			if !errors.Is(err, tc.wantErr) {
				t.Fatalf("want error matching %v, got %v", tc.wantErr, err)
			}
			t.Logf("scenario=%-24s rejected: %v", tc.name, err)
		})
	}
}

func TestNewCSCRejectsInvalidStructures(t *testing.T) {
	// col0 stores rows {1} then col1 stores row {0}: as column-wise layout
	// that is valid; unsorted must be within one column:
	_, err := NewCSC(2, 2, []int{0, 2, 2}, []int{1, 0}, []float64{1, 2})
	if !errors.Is(err, ErrInvalidStructure) {
		t.Fatalf("unsorted rows: want ErrInvalidStructure, got %v", err)
	}
	t.Logf("scenario=unsorted rows in column rejected: %v", err)

	_, err = NewCSC(2, 2, []int{0, 2, 2}, []int{0, 0}, []float64{1, 2})
	if !errors.Is(err, ErrDuplicateEntry) {
		t.Fatalf("duplicate row: want ErrDuplicateEntry, got %v", err)
	}
	t.Logf("scenario=duplicate row in column rejected: %v", err)
}

func TestConversionsRoundTrip(t *testing.T) {
	coo := sampleCOO(t)
	csr, err := coo.ToCSR()
	if err != nil {
		t.Fatalf("COO->CSR: %v", err)
	}
	csc, err := coo.ToCSC()
	if err != nil {
		t.Fatalf("COO->CSC: %v", err)
	}
	formats := map[string]Matrix{
		"COO":          coo,
		"CSR from COO": csr,
		"CSC from COO": csc,
	}
	csr2, err := csc.ToCSR()
	if err != nil {
		t.Fatalf("CSC->CSR: %v", err)
	}
	csc2, err := csr.ToCSC()
	if err != nil {
		t.Fatalf("CSR->CSC: %v", err)
	}
	coo2, err := csr.ToCOO()
	if err != nil {
		t.Fatalf("CSR->COO: %v", err)
	}
	coo3, err := csc.ToCOO()
	if err != nil {
		t.Fatalf("CSC->COO: %v", err)
	}
	formats["CSR from CSC"] = csr2
	formats["CSC from CSR"] = csc2
	formats["COO from CSR"] = coo2
	formats["COO from CSC"] = coo3

	for name, m := range formats {
		t.Run(name, func(t *testing.T) {
			r, c := m.Dims()
			if r != 3 || c != 3 {
				t.Fatalf("dims = %dx%d, want 3x3", r, c)
			}
			if m.NNZ() != 5 {
				t.Fatalf("nnz = %d, want 5", m.NNZ())
			}
			d := toDense(t, m)
			if !denseEqual(d, goldenDense) {
				t.Fatalf("dense =\n%v\nwant\n%v", d, goldenDense)
			}
			t.Logf("scenario=%-16s dims=3x3 nnz=%d dense=%v", name, m.NNZ(), d)
		})
	}
}

func TestMatVecAcrossFormats(t *testing.T) {
	coo := sampleCOO(t)
	csr, _ := coo.ToCSR()
	csc, _ := coo.ToCSC()
	x := []float64{1, 2, 3}
	// golden:
	// row0: 2*1 - 3 = -1
	// row1: 3*2 = 6
	// row2: -1 + 4*3 = 11
	want := []float64{-1, 6, 11}
	for name, m := range map[string]Matrix{"COO": coo, "CSR": csr, "CSC": csc} {
		t.Run(name, func(t *testing.T) {
			y, err := m.MatVec(x)
			if err != nil {
				t.Fatalf("MatVec: %v", err)
			}
			for i := range want {
				if y[i] != want[i] {
					t.Fatalf("y = %v, want %v", y, want)
				}
			}
			t.Logf("scenario=%-4s matvec x=%v -> y=%v", name, x, y)
		})
	}

	t.Run("dimension mismatch", func(t *testing.T) {
		_, err := coo.MatVec([]float64{1, 2})
		if !errors.Is(err, ErrDimensionMismatch) {
			t.Fatalf("want ErrDimensionMismatch, got %v", err)
		}
		t.Logf("scenario=x length 2 for 3-col matrix rejected: %v", err)
	})
	t.Run("NaN input", func(t *testing.T) {
		_, err := csr.MatVec([]float64{1, math.NaN(), 3})
		if !errors.Is(err, ErrNonFiniteValue) {
			t.Fatalf("want ErrNonFiniteValue, got %v", err)
		}
		t.Logf("scenario=NaN x rejected: %v", err)
	})
	t.Run("constructor copies inputs", func(t *testing.T) {
		rows := []int{0, 1}
		cols := []int{0, 1}
		data := []float64{1, 2}
		m, err := NewCOO(2, 2, rows, cols, data)
		if err != nil {
			t.Fatal(err)
		}
		data[0] = 999
		if m.Data()[0] != 1 {
			t.Fatal("constructor must copy input slices")
		}
		t.Logf("scenario=input slice mutation after construct does not affect matrix: A00=%v", m.Data()[0])
	})
}
