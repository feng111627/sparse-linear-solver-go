package sparse

import "math"

// CSR is a sparse matrix in compressed sparse row format.
//
// rowPtr has length rows+1 with rowPtr[0]==0; entries of row r are stored at
// positions [rowPtr[r], rowPtr[r+1]) with strictly increasing, unique column
// indices. NewCSR validates all of these properties.
type CSR struct {
	rows, cols int
	rowPtr     []int
	colInd     []int
	data       []float64
}

// NewCSR constructs a CSR matrix and validates its structure. Input slices
// are copied.
func NewCSR(rows, cols int, rowPtr, colInd []int, data []float64) (*CSR, error) {
	const op = "NewCSR"
	if rows <= 0 || cols <= 0 {
		return nil, structureError(op, ErrInvalidStructure,
			"dimensions must be positive")
	}
	if rowPtr == nil || colInd == nil || data == nil {
		return nil, structureError(op, ErrInvalidStructure,
			"index/value slices must not be nil")
	}
	if len(rowPtr) != rows+1 {
		return nil, structureError(op, ErrInvalidStructure,
			"rowPtr must have length rows+1")
	}
	if len(colInd) != len(data) {
		return nil, structureError(op, ErrInvalidStructure,
			"colInd and data must have equal length")
	}
	nnz := len(data)
	if rowPtr[0] != 0 || rowPtr[rows] != nnz {
		return nil, structureError(op, ErrInvalidStructure,
			"rowPtr must start at 0 and end at the number of entries")
	}
	m := &CSR{
		rows:   rows,
		cols:   cols,
		rowPtr: make([]int, rows+1),
		colInd: make([]int, nnz),
		data:   make([]float64, nnz),
	}
	copy(m.rowPtr, rowPtr)
	for r := 0; r < rows; r++ {
		start, end := rowPtr[r], rowPtr[r+1]
		if start < 0 || end < start || end > nnz {
			return nil, structureError(op, ErrInvalidStructure,
				"rowPtr slices must be non-decreasing and in range")
		}
		for k := start; k < end; k++ {
			c := colInd[k]
			if c < 0 || c >= cols {
				return nil, entryError(op, ErrIndexOutOfRange, k, r, c,
					"column index outside matrix dimensions")
			}
			if k > start && c <= colInd[k-1] {
				kind := ErrInvalidStructure
				reason := "column indices within a row must be strictly increasing"
				if c == colInd[k-1] {
					kind = ErrDuplicateEntry
					reason = "duplicate column index within a row"
				}
				return nil, entryError(op, kind, k, r, c, reason)
			}
			v := data[k]
			if math.IsNaN(v) || math.IsInf(v, 0) {
				return nil, entryError(op, ErrNonFiniteValue, k, r, c,
					"entry value is NaN or Inf")
			}
			m.colInd[k], m.data[k] = c, v
		}
	}
	return m, nil
}

func (m *CSR) Dims() (int, int) { return m.rows, m.cols }
func (m *CSR) NNZ() int         { return len(m.data) }

// RowPointers returns a copy of the row pointer array.
func (m *CSR) RowPointers() []int { return append([]int(nil), m.rowPtr...) }

// ColIndices returns a copy of the column index array.
func (m *CSR) ColIndices() []int { return append([]int(nil), m.colInd...) }

// Data returns a copy of the value array.
func (m *CSR) Data() []float64 { return append([]float64(nil), m.data...) }

// MatVec computes y = A*x in O(nnz).
func (m *CSR) MatVec(x []float64) ([]float64, error) {
	const op = "CSR.MatVec"
	if len(x) != m.cols {
		return nil, dimensionError(op, "matrix is %dx%d but x has length %d",
			m.rows, m.cols, len(x))
	}
	for i, v := range x {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return nil, fmtNonFinite(op, "x", i)
		}
	}
	y := make([]float64, m.rows)
	for r := 0; r < m.rows; r++ {
		s := 0.0
		for k := m.rowPtr[r]; k < m.rowPtr[r+1]; k++ {
			s += m.data[k] * x[m.colInd[k]]
		}
		y[r] = s
	}
	for i, v := range y {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return nil, fmtNonFinite(op, "y", i)
		}
	}
	return y, nil
}

// ToCOO expands the compressed storage into triplets.
func (m *CSR) ToCOO() (*COO, error) {
	nnz := len(m.data)
	rows := make([]int, nnz)
	cols := make([]int, nnz)
	data := make([]float64, nnz)
	i := 0
	for r := 0; r < m.rows; r++ {
		for k := m.rowPtr[r]; k < m.rowPtr[r+1]; k++ {
			rows[i], cols[i], data[i] = r, m.colInd[k], m.data[k]
			i++
		}
	}
	// Rebuild via NewCOO so the invariant of COO is guaranteed centrally.
	return NewCOO(m.rows, m.cols, rows, cols, data)
}

// ToCSC transposes the compressed layout via a triplet intermediate.
func (m *CSR) ToCSC() (*CSC, error) {
	t, err := m.ToCOO()
	if err != nil {
		return nil, err
	}
	return t.ToCSC()
}
