package sparse

import "math"

// CSC is a sparse matrix in compressed sparse column format.
//
// colPtr has length cols+1 with colPtr[0]==0; entries of column c are stored
// at positions [colPtr[c], colPtr[c+1]) with strictly increasing, unique row
// indices. NewCSC validates all of these properties.
type CSC struct {
	rows, cols int
	colPtr     []int
	rowInd     []int
	data       []float64
}

// NewCSC constructs a CSC matrix and validates its structure. Input slices
// are copied.
func NewCSC(rows, cols int, colPtr, rowInd []int, data []float64) (*CSC, error) {
	const op = "NewCSC"
	if rows <= 0 || cols <= 0 {
		return nil, structureError(op, ErrInvalidStructure,
			"dimensions must be positive")
	}
	if colPtr == nil || rowInd == nil || data == nil {
		return nil, structureError(op, ErrInvalidStructure,
			"index/value slices must not be nil")
	}
	if len(colPtr) != cols+1 {
		return nil, structureError(op, ErrInvalidStructure,
			"colPtr must have length cols+1")
	}
	if len(rowInd) != len(data) {
		return nil, structureError(op, ErrInvalidStructure,
			"rowInd and data must have equal length")
	}
	nnz := len(data)
	if colPtr[0] != 0 || colPtr[cols] != nnz {
		return nil, structureError(op, ErrInvalidStructure,
			"colPtr must start at 0 and end at the number of entries")
	}
	m := &CSC{
		rows:   rows,
		cols:   cols,
		colPtr: make([]int, cols+1),
		rowInd: make([]int, nnz),
		data:   make([]float64, nnz),
	}
	copy(m.colPtr, colPtr)
	for c := 0; c < cols; c++ {
		start, end := colPtr[c], colPtr[c+1]
		if start < 0 || end < start || end > nnz {
			return nil, structureError(op, ErrInvalidStructure,
				"colPtr slices must be non-decreasing and in range")
		}
		for k := start; k < end; k++ {
			r := rowInd[k]
			if r < 0 || r >= rows {
				return nil, entryError(op, ErrIndexOutOfRange, k, r, c,
					"row index outside matrix dimensions")
			}
			if k > start && r <= rowInd[k-1] {
				kind := ErrInvalidStructure
				reason := "row indices within a column must be strictly increasing"
				if r == rowInd[k-1] {
					kind = ErrDuplicateEntry
					reason = "duplicate row index within a column"
				}
				return nil, entryError(op, kind, k, r, c, reason)
			}
			v := data[k]
			if math.IsNaN(v) || math.IsInf(v, 0) {
				return nil, entryError(op, ErrNonFiniteValue, k, r, c,
					"entry value is NaN or Inf")
			}
			m.rowInd[k], m.data[k] = r, v
		}
	}
	return m, nil
}

func (m *CSC) Dims() (int, int) { return m.rows, m.cols }
func (m *CSC) NNZ() int         { return len(m.data) }

// ColPointers returns a copy of the column pointer array.
func (m *CSC) ColPointers() []int { return append([]int(nil), m.colPtr...) }

// RowIndices returns a copy of the row index array.
func (m *CSC) RowIndices() []int { return append([]int(nil), m.rowInd...) }

// Data returns a copy of the value array.
func (m *CSC) Data() []float64 { return append([]float64(nil), m.data...) }

// MatVec computes y = A*x in O(nnz).
func (m *CSC) MatVec(x []float64) ([]float64, error) {
	const op = "CSC.MatVec"
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
	for c := 0; c < m.cols; c++ {
		xc := x[c]
		for k := m.colPtr[c]; k < m.colPtr[c+1]; k++ {
			y[m.rowInd[k]] += m.data[k] * xc
		}
	}
	for i, v := range y {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return nil, fmtNonFinite(op, "y", i)
		}
	}
	return y, nil
}

// ToCOO expands the compressed storage into triplets.
func (m *CSC) ToCOO() (*COO, error) {
	nnz := len(m.data)
	rows := make([]int, nnz)
	cols := make([]int, nnz)
	data := make([]float64, nnz)
	i := 0
	for c := 0; c < m.cols; c++ {
		for k := m.colPtr[c]; k < m.colPtr[c+1]; k++ {
			rows[i], cols[i], data[i] = m.rowInd[k], c, m.data[k]
			i++
		}
	}
	return NewCOO(m.rows, m.cols, rows, cols, data)
}

// ToCSR converts through the triplet representation.
func (m *CSC) ToCSR() (*CSR, error) {
	t, err := m.ToCOO()
	if err != nil {
		return nil, err
	}
	return t.ToCSR()
}
