package sparse

import (
	"math"
	"sort"
)

// COO is a sparse matrix in coordinate (triplet) format.
//
// Every triplet (row[k], col[k], data[k]) denotes one matrix entry. The
// constructor rejects out-of-range indices, duplicate coordinates and
// non-finite values, so a *COO value returned without error always holds a
// structurally valid matrix with exactly NNZ() stored entries.
type COO struct {
	rows, cols int
	row, col   []int
	data       []float64
}

// NewCOO constructs a coordinate matrix from parallel triplet arrays.
// The input slices are copied; callers may reuse or mutate them afterwards.
func NewCOO(rows, cols int, rowIdx, colIdx []int, data []float64) (*COO, error) {
	const op = "NewCOO"
	if rows <= 0 || cols <= 0 {
		return nil, structureError(op, ErrInvalidStructure,
			"dimensions must be positive")
	}
	if rowIdx == nil || colIdx == nil || data == nil {
		return nil, structureError(op, ErrInvalidStructure,
			"index/value slices must not be nil")
	}
	nnz := len(data)
	if len(rowIdx) != nnz || len(colIdx) != nnz {
		return nil, structureError(op, ErrInvalidStructure,
			"rowIdx, colIdx and data must have equal length")
	}
	m := &COO{
		rows: rows,
		cols: cols,
		row:  make([]int, nnz),
		col:  make([]int, nnz),
		data: make([]float64, nnz),
	}
	seen := make(map[[2]int]struct{}, nnz)
	for k := 0; k < nnz; k++ {
		r, c := rowIdx[k], colIdx[k]
		if r < 0 || r >= rows || c < 0 || c >= cols {
			return nil, entryError(op, ErrIndexOutOfRange, k, r, c,
				"index outside matrix dimensions")
		}
		v := data[k]
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return nil, entryError(op, ErrNonFiniteValue, k, r, c,
				"entry value is NaN or Inf")
		}
		key := [2]int{r, c}
		if _, dup := seen[key]; dup {
			return nil, entryError(op, ErrDuplicateEntry, k, r, c,
				"same coordinate appears more than once")
		}
		seen[key] = struct{}{}
		m.row[k], m.col[k], m.data[k] = r, c, v
	}
	return m, nil
}

func (m *COO) Dims() (int, int) { return m.rows, m.cols }
func (m *COO) NNZ() int         { return len(m.data) }

// RowIndices returns a copy of the row index array.
func (m *COO) RowIndices() []int { return append([]int(nil), m.row...) }

// ColIndices returns a copy of the column index array.
func (m *COO) ColIndices() []int { return append([]int(nil), m.col...) }

// Data returns a copy of the value array.
func (m *COO) Data() []float64 { return append([]float64(nil), m.data...) }

// MatVec computes y = A*x in O(nnz).
func (m *COO) MatVec(x []float64) ([]float64, error) {
	const op = "COO.MatVec"
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
	for k := range m.data {
		y[m.row[k]] += m.data[k] * x[m.col[k]]
	}
	for i, v := range y {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return nil, fmtNonFinite(op, "y", i)
		}
	}
	return y, nil
}

// sortedOrder returns indices k sorted by (row, col), ascending.
func (m *COO) sortedOrder() []int {
	order := make([]int, len(m.data))
	for k := range order {
		order[k] = k
	}
	sort.Slice(order, func(i, j int) bool {
		a, b := order[i], order[j]
		if m.row[a] != m.row[b] {
			return m.row[a] < m.row[b]
		}
		return m.col[a] < m.col[b]
	})
	return order
}

// ToCSR converts to compressed sparse row format.
func (m *COO) ToCSR() (*CSR, error) {
	order := m.sortedOrder()
	rowPtr := make([]int, m.rows+1)
	colInd := make([]int, len(m.data))
	data := make([]float64, len(m.data))
	for pos, k := range order {
		colInd[pos] = m.col[k]
		data[pos] = m.data[k]
		rowPtr[m.row[k]+1]++
	}
	for i := 1; i <= m.rows; i++ {
		rowPtr[i] += rowPtr[i-1]
	}
	return &CSR{
		rows:   m.rows,
		cols:   m.cols,
		rowPtr: rowPtr,
		colInd: colInd,
		data:   data,
	}, nil
}

// ToCSC converts to compressed sparse column format.
func (m *COO) ToCSC() (*CSC, error) {
	order := make([]int, len(m.data))
	for k := range order {
		order[k] = k
	}
	sort.Slice(order, func(i, j int) bool {
		a, b := order[i], order[j]
		if m.col[a] != m.col[b] {
			return m.col[a] < m.col[b]
		}
		return m.row[a] < m.row[b]
	})
	colPtr := make([]int, m.cols+1)
	rowInd := make([]int, len(m.data))
	data := make([]float64, len(m.data))
	for pos, k := range order {
		rowInd[pos] = m.row[k]
		data[pos] = m.data[k]
		colPtr[m.col[k]+1]++
	}
	for j := 1; j <= m.cols; j++ {
		colPtr[j] += colPtr[j-1]
	}
	return &CSC{
		rows:   m.rows,
		cols:   m.cols,
		colPtr: colPtr,
		rowInd: rowInd,
		data:   data,
	}, nil
}
