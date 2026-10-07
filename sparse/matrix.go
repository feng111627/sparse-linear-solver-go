package sparse

import "math"

// Matrix 是所有稀疏矩阵格式共同满足的只读接口。
type Matrix interface {
	// NumRows 返回行数。
	NumRows() int
	// NumCols 返回列数。
	NumCols() int
	// NNZ 返回存储的非零元素个数。
	NNZ() int
}

// COO 坐标格式：逐元素存储 (row, col, value)。
type COO struct {
	Rows, Cols int
	RowIdx     []int
	ColIdx     []int
	Values     []float64
}

func (m *COO) NumRows() int { return m.Rows }
func (m *COO) NumCols() int { return m.Cols }
func (m *COO) NNZ() int     { return len(m.Values) }

// CSR 压缩行存储格式。
type CSR struct {
	Rows, Cols int
	RowPtr     []int // 长度 Rows+1
	ColIdx     []int // 长度 NNZ
	Values     []float64
}

func (m *CSR) NumRows() int { return m.Rows }
func (m *CSR) NumCols() int { return m.Cols }
func (m *CSR) NNZ() int     { return len(m.Values) }

// CSC 压缩列存储格式。
type CSC struct {
	Rows, Cols int
	ColPtr     []int // 长度 Cols+1
	RowIdx     []int // 长度 NNZ
	Values     []float64
}

func (m *CSC) NumRows() int { return m.Rows }
func (m *CSC) NumCols() int { return m.Cols }
func (m *CSC) NNZ() int     { return len(m.Values) }

func checkDims(rows, cols int) error {
	if rows < 0 || cols < 0 {
		return &ErrInvalidStructure{Detail: "negative matrix dimensions"}
	}
	return nil
}

func checkValue(v float64, row, col int) error {
	if math.IsNaN(v) || math.IsInf(v, 0) {
		return &ErrInvalidStructure{Detail: "non-finite value (NaN/Inf) in matrix entry"}
	}
	return nil
}

// NewCOO 构造并校验一个 COO 矩阵。重复元素、越界索引、长度不一致都会报错。
func NewCOO(rows, cols int, rowIdx, colIdx []int, values []float64) (*COO, error) {
	if err := checkDims(rows, cols); err != nil {
		return nil, err
	}
	if len(rowIdx) != len(colIdx) || len(rowIdx) != len(values) {
		return nil, &ErrInvalidStructure{Detail: "rowIdx, colIdx and values must have equal length"}
	}
	seen := make(map[[2]int]struct{}, len(values))
	for k := range values {
		r, c := rowIdx[k], colIdx[k]
		if r < 0 || r >= rows || c < 0 || c >= cols {
			return nil, &ErrIndexOutOfRange{Row: r, Col: c, NumRows: rows, NumCols: cols}
		}
		key := [2]int{r, c}
		if _, dup := seen[key]; dup {
			return nil, &ErrDuplicateEntry{Row: r, Col: c}
		}
		seen[key] = struct{}{}
		if err := checkValue(values[k], r, c); err != nil {
			return nil, err
		}
	}
	return &COO{Rows: rows, Cols: cols, RowIdx: rowIdx, ColIdx: colIdx, Values: values}, nil
}

// NewCSR 构造并校验一个 CSR 矩阵。
func NewCSR(rows, cols int, rowPtr, colIdx []int, values []float64) (*CSR, error) {
	if err := checkDims(rows, cols); err != nil {
		return nil, err
	}
	if len(rowPtr) != rows+1 {
		return nil, &ErrInvalidStructure{Detail: "rowPtr length must be rows+1"}
	}
	if len(colIdx) != len(values) {
		return nil, &ErrInvalidStructure{Detail: "colIdx and values must have equal length"}
	}
	if rowPtr[0] != 0 || rowPtr[rows] != len(values) {
		return nil, &ErrInvalidStructure{Detail: "rowPtr must start at 0 and end at nnz"}
	}
	for i := 0; i < rows; i++ {
		if rowPtr[i] > rowPtr[i+1] {
			return nil, &ErrInvalidStructure{Detail: "rowPtr must be non-decreasing"}
		}
		seen := make(map[int]struct{}, rowPtr[i+1]-rowPtr[i])
		for k := rowPtr[i]; k < rowPtr[i+1]; k++ {
			c := colIdx[k]
			if c < 0 || c >= cols {
				return nil, &ErrIndexOutOfRange{Row: i, Col: c, NumRows: rows, NumCols: cols}
			}
			if _, dup := seen[c]; dup {
				return nil, &ErrDuplicateEntry{Row: i, Col: c}
			}
			seen[c] = struct{}{}
			if err := checkValue(values[k], i, c); err != nil {
				return nil, err
			}
		}
	}
	return &CSR{Rows: rows, Cols: cols, RowPtr: rowPtr, ColIdx: colIdx, Values: values}, nil
}

// NewCSC 构造并校验一个 CSC 矩阵。
func NewCSC(rows, cols int, colPtr, rowIdx []int, values []float64) (*CSC, error) {
	if err := checkDims(rows, cols); err != nil {
		return nil, err
	}
	if len(colPtr) != cols+1 {
		return nil, &ErrInvalidStructure{Detail: "colPtr length must be cols+1"}
	}
	if len(rowIdx) != len(values) {
		return nil, &ErrInvalidStructure{Detail: "rowIdx and values must have equal length"}
	}
	if colPtr[0] != 0 || colPtr[cols] != len(values) {
		return nil, &ErrInvalidStructure{Detail: "colPtr must start at 0 and end at nnz"}
	}
	for j := 0; j < cols; j++ {
		if colPtr[j] > colPtr[j+1] {
			return nil, &ErrInvalidStructure{Detail: "colPtr must be non-decreasing"}
		}
		seen := make(map[int]struct{}, colPtr[j+1]-colPtr[j])
		for k := colPtr[j]; k < colPtr[j+1]; k++ {
			r := rowIdx[k]
			if r < 0 || r >= rows {
				return nil, &ErrIndexOutOfRange{Row: r, Col: j, NumRows: rows, NumCols: cols}
			}
			if _, dup := seen[r]; dup {
				return nil, &ErrDuplicateEntry{Row: r, Col: j}
			}
			seen[r] = struct{}{}
			if err := checkValue(values[k], r, j); err != nil {
				return nil, err
			}
		}
	}
	return &CSC{Rows: rows, Cols: cols, ColPtr: colPtr, RowIdx: rowIdx, Values: values}, nil
}
