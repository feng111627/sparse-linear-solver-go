package sparse

// MatVec 计算 y = A*x。维度不匹配时返回 *ErrDimensionMismatch。
func MatVec(m Matrix, x []float64) ([]float64, error) {
	if len(x) != m.NumCols() {
		return nil, &ErrDimensionMismatch{
			What:   "matvec",
			RowsA:  m.NumRows(), ColsA: m.NumCols(),
			RowsB: len(x), ColsB: 1,
		}
	}
	y := make([]float64, m.NumRows())
	switch a := m.(type) {
	case *CSR:
		for i := 0; i < a.Rows; i++ {
			sum := 0.0
			for k := a.RowPtr[i]; k < a.RowPtr[i+1]; k++ {
				sum += a.Values[k] * x[a.ColIdx[k]]
			}
			y[i] = sum
		}
	case *CSC:
		for j := 0; j < a.Cols; j++ {
			xj := x[j]
			for k := a.ColPtr[j]; k < a.ColPtr[j+1]; k++ {
				y[a.RowIdx[k]] += a.Values[k] * xj
			}
		}
	case *COO:
		for k := range a.Values {
			y[a.RowIdx[k]] += a.Values[k] * x[a.ColIdx[k]]
		}
	default:
		return nil, &ErrInvalidStructure{Detail: "unsupported matrix type for matvec"}
	}
	return y, nil
}
