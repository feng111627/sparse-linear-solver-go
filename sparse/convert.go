package sparse

// ToCSR 将 COO 转换为 CSR。行内元素按列索引升序排列，保证结果确定。
func (m *COO) ToCSR() (*CSR, error) {
	rowPtr := make([]int, m.Rows+1)
	for _, r := range m.RowIdx {
		rowPtr[r+1]++
	}
	for i := 0; i < m.Rows; i++ {
		rowPtr[i+1] += rowPtr[i]
	}
	n := len(m.Values)
	colIdx := make([]int, n)
	values := make([]float64, n)
	// 按 (row, col) 稳定排序填充：先按行桶，再在每个桶内按列插入排序。
	next := make([]int, m.Rows)
	copy(next, rowPtr[:m.Rows])
	order := make([]int, n)
	for i := range order {
		order[i] = i
	}
	// 简单插入排序，键为 (row, col)，数据量按 nnz 计；对重复元素已在构造时拒绝。
	for i := 1; i < n; i++ {
		for j := i; j > 0; j-- {
			a, b := order[j-1], order[j]
			if m.RowIdx[a] < m.RowIdx[b] ||
				(m.RowIdx[a] == m.RowIdx[b] && m.ColIdx[a] <= m.ColIdx[b]) {
				break
			}
			order[j-1], order[j] = order[j], order[j-1]
		}
	}
	for _, k := range order {
		r := m.RowIdx[k]
		p := next[r]
		colIdx[p] = m.ColIdx[k]
		values[p] = m.Values[k]
		next[r]++
	}
	return NewCSR(m.Rows, m.Cols, rowPtr, colIdx, values)
}

// ToCSC 将 COO 转换为 CSC（借助转置关系复用 ToCSR 逻辑）。
func (m *COO) ToCSC() (*CSC, error) {
	// 交换行列构造转置的 COO，再转 CSR，即得原矩阵的 CSC。
	tr, err := NewCOO(m.Cols, m.Rows, m.ColIdx, m.RowIdx, m.Values)
	if err != nil {
		return nil, err
	}
	tcsr, err := tr.ToCSR()
	if err != nil {
		return nil, err
	}
	return NewCSC(m.Rows, m.Cols, tcsr.RowPtr, tcsr.ColIdx, tcsr.Values)
}

// ToCOO 将 CSR 转换为 COO。
func (m *CSR) ToCOO() *COO {
	n := len(m.Values)
	rowIdx := make([]int, n)
	colIdx := make([]int, n)
	values := make([]float64, n)
	for i := 0; i < m.Rows; i++ {
		for k := m.RowPtr[i]; k < m.RowPtr[i+1]; k++ {
			rowIdx[k] = i
			colIdx[k] = m.ColIdx[k]
			values[k] = m.Values[k]
		}
	}
	return &COO{Rows: m.Rows, Cols: m.Cols, RowIdx: rowIdx, ColIdx: colIdx, Values: values}
}

// ToCSC 将 CSR 转换为 CSC。
func (m *CSR) ToCSC() (*CSC, error) {
	return m.ToCOO().ToCSC()
}

// ToCOO 将 CSC 转换为 COO。
func (m *CSC) ToCOO() *COO {
	n := len(m.Values)
	rowIdx := make([]int, n)
	colIdx := make([]int, n)
	values := make([]float64, n)
	for j := 0; j < m.Cols; j++ {
		for k := m.ColPtr[j]; k < m.ColPtr[j+1]; k++ {
			rowIdx[k] = m.RowIdx[k]
			colIdx[k] = j
			values[k] = m.Values[k]
		}
	}
	return &COO{Rows: m.Rows, Cols: m.Cols, RowIdx: rowIdx, ColIdx: colIdx, Values: values}
}

// ToCSR 将 CSC 转换为 CSR。
func (m *CSC) ToCSR() (*CSR, error) {
	return m.ToCOO().ToCSR()
}
