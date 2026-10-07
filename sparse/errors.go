// Package sparse 提供稀疏矩阵的多种存储格式、校验与相互转换。
package sparse

import "fmt"

// ErrDuplicateEntry 表示同一 (row, col) 位置出现重复元素。
type ErrDuplicateEntry struct {
	Row, Col int
}

func (e *ErrDuplicateEntry) Error() string {
	return fmt.Sprintf("sparse: duplicate entry at (%d, %d)", e.Row, e.Col)
}

// ErrIndexOutOfRange 表示行或列索引越界。
type ErrIndexOutOfRange struct {
	Row, Col       int
	NumRows, NumCols int
}

func (e *ErrIndexOutOfRange) Error() string {
	return fmt.Sprintf("sparse: index (%d, %d) out of range for %dx%d matrix",
		e.Row, e.Col, e.NumRows, e.NumCols)
}

// ErrDimensionMismatch 表示矩阵/向量维度不匹配。
type ErrDimensionMismatch struct {
	What             string
	RowsA, ColsA     int
	RowsB, ColsB     int
}

func (e *ErrDimensionMismatch) Error() string {
	return fmt.Sprintf("sparse: dimension mismatch in %s: (%dx%d) vs (%dx%d)",
		e.What, e.RowsA, e.ColsA, e.RowsB, e.ColsB)
}

// ErrInvalidStructure 表示存储格式内部结构不合法（如指针数组长度错误）。
type ErrInvalidStructure struct {
	Detail string
}

func (e *ErrInvalidStructure) Error() string {
	return fmt.Sprintf("sparse: invalid structure: %s", e.Detail)
}
