package sparse

// Matrix is the common interface implemented by all sparse storage formats.
type Matrix interface {
	// Dims returns the number of rows and columns.
	Dims() (rows, cols int)
	// NNZ returns the number of structurally stored (non-assembled) entries.
	NNZ() int
	// MatVec computes y = A * x.
	MatVec(x []float64) ([]float64, error)
}
