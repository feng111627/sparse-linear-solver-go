package sparse

import (
	"errors"
	"fmt"
)

// Sentinel errors. Callers can distinguish failure classes with errors.Is,
// and inspect offending positions/values with errors.As against *StructureError.
var (
	// ErrInvalidStructure covers malformed arrays (length mismatch, nil, etc.).
	ErrInvalidStructure = errors.New("sparse: invalid matrix structure")
	// ErrIndexOutOfRange indicates an index outside [0, dims).
	ErrIndexOutOfRange = errors.New("sparse: index out of range")
	// ErrDuplicateEntry indicates the same (row, col) pair appears more than once.
	ErrDuplicateEntry = errors.New("sparse: duplicate matrix entry")
	// ErrDimensionMismatch indicates incompatible vector/matrix dimensions.
	ErrDimensionMismatch = errors.New("sparse: dimension mismatch")
	// ErrNonFiniteValue indicates a NaN or +-Inf value in input or computation.
	ErrNonFiniteValue = errors.New("sparse: non-finite value")
	// ErrSingularMatrix indicates a zero pivot or a preconditioner that cannot be formed.
	ErrSingularMatrix = errors.New("sparse: singular or non-invertible matrix")
	// ErrBreakdown indicates the Krylov iteration broke down (e.g. zero denominator).
	ErrBreakdown = errors.New("sparse: solver breakdown")
	// ErrNotConverged indicates the iteration limit was reached before tolerance.
	ErrNotConverged = errors.New("sparse: solver did not converge")
	// ErrUnsupportedPreconditioner indicates a preconditioner unsuitable for the matrix.
	ErrUnsupportedPreconditioner = errors.New("sparse: unsupported preconditioner")
)

// StructureError wraps a structure-related sentinel with details such as the
// offending entry position. errors.Is(err, ErrDuplicateEntry) still succeeds.
type StructureError struct {
	Op     string // operation that detected the problem, e.g. "NewCOO"
	Kind   error  // one of ErrInvalidStructure / ErrIndexOutOfRange / ErrDuplicateEntry
	Index  int    // entry position in the supplied triplet/value arrays (-1 if n/a)
	Row    int    // row coordinate of the offending entry (-1 if n/a)
	Col    int    // column coordinate of the offending entry (-1 if n/a)
	Reason string
}

func (e *StructureError) Error() string {
	msg := e.Op + ": " + e.Kind.Error()
	if e.Reason != "" {
		msg += ": " + e.Reason
	}
	if e.Index >= 0 {
		msg += fmt.Sprintf(" (entry %d", e.Index)
		if e.Row >= 0 || e.Col >= 0 {
			msg += fmt.Sprintf(", row=%d col=%d", e.Row, e.Col)
		}
		msg += ")"
	}
	return msg
}

func (e *StructureError) Unwrap() error { return e.Kind }

func structureError(op string, kind error, reason string) *StructureError {
	return &StructureError{Op: op, Kind: kind, Index: -1, Row: -1, Col: -1, Reason: reason}
}

func entryError(op string, kind error, index, row, col int, reason string) *StructureError {
	return &StructureError{Op: op, Kind: kind, Index: index, Row: row, Col: col, Reason: reason}
}

// dimensionError builds an error matching ErrDimensionMismatch with context.
func dimensionError(op string, format string, args ...any) error {
	return fmt.Errorf("%w: %s: %s", ErrDimensionMismatch, op, fmt.Sprintf(format, args...))
}
