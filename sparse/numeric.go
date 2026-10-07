package sparse

import "fmt"

// fmtNonFinite builds an ErrNonFiniteValue error naming the offending vector.
func fmtNonFinite(op, vec string, index int) error {
	return fmt.Errorf("%w: %s: %s[%d] is NaN or Inf",
		ErrNonFiniteValue, op, vec, index)
}
