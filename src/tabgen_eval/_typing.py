"""Shared value annotations without importing numerical execution backends.

Domain-specific records belong with their owners. These aliases describe values
that cross multiple subsystem boundaries; none changes runtime validation.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, TypeAlias, TypeGuard

import numpy as np
from numpy.typing import NDArray

Precision: TypeAlias = Literal["float32", "float64"]
TaskKind: TypeAlias = Literal["classification", "regression"]
TaskChoice: TypeAlias = Literal["auto", "classification", "regression"]
UncertaintyMode: TypeAlias = Literal["none", "reference", "candidate", "both"]
# NumPy's bit-width parameter is intentionally open: SeedSequence accepts every
# signed/unsigned integer width, including platform-specific extended integers.
SeedScalar: TypeAlias = int | np.integer[Any]
SeedLike: TypeAlias = SeedScalar | Sequence["SeedLike"] | NDArray[np.integer[Any]]
SeedEntropy: TypeAlias = int | list["SeedEntropy"]
FloatArray: TypeAlias = NDArray[np.float32 | np.float64]
IntArray: TypeAlias = NDArray[np.int64]
IntegerArray: TypeAlias = NDArray[np.integer[Any]]
BoolArray: TypeAlias = NDArray[np.bool_]


def is_integer_scalar(value: object) -> TypeGuard[SeedScalar]:
    """Narrow Python/NumPy integer storage without losing its bit-width type."""
    return isinstance(value, (int, np.integer))
