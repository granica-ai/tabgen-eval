"""Finite-only means and guarded ratios used by Python metric calculations."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

NAN = float("nan")


def _finite(values: ArrayLike) -> NDArray[np.float64]:
    """Return the finite (non-NaN, non-inf) subset of *values* as a float array."""
    arr = np.asarray(values, dtype=float).ravel()
    return arr[np.isfinite(arr)]


def safe_nanmean(values: ArrayLike, default: float = NAN) -> float:
    """Mean of the finite entries of *values*; *default* if none are finite."""
    valid = _finite(values)
    if valid.size == 0:
        return default
    return float(np.mean(valid))


def safe_ratio(numerator: float, denominator: float) -> float:
    """``numerator / denominator`` with NaN on a zero/non-finite denominator."""
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator == 0.0:
        return NAN
    return float(numerator / denominator)
