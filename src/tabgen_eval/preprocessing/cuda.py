"""Lazy CUDA backend for the shared, model-independent preprocessing contract.

Callers reserve source, output and block-reduction storage before invoking these
routines. Tensor buffers remain on their input device; Python orchestrates the
compiled operations and small fitted metadata only.
"""

from tabgen_eval.execution.torch.cuda.triton.preprocessing import (
    apply_numeric,
    assemble,
    count_missing,
    fit_categories,
    fit_numeric,
    map_categories,
)

__all__ = [
    "fit_numeric",
    "apply_numeric",
    "fit_categories",
    "map_categories",
    "count_missing",
    "assemble",
]
