"""Bounded compiled CPU nearest-neighbor reductions for fitted compact geometry."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tabgen_eval.execution.torch.distance.ops import _validate

if TYPE_CHECKING:
    from torch import Tensor

    from .geometry import Features


def nearest_distances_cpu(
    reference: Features,
    query: Features,
    *,
    exclude_self: bool = False,
    threads: int = 1,
    scratch_bytes: int = 64 * 1024**2,
) -> Tensor:
    """Return Euclidean minima per query using a single native call.

    Inputs are finite contiguous CPU ``Features`` in shared FP32/FP64 precision.
    Categorical identities use the fitted reference vocabulary and ``-1`` for an
    unknown block. Inputs remain read-only. ``exclude_self`` masks equal row
    positions for aligned same-table inputs; other duplicate records are retained.

    ``scratch_bytes`` bounds packed reference coordinates and per-worker SIMD
    state, excluding inputs and the returned vector. Empty references produce
    infinity. Insufficient workspace raises before packing or distance work.
    """
    a, b = reference.numeric, query.numeric
    ca, cb = reference.categorical, query.categorical
    _validate(a, b, ca, cb)
    if a.is_cuda:
        raise ValueError("compiled CPU nearest distances require CPU inputs")
    from tabgen_eval.execution.torch.cpu.extension import library

    return library().nearest_distances(a, b, ca, cb, exclude_self, threads, scratch_bytes)
