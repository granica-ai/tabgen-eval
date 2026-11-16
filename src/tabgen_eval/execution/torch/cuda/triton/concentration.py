"""Exact multi-radius local masses with bounded register tiles on CUDA."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

from .distance import _distances


@triton.jit
def _count(
    A,
    C,
    THRESHOLDS,
    COUNTS,
    N,
    PN: tl.constexpr,
    PC: tl.constexpr,
    FP64: tl.constexpr,
    RADII: tl.constexpr,
):
    """Visit upper-triangular pairs and add each qualifying pair to both centers."""
    if tl.program_id(0) * 8 <= tl.program_id(1) * 128 + 127:
        q, r, distance = _distances(A, A, C, C, N, N, PN, PC, FP64)
        valid = (q[:, None] < N) & (r[None, :] < N) & (r[None, :] > q[:, None])
        for index in range(RADII):
            threshold = tl.load(THRESHOLDS + index)
            inside = valid & (distance <= threshold)
            tl.atomic_add(
                COUNTS + index * N + q, tl.sum(inside.to(tl.int32), axis=1), q < N, sem="relaxed"
            )
            tl.atomic_add(
                COUNTS + index * N + r, tl.sum(inside.to(tl.int32), axis=0), r < N, sem="relaxed"
            )


@triton.jit
def _pairs(
    A,
    C,
    LEFT,
    RIGHT,
    OUT,
    N,
    PN: tl.constexpr,
    PC: tl.constexpr,
    FP64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Evaluate a bounded real-only pair sample without feature-gather temporaries."""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    left = tl.load(LEFT + i, i < N, other=0)
    right = tl.load(RIGHT + i, i < N, other=0)
    kind = tl.float64 if FP64 else tl.float32
    distances = tl.full((BLOCK,), 0, kind)
    for column in range(PN):
        delta = tl.load(A + left * PN + column, i < N, other=0).to(kind) - tl.load(
            A + right * PN + column, i < N, other=0
        ).to(kind)
        distances += delta * delta
    categories = tl.full((BLOCK,), 0, tl.int32)
    for column in range(PC):
        a = tl.load(C + left * PC + column, i < N, other=0)
        b = tl.load(C + right * PC + column, i < N, other=0)
        categories += 2 * (a != b).to(tl.int32)
    tl.store(OUT + i, distances + categories.to(kind), i < N)


def local_counts(numeric: Tensor, categorical: Tensor, thresholds: Tensor) -> Tensor:
    """Return integer ``[radii, rows]`` counts on the current CUDA stream.

    Inputs are contiguous, finite, draw-local features. Category codes preserve
    observed identities and contain no missing code. Thresholds are already
    squared and rounded to numeric precision. The only retained numerical state
    is the count matrix; distances occupy one 8 by 128 register tile per program.
    """
    n = len(numeric)
    if n > torch.iinfo(torch.int32).max:
        raise ValueError("Concentration count range exceeds int32 CUDA capacity")
    counts = torch.zeros((len(thresholds), n), dtype=torch.int32, device=numeric.device)
    if n > 1:
        _count[(triton.cdiv(n, 8), triton.cdiv(n, 128))](
            numeric,
            categorical,
            thresholds,
            counts,
            n,
            numeric.shape[1],
            categorical.shape[1],
            numeric.dtype == torch.float64,
            len(thresholds),
            enable_fp_fusion=False,
        )
    return counts


def paired_distances(numeric: Tensor, categorical: Tensor, left: Tensor, right: Tensor) -> Tensor:
    """Return sampled squared distances in selected precision without host synchronization."""
    output = numeric.new_empty(len(left))
    if len(left):
        _pairs[(triton.cdiv(len(left), 256),)](
            numeric,
            categorical,
            left,
            right,
            output,
            len(left),
            numeric.shape[1],
            categorical.shape[1],
            numeric.dtype == torch.float64,
            256,
            enable_fp_fusion=False,
        )
    return output
