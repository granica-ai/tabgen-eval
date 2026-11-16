"""CUDA grouped custom-operator boundaries and matching fake-tensor contracts.

The common layout is contiguous [group,column,padded_row], with NaN numeric
padding and -1 categorical padding. True row/finite counts remain separate int64
metadata. Wrappers enforce metadata contracts and dispatch lazily to Triton;
planners guarantee value bounds, valid pair identities and workspace capacity.
All operators own their outputs and declare no input mutation. Fake functions
mirror layouts for tracing without launching kernels or reading source values.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from types import ModuleType


def kernels() -> ModuleType:
    """Import grouped Triton launchers only when a CUDA grouped operation executes.

    Registration/fake metadata remain importable without importing kernel code;
    actual device execution still requires the optional Triton dependency.
    """
    return importlib.import_module("tabgen_eval.execution.torch.cuda.triton.fidelity_grouped")


def inputs(*values: Tensor) -> None:
    """Require contiguous tensor arguments on the first argument's CUDA device.

    This metadata-only guard never gathers, casts or synchronizes input values.
    Each operation separately validates ranks and supported element types.
    """
    if any(
        v.device.type != "cuda" or v.device != values[0].device or not v.is_contiguous()
        for v in values
    ):
        raise ValueError("grouped inputs must be contiguous on one CUDA device")


@torch.library.custom_op("tabgen_eval::group_gather", mutates_args=(), device_types="cuda")
def group_gather(data: Tensor, order: Tensor, offsets: Tensor, groups: Tensor, rows: int) -> Tensor:
    """Gather retained ragged groups into padded, column-major device buffers.

    Parameters
    ----------
    data : Tensor
        Contiguous CUDA FP32/FP64 or int64 matrix ``[N, P]``.
    order, offsets, groups : Tensor
        Contiguous CUDA int64 row permutation, group boundaries and selected groups.
    rows : int
        Positive padded capacity, at least the largest selected group size.

    Returns
    -------
    Tensor
        ``[G, P, rows]`` in the data dtype/device, padded with NaN or int64 ``-1``.

    Notes
    -----
    Inputs are read-only. Fake dispatch returns the padded layout without gathering;
    the group planner reserves all padded buffers before allocation.
    """
    inputs(data, order, offsets, groups)
    if data.ndim != 2 or data.dtype not in {torch.float32, torch.float64, torch.int64}:
        raise ValueError("group gather data must be a numeric/category matrix")
    if any(v.ndim != 1 or v.dtype != torch.int64 for v in (order, offsets, groups)) or rows < 1:
        raise ValueError("group gather requires integer row maps and positive padding")
    return kernels().gather(data, order, offsets, groups, rows)


@group_gather.register_fake
def _fake_gather(data: Tensor, order: Tensor, offsets: Tensor, groups: Tensor, rows: int) -> Tensor:
    """Describe owned ``[G,P,rows]`` gathered output without inspecting row maps."""
    return data.new_empty((len(groups), data.shape[1], rows))


@torch.library.custom_op("tabgen_eval::group_gram", mutates_args=(), device_types="cuda")
def group_gram(a: Tensor, b: Tensor, counts: bool = False) -> Tensor:
    """Compute batched column Gram products or exact overlap counts.

    Parameters
    ----------
    a, b : Tensor
        Contiguous CUDA ``[G, P, R]``/``[G, Q, R]`` in shared FP32/FP64,
        or bool when ``counts=True``.
    counts : bool
        Select exact int64 count output instead of IEEE floating products.

    Returns
    -------
    Tensor
        ``[G, P, Q]`` in int64 or the declared floating dtype on the same device.

    Notes
    -----
    No input mutation; fake dispatch reproduces metadata. Row-chunk partial products
    are bounded by the group's scratch budget and reduced in deterministic order.
    """
    inputs(a, b)
    if a.ndim != 3 or b.ndim != 3 or a.shape[0] != b.shape[0] or a.shape[2] != b.shape[2]:
        raise ValueError("group Gram inputs must match groups and rows")
    if a.dtype != b.dtype or (
        a.dtype != torch.bool if counts else a.dtype not in {torch.float32, torch.float64}
    ):
        raise ValueError("group Gram requires shared selected precision or boolean masks")
    return kernels().gram(a, b, counts)


@group_gram.register_fake
def _fake_gram(a: Tensor, b: Tensor, counts: bool = False) -> Tensor:
    """Describe grouped ``[G,P,Q]`` products in int64 counts or selected floating precision."""
    return a.new_empty(
        (a.shape[0], a.shape[1], b.shape[1]), dtype=torch.int64 if counts else a.dtype
    )


@torch.library.custom_op("tabgen_eval::group_ks", mutates_args=(), device_types="cuda")
def group_ks(a: Tensor, b: Tensor, na: Tensor, nb: Tensor) -> Tensor:
    """Compute exact empirical KS complements for sorted finite prefixes.

    Parameters
    ----------
    a, b : Tensor
        Contiguous CUDA FP32/FP64 ``[G, P, NR]``/``[G, P, NS]`` sorted columns.
    na, nb : Tensor
        Contiguous CUDA int64 finite-prefix lengths ``[G, P]``.

    Returns
    -------
    Tensor
        ``[G, P]`` in the input floating dtype/device; empty prefixes give NaN.

    Notes
    -----
    Inputs are read-only. Tied values use right-sided empirical CDF counts and
    padding is masked. Fake dispatch returns only output metadata.
    """
    inputs(a, b, na, nb)
    if a.ndim != 3 or b.ndim != 3 or a.shape[:2] != b.shape[:2] or a.dtype != b.dtype:
        raise ValueError("group KS requires matching grouped numeric columns")
    if (
        a.dtype not in {torch.float32, torch.float64}
        or na.shape != a.shape[:2]
        or nb.shape != na.shape
        or na.dtype != torch.int64
        or nb.dtype != torch.int64
    ):
        raise ValueError("group KS requires finite integer counts")
    return kernels().ks(a, b, na, nb)


@group_ks.register_fake
def _fake_ks(a: Tensor, b: Tensor, na: Tensor, nb: Tensor) -> Tensor:
    """Return symbolic ``[G,P]`` KS-complement metadata without reading finite-prefix counts."""
    return a.new_empty(a.shape[:2])


@torch.library.custom_op("tabgen_eval::group_bins", mutates_args=(), device_types="cuda")
def group_bins(values: Tensor, edges: Tensor, levels: Tensor, clip: bool) -> Tensor:
    """Assign grouped reference-fitted right-closed bins.

    Parameters
    ----------
    values, edges : Tensor
        Contiguous CUDA floating ``[G, P, R]``/``[G, P, E]`` in shared precision.
    levels : Tensor
        Contiguous CUDA int64 ``[G, P]`` active bin counts.
    clip : bool
        Clip finite candidate observations to the reference edge range.

    Returns
    -------
    Tensor
        Int64 ``[G, P, R]`` codes; excluded/nonfinite cells are ``-1``.

    Notes
    -----
    No input mutation. Fake dispatch preserves code shape/device. Padding and
    inactive edges are masked; selected precision governs discrete boundary decisions.
    """
    inputs(values, edges, levels)
    if (
        values.ndim != 3
        or edges.ndim != 3
        or values.shape[:2] != edges.shape[:2]
        or values.dtype != edges.dtype
        or levels.shape != values.shape[:2]
        or levels.dtype != torch.int64
    ):
        raise ValueError("group bins require matching columns, edges and level counts")
    return kernels().bins(values, edges, levels, clip)


@group_bins.register_fake
def _fake_bins(values: Tensor, edges: Tensor, levels: Tensor, clip: bool) -> Tensor:
    """Describe int64 codes matching the grouped ``[G,P,R]`` numeric input shape."""
    return values.new_empty(values.shape, dtype=torch.int64)


@torch.library.custom_op("tabgen_eval::group_edges", mutates_args=(), device_types="cuda")
def group_edges(values: Tensor, counts: Tensor, bins: int) -> tuple[Tensor, Tensor]:
    """Fit and compress quantiles from sorted finite prefixes in one CUDA kernel.

    Read-only FP32/FP64 values [G,P,R], int64 counts [G,P]; outputs selected
    precision edges [G,P,bins+1] and int64 levels [G,P]. Constant columns retain
    padded equal-width edges, and missing/empty columns have zero levels.
    """
    inputs(values, counts)
    if values.ndim != 3 or counts.shape != values.shape[:2] or values.shape[2] < 1:
        raise ValueError("group edges require sorted columns and compatible counts")
    if values.dtype not in {torch.float32, torch.float64} or counts.dtype != torch.int64:
        raise ValueError("group edges require selected precision and int64 counts")
    if bins < 1:
        raise ValueError("group edges require positive bins")
    return kernels().edges(values, counts, bins)


@group_edges.register_fake
def _fake_edges(values: Tensor, counts: Tensor, bins: int) -> tuple[Tensor, Tensor]:
    """Describe fixed ``bins+1`` edge capacity and per-column int64 active-level metadata."""
    return values.new_empty((*values.shape[:2], bins + 1)), counts.new_empty(counts.shape)


@torch.library.custom_op("tabgen_eval::group_center", mutates_args=(), device_types="cuda")
def group_center(values: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
    """Center valid grouped columns stably, using bounded tiled scratch.

    Read-only FP32/FP64 values and bool validity [G,P,R] produce normalized
    columns and bool variation [G,P]. Validity can represent pairwise overlap.
    Anchor/scale arithmetic preserves overflow fallback and empty semantics.
    """
    inputs(values, valid)
    if values.ndim != 3 or valid.shape != values.shape:
        raise ValueError("group centering requires compatible columns and validity")
    if values.dtype not in {torch.float32, torch.float64} or valid.dtype != torch.bool:
        raise ValueError("group centering requires selected precision and bool validity")
    return kernels().center(values, valid)


@group_center.register_fake
def _fake_center(values: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
    """Describe centered ``[G,P,R]`` values and bool ``[G,P]`` variation flags."""
    return torch.empty_like(values), valid.new_empty(values.shape[:2])


def joint_inputs(codes: Tensor, pairs: Tensor, widths: Tensor) -> None:
    """Validate grouped contingency metadata without reading category identities.

    Codes are int64 ``[G,P,R]``, pairs ``[Q,2]``, and widths ``[G,Q]`` on one
    contiguous CUDA device. The caller guarantees pair bounds, nonnegative valid
    codes and int64 key capacity; negative codes are exclusions.
    """
    inputs(codes, pairs, widths)
    if (
        codes.ndim != 3
        or pairs.ndim != 2
        or pairs.shape[1] != 2
        or widths.shape != (codes.shape[0], len(pairs))
    ):
        raise ValueError("group joint inputs have incompatible shapes")
    if any(v.dtype != torch.int64 for v in (codes, pairs, widths)):
        raise ValueError("group joint inputs require int64")


@torch.library.custom_op("tabgen_eval::group_dense_counts", mutates_args=(), device_types="cuda")
def group_dense_counts(codes: Tensor, pairs: Tensor, widths: Tensor, grid: int) -> Tensor:
    """Count exact joint categories for batched retained groups.

    Parameters
    ----------
    codes : Tensor
        Contiguous CUDA int64 ``[G, P, R]`` padded with ``-1``.
    pairs, widths : Tensor
        Int64 ``[B, 2]`` column pairs and ``[G, B]`` second-axis widths.
    grid : int
        Power-of-two flattened joint-grid bound, at most 4096.

    Returns
    -------
    Tensor
        CUDA int64 ``[G, B, grid]`` counts; fake dispatch returns this metadata.

    Notes
    -----
    No inputs are mutated. The caller budgets dense scratch and ensures every
    occupied joint key fits ``grid``.
    """
    joint_inputs(codes, pairs, widths)
    if grid < 1 or grid > 4096 or grid & (grid - 1):
        raise ValueError("group dense grid must be a power of two up to 4096")
    return kernels().dense(codes, pairs, widths, grid)


@group_dense_counts.register_fake
def _fake_dense(codes: Tensor, pairs: Tensor, widths: Tensor, grid: int) -> Tensor:
    """Describe owned int64 ``[G,Q,grid]`` histograms without allocating device work."""
    return codes.new_empty((codes.shape[0], len(pairs), grid))


@torch.library.custom_op("tabgen_eval::group_dense_tv", mutates_args=(), device_types="cuda")
def group_dense_tv(
    a: Tensor,
    b: Tensor,
    pairs: Tensor,
    levels: Tensor,
    grid: int,
    cond: int,
    marginal: bool,
    prototype: Tensor,
) -> Tensor:
    """Count read-only grouped codes and fuse normalization, TV and exclusions.

    Codes are CUDA int64 [G,P,R/S], pairs int64 [Q,2], and levels int64 [G,P].
    Return [G,Q] scores in prototype precision. The caller reserves two integer
    histograms and output; excluded conditioning axes and empty slices yield NaN.
    """
    inputs(a, b, pairs, levels, prototype)
    if a.ndim != 3 or b.ndim != 3 or a.shape[:2] != b.shape[:2]:
        raise ValueError("group TV codes must match groups and columns")
    if pairs.ndim != 2 or pairs.shape[1] != 2 or levels.shape != a.shape[:2]:
        raise ValueError("group TV pairs/levels have incompatible shapes")
    if any(value.dtype != torch.int64 for value in (a, b, pairs, levels)):
        raise ValueError("group TV codes/pairs/levels require int64")
    if prototype.dtype not in {torch.float32, torch.float64}:
        raise ValueError("group TV requires selected floating precision")
    if grid < 1 or grid > 8192 or grid & (grid - 1):
        raise ValueError("group TV grid must be a power of two up to 8192")
    return kernels().dense_tv(a, b, pairs, levels, grid, cond, marginal, prototype)


@group_dense_tv.register_fake
def _fake_dense_tv(
    a: Tensor,
    b: Tensor,
    pairs: Tensor,
    levels: Tensor,
    grid: int,
    cond: int,
    marginal: bool,
    prototype: Tensor,
) -> Tensor:
    """Describe ``[G,Q]`` fused TV similarities in the scalar prototype's precision."""
    return prototype.new_empty((len(a), len(pairs)))


@torch.library.custom_op("tabgen_eval::group_keys", mutates_args=(), device_types="cuda")
def group_keys(codes: Tensor, pairs: Tensor, widths: Tensor) -> Tensor:
    """Encode occupied joint keys for sparse grouped contingency reductions.

    Parameters
    ----------
    codes, pairs, widths : Tensor
        Same CUDA int64 contracts as ``group_dense_counts``.

    Returns
    -------
    Tensor
        CUDA int64 ``[G * B, R]`` keys, with excluded cells ``-1``.

    Notes
    -----
    Inputs are read-only; fake dispatch returns the flattened group/pair metadata.
    Downstream sorting scales with occupied rows rather than category products.
    """
    joint_inputs(codes, pairs, widths)
    return kernels().keys(codes, pairs, widths)


@group_keys.register_fake
def _fake_keys(codes: Tensor, pairs: Tensor, widths: Tensor) -> Tensor:
    """Describe int64 ``[G*Q,R]`` keys using the same flattened group/pair order as CUDA."""
    return codes.new_empty((codes.shape[0] * len(pairs), codes.shape[2]))
