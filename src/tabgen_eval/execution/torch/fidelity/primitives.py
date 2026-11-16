"""Shared fidelity statistics over read-only selected-precision tensor columns.

Numeric missingness is per cell, never a complete-row filter. FP32/FP64 moments,
quantile interpolation and probabilities stay in their declared precision;
identity keys and counts remain int64 until normalization. Reference-fitted
numeric edges are reused on candidates. Dense contingencies are restricted to
small grids, while high-cardinality domains sort occupied identities. These
primitives return device scores; executors handle group/draw aggregation and
conversion to host report arrays.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations

import torch
from torch import Tensor

from tabgen_eval.execution.torch.fidelity.ops import (
    fidelity_gram,
    finite_pair_counts,
    joint_dense_counts,
    joint_keys,
    joint_sorted_tv,
)


def centered_columns(values: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
    """Build stable centered columns from FP32/FP64 ``[N,P]`` values and bool validity.

    Anchor each column at its first valid observation before scaling; this
    preserves small variation around large offsets. If subtraction of finite
    opposite extremes overflows, scale the original values first. Invalid cells
    become zero and columns without valid variation are flagged false in the
    returned ``[P]`` bool vector. Both outputs are owned; inputs are read-only.
    The caller handles empty row axes before requesting the first valid anchor.
    """
    p = values.shape[1]
    first = valid.to(torch.int64).argmax(dim=0)
    anchor = values[first, torch.arange(p, device=values.device)]
    anchor = torch.where(valid.any(dim=0), anchor, 0)
    clean = torch.where(valid, values, anchor)
    shifted = clean - anchor
    scale = shifted.abs().amax(dim=0)
    # Opposite-sign finite extremes may overflow their difference. Scale first
    # only for those columns; ordinary/high-offset columns retain direct shifts.
    absolute = clean.abs().amax(dim=0).clamp(min=torch.finfo(values.dtype).tiny)
    z = torch.where(
        torch.isfinite(scale),
        shifted / torch.where(scale > 0, scale, 1),
        clean / absolute - anchor / absolute,
    )
    means = z.sum(dim=0) / valid.sum(dim=0).clamp(min=1).to(values.dtype)
    z = torch.where(valid, z - means, 0)
    variable = (scale > 0) & valid.any(dim=0)
    return z, variable


def correlations(values: Tensor) -> Tensor:
    """Compute stable pairwise-finite correlations in selected precision.

    Parameters
    ----------
    values : Tensor
        FP32/FP64 ``[N, P]`` owned CPU/CUDA observations with per-cell missingness.

    Returns
    -------
    Tensor
        ``[P, P]`` correlations in the same dtype/device; unusable pairs are NaN.

    Notes
    -----
    Inputs are read-only. Shifted/scaled moments and exact integer overlap counts
    avoid losing small variation at large offsets. Near-degenerate pairs use a
    two-pass calculation in the same precision, without refinement.
    """
    p = values.shape[1]
    if not len(values) or not p:
        return values.new_full((p, p), float("nan"))
    valid = torch.isfinite(values)
    z, variable = centered_columns(values, valid)
    if bool(valid.all()):
        # Complete columns need only one Gram product; avoid the masked moment
        # products and integer matrix multiplication in this common CPU case.
        cross = fidelity_gram(z, z)
        norms = cross.diagonal()
        usable = variable[:, None] & variable[None, :] & (norms[:, None] > 0) & (norms[None, :] > 0)
        rho = cross / (norms[:, None] * norms[None, :]).sqrt()
        return torch.where(usable, rho.clamp(-1, 1), float("nan"))
    masks = valid.to(values.dtype)
    count = finite_pair_counts(valid)
    denominator = count.clamp(min=1).to(values.dtype)
    sums = fidelity_gram(z, masks)
    squares = fidelity_gram((z * z).contiguous(), masks)
    cross = fidelity_gram(z, z)
    correction = sums * sums / denominator
    variance = squares - correction
    covariance = cross - sums * sums.T / denominator
    rho = covariance / (variance * variance.T).clamp(min=0).sqrt()
    usable = (count >= 2) & variable[:, None] & variable[None, :]
    rho = torch.where(usable & (variance > 0) & (variance.T > 0), rho.clamp(-1, 1), float("nan"))
    uncertainty = 32 * torch.finfo(values.dtype).eps * (squares.abs() + correction.abs())
    suspect = usable & ((variance <= uncertainty) | (variance.T <= uncertainty.T))
    for i, j in torch.nonzero(torch.triu(suspect, diagonal=1), as_tuple=False).cpu().tolist():
        keep = valid[:, i] & valid[:, j]
        pair = values[:, [i, j]][keep].contiguous()
        pair, varies = centered_columns(pair, torch.ones_like(pair, dtype=torch.bool))
        norms = (pair * pair).sum(dim=0)
        result = (pair[:, 0] * pair[:, 1]).sum() / norms.prod().sqrt()
        result = torch.where(varies.all() & (norms > 0).all(), result.clamp(-1, 1), float("nan"))
        rho[i, j] = rho[j, i] = result
    return rho


def equal_width_edges(values: Tensor, num_bins: int = 10) -> Tensor | None:
    """Fit selected-precision equal-width boundaries from finite reference values.

    Return a 1-D edge tensor on the input device, or None when no usable interval
    exists. Constant columns receive relative/absolute padding before binning;
    duplicate rounded boundaries are removed. No candidate values affect this fit.
    """
    finite = values[torch.isfinite(values)]
    if not len(finite):
        return None
    lo, hi = finite.amin(), finite.amax()
    if bool(lo == hi):
        pad = lo.abs() * 1e-6 + 1e-9
        lo, hi = lo - pad, hi + pad
    positions = torch.arange(num_bins + 1, dtype=values.dtype, device=values.device)
    # CUDA scalar division may multiply by a rounded reciprocal. A tensor
    # divisor preserves the native CPU division and tied boundary membership.
    divisor = values.new_tensor(num_bins)
    edges = lo + positions * ((hi - lo) / divisor)
    edges[-1] = hi
    edges = edges.unique_consecutive()
    return edges if len(edges) >= 2 else None


def quantile_edges(values: Tensor, num_bins: int = 10) -> Tensor | None:
    """Fit linearly interpolated quantiles to finite reference observations.

    Sorting and interpolation stay in the input FP32/FP64 dtype/device. Repeated
    boundaries are compressed; constant columns use padded equal-width edges.
    Return None for absent usable observations. Interpolation is anchored to the
    nearer endpoint to limit cancellation and matches grouped edge construction.
    """
    ordered = values[torch.isfinite(values)].sort().values
    if not len(ordered):
        return None
    if bool(ordered[0] == ordered[-1]):
        return equal_width_edges(ordered, num_bins)
    q = torch.arange(num_bins + 1, device=values.device, dtype=values.dtype) * (1 / num_bins)
    positions = q.clamp(max=1) * (len(ordered) - 1)
    low = positions.floor().to(torch.int64)
    high = (low + 1).clamp(max=len(ordered) - 1)
    fraction = positions - low.to(values.dtype)
    difference = ordered[high] - ordered[low]
    edges = torch.where(
        fraction >= 0.5,
        ordered[high] - difference * (1 - fraction),
        ordered[low] + difference * fraction,
    ).unique_consecutive()
    return edges if len(edges) >= 2 else None


def bin_codes(
    values: Tensor, edges: Tensor | None, *, clip: bool = False, clip_infinite: bool = False
) -> Tensor:
    """Assign right-closed reference bins with an explicit infinity policy.

    Parameters
    ----------
    values : Tensor
        Floating observations on the edge dtype/device.
    edges : Tensor or None
        Sorted reference-fitted bin boundaries; absent edges exclude all cells.
    clip, clip_infinite : bool
        Candidate clipping and whether infinite values may be clipped into bins.

    Returns
    -------
    Tensor
        Int64 codes matching ``values.shape``; excluded cells use ``-1``.

    Notes
    -----
    Inputs are read-only. The caller distinguishes conditioning bins from
    within-slice contingency bins; precision can change boundary membership.
    """
    if edges is None:
        return torch.full(values.shape, -1, dtype=torch.int64, device=values.device)
    valid = ~values.isnan() if clip_infinite else torch.isfinite(values)
    v = values.clamp(min=edges[0], max=edges[-1]) if clip else values
    codes = torch.searchsorted(edges, v.contiguous(), right=False) - 1
    codes = torch.where(v == edges[0], 0, codes)
    return torch.where(valid & torch.isfinite(v) & (v >= edges[0]) & (v <= edges[-1]), codes, -1)


def ks_complement(a: Tensor, b: Tensor) -> Tensor:
    """Return scalar ``1-KS`` for two read-only floating observation vectors.

    Each side independently discards nonfinite cells and sorts in selected
    precision. Right-sided searches at observations from both sides preserve
    empirical-CDF ties. Either empty finite population yields NaN in input dtype.
    """
    a, b = a[torch.isfinite(a)].sort().values, b[torch.isfinite(b)].sort().values
    if not len(a) or not len(b):
        return a.new_full((), float("nan"))
    observed = torch.cat((a, b))
    ca = torch.searchsorted(a, observed, right=True).to(a.dtype) / len(a)
    cb = torch.searchsorted(b, observed, right=True).to(a.dtype) / len(b)
    return 1 - (ca - cb).abs().amax()


def tv_complement_codes(a: Tensor, b: Tensor, n_levels: int, prototype: Tensor) -> Tensor:
    """Return a scalar marginal TV complement for shared int64 category identities.

    Exclude negative missing codes on each side, count exactly before converting
    to ``prototype`` precision, and normalize by each side's own valid size.
    ``n_levels`` bounds the shared category vocabulary. Empty sides yield NaN;
    source code arrays are never modified.
    """
    a, b = a[a >= 0], b[b >= 0]
    if not len(a) or not len(b):
        return prototype.new_full((), float("nan"))
    ca = torch.bincount(a, minlength=n_levels).to(prototype.dtype) / len(a)
    cb = torch.bincount(b, minlength=n_levels).to(prototype.dtype) / len(b)
    return (1 - 0.5 * (ca - cb).abs().sum()).clamp(0, 1)


def contingency_scores(
    a: Tensor,
    b: Tensor,
    levels: list[int],
    numeric: int,
    prototype: Tensor,
    scratch_bytes: int,
    *,
    return_pairs: bool = False,
) -> Tensor | tuple[Tensor, list[tuple[int, int]]]:
    """Compare mixed/categorical column pairs using bounded dense or occupied counts.

    ``a`` and ``b`` are int64 ``[NR,P]``/``[NS,P]`` code matrices; ``numeric``
    marks the initial numeric-bin columns. Pure numeric pairs are skipped because
    Trend handles them by correlation. ``levels`` supplies the shared fitted
    axis widths; negative codes exclude an observation from that pair.

    Power-of-two products up to 4096 use integer histograms; larger domains sort
    occupied int64 joint keys rather than allocating the full Cartesian grid.
    Products exceeding int64 raise OverflowError. ``scratch_bytes`` determines
    the pair batch and raises MemoryError if even one pair cannot fit. The result
    is a flat vector of TV complements in ``prototype`` dtype/device, with NaN
    for empty distributions. With ``return_pairs=True``, also return the ordered
    host ``(column_1, column_2)`` identities corresponding to that vector. The
    identities follow histogram buckets and batches rather than canonical pair
    order. Inputs are read-only; only owned scratch is mutated.
    """
    groups: defaultdict[int, list[tuple[int, int]]] = defaultdict(list)
    for i, j in combinations(range(len(levels)), 2):
        if j < numeric or levels[i] == 0 or levels[j] == 0:
            continue
        product = levels[i] * levels[j]
        if product > torch.iinfo(torch.int64).max:
            raise OverflowError("joint category product exceeds int64 key capacity")
        grid = 1 << (product - 1).bit_length() if product <= 4096 else 0
        groups[grid].append((i, j))
    scores: list[Tensor] = []
    identities: list[tuple[int, int]] | None = [] if return_pairs else None
    for grid, pairs in groups.items():
        if grid and a.device.type == "cuda":
            per_pair = grid * 8 * 6 + 128
        else:
            per_pair = (len(a) + len(b)) * 8 * 8 + grid * 8 * 6 + 128
        if per_pair > scratch_bytes:
            raise MemoryError("one contingency pair exceeds the resolved workspace allowance")
        batch = max(1, min(64, scratch_bytes // per_pair))
        for start in range(0, len(pairs), batch):
            selected = pairs[start : start + batch]
            indices = torch.tensor(selected, dtype=torch.int64, device=a.device)
            widths = torch.tensor(
                [levels[j] for _, j in selected], dtype=torch.int64, device=a.device
            )
            if grid:
                ac, bc = (
                    joint_dense_counts(a, indices, widths, grid),
                    joint_dense_counts(b, indices, widths, grid),
                )
                na, nb = ac.sum(dim=1), bc.sum(dim=1)
                delta = (
                    (
                        ac.to(prototype.dtype) / na.clamp(min=1).to(prototype.dtype)[:, None]
                        - bc.to(prototype.dtype) / nb.clamp(min=1).to(prototype.dtype)[:, None]
                    )
                    .abs()
                    .sum(dim=1)
                )
                delta = torch.where((na > 0) & (nb > 0), delta, float("nan"))
            else:
                ak = joint_keys(a, indices, widths).sort(dim=1).values
                bk = joint_keys(b, indices, widths).sort(dim=1).values
                zero = indices.new_zeros((len(selected), 1))
                na = len(a) - torch.searchsorted(ak, zero).squeeze(1)
                nb = len(b) - torch.searchsorted(bk, zero).squeeze(1)
                delta = joint_sorted_tv(ak, bk, na, nb, prototype)
            scores.append((1 - 0.5 * delta).clamp(0, 1))
            if return_pairs:
                assert identities is not None
                identities.extend(selected)
    result = torch.cat(scores) if scores else prototype.new_empty(0)
    if return_pairs:
        assert identities is not None
        return result, identities
    return result
