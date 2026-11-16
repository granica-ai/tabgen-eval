"""Reference-fitted geometry and bounded pure-precision distance reductions.

Numeric scaling stays in the selected FP32/FP64 mode. Categories use compact
identities with exact integer contributions. Persistent pools are read-only.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch

from tabgen_eval.execution.runtime.components import PRDC_COMPONENTS, selected_components
from tabgen_eval.execution.torch.distance.ops import (
    nearest_squared_self_tile,
    nearest_squared_tile,
    prdc_cross_tile,
    squared_distance_tile,
)

if TYPE_CHECKING:
    from tabgen_eval.execution.torch.context import TorchContext
    from tabgen_eval.execution.torch.tensors import TensorTable

    from .categories import CategoryDomain


@dataclass
class Features:
    """Draw-local transformed numeric/category tensors.

    Parameters
    ----------
    numeric : Tensor
        Contiguous ``[N, PN]`` FP32/FP64 feature coordinates.
    categorical : Tensor
        Contiguous ``[N, PC]`` int64 known category identities or unknown ``-1``.

    Notes
    -----
    Both fields share a device. Row slices alias this draw's owned buffers, never
    caller DataFrames or persistent source tensors.
    """

    numeric: torch.Tensor
    categorical: torch.Tensor

    @property
    def n_rows(self) -> int:
        """Number of aligned rows in the numeric and categorical feature matrices."""
        return self.numeric.shape[0]

    def rows(self, start: int, end: int) -> Features:
        """Return row-slice views sharing this draw's storage; no gather or copy occurs."""
        return Features(self.numeric[start:end], self.categorical[start:end])


class ReferenceGeometry:
    """Fit once per draw; consume owned gathers while keeping input pools intact."""

    def __init__(
        self,
        reference: TensorTable,
        *,
        domain: CategoryDomain | None = None,
        workspace: DistanceWorkspace | None = None,
    ) -> None:
        """Fit numeric ranges and categorical membership from a complete reference draw.

        Numeric zero ranges become one. If an eligible ``domain`` and ``workspace`` can
        hold a presence buffer, fit it; otherwise retain compact sorted per-column
        vocabularies. CUDA unique results are cloned so capacity tracks occupied levels,
        not the number of reference rows. The supplied reference is not modified.
        """
        self.mins = reference.numeric.amin(dim=0)
        ranges = reference.numeric.amax(dim=0) - self.mins
        self.ranges = torch.where(ranges == 0, 1, ranges)
        self.presence = domain.fit(reference, workspace) if domain is not None else None
        self.domain = domain if self.presence is not None else None
        self.vocabularies: tuple[torch.Tensor, ...] | None = None
        if self.domain is None:
            # CUDA unique retains an N-element allocation behind its K logical
            # values. Compact each result before fitting the next column so the
            # retained storage matches the workspace's occupied-vocabulary bound.
            def vocabulary(c: int) -> torch.Tensor:
                """Fit sorted levels and compact CUDA unique storage to its logical length."""
                levels = torch.unique(reference.categorical[:, c], sorted=True)
                return levels.clone() if levels.device.type == "cuda" else levels

            self.vocabularies = tuple(vocabulary(c) for c in range(reference.categorical.shape[1]))

    def transform(self, table: TensorTable, *, reference: bool = False) -> Features:
        """Transform an owned draw using reference-fitted ranges/vocabularies.

        Parameters
        ----------
        table : TensorTable
            Draw-owned numeric/category buffers on the fitted device and precision.
        reference : bool
            Skip unknown-category remapping for the reference draw itself.

        Returns
        -------
        Features
            Views of the transformed owned draw.

        Notes
        -----
        This method mutates ``table`` in place. Never pass persistent input pools.
        Numerics are range-scaled without clipping; unknown categories become ``-1``.
        """
        table.numeric.sub_(self.mins).div_(self.ranges)
        if not reference:
            if self.domain is not None:
                assert self.presence is not None
                self.domain.mask(table, self.presence)
            else:
                assert self.vocabularies is not None
                for c, levels in enumerate(self.vocabularies):
                    values = table.categorical[:, c].contiguous()
                    positions = torch.searchsorted(levels, values)
                    known = (positions < len(levels)) & (
                        levels[positions.clamp(max=len(levels) - 1)] == values
                    )
                    table.categorical[:, c].masked_fill_(~known | (values < 0), -1)
        return Features(table.numeric, table.categorical)


class DistanceWorkspace:
    """Conservative GPU live-buffer reservation and two-axis scratch planning.

    Input pools are persistent and excluded. Tasks execute sequentially; the budget
    includes gathered/encoded rows, occupied vocabularies, indices, reductions and
    headroom for ATen sorting/selecting before reserving bounded operator scratch.
    """

    def __init__(
        self,
        context: TorchContext,
        rows: Sequence[int],
        original_rows: Sequence[int],
        *,
        k: int | None = None,
        category_metadata_bytes: int = 0,
    ) -> None:
        """Reserve draw-local live buffers before allocating bounded distance scratch.

        ``rows`` are complete-row counts by pool and ``original_rows`` are pre-filter
        counts, including duplicates. ``k`` selects PRDC neighbor-state accounting;
        ``None`` selects DCR minima. Persistent context pools are excluded, while lazy
        category metadata is included explicitly. CUDA raises ``MemoryError`` if minimal
        scratch cannot coexist with estimated live storage under ``gpu_workspace_mb``;
        CPU uses a small fixed tile scratch target to limit launch overhead.
        """
        element = context.reference.numeric.element_size()
        self.element = element
        self.cuda = context.inputs.device.type == "cuda"
        self.k = k
        largest = max(rows, default=0)
        row_bytes = context.n_numeric * element + context.n_categorical * 8 + 1
        data = sum(rows) * row_bytes
        vocab_sizes = [min(card, rows[0]) * 8 for card in context.inputs.cardinalities]
        # Compact CUDA vocabularies still own separate allocator-rounded buffers.
        vocab = sum(((size + 511) // 512) * 512 if self.cuda else size for size in vocab_sizes)
        self.vocabulary_bytes = vocab
        # Boolean numeric checks plus one column's contiguous values, unique
        # output, sorting/selecting scratch and query remapping temporaries.
        preprocessing = largest * (context.n_numeric + (64 if context.n_categorical else 0))
        preprocessing += context.n_numeric * element * 4
        indices = 8 * sum(original_rows) + 9 * max(original_rows, default=0)
        if k is None:
            # Reserve generated/holdout nearest-distance reductions.
            state = max(rows, default=0) * (4 * element + 8)
        else:
            state = sum(rows) * (element + 10)
        self.reserved_bytes = (
            data + vocab + preprocessing + indices + state + category_metadata_bytes + 16384
        )
        self.budget_bytes = context.gpu_workspace_mb * 1024**2
        if self.cuda:
            if self.reserved_bytes + 4096 > self.budget_bytes:
                raise MemoryError(
                    "distance draws and reductions exceed the resolved workspace allowance"
                )
            self.scratch_bytes = min(64 * 1024**2, (self.budget_bytes - self.reserved_bytes) // 3)
        else:
            # Small ATen CPU tiles avoid excessive parallel-launch overhead.
            self.scratch_bytes = 512 * 1024

    def reserve_categories(self, presence_bytes: int) -> bool:
        """Reserve presence storage instead of occupied-vocabulary capacity.

        Round CUDA storage to allocator granularity and charge only any increase
        over the existing vocabulary reservation. Return ``False`` without
        changing the plan if minimal scratch would no longer fit. On success,
        increase ``reserved_bytes`` and shrink ``scratch_bytes`` as needed.
        This updates accounting only; ``CategoryDomain.fit`` owns allocation.
        """
        if self.cuda:
            presence_bytes = ((presence_bytes + 511) // 512) * 512
        extra = max(0, presence_bytes - self.vocabulary_bytes)
        if self.reserved_bytes + extra + 4096 > self.budget_bytes:
            return False
        self.reserved_bytes += extra
        self.scratch_bytes = min(self.scratch_bytes, (self.budget_bytes - self.reserved_bytes) // 3)
        return True

    def shape(
        self, mode: Literal["radius", "nearest", "cross"], n_query: int, n_reference: int
    ) -> tuple[int, int]:
        """Choose query/reference tile extents within reserved scratch.

        Parameters
        ----------
        mode : str
            ``radius``, ``nearest`` or ``cross`` selects the live-buffer estimate.
        n_query, n_reference : int
            Row counts after the caller's complete-row filtering.

        Returns
        -------
        tuple of int
            Positive tile extents for both axes.

        Raises
        ------
        MemoryError
            Even the minimum tile would exceed available temporary workspace.
        """
        q = min(
            n_query,
            4096
            if self.cuda and mode == "nearest"
            else 512
            if self.cuda and mode == "cross"
            else 256
            if self.cuda
            else 32,
        )
        r = min(n_reference, 65536 if self.cuda else 256)

        def size() -> int:
            """Estimate bytes live for the current tile extents and selected reduction mode."""
            if not self.cuda:
                return q * r * (self.element * 3 + 32)
            if mode == "nearest":
                return q * (((r + 127) // 128) * self.element + 3 * self.element)
            if mode == "cross":
                return ((q + 7) // 8) * r * 4 + 2 * q * ((r + 127) // 128) + r * 8 + q * 2
            assert self.k is not None
            return q * (r * self.element + (self.k + 1) * (6 * self.element + 32))

        while size() > self.scratch_bytes:
            if r > 128:
                r = max(128, r // 2)
            elif q > 1:
                q = max(1, q // 2)
            elif r > 1:
                r = max(1, r // 2)
            else:
                raise MemoryError(
                    "distance tile/neighbor selection exceeds the resolved workspace allowance"
                )
        return max(q, 1), max(r, 1)


def finite(features: Features) -> bool:
    """Check numeric coordinates for NaN/infinity, synchronizing a device scalar to host."""
    return bool(torch.isfinite(features.numeric).all().item())


def kth_radii(features: Features, k: int, workspace: DistanceWorkspace) -> torch.Tensor:
    """Stream squared k-neighbor radii while retaining bootstrap duplicates.

    Parameters
    ----------
    features : Features
        Finite transformed draw on one device in the selected precision.
    k : int
        Positive neighbor rank; self occupies rank zero and ``k < N``.
    workspace : DistanceWorkspace
        Caller-owned live-buffer and tile-size budget.

    Returns
    -------
    Tensor
        Squared radii ``[N]`` in the numeric dtype/device.

    Notes
    -----
    Inputs are read-only; top-k state retains ``k+1`` neighbors across reference tiles.
    """
    n = features.n_rows
    qstep, rstep = workspace.shape("radius", n, n)
    output = features.numeric.new_empty(n)
    keep = k + 1
    for q in range(0, n, qstep):
        query = features.rows(q, min(q + qstep, n))
        best = query.numeric.new_full((query.n_rows, keep), float("inf"))
        for r in range(0, n, rstep):
            reference = features.rows(r, min(r + rstep, n))
            distances = squared_distance_tile(
                query.numeric, reference.numeric, query.categorical, reference.categorical
            )
            local = torch.topk(
                distances, min(keep, reference.n_rows), dim=1, largest=False, sorted=False
            ).values
            del distances
            combined = torch.cat((best, local), dim=1)
            best = torch.topk(combined, keep, dim=1, largest=False, sorted=False).values
            del local, combined
        output[q : q + query.n_rows] = best.amax(dim=1)
    return output


def nearest_distances(
    reference: Features,
    query: Features,
    workspace: DistanceWorkspace,
    *,
    exclude_self: bool = False,
) -> torch.Tensor:
    """Stream Euclidean closest-reference distances for each query.

    Parameters
    ----------
    reference, query : Features
        Transformed complete-row features sharing geometry, dtype and device.
    workspace : DistanceWorkspace
        Bound for gathers, reductions and two-axis kernel scratch.
    exclude_self : bool
        Exclude equal parent row positions for a same-table search. Other rows
        with identical values remain eligible neighbors.

    Returns
    -------
    Tensor
        ``[N_query]`` distances in selected precision, after square rooting minima.

    Notes
    -----
    Features are read-only; no full pairwise distance matrix is retained.
    """
    qstep, rstep = workspace.shape("nearest", query.n_rows, reference.n_rows)
    output = query.numeric.new_empty(query.n_rows)
    for q in range(0, query.n_rows, qstep):
        block = query.rows(q, min(q + qstep, query.n_rows))
        best = block.numeric.new_full((block.n_rows,), float("inf"))
        for r in range(0, reference.n_rows, rstep):
            ref = reference.rows(r, min(r + rstep, reference.n_rows))
            if exclude_self:
                local = nearest_squared_self_tile(
                    block.numeric, ref.numeric, block.categorical, ref.categorical, q, r
                )
            else:
                local = nearest_squared_tile(
                    block.numeric, ref.numeric, block.categorical, ref.categorical
                )
            torch.minimum(best, local, out=best)
        output[q : q + block.n_rows] = best.sqrt()
    return output


def _median_ordered(ordered: torch.Tensor) -> torch.Tensor:
    """Average the middle Euclidean distances for even-sized nonempty sorted data."""
    middle = len(ordered) // 2
    return (
        ordered[middle] if len(ordered) % 2 else ordered[middle - 1] * 0.5 + ordered[middle] * 0.5
    )


def prdc(
    reference: Features,
    candidate: Features,
    k: int,
    workspace: DistanceWorkspace,
    *,
    requested_components: Iterable[str] | None = None,
) -> dict[str, float]:
    """Reduce precision, recall, density and coverage from strict balls.

    Parameters
    ----------
    reference, candidate : Features
        Finite complete-row draws sharing reference-fitted geometry.
    k : int
        Neighbor rank with self counted at zero.
    workspace : DistanceWorkspace
        Reserved draw/reduction storage and bounded tile scratch.

    Returns
    -------
    dict
        Requested scalar scores for the validated finite draw; unrequested
        components remain present as NaN in the internal computation result.

    Notes
    -----
    Inputs remain unchanged; category counts are exact and strict comparisons use
    the selected precision. Tiles stream across both axes.
    """
    components = selected_components(requested_components, PRDC_COMPONENTS)
    mask = sum(1 << i for i, name in enumerate(PRDC_COMPONENTS) if name in components)
    wants_counts, wants_recall, wants_cover = bool(mask & 5), bool(mask & 2), bool(mask & 8)
    rr = (
        kth_radii(reference, k, workspace)
        if mask & 13
        else reference.numeric.new_zeros(reference.n_rows)
    )
    sr = (
        kth_radii(candidate, k, workspace)
        if wants_recall
        else candidate.numeric.new_zeros(candidate.n_rows)
    )
    nr, ns = reference.n_rows, candidate.n_rows
    counts = torch.zeros(ns, dtype=torch.int64, device=rr.device)
    recalled = torch.zeros(nr, dtype=torch.bool, device=rr.device)
    covered = torch.zeros_like(recalled)
    qstep, rstep = workspace.shape("cross", nr, ns)
    for q in range(0, nr, qstep):
        a = reference.rows(q, min(q + qstep, nr))
        for r in range(0, ns, rstep):
            b = candidate.rows(r, min(r + rstep, ns))
            containing, recall, cover = prdc_cross_tile(
                a.numeric,
                b.numeric,
                a.categorical,
                b.categorical,
                rr[q : q + a.n_rows],
                sr[r : r + b.n_rows],
                mask,
            )
            if wants_counts:
                counts[r : r + b.n_rows].add_(containing)
            if wants_recall:
                recalled[q : q + a.n_rows].logical_or_(recall)
            if wants_cover:
                covered[q : q + a.n_rows].logical_or_(cover)
    # Integer reductions are exact. Convert only scalar counts for the existing report API.
    summary = (
        torch.stack(((counts > 0).sum(), recalled.sum(), counts.sum(), covered.sum()))
        .cpu()
        .tolist()
    )
    values = dict(
        zip(
            ("precision", "recall", "density", "coverage"),
            (summary[0] / ns, summary[1] / nr, summary[2] / (k * ns), summary[3] / nr),
            strict=True,
        )
    )
    return {name: value if name in components else float("nan") for name, value in values.items()}


def _near_match_rate(values: torch.Tensor, threshold: float) -> torch.Tensor:
    """Reduce inclusive proximity without rounding a real cutoff upward in FP32."""
    cutoff: float | np.float32 = threshold
    if values.dtype == torch.float32:
        cutoff = np.float32(min(cutoff, np.finfo(np.float32).max))
        if float(cutoff) > threshold:
            cutoff = np.nextafter(cutoff, np.float32(-np.inf))
    return (values <= float(cutoff)).sum(dtype=torch.int64).to(torch.float64) / len(values)
