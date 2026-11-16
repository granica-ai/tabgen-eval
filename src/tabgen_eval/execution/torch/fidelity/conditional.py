"""Conditional fidelity over reference-defined slices with bounded CUDA scratch.

Each draw preserves duplicate source rows and per-cell missingness. Numeric
conditioning labels use reference equal-width bins; within-slice contingencies
fit reference quantiles and deliberately have a different infinity policy.
Retained reference groups are packed by padded size, then reused across batched
numeric KS, correlations and categorical TV work. Category compaction and pair
plans are fitted once per draw; generated-only identities cannot change reference
slice membership. Exact integer counts govern recovery and float64 rarity tests;
score arithmetic stays in selected precision. Group and conditioning-column
means are unweighted. Executors return six host arrays; sampling and uncertainty
remain outside this module.
"""

from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Iterable, Sequence
from itertools import combinations
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
import torch
from numpy.typing import ArrayLike

from tabgen_eval._typing import FloatArray, IntArray
from tabgen_eval.execution.runtime.components import CONDITIONAL_COMPONENTS, selected_components
from tabgen_eval.execution.torch.cpu.executor import CppExecutor
from tabgen_eval.execution.torch.fidelity.conditional_details import ConditionalDetails
from tabgen_eval.execution.torch.fidelity.details import host_array
from tabgen_eval.execution.torch.fidelity.grouped_ops import (
    group_bins,
    group_center,
    group_dense_tv,
    group_edges,
    group_gather,
    group_gram,
    group_keys,
    group_ks,
)
from tabgen_eval.execution.torch.fidelity.ops import joint_sorted_tv
from tabgen_eval.execution.torch.fidelity.primitives import bin_codes, equal_width_edges

if TYPE_CHECKING:
    from tabgen_eval.execution.runtime.details import DetailedOutcome, FidelityDetails
    from tabgen_eval.execution.torch.context import TorchContext
    from tabgen_eval.execution.torch.tensors import TensorTable
    from tabgen_eval.metrics.conditional import Conditional
    from tabgen_eval.schema import Schema

PairBuckets: TypeAlias = Sequence[tuple[int, torch.Tensor]]
SlicePlan: TypeAlias = tuple[int, int, list[int], int]
GroupedColumns: TypeAlias = tuple[torch.Tensor, torch.Tensor]


def mean(values: torch.Tensor, dim: int) -> torch.Tensor:
    """Average finite entries along ``dim`` with unweighted selected-precision arithmetic.

    Nonfinite scores contribute neither value nor count; an empty applicable
    set returns NaN. The input tensor is read-only. Conditional uses this first
    across retained groups, then equally across conditioning columns.
    """
    valid = torch.isfinite(values)
    count = valid.sum(dim=dim)
    result = torch.where(valid, values, 0).sum(dim=dim) / count.clamp(min=1).to(values.dtype)
    return torch.where(count > 0, result, float("nan"))


def centered(values: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize grouped FP32/FP64 ``[G,P,R]`` columns with explicit bool validity.

    Make contiguous kernel inputs if needed and return owned centered values
    plus bool ``[G,P]`` variation flags. Invalid/padded cells become zero;
    validity can also describe pairwise overlap in cancellation fallbacks.
    """
    return group_center(values.contiguous(), valid.contiguous())


def correlations(
    values: torch.Tensor, rows: torch.Tensor, scratch: int, complete: bool | None = None
) -> torch.Tensor:
    """Return pairwise-finite ``[G,P,P]`` correlations for padded grouped columns.

    ``values`` is FP32/FP64 ``[G,P,R]`` with NaN padding and ``rows`` gives each
    group's true int64 row count. A true ``complete`` hint certifies no missing
    numeric cells within real rows; padding is still excluded. Otherwise a
    host scalar check selects either one-Gram or masked-moment computation.

    Counts stay integer until normalization. Near-degenerate masked moments
    are recomputed in bounded batches over explicit pairwise overlap, using
    the same precision rather than refinement. ``scratch`` is a byte allowance
    and must fit one fallback pair. Inputs remain read-only; only owned output
    entries are replaced by fallback results. Unusable pairs yield NaN.
    """
    g, p, n = values.shape
    if not p:
        return values.new_empty((g, 0, 0))
    valid = torch.isfinite(values)
    z, variable = centered(values, valid)
    cross = group_gram(z.contiguous(), z.contiguous())
    if complete is None:
        complete = bool((valid.sum(dim=2) == rows[:, None]).all())
    if complete:
        norms = cross.diagonal(dim1=1, dim2=2)
        usable = (
            variable[:, :, None]
            & variable[:, None, :]
            & (norms[:, :, None] > 0)
            & (norms[:, None, :] > 0)
        )
        rho = cross / (norms[:, :, None] * norms[:, None, :]).sqrt()
        return torch.where(usable, rho.clamp(-1, 1), float("nan"))
    count = group_gram(valid.contiguous(), valid.contiguous(), True)
    denominator = count.clamp(min=1).to(values.dtype)
    masks = valid.to(values.dtype)
    sums = group_gram(z.contiguous(), masks.contiguous())
    squares = group_gram((z * z).contiguous(), masks.contiguous())
    correction = sums * sums / denominator
    variance = squares - correction
    covariance = cross - sums * sums.transpose(1, 2) / denominator
    usable = (count >= 2) & variable[:, :, None] & variable[:, None, :]
    rho = covariance / (variance * variance.transpose(1, 2)).clamp(min=0).sqrt()
    rho = torch.where(
        usable & (variance > 0) & (variance.transpose(1, 2) > 0), rho.clamp(-1, 1), float("nan")
    )
    uncertainty = 32 * torch.finfo(values.dtype).eps * (squares.abs() + correction.abs())
    suspect = usable & (
        (variance <= uncertainty) | (variance.transpose(1, 2) <= uncertainty.transpose(1, 2))
    )
    bad = torch.nonzero(torch.triu(suspect, diagonal=1))
    per_pair = n * (16 * values.element_size() + 32)
    if len(bad) and per_pair > scratch:
        raise MemoryError(
            "one conditional correlation pair exceeds the resolved workspace allowance"
        )
    batch = max(1, min(64, scratch // max(per_pair, 1)))
    for start in range(0, len(bad), batch):
        selected = bad[start : start + batch]
        a = values[selected[:, 0], selected[:, 1]]
        b = values[selected[:, 0], selected[:, 2]]
        pair = torch.stack((a, b), dim=1)
        keep = torch.isfinite(a) & torch.isfinite(b)
        z, varies = centered(pair, keep[:, None, :].expand_as(pair))
        norms = (z * z).sum(dim=2)
        score = (z[:, 0] * z[:, 1]).sum(dim=1) / norms.prod(dim=1).sqrt()
        score = torch.where(
            varies.all(dim=1) & (norms > 0).all(dim=1), score.clamp(-1, 1), float("nan")
        )
        rho[selected[:, 0], selected[:, 1], selected[:, 2]] = score
        rho[selected[:, 0], selected[:, 2], selected[:, 1]] = score
    return rho


def ordered(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Sort each ``[G,P,R]`` numeric column, placing nonfinite/padded cells last.

    Return owned selected-precision values and exact int64 ``[G,P]`` finite
    prefix sizes. The grouping rows need not be complete across columns.
    """
    valid = torch.isfinite(values)
    return torch.where(valid, values, float("inf")).sort(dim=2).values, valid.sum(dim=2)


def edges(
    ordered_values: torch.Tensor, counts: torch.Tensor, bins: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit within-slice quantile edges from sorted grouped finite prefixes.

    Delegate to the fused edge kernel. ``ordered_values`` is ``[G,P,R]``, ``counts``
    int64 ``[G,P]``; outputs are fixed-capacity ``[G,P,bins+1]`` boundaries and
    int64 active bin counts after duplicate compression. Inputs are read-only.
    """
    return group_edges(ordered_values.contiguous(), counts.contiguous(), bins)


def compact_categories(
    reference: torch.Tensor, candidate: torch.Tensor, cards: Sequence[int]
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Keep real-supported identities and merge generated-only TV mass.

    Every merged identity has zero reference probability, so its contribution
    to marginal or joint total variation is unchanged. Actual missing cells
    retain -1, and the original tables still define conditioning and Recovery.
    Compaction is fitted again for each draw, including ideal/bootstrap draws.
    """
    real, generated = torch.empty_like(reference), torch.empty_like(candidate)
    levels = []
    for col, card in enumerate(cards):
        if not card:
            real[:, col] = generated[:, col] = -1
            levels.append(reference.new_zeros(()))
            continue
        r, s = reference[:, col], candidate[:, col]
        present = reference.new_zeros((card,))
        present.scatter_add_(0, r.clamp(min=0), (r >= 0).to(torch.int64))
        present = present > 0
        known = present.sum()
        mapping = torch.where(present, present.to(torch.int64).cumsum(0) - 1, known)
        real[:, col] = torch.where(r >= 0, mapping[r.clamp(min=0)], -1)
        generated[:, col] = torch.where(s >= 0, mapping[s.clamp(min=0)], -1)
        levels.append(known + (known < card).to(torch.int64))
    return real, generated, torch.stack(levels).cpu().tolist() if levels else []


def _dense_extension_wins(plan: PairBuckets, groups: int, rows: int, scratch: int) -> bool:
    """Promote medium domains only when they halve a costly sparse launch loop.

    Scratch estimates match ``tv_scores``. Small slices and already efficient
    sparse batches keep the premerged fallback; no category product is truncated.
    """
    promoted = sum(len(pairs) for grid, pairs in plan if grid == 8192)
    if not promoted or groups <= 0:
        return False
    sparse = sum(len(pairs) for grid, pairs in plan if not grid)
    sparse_cost = groups * (rows * 80 + 128)
    dense_cost = groups * (8192 * 64 + 128)
    if dense_cost > min(sparse_cost, scratch):
        return False
    limit = min(1024, 65535 // max(groups, 1))
    old_batch = max(1, min(limit, scratch // sparse_cost))
    new_batch = max(1, min(limit, scratch // dense_cost))
    old_calls = (promoted + sparse + old_batch - 1) // old_batch
    new_calls = (promoted + new_batch - 1) // new_batch + (sparse + old_batch - 1) // old_batch
    return old_calls >= 8 and old_calls >= 2 * new_calls


def _extended_pairs_possible(pn: int, cards: Sequence[int], bins: int) -> bool:
    """Screen unchanged pool domains for an optional medium histogram bucket."""
    return (
        any(4096 < card <= 8192 for card in cards)
        or (pn > 0 and any(4096 < bins * card <= 8192 for card in cards))
        or any(4096 < a * b <= 8192 for a, b in combinations(cards, 2))
    )


class _PairPlan(list[tuple[int, torch.Tensor]]):
    """Own reusable CUDA pair views and an optional premerged sparse fallback.

    Elements are ``(power_of_two_grid_or_zero, int64[Q,2] view)`` into one
    uploaded pair table. ``sparse_fallback`` merges 8192-cell and occupied-key
    pairs without rebuilding Python lists/device tensors for every group batch.
    The plan is draw-local and reused across all conditioning columns.
    """

    sparse_fallback: list[tuple[int, torch.Tensor]] | None


def pair_plan(
    pn: int, cards: Sequence[int], bins: int, device: str | torch.device
) -> tuple[_PairPlan, _PairPlan]:
    """Upload marginal and mixed/categorical pair metadata once per draw.

    Numeric axes occupy [0,pn), categorical axes follow with sizes ``cards``;
    within-slice numeric axes have at most ``bins`` active intervals. Return
    marginal and joint _PairPlan objects referencing one owned int64 CUDA table.
    Marginals use diagonal categorical pairs; numeric/numeric joint pairs are
    handled by correlations and omitted here.

    Pair domains select power-of-two grids through 8192 or occupied-key sorting.
    The 8192 bucket includes a premerged sparse fallback so small slices need
    not pay extra launches. Products exceeding int64 fail before uploading.
    Plans are conservative upper bounds; active numeric levels remain group-specific.
    """
    maxima = [bins] * pn + list(cards)
    flat: list[tuple[int, int]] = []
    spans: list[tuple[list[tuple[int, int, int]], tuple[int, int] | None]] = []
    for marginal in (True, False):
        groups: defaultdict[int, list[tuple[int, int]]] = defaultdict(list)
        identities = (
            ((j, j) for j in range(pn, pn + len(cards)))
            if marginal
            else ((i, j) for i, j in combinations(range(len(maxima)), 2) if j >= pn)
        )
        for i, j in identities:
            product = maxima[j] if marginal else maxima[i] * maxima[j]
            if product > torch.iinfo(torch.int64).max:
                raise OverflowError("joint category product exceeds int64 key capacity")
            if product:
                grid = 1 << (product - 1).bit_length() if product <= 8192 else 0
                groups[grid].append((i, j))
        plan = []
        for grid, pairs in groups.items():
            start = len(flat)
            flat.extend(pairs)
            plan.append((grid, start, len(flat)))
        fallback = None
        if 8192 in groups:
            start = len(flat)
            flat.extend(groups[8192])
            flat.extend(groups.get(0, []))
            fallback = (start, len(flat))
        spans.append((plan, fallback))
    table = torch.tensor(flat, dtype=torch.int64, device=device).reshape(-1, 2)
    plans: list[_PairPlan] = []
    for plan, fallback in spans:
        batches = _PairPlan((grid, table[start:end]) for grid, start, end in plan)
        batches.sparse_fallback = (
            [(grid, pairs) for grid, pairs in batches if grid not in (0, 8192)]
            + [(0, table[fallback[0] : fallback[1]])]
            if fallback
            else None
        )
        plans.append(batches)
    return plans[0], plans[1]


def tv_scores(
    a: torch.Tensor,
    b: torch.Tensor,
    plan: PairBuckets,
    levels: torch.Tensor,
    prototype: torch.Tensor,
    scratch: int,
    *,
    marginal: bool = False,
    cond: int = -1,
) -> torch.Tensor:
    """Compute grouped marginal or joint TV similarities under a byte scratch cap.

    ``a``/``b`` are int64 ``[G,P,NR/NS]`` with -1 exclusions; ``levels`` is int64
    ``[G,P]``. ``plan`` holds shared ``[Q,2]`` pair views and dense-grid bounds.
    Return ``[G,total_pairs]`` similarities in ``prototype`` precision. Empty
    slices, inactive joint axes and pairs touching ``cond`` produce NaN.

    Choose dense integer histograms only when their live storage and launch
    count justify them; otherwise sort occupied keys. The optional native dense
    batch launcher consumes the same plan and scoring contract. Scratch and the
    CUDA group/pair grid bound cap each pair batch; a single pair exceeding the
    budget raises MemoryError. Inputs/plans are read-only and all count/sort
    scratch is draw-local. This routine excludes the conditioning axis rather
    than deleting source columns, preserving pair-table reuse.
    """
    g = len(a)
    fallback = getattr(plan, "sparse_fallback", None)
    if fallback is not None and not _dense_extension_wins(
        plan, g, a.shape[2] + b.shape[2], scratch
    ):
        plan = fallback
    result = []
    for grid, pairs in plan:
        # Larger grids use dense counting only when it is cheaper in scratch.
        if grid > 4096 and grid * 64 > (a.shape[2] + b.shape[2]) * 80:
            grid = 0
        per_pair = (
            g * (grid * 8 * 8 + 128) if grid else g * ((a.shape[2] + b.shape[2]) * 8 * 10 + 128)
        )
        if per_pair > scratch:
            raise MemoryError(
                "one grouped contingency pair exceeds the resolved workspace allowance"
            )
        batch = max(1, min(1024, 65535 // max(g, 1), scratch // per_pair))
        if grid and os.environ.get("TABGEN_EVAL_CONDITIONAL_NATIVE") == "1":
            from tabgen_eval.execution.torch.cuda.fidelity.native_tv import dense_tv_batches

            result.extend(
                dense_tv_batches(a, b, pairs, levels, grid, batch, cond, marginal, prototype)
            )
            continue
        for start in range(0, len(pairs), batch):
            indices = pairs[start : start + batch]
            if grid:
                result.append(
                    group_dense_tv(a, b, indices, levels, grid, cond, marginal, prototype)
                )
                continue
            widths = (
                torch.zeros((g, len(indices)), dtype=torch.int64, device=a.device)
                if marginal
                else levels[:, indices[:, 1]].contiguous()
            )
            ar = group_keys(a, indices, widths).sort(dim=1).values
            bs = group_keys(b, indices, widths).sort(dim=1).values
            zero = indices.new_zeros((len(ar), 1))
            nr = a.shape[2] - torch.searchsorted(ar, zero).squeeze(1)
            ns = b.shape[2] - torch.searchsorted(bs, zero).squeeze(1)
            delta = joint_sorted_tv(ar, bs, nr, ns, prototype).view(g, -1)
            nr, ns = nr.view(g, -1), ns.view(g, -1)
            score = torch.where((nr > 0) & (ns > 0), (1 - 0.5 * delta).clamp(0, 1), float("nan"))
            keep = (indices[:, 0] != cond) & (indices[:, 1] != cond)
            if not marginal:
                keep = keep & (levels[:, indices[:, 0]] > 0) & (levels[:, indices[:, 1]] > 0)
            result.append(torch.where(keep, score, float("nan")))
    return torch.cat(result, dim=1) if result else prototype.new_empty((g, 0))


def group_plan(
    metadata: Sequence[Sequence[int]],
    minimum: int,
    pn: int,
    pc: int,
    element: int,
    bins: int,
    cards: Sequence[int],
    available: int,
    grid: int | None = None,
) -> list[SlicePlan]:
    """Pack retained reference slices into size-batched padded gather plans.

    ``metadata`` contains host integer ``(reference_count,candidate_count)``
    per reference label. Retain groups meeting ``minimum`` on the reference side,
    sort by largest side, then pack at most 128 groups using power-of-two row
    capacities. An absent candidate still reserves one padded row.

    ``pn``, ``pc``, ``element``, ``bins`` and ``cards`` bound live gather, moments,
    quantile and contingency storage. ``available`` is the remaining byte budget
    and ``grid`` can supply an already computed maximum small dense grid.
    Each tuple is ``(reference_padding,candidate_padding,group_ids,live_bytes_per_group)``.
    Reserve additional pair scratch before accepting a group; reject a group
    that cannot fit alone. Only counts cross to the host, never grouped cell data.
    """
    if grid is None:
        maxima = [bins] * pn + list(cards)
        grids = [1 << (c - 1).bit_length() for c in cards if 0 < c <= 4096]
        for i, j in combinations(range(pn + pc), 2):
            product = maxima[i] * maxima[j]
            if j >= pn and 0 < product <= 4096:
                grids.append(1 << (product - 1).bit_length())
        grid = max(grids, default=0)

    def costs(nr: int, ns: int) -> tuple[int, int]:
        """Estimate per-group live bytes and live-plus-two-pair scratch for padded lengths.

        The conservative bound includes numeric copies/masks, exact category codes,
        Gram chunks, fitted edges and the larger dense/occupied pair requirement.
        It is a planning estimate, not an allocation or measurement.
        """
        gram_chunks = (max(nr, ns) + 4095) // 4096
        buffers = (
            (nr + ns) * (pn * (8 * element + 40) + pc * 40 + 32)
            + pn * pn * (gram_chunks + 16) * max(element, 8)
            + pn * (bins + 1) * (4 * element + 16)
            + 4096
        )
        pair = max(grid * 64 + 128, (nr + ns) * 80 + 128)
        return buffers, buffers + 2 * pair

    groups = sorted(
        (g for g, (nr, _) in enumerate(metadata) if nr >= minimum), key=lambda g: max(metadata[g])
    )
    plans: list[SlicePlan] = []
    selected: list[int] = []
    nr, ns = 1, 1
    for group in groups:
        a, b = metadata[group]
        a, b = 1 << (a - 1).bit_length(), 1 << (max(b, 1) - 1).bit_length()
        next_nr, next_ns = max(nr, a), max(ns, b)
        _, total = costs(next_nr, next_ns)
        if selected and (len(selected) == 128 or (len(selected) + 1) * total > available):
            plans.append((nr, ns, selected, costs(nr, ns)[0]))
            selected, nr, ns = [], 1, 1
            next_nr, next_ns = a, b
            _, total = costs(a, b)
        if total > available:
            raise MemoryError("one Conditional group exceeds the resolved workspace allowance")
        selected.append(group)
        nr, ns = next_nr, next_ns
    if selected:
        plans.append((nr, ns, selected, costs(nr, ns)[0]))
    return plans


class ConditionalExecutor:
    """Compute six unweighted Conditional statistics over retained slices.

    Parameters
    ----------
    context : TorchContext
        Read-only source pools, selected precision and bounded GPU workspace.

    Notes
    -----
    Per-column group plans batch retained slices by size. Integer counts govern
    recovery/rare structure; within-slice binning and numeric scores retain selected
    precision. Aggregation is unweighted within each column and across columns.
    """

    def __init__(self, context: TorchContext) -> None:
        """Retain source context and select whether draw-local category compaction may help.

        Pool-level vocabulary sizes only screen execution cost. Actual known-category
        membership is refit from each reference draw, preventing generated-only levels
        from inflating dense contingency domains without altering slice definitions.
        """
        self.context = context
        support = getattr(context, "reference_cardinalities", None)
        # Only pay for compaction when generated-only vocabulary dominates.
        # This pool-level screen selects execution, not draw-fitted membership.
        self._compact = support is None or sum(context.inputs.cardinalities) > 2 * sum(support)

    @staticmethod
    def batch_size(metric: Conditional, n: int) -> int:
        """Bound draw-task scheduling to 1..64 under an 8 MiB index allowance.

        The byte-intensive group plan is built separately for every actual draw;
        this hint never assumes equally sized retained slices across bootstrap tasks.
        """
        return max(1, min(64, (8 * 1024**2) // max(16 * n, 1)))

    @staticmethod
    def labels(
        table: TensorTable,
        cond: int,
        pn: int,
        bins: int,
        reference: TensorTable | None = None,
    ) -> torch.Tensor:
        """Return one conditioning axis as shared int64 category/bin identities.

        Categorical labels are read-only views of source identities. Numeric labels
        use equal-width reference-fitted edges; when ``reference`` is supplied,
        candidate values are clipped, including infinities, before bin assignment.
        Missing categories/NaNs use -1 and do not define conditioning groups.
        """
        if cond >= pn:
            return table.categorical[:, cond - pn]
        e = equal_width_edges((reference or table).numeric[:, cond], bins)
        return bin_codes(table.numeric[:, cond], e, clip=reference is not None, clip_infinite=True)

    def not_applicable(self, metric: Conditional, schema: Schema) -> set[str]:
        """Find structural and reference-data rare-slice exclusions for report routing.

        Width controls whether Shape or Trend can exclude a conditioning column and
        still have enough axes. Rare applicability uses exact retained reference
        counts and a float64 rarity comparison; generated data never determines
        whether a rare statistic exists. This scan does not fit draw-specific plans.
        """
        c = self.context
        width = c.n_numeric + c.n_categorical
        if not width:
            return set(metric.stat_names)
        result = {"cond_trend_all", "cond_trend_rare"} if width < 3 else set()
        if width < 2:
            result |= {"cond_shape_all", "cond_shape_rare"}
        _, minimum, bins, rare = CppExecutor._parameters(metric)
        with torch.autocast("cuda", enabled=False):
            for cond in range(width):
                labels = self.labels(c.reference, cond, c.n_numeric, bins)
                _, counts = labels[labels >= 0].unique(return_counts=True)
                if len(counts) and bool(
                    ((counts >= minimum) & (counts.to(torch.float64) / counts.sum() < rare)).any()
                ):
                    return result
        return result | {"cond_shape_rare", "cond_trend_rare", "recovery_rare"}

    def _scores(
        self,
        a: GroupedColumns,
        b: GroupedColumns,
        nr: torch.Tensor,
        ns: torch.Tensor,
        cond: int,
        bins: int,
        scratch: int,
        cards: Sequence[int] | None = None,
        pairs: tuple[PairBuckets, PairBuckets] | None = None,
        complete: Sequence[bool | None] = (None, None),
        requested_components: Sequence[str] = CONDITIONAL_COMPONENTS,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score a padded batch of slices without rebuilding its shared pair plan.

        ``a``/``b`` hold numeric/category tensors ``[G,PN/PC,R/S]``; ``nr``/``ns``
        are int64 true row counts. ``cards`` and ``pairs`` may supply draw-compacted
        domains. ``complete`` hints select numeric moment work independently per side.
        Within-slice numeric bins come only from the reference slice and exclude
        original infinities on joint axes. Conditioning labels follow a distinct
        clipping convention established before this routine.

        Produce Shape and Trend ``[G]`` in selected precision. Exclude ``cond`` from
        all marginal/pair means, preserve per-cell missingness, and score empty
        candidate slices as zero only for structurally applicable statistics.
        Inputs are read-only; masking updates owned KS output only. ``scratch``
        bounds contingency/correlation fallback buffers.
        """
        pn = self.context.n_numeric
        cards = self.context.inputs.cardinalities if cards is None else cards
        pairs = pair_plan(pn, cards, bins, a[0].device) if pairs is None else pairs
        an, ac, bn, bc = a[0], a[1], b[0], b[1]
        prototype = an.new_empty(())
        wants_shape = "cond_shape" in requested_components and pn + len(cards) >= 2
        wants_trend = "cond_trend" in requested_components and pn + len(cards) >= 3
        shape = an.new_full((len(an),), float("nan"))
        trend = torch.full_like(shape, float("nan"))
        if wants_shape or (wants_trend and cards):
            ao, na = ordered(an)
        if wants_shape:
            bo, nb = ordered(bn)
            shapes = group_ks(ao.contiguous(), bo.contiguous(), na.contiguous(), nb.contiguous())
        if wants_trend and cards:
            e, nl = edges(ao, na, bins)
            rnum = group_bins(an, e, nl, False)
            snum = group_bins(bn, e, nl, True)
        else:
            nl = torch.zeros((len(an), pn), dtype=torch.int64, device=an.device)
            rnum = torch.zeros(an.shape, dtype=torch.int64, device=an.device)
            snum = torch.zeros(bn.shape, dtype=torch.int64, device=an.device)
        cat_levels = torch.tensor(cards, device=an.device, dtype=torch.int64).expand(len(an), -1)
        levels = torch.cat((nl, cat_levels), dim=1)
        rc = torch.cat((rnum, ac), dim=1).contiguous()
        sc = torch.cat((snum, bc), dim=1).contiguous()
        if wants_shape:
            categorical = tv_scores(
                rc, sc, pairs[0], levels, prototype, scratch, marginal=True, cond=cond
            )
            if cond < pn:
                shapes[:, cond] = float("nan")
            shape = torch.where(ns > 0, mean(torch.cat((shapes, categorical), dim=1), 1), 0)
        if wants_trend:
            parts = []
            if pn >= 2:
                ar, bs = (
                    correlations(an, nr, scratch, complete[0]),
                    correlations(bn, ns, scratch, complete[1]),
                )
                i, j = torch.triu_indices(pn, pn, offset=1, device=an.device)
                keep = (i != cond) & (j != cond)
                parts.append(1 - 0.5 * (ar[:, i[keep], j[keep]] - bs[:, i[keep], j[keep]]).abs())
            parts.append(tv_scores(rc, sc, pairs[1], levels, prototype, scratch, cond=cond))
            trend = torch.where(ns > 0, mean(torch.cat(parts, dim=1), 1), 0)
        return shape, trend

    def _draw(
        self,
        other: TensorTable,
        rd: IntArray,
        sd: IntArray,
        minimum: int,
        bins: int,
        rare: float,
        *,
        detail_names: Sequence[str] | None = None,
        requested_components: Iterable[str] | None = CONDITIONAL_COMPONENTS,
    ) -> tuple[FloatArray, FidelityDetails | None]:
        """Compute six statistics for one owned reference/candidate draw pair.

        Reserve gathers, grouping, pair metadata and optional category compaction
        before reading source rows. For each conditioning axis, fit reference
        numeric edges once, stable-sort reference labels, and keep only candidate
        rows matching a reference label. Labels absent from the reference never
        create slices. Source duplicates retain their multiplicity throughout.

        Each size-batched plan computes Shape, Trend and relative frequency recovery.
        Recovery divides each side's group frequency by its own total nonmissing
        conditioning population; candidate-only valid labels still enter its total.
        Rarity uses float64 reference frequencies before retained-group averaging.
        Both group and conditioning-column means are unweighted; unsupported entries
        remain NaN. Return a six-element host NumPy array in the metric's fixed order;
        source pools are not mutated and batch-local gathers are released promptly.
        The second returned value contains optional descriptive host tables.
        """
        c = self.context
        pn, pc = c.n_numeric, c.n_categorical
        components = selected_components(requested_components, CONDITIONAL_COMPONENTS)
        wants_shape = "cond_shape" in components and pn + pc >= 2
        wants_trend = "cond_trend" in components and pn + pc >= 3
        wants_scores = wants_shape or wants_trend
        budget = c.gpu_workspace_mb * 1024**2
        element = c.reference.numeric.element_size()
        reserved = (len(rd) + len(sd)) * (pn * element + pc * 8 + 160) + 65536
        # One device pair table plus a worst-case merged sparse fallback,
        # shared by all conditioning/group batches.
        pair_bytes = (
            32
            if self._compact or _extended_pairs_possible(pn, c.inputs.cardinalities, bins)
            else 16
        )
        if wants_scores:
            reserved += (pc + pc * pn + pc * (pc - 1) // 2) * pair_bytes
        if pc and self._compact and wants_scores:
            # Owned compact tables plus the per-column fitting/encoding scratch.
            reserved += (len(rd) + len(sd)) * pc * 8
            reserved += max(len(rd), len(sd)) * 64
            reserved += max(c.inputs.cardinalities, default=0) * 64 + pc * 1024
        if reserved >= budget:
            raise MemoryError(
                "Conditional draws and grouping exceed the resolved workspace allowance"
            )
        a, b = c.reference.gather(rd), other.gather(sd)
        if self._compact and wants_scores:
            ac, bc, cards = compact_categories(a.categorical, b.categorical, c.inputs.cardinalities)
        else:
            ac, bc, cards = a.categorical, b.categorical, c.inputs.cardinalities
        pairs = pair_plan(pn, cards, bins, a.numeric.device) if wants_scores else ([], [])
        grid = max((width for plan in pairs for width, _ in plan if width <= 4096), default=0)
        # Complete draws skip every batch-local scalar check. Incomplete draws
        # retain the fast/masked slice choice, avoiding extra Gram products for
        # groups whose numeric rows are complete.
        complete = (None, None)
        if wants_trend:
            complete = torch.stack([torch.isfinite(x.numeric).all() for x in (a, b)]).cpu().tolist()
            complete = [True if value else None for value in complete]
        columns = a.numeric.new_full((pn + pc, 6), float("nan"))
        details = (
            ConditionalDetails(pn + pc, pn, detail_names, minimum, rare, a.numeric.dtype)
            if detail_names is not None
            else None
        )
        for cond in range(pn + pc):
            # One reference edge construction and one grouping pass per column.
            if cond < pn:
                e = equal_width_edges(a.numeric[:, cond], bins)
                r = bin_codes(a.numeric[:, cond], e, clip_infinite=True)
                s = bin_codes(b.numeric[:, cond], e, clip=True, clip_infinite=True)
            else:
                r, s = a.categorical[:, cond - pn], b.categorical[:, cond - pn]
            valid_r, valid_s = r >= 0, s >= 0
            rrows, srows = torch.nonzero(valid_r).flatten(), torch.nonzero(valid_s).flatten()
            rl, ro = r[rrows].sort(stable=True)
            labels, counts = rl.unique_consecutive(return_counts=True)
            if not len(labels):
                continue
            rorder = rrows[ro]
            rcounts = counts
            roffsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
            sl = torch.searchsorted(labels, s[srows])
            exists = (sl < len(labels)) & (labels[sl.clamp(max=len(labels) - 1)] == s[srows])
            srows, sl = srows[exists], sl[exists]
            sl, so = sl.sort(stable=True)
            sorder = srows[so]
            scounts = torch.bincount(sl, minlength=len(labels))
            soffsets = torch.cat((scounts.new_zeros(1), scounts.cumsum(0)))
            metadata = torch.stack((rcounts, scounts), dim=1).cpu().tolist()
            plans = (
                group_plan(metadata, minimum, pn, pc, element, bins, cards, budget - reserved, grid)
                if wants_scores
                else [(0, 0, [index for index, (nr, _) in enumerate(metadata) if nr >= minimum], 0)]
            )
            total_r, total_s = len(rrows), int(valid_s.sum().item())
            all_scores, rare_scores = [], []
            group_ids = [] if details is not None else None
            for nr_pad, ns_pad, groups, per_group in plans:
                if not groups:
                    continue
                selected = torch.tensor(groups, device=a.numeric.device, dtype=torch.int64)
                nr, ns = rcounts[selected], scounts[selected]
                scratch = (budget - reserved - len(selected) * per_group) // 2
                if wants_scores:
                    ar = (
                        group_gather(a.numeric, rorder, roffsets, selected, nr_pad),
                        group_gather(ac, rorder, roffsets, selected, nr_pad),
                    )
                    bs = (
                        group_gather(b.numeric, sorder, soffsets, selected, ns_pad),
                        group_gather(bc, sorder, soffsets, selected, ns_pad),
                    )
                    shape, trend = self._scores(
                        ar,
                        bs,
                        nr,
                        ns,
                        cond,
                        bins,
                        scratch,
                        cards,
                        pairs,
                        complete,
                        requested_components=components,
                    )
                    del ar, bs
                else:
                    shape = a.numeric.new_full((len(groups),), float("nan"))
                    trend = torch.full_like(shape, float("nan"))
                frequency = nr.to(a.numeric.dtype) / total_r
                recovery = (
                    (ns.to(a.numeric.dtype) / total_s) / frequency
                    if total_s
                    else torch.zeros_like(frequency)
                )
                if "recovery" not in components:
                    recovery.fill_(float("nan"))
                scores = torch.stack((shape, trend, recovery), dim=1)
                is_rare = nr.to(torch.float64) / total_r < rare
                all_scores.append(scores)
                rare_scores.append(torch.where(is_rare[:, None], scores, float("nan")))
                if details is not None:
                    group_ids.extend(groups)
                del selected, shape, trend, scores
            if all_scores:
                combined = torch.cat(all_scores)
                columns[cond, [0, 2, 4]] = mean(combined, 0)
                host_scores = host_array(combined) if details is not None else None
                del combined
                columns[cond, [1, 3, 5]] = mean(torch.cat(rare_scores), 0)
            else:
                host_scores = None
            if details is not None:
                details.add(
                    cond,
                    labels,
                    metadata,
                    total_r,
                    total_s,
                    group_ids,
                    host_scores,
                    e if cond < pn else None,
                )
        result = mean(columns, 0).cpu().numpy()
        return result, details.finish(columns) if details is not None else None

    @torch.no_grad()
    def compute_detailed(
        self,
        metric: Conditional,
        reference_draw: ArrayLike | torch.Tensor,
        candidate_draw: ArrayLike | torch.Tensor,
        *,
        ideal: bool = False,
    ) -> DetailedOutcome:
        """Capture one point's column and slice scores from its existing reductions.

        Full reference-observed slice structure is retained, including groups
        below the minimum count. Transfers operate on existing vectors/batches,
        leaving GPU workspace, group plans and scalar arithmetic unchanged.
        """
        from tabgen_eval.execution.runtime.details import DetailedOutcome
        from tabgen_eval.execution.torch.fidelity.executor import FidelityExecutor

        if metric.name != "conditional_fidelity":
            raise ValueError("conditional detail execution requires Conditional")
        _, minimum, bins, rare = CppExecutor._parameters(metric)
        c = self.context
        other = c.reference if ideal else c.candidate
        rd = FidelityExecutor._point_draw(reference_draw, c.reference.n_rows)
        sd = FidelityExecutor._point_draw(candidate_draw, other.n_rows)
        with torch.autocast("cuda", enabled=False):
            values, details = self._draw(
                other,
                rd,
                sd,
                minimum,
                bins,
                rare,
                detail_names=metric.stat_names,
                requested_components=metric.requested_components,
            )
        assert details is not None
        return DetailedOutcome(dict(zip(metric.stat_names, values, strict=True)), details)

    @torch.no_grad()
    def compute(
        self,
        metric: Conditional,
        reference_draws: ArrayLike | torch.Tensor,
        candidate_draws: ArrayLike | torch.Tensor,
        *,
        ideal: bool = False,
    ) -> dict[str, FloatArray]:
        """Compute ordered observed/ideal Conditional draw batches.

        Parameters
        ----------
        metric : Conditional
            Minimum reference count, conditioning bins and rare threshold.
        reference_draws, candidate_draws : array-like
            Integer ``[B, N]`` source indices; duplicates and row order are preserved.
        ideal : bool
            Draw both sides from the reference pool while preserving supplied indices.

        Returns
        -------
        dict of numpy.ndarray
            Six statistic arrays of length B, including NaN when no slices apply.

        Notes
        -----
        Sources are read-only, ambient autocast is locally disabled and draw-local
        buffers are released before the next task. Empty candidate slices score zero
        for applicable statistics; unavailable statistics remain NaN.
        """
        from tabgen_eval.execution.torch.fidelity.executor import FidelityExecutor

        _, minimum, bins, rare = CppExecutor._parameters(metric)
        c = self.context
        other = c.reference if ideal else c.candidate
        rd = FidelityExecutor._draws(reference_draws, c.reference.n_rows)
        sd = FidelityExecutor._draws(candidate_draws, other.n_rows)
        if len(rd) != len(sd):
            raise ValueError("draw arrays must have equal batch lengths")
        out = np.empty((len(rd), 6))
        with torch.autocast("cuda", enabled=False):
            for i, (r, s) in enumerate(zip(rd, sd, strict=True)):
                out[i], _ = self._draw(
                    other,
                    r,
                    s,
                    minimum,
                    bins,
                    rare,
                    requested_components=metric.requested_components,
                )
        return {name: out[:, i] for i, name in enumerate(metric.stat_names)}
