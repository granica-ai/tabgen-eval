"""Shape/Trend draw execution and conservative fidelity workspace accounting.

Draws retain duplicate source positions and per-column missingness. Executors
return host statistic arrays; sampling, bootstrap calibration and report
construction remain outside the numerical execution layer.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations
from typing import TYPE_CHECKING, Literal, cast, overload

import numpy as np
import torch
from numpy.typing import ArrayLike
from torch import Tensor

from tabgen_eval._typing import FloatArray, IntegerArray
from tabgen_eval.execution.runtime.details import DetailedOutcome, DetailTable, FidelityDetails
from tabgen_eval.execution.torch.fidelity.details import host_array
from tabgen_eval.execution.torch.fidelity.primitives import (
    bin_codes,
    contingency_scores,
    correlations,
    quantile_edges,
)

if TYPE_CHECKING:
    from tabgen_eval.execution.torch.context import TorchContext
    from tabgen_eval.execution.torch.tensors import TensorTable
    from tabgen_eval.metrics import Metric
    from tabgen_eval.schema import Schema


class FidelityWorkspace:
    """Estimate live Trend buffers before gathering a pair of draws.

    ``context`` supplies numeric width, category width, selected precision and
    workspace limit; ``rows`` includes each draw's repeated source positions.
    Owned draw storage, bin codes, preprocessing, and Gram partials are reserved
    separately from pair scratch. CUDA enforces the byte budget; the CPU/reference
    path uses a bounded default contingency scratch allowance.
    """

    def __init__(self, context: TorchContext, rows: Sequence[int]) -> None:
        """Compute reserved bytes and remaining pair scratch without owning tensors.

        CUDA requires fixed live buffers plus a minimum reduction allowance to fit
        ``gpu_workspace_mb``. ``scratch_bytes`` is capped at 64 MiB and divided from
        the remaining budget to allow concurrent contingency intermediates.
        """
        self.cuda = context.inputs.device.type == "cuda"
        element = context.reference.numeric.element_size()
        pn, pc = context.n_numeric, context.n_categorical
        largest, total = max(rows, default=0), sum(rows)
        data = total * (pn * element + pc * 8 + 1)
        codes = total * (pn + pc) * 8
        preprocessing = largest * (pn * (8 * element + 8) + 128)
        gram = ((largest + 4095) // 4096 + 16) * pn * pn * max(element, 8)
        self.reserved_bytes = data + codes + preprocessing + gram + total * 8 + 32768
        budget = context.gpu_workspace_mb * 1024**2
        if self.cuda and self.reserved_bytes + 4096 > budget:
            raise MemoryError(
                "Trend draws and preprocessing exceed the resolved workspace allowance"
            )
        self.scratch_bytes = (
            min(64 * 1024**2, (budget - self.reserved_bytes) // 3) if self.cuda else 64 * 1024**2
        )


class FidelityExecutor:
    """Execute bounded Shape or Trend draws on prepared CUDA tables."""

    def __init__(self, context: TorchContext) -> None:
        """Retain the prepared execution context; input pool storage remains shared/read-only."""
        self.context = context

    @staticmethod
    def _draws(draws: ArrayLike, n_rows: int) -> IntegerArray:
        """Validate host integer ``[tasks,rows]`` indices against one source pool.

        Unsigned and signed integer matrices are accepted; negative or out-of-range
        positions fail before gathering. This may return an array view and never
        rewrites caller indices. Repeated positions and row ordering are preserved.
        """
        draws = np.asarray(draws)
        if draws.ndim != 2 or draws.dtype.kind not in "iu":
            raise ValueError("draw arrays must be two-dimensional integer arrays")
        if draws.size and (draws.min() < 0 or draws.max() >= n_rows):
            raise ValueError("draw index is outside the input pool")
        return draws

    def batch_size(self, metric: Metric, n: int) -> int:
        """Bound host task batching by draw length and an 8 MiB index allowance.

        Return 1..64 tasks. This routing hint does not reserve metric scratch;
        individual Shape/Trend draws still perform their own byte-budget checks.
        """
        return max(1, min(64, (8 * 1024**2) // max(16 * n, 1)))

    def not_applicable(self, metric: Metric, schema: Schema) -> set[str]:
        """Delegate structural applicability to the metric's schema-only contract.

        No source rows are inspected here; Conditional has a separate data-dependent
        rare-slice check. The returned names are excluded by report construction.
        """
        return metric.not_applicable(None, None, schema)

    @overload
    def _trend_draw(
        self,
        other: TensorTable,
        rd: IntegerArray,
        sd: IntegerArray,
        *,
        detailed: Literal[False] = False,
    ) -> float: ...

    @overload
    def _trend_draw(
        self, other: TensorTable, rd: IntegerArray, sd: IntegerArray, *, detailed: Literal[True]
    ) -> DetailedOutcome: ...

    @overload
    def _trend_draw(
        self, other: TensorTable, rd: IntegerArray, sd: IntegerArray, *, detailed: bool
    ) -> float | DetailedOutcome: ...

    def _trend_draw(
        self, other: TensorTable, rd: IntegerArray, sd: IntegerArray, *, detailed: bool = False
    ) -> float | DetailedOutcome:
        """Gather one owned draw pair and compute its unweighted finite Trend mean.

        Numeric pairs use pairwise-finite correlations. Mixed/categorical pairs use
        reference-fitted quantile bins and exact shared identities; candidate numeric
        values, including infinities, are clipped under the global Trend convention.
        Only owned code buffers are modified. Scratch is budgeted before gathering;
        a draw without computable pair scores yields NaN.
        """
        context = self.context
        width = context.n_numeric + context.n_categorical
        pair_count = width * (width - 1) // 2
        workspace = FidelityWorkspace(context, [len(rd), len(sd)])
        # Fidelity preserves valid cells in rows that have other missing cells.
        a, b = context.reference.gather(rd), other.gather(sd)
        pn, pc = context.n_numeric, context.n_categorical
        prototype = a.numeric.new_empty(())
        parts: list[Tensor] = []
        identities: list[tuple[int, int]] | None = [] if detailed else None
        if pn >= 2:
            ar, bs = correlations(a.numeric), correlations(b.numeric)
            i, j = torch.triu_indices(pn, pn, offset=1, device=a.numeric.device)
            parts.append(1 - 0.5 * (ar[i, j] - bs[i, j]).abs())
            if detailed:
                assert identities is not None
                identities.extend(combinations(range(pn), 2))
        if pc:
            rc = torch.empty((len(rd), pn + pc), dtype=torch.int64, device=a.numeric.device)
            sc = torch.empty((len(sd), pn + pc), dtype=torch.int64, device=b.numeric.device)
            levels: list[int] = []
            for c in range(pn):
                edges = quantile_edges(a.numeric[:, c])
                levels.append(len(edges) - 1 if edges is not None else 0)
                rc[:, c] = bin_codes(a.numeric[:, c], edges)
                sc[:, c] = bin_codes(b.numeric[:, c], edges, clip=True, clip_infinite=True)
            rc[:, pn:], sc[:, pn:] = a.categorical, b.categorical
            levels.extend(context.inputs.cardinalities)
            contingency = contingency_scores(
                rc, sc, levels, pn, prototype, workspace.scratch_bytes, return_pairs=detailed
            )
            if detailed:
                scores, pair_ids = cast(tuple[Tensor, list[tuple[int, int]]], contingency)
                assert identities is not None
                parts.append(scores)
                identities.extend(pair_ids)
            else:
                parts.append(cast(Tensor, contingency))
        scores = torch.cat(parts) if parts else prototype.new_empty(0)
        value = scores[torch.isfinite(scores)].mean()
        scalar = float(value.item())
        if not detailed:
            return scalar
        # Preserve the scalar's bucket reduction order. Canonical ordering is
        # applied only to the descriptive host table after the one transfer.
        host_scores = host_array(scores)
        first, second = np.triu_indices(width, 1)
        ordered = np.full(pair_count, np.nan, dtype=host_scores.dtype)
        if identities:
            pairs = np.asarray(identities, dtype=np.int64)
            locations = pairs[:, 0] * (2 * width - pairs[:, 0] - 1) // 2
            locations += pairs[:, 1] - pairs[:, 0] - 1
            ordered[locations] = host_scores
        table = DetailTable(dict(column_1_index=first, column_2_index=second, score=ordered))
        return DetailedOutcome({"trend": scalar}, FidelityDetails({"pairs": table}))

    @overload
    def _shape_draw(
        self,
        other: TensorTable,
        rd: IntegerArray,
        sd: IntegerArray,
        *,
        detailed: Literal[False] = False,
    ) -> float: ...

    @overload
    def _shape_draw(
        self, other: TensorTable, rd: IntegerArray, sd: IntegerArray, *, detailed: Literal[True]
    ) -> DetailedOutcome: ...

    @overload
    def _shape_draw(
        self, other: TensorTable, rd: IntegerArray, sd: IntegerArray, *, detailed: bool
    ) -> float | DetailedOutcome: ...

    def _shape_draw(
        self, other: TensorTable, rd: IntegerArray, sd: IntegerArray, *, detailed: bool = False
    ) -> float | DetailedOutcome:
        """Gather one pair of read-only source draws and average marginal similarities.

        Reserve Shape scratch first; each numeric/categorical column uses its own
        valid observations. Every observed categorical identity participates;
        reduction includes finite column scores only.
        """
        from .shape import ShapeWorkspace, shape_scores

        context = self.context
        width = context.n_numeric + context.n_categorical
        workspace = ShapeWorkspace(context, (len(rd), len(sd)))
        a, b = context.reference.gather(rd), other.gather(sd)
        scores = shape_scores(a, b, context.inputs.cardinalities, workspace)
        value = scores[torch.isfinite(scores)].mean()
        scalar = float(value.item())
        if not detailed:
            return scalar
        table = DetailTable(
            dict(column_index=np.arange(width, dtype=np.int64), score=host_array(scores))
        )
        return DetailedOutcome({"shape": scalar}, FidelityDetails({"columns": table}))

    @staticmethod
    def _point_draw(draw: ArrayLike, n_rows: int) -> IntegerArray:
        """Validate one point comparison's one-dimensional positional indices."""
        draw = np.asarray(draw)
        if draw.ndim != 1:
            raise ValueError("point draws must be one-dimensional integer arrays")
        return FidelityExecutor._draws(draw[None, :], n_rows)[0]

    @torch.no_grad()
    def compute_detailed(
        self,
        metric: Metric,
        reference_draw: ArrayLike,
        candidate_draw: ArrayLike,
        *,
        ideal: bool = False,
    ) -> DetailedOutcome:
        """Return aggregate and column/pair scores from one numerical point pass.

        Index vectors address the prepared pools; ideal mode uses the reference
        pool for both sides. Existing device scores are copied in bounded
        transfers after their scalar reduction; no numerical work is repeated.
        """
        if metric.name not in {"shape", "trend"}:
            raise ValueError("tensor fidelity execution requires Shape or Trend")
        context = self.context
        other = context.reference if ideal else context.candidate
        rd = self._point_draw(reference_draw, context.reference.n_rows)
        sd = self._point_draw(candidate_draw, other.n_rows)
        compute_draw = self._shape_draw if metric.name == "shape" else self._trend_draw
        with torch.autocast(context.inputs.device.type, enabled=False):
            return compute_draw(other, rd, sd, detailed=True)

    @torch.no_grad()
    def compute(
        self,
        metric: Metric,
        reference_draws: ArrayLike,
        candidate_draws: ArrayLike,
        *,
        ideal: bool = False,
    ) -> dict[str, FloatArray]:
        """Return one host statistic per ordered source-row draw.

        Parameters
        ----------
        metric : Shape or Trend
            Built-in fidelity metric with an implemented CUDA kernel.
        reference_draws, candidate_draws : numpy.ndarray
            Integer matrices of shape ``(tasks, rows)``. Draw lengths may
            differ between pools. Repeated positions preserve multiplicity.
        ideal : bool, default False
            Compare both draws against the reference source pool.

        Returns
        -------
        dict of str to numpy.ndarray
            Statistic name mapped to a float64 host array of task results;
            computations use the context's selected FP32/FP64 precision.
        """
        if metric.name not in {"shape", "trend"}:
            raise ValueError("tensor fidelity execution requires Shape or Trend")
        context = self.context
        other = context.reference if ideal else context.candidate
        rd = self._draws(reference_draws, context.reference.n_rows)
        sd = self._draws(candidate_draws, other.n_rows)
        if len(rd) != len(sd):
            raise ValueError("draw arrays must have equal batch lengths")
        values = np.empty(len(rd))
        compute_draw = self._shape_draw if metric.name == "shape" else self._trend_draw
        with torch.autocast(context.inputs.device.type, enabled=False):
            for i, (r, s) in enumerate(zip(rd, sd, strict=True)):
                values[i] = compute_draw(other, r, s)
        return {metric.name: values}
