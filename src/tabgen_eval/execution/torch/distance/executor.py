"""PRDC and training-referenced memorization over original-source draws."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral
from typing import TYPE_CHECKING

import numpy as np
import torch
from numpy.typing import ArrayLike

from tabgen_eval._typing import FloatArray, IntegerArray
from tabgen_eval.execution.runtime.components import PRDC_COMPONENTS, selected_components
from tabgen_eval.execution.runtime.outcomes import MemorizationOutcome
from tabgen_eval.execution.torch.distance.categories import CategoryDomain
from tabgen_eval.execution.torch.distance.geometry import (
    DistanceWorkspace,
    ReferenceGeometry,
    finite,
    prdc,
)

if TYPE_CHECKING:
    from tabgen_eval.execution.torch.context import TorchContext
    from tabgen_eval.execution.torch.tensors import TensorTable
    from tabgen_eval.metrics import Metric
    from tabgen_eval.schema import Schema


class DistanceExecutor:
    """Execute GPU PRDC/DCR draws with shared read-only source pools.

    Parameters
    ----------
    context : TorchContext
        Owns selected-device inputs, precision and temporary workspace cap.

    Notes
    -----
    Tasks run sequentially on the caller's CUDA stream. Draw-local gathers preserve
    indices/duplicates and are released between comparisons.
    """

    def __init__(self, context: TorchContext) -> None:
        """Retain prepared source pools and cache identity-domain metadata without fitting draws."""
        self.context = context
        self.category_domain = CategoryDomain(context.inputs.cardinalities, context.inputs.device)

    @staticmethod
    def _draws(draws: ArrayLike, n_rows: int) -> IntegerArray:
        """Validate integer ``[B, N]`` source positions without copying or reordering.

        Return an array view when possible. Bounds include every position, even rows
        that later fail complete-row filtering. Invalid layouts or bounds raise
        ``ValueError``; this helper never mutates the caller's draw arrays.
        """
        draws = np.asarray(draws)
        if draws.ndim != 2 or draws.dtype.kind not in "iu":
            raise ValueError("draw arrays must be two-dimensional integer arrays")
        if draws.size and (draws.min() < 0 or draws.max() >= n_rows):
            raise ValueError("draw index is outside the input pool")
        return draws

    def _samples(
        self, pools: Sequence[TensorTable], draws: Sequence[IntegerArray], *, k: int | None = None
    ) -> tuple[list[TensorTable], DistanceWorkspace, list[int]]:
        # Filtering needs masks and nonzero/selection scratch in addition to the
        # retained indices. Guard this phase before any device index upload.
        """Filter incomplete source rows and gather owned draw-local tensors.

        ``pools`` and one-dimensional ``draws`` are aligned sequences. Repeated indices
        survive filtering. Guard index/filter scratch before device upload, then reserve
        all live gathered data and distance scratch before gathering. Return samples,
        their shared workspace, and retained row counts. A CUDA budget violation raises
        ``MemoryError`` before the corresponding allocation phase.
        """
        index_bytes = sum(len(d) for d in draws) * 8 + max(map(len, draws), default=0) * 32 + 16384
        if (
            self.context.inputs.device.type == "cuda"
            and index_bytes > self.context.gpu_workspace_mb * 1024**2
        ):
            raise MemoryError("draw index filtering exceeds the resolved workspace allowance")
        indices = []
        for pool, draw in zip(pools, draws, strict=True):
            positions = pool.row_indices(draw)
            indices.append(positions[pool.complete[positions]])
            del positions
        sizes = [len(i) for i in indices]
        workspace = DistanceWorkspace(
            self.context,
            sizes,
            [len(d) for d in draws],
            k=k,
            category_metadata_bytes=self.category_domain.metadata_bytes,
        )
        samples = [pool._gather_indices(i) for pool, i in zip(pools, indices, strict=True)]
        return samples, workspace, sizes

    def not_applicable(self, metric: Metric, schema: Schema) -> set[str]:
        """Delegate schema-only applicability to the distance metric."""
        return metric.not_applicable(None, None, schema)

    def batch_size(self, metric: Metric, n: int) -> int:
        # Only one device task is live at a time. Bounded CPU index batches do not
        # imply B simultaneous device draws; each draw has its own workspace guard.
        """Bound host index batch storage while device comparisons remain sequential.

        Use two source pools with an 8 MiB index target
        and at most 64 draws. This is not a simultaneous-device-draw allocation budget;
        each task independently checks ``gpu_workspace_mb``.
        """
        pools = 2
        return max(1, min(64, (8 * 1024**2) // max(pools * n * 8, 1)))

    def _prdc_draw(
        self, metric: Metric, other: TensorTable, draws: Sequence[IntegerArray], k: int
    ) -> dict[str, float]:
        # Draw-local scope releases all gathered/encoded buffers before the next
        # draw starts, including when the estimator submits a multi-draw batch.
        """Evaluate one PRDC draw in its own buffer lifetime.

        Fit geometry on the complete reference sample only. Required radius sources
        need more than k complete rows; the other source need only be nonempty.
        Nonfinite transformed coordinates produce NaN. Sources remain read-only.
        """
        samples, workspace, sizes = self._samples((self.context.reference, other), draws, k=k)
        result = dict.fromkeys(metric.stat_names, float("nan"))
        components = selected_components(metric.requested_components, PRDC_COMPONENTS)
        usable = (
            min(sizes) > 0
            and (not any(name != "recall" for name in components) or sizes[0] > k)
            and ("recall" not in components or sizes[1] > k)
        )
        if usable and components:
            geometry = ReferenceGeometry(
                samples[0], domain=self.category_domain, workspace=workspace
            )
            a = geometry.transform(samples[0], reference=True)
            b = geometry.transform(samples[1])
            if finite(a) and finite(b):
                result = prdc(a, b, k, workspace, requested_components=components)
        return result

    @torch.no_grad()
    def compute(
        self,
        metric: Metric,
        reference_draws: ArrayLike,
        candidate_draws: ArrayLike,
        *,
        ideal: bool = False,
    ) -> dict[str, FloatArray]:
        """Compute PRDC for a batch of matched source-index draws.

        Parameters
        ----------
        metric : PRDC
            Metric parameters including neighbor rank.
        reference_draws, candidate_draws : array-like
            Integer ``[B, N]`` row positions; invalid indices raise before gathering.
        ideal : bool
            Use the reference source for both draw pools when true.

        Returns
        -------
        dict of numpy.ndarray
            Four statistic arrays of length B, preserving batch order.

        Notes
        -----
        Sources are read-only; the caller controls sampling and the ideal protocol.
        """
        if metric.name != "prdc":
            raise ValueError("sample distance execution requires PRDC")
        k = metric.nearest_k
        if isinstance(k, bool) or not isinstance(k, Integral) or k < 1:
            raise ValueError("nearest_k must be a positive integer")
        k = int(k)
        context = self.context
        other = context.reference if ideal else context.candidate
        rd = self._draws(reference_draws, context.reference.n_rows)
        sd = self._draws(candidate_draws, other.n_rows)
        if len(rd) != len(sd):
            raise ValueError("draw arrays must have equal batch lengths")
        values = {name: np.empty(len(rd)) for name in metric.stat_names}
        for i, draws in enumerate(zip(rd, sd, strict=True)):
            result = self._prdc_draw(metric, other, draws, k)
            for name in values:
                values[name][i] = result[name]
        return values

    @torch.no_grad()
    def compute_memorization(
        self, metric: Metric, reference_draws: ArrayLike, candidate_draws: ArrayLike
    ) -> list[MemorizationOutcome]:
        """Compute only one requested memorization metric per supplied draw."""
        return [
            result[metric.name]
            for result in self.compute_memorization_group(
                [metric], reference_draws, candidate_draws
            )
        ]

    def _memorization_draw(
        self, metrics: Sequence[Metric], pools: Sequence[TensorTable], draws: Sequence[IntegerArray]
    ) -> dict[str, MemorizationOutcome]:
        """Release one draw's features, search indices and minima before the next gather."""
        from tabgen_eval.execution.runtime.distance_outcomes import memorization_metric_outcome
        from tabgen_eval.execution.torch.distance.planner import MemorizationPlan

        samples, workspace, sizes = self._samples(pools, draws)
        if not sizes[0]:
            return {
                metric.name: memorization_metric_outcome(
                    metric,
                    counts=sizes[:2],
                    holdout_count=sizes[2] if len(sizes) > 2 else None,
                )
                for metric in metrics
            }
        geometry = ReferenceGeometry(samples[0], domain=self.category_domain, workspace=workspace)
        train = geometry.transform(samples[0], reference=True)
        gen = geometry.transform(samples[1])
        holdout = geometry.transform(samples[2]) if len(samples) > 2 else None
        plan = MemorizationPlan(
            train,
            gen,
            workspace,
            metrics,
            threads=self.context.workers,
            holdout=holdout,
        )
        # Only the plan retains H's gathered feature buffers during its search phases.
        samples.clear()
        del geometry, holdout
        return plan.execute()

    @torch.no_grad()
    def compute_memorization_group(
        self,
        metrics: Sequence[Metric],
        reference_draws: ArrayLike,
        candidate_draws: ArrayLike,
        *,
        holdout_draws: ArrayLike | None = None,
    ) -> list[dict[str, MemorizationOutcome]]:
        """Fit one geometry per draw and share only the requested distance work."""
        if not metrics or any(
            metric.name not in {"dcr", "near_match_rate", "replication_ratio"} for metric in metrics
        ):
            raise ValueError("unsupported memorization metric group")
        context = self.context
        rd = self._draws(reference_draws, context.reference.n_rows)
        gd = self._draws(candidate_draws, context.candidate.n_rows)
        if len(rd) != len(gd):
            raise ValueError("draw arrays must have equal batch lengths")
        pools = (context.reference, context.candidate)
        all_draws = (rd, gd)
        if holdout_draws is not None:
            if context.holdout is None:
                raise ValueError("holdout draws require a prepared holdout pool")
            hd = self._draws(holdout_draws, context.holdout.n_rows)
            if len(hd) != len(rd):
                raise ValueError("draw arrays must have equal batch lengths")
            if any(metric.name == "dcr" for metric in metrics):
                pools += (context.holdout,)
                all_draws += (hd,)
        outcomes = []
        for draws in zip(*all_draws, strict=True):
            outcomes.append(self._memorization_draw(metrics, pools, draws))
        self.memorization_execution = {}
        for metric in metrics:
            diagnostics = [outcome[metric.name].diagnostics for outcome in outcomes]
            methods = sorted(
                {
                    item.get("search_method", item.get("method", "unavailable"))
                    for item in diagnostics
                }
            )
            self.memorization_execution[metric.name] = {
                "search_methods": methods,
            }
        return outcomes
