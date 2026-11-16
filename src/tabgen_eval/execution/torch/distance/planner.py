"""Shared exact forward and reverse searches in training-fitted geometry."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from tabgen_eval.execution.runtime.distance_outcomes import (
    NEAR_MATCH_THRESHOLD,
    memorization_metric_outcome,
)
from tabgen_eval.execution.runtime.outcomes import DCR_PERCENTILES, MemorizationOutcome

from .geometry import (
    DistanceWorkspace,
    Features,
    _median_ordered,
    _near_match_rate,
    finite,
    nearest_distances,
)

if TYPE_CHECKING:
    from tabgen_eval.metrics.memorization import _MemorizationMetric


class MemorizationPlan:
    """One draw's shared exact searches over training-fitted feature buffers."""

    def __init__(
        self,
        reference: Features,
        generated: Features,
        workspace: DistanceWorkspace,
        metrics: Sequence[_MemorizationMetric],
        *,
        threads: int = 1,
        holdout: Features | None = None,
    ) -> None:
        self.reference, self.generated, self.holdout = reference, generated, holdout
        self.workspace, self.metrics, self.threads = workspace, metrics, threads
        self.searches = {"forward": 0, "holdout_to_training": 0, "reverse": 0}

    @property
    def available_bytes(self) -> int:
        """Bounded CPU scratch allowance or remaining CUDA draw workspace."""
        if self.reference.numeric.device.type == "cpu":
            return 64 * 1024**2
        return max(0, self.workspace.budget_bytes - self.workspace.reserved_bytes)

    def _nearest(self, query: Features, *, reference: Features | None = None) -> torch.Tensor:
        reference = self.reference if reference is None else reference
        if self.reference.numeric.device.type == "cpu":
            from .cpu_nearest import nearest_distances_cpu

            return nearest_distances_cpu(
                reference,
                query,
                threads=self.threads,
                scratch_bytes=self.available_bytes,
            )
        return nearest_distances(reference, query, self.workspace)

    def execute(self) -> dict[str, MemorizationOutcome]:
        """Reduce requested exact query passes using one training-fitted geometry."""
        median = q05 = baseline = float("nan")
        generated_quantiles: dict[int, float] = {}
        holdout_quantiles: dict[int, float] = {}
        forward: torch.Tensor | None = None
        reverse: torch.Tensor | None = None
        wants_replication = any(metric.name == "replication_ratio" for metric in self.metrics)
        wants_forward = any(metric.name in {"dcr", "near_match_rate"} for metric in self.metrics)
        wants_dcr = any(metric.name == "dcr" for metric in self.metrics)

        def quantiles(ordered: torch.Tensor) -> dict[int, float]:
            reductions = []
            for percentile in DCR_PERCENTILES:
                position = (len(ordered) - 1) * (percentile / 100)
                lower = int(position)
                upper = min(lower + 1, len(ordered) - 1)
                reductions.append(
                    _median_ordered(ordered)
                    if percentile == 50
                    else ordered[lower].lerp(ordered[upper], position - lower)
                )
            return dict(zip(DCR_PERCENTILES, torch.stack(reductions).tolist(), strict=True))

        if finite(self.reference):
            if self.generated.n_rows and finite(self.generated):
                if wants_forward:
                    self.searches["forward"] += 1
                    forward = self._nearest(self.generated)
                if wants_replication:
                    self.searches["reverse"] += 1
                    reverse = self._nearest(self.reference, reference=self.generated)
            if wants_dcr and forward is not None:
                ordered = forward.sort().values
                generated_quantiles = quantiles(ordered)
                median = generated_quantiles[50]
                q05 = generated_quantiles[5]
            if (
                wants_dcr
                and self.holdout is not None
                and self.holdout.n_rows
                and finite(self.holdout)
            ):
                self.searches["holdout_to_training"] += 1
                ordered = self._nearest(self.holdout).sort().values
                holdout_quantiles = quantiles(ordered)
                baseline = holdout_quantiles[50]

        def rate(metric: _MemorizationMetric) -> float:
            distances = forward if metric.name == "near_match_rate" else reverse
            if (
                metric.name == "replication_ratio"
                and self.reference.n_rows
                and finite(self.reference)
                and not self.generated.n_rows
            ):
                return 0.0
            if distances is None or torch.isnan(distances).any():
                return float("nan")
            return float(_near_match_rate(distances, NEAR_MATCH_THRESHOLD).item())

        return {
            metric.name: memorization_metric_outcome(
                metric,
                counts=(self.reference.n_rows, self.generated.n_rows),
                median=median,
                q05=q05,
                holdout_median=baseline,
                generated_quantiles=generated_quantiles,
                holdout_quantiles=holdout_quantiles,
                holdout_count=None if self.holdout is None else self.holdout.n_rows,
                rate=rate(metric) if metric.name != "dcr" else float("nan"),
                diagnostics={"search_method": "exact", "searches": dict(self.searches)},
            )
            for metric in self.metrics
        }
