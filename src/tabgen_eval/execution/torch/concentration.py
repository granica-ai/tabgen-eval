"""Exact concentration execution with one bounded shared-reference profile cache."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeAlias, overload

import numpy as np
import torch
from numpy.typing import ArrayLike
from torch import Tensor

from tabgen_eval._typing import FloatArray, IntegerArray
from tabgen_eval.execution.python.concentration import (
    fitted_radii,
    outcome,
    pair_indices,
    squared_thresholds,
)
from tabgen_eval.execution.runtime.details import DetailedOutcome
from tabgen_eval.execution.torch.distance.executor import DistanceExecutor

_ReferenceProfile: TypeAlias = tuple[
    tuple[object, ...], Tensor, Tensor, FloatArray, FloatArray, int
]

if TYPE_CHECKING:
    from tabgen_eval.execution.torch.context import TorchContext
    from tabgen_eval.execution.torch.tensors import TensorTable
    from tabgen_eval.metrics import Concentration
    from tabgen_eval.schema import Schema


def _summarize(counts: Tensor, upper_quantile: float) -> FloatArray:
    """Reduce integer counts on-device and copy only the compact summary to host."""
    n = counts.shape[1]
    means = counts.sum(dim=1, dtype=torch.int64).to(torch.float64) / (n * (n - 1))
    # Quantiles require floating storage. Counts remain exact through conversion
    # to FP64, including CUDA's int32 counters and CPU's int64 counters.
    upper = torch.quantile(counts.to(torch.float64), upper_quantile, dim=1) / (n - 1)
    return torch.stack((means, upper), dim=1).cpu().numpy()


class ConcentrationExecutor:
    """Run complete-case comparisons with shared real-defined radii and geometry.

    Source tensors are read-only. One last reference cache retains only its draw
    key, numeric scale, radii and compact summaries; no count matrix or sampled
    feature pool survives a comparison. Thus paired generated/real and real/real
    calls reuse the expensive reference profile while memory stays bounded.
    """

    def __init__(self, context: TorchContext) -> None:
        self.context = context
        self._reference_cache: _ReferenceProfile | None = None

    @staticmethod
    def batch_size(metric: Concentration, n: int) -> int:
        """Keep each paired reference adjacent so its compact profile is reused."""
        return 1

    @staticmethod
    def not_applicable(metric: Concentration, schema: Schema) -> set[str]:
        """Concentration is defined for any feature roles with two usable records."""
        return metric.not_applicable(None, None, schema)

    def _reserve(self, metric: Concentration, n_reference: int, n_candidate: int) -> None:
        """Check gathers, count/quantile buffers, bounded pair samples and headroom."""
        context = self.context
        if context.inputs.device.type != "cuda":
            return
        element = context.reference.numeric.element_size()
        rows = n_reference + n_candidate
        radii = len(metric.fixed_radii) + len(metric.radius_quantiles)
        features = rows * (context.n_numeric * element + context.n_categorical * 8)
        # Counts plus FP64 conversion, quantile sorting/copies and allocator room.
        summaries = max(n_reference, n_candidate) * radii * 40
        preprocessing = rows * (context.n_numeric + 32)
        pairs = metric.radius_pairs * 40 if metric.radius_quantiles else 0
        live = features + summaries + preprocessing + pairs + 65536
        if live > context.gpu_workspace_mb * 1024**2:
            raise MemoryError(
                "Concentration draws and quantile workspace exceed the resolved workspace allowance"
            )

    @staticmethod
    def _key(metric: Concentration, draw: IntegerArray) -> tuple[object, ...]:
        return (
            draw.tobytes(),
            metric.fixed_radii,
            metric.radius_quantiles,
            metric.upper_quantile,
            metric.radius_pairs,
            metric.radius_seed,
        )

    def _sample(self, pool: TensorTable, draw: ArrayLike) -> TensorTable:
        positions = pool.row_indices(draw)
        positions = positions[pool.complete[positions]]
        return pool._gather_indices(positions)

    def _counts(self, numeric: Tensor, categorical: Tensor, radii: FloatArray) -> Tensor:
        thresholds = torch.as_tensor(
            squared_thresholds(radii, np.dtype(self.context.inputs.dtype)),
            dtype=numeric.dtype,
            device=numeric.device,
        )
        if numeric.device.type == "cuda":
            from tabgen_eval.execution.torch.cuda.triton.concentration import local_counts

            return local_counts(numeric, categorical, thresholds)
        from tabgen_eval.execution.torch.cpu.extension import library

        return library().local_concentration_counts(
            numeric,
            categorical,
            thresholds,
            self.context.workers,
        )

    def _radii(self, metric: Concentration, numeric: Tensor, categorical: Tensor) -> FloatArray:
        if not metric.radius_quantiles:
            return np.asarray(metric.fixed_radii, dtype=np.float64)
        left, right = pair_indices(len(numeric), metric.radius_pairs, metric.radius_seed)
        if numeric.device.type == "cuda":
            from tabgen_eval.execution.torch.cuda.triton.concentration import paired_distances

            left_device = torch.as_tensor(left, device=numeric.device)
            right_device = torch.as_tensor(right, device=numeric.device)
            distances = paired_distances(numeric, categorical, left_device, right_device)
            distances = distances.cpu().numpy()
        else:
            from tabgen_eval.execution.torch.cpu.extension import library

            distances = (
                library()
                .paired_concentration_distances(
                    numeric,
                    categorical,
                    torch.from_numpy(left),
                    torch.from_numpy(right),
                    self.context.workers,
                )
                .numpy()
            )
        return fitted_radii(metric, distances)

    @overload
    def _draw(
        self,
        metric: Concentration,
        reference_draw: IntegerArray,
        candidate_draw: IntegerArray,
        *,
        ideal: bool = False,
        details: Literal[False] = False,
    ) -> dict[str, float]: ...

    @overload
    def _draw(
        self,
        metric: Concentration,
        reference_draw: IntegerArray,
        candidate_draw: IntegerArray,
        *,
        ideal: bool = False,
        details: Literal[True],
    ) -> DetailedOutcome: ...

    @overload
    def _draw(
        self,
        metric: Concentration,
        reference_draw: IntegerArray,
        candidate_draw: IntegerArray,
        *,
        ideal: bool = False,
        details: bool,
    ) -> dict[str, float] | DetailedOutcome: ...

    def _draw(
        self,
        metric: Concentration,
        reference_draw: IntegerArray,
        candidate_draw: IntegerArray,
        *,
        ideal: bool = False,
        details: bool = False,
    ) -> dict[str, float] | DetailedOutcome:
        context = self.context
        pool = context.reference if ideal else context.candidate
        self._reserve(metric, len(reference_draw), len(candidate_draw))
        key = self._key(metric, reference_draw)
        cache = self._reference_cache
        other = self._sample(pool, candidate_draw)
        candidate_n = other.n_rows
        n_radii = len(metric.fixed_radii) + len(metric.radius_quantiles)
        empty = np.full((n_radii, 2), np.nan)
        if not (context.n_numeric or context.n_categorical):
            n = self._sample(context.reference, reference_draw).n_rows
            return outcome(
                metric, np.full(n_radii, np.nan), empty, empty, (n, candidate_n), details=details
            )
        new_reference = cache is None or cache[0] != key
        if candidate_n < 2:
            if not new_reference:
                assert cache is not None
                return outcome(
                    metric, cache[3], cache[4], empty, (cache[5], candidate_n), details=details
                )
            n = self._sample(context.reference, reference_draw).n_rows
            radii = np.asarray([*metric.fixed_radii, *([np.nan] * len(metric.radius_quantiles))])
            return outcome(metric, radii, empty, empty, (n, candidate_n), details=details)
        if new_reference:
            reference = self._sample(context.reference, reference_draw)
            n = reference.n_rows
            radii = np.asarray([*metric.fixed_radii, *([np.nan] * len(metric.radius_quantiles))])
            if n < 2:
                return outcome(metric, radii, empty, empty, (n, candidate_n), details=details)
            low = reference.numeric.amin(dim=0)
            scale = reference.numeric.amax(dim=0) - low
            scale.masked_fill_(scale == 0, 1)
            reference.numeric.sub_(low).div_(scale)
            other.numeric.sub_(low).div_(scale)
            if not bool(
                torch.isfinite(reference.numeric).all() & torch.isfinite(other.numeric).all()
            ):
                return outcome(metric, radii, empty, empty, (n, candidate_n), details=details)
            radii = self._radii(metric, reference.numeric, reference.categorical)
            profile = _summarize(
                self._counts(reference.numeric, reference.categorical, radii),
                metric.upper_quantile,
            )
            self._reference_cache = cache = (key, low, scale, radii, profile, n)
            del reference
        assert cache is not None
        _, low, scale, radii, profile, n = cache
        if not new_reference:
            other.numeric.sub_(low).div_(scale)
        candidate = empty
        if new_reference or bool(torch.isfinite(other.numeric).all()):
            candidate = _summarize(
                self._counts(other.numeric, other.categorical, radii),
                metric.upper_quantile,
            )
        return outcome(metric, radii, profile, candidate, (n, candidate_n), details=details)

    def compute_detailed(
        self,
        metric: Concentration,
        reference_draw: ArrayLike,
        candidate_draw: ArrayLike,
        *,
        ideal: bool = False,
    ) -> DetailedOutcome:
        """Capture raw point profiles from the same kernels that produce scalar ratios."""
        with torch.inference_mode(), torch.autocast(self.context.inputs.device.type, enabled=False):
            return self._draw(
                metric,
                np.asarray(reference_draw),
                np.asarray(candidate_draw),
                ideal=ideal,
                details=True,
            )

    def compute(
        self,
        metric: Concentration,
        reference_draws: ArrayLike,
        candidate_draws: ArrayLike,
        *,
        ideal: bool = False,
    ) -> dict[str, FloatArray]:
        """Return one scalar array per statistic for validated original-source draws."""
        reference_draws = DistanceExecutor._draws(reference_draws, self.context.reference.n_rows)
        pool = self.context.reference if ideal else self.context.candidate
        candidate_draws = DistanceExecutor._draws(candidate_draws, pool.n_rows)
        if len(reference_draws) != len(candidate_draws):
            raise ValueError("Reference and candidate draw batches must have equal length")
        result: dict[str, FloatArray] = {
            name: np.empty(len(reference_draws)) for name in metric.stat_names
        }
        with torch.inference_mode(), torch.autocast(self.context.inputs.device.type, enabled=False):
            for index, (rd, sd) in enumerate(zip(reference_draws, candidate_draws, strict=True)):
                values = self._draw(metric, rd, sd, ideal=ideal)
                for name in result:
                    result[name][index] = values[name]
        return result
