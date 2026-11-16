"""Adapt estimator-owned CPU draws to compiled ATen metric operators.

Persistent input tensors belong to the execution context and remain read-only.
This adapter validates and copies host draw indices, serializes metric options,
and decodes results; native operators own gathering, numerical work, and worker
budgets. It does not sample bootstrap draws or choose scoring partitions.
"""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral
from typing import TYPE_CHECKING

import numpy as np
import torch
from numpy.typing import ArrayLike
from torch import Tensor

from tabgen_eval._typing import FloatArray, SeedScalar
from tabgen_eval.execution.runtime.components import component_mask
from tabgen_eval.execution.runtime.context import cpu_batch_size
from tabgen_eval.execution.runtime.details import DetailedOutcome
from tabgen_eval.execution.runtime.outcomes import ClassifierOutcome
from tabgen_eval.execution.torch.cpu.diagnostics import classifier_fold
from tabgen_eval.execution.torch.cpu.extension import library
from tabgen_eval.execution.torch.distance.executor import DistanceExecutor
from tabgen_eval.preprocessing.reporting import fold_metadata

if TYPE_CHECKING:
    from tabgen_eval.execution.torch.context import TorchContext
    from tabgen_eval.execution.torch.tensors import TensorTable
    from tabgen_eval.metrics import LinearC2ST, Metric
    from tabgen_eval.schema import Schema


class CppExecutor:
    """CPU execution adapter sharing one prepared context and loaded operator namespace.

    Construction loads or builds the extension through ``library()``. Draw operators
    receive contiguous row-major CPU numerics/categories plus complete-row masks;
    output order follows the caller's batch and metric statistic order. Autocast is
    disabled around numerical calls to preserve the context's selected precision.
    """

    def __init__(self, context: TorchContext) -> None:
        """Retain the prepared context and load the native CPU operator namespace."""
        self.context = context
        self.ops = library()

    def batch_size(self, metric: Metric, n: int) -> int:
        """Bound host draw batches using metric width and encoded classifier storage."""
        c = self.context
        return cpu_batch_size(
            metric,
            n,
            n_numeric=c.n_numeric,
            n_categorical=c.n_categorical,
            classifier_width=c.classifier_width,
        )

    @staticmethod
    def _parameters(metric: Metric) -> tuple[int, int, int, float]:
        """Validate the common native sample-metric option packet.

        Return ``(nearest_k, min_count, num_bins, rare_threshold)`` with positive integral
        counts (excluding booleans) and a finite probability threshold. Defaults also
        fill fields unused by the selected operator; invalid values raise ``ValueError``.
        """
        values = []
        for name, default in [("nearest_k", 5), ("min_count", 10), ("num_bins", 10)]:
            value = getattr(metric, name, default)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
            values.append(int(value))
        rare = getattr(metric, "rare_threshold", 0.05)
        if not np.isfinite(rare) or not 0 <= rare <= 1:
            raise ValueError("rare_threshold must be finite and in [0, 1]")
        return *values, float(rare)

    @staticmethod
    def _draws(draws: ArrayLike, table: TensorTable) -> Tensor:
        """Copy valid integer ``[B, N]`` source indices to owned contiguous int64 CPU storage.

        Bounds are checked against ``table`` before conversion. Draw order and repeated
        indices are preserved, and caller arrays are never modified or retained as views.
        """
        values = DistanceExecutor._draws(draws, table.n_rows)
        return torch.from_numpy(np.array(values, dtype=np.int64, order="C", copy=True))

    def _pair(
        self, reference_draws: ArrayLike, candidate_draws: ArrayLike, *, ideal: bool = False
    ) -> tuple[TensorTable, Tensor, Tensor]:
        """Resolve the candidate/ideal pool and validate equal draw batch lengths.

        In ideal mode both draws address the reference pool. Return the selected pool
        and two owned CPU index tensors; draw widths may differ between source pools.
        """
        c = self.context
        other = c.reference if ideal else c.candidate
        rd = self._draws(reference_draws, c.reference)
        sd = self._draws(candidate_draws, other)
        if len(rd) != len(sd):
            raise ValueError("draw arrays must have equal batch lengths")
        return other, rd, sd

    def _arguments(self, other: TensorTable, rd: Tensor, sd: Tensor) -> tuple[Tensor, ...]:
        """Pack read-only source tensors, complete masks, and draws in native ABI order."""
        r = self.context.reference
        return (
            r.numeric,
            r.categorical,
            other.numeric,
            other.categorical,
            r.complete,
            other.complete,
            rd,
            sd,
        )

    def not_applicable(self, metric: Metric, schema: Schema) -> set[str]:
        """Identify structurally unavailable statistics using reference data only.

        Conditional rarity is queried in native code with the metric's grouping options;
        other metrics supply their own schema rules. This does not fit candidate data or
        compute a metric draw.
        """
        if metric.name != "conditional_fidelity":
            return metric.not_applicable(None, None, schema)
        c = self.context
        width = c.n_numeric + c.n_categorical
        if not width:
            return set(metric.stat_names)
        missing = {"cond_trend_all", "cond_trend_rare"} if width < 3 else set()
        if width < 2:
            missing |= {"cond_shape_all", "cond_shape_rare"}
        _, minimum, bins, rare = self._parameters(metric)
        if not self.ops.has_rare(c.reference.numeric, c.reference.categorical, minimum, bins, rare):
            missing |= {"cond_shape_rare", "cond_trend_rare", "recovery_rare"}
        return missing

    @torch.no_grad()
    def compute(
        self,
        metric: Metric,
        reference_draws: ArrayLike,
        candidate_draws: ArrayLike,
        *,
        ideal: bool = False,
    ) -> dict[str, FloatArray]:
        """Evaluate Shape, Trend, Conditional, or PRDC for matched source draws.

        Draws are integer matrices ``[B, NR]`` and ``[B, NS]``. ``ideal=True`` selects
        the reference pool for the second draw. Return one NumPy array of length B per
        statistic, preserving draw order. Unsupported metrics or malformed draws raise
        ``ValueError`` before numerical dispatch; persistent input pools remain unchanged.
        """
        if metric.name not in {"shape", "trend", "conditional_fidelity", "prdc"}:
            raise ValueError(f"unsupported compiled CPU sample metric: {metric.name}")
        other, rd, sd = self._pair(reference_draws, candidate_draws, ideal=ideal)
        c = self.context
        with torch.autocast("cpu", enabled=False):
            result = self.ops.samples(
                *self._arguments(other, rd, sd),
                list(c.inputs.cardinalities),
                metric.name,
                *self._parameters(metric),
                c.workers,
                component_mask(metric),
            ).numpy()
        return {name: result[:, i] for i, name in enumerate(metric.stat_names)}

    @torch.no_grad()
    def compute_detailed(
        self,
        metric: Metric,
        reference_draw: ArrayLike,
        candidate_draw: ArrayLike,
        *,
        ideal: bool = False,
    ) -> DetailedOutcome:
        """Collect one fixed fidelity comparison's descriptive tables during execution.

        Draws are one-dimensional source positions. The native operator returns
        the same aggregate reduction and typed column/pair/slice observations;
        this adapter only decodes owned host arrays and does not recompute scores.
        """
        from tabgen_eval.execution.runtime.details import (
            DetailedOutcome,
            DetailTable,
            FidelityDetails,
        )

        if metric.name not in {"shape", "trend", "conditional_fidelity"}:
            raise ValueError(f"unsupported detailed CPU metric: {metric.name}")
        draws = [np.asarray(reference_draw), np.asarray(candidate_draw)]
        if any(draw.ndim != 1 for draw in draws):
            raise ValueError("detailed fidelity draws must be one-dimensional")
        other, rd, sd = self._pair(draws[0][None, :], draws[1][None, :], ideal=ideal)
        _, minimum, bins, rare = self._parameters(metric)
        with torch.autocast("cpu", enabled=False):
            output = self.ops.fidelity_details(
                *self._arguments(other, rd, sd),
                list(self.context.inputs.cardinalities),
                metric.name,
                minimum,
                bins,
                rare,
                self.context.workers,
                component_mask(metric) & 7,
            )
        values = dict(zip(metric.stat_names, output.pop("values").tolist(), strict=True))
        tables = {}
        for key, tensor in output.items():
            table, column = key.split(".", 1)
            tables.setdefault(table, {})[column] = tensor.numpy()
        return DetailedOutcome(
            values,
            FidelityDetails({name: DetailTable(columns) for name, columns in tables.items()}),
        )

    @torch.no_grad()
    def compute_classifier(
        self,
        metric: LinearC2ST,
        reference_draws: ArrayLike,
        candidate_draws: ArrayLike,
        reference_folds: ArrayLike,
        candidate_folds: ArrayLike,
        seeds: Sequence[SeedScalar],
        *,
        ideal: bool = False,
        point: bool = False,
        fit_threads: int | None = None,
    ) -> list[ClassifierOutcome]:
        """Fit and score source-disjoint Linear C2ST folds for each draw.

        Index and fold matrices have matching shapes per source; repeated source rows
        must keep one fold, including across classes in ideal mode. ``seeds`` supplies
        one exact uint64 seed per task. Native execution handles preprocessing, solver
        selection, and held-out scoring; this adapter does not create folds or resample.

        ``point`` selects point-estimate solver options. An explicit ``fit_threads``
        bounds inner workers independently; otherwise point fits use the context
        budget and bootstrap fits cap inner threads to avoid oversubscription.
        Return ordered ``ClassifierOutcome`` objects with fold-level diagnostics; a
        returned AUC and its convergence diagnostics must be read together.
        """
        from tabgen_eval.execution.runtime.outcomes import ClassifierOutcome

        if metric.name != "linear_c2st":
            raise ValueError("compiled CPU classifier execution requires Linear C2ST")
        other, rd, sd = self._pair(reference_draws, candidate_draws, ideal=ideal)
        from tabgen_eval.execution.classifier.protocol import folds_and_seeds

        arrays, seed_items = folds_and_seeds(
            rd.numpy(),
            sd.numpy(),
            reference_folds,
            candidate_folds,
            seeds,
            metric.cv_folds,
            ideal=ideal,
        )
        folds = [torch.from_numpy(array) for array in arrays]
        # ATen carries seeds in int64 tensors; reinterpret uint64 bits rather
        # than numerically converting values above the signed range.
        seed_tensor = torch.from_numpy(np.array(seed_items, dtype=np.uint64).view(np.int64))
        # The outer draw scheduler and each inner fit share the worker budget.
        # Point estimates have no competing bootstrap fits in this dispatch.
        if fit_threads is None:
            fit_threads = (
                self.context.workers
                if point
                else min(metric.bootstrap_fit_threads, self.context.workers)
            )
        elif (
            isinstance(fit_threads, bool)
            or not isinstance(fit_threads, Integral)
            or not 1 <= fit_threads <= self.context.workers
        ):
            raise ValueError("fit_threads must be an integer within the context worker budget")
        fit_threads = int(fit_threads)
        from tabgen_eval.execution.runtime.memory import plan_classifier_memory

        plan = plan_classifier_memory(
            metric,
            self.context,
            rd.shape[1] + sd.shape[1],
            concurrent_tasks=min(len(rd), max(1, self.context.workers // fit_threads)),
        )
        workers = min(self.context.workers, plan.concurrent_tasks * fit_threads)
        options = metric.native_options(point=point)
        with torch.autocast("cpu", enabled=False):
            scores, diagnostics, solvers, reasons, preprocessing = self.ops.classifier(
                *self._arguments(other, rd, sd),
                *folds,
                seed_tensor,
                options["n_folds"],
                options["solver"],
                options["ridge"],
                options["tolerance"],
                options["max_iter"],
                options["direct_max_columns"],
                options["direct_max_work"],
                options["saga_max_epochs"],
                options["saga_batch_size"],
                options["saga_rank"],
                int(options["memory_limit_mb"] * 1024**2),
                workers,
                fit_threads,
                options["saga_backoff"],
                options["saga_sampled_preparation"],
                options["saga_preparation_rows"],
            )
        outcomes = []
        for task, value in enumerate(scores.numpy()):
            rows = []
            for fold, row in enumerate(diagnostics[task].numpy()):
                # Native string metadata is flattened task-major, while the
                # numeric record retains separate task and fold dimensions.
                index = task * metric.cv_folds + fold
                data = classifier_fold(
                    row, fold=fold, solver=solvers[index], reason=reasons[index], options=options
                )
                data["preprocessing"] = fold_metadata(
                    preprocessing[index],
                    self.context.inputs.numeric_names,
                    self.context.inputs.categorical_names,
                    self.context.category_values,
                )
                rows.append(data)
            outcomes.append(ClassifierOutcome(float(value), rows))
        return outcomes
