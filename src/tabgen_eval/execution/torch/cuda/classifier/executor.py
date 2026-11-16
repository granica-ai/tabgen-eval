"""Execute shared classifier draws through the CUDA implementation.

The estimator owns bootstrap positions, source-stable fold assignments, and
per-draw seeds. This adapter validates their protocol, executes folds/draws
sequentially, and returns chance-clipped held-out AUC with failed-fit diagnostics.
Training transforms and solver workspace belong to each fold; source tensor
pools remain owned by the surrounding context. Autocast is disabled locally
without changing process-wide precision settings or caller RNG streams.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch
from numpy.typing import ArrayLike

from tabgen_eval._typing import IntArray, SeedScalar
from tabgen_eval.execution.classifier.protocol import folds_and_seeds
from tabgen_eval.execution.runtime.outcomes import ClassifierOutcome
from tabgen_eval.execution.torch.cuda.classifier.data import prepare
from tabgen_eval.execution.torch.cuda.classifier.solver import auc, fit
from tabgen_eval.execution.torch.distance.executor import DistanceExecutor

if TYPE_CHECKING:
    from tabgen_eval.execution.torch.context import TorchContext
    from tabgen_eval.execution.torch.tensors import TensorTable
    from tabgen_eval.metrics import LinearC2ST, Metric
    from tabgen_eval.schema import Schema


class ClassifierExecutor:
    """Run source-stable CUDA classifier folds with isolated fit workspace.

    Parameters
    ----------
    context : TorchContext
        Owns read-only source tensors, declared precision and GPU workspace budget.

    Notes
    -----
    Draws/folds execute sequentially, retaining failed-fit diagnostics. Library handle
    reserve is separate from the model storage limit and global precision is preserved.
    """

    def __init__(self, context: TorchContext) -> None:
        """Retain the context that owns source tensors and GPU resource limits.

        Construction performs no fitting, draw gathering, or native-library load.
        The context must outlive this adapter and owns reference/candidate buffers.
        """
        self.context = context

    @staticmethod
    def batch_size(metric: Metric, n: int) -> int:
        """Bound the estimator's host draw batch by an 8 MiB index allowance.

        ``n`` is the maximum per-draw row count; two int64 index arrays determine
        the estimate. Return at least one and at most 64. Fits still execute one
        draw at a time; this value is unrelated to SAGA's training minibatch size.
        The metric argument exists to match the shared executor interface.
        """
        return max(1, min(64, (8 * 1024**2) // max(16 * n, 1)))

    @staticmethod
    def not_applicable(metric: Metric, schema: Schema) -> set[str]:
        """Delegate an inapplicable feature schema to the metric's result policy.

        No rows are materialized and no GPU operation is needed. The metric owns
        the returned NaN/diagnostic representation through the shared interface.
        """
        return metric.not_applicable(None, None, schema)

    def _draw(
        self,
        other: TensorTable,
        rd: IntArray,
        sd: IntArray,
        rf: IntArray,
        sf: IntArray,
        seed: int,
        metric: LinearC2ST,
        point: bool,
    ) -> ClassifierOutcome:
        """Gather and score one source-stable two-class draw fold by fold.

        ``rd``/``sd`` are source positions, ``rf``/``sf`` their validated fold labels,
        and ``seed`` selects private fit randomness. ``other`` is the candidate or
        reference pool for ordinary or ideal evaluation. Return one ClassifierOutcome
        with mean chance-clipped AUC and all fold states, retaining nonconvergence.

        The shared GPU allowance reserves 16 MiB for library handles; each model is
        additionally constrained by the metric's byte limit. An impossible draw-level
        reservation raises MemoryError. A fold that cannot prepare or fit instead
        returns NaN with its reason, and failed numerical/resource fits are never
        scored as chance placeholders. Fit-local designs/weights are released before
        the next fold; pool buffers and external RNG state remain unchanged.
        """
        c = self.context
        item = c.reference.numeric.element_size()
        width = sum(c.inputs.cardinalities) + c.n_categorical
        # Conservatively reserve stream-bound library handles against the GPU
        # budget. They are shared resources outside the model memory limit.
        gpu_limit = c.gpu_workspace_mb * 1024**2 - 16 * 1024**2
        reserved = (
            (len(rd) + len(sd)) * (c.n_numeric * item * 8 + c.n_categorical * 32 + 128)
            + width * 64
            + 65536
        )
        if reserved > gpu_limit:
            raise MemoryError(
                "C2ST draws and preprocessing exceed the resolved workspace allowance"
            )
        limit = int(min(gpu_limit, metric.memory_limit_mb * 1024**2))
        options = metric.native_options(point=point)
        result, values = [], []
        if reserved <= limit:
            a, b = c.reference.gather(rd), other.gather(sd)
            rf, sf = [
                torch.as_tensor(np.array(v, copy=True), device=c.inputs.device, dtype=torch.int64)
                for v in (rf, sf)
            ]
        for fold in range(metric.cv_folds):
            data = (
                prepare(a, b, rf, sf, fold, c.inputs.cardinalities, options, limit, reserved)
                if reserved <= limit
                else None
            )
            if data is None:
                result.append(
                    dict(
                        fold=fold,
                        training_rows=0,
                        scoring_rows=0,
                        features=0,
                        solver="none",
                        converged=False,
                        reason="preprocessing_or_memory_limit",
                        tolerance=options["tolerance"],
                    )
                )
                values.append(float("nan"))
                continue
            if data.train.d == 1:
                from tabgen_eval.preprocessing.reporting import fold_metadata

                result.append(
                    dict(
                        fold=fold,
                        training_rows=data.train.n,
                        scoring_rows=data.test.n,
                        features=0,
                        solver="none",
                        converged=False,
                        reason="preprocessing_or_memory_limit",
                        tolerance=options["tolerance"],
                        preprocessing=fold_metadata(
                            data.preprocessing_metadata,
                            c.inputs.numeric_names,
                            c.inputs.categorical_names,
                            c.category_values,
                        ),
                    )
                )
                values.append(float("nan"))
                continue
            weights, state = fit(data, options, (seed + fold * 0x9E3779B97F4A7C15) % 2**64)
            value = (
                float("nan")
                if state.get("reason") in {"memory_limit", "numerical_failure"}
                else auc(data.test.mv(weights), data.scoring_labels)
            )
            values.append(max(0.5, value) if np.isfinite(value) else float("nan"))
            for key in ["gradient_squared", "decrement_squared"]:
                if state.get(key) is None or not np.isfinite(state[key]):
                    state[key] = None
            from tabgen_eval.preprocessing.reporting import fold_metadata

            preprocessing = fold_metadata(
                data.preprocessing_metadata,
                c.inputs.numeric_names,
                c.inputs.categorical_names,
                c.category_values,
            )
            result.append(
                dict(
                    fold=fold,
                    training_rows=data.train.n,
                    scoring_rows=data.test.n,
                    features=data.train.d - 1,
                    preprocessing=preprocessing,
                    **state,
                )
            )
            del data, weights
        from tabgen_eval.execution.runtime.outcomes import ClassifierOutcome

        return ClassifierOutcome(float(np.mean(values)), result)

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
        """Execute classifier draws without changing source folds or RNG state.

        Parameters
        ----------
        metric : LinearC2ST
            Classifier protocol and solver options.
        reference_draws, candidate_draws : array-like
            Integer source positions ``[B, NR]``/``[B, NS]`` preserving duplicates.
        reference_folds, candidate_folds : array-like
            Matching original-source fold labels; conflicting identities raise.
        seeds : array-like
            One exact uint64 seed per draw.
        ideal : bool
            Use the real source for both classes with shared-source fold validation.
        point : bool
            Use point-fit direct-Newton cutoffs rather than bootstrap cutoffs.
        fit_threads : int or None
            Shared protocol CPU fit budget; validated but unused by GPU solvers.

        Returns
        -------
        list of ClassifierOutcome
            Chance-clipped mean held-out AUC and all fold diagnostics per draw.

        Notes
        -----
        Ambient autocast is disabled locally and restored. Sources remain read-only;
        NumPy/the estimator owns sampling and bootstrap ordering.
        """
        if fit_threads is not None and (
            isinstance(fit_threads, bool) or not isinstance(fit_threads, int) or fit_threads < 1
        ):
            raise ValueError("fit_threads must be a positive integer or None")
        c = self.context
        other = c.reference if ideal else c.candidate
        rd = DistanceExecutor._draws(reference_draws, c.reference.n_rows)
        sd = DistanceExecutor._draws(candidate_draws, other.n_rows)
        if len(rd) != len(sd):
            raise ValueError("draw arrays must have equal batch lengths")
        (rf, sf), seeds = folds_and_seeds(
            rd, sd, reference_folds, candidate_folds, seeds, metric.cv_folds, ideal=ideal
        )
        outcomes = []
        with torch.autocast("cuda", enabled=False):
            for r, s, a, b, seed in zip(rd, sd, rf, sf, seeds, strict=True):
                outcomes.append(self._draw(other, r, s, a, b, seed, metric, point))
        return outcomes
