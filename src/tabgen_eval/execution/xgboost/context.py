"""Sequential native categorical C2ST over reusable compiled source storage."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from joblib import effective_n_jobs
from numpy.typing import ArrayLike
from sklearn.metrics import roc_auc_score

from tabgen_eval._typing import IntArray, SeedScalar
from tabgen_eval.execution.classifier.protocol import folds_and_seeds
from tabgen_eval.execution.runtime.outcomes import ClassifierOutcome
from tabgen_eval.models import FitUnavailable, XGBoostModel
from tabgen_eval.schema import Schema

if TYPE_CHECKING:
    from tabgen_eval.metrics import ClassifierMetric, Metric
    from tabgen_eval.preprocessing.sources import PreparedSources


class XGBoostContext:
    """Own reusable sources; fit one native-category forest at a time.

    The workspace limit includes gathered inputs, fold matrices, and a
    conservative estimate for XGBoost histograms/quantization. XGBoost does not
    expose an allocator hard cap; that estimate is a preflight guard, not an
    allocation guarantee. Persistent source storage is reported separately.
    """

    _automatically_managed_memory: bool

    def __init__(
        self, sources: PreparedSources, *, n_jobs: int = 1, gpu_workspace_mb: int = 2048
    ) -> None:
        if n_jobs == 0:
            raise ValueError("n_jobs must be non-zero")
        if gpu_workspace_mb <= 0:
            raise ValueError("gpu_workspace_mb must be positive")
        self.inputs = sources
        self.reference, self.candidate = sources.reference, sources.candidate
        self.workers: int = effective_n_jobs(n_jobs)
        self.gpu_workspace_mb = gpu_workspace_mb
        self.n_numeric = len(sources.numeric_names)
        self.n_categorical = len(sources.categorical_names)
        self.classifier_width = self.n_numeric + self.n_categorical
        self.category_mismatch = sources.category_mismatch

    @property
    def execution_info(self) -> dict[str, object]:
        """Describe source precision separately from native engine precision."""
        return {
            "implementation": "xgboost",
            "device": str(self.reference.numeric.device),
            "dtype": self.inputs.precision["feature_dtype"],
            "preprocessing": dict(self.inputs.precision),
            "source_preparation_seconds": self.inputs.preparation_seconds,
            "source_pools": 1 if self.reference is self.candidate else 2,
            "persistent_input_bytes": self.reference.nbytes
            + (0 if self.reference is self.candidate else self.candidate.nbytes),
            "engine_feature_precision": "float32",
        }

    def execution_info_for(self, metric: Metric) -> dict[str, object]:
        info = self.execution_info
        if hasattr(metric, "_xgboost_routing_reason"):
            info["routing_reason"] = metric._xgboost_routing_reason
        if hasattr(metric, "_memory_plan"):
            info["memory_plan"] = dict(metric._memory_plan)
        return info

    def batch_size(self, metric: Metric, n: int) -> int:
        """Do not retain multiple gathered GPU comparisons simultaneously."""
        return 1

    def not_applicable(self, metric: Metric, schema: Schema) -> set[str]:
        return metric.not_applicable(None, None, schema)

    @staticmethod
    def _draws(draws: ArrayLike, n_rows: int) -> IntArray:
        array = np.asarray(draws)
        if array.ndim != 2 or array.dtype.kind not in "iu":
            raise ValueError("draw arrays must be two-dimensional integer arrays")
        if array.size and (array.min() < 0 or array.max() >= n_rows):
            raise ValueError("draw index is outside the input pool")
        return array

    def _budget(self, metric: ClassifierMetric, nr: int, ns: int) -> tuple[int, dict[str, object]]:
        """Reserve live gathers and native forest work before allocating folds."""
        limit = int(metric.memory_limit_mb * 1024**2)
        if self.reference.numeric.device.type == "cuda":
            limit = min(limit, int(self.gpu_workspace_mb * 1024**2))
        rows = nr + ns
        gathered = rows * (
            self.n_numeric * self.reference.numeric.element_size() + 8 * self.n_categorical + 1 + 16
        )
        # Quantile storage plus gradient/histogram and default depth-six tree
        # buffers. Compact native columns avoid category-cardinality expansion.
        from tabgen_eval.execution.runtime.memory import xgboost_workspace_bytes

        engine = xgboost_workspace_bytes(rows, self.n_numeric, self.inputs.cardinalities)
        remaining = limit - gathered - engine
        if remaining <= 0:
            raise FitUnavailable(
                "xgboost_workspace_limit",
                estimated_bytes=gathered + engine,
                memory_limit_bytes=limit,
            )
        return remaining, dict(
            gathered_bytes=gathered,
            engine_workspace_estimate_bytes=engine,
            memory_limit_bytes=limit,
            engine_workspace_is_estimated=True,
        )

    def compute_classifier(
        self,
        metric: ClassifierMetric,
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
        """Preserve supplied folds, duplicates, and comparison-specific seeds."""
        from tabgen_eval.preprocessing import fit_fold

        other = self.reference if ideal else self.candidate
        rd = self._draws(reference_draws, self.reference.n_rows)
        sd = self._draws(candidate_draws, other.n_rows)
        if len(rd) != len(sd):
            raise ValueError("draw arrays must have equal batch lengths")
        (rf, sf), seeds = folds_and_seeds(
            rd,
            sd,
            reference_folds,
            candidate_folds,
            seeds,
            metric.cv_folds,
            ideal=ideal,
        )
        # Load before noncomputability checks: missing optional dependencies are
        # installation errors, not statistically unavailable comparisons.
        XGBoostModel.load_backend()
        if fit_threads is not None and (
            not isinstance(fit_threads, int)
            or isinstance(fit_threads, bool)
            or not 1 <= fit_threads <= self.workers
        ):
            raise ValueError("fit_threads must be between 1 and the context worker budget")
        threads = (
            fit_threads
            if fit_threads is not None
            else (self.workers if point else min(self.workers, metric.bootstrap_fit_threads))
        )
        from tabgen_eval.execution.runtime.memory import plan_classifier_memory

        plan_classifier_memory(metric, self, rd.shape[1] + sd.shape[1])
        self.gpu_workspace_mb = metric.gpu_workspace_mb
        outcomes = []
        for r, s, a, b, seed in zip(rd, sd, rf, sf, seeds, strict=True):
            values, diagnostics = [], []
            reference = candidate = None
            try:
                remaining, workspace = self._budget(metric, len(r), len(s))
                reference, candidate = self.reference.gather(r), other.gather(s)
            except FitUnavailable as error:
                reference = candidate = None
                info = dict(error.diagnostics, reason=error.reason, converged=False)
                outcomes.append(ClassifierOutcome(float("nan"), [info]))
                continue
            for fold in range(metric.cv_folds):
                model = data = None
                try:
                    data = fit_fold(
                        reference,
                        candidate,
                        a,
                        b,
                        fold,
                        cardinalities=self.inputs.cardinalities,
                        numeric_names=self.inputs.numeric_names,
                        categorical_names=self.inputs.categorical_names,
                        category_values=self.inputs.category_values,
                        policy="xgboost",
                        memory_limit_bytes=remaining,
                    )
                    model = XGBoostModel(device=str(reference.numeric.device))
                    scores, info = model.fit_predict(
                        data.train,
                        data.training_labels,
                        data.test,
                        seed=int(seed) + fold,
                        fit_threads=threads,
                        feature_types=data.feature_types,
                        feature_names=data.feature_names,
                        categorical_cardinalities=data.categorical_cardinalities,
                    )
                    if not np.isfinite(scores).all():
                        raise FitUnavailable("nonfinite_predictions")
                    labels = data.scoring_labels.detach().cpu().numpy()
                    values.append(max(float(roc_auc_score(labels, scores)), 0.5))
                    info.update(
                        fold=fold,
                        training_rows=len(data.train),
                        scoring_rows=len(data.test),
                        features=data.train.shape[1],
                        preprocessing=dict(data.diagnostics),
                        workspace=workspace,
                    )
                    diagnostics.append(info)
                except (FitUnavailable, FloatingPointError) as error:
                    values.append(np.nan)
                    diagnostics.append(
                        dict(
                            getattr(error, "diagnostics", {}),
                            fold=fold,
                            converged=False,
                            reason=getattr(error, "reason", "preprocessing_or_memory_limit"),
                        )
                    )
                finally:
                    # Failed native fits can own substantial fold/engine buffers.
                    # Release them before preparing another fold or comparison.
                    model = data = None
            outcomes.append(ClassifierOutcome(float(np.mean(values)), diagnostics))
            del reference, candidate
        return outcomes
