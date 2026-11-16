"""Evaluate fixed paired scores and optional empirical score distributions."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd

from tabgen_eval._options import ResolvedOptions
from tabgen_eval._typing import SeedLike
from tabgen_eval.execution.classifier.protocol import uses_classifier_protocol
from tabgen_eval.execution.runtime.dispatch import ExecutionSession, supports, supports_details
from tabgen_eval.execution.runtime.domains import (
    RowUsage,
    SourceDomain,
    eligible_fixed_draw,
    metric_eligibility,
)
from tabgen_eval.execution.runtime.metadata import execution_info
from tabgen_eval.execution.runtime.outcomes import FixedMetricResult
from tabgen_eval.execution.runtime.paired import compute_fixed_metric
from tabgen_eval.execution.runtime.sampling import FixedDraw
from tabgen_eval.preprocessing.categories import CategoryVocabulary
from tabgen_eval.report.assembly import assemble_stat, point_stat_result, utility_metric_result
from tabgen_eval.report.diagnostics import DiagnosticFlag
from tabgen_eval.report.result import MetricResult
from tabgen_eval.uncertainty.paired import compute_replicates

if TYPE_CHECKING:
    from tabgen_eval.execution.runtime.context import ContextProvider, NumericalContext
    from tabgen_eval.metrics.base import Metric
    from tabgen_eval.schema import Schema

NAN = float("nan")


def _metric_seed(seed: SeedLike, metric_name: str) -> np.random.SeedSequence:
    """Isolate numerical randomness from shared source sampling."""
    metric_id = int.from_bytes(hashlib.sha256(metric_name.encode()).digest()[:4], "big")
    return np.random.SeedSequence(seed, spawn_key=(metric_id,))


class _ContextSession:
    """Expose an injected numerical context through the session interface."""

    def __init__(self, context: NumericalContext) -> None:
        self.context = context

    def context_for(self, metric: Metric) -> NumericalContext:
        return self.context


class Estimator:
    """Compute paired point scores and raw repeated-score summaries for one metric."""

    def __init__(
        self,
        metric: Metric,
        options: ResolvedOptions,
        *,
        context: NumericalContext | None = None,
        session: ContextProvider | None = None,
        category_mismatch: bool | None = None,
        details: bool = False,
        draw: FixedDraw | None = None,
        row_usage: RowUsage | None = None,
    ) -> None:
        self.metric, self.options = metric, options
        for option in ("device", "dtype"):
            if option not in vars(metric):
                setattr(metric, option, options[option])
        self.context = context
        if not supports(metric):
            self.context = None
        self.session = session
        self.category_mismatch = category_mismatch
        if details and not supports_details(metric):
            raise ValueError(f"Detailed output is unavailable for {metric.name!r}")
        self.capture_details = details or (
            supports_details(metric) and metric.name == "concentration"
        )
        self.draw = draw
        self.row_usage = row_usage

    def run(self, holdout: pd.DataFrame, generated: pd.DataFrame, schema: Schema) -> MetricResult:
        """Use one mode-specific resolution for points and repeated comparisons."""
        metric, options = self.metric, self.options
        metric.validate_inputs(holdout, generated, schema)
        draw = self.draw
        self._active_row_usage = self.row_usage
        policy, columns = metric_eligibility(metric, schema)
        if policy is not None and self.row_usage is None:
            domain = SourceDomain.prepare(
                {"test": holdout, "generated": generated}, policy=policy, columns=columns
            )
            holdout, generated = domain.pools["test"], domain.pools["generated"]
            if any(counts["excluded"] for counts in domain.counts.values()):
                self.context = self.session = None
                draw = None
            draw = draw or eligible_fixed_draw(
                len(holdout),
                len(generated),
                seed=options["seed"],
                uncertainty=options["uncertainty"],
            )
            self._active_row_usage = domain.row_usage(draw.n_res)
        from tabgen_eval.metrics.utility import MLE

        if (
            isinstance(metric, MLE)
            and "compute" not in vars(metric)
            and type(metric).compute is MLE.compute
        ):
            from tabgen_eval.execution.runtime.utility import run_utility

            outcome = run_utility(
                holdout, generated, schema, metric=metric, options=options, draw=draw
            )
            return self._with_row_usage(
                utility_metric_result(
                    metric,
                    outcome,
                    base_flags=self._base_flags(holdout, generated, schema),
                    unavailable=metric.not_applicable(holdout, generated, schema),
                )
            )
        if not metric.bootstrapped:
            raw = metric.compute(
                holdout,
                generated,
                schema,
                rng=np.random.default_rng(_metric_seed(options["seed"], metric.name)),
            )
            return self._with_row_usage(
                self._descriptive_result(
                    metric, raw, len(holdout), len(generated), min(len(holdout), len(generated))
                )
            )
        draw = draw or eligible_fixed_draw(
            len(holdout),
            len(generated),
            seed=options["seed"],
            uncertainty=options["uncertainty"],
        )
        if self.context is not None:
            session = _ContextSession(self.context)
        elif self.session is not None:
            session = self.session
        else:
            session = ExecutionSession(
                holdout,
                generated,
                schema,
                n_jobs=options["n_jobs"],
                n_eff=draw.n_res,
                n_bootstrap=options["n_replicates"] if options["uncertainty"] != "none" else 0,
                retain_labels=self.capture_details and metric.name == "conditional_fidelity",
                classifier_preprocessing=uses_classifier_protocol(metric),
            )
        insufficient = draw.n_res < max(1, metric.n_min)
        repeated = options["uncertainty"] != "none" and options["n_replicates"] > 0
        repeat_generated = repeated and options["uncertainty"] in {"candidate", "both"}
        repeat_reference = repeated and options["uncertainty"] in {"reference", "both"}
        if insufficient or (repeat_generated and repeat_reference):
            missing = {name: NAN for name in metric.stat_names}
            point = FixedMetricResult(
                metric.name,
                missing.copy(),
                missing.copy(),
                dict(implementation="unavailable") if insufficient else {},
                dict(reason="fewer sampled rows than the metric minimum", n_min=metric.n_min)
                if insufficient
                else {},
            )
        else:
            point = compute_fixed_metric(
                metric,
                holdout,
                generated,
                schema,
                draw,
                session=session,
                seed=options["seed"],
                n_jobs=options["n_jobs"],
                details=self.capture_details and not repeat_generated,
                include_reference=not repeat_reference,
                include_generated=not repeat_generated,
            )
        value_reps, ideal_reps, repeated_diagnostics = {}, {}, {}
        if repeated:
            if insufficient:
                if options["uncertainty"] in {"reference", "both"}:
                    ideal_reps = {
                        name: np.full(options["n_replicates"], NAN) for name in metric.stat_names
                    }
                if options["uncertainty"] in {"candidate", "both"}:
                    value_reps = {
                        name: np.full(options["n_replicates"], NAN) for name in metric.stat_names
                    }
            else:
                value_reps, ideal_reps, repeated_diagnostics, repeated_details = compute_replicates(
                    metric,
                    holdout,
                    generated,
                    schema,
                    draw,
                    session=session,
                    options=options,
                    details=self.capture_details and repeat_generated,
                )
                if repeated_details is not None:
                    point = replace(point, details=repeated_details)
        result = MetricResult(name=metric.name, category=metric.category)
        result.details = point.details
        result.diagnostics = dict(
            point=point.diagnostics,
            replicates=repeated_diagnostics,
            execution=point.execution
            if insufficient
            else execution_info(metric, session.context_for(metric)),
            sampling=dict(
                uncertainty=options["uncertainty"],
                n_res=draw.n_res,
                real_roles_disjoint=True,
                replacement=False,
                shared_real_reference=True,
                estimate="replicate_mean" if repeated else "single_comparison",
            ),
        )
        flags = self._base_flags(holdout, generated, schema)
        if (
            not insufficient
            and uses_classifier_protocol(metric)
            and any(
                not cast("Mapping[str, object]", fit).get("converged", True)
                for fit in (
                    *point.diagnostics.values(),
                    *(fit for side in repeated_diagnostics.values() for fit in side),
                )
            )
        ):
            flags.append(DiagnosticFlag.CLASSIFIER_NOT_CONVERGED)
        not_applicable = set()
        if not insufficient and repeat_generated:
            not_applicable = metric.not_applicable(holdout, generated, schema)
            # A slice can be rare in a sampled comparison even when it is not
            # rare in the full pool. Any finite repetition proves applicability.
            not_applicable.difference_update(
                name for name, values in value_reps.items() if np.isfinite(values).any()
            )
        elif not insufficient and any(not np.isfinite(value) for value in point.values.values()):
            not_applicable = metric.not_applicable(
                holdout.iloc[draw.reference],
                generated.iloc[draw.generated],
                schema,
            )
        for spec in metric.stats:
            result.stats[spec.name] = assemble_stat(
                spec=spec,
                value_reps=value_reps.get(spec.name),
                ideal_reps=ideal_reps.get(spec.name),
                value_point=point.values.get(spec.name, NAN),
                ideal_point=point.real_reference.get(spec.name, NAN),
                base_flags=flags,
                insufficient=insufficient
                or (uses_classifier_protocol(metric) and draw.n_res < metric.cv_folds),
                not_applicable=spec.name in not_applicable,
                n_holdout=len(holdout),
                n_generated=len(generated),
                n_eff=draw.n_res,
            )
        return self._with_row_usage(result)

    def _with_row_usage(self, result: MetricResult) -> MetricResult:
        """Retain pre-sampling exclusions independently of detailed metric output."""
        if self._active_row_usage is not None:
            result.diagnostics["row_usage"] = self._active_row_usage
        return result

    def _descriptive_result(
        self,
        metric: Metric,
        raw: Mapping[str, float],
        n_holdout: int,
        n_generated: int,
        n_eff: int,
    ) -> MetricResult:
        """Retain full-input descriptive statistics without repeated sampling."""
        result = MetricResult(name=metric.name, category=metric.category)
        for spec in metric.stats:
            result.stats[spec.name] = point_stat_result(
                spec,
                raw.get(spec.name, NAN),
                n_holdout=n_holdout,
                n_generated=n_generated,
                n_eff=n_eff,
            )
        return result

    def _base_flags(
        self, holdout: pd.DataFrame, generated: pd.DataFrame, schema: Schema
    ) -> list[DiagnosticFlag]:
        mismatch = self.category_mismatch
        if mismatch is None:
            mismatch = self._category_mismatch(holdout, generated, schema)
        return [DiagnosticFlag.CATEGORY_VOCAB_MISMATCH] if mismatch else []

    @staticmethod
    def _category_mismatch(holdout: pd.DataFrame, generated: pd.DataFrame, schema: Schema) -> bool:
        for col in schema.categorical:
            if col not in holdout.columns or col not in generated.columns:
                continue
            if set(CategoryVocabulary.fit(holdout[col]).keys) != set(
                CategoryVocabulary.fit(generated[col]).keys
            ):
                return True
        return False
