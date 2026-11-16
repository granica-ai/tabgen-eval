"""Unified public orchestration over explicit data roles and metric selections."""

from __future__ import annotations

from collections.abc import Sequence

from ._evaluation.memorization import evaluate_memorization
from ._evaluation.paired import evaluate_pair
from ._evaluation.real import evaluate_test_only
from ._options import UncertaintyOptions, describe_options, resolve_options
from ._typing import Precision, SeedLike
from ._validation import is_mapping, require_instance
from .data import EvalData
from .execution.runtime.domains import metric_input_schema
from .execution.runtime.memorization import validate_memorization_options
from .metrics.memorization import uses_memorization_protocol
from .models.xgboost import XGBoostModel
from .registry import build_metrics, resolve_selection
from .report import Report
from .report.evaluation import project_results
from .schema import validate


def evaluate(
    data: EvalData,
    *,
    group: Sequence[str] = (),
    metrics: Sequence[str] = (),
    uncertainty: UncertaintyOptions | None = None,
    seed: SeedLike = 42,
    device: str = "auto",
    dtype: Precision = "float64",
    n_jobs: int = 8,
    details: bool = False,
) -> Report:
    """Evaluate selected metrics using their required train/test/generated roles.

    With no filters, run every metric supported by the supplied data roles.
    Nonempty ``metrics`` overrides ``group``; both accept only registered names.
    Explicit requests with missing roles raise. Utility requires a target in
    ``data.schema``, including when selected by the default. Required optional
    dependencies are checked before computation.

    Test alone compares disjoint subsets of the same real pool. Train and
    generated support near-match rate and replication ratio; DCR ratio also
    requires test. Generated and test support all non-memorization metrics.
    Supplying all three roles supports every metric in one call.

    ``uncertainty=None`` requests fixed scores. To add uncertainty, supply a
    dictionary with required ``mode="reference"``, ``"candidate"`` or ``"both"``
    and optional positive integer ``n_replicates`` (default 10). Invalid dictionaries raise
    with the expected keys and an example before computation begins. The input
    dictionary is not modified. Sampling is without replacement. Reference-only
    uncertainty keeps the generated sample fixed. Candidate-only uncertainty
    repeats generated-real scores without a real-real interval. Candidate-only
    and both-side uncertainty use half the eligible holdout when generated size
    exceeds that half, otherwise
    half the eligible generated pool. Point scores and repetitions share that
    resolution; changing to ``"candidate"`` or ``"both"`` can therefore change
    point scores.
    Memorization remains point-only. Execution and seed are keyword-only
    arguments; registered metrics use fixed definitions and memory planning is
    automatic. Data diagnostics are available independently through ``diagnose``.
    """
    data = require_instance(data, EvalData, "data must be an EvalData instance")
    require_instance(details, bool, "details must be a boolean", error=ValueError)
    options = resolve_options(
        uncertainty=uncertainty,
        seed=seed,
        device=device,
        dtype=dtype,
        n_jobs=n_jobs,
    )
    if options["uncertainty"] == "candidate" and data.generated is None:
        raise ValueError("Candidate uncertainty requires EvalData.generated")
    selection = resolve_selection(group=group, metrics=metrics, roles=data.roles)
    chosen = build_metrics(selection, options)
    comparison = [metric for metric in chosen if metric.category != "memorization"]
    memory = [metric for metric in chosen if uses_memorization_protocol(metric)]
    if memory:
        validate_memorization_options(memory)
    if details and (data.generated is None or not comparison):
        raise ValueError("details requires a generated/test point comparison")

    if any(metric.needs_target for metric in chosen) and data.schema.target is None:
        raise ValueError("Utility requires EvalData.schema with an explicit Schema.target")
    # Actual transforms still fit only on assigned draws, training pools, or folds.
    source = data.test if comparison else data.train
    assert source is not None  # Registry role requirements guarantee the reference pool.
    schema = data.schema
    used_roles = {role for spec in selection for role in spec.required_roles}
    if comparison and data.generated is not None:
        used_roles.add("generated")
    for role in ("train", "test", "generated"):
        if role in used_roles:
            validate(
                source,
                getattr(data, role),
                metric_input_schema(schema, comparison if role != "train" else memory),
                allow_empty=role == "test" and not comparison,
                reference_name="test" if comparison else "train",
                candidate_name=role,
            )
    for metric in chosen:
        reference = data.train if metric.category == "memorization" else data.test
        candidate = data.generated if data.generated is not None else data.test
        assert reference is not None and candidate is not None
        metric.validate_inputs(reference, candidate, schema)
    if any(getattr(metric, "execution_backend", None) == "xgboost" for metric in chosen):
        XGBoostModel.load_backend()

    reports: dict[str, Report] = {}
    if comparison:
        assert data.test is not None
        if data.generated is None:
            reports["test_only"] = evaluate_test_only(
                data.test,
                schema,
                metrics=comparison,
                options=options,
            )
        else:
            reports["comparison"] = evaluate_pair(
                data.test,
                data.generated,
                schema,
                options=options,
                details=details,
                metrics=comparison,
            )
    if memory:
        assert data.train is not None and data.generated is not None
        reports["memorization"] = evaluate_memorization(
            data.train,
            data.generated,
            data.schema,
            holdout=data.test,
            options=options,
            metrics=memory,
        )

    results = project_results(
        [result for report in reports.values() for result in report.results], selection
    )
    primary = next(iter(reports.values()))
    provenance = dict(primary.provenance)
    execution: dict[str, object] = {}
    for spec in selection:
        procedure = (
            "memorization"
            if spec.group == "memorization"
            else ("test_only" if data.generated is None else "comparison")
        )
        recorded = reports[procedure].provenance.get("metric_execution", {})
        assert is_mapping(recorded)  # Procedure reports own this execution metadata subtree.
        execution[spec.name] = recorded.get(spec.computation, {})
    provenance.update(
        evaluation_kind="test_only" if data.generated is None else "evaluation",
        input_roles=sorted(data.roles),
        input_rows={
            role: len(getattr(data, role))
            for role in ("train", "test", "generated")
            if role in data.roles
        },
        schema_inferred=False,
        selected_metrics=[spec.name for spec in selection],
        procedures={
            name: {
                **report.provenance,
                "schema": report.schema_dict,
                "schema_inferred": False,
            }
            for name, report in reports.items()
        },
        metric_execution=execution,
    )
    return Report(
        results=results,
        schema_dict=primary.schema_dict,
        options_dict={**describe_options(options), "details": details},
        provenance=provenance,
    )
