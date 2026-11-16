"""Focused report views that share stored metric results and evaluation metadata."""

from __future__ import annotations

from typing import TypeVar

from .io import Report


class FidelityReport(Report):
    """Distributional, geometric and conditional fidelity results.

    Includes Shape, Trend, Conditional, PRDC, classifier two-sample tests,
    and Concentration. Access this view through ``report.fidelity``.
    Results and captured details are shared with the parent report. Metadata
    describes the family's own procedure; constructing a view performs no
    metric computation.
    """


class MemorizationReport(Report):
    """Training-referenced distances, generated near matches and replication ratios.

    Access this view through ``report.memorization``. Inherited
    exports and plotting use the stored point results and their evaluation
    metadata, including search choices, thresholds and selected row counts.
    """


class UtilityReport(Report):
    """Machine-learning efficacy and shared-test prediction comparisons.

    Access this view through ``report.utility`` after ``evaluate(...,
    metrics=["mle"])``. Raw generated/real-trained AUROC or MAE scores,
    their generated-to-real ratios and prediction agreement or MSE share the
    evaluation's fitted models and real scoring rows.
    """


_FAMILY_CATEGORIES = {
    FidelityReport: frozenset(
        {
            "fidelity",
            "conditional_fidelity",
            "geometric_fidelity",
            "concentration",
            "detection",
        }
    ),
    MemorizationReport: frozenset({"memorization"}),
    UtilityReport: frozenset({"utility"}),
}

_ReportT = TypeVar("_ReportT", bound=Report)


def family_view(report: Report, report_type: type[_ReportT]) -> _ReportT:
    """Select stored results and contextualize metadata for their procedure."""
    categories = _FAMILY_CATEGORIES[report_type]
    results = [result for result in report.results if result.category in categories]
    procedures = report.provenance.get("procedures", {})
    route = (
        "memorization"
        if report_type is MemorizationReport
        else (
            "test_only" if report.provenance.get("evaluation_kind") == "test_only" else "comparison"
        )
    )
    procedure = procedures.get(route)
    provenance, options, schema = report.provenance, report.options_dict, report.schema_dict
    if procedure is not None:
        schema = procedure.get("schema", schema)
        context = {
            name: provenance[name]
            for name in ("input_roles", "input_rows", "procedures")
            if name in provenance
        }
        provenance = {**context, **procedure}
        if procedure.get("n_eff") is None:
            resolutions = {stat.n_eff for result in results for stat in result.stats.values()}
            provenance["n_eff"] = next(iter(resolutions)) if len(resolutions) == 1 else None
        if "per_metric_sampling" in provenance:
            computations = {result.computation or result.name for result in results}
            provenance["per_metric_sampling"] = {
                name: sampling
                for name, sampling in provenance["per_metric_sampling"].items()
                if name in computations
            }
        provenance["selected_metrics"] = [result.name for result in results]
        provenance["metric_execution"] = {
            result.name: report.provenance.get("metric_execution", {}).get(
                result.name,
                procedure.get("metric_execution", {}).get(result.computation or result.name, {}),
            )
            for result in results
        }
        if "uncertainty" in procedure:
            mode = procedure["uncertainty"]
            options = {
                **options,
                "uncertainty": None
                if mode == "none"
                else {"mode": mode, "n_replicates": procedure["n_replicates"]},
            }
    return report_type(
        results=results,
        schema_dict=schema,
        options_dict=options,
        provenance=provenance,
    )
