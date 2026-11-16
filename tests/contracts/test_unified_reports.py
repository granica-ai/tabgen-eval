"""Unified reports preserve selected scores and distinguish real-only score roles."""

from __future__ import annotations

import json

import numpy as np
import pytest

from tabgen_eval.execution.runtime.details import DetailTable, FidelityDetails
from tabgen_eval.metrics import DCR, MLE, PRDC, Conditional, ReplicationRatio, Shape
from tabgen_eval.registry.selection import SELECTIONS
from tabgen_eval.report import MetricResult, Report
from tabgen_eval.report.assembly import point_metric_result
from tabgen_eval.report.evaluation import project_results, reference_report
from tabgen_eval.uncertainty.results import RealMetricScores, RealReferenceResult

pytestmark = pytest.mark.unit


def selected(*names):
    return tuple(item for item in SELECTIONS if item.name in names)


def source(metric, *, details=None):
    return point_metric_result(
        metric,
        {spec.name: index / 10 for index, spec in enumerate(metric.stats, 1)},
        n_holdout=80,
        n_generated=70,
        n_eff=35,
        diagnostics={"implementation": "fixture"},
        details=details,
    )


def real_only(name="shape", samples=(0.3, 0.7)):
    statistic = "shape" if name == "shape" else "score"
    return RealReferenceResult(
        {
            name: RealMetricScores(
                name,
                {statistic: np.asarray(samples, dtype=np.float64)},
                {"implementation": "fixture"},
                {"task": "classification"} if name == "mle" else {},
                ({"draw": 0},),
                sampling={
                    "n_source": 80,
                    "n_res": 20,
                    "policy": "without_replacement",
                    "replacement": False,
                    "real_roles_disjoint": True,
                },
            )
        },
        {
            "schema": {"numeric": ["x"]},
            "n_source": 80,
            "n_res": 20,
            "n_replicates": len(samples),
            "seed": 17,
        },
    )


def test_projection_retains_scalar_identity_and_public_primary_lookup():
    internal = source(DCR())
    results = project_results([internal], selected("dcr_ratio"))
    report = Report(results, {}, {})
    assert report["dcr_ratio"] is report["dcr_ratio/ratio"] is internal["ratio"]
    assert results[0].computation == "dcr"
    assert results[0].diagnostics is internal.diagnostics
    assert set(results[0].stats) == set(internal.stats)
    exported = json.loads(report.to_json())
    assert exported["schema_version"] == 2
    assert exported["metric_metadata"]["dcr_ratio"]["primary"] == "ratio"
    assert exported["metric_metadata"]["dcr_ratio"]["computation"] == "dcr"
    assert not hasattr(report["dcr_ratio"], "comparison")


def test_prdc_projection_only_exposes_selected_components():
    internal = source(PRDC())
    results = project_results([internal], selected("precision", "coverage"))
    report = Report(results, {}, {})
    assert [result.name for result in results] == ["precision", "coverage"]
    assert report["precision"] is internal["precision"]
    assert report["coverage"] is internal["coverage"]
    assert list(report.to_dataframe()["stat"]) == ["precision", "coverage"]
    assert all(result.category == "geometric_fidelity" for result in report.fidelity.results)
    with pytest.raises(KeyError, match="No stat"):
        report["density"]


def test_conditional_projection_keeps_rare_scores_and_only_relevant_details():
    metric = Conditional()
    details = FidelityDetails(
        {
            "columns": DetailTable(
                {
                    "column_index": np.array([0]),
                    **{spec.name: np.array([0.5]) for spec in metric.stats},
                }
            ),
            "slices": DetailTable(
                {
                    "column_index": np.array([0]),
                    "label_code": np.array([0]),
                    "shape": np.array([0.4]),
                    "trend": np.array([0.7]),
                    "recovery": np.array([0.8]),
                    "eligible": np.array([True]),
                    "rare": np.array([False]),
                    "reference_count": np.array([10]),
                    "candidate_count": np.array([11]),
                }
            ),
        },
        {"column_names": ["category"], "category_values": {0: ["a"]}},
    )
    internal = source(metric, details=details)
    results = project_results([internal], selected("cond_shape", "recovery"))
    report = Report(results, {}, {})
    assert report["cond_shape"] is internal["cond_shape_all"]
    assert set(results[0].stats) == {"cond_shape_all", "cond_shape_rare"}
    assert "cond_trend_all" not in report.details("cond_shape").columns
    assert "recovery" not in report.details("cond_shape", "slice").columns
    assert "shape" not in report.details("recovery", "slice").columns
    assert "trend" in internal.details.tables["slices"].columns
    assert set(json.loads(report.to_json())["details"]) == {"cond_shape", "recovery"}
    assert report.plot(kind="columns", metric="cond_shape").data
    assert report.plot(kind="slices", column="category").data


def test_projection_requires_a_completed_primary_score():
    with pytest.raises(ValueError, match="Missing computation"):
        project_results([], selected("precision"))
    with pytest.raises(ValueError, match="primary statistic"):
        project_results([MetricResult("prdc", "geometric_fidelity")], selected("precision"))


def test_test_only_distributions_do_not_invent_a_generated_score_or_point():
    report = reference_report(
        real_only(), options_dict={"uq": "test"}, point=False, metrics=[Shape()]
    )
    stat = report["shape"]
    assert stat.value == 0.5 and np.isnan(stat.ideal)
    assert stat.point_status is None and stat.ideal_status == "absent"
    assert stat.generated_distribution is None
    assert list(stat.reference_distribution.samples) == [0.3, 0.7]
    payload = json.loads(report.to_json(include_replicates=True))
    item = payload["results"]["fidelity"]["shape"]["shape"]
    assert item["value"] == 0.5 and item["ideal"] is None
    assert item["value_status"] == "finite"
    assert item["ideal_status"] == "absent"
    assert item["value_estimate"] == "mean"
    assert item["score_role"] == "test_test"
    assert item["generated_distribution"] is None
    assert item["reference_distribution"]["samples"] == [0.3, 0.7]
    frame = report.to_dataframe()
    assert frame.loc[0, "n_generated"] == 0
    assert frame.loc[0, "n_test"] == 80
    assert frame.loc[0, "value_status"] == "finite"
    assert "test=80" in report.summary()
    assert "Test–test mean 0.5000" in report.summary()
    assert "Generated" not in report.to_markdown()
    assert "Comparison" not in report.to_markdown()
    assert "95% score bands" in report.summary()
    assert item["reference_distribution"]["confidence_interval"] == 0.95
    assert "comparison" not in item
    assert "point NaN" not in report.summary()


def test_test_only_point_is_the_actual_draw_without_an_uncertainty_band():
    report = reference_report(
        real_only(samples=(0.3,)), options_dict={}, point=True, metrics=[Shape()]
    )
    stat = report["shape"]
    assert stat.value == 0.3
    assert stat.generated_distribution is stat.reference_distribution is None
    assert np.isnan(stat.ideal) and stat.ideal_status == "absent"
    assert "NO_CI_REQUESTED" in stat.flag_codes
    assert "Test–test point 0.3000" in report.summary()
    fig = report.plot()
    assert len(fig.data) == 1 and list(fig.data[0].x) == [0.3]
    assert fig.data[0].name == "Test–test point"
    with pytest.raises(ValueError, match="exactly one draw"):
        reference_report(real_only(), options_dict={}, point=True, metrics=[Shape()])


def test_test_only_undefined_point_differs_from_absent_point():
    report = reference_report(
        real_only(samples=(np.nan,)), options_dict={}, point=True, metrics=[Shape()]
    )
    entry = json.loads(report.to_json())["results"]["fidelity"]["shape"]["shape"]
    assert entry["value_status"] == "nan"
    assert entry["ideal_status"] == "absent"
    assert "NAN_VALUE" in report["shape"].flag_codes
    assert any("undefined" in annotation.text for annotation in report.plot().layout.annotations)


def test_test_only_utility_has_only_real_trained_score():
    report = reference_report(real_only("mle"), options_dict={}, point=False, metrics=[MLE()])
    report.results = project_results(report.results, selected("mle"))
    assert set(report.results[0].stats) == {"score"}
    assert report["mle"].score_role == "real_trained"
    assert report.utility.results[0] is report.results[0]
    for kind in ("scores", "utility"):
        fig = report.plot(kind=kind)
        assert all("Real-trained" in trace.name for trace in fig.data)
        assert all("point" not in trace.name for trace in fig.data)
    assert "Generated" not in report.to_markdown()
    assert "Real-trained mean" in report.summary()


def test_test_only_distribution_plot_shows_the_mean_without_an_extra_draw():
    report = reference_report(real_only(), options_dict={}, point=False, metrics=[Shape()])
    fig = report.plot()
    assert len(fig.data) == 2
    assert list(fig.data[1].x) == [0.5]
    assert fig.data[1].name == "Test–test mean"
    assert fig.data[0].name == "Test–test score band"
    assert list(fig.data[0].x) == pytest.approx([0.31, 0.69])
    assert all("NaN" not in annotation.text for annotation in fig.layout.annotations)


def test_test_only_unavailable_distribution_is_explicit_without_an_invented_point():
    report = reference_report(
        real_only(samples=(np.nan, np.inf)), options_dict={}, point=False, metrics=[Shape()]
    )
    fig = report.plot()
    assert not fig.data
    assert any(
        "Test–test: no finite replicate scores" in annotation.text
        for annotation in fig.layout.annotations
    )
    assert report["shape"].reference_distribution.unavailable_count == 2


def test_mixed_role_summary_and_family_views_keep_role_counts():
    results = project_results([source(Shape()), source(DCR())], selected("shape", "dcr_ratio"))
    report = Report(
        results,
        {},
        {},
        {
            "evaluation_kind": "combined",
            "input_rows": {"train": 100, "test": 80, "generated": 70},
            "n_replicates": 0,
        },
    )
    assert "train=100, test=80, generated=70 rows" in report.summary()
    assert report.to_dataframe()["n_train"].tolist() == [100] * len(report.to_dataframe())
    assert [result.name for result in report.fidelity.results] == ["shape"]
    assert [result.name for result in report.memorization.results] == ["dcr_ratio"]
    assert report.plot().data


def test_memorization_roles_are_preserved_in_every_readable_view():
    roles = {
        "median": ("generated_train", "Generated → train"),
        "holdout_q50": ("test_train", "Test → train"),
        "ratio": ("distance_ratio", "Generated / test distance"),
    }
    result = source(DCR())
    for name, (role, _) in roles.items():
        result[name].score_role = role
        result[name].ideal_status = "absent"
    result.stats = {name: result[name] for name in roles}
    replication = source(ReplicationRatio())
    replication.stats["rate"].score_role = "train_generated"
    replication.stats["rate"].ideal_status = "absent"
    report = Report([result, replication], {}, {})
    labels = [*(label for _, label in roles.values()), "Train → generated"]
    for label in labels:
        assert label in report.summary()
        assert label in report.to_markdown()
        assert any(trace.name == f"{label} point" for trace in report.plot().data)
    assert "Generated point" not in report.to_markdown()
    assert "Comparison" not in report.to_markdown()


def test_mixed_family_views_use_their_own_procedure_schema_and_uncertainty():
    results = project_results([source(Shape()), source(DCR())], selected("shape", "dcr_ratio"))
    comparison_schema = {"numeric": ["score"], "categorical": []}
    memory_schema = {"numeric": [], "categorical": ["score"]}
    report = Report(
        results,
        comparison_schema,
        {"uncertainty": {"mode": "both", "n_replicates": 4}},
        {
            "evaluation_kind": "evaluation",
            "input_rows": {"train": 100, "test": 80, "generated": 70},
            "metric_execution": {
                "shape": {"implementation": "paired"},
                "dcr_ratio": {"implementation": "memory"},
            },
            "n_eff": 35,
            "n_replicates": 4,
            "procedures": {
                "comparison": {
                    "schema": comparison_schema,
                    "n_eff": 35,
                    "n_replicates": 4,
                    "uncertainty": "both",
                    "evaluation_kind": "paired",
                },
                "memorization": {
                    "schema": memory_schema,
                    "n_eff": 17,
                    "n_replicates": 0,
                    "uncertainty": "none",
                    "evaluation_kind": "memorization",
                    "n_selected_training": 20,
                },
            },
        },
    )
    memory = report.memorization
    assert memory.schema_dict is memory_schema
    assert memory.provenance["n_eff"] == 17
    assert memory.provenance["n_replicates"] == 0
    assert memory.provenance["uncertainty"] == "none"
    assert "uq_config" not in memory.provenance
    assert memory.options_dict == {"uncertainty": None}
    assert memory.provenance["metric_execution"] == {"dcr_ratio": {"implementation": "memory"}}
    assert memory["dcr_ratio"] is report["dcr_ratio"]
    assert memory.provenance["input_rows"] is report.provenance["input_rows"]
    assert report.fidelity.schema_dict is comparison_schema
    assert report.fidelity.provenance["n_replicates"] == 4
    assert report.fidelity.options_dict == report.options_dict
    assert report.provenance["n_eff"] == 35
    assert report.options_dict["uncertainty"] == {"mode": "both", "n_replicates": 4}
    assert "uq_config" not in report.provenance
