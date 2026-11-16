"""Family views preserve stored results, metadata and the shared report interface."""

from __future__ import annotations

import importlib
import json

import numpy as np
import pandas as pd
import pytest

from tabgen_eval import Report, Schema
from tabgen_eval.registry.registry import REGISTRY
from tabgen_eval.report import FidelityReport, MemorizationReport, UtilityReport
from tabgen_eval.report.assembly import point_metric_result
from tests.support.evaluation import memorization_report as evaluate_memorization
from tests.support.evaluation import paired_report as evaluate
from tests.support.options import make_options

pytestmark = pytest.mark.unit


@pytest.fixture
def mixed_report():
    results = []
    for spec in REGISTRY:
        metric = spec.metric_cls()
        results.append(
            point_metric_result(
                metric,
                {name: 0.75 for name in metric.stat_names},
                n_holdout=40,
                n_generated=30,
                n_eff=20,
                diagnostics={"fitted": False},
            )
        )
    return Report(
        results,
        schema_dict={"numeric": ["x", "target"]},
        options_dict={"uncertainty": None},
        provenance={
            "sample_resolution": 20,
            "metric_execution": {"shape": {"implementation": "aten_cpp"}},
        },
    )


@pytest.mark.parametrize(
    ("view_name", "report_type", "expected"),
    [
        (
            "fidelity",
            FidelityReport,
            {
                "shape",
                "trend",
                "prdc",
                "conditional_fidelity",
                "concentration",
                "linear_c2st",
                "nonlinear_c2st",
            },
        ),
        (
            "memorization",
            MemorizationReport,
            {"dcr", "near_match_rate", "replication_ratio"},
        ),
        (
            "utility",
            UtilityReport,
            {"mle"},
        ),
    ],
)
def test_family_membership_and_shared_objects(mixed_report, view_name, report_type, expected):
    view = getattr(mixed_report, view_name)
    assert isinstance(view, report_type)
    assert isinstance(view, Report)
    assert {result.name for result in view.results} == expected
    assert view.results == [result for result in mixed_report.results if result.name in expected]
    for result in view.results:
        assert result is next(
            source for source in mixed_report.results if source.name == result.name
        )
        for name, stat in result.stats.items():
            assert view[f"{result.name}/{name}"] is mixed_report[f"{result.name}/{name}"] is stat
    for name in ("schema_dict", "options_dict", "provenance"):
        assert getattr(view, name) is getattr(mixed_report, name)


def test_family_views_partition_every_registered_metric(mixed_report):
    views = [mixed_report.fidelity, mixed_report.memorization, mixed_report.utility]
    names = [result.name for view in views for result in view.results]
    assert len(names) == len(set(names)) == len(REGISTRY)
    assert set(names) == {spec.name for spec in REGISTRY}


@pytest.mark.parametrize("view_name", ["fidelity", "memorization", "utility"])
def test_focused_views_retain_inherited_exports(mixed_report, view_name, tmp_path):
    view = getattr(mixed_report, view_name)
    names = {result.name for result in view.results}
    assert set(view.to_dataframe()["metric"]) == names
    payload = json.loads(view.to_json(str(tmp_path / "report.json")))
    assert payload["provenance"] == mixed_report.provenance
    assert payload["schema"] == mixed_report.schema_dict
    assert payload["options"] == mixed_report.options_dict
    assert {name for category in payload["results"].values() for name in category} == names
    assert all(name in view.to_markdown() for name in names)
    assert "TabGenEval report" in view.summary()


def test_views_do_not_share_the_membership_list(mixed_report):
    view = mixed_report.fidelity
    view.results.pop()
    assert len(mixed_report.results) == len(REGISTRY)
    assert len(mixed_report.fidelity.results) == 7


@pytest.mark.parametrize("view_name", ["fidelity", "memorization", "utility"])
def test_empty_family_views_are_allowed(view_name):
    report = Report([], {}, {}, {"source": "empty"})
    view = getattr(report, view_name)
    assert isinstance(view, Report)
    assert view.results == []
    assert json.loads(view.to_json())["results"] == {}
    assert view.provenance is report.provenance


def test_evaluate_remains_general_and_retains_fidelity_details():
    holdout = pd.DataFrame({"x": np.arange(80.0), "y": np.arange(80.0) ** 2})
    generated = holdout.iloc[::2].copy()
    report = evaluate(
        holdout,
        generated,
        Schema(numeric=["x", "y"]),
        metrics=["shape", "trend", "prdc"],
        details=True,
        options=make_options(device="cpu"),
    )
    assert type(report) is Report
    assert isinstance(report.fidelity, FidelityReport)
    assert report.fidelity.results[0].details is report.results[0].details
    pd.testing.assert_frame_equal(report.fidelity.details("shape"), report.details("shape"))
    assert report.utility.results == []
    assert "prdc" in {result.name for result in report.fidelity.results}


def test_memorization_entry_point_returns_focused_report():
    training = pd.DataFrame({"x": np.arange(12.0)})
    generated = pd.DataFrame({"x": [1.0, 3.2, 5.0, 7.1]})
    report = evaluate_memorization(
        training,
        generated,
        Schema(numeric=["x"]),
        holdout=pd.DataFrame({"x": [0.2, 2.2, 6.2, 8.2]}),
        options=make_options(device="cpu"),
    )
    assert type(report) is MemorizationReport
    assert report.provenance["evaluation_kind"] == "memorization"
    assert report.memorization.results[0] is report.results[0]
    assert report.fidelity.results == report.utility.results == []


def test_mle_view_preserves_unified_evaluation_results():
    from tabgen_eval.metrics import MLE

    result = point_metric_result(
        MLE(),
        {"score": 0.8, "ratio": 1.2, "prediction_agreement": 0.9, "prediction_mse": np.nan},
        n_holdout=40,
        n_generated=30,
        n_eff=20,
    )
    report = Report([result], {}, {}, {"evaluation_kind": "paired"})
    assert type(report.utility) is UtilityReport
    assert report.utility["mle/score"] is report["mle/score"]
    assert report.utility["mle/ratio"] is report["mle/ratio"]
    assert report.utility.results[0] is result
    assert report.fidelity.results == report.memorization.results == []


@pytest.mark.parametrize("report_type", [Report, FidelityReport, MemorizationReport, UtilityReport])
def test_plot_delegates_to_visualization_without_copying(report_type, monkeypatch):
    visualization = importlib.import_module("tabgen_eval.visualization")
    calls = []
    figure = object()

    def plot(report, *, kind, **kwargs):
        calls.append((report, kind, kwargs))
        return figure

    monkeypatch.setattr(visualization, "plot_report", plot)
    report = report_type([], {}, {})
    assert report.plot() is figure
    assert calls == [(report, "scores", {})]
    assert report.plot(kind="columns", metric="shape") is figure
    assert calls[-1] == (report, "columns", {"metric": "shape"})
