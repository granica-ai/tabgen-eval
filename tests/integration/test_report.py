"""Report views: dataframe / json / markdown / summary / getitem / replicate dump."""

from __future__ import annotations

import json

import pytest

from tests.helpers import make_frame
from tests.support.evaluation import paired_report as evaluate
from tests.support.options import make_options

pytestmark = pytest.mark.unit


@pytest.fixture
def report(schema):
    test = make_frame(400, seed=1)
    syn = make_frame(350, seed=2)
    return evaluate(
        test, syn, schema=schema, options=make_options(uncertainty="both", n_replicates=4, seed=1)
    )


def test_to_dataframe_columns(report):
    df = report.to_dataframe()
    for col in [
        "category",
        "metric",
        "stat",
        "value",
        "ci_low",
        "bootstrap_median",
        "ideal",
        "ideal_bootstrap_median",
        "generated_retained",
        "reference_retained",
        "flags",
    ]:
        assert col in df.columns


def test_to_json_roundtrips(report, tmp_path):
    text = report.to_json()
    data = json.loads(text)
    assert "results" in data and "provenance" in data
    path = tmp_path / "r.json"
    report.to_json(str(path))
    assert json.loads(path.read_text())["results"]


def test_to_markdown_has_table(report):
    md = report.to_markdown()
    assert md.startswith("| Category")
    assert "prdc" in md


def test_bootstrap_median_only_in_detailed_views(report):
    # The bootstrap median is kept as a diagnostic in the detailed df/json views only — the
    # human-facing markdown and console summary do not surface it.
    md = report.to_markdown()
    assert "Median" not in md
    txt = report.summary()
    assert "(med " not in txt
    # ...but it remains available in the dataframe and JSON.
    assert "bootstrap_median" in report.to_dataframe().columns
    assert '"bootstrap_median"' in report.to_json()


def test_summary_omits_allnan_and_has_legend(report):
    from tabgen_eval.report.diagnostics import DiagnosticFlag

    txt = report.summary()
    assert "TabGenEval report" in txt
    assert "Fidelity" in txt
    # The legend must *define* each flag it lists, not merely print the codes.
    legend = txt.split("diagnostics:")[-1]
    used = [f for f in DiagnosticFlag if f.code in txt]
    assert used, "the sample report should surface at least one flag"
    for flag in used:
        assert flag.description in legend, f"{flag.code} listed without its description"


def test_markdown_renders_flag_legend(report):
    from tabgen_eval.report.diagnostics import DiagnosticFlag

    md = report.to_markdown()
    used = [f for f in DiagnosticFlag if f.code in md]
    assert used
    assert "**Flags**" in md
    for flag in used:
        assert f"`{flag.code}` — {flag.description}" in md


def test_getitem_lookup(report):
    s = report["shape/shape"]
    assert s.name == "shape"
    s2 = report["precision"]
    assert s2.name == "precision"


def test_dump_replicates_requires_distribution(schema, tmp_path):
    point = evaluate(
        make_frame(200, seed=1),
        make_frame(200, seed=2),
        schema=schema,
        metrics=["shape"],
        options=make_options(uncertainty="none"),
    )
    with pytest.raises(ValueError, match="distribution"):
        point.dump_replicates(str(tmp_path / "d.json"))


def test_dump_replicates_writes_ordered_distribution_samples(schema, tmp_path):
    test = make_frame(300, seed=1)
    syn = make_frame(300, seed=2)
    rep = evaluate(
        test,
        syn,
        schema=schema,
        options=make_options(uncertainty="both", n_replicates=3),
    )
    path = tmp_path / "d.json"
    rep.dump_replicates(str(path))
    data = json.loads(path.read_text())
    shape = data["results"]["fidelity"]["shape"]["shape"]
    assert len(shape["generated_distribution"]["samples"]) == 3
    assert len(shape["reference_distribution"]["samples"]) == 3


@pytest.fixture
def classifier_lookup_report():
    """Construct distinct classifier results without fitting either detector."""
    from tabgen_eval import MetricResult, Report

    marker = object()
    linear = MetricResult("linear_c2st", "detection", {"auc": marker})
    nonlinear = MetricResult("nonlinear_c2st", "detection", {"auc": object()})
    return Report([linear, nonlinear], {}, {})


def test_getitem_rejects_ambiguous_bare_statistic(classifier_lookup_report):
    with pytest.raises(KeyError, match="Ambiguous stat 'auc'") as error:
        classifier_lookup_report["auc"]
    assert "linear_c2st/auc" in str(error.value)
    assert "nonlinear_c2st/auc" in str(error.value)


@pytest.mark.parametrize("metric_name", ["linear_c2st", "nonlinear_c2st"])
def test_getitem_qualified_statistic_resolves_ambiguity(classifier_lookup_report, metric_name):
    expected = next(
        result.stats["auc"]
        for result in classifier_lookup_report.results
        if result.name == metric_name
    )
    assert classifier_lookup_report[f"{metric_name}/auc"] is expected


def test_getitem_returns_unique_bare_statistic(classifier_lookup_report):
    classifier_lookup_report.results.pop()
    assert classifier_lookup_report["auc"] is classifier_lookup_report.results[0].stats["auc"]


@pytest.mark.parametrize("key", ["missing", "missing/auc", "linear_c2st/missing"])
def test_getitem_missing_statistic_raises(classifier_lookup_report, key):
    with pytest.raises(KeyError, match="No stat matching"):
        classifier_lookup_report[key]


@pytest.mark.parametrize(
    "key",
    [
        "",
        "/",
        "/auc",
        "linear_c2st/",
        "linear_c2st//auc",
        "unrelated/linear_c2st/auc",
        "linear_c2st/auc/auc",
        "linear_c2st/auc/",
    ],
)
def test_getitem_rejects_malformed_statistic_keys(classifier_lookup_report, key):
    with pytest.raises(KeyError, match="Invalid stat key"):
        classifier_lookup_report[key]
