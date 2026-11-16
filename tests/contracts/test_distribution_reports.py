"""Reports retain point estimates, raw bands and factual diagnostics."""

import json

import numpy as np
import pytest

from tabgen_eval.metrics.base import Direction, StatSpec
from tabgen_eval.report.assembly import assemble_stat
from tabgen_eval.report.diagnostics import DiagnosticFlag
from tabgen_eval.report.io import Report
from tabgen_eval.report.result import MetricResult
from tabgen_eval.statistics.distributions import ScoreDistribution

pytestmark = pytest.mark.unit


def assemble(**kwargs):
    defaults = dict(
        spec=StatSpec("score", "Score", Direction.HIGHER_BETTER),
        value_reps=[0, 1, np.nan, np.inf],
        ideal_reps=[0.02, 0.02],
        value_point=0.7,
        ideal_point=0.6,
        base_flags=[],
        insufficient=False,
        not_applicable=False,
        n_holdout=40,
        n_generated=30,
        n_eff=15,
    )
    defaults.update(kwargs)
    return assemble_stat(**defaults)


def report_for(stat):
    return Report([MetricResult("fixture", "fidelity", {"score": stat})], {}, {})


def test_assembly_reports_finite_means_and_raw_bands():
    stat = assemble()
    assert stat.value == 0.5
    assert stat.ideal == 0.02
    assert stat.value_estimate == stat.ideal_estimate == "mean"
    assert stat.bootstrap_median == stat.generated_distribution.mean == 0.5
    assert (stat.ci_low, stat.ci_high) == pytest.approx((0.025, 0.975))
    assert stat.ideal_bootstrap_median == 0.02
    assert not hasattr(stat, "comparison")
    assert stat.generated_distribution.finite_count == 2
    assert stat.generated_distribution.unavailable_count == 2
    assert DiagnosticFlag.NONFINITE_REPLICATES in stat.flags
    assert not hasattr(stat, "reliable")


def test_zero_width_and_nonconvergence_do_not_suppress_results():
    stat = assemble(value_reps=[0.9] * 10, base_flags=[DiagnosticFlag.CLASSIFIER_NOT_CONVERGED])
    assert stat.ci_low == stat.ci_high == 0.9
    assert stat.generated_distribution.finite_count == 10
    assert DiagnosticFlag.DEGENERATE_CI in stat.flags
    assert not hasattr(stat, "comparison")


def test_reference_only_and_point_only_keep_real_real_point():
    stat = assemble(value_reps=None)
    assert stat.value == 0.7 and stat.ideal == 0.02
    assert np.isnan(stat.ci_low)
    assert stat.reference_distribution.finite_count == 2
    assert not hasattr(stat, "comparison")
    point = assemble(value_reps=None, ideal_reps=None)
    assert point.value == 0.7 and point.ideal == 0.6
    assert not hasattr(point, "comparison")
    assert DiagnosticFlag.NO_CI_REQUESTED in point.flags


def test_all_nonfinite_distribution_is_explicit_not_hidden_or_replaced():
    stat = assemble(value_reps=[np.nan, np.inf], value_point=np.inf)
    assert np.isnan(stat.value)
    assert np.isnan(stat.ci_low)
    assert stat.generated_distribution.reason == "no finite replicate scores"
    assert not hasattr(stat, "comparison")
    report = report_for(stat)
    assert "mean NaN" in report.summary()
    exported = json.loads(report.to_json(include_replicates=True))
    item = exported["results"]["fidelity"]["fixture"]["score"]
    assert item["value"] is None
    assert item["value_status"] == "nan"
    assert item["generated_distribution"]["samples"] == [None, None]
    assert item["generated_distribution"]["discarded_reasons"] == {"nan": 1, "positive_infinity": 1}


def test_structural_nonapplicability_retains_explanation_and_other_scores():
    stat = assemble(not_applicable=True, value_point=np.nan, value_reps=[np.nan, np.nan])
    assert np.isnan(stat.value) and stat.ideal == 0.02
    assert DiagnosticFlag.NOT_APPLICABLE in stat.flags
    assert stat.generated_distribution.finite_count == 0
    assert stat.generated_distribution.requested_count == 2
    assert stat.reference_distribution.finite_count == 2
    report = report_for(stat)
    assert "Score" in report.summary()
    assert "NOT_APPLICABLE" in report.summary()


def test_json_dataframe_markdown_expose_counts_and_unshifted_bands(tmp_path):
    report = report_for(assemble())
    text = report.to_json(include_replicates=True)
    assert "NaN" not in text and "Infinity" not in text
    data = json.loads(text)["results"]["fidelity"]["fixture"]["score"]
    assert data["value"] == 0.5
    assert data["ci_low"] == pytest.approx(0.025)
    assert data["generated_distribution"]["finite_count"] == 2
    assert data["generated_distribution"]["requested_count"] == 4
    for key in ("generated_distribution", "reference_distribution"):
        assert data[key]["confidence_interval"] == 0.95
        assert "alpha" not in data[key]
    assert "reliable" not in data
    frame = report.to_dataframe()
    assert frame.loc[0, "generated_retained"] == 2
    assert frame.loc[0, "generated_discarded"] == 2
    assert "reliable" not in frame
    assert "comparison" not in frame and "comparison_criteria" not in frame
    assert "comparison" not in data
    assert "Comparison" not in report.to_markdown()
    markdown = report.to_markdown()
    assert "2/4 / 2/2" in markdown
    assert "Reliable" not in markdown
    report.to_json(str(tmp_path / "out.json"))
    assert json.loads((tmp_path / "out.json").read_text())["results"]


def test_reference_replicates_export_without_generated_replicates():
    report = report_for(assemble(value_reps=None))
    item = json.loads(report.to_json(include_replicates=True))["results"]["fidelity"]["fixture"][
        "score"
    ]
    assert item["reference_distribution"]["samples"] == [0.02, 0.02]
    assert item["generated_distribution"] is None
    assert "value_replicates" not in item and "ideal_replicates" not in item


def test_readable_views_report_means_without_an_independent_point():
    report = report_for(assemble())
    summary = report.summary()
    assert "mean 0.5000 [0.0250, 0.9750]" in summary
    assert "real-real mean 0.0200 [0.0200, 0.0200]" in summary
    markdown = report.to_markdown()
    assert "Generated score | Generated mean" in markdown
    assert "Real-real score | Reference mean" in markdown
    assert "0.5000 | 0.5000 | [0.0250, 0.9750]" in markdown
    assert "0.0200 | 0.0200 | [0.0200, 0.0200]" in markdown


def test_reference_mean_visible_when_reference_point_is_undefined():
    report = report_for(assemble(ideal_point=np.nan, value_reps=None))
    assert "real-real mean 0.0200" in report.summary()
    assert "point 0.7000, mean" not in report.summary()


def test_fixed_views_do_not_invent_replicate_means():
    report = report_for(assemble(value_reps=None, ideal_reps=None))
    assert ", mean" not in report.summary()
    assert "0.7000 | — | — | 0.6000 | — | —" in report.to_markdown()


@pytest.mark.parametrize("mode", ["none", "reference", "both"])
def test_all_uncertainty_modes_export_raw_scores_without_comparison_labels(mode):
    stat = assemble(
        value_reps=[0.2, 0.8] if mode == "both" else None,
        ideal_reps=[0.4, 0.6] if mode != "none" else None,
    )
    report = report_for(stat)
    assert stat.value == (0.5 if mode == "both" else 0.7)
    assert stat.ideal == (0.5 if mode != "none" else 0.6)
    assert not hasattr(stat, "comparison")
    item = json.loads(report.to_json(include_replicates=True))["results"]["fidelity"]["fixture"][
        "score"
    ]
    assert "comparison" not in item and "comparison_criteria" not in item
    assert (item["generated_distribution"] is not None) == (mode == "both")
    assert (item["reference_distribution"] is not None) == (mode != "none")
    assert "comparison" not in report.to_dataframe().columns
    for label in ("comparable", "deficit_detected", "reference_mismatch_detected", "inconclusive"):
        assert label not in report.summary() and label not in report.to_markdown()


@pytest.mark.parametrize("samples", [[1.0, 3.0, np.nan], [np.nan, np.inf]])
@pytest.mark.parametrize("probabilities", [0.5, [0.1, 0.5, 0.9], [[0.0, 1.0]]])
def test_quantiles_preserve_array_shape_for_finite_and_unavailable_scores(samples, probabilities):
    distribution = ScoreDistribution.from_samples(samples, confidence_interval=0.95)
    actual = distribution.quantiles(probabilities)
    assert isinstance(actual, np.ndarray)
    assert actual.shape == np.asarray(probabilities).shape
    assert actual.dtype == np.float64
    finite = np.asarray(samples)[np.isfinite(samples)]
    expected = np.quantile(finite, probabilities) if finite.size else np.full(actual.shape, np.nan)
    np.testing.assert_allclose(actual, expected, equal_nan=True)
