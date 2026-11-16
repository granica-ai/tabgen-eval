"""Eligibility precedes sampling without changing unrelated metric populations."""

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from tabgen_eval import EvalData, Schema, evaluate
from tests.support.options import make_options, public_kwargs, replace_options

pytestmark = pytest.mark.unit


@pytest.fixture
def incomplete_data():
    rng = np.random.default_rng(129)

    def frame(size):
        x = rng.normal(size=size)
        return pd.DataFrame(
            {
                "x": x,
                "target": 2 * x + rng.normal(0, 0.1, size),
                "c": rng.choice(["a", "b"], size),
                "ignored": [None] * size,
            }
        )

    train, test, generated = frame(60), frame(100), frame(80)
    train.loc[:7, "x"] = np.nan
    test.loc[:19, "x"] = np.nan
    test.loc[20:29, "target"] = np.nan
    generated.loc[:9, "x"] = np.nan
    generated.loc[10:19, "target"] = np.nan
    return EvalData(
        train=train,
        test=test,
        generated=generated,
        schema=Schema(numeric=["x", "target"], categorical=["c"], target="target"),
    )


def assert_scores_equal(actual, expected, names):
    for name in names:
        a, b = actual[name], expected[name]
        assert a.n_eff == b.n_eff
        np.testing.assert_equal([a.value, a.ideal], [b.value, b.ideal])
        for field in ("generated_distribution", "reference_distribution"):
            first, second = getattr(a, field), getattr(b, field)
            if second is None:
                assert first is None
            else:
                np.testing.assert_array_equal(first.samples, second.samples)


@pytest.mark.parametrize("mode", ["none", "reference", "both"])
def test_distance_prefilter_matches_explicit_complete_inputs(incomplete_data, mode):
    data = incomplete_data
    options = make_options(uncertainty=mode, n_replicates=3, seed=71)
    metrics = ["precision", "concentration"]
    actual = evaluate(data, metrics=metrics, **public_kwargs(options))
    features = data.schema.numeric + data.schema.categorical
    filtered = replace(
        data,
        test=data.test.dropna(subset=features),
        generated=data.generated.dropna(subset=features),
    )
    expected = evaluate(filtered, metrics=metrics, **public_kwargs(options))
    assert_scores_equal(actual, expected, metrics)
    assert actual["precision"].n_eff == actual["concentration"].n_eff == 35
    for result in actual.results:
        usage = result.diagnostics["row_usage"]
        assert usage["pools"]["test"] == {
            "input": 100,
            "retained": 70,
            "excluded": 30,
            "excluded_percent": 30.0,
        }
        assert usage["pools"]["generated"] == {
            "input": 80,
            "retained": 60,
            "excluded": 20,
            "excluded_percent": 25.0,
        }
    table = actual.row_usage()
    assert set(table.pool) == {"test", "generated"}
    assert "excluded" in actual.summary().lower()
    assert "25" in actual.to_markdown()


@pytest.mark.parametrize(
    "generated_size,expected_resolutions", [(80, [50, 35, 45]), (40, [20, 10, 15])]
)
def test_mixed_selection_preserves_each_metric_domain(
    incomplete_data, generated_size, expected_resolutions
):
    pytest.importorskip("xgboost")
    data = replace(incomplete_data, generated=incomplete_data.generated.iloc[:generated_size])
    options = make_options(uncertainty="both", n_replicates=2, seed=71)
    metrics = ["shape", "precision", "mle"]
    combined = evaluate(data, metrics=metrics, **public_kwargs(options))
    for name in metrics:
        single = evaluate(data, metrics=[name], **public_kwargs(options))
        assert_scores_equal(combined, single, [name])
    assert [combined[name].n_eff for name in metrics] == expected_resolutions
    reordered = evaluate(data, metrics=list(reversed(metrics)), **public_kwargs(options))
    assert_scores_equal(combined, reordered, metrics)
    assert combined.results[1].diagnostics["row_usage"]["n_res"] == expected_resolutions[1]
    assert combined.results[2].diagnostics["row_usage"]["n_res"] == expected_resolutions[2]
    assert data.test.x.isna().sum() == 20
    assert data.generated.x.isna().sum() == 10


@pytest.mark.parametrize("mode", ["none", "reference", "both"])
def test_mle_prefilters_only_targets(incomplete_data, mode):
    pytest.importorskip("xgboost")
    data = incomplete_data
    options = make_options(uncertainty=mode, n_replicates=2, seed=71)
    actual = evaluate(data, metrics=["mle"], **public_kwargs(options))
    labelled = replace(
        data,
        test=data.test.dropna(subset=["target"]),
        generated=data.generated.dropna(subset=["target"]),
    )
    assert labelled.test.x.isna().sum() == 20
    assert labelled.generated.x.isna().sum() == 10
    expected = evaluate(labelled, metrics=["mle"], **public_kwargs(options))
    assert_scores_equal(actual, expected, ["mle"])
    assert actual["mle"].n_eff == 45
    usage = actual.results[0].diagnostics["row_usage"]
    assert usage["policy"] == "observed_target"
    assert usage["pools"]["test"]["retained"] == 90
    assert usage["pools"]["generated"]["retained"] == 70


@pytest.mark.parametrize("generated_size", [40, 80])
def test_reference_uncertainty_preserves_unrepeated_fidelity_scores(
    incomplete_data, generated_size
):
    pytest.importorskip("xgboost")
    data = replace(incomplete_data, generated=incomplete_data.generated.iloc[:generated_size])
    options = make_options(seed=71)
    metrics = ["shape", "precision", "mle"]
    fixed = evaluate(data, metrics=metrics, **public_kwargs(options))
    repeated = evaluate(
        data,
        metrics=metrics,
        **public_kwargs(replace_options(options, uncertainty="reference", n_replicates=2)),
    )
    for name in metrics:
        if name != "mle":
            np.testing.assert_equal(fixed[name].value, repeated[name].value)
        else:
            assert repeated[name].value == repeated[name].generated_distribution.mean
        assert repeated[name].ideal == repeated[name].reference_distribution.mean
        assert fixed[name].n_eff == repeated[name].n_eff


def test_both_uncertainty_reports_means_at_the_resolved_resolution(incomplete_data):
    pytest.importorskip("xgboost")
    metrics = ["shape", "precision", "mle"]
    fixed = evaluate(incomplete_data, metrics=metrics, seed=71)
    repeated = evaluate(
        incomplete_data,
        metrics=metrics,
        seed=71,
        uncertainty={"mode": "both", "n_replicates": 2},
    )
    for name in metrics:
        assert repeated[name].value == repeated[name].generated_distribution.mean
        assert repeated[name].ideal == repeated[name].reference_distribution.mean
        assert fixed[name].n_eff == repeated[name].n_eff


def test_both_resolution_uses_eligible_counts_before_choosing_the_branch(incomplete_data):
    data = incomplete_data
    generated = data.generated.copy()
    generated.loc[:49, "x"] = np.nan
    report = evaluate(
        replace(data, generated=generated),
        metrics=["shape", "precision"],
        uncertainty={"mode": "both", "n_replicates": 3},
    )
    # Raw G=80 exceeds H/2=50; complete cases instead give G=30 and H=70.
    assert report["shape"].n_eff == 50
    assert report["precision"].n_eff == 15
    usage = report.results[1].diagnostics["row_usage"]
    assert usage["n_res"] == 15
    assert usage["pools"]["test"]["retained"] == 70
    assert usage["pools"]["generated"]["retained"] == 30
    assert report.provenance["input_rows"] == {"train": 60, "test": 100, "generated": 80}


@pytest.mark.parametrize("mode", ["none", "reference", "both"])
def test_test_only_prefilters_per_metric(incomplete_data, mode):
    data = replace(incomplete_data, train=None, generated=None)
    options = make_options(uncertainty=mode, n_replicates=3, seed=71)
    combined = evaluate(data, metrics=["shape", "precision"], **public_kwargs(options))
    assert combined["shape"].n_eff == 50
    assert combined["precision"].n_eff == 35
    features = data.schema.numeric + data.schema.categorical
    expected = evaluate(
        replace(data, test=data.test.dropna(subset=features)),
        metrics=["precision"],
        **public_kwargs(options),
    )
    assert_scores_equal(combined, expected, ["precision"])


def test_test_only_reference_draws_are_unchanged_by_both_mode(incomplete_data):
    data = replace(incomplete_data, train=None, generated=None)
    metrics = ["shape", "precision"]
    reports = [
        evaluate(
            data,
            metrics=metrics,
            seed=71,
            uncertainty={"mode": mode, "n_replicates": 3},
        )
        for mode in ("reference", "both")
    ]
    assert_scores_equal(reports[0], reports[1], metrics)


def test_public_memorization_stays_point_only_when_comparison_resolution_changes(incomplete_data):
    data = replace(incomplete_data, generated=incomplete_data.generated.iloc[:40])
    metrics = ["shape", "dcr_ratio", "near_match_rate"]
    fixed = evaluate(data, metrics=metrics, seed=71)
    repeated = evaluate(
        data,
        metrics=metrics,
        seed=71,
        uncertainty={"mode": "both", "n_replicates": 3},
    )
    assert fixed["shape"].n_eff == 40 and repeated["shape"].n_eff == 20
    assert_scores_equal(fixed, repeated, ["dcr_ratio", "near_match_rate"])
    for result in repeated.results[1:]:
        for stat in result.stats.values():
            assert stat.generated_distribution is stat.reference_distribution is None
    assert repeated.provenance["procedures"]["memorization"]["uncertainty"] == "none"


def test_exhausted_distance_pool_does_not_abort_fidelity(incomplete_data):
    data = incomplete_data
    generated = data.generated.copy()
    generated["x"] = np.nan
    report = evaluate(
        replace(data, generated=generated),
        metrics=["shape", "precision"],
        uncertainty={"mode": "both", "n_replicates": 2},
    )
    assert np.isfinite(report["shape"].value)
    assert np.isnan(report["precision"].value)
    usage = report.results[1].diagnostics["row_usage"]["pools"]["generated"]
    assert usage["retained"] == 0
    assert usage["excluded_percent"] == 100


def test_memorization_reports_each_involved_pool(incomplete_data):
    report = evaluate(
        incomplete_data,
        metrics=["dcr_ratio", "near_match_rate"],
        uncertainty=None,
    )
    usage = report.results[0].diagnostics["row_usage"]["pools"]
    assert set(usage) == {"train", "test", "generated"}
    assert [usage[role]["retained"] for role in ("train", "test", "generated")] == [52, 70, 60]
    assert "row_usage" in json.loads(report.to_json())


@pytest.mark.gpu
def test_filtered_domains_preserve_cuda_scores_and_source_accounting(incomplete_data, torch):
    from tests.support.backends import cuda_device

    metrics = ["shape", "precision", "concentration", "dcr_ratio"]
    options = make_options(device="cpu", uncertainty="both", n_replicates=3, seed=71)
    cpu = evaluate(incomplete_data, metrics=metrics, **public_kwargs(options))
    gpu = evaluate(
        incomplete_data,
        metrics=metrics,
        **public_kwargs(replace_options(options, device=cuda_device(torch))),
    )
    pd.testing.assert_frame_equal(cpu.row_usage(), gpu.row_usage())
    for result, expected in cpu.iter_stats():
        actual = gpu[f"{result.name}/{expected.name}"]
        assert actual.n_eff == expected.n_eff
        np.testing.assert_allclose(
            [actual.value, actual.ideal],
            [expected.value, expected.ideal],
            rtol=1e-10,
            atol=1e-12,
            equal_nan=True,
        )
        for field in ("generated_distribution", "reference_distribution"):
            first, second = getattr(actual, field), getattr(expected, field)
            if second is None:
                assert first is None
            else:
                np.testing.assert_allclose(
                    first.samples, second.samples, rtol=1e-10, atol=1e-12, equal_nan=True
                )


@pytest.mark.gpu
def test_cuda_mle_uses_observed_targets_and_retains_missing_predictors(incomplete_data, torch):
    pytest.importorskip("xgboost")
    from tests.support.backends import cuda_device

    options = make_options(device=cuda_device(torch), uncertainty="both", n_replicates=2, seed=71)
    actual = evaluate(
        incomplete_data, metrics=["shape", "precision", "mle"], **public_kwargs(options)
    )
    labelled = replace(
        incomplete_data,
        test=incomplete_data.test.dropna(subset=["target"]),
        generated=incomplete_data.generated.dropna(subset=["target"]),
    )
    expected = evaluate(labelled, metrics=["mle"], **public_kwargs(options))
    assert_scores_equal(actual, expected, ["mle"])
    assert actual["mle"].n_eff == 45
    assert actual.provenance["metric_execution"]["mle"]["device"] == cuda_device(torch)
