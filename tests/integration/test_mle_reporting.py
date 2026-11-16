"""Public MLE scores and ratios retain task-specific meanings in every report."""

import json

import numpy as np
import pandas as pd
import pytest

from tabgen_eval import EvalData, Schema, evaluate
from tabgen_eval.metrics.base import Direction

pytestmark = pytest.mark.unit


@pytest.fixture(params=["classification", "regression"])
def predictive_data(request):
    pytest.importorskip("xgboost", minversion="2.1")
    task = request.param
    rng = np.random.default_rng(137)

    def frame(size, generated=False):
        x = rng.normal(size=size)
        if task == "classification":
            signal = (0.3 if generated else 2.0) * x + rng.normal(size=size)
            target = np.where(signal > 0, "positive", "negative")
        else:
            target = (-0.5 if generated else 2.0) * x + rng.normal(0, 0.4, size)
        return pd.DataFrame({"x": x, "target": target})

    schema = Schema(
        numeric=["x", "target"] if task == "regression" else ["x"],
        categorical=["target"] if task == "classification" else [],
        target="target",
    )
    return task, frame(120), frame(100, generated=True), schema


def assert_score_metadata(stat, task):
    expected = Direction.LOWER_BETTER if task == "regression" else Direction.HIGHER_BETTER
    label = "MAE" if task == "regression" else "AUROC"
    assert stat.direction is expected
    assert stat.display == f"MLE ({label})"


@pytest.mark.parametrize("mode", ["none", "reference", "both"])
def test_public_mle_exposes_raw_scores_and_ratios(predictive_data, mode):
    task, test, generated, schema = predictive_data
    report = evaluate(
        EvalData(test=test, generated=generated, schema=schema),
        metrics=["mle"],
        device="cpu",
        uncertainty=None if mode == "none" else {"mode": mode, "n_replicates": 2},
    )
    score, ratio = report["mle/score"], report["mle/ratio"]
    assert_score_metadata(score, task)
    assert ratio.direction is score.direction
    assert ratio.display == f"MLE Ratio ({'MAE' if task == 'regression' else 'AUROC'})"
    assert score.value >= 0 and score.ideal > 0
    diagnostics = report.results[0].diagnostics
    if mode == "none":
        assert ratio.value == pytest.approx(score.value / score.ideal)
        assert score.generated_distribution is score.reference_distribution is None
        fits = [diagnostics["point"]]
    else:
        generated_scores = score.generated_distribution.samples
        reference_scores = score.reference_distribution.samples
        assert len(generated_scores) == len(reference_scores) == 2
        assert np.isfinite(generated_scores).all() and np.all(reference_scores > 0)
        expected_ratios = generated_scores / reference_scores
        np.testing.assert_allclose(ratio.generated_distribution.samples, expected_ratios)
        assert score.value == pytest.approx(np.mean(generated_scores))
        assert score.ideal == pytest.approx(np.mean(reference_scores))
        assert ratio.value == pytest.approx(np.mean(expected_ratios))
        assert score.value_estimate == score.ideal_estimate == ratio.value_estimate == "mean"
        assert not diagnostics["point"]
        fits = diagnostics["replicates"]
        assert len(fits) == 2
    exported = json.loads(report.to_json())["results"]["utility"]["mle"]
    for name in ("score", "ratio"):
        assert exported[name]["direction"] == score.direction.value
    for fit in fits:
        assert fit["diagnostics"]["scoring"] == ("mae" if task == "regression" else "auroc")
        assert "generated_skill" not in fit and "reference_skill" not in fit


@pytest.mark.parametrize("mode", ["none", "reference", "both"])
def test_test_only_mle_retains_raw_score_direction(predictive_data, mode):
    task, test, _, schema = predictive_data
    report = evaluate(
        EvalData(test=test, schema=schema),
        metrics=["mle"],
        device="cpu",
        uncertainty=None if mode == "none" else {"mode": mode, "n_replicates": 2},
    )
    assert set(report.results[0].stats) == {"score"}
    score = report["mle/score"]
    assert_score_metadata(score, task)
    assert score.score_role == "real_trained"
    if mode == "none":
        assert np.isfinite(score.value) and score.value >= 0
    else:
        assert score.point_status is None
        assert score.value_estimate == "mean"
        assert score.reference_distribution.requested_count == 2
        samples = score.reference_distribution.samples
        assert np.isfinite(samples).all() and np.all(samples >= 0)
        assert score.value == pytest.approx(np.mean(samples))
        np.testing.assert_allclose(
            [score.ideal_ci_low, score.ideal_ci_high], np.quantile(samples, [0.025, 0.975])
        )
    assert np.isnan(score.ideal) and score.ideal_status == "absent"
    exported = json.loads(report.to_json())["results"]["utility"]["mle"]["score"]
    assert exported["direction"] == score.direction.value
