"""MLE prediction comparison reuses paired fits and their shared scoring rows."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error, roc_auc_score

from tabgen_eval import Schema, list_metrics
from tabgen_eval.execution.runtime.sampling import fixed_draw
from tabgen_eval.models import FitUnavailable, XGBoostModel
from tests.support.evaluation import paired_report as evaluate
from tests.support.evaluation import reference_scores as calibrate
from tests.support.evaluation import summaries
from tests.support.options import make_options

pytestmark = pytest.mark.unit


def frames(task):
    rng = np.random.default_rng(12)
    x = np.arange(480, dtype=float)
    y = np.where(x % 3 == 0, "a", "b") if task == "classification" else x * 2 + rng.normal(size=480)
    real = pd.DataFrame({"x": x, "y": y})
    generated = real.iloc[:160].copy()
    generated["x"] += 1000
    schema = Schema(
        numeric=["x"] if task == "classification" else ["x", "y"],
        categorical=["y"] if task == "classification" else [],
        target="y",
    )
    return real, generated, schema


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_mle_report_matches_predictions_without_extra_fits(task, monkeypatch):
    pytest.importorskip("xgboost", minversion="2.1")
    real, generated, schema = frames(task)
    inputs, outputs = [], []
    original_fit, original_predict = XGBoostModel.fit, XGBoostModel.predict

    def fit(self, train, labels, **kwargs):
        assert self.device == "cpu" and str(train.dtype) == "torch.float64"
        inputs.append(train.cpu().numpy().copy())
        return original_fit(self, train, labels, **kwargs)

    def predict(self, test, **kwargs):
        output = original_predict(self, test, **kwargs)
        outputs.append((test.cpu().numpy().copy(), output.copy()))
        return output

    monkeypatch.setattr(XGBoostModel, "fit", fit)
    monkeypatch.setattr(XGBoostModel, "predict", predict)
    options = make_options(device="cpu", dtype="float64", seed=9)
    real_before, gen_before = real.copy(deep=True), generated.copy(deep=True)
    report = evaluate(real, generated, schema, metrics=["mle"], options=options)
    draw = fixed_draw(len(real), len(generated), seed=9)
    assert len(inputs) == len(outputs) == 2
    np.testing.assert_array_equal(inputs[0][:, 0], generated.iloc[draw.generated].x)
    np.testing.assert_array_equal(inputs[1][:, 0], real.iloc[draw.real_candidate].x)
    np.testing.assert_array_equal(outputs[0][0], outputs[1][0])
    assert not set(inputs[1][:, 0]) & set(outputs[0][0][:, 0])
    assert inputs[0].shape[1] == 1
    if task == "classification":
        active, inactive = "prediction_agreement", "prediction_mse"
        expected = np.mean(np.argmax(outputs[0][1], axis=1) == np.argmax(outputs[1][1], axis=1))
    else:
        active, inactive = "prediction_mse", "prediction_agreement"
        expected = np.mean((outputs[0][1] - outputs[1][1]) ** 2)
    truth = real.iloc[draw.reference].y.to_numpy()
    scores = [
        roc_auc_score(truth == "b", prediction[:, 1])
        if task == "classification"
        else mean_absolute_error(truth, prediction)
        for _, prediction in outputs
    ]
    assert report["mle/score"].value == pytest.approx(scores[0])
    assert report["mle/score"].ideal == pytest.approx(scores[1])
    assert report["mle/ratio"].value == pytest.approx(scores[0] / scores[1])
    point = report.results[0].diagnostics["point"]
    assert "generated_skill" not in point and "reference_skill" not in point
    stat = report[f"mle/{active}"]
    assert stat.value == pytest.approx(expected)
    assert stat.generated_distribution is None and stat.reference_distribution is None
    assert np.isnan(stat.ideal)
    assert "NOT_APPLICABLE" in report[f"mle/{inactive}"].flag_codes
    assert set(report.results[0].stats) == {
        "score",
        "ratio",
        "prediction_agreement",
        "prediction_mse",
    }
    assert [result.name for result in report.results] == ["mle"]
    assert report.provenance["metric_execution"]["mle"]["device"] == "cpu"
    assert f'"{active}"' in report.to_json()
    pd.testing.assert_frame_equal(real, real_before)
    pd.testing.assert_frame_equal(generated, gen_before)


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_uncertainty_reuses_existing_model_fits(task, monkeypatch):
    pytest.importorskip("xgboost", minversion="2.1")
    real, generated, schema = frames(task)
    outputs, fits = [], []
    original_fit, original_predict = XGBoostModel.fit, XGBoostModel.predict

    def fit(self, *args, **kwargs):
        fits.append(id(self))
        return original_fit(self, *args, **kwargs)

    def predict(self, test, **kwargs):
        result = original_predict(self, test, **kwargs)
        outputs.append(result.copy())
        return result

    monkeypatch.setattr(XGBoostModel, "fit", fit)
    monkeypatch.setattr(XGBoostModel, "predict", predict)
    active = "prediction_agreement" if task == "classification" else "prediction_mse"
    points = []
    for mode, expected_fits in [("none", 2), ("reference", 4), ("both", 6)]:
        outputs.clear()
        fits.clear()
        report = evaluate(
            real,
            generated,
            schema,
            metrics=["mle"],
            options=make_options(seed=17, uncertainty=mode, n_replicates=2),
        )
        assert len(fits) == expected_fits
        assert len(outputs) == (2 if mode == "none" else 6)
        stat = report[f"mle/{active}"]
        n_res = 80 if mode == "both" else 160
        assert stat.n_eff == n_res
        assert all(len(prediction) == n_res for prediction in outputs)
        expected_point = (
            np.mean(np.argmax(outputs[0], axis=1) == np.argmax(outputs[1], axis=1))
            if task == "classification"
            else np.mean((outputs[0] - outputs[1]) ** 2)
        )
        assert stat.value == pytest.approx(expected_point)
        points.append(stat.value)
        assert stat.reference_distribution is None
        assert not hasattr(stat, "comparison")
        if mode != "none":
            expected = [
                np.mean(np.argmax(outputs[i], axis=1) == np.argmax(outputs[i + 1], axis=1))
                if task == "classification"
                else np.mean((outputs[i] - outputs[i + 1]) ** 2)
                for i in [2, 4]
            ]
            np.testing.assert_allclose(stat.generated_distribution.samples, expected)
        else:
            assert stat.generated_distribution is None
    assert points[0] == points[1]


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_failed_models_do_not_claim_agreement(task, monkeypatch):
    def fail(self, *args, **kwargs):
        raise FitUnavailable("test_failure")

    monkeypatch.setattr(XGBoostModel, "load_backend", staticmethod(lambda: None))
    monkeypatch.setattr(XGBoostModel, "fit", fail)
    real, generated, schema = frames(task)
    report = evaluate(real, generated, schema, metrics=["mle"])
    key = "prediction_agreement" if task == "classification" else "prediction_mse"
    assert np.isnan(report[f"mle/{key}"].value)
    assert "CLASSIFIER_NOT_CONVERGED" in report[f"mle/{key}"].flag_codes


def test_agreement_is_not_a_separate_metric_and_calibration_stays_score_only():
    pytest.importorskip("xgboost", minversion="2.1")
    real, generated, schema = frames("regression")
    assert "prediction_agreement" not in {entry["name"] for entry in list_metrics()}
    with pytest.raises(ValueError, match="Unknown computation"):
        evaluate(real, generated, schema, metrics=["prediction_agreement"])
    calibration = calibrate(
        real,
        schema=schema,
        metrics=["mle"],
        n_replicates=2,
        options=make_options(seed=9),
    )
    assert set(summaries(calibration.metrics["mle"])) == {"score"}
