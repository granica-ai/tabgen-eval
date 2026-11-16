"""Unavailable target scores preserve valid paired predictions and real scores."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from tabgen_eval import Schema
from tabgen_eval.execution.runtime.sampling import fixed_draw
from tabgen_eval.metrics import MLE
from tabgen_eval.models import XGBoostModel
from tests.support.evaluation import paired_report as evaluate
from tests.support.evaluation import reference_scores as calibrate
from tests.support.evaluation import summaries
from tests.support.options import make_options

pytestmark = pytest.mark.unit


@pytest.fixture
def xgb():
    return pytest.importorskip("xgboost", minversion="2.1")


def capture_predictions(monkeypatch):
    """Observe actual model outputs without replacing fitting or prediction."""
    original = XGBoostModel.predict
    predictions = []

    def predict(self, test, **kwargs):
        values = original(self, test, **kwargs)
        predictions.append((test.detach().cpu().numpy().copy(), values.copy()))
        return values

    monkeypatch.setattr(XGBoostModel, "predict", predict)
    return predictions


def test_single_class_generated_training_preserves_valid_real_score(xgb, monkeypatch):
    x = np.tile([-1.0, 1.0], 120)
    real = pd.DataFrame({"x": x, "y": np.where(x < 0, "a", "b")})
    generated = real.iloc[:120].assign(y="a")
    schema = Schema(numeric=["x"], categorical=["y"], target="y")
    seed = 17
    draw = fixed_draw(len(real), len(generated), seed=seed)
    predictions = capture_predictions(monkeypatch)

    report = evaluate(
        real,
        generated,
        schema,
        metrics=["mle"],
        options=make_options(device="cpu", uncertainty="none", seed=seed),
    )

    assert len(predictions) == 1  # The unavailable generated model never predicts.
    scoring, probabilities = predictions[0]
    np.testing.assert_array_equal(scoring[:, 0], real.iloc[draw.reference].x)
    expected = roc_auc_score(real.iloc[draw.reference].y == "b", probabilities[:, 1])
    assert np.isfinite(expected)
    assert report["mle/score"].ideal == pytest.approx(expected)
    for name in ("score", "ratio", "prediction_agreement"):
        assert np.isnan(report[f"mle/{name}"].value)
    fits = report.results[0].diagnostics["point"]["diagnostics"]["fits"]
    assert fits["generated"]["reason"] == "classification_class_support"
    assert fits["generated"]["converged"] is False
    assert fits["reference"]["converged"] is True


def test_single_class_scoring_keeps_same_row_prediction_agreement(xgb, monkeypatch):
    seed = 19
    draw = fixed_draw(240, 120, seed=seed)
    real = pd.DataFrame({"x": np.tile([-1.0, 1.0], 120), "y": "a"})
    # Both training pools have two classes, while all sampled scoring labels are a.
    real.loc[draw.real_candidate, "x"] = np.tile([-1.0, 1.0], 60)
    real.loc[draw.real_candidate, "y"] = np.tile(["a", "b"], 60)
    generated = pd.DataFrame({"x": np.tile([-1.0, 1.0], 60), "y": np.tile(["b", "a"], 60)})
    schema = Schema(numeric=["x"], categorical=["y"], target="y")
    predictions = capture_predictions(monkeypatch)

    report = evaluate(
        real,
        generated,
        schema,
        metrics=["mle"],
        options=make_options(device="cpu", uncertainty="none", seed=seed),
    )

    assert real.iloc[draw.reference].y.nunique() == 1
    assert len(predictions) == 2
    np.testing.assert_array_equal(predictions[0][0], predictions[1][0])
    np.testing.assert_array_equal(predictions[0][0][:, 0], real.iloc[draw.reference].x)
    fits = report.results[0].diagnostics["point"]["diagnostics"]["fits"]
    labels = [
        np.asarray(fits[name]["classes"])[np.argmax(values, axis=1)]
        for name, (_, values) in zip(("generated", "reference"), predictions, strict=True)
    ]
    expected = float(np.mean(labels[0] == labels[1]))
    assert expected == 0.0  # The actual models learned opposite class decisions.
    assert report["mle/prediction_agreement"].value == pytest.approx(expected)
    assert "NAN_VALUE" not in report["mle/prediction_agreement"].flag_codes
    assert np.isnan(report["mle/score"].value)
    assert np.isnan(report["mle/score"].ideal)
    assert np.isnan(report["mle/ratio"].value)
    assert all(fit["reason"] == "classification_class_support" for fit in fits.values())


def test_all_missing_regression_targets_stay_unavailable_without_fitting(xgb, monkeypatch):
    frame = pd.DataFrame({"x": np.arange(48, dtype=float), "y": np.nan})
    schema = Schema(numeric=["x", "y"], target="y")

    def forbidden(*args, **kwargs):
        pytest.fail("all-missing targets must be rejected before model fitting")

    monkeypatch.setattr(XGBoostModel, "fit", forbidden)
    report = evaluate(
        frame,
        frame.copy(),
        schema,
        metrics=["mle"],
        options=make_options(device="cpu", uncertainty="both", n_replicates=2),
    )
    for name in ("score", "ratio", "prediction_mse"):
        stat = report[f"mle/{name}"]
        assert np.isnan(stat.value)
        assert stat.generated_distribution.unavailable_count == 2
    assert report["mle/score"].n_eff == 0
    assert "INSUFFICIENT_DATA" in report["mle/score"].flag_codes
    point = report.results[0].diagnostics["point"]
    assert point["reason"] == "insufficient eligible sampled rows"
    assert point["diagnostics"] == {"insufficient_data": True}
    pools = report.results[0].diagnostics["row_usage"]["pools"]
    for role in ("test", "generated"):
        assert pools[role] == {
            "input": 48,
            "retained": 0,
            "excluded": 48,
            "excluded_percent": 100.0,
        }

    calibration = calibrate(
        frame, schema, metrics=["mle"], n_replicates=2, options=make_options(device="cpu")
    ).metrics["mle"]
    assert summaries(calibration)["score"].unavailable_count == 2
    assert all(diagnostic["insufficient_data"] for diagnostic in calibration.diagnostics)
    assert all("eligible" in diagnostic["reason"] for diagnostic in calibration.diagnostics)
    assert calibration.sampling["n_res"] == 0
    assert calibration.row_usage["pools"]["test"]["excluded_percent"] == 100.0


def test_single_regression_scoring_row_has_valid_mae_and_prediction_mse(xgb, monkeypatch):
    training = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "y": [2.0, 4.0, 6.0, 8.0]})
    generated = training.assign(y=training.y + 1.0)
    scoring = pd.DataFrame({"x": [2.5], "y": [1.0]})
    schema = Schema(numeric=["x", "y"], target="y")
    predictions = capture_predictions(monkeypatch)
    outcome = MLE().compute_explicit(
        scoring, generated, training, schema, rng=np.random.default_rng(17)
    )
    assert len(predictions) == 2
    errors = [float(np.abs(values[0] - scoring.y.iloc[0])) for _, values in predictions]
    assert outcome.values["score"] == errors[0]
    assert outcome.reference_scores["score"] == errors[1]
    assert outcome.values["ratio"] == errors[0] / errors[1]
    expected_mse = float((predictions[0][1][0] - predictions[1][1][0]) ** 2)
    assert outcome.values["prediction_mse"] == pytest.approx(expected_mse)
    assert outcome.diagnostics["usable_rows"]["scoring"] == 1
