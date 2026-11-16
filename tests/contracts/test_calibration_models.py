"""Real-only fitted model calibration: disjoint roles and native model scores."""

import hashlib

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.preprocessing import LabelEncoder

from tabgen_eval import Schema
from tabgen_eval.execution.classifier.protocol import source_folds
from tabgen_eval.metrics import MLE, NonLinearC2ST
from tabgen_eval.models import XGBoostModel
from tabgen_eval.schema import ValidationError
from tests.support.evaluation import reference_scores as calibrate
from tests.support.evaluation import summaries
from tests.support.metrics import configured_metric
from tests.support.options import make_options
from tests.support.reports import without_memory_plans

pytestmark = pytest.mark.unit


def frames(task="regression", n=360):
    rng = np.random.default_rng(65)
    x = rng.normal(size=n)
    y = (
        np.where(x + rng.normal(scale=0.3, size=n) > 0, "yes", "no")
        if task == "classification"
        else 2 * x + rng.normal(scale=0.2, size=n)
    )
    schema = Schema(
        numeric=["x"] if task == "classification" else ["x", "y"],
        categorical=["y"] if task == "classification" else [],
        target="y",
    )
    return pd.DataFrame({"x": x, "y": y}), schema


def draws(n, seed, count):
    size = n // 2
    rng = np.random.default_rng(seed)
    for _ in range(count):
        indices = rng.choice(n, 2 * size, replace=False)
        yield indices[:size], indices[size:]


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_mle_score_matches_independent_real_trained_xgboost(task):
    xgb = pytest.importorskip("xgboost", minversion="2.1")
    frame, schema = frames(task)
    original = frame.copy(deep=True)
    result = calibrate(
        frame,
        schema,
        metrics=["mle"],
        n_replicates=3,
        seed=23,
        options=make_options(device="cpu"),
    )
    metric = result.metrics["mle"]
    assert set(summaries(metric)) == {"score"}
    assert without_memory_plans(metric.execution) == {
        "implementation": "xgboost",
        "device": "cpu",
        "routing_reason": "explicit_device",
    }
    assert metric.execution["memory_plan"]["policy"] == "available_memory_80_percent"
    scores = []
    for (scoring, training), diagnostic in zip(
        draws(len(frame), 23, 3), metric.diagnostics, strict=True
    ):
        assert set(diagnostic["fits"]) == {"baseline"}
        train, test = frame.iloc[training], frame.iloc[scoring]
        info = diagnostic["fits"]["baseline"]
        cls = xgb.XGBClassifier if task == "classification" else xgb.XGBRegressor
        model = cls(device="cpu", tree_method="hist", n_jobs=1, random_state=info["random_state"])
        if task == "classification":
            encoder = LabelEncoder().fit(train.y)
            model.fit(train[["x"]].to_numpy(), encoder.transform(train.y))
            scores.append(
                roc_auc_score(
                    encoder.transform(test.y), model.predict_proba(test[["x"]].to_numpy())[:, 1]
                )
            )
        else:
            model.fit(train[["x"]].to_numpy(), train.y)
            scores.append(mean_absolute_error(test.y, model.predict(test[["x"]].to_numpy())))
        assert info["training_rows"] == info["scoring_rows"] == len(frame) // 2
        assert diagnostic["scoring"] == ("mae" if task == "regression" else "auroc")
    distribution = summaries(metric)["score"]
    np.testing.assert_allclose(distribution.samples, scores, atol=1e-12)
    np.testing.assert_allclose(
        [distribution.band_low, distribution.band_high], np.quantile(scores, [0.025, 0.975])
    )
    pd.testing.assert_frame_equal(frame, original)


def test_nonlinear_scores_match_independent_partitioned_real_real_comparisons():
    pytest.importorskip("xgboost", minversion="2.1")
    frame, schema = frames()
    result = calibrate(
        frame,
        schema,
        metrics=["nonlinear_c2st"],
        n_replicates=2,
        seed=19,
        options=make_options(device="cpu"),
    )
    metric = NonLinearC2ST(device="cpu")
    expected = []
    metric_id = int.from_bytes(hashlib.sha256(metric.name.encode()).digest()[:4], "little")
    for i, (reference, candidate) in enumerate(draws(len(frame), 19, 2)):
        rng = np.random.default_rng(np.random.SeedSequence(19, spawn_key=(metric_id, i, 0xC2A1)))
        rf, sf = source_folds(len(reference), len(candidate), 3, rng)
        fit_seed = int(rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64))
        outcome = metric.compute_partitioned(
            frame.iloc[reference],
            frame.iloc[candidate],
            schema,
            rf,
            sf,
            seed=fit_seed,
            point=True,
            fit_threads=1,
        )
        expected.append(outcome.value)
    actual = result.metrics[metric.name]
    np.testing.assert_equal(summaries(actual)["auc"].samples, expected)
    assert actual.execution["implementation"] == "xgboost" and actual.execution["device"] == "cpu"
    assert all(len(diagnostic["folds"]) == 3 for diagnostic in actual.diagnostics)


@pytest.mark.parametrize("metric_name, expected_fits", [("mle", 3), ("nonlinear_c2st", 9)])
def test_native_device_precision_and_fit_counts(
    metric_name, expected_fits, monkeypatch, torch, device, dtype
):
    pytest.importorskip("xgboost", minversion="2.1")
    frame, schema = frames()
    calls = []
    original = XGBoostModel.fit

    def fit(self, train, labels, **kwargs):
        assert self.device == device and train.dtype == getattr(torch, dtype)
        calls.append(kwargs["fit_threads"])
        return original(self, train, labels, **kwargs)

    monkeypatch.setattr(XGBoostModel, "fit", fit)
    result = calibrate(
        frame,
        schema,
        metrics=[metric_name],
        n_replicates=3,
        options=make_options(device=device, dtype=dtype, n_jobs=1),
    )
    assert len(calls) == expected_fits and set(calls) == {1}
    assert result.metrics[metric_name].execution["device"] == device


def test_mle_uses_disjoint_full_source_draws_and_training_only_encodings(monkeypatch):
    pytest.importorskip("xgboost", minversion="2.1")
    frame, schema = frames()
    frame["id"] = np.arange(len(frame))
    schema.numeric.append("id")
    calls = []
    original_fit, original_predict = XGBoostModel.fit, XGBoostModel.predict

    def fit(self, train, labels, **kwargs):
        self.test_training_ids = train[:, 1].cpu().numpy().copy()
        return original_fit(self, train, labels, **kwargs)

    def predict(self, test, **kwargs):
        calls.append((self.test_training_ids.copy(), test[:, 1].cpu().numpy().copy()))
        return original_predict(self, test, **kwargs)

    monkeypatch.setattr(XGBoostModel, "fit", fit)
    monkeypatch.setattr(XGBoostModel, "predict", predict)
    calibrate(
        frame,
        schema,
        metrics=["mle"],
        n_replicates=3,
        seed=27,
        options=make_options(n_jobs=1),
    )
    assert len(calls) == 3
    for (train, scoring), (r, t) in zip(calls, draws(len(frame), 27, 3), strict=True):
        np.testing.assert_equal(train, t)
        np.testing.assert_equal(scoring, r)
        assert not np.intersect1d(train, scoring).size
        assert len(np.unique(train)) == len(np.unique(scoring)) == len(frame) // 2
    assert not set(calls[1][0]).issubset(set(calls[0][0]))


@pytest.mark.parametrize("name, stat", [("mle", "score"), ("nonlinear_c2st", "auc")])
def test_parallelism_metric_order_and_repetition_prefix_are_invariant(name, stat):
    pytest.importorskip("xgboost", minversion="2.1")
    frame, schema = frames()

    def run(metrics, count, jobs):
        return calibrate(
            frame,
            schema,
            metrics=metrics,
            n_replicates=count,
            seed=31,
            options=make_options(n_jobs=jobs),
        )

    alone = run([name], 2, 1)
    mixed = run(["shape", "nonlinear_c2st", "mle"], 4, 2)
    reversed_order = run(["mle", "nonlinear_c2st", "shape"], 2, 1)
    expected = summaries(alone.metrics[name])[stat].samples
    np.testing.assert_equal(summaries(mixed.metrics[name])[stat].samples[:2], expected)
    np.testing.assert_equal(summaries(reversed_order.metrics[name])[stat].samples, expected)
    assert all(
        fit["fit_threads"] == 1
        for diagnostic in mixed.metrics["mle"].diagnostics
        for fit in diagnostic["fits"].values()
    )


def test_mle_requires_target_and_preserves_minimum_and_inapplicability(monkeypatch):
    frame, schema = frames()
    with pytest.raises(ValidationError, match="explicit Schema.target"):
        calibrate(frame, Schema(numeric=["x", "y"]), metrics=["mle"])

    def forbidden(*args, **kwargs):
        raise AssertionError("unavailable comparisons must not fit")

    monkeypatch.setattr(XGBoostModel, "fit", forbidden)
    options = make_options()
    tiny = calibrate(
        frame.iloc[:100],
        schema,
        n_replicates=2,
        metrics=[configured_metric(MLE, options, n_min=100)],
        options=options,
    )
    assert summaries(tiny.metrics["mle"])["score"].unavailable_count == 2
    no_features = calibrate(
        frame[["y"]], Schema(numeric=["y"], target="y"), metrics=["mle"], n_replicates=2
    )
    assert summaries(no_features.metrics["mle"])["score"].unavailable_count == 2
    assert all(d["not_applicable"] for d in no_features.metrics["mle"].diagnostics)


def test_mle_missing_targets_and_failed_fits_remain_diagnosable(monkeypatch):
    xgb = pytest.importorskip("xgboost", minversion="2.1")
    frame, schema = frames("classification")
    frame.loc[::9, "y"] = None
    result = calibrate(
        frame,
        schema,
        metrics=["mle"],
        n_replicates=2,
        options=make_options(),
    )
    assert summaries(result.metrics["mle"])["score"].finite_count == 2
    assert all(
        d["usable_rows"]["scoring"] < len(frame) // 2 for d in result.metrics["mle"].diagnostics
    )

    def fail(*args, **kwargs):
        raise xgb.core.XGBoostError("calibration fit failed")

    monkeypatch.setattr(xgb.XGBClassifier, "fit", fail)
    failed = calibrate(
        frame, schema, metrics=["mle"], n_replicates=2, options=make_options(n_jobs=1)
    )
    assert summaries(failed.metrics["mle"])["score"].unavailable_count == 2
    assert all(
        d["fits"]["baseline"]["reason"] == "xgboost_fit_failed"
        for d in failed.metrics["mle"].diagnostics
    )


@pytest.mark.parametrize("metric_name", ["nonlinear_c2st", "mle"])
def test_missing_required_engine_has_actionable_error(metric_name, monkeypatch):
    from tabgen_eval.models import xgboost as adapter

    original = adapter.importlib.import_module

    def missing(name):
        if name == "xgboost":
            raise ModuleNotFoundError(name=name)
        return original(name)

    monkeypatch.setattr(adapter.importlib, "import_module", missing)
    frame, schema = frames()
    with pytest.raises(
        ModuleNotFoundError, match=r"XGBoost is a required dependency.*xgboost>=2\.1"
    ):
        calibrate(frame, schema, metrics=[metric_name], n_replicates=1)


@pytest.mark.parametrize("constant_target", [None, 0.0, 2.0])
@pytest.mark.parametrize("size", [2, 360])
def test_raw_mae_preserves_nonconvergence_single_rows_and_constant_targets(
    monkeypatch, constant_target, size
):
    monkeypatch.setattr(XGBoostModel, "load_backend", staticmethod(lambda: None))

    def fit(self, train, labels, **kwargs):
        self.diagnostics = {"converged": False, "reason": "iteration_limit"}
        return self

    monkeypatch.setattr(XGBoostModel, "fit", fit)
    monkeypatch.setattr(XGBoostModel, "predict", lambda self, test, **kwargs: np.zeros(len(test)))
    frame, schema = frames(n=size)
    if constant_target is not None:
        frame["y"] = constant_target
    result = calibrate(
        frame,
        schema,
        metrics=["mle"],
        n_replicates=3,
        seed=14,
        options=make_options(device="cpu", n_jobs=1),
    )
    score = summaries(result.metrics["mle"])["score"]
    expected = [np.mean(np.abs(frame.iloc[scoring].y)) for scoring, _ in draws(size, 14, 3)]
    np.testing.assert_allclose(score.samples, expected, atol=1e-14)
    assert score.finite_count == 3
    assert set(summaries(result.metrics["mle"])) == {"score"}
    assert all(not d["fits"]["baseline"]["converged"] for d in result.metrics["mle"].diagnostics)
