"""Native-engine defaults, nonlinear power, and the shared fitted C2ST contract."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from tabgen_eval import Schema, list_metrics
from tabgen_eval.execution.classifier.protocol import source_folds
from tabgen_eval.execution.runtime.dispatch import supports
from tabgen_eval.metrics import LinearC2ST, NonLinearC2ST
from tabgen_eval.models.xgboost import XGBoostModel
from tabgen_eval.preprocessing import fit_fold, prepare_sources
from tests.support.evaluation import paired_report as evaluate
from tests.support.metrics import configured_metric
from tests.support.options import make_options

pytestmark = pytest.mark.unit


@pytest.fixture
def xgb():
    return pytest.importorskip("xgboost", minversion="2.1")


def variance_shift(n=600):
    rng = np.random.default_rng(9)
    real = pd.DataFrame({"x": rng.normal(size=n), "y": rng.normal(size=n)})
    fake = pd.DataFrame({"x": rng.normal(scale=3, size=n), "y": rng.normal(scale=3, size=n)})
    return real, fake, Schema(numeric=["x", "y"])


def test_xgboost_is_registered_and_supports_shared_preprocessing_dispatch():
    entry = next(m for m in list_metrics() if m["name"] == "nonlinear_c2st")
    assert entry["category"] == "detection" and entry["enabled_by_default"]
    assert supports(NonLinearC2ST())


@pytest.mark.parametrize("subclass", [False, True])
def test_custom_nonlinear_fit_hook_is_not_bypassed(subclass):
    calls = []

    def fit(train, labels, test, *, seed, point, fit_threads=1):
        calls.append((len(train), len(test), fit_threads))
        return np.zeros(len(test)), {"converged": True}

    if subclass:

        class CustomDetector(NonLinearC2ST):
            _fit_predict = staticmethod(fit)

        metric = CustomDetector()
    else:
        metric = NonLinearC2ST()
        metric._fit_predict = fit
    real, fake, schema = variance_shift(n=30)
    rf, sf = source_folds(len(real), len(fake), 3, np.random.default_rng(7))
    result = metric.compute_partitioned(real, fake, schema, rf, sf, seed=3)
    assert result.value == 0.5
    assert calls == [(40, 20, 1)] * 3


def test_nonlinear_name_controls_selection_reports_and_seed(xgb):
    from tabgen_eval.metrics.c2st import NonLinearC2ST as DirectImport

    assert DirectImport is NonLinearC2ST
    real, fake, schema = variance_shift(n=80)
    options = make_options(uncertainty="none")
    report = evaluate(
        real,
        fake,
        schema,
        metrics=[configured_metric(NonLinearC2ST, options, n_min=20, cv_folds=2)],
        options=options,
    )
    assert np.isfinite(report["nonlinear_c2st/auc"].value)
    diagnostics = report.provenance["classifier_diagnostics"]["nonlinear_c2st"]
    assert len(diagnostics["point"]["generated"]["folds"]) == 2
    assert "nonlinear_c2st" in json.loads(report.to_json())["results"]["detection"]
    with pytest.raises(ValueError, match="Unknown computation"):
        evaluate(real, fake, schema, metrics=["xgboost_c2st"])

    # Numerical randomness remains metric-specific while sampling is shared.
    from tabgen_eval.execution.runtime.paired import comparison_rng

    first = comparison_rng(NonLinearC2ST(), seed=42, ideal=False).integers(2**31)
    renamed = NonLinearC2ST()
    renamed.name = "another_detector"
    second = comparison_rng(renamed, seed=42, ideal=False).integers(2**31)
    assert first != second


def test_missing_engine_has_actionable_install_error(monkeypatch):
    from tabgen_eval.models import xgboost as adapter

    def missing(name):
        raise ModuleNotFoundError(name=name)

    monkeypatch.setattr(adapter.importlib, "import_module", missing)
    with pytest.raises(
        ModuleNotFoundError, match=r"XGBoost is a required dependency.*xgboost>=2\.1"
    ):
        XGBoostModel.load_backend()


def test_broken_engine_dependency_is_not_hidden(monkeypatch):
    from tabgen_eval.models import xgboost as adapter

    def broken(name):
        raise ModuleNotFoundError(name="missing_xgboost_dependency")

    monkeypatch.setattr(adapter.importlib, "import_module", broken)
    with pytest.raises(ModuleNotFoundError) as error:
        XGBoostModel.load_backend()
    assert error.value.name == "missing_xgboost_dependency"


def test_defaults_match_direct_xgbclassifier_and_record_100_trees(xgb):
    real, fake, schema = variance_shift()
    rf, sf = source_folds(len(real), len(fake), 3, np.random.default_rng(7))
    outcome = NonLinearC2ST(device="cpu").compute_partitioned(real, fake, schema, rf, sf, seed=3)
    sources = prepare_sources(real, fake, schema)
    values = []
    for fold, info in enumerate(outcome.folds):
        data = fit_fold(
            sources.reference,
            sources.candidate,
            rf,
            sf,
            fold,
            cardinalities=sources.cardinalities,
            numeric_names=sources.numeric_names,
            categorical_names=sources.categorical_names,
            policy="xgboost",
        )
        model = xgb.XGBClassifier(
            device="cpu",
            tree_method="hist",
            n_jobs=1,
            random_state=3 + fold,
            feature_types=list(data.feature_types),
            enable_categorical=True,
            max_cat_to_onehot=max(data.categorical_cardinalities, default=0) + 1,
        )
        assert model.early_stopping_rounds is None
        model.fit(data.train.numpy(), data.training_labels.numpy())
        values.append(
            max(
                roc_auc_score(
                    data.scoring_labels.numpy(),
                    model.predict(data.test.numpy(), output_margin=True),
                ),
                0.5,
            )
        )
        assert info["trees"] == info["iterations"] == len(model.get_booster().get_dump()) == 100
        learner = info["model_config"]["learner"]
        params = learner["gradient_booster"]["tree_train_param"]
        assert int(params["max_depth"]) == 6
        assert float(params["learning_rate"]) == pytest.approx(0.3)
        assert float(params["reg_lambda"]) == 1.0
        assert learner["generic_param"]["device"] == "cpu"
        assert "rng_state" not in learner["generic_param"]
        assert info["training_completed"] and info["reason"] == "training_budget_completed"
        assert info["xgboost_version"] == xgb.__version__
    assert outcome.converged and outcome.value == np.mean(values)


def test_nonlinear_detector_catches_variance_shift_linear_misses(xgb):
    real, fake, schema = variance_shift(n=1200)
    values = [
        metric.compute(real, fake, schema, rng=np.random.default_rng(7))["auc"]
        for metric in [LinearC2ST(), NonLinearC2ST()]
    ]
    assert values[0] < 0.6 and values[1] > 0.8


def test_xgboost_repetitions_report_means_and_matched_ideal_without_extra_points(xgb):
    real, fake, schema = variance_shift(n=360)
    report = evaluate(
        real,
        fake,
        schema,
        metrics=["nonlinear_c2st"],
        options=make_options(uncertainty="both", n_replicates=3, seed=20),
    )
    stat = report["nonlinear_c2st/auc"]
    assert stat.value_estimate == stat.ideal_estimate == "mean"
    assert stat.n_eff == 180
    diagnostics = report.provenance["classifier_diagnostics"]["nonlinear_c2st"]
    assert not diagnostics["point"]
    for key, distribution, value, low, high in (
        ("generated", stat.generated_distribution, stat.value, stat.ci_low, stat.ci_high),
        (
            "real_reference",
            stat.reference_distribution,
            stat.ideal,
            stat.ideal_ci_low,
            stat.ideal_ci_high,
        ),
    ):
        fits = diagnostics["replicates"][key]
        assert len(fits) == 3 and all(fit["converged"] and fit["point"] for fit in fits)
        assert all(len(fit["folds"]) == 3 for fit in fits)
        np.testing.assert_array_equal([fit["raw_value"] for fit in fits], distribution.samples)
        assert np.isfinite(distribution.samples).all()
        assert value == pytest.approx(np.mean(distribution.samples), rel=0, abs=1e-15)
        np.testing.assert_allclose(
            [low, high], np.quantile(distribution.samples, [0.025, 0.975]), rtol=0, atol=1e-15
        )
        assert 0.5 <= value <= 1.0
        assert all(
            fold["training_rows"] + fold["scoring_rows"] == 2 * stat.n_eff
            for fit in fits
            for fold in fit["folds"]
        )
        assert {f["solver"] for fit in fits for f in fit["folds"]} == {"xgboost-hist"}
    assert json.loads(report.to_json())["provenance"]["classifier_diagnostics"] == {
        "nonlinear_c2st": diagnostics
    }


def test_xgboost_worker_budgets_preserve_scores_and_bound_every_fit(xgb):
    real, fake, schema = variance_shift(n=360)
    reports = [
        evaluate(
            real,
            fake,
            schema,
            metrics=["nonlinear_c2st"],
            options=make_options(
                n_jobs=jobs,
                uncertainty="both",
                n_replicates=3,
                seed=21,
            ),
        )
        for jobs in [1, 2]
    ]
    first, second = [r["nonlinear_c2st/auc"] for r in reports]
    np.testing.assert_equal(
        first.generated_distribution.samples, second.generated_distribution.samples
    )
    np.testing.assert_equal(
        first.reference_distribution.samples, second.reference_distribution.samples
    )
    assert first.value == second.value
    diagnostics = reports[1].provenance["classifier_diagnostics"]["nonlinear_c2st"]
    assert not diagnostics["point"]
    for key in ("generated", "real_reference"):
        fits = diagnostics["replicates"][key]
        assert len(fits) == 3
        assert all(len(fit["folds"]) == 3 for fit in fits)
        assert {f["fit_threads"] for fit in fits for f in fit["folds"]} == {1}
        for fold in [f for fit in fits for f in fit["folds"]]:
            assert (
                int(fold["model_config"]["learner"]["generic_param"]["nthread"])
                == fold["fit_threads"]
            )


def test_frontend_setting_keeps_xgboost_engine_with_shared_preprocessing(xgb, monkeypatch):
    from tabgen_eval.execution.torch.context import TorchContext

    def forbidden(*args, **kwargs):
        raise AssertionError("XGBoost must not load a Torch metric solver")

    monkeypatch.setattr(TorchContext, "_executor", forbidden)
    real, fake, schema = variance_shift(n=240)
    report = evaluate(
        real,
        fake,
        schema,
        metrics=["nonlinear_c2st"],
        options=make_options(device="cpu", uncertainty="none"),
    )
    assert np.isfinite(report["nonlinear_c2st/auc"].value)
    assert {
        name: info["implementation"] for name, info in report.provenance["metric_execution"].items()
    } == {"nonlinear_c2st": "xgboost"}


def test_native_engine_failure_is_flagged_and_failed_bootstraps_are_excluded(xgb, monkeypatch):
    def fail(*args, **kwargs):
        raise xgb.core.XGBoostError("simulated engine failure")

    monkeypatch.setattr(xgb.XGBClassifier, "fit", fail)
    real, fake, schema = variance_shift(n=240)
    report = evaluate(
        real,
        fake,
        schema,
        metrics=["nonlinear_c2st"],
        options=make_options(uncertainty="both", n_replicates=3),
    )
    stat = report["nonlinear_c2st/auc"]
    assert {"NAN_VALUE", "CLASSIFIER_NOT_CONVERGED", "NONFINITE_REPLICATES"} <= set(stat.flag_codes)
    diag = report.provenance["classifier_diagnostics"]["nonlinear_c2st"]
    assert not diag["point"]
    for key, distribution in (
        ("generated", stat.generated_distribution),
        ("real_reference", stat.reference_distribution),
    ):
        assert np.isnan(distribution.samples).all()
        assert distribution.finite_count == 0
        fits = diag["replicates"][key]
        assert len(fits) == 3 and all(not fit["converged"] for fit in fits)
        assert all(len(fit["folds"]) == 3 for fit in fits)
        assert all(
            fold["reason"] == "xgboost_fit_failed" and fold["error"] == "simulated engine failure"
            for fit in fits
            for fold in fit["folds"]
        )


def test_missing_and_unknown_categories_and_constant_features_are_supported(xgb):
    real, fake, schema = variance_shift(n=360)
    real.loc[::11, "x"] = np.nan
    fake.loc[::13, "y"] = np.nan
    real["c"] = np.resize(["a", "b", "nan", None], len(real))
    fake["c"] = np.resize(["a", "b", "unknown", None], len(fake))
    schema.categorical = ["c"]
    assert np.isfinite(
        NonLinearC2ST().compute(real, fake, schema, rng=np.random.default_rng(0))["auc"]
    )
    constant = pd.DataFrame({"x": np.ones(120)})
    assert (
        NonLinearC2ST().compute(constant, constant, Schema(numeric=["x"]), rng=None)["auc"] == 0.5
    )


def test_failed_fold_releases_preprocessing_and_model_buffers_before_next_fold(xgb, monkeypatch):
    import weakref

    import tabgen_eval.preprocessing as preparation
    from tabgen_eval.execution.xgboost.context import XGBoostContext
    from tabgen_eval.models import FitUnavailable

    real, fake, schema = variance_shift(n=30)
    context = XGBoostContext(prepare_sources(real, fake, schema, dtype="float32"))
    folds = (np.arange(len(real)) % 3)[None, :]
    draws = np.arange(len(real))[None, :]
    data_refs, model_refs = [], []
    original = preparation.fit_fold

    def track(*args, **kwargs):
        assert all(reference() is None for reference in data_refs + model_refs)
        data = original(*args, **kwargs)
        data_refs.append(weakref.ref(data))
        return data

    def fail(model, *args, **kwargs):
        model_refs.append(weakref.ref(model))
        raise FitUnavailable("simulated_native_failure")

    monkeypatch.setattr(preparation, "fit_fold", track)
    monkeypatch.setattr(XGBoostModel, "fit_predict", fail)
    outcome = context.compute_classifier(
        NonLinearC2ST(), draws, draws, folds, folds, [42], point=True
    )[0]
    assert np.isnan(outcome.value)
    assert len(outcome.folds) == 3
    assert all(fold["reason"] == "simulated_native_failure" for fold in outcome.folds)
    assert all(reference() is None for reference in data_refs + model_refs)
