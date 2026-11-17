"""Supervised utility, held-out source separation, and paired ratio uncertainty."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.preprocessing import LabelEncoder

from tabgen_eval import Schema, list_metrics
from tabgen_eval._evaluation.estimator import Estimator
from tabgen_eval.execution.runtime.dispatch import supports
from tabgen_eval.metrics import MLE, Direction
from tabgen_eval.models import XGBoostModel
from tabgen_eval.models.errors import FitUnavailable
from tabgen_eval.preprocessing.predictors import predictor_design
from tabgen_eval.schema.validate import ValidationError
from tests.support.evaluation import paired_report as evaluate
from tests.support.metrics import configured_metric
from tests.support.options import make_options

pytestmark = pytest.mark.unit


@pytest.fixture
def xgb():
    try:
        return XGBoostModel.load_backend()
    except ModuleNotFoundError as error:
        if error.name != "xgboost":
            raise
        pytest.skip(str(error))


def data(task="classification", n=320):
    rng = np.random.default_rng(51)
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    if task == "classification":
        y, g = np.where(x > 0, "yes", "no"), np.where(z > 0, "yes", "no")
        schema = Schema(numeric=["x"], categorical=["y"], target="y")
    elif task == "multiclass":
        y = np.select([x < -0.5, x > 0.5], ["low", "high"], default="middle")
        g = np.select([z < -0.5, z > 0.5], ["low", "high"], default="middle")
        schema = Schema(numeric=["x"], categorical=["y"], target="y")
    else:
        y, g = 2 * x + rng.normal(scale=0.1, size=n), 2 * z + rng.normal(scale=0.1, size=n)
        schema = Schema(numeric=["x", "y"], target="y")
    return pd.DataFrame({"x": x, "y": y}), pd.DataFrame({"x": z, "y": g}), schema


def test_utility_is_optional_registered_and_requires_an_official_target():
    entry = next(m for m in list_metrics() if m["name"] == "mle")
    assert entry["category"] == "utility" and entry["enabled_by_default"]
    assert not supports(MLE())
    real, fake, schema = data()
    schema.target = None
    for call in [
        lambda: evaluate(real, fake, schema, metrics=["utility"]),
        lambda: Estimator(MLE(), make_options()).run(real, fake, schema),
        lambda: MLE().compute(real, fake, schema, rng=None),
    ]:
        with pytest.raises(ValidationError, match="explicit Schema.target"):
            call()


def test_task_metadata_uses_roles_and_supports_numeric_class_overrides():
    real, fake, schema = data("regression")
    metric = MLE()
    metric.validate_inputs(real, fake, schema)
    assert metric.stats[0].display == "MLE (MAE)"
    assert metric.stats[1].display == "MLE Ratio (MAE)"
    assert all(s.direction is Direction.LOWER_BETTER for s in metric.stats[:2])
    assert metric.stats[0].bounds == (0.0, np.inf)
    assert metric.stats[2].direction is Direction.HIGHER_BETTER
    assert metric.stats[3].direction is Direction.LOWER_BETTER
    metric = MLE(task="classification")
    metric.validate_inputs(real, fake, schema)
    assert metric.stats[0].display == "MLE (AUROC)"
    assert metric.stats[1].display == "MLE Ratio (AUROC)"
    assert all(s.direction is Direction.HIGHER_BETTER for s in metric.stats[:2])
    assert metric.stats[0].bounds == (0.0, 1.0)
    for task, dtype in [("regression", float), ("classification", bool)]:
        real["y"] = real["y"].astype(dtype)
        metric = MLE()
        metric.validate_inputs(real, real, Schema(numeric=["x"], target="y"))
        assert metric._resolved_task == task
    for kwargs in [{"task": "bad"}, {"fit_threads": 0}, {"memory_limit_mb": 0}]:
        with pytest.raises(ValueError):
            MLE(**kwargs)


@pytest.mark.parametrize("task", ["classification", "multiclass", "regression"])
def test_scores_match_independent_default_xgboost_fits(xgb, task):
    real, generated, schema = data(task, n=480)
    test, baseline = real.iloc[:240], real.iloc[240:]
    generated = generated.iloc[:240]
    outcome = MLE().compute_explicit(
        test, generated, baseline, schema, rng=np.random.default_rng(7)
    )
    scores = []
    for training, key in [(generated, "generated"), (baseline, "reference")]:
        info = outcome.diagnostics["fits"][key]
        is_regression = task == "regression"
        cls = xgb.XGBRegressor if is_regression else xgb.XGBClassifier
        model = cls(
            device="cpu",
            tree_method="hist",
            n_jobs=1,
            random_state=info["random_state"],
            disable_default_eval_metric=True,
        )
        assert model.early_stopping_rounds is None
        if is_regression:
            y, truth = training.y, test.y
        else:
            labels = LabelEncoder().fit(training.y)
            y, truth = labels.transform(training.y), labels.transform(test.y)
        model.fit(training[["x"]].to_numpy(), y)
        if is_regression:
            score = mean_absolute_error(truth, model.predict(test[["x"]].to_numpy()))
        else:
            predictions = model.predict_proba(test[["x"]].to_numpy())
            score = (
                roc_auc_score(truth, predictions[:, 1])
                if task == "classification"
                else roc_auc_score(truth, predictions, multi_class="ovr", average="macro")
            )
        scores.append(score)
        assert info["features"] == 1  # The declared target is excluded from every model.
        assert info["iterations"] == 100 and info["training_completed"]
        assert info["trees"] == (300 if task == "multiclass" else 100)
        assert info["xgboost_version"] == xgb.__version__
    # GPU histogram fits may differ from the independent CPU reference fits.
    tolerance = 1e-2 if task in {"multiclass", "regression"} else 0.0
    assert outcome.values["score"] == pytest.approx(scores[0], abs=tolerance, rel=0)
    assert outcome.reference_scores["score"] == pytest.approx(scores[1], abs=tolerance, rel=0)
    assert outcome.values["ratio"] == pytest.approx(scores[0] / scores[1], abs=tolerance, rel=0)


def test_utility_predictor_encoding_never_fits_scoring_rows_or_target():
    train = pd.DataFrame({"x": [0.0, np.nan, 1.0], "c": ["a", "b", None], "y": [0, 1, 2]})
    test = pd.DataFrame({"x": [1e10, np.nan], "c": ["new", "b"], "y": [999, 999]})
    schema = Schema(numeric=["x", "y"], categorical=["c"], target="y")
    design, scoring = predictor_design(train, test, schema, 2**30)
    altered, _ = predictor_design(train, test.assign(x=-1e20, c="another", y=-999), schema, 2**30)
    np.testing.assert_equal(design, altered)
    assert design.shape == (3, 2) and scoring.shape == (2, 2)
    assert np.isnan(design[1, 0]) and design[0, 0] == 0
    assert scoring[0, 1] == 3 and scoring[1, 1] == design[1, 1]
    with pytest.raises(FitUnavailable, match="memory_limit"):
        predictor_design(train, test, schema, 1)


@pytest.mark.parametrize("generated_target", [0.0, 1.0])
def test_constant_scoring_target_has_valid_mae_and_explicit_zero_reference_ratio(
    xgb, generated_target
):
    real = pd.DataFrame({"x": np.arange(240), "y": np.zeros(240)})
    fake = real.assign(y=generated_target)
    schema = Schema(numeric=["x", "y"], target="y")
    report = evaluate(
        real,
        fake,
        schema,
        metrics=["mle"],
        options=make_options(uncertainty="both", n_replicates=2),
    )
    raw, ratio = report["mle/score"], report["mle/ratio"]
    assert raw.value == generated_target
    assert raw.ideal == 0.0
    np.testing.assert_array_equal(raw.generated_distribution.samples, [generated_target] * 2)
    np.testing.assert_array_equal(raw.reference_distribution.samples, [0.0] * 2)
    assert "NAN_VALUE" not in raw.flag_codes
    diagnostics = report.results[0].diagnostics
    assert diagnostics["point"] == {} and len(diagnostics["replicates"]) == 2
    reasons = [replicate["reason"] for replicate in diagnostics["replicates"]]
    assert np.isnan(ratio.value)
    assert "NAN_VALUE" in ratio.flag_codes
    if generated_target == 0:
        assert reasons == ["zero generated and reference scores"] * 2
        assert np.isnan(ratio.generated_distribution.samples).all()
        assert ratio.generated_distribution.discarded_reasons == {
            "zero generated and reference scores": 2
        }
    else:
        assert reasons == [None] * 2
        assert np.isposinf(ratio.generated_distribution.samples).all()
        assert ratio.generated_distribution.discarded_reasons == {"infinite_ratio": 2}
    assert np.isnan(ratio.ci_low)
    assert ratio.generated_distribution.unavailable_count == 2


@pytest.mark.parametrize("kind", ["missing", "constant", "unknown"])
def test_degenerate_classification_targets_are_flagged(xgb, kind):
    real, fake, schema = data()
    if kind == "missing":
        real["y"] = np.nan
    elif kind == "constant":
        real["y"] = "one"
    else:
        fake["y"] = "unseen"
    report = evaluate(
        real,
        fake,
        schema,
        metrics=["mle"],
        options=make_options(uncertainty="both", n_replicates=2),
    )
    raw = report["mle/score"]
    assert np.isnan(raw.value)
    diag = report.provenance["utility_diagnostics"]["mle"]
    if kind == "missing":
        assert "INSUFFICIENT_DATA" in raw.flag_codes
        assert "CLASSIFIER_NOT_CONVERGED" not in raw.flag_codes
        assert raw.n_eff == 0
        assert diag["row_usage"]["pools"]["test"]["excluded_percent"] == 100.0
        assert all(rep["diagnostics"] == {"insufficient_data": True} for rep in diag["replicates"])
    else:
        assert "CLASSIFIER_NOT_CONVERGED" in raw.flag_codes
        assert all(
            any(not fit["converged"] for fit in rep["diagnostics"]["fits"].values())
            for rep in diag["replicates"]
        )


def test_missing_targets_are_excluded_even_when_predictor_missingness_is_a_category(xgb):
    real, fake, schema = data()
    real.loc[::7, "y"], fake.loc[::9, "y"] = None, None
    real["c"], fake["c"] = np.resize(["a", None], len(real)), np.resize(["b", None], len(fake))
    schema.categorical.append("c")
    report = evaluate(
        real,
        fake,
        schema,
        metrics=["mle"],
        options=make_options(uncertainty="none"),
    )
    assert np.isfinite(report["mle/score"].value)
    diag = report.provenance["utility_diagnostics"]["mle"]["point"]["diagnostics"]
    assert all(n < 160 for n in diag["usable_rows"].values())
    assert all(f["classes"] == ["no", "yes"] for f in diag["fits"].values())


def test_failed_native_fits_are_flagged_and_excluded_from_intervals(xgb, monkeypatch):
    def fail(*args, **kwargs):
        raise xgb.core.XGBoostError("simulated utility failure")

    monkeypatch.setattr(xgb.XGBRegressor, "fit", fail)
    real, fake, schema = data("regression")
    report = evaluate(
        real,
        fake,
        schema,
        metrics=["mle"],
        options=make_options(uncertainty="both", n_replicates=2),
    )
    raw = report["mle/score"]
    assert {"NAN_VALUE", "CLASSIFIER_NOT_CONVERGED"} <= set(raw.flag_codes)
    assert np.isnan(raw.generated_distribution.samples).all()
    assert raw.generated_distribution.unavailable_count == 2
    diag = report.provenance["utility_diagnostics"]["mle"]
    assert all(
        not fit["converged"]
        for rep in diag["replicates"]
        for fit in rep["diagnostics"]["fits"].values()
    )
    assert all(
        f["reason"] == "xgboost_fit_failed" and f["error"] == "simulated utility failure"
        for rep in diag["replicates"]
        for f in rep["diagnostics"]["fits"].values()
    )
    assert diag["point"] == {} and len(diag["replicates"]) == 2


def test_mle_uses_shared_native_predictor_preparation(xgb, monkeypatch):
    from tabgen_eval.preprocessing.columns import PreparedColumns

    calls = []
    original = PreparedColumns.prepare

    def record(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(PreparedColumns, "prepare", record)
    real, fake, schema = data()
    report = evaluate(
        real,
        fake,
        schema,
        metrics=["mle"],
        options=make_options(device="cpu", uncertainty="none"),
    )
    assert np.isfinite(report["mle/score"].value)
    assert calls


def test_utility_auroc_is_not_clipped_at_chance(xgb):
    real, fake, schema = data()
    fake["y"] = np.where(fake.x > 0, "no", "yes")
    report = evaluate(real, fake, schema, metrics=["mle"], options=make_options(uncertainty="none"))
    assert report["mle/score"].value < 0.05
    assert report["mle/ratio"].value < 0.05


def test_target_only_schema_is_structurally_inapplicable():
    frame = pd.DataFrame({"y": np.arange(240)})
    report = evaluate(frame, frame, Schema(numeric=["y"], target="y"), metrics=["mle"])
    assert all("NOT_APPLICABLE" in s.flag_codes for s in report.results[0].stats.values())


def test_missing_utility_engine_raises_actionable_required_dependency_error(monkeypatch):
    from tabgen_eval.models import xgboost as adapter

    def missing(name):
        raise ModuleNotFoundError(name=name)

    monkeypatch.setattr(adapter.importlib, "import_module", missing)
    real, fake, schema = data()
    with pytest.raises(
        ModuleNotFoundError, match=r"XGBoost is a required dependency.*xgboost>=2\.1"
    ):
        MLE().compute(real, fake, schema, rng=None)


def test_mle_accepts_selected_precision_and_device(xgb, monkeypatch, torch, device, dtype):
    from tabgen_eval.models import XGBoostModel

    real, generated, schema = data("regression")
    original = XGBoostModel.fit
    calls = []

    def fit(self, training, labels, **kwargs):
        assert str(training.device) == device and training.dtype == getattr(torch, dtype)
        calls.append(1)
        return original(self, training, labels, **kwargs)

    monkeypatch.setattr(XGBoostModel, "fit", fit)
    report = evaluate(
        real,
        generated,
        schema,
        metrics=["mle"],
        options=make_options(device=device, dtype=dtype, uncertainty="both", n_replicates=2),
    )
    assert len(calls) == 4
    assert np.isfinite(report["mle/score"].value)
    assert report.provenance["metric_execution"]["mle"]["device"] == device


def test_original_mle_preserves_typed_target_categories(xgb):
    labels = pd.Series([1, True, 1.0, "1"] * 60, dtype=object)
    frame = pd.DataFrame({"x": np.tile(np.arange(4), 60), "y": labels})
    result = MLE().compute_explicit(
        frame,
        frame,
        frame,
        Schema(numeric=["x"], categorical=["y"], target="y"),
        rng=None,
    )
    assert np.isfinite(result.values["score"])
    assert all(len(fit["classes"]) == 4 for fit in result.diagnostics["fits"].values())


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_mle_repetition_means_use_raw_paired_bands(xgb, task):
    real, fake, schema = data(task)
    point = evaluate(real, fake, schema, metrics=["mle"], options=make_options(seed=9))
    report = evaluate(
        real,
        fake,
        schema,
        metrics=["mle"],
        options=make_options(
            seed=9,
            uncertainty="both",
            n_replicates=4,
        ),
    )
    for name in ["score", "ratio"]:
        raw = report[f"mle/{name}"]
        samples = raw.generated_distribution.samples
        assert raw.value == pytest.approx(np.mean(samples))
        assert raw.value_estimate == "mean"
        np.testing.assert_equal([raw.ci_low, raw.ci_high], np.quantile(samples, [0.025, 0.975]))
    raw, ratio = report["mle/score"], report["mle/ratio"]
    assert raw.ideal == pytest.approx(np.mean(raw.reference_distribution.samples))
    assert raw.ideal_estimate == "mean"
    assert raw.n_eff == 160 and ratio.target == 1

    np.testing.assert_equal(
        ratio.generated_distribution.samples,
        raw.generated_distribution.samples / raw.reference_distribution.samples,
    )
    assert np.isnan(ratio.ideal) and ratio.reference_distribution is None
    assert point["mle/score"].generated_distribution is None
    exported = json.loads(report.to_json(include_replicates=True))["results"]["utility"]["mle"][
        "score"
    ]
    np.testing.assert_equal(
        exported["generated_distribution"]["samples"], raw.generated_distribution.samples
    )
    assert "interval_method" not in exported and "value_replicates" not in exported
    assert (
        "samples"
        not in json.loads(report.to_json())["results"]["utility"]["mle"]["score"][
            "generated_distribution"
        ]
    )


def test_fresh_pools_have_no_replacement_and_share_shape_sampling(xgb, monkeypatch):
    from tabgen_eval.execution.runtime.sampling import fixed_draw, paired_draw_batches
    from tabgen_eval.models import XGBoostModel

    real, fake, schema = data("regression", n=500)
    real["id"], fake["id"] = np.arange(500), -np.arange(1, 501)
    schema.numeric.append("id")
    original_fit, original_predict = XGBoostModel.fit, XGBoostModel.predict
    calls = []

    def fit(self, train, labels, **kwargs):
        self.test_training_ids = train[:, 1].cpu().numpy().copy()
        return original_fit(self, train, labels, **kwargs)

    def predict(self, test, **kwargs):
        calls.append((self.test_training_ids.copy(), test[:, 1].cpu().numpy().copy()))
        return original_predict(self, test, **kwargs)

    monkeypatch.setattr(XGBoostModel, "fit", fit)
    monkeypatch.setattr(XGBoostModel, "predict", predict)
    cfg = make_options(uncertainty="both", n_replicates=3, seed=17)
    report = evaluate(real, fake, schema, metrics=["shape", "mle"], options=cfg)
    draw = fixed_draw(500, 500, seed=17)
    expected = []
    for batch in paired_draw_batches(
        draw, n_replicates=3, seed=17, batch_size=1, include_generated=True
    ):
        expected.append((batch.reference[0], batch.real_candidate[0], batch.generated[0]))
    assert len(calls) == 6
    for (generated_fit, baseline_fit), (scoring, baseline, candidate) in zip(
        zip(calls[::2], calls[1::2], strict=True),
        expected,
        strict=True,
    ):
        np.testing.assert_equal(generated_fit[0], -candidate - 1)
        np.testing.assert_equal(generated_fit[1], scoring)
        np.testing.assert_equal(baseline_fit[0], baseline)
        np.testing.assert_equal(baseline_fit[1], scoring)
        assert not np.intersect1d(baseline_fit[0], baseline_fit[1]).size
        assert all(len(np.unique(rows)) == 250 for rows in [*generated_fit, *baseline_fit])
    assert not set(calls[3][0]).issubset(set(calls[1][0]))
    assert report["mle/score"].n_eff == report["shape/shape"].n_eff == 250
    assert report.provenance["utility_diagnostics"]["mle"]["sampling"]["replacement"] is False
    alone = evaluate(real, fake, schema, metrics=["shape"], options=cfg)
    np.testing.assert_equal(
        report["shape/shape"].generated_distribution.samples,
        alone["shape/shape"].generated_distribution.samples,
    )


def test_reference_mode_only_refits_baseline_and_matches_both_mode(xgb, monkeypatch):
    from tabgen_eval.models import XGBoostModel

    real, fake, schema = data("regression")
    original = XGBoostModel.fit
    calls = []

    def fit(self, train, labels, **kwargs):
        calls.append(1)
        return original(self, train, labels, **kwargs)

    monkeypatch.setattr(XGBoostModel, "fit", fit)
    reference = evaluate(
        real,
        fake,
        schema,
        metrics=["mle"],
        options=make_options(
            uncertainty="reference",
            n_replicates=3,
        ),
    )
    assert len(calls) == 4  # one generated fit reused for three real-trained comparisons
    both = evaluate(
        real,
        fake,
        schema,
        metrics=["mle"],
        options=make_options(
            uncertainty="both",
            n_replicates=3,
        ),
    )
    for name in ["score", "ratio"]:
        for report in [reference, both]:
            stat = report[f"mle/{name}"]
            assert stat.value == pytest.approx(np.mean(stat.generated_distribution.samples))
    np.testing.assert_equal(
        reference["mle/score"].reference_distribution.samples,
        both["mle/score"].reference_distribution.samples,
    )
    assert reference["mle/ratio"].reference_distribution is None


def test_repetition_prefix_parallelism_and_metric_order_are_deterministic(xgb):
    real, fake, schema = data("regression")
    reports = [
        evaluate(
            real,
            fake,
            schema,
            metrics=metrics,
            options=make_options(
                uncertainty="both",
                n_replicates=count,
                n_jobs=jobs,
                seed=11,
            ),
        )
        for count, jobs, metrics in [
            (2, 1, ["mle"]),
            (4, 2, ["shape", "mle"]),
            (2, 2, ["mle", "shape"]),
        ]
    ]
    for name in ["score", "ratio"]:
        first = reports[0][f"mle/{name}"]
        for report in reports[1:]:
            actual = report[f"mle/{name}"]
            assert actual.value == pytest.approx(np.mean(actual.generated_distribution.samples))
            np.testing.assert_equal(
                actual.generated_distribution.samples[:2], first.generated_distribution.samples
            )
    assert all(
        fit["fit_threads"] == 1
        for rep in reports[1].results[1].diagnostics["replicates"]
        for fit in rep["diagnostics"]["fits"].values()
    )


def test_automatic_resolution_and_minimum_prevent_unusable_fits(monkeypatch):
    from tabgen_eval.models import XGBoostModel

    real, fake, schema = data("regression", n=100)

    def forbidden(*args, **kwargs):
        raise AssertionError("insufficient draws must not fit")

    monkeypatch.setattr(XGBoostModel, "fit", forbidden)
    options = make_options(uncertainty="both", n_replicates=2)
    report = evaluate(
        real,
        fake,
        schema,
        metrics=[configured_metric(MLE, options, n_min=100)],
        options=options,
    )
    assert report["mle/score"].n_eff == 50
    assert "INSUFFICIENT_DATA" in report["mle/score"].flag_codes
    assert report["mle/score"].generated_distribution.unavailable_count == 2


def test_finite_nonconverged_scores_remain_in_distributions(monkeypatch):
    from tabgen_eval.models import XGBoostModel

    monkeypatch.setattr(XGBoostModel, "load_backend", staticmethod(lambda: None))

    def fit(self, train, labels, **kwargs):
        self.diagnostics = {"converged": False, "reason": "iteration_limit"}
        return self

    monkeypatch.setattr(XGBoostModel, "fit", fit)
    monkeypatch.setattr(XGBoostModel, "predict", lambda self, test, **kwargs: np.zeros(len(test)))
    real, fake, schema = data("regression")
    report = evaluate(
        real,
        fake,
        schema,
        metrics=["mle"],
        options=make_options(
            uncertainty="both",
            n_replicates=2,
        ),
    )
    score = report["mle/score"]
    assert np.isfinite(score.value)
    assert "CLASSIFIER_NOT_CONVERGED" in score.flag_codes
    assert score.generated_distribution.finite_count == 2
    assert score.reference_distribution.finite_count == 2
    assert report["mle/ratio"].generated_distribution.finite_count == 2
    np.testing.assert_array_equal(report["mle/ratio"].generated_distribution.samples, [1.0, 1.0])


@pytest.mark.parametrize(
    "generated_probability, reference_probability, expected_scores",
    [
        ([0.5] * 4, [0.5] * 4, [0.5, 0.5]),
        ([0.4, 0.9, 0.1, 0.7], [0.5] * 4, [0.25, 0.5]),
        ([0.5] * 4, [0.4, 0.9, 0.1, 0.7], [0.5, 0.25]),
    ],
)
def test_chance_and_below_chance_auroc_use_raw_ratios(
    monkeypatch, generated_probability, reference_probability, expected_scores
):
    from tabgen_eval.models import XGBoostModel

    frame = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "y": ["a", "a", "b", "b"]})
    fits = []
    monkeypatch.setattr(XGBoostModel, "load_backend", staticmethod(lambda: None))

    def fit(self, train, labels, **kwargs):
        self.ordinal = len(fits)
        fits.append(self)
        self.diagnostics = {"converged": True}
        return self

    def predict(self, test, **kwargs):
        probabilities = generated_probability if self.ordinal == 0 else reference_probability
        values = np.asarray(probabilities)
        return np.column_stack([1 - values, values])

    monkeypatch.setattr(XGBoostModel, "fit", fit)
    monkeypatch.setattr(XGBoostModel, "predict", predict)
    outcome = MLE().compute_explicit(
        frame,
        frame,
        frame,
        Schema(numeric=["x"], categorical=["y"], target="y"),
        rng=np.random.default_rng(17),
    )
    assert len(fits) == 2
    assert outcome.values["score"] == expected_scores[0]
    assert outcome.reference_scores["score"] == expected_scores[1]
    assert outcome.values["ratio"] == expected_scores[0] / expected_scores[1]
