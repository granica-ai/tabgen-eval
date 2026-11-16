"""Python logistic reference objective, solver choice and honest fit diagnostics."""

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from tabgen_eval import Schema
from tabgen_eval.metrics import LinearC2ST
from tests.support import linear_c2st as classifier
from tests.support.metrics import configured_metric
from tests.support.options import make_options

pytestmark = pytest.mark.unit


def record_models(monkeypatch):
    models = []

    def constructor(**kwargs):
        model = LogisticRegression(**kwargs)
        models.append(model)
        return model

    monkeypatch.setattr(classifier, "LogisticRegression", constructor)
    return models


@pytest.mark.parametrize(
    "solver,columns,work,expected",
    [
        ("auto", 1000, 2e11, "lbfgs"),
        ("direct", 1, 1, "lbfgs"),
        ("sketchysaga", 1000, 2e11, "saga"),
        ("auto", 1, 2e11, "saga"),
        ("auto", 1000, 1, "saga"),
    ],
)
@pytest.mark.parametrize("ridge", [None, 0.025])
def test_python_solver_routing_and_shared_ridge_convention(
    monkeypatch, solver, columns, work, expected, ridge
):
    models = record_models(monkeypatch)
    metric = LinearC2ST(
        solver=solver,
        ridge=ridge,
        max_iter=17,
        saga_max_epochs=23,
        point_direct_max_columns=columns,
        point_direct_max_work=work,
    )
    train = np.zeros((8, 2))
    labels = np.tile([0.0, 1.0], 4)
    _, info = classifier.fit_predict(metric, train, labels, train[:2], seed=42, point=True)
    model = models[0]
    effective_ridge = 1 / len(train) if ridge is None else ridge
    assert model.solver == expected
    assert model.max_iter == (17 if expected == "lbfgs" else 23)
    assert model.C == pytest.approx(1 / (len(train) * effective_ridge))
    assert model.tol == pytest.approx(
        metric.tolerance / np.sqrt(3) if expected == "lbfgs" else metric.tolerance
    )
    assert info["solver"] == f"python-{expected}"
    assert info["effective_ridge"] == effective_ridge
    assert info["gradient_squared"] == (0 if expected == "lbfgs" else None)
    assert info["converged"]


def test_python_gradient_diagnostic_matches_penalized_objective_finite_difference(monkeypatch):
    models = record_models(monkeypatch)
    rng = np.random.default_rng(19)
    train = rng.normal(size=(80, 4))
    labels = rng.binomial(1, expit(train[:, 0] - 0.3)).astype(float)
    metric = LinearC2ST(ridge=0.025, max_iter=200, tolerance=1e-05)
    scores, info = classifier.fit_predict(metric, train, labels, train[:7], seed=42, point=True)
    model = models[0]
    parameters = np.r_[model.coef_[0], model.intercept_[0]]

    def objective(values):
        logits = train @ values[:-1] + values[-1]
        return np.mean(np.logaddexp(0, logits) - labels * logits) + 0.0125 * (
            values[:-1] @ values[:-1]
        )

    gradient = []
    for index in range(len(parameters)):
        direction = np.zeros_like(parameters)
        direction[index] = 1e-5
        gradient.append(
            (objective(parameters + direction) - objective(parameters - direction)) / 2e-5
        )
    expected = np.dot(gradient, gradient)
    assert info["gradient_squared"] == pytest.approx(expected, rel=1e-4, abs=1e-15)
    assert info["converged"]
    np.testing.assert_allclose(scores, train[:7] @ parameters[:-1] + parameters[-1])


def test_reference_preserves_prepared_fp32_values_but_optimizes_in_fp64(monkeypatch):
    observed = []

    class ObservedFit(LogisticRegression):
        def fit(self, train, labels):
            observed.append((self, train.copy(), labels.copy()))
            return super().fit(train, labels)

    monkeypatch.setattr(classifier, "LogisticRegression", ObservedFit)
    rng = np.random.default_rng(190)
    train = rng.normal(size=(160, 6)).astype(np.float32)
    labels = rng.binomial(1, expit(train[:, 0] - 0.3 * train[:, 1])).astype(np.float32)
    test = train[:7].copy()
    original = train.copy(), labels.copy(), test.copy()
    metric = LinearC2ST(solver="direct", ridge=0.025)
    scores, info = classifier.fit_predict(metric, train, labels, test, seed=42, point=True)
    model, fitted_values, fitted_labels = observed[0]
    assert fitted_values.dtype == fitted_labels.dtype == model.coef_.dtype == np.float64
    np.testing.assert_array_equal(fitted_values, original[0].astype(np.float64))
    np.testing.assert_array_equal(fitted_labels, original[1].astype(np.float64))
    residual = expit(fitted_values @ model.coef_[0] + model.intercept_[0]) - fitted_labels
    gradient = np.r_[
        fitted_values.T @ residual / len(train) + metric.ridge * model.coef_[0], residual.mean()
    ]
    assert np.linalg.norm(gradient) <= metric.tolerance
    assert info["converged"] and info["optimizer_dtype"] == "float64"
    assert info["gradient_squared"] == pytest.approx(float(gradient @ gradient), abs=1e-18)
    np.testing.assert_allclose(
        scores, test.astype(np.float64) @ model.coef_[0] + model.intercept_[0]
    )
    for actual, expected in zip((train, labels, test), original, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_warning_remains_nonconverged_even_if_gradient_already_meets_threshold(monkeypatch):
    class WarnedFit(LogisticRegression):
        def fit(self, x, y):
            result = super().fit(x, y)
            warnings.warn("Fit exhausted its budget", ConvergenceWarning, stacklevel=2)
            return result

    monkeypatch.setattr(classifier, "LogisticRegression", WarnedFit)
    train = np.zeros((8, 2))
    metric = LinearC2ST()
    scores, info = classifier.fit_predict(
        metric, train, np.tile([0.0, 1.0], 4), train[:2], seed=42, point=True
    )
    assert np.isfinite(scores).all() and info["gradient_squared"] == 0
    assert not info["converged"] and info["convergence_warning"]
    assert info["reason"] == "sklearn_convergence_warning"


def test_optimizer_success_without_a_warning_cannot_replace_the_gradient_certificate(monkeypatch):
    class ObjectiveStoppedFit(LogisticRegression):
        def fit(self, train, labels):
            result = super().fit(train, labels)
            # Model a successful ftol termination whose coefficients have not
            # reached the independently required gradient accuracy.
            self.coef_.fill(0)
            self.intercept_.fill(0)
            return result

    monkeypatch.setattr(classifier, "LogisticRegression", ObjectiveStoppedFit)
    train = np.linspace(-1, 1, 80)[:, None]
    labels = (train[:, 0] > 0).astype(float)
    metric = LinearC2ST(solver="direct")
    scores, info = classifier.fit_predict(metric, train, labels, train[:7], seed=42, point=True)
    assert np.isfinite(scores).all() and not info["convergence_warning"]
    assert info["gradient_squared"] > metric.tolerance**2
    assert not info["converged"] and info["reason"] == "not_converged"


def test_iteration_limit_retains_scores_and_evaluates_extra_diagnostic_only_after_fit(monkeypatch):
    calls = []

    class ObservedFit(LogisticRegression):
        def fit(self, x, y):
            calls.append("fit_start")
            result = super().fit(x, y)
            calls.append("fit_end")
            return result

        def decision_function(self, x):
            calls.append(("decision", len(x)))
            return super().decision_function(x)

    monkeypatch.setattr(classifier, "LogisticRegression", ObservedFit)
    rng = np.random.default_rng(29)
    train = rng.normal(size=(80, 4))
    labels = (train[:, 0] + 0.3 * train[:, 1] > 0).astype(float)
    metric = LinearC2ST(max_iter=1, tolerance=1e-08)
    scores, info = classifier.fit_predict(metric, train, labels, train[:7], seed=42, point=True)
    assert calls == ["fit_start", "fit_end", ("decision", 80), ("decision", 7)]
    assert np.isfinite(scores).all() and np.isfinite(info["gradient_squared"])
    assert info["iterations"] == 1 and not info["converged"]
    assert info["convergence_warning"]


def test_saga_retains_epoch_budget_without_an_extra_full_gradient_diagnostic(monkeypatch):
    scoring_rows = []

    class ObservedSAGA(LogisticRegression):
        def decision_function(self, x):
            scoring_rows.append(len(x))
            return super().decision_function(x)

    monkeypatch.setattr(classifier, "LogisticRegression", ObservedSAGA)
    rng = np.random.default_rng(29)
    train = rng.normal(size=(80, 4))
    labels = (train[:, 0] > 0).astype(float)
    metric = LinearC2ST(solver="sketchysaga", saga_max_epochs=1)
    scores, info = classifier.fit_predict(metric, train, labels, train[:7], seed=42, point=True)
    assert np.isfinite(scores).all() and scoring_rows == [7]
    assert info["solver"] == "python-saga" and info["epochs_completed"] == 1
    assert not info["converged"] and info["reason"] == "max_epochs"
    assert info["stopping_rule"] == "sklearn_coefficient_change"
    assert info["gradient_squared"] is None and info["full_gradient_evaluations"] == 0


def test_python_preprocessing_budget_does_not_reserve_a_dense_hessian():
    names = [f"x{i}" for i in range(400)]
    real = pd.DataFrame(np.zeros((12, 400)), columns=names)
    folds = np.arange(12) % 3
    metric = LinearC2ST(memory_limit_mb=1, solver="direct")
    outcome = classifier.compute_partitioned(
        metric, real, real, Schema(numeric=names), folds, folds, seed=42, point=True
    )
    assert outcome.value == 0.5 and outcome.converged
    assert all(fold["solver"] == "python-lbfgs" for fold in outcome.folds)
    guarded = classifier.compute_partitioned(
        LinearC2ST(memory_limit_mb=0.01),
        real,
        real,
        Schema(numeric=names),
        folds,
        folds,
        seed=42,
        point=True,
    )
    assert np.isnan(guarded.value)
    assert all(fold["reason"] == "preprocessing_memory_limit" for fold in guarded.folds)


def test_native_evaluation_matches_observation_precision_for_both_comparisons(monkeypatch):
    """The real-real yardstick must not recover precision absent in synthetic data."""
    from tabgen_eval.execution.torch.cpu.executor import CppExecutor
    from tests.support.evaluation import paired_report as evaluate

    observed = []
    original = CppExecutor.compute_classifier

    def record(self, *args, **kwargs):
        for source in (self.context.reference, self.context.candidate):
            values = source.numeric.numpy()
            assert values.dtype == np.float64
            np.testing.assert_array_equal(values, np.ones_like(values))
        observed.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(CppExecutor, "compute_classifier", record)
    real = pd.DataFrame({"x": 1 + np.linspace(-2e-8, 2e-8, 120)})
    generated = pd.DataFrame({"x": np.ones(60, dtype=np.float32)})
    options = make_options(
        device="cpu",
        dtype="float64",
        uncertainty="both",
        n_replicates=2,
        n_jobs=1,
    )
    report = evaluate(
        real,
        generated,
        Schema(numeric=["x"]),
        metrics=[configured_metric(LinearC2ST, options, solver="direct")],
        options=options,
    )
    assert observed
    execution = report.provenance["metric_execution"]["linear_c2st"]
    assert execution["implementation"] == "aten_cpp" and execution["implementation"] == "aten_cpp"
    assert execution["preprocessing"]["observation_bits"] == {"x": 32}
    diagnostics = report.provenance["classifier_diagnostics"]["linear_c2st"]
    folds = []
    for side in ("generated", "real_reference"):
        fits = diagnostics["replicates"][side]
        for fit in fits:
            assert fit["raw_value"] == 0.5
            folds.extend(fit["folds"])
    assert len(folds) == 12


def test_custom_python_linear_metric_keeps_its_partitioned_contract():
    from tabgen_eval.execution.runtime.dispatch import context_for

    class CustomLinear(LinearC2ST):
        pass

    real = pd.DataFrame({"x": np.arange(12, dtype=np.float64)})
    fake = real.astype(np.float32)
    assert context_for(CustomLinear(), real, fake, Schema(numeric=["x"])) is None


@pytest.mark.parametrize("kind", ["linear", "nonlinear"])
def test_class_level_partition_override_retains_python_dispatch(monkeypatch, kind):
    from tabgen_eval.execution.runtime.dispatch import context_for
    from tabgen_eval.metrics import NonLinearC2ST

    metric_type = LinearC2ST if kind == "linear" else NonLinearC2ST

    def partitioned(self, *args, **kwargs):
        raise AssertionError("A context must not bypass the replaced partition method")

    monkeypatch.setattr(metric_type, "compute_partitioned", partitioned)
    real = pd.DataFrame({"x": np.arange(12, dtype=np.float64)})
    assert context_for(metric_type(), real, real, Schema(numeric=["x"])) is None
