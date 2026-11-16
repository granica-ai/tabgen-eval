"""Compiled CPU objective, direct optimizer, and source-fold oracles."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from tabgen_eval import Schema
from tabgen_eval.execution.classifier.protocol import source_folds
from tabgen_eval.execution.runtime.dispatch import context_for
from tabgen_eval.metrics import LinearC2ST
from tests.support.linear_c2st import compute_partitioned as linear_reference

from .oracles import design, fit, independent_loss

pytestmark = pytest.mark.unit


def test_compiled_loss_gradient_and_curvature_match_independent_finite_differences(torch, ops):
    x, y = design(torch)
    w = torch.tensor([0.7, -0.4, 0.2, 0.3, 0.2], dtype=torch.float64)
    loss, gradient, curvature = ops.classifier_derivatives(x, y, w, 0.03)
    xn, yn, wn = x.numpy(), y.numpy(), w.numpy()
    assert loss.item() == pytest.approx(independent_loss(xn, yn, wn, 0.03), abs=1e-14)
    delta = 1e-5 * np.eye(len(w))
    finite_difference = [
        (independent_loss(xn, yn, wn + d, 0.03) - independent_loss(xn, yn, wn - d, 0.03)) / 2e-5
        for d in delta
    ]
    np.testing.assert_allclose(gradient.numpy(), finite_difference, atol=1e-10, rtol=0)
    p = expit(xn @ wn)
    np.testing.assert_allclose(curvature.numpy(), p * (1 - p) / len(y), atol=1e-16, rtol=0)
    z = x @ w
    expected = xn.T @ ((expit(z.numpy()) - yn) / len(y))
    expected[:-1] += 0.03 * wn[:-1]
    np.testing.assert_allclose(gradient.numpy(), expected, atol=1e-14, rtol=0)


def test_direct_matches_independent_optimum_and_recomputes_diagnostics(torch, ops):
    from scipy.optimize import minimize

    x, y = design(torch)
    xn, yn = x.numpy(), y.numpy()
    optimum = minimize(
        lambda w: independent_loss(xn, yn, w, 0.03),
        np.zeros(x.shape[1]),
        method="BFGS",
        options={"gtol": 1e-10},
    )
    fitted = []
    for solver in ["direct"]:
        w, diag, selected, reason = fit(ops, x, y, solver)
        assert selected == solver and diag[0] == 1, reason
        np.testing.assert_allclose(w.numpy(), optimum.x, atol=3e-6, rtol=0)
        loss, gradient, _ = ops.classifier_derivatives(x, y, w, 0.03)
        assert diag[3].item() == pytest.approx(float(gradient @ gradient), abs=1e-20)
        assert loss.item() == pytest.approx(optimum.fun, abs=1e-13)
        fitted.append(w)


def test_direct_tolerance_is_in_gradient_norm_units(torch, ops):
    x = torch.column_stack((torch.zeros(4, dtype=torch.float64), torch.ones(4)))
    y = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64)
    tolerance = 0.1
    # At zero ||g||2=.25 and ||g||2²=.0625. Comparing the squared gradient
    # directly with .1 would falsely accept the zero model without an update.
    weights, diagnostic, selected, reason = ops.classifier_fit(
        x, y, "direct", 0.03, tolerance, 1, 500, 5e10, 1, 4, 1, 31, 1
    )
    assert selected == "direct" and diagnostic[1] == 1
    assert weights[-1] != 0
    assert diagnostic[0] == (diagnostic[3] <= tolerance**2), reason


def test_stable_separated_logits_and_failed_fit_budgets_are_retained(torch, ops):
    x = torch.column_stack([torch.linspace(-1, 1, 100, dtype=torch.float64), torch.ones(100)])
    y = (x[:, 0] > 0).double()
    for solver in ["direct"]:
        w, diag, _, reason = fit(ops, x, y, solver, max_iter=1)
        assert diag[0] == 0 and reason == "max_iter"
        assert diag[1] == 1 and torch.isfinite(w).all()
    w = torch.tensor([1000.0, 0.0], dtype=torch.float64)
    loss, gradient, curvature = ops.classifier_derivatives(x, y, w, 1e-8)
    assert torch.isfinite(loss) and torch.isfinite(gradient).all()
    assert torch.isfinite(curvature).all() and (curvature > 0).any()


@pytest.mark.parametrize("threads", [1, 4])
def test_wide_tiled_hessian_reaches_stationary_solution(torch, ops, threads):
    rng = np.random.default_rng(711)
    values = np.column_stack([rng.normal(size=(315, 260)), np.ones(315)])
    labels = rng.binomial(1, expit(values[:, 0] - values[:, 1]))
    x, y = torch.from_numpy(values), torch.from_numpy(labels.astype(float))
    fitted = []
    for solver in ["direct"]:
        w, diag, _, reason = fit(ops, x, y, solver, threads=threads)
        assert diag[0] == 1, reason
        gradient = values.T @ ((expit(values @ w.numpy()) - labels) / len(labels))
        gradient[:-1] += 0.03 * w.numpy()[:-1]
        assert gradient @ gradient < 1e-14
        fitted.append(w)


@pytest.mark.parametrize("solver", ["direct"])
def test_classifier_fold_preprocessing_and_auc_match_python(torch, solver):
    rng = np.random.default_rng(4)
    tables = []
    for shift in [0.0, 0.8]:
        x = rng.normal(shift, 1, 360)
        table = pd.DataFrame(
            {
                "x": x,
                "y": x + rng.normal(0, 0.3, 360),
                "constant": 1.0,
                "c": rng.choice(["a", "b", "nan"], 360),
            }
        )
        table.loc[::19, "c"] = None
        table.loc[::17, "x"] = np.nan
        tables.append(table)
    r, s = tables
    schema = Schema(numeric=["x", "y", "constant"], categorical=["c"])
    rf, sf = source_folds(len(r), len(s), 3, np.random.default_rng(13))
    metric = LinearC2ST(device="cpu", solver=solver, tolerance=1e-07)
    context = context_for(metric, r, s, schema, n_jobs=4)
    outcome = context.compute_classifier(
        metric,
        np.arange(len(r))[None],
        np.arange(len(s))[None],
        rf[None],
        sf[None],
        [2**64 - 2],
        point=True,
    )[0]
    oracle = linear_reference(metric, r, s, schema, rf, sf, seed=8, point=True)
    assert outcome.converged and oracle.converged
    assert outcome.value == pytest.approx(oracle.value, abs=2e-4)
    assert {d["solver"] for d in outcome.folds} == {solver}


def test_classifier_original_source_folds_reject_conflicting_copies_and_shared_classes(torch):
    r = pd.DataFrame({"x": np.arange(30.0)})
    schema = Schema(numeric=["x"])
    metric = LinearC2ST(device="cpu")
    context = context_for(metric, r, r, schema)
    draw = np.array([[0, 0, 1, 2]])
    fold = np.array([[0, 1, 1, 2]])
    with pytest.raises(ValueError, match="copies"):
        context.compute_classifier(metric, draw, draw, fold, fold, [1])
    draw = np.array([[0, 1, 2]])
    with pytest.raises(ValueError, match="across classes"):
        context.compute_classifier(
            metric, draw, draw, np.array([[0, 1, 2]]), np.array([[1, 0, 2]]), [1], ideal=True
        )


def test_constant_auc_and_noncomputable_design_diagnostics(torch):
    r = pd.DataFrame({"x": np.ones(160)})
    schema = Schema(numeric=["x"])
    metric = LinearC2ST(device="cpu")
    assert metric.compute(r, r, schema, rng=np.random.default_rng(0))["auc"] == 0.5
    missing = r.assign(x=np.nan)
    context = context_for(metric, missing, missing, schema)
    draw = np.arange(len(r))[None]
    folds = (draw % metric.cv_folds).astype(np.int64)
    outcome = context.compute_classifier(metric, draw, draw, folds, folds, [1])[0]
    assert np.isnan(outcome.value)
    diag = outcome.folds
    assert all(
        f["preprocessing"]["dropped_features"]
        == [{"column": "x", "reason": "no_finite_training_values"}]
        for f in diag
    )
    assert all(
        f["reason"] == "preprocessing_or_memory_limit" and f["solver"] == "none" for f in diag
    )


def test_explicit_fit_budget_is_independent_of_point_solver_policy(torch):
    rng = np.random.default_rng(19)
    reference = pd.DataFrame({"x": rng.normal(size=60)})
    candidate = reference + 0.3
    metric = LinearC2ST(device="cpu", solver="direct")
    context = context_for(metric, reference, candidate, Schema(numeric=["x"]), n_jobs=4)
    draw = np.arange(len(reference))[None]
    folds = draw % 3
    outcome = context.compute_classifier(
        metric, draw, draw, folds, folds, [17], point=True, fit_threads=1
    )[0]
    assert np.isfinite(outcome.value)
    assert all(fold["solver"] == "direct" for fold in outcome.folds)
    for budget in [0, 5, True]:
        with pytest.raises(ValueError, match="fit_threads"):
            context.compute_classifier(
                metric, draw, draw, folds, folds, [17], point=True, fit_threads=budget
            )
