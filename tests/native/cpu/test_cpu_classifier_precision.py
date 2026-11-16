"""Selected-precision CPU logistic objectives, fits, and compact storage."""

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import roc_auc_score

from tabgen_eval import Schema
from tabgen_eval.execution.classifier.protocol import source_folds
from tabgen_eval.execution.runtime.dispatch import context_for
from tabgen_eval.metrics import LinearC2ST

from .oracles import independent_loss
from .test_cpu_saga import expanded, native_classifier

pytestmark = pytest.mark.unit


def problem(n=1536, p=24):
    rng = np.random.default_rng(609)
    numbers = rng.normal(size=(n, p))
    # Correlated and low-variance coordinates exercise scaling/conditioning.
    numbers[:, -1] = numbers[:, 0] + 1e-4 * numbers[:, -1]
    numbers[:, -2] *= 1e-3
    x = np.column_stack([numbers, np.ones(n)])
    labels = rng.binomial(1, expit(x[:, 0] - 0.8 * x[:, 1] + 0.25))
    return x, labels.astype(float)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_selected_precision_objective_and_newton_match_independent_optimum(torch, ops, dtype):
    x, labels = problem(n=384, p=12)
    tensors = [torch.from_numpy(a.astype(dtype)) for a in (x, labels)]
    data, y = tensors
    initial = data.new_tensor(np.linspace(-0.2, 0.3, data.shape[1]))
    ridge = 1 / len(x)
    loss, gradient, curvature = ops.classifier_derivatives(data, y, initial, ridge)
    xn, yn, w = [a.numpy().astype(float) for a in (data, y, initial)]
    residual = expit(xn @ w) - yn
    expected = xn.T @ residual / len(xn)
    expected[:-1] += ridge * w[:-1]
    tol = 2e-7 if dtype == "float32" else 1e-14
    assert gradient.dtype == curvature.dtype == data.dtype
    assert loss.item() == pytest.approx(independent_loss(xn, yn, w, ridge), abs=tol)
    np.testing.assert_allclose(gradient, expected, atol=tol, rtol=0)
    coefficients, diagnostics, selected, reason = ops.classifier_fit(
        data, y, "direct", ridge, 1e-5, 100, 512, 2e11, 500, 4096, 8, 31, 4
    )
    assert coefficients.dtype == data.dtype
    assert selected == "direct" and diagnostics[0] == 1, reason
    reference = minimize(
        lambda z: independent_loss(xn, yn, z, ridge),
        np.zeros(x.shape[1]),
        method="BFGS",
        options={"gtol": 1e-9},
    )
    np.testing.assert_allclose(coefficients.numpy(), reference.x, atol=3e-4, rtol=0)
    assert independent_loss(xn, yn, coefficients.numpy(), ridge) == pytest.approx(
        reference.fun, abs=2e-8
    )
    assert roc_auc_score(yn, xn @ coefficients.numpy()) == pytest.approx(
        roc_auc_score(yn, xn @ reference.x), abs=1e-5
    )


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_compact_saga_preserves_precision_and_predictive_quality(torch, ops, dtype):
    x, labels = problem()
    data = torch.from_numpy(x[:, :-1].astype(dtype).copy())
    y = torch.from_numpy(labels.astype(dtype))
    ids = torch.empty((len(x), 0), dtype=torch.int64)
    empty = data.new_empty(0)
    before = torch.random.get_rng_state().clone()
    coefficients, diagnostics, reason = ops.classifier_saga_fit(
        data, ids, empty, empty, y, 1 / len(x), 1e-4, 500, 512, 12, 31, 10**8, 4
    )
    assert coefficients.dtype == data.dtype and diagnostics.dtype == torch.float64
    assert diagnostics[0] == 1 and reason == "online_gradient"
    assert diagnostics[8] <= diagnostics[10]
    assert diagnostics[25] == len(x)  # Moderate data preparation uses all rows.
    torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)
    optimum = minimize(
        lambda w: independent_loss(x, labels, w, 1 / len(x)),
        np.zeros(x.shape[1]),
        method="BFGS",
        options={"gtol": 1e-8},
    )
    assert independent_loss(x, labels, coefficients.numpy(), 1 / len(x)) - optimum.fun < 1e-4
    assert roc_auc_score(labels, x @ coefficients.numpy()) == pytest.approx(
        roc_auc_score(labels, x @ optimum.x), abs=5e-4
    )
    with pytest.raises(RuntimeError, match="memory"):
        ops.classifier_saga_fit(data, ids, empty, empty, y, 1 / len(x), 1e-4, 1, 512, 12, 31, 1, 4)


def test_fp32_compact_categorical_preparation_and_scoring_remain_fp32(torch, ops):
    rng = np.random.default_rng(610)
    real = torch.from_numpy(rng.normal(size=(180, 3)).astype("float32"))
    candidate = real + 0.2
    ids = torch.tensor(rng.integers(0, 15, size=(180, 1)), dtype=torch.int64)
    folds = torch.arange(180, dtype=torch.int64) % 3
    buffers = ops.classifier_saga_prepare(real, ids, candidate, ids, folds, folds, 0, 10**7)
    numbers, identities, test_numbers, test_ids, centers, scales, labels, scoring = buffers
    for buffer in [numbers, test_numbers, centers, scales, labels, scoring]:
        assert buffer.dtype == torch.float32
    coefficients = torch.linspace(-1, 1, numbers.shape[1] + centers.numel() + 1)
    prediction = ops.classifier_saga_mv(test_numbers, test_ids, centers, scales, coefficients)
    assert prediction.dtype == torch.float32
    dense = expanded(test_numbers.numpy(), test_ids.numpy(), centers.numpy(), scales.numpy())
    np.testing.assert_allclose(prediction, dense @ coefficients.numpy(), rtol=1e-6, atol=1e-6)
    assert identities.dtype == torch.int64


@pytest.mark.parametrize("solver", ["direct", "sketchysaga"])
def test_public_cpu_fp32_route_matches_fp64_auc(torch, solver):
    rng = np.random.default_rng(611)
    reference = pd.DataFrame(rng.normal(size=(720, 8)), columns=list("abcdefgh"))
    candidate = reference + rng.normal(0.25, 0.2, size=reference.shape)
    reference["category"] = rng.choice(["a", "b", "c"], len(reference))
    candidate["category"] = rng.choice(["a", "b", "c"], len(candidate))
    schema = Schema(numeric=list("abcdefgh"), categorical=["category"])
    folds = source_folds(len(reference), len(candidate), 3, np.random.default_rng(31))
    outcomes = []
    for dtype in ["float32", "float64"]:
        metric = LinearC2ST(device="cpu", dtype=dtype, solver=solver)
        context = context_for(metric, reference, candidate, schema, n_jobs=4)
        assert context.inputs.reference.numeric.dtype == getattr(torch, dtype)
        outcome = context.compute_classifier(
            metric,
            np.arange(len(reference))[None],
            np.arange(len(candidate))[None],
            folds[0][None],
            folds[1][None],
            [31],
            point=True,
        )[0]
        assert np.isfinite(outcome.value)
        assert all(fold["solver"] == solver for fold in outcome.folds)
        outcomes.append(outcome)
    assert outcomes[0].value == pytest.approx(outcomes[1].value, abs=5e-4)


def test_large_fp32_inputs_standardize_safely_and_mixed_precision_state_is_rejected(torch, ops):
    real = np.zeros((30, 1), dtype="float32")
    real[:6, 0] = [1e25, -1e25, 1e25, -1e25, 1e25, -1e25]
    result = native_classifier(torch, ops, real, real.copy(), draws=2)
    assert torch.isfinite(result[0]).all()
    x, y = problem(n=30, p=3)
    data = torch.from_numpy(x.astype("float32"))
    with pytest.raises(RuntimeError, match="design"):
        ops.classifier_derivatives(data, torch.from_numpy(y), data.new_zeros(4), 1 / len(x))


def test_fp32_ill_conditioned_saga_uses_stable_inverse(torch, ops):
    rng = np.random.default_rng(612)
    base = rng.normal(size=(1024, 1))
    values = np.repeat(base, 6, axis=1) + 1e-5 * rng.normal(size=(1024, 6))
    x = torch.from_numpy(values.astype("float32"))
    y = torch.from_numpy(rng.binomial(1, expit(base[:, 0])).astype("float32"))
    empty = x.new_empty(0)
    ids = torch.empty((len(x), 0), dtype=torch.int64)
    coefficients, diagnostics, reason = ops.classifier_saga_fit(
        x, ids, empty, empty, y, 1e-5, 1e-4, 500, 1024, 6, 31, 10**8, 4
    )
    assert coefficients.dtype == torch.float32
    assert diagnostics[7] == 1  # Reflectors protect small inverse eigenvalues.
    assert diagnostics[0] == 1 and reason == "online_gradient"
    reference = minimize(
        lambda w: independent_loss(np.column_stack([values, np.ones(len(x))]), y.numpy(), w, 1e-5),
        np.zeros(7),
        method="BFGS",
    )
    assert independent_loss(
        np.column_stack([values, np.ones(len(x))]), y.numpy(), coefficients.numpy(), 1e-5
    ) == pytest.approx(reference.fun, abs=2e-5)


def test_fp32_saga_workspace_budget_accounts_for_scalar_width(torch, ops):
    x, labels = problem(n=2048, p=32)
    for dtype in ["float32", "float64"]:
        numbers = torch.from_numpy(x[:, :-1].astype(dtype).copy())
        y = torch.from_numpy(labels.astype(dtype))
        empty = numbers.new_empty(0)
        ids = torch.empty((len(x), 0), dtype=torch.int64)
        args = (numbers, ids, empty, empty, y, 1 / len(x), 1e-4, 1, 512, 16, 31, 2 * 1024**2, 4)
        if dtype == "float32":
            coefficients, _, reason = ops.classifier_saga_fit(*args)
            assert torch.isfinite(coefficients).all() and reason == "max_epochs"
        else:
            with pytest.raises(RuntimeError, match="workspace exceeds"):
                ops.classifier_saga_fit(*args)


def test_fp32_dense_fold_memory_budget_accounts_for_scalar_width(torch):
    rng = np.random.default_rng(613)
    reference = pd.DataFrame(rng.normal(size=(300, 70)), columns=[f"x{i}" for i in range(70)])
    candidate = reference + 0.2
    schema = Schema(numeric=list(reference.columns))
    folds = source_folds(len(reference), len(candidate), 3, np.random.default_rng(31))
    for dtype in ["float32", "float64"]:
        metric = LinearC2ST(device="cpu", dtype=dtype, solver="direct", memory_limit_mb=1)
        context = context_for(metric, reference, candidate, schema, n_jobs=4)
        outcome = context.compute_classifier(
            metric,
            np.arange(len(reference))[None],
            np.arange(len(candidate))[None],
            folds[0][None],
            folds[1][None],
            [31],
            point=True,
        )[0]
        if dtype == "float32":
            assert np.isfinite(outcome.value) and outcome.converged
        else:
            assert np.isnan(outcome.value)
            assert all(fold["reason"] == "direct_memory_limit" for fold in outcome.folds)
