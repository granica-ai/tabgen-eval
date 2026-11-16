"""Native Newton precision, stopping, isolation and packet validation."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import expit

from tabgen_eval.metrics import LinearC2ST

pytestmark = [pytest.mark.unit, pytest.mark.gpu]


def _fold(torch, device, dtype):
    from tabgen_eval.execution.torch.cuda.classifier.data import Design, Fold

    rng = np.random.default_rng(9401)
    x = np.column_stack((rng.normal(size=(257, 7)), np.ones(257))).astype(dtype)
    y = rng.binomial(1, expit(x @ np.array([0.6, -0.5, 0.1, 0.2, -0.2, 0.3, 0.4, -0.7])))
    matrix = torch.tensor(x, device=device).T.contiguous().T
    labels = torch.tensor(y, dtype=matrix.dtype, device=device)
    design = Design(
        matrix[:, :-1].contiguous(),
        torch.empty((len(matrix), 0), device=device, dtype=torch.int64),
        matrix.new_empty(0),
        matrix.new_empty(0),
        dense=matrix,
    )
    return Fold(design, design, labels, labels, "direct", "none", 64 * 1024**2), x, y


def test_compiled_newton_owns_iteration_and_preserves_rng_and_inputs(
    torch, device, dtype, monkeypatch
):
    from tabgen_eval.execution.torch.cuda.classifier import solver
    from tabgen_eval.execution.torch.cuda.classifier.device import operators

    fold, x, y = _fold(torch, device, dtype)
    # Prepare lazy modules before asserting numerical Python helpers are absent.
    operators(fold.train.dense, algorithm="newton")
    before = fold.train.dense.clone(), fold.labels.clone()
    cpu_rng = torch.random.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone()

    def forbidden(*args, **kwargs):
        pytest.fail("native Newton dispatched a Python numerical helper")

    for name in ("derivatives", "hessian_vector", "logistic"):
        monkeypatch.setattr(solver, name, forbidden)
    for name in ("dot", "isfinite", "sqrt"):
        monkeypatch.setattr(torch, name, forbidden)
    tolerance = 1e-6 if dtype == "float64" else np.sqrt(1e-9)
    metric = LinearC2ST(ridge=0.03, tolerance=tolerance)
    weights, diagnostics = solver.fit(fold, metric.native_options(), 319)
    assert weights.dtype == getattr(torch, dtype)
    assert diagnostics["converged"], diagnostics
    # An independent full gradient checks objective normalization and excludes
    # the intercept from ridge. The online SAGA gradient is not involved here.
    w = weights.cpu().numpy().astype(float)
    gradient = x.astype(float).T @ ((expit(x.astype(float) @ w) - y) / len(y))
    gradient[:-1] += 0.03 * w[:-1]
    squared = float(gradient @ gradient)
    assert diagnostics["gradient_squared"] == pytest.approx(
        squared, rel=1e-3 if dtype == "float32" else 1e-8, abs=1e-14
    )
    assert squared <= tolerance**2 * (1.01 if dtype == "float32" else 1.00001)
    assert torch.equal(fold.train.dense, before[0])
    assert torch.equal(fold.labels, before[1])
    assert torch.equal(torch.random.get_rng_state(), cpu_rng)
    assert torch.equal(torch.cuda.get_rng_state(device), cuda_rng)


def test_native_newton_reports_final_full_gradient_after_iteration_cap(torch, device, dtype):
    from tabgen_eval.execution.torch.cuda.classifier.solver import fit

    fold, x, y = _fold(torch, device, dtype)
    metric = LinearC2ST(ridge=0.03, max_iter=1, tolerance=1e-15)
    weights, state = fit(fold, metric.native_options(), 1)
    w = weights.cpu().numpy().astype(float)
    gradient = x.astype(float).T @ ((expit(x.astype(float) @ w) - y) / len(y))
    gradient[:-1] += 0.03 * w[:-1]
    assert not state["converged"]
    assert state["reason"] == "max_iter"
    assert state["iterations"] == 1
    assert state["gradient_squared"] == pytest.approx(
        float(gradient @ gradient), rel=2e-5 if dtype == "float32" else 1e-12
    )
    assert state["stopping_rule"] == "gradient_squared"


def test_native_newton_tolerance_is_in_gradient_norm_units(torch, device, dtype):
    from tabgen_eval.execution.torch.cuda.classifier.device import operators

    matrix = (
        torch.tensor([[0.0, 1.0]] * 4, device=device, dtype=getattr(torch, dtype)).T.contiguous().T
    )
    labels = matrix.new_tensor([0.0, 0.0, 0.0, 1.0])
    ops, ptx = operators(matrix, algorithm="newton")
    tolerance = 0.1
    # The initial full gradient has norm .25 but squared norm .0625.
    weights, diagnostic, status, iterations = ops.newton_fit(
        matrix, labels, ptx, 0.03, tolerance, 1, 64 * 1024**2
    )
    assert iterations == 1 and weights[-1] != 0
    assert (status == 1) == (diagnostic[0] <= tolerance**2)


def test_native_newton_checks_allocation_budget_and_packet_layout(torch, device, dtype):
    from tabgen_eval.execution.torch.cuda.classifier.device import operators

    fold, _, _ = _fold(torch, device, dtype)
    matrix, labels = fold.train.dense, fold.labels
    ops, ptx = operators(matrix, algorithm="newton")
    with pytest.raises(torch.OutOfMemoryError, match="workspace exceed"):
        ops.newton_fit(matrix, labels, ptx, 0.03, 1e-5, 20, 1)
    with pytest.raises(RuntimeError, match="column-major"):
        ops.newton_fit(matrix.contiguous(), labels, ptx, 0.03, 1e-5, 20, 64 * 1024**2)
    other = labels.to(torch.float64 if dtype == "float32" else torch.float32)
    with pytest.raises(RuntimeError, match="same-device/dtype"):
        ops.newton_fit(matrix, other, ptx, 0.03, 1e-5, 20, 64 * 1024**2)


def test_native_newton_nonfinite_gradient_is_an_honest_failure(torch, device, dtype):
    from tabgen_eval.execution.torch.cuda.classifier.solver import fit

    fold, _, _ = _fold(torch, device, dtype)
    fold.train.dense[0, 0] = float("nan")
    _, state = fit(fold, LinearC2ST().native_options(), 1)
    assert not state["converged"]
    assert state["reason"] == "nonfinite_gradient"
    assert state["iterations"] == 0
    assert not np.isfinite(state["gradient_squared"])


def test_native_newton_uses_callers_nondefault_stream(torch, device, dtype):
    from tabgen_eval.execution.torch.cuda.classifier.solver import fit

    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        fold, _, _ = _fold(torch, device, dtype)
        weights, state = fit(fold, LinearC2ST(ridge=0.03).native_options(), 81)
        assert torch.cuda.current_stream(device) == stream
        assert state["converged"], state
        assert weights.dtype == getattr(torch, dtype)
    stream.synchronize()


def test_small_decrement_cannot_replace_full_gradient_stopping(torch, device, dtype):
    from tabgen_eval.execution.torch.cuda.classifier.data import Design, Fold
    from tabgen_eval.execution.torch.cuda.classifier.solver import fit

    # At zero the exact full gradient is [-1, 0], but the large Hessian
    # produces a decrement around 1e-12. A decrement shortcut would accept
    # zero coefficients despite a gradient squared of one.
    matrix = (
        torch.tensor(
            [[2000008.0, 1.0], [-2000000.0, 1.0], [2000000.0, 1.0], [-2000000.0, 1.0]],
            dtype=getattr(torch, dtype),
            device=device,
        )
        .T.contiguous()
        .T
    )
    labels = matrix.new_tensor([1.0, 1.0, 0.0, 0.0])
    design = Design(
        matrix[:, :-1].contiguous(),
        torch.empty((4, 0), device=device, dtype=torch.int64),
        matrix.new_empty(0),
        matrix.new_empty(0),
        dense=matrix,
    )
    fold = Fold(design, design, labels, labels, "direct", "none", 64 * 1024**2)
    metric = LinearC2ST(ridge=0.03, max_iter=1, tolerance=0.0001)
    weights, state = fit(fold, metric.native_options(), 81)
    assert state["iterations"] == 1
    assert weights[0].item() != 0
    assert 0 < state["decrement_squared"] < metric.tolerance**2
    assert state["converged"] == (state["gradient_squared"] <= metric.tolerance**2)
    # The deliberately unscaled FP32 case can reach its arithmetic floor.
    # It must retain nonconvergence rather than use the tiny decrement.
    if not state["converged"]:
        assert state["reason"] == "max_iter"
