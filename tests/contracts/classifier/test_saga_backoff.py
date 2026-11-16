"""Objective and state-transition oracles for the CUDA-only fitting guard."""

# Optional torch installations skip this module before importing GPU helpers.
# ruff: noqa: E402

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tabgen_eval.execution.torch.cuda.classifier import saga
from tabgen_eval.execution.torch.cuda.classifier.data import Design, Fold
from tabgen_eval.execution.torch.cuda.classifier.saga_guard import ObjectiveGuard
from tabgen_eval.metrics import LinearC2ST

pytestmark = pytest.mark.unit


@pytest.fixture
def fold(monkeypatch):
    def independent_mv(self, weights):
        dense = torch.cat((self.numbers, torch.ones(self.n, 1, dtype=weights.dtype)), dim=1)
        return dense @ weights

    monkeypatch.setattr(Design, "mv", independent_mv)
    numbers = torch.ones(8, 1, dtype=torch.float64)
    design = Design(
        numbers, torch.empty(8, 0, dtype=torch.int64), numbers.new_empty(0), numbers.new_empty(0)
    )
    return Fold(
        design,
        design,
        numbers.new_zeros(8),
        numbers.new_ones(8),
        "sketchysaga",
        "nystrom",
        64 * 1024**2,
    )


def fake_native(monkeypatch, proposals):
    calls = []
    factor = SimpleNamespace(
        inverse_root=torch.ones(2, dtype=torch.float64),
        reflectors=torch.ones(2, 1, dtype=torch.float64),
        rotations=torch.ones(1, 1, dtype=torch.float64),
        tau=torch.ones(1, dtype=torch.float64),
        ratios=torch.ones(1, dtype=torch.float64),
        basis=lambda: torch.ones(2, 1, dtype=torch.float64),
    )
    monkeypatch.setattr(
        saga, "build", lambda *args, **kwargs: (factor, 0.5, {"minimum_inverse_ratio": 1.0})
    )

    def fit(packet, ptx, rate, ridge, tol, mode, stopping, averaged, limit, initial):
        calls.append(
            dict(
                schedule=packet[5].clone(),
                state=tuple(t.clone() for t in packet[-3:]),
                rate=rate,
                initial=initial,
            )
        )
        value = proposals[len(calls) - 1]
        if isinstance(value, Exception):
            raise value
        done, steps = packet[5].shape[:2]
        gradient = 0.01
        if isinstance(value, tuple):
            value, done, gradient = value
        w, table, average = (t.clone() for t in packet[-3:])
        w[0] = value
        table.add_(1)
        average.add_(1)
        finite = float(np.isfinite(value))
        initial = 7.0 if initial < 0 else initial
        history = w.new_tensor([[abs(value), abs(value), finite, 0.0, gradient, initial]] * done)
        return (
            w,
            table,
            average,
            w.new_empty(0),
            history,
            done,
            done * steps,
            w.new_tensor([gradient, 0.0]),
        )

    monkeypatch.setattr(saga, "operators", lambda *args: (SimpleNamespace(saga_fit=fit), "ptx"))
    return calls


def options(epochs, **kwargs):
    return dict(
        ridge=0.125,
        saga_max_epochs=epochs,
        saga_batch_size=4,
        saga_rank=1,
        tolerance=1e-8,
        **kwargs,
    )


def test_probe_matches_independent_logistic_loss_and_excludes_intercept(fold):
    guard = ObjectiveGuard(fold, 42, 0.125)
    weights = torch.tensor([-0.3, 4.0], dtype=torch.float64)
    expected = np.logaddexp(0, 3.7) + 0.5 * 0.125 * 0.3**2
    assert guard.objective(weights) == pytest.approx(expected, rel=1e-15)
    boundary = guard.previous + max(1e-4, 0.02 * guard.previous)
    assert not guard.rejects(boundary, True)
    assert guard.rejects(np.nextafter(boundary, np.inf), True)
    assert guard.rejects(float("nan"), True)


def test_rejection_replays_same_prefix_restores_all_states_and_counts_failed_work(
    fold, monkeypatch
):
    calls = fake_native(monkeypatch, [1000.0, -0.1])
    weights, state = saga.fit(fold, options(20), 42)
    assert state["epochs_completed"] == 20
    assert state["accepted_epochs"] == 4
    assert state["iterations"] == 40
    assert state["accepted_updates"] == 8
    assert state["backoff_retries"] == 1
    assert state["rejected_epochs"] == 16
    assert state["reason"] == "max_epochs"
    torch.testing.assert_close(calls[1]["schedule"], calls[0]["schedule"][:4], atol=0, rtol=0)
    for before, retry in zip(calls[0]["state"], calls[1]["state"], strict=True):
        torch.testing.assert_close(before, retry, atol=0, rtol=0)
    assert calls[0]["initial"] == -1
    assert calls[1]["initial"] == 7
    assert state["initial_online_gradient_infinity"] == 7
    assert calls[1]["rate"] == calls[0]["rate"] / 2
    assert weights[0] == -0.1


def test_budget_rejection_returns_last_accepted_model_and_its_diagnostics(fold, monkeypatch):
    fake_native(monkeypatch, [-0.1, 1000.0])
    weights, state = saga.fit(fold, options(20), 42)
    assert weights[0] == -0.1
    assert state["reason"] == "max_epochs"
    assert state["accepted_epochs"] == 16
    assert state["rejected_epochs"] == 4
    assert state["maximum_feature_coefficient"] == 0.1
    assert state["online_gradient_infinity"] == 0.01
    assert state["final_learning_rate"] == 0.5


def test_guard_rejects_native_convergence_before_accepting_stop(fold, monkeypatch):
    calls = fake_native(monkeypatch, [(1000.0, 1, 0.0), (-0.1, 1, 0.0)])
    weights, state = saga.fit(fold, options(20), 42)
    assert state["converged"]
    assert state["epochs_completed"] == 2
    assert state["accepted_epochs"] == 1
    assert state["rejected_epochs"] == 1
    assert state["backoff_retries"] == 1
    assert weights[0] == -0.1
    torch.testing.assert_close(calls[0]["schedule"][0], calls[1]["schedule"][0], atol=0, rtol=0)


def test_initial_gradient_does_not_relax_the_absolute_tolerance(fold, monkeypatch):
    fake_native(monkeypatch, [-0.1])
    tolerance = 0.005
    _, state = saga.fit(fold, {**options(1), "tolerance": tolerance}, 42)
    assert state["initial_online_gradient_infinity"] == 7
    assert tolerance < state["online_gradient_infinity"] < tolerance * 7
    assert state["gradient_stopping_threshold"] == state["tolerance"] == tolerance
    assert not state["converged"] and state["reason"] == "max_epochs"


@pytest.mark.parametrize("value", [1000.0, float("nan")])
def test_zero_accepted_state_and_retry_cap_are_unavailable(fold, monkeypatch, value):
    fake_native(monkeypatch, [value] * 5)
    _, state = saga.fit(fold, options(128), 42)
    assert state["reason"] == "numerical_failure"
    assert state["accepted_epochs"] == 0
    assert state["epochs_completed"] == 80
    assert state["backoff_retries"] == 4
    assert state["rejected_epochs"] == 80


def test_resource_and_programming_errors_are_not_retried(fold, monkeypatch):
    calls = fake_native(monkeypatch, [MemoryError("allocation")])
    _, state = saga.fit(fold, options(128), 42)
    assert state["reason"] == "memory_limit"
    assert len(calls) == 1
    fake_native(monkeypatch, [RuntimeError("incorrect packet")])
    with pytest.raises(RuntimeError, match="incorrect packet"):
        saga.fit(fold, options(128), 42)


def test_stable_guard_preserves_trajectory_and_private_sampling(fold, monkeypatch):
    calls = fake_native(monkeypatch, [-0.1, -0.2, -0.1, -0.2])
    guarded, a = saga.fit(fold, options(20), 42)
    unguarded, b = saga.fit(fold, options(20, saga_backoff=False), 42)
    torch.testing.assert_close(guarded, unguarded, atol=0, rtol=0)
    for left, right in zip(calls[:2], calls[2:], strict=True):
        torch.testing.assert_close(left["schedule"], right["schedule"], atol=0, rtol=0)
    assert a["backoff_retries"] == 0
    assert a["online_gradient_squared"] == b["online_gradient_squared"]


@pytest.mark.parametrize("value", [0, 1, "yes", None])
def test_guard_option_requires_boolean(value):
    with pytest.raises(ValueError, match="saga_backoff"):
        LinearC2ST(saga_backoff=value)


@pytest.mark.gpu
def test_actual_native_guard_preserves_stable_fit_and_heldout_isolation(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    device = "cuda:0"
    generator = torch.Generator(device=device).manual_seed(49)
    numbers = torch.randn(512, 4, device=device, dtype=torch.float64, generator=generator)
    design = Design(
        numbers,
        torch.empty(512, 0, device=device, dtype=torch.int64),
        numbers.new_empty(0),
        numbers.new_empty(0),
    )
    labels = (numbers[:, 0] > 0).to(torch.float64)
    fold = Fold(design, design, labels, 1 - labels, "sketchysaga", "nystrom", 128 * 1024**2)
    original_operators = saga.operators
    input_checks = []

    def checked_operators(*args):
        native, ptx = original_operators(*args)

        def checked_fit(packet, *parameters):
            saved = tuple(t.clone() for t in packet[-3:])
            result = native.saga_fit(packet, *parameters)
            for current, previous in zip(packet[-3:], saved, strict=True):
                torch.testing.assert_close(current, previous, atol=0, rtol=0)
            input_checks.append(True)
            return result

        return SimpleNamespace(saga_fit=checked_fit), ptx

    monkeypatch.setattr(saga, "operators", checked_operators)
    kwargs = dict(
        ridge=1 / 512,
        saga_max_epochs=20,
        saga_batch_size=128,
        saga_rank=4,
        tolerance=1e-30,
    )
    guarded, a = saga.fit(fold, kwargs, 42)
    plain, b = saga.fit(fold, dict(kwargs, saga_backoff=False), 42)
    torch.testing.assert_close(guarded, plain, atol=0, rtol=0)
    assert a["backoff_retries"] == 0
    assert a["accepted_epochs"] == b["epochs_completed"] == 20
    fold.scoring_labels = torch.full_like(labels, float("nan"))
    repeated, c = saga.fit(fold, kwargs, 42)
    torch.testing.assert_close(repeated, guarded, atol=0, rtol=0)
    assert c["final_probe_objective"] == a["final_probe_objective"]
    assert len(input_checks) == 6
