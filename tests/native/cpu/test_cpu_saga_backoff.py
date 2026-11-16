"""Independent state, objective and work-budget oracles for native CPU backoff."""

import numpy as np
import pytest

pytestmark = pytest.mark.unit


def packet(torch, numbers, labels):
    numbers = torch.as_tensor(numbers, dtype=torch.float64)
    labels = torch.as_tensor(labels, dtype=torch.float64)
    n, p = numbers.shape
    empty = numbers.new_empty(0)
    return [
        numbers,
        torch.empty(n, 0, dtype=torch.int64),
        empty,
        empty,
        labels,
        numbers.new_zeros(p + 1),
        numbers.new_zeros(n),
        numbers.new_zeros(p + 1),
        empty,
        numbers.new_empty(p + 1, 0),
        empty,
    ]


def run(
    torch, ops, inputs, *, epochs, rate, tolerance=1e-30, batch=None, schedule=None, backoff=True
):
    n = len(inputs[0])
    batch = n if batch is None else batch
    if schedule is None:
        schedule = torch.arange(n)[None, None].repeat(epochs, 1, 1)
    return ops.classifier_saga_guarded_updates(
        inputs, schedule, 0.125, rate, tolerance, 42, epochs, backoff, 1
    )


def test_rejected_convergence_replays_and_probe_excludes_intercept(torch, ops):
    inputs = packet(torch, np.ones((8, 1)), np.zeros(8))
    originals = [t.clone() for t in inputs]
    # Each first update meets this online threshold, but large steps increase
    # the training objective. Guard approval must precede convergence approval.
    states, diag, reason = run(torch, ops, inputs, epochs=20, rate=32, tolerance=1)
    assert reason == "online_gradient"
    assert diag[1] == diag[2] == 4  # Three rejected epochs, one retained.
    assert diag[11] == diag[12] == 1
    assert diag[19] == diag[20] == 3
    assert diag[9] == 0.5
    assert diag[13] == 32 and diag[14] == diag[15] == 4
    torch.testing.assert_close(states[0], inputs[0].new_tensor([-2, -2]), atol=0, rtol=0)
    expected = np.logaddexp(0, -4) + 0.5 * 0.125 * 2**2
    assert float(diag[22]) == pytest.approx(expected, abs=1e-15)
    for current, old in zip(inputs, originals, strict=True):
        torch.testing.assert_close(current, old, atol=0, rtol=0)


@pytest.mark.parametrize("private_sampling", [False, True])
def test_rolls_back_every_state_and_replays_identical_draw_prefix(torch, ops, private_sampling):
    inputs = packet(torch, np.ones((8, 1)), np.zeros(8))
    batch = 4 if private_sampling else 8
    schedule = torch.empty(0, 8 // batch, batch, dtype=torch.int64) if private_sampling else None
    states, diag, reason = run(
        torch, ops, inputs, epochs=20, rate=16, batch=batch, schedule=schedule
    )
    assert reason == "max_epochs"
    assert diag[1] == 20 and diag[20] == 16
    assert diag[11] == 4 and diag[19] == 1
    reference, control, _ = run(
        torch, ops, inputs, epochs=4, rate=8, batch=batch, schedule=schedule, backoff=False
    )
    for actual, expected in zip(states, reference, strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert diag[8] == control[8]
    assert diag[3] == control[3]
    assert diag[2] == 20 * (8 // batch)
    assert diag[12] == 4 * (8 // batch)


def test_budget_rejection_returns_accepted_state_and_accepted_diagnostics(torch, ops):
    rng = np.random.default_rng(5)
    inputs = packet(torch, rng.normal(size=(16, 2)) * 3, rng.integers(2, size=16))
    schedule = torch.empty(20, 4, 4, dtype=torch.int64)
    schedule[:16] = torch.arange(4)
    schedule[16:] = torch.arange(4, 8)
    states, diag, reason = run(torch, ops, inputs, epochs=20, rate=0.1, schedule=schedule)
    assert reason == "max_epochs"
    assert diag[11] == 16 and diag[20] == 4 and diag[19] == 0
    reference, control, _ = run(
        torch, ops, inputs, epochs=16, rate=0.1, schedule=schedule, backoff=False
    )
    for actual, expected in zip(states, reference, strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for i in [3, 4, 8, 9, 10]:
        assert diag[i] == control[i]


@pytest.mark.parametrize("rate", [1e8, 1e308])
def test_retry_limit_and_zero_accepted_model_are_numerical_failure(torch, ops, rate):
    inputs = packet(torch, np.ones((8, 1)), np.zeros(8))
    states, diag, reason = run(torch, ops, inputs, epochs=128, rate=rate)
    assert reason == "numerical_failure" and diag[0] == 0
    assert diag[11] == diag[12] == 0 and diag[19] == diag[21] == 4
    assert diag[1] == diag[20]
    assert torch.isnan(diag[3]) and torch.isnan(diag[8])
    for state in states:
        assert (state == 0).all()


def test_guarded_native_fit_keeps_normal_trajectory_and_global_rng(torch, ops):
    rng = np.random.default_rng(619)
    x = torch.from_numpy(rng.normal(size=(512, 4)))
    labels = (x[:, 0] > 0).double()
    ids = torch.empty(512, 0, dtype=torch.int64)
    empty = x.new_empty(0)
    args = x, ids, empty, empty, labels, 1 / 512, 1e-30, 20, 128, 4, 42, 10**7, 4
    before = torch.random.get_rng_state().clone()
    guarded = ops.classifier_saga_fit(*args, True)
    control = ops.classifier_saga_fit(*args, False)
    torch.testing.assert_close(guarded[0], control[0], atol=0, rtol=0)
    torch.testing.assert_close(guarded[1][:11], control[1][:11], atol=0, rtol=0)
    torch.testing.assert_close(torch.random.get_rng_state(), before, atol=0, rtol=0)
    assert guarded[1][19] == 0 and guarded[1][18] == 2
    assert guarded[1][11] == guarded[1][1] == 20


def test_native_memory_limit_includes_guard_state_and_probe(torch, ops):
    x = torch.ones(512, 4, dtype=torch.float64)
    labels = (torch.arange(512) % 2).double()
    empty = x.new_empty(0)
    ids = torch.empty(512, 0, dtype=torch.int64)
    args = x, ids, empty, empty, labels, 1 / 512, 1e-4, 1, 128, 4, 42
    # Exactly the original native fit allocation bound; guard needs extra
    # probe and complete-state storage, checked before any factorization.
    baseline_bytes = 8 * (512 * 4 + 2 * 512 + 16 * 5 * 4 + 16 * 5 + 128 * (4 + 8))
    baseline_bytes += 8 * (512 * (2 * 4 + 2 * 4 + 8) + 4 * 5 * 4 + 8 * 5)
    with pytest.raises(RuntimeError, match="memory limit"):
        ops.classifier_saga_fit(*args, baseline_bytes, 1, True)
    assert torch.isfinite(ops.classifier_saga_fit(*args, baseline_bytes, 1, False)[0]).all()


def test_standalone_dense_dispatch_honors_default_and_disable_flag(torch, ops):
    generator = torch.Generator().manual_seed(721)
    numeric = torch.randn(128, 3, dtype=torch.float64, generator=generator)
    x = torch.cat((numeric, torch.ones(128, 1, dtype=torch.float64)), dim=1)
    labels = (numeric[:, 0] > 0).double()
    args = x, labels, "sketchysaga", 0.0, 1e-30, 100, 500, 2e11, 20, 64, 3, 42, 1
    guarded = ops.classifier_fit(*args)
    disabled = ops.classifier_fit(*args, False)
    torch.testing.assert_close(guarded[0], disabled[0], rtol=0, atol=0)
    torch.testing.assert_close(guarded[1][:5], disabled[1][:5], rtol=0, atol=0)
    assert guarded[1][10] == 1 and disabled[1][10] == 0
    assert guarded[1][13] == 0  # No backoffs in this stable case.
