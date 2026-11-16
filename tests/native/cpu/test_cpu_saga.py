"""Independent dense oracles for the native compact CPU SAGA implementation."""

import numpy as np
import pytest
from scipy.special import expit

pytestmark = pytest.mark.unit


def expanded(numbers, ids, centers, scales):
    cats = np.zeros((len(numbers), len(centers)))
    for i, row in enumerate(ids):
        for category in row:
            if category >= 0:
                cats[i, category] = 1
    return np.column_stack([numbers, (cats - centers) / scales, np.ones(len(numbers))])


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("preconditioned", [False, True])
@pytest.mark.parametrize("warm", [False, True])
@pytest.mark.parametrize("threads", [1, 4])
def test_compact_native_updates_match_dense_oracle_and_preserve_state(
    torch, ops, preconditioned, warm, threads, dtype
):
    rng = np.random.default_rng(5)
    numbers = rng.normal(size=(17, 3))
    ids = np.column_stack([rng.integers(-1, 3, 17), rng.integers(3, 5, 17)])
    centers, scales = np.array([0.5, 0, 0.3, 0.0, 1.0]), np.array([1, 0.5, 2, 1, 0.4])
    x = expanded(numbers, ids, centers, scales)
    labels = rng.integers(0, 2, len(numbers)).astype(float)
    initial = rng.normal(size=x.shape[1]) * 0.1 if warm else np.zeros(x.shape[1])
    cache = expit(x @ initial) - labels if warm else np.zeros(len(x))
    average = x.T @ cache / len(x)
    if preconditioned:
        root = rng.uniform(0.7, 1.4, x.shape[1])
        basis = np.linalg.qr(rng.normal(size=(x.shape[1], 3)))[0]
        ratios = np.array([0.1, 0.5, 0.9])
        inverse = (
            np.diag(root)
            @ (np.eye(len(root)) + basis @ np.diag(ratios - 1) @ basis.T)
            @ np.diag(root)
        )
    else:
        root, basis, ratios = np.empty(0), np.empty((0, 0)), np.empty(0)
        inverse = np.eye(x.shape[1])
    schedule = np.array([rng.choice(len(x), 7, replace=False) for _ in range(23)])
    packet = [numbers, ids, centers, scales, labels, initial, cache, average, root, basis, ratios]
    tensors = [
        torch.from_numpy(a.copy() if a.dtype.kind == "i" else a.astype(dtype)) for a in packet
    ]
    originals = [a.clone() for a in tensors]
    before_rng = torch.random.get_rng_state().clone()
    weights, table, cached_mean, gradient, diagnostics = ops.classifier_saga_updates(
        tensors, torch.from_numpy(schedule), 0.02, 0.03, threads
    )
    w, saved, mean = initial.copy(), cache.copy(), average.copy()
    for rows in schedule:
        current = expit(x[rows] @ w) - labels[rows]
        correction = x[rows].T @ (current - saved[rows]) / len(rows)
        grad = mean + correction
        grad[:-1] += 0.02 * w[:-1]
        mean += correction * len(rows) / len(x)
        saved[rows] = current
        w -= 0.03 * inverse @ grad
    rtol, atol = (3e-5, 3e-7) if dtype == "float32" else (1e-13, 1e-14)
    for actual, expected in [(weights, w), (table, saved), (cached_mean, mean), (gradient, grad)]:
        assert actual.dtype == getattr(torch, dtype)
        np.testing.assert_allclose(actual.numpy(), expected, rtol=rtol, atol=atol)
    np.testing.assert_allclose(
        cached_mean.numpy(), x.T @ table.numpy() / len(x), rtol=rtol, atol=atol
    )
    assert diagnostics[-1] == pytest.approx(grad @ grad, abs=3e-8 if dtype == "float32" else 1e-15)
    for a, old in zip(tensors, originals, strict=True):
        torch.testing.assert_close(a, old, rtol=0, atol=0)
    torch.testing.assert_close(torch.random.get_rng_state(), before_rng, rtol=0, atol=0)


def test_training_only_preparation_matches_independent_transform_and_unknown_vocabulary(torch, ops):
    real = np.array([[1, np.nan], [2, np.nan], [3, 5], [1e5, 6], [4, np.nan], [5, 8]], float)
    fake = np.array([[3, 100], [4, 200], [5, 300], [-1e5, 400], [6, 500], [7, 600]], float)
    rc = np.array([[0], [1], [0], [7], [1], [0]], dtype=np.int64)
    sc = np.array([[0], [2], [1], [9], [-1], [2]], dtype=np.int64)
    folds = np.array([1, 2, 1, 0, 2, 0], dtype=np.int64)
    tensors = [torch.from_numpy(v) for v in [real, rc, fake, sc, folds, folds]]
    result = ops.classifier_saga_prepare(*tensors, 0, 10**7)
    train_num, train_ids, test_num, test_ids, centers, scales, labels, scoring = result
    training_rows = folds != 0
    impute = np.nanmean(real[training_rows], axis=0)
    train = np.vstack([real[training_rows], fake[training_rows]])
    train = np.where(np.isfinite(train), train, impute)
    expected = np.column_stack(((train - train.mean(0)) / train.std(0), [1, 1, 0, 1, 0, 0, 0, 0]))
    np.testing.assert_allclose(train_num, expected, rtol=0, atol=1e-14)
    # Both training classes define the vocabulary; missing is distinct. Held-out
    # categories 7 and 9 still do not acquire indicator columns.
    np.testing.assert_array_equal(centers, np.zeros(4))
    np.testing.assert_array_equal(scales, np.ones(4))
    np.testing.assert_array_equal(train_ids[:, 0], [1, 2, 1, 2, 1, 3, 2, 0])
    np.testing.assert_array_equal(test_ids[:, 0], [-1, 1, -1, 3])
    np.testing.assert_array_equal(labels, [0] * 4 + [1] * 4)
    np.testing.assert_array_equal(scoring, [0, 0, 1, 1])
    coefficients = torch.arange(train_num.shape[1] + centers.numel() + 1, dtype=torch.float64)
    actual = ops.classifier_saga_mv(test_num, test_ids, centers, scales, coefficients)
    dense = expanded(test_num.numpy(), test_ids.numpy(), centers.numpy(), scales.numpy())
    np.testing.assert_allclose(actual, dense @ coefficients.numpy(), atol=1e-12, rtol=1e-14)


def test_saga_fit_is_private_seeded_and_enforces_memory_before_factorization(torch, ops):
    rng = np.random.default_rng(65)
    x = torch.from_numpy(rng.normal(size=(50, 3)))
    identities = torch.empty((50, 0), dtype=torch.int64)
    empty = x.new_empty(0)
    y = (x[:, 0] > 0).double()
    state = torch.random.get_rng_state().clone()
    args = x, identities, empty, empty, y, 1 / 50, 1e-4, 20, 8, 3, 31, 10**7, 1
    first = ops.classifier_saga_fit(*args)
    second = ops.classifier_saga_fit(*args)
    torch.testing.assert_close(first[0], second[0], rtol=0, atol=0)
    torch.testing.assert_close(first[1], second[1], rtol=0, atol=0)
    torch.testing.assert_close(torch.random.get_rng_state(), state, rtol=0, atol=0)
    assert first[1][2] == first[1][1] * 7
    with pytest.raises(RuntimeError, match="memory limit"):
        ops.classifier_saga_fit(*args[:-2], 1, 1)


def native_classifier(torch, ops, real, candidate, *, epochs=1, draws=1):
    """One source-stable native draw; all option/shape inputs are explicit."""
    reference = torch.from_numpy(real)
    generated = torch.from_numpy(candidate)
    rows = torch.arange(len(real), dtype=torch.int64)[None].repeat(draws, 1)
    folds = rows % 3
    categories = torch.empty((len(real), 0), dtype=torch.int64)
    complete = torch.ones(len(real), dtype=torch.bool)
    return ops.classifier(
        reference,
        categories,
        generated,
        categories,
        complete,
        complete,
        rows,
        rows,
        folds,
        folds,
        torch.full((draws,), 42, dtype=torch.int64),
        3,
        "sketchysaga",
        0.0,
        1e-12,
        100,
        512,
        2e11,
        epochs,
        8,
        2,
        10**7,
        4,
        1,
    )


def test_terminal_numerical_failure_has_nan_auc_instead_of_zero_model_chance(torch, ops):
    # These FP64 observations exceed representable variance and curvature.
    # The diagonal fallback also cannot represent this curvature. This failure
    # must remain a missing score, including under native parallel draw work.
    real = np.zeros((30, 1))
    real[:6, 0] = [1e155, -1e155, 1e155, -1e155, 1e155, -1e155]
    result = native_classifier(torch, ops, real, real.copy(), draws=2)
    assert torch.isnan(result[0]).all()
    assert result[2] == ["sketchysaga"] * 6
    assert result[3] == ["numerical_failure"] * 6
    assert not result[1][..., 3].any()
    assert (result[1][..., 4:6] == 0).all()


def test_valid_finite_epoch_limit_retains_auc(torch, ops):
    rng = np.random.default_rng(157)
    real = rng.normal(size=(30, 2))
    candidate = rng.normal(0.7, 1, size=(30, 2))
    result = native_classifier(torch, ops, real, candidate)
    assert torch.isfinite(result[0]).all()
    assert 0.5 <= result[0][0] <= 1
    assert result[2] == ["sketchysaga"] * 3
    assert result[3] == ["max_epochs"] * 3
    assert (result[1][..., 5] == 1).all()
    assert not result[1][..., 3].any()


def test_online_gradient_stopping_uses_the_absolute_tolerance(torch, ops):
    rng = np.random.default_rng(812)
    x = torch.from_numpy(rng.normal(size=(48, 4)))
    ids = torch.empty((48, 0), dtype=torch.int64)
    empty = x.new_empty(0)
    y = (x[:, 0] + 0.3 * x[:, 1] > 0).double()
    args = (x, ids, empty, empty, y, 1 / 48)
    # Full batch gives an independent initial online-gradient oracle.
    initial = torch.cat((x.T @ (0.5 - y) / len(y), (0.5 - y).mean()[None]))
    bounded = ops.classifier_saga_fit(*args, 1e-30, 2, 48, 3, 42, 10**7, 1)
    diag = bounded[1]
    assert diag[0] == 0 and diag[1] == 2
    assert float(diag[9]) == pytest.approx(float(initial.abs().max()), abs=1e-14)
    assert float(diag[8]) != float(diag[9])
    # At full batch, the existing direction is the exact gradient before the
    # final update, including ridge on features and no ridge on intercept.
    first = ops.classifier_saga_fit(*args, 1e-30, 1, 48, 3, 42, 10**7, 1)
    first_weights = first[0]
    residual = torch.sigmoid(x @ first_weights[:-1] + first_weights[-1]) - y
    exact = torch.cat((x.T @ residual / len(y) + first_weights[:-1] / 48, residual.mean()[None]))
    assert float(diag[8]) == pytest.approx(float(exact.abs().max()), abs=1e-14)
    threshold = float(diag[8])
    absolute = ops.classifier_saga_fit(*args, threshold, 5, 48, 3, 42, 10**7, 1)
    assert absolute[2] == "online_gradient" and absolute[1][0] == 1
    assert absolute[1][1] == 2  # Equality must satisfy the rule.
    assert float(absolute[1][10]) == threshold


def test_initial_gradient_above_one_does_not_relax_the_stopping_threshold(torch, ops):
    from .test_cpu_saga_backoff import packet, run

    inputs = packet(torch, np.full((8, 1), 20.0), np.zeros(8))
    tolerance = 0.1
    _, diagnostic, reason = run(
        torch, ops, inputs, epochs=2, rate=0.1, tolerance=tolerance, backoff=False
    )
    assert diagnostic[9] == 10  # The initial online-gradient infinity norm.
    # A relative tolerance multiplied by the initial norm would accept this
    # second direction. The single absolute tolerance must retain nonconvergence.
    assert tolerance < diagnostic[8] < tolerance * diagnostic[9]
    assert diagnostic[10] == tolerance
    assert diagnostic[0] == 0 and reason == "max_epochs"
