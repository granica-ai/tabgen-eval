"""Independent dense CPU SAGA update oracles for native FP32/FP64 CUDA kernels."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tabgen_eval.execution.torch.cuda.classifier.device import operators
from tabgen_eval.execution.torch.cuda.classifier.preconditioner import CurvatureFactor

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


def library(dtype):
    ops, ptx = operators(torch.empty(1, device="cuda", dtype=dtype))
    return SimpleNamespace(fit=ops.saga_fit), ptx


def low_rank_diagonal(diagonal, low_rank):
    residual = diagonal - low_rank.square().sum(1)
    root = residual.rsqrt()
    qr, tau = torch.geqrf(root[:, None] * low_rank)
    rotations, singular, _ = torch.linalg.svd(
        torch.triu(qr[: low_rank.shape[1]]), full_matrices=False
    )
    return CurvatureFactor(root, qr, tau, rotations, 1 / (1 + singular.square())), residual


def stable_derivative(z, labels):
    e = torch.exp(-z.abs())
    return torch.where(
        labels == 0,
        torch.where(z >= 0, 1 / (1 + e), e / (1 + e)),
        torch.where(z >= 0, -e / (1 + e), -1 / (1 + e)),
    )


def column(matrix):
    # Explicit layout also handles the ambiguous stride of a single column.
    out = torch.empty_strided(matrix.shape, (1, matrix.shape[0]), dtype=matrix.dtype, device="cuda")
    out.copy_(matrix)
    return out


def fixture(kind, mode, dtype, epochs=3, steps=2, b=7):
    n = 23
    generator = torch.Generator().manual_seed(93)
    rng = np.random.default_rng(93)
    pn = 0 if kind == "categorical_only" else 2
    cards = [] if kind == "numeric_only" else [4, 3, 2]
    pc, k = len(cards), sum(cards)
    d = pn + k + 1
    numbers = torch.randn(n, pn, dtype=dtype, generator=generator)
    identities = torch.full((n, pc), -1, dtype=torch.int64)
    offset = 0
    for c, width in enumerate(cards):
        local = torch.as_tensor(rng.integers(-1, width, size=n))
        identities[:, c] = torch.where(local < 0, -1, local + offset)
        offset += width
    center = torch.linspace(0, 0.6, k, dtype=dtype)
    scales = torch.linspace(0.25, 1.25, k, dtype=dtype)
    categorical = torch.zeros(n, k, dtype=dtype)
    for row in range(n):
        for c in range(pc):
            identity = int(identities[row, c])
            if identity >= 0:
                categorical[row, identity] = 1
    if k:
        categorical = (categorical - center) / scales
    x = torch.cat((numbers, categorical, torch.ones(n, 1, dtype=dtype)), 1)
    labels = (torch.arange(n) % 2).to(dtype)
    initial = torch.randn(d, dtype=dtype, generator=generator) * 0.1
    if kind == "extreme_logits":
        initial[-1] = 1000
    table = torch.randn(n, dtype=dtype, generator=generator) * 0.2
    average = x.T @ table / n
    schedule = (
        np.stack([rng.choice(n, b, replace=False) for _ in range(epochs * steps)])
        .reshape(epochs, steps, b)
        .astype(np.int32)
    )
    rank = min(3, d)
    low = torch.randn(d, rank, dtype=dtype, generator=generator)
    diagonal = low.square().sum(1) + torch.linspace(0.75, 1.25, d, dtype=dtype)
    factor, residual = low_rank_diagonal(diagonal, low)
    basis = torch.orgqr(factor.reflectors, factor.tau) @ factor.rotations
    p = torch.eye(d, dtype=dtype) if mode == 0 else torch.diag(residual) + low @ low.T
    weights, cached, running, summed = (
        initial.clone(),
        table.clone(),
        average.clone(),
        torch.zeros_like(initial),
    )
    history = []
    initial_gradient_norm = None
    eta, ridge = 0.015, 0.025
    for epoch in range(epochs):
        previous = weights.clone()
        for batch_rows in schedule[epoch]:
            rows = torch.as_tensor(batch_rows, dtype=torch.int64)
            current = stable_derivative(x[rows] @ weights, labels[rows])
            delta = current - cached[rows]
            correction = x[rows].T @ (delta / b)
            gradient = running + correction
            gradient[:-1] += ridge * weights[:-1]
            if initial_gradient_norm is None:
                initial_gradient_norm = gradient.abs().max().item()
            last_gradient_estimate = gradient.clone()
            running += correction * b / n
            cached[rows] = current
            summed += weights
            weights -= eta * torch.linalg.solve(p, gradient)
        history.append(
            [
                weights[:-1].abs().max().item(),
                (weights[:-1] - previous[:-1]).abs().max().item(),
                1.0,
                abs((weights[-1] - previous[-1]).item()),
                gradient.abs().max().item(),
                initial_gradient_norm,
            ]
        )

    def gpu(value):
        return value.cuda()

    ratio = factor.ratios if mode != 2 else factor.ratios - 1
    packet = [
        gpu(numbers),
        gpu(identities),
        gpu(center),
        gpu(scales),
        gpu(labels),
        gpu(torch.from_numpy(schedule)),
        gpu(factor.inverse_root),
        column(factor.reflectors),
        gpu(factor.tau),
        column(factor.rotations),
        gpu(ratio),
        column(basis),
        gpu(initial),
        gpu(table),
        gpu(average),
    ]
    ops, ptx = library(dtype)
    output = ops.fit(packet, ptx, eta, ridge, 1e-4, mode, False, True, 64 * 1024**2)
    errors = {}
    for label, actual, expected in zip(
        ["weights", "table", "average", "averaged_sum", "epoch_diagnostics"],
        output[:5],
        [weights, cached, running, summed, torch.tensor(history, dtype=torch.float64)],
        strict=False,
    ):
        actual = actual.cpu()
        difference = float((actual - expected).abs().max().item())
        torch.testing.assert_close(
            actual,
            expected,
            rtol=(2e-5 if dtype == torch.float32 else 1e-11),
            atol=(2e-6 if dtype == torch.float32 else 1e-12),
            check_dtype=False,
        )
        errors[label] = difference
    assert output[5:7] == (epochs, epochs * steps)
    torch.testing.assert_close(
        output[7].cpu(),
        last_gradient_estimate,
        rtol=(2e-5 if dtype == torch.float32 else 1e-11),
        atol=(2e-6 if dtype == torch.float32 else 1e-12),
        check_dtype=False,
    )
    errors["last_online_gradient"] = float(
        (output[7].cpu() - last_gradient_estimate).abs().max().item()
    )
    # Fresh repeated execution must be exact, and read-only packet inputs stay intact.
    again = ops.fit(packet, ptx, eta, ridge, 1e-4, mode, False, True, 64 * 1024**2)
    for a, c in zip(output[:5], again[:5], strict=False):
        torch.testing.assert_close(a, c, rtol=0, atol=0)
    torch.testing.assert_close(output[7], again[7], rtol=0, atol=0)
    torch.testing.assert_close(packet[12].cpu(), initial, rtol=0, atol=0)
    torch.testing.assert_close(packet[13].cpu(), table, rtol=0, atol=0)
    torch.testing.assert_close(packet[14].cpu(), average, rtol=0, atol=0)
    return dict(
        kind=kind,
        inverse_mode=mode,
        errors=errors,
        repeated_execution_bitexact=True,
        original_state_preserved=True,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("kind", ["mixed", "numeric_only", "categorical_only", "extreme_logits"])
@pytest.mark.parametrize("mode", [0, 1, 2])
def test_native_updates_dense_oracle(dtype, kind, mode):
    fixture(kind, mode, dtype)


def test_householder_inverse_avoids_subtractive_cancellation():
    value = torch.ones(2, dtype=torch.float64, device="cuda")
    qr, tau = torch.geqrf(value.new_tensor([[1e5], [0.0]]))
    factor = CurvatureFactor(
        value, qr, tau, value.new_ones((1, 1)), value.new_tensor([1 / (1 + 1e10)])
    )
    torch.testing.assert_close(
        factor.solve_stable(value.new_tensor([1.0, 0.0])),
        value.new_tensor([1 / (1 + 1e10), 0.0]),
        rtol=1e-14,
        atol=0.0,
    )


def test_fp64_curvature_application_matches_dense_solve_without_householder(monkeypatch):
    low = torch.tensor([[3.0], [4.0]], dtype=torch.float64, device="cuda")
    qr, tau = torch.geqrf(low)
    factor = CurvatureFactor(
        low.new_ones(2), qr, tau, low.new_ones((1, 1)), low.new_tensor([1 / 26])
    )
    matrix = torch.eye(2, dtype=low.dtype, device=low.device) + low @ low.T

    def forbidden(*args, **kwargs):
        raise AssertionError("FP64 application must not dispatch Householder transforms")

    monkeypatch.setattr(torch, "ormqr", forbidden)
    for rhs in [low.new_tensor([1.0, -1.0]), low.new_tensor([0.25, 2.0])]:
        torch.testing.assert_close(
            factor.solve(rhs), torch.linalg.solve(matrix, rhs), rtol=1e-13, atol=1e-14
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_native_chunk_boundaries_preserve_trajectory(dtype):
    # Reuse one known draw sequence; starting the next native call must carry
    # all three persistent states, including the scalar loss-gradient cache.
    packet, _ = _fixture_packet(dtype)
    ops, ptx = library(dtype)
    full = ops.fit(packet, ptx, 0.015, 0.025, 0.0, 2, False, False, 64 * 1024**2)
    carried = packet.copy()
    history = []
    initial_norm = -1.0
    for epoch in range(packet[5].shape[0]):
        carried[5] = packet[5][epoch : epoch + 1]
        output = ops.fit(
            carried, ptx, 0.015, 0.025, 0.0, 2, False, False, 64 * 1024**2, initial_norm
        )
        initial_norm = float(output[4][0, 5])
        carried[12:15] = output[:3]
        history.append(output[4])
    for first, second in zip(full[:3], output[:3], strict=True):
        assert torch.equal(first, second)
    assert torch.equal(full[4], torch.cat(history))
    assert torch.equal(full[7], output[7])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_initial_gradient_diagnostic_cannot_relax_native_stopping(dtype):
    packet, _ = _fixture_packet(dtype)
    ops, ptx = library(dtype)
    arguments = packet, ptx, 0.015, 0.025, 1e-30, 2, True, False, 64 * 1024**2
    small = ops.fit(*arguments, 0.0)
    large = ops.fit(*arguments, 1e100)
    assert small[5:7] == large[5:7]
    for first, second in zip(small[:4], large[:4], strict=True):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(small[4][:, :5], large[4][:, :5], rtol=0, atol=0)
    assert (small[4][:, 5] == 0).all() and (large[4][:, 5] == 1e100).all()


def _fixture_packet(dtype):
    # Small mixed design with repeated, unknown and absent category levels.
    gen = torch.Generator().manual_seed(17)
    n, pn, d = 23, 2, 7
    numbers = torch.randn(n, pn, generator=gen, dtype=dtype, device="cpu").cuda()
    ids = ((torch.arange(n) % 5) - 1)[:, None].cuda()
    centers, scales = (
        numbers.new_tensor([0.0, 0.5, 0.0, 0.0]),
        numbers.new_tensor([1.0, 0.5, 1.0, 1.0]),
    )
    labels = (torch.arange(n) % 2).to(dtype).cuda()
    rng = np.random.default_rng(71)
    schedule = torch.tensor(
        np.stack([rng.choice(n, 7, replace=False) for _ in range(6)]).reshape(3, 2, 7),
        dtype=torch.int32,
        device="cuda",
    )
    low = torch.randn(d, 3, generator=gen, dtype=dtype).cuda()
    diagonal = low.square().sum(1) + 1
    factor, _ = low_rank_diagonal(diagonal, low)
    initial = numbers.new_zeros(d)
    packet = [
        numbers,
        ids,
        centers,
        scales,
        labels,
        schedule,
        factor.inverse_root,
        column(factor.reflectors),
        factor.tau,
        column(factor.rotations),
        factor.ratios - 1,
        column(factor.basis()),
        initial,
        numbers.new_zeros(n),
        initial.clone(),
    ]
    return packet, labels


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_production_saga_bounded_schedule_private_rng_and_online_only(monkeypatch, dtype):
    from tabgen_eval.execution.torch.cuda.classifier import newton, saga, solver
    from tabgen_eval.execution.torch.cuda.classifier.data import Design, Fold

    packet, labels = _fixture_packet(dtype)
    design = Design(*packet[:4])
    fold = Fold(design, design, labels, labels, "sketchysaga", "diagonal", 64 * 1024**2)
    ops, ptx = operators(design.numbers)
    captured = []
    captured_initial = []

    def native_fit(*args):
        captured.append(args[0][5].cpu().clone())
        result = ops.saga_fit(*args)
        captured_initial.append((args[-1], float(result[4][0, 5])))
        return result

    monkeypatch.setattr(saga, "operators", lambda _: (SimpleNamespace(saga_fit=native_fit), ptx))

    def forbidden(*_):
        raise AssertionError("No exact full-gradient or objective diagnostics may run")

    monkeypatch.setattr(solver, "derivatives", forbidden)
    monkeypatch.setattr(newton, "fit", forbidden)
    np_state = np.random.get_state()
    cpu_state = torch.random.get_rng_state().clone()
    gpu_state = torch.cuda.get_rng_state().clone()
    seed = 1234
    options = dict(
        ridge=1 / design.n,
        saga_max_epochs=17,
        saga_batch_size=7,
        saga_rank=3,
        tolerance=0.0,
    )
    weights, state = saga.fit(fold, options, seed)
    assert weights.dtype == dtype
    assert state["epochs_completed"] == 17 and state["iterations"] == 68
    assert [x.shape[0] for x in captured] == [16, 1]
    assert captured_initial[0][0] == -1.0
    assert captured_initial[1][0] == captured_initial[0][1]
    assert (
        captured_initial[1][1]
        == state["initial_online_gradient_infinity"]
        == captured_initial[0][1]
    )
    rng = np.random.default_rng(seed ^ 0xA617)
    expected = np.stack([rng.choice(design.n, 7, replace=False) for _ in range(68)]).reshape(
        17, 4, 7
    )
    assert np.array_equal(torch.cat(captured).numpy(), expected)
    assert state["gradient_squared"] is None
    assert state["full_gradient_evaluations"] == 0
    assert state["online_gradient_squared"] >= 0
    assert torch.equal(cpu_state, torch.random.get_rng_state())
    assert torch.equal(gpu_state, torch.cuda.get_rng_state())
    after = np.random.get_state()
    assert (
        np_state[0] == after[0]
        and np.array_equal(np_state[1], after[1])
        and np_state[2:] == after[2:]
    )
    tiny = Fold(design, design, labels, labels, "sketchysaga", "diagonal", 1)
    _, failed = saga.fit(tiny, options, seed)
    assert failed["reason"] == "memory_limit" and not failed["converged"]


def test_native_memory_guard_raises_typed_resource_error():
    packet, _ = _fixture_packet(torch.float64)
    ops, ptx = library(torch.float64)
    with pytest.raises(torch.OutOfMemoryError):
        ops.fit(packet, ptx, 0.015, 0.025, 1e-4, 2, False, False, 1)


def test_saga_programming_errors_propagate_instead_of_returning_a_score(monkeypatch):
    from tabgen_eval.execution.torch.cuda.classifier import saga
    from tabgen_eval.execution.torch.cuda.classifier.data import Design, Fold

    packet, labels = _fixture_packet(torch.float64)
    design = Design(*packet[:4])
    fold = Fold(design, design, labels, labels, "sketchysaga", "diagonal", 64 * 1024**2)

    def incompatible_kernel(*_, **kwargs):
        raise RuntimeError("CUDA kernel ABI mismatch")

    monkeypatch.setattr(saga, "build", incompatible_kernel)
    with pytest.raises(RuntimeError, match="ABI mismatch"):
        saga.fit(fold, dict(ridge=1 / design.n, tolerance=1e-4), 17)


def test_nystrom_failure_reports_diagonal_rank_zero_and_keeps_its_exact_inverse():
    from tabgen_eval.execution.torch.cuda.classifier import preconditioner

    diagonal = torch.linspace(0.5, 2, 12, dtype=torch.float64, device="cuda")
    omega = torch.eye(12, dtype=diagonal.dtype, device="cuda")[:, :3].contiguous()
    ops, _ = operators(diagonal)
    # A negative-definite core exercises native POTRF failure and its genuine
    # numerical-only fallback. Python linalg mocks cannot reach native C++.
    root, qr, tau, rotation, ratio, basis, residual, fallback = ops.saga_factor(
        diagonal, omega, -omega, 1e-4, 64 * 1024**2
    )
    factor = preconditioner.CurvatureFactor(root, qr, tau, rotation, ratio)
    factor._basis = basis
    assert fallback.item() == 1
    assert torch.equal(residual, diagonal)
    rhs = diagonal.new_ones(12)
    torch.testing.assert_close(factor.solve(rhs), rhs / diagonal, rtol=1e-13, atol=1e-13)


def test_fp32_factor_and_basis_ignore_ambient_tf32_and_preserve_policy():
    from tabgen_eval.execution.torch.cuda.classifier import preconditioner
    from tabgen_eval.execution.torch.cuda.classifier.data import Design

    gen = torch.Generator().manual_seed(89)
    numbers = torch.randn((384, 128), generator=gen, dtype=torch.float32).cuda()
    ids = (torch.arange(384) % 128)[:, None].cuda()
    design = Design(numbers, ids, numbers.new_zeros(128), numbers.new_ones(128))
    previous = torch.get_float32_matmul_precision()
    factors = []
    try:
        for policy in ["highest", "high"]:
            torch.set_float32_matmul_precision(policy)
            assert torch.backends.cuda.matmul.allow_tf32 == (policy == "high")
            factor, rate, metadata = preconditioner.build(
                design,
                1 / design.n,
                64,
                torch.Generator(device="cuda").manual_seed(91),
                128 * 1024**2,
            )
            basis = factor.basis()
            assert torch.get_float32_matmul_precision() == policy
            assert torch.backends.cuda.matmul.allow_tf32 == (policy == "high")
            assert metadata["factorization_fallbacks"] == 0
            factors.append((factor, rate, basis))
        first, second = factors
        for name in ["inverse_root", "reflectors", "tau", "rotations", "ratios"]:
            assert torch.equal(getattr(first[0], name), getattr(second[0], name))
        assert first[1] == second[1]
        assert torch.equal(first[2], second[2])
    finally:
        torch.set_float32_matmul_precision(previous)
    assert torch.get_float32_matmul_precision() == previous


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("column_a", [False, True])
@pytest.mark.parametrize("column_b", [False, True])
def test_pedantic_gemm_row_column_layouts_match_independent_double_product(
    dtype, column_a, column_b
):
    from tabgen_eval.execution.torch.cuda.extension import library as cuda_library

    gen = torch.Generator().manual_seed(109)
    a = torch.randn((13, 17), generator=gen, dtype=dtype).cuda()
    b = torch.randn((17, 11), generator=gen, dtype=dtype).cuda()
    if column_a:
        a = column(a)
    if column_b:
        b = column(b)
    ac, bc = a.clone(), b.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        actual = cuda_library().gemm(a, b)
    stream.synchronize()
    expected = a.cpu().double() @ b.cpu().double()
    tol = 2e-6 if dtype == torch.float32 else 1e-13
    torch.testing.assert_close(actual.cpu().double(), expected, rtol=tol, atol=tol)
    assert actual.dtype == dtype and actual.stride() == (1, 13)
    assert torch.equal(a, ac) and torch.equal(b, bc)


def test_native_online_gradient_stopping_absolute_equality_and_nonfinite():
    packet, _ = _fixture_packet(torch.float64)
    ops, ptx = library(torch.float64)
    limit = 64 * 1024**2
    one = packet.copy()
    one[5] = packet[5][:1]
    observed = ops.fit(one, ptx, 0.015, 0.025, 0.0, 2, False, False, limit)
    norm, initial = observed[4][0, 4:].tolist()
    assert norm != initial
    # Capturing first g0 includes the nonzero initial cached state and ridge;
    # the independent dense fixture validates the numeric values directly.
    absolute = ops.fit(packet, ptx, 0.015, 0.025, norm, 2, True, False, limit)
    assert absolute[5] == 1 and absolute[4][0, 4] == norm
    for bad in [float("nan"), float("inf")]:
        nonfinite = packet.copy()
        nonfinite[14] = packet[14].clone()
        nonfinite[14][0] = bad
        result = ops.fit(nonfinite, ptx, 0.015, 0.025, 1e30, 2, True, False, limit)
        assert result[4][0, 2] == 0
    diagnostic_only = ops.fit(packet, ptx, 0.015, 0.025, 1e30, 2, True, False, limit, float("inf"))
    assert diagnostic_only[4][0, 2] == 1
    assert torch.isinf(diagnostic_only[4][0, 5])
