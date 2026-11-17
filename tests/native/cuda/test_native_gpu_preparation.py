"""Independent dense oracles for native compact GPU curvature preparation."""

# Check the optional dependency before importing GPU helpers during collection.
# ruff: noqa: E402

import pytest
import torch

pytest.importorskip("triton")

from tabgen_eval.execution.torch.cuda.classifier.data import Design
from tabgen_eval.execution.torch.cuda.classifier.device import operators
from tabgen_eval.execution.torch.cuda.classifier.preconditioner import build

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


def design_fixture(dtype, kind):
    generator = torch.Generator().manual_seed(32)
    pn = 0 if kind == "categorical" else 8
    pc = 0 if kind == "numeric" else 2
    numbers = torch.randn(128, pn, dtype=dtype, generator=generator).cuda()
    ids = torch.stack((torch.arange(128) % 5, 5 + torch.arange(128) % 3), 1)[:, :pc].cuda()
    if pc:
        ids[0] = -1
    k = 8 if pc else 0
    centers = torch.linspace(0, 0.6, k, dtype=dtype, device="cuda")
    scales = torch.linspace(0.25, 1.25, k, dtype=dtype, device="cuda")
    design = Design(numbers, ids, centers, scales)
    design.invert()
    categorical = numbers.new_zeros((128, k))
    if k:
        for column in range(pc):
            valid = ids[:, column] >= 0
            categorical[torch.arange(128, device="cuda")[valid], ids[valid, column]] = 1
        categorical = (categorical - centers) / scales
    dense = torch.cat((numbers, categorical, numbers.new_ones((128, 1))), dim=1)
    return design, dense


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("kind", ["numeric", "mixed", "categorical"])
def test_native_curvature_matches_independent_dense_inverse_and_rate(dtype, kind):
    design, dense = design_fixture(dtype, kind)
    ridge, rank, seed = 0.025, 3, 31
    global_state = torch.cuda.get_rng_state().clone()
    generator = torch.Generator(device="cuda").manual_seed(seed)
    factor, rate, metadata = build(design, ridge, rank, generator, 128 * 1024**2)
    assert torch.equal(global_state, torch.cuda.get_rng_state())
    assert metadata["factorization_fallbacks"] == 0

    # Build the represented P from dense H0 and an independent eigendecomposition,
    # without the compact kernels or inverse-factor construction under test.
    reference_generator = torch.Generator(device="cuda").manual_seed(seed)
    omega = torch.linalg.qr(
        torch.randn(design.d, rank, device="cuda", dtype=dtype, generator=reference_generator)
    ).Q
    majorant = dense.T @ dense / (4 * design.n)
    majorant.diagonal()[:-1].add_(ridge)
    images = majorant @ omega
    shift = max(float(images.norm()) * torch.finfo(dtype).eps, torch.finfo(dtype).tiny)
    shifted = images + shift * omega
    core = omega.T @ shifted
    approximate = shifted @ torch.linalg.solve((core + core.T) / 2, shifted.T)
    eigenvalues, eigenvectors = torch.linalg.eigh((approximate + approximate.T) / 2)
    low = eigenvectors[:, -rank:] * (eigenvalues[-rank:] - shift).clamp(min=0).sqrt()
    diagonal = majorant.diagonal()
    residual = torch.maximum(
        diagonal - low.square().sum(1), diagonal * (1e-6 if dtype == torch.float32 else 1e-10)
    )
    represented = torch.diag(residual) + low @ low.T
    rhs = torch.linspace(-1, 1, design.d, dtype=dtype, device="cuda")
    tolerance = 4e-5 if dtype == torch.float32 else 2e-12
    torch.testing.assert_close(
        factor.solve(rhs), torch.linalg.solve(represented, rhs), rtol=tolerance, atol=tolerance
    )

    direction = torch.randn(design.d, device="cuda", dtype=dtype, generator=reference_generator)
    direction /= direction.norm()
    for _ in range(30):
        image = (dense.T @ dense / (4 * design.n)) @ direction
        solved = torch.linalg.solve(represented, image)
        eigenvalue = float((image @ solved) / (direction @ image))
        direction = solved / solved.norm()
    estimate = 2 * eigenvalue
    inverse = torch.linalg.inv(represented)
    row_maximum = float(((dense @ inverse) * dense).sum(1).max() / 4)
    ridge_bound = ridge * float(factor.inverse_root[:-1].square().max())
    expected_rate = 1 / (estimate + ridge_bound)
    assert rate == pytest.approx(expected_rate, rel=tolerance)
    assert metadata["curvature_row_maximum_raw"] == pytest.approx(row_maximum, rel=tolerance)
    assert metadata["curvature_data_used"] == pytest.approx(estimate, rel=tolerance)
    assert metadata["curvature_ridge_bound"] == pytest.approx(ridge_bound, rel=tolerance)
    assert metadata["curvature_calibration_rows"] == design.n
    assert metadata["curvature_calibration_batch"] == design.n
    assert torch.equal(generator.get_state(), reference_generator.get_state())


def test_rate_uses_actual_batch_with_unchanged_preconditioner_and_private_calibration():
    design, dense = design_fixture(torch.float64, "mixed")
    results = [
        build(
            design,
            0.025,
            3,
            torch.Generator(device="cuda").manual_seed(31),
            128 * 1024**2,
            batch=batch,
        )
        for batch in (1, 8, 128, 4096)
    ]
    reference, _, reference_metadata = results[0]
    for batch, (factor, rate, metadata) in zip((1, 8, 128, 4096), results, strict=True):
        torch.testing.assert_close(factor.inverse_root, reference.inverse_root, rtol=0, atol=0)
        torch.testing.assert_close(factor.basis(), reference.basis(), rtol=0, atol=0)
        assert metadata["curvature_data_raw"] == reference_metadata["curvature_data_raw"]
        assert (
            metadata["curvature_row_maximum_raw"] == reference_metadata["curvature_row_maximum_raw"]
        )
        b = min(batch, design.n)
        a = (design.n - b) / (b * (design.n - 1))
        c = design.n * (b - 1) / (b * (design.n - 1))
        expected = 1 / (
            c * metadata["curvature_data_used"]
            + a * metadata["curvature_row_maximum_used"]
            + metadata["curvature_ridge_bound"]
        )
        assert rate == pytest.approx(expected, rel=1e-14)
        assert metadata["curvature_calibration_batch"] == b
    assert [r[1] for r in results] == sorted(r[1] for r in results)


def test_native_preparation_owns_math_and_preserves_current_stream(monkeypatch):
    design, _ = design_fixture(torch.float64, "mixed")
    operators(design.numbers)
    inputs = [
        value.clone()
        for value in (design.numbers, design.identities, design.centers, design.scales)
    ]

    def forbidden(*args, **kwargs):
        raise AssertionError("Python math was called during native preparation")

    for name in ("qr", "svd", "cholesky", "solve_triangular"):
        monkeypatch.setattr(torch.linalg, name, forbidden)
    for name in ("geqrf", "orgqr", "ormqr", "randn"):
        monkeypatch.setattr(torch, name, forbidden)
    for name in ("mv", "transpose_mv", "diagonal"):
        monkeypatch.setattr(design, name, forbidden)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        factor, rate, _ = build(
            design, 0.025, 3, torch.Generator(device="cuda").manual_seed(31), 128 * 1024**2
        )
        result = factor.solve(design.numbers.new_ones(design.d))
        assert torch.cuda.current_stream() == stream
    stream.synchronize()
    assert rate > 0 and bool(torch.isfinite(result).all())
    for actual, original in zip(
        (design.numbers, design.identities, design.centers, design.scales), inputs, strict=True
    ):
        assert torch.equal(actual, original)


def test_native_workspace_guard_precedes_random_draws():
    design, _ = design_fixture(torch.float64, "mixed")
    generator = torch.Generator(device="cuda").manual_seed(31)
    initial = generator.get_state()
    with pytest.raises(torch.OutOfMemoryError, match="curvature exceeds model workspace"):
        build(design, 0.025, 3, generator, 8)
    assert torch.equal(initial, generator.get_state())


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_native_numerical_factor_failure_uses_exact_diagonal_inverse(dtype):
    from tabgen_eval.execution.torch.cuda.classifier.preconditioner import CurvatureFactor

    diagonal = torch.linspace(0.5, 2, 12, dtype=dtype, device="cuda")
    omega = torch.eye(12, dtype=dtype, device="cuda")[:, :3].contiguous()
    ops, _ = operators(diagonal)
    # This invalid sketch has a negative-definite core: native POTRF genuinely
    # fails, unlike a Python monkeypatch that cannot reach C++ preparation.
    root, qr, tau, rotation, ratio, basis, residual, fallback = ops.saga_factor(
        diagonal, omega, -omega, 1e-4, 128 * 1024**2
    )
    assert fallback.item() == 1
    factor = CurvatureFactor(root, qr, tau, rotation, ratio)
    factor._basis = basis
    rhs = diagonal.new_ones(12)
    torch.testing.assert_close(factor.solve(rhs), rhs / diagonal)
    assert torch.equal(residual, diagonal)


def test_native_factor_programming_errors_are_not_diagonal_fallbacks():
    prototype = torch.ones(12, dtype=torch.float64, device="cuda")
    ops, _ = operators(prototype)
    omega = torch.eye(12, dtype=prototype.dtype, device="cuda")[:, :3].contiguous()
    with pytest.raises(RuntimeError, match="invalid native Nyström factor inputs"):
        ops.saga_factor(prototype[:-1], omega, omega, 1e-4, 128 * 1024**2)
