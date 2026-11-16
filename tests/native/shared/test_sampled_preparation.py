"""Policy boundaries and independent coherent-sample curvature oracles."""

import numpy as np
import pytest

from tabgen_eval.metrics.c2st import LinearC2ST


@pytest.mark.parametrize("value", [True, 0, -1, 1.5])
def test_preparation_rows_require_positive_integer(value):
    with pytest.raises(ValueError, match="saga_preparation_rows"):
        LinearC2ST(saga_preparation_rows=value)


@pytest.mark.parametrize("value", [0, 1, "auto", None])
def test_sampled_preparation_requires_boolean(value):
    with pytest.raises(ValueError, match="saga_sampled_preparation"):
        LinearC2ST(saga_sampled_preparation=value)


def test_preparation_options_forward_without_point_bootstrap_difference():
    metric = LinearC2ST(saga_sampled_preparation=False, saga_preparation_rows=16384)
    for point in (True, False):
        options = metric.native_options(point=point)
        assert options["saga_sampled_preparation"] is False
        assert options["saga_preparation_rows"] == 16384


def test_cpu_native_policy_and_large_fit_keep_full_training_objective():
    import torch

    from tabgen_eval.execution.torch.cpu.extension import library

    ops = library()
    for n in (499999, 500000, 500001):
        assert ops.classifier_saga_preparation_rows(n) == (n if n < 500000 else 4096)
        assert ops.classifier_saga_preparation_rows(n, False) == n
    n, seed = 500000, 19
    generator = torch.Generator().manual_seed(68)
    numbers = torch.randn(n, 4, dtype=torch.float64, generator=generator)
    identities = torch.empty((n, 0), dtype=torch.int64)
    centers = scales = torch.empty(0, dtype=torch.float64)
    labels = (numbers[:, 0] + 0.3 * numbers[:, 1] > 0).to(torch.float64)
    before_rng = torch.random.get_rng_state().clone()
    weights, diagnostic, _ = ops.classifier_saga_fit(
        numbers,
        identities,
        centers,
        scales,
        labels,
        1 / n,
        1e-12,
        1,
        4096,
        3,
        seed,
        128 * 1024**2,
        4,
        True,
        True,
        4096,
    )
    assert torch.equal(before_rng, torch.random.get_rng_state())
    assert torch.isfinite(weights).all()
    assert weights[0] > 0
    # Standalone ABI: 11 fit fields + 13 guard fields + coherent prep metadata.
    assert diagnostic[24:27].tolist() == [n, 4096, 1]
    assert diagnostic[35:37].tolist() == [n, 4096]
    assert diagnostic[2] == (n + 4095) // 4096


@pytest.mark.gpu
def test_native_threshold_coherence_and_private_sampling():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from tabgen_eval.execution.torch.cuda.classifier.data import Design
    from tabgen_eval.execution.torch.cuda.classifier.device import operators
    from tabgen_eval.execution.torch.cuda.classifier.preconditioner import build

    n, seed, rank = 500000, 91, 3
    data_generator = torch.Generator().manual_seed(77)
    numbers = torch.randn(n, 5, dtype=torch.float64, generator=data_generator).cuda()
    identities = torch.stack((torch.arange(n) % 5, 5 + torch.arange(n) % 3), dim=1).cuda()
    identities[::29] = -1
    centers = torch.linspace(0, 0.6, 8, dtype=torch.float64, device="cuda")
    scales = torch.linspace(0.4, 1.2, 8, dtype=torch.float64, device="cuda")
    design = Design(numbers, identities, centers, scales)
    design.invert()
    ops, _ = operators(numbers)
    for count in (499999, 500000, 500001):
        assert ops.saga_preparation_policy(count) == (count if count < 500000 else 4096)
        assert ops.saga_preparation_policy(count, False) == count
    assert ops.saga_preparation_policy(n, True, n + 1) == n
    rows = ops.saga_preparation_sample(n, 4096, seed)
    assert rows.unique().numel() == 4096
    assert rows.min() >= 0 and rows.max() < n
    assert torch.equal(rows, ops.saga_preparation_sample(n, 4096, seed))
    assert not torch.equal(rows, ops.saga_preparation_sample(n, 4096, seed + 1))
    sample = Design(numbers[rows.cuda()], identities[rows.cuda()], centers, scales)
    sample.invert()
    initial_rng = torch.cuda.get_rng_state().clone()
    generator = torch.Generator(device="cuda").manual_seed(seed)
    factor, _, metadata = build(design, 1 / n, rank, generator, 256 * 1024**2)
    reference_generator = torch.Generator(device="cuda").manual_seed(seed)
    reference, _, reference_metadata = build(
        sample, 1 / n, rank, reference_generator, 256 * 1024**2, sampled=False
    )
    # Both low-rank and diagonal curvature must use this same sample, including
    # absent categories and the full-data ridge. Independent sample construction
    # would change either represented inverse or sketch/power RNG trajectory.
    rhs = torch.linspace(-1, 1, design.d, dtype=torch.float64, device="cuda")
    torch.testing.assert_close(factor.solve(rhs), reference.solve(rhs), rtol=1e-12, atol=1e-12)
    assert torch.equal(generator.get_state(), reference_generator.get_state())
    assert torch.equal(initial_rng, torch.cuda.get_rng_state())
    assert metadata["sampled_preparation"]
    assert metadata["preparation_training_rows"] == n
    assert metadata["preparation_rows"] == metadata["curvature_calibration_rows"] == 4096
    assert metadata["curvature_calibration_population"] == n
    assert metadata["curvature_calibration_batch"] == 4096
    assert not reference_metadata["sampled_preparation"]
    assert reference_metadata["preparation_training_rows"] == 4096
    # Reproduce the diagnostic hash without any package sampling implementation.
    value = 1469598103934665603
    for row in rows.tolist():
        value = ((value ^ row) * 1099511628211) & ((1 << 64) - 1)
    assert metadata["preparation_sample_hash"] == value & ((1 << 53) - 1)
    assert np.isfinite(metadata["learning_rate"]) and metadata["learning_rate"] > 0
