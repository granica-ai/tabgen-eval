"""CUDA paired evaluation follows mode-specific resolution and stable draw streams."""

import numpy as np
import pandas as pd
import pytest

from tabgen_eval import Schema
from tests.support.evaluation import paired_report as evaluate
from tests.support.options import make_options

pytestmark = [pytest.mark.unit, pytest.mark.gpu]


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("generated_size,both_resolution", [(191, 128), (83, 41)])
def test_cuda_uncertainty_modes_use_matched_points_and_reference_draws(
    dtype, generated_size, both_resolution
):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    pytest.importorskip("triton")
    rng = np.random.default_rng(842)
    real = pd.DataFrame(
        {
            "x": rng.normal(size=257),
            "y": rng.normal(size=257),
            "z": rng.normal(size=257),
            "c": rng.choice(["a", "b", "rare"], 257),
        }
    )
    generated = pd.DataFrame(
        {
            "x": rng.normal(0.6, size=generated_size),
            "y": rng.normal(size=generated_size),
            "z": rng.normal(size=generated_size),
            "c": rng.choice(["a", "b"], generated_size),
        }
    )
    schema = Schema(numeric=["x", "y", "z"], categorical=["c"])
    metrics = ["shape", "trend", "conditional_fidelity", "prdc", "linear_c2st"]
    reports = {}
    for mode in ("none", "reference", "both"):
        reports[mode] = evaluate(
            real,
            generated,
            schema,
            metrics=metrics,
            details=True,
            options=make_options(
                device="cuda:0",
                dtype=dtype,
                uncertainty=mode,
                n_replicates=3,
                seed=73,
            ),
        )
        expected = both_resolution if mode == "both" else min(generated_size, 128)
        assert reports[mode].provenance["n_res"] == expected
        assert all(stat.n_eff == expected for _, stat in reports[mode].iter_stats())
        assert reports[mode].details("shape") is not None
    for metric in reports["none"].results:
        for name, point in metric.stats.items():
            key = f"{metric.name}/{name}"
            reference, both = reports["reference"][key], reports["both"][key]
            np.testing.assert_equal([point.value, point.ideal], [reference.value, reference.ideal])
            if generated_size > 128:
                np.testing.assert_equal([point.value, point.ideal], [both.value, both.ideal])
            assert point.generated_distribution is point.reference_distribution is None
            assert reference.generated_distribution is None
            assert reference.reference_distribution.requested_count == 3
            assert both.generated_distribution.requested_count == 3
            if generated_size > 128:
                np.testing.assert_equal(
                    reference.reference_distribution.samples, both.reference_distribution.samples
                )
    for mode, report in reports.items():
        fits = report.provenance["classifier_diagnostics"]["linear_c2st"]
        assert all(fit["point"] and fit["converged"] for fit in fits["point"].values())
        assert len(fits["replicates"].get("generated", [])) == (3 if mode == "both" else 0)
        assert len(fits["replicates"].get("real_reference", [])) == (0 if mode == "none" else 3)
    cpu = evaluate(
        real,
        generated,
        schema,
        metrics=metrics[:-1],
        options=make_options(
            device="cpu", dtype=dtype, uncertainty="both", n_replicates=3, seed=73
        ),
    )
    tolerance = 2e-5 if dtype == "float32" else 1e-10
    for metric, stat in cpu.iter_stats():
        actual = reports["both"][f"{metric.name}/{stat.name}"]
        for values, expected in (
            ([actual.value, actual.ideal], [stat.value, stat.ideal]),
            (actual.generated_distribution.samples, stat.generated_distribution.samples),
            (actual.reference_distribution.samples, stat.reference_distribution.samples),
        ):
            np.testing.assert_allclose(
                values, expected, rtol=tolerance, atol=tolerance, equal_nan=True
            )
