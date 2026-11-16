"""Real-real draw orchestration agrees across existing CPU and CUDA executors."""

import numpy as np
import pandas as pd
import pytest

from tabgen_eval.schema import Schema
from tests.support.evaluation import reference_scores as calibrate
from tests.support.evaluation import summaries
from tests.support.options import make_options

pytestmark = [pytest.mark.unit, pytest.mark.gpu]


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_calibration_cuda_matches_cpu_for_sample_and_fitted_metrics(dtype):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    pytest.importorskip("triton")
    rng = np.random.default_rng(9107)
    real = pd.DataFrame(
        {
            "x": rng.normal(size=383),
            "y": rng.normal(size=383),
            "c": rng.choice(["A", " A ", "nan", "<MISSING>"], 383),
        }
    )
    real.loc[::31, "x"] = np.nan
    real.loc[::37, "c"] = None
    schema = Schema(numeric=["x", "y"], categorical=["c"])
    args = dict(
        schema=schema,
        metrics=["shape", "trend", "conditional_fidelity", "prdc", "linear_c2st"],
        n_replicates=3,
        seed=318,
    )
    cpu = calibrate(
        real,
        **args,
        options=make_options(device="cpu"),
    )
    cuda = calibrate(
        real,
        **args,
        options=make_options(device="cuda:0", dtype=dtype),
    )
    for name, metric in cuda.metrics.items():
        assert metric.execution["device"].startswith("cuda")
        for stat, summary in summaries(metric).items():
            np.testing.assert_allclose(
                summary.samples,
                summaries(cpu.metrics[name])[stat].samples,
                rtol=1e-5,
                atol=2e-4,
                equal_nan=True,
            )
    assert all(draw["converged"] for draw in cuda.metrics["linear_c2st"].diagnostics)
