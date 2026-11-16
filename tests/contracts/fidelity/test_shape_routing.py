"""Host-only Shape crossover and workspace policy, without importing PyTorch."""

from __future__ import annotations

import pytest

from tabgen_eval.execution.runtime.routing import Workload
from tabgen_eval.metrics import Shape

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("rows", "bootstrap", "resident", "cuda"),
    [
        (1000, 0, False, False),
        (5000, 2, False, False),
        (20000, 0, False, False),
        (20000, 2, False, True),
        (20000, 0, True, True),
        (50000, 0, False, True),
    ],
)
def test_shape_routing_uses_effective_workload(rows, bootstrap, resident, cuda):
    workload = Workload(rows, 50, 50, (10,) * 50, 200_000, 200_000, rows, bootstrap, resident)
    selected, reason = workload.cuda_candidate(Shape())
    assert selected is cuda
    assert reason == ("workload_favors_cuda" if cuda else "small_workload_cpu")


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_shape_oversized_auto_draw_stays_cpu_without_gpu_probe(dtype, monkeypatch):
    torch = pytest.importorskip("torch")
    from tabgen_eval.execution.torch.context import TorchContext

    workload = Workload(100_000, 50, 50, (15_000,) * 50, 200_000, 200_000, 100_000)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("unexpected CUDA probe"))
    assert TorchContext.select_device(Shape(dtype=dtype, gpu_workspace_mb=64), workload) == (
        "cpu",
        "gpu_workspace_too_small",
    )
