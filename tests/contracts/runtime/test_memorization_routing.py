"""Automatic placement accounts for independently sized memorization roles."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tabgen_eval import Schema
from tabgen_eval.execution.runtime.dispatch import ExecutionSession
from tabgen_eval.metrics import DCR
from tabgen_eval.preprocessing.columns import HostTable, PreparedColumns

pytestmark = pytest.mark.unit


def table(rows, numeric=100, categorical=0, *, complete=True):
    """Describe large source pools with broadcast views over tiny owned buffers."""
    return HostTable(
        np.broadcast_to(np.zeros((numeric, 1)), (numeric, rows)),
        np.broadcast_to(np.zeros((categorical, 1), dtype=np.int64), (categorical, rows)),
        np.broadcast_to(np.array([complete]), (rows,)),
    )


def session(training=50000, generated=50000, holdout=1000000, *, complete=True):
    columns = PreparedColumns(
        table(training),
        table(generated),
        tuple(f"x{i}" for i in range(100)),
        (),
        (),
        False,
        holdout=None if holdout is None else table(holdout, complete=complete),
    )
    return ExecutionSession(
        pd.DataFrame(),
        pd.DataFrame(),
        Schema(numeric=list(columns.numeric_names)),
        n_eff=max(training, generated),
        columns=columns,
    )


def mock_cuda(monkeypatch, free_bytes):
    torch = pytest.importorskip("torch")
    from tabgen_eval.execution.torch import context
    from tabgen_eval.execution.torch.distance import ops

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (free_bytes, free_bytes))
    monkeypatch.setattr(context, "triton_installed", lambda: True)
    monkeypatch.setattr(context, "cuda_supported", lambda torch: True)
    monkeypatch.setattr(context, "resolve_device", lambda device: torch.device(device))
    monkeypatch.setattr(ops, "cuda_kernels", lambda: None)

    def from_columns(cls, columns, *, device, **kwargs):
        tables = {
            id(table): table
            for table in (columns.reference, columns.candidate, columns.holdout)
            if table is not None
        }
        storage = sum(
            table.numeric.nbytes + table.categorical.nbytes + table.complete.nbytes
            for table in tables.values()
        )
        return SimpleNamespace(
            device=device,
            routes={},
            execution_info={"device": device, "persistent_input_bytes": storage},
        )

    monkeypatch.setattr(context.TorchContext, "from_columns", classmethod(from_columns))
    return torch


def test_holdout_storage_rejects_auto_cuda_before_tensor_transfer(monkeypatch):
    mock_cuda(monkeypatch, 512 * 1024**2)
    metric = DCR(search_method="exact", gpu_workspace_mb=4096)
    small = session(holdout=None).context_for(metric)
    large = session().context_for(metric)
    assert small.device == "cuda"
    assert large.device == "cpu"
    assert large.routes["dcr"] == "insufficient_free_cuda_memory"


def test_holdout_draw_workspace_rejects_auto_cuda_before_device_probe(monkeypatch):
    torch = mock_cuda(monkeypatch, 16 * 1024**3)

    def forbidden():
        pytest.fail("known oversized holdout draws must be rejected before probing CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    metric = DCR(search_method="exact", gpu_workspace_mb=512)
    selected = session().context_for(metric)
    assert selected.device == "cpu"
    assert selected.routes["dcr"] == "gpu_workspace_too_small"


@pytest.mark.parametrize("dtype,element", [("float32", 4), ("float64", 8)])
def test_source_storage_matches_tensor_role_ownership(dtype, element):
    evaluation = session(training=7, generated=11, holdout=13)
    evaluation._columns = replace(
        evaluation.columns,
        reference=table(7, categorical=2),
        candidate=table(11, categorical=2),
        holdout=table(13, categorical=2),
        categorical_names=("a", "b"),
        cardinalities=(1, 1),
    )
    workload = evaluation._workload(DCR())
    assert workload.persistent_bytes(dtype) == (7 + 11 + 13) * (100 * element + 2 * 8 + 1)
    assert workload.persistent_bytes(dtype) == evaluation.columns.to_tensors(dtype=dtype).nbytes
    shared = replace(evaluation.columns, candidate=evaluation.columns.reference)
    evaluation._columns = shared
    workload = evaluation._workload(DCR())
    assert workload.persistent_bytes(dtype) == (7 + 13) * (100 * element + 2 * 8 + 1)
    assert workload.persistent_bytes(dtype) == shared.to_tensors(dtype=dtype).nbytes
    assert workload.draw_rows == (7, 7, 13)


def test_missing_holdout_still_reserves_selected_index_filtering(monkeypatch):
    mock_cuda(monkeypatch, 16 * 1024**3)
    evaluation = session(training=5000, generated=9000, holdout=1000000, complete=False)
    metric = DCR(search_method="exact", gpu_workspace_mb=16)
    workload = evaluation._workload(metric)
    assert workload.draw_rows == (5000, 9000, 1000000)
    assert workload.usable_rows == (5000, 9000, 0)
    assert workload.minimum_workspace(metric) > metric.gpu_workspace_mb * 1024**2
    selected = evaluation.context_for(metric)
    assert selected.device == "cpu"
    assert selected.routes["dcr"] == "gpu_workspace_too_small"
