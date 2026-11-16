"""Cached executors must not keep evaluation inputs alive through reference cycles."""

import gc
import weakref

import pandas as pd
import pytest

from tabgen_eval import Schema
from tabgen_eval.execution.torch.context import TorchContext
from tabgen_eval.metrics import Concentration, Shape
from tabgen_eval.preprocessing.columns import PreparedColumns

pytestmark = pytest.mark.unit


def context():
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    return TorchContext.from_columns(PreparedColumns.prepare(frame, frame, Schema(numeric=["x"])))


@pytest.mark.parametrize("route", ["cached_cpu", "cached_concentration", "cached_memorization"])
def test_cached_executor_releases_context_and_inputs_without_cyclic_gc(monkeypatch, route):
    from tabgen_eval.execution.torch.distance.executor import DistanceExecutor
    from tabgen_eval.metrics import DCR

    enabled = gc.isenabled()
    gc.disable()
    try:
        owner = context()
        if route == "cached_cpu":
            import tabgen_eval.execution.torch.cpu.executor as cpu

            monkeypatch.setattr(cpu, "library", lambda: object())
            executor = owner._executor(Shape(device="cpu"))
        elif route == "cached_concentration":
            executor = owner._executor(Concentration(device="cpu"))
        else:
            monkeypatch.setattr(DistanceExecutor, "compute_memorization_group", lambda *a, **k: [])
            owner.compute_memorization_group([DCR(device="cpu")], [], [])
            executor = owner._distances
        owner_ref = weakref.ref(owner)
        input_ref = weakref.ref(owner.reference.numeric)
        del owner
        assert owner_ref() is None
        assert input_ref() is None
        with pytest.raises(ReferenceError):
            _ = executor.context.inputs
    finally:
        if enabled:
            gc.enable()


def test_explicit_standalone_executor_still_owns_its_context():
    from tabgen_eval.execution.torch.distance.executor import DistanceExecutor

    executor = DistanceExecutor(context())
    assert executor.context.reference.n_rows == 3
