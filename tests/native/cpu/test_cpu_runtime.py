"""CPU thread, precision, and sample-boundary isolation."""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tabgen_eval import Schema
from tabgen_eval.metrics import Conditional, Shape, Trend

from .oracles import design, fit

pytestmark = pytest.mark.unit


def test_cpu_native_thread_budgets_and_autocast_do_not_change_global_settings(torch, ops):
    x, y = design(torch)
    if sys.platform == "darwin":
        from tabgen_eval.execution.torch.cpu.extension import _mac_openmp

        _, runtime = _mac_openmp(torch)
        lib = ctypes.CDLL(str(runtime))
        maximum = lib.omp_get_max_threads
    else:
        lib = ctypes.CDLL(str(Path(torch.__file__).parent / "lib" / "libtorch_cpu.so"))
        maximum = lib.MKL_Get_Max_Threads
    maximum.restype = ctypes.c_int
    before = (torch.get_num_threads(), maximum(), torch.get_float32_matmul_precision())
    for threads in [1, 4]:
        for solver in ["direct", "sketchysaga"]:
            fit(ops, x, y, solver, threads=threads)
            assert (
                torch.get_num_threads(),
                maximum(),
                torch.get_float32_matmul_precision(),
            ) == before
    frame = pd.DataFrame(x[:, :2].numpy(), columns=["x", "y"])
    schema = Schema(numeric=["x", "y"])
    for cls in [Shape, Trend, Conditional]:
        metric = cls(device="cpu", dtype="float32")
        expected = metric.compute(frame, frame, schema, rng=None)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = metric.compute(frame, frame, schema, rng=None)
        np.testing.assert_allclose(
            list(actual.values()), list(expected.values()), rtol=0, atol=0, equal_nan=True
        )
    assert (torch.get_num_threads(), maximum(), torch.get_float32_matmul_precision()) == before


def test_sample_operator_validates_consumed_category_codes_before_parallel_work(torch, ops):
    numeric = torch.empty((5, 0), dtype=torch.float64)
    codes = torch.tensor([[0], [1], [99], [0], [1]], dtype=torch.int64)
    mask = torch.ones(5, dtype=torch.bool)

    def compute(draw):
        return ops.samples(
            numeric,
            codes,
            numeric,
            codes,
            mask,
            mask,
            draw,
            draw,
            [2],
            "shape",
            5,
            10,
            10,
            0.05,
            4,
        )

    torch.testing.assert_close(
        compute(torch.tensor([[0, 1]])), torch.ones((1, 1), dtype=torch.float64)
    )
    assert compute(torch.empty((0, 2), dtype=torch.int64)).shape == (0, 1)
    with pytest.raises(RuntimeError, match="outside declared cardinality"):
        compute(torch.tensor([[0, 2], [1, 0]]))
