"""Intrinsic Conditional applicability agrees across scalar and descriptive output."""

import numpy as np
import pandas as pd
import pytest

from tabgen_eval.execution.python.conditional import compute_conditional
from tabgen_eval.execution.python.stats import safe_nanmean
from tabgen_eval.metrics import Conditional
from tabgen_eval.preprocessing.columns import PreparedColumns
from tabgen_eval.schema import Schema

pytestmark = pytest.mark.unit


@pytest.fixture(
    params=[
        ("reference", "float64"),
        ("cpu", "float32"),
        ("cpu", "float64"),
        pytest.param(("cuda", "float32"), marks=pytest.mark.gpu),
        pytest.param(("cuda", "float64"), marks=pytest.mark.gpu),
    ]
)
def backend(request):
    device, dtype = request.param
    if device != "reference":
        torch = pytest.importorskip("torch")
        if device == "cuda":
            if not torch.cuda.is_available():
                pytest.skip("CUDA unavailable")
            pytest.importorskip("triton")
            from tests.support.backends import cuda_device

            device = cuda_device(torch)
    return device, dtype


CASES = [(0, "categorical", "matched")] + [
    (width, kind, candidate)
    for width in [1, 2, 3]
    for kind in ["categorical", "numeric", "mixed"]
    for candidate in ["matched", "absent", "missing"]
    if not (width == 1 and kind == "mixed")
]


def frames(width, kind, candidate):
    reference = pd.DataFrame(index=np.arange(12))
    generated = pd.DataFrame(index=np.arange(12))
    numeric, categorical = [], []
    for index in range(width):
        name = f"x{index}"
        is_numeric = kind == "numeric" or (kind == "mixed" and index == 0)
        if is_numeric:
            numeric.append(name)
            reference[name] = np.arange(12, dtype=float)
            generated[name] = reference[name] if candidate == "matched" else 100.0
            if candidate == "missing":
                generated[name] = np.nan
        else:
            categorical.append(name)
            reference[name] = ["a"] * 8 + ["b"] * 4
            generated[name] = reference[name] if candidate == "matched" else "novel"
            if candidate == "missing":
                generated[name] = None
    # Applicability counts modeled axes, independently of other source columns.
    reference["unused"] = generated["unused"] = np.arange(12)
    return reference, generated, Schema(numeric=numeric, categorical=categorical)


@pytest.mark.parametrize("width,kind,candidate", CASES)
def test_scalar_and_details_obey_intrinsic_width_contract(backend, width, kind, candidate):
    device, dtype = backend
    reference, generated, schema = frames(width, kind, candidate)
    options = dict(min_count=1, num_bins=2, rare_threshold=0.75)
    metric = Conditional(**options)
    if device == "reference":
        values = compute_conditional(reference, generated, schema, **options)
        captured = compute_conditional(reference, generated, schema, details=True, **options)
    else:
        from tabgen_eval.execution.torch.context import TorchContext

        columns = PreparedColumns.prepare(reference, generated, schema, retain_labels=True)
        context = TorchContext.from_columns(columns, device=device, dtype=dtype)
        draw = np.arange(12)
        batch = context.compute(metric, draw[None, :], draw[None, :])
        values = {name: result[0] for name, result in batch.items()}
        captured = context.compute_detailed(metric, draw, draw)

    columns = captured.details.tables["columns"].columns
    slices = captured.details.tables["slices"].columns
    np.testing.assert_array_equal(columns["column_index"], np.arange(width))
    if width:
        assert len(slices["reference_count"]) == 2 * width
        assert slices["eligible"].all()
        assert slices["rare"].all()
        assert (slices["reference_total"] == 12).all()
        for column in range(width):
            assert slices["reference_count"][slices["column_index"] == column].sum() == 12
    for name, value in values.items():
        np.testing.assert_equal(captured.values[name], value)
        np.testing.assert_allclose(value, safe_nanmean(columns[name]), atol=1e-6, equal_nan=True)
        if (width < 2 and not name.startswith("recovery")) or (
            width == 2 and name.startswith("cond_trend")
        ):
            assert np.isnan(value), name
            assert np.isnan(columns[name]).all(), name

    if width < 2:
        assert all(np.isnan(slices[name]).all() for name in ["shape", "trend"])
    elif width == 2:
        assert np.isnan(slices["trend"]).all()

    # A categorical-only table gives an exact, independent oracle: identical
    # joint tables score one; completely absent slices score zero when defined.
    if kind == "categorical" and width >= 2:
        expected = float(candidate == "matched")
        for name, value in values.items():
            if width >= 3 or not name.startswith("cond_trend"):
                assert value == expected, name
        assert slices["eligible"].all()
        assert slices["rare"].all()
        assert (slices["reference_total"] == 12).all()
        if candidate != "matched":
            assert (slices["candidate_count"] == 0).all()
