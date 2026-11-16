"""Reference-precision bin boundaries retain finite FP32 observations."""

import numpy as np
import pandas as pd
import pytest

from tabgen_eval.execution.python.conditional import _bin_labels
from tabgen_eval.execution.python.primitives import _as_labels

pytestmark = pytest.mark.unit


def expected_codes(values, edges, *, clip):
    """Locate observations by literal right-closed interval membership."""
    values = values.to_numpy(dtype=np.float64, na_value=np.nan)
    result = []
    for value in values:
        code = pd.NA
        if not np.isnan(value):
            if clip:
                value = min(max(value, edges[0]), edges[-1])
            for index, (left, right) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
                if (value >= left if index == 0 else value > left) and value <= right:
                    code = index
                    break
        result.append(code)
    return pd.array(result, dtype="Int64")


@pytest.mark.parametrize("scale", [1.0, 1e-13])
def test_right_closed_labels_include_exact_upper_bound_and_clip_outliers_at_any_scale(scale):
    from tabgen_eval.execution.python.primitives import _bin_codes

    edges = np.array([0.0, 2.0, 4.0]) * scale
    source = pd.Series([-10, -1, 0, 1, 2, 3, 4, 5, 10, np.nan]) * scale
    expected = pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, pd.NA], dtype="Int64")
    pd.testing.assert_series_equal(_as_labels(source, True, edges, clip=True), expected)
    expected_array = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, -1])
    np.testing.assert_array_equal(_bin_codes(source.to_numpy(), edges, clip=True), expected_array)
    # Raw array kernels still defend against nonfinite coordinates even though
    # numeric Series must already satisfy the public representation contract.
    raw = source.to_numpy(copy=True)
    raw[[0, 8]] = [-np.inf, np.inf]
    expected_array[[0, 8]] = -1
    np.testing.assert_array_equal(_bin_codes(raw, edges, clip=True), expected_array)
    unbounded = _as_labels(source, True, edges)
    pd.testing.assert_series_equal(unbounded, pd.Series(expected_codes(source, edges, clip=False)))


@pytest.mark.parametrize("dtype", ["float32", "Float32", "float64", "Float64"])
def test_clipped_numeric_labels_retain_both_bounds_outliers_and_missing_cells(dtype):
    # These bounds round outward in FP32 on opposite sides of the interval.
    edges = np.array([0.7, 0.75, 0.8], dtype=np.float64)
    assert float(np.float32(edges[0])) < edges[0]
    assert float(np.float32(edges[-1])) > edges[-1]
    index = pd.Index([9, 2, 7, 1, 5, 4], name="row")
    source = pd.Series([-10, 0.7, 0.75, 0.8, 10, np.nan], dtype=dtype, index=index, name="x")
    untouched = source.copy(deep=True)
    actual = _as_labels(source, True, edges, clip=True)
    expected = pd.Series(expected_codes(source, edges, clip=True), index=index)
    pd.testing.assert_series_equal(actual, expected)
    assert actual.notna().tolist() == [True, True, True, True, True, False]
    pd.testing.assert_series_equal(source, untouched)


def test_unclipped_numeric_labels_retain_the_reference_interval_policy():
    edges = np.array([0.7, 0.75, 0.8])
    source = pd.Series([0.69, 0.7, 0.75, 0.8, 0.81, pd.NA], dtype="Float64", name="x")
    actual = _as_labels(source, True, edges)
    expected = pd.Series(expected_codes(source, edges, clip=False))
    pd.testing.assert_series_equal(actual, expected)
    assert actual.isna().tolist() == [True, False, False, False, True, True]


@pytest.mark.parametrize(
    "reference_dtype,candidate_dtype",
    [
        ("float64", "float32"),
        ("float32", "float64"),
        ("float32", "float32"),
        ("float64", "float64"),
        ("Float64", "Float32"),
        ("Float32", "Float64"),
    ],
)
def test_conditioning_labels_match_float64_oracle_for_mixed_precision_and_nullable_values(
    reference_dtype, candidate_dtype
):
    reference = pd.Series(
        [0.7, 0.725, 0.75, 0.775, 0.8, np.nan],
        dtype=reference_dtype,
        index=[7, 5, 3, 1, 9, 11],
        name="x",
    )
    candidate = pd.Series(
        [-10, 0.7, 0.75, 0.8, 10, np.nan],
        dtype=candidate_dtype,
        index=[8, 6, 4, 2, 10, 12],
        name="x",
    )
    original_reference, original_candidate = reference.copy(deep=True), candidate.copy(deep=True)
    real_labels, generated_labels, edges = _bin_labels(reference, candidate, 4, return_edges=True)
    finite_reference = reference.to_numpy(dtype=np.float64, na_value=np.nan)[:-1]
    expected_edges = np.linspace(finite_reference.min(), finite_reference.max(), 5)
    np.testing.assert_array_equal(edges, expected_edges)
    for source, actual, clip in [
        (reference, real_labels, False),
        (candidate, generated_labels, True),
    ]:
        codes = expected_codes(source, expected_edges, clip=clip)
        labels = [np.nan if pd.isna(code) else str(code) for code in codes]
        expected = pd.Series(labels, index=source.index, name=source.name, dtype=object)
        pd.testing.assert_series_equal(actual, expected)
        assert actual.notna().sum() == 5
    pd.testing.assert_series_equal(reference, original_reference)
    pd.testing.assert_series_equal(candidate, original_candidate)


@pytest.mark.parametrize("dtype", ["float32", "Float32", "Float64", "Int64"])
def test_constant_numeric_conditioning_retains_finite_values_and_real_missingness(dtype):
    real = pd.Series([1, 1, np.nan], dtype=dtype, index=[3, 7, 2])
    candidate = pd.Series([-1, 1, 3, np.nan], dtype=dtype, index=[6, 4, 8, 0])
    real_labels, candidate_labels, edges = _bin_labels(real, candidate, 2, return_edges=True)
    expected_edges = np.array([1 - 1.001e-6, 1, 1 + 1.001e-6])
    np.testing.assert_array_equal(edges, expected_edges)
    assert real_labels.tolist()[:2] == ["0", "0"]
    assert candidate_labels.tolist()[:3] == ["0", "0", "1"]
    assert pd.isna(real_labels.iloc[-1]) and pd.isna(candidate_labels.iloc[-1])
    pd.testing.assert_index_equal(real_labels.index, real.index)
    pd.testing.assert_index_equal(candidate_labels.index, candidate.index)


def test_all_missing_nullable_reference_is_unbinnable_and_categories_retain_identity():
    assert _bin_labels(pd.Series([pd.NA], dtype="Float32"), pd.Series([1.0]), 2) is None
    categorical = pd.Series([1, "1", 1.0, " A ", "A", pd.NA], dtype=object)
    assert _as_labels(categorical, False, None, clip=True) is categorical
