"""Category-sensitive metric expectations from explicit, small probability tables."""

import numpy as np
import pandas as pd
import pytest

from tabgen_eval import Schema, diagnose
from tabgen_eval.execution.python.conditional import compute_conditional
from tabgen_eval.execution.python.encoding.features import fit_encoder
from tabgen_eval.execution.python.primitives import contingency_similarity, tv_complement

pytestmark = pytest.mark.unit


def _series(values):
    return pd.Series(values, dtype=object)


@pytest.mark.parametrize(
    "first,second", [(1, "1"), (1, 1.0), (1, True), (1.0, True), (" A ", "A"), ("1.0", "1")]
)
def test_shape_distinguishes_category_identity(first, second):
    assert tv_complement(_series([first]), _series([second])) == 0.0


@pytest.mark.parametrize(
    "first,second",
    [(1, np.int64(1)), (1.0, np.float32(1.0)), (True, np.bool_(True)), ("x", np.str_("x"))],
)
def test_shape_preserves_logical_scalar_family(first, second):
    assert tv_complement(_series([first]), _series([second])) == 1.0


def test_shape_actual_missing_does_not_consume_literal_category_mass():
    reference = _series([None, np.nan, pd.NA, pd.NaT, "nan", "<MISSING>"])
    candidate = _series(["nan", "nan"])
    assert tv_complement(reference, candidate) == 0.5


@pytest.mark.parametrize("values", [[1, "1"], [1, 1.0], [1, True], ["nan", "x"]])
def test_trend_retains_distinct_category_associations(values):
    score = contingency_similarity(
        _series(values),
        _series(["left", "right"]),
        _series(values),
        _series(["right", "left"]),
        col1_numeric=False,
        col2_numeric=False,
    )
    # The two joint distributions have disjoint support, including literal "nan".
    assert score == 0.0


def test_numeric_bins_drop_actual_missing_without_dropping_literal_nan():
    score = contingency_similarity(
        pd.Series([0.0, 1.0, np.nan]),
        _series(["nan", "x", "ignored"]),
        pd.Series([0.0, 1.0]),
        _series(["nan", "x"]),
        col1_numeric=True,
        col2_numeric=False,
    )
    assert score == 1.0


def test_conditional_uses_typed_slices_and_unweighted_slice_averages():
    reference = pd.DataFrame(
        {"condition": _series([1, 1, "1", "1"]), "value": ["a", "a", "b", "b"]}
    )
    candidate = pd.DataFrame({"condition": _series([1, 1, 1, 1]), "value": ["a", "a", "b", "b"]})
    result = compute_conditional(
        reference,
        candidate,
        Schema(categorical=["condition", "value"]),
        num_bins=2,
        min_count=1,
        rare_threshold=0.1,
    )
    # Conditioning on condition: mean(0.5, 0) = 0.25. On value: mean(1, 0) = 0.5.
    assert result["cond_shape_all"] == 0.375


def test_data_diagnostics_duplicate_and_support_checks_use_typed_categories():
    reference = pd.DataFrame({"category": _series([1])})
    candidate = pd.DataFrame({"category": _series([1, np.int64(1), 1.0, "1", True, " A ", "A"])})
    schema = Schema(categorical=["category"])
    result = diagnose(reference, candidate, schema=schema)
    assert result.inputs["generated"]["duplicate_row_rate"] == pytest.approx(1 / 7)
    assert result.categorical_support["category"]["n_unseen_categories"] == 5


def test_distance_encoder_preserves_identity_and_reference_fitted_unknowns():
    reference = pd.DataFrame({"category": _series([1, 1.0, "1", True, "<MISSING>", None])})
    candidate = pd.DataFrame({"category": _series([np.int64(1), "new", "<MISSING>", np.nan])})
    encoder = fit_encoder(reference, Schema(categorical=["category"]))
    dense = encoder.transform(reference)
    assert encoder.onehot_dim == 6
    np.testing.assert_array_equal(dense @ dense.T, np.eye(6))
    before = encoder.onehot_dim
    encoded = encoder.transform(candidate)
    np.testing.assert_array_equal(encoded[0], dense[0])
    np.testing.assert_array_equal(encoded[1], np.zeros(6))
    np.testing.assert_array_equal(encoded[2], dense[4])
    np.testing.assert_array_equal(encoded[3], dense[5])
    assert encoder.onehot_dim == before
    structured = encoder.transform_structured(candidate)
    assert structured.codes[1, 0] == -1
    assert structured.oov[:, 0].tolist() == [False, True, False, False]
    np.testing.assert_array_equal(structured.codes[[0, 2, 3], 0], [0, 4, 5])


def test_bootstrap_copies_preserve_duplicate_categories():
    frame = pd.DataFrame({"category": _series([1, "1", 1.0, True])})
    copy = frame.iloc[[0, 1, 2, 3, 0, 1, 2, 3]].reset_index(drop=True)
    result = diagnose(frame, copy, schema=Schema(categorical=["category"]))
    assert result.inputs["generated"]["duplicate_row_rate"] == 0.5
    assert result.categorical_support["category"]["n_unseen_categories"] == 0


def test_data_diagnostics_duplicate_scope_includes_undeclared_columns_without_mutation():
    frame = pd.DataFrame({"category": _series([1, np.int64(1), 1.0]), "extra": [0, 1, 0]})
    before = frame.copy(deep=True)
    result = diagnose(frame, frame, schema=Schema(categorical=["category"]))
    assert result.inputs["generated"]["duplicate_row_rate"] == 0.0
    pd.testing.assert_frame_equal(frame, before)


def test_data_diagnostics_null_categories_share_missing_identity():
    frame = pd.DataFrame({"category": _series([None, np.nan, pd.NA, pd.NaT])})
    result = diagnose(frame, frame, schema=Schema(categorical=["category"]))
    assert result.inputs["generated"]["duplicate_row_rate"] == 0.75
    assert result.categorical_support["category"]["n_unseen_categories"] == 0


@pytest.mark.parametrize(
    "first,second",
    [
        (np.datetime64("2024-01-01"), pd.Timestamp("2024-01-01")),
        (np.timedelta64(24, "h"), pd.Timedelta(days=1)),
        (np.uint64(2**64 - 1), 2**64 - 1),
        (np.int64(-(2**63)), -(2**63)),
    ],
)
def test_temporal_and_wide_integer_categories_share_logical_identity(first, second):
    assert tv_complement(_series([first]), _series([second])) == 1.0


@pytest.mark.parametrize(
    "first,second",
    [
        (np.datetime64("2024-01-01"), "2024-01-01"),
        (np.timedelta64(1, "D"), 1),
        (np.uint64(2**64 - 1), np.uint64(2**64 - 2)),
        ((1,), (1.0,)),
        ((1,), ("1",)),
    ],
)
def test_temporal_and_composite_category_values_remain_distinct(first, second):
    assert tv_complement(_series([first]), _series([second])) == 0.0


class _DisplayedString(str):
    def __str__(self):
        return "display-only"


def test_string_subclasses_use_exact_contents_in_fidelity_and_distance():
    reference = pd.DataFrame(
        {"category": _series([_DisplayedString(" A "), _DisplayedString("A"), "display-only"])}
    )
    candidate = pd.DataFrame({"category": _series([" A ", "A", _DisplayedString("display-only")])})
    assert tv_complement(reference["category"], candidate["category"]) == 1.0
    encoder = fit_encoder(reference, Schema(categorical=["category"]))
    assert encoder.onehot_dim == 3
    np.testing.assert_array_equal(encoder.transform(reference), encoder.transform(candidate))


@pytest.mark.parametrize("dtype", [object, np.longdouble])
def test_extended_float_categories_preserve_observed_precision(dtype):
    if np.finfo(np.longdouble).eps >= np.finfo(np.float64).eps:
        pytest.skip("Platform has no floating precision beyond FP64")
    low = np.longdouble(1)
    high = low + np.finfo(np.longdouble).eps
    reference = pd.Series(np.array([low, high], dtype=np.longdouble), dtype=dtype)
    candidate = pd.Series(np.array([low, low], dtype=np.longdouble), dtype=dtype)
    assert tv_complement(reference, candidate) == 0.5


class _EqualInteger(int):
    def __eq__(self, other):
        return True

    def __hash__(self):
        return 1


def test_integer_subclasses_use_logical_values_before_native_equality():
    reference = _series([_EqualInteger(1), _EqualInteger(2)])
    assert tv_complement(reference, _series([1, 2])) == 1.0
    frame = pd.DataFrame({"category": reference})
    assert (
        diagnose(frame, frame, schema=Schema(categorical=["category"])).inputs["generated"][
            "duplicate_row_rate"
        ]
        == 0.0
    )
