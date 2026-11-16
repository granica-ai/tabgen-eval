"""Prepared input validation preserves values and metric-specific missingness."""

import numpy as np
import pandas as pd
import pytest

from tabgen_eval import EvalData, Schema, ValidationError, evaluate
from tabgen_eval.preprocessing.columns import PreparedColumns
from tests.support.options import make_options

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("field", ["clean", "categorical_nan_policy"])
def test_cleaning_options_are_removed(field):
    assert field not in dict(make_options())
    frame = pd.DataFrame({"x": [0.0, 1.0]})
    with pytest.raises(TypeError, match=field):
        evaluate(EvalData(test=frame, schema=Schema(numeric=["x"])), **{field: "off"})


def test_evaluation_requires_an_explicit_schema():
    frame = pd.DataFrame({"x": np.linspace(0, 1, 40)})
    with pytest.raises(TypeError, match="schema"):
        EvalData(test=frame)
    with pytest.raises(TypeError, match="schema"):
        EvalData(test=frame, schema=None)
    schema = Schema.infer(frame)
    data = EvalData(test=frame, generated=frame, schema=schema)
    assert np.isfinite(evaluate(data, metrics=["shape"])["shape"].value)


@pytest.mark.parametrize("role", ["train", "test", "generated"])
@pytest.mark.parametrize("kind", ["numeric_text", "invalid_text", "infinity", "datetime"])
def test_invalid_numeric_representations_raise_before_computation(role, kind):
    frame = pd.DataFrame({"x": np.linspace(0, 1, 40)})
    invalid = frame.copy()
    if kind in {"numeric_text", "invalid_text"}:
        invalid["x"] = invalid.x.astype(str)
        if kind == "invalid_text":
            invalid.loc[0, "x"] = "not-a-number"
    elif kind == "infinity":
        invalid.loc[0, "x"] = np.inf
    else:
        invalid["x"] = pd.date_range("2020-01-01", periods=len(invalid))
    frames = dict.fromkeys(("train", "test", "generated"), frame)
    frames[role] = invalid
    data = EvalData(**frames, schema=Schema(numeric=["x"]))
    with pytest.raises(ValidationError, match="x"):
        evaluate(data, metrics=["shape", "near_match_rate"])


def test_fractional_generated_values_are_not_rounded():
    train = pd.DataFrame({"x": np.arange(30)})
    generated = pd.DataFrame({"x": np.arange(30) + 0.1})
    before = generated.copy(deep=True)
    report = evaluate(
        EvalData(train=train, generated=generated, schema=Schema(numeric=["x"])),
        metrics=["near_match_rate"],
        uncertainty=None,
    )
    assert report["near_match_rate"].value == 0
    pd.testing.assert_frame_equal(generated, before)


def test_shape_explicitly_compares_observed_categories():
    test = pd.DataFrame({"c": ["a"] * 40})
    generated = pd.DataFrame({"c": ["a"] * 4 + [None] * 36})
    report = evaluate(
        EvalData(test=test, generated=generated, schema=Schema(categorical=["c"])),
        metrics=["shape"],
        uncertainty=None,
    )
    assert report["shape"].value == 1
    assert generated.c.isna().sum() == 36


def test_packing_preserves_typed_categories_and_ignores_unused_missing_columns():
    frame = pd.DataFrame(
        {
            "c": pd.Series([1, 1.0, True, "1", " 1 ", "nan", None], dtype=object),
            "ignored": [None] * 7,
        }
    )
    before = frame.copy(deep=True)
    packed = PreparedColumns.prepare_single(frame, Schema(categorical=["c"]), retain_labels=True)
    assert packed.cardinalities == (6,)
    assert packed.reference.complete.tolist() == [True] * 6 + [False]
    assert len(set(packed.reference.categorical[0, :6])) == 6
    assert packed.reference.categorical[0, 6] == -1
    pd.testing.assert_frame_equal(frame, before)
