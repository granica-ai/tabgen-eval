"""Arrow and pandas packing preserve declared values, typed identities, and sampling."""

import json

import numpy as np
import pandas as pd
import pytest

from tabgen_eval import EvalData, Schema, evaluate
from tabgen_eval.preprocessing.arrow import try_prepare
from tabgen_eval.preprocessing.columns import PreparedColumns
from tests.support.options import make_options
from tests.support.reports import without_memory_plans

pytestmark = pytest.mark.unit


def prepare(real, fake, schema):
    pytest.importorskip("pyarrow")
    result = try_prepare(real, fake, schema)
    return PreparedColumns.prepare(real, fake, schema) if result is None else result


def parity(real, fake, schema):
    source_r, source_s = real.copy(deep=True), fake.copy(deep=True)
    expected = PreparedColumns.prepare(real, fake, schema)
    actual = prepare(real, fake, schema)
    for actual_pool, expected_pool in [
        (actual.reference, expected.reference),
        (actual.candidate, expected.candidate),
    ]:
        for field in ["numeric", "categorical", "complete"]:
            np.testing.assert_array_equal(
                getattr(actual_pool, field), getattr(expected_pool, field)
            )
    for field in [
        "numeric_names",
        "categorical_names",
        "cardinalities",
        "reference_cardinalities",
        "category_mismatch",
    ]:
        assert getattr(actual, field) == getattr(expected, field)
    pd.testing.assert_frame_equal(real, source_r, check_exact=True)
    pd.testing.assert_frame_equal(fake, source_s, check_exact=True)


@pytest.mark.parametrize(
    "dtype",
    [
        object,
        "string[python]",
        "string[pyarrow]",
        "category",
        "float16",
        "float32",
        "float64",
        "int64",
        "uint64",
        "bool",
        "Int64",
        "Float32",
        "boolean",
    ],
)
def test_dtype_rules(dtype):
    if dtype in [object, "string[python]", "string[pyarrow]", "category"]:
        values = ["01", "1.0", "1", "  eng ", "", "nan", "NaN", "١.٠", None, np.nan]
    elif dtype in ["bool", "boolean"]:
        values = [True, False] + ([pd.NA] if dtype == "boolean" else [])
    elif dtype in ["int64", "Int64"]:
        values = [0, -1, 2**53 + 1] + ([pd.NA] if dtype == "Int64" else [])
    elif dtype == "uint64":
        values = [0, 2**63 + 1, 2**64 - 1]
    else:
        values = [0, -0.0, -1, 1.234567, np.inf, -np.inf, np.nan, 1e-20]
    real = pd.DataFrame({"c": pd.Series(values * 150, dtype=dtype)})
    fake = real.iloc[::-1].copy()
    parity(real, fake, Schema(categorical=["c"]))


@pytest.mark.parametrize(
    "values",
    [
        [1, True, 1.0, "1", False, 0],
        [None, np.nan, pd.NA, pd.NaT, np.float32(np.nan), "nan"],
        [pd.Timestamp("2020-01-01"), pd.NaT, "NaT"],
    ],
)
def test_unsupported_mixed_fallback(values):
    real = pd.DataFrame({"c": pd.Series(values * 300, dtype=object)})
    parity(real, real.iloc[::-1].copy(), Schema(categorical=["c"]))


def test_declared_feature_missingness_excludes_unmodeled_columns():
    real = pd.DataFrame(
        {
            "x": [1.1, 2.0, np.nan, 4.0],
            "c": [" 1.0 ", "01", None, "nan"],
            "ignored": [None, None, "present", "present"],
        }
    )
    schema = Schema(numeric=["x"], categorical=["c"])
    parity(real, real.iloc[::-1].copy(), schema)
    columns = prepare(real, real, schema)
    np.testing.assert_array_equal(columns.reference.complete, [True, True, False, True])


@pytest.mark.parametrize("dtype", ["float32", "float64", "Float32", "Float64", "Int64"])
def test_numeric_nullable_storage_and_missingness_match(dtype):
    source = pd.DataFrame({"x": pd.Series([1, None, 3], dtype=dtype)})
    parity(source, source.iloc[::-1].copy(), Schema(numeric=["x"]))


@pytest.mark.parametrize(
    "values",
    [
        ["1", "bad"],
        [1.0, np.inf],
        pd.date_range("2020-01-01", periods=2),
        pd.to_timedelta([1, 2], unit="s"),
    ],
)
def test_arrow_does_not_repair_invalid_numeric_values(values):
    source = pd.DataFrame({"x": values})
    schema = Schema(numeric=["x"])
    with pytest.raises(ValueError):
        prepare(source, source, schema)
    with pytest.raises(ValueError):
        PreparedColumns.prepare(source, source, schema)


@pytest.mark.parametrize("rows", [0, 1, 1023, 4096])
def test_empty_all_missing_and_high_cardinality(rows):
    real = pd.DataFrame(
        {
            "missing": pd.Series([None] * rows, dtype=object),
            "unique": pd.Series([str(i) for i in range(rows)], dtype=object),
        }
    )
    fake = real.iloc[: max(0, rows - 1)].copy()
    parity(real, fake, Schema(categorical=["missing", "unique"]))


def test_shared_vocab_order_and_category_mismatch():
    real = pd.DataFrame({"c": ["red", "blue", "1.0", "1", "nan", None] * 300})
    fake = pd.DataFrame({"c": ["blue", "red", "1", "01", "nan", None] * 250})
    real.index = pd.MultiIndex.from_tuples([("a", 1)] * len(real))
    parity(real, fake, Schema(categorical=["c"]))


def test_no_modeled_columns_still_preserves_complete_rows():
    real = pd.DataFrame({"ignored": [None, "value"] * 600})
    parity(real, real.iloc[::-1].copy(), Schema())


def test_dataframe_subclasses_use_pandas_packing():
    class CustomFrame(pd.DataFrame):
        pass

    real = pd.DataFrame({"c": [" 1.0 ", None]})
    schema = Schema(categorical=["c"])
    assert try_prepare(CustomFrame(real), real, schema) is None


def test_missing_arrow_uses_pandas_route(monkeypatch):
    import sys

    real = pd.DataFrame({"c": [" 1.0 ", None]})
    monkeypatch.setitem(sys.modules, "pyarrow", None)
    assert try_prepare(real, real, Schema(categorical=["c"])) is None


def test_unrepresentable_unicode_and_string_subclass_fallback():
    class Alias(str):
        def __str__(self):
            return "other label"

    for values in [["\ud800", "normal", None], ["same", Alias("same"), None]]:
        real = pd.DataFrame({"c": pd.Series(values * 500, dtype=object)})
        schema = Schema(categorical=["c"])
        parity(real, real.iloc[::-1].copy(), schema)


@pytest.mark.parametrize("label", ["\ud800", "\udfff"])
@pytest.mark.parametrize("details", [False, True])
def test_public_shape_preserves_unrepresentable_unicode_with_missing_category(label, details):
    pytest.importorskip("pyarrow")
    real = pd.DataFrame({"c": pd.Series([label, "normal", None], dtype=object)})
    generated = real.copy(deep=True)
    original = real.copy(deep=True)
    schema = Schema(categorical=["c"])

    columns = try_prepare(real, generated, schema, retain_labels=True)
    assert columns.category_values == ((label, "normal"),)
    np.testing.assert_array_equal(columns.reference.categorical, [[0, 1, -1]])
    np.testing.assert_array_equal(columns.candidate.categorical, [[0, 1, -1]])
    np.testing.assert_array_equal(columns.reference.complete, [True, True, False])

    report = evaluate(
        EvalData(test=real, generated=generated, schema=schema),
        metrics=["shape"],
        device="cpu",
        details=details,
    )
    assert report["shape/shape"].value == 1.0
    pd.testing.assert_frame_equal(real, original, check_exact=True)
    pd.testing.assert_frame_equal(generated, original, check_exact=True)


def test_prepared_columns_own_values_after_source_mutation():
    pytest.importorskip("pyarrow")
    real = pd.DataFrame({"x": [1.0, 2.0], "c": ["red", "blue"]})
    schema = Schema(numeric=["x"], categorical=["c"])
    columns = try_prepare(real, real, schema)
    real.loc[0, "x"], real.loc[0, "c"] = 100, "changed"
    np.testing.assert_array_equal(columns.reference.numeric, [[1, 2]])
    np.testing.assert_array_equal(columns.reference.categorical, [[0, 1]])
    assert not columns.reference.numeric.flags.writeable
    assert not columns.reference.categorical.flags.writeable


@pytest.mark.parametrize("B", [0, 3])
def test_integrated_tensor_evaluation_preserves_reports_and_draws(monkeypatch, B):
    import importlib

    from tests.support.evaluation import memorization_report as evaluate_memorization
    from tests.support.evaluation import paired_report as evaluate

    pytest.importorskip("torch")
    pytest.importorskip("pyarrow")
    module = importlib.import_module("tabgen_eval.execution.runtime.preparation")
    memorization = importlib.import_module("tabgen_eval.execution.runtime.memorization")
    rng = np.random.default_rng(48)
    real = pd.DataFrame(
        {
            "x": rng.normal(size=260),
            "y": rng.normal(size=260),
            "c": [" red ", "blue", None, "nan"] * 65,
            "ignored": [None, "ok", "ok", "ok"] * 65,
        }
    )
    fake = real.iloc[::-1].copy()
    fake.loc[0, "c"] = "unknown"
    schema = Schema(numeric=["x", "y"], categorical=["c"])
    metrics = ["shape", "trend", "conditional_fidelity", "prdc", "linear_c2st"]
    options = make_options(
        device="cpu",
        uncertainty="both" if B else "none",
        n_replicates=max(1, B),
        seed=19,
    )
    with monkeypatch.context() as context:
        context.setattr(module, "try_prepare", lambda *args, **kwargs: None)
        expected = evaluate(real, fake, schema, metrics=metrics, options=options)
        context.setattr(memorization, "try_prepare", lambda *args, **kwargs: None)
        expected_dcr = evaluate_memorization(real, fake, schema, options=options)

    def forbidden(*args, **kwargs):
        pytest.fail("Eligible evaluation must use combined preparation only")

    monkeypatch.setattr(PreparedColumns, "prepare", forbidden)
    actual = evaluate(real, fake, schema, metrics=metrics, options=options)
    actual_dcr = evaluate_memorization(real, fake, schema, options=options)
    assert without_memory_plans(json.loads(actual_dcr.to_json())) == without_memory_plans(
        json.loads(expected_dcr.to_json())
    )
    assert np.isfinite(actual["shape/shape"].value)
    assert actual.provenance["per_metric_sampling"]["shape"]["n_res"] >= 100
    for mr, old in expected.iter_stats():
        new = actual[f"{mr.name}/{old.name}"]
        assert old.flags == new.flags
        for field in [
            "value",
            "ideal",
            "ci_low",
            "ci_high",
            "ideal_ci_low",
            "ideal_ci_high",
            "generated_distribution",
            "reference_distribution",
        ]:
            a, b = getattr(old, field), getattr(new, field)
            if field.endswith("_distribution") and a is not None:
                a, b = a.samples, b.samples
            if a is None:
                assert b is None
            else:
                np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("route", ["python", "mixed", "no_arrow", "validation"])
def test_evaluation_fallbacks_preserve_supplied_values(monkeypatch, route):
    import importlib
    import sys

    from tabgen_eval.metrics import Shape
    from tests.support.evaluation import paired_report as evaluate

    real = pd.DataFrame({"c": [" 1.0 ", "01", None] * 50})
    expected = real.copy(deep=True)
    schema = Schema(categorical=["c"])
    options = make_options(device="cpu")
    metric = Shape(device=options["device"], bootstrapped=route != "mixed")
    if route != "python":
        pytest.importorskip("torch")
    if route == "no_arrow":
        monkeypatch.setitem(sys.modules, "pyarrow", None)
    elif route == "validation":
        metric.validate_inputs = lambda r, s, schema: None
    preparation = importlib.import_module("tabgen_eval.execution.runtime.preparation")
    original = preparation.prepare_evaluation_sources
    seen = []

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        pd.testing.assert_frame_equal(result.reference, expected)
        pd.testing.assert_frame_equal(result.generated, expected)
        seen.append(result)
        return result

    monkeypatch.setattr(preparation, "prepare_evaluation_sources", capture)
    evaluate(real, real, schema, metrics=[metric], options=options)
    assert len(seen) == 1
    pd.testing.assert_frame_equal(real, expected)


def test_missing_arrow_module_preserves_existing_arrow_backed_input(monkeypatch):
    import sys

    from tabgen_eval import EvalData, evaluate

    pytest.importorskip("pyarrow")
    pytest.importorskip("torch")
    frame = pd.DataFrame(
        {
            "x": pd.Series([1.0, 2.0, None, 4.0] * 20, dtype="float64[pyarrow]"),
            "c": pd.Series([" a ", "a", None, "nan"] * 20, dtype="string[pyarrow]"),
        }
    )
    before = frame.copy(deep=True)
    monkeypatch.setitem(sys.modules, "pyarrow", None)
    report = evaluate(
        EvalData(
            test=frame, generated=frame.copy(), schema=Schema(numeric=["x"], categorical=["c"])
        ),
        metrics=["shape"],
        device="cpu",
        uncertainty=None,
    )
    assert np.isfinite(report["shape"].value)
    pd.testing.assert_frame_equal(frame, before)


def test_no_torch_or_custom_contract_prevents_prepared_evaluation(monkeypatch):
    from tabgen_eval.execution.runtime.dispatch import supports_prepared_evaluation
    from tabgen_eval.execution.torch import tensors
    from tabgen_eval.metrics import DCR, Shape

    schema = Schema(numeric=["x"])
    assert not supports_prepared_evaluation([], schema)
    assert not supports_prepared_evaluation([DCR()], schema)  # small-numeric auto uses Python
    assert not supports_prepared_evaluation([Shape(bootstrapped=False)], schema)
    monkeypatch.setattr(tensors, "torch_module", lambda **kwargs: None)
    assert not supports_prepared_evaluation([Shape()], schema)


def test_typed_identity_and_literal_missing_labels():
    real = pd.DataFrame(
        {"c": pd.Series([1, 1.0, "1", True, " A ", "A", "nan", "<MISSING>", None], dtype=object)}
    )
    fake = pd.DataFrame(
        {
            "c": pd.Series(
                [
                    np.int64(1),
                    np.float32(1),
                    "1",
                    np.bool_(True),
                    " A ",
                    "A",
                    "nan",
                    "<MISSING>",
                    pd.NA,
                ],
                dtype=object,
            )
        }
    )
    schema = Schema(categorical=["c"])
    parity(real, fake, schema)
    columns = prepare(real, fake, schema)
    np.testing.assert_array_equal(columns.reference.categorical, columns.candidate.categorical)
    observed = columns.reference.categorical[0, :8]
    assert len(set(observed)) == 8
    assert columns.cardinalities == (8,)
    assert columns.reference.categorical[0, 8] == -1
    assert not columns.reference.complete[8]


def test_arrow_null_predictors_and_targets_remain_missing():
    source = pd.DataFrame({"c": [None, "<MISSING>"], "target": [None, "nan"]})
    schema = Schema(categorical=["c", "target"], target="target")
    parity(source, source, schema)
    columns = prepare(source, source, schema)
    assert columns.reference.categorical[0, 0] == -1
    assert columns.reference.categorical[1, 0] == -1
    assert columns.reference.categorical[1, 1] >= 0


def test_arrow_and_pandas_share_float_family_without_string_rounding():
    value = np.float32(0.125)
    real = pd.DataFrame({"c": pd.Series([value, np.float32(1)], dtype="float32")})
    fake = pd.DataFrame({"c": pd.Series([np.float64(value), np.float64(1)], dtype="float64")})
    schema = Schema(categorical=["c"])
    parity(real, fake, schema)
    columns = prepare(real, fake, schema)
    assert columns.cardinalities == (2,)
    assert not columns.category_mismatch
    np.testing.assert_array_equal(columns.reference.categorical, columns.candidate.categorical)


def test_mixed_mle_preparation_preserves_supplied_host_frames():
    from tabgen_eval.execution.runtime.preparation import prepare_evaluation_sources
    from tabgen_eval.metrics import MLE, Shape

    real = pd.DataFrame({"x": [1.1, 2.2, np.nan, 4.4], "y": [1, 2, 3, 4]})
    generated = real.iloc[::-1].copy()
    prepared = prepare_evaluation_sources(
        real,
        generated,
        Schema(numeric=["x", "y"], target="y"),
        metrics=[Shape(), MLE()],
    )
    pd.testing.assert_frame_equal(prepared.reference, real)
    pd.testing.assert_frame_equal(prepared.generated, generated)
