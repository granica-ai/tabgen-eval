"""Category identity contracts with explicit, independently assigned outcomes."""

import pickle

import numpy as np
import pandas as pd
import pytest

from tabgen_eval.preprocessing.categories import (
    MISSING_CATEGORY,
    CategoryVocabulary,
    category_key,
    factorize_categories,
)


def test_equal_values_in_distinct_families_have_distinct_codes():
    values = pd.Series(
        [1, "1", 1.0, True, np.int64(1), np.float32(1), np.bool_(True)], dtype=object
    )
    (codes,), vocabulary = factorize_categories(values)
    assert len(vocabulary) == 4
    np.testing.assert_array_equal(codes, [0, 1, 2, 3, 0, 2, 3])


def test_python_sequence_is_not_coerced_before_identity_encoding():
    (codes,), vocabulary = factorize_categories([1, 1.0, True, "1"])
    np.testing.assert_array_equal(codes, [0, 1, 2, 3])
    assert len(vocabulary) == 4


def test_exact_labels_and_missing_categories_cannot_collide():
    values = pd.Series(
        [
            " A ",
            "A",
            "1.0",
            "1",
            "",
            " ",
            "nan",
            "NaN",
            "None",
            "<MISSING>",
            None,
            np.nan,
            pd.NA,
            pd.NaT,
            MISSING_CATEGORY,
        ],
        dtype=object,
    )
    (codes,), vocabulary = factorize_categories(values)
    np.testing.assert_array_equal(codes, list(range(10)) + [-1] * 4 + [10])
    assert len(vocabulary) == 11
    assert pickle.loads(pickle.dumps(MISSING_CATEGORY)) is MISSING_CATEGORY
    restored = pickle.loads(pickle.dumps(vocabulary))
    np.testing.assert_array_equal(restored.transform(values), codes)


@pytest.mark.parametrize(
    "dtype",
    [
        "int8",
        "int64",
        "uint64",
        "float16",
        "float32",
        "float64",
        "Int64",
        "Float32",
        "boolean",
        "bool",
        "string",
        "category",
        object,
    ],
)
def test_dtype_dictionary_matches_scalar_identity(dtype):
    values = [True, False, True] if dtype in {"bool", "boolean"} else [1, 2, 1]
    if dtype in {"string", "category", object}:
        values = ["1.0", " 1 ", "1.0"]
    series = pd.Series(values * 500, dtype=dtype)
    (codes,), vocabulary = factorize_categories(series)
    assert len(vocabulary) == 2
    np.testing.assert_array_equal(codes, [0, 1, 0] * 500)
    assert vocabulary.keys == tuple(category_key(v) for v in series.iloc[:2])


def test_training_vocabulary_keeps_null_and_unseen_distinct():
    vocabulary = CategoryVocabulary.fit(pd.Series(["a", None, "<MISSING>"]))
    scoring = pd.Series(["a", "b", None, MISSING_CATEGORY, "<MISSING>"])
    np.testing.assert_array_equal(vocabulary.transform(scoring), [0, -2, -1, -2, 1])
    assert vocabulary.values == ("a", "<MISSING>")


def test_unused_categorical_dictionary_values_are_not_observed():
    series = pd.Series(pd.Categorical(["b", None, "b"], categories=["unused", "b"]))
    (codes,), vocabulary = factorize_categories(series)
    assert vocabulary.values == ("b",)
    np.testing.assert_array_equal(codes, [0, -1, 0])


def test_bootstrap_rows_and_cross_pool_identities_keep_multiplicity():
    real = pd.Series([1, "1", True, 1.0], dtype=object)
    fake = real.iloc[[3, 3, 1, 0, 2, 2]]
    (a, b), vocabulary = factorize_categories(real, fake)
    np.testing.assert_array_equal(a, [0, 1, 2, 3])
    np.testing.assert_array_equal(b, [3, 3, 1, 0, 2, 2])
    assert len(vocabulary) == 4


def test_large_unsigned_integers_are_not_rounded_to_float():
    values = pd.Series([2**63 + 1, 2**63 + 2, 2**64 - 1], dtype="uint64")
    (codes,), vocabulary = factorize_categories(values)
    np.testing.assert_array_equal(codes, [0, 1, 2])
    assert vocabulary.keys == tuple(("integer", int(v)) for v in values)


def test_string_subclass_content_is_not_custom_display():
    class Label(str):
        def __str__(self):
            return "display"

    (codes,), vocabulary = factorize_categories(pd.Series([Label("actual"), "actual", "display"]))
    np.testing.assert_array_equal(codes, [0, 0, 1])
    assert len(vocabulary) == 2


def test_homogeneous_columns_only_key_dictionary_labels(monkeypatch):
    import tabgen_eval.preprocessing.categories as categories

    original = categories.category_key
    calls = []

    def counted(value):
        calls.append(value)
        return original(value)

    monkeypatch.setattr(categories, "category_key", counted)
    (codes,), vocabulary = factorize_categories(pd.Series(["a", "b", None] * 100_000))
    assert len(codes) == 300_000
    assert len(vocabulary) == len(calls) == 2
