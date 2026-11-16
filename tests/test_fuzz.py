"""Fuzz / adversarial: the dichotomy — a flagged Report OR a typed error, never a crash.

Every hostile frame must yield *either* a ``Report`` (possibly all-flagged) *or* a
``ValidationError``/``ValueError`` — never an uncaught exception, hang, or segfault.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabgen_eval import Schema, ValidationError
from tests.support.evaluation import paired_report as evaluate
from tests.support.options import make_options

pytestmark = pytest.mark.fuzz

CFG = make_options(uncertainty="both", n_replicates=10, seed=0)


def _run_ok(test, syn, schema):
    """Assert the dichotomy holds for one (test, syn, schema) triple."""
    try:
        rep = evaluate(test, syn, schema=schema, options=CFG)
    except (ValidationError, ValueError, ImportError):
        return  # typed error is acceptable
    rep.to_dataframe()
    rep.summary()  # rendering must not crash either


def test_all_nan_frame():
    schema = Schema(numeric=["a"], categorical=["c"])
    test = pd.DataFrame({"a": [np.nan] * 50, "c": [None] * 50})
    syn = pd.DataFrame({"a": [np.nan] * 50, "c": [None] * 50})
    _run_ok(test, syn, schema)


def test_single_row():
    schema = Schema(numeric=["a"], categorical=["c"])
    test = pd.DataFrame({"a": [1.0], "c": ["x"]})
    syn = pd.DataFrame({"a": [2.0], "c": ["y"]})
    _run_ok(test, syn, schema)


def test_constant_columns():
    schema = Schema(numeric=["a", "b"], categorical=["c"])
    test = pd.DataFrame({"a": [5.0] * 80, "b": np.arange(80.0), "c": ["x"] * 80})
    syn = pd.DataFrame({"a": [5.0] * 80, "b": np.arange(80.0), "c": ["x"] * 80})
    _run_ok(test, syn, schema)


def test_unicode_and_weird_categories():
    schema = Schema(categorical=["c"])
    weird = ["🙂", "a\nb", "x" * 500, "\x00", "  spaced  "]
    rng = np.random.default_rng(0)
    test = pd.DataFrame({"c": rng.choice(weird, 100)})
    syn = pd.DataFrame({"c": rng.choice(weird, 100)})
    _run_ok(test, syn, schema)


def test_inf_and_denormals():
    schema = Schema(numeric=["a"])
    test = pd.DataFrame({"a": [1e308, -1e308, np.inf, -np.inf, 5e-324] * 20})
    syn = pd.DataFrame({"a": np.random.default_rng(0).normal(size=100)})
    _run_ok(test, syn, schema)


def test_high_cardinality_categorical():
    # 700 distinct levels -> the one-hot width exceeds PRDC's structured-path threshold, so this
    # also exercises the code-based distance (no dense 700-wide one-hot matrix is materialised),
    # with fully disjoint syn levels stressing the out-of-vocabulary path.
    schema = Schema(categorical=["c"])
    test = pd.DataFrame({"c": [f"id_{i}" for i in range(700)]})
    syn = pd.DataFrame({"c": [f"id_{i}" for i in range(700, 1400)]})
    _run_ok(test, syn, schema)


def test_wrong_type_raises_typed():
    schema = Schema(numeric=["a"])
    with pytest.raises(ValidationError):
        evaluate(np.zeros((10, 1)), pd.DataFrame({"a": [1.0]}), schema=schema, options=CFG)


def test_column_mismatch_raises_typed():
    schema = Schema(numeric=["a"])
    test = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    syn = pd.DataFrame({"a": [1.0, 2.0]})
    with pytest.raises(ValidationError):
        evaluate(test, syn, schema=schema, options=CFG)
