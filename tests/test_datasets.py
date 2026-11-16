"""Unit tests for the OpenML on-ramp (``datasets/openml.py``).

All offline: the network fetch (``_sklearn_fetch``) is monkeypatched, and its scikit-learn
routing is exercised by monkeypatching ``fetch_openml`` itself, so nothing here touches OpenML.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabgen_eval import Schema, load_openml
from tabgen_eval.datasets.openml import _sklearn_fetch
from tests.helpers import make_frame
from tests.support.evaluation import paired_report as evaluate
from tests.support.options import make_options

pytestmark = pytest.mark.unit


def _patch_fetch(monkeypatch, frame: pd.DataFrame) -> None:
    """Make ``_sklearn_fetch`` return *frame* instead of hitting OpenML."""
    monkeypatch.setattr(
        "tabgen_eval.datasets.openml._sklearn_fetch",
        lambda dataset, version, cache_dir: frame.copy(),
    )


def test_returns_disjoint_train_test_covering_all_rows(monkeypatch):
    df = make_frame(200, seed=1)
    _patch_fetch(monkeypatch, df)

    train, test = load_openml("anything", test_size=0.25, seed=0)

    assert len(test) == 50 and len(train) == 150
    assert list(train.columns) == list(df.columns)
    # A partition: disjoint and together they reconstruct the original row multiset.
    combined = pd.concat([train, test], ignore_index=True).sort_values(list(df.columns))
    original = df.sort_values(list(df.columns))
    assert np.array_equal(combined.to_numpy(), original.to_numpy())


def test_split_is_deterministic_given_seed(monkeypatch):
    df = make_frame(120, seed=2)
    _patch_fetch(monkeypatch, df)

    a_train, a_test = load_openml("d", seed=7)
    b_train, b_test = load_openml("d", seed=7)
    c_train, c_test = load_openml("d", seed=8)

    pd.testing.assert_frame_equal(a_train, b_train)
    pd.testing.assert_frame_equal(a_test, b_test)
    # A different seed yields a different partition.
    assert not a_test.equals(c_test)


def test_reset_index_on_both_frames(monkeypatch):
    df = make_frame(80, seed=3)
    _patch_fetch(monkeypatch, df)
    train, test = load_openml("d")
    assert list(train.index) == list(range(len(train)))
    assert list(test.index) == list(range(len(test)))


def test_load_then_evaluate_smoke(monkeypatch):
    """The whole point: a loaded split flows straight into evaluate() and yields a Report."""
    df = make_frame(300, seed=4)
    _patch_fetch(monkeypatch, df)
    train, test = load_openml("d")

    syn = train.copy()  # a "perfect" generator: train rows stand in for synthetic
    report = evaluate(
        test, syn, Schema.infer(test), options=make_options(uncertainty="both", n_replicates=20)
    )
    assert len(report.to_dataframe()) > 0


def test_sklearn_fetch_routes_numeric_id_to_data_id(monkeypatch):
    captured: dict = {}

    class _Bunch:
        def __init__(self, frame):
            self.frame = frame

    def fake_fetch_openml(**kwargs):
        captured.update(kwargs)
        return _Bunch(pd.DataFrame({"a": [1, 2, 3]}))

    monkeypatch.setattr("sklearn.datasets.fetch_openml", fake_fetch_openml)

    _sklearn_fetch("45068", "active", None)
    assert captured["data_id"] == 45068
    assert "name" not in captured


def test_sklearn_fetch_routes_name_with_version_and_cache(monkeypatch):
    captured: dict = {}

    class _Bunch:
        def __init__(self, frame):
            self.frame = frame

    def fake_fetch_openml(**kwargs):
        captured.update(kwargs)
        return _Bunch(pd.DataFrame({"a": [1, 2, 3]}))

    monkeypatch.setattr("sklearn.datasets.fetch_openml", fake_fetch_openml)

    _sklearn_fetch("adult", 1, "/tmp/cache")
    assert captured["name"] == "adult"
    assert captured["version"] == 1
    assert captured["data_home"] == "/tmp/cache"
