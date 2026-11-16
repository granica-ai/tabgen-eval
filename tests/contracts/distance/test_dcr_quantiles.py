"""Independent empirical closest-record quantiles across execution backends."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabgen_eval import Schema
from tabgen_eval.metrics import DCR
from tabgen_eval.schema.validate import ValidationError
from tests.support.distances import distance_matrices
from tests.support.distances import memorization as _compute
from tests.support.evaluation import memorization_report as evaluate_memorization
from tests.support.options import make_options

pytestmark = pytest.mark.unit


@pytest.fixture
def dtype(backend_dtype):
    return backend_dtype


def _nearest_oracle(reference, query, schema, dtype):
    features = list(schema.numeric) + list(schema.categorical)
    if not len(reference.dropna(subset=features)) or not len(query.dropna(subset=features)):
        return np.array([np.nan])
    (matrix, _) = distance_matrices(reference, [query], schema, dtype)
    return np.sqrt(matrix(1, 0).min(axis=1))


@pytest.mark.parametrize(
    "distances", [[3.0], [1.0, 3.0], [5.0, 1.0, 3.0], [8, 2, 0, 4, 1], list(range(20, -1, -1))]
)
def test_quantile_interpolates_euclidean_distances_and_retains_median(backend, dtype, distances):
    reference = pd.DataFrame({"x": [0.0, 8.0]})
    generated = pd.DataFrame({"x": 8 * (1 + np.asarray(distances))})
    outcome = _compute(
        reference,
        generated,
        Schema(numeric=["x"]),
        backend,
        dtype,
        holdout=pd.DataFrame({"x": [16.0]}),
    )
    assert outcome.values["median"] == np.median(distances)
    assert outcome.values["q05"] == pytest.approx(np.quantile(distances, 0.05), rel=2e-06)
    assert outcome.values["ratio"] == np.median(distances)


@pytest.mark.parametrize("wide", [False, True])
def test_quantile_preserves_typed_categories_duplicates_and_missing_rows(backend, dtype, wide):
    levels = [1, 1.0, "1"] + ([f"level-{i}" for i in range(520)] if wide else ["a"])
    reference = pd.DataFrame({"x": np.arange(len(levels)), "c": pd.Series(levels, dtype=object)})
    generated = reference.iloc[[0, 0, 1, 2, 3]].copy()
    generated["x"] += np.array([0.0, 0.0, 0.25, 0.5, 0.75])
    generated.iloc[2, 1] = "unknown"
    generated = pd.concat([generated, pd.DataFrame({"x": [np.nan], "c": [1]})])
    schema = Schema(numeric=["x"], categorical=["c"])
    outcome = _compute(reference, generated, schema, backend, dtype)
    precision = "float64" if backend == "python" else dtype
    for pool, actual in [(generated, outcome.values)]:
        expected = _nearest_oracle(reference, pool, schema, precision)
        assert actual["median"] == pytest.approx(np.median(expected), abs=2e-06)
        assert actual["q05"] == pytest.approx(np.quantile(expected, 0.05), abs=2e-06)
    assert outcome.diagnostics["usable_rows"] == {
        "training": len(reference),
        "generated": 5,
        "holdout": None,
    }


@pytest.mark.parametrize("invalid", ["reference", "candidate"])
@pytest.mark.parametrize("kind", ["empty", "missing", "numeric_text", "invalid_text", "infinity"])
def test_undefined_quantile_preserves_independently_valid_comparison(backend, dtype, invalid, kind):
    pools = {
        "reference": pd.DataFrame({"x": [0.0, 8.0]}),
        "candidate": pd.DataFrame({"x": [16.0, 32.0]}),
    }
    values = {
        "empty": [],
        "missing": [np.nan],
        "numeric_text": ["16.0"],
        "invalid_text": ["bad"],
        "infinity": [np.inf],
    }[kind]
    pools[invalid] = pd.DataFrame({"x": values})
    if kind not in {"empty", "missing"}:
        with pytest.raises(ValidationError, match="declared numeric|numeric infinity"):
            _compute(*pools.values(), Schema(numeric=["x"]), backend, dtype)
        return
    outcome = _compute(*pools.values(), Schema(numeric=["x"]), backend, dtype)
    assert np.isnan(outcome.values["q05"])
    if invalid == "candidate":
        assert outcome.diagnostics["holdout_median"] is None


def test_python_quantiles_reuse_one_nearest_query_per_pool(monkeypatch):
    from tabgen_eval.execution.python.distance import NearestRecordIndex

    query = NearestRecordIndex.query
    sizes = []

    def counted(self, features):
        sizes.append(features.n_rows)
        return query(self, features)

    monkeypatch.setattr(NearestRecordIndex, "query", counted)
    _compute(
        pd.DataFrame({"x": [0.0, 8.0]}),
        pd.DataFrame({"x": [16.0, 32.0, 40.0]}),
        Schema(numeric=["x"]),
        "reference",
        "float64",
    )
    assert sizes == [3]


@pytest.mark.gpu
def test_cuda_quantiles_share_one_nearest_search_per_query(monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from tabgen_eval.execution.torch.distance import planner

    original = planner.nearest_distances
    calls = []

    def counted(reference, query, workspace, **kwargs):
        calls.append(query.n_rows)
        return original(reference, query, workspace, **kwargs)

    monkeypatch.setattr(planner, "nearest_distances", counted)
    _compute(
        pd.DataFrame({"x": [0.0, 8.0]}),
        pd.DataFrame({"x": [16.0, 32.0, 40.0]}),
        Schema(numeric=["x"]),
        "cuda",
        "float64",
    )
    assert calls == [3]


def test_memorization_point_includes_q05_and_independent_holdout_spacing():
    training = pd.DataFrame({"x": [0.0, 2.0, 8.0]})
    generated = pd.DataFrame({"x": [16.0, 32.0]})
    point = evaluate_memorization(
        training,
        generated,
        Schema(numeric=["x"]),
        holdout=pd.DataFrame({"x": [4.0]}),
        options=make_options(device="cpu", uncertainty="both", n_replicates=3),
    )
    assert point["dcr/q05"].value == pytest.approx(1.1)
    assert point["dcr/median"].value == 2.0
    assert point.results[0].diagnostics["holdout_median"] == 0.25
    assert point["dcr/ratio"].value == 8.0
    assert DCR().stat_spec("q05").display == "DCR (5th percentile)"
    assert point.provenance["n_retained_training"] == 3
    assert point.provenance["n_retained_generated"] == 2
    assert not hasattr(point, "value_replicates")


@pytest.mark.parametrize("query_count", [1, 2, 17])
def test_all_generated_and_holdout_percentiles_match_independent_distances(
    backend, dtype, query_count
):
    training = pd.DataFrame({"x": [0.0, 2.0, 7.0], "c": ["a", "b", "a"]})
    generated = pd.DataFrame({"x": np.linspace(-3, 11, query_count), "c": ["a"] * query_count})
    holdout = pd.DataFrame({"x": [0.0, 1.0, 3.0, 8.0], "c": ["b", "a", "a", "b"]})
    schema = Schema(numeric=["x"], categorical=["c"])
    outcome = _compute(training, generated, schema, backend, dtype, holdout=holdout)
    precision = "float64" if backend == "python" else dtype
    for prefix, pool in [("", generated), ("holdout_", holdout)]:
        distances = _nearest_oracle(training, pool, schema, precision)
        for percentile in [5, 10, 25, 50, 75, 90, 95, 99]:
            assert outcome.values[f"{prefix}q{percentile:02d}"] == pytest.approx(
                np.quantile(distances, percentile / 100, method="linear"), abs=2e-6
            )
    assert outcome.values["q50"] == pytest.approx(outcome.values["median"])
    assert outcome.values["holdout_q50"] == pytest.approx(outcome.diagnostics["holdout_median"])


@pytest.mark.parametrize("holdout_kind", ["missing", "empty", "all_nan", "valid"])
def test_public_percentile_report_and_exports(holdout_kind):
    import json

    training = pd.DataFrame({"x": [0.0, 1.0]})
    generated = pd.DataFrame({"x": [2.0, 4.0]})
    holdout = {
        "missing": None,
        "empty": pd.DataFrame({"x": pd.Series([], dtype=float)}),
        "all_nan": pd.DataFrame({"x": [np.nan]}),
        "valid": pd.DataFrame({"x": [3.0, 7.0]}),
    }[holdout_kind]
    report = evaluate_memorization(
        training,
        generated,
        Schema(numeric=["x"]),
        holdout=holdout,
        options=make_options(device="cpu"),
    )
    exported = json.loads(report.to_json())
    exported_stats = exported["results"]["memorization"]["dcr"]
    table = report.to_dataframe()
    assert len(table) == 18
    for percentile in [5, 10, 25, 50, 75, 90, 95, 99]:
        key = f"q{percentile:02d}"
        assert report[f"dcr/{key}"].value == pytest.approx(
            np.quantile([1.0, 3.0], percentile / 100)
        )
        assert exported_stats[key]["value"] == report[f"dcr/{key}"].value
        held = report[f"dcr/holdout_{key}"]
        if holdout_kind == "valid":
            assert held.value == pytest.approx(np.quantile([2.0, 6.0], percentile / 100))
        else:
            assert exported_stats[f"holdout_{key}"]["value"] is None
            assert np.isnan(held.value)
            flag = "MISSING_HOLDOUT" if holdout_kind == "missing" else "INSUFFICIENT_HOLDOUT_ROWS"
            assert flag in held.flag_codes


def test_categorical_percentile_report_matches_oracle_and_exports(backend, dtype):
    import json

    labels = [1, "1", 1.0, True, "a", "b", "c"]
    training = pd.DataFrame(
        {"x": np.arange(21, dtype=float), "c": pd.Series(labels * 3, dtype=object)}
    )
    generated = training.iloc[[0, 0, 2, 4, 8, 12, 16]].copy()
    generated["x"] += [0, 0, 0.2, 0.5, 0.8, 1, 3]
    generated.iloc[2, 1] = "new-generated"
    generated.iloc[3, 1] = None
    holdout = training.iloc[[1, 3, 5, 9, 15]].copy()
    holdout["x"] += [0.1, 0.3, 0.7, 1, 2]
    holdout.iloc[1, 1] = "new-holdout"
    holdout.iloc[2, 1] = None
    schema = Schema(numeric=["x"], categorical=["c"])
    originals = [frame.copy(deep=True) for frame in [training, generated, holdout]]
    report = evaluate_memorization(
        training,
        generated,
        schema,
        holdout=holdout,
        options=make_options(
            device="cpu" if backend == "python" else backend,
            dtype=dtype,
        ),
    )
    precision = "float64" if backend == "python" else dtype
    exported = json.loads(report.to_json())["results"]["memorization"]["dcr"]
    assert len(report.to_dataframe()) == len(exported) == 18
    for prefix, pool in [("", generated), ("holdout_", holdout)]:
        distances = _nearest_oracle(training, pool, schema, precision)
        for percentile in [5, 10, 25, 50, 75, 90, 95, 99]:
            key = f"{prefix}q{percentile:02d}"
            actual = report[f"dcr/{key}"].value
            assert actual == pytest.approx(np.quantile(distances, percentile / 100), abs=2e-6)
            assert exported[key]["value"] == actual
    assert report["dcr/q50"].value == report["dcr/median"].value
    assert report["dcr/ratio"].value == pytest.approx(
        report["dcr/q50"].value / report["dcr/holdout_q50"].value
    )
    if backend != "python":
        execution = report.provenance["metric_execution"]["dcr"]
        assert execution["dtype"] == dtype
        assert execution["device"].startswith("cuda" if backend.startswith("cuda") else "cpu")
    for original, actual in zip(originals, [training, generated, holdout], strict=True):
        pd.testing.assert_frame_equal(original, actual)
