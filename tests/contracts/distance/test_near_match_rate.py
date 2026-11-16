"""Restored inclusive generated-to-training near matches across exact backends."""

import json

import numpy as np
import pandas as pd
import pytest

from tabgen_eval import Schema, list_metrics
from tabgen_eval.metrics import DCR, NearMatchRate
from tests.support.distances import distance_matrices
from tests.support.evaluation import memorization_report as evaluate_memorization
from tests.support.evaluation import paired_report as evaluate
from tests.support.evaluation import reference_scores as calibrate
from tests.support.metrics import configured_metric
from tests.support.options import make_options

pytestmark = pytest.mark.unit


def options(backend, dtype):
    return make_options(
        device="cpu" if backend == "python" else backend,
        dtype=dtype,
        uncertainty="both",
        n_replicates=3,
    )


def test_inclusive_threshold_and_repeated_generated_copies(backend_precision):
    backend, dtype = backend_precision
    training = pd.DataFrame({"x": [0.0, 1.0]})
    generated = pd.DataFrame({"x": [0.0, 0.0005, 0.01, 1.0, 1.0]})
    schema = Schema(numeric=["x"])
    report = evaluate_memorization(
        training,
        generated,
        schema,
        metrics=["dcr", "near_match_rate"],
        options=options(backend, dtype),
    )
    stat = report["near_match_rate/rate"]
    assert stat.value == pytest.approx(4 / 5)
    assert stat.generated_distribution is None and np.isnan(stat.ideal)
    assert report.provenance["uncertainty"] == "none"
    diagnostic = report.provenance["metric_diagnostics"]["near_match_rate"]
    assert diagnostic["near_match_threshold"] == 1e-3
    assert diagnostic["denominator"] == 5 and diagnostic["query_role"] == "generated"
    assert (
        json.loads(report.to_json())["results"]["memorization"]["near_match_rate"]["rate"]["value"]
        == 4 / 5
    )
    assert len(report.to_dataframe()) == 19


def test_typed_categories_missing_rows_and_unseen_values_match_onehot_oracle(backend_precision):
    backend, dtype = backend_precision
    training = pd.DataFrame(
        {"x": [0.0, 2.0, 8.0, 16.0], "c": pd.Series([1, "1", 1.0, True], dtype=object)}
    )
    generated = training.iloc[[0, 0, 1, 2, 3]].copy()
    generated["x"] += [0, 0, 1, 2, 3]
    generated.iloc[2, 1] = "unseen"
    generated = pd.concat([generated, pd.DataFrame({"x": [np.nan], "c": [1]})], ignore_index=True)
    schema = Schema(numeric=["x"], categorical=["c"])
    threshold = 1e-3
    report = evaluate_memorization(
        training,
        generated,
        schema,
        metrics=["near_match_rate"],
        options=options(backend, dtype),
    )
    matrix, _ = distance_matrices(
        training, [generated], schema, "float64" if backend == "python" else dtype
    )
    nearest = np.sqrt(matrix(1, 0).min(axis=1)).astype(np.float64)
    assert report["near_match_rate/rate"].value == float((nearest <= threshold).mean())
    assert report.provenance["n_retained_generated"] == 5


def test_fixed_threshold_boundary_in_selected_precision(backend_precision):
    backend, dtype = backend_precision
    kind = np.float64 if backend == "python" or dtype == "float64" else np.float32
    below = np.nextafter(kind(1e-3), kind(0))
    above = np.nextafter(kind(1e-3), kind(1))
    training = pd.DataFrame({"x": [0.0, 1.0]})
    for distance, expected in [(below, 1.0), (above, 0.0)]:
        report = evaluate_memorization(
            training,
            pd.DataFrame({"x": [float(distance)]}),
            Schema(numeric=["x"]),
            metrics=["near_match_rate"],
            options=options(backend, dtype),
        )
        assert report["near_match_rate/rate"].value == expected


@pytest.mark.parametrize("missing_role", ["training", "generated"])
def test_no_usable_rows_leave_rate_undefined(backend_precision, missing_role):
    backend, dtype = backend_precision
    pools = {
        "training": pd.DataFrame({"x": [0.0, 1.0]}),
        "generated": pd.DataFrame({"x": [0.0, 1.0]}),
    }
    pools[missing_role]["x"] = np.nan
    report = evaluate_memorization(
        **pools,
        schema=Schema(numeric=["x"]),
        metrics=["near_match_rate"],
        options=options(backend, dtype),
    )
    assert np.isnan(report["near_match_rate/rate"].value)
    assert "NAN_VALUE" in report["near_match_rate/rate"].flag_codes


@pytest.mark.parametrize("threshold", [1e-3, 0, -1, float("nan"), float("inf"), True, "0.1", None])
def test_threshold_is_not_configurable(threshold):
    with pytest.raises(TypeError, match="unexpected kwarg.*near_match_threshold"):
        NearMatchRate(near_match_threshold=threshold)


def test_public_registry_and_data_role_guards():
    entries = {m["name"]: m for m in list_metrics()}
    assert entries["near_match_rate"]["category"] == "memorization"
    assert entries["near_match_rate"]["enabled_by_default"]
    assert "training_coverage" not in entries
    frame = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    with pytest.raises(ValueError, match="comparison metrics"):
        evaluate(frame, frame, Schema.infer(frame), metrics=["near_match_rate"])
    with pytest.raises(ValueError, match="comparison metrics"):
        calibrate(frame, Schema(numeric=["x"]), metrics=["near_match_rate"])
    result = NearMatchRate().compute(frame, frame, Schema(numeric=["x"]), rng=None)
    assert result == {"rate": 1.0}


def test_holdout_independence_row_caps_and_default_selection():
    training = pd.DataFrame({"x": np.arange(20, dtype=float)})
    generated = training.copy()
    schema = Schema(numeric=["x"])
    cfg = make_options(
        device="cpu",
        seed=9,
    )
    alone = evaluate_memorization(
        training,
        generated,
        schema,
        metrics=[configured_metric(NearMatchRate, cfg, max_samples=8)],
        options=cfg,
    )
    for holdout in [training, training + 100]:
        mixed = evaluate_memorization(
            training,
            generated,
            schema,
            holdout=holdout,
            metrics=[configured_metric(cls, cfg, max_samples=8) for cls in [DCR, NearMatchRate]],
            options=cfg,
        )
        assert mixed["near_match_rate/rate"].value == alone["near_match_rate/rate"].value
        assert (
            mixed.provenance["n_selected_training"] == mixed.provenance["n_selected_generated"] == 8
        )
    default = evaluate_memorization(training, generated, schema, options=cfg)
    assert [metric.name for metric in default.results] == ["dcr"]


def test_python_dcr_and_near_matches_share_one_generated_search(monkeypatch):
    from tabgen_eval.execution.python.distance import NearestRecordIndex, compute_memorization_group
    from tabgen_eval.metrics import DCR, NearMatchRate

    calls = []
    original = NearestRecordIndex.query

    def query(self, features):
        calls.append(features.n_rows)
        return original(self, features)

    monkeypatch.setattr(NearestRecordIndex, "query", query)
    training = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    generated = pd.DataFrame({"x": [0.0, 1.1]})
    compute_memorization_group(
        training,
        generated,
        Schema(numeric=["x"]),
        metrics=[DCR(), NearMatchRate()],
    )
    assert calls == [2]


def test_tensor_dcr_and_near_matches_share_one_generated_search(monkeypatch, device, dtype):
    from tabgen_eval.execution.torch.distance.planner import MemorizationPlan

    calls = []
    original = MemorizationPlan._nearest

    def nearest(self, query):
        calls.append(query.n_rows)
        return original(self, query)

    monkeypatch.setattr(MemorizationPlan, "_nearest", nearest)
    training = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    generated = pd.DataFrame({"x": [0.0, 1.1]})
    evaluate_memorization(
        training,
        generated,
        Schema(numeric=["x"]),
        metrics=["dcr", "near_match_rate"],
        options=make_options(device=device, dtype=dtype),
    )
    assert calls == [2]
