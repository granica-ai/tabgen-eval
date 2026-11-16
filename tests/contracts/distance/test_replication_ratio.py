"""Restored training coverage uses reverse exact search in training geometry."""

import json

import numpy as np
import pandas as pd
import pytest

from tabgen_eval import Schema, list_metrics
from tabgen_eval.metrics import DCR, NearMatchRate, ReplicationRatio
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
    )


def test_reverse_query_inclusive_threshold_and_training_denominator(backend_precision):
    backend, dtype = backend_precision
    training = pd.DataFrame({"x": [0.0, 0.0, 0.0005, 0.01, 1.0]})
    generated = pd.DataFrame({"x": [0.0] * 9})
    report = evaluate_memorization(
        training,
        generated,
        Schema(numeric=["x"]),
        metrics=["dcr", "near_match_rate", "replication_ratio"],
        options=options(backend, dtype),
    )
    assert report["replication_ratio/rate"].value == pytest.approx(3 / 5)
    assert report["near_match_rate/rate"].value == 1
    stat = report["replication_ratio/rate"]
    assert stat.generated_distribution is None and np.isnan(stat.ideal)
    details = report.provenance["metric_diagnostics"]["replication_ratio"]
    assert details["denominator"] == 5 and details["query_role"] == "training"
    assert details["near_match_threshold"] == 1e-3
    exported = json.loads(report.to_json())
    assert exported["results"]["memorization"]["replication_ratio"]["rate"]["value"] == stat.value


def test_reverse_geometry_and_categories_match_training_fitted_oracle(backend_precision):
    backend, dtype = backend_precision
    training = pd.DataFrame({"x": [0.0, 2.0, 8.0, 16.0], "c": ["a", "a", "b", "b"]})
    generated = pd.DataFrame({"x": [0.0, 1.0, 3.0, np.nan], "c": ["a", "a", "new", "b"]})
    schema = Schema(numeric=["x"], categorical=["c"])
    threshold = 1e-3
    result = evaluate_memorization(
        training,
        generated,
        schema,
        metrics=["replication_ratio"],
        options=options(backend, dtype),
    )
    matrix, _ = distance_matrices(
        training, [generated], schema, "float64" if backend == "python" else dtype
    )
    nearest = np.sqrt(matrix(0, 1).min(axis=1)).astype(np.float64)
    assert result["replication_ratio/rate"].value == pytest.approx((nearest <= threshold).mean())
    assert result.provenance["n_retained_generated"] == 3


@pytest.mark.parametrize("missing_role", ["training", "generated"])
def test_no_usable_rows(backend_precision, missing_role):
    backend, dtype = backend_precision
    pools = {role: pd.DataFrame({"x": [0.0, 1.0]}) for role in ["training", "generated"]}
    pools[missing_role]["x"] = np.nan
    report = evaluate_memorization(
        **pools,
        schema=Schema(numeric=["x"]),
        metrics=["replication_ratio"],
        options=options(backend, dtype),
    )
    value = report["replication_ratio/rate"].value
    assert np.isnan(value) if missing_role == "training" else value == 0


def test_registry_guards_and_direct_compute():
    entries = {m["name"]: m for m in list_metrics()}
    assert entries["replication_ratio"]["category"] == "memorization"
    assert entries["replication_ratio"]["enabled_by_default"]
    assert "training_coverage" not in entries
    frame = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    with pytest.raises(ValueError, match="comparison metrics"):
        evaluate(frame, frame, Schema.infer(frame), metrics=["replication_ratio"])
    with pytest.raises(ValueError, match="comparison metrics"):
        calibrate(frame, Schema(numeric=["x"]), metrics=["replication_ratio"])
    assert ReplicationRatio().compute(frame, frame, Schema(numeric=["x"]), rng=None) == {"rate": 1}
    with pytest.raises(TypeError, match="unexpected kwarg.*near_match_threshold"):
        ReplicationRatio(near_match_threshold=-1)


def test_sampling_holdout_independence_and_category_selection():
    frame = pd.DataFrame({"x": np.arange(20, dtype=float)})
    schema = Schema(numeric=["x"])
    options = make_options(
        device="cpu",
        seed=9,
    )
    alone = evaluate_memorization(
        frame,
        frame,
        schema,
        metrics=[configured_metric(ReplicationRatio, options, max_samples=8)],
        options=options,
    )
    combined = evaluate_memorization(
        frame,
        frame,
        schema,
        holdout=frame + 100,
        metrics=[
            configured_metric(cls, options, max_samples=8)
            for cls in [DCR, NearMatchRate, ReplicationRatio]
        ],
        options=options,
    )
    assert {m.name for m in combined.results} == {"dcr", "near_match_rate", "replication_ratio"}
    assert alone["replication_ratio/rate"].value == combined["replication_ratio/rate"].value
    assert (
        combined.provenance["n_selected_training"]
        == combined.provenance["n_selected_generated"]
        == 8
    )
    assert [
        m.name for m in evaluate_memorization(frame, frame, schema, options=options).results
    ] == ["dcr"]


def test_reverse_fixed_threshold_boundary(backend_precision):
    backend, dtype = backend_precision
    kind = np.float64 if backend == "python" or dtype == "float64" else np.float32
    below = np.nextafter(kind(1e-3), kind(0))
    above = np.nextafter(kind(1e-3), kind(1))
    for distance, expected in [(below, 1 / 3), (above, 0.0)]:
        training = pd.DataFrame({"x": [0.0, float(distance), 1.0]})
        generated = pd.DataFrame({"x": [2 * float(distance)]})
        report = evaluate_memorization(
            training,
            generated,
            Schema(numeric=["x"]),
            metrics=["replication_ratio"],
            options=options(backend, dtype),
        )
        assert report["replication_ratio/rate"].value == pytest.approx(expected)


def test_shared_searches_and_replication_only_skips_forward(backend_precision, monkeypatch):
    backend, dtype = backend_precision
    calls = []
    if backend == "python":
        from tabgen_eval.execution.python.distance import NearestRecordIndex

        original = NearestRecordIndex.query

        def query(self, features):
            calls.append(features.n_rows)
            return original(self, features)

        monkeypatch.setattr(NearestRecordIndex, "query", query)
    else:
        from tabgen_eval.execution.torch.distance.planner import MemorizationPlan

        original = MemorizationPlan._nearest

        def query(self, features, **kwargs):
            calls.append(features.n_rows)
            return original(self, features, **kwargs)

        monkeypatch.setattr(MemorizationPlan, "_nearest", query)
    training = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    generated = pd.DataFrame({"x": [0.0, 1.1]})
    for metrics, expected_calls in [
        (["dcr", "near_match_rate", "replication_ratio"], [2, 3]),
        (["replication_ratio"], [3]),
    ]:
        calls.clear()
        evaluate_memorization(
            training,
            generated,
            Schema(numeric=["x"]),
            metrics=metrics,
            options=options(backend, dtype),
        )
        assert calls == expected_calls


@pytest.mark.parametrize("metric", [NearMatchRate, ReplicationRatio])
@pytest.mark.parametrize("threshold", [1e-3, 0.1])
def test_metric_threshold_override_is_rejected(metric, threshold):
    with pytest.raises(TypeError, match="unexpected kwarg.*near_match_threshold"):
        metric(near_match_threshold=threshold)
