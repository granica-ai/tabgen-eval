"""Training-referenced DCR points and independent distance geometry checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabgen_eval import Schema, list_metrics
from tabgen_eval.execution.runtime.dispatch import context_for
from tabgen_eval.metrics import DCR
from tests.support.distances import distance_matrices, memorization
from tests.support.evaluation import memorization_report as evaluate_memorization
from tests.support.evaluation import paired_report as evaluate
from tests.support.options import make_options
from tests.support.reports import without_memory_plans

pytestmark = pytest.mark.unit


def _values(report):
    return {
        metric.name if name == "rate" else name: stat.value
        for metric in report.results
        for (name, stat) in metric.stats.items()
    }


def _numerical_flags(report):
    return report.results[0].diagnostics.get("stat_flags", {})


@pytest.fixture(autouse=True)
def tensor_backend_dependency(request):
    if (
        getattr(request.node, "callspec", None)
        and request.node.callspec.params.get("backend") == "torch"
    ):
        pytest.importorskip("torch")


def cpu_context(*frames):
    pytest.importorskip("torch")
    return context_for(DCR(device="cpu"), *frames)


def frames(seed=0, n=260, wide=False):
    rng = np.random.default_rng(seed)
    real = pd.DataFrame({"x": rng.normal(size=n), "y": rng.normal(size=n)})
    fake = pd.DataFrame({"x": rng.normal(0.3, size=n), "y": rng.normal(size=n)})
    for df in [real, fake]:
        df["c"] = [f"level-{i}" for i in range(n)] if wide else rng.choice(["a", "b", "nan"], n)
    if wide:
        (real["c2"], fake["c2"]) = (real.c, fake.c)
    real.loc[::23, "x"] = np.nan
    fake.loc[::19, "c"] = None
    fake.loc[1, "c"] = "unknown"
    return (real, fake, Schema(numeric=["x", "y"], categorical=["c", "c2"] if wide else ["c"]))


def oracle(reference, candidate, holdout, schema):
    (matrix, _) = distance_matrices(reference, [candidate, holdout], schema)
    held = matrix(2, 0)
    medians = (np.median(np.sqrt(matrix(1, 0).min(1))), np.median(np.sqrt(held.min(1))))
    return (*medians, medians[0] / medians[1])


@pytest.mark.parametrize("wide", [False, True])
def test_dcr_native_and_reference_match_float64_embedding_oracle(wide):
    (real, fake, schema) = frames(n=700 if wide else 260, wide=wide)
    holdout = fake.iloc[::3].copy()
    expected = oracle(real, fake, holdout, schema)
    for backend in ["reference", "cpu"]:
        outcome = memorization(real, fake, schema, backend, holdout=holdout)
        np.testing.assert_allclose(
            [
                outcome.values["median"],
                outcome.diagnostics["holdout_median"],
                outcome.values["ratio"],
            ],
            expected,
            atol=1e-06,
            rtol=1e-06,
        )


def test_even_median_uses_euclidean_distances_and_ratio_of_medians():
    real = pd.DataFrame({"x": [0.0, 8.0]})
    fake = pd.DataFrame({"x": [16.0, 32.0]})
    for backend in ["reference", "cpu"]:
        outcome = memorization(
            real, fake, Schema(numeric=["x"]), backend, holdout=pd.DataFrame({"x": [16.0]})
        )
        assert outcome.values["median"] == 2.0
        assert outcome.diagnostics["holdout_median"] == 1.0
        assert outcome.values["ratio"] == 2.0


def test_generated_queries_cannot_change_the_training_embedding():
    training = pd.DataFrame({"x": [0.0, 8.0], "c": ["a", "b"]})
    generated = pd.DataFrame({"x": [16.0, 32.0], "c": ["a", "unknown"]})
    schema = Schema(numeric=["x"], categorical=["c"])
    holdout = training.assign(x=training.x + 1)
    first = memorization(training, generated, schema, holdout=holdout)
    second = memorization(
        training, generated.assign(x=1000000.0, c="another"), schema, holdout=holdout
    )
    assert first.diagnostics["holdout_median"] == second.diagnostics["holdout_median"]
    assert first.values["median"] != second.values["median"]


def test_memorization_point_ignores_general_uncertainty_settings():
    (training, generated, schema) = frames()
    generated = generated.iloc[:117]
    options = dict(device="cpu", seed=20)
    point = evaluate_memorization(
        training, generated, schema, holdout=generated.iloc[::3], options=make_options(**options)
    )
    configured = evaluate_memorization(
        training,
        generated,
        schema,
        holdout=generated.iloc[::3],
        options=make_options(
            **options,
            uncertainty="both",
            n_replicates=8,
        ),
    )
    np.testing.assert_equal(_values(configured), _values(point))
    assert without_memory_plans(configured.provenance) == without_memory_plans(point.provenance)
    assert configured.results[0].diagnostics == point.results[0].diagnostics
    assert configured.provenance["n_selected_training"] == len(
        training.dropna(subset=schema.numeric + schema.categorical)
    )
    assert configured.provenance["n_selected_generated"] == len(
        generated.dropna(subset=schema.numeric + schema.categorical)
    )
    assert (
        _values(configured)["ratio"]
        == _values(configured)["median"] / configured.results[0].diagnostics["holdout_median"]
    )
    for name in ("ideal", "ci_low", "ci_high", "value_replicates", "ideal_replicates"):
        assert not hasattr(configured, name)
        assert name not in _values(configured)


def test_native_and_python_match_all_memorization_point_statistics():
    pytest.importorskip("torch")
    (training, generated, schema) = frames()
    outcomes = [
        evaluate_memorization(
            training,
            generated.iloc[:173],
            schema,
            options=make_options(device="cpu", seed=23),
        )
        for backend in ["auto", "torch"]
    ]
    for name in _values(outcomes[0]):
        np.testing.assert_allclose(
            _values(outcomes[0])[name],
            _values(outcomes[1])[name],
            atol=1e-06,
            rtol=1e-06,
            equal_nan=True,
        )
    for field in ("holdout_median", "usable_rows", "zero_baseline"):
        assert (
            outcomes[0].results[0].diagnostics[field] == outcomes[1].results[0].diagnostics[field]
        )
    assert _numerical_flags(outcomes[0]) == _numerical_flags(outcomes[1])


def test_parallelism_preserves_memorization_point_and_training_spacing():
    (training, generated, schema) = frames()
    outcomes = [
        evaluate_memorization(
            training,
            generated,
            schema,
            options=make_options(device="cpu", n_jobs=jobs),
        )
        for jobs in [1, 4]
    ]
    np.testing.assert_equal(_values(outcomes[0]), _values(outcomes[1]))
    assert outcomes[0].results[0].diagnostics == outcomes[1].results[0].diagnostics


def test_duplicate_training_zero_spacing_flags_only_ratio():
    training = pd.DataFrame({"x": np.zeros(240)})
    point = evaluate_memorization(
        training,
        training.iloc[:17].copy(),
        Schema(numeric=["x"]),
        holdout=training.iloc[:19].copy(),
        options=make_options(device="cpu", uncertainty="both", n_replicates=3),
    )
    assert _values(point)["median"] == _values(point)["q05"] == 0.0
    assert np.isnan(_values(point)["ratio"])
    assert _numerical_flags(point) == {"ratio": ["ZERO_BASELINE"]}
    assert point.results[0].diagnostics["holdout_median"] == 0.0
    assert point.provenance["n_retained_training"] == 240
    assert point.provenance["n_retained_generated"] == 17


def test_single_training_row_preserves_raw_scores_and_withholds_ratio():
    training = pd.DataFrame({"x": [2.0]})
    generated = pd.DataFrame({"x": [2.0, 3.0]})
    point = evaluate_memorization(
        training,
        generated,
        Schema(numeric=["x"]),
        options=make_options(device="cpu"),
    )
    assert _values(point)["median"] == 0.5
    assert _values(point)["q05"] == pytest.approx(0.05)
    assert np.isnan(_values(point)["ratio"])
    assert point.results[0].diagnostics["holdout_median"] is None
    assert _numerical_flags(point) == {
        name: ["MISSING_HOLDOUT"]
        for name in ["ratio", *[f"holdout_q{p:02d}" for p in [5, 10, 25, 50, 75, 90, 95, 99]]]
    }


@pytest.mark.parametrize("metrics", [["dcr"], ["shape", "dcr"], ["memorization"]])
def test_holdout_evaluation_rejects_training_referenced_dcr(metrics):
    (real, generated, schema) = frames()
    with pytest.raises(ValueError, match="comparison metrics"):
        evaluate(real, generated, schema, metrics=metrics, options=make_options(uncertainty="none"))


def test_native_boundary_rejects_invalid_indices_and_preserves_empty_batches():
    (real, fake, schema) = frames()
    context = cpu_context(real, fake, schema)
    for bad in [-1, len(real)]:
        with pytest.raises(ValueError, match="index"):
            context.compute_memorization_group([DCR()], np.array([[bad]]), np.array([[0]]))
    with pytest.raises(ValueError, match="batch lengths"):
        context.compute_memorization_group([DCR()], np.array([[0]]), np.array([[0], [1]]))
    empty = np.empty((0, 2), dtype=np.int64)
    assert context.compute_memorization_group([DCR()], empty, empty) == []


def test_no_complete_training_yields_undefined_point_scores():
    training = pd.DataFrame({"x": [np.nan] * 240})
    generated = pd.DataFrame({"x": np.zeros(173)})
    point = evaluate_memorization(
        training,
        generated,
        Schema(numeric=["x"]),
        options=make_options(device="cpu"),
    )
    assert all(np.isnan(value) for value in _values(point).values())
    assert _numerical_flags(point) == {
        name: ["MISSING_HOLDOUT"]
        for name in ["ratio", *[f"holdout_q{p:02d}" for p in [5, 10, 25, 50, 75, 90, 95, 99]]]
    }
    assert point.provenance["n_selected_training"] == 0
    assert point.provenance["n_selected_generated"] == 173
    usage = point.results[0].diagnostics["row_usage"]["pools"]["train"]
    assert usage == {"input": 240, "retained": 0, "excluded": 240, "excluded_percent": 100.0}
    assert point.provenance["n_retained_training"] == 0
    assert point.provenance["n_retained_generated"] == 173


def test_dcr_ratio_is_registered_and_custom_compute_is_preserved():
    from tabgen_eval._evaluation.estimator import Estimator

    assert next(m for m in list_metrics() if m["name"] == "dcr_ratio")["enabled_by_default"]

    class CustomDCR(DCR):
        def compute(self, reference, candidate, schema, *, rng):
            return {"median": 4.0, "ratio": 2.0}

    (real, fake, schema) = frames()
    result = Estimator(CustomDCR(), make_options(uncertainty="none")).run(real, fake, schema)
    assert result["median"].value == 4.0 and result["ratio"].value == 2.0


def test_auto_low_dimensional_numeric_dcr_uses_native_exact_search():
    rng = np.random.default_rng(9)
    training = pd.DataFrame({"x": rng.normal(size=240)})
    generated = pd.DataFrame({"x": rng.normal(size=137)})
    point = evaluate_memorization(training, generated, Schema(numeric=["x"]))
    assert np.isfinite(_values(point)["median"])
    assert np.isnan(_values(point)["ratio"])
    assert point.provenance["metric_execution"]["dcr"]["implementation"] == "aten_cpp"
    assert point.provenance["n_selected_training"] == 240
    assert point.provenance["n_selected_generated"] == 137
    assert DCR(device="cpu").device == "cpu"


@pytest.mark.parametrize("dimension", [1, 16])
def test_python_and_native_float64_dcr_preserve_near_copy_distance(dimension):
    columns = [f"x{i}" for i in range(dimension)]
    reference = pd.DataFrame([np.zeros(dimension), np.ones(dimension)], columns=columns)
    candidate = pd.DataFrame([np.ones(dimension)], columns=columns)
    candidate.iloc[0, 0] += 1e-09
    for backend in ["reference", "cpu"]:
        outcome = memorization(reference, candidate, Schema(numeric=columns), backend)
        assert outcome.values["median"] == candidate.iloc[0, 0] - 1.0
        assert outcome.diagnostics["holdout_median"] is None


@pytest.mark.parametrize("backend", ["reference", "cpu"])
@pytest.mark.parametrize("invalid_pool", [0, 1])
def test_invalid_numeric_dcr_raises_before_geometry(backend, invalid_pool):
    schema = Schema(numeric=["x"])
    pools = [pd.DataFrame({"x": [0.0, 1.0, 2.0]}), pd.DataFrame({"x": [0.5, 1.5]})]
    pools[invalid_pool] = pd.concat(
        [pools[invalid_pool], pd.DataFrame({"x": ["bad"]})], ignore_index=True
    )
    with pytest.raises(ValueError, match="declared numeric"):
        memorization(*pools, schema, backend)
