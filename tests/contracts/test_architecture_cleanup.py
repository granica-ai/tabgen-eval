"""Shared execution preserves source draws, numerical streams and single-pool ownership."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

from tabgen_eval import Schema
from tabgen_eval.execution.classifier.protocol import source_folds
from tabgen_eval.execution.runtime.comparisons import comparison_task
from tabgen_eval.execution.runtime.dispatch import ExecutionSession
from tabgen_eval.execution.runtime.paired import comparison_rng
from tabgen_eval.metrics import LinearC2ST, Shape
from tabgen_eval.preprocessing.columns import PreparedColumns
from tests.support.evaluation import paired_report as evaluate
from tests.support.evaluation import reference_scores as calibrate
from tests.support.evaluation import summaries
from tests.support.metrics import configured_metric
from tests.support.options import make_options

pytestmark = pytest.mark.unit


def frames():
    rng = np.random.default_rng(19)
    return pd.DataFrame({"x": rng.normal(size=48), "c": rng.choice(["a", "b"], 48)})


def test_explicit_minimum_applies_to_evaluation_and_real_only_calibration(monkeypatch):
    real = frames()
    schema = Schema(numeric=["x"], categorical=["c"])
    options = make_options(
        device="cpu",
        uncertainty="reference",
        n_replicates=2,
    )

    def unexpected(*args, **kwargs):
        raise AssertionError("An explicitly insufficient draw must not enter a numerical core")

    from tabgen_eval.execution.torch.context import TorchContext

    monkeypatch.setattr(TorchContext, "compute", unexpected)
    evaluation = evaluate(
        real,
        real.iloc[:24],
        schema=schema,
        metrics=[configured_metric(Shape, options, n_min=30)],
        options=options,
    )
    reference = calibrate(
        real,
        schema=schema,
        metrics=[configured_metric(Shape, options, n_min=30)],
        options=options,
        n_replicates=2,
    )
    assert np.isnan(evaluation["shape/shape"].value)
    assert np.isnan(evaluation["shape/shape"].reference_distribution.samples).all()
    distribution = summaries(reference.metrics["shape"])["shape"]
    assert np.isnan(distribution.samples).all()
    assert distribution.requested_count == distribution.unavailable_count == 2
    assert reference.metrics["shape"].diagnostics[0]["n_min"] == 30


def test_real_only_python_preparation_keeps_caller_values(monkeypatch):
    import tabgen_eval.execution.runtime.preparation as engine

    real = pd.DataFrame(
        {"x": [1.0, np.nan, 3.0, 4.0, 5.0, 6.0], "c": [None, " a ", "b", "a", "nan", "a"]}
    )
    before = real.copy(deep=True)
    schema = Schema(numeric=["x"], categorical=["c"])
    seen = []
    original = engine.prepare_evaluation_sources

    def capture(frame, candidate, *args, **kwargs):
        seen.append((frame, candidate))
        return original(frame, candidate, *args, **kwargs)

    monkeypatch.setattr(engine, "prepare_evaluation_sources", capture)
    result = calibrate(real, schema, metrics=["shape"], n_replicates=2, options=make_options())
    assert len(seen) == 1 and seen[0][0] is real and seen[0][1] is real
    assert "cleaning" not in result.metadata
    pd.testing.assert_frame_equal(real, before)


def test_real_only_arrow_packs_and_factorizes_one_pool(monkeypatch):
    pytest.importorskip("pyarrow")
    from tabgen_eval.preprocessing import arrow as arrow_prepared

    real = pd.DataFrame({"x": [1.1, 2.2, 3.3, 4.4], "c": ["1", " 1 ", None, "nan"]})
    before = real.copy(deep=True)
    schema = Schema(numeric=["x"], categorical=["c"])
    pools, categories = [], []
    original_pool = arrow_prepared._pool_columns
    original_factor = arrow_prepared.factorize_categories

    def pool(*args, **kwargs):
        pools.append(args[0])
        return original_pool(*args, **kwargs)

    def factor(*args):
        categories.append(len(args))
        return original_factor(*args)

    monkeypatch.setattr(arrow_prepared, "_pool_columns", pool)
    monkeypatch.setattr(arrow_prepared, "factorize_categories", factor)
    columns = arrow_prepared.try_prepare_single(real, schema, retain_labels=True)
    assert len(pools) == 1 and pools[0] is real
    assert categories == [1]
    assert columns.reference is columns.candidate
    assert not columns.reference.numeric.flags.writeable
    assert set(columns.category_values[0]) >= {"1", " 1 ", "nan"}
    pd.testing.assert_frame_equal(real, before)


def test_single_pool_pandas_factorizes_once_and_routing_counts_unique_storage(monkeypatch):
    import tabgen_eval.preprocessing.columns as prepared

    real = frames()
    schema = Schema(numeric=["x"], categorical=["c"])
    calls = []
    original = prepared.factorize_categories

    def factor(*args):
        calls.append(len(args))
        return original(*args)

    monkeypatch.setattr(prepared, "factorize_categories", factor)
    session = ExecutionSession(real, real, schema)
    columns = session.columns
    assert columns.reference is columns.candidate
    assert calls == [1]
    metric = Shape(device="cpu")
    workload = session._workload(metric)
    expected = len(real) * (8 + 8 + 1)
    assert workload.shared_source
    assert workload.persistent_bytes("float64") == expected
    assert workload.persistent_bytes("float32") == len(real) * (4 + 8 + 1)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_single_pool_transfer_and_precision_copy_keep_owned_storage_shared(device, monkeypatch):
    torch = pytest.importorskip("torch")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from tabgen_eval.execution.torch.tensors import TensorInputs, TensorTable

    real = frames()
    columns = PreparedColumns.prepare_single(real, Schema(numeric=["x"], categorical=["c"]))
    original = TensorTable.from_host
    calls = []

    def from_host(cls, table, **kwargs):
        calls.append(table)
        return original(table, **kwargs)

    monkeypatch.setattr(TensorTable, "from_host", classmethod(from_host))
    inputs = TensorInputs.from_columns(columns, device=device)
    assert len(calls) == 1
    assert inputs.reference is inputs.candidate
    assert inputs.nbytes == inputs.reference.nbytes
    copied = inputs.to(device=device, dtype="float32")
    assert copied.reference is copied.candidate
    assert copied.reference is not inputs.reference
    assert copied.nbytes == copied.reference.nbytes
    # Source arrays stay independent of tensor storage, including on CPU.
    before = columns.reference.numeric.copy()
    inputs.reference.numeric.fill_(123)
    np.testing.assert_array_equal(columns.reference.numeric, before)


@pytest.mark.parametrize("ideal,replicate", [(False, None), (True, None), (False, 2), (True, 2)])
def test_common_classifier_task_preserves_paired_fit_stream(ideal, replicate):
    metric = LinearC2ST()
    left, right = np.arange(24), np.arange(24, 48)
    key = int.from_bytes(hashlib.sha256(metric.name.encode()).digest()[:4], "little")
    spawn = (0xF17ED, key, int(ideal)) + (() if replicate is None else (replicate + 1,))
    rng = np.random.default_rng(np.random.SeedSequence(137, spawn_key=spawn))
    expected_folds = source_folds(24, 24, metric.cv_folds, rng)
    expected_seed = int(rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64))
    task = comparison_task(
        metric, left, right, rng=comparison_rng(metric, seed=137, ideal=ideal, replicate=replicate)
    )
    np.testing.assert_array_equal(task.reference_folds, expected_folds[0])
    np.testing.assert_array_equal(task.candidate_folds, expected_folds[1])
    assert task.fit_seed == expected_seed


def test_all_classifier_routes_retain_common_diagnostics_without_changing_seeds(monkeypatch):
    from tabgen_eval.execution.runtime.outcomes import ClassifierOutcome

    real = frames()
    schema = Schema(numeric=["x"], categorical=["c"])

    def fitted(self, reference, candidate, schema, rf, sf, *, seed, point=False, fit_threads=1):
        return ClassifierOutcome(0.5 + (seed % 17) / 100, [{"converged": False}])

    monkeypatch.setattr(LinearC2ST, "compute_partitioned", fitted)
    # Keep the monkeypatched fit callback in this process for every comparison.
    options = make_options(uncertainty="both", n_replicates=2, seed=137, n_jobs=1)
    evaluated = evaluate(
        real, real.iloc[:24], schema=schema, metrics=["linear_c2st"], options=options
    )
    calibrated = calibrate(
        real, schema=schema, metrics=["linear_c2st"], options=options, n_replicates=2
    )
    diagnostics = evaluated.results[0].diagnostics
    records = [
        *diagnostics["point"].values(),
        *diagnostics["replicates"]["generated"],
        *diagnostics["replicates"]["real_reference"],
        *calibrated.metrics["linear_c2st"].diagnostics,
    ]
    for record in records:
        assert {"raw_value", "converged", "folds", "fit_seed", "point"} <= record.keys()
        assert record["point"] is True and record["converged"] is False
    assert summaries(calibrated.metrics["linear_c2st"])["auc"].finite_count == 2


@pytest.mark.parametrize("point", [False, True])
def test_explicit_builtin_classifier_partitions_use_native_execution(monkeypatch, point):
    import tabgen_eval.metrics.c2st as classifiers
    from tabgen_eval.execution.runtime.outcomes import ClassifierOutcome

    real, generated = frames(), frames().iloc[:30].copy()
    schema = Schema(numeric=["x"], categorical=["c"])
    metric = LinearC2ST(device="cpu")
    real_folds = np.arange(len(real)) % metric.cv_folds
    generated_folds = np.arange(len(generated)) % metric.cv_folds
    expected = ClassifierOutcome(0.625, [{"converged": True}])
    calls = []

    class NativeContext:
        def compute_classifier(self, current, rd, sd, rf, sf, seeds, *, point, fit_threads):
            assert current is metric
            np.testing.assert_array_equal(rd, np.arange(len(real))[None])
            np.testing.assert_array_equal(sd, np.arange(len(generated))[None])
            np.testing.assert_array_equal(rf, real_folds[None])
            np.testing.assert_array_equal(sf, generated_folds[None])
            assert seeds == [137] and fit_threads == 3
            calls.append(point)
            return [expected]

    def context(current, reference, candidate, declared, *, n_jobs):
        assert current is metric and reference is real and candidate is generated
        assert declared is schema and n_jobs == 3
        return NativeContext()

    def unexpected(*args, **kwargs):
        pytest.fail("An unchanged built-in classifier must use its native implementation")

    monkeypatch.setattr(classifiers, "context_for", context)
    monkeypatch.setattr(classifiers.classifier_callbacks, "compute_partitioned", unexpected)
    result = metric.compute_partitioned(
        real,
        generated,
        schema,
        real_folds,
        generated_folds,
        seed=137,
        point=point,
        fit_threads=3,
    )
    assert result is expected and calls == [point]


@pytest.mark.parametrize("override", ["class", "instance"])
def test_custom_classifier_fit_hook_retains_generic_partition_runner(monkeypatch, override):
    import tabgen_eval.metrics.c2st as classifiers
    from tabgen_eval.execution.runtime.outcomes import ClassifierOutcome

    def fit(train, labels, test, *, seed, point, fit_threads=1):
        return np.zeros(len(test)), {"converged": True}

    if override == "class":

        class CustomLinear(LinearC2ST):
            _fit_predict = staticmethod(fit)

        metric = CustomLinear()
    else:
        metric = LinearC2ST()
        metric._fit_predict = fit
    real, generated = frames(), frames().iloc[:30].copy()
    schema = Schema(numeric=["x"], categorical=["c"])
    real_folds = np.arange(len(real)) % metric.cv_folds
    generated_folds = np.arange(len(generated)) % metric.cv_folds
    expected = ClassifierOutcome(0.5, [{"converged": True}])
    calls = []

    def partitioned(reference, candidate, declared, rf, sf, **kwargs):
        assert reference is real and candidate is generated and declared is schema
        assert rf is real_folds and sf is generated_folds
        assert kwargs["fit_predict"] is fit
        assert kwargs["seed"] == 137 and kwargs["point"] is False
        assert kwargs["fit_threads"] == 3
        assert kwargs["cv_folds"] == metric.cv_folds
        assert kwargs["memory_limit_mb"] == metric.memory_limit_mb
        assert kwargs["dtype"] == metric.dtype
        calls.append(1)
        return expected

    def unexpected(*args, **kwargs):
        pytest.fail("Native execution must not bypass a custom classifier fit hook")

    monkeypatch.setattr(classifiers, "context_for", unexpected)
    monkeypatch.setattr(classifiers.classifier_callbacks, "compute_partitioned", partitioned)
    result = metric.compute_partitioned(
        real,
        generated,
        schema,
        real_folds,
        generated_folds,
        seed=137,
        point=False,
        fit_threads=3,
    )
    assert result is expected and calls == [1]


def test_custom_classifier_without_fit_hook_has_a_clear_error():
    class IncompleteLinear(LinearC2ST):
        pass

    metric = IncompleteLinear()
    data = np.zeros((6, 1))
    with pytest.raises(NotImplementedError, match="_fit_predict"):
        metric._fit_predict(data, np.zeros(6), data, seed=137, point=True)
