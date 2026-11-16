"""Independent contracts for fixed generated and paired real-real comparisons."""

from __future__ import annotations

from dataclasses import fields

import numpy as np
import pandas as pd
import pytest

from tabgen_eval import Schema
from tabgen_eval.execution.runtime.outcomes import FixedMetricResult
from tabgen_eval.execution.runtime.sampling import fixed_draw
from tabgen_eval.metrics import PRDC, Conditional, Shape, Trend
from tests.support.options import make_options, replace_options

pytestmark = pytest.mark.unit


def _evaluate(*args, seed=None, options=None, **kwargs):
    from tests.support.evaluation import paired_report as evaluate

    options = options or make_options()
    options = replace_options(options, seed=options["seed"] if seed is None else seed)
    return evaluate(*args, options=options, **kwargs)


def _metric(report, name):
    return next(result for result in report.results if result.name == name)


def _values(result, *, reference=False):
    return {name: stat.ideal if reference else stat.value for name, stat in result.stats.items()}


def _oracle(n_holdout, n_generated, seed):
    n = min(n_generated, n_holdout // 2)
    real_seed, generated_seed = np.random.SeedSequence(seed).spawn(2)
    positions = np.random.default_rng(real_seed).choice(n_holdout, 2 * n, replace=False)
    generated = (
        np.arange(n_generated)
        if n_generated == n
        else np.random.default_rng(generated_seed).choice(n_generated, n, replace=False)
    )
    return positions[:n], positions[n:], generated


@pytest.mark.parametrize(
    "n_holdout,n_generated",
    [(23, 3), (23, 30), (101, 30), (3, 1)],
)
def test_fixed_draw_matches_independent_ordered_full_source_oracle(n_holdout, n_generated):
    actual = fixed_draw(n_holdout, n_generated, seed=93)
    expected = _oracle(n_holdout, n_generated, 93)
    for values, oracle in zip(
        [actual.reference, actual.real_candidate, actual.generated], expected, strict=True
    ):
        np.testing.assert_array_equal(values, oracle)
        assert values.dtype == np.int64
        assert not values.flags.writeable
        assert len(values) == actual.n_res
        assert len(np.unique(values)) == len(values)
    assert set(actual.reference).isdisjoint(actual.real_candidate)
    assert actual.n_res == min(n_generated, n_holdout // 2)
    if n_generated == actual.n_res:
        np.testing.assert_array_equal(actual.generated, np.arange(n_generated))


def test_automatic_draw_does_not_truncate_sources_to_a_prefix():
    draw = fixed_draw(101, 200, seed=114)
    assert len(draw.reference) == len(draw.real_candidate) == len(draw.generated) == 50
    assert max(draw.reference.max(), draw.real_candidate.max()) >= 10
    assert draw.generated.max() >= 5


@pytest.mark.parametrize("budget", [True, 0, 1, -1, 24, 2.5, "10"])
def test_removed_real_budget_is_rejected(budget):
    with pytest.raises((TypeError, ValueError)):
        fixed_draw(23, 10, sample_budget=budget, seed=1)


@pytest.mark.parametrize("n_generated", [True, 0, -1, 2.5])
def test_invalid_generated_count_is_rejected(n_generated):
    with pytest.raises((TypeError, ValueError)):
        fixed_draw(23, n_generated, seed=1)


def _frames():
    rng = np.random.default_rng(291)
    frames = []
    for n, shift in [(241, 0.0), (163, 0.3)]:
        x = rng.normal(shift, size=n)
        frame = pd.DataFrame(
            {"x": x, "y": x * 0.4 + rng.normal(size=n), "c": rng.choice(["a", "b", "nan"], n)}
        )
        frame.index = np.repeat(7, n)
        frames.append(frame)
    return *frames, Schema(numeric=["x", "y"], categorical=["c"])


@pytest.mark.parametrize(
    "metric",
    [
        Shape(),
        Trend(),
        Conditional(min_count=2, num_bins=3, rare_threshold=0.2),
        PRDC(nearest_k=3),
    ],
    ids=lambda metric: metric.name,
)
def test_fixed_metric_matches_two_manual_comparisons_against_exact_same_reference(metric):
    holdout, generated, schema = _frames()
    options = make_options()
    result = _evaluate(
        holdout,
        generated,
        schema,
        metrics=[metric],
        seed=111,
        options=options,
    )
    reference, real_candidate, fake = _oracle(len(holdout), len(generated), 111)
    expected_generated = metric.compute(
        holdout.iloc[reference], generated.iloc[fake], schema, rng=np.random.default_rng(0)
    )
    expected_real = metric.compute(
        holdout.iloc[reference], holdout.iloc[real_candidate], schema, rng=np.random.default_rng(0)
    )
    for name in metric.stat_names:
        np.testing.assert_equal(
            _values(_metric(result, metric.name))[name], expected_generated[name]
        )
        np.testing.assert_equal(
            _values(_metric(result, metric.name), reference=True)[name], expected_real[name]
        )
    assert result.provenance["n_res"] == 120
    assert "sample_budget" not in result.provenance


def test_custom_metric_receives_shared_reference_positions_and_two_only_comparisons(monkeypatch):
    holdout, generated, schema = _frames()
    holdout["source_id"] = np.arange(len(holdout))
    generated["source_id"] = -np.arange(1, len(generated) + 1)
    calls = []

    def record(self, reference, candidate, declared, *, rng):
        calls.append((reference.source_id.to_numpy().copy(), candidate.source_id.to_numpy().copy()))
        return {"shape": 0.25}

    class RecordingShape(Shape):
        compute = record

    result = _evaluate(
        holdout,
        generated,
        schema,
        metrics=[RecordingShape()],
        seed=121,
        options=make_options(),
    )
    reference, real_candidate, fake = _oracle(len(holdout), len(generated), 121)
    assert len(calls) == 2
    generated_call = next(call for call in calls if np.all(call[1] < 0))
    real_call = next(call for call in calls if np.all(call[1] >= 0))
    np.testing.assert_array_equal(generated_call[0], reference)
    np.testing.assert_array_equal(real_call[0], reference)
    np.testing.assert_array_equal(generated_call[1], -(fake + 1))
    np.testing.assert_array_equal(real_call[1], real_candidate)
    assert _values(_metric(result, "shape"))["shape"] == 0.25
    assert _values(_metric(result, "shape"), reference=True)["shape"] == 0.25


def test_uncertainty_reports_repetition_means_instead_of_extra_points():
    holdout, generated, schema = _frames()
    first = _evaluate(
        holdout,
        generated,
        schema,
        metrics=["shape", "trend"],
        seed=111,
        options=make_options(uncertainty="none"),
    )
    other = _evaluate(
        holdout,
        generated,
        schema,
        metrics=["shape", "trend"],
        seed=111,
        options=make_options(
            uncertainty="both",
            n_replicates=2,
        ),
    )
    assert first.provenance["n_res"] == other.provenance["n_res"] == 120
    for name in ["shape", "trend"]:
        for stat in _metric(other, name).stats.values():
            assert stat.value == stat.generated_distribution.mean
            assert stat.ideal == stat.reference_distribution.mean
            assert stat.value_estimate == stat.ideal_estimate == "mean"
    assert set(field.name for field in fields(FixedMetricResult)) == {
        "name",
        "values",
        "real_reference",
        "execution",
        "diagnostics",
        "details",
    }


def test_metric_order_and_unrelated_metrics_do_not_change_fixed_rows_or_scores():
    holdout, generated, schema = _frames()
    common = dict(seed=111, options=make_options())
    first = _evaluate(holdout, generated, schema, metrics=["shape", "trend"], **common)
    second = _evaluate(holdout, generated, schema, metrics=["trend", "shape"], **common)
    alone = _evaluate(holdout, generated, schema, metrics=["trend"], **common)
    for result in [first, second]:
        assert _values(_metric(result, "trend")) == _values(_metric(alone, "trend"))
        assert _values(_metric(result, "trend"), reference=True) == _values(
            _metric(alone, "trend"), reference=True
        )


def test_small_fixed_comparisons_have_no_global_minimum_size_gate():
    holdout = pd.DataFrame({"x": np.arange(3.0)})
    generated = pd.DataFrame({"x": [1.0]})
    result = _evaluate(
        holdout,
        generated,
        Schema(numeric=["x"]),
        metrics=["shape"],
        seed=23,
        options=make_options(),
    )
    assert result.provenance["n_res"] == 1
    assert np.isfinite(_values(_metric(result, "shape"))["shape"])
    assert np.isfinite(_values(_metric(result, "shape"), reference=True)["shape"])


def test_input_frames_are_not_modified_by_fixed_preparation_and_scoring():
    holdout, generated, schema = _frames()
    before = [holdout.copy(deep=True), generated.copy(deep=True)]
    _evaluate(
        holdout,
        generated,
        schema,
        metrics=["shape", "trend", "conditional_fidelity"],
        seed=71,
        options=make_options(),
        details=True,
    )
    pd.testing.assert_frame_equal(holdout, before[0], check_exact=True)
    pd.testing.assert_frame_equal(generated, before[1], check_exact=True)


@pytest.mark.parametrize("unsupported", ["dcr", "mle"])
def test_metrics_requiring_different_protocols_are_rejected(unsupported):
    holdout, generated, schema = _frames()
    with pytest.raises(ValueError):
        _evaluate(
            holdout,
            generated,
            schema,
            metrics=[unsupported],
            options=make_options(),
        )


def test_details_collect_only_the_generated_point_without_a_second_metric_run(monkeypatch):
    from tabgen_eval.execution.torch.cpu.executor import CppExecutor

    holdout, generated, schema = _frames()
    calls = []

    def instrument(name):
        original = getattr(CppExecutor, name)

        def counted(self, *args, **kwargs):
            calls.append(1)
            return original(self, *args, **kwargs)

        monkeypatch.setattr(CppExecutor, name, counted)

    instrument("compute")
    instrument("compute_detailed")
    common = dict(
        metrics=["shape"],
        seed=121,
        options=make_options(),
    )
    ordinary = _evaluate(holdout, generated, schema, **common)
    count = len(calls)
    calls.clear()
    captured = _evaluate(holdout, generated, schema, details=True, **common)
    assert len(calls) == count == 2
    assert _values(_metric(captured, "shape")) == _values(_metric(ordinary, "shape"))
    assert _values(_metric(captured, "shape"), reference=True) == _values(
        _metric(ordinary, "shape"), reference=True
    )
    assert _metric(ordinary, "shape").details is None
    assert _metric(captured, "shape").details.metadata["n_reference"] == 120
    assert _metric(captured, "shape").details.metadata["n_candidate"] == 120


def test_python_classifier_uses_private_comparison_streams_and_retains_failed_point_values(
    monkeypatch,
):
    import hashlib

    from tabgen_eval.execution.classifier.protocol import source_folds
    from tabgen_eval.execution.runtime.outcomes import ClassifierOutcome
    from tabgen_eval.metrics import LinearC2ST

    holdout, generated, schema = _frames()
    holdout["source_id"] = np.arange(len(holdout))
    generated["source_id"] = -np.arange(1, len(generated) + 1)
    reference, real_candidate, fake = _oracle(len(holdout), len(generated), 57)
    calls = []

    def record(self, real, candidate, declared, rf, sf, *, seed, point=False, fit_threads=1):
        comparison = 0 if np.all(candidate.source_id < 0) else 1
        metric_key = int.from_bytes(hashlib.sha256(self.name.encode()).digest()[:4], "little")
        rng = np.random.default_rng(
            np.random.SeedSequence(57, spawn_key=(0xF17ED, metric_key, comparison))
        )
        expected_rf, expected_sf = source_folds(120, 120, self.cv_folds, rng)
        expected_seed = int(rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64))
        assert point is True
        assert seed == expected_seed
        np.testing.assert_array_equal(real.source_id, reference)
        np.testing.assert_array_equal(
            candidate.source_id, -(fake + 1) if comparison == 0 else real_candidate
        )
        np.testing.assert_array_equal(rf, expected_rf)
        np.testing.assert_array_equal(sf, expected_sf)
        calls.append(comparison)
        return ClassifierOutcome(
            0.625 if comparison == 0 else 0.5, [{"converged": comparison == 1}]
        )

    monkeypatch.setattr(LinearC2ST, "compute_partitioned", record)
    result = _evaluate(
        holdout,
        generated,
        schema,
        metrics=["linear_c2st"],
        seed=57,
        options=make_options(),
    )
    assert sorted(calls) == [0, 1]
    metric = _metric(result, "linear_c2st")
    assert metric.stats["auc"].value == 0.625
    assert metric.stats["auc"].ideal == 0.5
    assert metric.diagnostics["point"]["generated"]["converged"] is False
    assert metric.diagnostics["point"]["generated"]["raw_value"] == 0.625
    assert metric.diagnostics["point"]["real_reference"]["converged"] is True


def test_fixed_sampler_accepts_sequence_entropy_without_changing_global_rng():
    state = np.random.get_state()
    seed = [7, 2**40, 99]
    actual = fixed_draw(401, 300, seed=seed)
    expected = _oracle(401, 300, seed)
    for a, e in zip(
        [actual.reference, actual.real_candidate, actual.generated], expected, strict=True
    ):
        np.testing.assert_array_equal(a, e)
    after = np.random.get_state()
    assert state[0] == after[0]
    np.testing.assert_array_equal(state[1], after[1])
    assert state[2:] == after[2:]


def test_cpu_fixed_path_does_not_import_optional_gpu_or_model_engines():
    import subprocess
    import sys

    code = """
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'triton', 'xgboost'}:
            raise AssertionError('unexpected optional engine import: ' + fullname)
sys.meta_path.insert(0, Block())
import numpy as np
import pandas as pd
from tabgen_eval import Schema
from tests.support.options import make_options
from tests.support.evaluation import paired_report as evaluate
real = pd.DataFrame({'x': np.arange(240.), 'y': np.arange(240.) ** 2, 'c': [' A ', 'nan'] * 120})
result = evaluate(real, real.iloc[:120], Schema(numeric=['x', 'y'], categorical=['c']),
    metrics=['shape', 'trend', 'conditional_fidelity'],
    options=make_options(uncertainty="none"), details=True)
assert result.provenance['n_res'] == 120
assert result.details('shape') is not None
assert 'torch' in sys.modules
assert not {'triton', 'xgboost'}.intersection(sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], capture_output=True, check=True, timeout=30)
