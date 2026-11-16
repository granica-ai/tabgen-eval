"""Public role-based evaluation contracts and numerical procedure parity."""

import importlib
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import tabgen_eval
from tabgen_eval import EvalData, Report, Schema, evaluate, list_metrics
from tests.support.evaluation import memorization_report as evaluate_memorization
from tests.support.evaluation import paired_report as evaluate_pair
from tests.support.evaluation import reference_scores as _calibrate
from tests.support.evaluation import summaries
from tests.support.options import make_options, public_kwargs, replace_options

pytestmark = pytest.mark.unit


@pytest.fixture
def data():
    rng = np.random.default_rng(821)

    def frame(n, shift=0):
        x = rng.normal(shift, 1, n)
        return pd.DataFrame({"x": x, "y": x + rng.normal(size=n), "c": rng.choice(["a", "b"], n)})

    return EvalData(
        train=frame(90),
        test=frame(100),
        generated=frame(80, 0.2),
        schema=Schema(numeric=["x", "y"], categorical=["c"], target="y"),
    )


def test_only_evaluate_is_exported_as_evaluation_entrypoint():
    assert not hasattr(tabgen_eval, "calibrate")
    assert not hasattr(tabgen_eval, "evaluate_memorization")
    assert not hasattr(tabgen_eval, "CalibrationResult")
    assert not hasattr(tabgen_eval, "UQConfig")
    with pytest.raises(TypeError, match="EvalData"):
        evaluate({})


@pytest.mark.parametrize(
    "roles",
    [("test",), ("test", "generated"), ("train", "generated"), ("train", "test", "generated")],
)
def test_exhaustive_defaults_for_every_valid_role_combination(data, roles):
    pytest.importorskip("xgboost", minversion="2.1")
    data = replace(
        data, **{role: None for role in ("train", "test", "generated") if role not in roles}
    )
    report = evaluate(data, uncertainty=None)
    expected = [m["name"] for m in list_metrics() if set(m["required_roles"]) <= set(roles)]
    assert [metric.name for metric in report.results] == expected
    assert isinstance(report, Report)
    payload = json.loads(report.to_json(include_replicates=True))
    assert payload["schema_version"] == 2

    def check_metadata(value):
        if isinstance(value, dict):
            assert not {"backend", "metric_backends"}.intersection(value)
            for item in value.values():
                check_metadata(item)
        elif isinstance(value, list):
            for item in value:
                check_metadata(item)

    check_metadata(payload)
    for info in report.provenance["metric_execution"].values():
        assert "implementation" in info and "device" in info
    assert report.provenance["input_rows"] == {role: len(getattr(data, role)) for role in roles}
    assert "sanity" not in payload["results"]


@pytest.mark.parametrize("metrics", [(), ("mle",)])
def test_missing_target_raises_before_any_metric_runs(data, metrics, monkeypatch):
    paired = importlib.import_module("tabgen_eval._evaluation.paired")
    monkeypatch.setattr(
        paired.Estimator, "run", lambda *a: pytest.fail("computed before validation")
    )
    with pytest.raises(ValueError, match="target"):
        evaluate(
            replace(data, schema=replace(data.schema, target=None)),
            metrics=metrics,
            uncertainty=None,
        )


def test_missing_dependency_raises_before_any_metric_runs(data, monkeypatch):
    from tabgen_eval.models.xgboost import XGBoostModel

    paired = importlib.import_module("tabgen_eval._evaluation.paired")

    def missing():
        raise ModuleNotFoundError("XGBoost required")

    monkeypatch.setattr(XGBoostModel, "load_backend", missing)
    monkeypatch.setattr(
        paired.Estimator, "run", lambda *a: pytest.fail("computed before dependency check")
    )
    with pytest.raises(ModuleNotFoundError, match="XGBoost"):
        evaluate(
            data,
            metrics=["shape", "nonlinear_c2st"],
            uncertainty=None,
        )


def test_metrics_override_group_and_shared_jobs_run_once(data, monkeypatch):
    paired = importlib.import_module("tabgen_eval._evaluation.paired")
    calls = []
    original = paired.Estimator.run

    def counted(self, *args):
        calls.append(self.metric.name)
        return original(self, *args)

    monkeypatch.setattr(paired.Estimator, "run", counted)
    report = evaluate(
        data,
        group=["memorization"],
        metrics=["precision", "coverage", "cond_shape", "recovery"],
        uncertainty=None,
    )
    assert calls == ["prdc", "conditional_fidelity"]
    assert {result.name for result in report.results} == {
        "precision",
        "coverage",
        "cond_shape",
        "recovery",
    }
    assert set(report.results[-1].stats) == {"recovery_all", "recovery_rare"}
    assert report["precision"] is report["precision/precision"]


@pytest.mark.parametrize("mode", ["none", "reference", "candidate", "both"])
def test_paired_scores_draws_and_uq_match_preserved_procedure(data, mode):
    options = make_options(seed=182, uncertainty=mode, n_replicates=3)
    actual = evaluate(
        data,
        metrics=["shape", "precision", "recall", "density", "coverage"],
        **public_kwargs(options),
    )
    expected = evaluate_pair(
        data.test,
        data.generated,
        data.schema,
        metrics=["shape", "prdc"],
        options=replace_options(options, uncertainty=mode, n_replicates=3),
    )
    assert actual.provenance["procedures"]["comparison"]["n_res"] == min(
        len(data.generated), len(data.test) // 2
    )
    assert {"sample_budget", "generated_tail_allowance", "alpha"}.isdisjoint(actual.options_dict)
    for key in ("shape", "precision", "recall", "density", "coverage"):
        old = expected["shape/shape" if key == "shape" else f"prdc/{key}"]
        new = actual[key]
        assert not hasattr(new, "comparison")
        for attr in ("value", "ideal", "ci_low", "ci_high", "ideal_ci_low", "ideal_ci_high"):
            np.testing.assert_equal(getattr(new, attr), getattr(old, attr))
        for attr in ("generated_distribution", "reference_distribution"):
            a, b = getattr(new, attr), getattr(old, attr)
            if b is None:
                assert a is None
            else:
                np.testing.assert_array_equal(a.samples, b.samples)
                assert a.confidence_interval == b.confidence_interval == 0.95
    exported = json.loads(actual.to_json(include_replicates=True))["results"]["fidelity"]["shape"][
        "shape"
    ]
    assert {"comparison", "comparison_criteria"}.isdisjoint(exported)


def test_dcr_ratio_requires_test_and_keeps_supporting_results(data):
    options = make_options()
    with pytest.raises(ValueError, match="dcr_ratio requires test"):
        evaluate(replace(data, test=None), metrics=["dcr_ratio"], **public_kwargs(options))
    actual = evaluate(
        data,
        metrics=["dcr_ratio", "near_match_rate", "replication_ratio"],
        **public_kwargs(options),
    )
    expected = evaluate_memorization(
        data.train,
        data.generated,
        data.schema,
        holdout=data.test,
        metrics=["dcr", "near_match_rate", "replication_ratio"],
        options=options,
    )
    for stat in expected.results[0].stats:
        np.testing.assert_equal(actual[f"dcr_ratio/{stat}"].value, expected[f"dcr/{stat}"].value)
    assert actual["dcr_ratio"].value == actual["dcr_ratio/ratio"].value
    assert actual.provenance["procedures"]["memorization"]["uncertainty"] == "none"


@pytest.mark.parametrize("mode", ["none", "reference", "both"])
def test_test_only_preserves_reference_draws_and_truthful_results(data, mode):
    options = make_options(seed=219, uncertainty=mode, n_replicates=3)
    actual = evaluate(
        replace(data, train=None, generated=None), metrics=["shape"], **public_kwargs(options)
    )
    count = 1 if mode == "none" else 3
    expected = _calibrate(
        data.test,
        data.schema,
        metrics=["shape"],
        n_replicates=count,
        seed=219,
        options=options,
    )
    samples = summaries(expected.metrics["shape"])["shape"].samples
    score = actual["shape"]
    assert score.n_eff == len(data.test) // 2
    if mode == "none":
        np.testing.assert_equal(score.value, samples[0])
        assert score.reference_distribution is None
    else:
        assert score.value == score.reference_distribution.mean
        assert score.value_estimate == "mean"
        np.testing.assert_array_equal(score.reference_distribution.samples, samples)
    assert score.generated_distribution is None
    assert not hasattr(score, "comparison")
    assert score.n_generated == 0


def test_data_diagnostics_are_independent_of_evaluation(data, monkeypatch):
    diagnostics = importlib.import_module("tabgen_eval.diagnostics")
    monkeypatch.setattr(diagnostics, "diagnose", lambda *a, **k: pytest.fail("diagnostics called"))
    report = evaluate(data, metrics=["shape"], uncertainty=None)
    assert [result.name for result in report.results] == ["shape"]
    assert not hasattr(report, "data_diagnostics")
    assert all("CATEGORY_VOCAB_MISMATCH" not in stat.flag_codes for _, stat in report.iter_stats())


def test_public_filters_preserve_caller_frames(data):
    before = {role: getattr(data, role).copy(deep=True) for role in data.roles}
    evaluate(data, metrics=["shape", "dcr_ratio"], uncertainty=None)
    for role, frame in before.items():
        pd.testing.assert_frame_equal(frame, getattr(data, role))


def test_test_only_utility_bounds_worker_leases_without_changing_draws(data, monkeypatch):
    pytest.importorskip("xgboost", minversion="2.1")
    from tabgen_eval.execution.runtime import memory
    from tabgen_eval.uncertainty import reference as execution

    available = 512 * 1024**2
    worker_observations = []
    draws = []
    original_task = execution._mle_reference_task

    def worker_overhead():
        worker_observations.append(True)
        return 300 * 1024**2

    def recorded_task(metric, real, schema, reference, training, **kwargs):
        draws.append((reference.copy(), training.copy()))
        return original_task(metric, real, schema, reference, training, **kwargs)

    def no_processes(*args, **kwargs):
        pytest.fail("memory pressure must serialize these model fits")

    monkeypatch.setattr(memory, "host_available_bytes", lambda: available)
    monkeypatch.setattr(memory, "process_worker_bytes", worker_overhead)
    monkeypatch.setattr(execution, "_mle_reference_task", recorded_task)
    monkeypatch.setattr(execution, "Parallel", no_processes)
    data = replace(data, train=None, generated=None)
    options = make_options(
        device="cpu", uncertainty="reference", n_replicates=2, seed=197, n_jobs=4
    )
    constrained = evaluate(data, metrics=["mle"], **public_kwargs(options))
    serial = evaluate(data, metrics=["mle"], **public_kwargs(replace_options(options, n_jobs=1)))

    assert worker_observations
    actual = constrained["mle"].reference_distribution
    expected = serial["mle"].reference_distribution
    assert actual.finite_count == actual.requested_count == 2
    np.testing.assert_array_equal(actual.samples, expected.samples)
    rng = np.random.default_rng(options["seed"])
    size = len(data.test) // 2
    oracle = [rng.choice(len(data.test), 2 * size, replace=False) for _ in range(2)]
    for observed, positions in zip(draws, oracle * 2, strict=True):
        np.testing.assert_array_equal(observed[0], positions[:size])
        np.testing.assert_array_equal(observed[1], positions[size:])
        assert set(observed[0]).isdisjoint(observed[1])

    plan = constrained.provenance["metric_execution"]["mle"]["memory_plan"]
    assert plan["concurrent_tasks"] == 1
    assert plan["worker_overhead_bytes"] == 0
    assert plan["task_workspace_bytes"] == plan["aggregate_workspace_bytes"] == available * 4 // 5
    diagnostics = constrained.results[0].diagnostics["replicates"]
    assert len(diagnostics) == 2
    for diagnostic in diagnostics:
        fit = diagnostic["fits"]["baseline"]
        preparation = fit["preparation_memory_plan"]
        assert preparation["owned_task_bytes"] == 0
        assert preparation["task_workspace_bytes"] == plan["task_workspace_bytes"]
        for stage in ("memory_plan", "scoring_memory_plan"):
            child = fit[stage]
            assert child["owned_task_bytes"] > 0
            assert child["concurrent_tasks"] == 1
            assert child["task_workspace_bytes"] == (
                plan["task_workspace_bytes"] - child["owned_task_bytes"]
            )
        assert fit["engine_workspace_estimate_bytes"] <= fit["memory_plan"]["task_workspace_bytes"]
