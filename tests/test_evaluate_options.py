"""Public argument validation, ownership, and uncertainty reporting contracts."""

from __future__ import annotations

import importlib
import inspect
import json
from array import array
from collections import UserDict

import numpy as np
import pandas as pd
import pytest
from scipy.stats import ks_2samp

import tabgen_eval
from tabgen_eval import EvalData, Schema, UncertaintyOptions, evaluate
from tabgen_eval._evaluation.estimator import _metric_seed
from tabgen_eval._options import resolve_options
from tabgen_eval.execution.runtime.sampling import fixed_draw
from tests.support.evaluation import summaries

pytestmark = pytest.mark.unit


@pytest.fixture
def data():
    real = pd.DataFrame({"x": np.arange(100, dtype=float)})
    generated = pd.DataFrame({"x": np.linspace(0, 110, 90)})
    return EvalData(test=real, generated=generated, schema=Schema(numeric=["x"]))


def test_public_signature_has_direct_keyword_only_controls():
    parameters = inspect.signature(evaluate).parameters
    assert list(parameters) == [
        "data",
        "group",
        "metrics",
        "uncertainty",
        "seed",
        "device",
        "dtype",
        "n_jobs",
        "details",
    ]
    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in list(parameters.values())[1:])
    assert not hasattr(tabgen_eval, "EvalConfig")
    assert not hasattr(tabgen_eval, "FileReferenceCache")


def test_uncertainty_type_describes_an_ordinary_dictionary():
    assert UncertaintyOptions.__required_keys__ == {"mode"}
    assert UncertaintyOptions.__optional_keys__ == {"n_replicates"}
    assert UncertaintyOptions(mode="both", n_replicates=3) == {
        "mode": "both",
        "n_replicates": 3,
    }


@pytest.mark.parametrize("value", ["auto", "torch", "python", None])
def test_removed_backend_parameter_is_rejected(data, value):
    from tabgen_eval.metrics import Shape

    with pytest.raises(TypeError, match="backend"):
        evaluate(data, metrics=["shape"], backend=value)
    with pytest.raises(TypeError, match="backend"):
        resolve_options(backend=value)
    with pytest.raises(TypeError, match="backend"):
        Shape(backend=value)


@pytest.mark.parametrize(
    "uncertainty, error_type, problem",
    [
        ("both", TypeError, "dictionary or None"),
        (True, TypeError, "dictionary or None"),
        ([], TypeError, "dictionary or None"),
        (UserDict(mode="both"), TypeError, "dictionary or None"),
        ({}, ValueError, "missing required key 'mode'"),
        ({"n_replicates": 3}, ValueError, "missing required key 'mode'"),
        ({"mode": "none"}, ValueError, "mode must be"),
        ({"mode": "bootstrap"}, ValueError, "mode must be"),
        ({"mode": None}, ValueError, "mode must be"),
        ({"mode": ["both"]}, ValueError, "mode must be"),
        ({"mode": "both", "replicates": 3}, ValueError, "unknown key"),
        ({"mode": "both", 1: 3, "other": 4}, ValueError, "unknown key"),
        *[
            ({"mode": "both", "n_replicates": count}, ValueError, "n_replicates")
            for count in (0, -1, 1.5, 1.0, True, np.bool_(True), "3", None)
        ],
    ],
)
def test_invalid_uncertainty_explains_contract_before_computation(
    data, monkeypatch, uncertainty, error_type, problem
):
    module = importlib.import_module("tabgen_eval.evaluate")
    monkeypatch.setattr(
        module, "build_metrics", lambda *a: pytest.fail("metric construction before validation")
    )
    with pytest.raises(error_type) as error:
        evaluate(data, metrics=["shape"], uncertainty=uncertainty)
    message = str(error.value)
    assert problem in message
    assert "Required: mode" in message
    assert "Optional: n_replicates" in message
    assert "default 10" in message
    assert "uncertainty={'mode': 'both', 'n_replicates': 100}" in message
    assert "uncertainty=None" in message


@pytest.mark.parametrize("mode", ["reference", "candidate", "both"])
def test_mode_only_uses_ten_repetitions_and_reports_requested_means(data, mode):
    point = evaluate(data, metrics=["shape"])
    repeated = evaluate(data, metrics=["shape"], uncertainty={"mode": mode})
    explicit = evaluate(data, metrics=["shape"], uncertainty={"mode": mode, "n_replicates": 10})
    assert repeated.options_dict["uncertainty"] == {"mode": mode, "n_replicates": 10}
    score = repeated["shape"]
    assert point["shape"].n_eff == score.n_eff == 50
    if mode in {"reference", "both"}:
        samples = score.reference_distribution.samples
        assert len(samples) == 10 and np.isfinite(samples).all()
        assert score.ideal == pytest.approx(np.mean(samples))
        assert score.ideal_estimate == "mean"
        np.testing.assert_allclose(
            [score.ideal_ci_low, score.ideal_ci_high], np.quantile(samples, [0.025, 0.975])
        )
        np.testing.assert_array_equal(samples, explicit["shape"].reference_distribution.samples)
    else:
        assert score.reference_distribution is None
        assert score.ideal == point["shape"].ideal and score.ideal_estimate == "point"
        assert np.isnan(score.ideal_ci_low) and np.isnan(score.ideal_ci_high)
    if mode in {"candidate", "both"}:
        samples = score.generated_distribution.samples
        assert len(samples) == 10 and np.isfinite(samples).all()
        assert score.value == pytest.approx(np.mean(samples))
        assert score.value_estimate == "mean"
        np.testing.assert_allclose(
            [score.ci_low, score.ci_high], np.quantile(samples, [0.025, 0.975])
        )
        np.testing.assert_array_equal(samples, explicit["shape"].generated_distribution.samples)
    else:
        assert score.generated_distribution is None
        assert score.value == point["shape"].value and score.value_estimate == "point"
        assert np.isnan(score.ci_low) and np.isnan(score.ci_high)


def test_default_disables_distributions(data):
    report = evaluate(data, metrics=["shape"])
    assert report.options_dict["uncertainty"] is None
    assert report["shape"].generated_distribution is None
    assert report["shape"].reference_distribution is None


@pytest.mark.parametrize("generated_size,n_res", [(40, 20), (90, 50)])
def test_candidate_interval_matches_both_without_reference_distribution(
    data, generated_size, n_res
):
    data = EvalData(
        test=data.test, generated=data.generated.head(generated_size), schema=data.schema
    )
    reports = {
        mode: evaluate(
            data,
            metrics=["shape"],
            device="cpu",
            uncertainty={"mode": mode, "n_replicates": 4},
        )
        for mode in ("candidate", "both")
    }
    candidate, both = reports["candidate"]["shape"], reports["both"]["shape"]
    assert candidate.n_eff == both.n_eff == n_res
    assert candidate.value == both.value
    assert candidate.value_estimate == both.value_estimate == "mean"
    np.testing.assert_array_equal(
        candidate.generated_distribution.samples, both.generated_distribution.samples
    )
    assert candidate.value == pytest.approx(np.mean(candidate.generated_distribution.samples))
    assert (candidate.ci_low, candidate.ci_high) == (both.ci_low, both.ci_high)
    draw = fixed_draw(len(data.test), len(data.generated), seed=42, uncertainty="candidate")
    expected_reference = (
        1
        - ks_2samp(
            data.test.iloc[draw.reference].x, data.test.iloc[draw.real_candidate].x
        ).statistic
    )
    assert candidate.ideal == pytest.approx(expected_reference)
    assert candidate.ideal_estimate == "point"
    assert both.ideal_estimate == "mean"
    assert len(both.reference_distribution.samples) == 4
    assert both.ideal == pytest.approx(np.mean(both.reference_distribution.samples))
    assert candidate.reference_distribution is None
    assert np.isnan(candidate.ideal_ci_low) and np.isnan(candidate.ideal_ci_high)
    payload = json.loads(reports["candidate"].to_json(include_replicates=True))
    assert payload["options"]["uncertainty"] == {"mode": "candidate", "n_replicates": 4}
    assert payload["results"]["fidelity"]["shape"]["shape"]["reference_distribution"] is None


def test_candidate_uncertainty_requires_generated_data(data):
    with pytest.raises(ValueError, match="Candidate uncertainty requires EvalData.generated"):
        evaluate(
            EvalData(test=data.test, schema=data.schema),
            metrics=["shape"],
            uncertainty={"mode": "candidate"},
        )


def test_options_are_owned_reusable_and_serializable(data):
    uncertainty = {"mode": "both", "n_replicates": np.int64(3)}
    seed = np.array([42, 7], dtype=np.uint64)
    kwargs = dict(
        uncertainty=uncertainty,
        seed=seed,
        device="cpu",
        n_jobs=np.int64(1),
        details=True,
    )
    first = evaluate(data, metrics=["shape"], **kwargs)
    second = evaluate(data, metrics=["shape"], **kwargs)
    np.testing.assert_array_equal(
        first["shape"].generated_distribution.samples,
        second["shape"].generated_distribution.samples,
    )
    assert type(uncertainty["n_replicates"]) is np.int64
    assert first.options_dict == {
        "uncertainty": {"mode": "both", "n_replicates": 3},
        "seed": [42, 7],
        "device": "cpu",
        "dtype": "float64",
        "n_jobs": 1,
        "details": True,
    }
    assert type(first.options_dict["uncertainty"]["n_replicates"]) is int
    uncertainty["mode"] = "reference"
    seed[0] = 99
    assert first.options_dict["uncertainty"]["mode"] == "both"
    assert first.options_dict["seed"] == [42, 7]
    payload = json.loads(first.to_json(include_replicates=True))
    assert payload["options"] == first.options_dict
    assert "config" not in payload
    assert "n_replicates" not in payload["options"]
    np.testing.assert_array_equal(
        payload["results"]["fidelity"]["shape"]["shape"]["generated_distribution"]["samples"],
        first["shape"].generated_distribution.samples,
    )


@pytest.mark.parametrize("value", [1.5, 1.0, True, np.bool_(True), "2", None, 0])
def test_invalid_worker_count_is_rejected(data, value):
    with pytest.raises(ValueError, match="n_jobs"):
        evaluate(data, metrics=["shape"], n_jobs=value)


@pytest.mark.parametrize("workers", [1, 2, -1, -2, -100])
def test_negative_worker_convention_is_preserved(workers):
    assert resolve_options(n_jobs=workers)["n_jobs"] == workers


@pytest.mark.parametrize(
    "kwargs, field",
    [
        ({"dtype": False}, "dtype"),
        ({"dtype": "float16"}, "dtype"),
        ({"device": "cuda:-1"}, "device"),
        ({"device": None}, "device"),
        ({"details": "yes"}, "details"),
    ],
)
def test_invalid_execution_arguments_identify_field(data, kwargs, field):
    with pytest.raises(ValueError, match=field):
        evaluate(data, metrics=["shape"], **kwargs)


@pytest.mark.parametrize(
    "seed",
    [
        0,
        2**256,
        np.uint64(2**63),
        [1, 2],
        (1, 2),
        np.array([1, 2]),
        range(3),
        True,
    ],
)
def test_seed_validation_preserves_existing_metric_rng_stream(seed):
    expected = _metric_seed(seed, "shape").generate_state(10)
    options = resolve_options(seed=seed)
    np.testing.assert_array_equal(
        _metric_seed(options["seed"], "shape").generate_state(10), expected
    )
    json.dumps(options["seed"])


def test_general_sequence_seed_exports_as_owned_list(data):
    seed = range(3)
    report = evaluate(data, metrics=["shape"], seed=seed)
    expected = list(seed)
    assert report.options_dict["seed"] == expected
    assert json.loads(report.to_json())["options"]["seed"] == expected


@pytest.mark.parametrize(
    "seed",
    [
        -1,
        None,
        "42",
        1.0,
        [1, -1],
        [1, 2.5],
        np.array([1.0, 2.0]),
        np.bool_(True),
        array("I", [1, 2]),
    ],
)
def test_invalid_entropy_fails_before_evaluation(data, seed):
    with pytest.raises(ValueError, match="seed.*NumPy SeedSequence"):
        evaluate(data, metrics=["shape"], seed=seed)


@pytest.mark.parametrize("budget", [101, 200])
def test_removed_sampling_option_is_rejected(data, budget):
    with pytest.raises(TypeError, match="sample_budget"):
        evaluate(data, metrics=["shape"], sample_budget=budget)


@pytest.mark.parametrize(
    "field, value",
    [
        ("config", {}),
        ("n_replicates", 3),
        ("metric_kwargs", {}),
        ("gpu_workspace_mb", 32),
        ("workspace_mb", 32),
        ("memory_limit_mb", 32),
        ("generated_tail_allowance", 0.1),
        ("n_bootstrap", 3),
        ("alpha", 0.05),
    ],
)
def test_removed_options_are_rejected(data, field, value):
    with pytest.raises(TypeError, match=field):
        evaluate(data, metrics=["shape"], **{field: value})


def test_small_finite_samples_produce_bands_without_a_reliability_gate():
    frame = pd.DataFrame({"x": np.arange(30, dtype=float)})
    data = EvalData(test=frame, generated=frame, schema=Schema.infer(frame))
    report = evaluate(
        data,
        metrics=["shape"],
        uncertainty={"mode": "both", "n_replicates": 3},
    )
    score = report["shape"]
    assert np.isfinite(score.ci_low)
    assert np.isfinite(score.ideal_ci_low)
    assert not hasattr(score, "reliable")


def test_data_diagnostics_do_not_require_real_halves(monkeypatch):
    from tabgen_eval import diagnose
    from tabgen_eval.execution.runtime import sampling

    monkeypatch.setattr(sampling, "fixed_draw", lambda *a, **k: pytest.fail("unexpected sampling"))
    frame = pd.DataFrame({"x": [1.0]})
    report = diagnose(frame, frame, schema=Schema(numeric=["x"]))
    assert report.inputs["train"]["n_rows"] == 1
    assert report.inputs["generated"]["n_rows"] == 1


def test_internal_reference_distribution_needs_no_generated_pool():
    from tests.support.evaluation import reference_scores as _calibrate

    frame = pd.DataFrame({"x": np.arange(100, dtype=float)})
    result = _calibrate(
        frame,
        Schema(numeric=["x"]),
        metrics=["shape"],
        n_replicates=3,
        options=resolve_options(),
    )
    assert result.metadata["n_res"] == 50
    assert len(summaries(result.metrics["shape"])["shape"].samples) == 3
