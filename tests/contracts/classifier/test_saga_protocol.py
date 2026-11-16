"""Independent fold semantics for the production direct/SketchySAGA classifier."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from tabgen_eval import Schema
from tabgen_eval.execution.runtime.dispatch import context_for
from tabgen_eval.metrics import LinearC2ST
from tabgen_eval.preprocessing.frames import fold_design

pytestmark = pytest.mark.unit


@pytest.fixture(params=["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)])
def device(request):
    torch = pytest.importorskip("torch")
    if request.param.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    return request.param


def numeric_frames(n=180):
    generator = np.random.default_rng(490)
    reference = pd.DataFrame(generator.normal(size=(n, 3)), columns=["x", "y", "z"])
    candidate = pd.DataFrame(generator.normal(0.35, size=(n, 3)), columns=reference.columns)
    return reference, candidate, Schema(numeric=list(reference.columns))


def outcome(metric, reference, candidate, schema, rows=None, *, ideal=False):
    context = context_for(metric, reference, candidate, schema)
    assignments = np.arange(len(reference)) % 3
    rows = np.arange(len(reference)) if rows is None else rows
    return context.compute_classifier(
        metric,
        rows[None],
        rows[None],
        assignments[rows][None],
        assignments[rows][None],
        [1234567],
        ideal=ideal,
        point=True,
    )[0]


def test_standard_numeric_scaler_matches_independent_sklearn_training_fit():
    reference, candidate, schema = numeric_frames(n=12)
    reference.loc[0, "x"] = np.nan
    reference["constant"] = candidate["constant"] = 7.0
    schema.numeric.append("constant")
    reference["c"] = ["a", "b"] * 6
    candidate["c"] = ["a", "unseen"] * 6
    schema.categorical.append("c")
    heldout = reference.copy()
    heldout["x"] = 1e20
    heldout["c"] = "heldout-only"
    train, test, _, _ = fold_design(reference, candidate, heldout, heldout, schema, 2**30)
    # Imputation uses reference-training means; scaling uses both training classes.
    r = reference[schema.numeric].to_numpy().copy()
    s = candidate[schema.numeric].to_numpy().copy()
    means = np.nanmean(r, axis=0)
    r = np.where(np.isfinite(r), r, means)
    expected = StandardScaler().fit(np.concatenate([r, s]))
    np.testing.assert_allclose(train[:, :4], expected.transform(np.concatenate([r, s])), atol=1e-14)
    np.testing.assert_allclose(
        test[:, :4],
        np.tile(expected.transform(heldout[schema.numeric].to_numpy()), (2, 1)),
        atol=1e-14,
    )
    assert train.shape[1] == 8  # Four values, one missing flag, and three training categories.
    np.testing.assert_array_equal(train[:, 4], np.r_[1, np.zeros(23)])
    # Held-out-only categories have no learned indicator.
    np.testing.assert_array_equal(test[:, 5:], np.zeros((24, 3)))


@pytest.mark.parametrize("solver", ["auto", "direct", "sketchysaga"])
def test_default_ridge_counts_actual_fold_training_rows_and_resampled_copies(solver, device):
    pytest.importorskip("torch")
    reference, candidate, schema = numeric_frames()
    # Repeated source rows remain in their original folds; ridge counts copies.
    rows = np.repeat(np.arange(len(reference)), 1 + np.arange(len(reference)) % 4)
    result = outcome(
        LinearC2ST(device=device, solver=solver, saga_max_epochs=3, saga_rank=3),
        reference,
        candidate,
        schema,
        rows,
    )
    for fold in result.folds:
        expected = 2 * int(np.count_nonzero(rows % 3 != fold["fold"]))
        assert fold["training_rows"] == expected
        assert fold["effective_ridge"] == pytest.approx(1 / expected, rel=0, abs=1e-18)
        assert fold["tolerance"] == 1e-4
        assert "gradient_absolute_tolerance" not in fold
        assert "gradient_relative_tolerance" not in fold
        if fold["solver"] == "sketchysaga":
            assert fold["gradient_stopping_threshold"] == fold["tolerance"]


def test_one_tolerance_is_shared_by_both_solver_policies():
    assert LinearC2ST().tolerance == 1e-4
    for solver in ("direct", "sketchysaga"):
        metric = LinearC2ST(solver=solver, tolerance=3e-5)
        for point in (False, True):
            options = metric.native_options(point=point)
            assert options["tolerance"] == 3e-5
            assert {name for name in options if "tolerance" in name} == {"tolerance"}


@pytest.mark.parametrize("name", ["squared_tolerance", "saga_tolerance", "saga_relative_tolerance"])
def test_removed_tolerance_aliases_are_rejected(name):
    with pytest.raises(TypeError, match=name):
        LinearC2ST(**{name: 1e-4})


@pytest.mark.parametrize("ideal", [False, True])
def test_compiled_saga_preserves_rng_and_reports_budget_without_exact_gradient(ideal, device):
    torch = pytest.importorskip("torch")
    reference, candidate, schema = numeric_frames()
    before_torch = torch.random.get_rng_state().clone()
    before_numpy = np.random.get_state()
    before_threads = torch.get_num_threads()
    metric = LinearC2ST(
        device=device,
        solver="sketchysaga",
        saga_max_epochs=3,
        saga_batch_size=64,
        saga_rank=3,
        tolerance=1e-30,
    )
    first = outcome(metric, reference, candidate, schema, ideal=ideal)
    second = outcome(metric, reference, candidate, schema, ideal=ideal)
    assert first.value == second.value
    for a, b in zip(first.folds, second.folds, strict=True):
        assert a["solver"] == "sketchysaga"
        assert a["gradient_squared"] is None
        assert a["full_gradient_evaluations"] == 0
        assert np.isfinite(a["online_gradient_squared"])
        assert a["stopping_rule"] == "online_gradient_infinity"
        assert a["epochs_completed"] <= 3
        assert a["relative_coefficient_change"] == b["relative_coefficient_change"]
        if not a["converged"]:
            assert a["epochs_completed"] == 3
    torch.testing.assert_close(torch.random.get_rng_state(), before_torch, atol=0, rtol=0)
    after_numpy = np.random.get_state()
    assert before_numpy[0] == after_numpy[0]
    np.testing.assert_array_equal(before_numpy[1], after_numpy[1])
    assert before_numpy[2:] == after_numpy[2:]
    assert torch.get_num_threads() == before_threads


def test_saga_stopping_and_diagnostics_do_not_consult_heldout_rows(device):
    pytest.importorskip("torch")
    reference, candidate, schema = numeric_frames()
    metric = LinearC2ST(
        device=device,
        solver="sketchysaga",
        saga_max_epochs=8,
        saga_batch_size=64,
        saga_rank=3,
    )
    first = outcome(metric, reference, candidate, schema)
    heldout_reference, heldout_candidate = reference.copy(), candidate.copy()
    scoring = np.arange(len(reference)) % 3 == 0
    heldout_reference.loc[scoring] = 1e12
    heldout_candidate.loc[scoring] = -1e12
    second = outcome(metric, heldout_reference, heldout_candidate, schema)
    # Only fold zero has the same training set. Other folds intentionally differ.
    a, b = first.folds[0], second.folds[0]
    for field in [
        "epochs_completed",
        "iterations",
        "relative_coefficient_change",
        "online_gradient_squared",
        "online_gradient_infinity",
        "initial_online_gradient_infinity",
        "gradient_stopping_threshold",
        "accepted_epochs",
        "accepted_updates",
        "backoff_probe_checks",
        "backoff_retries",
        "rejected_epochs",
        "final_probe_objective",
        "initial_learning_rate",
        "final_learning_rate",
        "effective_ridge",
        "converged",
        "reason",
    ]:
        assert a[field] == b[field]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"solver": "newton-cg"},
        {"max_cg_iter": 20},
        {"sketch_sparsity": 8},
    ],
)
def test_removed_newton_cg_configuration_fails_explicitly(kwargs):
    with pytest.raises((ValueError, TypeError)):
        LinearC2ST(**kwargs)


def test_auto_routes_to_implicit_saga_when_direct_factor_memory_does_not_fit(device):
    reference = pd.DataFrame(
        {
            "x": np.arange(360, dtype=float),
            "a": np.arange(360).astype(str),
            "b": np.arange(360)[::-1].astype(str),
        }
    )
    schema = Schema(numeric=["x"], categorical=["a", "b"])
    candidate = reference.copy()
    options = dict(
        device=device,
        # Native CUDA preparation charges its queried ~6.6 MiB cuSOLVER scratch.
        # This remains below the dense direct-fit requirement; CPU needs less.
        memory_limit_mb=8 if str(device).startswith("cuda") else 2,
        direct_max_columns=1000,
        direct_max_work=1e15,
        saga_rank=8,
        saga_max_epochs=3,
    )
    automatic = outcome(LinearC2ST(**options), reference, candidate, schema)
    assert np.isfinite(automatic.value)
    assert all(fold["solver"] == "sketchysaga" for fold in automatic.folds)
    assert all(fold["features"] >= 400 for fold in automatic.folds)
    explicit = outcome(LinearC2ST(solver="direct", **options), reference, candidate, schema)
    assert not np.isfinite(explicit.value)
    assert not explicit.converged


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ridge": 0},
        {"ridge": np.inf},
        {"ridge": True},
        {"saga_max_epochs": 0},
        {"saga_batch_size": 1.5},
        {"saga_rank": True},
        {"tolerance": np.nan},
        {"tolerance": 0},
        {"tolerance": -1},
        {"tolerance": np.inf},
        {"tolerance": True},
    ],
)
def test_invalid_saga_objective_and_budget_configuration_fail_before_execution(kwargs):
    with pytest.raises(ValueError, match=next(iter(kwargs))):
        LinearC2ST(**kwargs)


@pytest.mark.gpu
def test_cuda_saga_never_calls_full_gradient_or_full_objective(monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from tabgen_eval.execution.torch.cuda.classifier import newton, solver

    def forbidden(*args, **kwargs):
        raise AssertionError("SAGA must not evaluate a full training gradient or objective")

    monkeypatch.setattr(solver, "derivatives", forbidden)
    monkeypatch.setattr(newton, "fit", forbidden)
    reference, candidate, schema = numeric_frames()
    metric = LinearC2ST(
        device="cuda:0",
        solver="sketchysaga",
        saga_rank=3,
        saga_batch_size=64,
        saga_max_epochs=3,
    )
    result = outcome(metric, reference, candidate, schema)
    assert np.isfinite(result.value)
    assert all(fold["full_gradient_evaluations"] == 0 for fold in result.folds)
    assert all(np.isfinite(fold["online_gradient_squared"]) for fold in result.folds)


@pytest.mark.gpu
@pytest.mark.parametrize("reason", ["memory_limit", "numerical_failure", "max_epochs"])
def test_cuda_terminal_saga_failures_are_not_scored_as_valid_models(monkeypatch, reason):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from sklearn.metrics import roc_auc_score

    from tabgen_eval.execution.torch.cuda.classifier import executor

    def mock_fit(fold, options, seed):
        # Finite placeholder weights would yield a valid AUC if scored. Terminal
        # failures must preserve the failure rather than silently accepting it.
        weights = fold.train.numbers.new_zeros(fold.train.d)
        weights[0] = 1
        return weights, dict(
            solver="sketchysaga",
            reason=reason,
            converged=False,
            gradient_squared=None,
            decrement_squared=None,
        )

    monkeypatch.setattr(executor, "fit", mock_fit)
    reference, candidate, schema = numeric_frames()
    metric = LinearC2ST(device="cuda:0", solver="sketchysaga", saga_rank=3)
    result = outcome(metric, reference, candidate, schema)
    assert not result.converged
    assert all(fold["reason"] == reason for fold in result.folds)
    if reason == "max_epochs":
        # Positive affine scaling of the first feature preserves its ranking;
        # this independent oracle needs no native transforms or fit code.
        assignments = np.arange(len(reference)) % 3
        expected = []
        for fold in range(3):
            heldout = assignments == fold
            margins = np.concatenate([reference.loc[heldout, "x"], candidate.loc[heldout, "x"]])
            labels = np.repeat([0, 1], heldout.sum())
            expected.append(max(0.5, roc_auc_score(labels, margins)))
        assert result.value == pytest.approx(np.mean(expected), abs=1e-15)
    else:
        assert np.isnan(result.value)
