"""Held-out classifier protocol, solver accuracy, and bootstrap reliability."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from tabgen_eval import Schema, list_metrics
from tabgen_eval.execution.classifier.protocol import source_folds
from tabgen_eval.execution.runtime.dispatch import context_for
from tabgen_eval.execution.runtime.sampling import fixed_draw, paired_draw_batches
from tabgen_eval.metrics import ClassifierMetric, LinearC2ST
from tabgen_eval.preprocessing.frames import fold_design
from tests.support.evaluation import paired_report as evaluate
from tests.support.linear_c2st import compute_partitioned as linear_reference
from tests.support.metrics import configured_metric
from tests.support.options import make_options

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def tensor_backend_dependency(request):
    if (
        getattr(request.node, "callspec", None)
        and request.node.callspec.params.get("backend") == "torch"
    ):
        pytest.importorskip("torch")


def cpu_context(*frames):
    pytest.importorskip("torch")
    return context_for(LinearC2ST(device="cpu"), *frames)


def mixed_data(seed=4, n=360):
    rng = np.random.default_rng(seed)
    schema = Schema(numeric=["x", "y", "constant"], categorical=["c"])
    frames = []
    for shift in [0, 0.8]:
        x = rng.normal(shift, 1, n)
        df = pd.DataFrame({"x": x, "y": x + rng.normal(0, 0.3, n), "constant": 1.0})
        df["c"] = rng.choice(["a", "b", "nan"], n)
        df.loc[np.arange(n) % 19 == 0, "c"] = None
        df.loc[np.arange(n) % 17 == 0, "x"] = np.nan
        frames.append(df)
    return *frames, schema


def test_fold_preprocessing_fits_both_training_classes_but_never_scoring_rows():
    r = pd.DataFrame({"x": [0.0, 2.0, np.nan], "c": ["a", "b", "a"]})
    s = pd.DataFrame({"x": [2.0, 4.0, np.nan], "c": ["b", "unknown", "b"]})
    rt = pd.DataFrame({"x": [np.nan, 1e9], "c": ["unknown", "test-only"]})
    st = rt.copy()
    schema = Schema(numeric=["x"], categorical=["c"])
    train, test, _, _ = fold_design(r, s, rt, st, schema, 2**30)
    train2, _, _, _ = fold_design(r, s, rt.assign(x=-1e9), st, schema, 2**30)
    np.testing.assert_equal(train, train2)
    assert train.shape[1] == 5  # numeric, missingness, and three training category levels
    assert test[0, 0] == train[2, 0]  # impute scoring NaN from real training mean
    np.testing.assert_array_equal(train[:, 1], [0, 0, 1, 0, 0, 1])
    np.testing.assert_array_equal(
        train[:, 2:], [[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]]
    )
    # Generated-only training category is learned; held-out-only category is not.
    np.testing.assert_array_equal(test[:, 2:], [[0, 0, 1], [0, 0, 0]] * 2)


def test_resampled_classifier_roles_have_unique_disjoint_source_rows():
    draw = fixed_draw(401, 201, seed=5)
    for batch in paired_draw_batches(
        draw, n_replicates=5, seed=43, batch_size=2, include_generated=True
    ):
        for reference, candidate in zip(batch.reference, batch.real_candidate, strict=True):
            assert len(np.unique(reference)) == len(reference)
            assert len(np.unique(candidate)) == len(candidate)
            assert not np.intersect1d(reference, candidate).size
            rf, sf = source_folds(len(reference), len(candidate), 3, np.random.default_rng(5))
            assert set(rf) == set(sf) == {0, 1, 2}


def test_native_classifier_matches_independent_sklearn_objective():
    solver = "direct"
    real, fake, schema = mixed_data()
    rf, sf = source_folds(len(real), len(fake), 3, np.random.default_rng(13))
    # Tight tolerance makes solver differences smaller than one held-out AUC pair.
    metric = LinearC2ST(solver=solver, tolerance=1e-07)
    native = cpu_context(real, fake, schema)
    outcome = native.compute_classifier(
        metric,
        np.arange(len(real))[None],
        np.arange(len(fake))[None],
        rf[None],
        sf[None],
        [8],
        point=True,
    )[0]
    oracle = linear_reference(metric, real, fake, schema, rf, sf, seed=8, point=True)
    assert outcome.converged and oracle.converged
    assert outcome.value == pytest.approx(oracle.value, abs=2e-4)
    assert {f["solver"] for f in outcome.folds} == {solver}


def test_classifier_retains_scale_shift_and_constant_data_ties():
    metric = LinearC2ST()
    real = pd.DataFrame({"x": np.linspace(1.0, 2.0, 150)})
    fake = pd.DataFrame({"x": np.linspace(3.0, 4.0, 150)})
    schema = Schema(numeric=["x"])
    assert metric.compute(real, fake, schema, rng=np.random.default_rng(0))["auc"] == 1.0
    constant = real.assign(x=1.0)
    assert metric.compute(constant, constant, schema, rng=np.random.default_rng(0))["auc"] == 0.5


def test_scoring_clips_each_fold_before_averaging():
    class MixedDirectionClassifier(ClassifierMetric):
        name = "mixed_direction_test"
        stats = LinearC2ST.stats

        def _fit_predict(self, train, labels, test, *, seed, point, fit_threads=1):
            return test[:, 0] * (1 if seed % 3 == 0 else -1), {"converged": True}

    r = pd.DataFrame({"x": np.linspace(-2, -1, 30)})
    s = pd.DataFrame({"x": np.linspace(1, 2, 30)})
    folds = np.arange(30) % 3
    score = MixedDirectionClassifier().compute_partitioned(
        r, s, Schema(numeric=["x"]), folds, folds, seed=0
    )
    assert roc_auc_score(np.repeat([0, 1], 30), np.r_[-r.x, -s.x]) == 0.0
    assert score.value == pytest.approx(2 / 3)  # clip BEFORE averaging, no final inversion


def test_single_classifier_hook_receives_the_fit_budget():
    class BudgetClassifier(ClassifierMetric):
        name = "budget_test"
        stats = LinearC2ST.stats

        def _fit_predict(self, train, labels, test, *, seed, point, fit_threads=1):
            return test[:, 0], {"converged": True, "fit_threads": fit_threads}

    reference = pd.DataFrame({"x": np.linspace(-2, -1, 30)})
    candidate = pd.DataFrame({"x": np.linspace(1, 2, 30)})
    folds = np.arange(30) % 3
    outcome = BudgetClassifier().compute_partitioned(
        reference, candidate, Schema(numeric=["x"]), folds, folds, seed=0, fit_threads=3
    )
    assert outcome.value == 1.0
    assert all(fold["fit_threads"] == 3 for fold in outcome.folds)


def test_fitted_report_reports_repeated_means_without_extra_fits():
    real, fake, schema = mixed_data(n=300)
    kwargs = dict(seed=15)
    point = evaluate(
        real,
        fake,
        schema,
        metrics=["linear_c2st"],
        options=make_options(uncertainty="none", **kwargs),
    )
    report = evaluate(
        real,
        fake,
        schema,
        metrics=["linear_c2st"],
        options=make_options(uncertainty="both", n_replicates=4, **kwargs),
    )
    stat = report["linear_c2st/auc"]
    assert stat.value == stat.generated_distribution.mean
    assert stat.ideal == stat.reference_distribution.mean
    assert point["linear_c2st/auc"].generated_distribution is None
    assert stat.n_eff == 150
    assert stat.ci_low <= stat.ci_high
    diag = report.provenance["classifier_diagnostics"]["linear_c2st"]
    assert not diag["point"]
    for comparison in ("generated", "real_reference"):
        assert len(diag["replicates"][comparison]) == 4
        for fit in diag["replicates"][comparison]:
            assert fit["point"] and fit["converged"]
            for fold in fit["folds"]:
                assert fold["training_rows"] + fold["scoring_rows"] == 2 * stat.n_eff
    assert json.loads(report.to_json())["provenance"]["classifier_diagnostics"] == {
        "linear_c2st": diag
    }


def test_native_fitted_parallelism_preserves_replicates_and_solver_policy():
    pytest.importorskip("torch")
    real, fake, schema = mixed_data(n=300)
    results = []
    for jobs in [1, 4]:
        options = make_options(device="cpu", n_jobs=jobs, uncertainty="both", n_replicates=5)
        report = evaluate(
            real,
            fake,
            schema,
            metrics=[
                configured_metric(
                    LinearC2ST,
                    options,
                    direct_max_columns=1,
                    point_direct_max_columns=1,
                    tolerance=0.01,
                )
            ],
            options=options,
        )
        results.append(report["linear_c2st/auc"])
        diag = report.provenance["classifier_diagnostics"]["linear_c2st"]
        assert not diag["point"]
        assert {f["solver"] for fit in diag["replicates"]["generated"] for f in fit["folds"]} == {
            "sketchysaga"
        }
        assert len(diag["replicates"]["generated"]) == 5
    np.testing.assert_equal(
        results[0].generated_distribution.samples, results[1].generated_distribution.samples
    )
    np.testing.assert_equal(
        results[0].reference_distribution.samples, results[1].reference_distribution.samples
    )


def test_finite_nonconverged_fits_remain_in_distributions():
    real, fake, schema = mixed_data()
    options = make_options(uncertainty="both", n_replicates=3)
    report = evaluate(
        real,
        fake,
        schema,
        metrics=[
            configured_metric(
                LinearC2ST,
                options,
                max_iter=1,
                tolerance=1e-08,
            )
        ],
        options=options,
    )
    stat = report["linear_c2st/auc"]
    assert "CLASSIFIER_NOT_CONVERGED" in stat.flag_codes
    assert np.isfinite(stat.generated_distribution.samples).all()
    assert np.isfinite(stat.ci_low)
    fits = report.provenance["classifier_diagnostics"]["linear_c2st"]["replicates"]["generated"]
    assert len(fits) == 3 and all(not fit["converged"] for fit in fits)


def test_native_boundary_rejects_a_source_row_in_two_folds():
    real, fake, schema = mixed_data()
    native = cpu_context(real, fake, schema)
    with pytest.raises(ValueError, match="copies|across classes"):
        native.compute_classifier(
            LinearC2ST(),
            np.array([[0, 0, 1]]),
            np.array([[0, 1, 2]]),
            np.array([[0, 1, 2]]),
            np.array([[0, 1, 2]]),
            [0],
        )


def test_linear_c2st_is_registered_as_explicit_selection():
    entry = next(m for m in list_metrics() if m["name"] == "linear_c2st")
    assert entry["category"] == "detection" and entry["enabled_by_default"]


def test_high_cardinality_design_is_flagged_before_exceeding_memory_guard():
    real = pd.DataFrame({"c": [f"category-{i}" for i in range(6000)]})
    options = make_options(uncertainty="none")
    report = evaluate(
        real,
        real.copy(),
        Schema(categorical=["c"]),
        metrics=[
            configured_metric(
                LinearC2ST,
                options,
                memory_limit_mb=1,
            )
        ],
        options=options,
    )
    stat = report["linear_c2st/auc"]
    assert np.isnan(stat.value) and "CLASSIFIER_NOT_CONVERGED" in stat.flag_codes
    assert all(
        f["reason"]
        in {"preprocessing_memory_limit", "preprocessing_or_memory_limit", "saga_memory_limit"}
        for f in report.provenance["classifier_diagnostics"]["linear_c2st"]["point"]["generated"][
            "folds"
        ]
    )


def test_entirely_missing_numeric_design_returns_flagged_report():
    real = pd.DataFrame({"x": [np.nan] * 240})
    report = evaluate(
        real,
        real.copy(),
        Schema(numeric=["x"]),
        metrics=["linear_c2st"],
        options=make_options(uncertainty="none"),
    )
    assert "NAN_VALUE" in report["linear_c2st/auc"].flag_codes


def test_classifier_with_fewer_rows_than_folds_is_insufficient():
    real = pd.DataFrame({"x": np.arange(8)})
    options = make_options(uncertainty="both", n_replicates=2)
    report = evaluate(
        real,
        real.copy(),
        Schema(numeric=["x"]),
        metrics=[
            configured_metric(
                LinearC2ST,
                options,
                n_min=1,
                cv_folds=16,
            )
        ],
        options=options,
    )
    assert "INSUFFICIENT_DATA" in report["linear_c2st/auc"].flag_codes


def test_native_shared_pool_checks_fold_identity_across_classes():
    real, fake, schema = mixed_data()
    native = cpu_context(real, fake, schema)
    with pytest.raises(ValueError, match="copies|across classes"):
        native.compute_classifier(
            LinearC2ST(),
            np.array([[0, 1, 2]]),
            np.array([[0, 1, 2]]),
            np.array([[0, 1, 2]]),
            np.array([[1, 1, 2]]),
            [0],
            ideal=True,
        )
