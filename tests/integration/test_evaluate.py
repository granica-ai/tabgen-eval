"""End-to-end evaluate(): contract, explicit schema, metric selection, n_generated != n_holdout."""

from __future__ import annotations

import numpy as np
import pytest

from tabgen_eval import Schema, list_metrics
from tabgen_eval.report import Report
from tests.helpers import make_frame
from tests.support.evaluation import paired_report as evaluate
from tests.support.options import make_options

pytestmark = pytest.mark.unit


def test_n_jobs_minus_one_means_all_cores():
    """n_jobs=-1 follows the joblib/scikit-learn convention (all cores), not 'serial'."""
    import os

    from joblib import effective_n_jobs

    assert effective_n_jobs(-1) == (os.cpu_count() or 1)
    assert effective_n_jobs(-2) == max(1, (os.cpu_count() or 1) - 1)


def test_n_jobs_zero_rejected():
    with pytest.raises(ValueError, match="n_jobs"):
        make_options(n_jobs=0)


def test_automatic_resolution_is_recorded(schema):
    test, syn = make_frame(800, seed=1), make_frame(800, seed=2)
    rep = evaluate(test, syn, schema=schema, metrics=["shape"], options=make_options())
    assert "sample_budget" not in rep.options_dict
    assert "sample_budget" not in rep.provenance
    assert rep.provenance["n_res"] == rep.provenance["n_eff"] == 400


def test_removed_sampling_option_is_rejected(schema):
    test, syn = make_frame(200, seed=1), make_frame(200, seed=2)
    with pytest.raises(TypeError, match="sample_budget"):
        evaluate(
            test, syn, schema=schema, metrics=["shape"], options=make_options(sample_budget=201)
        )


def test_near_saturation_bands_remain_raw_and_in_range():
    import pandas as pd

    from tabgen_eval import Schema

    rng = np.random.default_rng(1)
    hold = pd.DataFrame({"x": rng.integers(0, 3, 200).astype(float)})
    gen = pd.DataFrame({"x": rng.integers(0, 3, 200).astype(float)})
    stat = evaluate(
        hold,
        gen,
        Schema(numeric=["x"]),
        metrics=["shape"],
        options=make_options(uncertainty="both", n_replicates=10, seed=1),
    )["shape/shape"]
    for distribution, endpoints in [
        (stat.generated_distribution, (stat.ci_low, stat.ci_high)),
        (stat.reference_distribution, (stat.ideal_ci_low, stat.ideal_ci_high)),
    ]:
        np.testing.assert_array_equal(endpoints, np.quantile(distribution.samples, [0.025, 0.975]))
        assert 0 <= endpoints[0] <= endpoints[1] <= 1


def test_continuous_bands_match_raw_score_quantiles():
    import pandas as pd

    from tabgen_eval import Schema

    rng = np.random.default_rng(2)
    hold, gen = [pd.DataFrame({"x": rng.normal(loc, 1, 500)}) for loc in (0, 0.6)]
    stat = evaluate(
        hold,
        gen,
        Schema(numeric=["x"]),
        metrics=["shape"],
        options=make_options(uncertainty="both", n_replicates=10),
    )["shape/shape"]
    for distribution in [stat.generated_distribution, stat.reference_distribution]:
        assert 0 <= distribution.band_low <= distribution.band_high <= 1
        assert distribution.requested_count == distribution.finite_count == 10


def test_evaluate_returns_full_contract(test_frame, syn_frame, schema):
    rep = evaluate(
        test_frame,
        syn_frame,
        schema=schema,
        options=make_options(uncertainty="both", n_replicates=10, seed=1),
    )
    assert isinstance(rep, Report)
    df = rep.to_dataframe()
    assert "reliable" not in df
    assert {
        "value",
        "ideal",
        "generated_retained",
        "reference_retained",
    } <= set(df)
    for _metric, stat in rep.iter_stats():
        for distribution in [stat.generated_distribution, stat.reference_distribution]:
            assert distribution.requested_count == 10
            assert distribution.finite_count + distribution.unavailable_count == 10
            finite = distribution.samples[np.isfinite(distribution.samples)]
            if len(finite):
                np.testing.assert_allclose(
                    [distribution.band_low, distribution.band_high],
                    np.quantile(finite, [0.025, 0.975]),
                    rtol=1e-14,
                    atol=1e-15,
                )
            else:
                assert np.isnan(distribution.band_low) and np.isnan(distribution.band_high)
        assert not hasattr(stat, "comparison")


def test_evaluate_includes_conditional_trend(test_frame, syn_frame, schema):
    # Conditional Trend flows end-to-end through the Estimator: the long-form report carries a
    # (conditional, cond_trend_all) row with a value and a bootstrap CI.
    rep = evaluate(
        test_frame,
        syn_frame,
        schema=schema,
        options=make_options(uncertainty="both", n_replicates=10, seed=1),
    )
    df = rep.to_dataframe().set_index(["metric", "stat"])
    assert ("conditional_fidelity", "cond_trend_all") in df.index
    row = df.loc[("conditional_fidelity", "cond_trend_all")]
    assert np.isfinite(row["value"])
    assert np.isfinite(row["ci_low"]) and np.isfinite(row["ci_high"])


def test_categorical_missing_cells_are_excluded_from_shape():
    import pandas as pd

    test = pd.DataFrame({"c": ["a"] * 200 + [None] * 200})
    syn = pd.DataFrame({"c": ["a"] * 400})
    result = evaluate(
        test,
        syn,
        schema=Schema(categorical=["c"]),
        metrics=["shape"],
        options=make_options(uncertainty="both", n_replicates=10),
    )["shape/shape"]
    assert result.value == result.ideal == 1.0
    np.testing.assert_array_equal(result.generated_distribution.samples, np.ones(10))
    np.testing.assert_array_equal(result.reference_distribution.samples, np.ones(10))
    assert test["c"].isna().sum() == 200


@pytest.mark.parametrize("n_test,n_syn", [(800, 500), (500, 800), (600, 600)])
def test_raw_bands_and_means_are_reported_for_any_size_ordering(schema, n_test, n_syn):
    test, syn = make_frame(n_test, seed=1), make_frame(n_syn, seed=2, shift=0.5)
    rep = evaluate(
        test,
        syn,
        schema=schema,
        metrics=["shape", "trend"],
        options=make_options(uncertainty="both", n_replicates=10, seed=1),
    )
    fixed = evaluate(
        test, syn, schema=schema, metrics=["shape", "trend"], options=make_options(seed=1)
    )
    for _metric, stat in rep.iter_stats():
        point = fixed[f"{_metric.name}/{stat.name}"]
        assert stat.value == stat.generated_distribution.mean
        assert stat.ideal == stat.reference_distribution.mean
        assert point.generated_distribution is None
        np.testing.assert_array_equal(
            [stat.ci_low, stat.ci_high],
            np.quantile(stat.generated_distribution.samples, [0.025, 0.975]),
        )


def test_fewer_generated_rows_define_matched_resolution(schema):
    rep = evaluate(
        make_frame(900, seed=1), make_frame(150, seed=2), schema=schema, metrics=["shape", "prdc"]
    )
    assert rep.provenance["n_eff"] == 150
    assert rep.provenance["n_holdout"] == 900 and rep.provenance["n_generated"] == 150
    assert np.isfinite(rep["shape/shape"].ideal) and np.isfinite(rep["prdc/coverage"].ideal)


def test_half_holdout_limits_resolution_when_generated_is_larger(schema):
    rep = evaluate(
        make_frame(150, seed=1), make_frame(900, seed=2), schema=schema, metrics=["shape"]
    )
    assert rep.provenance["n_eff"] == 75
    assert np.isfinite(rep["shape/shape"].ideal)


def test_generated_size_limits_automatic_resolution(schema):
    test, syn = make_frame(1000, seed=1), make_frame(800, seed=2)
    full = evaluate(test, syn, schema=schema, metrics=["shape"])
    small = evaluate(test, syn.iloc[:200], schema=schema, metrics=["shape"])
    assert full.provenance["n_eff"] == 500 and small.provenance["n_eff"] == 200
    assert full.provenance["n_holdout"] == small.provenance["n_holdout"] == 1000
    assert np.isfinite(full["shape/shape"].ideal) and np.isfinite(small["shape/shape"].ideal)


def test_evaluate_handles_different_sizes(schema):
    test = make_frame(600, seed=1)
    syn = make_frame(200, seed=2)
    rep = evaluate(
        test, syn, schema=schema, options=make_options(uncertainty="both", n_replicates=10)
    )
    prov = rep.provenance
    assert prov["n_holdout"] == 600 and prov["n_generated"] == 200
    # Both-side uncertainty halves a generated pool no larger than half the holdout.
    assert prov["n_eff"] == 100
    assert "selected rows per role=100" in rep.summary()


def test_evaluate_records_the_explicitly_inferred_schema(test_frame, syn_frame):
    rep = evaluate(
        test_frame,
        syn_frame,
        Schema.infer(test_frame),
        options=make_options(uncertainty="both", n_replicates=10),
    )
    assert rep.provenance["schema_inferred"] is False
    assert rep.schema_dict["numeric"]


def test_metric_selection_by_category(test_frame, syn_frame, schema):
    rep = evaluate(
        test_frame,
        syn_frame,
        schema=schema,
        metrics=["fidelity"],
        options=make_options(uncertainty="both", n_replicates=10),
    )
    cats = set(rep.to_dataframe()["category"])
    assert cats == {"fidelity"}


def test_metric_selection_by_name(test_frame, syn_frame, schema):
    rep = evaluate(
        test_frame,
        syn_frame,
        schema=schema,
        metrics=["prdc"],
        options=make_options(uncertainty="both", n_replicates=10),
    )
    assert set(rep.to_dataframe()["metric"]) == {"prdc"}


def test_unknown_metric_raises(test_frame, syn_frame, schema):
    with pytest.raises(ValueError):
        evaluate(test_frame, syn_frame, schema=schema, metrics=["nope"])


def test_list_metrics_introspection():
    names = {m["name"] for m in list_metrics()}
    assert {
        "shape",
        "trend",
        "cond_shape",
        "cond_trend",
        "recovery",
        "precision",
        "recall",
        "density",
        "coverage",
        "dcr_ratio",
    } <= names


def test_parallel_matches_serial(test_frame, syn_frame, schema):
    serial = evaluate(
        test_frame,
        syn_frame,
        schema=schema,
        options=make_options(uncertainty="both", n_replicates=10, seed=5, n_jobs=1),
    )
    parallel = evaluate(
        test_frame,
        syn_frame,
        schema=schema,
        options=make_options(uncertainty="both", n_replicates=10, seed=5, n_jobs=4),
    )
    s_df = serial.to_dataframe().set_index(["metric", "stat"])["value"]
    p_df = parallel.to_dataframe().set_index(["metric", "stat"])["value"]
    assert (s_df.fillna(-1) == p_df.fillna(-1)).all()
