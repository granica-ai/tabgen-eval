"""Independent row budgets and training-only point evaluation contracts."""

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import cdist

from tabgen_eval import Schema
from tabgen_eval.execution.runtime.memorization_sampling import memorization_draw
from tabgen_eval.execution.runtime.outcomes import MemorizationOutcome
from tabgen_eval.metrics import DCR
from tests.support.evaluation import memorization_report as evaluate_memorization
from tests.support.evaluation import paired_report as evaluate
from tests.support.metrics import configured_metric
from tests.support.options import make_options, replace_options
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


@pytest.mark.parametrize("sizes", [(60003, 51007), (10, 60003), (60003, 10), (0, 3)])
def test_explicit_cap_is_independent_uniform_and_reproducible(sizes):
    actual = memorization_draw(*sizes, max_samples=50000, seed=137)
    streams = np.random.SeedSequence(137).spawn(2)
    for positions, size, stream in zip(
        [actual.training, actual.generated], sizes, streams, strict=True
    ):
        expected = (
            np.arange(size)
            if size <= 50000
            else np.random.default_rng(stream).choice(size, 50000, replace=False)
        )
        np.testing.assert_array_equal(positions, expected)
        assert len(positions) == min(size, 50000)
        assert len(np.unique(positions)) == len(positions)
        assert not positions.flags.writeable
    same_training = memorization_draw(sizes[0], sizes[1] + 60000, max_samples=50000, seed=137)
    np.testing.assert_array_equal(actual.training, same_training.training)
    same_generated = memorization_draw(sizes[0] + 60000, sizes[1], max_samples=50000, seed=137)
    np.testing.assert_array_equal(actual.generated, same_generated.generated)


@pytest.mark.parametrize("limit", [True, np.bool_(False), 0, -1, 2.5, "50000", np.inf])
def test_invalid_metric_row_caps_rejected(limit):
    with pytest.raises(ValueError, match="max_samples"):
        DCR(max_samples=limit)


@pytest.mark.parametrize("limit", [None, 1, np.int64(7)])
def test_explicit_cap_and_uncapped_selection(limit):
    metric = DCR(max_samples=limit)
    draw = memorization_draw(11, 3, max_samples=metric.max_samples)
    assert len(draw.training) == (11 if limit is None else min(11, limit))
    assert len(draw.generated) == (3 if limit is None else min(3, limit))
    if limit is None:
        np.testing.assert_array_equal(draw.training, np.arange(11))
        np.testing.assert_array_equal(draw.generated, np.arange(3))


@pytest.mark.parametrize("limit", [None, 50000])
def test_selection_precedes_encoding_and_context_creation(limit, monkeypatch):
    import tabgen_eval.execution.runtime.memorization as module

    seen = []
    sizes = (60003, 51007)
    selected = tuple(size if limit is None else min(size, limit) for size in sizes)

    class Session:
        def __init__(self, training, generated, schema, **kwargs):
            assert (len(training), len(generated)) == selected
            assert kwargs["n_bootstrap"] == 0
            seen.append("session")

        def context_for(self, metric):
            return None

    def prepare(training, generated, *args, **kwargs):
        assert (len(training), len(generated)) == selected
        seen.append("prepare")
        return None

    def numerical(training, generated, schema, metrics):
        assert (len(training), len(generated)) == selected
        draw = memorization_draw(*sizes, max_samples=limit, seed=137)
        np.testing.assert_array_equal(training["x"], draw.training)
        np.testing.assert_array_equal(generated["x"], draw.generated)
        seen.append("numerical")
        return {
            metric.name: MemorizationOutcome(
                {name: 0.0 for name in metric.stat_names},
                {"usable_rows": {"training": len(training), "generated": len(generated)}},
            )
            for metric in metrics
        }

    monkeypatch.setattr(module, "ExecutionSession", Session)
    monkeypatch.setattr(module, "try_prepare", prepare)
    monkeypatch.setattr(module, "compute_memorization_group", numerical)
    training = pd.DataFrame({"x": np.arange(60003)})
    generated = pd.DataFrame({"x": np.arange(51007)})
    result = evaluate_memorization(
        training,
        generated,
        Schema(numeric=["x"]),
        options=make_options(device="cpu", seed=137),
        metrics=[DCR(device="cpu", max_samples=limit)],
    )
    assert seen == ["prepare", "session", "numerical"]
    assert result.provenance["n_training"] == 60003
    assert result.provenance["n_generated"] == 51007
    assert result.provenance["max_samples"] == limit
    for role, count in zip(("training", "generated"), selected, strict=True):
        assert result.provenance[f"n_selected_{role}"] == count
        assert result.provenance[f"n_retained_{role}"] == count


def test_point_scores_match_independently_sampled_training_geometry():
    pytest.importorskip("torch")
    training = pd.DataFrame({"x": [0.0, 1.0, 3.0, 8.0, 11.0, 19.0, 35.0, 60.0, 100.0]})
    generated = pd.DataFrame({"x": [0.25, 1.25, 4.5, 8.5, 90.0, 110.0]})
    holdout = pd.DataFrame({"x": [2.0, 6.0, 15.0, 80.0]})
    originals = (training.copy(deep=True), generated.copy(deep=True))
    options = make_options(device="cpu", seed=29)
    metrics = [configured_metric(DCR, options, max_samples=7)]
    result = evaluate_memorization(
        training,
        generated,
        Schema(numeric=["x"]),
        holdout=holdout,
        options=options,
        metrics=metrics,
    )
    seeds = np.random.SeedSequence(29).spawn(2)
    chosen = np.random.default_rng(seeds[0]).choice(len(training), 7, replace=False)
    t = training.iloc[chosen].to_numpy()
    (minimum, scale) = (t.min(axis=0), np.ptp(t, axis=0))
    t = (t - minimum) / scale
    g = (generated.to_numpy() - minimum) / scale
    cross = cdist(g, t)
    held_distances = cdist((holdout.to_numpy() - minimum) / scale, t)
    distances = cross.min(axis=1)
    baseline = np.median(held_distances.min(axis=1))
    expected = {
        "median": np.median(distances),
        "q05": np.quantile(distances, 0.05),
        "ratio": np.median(distances) / baseline,
    }
    for name, value in expected.items():
        assert _values(result)[name] == pytest.approx(value, abs=1e-12)
    assert result.results[0].diagnostics["holdout_median"] == pytest.approx(baseline)
    assert result.provenance["n_selected_training"] == 7
    assert result.provenance["n_selected_generated"] == 6
    assert result.provenance["baseline"] == "holdout_to_selected_training"
    assert result.provenance["n_selected_holdout"] == 4
    assert result.provenance["n_retained_training"] == 7
    assert not hasattr(result, "real_reference")
    assert not hasattr(result, "value_replicates")
    for before, after in zip(originals, [training, generated], strict=True):
        pd.testing.assert_frame_equal(before, after)
    alternate = replace_options(
        options,
        uncertainty="both",
        n_replicates=999,
    )
    again = evaluate_memorization(
        training,
        generated,
        Schema(numeric=["x"]),
        holdout=holdout,
        options=alternate,
        metrics=metrics,
    )
    assert _values(result) == _values(again)

    assert without_memory_plans(result.provenance) == without_memory_plans(again.provenance)


def test_direct_metric_uses_uncapped_training_point_path():
    training = pd.DataFrame({"x": [0.0, 2.0, 4.0, 6.0]})
    generated = pd.DataFrame({"x": [1.0, 3.0, 5.0]})
    values = DCR().compute(
        training, generated, Schema(numeric=["x"]), rng=None, holdout=pd.DataFrame({"x": [8.0]})
    )
    assert values["ratio"] == pytest.approx(0.5)
    assert DCR().max_samples is None
    assert DCR().search_method == "exact"
    assert not DCR().bootstrapped


@pytest.mark.parametrize("metric", [DCR])
def test_default_row_budget_retains_unequal_large_pools(metric):
    options = metric()
    assert options.max_samples is None
    draw = memorization_draw(60003, 51007, max_samples=options.max_samples)
    np.testing.assert_array_equal(draw.training, np.arange(60003))
    np.testing.assert_array_equal(draw.generated, np.arange(51007))


def test_auto_search_uses_native_routing_for_small_numeric_schema():
    assert DCR().device == "auto"
    assert DCR(device="cpu").device == "cpu"
    assert not hasattr(DCR(search_method="exact"), "backend")
    with pytest.raises(ValueError, match="exact"):
        DCR(search_method="approximate")


def test_holdout_path_rejects_training_metrics_instead_of_resampling():
    training = pd.DataFrame({"x": [0.0, 2.0, 4.0, 6.0]})
    with pytest.raises(ValueError, match="comparison metrics"):
        evaluate(training, training, Schema(numeric=["x"]), metrics=["shape", "dcr"])


def test_no_modeled_features_are_unavailable_without_distance_work(monkeypatch):
    import tabgen_eval.execution.runtime.memorization as module

    def forbidden(*args, **kwargs):
        pytest.fail("a schema without distance axes must not launch a neighbor search")

    monkeypatch.setattr(module, "ExecutionSession", forbidden)
    monkeypatch.setattr(module, "compute_memorization_group", forbidden)
    training = pd.DataFrame({"unused": [0.0, 1.0, 2.0]})
    result = evaluate_memorization(training, training, Schema())
    assert all(np.isnan(value) for value in _values(result).values())
    assert all(flags == ["NOT_APPLICABLE"] for flags in _numerical_flags(result).values())


def test_holdout_budget_is_independent_and_preserves_training_generated_draws():
    training = pd.DataFrame({"x": np.arange(17, dtype=float)})
    generated = pd.DataFrame({"x": np.arange(11, dtype=float) + 0.25})
    holdout = pd.DataFrame({"x": np.arange(31, dtype=float) + 0.75})
    options = make_options(seed=71)
    chosen = [configured_metric(DCR, options, max_samples=9, max_holdout_samples=13)]
    first = evaluate_memorization(
        training, generated, Schema(numeric=["x"]), holdout=holdout, options=options, metrics=chosen
    )
    second = evaluate_memorization(
        training,
        generated,
        Schema(numeric=["x"]),
        holdout=holdout.iloc[:5],
        options=options,
        metrics=chosen,
    )
    missing = evaluate_memorization(
        training, generated, Schema(numeric=["x"]), options=options, metrics=chosen
    )
    for result in [first, second, missing]:
        assert result.provenance["n_selected_training"] == 9
        assert result.provenance["n_selected_generated"] == 9
    for key in ["dcr/median", "dcr/q05"]:
        assert first[key].value == second[key].value == missing[key].value
    assert first.provenance["n_selected_holdout"] == 13
    assert second.provenance["n_selected_holdout"] == 5
    assert missing.provenance["n_selected_holdout"] is None
