"""Memorization searches are exact regardless of workload or optional ANN packages."""

import builtins
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from tabgen_eval import Schema
from tabgen_eval.execution.torch.distance.planner import MemorizationPlan
from tabgen_eval.metrics import DCR
from tests.support.evaluation import memorization_report as evaluate_memorization
from tests.support.options import make_options

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "options",
    [
        {"projection_dim": 128},
        {"ivf_lists": 256},
        {"ivf_probes": 16},
        {"ivf_candidates": 256},
        {"cpu_ann_workspace_mb": 2048},
    ],
)
def test_removed_search_options_are_rejected(options):
    with pytest.raises(TypeError, match="unexpected kwarg"):
        DCR(**options)


def test_public_exact_results_match_independent_oracle_without_ann_imports(monkeypatch):
    original = builtins.__import__

    def without_ann(name, *args, **kwargs):
        if name.split(".")[0] in {"cuvs", "libcuvs", "rmm", "pylibraft", "faiss"}:
            pytest.fail(f"Exact memorization attempted an ANN import: {name}")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_ann)
    train = pd.DataFrame({"x": [0.0, 2.0, 4.0], "c": ["a", "b", "a"]})
    generated = pd.DataFrame({"x": [1.0, 3.0, 5.0], "c": ["a", "b", "new"]})
    held = pd.DataFrame({"x": [0.5, 2.5], "c": ["a", "b"]})
    schema = Schema(numeric=["x"], categorical=["c"])

    def distances(query):
        numeric = (query.x.to_numpy()[:, None] - train.x.to_numpy()[None, :]) / 4.0
        reference_codes = train.c.to_numpy()[None, :]
        query_codes = query.c.to_numpy()[:, None]
        categorical = np.where(
            np.isin(query_codes, ["a", "b"]), 2 * (query_codes != reference_codes), 1
        )
        return np.sqrt(numeric**2 + categorical)

    forward, held_distances = distances(generated), distances(held)
    expected = {
        "dcr/median": np.median(forward.min(1)),
        "dcr/q05": np.quantile(forward.min(1), 0.05),
        "dcr/ratio": np.median(forward.min(1)) / np.median(held_distances.min(1)),
        "near_match_rate/rate": (forward.min(1) <= 1e-3).mean(),
        "replication_ratio/rate": (forward.min(0) <= 1e-3).mean(),
    }
    report = evaluate_memorization(
        train,
        generated,
        schema,
        holdout=held,
        metrics=["dcr", "near_match_rate", "replication_ratio"],
        options=make_options(
            device="cpu",
            uncertainty="none",
        ),
    )
    for key, value in expected.items():
        assert report[key].value == pytest.approx(value)
    assert report.results[0].diagnostics["search_method"] == "exact"


def test_large_workload_runs_requested_exact_searches_without_a_work_refusal(monkeypatch):
    from tabgen_eval.execution.torch.distance import planner

    def features(rows):
        return SimpleNamespace(
            n_rows=rows,
            numeric=SimpleNamespace(shape=(rows, 98), device=SimpleNamespace(type="cpu")),
            categorical=SimpleNamespace(shape=(rows, 47)),
        )

    monkeypatch.setattr(planner, "finite", lambda _: True)
    training, generated, holdout = features(1_000_000), features(125_000), features(125_000)
    plan = MemorizationPlan(training, generated, SimpleNamespace(), [DCR()], holdout=holdout)
    calls = []

    def nearest(query, *, reference=training):
        calls.append((reference.n_rows, query.n_rows))
        return torch.full((query.n_rows,), 2.0 if query is generated else 4.0)

    monkeypatch.setattr(plan, "_nearest", nearest)
    outcome = plan.execute()["dcr"]
    assert calls == [(1_000_000, 125_000), (1_000_000, 125_000)]
    assert outcome.values["median"] == 2.0
    assert outcome.values["ratio"] == 0.5
    for percentile in (5, 10, 25, 50, 75, 90, 95, 99):
        assert outcome.values[f"q{percentile:02d}"] == 2.0
        assert outcome.values[f"holdout_q{percentile:02d}"] == 4.0
    assert outcome.diagnostics["search_method"] == "exact"
    assert outcome.diagnostics["searches"] == {"forward": 1, "holdout_to_training": 1, "reverse": 0}


@pytest.mark.parametrize("search_method", ["auto", "approximate"])
def test_nonexact_searches_are_rejected(search_method):
    with pytest.raises(ValueError, match="only supports"):
        DCR(search_method=search_method)
