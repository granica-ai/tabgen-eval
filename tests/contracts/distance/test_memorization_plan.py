"""Exact DCR draw resources are released before gathering the next draw."""

import weakref

import pytest
import torch

from tabgen_eval.execution.torch.distance.planner import MemorizationPlan
from tabgen_eval.metrics import DCR

pytestmark = pytest.mark.unit


def test_batched_memorization_releases_plan_and_features_before_next_gather(monkeypatch):
    import numpy as np
    import pandas as pd

    from tabgen_eval import Schema
    from tabgen_eval.execution.runtime.distance_outcomes import memorization_metric_outcome
    from tabgen_eval.execution.runtime.memorization import prepare_memorization_columns
    from tabgen_eval.execution.torch.context import TorchContext
    from tabgen_eval.execution.torch.distance.executor import DistanceExecutor

    training = pd.DataFrame({"x": [0.0, 2.0, 8.0]})
    generated = training + 0.25
    held = training + 0.5
    columns = prepare_memorization_columns(training, generated, held, Schema(numeric=["x"]))
    executor = DistanceExecutor(TorchContext.from_columns(columns))
    retained = []
    gathers = []
    original = executor._samples

    def gather(pools, draws, **kwargs):
        # Check at the boundary before gathering any buffer for the next draw.
        assert all(reference() is None for draw in retained for reference in draw)
        gathers.append(len(pools))
        return original(pools, draws, **kwargs)

    def execute(plan):
        plan._forward = torch.ones(plan.generated.n_rows)
        retained.append(
            tuple(
                weakref.ref(item)
                for item in (
                    plan,
                    plan._forward,
                    plan.reference.numeric,
                    plan.generated.numeric,
                    plan.holdout.numeric,
                )
            )
        )
        return {
            metric.name: memorization_metric_outcome(
                metric,
                counts=(plan.reference.n_rows, plan.generated.n_rows),
                median=0.25,
                q05=0.25,
                holdout_median=0.5,
                holdout_count=plan.holdout.n_rows,
            )
            for metric in plan.metrics
        }

    monkeypatch.setattr(executor, "_samples", gather)
    monkeypatch.setattr(MemorizationPlan, "execute", execute)
    draws = np.array([[0, 1, 2], [2, 1, 0]])
    outcomes = executor.compute_memorization_group([DCR()], draws, draws, holdout_draws=draws)
    assert gathers == [3, 3]
    assert len(outcomes) == 2
    assert all(reference() is None for draw in retained for reference in draw)
    assert [outcome["dcr"].values["ratio"] for outcome in outcomes] == [0.5, 0.5]
