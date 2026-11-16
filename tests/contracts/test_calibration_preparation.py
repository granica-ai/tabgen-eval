"""Calibration preserves shared categorical identity and missing-data preparation."""

import numpy as np
import pandas as pd
import pytest

from tabgen_eval import ValidationError
from tabgen_eval.execution.runtime.sampling import SamplingPlan, draw_batches
from tabgen_eval.metrics import Conditional, Shape, Trend
from tabgen_eval.schema import Schema
from tests.support.evaluation import reference_scores as calibrate
from tests.support.evaluation import summaries
from tests.support.options import make_options

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("mixed", [False, True])
def test_calibration_observed_values_match_each_draw_and_preserve_caller(mixed):
    rng = np.random.default_rng(913)
    labels = (
        [1, 1.0, "1", " 1 ", "nan", "<MISSING>", None]
        if mixed
        else ["A", " A ", "1.0", "1", "nan", "<MISSING>", None]
    )
    c = np.empty(350, dtype=object)
    c[:] = [labels[index % len(labels)] for index in range(len(c))]
    real = pd.DataFrame({"x": rng.normal(size=350), "y": rng.normal(size=350), "c": c})
    real.loc[::17, "x"] = np.nan
    before = real.copy(deep=True)
    schema = Schema(numeric=["x", "y"], categorical=["c"])
    options = make_options()
    result = calibrate(
        real,
        schema,
        metrics=["shape", "trend", "conditional_fidelity"],
        n_replicates=3,
        seed=27,
        options=options,
    )
    expected = {metric.name: [] for metric in [Shape(), Trend(), Conditional()]}
    for batch in draw_batches(SamplingPlan.create(350), n_replicates=3, seed=27, batch_size=1):
        r = real.iloc[batch.reference[0]].reset_index(drop=True)
        s = real.iloc[batch.candidate[0]].reset_index(drop=True)
        for cls in [Shape, Trend, Conditional]:
            metric = cls()
            expected[metric.name].append(metric.compute(r, s, schema, rng=None))
    for name, draws in expected.items():
        for stat, summary in summaries(result.metrics[name]).items():
            np.testing.assert_allclose(
                summary.samples, [draw[stat] for draw in draws], equal_nan=True
            )
    pd.testing.assert_frame_equal(real, before)
    assert "sample_budget" not in result.metadata
    assert result.metadata["n_res"] == 175


def test_calibration_rejects_numeric_text_without_conversion():
    real = pd.DataFrame({"x": ["1", "bad", "2", "3"]})
    with pytest.raises(ValidationError, match="x"):
        calibrate(
            real,
            Schema(numeric=["x"]),
            metrics=["shape"],
            options=make_options(),
        )
