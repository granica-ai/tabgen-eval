"""Supervised native-category contracts for predictive MLE."""

import numpy as np
import pandas as pd
import pytest

from tabgen_eval import Schema
from tabgen_eval.preprocessing.predictors import PredictorPreprocessor, match_predictor_precision


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)])
def test_high_cardinality_predictors_stay_compact_under_a_small_memory_budget(device):
    import torch

    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    size = 10_000
    frame = pd.DataFrame(
        {
            "x": np.arange(size, dtype=float),
            "c": [f"category-{i}" for i in range(size)],
            "target": 0.0,
        }
    )
    schema = Schema(numeric=["x", "target"], categorical=["c"], target="target")
    state, design = PredictorPreprocessor.fit_transform(frame, schema, 32 * 1024**2, device=device)
    assert str(design.device) == device
    assert design.shape == (size, 2)
    assert state.feature_names == ("x", "c")
    assert state.feature_types == ("q", "c")
    assert state.categorical_cardinalities == (size,)
    np.testing.assert_array_equal(design[:, 0].cpu(), frame.x)
    assert len(np.unique(design[:, 1].cpu())) == size


def test_supervised_state_is_training_only_and_keeps_missing_distinct_from_unknown():
    train = pd.DataFrame(
        {
            "x": [100.0, 200, np.nan, 400],
            "empty": [np.nan] * 4,
            "c": ["A", " A ", None, "nan"],
            "target": [0, 1, 0, 1],
        }
    )
    schema = Schema(numeric=["x", "empty", "target"], categorical=["c"], target="target")
    state, design = PredictorPreprocessor.fit_transform(train, schema, 1024**2)
    test = pd.DataFrame(
        {"x": [1e20, np.nan, 1], "empty": [3, 4, 5], "c": ["unseen", None, "A"], "target": [99] * 3}
    )
    scoring = state.transform(test).numpy()
    assert state.feature_names == ("x", "c")
    assert scoring.shape == (3, 2) and scoring[0, 0] == 1e20
    assert np.isnan(design[2, 0]) and np.isnan(scoring[1, 0])
    assert scoring[0, 1] == 4
    assert scoring[1, 1] == design[2, 1] != scoring[0, 1]
    assert scoring[2, 1] == design[0, 1]
    np.testing.assert_array_equal(state.transform(train).numpy(), design.numpy())


def test_supervised_precision_matching_is_symmetric_and_does_not_mutate_sources():
    low = pd.DataFrame({"x": np.array([0.1, 0.2, 0.3], dtype=np.float32)})
    high = pd.DataFrame({"x": [0.1, 0.2, 0.3]})
    original = high.copy()
    schema = Schema(numeric=["x"])
    (a, b, c), precision = match_predictor_precision((low, high, high), schema)
    assert b is c
    assert precision["observation_bits"] == {"x": 32}
    np.testing.assert_array_equal(a.x, b.x)
    pd.testing.assert_frame_equal(high, original)
    (b, a), _ = match_predictor_precision((high, low), schema)
    np.testing.assert_array_equal(a.x, b.x)


@pytest.mark.parametrize("target_role", ["numeric", "categorical"])
def test_target_role_controls_observation_matching_and_label_identity(target_role):
    """Numeric targets share source precision; categorical class identities do not."""
    labels = np.array([1.0, 1.0 + 2**-30], dtype=np.float64)
    real = pd.DataFrame({"x": [0.0, 1.0], "target": labels})
    generated = real.assign(target=labels.astype(np.float32))
    originals = [frame.copy(deep=True) for frame in (real, generated)]
    schema = Schema(
        numeric=["x", "target"] if target_role == "numeric" else ["x"],
        categorical=["target"] if target_role == "categorical" else [],
        target="target",
    )
    (training, scoring, _), precision = match_predictor_precision((real, real, generated), schema)
    assert training is scoring
    if target_role == "numeric":
        assert precision["observation_bits"]["target"] == 32
        np.testing.assert_array_equal(scoring.target, [1.0, 1.0])
        assert scoring.target.nunique() == 1
    else:
        assert "target" not in precision["observation_bits"]
        np.testing.assert_array_equal(scoring.target, labels)
        assert scoring.target.nunique() == 2
    for frame, original in zip((real, generated), originals, strict=True):
        pd.testing.assert_frame_equal(frame, original)
