"""Nonfinite linear transforms fail safely without rejecting tree missing values."""

import numpy as np
import pandas as pd
import pytest

from tabgen_eval import Schema
from tabgen_eval.models import FitUnavailable
from tabgen_eval.preprocessing import fit_fold, prepare_sources
from tests.support.evaluation import paired_report as evaluate
from tests.support.options import make_options


def test_finite_extreme_observations_flag_unusable_linear_transforms(device, dtype):
    large = 3e38 if dtype == "float32" else 1e308
    frame = pd.DataFrame({"x": np.array([large] * 4 + [-large] * 8, dtype=dtype)})
    sources = prepare_sources(
        frame, frame.copy(), Schema(numeric=["x"]), device=device, dtype=dtype
    )
    folds = np.repeat(np.arange(3), 4)
    with pytest.raises(FitUnavailable, match="^nonfinite_classifier_preprocessing$"):
        fit_fold(
            sources.reference,
            sources.candidate,
            folds,
            folds,
            0,
            cardinalities=sources.cardinalities,
            numeric_names=sources.numeric_names,
            categorical_names=sources.categorical_names,
            policy="linear",
        )


def test_tree_preprocessing_retains_missing_heldout_observations(torch, device, dtype):
    frame = pd.DataFrame({"x": np.array([np.nan] * 4 + [1.0] * 8, dtype=dtype)})
    sources = prepare_sources(
        frame, frame.copy(), Schema(numeric=["x"]), device=device, dtype=dtype
    )
    folds = np.repeat(np.arange(3), 4)
    data = fit_fold(
        sources.reference,
        sources.candidate,
        folds,
        folds,
        0,
        cardinalities=sources.cardinalities,
        numeric_names=sources.numeric_names,
        categorical_names=sources.categorical_names,
        policy="xgboost",
    )
    assert torch.isfinite(data.train).all()
    assert torch.isnan(data.test).all()


def test_extreme_finite_frames_return_flagged_classifier_reports(dtype):
    large = 3e38 if dtype == "float32" else 1e308
    frame = pd.DataFrame({"x": np.array([-large] * 59 + [large], dtype=dtype)})
    report = evaluate(
        frame,
        frame.copy(),
        Schema(numeric=["x"]),
        metrics=["linear_c2st"],
        options=make_options(uncertainty="none", seed=0, device="cpu", dtype=dtype),
    )
    row = report["linear_c2st/auc"]
    assert np.isnan(row.value)
    assert any(flag.name == "NAN_VALUE" for flag in row.flags)
