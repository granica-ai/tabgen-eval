"""Raw regression scoring over caller-supplied truth and predictions.

Mean absolute error preserves undefined inputs as NaN. Fitting, predictor
preprocessing, source selection, and report construction belong to callers."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import mean_absolute_error


def regression_mae(truth: ArrayLike, predictions: ArrayLike) -> float:
    """Return raw mean absolute error for nonempty finite paired observations.

    One scoring observation and constant targets are valid. Empty, misaligned,
    or nonfinite inputs have no defined score and return NaN.
    """
    truth, predictions = np.asarray(truth), np.asarray(predictions)
    if (
        truth.ndim != 1
        or predictions.shape != truth.shape
        or not len(truth)
        or not np.isfinite(truth).all()
        or not np.isfinite(predictions).all()
    ):
        return float("nan")
    return float(mean_absolute_error(truth, predictions))
