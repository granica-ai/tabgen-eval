"""Unit tests for the NaN-safe numeric helpers."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tabgen_eval.execution.python.stats import safe_nanmean, safe_ratio

pytestmark = pytest.mark.unit


def test_safe_nanmean_drops_nonfinite():
    assert safe_nanmean([1.0, 2.0, np.nan, np.inf]) == 1.5


def test_safe_nanmean_all_nan_returns_default():
    assert math.isnan(safe_nanmean([np.nan, np.inf]))
    assert safe_nanmean([], default=0.0) == 0.0


def test_safe_ratio_zero_denominator_is_nan():
    assert math.isnan(safe_ratio(1.0, 0.0))
    assert math.isnan(safe_ratio(1.0, np.nan))
    assert safe_ratio(1.0, 2.0) == 0.5
