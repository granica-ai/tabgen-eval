"""Reusable model fitting, prediction, and statistical fit-failure contracts.

Optional engines are loaded when used, independently of metric definitions.
"""

from .errors import FitUnavailable
from .xgboost import XGBoostModel

__all__ = ["FitUnavailable", "XGBoostModel"]
