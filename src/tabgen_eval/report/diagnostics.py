"""Factual diagnostics accompanying scalar values and replicate summaries."""

from __future__ import annotations

from enum import Enum


class DiagnosticFlag(Enum):
    """An observed applicability, numerical, fitting, or sampling condition.

    Diagnostics describe what was computed and what happened. They do not
    establish statistical reliability or make a quality judgment about a
    generator. Counts and fitted-model details remain available in the result.
    """

    INSUFFICIENT_DATA = (
        "INSUFFICIENT_DATA",
        "the metric's minimum data requirement is not satisfied",
    )
    NAN_VALUE = ("NAN_VALUE", "the point score is nonfinite")
    MISSING_HOLDOUT = ("MISSING_HOLDOUT", "an independent real holdout was not supplied")
    INSUFFICIENT_HOLDOUT_ROWS = (
        "INSUFFICIENT_HOLDOUT_ROWS",
        "the independent real holdout has no usable rows",
    )
    INSUFFICIENT_TRAINING_ROWS = (
        "INSUFFICIENT_TRAINING_ROWS",
        "the training pool has no usable rows",
    )
    ZERO_BASELINE = ("ZERO_BASELINE", "the baseline is zero; the ratio is undefined")
    CLASSIFIER_NOT_CONVERGED = (
        "CLASSIFIER_NOT_CONVERGED",
        "a model fit did not meet its stopping criterion; finite scores are retained",
    )
    DEGENERATE_CI = ("DEGENERATE_CI", "all finite replicate scores are identical")
    NONFINITE_REPLICATES = (
        "NONFINITE_REPLICATES",
        "nonfinite replicate scores are excluded from summaries; counts and reasons are reported",
    )
    CATEGORY_VOCAB_MISMATCH = (
        "CATEGORY_VOCAB_MISMATCH",
        "the input pools contain different categorical vocabularies",
    )
    NOT_APPLICABLE = ("NOT_APPLICABLE", "the statistic is structurally undefined for this data")
    NO_CI_REQUESTED = ("NO_CI_REQUESTED", "no repeated-score distribution was requested")

    @property
    def code(self) -> str:
        """Machine-readable diagnostic identifier."""
        return self.value[0]

    @property
    def description(self) -> str:
        """Factual explanation displayed in report legends."""
        return self.value[1]

    @classmethod
    def by_code(cls, code: str) -> DiagnosticFlag | None:
        """Return a known diagnostic by code, or None for an unknown code."""
        return _BY_CODE.get(code)

    def __str__(self) -> str:
        return self.code


_BY_CODE = {flag.code: flag for flag in DiagnosticFlag}
