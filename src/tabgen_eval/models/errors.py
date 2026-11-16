"""Shared failure contract for fitted metrics, independent of any classifier.

Metric adapters translate noncomputable fits into flagged results. Input misuse
and missing/broken optional dependencies remain ordinary exceptions and are not
caught as statistical failures.
"""

from __future__ import annotations


class FitUnavailable(Exception):
    """A valid comparison cannot produce a usable fitted score.

    Examples include empty labelled pools, unsupported class sets, no usable
    features, a preprocessing memory limit, or an engine fit/prediction failure.
    This is not specific to cross-validation: utility models use it too.

    ``reason`` is a stable diagnostic code; ``diagnostics`` retains engine
    details. C2ST and utility catch this exception and report an undefined score
    with fit diagnostics. Nonfinite replicate scores are excluded from summaries
    with counts and reasons; finite nonconverged scores remain available.

    Do not use this exception for malformed API inputs, missing XGBoost, or a
    broken installation; those errors must remain visible to the caller.
    """

    def __init__(
        self, reason: str = "preprocessing_or_memory_limit", **diagnostics: object
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.diagnostics = diagnostics
