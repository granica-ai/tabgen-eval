"""Unified tabular evaluation and an independent data-diagnostics workflow.

Use ``evaluate(EvalData(...))`` for metrics and optional repeated-score bands.
Use ``diagnose(train=..., generated=...)`` for a separate DiagnosticReport.
"""

from __future__ import annotations

from ._options import UncertaintyOptions
from .data import EvalData
from .datasets import load_openml
from .diagnostics import DiagnosticReport, diagnose
from .evaluate import evaluate
from .registry import list_groups, list_metrics
from .report import (
    FidelityReport,
    MemorizationReport,
    MetricResult,
    Report,
    StatResult,
    UtilityReport,
)
from .report.diagnostics import DiagnosticFlag
from .schema import Schema, ValidationError

__version__ = "0.1.0"

__all__ = [
    "evaluate",
    "EvalData",
    "UncertaintyOptions",
    "Schema",
    "Report",
    "FidelityReport",
    "MemorizationReport",
    "UtilityReport",
    "MetricResult",
    "StatResult",
    "DiagnosticFlag",
    "ValidationError",
    "diagnose",
    "DiagnosticReport",
    "list_metrics",
    "list_groups",
    "load_openml",
    "__version__",
]
