"""Data profiling outside the metric-evaluation workflow."""

from .api import diagnose
from .report import DiagnosticReport

__all__ = ["DiagnosticReport", "diagnose"]
