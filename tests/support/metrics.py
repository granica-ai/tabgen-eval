"""Explicit metric fixtures for private numerical engine contracts."""

from typing import TypeVar

from tabgen_eval._options import ResolvedOptions
from tabgen_eval.metrics import Metric

MetricType = TypeVar("MetricType", bound=Metric)


def configured_metric(
    metric_cls: type[MetricType], options: ResolvedOptions, **overrides: object
) -> MetricType:
    """Construct a metric with global execution settings and explicit fixture options.

    Public evaluation uses metric defaults. Numerical contracts pass these instances
    directly to private engines so solver and edge-case fixtures remain independently
    configurable without exposing a second public configuration path.
    """
    options = {
        name: options[name]
        for name in ("device", "dtype")
        if name not in metric_cls.fixed_execution_options
    }
    return metric_cls(**(options | overrides))
