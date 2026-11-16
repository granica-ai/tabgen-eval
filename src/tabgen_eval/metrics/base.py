"""Public metric interfaces and named statistic specifications.

Metric adapters declare their input requirements, statistic directions and
execution options. Numerical backends compute values; evaluation and report
layers supply sampling, comparison context and factual diagnostics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from numbers import Integral
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .._typing import Precision
from ..schema import Schema
from ..schema.validate import ValidationError

if TYPE_CHECKING:
    from tabgen_eval.execution.runtime.memory import MemoryPlanInfo


class MetricFamily(Enum):
    """The numerical work required to evaluate a metric comparison."""

    SAMPLE_BASED = "sample_based"  # statistic of the samples: resample + recompute
    ESTIMATOR_BASED = (
        "estimator_based"  # classifier fitting uses training-only transforms and refitting
    )


class Direction(Enum):
    """A statistic's preferred direction, retained as interpretation metadata."""

    HIGHER_BETTER = "higher_better"
    LOWER_BETTER = "lower_better"
    TARGET = "target"  # departures on either side of the target may matter


@dataclass(frozen=True)
class StatSpec:
    """One scalar a metric emits, with metadata needed to render and interpret it."""

    name: str  # "precision"
    display: str  # "Precision"
    direction: Direction
    bounds: tuple[float, float] | None = (0.0, 1.0)  # theoretical score range
    target: float | None = None  # for TARGET stats (e.g. recovery -> 1.0)
    fmt: str = ".4f"
    compare_to_ideal: bool = True  # is the real-real ideal meaningful here?


class Metric(ABC):
    """Base class for every metric.

    Subclasses set the class attributes below and implement :meth:`compute`. The
    engine supplies matched data roles, repeated scores, comparisons, and
    factual diagnostics around ``compute``.

    Shared execution options
    ------------------------
    All metrics accept ``device="auto"`` and ``dtype="float64"``,
    with automatic memory planning. Automatic routing selects an implemented CPU/
    CUDA path through Torch. ``device`` also
    accepts ``"cpu"``, ``"cuda"``, or ``"cuda:<index>"``; tensor ``dtype`` is
    ``"float32"`` or ``"float64"``. LinearC2ST supports both precisions on CPU
    and CUDA. Private constructor workspace overrides remain available for
    backend experiments; normal evaluation derives finite allowances from
    currently available memory. Unsupported execution
    combinations raise rather than silently changing requested precision/device.
    Per-metric statistical options are documented on the concrete metric classes.
    """

    name: str  # "prdc"
    device: str = "auto"
    dtype: Precision = "float64"
    gpu_workspace_mb: int = 2048
    memory_limit_mb: float
    _memory_plan: MemoryPlanInfo
    _memory_lease_bytes: int
    _memory_concurrent_tasks: int
    _explicit_memory_options: frozenset[str]
    _xgboost_device: str
    _xgboost_routing_reason: str
    requested_components: frozenset[str] | None
    fixed_execution_options: dict[str, str] = {}
    """Execution settings fixed by the metric instead of inherited from evaluate arguments."""
    category: str  # Public grouping for metric selection and reports.
    family: MetricFamily = MetricFamily.SAMPLE_BASED
    stats: tuple[StatSpec, ...] = ()
    n_min: int = 0
    """Optional required sampled rows per side; numerical cores enforce intrinsic requirements."""
    arity: int = 2  # 2 = (reference, candidate)
    needs_target: bool = False
    symmetric: bool = False  # M(a, b) == M(b, a)?
    bootstrapped: bool = True  # False => point-only: no repeated scores or paired baseline

    def __init__(self, **kwargs: object) -> None:
        """Accept internal computation overrides and remember explicit memory budgets."""
        self._explicit_memory_options = frozenset(kwargs) & {"gpu_workspace_mb", "memory_limit_mb"}
        fixed = type(self).fixed_execution_options
        for key, value in kwargs.items():
            if key == "fixed_execution_options" or not hasattr(self, key):
                raise TypeError(f"{type(self).__name__} got unexpected kwarg {key!r}")
            setattr(self, key, value)
        for key, required in fixed.items():
            requested = getattr(self, key)
            if requested != required and not (key == "device" and requested == "auto"):
                raise ValueError(f"{self.name} only supports {key}={required!r}, got {requested!r}")
            setattr(self, key, required)
        from tabgen_eval.execution.runtime.options import validate_options

        validate_options(self.device, self.dtype, self.gpu_workspace_mb)
        if (
            isinstance(self.n_min, (bool, np.bool_))
            or not isinstance(self.n_min, Integral)
            or self.n_min < 0
        ):
            raise ValueError("n_min must be a nonnegative integer")

    @abstractmethod
    def compute(
        self,
        reference: pd.DataFrame,
        candidate: pd.DataFrame,
        schema: Schema,
        *,
        rng: np.random.Generator,
    ) -> dict[str, float]:
        """Return ``{stat_name: value}``.

        Statistically undefined outcomes return NaN and accompanying diagnostics.
        Invalid inputs, unavailable dependencies, resource exhaustion, and
        programming errors may raise rather than being represented as scores.
        """

    def not_applicable(
        self, reference: pd.DataFrame | None, candidate: pd.DataFrame | None, schema: Schema
    ) -> set[str]:
        """Stats that are *structurally* undefined for this data (NOT_APPLICABLE, not failure).

        Example: Trend needs >= 2 columns; Conditional ``*_rare`` needs >= 1 rare slice.
        Default: nothing structurally inapplicable.
        """
        return set()

    def validate_inputs(
        self, reference: pd.DataFrame, candidate: pd.DataFrame, schema: Schema
    ) -> None:
        """Check metric-specific user requirements before fitting or sampling."""
        if self.needs_target and (
            schema.target is None
            or schema.target not in reference.columns
            or schema.target not in candidate.columns
        ):
            raise ValidationError(
                f"{self.name} requires an explicit Schema.target column in both frames"
            )

    @property
    def stat_names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.stats)

    def stat_spec(self, name: str) -> StatSpec:
        for s in self.stats:
            if s.name == name:
                return s
        raise KeyError(f"{self.name} has no stat {name!r}")
