"""Explicit input roles for unified evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ._validation import require_instance
from .schema import Schema

_ROLE_NAMES = ("train", "test", "generated")
_VALID_ROLES = (
    frozenset({"test"}),
    frozenset({"test", "generated"}),
    frozenset({"train", "generated"}),
    frozenset({"train", "test", "generated"}),
)
_VALID_CONFIGURATIONS = "test; generated + test; generated + train; train + test + generated"


def validate_roles(roles: frozenset[str]) -> None:
    """Require one supported configuration for evaluation or input discovery."""
    if roles not in _VALID_ROLES:
        supplied = ", ".join(sorted(roles)) or "none"
        raise ValueError(
            f"Invalid EvalData configuration ({supplied}). "
            f"Valid configurations: {_VALID_CONFIGURATIONS}."
        )


@dataclass(frozen=True, kw_only=True, eq=False)
class EvalData:
    """DataFrames in their statistical roles, with explicit column declarations.

    ``train`` is the data used to fit the generator, ``test`` is independent real
    data, and ``generated`` is the generator's output. Test alone compares
    disjoint real subsets. A ``Schema`` is required; ``Schema.infer`` can propose
    roles for callers to inspect and accept explicitly. Utility requires
    ``schema.target``.

    Freezing prevents rebinding fields; callers still own the DataFrames and
    should not modify them while an evaluation is running.

    Construction prints the default metric selection for the supplied roles.
    Evaluation prerequisites, including the utility target, are checked by
    ``evaluate`` after any metric filters have been applied.
    """

    train: pd.DataFrame | None = None
    test: pd.DataFrame | None = None
    generated: pd.DataFrame | None = None
    schema: Schema

    def __post_init__(self) -> None:
        for role in _ROLE_NAMES:
            frame = getattr(self, role)
            if frame is not None and not isinstance(frame, pd.DataFrame):
                raise TypeError(f"EvalData.{role} must be a pandas DataFrame or None")
        require_instance(
            self.schema,
            Schema,
            "EvalData.schema must be a Schema; use Schema.infer explicitly if needed",
        )
        validate_roles(self.roles)
        print(_default_metric_summary(self.roles, self.schema))

    @property
    def roles(self) -> frozenset[str]:
        """Supplied roles; an empty DataFrame is supplied, rather than absent."""
        return frozenset(role for role in _ROLE_NAMES if getattr(self, role) is not None)


def _default_metric_summary(roles: frozenset[str], schema: Schema) -> str:
    """Describe registry defaults without preparing data or loading metric engines."""
    from .registry import resolve_selection

    selection = resolve_selection(roles=roles)
    grouped: dict[str, list[str]] = {}
    for metric in selection:
        grouped.setdefault(metric.group, []).append(metric.name)
    supplied = " + ".join(role for role in _ROLE_NAMES if role in roles)
    lines = [f"EvalData ({supplied}) — default evaluate metrics:"]
    lines.extend(f"  {group}: {', '.join(names)}" for group, names in grouped.items())
    if any(metric.name == "mle" for metric in selection) and schema.target is None:
        lines.append(
            "  mle requires schema.target; supply it or filter out utility when evaluating."
        )
    return "\n".join(lines)
