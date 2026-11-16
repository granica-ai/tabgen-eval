"""Public metric names, role requirements, and shared computation planning."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TypedDict, cast

from .._options import ResolvedOptions
from ..metrics import Metric
from .registry import get_by_name, resolve_metrics


@dataclass(frozen=True)
class MetricSelection:
    """One independently selectable result and the computation supplying it."""

    name: str
    group: str
    computation: str
    outputs: tuple[str, ...]
    primary: str
    required_roles: frozenset[str]


def _selection(
    name: str,
    group: str,
    computation: str | None = None,
    outputs: tuple[str, ...] | None = None,
    *,
    primary: str | None = None,
    required_roles: frozenset[str] = frozenset({"test"}),
) -> MetricSelection:
    computation = computation or name
    spec = get_by_name(computation)
    if spec is None:
        raise RuntimeError(f"Unregistered computation {computation!r}")
    outputs = outputs if outputs is not None else tuple(s.name for s in spec.metric_cls.stats)
    return MetricSelection(name, group, computation, outputs, primary or outputs[0], required_roles)


_MEMORIZATION_ROLES = frozenset({"train", "generated"})

# Order controls selection, execution planning, and discovery display order.
SELECTIONS: tuple[MetricSelection, ...] = (
    _selection("shape", "fidelity"),
    _selection("trend", "fidelity"),
    *(
        _selection(name, "geometric_fidelity", "prdc", (name,))
        for name in ("precision", "recall", "density", "coverage")
    ),
    *(
        _selection(name, "conditional_fidelity", "conditional_fidelity", outputs)
        for name, outputs in (
            ("cond_shape", ("cond_shape_all", "cond_shape_rare")),
            ("cond_trend", ("cond_trend_all", "cond_trend_rare")),
            ("recovery", ("recovery_all", "recovery_rare")),
        )
    ),
    _selection("concentration", "concentration"),
    _selection("linear_c2st", "detection"),
    _selection("nonlinear_c2st", "detection"),
    _selection(
        "dcr_ratio",
        "memorization",
        "dcr",
        primary="ratio",
        required_roles=_MEMORIZATION_ROLES | {"test"},
    ),
    _selection("near_match_rate", "memorization", required_roles=_MEMORIZATION_ROLES),
    _selection("replication_ratio", "memorization", required_roles=_MEMORIZATION_ROLES),
    _selection("mle", "utility"),
)

_BY_NAME = {selection.name: selection for selection in SELECTIONS}
_GROUPS = tuple(dict.fromkeys(selection.group for selection in SELECTIONS))

MetricDescription = TypedDict(
    "MetricDescription",
    {
        "name": str,
        "group": str,
        "category": str,
        "computation": str,
        "outputs": list[str],
        "primary": str,
        "required_roles": list[str],
        "class": str,
        "enabled_by_default": bool,
    },
)


def list_groups() -> list[str]:
    """Return the registered group names in display order."""
    return list(_GROUPS)


def list_metrics() -> list[MetricDescription]:
    """Describe registered public names, companion results, and required roles."""
    return [
        {
            "name": selection.name,
            "group": selection.group,
            "category": selection.group,
            "computation": selection.computation,
            "outputs": list(selection.outputs),
            "primary": selection.primary,
            "required_roles": sorted(selection.required_roles),
            "class": _computation_class_name(selection.computation),
            "enabled_by_default": True,
        }
        for selection in SELECTIONS
    ]


def _computation_class_name(name: str) -> str:
    spec = get_by_name(name)
    if spec is None:
        raise RuntimeError(f"Unregistered computation {name!r}")
    return spec.metric_cls.__name__


def _tokens(value: Iterable[object] | None, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        raise TypeError(f"{name} must be a sequence of registered names, not a string")
    try:
        tokens = tuple(value)
    except TypeError as error:
        raise TypeError(f"{name} must be a sequence of registered names") from error
    if any(not isinstance(token, str) for token in tokens):
        raise TypeError(f"{name} names must be strings")
    return cast("tuple[str, ...]", tokens)


def resolve_selection(
    *, group: Iterable[str] = (), metrics: Iterable[str] = (), roles: Iterable[str]
) -> tuple[MetricSelection, ...]:
    """Validate filters and select all requested results in registry order.

    A nonempty metric filter overrides the group filter. Both filters' names
    are validated, while compatibility applies to the effective selection.
    Empty filters select every result supported by the supplied data roles.
    Explicit groups are complete: a missing role for any member raises.
    """
    groups, names = _tokens(group, "group"), _tokens(metrics, "metrics")
    for name, tokens, known in (
        ("group", groups, _GROUPS),
        ("metric", names, _BY_NAME),
    ):
        unknown = sorted(set(tokens) - set(known))
        if unknown:
            raise ValueError(f"Unknown {name} name(s): {unknown}. Available {name}s: {list(known)}")
    roles = frozenset(roles)
    if names:
        selected = tuple(selection for selection in SELECTIONS if selection.name in names)
    elif groups:
        selected = tuple(selection for selection in SELECTIONS if selection.group in groups)
    else:
        return tuple(selection for selection in SELECTIONS if selection.required_roles <= roles)

    incompatible = [
        f"{selection.name} requires {', '.join(sorted(selection.required_roles - roles))}"
        for selection in selected
        if not selection.required_roles <= roles
    ]
    if incompatible:
        supplied = ", ".join(sorted(roles)) or "none"
        raise ValueError(
            f"Metrics incompatible with supplied data roles ({supplied}): "
            + "; ".join(incompatible)
        )
    return selected


def build_metrics(selection: Iterable[MetricSelection], options: ResolvedOptions) -> list[Metric]:
    """Instantiate each shared computation once and declare requested components."""
    components: dict[str, set[str]] = {}
    for item in selection:
        components.setdefault(item.computation, set()).add(item.name)
    chosen = resolve_metrics(list(components), options)
    for metric in chosen:
        metric.requested_components = frozenset(components[metric.name])
    return chosen
