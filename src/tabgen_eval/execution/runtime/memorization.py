"""Prepare independently selected pools and share memorization searches."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, TypedDict, cast

import numpy as np
import pandas as pd

from tabgen_eval._options import ResolvedOptions, resolve_options
from tabgen_eval.execution.python.distance import compute_memorization_group
from tabgen_eval.preprocessing.arrow import try_prepare, try_prepare_memorization
from tabgen_eval.preprocessing.columns import HostTable, PreparedColumns
from tabgen_eval.schema import Schema, validate
from tabgen_eval.schema.schema import SchemaDeclaration

from .dispatch import ExecutionSession
from .domains import SourceDomain, metric_input_schema
from .memorization_sampling import memorization_draw
from .outcomes import MemorizationOutcome

if TYPE_CHECKING:
    from tabgen_eval._typing import IntArray, SeedEntropy
    from tabgen_eval.metrics.memorization import DCR, _MemorizationMetric

    from .context import ExecutionContext, NumericalContext


class _MemorizationHoldout(TypedDict, total=False):
    baseline: str
    n_holdout: int | None
    n_eligible_holdout: int | None
    max_holdout_samples: int | None
    n_selected_holdout: int | None
    n_retained_holdout: int | None


class MemorizationMetadata(_MemorizationHoldout):
    """Original, eligible, and selected source counts for distance comparisons."""

    n_training: int
    n_generated: int
    n_eligible_training: int
    n_eligible_generated: int
    max_samples: int | None
    n_selected_training: int
    n_selected_generated: int
    n_retained_training: int
    n_retained_generated: int
    seed: SeedEntropy
    schema: SchemaDeclaration
    schema_inferred: bool
    reference: str
    sampling: str


@dataclass(frozen=True)
class MemorizationExecution:
    """Per-metric outcomes and execution choices over shared selected source pools."""

    outcomes: dict[str, MemorizationOutcome]
    execution: dict[str, dict[str, object]]
    metadata: MemorizationMetadata


def prepare_memorization_columns(
    training: pd.DataFrame, generated: pd.DataFrame, holdout: pd.DataFrame, schema: Schema
) -> PreparedColumns:
    """Encode three eligible pools with shared identities, preserving their row roles."""
    # Preserve typed categorical scalars when pandas reconciles pool dtypes.
    pools = [
        frame.assign(**{c: frame[c].astype(object) for c in schema.categorical})
        for frame in (generated, holdout)
    ]
    queries = pd.concat(pools, ignore_index=True)
    columns = PreparedColumns.prepare(training, queries, schema)
    table = columns.candidate

    def rows(start: int, stop: int) -> HostTable:
        return HostTable(
            table.numeric[:, start:stop],
            table.categorical[:, start:stop],
            table.complete[start:stop],
        )

    return replace(
        columns, candidate=rows(0, len(generated)), holdout=rows(len(generated), len(queries))
    )


def validate_memorization_options(metrics: Sequence[_MemorizationMetric]) -> None:
    """Reject incompatible shared source budgets before any metric computation."""
    if not metrics:
        raise ValueError("At least one memorization metric is required")
    limits = {metric.max_samples for metric in metrics}
    if len(limits) != 1:
        raise ValueError("Memorization metrics in one evaluation must use the same max_samples")


def run_memorization(
    training: pd.DataFrame,
    generated: pd.DataFrame,
    schema: Schema,
    metrics: Sequence[_MemorizationMetric],
    options: ResolvedOptions,
    *,
    holdout: pd.DataFrame | None = None,
) -> MemorizationExecution:
    """Select and prepare rows once, then execute compatible metrics together.

    Selected metrics must share ``max_samples`` so their scores describe the same
    source draws. Numerical contexts may differ by explicitly requested execution
    options; compatible contexts share fitted geometry and distance searches.
    """
    validate_memorization_options(metrics)
    if not isinstance(schema, Schema):
        raise TypeError("schema must be an explicit Schema instance")
    validation_schema = metric_input_schema(schema, metrics)
    validate(training, generated, validation_schema, reference_name="train")
    dcr = next((cast("DCR", metric) for metric in metrics if metric.name == "dcr"), None)
    if holdout is not None and dcr is not None:
        validate(
            training,
            holdout,
            validation_schema,
            allow_empty=True,
            reference_name="train",
            candidate_name="test",
        )
    pools = {"train": training, "generated": generated}
    if holdout is not None and dcr is not None:
        pools["test"] = holdout
    domain = SourceDomain.prepare(
        pools,
        policy="complete_case",
        columns=tuple(schema.numeric) + tuple(schema.categorical),
    )
    training, generated = domain.pools["train"], domain.pools["generated"]
    if "test" in domain.pools:
        holdout = domain.pools["test"]
    draw = memorization_draw(
        len(training), len(generated), max_samples=metrics[0].max_samples, seed=options["seed"]
    )
    selected_training, selected_generated = (
        training.iloc[draw.training],
        generated.iloc[draw.generated],
    )
    holdout_positions: IntArray | None = None
    if holdout is not None and dcr is not None:
        cap = dcr.max_holdout_samples
        holdout_positions = np.arange(len(holdout), dtype=np.int64)
        if cap is not None and len(holdout) > cap:
            stream = np.random.SeedSequence(options["seed"]).spawn(3)[2]
            holdout_positions = np.random.default_rng(stream).choice(
                len(holdout), cap, replace=False
            )
        selected_holdout = holdout.iloc[holdout_positions]
    else:
        selected_holdout = None
    unavailable: dict[str, set[str]] = {}
    for metric in metrics:
        metric.validate_inputs(selected_training, selected_generated, schema)
        unavailable[metric.name] = metric.not_applicable(
            selected_training, selected_generated, schema
        )
    active = [metric for metric in metrics if not unavailable[metric.name]]
    populated = bool(len(selected_training) and len(selected_generated))
    tensor_only = bool(active and populated)
    if selected_holdout is not None and tensor_only:
        prepared = try_prepare_memorization(
            selected_training,
            selected_generated,
            selected_holdout,
            schema,
        )
    elif tensor_only:
        prepared = try_prepare(
            selected_training,
            selected_generated,
            schema,
        )
    else:
        prepared = None
    columns = prepared
    if columns is None and selected_holdout is not None and active and populated:
        columns = prepare_memorization_columns(
            selected_training, selected_generated, selected_holdout, schema
        )
    session = (
        ExecutionSession(
            selected_training,
            selected_generated,
            schema,
            n_jobs=options["n_jobs"],
            n_eff=max(len(draw.training), len(draw.generated)),
            n_bootstrap=0,
            columns=columns,
        )
        if active and populated
        else None
    )
    groups: dict[int, tuple[NumericalContext | None, list[_MemorizationMetric]]] = {}
    execution: dict[str, dict[str, object]] = {}
    for metric in active:
        context = session.context_for(metric) if session is not None else None
        key = id(context)
        if key not in groups:
            groups[key] = (context, [])
        groups[key][1].append(metric)
        execution[metric.name] = (
            {"implementation": "python", "device": "cpu", "dtype": "float64"}
            if context is None
            else context.execution_info_for(metric)
        )
    outcomes: dict[str, MemorizationOutcome] = {}
    for context, group in groups.values():
        if context is None:
            outcomes.update(
                compute_memorization_group(
                    selected_training,
                    selected_generated,
                    schema,
                    group,
                    **({} if selected_holdout is None else {"holdout": selected_holdout}),
                )
            )
        else:
            outcomes.update(
                cast("ExecutionContext", context).compute_memorization_group(
                    group,
                    np.arange(len(draw.training), dtype=np.int64)[None],
                    np.arange(len(draw.generated), dtype=np.int64)[None],
                    **(
                        {}
                        if selected_holdout is None
                        else {
                            "holdout_draws": np.arange(len(selected_holdout), dtype=np.int64)[None]
                        }
                    ),
                )[0]
            )
            for metric in group:
                execution[metric.name] = context.execution_info_for(metric)
    counts = (
        cast("dict[str, int]", next(iter(outcomes.values())).diagnostics["usable_rows"])
        if outcomes
        else {
            "training": len(selected_training),
            "generated": len(selected_generated),
        }
    )
    for metric in metrics:
        if not unavailable[metric.name]:
            continue
        outcomes[metric.name] = MemorizationOutcome(
            dict.fromkeys(metric.stat_names, float("nan")),
            {"usable_rows": dict(counts)},
            {name: ["NOT_APPLICABLE"] for name in metric.stat_names},
        )
        execution[metric.name] = {"implementation": "none", "device": "cpu", "dtype": metric.dtype}
    for metric in metrics:
        usage = domain.row_usage()
        assert usage is not None
        if metric.name != "dcr":
            usage["pools"].pop("test", None)
        outcomes[metric.name].diagnostics["row_usage"] = usage
    metadata: MemorizationMetadata = {
        "n_training": domain.counts["train"]["input"],
        "n_generated": domain.counts["generated"]["input"],
        "n_eligible_training": draw.n_training,
        "n_eligible_generated": draw.n_generated,
        "max_samples": draw.max_samples,
        "n_selected_training": len(draw.training),
        "n_selected_generated": len(draw.generated),
        "n_retained_training": counts["training"],
        "n_retained_generated": counts["generated"],
        "seed": options["seed"],
        "schema": schema.to_dict(),
        "schema_inferred": False,
        "reference": "selected_training",
        "sampling": "independent_uniform_without_replacement",
    }
    if any(metric.name == "dcr" for metric in metrics):
        assert dcr is not None
        metadata.update(
            baseline="holdout_to_selected_training",
            n_holdout=None if "test" not in domain.counts else domain.counts["test"]["input"],
            n_eligible_holdout=None if holdout is None else len(holdout),
            max_holdout_samples=dcr.max_holdout_samples,
            n_selected_holdout=None if holdout_positions is None else len(holdout_positions),
            n_retained_holdout=None
            if selected_holdout is None
            else (
                int(cast("HostTable", columns.holdout).complete.sum())
                if columns is not None
                else len(selected_holdout)
            ),
        )
    return MemorizationExecution(outcomes, execution, metadata)


def compute_point(
    metric: _MemorizationMetric,
    training: pd.DataFrame,
    generated: pd.DataFrame,
    schema: Schema,
    *,
    rng: np.random.Generator | None,
    holdout: pd.DataFrame | None = None,
) -> dict[str, float]:
    """Apply a metric's row budget and return only its requested point statistics."""
    seed = 42 if rng is None else int(rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64))
    options = resolve_options(
        seed=seed,
        device=metric.device,
        dtype=metric.dtype,
    )
    return (
        run_memorization(training, generated, schema, [metric], options, holdout=holdout)
        .outcomes[metric.name]
        .values
    )
