"""Resolve metric-specific eligible source pools before selecting row draws.

Eligibility concerns whole observations only. Marginal, pairwise and conditional
fidelity keep their existing observed-value rules inside each sampled comparison.
No values are repaired and duplicate observations remain distinct source rows.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import copy
from dataclasses import dataclass
from typing import Literal, TypedDict

import numpy as np
import pandas as pd

from tabgen_eval._typing import BoolArray, SeedLike, UncertaintyMode
from tabgen_eval.metrics.base import Metric
from tabgen_eval.schema import Schema

from .sampling import FixedDraw, fixed_draw

EligibilityPolicy = Literal["complete_case", "observed_target"]
StorageKey = tuple[tuple[str, int], ...]


class RowCounts(TypedDict):
    """Input and eligible record counts for one unmodified source pool."""

    input: int
    retained: int
    excluded: int
    excluded_percent: float | None


class RowUsage(TypedDict):
    """Metric-specific exclusions recorded before any point or repeated draw."""

    policy: EligibilityPolicy
    columns: list[str]
    pools: dict[str, RowCounts]
    n_res: int | None


def metric_input_schema(schema: Schema, metrics: Sequence[Metric]) -> Schema:
    """Validate a separate target only when a selected computation consumes it."""
    if schema.target is None or any(metric.needs_target for metric in metrics):
        return schema
    return Schema(numeric=schema.numeric, categorical=schema.categorical)


def metric_eligibility(
    metric: Metric, schema: Schema
) -> tuple[EligibilityPolicy | None, tuple[str, ...]]:
    """Return the shared source-domain key for an internal computation."""
    if metric.name in {"prdc", "concentration", "dcr", "near_match_rate", "replication_ratio"}:
        return "complete_case", tuple(schema.numeric) + tuple(schema.categorical)
    if metric.name == "mle":
        if schema.target is None:
            raise ValueError("Utility requires an explicit Schema.target")
        return "observed_target", (schema.target,)
    return None, ()


@dataclass(frozen=True)
class SourceDomain:
    """Eligible frames and factual exclusions from the original source pools."""

    policy: EligibilityPolicy | None
    columns: tuple[str, ...]
    pools: dict[str, pd.DataFrame]
    counts: dict[str, RowCounts]

    @property
    def storage_key(self) -> StorageKey:
        """Share unchanged source objects even when their exclusion policies differ."""
        return tuple((role, id(frame)) for role, frame in self.pools.items())

    @classmethod
    def prepare(
        cls,
        pools: Mapping[str, pd.DataFrame],
        *,
        policy: EligibilityPolicy | None = None,
        columns: Sequence[str] = (),
    ) -> SourceDomain:
        """Exclude missing records using only the computation's declared columns."""
        columns = tuple(columns)
        frames: dict[str, pd.DataFrame] = {}
        counts: dict[str, RowCounts] = {}
        for role, source in pools.items():
            if policy is None or not columns:
                retained = source
            else:
                features = source[list(columns)]
                keep: BoolArray = ~features.isna().any(axis=1).to_numpy()
                retained = source if keep.all() else source.iloc[np.flatnonzero(keep)]
            total, usable = len(source), len(retained)
            excluded = total - usable
            frames[role] = retained
            counts[role] = {
                "input": total,
                "retained": usable,
                "excluded": excluded,
                "excluded_percent": 100.0 * excluded / total if total else None,
            }
        return cls(policy, columns, frames, counts)

    def row_usage(self, n_res: int | None = None) -> RowUsage | None:
        """Describe pre-sampling exclusions once for every involved source pool."""
        if self.policy is None:
            return None
        return {
            "policy": self.policy,
            "columns": list(self.columns),
            "pools": {role: copy(counts) for role, counts in self.counts.items()},
            "n_res": n_res,
        }


def prepare_metric_domains(
    metrics: Sequence[Metric], pools: Mapping[str, pd.DataFrame], schema: Schema
) -> list[tuple[SourceDomain, list[Metric]]]:
    """Resolve each eligibility policy once, retaining computation order within it."""
    groups: dict[tuple[EligibilityPolicy | None, tuple[str, ...]], list[Metric]] = {}
    for metric in metrics:
        groups.setdefault(metric_eligibility(metric, schema), []).append(metric)
    return [
        (SourceDomain.prepare(pools, policy=policy, columns=columns), members)
        for (policy, columns), members in groups.items()
    ]


def eligible_fixed_draw(
    n_reference: int,
    n_generated: int,
    *,
    seed: SeedLike,
    uncertainty: UncertaintyMode = "none",
) -> FixedDraw:
    """Select mode-specific draws or represent exhausted eligible pools explicitly."""
    if n_reference >= 2 and n_generated:
        return fixed_draw(n_reference, n_generated, seed=seed, uncertainty=uncertainty)
    empty = np.empty(0, dtype=np.int64)
    return FixedDraw(empty, empty, empty, n_reference, n_generated, 0)
