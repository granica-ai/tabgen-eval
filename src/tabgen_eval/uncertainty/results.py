"""Raw real-only scores awaiting point or uncertainty report assembly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

from tabgen_eval._typing import FloatArray, SeedEntropy
from tabgen_eval.execution.runtime.domains import RowUsage
from tabgen_eval.schema.schema import SchemaDeclaration


class RealSampling(TypedDict):
    """Per-metric source resolution for disjoint real-only draws."""

    n_source: int
    n_res: int
    policy: str
    replacement: bool
    real_roles_disjoint: bool


class _OptionalReferenceMetadata(TypedDict, total=False):
    confidence_interval: float
    schema_inferred: bool
    per_metric_sampling: dict[str, RealSampling]


class RealReferenceMetadata(_OptionalReferenceMetadata):
    """Source and schema facts needed to assemble a real-only report."""

    n_source: int
    n_res: int | None
    n_replicates: int
    seed: SeedEntropy
    schema: SchemaDeclaration


@dataclass(frozen=True)
class RealMetricScores:
    """Ordered numerical samples, execution choices, and source exclusions.

    Values are raw arrays, including unavailable entries. The report layer
    chooses a scalar for point evaluation or constructs descriptive empirical
    distributions when uncertainty was requested.
    """

    name: str
    values: dict[str, FloatArray]
    execution: dict[str, object]
    options: dict[str, object]
    diagnostics: tuple[dict[str, object], ...]
    sampling: RealSampling
    row_usage: RowUsage | None = None


@dataclass(frozen=True)
class RealReferenceResult:
    """Real-only raw comparison outcomes and shared sampling provenance."""

    metrics: dict[str, RealMetricScores]
    metadata: RealReferenceMetadata
