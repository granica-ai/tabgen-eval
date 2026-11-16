"""Immutable state exchanged by compiled classifier preprocessing backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

POLICY_VERSION = "c2st-2"


@dataclass(frozen=True)
class NumericState:
    """Training-selected source columns and fitted numeric parameters.

    ``keep`` and ``indicators`` contain source-column indices. Parameters follow
    ``keep`` order; indicators are appended after the retained value columns.
    Tensor storage stays on its backend device and is read-only to consumers.
    """

    keep: Tensor
    indicators: Tensor
    impute: Tensor
    center: Tensor
    scale: Tensor


@dataclass(frozen=True)
class CategoryState:
    """Training-only vocabulary over previously encoded source identities.

    Each source column has a domain of cardinality + 1, with actual missing
    identity -1 occupying the first slot. ``mapping`` maps domain slots to local
    active indices, or -1 for unseen. ``feature_offsets`` places local indices
    into the full linear indicator space; ``active`` lists fitted domain slots.
    """

    mapping: Tensor
    domain_offsets: Tensor
    feature_offsets: Tensor
    active: Tensor


@dataclass(frozen=True)
class PreparedFold:
    """Detector-ready fold inputs plus their training-fitted interpretation."""

    train: Tensor
    test: Tensor
    training_labels: Tensor
    scoring_labels: Tensor
    numeric: NumericState
    categorical: CategoryState
    feature_names: tuple[str, ...]
    feature_types: tuple[str, ...]
    categorical_cardinalities: tuple[int, ...]
    diagnostics: dict[str, object]
