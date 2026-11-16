"""Small metadata decoding for transparent, training-fitted preprocessing."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from .categories import category_display
from .contracts import POLICY_VERSION

if TYPE_CHECKING:
    from torch import Tensor


def _rows(tensor: Tensor) -> list[int] | list[list[int]]:
    """Copy the compiled operator's integer vector/matrix metadata to the host."""
    return tensor.detach().cpu().tolist()


def fold_metadata(
    metadata: Sequence[Tensor],
    numeric_names: Sequence[str],
    categorical_names: Sequence[str],
    category_values: tuple[tuple[object, ...], ...] | None = None,
) -> dict[str, object]:
    """Decode compiled fold metadata; numerical data are never rescanned here.

    Six tensors describe kept/indicator columns, active category triples,
    numeric/categorical missing counts (four class/partition roles), and unseen
    held-out counts (two classes). Only these small records cross to the host.
    """
    if not metadata:
        return dict(policy_version=POLICY_VERSION, status="unavailable")
    keep, indicators, active, numeric_missing, categorical_missing, unseen = map(_rows, metadata)
    retained = set(keep)
    categories = []
    for column, identity, feature in active:
        label = "<missing>" if identity < 0 else identity
        if identity >= 0 and category_values is not None:
            label = category_display(category_values[column][identity])
        categories.append(
            dict(
                column=categorical_names[column], identity=identity, value=label, indicator=feature
            )
        )
    return dict(
        policy_version=POLICY_VERSION,
        policy="linear",
        numerical_scaling="guarded_standard",
        categorical_scaling="none",
        categorical_encoding="indicators",
        numeric_features=[numeric_names[index] for index in keep],
        missing_indicators=[numeric_names[index] for index in indicators],
        dropped_features=[
            dict(column=name, reason="no_finite_training_values")
            for index, name in enumerate(numeric_names)
            if index not in retained
        ],
        category_mapping=categories,
        count_roles=["real_train", "generated_train", "real_holdout", "generated_holdout"],
        numeric_missing=dict(zip(numeric_names, numeric_missing, strict=True)),
        categorical_missing=dict(zip(categorical_names, categorical_missing, strict=True)),
        unseen_holdout=dict(zip(categorical_names, unseen, strict=True)),
    )
