"""Fit training-only preprocessing and assemble detector-specific fold inputs."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

from numpy.typing import ArrayLike

from tabgen_eval.models import FitUnavailable

from .contracts import CategoryState, NumericState, PreparedFold
from .reporting import fold_metadata

if TYPE_CHECKING:
    from tabgen_eval.execution.torch.tensors import TensorTable


def fit_fold(
    reference: TensorTable,
    candidate: TensorTable,
    reference_folds: ArrayLike,
    candidate_folds: ArrayLike,
    fold: int,
    *,
    cardinalities: Sequence[int],
    numeric_names: Sequence[str],
    categorical_names: Sequence[str],
    category_values: tuple[tuple[object, ...], ...] | None = None,
    policy: Literal["linear", "xgboost"] = "xgboost",
    memory_limit_bytes: float = 2048 * 1024**2,
) -> PreparedFold:
    """Fit only on rows outside ``fold`` and transform its held-out rows.

    The fixed ``linear`` and ``xgboost`` policies select compiled operations at
    this boundary. Arrays are never inspected to choose classifier features
    using held-out values. Output features contain no model intercept column.
    """
    import torch

    if policy not in {"linear", "xgboost"}:
        raise ValueError("Unknown classifier preprocessing policy")
    linear = policy == "linear"
    backend = __import__(
        f"tabgen_eval.preprocessing.{reference.numeric.device.type}", fromlist=["fit_numeric"]
    )
    start = time.perf_counter()
    device = reference.numeric.device
    rf, sf = [
        torch.as_tensor(value, device=device, dtype=torch.int64)
        for value in (reference_folds, candidate_folds)
    ]
    count = len(rf) + len(sf)
    pn, pc = len(numeric_names), len(categorical_names)
    item = reference.numeric.element_size()
    # Include gathers, fitted outputs, masks, counts and potential dense linear
    # expansion before allocating any of those buffers. Source pools are owned
    # by the evaluation context and budgeted independently.
    max_width = 2 * pn + sum(cardinalities) + pc if linear else pn + pc
    estimated = count * (item * (4 * pn + 2 * max_width + 4) + 40 * pc + 32)
    blocks = (count + 1023) // 1024
    estimated += (sum(cardinalities) + pc) * 64 + pn * (32 * blocks + 128) + 65536
    if estimated > memory_limit_bytes:
        raise FitUnavailable("preprocessing_memory_limit")
    indices = [
        torch.nonzero(mask).flatten() for mask in (rf != fold, sf != fold, rf == fold, sf == fold)
    ]
    if any(index.numel() == 0 for index in indices):
        raise FitUnavailable("empty_classifier_partition")
    tables = (reference, candidate, reference, candidate)
    numbers = [
        table.numeric.index_select(0, index) for table, index in zip(tables, indices, strict=True)
    ]
    codes = [
        table.categorical.index_select(0, index)
        for table, index in zip(tables, indices, strict=True)
    ]
    numeric = NumericState(*backend.fit_numeric(numbers[0], numbers[1], linear))
    domain = torch.tensor(cardinalities, device=device, dtype=torch.int64)
    categorical = CategoryState(*backend.fit_categories(codes[0], codes[1], domain))
    if len(numeric.keep) == 0 and len(categorical.active) == 0:
        from .contracts import POLICY_VERSION

        raise FitUnavailable(
            "no_usable_classifier_features",
            preprocessing=dict(
                policy_version=POLICY_VERSION,
                dropped_features=[
                    dict(column=name, reason="no_finite_training_values") for name in numeric_names
                ],
            ),
        )
    transformed = [
        backend.apply_numeric(
            value,
            numeric.keep,
            numeric.indicators,
            numeric.impute,
            numeric.center,
            numeric.scale,
            linear,
        )
        for value in numbers
    ]
    if linear and any(not bool(torch.isfinite(value).all()) for value in transformed):
        raise FitUnavailable("nonfinite_classifier_preprocessing")
    mapped = [
        backend.map_categories(
            value,
            categorical.mapping,
            categorical.domain_offsets,
            categorical.feature_offsets,
            linear,
        )
        for value in codes
    ]
    matrices = [
        backend.assemble(value, ids, len(categorical.active), linear)
        for value, ids in zip(transformed, mapped, strict=True)
    ]
    train, test = torch.cat(matrices[:2]), torch.cat(matrices[2:])
    labels = [
        torch.cat((train.new_zeros(len(indices[offset])), train.new_ones(len(indices[offset + 1]))))
        for offset in (0, 2)
    ]
    offsets = categorical.domain_offsets
    active_columns = torch.searchsorted(offsets[1:], categorical.active, right=True)
    active_identities = categorical.active - offsets[active_columns] - 1
    active_features = (
        categorical.mapping[categorical.active] + categorical.feature_offsets[active_columns]
    )
    active = torch.stack((active_columns, active_identities, active_features), dim=1)
    numeric_missing = torch.stack([backend.count_missing(value) for value in numbers], dim=1)
    categorical_missing = torch.stack([(value == -1).sum(dim=0) for value in codes], dim=1)
    unknown = []
    for value, ids in zip(codes[2:], mapped[2:], strict=True):
        if linear:
            unknown.append(((ids < 0) & (value >= 0)).sum(dim=0))
        else:
            sizes = categorical.feature_offsets[1:] - categorical.feature_offsets[:-1]
            unknown.append(((ids == sizes) & (value >= 0)).sum(dim=0))
    metadata = [
        numeric.keep,
        numeric.indicators,
        active,
        numeric_missing,
        categorical_missing,
        torch.stack(unknown, dim=1),
    ]
    diagnostics = fold_metadata(metadata, numeric_names, categorical_names, category_values)
    keep = numeric.keep.cpu().tolist()
    indicators = numeric.indicators.cpu().tolist() if linear else []
    names = [numeric_names[index] for index in keep]
    names.extend(f"{numeric_names[index]}__missing" for index in indicators)
    sizes = tuple(
        (categorical.feature_offsets[1:] - categorical.feature_offsets[:-1]).cpu().tolist()
    )
    if linear:
        names.extend(
            f"{entry['column']}={entry['value']!r}" for entry in diagnostics["category_mapping"]
        )
        feature_types = ("q",) * len(names)
    else:
        names.extend(categorical_names)
        feature_types = ("q",) * len(keep) + ("c",) * pc
    diagnostics.update(
        policy=policy,
        numerical_scaling="guarded_standard" if linear else "none",
        categorical_encoding="indicators" if linear else "native_one_vs_rest",
        categorical_scaling="none",
        estimated_bytes=estimated,
        preparation_seconds=time.perf_counter() - start,
    )
    return PreparedFold(
        train, test, *labels, numeric, categorical, tuple(names), feature_types, sizes, diagnostics
    )
