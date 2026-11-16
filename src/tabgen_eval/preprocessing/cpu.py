"""Dispatch CPU preprocessing to the shared compiled operator library."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor
    from torch._ops import _OpNamespace


def _ops() -> _OpNamespace:
    from tabgen_eval.execution.torch.cpu.extension import library

    return library()


def fit_numeric(
    reference: Tensor, candidate: Tensor, linear: bool
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    return tuple(_ops().preprocessing_fit_numeric(reference, candidate, linear))


def apply_numeric(
    values: Tensor,
    keep: Tensor,
    indicators: Tensor,
    impute: Tensor,
    center: Tensor,
    scale: Tensor,
    linear: bool,
) -> Tensor:
    return _ops().preprocessing_apply_numeric(
        values, keep, indicators, impute, center, scale, linear
    )


def fit_categories(
    reference: Tensor, candidate: Tensor, cardinalities: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return tuple(_ops().preprocessing_fit_categories(reference, candidate, cardinalities))


def map_categories(
    codes: Tensor,
    mapping: Tensor,
    domain_offsets: Tensor,
    feature_offsets: Tensor,
    linear: bool,
) -> Tensor:
    return _ops().preprocessing_map_categories(
        codes, mapping, domain_offsets, feature_offsets, linear
    )


def assemble(numbers: Tensor, codes: Tensor, feature_count: int, linear: bool) -> Tensor:
    return _ops().preprocessing_assemble(numbers, codes, feature_count, linear)


def count_missing(values: Tensor) -> Tensor:
    return _ops().preprocessing_count_missing(values)
