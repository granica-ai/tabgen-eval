"""Evaluation-local typed sources and symmetric observation precision."""

from __future__ import annotations

import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal, TypedDict

import numpy as np
import pandas as pd

from tabgen_eval._typing import Precision
from tabgen_eval.schema import Schema

from .categories import MISSING_CATEGORY, category_key
from .columns import HostTable, PreparedColumns
from .contracts import POLICY_VERSION

if TYPE_CHECKING:
    from tabgen_eval.execution.torch.tensors import TensorInputs, TensorTable


class ObservationPrecision(TypedDict):
    """Per-feature precision chosen symmetrically from the compared sources."""

    policy_version: str
    feature_dtype: Precision
    accumulation_dtype: Literal["float64"]
    input_dtypes: dict[str, list[str]]
    observation_bits: dict[str, int]


def observation_precision(
    reference: pd.DataFrame, candidate: pd.DataFrame, names: Sequence[str], dtype: Precision
) -> ObservationPrecision:
    """Choose common per-column float widths from current input dtypes only.

    Integer columns and parsed object columns do not imply a lower floating
    precision. The requested execution precision caps all observation widths.
    """
    cap = np.dtype(dtype).itemsize * 8
    result = []
    source_types = []
    for name in names:
        pair, widths = [], [cap]
        for frame in (reference, candidate):
            source = frame[name].dtype
            pair.append(str(source))
            try:
                kind = np.dtype(getattr(source, "numpy_dtype", source))
            except TypeError:
                continue
            if kind.kind == "f":
                widths.append(kind.itemsize * 8)
        result.append(min(widths))
        source_types.append(pair)
    return dict(
        policy_version=POLICY_VERSION,
        feature_dtype=dtype,
        accumulation_dtype="float64",
        input_dtypes=dict(zip(names, source_types, strict=True)),
        observation_bits=dict(zip(names, result, strict=True)),
    )


@dataclass(frozen=True)
class PreparedSources:
    """Read-only source tensors shared by generated and real-real comparisons."""

    reference: TensorTable
    candidate: TensorTable
    numeric_names: tuple[str, ...]
    categorical_names: tuple[str, ...]
    cardinalities: tuple[int, ...]
    category_values: tuple[tuple[object, ...], ...]
    precision: ObservationPrecision
    preparation_seconds: float
    category_mismatch: bool
    reference_cardinalities: tuple[int, ...] | None
    tensors: TensorInputs


def ensure_category_labels(
    columns: PreparedColumns, reference: pd.DataFrame, candidate: pd.DataFrame
) -> PreparedColumns:
    """Recover missing dictionary metadata without re-encoding source columns.

    Advanced callers may supply unlabeled prepared storage. First occurrence
    positions come from its existing codes; only unique labels visit Python.
    Actual missing cells use the existing missing identity; values are not cleaned.
    """
    if columns.category_values is not None:
        return columns
    dictionaries = []
    for index, name in enumerate(columns.categorical_names):
        labels = [None] * columns.cardinalities[index]
        filled = set()
        for frame, table in ((reference, columns.reference), (candidate, columns.candidate)):
            identities, positions = np.unique(table.categorical[index], return_index=True)
            for identity, position in zip(identities, positions, strict=True):
                if identity < 0 or identity in filled:
                    continue
                value = frame[name].iloc[int(position)]
                missing = pd.isna(value)
                labels[int(identity)] = (
                    MISSING_CATEGORY if isinstance(missing, (bool, np.bool_)) and missing else value
                )
                filled.add(int(identity))
        dictionaries.append(tuple(labels))
    return replace(columns, category_values=tuple(dictionaries))


def matching_is_noop(
    reference: pd.DataFrame, candidate: pd.DataFrame, names: Sequence[str], dtype: Precision
) -> bool:
    """Whether common precision changes nothing beyond ordinary dtype transfer."""
    precision = observation_precision(reference, candidate, names, dtype)
    cap = np.dtype(dtype).itemsize * 8
    for name, common in precision["observation_bits"].items():
        for frame in (reference, candidate):
            source = frame[name].dtype
            try:
                kind = np.dtype(getattr(source, "numpy_dtype", source))
                bits = kind.itemsize * 8 if kind.kind == "f" else cap
            except TypeError:
                bits = cap
            if common < min(cap, bits):
                return False
    return True


def matched_columns(
    columns: PreparedColumns,
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    dtype: Precision,
) -> tuple[PreparedColumns, ObservationPrecision]:
    """Quantize and repack host numerics with one compiled pass per source.

    Category identities are reused unchanged. Precision is chosen once from the
    real/generated pair and therefore also applies to its real-real reference.
    """
    import torch

    from tabgen_eval.execution.torch.cpu.extension import library

    if columns.holdout is not None:
        raise ValueError("Classifier preprocessing accepts two pools, not an independent holdout")
    from tabgen_eval.preprocessing.columns import HostTable

    precision = observation_precision(reference, candidate, columns.numeric_names, dtype)
    bits = torch.tensor(list(precision["observation_bits"].values()), dtype=torch.int64)
    output_dtype = getattr(torch, dtype)

    # Canonical identity order prevents held-out first appearances from changing
    # the order of learned category features. This sorts dictionary metadata,
    # never source rows, and fits no frequencies or classifier parameters.
    dictionaries = columns.category_values
    reordered = dictionaries
    if dictionaries is not None:
        domains, features, mapping, reordered = [0], [0], [], []
        for labels in dictionaries:
            order = sorted(range(len(labels)), key=lambda index: repr(category_key(labels[index])))
            ranks = [0] * len(order)
            for rank, index in enumerate(order):
                ranks[index] = rank
            mapping.extend([-1, *ranks])
            domains.append(domains[-1] + len(labels) + 1)
            features.append(features[-1] + len(labels))
            reordered.append(tuple(labels[index] for index in order))
        domains = torch.tensor(domains, dtype=torch.int64)
        features = torch.tensor(features, dtype=torch.int64)
        mapping = torch.tensor(mapping, dtype=torch.int64)
        reordered = tuple(reordered)

    def quantize(table: HostTable) -> HostTable:
        # The source NumPy arrays are deliberately immutable. Native quantize
        # only reads its strided input and always returns fresh owned storage.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
            view = torch.from_numpy(table.numeric.T)
        result = library().preprocessing_quantize(view, bits, output_dtype)
        categories = table.categorical
        if dictionaries is not None and len(dictionaries):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
                codes = torch.from_numpy(table.categorical.T).contiguous()
            remapped = library().preprocessing_map_categories(
                codes, mapping, domains, features, True
            )
            remapped = torch.where(remapped >= 0, remapped - features[:-1], -1)
            categories = remapped.numpy().T
        return HostTable(result.numpy().T, categories, table.complete)

    real = quantize(columns.reference)
    fake = real if columns.reference is columns.candidate else quantize(columns.candidate)
    return replace(columns, reference=real, candidate=fake, category_values=reordered), precision


def prepare_sources(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    schema: Schema,
    *,
    device: str = "cpu",
    dtype: Precision = "float64",
    columns: PreparedColumns | None = None,
) -> PreparedSources:
    """Own reusable typed tensors from validated, explicitly declared source values.

    ``columns`` optionally shares a previously prepared identity encoding. No
    imputation, scaling or classifier vocabulary is fitted at this boundary.
    """
    from tabgen_eval.preprocessing.columns import PreparedColumns

    start = time.perf_counter()
    if columns is None:
        columns = PreparedColumns.prepare(reference, candidate, schema, retain_labels=True)
    columns = ensure_category_labels(columns, reference, candidate)
    matched, precision = matched_columns(columns, reference, candidate, dtype)
    # Numeric and remapped category buffers are fresh owned outputs. Transfer
    # their ownership instead of retaining a second matched host copy on CPU.
    tensors = matched.to_tensors(device=device, dtype=dtype, copy=False)
    assert matched.category_values is not None  # ensure_category_labels supplied every dictionary.
    return PreparedSources(
        tensors.reference,
        tensors.candidate,
        matched.numeric_names,
        matched.categorical_names,
        matched.cardinalities,
        matched.category_values,
        precision,
        time.perf_counter() - start,
        matched.category_mismatch,
        matched.reference_cardinalities,
        tensors,
    )
