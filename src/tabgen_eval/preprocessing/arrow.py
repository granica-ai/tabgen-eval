"""Optional Arrow packing with compact, exact typed category dictionaries.

This is storage preparation only. It extracts declared numeric values and encodes
category identities without repairing input, changing roles, or fitting metric
geometry. Unsupported representations use the shared typed factorizer; absent
Arrow returns control to ordinary pandas packing.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType

import numpy as np
import pandas as pd

from tabgen_eval._typing import BoolArray, FloatArray, IntegerArray
from tabgen_eval.preprocessing.categories import factorize_categories
from tabgen_eval.preprocessing.columns import HostTable, PreparedColumns, numeric_values
from tabgen_eval.schema import Schema


@dataclass
class _Category:
    """A compact dictionary, nonnegative row codes, and true missing-cell mask."""

    codes: IntegerArray
    labels: pd.Series
    missing: BoolArray


def _supported(series: pd.Series) -> bool:
    """Whether Arrow dictionary encoding preserves these scalar identities."""
    if type(series) is not pd.Series:
        return False
    dtype = series.dtype
    if isinstance(dtype, np.dtype) and dtype.isnative and dtype.kind in "biuf":
        return dtype != np.dtype("float16")
    if isinstance(dtype, pd.StringDtype):
        return True
    if pd.api.types.is_object_dtype(dtype):
        return pd.api.types.infer_dtype(series, skipna=True) in {"string", "unicode", "empty"}
    return False


def _category(series: pd.Series, pa: ModuleType, pc: ModuleType) -> _Category:
    """Pack one pool's observed labels; actual missing cells remain missing."""
    labels = None
    if _supported(series):
        try:
            source = pa.array(series, from_pandas=True)
            if isinstance(source, pa.ChunkedArray):
                source = source.combine_chunks()
            encoded = pc.dictionary_encode(source, null_encoding="encode")
            codes = encoded.indices.to_numpy()
            labels = pd.Series(encoded.dictionary.to_pylist(), dtype=series.dtype)
        except (
            pa.ArrowInvalid,
            pa.ArrowNotImplementedError,
            pa.ArrowTypeError,
            UnicodeEncodeError,
        ):
            labels = None
    missing = series.isna().to_numpy() if labels is None else labels.isna().to_numpy()[codes]
    if labels is None:
        (codes,), vocabulary = factorize_categories(series)
        values = vocabulary.values
        if (codes < 0).any():
            codes[codes < 0] = len(values)
            values = values + (None,)
        # Growing a Series can infer Arrow string storage again, even when its
        # original dtype was object. Keep fallback labels in explicit object storage.
        labels = pd.Series(values, dtype=object)
    return _Category(codes, labels, missing)


def _pool_columns(
    frame: pd.DataFrame, schema: Schema, pa: ModuleType, pc: ModuleType
) -> tuple[FloatArray, list[_Category], BoolArray]:
    """Own feature-major numerics, category dictionaries, and feature completeness."""
    numeric = (
        np.stack([numeric_values(frame[column]) for column in schema.numeric])
        if schema.numeric
        else np.empty((0, len(frame)), dtype=np.float64)
    )
    categories = [_category(frame[column], pa, pc) for column in schema.categorical]
    complete = ~np.isnan(numeric).any(axis=0)
    for category in categories:
        complete &= ~category.missing
    return numeric, categories, complete


def _try_prepare(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    schema: Schema,
    *,
    retain_labels: bool = False,
    holdout: pd.DataFrame | None = None,
) -> PreparedColumns | None:
    """Pack independent source pools, returning ``None`` only for route ineligibility.

    Exact DataFrames are required to avoid bypassing subclass behavior. A broken
    installed Arrow dependency propagates its import error. Categories share typed
    identities across pools, including optional holdout storage, but no fitted
    classifier vocabularies, bins, ranges or imputation state.
    """
    if (
        type(reference) is not pd.DataFrame
        or type(candidate) is not pd.DataFrame
        or (holdout is not None and type(holdout) is not pd.DataFrame)
    ):
        return None
    try:
        import pyarrow as pa
        import pyarrow.compute as pc
    except ModuleNotFoundError as exc:
        if exc.name != "pyarrow":
            raise
        return None

    shared = reference is candidate
    rn, rcat, rcomplete = _pool_columns(reference, schema, pa, pc)
    sn, scat, scomplete = (
        (rn, rcat, rcomplete) if shared else _pool_columns(candidate, schema, pa, pc)
    )
    if holdout is not None:
        hn, hcat, hcomplete = _pool_columns(holdout, schema, pa, pc)
    else:
        hcat = ()
    rc, sc, hc, cards, reference_cards = [], [], [], [], []
    category_values = [] if retain_labels else None
    mismatch = False
    for i, (real_category, candidate_category) in enumerate(zip(rcat, scat, strict=True)):
        if holdout is not None:
            (rmap, smap, hmap), vocabulary = factorize_categories(
                real_category.labels, candidate_category.labels, hcat[i].labels
            )
            hc.append(hmap[hcat[i].codes])
        elif shared:
            (rmap,), vocabulary = factorize_categories(real_category.labels)
            smap = rmap
        else:
            (rmap, smap), vocabulary = factorize_categories(
                real_category.labels, candidate_category.labels
            )
        if category_values is not None:
            category_values.append(vocabulary.values)
        a = rmap[real_category.codes]
        b = a if shared else smap[candidate_category.codes]
        rc.append(a)
        sc.append(b)
        cards.append(len(vocabulary))
        present_r = np.bincount(a[a >= 0], minlength=len(vocabulary)) > 0
        present_s = present_r if shared else np.bincount(b[b >= 0], minlength=len(vocabulary)) > 0
        reference_cards.append(int(present_r.sum()))
        mismatch |= not np.array_equal(present_r, present_s)
    rarray = np.stack(rc) if rc else np.empty((0, len(reference)), dtype=np.int64)
    sarray = (
        rarray if shared else np.stack(sc) if sc else np.empty((0, len(candidate)), dtype=np.int64)
    )
    real = HostTable(rn, rarray, rcomplete)
    return PreparedColumns(
        real,
        real if shared else HostTable(sn, sarray, scomplete),
        tuple(schema.numeric),
        tuple(schema.categorical),
        tuple(cards),
        bool(mismatch),
        tuple(reference_cards),
        tuple(category_values) if category_values is not None else None,
        None
        if holdout is None
        else HostTable(
            hn, np.stack(hc) if hc else np.empty((0, len(holdout)), dtype=np.int64), hcomplete
        ),
    )


def try_prepare(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    schema: Schema,
    *,
    retain_labels: bool = False,
) -> PreparedColumns | None:
    """Pack two pools into ``PreparedColumns``, or return ``None`` without Arrow."""
    return _try_prepare(reference, candidate, schema, retain_labels=retain_labels)


def try_prepare_memorization(
    training: pd.DataFrame, generated: pd.DataFrame, holdout: pd.DataFrame, schema: Schema
) -> PreparedColumns | None:
    """Pack three independent pools without concatenation or input conversions."""
    return _try_prepare(training, generated, schema, holdout=holdout)


def try_prepare_single(
    source: pd.DataFrame, schema: Schema, *, retain_labels: bool = False
) -> PreparedColumns | None:
    """Own one encoded source pool shared by two sampled real comparison roles."""
    return try_prepare(source, source, schema, retain_labels=retain_labels)
