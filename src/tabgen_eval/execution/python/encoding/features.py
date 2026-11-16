"""Metric-internal feature space: min-max-scaled numerics + one-hot categoricals.

PRDC fits this space on its real reference draw; memorization fits it on the
generator training pool. Each fitted transform is applied consistently to its
query pools without learning from generated or holdout observations. This always-on metric encoding does not repair input values or change declared roles.

Numerics are min-max scaled using the reference's per-column min/max::

        x_scaled = (x − min_ref) / (max_ref − min_ref)        (constant column → divisor 1.0)

so the reference's real values land in ``[0, 1]`` (generated values map with the *same* transform
and may fall outside ``[0, 1]`` — that is fine and intended). Min/max come from the
**reference only**, never the generated set: the real distribution defines the space, and one fixed
transform is applied to both sides so distances are comparable. Subtracting the min is a
per-column translation, which does not change Euclidean distances — so this is
distance-equivalent to plain range scaling; the ``[0, 1]`` framing is for readability/sanity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, NamedTuple, overload

import numpy as np
import pandas as pd
from numpy.typing import DTypeLike, NDArray
from sklearn.preprocessing import OneHotEncoder

from tabgen_eval._typing import BoolArray, FloatArray, IntArray
from tabgen_eval.preprocessing.categories import MISSING_CATEGORY, CategoryVocabulary
from tabgen_eval.preprocessing.columns import numeric_values
from tabgen_eval.schema import Schema


class StructuredFeatures(NamedTuple):
    """The same feature space as :meth:`FeatureEncoder.transform`, *without* expanding one-hot.

    ``numeric`` is the dense range-scaled numeric block (``(n, p_num)``) in the requested
    floating-point dtype. ``codes`` is
    an integer label matrix (``(n, p_cat)``): each entry is the category's index in the fitted
    reference vocabulary, or ``-1`` for an out-of-vocabulary value (the one-hot
    ``handle_unknown="ignore"`` all-zero row). ``oov`` is the boolean ``codes < 0`` mask, kept
    precomputed because the squared-distance formula needs it. The squared Euclidean distance
    between two rows is recovered exactly from these blocks (see ``execution/python/encoding/distances.py``),
    never materialising the ``sum(levels)``-wide one-hot columns.
    """

    # Explicit NumPy dtype casts stay flexible; distance kernels promote to FP64.
    numeric: NDArray[np.generic]
    codes: IntArray
    oov: BoolArray

    @property
    def n_rows(self) -> int:
        """Return the number of encoded rows, including repeated source records."""
        return self.numeric.shape[0]

    def rows(self, start: int, end: int) -> StructuredFeatures:
        """The row slice ``[start:end)`` as a :class:`StructuredFeatures` view."""
        return StructuredFeatures(
            self.numeric[start:end], self.codes[start:end], self.oov[start:end]
        )


@dataclass
class FeatureEncoder:
    """Fitted encoder: reference per-column min + range, and a one-hot encoder."""

    numeric: list[str]
    categorical: list[str]
    num_min: FloatArray
    num_ranges: FloatArray
    onehot: OneHotEncoder | None
    vocabularies: list[CategoryVocabulary] = field(default_factory=list)

    @property
    def onehot_dim(self) -> int:
        """Width of the dense one-hot feature space (``p_num`` + total category levels)."""
        return len(self.numeric) + sum(len(vocabulary) for vocabulary in self.vocabularies)

    def _scaled_numeric(self, df: pd.DataFrame) -> FloatArray:
        """Apply fitted ranges to declared numeric values, preserving actual missing cells."""
        if not self.numeric:
            return np.zeros((len(df), 0))
        num = np.column_stack([numeric_values(df[column]) for column in self.numeric]).astype(
            float, copy=False
        )
        return (num - self.num_min) / self.num_ranges

    def _categorical_codes(self, df: pd.DataFrame) -> IntArray:
        """Apply each reference-fitted vocabulary; unknown categories receive -1."""
        codes = np.empty((len(df), len(self.categorical)), dtype=np.int64)
        for j, column in enumerate(self.categorical):
            codes[:, j] = self.vocabularies[j].transform(_category_values(df[column]))
        codes[codes < 0] = -1
        return codes

    @overload
    def transform(
        self,
        df: pd.DataFrame,
        *,
        dtype: Literal["float32"] | type[np.float32] | np.dtype[np.float32] = np.float32,
    ) -> NDArray[np.float32]: ...

    @overload
    def transform(
        self,
        df: pd.DataFrame,
        *,
        dtype: Literal["float64"] | type[np.float64] | np.dtype[np.float64],
    ) -> NDArray[np.float64]: ...

    @overload
    def transform(self, df: pd.DataFrame, *, dtype: DTypeLike) -> NDArray[np.generic]: ...

    def transform(self, df: pd.DataFrame, *, dtype: DTypeLike = np.float32) -> NDArray[np.generic]:
        """Encode *df* as dense scaled numeric/one-hot coordinates in *dtype*.

        Scaling uses float64 reference parameters before the requested output cast.
        Actual missing values remain NaN rather than representing an observed zero.
        """
        num_scaled = self._scaled_numeric(df)
        if self.categorical and self.onehot is not None:
            cat_enc = self.onehot.transform(self._categorical_codes(df))
        else:
            cat_enc = np.zeros((len(df), 0))
        return np.concatenate([num_scaled, cat_enc], axis=1).astype(dtype)

    def transform_structured(
        self, df: pd.DataFrame, *, dtype: DTypeLike = np.float32
    ) -> StructuredFeatures:
        """Encode numeric coordinates in *dtype* and categories as compact integer codes.

        The numeric transform matches :meth:`transform`, including its handling of
        actual missing values. Category codes and unknown-level geometry are independent
        of the numeric output precision.
        """
        num_scaled = self._scaled_numeric(df).astype(dtype)
        if self.categorical:
            codes = self._categorical_codes(df)
        else:
            codes = np.zeros((len(df), 0), dtype=np.int64)
        return StructuredFeatures(num_scaled, codes, codes < 0)


def fit_encoder(df: pd.DataFrame, schema: Schema) -> FeatureEncoder:
    """Fit a :class:`FeatureEncoder` on *df* (computes ranges + one-hot vocabulary)."""
    numeric = [c for c in schema.numeric if c in df.columns]
    categorical = [c for c in schema.categorical if c in df.columns]

    if numeric:
        num = np.column_stack([numeric_values(df[column]) for column in numeric]).astype(
            float, copy=False
        )
        mins = num.min(axis=0)  # reference min/max only — never the candidate
        ranges = num.max(axis=0) - mins
        ranges[ranges == 0] = 1.0  # Constant reference columns retain unscaled query differences.
    else:
        mins = np.array([])
        ranges = np.array([])

    onehot = None
    vocabularies = [CategoryVocabulary.fit(_category_values(df[col])) for col in categorical]
    encoder = FeatureEncoder(numeric, categorical, mins, ranges, onehot, vocabularies)
    if categorical:
        onehot = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        onehot.fit(encoder._categorical_codes(df))
        encoder.onehot = onehot
    return encoder


def _category_values(series: pd.Series) -> pd.Series | NDArray[np.object_]:
    """Use an observed, collision-free category for residual missing encoder inputs."""
    missing = series.isna().to_numpy()
    if not missing.any():
        return series
    values = series.to_numpy(dtype=object, copy=True)
    values[missing] = MISSING_CATEGORY
    return values


def build_feature_pair(
    reference: pd.DataFrame, candidate: pd.DataFrame, schema: Schema
) -> tuple[FloatArray, FloatArray]:
    """Fit the encoder on *reference*, transform both; return ``(ref_features, cand_features)``."""
    enc = fit_encoder(reference, schema)
    return enc.transform(reference), enc.transform(candidate)
