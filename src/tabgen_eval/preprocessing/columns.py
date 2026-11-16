"""Pack validated DataFrames into evaluation-local host columns and identities.

The pandas route runs after schema validation. It owns independent
FP32/FP64 numeric arrays, integer category codes and the complete-row mask used by
metrics that require complete cases. Arrays are feature-major ``(P, N)`` so each
source column is contiguous; tensor transfer later repacks them as ``(N, P)``.

Both source pools share typed category identities for storage and comparisons.
This is not a fitted metric vocabulary: ranges, bins, reference membership and
classifier training transforms remain draw/fold-local. No Torch import or device
probe is needed until a caller requests tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from tabgen_eval._typing import BoolArray, FloatArray, IntArray, Precision
from tabgen_eval.preprocessing.categories import factorize_categories
from tabgen_eval.schema import Schema
from tabgen_eval.schema.validate import numeric_array

if TYPE_CHECKING:
    from torch import device as TorchDevice

    from tabgen_eval.execution.torch.tensors import TensorInputs


def numeric_values(series: pd.Series) -> FloatArray:
    """Extract validated numeric values without text parsing or missing-value repair."""
    return numeric_array(series)


@dataclass(frozen=True)
class HostTable:
    """Read-only host arrays for one source pool in feature-major layout.

    Attributes
    ----------
    numeric : numpy.ndarray
        FP32/FP64 array ``(P_numeric, N)``; preparation extracts declared values but does
        not fit scaling, imputation or bin edges.
    categorical : numpy.ndarray
        Int64 array ``(P_categorical, N)`` with shared nonnegative identities and
        ``-1`` for actual missing categories.
    complete : numpy.ndarray
        Boolean ``(N,)`` mask reflecting missingness in declared numeric and
        categorical features. Excluded source columns have no effect.

    Notes
    -----
    Factories provide owned arrays. Construction marks supplied arrays read-only;
    this frozen dataclass does not itself copy or validate arbitrary input arrays.
    """

    numeric: FloatArray
    categorical: IntArray
    complete: BoolArray

    def __post_init__(self) -> None:
        """Mark each supplied NumPy array read-only without copying its storage."""
        for array in (self.numeric, self.categorical, self.complete):
            array.setflags(write=False)

    @property
    def n_rows(self) -> int:
        """Number of source positions, including incomplete or duplicate rows."""
        return len(self.complete)


@dataclass(frozen=True)
class PreparedColumns:
    """Owned comparison sources with shared category identities and schema metadata.

    Attributes
    ----------
    reference, candidate : HostTable
        Feature-major host arrays; source row order and multiplicities are intact.
    numeric_names, categorical_names : tuple of str
        Feature order shared by both pools and all downstream tensor transfers.
    cardinalities : tuple of int
        Shared observed vocabulary sizes per categorical column. Actual missing
        values have code ``-1`` and do not occupy vocabulary entries.
    category_mismatch : bool
        Whether observed nonmissing category sets differ between the pools.
    reference_cardinalities : tuple of int or None
        Observed reference-only category counts used for reference-domain sizing.
    holdout : HostTable or None
        Optional independent real pool for training-referenced distance ratios.
    category_values : tuple of tuple or None
        Optional typed label dictionaries retained for descriptive slice output.
        Each tuple is indexed by the existing shared categorical storage code.

    Notes
    -----
    Arrays are read-only. A shared encoding never authorizes fitting geometry,
    training vocabulary, binning or scaling on candidate/scoring rows.
    Two roles over one source share the same immutable HostTable.
    """

    reference: HostTable
    candidate: HostTable
    numeric_names: tuple[str, ...]
    categorical_names: tuple[str, ...]
    cardinalities: tuple[int, ...]
    category_mismatch: bool
    reference_cardinalities: tuple[int, ...] | None = None
    category_values: tuple[tuple[object, ...], ...] | None = None
    holdout: HostTable | None = None

    @property
    def n_numeric(self) -> int:
        """Number of numeric feature rows in either host pool."""
        return len(self.numeric_names)

    @property
    def n_categorical(self) -> int:
        """Number of categorical feature rows in either host pool."""
        return len(self.categorical_names)

    @property
    def classifier_width(self) -> int:
        """Conservative expanded classifier width used for resource estimates."""
        return self.n_numeric + sum(self.cardinalities) + self.n_categorical

    @classmethod
    def prepare(
        cls,
        reference: pd.DataFrame,
        candidate: pd.DataFrame,
        schema: Schema,
        *,
        retain_labels: bool = False,
    ) -> PreparedColumns:
        """Copy validated source tables into shared, immutable-by-contract host storage.

        Parameters
        ----------
        reference, candidate : pandas.DataFrame
            Validated source pools. Input values/storage are not modified;
            row positions, duplicates and declared-feature completeness are retained.
        schema : Schema
            Effective numeric/categorical schema in the required feature order.
        retain_labels : bool, default=False
            Retain existing typed dictionaries for human-readable slice details.

        Returns
        -------
        PreparedColumns
            Owned feature-major numeric FP32/FP64 and category int64 arrays, boolean row
            masks, and category metadata shared across the two pools. A later transfer
            selects FP32/FP64 without re-factorizing category identities.

        Notes
        -----
        Categories are factorized together using exact typed identities, with
        actual missing cells coded ``-1``. Literal string labels remain ordinary
        identities, including ``"nan"`` and whitespace-bearing labels.
        Completeness is measured only in declared numeric and categorical features.
        No values are repaired, imputed, rounded, parsed, or converted between roles.
        """

        def numerics(df: pd.DataFrame) -> FloatArray:
            """Stack numeric columns without unnecessarily widening FP32 sources."""
            if not schema.numeric:
                return np.empty((0, len(df)), dtype=np.float64)
            return np.stack([numeric_values(df[c]) for c in schema.numeric])

        shared = reference is candidate
        rc, sc, cardinalities, reference_cardinalities = [], [], [], []
        category_values = [] if retain_labels else None
        mismatch = False
        for col in schema.categorical:
            r, s = reference[col], candidate[col]
            if shared:
                (a,), vocabulary = factorize_categories(r)
                b = a
            else:
                (a, b), vocabulary = factorize_categories(r, s)
            if category_values is not None:
                category_values.append(vocabulary.values)
            rc.append(a)
            sc.append(b)
            cardinalities.append(len(vocabulary))
            present_r = np.bincount(a[a >= 0], minlength=len(vocabulary)) > 0
            present_s = (
                present_r if shared else np.bincount(b[b >= 0], minlength=len(vocabulary)) > 0
            )
            reference_cardinalities.append(int(present_r.sum()))
            mismatch |= not np.array_equal(present_r, present_s)

        def table(df: pd.DataFrame, codes: list[IntArray]) -> HostTable:
            """Combine owned arrays with declared-feature source completeness."""
            categories = np.stack(codes) if codes else np.empty((0, len(df)), dtype=np.int64)
            features = list(schema.numeric) + list(schema.categorical)
            complete = ~df[features].isna().any(axis=1).to_numpy()
            return HostTable(numerics(df), categories, complete)

        real = table(reference, rc)
        return cls(
            real,
            real if shared else table(candidate, sc),
            tuple(schema.numeric),
            tuple(schema.categorical),
            tuple(cardinalities),
            bool(mismatch),
            tuple(reference_cardinalities),
            tuple(category_values) if category_values is not None else None,
        )

    @classmethod
    def prepare_single(
        cls, source: pd.DataFrame, schema: Schema, *, retain_labels: bool = False
    ) -> PreparedColumns:
        """Own one immutable encoded pool, shared by two sampled comparison roles."""
        return cls.prepare(source, source, schema, retain_labels=retain_labels)

    def to_tensors(
        self,
        *,
        device: str | TorchDevice = "cpu",
        dtype: Precision = "float64",
        copy: bool = True,
    ) -> TensorInputs:
        """Copy unique source pools into contiguous row-major tensor storage.

        Parameters
        ----------
        device : str, default="cpu"
            CPU/CUDA destination accepted by ``resolve_device``; ``auto`` here chooses
            available CUDA directly rather than applying metric workload routing.
        dtype : {"float32", "float64"}, default="float64"
            Numeric tensor precision; category codes stay int64 and masks stay bool.
        copy : bool, default=True
            Preserve independent storage by default. Compiled preprocessing can
            transfer fresh owned buffers with False to avoid a second CPU copy.

        Returns
        -------
        TensorInputs
            Host-independent tensor pools and unchanged schema/category metadata.
            Two roles over one owned source retain shared tensor storage.
            Torch is imported lazily when this method is called.
        """
        from tabgen_eval.execution.torch.tensors import TensorInputs

        return TensorInputs.from_columns(self, device=device, dtype=dtype, copy=copy)
