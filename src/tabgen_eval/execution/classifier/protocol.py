"""Assign and validate backend-independent C2ST source folds and seeds.

Estimators choose draws and scoring partitions. Both native CPU and CUDA
adapters use this module before fitting so repeated original-source identities
cannot leak between training and held-out scoring folds. Seed validation keeps
the full uint64 range without an intermediate floating-point conversion.
"""

from __future__ import annotations

from collections.abc import Iterable
from numbers import Integral
from typing import TYPE_CHECKING, Protocol, TypeGuard

import numpy as np
from numpy.typing import ArrayLike

from tabgen_eval._typing import FloatArray, IntArray, SeedScalar

if TYPE_CHECKING:
    from tabgen_eval.metrics import ClassifierMetric


class ClassifierFit(Protocol):
    """Fit a prepared classifier fold and return scores plus factual diagnostics."""

    def __call__(
        self,
        train: FloatArray,
        labels: FloatArray,
        test: FloatArray,
        *,
        seed: int,
        point: bool,
        fit_threads: int = 1,
    ) -> tuple[FloatArray, dict[str, object]]: ...


def folds_and_seeds(
    rd: IntArray,
    sd: IntArray,
    reference_folds: ArrayLike,
    candidate_folds: ArrayLike,
    seeds: Iterable[SeedScalar],
    n_folds: int,
    *,
    ideal: bool = False,
) -> tuple[list[IntArray], list[SeedScalar]]:
    """Validate source-stable classifier folds and exact uint64 seeds.

    Parameters
    ----------
    rd, sd : numpy.ndarray
        Previously validated integer source indices ``[B, NR]``/``[B, NS]``.
        This helper checks fold consistency, not source-pool bounds.
    reference_folds, candidate_folds : array-like
        Integer fold labels matching each source draw, in ``[0, n_folds)``.
    seeds : iterable of int
        Exactly B integral values in ``[0, 2**64)``; booleans are rejected and
        values are never converted through float64.
    n_folds : int
        Configured number of original-source scoring partitions.
    ideal : bool, default=False
        Also require any reference identities shared across classes to keep the
        same fold. This cross-class identity rule applies only in ideal mode.

    Returns
    -------
    tuple
        A list of two owned contiguous int64 fold arrays, and a list of the
        validated seed objects. Per-draw source order and repeated positions
        are preserved; no input is mutated.

    Raises
    ------
    ValueError
        Fold shapes, labels, seed counts/ranges, or source-fold assignments are
        invalid. Every copy of a source identity must keep one scoring fold.
    """
    folds, mappings = [], []
    for labels, draw in [(reference_folds, rd), (candidate_folds, sd)]:
        array = np.asarray(labels)
        if array.shape != draw.shape or array.dtype.kind not in "iu":
            raise ValueError("fold arrays must be integer arrays matching draw shapes")
        if array.size and (array.min() < 0 or array.max() >= n_folds):
            raise ValueError("fold labels are outside the configured fold range")
        groups = []
        for positions, row in zip(draw, array, strict=True):
            mapping = {}
            for position, label in zip(positions, row, strict=True):
                if position in mapping and mapping[position] != label:
                    raise ValueError("copies of a source row must share a fold")
                mapping[position] = label
            groups.append(mapping)
        mappings.append(groups)
        folds.append(np.array(array, dtype=np.int64, order="C", copy=True))
    if ideal:
        for a, b in zip(*mappings, strict=True):
            if any(a[key] != b[key] for key in a.keys() & b.keys()):
                raise ValueError("shared source rows must share a fold across classes")
    items = list(seeds)
    if len(items) != len(rd) or any(
        isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2**64
        for seed in items
    ):
        raise ValueError("seeds must be nonnegative integers matching the draw batch")
    return folds, items


def source_folds(
    n_reference: int,
    n_candidate: int,
    n_folds: int,
    rng: np.random.Generator,
    *,
    shared: bool = False,
) -> tuple[IntArray, IntArray]:
    """Assign folds before resampling; shared source pools get identical assignments."""
    from sklearn.model_selection import StratifiedKFold

    labels = np.repeat([0, 1], [n_reference, n_candidate])
    if shared:
        if n_reference != n_candidate:
            raise ValueError("shared classifier pools must have equal row counts")
        folds = np.arange(n_reference, dtype=np.int64) % n_folds
        rng.shuffle(folds)
        return folds, folds.copy()
    folds = np.empty(len(labels), dtype=np.int64)
    seed = int(rng.integers(0, 2**32 - 1))
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (_, test) in enumerate(cv.split(np.empty((len(labels), 0)), labels)):
        folds[test] = fold
    return folds[:n_reference], folds[n_reference:]


def uses_classifier_protocol(metric: object) -> TypeGuard[ClassifierMetric]:
    """Dispatch built-in classifier fitting while respecting custom compute overrides."""
    from tabgen_eval.metrics.c2st import ClassifierMetric

    return (
        isinstance(metric, ClassifierMetric)
        and "compute" not in vars(metric)
        and type(metric).compute is ClassifierMetric.compute
    )
