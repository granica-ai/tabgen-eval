"""Conditional Shape, Trend, and frequency recovery within reference-defined slices.

Numeric conditioning columns use equal-width reference bins; categorical
columns use their observed reference identities. Slice eligibility and rarity
come from the assigned reference draw. Similarities compare observed values
within each column or pair, preserving the fidelity missingness contract.

The six outputs aggregate all eligible slices and their rare subset. Slices
receive equal weight within each conditioning column, then columns receive
equal weight. Undefined components are excluded from their means. Recovery is
the generated-to-reference slice-frequency ratio, with one as its target.

This module computes one supplied comparison and optional descriptive tables.
Evaluation owns source draws and repeated-score uncertainty; empirical score
bands remain descriptive and do not assign quality or significance labels.
"""

from __future__ import annotations

from itertools import combinations
from typing import Literal, overload

import numpy as np
import pandas as pd

from tabgen_eval._typing import BoolArray, FloatArray, IntArray
from tabgen_eval.execution.runtime.components import CONDITIONAL_COMPONENTS, selected_components
from tabgen_eval.execution.runtime.details import DetailedOutcome
from tabgen_eval.preprocessing.categories import factorize_categories
from tabgen_eval.preprocessing.columns import numeric_values
from tabgen_eval.schema import Schema

from .conditional_details import ConditionalDetails
from .primitives import (
    contingency_similarity_codes,
    correlation_similarity_arr,
    digitize_reference,
    equal_width_edges,
    factorize_shared,
    ks_complement_arr,
    tv_complement_codes,
)
from .stats import safe_nanmean, safe_ratio

NAN = float("nan")


@overload
def _bin_labels(
    real: pd.Series, syn: pd.Series, num_bins: int, *, return_edges: Literal[False] = False
) -> tuple[pd.Series, pd.Series] | None: ...


@overload
def _bin_labels(
    real: pd.Series, syn: pd.Series, num_bins: int, *, return_edges: Literal[True]
) -> tuple[pd.Series, pd.Series, FloatArray] | None: ...


@overload
def _bin_labels(
    real: pd.Series, syn: pd.Series, num_bins: int, *, return_edges: bool
) -> tuple[pd.Series, pd.Series] | tuple[pd.Series, pd.Series, FloatArray] | None: ...


def _bin_labels(
    real: pd.Series, syn: pd.Series, num_bins: int, *, return_edges: bool = False
) -> tuple[pd.Series, pd.Series] | tuple[pd.Series, pd.Series, FloatArray] | None:
    """Equal-width bin labels for a numeric conditioning column over the real range.

    Edges come from the shared :func:`~tabgen_eval.execution.python.primitives.equal_width_edges` (so a
    degenerate column is handled identically to contingency binning). Syn values outside the
    real range are clipped to the nearest edge before binning. Returns ``None`` if the real
    column has no finite values or a range too degenerate to bin. Optional
    interval metadata retains the same edges used for these labels.
    """
    rv = pd.Series(numeric_values(real), index=real.index, name=real.name, dtype=np.float64)
    edges = equal_width_edges(rv.to_numpy(dtype=float), num_bins)
    if edges is None:
        return None
    labels = [str(i) for i in range(edges.size - 1)]
    real_lab = pd.cut(rv, bins=edges, labels=labels, include_lowest=True).astype(object)
    sv = pd.Series(numeric_values(syn), index=syn.index, name=syn.name, dtype=np.float64).clip(
        lower=edges[0], upper=edges[-1]
    )
    syn_lab = pd.cut(sv, bins=edges, labels=labels, include_lowest=True).astype(object)
    return (real_lab, syn_lab, edges) if return_edges else (real_lab, syn_lab)


def _slice_labels(
    real: pd.DataFrame, syn: pd.DataFrame, cond: str, is_numeric: bool, num_bins: int
) -> tuple[pd.Series, pd.Series, list[str | int]] | None:
    """Return ``(real_labels, syn_labels, slice_values)`` for conditioning column *cond*."""
    if is_numeric:
        binned = _bin_labels(real[cond], syn[cond], num_bins)
        if binned is None:
            return None
        real_lab, syn_lab = binned
    else:
        real_codes, syn_codes, _ = factorize_shared(real[cond], syn[cond])
        real_lab = pd.Series(real_codes).where(real_codes >= 0)
        syn_lab = pd.Series(syn_codes).where(syn_codes >= 0)
    values = sorted(real_lab.dropna().unique().tolist())
    return real_lab, syn_lab, values


class _Encoded:
    """Column encodings shared by all conditional slices of one comparison.

    Reference and candidate columns are stored in positionally aligned arrays.
    Slice methods select rows with boolean masks and exclude the conditioning
    column from Shape and Trend calculations.

    Numeric columns store paired floating arrays; each slice filters finite values
    and fits reference-anchored contingency bins where needed. Categorical columns
    store shared identity codes and domain widths; -1 denotes missing cells.
    """

    def __init__(
        self,
        reference: pd.DataFrame,
        candidate: pd.DataFrame,
        schema: Schema,
        num_bins: int,
        *,
        details: bool = False,
    ) -> None:
        self.numeric = set(schema.numeric)
        self.cols = list(schema.numeric) + list(schema.categorical)
        self.num_bins = num_bins
        self.num: dict[str, tuple[FloatArray, FloatArray]] = {}
        self.cat: dict[str, tuple[IntArray, IntArray, int]] = {}
        self.category_values = {}
        for col in schema.numeric:
            ref = numeric_values(reference[col]).astype(float, copy=False)
            cand = numeric_values(candidate[col]).astype(float, copy=False)
            self.num[col] = (ref, cand)
        for category_index, col in enumerate(schema.categorical):
            if details:
                (ref, cand), vocabulary = factorize_categories(reference[col], candidate[col])
                self.cat[col] = ref, cand, len(vocabulary)
                self.category_values[len(schema.numeric) + category_index] = vocabulary.values
            else:
                self.cat[col] = factorize_shared(reference[col], candidate[col])

    def slice_shape(self, real_mask: BoolArray, syn_mask: BoolArray, cond: str) -> float:
        """Marginal similarity of every column except *cond*, on a positional slice mask."""
        scores: list[float] = []
        for col in self.cols:
            if col == cond:
                continue
            if col in self.numeric:
                ref, cand = self.num[col]
                r = ref[real_mask]
                s = cand[syn_mask]
                scores.append(ks_complement_arr(r[np.isfinite(r)], s[np.isfinite(s)]))
            else:
                ref_codes, cand_codes, n_cats = self.cat[col]
                scores.append(
                    tv_complement_codes(ref_codes[real_mask], cand_codes[syn_mask], n_cats)
                )
        return safe_nanmean(scores)

    def _pair_codes(
        self, col: str, is_numeric: bool, real_mask: BoolArray, syn_mask: BoolArray
    ) -> tuple[IntArray, IntArray, int] | None:
        """Return shared category codes or reference-fitted numeric bins for a slice.

        The result contains reference codes, candidate codes, and domain width. Return
        None when the reference numeric slice cannot define bins.
        """
        if is_numeric:
            ref, cand = self.num[col]
            return digitize_reference(ref[real_mask], cand[syn_mask], self.num_bins)
        ref_codes, cand_codes, n_cats = self.cat[col]
        return ref_codes[real_mask], cand_codes[syn_mask], n_cats

    def slice_trend(self, real_mask: BoolArray, syn_mask: BoolArray, cond: str) -> float:
        """Average association similarity over nonconditioning column pairs in a slice.

        Numeric pairs compare Pearson correlations; other pairs compare contingency
        distributions with reference-fitted numeric bins. Each contributing column's
        slice codes are computed at most once and cached for the remaining pairs.
        """
        cols = [c for c in self.cols if c != cond]
        # The slice masks are fixed for this call, so a column's codes are identical across every
        # pair it appears in -- bin/mask each contributing column at most once (lazily, so an
        # all-numeric pair that never needs codes pays nothing).
        code_cache: dict[str, tuple[IntArray, IntArray, int] | None] = {}

        def codes(col: str, is_numeric: bool) -> tuple[IntArray, IntArray, int] | None:
            if col not in code_cache:
                code_cache[col] = self._pair_codes(col, is_numeric, real_mask, syn_mask)
            return code_cache[col]

        scores: list[float] = []
        for c1, c2 in combinations(cols, 2):
            c1_num, c2_num = c1 in self.numeric, c2 in self.numeric
            if c1_num and c2_num:
                ref1, cand1 = self.num[c1]
                ref2, cand2 = self.num[c2]
                scores.append(
                    correlation_similarity_arr(
                        ref1[real_mask], ref2[real_mask], cand1[syn_mask], cand2[syn_mask]
                    )
                )
                continue
            cod1 = codes(c1, c1_num)
            cod2 = codes(c2, c2_num)
            if cod1 is None or cod2 is None:  # An unbinnable numeric column contributes NaN.
                scores.append(NAN)
                continue
            ref_a, cand_a, n_a = cod1
            ref_b, cand_b, n_b = cod2
            scores.append(contingency_similarity_codes(ref_a, ref_b, cand_a, cand_b, n_a, n_b))
        return safe_nanmean(scores)


def any_rare_slice(
    real: pd.DataFrame, schema: Schema, *, num_bins: int, min_count: int, rare_threshold: float
) -> bool:
    """Test whether a reference-defined retained slice meets the rarity threshold.

    Only nonmissing conditioning labels enter its population denominator.
    The scan inspects reference data without mutation and consumes no RNG.
    """
    numeric = set(schema.numeric)
    for cond in list(schema.numeric) + list(schema.categorical):
        sliced = _slice_labels(real, real, cond, cond in numeric, num_bins)
        if sliced is None:
            continue
        real_lab, _, values = sliced
        total = int(real_lab.notna().sum())
        if total == 0:
            continue
        counts = real_lab.value_counts()
        for v in values:
            rc = int(counts.get(v, 0))
            if rc >= min_count and (rc / total) < rare_threshold:
                return True
    return False


@overload
def compute_conditional(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    schema: Schema,
    *,
    num_bins: int,
    min_count: int,
    rare_threshold: float,
    details: Literal[False] = False,
    requested_components: frozenset[str] | None = None,
) -> dict[str, float]: ...


@overload
def compute_conditional(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    schema: Schema,
    *,
    num_bins: int,
    min_count: int,
    rare_threshold: float,
    details: Literal[True],
    requested_components: frozenset[str] | None = None,
) -> DetailedOutcome: ...


@overload
def compute_conditional(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    schema: Schema,
    *,
    num_bins: int,
    min_count: int,
    rare_threshold: float,
    details: bool,
    requested_components: frozenset[str] | None = None,
) -> dict[str, float] | DetailedOutcome: ...


def compute_conditional(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    schema: Schema,
    *,
    num_bins: int,
    min_count: int,
    rare_threshold: float,
    details: bool = False,
    requested_components: frozenset[str] | None = None,
) -> dict[str, float] | DetailedOutcome:
    """Compute six Conditional statistics from explicit slice options.

    Column encodings are owned by this comparison and reused across slices. The
    reference defines slices, minimum-count eligibility, and rarity. Finite scores
    are averaged across retained slices, then across conditioning columns. Source
    frames remain read-only. Optional details retain the existing column and
    slice scores, including counts and ineligible observed reference groups.
    Recovery needs one modeled column, Shape two, and Trend three. Unselected
    components remain NaN and their slice calculations are skipped.
    """
    numeric = set(schema.numeric)
    conds = list(schema.numeric) + list(schema.categorical)
    width = len(conds)
    components = selected_components(requested_components, CONDITIONAL_COMPONENTS)
    wants_shape = "cond_shape" in components and width >= 2
    wants_trend = "cond_trend" in components and width >= 3
    wants_recovery = "recovery" in components

    # Recovery alone needs conditioning labels/counts, without within-slice encodings.
    enc = (
        _Encoded(reference, candidate, schema, num_bins, details=details)
        if details or wants_shape or wants_trend
        else None
    )
    collector = ConditionalDetails(conds, enc.category_values) if details else None

    col_shape_all, col_shape_rare = [], []
    col_trend_all, col_trend_rare = [], []
    col_rec_all, col_rec_rare = [], []

    for column_index, cond in enumerate(conds):
        edges = None
        if collector is not None and cond in numeric:
            binned = _bin_labels(reference[cond], candidate[cond], num_bins, return_edges=True)
            sliced = None
            if binned is not None:
                real_lab, syn_lab, edges = binned
                sliced = real_lab, syn_lab, sorted(real_lab.dropna().unique().tolist())
        elif collector is not None:
            ref_codes, syn_codes, _ = enc.cat[cond]
            real_lab = pd.Series(ref_codes).where(ref_codes >= 0)
            syn_lab = pd.Series(syn_codes).where(syn_codes >= 0)
            sliced = real_lab, syn_lab, sorted(real_lab.dropna().unique().tolist())
        else:
            sliced = _slice_labels(reference, candidate, cond, cond in numeric, num_bins)
        if sliced is None:
            continue
        real_lab, syn_lab, values = sliced
        real_lab_arr = real_lab.to_numpy()
        syn_lab_arr = syn_lab.to_numpy()
        total_real = int(real_lab.notna().sum())
        total_syn = int(syn_lab.notna().sum())
        if total_real == 0:
            continue
        real_counts = real_lab.value_counts()
        syn_counts = syn_lab.value_counts()

        shapes_all, shapes_rare = [], []
        trends_all, trends_rare = [], []
        recs_all, recs_rare = [], []
        for v in values:
            real_count = int(real_counts.get(v, 0))
            if real_count < min_count:
                if collector is not None:
                    collector.add_slice(
                        column_index,
                        int(v),
                        real_count,
                        int(syn_counts.get(v, 0)),
                        total_real,
                        total_syn,
                        False,
                        real_count / total_real < rare_threshold,
                        NAN,
                        NAN,
                        NAN,
                        edges,
                    )
                continue
            freq = real_count / total_real
            is_rare = freq < rare_threshold
            syn_count = int(syn_counts.get(v, 0))

            # Recovery (rate of this subpopulation).
            synth_freq = (syn_count / total_syn) if total_syn else 0.0
            recovery = safe_ratio(synth_freq, freq) if wants_recovery else NAN

            # An absent candidate slice scores zero only when enough other axes exist.
            shape = 0.0 if wants_shape else NAN
            trend = 0.0 if wants_trend else NAN
            if syn_count and (wants_shape or wants_trend):
                real_mask = real_lab_arr == v
                syn_mask = syn_lab_arr == v
                if wants_shape:
                    shape = enc.slice_shape(real_mask, syn_mask, cond)
                if wants_trend:
                    trend = enc.slice_trend(real_mask, syn_mask, cond)

            if collector is not None:
                collector.add_slice(
                    column_index,
                    int(v),
                    real_count,
                    syn_count,
                    total_real,
                    total_syn,
                    True,
                    is_rare,
                    shape,
                    trend,
                    recovery,
                    edges,
                )

            shapes_all.append(shape)
            trends_all.append(trend)
            recs_all.append(recovery)
            if is_rare:
                shapes_rare.append(shape)
                trends_rare.append(trend)
                recs_rare.append(recovery)

        col_shape_all.append(safe_nanmean(shapes_all))
        col_shape_rare.append(safe_nanmean(shapes_rare))
        col_trend_all.append(safe_nanmean(trends_all))
        col_trend_rare.append(safe_nanmean(trends_rare))
        col_rec_all.append(safe_nanmean(recs_all))
        col_rec_rare.append(safe_nanmean(recs_rare))
        if collector is not None:
            collector.add_column(
                column_index,
                (
                    col_shape_all[-1],
                    col_shape_rare[-1],
                    col_trend_all[-1],
                    col_trend_rare[-1],
                    col_rec_all[-1],
                    col_rec_rare[-1],
                ),
            )

    values = {
        "cond_shape_all": safe_nanmean(col_shape_all),
        "cond_shape_rare": safe_nanmean(col_shape_rare),
        "cond_trend_all": safe_nanmean(col_trend_all),
        "cond_trend_rare": safe_nanmean(col_trend_rare),
        "recovery_all": safe_nanmean(col_rec_all),
        "recovery_rare": safe_nanmean(col_rec_rare),
    }
    return DetailedOutcome(values, collector.finish()) if collector is not None else values
