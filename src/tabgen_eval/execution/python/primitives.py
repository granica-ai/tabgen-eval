r"""NaN-safe similarity primitives — the four building blocks of Fidelity & Conditional.

Each returns a score in ``[0, 1]`` (1 = identical) and returns NaN (never raises) on
degenerate input (empty input, or — for ``correlation_similarity`` only — a constant column,
where Pearson ρ is undefined). A constant column is still well-defined for KS, TV, and
contingency, which return a real score. The NaN-safe means in Fidelity/Conditional simply drop
a NaN — never count it as 0. Implemented directly on numpy / scipy / scikit-learn::

    ks_complement(a, b)      = 1 − sup_x |F̂_a(x) − F̂_b(x)|         (KS statistic; numeric marginal)
    tv_complement(a, b)      = 1 − ½ · Σ_c |p_a(c) − p_b(c)|        (total variation; categorical)
    correlation_similarity   = 1 − ½ · |ρ(a1,a2) − ρ(b1,b2)|       (Pearson ρ; numeric pair)
    contingency_similarity   = 1 − ½ · Σ_{i,j} |P_a(i,j) − P_b(i,j)|   (normalised joint table;
                                                                       numerics binned first)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from tabgen_eval._typing import FloatArray, IntArray
from tabgen_eval.preprocessing.categories import factorize_categories
from tabgen_eval.preprocessing.columns import numeric_values

NAN = float("nan")

# Number of bins used to discretise a numeric column for contingency tables (quantile bins, see
# ``_reference_edges``) and for numeric conditioning slices (equal-width, see ``conditional.py``).
NUM_BINS = 10


def _finite_numeric(series: pd.Series) -> NDArray[np.float64]:
    arr = numeric_values(series).astype(float, copy=False)
    return arr[np.isfinite(arr)]


def ks_complement_arr(x: FloatArray, y: FloatArray) -> float:
    """``1 - KS D-statistic`` between two 1-D finite arrays; NaN if either empty.

    Equals ``1 - scipy.stats.ks_2samp(x, y).statistic`` to floating round-off, but
    skips ``ks_2samp``'s p-value machinery (the asymptotic/exact
    ``_compute_outer_prob_inside_method`` — pure overhead when only the statistic
    is wanted). The D-statistic is ``sup_t |F̂_x(t) − F̂_y(t)|`` evaluated at every
    observed value via two ``searchsorted`` calls on the sorted samples.
    """
    if x.size == 0 or y.size == 0:
        return NAN
    x = np.sort(x, kind="quicksort")
    y = np.sort(y, kind="quicksort")
    allv = np.concatenate([x, y])
    cdf_x = np.searchsorted(x, allv, side="right") / x.size
    cdf_y = np.searchsorted(y, allv, side="right") / y.size
    return float(1.0 - np.max(np.abs(cdf_x - cdf_y)))


def tv_complement_codes(real_codes: IntArray, syn_codes: IntArray, n_cats: int) -> float:
    """``1 - 0.5 * sum_c |p_real(c) - p_syn(c)|`` from shared integer category codes.

    ``real_codes`` / ``syn_codes`` are codes in ``[0, n_cats)`` over a *shared*
    category space (equal codes ⇒ equal categories); negative codes denote NaN and
    are dropped, mirroring ``Series.dropna()``. `
    """
    rc = real_codes[real_codes >= 0]
    sc = syn_codes[syn_codes >= 0]
    if rc.size == 0 or sc.size == 0:
        return NAN
    fr = np.bincount(rc, minlength=n_cats) / rc.size
    fs = np.bincount(sc, minlength=n_cats) / sc.size
    return float(1.0 - 0.5 * np.abs(fr - fs).sum())


def factorize_shared(a: pd.Series, b: pd.Series) -> tuple[IntArray, IntArray, int]:
    """Jointly encode typed category identities; actual missing values receive -1."""
    (ca, cb), vocabulary = factorize_categories(a, b)
    return ca, cb, len(vocabulary)


def ks_complement(a: pd.Series, b: pd.Series) -> float:
    """``1 - KS_statistic(a, b)`` over finite values; NaN if either side is empty.

    Range ``[0, 1]``; 1.0 = identical empirical CDFs.
    """
    return ks_complement_arr(_finite_numeric(a), _finite_numeric(b))


def tv_complement(a: pd.Series, b: pd.Series) -> float:
    """``1 - 0.5 * sum_c |p_a(c) - p_b(c)|`` over the union of categories; NaN if either empty.

    Range ``[0, 1]``; 1.0 = identical category frequencies.
    """
    ca, cb, n_cats = factorize_shared(a, b)
    return tv_complement_codes(ca, cb, n_cats)


def _pearson(x: FloatArray, y: FloatArray) -> float:
    """Pearson correlation; NaN if fewer than 2 points or a side is constant."""
    if x.size < 2 or np.all(x == x[0]) or np.all(y == y[0]):
        return NAN
    sx, sy = x.std(), y.std()
    if sx == 0.0 or sy == 0.0:
        return NAN
    return float(np.corrcoef(x, y)[0, 1])


def correlation_similarity(a1: pd.Series, a2: pd.Series, b1: pd.Series, b2: pd.Series) -> float:
    """``1 - |corr(a1, a2) - corr(b1, b2)| / 2`` (Pearson); NaN if either corr is undefined.

    ``|corr_a - corr_b|`` lies in ``[0, 2]`` so the score lies in ``[0, 1]``.
    """
    xa1, xa2 = _aligned_numeric(a1, a2)
    xb1, xb2 = _aligned_numeric(b1, b2)
    ca = _pearson(xa1, xa2)
    cb = _pearson(xb1, xb2)
    if not (np.isfinite(ca) and np.isfinite(cb)):
        return NAN
    return 1.0 - 0.5 * abs(ca - cb)


def _aligned_numeric(
    s1: pd.Series, s2: pd.Series
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Row-aligned finite numeric pairs (drop rows where either value is non-finite)."""
    x = numeric_values(s1).astype(float, copy=False)
    y = numeric_values(s2).astype(float, copy=False)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def correlation_similarity_arr(
    ax: FloatArray, ay: FloatArray, bx: FloatArray, by: FloatArray
) -> float:
    """Compare Pearson correlations from row-aligned floating arrays.

    Each side excludes rows with a nonfinite value in either column. Return
    1 - 0.5*abs(rho_reference - rho_candidate), or NaN when either correlation is
    undefined. Arrays are read-only and require no pandas coercion.
    """
    ma = np.isfinite(ax) & np.isfinite(ay)
    mb = np.isfinite(bx) & np.isfinite(by)
    ca = _pearson(ax[ma], ay[ma])
    cb = _pearson(bx[mb], by[mb])
    if not (np.isfinite(ca) and np.isfinite(cb)):
        return NAN
    return 1.0 - 0.5 * abs(ca - cb)


def _as_labels(
    series: pd.Series, is_numeric: bool, edges: FloatArray | None, *, clip: bool = False
) -> pd.Series:
    """Turn a column into categorical labels; bin numerics with the given *edges*.

    ``clip=True`` clamps candidate values to the reference edge range, assigning
    out-of-range observations to the nearest edge bin. Categorical values retain
    their type and exact string content. Actual missing cells and unassigned
    numeric bins remain missing.
    """
    if is_numeric:
        vals = pd.Series(
            numeric_values(series), index=series.index, name=series.name, dtype=np.float64
        )
        if clip and edges is not None:
            vals = vals.clip(lower=edges[0], upper=edges[-1])
        binned = pd.cut(vals, bins=edges, include_lowest=True, duplicates="drop")
        codes = binned.cat.codes.astype("Int64")
        return codes.where(codes >= 0)
    return series


def equal_width_edges(values: FloatArray, num_bins: int) -> NDArray[np.float64] | None:
    """Unique, monotonic equal-width bin edges spanning the finite range of *values*.

    The single binning helper shared by every numeric-discretisation path (contingency tables
    here, conditioning slices in ``metrics/conditional.py``), so they handle degenerate columns
    identically. Robustness:

    - **no finite values** -> ``None`` (the caller scores NaN / skips the column);
    - **single distinct value** (constant column) -> the range is widened by a tiny relative
      amount so the value occupies one bin rather than collapsing to a single edge;
    - **large-magnitude narrow range** whose ``linspace`` edges round to the same float ->
      duplicates are collapsed by ``np.unique``; if fewer than two distinct edges survive the
      range cannot be binned and we return ``None`` (rather than letting ``pd.cut`` raise
      "Bin edges must be unique").
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    lo, hi = float(finite.min()), float(finite.max())
    if lo == hi:
        pad = abs(lo) * 1e-6 + 1e-9
        lo, hi = lo - pad, hi + pad
    edges = np.unique(np.linspace(lo, hi, num_bins + 1))
    return edges if edges.size >= 2 else None


def quantile_edges(values: FloatArray, num_bins: int) -> FloatArray | None:
    """Unique, monotonic equal-frequency (quantile) bin edges over the finite values of *values*.

    Drop-in alternative to :func:`equal_width_edges`: the edges are the ``k / num_bins`` empirical
    quantiles, so each bin holds ~``1 / num_bins`` of the values and a skewed or heavy-tailed column
    is not collapsed into one or two bins. The outer edges are the finite min and max, so every
    finite value falls inside the grid. Robustness:

    - **no finite values** -> ``None`` (the caller scores NaN / skips the column);
    - **single distinct value** (constant column) -> delegated to :func:`equal_width_edges`, so a
      constant column is binned identically under either scheme;
    - **ties** (a value holding more than ``1 / num_bins`` of the mass) make adjacent quantiles
      coincide; duplicates are collapsed by ``np.unique``, yielding *fewer* than ``num_bins`` bins
      rather than letting ``pd.cut`` raise "Bin edges must be unique". Fewer than two distinct
      edges -> ``None``.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    if finite.min() == finite.max():
        return equal_width_edges(finite, num_bins)
    edges = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, num_bins + 1)))
    return edges if edges.size >= 2 else None


def _bin_codes(values: FloatArray, edges: FloatArray, *, clip: bool) -> IntArray:
    """Integer bin codes for *values* under *edges*, matching ``pd.cut(bins=edges, right=True,
    include_lowest=True)``: bin ``j`` is ``(edges[j], edges[j+1]]`` with the first bin closed on the
    left, so the minimum edge lands in bin 0. ``-1`` marks non-finite or out-of-range
    values, which joint frequencies drop. ``clip=True`` clamps finite values into
    the edge range first (the candidate side, mirroring :func:`_as_labels`'s ``clip=True``).
    """
    v = np.asarray(values, dtype=float)
    out = np.full(v.shape, -1, dtype=np.int64)
    finite = np.isfinite(v)
    vv = v[finite]
    if clip:
        vv = np.clip(vv, edges[0], edges[-1])
    code = np.searchsorted(edges, vv, side="left") - 1
    code[vv == edges[0]] = 0  # include_lowest: the minimum edge falls in bin 0, not bin -1
    code[(vv < edges[0]) | (vv > edges[-1])] = -1  # outside the real range -> dropped
    out[finite] = code
    return out


def digitize_reference(
    real_vals: FloatArray, cand_vals: FloatArray, num_bins: int
) -> tuple[IntArray, IntArray, int] | None:
    """Bin a numeric column into shared, **real-anchored** bins -> integer codes for the real and
    candidate sides, matching the numeric path of :func:`contingency_similarity` (edges from the real
    values via :func:`quantile_edges`, candidate clipped to them). Returns ``(real_codes, cand_codes,
    n_bins)`` (``-1`` = dropped), or ``None`` if the real range is unbinnable — exactly when
    :func:`_reference_edges` would return ``None``.
    """
    edges = quantile_edges(real_vals, num_bins)
    if edges is None:
        return None
    return (
        _bin_codes(real_vals, edges, clip=False),
        _bin_codes(cand_vals, edges, clip=True),
        edges.size - 1,
    )


def _reference_edges(a: pd.Series) -> FloatArray | None:
    """Quantile bin edges over the **reference (real)** finite values of *a*.

    Quantile (equal-frequency) rather than equal-width bins, so a skewed or heavy-tailed column is
    not collapsed into one or two occupied bins — which would hide a scrambled numeric–categorical
    association behind a near-perfect contingency score.

    The grid is anchored to the reference distribution and the candidate is later clipped to
    these edges — so the candidate cannot move the bin boundaries. This keeps the discretisation
    robust to a generated outlier that would otherwise stretch a combined range and collapse every row
    into one bin (which would mask poor fidelity as a perfect score), and is consistent with the
    real-anchored conditioning-slice binning in ``metrics/conditional.py`` and the reference-fit
    min-max scaling used by PRDC.
    """
    return quantile_edges(_finite_numeric(a), NUM_BINS)


def contingency_similarity(
    a1: pd.Series,
    a2: pd.Series,
    b1: pd.Series,
    b2: pd.Series,
    *,
    col1_numeric: bool,
    col2_numeric: bool,
) -> float:
    """``1 - 0.5 * sum_{i,j} |joint_a(i,j) - joint_b(i,j)|`` over normalised contingency tables.

    Numeric columns are discretised into quantile bins anchored to the **reference** values
    (the ``a*`` columns), with the candidate (``b*``) clipped to those edges — so a num-cat or
    num-num pair becomes cat-cat on a shared, real-anchored grid the candidate cannot move. NaN
    if either joint table is empty.
    """
    edges1 = _reference_edges(a1) if col1_numeric else None
    edges2 = _reference_edges(a2) if col2_numeric else None
    if (col1_numeric and edges1 is None) or (col2_numeric and edges2 is None):
        return NAN

    la1 = _as_labels(a1, col1_numeric, edges1)
    la2 = _as_labels(a2, col2_numeric, edges2)
    lb1 = _as_labels(b1, col1_numeric, edges1, clip=True)
    lb2 = _as_labels(b2, col2_numeric, edges2, clip=True)

    ra, sa, _ = factorize_shared(la1, lb1)
    rb, sb, _ = factorize_shared(la2, lb2)
    ja = _joint_freq(pd.Series(ra).where(ra >= 0), pd.Series(rb).where(rb >= 0))
    jb = _joint_freq(pd.Series(sa).where(sa >= 0), pd.Series(sb).where(sb >= 0))
    if ja.empty or jb.empty:
        return NAN

    idx = ja.index.union(jb.index)
    ja = ja.reindex(idx, fill_value=0.0)
    jb = jb.reindex(idx, fill_value=0.0)
    tv = 0.5 * float(np.abs(ja.to_numpy() - jb.to_numpy()).sum())
    return 1.0 - tv


def _joint_freq(c1: pd.Series, c2: pd.Series) -> pd.Series:
    """Normalised joint frequency over (c1, c2) cells, dropping rows with any NaN label."""
    df = pd.DataFrame({"a": c1.to_numpy(), "b": c2.to_numpy()}).dropna()
    if df.empty:
        return pd.Series(dtype=float)
    counts = df.groupby(["a", "b"]).size()
    return counts / counts.sum()


def contingency_similarity_codes(
    real_a: IntArray,
    real_b: IntArray,
    cand_a: IntArray,
    cand_b: IntArray,
    n_a: int,
    n_b: int,
) -> float:
    """Array form of :func:`contingency_similarity` from shared integer codes (one axis per column).

    Codes are in ``[0, n_a)`` / ``[0, n_b)`` over a shared real/candidate space (categorical codes
    from :func:`factorize_shared`, or numeric bins from :func:`digitize_reference`); a negative code
    on *either* axis drops that row (mirroring :func:`_joint_freq`'s NaN drop). The TV is taken over
    the full ``n_a * n_b`` grid, so cells present in only one side (the reindexed union of
    :func:`contingency_similarity`) and cells absent from both (which contribute 0) are handled
    identically. NaN if either side has no surviving rows.
    """

    # TODO: densifies the full n_a*n_b grid using the GLOBAL category cardinalities; for a
    # high-cardinality cat x cat pair this is a memory/time regression vs a sparse joint count.
    def _freq(a: IntArray, b: IntArray) -> NDArray[np.float64] | None:
        keep = (a >= 0) & (b >= 0)
        a, b = a[keep], b[keep]
        if a.size == 0:
            return None
        flat = a.astype(np.int64) * n_b + b.astype(np.int64)
        return np.bincount(flat, minlength=n_a * n_b) / a.size

    fr = _freq(real_a, real_b)
    fs = _freq(cand_a, cand_b)
    if fr is None or fs is None:
        return NAN
    return float(1.0 - 0.5 * np.abs(fr - fs).sum())
