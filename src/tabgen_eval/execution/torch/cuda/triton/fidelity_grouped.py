"""CUDA grouped gather, KS, IEEE Gram and exact contingency kernels.

Grouped numeric/category layouts are contiguous [groups,columns,padded_rows],
with NaN/-1 padding. Gathers and binning use 256-row programs; joint keys use
1024-row programs. Grams use masked 16x16 outputs and 32-row inner tiles, chunking
scratch before final reduction. KS searches sorted finite prefixes at observations
from both sides, preserving ties. Numeric work stays FP32/FP64; counts are exact
integers and only integer histograms use atomics. Host wrappers select the caller
device/stream and the group planner bounds padding, partials and pair batches.
"""

import torch
import triton
import triton.language as tl
from torch import Tensor

from tabgen_eval.execution.torch.cuda.triton.host import install

install()

_NATIVE_FP64_DOT = tuple(map(int, triton.__version__.split(".")[:2])) >= (3, 6)


@triton.jit
def _gather(
    X, ORDER, OFFSETS, GROUPS, OUT, P: tl.constexpr, R: tl.constexpr, INTEGER: tl.constexpr
):
    """Gather selected ragged rows into [G,P,R] padded output.

    Grid (ceil(R/256), G, P). ORDER/OFFSETS identify source rows; masks
    protect ragged and padded tails. OUT alone is written, with NaN/-1 padding.
    """
    g, c = tl.program_id(1), tl.program_id(2)
    r = tl.program_id(0) * 256 + tl.arange(0, 256)
    group = tl.load(GROUPS + g)
    start, end = tl.load(OFFSETS + group), tl.load(OFFSETS + group + 1)
    row = tl.load(ORDER + start + r, (r < R) & (start + r < end), 0)
    value = tl.load(X + row * P + c, (r < R) & (start + r < end), -1 if INTEGER else float("nan"))
    tl.store(OUT + (g * P + c) * R + r, value, r < R)


@triton.jit
def _gram(
    A,
    B,
    OUT,
    N: tl.constexpr,
    PA: tl.constexpr,
    PB: tl.constexpr,
    CHUNKS: tl.constexpr,
    ROWS: tl.constexpr,
    COUNTS: tl.constexpr,
    FP64: tl.constexpr,
    NATIVE_FP64_DOT: tl.constexpr,
):
    """Compute masked grouped 16x16 products for one row chunk.

    Grid axes cover PA, PB and flattened group/chunk. [G,P,N] inputs are
    read-only; OUT [G,chunks,PA,PB] uses IEEE selected precision or exact
    int32 count partials, summed by the wrapper.
    """
    i, j = tl.program_id(0) * 16 + tl.arange(0, 16), tl.program_id(1) * 16 + tl.arange(0, 16)
    group, chunk = tl.program_id(2) // CHUNKS, tl.program_id(2) % CHUNKS
    kind = tl.int32 if COUNTS else tl.float64 if FP64 else tl.float32
    value = tl.full((16, 16), 0, kind)
    for offset in range(0, ROWS, 32):
        r = chunk * ROWS + offset + tl.arange(0, 32)
        a = tl.load(
            A + (group * PA + i[:, None]) * N + r[None, :], (i[:, None] < PA) & (r[None, :] < N), 0
        )
        b = tl.load(
            B + (group * PB + j[None, :]) * N + r[:, None], (j[None, :] < PB) & (r[:, None] < N), 0
        )
        if COUNTS and not NATIVE_FP64_DOT:
            value += tl.sum(a.to(tl.int32)[:, None, :] * tl.trans(b).to(tl.int32)[None, :, :], 2)
        elif COUNTS:
            value = tl.dot(a.to(tl.int8), b.to(tl.int8), value, out_dtype=kind)
        elif FP64 and not NATIVE_FP64_DOT:
            value += tl.sum(a[:, None, :] * tl.trans(b)[None, :, :], 2)
        else:
            value = tl.dot(a, b, value, input_precision="ieee", out_dtype=kind)
    address = ((group * CHUNKS + chunk) * PA + i[:, None]) * PB + j[None, :]
    tl.store(OUT + address, value, (i[:, None] < PA) & (j[None, :] < PB))


@triton.jit
def _search(X, column, STRIDE: tl.constexpr, n, query, STEPS: tl.constexpr):
    """Return upper-bound positions for 256 queries in a sorted finite prefix.

    ``column`` selects a flattened [G,P] column with padded STRIDE; ``n`` is its
    true finite length. STEPS covers the maximum stride and masked binary-search
    loads never consume padding. Right bounds preserve empirical-CDF ties;
    the helper writes no device storage.
    """
    lo, hi = tl.full((256,), 0, tl.int64), tl.full((256,), n, tl.int64)
    for _ in range(STEPS):
        mid = (lo + hi) // 2
        value = tl.load(X + column * STRIDE + mid, (lo < hi) & (mid < n), 0)
        lower = value <= query
        active = lo < hi
        lo, hi = tl.where(active & lower, mid + 1, lo), tl.where(active & ~lower, mid, hi)
    return lo


@triton.jit
def _ks(
    A,
    B,
    NA,
    NB,
    OUT,
    NR: tl.constexpr,
    NS: tl.constexpr,
    BLOCKS: tl.constexpr,
    SIDE: tl.constexpr,
    FP64: tl.constexpr,
    STEPS: tl.constexpr,
):
    """Evaluate finite-prefix empirical CDF gaps in 256-observation blocks.

    Grid (observation blocks, G*P); SIDE chooses reference/candidate query
    values. NR/NS define padded strides, NA/NB finite counts. Right-sided
    binary searches preserve ties; invalid query tails contribute zero.
    """
    c = tl.program_id(1)
    r = tl.program_id(0) * 256 + tl.arange(0, 256)
    nr, ns = tl.load(NA + c), tl.load(NB + c)
    query = tl.load(A + c * NR + r, r < nr, 0) if SIDE == 0 else tl.load(B + c * NS + r, r < ns, 0)
    ar, bs = _search(A, c, NR, nr, query, STEPS), _search(B, c, NS, ns, query, STEPS)
    kind = tl.float64 if FP64 else tl.float32
    gap = tl.abs(
        ar.to(kind) / tl.maximum(nr, 1).to(kind) - bs.to(kind) / tl.maximum(ns, 1).to(kind)
    )
    valid = r < (nr if SIDE == 0 else ns)
    tl.store(OUT + c * BLOCKS + tl.program_id(0), tl.max(tl.where(valid, gap, 0), 0))


@triton.jit
def _bins(
    X, EDGES, LEVELS, OUT, N: tl.constexpr, E: tl.constexpr, CLIP: tl.constexpr, STEPS: tl.constexpr
):
    """Assign right-closed grouped bins with selected-precision edge comparisons.

    Grid (ceil(N/256), G*P); EDGES stores active prefixes and LEVELS their
    bin counts. Input infinities/NaNs are excluded before optional clipping;
    OUT [G,P,N] uses int64 -1 for invalid cells.
    """
    c = tl.program_id(1)
    r = tl.program_id(0) * 256 + tl.arange(0, 256)
    n = tl.load(LEVELS + c)
    first = tl.load(EDGES + c * E)
    last = tl.load(EDGES + c * E + n, n > 0, 0)
    value = tl.load(X + c * N + r, r < N, float("nan"))
    original = value
    if CLIP:
        value = tl.minimum(tl.maximum(value, first), last)
    lo, hi = tl.full((256,), 0, tl.int64), tl.full((256,), n + 1, tl.int64)
    for _ in range(STEPS):
        mid = (lo + hi) // 2
        edge = tl.load(EDGES + c * E + mid, (lo < hi) & (mid <= n), 0)
        lower = edge < value
        active = lo < hi
        lo, hi = tl.where(active & lower, mid + 1, lo), tl.where(active & ~lower, mid, hi)
    code = tl.where(value == first, 0, lo - 1)
    # Conditional numeric joint axes drop infinities before clipping, unlike
    # conditioning labels and the global Trend infinity convention.
    valid = (
        (original == original)
        & (tl.abs(original) != float("inf"))
        & (value >= first)
        & (value <= last)
        & (n > 0)
    )
    tl.store(OUT + c * N + r, tl.where(valid, code, -1), r < N)


@triton.jit(do_not_specialize=["N", "P", "Q", "COND"])
def _joint(
    C,
    PAIRS,
    WIDTHS,
    OUT,
    N,
    P,
    Q,
    GRID: tl.constexpr,
    DENSE: tl.constexpr,
    LEVEL_WIDTHS: tl.constexpr = False,
    MARGINAL: tl.constexpr = False,
    COND=-1,
):
    """Count bounded integer grids or store occupied keys per group/pair.

    Grid (ceil(N/1024), Q, G); negative codes and row tails are masked.
    DENSE updates zeroed [G,Q,GRID] int64 histograms with integer atomics;
    otherwise OUT [G*Q,N] stores int64 keys. No full category product is allocated.
    N/P/Q/COND remain runtime arguments: they affect address strides and axis
    exclusion, not register-tile dimensions. Keeping them unspecialized avoids
    recompiling one histogram variant for each padded group or pair batch.
    """
    q, g = tl.program_id(1), tl.program_id(2).to(tl.int64)
    r = tl.program_id(0) * 1024 + tl.arange(0, 1024)
    i, j = tl.load(PAIRS + q * 2), tl.load(PAIRS + q * 2 + 1)
    if LEVEL_WIDTHS:
        width = 0 if MARGINAL else tl.load(WIDTHS + g * P + j)
    else:
        width = tl.load(WIDTHS + g * Q + q)
    a = tl.load(C + (g * P + i) * N + r, r < N, -1)
    b = tl.load(C + (g * P + j) * N + r, r < N, -1)
    key = a * width + b
    valid = (r < N) & (a >= 0) & (b >= 0)
    if LEVEL_WIDTHS:
        valid = valid & (i != COND) & (j != COND)
    if DENSE:
        counts = tl.histogram(tl.where(valid, key, 0).to(tl.int32), GRID)
        cells = tl.arange(0, GRID)
        counts -= tl.where(cells == 0, tl.sum((~valid).to(tl.int32), 0), 0)
        tl.atomic_add(
            OUT + (g * Q + q) * GRID + cells, counts.to(tl.int64), mask=counts != 0, sem="relaxed"
        )
    else:
        tl.store(OUT + (g * Q + q) * N + r, tl.where(valid, key, -1), r < N)


@triton.jit
def _divide(a, b, FP64: tl.constexpr):
    """Divide in declared precision while preserving FP32 subnormal denominators.

    FP64 uses ordinary division. Emit FP32 round-to-nearest PTX explicitly:
    Triton 3.2 lowers ``tl.div_rn`` to ``div.rn.ftz.f32``, which flushes tiny
    operands to zero. Used by centering, edge construction and probability
    normalization; no storage is mutated.
    """
    if FP64:
        return a / b
    else:
        return tl.inline_asm_elementwise(
            "div.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[a, b],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )


@triton.jit(do_not_specialize=["P", "Q", "COND"])
def _dense_tv(
    A,
    B,
    PAIRS,
    LEVELS,
    OUT,
    Q,
    P,
    GRID: tl.constexpr,
    COND,
    MARGINAL: tl.constexpr,
    FP64: tl.constexpr,
):
    """Convert two exact joint histograms into one similarity per group/pair.

    A/B are zero-initialized accumulated int64 [G,Q,GRID] counts; OUT is [G,Q]
    in selected precision. One program reduces one grid, normalizes each side
    by its own valid count, and clamps 1-0.5*L1 to [0,1]. Empty distributions,
    conditioning-axis pairs and inactive joint levels produce NaN. Only OUT
    is written; the histogram/count buffers remain read-only.
    """
    q, g = tl.program_id(0), tl.program_id(1)
    cells = tl.arange(0, GRID)
    a = tl.load(A + (g * Q + q) * GRID + cells)
    b = tl.load(B + (g * Q + q) * GRID + cells)
    nr, ns = tl.sum(a, 0), tl.sum(b, 0)
    kind = tl.float64 if FP64 else tl.float32
    delta = tl.sum(
        tl.abs(
            _divide(a.to(kind), tl.maximum(nr, 1).to(kind), FP64)
            - _divide(b.to(kind), tl.maximum(ns, 1).to(kind), FP64)
        ),
        0,
    )
    i, j = tl.load(PAIRS + q * 2), tl.load(PAIRS + q * 2 + 1)
    keep = (nr > 0) & (ns > 0) & (i != COND) & (j != COND)
    if not MARGINAL:
        keep = keep & (tl.load(LEVELS + g * P + i) > 0) & (tl.load(LEVELS + g * P + j) > 0)
    score = tl.minimum(tl.maximum(1 - 0.5 * delta, 0), 1)
    tl.store(OUT + g * Q + q, tl.where(keep, score, float("nan")))


@triton.jit(do_not_specialize=["N"])
def _edges(X, COUNTS, OUT, LEVELS, N, BINS: tl.constexpr, BLOCK: tl.constexpr, FP64: tl.constexpr):
    """Fit and compress selected-precision quantile edges per group/column.

    X is sorted [G,P,N] with finite-prefix COUNTS [G,P]. OUT reserves BINS+1
    edges and LEVELS records the active bin count after stable duplicate
    compression. Nearest-endpoint interpolation limits cancellation; constant
    columns receive padded equal-width edges, and empty columns have zero
    levels. N is runtime to avoid one JIT variant per padded row count. Only
    owned OUT/LEVELS storage is written; inactive edge capacity is padded.
    """
    c = tl.program_id(0)
    q = tl.arange(0, BLOCK)
    kind = tl.float64 if FP64 else tl.float32
    n = tl.load(COUNTS + c)
    last = tl.maximum(n, 1) - 1
    fraction_q = tl.where(q == BINS, 1, q.to(kind) * (1.0 / BINS))
    position = last.to(kind) * fraction_q
    low = tl.floor(position).to(tl.int64)
    high = tl.minimum(low + 1, last)
    fraction = position - low.to(kind)
    a = tl.load(X + c * N + low, q <= BINS, 0)
    b = tl.load(X + c * N + high, q <= BINS, 0)
    result = tl.where(fraction >= 0.5, b - (b - a) * (1 - fraction), a + (b - a) * fraction)
    lo = tl.load(X + c * N)
    hi = tl.load(X + c * N + last)
    pad = tl.abs(lo) * 1e-6 + 1e-9
    left, right = lo - pad, hi + pad
    equal = left + q.to(kind) * _divide(right - left, tl.full((), BINS, kind), FP64)
    equal = tl.where(q == BINS, right, equal)
    result = tl.where(lo == hi, equal, result)
    # Recompute the preceding quantile in linear space. Triton 3.2 does not
    # support the gather used by newer versions; a one-hot selection would
    # instead require quadratic space for large bin counts.
    previous_q = tl.maximum(q - 1, 0)
    previous_position = last.to(kind) * (previous_q.to(kind) * (1.0 / BINS))
    previous_low = tl.floor(previous_position).to(tl.int64)
    previous_high = tl.minimum(previous_low + 1, last)
    previous_fraction = previous_position - previous_low.to(kind)
    previous_a = tl.load(X + c * N + previous_low, q <= BINS, 0)
    previous_b = tl.load(X + c * N + previous_high, q <= BINS, 0)
    previous = tl.where(
        previous_fraction >= 0.5,
        previous_b - (previous_b - previous_a) * (1 - previous_fraction),
        previous_a + (previous_b - previous_a) * previous_fraction,
    )
    previous_equal = left + previous_q.to(kind) * _divide(
        right - left, tl.full((), BINS, kind), FP64
    )
    previous = tl.where(lo == hi, previous_equal, previous)
    distinct = (q <= BINS) & ((q == 0) | (result != previous))
    location = tl.cumsum(distinct.to(tl.int32), 0) - 1
    tl.store(OUT + c * (BINS + 1) + q, float("inf"), q <= BINS)
    tl.debug_barrier()
    tl.store(OUT + c * (BINS + 1) + location, result, distinct)
    levels = tl.where(n > 0, tl.sum(distinct.to(tl.int64), 0) - 1, 0)
    tl.store(LEVELS + c, levels)


@triton.jit(do_not_specialize=["N"])
def _center(X, VALID, OUT, VARIABLE, N, FP64: tl.constexpr, BLOCK: tl.constexpr):
    """Stable anchor/scale centering with bounded tiles and runtime row count.

    Four streaming passes avoid row-sized intermediates. VALID may encode
    pairwise overlap for fallback correlations, rather than just finiteness.
    """
    c = tl.program_id(0)
    r = tl.arange(0, BLOCK)
    kind = tl.float64 if FP64 else tl.float32
    tiny: tl.constexpr = 2.2250738585072014e-308 if FP64 else 1.1754943508222875e-38
    first = tl.full((), N, tl.int64)
    absolute = tl.full((), tiny, kind)
    for offset in range(0, N, BLOCK):
        row = offset + r
        keep = tl.load(VALID + c * N + row, row < N, False)
        x = tl.load(X + c * N + row, row < N, 0)
        first = tl.minimum(first, tl.min(tl.where(keep, row.to(tl.int64), N), 0))
        absolute = tl.maximum(absolute, tl.max(tl.abs(tl.where(keep, x, 0)), 0))
    anchor = tl.load(X + c * N + first, first < N, 0)
    scale = tl.full((), 0, kind)
    for offset in range(0, N, BLOCK):
        row = offset + r
        keep = tl.load(VALID + c * N + row, row < N, False)
        x = tl.load(X + c * N + row, row < N, 0)
        shifted = tl.where(keep, x, anchor) - anchor
        scale = tl.maximum(scale, tl.max(tl.abs(shifted), 0))
    denominator = tl.where(scale > 0, scale, 1)
    total = tl.full((), 0, kind)
    count = tl.full((), 0, tl.int64)
    for offset in range(0, N, BLOCK):
        row = offset + r
        keep = tl.load(VALID + c * N + row, row < N, False)
        x = tl.load(X + c * N + row, row < N, 0)
        clean = tl.where(keep, x, anchor)
        z = tl.where(
            scale < float("inf"),
            _divide(clean - anchor, denominator, FP64),
            _divide(clean, absolute, FP64) - _divide(anchor, absolute, FP64),
        )
        total += tl.sum(z, 0)
        count += tl.sum(keep.to(tl.int64), 0)
    center = _divide(total, tl.maximum(count, 1).to(kind), FP64)
    for offset in range(0, N, BLOCK):
        row = offset + r
        keep = tl.load(VALID + c * N + row, row < N, False)
        x = tl.load(X + c * N + row, row < N, 0)
        clean = tl.where(keep, x, anchor)
        z = tl.where(
            scale < float("inf"),
            _divide(clean - anchor, denominator, FP64),
            _divide(clean, absolute, FP64) - _divide(anchor, absolute, FP64),
        )
        tl.store(OUT + c * N + row, tl.where(keep, z - center, 0), row < N)
    tl.store(VARIABLE + c, (scale > 0) & (count > 0))


def gather(data: Tensor, order: Tensor, offsets: Tensor, groups: Tensor, rows: int) -> Tensor:
    """Allocate padded [G,P,rows] storage and gather selected ragged source groups.

    Data is row-major [N,P] FP32/FP64 or int64. ORDER and OFFSETS encode the
    source-row permutation and group boundaries; GROUPS selects retained slices.
    The caller reserves padding capacity. 256-row programs write NaN numeric
    padding or -1 category padding, preserve source order/duplicates, and never
    modify inputs. Empty output dimensions skip the launch.
    """
    out = data.new_empty((len(groups), data.shape[1], rows))
    if out.numel():
        with torch.cuda.device(data.device):
            _gather[(triton.cdiv(rows, 256), len(groups), data.shape[1])](
                data, order, offsets, groups, out, data.shape[1], rows, data.dtype == torch.int64
            )
    return out


def gram(a: Tensor, b: Tensor, counts: bool) -> Tensor:
    """Compute grouped [G,P,R] by [G,Q,R] column products using bounded row chunks.

    Use 16x16 output tiles and 32-row inner products; chunks of at most 4096
    rows bound partial scratch and exact int32 bool counts. Return [G,P,Q]
    with int64 count reduction or input FP32/FP64 arithmetic. Inputs remain
    read-only; launches use their CUDA device/current stream and disable
    floating fusion, avoiding ambient TF32/autocast precision changes.
    """
    rows = min(4096, max(32, triton.next_power_of_2(a.shape[2])))
    chunks = triton.cdiv(a.shape[2], rows)
    out = a.new_empty(
        (len(a), chunks, a.shape[1], b.shape[1]), dtype=torch.int32 if counts else a.dtype
    )
    if out.numel():
        with torch.cuda.device(a.device):
            _gram[(triton.cdiv(a.shape[1], 16), triton.cdiv(b.shape[1], 16), len(a) * chunks)](
                a,
                b,
                out,
                a.shape[2],
                a.shape[1],
                b.shape[1],
                chunks,
                rows,
                counts,
                a.dtype == torch.float64,
                _NATIVE_FP64_DOT,
                enable_fp_fusion=False,
            )
    return out.sum(dim=1, dtype=torch.int64 if counts else a.dtype)


def ks(a: Tensor, b: Tensor, na: Tensor, nb: Tensor) -> Tensor:
    """Return [G,P] empirical KS complements for sorted grouped finite prefixes.

    A/B are FP32/FP64 [G,P,NR/NS] with exact int64 counts NA/NB [G,P]. Launch
    256-query programs from both populations and take their maximum CDF gap.
    Right-sided searches preserve ties and true-prefix masks ignore padding.
    Empty prefixes produce NaN. The caller budgets both partial arrays and
    provides nonempty padded row capacities; all source storage is read-only.
    """
    parts = []
    for side, rows in enumerate((a.shape[2], b.shape[2])):
        blocks = triton.cdiv(rows, 256)
        out = a.new_empty((len(a) * a.shape[1], blocks))
        if out.numel():
            with torch.cuda.device(a.device):
                _ks[(blocks, len(out))](
                    a,
                    b,
                    na,
                    nb,
                    out,
                    a.shape[2],
                    b.shape[2],
                    blocks,
                    side,
                    a.dtype == torch.float64,
                    max(a.shape[2], b.shape[2]).bit_length(),
                    enable_fp_fusion=False,
                )
        parts.append(out.amax(dim=1).view(a.shape[:2]))
    return torch.where((na > 0) & (nb > 0), 1 - torch.maximum(*parts), float("nan"))


def bins(values: Tensor, edges: Tensor, levels: Tensor, clip: bool) -> Tensor:
    """Assign int64 [G,P,R] codes from read-only grouped numeric values and edges.

    Edges [G,P,E] and active LEVELS [G,P] are reference-fitted. 256-row programs
    use right-closed bins and optional candidate clipping; original NaN and
    infinities remain excluded on within-slice contingency axes. Return owned
    codes with -1 exclusions on the input CUDA device/current stream.
    """
    out = values.new_empty(values.shape, dtype=torch.int64)
    if out.numel():
        with torch.cuda.device(values.device):
            _bins[(triton.cdiv(values.shape[2], 256), values.shape[0] * values.shape[1])](
                values,
                edges,
                levels,
                out,
                values.shape[2],
                edges.shape[2],
                clip,
                edges.shape[2].bit_length(),
                enable_fp_fusion=False,
            )
    return out


def edges(values: Tensor, counts: Tensor, bins: int) -> tuple[Tensor, Tensor]:
    """Allocate fixed-capacity reference quantile edges and active group-level counts.

    Sorted FP32/FP64 values [G,P,R] and int64 counts [G,P] are read-only.
    One fused program per group/column emits [G,P,bins+1] selected-precision
    boundaries plus int64 [G,P] levels; duplicated quantiles compress without
    host loops or per-column synchronization. Caller reserves all output bytes.
    """
    out = values.new_empty((*values.shape[:2], bins + 1))
    levels = counts.new_empty(counts.shape)
    if levels.numel():
        with torch.cuda.device(values.device):
            _edges[(levels.numel(),)](
                values,
                counts,
                out,
                levels,
                values.shape[2],
                bins,
                triton.next_power_of_2(bins + 1),
                values.dtype == torch.float64,
                enable_fp_fusion=False,
            )
    return out, levels


def center(values: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
    """Allocate stable normalized [G,P,R] columns and bool [G,P] variation flags.

    Read-only numeric values and bool validity share contiguous grouped layout.
    One program streams each column in 1024-element tiles, avoiding row-sized
    intermediates, while anchor/scale shifting protects large offsets and finite
    extreme differences. Invalid cells become zero. Runtime row counts reduce
    JIT specialization; division/fusion preserve selected precision semantics.
    """
    out = torch.empty_like(values)
    variable = valid.new_empty(values.shape[:2])
    if variable.numel():
        with torch.cuda.device(values.device):
            _center[(variable.numel(),)](
                values,
                valid,
                out,
                variable,
                values.shape[2],
                values.dtype == torch.float64,
                1024,
                enable_fp_fusion=False,
            )
    return out, variable


def dense(codes: Tensor, pairs: Tensor, widths: Tensor, grid: int) -> Tensor:
    """Count exact int64 [G,Q,grid] joint histograms for prevalidated small grids.

    Codes are [G,P,R], pairs [Q,2], widths [G,Q]. 1024-row programs ignore
    negative/padded identities and atomically add local integer histograms into
    fresh zeroed storage. The caller validates capacity and budgets counts;
    input codes and metadata remain read-only.
    """
    out = codes.new_zeros((len(codes), len(pairs), grid))
    if out.numel():
        with torch.cuda.device(codes.device):
            _joint[(triton.cdiv(codes.shape[2], 1024), len(pairs), len(codes))](
                codes, pairs, widths, out, codes.shape[2], codes.shape[1], len(pairs), grid, True
            )
    return out


def dense_tv(
    a: Tensor,
    b: Tensor,
    pairs: Tensor,
    levels: Tensor,
    grid: int,
    cond: int,
    marginal: bool,
    prototype: Tensor,
) -> Tensor:
    """Fuse exact histograms and TV normalization for a bounded group/pair batch.

    Codes A/B are int64 [G,P,R/S], pairs [Q,2], and LEVELS [G,P] supplies
    per-group axis widths. The prototype selects FP32/FP64 [G,Q] similarities.
    Marginal mode uses diagonal pairs; pairs touching COND and empty joint axes
    are excluded. Two owned integer count tensors are zeroed before launch.

    Grids through 4096 use one local histogram; 8192 uses two 4096-cell banks
    to avoid large-grid register spilling. The reduction runs once per group/pair
    on the current stream and leaves input pools/plans untouched. The caller
    must reserve both histograms plus output; no approximate floating atomics
    or full-vocabulary truncation enter the score.
    """
    out = prototype.new_empty((len(a), len(pairs)))
    if out.numel():
        ar, bs = [codes.new_zeros((len(codes), len(pairs), grid)) for codes in (a, b)]
        histogram = _joint_banked if grid > 4096 else _joint
        with torch.cuda.device(a.device):
            for codes, counts in ((a, ar), (b, bs)):
                histogram[
                    (
                        triton.cdiv(codes.shape[2], 1024) * (2 if grid > 4096 else 1),
                        len(pairs),
                        len(codes),
                    )
                ](
                    codes,
                    pairs,
                    levels,
                    counts,
                    codes.shape[2],
                    codes.shape[1],
                    len(pairs),
                    grid,
                    True,
                    True,
                    marginal,
                    cond,
                    num_warps=8 if grid > 4096 else 4,
                )
            _dense_tv[(len(pairs), len(a))](
                ar,
                bs,
                pairs,
                levels,
                out,
                len(pairs),
                a.shape[1],
                grid,
                cond,
                marginal,
                prototype.dtype == torch.float64,
                num_warps=8 if grid > 4096 else 4,
                enable_fp_fusion=False,
            )
    return out


def keys(codes: Tensor, pairs: Tensor, widths: Tensor) -> Tensor:
    """Encode grouped occupied identities into int64 [G*Q,R] joint-key rows.

    Inputs are [G,P,R] codes, shared [Q,2] pairs and [G,Q] second-axis widths.
    Flattening preserves group-major/pair-major order for downstream sorting.
    Negative codes become -1; products must already fit int64. Owned output
    scales with occupied rows, avoiding allocation of the full category product.
    """
    out = codes.new_empty((len(codes) * len(pairs), codes.shape[2]))
    if out.numel():
        with torch.cuda.device(codes.device):
            _joint[(triton.cdiv(codes.shape[2], 1024), len(pairs), len(codes))](
                codes, pairs, widths, out, codes.shape[2], codes.shape[1], len(pairs), 1, False
            )
    return out


@triton.jit(do_not_specialize=["N", "P", "Q", "COND"])
def _joint_banked(
    C,
    PAIRS,
    WIDTHS,
    OUT,
    N,
    P,
    Q,
    GRID: tl.constexpr,
    DENSE: tl.constexpr,
    LEVEL_WIDTHS: tl.constexpr = False,
    MARGINAL: tl.constexpr = False,
    COND=-1,
):
    """Count an 8192-cell grid using two independent 4096-cell local banks.

    Grid (2*ceil(N/1024), Q, G) reads each row block once per bank. Limiting
    each local histogram to 4096 cells avoids large-grid register spilling.
    Negative codes and row tails are masked; exact int64 atomics update the
    caller's zeroed [G,Q,GRID] counts. The original small-grid kernel is retained.
    """
    q, g = tl.program_id(1), tl.program_id(2).to(tl.int64)
    BANKS: tl.constexpr = 2 if DENSE and GRID > 4096 else 1
    bank = tl.program_id(0) % BANKS
    r = (tl.program_id(0) // BANKS) * 1024 + tl.arange(0, 1024)
    i, j = tl.load(PAIRS + q * 2), tl.load(PAIRS + q * 2 + 1)
    if LEVEL_WIDTHS:
        width = 0 if MARGINAL else tl.load(WIDTHS + g * P + j)
    else:
        width = tl.load(WIDTHS + g * Q + q)
    a = tl.load(C + (g * P + i) * N + r, r < N, -1)
    b = tl.load(C + (g * P + j) * N + r, r < N, -1)
    key = a * width + b
    valid = (r < N) & (a >= 0) & (b >= 0)
    if LEVEL_WIDTHS:
        valid = valid & (i != COND) & (j != COND)
    if DENSE:
        HIST: tl.constexpr = GRID if GRID < 4096 else 4096
        shifted = key - bank * HIST
        in_bank = valid & (shifted >= 0) & (shifted < HIST)
        counts = tl.histogram(tl.where(in_bank, shifted, 0).to(tl.int32), HIST)
        counts -= tl.where(tl.arange(0, HIST) == 0, tl.sum((~in_bank).to(tl.int32), 0), 0)
        cells = bank * HIST + tl.arange(0, HIST)
        tl.atomic_add(
            OUT + (g * Q + q) * GRID + cells, counts.to(tl.int64), mask=counts != 0, sem="relaxed"
        )
    else:
        tl.store(OUT + (g * Q + q) * N + r, tl.where(valid, key, -1), r < N)
