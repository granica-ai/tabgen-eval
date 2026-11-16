"""CUDA IEEE Gram and exact occupied-category reductions.

Numeric [N,P] matrices and int64 [N,P] code matrices are row-major. Gram programs
use 16x16 output tiles, 32-row inner products and bounded row chunks, masking all
column/row tails. Numeric dots request IEEE FP32/FP64; boolean overlap products
use exact integer accumulation. Contingencies count at most 4096 dense cells or
sort occupied int64 keys. Host wrappers reserve partial scratch and current-device
launches; no floating atomics, TF32, refinement or input mutation is introduced.
"""

import torch
import triton
import triton.language as tl
from torch import Tensor

from tabgen_eval.execution.torch.cuda.triton.host import install

install()

_NATIVE_FP64_DOT = tuple(map(int, triton.__version__.split(".")[:2])) >= (3, 6)


@triton.jit
def _gram(
    A,
    B,
    OUT,
    N,
    PA: tl.constexpr,
    PB: tl.constexpr,
    FP64: tl.constexpr,
    COUNTS: tl.constexpr,
    ROWS: tl.constexpr,
    NATIVE_FP64_DOT: tl.constexpr,
):
    """Produce masked 16x16 Gram partials for one bounded row chunk.

    Grid (ceil(PA/16), ceil(PB/16), row_chunks). OUT is [chunks,PA,PB];
    bool inputs use int8 products/int32 partials later summed to int64,
    while floating inputs use the declared IEEE precision.
    """
    i = tl.program_id(0) * 16 + tl.arange(0, 16)
    j = tl.program_id(1) * 16 + tl.arange(0, 16)
    k = tl.program_id(2) * ROWS + tl.arange(0, 32)
    kind = tl.int32 if COUNTS else tl.float64 if FP64 else tl.float32
    result = tl.full((16, 16), 0, kind)
    for offset in range(0, ROWS, 32):
        r = k + offset
        a = tl.load(A + r[None, :] * PA + i[:, None], (r[None, :] < N) & (i[:, None] < PA), 0)
        b = tl.load(B + r[:, None] * PB + j[None, :], (r[:, None] < N) & (j[None, :] < PB), 0)
        if COUNTS and not NATIVE_FP64_DOT:
            result += tl.sum(a.to(tl.int32)[:, None, :] * tl.trans(b).to(tl.int32)[None, :, :], 2)
        elif COUNTS:
            result = tl.dot(a.to(tl.int8), b.to(tl.int8), result, out_dtype=kind)
        elif FP64 and not NATIVE_FP64_DOT:
            # Triton 3.2 has no FP64 dot lowering. Keep all products and
            # reductions in FP64 rather than silently narrowing the inputs.
            result += tl.sum(a[:, None, :] * tl.trans(b)[None, :, :], 2)
        else:
            result = tl.dot(a, b, result, input_precision="ieee", out_dtype=kind)
    location = (tl.program_id(2) * PA + i[:, None]) * PB + j[None, :]
    tl.store(OUT + location, result, (i[:, None] < PA) & (j[None, :] < PB))


@triton.jit
def _keys(C, PAIRS, WIDTHS, OUT, N, P: tl.constexpr, BLOCK: tl.constexpr):
    """Encode [pair,N] int64 joint keys from row-major category columns.

    Grid (ceil(N/BLOCK), pairs); row tails and negative category identities
    are excluded with key -1. Category products must fit int64.
    """
    pair = tl.program_id(1)
    rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    i, j = tl.load(PAIRS + pair * 2), tl.load(PAIRS + pair * 2 + 1)
    width = tl.load(WIDTHS + pair)
    a = tl.load(C + rows * P + i, rows < N, -1)
    b = tl.load(C + rows * P + j, rows < N, -1)
    value = tl.where((a >= 0) & (b >= 0), a * width + b, -1)
    tl.store(OUT + pair * N + rows, value, rows < N)


@triton.jit
def _dense(C, PAIRS, WIDTHS, OUT, N, P: tl.constexpr, GRID: tl.constexpr):
    """Accumulate exact int64 histograms for bounded flattened grids.

    Grid (ceil(N/1024), pairs); valid rows/codes enter GRID<=4096 cells.
    Only integer histogram atomics update zero-initialized [pairs,GRID] OUT.
    """
    pair = tl.program_id(1)
    rows = tl.program_id(0) * 1024 + tl.arange(0, 1024)
    i, j = tl.load(PAIRS + pair * 2), tl.load(PAIRS + pair * 2 + 1)
    width = tl.load(WIDTHS + pair)
    a = tl.load(C + rows * P + i, rows < N, -1)
    b = tl.load(C + rows * P + j, rows < N, -1)
    valid = (rows < N) & (a >= 0) & (b >= 0)
    # Older histograms do not support masks or define out-of-range inputs.
    # Put invalid lanes in bin zero, then subtract their exact integer count.
    histogram = tl.histogram(tl.where(valid, a * width + b, 0).to(tl.int32), GRID)
    cells = tl.arange(0, GRID)
    histogram -= tl.where(cells == 0, tl.sum((~valid).to(tl.int32), 0), 0)
    tl.atomic_add(OUT + pair * GRID + cells, histogram.to(tl.int64), sem="relaxed")


@triton.jit
def _search(X, N, pair, values, RIGHT: tl.constexpr, STEPS: tl.constexpr, BLOCK: tl.constexpr):
    """Find per-lane lower/upper bounds in one sorted int64 pair-key row.

    ``X`` has [Q,N] row stride N; ``values`` has BLOCK query lanes. RIGHT chooses
    upper rather than lower bounds so duplicate key multiplicities come from
    subtracting the two. STEPS must cover N; masks protect exhausted/tail loads.
    Only registers change and no input/output buffer is written.
    """
    lo = tl.full((BLOCK,), 0, tl.int32)
    hi = tl.full((BLOCK,), N, tl.int32)
    for _ in range(STEPS):
        mid = (lo + hi) // 2
        value = tl.load(X + pair * N + mid, (lo < hi) & (mid < N), other=0)
        lower = (value <= values) if RIGHT else (value < values)
        active = lo < hi
        lo = tl.where(active & lower, mid + 1, lo)
        hi = tl.where(active & ~lower, mid, hi)
    return lo


@triton.jit
def _sparse(
    A, B, NA, NB, OUT, NR, NS, STRIDE, SIDE: tl.constexpr, FP64: tl.constexpr, STEPS: tl.constexpr
):
    """Reduce occupied key-frequency differences in 256-row programs.

    Sorted A/B are [pairs,NR/NS]; SIDE selects reference/candidate occupancy.
    Only each key's first occurrence contributes; -1/tails are masked. OUT
    stores [pairs,STRIDE] selected-precision partials for a bounded reduction.
    """
    pair = tl.program_id(1)
    rows = tl.program_id(0) * 256 + tl.arange(0, 256)
    kind = tl.float64 if FP64 else tl.float32
    ar, bs = tl.load(NA + pair), tl.load(NB + pair)
    if SIDE == 0:
        key = tl.load(A + pair * NR + rows, rows < NR, -1)
        previous = tl.load(A + pair * NR + rows - 1, (rows > 0) & (rows < NR), -1)
        valid = (rows < NR) & (key >= 0) & (key != previous)
        end = _search(A, NR, pair, key, True, STEPS, 256)
        left = _search(B, NS, pair, key, False, STEPS, 256)
        right = _search(B, NS, pair, key, True, STEPS, 256)
        delta = tl.abs(
            (end - rows).to(kind) / tl.maximum(ar, 1).to(kind)
            - (right - left).to(kind) / tl.maximum(bs, 1).to(kind)
        )
    else:
        key = tl.load(B + pair * NS + rows, rows < NS, -1)
        previous = tl.load(B + pair * NS + rows - 1, (rows > 0) & (rows < NS), -1)
        end = _search(B, NS, pair, key, True, STEPS, 256)
        left = _search(A, NR, pair, key, False, STEPS, 256)
        right = _search(A, NR, pair, key, True, STEPS, 256)
        valid = (rows < NS) & (key >= 0) & (key != previous) & (left == right)
        delta = (end - rows).to(kind) / tl.maximum(bs, 1).to(kind)
    result = tl.sum(tl.where(valid, delta, 0), axis=0)
    tl.store(OUT + pair * STRIDE + tl.program_id(0), result)


def gram(a: Tensor, b: Tensor, *, counts: bool = False) -> Tensor:
    """Launch IEEE ``a.T@b`` or exact bool overlap products with bounded partials.

    Inputs are read-only contiguous [N,P]/[N,Q] on one CUDA device. Row chunks
    are powers of two between 32 and 4096, bounding count partials to int32 and
    limiting floating scratch. Reduce partials to [P,Q] in int64 for counts or
    the declared FP32/FP64 dtype for numeric work. Fresh buffers and launches
    use the input device/current stream; fusion is disabled for floating dots.
    """
    block_rows = min(4096, max(32, triton.next_power_of_2(max(len(a), 1))))
    chunks = triton.cdiv(len(a), block_rows)
    out = torch.empty(
        (chunks, a.shape[1], b.shape[1]), device=a.device, dtype=torch.int32 if counts else a.dtype
    )
    if out.numel():
        with torch.cuda.device(a.device):
            _gram[(triton.cdiv(a.shape[1], 16), triton.cdiv(b.shape[1], 16), chunks)](
                a,
                b,
                out,
                len(a),
                a.shape[1],
                b.shape[1],
                a.dtype == torch.float64,
                counts,
                block_rows,
                _NATIVE_FP64_DOT,
                enable_fp_fusion=False,
            )
    return out.sum(dim=0, dtype=torch.int64 if counts else a.dtype)


def keys(codes: Tensor, pairs: Tensor, widths: Tensor) -> Tensor:
    """Allocate owned int64 [Q,N] joint keys from read-only row-major category codes.

    ``pairs`` is [Q,2] and ``widths`` gives each second-axis size. The higher-level
    plan validates key capacity; negative identities map to -1. 512-row programs
    mask row tails and launch on the source device's current stream.
    """
    out = codes.new_empty((len(pairs), len(codes)))
    if out.numel():
        with torch.cuda.device(codes.device):
            _keys[(triton.cdiv(len(codes), 512), len(pairs))](
                codes,
                pairs,
                widths,
                out,
                len(codes),
                codes.shape[1],
                512,
            )
    return out


def dense_counts(codes: Tensor, pairs: Tensor, widths: Tensor, grid: int) -> Tensor:
    """Return zero-initialized int64 [Q,grid] joint histograms on the code device.

    Code rows are [N,P], pairs [Q,2] and widths [Q]; the caller checks grid bounds
    and key capacity. 1024-row programs use local histograms followed by exact
    integer atomics. Empty row/pair dimensions return correctly shaped zeros
    without a kernel launch. Input tensors remain read-only.
    """
    out = codes.new_zeros((len(pairs), grid))
    if len(codes) and len(pairs):
        with torch.cuda.device(codes.device):
            _dense[(triton.cdiv(len(codes), 1024), len(pairs))](
                codes,
                pairs,
                widths,
                out,
                len(codes),
                codes.shape[1],
                grid,
            )
    return out


def sorted_tv(a: Tensor, b: Tensor, na: Tensor, nb: Tensor, prototype: Tensor) -> Tensor:
    """Reduce sorted occupied-key rows to L1 probability-distance scalars.

    ``a``/``b`` are int64 [Q,NR/NS], ``na``/``nb`` valid counts [Q], and
    ``prototype`` selects FP32/FP64 output. Two 256-row launch sequences cover
    reference keys and candidate-only keys without double-counting duplicates.
    Return [Q] distances in [0,2], with NaN for either empty distribution; callers
    form similarities as 1-0.5*distance. Sources are read-only and scratch is
    owned. Searches/counts stay integer until selected-precision normalization.
    """
    nr, ns = a.shape[1], b.shape[1]
    parts = []
    steps = max(nr, ns, 1).bit_length()
    for side, rows in enumerate((nr, ns)):
        blocks = triton.cdiv(rows, 256)
        out = prototype.new_empty((len(a), blocks))
        if out.numel():
            with torch.cuda.device(a.device):
                _sparse[(blocks, len(a))](
                    a,
                    b,
                    na,
                    nb,
                    out,
                    nr,
                    ns,
                    blocks,
                    side,
                    prototype.dtype == torch.float64,
                    steps,
                    enable_fp_fusion=False,
                )
        parts.append(out.sum(dim=1))
    return torch.where((na > 0) & (nb > 0), parts[0] + parts[1], float("nan"))
