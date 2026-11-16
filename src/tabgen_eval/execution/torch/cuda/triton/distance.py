"""CUDA direct-distance kernels for compact mixed feature tables.

Inputs are contiguous row-major numeric FP32/FP64 and int64 category matrices.
Programs cover 8 query by 128 reference rows; every tail load/store is masked.
Numeric differences accumulate in selected precision, category contributions in
exact integers; floating fusion/TF32/refinement are disabled. Host wrappers select
the caller device/current stream and allocate only caller-bounded tile or partial
reduction buffers. These kernels never mutate input coordinates or codes.
"""

from collections.abc import Sequence

import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.runtime.jit import JITFunction

from tabgen_eval.execution.torch.cuda.triton.host import install

install()

_BQ = 8
_BR = 128


@triton.jit
def _distances(A, B, CA, CB, NQ, NR, PN: tl.constexpr, PC: tl.constexpr, FP64: tl.constexpr):
    """Load one 8x128 row-major mixed-distance block.

    Grid axes are query/reference blocks. Runtime NQ/NR mask incomplete tails;
    PN/PC and FP64 fix feature layout/precision. Negative categories preserve
    unknown-zero-one-hot semantics. Returns row indices and register distances.
    """
    q = tl.program_id(0) * 8 + tl.arange(0, 8)
    r = tl.program_id(1) * 128 + tl.arange(0, 128)
    kind = tl.float64 if FP64 else tl.float32
    numeric = tl.full((8, 128), 0, kind)
    for c in range(PN):
        a = tl.load(A + q * PN + c, q < NQ, other=0).to(kind)
        b = tl.load(B + r * PN + c, r < NR, other=0).to(kind)
        delta = a[:, None] - b[None, :]
        numeric += delta * delta
    category = tl.full((8, 128), 0, tl.int32)
    for c in range(PC):
        a = tl.load(CA + q * PC + c, q < NQ, other=-1)
        b = tl.load(CB + r * PC + c, r < NR, other=-1)
        category += 2 * (a[:, None] != b[None, :]).to(tl.int32) - (
            (a[:, None] < 0) != (b[None, :] < 0)
        ).to(tl.int32)
    return q, r, numeric + category.to(kind)


@triton.jit
def _tile(A, B, CA, CB, OUT, NQ, NR, PN: tl.constexpr, PC: tl.constexpr, FP64: tl.constexpr):
    """Store selected-precision [NQ, NR] output for the caller's bounded tile.

    Grid (ceil(NQ/8), ceil(NR/128)); both store axes mask row tails. Inputs
    remain read-only and OUT is the only writable buffer.
    """
    q, r, distance = _distances(A, B, CA, CB, NQ, NR, PN, PC, FP64)
    tl.store(OUT + q[:, None] * NR + r[None, :], distance, (q[:, None] < NQ) & (r[None, :] < NR))


@triton.jit
def _minimum(A, B, CA, CB, OUT, NQ, NR, PN: tl.constexpr, PC: tl.constexpr, FP64: tl.constexpr):
    """Store one minimum per query/reference block, masking reference tails.

    OUT is row-major [NQ, ceil(NR/128)] in selected precision; invalid
    reference lanes become infinity before reduction. Host code merges blocks.
    """
    q, r, distance = _distances(A, B, CA, CB, NQ, NR, PN, PC, FP64)
    distance = tl.where(r[None, :] < NR, distance, float("inf"))
    blocks = tl.cdiv(NR, 128)
    tl.store(OUT + q * blocks + tl.program_id(1), tl.min(distance, axis=1), q < NQ)


@triton.jit(do_not_specialize=["QUERY_START", "REFERENCE_START"])
def _self_minimum(
    A,
    B,
    CA,
    CB,
    OUT,
    NQ,
    NR,
    PN: tl.constexpr,
    PC: tl.constexpr,
    FP64: tl.constexpr,
    QUERY_START,
    REFERENCE_START,
):
    """Reduce block minima after excluding equal global self-search row positions."""
    q, r, distance = _distances(A, B, CA, CB, NQ, NR, PN, PC, FP64)
    valid = (r[None, :] < NR) & (q[:, None] + QUERY_START != r[None, :] + REFERENCE_START)
    distance = tl.where(valid, distance, float("inf"))
    blocks = tl.cdiv(NR, 128)
    tl.store(OUT + q * blocks + tl.program_id(1), tl.min(distance, axis=1), q < NQ)


@triton.jit
def _cross(
    A,
    B,
    CA,
    CB,
    RA,
    RB,
    COUNT,
    RECALL,
    COVER,
    NQ,
    NR,
    PN: tl.constexpr,
    PC: tl.constexpr,
    FP64: tl.constexpr,
    COMPONENTS: tl.constexpr,
):
    """Fuse strict squared-radius comparisons with exact integer/boolean outputs.

    Each 8x128 program stores int32 counts by query block and reference row,
    and recall/coverage partial flags by query row and reference block. Row
    masks exclude padded cells; host reduction yields exact int64 totals.
    """
    q, r, distance = _distances(A, B, CA, CB, NQ, NR, PN, PC, FP64)
    valid = (q[:, None] < NQ) & (r[None, :] < NR)
    blocks = tl.cdiv(NR, 128)
    if COMPONENTS & 13:
        real_radius = tl.load(RA + q, q < NQ, other=0)
        inside = valid & (distance < real_radius[:, None])
        if COMPONENTS & 5:
            tl.store(COUNT + tl.program_id(0) * NR + r, tl.sum(inside.to(tl.int32), axis=0), r < NR)
        if COMPONENTS & 8:
            tl.store(
                COVER + q * blocks + tl.program_id(1), tl.max(inside.to(tl.int32), axis=1), q < NQ
            )
    if COMPONENTS & 2:
        fake_radius = tl.load(RB + r, r < NR, other=0)
        recalled = valid & (distance < fake_radius[None, :])
        tl.store(
            RECALL + q * blocks + tl.program_id(1), tl.max(recalled.to(tl.int32), axis=1), q < NQ
        )


def _launch(
    kernel: JITFunction,
    a: Tensor,
    b: Tensor,
    ca: Tensor,
    cb: Tensor,
    *outputs: Tensor,
    **options: int | bool,
) -> None:
    """Launch a validated mixed-distance kernel on the input device/current stream.

    Inputs have the layout documented by ``distance.ops``. Outputs are owned by the
    wrapper; this helper allocates no tile buffers. Both row axes use masked 8x128
    blocks, and floating fusion is disabled to preserve selected-precision arithmetic.
    """
    with torch.cuda.device(a.device):
        kernel[(triton.cdiv(len(a), _BQ), triton.cdiv(len(b), _BR))](
            a,
            b,
            ca,
            cb,
            *outputs,
            len(a),
            len(b),
            a.shape[1],
            ca.shape[1],
            a.dtype == torch.float64,
            **options,
            enable_fp_fusion=False,
        )


def distance_tile(a: Tensor, b: Tensor, ca: Tensor, cb: Tensor) -> Tensor:
    """Return owned ``[Q, R]`` squared distances for one caller-bounded tile.

    Inputs are contiguous CUDA numeric FP32/FP64 and category int64 matrices with
    aligned widths/rows. Validation and workspace limits belong to ``distance.ops``
    and its caller. Empty axes return a correctly shaped empty tensor; inputs are
    read-only and output precision/device follow the numeric inputs.
    """
    out = a.new_empty((len(a), len(b)))
    if len(a) and len(b):
        _launch(_tile, a, b, ca, cb, out)
    return out


def nearest_tile(a: Tensor, b: Tensor, ca: Tensor, cb: Tensor) -> Tensor:
    """Return squared reference minima ``[Q]`` using bounded block partials.

    The validated mixed-feature input contract matches ``distance_tile``. Temporary
    storage is ``[Q, ceil(R/128)]`` rather than a full distance tile. An empty
    reference returns infinity per query; an empty query returns an empty tensor.
    """
    if not len(a) or not len(b):
        return a.new_full((len(a),), float("inf"))
    partial = a.new_empty((len(a), triton.cdiv(len(b), _BR)))
    _launch(_minimum, a, b, ca, cb, partial)
    return partial.amin(dim=1)


def nearest_self_tile(
    a: Tensor, b: Tensor, ca: Tensor, cb: Tensor, query_start: int, reference_start: int
) -> Tensor:
    """Bounded squared minima with only each row's own parent position excluded."""
    if not len(a) or not len(b):
        return a.new_full((len(a),), float("inf"))
    partial = a.new_empty((len(a), triton.cdiv(len(b), _BR)))
    with torch.cuda.device(a.device):
        _self_minimum[(triton.cdiv(len(a), _BQ), triton.cdiv(len(b), _BR))](
            a,
            b,
            ca,
            cb,
            partial,
            len(a),
            len(b),
            a.shape[1],
            ca.shape[1],
            a.dtype == torch.float64,
            query_start,
            reference_start,
            enable_fp_fusion=False,
        )
    return partial.amin(dim=1)


def cross_tile(
    a: Tensor, b: Tensor, ca: Tensor, cb: Tensor, ra: Tensor, rb: Tensor, components: int = 15
) -> tuple[Tensor, Tensor, Tensor]:
    """Return PRDC counts ``[R]`` and recall/coverage flags ``[Q]`` for one tile.

    ``ra``/``rb`` are squared radii with the numeric dtype/device and lengths Q/R.
    Strict ball comparisons use selected precision; block counts are accumulated as
    integers and returned as int64, flags as bool. Empty axes return zero counts and
    false flags. Caller-selected tile sizes bound partial buffers; inputs are read-only.
    """
    if not len(a) or not len(b):
        return (
            torch.zeros(len(b), dtype=torch.int64, device=a.device),
            torch.zeros(len(a), dtype=torch.bool, device=a.device),
            torch.zeros(len(a), dtype=torch.bool, device=a.device),
        )
    counts = torch.empty((triton.cdiv(len(a), _BQ), len(b)), dtype=torch.int32, device=a.device)
    recall = torch.empty((len(a), triton.cdiv(len(b), _BR)), dtype=torch.bool, device=a.device)
    coverage = torch.empty_like(recall)
    _launch(_cross, a, b, ca, cb, ra, rb, counts, recall, coverage, COMPONENTS=components)
    return (
        counts.sum(dim=0)
        if components & 5
        else torch.zeros(len(b), dtype=torch.int64, device=a.device),
        recall.any(dim=1)
        if components & 2
        else torch.zeros(len(a), dtype=torch.bool, device=a.device),
        coverage.any(dim=1)
        if components & 8
        else torch.zeros(len(a), dtype=torch.bool, device=a.device),
    )


@triton.jit(do_not_specialize=["N", "P"])
def _presence(CODES, COLS, OFFSETS, CARDS, PRESENT, N, P, BINS: tl.constexpr, BLOCK: tl.constexpr):
    """Merge block-local histograms into reference-only int32 presence.

    Column groups share a power-of-two domain; 1024 strided source rows form
    one program. Histogram reduction limits duplicate atomic contention. All
    category tails and negative codes are masked; integer OR is exact.
    """
    col = tl.load(COLS + tl.program_id(0))
    rows = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    code = tl.load(CODES + rows * P + col, rows < N, -1)
    card = tl.load(CARDS + col)
    valid = (rows < N) & (code >= 0) & (code < card)
    # Older Triton has no histogram mask argument. Out-of-range values do not
    # contribute to any bin, preserving exclusion of missing codes and tails.
    safe_code = tl.where(valid, code, BINS).to(tl.int32)
    counts = tl.histogram(safe_code, BINS)
    levels = tl.arange(0, BINS)
    offset = tl.load(OFFSETS + col)
    tl.atomic_or(PRESENT + offset + levels, 1, (levels < card) & (counts > 0), sem="relaxed")


@triton.jit(do_not_specialize=["N", "P"])
def _mask(CODES, OFFSETS, CARDS, PRESENT, N, P, BLOCK: tl.constexpr):
    """Mask an owned row-major query in one launch using fitted membership."""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    col = i % P
    code = tl.load(CODES + i, i < N * P, -1)
    offset = tl.load(OFFSETS + col)
    card = tl.load(CARDS + col)
    valid = (i < N * P) & (code >= 0) & (code < card)
    present = tl.load(PRESENT + offset + code, valid, 0)
    tl.store(CODES + i, tl.where(present != 0, code, -1), i < N * P)


def category_presence(
    codes: Tensor, offsets: Tensor, cards: Tensor, groups: Sequence[tuple[int, Tensor]], total: int
) -> Tensor:
    """Fit a fresh int32 reference-only presence buffer in a compact identity domain.

    ``codes`` is contiguous int64 ``[N, P]``; ``offsets`` and ``cards`` are int64
    ``[P]`` metadata on the same device. ``groups`` pairs power-of-two bin counts
    with int64 column lists. ``total`` is the sum of cardinalities. Negative codes
    are ignored. Integer atomic OR merges block histograms exactly; input codes and
    cached metadata remain unchanged, and the returned zero-initialized buffer is owned.
    """
    presence = torch.zeros(total, dtype=torch.int32, device=codes.device)
    if codes.numel():
        with torch.cuda.device(codes.device):
            for bins, columns in groups:
                _presence[(len(columns), triton.cdiv(len(codes), 1024))](
                    codes,
                    columns,
                    offsets,
                    cards,
                    presence,
                    len(codes),
                    codes.shape[1],
                    BINS=bins,
                    BLOCK=1024,
                )
    return presence


def mask_categories(codes: Tensor, offsets: Tensor, cards: Tensor, presence: Tensor) -> None:
    """Mutate draw-owned category codes using reference-fitted compact presence.

    Inputs follow ``category_presence``'s device/layout contract. Identities absent
    from ``presence`` and invalid/negative codes become ``-1``; valid observed codes
    are retained. Metadata and presence are read-only. Empty codes skip launch;
    execution uses the codes' device/current stream without a host synchronization.
    """
    if codes.numel():
        with torch.cuda.device(codes.device):
            _mask[(triton.cdiv(codes.numel(), 1024),)](
                codes, offsets, cards, presence, len(codes), codes.shape[1], BLOCK=1024
            )
