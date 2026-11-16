"""CUDA fused logistic and exact full-one-hot classifier kernels.

Floating tensors retain declared FP32/FP64 and use libdevice exponential/logarithm
routines without TF32, autocast or wider refinement. Elementwise logistic work
uses 256-row programs; mapping/one-hot updates use 1024-element programs. Integer
histograms are exact. Categorical adjoints use stable row segments
instead of floating atomics. Host wrappers own outputs/scratch, guard device/current
stream and preserve private RNG state; full dense designs are column-major while
numeric and identity source buffers remain row-major.
"""

import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.language.extra.cuda import libdevice

from tabgen_eval.execution.torch.cuda.triton.host import install

install()


@triton.jit
def _logistic(Z, Y, R, H, LOSS, N, BLOCK: tl.constexpr):
    """Write residual, curvature and stable binary loss for masked 256-row blocks.

    Z/Y are [N] in selected precision; R/H/LOSS are fresh [N] buffers.
    Tail lanes are masked on load/store and exponential/logarithmic arithmetic
    never promotes FP32 inputs to FP64.
    """
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    z, y = tl.load(Z + i, i < N, 0), tl.load(Y + i, i < N, 0)
    e = libdevice.exp(-tl.abs(z))
    denominator = 1 + e
    probability = tl.where(z >= 0, 1 / denominator, e / denominator)
    residual = tl.where(y == 0, probability, tl.where(z >= 0, -e / denominator, -1 / denominator))
    signed = tl.where(y == 1, -z, z)
    loss = tl.maximum(signed, 0) + libdevice.log1p(libdevice.exp(-tl.abs(signed)))
    tl.store(R + i, residual / N, i < N)
    tl.store(H + i, e / (denominator * denominator * N), i < N)
    tl.store(LOSS + i, loss, i < N)


@triton.jit
def _loss_difference(Z, DELTA, Y, OUT, N, BLOCK: tl.constexpr):
    """Write stable selected-precision logistic loss changes in 256-row blocks.

    Read-only Z/DELTA/Y are [N]; OUT is [N]. Small changes use log1p/expm1
    to preserve Armijo decisions without subtracting near-equal total losses.
    N masks every tail; objective ridge is handled by the solver.
    """
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    z, delta, y = tl.load(Z + i, i < N, 0), tl.load(DELTA + i, i < N, 0), tl.load(Y + i, i < N, 0)
    signed = tl.where(y == 1, -z, z)
    change = tl.where(y == 1, -delta, delta)
    e = libdevice.exp(-tl.abs(signed))
    probability = tl.where(signed >= 0, 1 / (1 + e), e / (1 + e))
    small = libdevice.log1p(probability * libdevice.expm1(change))
    updated = signed + change
    old_loss = tl.maximum(signed, 0) + libdevice.log1p(e)
    new_loss = tl.maximum(updated, 0) + libdevice.log1p(libdevice.exp(-tl.abs(updated)))
    # At large negative changes, p can round to one and log1p's argument
    # approaches -1. The ordinary difference is well-resolved in that regime.
    difference = tl.where(tl.abs(change) <= 10, small, new_loss - old_loss)
    tl.store(OUT + i, difference, i < N)


@triton.jit
def _dense(IDS, CENTER, SCALE, OUT, N, P: tl.constexpr, PN: tl.constexpr, BLOCK: tl.constexpr):
    """Update one-hot positions in a prefilled column-major [N,D] design.

    Grid ceil(N*P/BLOCK); IDs select full features after numeric prefix PN.
    Training CENTER/SCALE and N/P define layout; each valid row/feature write
    is unique. Missing/unknown indices and element tails are masked.
    """
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    index = tl.load(IDS + i, i < N * P, -1)
    center, scale = tl.load(CENTER + index, index >= 0, 0), tl.load(SCALE + index, index >= 0, 1)
    value = (1 - center) / scale
    tl.store(OUT + (index + PN) * N + i // P, value, (i < N * P) & (index >= 0))


@triton.jit
def _cat_mv(IDS, W, SCALE, OUT, N, P: tl.constexpr):
    """Sum implicit categorical forward products for 128-row programs.

    IDs are row-major [N,P], W/SCALE are full feature vectors. Unknown
    indices contribute zero; selected-precision OUT [N] is written once per row.
    Center and intercept terms are added by the Design adapter.
    """
    r = tl.program_id(0) * 128 + tl.arange(0, 128)
    cols = tl.arange(0, 32)
    value = tl.full((128, 32), 0, OUT.dtype.element_ty)
    for start in range(0, P, 32):
        c = start + cols
        index = tl.load(IDS + r[:, None] * P + c[None, :], (r[:, None] < N) & (c[None, :] < P), -1)
        w = tl.load(W + index, index >= 0, 0)
        scale = tl.load(SCALE + index, index >= 0, 1)
        value += w / scale
    tl.store(OUT + r, tl.sum(value, axis=1), r < N)


@triton.jit
def _segment(ORDER, OFFSETS, V, OUT, K: tl.constexpr):
    """Sum stable row segments for 32-feature programs.

    ORDER lists sorted source rows, OFFSETS bounds each feature segment and
    V is [N]. Each output owns its reduction, avoiding floating atomics;
    feature/segment tails are masked and OUT is [K] in selected precision.
    """
    f = tl.program_id(0) * 32 + tl.arange(0, 32)
    r = tl.arange(0, 128)
    begin = tl.load(OFFSETS + f, f < K, 0)
    end = tl.load(OFFSETS + f + 1, f < K, 0)
    largest = tl.max(end - begin, 0)
    result = tl.full((32, 128), 0, V.dtype.element_ty)
    for start in range(0, largest, 128):
        location = begin[:, None] + start + r[None, :]
        valid = (f[:, None] < K) & (location < end[:, None])
        row = tl.load(ORDER + location, valid, 0)
        value = tl.load(V + row, valid, 0)
        result += value
    tl.store(OUT + f, tl.sum(result, axis=1), f < K)


def logistic(z: Tensor, y: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Return stable logistic derivatives and unnormalized per-row losses.

    Parameters
    ----------
    z, y : Tensor
        Matching contiguous FP32/FP64 CUDA ``[N]`` logits and binary labels.

    Returns
    -------
    tuple of Tensor
        Fresh residual ``(sigmoid(z)-y)/N``, curvature ``p*(1-p)/N``, and binary
        loss vectors, all ``[N]`` in the input dtype/device. Loss is not divided
        by N: the solver reduces it by mean. Empty inputs return empty buffers.

    Notes
    -----
    Inputs are read-only. Sign-aware residuals and exponential/log1p formulas
    avoid overflow and cancellation near probabilities zero/one. Fusion is
    explicitly disabled; no ambient TF32 or autocast policy is changed.
    """
    residual, curvature, loss = (torch.empty_like(z) for _ in range(3))
    if len(z):
        with torch.cuda.device(z.device):
            _logistic[(triton.cdiv(len(z), 256),)](
                z, y, residual, curvature, loss, len(z), 256, enable_fp_fusion=False
            )
    return residual, curvature, loss


def loss_difference(logits: Tensor, change: Tensor, labels: Tensor) -> Tensor:
    """Return stable per-row logistic loss changes for Armijo search.

    Parameters
    ----------
    logits, change, labels : Tensor
        Contiguous CUDA [N] vectors in selected FP32/FP64.

    Returns
    -------
    Tensor
        Fresh [N] loss differences; no input mutation or hidden tolerance adjustment.
    """
    out = torch.empty_like(logits)
    if len(logits):
        with torch.cuda.device(logits.device):
            _loss_difference[(triton.cdiv(len(logits), 256),)](
                logits, change, labels, out, len(logits), 256, enable_fp_fusion=False
            )
    return out


def dense(numbers: Tensor, ids: Tensor, centers: Tensor, scales: Tensor) -> Tensor:
    """Expand compact features into a fresh column-major CUDA ``[N, D]`` design.

    ``numbers[N, PN]`` is row-major selected-precision CUDA, ``ids[N, PC]`` is
    row-major int64 with compact categorical indices or unknown -1. Matching
    center/scale vectors have length K and ``D=PN+K+1``. Copy numeric columns,
    fill centered/scaled indicator baselines, scatter valid ones, and set the
    last column to the intercept. Inputs remain read-only; callers must budget
    the full N-by-D allocation before invoking this wrapper.
    """
    n, pn = numbers.shape
    d = pn + len(centers) + 1
    out = numbers.new_empty((d, n)).T
    out[:, :pn] = numbers
    out[:, pn:-1] = (-centers / scales)[None]
    out[:, -1] = 1
    if ids.numel():
        with torch.cuda.device(numbers.device):
            _dense[(triton.cdiv(ids.numel(), 1024),)](
                ids, centers, scales, out, n, ids.shape[1], pn, 1024, enable_fp_fusion=False
            )
    return out


def cat_mv(ids: Tensor, coefficients: Tensor, scales: Tensor) -> Tensor:
    """Return the uncentered implicit categorical forward product for each row.

    ``ids[N, PC]`` is row-major int64 CUDA; selected-precision ``coefficients[K]``
    and positive ``scales[K]`` share its device. Return fresh ``[N]`` sums of
    coefficient/scale for valid IDs, with unknown -1 contributing zero. Center
    correction and intercept addition belong to Design.mv. Inputs are read-only;
    empty identity matrices return zeros without a kernel launch.
    """
    out = coefficients.new_zeros(len(ids))
    if ids.numel():
        with torch.cuda.device(ids.device):
            _cat_mv[(triton.cdiv(len(ids), 128),)](
                ids, coefficients, scales, out, len(ids), ids.shape[1], enable_fp_fusion=False
            )
    return out


def inverse(ids: Tensor, width: int) -> tuple[Tensor, Tensor]:
    """Build stable category-to-observation row segments on the CUDA device.

    ``ids[N, PC]`` is contiguous int64 with fitted indices in ``[0, width)`` or
    unknown -1. Stable-sort flattened identities, discard unknowns, and return
    contiguous int64 row indices plus ``[width+1]`` cumulative offsets. Repeated
    rows are retained when their feature occurrences repeat. Stable ordering
    makes later floating adjoints deterministic without atomics. The input is
    read-only; temporary sorted keys/order can scale with N*PC, not just width.
    """
    keys, order = ids.flatten().sort(stable=True)
    keep = keys >= 0
    valid = keys[keep]
    order = (order[keep] // max(ids.shape[1], 1)).contiguous()
    counts = torch.bincount(valid, minlength=width)
    offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    return order, offsets


def segment(order: Tensor, offsets: Tensor, values: Tensor) -> Tensor:
    """Sum a read-only CUDA vector over cached stable category row segments.

    ``order``/``offsets`` are int64 arrays from inverse; ``values[N]`` is a
    contiguous selected-precision vector on their device. Return fresh ``[K]``
    sums, including zero for empty segments, where K=len(offsets)-1. Each feature
    owns its reduction, so no floating atomic accumulation or input mutation is
    needed. The caller must ensure row bounds and compatible segment metadata.
    """
    k = len(offsets) - 1
    out = values.new_empty(k)
    if k:
        with torch.cuda.device(values.device):
            _segment[(triton.cdiv(k, 32),)](order, offsets, values, out, k, enable_fp_fusion=False)
    return out
