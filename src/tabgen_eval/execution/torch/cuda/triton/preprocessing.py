"""Shared GPU preprocessing kernels; no classifier or model dependencies.

Numeric statistics use bounded block partials and FP64 accumulators, rather than
FP64 copies of the observation matrix. Application fuses gather, imputation,
standardization and missing-indicator construction into one output pass. Strides
are compile-time layout metadata, keeping portable host-launcher signatures
bounded while supporting noncontiguous source views.
"""

from collections.abc import Sequence
from typing import Literal, TypeAlias, overload

import torch
import triton
import triton.language as tl
from torch import Tensor

from tabgen_eval.execution.torch.cuda.triton.host import install

_NumericParameters: TypeAlias = tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
_CategoryParameters: TypeAlias = tuple[Tensor, Tensor, Tensor, Tensor]

install()


@triton.jit
def _finite(value):
    return (value == value) & (tl.abs(value) != float("inf"))


@triton.jit
def _moments(
    A,
    SUM,
    COUNT,
    N,
    P: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    NB: tl.constexpr,
    CLASS: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
):
    b = tl.program_id(0)
    r = b * ROWS + tl.arange(0, ROWS)
    c = tl.program_id(1) * COLS + tl.arange(0, COLS)
    value = tl.load(
        A + r[:, None] * S0 + c[None, :] * S1, (r[:, None] < N) & (c[None, :] < P), float("nan")
    ).to(tl.float64)
    valid = (r[:, None] < N) & (c[None, :] < P) & _finite(value)
    total = tl.sum(tl.where(valid, value, 0), 0)
    count = tl.sum(valid.to(tl.int64), 0)
    pos = (CLASS * NB + b) * P + c
    tl.store(SUM + pos, total, c < P)
    tl.store(COUNT + pos, count, c < P)


@triton.jit
def _means(
    SUM,
    COUNT,
    IMPUTE,
    CENTER,
    RC,
    SC,
    P: tl.constexpr,
    NR,
    NS,
    NB: tl.constexpr,
    BLOCK: tl.constexpr,
):
    c = tl.program_id(0)
    b = tl.arange(0, BLOCK)
    rs = tl.sum(tl.load(SUM + b * P + c, b < NB, 0), 0)
    ss = tl.sum(tl.load(SUM + (NB + b) * P + c, b < NB, 0), 0)
    rn = tl.sum(tl.load(COUNT + b * P + c, b < NB, 0), 0)
    sn = tl.sum(tl.load(COUNT + (NB + b) * P + c, b < NB, 0), 0)
    impute = tl.where(rn > 0, rs / tl.maximum(rn, 1), ss / tl.maximum(sn, 1))
    center = (rs + ss + (NR + NS - rn - sn) * impute) / (NR + NS)
    tl.store(IMPUTE + c, impute)
    tl.store(CENTER + c, center)
    tl.store(RC + c, rn)
    tl.store(SC + c, sn)


@triton.jit
def _deviations(
    A,
    B,
    IMPUTE,
    CENTER,
    DEV,
    SQUARE,
    NR,
    NS,
    P: tl.constexpr,
    A0: tl.constexpr,
    A1: tl.constexpr,
    B0: tl.constexpr,
    B1: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
):
    block = tl.program_id(0)
    r = block * ROWS + tl.arange(0, ROWS)
    c = tl.program_id(1) * COLS + tl.arange(0, COLS)
    va = tl.load(A + r[:, None] * A0 + c[None, :] * A1, (r[:, None] < NR) & (c[None, :] < P), 0).to(
        tl.float64
    )
    vb = tl.load(
        B + (r[:, None] - NR) * B0 + c[None, :] * B1,
        (r[:, None] >= NR) & (r[:, None] < NR + NS) & (c[None, :] < P),
        0,
    ).to(tl.float64)
    value = tl.where(r[:, None] < NR, va, vb)
    impute = tl.load(IMPUTE + c, c < P, 0)
    center = tl.load(CENTER + c, c < P, 0)
    value = tl.where(_finite(value), value, impute[None, :])
    delta = tl.where(r[:, None] < NR + NS, value - center[None, :], 0)
    tl.store(DEV + block * P + c, tl.sum(delta, 0), c < P)
    tl.store(SQUARE + block * P + c, tl.sum(delta * delta, 0), c < P)


@triton.jit
def _scales(DEV, SQUARE, CENTER, SCALE, P: tl.constexpr, N, NB: tl.constexpr, BLOCK: tl.constexpr):
    c = tl.program_id(0)
    b = tl.arange(0, BLOCK)
    correction = tl.sum(tl.load(DEV + b * P + c, b < NB, 0), 0)
    square = tl.sum(tl.load(SQUARE + b * P + c, b < NB, 0), 0)
    variance = tl.maximum((square - correction * correction / N) / N, 0)
    center = tl.load(CENTER + c)
    eps: tl.constexpr = 2.220446049250313e-16
    constant = variance <= N * eps * variance + (N * center * eps) * (N * center * eps)
    tl.store(SCALE + c, tl.where(constant, 1, tl.sqrt(variance)))


@triton.jit
def _apply_numeric(
    A,
    KEEP,
    IND,
    IMPUTE,
    CENTER,
    SCALE,
    OUT,
    N,
    P: tl.constexpr,
    K: tl.constexpr,
    NI: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    LINEAR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    width: tl.constexpr = K + NI
    row, feature = index // width, index % width
    numeric = feature < K
    source = tl.load(KEEP + feature, numeric, 0)
    if LINEAR:
        indicator = tl.load(IND + feature - K, ~numeric, 0)
        source = tl.where(numeric, source, indicator)
    value = tl.load(A + row * S0 + source * S1, row < N, float("nan"))
    if LINEAR:
        finite = _finite(value)
        impute = tl.load(IMPUTE + feature, numeric, 0)
        center = tl.load(CENTER + feature, numeric, 0)
        scale = tl.load(SCALE + feature, numeric, 1)
        transformed = (tl.where(finite, value, impute) - center) / scale
        value = tl.where(numeric, transformed, (~finite).to(value.dtype))
    else:
        # Preserve established classifier treatment of every nonfinite value as
        # missing, while leaving all finite numerical observations unscaled.
        value = tl.where(_finite(value), value, float("nan"))
    tl.store(OUT + index, value, row < N)


@triton.jit
def _hist(C, OFFSETS, OUT, INVALID, N, P: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c = index % P
    code = tl.load(C + index, index < N * P, -1)
    offset = tl.load(OFFSETS + c)
    end = tl.load(OFFSETS + c + 1)
    valid = (index < N * P) & (code >= -1) & (code < end - offset - 1)
    # Sanitize before pointer arithmetic too: an arbitrary int64 code may overflow.
    safe = tl.where(valid, code, -1)
    tl.atomic_add(OUT + offset + safe + 1, 1, valid, sem="relaxed")
    # Boolean inversion is broken in Triton 3.2's CPU interpreter.
    invalid = tl.sum(((index < N * P) & (valid == 0)).to(tl.int32), 0)
    tl.store(INVALID + tl.program_id(0), invalid)


@triton.jit
def _map(
    C,
    DOMAIN,
    MAP,
    FEATURES,
    OUT,
    MISSING,
    UNSEEN,
    N,
    P: tl.constexpr,
    LINEAR: tl.constexpr,
    REPORT: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
):
    r = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    c = tl.program_id(1) * COLS + tl.arange(0, COLS)
    mask = (r[:, None] < N) & (c[None, :] < P)
    code = tl.load(C + r[:, None] * P + c[None, :], mask, -2)
    start = tl.load(DOMAIN + c, c < P, 0)
    end = tl.load(DOMAIN + c + 1, c < P, 0)
    valid = mask & (code >= -1) & (code < end[None, :] - start[None, :] - 1)
    safe = tl.where(valid, code, -1)
    local = tl.load(MAP + start[None, :] + safe + 1, valid, -1)
    offset = tl.load(FEATURES + c, c < P, 0)
    if LINEAR:
        result = tl.where(local >= 0, offset[None, :] + local, -1)
    else:
        count = tl.load(FEATURES + c + 1, c < P, 0) - offset
        result = tl.where(local >= 0, local, count[None, :])
    tl.store(OUT + r[:, None] * P + c[None, :], result, mask)
    if REPORT:
        missing = tl.sum((mask & (code == -1)).to(tl.int64), 0)
        unseen = tl.sum((mask & (local < 0) & (code >= 0)).to(tl.int64), 0)
        tl.atomic_add(MISSING + c, missing, c < P, sem="relaxed")
        tl.atomic_add(UNSEEN + c, unseen, c < P, sem="relaxed")


@triton.jit
def _missing(
    A,
    OUT,
    N,
    P: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
):
    r = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    c = tl.program_id(1) * COLS + tl.arange(0, COLS)
    mask = (r[:, None] < N) & (c[None, :] < P)
    value = tl.load(A + r[:, None] * S0 + c[None, :] * S1, mask, 0)
    count = tl.sum((mask & ~_finite(value)).to(tl.int64), 0)
    tl.atomic_add(OUT + c, count, c < P, sem="relaxed")


@overload
def fit_numeric(
    reference: Tensor,
    candidate: Tensor,
    linear: bool = True,
    *,
    return_diagnostics: Literal[False] = False,
) -> _NumericParameters: ...


@overload
def fit_numeric(
    reference: Tensor, candidate: Tensor, linear: bool = True, *, return_diagnostics: Literal[True]
) -> tuple[_NumericParameters, Tensor]: ...


@overload
def fit_numeric(
    reference: Tensor, candidate: Tensor, linear: bool = True, *, return_diagnostics: bool
) -> _NumericParameters | tuple[_NumericParameters, Tensor]: ...


def fit_numeric(
    reference: Tensor, candidate: Tensor, linear: bool = True, *, return_diagnostics: bool = False
) -> _NumericParameters | tuple[_NumericParameters, Tensor]:
    """Fit training-only numeric state with O(P * ceil(N/256)) scratch.

    Return source-column indices, missing-indicator source indices, imputation,
    center and scale. Parameters retain input precision; only block accumulators
    use FP64. Both sources must share feature width, dtype and CUDA device.
    """
    p = reference.shape[1]
    if not p:
        index = torch.empty(0, device=reference.device, dtype=torch.int64)
        empty = reference.new_empty(0)
        state = (index, index.clone(), empty, empty.clone(), empty.clone())
        return (state, index.new_empty((0, 2))) if return_diagnostics else state
    nr, ns = len(reference), len(candidate)
    nb = triton.cdiv(max(nr, ns), 256)
    sums = torch.zeros((2, nb, p), dtype=torch.float64, device=reference.device)
    counts = torch.zeros((2, nb, p), dtype=torch.int64, device=reference.device)
    impute = torch.empty(p, dtype=torch.float64, device=reference.device)
    center = torch.empty_like(impute)
    rc = torch.empty(p, dtype=torch.int64, device=reference.device)
    sc = torch.empty_like(rc)
    with torch.cuda.device(reference.device):
        for cls, values in enumerate((reference, candidate)):
            if len(values):
                _moments[(triton.cdiv(len(values), 256), triton.cdiv(p, 8))](
                    values,
                    sums,
                    counts,
                    len(values),
                    p,
                    *values.stride(),
                    nb,
                    cls,
                    256,
                    8,
                    enable_fp_fusion=False,
                )
        _means[(p,)](
            sums,
            counts,
            impute,
            center,
            rc,
            sc,
            p,
            nr,
            ns,
            nb,
            triton.next_power_of_2(nb),
            enable_fp_fusion=False,
        )
        keep = torch.nonzero((rc + sc) > 0).flatten()
        if linear:
            indicators = torch.nonzero(((rc + sc) > 0) & ((rc < nr) | (sc < ns))).flatten()
            blocks = triton.cdiv(nr + ns, 256)
            deviations = torch.empty((blocks, p), dtype=torch.float64, device=reference.device)
            squares = torch.empty_like(deviations)
            scale = torch.empty_like(center)
            _deviations[(blocks, triton.cdiv(p, 8))](
                reference,
                candidate,
                impute,
                center,
                deviations,
                squares,
                nr,
                ns,
                p,
                *reference.stride(),
                *candidate.stride(),
                256,
                8,
                enable_fp_fusion=False,
            )
            _scales[(p,)](
                deviations,
                squares,
                center,
                scale,
                p,
                nr + ns,
                blocks,
                triton.next_power_of_2(blocks),
                enable_fp_fusion=False,
            )
        else:
            indicators = keep.new_empty(0)
            center = torch.zeros_like(center)
            scale = torch.ones_like(center)
    state = (
        keep,
        indicators,
        impute[keep].to(reference.dtype),
        center[keep].to(reference.dtype),
        scale[keep].to(reference.dtype),
    )
    return (state, torch.stack((nr - rc, ns - sc), dim=1)) if return_diagnostics else state


def apply_numeric(
    values: Tensor,
    keep: Tensor,
    indicators: Tensor,
    impute: Tensor,
    center: Tensor,
    scale: Tensor,
    linear: bool = True,
) -> Tensor:
    """Write kept features and optional raw missing flags; trees retain NaNs.

    Missing counts/flags include all nonfinite values, matching the established
    classifier policy. Tree application canonicalizes infinities to NaN.
    """
    k, i = len(keep), len(indicators) if linear else 0
    out = values.new_empty((len(values), k + i))
    if out.numel():
        with torch.cuda.device(values.device):
            _apply_numeric[(triton.cdiv(out.numel(), 1024),)](
                values,
                keep,
                indicators,
                impute,
                center,
                scale,
                out,
                len(values),
                values.shape[1],
                k,
                i,
                *values.stride(),
                linear,
                1024,
                enable_fp_fusion=False,
            )
    return out


def histogram(codes: Tensor, offsets: Tensor, width: int) -> Tensor:
    """Count category identities, rejecting invalid codes without unsafe writes."""
    if codes.ndim != 2 or offsets.ndim != 1 or len(offsets) != codes.shape[1] + 1:
        raise ValueError("category offsets must contain one boundary per column plus the end")
    if codes.dtype != torch.int64 or offsets.dtype != torch.int64:
        raise ValueError("category codes and offsets must use int64")
    if codes.device != offsets.device or not codes.is_contiguous() or not offsets.is_contiguous():
        raise ValueError("category codes and offsets must be contiguous on the same device")
    if width < 0:
        raise ValueError("category histogram width must be nonnegative")
    # Validate boundaries before any kernel can derive an output address from them.
    valid_offsets = (offsets[0] == 0) & (offsets[-1] == width)
    valid_offsets &= (offsets[1:] > offsets[:-1]).all()
    if not bool(valid_offsets.item()):
        raise ValueError("category offsets must increase strictly from zero to histogram width")
    out = codes.new_zeros(width)
    if codes.numel():
        with torch.cuda.device(codes.device):
            blocks = triton.cdiv(codes.numel(), 1024)
            invalid = torch.empty(blocks, dtype=torch.int32, device=codes.device)
            _hist[(blocks,)](
                codes,
                offsets,
                out,
                invalid,
                len(codes),
                codes.shape[1],
                1024,
            )
            if bool(invalid.any().item()):
                raise ValueError("category code is outside its declared domain (missing is -1)")
    return out


@overload
def fit_categories(
    reference: Tensor,
    candidate: Tensor,
    cardinalities: Sequence[int] | Tensor,
    *,
    return_diagnostics: Literal[False] = False,
) -> _CategoryParameters: ...


@overload
def fit_categories(
    reference: Tensor,
    candidate: Tensor,
    cardinalities: Sequence[int] | Tensor,
    *,
    return_diagnostics: Literal[True],
) -> tuple[_CategoryParameters, Tensor]: ...


@overload
def fit_categories(
    reference: Tensor,
    candidate: Tensor,
    cardinalities: Sequence[int] | Tensor,
    *,
    return_diagnostics: bool,
) -> _CategoryParameters | tuple[_CategoryParameters, Tensor]: ...


def fit_categories(
    reference: Tensor,
    candidate: Tensor,
    cardinalities: Sequence[int] | Tensor,
    *,
    return_diagnostics: bool = False,
) -> _CategoryParameters | tuple[_CategoryParameters, Tensor]:
    """Fit union-training category domains; missing owns each domain's first slot."""
    cardinalities = torch.as_tensor(cardinalities, device=reference.device, dtype=torch.int64)
    domains = torch.cat((cardinalities.new_zeros(1), (cardinalities + 1).cumsum(0)))
    width = int(domains[-1].item())
    counts = histogram(reference.contiguous(), domains, width)
    candidate_counts = histogram(candidate.contiguous(), domains, width)
    missing = torch.stack((counts[domains[:-1]], candidate_counts[domains[:-1]]), dim=1)
    counts.add_(candidate_counts)
    active = torch.nonzero(counts > 0).flatten()
    columns = torch.bucketize(active, domains[1:], right=True)
    sizes = torch.bincount(columns, minlength=len(cardinalities))
    features = torch.cat((sizes.new_zeros(1), sizes.cumsum(0)))
    mapping = counts.new_full((width,), -1)
    mapping[active] = torch.arange(len(active), device=reference.device) - features[columns]
    state = (mapping, domains, features, active)
    return (state, missing) if return_diagnostics else state


@overload
def map_categories(
    codes: Tensor,
    mapping: Tensor,
    domain_offsets: Tensor,
    feature_offsets: Tensor,
    linear: bool = True,
    *,
    return_diagnostics: Literal[False] = False,
) -> Tensor: ...


@overload
def map_categories(
    codes: Tensor,
    mapping: Tensor,
    domain_offsets: Tensor,
    feature_offsets: Tensor,
    linear: bool = True,
    *,
    return_diagnostics: Literal[True],
) -> tuple[Tensor, Tensor, Tensor]: ...


@overload
def map_categories(
    codes: Tensor,
    mapping: Tensor,
    domain_offsets: Tensor,
    feature_offsets: Tensor,
    linear: bool = True,
    *,
    return_diagnostics: bool,
) -> Tensor | tuple[Tensor, Tensor, Tensor]: ...


def map_categories(
    codes: Tensor,
    mapping: Tensor,
    domain_offsets: Tensor,
    feature_offsets: Tensor,
    linear: bool = True,
    *,
    return_diagnostics: bool = False,
) -> Tensor | tuple[Tensor, Tensor, Tensor]:
    """Map fitted categories; unknown is -1 for linear or nonmatching K for trees."""
    codes = codes.contiguous()
    out = torch.empty_like(codes)
    missing = codes.new_zeros(codes.shape[1]) if return_diagnostics else codes
    unseen = torch.zeros_like(missing) if return_diagnostics else codes
    if codes.numel():
        with torch.cuda.device(codes.device):
            _map[(triton.cdiv(len(codes), 256), triton.cdiv(codes.shape[1], 8))](
                codes,
                domain_offsets,
                mapping,
                feature_offsets,
                out,
                missing,
                unseen,
                len(codes),
                codes.shape[1],
                linear,
                return_diagnostics,
                256,
                8,
            )
    return (out, missing, unseen) if return_diagnostics else out


def count_missing(values: Tensor) -> Tensor:
    """Count nonfinite held-out observations with bounded integer reductions."""
    out = torch.zeros(values.shape[1], dtype=torch.int64, device=values.device)
    if values.numel():
        with torch.cuda.device(values.device):
            _missing[(triton.cdiv(len(values), 256), triton.cdiv(values.shape[1], 8))](
                values,
                out,
                len(values),
                values.shape[1],
                *values.stride(),
                256,
                8,
            )
    return out


@triton.jit
def _assemble(
    NUMBERS,
    CODES,
    OUT,
    N,
    PN: tl.constexpr,
    PC: tl.constexpr,
    WIDTH: tl.constexpr,
    LINEAR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, feature = index // WIDTH, index % WIDTH
    numeric = feature < PN
    value = tl.load(NUMBERS + row * PN + feature, (row < N) & numeric, 0)
    if not LINEAR:
        category = tl.load(CODES + row * PC + feature - PN, (row < N) & ~numeric, 0)
        value = tl.where(numeric, value, category.to(value.dtype))
    tl.store(OUT + index, value, row < N)


@triton.jit
def _indicators(
    CODES,
    OUT,
    INVALID,
    N,
    PN: tl.constexpr,
    PC: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    category = tl.load(CODES + index, index < N * PC, -1)
    valid = (index < N * PC) & (category >= 0) & (category < WIDTH - PN)
    safe = tl.where(valid, category, 0)
    tl.store(OUT + (index // PC) * WIDTH + PN + safe, 1, valid)
    invalid = (index < N * PC) & ((category < -1) | (category >= WIDTH - PN))
    tl.store(INVALID + tl.program_id(0), tl.sum(invalid.to(tl.int32), 0))


def assemble(numbers: Tensor, codes: Tensor, feature_count: int, linear: bool = True) -> Tensor:
    """Materialize model input without an intercept; caller budgets the output.

    Linear designs contain full unscaled indicators. Tree designs retain one
    local category code per original column. This routine owns no model policy.
    """
    numbers, codes = numbers.contiguous(), codes.contiguous()
    if numbers.ndim != 2 or codes.ndim != 2 or len(numbers) != len(codes):
        raise ValueError("numeric and categorical matrices must have matching row counts")
    if numbers.device != codes.device or codes.dtype != torch.int64:
        raise ValueError("categorical codes must use int64 on the numeric matrix's device")
    if isinstance(feature_count, bool) or not isinstance(feature_count, int) or feature_count < 0:
        raise ValueError("categorical feature count must be a nonnegative integer")
    n, pn = numbers.shape
    pc = codes.shape[1]
    width = pn + (feature_count if linear else pc)
    out = numbers.new_empty((n, width))
    if out.numel() or (linear and codes.numel()):
        with torch.cuda.device(numbers.device):
            if out.numel():
                _assemble[(triton.cdiv(out.numel(), 1024),)](
                    numbers,
                    codes,
                    out,
                    n,
                    pn,
                    pc,
                    width,
                    linear,
                    1024,
                )
            if linear and codes.numel():
                blocks = triton.cdiv(codes.numel(), 1024)
                invalid = torch.empty(blocks, dtype=torch.int32, device=codes.device)
                _indicators[(blocks,)](
                    codes,
                    out,
                    invalid,
                    n,
                    pn,
                    pc,
                    width,
                    1024,
                )
                if bool(invalid.any().item()):
                    raise ValueError(
                        "categorical indicator index is outside the fitted feature domain"
                    )
    return out
