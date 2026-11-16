"""Registered CPU/CUDA fidelity primitives and metadata-only fake implementations.

Public custom operators validate dtype, device and contiguous layout before
routing to compiled ATen CPU or lazily imported Triton CUDA implementations.
They declare no input mutation. Fake kernels reproduce output shape/device/dtype
for tracing without reading observations. Pair counts and joint identities stay
integer; numeric products and normalized frequency distances use the caller's
FP32/FP64 precision. These are numerical building blocks, not report estimators.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from tabgen_eval.execution.torch.cpu.extension import library

if TYPE_CHECKING:
    from types import ModuleType


def cuda_kernels() -> ModuleType:
    """Lazily import CUDA fidelity launchers without making Triton a CPU dependency.

    Translate only a missing Triton installation into the optional-extra error;
    missing internal modules and their original exceptions remain visible.
    """
    try:
        return importlib.import_module("tabgen_eval.execution.torch.cuda.triton.fidelity")
    except ModuleNotFoundError as exc:
        if exc.name != "triton":
            raise
        raise ImportError("CUDA fidelity requires Triton; install 'tabgen-eval[cuda]'.") from exc


def _same_device(*tensors: Tensor) -> None:
    """Require every nonempty tensor argument list to be contiguous on one device.

    This shared layout check neither copies inputs nor validates their values;
    operator-specific helpers impose ranks and dtypes afterward.
    """
    if any(t.device != tensors[0].device or not t.is_contiguous() for t in tensors):
        raise ValueError("fidelity inputs must be contiguous on one device")


def _gram_inputs(a: Tensor, b: Tensor) -> None:
    """Validate shared FP32/FP64 ``[N,P]``/``[N,Q]`` Gram layouts without coercion."""
    _same_device(a, b)
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[0]:
        raise ValueError("Gram inputs must be matrices with matching row counts")
    if a.dtype not in {torch.float32, torch.float64} or b.dtype != a.dtype:
        raise ValueError("Gram inputs must share float32 or float64 dtype")


@torch.library.custom_op("tabgen_eval::fidelity_gram", mutates_args=(), device_types="cpu")
def fidelity_gram(a: Tensor, b: Tensor) -> Tensor:
    """Compute selected-precision ``a.T @ b`` without input mutation.

    Parameters
    ----------
    a, b : Tensor
        Contiguous FP32/FP64 ``[N, P]``/``[N, Q]`` matrices on one CPU/CUDA device.

    Returns
    -------
    Tensor
        ``[P, Q]`` in the input numeric dtype/device.

    Notes
    -----
    CPU disables ambient autocast locally; CUDA requests IEEE products rather
    than TF32. Fake dispatch validates layouts and returns output metadata only.
    """
    _gram_inputs(a, b)
    with torch.autocast("cpu", enabled=False):
        return library().gram(a, b)


@fidelity_gram.register_kernel("cuda")
def _cuda_gram(a: Tensor, b: Tensor) -> Tensor:
    """Validate the public Gram contract and launch selected-precision CUDA products.

    The kernel owns partial scratch and returns [P,Q]; input matrices remain
    read-only. IEEE dots do not inherit ambient TF32 matrix settings.
    """
    _gram_inputs(a, b)
    return cuda_kernels().gram(a, b)


@fidelity_gram.register_fake
def _fake_gram(a: Tensor, b: Tensor) -> Tensor:
    """Validate Gram metadata and produce symbolic ``[P,Q]`` output without reading data."""
    _gram_inputs(a, b)
    return a.new_empty((a.shape[1], b.shape[1]))


def _mask_inputs(mask: Tensor) -> None:
    """Require a contiguous bool ``[N,P]`` finite-cell mask on one device."""
    _same_device(mask)
    if mask.ndim != 2 or mask.dtype != torch.bool:
        raise ValueError("finite-pair masks must be boolean matrices")


@torch.library.custom_op("tabgen_eval::finite_pair_counts", mutates_args=(), device_types="cpu")
def finite_pair_counts(mask: Tensor) -> Tensor:
    """Count exact pairwise valid observations from finite-cell masks.

    Parameters
    ----------
    mask : Tensor
        Contiguous bool ``[N, P]`` on CPU or CUDA.

    Returns
    -------
    Tensor
        Int64 ``[P, P]`` overlap counts on the input device.

    Notes
    -----
    Read-only inputs; fake dispatch returns the same count metadata. Integer
    products/count reductions remain exact rather than using floating probabilities.
    """
    _mask_inputs(mask)
    return library().pair_counts(mask)


@finite_pair_counts.register_kernel("cuda")
def _cuda_pair_counts(mask: Tensor) -> Tensor:
    """Form exact CUDA ``[P,P]`` overlaps from bool masks using integer Gram partials.

    Partial int32 counts are bounded by row chunks and reduced to int64; floating
    normalization occurs later in the caller, not in this counting operator.
    """
    _mask_inputs(mask)
    return cuda_kernels().gram(mask, mask, counts=True)


@finite_pair_counts.register_fake
def _fake_pair_counts(mask: Tensor) -> Tensor:
    """Return int64 ``[P,P]`` overlap metadata after validating bool mask layout."""
    _mask_inputs(mask)
    return mask.new_empty((mask.shape[1], mask.shape[1]), dtype=torch.int64)


def _joint_inputs(codes: Tensor, pairs: Tensor, widths: Tensor) -> None:
    """Validate shared-device int64 code ``[N,P]``, pair ``[B,2]`` and width ``[B]`` layouts.

    Value bounds and flattened-key overflow are the planning/native caller's
    responsibility; this helper only checks metadata and does not synchronize.
    """
    _same_device(codes, pairs, widths)
    if codes.ndim != 2 or pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("joint codes and pair indices must be matrices")
    if widths.shape != (pairs.shape[0],):
        raise ValueError("joint widths must match the pair count")
    if any(t.dtype != torch.int64 for t in (codes, pairs, widths)):
        raise ValueError("joint codes, pairs and widths must use int64")


@torch.library.custom_op("tabgen_eval::joint_keys", mutates_args=(), device_types="cpu")
def joint_keys(codes: Tensor, pairs: Tensor, widths: Tensor) -> Tensor:
    """Encode occupied joint categories without category-product allocation.

    Parameters
    ----------
    codes : Tensor
        Contiguous int64 ``[N, P]`` with negative identities for excluded cells.
    pairs, widths : Tensor
        Int64 ``[B, 2]`` column indices and ``[B]`` positive second-axis widths.
        All tensors share one CPU/CUDA device.

    Returns
    -------
    Tensor
        Int64 ``[B, N]`` keys, with excluded pairs encoded as ``-1``.

    Notes
    -----
    Inputs are read-only and keys must fit int64. Fake dispatch preserves output
    metadata; the occupied-key path avoids allocating the full joint grid.
    """
    _joint_inputs(codes, pairs, widths)
    return library().keys(codes, pairs, widths)


@joint_keys.register_kernel("cuda")
def _cuda_keys(codes: Tensor, pairs: Tensor, widths: Tensor) -> Tensor:
    """Encode read-only CUDA pair identities into owned int64 ``[B,N]`` keys.

    Negative input identities map to -1; positive key capacity is checked by the
    higher-level contingency plan before entering the kernel.
    """
    _joint_inputs(codes, pairs, widths)
    return cuda_kernels().keys(codes, pairs, widths)


@joint_keys.register_fake
def _fake_keys(codes: Tensor, pairs: Tensor, widths: Tensor) -> Tensor:
    """Validate joint layouts and return symbolic int64 ``[B,N]`` occupied-key metadata."""
    _joint_inputs(codes, pairs, widths)
    return codes.new_empty((pairs.shape[0], codes.shape[0]))


def _dense_inputs(codes: Tensor, pairs: Tensor, widths: Tensor, grid: int) -> None:
    """Validate joint layouts and a power-of-two dense histogram capacity in [1,4096]."""
    _joint_inputs(codes, pairs, widths)
    if grid < 1 or grid > 4096 or grid & (grid - 1):
        raise ValueError("dense joint grid must be a power of two no greater than 4096")


@torch.library.custom_op("tabgen_eval::joint_dense_counts", mutates_args=(), device_types="cpu")
def joint_dense_counts(codes: Tensor, pairs: Tensor, widths: Tensor, grid: int) -> Tensor:
    """Count bounded joint grids with exact integer accumulation.

    Parameters
    ----------
    codes, pairs, widths : Tensor
        Same contiguous int64 layout as ``joint_keys``.
    grid : int
        Power-of-two flattened joint-grid capacity, between 1 and 4096.

    Returns
    -------
    Tensor
        Int64 ``[B, grid]`` on the input CPU/CUDA device.

    Notes
    -----
    Inputs are read-only. The caller ensures encoded keys fit the grid and reserves
    histogram scratch. Fake dispatch reproduces only output metadata.
    """
    _dense_inputs(codes, pairs, widths, grid)
    return library().dense_counts(codes, pairs, widths, grid)


@joint_dense_counts.register_kernel("cuda")
def _cuda_dense(codes: Tensor, pairs: Tensor, widths: Tensor, grid: int) -> Tensor:
    """Allocate exact CUDA ``[B,grid]`` counts and launch bounded joint histograms.

    The caller guarantees occupied keys fit ``grid``. Integer atomics update only
    zero-initialized owned output; source codes and pair metadata are read-only.
    """
    _dense_inputs(codes, pairs, widths, grid)
    return cuda_kernels().dense_counts(codes, pairs, widths, grid)


@joint_dense_counts.register_fake
def _fake_dense(codes: Tensor, pairs: Tensor, widths: Tensor, grid: int) -> Tensor:
    """Validate dense metadata and describe int64 ``[B,grid]`` output without counting."""
    _dense_inputs(codes, pairs, widths, grid)
    return codes.new_empty((pairs.shape[0], grid))


def _tv_inputs(a: Tensor, b: Tensor, na: Tensor, nb: Tensor, prototype: Tensor) -> None:
    """Validate sorted-key/count layouts and the scalar output-precision prototype.

    Keys are int64 ``[B,NR/NS]`` and valid sizes int64 ``[B]`` on one device.
    Sorted order, count contents and sentinel policy are trusted numerical
    preconditions; validation does not read or mutate the observations.
    """
    _same_device(a, b, na, nb, prototype)
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[0]:
        raise ValueError("sorted joint keys must have matching pair counts")
    if na.shape != (a.shape[0],) or nb.shape != na.shape:
        raise ValueError("valid joint counts must match the pair count")
    if any(t.dtype != torch.int64 for t in (a, b, na, nb)):
        raise ValueError("joint keys and valid counts must use int64")
    if prototype.ndim != 0 or prototype.dtype not in {torch.float32, torch.float64}:
        raise ValueError("TV prototype must be a float32 or float64 scalar")


@torch.library.custom_op("tabgen_eval::joint_sorted_tv", mutates_args=(), device_types="cpu")
def joint_sorted_tv(a: Tensor, b: Tensor, na: Tensor, nb: Tensor, prototype: Tensor) -> Tensor:
    """Return L1 distances between occupied joint-key frequency distributions.

    Parameters
    ----------
    a, b : Tensor
        Sorted contiguous int64 keys ``[B,NR]``/``[B,NS]`` on one CPU/CUDA device.
        Excluded observations use negative keys, which form the sorted prefix.
    na, nb : Tensor
        Int64 valid-observation counts ``[B]`` excluding that negative prefix.
    prototype : Tensor
        FP32/FP64 scalar selecting the output dtype/device.

    Returns
    -------
    Tensor
        ``[B]`` sums of absolute normalized-frequency differences in [0,2]. Either
        empty distribution yields NaN. Callers form TV complements as
        ``1-0.5*distance``; this operator deliberately returns the intermediate.

    Notes
    -----
    Inputs are read-only. Storage follows occupied rows, independently of the
    full category-domain product. Fake dispatch returns matching metadata.
    """
    _tv_inputs(a, b, na, nb, prototype)
    return library().sorted_tv(a, b, na, nb, prototype)


@joint_sorted_tv.register_kernel("cuda")
def _cuda_tv(a: Tensor, b: Tensor, na: Tensor, nb: Tensor, prototype: Tensor) -> Tensor:
    """Compute CUDA L1 differences of normalized occupied-key frequencies.

    The output is a distance in [0,2], not a complement. Callers turn this
    intermediate into TV similarity using ``1-0.5*distance``.
    """
    _tv_inputs(a, b, na, nb, prototype)
    return cuda_kernels().sorted_tv(a, b, na, nb, prototype)


@joint_sorted_tv.register_fake
def _fake_tv(a: Tensor, b: Tensor, na: Tensor, nb: Tensor, prototype: Tensor) -> Tensor:
    """Validate key metadata and return a ``[B]`` distance tensor in prototype precision."""
    _tv_inputs(a, b, na, nb, prototype)
    return prototype.new_empty((a.shape[0],))
