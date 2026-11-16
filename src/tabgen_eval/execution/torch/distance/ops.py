"""Registered PyTorch distance primitives with CPU and CUDA implementations.

Only imported for tensor distance metrics. CPU uses compiled ATen/C++ operators;
CUDA loads Triton lazily. Direct differences avoid dot-product cancellation,
without refinement, autocast, TF32 or changing process-wide precision settings.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from types import ModuleType


def cuda_kernels() -> ModuleType:
    """Import CUDA distance kernels lazily without hiding unrelated import failures.

    Missing Triton raises an installation-oriented ``ImportError``; other missing
    modules propagate unchanged so packaging or implementation errors stay visible.
    """
    try:
        return importlib.import_module("tabgen_eval.execution.torch.cuda.triton.distance")
    except ModuleNotFoundError as exc:
        if exc.name != "triton":
            raise
        raise ImportError("CUDA distances require Triton; install 'tabgen-eval[cuda]'.") from exc


def _validate(a: Tensor, b: Tensor, ca: Tensor, cb: Tensor) -> None:
    """Require aligned contiguous row-major numeric/category matrices on one device.

    Numeric pairs share FP32/FP64 dtype and width; category pairs share int64 width
    and row counts with their numeric partner. Raise ``ValueError`` for a layout,
    dtype, or device mismatch. No content scan, copy, or mutation is performed.
    """
    if any(x.ndim != 2 for x in (a, b, ca, cb)):
        raise ValueError("distance inputs must be matrices")
    if (
        a.shape[1] != b.shape[1]
        or ca.shape[1] != cb.shape[1]
        or a.shape[0] != ca.shape[0]
        or b.shape[0] != cb.shape[0]
    ):
        raise ValueError("incompatible numeric/category feature layouts")
    if a.dtype not in {torch.float32, torch.float64} or a.dtype != b.dtype:
        raise ValueError("numeric inputs must share float32 or float64 dtype")
    if ca.dtype != torch.int64 or cb.dtype != torch.int64:
        raise ValueError("category codes must use int64")
    if any(x.device != a.device or not x.is_contiguous() for x in (a, b, ca, cb)):
        raise ValueError("distance inputs must be contiguous on one device")


def _cpu_tile(a: Tensor, b: Tensor, ca: Tensor, cb: Tensor) -> Tensor:
    """Validate and call the lazily loaded native CPU squared-distance operator."""
    _validate(a, b, ca, cb)
    from tabgen_eval.execution.torch.cpu.extension import library

    return library().distance_tile(a, b, ca, cb)


@torch.library.custom_op("tabgen_eval::squared_distance_tile", mutates_args=(), device_types="cpu")
def squared_distance_tile(a: Tensor, b: Tensor, ca: Tensor, cb: Tensor) -> Tensor:
    """Compute direct squared distances for one caller-bounded tile.

    Parameters
    ----------
    a, b : Tensor
        Contiguous FP32/FP64 matrices ``[Q, PN]`` and ``[R, PN]``.
    ca, cb : Tensor
        Contiguous int64 matrices ``[Q, PC]`` and ``[R, PC]``. Negative
        identities encode unknown all-zero one-hot vectors. All inputs share a device.

    Returns
    -------
    Tensor
        ``[Q, R]`` distances on the input CPU/CUDA device and numeric dtype.

    Notes
    -----
    The operator does not mutate inputs. CPU uses ATen/C++; CUDA uses direct
    Triton differences with IEEE precision. The fake kernel returns only matching
    output metadata; the caller owns tile-size and temporary-memory limits.
    """
    return _cpu_tile(a, b, ca, cb)


@squared_distance_tile.register_kernel("cuda")
def _cuda_tile(a: Tensor, b: Tensor, ca: Tensor, cb: Tensor) -> Tensor:
    """Validate inputs and launch the CUDA bounded full-tile implementation."""
    _validate(a, b, ca, cb)
    return cuda_kernels().distance_tile(a, b, ca, cb)


@squared_distance_tile.register_fake
def _fake_tile(a: Tensor, b: Tensor, ca: Tensor, cb: Tensor) -> Tensor:
    """Validate layout and return only ``[Q, R]`` output metadata for tracing."""
    _validate(a, b, ca, cb)
    return a.new_empty((a.shape[0], b.shape[0]))


@torch.library.custom_op("tabgen_eval::nearest_squared_tile", mutates_args=(), device_types="cpu")
def nearest_squared_tile(a: Tensor, b: Tensor, ca: Tensor, cb: Tensor) -> Tensor:
    """Reduce each query against one reference tile without mutating inputs.

    Parameters
    ----------
    a, b, ca, cb : Tensor
        Same contiguous numeric/category contract as ``squared_distance_tile``.

    Returns
    -------
    Tensor
        ``[Q]`` minima in the numeric dtype/device; empty references produce infinity.

    Notes
    -----
    CUDA retains only block minima, while CPU reduces a bounded distance tile.
    The fake kernel returns the same one-dimensional metadata without computing.
    """
    _validate(a, b, ca, cb)
    if b.shape[0] == 0:
        return a.new_full((a.shape[0],), float("inf"))
    return _cpu_tile(a, b, ca, cb).amin(dim=1)


@nearest_squared_tile.register_kernel("cuda")
def _cuda_nearest(a: Tensor, b: Tensor, ca: Tensor, cb: Tensor) -> Tensor:
    """Validate and reduce CUDA reference blocks to one squared minimum per query."""
    _validate(a, b, ca, cb)
    return cuda_kernels().nearest_tile(a, b, ca, cb)


@nearest_squared_tile.register_fake
def _fake_nearest(a: Tensor, b: Tensor, ca: Tensor, cb: Tensor) -> Tensor:
    """Validate layout and return only ``[Q]`` minimum-output metadata."""
    _validate(a, b, ca, cb)
    return a.new_empty((a.shape[0],))


@torch.library.custom_op(
    "tabgen_eval::nearest_squared_self_tile", mutates_args=(), device_types="cpu"
)
def nearest_squared_self_tile(
    a: Tensor, b: Tensor, ca: Tensor, cb: Tensor, query_start: int, reference_start: int
) -> Tensor:
    """Squared minima excluding equal global row positions in one self-search tile.

    Tile coordinates obey ``squared_distance_tile``. Nonnegative offsets identify
    each tile's position in the same parent table. Only equal positions are masked;
    equal-valued records elsewhere remain valid neighbors. Inputs remain read-only.
    """
    _validate(a, b, ca, cb)
    if query_start < 0 or reference_start < 0:
        raise ValueError("self-search row offsets must be nonnegative")
    if not b.shape[0]:
        return a.new_full((a.shape[0],), float("inf"))
    distances = _cpu_tile(a, b, ca, cb)
    start = max(query_start, reference_start)
    stop = min(query_start + len(a), reference_start + len(b))
    if start < stop:
        positions = torch.arange(start, stop, device=a.device)
        distances[positions - query_start, positions - reference_start] = float("inf")
    return distances.amin(dim=1)


@nearest_squared_self_tile.register_kernel("cuda")
def _cuda_self_nearest(
    a: Tensor, b: Tensor, ca: Tensor, cb: Tensor, query_start: int, reference_start: int
) -> Tensor:
    """Reduce CUDA self-search partials with positional exclusion inside the kernel."""
    _validate(a, b, ca, cb)
    if query_start < 0 or reference_start < 0:
        raise ValueError("self-search row offsets must be nonnegative")
    return cuda_kernels().nearest_self_tile(a, b, ca, cb, query_start, reference_start)


@nearest_squared_self_tile.register_fake
def _fake_self_nearest(
    a: Tensor, b: Tensor, ca: Tensor, cb: Tensor, query_start: int, reference_start: int
) -> Tensor:
    """Validate row-offset metadata and return only the query-minimum shape."""
    _validate(a, b, ca, cb)
    if query_start < 0 or reference_start < 0:
        raise ValueError("self-search row offsets must be nonnegative")
    return a.new_empty((a.shape[0],))


def _validate_radii(a: Tensor, b: Tensor, ra: Tensor, rb: Tensor) -> None:
    """Require contiguous squared radii aligned with each numeric feature matrix.

    ``ra``/``rb`` have lengths Q/R and match their features' dtype and device.
    Raise ``ValueError`` on metadata mismatch; numerical radius values are not scanned.
    """
    if (
        ra.shape != (a.shape[0],)
        or rb.shape != (b.shape[0],)
        or ra.dtype != a.dtype
        or rb.dtype != a.dtype
        or ra.device != a.device
        or rb.device != a.device
        or not ra.is_contiguous()
        or not rb.is_contiguous()
    ):
        raise ValueError(
            "radii must be contiguous and match the feature rows, device and numeric dtype"
        )


@torch.library.custom_op("tabgen_eval::prdc_cross_tile", mutates_args=(), device_types="cpu")
def prdc_cross_tile(
    a: Tensor, b: Tensor, ca: Tensor, cb: Tensor, ra: Tensor, rb: Tensor, components: int = 15
) -> tuple[Tensor, Tensor, Tensor]:
    """Reduce strict PRDC ball membership for a bounded cross tile.

    Parameters
    ----------
    a, b, ca, cb : Tensor
        Same numeric/category contract as ``squared_distance_tile``.
    ra, rb : Tensor
        Contiguous squared radii ``[Q]``/``[R]`` in the numeric dtype/device.

    Returns
    -------
    tuple of Tensor
        Int64 real-ball counts ``[R]``, bool recall ``[Q]`` and bool coverage
        ``[Q]``. Distances use strict ``<`` comparisons in the selected precision.

    Notes
    -----
    Inputs are read-only. CUDA fuses counts/flags without retaining all distances;
    fake dispatch reproduces only the three output shapes, dtypes and device.
    """
    _validate_radii(a, b, ra, rb)
    distances = _cpu_tile(a, b, ca, cb)
    inside = distances < ra[:, None] if components & 13 else None
    counts = inside.sum(dim=0) if components & 5 else a.new_zeros((len(b),), dtype=torch.int64)
    recall = (
        (distances < rb[None, :]).any(dim=1)
        if components & 2
        else a.new_zeros((len(a),), dtype=torch.bool)
    )
    cover = inside.any(dim=1) if components & 8 else a.new_zeros((len(a),), dtype=torch.bool)
    return counts, recall, cover


@prdc_cross_tile.register_kernel("cuda")
def _cuda_cross(
    a: Tensor, b: Tensor, ca: Tensor, cb: Tensor, ra: Tensor, rb: Tensor, components: int = 15
) -> tuple[Tensor, Tensor, Tensor]:
    """Validate feature/radius metadata and launch fused CUDA PRDC cross reductions."""
    _validate(a, b, ca, cb)
    _validate_radii(a, b, ra, rb)
    return cuda_kernels().cross_tile(a, b, ca, cb, ra, rb, components)


@prdc_cross_tile.register_fake
def _fake_cross(
    a: Tensor, b: Tensor, ca: Tensor, cb: Tensor, ra: Tensor, rb: Tensor, components: int = 15
) -> tuple[Tensor, Tensor, Tensor]:
    """Return int64 counts ``[R]`` and bool recall/coverage ``[Q]`` metadata only."""
    _validate(a, b, ca, cb)
    _validate_radii(a, b, ra, rb)
    return (
        a.new_empty((b.shape[0],), dtype=torch.int64),
        a.new_empty((a.shape[0],), dtype=torch.bool),
        a.new_empty((a.shape[0],), dtype=torch.bool),
    )
