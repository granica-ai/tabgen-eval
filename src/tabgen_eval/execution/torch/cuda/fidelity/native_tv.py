"""Opt-in native orchestration of retained Triton histogram/TV functions.

This bridge deliberately depends on Triton's internal CompiledKernel loading
API. It validates runtime signatures, scratch requirements, and launch metadata
before borrowing CUDA function handles for C++ batching. Cached Python owners
keep their modules alive. Device, histogram grid, marginal mode, and precision
belong to the cache key; handles never move between devices.

Metric preparation and workspace policy remain in ``execution.torch.fidelity``. The
bridge loads/compiles on first use and performs no metric-specific sampling or
score aggregation; native code returns per-batch score tensors to its caller.
"""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

import torch

from tabgen_eval.execution.torch.cuda.extension import library

if TYPE_CHECKING:
    from triton.compiler.compiler import CompiledKernel


def _spec(kernel: CompiledKernel, expected: list[str]) -> list[int]:
    """Validate a compiled Triton runtime ABI and borrow its launch metadata.

    ``expected`` lists non-constexpr parameter type strings in operator order.
    Reject mismatched signatures, multi-CTA/cooperative/PDL launches, or nonzero
    global/profile scratch with RuntimeError. Initialize lazy handles, then return
    ``[CUfunction, num_warps, shared_bytes]``. The caller must retain the original
    CompiledKernel owner for the full native invocation and all asynchronous work.
    """
    actual = [kind for kind in kernel.src.signature.values() if kind != "constexpr"]
    if actual != expected:
        raise RuntimeError(f"unsupported native Conditional kernel ABI: {actual}")
    metadata = kernel.metadata
    if (
        metadata.num_ctas != 1
        or getattr(metadata, "launch_cooperative_grid", False)
        or getattr(metadata, "launch_pdl", False)
        or getattr(metadata, "global_scratch_size", 0)
        or getattr(metadata, "profile_scratch_size", 0)
    ):
        raise RuntimeError("unsupported native Conditional kernel launch metadata")
    kernel._init_handles()
    return [kernel.function, metadata.num_warps, metadata.shared]


@cache
def _compiled(
    device: torch.device, grid: int, marginal: bool, dtype: torch.dtype
) -> tuple[CompiledKernel, CompiledKernel, list[int], list[int]]:
    """Cache warmed histogram/TV owners and ABI specs for one device/grid/mode.

    ``device`` is a CUDA device, ``grid`` a supported histogram bucket size,
    ``marginal`` the compile-time reduction mode, and ``dtype`` FP32/FP64 output.
    Warmup compiles without launching, using nonspecialized i32 placeholders to
    establish native scalar signatures. Large grids choose the banked histogram.
    Return both CompiledKernel owners plus their three-integer launch specs.
    ``functools.cache`` retains these owners for the process lifetime and the
    current CUDA device is restored on exit.
    """
    from tabgen_eval.execution.torch.cuda.triton.fidelity_grouped import (
        _dense_tv,
        _joint,
        _joint_banked,
    )

    with torch.cuda.device(device):
        codes = torch.empty((1,), dtype=torch.int64, device=device)
        output = torch.empty((1,), dtype=dtype, device=device)
        # warmup compiles without launching. Representative non-specialized
        # signed integers establish the i32 runtime ABI verified below.
        histogram_kernel = _joint_banked if grid > 4096 else _joint
        histogram = histogram_kernel.warmup(
            codes,
            codes,
            codes,
            codes,
            17,
            3,
            7,
            grid,
            True,
            True,
            marginal,
            -1,
            grid=(1, 1, 1),
            num_warps=8 if grid > 4096 else 4,
        )
        reduction = _dense_tv.warmup(
            codes,
            codes,
            codes,
            codes,
            output,
            7,
            3,
            grid,
            -1,
            marginal,
            dtype == torch.float64,
            grid=(1, 1, 1),
            num_warps=8 if grid > 4096 else 4,
            enable_fp_fusion=False,
        )
        hist_spec = _spec(histogram, ["*i64"] * 4 + ["i32"] * 4)
        tv_spec = _spec(
            reduction,
            ["*i64"] * 4 + ["*fp64" if dtype == torch.float64 else "*fp32"] + ["i32"] * 3,
        )
    return histogram, reduction, hist_spec, tv_spec


def dense_tv_batches(
    a: torch.Tensor,
    b: torch.Tensor,
    pairs: torch.Tensor,
    levels: torch.Tensor,
    grid: int,
    batch: int,
    cond: int,
    marginal: bool,
    prototype: torch.Tensor,
) -> list[torch.Tensor]:
    """Compute dense conditional-TV pair batches through one native operator.

    Parameters
    ----------
    a, b : Tensor
        Contiguous aligned int64 CUDA category codes ``[G, P, N_real]`` and
        ``[G, P, N_generated]`` on one device. Inputs remain read-only.
    pairs : Tensor
        Contiguous int64 CUDA ``[Q, 2]`` column pairs for this histogram grid.
    levels : Tensor
        Contiguous int64 CUDA ``[G, P]`` per-draw category/domain widths.
    grid : int
        Supported power-of-two histogram bucket, at most 8192.
    batch : int
        Caller-selected pair count per workspace-bounded native batch, at most 1024.
    cond : int
        Conditioning column or -1; native code receives it as a runtime i32.
    marginal : bool
        Compile-time histogram/reduction mode shared with the Triton caller.
    prototype : Tensor
        Supplies FP32/FP64 score dtype/device; its values are not read.

    Returns
    -------
    list of Tensor
        Fresh contiguous ``[G, q]`` score tensors in original pair-batch order;
        the caller concatenates once across histogram grid buckets. Empty draws
        or pairs return one empty tensor without compiling native operators.

    Raises
    ------
    RuntimeError
        The selected library lacks the bridge, or Triton's ABI/launch metadata is
        unsupported. Native input validation also rejects unsafe shapes/alignment.

    Notes
    -----
    The caller owns memory policy, input lifetimes, and final aggregation. Native
    allocations/launches use the caller's stream and unchanged histogram/TV kernels;
    this bridge retains module owners through the cached compiled-kernel tuple.
    """
    if not len(a) or not len(pairs):
        return [prototype.new_empty((len(a), len(pairs)))]
    _, _, histogram, reduction = _compiled(a.device, grid, marginal, prototype.dtype)
    native = library()
    if not hasattr(native, "dense_tv_batches"):
        raise RuntimeError("native Conditional prototype requires a rebuilt CUDA extension")
    return native.dense_tv_batches(
        a, b, pairs, levels, prototype, grid, batch, cond, histogram, reduction
    )
