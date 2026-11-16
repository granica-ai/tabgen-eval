"""Bounded host transfers of existing fidelity score and grouping buffers."""

from __future__ import annotations

import numpy as np
from torch import Tensor

from tabgen_eval._typing import BoolArray, FloatArray, IntArray


def host_array(values: Tensor, max_bytes: int = 64 * 1024**2) -> FloatArray | IntArray | BoolArray:
    """Copy existing point-score buffers to owned host storage in bounded chunks.

    No device-side packing or score reordering is introduced. Keeping the
    original workspace and group-plan choices preserves numerical reductions.
    Transfers are bounded by rows of a vector or matrix, not individual slices.
    """
    if values.numel() * values.element_size() <= max_bytes:
        host = values.cpu().numpy()
        return host.copy() if values.device.type == "cpu" else host
    shape = tuple(values.shape)
    dtype = np.dtype(str(values.dtype).removeprefix("torch."))
    result = np.empty(shape, dtype=dtype)
    row_bytes = values.element_size() * int(np.prod(shape[1:], dtype=np.int64))
    batch = max(1, max_bytes // max(row_bytes, 1))
    for start in range(0, len(values), batch):
        result[start : start + batch] = values[start : start + batch].cpu().numpy()
    return result
