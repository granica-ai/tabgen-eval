"""Validate execution option syntax without loading Torch, Triton or CUDA.

Metric constructors and evaluation sessions use this lightweight boundary before
choosing an implementation. Valid options do not imply that a requested device,
framework version or metric kernel is installed; runtime selection checks those
capabilities separately.
"""

from __future__ import annotations

import re
from numbers import Integral

DTYPES = ("float32", "float64")


def validate_execution_options(device: object, dtype: object) -> None:
    """Validate device and precision without probing numerical implementations."""
    if not isinstance(device, str) or not re.fullmatch(r"auto|cpu|cuda(?::\d+)?", device):
        raise ValueError("device must be 'auto', 'cpu', 'cuda', or 'cuda:<index>'")
    if dtype not in DTYPES:
        raise ValueError("dtype must be 'float32' or 'float64'")


def validate_options(device: object, dtype: object, gpu_workspace_mb: object) -> None:
    """Reject malformed device, precision and workspace requests.

    Parameters
    ----------
    device : str
        ``auto``, ``cpu``, ``cuda`` or ``cuda:<nonnegative index>``.
    dtype : str
        ``float32`` or ``float64`` for numerical execution.
    gpu_workspace_mb : numbers.Integral
        Positive temporary-workspace budget in MiB; booleans are rejected.

    Raises
    ------
    ValueError
        An option is outside its accepted syntax or range. No numerical backend
        is imported and no device availability probe is performed.
    """
    validate_execution_options(device, dtype)
    if (
        isinstance(gpu_workspace_mb, bool)
        or not isinstance(gpu_workspace_mb, Integral)
        or gpu_workspace_mb < 1
    ):
        raise ValueError("gpu_workspace_mb must be a positive integer")
