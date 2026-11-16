"""Owned row-major tensor pools and lazy numerical dependency boundaries.

Host preparation stores contiguous source columns as ``(P, N)`` arrays. The
factories here copy/repack them into contiguous ``(N, P)`` tensors, retain integer
category identities and select numeric precision exactly once. No metric
scaling, binning, fitting or source-row sampling occurs during transfer.

Executors treat tables as read-only. Frozen containers prevent attribute
replacement but cannot make Torch storage intrinsically immutable; arbitrary
constructor inputs may alias caller tensors. Factory-created tables own their
storage and transfers/gathers return fresh storage. Torch and CUDA are only
loaded/probed when a relevant helper is called.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING, Literal, TypeGuard, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tabgen_eval._typing import Precision
from tabgen_eval.execution.runtime.options import DTYPES

if TYPE_CHECKING:
    from torch import Tensor
    from torch import device as TorchDevice
    from torch import dtype as TorchDType

    from tabgen_eval.preprocessing.columns import HostTable, PreparedColumns


@overload
def torch_module(*, required: Literal[True] = True) -> ModuleType: ...


@overload
def torch_module(*, required: Literal[False]) -> ModuleType | None: ...


@overload
def torch_module(*, required: bool) -> ModuleType | None: ...


def torch_module(*, required: bool = True) -> ModuleType | None:
    """Import Torch lazily while distinguishing absence from a broken installation.

    Parameters
    ----------
    required : bool, default=True
        If false, return ``None`` only when the top-level ``torch`` package itself
        is absent. Otherwise an absent package produces an actionable ImportError.

    Returns
    -------
    module or None
        Imported Torch module, or the permitted missing-dependency sentinel.

    Raises
    ------
    ImportError
        Required Torch is absent, or an installed package/transitive dependency
        fails to import. Errors inside an installed Torch are never suppressed.
    """
    try:
        return importlib.import_module("torch")
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        if not required:
            return None
        raise ImportError(
            "PyTorch execution requires torch; install 'tabgen-eval[torch]'."
        ) from exc


def resolve_device(device: str | TorchDevice) -> TorchDevice:
    """Resolve supported tensor storage placement without changing caller devices.

    Parameters
    ----------
    device : str or torch.device
        CPU or CUDA destination; ``auto`` chooses available CUDA, otherwise CPU.
        An unindexed CUDA request uses the caller's current CUDA device.

    Returns
    -------
    torch.device
        Canonical CPU device or CUDA device with an explicit available index.

    Raises
    ------
    ValueError
        Device syntax/type/index is unsupported, including indexed CPU devices.
    RuntimeError
        A CUDA destination was requested but CUDA is unavailable.

    Notes
    -----
    This storage helper does not use metric crossover or workspace routing and
    may initialize/probe CUDA. It never selects a different current device for
    other code or changes streams, precision settings or metric algorithms.
    """
    torch = torch_module()
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        resolved = torch.device(device)
    except (RuntimeError, ValueError, TypeError) as exc:
        raise ValueError("device must be 'auto', 'cpu', 'cuda', or 'cuda:<index>'") from exc
    if resolved.type not in {"cpu", "cuda"} or (
        resolved.type == "cpu" and resolved.index is not None
    ):
        raise ValueError("only 'cpu' and CUDA devices are supported")
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device {device!r} requested, but CUDA is unavailable")
        index = torch.cuda.current_device() if resolved.index is None else resolved.index
        if index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index {index} is unavailable")
        resolved = torch.device("cuda", index)
    return resolved


def numeric_dtype(dtype: Precision) -> TorchDType:
    """Resolve the accepted FP32/FP64 spelling after validating it.

    Parameters
    ----------
    dtype : {"float32", "float64"}
        Explicit floating arithmetic/storage precision.

    Returns
    -------
    torch.dtype
        Corresponding dtype from a lazily imported Torch module.

    Raises
    ------
    ValueError
        A spelling outside the two supported precisions was supplied.
    """
    if dtype not in DTYPES:
        raise ValueError("dtype must be 'float32' or 'float64'")
    return getattr(torch_module(), dtype)


def _is_tensor(value: object) -> TypeGuard[Tensor]:
    """Narrow tensors while preserving the lazy Torch import boundary."""
    return isinstance(value, torch_module().Tensor)


@dataclass(frozen=True)
class TensorTable:
    """Validated contiguous tensors for one fixed source pool.

    Attributes
    ----------
    numeric : torch.Tensor
        Row-major FP32/FP64 matrix ``(N, P_numeric)``; missing numeric cells remain
        available for metric-specific pairwise, complete-case or imputation rules.
    categorical : torch.Tensor
        Row-major int64 matrix ``(N, P_categorical)`` with shared category identities
        and ``-1`` for actual missing values.
    complete : torch.Tensor
        Boolean vector ``(N,)`` copied from full-frame host completeness.

    Notes
    -----
    All fields share a device, require no gradients and must be contiguous.
    Construction validates this boundary but does not copy supplied tensors or
    validate category-code ranges against a vocabulary. Factories own their
    storage; callers/executors must treat all source tensors as read-only.
    """

    numeric: Tensor
    categorical: Tensor
    complete: Tensor

    def __post_init__(self) -> None:
        """Reject mismatched row counts, devices, precision, layout or autograd state."""
        torch = torch_module()
        if not all(
            isinstance(t, torch.Tensor) for t in (self.numeric, self.categorical, self.complete)
        ):
            raise TypeError("prepared table fields must be tensors")
        if (
            self.numeric.ndim != 2
            or self.categorical.ndim != 2
            or self.complete.ndim != 1
            or self.numeric.shape[0] != self.complete.shape[0]
            or self.categorical.shape[0] != self.complete.shape[0]
        ):
            raise ValueError("prepared tensors must have matching row counts")
        if self.numeric.dtype not in {torch.float32, torch.float64}:
            raise ValueError("numeric tensors must use float32 or float64")
        if self.categorical.dtype != torch.int64 or self.complete.dtype != torch.bool:
            raise ValueError("category codes must be int64 and complete-row masks must be bool")
        if len({t.device for t in (self.numeric, self.categorical, self.complete)}) != 1:
            raise ValueError("prepared tensors must share a device")
        if any(
            t.requires_grad or not t.is_contiguous()
            for t in (self.numeric, self.categorical, self.complete)
        ):
            raise ValueError("prepared tensors must be contiguous and require no gradients")

    @property
    def n_rows(self) -> int:
        """Number of source positions, retaining incomplete and repeated rows."""
        return len(self.complete)

    @property
    def nbytes(self) -> int:
        """Sum field storage sizes in bytes, excluding tensor/container overhead."""
        return sum(
            t.numel() * t.element_size() for t in (self.numeric, self.categorical, self.complete)
        )

    @classmethod
    def from_host(
        cls,
        table: HostTable,
        *,
        device: str | TorchDevice,
        dtype: TorchDType,
        copy: bool = True,
    ) -> TensorTable:
        """Repack feature-major host columns into row-major tensors.

        Parameters
        ----------
        table : HostTable
            Numeric/category arrays ``(P, N)`` and a boolean complete-row vector.
        device : str or torch.device
            Resolved CPU/CUDA storage destination.
        dtype : torch.dtype
            Selected numeric dtype; categorical and complete fields stay int64/bool.
        copy : bool, default=True
            Copy caller-owned inputs. Internal callers may transfer ownership of
            fresh compiled buffers with ``False``; executors never mutate them.

        Returns
        -------
        TensorTable
            Contiguous tensors with copied storage by default; source masks/codes
            and numeric NaNs survive.
        """
        torch = torch_module()

        def tensor(array: NDArray[np.generic], target_dtype: TorchDType) -> Tensor:
            """Copy public inputs or take ownership of fresh compiled buffers."""
            import warnings

            array_dtype = {
                torch.float32: np.float32,
                torch.float64: np.float64,
                torch.int64: np.int64,
                torch.bool: np.bool_,
            }[target_dtype]
            # Cast while packing the owned host buffer. A subsequent Torch
            # cast would retain an unnecessary full-width staging allocation.
            owned = (
                np.array(array, copy=True, order="C", dtype=array_dtype)
                if copy or target_dtype == torch.bool
                else np.ascontiguousarray(array, dtype=array_dtype)
            )
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
                result = torch.from_numpy(owned)
            return result.to(device=device)

        return cls(
            tensor(np.transpose(table.numeric), dtype),
            tensor(np.transpose(table.categorical), torch.int64),
            tensor(table.complete, torch.bool),
        )

    def to(self, *, device: str | TorchDevice, dtype: Precision) -> TensorTable:
        """Copy all fields to an explicitly resolved device and numeric precision.

        Parameters
        ----------
        device : str or torch.device
            Supported destination; ``auto`` uses storage availability selection.
        dtype : {"float32", "float64"}
            Numeric destination precision; other field dtypes do not change.

        Returns
        -------
        TensorTable
            Independently owned tensors even if device/precision already match.
            No fitted transforms or missingness filtering are applied.
        """
        resolved, kind = resolve_device(device), numeric_dtype(dtype)
        return type(self)(
            self.numeric.to(device=resolved, dtype=kind, copy=True),
            self.categorical.to(device=resolved, copy=True),
            self.complete.to(device=resolved, copy=True),
        )

    def row_indices(self, positions: ArrayLike | Tensor) -> Tensor:
        """Validate and place one draw's source positions before gathering.

        Parameters
        ----------
        positions : array-like or torch.Tensor
            One-dimensional integer indices in ``[0, n_rows)``. NumPy-compatible
            arrays accept signed/unsigned integer dtypes; tensors accept int32/int64.
            Empty draws are valid and duplicates/order are preserved.

        Returns
        -------
        torch.Tensor
            Int64 indices on the table's device. An already compatible tensor can
            share the caller's index storage; NumPy positions are copied.

        Raises
        ------
        ValueError
            Indices have an invalid shape/dtype or address positions outside the pool.
            GPU min/max validation can synchronize before numerical gathering.
        """
        torch = torch_module()
        if _is_tensor(positions):
            if positions.ndim != 1 or positions.dtype not in {torch.int32, torch.int64}:
                raise ValueError("draw positions must be a one-dimensional integer array")
            if positions.numel() and (
                positions.min().item() < 0 or positions.max().item() >= self.n_rows
            ):
                raise ValueError("draw index is outside the input pool")
            indices = positions.to(device=self.numeric.device, dtype=torch.int64)
        else:
            positions = np.asarray(positions)
            if positions.ndim != 1 or positions.dtype.kind not in "iu":
                raise ValueError("draw positions must be a one-dimensional integer array")
            if positions.size and (positions.min() < 0 or positions.max() >= self.n_rows):
                raise ValueError("draw index is outside the input pool")
            indices = torch.from_numpy(np.array(positions, dtype=np.int64, copy=True)).to(
                self.numeric.device
            )
        return indices

    def gather(self, positions: ArrayLike | Tensor) -> TensorTable:
        """Copy a single source draw in its requested order, including duplicates.

        Parameters
        ----------
        positions : array-like or torch.Tensor
            One-dimensional integer source positions validated by ``row_indices``.

        Returns
        -------
        TensorTable
            Contiguous owned draw-local storage. The complete-row mask is gathered,
            not applied; missingness filtering remains the metric's responsibility.
        """
        return self._gather_indices(self.row_indices(positions))

    def _gather_indices(self, indices: Tensor) -> TensorTable:
        """Gather trusted device-local indices with autograd disabled.

        Parameters
        ----------
        indices : torch.Tensor
            Previously validated int64 source positions on this table's device.

        Returns
        -------
        TensorTable
            Fresh contiguous numeric/category/mask storage in index order. This private
            path intentionally omits repeated bounds/type checks; callers must use
            ``row_indices`` before passing untrusted positions.
        """
        torch = torch_module()
        with torch.no_grad():
            return type(self)(
                *(
                    t.index_select(0, indices)
                    for t in (self.numeric, self.categorical, self.complete)
                )
            )


@dataclass(frozen=True)
class TensorInputs:
    """Compatible tensor source pools plus their shared column metadata.

    Attributes
    ----------
    reference, candidate : TensorTable
        Pools in the same floating precision and on one CPU/CUDA device; row
        counts may differ and neither pool is mutated during evaluation.
    numeric_names, categorical_names : tuple of str
        Ordered feature names matching the corresponding tensor widths.
    cardinalities : tuple of int
        Shared observed category domain sizes. Actual nulls have code ``-1``
        and do not occupy dictionary entries.
    category_mismatch : bool
        Whether the host pools had different observed nonmissing category sets.
    holdout : TensorTable or None
        Optional independent real pool sharing storage identities and precision.

    Notes
    -----
    Metadata describes storage identities rather than a learned feature map.
    Factories transfer owned tensors; the dataclass itself only validates shared
    shape/device/precision and category metadata boundaries.
    """

    reference: TensorTable
    candidate: TensorTable
    numeric_names: tuple[str, ...]
    categorical_names: tuple[str, ...]
    cardinalities: tuple[int, ...]
    category_mismatch: bool
    holdout: TensorTable | None = None

    def __post_init__(self) -> None:
        """Validate pool compatibility, feature widths and category metadata ranges."""
        tables = (self.reference, self.candidate) + (
            () if self.holdout is None else (self.holdout,)
        )
        if any(table.numeric.device != self.reference.numeric.device for table in tables):
            raise ValueError("input pools must share a device")
        if any(table.numeric.dtype != self.reference.numeric.dtype for table in tables):
            raise ValueError("input pools must share a numeric dtype")
        if any(
            table.numeric.shape[1] != len(self.numeric_names)
            or table.categorical.shape[1] != len(self.categorical_names)
            for table in tables
        ):
            raise ValueError("tensor widths must match column metadata")
        if len(self.cardinalities) != len(self.categorical_names):
            raise ValueError("category metadata must match categorical width")
        if any(card < 0 for card in self.cardinalities):
            raise ValueError("invalid category metadata")

    @classmethod
    def from_columns(
        cls,
        columns: PreparedColumns,
        *,
        device: str | TorchDevice = "cpu",
        dtype: Precision = "float64",
        copy: bool = True,
    ) -> TensorInputs:
        """Resolve placement/precision and copy each unique prepared source once.

        Parameters
        ----------
        columns : PreparedColumns
            Feature-major host arrays and shared categorical metadata.
        device : str, default="cpu"
            Storage destination; unindexed CUDA uses the caller's current device.
        dtype : {"float32", "float64"}, default="float64"
            Numeric precision selected for both pools.
        copy : bool, default=True
            Copy source buffers, or take ownership of fresh internal buffers.

        Returns
        -------
        TensorInputs
            Row-major owned tensors; column order, identities and mismatch metadata
            are retained. Real-only roles share one owned tensor table. This
            method does not perform workload-based routing.
        """
        resolved, kind = resolve_device(device), numeric_dtype(dtype)
        reference = TensorTable.from_host(columns.reference, device=resolved, dtype=kind, copy=copy)
        candidate = (
            reference
            if columns.reference is columns.candidate
            else TensorTable.from_host(columns.candidate, device=resolved, dtype=kind, copy=copy)
        )
        return cls(
            reference,
            candidate,
            columns.numeric_names,
            columns.categorical_names,
            columns.cardinalities,
            columns.category_mismatch,
            None
            if columns.holdout is None
            else TensorTable.from_host(columns.holdout, device=resolved, dtype=kind, copy=copy),
        )

    @property
    def device(self) -> TorchDevice:
        """Shared resolved Torch device of both source pools."""
        return self.reference.numeric.device

    @property
    def dtype(self) -> Precision:
        """Shared numeric precision as the string ``float32`` or ``float64``."""
        name = str(self.reference.numeric.dtype).removeprefix("torch.")
        return "float32" if name == "float32" else "float64"

    @property
    def nbytes(self) -> int:
        """Unique persistent tensor bytes, counting a shared immutable source once."""
        return (
            self.reference.nbytes
            + (0 if self.reference is self.candidate else self.candidate.nbytes)
            + (0 if self.holdout is None else self.holdout.nbytes)
        )

    def to(self, *, device: str | TorchDevice, dtype: Precision) -> TensorInputs:
        """Copy both pools to a destination while retaining all storage metadata.

        Parameters
        ----------
        device : str or torch.device
            Supported CPU/CUDA destination resolved independently of workload routing.
        dtype : {"float32", "float64"}
            Numeric precision for both copied pools.

        Returns
        -------
        TensorInputs
            Freshly owned source storage, even when the destination already matches.
            Row order, incomplete rows and category identities are unchanged.
        """
        reference = self.reference.to(device=device, dtype=dtype)
        candidate = (
            reference
            if self.reference is self.candidate
            else self.candidate.to(device=device, dtype=dtype)
        )
        return type(self)(
            reference,
            candidate,
            self.numeric_names,
            self.categorical_names,
            self.cardinalities,
            self.category_mismatch,
            None if self.holdout is None else self.holdout.to(device=device, dtype=dtype),
        )
