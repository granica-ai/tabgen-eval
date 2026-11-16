"""Batched CUDA marginal comparisons for Shape.

Numeric columns retain their own finite observations, sort in the selected
precision, and use the grouped Triton KS reduction. Categorical counts remain
int64 until probability normalization. Actual missing categories (-1) are
excluded; a literal ``"nan"`` category is retained. Column batches bound scratch
memory independently of the number of bootstrap draws.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from .grouped_ops import group_ks

if TYPE_CHECKING:
    from tabgen_eval.execution.torch.context import TorchContext
    from tabgen_eval.execution.torch.tensors import TensorTable


class ShapeWorkspace:
    """Account for gathered draws and conservatively bound per-column scratch.

    Parameters
    ----------
    context : TorchContext
        Prepared CUDA inputs and temporary workspace budget. Persistent input
        pools are accounted for separately by runtime routing.
    rows : tuple of int
        Reference and comparison draw sizes, including repeated source rows.

    Raises
    ------
    MemoryError
        The draw buffers and one column's preprocessing exceed the budget.
    """

    def __init__(self, context: TorchContext, rows: Sequence[int]) -> None:
        """Reserve draw ownership and sorting/histogram scratch before allocation.

        ``rows`` contains the two source-draw lengths, including duplicates. Numeric
        item size comes from the prepared FP32/FP64 pool; category identities and
        sort indices are charged at integer widths. The object stores byte counts,
        not device buffers, and checks that each present column family can run.
        """
        self.rows = sum(rows)
        self.element = context.reference.numeric.element_size()
        self.reserved_bytes = (
            self.rows * (context.n_numeric * self.element + context.n_categorical * 8 + 9) + 32768
        )
        self.available = context.gpu_workspace_mb * 1024**2 - self.reserved_bytes - 1024**2
        if self.available < 0:
            raise MemoryError(
                "Shape draws and preprocessing exceed the resolved workspace allowance"
            )
        if context.n_numeric:
            self.numeric_batch()
        if context.n_categorical:
            self.categorical_batch(max(context.inputs.cardinalities, default=0))

    def numeric_batch(self) -> int:
        # Selected-precision copies/values, masks, sort indices and a generous
        # allowance for ATen sort scratch plus Triton blockwise KS maxima.
        """Return at most 32 numeric columns fitting copies, sorting and KS partials.

        The bound includes both draw lengths and selected-precision item size;
        it does not assume that allocator reuse makes live sort scratch free.
        """
        per_column = self.rows * (6 * self.element + 64) + 4096
        return self._batch(per_column)

    def categorical_batch(self, levels: int) -> int:
        # Transposed codes, safe indices, exact integer scatter sources/counts,
        # validity masks, and selected-precision probability/reduction buffers.
        """Return the column batch fitting an exact histogram with ``levels`` cells.

        A batch shares its largest category domain. Accounting includes int64
        counts and FP32/FP64 normalized probabilities, so wide vocabulary sizes
        reduce the batch independently of the number of source rows.
        """
        per_column = self.rows * 40 + levels * (16 + 4 * self.element) + 4096
        return self._batch(per_column)

    def _batch(self, per_column: int) -> int:
        """Clamp a byte-derived column capacity to [1,32], or reject one-column overflow.

        ``per_column`` includes transient reductions; ``available`` already excludes
        owned draws and fixed overhead. No device allocation occurs here.
        """
        if per_column > self.available:
            raise MemoryError("one Shape column exceeds the resolved workspace allowance")
        return max(1, min(32, self.available // max(per_column, 1)))


def _ordered(values: Tensor) -> tuple[Tensor, Tensor]:
    """Sort finite cells in owned ``[P,N]`` columns for grouped KS.

    Return values ``[1,P,N]`` with nonfinite cells replaced by positive infinity,
    and exact int64 prefix lengths ``[1,P]``. The singleton group axis reuses the
    Conditional kernel; source values retain dtype/device and are not mutated.
    """
    valid = torch.isfinite(values)
    counts = valid.sum(dim=1, dtype=torch.int64).unsqueeze(0)
    ordered = torch.where(valid, values, float("inf")).sort(dim=1).values
    return ordered.unsqueeze(0), counts


def _counts(codes: Tensor, levels: int) -> Tensor:
    """Count nonnegative codes from contiguous int64 ``[P,N]`` into ``[P,levels]``.

    Negative missing codes contribute zero at a clamped safe index. The caller
    ensures all valid codes fit ``levels``. Scatter mutates only a newly allocated
    integer histogram, keeping probabilities out of the counting stage.
    """
    counts = codes.new_zeros((codes.shape[0], levels))
    counts.scatter_add_(1, codes.clamp(min=0), (codes >= 0).to(torch.int64))
    return counts


def shape_scores(
    reference: TensorTable,
    candidate: TensorTable,
    cardinalities: Sequence[int],
    workspace: ShapeWorkspace,
) -> Tensor:
    """Compute one selected-precision marginal similarity per column.

    Parameters
    ----------
    reference, candidate : TensorTable
        Draw-local row-major CUDA tables, with FP32/FP64 numeric values and
        shared int64 categorical identities. Inputs are read-only.
    cardinalities : tuple of int
        Shared category domain sizes in categorical-column order.
    workspace : ShapeWorkspace
        Budget checked before gathering and before each column batch.

    Returns
    -------
    torch.Tensor
        One-dimensional numeric-then-categorical scores on the input device
        and in its numeric dtype. Uncomputable columns contain NaN. The caller
        takes the unweighted mean over finite scores.

    Notes
    -----
    No complete-row filter is applied. Missingness in another column does not
    discard a valid observation. Counts and KS search positions stay integer;
    CDFs, probabilities and reductions use the selected floating precision.
    The caller disables autocast and executes without gradient recording.
    """
    pn, pc = reference.numeric.shape[1], reference.categorical.shape[1]
    prototype = reference.numeric
    if not reference.n_rows or not candidate.n_rows:
        return prototype.new_full((pn + pc,), float("nan"))
    parts = []
    if pn:
        batch = workspace.numeric_batch()
        for start in range(0, pn, batch):
            a, na = _ordered(reference.numeric[:, start : start + batch].T.contiguous())
            b, nb = _ordered(candidate.numeric[:, start : start + batch].T.contiguous())
            parts.append(group_ks(a, b, na, nb).flatten())
    if pc:
        # One common grid for a small column batch avoids allocating a global
        # columns-by-cardinality matrix and keeps high-cardinality domains bounded.
        maximum = max(cardinalities, default=0)
        batch = workspace.categorical_batch(maximum)
        for start in range(0, pc, batch):
            levels = max(cardinalities[start : start + batch], default=0)
            if not levels:
                parts.append(prototype.new_full((min(batch, pc - start),), float("nan")))
                continue
            ac = _counts(reference.categorical[:, start : start + batch].T.contiguous(), levels)
            bc = _counts(candidate.categorical[:, start : start + batch].T.contiguous(), levels)
            na, nb = ac.sum(dim=1), bc.sum(dim=1)
            pa = ac.to(prototype.dtype) / na.clamp(min=1).to(prototype.dtype)[:, None]
            pb = bc.to(prototype.dtype) / nb.clamp(min=1).to(prototype.dtype)[:, None]
            score = (1 - 0.5 * (pa - pb).abs().sum(dim=1)).clamp(0, 1)
            parts.append(torch.where((na > 0) & (nb > 0), score, float("nan")))
    return torch.cat(parts) if parts else prototype.new_empty(0)
