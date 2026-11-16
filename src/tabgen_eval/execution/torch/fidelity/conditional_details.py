"""Host table collection for reference-defined Conditional slices."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from tabgen_eval.execution.runtime.details import DetailTable, FidelityDetails

from .details import host_array

if TYPE_CHECKING:
    from torch import Tensor
    from torch import dtype as TorchDtype

    from tabgen_eval._typing import FloatArray


class ConditionalDetails:
    """Stream existing group results into host tables without device detail buffers.

    Group plans may reorder slices by padded size. Their original group IDs
    restore the reference-label ordering before scores enter the returned table.
    Count and boundary arrays come from the same grouping/binning pass as the
    scalar computation; no source rows are inspected again.
    """

    def __init__(
        self,
        width: int,
        numeric: int,
        names: Sequence[str],
        minimum: int,
        rare: float,
        dtype: TorchDtype,
    ) -> None:
        self.width, self.numeric = width, numeric
        self.names, self.minimum, self.rare = names, minimum, rare
        self.dtype = np.dtype(str(dtype).removeprefix("torch."))
        self.parts: list[dict[str, NDArray[np.generic]]] = []

    def add(
        self,
        cond: int,
        labels: Tensor,
        metadata: Sequence[Sequence[int]],
        total_r: int,
        total_s: int,
        group_ids: Sequence[int] | None,
        scores: FloatArray | None,
        edges: Tensor | None,
    ) -> None:
        """Retain one conditioning column, including excluded reference groups."""
        codes = host_array(labels)
        counts = np.asarray(metadata, dtype=np.int64).reshape(-1, 2)
        size = len(codes)
        values = np.full((size, 3), np.nan, dtype=self.dtype)
        if group_ids:
            assert scores is not None
            values[np.asarray(group_ids, dtype=np.int64)] = scores
        if self.width < 2:
            values[:, 0] = np.nan
        if self.width < 3:
            values[:, 1] = np.nan
        lower = np.full(size, np.nan, dtype=self.dtype)
        upper = np.full(size, np.nan, dtype=self.dtype)
        left = np.zeros(size, dtype=bool)
        right = np.zeros(size, dtype=bool)
        if cond < self.numeric:
            assert edges is not None
            fitted = host_array(edges)
            lower, upper = fitted[codes], fitted[codes + 1]
            left = codes == 0
            right[:] = True
        self.parts.append(
            dict(
                column_index=np.full(size, cond, dtype=np.int64),
                label_code=codes,
                reference_count=counts[:, 0],
                candidate_count=counts[:, 1],
                reference_total=np.full(size, total_r, dtype=np.int64),
                candidate_total=np.full(size, total_s, dtype=np.int64),
                eligible=counts[:, 0] >= self.minimum,
                rare=counts[:, 0].astype(np.float64) / total_r < self.rare,
                shape=values[:, 0],
                trend=values[:, 1],
                recovery=values[:, 2],
                lower=lower,
                upper=upper,
                left_closed=left,
                right_closed=right,
            )
        )

    def finish(self, columns: Tensor) -> FidelityDetails:
        """Return descriptive host tables while leaving scalar reductions intact."""
        host = host_array(columns)
        if self.width < 2:
            host[:, [0, 1]] = np.nan
        if self.width < 3:
            host[:, [2, 3]] = np.nan
        column_table = DetailTable(
            dict(
                column_index=np.arange(self.width, dtype=np.int64),
                **{name: host[:, index] for index, name in enumerate(self.names)},
            )
        )
        types = dict(
            column_index=np.int64,
            label_code=np.int64,
            reference_count=np.int64,
            candidate_count=np.int64,
            reference_total=np.int64,
            candidate_total=np.int64,
            eligible=bool,
            rare=bool,
            shape=self.dtype,
            trend=self.dtype,
            recovery=self.dtype,
            lower=self.dtype,
            upper=self.dtype,
            left_closed=bool,
            right_closed=bool,
        )
        slices = {
            name: (
                np.concatenate([part[name] for part in self.parts])
                if self.parts
                else np.empty(0, dtype=dtype)
            )
            for name, dtype in types.items()
        }
        return FidelityDetails({"columns": column_table, "slices": DetailTable(slices)})
