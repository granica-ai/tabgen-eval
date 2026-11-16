"""Shared identity-domain metadata and reference-draw-only CUDA membership."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import accumulate
from typing import TYPE_CHECKING

import torch

from tabgen_eval.execution.torch.distance.ops import cuda_kernels

if TYPE_CHECKING:
    from tabgen_eval.execution.torch.tensors import TensorTable

    from .geometry import DistanceWorkspace

MAX_CARDINALITY = 4096
MAX_PRESENCE_BYTES = 16 * 1024**2


class CategoryDomain:
    """Cache read-only global identity offsets, never draw-fitted membership.

    CUDA buffers are created only when an eligible draw actually fits presence.
    Large domains and draws whose workspace cannot hold presence retain sorted
    reference vocabularies. No category counts or membership are fitted here.
    """

    def __init__(self, cardinalities: Sequence[int], device: torch.device) -> None:
        """Cache domain sizes and conservative CUDA metadata allocation bounds.

        ``cardinalities`` are shared encoding sizes, not reference-draw membership.
        Presence is eligible only on CUDA with per-column sizes at most 4096 and a
        combined int32 presence buffer at most 16 MiB. No device buffers are allocated
        until ``fit`` finds sufficient draw workspace; unsupported domains use sorting.
        """
        self.cardinalities = tuple(map(int, cardinalities))
        self.device = device
        self.total = sum(self.cardinalities)
        self.presence_bytes = self.total * 4
        self.supported = (
            device.type == "cuda"
            and all(0 <= card <= MAX_CARDINALITY for card in self.cardinalities)
            and self.presence_bytes <= MAX_PRESENCE_BYTES
        )
        buckets: dict[int, list[int]] = {}
        if self.supported:
            for col, card in enumerate(self.cardinalities):
                if card:
                    bins = 1 << (card - 1).bit_length()
                    buckets.setdefault(bins, []).append(col)
        self._groups = tuple((bins, tuple(cols)) for bins, cols in sorted(buckets.items()))

        # CUDA allocation accounting rounds even tiny int64 arrays to 512 bytes.
        # Reserve metadata before lazy allocation so later draws stay bounded.
        def allocation(size: int) -> int:
            """Round metadata bytes to the CUDA allocator granularity used by this budget."""
            return ((size + 511) // 512) * 512

        self.metadata_bytes = (
            2 * allocation(len(self.cardinalities) * 8)
            + sum(allocation(len(cols) * 8) for _, cols in self._groups)
            if self.supported
            else 0
        )
        self._metadata: (
            tuple[torch.Tensor, torch.Tensor, tuple[tuple[int, torch.Tensor], ...]] | None
        ) = None

    def fit(
        self, reference: TensorTable, workspace: DistanceWorkspace | None
    ) -> torch.Tensor | None:
        """Fit fresh reference-only category presence within the draw workspace.

        ``reference.categorical`` contains int64 row-major shared identities; negative
        codes are ignored. Metadata may be cached on first use, but the returned int32
        presence buffer is new for every draw. Return ``None`` when unsupported or the
        workspace cannot reserve presence, allowing sorted-vocabulary fallback.
        """
        if not self.supported:
            return None
        assert workspace is not None
        if not workspace.reserve_categories(self.presence_bytes):
            return None
        if self._metadata is None:
            offsets = tuple(accumulate((0, *self.cardinalities)))[:-1]
            self._metadata = (
                torch.tensor(offsets, dtype=torch.int64, device=self.device),
                torch.tensor(self.cardinalities, dtype=torch.int64, device=self.device),
                tuple(
                    (bins, torch.tensor(cols, dtype=torch.int64, device=self.device))
                    for bins, cols in self._groups
                ),
            )
        offsets, cards, groups = self._metadata
        return cuda_kernels().category_presence(
            reference.categorical, offsets, cards, groups, self.total
        )

    def mask(self, table: TensorTable, presence: torch.Tensor) -> None:
        """Replace absent category identities with ``-1`` in an owned query draw.

        Call only after successful ``fit`` on this domain. ``presence`` is the fitted
        reference-draw buffer, not pooled membership; cached offsets/cardinalities remain
        read-only. ``table.categorical`` is mutated on its device's current stream.
        """
        assert self._metadata is not None
        offsets, cards, _ = self._metadata
        cuda_kernels().mask_categories(table.categorical, offsets, cards, presence)
