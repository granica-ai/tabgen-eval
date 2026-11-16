"""Choose automatic CPU/CUDA placement using profiled workload estimates.

``ExecutionSession`` constructs a host-only ``Workload`` after size caps and
preparation; ``TorchContext.select_device`` combines its crossover decision with
kernel availability, configured workspace and current-device free memory. The
thresholds reflect recorded CPU/A100 profiles and do not choose another user's
GPU or change a metric's statistical protocol.

Estimates only select a device: rows, fitting, folds and precision are unchanged.
Explicit devices bypass this crossover policy. Live executor workspace checks
remain authoritative after data-dependent preprocessing and category expansion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from tabgen_eval._typing import Precision

if TYPE_CHECKING:
    from tabgen_eval.metrics.base import Metric
    from tabgen_eval.metrics.concentration import Concentration


@dataclass(frozen=True)
class Workload:
    """Host-only estimates used to choose a numerical execution device.

    Attributes
    ----------
    n_eff : int
        Effective draw size after evaluation size caps, rather than source size.
    numeric, categorical : int
        Numeric and categorical column counts.
    cardinalities : tuple of int
        Shared encoded category domains, including observed string identities.
    reference_rows, candidate_rows : int
        Persistent source-pool sizes before sampling.
    complete_rows : int
        Estimated retained rows per draw for complete-row distance metrics.
    n_bootstrap : int
        Requested replicate count; zero denotes a point-only report.
    cuda_resident : bool
        Whether this evaluation already holds compatible CUDA storage on the
        current device, in the same precision and workspace configuration.
    shared_source : bool
        Both roles address one immutable source. Persistent storage is counted
        once; draw-local gathers remain separate.
    holdout_rows : int
        Optional independent holdout storage copied with the prepared source pools.
    draw_rows, usable_rows : tuple of int or None
        Selected and complete row counts by distance role, in reference/candidate/
        optional holdout order. Memorization supplies exact counts after its
        independent source selection. Other procedures use conservative equal draws.

    Notes
    -----
    These conservative estimates do not fit transforms, alter sampling or
    guarantee a data-dependent model will fit. Executors enforce live workspace
    bounds. No PyTorch import or device probe is needed to construct this class.
    """

    n_eff: int
    numeric: int
    categorical: int
    cardinalities: tuple[int, ...]
    reference_rows: int
    candidate_rows: int
    complete_rows: int
    n_bootstrap: int = 0
    cuda_resident: bool = False
    shared_source: bool = False
    holdout_rows: int = 0
    draw_rows: tuple[int, ...] | None = None
    usable_rows: tuple[int, ...] | None = None

    @property
    def columns(self) -> int:
        """Total number of source numeric and categorical features."""
        return self.numeric + self.categorical

    @property
    def pairs(self) -> int:
        """Number of unordered feature pairs used by the Trend workload proxy."""
        return self.columns * (self.columns - 1) // 2

    @property
    def encoded_width(self) -> int:
        """Approximate expanded classifier width, capped per category by draw size."""
        return self.numeric + sum(min(card + 1, self.n_eff) for card in self.cardinalities)

    def cuda_candidate(self, metric: Metric) -> tuple[bool, str]:
        """Compare metric-specific work proxies with measured device crossovers.

        Parameters
        ----------
        metric : Metric
            Metric name and requested precision. Decisions combine draw size,
            pairs, categories and pool reuse.

        Returns
        -------
        tuple of bool and str
            Whether CUDA is a candidate and an auditable routing reason. The caller
            must still check dependencies, supported kernels and memory availability.

        Notes
        -----
        Bootstrap work or existing compatible CUDA storage lowers some transfer-cost
        thresholds. This method imports no numerical backend and probes no device.
        """
        if metric.name == "linear_c2st" and metric.dtype == "float32":
            return True, "float32_classifier_cuda_preferred"
        if metric.name == "shape":
            threshold = 2_000_000 if self.n_bootstrap or self.cuda_resident else 5_000_000
            use = self.n_eff * self.columns >= threshold
        elif metric.name == "conditional_fidelity":
            use = self.n_eff >= 2000 and self.n_eff * self.columns**2 >= 50_000_000
        elif metric.name == "trend":
            use = self.n_eff * self.pairs >= 10_000_000
        elif metric.name == "linear_c2st":
            use = self.n_eff * self.encoded_width >= 500_000
        elif metric.name in {
            "prdc",
            "dcr",
            "near_match_rate",
            "replication_ratio",
            "concentration",
        }:
            threshold = 20_000_000 if self.n_bootstrap or self.cuda_resident else 100_000_000
            rows = self.usable_rows
            pairs = (
                rows[0] * sum(rows[1:] if metric.name == "dcr" else rows[1:2])
                if rows is not None
                else self.complete_rows**2
            )
            use = pairs * max(self.columns, 1) >= threshold
        else:
            use = False
        return use, "workload_favors_cuda" if use else "small_workload_cpu"

    def minimum_workspace(self, metric: Metric) -> int:
        """Estimate minimum per-draw GPU buffers plus library/kernel scratch reserves.

        Parameters
        ----------
        metric : Metric
            Name and precision used to size numeric cells and family-specific
            preprocessing, category, index and state buffers.

        Returns
        -------
        int
            Estimated temporary bytes, excluding persistent full source tensors.
            This is an early rejection/routing estimate, not a promise a fitted model
            or data-dependent slice will fit its actual algorithm workspace.

        Notes
        -----
        Index filtering uses every selected row. Memorization provides exact usable
        counts for gathered distance buffers; other draws conservatively retain all
        sampled rows. Executors subsequently enforce their live workspace bounds.
        """
        n, pn, pc = self.n_eff, self.numeric, self.categorical
        element = 4 if metric.dtype == "float32" else 8
        width = sum(self.cardinalities) + pc
        if metric.name == "shape":
            data = 2 * n * (pn * element + pc * 8 + 9) + 32768
            numeric = 2 * n * (6 * element + 64) + 4096 if pn else 0
            categorical = (
                2 * n * 40 + max(self.cardinalities, default=0) * (16 + 4 * element) + 4096
                if pc
                else 0
            )
            return data + max(numeric, categorical) + 1024**2
        if metric.name == "concentration":
            concentration = cast("Concentration", metric)
            return (
                2 * n * (pn * element + pc * 8 + pn + 32)
                + n * (len(concentration.fixed_radii) + len(concentration.radius_quantiles)) * 40
                + (concentration.radius_pairs * 40 if concentration.radius_quantiles else 0)
                + 65536
            )
        if metric.name == "linear_c2st":
            return 2 * n * (pn * element * 8 + pc * 32 + 128) + width * 64 + 65536 + 16 * 1024**2
        if metric.name == "conditional_fidelity":
            return 2 * n * (pn * element + pc * 8 + 160) + 65536 + 1024**2
        if metric.name == "trend":
            return (
                2 * n * (pn * element + pc * 8 + 1)
                + 2 * n * (pn + pc) * 8
                + n * (pn * (8 * element + 8) + 128)
                + ((n + 4095) // 4096 + 16) * pn * pn * max(element, 8)
                + 2 * n * 8
                + 32768
                + 1024**2
            )
        selected = self.draw_rows or (n, n)
        rows = self.usable_rows if self.usable_rows is not None else selected
        largest = max(rows, default=0)
        original_largest = max(selected, default=0)
        data = sum(rows) * (pn * element + pc * 8 + 1)
        vocab = sum(((min(card, rows[0]) * 8 + 511) // 512) * 512 for card in self.cardinalities)
        preprocessing = largest * (pn + (64 if pc else 0)) + pn * element * 4
        indices = sum(selected) * 8 + original_largest * 9
        state = sum(rows) * (element + 10) if metric.name == "prdc" else largest * (4 * element + 8)
        filtering = sum(selected) * 8 + original_largest * 32 + 16384
        return max(
            filtering,
            data + vocab + preprocessing + indices + state + 16384 + 1024**2,
        )

    def persistent_bytes(self, dtype: Precision) -> int:
        """Estimate full prepared source-pool tensor storage for one precision.

        Parameters
        ----------
        dtype : {"float32", "float64"}
            Validated numeric precision; category codes use eight bytes and complete
            masks one byte per source row regardless of this choice.

        Returns
        -------
        int
            Unique source tensor bytes, including an independent holdout, before
            draws are gathered.
            Routing adds these bytes only when compatible current-device storage is
            not already resident; tensor/container allocator overhead is excluded.
        """
        element = 4 if dtype == "float32" else 8
        return (
            self.reference_rows
            + (0 if self.shared_source else self.candidate_rows)
            + self.holdout_rows
        ) * (self.numeric * element + self.categorical * 8 + 1)
