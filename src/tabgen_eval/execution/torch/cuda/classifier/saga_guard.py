"""Fixed training-only objective probe for bounded native SAGA backoff.

Probe rows come from an independent private NumPy generator and reuse the fold's
training-fitted transforms. Held-out labels and AUC never affect acceptance.
The fitter owns rollback checkpoints and rate changes; this module supplies
the scalar decision and counters only. Native updates return new state tensors,
so fit checkpoints retain references rather than cloning a full scalar cache.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

from tabgen_eval.execution.torch.cuda.classifier.data import Design

if TYPE_CHECKING:
    from tabgen_eval.execution.torch.cuda.classifier.data import Fold

PROBE_ROWS = 4096
MAX_BACKOFFS = 4


def reserve_bytes(design: Design) -> int:
    """Estimate probe, gather scratch, and retained checkpoint bytes for a design.

    Use at most 4,096 training rows, declared floating item size, compact identity
    width, and one retained table plus two D-vectors. Return an integer allowance
    without allocating or changing the design; the SAGA adapter subtracts it
    before deciding whether a schedule chunk fits its remaining workspace.
    """
    rows = min(PROBE_ROWS, design.n)
    item = design.numbers.element_size()
    probe = rows * (design.numeric * item + design.identities.shape[1] * 8 + 8 * item + 8)
    return probe + (design.n + 2 * design.d) * item


class ObjectiveGuard:
    """Own a private fixed probe and the last accepted objective checkpoint.

    The probe has at most 4,096 training observations and preserves full fitted
    feature identities/scales. It is an empirical divergence guard, not a full-
    data convergence test or monotonicity certificate. ``previous`` starts at
    log(2), the exact zero-weight average logistic objective. The fitter updates
    accepted state and replay/backoff counters; no global RNG is consumed.
    """

    def __init__(self, fold: Fold, seed: int, ridge: float) -> None:
        """Gather a reproducible probe using a private seed-domain-separated sampler.

        ``fold`` supplies training labels/design and ``ridge`` the full-training
        feature penalty. Gather new numeric/identity/label buffers and share only
        read-only categorical transform tensors. No held-out row is inspected.
        """
        design = fold.train
        rng = np.random.default_rng(int(seed) ^ 0xC19E)
        rows = torch.as_tensor(
            rng.choice(design.n, min(PROBE_ROWS, design.n), replace=False),
            device=design.numbers.device,
        )
        self.design = Design(
            design.numbers[rows], design.identities[rows], design.centers, design.scales
        )
        self.labels = fold.labels[rows]
        self.ridge = ridge
        self.previous = math.log(2.0)  # Exact initial zero-weight objective.
        self.checks = self.backoffs = self.rejected_epochs = 0

    def objective(self, weights: torch.Tensor) -> float:
        """Return probe mean logistic loss plus full feature ridge for ``weights[D]``.

        Weights and fitted probe buffers are read-only. Allocate logits and stable
        softplus losses on their CUDA device, then synchronize only the host scalar.
        The final intercept is excluded from ridge. Only retained probe rows are
        evaluated; no additional scan of source rows occurs.
        """
        logits = self.design.mv(weights)
        loss = (F.softplus(logits) - self.labels * logits).mean()
        # The final coefficient is the unpenalized intercept.
        return float((loss + 0.5 * self.ridge * weights[:-1].square().sum()).item())

    def rejects(self, objective: float, finite: bool) -> bool:
        """Count a check and flag nonfinite or excessively increased probe objective.

        ``finite`` is the native epoch's finite-state flag. Reject an increase above
        ``previous + max(1e-4, 0.02 * abs(previous))``; do not change the accepted
        objective, restore tensors, or halve a rate here. The fit controller owns
        those actions and enforces the shared four-backoff budget.
        """
        self.checks += 1
        return (
            not finite
            or not math.isfinite(objective)
            or objective > self.previous + max(1e-4, 0.02 * abs(self.previous))
        )

    def diagnostics(self) -> dict[str, int | float]:
        """Snapshot serializable probe/checkpoint counters without resetting them.

        Return probe size, checks, retries, rejected epochs, and the last accepted
        objective. These diagnostics describe guard activity rather than held-out
        quality or an exact optimization convergence measure.
        """
        return dict(
            backoff_probe_rows=self.design.n,
            backoff_probe_checks=self.checks,
            backoff_retries=self.backoffs,
            backoff_retry_limit=MAX_BACKOFFS,
            rejected_epochs=self.rejected_epochs,
            final_probe_objective=self.previous,
        )
