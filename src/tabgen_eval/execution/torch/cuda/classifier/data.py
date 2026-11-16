"""Training-only transforms and compact full-one-hot CUDA designs.

``prepare`` separates scoring rows before fitting imputation, numeric scaling,
and the union-training categorical vocabulary. Numeric columns use guarded
standardization; categorical and numerical-missingness indicators are unscaled.
``Design`` either owns a column-major dense matrix for direct Newton or compact
row-major numbers/identities for native SketchySAGA. Dense and implicit products
represent the same features and final unpenalized intercept.

Torch/Triton are imported by this implementation module; native host operators
are loaded only when products are requested. No dataset sampling is performed
here: bootstrap positions and source-stable folds come from the shared protocol.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from tabgen_eval.execution.torch.cuda.extension import library
from tabgen_eval.execution.torch.cuda.triton import classifier as kernels
from tabgen_eval.preprocessing import cuda as preprocessing

if TYPE_CHECKING:
    from tabgen_eval.execution.torch.tensors import TensorTable
    from tabgen_eval.metrics.c2st import LinearOptions


@dataclass
class Design:
    """Full-feature classifier design with optional implicit categoricals.

    Parameters
    ----------
    numbers : Tensor
        Scaled FP32/FP64 CUDA numeric matrix ``[N, PN]``.
    identities : Tensor
        CUDA int64 ``[N, PC]`` full-vocabulary feature identities, unknown ``-1``.
    centers, scales : Tensor
        ``[D_cat]`` training-fitted categorical scaling in the numeric dtype/device.
    dense : Tensor, optional
        Column-major ``[N, D]`` full design, including unpenalized final intercept.

    Notes
    -----
    Implicit products preserve full one-hot feature meaning. Inverse row lists
    support deterministic adjoints; they must be prepared before transpose products.
    """

    numbers: torch.Tensor
    identities: torch.Tensor
    centers: torch.Tensor
    scales: torch.Tensor
    dense: torch.Tensor | None = None
    inverse_rows: torch.Tensor | None = None
    inverse_offsets: torch.Tensor | None = None

    @property
    def n(self) -> int:
        """Number of observations in the numeric and identity buffers."""
        return len(self.numbers)

    @property
    def numeric(self) -> int:
        """Number of numeric features before the categorical feature block."""
        return self.numbers.shape[1]

    @property
    def d(self) -> int:
        """Total feature count, including full categorical indicators and intercept."""
        return self.numeric + len(self.centers) + 1

    def expand(self) -> None:
        """Materialize and retain the full column-major CUDA ``[N, D]`` design.

        Replace ``self.dense`` with a fresh allocation, preserving compact buffers.
        The caller must reserve dense expansion storage before invoking this method;
        it performs no workspace-policy check itself. The final column is all ones.
        """
        self.dense = kernels.dense(self.numbers, self.identities, self.centers, self.scales)

    def invert(self) -> None:
        """Cache stable category-to-row segments for implicit transpose products.

        Replace ``inverse_rows`` and ``inverse_offsets`` with CUDA int64 tensors.
        The stable sort preserves source-row order within each feature; unknown
        identity ``-1`` is excluded. This cache is fit-local, shares no sampler state,
        and must be built before implicit ``transpose_mv`` or ``diagonal`` calls.
        """
        self.inverse_rows, self.inverse_offsets = kernels.inverse(
            self.identities, len(self.centers)
        )

    def mv(self, weights: torch.Tensor) -> torch.Tensor:
        """Return ``X @ weights`` as selected-precision CUDA ``[N]``.

        Parameters
        ----------
        weights : Tensor
            ``[D]`` coefficients in the design dtype/device, intercept last.

        Returns
        -------
        Tensor
            Fresh logits ``[N]``. Design buffers and coefficients are read-only.
        """
        ops = library()
        if self.dense is not None:
            return ops.mv(self.dense, weights, False)
        out = ops.mv(self.numbers, weights[: self.numeric].contiguous(), False)
        cat = weights[self.numeric : -1].contiguous()
        if len(cat):
            out.add_(kernels.cat_mv(self.identities, cat, self.scales))
            out.sub_(torch.dot(self.centers / self.scales, cat))
        return out.add_(weights[-1])

    def transpose_mv(self, values: torch.Tensor) -> torch.Tensor:
        """Return the deterministic full-feature adjoint ``X.T @ values``.

        Parameters
        ----------
        values : Tensor
            ``[N]`` vector in the design dtype/device.

        Returns
        -------
        Tensor
            Fresh ``[D]`` adjoint, including intercept sum.

        Notes
        -----
        Implicit categories use precomputed stable row segments; no floating atomic
        accumulation or input mutation is introduced.
        """
        ops = library()
        if self.dense is not None:
            return ops.mv(self.dense, values.contiguous(), True)
        numeric = ops.mv(self.numbers, values.contiguous(), True)
        total = values.sum()
        cats = kernels.segment(self.inverse_rows, self.inverse_offsets, values.contiguous())
        cats = (cats - self.centers * total) / self.scales
        return torch.cat((numeric, cats, total[None]))

    def diagonal(self, curvature: torch.Tensor, ridge: float) -> torch.Tensor:
        """Return ``diag(X.T @ diag(curvature) @ X)`` plus feature ridge.

        ``curvature`` is a contiguous selected-precision CUDA ``[N]`` vector and
        ``ridge`` is a scalar penalty; the fresh ``[D]`` result excludes intercept
        ridge. Implicit category segments must already be cached. Numeric square/
        weight intermediates allocate ``[N, PN]`` storage, so this reference utility
        is not the memory-bounded native SAGA preparation path. Inputs are read-only.
        """
        numeric = (self.numbers.square() * curvature[:, None]).sum(dim=0)
        total = curvature.sum()
        counts = kernels.segment(self.inverse_rows, self.inverse_offsets, curvature.contiguous())
        categorical = (
            counts * (1 - 2 * self.centers) + self.centers.square() * total
        ) / self.scales.square()
        out = torch.cat((numeric, categorical, total[None]))
        out[:-1].add_(ridge)
        return out


@dataclass
class Fold:
    """Own one training/scoring split and its remaining model workspace.

    ``train``/``test`` share fitted categorical transforms but own their row
    buffers. ``labels`` and ``scoring_labels`` are FP32/FP64 binary CUDA vectors
    in the corresponding row order. ``solver`` selects direct Newton or native
    SketchySAGA; ``preconditioner`` records that choice. ``scratch_limit`` is a
    byte allowance after reserving draw/design storage, not total GPU memory.
    """

    train: Design
    test: Design
    labels: torch.Tensor
    scoring_labels: torch.Tensor
    solver: str
    preconditioner: str
    scratch_limit: int
    preprocessing_metadata: list[torch.Tensor] | None = None


def prepare(
    a: TensorTable,
    b: TensorTable,
    af: torch.Tensor,
    bf: torch.Tensor,
    fold: int,
    cardinalities: Sequence[int],
    options: LinearOptions,
    limit: int,
    reserved: int,
) -> Fold | None:
    """Fit one training-only design and select a memory-bounded solver.

    Parameters
    ----------
    a, b : TensorTable
        Draw-owned reference/candidate CUDA pools in selected FP32/FP64.
    af, bf : Tensor
        CUDA int64 source-stable fold labels matching each pool's rows.
    fold : int
        Held-out scoring partition; all fitted transforms exclude these rows.
    cardinalities : tuple of int
        Global identity-domain widths, including lookup space for missing codes.
    options : dict
        Solver selection and dense work/column cutoffs; scaling policy is fixed.
    limit, reserved : int
        Model/temporary byte limit and preprocessing reservation, respectively.

    Returns
    -------
    Fold or None
        Training/scoring designs and labels with remaining fit allowance. Return
        None for empty partitions, nonfinite transformed values, or a design
        that cannot satisfy allocation policy. A zero-feature fold retains drop
        metadata and is reported as unscorable before model fitting.

    Notes
    -----
    Numeric imputation prefers finite reference training means, falling back to
    candidate training means where necessary. Guarded standardization
    uses both training classes after imputation. Missingness indicators and union-
    training categorical indicators are unscaled. Held-out identities never expand
    this vocabulary. Full one-hot semantics are
    preserved without feature hashing, dropping a baseline, or fitting on scores.

    Source tables and fold arrays are read-only. Preprocessing and fitting peaks
    are combined by maximum, with retained storage counted explicitly. Auto routing
    uses encoded width, estimated dense work, and byte limits; it does not use a
    row-count cutoff. Forced direct fits return None if dense allocation fails the
    policy rather than silently switching algorithms.
    """
    masks = [af != fold, bf != fold, af == fold, bf == fold]
    rows = [torch.nonzero(mask).flatten() for mask in masks]
    if any(not len(row) for row in rows):
        return None
    nt, nv = len(rows[0]) + len(rows[1]), len(rows[2]) + len(rows[3])
    item = a.numeric.element_size()
    width = sum(cardinalities) + len(cardinalities)
    # Guard gathered sources, numeric outputs and FP64 block reductions before
    # allocating feature-sized preprocessing buffers. Missing flags can double PN.
    draw_bytes = (a.n_rows + b.n_rows) * (a.numeric.shape[1] * item + len(cardinalities) * 8 + 128)
    preprocessing_bytes = (
        draw_bytes
        + (nt + nv) * (a.numeric.shape[1] * item * 3 + len(cardinalities) * 16 + 32)
        + ((nt + 255) // 256 + 2) * a.numeric.shape[1] * 96
        + width * 64
    )
    if max(reserved, preprocessing_bytes) > limit:
        return None
    numbers = [table.numeric[index] for table, index in zip((a, b, a, b), rows, strict=True)]
    codes = [table.categorical[index] for table, index in zip((a, b, a, b), rows, strict=True)]
    numeric_state, numeric_train_missing = preprocessing.fit_numeric(
        numbers[0], numbers[1], linear=True, return_diagnostics=True
    )
    numeric_missing = torch.cat(
        (
            numeric_train_missing,
            torch.stack([preprocessing.count_missing(v) for v in numbers[2:]], dim=1),
        ),
        dim=1,
    )
    numbers = [preprocessing.apply_numeric(v, *numeric_state, linear=True) for v in numbers]
    if any(not bool(torch.isfinite(v).all()) for v in numbers):
        return None
    category_state, category_train_missing = preprocessing.fit_categories(
        codes[0], codes[1], cardinalities, return_diagnostics=True
    )
    mapping, domains, feature_offsets, active = category_state
    pn, pc = numbers[0].shape[1], len(active)
    d = pn + pc + 1
    centers, scales = numbers[0].new_zeros(pc), numbers[0].new_ones(pc)
    mapped = [
        preprocessing.map_categories(v, mapping, domains, feature_offsets, linear=True)
        for v in codes[:2]
    ]
    heldout = [
        preprocessing.map_categories(
            v, mapping, domains, feature_offsets, linear=True, return_diagnostics=True
        )
        for v in codes[2:]
    ]
    mapped.extend(v[0] for v in heldout)
    categorical_missing = torch.cat(
        (category_train_missing, torch.stack([v[1] for v in heldout], dim=1)), dim=1
    )
    categorical_unseen = torch.stack([v[2] for v in heldout], dim=1)
    source_columns = torch.bucketize(active, domains[1:], right=True)
    active_categories = torch.stack(
        (
            source_columns,
            active - domains[source_columns] - 1,
            torch.arange(pc, device=active.device, dtype=torch.int64),
        ),
        dim=1,
    )
    preprocessing_metadata = [
        numeric_state[0],
        numeric_state[1],
        active_categories,
        numeric_missing,
        categorical_missing,
        categorical_unseen,
    ]
    del codes
    train = Design(torch.cat(numbers[:2]), torch.cat(mapped[:2]), centers, scales)
    test = Design(torch.cat(numbers[2:]), torch.cat(mapped[2:]), centers, scales)
    labels = torch.cat((numbers[0].new_zeros(len(rows[0])), numbers[0].new_ones(len(rows[1]))))
    scoring = torch.cat((numbers[0].new_zeros(len(rows[2])), numbers[0].new_ones(len(rows[3]))))
    del numbers, mapped, heldout
    if d == 1:
        # Retain training-only drop explanations for an unscorable fold.
        return Fold(train, test, labels, scoring, "none", "none", 0, preprocessing_metadata)
    direct = options["solver"] == "direct" or (
        options["solver"] == "auto"
        and d - 1 <= options["direct_max_columns"]
        and nt * d * d <= options["direct_max_work"]
    )
    # Dense expansion, weighted designs, factors and sort/inversion
    # scratch are guarded before allocation. Wide designs retain all features
    # through implicit one-hot operators and a bounded Nyström preconditioner.
    # Preprocessing and fitting peaks occur at different times. Their bounds
    # combine by maximum, while storage that survives both phases is included
    # explicitly. Summing their entire peaks rejects otherwise bounded fits.
    dense_bytes = max(
        reserved,
        preprocessing_bytes,
        draw_bytes
        + item * ((nt + nv) * d + max(nt * d, 2 * d * d) + 7 * d * d)
        + (nt + nv) * (pn * item + len(cardinalities) * 8 + 32),
    )
    compact_bytes = max(
        reserved,
        preprocessing_bytes,
        draw_bytes
        + nt * (pn * item * 4 + len(cardinalities) * 64 + 32)
        + nv * (pn * item * 2 + len(cardinalities) * 16 + 16)
        + d * (40 * item + 64)
        + width * 64,
    )
    dense = dense_bytes <= limit and direct
    if dense:
        train.expand()
        test.expand()
        preconditioner = "none"
    elif options["solver"] != "direct" and compact_bytes <= limit:
        direct = False
        train.invert()
        preconditioner = "nystrom_residual_diagonal"
    else:
        return None
    fit_reserved = draw_bytes + (nt + nv) * (pn * item + len(cardinalities) * 8 + 32)
    if dense:
        fit_reserved += item * ((nt + nv) * d + nt * d + 2 * d * d)
    else:
        fit_reserved += train.inverse_rows.numel() * train.inverse_rows.element_size()
        fit_reserved += train.inverse_offsets.numel() * train.inverse_offsets.element_size()
    return Fold(
        train,
        test,
        labels,
        scoring,
        "direct" if direct else "sketchysaga",
        preconditioner,
        limit - fit_reserved,
        preprocessing_metadata,
    )
