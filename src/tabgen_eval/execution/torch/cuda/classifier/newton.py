"""Adapt dense CUDA Newton packets and terminal diagnostics.

The compiled optimizer owns all gradient, factorization, damping, iteration and
Armijo work. Designs are read-only column-major FP32/FP64 tensors, with an
unpenalized final intercept. The only convergence certificate is the full
average-loss gradient L2 norm; the Newton decrement remains a diagnostic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tabgen_eval.execution.torch.cuda.classifier.device import operators

if TYPE_CHECKING:
    from torch import Tensor

    from tabgen_eval.execution.torch.cuda.classifier.data import Fold
    from tabgen_eval.metrics.c2st import LinearOptions

_REASONS = (
    "max_iter",
    "gradient_squared",
    "nonfinite_gradient",
    "factorization_failed",
    "invalid_direction",
    "line_search_failed",
)


def fit(fold: Fold, options: LinearOptions) -> tuple[Tensor, dict[str, object]]:
    """Return selected-precision coefficients and native fit diagnostics.

    ``fold.train.dense`` includes the intercept and shares the labels' CUDA
    device/dtype. ``options`` supplies resolved ridge, gradient L2-norm tolerance
    and iteration cap. The fold budget excludes a reserved weighted design and
    two square buffers, which are credited once when passing the total native
    allocation allowance. Querying cuSOLVER scratch precedes large allocations;
    budget exhaustion raises ``torch.OutOfMemoryError``. Input storage, current
    stream, library math modes and global RNG state remain unchanged.
    """
    design = fold.train
    ops, ptx = operators(design.dense, algorithm="newton")
    reserved = design.numbers.element_size() * (design.n * design.d + 2 * design.d**2)
    weights, diagnostics, status, iterations = ops.newton_fit(
        design.dense,
        fold.labels,
        ptx,
        options["ridge"],
        options["tolerance"],
        options["max_iter"],
        fold.scratch_limit + reserved,
    )
    squared, decrement, damping = diagnostics.tolist()
    return weights, dict(
        solver="direct",
        preconditioner=fold.preconditioner,
        converged=status == 1,
        reason=_REASONS[status],
        iterations=iterations,
        gradient_squared=squared,
        decrement_squared=decrement,
        factorization_damping=damping,
        effective_ridge=options["ridge"],
        tolerance=options["tolerance"],
        stopping_rule="gradient_squared",
    )
