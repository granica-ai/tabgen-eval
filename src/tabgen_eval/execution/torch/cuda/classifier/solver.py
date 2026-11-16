"""Dispatch compiled CUDA logistic fits and compute held-out ROC AUC.

Dense designs use native Newton with direct cuBLAS/cuSOLVER operations;
compact designs use native SketchySAGA. Each route returns coefficients and
honest convergence diagnostics. ``derivatives`` and ``hessian_vector`` are
full-data numerical references for independent gradient and Hessian tests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from tabgen_eval.execution.torch.cuda.triton.classifier import logistic

if TYPE_CHECKING:
    from tabgen_eval.execution.torch.cuda.classifier.data import Design, Fold
    from tabgen_eval.metrics.c2st import LinearOptions


def derivatives(
    design: Design, labels: torch.Tensor, weights: torch.Tensor, ridge: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return full average-loss gradient and per-row logistic curvature.

    Matching selected-precision CUDA weights/labels are read-only. Return fresh
    ``[D]`` gradient and ``[N]`` curvature; the Triton residual and curvature
    already include division by N. Add feature ridge to the gradient without
    penalizing the final intercept. Compact designs require cached inverse rows.
    This full-data routine supports numerical oracles.
    """
    residual, curvature, _ = logistic(design.mv(weights), labels)
    gradient = design.transpose_mv(residual)
    gradient[:-1].add_(weights[:-1], alpha=ridge)
    return gradient, curvature


def hessian_vector(
    design: Design, curvature: torch.Tensor, vector: torch.Tensor, ridge: float
) -> torch.Tensor:
    """Apply the exact penalized Hessian to a CUDA vector for numerical oracles.

    ``curvature[N]`` includes average-loss normalization from ``derivatives``;
    ``vector[D]`` matches the design dtype/device. Return a fresh ``[D]`` product,
    adding ridge only to features. Implicit designs need cached inverse rows.
    This reference helper is not used by production optimizer iterations.
    """
    out = design.transpose_mv(curvature * design.mv(vector))
    out[:-1].add_(vector[:-1], alpha=ridge)
    return out


def fit(fold: Fold, options: LinearOptions, seed: int) -> tuple[torch.Tensor, dict[str, object]]:
    """Fit average-loss ridge logistic regression by direct Newton or native SAGA.

    Parameters
    ----------
    fold : Fold
        Training/scoring designs, labels, selected solver, and remaining fit bytes.
    options : dict
        Solver budgets and tolerance. Zero ridge is resolved to 1/training_rows.
        Direct Newton bounds the full-gradient L2 norm; forwarded SAGA fits
        bound the online-gradient infinity norm by the same tolerance.
    seed : int
        Private uint64 SAGA seed; direct Newton is deterministic and does not use it.

    Returns
    -------
    tuple of Tensor and dict
        Selected-precision CUDA ``[D]`` coefficients and optimization diagnostics,
        retaining iteration caps and numerical failures rather than hiding them.

    Raises
    ------
    ValueError
        A direct fit does not have an expanded dense design or known solver.
    torch.OutOfMemoryError
        Native Newton buffers or cuSOLVER scratch exceed the fit allowance.

    Notes
    -----
    Design inputs are read-only. Direct Newton takes damped Cholesky directions
    and at most forty Armijo backtracks per accepted step. Stable per-row loss
    differences reduce cancellation near an optimum. The decrement is diagnostic;
    only the full-gradient L2-norm threshold certifies direct convergence. The
    last coefficient is never ridge-penalized. Native SAGA owns its update loop
    and handles its recognized memory/numerical failures in the delegated adapter.
    """
    design = fold.train
    options = options.copy()
    if options["ridge"] == 0:
        options["ridge"] = 1 / design.n
    if fold.solver == "sketchysaga":
        from tabgen_eval.execution.torch.cuda.classifier.saga import fit as saga_fit

        return saga_fit(fold, options, seed)
    if fold.solver != "direct" or design.dense is None:
        raise ValueError("Direct Newton requires an expanded dense design")
    from tabgen_eval.execution.torch.cuda.classifier.newton import fit as newton_fit

    return newton_fit(fold, options)


def auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute held-out ROC AUC with exact tie groups.

    Parameters
    ----------
    scores, labels : Tensor
        CUDA ``[N]`` finite margins and binary labels on one device. Scores retain
        selected FP32/FP64; source tensors are read-only.

    Returns
    -------
    float
        Rank AUC, NaN for nonfinite scores or an absent class. Chance clipping is
        applied by the executor, not here.
    """
    if not bool(torch.isfinite(scores).all()):
        return float("nan")
    order = scores.argsort(stable=True)
    values, y = scores[order], labels[order].to(torch.int64)
    _, inverse, counts = values.unique_consecutive(return_inverse=True, return_counts=True)
    positives = torch.bincount(inverse, weights=y.to(scores.dtype), minlength=len(counts))
    negatives = counts.to(scores.dtype) - positives
    total = int(y.sum().item())
    if not total or total == len(y):
        return float("nan")
    below = negatives.cumsum(0) - negatives
    wins = (positives * (below + 0.5 * negatives)).sum()
    return float((wins / (total * (len(y) - total))).item())
