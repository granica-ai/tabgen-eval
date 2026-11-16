"""Orchestrate native CUDA SketchySAGA chunks and bounded rollback.

C++/CUDA owns minibatch gathering, scalar-cache updates, inverse application,
and epoch reductions. Python generates at most sixteen epochs of independent
without-replacement minibatches per chunk, owns fit state/diagnostics, and uses
a fixed training probe for backoff. The scalar loss-gradient table is O(N),
not an N-by-D gradient table; compact categories are never densely expanded.

Sketch, schedule, and probe randomness use private domain-separated generators.
Native calls return fresh state, allowing reference checkpoints and exact
scheduled-prefix replay without modifying input designs. Full exact gradient
passes and held-out early stopping are absent from this production route.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch

from tabgen_eval.execution.torch.cuda.classifier.device import operators
from tabgen_eval.execution.torch.cuda.classifier.packets import EpochHistory, FitInputs
from tabgen_eval.execution.torch.cuda.classifier.preconditioner import build
from tabgen_eval.execution.torch.cuda.classifier.saga_guard import (
    MAX_BACKOFFS,
    ObjectiveGuard,
    reserve_bytes,
)

if TYPE_CHECKING:
    from tabgen_eval.execution.torch.cuda.classifier.data import Design, Fold
    from tabgen_eval.metrics.c2st import LinearOptions


def _failure(
    design: Design, ridge: float, reason: str, tolerance: float
) -> tuple[torch.Tensor, dict[str, object]]:
    """Return unscored zero coefficients and a recognized SAGA failure record.

    ``design`` supplies output dtype/device/width and ``ridge`` the effective
    feature penalty. Return fresh ``[D]`` placeholder weights with zero attempted
    work and no online-gradient estimate. The executor suppresses AUC scoring for
    these memory/numerical failure reasons; the zeros are not a valid chance fit.
    """
    return design.numbers.new_zeros(design.d), dict(
        solver="sketchysaga",
        converged=False,
        reason=reason,
        iterations=0,
        epochs_completed=0,
        gradient_squared=None,
        online_gradient_squared=None,
        online_gradient_infinity=None,
        initial_online_gradient_infinity=None,
        gradient_stopping_threshold=tolerance,
        decrement_squared=float("nan"),
        effective_ridge=ridge,
        tolerance=tolerance,
        full_gradient_evaluations=0,
    )


def fit(fold: Fold, options: LinearOptions, seed: int) -> tuple[torch.Tensor, dict[str, object]]:
    """Fit full-feature logistic regression with native online-gradient stopping.

    Parameters
    ----------
    fold : Fold
        Compact training design, binary labels, and remaining model scratch bytes.
    options : dict
        Resolved average-loss ridge and SAGA rank, batch, epoch, tolerance,
        sampled-preparation, and backoff settings. Batch is capped at training N.
    seed : int
        Private fit seed for independent sketch/power, minibatch, and probe streams.

    Returns
    -------
    tuple of Tensor and dict
        Last accepted CUDA ``[D]`` coefficients and serializable fit diagnostics.
        Recognized allocation/curvature failures instead return placeholder weights
        marked as failed, which callers must not score. Programming errors propagate.

    Notes
    -----
    Native preparation builds one fixed curvature factor. FP64 uses cached-basis
    inverse application; FP32 selects Householder coordinates only when inverse
    ratios are small. Factor, sort/cache, rollback/probe, and int32 schedule bytes
    bound a chunk of at most sixteen epochs before any native fit call.

    Each epoch contains ceil(N/batch) independent uniformly sampled minibatches;
    rows within a minibatch are distinct, but the epoch is not a shuffled full-data
    partition. Rejected chunks replay the identical scheduled prefix from accepted
    state after halving the rate, up to four backoffs. Attempted work, including
    retries, consumes the global epoch budget; exhaustion retains an accepted fit
    if available and reports nonconvergence honestly.

    The stopping test is ``||g_t||_inf <= tolerance``. The estimate is the existing
    unpreconditioned online direction, including feature ridge, measured before
    the corresponding update; no full-gradient pass occurs.
    Coefficient changes are diagnostics only. The fixed training-only objective
    probe controls backoff, and held-out AUC never influences fitting or stopping.
    """
    design = fold.train
    ridge = options["ridge"]
    epochs = options.get("saga_max_epochs", 10000)
    batch = min(options.get("saga_batch_size", 4096), design.n)
    rank = min(options.get("saga_rank", 64), design.d)
    tolerance = options["tolerance"]
    steps = math.ceil(design.n / batch)
    item = design.numbers.element_size()
    guarded = options.get("saga_backoff", True)
    guard_bytes = reserve_bytes(design) if guarded else 0
    try:
        generator = torch.Generator(device=design.numbers.device).manual_seed(int(seed) ^ 0x5A6A)
        factor, rate, metadata = build(
            design,
            ridge,
            rank,
            generator,
            fold.scratch_limit,
            sampled=options.get("saga_sampled_preparation", True),
            preparation_rows=options.get("saga_preparation_rows", 4096),
            batch=batch,
        )
        inverse_mode = 1 if item == 4 and metadata["minimum_inverse_ratio"] < 1e-3 else 2
        basis = factor.basis() if inverse_mode == 2 else design.numbers.new_empty((0, 0))
        qr, rotation = factor.reflectors.T.contiguous().T, factor.rotations.T.contiguous().T
        ratios = factor.ratios if inverse_mode == 1 else factor.ratios - 1
        factor_bytes = sum(
            t.numel() * t.element_size()
            for t in [factor.inverse_root, qr, factor.tau, rotation, ratios, basis]
        )
        # One cached ORMQR workspace plus deterministic sort/fitting buffers.
        sort_n = 1 << (batch - 1).bit_length()
        fixed = item * (
            10 * design.d + 2 * design.n + batch * design.numeric + 2 * batch + len(design.centers)
        )
        fixed += 12 * design.identities.shape[1] * sort_n + 8 * len(design.centers) + 65536
        if inverse_mode == 1:
            fixed += 128 * 1024**2
        allowance = fold.scratch_limit - factor_bytes - fixed - guard_bytes
        chunk = min(16, epochs, allowance // max(steps * batch * 4, 1))
        if chunk < 1:
            raise MemoryError("SketchySAGA buffers exceed the resolved model workspace allowance")
        ops, ptx = operators(design.numbers)
        weights = design.numbers.new_zeros(design.d)
        table = weights.new_zeros(design.n)
        average = weights.new_zeros(design.d)
        rng = np.random.default_rng(int(seed) ^ 0xA617)
        completed = updates = accepted_epochs = accepted_updates = 0
        converged = False
        last = None
        online = weights
        initial_norm = -1.0
        initial_rate = rate
        accepted_rate = None
        guard = ObjectiveGuard(fold, seed, ridge) if guarded else None
        terminal_numerical = False
        gradient_infinity = float("nan")
        while completed < epochs:
            count = min(chunk, epochs - completed)
            schedule = np.empty((count, steps, batch), dtype=np.int32)
            for e in range(count):
                for j in range(steps):
                    schedule[e, j] = rng.choice(design.n, batch, replace=False)
            schedule_gpu = torch.as_tensor(schedule, device=weights.device)
            # Native saga_fit reads inputs and produces fresh output buffers.
            # Keep only state references; the packet/schedule must be released.
            checkpoint = weights, table, average
            accepted = False
            while completed < epochs:
                allowed = min(count, epochs - completed)
                packet = list(
                    FitInputs(
                        numbers=design.numbers,
                        categories=design.identities,
                        category_centers=design.centers,
                        category_scales=design.scales,
                        labels=fold.labels,
                        schedule=schedule_gpu[:allowed],
                        inverse_root=factor.inverse_root,
                        reflectors=qr,
                        tau=factor.tau,
                        rotations=rotation,
                        ratios=ratios,
                        basis=basis,
                        initial_weights=checkpoint[0],
                        initial_table=checkpoint[1],
                        initial_average=checkpoint[2],
                    )
                )
                (
                    proposed_weights,
                    proposed_table,
                    proposed_average,
                    _,
                    history,
                    done,
                    count_updates,
                    proposed_online,
                ) = ops.saga_fit(
                    packet,
                    ptx,
                    rate,
                    ridge,
                    tolerance,
                    inverse_mode,
                    True,
                    False,
                    fold.scratch_limit - factor_bytes - guard_bytes,
                    initial_norm,
                )
                completed += done
                updates += count_updates
                epoch = EpochHistory._make(history[done - 1].tolist())
                maximum = epoch.maximum_abs_coefficient
                change = epoch.maximum_abs_change
                finite = epoch.finite
                intercept_change = epoch.intercept_abs_change
                proposed_infinity = epoch.online_gradient_infinity
                reported_initial = epoch.initial_gradient_infinity
                if initial_norm < 0:
                    initial_norm = reported_initial
                objective = guard.objective(proposed_weights) if guard else None
                rejected = guard and guard.rejects(objective, bool(finite))
                del packet, history
                if rejected:
                    guard.rejected_epochs += done
                    del proposed_weights, proposed_table, proposed_average, proposed_online
                    if guard.backoffs >= MAX_BACKOFFS:
                        terminal_numerical = True
                        break
                    if completed >= epochs:
                        break
                    guard.backoffs += 1
                    rate *= 0.5
                    # Replay this exact scheduled prefix from the accepted state.
                    continue
                weights, table, average = proposed_weights, proposed_table, proposed_average
                online = proposed_online
                accepted_epochs += done
                accepted_updates += count_updates
                accepted_rate = rate
                gradient_infinity = proposed_infinity
                relative = change / maximum if maximum else (0.0 if change == 0 else float("inf"))
                last = dict(
                    maximum_feature_coefficient=maximum,
                    maximum_feature_change=change,
                    relative_coefficient_change=relative,
                    intercept_change=intercept_change,
                )
                if guard:
                    guard.previous = objective
                converged = bool(finite) and gradient_infinity <= tolerance
                terminal_numerical = not finite
                accepted = True
                # Avoid retaining old state through proposed_* aliases.
                del proposed_weights, proposed_table, proposed_average, proposed_online
                break
            del checkpoint, schedule_gpu, schedule
            if converged or terminal_numerical or not accepted:
                break
        if not accepted_updates:
            terminal_numerical = True
        squared = float(torch.dot(online, online).item()) if accepted_updates else float("nan")
        reason = (
            "online_gradient"
            if converged
            else ("numerical_failure" if terminal_numerical else "max_epochs")
        )
        state = dict(
            solver="sketchysaga",
            stopping_rule="online_gradient_infinity",
            converged=converged,
            reason=reason,
            iterations=updates,
            epochs_completed=completed,
            epoch_budget=epochs,
            accepted_epochs=accepted_epochs,
            accepted_updates=accepted_updates,
            initial_learning_rate=initial_rate,
            final_learning_rate=accepted_rate,
            current_attempt_learning_rate=rate,
            saga_backoff_enabled=guarded,
            gradient_squared=None,
            decrement_squared=float("nan"),
            online_gradient_squared=squared if math.isfinite(squared) else None,
            online_gradient_norm=math.sqrt(squared) if math.isfinite(squared) else None,
            gradient_estimate_kind="saga_variance_reduced_minibatch",
            gradient_estimate_iterate="before_final_update",
            gradient_estimate_includes_ridge=True,
            gradient_estimate_preconditioned=False,
            full_gradient_evaluations=0,
            effective_ridge=ridge,
            tolerance=tolerance,
            online_gradient_infinity=gradient_infinity
            if math.isfinite(gradient_infinity)
            else None,
            initial_online_gradient_infinity=initial_norm if math.isfinite(initial_norm) else None,
            gradient_stopping_threshold=tolerance,
            inverse_application="householder" if inverse_mode == 1 else "basis",
            schedule_chunk_epochs=chunk,
            scalar_gradient_table_bytes=design.n * item,
            **(last or {}),
            **(guard.diagnostics() if guard else {}),
            **metadata,
        )
        return weights, state
    except (MemoryError, torch.OutOfMemoryError):
        return _failure(design, ridge, "memory_limit", tolerance)
    except (FloatingPointError, torch.linalg.LinAlgError) as error:
        weights, state = _failure(design, ridge, "numerical_failure", tolerance)
        state["detail"] = str(error)
        return weights, state
