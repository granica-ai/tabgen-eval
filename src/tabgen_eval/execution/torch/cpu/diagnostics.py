"""Decode CPU classifier diagnostics separately from numerical execution.

Indices mirror csrc/cpu/classifier/diagnostics.h; the wire ABI is unchanged.
"""

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np

from tabgen_eval._typing import FloatArray

if TYPE_CHECKING:
    from tabgen_eval.metrics.c2st import LinearOptions


class FoldField(IntEnum):
    """Consumed offsets in the 46-value native per-fold diagnostic record.

    The matching C++ enum lives in ``csrc/cpu/classifier/diagnostics.h``. Slots are
    shared across direct Newton and SketchySAGA; fields not populated by a solver
    may contain NaN. Explicit values preserve the native wire contract, including
    slots that this decoder does not consume.
    """

    TRAINING_ROWS = 0
    VALIDATION_ROWS = 1
    ENCODED_COLUMNS = 2
    CONVERGED = 3
    ITERATIONS = 4
    EPOCHS_COMPLETED = 5
    GRADIENT_SQUARED = 6
    RELATIVE_COEFFICIENT_CHANGE = 7
    EFFECTIVE_RIDGE = 8
    PRECONDITIONER_RANK = 9
    LEARNING_RATE = 10
    MAXIMUM_FEATURE_COEFFICIENT = 11
    MAXIMUM_FEATURE_CHANGE = 12
    INTERCEPT_CHANGE = 13
    STABLE_INVERSE = 14
    ESTIMATED_EIGENVALUE = 15
    DECREMENT_SQUARED = 16
    ONLINE_GRADIENT_INFINITY = 17
    INITIAL_ONLINE_GRADIENT_INFINITY = 18
    GRADIENT_STOPPING_THRESHOLD = 19
    ACCEPTED_EPOCHS = 20
    ACCEPTED_UPDATES = 21
    INITIAL_LEARNING_RATE = 22
    FINAL_LEARNING_RATE = 23
    CURRENT_ATTEMPT_LEARNING_RATE = 24
    BACKOFF_PROBE_ROWS = 26
    BACKOFF_PROBE_CHECKS = 27
    BACKOFF_RETRIES = 28
    REJECTED_EPOCHS = 29
    FINAL_PROBE_OBJECTIVE = 31
    SCHEDULE_CHUNK_EPOCHS = 32
    PREPARATION_TRAINING_ROWS = 33
    PREPARATION_ROWS = 34
    SAMPLED_PREPARATION = 35
    PREPARATION_SAMPLE_HASH = 36
    CURVATURE_DATA_RAW = 37
    CURVATURE_DATA_USED = 38
    CURVATURE_ROW_MAXIMUM_RAW = 39
    CURVATURE_ROW_MAXIMUM_USED = 40
    CURVATURE_RIDGE_BOUND = 41
    CURVATURE_PRACTICAL = 42
    CURVATURE_CALIBRATION_ROWS = 43
    CURVATURE_CALIBRATION_POPULATION = 44
    CURVATURE_CALIBRATION_BATCH = 45


def classifier_fold(
    row: FloatArray, *, fold: int, solver: str, reason: str, options: LinearOptions
) -> dict[str, object]:
    """Decode a native diagnostic row into JSON-compatible report metadata.

    ``row`` follows ``FoldField``; ``solver``/``reason`` come from the parallel native
    string arrays and ``options`` is the metric's native option mapping. Unknown
    numeric diagnostics become ``None`` or the established count/flag defaults.
    SketchySAGA's online gradient is labeled separately from an exact full gradient;
    this function never performs a gradient evaluation or changes native outcomes.
    The input row is read-only and the returned dictionary owns its scalar values.
    """
    data = dict(
        fold=fold,
        training_rows=int(row[FoldField.TRAINING_ROWS])
        if np.isfinite(row[FoldField.TRAINING_ROWS])
        else 0,
        scoring_rows=int(row[FoldField.VALIDATION_ROWS])
        if np.isfinite(row[FoldField.VALIDATION_ROWS])
        else 0,
        features=int(row[FoldField.ENCODED_COLUMNS])
        if np.isfinite(row[FoldField.ENCODED_COLUMNS])
        else 0,
        converged=bool(row[FoldField.CONVERGED])
        if np.isfinite(row[FoldField.CONVERGED])
        else False,
        solver=solver,
        reason=reason,
        tolerance=options["tolerance"],
    )
    if solver != "none":
        data.update(
            iterations=int(row[FoldField.ITERATIONS])
            if np.isfinite(row[FoldField.ITERATIONS])
            else 0,
            effective_ridge=float(row[FoldField.EFFECTIVE_RIDGE])
            if np.isfinite(row[FoldField.EFFECTIVE_RIDGE])
            else None,
        )
        if solver == "sketchysaga":
            # The legacy slot is shared with Newton's exact gradient diagnostic.
            # Relabel it here so reports never imply a full SAGA gradient scan.
            online = (
                float(row[FoldField.GRADIENT_SQUARED])
                if np.isfinite(row[FoldField.GRADIENT_SQUARED])
                else None
            )
            fallback = (
                np.isfinite(row[FoldField.PRECONDITIONER_RANK])
                and row[FoldField.PRECONDITIONER_RANK] == 0
                and np.isfinite(row[FoldField.LEARNING_RATE])
            )
            # A finite rate with rank zero identifies successful diagonal
            # fallback; a missing rate can instead indicate preparation failure.
            data.update(
                epochs_completed=int(row[FoldField.EPOCHS_COMPLETED])
                if np.isfinite(row[FoldField.EPOCHS_COMPLETED])
                else 0,
                stopping_rule="online_gradient_infinity",
                gradient_squared=None,
                decrement_squared=None,
                online_gradient_squared=online,
                online_gradient_norm=float(np.sqrt(online)) if online is not None else None,
                relative_coefficient_change=float(row[FoldField.RELATIVE_COEFFICIENT_CHANGE])
                if np.isfinite(row[FoldField.RELATIVE_COEFFICIENT_CHANGE])
                else None,
                preconditioner_rank=int(row[FoldField.PRECONDITIONER_RANK])
                if np.isfinite(row[FoldField.PRECONDITIONER_RANK])
                else 0,
                preconditioner=("diagonal_fallback" if fallback else "nystrom_residual_diagonal")
                if np.isfinite(row[FoldField.LEARNING_RATE])
                else None,
                factorization_fallbacks=int(fallback),
                epoch_budget=options["saga_max_epochs"],
                online_gradient_infinity=float(row[FoldField.ONLINE_GRADIENT_INFINITY])
                if np.isfinite(row[FoldField.ONLINE_GRADIENT_INFINITY])
                else None,
                initial_online_gradient_infinity=float(
                    row[FoldField.INITIAL_ONLINE_GRADIENT_INFINITY]
                )
                if np.isfinite(row[FoldField.INITIAL_ONLINE_GRADIENT_INFINITY])
                else None,
                gradient_stopping_threshold=float(row[FoldField.GRADIENT_STOPPING_THRESHOLD])
                if np.isfinite(row[FoldField.GRADIENT_STOPPING_THRESHOLD])
                else None,
                learning_rate=float(row[FoldField.LEARNING_RATE])
                if np.isfinite(row[FoldField.LEARNING_RATE])
                else None,
                accepted_epochs=int(row[FoldField.ACCEPTED_EPOCHS])
                if np.isfinite(row[FoldField.ACCEPTED_EPOCHS])
                else 0,
                accepted_updates=int(row[FoldField.ACCEPTED_UPDATES])
                if np.isfinite(row[FoldField.ACCEPTED_UPDATES])
                else 0,
                initial_learning_rate=float(row[FoldField.INITIAL_LEARNING_RATE])
                if np.isfinite(row[FoldField.INITIAL_LEARNING_RATE])
                else None,
                final_learning_rate=float(row[FoldField.FINAL_LEARNING_RATE])
                if np.isfinite(row[FoldField.FINAL_LEARNING_RATE])
                else None,
                current_attempt_learning_rate=float(row[FoldField.CURRENT_ATTEMPT_LEARNING_RATE])
                if np.isfinite(row[FoldField.CURRENT_ATTEMPT_LEARNING_RATE])
                else None,
                saga_backoff_enabled=options["saga_backoff"],
                schedule_chunk_epochs=int(row[FoldField.SCHEDULE_CHUNK_EPOCHS])
                if np.isfinite(row[FoldField.SCHEDULE_CHUNK_EPOCHS])
                else 16,
                preparation_training_rows=int(row[FoldField.PREPARATION_TRAINING_ROWS])
                if np.isfinite(row[FoldField.PREPARATION_TRAINING_ROWS])
                else 0,
                preparation_rows=int(row[FoldField.PREPARATION_ROWS])
                if np.isfinite(row[FoldField.PREPARATION_ROWS])
                else 0,
                sampled_preparation=bool(row[FoldField.SAMPLED_PREPARATION])
                if np.isfinite(row[FoldField.SAMPLED_PREPARATION])
                else False,
                preparation_sample_hash=int(row[FoldField.PREPARATION_SAMPLE_HASH])
                if np.isfinite(row[FoldField.PREPARATION_SAMPLE_HASH])
                else 0,
                preparation_sampling_threshold=500000,
                learning_rate_calibration="Gazagnadou practical expected smoothness; empirical power/sample margins",
                curvature_data_raw=float(row[FoldField.CURVATURE_DATA_RAW])
                if np.isfinite(row[FoldField.CURVATURE_DATA_RAW])
                else None,
                curvature_data_used=float(row[FoldField.CURVATURE_DATA_USED])
                if np.isfinite(row[FoldField.CURVATURE_DATA_USED])
                else None,
                curvature_row_maximum_raw=float(row[FoldField.CURVATURE_ROW_MAXIMUM_RAW])
                if np.isfinite(row[FoldField.CURVATURE_ROW_MAXIMUM_RAW])
                else None,
                curvature_row_maximum_used=float(row[FoldField.CURVATURE_ROW_MAXIMUM_USED])
                if np.isfinite(row[FoldField.CURVATURE_ROW_MAXIMUM_USED])
                else None,
                curvature_ridge_bound=float(row[FoldField.CURVATURE_RIDGE_BOUND])
                if np.isfinite(row[FoldField.CURVATURE_RIDGE_BOUND])
                else None,
                curvature_practical=float(row[FoldField.CURVATURE_PRACTICAL])
                if np.isfinite(row[FoldField.CURVATURE_PRACTICAL])
                else None,
                curvature_calibration_rows=int(row[FoldField.CURVATURE_CALIBRATION_ROWS])
                if np.isfinite(row[FoldField.CURVATURE_CALIBRATION_ROWS])
                else 0,
                curvature_calibration_population=int(
                    row[FoldField.CURVATURE_CALIBRATION_POPULATION]
                )
                if np.isfinite(row[FoldField.CURVATURE_CALIBRATION_POPULATION])
                else 0,
                curvature_calibration_batch=int(row[FoldField.CURVATURE_CALIBRATION_BATCH])
                if np.isfinite(row[FoldField.CURVATURE_CALIBRATION_BATCH])
                else 0,
                maximum_feature_coefficient=float(row[FoldField.MAXIMUM_FEATURE_COEFFICIENT])
                if np.isfinite(row[FoldField.MAXIMUM_FEATURE_COEFFICIENT])
                else None,
                maximum_feature_change=float(row[FoldField.MAXIMUM_FEATURE_CHANGE])
                if np.isfinite(row[FoldField.MAXIMUM_FEATURE_CHANGE])
                else None,
                intercept_change=float(row[FoldField.INTERCEPT_CHANGE])
                if np.isfinite(row[FoldField.INTERCEPT_CHANGE])
                else None,
                stable_inverse=bool(row[FoldField.STABLE_INVERSE])
                if np.isfinite(row[FoldField.STABLE_INVERSE])
                else False,
                estimated_eigenvalue=float(row[FoldField.ESTIMATED_EIGENVALUE])
                if np.isfinite(row[FoldField.ESTIMATED_EIGENVALUE])
                else None,
                gradient_estimate_kind="saga_variance_reduced_minibatch",
                gradient_estimate_iterate="before_final_update",
                gradient_estimate_includes_ridge=True,
                gradient_estimate_preconditioned=False,
                full_gradient_evaluations=0,
            )
            if options["saga_backoff"]:
                data.update(
                    backoff_probe_rows=int(row[FoldField.BACKOFF_PROBE_ROWS])
                    if np.isfinite(row[FoldField.BACKOFF_PROBE_ROWS])
                    else 0,
                    backoff_probe_checks=int(row[FoldField.BACKOFF_PROBE_CHECKS])
                    if np.isfinite(row[FoldField.BACKOFF_PROBE_CHECKS])
                    else 0,
                    backoff_retries=int(row[FoldField.BACKOFF_RETRIES])
                    if np.isfinite(row[FoldField.BACKOFF_RETRIES])
                    else 0,
                    rejected_epochs=int(row[FoldField.REJECTED_EPOCHS])
                    if np.isfinite(row[FoldField.REJECTED_EPOCHS])
                    else 0,
                    backoff_retry_limit=4,
                    final_probe_objective=float(row[FoldField.FINAL_PROBE_OBJECTIVE])
                    if np.isfinite(row[FoldField.FINAL_PROBE_OBJECTIVE])
                    else None,
                )
        else:
            data.update(
                stopping_rule="gradient_squared",
                gradient_squared=float(row[FoldField.GRADIENT_SQUARED])
                if np.isfinite(row[FoldField.GRADIENT_SQUARED])
                else None,
                decrement_squared=float(row[FoldField.DECREMENT_SQUARED])
                if np.isfinite(row[FoldField.DECREMENT_SQUARED])
                else None,
            )
    return data
