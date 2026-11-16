"""Independent typed one-hot geometry and current two-pool numerical adapters."""

import numpy as np


def typed_key(value):
    """Independent logical families for ordinary Python/NumPy scalar test labels."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return "bool", value
    if isinstance(value, int):
        return "int", value
    if isinstance(value, float):
        return "float", value
    if isinstance(value, str):
        return "str", value
    return type(value), value


def geometry(reference, queries, schema, dtype="float64", *, scaling_dtype=None):
    """Build tiny explicit one-hot tables without production encoders or category keys.

    ``scaling_dtype`` distinguishes Python coordinate rounding after FP64 scaling
    from tensor geometry that scales directly in the selected arithmetic type.
    Categorical unknowns have an all-zero reference-vocabulary block.
    """
    features = list(schema.numeric) + list(schema.categorical)
    frames = [frame.dropna(subset=features) for frame in (reference, *queries)]
    numbers = [
        frame[list(schema.numeric)].to_numpy(dtype=scaling_dtype or dtype) for frame in frames
    ]
    if schema.numeric:
        low = numbers[0].min(0)
        ranges = numbers[0].max(0) - low
        ranges[ranges == 0] = 1
        numbers = [(array - low) / ranges for array in numbers]
    numbers = [array.astype(dtype) for array in numbers]
    categories = [[] for _ in frames]
    for column in schema.categorical:
        keys = [[typed_key(value) for value in frame[column]] for frame in frames]
        levels = tuple(dict.fromkeys(keys[0]))
        for blocks, values in zip(categories, keys, strict=True):
            blocks.append(
                np.asarray(
                    [[key == level for level in levels] for key in values], dtype=np.int64
                ).reshape(len(values), len(levels))
            )
    onehot = [
        np.column_stack(blocks) if blocks else np.empty((len(frame), 0), dtype=np.int64)
        for blocks, frame in zip(categories, frames, strict=True)
    ]
    return numbers, onehot, [len(frame) for frame in frames]


def squared_distances(a, b, ca, cb, *, dtype=None):
    """Feature-order arithmetic with one exact categorical contribution addition."""
    kind = np.dtype(dtype or a.dtype)
    result = np.zeros((len(a), len(b)), dtype=kind)
    for c in range(a.shape[1]):
        delta = a[:, c, None].astype(kind) - b[None, :, c].astype(kind)
        result += delta * delta
    categories = np.zeros(result.shape, dtype=np.int64)
    for c in range(ca.shape[1]):
        delta = ca[:, c, None] - cb[None, :, c]
        categories += delta * delta
    return result + categories.astype(kind)


def distance_matrices(
    reference, queries, schema, dtype="float64", *, scaling_dtype=None, accumulation_dtype=None
):
    """Return a tiny matrix oracle and complete-row counts for fitted geometry."""
    numeric, categorical, sizes = geometry(
        reference, queries, schema, dtype, scaling_dtype=scaling_dtype
    )

    def matrix(i, j):
        return squared_distances(
            numeric[i], numeric[j], categorical[i], categorical[j], dtype=accumulation_dtype
        )

    return matrix, sizes


def memorization(
    training,
    generated,
    schema,
    backend="cpu",
    dtype="float64",
    *,
    draws=None,
    holdout=None,
):
    """Exercise shared execution with explicit source roles; this is not an oracle."""
    from tabgen_eval.execution.python.distance import compute_memorization_group
    from tabgen_eval.execution.runtime.dispatch import ExecutionSession
    from tabgen_eval.execution.runtime.memorization import prepare_memorization_columns
    from tabgen_eval.metrics import DCR

    options = dict(
        device="cpu" if backend == "reference" else backend,
        dtype=dtype,
    )
    metrics = [DCR(**options)]
    if draws is not None:
        training, generated = training.iloc[draws[0]], generated.iloc[draws[1]]
    if backend == "reference":
        results = compute_memorization_group(training, generated, schema, metrics, holdout=holdout)
    else:
        columns = (
            None
            if holdout is None
            else prepare_memorization_columns(training, generated, holdout, schema)
        )
        context = ExecutionSession(training, generated, schema, columns=columns).context_for(
            metrics[0]
        )
        results = context.compute_memorization_group(
            metrics,
            np.arange(len(training), dtype=np.int64)[None],
            np.arange(len(generated), dtype=np.int64)[None],
            holdout_draws=None
            if holdout is None
            else np.arange(len(holdout), dtype=np.int64)[None],
        )[0]
    return results["dcr"]
