"""Validate evaluation arguments once and describe resolved call options."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from numbers import Integral
from typing import Literal, NoReturn, TypedDict, TypeGuard, cast

import numpy as np
from numpy.typing import NDArray

from ._typing import (
    Precision,
    SeedEntropy,
    SeedLike,
    UncertaintyMode,
    is_integer_scalar,
)
from .execution.runtime.options import DTYPES, validate_execution_options


class _OptionalUncertaintyOptions(TypedDict, total=False):
    n_replicates: int


class UncertaintyOptions(_OptionalUncertaintyOptions):
    """Ordinary dictionary accepted by ``evaluate(uncertainty=...)``.

    ``mode`` is required and selects ``"reference"``, ``"candidate"`` or ``"both"``.
    The optional positive integer ``n_replicates`` defaults to 10. Pass ``None`` to disable
    uncertainty. Sampling remains without replacement within each replicate.
    """

    mode: Literal["reference", "candidate", "both"]


class ResolvedOptions(TypedDict):
    """Private normalized values shared by evaluation's execution layers."""

    uncertainty: UncertaintyMode
    n_replicates: int
    seed: SeedEntropy
    n_jobs: int
    device: str
    dtype: Precision


class _OptionalEvaluationOptions(TypedDict, total=False):
    details: bool


class EvaluationOptions(_OptionalEvaluationOptions):
    """Report metadata in the public argument shape, after validation."""

    uncertainty: UncertaintyOptions | None
    seed: SeedEntropy
    n_jobs: int
    device: str
    dtype: Precision


_UNCERTAINTY_GUIDE = (
    "Required: mode — 'reference', 'candidate' or 'both'.\n"
    "Optional: n_replicates — positive integer, default 10.\n"
    "Example: uncertainty={'mode': 'both', 'n_replicates': 100}\n"
    "Use uncertainty=None to disable uncertainty."
)


def _uncertainty_error(problem: str, *, wrong_type: bool = False) -> NoReturn:
    error = TypeError if wrong_type else ValueError
    raise error(f"Invalid uncertainty options: {problem}.\n\n{_UNCERTAINTY_GUIDE}")


def _integer(value: object, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    value = int(value)
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _is_dictionary(value: object) -> TypeGuard[dict[object, object]]:
    return isinstance(value, dict)


def _resolve_uncertainty(value: object) -> tuple[UncertaintyMode, int]:
    if value is None:
        return "none", 10
    if not _is_dictionary(value):
        _uncertainty_error("expected a dictionary or None", wrong_type=True)
    supplied = dict(value)
    unknown = supplied.keys() - {"mode", "n_replicates"}
    if unknown:
        names = ", ".join(sorted(repr(key) for key in unknown))
        _uncertainty_error(f"unknown key(s) {names}")
    if "mode" not in supplied:
        _uncertainty_error("missing required key 'mode'")
    mode = supplied["mode"]
    if not isinstance(mode, str) or mode not in {"reference", "candidate", "both"}:
        _uncertainty_error(f"mode must be 'reference', 'candidate' or 'both', got {mode!r}")
    try:
        count = _integer(supplied.get("n_replicates", 10), "n_replicates", minimum=1)
    except ValueError as error:
        _uncertainty_error(str(error))
    # Explicit branches narrow the runtime-validated string to the accepted modes.
    if mode == "candidate":
        return "candidate", count
    return ("reference" if mode == "reference" else "both"), count


def _is_array(value: object) -> TypeGuard[NDArray[np.generic]]:
    return isinstance(value, np.ndarray)


def _is_sequence(value: object) -> TypeGuard[Sequence[object]]:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _seed_value(value: object) -> SeedEntropy:
    """Own NumPy's accepted deterministic entropy in JSON-compatible storage."""
    if _is_array(value):
        return _seed_value(value.tolist())
    if isinstance(value, int):
        return value
    if is_integer_scalar(value):
        return int(value)
    if _is_sequence(value):
        return [_seed_value(item) for item in value]
    raise ValueError("seed must contain only integers")


def _validate_seed(seed: object) -> SeedEntropy:
    try:
        # NumPy validates raw entropy, including unsupported containers. Its
        # ArrayLike annotation omits accepted runtime inputs such as range;
        # adapt only this validation boundary, without converting the input.
        seed_sequence = cast(Callable[[object], np.random.SeedSequence], np.random.SeedSequence)
        if seed is None:
            raise ValueError("seed must be deterministic")
        seed_sequence(seed)
        entropy = _seed_value(seed)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "seed must be deterministic entropy accepted by NumPy SeedSequence "
            "(a non-negative integer or sequence of non-negative integers)"
        ) from error
    return entropy


def resolve_options(
    *,
    uncertainty: UncertaintyOptions | None = None,
    seed: SeedLike = 42,
    n_jobs: int = 8,
    device: str = "auto",
    dtype: Precision = "float64",
) -> ResolvedOptions:
    """Validate public arguments without mutating their caller-owned values."""
    mode, count = _resolve_uncertainty(uncertainty)
    _validate_choice(dtype, "dtype", DTYPES)
    n_jobs = _integer(n_jobs, "n_jobs")
    if n_jobs == 0:
        raise ValueError("n_jobs must be non-zero (1 serial, >1 workers, -1 all cores)")
    validate_execution_options(device, dtype)
    entropy = _validate_seed(seed)
    return {
        "uncertainty": mode,
        "n_replicates": count,
        "seed": entropy,
        "n_jobs": n_jobs,
        "device": device,
        "dtype": dtype,
    }


def _validate_choice(value: object, name: str, choices: Sequence[str]) -> None:
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f"{name} must be one of {choices}, got {value!r}")


def describe_options(options: ResolvedOptions) -> EvaluationOptions:
    """Describe resolved arguments using the public uncertainty dictionary shape."""
    return {
        "uncertainty": (
            None
            if options["uncertainty"] == "none"
            else {"mode": options["uncertainty"], "n_replicates": options["n_replicates"]}
        ),
        "seed": _seed_value(options["seed"]),
        "device": options["device"],
        "dtype": options["dtype"],
        "n_jobs": options["n_jobs"],
    }
