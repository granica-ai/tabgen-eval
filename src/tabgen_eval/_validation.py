"""Small runtime checks shared by typed public input boundaries."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeGuard, TypeVar

_Value = TypeVar("_Value")


def require_instance(
    value: object,
    expected: type[_Value],
    message: str,
    *,
    error: type[Exception] = TypeError,
) -> _Value:
    """Validate dynamically supplied Python inputs and retain their narrowed type."""
    if not isinstance(value, expected):
        raise error(message)
    return value


def is_mapping(value: object) -> TypeGuard[Mapping[object, object]]:
    """Narrow an extensible metadata mapping without guessing its value schema."""
    return isinstance(value, Mapping)
