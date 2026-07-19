"""
Precondition helpers.

These functions exist so that argument checking reads as one line at the
top of a constructor and always produces a
:class:`~multilingual_embedding.core.exceptions.ValidationError` carrying
the offending value. Each returns the validated value, which allows the
check to be inlined into an assignment::

    self.window = require_positive(window, name="window")
"""

from __future__ import annotations

from collections.abc import Collection, Sized

from multilingual_embedding.core.exceptions import ValidationError

__all__ = [
    "require_in_range",
    "require_non_empty_collection",
    "require_non_empty_string",
    "require_non_negative",
    "require_one_of",
    "require_positive",
    "require_same_length",
]


def require_non_empty_string(value: object, *, name: str) -> str:
    """Require a string with at least one non-whitespace character."""

    if not isinstance(value, str):
        raise ValidationError(
            f"{name} must be a string",
            name=name,
            received=type(value).__name__,
        )

    if not value.strip():
        raise ValidationError(f"{name} must not be empty", name=name)

    return value


def require_positive[Number: (int, float)](value: Number, *, name: str) -> Number:
    """Require a strictly positive number."""

    if value <= 0:
        raise ValidationError(f"{name} must be > 0", name=name, value=value)

    return value


def require_non_negative[Number: (int, float)](value: Number, *, name: str) -> Number:
    """Require a number that is zero or greater."""

    if value < 0:
        raise ValidationError(f"{name} must be >= 0", name=name, value=value)

    return value


def require_in_range[Number: (int, float)](
    value: Number,
    *,
    name: str,
    minimum: float,
    maximum: float,
    inclusive: bool = True,
) -> Number:
    """
    Require a number inside an interval.

    Parameters
    ----------
    inclusive:
        When True the bounds themselves are permitted.
    """

    in_range = minimum <= value <= maximum if inclusive else minimum < value < maximum

    if not in_range:
        bounds = f"[{minimum}, {maximum}]" if inclusive else f"({minimum}, {maximum})"

        raise ValidationError(
            f"{name} must lie in {bounds}",
            name=name,
            value=value,
        )

    return value


def require_one_of[T](value: T, *, name: str, allowed: Collection[T]) -> T:
    """Require membership in a fixed set of permitted values."""

    if value not in allowed:
        raise ValidationError(
            f"{name} is not a permitted value",
            name=name,
            value=value,
            allowed=sorted(allowed, key=str),
        )

    return value


def require_non_empty_collection[T](value: T, *, name: str) -> T:
    """Require a sized collection holding at least one element."""

    if not isinstance(value, Sized):
        raise ValidationError(
            f"{name} must be a sized collection",
            name=name,
            received=type(value).__name__,
        )

    if len(value) == 0:
        raise ValidationError(f"{name} must not be empty", name=name)

    return value


def require_same_length(
    first: Sized,
    second: Sized,
    *,
    first_name: str,
    second_name: str,
) -> None:
    """Require two collections to be the same length."""

    if len(first) != len(second):
        raise ValidationError(
            f"{first_name} and {second_name} must have the same length",
            **{
                f"{first_name}_length": len(first),
                f"{second_name}_length": len(second),
            },
        )
