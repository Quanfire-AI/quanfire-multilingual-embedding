"""
Conversion between framework objects and plain data.

Everything the framework persists — metadata, configuration, statistics
— passes through :func:`to_primitive`, which reduces dataclasses, enums,
datetimes, paths and numpy scalars to JSON compatible values.

Deserialisation is deliberately narrow: :func:`from_primitive` populates
a known dataclass type from a mapping. The framework never reconstructs
arbitrary classes from serialised input, since that would let a corpus
file decide which code runs.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeGuard, get_args, get_origin, get_type_hints

from multilingual_embedding.core.exceptions import SerializationError

if TYPE_CHECKING:
    # `DataclassInstance` is the protocol `dataclasses.fields()` accepts. It
    # exists only in typeshed, so it must not be imported at runtime.
    from _typeshed import DataclassInstance

__all__ = [
    "from_primitive",
    "is_dataclass_type",
    "to_primitive",
]


def is_dataclass_type(candidate: Any) -> TypeGuard[type[DataclassInstance]]:
    """Return True if ``candidate`` is a dataclass type (not an instance)."""

    return isinstance(candidate, type) and dataclasses.is_dataclass(candidate)


def to_primitive(value: Any) -> Any:
    """
    Recursively reduce a value to JSON compatible primitives.

    Handled specially:

    - dataclasses become dictionaries
    - enums become their ``value``
    - ``datetime`` and ``date`` become ISO 8601 strings
    - ``Path`` becomes a string
    - ``set`` and ``frozenset`` become sorted lists, for stable output
    - numpy scalars and arrays become Python scalars and lists

    Unknown types raise rather than being silently stringified, so a
    persistence gap is caught during development instead of producing a
    file that cannot be read back.
    """

    # Enums are checked first: the framework's enums subclass `str`, so a
    # scalar check would match them and return the member itself rather
    # than its value, which YAML cannot represent.
    if isinstance(value, Enum):
        return to_primitive(value.value)

    if value is None or isinstance(value, (str, bool, int, float)):
        return value

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if isinstance(value, Path):
        return str(value)

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: to_primitive(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }

    if isinstance(value, Mapping):
        return {str(key): to_primitive(item) for key, item in value.items()}

    if isinstance(value, (set, frozenset)):
        return [to_primitive(item) for item in sorted(value, key=str)]

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [to_primitive(item) for item in value]

    # numpy is a hard dependency, but importing it lazily keeps this
    # module usable in contexts where the array stack is not needed.
    primitive = _numpy_to_primitive(value)

    if primitive is not _UNHANDLED:
        return primitive

    raise SerializationError(
        "Value is not serialisable",
        type=type(value).__name__,
    )


def from_primitive[T](target_type: type[T], data: Any) -> T:
    """
    Populate a dataclass instance from primitive data.

    Nested dataclasses, enums, datetimes and optional fields are
    resolved using the target's type hints. Keys absent from ``data``
    fall back to the field default; unknown keys are rejected so that a
    renamed configuration field does not silently lose its value.

    Parameters
    ----------
    target_type:
        The dataclass type to construct.

    data:
        A mapping of field names to primitive values.
    """

    # Bound through a separate name so the TypeGuard narrowing survives for
    # `dataclasses.fields()` without discarding the `T` on ``target_type``.
    dataclass_type = target_type if is_dataclass_type(target_type) else None

    if dataclass_type is None:
        coerced: T = _coerce_scalar(target_type, data)

        return coerced

    if not isinstance(data, Mapping):
        raise SerializationError(
            "Expected a mapping to build a dataclass",
            target=target_type.__name__,
            received=type(data).__name__,
        )

    hints = get_type_hints(target_type)

    field_names = {field.name for field in dataclasses.fields(dataclass_type)}

    unknown = sorted(set(data) - field_names)

    if unknown:
        raise SerializationError(
            "Unknown fields for target type",
            target=target_type.__name__,
            unknown=unknown,
            known=sorted(field_names),
        )

    kwargs: dict[str, Any] = {}

    for field in dataclasses.fields(dataclass_type):
        if field.name not in data:
            continue

        kwargs[field.name] = _coerce_scalar(
            hints.get(field.name, Any),
            data[field.name],
        )

    try:
        return target_type(**kwargs)
    except TypeError as error:
        raise SerializationError(
            "Could not construct target type",
            target=target_type.__name__,
            reason=str(error),
        ) from error


def _coerce_scalar(annotation: Any, value: Any) -> Any:
    """Convert one primitive value according to its type annotation."""

    if annotation is Any or annotation is None:
        return value

    origin = get_origin(annotation)

    # Optional[X] / Union[X, None]
    if origin is not None and type(None) in get_args(annotation):
        if value is None:
            return None

        remaining = [arg for arg in get_args(annotation) if arg is not type(None)]

        if len(remaining) == 1:
            return _coerce_scalar(remaining[0], value)

        return value

    if origin in (list, Sequence) and isinstance(value, Sequence) and not isinstance(value, str):
        args = get_args(annotation)

        item_type = args[0] if args else Any

        return [_coerce_scalar(item_type, item) for item in value]

    if origin is dict and isinstance(value, Mapping):
        args = get_args(annotation)

        value_type = args[1] if len(args) == 2 else Any

        return {key: _coerce_scalar(value_type, item) for key, item in value.items()}

    if is_dataclass_type(annotation):
        return from_primitive(annotation, value)

    if isinstance(annotation, type) and issubclass(annotation, Enum):
        try:
            return annotation(value)
        except ValueError as error:
            raise SerializationError(
                "Invalid enum value",
                enum=annotation.__name__,
                value=value,
            ) from error

    if annotation is datetime and isinstance(value, str):
        return datetime.fromisoformat(value)

    if annotation is date and isinstance(value, str):
        return date.fromisoformat(value)

    if annotation is Path and isinstance(value, str):
        return Path(value)

    return value


_UNHANDLED = object()


def _numpy_to_primitive(value: Any) -> Any:
    """Reduce numpy scalars and arrays, or return the sentinel."""

    try:
        import numpy
    except ImportError:  # pragma: no cover - numpy is a declared dependency
        return _UNHANDLED

    if isinstance(value, numpy.generic):
        return value.item()

    if isinstance(value, numpy.ndarray):
        return value.tolist()

    return _UNHANDLED
