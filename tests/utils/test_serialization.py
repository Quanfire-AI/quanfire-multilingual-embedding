from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

import numpy
import pytest

from multilingual_embedding.core.exceptions import SerializationError
from multilingual_embedding.utils.serialization import (
    from_primitive,
    is_dataclass_type,
    to_primitive,
)


class Colour(StrEnum):
    RED = "red"

    BLUE = "blue"


@dataclass(slots=True)
class Inner:
    value: int = 1


@dataclass(slots=True)
class Outer:
    name: str = "x"

    colour: Colour = Colour.RED

    inner: Inner = field(default_factory=Inner)

    items: list[int] = field(default_factory=list)

    optional: str | None = None

    path: Path | None = None


def test_to_primitive_handles_scalars() -> None:
    assert to_primitive(1) == 1

    assert to_primitive("a") == "a"

    assert to_primitive(None) is None

    assert to_primitive(True) is True


def test_to_primitive_handles_enum_and_datetime() -> None:
    assert to_primitive(Colour.RED) == "red"

    moment = datetime(2026, 1, 1, tzinfo=UTC)

    assert to_primitive(moment) == moment.isoformat()


def test_to_primitive_handles_path_and_sets() -> None:
    assert to_primitive(Path("a/b")) == "a/b"

    assert to_primitive({"b", "a"}) == ["a", "b"]


def test_to_primitive_handles_numpy() -> None:
    assert to_primitive(numpy.int64(5)) == 5

    assert to_primitive(numpy.array([1, 2])) == [1, 2]


def test_to_primitive_rejects_unknown_type() -> None:
    class Opaque:
        pass

    with pytest.raises(SerializationError):
        to_primitive(Opaque())


def test_dataclass_round_trip() -> None:
    original = Outer(
        name="trial",
        colour=Colour.BLUE,
        inner=Inner(value=7),
        items=[1, 2],
        path=Path("a/b"),
    )

    rebuilt = from_primitive(Outer, to_primitive(original))

    assert rebuilt == original

    assert isinstance(rebuilt.colour, Colour)

    assert isinstance(rebuilt.inner, Inner)

    assert isinstance(rebuilt.path, Path)


def test_missing_fields_use_defaults() -> None:
    rebuilt = from_primitive(Outer, {"name": "only"})

    assert rebuilt.name == "only"

    assert rebuilt.inner == Inner()


def test_unknown_field_rejected() -> None:
    """A renamed config key must fail loudly rather than be dropped."""

    with pytest.raises(SerializationError) as error:
        from_primitive(Outer, {"nmae": "typo"})

    assert "nmae" in str(error.value)


def test_invalid_enum_value_rejected() -> None:
    with pytest.raises(SerializationError):
        from_primitive(Outer, {"colour": "chartreuse"})


def test_non_mapping_rejected() -> None:
    with pytest.raises(SerializationError):
        from_primitive(Outer, ["not", "a", "mapping"])


def test_optional_none_is_preserved() -> None:
    assert from_primitive(Outer, {"optional": None}).optional is None


def test_is_dataclass_type() -> None:
    assert is_dataclass_type(Inner)

    assert not is_dataclass_type(Inner())

    assert not is_dataclass_type(int)
