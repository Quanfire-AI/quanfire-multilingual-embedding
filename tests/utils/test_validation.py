from __future__ import annotations

import pytest

from multilingual_embedding.core.exceptions import ValidationError
from multilingual_embedding.utils.validation import (
    require_in_range,
    require_non_empty_collection,
    require_non_empty_string,
    require_non_negative,
    require_one_of,
    require_positive,
    require_same_length,
)


def test_non_empty_string_returns_value() -> None:
    assert require_non_empty_string("hello", name="value") == "hello"


@pytest.mark.parametrize("value", ["", "   ", None, 5])
def test_non_empty_string_rejects(value: object) -> None:
    with pytest.raises(ValidationError):
        require_non_empty_string(value, name="value")


def test_positive_accepts_and_rejects() -> None:
    assert require_positive(3, name="value") == 3

    with pytest.raises(ValidationError):
        require_positive(0, name="value")

    with pytest.raises(ValidationError):
        require_positive(-1, name="value")


def test_non_negative_allows_zero() -> None:
    assert require_non_negative(0, name="value") == 0

    with pytest.raises(ValidationError):
        require_non_negative(-1, name="value")


def test_in_range_inclusive_and_exclusive() -> None:
    assert require_in_range(1.0, name="value", minimum=0.0, maximum=1.0) == 1.0

    with pytest.raises(ValidationError):
        require_in_range(1.0, name="value", minimum=0.0, maximum=1.0, inclusive=False)


def test_one_of() -> None:
    assert require_one_of("a", name="value", allowed={"a", "b"}) == "a"

    with pytest.raises(ValidationError):
        require_one_of("z", name="value", allowed={"a", "b"})


def test_non_empty_collection() -> None:
    assert require_non_empty_collection([1], name="value") == [1]

    with pytest.raises(ValidationError):
        require_non_empty_collection([], name="value")

    with pytest.raises(ValidationError):
        require_non_empty_collection(5, name="value")


def test_same_length() -> None:
    require_same_length([1, 2], "ab", first_name="ids", second_name="tokens")

    with pytest.raises(ValidationError):
        require_same_length([1], "ab", first_name="ids", second_name="tokens")


def test_error_carries_structured_context() -> None:
    with pytest.raises(ValidationError) as error:
        require_positive(-5, name="dimension")

    assert error.value.context["value"] == -5

    assert error.value.context["name"] == "dimension"
