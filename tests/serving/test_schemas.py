"""
What the endpoint refuses before any model is loaded.

Validation here is not ceremony. Every rule below rejects an input that
would otherwise produce a well-formed response: a vector of the right
shape and norm, indistinguishable from a good one, computed from
something the caller did not mean to send. An empty-string field passed
through a template embeds to a point in the space and takes a rank in a
result list. A 5,000-item batch is embedded in one pass and blocks every
other caller behind it.

These tests need pydantic, which arrives with the ``serve`` extra.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pydantic", reason="requires the serve extra")

from pydantic import ValidationError

from multilingual_embedding.serving.schemas import (
    MAX_BATCH,
    MAX_INPUT_CHARACTERS,
    EmbeddingsRequest,
)


class TestOneInputOrMany:
    def test_a_bare_string_becomes_a_one_item_batch(self) -> None:
        assert EmbeddingsRequest(input="नमस्ते").texts() == ["नमस्ते"]

    def test_a_list_arrives_in_order(self) -> None:
        request = EmbeddingsRequest(input=["first", "second", "third"])

        assert request.texts() == ["first", "second", "third"]

    def test_texts_returns_a_copy(self) -> None:
        """Mutating the returned list must not edit the request."""

        request = EmbeddingsRequest(input=["a", "b"])

        request.texts().append("c")

        assert request.texts() == ["a", "b"]


class TestWhatIsRefused:
    def test_an_empty_batch(self) -> None:
        with pytest.raises(ValidationError, match="at least one string"):
            EmbeddingsRequest(input=[])

    def test_an_empty_string(self) -> None:
        with pytest.raises(ValidationError, match=r"input\[0\] is empty"):
            EmbeddingsRequest(input="")

    def test_a_whitespace_only_string(self) -> None:
        """
        The one worth spelling out.

        This is what an unfilled template field looks like by the time it
        reaches a wire format, and it is the only invalid input here that
        would otherwise succeed all the way to a vector.
        """

        with pytest.raises(ValidationError, match=r"input\[1\] is empty or whitespace"):
            EmbeddingsRequest(input=["real text", "   \n\t "])

    def test_a_batch_over_the_ceiling(self) -> None:
        with pytest.raises(ValidationError, match=f"the maximum is {MAX_BATCH}"):
            EmbeddingsRequest(input=["x"] * (MAX_BATCH + 1))

    def test_a_batch_exactly_at_the_ceiling_is_allowed(self) -> None:
        """An off-by-one here would reject a caller who read the docs."""

        assert len(EmbeddingsRequest(input=["x"] * MAX_BATCH).texts()) == MAX_BATCH

    def test_a_string_over_the_character_ceiling(self) -> None:
        with pytest.raises(ValidationError, match=f"the maximum is {MAX_INPUT_CHARACTERS}"):
            EmbeddingsRequest(input="x" * (MAX_INPUT_CHARACTERS + 1))

    def test_an_unknown_input_type(self) -> None:
        """
        A typo must not silently mean "no side".

        ``input_type="Query"`` reaching prefix_for as an unrecognised
        value would fall through to the passage branch and embed a query
        as a passage, which is precisely the failure the whole module
        exists to prevent.
        """

        with pytest.raises(ValidationError):
            EmbeddingsRequest(input="x", input_type="Query")  # type: ignore[arg-type]

    def test_an_unsupported_encoding_format(self) -> None:
        """Base64 is in the standard schema; this server does not offer it."""

        with pytest.raises(ValidationError):
            EmbeddingsRequest(input="x", encoding_format="base64")  # type: ignore[arg-type]


class TestWhatIsOptional:
    def test_input_type_defaults_to_unset(self) -> None:
        """Unset must be distinguishable from a chosen side, not empty."""

        assert EmbeddingsRequest(input="x").input_type is None

    def test_model_defaults_to_unset(self) -> None:
        assert EmbeddingsRequest(input="x").model is None
