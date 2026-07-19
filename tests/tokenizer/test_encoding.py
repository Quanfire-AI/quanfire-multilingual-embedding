"""Tests for the Encoding value object."""

from __future__ import annotations

import pytest

from multilingual_embedding.common.span import Span
from multilingual_embedding.core.exceptions import ValidationError
from multilingual_embedding.tokenizer.encoding import Encoding


def make_encoding() -> Encoding:
    return Encoding(
        ids=[5, 6, 7],
        tokens=["नमस्ते", "दुनिया", "!"],
        spans=[Span(0, 6), Span(7, 13), Span(13, 14)],
    )


class TestConstruction:
    def test_length_and_len_agree(self) -> None:
        encoding = make_encoding()

        assert encoding.length == 3

        assert len(encoding) == 3

    def test_spans_and_mask_are_optional(self) -> None:
        encoding = Encoding(ids=[1], tokens=["a"])

        assert encoding.spans is None

        assert encoding.attention_mask is None

    def test_mismatched_ids_and_tokens_are_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Encoding(ids=[1, 2], tokens=["a"])

    def test_mismatched_spans_are_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Encoding(ids=[1], tokens=["a"], spans=[Span(0, 1), Span(1, 2)])

    def test_mismatched_attention_mask_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Encoding(ids=[1], tokens=["a"], attention_mask=[1, 1])

    def test_empty_encoding_is_valid(self) -> None:
        assert len(Encoding(ids=[], tokens=[])) == 0


class TestToDict:
    def test_spans_become_pairs(self) -> None:
        payload = make_encoding().to_dict()

        assert payload["ids"] == [5, 6, 7]

        assert payload["spans"] == [[0, 6], [7, 13], [13, 14]]

        assert payload["attention_mask"] is None

    def test_absent_spans_stay_none(self) -> None:
        assert Encoding(ids=[1], tokens=["a"]).to_dict()["spans"] is None

    def test_result_is_a_copy(self) -> None:
        encoding = make_encoding()

        payload = encoding.to_dict()

        payload["ids"].append(99)

        assert encoding.ids == [5, 6, 7]


class TestTruncate:
    def test_keeps_the_prefix(self) -> None:
        truncated = make_encoding().truncate(2)

        assert truncated.ids == [5, 6]

        assert truncated.tokens == ["नमस्ते", "दुनिया"]

        assert truncated.spans == [Span(0, 6), Span(7, 13)]

    def test_longer_budget_is_a_no_op(self) -> None:
        encoding = make_encoding()

        assert encoding.truncate(10).ids == encoding.ids

    def test_returns_a_new_instance(self) -> None:
        encoding = make_encoding()

        assert encoding.truncate(1) is not encoding

        assert encoding.length == 3

    def test_truncate_to_zero(self) -> None:
        assert len(make_encoding().truncate(0)) == 0

    def test_negative_budget_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_encoding().truncate(-1)

    def test_attention_mask_is_truncated_too(self) -> None:
        encoding = Encoding(ids=[1, 2], tokens=["a", "b"], attention_mask=[1, 1])

        assert encoding.truncate(1).attention_mask == [1]


class TestPadTo:
    def test_pads_ids_tokens_and_mask(self) -> None:
        padded = Encoding(ids=[5], tokens=["hi"]).pad_to(3, pad_id=0)

        assert padded.ids == [5, 0, 0]

        assert padded.tokens == ["hi", "<pad>", "<pad>"]

        assert padded.attention_mask == [1, 0, 0]

    def test_pad_token_is_configurable(self) -> None:
        padded = Encoding(ids=[5], tokens=["hi"]).pad_to(2, pad_id=0, pad_token="[PAD]")

        assert padded.tokens == ["hi", "[PAD]"]

    def test_spans_are_dropped_because_padding_has_no_source_position(self) -> None:
        padded = make_encoding().pad_to(5, pad_id=0)

        assert padded.spans is None

    def test_padding_to_the_current_length_is_a_no_op_but_adds_a_mask(self) -> None:
        padded = make_encoding().pad_to(3, pad_id=0)

        assert padded.ids == [5, 6, 7]

        assert padded.attention_mask == [1, 1, 1]

    def test_existing_mask_is_preserved(self) -> None:
        encoding = Encoding(ids=[1, 2], tokens=["a", "<pad>"], attention_mask=[1, 0])

        assert encoding.pad_to(3, pad_id=0).attention_mask == [1, 0, 0]

    def test_shorter_target_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_encoding().pad_to(1, pad_id=0)

    def test_negative_target_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_encoding().pad_to(-1, pad_id=0)

    def test_truncate_then_pad_yields_the_requested_length(self) -> None:
        result = make_encoding().truncate(2).pad_to(4, pad_id=0)

        assert len(result) == 4

        assert result.attention_mask == [1, 1, 0, 0]
