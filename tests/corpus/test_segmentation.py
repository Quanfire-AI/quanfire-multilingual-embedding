from __future__ import annotations

import pytest

from multilingual_embedding.corpus.segmentation import (
    split_paragraphs,
    split_sentences,
    split_words,
)


def sentences(text: str, language: str | None = None) -> list[str]:
    return [span.slice(text) for span in split_sentences(text, language=language)]


class TestSpanIntegrity:
    """Every returned span must slice back to the text it represents."""

    @pytest.mark.parametrize(
        "text",
        [
            "Hello there. How are you?",
            "नमस्ते। आप कैसे हैं?",
            "今天天气很好。你好吗？",
            "مرحبا. كيف حالك؟",
        ],
    )
    def test_spans_are_ordered_and_within_bounds(self, text: str) -> None:
        spans = split_sentences(text)

        previous_end = 0

        for span in spans:
            assert span.start >= previous_end

            assert span.end <= len(text)

            previous_end = span.end


class TestLatinSegmentation:
    def test_basic_split(self) -> None:
        assert sentences("Hello there. How are you?") == [
            "Hello there.",
            "How are you?",
        ]

    def test_abbreviation_does_not_split(self) -> None:
        assert sentences("Dr. Smith arrived. He was late.", "en") == [
            "Dr. Smith arrived.",
            "He was late.",
        ]

    def test_initials_do_not_split(self) -> None:
        assert sentences("J. R. R. Tolkien wrote books. They are long.", "en") == [
            "J. R. R. Tolkien wrote books.",
            "They are long.",
        ]

    def test_decimal_number_does_not_split(self) -> None:
        assert sentences("The value is 3.14 exactly. Nothing more.", "en") == [
            "The value is 3.14 exactly.",
            "Nothing more.",
        ]

    def test_closing_quote_stays_with_sentence(self) -> None:
        assert sentences('He said "stop!" Then he left.', "en") == [
            'He said "stop!"',
            "Then he left.",
        ]

    def test_repeated_terminators_are_one_boundary(self) -> None:
        assert sentences("Really?! Yes.", "en") == ["Really?!", "Yes."]

    def test_final_sentence_without_terminator(self) -> None:
        assert sentences("First one. Trailing text", "en") == [
            "First one.",
            "Trailing text",
        ]


class TestNonLatinSegmentation:
    def test_devanagari_danda(self) -> None:
        assert sentences("नमस्ते। आप कैसे हैं?", "hi") == ["नमस्ते।", "आप कैसे हैं?"]

    def test_chinese_needs_no_following_space(self) -> None:
        """CJK terminators are not followed by whitespace."""

        assert sentences("今天天气很好。你好吗？我很好。", "zh") == [
            "今天天气很好。",
            "你好吗？",
            "我很好。",
        ]

    def test_arabic_question_mark(self) -> None:
        assert sentences("مرحبا. كيف حالك؟", "ar") == ["مرحبا.", "كيف حالك؟"]


class TestEdgeCases:
    @pytest.mark.parametrize("text", ["", "   ", "\n\n\t "])
    def test_blank_input_yields_nothing(self, text: str) -> None:
        assert split_sentences(text) == []

        assert split_paragraphs(text) == []

    def test_text_without_terminator_is_one_sentence(self) -> None:
        assert sentences("no terminator here") == ["no terminator here"]

    def test_whitespace_is_trimmed_from_spans(self) -> None:
        assert sentences("  Hello.   World.  ") == ["Hello.", "World."]


class TestParagraphs:
    def test_blank_line_separates(self) -> None:
        text = "First para.\nStill first.\n\nSecond para."

        assert [span.slice(text) for span in split_paragraphs(text)] == [
            "First para.\nStill first.",
            "Second para.",
        ]

    def test_repeated_blank_lines_produce_no_empty_paragraph(self) -> None:
        text = "One.\n\n\n\nTwo."

        assert len(split_paragraphs(text)) == 2

    def test_single_paragraph(self) -> None:
        assert len(split_paragraphs("Just one paragraph here.")) == 1


class TestWords:
    def test_basic_words(self) -> None:
        text = "Hello, world! It's fine."

        assert [span.slice(text) for span in split_words(text)] == [
            "Hello",
            "world",
            "It's",
            "fine",
        ]

    def test_devanagari_words(self) -> None:
        text = "नमस्ते दुनिया"

        assert len(split_words(text)) == 2

    def test_spans_slice_correctly(self) -> None:
        text = "one two three"

        for span in split_words(text):
            assert span.slice(text) == text[span.start : span.end]
