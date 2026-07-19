"""
Tests for statistics, validators, offsets, iteration and language utilities.
"""

from __future__ import annotations

import pytest

from multilingual_embedding.common.span import Span
from multilingual_embedding.core.exceptions import ValidationError
from multilingual_embedding.corpus.corpus import Corpus
from multilingual_embedding.corpus.document import Document
from multilingual_embedding.corpus.iterator import SentenceStream, batched, take
from multilingual_embedding.corpus.language import (
    expected_script,
    infer_language,
    language_name,
    normalize_language_code,
)
from multilingual_embedding.corpus.offsets import (
    invert_spans,
    merge_overlapping,
    resolve_chain,
    spans_are_ordered,
    spans_within,
)
from multilingual_embedding.corpus.script import Script
from multilingual_embedding.corpus.statistics import (
    LengthSummary,
    StatisticsAccumulator,
    compute_statistics,
)
from multilingual_embedding.corpus.validators import (
    DocumentDeduplicator,
    SentenceFilter,
    validate_document,
)


class TestLanguage:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("en", "en"), ("EN", "en"), ("en-GB", "en"), ("en_gb", "en"), ("  hi  ", "hi")],
    )
    def test_normalization(self, raw: str, expected: str) -> None:
        assert normalize_language_code(raw) == expected

    @pytest.mark.parametrize("raw", ["", "e", "eng", "1a", None])
    def test_invalid_codes_rejected(self, raw: object) -> None:
        with pytest.raises(ValidationError):
            normalize_language_code(raw)  # type: ignore[arg-type]

    def test_language_name(self) -> None:
        assert language_name("hi") == "Hindi"

        assert language_name("zz") is None

    def test_expected_script(self) -> None:
        assert expected_script("ta") is Script.TAMIL

        assert expected_script("zz") is None

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("नमस्ते दुनिया", "hi"),
            ("வணக்கம்", "ta"),
            ("안녕하세요", "ko"),
            ("สวัสดี", "th"),
            ("こんにちは", "ja"),
        ],
    )
    def test_inference_for_unambiguous_scripts(self, text: str, expected: str) -> None:
        assert infer_language(text) == expected

    @pytest.mark.parametrize("text", ["hello world", "مرحبا", "Привет", "中文"])
    def test_no_inference_for_shared_scripts(self, text: str) -> None:
        """Latin, Arabic, Cyrillic and Han each serve many languages."""

        assert infer_language(text) is None

    def test_no_inference_for_mixed_or_empty(self) -> None:
        assert infer_language("") is None

        assert infer_language("hello नमस्ते hello नमस्ते") is None


class TestOffsets:
    def test_resolve_chain(self) -> None:
        assert resolve_chain([Span(100, 400), Span(10, 20)]) == Span(110, 120)

    def test_resolve_deep_chain(self) -> None:
        assert resolve_chain([Span(10, 100), Span(5, 50), Span(2, 8)]) == Span(17, 23)

    def test_resolve_single(self) -> None:
        assert resolve_chain([Span(3, 9)]) == Span(3, 9)

    def test_resolve_empty_rejected(self) -> None:
        with pytest.raises(ValidationError):
            resolve_chain([])

    def test_ordering_check(self) -> None:
        assert spans_are_ordered([Span(0, 5), Span(5, 10)])

        assert not spans_are_ordered([Span(5, 10), Span(0, 5)])

        assert not spans_are_ordered([Span(0, 6), Span(5, 10)])

    def test_within_check(self) -> None:
        assert spans_within([Span(0, 5)], length=10)

        assert not spans_within([Span(0, 15)], length=10)

    def test_invert_recovers_gaps(self) -> None:
        assert invert_spans([Span(2, 5), Span(8, 10)], length=12) == [
            Span(0, 2),
            Span(5, 8),
            Span(10, 12),
        ]

    def test_invert_with_no_gaps(self) -> None:
        assert invert_spans([Span(0, 10)], length=10) == []

    def test_merge_overlapping(self) -> None:
        assert merge_overlapping([Span(0, 5), Span(3, 8), Span(20, 25)]) == [
            Span(0, 8),
            Span(20, 25),
        ]

    def test_merge_empty(self) -> None:
        assert merge_overlapping([]) == []


class TestIteration:
    def test_stream_restarts_each_pass(self) -> None:
        calls = {"count": 0}

        def factory():
            calls["count"] += 1

            return iter(["a", "b", "c"])

        stream = SentenceStream(factory)

        assert list(stream) == ["a", "b", "c"]

        assert list(stream) == ["a", "b", "c"]

        assert calls["count"] == 2

    def test_stream_limit(self) -> None:
        stream = SentenceStream(lambda: iter(["a", "b", "c"]), limit=2)

        assert list(stream) == ["a", "b"]

    def test_stream_map_composes(self) -> None:
        stream = SentenceStream(lambda: iter(["a", "b"]))

        mapped = stream.map(str.upper).map(lambda text: text + "!")

        assert list(mapped) == ["A!", "B!"]

    def test_map_does_not_mutate_original(self) -> None:
        stream = SentenceStream(lambda: iter(["a"]))

        stream.map(str.upper)

        assert list(stream) == ["a"]

    def test_count(self) -> None:
        assert SentenceStream(lambda: iter(["a", "b"])).count() == 2

    def test_batched(self) -> None:
        assert list(batched(["a", "b", "c"], 2)) == [["a", "b"], ["c"]]

    def test_batched_rejects_zero(self) -> None:
        with pytest.raises(ValueError):
            list(batched(["a"], 0))

    def test_take(self) -> None:
        assert take(["a", "b", "c"], 2) == ["a", "b"]


class TestStatistics:
    def test_counts(self, small_corpus: Corpus) -> None:
        statistics = compute_statistics(small_corpus)

        assert statistics.document_count == 2

        assert statistics.sentence_count == 6

        assert statistics.word_count > 0

    def test_language_and_script_breakdown(self, small_corpus: Corpus) -> None:
        statistics = compute_statistics(small_corpus)

        assert statistics.languages == {"en": 1, "hi": 1}

        assert set(statistics.scripts) == {"Latn", "Deva"}

    def test_devanagari_words_counted_correctly(self) -> None:
        """
        Combining marks must not split words.

        A naive word regex counts "नमस्ते" as five words; this guards the
        fix in the segmenter.
        """

        statistics = compute_statistics([Document.from_text("नमस्ते दुनिया।", language="hi")])

        assert statistics.word_count == 2

    def test_type_token_ratio(self) -> None:
        statistics = compute_statistics([Document.from_text("a a a b.", language="en")])

        assert 0.0 < statistics.type_token_ratio <= 1.0

    def test_top_words_are_sorted(self) -> None:
        statistics = compute_statistics(
            [Document.from_text("the the the cat cat dog.", language="en")]
        )

        assert statistics.top_words[0] == ("the", 3)

    def test_word_counting_is_case_folded(self) -> None:
        statistics = compute_statistics([Document.from_text("The the THE.", language="en")])

        assert statistics.unique_words == 1

    def test_accumulator_is_incremental(self, small_corpus: Corpus) -> None:
        accumulator = StatisticsAccumulator()

        for document in small_corpus:
            accumulator.add(document)

        assert accumulator.result().document_count == 2

    def test_word_cap_sets_truncation_flag(self) -> None:
        accumulator = StatisticsAccumulator(max_tracked_words=2)

        accumulator.add(Document.from_text("alpha beta gamma delta.", language="en"))

        result = accumulator.result()

        assert result.truncated_vocabulary

        assert result.unique_words == 2

    def test_to_dict_is_serialisable(self, small_corpus: Corpus) -> None:
        import json

        json.dumps(compute_statistics(small_corpus).to_dict())

    def test_empty_input(self) -> None:
        statistics = compute_statistics([])

        assert statistics.document_count == 0

        assert statistics.type_token_ratio == 0.0


class TestLengthSummary:
    def test_empty(self) -> None:
        summary = LengthSummary.from_values([])

        assert summary.count == 0

        assert summary.mean == 0.0

    def test_single_value(self) -> None:
        summary = LengthSummary.from_values([5])

        assert summary.median == 5.0

        assert summary.p99 == 5.0

    def test_matches_numpy_percentiles(self) -> None:
        """Reported figures must agree with downstream numpy analysis."""

        import numpy

        values = list(range(1, 101))

        summary = LengthSummary.from_values(values)

        assert summary.median == pytest.approx(float(numpy.percentile(values, 50)))

        assert summary.p95 == pytest.approx(float(numpy.percentile(values, 95)))

        assert summary.p99 == pytest.approx(float(numpy.percentile(values, 99)))


class TestSentenceFilter:
    def test_length_bounds(self) -> None:
        sentence_filter = SentenceFilter(min_characters=3, max_characters=10)

        assert not sentence_filter.accepts("ab")

        assert sentence_filter.accepts("abcd")

        assert not sentence_filter.accepts("a" * 20)

    def test_blank_rejected(self) -> None:
        sentence_filter = SentenceFilter()

        assert not sentence_filter.accepts("   ")

        assert sentence_filter.report.rejected_blank == 1

    def test_letterless_rejected(self) -> None:
        assert not SentenceFilter().accepts("12345 !!!")

    def test_non_latin_letters_accepted(self) -> None:
        """Letter detection must be Unicode aware, not ASCII."""

        sentence_filter = SentenceFilter()

        assert sentence_filter.accepts("नमस्ते दुनिया")

        assert sentence_filter.accepts("こんにちは")

    def test_encoding_damage_rejected(self) -> None:
        assert not SentenceFilter().accepts("hi ���������������")

    def test_script_mismatch_rejected_when_configured(self) -> None:
        sentence_filter = SentenceFilter(expected_script=Script.DEVANAGARI)

        assert sentence_filter.accepts("नमस्ते दुनिया")

        assert not sentence_filter.accepts("hello world")

    def test_report_totals(self) -> None:
        sentence_filter = SentenceFilter(min_characters=3)

        for text in ["ok text", "ab", "  ", "also fine"]:
            sentence_filter.accepts(text)

        report = sentence_filter.report

        assert report.total == 4

        assert report.accepted == 2

        assert report.rejected == 2

        assert report.acceptance_rate == 0.5

    def test_apply_prunes_document(self) -> None:
        document = Document.from_text("Good sentence here. A. Another good one.", language="en")

        SentenceFilter(min_characters=5).apply(document)

        assert document.sentence_count == 2

    def test_apply_drops_empty_paragraphs(self) -> None:
        document = Document.from_text("A.\n\nA real sentence here.", language="en")

        SentenceFilter(min_characters=5).apply(document)

        assert document.paragraph_count == 1


class TestDeduplicator:
    def test_exact_duplicates_detected(self) -> None:
        deduplicator = DocumentDeduplicator()

        assert not deduplicator.is_duplicate(Document.from_text("Same text."))

        assert deduplicator.is_duplicate(Document.from_text("Same text."))

        assert deduplicator.duplicate_count == 1

    def test_whitespace_variation_is_a_duplicate(self) -> None:
        deduplicator = DocumentDeduplicator()

        deduplicator.is_duplicate(Document.from_text("Same   text."))

        assert deduplicator.is_duplicate(Document.from_text("Same text."))

    def test_different_text_is_not_duplicate(self) -> None:
        deduplicator = DocumentDeduplicator()

        deduplicator.is_duplicate(Document.from_text("One."))

        assert not deduplicator.is_duplicate(Document.from_text("Two."))

    def test_filter_keeps_first_occurrence(self) -> None:
        documents = [Document.from_text(text) for text in ["a.", "b.", "a."]]

        assert len(list(DocumentDeduplicator().filter(documents))) == 2


class TestValidateDocument:
    def test_sound_document_has_no_problems(self, english_document: Document) -> None:
        assert validate_document(english_document) == []

    def test_empty_document_is_reported(self) -> None:
        assert validate_document(Document.from_text("")) != []

    def test_span_inconsistency_is_reported(self, english_document: Document) -> None:
        english_document.paragraphs[0].sentences[0].text = "tampered"

        problems = validate_document(english_document)

        assert any("span" in problem for problem in problems)
