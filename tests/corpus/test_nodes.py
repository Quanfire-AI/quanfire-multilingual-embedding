from __future__ import annotations

import pytest

from multilingual_embedding.common.span import Span
from multilingual_embedding.corpus.document import Document
from multilingual_embedding.corpus.exceptions import CorpusError
from multilingual_embedding.corpus.paragraph import Paragraph
from multilingual_embedding.corpus.script import Script
from multilingual_embedding.corpus.sentence import Sentence
from multilingual_embedding.corpus.token import Token


class TestToken:
    def test_create_derives_span(self) -> None:
        token = Token.create("hello", start=4)

        assert token.span == Span(4, 9)

    def test_script_detection(self) -> None:
        assert Token.create("hello").script is Script.LATIN

        assert Token.create("नमस्ते").script is Script.DEVANAGARI

    def test_punctuation_detection(self) -> None:
        assert Token.create("...").is_punctuation

        assert not Token.create("word").is_punctuation


class TestSentence:
    def test_create_and_language(self) -> None:
        sentence = Sentence.create("Hello world", start=2, language="en")

        assert sentence.span == Span(2, 13)

        assert sentence.language == "en"

    def test_word_count_falls_back_without_tokens(self) -> None:
        assert Sentence.create("one two three").word_count() == 3

    def test_word_count_uses_tokens_when_present(self) -> None:
        sentence = Sentence.create("one two three")

        sentence.set_tokens([Token.create("one"), Token.create("two")])

        assert sentence.word_count() == 2

    def test_round_trip(self) -> None:
        sentence = Sentence.create("Hello", start=3, language="en")

        sentence.add(Token.create("Hello", start=0))

        rebuilt = Sentence.from_dict(sentence.to_dict())

        assert rebuilt.text == sentence.text

        assert rebuilt.span.start == 3

        assert rebuilt.token_count == 1

    def test_is_blank(self) -> None:
        assert Sentence.create("   ").is_blank

        assert not Sentence.create("x").is_blank


class TestParagraph:
    def test_from_text_segments(self) -> None:
        paragraph = Paragraph.from_text("One. Two.", language="en")

        assert paragraph.sentence_count == 2

    def test_segment_false_keeps_one_sentence(self) -> None:
        paragraph = Paragraph.from_text("One. Two.", segment=False)

        assert paragraph.sentence_count == 1

    def test_child_spans_are_relative_to_paragraph(self) -> None:
        paragraph = Paragraph.from_text("One. Two.", language="en")

        for sentence in paragraph.sentences:
            assert sentence.text == paragraph.text[sentence.span.start : sentence.span.end]

    def test_round_trip(self) -> None:
        paragraph = Paragraph.from_text("One. Two.", language="en", index=3)

        rebuilt = Paragraph.from_dict(paragraph.to_dict())

        assert rebuilt.sentence_count == 2

        assert rebuilt.metadata.paragraph_index == 3


class TestDocument:
    def test_structure(self, english_document: Document) -> None:
        assert english_document.paragraph_count == 2

        assert english_document.sentence_count == 3

    def test_language_is_inferred_from_script(self) -> None:
        document = Document.from_text("नमस्ते दुनिया।")

        assert document.metadata.base.language == "hi"

    def test_ambiguous_script_infers_nothing(self) -> None:
        """Latin is shared by many languages, so no guess is made."""

        assert Document.from_text("hello world").metadata.base.language is None

    def test_explicit_language_wins(self) -> None:
        assert Document.from_text("नमस्ते", language="mr").metadata.base.language == "mr"

    def test_verify_passes_for_constructed_document(self, english_document: Document) -> None:
        english_document.verify()

    def test_verify_detects_inconsistent_span(self, english_document: Document) -> None:
        english_document.paragraphs[0].sentences[0].text = "tampered"

        with pytest.raises(CorpusError):
            english_document.verify()

    def test_verify_detects_out_of_bounds_child(self) -> None:
        document = Document.from_text("short", segment=False)

        document.add(Paragraph.from_text("way beyond the end", start=100, segment=False))

        with pytest.raises(CorpusError):
            document.verify()

    def test_round_trip_preserves_everything(self, english_document: Document) -> None:
        english_document.metadata.license = "CC-BY-4.0"

        rebuilt = Document.from_dict(english_document.to_dict())

        assert rebuilt.to_dict() == english_document.to_dict()

        assert rebuilt.metadata.license == "CC-BY-4.0"

        rebuilt.verify()

    def test_sentences_and_tokens_iterate_in_order(self, english_document: Document) -> None:
        texts = [sentence.text for sentence in english_document.sentences()]

        assert texts[0].startswith("Dr. Smith")

        assert len(texts) == 3

    def test_multilingual_documents(self, multilingual_texts: dict[str, str]) -> None:
        for language, text in multilingual_texts.items():
            document = Document.from_text(text, language=language)

            assert document.sentence_count >= 1

            document.verify()
