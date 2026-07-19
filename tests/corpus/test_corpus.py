from __future__ import annotations

from pathlib import Path

import pytest

from multilingual_embedding.corpus.corpus import Corpus
from multilingual_embedding.corpus.document import Document
from multilingual_embedding.corpus.exceptions import EmptyCorpusError


def test_counts(small_corpus: Corpus) -> None:
    assert small_corpus.document_count == 2

    assert small_corpus.sentence_count == 6

    assert small_corpus.character_count > 0


def test_languages(small_corpus: Corpus) -> None:
    assert small_corpus.languages == ["en", "hi"]


def test_by_language(small_corpus: Corpus) -> None:
    assert small_corpus.by_language("hi").document_count == 1


def test_filter(small_corpus: Corpus) -> None:
    filtered = small_corpus.filter(lambda document: document.sentence_count > 100)

    assert filtered.document_count == 0


def test_from_texts_assigns_ids() -> None:
    corpus = Corpus.from_texts(["One. Two.", "Three."], language="en")

    assert [document.identifier for document in corpus] == ["doc-0", "doc-1"]


def test_iteration_and_indexing(small_corpus: Corpus) -> None:
    assert len(list(small_corpus)) == 2

    assert small_corpus[0] is small_corpus.documents[0]


def test_sentence_texts_are_strings(small_corpus: Corpus) -> None:
    assert all(isinstance(text, str) for text in small_corpus.sentence_texts())


class TestSplit:
    def test_split_is_document_level_and_exhaustive(self) -> None:
        corpus = Corpus.from_texts([f"Document {index}." for index in range(10)])

        train, evaluation = corpus.split(train_fraction=0.8, seed=1)

        assert train.document_count + evaluation.document_count == 10

        assert train.document_count == 8

    def test_split_is_reproducible(self) -> None:
        corpus = Corpus.from_texts([f"Document {index}." for index in range(10)])

        first, _ = corpus.split(seed=7)

        second, _ = corpus.split(seed=7)

        assert [d.identifier for d in first] == [d.identifier for d in second]

    def test_split_differs_by_seed(self) -> None:
        corpus = Corpus.from_texts([f"Document {index}." for index in range(20)])

        first, _ = corpus.split(seed=1)

        second, _ = corpus.split(seed=2)

        assert [d.identifier for d in first] != [d.identifier for d in second]

    def test_neither_side_is_empty(self) -> None:
        """A tiny corpus must still yield a usable evaluation split."""

        corpus = Corpus.from_texts(["One.", "Two."])

        train, evaluation = corpus.split(train_fraction=0.99)

        assert train.document_count == 1

        assert evaluation.document_count == 1

    def test_empty_corpus_rejected(self) -> None:
        with pytest.raises(EmptyCorpusError):
            Corpus().split()

    @pytest.mark.parametrize("fraction", [0.0, 1.0, -0.5, 1.5])
    def test_invalid_fraction_rejected(self, fraction: float) -> None:
        corpus = Corpus.from_texts(["One.", "Two."])

        with pytest.raises(ValueError):
            corpus.split(train_fraction=fraction)


class TestPersistence:
    def test_round_trip(self, small_corpus: Corpus, tmp_path: Path) -> None:
        path = tmp_path / "corpus.jsonl"

        small_corpus.metadata.dataset_name = "test-corpus"

        small_corpus.metadata.version = "1.0"

        small_corpus.save(path)

        reloaded = Corpus.load(path)

        assert reloaded.document_count == small_corpus.document_count

        assert reloaded.sentence_count == small_corpus.sentence_count

        assert reloaded.metadata.dataset_name == "test-corpus"

        assert reloaded.metadata.version == "1.0"

    def test_reloaded_documents_verify(self, small_corpus: Corpus, tmp_path: Path) -> None:
        path = tmp_path / "corpus.jsonl"

        small_corpus.save(path)

        Corpus.load(path).verify()

    def test_round_trip_preserves_unicode(self, tmp_path: Path) -> None:
        corpus = Corpus.from_documents([Document.from_text("नमस्ते दुनिया।", language="hi")])

        path = tmp_path / "corpus.jsonl"

        corpus.save(path)

        assert "नमस्ते" in Corpus.load(path)[0].text
