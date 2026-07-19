"""
Shared pytest fixtures.

The multilingual sample texts here are used across the whole suite. They
are deliberately real sentences in each script rather than lorem ipsum,
because several code paths — segmentation, script detection, combining
mark handling — behave differently on genuine text.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from multilingual_embedding.corpus.corpus import Corpus
from multilingual_embedding.corpus.document import Document

ENGLISH_TEXT = "Dr. Smith arrived late. He was tired.\n\nThe next day was better."

HINDI_TEXT = "नमस्ते दुनिया। यह एक परीक्षा है।\n\nआज मौसम अच्छा है।"

JAPANESE_TEXT = "今日はいい天気です。明日も晴れるでしょう。"

ARABIC_TEXT = "مرحبا بالعالم. كيف حالك؟"

CHINESE_TEXT = "今天天气很好。你好吗？"

TAMIL_TEXT = "வணக்கம் உலகம். இது ஒரு சோதனை."


@pytest.fixture
def multilingual_texts() -> dict[str, str]:
    """Sample text keyed by ISO 639-1 language code."""

    return {
        "en": ENGLISH_TEXT,
        "hi": HINDI_TEXT,
        "ja": JAPANESE_TEXT,
        "ar": ARABIC_TEXT,
        "zh": CHINESE_TEXT,
        "ta": TAMIL_TEXT,
    }


@pytest.fixture
def english_document() -> Document:
    """A segmented English document."""

    return Document.from_text(ENGLISH_TEXT, identifier="en-1", language="en")


@pytest.fixture
def hindi_document() -> Document:
    """A segmented Hindi document."""

    return Document.from_text(HINDI_TEXT, identifier="hi-1", language="hi")


@pytest.fixture
def small_corpus(english_document: Document, hindi_document: Document) -> Corpus:
    """A two document, two language corpus."""

    return Corpus.from_documents([english_document, hindi_document], name="test-corpus")


@pytest.fixture
def text_corpus_directory(tmp_path: Path) -> Path:
    """A directory holding two plain text corpus files."""

    directory = tmp_path / "text_corpus"

    directory.mkdir()

    (directory / "english.txt").write_text(ENGLISH_TEXT, encoding="utf-8")

    (directory / "hindi.txt").write_text(HINDI_TEXT, encoding="utf-8")

    return directory


@pytest.fixture
def jsonl_corpus_file(tmp_path: Path) -> Path:
    """A JSON Lines corpus file with language and extra fields."""

    path = tmp_path / "corpus.jsonl"

    records = [
        {"id": "d1", "text": ENGLISH_TEXT, "language": "en", "topic": "weather"},
        {"id": "d2", "text": HINDI_TEXT, "language": "hi"},
        {"id": "d3", "text": TAMIL_TEXT, "language": "ta"},
    ]

    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records),
        encoding="utf-8",
    )

    return path


@pytest.fixture
def training_sentences() -> list[str]:
    """
    A synthetic corpus with two disjoint topics.

    Used to check that an embedding model learns structure: tokens from
    the same topic should end up closer than tokens from different ones.
    """

    weather = ["sun", "rain", "cloud", "storm", "wind"]

    finance = ["bank", "loan", "credit", "market", "stock"]

    sentences: list[str] = []

    for _ in range(60):
        sentences.append(" ".join(weather))

        sentences.append(" ".join(reversed(weather)))

        sentences.append(" ".join(finance))

        sentences.append(" ".join(reversed(finance)))

    return sentences
