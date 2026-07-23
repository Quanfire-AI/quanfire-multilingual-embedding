"""
Configuration driven corpus loading.

The rest of the framework asks for text through this module rather than
constructing readers itself. Two entry points cover the two access
patterns:

:func:`load_corpus`
    Materialise everything into memory. For random access, splitting and
    repeated passes.

:func:`stream_sentences`
    A re-iterable stream that never materialises the corpus. This is
    what tokenizer and embedding training use.

Both apply the filtering configured in
:class:`~multilingual_embedding.config.base.CorpusConfig`, so cleaning
rules cannot be accidentally skipped by one caller and applied by
another.

Each has a config-free twin — :func:`documents_from`, :func:`corpus_from`
and :func:`sentences_from` — taking a source and plain settings. Those
are what a repository pinning this one should call: ``CorpusConfig``
belongs to ``config``, which is internal and free to change in a patch
release, so a public function that can only be reached through it is a
promise with a hole in it.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from multilingual_embedding.config.base import CorpusConfig
from multilingual_embedding.core.exceptions import ConfigurationError
from multilingual_embedding.core.logging import get_logger

from .corpus import Corpus
from .document import Document
from .iterator import SentenceStream
from .reader import CorpusReader, reader_for
from .validators import DocumentDeduplicator, SentenceFilter

__all__ = [
    "build_filter",
    "build_reader",
    "corpus_from",
    "documents_from",
    "load_corpus",
    "sentences_from",
    "stream_documents",
    "stream_sentences",
]

_logger = get_logger(__name__)


def build_reader(config: CorpusConfig) -> CorpusReader:
    """
    Construct the reader described by a corpus configuration.

    Raises
    ------
    ConfigurationError
        If no source is configured.
    """

    if config.source is None:
        raise ConfigurationError("Corpus configuration has no source")

    return reader_for(
        config.source,
        format=config.format,
        language=config.language,
        encoding=config.encoding,
        patterns=list(config.patterns),
        text_field=config.text_field,
    )


def build_filter(config: CorpusConfig) -> SentenceFilter:
    """Construct the sentence filter described by a corpus configuration."""

    return SentenceFilter(
        min_characters=config.min_sentence_characters,
        max_characters=config.max_sentence_characters,
    )


def stream_documents(
    config: CorpusConfig,
    *,
    deduplicate: bool = True,
) -> Iterator[Document]:
    """
    Yield filtered documents from the configured source, lazily.

    Parameters
    ----------
    deduplicate:
        Drop documents whose text was already seen. Memory grows by one
        hash per distinct document.
    """

    reader = build_reader(config)

    sentence_filter = build_filter(config)

    deduplicator = DocumentDeduplicator() if deduplicate else None

    for document in reader.iter_documents():
        if deduplicator is not None and deduplicator.is_duplicate(document):
            continue

        sentence_filter.apply(document)

        if config.lowercase:
            _lowercase_document(document)

        if document.sentence_count == 0:
            continue

        yield document


def load_corpus(
    config: CorpusConfig,
    *,
    deduplicate: bool = True,
    name: str | None = None,
) -> Corpus:
    """
    Load the configured source fully into memory.

    Prefer :func:`stream_sentences` for training; use this when random
    access or splitting is required.
    """

    corpus = Corpus.from_documents(
        stream_documents(config, deduplicate=deduplicate),
        name=name,
    )

    _logger.info(
        "Loaded corpus",
        extra={
            "documents": corpus.document_count,
            "sentences": corpus.sentence_count,
            "languages": corpus.languages,
        },
    )

    return corpus


def stream_sentences(
    config: CorpusConfig,
    *,
    deduplicate: bool = True,
    limit: int | None = None,
) -> SentenceStream:
    """
    Return a re-iterable stream of filtered sentence strings.

    The returned stream restarts the reader on every pass, so it can be
    iterated once per training epoch without holding the corpus in
    memory.

    Parameters
    ----------
    limit:
        Optional cap on sentences per pass, for smoke tests against a
        large corpus.
    """

    def factory() -> Iterator[str]:
        for document in stream_documents(config, deduplicate=deduplicate):
            for sentence in document.sentences():
                yield sentence.text

    return SentenceStream(factory, limit=limit)


def _settings(
    source: str | Path,
    *,
    format: str | None,
    patterns: Sequence[str] | None,
    language: str | None,
    encoding: str | None,
    text_field: str | None,
    min_sentence_characters: int | None,
    max_sentence_characters: int | None,
    lowercase: bool | None,
) -> CorpusConfig:
    """
    Assemble the internal config the three public twins delegate to.

    Private, and deliberately so: it returns the very type those
    functions exist to keep out of a consumer's imports. ``None`` means
    *keep the framework default* rather than restating it here, because
    two copies of a default drift and the copy in a signature drifts
    silently.
    """

    given: dict[str, Any] = {}

    for name, value in (
        ("format", format),
        ("patterns", patterns),
        ("language", language),
        ("encoding", encoding),
        ("text_field", text_field),
        ("min_sentence_characters", min_sentence_characters),
        ("max_sentence_characters", max_sentence_characters),
        ("lowercase", lowercase),
    ):
        if value is None:
            continue

        given[name] = value

    # Copied rather than referenced: the config owns the list and would
    # otherwise share it with the caller's literal.
    if "patterns" in given:
        given["patterns"] = list(given["patterns"])

    return CorpusConfig(source=Path(source), **given)


def documents_from(
    source: str | Path,
    *,
    format: str | None = None,
    patterns: Sequence[str] | None = None,
    language: str | None = None,
    encoding: str | None = None,
    text_field: str | None = None,
    min_sentence_characters: int | None = None,
    max_sentence_characters: int | None = None,
    lowercase: bool | None = None,
    deduplicate: bool = True,
) -> Iterator[Document]:
    """
    :func:`stream_documents`, from a source and plain settings.

    The config-taking form is for callers inside this repository, which
    already hold a ``CorpusConfig`` because a YAML file produced one.
    This is the form for callers outside it.

    Parameters
    ----------
    source:
        File or directory to read.

    format:
        Reader name, or ``"auto"`` to choose by extension.

    patterns:
        Glob patterns used when ``source`` is a directory.

    language:
        Language code applied to documents that do not declare one.

    encoding:
        Character encoding of the source files.

    text_field:
        Record key holding the text, for record-structured formats.

    min_sentence_characters, max_sentence_characters:
        Sentence length bounds, applied as documents stream past.

    lowercase:
        Fold case while loading. Off by default — casing is a tokenizer
        concern, and folding it here destroys it for every consumer of
        the corpus at once.

    deduplicate:
        Drop documents whose text was already seen. Memory grows by one
        hash per distinct document.

    Notes
    -----
    A generator: nothing is read, and nothing about the corpus raises,
    until the caller iterates. A ``try`` around this call translates no
    missing file and no malformed line — wrap the iteration instead.

    Omit any setting to keep the framework default rather than restating
    it at the call site.
    """

    return stream_documents(
        _settings(
            source,
            format=format,
            patterns=patterns,
            language=language,
            encoding=encoding,
            text_field=text_field,
            min_sentence_characters=min_sentence_characters,
            max_sentence_characters=max_sentence_characters,
            lowercase=lowercase,
        ),
        deduplicate=deduplicate,
    )


def corpus_from(
    source: str | Path,
    *,
    format: str | None = None,
    patterns: Sequence[str] | None = None,
    language: str | None = None,
    encoding: str | None = None,
    text_field: str | None = None,
    min_sentence_characters: int | None = None,
    max_sentence_characters: int | None = None,
    lowercase: bool | None = None,
    deduplicate: bool = True,
    name: str | None = None,
) -> Corpus:
    """
    :func:`load_corpus`, from a source and plain settings.

    Materialises everything into memory; prefer :func:`sentences_from`
    for training and use this when random access or splitting is
    required. Arguments are :func:`documents_from`'s, plus ``name`` for
    the resulting corpus.
    """

    return load_corpus(
        _settings(
            source,
            format=format,
            patterns=patterns,
            language=language,
            encoding=encoding,
            text_field=text_field,
            min_sentence_characters=min_sentence_characters,
            max_sentence_characters=max_sentence_characters,
            lowercase=lowercase,
        ),
        deduplicate=deduplicate,
        name=name,
    )


def sentences_from(
    source: str | Path,
    *,
    format: str | None = None,
    patterns: Sequence[str] | None = None,
    language: str | None = None,
    encoding: str | None = None,
    text_field: str | None = None,
    min_sentence_characters: int | None = None,
    max_sentence_characters: int | None = None,
    lowercase: bool | None = None,
    deduplicate: bool = True,
    limit: int | None = None,
) -> SentenceStream:
    """
    :func:`stream_sentences`, from a source and plain settings.

    Returns a re-iterable stream that restarts the reader on every pass,
    so it can be iterated once per training epoch without holding the
    corpus in memory. Arguments are :func:`documents_from`'s, plus
    ``limit`` as a cap on sentences per pass.
    """

    return stream_sentences(
        _settings(
            source,
            format=format,
            patterns=patterns,
            language=language,
            encoding=encoding,
            text_field=text_field,
            min_sentence_characters=min_sentence_characters,
            max_sentence_characters=max_sentence_characters,
            lowercase=lowercase,
        ),
        deduplicate=deduplicate,
        limit=limit,
    )


def _lowercase_document(document: Document) -> None:
    """
    Lowercase a document's text and every node beneath it.

    Applied consistently at all levels so that spans stay valid; case
    folding can change string length in some scripts, so the node texts
    and the parent text must be folded with the same operation.
    """

    document.text = document.text.lower()

    for paragraph in document.paragraphs:
        paragraph.text = paragraph.text.lower()

        for sentence in paragraph.sentences:
            sentence.text = sentence.text.lower()

            for token in sentence.tokens:
                token.text = token.text.lower()
