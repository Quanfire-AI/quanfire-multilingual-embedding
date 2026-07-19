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
"""

from __future__ import annotations

from collections.abc import Iterator

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
    "load_corpus",
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
