"""
Corpus layer: text representation, segmentation, reading and statistics.

The hierarchy is::

    Corpus
     └── Document
          └── Paragraph
               └── Sentence
                    └── Token

Each node stores its span relative to its parent, so any unit can be
mapped back to its exact position in the source text.

Load text with :func:`reader_for`, :func:`sentences_from`,
:func:`documents_from` or :func:`corpus_from`, all of which take a source
and plain settings. The ``build_*``/``stream_*``/``load_corpus`` forms
take a ``CorpusConfig``, and ``config`` is internal — so calling them
means importing a type this package does not promise to keep.
"""

from __future__ import annotations

from .audit import CorpusAudit, Finding, Severity, audit_corpus
from .corpus import Corpus
from .document import Document
from .exceptions import (
    CorpusError,
    CorpusFormatError,
    EmptyCorpusError,
    SegmentationError,
)
from .iterator import SentenceStream, batched, take
from .language import (
    LANGUAGE_NAMES,
    Language,
    expected_script,
    infer_language,
    language_name,
    normalize_language_code,
)
from .loader import (
    build_filter,
    build_reader,
    corpus_from,
    documents_from,
    load_corpus,
    sentences_from,
    stream_documents,
    stream_sentences,
)
from .paragraph import Paragraph
from .reader import (
    READERS,
    CorpusReader,
    JsonlReader,
    LineReader,
    TextFileReader,
    reader_for,
    resolve_reader_type,
)
from .script import (
    Script,
    ScriptProfile,
    detect_script,
    is_whitespace_delimited,
    script_histogram,
)
from .segmentation import (
    SENTENCE_TERMINATORS,
    split_paragraphs,
    split_sentences,
    split_words,
)
from .sentence import Sentence
from .statistics import (
    CorpusStatistics,
    LengthSummary,
    StatisticsAccumulator,
    compute_statistics,
)
from .token import Token
from .validators import (
    DocumentDeduplicator,
    FilterReport,
    SentenceFilter,
    validate_document,
)
from .writer import (
    CorpusWriter,
    JsonlCorpusWriter,
    PlainTextCorpusWriter,
    write_sentences,
)

__all__ = [
    "LANGUAGE_NAMES",
    "READERS",
    "SENTENCE_TERMINATORS",
    "Corpus",
    "CorpusAudit",
    "CorpusError",
    "CorpusFormatError",
    "CorpusReader",
    "CorpusStatistics",
    "CorpusWriter",
    "Document",
    "DocumentDeduplicator",
    "EmptyCorpusError",
    "FilterReport",
    "Finding",
    "JsonlCorpusWriter",
    "JsonlReader",
    "Language",
    "LengthSummary",
    "LineReader",
    "Paragraph",
    "PlainTextCorpusWriter",
    "Script",
    "ScriptProfile",
    "SegmentationError",
    "Sentence",
    "SentenceFilter",
    "SentenceStream",
    "Severity",
    "StatisticsAccumulator",
    "TextFileReader",
    "Token",
    "audit_corpus",
    "batched",
    "build_filter",
    "build_reader",
    "compute_statistics",
    "corpus_from",
    "detect_script",
    "documents_from",
    "expected_script",
    "infer_language",
    "is_whitespace_delimited",
    "language_name",
    "load_corpus",
    "normalize_language_code",
    "reader_for",
    "resolve_reader_type",
    "script_histogram",
    "sentences_from",
    "split_paragraphs",
    "split_sentences",
    "split_words",
    "stream_documents",
    "stream_sentences",
    "take",
    "validate_document",
    "write_sentences",
]
