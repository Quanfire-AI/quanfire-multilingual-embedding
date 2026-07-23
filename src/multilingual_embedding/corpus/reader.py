"""
Corpus readers.

Readers turn files on disk into :class:`Document` objects. Every reader
is lazy: :meth:`CorpusReader.iter_documents` is a generator, so a corpus
larger than memory streams through the pipeline one document at a time.

Readers are registered by name so a configuration file can select one
without importing it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from pathlib import Path

from multilingual_embedding.common.constants import DEFAULT_ENCODING
from multilingual_embedding.core.logging import get_logger
from multilingual_embedding.core.registry import Registry
from multilingual_embedding.utils.filesystem import iter_files, require_file
from multilingual_embedding.utils.hashing import hash_text
from multilingual_embedding.utils.io import open_text, read_jsonl

from .document import Document
from .exceptions import CorpusFormatError
from .paragraph import Paragraph
from .sentence import Sentence

__all__ = [
    "READERS",
    "CorpusReader",
    "JsonlReader",
    "LineReader",
    "TextFileReader",
    "reader_for",
    "resolve_reader_type",
]

_logger = get_logger(__name__)

READERS: Registry[CorpusReader] = Registry("corpus_reader")


class CorpusReader(ABC):
    """
    Base class for corpus readers.

    Parameters
    ----------
    source:
        File or directory to read.

    language:
        Language code applied to documents that do not declare one.

    encoding:
        Character encoding of the source files.

    patterns:
        Glob patterns used when ``source`` is a directory.

    Notes
    -----
    :meth:`iter_documents` is a generator, so constructing a reader
    raises nothing about the corpus — not a missing file, not a malformed
    line. Its body does not run until the caller iterates, by which time
    a ``try`` around the construction has already exited. Wrap the
    iteration to translate corpus errors::

        reader = reader_for(path)             # raises nothing yet

        try:
            for document in reader.iter_documents():
                ...
        except MultilingualEmbeddingError as error:
            ...

    The one error that does surface at construction is an explicit
    ``format`` naming no registered reader, which
    :func:`reader_for` raises as a ``RegistryError``. Under
    ``format="auto"`` even that is silent: an unrecognised extension
    falls back to :class:`TextFileReader`.
    """

    def __init__(
        self,
        source: str | Path,
        *,
        language: str | None = None,
        encoding: str = DEFAULT_ENCODING,
        patterns: Sequence[str] | None = None,
    ) -> None:
        self.source = Path(source).expanduser()

        self.language = language

        self.encoding = encoding

        self.patterns = tuple(patterns) if patterns else self.default_patterns()

    @staticmethod
    def default_patterns() -> tuple[str, ...]:
        """Glob patterns this reader matches when given a directory."""

        return ("*",)

    @abstractmethod
    def read_file(self, path: Path) -> Iterator[Document]:
        """Yield the documents contained in one file."""

    def iter_documents(self) -> Iterator[Document]:
        """
        Yield every document from the source, lazily.

        Files are visited in sorted order so that two runs over the same
        directory produce the same document sequence.
        """

        for path in self.paths():
            _logger.debug("Reading corpus file", extra={"path": str(path)})

            yield from self.read_file(path)

    def iter_sentences(self) -> Iterator[Sentence]:
        """Yield every sentence from the source, lazily."""

        for document in self.iter_documents():
            yield from document.sentences()

    def iter_sentence_texts(self) -> Iterator[str]:
        """Yield every sentence as a string, lazily."""

        for sentence in self.iter_sentences():
            yield sentence.text

    def paths(self) -> list[Path]:
        """Resolve the source into a sorted list of files."""

        if self.source.is_dir():
            return list(iter_files(self.source, patterns=self.patterns))

        return [require_file(self.source)]

    def __repr__(self) -> str:
        return f"{type(self).__name__}(source={str(self.source)!r})"


@READERS.register("text")
class TextFileReader(CorpusReader):
    """
    One plain text file becomes one document.

    Paragraphs are recovered from blank lines and sentences from the
    segmenter, so a file with normal prose layout yields a full
    document/paragraph/sentence tree.
    """

    @staticmethod
    def default_patterns() -> tuple[str, ...]:
        return ("*.txt", "*.text", "*.txt.gz")

    def read_file(self, path: Path) -> Iterator[Document]:
        with open_text(path, "r", encoding=self.encoding) as handle:
            text = handle.read()

        if not text.strip():
            return

        yield Document.from_text(
            text,
            identifier=hash_text(text),
            language=self.language,
            source=str(path),
            title=path.stem,
        )


@READERS.register("lines")
class LineReader(CorpusReader):
    """
    Each non-blank line becomes its own single-sentence document.

    This is the right reader for corpora already segmented one sentence
    per line, which is how most public training sets are distributed.
    Segmentation is skipped entirely so the framework does not
    second-guess the source's own boundaries.
    """

    @staticmethod
    def default_patterns() -> tuple[str, ...]:
        return ("*.txt", "*.text", "*.txt.gz")

    def read_file(self, path: Path) -> Iterator[Document]:
        with open_text(path, "r", encoding=self.encoding) as handle:
            for line_number, line in enumerate(handle, start=1):
                text = line.strip()

                if not text:
                    continue

                document = Document.from_text(
                    text,
                    identifier=f"{path.stem}:{line_number}",
                    language=self.language,
                    source=str(path),
                    segment=False,
                )

                document.add(
                    _single_sentence_paragraph(text, language=document.metadata.base.language)
                )

                yield document


@READERS.register("jsonl")
class JsonlReader(CorpusReader):
    """
    Each JSON Lines record becomes one document.

    Parameters
    ----------
    text_field:
        Record key holding the document text.

    id_field, language_field:
        Optional keys carrying an identifier and a language code. When
        absent, the id is derived from the text hash and the language
        falls back to the reader's default.

    segment:
        Whether to segment the text. Set False when each record already
        holds exactly one sentence.
    """

    def __init__(
        self,
        source: str | Path,
        *,
        language: str | None = None,
        encoding: str = DEFAULT_ENCODING,
        patterns: Sequence[str] | None = None,
        text_field: str = "text",
        id_field: str = "id",
        language_field: str = "language",
        segment: bool = True,
    ) -> None:
        super().__init__(
            source,
            language=language,
            encoding=encoding,
            patterns=patterns,
        )

        self.text_field = text_field

        self.id_field = id_field

        self.language_field = language_field

        self.segment = segment

    @staticmethod
    def default_patterns() -> tuple[str, ...]:
        return ("*.jsonl", "*.jsonl.gz", "*.ndjson")

    def read_file(self, path: Path) -> Iterator[Document]:
        for index, record in enumerate(read_jsonl(path), start=1):
            if not isinstance(record, dict):
                raise CorpusFormatError(
                    "JSON Lines record must be an object",
                    path=str(path),
                    line=index,
                    received=type(record).__name__,
                )

            text = record.get(self.text_field)

            if text is None:
                raise CorpusFormatError(
                    "JSON Lines record is missing the text field",
                    path=str(path),
                    line=index,
                    text_field=self.text_field,
                    keys=sorted(record),
                )

            if not isinstance(text, str):
                raise CorpusFormatError(
                    "JSON Lines text field must be a string",
                    path=str(path),
                    line=index,
                    received=type(text).__name__,
                )

            if not text.strip():
                continue

            document = Document.from_text(
                text,
                identifier=str(record.get(self.id_field) or hash_text(text)),
                language=record.get(self.language_field) or self.language,
                source=str(path),
                title=record.get("title"),
                segment=self.segment,
            )

            if not self.segment:
                document.add(
                    _single_sentence_paragraph(text, language=document.metadata.base.language)
                )

            # Carry through any additional fields, so downstream stages
            # can use dataset-specific columns the framework knows nothing about.
            known = {self.text_field, self.id_field, self.language_field, "title"}

            document.metadata.base.attributes = {
                key: value for key, value in record.items() if key not in known
            }

            yield document


def reader_for(
    source: str | Path,
    *,
    format: str = "auto",
    text_field: str | None = None,
    **options: object,
) -> CorpusReader:
    """
    Construct the appropriate reader for a source.

    Parameters
    ----------
    source:
        File or directory.

    format:
        Reader name, or ``"auto"`` to choose by file extension. For a
        directory, ``"auto"`` inspects the first matching file.

    text_field:
        Record key holding the text, for record-structured formats.
        Forwarded only to readers that have records; a plain text file
        has no fields to name, so the argument is inapplicable rather
        than ignored there. Omit to keep the reader's own default.

    options:
        Passed through to the reader constructor.
    """

    resolved = Path(source).expanduser()

    reader_type = resolve_reader_type(resolved, format=format)

    # Resolving the type before constructing is what lets record-only
    # options be routed correctly under ``format="auto"``, where the
    # caller cannot know which reader will be chosen.
    if text_field is not None and issubclass(reader_type, JsonlReader):
        options["text_field"] = text_field

    return reader_type(resolved, **options)  # type: ignore[arg-type]


def resolve_reader_type(source: str | Path, *, format: str = "auto") -> type[CorpusReader]:
    """
    Return the reader class that :func:`reader_for` would construct.

    Parameters
    ----------
    source:
        File or directory.

    format:
        Reader name, or ``"auto"`` to choose by file extension.

    Returns
    -------
    The reader class, uninstantiated.

    Raises
    ------
    RegistryError
        If ``format`` names no registered reader.
    """

    resolved = Path(source).expanduser()

    if format != "auto":
        return READERS.get(format)

    suffixes = _meaningful_suffixes(resolved)

    if not suffixes and resolved.is_dir():
        for candidate in iter_files(resolved, patterns=("*.jsonl", "*.ndjson", "*.txt")):
            suffixes = _meaningful_suffixes(candidate)

            break

    if any(suffix in {".jsonl", ".ndjson"} for suffix in suffixes):
        return JsonlReader

    return TextFileReader


def _meaningful_suffixes(path: Path) -> list[str]:
    """Return suffixes with a trailing ``.gz`` removed."""

    suffixes = [suffix.lower() for suffix in path.suffixes]

    return [suffix for suffix in suffixes if suffix != ".gz"]


def _single_sentence_paragraph(text: str, *, language: str | None) -> Paragraph:
    """
    Wrap pre-segmented text as a paragraph holding exactly one sentence.

    Used by readers whose input is already one sentence per record, so
    that those records still produce a well formed document tree.
    """

    return Paragraph.from_text(
        text,
        start=0,
        language=language,
        index=0,
        segment=False,
    )
