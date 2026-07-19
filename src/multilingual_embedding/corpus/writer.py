"""
Corpus writers.

Two output shapes matter downstream:

:class:`JsonlCorpusWriter`
    Full fidelity. Preserves the document tree, metadata and spans, so a
    corpus survives a round trip unchanged.

:class:`PlainTextCorpusWriter`
    One sentence per line. This is what SentencePiece consumes for
    training, and it is the format in which intermediate training files
    are staged.

Both stream their input and write atomically, so an interrupted export
leaves no partial file behind.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from multilingual_embedding.common.constants import DEFAULT_ENCODING
from multilingual_embedding.core.logging import get_logger
from multilingual_embedding.utils.filesystem import atomic_write_path
from multilingual_embedding.utils.io import write_jsonl

from .document import Document

__all__ = [
    "CorpusWriter",
    "JsonlCorpusWriter",
    "PlainTextCorpusWriter",
    "write_sentences",
]

_logger = get_logger(__name__)


class CorpusWriter(ABC):
    """Base class for corpus writers."""

    def __init__(self, path: str | Path, *, encoding: str = DEFAULT_ENCODING) -> None:
        self.path = Path(path).expanduser()

        self.encoding = encoding

    @abstractmethod
    def write(self, documents: Iterable[Document]) -> int:
        """Write documents and return the number of records emitted."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(path={str(self.path)!r})"


class JsonlCorpusWriter(CorpusWriter):
    """
    Write documents as JSON Lines, one document per line.

    Round trips through
    :meth:`~multilingual_embedding.corpus.document.Document.from_dict`
    without loss.
    """

    def write(self, documents: Iterable[Document]) -> int:
        def records() -> Iterator[dict[str, Any]]:
            for document in documents:
                yield document.to_dict()

        count = write_jsonl(self.path, records(), encoding=self.encoding)

        _logger.info(
            "Wrote corpus",
            extra={"path": str(self.path), "documents": count},
        )

        return count


class PlainTextCorpusWriter(CorpusWriter):
    """
    Write one sentence per line.

    Newlines inside a sentence would corrupt the line-per-sentence
    contract, so any embedded newline is collapsed to a space.
    """

    def write(self, documents: Iterable[Document]) -> int:
        def sentences() -> Iterator[str]:
            for document in documents:
                for sentence in document.sentences():
                    yield sentence.text

        return write_sentences(self.path, sentences(), encoding=self.encoding)


def write_sentences(
    path: str | Path,
    sentences: Iterable[str],
    *,
    encoding: str = DEFAULT_ENCODING,
) -> int:
    """
    Write sentences one per line and return the count.

    Blank sentences are skipped, and internal newlines are collapsed, so
    the output always holds exactly one sentence per line.
    """

    target = Path(path).expanduser()

    written = 0

    with (
        atomic_write_path(target) as temporary,
        temporary.open("w", encoding=encoding) as handle,
    ):
        for sentence in sentences:
            flattened = " ".join(sentence.split())

            if not flattened:
                continue

            handle.write(flattened)
            handle.write("\n")

            written += 1

    _logger.info(
        "Wrote sentences",
        extra={"path": str(target), "sentences": written},
    )

    return written
