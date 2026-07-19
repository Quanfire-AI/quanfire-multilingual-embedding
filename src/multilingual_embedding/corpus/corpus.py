"""
Corpus aggregate.

A :class:`Corpus` is an in-memory collection of documents plus the
metadata describing the dataset as a whole.

Not every workload should build one. Tokenizer training and embedding
training both stream sentences and never need the full corpus resident,
which is why the readers in
:mod:`multilingual_embedding.corpus.reader` yield documents lazily.
Build a Corpus when you need random access, splitting or repeated
passes; stream otherwise.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from multilingual_embedding.core.logging import get_logger
from multilingual_embedding.corpus.metadata.corpus import CorpusMetadata
from multilingual_embedding.utils.io import read_jsonl, write_jsonl

from .document import Document
from .exceptions import EmptyCorpusError
from .sentence import Sentence

__all__ = ["Corpus"]

_logger = get_logger(__name__)


@dataclass(slots=True)
class Corpus:
    """
    A collection of documents.

    Attributes
    ----------
    documents:
        Documents in insertion order.

    metadata:
        Dataset level metadata.
    """

    documents: list[Document] = field(default_factory=list)

    metadata: CorpusMetadata = field(default_factory=CorpusMetadata)

    def __iter__(self) -> Iterator[Document]:
        return iter(self.documents)

    def __len__(self) -> int:
        return len(self.documents)

    def __getitem__(self, index: int) -> Document:
        return self.documents[index]

    def __repr__(self) -> str:
        return (
            f"Corpus(documents={len(self.documents)}, "
            f"sentences={self.sentence_count}, "
            f"name={self.metadata.dataset_name!r})"
        )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_documents(
        cls,
        documents: Iterable[Document],
        *,
        name: str | None = None,
        version: str | None = None,
        description: str | None = None,
    ) -> Corpus:
        """Build a corpus from an iterable of documents."""

        metadata = CorpusMetadata()

        metadata.dataset_name = name

        metadata.version = version

        metadata.description = description

        return cls(documents=list(documents), metadata=metadata)

    @classmethod
    def from_texts(
        cls,
        texts: Iterable[str],
        *,
        language: str | None = None,
        name: str | None = None,
    ) -> Corpus:
        """
        Build a corpus by segmenting each string into a document.

        Convenient for tests and small experiments. Document ids are
        assigned positionally.
        """

        documents = [
            Document.from_text(text, identifier=f"doc-{index}", language=language)
            for index, text in enumerate(texts)
        ]

        return cls.from_documents(documents, name=name)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, document: Document) -> Document:
        """Append a document and return it."""

        self.documents.append(document)

        return document

    def extend(self, documents: Iterable[Document]) -> None:
        """Append several documents."""

        self.documents.extend(documents)

    # ------------------------------------------------------------------
    # Traversal
    # ------------------------------------------------------------------

    def sentences(self) -> Iterator[Sentence]:
        """Iterate every sentence across every document."""

        for document in self.documents:
            yield from document.sentences()

    def sentence_texts(self) -> Iterator[str]:
        """
        Iterate sentence strings.

        This is the input format the tokenizer trainer expects.
        """

        for sentence in self.sentences():
            yield sentence.text

    @property
    def document_count(self) -> int:
        """Number of documents."""

        return len(self.documents)

    @property
    def sentence_count(self) -> int:
        """Number of sentences across all documents."""

        return sum(document.sentence_count for document in self.documents)

    @property
    def character_count(self) -> int:
        """Total characters across all documents."""

        return sum(document.character_count for document in self.documents)

    @property
    def languages(self) -> list[str]:
        """Sorted list of distinct declared languages."""

        found = {
            document.metadata.base.language
            for document in self.documents
            if document.metadata.base.language
        }

        return sorted(found)

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def filter(self, predicate: Callable[[Document], bool]) -> Corpus:
        """Return a new corpus holding documents matching ``predicate``."""

        return Corpus(
            documents=[document for document in self.documents if predicate(document)],
            metadata=self.metadata,
        )

    def by_language(self, language: str) -> Corpus:
        """Return a new corpus holding only documents in ``language``."""

        return self.filter(lambda document: document.metadata.base.language == language)

    def split(
        self,
        *,
        train_fraction: float = 0.9,
        seed: int = 42,
    ) -> tuple[Corpus, Corpus]:
        """
        Split into train and evaluation corpora at the document level.

        Splitting by document rather than by sentence is what keeps the
        evaluation set honest: sentences from one document are highly
        correlated, so dividing them across the boundary would let a
        model see near-duplicates of what it is scored on.

        Parameters
        ----------
        train_fraction:
            Share of documents assigned to the training split.

        seed:
            Seed for the shuffle, so a split is reproducible.

        Raises
        ------
        EmptyCorpusError
            If the corpus holds no documents.
        """

        if not self.documents:
            raise EmptyCorpusError("Cannot split an empty corpus")

        if not 0.0 < train_fraction < 1.0:
            raise ValueError("train_fraction must lie in (0, 1)")

        shuffled = list(self.documents)

        random.Random(seed).shuffle(shuffled)

        # At least one document on each side, so neither split is empty
        # for a corpus of two or more documents.
        pivot = max(1, min(len(shuffled) - 1, round(len(shuffled) * train_fraction)))

        train = Corpus(documents=shuffled[:pivot], metadata=self.metadata)

        evaluation = Corpus(documents=shuffled[pivot:], metadata=self.metadata)

        _logger.debug(
            "Split corpus",
            extra={"train": len(train), "evaluation": len(evaluation)},
        )

        return train, evaluation

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        """
        Write the corpus as JSON Lines, one document per line.

        Dataset metadata is written as a leading header record so that a
        corpus file is self describing.
        """

        target = Path(path)

        def records() -> Iterator[dict[str, Any]]:
            yield {
                "_header": True,
                "dataset_name": self.metadata.dataset_name,
                "version": self.metadata.version,
                "description": self.metadata.description,
            }

            for document in self.documents:
                yield document.to_dict()

        count = write_jsonl(target, records())

        _logger.info(
            "Saved corpus",
            extra={"path": str(target), "records": count},
        )

        return target

    @classmethod
    def load(cls, path: str | Path) -> Corpus:
        """Read a corpus written by :meth:`save`."""

        corpus = cls()

        for record in read_jsonl(path):
            if record.get("_header"):
                corpus.metadata.dataset_name = record.get("dataset_name")

                corpus.metadata.version = record.get("version")

                corpus.metadata.description = record.get("description")

                continue

            corpus.add(Document.from_dict(record))

        return corpus

    def verify(self) -> None:
        """Recursively verify span consistency across every document."""

        for document in self.documents:
            document.verify()
