"""
Brute-force semantic search over encoded items.

Search is exact: every query is scored against every item with a single
matmul over an L2-normalised matrix, so cosine similarity reduces to a
dot product and the result is the true nearest neighbour set, not an
approximation.

That costs O(n·d) per query. On modern BLAS this is the right choice up
to roughly 10^5-10^6 items, where a query still lands in single-digit
milliseconds and the index needs no build step, no tuning, and no recall
measurement. Past that an approximate index (HNSW, IVF-PQ) becomes
necessary, and it is deliberately out of scope here: an ANN index is a
serious piece of software and wrapping a poor one would be worse than
being honest about the ceiling.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from multilingual_embedding.core.exceptions import (
    ResourceNotFoundError,
    ValidationError,
)
from multilingual_embedding.core.logging import get_logger
from multilingual_embedding.utils.filesystem import (
    atomic_write_path,
    ensure_directory,
    require_directory,
)
from multilingual_embedding.utils.io import read_json, write_json

from .sentence import SentenceEncoder

__all__ = ["SearchResult", "SimilarityIndex"]

_logger = get_logger(__name__)

_FORMAT_VERSION = 1

_VECTORS_FILENAME = "index_vectors.npy"

_METADATA_FILENAME = "index.json"

_NORM_EPSILON = 1e-12


@dataclass(slots=True, frozen=True)
class SearchResult:
    """
    One hit from a similarity search.

    Attributes
    ----------
    label:
        The payload stored alongside the vector, typically the source
        text.

    score:
        Cosine similarity to the query, in ``[-1, 1]``.

    index:
        Position of the item in the index, for joining back to a caller's
        own records.
    """

    label: str

    score: float

    index: int


class SimilarityIndex:
    """
    Exact cosine search over a set of labelled vectors.

    Parameters
    ----------
    dimension:
        Vector length every item must match.

    encoder:
        Optional sentence encoder, required only for text queries.

    Example
    -------
    ::

        index = SimilarityIndex.build(zip(labels, vectors), encoder=encoder)

        for hit in index.search("a wide river", top_k=3):
            print(hit.label, hit.score)
    """

    __slots__ = ("_dimension", "_encoder", "_labels", "_vectors")

    def __init__(
        self,
        dimension: int,
        encoder: SentenceEncoder | None = None,
    ) -> None:
        if dimension < 1:
            raise ValidationError("dimension must be >= 1", name="dimension", value=dimension)

        self._dimension = dimension

        self._encoder = encoder

        self._labels: list[str] = []

        # Rows are stored already normalised, so a query costs one matmul
        # rather than a matmul plus a per-row division.
        self._vectors = np.zeros((0, dimension), dtype=np.float32)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        items: Iterable[tuple[str, np.ndarray]],
        encoder: SentenceEncoder | None = None,
        dimension: int | None = None,
    ) -> SimilarityIndex:
        """
        Create an index from ``(label, vector)`` pairs.

        Parameters
        ----------
        items:
            Labelled vectors, all of the same length.

        encoder:
            Optional encoder enabling text queries.

        dimension:
            Vector length. Inferred from the first item when omitted.

        Raises
        ------
        ValidationError
            If no items are supplied and no dimension is given.
        """

        materialised = [
            (str(label), np.asarray(vector, dtype=np.float32)) for label, vector in items
        ]

        if dimension is None:
            if not materialised:
                raise ValidationError("Cannot infer dimension from an empty item sequence")

            dimension = int(materialised[0][1].reshape(-1).shape[0])

        index = cls(dimension, encoder=encoder)

        index.add(materialised)

        return index

    @classmethod
    def from_texts(
        cls,
        texts: Sequence[str],
        encoder: SentenceEncoder,
    ) -> SimilarityIndex:
        """Encode ``texts`` with ``encoder`` and index them under their own text."""

        vectors = encoder.encode_batch(texts)

        index = cls(encoder.dimension, encoder=encoder)

        index.add(zip(texts, vectors, strict=True))

        return index

    def add(self, items: Iterable[tuple[str, np.ndarray]]) -> int:
        """
        Append labelled vectors to the index.

        Returns
        -------
        int
            The new item count.

        Raises
        ------
        ValidationError
            If any vector has the wrong length.
        """

        labels: list[str] = []

        rows: list[np.ndarray] = []

        for label, vector in items:
            row = np.asarray(vector, dtype=np.float32).reshape(-1)

            if row.shape[0] != self._dimension:
                raise ValidationError(
                    "Indexed vector has the wrong dimension",
                    label=str(label),
                    found=int(row.shape[0]),
                    expected=self._dimension,
                )

            labels.append(str(label))

            rows.append(row)

        if rows:
            stacked = _normalize_rows(np.vstack(rows))

            self._vectors = np.vstack([self._vectors, stacked])

            self._labels.extend(labels)

        return len(self._labels)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def search(self, query: str | np.ndarray, top_k: int = 10) -> list[SearchResult]:
        """
        Return the ``top_k`` most similar items.

        Parameters
        ----------
        query:
            A query vector, or text to be encoded. Text requires the
            index to have been given an encoder.

        top_k:
            Maximum number of results. Fewer are returned when the index
            holds fewer items.

        Returns
        -------
        list[SearchResult]
            Descending by score.

        Raises
        ------
        ValidationError
            If ``top_k`` is not positive, the vector has the wrong
            length, or text is queried without an encoder.
        """

        if top_k < 1:
            raise ValidationError("top_k must be >= 1", name="top_k", value=top_k)

        if not self._labels:
            return []

        vector = self._resolve_query(query)

        scores = self._vectors @ vector

        count = min(top_k, len(self._labels))

        candidates = np.argpartition(-scores, count - 1)[:count]

        ordered = candidates[np.argsort(-scores[candidates], kind="stable")]

        return [
            SearchResult(
                label=self._labels[int(position)],
                score=float(scores[position]),
                index=int(position),
            )
            for position in ordered
        ]

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def dimension(self) -> int:
        """Vector length of every indexed item."""

        return self._dimension

    @property
    def labels(self) -> list[str]:
        """Indexed labels in insertion order."""

        return list(self._labels)

    @property
    def vectors(self) -> np.ndarray:
        """The normalised vectors, shape ``(size, dimension)``."""

        return self._vectors

    def __len__(self) -> int:
        return len(self._labels)

    def __repr__(self) -> str:
        return f"SimilarityIndex(size={len(self)}, dimension={self._dimension})"

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str | Path) -> Path:
        """Write vectors and labels to a directory."""

        target = ensure_directory(directory)

        with (
            atomic_write_path(target / _VECTORS_FILENAME) as temporary,
            temporary.open("wb") as handle,
        ):
            np.save(handle, self._vectors, allow_pickle=False)

        write_json(
            target / _METADATA_FILENAME,
            {
                "format_version": _FORMAT_VERSION,
                "dimension": self._dimension,
                "labels": self._labels,
            },
        )

        _logger.info("Saved similarity index", extra={"path": str(target), "size": len(self)})

        return target

    @classmethod
    def load(
        cls,
        directory: str | Path,
        encoder: SentenceEncoder | None = None,
    ) -> SimilarityIndex:
        """
        Read an index written by :meth:`save`.

        Raises
        ------
        ResourceNotFoundError
            If the directory or either file is missing.

        ValidationError
            If the format version is unsupported, or the stored vectors
            and labels disagree in length.
        """

        source = require_directory(directory)

        metadata_path = source / _METADATA_FILENAME

        vectors_path = source / _VECTORS_FILENAME

        for path in (metadata_path, vectors_path):
            if not path.is_file():
                raise ResourceNotFoundError("Index file is missing", path=str(path))

        metadata: dict[str, Any] = read_json(metadata_path)

        if metadata.get("format_version") != _FORMAT_VERSION:
            raise ValidationError(
                "Unsupported index format version",
                found=metadata.get("format_version"),
                expected=_FORMAT_VERSION,
            )

        labels = [str(label) for label in metadata.get("labels", [])]

        with vectors_path.open("rb") as handle:
            vectors = np.load(handle, allow_pickle=False)

        if vectors.shape[0] != len(labels):
            raise ValidationError(
                "Stored index vectors and labels differ in length",
                rows=int(vectors.shape[0]),
                labels=len(labels),
            )

        index = cls(int(metadata["dimension"]), encoder=encoder)

        index._labels = labels

        index._vectors = vectors.astype(np.float32)

        return index

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_query(self, query: str | np.ndarray) -> np.ndarray:
        """Return a unit-length query vector."""

        if isinstance(query, str):
            if self._encoder is None:
                raise ValidationError(
                    "Text queries require an encoder",
                    name="query",
                    value=query,
                )

            vector = np.asarray(self._encoder.encode(query), dtype=np.float32).reshape(-1)
        else:
            vector = np.asarray(query, dtype=np.float32).reshape(-1)

        if vector.shape[0] != self._dimension:
            raise ValidationError(
                "Query vector has the wrong dimension",
                found=int(vector.shape[0]),
                expected=self._dimension,
            )

        norm = float(np.linalg.norm(vector))

        return (vector / max(norm, _NORM_EPSILON)).astype(np.float32)


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    """Scale each row to unit length, leaving zero rows at zero."""

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)

    scaled: np.ndarray = vectors / np.maximum(norms, _NORM_EPSILON)

    return scaled.astype(np.float32)
