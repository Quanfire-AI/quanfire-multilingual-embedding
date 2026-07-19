"""
The embedding matrix: a vocabulary bound to its vectors.

A bare numpy array is not an embedding. Row 4179 means nothing without
the vocabulary that assigned that id, and the two drift apart easily —
rebuild the vocabulary with a different ``min_count`` and every row now
refers to a different token, silently, with no exception and no NaN to
warn you. Pairing them in one object, validating the shapes at
construction, and persisting them together is what stops that.

All arithmetic here is vectorised. Neighbour search over a 100k
vocabulary is a single matmul; a Python loop over rows would be three
orders of magnitude slower and is never the right answer.
"""

from __future__ import annotations

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
from multilingual_embedding.vocabulary.vocabulary import Vocabulary

__all__ = ["EmbeddingMatrix"]

_logger = get_logger(__name__)

_FORMAT_VERSION = 1

_VECTORS_FILENAME = "vectors.npy"

_VOCABULARY_FILENAME = "vocabulary.json"

_METADATA_FILENAME = "metadata.json"

# Floor for L2 norms. The pad row is all zeros by construction, and
# dividing it by its own norm would produce NaN across the whole row.
_NORM_EPSILON = 1e-12


class EmbeddingMatrix:
    """
    Vectors indexed by a vocabulary's token ids.

    Parameters
    ----------
    vocabulary:
        Token to id mapping. Row ``i`` of ``vectors`` belongs to the
        token with id ``i``.

    vectors:
        Array of shape ``(len(vocabulary), dimension)``. Cast to
        float32, which halves memory against float64 at no measurable
        cost in neighbour quality.

    Raises
    ------
    ValidationError
        If the array is not two dimensional, or its row count differs
        from the vocabulary size.

    Example
    -------
    ::

        matrix = EmbeddingMatrix(vocabulary, vectors)

        matrix.most_similar("king", top_k=5)
    """

    __slots__ = ("_vectors", "_vocabulary")

    def __init__(self, vocabulary: Vocabulary, vectors: np.ndarray) -> None:
        array = np.asarray(vectors, dtype=np.float32)

        if array.ndim != 2:
            raise ValidationError(
                "Embedding vectors must be a two dimensional array",
                ndim=array.ndim,
                shape=tuple(array.shape),
            )

        if array.shape[0] != len(vocabulary):
            raise ValidationError(
                "Embedding rows do not match vocabulary size",
                rows=int(array.shape[0]),
                vocabulary_size=len(vocabulary),
            )

        self._vocabulary = vocabulary

        self._vectors = array

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def vocabulary(self) -> Vocabulary:
        """The vocabulary indexing these vectors."""

        return self._vocabulary

    @property
    def vectors(self) -> np.ndarray:
        """The underlying array, shape ``(size, dimension)``."""

        return self._vectors

    @property
    def dimension(self) -> int:
        """Length of a single vector."""

        return int(self._vectors.shape[1])

    def __len__(self) -> int:
        return int(self._vectors.shape[0])

    def __contains__(self, token: object) -> bool:
        return isinstance(token, str) and token in self._vocabulary

    def __repr__(self) -> str:
        return f"EmbeddingMatrix(size={len(self)}, dimension={self.dimension})"

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def vector_for(self, token: str) -> np.ndarray:
        """
        Return the vector for a token.

        Out-of-vocabulary tokens resolve to the unknown row rather than
        raising: encountering an unseen word at inference time is normal,
        not a defect.
        """

        row: np.ndarray = self._vectors[self._vocabulary.id_of(token)]

        return row

    def vector_for_id(self, token_id: int) -> np.ndarray:
        """
        Return the vector for a token id.

        Raises
        ------
        ValidationError
            If the id falls outside the matrix.
        """

        if not 0 <= token_id < len(self):
            raise ValidationError(
                "Token id is out of range",
                token_id=token_id,
                size=len(self),
            )

        row: np.ndarray = self._vectors[token_id]

        return row

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def normalized(self) -> EmbeddingMatrix:
        """
        Return a copy with every row scaled to unit L2 length.

        Returns
        -------
        EmbeddingMatrix
            A new matrix sharing the same vocabulary. Zero rows stay
            zero instead of becoming NaN.
        """

        return EmbeddingMatrix(self._vocabulary, _normalize_rows(self._vectors))

    def similarity(self, a: str, b: str) -> float:
        """
        Cosine similarity between two tokens.

        Returns
        -------
        float
            In ``[-1, 1]``, or ``0.0`` when either vector is all zeros.

        Example
        -------
        ::

            matrix.similarity("cat", "dog")
        """

        return _cosine(self.vector_for(a), self.vector_for(b))

    def most_similar(
        self,
        token_or_vector: str | np.ndarray,
        top_k: int = 10,
        exclude_special: bool = True,
    ) -> list[tuple[str, float]]:
        """
        Return the nearest neighbours by cosine similarity.

        Parameters
        ----------
        token_or_vector:
            Query token, or a raw vector of length ``dimension``.

        top_k:
            Number of neighbours to return. Fewer are returned when the
            vocabulary is smaller than the request.

        exclude_special:
            Drop pad/unk/bos/eos rows from the results. Their vectors
            are artefacts of training rather than meaningful points, so
            they are noise in a neighbour list.

        Returns
        -------
        list[tuple[str, float]]
            ``(token, score)`` pairs in descending score order.

        Raises
        ------
        ValidationError
            If ``top_k`` is not positive, or a supplied vector has the
            wrong length.
        """

        query, excluded = self._resolve_query(token_or_vector)

        return self._rank(query, top_k=top_k, excluded=excluded, exclude_special=exclude_special)

    def analogy(
        self,
        positive: list[str],
        negative: list[str],
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """
        Solve a vector analogy such as king - man + woman.

        Parameters
        ----------
        positive:
            Tokens whose vectors are added.

        negative:
            Tokens whose vectors are subtracted.

        top_k:
            Number of results.

        Returns
        -------
        list[tuple[str, float]]
            ``(token, score)`` pairs in descending score order.

        Raises
        ------
        ValidationError
            If no positive or negative token is supplied.

        Example
        -------
        ::

            matrix.analogy(["king", "woman"], ["man"], top_k=1)
        """

        if not positive and not negative:
            raise ValidationError("Analogy requires at least one input token")

        query = np.zeros(self.dimension, dtype=np.float32)

        for token in positive:
            query = query + _unit(self.vector_for(token))

        for token in negative:
            query = query - _unit(self.vector_for(token))

        # The inputs sit closest to their own sum, so leaving them in
        # would return "king, woman, man" and bury the actual answer.
        excluded = {self._vocabulary.id_of(token) for token in (*positive, *negative)}

        return self._rank(query, top_k=top_k, excluded=excluded, exclude_special=True)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str | Path) -> Path:
        """
        Write vectors, vocabulary and metadata to a directory.

        Parameters
        ----------
        directory:
            Destination. Created if missing.

        Returns
        -------
        Path
            The directory written to.
        """

        target = ensure_directory(directory)

        # np.save appends .npy to a path but not to an already open handle,
        # which would otherwise defeat the atomic rename.
        with (
            atomic_write_path(target / _VECTORS_FILENAME) as temporary,
            temporary.open("wb") as handle,
        ):
            np.save(handle, self._vectors, allow_pickle=False)

        self._vocabulary.save(target / _VOCABULARY_FILENAME)

        write_json(
            target / _METADATA_FILENAME,
            {
                "format_version": _FORMAT_VERSION,
                "dimension": self.dimension,
                "size": len(self),
                "dtype": str(self._vectors.dtype),
            },
        )

        _logger.info(
            "Saved embedding matrix",
            extra={"path": str(target), "size": len(self), "dimension": self.dimension},
        )

        return target

    @classmethod
    def load(cls, directory: str | Path) -> EmbeddingMatrix:
        """
        Read a matrix written by :meth:`save`.

        Raises
        ------
        ResourceNotFoundError
            If the directory or any expected file is missing.

        ValidationError
            If the format version is unsupported, or the stored vectors
            disagree with the stored vocabulary.
        """

        source = require_directory(directory)

        metadata_path = source / _METADATA_FILENAME

        if not metadata_path.is_file():
            raise ResourceNotFoundError(
                "Embedding metadata is missing",
                path=str(metadata_path),
            )

        metadata: dict[str, Any] = read_json(metadata_path)

        version = metadata.get("format_version")

        if version != _FORMAT_VERSION:
            raise ValidationError(
                "Unsupported embedding format version",
                found=version,
                expected=_FORMAT_VERSION,
            )

        vectors_path = source / _VECTORS_FILENAME

        if not vectors_path.is_file():
            raise ResourceNotFoundError("Embedding vectors are missing", path=str(vectors_path))

        vocabulary = Vocabulary.load(source / _VOCABULARY_FILENAME)

        with vectors_path.open("rb") as handle:
            vectors = np.load(handle, allow_pickle=False)

        if vectors.shape[0] != len(vocabulary):
            raise ValidationError(
                "Stored vectors do not match the stored vocabulary",
                rows=int(vectors.shape[0]),
                vocabulary_size=len(vocabulary),
            )

        expected_dimension = metadata.get("dimension")

        if expected_dimension is not None and int(vectors.shape[1]) != int(expected_dimension):
            raise ValidationError(
                "Stored vectors do not match the recorded dimension",
                found=int(vectors.shape[1]),
                expected=int(expected_dimension),
            )

        return cls(vocabulary, vectors)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_query(self, token_or_vector: str | np.ndarray) -> tuple[np.ndarray, set[int]]:
        """Return the query vector and the ids to suppress from results."""

        if isinstance(token_or_vector, str):
            return self.vector_for(token_or_vector), {self._vocabulary.id_of(token_or_vector)}

        query = np.asarray(token_or_vector, dtype=np.float32).reshape(-1)

        if query.shape[0] != self.dimension:
            raise ValidationError(
                "Query vector has the wrong dimension",
                found=int(query.shape[0]),
                expected=self.dimension,
            )

        return query, set()

    def _rank(
        self,
        query: np.ndarray,
        *,
        top_k: int,
        excluded: set[int],
        exclude_special: bool,
    ) -> list[tuple[str, float]]:
        """Score every row against the query and return the best ``top_k``."""

        if top_k < 1:
            raise ValidationError("top_k must be >= 1", top_k=top_k)

        scores = _normalize_rows(self._vectors) @ _unit(query)

        if exclude_special:
            excluded = excluded | set(range(self._vocabulary.special_tokens.count))

        if excluded:
            scores = scores.copy()

            scores[list(excluded)] = -np.inf

        available = len(self) - len(excluded)

        if available < 1:
            return []

        count = min(top_k, available)

        # argpartition finds the top-k boundary in O(n); only those k are
        # then sorted, which matters once the vocabulary is large.
        candidates = np.argpartition(-scores, count - 1)[:count]

        ordered = candidates[np.argsort(-scores[candidates], kind="stable")]

        return [(self._vocabulary.token_of(int(index)), float(scores[index])) for index in ordered]


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    """Scale each row to unit length, leaving zero rows at zero."""

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)

    scaled: np.ndarray = vectors / np.maximum(norms, _NORM_EPSILON)

    return scaled.astype(np.float32)


def _unit(vector: np.ndarray) -> np.ndarray:
    """Scale a single vector to unit length, leaving a zero vector at zero."""

    norm = float(np.linalg.norm(vector))

    return (vector / max(norm, _NORM_EPSILON)).astype(np.float32)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity, defined as zero when either side has no length."""

    denominator = float(np.linalg.norm(a)) * float(np.linalg.norm(b))

    if denominator < _NORM_EPSILON:
        return 0.0

    return float(np.dot(a, b) / denominator)
