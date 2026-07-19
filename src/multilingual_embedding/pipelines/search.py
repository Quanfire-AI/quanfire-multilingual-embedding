"""
Semantic search pipeline.

Loads a trained tokenizer and embedding matrix from an experiment
directory, encodes a set of sentences into vectors, and answers nearest
neighbour queries over them.

This is the inference counterpart to
:mod:`multilingual_embedding.pipelines.training`. It deliberately loads
artefacts from disk rather than accepting in-memory objects, because
that is the path a deployed service takes and it is the one that needs
to be exercised.

Cross-lingual note: because tokenizer and embeddings are trained jointly
over a multilingual corpus, all languages share one vector space and a
query in one language can retrieve sentences in another — but only to
the extent the training corpus contained parallel or comparable content.
Without that signal the languages occupy separate regions of the space.
The honest way to check is the cross-lingual retrieval metric, not the
assumption that a shared space implies alignment.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from multilingual_embedding.config.base import ExperimentConfig
from multilingual_embedding.core.exceptions import ResourceNotFoundError
from multilingual_embedding.core.logging import get_logger
from multilingual_embedding.embedding.index import SearchResult, SimilarityIndex
from multilingual_embedding.embedding.matrix import EmbeddingMatrix
from multilingual_embedding.embedding.sentence import MeanPoolingEncoder, SentenceEncoder
from multilingual_embedding.tokenizer.tokenizer import SentencePieceTokenizer

__all__ = ["SearchHit", "SemanticSearchPipeline"]

_logger = get_logger(__name__)


@dataclass(slots=True, frozen=True)
class SearchHit:
    """
    One search result.

    Attributes
    ----------
    text:
        The matching sentence.

    score:
        Cosine similarity to the query, in [-1, 1].

    rank:
        One based position in the result list.
    """

    text: str

    score: float

    rank: int


class SemanticSearchPipeline:
    """
    Nearest neighbour search over sentence embeddings.

    Parameters
    ----------
    tokenizer:
        Tokenizer matching the one the embeddings were trained with.
        Using a different tokenizer would produce pieces that index into
        the wrong embedding rows.

    matrix:
        The trained embedding matrix.

    encoder:
        Sentence encoder. Defaults to mean pooling.

    Example
    -------
    ::

        pipeline = SemanticSearchPipeline.from_directory("artifacts/demo")

        pipeline.index(corpus.sentence_texts())

        for hit in pipeline.search("machine learning", top_k=5):
            print(hit.rank, round(hit.score, 3), hit.text)
    """

    __slots__ = ("_encoder", "_index", "_matrix", "_texts", "_tokenizer")

    def __init__(
        self,
        tokenizer: SentencePieceTokenizer,
        matrix: EmbeddingMatrix,
        encoder: SentenceEncoder | None = None,
    ) -> None:
        self._tokenizer = tokenizer

        self._matrix = matrix

        self._encoder = (
            encoder
            if encoder is not None
            else MeanPoolingEncoder(matrix, tokenize=tokenizer.tokenize)
        )

        self._index: SimilarityIndex | None = None

        self._texts: list[str] = []

    @classmethod
    def from_directory(
        cls,
        experiment_directory: str | Path,
        *,
        encoder: SentenceEncoder | None = None,
    ) -> SemanticSearchPipeline:
        """
        Load a pipeline from an experiment directory.

        Expects the layout :class:`TrainingPipeline` writes:
        ``tokenizer/`` and ``embedding/`` subdirectories.

        Raises
        ------
        ResourceNotFoundError
            If either subdirectory is missing, naming which one.
        """

        root = Path(experiment_directory).expanduser()

        tokenizer_directory = root / "tokenizer"

        embedding_directory = root / "embedding"

        for directory, label in (
            (tokenizer_directory, "tokenizer"),
            (embedding_directory, "embedding"),
        ):
            if not directory.is_dir():
                raise ResourceNotFoundError(
                    "Experiment directory is missing a required component",
                    component=label,
                    expected=str(directory),
                )

        tokenizer = SentencePieceTokenizer.load(tokenizer_directory)

        matrix = EmbeddingMatrix.load(embedding_directory)

        _logger.info(
            "Loaded search pipeline",
            extra={"directory": str(root), "vocabulary_size": len(matrix)},
        )

        return cls(tokenizer, matrix, encoder=encoder)

    @classmethod
    def from_config(cls, config: ExperimentConfig) -> SemanticSearchPipeline:
        """Load a pipeline from the directories an experiment config names."""

        return cls.from_directory(config.experiment_directory)

    @property
    def matrix(self) -> EmbeddingMatrix:
        """The embedding matrix backing this pipeline."""

        return self._matrix

    @property
    def tokenizer(self) -> SentencePieceTokenizer:
        """
        The tokenizer queries are encoded with.

        Exposed alongside :attr:`matrix` because the two must agree:
        a tokenizer that normalizes differently from the one that
        produced the vectors yields pieces the matrix has no rows for.
        """

        return self._tokenizer

    @property
    def indexed_count(self) -> int:
        """Number of sentences currently indexed."""

        return len(self._texts)

    def index(self, sentences: Iterable[str]) -> int:
        """
        Encode and index a set of sentences, replacing any previous index.

        Returns the number indexed. Sentences that encode to a zero
        vector — because every token was out of vocabulary — are skipped,
        since they would otherwise match every query equally poorly and
        pollute the results.
        """

        texts: list[str] = []

        vectors: list[np.ndarray] = []

        for sentence in sentences:
            if not sentence.strip():
                continue

            vector = self._encoder.encode(sentence)

            if not np.any(vector):
                continue

            texts.append(sentence)

            vectors.append(vector)

        self._texts = texts

        # The dimension is passed explicitly so that indexing a set where
        # every sentence was skipped yields an empty index rather than
        # failing on an inference that has nothing to infer from.
        self._index = SimilarityIndex.build(
            zip(texts, vectors, strict=True),
            dimension=self._matrix.dimension,
        )

        _logger.info("Indexed sentences", extra={"count": len(texts)})

        return len(texts)

    def search(self, query: str, *, top_k: int = 10) -> list[SearchHit]:
        """
        Return the ``top_k`` sentences most similar to ``query``.

        Returns an empty list when nothing has been indexed or when the
        query encodes to a zero vector, rather than raising: an
        unanswerable query is a normal condition for a search service.
        """

        if self._index is None or not self._texts:
            return []

        vector = self._encoder.encode(query)

        if not np.any(vector):
            _logger.debug("Query encoded to a zero vector", extra={"query": query})

            return []

        results: Sequence[SearchResult] = self._index.search(vector, top_k=top_k)

        return [
            SearchHit(text=result.label, score=result.score, rank=rank)
            for rank, result in enumerate(results, start=1)
        ]

    def similar_tokens(self, token: str, *, top_k: int = 10) -> list[tuple[str, float]]:
        """
        Return vocabulary tokens nearest to ``token``.

        Operates on word vectors directly rather than sentence vectors,
        which is the quickest way to sanity check what a model learned.
        """

        if token not in self._matrix:
            return []

        return self._matrix.most_similar(token, top_k)

    def __repr__(self) -> str:
        return (
            f"SemanticSearchPipeline(indexed={self.indexed_count}, "
            f"vocabulary_size={len(self._matrix)})"
        )
