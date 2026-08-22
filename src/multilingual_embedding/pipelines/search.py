"""
Semantic search pipeline.

Encodes a set of sentences into vectors and answers nearest neighbour
queries over them.

The pipeline depends on the
:class:`~multilingual_embedding.embedding.encoder.TextEncoder` contract
rather than on an embedding matrix, so it serves static and contextual
models alike. :meth:`SemanticSearchPipeline.from_directory` builds the
static path from a trained experiment directory;
:meth:`SemanticSearchPipeline.from_adapter` builds the adapted
contextual one from a saved LoRA adapter.

**Queries and passages are not encoded identically, and the pipeline is
where that asymmetry lives.** An E5-family model is trained with
``query:`` on one side and ``passage:`` on the other, and served without
them it still returns vectors — just worse ones, with nothing saying so.
The encoder cannot apply them, because ``encode`` takes text and has no
idea which side it is on. Only the caller of :meth:`index` versus
:meth:`search` knows, and that caller is this class. So the prefixes are
held here and applied automatically, rather than documented and left to
be remembered.

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
from numpy.typing import NDArray

from multilingual_embedding.config.base import ExperimentConfig
from multilingual_embedding.core.exceptions import ResourceNotFoundError
from multilingual_embedding.core.logging import get_logger
from multilingual_embedding.embedding.encoder import TextEncoder
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

    Depends on the :class:`~multilingual_embedding.embedding.encoder.TextEncoder`
    contract — text in, vectors out — rather than on an embedding matrix.
    That is what lets it serve a contextual model, which computes vectors
    at call time and has no per-token table to hold.

    Parameters
    ----------
    encoder:
        Anything satisfying ``TextEncoder``.

    matrix:
        The embedding matrix, when the encoder happens to be backed by
        one. Optional, and ``None`` for contextual models. It is kept
        only to support :meth:`similar_tokens`, which is meaningful for
        static models alone.

    tokenizer:
        The tokenizer the static encoder segments with, when there is a
        separable one. ``None`` for a contextual model, whose tokenizer
        is internal to it.

    query_prefix, passage_prefix:
        Markers the model was trained to expect on each side, prepended
        by :meth:`search` and :meth:`index` respectively. Empty for a
        symmetric model, which is every static one. Prefer
        :meth:`from_adapter`, which reads them from the artefact instead
        of asking the caller to know them.

    Example
    -------
    ::

        pipeline = SemanticSearchPipeline.from_directory("artifacts/demo")

        pipeline.index(corpus.sentence_texts())

        for hit in pipeline.search("machine learning", top_k=5):
            print(hit.rank, round(hit.score, 3), hit.text)
    """

    __slots__ = (
        "_encoder",
        "_index",
        "_matrix",
        "_passage_prefix",
        "_query_prefix",
        "_texts",
        "_tokenizer",
    )

    def __init__(
        self,
        encoder: TextEncoder,
        *,
        matrix: EmbeddingMatrix | None = None,
        tokenizer: SentencePieceTokenizer | None = None,
        query_prefix: str = "",
        passage_prefix: str = "",
    ) -> None:
        self._encoder = encoder

        self._matrix = matrix

        self._tokenizer = tokenizer

        self._query_prefix = query_prefix

        self._passage_prefix = passage_prefix

        self._index: SimilarityIndex | None = None

        self._texts: list[str] = []

    @classmethod
    def from_static(
        cls,
        tokenizer: SentencePieceTokenizer,
        matrix: EmbeddingMatrix,
        encoder: SentenceEncoder | None = None,
    ) -> SemanticSearchPipeline:
        """
        Build a pipeline over a static embedding matrix.

        The tokenizer must be the one the embeddings were trained with;
        a different one produces pieces that index into the wrong rows.

        Parameters
        ----------
        encoder:
            Sentence encoder over the matrix. Defaults to mean pooling.
        """

        resolved = (
            encoder
            if encoder is not None
            else MeanPoolingEncoder(matrix, tokenize=tokenizer.tokenize)
        )

        return cls(resolved, matrix=matrix, tokenizer=tokenizer)

    @classmethod
    def from_directory(
        cls,
        experiment_directory: str | Path,
        *,
        encoder: SentenceEncoder | None = None,
    ) -> SemanticSearchPipeline:
        """
        Load a pipeline from an experiment directory.

        Serves whichever model the run wrote. ``qfme train`` writes an
        ``embedding/`` matrix; ``qfme pretrain`` and its contrastive
        fine-tune write an ``encoder/``. When both are present the
        contextual encoder wins, because it is the more capable model and
        a directory that holds both was used to build both.

        The ``encoder`` override applies to the static path alone — it
        customises how sentences pool over the matrix, which a contextual
        model computes for itself. Passing it while serving an encoder is
        a caller mistake; prefer :meth:`from_encoder`, which is device
        aware, when the model is known to be contextual.

        Raises
        ------
        ResourceNotFoundError
            If ``tokenizer/`` is missing, or neither ``encoder/`` nor
            ``embedding/`` is present, naming what was expected.
        """

        root = Path(experiment_directory).expanduser()

        tokenizer_directory = root / "tokenizer"

        if not tokenizer_directory.is_dir():
            raise ResourceNotFoundError(
                "Experiment directory is missing a required component",
                component="tokenizer",
                expected=str(tokenizer_directory),
            )

        encoder_directory = root / "encoder"

        embedding_directory = root / "embedding"

        # The contextual encoder wins when a run wrote one, so a service
        # pointed at an experiment directory serves the from-scratch model
        # without the caller having to know which kind it is.
        if encoder_directory.is_dir():
            return cls.from_encoder(experiment_directory)

        if not embedding_directory.is_dir():
            raise ResourceNotFoundError(
                "Experiment directory holds neither a trained encoder nor an embedding matrix",
                component="encoder-or-embedding",
                expected=str(embedding_directory),
            )

        tokenizer = SentencePieceTokenizer.load(tokenizer_directory)

        matrix = EmbeddingMatrix.load(embedding_directory)

        _logger.info(
            "Loaded search pipeline",
            extra={"directory": str(root), "vocabulary_size": len(matrix)},
        )

        return cls.from_static(tokenizer, matrix, encoder)

    @classmethod
    def from_encoder(
        cls,
        experiment_directory: str | Path,
        *,
        device: str | None = None,
        batch_size: int = 32,
    ) -> SemanticSearchPipeline:
        """
        Build a pipeline over a from-scratch contextual encoder.

        This is what turns a ``qfme pretrain`` run — or its contrastive
        fine-tune — into something that answers queries. It loads the
        encoder the run saved and the tokenizer trained alongside it, from
        the ``encoder/`` and ``tokenizer/`` subdirectories that pipeline
        writes.

        A from-scratch encoder is symmetric: it was never trained with the
        ``query:``/``passage:`` asymmetry an E5-family checkpoint carries,
        so no prefixes are applied. And it has no per-token matrix, so
        :meth:`similar_tokens` returns nothing — the same as for an
        adapter, and for the same reason.

        Parameters
        ----------
        experiment_directory:
            The directory ``qfme pretrain`` wrote, holding ``tokenizer/``
            and ``encoder/``.

        device:
            Where to run, e.g. ``"cuda"`` or ``"cpu"``. ``None`` picks the
            best available.

        batch_size:
            Sentences per forward pass when indexing.

        Raises
        ------
        ResourceNotFoundError
            If ``tokenizer/`` or ``encoder/`` is missing, naming which.

        ImportError
            If the neural extra is not installed. A contextual encoder
            needs torch; the static paths do not, which is why the import
            is here rather than at module scope.
        """

        root = Path(experiment_directory).expanduser()

        tokenizer_directory = root / "tokenizer"

        encoder_directory = root / "encoder"

        for directory, label in (
            (tokenizer_directory, "tokenizer"),
            (encoder_directory, "encoder"),
        ):
            if not directory.is_dir():
                raise ResourceNotFoundError(
                    "Experiment directory is missing a required component",
                    component=label,
                    expected=str(directory),
                )

        # Imported lazily for the same reason as :meth:`from_adapter`:
        # torch is an optional extra, and importing it at module scope
        # would make the static search path uninstallable without it.
        from multilingual_embedding.embedding.neural import NeuralTextEncoder

        tokenizer = SentencePieceTokenizer.load(tokenizer_directory)

        encoder = NeuralTextEncoder.load(
            encoder_directory,
            tokenizer,
            device=device,
            batch_size=batch_size,
        )

        _logger.info(
            "Loaded contextual search pipeline",
            extra={"directory": str(root), "dimension": encoder.dimension},
        )

        return cls(encoder)

    @classmethod
    def from_config(cls, config: ExperimentConfig) -> SemanticSearchPipeline:
        """Load a pipeline from the directories an experiment config names."""

        return cls.from_directory(config.experiment_directory)

    @classmethod
    def from_adapter(
        cls,
        directory: str | Path,
        *,
        device: str | None = None,
        local_files_only: bool = False,
    ) -> SemanticSearchPipeline:
        """
        Build a pipeline over a saved LoRA adapter.

        This is what turns an adaptation run's output into something that
        answers queries. The base checkpoint named in the adapter's
        metadata is loaded, the low-rank update is applied on top, and
        **the recorded prefixes are carried into the pipeline** — which
        is the whole reason this factory exists rather than leaving
        callers to write ``cls(load_adapter(...)[0])``. That spelling
        loads the right weights and then uses them wrongly, silently.

        There is no matrix and no separable tokenizer, so
        :meth:`similar_tokens` returns nothing here. A contextual model
        has no per-token table to be similar in.

        Parameters
        ----------
        directory:
            An adapter directory as :func:`save_adapter` writes it.

        device:
            Where to run, e.g. ``"cuda"`` or ``"cpu"``. ``None`` picks
            the best available.

        local_files_only:
            Refuse to fetch the base checkpoint from the network. Worth
            setting whenever a served model must not change underneath
            you because an upstream repository moved.

        Raises
        ------
        ImportError
            If the neural extra is not installed. A contextual encoder
            needs torch; the static paths above do not, and that is why
            the import is here rather than at module scope.

        Example
        -------
        ::

            pipeline = SemanticSearchPipeline.from_adapter("models/indic-v1")

            pipeline.index(passages)

            pipeline.search("संविधान में मौलिक अधिकार")
        """

        # Imported lazily: torch is an optional extra, and importing it
        # at module scope would make the static search path — which
        # needs none of it — uninstallable without a training stack.
        from multilingual_embedding.embedding.neural.adapter import load_adapter

        encoder, metadata = load_adapter(
            directory,
            device=device,
            local_files_only=local_files_only,
        )

        _logger.info(
            "Loaded adapted search pipeline",
            extra={
                "directory": str(directory),
                "base": metadata.checkpoint,
                "asymmetric": bool(metadata.query_prefix or metadata.passage_prefix),
            },
        )

        return cls(
            encoder,
            query_prefix=metadata.query_prefix,
            passage_prefix=metadata.passage_prefix,
        )

    @property
    def encoder(self) -> TextEncoder:
        """The encoder queries and documents are embedded with."""

        return self._encoder

    @property
    def tokenizer(self) -> SentencePieceTokenizer | None:
        """
        The tokenizer a static encoder segments with.

        Exposed alongside :attr:`matrix` because the two must agree: a
        tokenizer that normalizes differently from the one that produced
        the vectors yields pieces the matrix has no rows for. ``None``
        for a contextual model, whose tokenizer is not separable.
        """

        return self._tokenizer

    @property
    def matrix(self) -> EmbeddingMatrix | None:
        """
        The embedding matrix, when the encoder is backed by one.

        ``None`` for a contextual model, which computes vectors at call
        time and holds no per-token table.
        """

        return self._matrix

    @property
    def prefixes(self) -> tuple[str, str]:
        """
        The query and passage markers, in that order.

        Exposed so a serving layer can report what it is applying.
        ``("", "")`` means the model is symmetric, which is not the same
        as the prefixes having been forgotten — and the difference is
        invisible in the vectors, which is why it is worth being able to
        read off a running service.
        """

        return self._query_prefix, self._passage_prefix

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

        The passage prefix is applied here and nowhere else, and the
        stored text is the sentence as given: a result should return what
        the caller indexed, not the marker the model needed to see.

        Encoding goes through ``encode_batch`` in one call rather than
        ``encode`` per sentence. For a transformer that is the difference
        between one padded batch per forward pass and one sentence per
        forward pass — roughly the difference between indexing a corpus
        and waiting for it.

        It also matters for
        :class:`~multilingual_embedding.embedding.sentence.SifEncoder`,
        whose common component is estimated from a batch and reused by
        ``encode``. Indexing one sentence at a time never gave it a batch
        to estimate from, so the component was never removed and SIF
        quietly degraded to a weighted average. Corpus in :meth:`index`,
        query in :meth:`search`, is the shape that encoder was written
        for.
        """

        given = [sentence for sentence in sentences if sentence.strip()]

        encoded = self._encoder.encode_batch([self._passage_prefix + text for text in given])

        keep = [index for index, vector in enumerate(encoded) if np.any(vector)]

        texts = [given[index] for index in keep]

        vectors: list[NDArray[np.float32]] = [encoded[index] for index in keep]

        self._texts = texts

        # The dimension is passed explicitly so that indexing a set where
        # every sentence was skipped yields an empty index rather than
        # failing on an inference that has nothing to infer from.
        self._index = SimilarityIndex.build(
            zip(texts, vectors, strict=True),
            dimension=self._encoder.dimension,
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

        vector = self._encoder.encode(self._query_prefix + query)

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

        if self._matrix is None or token not in self._matrix:
            return []

        return self._matrix.most_similar(token, top_k)

    def __repr__(self) -> str:
        # The prefixes appear only when set, so a static pipeline reads
        # as it always did, and an asymmetric one cannot be mistaken for
        # a symmetric one at a glance.
        markers = (
            f", query_prefix={self._query_prefix!r}, passage_prefix={self._passage_prefix!r}"
            if self._query_prefix or self._passage_prefix
            else ""
        )

        return (
            f"SemanticSearchPipeline(indexed={self.indexed_count}, "
            f"dimension={self._encoder.dimension}{markers})"
        )
