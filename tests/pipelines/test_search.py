"""
The search pipeline's asymmetry: a query is not encoded like a passage.

An E5-family model is trained with ``query:`` on one side and
``passage:`` on the other. Served without them it returns vectors that
look entirely normal and encode the wrong thing, and no norm check, no
shape check and no exception will say so. The only symptom is a lower
score, which is indistinguishable from the model simply not being very
good.

That failure cannot be caught downstream, so it is caught here. These
tests assert on the strings the encoder was actually handed, rather than
on retrieval quality, because retrieval quality is exactly the signal
that goes quietly wrong.
"""

from __future__ import annotations

import zlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from multilingual_embedding.embedding.matrix import EmbeddingMatrix
from multilingual_embedding.embedding.sentence import MeanPoolingEncoder, SifEncoder
from multilingual_embedding.pipelines.search import SemanticSearchPipeline
from multilingual_embedding.vocabulary.vocabulary import Vocabulary

DIMENSION = 8

CORPUS = ["alpha beta", "beta gamma", "gamma alpha"]


class RecordingEncoder:
    """
    A ``TextEncoder`` that remembers every string it was given.

    Deterministic across processes: seeded from ``crc32`` rather than the
    built-in ``hash``, whose result for a `str` is salted per interpreter
    by ``PYTHONHASHSEED``. Nothing here depends on the vectors, but a
    test that changes its data every run is not a test.
    """

    def __init__(self, dimension: int = DIMENSION) -> None:
        self._dimension = dimension

        self.encoded: list[str] = []

        self.batches: list[list[str]] = []

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(self, text: str) -> np.ndarray:
        self.encoded.append(text)

        return self._vector(text)

    def encode_batch(self, texts: Sequence[str]) -> np.ndarray:
        self.batches.append(list(texts))

        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float32)

        return np.vstack([self._vector(text) for text in texts])

    def _vector(self, text: str) -> np.ndarray:
        if not text.strip():
            return np.zeros(self._dimension, dtype=np.float32)

        generator = np.random.default_rng(zlib.crc32(text.encode("utf-8")))

        return generator.standard_normal(self._dimension).astype(np.float32)


@pytest.fixture
def static_matrix() -> EmbeddingMatrix:
    vocabulary = Vocabulary.from_counter({"alpha": 5, "beta": 4, "gamma": 3}, min_count=1)

    generator = np.random.default_rng(7)

    vectors = generator.standard_normal((len(vocabulary), DIMENSION)).astype(np.float32)

    return EmbeddingMatrix(vocabulary, vectors)


class TestPrefixesReachTheEncoder:
    def test_passages_are_prefixed_when_indexed(self) -> None:
        encoder = RecordingEncoder()

        pipeline = SemanticSearchPipeline(encoder, passage_prefix="passage: ")

        pipeline.index(CORPUS)

        assert encoder.batches == [["passage: " + text for text in CORPUS]]

    def test_queries_are_prefixed_when_searched(self) -> None:
        encoder = RecordingEncoder()

        pipeline = SemanticSearchPipeline(encoder, query_prefix="query: ")

        pipeline.index(CORPUS)

        pipeline.search("alpha")

        assert encoder.encoded == ["query: alpha"]

    def test_the_two_sides_get_different_prefixes(self) -> None:
        """
        The whole point of the asymmetry.

        A single ``prefix`` setting would encode a query as a passage,
        which is the failure this class exists to prevent.
        """

        encoder = RecordingEncoder()

        pipeline = SemanticSearchPipeline(
            encoder, query_prefix="query: ", passage_prefix="passage: "
        )

        pipeline.index(["alpha beta"])

        pipeline.search("alpha")

        assert encoder.batches == [["passage: alpha beta"]]

        assert encoder.encoded == ["query: alpha"]

    def test_no_prefix_is_applied_by_default(self) -> None:
        """Static models are symmetric; the default must not change them."""

        encoder = RecordingEncoder()

        pipeline = SemanticSearchPipeline(encoder)

        pipeline.index(CORPUS)

        pipeline.search("alpha")

        assert encoder.batches == [CORPUS]

        assert encoder.encoded == ["alpha"]


class TestPrefixesDoNotLeakIntoResults:
    def test_hits_return_the_text_that_was_indexed(self) -> None:
        """
        The prefix is a detail of the model, not of the caller's corpus.

        A search result carrying ``passage: `` on the front is a result
        that cannot be shown to anyone or looked up in the source.
        """

        pipeline = SemanticSearchPipeline(
            RecordingEncoder(), query_prefix="query: ", passage_prefix="passage: "
        )

        pipeline.index(CORPUS)

        hits = pipeline.search("alpha", top_k=3)

        assert {hit.text for hit in hits} == set(CORPUS)


class TestPrefixesAreVisible:
    def test_prefixes_are_readable_off_the_pipeline(self) -> None:
        pipeline = SemanticSearchPipeline(
            RecordingEncoder(), query_prefix="query: ", passage_prefix="passage: "
        )

        assert pipeline.prefixes == ("query: ", "passage: ")

    def test_a_symmetric_pipeline_reports_empty_prefixes(self) -> None:
        assert SemanticSearchPipeline(RecordingEncoder()).prefixes == ("", "")

    def test_repr_shows_prefixes_only_when_they_are_set(self) -> None:
        """
        Serving without prefixes and serving a symmetric model look the
        same in every metric, so they must not look the same in the repr.
        """

        symmetric = repr(SemanticSearchPipeline(RecordingEncoder()))

        asymmetric = repr(SemanticSearchPipeline(RecordingEncoder(), query_prefix="query: "))

        assert "prefix" not in symmetric

        assert "query: " in asymmetric


class TestIndexingEncodesAsABatch:
    def test_the_corpus_is_encoded_in_one_call(self) -> None:
        """
        A transformer encodes a padded batch per forward pass or one
        sentence per forward pass, and the difference is roughly the
        difference between indexing a corpus and waiting for it.
        """

        encoder = RecordingEncoder()

        SemanticSearchPipeline(encoder).index(CORPUS)

        assert len(encoder.batches) == 1

        assert encoder.encoded == []

    def test_sif_fits_its_common_component_during_indexing(
        self, static_matrix: EmbeddingMatrix
    ) -> None:
        """
        Regression: indexing one sentence at a time never fitted it.

        ``SifEncoder`` estimates the direction shared by all sentences
        from a batch and removes it; ``encode`` reuses that estimate.
        Given no batch it has nothing to estimate from, so SIF degraded
        to a plain weighted average — silently, since the vectors were
        still the right shape and the searches still returned results.
        """

        encoder = SifEncoder(static_matrix)

        assert not encoder.is_fitted

        SemanticSearchPipeline(encoder, matrix=static_matrix).index(CORPUS)

        assert encoder.is_fitted

    def test_unencodable_sentences_are_still_skipped(self, static_matrix: EmbeddingMatrix) -> None:
        """
        Batching must not lose the zero-vector filter.

        A fully out-of-vocabulary sentence matches every query equally
        poorly, so indexing it pollutes every result list.
        """

        pipeline = SemanticSearchPipeline(MeanPoolingEncoder(static_matrix), matrix=static_matrix)

        indexed = pipeline.index(["alpha beta", "   ", "quux corge"])

        assert indexed == 1

        assert pipeline.indexed_count == 1

    def test_indexing_nothing_leaves_an_empty_index(self) -> None:
        pipeline = SemanticSearchPipeline(RecordingEncoder())

        assert pipeline.index([]) == 0

        assert pipeline.search("alpha") == []


class TestFromDirectoryChoosesTheModel:
    """
    Which model an experiment directory serves, and how a broken one fails.

    These assert on the directory-shape logic alone — which subdirectory
    is looked for and in what order — so they need no trained model and no
    torch. The end-to-end proof that a real pretrained encoder is loaded
    and answers queries lives beside the pretraining pipeline.
    """

    def test_a_missing_tokenizer_is_named(self, tmp_path: Path) -> None:
        from multilingual_embedding.core.exceptions import ResourceNotFoundError

        # Neither subdirectory exists; the tokenizer is required for both
        # paths, so it is the one reported.
        with pytest.raises(ResourceNotFoundError) as caught:
            SemanticSearchPipeline.from_directory(tmp_path)

        assert caught.value.context["component"] == "tokenizer"

    def test_a_directory_with_neither_model_is_named(self, tmp_path: Path) -> None:
        from multilingual_embedding.core.exceptions import ResourceNotFoundError

        # A tokenizer but no model at all — neither the static matrix nor a
        # contextual encoder — is a distinct, more specific failure than a
        # missing tokenizer, and the error says so rather than pointing at
        # `embedding/` alone as though the encoder path did not exist.
        (tmp_path / "tokenizer").mkdir()

        with pytest.raises(ResourceNotFoundError) as caught:
            SemanticSearchPipeline.from_directory(tmp_path)

        assert caught.value.context["component"] == "encoder-or-embedding"

    def test_an_encoder_directory_takes_the_contextual_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        When ``encoder/`` is present, ``from_directory`` must delegate to
        the contextual factory rather than trying to load a matrix — even
        if an ``embedding/`` also happens to be there. Proven without torch
        by intercepting the delegation, so the branch is covered on a
        machine with no training stack.
        """

        (tmp_path / "tokenizer").mkdir()

        (tmp_path / "encoder").mkdir()

        (tmp_path / "embedding").mkdir()

        seen: dict[str, object] = {}

        def _fake_from_encoder(cls: object, directory: object) -> str:
            seen["directory"] = directory

            return "contextual"

        monkeypatch.setattr(
            SemanticSearchPipeline, "from_encoder", classmethod(_fake_from_encoder)
        )

        result = SemanticSearchPipeline.from_directory(tmp_path)

        assert result == "contextual"

        assert seen["directory"] == tmp_path
