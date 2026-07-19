"""
The TextEncoder contract, and the decoupling it exists for.

The point of this contract is that the search pipeline should work with
a model that has no embedding matrix, because a contextual encoder
computes vectors at call time and has no per-token table to hold. The
decisive test in this module builds a pipeline from an encoder backed by
no model at all and searches with it — if that works, the pipeline is
genuinely decoupled rather than merely refactored.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from multilingual_embedding.embedding.encoder import TextEncoder, encoder_dimension
from multilingual_embedding.embedding.matrix import EmbeddingMatrix
from multilingual_embedding.embedding.sentence import MeanPoolingEncoder, SifEncoder
from multilingual_embedding.pipelines.search import SemanticSearchPipeline
from multilingual_embedding.vocabulary.vocabulary import Vocabulary

DIMENSION = 8


class HashingEncoder:
    """
    A deterministic encoder backed by no model whatsoever.

    It stands in for a contextual model in exactly the respect that
    matters here: there is no vocabulary, no matrix, and nothing to look
    up. Each text is hashed to a fixed vector at call time.
    """

    def __init__(self, dimension: int = DIMENSION) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(self, text: str) -> np.ndarray:
        if not text.strip():
            return np.zeros(self._dimension, dtype=np.float32)

        generator = np.random.default_rng(abs(hash(text)) % (2**32))

        return generator.standard_normal(self._dimension).astype(np.float32)

    def encode_batch(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float32)

        return np.vstack([self.encode(text) for text in texts])


@pytest.fixture
def static_matrix() -> EmbeddingMatrix:
    vocabulary = Vocabulary.from_counter({"alpha": 5, "beta": 4, "gamma": 3}, min_count=1)

    generator = np.random.default_rng(7)

    vectors = generator.standard_normal((len(vocabulary), DIMENSION)).astype(np.float32)

    return EmbeddingMatrix(vocabulary, vectors)


class TestProtocolConformance:
    def test_stub_encoder_satisfies_the_protocol(self) -> None:
        """Structural typing: no inheritance from anything in the framework."""

        assert isinstance(HashingEncoder(), TextEncoder)

    def test_existing_static_encoders_already_satisfy_it(
        self, static_matrix: EmbeddingMatrix
    ) -> None:
        """
        The contract was chosen to fit what already existed.

        Both shipped encoders conform without modification, which is why
        the refactor changed no encoder behaviour.
        """

        assert isinstance(MeanPoolingEncoder(static_matrix), TextEncoder)

        assert isinstance(SifEncoder(static_matrix), TextEncoder)

    def test_an_object_missing_a_method_does_not_conform(self) -> None:
        class Incomplete:
            dimension = DIMENSION

            def encode(self, text: str) -> np.ndarray:
                return np.zeros(DIMENSION, dtype=np.float32)

        assert not isinstance(Incomplete(), TextEncoder)


class TestContractGuarantees:
    def test_encode_returns_a_vector_of_the_declared_width(self) -> None:
        assert HashingEncoder().encode("text").shape == (DIMENSION,)

    def test_encode_batch_returns_one_row_per_input_in_order(self) -> None:
        encoder = HashingEncoder()

        texts = ["one", "two", "three"]

        batch = encoder.encode_batch(texts)

        assert batch.shape == (3, DIMENSION)

        for row, text in enumerate(texts):
            np.testing.assert_array_equal(batch[row], encoder.encode(text))

    def test_unencodable_input_gives_zeros_not_nan(self) -> None:
        """
        A caller detects the degenerate case with a norm check.

        NaN would silently poison every similarity computed against it.
        """

        vector = HashingEncoder().encode("   ")

        assert not np.any(vector)

        assert not np.isnan(vector).any()

    def test_empty_batch_keeps_its_shape(self) -> None:
        assert HashingEncoder().encode_batch([]).shape == (0, DIMENSION)


class TestDimensionVerification:
    def test_accepts_an_honest_encoder(self) -> None:
        assert encoder_dimension(HashingEncoder()) == DIMENSION

    def test_rejects_a_lying_encoder(self) -> None:
        """
        A declared width that does not match real output would otherwise
        surface much later as a corrupt index or a shape error deep
        inside a matrix multiplication.
        """

        class Liar(HashingEncoder):
            @property
            def dimension(self) -> int:
                return DIMENSION + 5

        with pytest.raises(ValueError, match="declares dimension"):
            encoder_dimension(Liar())


class TestPipelineIsGenuinelyDecoupled:
    """
    The exit criterion for the decoupling.

    Each test here would have been impossible before it: the pipeline
    required an EmbeddingMatrix, and none of these have one.
    """

    def test_pipeline_accepts_an_encoder_with_no_matrix(self) -> None:
        pipeline = SemanticSearchPipeline(HashingEncoder())

        assert pipeline.matrix is None

        assert pipeline.tokenizer is None

        assert pipeline.encoder.dimension == DIMENSION

    def test_search_works_without_any_matrix(self) -> None:
        pipeline = SemanticSearchPipeline(HashingEncoder())

        corpus = ["the first sentence", "a second one", "and a third"]

        assert pipeline.index(corpus) == 3

        hits = pipeline.search("the first sentence", top_k=3)

        assert len(hits) == 3

        assert [hit.rank for hit in hits] == [1, 2, 3]

    def test_an_exact_query_ranks_its_own_sentence_first(self) -> None:
        """
        The encoder is deterministic, so an exact query must match
        itself perfectly. This proves vectors survive the round trip
        through indexing and search unchanged.
        """

        pipeline = SemanticSearchPipeline(HashingEncoder())

        pipeline.index(["alpha text", "beta text", "gamma text"])

        top = pipeline.search("beta text", top_k=1)[0]

        assert top.text == "beta text"

        assert top.score == pytest.approx(1.0, abs=1e-5)

    def test_results_are_ordered_by_descending_score(self) -> None:
        pipeline = SemanticSearchPipeline(HashingEncoder())

        pipeline.index([f"sentence number {index}" for index in range(20)])

        hits = pipeline.search("sentence number 7", top_k=10)

        assert all(hits[index].score >= hits[index + 1].score for index in range(len(hits) - 1))

    def test_similar_tokens_is_empty_without_a_matrix(self) -> None:
        """
        Token neighbourhoods are meaningful for static models only.

        A contextual model has no per-token table, so the honest answer
        is nothing rather than an error or a fabricated result.
        """

        pipeline = SemanticSearchPipeline(HashingEncoder())

        assert pipeline.similar_tokens("anything") == []

    def test_static_path_still_exposes_its_matrix(self, static_matrix: EmbeddingMatrix) -> None:
        """The decoupling must not cost the static model its extras."""

        pipeline = SemanticSearchPipeline(MeanPoolingEncoder(static_matrix), matrix=static_matrix)

        assert pipeline.matrix is not None

        assert pipeline.similar_tokens("alpha", top_k=2) != []
