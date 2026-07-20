"""
Embedding evaluation.

Word vectors are assessed two ways here.

*Intrinsic* metrics compare the geometry against human judgement — do
pairs humans call similar actually sit close together, and do analogies
resolve. These need a labelled dataset.

*Structural* metrics need no labels and describe the geometry itself:
how much of the space the vectors actually occupy, and whether the
neighbourhoods are coherent. They are the ones available on a fresh
corpus in a language with no benchmark, which is the normal situation
for most of the languages this framework targets.

Isotropy in particular is worth watching. Embedding matrices routinely
collapse toward a narrow cone, where every pair looks similar and cosine
similarity stops discriminating. A high mean pairwise similarity on
random tokens is the symptom.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from multilingual_embedding.core.logging import get_logger
from multilingual_embedding.embedding.matrix import EmbeddingMatrix
from multilingual_embedding.utils.io import read_jsonl

from .metrics import spearman_correlation

__all__ = [
    "AnalogyQuestion",
    "EmbeddingEvaluator",
    "EmbeddingMetrics",
    "SimilarityPair",
    "load_analogy_dataset",
    "load_similarity_dataset",
]

_logger = get_logger(__name__)


@dataclass(slots=True, frozen=True)
class SimilarityPair:
    """A human-judged similarity between two words."""

    word_a: str

    word_b: str

    score: float


@dataclass(slots=True, frozen=True)
class AnalogyQuestion:
    """
    An analogy of the form ``a : b :: c : expected``.

    Example: ``man : king :: woman : queen``.
    """

    a: str

    b: str

    c: str

    expected: str


@dataclass(slots=True)
class EmbeddingMetrics:
    """
    Quality figures for an embedding matrix.

    Attributes
    ----------
    vocabulary_size, dimension:
        Shape of the matrix.

    similarity_correlation:
        Spearman correlation against human similarity judgements, in
        [-1, 1]. ``None`` when no dataset was supplied.

    similarity_coverage:
        Share of dataset pairs where both words were in vocabulary. A
        strong correlation over 10% of the pairs says little, so this
        must be read alongside the correlation.

    analogy_accuracy, analogy_coverage:
        Share of analogy questions answered correctly, and the share
        that could be attempted at all.

    mean_pairwise_similarity:
        Mean cosine similarity between random token pairs. Near zero is
        healthy; a high value means the space has collapsed into a
        narrow cone and cosine similarity has lost its power to
        discriminate.

    isotropy:
        Ratio of the smallest to largest singular value of the vector
        matrix. Higher means the vectors use the available dimensions
        more evenly.

    effective_dimensions:
        Number of principal components needed to explain 95% of the
        variance. Far below ``dimension`` means capacity is being wasted.

    zero_vector_count:
        Rows that never received an update. Nonzero for tokens present
        in the vocabulary but absent from training.
    """

    vocabulary_size: int = 0

    dimension: int = 0

    similarity_correlation: float | None = None

    similarity_coverage: float | None = None

    analogy_accuracy: float | None = None

    analogy_coverage: float | None = None

    mean_pairwise_similarity: float = 0.0

    isotropy: float = 0.0

    effective_dimensions: int = 0

    zero_vector_count: int = 0

    neighbour_examples: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Reduce to primitives for report files."""

        return {
            "vocabulary_size": self.vocabulary_size,
            "dimension": self.dimension,
            "similarity_correlation": _rounded(self.similarity_correlation),
            "similarity_coverage": _rounded(self.similarity_coverage),
            "analogy_accuracy": _rounded(self.analogy_accuracy),
            "analogy_coverage": _rounded(self.analogy_coverage),
            "mean_pairwise_similarity": round(self.mean_pairwise_similarity, 5),
            "isotropy": round(self.isotropy, 5),
            "effective_dimensions": self.effective_dimensions,
            "zero_vector_count": self.zero_vector_count,
            "neighbour_examples": self.neighbour_examples,
        }


class EmbeddingEvaluator:
    """
    Scores an :class:`EmbeddingMatrix`.

    Parameters
    ----------
    matrix:
        The embedding matrix to evaluate.

    seed:
        Seed for the random pair sampling used by the structural
        metrics, so a report is reproducible.
    """

    __slots__ = ("_matrix", "_seed")

    def __init__(self, matrix: EmbeddingMatrix, *, seed: int = 42) -> None:
        self._matrix = matrix

        self._seed = seed

    def evaluate(
        self,
        *,
        similarity_pairs: Sequence[SimilarityPair] | None = None,
        analogies: Sequence[AnalogyQuestion] | None = None,
        neighbour_probes: Sequence[str] = (),
        top_k: int = 5,
        sample_size: int = 2_000,
    ) -> EmbeddingMetrics:
        """
        Compute every available metric.

        Parameters
        ----------
        similarity_pairs, analogies:
            Optional labelled datasets. Omitted metrics stay ``None``
            rather than defaulting to zero, so a missing benchmark is
            never mistaken for a failing score.

        neighbour_probes:
            Tokens whose nearest neighbours are recorded in the report,
            for eyeballing quality.

        sample_size:
            Number of random pairs drawn for the structural metrics.
        """

        metrics = EmbeddingMetrics(
            vocabulary_size=len(self._matrix),
            dimension=self._matrix.dimension,
        )

        if similarity_pairs:
            correlation, coverage = self.similarity_correlation(similarity_pairs)

            metrics.similarity_correlation = correlation

            metrics.similarity_coverage = coverage

        if analogies:
            accuracy, coverage = self.analogy_accuracy(analogies)

            metrics.analogy_accuracy = accuracy

            metrics.analogy_coverage = coverage

        metrics.mean_pairwise_similarity = self.mean_pairwise_similarity(sample_size=sample_size)

        metrics.isotropy, metrics.effective_dimensions = self.spectrum()

        metrics.zero_vector_count = self.zero_vector_count()

        metrics.neighbour_examples = {
            probe: [token for token, _ in self._matrix.most_similar(probe, top_k)]
            for probe in neighbour_probes
            if probe in self._matrix
        }

        _logger.info(
            "Evaluated embeddings",
            extra={
                "vocabulary_size": metrics.vocabulary_size,
                "isotropy": round(metrics.isotropy, 4),
                "mean_pairwise_similarity": round(metrics.mean_pairwise_similarity, 4),
            },
        )

        return metrics

    def similarity_correlation(
        self,
        pairs: Sequence[SimilarityPair],
    ) -> tuple[float, float]:
        """
        Spearman correlation against human similarity judgements.

        Returns the correlation and the share of pairs that could be
        scored. Pairs with an out-of-vocabulary word are skipped rather
        than scored as zero, which would understate a model that simply
        has a smaller vocabulary.
        """

        predicted: list[float] = []

        expected: list[float] = []

        for pair in pairs:
            if pair.word_a not in self._matrix or pair.word_b not in self._matrix:
                continue

            predicted.append(self._matrix.similarity(pair.word_a, pair.word_b))

            expected.append(pair.score)

        coverage = len(predicted) / len(pairs) if pairs else 0.0

        if len(predicted) < 2:
            return 0.0, coverage

        return spearman_correlation(predicted, expected), coverage

    def analogy_accuracy(
        self,
        questions: Sequence[AnalogyQuestion],
    ) -> tuple[float, float]:
        """
        Share of analogy questions whose top answer is correct.

        Returns accuracy over attempted questions plus the share
        attempted.
        """

        attempted = 0

        correct = 0

        for question in questions:
            terms = (question.a, question.b, question.c, question.expected)

            if any(term not in self._matrix for term in terms):
                continue

            attempted += 1

            results = self._matrix.analogy(
                positive=[question.b, question.c],
                negative=[question.a],
                top_k=1,
            )

            if results and results[0][0] == question.expected:
                correct += 1

        coverage = attempted / len(questions) if questions else 0.0

        return (correct / attempted if attempted else 0.0), coverage

    def mean_pairwise_similarity(self, *, sample_size: int = 2_000) -> float:
        """
        Mean cosine similarity between randomly drawn token pairs.

        Special tokens and zero rows are excluded, since they would drag
        the figure toward zero for reasons unrelated to the geometry.
        """

        vectors = self._informative_vectors()

        if len(vectors) < 2:
            return 0.0

        generator = np.random.default_rng(self._seed)

        left = generator.integers(0, len(vectors), size=sample_size)

        right = generator.integers(0, len(vectors), size=sample_size)

        keep = left != right

        if not keep.any():
            return 0.0

        normalized = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)

        similarities = np.sum(normalized[left[keep]] * normalized[right[keep]], axis=1)

        return float(np.mean(similarities))

    def spectrum(self) -> tuple[float, int]:
        """
        Return the isotropy ratio and the effective dimension count.

        Isotropy is the ratio of smallest to largest singular value.
        Effective dimensions is how many principal components explain
        95% of the variance.
        """

        vectors = self._informative_vectors()

        if len(vectors) < 2:
            return 0.0, 0

        centred = vectors - vectors.mean(axis=0, keepdims=True)

        singular_values = np.linalg.svd(centred, compute_uv=False)

        if singular_values.size == 0 or singular_values[0] == 0.0:
            return 0.0, 0

        isotropy = float(singular_values[-1] / singular_values[0])

        variance = singular_values**2

        cumulative = np.cumsum(variance) / np.sum(variance)

        effective = int(np.searchsorted(cumulative, 0.95) + 1)

        return isotropy, effective

    def zero_vector_count(self) -> int:
        """Count rows that are entirely zero."""

        return int(np.sum(np.all(self._matrix.vectors == 0.0, axis=1)))

    def _informative_vectors(self) -> NDArray[np.float32]:
        """Return non-special, non-zero rows."""

        special_count = self._matrix.vocabulary.special_tokens.count

        vectors = self._matrix.vectors[special_count:]

        if len(vectors) == 0:
            return vectors

        non_zero = ~np.all(vectors == 0.0, axis=1)

        informative: NDArray[np.float32] = vectors[non_zero]

        return informative


def load_similarity_dataset(path: str | Path) -> list[SimilarityPair]:
    """
    Read a word similarity dataset from JSON Lines.

    Each record must hold ``word_a``, ``word_b`` and ``score``.
    """

    pairs: list[SimilarityPair] = []

    for record in read_jsonl(path):
        pairs.append(
            SimilarityPair(
                word_a=str(record["word_a"]),
                word_b=str(record["word_b"]),
                score=float(record["score"]),
            )
        )

    return pairs


def load_analogy_dataset(path: str | Path) -> list[AnalogyQuestion]:
    """
    Read an analogy dataset from JSON Lines.

    Each record must hold ``a``, ``b``, ``c`` and ``expected``.
    """

    questions: list[AnalogyQuestion] = []

    for record in read_jsonl(path):
        questions.append(
            AnalogyQuestion(
                a=str(record["a"]),
                b=str(record["b"]),
                c=str(record["c"]),
                expected=str(record["expected"]),
            )
        )

    return questions


def sample_neighbour_probes(
    matrix: EmbeddingMatrix,
    count: int = 10,
) -> list[str]:
    """
    Pick frequent tokens to use as nearest-neighbour probes.

    Frequent tokens are chosen because they have the most training
    signal, so their neighbourhoods are the fairest illustration of what
    the model learned.
    """

    return [token for token, _ in matrix.vocabulary.most_common(count)]


def _rounded(value: float | None, digits: int = 5) -> float | None:
    """Round a value that may be absent."""

    return round(value, digits) if value is not None else None
