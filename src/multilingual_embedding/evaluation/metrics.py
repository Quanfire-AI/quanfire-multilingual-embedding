"""
Metric primitives.

Plain functions over plain data. Nothing here knows about tokenizers,
models or corpora, which keeps the metrics independently testable and
reusable across evaluation suites.

Ranking metrics all take a ranked list of predictions and a set of
relevant items.

A query with an empty relevant set has no defined score, and these
functions return ``0.0`` for it rather than raising or returning a
sentinel. That keeps them total and cheap to compose, but it pushes one
responsibility onto the caller: **filter out queries with no relevant
items before averaging**, or those zeros will drag the mean down and
understate the system being measured. :class:`EmbeddingEvaluator` does
this filtering and reports the surviving share as a coverage figure
alongside each score.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from multilingual_embedding.utils.validation import require_positive

__all__ = [
    "accuracy",
    "average_precision",
    "cosine_similarity",
    "cosine_similarity_matrix",
    "f1_score",
    "mean_reciprocal_rank",
    "ndcg_at_k",
    "pearson_correlation",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "spearman_correlation",
]


def cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    """
    Cosine similarity between two vectors.

    Returns 0.0 when either vector has zero magnitude. A zero vector has
    no direction, so any other answer would be arbitrary.
    """

    first_norm = float(np.linalg.norm(first))

    second_norm = float(np.linalg.norm(second))

    if first_norm == 0.0 or second_norm == 0.0:
        return 0.0

    return float(np.dot(first, second) / (first_norm * second_norm))


def cosine_similarity_matrix(vectors: np.ndarray, other: np.ndarray | None = None) -> np.ndarray:
    """
    Pairwise cosine similarity between two sets of row vectors.

    Parameters
    ----------
    vectors:
        Array of shape ``(n, d)``.

    other:
        Array of shape ``(m, d)``. Defaults to ``vectors``.

    Returns
    -------
    Array of shape ``(n, m)``.
    """

    target = vectors if other is None else other

    left = _row_normalized(vectors)

    right = _row_normalized(target)

    similarities: np.ndarray = left @ right.T

    return similarities


def precision_at_k(predictions: Sequence[str], relevant: set[str], k: int) -> float:
    """
    Share of the top ``k`` predictions that are relevant.

    The denominator is ``k``, not the number of predictions returned, so
    a system that returns fewer than ``k`` results is penalised for the
    shortfall rather than rewarded for it.
    """

    require_positive(k, name="k")

    if not relevant:
        return 0.0

    hits = sum(1 for item in predictions[:k] if item in relevant)

    return hits / k


def recall_at_k(predictions: Sequence[str], relevant: set[str], k: int) -> float:
    """Share of relevant items appearing in the top ``k`` predictions."""

    require_positive(k, name="k")

    if not relevant:
        return 0.0

    hits = sum(1 for item in predictions[:k] if item in relevant)

    return hits / len(relevant)


def f1_score(precision: float, recall: float) -> float:
    """Harmonic mean of precision and recall, or 0.0 if both are zero."""

    if precision + recall == 0.0:
        return 0.0

    return 2.0 * precision * recall / (precision + recall)


def average_precision(predictions: Sequence[str], relevant: set[str]) -> float:
    """
    Average of the precision values at each relevant hit.

    Rewards ranking correct answers early, unlike precision at a fixed
    cutoff which is blind to ordering within the cutoff.
    """

    if not relevant:
        return 0.0

    hits = 0

    total = 0.0

    for position, item in enumerate(predictions, start=1):
        if item in relevant:
            hits += 1

            total += hits / position

    return total / len(relevant)


def reciprocal_rank(predictions: Sequence[str], relevant: set[str]) -> float:
    """One divided by the rank of the first relevant item, or 0.0."""

    for position, item in enumerate(predictions, start=1):
        if item in relevant:
            return 1.0 / position

    return 0.0


def mean_reciprocal_rank(
    predictions: Sequence[Sequence[str]],
    relevant: Sequence[set[str]],
) -> float:
    """Mean of :func:`reciprocal_rank` across queries."""

    if not predictions:
        return 0.0

    if len(predictions) != len(relevant):
        raise ValueError("predictions and relevant must have the same length")

    scores = [
        reciprocal_rank(prediction, truth)
        for prediction, truth in zip(predictions, relevant, strict=True)
    ]

    return sum(scores) / len(scores)


def ndcg_at_k(predictions: Sequence[str], relevant: set[str], k: int) -> float:
    """
    Normalised discounted cumulative gain at ``k``.

    Binary relevance. Discounts gains logarithmically by rank, so moving
    a correct answer from position five to position one matters more
    than moving it from fifty to forty-six.
    """

    require_positive(k, name="k")

    if not relevant:
        return 0.0

    gain = sum(
        1.0 / math.log2(position + 1)
        for position, item in enumerate(predictions[:k], start=1)
        if item in relevant
    )

    ideal = sum(1.0 / math.log2(position + 1) for position in range(1, min(k, len(relevant)) + 1))

    return gain / ideal if ideal else 0.0


def accuracy(predicted: Sequence[object], expected: Sequence[object]) -> float:
    """Share of positions where prediction equals expectation."""

    if len(predicted) != len(expected):
        raise ValueError("predicted and expected must have the same length")

    if not predicted:
        return 0.0

    correct = sum(1 for left, right in zip(predicted, expected, strict=True) if left == right)

    return correct / len(predicted)


def pearson_correlation(first: Sequence[float], second: Sequence[float]) -> float:
    """
    Pearson correlation coefficient.

    Returns 0.0 when either input has zero variance, since correlation
    is undefined against a constant.
    """

    if len(first) != len(second):
        raise ValueError("inputs must have the same length")

    if len(first) < 2:
        return 0.0

    left = np.asarray(first, dtype=np.float64)

    right = np.asarray(second, dtype=np.float64)

    if left.std() == 0.0 or right.std() == 0.0:
        return 0.0

    return float(np.corrcoef(left, right)[0, 1])


def spearman_correlation(first: Sequence[float], second: Sequence[float]) -> float:
    """
    Spearman rank correlation.

    This is the standard measure for word similarity benchmarks: human
    similarity judgements are ordinal, so what matters is whether the
    model ranks pairs in the same order as annotators, not whether it
    reproduces their absolute numbers.

    Ties receive averaged ranks.
    """

    if len(first) != len(second):
        raise ValueError("inputs must have the same length")

    if len(first) < 2:
        return 0.0

    return pearson_correlation(_average_ranks(first), _average_ranks(second))


def _average_ranks(values: Sequence[float]) -> list[float]:
    """Rank values, assigning tied entries the average of their ranks."""

    ordered = sorted(range(len(values)), key=lambda index: values[index])

    ranks = [0.0] * len(values)

    position = 0

    while position < len(ordered):
        end = position

        while end + 1 < len(ordered) and values[ordered[end + 1]] == values[ordered[position]]:
            end += 1

        average = (position + end) / 2.0 + 1.0

        for index in range(position, end + 1):
            ranks[ordered[index]] = average

        position = end + 1

    return ranks


def _row_normalized(vectors: np.ndarray) -> np.ndarray:
    """L2 normalise rows, leaving zero rows as zero rather than NaN."""

    array = np.asarray(vectors, dtype=np.float64)

    if array.ndim == 1:
        array = array.reshape(1, -1)

    norms = np.linalg.norm(array, axis=1, keepdims=True)

    normalized: np.ndarray = array / np.maximum(norms, 1e-12)

    return normalized
