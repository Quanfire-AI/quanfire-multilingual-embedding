from __future__ import annotations

import numpy as np
import pytest

from multilingual_embedding.core.exceptions import ValidationError
from multilingual_embedding.evaluation.metrics import (
    accuracy,
    average_precision,
    cosine_similarity,
    cosine_similarity_matrix,
    f1_score,
    mean_reciprocal_rank,
    ndcg_at_k,
    pearson_correlation,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    spearman_correlation,
)


class TestCosine:
    def test_identical_vectors(self) -> None:
        vector = np.array([1.0, 2.0, 3.0])

        assert cosine_similarity(vector, vector) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        assert cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)

    def test_opposite_vectors(self) -> None:
        assert cosine_similarity(np.array([1.0, 0.0]), np.array([-1.0, 0.0])) == pytest.approx(-1.0)

    def test_zero_vector_returns_zero_not_nan(self) -> None:
        """A zero vector has no direction; NaN would poison downstream means."""

        result = cosine_similarity(np.zeros(3), np.array([1.0, 2.0, 3.0]))

        assert result == 0.0

        assert not np.isnan(result)

    def test_matrix_shape_and_values(self) -> None:
        left = np.array([[1.0, 0.0], [0.0, 1.0]])

        similarities = cosine_similarity_matrix(left)

        assert similarities.shape == (2, 2)

        assert similarities[0, 0] == pytest.approx(1.0)

        assert similarities[0, 1] == pytest.approx(0.0)

    def test_matrix_against_other(self) -> None:
        left = np.array([[1.0, 0.0]])

        right = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])

        assert cosine_similarity_matrix(left, right).shape == (1, 3)

    def test_matrix_handles_zero_rows(self) -> None:
        vectors = np.array([[0.0, 0.0], [1.0, 0.0]])

        assert not np.isnan(cosine_similarity_matrix(vectors)).any()


class TestRanking:
    def test_precision_denominator_is_k(self) -> None:
        """
        Returning fewer than k results is a shortfall, not a free pass.
        """

        assert precision_at_k(["a"], {"a", "b"}, k=5) == pytest.approx(0.2)

    def test_precision_and_recall(self) -> None:
        predictions = ["a", "x", "b", "y"]

        relevant = {"a", "b", "c"}

        assert precision_at_k(predictions, relevant, k=4) == pytest.approx(0.5)

        assert recall_at_k(predictions, relevant, k=4) == pytest.approx(2 / 3)

    def test_empty_relevant_set_scores_zero(self) -> None:
        assert precision_at_k(["a"], set(), k=1) == 0.0

        assert recall_at_k(["a"], set(), k=1) == 0.0

        assert average_precision(["a"], set()) == 0.0

    def test_invalid_k_rejected(self) -> None:
        with pytest.raises(ValidationError):
            precision_at_k(["a"], {"a"}, k=0)

    def test_f1(self) -> None:
        assert f1_score(0.5, 0.5) == pytest.approx(0.5)

        assert f1_score(1.0, 0.0) == 0.0

        assert f1_score(0.0, 0.0) == 0.0

    def test_average_precision_rewards_early_hits(self) -> None:
        early = average_precision(["a", "b", "x", "y"], {"a", "b"})

        late = average_precision(["x", "y", "a", "b"], {"a", "b"})

        assert early > late

        assert early == pytest.approx(1.0)

    def test_reciprocal_rank(self) -> None:
        assert reciprocal_rank(["x", "a"], {"a"}) == pytest.approx(0.5)

        assert reciprocal_rank(["x", "y"], {"a"}) == 0.0

    def test_mean_reciprocal_rank(self) -> None:
        result = mean_reciprocal_rank([["a"], ["x", "b"]], [{"a"}, {"b"}])

        assert result == pytest.approx(0.75)

    def test_mrr_length_mismatch_rejected(self) -> None:
        with pytest.raises(ValueError):
            mean_reciprocal_rank([["a"]], [{"a"}, {"b"}])

    def test_mrr_of_empty(self) -> None:
        assert mean_reciprocal_rank([], []) == 0.0

    def test_ndcg_perfect_ranking_is_one(self) -> None:
        assert ndcg_at_k(["a", "b"], {"a", "b"}, k=2) == pytest.approx(1.0)

    def test_ndcg_rewards_earlier_placement(self) -> None:
        assert ndcg_at_k(["a", "x"], {"a"}, k=2) > ndcg_at_k(["x", "a"], {"a"}, k=2)


class TestAccuracyAndCorrelation:
    def test_accuracy(self) -> None:
        assert accuracy(["a", "b", "c"], ["a", "x", "c"]) == pytest.approx(2 / 3)

    def test_accuracy_length_mismatch_rejected(self) -> None:
        with pytest.raises(ValueError):
            accuracy(["a"], ["a", "b"])

    def test_accuracy_of_empty(self) -> None:
        assert accuracy([], []) == 0.0

    def test_pearson_perfect(self) -> None:
        assert pearson_correlation([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)

    def test_pearson_inverse(self) -> None:
        assert pearson_correlation([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)

    def test_pearson_constant_input_returns_zero(self) -> None:
        """Correlation against a constant is undefined, not 1.0."""

        assert pearson_correlation([1, 1, 1], [1, 2, 3]) == 0.0

    def test_spearman_is_monotonic_not_linear(self) -> None:
        """
        Rank correlation must be perfect for any monotonic relation.

        Pearson would not be, which is exactly why word similarity
        benchmarks are scored with Spearman.
        """

        assert spearman_correlation([1, 2, 3, 4], [1, 4, 9, 16]) == pytest.approx(1.0)

    def test_spearman_handles_ties(self) -> None:
        result = spearman_correlation([1, 2, 2, 3], [1, 2, 2, 3])

        assert result == pytest.approx(1.0)

    def test_spearman_matches_closed_form_reference(self) -> None:
        """
        Checked against the rank-difference formula.

        For untied ranks, rho = 1 - 6*sum(d^2) / (n*(n^2 - 1)). With
        first=[1,2,3,4,5] and second=[2,1,4,3,5] the rank differences are
        [-1,1,-1,1,0], so sum(d^2)=4 and rho = 1 - 24/120 = 0.8.
        """

        assert spearman_correlation([1, 2, 3, 4, 5], [2, 1, 4, 3, 5]) == pytest.approx(0.8)

    def test_too_short_input_returns_zero(self) -> None:
        assert spearman_correlation([1], [1]) == 0.0

        assert pearson_correlation([1], [1]) == 0.0

    def test_length_mismatch_rejected(self) -> None:
        with pytest.raises(ValueError):
            spearman_correlation([1, 2], [1])
