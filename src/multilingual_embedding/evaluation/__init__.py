"""
Evaluation layer: metrics, tokenizer scoring, embedding scoring, reports.

The metric primitives in :mod:`.metrics` know nothing about the rest of
the framework, and the tokenizer evaluator takes a plain callable rather
than a Tokenizer object, so any segmentation function can be scored
against a baseline without adapters.
"""

from __future__ import annotations

from .embedding_eval import (
    AnalogyQuestion,
    EmbeddingEvaluator,
    EmbeddingMetrics,
    SimilarityPair,
    load_analogy_dataset,
    load_similarity_dataset,
    sample_neighbour_probes,
)
from .metrics import (
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
from .report import EvaluationReport
from .retrieval import (
    RetrievalReport,
    RetrievalScores,
    evaluate_retrieval,
)
from .tokenizer_eval import (
    TokenizerEvaluator,
    TokenizerMetrics,
    evaluate_tokenizer,
    language_fairness,
)

__all__ = [
    "AnalogyQuestion",
    "EmbeddingEvaluator",
    "EmbeddingMetrics",
    "EvaluationReport",
    "RetrievalReport",
    "RetrievalScores",
    "SimilarityPair",
    "TokenizerEvaluator",
    "TokenizerMetrics",
    "accuracy",
    "average_precision",
    "cosine_similarity",
    "cosine_similarity_matrix",
    "evaluate_retrieval",
    "evaluate_tokenizer",
    "f1_score",
    "language_fairness",
    "load_analogy_dataset",
    "load_similarity_dataset",
    "mean_reciprocal_rank",
    "ndcg_at_k",
    "pearson_correlation",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "sample_neighbour_probes",
    "spearman_correlation",
]
