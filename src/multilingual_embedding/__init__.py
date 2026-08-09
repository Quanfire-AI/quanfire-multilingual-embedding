"""
Quanfire Multilingual Embedding Framework.

A layered pipeline from raw multilingual text to searchable embeddings::

    corpus -> tokenizer -> vocabulary -> embeddings -> evaluation

Quick start
-----------
::

    from multilingual_embedding import ExperimentConfig, TrainingPipeline

    config = ExperimentConfig(name="demo")

    config.corpus.source = Path("data/sample/corpus.jsonl")

    result = TrainingPipeline(config).run()

Only the most commonly used names are re-exported here. Everything else
is reached through its own package — ``multilingual_embedding.corpus``,
``.tokenizer``, ``.embedding``, ``.evaluation`` — which keeps this
module's import cost proportional to what a caller actually uses.
"""

from __future__ import annotations

from .common import __version__
from .config import (
    CorpusConfig,
    EmbeddingConfig,
    EvaluationConfig,
    ExperimentConfig,
    TokenizerConfig,
    load_config,
)
from .core import (
    ConfigurationError,
    MultilingualEmbeddingError,
    ValidationError,
    configure_logging,
    get_logger,
)
from .corpus import Corpus, Document, Paragraph, Sentence, Token
from .pipelines import SemanticSearchPipeline, TrainingPipeline, TrainingResult
from .vocabulary import Vocabulary

__all__ = [
    "ConfigurationError",
    "Corpus",
    "CorpusConfig",
    "Document",
    "EmbeddingConfig",
    "EvaluationConfig",
    "ExperimentConfig",
    "MultilingualEmbeddingError",
    "Paragraph",
    "SemanticSearchPipeline",
    "Sentence",
    "Token",
    "TokenizerConfig",
    "TrainingPipeline",
    "TrainingResult",
    "ValidationError",
    "Vocabulary",
    "__version__",
    "configure_logging",
    "get_logger",
    "load_config",
]
