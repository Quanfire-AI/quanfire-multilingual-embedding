"""
Embedding layer: word vectors, sentence vectors and search over them.

The layer is built bottom up. :class:`Word2Vec` learns word vectors from
a corpus and hands back an :class:`EmbeddingMatrix`, which binds those
vectors to the vocabulary that indexes them. The sentence encoders
compose word vectors into sentence vectors, and :class:`SimilarityIndex`
searches a set of those.

:class:`EmbeddingModel` is the contract a future static model would
implement. Contextual encoders need a broader abstraction; see ROADMAP.md.

:func:`mine_negatives` is here rather than in ``corpus`` because finding
a passage a model confuses with the right answer requires a model, and
``corpus`` sits below every encoder. It is torch-free like the rest of
this module: it takes any :class:`TextEncoder`, so whoever built the
encoder decided which training stack to install.
"""

from __future__ import annotations

from .base import EmbeddingModel
from .encoder import TextEncoder, encoder_dimension
from .index import SearchResult, SimilarityIndex
from .matrix import EmbeddingMatrix
from .negatives import (
    AuditRecord,
    NegativeConfig,
    NegativeStatistics,
    mine_negatives,
)
from .sentence import (
    SENTENCE_ENCODERS,
    MeanPoolingEncoder,
    SentenceEncoder,
    SifEncoder,
)
from .word2vec import Word2Vec

__all__ = [
    "SENTENCE_ENCODERS",
    "AuditRecord",
    "EmbeddingMatrix",
    "EmbeddingModel",
    "MeanPoolingEncoder",
    "NegativeConfig",
    "NegativeStatistics",
    "SearchResult",
    "SentenceEncoder",
    "SifEncoder",
    "SimilarityIndex",
    "TextEncoder",
    "Word2Vec",
    "encoder_dimension",
    "mine_negatives",
]
