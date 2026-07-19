"""
Embedding layer: word vectors, sentence vectors and search over them.

The layer is built bottom up. :class:`Word2Vec` learns word vectors from
a corpus and hands back an :class:`EmbeddingMatrix`, which binds those
vectors to the vocabulary that indexes them. The sentence encoders
compose word vectors into sentence vectors, and :class:`SimilarityIndex`
searches a set of those.

:class:`EmbeddingModel` is the contract a future static model would
implement. Contextual encoders need a broader abstraction; see ROADMAP.md.
"""

from __future__ import annotations

from .base import EmbeddingModel
from .encoder import TextEncoder, encoder_dimension
from .index import SearchResult, SimilarityIndex
from .matrix import EmbeddingMatrix
from .sentence import (
    SENTENCE_ENCODERS,
    MeanPoolingEncoder,
    SentenceEncoder,
    SifEncoder,
)
from .word2vec import Word2Vec

__all__ = [
    "SENTENCE_ENCODERS",
    "EmbeddingMatrix",
    "EmbeddingModel",
    "MeanPoolingEncoder",
    "SearchResult",
    "SentenceEncoder",
    "SifEncoder",
    "SimilarityIndex",
    "TextEncoder",
    "Word2Vec",
    "encoder_dimension",
]
