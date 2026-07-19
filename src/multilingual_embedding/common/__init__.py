"""
Shared infrastructure for the multilingual embedding framework.
"""

from __future__ import annotations

from .constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHARACTER_COVERAGE,
    DEFAULT_ENCODING,
    DEFAULT_RANDOM_SEED,
    DEFAULT_VOCAB_SIZE,
)
from .enums import (
    SpecialToken,
    TokenizerModel,
)
from .span import Span
from .types import (
    CorpusText,
    DocumentText,
    ParagraphText,
    SentenceText,
    TokenId,
    TokenIds,
)
from .version import __version__

__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_CHARACTER_COVERAGE",
    "DEFAULT_ENCODING",
    "DEFAULT_RANDOM_SEED",
    "DEFAULT_VOCAB_SIZE",
    "CorpusText",
    "DocumentText",
    "ParagraphText",
    "SentenceText",
    "Span",
    "SpecialToken",
    "TokenId",
    "TokenIds",
    "TokenizerModel",
    "__version__",
]
