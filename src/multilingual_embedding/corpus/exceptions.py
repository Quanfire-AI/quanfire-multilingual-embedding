"""
Corpus specific errors.
"""

from __future__ import annotations

from multilingual_embedding.core.exceptions import MultilingualEmbeddingError

__all__ = [
    "CorpusError",
    "CorpusFormatError",
    "EmptyCorpusError",
    "SegmentationError",
]


class CorpusError(MultilingualEmbeddingError):
    """Base class for corpus layer failures."""


class CorpusFormatError(CorpusError):
    """Raised when a source file does not match the expected layout."""


class SegmentationError(CorpusError):
    """Raised when text cannot be segmented into the requested units."""


class EmptyCorpusError(CorpusError):
    """Raised when an operation requires at least one document."""
