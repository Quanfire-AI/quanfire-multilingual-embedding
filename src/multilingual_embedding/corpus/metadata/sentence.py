"""
Metadata associated with a sentence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .base import BaseMetadata

__all__ = ["SentenceMetadata"]


@dataclass(slots=True)
class SentenceMetadata:
    """
    Metadata describing a sentence.

    Attributes
    ----------
    base:
        Shared metadata (identifier, language, script, provenance).

    sentiment:
        Optional sentiment score, if a downstream analyser assigned one.

    language_confidence:
        Confidence in ``base.language`` when it was inferred rather than
        declared. ``None`` means the language was given, not guessed.
    """

    base: BaseMetadata = field(default_factory=BaseMetadata)

    sentiment: float | None = None

    language_confidence: float | None = None
