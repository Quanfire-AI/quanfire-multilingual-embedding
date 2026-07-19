"""
Metadata associated with a token.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .base import BaseMetadata

__all__ = ["TokenMetadata"]


@dataclass(slots=True)
class TokenMetadata:
    """
    Metadata describing a token.

    Attributes
    ----------
    base:
        Shared metadata (identifier, language, script, provenance).

    lemma, part_of_speech:
        Optional linguistic annotations, populated only if an external
        analyser supplied them.

    frequency:
        Corpus frequency, filled in once vocabulary counts are known.

    is_stopword:
        Whether the token was filtered as a stopword.
    """

    base: BaseMetadata = field(default_factory=BaseMetadata)

    lemma: str | None = None

    part_of_speech: str | None = None

    frequency: int | None = None

    is_stopword: bool = False
