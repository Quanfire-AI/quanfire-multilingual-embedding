"""
Metadata associated with a paragraph.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .base import BaseMetadata

__all__ = ["ParagraphMetadata"]


@dataclass(slots=True)
class ParagraphMetadata:
    """
    Metadata describing a paragraph.

    Attributes
    ----------
    base:
        Shared metadata (identifier, language, script, provenance).

    paragraph_index:
        Zero based position within the parent document.
    """

    base: BaseMetadata = field(default_factory=BaseMetadata)

    paragraph_index: int | None = None
