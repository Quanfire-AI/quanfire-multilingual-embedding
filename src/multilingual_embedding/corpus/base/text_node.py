"""
Base class for all text nodes.

Every textual object in the framework derives from TextNode:

Token
Sentence
Paragraph
Document

A node owns three things: the text it covers, the span locating that
text inside its parent, and its metadata.

Spans are relative to the immediate parent, not to the document root.
That keeps segmentation local — a paragraph can be re-segmented without
renumbering everything after it — at the cost of needing
:mod:`multilingual_embedding.corpus.offsets` to recover absolute
positions.

The class is generic in its metadata type so that ``sentence.metadata``
is known to be a ``SentenceMetadata`` rather than a bare ``BaseMetadata``.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass

from multilingual_embedding.common.span import Span
from multilingual_embedding.corpus.metadata.base import BaseMetadata

__all__ = ["TextNode"]


@dataclass(slots=True)
class TextNode[MetadataT](ABC):
    """
    Base class for all textual objects.

    Attributes
    ----------
    text:
        The text this node covers.

    span:
        Position of ``text`` within the parent node's text.

    metadata:
        Metadata for this node.
    """

    text: str

    span: Span

    metadata: MetadataT

    def __len__(self) -> int:
        """
        Number of characters.
        """
        return len(self.text)

    @property
    def character_count(self) -> int:
        """
        Number of Unicode characters.
        """
        return len(self.text)

    @property
    def is_empty(self) -> bool:
        """
        Returns True if text is empty.
        """
        return len(self.text) == 0

    @property
    def is_blank(self) -> bool:
        """
        Returns True if text is empty or only whitespace.
        """
        return not self.text.strip()

    @property
    def base_metadata(self) -> BaseMetadata | None:
        """
        Shared metadata, whether held directly or wrapped.

        Metadata classes for the concrete node types hold a
        ``BaseMetadata`` under ``.base``, while a plain node may hold one
        directly. This resolves either shape so that callers reading
        common fields do not need to know which.
        """

        metadata = self.metadata

        if isinstance(metadata, BaseMetadata):
            return metadata

        nested = getattr(metadata, "base", None)

        return nested if isinstance(nested, BaseMetadata) else None

    @property
    def language(self) -> str | None:
        """
        Language code from metadata, if set.
        """

        base = self.base_metadata

        return base.language if base is not None else None

    def strip(self) -> str:
        """
        Returns stripped text.
        """
        return self.text.strip()

    def startswith(self, prefix: str) -> bool:
        return self.text.startswith(prefix)

    def endswith(self, suffix: str) -> bool:
        return self.text.endswith(suffix)

    def contains(self, value: str) -> bool:
        return value in self.text

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        preview = self.text

        if len(preview) > 40:
            preview = preview[:37] + "..."

        return f"{self.__class__.__name__}(length={len(self)}, text={preview!r})"
