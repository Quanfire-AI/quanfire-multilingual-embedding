"""
Metadata associated with a document.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .base import BaseMetadata

__all__ = ["DocumentMetadata"]


@dataclass(slots=True)
class DocumentMetadata:
    """
    Metadata describing a document.

    Attributes
    ----------
    base:
        Shared metadata (identifier, language, script, provenance).

    title, author, url, license:
        Bibliographic provenance. ``license`` in particular should be
        carried through the pipeline, since a trained model inherits the
        licensing constraints of the text it was trained on.
    """

    base: BaseMetadata = field(default_factory=BaseMetadata)

    title: str | None = None

    author: str | None = None

    url: str | None = None

    license: str | None = None
