"""
Metadata associated with a corpus.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .base import BaseMetadata

__all__ = ["CorpusMetadata"]


@dataclass(slots=True)
class CorpusMetadata:
    """
    Metadata describing a corpus.

    Attributes
    ----------
    base:
        Shared metadata (identifier, language, script, provenance).

    dataset_name, version, description:
        Dataset identity. ``version`` should change whenever the
        underlying text changes, so that a model can be traced to the
        exact corpus revision it was trained on.
    """

    base: BaseMetadata = field(default_factory=BaseMetadata)

    dataset_name: str | None = None

    version: str | None = None

    description: str | None = None
