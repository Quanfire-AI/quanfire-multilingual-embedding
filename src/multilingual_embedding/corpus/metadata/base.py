"""
Base metadata shared by all corpus objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["BaseMetadata"]


@dataclass(slots=True)
class BaseMetadata:
    """
    Common metadata for corpus objects.

    Attributes
    ----------
    id:
        Optional unique identifier.

    language:
        ISO 639 language code (e.g. "en", "hi").

    script:
        ISO 15924 script code (e.g. "Latn", "Deva").

    source:
        Origin of the data.

    attributes:
        Arbitrary custom metadata.
    """

    id: str | None = None

    language: str | None = None

    script: str | None = None

    source: str | None = None

    attributes: dict[str, Any] = field(default_factory=dict)

    # There were `created_at` and `updated_at` fields here, each calling
    # `datetime.now(UTC)` from a default factory — so twice per document,
    # a quarter of a million clock calls for one Wikipedia extraction.
    # Nothing read them and nothing serialised them.
    #
    # They also failed a real run. On CPython 3.12.3 under WSL,
    # `datetime.now(UTC)` raised `SystemError: attempting to create
    # PyCMethod with a METH_METHOD flag but no class`, which took the
    # corpus audit down partway through 118,571 Hindi articles.
    #
    # A field nobody reads should not be able to fail a pipeline. This is
    # the third populated-but-unread field removed from this project; the
    # others were EmbeddingConfig.batch_size and ComputeConfig.workers.
