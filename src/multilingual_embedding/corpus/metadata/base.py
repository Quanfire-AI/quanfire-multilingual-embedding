"""
Base metadata shared by all corpus objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
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

    created_at:
        Creation timestamp (UTC).

    updated_at:
        Last update timestamp (UTC).

    attributes:
        Arbitrary custom metadata.
    """

    id: str | None = None

    language: str | None = None

    script: str | None = None

    source: str | None = None

    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    attributes: dict[str, Any] = field(default_factory=dict)
