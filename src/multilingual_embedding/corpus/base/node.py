"""
Structural protocols shared by corpus nodes.

These describe shape rather than inheritance, so that a caller can accept
"anything with text and a span" without importing the concrete classes.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from multilingual_embedding.common.span import Span

__all__ = ["Composite", "Spanned"]


@runtime_checkable
class Spanned(Protocol):
    """Anything carrying text and a position within a parent."""

    text: str

    span: Span


@runtime_checkable
class Composite(Protocol):
    """Anything carrying ordered child nodes."""

    @property
    def children(self) -> Sequence[Spanned]:  # pragma: no cover - protocol
        ...
