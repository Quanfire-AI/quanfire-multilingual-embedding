"""
Character span.

Represents a half-open interval:

[start, end)

where end is exclusive.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Span"]


@dataclass(slots=True, frozen=True)
class Span:
    """
    Character span.

    Examples
    --------
    text = "Hello world"

    Span(0, 5) -> "Hello"

    Span(6, 11) -> "world"
    """

    start: int

    end: int

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError("start must be >= 0")

        if self.end < self.start:
            raise ValueError("end must be >= start")

    @property
    def length(self) -> int:
        """Length of the span."""

        return self.end - self.start

    def slice(self, text: str) -> str:
        """Extract text covered by this span."""

        return text[self.start : self.end]

    def contains(self, index: int) -> bool:
        """Return True if index lies inside the span."""

        return self.start <= index < self.end

    def overlaps(self, other: Span) -> bool:
        """Return True if spans overlap."""

        return self.start < other.end and other.start < self.end

    def touches(self, other: Span) -> bool:
        """
        Return True if spans touch.

        Example

        [0,5)
             [5,10)

        """
        return self.end == other.start or other.end == self.start

    def merge(self, other: Span) -> Span:
        """
        Merge two overlapping/touching spans.
        """

        if not (self.overlaps(other) or self.touches(other)):
            raise ValueError("Cannot merge non-adjacent spans.")

        return Span(
            min(self.start, other.start),
            max(self.end, other.end),
        )

    def shift(self, offset: int) -> Span:
        """
        Shift span.
        """

        return Span(
            self.start + offset,
            self.end + offset,
        )
