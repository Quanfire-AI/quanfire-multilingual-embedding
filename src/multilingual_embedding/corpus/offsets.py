"""
Offset arithmetic for nested text spans.

Every node stores its span relative to its immediate parent's text. That
keeps segmentation local and composable: a paragraph can be re-segmented
without touching the document around it.

The cost is that mapping a sentence back to a position in the original
document requires walking the chain of parents. These helpers do that,
and verify that child spans stay inside their parent.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from multilingual_embedding.common.span import Span
from multilingual_embedding.core.exceptions import ValidationError

__all__ = [
    "invert_spans",
    "merge_overlapping",
    "resolve_chain",
    "spans_are_ordered",
    "spans_within",
    "to_absolute",
]


def to_absolute(span: Span, parent_offset: int) -> Span:
    """Shift a relative span into its parent's coordinate system."""

    return span.shift(parent_offset)


def resolve_chain(spans: Sequence[Span]) -> Span:
    """
    Collapse a root-to-leaf chain of relative spans into one absolute span.

    Parameters
    ----------
    spans:
        Spans ordered outermost first, each relative to the previous.

    Example
    -------
    A sentence at ``[10, 20)`` of a paragraph that starts at ``[100, ...)``
    of a document resolves to ``[110, 120)``::

        resolve_chain([Span(100, 400), Span(10, 20)])  -> Span(110, 120)
    """

    if not spans:
        raise ValidationError("Cannot resolve an empty span chain")

    offset = 0

    current = spans[0]

    for child in spans[1:]:
        offset += current.start

        current = child

    return current.shift(offset)


def spans_are_ordered(spans: Iterable[Span]) -> bool:
    """
    Return True if spans are sorted and non-overlapping.

    Segmentation output should always satisfy this; a violation means a
    segmenter emitted overlapping or out-of-order units.
    """

    previous_end = 0

    for span in spans:
        if span.start < previous_end:
            return False

        previous_end = span.end

    return True


def spans_within(spans: Iterable[Span], *, length: int) -> bool:
    """Return True if every span lies inside ``[0, length)``."""

    return all(span.start >= 0 and span.end <= length for span in spans)


def invert_spans(spans: Sequence[Span], *, length: int) -> list[Span]:
    """
    Return the gaps between ``spans`` across ``[0, length)``.

    Used to recover the material segmentation discarded — the whitespace
    and punctuation between sentences — which is what makes it possible
    to reconstruct the original text exactly.
    """

    gaps: list[Span] = []

    cursor = 0

    for span in sorted(spans, key=lambda item: item.start):
        if span.start > cursor:
            gaps.append(Span(cursor, span.start))

        cursor = max(cursor, span.end)

    if cursor < length:
        gaps.append(Span(cursor, length))

    return gaps


def merge_overlapping(spans: Iterable[Span]) -> list[Span]:
    """Merge overlapping or touching spans into a minimal covering set."""

    ordered = sorted(spans, key=lambda span: (span.start, span.end))

    if not ordered:
        return []

    merged = [ordered[0]]

    for span in ordered[1:]:
        last = merged[-1]

        if last.overlaps(span) or last.touches(span):
            merged[-1] = last.merge(span)
        else:
            merged.append(span)

    return merged
