"""
Base class for text nodes holding ordered children.

A container keeps both its own text and its children. Storing the text
rather than deriving it from the children is deliberate: the material
*between* children — whitespace, punctuation, markup — is part of the
source and would be lost if the text were reconstructed by joining.
Keeping it means a document survives a round trip through segmentation
unchanged.

:meth:`ContainerNode.verify_children` checks that the two views agree.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from multilingual_embedding.corpus.exceptions import CorpusError

from .text_node import TextNode

__all__ = ["ContainerNode"]


@dataclass(slots=True)
class ContainerNode[MetadataT, ChildT: TextNode[Any]](TextNode[MetadataT]):
    """
    A text node composed of ordered child nodes.

    Attributes
    ----------
    children:
        Child nodes in document order. Each child's span is relative to
        this node's ``text``.
    """

    children: list[ChildT] = field(default_factory=list)

    def __iter__(self) -> Iterator[ChildT]:
        return iter(self.children)

    def __getitem__(self, index: int) -> ChildT:
        return self.children[index]

    @property
    def child_count(self) -> int:
        """Number of direct children."""

        return len(self.children)

    @property
    def is_leaf(self) -> bool:
        """True when this node has no children."""

        return not self.children

    def add(self, child: ChildT) -> ChildT:
        """Append a child and return it."""

        self.children.append(child)

        return child

    def extend(self, children: Sequence[ChildT]) -> None:
        """Append several children in order."""

        self.children.extend(children)

    def child_texts(self) -> list[str]:
        """Return each child's text."""

        return [child.text for child in self.children]

    def verify_children(self) -> None:
        """
        Check that child spans are consistent with this node's text.

        Verifies that every child lies within bounds, that children are
        ordered and non-overlapping, and that each child's ``text``
        matches the slice its span designates.

        Raises
        ------
        CorpusError
            On the first inconsistency found, naming the child index.
        """

        length = len(self.text)

        previous_end = 0

        for index, child in enumerate(self.children):
            if child.span.start < 0 or child.span.end > length:
                raise CorpusError(
                    "Child span falls outside parent text",
                    node=type(self).__name__,
                    child_index=index,
                    span=(child.span.start, child.span.end),
                    parent_length=length,
                )

            if child.span.start < previous_end:
                raise CorpusError(
                    "Child spans overlap or are out of order",
                    node=type(self).__name__,
                    child_index=index,
                    span_start=child.span.start,
                    previous_end=previous_end,
                )

            expected = self.text[child.span.start : child.span.end]

            if child.text != expected:
                raise CorpusError(
                    "Child text does not match the slice its span designates",
                    node=type(self).__name__,
                    child_index=index,
                    expected=expected[:40],
                    actual=child.text[:40],
                )

            previous_end = child.span.end
