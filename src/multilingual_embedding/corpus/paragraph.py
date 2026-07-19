"""
Paragraph node.

A paragraph groups sentences. It exists as a level in its own right
because paragraph boundaries carry discourse information that is lost
when a document is flattened straight to sentences.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from multilingual_embedding.common.span import Span
from multilingual_embedding.corpus.base.container_node import ContainerNode
from multilingual_embedding.corpus.metadata.paragraph import ParagraphMetadata

from .segmentation import split_sentences
from .sentence import Sentence

__all__ = ["Paragraph"]


@dataclass(slots=True)
class Paragraph(ContainerNode[ParagraphMetadata, Sentence]):
    """
    A paragraph and its sentences.

    Children are :class:`~multilingual_embedding.corpus.sentence.Sentence`
    instances whose spans are relative to this paragraph's text.
    """

    metadata: ParagraphMetadata = field(default_factory=ParagraphMetadata)

    @classmethod
    def from_text(
        cls,
        text: str,
        *,
        start: int = 0,
        language: str | None = None,
        index: int | None = None,
        segment: bool = True,
    ) -> Paragraph:
        """
        Build a paragraph, optionally segmenting it into sentences.

        Parameters
        ----------
        text:
            Paragraph text.

        start:
            Offset within the parent document.

        language:
            Language code applied to the paragraph and its sentences.

        index:
            Position within the parent document.

        segment:
            When False the paragraph is created with no sentences. Use
            this for input that is already one sentence per record, so
            that segmentation does not second-guess the source.
        """

        metadata = ParagraphMetadata()

        metadata.base.language = language

        metadata.paragraph_index = index

        paragraph = cls(
            text=text,
            span=Span(start, start + len(text)),
            metadata=metadata,
        )

        if segment:
            for span in split_sentences(text, language=language):
                paragraph.add(
                    Sentence.create(
                        text=span.slice(text),
                        start=span.start,
                        language=language,
                    )
                )
        elif text.strip():
            paragraph.add(Sentence.create(text=text, start=0, language=language))

        return paragraph

    @property
    def sentences(self) -> list[Sentence]:
        """Sentences in this paragraph."""

        return self.children

    @property
    def sentence_count(self) -> int:
        """Number of sentences in this paragraph."""

        return len(self.children)

    def token_count(self) -> int:
        """Total tokens across this paragraph's sentences."""

        return sum(sentence.token_count for sentence in self.children)

    def to_dict(self) -> dict[str, Any]:
        """Reduce to primitives."""

        payload: dict[str, Any] = {
            "text": self.text,
            "start": self.span.start,
            "sentences": [sentence.to_dict() for sentence in self.children],
        }

        if self.metadata.paragraph_index is not None:
            payload["index"] = self.metadata.paragraph_index

        if self.metadata.base.language:
            payload["language"] = self.metadata.base.language

        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Paragraph:
        """Rebuild a paragraph from :meth:`to_dict` output."""

        paragraph = cls.from_text(
            text=data["text"],
            start=int(data.get("start", 0)),
            language=data.get("language"),
            index=data.get("index"),
            segment=False,
        )

        paragraph.children = [
            Sentence.from_dict(sentence_data) for sentence_data in data.get("sentences", [])
        ]

        return paragraph
