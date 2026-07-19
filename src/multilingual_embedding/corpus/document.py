"""
Document node.

A document is the unit of provenance. Licence, author and source URL
attach here, and a document is the granularity at which corpora are
split into train and evaluation sets — splitting at sentence level would
leak content from the same document across the boundary.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from multilingual_embedding.common.span import Span
from multilingual_embedding.corpus.base.container_node import ContainerNode
from multilingual_embedding.corpus.metadata.document import DocumentMetadata

from .language import infer_language
from .paragraph import Paragraph
from .script import Script, detect_script
from .segmentation import split_paragraphs
from .sentence import Sentence
from .token import Token

__all__ = ["Document"]


@dataclass(slots=True)
class Document(ContainerNode[DocumentMetadata, Paragraph]):
    """
    A document and its paragraphs.

    Children are
    :class:`~multilingual_embedding.corpus.paragraph.Paragraph` instances
    whose spans are relative to this document's text.
    """

    metadata: DocumentMetadata = field(default_factory=DocumentMetadata)

    @classmethod
    def from_text(
        cls,
        text: str,
        *,
        identifier: str | None = None,
        language: str | None = None,
        source: str | None = None,
        title: str | None = None,
        segment: bool = True,
        detect_language: bool = True,
    ) -> Document:
        """
        Build a document from raw text.

        Parameters
        ----------
        text:
            Full document text.

        identifier:
            Stable document id. Callers that do not supply one should
            derive it from content — see
            :func:`multilingual_embedding.utils.hashing.hash_text` — so
            that the same text yields the same id across runs.

        language:
            Language code. When omitted and ``detect_language`` is set,
            it is inferred from the dominant script; that inference
            returns None for scripts shared by several languages.

        source:
            Origin of the text, typically a file path or URL.

        title:
            Document title, if known.

        segment:
            Whether to split into paragraphs and sentences. Pass False
            for input that is already segmented.

        detect_language:
            Whether to infer the language when none is supplied.
        """

        resolved_language = language

        if resolved_language is None and detect_language:
            resolved_language = infer_language(text)

        metadata = DocumentMetadata()

        metadata.base.id = identifier

        metadata.base.language = resolved_language

        metadata.base.source = source

        metadata.base.script = detect_script(text).dominant.value if text.strip() else None

        metadata.title = title

        document = cls(
            text=text,
            span=Span(0, len(text)),
            metadata=metadata,
        )

        if not segment:
            return document

        for index, span in enumerate(split_paragraphs(text)):
            document.add(
                Paragraph.from_text(
                    text=span.slice(text),
                    start=span.start,
                    language=resolved_language,
                    index=index,
                )
            )

        return document

    @property
    def paragraphs(self) -> list[Paragraph]:
        """Paragraphs in this document."""

        return self.children

    @property
    def identifier(self) -> str | None:
        """Stable document id, if assigned."""

        return self.metadata.base.id

    @property
    def script(self) -> Script:
        """Dominant script of this document."""

        return detect_script(self.text).dominant

    def sentences(self) -> Iterator[Sentence]:
        """Iterate every sentence in document order."""

        for paragraph in self.children:
            yield from paragraph.sentences

    def tokens(self) -> Iterator[Token]:
        """Iterate every token in document order."""

        for sentence in self.sentences():
            yield from sentence.tokens

    @property
    def paragraph_count(self) -> int:
        """Number of paragraphs."""

        return len(self.children)

    @property
    def sentence_count(self) -> int:
        """Number of sentences across all paragraphs."""

        return sum(paragraph.sentence_count for paragraph in self.children)

    def token_count(self) -> int:
        """Number of tokens across all sentences."""

        return sum(paragraph.token_count() for paragraph in self.children)

    def verify(self) -> None:
        """
        Recursively check that spans agree with text at every level.

        Raises
        ------
        CorpusError
            On the first inconsistency found.
        """

        self.verify_children()

        for paragraph in self.children:
            paragraph.verify_children()

            for sentence in paragraph.sentences:
                sentence.verify_children()

    def to_dict(self) -> dict[str, Any]:
        """Reduce to primitives for JSON Lines persistence."""

        base = self.metadata.base

        payload: dict[str, Any] = {
            "text": self.text,
            "paragraphs": [paragraph.to_dict() for paragraph in self.children],
        }

        for key, value in (
            ("id", base.id),
            ("language", base.language),
            ("script", base.script),
            ("source", base.source),
            ("title", self.metadata.title),
            ("author", self.metadata.author),
            ("url", self.metadata.url),
            ("license", self.metadata.license),
        ):
            if value:
                payload[key] = value

        if base.attributes:
            payload["attributes"] = dict(base.attributes)

        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Document:
        """Rebuild a document from :meth:`to_dict` output."""

        document = cls.from_text(
            text=data["text"],
            identifier=data.get("id"),
            language=data.get("language"),
            source=data.get("source"),
            title=data.get("title"),
            segment=False,
            detect_language=False,
        )

        document.metadata.base.script = data.get("script")

        document.metadata.author = data.get("author")

        document.metadata.url = data.get("url")

        document.metadata.license = data.get("license")

        document.metadata.base.attributes = dict(data.get("attributes", {}))

        document.children = [
            Paragraph.from_dict(paragraph_data) for paragraph_data in data.get("paragraphs", [])
        ]

        return document
