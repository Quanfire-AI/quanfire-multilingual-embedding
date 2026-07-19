"""
Sentence node.

A sentence is the unit the tokenizer trains on and the unit an embedding
model treats as a context window. It holds its surface tokens once a
pre-tokenizer has run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from multilingual_embedding.common.span import Span
from multilingual_embedding.corpus.base.container_node import ContainerNode
from multilingual_embedding.corpus.metadata.sentence import SentenceMetadata

from .script import Script, detect_script
from .segmentation import split_words
from .token import Token

__all__ = ["Sentence"]


@dataclass(slots=True)
class Sentence(ContainerNode[SentenceMetadata, Token]):
    """
    A single sentence and its tokens.

    Children are :class:`~multilingual_embedding.corpus.token.Token`
    instances whose spans are relative to this sentence's text.
    """

    metadata: SentenceMetadata = field(default_factory=SentenceMetadata)

    @classmethod
    def create(
        cls,
        text: str,
        *,
        start: int = 0,
        language: str | None = None,
        metadata: SentenceMetadata | None = None,
    ) -> Sentence:
        """
        Build a sentence, deriving its span from its own length.

        Parameters
        ----------
        text:
            Sentence text.

        start:
            Offset within the parent paragraph.

        language:
            Language code recorded in metadata. Overrides any value
            already present in ``metadata``.
        """

        resolved = metadata if metadata is not None else SentenceMetadata()

        if language is not None:
            resolved.base.language = language

        return cls(
            text=text,
            span=Span(start, start + len(text)),
            metadata=resolved,
        )

    @property
    def tokens(self) -> list[Token]:
        """Surface tokens, empty until a pre-tokenizer has run."""

        return self.children

    @property
    def token_count(self) -> int:
        """Number of tokens attached to this sentence."""

        return len(self.children)

    @property
    def script(self) -> Script:
        """Dominant script of this sentence."""

        return detect_script(self.text).dominant

    def word_count(self) -> int:
        """
        Approximate word count.

        Counts attached tokens when present, otherwise falls back to
        Unicode word matching. The fallback is unreliable for scripts
        without whitespace word boundaries; :attr:`character_count` is
        the comparable measure across scripts.
        """

        if self.children:
            return len(self.children)

        return len(split_words(self.text))

    def set_tokens(self, tokens: list[Token]) -> None:
        """Replace this sentence's tokens."""

        self.children = tokens

    def to_dict(self) -> dict[str, Any]:
        """Reduce to primitives, omitting empty token lists."""

        payload: dict[str, Any] = {
            "text": self.text,
            "start": self.span.start,
        }

        if self.metadata.base.language:
            payload["language"] = self.metadata.base.language

        if self.children:
            payload["tokens"] = [
                {"text": token.text, "start": token.span.start} for token in self.children
            ]

        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Sentence:
        """Rebuild a sentence from :meth:`to_dict` output."""

        sentence = cls.create(
            text=data["text"],
            start=int(data.get("start", 0)),
            language=data.get("language"),
        )

        for token_data in data.get("tokens", []):
            sentence.add(
                Token.create(
                    text=token_data["text"],
                    start=int(token_data.get("start", 0)),
                )
            )

        return sentence

    def with_language(self, language: str) -> Sentence:
        """Return this sentence with its language set, for chaining."""

        self.metadata.base.language = language

        return self
