"""
Surface token.

A :class:`Token` is a stretch of a sentence's text together with its
position. It is the boundary object between the corpus layer and the
tokenizer layer: pre-tokenizers emit tokens, and subword models consume
them.

Note the distinction from a *vocabulary* entry. A Token is an occurrence
in text and knows where it came from; a vocabulary entry is a type with
an integer id and a corpus frequency.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from multilingual_embedding.common.span import Span
from multilingual_embedding.corpus.base.text_node import TextNode
from multilingual_embedding.corpus.metadata.token import TokenMetadata

from .script import Script, detect_script

__all__ = ["Token"]


@dataclass(slots=True)
class Token(TextNode[TokenMetadata]):
    """
    A single token occurrence.

    Attributes
    ----------
    token_id:
        Vocabulary id, assigned once a vocabulary exists. ``None`` for a
        token that has been segmented but not yet mapped.
    """

    metadata: TokenMetadata = field(default_factory=TokenMetadata)

    token_id: int | None = None

    @classmethod
    def create(
        cls,
        text: str,
        *,
        start: int = 0,
        token_id: int | None = None,
        metadata: TokenMetadata | None = None,
    ) -> Token:
        """
        Build a token whose span is derived from its own length.

        Convenient when constructing tokens directly in tests or when the
        producer already knows the start offset.
        """

        return cls(
            text=text,
            span=Span(start, start + len(text)),
            metadata=metadata if metadata is not None else TokenMetadata(),
            token_id=token_id,
        )

    @property
    def script(self) -> Script:
        """Dominant script of this token."""

        return detect_script(self.text).dominant

    @property
    def is_punctuation(self) -> bool:
        """
        True when the token carries no script-bearing characters.

        Punctuation and digits are frequently excluded from embedding
        training, so this is checked often enough to belong here.
        """

        return detect_script(self.text).dominant is Script.UNKNOWN or not any(
            character.isalnum() for character in self.text
        )
