"""
The result of tokenizing one piece of text.

:class:`Encoding` is the object that crosses the boundary from the
tokenizer into model code. It deliberately carries the surface pieces
and their spans alongside the ids: without them a prediction over token
positions cannot be mapped back onto the characters that produced it,
and debugging a tokenizer becomes guesswork.

Truncation and padding return new instances rather than mutating in
place, because the same encoding is frequently reused across batches
with different length budgets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from multilingual_embedding.common.span import Span
from multilingual_embedding.core.exceptions import ValidationError
from multilingual_embedding.utils.validation import (
    require_non_negative,
    require_same_length,
)

__all__ = ["Encoding"]


@dataclass(slots=True)
class Encoding:
    """
    Ids, surface pieces and optional alignment for one text.

    Attributes
    ----------
    ids:
        Vocabulary ids, one per token.

    tokens:
        Surface pieces, one per id. Same length as ``ids``.

    spans:
        Character spans into the text that was tokenized, when the
        tokenizer can supply them. ``None`` for tokenizers that cannot,
        rather than a list of fabricated positions.

    attention_mask:
        ``1`` for real tokens and ``0`` for padding. ``None`` until the
        encoding is padded.

    Example
    -------
    ::

        encoding = tokenizer.encode("hello world")

        encoding.pad_to(8, pad_id=0).attention_mask
        -> [1, 1, 0, 0, 0, 0, 0, 0]
    """

    ids: list[int]

    tokens: list[str]

    spans: list[Span] | None = None

    attention_mask: list[int] | None = None

    def __post_init__(self) -> None:
        require_same_length(self.ids, self.tokens, first_name="ids", second_name="tokens")

        if self.spans is not None:
            require_same_length(self.ids, self.spans, first_name="ids", second_name="spans")

        if self.attention_mask is not None:
            require_same_length(
                self.ids,
                self.attention_mask,
                first_name="ids",
                second_name="attention_mask",
            )

    @property
    def length(self) -> int:
        """Number of tokens in this encoding."""

        return len(self.ids)

    def __len__(self) -> int:
        return len(self.ids)

    def to_dict(self) -> dict[str, Any]:
        """
        Reduce to primitives for persistence or transport.

        Spans are rendered as ``[start, end]`` pairs so the payload
        stays JSON-serialisable.
        """

        return {
            "ids": list(self.ids),
            "tokens": list(self.tokens),
            "spans": None if self.spans is None else [[s.start, s.end] for s in self.spans],
            "attention_mask": None if self.attention_mask is None else list(self.attention_mask),
        }

    def truncate(self, max_length: int) -> Encoding:
        """
        Return a new encoding keeping at most ``max_length`` tokens.

        Parameters
        ----------
        max_length:
            Maximum number of tokens to keep. Encodings already at or
            below this length are returned as an equal copy.

        Returns
        -------
        A new :class:`Encoding`.

        Raises
        ------
        ValidationError
            If ``max_length`` is negative.

        Example
        -------
        ::

            Encoding([1, 2, 3], ["a", "b", "c"]).truncate(2).tokens
            -> ["a", "b"]
        """

        require_non_negative(max_length, name="max_length")

        return Encoding(
            ids=self.ids[:max_length],
            tokens=self.tokens[:max_length],
            spans=None if self.spans is None else self.spans[:max_length],
            attention_mask=None
            if self.attention_mask is None
            else self.attention_mask[:max_length],
        )

    def pad_to(self, length: int, pad_id: int, *, pad_token: str = "<pad>") -> Encoding:
        """
        Return a new encoding right-padded to ``length`` tokens.

        The result always carries an attention mask, since padding is
        exactly the situation in which one becomes necessary.

        Spans are dropped from a padded encoding: a padding token has no
        position in the source text, and inventing one would let a
        downstream offset lookup return a plausible but wrong answer.

        Parameters
        ----------
        length:
            Target token count.

        pad_id:
            Vocabulary id of the padding token, normally ``0``.

        pad_token:
            Surface form recorded for padding positions.

        Returns
        -------
        A new :class:`Encoding` of exactly ``length`` tokens.

        Raises
        ------
        ValidationError
            If ``length`` is negative or shorter than the encoding.

        Example
        -------
        ::

            Encoding([5], ["hi"]).pad_to(3, pad_id=0).ids
            -> [5, 0, 0]
        """

        require_non_negative(length, name="length")

        if length < self.length:
            raise ValidationError(
                "Cannot pad to a length shorter than the encoding; truncate instead",
                length=length,
                encoding_length=self.length,
            )

        missing = length - self.length

        existing_mask = (
            self.attention_mask if self.attention_mask is not None else [1] * self.length
        )

        return Encoding(
            ids=self.ids + [pad_id] * missing,
            tokens=self.tokens + [pad_token] * missing,
            spans=None,
            attention_mask=existing_mask + [0] * missing,
        )
