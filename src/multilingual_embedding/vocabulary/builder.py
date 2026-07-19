"""
Streaming vocabulary construction.

Counting every token of a large corpus in a plain dictionary is the
usual way vocabulary building runs out of memory: word frequencies are
Zipfian, so the table fills with singletons that will be discarded by
``min_count`` anyway.

:class:`VocabularyBuilder` bounds the table. When it exceeds
``max_tracked_tokens`` it prunes the entries seen only once, which are
overwhelmingly the ones destined for removal. Pruning is lossy — a token
that would have crossed ``min_count`` later can be undercounted — so the
cap defaults high enough that ordinary corpora never trigger it, and the
builder reports whether it did.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence

from multilingual_embedding.core.logging import get_logger

from .special_tokens import SpecialTokenSet
from .vocabulary import Vocabulary

__all__ = ["VocabularyBuilder"]

_logger = get_logger(__name__)


class VocabularyBuilder:
    """
    Accumulates token counts and produces a :class:`Vocabulary`.

    Parameters
    ----------
    min_count:
        Minimum occurrences for a token to be kept.

    max_size:
        Cap on the resulting vocabulary size, including special tokens.

    max_tracked_tokens:
        Cap on distinct tokens held while counting. Exceeding it triggers
        pruning of singletons.

    special_tokens:
        Reserved tokens for the resulting vocabulary.

    Example
    -------
    ::

        builder = VocabularyBuilder(min_count=5, max_size=32_000)

        for sentence in corpus.sentence_texts():
            builder.add_tokens(sentence.split())

        vocabulary = builder.build()
    """

    __slots__ = (
        "_counts",
        "_max_size",
        "_max_tracked_tokens",
        "_min_count",
        "_pruned",
        "_special",
        "_total_tokens",
    )

    def __init__(
        self,
        *,
        min_count: int = 1,
        max_size: int | None = None,
        max_tracked_tokens: int = 5_000_000,
        special_tokens: SpecialTokenSet | None = None,
    ) -> None:
        self._counts: Counter[str] = Counter()

        self._min_count = min_count

        self._max_size = max_size

        self._max_tracked_tokens = max_tracked_tokens

        self._special = special_tokens

        self._total_tokens = 0

        self._pruned = 0

    @property
    def distinct_tokens(self) -> int:
        """Number of distinct tokens currently tracked."""

        return len(self._counts)

    @property
    def total_tokens(self) -> int:
        """Total token occurrences seen."""

        return self._total_tokens

    @property
    def pruned_count(self) -> int:
        """
        Number of tokens dropped by pruning.

        Non-zero means counts are approximate; raise
        ``max_tracked_tokens`` if exactness matters.
        """

        return self._pruned

    def add(self, token: str) -> None:
        """Count a single token."""

        if not token:
            return

        self._total_tokens += 1

        self._counts[token] += 1

        if len(self._counts) > self._max_tracked_tokens:
            self._prune_singletons()

    def add_tokens(self, tokens: Iterable[str]) -> None:
        """Count an iterable of tokens."""

        for token in tokens:
            self.add(token)

    def add_all(self, sequences: Iterable[Sequence[str]]) -> None:
        """Count tokens from an iterable of token sequences."""

        for sequence in sequences:
            self.add_tokens(sequence)

    def build(self) -> Vocabulary:
        """Produce the vocabulary from the accumulated counts."""

        if self._pruned:
            _logger.warning(
                "Vocabulary counts are approximate; singleton pruning occurred",
                extra={"pruned": self._pruned, "cap": self._max_tracked_tokens},
            )

        return Vocabulary.from_counter(
            self._counts,
            min_count=self._min_count,
            max_size=self._max_size,
            special_tokens=self._special,
        )

    def _prune_singletons(self) -> None:
        """
        Drop tokens seen exactly once to reclaim space.

        If no singletons exist the cap cannot be honoured by pruning
        alone; the table is left as is rather than discarding tokens
        that have real evidence behind them.
        """

        singletons = [token for token, count in self._counts.items() if count == 1]

        if not singletons:
            return

        for token in singletons:
            del self._counts[token]

        self._pruned += len(singletons)
