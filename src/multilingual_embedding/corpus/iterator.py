"""
Streaming iteration helpers.

Tokenizer and embedding training both make several passes over the
sentences of a corpus. Materialising those sentences once per pass would
bound corpus size by available memory, so this module provides
re-iterable views that pull from the source each time.

:class:`SentenceStream` is the key abstraction: it is an iterable, not an
iterator, so ``for _ in stream`` can run repeatedly and each run restarts
the underlying source.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable, Iterator

__all__ = [
    "SentenceStream",
    "batched",
    "count_iterable",
    "take",
]


class SentenceStream(Iterable[str]):
    """
    A re-iterable stream of sentence strings.

    Parameters
    ----------
    factory:
        Callable returning a fresh iterator over sentences. Called once
        per iteration pass, which is what makes multi-epoch training
        possible without holding the corpus in memory.

    limit:
        Optional cap on sentences yielded per pass. Useful for smoke
        tests against a large corpus.

    transform:
        Optional per sentence transformation, applied lazily. Used to
        thread normalization into the stream without a separate pass.

    Example
    -------
    ::

        stream = SentenceStream(lambda: reader.iter_sentences())

        for epoch in range(5):
            for sentence in stream:   # restarts the reader each epoch
                ...
    """

    __slots__ = ("_factory", "_limit", "_transform")

    def __init__(
        self,
        factory: Callable[[], Iterator[str]],
        *,
        limit: int | None = None,
        transform: Callable[[str], str] | None = None,
    ) -> None:
        self._factory = factory

        self._limit = limit

        self._transform = transform

    def __iter__(self) -> Iterator[str]:
        source = self._factory()

        if self._transform is not None:
            source = (self._transform(sentence) for sentence in source)

        if self._limit is not None:
            source = itertools.islice(source, self._limit)

        return iter(source)

    def map(self, transform: Callable[[str], str]) -> SentenceStream:
        """Return a new stream with ``transform`` applied to each sentence."""

        existing = self._transform

        composed: Callable[[str], str]

        if existing is None:
            composed = transform
        else:

            def composed(sentence: str) -> str:
                return transform(existing(sentence))

        return SentenceStream(self._factory, limit=self._limit, transform=composed)

    def limited(self, limit: int) -> SentenceStream:
        """Return a new stream yielding at most ``limit`` sentences."""

        return SentenceStream(self._factory, limit=limit, transform=self._transform)

    def count(self) -> int:
        """
        Count sentences by consuming one full pass.

        Costs a complete pass over the source; cache the result rather
        than calling it inside a loop.
        """

        return sum(1 for _ in self)


def batched(iterable: Iterable[str], size: int) -> Iterator[list[str]]:
    """
    Yield consecutive lists of at most ``size`` items.

    The final batch is short rather than padded.
    """

    if size <= 0:
        raise ValueError("size must be > 0")

    iterator = iter(iterable)

    while batch := list(itertools.islice(iterator, size)):
        yield batch


def take(iterable: Iterable[str], count: int) -> list[str]:
    """Return the first ``count`` items as a list."""

    return list(itertools.islice(iterable, count))


def count_iterable(iterable: Iterable[object]) -> int:
    """Count items in an iterable, consuming it."""

    return sum(1 for _ in iterable)
