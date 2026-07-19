"""
Corpus statistics.

Statistics are gathered through an accumulator that consumes one
document at a time, so a report can be produced for a corpus far larger
than memory.

What is tracked is chosen to answer the questions that actually decide
tokenizer and embedding settings: how much text is there, which
languages and scripts is it in, how long are the sentences, and how
skewed is the vocabulary.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from .document import Document
from .script import Script, script_histogram
from .segmentation import split_words

__all__ = [
    "CorpusStatistics",
    "LengthSummary",
    "StatisticsAccumulator",
    "compute_statistics",
]


@dataclass(slots=True, frozen=True)
class LengthSummary:
    """
    Distribution summary for a length measurement.

    Percentiles matter more than the mean here: a corpus with a handful
    of enormous unsegmented sentences will show a reasonable mean and a
    p99 that reveals the problem.
    """

    count: int

    total: int

    minimum: int

    maximum: int

    mean: float

    median: float

    p95: float

    p99: float

    @classmethod
    def from_values(cls, values: list[int]) -> LengthSummary:
        """Summarise a list of measurements."""

        if not values:
            return cls(
                count=0,
                total=0,
                minimum=0,
                maximum=0,
                mean=0.0,
                median=0.0,
                p95=0.0,
                p99=0.0,
            )

        ordered = sorted(values)

        total = sum(ordered)

        return cls(
            count=len(ordered),
            total=total,
            minimum=ordered[0],
            maximum=ordered[-1],
            mean=total / len(ordered),
            median=_percentile(ordered, 0.50),
            p95=_percentile(ordered, 0.95),
            p99=_percentile(ordered, 0.99),
        )

    def to_dict(self) -> dict[str, Any]:
        """Reduce to primitives for reporting."""

        return {
            "count": self.count,
            "total": self.total,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "mean": round(self.mean, 3),
            "median": round(self.median, 3),
            "p95": round(self.p95, 3),
            "p99": round(self.p99, 3),
        }


@dataclass(slots=True)
class CorpusStatistics:
    """
    Aggregate description of a corpus.

    Attributes
    ----------
    document_count, paragraph_count, sentence_count:
        Structural counts.

    character_count, word_count:
        Volume measures. ``word_count`` is whitespace based and therefore
        not comparable across scripts; use ``character_count`` when
        comparing a Chinese corpus against an English one.

    languages, scripts:
        Document counts per declared language and per dominant script.

    sentence_characters, sentence_words:
        Sentence length distributions.

    unique_words, top_words:
        Vocabulary size and the most frequent entries. Both are capped
        by the accumulator's ``max_tracked_words``.

    truncated_vocabulary:
        True when the word cap was reached, meaning ``unique_words``
        understates the true figure.
    """

    document_count: int = 0

    paragraph_count: int = 0

    sentence_count: int = 0

    character_count: int = 0

    word_count: int = 0

    languages: dict[str, int] = field(default_factory=dict)

    scripts: dict[str, int] = field(default_factory=dict)

    sentence_characters: LengthSummary = field(
        default_factory=lambda: LengthSummary.from_values([])
    )

    sentence_words: LengthSummary = field(default_factory=lambda: LengthSummary.from_values([]))

    unique_words: int = 0

    top_words: list[tuple[str, int]] = field(default_factory=list)

    truncated_vocabulary: bool = False

    @property
    def type_token_ratio(self) -> float:
        """
        Unique words divided by total words.

        A very low ratio suggests repetitive or templated text; a very
        high one suggests the corpus is too small to train on.
        """

        return self.unique_words / self.word_count if self.word_count else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Reduce to primitives for report files."""

        return {
            "document_count": self.document_count,
            "paragraph_count": self.paragraph_count,
            "sentence_count": self.sentence_count,
            "character_count": self.character_count,
            "word_count": self.word_count,
            "unique_words": self.unique_words,
            "type_token_ratio": round(self.type_token_ratio, 5),
            "truncated_vocabulary": self.truncated_vocabulary,
            "languages": dict(sorted(self.languages.items())),
            "scripts": dict(sorted(self.scripts.items())),
            "sentence_characters": self.sentence_characters.to_dict(),
            "sentence_words": self.sentence_words.to_dict(),
            "top_words": [list(entry) for entry in self.top_words],
        }


class StatisticsAccumulator:
    """
    Builds :class:`CorpusStatistics` one document at a time.

    Parameters
    ----------
    max_tracked_words:
        Cap on distinct words held in the frequency table. Word
        frequencies are Zipfian, so an uncapped table over a large
        corpus is dominated by singletons and can exhaust memory. Once
        the cap is hit, new words are ignored and
        ``truncated_vocabulary`` is set.

    top_word_count:
        How many of the most frequent words to retain in the result.

    max_tracked_lengths:
        Cap on retained sentence lengths. Percentiles need the values,
        but a reservoir this size is ample and keeps memory flat.
    """

    __slots__ = (
        "_characters",
        "_documents",
        "_languages",
        "_max_tracked_lengths",
        "_max_tracked_words",
        "_paragraphs",
        "_scripts",
        "_sentence_characters",
        "_sentence_words",
        "_sentences",
        "_top_word_count",
        "_truncated",
        "_word_counts",
        "_words",
    )

    def __init__(
        self,
        *,
        max_tracked_words: int = 1_000_000,
        top_word_count: int = 50,
        max_tracked_lengths: int = 500_000,
    ) -> None:
        self._max_tracked_words = max_tracked_words

        self._top_word_count = top_word_count

        self._max_tracked_lengths = max_tracked_lengths

        self._documents = 0

        self._paragraphs = 0

        self._sentences = 0

        self._characters = 0

        self._words = 0

        self._languages: Counter[str] = Counter()

        self._scripts: Counter[str] = Counter()

        self._sentence_characters: list[int] = []

        self._sentence_words: list[int] = []

        self._word_counts: Counter[str] = Counter()

        self._truncated = False

    def add(self, document: Document) -> None:
        """Fold one document into the running totals."""

        self._documents += 1

        self._paragraphs += document.paragraph_count

        self._characters += document.character_count

        language = document.metadata.base.language

        if language:
            self._languages[language] += 1

        self._scripts[_dominant_script_name(document.text)] += 1

        for sentence in document.sentences():
            self._sentences += 1

            words = [span.slice(sentence.text) for span in split_words(sentence.text)]

            self._words += len(words)

            if len(self._sentence_characters) < self._max_tracked_lengths:
                self._sentence_characters.append(sentence.character_count)

                self._sentence_words.append(len(words))

            self._add_words(words)

    def extend(self, documents: Iterable[Document]) -> None:
        """Fold several documents into the running totals."""

        for document in documents:
            self.add(document)

    def result(self) -> CorpusStatistics:
        """Produce the accumulated statistics."""

        return CorpusStatistics(
            document_count=self._documents,
            paragraph_count=self._paragraphs,
            sentence_count=self._sentences,
            character_count=self._characters,
            word_count=self._words,
            languages=dict(self._languages),
            scripts=dict(self._scripts),
            sentence_characters=LengthSummary.from_values(self._sentence_characters),
            sentence_words=LengthSummary.from_values(self._sentence_words),
            unique_words=len(self._word_counts),
            top_words=self._word_counts.most_common(self._top_word_count),
            truncated_vocabulary=self._truncated,
        )

    def _add_words(self, words: list[str]) -> None:
        """Update the frequency table, respecting the tracking cap."""

        for word in words:
            folded = word.casefold()

            if folded in self._word_counts:
                self._word_counts[folded] += 1
            elif len(self._word_counts) < self._max_tracked_words:
                self._word_counts[folded] = 1
            else:
                self._truncated = True


def compute_statistics(
    documents: Iterable[Document],
    **accumulator_options: Any,
) -> CorpusStatistics:
    """
    Compute statistics over an iterable of documents.

    Accepts any iterable, so it works equally on a
    :class:`~multilingual_embedding.corpus.corpus.Corpus` and on a
    streaming reader.
    """

    accumulator = StatisticsAccumulator(**accumulator_options)

    accumulator.extend(documents)

    return accumulator.result()


def _dominant_script_name(text: str) -> str:
    """Return the name of the dominant script, ignoring shared characters."""

    counts = script_histogram(text)

    informative = {
        script: count
        for script, count in counts.items()
        if script not in {Script.COMMON, Script.UNKNOWN}
    }

    if not informative:
        return Script.UNKNOWN.value

    return max(informative.items(), key=lambda item: (item[1], item[0].value))[0].value


def _percentile(ordered: list[int], fraction: float) -> float:
    """
    Linear interpolation percentile over a pre-sorted list.

    Matches the default method used by numpy, so figures reported here
    agree with any downstream analysis done in numpy.
    """

    if not ordered:
        return 0.0

    if len(ordered) == 1:
        return float(ordered[0])

    position = fraction * (len(ordered) - 1)

    lower_index = math.floor(position)

    upper_index = math.ceil(position)

    if lower_index == upper_index:
        return float(ordered[lower_index])

    weight = position - lower_index

    return ordered[lower_index] * (1.0 - weight) + ordered[upper_index] * weight
