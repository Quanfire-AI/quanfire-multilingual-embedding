"""
Tokenizer evaluation.

A tokenizer is judged on how efficiently it represents text and how
evenly it does so across languages. The second point is the one that
matters for a multilingual model and the one most often missed: a
vocabulary trained on a corpus that is 90% English will encode English
in few tokens and Tamil in many, so the Tamil half of the model gets a
fraction of the effective context length for the same content.

:class:`TokenizerEvaluator` takes a plain ``tokenize`` callable rather
than a Tokenizer object, so it can score any segmentation function —
including a baseline such as whitespace splitting — without adapters.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from multilingual_embedding.core.logging import get_logger
from multilingual_embedding.corpus.script import detect_script

__all__ = [
    "TokenizerEvaluator",
    "TokenizerMetrics",
    "evaluate_tokenizer",
]

_logger = get_logger(__name__)

Tokenize = Callable[[str], list[str]]


@dataclass(slots=True)
class TokenizerMetrics:
    """
    Efficiency and coverage figures for one tokenizer.

    Attributes
    ----------
    sentence_count, token_count, character_count:
        Raw totals over the evaluated text.

    tokens_per_sentence:
        Mean tokens per sentence.

    characters_per_token:
        Mean characters represented by each token — the compression
        ratio. Higher means a more efficient vocabulary. Comparable
        across languages in a way that token counts alone are not.

    unknown_rate:
        Share of tokens mapping to the unknown token. A subword model
        with byte fallback should be near zero; anything appreciable
        means the vocabulary does not fit the corpus.

    distinct_tokens:
        Number of distinct token types produced.

    vocabulary_utilisation:
        Share of the vocabulary actually used, when the size is known.
        A low figure means capacity is being spent on tokens the corpus
        never produces.

    fertility:
        Mean subword tokens produced per whitespace word. The standard
        measure of over-segmentation; a value near 1.0 means words
        mostly survive intact.
    """

    sentence_count: int = 0

    token_count: int = 0

    character_count: int = 0

    tokens_per_sentence: float = 0.0

    characters_per_token: float = 0.0

    unknown_rate: float = 0.0

    distinct_tokens: int = 0

    vocabulary_utilisation: float | None = None

    fertility: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Reduce to primitives for report files."""

        return {
            "sentence_count": self.sentence_count,
            "token_count": self.token_count,
            "character_count": self.character_count,
            "tokens_per_sentence": round(self.tokens_per_sentence, 4),
            "characters_per_token": round(self.characters_per_token, 4),
            "unknown_rate": round(self.unknown_rate, 6),
            "distinct_tokens": self.distinct_tokens,
            "vocabulary_utilisation": (
                round(self.vocabulary_utilisation, 6)
                if self.vocabulary_utilisation is not None
                else None
            ),
            "fertility": round(self.fertility, 4),
        }


@dataclass(slots=True)
class TokenizerEvaluator:
    """
    Scores a tokenization function over text.

    Parameters
    ----------
    tokenize:
        Callable turning a sentence into a list of token strings.

    unknown_token:
        Surface form counted as unknown.

    vocabulary_size:
        Total vocabulary size, when known, for the utilisation figure.
    """

    tokenize: Tokenize

    unknown_token: str = "<unk>"

    vocabulary_size: int | None = None

    _distinct: Counter[str] = field(default_factory=Counter, init=False, repr=False)

    def evaluate(self, sentences: Iterable[str]) -> TokenizerMetrics:
        """
        Score a stream of sentences.

        Consumes the iterable once, holding only the distinct token set
        in memory, so it runs over corpora larger than memory.
        """

        self._distinct = Counter()

        sentence_count = 0

        token_count = 0

        character_count = 0

        unknown_count = 0

        word_count = 0

        for sentence in sentences:
            tokens = self.tokenize(sentence)

            sentence_count += 1

            token_count += len(tokens)

            character_count += len(sentence)

            word_count += len(sentence.split())

            for token in tokens:
                self._distinct[token] += 1

                if token == self.unknown_token:
                    unknown_count += 1

        metrics = TokenizerMetrics(
            sentence_count=sentence_count,
            token_count=token_count,
            character_count=character_count,
            tokens_per_sentence=token_count / sentence_count if sentence_count else 0.0,
            characters_per_token=character_count / token_count if token_count else 0.0,
            unknown_rate=unknown_count / token_count if token_count else 0.0,
            distinct_tokens=len(self._distinct),
            vocabulary_utilisation=(
                len(self._distinct) / self.vocabulary_size if self.vocabulary_size else None
            ),
            fertility=token_count / word_count if word_count else 0.0,
        )

        _logger.info(
            "Evaluated tokenizer",
            extra={
                "sentences": sentence_count,
                "characters_per_token": round(metrics.characters_per_token, 3),
                "unknown_rate": round(metrics.unknown_rate, 6),
            },
        )

        return metrics

    def evaluate_by_language(
        self,
        sentences_by_language: Mapping[str, Iterable[str]],
    ) -> dict[str, TokenizerMetrics]:
        """
        Score each language separately.

        The per language ``characters_per_token`` spread is the headline
        fairness number: a large gap means the vocabulary serves some
        languages far better than others.
        """

        return {
            language: self.evaluate(sentences)
            for language, sentences in sentences_by_language.items()
        }

    def evaluate_by_script(self, sentences: Iterable[str]) -> dict[str, TokenizerMetrics]:
        """
        Score sentences grouped by their dominant script.

        Useful when the corpus carries no language labels, which is
        common for scraped text.
        """

        grouped: dict[str, list[str]] = {}

        for sentence in sentences:
            grouped.setdefault(detect_script(sentence).dominant.value, []).append(sentence)

        return {script: self.evaluate(group) for script, group in grouped.items()}

    def most_common_tokens(self, count: int = 20) -> list[tuple[str, int]]:
        """Most frequent tokens from the last :meth:`evaluate` call."""

        return self._distinct.most_common(count)


def evaluate_tokenizer(
    tokenize: Tokenize,
    sentences: Iterable[str],
    *,
    unknown_token: str = "<unk>",
    vocabulary_size: int | None = None,
) -> TokenizerMetrics:
    """Convenience wrapper around :class:`TokenizerEvaluator`."""

    return TokenizerEvaluator(
        tokenize=tokenize,
        unknown_token=unknown_token,
        vocabulary_size=vocabulary_size,
    ).evaluate(sentences)


def language_fairness(metrics: Mapping[str, TokenizerMetrics]) -> dict[str, float]:
    """
    Summarise how evenly a tokenizer treats languages.

    Returns the minimum, maximum and ratio of ``characters_per_token``
    across languages. A ratio near 1.0 means comparable efficiency; a
    ratio of 3 means one language needs three times as many tokens for
    the same amount of text.
    """

    values = [
        item.characters_per_token for item in metrics.values() if item.characters_per_token > 0.0
    ]

    if not values:
        return {"minimum": 0.0, "maximum": 0.0, "ratio": 0.0}

    lowest = min(values)

    highest = max(values)

    return {
        "minimum": round(lowest, 4),
        "maximum": round(highest, 4),
        "ratio": round(highest / lowest, 4) if lowest else 0.0,
    }
