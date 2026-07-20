"""
Scoring an encoder at the task it exists for.

Everything else in this package measures a property of a model: how
isotropic its vectors are, how efficiently a tokenizer segments, whether
two words that should be similar are. None of that answers the only
question that matters for a retrieval encoder — given a query, does the
right passage come back?

Without this, a training run reports that its loss fell. A falling loss
is compatible with having learned nothing useful: contrastive training
on pairs whose anchor words already appear in their positive can be
solved by string matching, and the loss will fall beautifully while the
model learns to match substrings. That is not hypothetical here; Hindi
Wikipedia's title/lead pairs average 0.977 overlap.

So this measures retrieval, and it breaks the result down three ways —
by language, by pair kind, and by lexical overlap. The last is the one
that catches a model which only appears to work.

**Three things decide whether a number here means anything**, and all
three are reported alongside it:

*The candidate pool.* Recall@1 against 100 candidates and against
100,000 are different tasks. A number without its pool size is not
interpretable, so the pool is never omitted.

*The random baseline.* Ranking at chance gives recall@1 of 1/N. Printing
it next to the measurement is the difference between "0.42" and "0.42
against a chance level of 0.001".

*Duplicate positives.* If two queries share an identical passage,
neither the model nor the metric can tell which was meant, and the
scoring silently punishes correct behaviour. They are removed before
scoring, and the count of what was removed is reported.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from multilingual_embedding.core.exceptions import ValidationError
from multilingual_embedding.core.logging import get_logger
from multilingual_embedding.utils.hashing import hash_text

__all__ = [
    "RetrievalReport",
    "RetrievalScores",
    "evaluate_retrieval",
]

_logger = get_logger(__name__)

# Overlap bands for the leakage breakdown. A model that scores well only
# in the top band has learned to match strings rather than meaning.
_OVERLAP_BANDS: tuple[tuple[str, float, float], ...] = (
    ("low <0.3", 0.0, 0.3),
    ("mid 0.3-0.7", 0.3, 0.7),
    ("high >0.7", 0.7, 1.01),
)

# Encoding and scoring happen in blocks so that a large evaluation does
# not build an N-by-N score matrix in one allocation.
_BLOCK = 512


class Pair(Protocol):
    """
    The slice of a mined pair this module needs.

    Structural rather than a concrete import, so a caller can score
    against pairs from anywhere — a hand-built evaluation set, another
    project's data — without constructing the corpus layer's type.
    """

    @property
    def anchor(self) -> str: ...

    @property
    def positive(self) -> str: ...


class Encodes(Protocol):
    """The slice of ``TextEncoder`` this module needs."""

    def encode_batch(self, texts: Sequence[str]) -> NDArray[np.float32]: ...


@dataclass(slots=True)
class RetrievalScores:
    """
    How well queries found their passages.

    Attributes
    ----------
    queries:
        Number of queries scored.

    candidates:
        Size of the pool each query ranked against. Recall means nothing
        without it.

    recall_at_1, recall_at_5, recall_at_10:
        Share of queries whose own passage appeared in the top k.

    mrr:
        Mean reciprocal rank of the correct passage.

    ndcg_at_10:
        Normalised discounted cumulative gain, which rewards ranking the
        answer higher rather than merely inside the cut-off.

    random_recall_at_1:
        What chance alone would score, ``1 / candidates``. Included so a
        reader can tell a real result from an impressive-looking one.
    """

    queries: int = 0

    candidates: int = 0

    recall_at_1: float = 0.0

    recall_at_5: float = 0.0

    recall_at_10: float = 0.0

    mrr: float = 0.0

    ndcg_at_10: float = 0.0

    @property
    def random_recall_at_1(self) -> float:
        """Chance level, for reading the numbers above against."""

        return 1.0 / self.candidates if self.candidates else 0.0

    @property
    def lift_over_chance(self) -> float:
        """
        How many times better than chance recall@1 is.

        A model that has learned nothing scores about 1.0 here whatever
        its raw recall looks like on a small pool.
        """

        chance = self.random_recall_at_1

        return self.recall_at_1 / chance if chance else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Reduce to primitives for reporting."""

        return {
            "queries": self.queries,
            "candidates": self.candidates,
            "recall_at_1": round(self.recall_at_1, 4),
            "recall_at_5": round(self.recall_at_5, 4),
            "recall_at_10": round(self.recall_at_10, 4),
            "mrr": round(self.mrr, 4),
            "ndcg_at_10": round(self.ndcg_at_10, 4),
            "random_recall_at_1": round(self.random_recall_at_1, 6),
            "lift_over_chance": round(self.lift_over_chance, 1),
        }


@dataclass(slots=True)
class RetrievalReport:
    """
    The whole picture, including the breakdowns that catch a fake result.

    Attributes
    ----------
    overall:
        Scores across every query.

    by_language, by_kind:
        The same scores per language and per pair kind.

    by_overlap:
        Scores per lexical-overlap band. **This is the one to read
        first.** A model scoring well only in the high band has learned
        string matching, and its loss curve will look identical to one
        that learned meaning.

    dropped_duplicate_positives:
        Pairs removed because another pair had the same passage. Scoring
        them is impossible, and a large number here means the pair set
        needs deduplicating rather than the model needing work.
    """

    overall: RetrievalScores = field(default_factory=RetrievalScores)

    by_language: dict[str, RetrievalScores] = field(default_factory=dict)

    by_kind: dict[str, RetrievalScores] = field(default_factory=dict)

    by_overlap: dict[str, RetrievalScores] = field(default_factory=dict)

    dropped_duplicate_positives: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Reduce to primitives for reporting."""

        return {
            "overall": self.overall.to_dict(),
            "by_language": {k: v.to_dict() for k, v in sorted(self.by_language.items())},
            "by_kind": {k: v.to_dict() for k, v in sorted(self.by_kind.items())},
            "by_overlap": {k: v.to_dict() for k, v in sorted(self.by_overlap.items())},
            "dropped_duplicate_positives": self.dropped_duplicate_positives,
        }

    def summary(self) -> str:
        """A few lines a human reads before deciding to look further."""

        lines = [
            f"queries {self.overall.queries:,} against {self.overall.candidates:,} candidates",
            f"recall@1 {self.overall.recall_at_1:.3f}  "
            f"recall@10 {self.overall.recall_at_10:.3f}  "
            f"MRR {self.overall.mrr:.3f}",
            f"chance recall@1 is {self.overall.random_recall_at_1:.5f}, "
            f"so this is {self.overall.lift_over_chance:.0f}x chance",
        ]

        if self.by_overlap:
            lines.append("")

            lines.append("by lexical overlap — read this before believing the rest:")

            for band, scores in sorted(self.by_overlap.items()):
                lines.append(
                    f"  {band:14} recall@1 {scores.recall_at_1:.3f}  ({scores.queries:,} queries)"
                )

        return "\n".join(lines)


def _score_block(
    anchors: NDArray[np.float32],
    positives: NDArray[np.float32],
    offset: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """
    Rank every positive for a block of anchors.

    Returns the rank of each anchor's own positive, and the index of
    whatever ranked first — the latter so a caller can tell a near miss
    from a wild one.
    """

    similarity = anchors @ positives.T

    correct = np.arange(offset, offset + len(anchors))

    # The score the right answer got, against the scores of everything
    # else. Rank is how many candidates beat it, so a rank of 0 means it
    # came first. Computed by comparison rather than by sorting, which
    # would cost N log N per query for a number one comparison gives.
    own = similarity[np.arange(len(anchors)), correct]

    ranks = (similarity > own[:, None]).sum(axis=1)

    return ranks, similarity.argmax(axis=1)


def _scores_from_ranks(ranks: NDArray[np.int64], candidates: int) -> RetrievalScores:
    """Turn a set of ranks into the metrics, without re-ranking anything."""

    if len(ranks) == 0:
        return RetrievalScores(candidates=candidates)

    reciprocal = 1.0 / (ranks + 1)

    # nDCG with a single relevant document reduces to 1/log2(rank+2),
    # since the ideal ranking puts it first and scores 1.
    gains = np.where(ranks < 10, 1.0 / np.log2(ranks + 2), 0.0)

    return RetrievalScores(
        queries=len(ranks),
        candidates=candidates,
        recall_at_1=float((ranks < 1).mean()),
        recall_at_5=float((ranks < 5).mean()),
        recall_at_10=float((ranks < 10).mean()),
        mrr=float(reciprocal.mean()),
        ndcg_at_10=float(gains.mean()),
    )


def evaluate_retrieval(
    encoder: Encodes,
    pairs: Sequence[Any],
    *,
    limit: int | None = 2000,
) -> RetrievalReport:
    """
    Measure whether an encoder retrieves the right passage.

    Each pair's anchor is a query and its positive is the one correct
    answer; every other pair's positive is a distractor. So the candidate
    pool is the pair set, and its size is what makes the numbers mean
    something.

    Parameters
    ----------
    encoder:
        Anything with ``encode_batch``, so this scores the static and the
        contextual models through one code path.

    pairs:
        Mined or hand-built pairs. ``language``, ``kind`` and ``overlap``
        are used for the breakdowns when present and skipped when not.

    limit:
        Queries to score. The default keeps an evaluation to seconds;
        scoring is quadratic in the pool, so raising it raises both cost
        and difficulty. ``None`` uses everything.

    Raises
    ------
    ValidationError
        If fewer than two scoreable pairs remain, since a pool of one
        makes every query trivially correct.

    Example
    -------
    ::

        report = evaluate_retrieval(encoder, pairs, limit=1000)

        print(report.summary())
    """

    selected = list(pairs)[:limit] if limit is not None else list(pairs)

    # Identical passages make a query unscoreable: the model cannot know
    # which of two identical texts was meant, and neither can the metric.
    seen: set[str] = set()

    kept: list[Any] = []

    duplicates = 0

    for pair in selected:
        digest = hash_text(" ".join(pair.positive.split()))

        if digest in seen:
            duplicates += 1

            continue

        seen.add(digest)

        kept.append(pair)

    if len(kept) < 2:
        raise ValidationError(
            "retrieval evaluation needs at least two pairs with distinct "
            "positives, since a pool of one is trivially correct",
            supplied=len(selected),
            after_deduplication=len(kept),
        )

    anchors = encoder.encode_batch([pair.anchor for pair in kept])

    positives = encoder.encode_batch([pair.positive for pair in kept])

    ranks = np.concatenate(
        [
            _score_block(anchors[start : start + _BLOCK], positives, start)[0]
            for start in range(0, len(kept), _BLOCK)
        ]
    )

    report = RetrievalReport(
        overall=_scores_from_ranks(ranks, len(kept)),
        dropped_duplicate_positives=duplicates,
    )

    report.by_language = _grouped(kept, ranks, lambda p: getattr(p, "language", None))

    report.by_kind = _grouped(kept, ranks, lambda p: getattr(p, "kind", None))

    report.by_overlap = _grouped(kept, ranks, _band)

    _logger.info(
        "Evaluated retrieval",
        extra={
            "queries": report.overall.queries,
            "candidates": report.overall.candidates,
            "recall_at_1": round(report.overall.recall_at_1, 4),
            "lift_over_chance": round(report.overall.lift_over_chance, 1),
        },
    )

    return report


def _band(pair: Any) -> str | None:
    """Which overlap band a pair falls in, if it records an overlap."""

    overlap = getattr(pair, "overlap", None)

    if overlap is None:
        return None

    for name, low, high in _OVERLAP_BANDS:
        if low <= overlap < high:
            return name

    return None


def _grouped(
    pairs: Sequence[Any],
    ranks: NDArray[np.int64],
    key: Any,
) -> dict[str, RetrievalScores]:
    """
    Score each group separately, against the whole pool.

    The pool stays the full candidate set rather than shrinking to the
    group, because otherwise a small group would look easy for no reason
    other than being small — which is exactly the artefact these
    breakdowns exist to avoid.
    """

    buckets: dict[str, list[int]] = {}

    for index, pair in enumerate(pairs):
        name = key(pair)

        if name is not None:
            buckets.setdefault(str(name), []).append(index)

    return {
        name: _scores_from_ranks(ranks[indices], len(pairs)) for name, indices in buckets.items()
    }
