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

**On languages.** ``by_language`` scores each language separately and
answers "how well does this model serve Tamil". It does not answer "can a
Tamil query find a Hindi passage", which is a different question and the
one a shared multilingual vector space is usually assumed to settle.
Answering it properly needs aligned pairs — a query in one language whose
correct answer is in another — and nothing here mines those.

What *is* measurable without them is the prerequisite: whether the space
is organised by language at all. See :class:`LanguageSeparation`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from multilingual_embedding.core.exceptions import ValidationError
from multilingual_embedding.core.logging import get_logger
from multilingual_embedding.utils.hashing import hash_text

__all__ = [
    "LanguageSeparation",
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

# How many near misses per query the language-separation measure looks
# at. Small, because the question is what the model *nearly* retrieved,
# and a long tail of candidates says nothing about that.
_SEPARATION_K = 5


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
    def confidence_interval(self) -> tuple[float, float]:
        """
        A 95% interval on ``recall_at_1``, by the Wilson method.

        Reported because a percentage hides how few queries produced it.
        Hindi's high-overlap band moved from 0.5828 to 0.6290 — a
        confident-sounding +7.9% — which is 366 of 628 against 395 of
        628, and those intervals overlap. The low-overlap band moved 22
        of 155 to 54 of 155, which does not.

        Wilson rather than the normal approximation because the
        interesting bands are small and their rates are near the ends,
        where the normal interval misbehaves and can even leave [0, 1].
        """

        if not self.queries:
            return (0.0, 0.0)

        z = 1.96

        n = self.queries

        centre = (self.recall_at_1 + z * z / (2 * n)) / (1 + z * z / n)

        spread = (
            z
            * math.sqrt(self.recall_at_1 * (1 - self.recall_at_1) / n + z * z / (4 * n * n))
            / (1 + z * z / n)
        )

        return (max(0.0, centre - spread), min(1.0, centre + spread))

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
            "recall_at_1_hits": round(self.recall_at_1 * self.queries),
            "recall_at_1_ci95": [round(bound, 4) for bound in self.confidence_interval],
            "recall_at_5": round(self.recall_at_5, 4),
            "recall_at_10": round(self.recall_at_10, 4),
            "mrr": round(self.mrr, 4),
            "ndcg_at_10": round(self.ndcg_at_10, 4),
            "random_recall_at_1": round(self.random_recall_at_1, 6),
            "lift_over_chance": round(self.lift_over_chance, 1),
        }


@dataclass(slots=True)
class LanguageSeparation:
    """
    Whether the vector space is organised by language or by meaning.

    **This is not cross-lingual retrieval evaluation.** That needs
    aligned pairs — a query in one language whose correct answer is in
    another — and nothing in this project mines them. What this measures
    is the prerequisite, and it can be computed from the ordinary
    multilingual pair sets that already exist.

    The method. For each query, look at the near misses: the highest
    scoring candidates that are not the correct answer. Count how many
    are in the query's own language, and compare that against how many
    would be expected if the model ranked purely on meaning and ignored
    language entirely — which is just that language's share of the pool.

    A model that has genuinely aligned its languages puts a Tamil
    passage about tenancy near a Hindi query about tenancy, so its near
    misses look like the pool. A model that has merely learned to encode
    *which language this is* fills every query's near misses with the
    query's own language, whatever they are about.

    Attributes
    ----------
    queries:
        Queries contributing to the measure. Zero when the pair set has
        fewer than two languages, in which case the question is not
        being asked.

    pool:
        Language composition of the candidate pool, which is what the
        expectation is computed from.

    observed:
        Share of near misses in the query's own language, averaged over
        queries.

    expected:
        The same share for a language-blind model.

    by_language:
        Observed share per query language. A model can be aligned in one
        direction and not the other, and an average hides that.

    Notes
    -----
    ``separation`` is the number to read: 1.0 means language plays no
    part in the ranking, and the pool's own composition is the ceiling
    on how high it can go. It is a diagnostic, not a score — a strongly
    separated space is the *expected* outcome of training on pairs whose
    two sides are always the same language, which is what every pair
    this project mines looks like. Reading a high value as a defect
    would be reading it wrong; reading it as evidence that cross-lingual
    retrieval will work anyway would be worse.
    """

    queries: int = 0

    pool: dict[str, int] = field(default_factory=dict)

    observed: float = 0.0

    expected: float = 0.0

    by_language: dict[str, float] = field(default_factory=dict)

    @property
    def measured(self) -> bool:
        """False when the pair set was monolingual, so nothing was asked."""

        return self.queries > 0 and len(self.pool) > 1

    @property
    def separation(self) -> float:
        """
        Observed over expected. 1.0 is a language-blind space.

        Above 1.0 the model prefers its own language among near misses
        by more than the pool composition explains.
        """

        return self.observed / self.expected if self.expected else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Reduce to primitives for reporting."""

        return {
            "measured": self.measured,
            "queries": self.queries,
            "pool": dict(sorted(self.pool.items())),
            "observed_same_language_share": round(self.observed, 4),
            "expected_same_language_share": round(self.expected, 4),
            "separation": round(self.separation, 3),
            "by_language": {k: round(v, 4) for k, v in sorted(self.by_language.items())},
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

    language_separation:
        Whether the space is organised by language rather than by
        meaning. Empty and ``measured`` False for a monolingual pair set.
        Read :class:`LanguageSeparation` before quoting it — it is not a
        cross-lingual retrieval score.

    dropped_duplicate_positives:
        Pairs removed because another pair had the same passage. Scoring
        them is impossible, and a large number here means the pair set
        needs deduplicating rather than the model needing work.
    """

    overall: RetrievalScores = field(default_factory=RetrievalScores)

    by_language: dict[str, RetrievalScores] = field(default_factory=dict)

    by_kind: dict[str, RetrievalScores] = field(default_factory=dict)

    by_overlap: dict[str, RetrievalScores] = field(default_factory=dict)

    language_separation: LanguageSeparation = field(default_factory=LanguageSeparation)

    dropped_duplicate_positives: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Reduce to primitives for reporting."""

        return {
            "overall": self.overall.to_dict(),
            "by_language": {k: v.to_dict() for k, v in sorted(self.by_language.items())},
            "by_kind": {k: v.to_dict() for k, v in sorted(self.by_kind.items())},
            "by_overlap": {k: v.to_dict() for k, v in sorted(self.by_overlap.items())},
            "language_separation": self.language_separation.to_dict(),
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

        separation = self.language_separation

        if separation.measured:
            lines.append("")

            lines.append(
                f"language separation {separation.separation:.2f}x — near misses are "
                f"{separation.observed:.0%} same-language against {separation.expected:.0%} "
                f"expected of a language-blind space"
            )

            lines.append("  (not a cross-lingual retrieval score; see LanguageSeparation)")

        return "\n".join(lines)


def _score_block(
    anchors: NDArray[np.float32],
    positives: NDArray[np.float32],
    offset: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """
    Rank every positive for a block of anchors.

    Returns the rank of each anchor's own positive, and the indices of
    its top near misses — the correct answer masked out, so every index
    returned is something the model preferred or nearly preferred over
    it. :class:`LanguageSeparation` reads the second.
    """

    similarity = anchors @ positives.T

    correct = np.arange(offset, offset + len(anchors))

    rows = np.arange(len(anchors))

    # The score the right answer got, against the scores of everything
    # else. Rank is how many candidates beat it, so a rank of 0 means it
    # came first. Computed by comparison rather than by sorting, which
    # would cost N log N per query for a number one comparison gives.
    own = similarity[rows, correct]

    # Ties are counted *against* the correct answer — the pessimistic rank
    # (candidates scoring >= own, less the correct answer itself). The
    # optimistic `> own` would hand a rank of 0 to a degenerate encoder
    # that has collapsed the space: when every passage encodes to nearly
    # the same vector, every similarity ties, so `> own` is all-False and
    # the model that learned nothing scores a perfect recall@1. Counting
    # ties as misses turns that collapse into the worst rank, not the best,
    # so a fine-tune that destroyed the geometry reports as harmful — which
    # it is — rather than as flawless.
    ranks = (similarity >= own[:, None]).sum(axis=1) - 1

    # Masked rather than filtered afterwards: dropping the correct index
    # from a top-k list per row would leave rows of different lengths,
    # and the whole point of doing this in blocks is to stay in arrays.
    similarity[rows, correct] = -np.inf

    width = min(_SEPARATION_K, similarity.shape[1])

    top = np.argpartition(-similarity, width - 1, axis=1)[:, :width]

    return ranks, top


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

    blocks = [
        _score_block(anchors[start : start + _BLOCK], positives, start)
        for start in range(0, len(kept), _BLOCK)
    ]

    ranks = np.concatenate([block[0] for block in blocks])

    near_misses = np.concatenate([block[1] for block in blocks])

    report = RetrievalReport(
        overall=_scores_from_ranks(ranks, len(kept)),
        language_separation=_language_separation(kept, near_misses),
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


def _language_separation(
    pairs: Sequence[Any],
    near_misses: NDArray[np.int64],
) -> LanguageSeparation:
    """
    How much of a query's near misses its own language accounts for.

    Returns an unmeasured result rather than raising when the pair set
    is monolingual or carries no language labels. There is no meaningful
    answer in either case, and a zero would be indistinguishable from a
    perfectly aligned space — the same reason the evaluators elsewhere
    in this package leave a missing metric as ``None``.
    """

    languages = [getattr(pair, "language", None) for pair in pairs]

    if any(language is None for language in languages):
        return LanguageSeparation()

    pool: dict[str, int] = {}

    for language in languages:
        pool[str(language)] = pool.get(str(language), 0) + 1

    if len(pool) < 2:
        return LanguageSeparation(pool=pool)

    total = len(pairs)

    observed: list[float] = []

    expected: list[float] = []

    per_language: dict[str, list[float]] = {}

    for index, language in enumerate(languages):
        name = str(language)

        candidates = [int(other) for other in near_misses[index] if int(other) != index]

        if not candidates:
            continue

        share = sum(1 for other in candidates if str(languages[other]) == name) / len(candidates)

        # The query's own passage is not a candidate for itself, so the
        # language-blind expectation excludes it from both counts.
        chance = (pool[name] - 1) / (total - 1)

        observed.append(share)

        expected.append(chance)

        per_language.setdefault(name, []).append(share)

    if not observed:
        return LanguageSeparation(pool=pool)

    return LanguageSeparation(
        queries=len(observed),
        pool=pool,
        observed=sum(observed) / len(observed),
        expected=sum(expected) / len(expected),
        by_language={
            name: sum(shares) / len(shares) for name, shares in sorted(per_language.items())
        },
    )


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
