"""
Ingest a pre-aligned parallel corpus into :class:`MinedPair` records.

Every other pair source in this package *manufactures* pairs — from
document structure (:mod:`.pairs`), from a langlinks join
(:mod:`.aligned`), or from a model (:mod:`..embedding.negatives`). This
one does not mine anything. It reads a corpus that is *already* sentence
aligned — two files whose i-th lines are translations of each other, the
canonical format of Samanantar, BPCC, and every OPUS release — and turns
each line pair into the record the training pipeline already consumes.

Why it exists is written in the FLORES-200 result: an adapter trained on
Wikipedia *article leads* learns topical alignment and regresses on
sentence-level public bitext. The distribution it never saw is exactly
what this reads. The miner cannot produce it, because no monolingual
Wikipedia dump contains a Hindi sentence next to its Tamil translation;
that alignment was done by the people who built the parallel corpus, and
this module's whole job is to not throw their work away.

The one failure mode it guards hard is silent misalignment. Two parallel
files that disagree on line count mean every pair past the first missing
line is a sentence matched to the *wrong* translation — a corpus that
trains the model to align text with its non-translation, with no error to
point at. :func:`iter_parallel_pairs` refuses to truncate to the shorter
file: it raises when the two streams do not exhaust together.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

from multilingual_embedding.core.exceptions import ValidationError
from multilingual_embedding.utils.validation import require_positive

from .pairs import MinedPair, token_overlap

__all__ = [
    "ParallelConfig",
    "ParallelStatistics",
    "iter_parallel_pairs",
]


@dataclass(slots=True)
class ParallelConfig:
    """
    What counts as a usable line pair from a parallel corpus.

    The character thresholds are *sentence* thresholds, deliberately far
    below :class:`.pairs.PairConfig`'s passage thresholds. A parallel
    corpus is sentences, not article leads, so a 100-character floor would
    discard most of it — and the short sentences are where translation
    equivalence is hardest to fake by lexical overlap, which is the signal
    worth training on.

    Attributes
    ----------
    minimum_anchor_characters:
        Below this the anchor is a fragment — a stray number, a lone
        punctuation mark left by a sentence splitter — and carries no
        query signal.

    minimum_positive_characters:
        Below this the positive is a fragment for the same reason.

    maximum_positive_characters:
        Positives longer than this are truncated by the encoder anyway and
        are usually a splitter failure that ran two sentences together.
        Rejected rather than truncated, so the count is visible.

    deduplicate:
        Drop a line pair whose ``(anchor, positive)`` has already been
        seen. Mined parallel corpora repeat — Samanantar's own paper
        reports duplicate mining across shards — and a duplicated pair is
        a silent up-weighting of whatever it says. Costs a set of hashes
        over the run.

    drop_identical:
        Drop a pair whose two sides are byte-identical after stripping. In
        a cross-lingual corpus these are never translations: they are
        untranslated boilerplate, bare numerals, or URLs the aligner
        passed through. Teaching the model that a string matches itself
        across languages is teaching it nothing and mislabelling the
        space.
    """

    minimum_anchor_characters: int = 10

    minimum_positive_characters: int = 10

    maximum_positive_characters: int = 2000

    deduplicate: bool = True

    drop_identical: bool = True

    def __post_init__(self) -> None:
        require_positive(self.minimum_anchor_characters, name="minimum_anchor_characters")
        require_positive(self.minimum_positive_characters, name="minimum_positive_characters")
        require_positive(self.maximum_positive_characters, name="maximum_positive_characters")

        if self.minimum_positive_characters > self.maximum_positive_characters:
            raise ValidationError(
                "minimum_positive_characters must not exceed maximum_positive_characters",
                minimum=self.minimum_positive_characters,
                maximum=self.maximum_positive_characters,
            )


@dataclass(slots=True)
class ParallelStatistics:
    """
    What the ingest read, produced, and refused — and why it refused.

    The rejection breakdown is the point. A parallel corpus advertised as
    clean that loses a third of its lines to ``rejected_identical`` was not
    clean, and that is a fact about the corpus worth surfacing before it
    reaches a training run rather than after.
    """

    read: int = 0

    produced: int = 0

    rejected_short_anchor: int = 0

    rejected_short_positive: int = 0

    rejected_long_positive: int = 0

    rejected_identical: int = 0

    rejected_duplicate: int = 0

    overlap_total: float = field(default=0.0)

    def to_dict(self) -> dict[str, Any]:
        """Reduce to primitives for a report, mean overlap included."""

        mean_overlap = self.overlap_total / self.produced if self.produced else 0.0

        return {
            "read": self.read,
            "produced": self.produced,
            "rejected": {
                "short_anchor": self.rejected_short_anchor,
                "short_positive": self.rejected_short_positive,
                "long_positive": self.rejected_long_positive,
                "identical": self.rejected_identical,
                "duplicate": self.rejected_duplicate,
            },
            "mean_overlap": round(mean_overlap, 4),
        }


def iter_parallel_pairs(
    anchor_lines: Iterable[str],
    positive_lines: Iterable[str],
    *,
    anchor_language: str | None,
    positive_language: str | None,
    kind: str = "aligned_sentence",
    document_prefix: str = "parallel",
    config: ParallelConfig | None = None,
    statistics: ParallelStatistics | None = None,
) -> Iterator[MinedPair]:
    """
    Zip two line-aligned streams into :class:`MinedPair` records.

    ``anchor_lines`` and ``positive_lines`` are iterables whose i-th
    elements are translations of each other — typically the two files of a
    parallel corpus, opened line by line so a corpus larger than memory
    streams. Each surviving line pair becomes one pair with a *unique*
    ``document`` id: parallel sentences share no subject the way two pairs
    from one Wikipedia article do, so nothing here should ever land in the
    same batch by provenance, and a per-line id says exactly that to the
    sampler and to the same-document guard in negative mining.

    ``language`` on the emitted record is the positive's language, matching
    :class:`.aligned.AlignedPair`, so the evaluator's per-language
    breakdown reads as "how well is language X retrieved". ``overlap`` is
    computed and carried for the same diagnostic reason the other miners
    carry it — a cross-lingual corpus with high mean overlap is not
    cross-lingual.

    Raises
    ------
    ValidationError
        If the two streams do not exhaust together. A parallel corpus with
        unequal line counts is misaligned somewhere above the shorter
        file's length, and every pair past that point matches a sentence
        to the wrong translation. Truncating to the shorter file would
        hide exactly the corruption most worth catching, so this refuses.
    """

    settings = config or ParallelConfig()
    stats = statistics if statistics is not None else ParallelStatistics()
    seen: set[int] = set()

    anchors = iter(anchor_lines)
    positives = iter(positive_lines)
    index = 0

    _sentinel = object()

    while True:
        anchor_raw = next(anchors, _sentinel)
        positive_raw = next(positives, _sentinel)

        anchor_done = anchor_raw is _sentinel
        positive_done = positive_raw is _sentinel

        if anchor_done and positive_done:
            break

        if anchor_done or positive_done:
            raise ValidationError(
                "parallel files disagree on line count; the corpus is misaligned",
                anchor_exhausted=anchor_done,
                positive_exhausted=positive_done,
                lines_read=stats.read,
            )

        stats.read += 1
        index += 1

        anchor = str(anchor_raw).strip()
        positive = str(positive_raw).strip()

        if len(anchor) < settings.minimum_anchor_characters:
            stats.rejected_short_anchor += 1
            continue

        if len(positive) < settings.minimum_positive_characters:
            stats.rejected_short_positive += 1
            continue

        if len(positive) > settings.maximum_positive_characters:
            stats.rejected_long_positive += 1
            continue

        if settings.drop_identical and anchor == positive:
            stats.rejected_identical += 1
            continue

        if settings.deduplicate:
            fingerprint = hash((anchor, positive))
            if fingerprint in seen:
                stats.rejected_duplicate += 1
                continue
            seen.add(fingerprint)

        overlap = token_overlap(anchor, positive)
        stats.overlap_total += overlap
        stats.produced += 1

        yield MinedPair(
            anchor=anchor,
            positive=positive,
            kind=kind,
            document=f"{document_prefix}:{index}",
            language=positive_language,
            overlap=overlap,
        )
