"""
Mining hard negatives against a model.

Contrastive training already has negatives: every other passage in the
batch. They are free, and they are mostly useless. A batch drawn at
random from a Wikipedia pair set contrasts *"When was the Chola dynasty
founded?"* against passages on protein folding and Norwegian ferry
timetables, and separating those is a problem the model solved in its
first hundred steps. After that the loss keeps falling while almost
every gradient carries no information.

A **hard negative** is a passage that the current model ranks near the
correct answer and that is nonetheless wrong — a different dynasty, a
different date, the same subject one section over. Contrasting against
those is where the remaining quality is. Finding them needs a model,
which is why this module sits here and not in
:mod:`multilingual_embedding.corpus.pairs`: manufacturing a pair is a
property of document structure, but manufacturing a *negative* is a
property of what some particular encoder currently confuses.

The candidate pool is the pair set's own positives. That is not a
convenience — it is the same pool the training objective draws in-batch
negatives from, so a negative mined here is a passage the model could
actually have been asked about, rather than text from a corpus the
training set never sees.

**The failure this module exists to avoid is the false negative.** Rank
every passage by similarity to a query and the passages at the top are,
by construction, the ones most likely to *also be correct answers*. A
mined negative that is really a right answer does not merely waste a
gradient: it teaches the model to push the correct passage away, and it
does so with the largest gradient in the batch because the model
currently scores it highly. Mining hard negatives carelessly is one of
the few operations in this framework that reliably makes a model worse
while the loss curve looks better than before.

Three guards run, and all three are counted rather than assumed:

1. **Provenance.** A candidate from the anchor's own document is
   rejected. Two passages mined from one article are about one subject;
   :class:`~multilingual_embedding.corpus.pairs.MinedPair` has carried
   ``document`` since it was written, for exactly this.
2. **A similarity ceiling.** Above
   :attr:`NegativeConfig.maximum_similarity` a candidate is treated as a
   paraphrase of the positive rather than a hard negative. This is the
   standard denoising heuristic and it is a heuristic: it cannot tell a
   paraphrase from a genuinely distinct passage that a weak model has
   collapsed onto the same point.
3. **A similarity floor.** Below
   :attr:`NegativeConfig.minimum_similarity` a candidate is no harder
   than what in-batch sampling already supplies, so storing it spends
   memory in the training step for nothing.

**What is measured, and what is not.** The statistics report how many
candidates each guard rejected, and how many surviving negatives the
model scores *above* the pair's own positive
(:attr:`NegativeStatistics.outranking_the_positive`). That last number is
the honest proxy for the false-negative rate and it is a proxy, not a
measurement: it counts negatives the model finds more convincing than
the truth, which is the population false negatives are drawn from, not
the false negatives themselves. A real rate needs someone to read a
sample and say which mined negatives are actually correct answers.
:attr:`NegativeStatistics.audit` is that sample, drawn from the most
suspicious end, ready to be labelled. Nothing here will produce that
number on its own, and reporting one as though it had would be the kind
of confident wrong figure this project exists to avoid.

Prefixes
--------

Pairs go in and come out **unprefixed**. An E5-family model needs
``query: `` on the anchor and ``passage: `` on the candidates to rank
anything correctly, so the prefixes are applied here for encoding and
stripped back off before storage. Mining against unprefixed text finds
the negatives of a model nobody deploys; storing prefixed text would
double the prefix the moment the pair set reaches
:func:`multilingual_embedding.pipelines.adaptation.prefixed`. Both
failures are silent, so both are handled in one place rather than left
to the caller's memory.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from multilingual_embedding.core.exceptions import ValidationError
from multilingual_embedding.core.logging import get_logger
from multilingual_embedding.corpus.pairs import MinedPair
from multilingual_embedding.utils.validation import require_positive

from .encoder import TextEncoder

__all__ = [
    "AuditRecord",
    "NegativeConfig",
    "NegativeStatistics",
    "mine_negatives",
]

_logger = get_logger(__name__)


@dataclass(slots=True, frozen=True)
class AuditRecord:
    """
    One mined negative, with enough context for a human to judge it.

    A sample of these is the only route to a real false-negative rate.
    Everything else this module reports is a count of what a heuristic
    rejected.

    Attributes
    ----------
    anchor, positive, negative:
        The three texts, unprefixed, as a reader would see them.

    similarity:
        What the mining encoder scored the anchor against this negative.

    positive_similarity:
        What it scored the anchor against the true positive. When
        ``similarity`` exceeds this, the model already prefers the
        negative to the right answer.
    """

    anchor: str

    positive: str

    negative: str

    similarity: float

    positive_similarity: float

    @property
    def outranks_the_positive(self) -> bool:
        """Whether the model scored this negative above the true answer."""

        return self.similarity > self.positive_similarity

    def to_record(self) -> dict[str, Any]:
        """Reduce to a JSON Lines record, with a blank field to fill in."""

        return {
            "anchor": self.anchor,
            "positive": self.positive,
            "negative": self.negative,
            "similarity": round(self.similarity, 4),
            "positive_similarity": round(self.positive_similarity, 4),
            "outranks_the_positive": self.outranks_the_positive,
            # The point of the file. A labeller writes true when the
            # "negative" above is in fact a correct answer to the anchor.
            "is_actually_correct": None,
        }


@dataclass(slots=True, frozen=True)
class NegativeConfig:
    """
    Mining rules.

    Attributes
    ----------
    count:
        Negatives kept per pair. Each one becomes an extra candidate
        column in every training batch that contains its pair, so this
        multiplies the memory a contrastive step needs. Four is the
        usual published choice and the default here.

    pool:
        Candidates examined per anchor before filtering. Must exceed
        ``count`` by enough to survive the guards; the ceiling and the
        provenance rule both bite hardest at the top of the ranking,
        which is precisely where the pool is spent.

    maximum_similarity:
        Reject a candidate at or above this cosine score as a probable
        paraphrase of the positive. Lower is safer and yields easier
        negatives.

    minimum_similarity:
        Reject a candidate at or below this score as no harder than an
        in-batch negative. Cosine's domain is ``[-1, 1]`` and so is this,
        because an encoder that has not been trained yet — the
        from-scratch path, or a first mining pass — puts most of its
        candidates below zero, and refusing to look at them would leave
        that path with no negatives at all rather than with easy ones.

        This does **not** double as the guard against unencodable text.
        A zero vector scores exactly 0.0 against everything, so a floor
        of 0.0 happens to exclude it and a floor of ``-1.0`` happens not
        to; a guard that works by coincidence of a default is not a
        guard. Unencodable text is rejected by name, at every setting.

    allow_same_document:
        Keep candidates mined from the anchor's own document. Off by
        default, because two passages from one article are about one
        subject and the model is usually right to score them together.

    query_prefix, passage_prefix:
        Applied for encoding only, never stored. See the module
        docstring.

    batch_size:
        Texts per ``encode_batch`` call.

    audit_sample:
        How many mined negatives to keep in
        :attr:`NegativeStatistics.audit` for labelling. Drawn from the
        highest-scoring end, which is where false negatives concentrate,
        so it is a sample of the suspicious rather than of the whole —
        deliberately, and it means the labelled rate is an upper bound
        rather than an estimate of the population.
    """

    count: int = 4

    pool: int = 32

    maximum_similarity: float = 0.95

    minimum_similarity: float = 0.0

    allow_same_document: bool = False

    query_prefix: str = ""

    passage_prefix: str = ""

    batch_size: int = 64

    audit_sample: int = 50

    def __post_init__(self) -> None:
        require_positive(self.count, name="count")

        require_positive(self.pool, name="pool")

        require_positive(self.batch_size, name="batch_size")

        if self.pool < self.count:
            raise ValidationError(
                "pool must be at least count, since every candidate examined may be rejected",
                pool=self.pool,
                count=self.count,
            )

        if not -1.0 <= self.minimum_similarity < self.maximum_similarity <= 1.0:
            raise ValidationError(
                "similarity bounds must satisfy -1 <= minimum < maximum <= 1",
                minimum_similarity=self.minimum_similarity,
                maximum_similarity=self.maximum_similarity,
            )

        if self.audit_sample < 0:
            raise ValidationError("audit_sample cannot be negative", audit_sample=self.audit_sample)


@dataclass(slots=True)
class NegativeStatistics:
    """
    What mining produced, and what it threw away.

    Every rejection reason is counted separately because they mean
    different things. A large ``rejected_same_document`` says the pair
    set is dense in a few articles; a large ``rejected_too_similar``
    says the pool is full of near-duplicate passages; a large
    ``rejected_too_easy`` says the pool is too small or the model too
    weak for hard negatives to exist in it yet.

    Attributes
    ----------
    outranking_the_positive:
        Accepted negatives the model scored above the pair's own
        positive. The honest proxy for the false-negative rate — see the
        module docstring for why it is a proxy and not the rate.

    audit:
        A sample of the highest-scoring accepted negatives, for
        labelling by hand. Not a metric; excluded from :meth:`to_dict`.
    """

    pairs: int = 0

    candidates: int = 0

    rejected_as_positive: int = 0

    rejected_same_document: int = 0

    rejected_too_similar: int = 0

    rejected_too_easy: int = 0

    rejected_unencodable: int = 0

    accepted: int = 0

    pairs_without_negatives: int = 0

    anchors_unencodable: int = 0

    outranking_the_positive: int = 0

    mean_similarity: float = 0.0

    audit: list[AuditRecord] = field(default_factory=list)

    @property
    def suspicion_rate(self) -> float:
        """Share of accepted negatives the model prefers to the truth."""

        return self.outranking_the_positive / self.accepted if self.accepted else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Reduce the counts to a serialisable mapping."""

        return {
            "pairs": self.pairs,
            "candidates": self.candidates,
            "rejected_as_positive": self.rejected_as_positive,
            "rejected_same_document": self.rejected_same_document,
            "rejected_too_similar": self.rejected_too_similar,
            "rejected_too_easy": self.rejected_too_easy,
            "rejected_unencodable": self.rejected_unencodable,
            "accepted": self.accepted,
            "pairs_without_negatives": self.pairs_without_negatives,
            "anchors_unencodable": self.anchors_unencodable,
            "outranking_the_positive": self.outranking_the_positive,
            "suspicion_rate": round(self.suspicion_rate, 4),
            "mean_similarity": round(self.mean_similarity, 4),
        }


def _encode(
    encoder: TextEncoder,
    texts: Sequence[str],
    prefix: str,
    batch_size: int,
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """
    Encode in batches, normalise to unit length, and say which rows the
    encoder could not encode.

    Normalising here rather than trusting the encoder is deliberate: the
    ``TextEncoder`` contract does not promise unit vectors, and a dot
    product over unnormalised ones is not a cosine.

    The contract's other promise is that unencodable text yields a zero
    vector rather than NaN, so the caller can detect it with a norm
    check. This is that check. Those rows are returned as zero rather
    than divided by zero, and flagged, because a zero row scores 0.0
    against everything — a number that means "no information" and reads
    exactly like a mediocre similarity.
    """

    blocks: list[NDArray[np.float32]] = []

    for start in range(0, len(texts), batch_size):
        chunk = [prefix + text for text in texts[start : start + batch_size]]

        blocks.append(np.asarray(encoder.encode_batch(chunk), dtype=np.float32))

    vectors = np.vstack(blocks) if blocks else np.zeros((0, encoder.dimension), dtype=np.float32)

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)

    unencodable = norms.reshape(-1) == 0.0

    normalised = np.asarray(vectors / np.where(norms == 0.0, 1.0, norms), dtype=np.float32)

    return normalised, unencodable


def mine_negatives(
    pairs: Sequence[MinedPair],
    encoder: TextEncoder,
    config: NegativeConfig | None = None,
) -> tuple[list[MinedPair], NegativeStatistics]:
    """
    Attach hard negatives to a pair set, and report what happened.

    Every pair is returned, in the order given, whether or not it gained
    negatives. A pair that gained none trains exactly as it did before —
    on in-batch negatives — so dropping it would silently shrink the
    training set, and this framework has fixed that bug once already.

    Encodes each anchor once and each distinct positive once, then scores
    anchors against candidates in blocks. Memory is bounded by the
    vectors, not by the ``pairs x candidates`` score matrix, which is
    never materialised whole.

    Parameters
    ----------
    pairs:
        Unprefixed pairs. Any negatives already attached are replaced,
        so re-mining against a newer model is idempotent rather than
        cumulative.

    encoder:
        The model whose confusions are being mined. Normally the adapter
        about to be retrained: hard negatives are relative to a model,
        and negatives mined against a different one are merely
        moderately similar passages.

    config:
        Mining rules. The defaults are the conservative end — see
        :class:`NegativeConfig`.

    Returns
    -------
    The pairs with negatives attached, and the statistics.

    Raises
    ------
    ValidationError
        If fewer than two pairs are given. One pair is its own only
        candidate, so there is nothing to mine against.

    Example
    -------
    ::

        pairs, stats = mine_negatives(
            sample_pairs(path, 50_000),
            pipeline.encoder,
            NegativeConfig(query_prefix="query: ", passage_prefix="passage: "),
        )

        print(stats.to_dict()["suspicion_rate"])
    """

    settings = config if config is not None else NegativeConfig()

    if len(pairs) < 2:
        raise ValidationError(
            "mining hard negatives needs at least two pairs, since the "
            "candidate pool is the pair set's own positives",
            pairs=len(pairs),
        )

    statistics = NegativeStatistics(pairs=len(pairs))

    # The pool is de-duplicated by text. Two pairs sharing a positive
    # would otherwise put the same passage in the ranking twice, and the
    # provenance guard would clear one copy and keep the other.
    candidates: list[str] = []

    candidate_index: dict[str, int] = {}

    # First document wins. Mined pairs are de-duplicated by content hash
    # upstream, so a text reaching here from two documents is rare; when
    # it happens the guard protects one of them rather than neither.
    candidate_document: list[str] = []

    for pair in pairs:
        if pair.positive not in candidate_index:
            candidate_index[pair.positive] = len(candidates)

            candidates.append(pair.positive)

            candidate_document.append(pair.document)

    _logger.info(
        "Encoding for hard-negative mining",
        extra={"anchors": len(pairs), "candidates": len(candidates)},
    )

    anchor_vectors, anchor_unencodable = _encode(
        encoder,
        [pair.anchor for pair in pairs],
        settings.query_prefix,
        settings.batch_size,
    )

    candidate_vectors, candidate_unencodable = _encode(
        encoder,
        candidates,
        settings.passage_prefix,
        settings.batch_size,
    )

    # One more than the pool, because the pair's own positive occupies a
    # slot at the top of nearly every ranking and is always rejected.
    depth = min(settings.pool + 1, len(candidates))

    mined: list[MinedPair] = []

    similarity_total = 0.0

    for start in range(0, len(pairs), settings.batch_size):
        block = pairs[start : start + settings.batch_size]

        scores = anchor_vectors[start : start + settings.batch_size] @ candidate_vectors.T

        # argpartition puts the `depth` largest scores in the leading
        # slice in arbitrary order, which is O(n) against a full sort's
        # O(n log n) over a pool that is the whole pair set.
        frontier = np.argpartition(-scores, depth - 1, axis=1)[:, :depth]

        for row, pair in enumerate(block):
            # An anchor the encoder could not encode scores 0.0 against
            # the entire pool, so its "ranking" is arbitrary order and
            # any negative taken from it would be a negative for no
            # reason. Skipped by name rather than left to a threshold.
            if anchor_unencodable[start + row]:
                statistics.anchors_unencodable += 1

                statistics.pairs_without_negatives += 1

                mined.append(pair)

                continue

            positive_column = candidate_index[pair.positive]

            positive_similarity = float(scores[row, positive_column])

            ranked = sorted(
                frontier[row].tolist(),
                key=lambda column: -scores[row, column],
            )

            kept: list[str] = []

            for column in ranked:
                if len(kept) == settings.count:
                    break

                statistics.candidates += 1

                if column == positive_column:
                    statistics.rejected_as_positive += 1

                    continue

                # Checked by name rather than by score. A zero vector
                # scores exactly 0.0 against everything, which the
                # default floor happens to reject — but the floor is a
                # difficulty setting, and a run that lowers it to mine
                # against an untrained encoder would otherwise start
                # collecting text the encoder cannot read at all.
                if candidate_unencodable[column]:
                    statistics.rejected_unencodable += 1

                    continue

                if not settings.allow_same_document and (
                    pair.document and candidate_document[column] == pair.document
                ):
                    statistics.rejected_same_document += 1

                    continue

                similarity = float(scores[row, column])

                if similarity >= settings.maximum_similarity:
                    statistics.rejected_too_similar += 1

                    continue

                if similarity <= settings.minimum_similarity:
                    # Sorted descending, so everything below this is
                    # also too easy. Counting only the one examined
                    # keeps `candidates` a count of what was examined.
                    statistics.rejected_too_easy += 1

                    break

                kept.append(candidates[column])

                statistics.accepted += 1

                similarity_total += similarity

                if similarity > positive_similarity:
                    statistics.outranking_the_positive += 1

                if len(statistics.audit) < settings.audit_sample:
                    statistics.audit.append(
                        AuditRecord(
                            anchor=pair.anchor,
                            positive=pair.positive,
                            negative=candidates[column],
                            similarity=similarity,
                            positive_similarity=positive_similarity,
                        )
                    )

            if not kept:
                statistics.pairs_without_negatives += 1

            mined.append(
                MinedPair(
                    anchor=pair.anchor,
                    positive=pair.positive,
                    kind=pair.kind,
                    document=pair.document,
                    language=pair.language,
                    overlap=pair.overlap,
                    negatives=tuple(kept),
                )
            )

    statistics.mean_similarity = (
        similarity_total / statistics.accepted if statistics.accepted else 0.0
    )

    statistics.audit.sort(key=lambda record: -record.similarity)

    _report(statistics)

    return mined, statistics


def _report(statistics: NegativeStatistics) -> None:
    """Log the yield, and warn when the result is not worth training on."""

    _logger.info("Mined hard negatives", extra=statistics.to_dict())

    if statistics.pairs_without_negatives:
        share = statistics.pairs_without_negatives / statistics.pairs

        if share > 0.5:
            _logger.warning(
                "Most pairs gained no hard negative, so this run mostly "
                "reproduces in-batch training",
                extra={
                    "pairs_without_negatives": statistics.pairs_without_negatives,
                    "pairs": statistics.pairs,
                    "remedy": "raise pool, or lower minimum_similarity",
                },
            )

    if statistics.suspicion_rate > 0.25:
        _logger.warning(
            "A quarter of the mined negatives score above their own positive; "
            "training on these may teach the model to reject correct answers",
            extra={
                "outranking_the_positive": statistics.outranking_the_positive,
                "accepted": statistics.accepted,
                "remedy": "lower maximum_similarity, and label the audit sample",
            },
        )
