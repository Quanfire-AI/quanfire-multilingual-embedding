"""
Hard-negative mining.

The thing under test is not "does it return negatives" — anything
returns negatives. It is whether the *rejections* are right, because a
mined negative that is really a correct answer trains the model to push
the right passage away, and it does so with the largest gradient in the
batch. That failure is silent twice over: nothing raises, and the loss
curve improves, because a model taught to reject correct answers is
being taught something and learns it.

So the geometry here is planted rather than learned. A lookup encoder
places each text at a chosen angle on the unit circle, which makes every
cosine score in these tests an exact number the test author picked. A
real encoder would make the same assertions unfalsifiable — a candidate
that "should" be rejected as too similar is only rejected if the model
happens to score it that way.

No torch. :func:`mine_negatives` takes a ``TextEncoder`` and nothing
more, so it is tested against the contract rather than against the one
implementation that currently satisfies it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
import pytest
from numpy.typing import NDArray

from multilingual_embedding.core.exceptions import ValidationError
from multilingual_embedding.corpus.pairs import MinedPair
from multilingual_embedding.embedding.encoder import TextEncoder
from multilingual_embedding.embedding.negatives import (
    NegativeConfig,
    NegativeStatistics,
    mine_negatives,
)


def at(degrees: float) -> list[float]:
    """A unit vector at a chosen angle, so cosine scores are exact."""

    radians = math.radians(degrees)

    return [math.cos(radians), math.sin(radians)]


class LookupEncoder:
    """
    A ``TextEncoder`` whose vectors the test chooses.

    Text with no entry encodes to a zero vector, which is what the
    encoder contract promises for input it cannot handle. That is not
    padding for convenience: it is the degenerate case the miner has to
    recognise by norm rather than by score, so it needs to be reachable.
    """

    def __init__(self, vectors: Mapping[str, Sequence[float]]) -> None:
        self._vectors = dict(vectors)

        self.seen: list[str] = []

    @property
    def dimension(self) -> int:
        return 2

    def encode(self, text: str) -> NDArray[np.float32]:
        return self.encode_batch([text])[0]

    def encode_batch(self, texts: Sequence[str]) -> NDArray[np.float32]:
        self.seen.extend(texts)

        return np.asarray(
            [self._vectors.get(text, (0.0, 0.0)) for text in texts],
            dtype=np.float32,
        )


def pair(anchor: str, positive: str, document: str) -> MinedPair:
    return MinedPair(
        anchor=anchor,
        positive=positive,
        kind="title_lead",
        document=document,
        language="en",
        overlap=0.0,
    )


# Three pairs whose positives are each other's candidate pool. Measured
# from anchor a0, the pool ranks q1 (cos 10 = 0.985), then q0 (cos 60 =
# 0.500, and its own positive), then q2 (cos 80 = 0.174). One fixture
# therefore reaches every guard by moving one threshold at a time.
PAIRS = [
    pair("a0", "q0", "d0"),
    pair("a1", "q1", "d1"),
    pair("a2", "q2", "d2"),
]

VECTORS = {
    "a0": at(0),
    "a1": at(90),
    "a2": at(180),
    "q0": at(60),
    "q1": at(10),
    "q2": at(80),
}


@pytest.fixture
def encoder() -> LookupEncoder:
    return LookupEncoder(VECTORS)


class TestWhatGetsMined:
    def test_the_hardest_surviving_candidate_is_chosen(self, encoder: LookupEncoder) -> None:
        """
        Ranked by similarity, not sampled: the point is difficulty.

        ``q1`` at 0.985 beats ``q2`` at 0.174, so a miner that returned
        any surviving candidate rather than the hardest one would fail
        here and nowhere else.
        """

        mined, _ = mine_negatives(
            PAIRS,
            encoder,
            NegativeConfig(count=1, pool=8, maximum_similarity=0.999),
        )

        assert mined[0].negatives == ("q1",)

    def test_negatives_are_capped_at_the_requested_count(self, encoder: LookupEncoder) -> None:
        mined, _ = mine_negatives(PAIRS, encoder, NegativeConfig(count=1, pool=8))

        assert all(len(item.negatives) <= 1 for item in mined)

    def test_every_pair_comes_back_in_order(self, encoder: LookupEncoder) -> None:
        """
        A pair that gained no negative is still a training example.

        Dropping it would shrink the training set by however many pairs
        the filters happened to starve, without saying so. This
        framework has already shipped that bug once, in a pair-kind
        filter, and it is the reason the count is asserted here rather
        than assumed.
        """

        mined, statistics = mine_negatives(
            PAIRS,
            encoder,
            # A floor above every planted score starves every pair.
            NegativeConfig(count=4, pool=8, minimum_similarity=0.99, maximum_similarity=1.0),
        )

        assert [item.anchor for item in mined] == ["a0", "a1", "a2"]

        assert all(item.negatives == () for item in mined)

        assert statistics.pairs_without_negatives == 3

    def test_provenance_survives_mining(self, encoder: LookupEncoder) -> None:
        """Only ``negatives`` changes; the rest of the record is carried."""

        mined, _ = mine_negatives(PAIRS, encoder, NegativeConfig(count=1, pool=8))

        assert mined[0].kind == "title_lead"

        assert mined[0].document == "d0"

        assert mined[0].language == "en"

    def test_re_mining_replaces_rather_than_appends(self, encoder: LookupEncoder) -> None:
        """
        Negatives are relative to a model, so a second run against a
        newer adapter must not leave the first run's negatives in place.
        Accumulating them would train the new model against the old
        model's confusions and call the result an improvement.
        """

        once, _ = mine_negatives(PAIRS, encoder, NegativeConfig(count=1, pool=8))

        twice, _ = mine_negatives(once, encoder, NegativeConfig(count=1, pool=8))

        assert twice[0].negatives == once[0].negatives


class TestTheGuards:
    def test_the_pairs_own_positive_is_never_mined_against_it(self, encoder: LookupEncoder) -> None:
        """
        The positive sits near the top of its own ranking by
        construction, so this is the rejection that fires most often and
        the one whose absence would be most destructive.
        """

        mined, statistics = mine_negatives(PAIRS, encoder, NegativeConfig(count=4, pool=8))

        assert all(item.positive not in item.negatives for item in mined)

        assert statistics.rejected_as_positive == 3

    def test_a_candidate_from_the_anchors_own_document_is_rejected(self) -> None:
        """Two passages from one article are about one subject."""

        same_document = [
            pair("a0", "q0", "d0"),
            pair("a1", "q1", "d0"),
            pair("a2", "q2", "d2"),
        ]

        mined, statistics = mine_negatives(
            same_document,
            LookupEncoder(VECTORS),
            NegativeConfig(count=4, pool=8),
        )

        assert "q1" not in mined[0].negatives

        assert statistics.rejected_same_document == 2

    def test_the_provenance_guard_can_be_turned_off(self) -> None:
        """
        Off is a real choice — a corpus of one long document has no
        other candidates — so it is a flag rather than a rule.
        """

        same_document = [
            pair("a0", "q0", "d0"),
            pair("a1", "q1", "d0"),
            pair("a2", "q2", "d2"),
        ]

        mined, _ = mine_negatives(
            same_document,
            LookupEncoder(VECTORS),
            NegativeConfig(count=1, pool=8, allow_same_document=True, maximum_similarity=0.999),
        )

        assert mined[0].negatives == ("q1",)

    def test_a_candidate_above_the_ceiling_is_rejected(self, encoder: LookupEncoder) -> None:
        """
        ``q1`` scores 0.985 against ``a0`` — near enough to be a
        paraphrase of the answer rather than a wrong answer.
        """

        mined, statistics = mine_negatives(
            PAIRS,
            encoder,
            NegativeConfig(count=4, pool=8, maximum_similarity=0.95),
        )

        assert mined[0].negatives == ("q2",)

        # Twice: ``q1`` is too close to ``a0`` and ``q2`` is too close
        # to ``a1``. The guard is per anchor, not per candidate.
        assert statistics.rejected_too_similar == 2

    def test_a_candidate_below_the_floor_is_rejected(self, encoder: LookupEncoder) -> None:
        """
        ``q2`` scores 0.174, which in-batch sampling supplies for free.
        Storing it would widen every training batch for nothing.
        """

        mined, statistics = mine_negatives(
            PAIRS,
            encoder,
            NegativeConfig(count=4, pool=8, maximum_similarity=0.95, minimum_similarity=0.2),
        )

        assert mined[0].negatives == ()

        assert statistics.rejected_too_easy >= 1

    def test_text_the_encoder_cannot_encode_is_never_mined(self) -> None:
        """
        Asserted at a floor of -1.0, which is the setting that makes the
        guard load-bearing.

        The contract returns a zero vector, which scores 0.0 against
        everything, so the default floor of 0.0 discards it as a side
        effect. That is a coincidence of a default, not a guard: a run
        against an untrained encoder lowers the floor precisely because
        most honest candidates score below zero, and it would then start
        collecting text the encoder never read.
        """

        with_junk = [*PAIRS, pair("a3", "unencodable", "d3")]

        mined, statistics = mine_negatives(
            with_junk,
            LookupEncoder(VECTORS),
            NegativeConfig(pool=8, minimum_similarity=-1.0),
        )

        assert all("unencodable" not in item.negatives for item in mined)

        assert statistics.rejected_unencodable > 0

    def test_an_anchor_the_encoder_cannot_encode_mines_nothing(self) -> None:
        """
        Its scores against the whole pool are zero, so the ranking is
        arbitrary order rather than a ranking. Four negatives drawn from
        it would be four passages chosen by nothing at all, and they
        would look exactly like mined ones in the file.
        """

        with_junk = [*PAIRS, pair("unencodable", "q3", "d3")]

        mined, statistics = mine_negatives(
            with_junk,
            LookupEncoder({**VECTORS, "q3": at(45)}),
            NegativeConfig(pool=8, minimum_similarity=-1.0),
        )

        assert mined[3].negatives == ()

        assert statistics.anchors_unencodable == 1


class TestWhatIsCounted:
    def test_every_rejection_reason_is_counted_separately(self, encoder: LookupEncoder) -> None:
        """
        They mean different things. Same-document says the pair set is
        dense in a few articles; above-the-ceiling says the pool is full
        of near-duplicates; below-the-floor says there are no hard
        negatives to find yet. One total would say none of that.
        """

        _, statistics = mine_negatives(
            PAIRS,
            encoder,
            NegativeConfig(count=4, pool=8, maximum_similarity=0.95, minimum_similarity=0.2),
        )

        summary = statistics.to_dict()

        assert summary["rejected_as_positive"]

        assert summary["rejected_too_similar"]

        assert summary["rejected_too_easy"]

        assert summary["rejected_same_document"] == 0

    def test_negatives_the_model_prefers_to_the_truth_are_counted(
        self, encoder: LookupEncoder
    ) -> None:
        """
        ``q1`` scores 0.985 against ``a0`` while its own positive
        ``q0`` scores 0.500. Kept with a permissive ceiling, it is
        exactly the kind of candidate most likely to be a correct answer
        in disguise, and the count is what makes that visible.

        The same happens for ``a1``, whose positive ``q1`` scores 0.174
        while ``q2`` scores 0.985. ``a2`` keeps nothing — every
        candidate scores below zero against it — so the run accepts two
        negatives and both outrank their own answer.
        """

        _, statistics = mine_negatives(
            PAIRS,
            encoder,
            NegativeConfig(count=1, pool=8, maximum_similarity=0.999),
        )

        assert statistics.accepted == 2

        assert statistics.outranking_the_positive == 2

        assert statistics.suspicion_rate == pytest.approx(1.0)

    def test_the_counts_are_not_published_as_a_false_negative_rate(self) -> None:
        """
        The guarded honesty of this module, asserted rather than
        documented.

        ``outranking_the_positive`` counts the population false
        negatives are drawn from, not the false negatives. A key named
        for the rate would be read as the rate, quoted as the rate, and
        would be wrong by an unknown factor in an unknown direction. The
        only route to the real number is labelling the audit sample by
        hand, so no field here may claim to be it.
        """

        keys = NegativeStatistics().to_dict()

        assert not any("false_negative" in key for key in keys)

    def test_the_mean_similarity_is_over_accepted_negatives_only(
        self, encoder: LookupEncoder
    ) -> None:
        _, statistics = mine_negatives(
            PAIRS,
            encoder,
            NegativeConfig(count=1, pool=8, maximum_similarity=0.95),
        )

        # a0 keeps q2 (0.174); a1 and a2 keep whatever survives for them.
        assert 0.0 < statistics.mean_similarity < 0.95

    def test_an_empty_run_reports_a_zero_rate_rather_than_dividing_by_zero(self) -> None:
        assert NegativeStatistics().suspicion_rate == 0.0


class TestTheAuditSample:
    def test_it_holds_the_most_suspicious_end(self, encoder: LookupEncoder) -> None:
        """
        Sorted by similarity descending, because that is where false
        negatives concentrate. A uniform sample would spend a labeller's
        afternoon confirming that unrelated passages are unrelated.
        """

        _, statistics = mine_negatives(
            PAIRS,
            encoder,
            NegativeConfig(count=4, pool=8, maximum_similarity=0.999),
        )

        scores = [record.similarity for record in statistics.audit]

        assert scores == sorted(scores, reverse=True)

    def test_a_record_carries_all_three_texts_and_both_scores(self, encoder: LookupEncoder) -> None:
        """A labeller cannot judge a negative without the pair it is for."""

        _, statistics = mine_negatives(
            PAIRS,
            encoder,
            NegativeConfig(count=1, pool=8, maximum_similarity=0.999),
        )

        record = statistics.audit[0].to_record()

        assert record["anchor"] and record["positive"] and record["negative"]

        assert record["similarity"] > record["positive_similarity"]

        assert record["outranks_the_positive"] is True

        # The field the labeller fills in. Present and unset, so an
        # unlabelled file is distinguishable from a file labelled "no".
        assert record["is_actually_correct"] is None

    def test_it_is_bounded(self, encoder: LookupEncoder) -> None:
        _, statistics = mine_negatives(
            PAIRS,
            encoder,
            NegativeConfig(count=4, pool=8, maximum_similarity=0.999, audit_sample=1),
        )

        assert len(statistics.audit) == 1

    def test_it_can_be_turned_off(self, encoder: LookupEncoder) -> None:
        _, statistics = mine_negatives(
            PAIRS,
            encoder,
            NegativeConfig(pool=8, audit_sample=0),
        )

        assert statistics.audit == []


class TestPrefixes:
    def test_the_two_sides_are_prefixed_differently(self) -> None:
        """
        An E5 model mined without its prefixes ranks by the wrong
        geometry and still returns a full set of negatives, so this is
        checked at the encoder boundary rather than inferred from the
        result.
        """

        encoder = LookupEncoder({})

        mine_negatives(
            PAIRS,
            encoder,
            NegativeConfig(query_prefix="query: ", passage_prefix="passage: "),
        )

        assert "query: a0" in encoder.seen

        assert "passage: q0" in encoder.seen

    def test_stored_negatives_carry_no_prefix(self) -> None:
        """
        The pair file is prefix-free by convention. A prefix stored here
        would be applied a second time by
        ``pipelines.adaptation.prefixed`` at training time, producing
        ``passage: passage: ...`` — which no checkpoint was trained on
        and nothing would raise about.
        """

        encoder = LookupEncoder({f"query: {k}": v for k, v in VECTORS.items()} | VECTORS)

        mined, _ = mine_negatives(
            PAIRS,
            encoder,
            NegativeConfig(count=1, pool=8, query_prefix="query: ", maximum_similarity=0.999),
        )

        assert mined[0].negatives == ("q1",)


class TestRefusals:
    def test_one_pair_cannot_be_mined(self, encoder: LookupEncoder) -> None:
        """The candidate pool is the pair set's own positives."""

        with pytest.raises(ValidationError):
            mine_negatives([PAIRS[0]], encoder)

    def test_a_pool_smaller_than_the_count_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            NegativeConfig(count=8, pool=4)

    def test_inverted_similarity_bounds_are_refused(self) -> None:
        with pytest.raises(ValidationError):
            NegativeConfig(minimum_similarity=0.9, maximum_similarity=0.5)

    def test_a_floor_below_minus_one_is_refused(self) -> None:
        """
        Cosine has a domain, and a floor outside it is a typo rather
        than a permissive setting.
        """

        with pytest.raises(ValidationError):
            NegativeConfig(minimum_similarity=-1.5)

    def test_a_negative_floor_inside_the_domain_is_allowed(self) -> None:
        """
        An encoder that has not been trained yet puts most of its
        candidates below zero. Refusing to look there would leave that
        run with no negatives at all rather than with easy ones, which
        is the opposite of the setting's purpose.
        """

        assert NegativeConfig(minimum_similarity=-1.0).minimum_similarity == -1.0

    @pytest.mark.parametrize("field", ["count", "pool", "batch_size"])
    def test_non_positive_sizes_are_refused(self, field: str) -> None:
        with pytest.raises(ValidationError):
            NegativeConfig(**{field: 0})

    def test_a_negative_audit_sample_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            NegativeConfig(audit_sample=-1)


class TestScale:
    def test_a_pool_larger_than_the_candidate_set_is_clamped(self, encoder: LookupEncoder) -> None:
        """
        Three pairs and a pool of 32 must not index past the end of the
        ranking. Small pair sets are what a trial run uses.
        """

        mined, _ = mine_negatives(PAIRS, encoder, NegativeConfig(pool=32))

        assert len(mined) == 3

    def test_scoring_is_blocked_rather_than_materialised(self, encoder: LookupEncoder) -> None:
        """
        Anchors are scored in blocks of ``batch_size``, so the full
        ``pairs x candidates`` matrix is never allocated. A batch size
        below the pair count exercises the seam where a block boundary
        could mis-index an anchor against another anchor's positive.
        """

        blocked, _ = mine_negatives(PAIRS, encoder, NegativeConfig(count=1, pool=8, batch_size=1))

        whole, _ = mine_negatives(PAIRS, encoder, NegativeConfig(count=1, pool=8, batch_size=64))

        assert [item.negatives for item in blocked] == [item.negatives for item in whole]


def test_the_miner_satisfies_the_encoder_contract_it_asks_for() -> None:
    """
    The miner takes a ``TextEncoder``, not a torch model.

    That is what keeps it in ``embedding/`` rather than
    ``embedding/neural/``, and it is why this whole file runs on a base
    install with no training stack present.
    """

    assert isinstance(LookupEncoder({}), TextEncoder)
