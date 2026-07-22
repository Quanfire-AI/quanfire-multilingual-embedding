"""
Retrieval evaluation.

This is the instrument that decides whether a trained encoder is any
good, so the tests are mostly about whether the instrument itself can be
trusted: does it separate a model that learned from one that did not,
does it report the context that makes a number interpretable, and does
it refuse to score what cannot be scored.

The controls matter more than usual here. A retrieval metric that always
returns a flattering number is worse than none, because it launders a
guess into a measurement.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pytest

from multilingual_embedding.core.exceptions import ValidationError
from multilingual_embedding.evaluation.retrieval import evaluate_retrieval


@dataclass
class Pair:
    anchor: str

    positive: str

    language: str = "en"

    kind: str = "title_lead"

    overlap: float = 0.0


def pairs(count: int = 200) -> list[Pair]:
    return [
        Pair(f"query about topic {i}", f"passage discussing topic {i} at length")
        for i in range(count)
    ]


def topic_of(text: str) -> int:
    return int(text.split("topic ")[1].split()[0])


class Perfect:
    """Encodes the topic exactly. The ceiling."""

    def __init__(self, count: int = 200) -> None:
        self.count = count

    def encode_batch(self, texts):  # type: ignore[no-untyped-def]
        vectors = np.zeros((len(texts), self.count), dtype=np.float32)

        for row, text in enumerate(texts):
            vectors[row, topic_of(text)] = 1.0

        return vectors


class Chance:
    """
    Vectors unrelated to meaning. The floor.

    Seeded from the text rather than from a constant. An encoder seeded
    per call returns identical vectors for anchors and positives and
    scores a perfect 1.0 — which is what the first version of this did,
    and it made a broken control look like a working one.
    """

    def encode_batch(self, texts):  # type: ignore[no-untyped-def]
        rows = []

        for text in texts:
            seed = int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), "big")

            vector = np.random.default_rng(seed).normal(size=32).astype(np.float32)

            rows.append(vector / np.linalg.norm(vector))

        return np.stack(rows)


class TestTheInstrumentDiscriminates:
    def test_a_perfect_encoder_scores_one(self) -> None:
        report = evaluate_retrieval(Perfect(), pairs())

        assert report.overall.recall_at_1 == pytest.approx(1.0)

        assert report.overall.mrr == pytest.approx(1.0)

    def test_an_encoder_that_learned_nothing_scores_near_chance(self) -> None:
        """
        The control that matters. If this passed for a random encoder,
        every number the harness produces would be worthless.
        """

        report = evaluate_retrieval(Chance(), pairs())

        assert report.overall.recall_at_1 < 0.05

        assert report.overall.lift_over_chance < 15, (
            "a random encoder is scoring far above chance; the harness is "
            "measuring something other than retrieval"
        )

    def test_chance_level_is_reported_not_assumed(self) -> None:
        report = evaluate_retrieval(Perfect(), pairs(count=200))

        assert report.overall.random_recall_at_1 == pytest.approx(1 / 200)


class TestNumbersCarryTheirContext:
    def test_the_candidate_pool_is_reported(self) -> None:
        """
        Recall@1 against 100 candidates and 100,000 are different tasks,
        so a number without its pool is not interpretable.
        """

        report = evaluate_retrieval(Perfect(), pairs(count=150))

        assert report.overall.candidates == 150

    def test_lift_over_chance_is_one_for_a_useless_model(self) -> None:
        """
        Raw recall flatters a small pool. Lift does not, which is why it
        is the number to read on an unfamiliar evaluation.
        """

        report = evaluate_retrieval(Chance(), pairs(count=400))

        assert report.overall.lift_over_chance < 20

    def test_ranking_higher_scores_better_than_merely_ranking_inside_k(self) -> None:
        """nDCG must reward position, or it is measuring recall twice."""

        from multilingual_embedding.evaluation.retrieval import _scores_from_ranks

        first = _scores_from_ranks(np.zeros(10, dtype=np.int64), 100)

        ninth = _scores_from_ranks(np.full(10, 8, dtype=np.int64), 100)

        assert first.recall_at_10 == ninth.recall_at_10 == 1.0

        assert first.ndcg_at_10 > ninth.ndcg_at_10


class TestBreakdowns:
    def test_scores_are_grouped_by_language(self) -> None:
        mixed = pairs(100)

        for pair in mixed[:40]:
            pair.language = "hi"

        report = evaluate_retrieval(Perfect(100), mixed)

        assert set(report.by_language) == {"en", "hi"}

        assert report.by_language["hi"].queries == 40

    def test_scores_are_grouped_by_overlap_band(self) -> None:
        """
        The breakdown that catches a model which only matches strings.
        """

        mixed = pairs(100)

        for index, pair in enumerate(mixed):
            pair.overlap = 0.1 if index < 50 else 0.9

        report = evaluate_retrieval(Perfect(100), mixed)

        assert set(report.by_overlap) == {"low <0.3", "high >0.7"}

        assert report.by_overlap["low <0.3"].queries == 50

    def test_a_group_is_scored_against_the_whole_pool(self) -> None:
        """
        Otherwise a small group looks easy purely for being small, which
        is the artefact these breakdowns exist to avoid.
        """

        mixed = pairs(100)

        for pair in mixed[:5]:
            pair.language = "ta"

        report = evaluate_retrieval(Perfect(100), mixed)

        assert report.by_language["ta"].candidates == 100

    def test_pairs_without_metadata_are_simply_not_grouped(self) -> None:
        """A hand-built evaluation set need not carry any of it."""

        @dataclass
        class Bare:
            anchor: str

            positive: str

        bare = [Bare(f"query about topic {i}", f"passage discussing topic {i}") for i in range(20)]

        report = evaluate_retrieval(Perfect(20), bare)

        assert report.overall.queries == 20

        assert report.by_language == {}


class TestWhatCannotBeScored:
    def test_duplicate_positives_are_removed_and_counted(self) -> None:
        """
        Two queries sharing a passage cannot both be right, and the
        metric would silently punish correct behaviour.
        """

        duplicated = pairs(50)

        duplicated[10].positive = duplicated[0].positive

        report = evaluate_retrieval(Perfect(50), duplicated)

        assert report.dropped_duplicate_positives == 1

        assert report.overall.queries == 49

    def test_too_few_pairs_is_refused_rather_than_reported(self) -> None:
        """A pool of one makes every query trivially correct."""

        with pytest.raises(ValidationError, match="at least two"):
            evaluate_retrieval(Perfect(), pairs(1))

    def test_a_pool_of_identical_positives_is_refused(self) -> None:
        identical = [Pair(f"query about topic {i}", "the same passage") for i in range(20)]

        with pytest.raises(ValidationError):
            evaluate_retrieval(Perfect(20), identical)


class TestItWorksOnTheRealEncoders:
    def test_it_scores_a_neural_encoder(self) -> None:
        """One code path must serve both model families."""

        torch = pytest.importorskip("torch", reason="requires the neural extra")

        from multilingual_embedding.embedding.neural import (
            EncoderConfig,
            NeuralTextEncoder,
            TransformerEncoderModel,
        )

        class Encoding:
            def __init__(self, ids: list[int]) -> None:
                self.ids = ids

        class Tokenizer:
            def encode(self, text: str) -> Encoding:
                return Encoding([1 + (abs(hash(word)) % 60) for word in text.split()])

        torch.manual_seed(0)

        model = TransformerEncoderModel(
            EncoderConfig(vocabulary_size=64, dimension=32, layers=1, heads=4, max_length=32)
        )

        encoder = NeuralTextEncoder(model, Tokenizer(), device="cpu")

        report = evaluate_retrieval(encoder, pairs(40))

        assert report.overall.queries == 40

        assert 0.0 <= report.overall.recall_at_1 <= 1.0


class TestLanguageSeparation:
    """
    Whether the space is organised by language rather than by meaning.

    This measure is the easiest thing in the module to misread, so the
    tests are mostly about what it must *refuse* to say: it is not a
    cross-lingual retrieval score, it declines to answer on a
    monolingual pair set, and its baseline has to come from the pool's
    own composition rather than from a constant.
    """

    @staticmethod
    def bilingual(count: int = 40) -> list[Pair]:
        """
        Alternating languages, with topics deliberately decoupled from them.

        Deriving the topic from the index would make topic and language
        correlate, and a language-blind encoder would then look
        separated for a reason that has nothing to do with language.
        That is a real way to fool this measure, so the fixture is built
        not to.
        """

        order = list(np.random.default_rng(7).permutation(count))

        return [
            Pair(
                anchor=f"query about topic {order[i]}",
                positive=f"passage discussing topic {order[i]} at length",
                language="en" if i % 2 == 0 else "hi",
            )
            for i in range(count)
        ]

    def test_it_declines_to_answer_on_one_language(self) -> None:
        separation = evaluate_retrieval(Perfect(), pairs(40)).language_separation

        assert not separation.measured

        assert separation.separation == 0.0

    def test_it_declines_when_pairs_carry_no_language(self) -> None:
        @dataclass
        class Unlabelled:
            anchor: str

            positive: str

        unlabelled = [
            Unlabelled(f"query about topic {i}", f"passage discussing topic {i} at length")
            for i in range(20)
        ]

        assert not evaluate_retrieval(Perfect(), unlabelled).language_separation.measured

    def test_a_language_blind_encoder_scores_near_one(self) -> None:
        """
        The calibration. Without it a high number proves nothing, because
        every number this measure produces would be high.
        """

        report = evaluate_retrieval(Perfect(count=40), self.bilingual())

        separation = report.language_separation

        assert separation.measured

        assert separation.separation < 1.35

    def test_a_language_clustered_encoder_scores_well_above_one(self) -> None:
        class ByLanguage:
            """Same-language texts are near-identical whatever they say."""

            def encode_batch(self, texts):  # type: ignore[no-untyped-def]
                vectors = np.zeros((len(texts), 8), dtype=np.float32)

                for row, text in enumerate(texts):
                    vectors[row, 0] = 5.0 if "topic" in text else 0.0

                    # The language axis, inferred from the pair set's own
                    # alternation, dwarfs the topic axis.
                    vectors[row, 1] = 5.0 * (topic_of(text) % 2)

                    vectors[row, 2 + topic_of(text) % 6] = 1.0

                norms = np.linalg.norm(vectors, axis=1, keepdims=True)

                return (vectors / np.maximum(norms, 1e-9)).astype(np.float32)

        aligned = [
            Pair(
                anchor=f"query about topic {i}",
                positive=f"passage discussing topic {i} at length",
                language="en" if i % 2 == 0 else "hi",
            )
            for i in range(40)
        ]

        separation = evaluate_retrieval(ByLanguage(), aligned).language_separation

        assert separation.separation > 1.5

    def test_the_expectation_comes_from_the_pool(self) -> None:
        """
        An unbalanced pool must move the baseline, not the verdict.

        Nine tenths of the passages being English makes "the near misses
        were English" unremarkable, and a measure that ignored that would
        report every skewed corpus as language-separated.
        """

        skewed = [
            Pair(
                anchor=f"query about topic {i}",
                positive=f"passage discussing topic {i} at length",
                language="en" if i % 10 else "hi",
            )
            for i in range(40)
        ]

        separation = evaluate_retrieval(Perfect(count=40), skewed).language_separation

        assert separation.pool == {"en": 36, "hi": 4}

        # Averaged per query, excluding the query's own passage from both
        # counts: 36 English queries each expect 35/39, and 4 Hindi ones
        # expect 3/39.
        assert separation.expected == pytest.approx((36 * 35 / 39 + 4 * 3 / 39) / 40)

        # And the verdict does not move with the skew, which is the point.
        assert separation.separation < 1.2

    def test_it_reports_each_language_separately(self) -> None:
        separation = evaluate_retrieval(Perfect(count=40), self.bilingual()).language_separation

        assert set(separation.by_language) == {"en", "hi"}

    def test_the_summary_says_what_it_is_not(self) -> None:
        """
        The one number here that would be quoted out of context, so the
        line that stops it being quoted travels with it.
        """

        summary = evaluate_retrieval(Perfect(count=40), self.bilingual()).summary()

        assert "language separation" in summary

        assert "not a cross-lingual retrieval score" in summary

    def test_it_survives_a_round_trip_to_primitives(self) -> None:
        payload = evaluate_retrieval(Perfect(count=40), self.bilingual()).to_dict()

        assert payload["language_separation"]["measured"] is True

        assert set(payload["language_separation"]) == {
            "measured",
            "queries",
            "pool",
            "observed_same_language_share",
            "expected_same_language_share",
            "separation",
            "by_language",
        }
