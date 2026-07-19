from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from multilingual_embedding.core.exceptions import ValidationError
from multilingual_embedding.vocabulary.builder import VocabularyBuilder
from multilingual_embedding.vocabulary.special_tokens import (
    BOS_ID,
    EOS_ID,
    PAD_ID,
    UNK_ID,
    SpecialTokenSet,
)
from multilingual_embedding.vocabulary.vocabulary import Vocabulary


@pytest.fixture
def counts() -> Counter[str]:
    return Counter({"the": 10, "cat": 5, "dog": 5, "rare": 1, "नमस्ते": 3})


class TestSpecialTokens:
    def test_ids_are_fixed(self) -> None:
        """
        These ids are baked into trained models, so they must not drift.

        Padding in particular must be zero, so a zero-filled array is a
        valid padded batch.
        """

        assert (PAD_ID, UNK_ID, BOS_ID, EOS_ID) == (0, 1, 2, 3)

    def test_ordering_matches_ids(self) -> None:
        assert SpecialTokenSet().as_tuple() == ("<pad>", "<unk>", "<bos>", "<eos>")

    def test_mapping(self) -> None:
        assert SpecialTokenSet().as_mapping()["<unk>"] == UNK_ID

    def test_membership(self) -> None:
        assert "<pad>" in SpecialTokenSet()

        assert "cat" not in SpecialTokenSet()


class TestConstruction:
    def test_special_tokens_occupy_lowest_ids(self, counts: Counter[str]) -> None:
        vocabulary = Vocabulary.from_counter(counts)

        assert vocabulary.token_of(PAD_ID) == "<pad>"

        assert vocabulary.token_of(UNK_ID) == "<unk>"

    def test_ordered_by_descending_frequency(self, counts: Counter[str]) -> None:
        vocabulary = Vocabulary.from_counter(counts)

        assert vocabulary.most_common(1)[0][0] == "the"

    def test_ordering_is_deterministic(self, counts: Counter[str]) -> None:
        """
        Ties break on the token string, so ordering cannot depend on the
        iteration order of the input mapping.
        """

        forward = Vocabulary.from_counter(dict(counts))

        reversed_counts = dict(reversed(list(counts.items())))

        assert Vocabulary.from_counter(reversed_counts).tokens() == forward.tokens()

    def test_min_count_excludes_rare(self, counts: Counter[str]) -> None:
        vocabulary = Vocabulary.from_counter(counts, min_count=3)

        assert "rare" not in vocabulary

        assert "cat" in vocabulary

    def test_max_size_keeps_most_frequent(self, counts: Counter[str]) -> None:
        vocabulary = Vocabulary.from_counter(counts, max_size=6)

        assert len(vocabulary) == 6

        assert "the" in vocabulary

    def test_max_size_below_special_count_rejected(self, counts: Counter[str]) -> None:
        with pytest.raises(ValidationError):
            Vocabulary.from_counter(counts, max_size=2)

    def test_min_count_below_one_rejected(self, counts: Counter[str]) -> None:
        with pytest.raises(ValidationError):
            Vocabulary.from_counter(counts, min_count=0)

    def test_from_tokens(self) -> None:
        vocabulary = Vocabulary.from_tokens(["a", "b", "a"])

        assert vocabulary.frequency_of("a") == 2

    def test_empty_counter_yields_specials_only(self) -> None:
        assert len(Vocabulary.from_counter({})) == 4


class TestLookup:
    def test_unknown_token_maps_to_unk(self, counts: Counter[str]) -> None:
        """OOV is an expected condition at inference, never an error."""

        assert Vocabulary.from_counter(counts).id_of("absent") == UNK_ID

    def test_out_of_range_id_raises(self, counts: Counter[str]) -> None:
        """A bad id means model and vocabulary disagree, which is a defect."""

        with pytest.raises(ValidationError):
            Vocabulary.from_counter(counts).token_of(9999)

    def test_encode_and_decode_round_trip(self, counts: Counter[str]) -> None:
        vocabulary = Vocabulary.from_counter(counts)

        tokens = ["the", "cat", "नमस्ते"]

        assert vocabulary.decode(vocabulary.encode(tokens)) == tokens

    def test_decode_skips_special_by_default(self, counts: Counter[str]) -> None:
        vocabulary = Vocabulary.from_counter(counts)

        ids = [BOS_ID, vocabulary.id_of("the"), EOS_ID]

        assert vocabulary.decode(ids) == ["the"]

        assert len(vocabulary.decode(ids, skip_special=False)) == 3

    def test_encode_substitutes_unknown(self, counts: Counter[str]) -> None:
        vocabulary = Vocabulary.from_counter(counts)

        assert vocabulary.encode(["absent"]) == [UNK_ID]

    def test_frequency_lookup(self, counts: Counter[str]) -> None:
        vocabulary = Vocabulary.from_counter(counts)

        assert vocabulary.frequency_of("the") == 10

        assert vocabulary.frequency_of("absent") == 0

    def test_coverage(self, counts: Counter[str]) -> None:
        vocabulary = Vocabulary.from_counter(counts)

        assert vocabulary.coverage(["the", "cat", "absent", "absent"]) == 0.5

    def test_coverage_of_empty_is_zero(self, counts: Counter[str]) -> None:
        assert Vocabulary.from_counter(counts).coverage([]) == 0.0

    def test_dunder_access(self, counts: Counter[str]) -> None:
        vocabulary = Vocabulary.from_counter(counts)

        assert vocabulary["the"] == vocabulary.id_of("the")

        assert "the" in vocabulary

        assert len(list(vocabulary)) == len(vocabulary)


class TestMutation:
    def test_add_new_token(self) -> None:
        vocabulary = Vocabulary()

        token_id = vocabulary.add("new")

        assert vocabulary.token_of(token_id) == "new"

    def test_add_existing_increments_frequency(self) -> None:
        vocabulary = Vocabulary()

        vocabulary.add("word", frequency=2)

        vocabulary.add("word", frequency=3)

        assert vocabulary.frequency_of("word") == 5

    def test_add_empty_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Vocabulary().add("")

    def test_frozen_vocabulary_rejects_addition(self) -> None:
        """
        Adding after training would create an id with no embedding row.
        """

        vocabulary = Vocabulary().freeze()

        assert vocabulary.is_frozen

        with pytest.raises(ValidationError):
            vocabulary.add("new")


class TestPersistence:
    def test_round_trip(self, counts: Counter[str], tmp_path: Path) -> None:
        vocabulary = Vocabulary.from_counter(counts)

        path = tmp_path / "vocabulary.json"

        vocabulary.save(path)

        reloaded = Vocabulary.load(path)

        assert reloaded == vocabulary

        assert reloaded.tokens() == vocabulary.tokens()

        assert reloaded.frequencies() == vocabulary.frequencies()

    def test_round_trip_preserves_non_latin(self, tmp_path: Path) -> None:
        vocabulary = Vocabulary.from_tokens(["नमस्ते", "こんにちは", "مرحبا"])

        path = tmp_path / "vocabulary.json"

        vocabulary.save(path)

        assert "नमस्ते" in Vocabulary.load(path)

    def test_version_mismatch_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Vocabulary.from_dict({"format_version": 999, "tokens": [], "frequencies": []})

    def test_length_mismatch_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Vocabulary.from_dict(
                {
                    "format_version": 1,
                    "special_tokens": ["<pad>", "<unk>", "<bos>", "<eos>"],
                    "tokens": ["<pad>"],
                    "frequencies": [],
                }
            )

    def test_missing_special_prefix_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Vocabulary.from_dict(
                {
                    "format_version": 1,
                    "special_tokens": ["<pad>", "<unk>", "<bos>", "<eos>"],
                    "tokens": ["wrong", "a", "b", "c"],
                    "frequencies": [1, 1, 1, 1],
                }
            )


class TestBuilder:
    def test_build_from_sequences(self) -> None:
        builder = VocabularyBuilder(min_count=2)

        builder.add_all([["a", "b"], ["a", "c"], ["a", "b"]])

        vocabulary = builder.build()

        assert "a" in vocabulary

        assert "c" not in vocabulary

    def test_totals(self) -> None:
        builder = VocabularyBuilder()

        builder.add_tokens(["a", "b", "a"])

        assert builder.total_tokens == 3

        assert builder.distinct_tokens == 2

    def test_empty_tokens_ignored(self) -> None:
        builder = VocabularyBuilder()

        builder.add_tokens(["a", "", "b"])

        assert builder.total_tokens == 2

    def test_pruning_reports_itself(self) -> None:
        """
        Approximate counts must be visible to the caller, not silent.
        """

        builder = VocabularyBuilder(max_tracked_tokens=2)

        builder.add_tokens(["a", "a", "b", "c", "d", "e"])

        assert builder.pruned_count > 0

    def test_max_size_is_honoured(self) -> None:
        builder = VocabularyBuilder(max_size=6)

        builder.add_tokens([f"token{index}" for index in range(50)])

        assert len(builder.build()) == 6
