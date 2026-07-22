"""
Reading a mined pair file back.

Mining writes millions of pairs; training reads a few thousand of them.
The step in between is a sampler, and it is the part of the adaptation
path most able to be wrong without saying so — a biased sample produces
a report about a subset of the corpus while claiming to be about the
corpus, and every number in it is internally consistent.

That is not hypothetical. The first implementation read the leading
``count * 4`` lines and shuffled those. A mined pair file is in corpus
order, and two concatenated files are in language order: for a Hindi and
Tamil pair set joined together, the window covered the first 168,000 of
642,536 Hindi lines and never reached a Tamil pair. The run reported
``by_language: {"hi": ...}`` and was read as a joint experiment.

So the test that matters here is not "it returns pairs". It is that a
pair sitting at the very end of the file can be selected at all.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from multilingual_embedding.core.exceptions import ValidationError
from multilingual_embedding.corpus.pairs import MinedPair, sample_pairs


def write(path: Path, records: list[dict[str, object]]) -> Path:
    """Write records as JSON Lines, gzipped when the suffix says so."""

    opener = gzip.open if path.suffix == ".gz" else open

    with opener(path, "wt", encoding="utf-8") as handle:  # type: ignore[operator]
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    return path


def ordered(count: int, language: str = "hi") -> list[dict[str, object]]:
    """Pairs numbered in order, so a sample's positions are readable."""

    return [
        MinedPair(
            anchor=f"anchor {index}",
            positive=f"positive {index} at some length",
            kind="adjacent",
            document=f"doc-{index}",
            language=language,
            overlap=0.1,
        ).to_record()
        for index in range(count)
    ]


class TestFromRecord:
    """What a pair file is allowed to leave out."""

    def test_a_full_record_round_trips(self) -> None:
        original = MinedPair(
            anchor="किरायेदारी विवाद",
            positive="यह मामला लघु वाद न्यायालय में लंबित है।",
            kind="title_lead",
            document="matter-3104",
            language="hi",
            overlap=0.2341,
        )

        assert MinedPair.from_record(original.to_record()) == original

    def test_provenance_may_be_absent(self) -> None:
        """
        A hand-written evaluation set is the natural first artefact for a
        domain nobody has mined yet, and requiring six fields to score
        twenty pairs would make that harder than writing the encoder.
        """

        pair = MinedPair.from_record({"anchor": "a query", "positive": "an answer"})

        assert pair.kind == ""

        assert pair.document == ""

        assert pair.language is None

        assert pair.overlap == 0.0

    def test_what_is_absent_is_not_guessed(self) -> None:
        """
        An empty kind must not match a kind filter.

        Defaulting the missing fields to something plausible — the
        commonest kind, the file's language — would make a filtered run
        train on pairs it was told to exclude, and nothing would say so.
        """

        pair = MinedPair.from_record({"anchor": "a query", "positive": "an answer"})

        assert pair.kind not in {"adjacent", "title_lead", "heading_section"}

    @pytest.mark.parametrize(
        "record",
        [
            {"positive": "an answer"},
            {"anchor": "a query"},
            {"anchor": "", "positive": "an answer"},
            {"anchor": "a query", "positive": "   "},
        ],
    )
    def test_a_half_pair_is_refused(self, record: dict[str, object]) -> None:
        """A pair missing one side trains the model on nothing, silently."""

        with pytest.raises(ValidationError):
            MinedPair.from_record(record)


class TestSampling:
    def test_it_draws_the_requested_count(self, tmp_path: Path) -> None:
        path = write(tmp_path / "pairs.jsonl", ordered(500))

        assert len(sample_pairs(path, 50, seed=1)) == 50

    def test_a_short_file_yields_all_of_it(self, tmp_path: Path) -> None:
        """
        Not an error. The caller checks the length and says so — the
        pipeline prints "only N pairs available" — because a file being
        smaller than expected is a fact about the corpus, not a fault.
        """

        path = write(tmp_path / "pairs.jsonl", ordered(30))

        assert len(sample_pairs(path, 100, seed=1)) == 30

    def test_the_same_seed_draws_the_same_pairs(self, tmp_path: Path) -> None:
        path = write(tmp_path / "pairs.jsonl", ordered(400))

        assert sample_pairs(path, 40, seed=7) == sample_pairs(path, 40, seed=7)

    def test_a_different_seed_draws_different_pairs(self, tmp_path: Path) -> None:
        path = write(tmp_path / "pairs.jsonl", ordered(400))

        first = {pair.document for pair in sample_pairs(path, 40, seed=1)}

        second = {pair.document for pair in sample_pairs(path, 40, seed=2)}

        assert first != second

    def test_the_tail_of_the_file_is_reachable(self, tmp_path: Path) -> None:
        """
        The regression this function exists for.

        A window over the head of the file cannot select anything past
        it, and the failure is invisible: the run reports on whatever it
        did see, consistently. Here the last tenth of the file must be
        represented, across seeds.
        """

        path = write(tmp_path / "pairs.jsonl", ordered(2_000))

        seen_in_tail = 0

        for seed in range(5):
            drawn = sample_pairs(path, 100, seed=seed)

            positions = [int(pair.document.split("-")[1]) for pair in drawn]

            seen_in_tail += sum(1 for position in positions if position >= 1_800)

        assert seen_in_tail > 0, "no pair from the final tenth of the file was ever drawn"

    def test_two_concatenated_languages_are_both_represented(self, tmp_path: Path) -> None:
        """
        The exact shape that produced the wrong result.

        Concatenating a Hindi pair file and a Tamil one puts every Tamil
        pair after every Hindi pair. A head-window sampler reports
        `by_language: {"hi": ...}` on what was set up as a joint run.
        """

        path = write(tmp_path / "joint.jsonl", ordered(1_500, "hi") + ordered(500, "ta"))

        languages = {pair.language for pair in sample_pairs(path, 200, seed=3)}

        assert languages == {"hi", "ta"}

    def test_gzip_is_read_by_extension(self, tmp_path: Path) -> None:
        """Mined pair files are written compressed; nothing should have to say so."""

        path = write(tmp_path / "pairs.jsonl.gz", ordered(200))

        assert len(sample_pairs(path, 20, seed=1)) == 20

    def test_blank_lines_are_skipped(self, tmp_path: Path) -> None:
        """A trailing newline is not a pair, and must not become one."""

        path = tmp_path / "pairs.jsonl"

        write(path, ordered(20))

        path.write_text(path.read_text(encoding="utf-8") + "\n\n", encoding="utf-8")

        assert len(sample_pairs(path, 100, seed=1)) == 20
