"""
The ``qfme mine-negatives`` wiring.

Parser-level only. The mining itself is covered against planted geometry
in :mod:`tests.embedding.test_negatives`, and the handler's remaining
job — loading an adapter and reading its prefixes off the manifest —
needs a real checkpoint and belongs with the tests that have one.

The defaults are the interesting part. Every one of them is a decision
about what to throw away, and a default that quietly kept more would
produce more negatives, a better-looking loss curve and a worse model.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from multilingual_embedding.cli import build_parser


@pytest.fixture(scope="module")
def parse():  # type: ignore[no-untyped-def]
    parser = build_parser()

    return lambda *argv: parser.parse_args(["mine-negatives", *argv])


REQUIRED = ("--pairs", "p.jsonl", "--adapter", "models/indic-v1", "--output", "out.jsonl")


class TestWhatIsRequired:
    @pytest.mark.parametrize("missing", ["--pairs", "--adapter", "--output"])
    def test_each_required_flag_is_required(self, missing: str) -> None:
        argv = list(REQUIRED)

        index = argv.index(missing)

        del argv[index : index + 2]

        with pytest.raises(SystemExit):
            build_parser().parse_args(["mine-negatives", *argv])

    def test_paths_arrive_as_paths(self, parse) -> None:  # type: ignore[no-untyped-def]
        args = parse(*REQUIRED)

        assert args.pairs == Path("p.jsonl")

        assert args.adapter == Path("models/indic-v1")

        assert args.output == Path("out.jsonl")


class TestTheDefaults:
    def test_four_negatives_per_pair(self, parse) -> None:  # type: ignore[no-untyped-def]
        assert parse(*REQUIRED).count == 4

    def test_the_ceiling_is_on_by_default(self, parse) -> None:  # type: ignore[no-untyped-def]
        """
        Off would mean mining the candidates most likely to be correct
        answers, which is the one setting that reliably makes a model
        worse while the loss curve improves.
        """

        assert parse(*REQUIRED).max_similarity == 0.95

    def test_the_floor_discards_what_in_batch_sampling_already_supplies(self, parse) -> None:  # type: ignore[no-untyped-def]
        """
        A candidate below zero is one the model already separates, so
        storing it widens every training batch for nothing. Lower it
        when mining against a checkpoint too weak to rank anything above
        the floor — unencodable text is caught by norm, not by this.
        """

        assert parse(*REQUIRED).min_similarity == 0.0

    def test_the_provenance_guard_is_on_by_default(self, parse) -> None:  # type: ignore[no-untyped-def]
        """
        Two passages from one article are usually about one subject, so
        keeping them has to be asked for rather than fallen into.
        """

        assert parse(*REQUIRED).allow_same_document is False

    def test_the_guard_can_be_turned_off(self, parse) -> None:  # type: ignore[no-untyped-def]
        assert parse(*REQUIRED, "--allow-same-document").allow_same_document is True

    def test_nothing_is_written_unless_asked(self, parse) -> None:  # type: ignore[no-untyped-def]
        args = parse(*REQUIRED)

        assert args.audit is None

        assert args.report is None

        assert args.limit is None

    def test_the_network_is_allowed_unless_refused(self, parse) -> None:  # type: ignore[no-untyped-def]
        assert parse(*REQUIRED).local_files_only is False


class TestTrialRuns:
    def test_a_sample_can_be_taken(self, parse) -> None:  # type: ignore[no-untyped-def]
        """
        Mining cannot stream — the candidate pool is the pair set's own
        positives, so everything has to be resident and encoded first.
        A sample is how the filters get inspected before an hour is
        spent on the full set.
        """

        assert parse(*REQUIRED, "--limit", "5000").limit == 5000

    def test_the_audit_file_has_a_bounded_size(self, parse) -> None:  # type: ignore[no-untyped-def]
        assert parse(*REQUIRED, "--audit", "a.jsonl").audit_sample == 50
