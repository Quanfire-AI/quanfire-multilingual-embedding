"""
Mining pairs from a corpus that is not Wikipedia.

Every pair-mining result this project has published came from a
MediaWiki dump, which leaves an obvious question open: is the miner
coupled to Wikipedia, or does it merely happen to have been pointed at
one? `ROADMAP.md` Phase C depends on the second answer being true, and
until this module existed nothing checked it.

Nothing here trains anything. These tests pin the *contract* a domain
export has to satisfy, so that an exporter written against it produces a
file the miner can consume — and so that a change to the reader or the
miner which quietly breaks that contract fails here rather than on a
training box.

**On the fixture.** `data/sample/domain-corpus.jsonl` is synthetic
professional-services text written for this purpose. It demonstrates
that the path runs and fixes the record shape; it is not evidence about
how real client documents behave, and the overlap figures it produces
are a hypothesis to check against a real export rather than a result.
That distinction matters enough to state twice, so it is stated again in
the one test that measures overlap.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from multilingual_embedding.corpus.pairs import PairConfig, PairKind, PairStatistics, iter_pairs
from multilingual_embedding.corpus.reader import JsonlReader

CORPUS = Path(__file__).resolve().parents[2] / "data" / "sample" / "domain-corpus.jsonl"


@pytest.fixture(scope="module")
def documents() -> list:
    """The domain corpus, read exactly as `qfme mine-pairs --source` reads it."""

    return list(JsonlReader(CORPUS).iter_documents())


@pytest.fixture(scope="module")
def mined(documents: list) -> tuple[list, PairStatistics]:
    statistics = PairStatistics()

    pairs = list(iter_pairs(documents, statistics=statistics))

    return pairs, statistics


class TestTheFixtureItself:
    """The corpus has to be a corpus before it can be a test of anything."""

    def test_the_file_is_committed(self) -> None:
        assert CORPUS.exists(), (
            "data/sample/ is one of the few paths not gitignored under data/; "
            "the domain corpus belongs there so this test runs on a fresh clone"
        )

    def test_every_record_parses_as_json(self) -> None:
        for line in CORPUS.read_text(encoding="utf-8").splitlines():
            if line.strip():
                json.loads(line)  # a raise here names the offending line

    def test_it_is_not_wikipedia(self, documents: list) -> None:
        sources = {document.metadata.base.attributes.get("source") for document in documents}

        assert sources == {"synthetic-domain-sample"}

    def test_it_carries_more_than_one_language(self, documents: list) -> None:
        languages = {document.metadata.base.language for document in documents}

        assert "en" in languages and "hi" in languages, (
            "a domain corpus that is English-only would not exercise the part "
            "of the overlap measure that switches to character bigrams"
        )


class TestTheRecordShape:
    """
    Where `sections` has to live, which is the trap.

    `Document.from_dict` nests extra fields under an `attributes` key,
    while `JsonlReader` — the reader `qfme mine-pairs` actually uses —
    flattens every unrecognised top-level field into the attributes
    mapping. So `sections` goes at the top level of the record, as the
    Wikipedia extractor's `to_record` writes it. Nested under an
    `attributes` key it is silently invisible: mining succeeds, and
    produces no `heading_section` pairs at all.

    That failure has no error message, which is why it gets a test.
    """

    def test_sections_are_read_from_the_top_level(self, documents: list) -> None:
        with_sections = [
            document for document in documents if document.metadata.base.attributes.get("sections")
        ]

        assert len(with_sections) >= 5

    def test_a_section_is_a_heading_and_a_text(self, documents: list) -> None:
        for document in documents:
            for section in document.metadata.base.attributes.get("sections") or []:
                assert set(section) == {"heading", "text"}

                assert section["heading"].strip()

                assert section["text"].strip()

    def test_titles_survive_the_reader(self, documents: list) -> None:
        assert all(document.metadata.title for document in documents)

    def test_nesting_sections_under_attributes_yields_no_section_pairs(self) -> None:
        """The trap, demonstrated rather than described."""

        record = json.loads(CORPUS.read_text(encoding="utf-8").splitlines()[0])

        sections = record.pop("sections")

        record["attributes"] = {"sections": sections}

        nested = CORPUS.parent / "_nested.jsonl"

        try:
            nested.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")

            pairs = list(iter_pairs(JsonlReader(nested).iter_documents()))

            assert not [pair for pair in pairs if pair.kind == PairKind.HEADING_SECTION]
        finally:
            nested.unlink(missing_ok=True)


class TestMining:
    """The claim Phase C rests on: structure the miner understands is not Wikipedia-specific."""

    def test_all_three_kinds_are_produced(self, mined: tuple[list, PairStatistics]) -> None:
        pairs, _ = mined

        kinds = {pair.kind for pair in pairs}

        assert kinds == {
            PairKind.TITLE_LEAD,
            PairKind.HEADING_SECTION,
            PairKind.ADJACENT,
        }

    def test_it_produces_more_pairs_than_documents(
        self, documents: list, mined: tuple[list, PairStatistics]
    ) -> None:
        """
        The multiplier is the whole economic argument for structural mining.

        Ten documents yielding ten pairs would not be worth a pipeline.
        """

        pairs, _ = mined

        assert len(pairs) > 3 * len(documents)

    def test_every_pair_traces_back_to_its_document(
        self, documents: list, mined: tuple[list, PairStatistics]
    ) -> None:
        """
        Two pairs from one document are false negatives for each other.

        A sampler can only keep them apart if the provenance survives,
        and a domain corpus makes this worse than Wikipedia does: one
        matter can produce dozens of closely related pairs.
        """

        pairs, _ = mined

        identifiers = {document.identifier for document in documents}

        assert all(pair.document in identifiers for pair in pairs)

    def test_the_hindi_document_produces_hindi_pairs(
        self, mined: tuple[list, PairStatistics]
    ) -> None:
        pairs, _ = mined

        hindi = [pair for pair in pairs if pair.language == "hi"]

        assert hindi

        assert all(pair.overlap <= 1.0 for pair in hindi)

    def test_rejections_are_accounted_for(self, mined: tuple[list, PairStatistics]) -> None:
        """Nothing may vanish; a dropped pair is dropped for a named reason."""

        _, statistics = mined

        counts = statistics.to_dict()

        assert set(counts["rejected"]) == {
            "short_anchor",
            "short_positive",
            "overlap",
            "duplicate",
        }

        assert counts["produced"] + sum(counts["rejected"].values()) > counts["produced"]


class TestLexicalLeakage:
    """
    The measurement that decides whether mined pairs are worth training on.

    Restated, because it is the easiest thing in this file to
    over-read: the corpus is synthetic and was written by someone who
    knew overlap would be measured. What follows is a property of this
    fixture. It is a hypothesis about real business documents, not a
    finding about them, and the way to settle it is to run
    `qfme mine-pairs --report` over a real export.
    """

    def test_overlap_is_measured_for_every_pair(self, mined: tuple[list, PairStatistics]) -> None:
        pairs, _ = mined

        assert all(0.0 <= pair.overlap <= 1.0 for pair in pairs)

    def test_title_lead_leaks_far_less_than_it_does_on_wikipedia(
        self, mined: tuple[list, PairStatistics]
    ) -> None:
        """
        Hindi Wikipedia's `title_lead` pairs average 0.977 overlap, because
        an encyclopedia lead restates its title by convention. Business
        prose has no such convention — an invoice's payment-terms section
        does not open by repeating the words "payment terms" — so the
        single most contaminated pair source on Wikipedia is not obviously
        contaminated here.

        If that survives contact with a real export it is the most useful
        thing this fixture has to say, because leakage was the dominant
        difficulty in all of the Wikipedia work.
        """

        pairs, _ = mined

        leads = [pair.overlap for pair in pairs if pair.kind == PairKind.TITLE_LEAD]

        assert leads

        assert sum(leads) / len(leads) < 0.7

    def test_the_overlap_filter_bites(self, documents: list) -> None:
        """
        A permissive default and a strict setting must not agree, or the
        filter is decorative and the band breakdown means nothing.
        """

        strict = PairStatistics()

        list(iter_pairs(documents, PairConfig(maximum_overlap=0.15), statistics=strict))

        permissive = PairStatistics()

        list(iter_pairs(documents, PairConfig(maximum_overlap=1.0), statistics=permissive))

        assert strict.produced < permissive.produced

        assert strict.rejected_overlap > 0

        assert permissive.rejected_overlap == 0
