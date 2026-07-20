"""
Pair mining.

The tests concentrate on lexical leakage, because that is the failure
this module exists to prevent and the only one that cannot be seen from
the training loss. A pair whose anchor words already appear in its
positive can be scored correctly by string matching, so a model trained
on such pairs shows a falling loss while learning nothing about meaning.

Everything else here — length filters, deduplication — is ordinary and
tested ordinarily.
"""

from __future__ import annotations

import pytest

from multilingual_embedding.core.exceptions import ValidationError
from multilingual_embedding.corpus.document import Document
from multilingual_embedding.corpus.pairs import (
    PairConfig,
    PairKind,
    mine_pairs,
    token_overlap,
)

LEAD = (
    "This is a sufficiently long opening paragraph about a subject, written "
    "so that it comfortably exceeds the minimum positive length required."
)

BODY = (
    "A second block of text belonging to a section, also written long enough "
    "to pass the length filter that rejects fragments rather than passages."
)


def article(
    *,
    identifier: str = "1",
    title: str = "An Article Title",
    text: str | None = None,
    sections: list[dict[str, str]] | None = None,
    language: str = "en",
) -> Document:
    """A document shaped as the Wikipedia extractor produces."""

    document = Document.from_text(
        text if text is not None else LEAD,
        identifier=identifier,
        language=language,
        title=title,
    )

    if sections is not None:
        document.metadata.base.attributes = {"sections": sections}

    return document


class TestTokenOverlap:
    def test_full_containment_scores_one(self) -> None:
        assert token_overlap("Mumbai", "Mumbai is a city") == 1.0

    def test_no_shared_words_scores_zero(self) -> None:
        assert token_overlap("climate", "Mumbai is a city") == 0.0

    def test_it_is_case_insensitive(self) -> None:
        """A title restated in a lead usually changes case."""

        assert token_overlap("Mumbai", "mumbai is a city") == 1.0

    def test_partial_overlap_is_proportional(self) -> None:
        assert token_overlap("Mumbai climate", "Mumbai is a city") == pytest.approx(0.5)

    def test_an_anchor_with_no_words_does_not_divide_by_zero(self) -> None:
        assert token_overlap("!!! ---", "some text here") == 0.0

    def test_it_works_without_whitespace_word_boundaries(self) -> None:
        """
        The measure must mean something for scripts that do not separate
        words with spaces, or it would silently report 0.0 for Japanese
        and pass every pair regardless of leakage.
        """

        assert token_overlap("機械学習", "機械学習は面白い") == 1.0


class TestWhatGetsMined:
    def test_title_and_lead_become_a_pair(self) -> None:
        pairs, _ = mine_pairs([article()])

        assert [p.kind for p in pairs] == [PairKind.TITLE_LEAD]

        assert pairs[0].anchor == "An Article Title"

    def test_heading_and_section_become_a_pair(self) -> None:
        pairs, _ = mine_pairs([article(sections=[{"heading": "Early History", "text": BODY}])])

        kinds = {p.kind for p in pairs}

        assert PairKind.HEADING_SECTION in kinds

    def test_adjacent_paragraphs_become_a_pair(self) -> None:
        pairs, _ = mine_pairs([article(text=f"{LEAD}\n\n{BODY}", title="")])

        assert [p.kind for p in pairs] == [PairKind.ADJACENT]

    def test_kinds_can_be_restricted(self) -> None:
        document = article(
            text=f"{LEAD}\n\n{BODY}",
            sections=[{"heading": "Early History", "text": BODY}],
        )

        pairs, _ = mine_pairs([document], PairConfig(kinds=(PairKind.TITLE_LEAD,)))

        assert {p.kind for p in pairs} == {PairKind.TITLE_LEAD}

    def test_provenance_is_recorded(self) -> None:
        """
        Two pairs from one article are about the same subject. A batch
        holding both teaches the model that a correct match is a
        negative, so a sampler needs to be able to tell.
        """

        pairs, _ = mine_pairs(
            [article(identifier="42", sections=[{"heading": "Early History", "text": BODY}])]
        )

        assert {p.document for p in pairs} == {"42"}

        assert {p.language for p in pairs} == {"en"}

    def test_a_corpus_without_structure_still_yields_pairs(self) -> None:
        """Plain text with no title and no sections still has adjacency."""

        document = Document.from_text(f"{LEAD}\n\n{BODY}", identifier="1")

        pairs, _ = mine_pairs([document])

        assert pairs

        assert {p.kind for p in pairs} == {PairKind.ADJACENT}


class TestLexicalLeakage:
    """
    The reason this module exists.

    Measured on the Meetei Mayek Wikipedia, title/lead pairs average
    0.89 overlap and 70% of them exceed 0.9 — the most obvious pair
    source is also the most contaminated.
    """

    def test_overlap_is_recorded_on_every_pair(self) -> None:
        pairs, _ = mine_pairs([article(title="opening paragraph")])

        assert pairs[0].overlap > 0.0

    def test_a_leaky_pair_can_be_rejected(self) -> None:
        """The title is restated verbatim, which is the Wikipedia norm."""

        document = article(title="Subject Matter", text="Subject Matter " + LEAD)

        kept, statistics = mine_pairs(
            [document], PairConfig(maximum_overlap=0.5, kinds=(PairKind.TITLE_LEAD,))
        )

        assert kept == []

        assert statistics.rejected_overlap == 1

    def test_a_clean_pair_survives_the_same_filter(self) -> None:
        """The filter must discriminate, not merely reject."""

        document = article(title="Something Entirely Different", text=LEAD)

        kept, _ = mine_pairs(
            [document], PairConfig(maximum_overlap=0.5, kinds=(PairKind.TITLE_LEAD,))
        )

        assert len(kept) == 1

    def test_mean_overlap_is_reported_per_kind(self) -> None:
        """
        Per kind rather than overall, because the kinds differ sharply
        and an average across them hides which one is the problem.
        """

        document = article(
            text=f"{LEAD}\n\n{BODY}",
            sections=[{"heading": "Early History", "text": BODY}],
        )

        _, statistics = mine_pairs([document])

        assert set(statistics.mean_overlap_by_kind) == set(statistics.by_kind)

        assert all(0.0 <= v <= 1.0 for v in statistics.mean_overlap_by_kind.values())

    def test_a_leaky_kind_is_warned_about(self) -> None:
        """
        Silence here would be the worst outcome: the loss falls, the
        model looks trained, and nobody learns that the task was
        solvable by string matching.

        A handler is attached directly rather than using ``caplog``.
        The framework's loggers set ``propagate = False`` on purpose, so
        records never reach the root logger that ``caplog`` inspects —
        which makes ``caplog`` silently pass on any assertion of the form
        "no warning was logged".
        """

        import logging

        from multilingual_embedding.corpus import pairs as pairs_module

        records: list[logging.LogRecord] = []

        class Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        handler = Capture(level=logging.WARNING)

        pairs_module._logger.addHandler(handler)

        try:
            document = article(title="Subject Matter", text="Subject Matter " + LEAD)

            mine_pairs([document], PairConfig(kinds=(PairKind.TITLE_LEAD,)))
        finally:
            pairs_module._logger.removeHandler(handler)

        assert any("string matching" in record.getMessage() for record in records)


class TestQualityFilters:
    def test_short_anchors_are_rejected(self) -> None:
        _, statistics = mine_pairs(
            [article(title="ab", sections=None)],
            PairConfig(kinds=(PairKind.TITLE_LEAD,)),
        )

        assert statistics.rejected_short_anchor == 1

    def test_short_positives_are_rejected(self) -> None:
        _, statistics = mine_pairs(
            [article(text="Too short to be a passage.")],
            PairConfig(kinds=(PairKind.TITLE_LEAD,)),
        )

        assert statistics.rejected_short_positive == 1

    def test_long_positives_are_truncated_not_dropped(self) -> None:
        document = article(text="word " * 2000)

        pairs, _ = mine_pairs([document], PairConfig(maximum_positive_characters=500))

        assert pairs

        assert len(pairs[0].positive) <= 500

    def test_identical_pairs_are_deduplicated(self) -> None:
        documents = [article(identifier="1"), article(identifier="2")]

        pairs, statistics = mine_pairs(documents, PairConfig(kinds=(PairKind.TITLE_LEAD,)))

        assert len(pairs) == 1

        assert statistics.rejected_duplicate == 1

    def test_whitespace_is_normalised(self) -> None:
        """Otherwise trivially different spacing defeats deduplication."""

        pairs, _ = mine_pairs(
            [article(title="An   Article\n Title")],
            PairConfig(kinds=(PairKind.TITLE_LEAD,)),
        )

        assert pairs[0].anchor == "An Article Title"


class TestConfigurationIsValidated:
    @pytest.mark.parametrize(
        "settings",
        [
            {"minimum_anchor_characters": 0},
            {"minimum_positive_characters": 0},
            {"maximum_overlap": 1.5},
            {"maximum_overlap": -0.1},
            {"kinds": ("nonsense",)},
            {"minimum_positive_characters": 500, "maximum_positive_characters": 100},
        ],
    )
    def test_bad_settings_fail_at_construction(self, settings: dict[str, object]) -> None:
        with pytest.raises(ValidationError):
            PairConfig(**settings)  # type: ignore[arg-type]


class TestStatisticsAreReportable:
    def test_statistics_serialise(self) -> None:
        import json

        _, statistics = mine_pairs([article()])

        json.dumps(statistics.to_dict())

    def test_a_pair_serialises_to_the_trainer_format(self) -> None:
        pairs, _ = mine_pairs([article()])

        record = pairs[0].to_record()

        assert set(record) == {
            "anchor",
            "positive",
            "kind",
            "document",
            "language",
            "overlap",
        }
