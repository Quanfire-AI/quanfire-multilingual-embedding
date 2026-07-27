"""
Mining aligned cross-lingual pairs.

Two properties carry the module's purpose and get the most attention.
First, both directions are emitted from one aligned article, so a mixed
set queries in both languages rather than only the source's. Second, the
positive's language lands in the single ``language`` field the retrieval
evaluator groups by, while both per-side languages survive on the
record. The join accounting — why a source document failed to align — is
tested too, because a silent join failure shrinks the evaluation set
without complaint.
"""

from __future__ import annotations

import pytest

from multilingual_embedding.core.exceptions import ValidationError
from multilingual_embedding.corpus.aligned import (
    AlignedDocument,
    AlignedPairConfig,
    AlignedPairKind,
    AlignedStatistics,
    iter_aligned_documents,
    iter_aligned_pairs,
    to_aligned_document,
)
from multilingual_embedding.corpus.document import Document
from multilingual_embedding.corpus.pairs import MinedPair

# Long enough to clear the positive-length floor; different scripts so
# cross-lingual overlap is genuinely zero.
HI_LEAD = (
    "भारत एक विशाल देश है जिसकी संस्कृति और इतिहास बहुत पुराना और समृद्ध है। "
    "यह पैराग्राफ इतना लंबा है कि यह न्यूनतम लंबाई की शर्त को आसानी से पूरा करता है।"
)

TA_LEAD = (
    "இந்தியா ஒரு பரந்த நாடு, அதன் கலாச்சாரமும் வரலாறும் மிகவும் பழமையானது. "
    "இந்த பத்தி குறைந்தபட்ச நீளத் தேவையை எளிதாக பூர்த்தி செய்யும் அளவுக்கு நீளமானது."
)


def hi() -> AlignedDocument:
    return AlignedDocument(identifier="12", title="भारत", language="hi", lead=HI_LEAD)


def ta() -> AlignedDocument:
    return AlignedDocument(identifier="99", title="இந்தியா", language="ta", lead=TA_LEAD)


class TestToAlignedDocument:
    def test_it_takes_the_first_paragraph_as_the_lead(self) -> None:
        document = Document.from_text(
            f"{HI_LEAD}\n\nA second paragraph that should not be the lead.",
            identifier="12",
            language="hi",
            title="भारत",
        )

        view = to_aligned_document(document)

        assert view is not None

        assert view.lead.startswith("भारत")

    def test_no_title_means_no_view(self) -> None:
        document = Document.from_text(HI_LEAD, identifier="12", language="hi")

        assert to_aligned_document(document) is None

    def test_no_lead_means_no_view(self) -> None:
        document = Document.from_text("   ", identifier="12", language="hi", title="भारत")

        assert to_aligned_document(document) is None

    def test_the_language_fallback_fills_a_missing_code(self) -> None:
        """Devanagari is shared, so a document's language may be absent."""

        document = Document.from_text(HI_LEAD, identifier="12", title="भारत", detect_language=False)

        view = to_aligned_document(document, language="hi")

        assert view is not None

        assert view.language == "hi"


class TestJoin:
    def test_a_linked_document_aligns(self) -> None:
        source = Document.from_text(HI_LEAD, identifier="12", language="hi", title="भारत")

        target_index = {"இந்தியா".casefold(): ta()}

        stats = AlignedStatistics()

        pairs = list(
            iter_aligned_documents([source], target_index, {"12": "இந்தியா"}, statistics=stats)
        )

        assert len(pairs) == 1

        assert stats.aligned == 1

    def test_no_interlanguage_link_is_counted(self) -> None:
        source = Document.from_text(HI_LEAD, identifier="12", language="hi", title="भारत")

        stats = AlignedStatistics()

        pairs = list(iter_aligned_documents([source], {}, {}, statistics=stats))

        assert pairs == []

        assert stats.missing_langlink == 1

        assert stats.missing_target == 0

    def test_a_link_to_an_absent_target_is_counted(self) -> None:
        source = Document.from_text(HI_LEAD, identifier="12", language="hi", title="भारत")

        stats = AlignedStatistics()

        # The link exists, but the target corpus has no such title.
        pairs = list(iter_aligned_documents([source], {}, {"12": "இந்தியா"}, statistics=stats))

        assert pairs == []

        assert stats.missing_target == 1

        assert stats.missing_langlink == 0

    def test_a_titleless_source_is_not_a_join_failure(self) -> None:
        """It cannot form a pair, but it did not fail to align — do not miscount."""

        source = Document.from_text(HI_LEAD, identifier="12", language="hi")

        stats = AlignedStatistics()

        list(iter_aligned_documents([source], {}, {"12": "x"}, statistics=stats))

        assert stats.source_documents == 0

        assert stats.missing_langlink == 0


class TestBothDirections:
    def test_each_article_queries_in_both_languages(self) -> None:
        pairs = list(iter_aligned_pairs([(hi(), ta())]))

        query_languages = {pair.anchor_language for pair in pairs}

        assert query_languages == {"hi", "ta"}

    def test_the_scored_language_is_the_positive_side(self) -> None:
        pairs = list(iter_aligned_pairs([(hi(), ta())]))

        for pair in pairs:
            record = pair.to_record()

            assert record["language"] == pair.positive_language

            assert record["language"] != pair.anchor_language

    def test_both_languages_survive_on_the_record(self) -> None:
        pairs = list(iter_aligned_pairs([(hi(), ta())]))

        record = pairs[0].to_record()

        assert "anchor_language" in record

        assert "positive_language" in record


class TestKinds:
    def test_default_emits_title_lead_and_lead_lead_both_ways(self) -> None:
        pairs = list(iter_aligned_pairs([(hi(), ta())]))

        kinds = {pair.kind for pair in pairs}

        assert kinds == {AlignedPairKind.TITLE_LEAD, AlignedPairKind.LEAD_LEAD}

        # title/lead ×2 directions + lead/lead ×2 directions.
        assert len(pairs) == 4

    def test_selecting_one_kind_drops_the_other(self) -> None:
        config = AlignedPairConfig(kinds=(AlignedPairKind.TITLE_LEAD,))

        pairs = list(iter_aligned_pairs([(hi(), ta())], config))

        assert {pair.kind for pair in pairs} == {AlignedPairKind.TITLE_LEAD}

        assert len(pairs) == 2


class TestQualityRules:
    def test_a_short_title_anchor_is_still_kept(self) -> None:
        """A bare title is a real cross-lingual query; the floor is low."""

        pairs = list(iter_aligned_pairs([(hi(), ta())]))

        title_anchors = [pair.anchor for pair in pairs if pair.kind == AlignedPairKind.TITLE_LEAD]

        # "भारत" is four characters — above the floor of three, below the
        # monolingual miner's eight.
        assert "भारत" in title_anchors

    def test_a_short_positive_is_rejected(self) -> None:
        short = AlignedDocument(identifier="99", title="இந்தியா", language="ta", lead="குறு")

        stats = AlignedStatistics()

        list(iter_aligned_pairs([(hi(), short)], statistics=stats))

        assert stats.rejected_short_positive > 0

    def test_identical_text_is_written_once(self) -> None:
        stats = AlignedStatistics()

        # The same aligned article twice must not double the pairs.
        list(iter_aligned_pairs([(hi(), ta()), (hi(), ta())], statistics=stats))

        assert stats.rejected_duplicate > 0

    def test_cross_script_overlap_is_zero(self) -> None:
        """Different alphabets share no units — this is the whole point."""

        pairs = list(iter_aligned_pairs([(hi(), ta())]))

        assert all(pair.overlap == 0.0 for pair in pairs)

    def test_the_document_id_pairs_both_sides(self) -> None:
        """Two pairs from one article share a document id, so a sampler keeps them apart."""

        pairs = list(iter_aligned_pairs([(hi(), ta())]))

        assert all("12" in pair.document and "99" in pair.document for pair in pairs)


class TestRecordRoundTrips:
    def test_it_loads_as_an_ordinary_mined_pair(self) -> None:
        """An aligned file must be readable wherever the richer view is not needed."""

        pair = next(iter(iter_aligned_pairs([(hi(), ta())])))

        loaded = MinedPair.from_record(pair.to_record())

        assert loaded.anchor == pair.anchor

        assert loaded.language == pair.positive_language


class TestConfigValidation:
    def test_an_unknown_kind_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            AlignedPairConfig(kinds=("title_lead",))  # the monolingual name, not ours

    def test_a_positive_floor_above_the_ceiling_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            AlignedPairConfig(minimum_positive_characters=2000, maximum_positive_characters=100)
