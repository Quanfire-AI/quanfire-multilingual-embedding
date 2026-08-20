"""
Reading Indian import/export material into the training corpus.

Trade has two shapes. Layers B/C (notifications, FTP) arrive as flat
``<number, date, subject, body>`` rows, so — as for KCC — the property that
makes the output trainable is not "find every section" but "remove the feed's
noise without over-cleaning terse-but-real subjects". That filter is pinned
oracle-diff style: the disciplined pass is diffed against a deliberately naive
reference (:meth:`TradeFilterConfig.naive`) on a fixture carrying one row of
every noise category, and the two are required to *disagree* by exactly those
rows — the removals reconcile to the naive population exactly, so the junk
removal is loud and measured instead of an opaque "fewer pairs". Each category
also has its own pinning test so a regression names the category it broke.

Layer A (the bilingual Acts) is pinned differently: the cross-lingual helper
must emit both directions of an aligned section, symmetric, with empty E5
prefixes — the register decision the pair design turns on. The two §52 legal
bases (q(ii) for Acts, q(i) for Gazette matter) and the attribution are asserted
present so a mechanical pipeline cannot strip the provenance off the text.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from multilingual_embedding.corpus.annotated_acts import StatuteSection
from multilingual_embedding.corpus.trade import (
    TRADE_ATTRIBUTION,
    TRADE_LEGAL_BASIS_ACT,
    TRADE_LEGAL_BASIS_GAZETTE,
    TRADE_LICENSE,
    TRADE_NOTIFICATION_KIND,
    TRADE_SECTION_XLING_KIND,
    TRADE_SOURCE,
    BilingualAlignmentStatistics,
    BilingualSection,
    TradeFilterConfig,
    TradeNotification,
    TradeReadError,
    TradeStatistics,
    align_bilingual_sections,
    extract_notification_pairs,
    iter_bilingual_section_pairs,
    iter_notification_pairs,
    prefix_regime,
    read_notification_rows,
)


def _note(subject: str, body: str, **facets: str) -> TradeNotification:
    return TradeNotification(subject=subject, body=body, **facets)


# A genuine notification: a terse-but-contentful subject over a long operative
# body. body->subject overlap is low (the body says far more than the headline),
# so the pair is not a stub. Kept by both the filter and the naive reference.
_GENUINE = _note(
    "Revised tariff value of crude palm oil",
    "The Central Government hereby lowers the levy chargeable on imported edible "
    "commodities to support domestic prices, effective across all customs "
    "stations with immediate effect.",
    notification_number="34/2024-Customs",
    instrument="notification",
    layer="B",
    year="2024",
)


def test_genuine_notification_pair_is_kept_and_well_formed() -> None:
    stats = TradeStatistics()

    pairs = list(iter_notification_pairs([_GENUINE], TradeFilterConfig(), stats))

    assert len(pairs) == 1
    pair = pairs[0]
    assert pair.anchor == "Revised tariff value of crude palm oil"
    assert pair.positive.startswith("The Central Government")
    assert pair.kind == TRADE_NOTIFICATION_KIND
    # Provenance rides in the document id in place of a per-row licence field.
    assert pair.document == f"{TRADE_SOURCE}:34/2024-Customs"
    assert stats.produced == 1
    assert stats.rejected == {}


def test_headline_subject_over_rich_body_is_kept() -> None:
    # Regression (real-data catch, 2026-08-15): a live DGFT trade notice's subject
    # is a headline drawn from the body's own words, so subject->body containment
    # ran to 0.84 and the old subject->body cap wrongly cut it. The body adds far
    # more than the headline (body->subject was 0.05), so it MUST be kept. This is
    # the common shape of a genuine notification, not an edge case.
    stats = TradeStatistics()
    rows = [
        _note(
            "Standard operating procedure for reporting inward remittances by NBFC factors",
            "This trade notice lays down the standard operating procedure for reporting "
            "of inward remittance messages pertaining to NBFC factors. Exporters and "
            "authorised dealers shall follow the reconciliation steps set out below, "
            "uploading each realisation against the corresponding shipping bill within "
            "the timelines the Directorate specifies, so that electronic bank realisation "
            "certificates are generated without manual intervention.",
        )
    ]
    pairs = list(iter_notification_pairs(rows, TradeFilterConfig(), stats))
    assert len(pairs) == 1
    assert stats.rejected == {}
    # The stored overlap is body->subject redundancy: low, because the body is rich.
    assert pairs[0].overlap < 0.3


def test_notification_pair_trains_with_asymmetric_prefixes() -> None:
    # A notification is a short query against a long passage, so it takes the
    # non-empty E5 prefixes — the asymmetric half of the pair-design decision.
    assert prefix_regime(TRADE_NOTIFICATION_KIND) == ("query: ", "passage: ")


def test_withdrawn_notification_is_dropped() -> None:
    stats = TradeStatistics()
    rows = [
        _note(
            "Rescission of notification 5/2023-Customs",
            "Notification 5/2023-Customs is hereby withdrawn with effect from the "
            "date of publication in the Gazette.",
        )
    ]
    assert list(iter_notification_pairs(rows, TradeFilterConfig(), stats)) == []
    assert stats.rejected["withdrawn"] == 1


def test_superseded_notification_is_dropped() -> None:
    stats = TradeStatistics()
    rows = [
        _note(
            "Valuation guidance for imported machinery",
            "The earlier circular on this subject stands superseded by the "
            "revised procedure set out in the enclosed guidelines below.",
        )
    ]
    assert list(iter_notification_pairs(rows, TradeFilterConfig(), stats)) == []
    assert stats.rejected["superseded"] == 1


def test_empty_subject_is_dropped() -> None:
    stats = TradeStatistics()
    rows = [
        _note(
            "",
            "The Central Government revises the tariff value of gold to USD 700 "
            "per ten grams for the current fortnight.",
        )
    ]
    assert list(iter_notification_pairs(rows, TradeFilterConfig(), stats)) == []
    assert stats.rejected["empty_subject"] == 1


def test_annexure_only_body_is_dropped() -> None:
    stats = TradeStatistics()
    rows = [_note("Tariff values notified", "Annexure-I enclosed herewith.")]
    assert list(iter_notification_pairs(rows, TradeFilterConfig(), stats)) == []
    assert stats.rejected["annexure_only"] == 1


def test_short_body_is_dropped() -> None:
    stats = TradeStatistics()
    rows = [_note("Revised customs duty on electronics imports", "Noted.")]
    assert list(iter_notification_pairs(rows, TradeFilterConfig(), stats)) == []
    assert stats.rejected["short_body"] == 1


def test_short_subject_is_dropped() -> None:
    stats = TradeStatistics()
    rows = [
        _note(
            "BCD X",
            "The basic customs duty on the specified goods is revised as set out "
            "in the operative paragraphs of this notification.",
        )
    ]
    assert list(iter_notification_pairs(rows, TradeFilterConfig(), stats)) == []
    assert stats.rejected["short_subject"] == 1


def test_stub_body_restating_subject_is_overlap_capped() -> None:
    # A STUB: the body is essentially just the subject restated, adding no
    # operative content — body->subject overlap is high, so nothing to retrieve.
    # (Measured body->subject on purpose: a genuine terse notice whose long body
    # merely *opens* with the subject is NOT a stub and must survive — see
    # test_headline_subject_over_rich_body_is_kept.)
    stats = TradeStatistics()
    rows = [
        _note(
            "Revised tariff value of gold and silver bars",
            "Revised tariff value of gold and silver bars is notified.",
        )
    ]
    assert list(iter_notification_pairs(rows, TradeFilterConfig(), stats)) == []
    assert stats.rejected["overlap"] == 1


def test_exact_duplicate_notifications_are_deduplicated() -> None:
    stats = TradeStatistics()
    kept = list(iter_notification_pairs([_GENUINE, _GENUINE, _GENUINE], TradeFilterConfig(), stats))
    assert len(kept) == 1
    assert stats.rejected["duplicate"] == 2


def test_oracle_diff_naive_keeps_the_noise_the_filter_removes() -> None:
    """The disciplined filter and the naive reference disagree by exactly the noise."""

    rows = [
        _GENUINE,  # genuine, kept by both
        _note(
            "Amendment to Foreign Trade Policy 2023 export incentives",
            "The Directorate General of Foreign Trade amends the scheme so that "
            "eligible exporters may claim the enhanced benefit under the revised "
            "provisions from the notified date.",
            layer="C",
        ),  # a second genuine pair (FTP, layer C)
        _note(
            "Rescission of notification 5/2023-Customs",
            "Notification 5/2023-Customs is hereby withdrawn from the date of "
            "publication in the Official Gazette.",
        ),  # withdrawn
        _note(
            "Valuation guidance for imported machinery",
            "The earlier circular on the subject stands superseded by the revised "
            "procedure notified separately today.",
        ),  # superseded
        _note("Tariff values notified", "Annexure-I enclosed herewith."),  # annexure-only
        _note(
            "",
            "The Central Government revises the tariff value of gold to USD 700 "
            "per ten grams for the current fortnight.",
        ),  # empty subject
        # short subject
        _note("BCD X", "The basic customs duty on the specified goods is revised as notified."),
        _note("Revised customs duty on electronics imports", "Noted."),  # short body
        _note(
            "Revised tariff value of gold and silver bars",
            "Revised tariff value of gold and silver bars is notified.",
        ),  # stub: body restates subject, nothing added -> overlap
        _GENUINE,  # exact duplicate of the genuine row
    ]

    naive_stats = TradeStatistics()
    filtered_stats = TradeStatistics()

    naive = list(iter_notification_pairs(rows, TradeFilterConfig.naive(), naive_stats))
    filtered = list(iter_notification_pairs(rows, TradeFilterConfig(), filtered_stats))

    # The naive reference keeps every row (nothing here has an empty body, the
    # one structural drop), so it is the full population to diff against.
    assert naive_stats.produced == len(rows)
    assert len(naive) == len(rows)

    # The filter keeps only the two genuine pairs.
    assert filtered_stats.produced == 2
    assert {pair.anchor for pair in filtered} == {
        "Revised tariff value of crude palm oil",
        "Amendment to Foreign Trade Policy 2023 export incentives",
    }

    # Every removal is a named reason, one per row.
    assert dict(filtered_stats.rejected) == {
        "withdrawn": 1,
        "superseded": 1,
        "annexure_only": 1,
        "empty_subject": 1,
        "short_subject": 1,
        "short_body": 1,
        "overlap": 1,
        "duplicate": 1,
    }

    # The counts reconcile exactly: the naive population equals what the filter
    # kept plus every reason it dropped — the oracle-diff invariant.
    assert naive_stats.produced == filtered_stats.produced + sum(filtered_stats.rejected.values())


def test_bilingual_helper_emits_both_directions_with_empty_prefixes() -> None:
    section = BilingualSection(
        identifier="customs-act-1962:s.12",
        english="Dutiable goods imported into India shall be liable to customs duty.",
        hindi="भारत में आयातित शुल्क योग्य माल सीमा शुल्क के लिए उत्तरदायी होगा।",
        heading="Dutiable goods",
    )

    pairs = list(iter_bilingual_section_pairs([section]))

    # Both directions, and no more.
    assert len(pairs) == 2
    directions = {(pair.anchor, pair.positive) for pair in pairs}
    assert (section.english, section.hindi) in directions
    assert (section.hindi, section.english) in directions

    for pair in pairs:
        # Symmetric register -> empty E5 prefixes, the Layer A decision.
        assert prefix_regime(pair.kind) == ("", "")
        assert pair.kind == TRADE_SECTION_XLING_KIND
        # Both halves share one document id, so a sampler keeps them apart.
        assert pair.document == f"{TRADE_SOURCE}:customs-act-1962:s.12"

    # The two directions carry the two languages, not one.
    assert {pair.language for pair in pairs} == {"en", "hi"}


def test_bilingual_helper_skips_unaligned_section() -> None:
    section = BilingualSection(identifier="s.1", english="Short title.", hindi="")
    assert list(iter_bilingual_section_pairs([section])) == []


def _statute_section(number: str, text: str, heading: str = "") -> StatuteSection:
    return StatuteSection(number=number, heading=heading, text=text)


def test_align_bilingual_sections_pairs_by_number_and_feeds_the_helper() -> None:
    # A concrete StatuteSection (the real producer) satisfies the aligner's
    # structural input — so the reader → aligner → pair-helper path is whole.
    english = [
        _statute_section("12", "Dutiable goods shall be liable to customs duty.", "Dutiable goods"),
    ]
    hindi = [
        _statute_section("12", "शुल्क योग्य माल सीमा शुल्क के लिए उत्तरदायी होगा।", "शुल्क योग्य माल"),
    ]

    aligned = list(
        align_bilingual_sections(english, hindi, act_identifier="customs-act-1962")
    )

    assert len(aligned) == 1
    section = aligned[0]
    assert section.identifier == "customs-act-1962:s.12"
    assert section.english == english[0].text
    assert section.hindi == hindi[0].text
    # The aligned section drops straight into the cross-lingual helper: both
    # directions, symmetric empty prefixes, one shared document id.
    pairs = list(iter_bilingual_section_pairs([section]))
    assert len(pairs) == 2
    assert {pair.language for pair in pairs} == {"en", "hi"}
    for pair in pairs:
        assert pair.kind == TRADE_SECTION_XLING_KIND
        assert pair.document == f"{TRADE_SOURCE}:customs-act-1962:s.12"


def test_align_matches_across_case_and_whitespace_but_keeps_english_spelling() -> None:
    # "12A" and "12 a" are the same section — folded, whitespace-free key — but
    # the id carries the English reading's own spelling for a readable trail.
    english = [_statute_section("12A", "Provisional assessment of duty.")]
    hindi = [_statute_section("12 a", "शुल्क का अनंतिम निर्धारण।")]

    aligned = list(
        align_bilingual_sections(english, hindi, act_identifier="customs-act-1962")
    )

    assert len(aligned) == 1
    assert aligned[0].identifier == "customs-act-1962:s.12A"


def test_align_ledger_counts_every_unmatched_reason_and_reconciles() -> None:
    # One section of each fate, oracle-diff style: the distinct section numbers
    # across both readings must reconcile to aligned + the union-level reasons,
    # with duplicate/no_number as parse-health counters outside that sum.
    english = [
        _statute_section("10", "Rate of duty and tariff valuation."),  # aligned
        _statute_section("10", "A re-parse doubled this section."),  # duplicate
        _statute_section("11", "Power to prohibit importation."),  # english_only
        _statute_section("12", "Has an English side."),  # empty_side (hi blank)
        _statute_section("", "No number to key on."),  # no_number
    ]
    hindi = [
        _statute_section("10", "शुल्क की दर और प्रशुल्क मूल्यांकन।"),  # aligned
        _statute_section("12", ""),  # empty_side
        _statute_section("13", "कुछ मालों के आयात पर रोक।"),  # hindi_only
    ]

    stats = BilingualAlignmentStatistics()
    aligned = list(
        align_bilingual_sections(
            english, hindi, act_identifier="customs-act-1962", stats=stats
        )
    )

    assert len(aligned) == 1 and stats.aligned == 1
    assert stats.unmatched["english_only"] == 1
    assert stats.unmatched["hindi_only"] == 1
    assert stats.unmatched["empty_side"] == 1
    assert stats.unmatched["duplicate"] == 1
    assert stats.unmatched["no_number"] == 1

    # Reconciliation over numbered-distinct sections: {10, 11, 12, 13} == 4.
    distinct_numbers = {"10", "11", "12", "13"}
    union_reasons = (
        stats.aligned
        + stats.unmatched["english_only"]
        + stats.unmatched["hindi_only"]
        + stats.unmatched["empty_side"]
    )
    assert union_reasons == len(distinct_numbers)
    # The JSON view a build report carries is faithful.
    assert stats.to_dict()["aligned"] == 1


def test_extract_writes_pair_jsonl_and_carries_provenance(tmp_path: Path) -> None:
    source = tmp_path / "notifications.csv"
    source.write_text(
        "NotificationNumber,Date,Subject,Body,Instrument,Layer,Year\n"
        "34/2024-Customs,2024-06-01,Revised tariff value of crude palm oil,"
        "The Central Government hereby lowers the levy on imported edible commodities"
        " across all ports.,notification,B,2024\n"
        "5/2023-Customs,2023-01-01,Rescission of earlier notification,"
        "This notification is hereby withdrawn with immediate effect from"
        " publication.,notification,B,2023\n",
        encoding="utf-8",
    )
    out = tmp_path / "pairs-trade.jsonl.gz"

    stats = extract_notification_pairs(source, out)

    assert stats.produced == 1
    assert stats.rejected["withdrawn"] == 1
    with gzip.open(out, "rt", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle]
    assert len(records) == 1
    assert records[0]["kind"] == TRADE_NOTIFICATION_KIND
    assert records[0]["anchor"] == "Revised tariff value of crude palm oil"

    # The provenance constants exist for the card/report to carry. Both §52
    # bases are encoded and differ per layer (q(ii) Act vs q(i) Gazette).
    assert TRADE_LICENSE == "GoI-statutory"
    assert "CBIC" in TRADE_ATTRIBUTION
    assert "s.52(1)(q)(ii)" in TRADE_LEGAL_BASIS_ACT
    assert "s.52(1)(q)(i)" in TRADE_LEGAL_BASIS_GAZETTE
    assert TRADE_LEGAL_BASIS_ACT != TRADE_LEGAL_BASIS_GAZETTE


def test_missing_required_column_fails_loudly(tmp_path: Path) -> None:
    source = tmp_path / "bad.csv"
    # no body column
    source.write_text("NotificationNumber,Subject\n1/2024,leaf yellow\n", encoding="utf-8")
    with pytest.raises(TradeReadError):
        list(read_notification_rows(source))
