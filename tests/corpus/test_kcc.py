"""
Reading Kisan Call Centre (KCC) transcripts into the training corpus.

KCC is the *training* front door for the agriculture-advisory domain, and its
rows arrive already pair-shaped (query, answer). So the property that makes the
output trainable is not "find every section" as it is for statutes — it is
"remove the call-centre noise without over-cleaning terse-but-real queries".

That filter is pinned oracle-diff style: the disciplined pass is diffed against
a deliberately naive reference (:meth:`KccFilterConfig.naive`) on a fixture that
carries one row of every noise category, and the two are required to *disagree*
by exactly those rows — which makes the junk removal loud and measured instead
of an opaque "fewer pairs". Each noise category also has its own pinning test so
a regression names the category it broke. The GODL attribution and licence are
asserted present so a mechanical pipeline cannot strip the credit obligation.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from multilingual_embedding.corpus.kcc import (
    KCC_ATTRIBUTION,
    KCC_LICENSE,
    KCC_SOURCE,
    KccFilterConfig,
    KccReadError,
    KccRow,
    KccStatistics,
    extract_kcc_pairs,
    iter_kcc_pairs,
    read_kcc_rows,
)


def _row(query: str, answer: str, **facets: str) -> KccRow:
    return KccRow(query=query, answer=answer, **facets)


# One genuine agri Q→A. Terse and code-mixed on purpose: the query is 11
# characters, below a Wikipedia floor but a real farmer question, so the filter
# must keep it.
_GENUINE = _row("leaf yellow paddy", "Apply nitrogen fertiliser and ensure proper drainage in the field.", crop="Paddy")


def test_genuine_pair_is_kept_and_well_formed():
    stats = KccStatistics()

    pairs = list(iter_kcc_pairs([_GENUINE], KccFilterConfig(), stats))

    assert len(pairs) == 1
    pair = pairs[0]
    assert pair.anchor == "leaf yellow paddy"
    assert pair.positive.startswith("Apply nitrogen")
    assert pair.kind == "kcc_qa"
    # Every pair traces back to the source through its document id, which is the
    # provenance a pair record carries in place of a per-row licence field.
    assert pair.document.startswith(f"{KCC_SOURCE}:")
    assert stats.produced == 1
    assert stats.rejected == {}


def test_terse_real_query_survives_the_length_floor():
    # "leaf yellow" is 11 chars — terse is not junk, the statute lesson.
    stats = KccStatistics()
    kept = list(iter_kcc_pairs([_row("leaf yellow", "Nitrogen deficiency; top-dress with urea.")], KccFilterConfig(), stats))
    assert len(kept) == 1


def test_junk_queries_are_dropped():
    rows = [
        _row("test", "some answer that is long enough to pass the floor"),
        _row("123456", "another sufficiently long answer for the floor here"),
        _row(".....", "yet another sufficiently long answer for the floor"),
    ]
    stats = KccStatistics()
    kept = list(iter_kcc_pairs(rows, KccFilterConfig(), stats))
    assert kept == []
    assert stats.rejected["junk_query"] == 3


def test_short_answer_is_dropped():
    stats = KccStatistics()
    kept = list(iter_kcc_pairs([_row("when to sow wheat", "solved")], KccFilterConfig(), stats))
    assert kept == []
    assert stats.rejected["short_answer"] == 1


def test_boilerplate_answer_is_dropped_across_all_its_rows():
    template = "Contact your nearest Krishi Vigyan Kendra for details."
    rows = [_row(f"crop query number {i}", template) for i in range(6)]
    # A genuine, non-repeated pair to prove the cut is frequency-based, not blanket.
    rows.append(_row("how to control stem borer in rice", "Spray cartap hydrochloride at recommended dose."))
    stats = KccStatistics()
    kept = list(iter_kcc_pairs(rows, KccFilterConfig(boilerplate_answer_count=5), stats))
    assert len(kept) == 1
    assert kept[0].anchor.startswith("how to control")
    assert stats.rejected["boilerplate_answer"] == 6


def test_exact_duplicates_are_deduplicated():
    row = _row("price of tomato today", "Check the nearest APMC mandi rate for tomato.")
    stats = KccStatistics()
    kept = list(iter_kcc_pairs([row, row, row], KccFilterConfig(), stats))
    assert len(kept) == 1
    assert stats.rejected["duplicate"] == 2


def test_query_restated_in_answer_is_overlap_capped():
    # The answer is the query verbatim — a string-solvable pair the model should
    # not be rewarded for.
    stats = KccStatistics()
    kept = list(
        iter_kcc_pairs(
            [_row("how much urea per acre for wheat", "how much urea per acre for wheat")],
            KccFilterConfig(minimum_answer_characters=1),
            stats,
        )
    )
    assert kept == []
    assert stats.rejected["overlap"] == 1


def test_oracle_diff_naive_keeps_the_noise_the_filter_removes():
    """The disciplined filter and the naive reference must disagree by the noise."""

    boiler = "Information provided to the farmer as per query."
    rows = [
        _GENUINE,  # genuine, kept by both
        _row("test", "some sufficiently long answer to clear the floor here"),  # junk query
        _row("okay", "another sufficiently long answer to clear the floor"),  # junk query
        _row("when does the monsoon arrive", "yes"),  # short answer
        *[_row(f"generic advisory query {i}", boiler) for i in range(60)],  # boilerplate
        _GENUINE,  # exact duplicate of the genuine row
    ]

    naive_stats = KccStatistics()
    filtered_stats = KccStatistics()

    naive = list(iter_kcc_pairs(rows, KccFilterConfig.naive(), naive_stats))
    filtered = list(iter_kcc_pairs(rows, KccFilterConfig(), filtered_stats))

    # Naive keeps essentially everything (only structurally-empty rows never
    # reach it, and there are none here).
    assert naive_stats.produced == len(rows)
    # The filter must remove strictly more — the noise — and must keep the one
    # genuine pair exactly once.
    assert filtered_stats.produced == 1
    assert filtered[0].anchor == "leaf yellow paddy"
    # And the removals are a named ledger, not an opaque count.
    assert set(filtered_stats.rejected) >= {"junk_query", "short_answer", "boilerplate_answer", "duplicate"}
    assert naive_stats.produced > filtered_stats.produced


def test_extract_writes_pair_jsonl_and_carries_licence(tmp_path: Path):
    source = tmp_path / "kcc-sample.csv"
    source.write_text(
        "StateName,DistrictName,Crop,QueryType,QueryText,KccAns,CreatedOn\n"
        "Punjab,Ludhiana,Paddy,Nutrient,leaf yellow paddy,Apply nitrogen and drain the field properly.,2024-06-01\n"
        "Punjab,Ludhiana,Paddy,General,test,ignored short-nothing junk answer row here,2024-06-01\n",
        encoding="utf-8",
    )
    out = tmp_path / "pairs.jsonl.gz"

    stats = extract_kcc_pairs(source, out)

    assert stats.produced == 1
    with gzip.open(out, "rt", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle]
    assert len(records) == 1
    assert records[0]["kind"] == "kcc_qa"
    assert records[0]["anchor"] == "leaf yellow paddy"
    # The licence + attribution constants exist for the card/report to carry.
    assert KCC_LICENSE == "GODL-India"
    assert "GODL-India" in KCC_ATTRIBUTION


def test_missing_required_column_fails_loudly(tmp_path: Path):
    source = tmp_path / "bad.csv"
    source.write_text("StateName,QueryText\nPunjab,leaf yellow\n", encoding="utf-8")  # no answer column
    with pytest.raises(KccReadError):
        list(read_kcc_rows(source))
