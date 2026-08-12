"""
Reading Indian Central Acts into the training corpus.

Central Acts are the *training* front door for the statutory-law domain, so
the properties pinned here are the ones that make the output trainable and
honest: every section is found no matter which container an Act uses,
sections become the heading/section pairs a retriever learns from, and the
CC-BY attribution and the caveated §52 legal basis survive onto every record
so a mechanical pipeline can never strip the licence off the text.

The one structural risk — that a reader keyed to a single container silently
misses the sections of Acts that use another — is pinned by an oracle-diff
test: the recursive reader is diffed against a deliberately naive
Parts-only reference on a mixed Act, and the two are required to *disagree*,
which is what makes the undercount loud instead of silent.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from multilingual_embedding.corpus.annotated_acts import (
    ANNOTATED_ACTS_ATTRIBUTION,
    ANNOTATED_ACTS_LEGAL_BASIS,
    ANNOTATED_ACTS_LICENSE,
    ANNOTATED_ACTS_SOURCE,
    StatuteReadError,
    extract_annotated_acts,
    iter_sections,
    read_annotated_acts,
)
from multilingual_embedding.corpus.pairs import PairKind, mine_pairs
from multilingual_embedding.corpus.reader import reader_for


def _section(heading: str, *paragraphs: str) -> dict:
    """One section in the dataset's shape: a heading and numbered paragraphs."""

    return {"heading": heading, "paragraphs": {str(i): p for i, p in enumerate(paragraphs)}}


def _sections_map(*headings_and_bodies: tuple[str, str]) -> dict:
    """A ``Sections`` map keyed by section label."""

    return {
        f"Section {i + 1}.": _section(heading, body)
        for i, (heading, body) in enumerate(headings_and_bodies)
    }


# An Act that puts its sections under each of the three containers at once —
# top-level ``Sections``, a ``Chapters`` member, and a ``Parts`` member — so
# a reader that only understands one of them is provably incomplete on it.
_MIXED_ACT = {
    "Act Title": "THE MIXED CONTAINER ACT, 1999",
    "Act ID": "ACT NO. 1 OF 1999",
    "Enactment Date": "[1st January, 1999.]",
    "Sections": _sections_map(("Short title", "This Act may be called the Mixed Container Act.")),
    "Chapters": {
        "0": {
            "Name": "PRELIMINARY",
            "Sections": _sections_map(("Definitions", "In this Act, unless the context requires.")),
        }
    },
    "Parts": {
        "0": {
            "Name": "ENFORCEMENT",
            "Sections": _sections_map(("Penalty", "Whoever contravenes shall be punished.")),
        }
    },
}


def _naive_parts_only_sections(act: dict) -> list[str]:
    """
    A deliberately naive reference: sections found by looking under
    ``Parts`` only. This is the reader that would have shipped, and the
    oracle-diff exists so the gap between it and reality is a failing test,
    not a silently smaller corpus.
    """

    headings: list[str] = []

    for part in act.get("Parts", {}).values():
        for section in part.get("Sections", {}).values():
            headings.append(section.get("heading", ""))

    return headings


def test_recursive_discovery_finds_every_container() -> None:
    found = {section.heading for section in iter_sections(_MIXED_ACT)}

    assert found == {"Short title", "Definitions", "Penalty"}


def test_oracle_diff_recursive_beats_naive_parts_only() -> None:
    # The naive reference sees only the Parts section; the recursive reader
    # sees all three. The point of the test is that they DISAGREE — a
    # single-container reader is provably missing sections that exist.
    naive = set(_naive_parts_only_sections(_MIXED_ACT))

    recursive = {section.heading for section in iter_sections(_MIXED_ACT)}

    assert naive == {"Penalty"}

    assert recursive - naive == {"Short title", "Definitions"}

    assert len(recursive) > len(naive)


def test_paragraphs_join_in_numeric_not_lexical_order() -> None:
    section = {
        "heading": "Long section",
        "paragraphs": {str(i): f"para-{i}" for i in range(12)},
    }

    (found,) = list(iter_sections({"Sections": {"Section 1.": section}}))

    # "10" and "11" must follow "9", not sort between "1" and "2".
    assert found.text.splitlines() == [f"para-{i}" for i in range(12)]


def test_nested_subclauses_are_not_dropped() -> None:
    section = {
        "heading": "Nested",
        "paragraphs": {"0": "opening", "1": {"a": "clause a", "b": "clause b"}},
    }

    (found,) = list(iter_sections({"Sections": {"Section 1.": section}}))

    assert found.text == "opening\nclause a\nclause b"


def _write_act(directory: Path, name: str, act: dict) -> None:
    (directory / name).write_text(json.dumps(act), encoding="utf-8")


def test_read_yields_document_with_sections_and_provenance(tmp_path: Path) -> None:
    _write_act(tmp_path, "1999.json", _MIXED_ACT)

    (document,) = list(read_annotated_acts(tmp_path))

    assert document.identifier == "ACT NO. 1 OF 1999"

    assert document.title == "THE MIXED CONTAINER ACT, 1999"

    assert len(document.sections) == 3

    record = document.to_record()

    assert record["source"] == ANNOTATED_ACTS_SOURCE

    assert record["license"] == ANNOTATED_ACTS_LICENSE

    # The CC-BY grant obliges attribution; it must ride on the record.
    assert record["attribution"] == ANNOTATED_ACTS_ATTRIBUTION

    # The caveated §52 footing is flagged, not waved away.
    assert record["legal_basis"] == ANNOTATED_ACTS_LEGAL_BASIS

    assert {s["heading"] for s in record["sections"]} == {"Short title", "Definitions", "Penalty"}


def test_act_with_no_findable_sections_is_dropped(tmp_path: Path) -> None:
    _write_act(tmp_path, "empty.json", {"Act Title": "THE EMPTY ACT", "Act ID": "X"})

    assert list(read_annotated_acts(tmp_path)) == []


def test_empty_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(StatuteReadError):
        list(read_annotated_acts(tmp_path))


def test_invalid_json_raises(tmp_path: Path) -> None:
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(StatuteReadError):
        list(read_annotated_acts(tmp_path))


# A section body long enough to clear the default positive-length floor
# (minimum_positive_characters = 100). Real statute provisions that are this
# long or longer are the ones a retriever should mine; the short ones (extent,
# commencement) are a separate config decision, exercised below.
_LONG_BODY = (
    "In this Act, unless the context otherwise requires, the expressions used "
    "herein shall have the meanings respectively assigned to them, and every "
    "reference to a section shall be construed as a reference to a section of "
    "this Act."
)

# A realistic mixed-container Act whose section bodies clear the default
# positive-length floor, so the round-trip exercises real mining, not a
# fixture the filter happens to drop.
_REALISTIC_ACT = {
    "Act Title": "THE REALISTIC ACT, 2001",
    "Act ID": "ACT NO. 2 OF 2001",
    "Sections": {"Section 1.": _section("Definitions clause", _LONG_BODY)},
    "Chapters": {
        "0": {"Name": "GENERAL", "Sections": {"Section 2.": _section("Application", _LONG_BODY)}}
    },
    "Parts": {
        "0": {"Name": "OFFENCES", "Sections": {"Section 3.": _section("Punishment", _LONG_BODY)}}
    },
}


def test_roundtrip_extract_then_mine_yields_heading_section_pairs(tmp_path: Path) -> None:
    # The integration proof: sections written to the corpus file survive the
    # reader→attributes→mine_pairs path and produce heading/section pairs.
    _write_act(tmp_path, "2001.json", _REALISTIC_ACT)

    corpus = tmp_path / "statutes.jsonl.gz"

    written = extract_annotated_acts(tmp_path, corpus)

    assert written == 1

    # The record round-trips and carries its sections.
    with gzip.open(corpus, "rt", encoding="utf-8") as handle:
        record = json.loads(handle.readline())

    assert len(record["sections"]) == 3

    documents = list(reader_for(corpus).iter_documents())

    pairs, _ = mine_pairs(documents)

    heading_pairs = {pair.anchor for pair in pairs if pair.kind == PairKind.HEADING_SECTION}

    assert heading_pairs == {"Definitions clause", "Application", "Punishment"}
