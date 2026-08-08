"""
Reading public court judgments into the training corpus.

Judgments are the *training* front door for the legal domain, so the
properties pinned here are the ones that make the output trainable: a
document becomes the same corpus record the readers already consume, it
carries the CC BY licence that permits training where MILPaC's does not,
and a failed extraction is dropped rather than emitted as an empty record
that reads downstream as a real judgment.

The one part that depends on a real PDF — pulling text out of it — is a
seam these tests inject a fake for. That is deliberate: PDF fidelity
cannot be verified until a real collection is downloaded and audited, so
everything *around* the extraction is tested exhaustively here, and the
extraction itself is isolated behind a boundary rather than faked into
looking verified.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from multilingual_embedding.corpus.judgments import (
    JUDGMENTS_LICENSE,
    JUDGMENTS_SOURCE,
    JudgmentDocument,
    JudgmentExtractionError,
    extract_judgments,
    iter_judgments,
)

# A body long enough to clear the failed-extraction floor. Real judgments
# run to thousands of characters; this stands in for one.
_BODY = (
    "The appellant challenges the order of the High Court dismissing the "
    "writ petition. Having heard learned counsel for the parties and "
    "perused the record, we are of the considered view that the impugned "
    "order cannot be sustained. The appeal is accordingly allowed and the "
    "matter is remitted for fresh consideration in accordance with law."
) * 3


def touch_pdfs(directory: Path, *names: str) -> Path:
    """Create empty ``.pdf`` files so path resolution has something to find."""

    directory.mkdir(parents=True, exist_ok=True)

    for name in names:
        (directory / name).write_bytes(b"%PDF-1.4\n")

    return directory


def fake_reader(pages_by_stem: dict[str, tuple[str | None, list[str]]]):
    """
    A PDF-reading seam that returns canned text keyed by filename stem.

    Lets a test exercise the whole filter-and-record pipeline without a
    real PDF — the extraction fidelity is the one thing these tests
    deliberately do not assert, because it cannot be known until a real
    collection is read.
    """

    def read(path: Path) -> tuple[str | None, list[str]]:
        return pages_by_stem[path.stem]

    return read


def test_a_pdf_becomes_the_record_the_corpus_reader_accepts(tmp_path: Path) -> None:
    from multilingual_embedding.corpus.reader import JsonlReader

    touch_pdfs(tmp_path, "case-1.pdf")

    reader = fake_reader({"case-1": ("Union of India v. Ors", [_BODY])})

    output = tmp_path / "out.jsonl"

    extract_judgments(tmp_path, output, reader=reader)

    # The decisive assertion: the written record round-trips through the
    # same reader the training corpus is loaded with, without conversion.
    documents = list(JsonlReader(output).iter_documents())

    assert len(documents) == 1

    assert documents[0].identifier == "case-1"

    assert documents[0].metadata.base.language == "en"


def test_every_record_carries_the_permissive_licence(tmp_path: Path) -> None:
    touch_pdfs(tmp_path, "case-1.pdf")

    reader = fake_reader({"case-1": (None, [_BODY])})

    record = next(iter_judgments(tmp_path, reader=reader)).to_record()

    # Statutory public domain (§52(1)(q)), not NC: this is the reason
    # judgments train and MILPaC only scores, so the basis is stamped on the
    # record at the point of use.
    assert record["source"] == JUDGMENTS_SOURCE

    assert (
        record["license"]
        == JUDGMENTS_LICENSE
        == "Public Domain (India, Copyright Act §52(1)(q))"
    )


def test_the_language_defaults_to_english_and_is_overridable(tmp_path: Path) -> None:
    touch_pdfs(tmp_path, "case-1.pdf")

    reader = fake_reader({"case-1": (None, [_BODY])})

    assert next(iter_judgments(tmp_path, reader=reader)).language == "en"

    # A Hindi collection is read by asserting the language, because the
    # PDF does not state one worth trusting.
    assert next(iter_judgments(tmp_path, language="hi", reader=reader)).language == "hi"


def test_the_title_comes_from_metadata_when_present(tmp_path: Path) -> None:
    touch_pdfs(tmp_path, "2019-SC-4461.pdf")

    reader = fake_reader({"2019-SC-4461": ("Kesavananda Bharati v. State of Kerala", [_BODY])})

    document = next(iter_judgments(tmp_path, reader=reader))

    assert document.title == "Kesavananda Bharati v. State of Kerala"

    # The identifier is the file stem regardless, so a record traces back
    # to the exact PDF.
    assert document.identifier == "2019-SC-4461"


def test_the_title_falls_back_to_the_file_stem(tmp_path: Path) -> None:
    touch_pdfs(tmp_path, "2019-SC-4461.pdf")

    reader = fake_reader({"2019-SC-4461": (None, [_BODY])})

    document = next(iter_judgments(tmp_path, reader=reader))

    # No metadata title, so the stem stands in — traceable, not invented
    # off the first line of prose whose layout no parser can trust.
    assert document.title == "2019-SC-4461"


def test_pages_are_joined_into_one_document(tmp_path: Path) -> None:
    touch_pdfs(tmp_path, "case-1.pdf")

    reader = fake_reader(
        {"case-1": (None, ["First page prose. " * 20, "Second page prose. " * 20])}
    )

    document = next(iter_judgments(tmp_path, reader=reader))

    assert "First page prose." in document.text

    assert "Second page prose." in document.text


def test_a_failed_extraction_is_dropped_not_emitted(tmp_path: Path) -> None:
    touch_pdfs(tmp_path, "scanned.pdf", "real.pdf")

    # A scanned judgment's text layer is an image: extraction yields almost
    # nothing. That is a failure, not a short document, and it must not
    # become an empty record that reads downstream as a real judgment.
    reader = fake_reader(
        {
            "scanned": (None, ["", "  \n  "]),
            "real": ("A Real Judgment", [_BODY]),
        }
    )

    kept = [document.identifier for document in iter_judgments(tmp_path, reader=reader)]

    assert kept == ["real"]


def test_the_minimum_characters_floor_is_adjustable(tmp_path: Path) -> None:
    touch_pdfs(tmp_path, "short.pdf")

    reader = fake_reader({"short": (None, ["A short order of the Court."])})

    # Above the default floor it is a failed extraction; lower the floor
    # and the same document is kept.
    assert list(iter_judgments(tmp_path, reader=reader)) == []

    assert len(list(iter_judgments(tmp_path, minimum_characters=10, reader=reader))) == 1


def test_a_directory_reads_every_pdf_sorted(tmp_path: Path) -> None:
    touch_pdfs(tmp_path, "b.pdf", "a.pdf")

    reader = fake_reader({"a": (None, [_BODY]), "b": (None, [_BODY])})

    identifiers = [document.identifier for document in iter_judgments(tmp_path, reader=reader)]

    # Sorted, so a regenerated corpus lines up with itself.
    assert identifiers == ["a", "b"]


def test_the_limit_stops_early(tmp_path: Path) -> None:
    touch_pdfs(tmp_path, "a.pdf", "b.pdf", "c.pdf")

    reader = fake_reader({stem: (None, [_BODY]) for stem in ("a", "b", "c")})

    assert len(list(iter_judgments(tmp_path, limit=2, reader=reader))) == 2


def test_a_directory_with_no_pdfs_is_an_error(tmp_path: Path) -> None:
    empty = tmp_path / "empty"

    empty.mkdir()

    with pytest.raises(JudgmentExtractionError):
        list(iter_judgments(empty, reader=fake_reader({})))


def test_a_missing_path_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(JudgmentExtractionError):
        list(iter_judgments(tmp_path / "nope.pdf", reader=fake_reader({})))


def test_extract_writes_one_line_per_judgment(tmp_path: Path) -> None:
    touch_pdfs(tmp_path, "a.pdf", "b.pdf")

    reader = fake_reader({"a": (None, [_BODY]), "b": (None, [_BODY])})

    output = tmp_path / "corpus.jsonl"

    count = extract_judgments(tmp_path, output, reader=reader)

    assert count == 2

    lines = [line for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert len(lines) == 2

    for line in lines:
        record = json.loads(line)

        assert record["source"] == JUDGMENTS_SOURCE


def test_extract_gzips_by_extension(tmp_path: Path) -> None:
    touch_pdfs(tmp_path, "a.pdf")

    reader = fake_reader({"a": (None, [_BODY])})

    output = tmp_path / "corpus.jsonl.gz"

    extract_judgments(tmp_path, output, reader=reader)

    with gzip.open(output, "rt", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]

    assert len(records) == 1

    assert records[0]["license"] == JUDGMENTS_LICENSE


def test_the_document_is_a_dataclass_round_trip() -> None:
    document = JudgmentDocument(
        identifier="case-1",
        title="A v. B",
        text=_BODY,
        language="en",
    )

    record = document.to_record()

    assert record["id"] == "case-1"

    assert record["title"] == "A v. B"

    assert record["text"] == _BODY
