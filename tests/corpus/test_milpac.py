"""
MILPaC evaluation-set reading.

MILPaC is the held-out evaluation front door, and the properties these
tests pin are the ones that make it a *held-out* set rather than just
another corpus: it produces pair records the adapter's evaluation loader
accepts, every record carries the non-commercial licence at the point it
could be misused, and a row in a language or dataset the run did not ask
for is dropped rather than quietly scored.

The workbook fixtures are built in ``tmp_path`` with openpyxl, so nothing
here reaches the network or depends on a downloaded corpus — the same
discipline as the Wikipedia tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("openpyxl", reason="requires the milpac extra")

from openpyxl import Workbook

from multilingual_embedding.corpus.milpac import (
    MILPAC_LICENSE,
    MILPAC_SOURCE,
    MilpacExtractionError,
    MilpacUnit,
    extract_milpac,
    iter_units,
)
from multilingual_embedding.corpus.pairs import MinedPair

_HEADER = ("dataset", "id", "src_lang", "src", "tgt_lang", "tgt")

# Long enough that the pair is a real passage rather than a fragment; the
# adapter's evaluation does not care, but a realistic fixture keeps the
# tests honest about what a MILPaC row looks like.
_EN = "The appropriate Government may, by notification, exempt any establishment."

_HI = "उपयुक्त सरकार, अधिसूचना द्वारा, किसी भी स्थापना को छूट दे सकती है।"

_TA = "தகுந்த அரசாங்கம், அறிவிப்பின் மூலம், எந்தவொரு நிறுவனத்தையும் விலக்களிக்கலாம்."


def workbook(
    path: Path,
    rows: list[tuple[object, ...]],
    *,
    header: tuple[str, ...] = _HEADER,
) -> Path:
    """Write one MILPaC-shaped workbook and return its path."""

    book = Workbook()

    sheet = book.active

    sheet.append(header)

    for row in rows:
        sheet.append(row)

    book.save(path)

    return path


def test_a_row_becomes_a_pair_record_the_eval_loader_accepts(tmp_path: Path) -> None:
    path = workbook(
        tmp_path / "acts.xlsx",
        [("Acts", "1", "EN", _EN, "HI", _HI)],
    )

    units = list(iter_units(path))

    assert len(units) == 1

    record = units[0].to_pair_record()

    # The decisive assertion: the record round-trips through the same
    # loader the adapter uses for --eval-pairs-file, without conversion.
    pair = MinedPair.from_record(record)

    assert pair.anchor == _EN

    assert pair.positive == _HI

    assert pair.language == "hi"


def test_every_record_carries_the_non_commercial_licence(tmp_path: Path) -> None:
    path = workbook(tmp_path / "acts.xlsx", [("Acts", "1", "EN", _EN, "TA", _TA)])

    record = next(iter_units(path)).to_pair_record()

    # Stamped on the record itself, so the restriction is legible in the
    # raw file at the point someone might feed it to --pairs.
    assert record["source"] == MILPAC_SOURCE

    assert record["license"] == MILPAC_LICENSE == "CC BY-NC-SA 4.0"


def test_the_target_language_is_the_facet(tmp_path: Path) -> None:
    path = workbook(
        tmp_path / "acts.xlsx",
        [
            ("Acts", "1", "EN", _EN, "HI", _HI),
            ("Acts", "2", "EN", _EN, "TA", _TA),
        ],
    )

    by_language = {unit.identifier: unit.target_language for unit in iter_units(path)}

    # The Indic target, not the English source, is what a per-language
    # recall table is broken down by.
    assert by_language == {"1": "hi", "2": "ta"}


def test_a_language_not_asked_for_is_dropped(tmp_path: Path) -> None:
    path = workbook(
        tmp_path / "acts.xlsx",
        [
            ("Acts", "1", "EN", _EN, "HI", _HI),
            ("Acts", "2", "EN", _EN, "BN", "কোনো প্রতিষ্ঠানকে ছাড় দিতে পারে।"),
        ],
    )

    languages = {unit.target_language for unit in iter_units(path)}

    # Bengali is in MILPaC but not in this project's locked pair, so it is
    # skipped rather than scored — a recall over it would measure the base
    # checkpoint, not the adaptation.
    assert languages == {"hi"}


def test_widening_the_languages_keeps_more(tmp_path: Path) -> None:
    path = workbook(
        tmp_path / "acts.xlsx",
        [("Acts", "2", "EN", _EN, "BN", "কোনো প্রতিষ্ঠানকে ছাড় দিতে পারে।")],
    )

    assert list(iter_units(path, languages=("hi", "ta"))) == []

    assert len(list(iter_units(path, languages=("bn",)))) == 1


def test_a_dataset_filter_selects_by_register(tmp_path: Path) -> None:
    path = workbook(
        tmp_path / "mixed.xlsx",
        [
            ("Acts", "1", "EN", _EN, "HI", _HI),
            ("IP", "2", "EN", _EN, "HI", _HI),
        ],
    )

    datasets = {unit.dataset for unit in iter_units(path, datasets=("acts",))}

    assert datasets == {"Acts"}


def test_a_blank_side_is_skipped(tmp_path: Path) -> None:
    path = workbook(
        tmp_path / "acts.xlsx",
        [
            ("Acts", "1", "EN", _EN, "HI", ""),
            ("Acts", "2", "EN", "", "HI", _HI),
            ("Acts", "3", "EN", _EN, "HI", _HI),
        ],
    )

    kept = list(iter_units(path))

    # A half-empty aligned unit is not a pair; the one complete row
    # survives.
    assert [unit.identifier for unit in kept] == ["3"]


def test_a_non_english_source_is_skipped(tmp_path: Path) -> None:
    path = workbook(
        tmp_path / "acts.xlsx",
        [
            ("Acts", "1", "HI", _HI, "TA", _TA),
            ("Acts", "2", "EN", _EN, "TA", _TA),
        ],
    )

    kept = [unit.identifier for unit in iter_units(path)]

    # MILPaC is English-centric; a non-English source row is a schema
    # surprise, not a usable cross-lingual anchor.
    assert kept == ["2"]


def test_a_missing_column_is_refused_loudly(tmp_path: Path) -> None:
    path = workbook(
        tmp_path / "broken.xlsx",
        [("Acts", "1", "EN", _EN, _HI)],
        header=("dataset", "id", "src_lang", "src", "tgt"),  # no tgt_lang
    )

    # Reading the wrong column would produce a plausible evaluation file
    # rather than an error, so a missing documented column must raise.
    with pytest.raises(MilpacExtractionError):
        list(iter_units(path))


def test_columns_are_matched_by_name_not_position(tmp_path: Path) -> None:
    path = workbook(
        tmp_path / "reordered.xlsx",
        [("HI", _HI, "1", "EN", _EN, "Acts")],
        header=("tgt_lang", "tgt", "id", "src_lang", "src", "dataset"),
    )

    unit = next(iter_units(path))

    # A reordered export still reads correctly because columns are found
    # by header name.
    assert unit.source_text == _EN

    assert unit.target_text == _HI

    assert unit.dataset == "Acts"


def test_a_directory_reads_every_workbook(tmp_path: Path) -> None:
    corpus = tmp_path / "milpac"

    corpus.mkdir()

    workbook(corpus / "acts.xlsx", [("Acts", "1", "EN", _EN, "HI", _HI)])

    workbook(corpus / "ip.xlsx", [("IP", "2", "EN", _EN, "TA", _TA)])

    datasets = {unit.dataset for unit in iter_units(corpus)}

    assert datasets == {"Acts", "IP"}


def test_a_directory_with_no_workbooks_is_an_error(tmp_path: Path) -> None:
    empty = tmp_path / "empty"

    empty.mkdir()

    with pytest.raises(MilpacExtractionError):
        list(iter_units(empty))


def test_a_missing_path_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(MilpacExtractionError):
        list(iter_units(tmp_path / "nope.xlsx"))


def test_extract_writes_one_line_per_unit(tmp_path: Path) -> None:
    source = workbook(
        tmp_path / "acts.xlsx",
        [
            ("Acts", "1", "EN", _EN, "HI", _HI),
            ("Acts", "2", "EN", _EN, "TA", _TA),
        ],
    )

    output = tmp_path / "eval.jsonl"

    count = extract_milpac(source, output)

    assert count == 2

    lines = [line for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert len(lines) == 2

    # Every written line is loadable as an evaluation pair.
    for line in lines:
        import json

        MinedPair.from_record(json.loads(line))


def test_extract_gzips_by_extension(tmp_path: Path) -> None:
    import gzip
    import json

    source = workbook(tmp_path / "acts.xlsx", [("Acts", "1", "EN", _EN, "HI", _HI)])

    output = tmp_path / "eval.jsonl.gz"

    extract_milpac(source, output)

    with gzip.open(output, "rt", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]

    assert len(records) == 1

    assert records[0]["source"] == MILPAC_SOURCE


def test_document_id_encodes_provenance(tmp_path: Path) -> None:
    path = workbook(tmp_path / "acts.xlsx", [("Acts", "42", "EN", _EN, "HI", _HI)])

    unit = next(iter_units(path))

    # dataset, language pair and row id, so a suspicious eval hit can be
    # traced back to the exact source row.
    assert unit.to_pair_record()["document"] == "milpac-Acts-en-hi-42"


def test_the_unit_is_a_dataclass_round_trip() -> None:
    unit = MilpacUnit(
        dataset="Acts",
        identifier="1",
        source_language="en",
        target_language="hi",
        source_text=_EN,
        target_text=_HI,
    )

    record = unit.to_pair_record()

    assert record["anchor"] == _EN

    assert record["kind"] == "parallel"
