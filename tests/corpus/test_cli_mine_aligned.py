"""
The ``qfme mine-aligned`` wiring.

A parser check for the required flags, and one end-to-end run over three
tiny files — a source corpus, a target corpus and a langlinks dump — to
prove the join actually happens and writes cross-lingual pairs. The
end-to-end test is worth its cost here because the handler's whole job is
the join, and a join that silently produced nothing would pass every
unit test of its parts.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from multilingual_embedding.cli import EXIT_SUCCESS, build_parser, main

HI_LEAD = (
    "भारत एक विशाल देश है जिसकी संस्कृति और इतिहास बहुत पुराना और समृद्ध है। "
    "यह पैराग्राफ इतना लंबा है कि यह न्यूनतम लंबाई की शर्त को आसानी से पूरा करता है।"
)

TA_LEAD = (
    "இந்தியா ஒரு பரந்த நாடு, அதன் கலாச்சாரமும் வரலாறும் மிகவும் பழமையானது. "
    "இந்த பத்தி குறைந்தபட்ச நீளத் தேவையை எளிதாக பூர்த்தி செய்யும் அளவுக்கு நீளமானது."
)


def _jsonl(path: Path, records: list[dict[str, str]]) -> Path:
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    return path


class TestRequiredFlags:
    REQUIRED = (
        "--source",
        "hi.jsonl",
        "--target",
        "ta.jsonl",
        "--langlinks",
        "ll.sql",
        "--source-language",
        "hi",
        "--target-language",
        "ta",
        "--output",
        "out.jsonl",
    )

    @pytest.mark.parametrize(
        "missing",
        [
            "--source",
            "--target",
            "--langlinks",
            "--source-language",
            "--target-language",
            "--output",
        ],
    )
    def test_each_required_flag_is_required(self, missing: str) -> None:
        argv = list(self.REQUIRED)

        index = argv.index(missing)

        del argv[index : index + 2]

        with pytest.raises(SystemExit):
            build_parser().parse_args(["mine-aligned", *argv])


class TestEndToEnd:
    def test_it_joins_and_writes_cross_lingual_pairs(self, tmp_path: Path) -> None:
        source = _jsonl(
            tmp_path / "hi.jsonl",
            [{"id": "12", "language": "hi", "title": "भारत", "text": HI_LEAD}],
        )

        target = _jsonl(
            tmp_path / "ta.jsonl",
            [{"id": "99", "language": "ta", "title": "இந்தியா", "text": TA_LEAD}],
        )

        langlinks = tmp_path / "ll.sql"

        langlinks.write_text(
            "INSERT INTO `langlinks` VALUES (12,'ta','இந்தியா');\n",
            encoding="utf-8",
        )

        output = tmp_path / "aligned.jsonl.gz"

        code = main(
            [
                "mine-aligned",
                "--source",
                str(source),
                "--target",
                str(target),
                "--langlinks",
                str(langlinks),
                "--source-language",
                "hi",
                "--target-language",
                "ta",
                "--output",
                str(output),
            ]
        )

        assert code == EXIT_SUCCESS

        with gzip.open(output, "rt", encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]

        # title/lead and lead/lead, in both directions.
        assert len(records) == 4

        languages = {record["language"] for record in records}

        assert languages == {"hi", "ta"}

    def test_a_missing_link_writes_nothing_but_still_succeeds(self, tmp_path: Path) -> None:
        """No alignment is a normal, empty outcome — not a failure."""

        source = _jsonl(
            tmp_path / "hi.jsonl",
            [{"id": "12", "language": "hi", "title": "भारत", "text": HI_LEAD}],
        )

        target = _jsonl(
            tmp_path / "ta.jsonl",
            [{"id": "99", "language": "ta", "title": "இந்தியா", "text": TA_LEAD}],
        )

        # A langlinks dump that points 12 at a Malayalam page, not Tamil.
        langlinks = tmp_path / "ll.sql"

        langlinks.write_text(
            "INSERT INTO `langlinks` VALUES (12,'ml','something');\n",
            encoding="utf-8",
        )

        output = tmp_path / "aligned.jsonl"

        code = main(
            [
                "mine-aligned",
                "--source",
                str(source),
                "--target",
                str(target),
                "--langlinks",
                str(langlinks),
                "--source-language",
                "hi",
                "--target-language",
                "ta",
                "--output",
                str(output),
            ]
        )

        assert code == EXIT_SUCCESS

        assert output.read_text(encoding="utf-8") == ""
