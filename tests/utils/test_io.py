from __future__ import annotations

from pathlib import Path

import pytest

from multilingual_embedding.core.exceptions import SerializationError
from multilingual_embedding.utils.io import (
    count_lines,
    read_json,
    read_jsonl,
    read_text,
    read_yaml,
    write_json,
    write_jsonl,
    write_text,
    write_yaml,
)

HINDI = "नमस्ते दुनिया"


def test_text_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "file.txt"

    write_text(path, HINDI)

    assert read_text(path) == HINDI


def test_text_round_trip_gzip(tmp_path: Path) -> None:
    """A .gz suffix must be handled transparently by both directions."""

    path = tmp_path / "file.txt.gz"

    write_text(path, HINDI)

    assert path.read_bytes()[:2] == b"\x1f\x8b"

    assert read_text(path) == HINDI


def test_json_round_trip_preserves_unicode(tmp_path: Path) -> None:
    path = tmp_path / "file.json"

    write_json(path, {"text": HINDI})

    assert read_json(path)["text"] == HINDI

    assert HINDI in path.read_text(encoding="utf-8")


def test_invalid_json_raises(tmp_path: Path) -> None:
    path = tmp_path / "file.json"

    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(SerializationError):
        read_json(path)


def test_jsonl_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "file.jsonl"

    records = [{"id": 1, "text": HINDI}, {"id": 2, "text": "hello"}]

    assert write_jsonl(path, records) == 2

    assert list(read_jsonl(path)) == records


def test_jsonl_round_trip_gzip(tmp_path: Path) -> None:
    path = tmp_path / "file.jsonl.gz"

    write_jsonl(path, [{"a": 1}])

    assert list(read_jsonl(path)) == [{"a": 1}]


def test_jsonl_reports_bad_line_number(tmp_path: Path) -> None:
    path = tmp_path / "file.jsonl"

    path.write_text('{"a": 1}\n{bad}\n', encoding="utf-8")

    with pytest.raises(SerializationError) as error:
        list(read_jsonl(path))

    assert error.value.context["line"] == 2


def test_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "file.jsonl"

    path.write_text('{"a": 1}\n\n{"a": 2}\n', encoding="utf-8")

    assert len(list(read_jsonl(path))) == 2


def test_yaml_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "file.yaml"

    write_yaml(path, {"name": "trial", "text": HINDI})

    assert read_yaml(path) == {"name": "trial", "text": HINDI}


def test_yaml_rejects_arbitrary_objects(tmp_path: Path) -> None:
    """safe_load must refuse Python object tags."""

    path = tmp_path / "file.yaml"

    path.write_text("!!python/object/apply:os.system ['echo']\n", encoding="utf-8")

    with pytest.raises(SerializationError):
        read_yaml(path)


def test_count_lines(tmp_path: Path) -> None:
    path = tmp_path / "file.txt"

    path.write_text("a\nb\nc\n", encoding="utf-8")

    assert count_lines(path) == 3


def test_write_creates_parent_directories(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "deep" / "file.txt"

    write_text(path, "x")

    assert path.exists()
