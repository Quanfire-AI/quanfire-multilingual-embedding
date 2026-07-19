from __future__ import annotations

from pathlib import Path

import pytest

from multilingual_embedding.core.exceptions import ResourceNotFoundError
from multilingual_embedding.utils.filesystem import (
    atomic_write_path,
    ensure_directory,
    human_readable_size,
    iter_files,
    require_directory,
    require_file,
)


def test_ensure_directory_creates_parents(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "c"

    assert ensure_directory(target).is_dir()


def test_ensure_directory_is_idempotent(tmp_path: Path) -> None:
    ensure_directory(tmp_path / "a")

    assert ensure_directory(tmp_path / "a").is_dir()


def test_require_file_rejects_missing(tmp_path: Path) -> None:
    with pytest.raises(ResourceNotFoundError):
        require_file(tmp_path / "missing.txt")


def test_require_file_rejects_directory(tmp_path: Path) -> None:
    with pytest.raises(ResourceNotFoundError):
        require_file(tmp_path)


def test_require_directory_rejects_file(tmp_path: Path) -> None:
    path = tmp_path / "file.txt"

    path.write_text("x", encoding="utf-8")

    with pytest.raises(ResourceNotFoundError):
        require_directory(path)


def test_iter_files_is_sorted_and_recursive(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()

    for name in ["c.txt", "a.txt", "b.txt"]:
        (tmp_path / name).write_text("x", encoding="utf-8")

    (tmp_path / "sub" / "d.txt").write_text("x", encoding="utf-8")

    found = [path.name for path in iter_files(tmp_path, patterns=["*.txt"])]

    assert found == ["a.txt", "b.txt", "c.txt", "d.txt"]


def test_iter_files_non_recursive(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()

    (tmp_path / "a.txt").write_text("x", encoding="utf-8")

    (tmp_path / "sub" / "b.txt").write_text("x", encoding="utf-8")

    found = [path.name for path in iter_files(tmp_path, patterns=["*.txt"], recursive=False)]

    assert found == ["a.txt"]


def test_iter_files_deduplicates_overlapping_patterns(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")

    found = list(iter_files(tmp_path, patterns=["*.txt", "*"]))

    assert len(found) == 1


def test_atomic_write_moves_into_place(tmp_path: Path) -> None:
    target = tmp_path / "out" / "file.txt"

    with atomic_write_path(target) as temporary:
        temporary.write_text("done", encoding="utf-8")

    assert target.read_text(encoding="utf-8") == "done"


def test_atomic_write_leaves_original_on_failure(tmp_path: Path) -> None:
    """A failed write must not destroy the previous contents."""

    target = tmp_path / "file.txt"

    target.write_text("original", encoding="utf-8")

    with pytest.raises(RuntimeError), atomic_write_path(target) as temporary:
        temporary.write_text("partial", encoding="utf-8")

        raise RuntimeError("interrupted")

    assert target.read_text(encoding="utf-8") == "original"


def test_atomic_write_cleans_up_temporary(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError), atomic_write_path(tmp_path / "file.txt"):
        raise RuntimeError("interrupted")

    assert list(tmp_path.iterdir()) == []


def test_human_readable_size() -> None:
    assert human_readable_size(512) == "512 B"

    assert human_readable_size(2048) == "2.0 KB"

    assert human_readable_size(5 * 1024 * 1024) == "5.0 MB"
