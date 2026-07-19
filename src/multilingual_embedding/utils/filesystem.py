"""
Filesystem helpers.

Centralises the path handling that would otherwise be repeated across
readers, writers and model persistence: directory creation, existence
checks that raise framework errors, atomic replacement and recursive
file discovery.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

from multilingual_embedding.core.exceptions import ResourceNotFoundError

__all__ = [
    "atomic_write_path",
    "ensure_directory",
    "human_readable_size",
    "iter_files",
    "require_directory",
    "require_file",
]


def ensure_directory(path: str | Path) -> Path:
    """
    Create a directory and any missing parents.

    Returns the resolved path. Existing directories are left untouched.
    """

    resolved = Path(path).expanduser()

    resolved.mkdir(parents=True, exist_ok=True)

    return resolved


def require_file(path: str | Path) -> Path:
    """
    Return the path if it is an existing file, otherwise raise.

    Raises
    ------
    ResourceNotFoundError
        If the path is missing or is a directory.
    """

    resolved = Path(path).expanduser()

    if not resolved.exists():
        raise ResourceNotFoundError("File does not exist", path=str(resolved))

    if not resolved.is_file():
        raise ResourceNotFoundError("Path is not a file", path=str(resolved))

    return resolved


def require_directory(path: str | Path) -> Path:
    """
    Return the path if it is an existing directory, otherwise raise.

    Raises
    ------
    ResourceNotFoundError
        If the path is missing or is not a directory.
    """

    resolved = Path(path).expanduser()

    if not resolved.exists():
        raise ResourceNotFoundError("Directory does not exist", path=str(resolved))

    if not resolved.is_dir():
        raise ResourceNotFoundError("Path is not a directory", path=str(resolved))

    return resolved


def iter_files(
    directory: str | Path,
    *,
    patterns: Sequence[str] = ("*",),
    recursive: bool = True,
) -> Iterator[Path]:
    """
    Yield files under a directory in deterministic order.

    Ordering matters for reproducibility: a corpus built from a
    directory must produce the same document sequence on every machine,
    and filesystem iteration order is not guaranteed.

    Parameters
    ----------
    directory:
        Root to search.

    patterns:
        Glob patterns, e.g. ``("*.txt", "*.jsonl")``.

    recursive:
        Whether to descend into subdirectories.
    """

    root = require_directory(directory)

    seen: set[Path] = set()

    matches: list[Path] = []

    for pattern in patterns:
        glob_pattern = f"**/{pattern}" if recursive else pattern

        for candidate in root.glob(glob_pattern):
            if candidate.is_file() and candidate not in seen:
                seen.add(candidate)
                matches.append(candidate)

    yield from sorted(matches)


@contextmanager
def atomic_write_path(path: str | Path, *, suffix: str = ".tmp") -> Iterator[Path]:
    """
    Yield a temporary path that is moved into place on success.

    A partially written model or corpus file is worse than no file at
    all, because the next run will happily load the truncated version.
    Writing to a sibling temporary file and renaming makes the
    replacement atomic on POSIX filesystems, so readers observe either
    the old contents or the new ones.

    On failure the temporary file is removed and the original is left
    untouched.

    Example
    -------
    ::

        with atomic_write_path("model.npz") as tmp:
            numpy.savez(tmp, **arrays)
    """

    target = Path(path).expanduser()

    ensure_directory(target.parent)

    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(target.parent),
        prefix=f".{target.name}.",
        suffix=suffix,
    )

    os.close(descriptor)

    temporary = Path(temporary_name)

    try:
        yield temporary
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    if not temporary.exists():
        raise ResourceNotFoundError(
            "Atomic write produced no file",
            path=str(target),
        )

    os.replace(temporary, target)


def human_readable_size(byte_count: int) -> str:
    """Format a byte count for logs and reports."""

    size = float(byte_count)

    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size) < 1024.0 or unit == "TB":
            if unit == "B":
                return f"{int(size)} {unit}"

            return f"{size:.1f} {unit}"

        size /= 1024.0

    return f"{size:.1f} TB"
