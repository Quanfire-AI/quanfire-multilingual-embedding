"""
Text, JSON, JSON Lines and YAML input/output.

Every reader here transparently handles gzip, selected by a ``.gz``
suffix. Corpora are routinely distributed compressed, and making that
invisible at the call site means readers never branch on it.

Every writer routes through
:func:`~multilingual_embedding.utils.filesystem.atomic_write_path`, so an
interrupted run cannot leave a half-written file where a complete one is
expected.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any

import yaml

from multilingual_embedding.common.constants import DEFAULT_ENCODING
from multilingual_embedding.core.exceptions import SerializationError
from multilingual_embedding.utils.filesystem import atomic_write_path, require_file

__all__ = [
    "count_lines",
    "open_text",
    "read_json",
    "read_jsonl",
    "read_text",
    "read_yaml",
    "write_json",
    "write_jsonl",
    "write_text",
    "write_yaml",
]


@contextmanager
def open_text(
    path: str | Path,
    mode: str = "r",
    *,
    encoding: str = DEFAULT_ENCODING,
) -> Iterator[IO[str]]:
    """
    Open a text file, decompressing transparently when it ends in ``.gz``.

    Parameters
    ----------
    path:
        File to open.

    mode:
        Text mode such as ``"r"``, ``"w"`` or ``"a"``. A ``"t"`` is added
        automatically for the gzip path.

    encoding:
        Character encoding. Defaults to UTF-8, which every multilingual
        corpus in this framework is assumed to use.
    """

    resolved = Path(path).expanduser()

    if resolved.suffix == ".gz":
        text_mode = mode if "t" in mode else f"{mode}t"

        # The handle is closed in this contextmanager's own `finally` below;
        # a contextmanager cannot use `with` for the handle it yields.
        handle: IO[str] = gzip.open(  # type: ignore[assignment]  # noqa: SIM115
            resolved, text_mode, encoding=encoding
        )
    else:
        handle = resolved.open(mode, encoding=encoding)

    try:
        yield handle
    finally:
        handle.close()


def read_text(path: str | Path, *, encoding: str = DEFAULT_ENCODING) -> str:
    """Read an entire text file into memory."""

    require_file(path)

    with open_text(path, "r", encoding=encoding) as handle:
        return handle.read()


def write_text(
    path: str | Path,
    content: str,
    *,
    encoding: str = DEFAULT_ENCODING,
) -> Path:
    """Atomically write a string to a file, creating parent directories."""

    target = Path(path).expanduser()

    with atomic_write_path(target) as temporary:
        if target.suffix == ".gz":
            with gzip.open(temporary, "wt", encoding=encoding) as handle:
                handle.write(content)
        else:
            temporary.write_text(content, encoding=encoding)

    return target


def read_json(path: str | Path, *, encoding: str = DEFAULT_ENCODING) -> Any:
    """Read a JSON document."""

    raw = read_text(path, encoding=encoding)

    try:
        return json.loads(raw)
    except json.JSONDecodeError as error:
        raise SerializationError(
            "Invalid JSON document",
            path=str(path),
            reason=str(error),
        ) from error


def write_json(
    path: str | Path,
    data: Any,
    *,
    indent: int = 2,
    sort_keys: bool = False,
    encoding: str = DEFAULT_ENCODING,
) -> Path:
    """
    Write a JSON document.

    ``ensure_ascii`` is disabled so that non-Latin scripts are stored as
    readable characters rather than escape sequences — important when a
    human needs to inspect a Hindi or Arabic corpus file.
    """

    content = json.dumps(
        data,
        indent=indent,
        sort_keys=sort_keys,
        ensure_ascii=False,
    )

    return write_text(path, content + "\n", encoding=encoding)


def read_jsonl(
    path: str | Path,
    *,
    encoding: str = DEFAULT_ENCODING,
    skip_blank: bool = True,
) -> Iterator[Any]:
    """
    Stream a JSON Lines file one record at a time.

    Lazily evaluated, so corpora far larger than memory can be read.
    Decode failures name the line number, which matters when a single
    malformed record sits inside a million line file.
    """

    require_file(path)

    with open_text(path, "r", encoding=encoding) as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()

            if not stripped:
                if skip_blank:
                    continue

                raise SerializationError(
                    "Blank line in JSON Lines file",
                    path=str(path),
                    line=line_number,
                )

            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as error:
                raise SerializationError(
                    "Invalid JSON Lines record",
                    path=str(path),
                    line=line_number,
                    reason=str(error),
                ) from error


def write_jsonl(
    path: str | Path,
    records: Iterable[Any],
    *,
    encoding: str = DEFAULT_ENCODING,
) -> int:
    """
    Write records as JSON Lines and return the number written.

    Records are streamed to the temporary file rather than buffered, so
    memory use stays flat regardless of corpus size.
    """

    target = Path(path).expanduser()

    written = 0

    with atomic_write_path(target) as temporary:
        if target.suffix == ".gz":
            # Closed in the `finally` below; the handle must outlive this
            # branch, so it cannot be bound by a `with` here.
            handle: IO[str] = gzip.open(temporary, "wt", encoding=encoding)  # noqa: SIM115
        else:
            handle = temporary.open("w", encoding=encoding)

        try:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False))
                handle.write("\n")

                written += 1
        finally:
            handle.close()

    return written


def read_yaml(path: str | Path, *, encoding: str = DEFAULT_ENCODING) -> Any:
    """
    Read a YAML document.

    Uses ``safe_load``: configuration files must not be able to
    instantiate arbitrary Python objects.
    """

    raw = read_text(path, encoding=encoding)

    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError as error:
        raise SerializationError(
            "Invalid YAML document",
            path=str(path),
            reason=str(error),
        ) from error


def write_yaml(
    path: str | Path,
    data: Any,
    *,
    encoding: str = DEFAULT_ENCODING,
) -> Path:
    """Write a YAML document, preserving key order and Unicode."""

    content = yaml.safe_dump(
        data,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    )

    return write_text(path, content, encoding=encoding)


def count_lines(path: str | Path, *, encoding: str = DEFAULT_ENCODING) -> int:
    """Count lines in a text file without holding it in memory."""

    require_file(path)

    with open_text(path, "r", encoding=encoding) as handle:
        return sum(1 for _ in handle)
