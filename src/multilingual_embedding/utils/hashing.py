"""
Stable content hashing.

Reproducibility depends on identifiers that do not change between
processes. Python's builtin :func:`hash` is randomised per interpreter
run, so the framework uses SHA-256 over a canonical UTF-8 encoding
instead.

Text is normalised to Unicode NFC before hashing so that two byte
sequences representing the same characters — a real concern for the
Devanagari and Hangul content this framework targets — produce the same
identifier.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from multilingual_embedding.common.constants import DEFAULT_ENCODING

__all__ = [
    "DEFAULT_DIGEST_SIZE",
    "hash_bytes",
    "hash_file",
    "hash_iterable",
    "hash_object",
    "hash_text",
]

DEFAULT_DIGEST_SIZE = 16

_FILE_CHUNK_SIZE = 1 << 20


def hash_text(text: str, *, digest_size: int = DEFAULT_DIGEST_SIZE) -> str:
    """
    Return a stable hex digest for a string.

    Parameters
    ----------
    text:
        Input text. Normalised to NFC before hashing.

    digest_size:
        Number of hex characters to return. The full SHA-256 digest is
        64 characters; the default truncation is ample for identifying
        documents within a corpus while staying readable in logs.
    """

    normalized = unicodedata.normalize("NFC", text)

    return hash_bytes(normalized.encode(DEFAULT_ENCODING), digest_size=digest_size)


def hash_bytes(data: bytes, *, digest_size: int = DEFAULT_DIGEST_SIZE) -> str:
    """Return a stable hex digest for a byte string."""

    if digest_size <= 0 or digest_size > 64:
        raise ValueError("digest_size must lie in [1, 64]")

    return hashlib.sha256(data).hexdigest()[:digest_size]


def hash_file(path: str | Path, *, digest_size: int = DEFAULT_DIGEST_SIZE) -> str:
    """
    Return a stable hex digest for a file's contents.

    The file is streamed in chunks, so corpora larger than memory are
    handled without issue.
    """

    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:
        while chunk := handle.read(_FILE_CHUNK_SIZE):
            digest.update(chunk)

    if digest_size <= 0 or digest_size > 64:
        raise ValueError("digest_size must lie in [1, 64]")

    return digest.hexdigest()[:digest_size]


def hash_object(obj: Any, *, digest_size: int = DEFAULT_DIGEST_SIZE) -> str:
    """
    Return a stable hex digest for a JSON serialisable object.

    Mapping keys are sorted so that two dictionaries with the same
    contents but different insertion order hash identically. This makes
    the digest usable as a cache key for a configuration block.
    """

    encoded = json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )

    return hash_text(encoded, digest_size=digest_size)


def hash_iterable(values: Iterable[str], *, digest_size: int = DEFAULT_DIGEST_SIZE) -> str:
    """
    Return a stable, order sensitive digest for a sequence of strings.

    A length prefix delimits each element so that ``["ab", "c"]`` and
    ``["a", "bc"]`` do not collide.
    """

    digest = hashlib.sha256()

    for value in values:
        encoded = unicodedata.normalize("NFC", value).encode(DEFAULT_ENCODING)

        digest.update(str(len(encoded)).encode("ascii"))
        digest.update(b":")
        digest.update(encoded)

    if digest_size <= 0 or digest_size > 64:
        raise ValueError("digest_size must lie in [1, 64]")

    return digest.hexdigest()[:digest_size]
