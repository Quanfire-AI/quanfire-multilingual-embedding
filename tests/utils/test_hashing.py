from __future__ import annotations

import unicodedata
from pathlib import Path

import pytest

from multilingual_embedding.utils.hashing import (
    hash_bytes,
    hash_file,
    hash_iterable,
    hash_object,
    hash_text,
)


def test_hash_is_stable_across_calls() -> None:
    assert hash_text("hello") == hash_text("hello")


def test_hash_differs_for_different_input() -> None:
    assert hash_text("hello") != hash_text("world")


def test_hash_normalizes_unicode() -> None:
    """
    The same characters in NFC and NFD must hash identically.

    Devanagari and Hangul text routinely arrives in either form, and two
    spellings of one document must not become two corpus entries.
    """

    composed = unicodedata.normalize("NFC", "é")

    decomposed = unicodedata.normalize("NFD", "é")

    assert composed != decomposed

    assert hash_text(composed) == hash_text(decomposed)


def test_digest_size_is_respected() -> None:
    assert len(hash_text("hello", digest_size=8)) == 8

    assert len(hash_text("hello", digest_size=64)) == 64


@pytest.mark.parametrize("size", [0, -1, 65])
def test_invalid_digest_size_rejected(size: int) -> None:
    with pytest.raises(ValueError):
        hash_text("hello", digest_size=size)


def test_hash_bytes() -> None:
    assert hash_bytes(b"abc") == hash_bytes(b"abc")


def test_hash_file(tmp_path: Path) -> None:
    path = tmp_path / "file.txt"

    path.write_text("content", encoding="utf-8")

    assert hash_file(path) == hash_text("content")


def test_hash_object_is_key_order_independent() -> None:
    assert hash_object({"a": 1, "b": 2}) == hash_object({"b": 2, "a": 1})


def test_hash_object_differs_on_value_change() -> None:
    assert hash_object({"a": 1}) != hash_object({"a": 2})


def test_hash_iterable_is_order_sensitive() -> None:
    assert hash_iterable(["a", "b"]) != hash_iterable(["b", "a"])


def test_hash_iterable_avoids_boundary_collision() -> None:
    """Length prefixing must keep ["ab","c"] distinct from ["a","bc"]."""

    assert hash_iterable(["ab", "c"]) != hash_iterable(["a", "bc"])
