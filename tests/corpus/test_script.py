from __future__ import annotations

from itertools import pairwise

import pytest

from multilingual_embedding.corpus.script import (
    _SCRIPT_RANGES,
    Script,
    detect_script,
    is_whitespace_delimited,
    script_histogram,
    script_of_character,
)


def test_range_table_is_sorted_and_disjoint() -> None:
    """
    Lookup binary searches this table, so ordering is load bearing.

    The module sorts and checks at import; this asserts the invariant
    survives future edits.
    """

    starts = [entry[0] for entry in _SCRIPT_RANGES]

    assert starts == sorted(starts)

    for previous, current in pairwise(_SCRIPT_RANGES):
        assert current[0] > previous[1]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hello world", Script.LATIN),
        ("नमस्ते दुनिया", Script.DEVANAGARI),
        ("வணக்கம்", Script.TAMIL),
        ("مرحبا", Script.ARABIC),
        ("中文测试", Script.HAN),
        ("こんにちは", Script.HIRAGANA),
        ("カタカナ", Script.KATAKANA),
        ("안녕하세요", Script.HANGUL),
        ("Привет", Script.CYRILLIC),
        ("Γειά", Script.GREEK),
        ("שלום", Script.HEBREW),
        ("สวัสดี", Script.THAI),
        ("বাংলা", Script.BENGALI),
    ],
)
def test_dominant_script_detection(text: str, expected: Script) -> None:
    assert detect_script(text).dominant is expected


def test_punctuation_does_not_dilute_confidence() -> None:
    """
    Shared characters must be excluded from the confidence denominator.

    Otherwise "hello, world!" would score below 1.0 for Latin purely
    because of the comma and space.
    """

    assert detect_script("Hello, world!").confidence == pytest.approx(1.0)


def test_text_without_letters_is_unknown() -> None:
    profile = detect_script("123 !!! ...")

    assert profile.dominant is Script.UNKNOWN

    assert profile.confidence == 0.0


def test_empty_text_is_unknown() -> None:
    assert detect_script("").dominant is Script.UNKNOWN


def test_mixed_script_is_flagged() -> None:
    profile = detect_script("Hello नमस्ते Hello नमस्ते")

    assert profile.is_mixed


def test_single_script_is_not_mixed() -> None:
    assert not detect_script("Hello world").is_mixed


@pytest.mark.parametrize("character", [" ", ",", "!", "5", "\n"])
def test_shared_characters_map_to_common(character: str) -> None:
    assert script_of_character(character) is Script.COMMON


def test_script_of_empty_string() -> None:
    assert script_of_character("") is Script.UNKNOWN


def test_histogram_counts_every_character() -> None:
    text = "ab नम"

    assert sum(script_histogram(text).values()) == len(text)


@pytest.mark.parametrize(
    ("script", "expected"),
    [
        (Script.LATIN, True),
        (Script.DEVANAGARI, True),
        (Script.ARABIC, True),
        (Script.HAN, False),
        (Script.HIRAGANA, False),
        (Script.KATAKANA, False),
        (Script.THAI, False),
    ],
)
def test_whitespace_delimitation(script: Script, expected: bool) -> None:
    assert is_whitespace_delimited(script) is expected
