"""
Unicode script identification.

Script awareness is what makes the rest of the pipeline genuinely
multilingual rather than Latin-with-exceptions. Segmentation rules,
tokenizer character coverage and corpus statistics all branch on it.

Detection uses explicit codepoint ranges rather than
:func:`unicodedata.name` lookups: ranges are an order of magnitude
faster over large corpora and give the same answer for the scripts this
framework targets.
"""

from __future__ import annotations

import unicodedata
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise

__all__ = [
    "Script",
    "detect_script",
    "is_whitespace_delimited",
    "script_histogram",
    "script_of_character",
]


class Script(StrEnum):
    """
    ISO 15924 script codes for the scripts the framework recognises.

    ``UNKNOWN`` covers unassigned or unrecognised codepoints;
    ``COMMON`` covers punctuation, digits and symbols shared across
    scripts, which must not be counted as evidence for any one script.
    """

    LATIN = "Latn"
    DEVANAGARI = "Deva"
    BENGALI = "Beng"
    GUJARATI = "Gujr"
    GURMUKHI = "Guru"
    TAMIL = "Taml"
    TELUGU = "Telu"
    KANNADA = "Knda"
    MALAYALAM = "Mlym"
    ORIYA = "Orya"
    ARABIC = "Arab"
    HEBREW = "Hebr"
    CYRILLIC = "Cyrl"
    GREEK = "Grek"
    HAN = "Hani"
    HIRAGANA = "Hira"
    KATAKANA = "Kana"
    HANGUL = "Hang"
    THAI = "Thai"
    ETHIOPIC = "Ethi"
    COMMON = "Zyyy"
    UNKNOWN = "Zzzz"


# (start, end, script) triples, inclusive on both ends, grouped by script
# for readability. Sorted by start codepoint at import time rather than by
# hand, because lookup binary searches this table and a manually misplaced
# row would silently misclassify a whole script.
_UNSORTED_SCRIPT_RANGES: tuple[tuple[int, int, Script], ...] = (
    (0x0041, 0x005A, Script.LATIN),
    (0x0061, 0x007A, Script.LATIN),
    (0x00C0, 0x024F, Script.LATIN),
    (0x0370, 0x03FF, Script.GREEK),
    (0x0400, 0x04FF, Script.CYRILLIC),
    (0x0500, 0x052F, Script.CYRILLIC),
    (0x0590, 0x05FF, Script.HEBREW),
    (0x0600, 0x06FF, Script.ARABIC),
    (0x0750, 0x077F, Script.ARABIC),
    (0x0900, 0x097F, Script.DEVANAGARI),
    (0x0980, 0x09FF, Script.BENGALI),
    (0x0A00, 0x0A7F, Script.GURMUKHI),
    (0x0A80, 0x0AFF, Script.GUJARATI),
    (0x0B00, 0x0B7F, Script.ORIYA),
    (0x0B80, 0x0BFF, Script.TAMIL),
    (0x0C00, 0x0C7F, Script.TELUGU),
    (0x0C80, 0x0CFF, Script.KANNADA),
    (0x0D00, 0x0D7F, Script.MALAYALAM),
    (0x0E00, 0x0E7F, Script.THAI),
    (0x1200, 0x137F, Script.ETHIOPIC),
    (0x1E00, 0x1EFF, Script.LATIN),
    (0x2C60, 0x2C7F, Script.LATIN),
    (0x3040, 0x309F, Script.HIRAGANA),
    (0x30A0, 0x30FF, Script.KATAKANA),
    (0x3400, 0x4DBF, Script.HAN),
    (0x4E00, 0x9FFF, Script.HAN),
    (0xA720, 0xA7FF, Script.LATIN),
    (0xAC00, 0xD7AF, Script.HANGUL),
    (0x1100, 0x11FF, Script.HANGUL),
    (0x3130, 0x318F, Script.HANGUL),
    (0xF900, 0xFAFF, Script.HAN),
    (0xFB50, 0xFDFF, Script.ARABIC),
    (0xFE70, 0xFEFF, Script.ARABIC),
    (0x20000, 0x2A6DF, Script.HAN),
)

_SCRIPT_RANGES: tuple[tuple[int, int, Script], ...] = tuple(
    sorted(_UNSORTED_SCRIPT_RANGES, key=lambda entry: entry[0])
)


def _assert_ranges_disjoint() -> None:
    """
    Verify no two ranges overlap.

    An overlap would make lookup depend on which row the binary search
    happened to land on, so it is caught at import rather than becoming
    an intermittent misclassification.
    """

    for previous, current in pairwise(_SCRIPT_RANGES):
        if current[0] <= previous[1]:
            raise RuntimeError(f"Overlapping script ranges: {previous!r} and {current!r}")


_assert_ranges_disjoint()


# Scripts written without spaces between words. Whitespace pre-tokenization
# is meaningless for these, so the tokenizer must fall back to subword or
# character segmentation.
_NON_WHITESPACE_DELIMITED = frozenset(
    {
        Script.HAN,
        Script.HIRAGANA,
        Script.KATAKANA,
        Script.THAI,
    }
)


@dataclass(slots=True, frozen=True)
class ScriptProfile:
    """
    Result of analysing the script composition of a piece of text.

    Attributes
    ----------
    dominant:
        Script accounting for the largest share of script-bearing
        characters.

    confidence:
        Dominant script's share of script-bearing characters, in [0, 1].
        Punctuation, digits and whitespace are excluded from the
        denominator, so ``"hello, world!"`` scores 1.0 for Latin rather
        than being diluted by the comma and space.

    counts:
        Per script character counts, including ``COMMON``.
    """

    dominant: Script

    confidence: float

    counts: dict[Script, int]

    @property
    def is_mixed(self) -> bool:
        """True when no single script accounts for 80% of the text."""

        return self.confidence < 0.8


def script_of_character(character: str) -> Script:
    """
    Return the script of a single character.

    Whitespace, punctuation, digits and symbols map to
    :attr:`Script.COMMON`, since they carry no script evidence.
    """

    if not character:
        return Script.UNKNOWN

    codepoint = ord(character[0])

    category = unicodedata.category(character[0])

    # Separators, punctuation, symbols, numbers and control characters
    # appear across every script and must not sway detection.
    if category[0] in {"Z", "P", "S", "N", "C"}:
        return Script.COMMON

    low = 0

    high = len(_SCRIPT_RANGES) - 1

    while low <= high:
        middle = (low + high) // 2

        start, end, script = _SCRIPT_RANGES[middle]

        if codepoint < start:
            high = middle - 1
        elif codepoint > end:
            low = middle + 1
        else:
            return script

    # Combining marks belong to the script of the base character they
    # follow; callers see them through detect_script, which has context.
    if category == "Mn":
        return Script.COMMON

    return Script.UNKNOWN


def script_histogram(text: str) -> dict[Script, int]:
    """Return per script character counts for ``text``."""

    counter: Counter[Script] = Counter()

    for character in text:
        counter[script_of_character(character)] += 1

    return dict(counter)


def detect_script(text: str) -> ScriptProfile:
    """
    Identify the dominant script of a string.

    Example
    -------
    ::

        detect_script("Hello world").dominant     -> Script.LATIN
        detect_script("नमस्ते दुनिया").dominant      -> Script.DEVANAGARI
        detect_script("こんにちは").dominant        -> Script.HIRAGANA
    """

    counts = script_histogram(text)

    informative = {
        script: count
        for script, count in counts.items()
        if script not in {Script.COMMON, Script.UNKNOWN}
    }

    if not informative:
        return ScriptProfile(dominant=Script.UNKNOWN, confidence=0.0, counts=counts)

    total = sum(informative.values())

    dominant, dominant_count = max(informative.items(), key=lambda item: (item[1], item[0].value))

    return ScriptProfile(
        dominant=dominant,
        confidence=dominant_count / total,
        counts=counts,
    )


def is_whitespace_delimited(script: Script) -> bool:
    """
    Return True if words in this script are separated by whitespace.

    Drives the choice of pre-tokenizer: splitting Japanese or Thai on
    whitespace would yield whole sentences as single tokens.
    """

    return script not in _NON_WHITESPACE_DELIMITED
