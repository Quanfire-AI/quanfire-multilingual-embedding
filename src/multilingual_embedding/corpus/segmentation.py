"""
Paragraph and sentence segmentation.

Segmentation is where a framework either is or is not multilingual. A
period-and-space rule handles English and silently destroys Hindi, which
ends sentences with the danda (``।``), and Chinese, which uses ``。`` and
inserts no space afterwards.

This module returns spans rather than strings, so the caller keeps the
ability to map any unit back to its exact position in the source.

The sentence splitter is rule based. It is fast, dependency free and
predictable, and it handles the terminator inventory of every script the
framework supports. It does not attempt to resolve genuinely ambiguous
cases — an unknown abbreviation followed by a capitalised proper noun
will split. Callers needing higher accuracy should segment upstream and
feed the framework pre-split sentences, which every reader supports.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

from multilingual_embedding.common.span import Span

from .script import Script, detect_script

__all__ = [
    "SENTENCE_TERMINATORS",
    "split_paragraphs",
    "split_sentences",
    "split_words",
]

# Terminators across the supported scripts.
#
# ASCII    . ! ?      Latin, Cyrillic, Greek and most alphabetic scripts
# ।  ॥               Devanagari danda and double danda. Shared by Hindi,
#                     Marathi, Nepali, Sanskrit, Maithili, Konkani, Dogri,
#                     Bodo, and used by Bengali, Odia and Gurmukhi too.
# ۔  ؟               Urdu full stop, Arabic question mark. Also Sindhi
#                     and Kashmiri, which are written in Perso-Arabic.
# 。 ！ ？ ．          CJK fullwidth forms, used without a following space
# ።                  Ethiopic full stop
# ᱾  ᱿               Ol Chiki mucaad and double mucaad (Santali)
# ꯫                  Meetei Mayek cheikhei (Meitei/Manipuri)
SENTENCE_TERMINATORS: frozenset[str] = frozenset(".!?।॥۔؟。！？．።…᱾᱿꯫")

# Terminators that end a sentence unconditionally. CJK and Indic scripts
# do not require whitespace after a terminator, so the usual
# "terminator followed by space and a capital" heuristic never fires.
_UNCONDITIONAL_TERMINATORS: frozenset[str] = frozenset("।॥。！？．።᱾᱿꯫")

# Characters permitted between a terminator and the sentence boundary,
# so that `He said "stop!"` keeps the quote with the sentence it closes.
_TRAILING_CHARACTERS: frozenset[str] = frozenset("\"'’”)]}»›")

# Latin abbreviations that end in a period without ending a sentence.
_ABBREVIATIONS: frozenset[str] = frozenset(
    {
        "mr",
        "mrs",
        "ms",
        "dr",
        "prof",
        "sr",
        "jr",
        "st",
        "vs",
        "etc",
        "inc",
        "ltd",
        "co",
        "corp",
        "dept",
        "est",
        "fig",
        "no",
        "vol",
        "approx",
        "e.g",
        "i.e",
        "cf",
        "al",
        "ca",
        "ed",
        "eds",
        "pp",
    }
)

_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n[ \t\n]*")


@lru_cache(maxsize=1)
def _combining_mark_class() -> str:
    """
    Build a regex character class covering Unicode combining marks.

    Python's ``\\w`` does not match combining marks, so a naive word
    pattern both fragments words and silently discards characters:
    ``"नमस्ते दुनिया"`` (two words) comes back as five matches,
    ``['नमस', 'त', 'द', 'न', 'य']``, and ``"हैं"`` collapses to ``['ह']``,
    losing two of its three codepoints. Devanagari, Arabic, Hebrew and
    Thai are all affected.

    The class is derived from the Unicode database rather than hardcoded,
    so it stays correct as the database is updated. The scan covers the
    Basic Multilingual Plane, which contains every combining mark used by
    the scripts this framework supports.
    """

    ranges: list[tuple[int, int]] = []

    start: int | None = None

    previous = -2

    for codepoint in range(0x0300, 0x10000):
        if unicodedata.category(chr(codepoint)) in {"Mn", "Mc", "Me"}:
            if start is None:
                start = codepoint
            elif codepoint != previous + 1:
                ranges.append((start, previous))

                start = codepoint

            previous = codepoint

    if start is not None:
        ranges.append((start, previous))

    return "".join(
        f"\\u{low:04x}" if low == high else f"\\u{low:04x}-\\u{high:04x}" for low, high in ranges
    )


# Zero-width joiner and non-joiner are format characters, not marks, but
# they occur word-internally in Devanagari and Arabic and must not split
# a word.
_ZERO_WIDTH_JOINERS = "\\u200c\\u200d"


@lru_cache(maxsize=1)
def _word_pattern() -> re.Pattern[str]:
    """
    Compile the word pattern.

    Built on first use rather than at import. Scanning the Basic
    Multilingual Plane for combining marks measures at roughly 15 ms and
    finds about 1,300 of them — not enormous, but it is pure waste for
    the many callers that never split words, and import-time cost is
    paid by every consumer of the library whether they use the feature
    or not. Cached, so it is paid at most once per process.
    """

    character = f"[\\w{_combining_mark_class()}{_ZERO_WIDTH_JOINERS}]"

    return re.compile(f"{character}+(?:['’]{character}+)*", re.UNICODE)


def split_paragraphs(text: str) -> list[Span]:
    """
    Split text into paragraph spans on blank lines.

    Leading and trailing whitespace is trimmed from each paragraph, and
    empty paragraphs are dropped, so consecutive blank lines do not
    produce empty units.

    Returns
    -------
    Ordered, non-overlapping spans into ``text``.
    """

    if not text.strip():
        return []

    spans: list[Span] = []

    cursor = 0

    for match in _PARAGRAPH_BREAK.finditer(text):
        span = _trimmed_span(text, cursor, match.start())

        if span is not None:
            spans.append(span)

        cursor = match.end()

    final = _trimmed_span(text, cursor, len(text))

    if final is not None:
        spans.append(final)

    return spans


def split_sentences(text: str, *, language: str | None = None) -> list[Span]:
    """
    Split text into sentence spans.

    Parameters
    ----------
    text:
        Text to segment. Usually a single paragraph.

    language:
        Optional ISO 639-1 hint. Currently used only to skip the Latin
        abbreviation checks for scripts where they do not apply, which
        is a small speed win on large non-Latin corpora.

    Returns
    -------
    Ordered, non-overlapping spans covering the non-whitespace content.

    Example
    -------
    ::

        split_sentences("Hello there. How are you?")
        -> [Span(0, 12), Span(13, 25)]

        split_sentences("नमस्ते। आप कैसे हैं?")
        -> two spans, split on the danda
    """

    if not text.strip():
        return []

    check_abbreviations = _should_check_abbreviations(text, language)

    spans: list[Span] = []

    start = 0

    index = 0

    length = len(text)

    while index < length:
        character = text[index]

        if character not in SENTENCE_TERMINATORS:
            index += 1
            continue

        boundary = index + 1

        # Absorb repeated terminators ("...", "?!") into one boundary.
        while boundary < length and text[boundary] in SENTENCE_TERMINATORS:
            boundary += 1

        # Absorb closing quotes and brackets.
        while boundary < length and text[boundary] in _TRAILING_CHARACTERS:
            boundary += 1

        if _is_sentence_boundary(
            text,
            terminator_index=index,
            boundary=boundary,
            check_abbreviations=check_abbreviations,
        ):
            span = _trimmed_span(text, start, boundary)

            if span is not None:
                spans.append(span)

            start = boundary

            index = boundary
        else:
            index = boundary

    final = _trimmed_span(text, start, length)

    if final is not None:
        spans.append(final)

    return spans


def split_words(text: str) -> list[Span]:
    """
    Split text into word spans using Unicode word characters.

    Intended for statistics and for whitespace pre-tokenization of
    alphabetic scripts. For scripts without whitespace word boundaries —
    Han, Hiragana, Katakana, Thai — this returns whole runs rather than
    words, which is why the tokenizer layer relies on subword models
    there rather than on this function.
    """

    return [Span(match.start(), match.end()) for match in _word_pattern().finditer(text)]


def _trimmed_span(text: str, start: int, end: int) -> Span | None:
    """
    Return the span with surrounding whitespace removed, or None if blank.
    """

    while start < end and text[start].isspace():
        start += 1

    while end > start and text[end - 1].isspace():
        end -= 1

    return Span(start, end) if end > start else None


def _should_check_abbreviations(text: str, language: str | None) -> bool:
    """
    Decide whether Latin abbreviation rules are worth applying.

    They only matter for scripts that use a bare period as terminator.
    """

    if language is not None:
        return language.lower() in {
            "en",
            "fr",
            "de",
            "es",
            "pt",
            "it",
            "nl",
            "id",
            "vi",
            "tr",
        }

    return detect_script(text).dominant in {
        Script.LATIN,
        Script.CYRILLIC,
        Script.GREEK,
        Script.UNKNOWN,
    }


def _is_sentence_boundary(
    text: str,
    *,
    terminator_index: int,
    boundary: int,
    check_abbreviations: bool,
) -> bool:
    """
    Decide whether a terminator actually ends a sentence.
    """

    terminator = text[terminator_index]

    # Indic and CJK terminators are unambiguous and need no following space.
    if terminator in _UNCONDITIONAL_TERMINATORS:
        return True

    # End of text always closes the final sentence.
    if boundary >= len(text):
        return True

    following = text[boundary]

    # A terminator not followed by whitespace is usually internal:
    # a decimal point, a URL, a version number, an ellipsis mid-word.
    if not following.isspace():
        return False

    if (
        terminator == "."
        and check_abbreviations
        and _ends_with_abbreviation(text, terminator_index)
    ):
        return False

    # Require the next sentence to begin with something that can start
    # one. A lowercase letter after a period nearly always means the
    # period was internal to an abbreviation this module does not know.
    next_index = boundary

    while next_index < len(text) and text[next_index].isspace():
        next_index += 1

    if next_index >= len(text):
        return True

    successor = text[next_index]

    return not successor.islower()


def _ends_with_abbreviation(text: str, period_index: int) -> bool:
    """
    Return True if the period at ``period_index`` closes an abbreviation.

    Covers two cases: a known abbreviation from the list above, and a
    single-letter initial such as the ``J.`` in ``J. R. R. Tolkien``.
    """

    start = period_index

    while start > 0 and (text[start - 1].isalpha() or text[start - 1] == "."):
        start -= 1

    word = text[start:period_index].lower().strip(".")

    if not word:
        return False

    if word in _ABBREVIATIONS:
        return True

    # Single letter followed by a period: an initial, not a sentence end.
    return len(word) == 1 and word.isalpha()
