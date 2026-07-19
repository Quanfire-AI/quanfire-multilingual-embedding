"""
Language codes and script based language inference.

The framework identifies languages by ISO 639-1 code. Where a document
does not declare one, :func:`infer_language` guesses from the dominant
Unicode script.

That inference is deliberately conservative. A script maps to a language
only when the correspondence is effectively one to one — Devanagari to
Hindi, Tamil to Tamil, Hangul to Korean. Latin, Arabic and Han are each
shared by many languages, so they return ``None`` rather than a guess.
Statistical language identification is out of scope for this layer; a
caller that needs it should set the language explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass

from multilingual_embedding.core.exceptions import ValidationError

from .script import Script, detect_script

__all__ = [
    "LANGUAGE_NAMES",
    "Language",
    "expected_script",
    "infer_language",
    "language_name",
    "normalize_language_code",
]


@dataclass(slots=True, frozen=True)
class Language:
    """
    A language with its conventional script.

    Attributes
    ----------
    code:
        ISO 639-1 two letter code.

    name:
        English name of the language.

    script:
        Script the language is normally written in.
    """

    code: str

    name: str

    script: Script


_LANGUAGES: tuple[Language, ...] = (
    Language("en", "English", Script.LATIN),
    Language("fr", "French", Script.LATIN),
    Language("de", "German", Script.LATIN),
    Language("es", "Spanish", Script.LATIN),
    Language("pt", "Portuguese", Script.LATIN),
    Language("it", "Italian", Script.LATIN),
    Language("nl", "Dutch", Script.LATIN),
    Language("id", "Indonesian", Script.LATIN),
    Language("vi", "Vietnamese", Script.LATIN),
    Language("tr", "Turkish", Script.LATIN),
    Language("hi", "Hindi", Script.DEVANAGARI),
    Language("mr", "Marathi", Script.DEVANAGARI),
    Language("ne", "Nepali", Script.DEVANAGARI),
    Language("bn", "Bengali", Script.BENGALI),
    Language("gu", "Gujarati", Script.GUJARATI),
    Language("pa", "Punjabi", Script.GURMUKHI),
    Language("ta", "Tamil", Script.TAMIL),
    Language("te", "Telugu", Script.TELUGU),
    Language("kn", "Kannada", Script.KANNADA),
    Language("ml", "Malayalam", Script.MALAYALAM),
    Language("or", "Odia", Script.ORIYA),
    Language("ar", "Arabic", Script.ARABIC),
    Language("ur", "Urdu", Script.ARABIC),
    Language("fa", "Persian", Script.ARABIC),
    Language("he", "Hebrew", Script.HEBREW),
    Language("ru", "Russian", Script.CYRILLIC),
    Language("uk", "Ukrainian", Script.CYRILLIC),
    Language("el", "Greek", Script.GREEK),
    Language("zh", "Chinese", Script.HAN),
    Language("ja", "Japanese", Script.HIRAGANA),
    Language("ko", "Korean", Script.HANGUL),
    Language("th", "Thai", Script.THAI),
    Language("am", "Amharic", Script.ETHIOPIC),
)

_BY_CODE: dict[str, Language] = {language.code: language for language in _LANGUAGES}

LANGUAGE_NAMES: dict[str, str] = {language.code: language.name for language in _LANGUAGES}

# Scripts used by exactly one supported language, or with one
# overwhelmingly dominant member. Only these support inference.
_UNAMBIGUOUS_SCRIPTS: dict[Script, str] = {
    Script.DEVANAGARI: "hi",
    Script.BENGALI: "bn",
    Script.GUJARATI: "gu",
    Script.GURMUKHI: "pa",
    Script.TAMIL: "ta",
    Script.TELUGU: "te",
    Script.KANNADA: "kn",
    Script.MALAYALAM: "ml",
    Script.ORIYA: "or",
    Script.HEBREW: "he",
    Script.GREEK: "el",
    Script.HANGUL: "ko",
    Script.THAI: "th",
    Script.ETHIOPIC: "am",
    Script.HIRAGANA: "ja",
    Script.KATAKANA: "ja",
}


def normalize_language_code(code: str) -> str:
    """
    Normalise a language tag to a bare lowercase ISO 639-1 code.

    Regional and script subtags are stripped, so ``"en-GB"``,
    ``"EN_gb"`` and ``"en"`` all yield ``"en"``.

    Raises
    ------
    ValidationError
        If the value is not a two letter code.
    """

    if not isinstance(code, str) or not code.strip():
        raise ValidationError("Language code must be a non-empty string", code=code)

    primary = code.strip().replace("_", "-").split("-")[0].lower()

    if len(primary) != 2 or not primary.isalpha():
        raise ValidationError(
            "Language code must be a two letter ISO 639-1 code",
            code=code,
        )

    return primary


def language_name(code: str) -> str | None:
    """Return the English name for a language code, or None if unknown."""

    language = _BY_CODE.get(normalize_language_code(code))

    return language.name if language else None


def expected_script(code: str) -> Script | None:
    """
    Return the script a language is conventionally written in.

    Returns None for languages the framework does not know about.
    """

    language = _BY_CODE.get(normalize_language_code(code))

    return language.script if language else None


def infer_language(text: str) -> str | None:
    """
    Guess a language code from the dominant script of ``text``.

    Returns None when the script is shared across several languages
    (Latin, Arabic, Cyrillic, Han) or when the text carries no script
    evidence at all. Callers should treat a None result as "unknown" and
    fall back to a configured default rather than to English.

    Example
    -------
    ::

        infer_language("नमस्ते")      -> "hi"
        infer_language("안녕하세요")   -> "ko"
        infer_language("hello")      -> None   (Latin is ambiguous)
    """

    profile = detect_script(text)

    if profile.dominant is Script.UNKNOWN:
        return None

    # A mixed-script string gives no reliable single answer.
    if profile.is_mixed:
        return None

    return _UNAMBIGUOUS_SCRIPTS.get(profile.dominant)
