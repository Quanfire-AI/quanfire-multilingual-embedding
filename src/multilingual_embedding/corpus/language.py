"""
Language codes and script based language inference.

Languages are identified by ISO 639 code. Two-letter ISO 639-1 codes are
used where one exists; the three-letter ISO 639-2/3 form is accepted for
languages that have no two-letter code, which includes several scheduled
languages of India — Maithili (``mai``), Santali (``sat``), Konkani
(``kok``), Dogri (``doi``), Meitei (``mni``) and Bodo (``brx``).

Where a document declares no language, :func:`infer_language` guesses
from the dominant Unicode script. That inference is deliberately
limited, and it is worth being precise about what it does and does not
claim:

*It returns nothing for scripts serving many languages with no dominant
one.* Latin, Arabic, Cyrillic and Han all return ``None``. A
plausible-looking wrong answer is worse than an honest absence.

*Where it does answer, it returns the most widely used language of that
script, not an identification.* Devanagari resolves to Hindi even though
Marathi, Nepali, Sanskrit, Maithili, Konkani, Dogri and Bodo also use
it; Bengali resolves to Bengali though Assamese shares the script. These
are defensible defaults because one language dominates usage by an order
of magnitude, but they are defaults, not detections.

Statistical language identification is out of scope for this layer. A
caller that needs to distinguish Marathi from Hindi, or Assamese from
Bengali, must set the language explicitly or use a dedicated identifier.
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
    # The 22 scheduled languages of India, plus English. Several have no
    # ISO 639-1 two-letter code and are identified by their 639-2/3 form.
    Language("hi", "Hindi", Script.DEVANAGARI),
    Language("mr", "Marathi", Script.DEVANAGARI),
    Language("ne", "Nepali", Script.DEVANAGARI),
    Language("sa", "Sanskrit", Script.DEVANAGARI),
    Language("mai", "Maithili", Script.DEVANAGARI),
    Language("kok", "Konkani", Script.DEVANAGARI),
    Language("doi", "Dogri", Script.DEVANAGARI),
    Language("brx", "Bodo", Script.DEVANAGARI),
    Language("bn", "Bengali", Script.BENGALI),
    Language("as", "Assamese", Script.BENGALI),
    Language("gu", "Gujarati", Script.GUJARATI),
    Language("pa", "Punjabi", Script.GURMUKHI),
    Language("ta", "Tamil", Script.TAMIL),
    Language("te", "Telugu", Script.TELUGU),
    Language("kn", "Kannada", Script.KANNADA),
    Language("ml", "Malayalam", Script.MALAYALAM),
    Language("or", "Odia", Script.ORIYA),
    Language("sat", "Santali", Script.OL_CHIKI),
    Language("mni", "Meitei", Script.MEETEI_MAYEK),
    Language("ar", "Arabic", Script.ARABIC),
    Language("ur", "Urdu", Script.ARABIC),
    # Sindhi and Kashmiri are written in Perso-Arabic in India and
    # Pakistan; both also have Devanagari orthographies, which this
    # single-script mapping cannot express.
    Language("sd", "Sindhi", Script.ARABIC),
    Language("ks", "Kashmiri", Script.ARABIC),
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

# Scripts served by exactly one language, or by one that dominates usage
# by an order of magnitude. Only these support inference; see the module
# docstring on what that inference does and does not claim.
_UNAMBIGUOUS_SCRIPTS: dict[Script, str] = {
    Script.DEVANAGARI: "hi",
    Script.BENGALI: "bn",
    Script.OL_CHIKI: "sat",
    Script.MEETEI_MAYEK: "mni",
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
    Normalise a language tag to a bare lowercase ISO 639 code.

    Regional and script subtags are stripped, so ``"en-GB"``, ``"EN_gb"``
    and ``"en"`` all yield ``"en"``, and ``"mai-Deva"`` yields ``"mai"``.

    Both two-letter ISO 639-1 and three-letter ISO 639-2/3 codes are
    accepted. The three-letter form is not a convenience: several
    scheduled languages of India — Maithili, Santali, Konkani, Dogri,
    Meitei and Bodo — have no two-letter code at all, so rejecting it
    would make them impossible to label.

    Raises
    ------
    ValidationError
        If the value is not a two or three letter alphabetic code.

    Example
    -------
    ::

        normalize_language_code("en-GB")    -> "en"
        normalize_language_code("MAI")      -> "mai"
        normalize_language_code("sat-Olck") -> "sat"
    """

    if not isinstance(code, str) or not code.strip():
        raise ValidationError("Language code must be a non-empty string", code=code)

    primary = code.strip().replace("_", "-").split("-")[0].lower()

    if len(primary) not in (2, 3) or not primary.isalpha():
        raise ValidationError(
            "Language code must be a two letter ISO 639-1 or three letter ISO 639-2/3 code",
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
