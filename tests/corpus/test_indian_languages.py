"""
Coverage for the 22 scheduled languages of India, plus English.

These are given their own module because they exercise the parts of the
corpus layer most likely to break quietly: eight of them share the
Devanagari script, two share Bengali, three share Perso-Arabic, and two
use scripts (Ol Chiki, Meetei Mayek) that no other supported language
uses. Six have no ISO 639-1 two-letter code at all.

Each case asserts the full chain — the script is recognised, the text
segments on its own terminator, words survive combining marks, and the
language code is both accepted and named.
"""

from __future__ import annotations

import pytest

from multilingual_embedding.corpus.language import (
    expected_script,
    language_name,
    normalize_language_code,
)
from multilingual_embedding.corpus.script import Script, detect_script
from multilingual_embedding.corpus.segmentation import split_sentences, split_words

# (code, English name, script, two-sentence sample, expected word count)
SCHEDULED_LANGUAGES: tuple[tuple[str, str, Script, str, int], ...] = (
    ("hi", "Hindi", Script.DEVANAGARI, "नमस्ते दुनिया। आज मौसम अच्छा है।", 6),
    ("en", "English", Script.LATIN, "Hello world. The weather is fine today.", 7),
    ("bn", "Bengali", Script.BENGALI, "নমস্কার বিশ্ব। আজ আবহাওয়া ভালো।", 5),
    ("mr", "Marathi", Script.DEVANAGARI, "नमस्कार जग। आज हवामान चांगले आहे।", 6),
    ("te", "Telugu", Script.TELUGU, "నమస్కారం ప్రపంచం. ఈరోజు వాతావరణం బాగుంది.", 5),
    ("ta", "Tamil", Script.TAMIL, "வணக்கம் உலகம். இன்று வானிலை நன்றாக உள்ளது.", 6),
    ("ur", "Urdu", Script.ARABIC, "ہیلو دنیا۔ آج موسم اچھا ہے۔", 6),
    ("gu", "Gujarati", Script.GUJARATI, "નમસ્તે વિશ્વ. આજે હવામાન સારું છે.", 6),
    ("kn", "Kannada", Script.KANNADA, "ನಮಸ್ಕಾರ ಜಗತ್ತು. ಇಂದು ಹವಾಮಾನ ಚೆನ್ನಾಗಿದೆ.", 5),
    ("or", "Odia", Script.ORIYA, "ନମସ୍କାର ଦୁନିଆ। ଆଜି ପାଗ ଭଲ ଅଛି।", 6),
    ("pa", "Punjabi", Script.GURMUKHI, "ਸਤ ਸ੍ਰੀ ਅਕਾਲ ਦੁਨੀਆ। ਅੱਜ ਮੌਸਮ ਚੰਗਾ ਹੈ।", 8),
    ("ml", "Malayalam", Script.MALAYALAM, "നമസ്കാരം ലോകം. ഇന്ന് കാലാവസ്ഥ നല്ലതാണ്.", 5),
    ("as", "Assamese", Script.BENGALI, "নমস্কাৰ পৃথিৱী। আজি বতৰ ভাল।", 5),
    ("mai", "Maithili", Script.DEVANAGARI, "प्रणाम दुनिया। आइ मौसम नीक अछि।", 6),
    ("sat", "Santali", Script.OL_CHIKI, "ᱡᱚᱦᱟᱨ ᱫᱤᱥᱳᱢ᱾ ᱛᱮᱦᱮᱧ ᱦᱚᱭ ᱵᱮᱥ᱾", 5),
    ("ks", "Kashmiri", Script.ARABIC, "سلام دُنیا۔ اَز موسم ژھُ خوش۔", 6),
    ("ne", "Nepali", Script.DEVANAGARI, "नमस्ते संसार। आज मौसम राम्रो छ।", 6),
    ("sa", "Sanskrit", Script.DEVANAGARI, "नमो लोकाय। अद्य ऋतुः शोभनः अस्ति।", 6),
    ("sd", "Sindhi", Script.ARABIC, "سلام دنيا۔ اڄ موسم سٺو آهي۔", 6),
    ("doi", "Dogri", Script.DEVANAGARI, "नमस्कार दुनिया। अज्ज मौसम खरा ऐ।", 6),
    ("kok", "Konkani", Script.DEVANAGARI, "नमस्कार संवसार। आयज हवामान बरें आसा।", 6),
    ("mni", "Meitei", Script.MEETEI_MAYEK, "ꯍꯦꯂꯣ ꯃꯥꯂꯦꯝ꯫ ꯉꯁꯤ ꯅꯨꯡꯁꯥ ꯐꯦ꯫", 5),
    ("brx", "Bodo", Script.DEVANAGARI, "खुलुमबाय बुहुम। दिनै मुसुख मोजां।", 5),
)

IDS = [entry[0] for entry in SCHEDULED_LANGUAGES]


def test_all_twenty_two_scheduled_languages_plus_english_are_present() -> None:
    """The list must not silently lose a language."""

    assert len(SCHEDULED_LANGUAGES) == 23


@pytest.mark.parametrize(("code", "name", "script", "text", "words"), SCHEDULED_LANGUAGES, ids=IDS)
class TestScheduledLanguage:
    def test_script_is_detected(
        self, code: str, name: str, script: Script, text: str, words: int
    ) -> None:
        assert detect_script(text).dominant is script

    def test_language_code_is_accepted(
        self, code: str, name: str, script: Script, text: str, words: int
    ) -> None:
        """
        Six of these have no ISO 639-1 code.

        Maithili, Santali, Konkani, Dogri, Meitei and Bodo are only
        expressible in the three-letter form, so rejecting it would make
        them impossible to label.
        """

        assert normalize_language_code(code) == code

    def test_language_is_named(
        self, code: str, name: str, script: Script, text: str, words: int
    ) -> None:
        assert language_name(code) == name

    def test_expected_script_is_recorded(
        self, code: str, name: str, script: Script, text: str, words: int
    ) -> None:
        assert expected_script(code) is script

    def test_segments_into_two_sentences(
        self, code: str, name: str, script: Script, text: str, words: int
    ) -> None:
        """
        Each sample carries its own script's terminator.

        Devanagari and most Indic text uses the danda, Perso-Arabic the
        Urdu full stop, Ol Chiki the mucaad and Meetei Mayek the
        cheikhei. A period-and-space rule would find one sentence in
        nearly all of these.
        """

        segments = split_sentences(text, language=code if len(code) == 2 else None)

        assert len(segments) == 2

    def test_sentence_spans_slice_back_correctly(
        self, code: str, name: str, script: Script, text: str, words: int
    ) -> None:
        for span in split_sentences(text):
            assert span.slice(text) == text[span.start : span.end]

    def test_words_survive_combining_marks(
        self, code: str, name: str, script: Script, text: str, words: int
    ) -> None:
        """
        Indic and Perso-Arabic text is dense with combining marks.

        A naive ``\\w+`` fragments these words and silently discards the
        marks; the word splitter builds its own class from the Unicode
        database to avoid that.
        """

        assert len(split_words(text)) == words


class TestSharedScripts:
    """
    Several scheduled languages share a script, which bounds what
    script-based inference can honestly claim.
    """

    def test_eight_languages_share_devanagari(self) -> None:
        devanagari = [
            code for code, _, script, _, _ in SCHEDULED_LANGUAGES if script is Script.DEVANAGARI
        ]

        assert set(devanagari) == {"hi", "mr", "ne", "sa", "mai", "doi", "kok", "brx"}

    def test_bengali_script_serves_two_languages(self) -> None:
        bengali = [
            code for code, _, script, _, _ in SCHEDULED_LANGUAGES if script is Script.BENGALI
        ]

        assert set(bengali) == {"bn", "as"}

    def test_perso_arabic_serves_three(self) -> None:
        arabic = [code for code, _, script, _, _ in SCHEDULED_LANGUAGES if script is Script.ARABIC]

        assert set(arabic) == {"ur", "ks", "sd"}

    def test_two_scripts_are_used_by_exactly_one_language(self) -> None:
        """Ol Chiki and Meetei Mayek admit unambiguous inference."""

        from multilingual_embedding.corpus.language import infer_language

        assert infer_language("ᱡᱚᱦᱟᱨ ᱫᱤᱥᱳᱢ") == "sat"

        assert infer_language("ꯍꯦꯂꯣ ꯃꯥꯂꯦꯝ") == "mni"
