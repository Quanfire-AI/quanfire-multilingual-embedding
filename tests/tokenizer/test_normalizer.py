"""Tests for the normalizer registry, implementations and pipeline."""

from __future__ import annotations

import pytest

from multilingual_embedding.core.exceptions import ConfigurationError, RegistryError
from multilingual_embedding.tokenizer.normalizer import (
    NORMALIZERS,
    DigitNormalizer,
    LowercaseNormalizer,
    NFCNormalizer,
    NFDNormalizer,
    NFKCNormalizer,
    NFKDNormalizer,
    Normalizer,
    NormalizerPipeline,
    StripAccentsNormalizer,
    WhitespaceNormalizer,
)

ZERO_WIDTH_JOINER = "‍"

ZERO_WIDTH_NON_JOINER = "‌"

ZERO_WIDTH_SPACE = "​"

BYTE_ORDER_MARK = "﻿"


class TestRegistry:
    def test_every_documented_key_is_registered(self) -> None:
        assert set(NORMALIZERS.keys()) == {
            "nfc",
            "nfd",
            "nfkc",
            "nfkd",
            "lowercase",
            "whitespace",
            "strip_accents",
            "digits",
        }

    @pytest.mark.parametrize("key", sorted(NORMALIZERS.keys()))
    def test_created_instances_are_normalizers(self, key: str) -> None:
        instance = NORMALIZERS.create(key)

        assert isinstance(instance, Normalizer)

        assert isinstance(instance.normalize("hello"), str)


class TestUnicodeNormalizers:
    def test_nfkc_folds_fullwidth_latin_used_in_cjk_text(self) -> None:
        assert NFKCNormalizer().normalize("ＡＢＣ１２３") == "ABC123"

    def test_nfkc_folds_arabic_presentation_forms(self) -> None:
        # U+FE8E is the final-form glyph variant of alef; NFKC folds it
        # back onto the plain letter U+0627 so the two spellings of the
        # same word do not become two vocabulary entries.
        assert NFKCNormalizer().normalize("ﺎ") == "ا"

        assert NFKCNormalizer().normalize("ﻋﺎﻟﻢ") == "عالم"

    def test_nfc_composes_decomposed_accents(self) -> None:
        decomposed = "café"

        assert NFCNormalizer().normalize(decomposed) == "café"

        assert len(NFCNormalizer().normalize(decomposed)) == 4

    def test_nfc_preserves_fullwidth_forms_that_nfkc_folds(self) -> None:
        assert NFCNormalizer().normalize("ＡＢＣ") == "ＡＢＣ"

    def test_nfd_decomposes(self) -> None:
        assert NFDNormalizer().normalize("café") == "café"

    def test_nfkd_decomposes_and_folds_compatibility(self) -> None:
        assert NFKDNormalizer().normalize("Ａé") == "Aé"

    def test_devanagari_survives_nfkc_unchanged(self) -> None:
        text = "नमस्ते दुनिया"

        assert NFKCNormalizer().normalize(text) == text

    def test_invalid_form_is_rejected(self) -> None:
        from multilingual_embedding.core.exceptions import ValidationError
        from multilingual_embedding.tokenizer.normalizer import _UnicodeNormalizer

        with pytest.raises(ValidationError):
            _UnicodeNormalizer("NFX")


class TestLowercaseNormalizer:
    def test_ascii(self) -> None:
        assert LowercaseNormalizer().normalize("HELLO World") == "hello world"

    def test_casefold_unifies_german_sharp_s(self) -> None:
        normalizer = LowercaseNormalizer()

        # This is exactly what `.lower()` would fail to do: it leaves ß
        # alone, so STRASSE and straße would stay distinct types.
        assert normalizer.normalize("STRASSE") == normalizer.normalize("straße")

    def test_casefold_unifies_greek_final_sigma(self) -> None:
        normalizer = LowercaseNormalizer()

        # `.lower()` leaves final sigma as ς, so the same word spelled
        # with medial and final sigma would not unify.
        assert normalizer.normalize("ς") == "σ"

        assert normalizer.normalize("ΟΔΟΣ") == "οδοσ"

    def test_uncased_scripts_are_untouched(self) -> None:
        for text in ("नमस्ते", "こんにちは", "مرحبا", "世界"):
            assert LowercaseNormalizer().normalize(text) == text


class TestWhitespaceNormalizer:
    def test_collapses_runs_and_strips(self) -> None:
        assert WhitespaceNormalizer().normalize("  hello   world \n\t ") == "hello world"

    def test_folds_no_break_and_ideographic_spaces(self) -> None:
        assert WhitespaceNormalizer().normalize("a b　c") == "a b c"

    def test_removes_zero_width_space_and_byte_order_mark(self) -> None:
        text = f"{BYTE_ORDER_MARK}hello{ZERO_WIDTH_SPACE}world"

        assert WhitespaceNormalizer().normalize(text) == "helloworld"

    def test_preserves_joiners_in_devanagari(self) -> None:
        # क् + ZWNJ + ष suppresses the क्ष conjunct; क् + ZWJ + ष forces the
        # half-form. Stripping either silently merges distinct spellings.
        text = f"क्{ZERO_WIDTH_NON_JOINER}ष और क्{ZERO_WIDTH_JOINER}ष"

        result = WhitespaceNormalizer().normalize(text)

        assert ZERO_WIDTH_NON_JOINER in result

        assert ZERO_WIDTH_JOINER in result

        assert result == text

    def test_preserves_joiner_in_arabic(self) -> None:
        text = f"ال{ZERO_WIDTH_NON_JOINER}عالم"

        assert WhitespaceNormalizer().normalize(text) == text

    def test_empty_input(self) -> None:
        assert WhitespaceNormalizer().normalize("   ") == ""


class TestStripAccentsNormalizer:
    def test_removes_latin_accents(self) -> None:
        assert StripAccentsNormalizer().normalize("café naïve") == "cafe naive"

    def test_removes_arabic_harakat(self) -> None:
        vocalised = "مَرْحَبًا"

        result = StripAccentsNormalizer().normalize(vocalised)

        assert result == "مرحبا"

    def test_leaves_unaccented_text_alone(self) -> None:
        assert StripAccentsNormalizer().normalize("hello 世界") == "hello 世界"


class TestDigitNormalizer:
    def test_devanagari_numerals_unify_with_ascii(self) -> None:
        normalizer = DigitNormalizer()

        assert normalizer.normalize("२०२४") == "2024"

        assert normalizer.normalize("२०२४") == normalizer.normalize("2024")

    def test_arabic_indic_numerals(self) -> None:
        assert DigitNormalizer().normalize("٢٠٢٤") == "2024"

    def test_fullwidth_numerals(self) -> None:
        assert DigitNormalizer().normalize("２０２４") == "2024"

    def test_surrounding_text_is_preserved(self) -> None:
        assert DigitNormalizer().normalize("साल २०२४ में") == "साल 2024 में"

    def test_non_decimal_numeric_forms_are_untouched(self) -> None:
        # U+00BD VULGAR FRACTION ONE HALF has no single ASCII digit.
        assert DigitNormalizer().normalize("½") == "½"

    def test_text_without_digits_is_returned_unchanged(self) -> None:
        assert DigitNormalizer().normalize("hello") == "hello"


class TestNormalizerPipeline:
    def test_applies_steps_in_order(self) -> None:
        pipeline = NormalizerPipeline([NFKCNormalizer(), LowercaseNormalizer()])

        assert pipeline.normalize("ＨＥＬＬＯ") == "hello"

        assert len(pipeline) == 2

    def test_empty_pipeline_is_identity(self) -> None:
        assert NormalizerPipeline().normalize("Hello") == "Hello"

        assert NormalizerPipeline.from_config(None).normalize("Hello") == "Hello"

    def test_from_config_accepts_mappings_and_bare_strings(self) -> None:
        pipeline = NormalizerPipeline.from_config(
            [{"type": "nfkc"}, "lowercase", {"type": "whitespace"}]
        )

        assert len(pipeline) == 3

        assert pipeline.normalize("  ＨＥＬＬＯ　 Ｗｏｒｌｄ ") == "hello world"

    def test_from_config_rejects_unknown_type(self) -> None:
        with pytest.raises(RegistryError):
            NormalizerPipeline.from_config([{"type": "no_such_normalizer"}])

    def test_from_config_rejects_missing_type_key(self) -> None:
        with pytest.raises(ConfigurationError):
            NormalizerPipeline.from_config([{"form": "nfkc"}])

    def test_multilingual_pipeline_end_to_end(self) -> None:
        pipeline = NormalizerPipeline.from_config(
            [{"type": "nfkc"}, {"type": "lowercase"}, {"type": "digits"}, {"type": "whitespace"}]
        )

        assert pipeline.normalize("  Hello　ＷＯＲＬＤ  २०२४ ") == "hello world 2024"

        assert pipeline.normalize("नमस्ते  दुनिया") == "नमस्ते दुनिया"

    def test_repr_lists_steps(self) -> None:
        pipeline = NormalizerPipeline([LowercaseNormalizer()])

        assert "LowercaseNormalizer" in repr(pipeline)

    def test_call_is_equivalent_to_normalize(self) -> None:
        pipeline = NormalizerPipeline([LowercaseNormalizer()])

        assert pipeline("ABC") == pipeline.normalize("ABC")
