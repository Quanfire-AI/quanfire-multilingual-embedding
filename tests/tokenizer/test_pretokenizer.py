"""Tests for pre-tokenizers, with span correctness as the core invariant."""

from __future__ import annotations

import pytest

from multilingual_embedding.corpus.token import Token
from multilingual_embedding.tokenizer.pretokenizer import (
    PRETOKENIZERS,
    CharacterPreTokenizer,
    PreTokenizer,
    PunctuationPreTokenizer,
    ScriptAwarePreTokenizer,
    WhitespacePreTokenizer,
)

ENGLISH = "The quick brown fox, jumps!"

HINDI = "नमस्ते दुनिया, आप कैसे हैं?"

JAPANESE = "これは日本語のテキストです。"

ARABIC = "مرحبا بالعالم، كيف حالك؟"

CHINESE = "这是一个中文句子。"

MIXED = "Hello 世界 नमस्ते world 123!"

SAMPLES = [ENGLISH, HINDI, JAPANESE, ARABIC, CHINESE, MIXED]

ALL_KEYS = sorted(PRETOKENIZERS.keys())


def assert_spans_are_exact(tokens: list[Token], text: str) -> None:
    """Every token must slice back to itself, in left-to-right order."""

    previous_end = 0

    for token in tokens:
        assert token.text == text[token.span.start : token.span.end]

        assert token.span.length == len(token.text)

        assert token.span.start >= previous_end

        previous_end = token.span.end

    assert previous_end <= len(text)


class TestRegistry:
    def test_every_documented_key_is_registered(self) -> None:
        assert set(ALL_KEYS) == {"whitespace", "char", "punctuation", "script"}

    @pytest.mark.parametrize("key", ALL_KEYS)
    @pytest.mark.parametrize("text", SAMPLES)
    def test_spans_are_exact_for_every_implementation(self, key: str, text: str) -> None:
        pretokenizer = PRETOKENIZERS.create(key)

        assert isinstance(pretokenizer, PreTokenizer)

        assert_spans_are_exact(pretokenizer.pre_tokenize(text), text)

    @pytest.mark.parametrize("key", ALL_KEYS)
    def test_empty_text_yields_no_tokens(self, key: str) -> None:
        assert PRETOKENIZERS.create(key).pre_tokenize("") == []

    @pytest.mark.parametrize("key", ALL_KEYS)
    def test_whitespace_only_text_yields_no_tokens(self, key: str) -> None:
        assert PRETOKENIZERS.create(key).pre_tokenize("   \n\t ") == []


class TestWhitespacePreTokenizer:
    def test_splits_english_on_spaces(self) -> None:
        tokens = WhitespacePreTokenizer().pre_tokenize(ENGLISH)

        assert [token.text for token in tokens] == [
            "The",
            "quick",
            "brown",
            "fox,",
            "jumps!",
        ]

    def test_leading_and_repeated_whitespace(self) -> None:
        text = "  a\t\tb\n c "

        tokens = WhitespacePreTokenizer().pre_tokenize(text)

        assert [token.text for token in tokens] == ["a", "b", "c"]

        assert_spans_are_exact(tokens, text)

    def test_japanese_becomes_one_token_which_is_why_script_exists(self) -> None:
        # The failure mode ScriptAwarePreTokenizer is designed to avoid.
        tokens = WhitespacePreTokenizer().pre_tokenize(JAPANESE)

        assert len(tokens) == 1


class TestCharacterPreTokenizer:
    def test_one_token_per_non_space_character(self) -> None:
        text = "ab c"

        tokens = CharacterPreTokenizer().pre_tokenize(text)

        assert [token.text for token in tokens] == ["a", "b", "c"]

        assert [token.span.start for token in tokens] == [0, 1, 3]

    def test_devanagari_combining_marks_are_separate_codepoints(self) -> None:
        tokens = CharacterPreTokenizer().pre_tokenize("नमस्ते")

        assert len(tokens) == len("नमस्ते")

        assert_spans_are_exact(tokens, "नमस्ते")


class TestPunctuationPreTokenizer:
    def test_punctuation_is_peeled_into_its_own_tokens(self) -> None:
        tokens = PunctuationPreTokenizer().pre_tokenize("hello, world!")

        assert [token.text for token in tokens] == ["hello", ",", "world", "!"]

    def test_english_sample(self) -> None:
        tokens = PunctuationPreTokenizer().pre_tokenize(ENGLISH)

        assert [token.text for token in tokens] == [
            "The",
            "quick",
            "brown",
            "fox",
            ",",
            "jumps",
            "!",
        ]

        assert_spans_are_exact(tokens, ENGLISH)

    def test_hindi_danda_is_separated(self) -> None:
        text = "यह एक वाक्य है।"

        tokens = PunctuationPreTokenizer().pre_tokenize(text)

        assert tokens[-1].text == "।"

        assert_spans_are_exact(tokens, text)

    def test_arabic_comma_and_question_mark(self) -> None:
        tokens = PunctuationPreTokenizer().pre_tokenize(ARABIC)

        surfaces = [token.text for token in tokens]

        assert "،" in surfaces

        assert "؟" in surfaces

    def test_symbols_are_treated_as_punctuation(self) -> None:
        tokens = PunctuationPreTokenizer().pre_tokenize("cost=5$")

        assert [token.text for token in tokens] == ["cost", "=", "5", "$"]


class TestScriptAwarePreTokenizer:
    def test_japanese_yields_one_token_per_character(self) -> None:
        tokens = ScriptAwarePreTokenizer().pre_tokenize(JAPANESE)

        assert [token.text for token in tokens] == list(JAPANESE)

        assert_spans_are_exact(tokens, JAPANESE)

    def test_chinese_yields_one_token_per_character(self) -> None:
        tokens = ScriptAwarePreTokenizer().pre_tokenize(CHINESE)

        assert [token.text for token in tokens] == list(CHINESE)

    def test_english_yields_word_tokens_not_characters(self) -> None:
        tokens = ScriptAwarePreTokenizer().pre_tokenize(ENGLISH)

        assert [token.text for token in tokens] == [
            "The",
            "quick",
            "brown",
            "fox",
            ",",
            "jumps",
            "!",
        ]

    def test_hindi_yields_word_tokens(self) -> None:
        tokens = ScriptAwarePreTokenizer().pre_tokenize(HINDI)

        assert [token.text for token in tokens] == [
            "नमस्ते",
            "दुनिया",
            ",",
            "आप",
            "कैसे",
            "हैं",
            "?",
        ]

        assert_spans_are_exact(tokens, HINDI)

    def test_arabic_yields_word_tokens(self) -> None:
        tokens = ScriptAwarePreTokenizer().pre_tokenize("مرحبا بالعالم")

        assert [token.text for token in tokens] == ["مرحبا", "بالعالم"]

    def test_script_boundaries_split_runs_without_whitespace(self) -> None:
        text = "abc世界def"

        tokens = ScriptAwarePreTokenizer().pre_tokenize(text)

        assert [token.text for token in tokens] == ["abc", "世", "界", "def"]

        assert_spans_are_exact(tokens, text)

    def test_mixed_script_sentence(self) -> None:
        tokens = ScriptAwarePreTokenizer().pre_tokenize(MIXED)

        surfaces = [token.text for token in tokens]

        assert surfaces == ["Hello", "世", "界", "नमस्ते", "world", "123", "!"]

        assert_spans_are_exact(tokens, MIXED)

    def test_digits_and_punctuation_alone_do_not_start_a_run(self) -> None:
        text = "123 !!! ???"

        tokens = ScriptAwarePreTokenizer().pre_tokenize(text)

        assert "".join(token.text for token in tokens) == "123!!!???"

        assert_spans_are_exact(tokens, text)

    def test_japanese_kana_and_kanji_all_split_per_character(self) -> None:
        text = "カタカナひらがな漢字"

        tokens = ScriptAwarePreTokenizer().pre_tokenize(text)

        assert len(tokens) == len(text)

    def test_thai_splits_per_character(self) -> None:
        text = "สวัสดีชาวโลก"

        tokens = ScriptAwarePreTokenizer().pre_tokenize(text)

        assert len(tokens) == len(text)

        assert_spans_are_exact(tokens, text)


class TestCallInterface:
    @pytest.mark.parametrize("key", ALL_KEYS)
    def test_call_matches_pre_tokenize(self, key: str) -> None:
        pretokenizer = PRETOKENIZERS.create(key)

        # Tokens carry creation timestamps, so compare the parts that
        # define the segmentation rather than the objects themselves.
        called = [(token.text, token.span) for token in pretokenizer(MIXED)]

        direct = [(token.text, token.span) for token in pretokenizer.pre_tokenize(MIXED)]

        assert called == direct
