"""Tests for the SentencePiece and word tokenizers."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pytest

from multilingual_embedding.config.base import TokenizerConfig
from multilingual_embedding.core.exceptions import (
    NotFittedError,
    ResourceNotFoundError,
)
from multilingual_embedding.tokenizer.encoding import Encoding
from multilingual_embedding.tokenizer.tokenizer import (
    TOKENIZERS,
    SentencePieceTokenizer,
    Tokenizer,
    WordTokenizer,
)
from multilingual_embedding.tokenizer.trainer import SentencePieceTrainerAdapter
from multilingual_embedding.vocabulary.special_tokens import BOS_ID, EOS_ID, PAD_ID, UNK_ID

TRAINABLE_VOCAB_SIZE = 150


@pytest.fixture(scope="module")
def trained_model_path(
    tmp_path_factory: pytest.TempPathFactory,
    multilingual_sentences: list[str],
) -> Path:
    """Train one SentencePiece model and share it across the module."""

    directory = tmp_path_factory.mktemp("sentencepiece-model")

    config = TokenizerConfig(
        vocab_size=TRAINABLE_VOCAB_SIZE,
        character_coverage=0.9995,
        model_prefix="tokenizer",
    )

    return SentencePieceTrainerAdapter(config).train(multilingual_sentences, directory)


@pytest.fixture
def sentencepiece_tokenizer(trained_model_path: Path) -> SentencePieceTokenizer:
    return SentencePieceTokenizer(trained_model_path)


@pytest.fixture
def word_tokenizer(distinct_sentences: list[str]) -> WordTokenizer:
    return WordTokenizer(pretokenizer={"type": "script"}).train(distinct_sentences)


class TestRegistry:
    def test_both_implementations_are_registered(self) -> None:
        assert set(TOKENIZERS.keys()) == {"sentencepiece", "word"}

    def test_registry_creates_usable_instances(self) -> None:
        assert isinstance(TOKENIZERS.create("word"), Tokenizer)

        assert isinstance(TOKENIZERS.create("sentencepiece"), Tokenizer)


class TestSentencePieceTokenizer:
    def test_encode_returns_ids_and_pieces(
        self, sentencepiece_tokenizer: SentencePieceTokenizer
    ) -> None:
        encoding = sentencepiece_tokenizer.encode("hello world this is a small test sentence")

        assert isinstance(encoding, Encoding)

        assert encoding.length > 0

        assert len(encoding.ids) == len(encoding.tokens)

        assert encoding.attention_mask == [1] * encoding.length

    def test_ids_are_within_the_vocabulary(
        self, sentencepiece_tokenizer: SentencePieceTokenizer
    ) -> None:
        encoding = sentencepiece_tokenizer.encode("नमस्ते दुनिया")

        assert all(0 <= i < sentencepiece_tokenizer.vocabulary_size for i in encoding.ids)

    def test_vocabulary_size_matches_the_trained_model(
        self, sentencepiece_tokenizer: SentencePieceTokenizer
    ) -> None:
        assert sentencepiece_tokenizer.vocabulary_size == TRAINABLE_VOCAB_SIZE

    def test_tokenize_matches_encode_tokens(
        self, sentencepiece_tokenizer: SentencePieceTokenizer
    ) -> None:
        text = "机器学习模型需要大量的数据"

        assert sentencepiece_tokenizer.tokenize(text) == sentencepiece_tokenizer.encode(text).tokens

    def test_round_trip_across_every_script(
        self,
        sentencepiece_tokenizer: SentencePieceTokenizer,
        distinct_sentences: list[str],
    ) -> None:
        for sentence in distinct_sentences:
            encoding = sentencepiece_tokenizer.encode(sentence)

            assert sentencepiece_tokenizer.decode(encoding.ids) == sentence

    def test_encode_all(
        self,
        sentencepiece_tokenizer: SentencePieceTokenizer,
        distinct_sentences: list[str],
    ) -> None:
        encodings = sentencepiece_tokenizer.encode_all(distinct_sentences)

        assert len(encodings) == len(distinct_sentences)

    def test_to_vocabulary_reproduces_the_model_id_space(
        self, sentencepiece_tokenizer: SentencePieceTokenizer
    ) -> None:
        vocabulary = sentencepiece_tokenizer.to_vocabulary()

        assert len(vocabulary) == sentencepiece_tokenizer.vocabulary_size

        # Every id the tokenizer emits must resolve to the same piece
        # through the vocabulary, or the embedding matrix is misindexed.
        for sentence in ("hello world", "नमस्ते दुनिया", "これは日本語のテキストです"):
            encoding = sentencepiece_tokenizer.encode(sentence)

            assert [vocabulary.token_of(i) for i in encoding.ids] == encoding.tokens

    def test_to_vocabulary_places_special_tokens_at_the_reserved_ids(
        self, sentencepiece_tokenizer: SentencePieceTokenizer
    ) -> None:
        vocabulary = sentencepiece_tokenizer.to_vocabulary()

        assert vocabulary.token_of(PAD_ID) == "<pad>"

        assert vocabulary.token_of(UNK_ID) == "<unk>"

        assert vocabulary.token_of(BOS_ID) == "<bos>"

        assert vocabulary.token_of(EOS_ID) == "<eos>"

    def test_to_vocabulary_is_frozen(self, sentencepiece_tokenizer: SentencePieceTokenizer) -> None:
        assert sentencepiece_tokenizer.to_vocabulary().is_frozen

    def test_unfitted_tokenizer_raises_not_fitted(self) -> None:
        tokenizer = SentencePieceTokenizer()

        assert not tokenizer.is_fitted

        with pytest.raises(NotFittedError):
            tokenizer.encode("hello")

        with pytest.raises(NotFittedError):
            tokenizer.decode([1, 2])

        with pytest.raises(NotFittedError):
            tokenizer.tokenize("hello")

        with pytest.raises(NotFittedError):
            tokenizer.to_vocabulary()

        with pytest.raises(NotFittedError):
            _ = tokenizer.vocabulary_size

    def test_saving_an_unfitted_tokenizer_raises(self, tmp_path: Path) -> None:
        with pytest.raises(NotFittedError):
            SentencePieceTokenizer().save(tmp_path)

    def test_missing_model_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ResourceNotFoundError):
            SentencePieceTokenizer(tmp_path / "absent.model")

    def test_save_then_load_round_trips(
        self,
        sentencepiece_tokenizer: SentencePieceTokenizer,
        tmp_path: Path,
        distinct_sentences: list[str],
    ) -> None:
        directory = sentencepiece_tokenizer.save(tmp_path / "saved")

        reloaded = SentencePieceTokenizer.load(directory)

        assert reloaded.vocabulary_size == sentencepiece_tokenizer.vocabulary_size

        for sentence in distinct_sentences:
            assert reloaded.encode(sentence).ids == sentencepiece_tokenizer.encode(sentence).ids

            assert reloaded.decode(reloaded.encode(sentence).ids) == sentence

    def test_load_finds_a_model_under_a_custom_prefix(
        self, sentencepiece_tokenizer: SentencePieceTokenizer, tmp_path: Path
    ) -> None:
        import shutil

        assert sentencepiece_tokenizer.model_path is not None

        shutil.copyfile(sentencepiece_tokenizer.model_path, tmp_path / "custom-prefix.model")

        assert SentencePieceTokenizer.load(tmp_path).vocabulary_size == TRAINABLE_VOCAB_SIZE

    def test_load_from_a_directory_without_a_model_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ResourceNotFoundError):
            SentencePieceTokenizer.load(tmp_path)

    def test_load_from_a_missing_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ResourceNotFoundError):
            SentencePieceTokenizer.load(tmp_path / "absent")


class TestSentencePieceNormalization:
    """
    Configured normalizers must apply at training *and* encode time.

    Applying them at training only would be worse than not applying them
    at all: the model would learn pieces of normalized text while encode
    fed it raw text, and the mismatch would surface only as quietly
    degraded results.
    """

    NORMALIZERS: ClassVar[list[dict[str, str]]] = [
        {"type": "nfkc"},
        {"type": "lowercase"},
        {"type": "whitespace"},
    ]

    # Class-scoped fixtures are classmethods deliberately. pytest builds a
    # fresh instance per test, so `self` inside a class-scoped fixture is
    # not the instance any test receives — binding to the class says what
    # is actually true about the lifetime, and pytest has signalled it
    # intends to stop accepting the instance form.
    @pytest.fixture(scope="class")
    @classmethod
    def mixed_case_corpus(cls) -> list[str]:
        """A corpus whose Latin sentences are shouted."""

        return [
            sentence.upper() if sentence.isascii() else sentence
            for _ in range(120)
            for sentence in (
                "the quick brown fox jumps over the lazy dog",
                "hello world this is a small test sentence",
                "machine learning models need plenty of training data",
                "नमस्ते दुनिया यह एक परीक्षण वाक्य है",
                "これは日本語のテキストです",
            )
        ]

    @pytest.fixture(scope="class")
    @classmethod
    def normalizing_model_path(
        cls,
        tmp_path_factory: pytest.TempPathFactory,
        mixed_case_corpus: list[str],
    ) -> Path:
        # Deliberately small: this corpus is five distinct sentences, so
        # there are few pieces to find, and the test only needs a model
        # that encodes rather than a realistic vocabulary.
        config = TokenizerConfig(
            vocab_size=85,
            character_coverage=0.9995,
            normalizers=cls.NORMALIZERS,
        )

        directory = tmp_path_factory.mktemp("normalizing-model")

        return SentencePieceTrainerAdapter(config).train(mixed_case_corpus, directory)

    @pytest.fixture
    def normalizing_tokenizer(self, normalizing_model_path: Path) -> SentencePieceTokenizer:
        return SentencePieceTokenizer(normalizing_model_path, normalizers=self.NORMALIZERS)

    def test_training_corpus_was_normalized(self, normalizing_model_path: Path) -> None:
        """No shouted piece may survive: the trainer saw folded text."""

        tokenizer = SentencePieceTokenizer(normalizing_model_path)

        pieces = [
            tokenizer._require_processor().IdToPiece(index)
            for index in range(tokenizer.vocabulary_size)
        ]

        assert [piece for piece in pieces if piece != piece.casefold()] == []

    def test_mixed_case_query_encodes_as_its_folded_form(
        self, normalizing_tokenizer: SentencePieceTokenizer
    ) -> None:
        assert (
            normalizing_tokenizer.encode("THE Quick Brown FOX").ids
            == normalizing_tokenizer.encode("the quick brown fox").ids
        )

    def test_tokenize_normalizes_too(self, normalizing_tokenizer: SentencePieceTokenizer) -> None:
        """Embedding training goes through tokenize, not encode."""

        assert normalizing_tokenizer.tokenize("HELLO WORLD") == normalizing_tokenizer.tokenize(
            "hello world"
        )

    def test_an_unnormalizing_tokenizer_over_the_same_model_differs(
        self, normalizing_model_path: Path, normalizing_tokenizer: SentencePieceTokenizer
    ) -> None:
        """
        Guards the test above against passing for the wrong reason.

        If SentencePiece happened to fold case itself, the parity
        assertions would hold with no normalizer wired in at all.
        """

        bare = SentencePieceTokenizer(normalizing_model_path)

        assert (
            bare.encode("THE Quick Brown FOX").ids
            != normalizing_tokenizer.encode("THE Quick Brown FOX").ids
        )

    def test_reloaded_tokenizer_normalizes_identically(
        self, normalizing_tokenizer: SentencePieceTokenizer, tmp_path: Path
    ) -> None:
        """The chain must be persisted, or search would drift from training."""

        reloaded = SentencePieceTokenizer.load(normalizing_tokenizer.save(tmp_path / "saved"))

        assert [type(step) for step in reloaded.normalizer.normalizers] == [
            type(step) for step in normalizing_tokenizer.normalizer.normalizers
        ]

        for text in ("THE Quick Brown FOX", "Hello World", "नमस्ते दुनिया"):
            assert reloaded.encode(text).ids == normalizing_tokenizer.encode(text).ids

    def test_a_model_directory_without_a_sidecar_loads_unnormalized(
        self, normalizing_model_path: Path, tmp_path: Path
    ) -> None:
        """A bare model was trained without a chain; do not invent one."""

        import shutil

        shutil.copyfile(normalizing_model_path, tmp_path / "tokenizer.model")

        assert len(SentencePieceTokenizer.load(tmp_path).normalizer) == 0

    def test_default_construction_normalizes_nothing(self, normalizing_model_path: Path) -> None:
        assert len(SentencePieceTokenizer(normalizing_model_path).normalizer) == 0


class TestWordTokenizer:
    def test_train_builds_a_vocabulary(self, distinct_sentences: list[str]) -> None:
        tokenizer = WordTokenizer().train(distinct_sentences)

        assert tokenizer.is_fitted

        assert tokenizer.vocabulary_size > 4

    def test_round_trip_for_whitespace_delimited_scripts(self) -> None:
        sentences = ["hello world", "नमस्ते दुनिया", "mercado libre"]

        tokenizer = WordTokenizer(pretokenizer={"type": "whitespace"}).train(sentences)

        for sentence in sentences:
            assert tokenizer.decode(tokenizer.encode(sentence).ids) == sentence

    def test_round_trip_is_modulo_normalization(self) -> None:
        tokenizer = WordTokenizer(
            normalizers=[{"type": "nfkc"}, {"type": "lowercase"}, {"type": "whitespace"}],
        ).train(["hello world"])

        assert tokenizer.decode(tokenizer.encode("  HELLO   World  ").ids) == "hello world"

    def test_unknown_tokens_map_to_the_unknown_id(self) -> None:
        tokenizer = WordTokenizer().train(["hello world"])

        encoding = tokenizer.encode("hello unseen")

        assert encoding.ids[0] != UNK_ID

        assert encoding.ids[1] == UNK_ID

    def test_encode_returns_spans_into_the_normalized_text(self) -> None:
        tokenizer = WordTokenizer(
            normalizers=[{"type": "whitespace"}],
            pretokenizer={"type": "whitespace"},
        ).train(["hello world"])

        encoding = tokenizer.encode("  hello   world ")

        assert encoding.spans is not None

        normalized = tokenizer.normalizer.normalize("  hello   world ")

        for token, span in zip(encoding.tokens, encoding.spans, strict=True):
            assert token == normalized[span.start : span.end]

    def test_script_pretokenizer_gives_character_tokens_for_japanese(self) -> None:
        tokenizer = WordTokenizer(pretokenizer={"type": "script"}).train(
            ["これは日本語のテキストです"]
        )

        assert tokenizer.tokenize("これは日本語") == list("これは日本語")

    def test_pretokenizer_and_normalizer_are_exposed(self, word_tokenizer: WordTokenizer) -> None:
        from multilingual_embedding.tokenizer.normalizer import NormalizerPipeline
        from multilingual_embedding.tokenizer.pretokenizer import ScriptAwarePreTokenizer

        assert isinstance(word_tokenizer.normalizer, NormalizerPipeline)

        assert isinstance(word_tokenizer.pretokenizer, ScriptAwarePreTokenizer)

    def test_min_count_filters_rare_tokens(self) -> None:
        tokenizer = WordTokenizer(min_count=2).train(["common common", "rare"])

        assert tokenizer.vocabulary.contains("common")

        assert not tokenizer.vocabulary.contains("rare")

    def test_max_size_caps_the_vocabulary(self, distinct_sentences: list[str]) -> None:
        tokenizer = WordTokenizer(max_size=10).train(distinct_sentences)

        assert tokenizer.vocabulary_size == 10

    def test_tokenize_works_before_training(self) -> None:
        # Segmentation does not depend on the vocabulary, so it is
        # available before fitting; only id lookup requires training.
        assert WordTokenizer().tokenize("hello world") == ["hello", "world"]

    def test_unfitted_tokenizer_raises_not_fitted(self, tmp_path: Path) -> None:
        tokenizer = WordTokenizer()

        assert not tokenizer.is_fitted

        with pytest.raises(NotFittedError):
            tokenizer.encode("hello")

        with pytest.raises(NotFittedError):
            tokenizer.decode([5])

        with pytest.raises(NotFittedError):
            _ = tokenizer.vocabulary_size

        with pytest.raises(NotFittedError):
            tokenizer.save(tmp_path / "unwritten")

        assert not (tmp_path / "unwritten").exists()

    def test_save_then_load_round_trips(
        self, tmp_path: Path, distinct_sentences: list[str]
    ) -> None:
        tokenizer = WordTokenizer(
            normalizers=[{"type": "nfkc"}, {"type": "lowercase"}],
            pretokenizer={"type": "punctuation"},
            min_count=1,
            max_size=200,
        ).train(distinct_sentences)

        directory = tokenizer.save(tmp_path / "word")

        reloaded = WordTokenizer.load(directory)

        assert reloaded.vocabulary_size == tokenizer.vocabulary_size

        for sentence in distinct_sentences:
            assert reloaded.encode(sentence).ids == tokenizer.encode(sentence).ids

            assert reloaded.decode(reloaded.encode(sentence).ids) == tokenizer.decode(
                tokenizer.encode(sentence).ids
            )

    def test_load_accepts_bare_string_specifications(self, tmp_path: Path) -> None:
        tokenizer = WordTokenizer(normalizers=["nfkc"], pretokenizer="whitespace").train(
            ["hello world"]
        )

        reloaded = WordTokenizer.load(tokenizer.save(tmp_path))

        assert reloaded.tokenize("hello world") == ["hello", "world"]

    def test_load_from_an_incomplete_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ResourceNotFoundError):
            WordTokenizer.load(tmp_path)

    def test_repr_reports_vocabulary_size(self, word_tokenizer: WordTokenizer) -> None:
        assert str(word_tokenizer.vocabulary_size) in repr(word_tokenizer)


class TestEncodeLengthInvariantIsStructural:
    """
    Regression: ``encode`` used to derive ids and pieces from two independent
    SentencePiece calls, which rarely disagreed in length and killed a training
    run mid-flight (reported by the generative-text pillar, 2026-08-20).

    The bug was nasty to attribute because it was non-deterministic: it depended
    on the encode sequence rather than the text, so the offending record almost
    never reproduced in isolation and retrying the same string always worked.
    These tests pin the property that makes it impossible, not the symptom.
    """

    def test_ids_and_tokens_agree_across_scripts_and_edge_cases(
        self, sentencepiece_tokenizer: SentencePieceTokenizer
    ) -> None:
        for text in (
            "The Companies Act, 2013 provides for incorporation.",
            "ಕರ್ನಾಟಕ ಗ್ರಾಮ ಸ್ವರಾಜ್",
            "தமிழ்நாடு கரும்பு மேல்வரி",
            "‍‌ emoji \U0001F1EE\U0001F1F3 ﷽ ",
            "",
            "   ",
            "A" * 600,
        ):
            encoding = sentencepiece_tokenizer.encode(text)
            assert len(encoding.ids) == len(encoding.tokens)
            assert len(encoding.attention_mask) == len(encoding.ids)

    def test_ids_are_derived_from_the_pieces_not_a_second_segmentation(
        self, sentencepiece_tokenizer: SentencePieceTokenizer
    ) -> None:
        """
        The guarantee is structural: every id must be exactly the id OF the
        piece at the same index. A second independent segmentation could pass a
        length check by luck; it cannot pass this.
        """
        processor = sentencepiece_tokenizer._require_processor()

        for text in ("Incorporation of companies", "ಕರ್ನಾಟಕ ಗ್ರಾಮ ಸ್ವರಾಜ್", "A" * 200):
            encoding = sentencepiece_tokenizer.encode(text)
            assert encoding.ids == [processor.PieceToId(p) for p in encoding.tokens]

    def test_survives_a_processor_whose_two_calls_DISAGREE(
        self, sentencepiece_tokenizer: SentencePieceTokenizer
    ) -> None:
        """
        The test that actually discriminates.

        On a healthy model the two SentencePiece calls agree, so a test written
        against real text passes with the OLD two-call code too — it pins a
        property that holds either way and proves nothing. (Confirmed by
        mutation: reverting the fix left such a test green.)

        The defect only appears when the two calls disagree, which is rare and
        non-deterministic and cannot be reproduced on demand. So force it: a
        processor whose EncodeAsIds returns one MORE element than
        EncodeAsPieces is exactly the reported failure, deterministically.

        The fixed encode never calls EncodeAsIds, so it is immune by
        construction. The old form would raise here.
        """

        processor = sentencepiece_tokenizer._require_processor()
        real_pieces = processor.EncodeAsPieces
        real_piece_to_id = processor.PieceToId

        class DivergentProcessor:
            def EncodeAsPieces(self, text: str) -> list[str]:
                return list(real_pieces(text))

            def EncodeAsIds(self, text: str) -> list[int]:
                # one element longer — the reported 506 vs 507 signature
                return [0] * (len(real_pieces(text)) + 1)

            def PieceToId(self, piece: str) -> int:
                return real_piece_to_id(piece)

        sentencepiece_tokenizer._processor = DivergentProcessor()

        encoding = sentencepiece_tokenizer.encode("The Companies Act, 2013")

        assert len(encoding.ids) == len(encoding.tokens), (
            "encode must not depend on EncodeAsIds agreeing with EncodeAsPieces"
        )

    def test_repeated_encoding_is_stable(
        self, sentencepiece_tokenizer: SentencePieceTokenizer
    ) -> None:
        """The original failure was non-deterministic, so encode many times."""
        text = "The Karnataka Gram Swaraj and Panchayat Raj Act, 1993"
        first = sentencepiece_tokenizer.encode(text)
        for _ in range(200):
            again = sentencepiece_tokenizer.encode(text)
            assert again.ids == first.ids
            assert again.tokens == first.tokens
