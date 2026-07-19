"""Tests for the SentencePiece trainer adapter."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest
import sentencepiece

from multilingual_embedding.common.enums import TokenizerModel
from multilingual_embedding.config.base import TokenizerConfig
from multilingual_embedding.core.exceptions import ConfigurationError, ValidationError
from multilingual_embedding.tokenizer.trainer import SentencePieceTrainerAdapter
from multilingual_embedding.vocabulary.special_tokens import BOS_ID, EOS_ID, PAD_ID, UNK_ID

TRAINABLE_VOCAB_SIZE = 150


class _RecordCollector(logging.Handler):
    """Handler that keeps every record it is given."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)

        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def trainer_records() -> Iterator[list[logging.LogRecord]]:
    """
    Warnings emitted by the trainer module.

    A dedicated handler rather than ``caplog``: ``configure_logging``
    sets ``propagate = False`` on the framework logger, so whether
    caplog's root handler sees a framework record depends on which
    earlier test last configured logging.
    """

    logger = logging.getLogger("multilingual_embedding.tokenizer.trainer")

    collector = _RecordCollector()

    previous_level = logger.level

    logger.setLevel(logging.WARNING)

    logger.addHandler(collector)

    try:
        yield collector.records
    finally:
        logger.removeHandler(collector)

        logger.setLevel(previous_level)


def make_config(**overrides: object) -> TokenizerConfig:
    settings: dict[str, object] = {
        "vocab_size": TRAINABLE_VOCAB_SIZE,
        "character_coverage": 0.9995,
        "model_prefix": "tokenizer",
    }

    settings.update(overrides)

    return TokenizerConfig(**settings)  # type: ignore[arg-type]


class TestTraining:
    def test_produces_a_model_and_a_vocab_file(
        self, tmp_path: Path, multilingual_sentences: list[str]
    ) -> None:
        adapter = SentencePieceTrainerAdapter(make_config())

        model_path = adapter.train(multilingual_sentences, tmp_path)

        assert model_path == tmp_path / "tokenizer.model"

        assert model_path.is_file()

        assert (tmp_path / "tokenizer.vocab").is_file()

        assert model_path.stat().st_size > 0

    def test_output_directory_is_created(
        self, tmp_path: Path, multilingual_sentences: list[str]
    ) -> None:
        target = tmp_path / "nested" / "artifacts"

        model_path = SentencePieceTrainerAdapter(make_config()).train(
            multilingual_sentences, target
        )

        assert model_path.is_file()

    def test_model_prefix_is_honoured(
        self, tmp_path: Path, multilingual_sentences: list[str]
    ) -> None:
        adapter = SentencePieceTrainerAdapter(make_config(model_prefix="multilingual"))

        model_path = adapter.train(multilingual_sentences, tmp_path)

        assert model_path.name == "multilingual.model"

    def test_trained_model_is_loadable_and_has_the_requested_size(
        self, tmp_path: Path, multilingual_sentences: list[str]
    ) -> None:
        model_path = SentencePieceTrainerAdapter(make_config()).train(
            multilingual_sentences, tmp_path
        )

        processor = sentencepiece.SentencePieceProcessor()

        processor.Load(str(model_path))

        assert processor.GetPieceSize() == TRAINABLE_VOCAB_SIZE

    def test_special_token_ids_match_the_framework_vocabulary(
        self, tmp_path: Path, multilingual_sentences: list[str]
    ) -> None:
        # If these drift, ids from the tokenizer index the wrong rows of
        # an embedding matrix built against our Vocabulary, and nothing
        # raises: the model simply trains on scrambled inputs.
        model_path = SentencePieceTrainerAdapter(make_config()).train(
            multilingual_sentences, tmp_path
        )

        processor = sentencepiece.SentencePieceProcessor()

        processor.Load(str(model_path))

        assert processor.IdToPiece(PAD_ID) == "<pad>"

        assert processor.IdToPiece(UNK_ID) == "<unk>"

        assert processor.IdToPiece(BOS_ID) == "<bos>"

        assert processor.IdToPiece(EOS_ID) == "<eos>"

    def test_bpe_model_type_is_supported(
        self, tmp_path: Path, multilingual_sentences: list[str]
    ) -> None:
        adapter = SentencePieceTrainerAdapter(make_config(model_type=TokenizerModel.BPE))

        assert adapter.train(multilingual_sentences, tmp_path).is_file()

    def test_config_defaults_to_a_tokenizer_config(self) -> None:
        assert isinstance(SentencePieceTrainerAdapter().config, TokenizerConfig)

    def test_sentences_are_consumed_from_a_generator(
        self, tmp_path: Path, multilingual_sentences: list[str]
    ) -> None:
        def stream() -> Iterator[str]:
            yield from multilingual_sentences

        model_path = SentencePieceTrainerAdapter(make_config()).train(stream(), tmp_path)

        assert model_path.is_file()


class TestFailureModes:
    def test_vocab_size_larger_than_the_corpus_supports_raises_configuration_error(
        self, tmp_path: Path
    ) -> None:
        tiny_corpus = ["hello world", "नमस्ते दुनिया", "こんにちは"] * 5

        adapter = SentencePieceTrainerAdapter(make_config(vocab_size=5_000))

        with pytest.raises(ConfigurationError) as error:
            adapter.train(tiny_corpus, tmp_path)

        assert error.value.context["requested_vocab_size"] == 5_000

        assert "vocab_size" in str(error.value)

        assert not (tmp_path / "tokenizer.model").exists()

    def test_vocab_size_too_small_for_the_character_inventory_is_reported_separately(
        self, tmp_path: Path, multilingual_sentences: list[str]
    ) -> None:
        adapter = SentencePieceTrainerAdapter(make_config(vocab_size=20))

        with pytest.raises(ConfigurationError) as error:
            adapter.train(multilingual_sentences, tmp_path)

        # The opposite problem must not be reported as "too large".
        assert "too small" in str(error.value)

        assert error.value.context["requested_vocab_size"] == 20

    def test_empty_corpus_raises_validation_error(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError):
            SentencePieceTrainerAdapter(make_config()).train([], tmp_path)

    def test_blank_only_corpus_raises_validation_error(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError):
            SentencePieceTrainerAdapter(make_config()).train(["", "   ", "\n"], tmp_path)


class TestNormalizerApplication:
    """
    ``tokenizer.normalizers`` must shape the corpus SentencePiece sees.

    The setting is validated and persisted into ``config.yaml``, so a
    trainer that read it and did nothing would leave a record claiming
    normalization that never happened.
    """

    def test_staging_applies_the_configured_chain(self, tmp_path: Path) -> None:
        corpus_path = tmp_path / "corpus.txt"

        adapter = SentencePieceTrainerAdapter(
            make_config(normalizers=[{"type": "lowercase"}, {"type": "whitespace"}])
        )

        adapter._stage_corpus(["HELLO World", "२०२४ ठीक है"], corpus_path)

        assert corpus_path.read_text(encoding="utf-8").splitlines() == [
            "hello world",
            "२०२४ ठीक है",
        ]

    def test_digit_normalizer_reaches_the_staged_corpus(self, tmp_path: Path) -> None:
        corpus_path = tmp_path / "corpus.txt"

        adapter = SentencePieceTrainerAdapter(make_config(normalizers=[{"type": "digits"}]))

        adapter._stage_corpus(["२०२४"], corpus_path)

        assert corpus_path.read_text(encoding="utf-8").strip() == "2024"

    def test_an_empty_chain_leaves_text_alone(self, tmp_path: Path) -> None:
        corpus_path = tmp_path / "corpus.txt"

        adapter = SentencePieceTrainerAdapter(make_config(normalizers=[]))

        adapter._stage_corpus(["HELLO World"], corpus_path)

        assert corpus_path.read_text(encoding="utf-8").strip() == "HELLO World"

    def test_the_chain_is_exposed_for_the_tokenizer_to_reuse(self) -> None:
        """The pipeline reads this to build a matching tokenizer."""

        adapter = SentencePieceTrainerAdapter(make_config(normalizers=[{"type": "lowercase"}]))

        assert adapter.normalizer.normalize("ABC") == "abc"


class TestPretokenizerIsReportedAsInapplicable:
    """
    SentencePiece cannot honour a pre-tokenizer, and must say so.

    It consumes a raw character stream by design — the property that
    lets one model serve scripts with no whitespace word boundaries — so
    there is nowhere to insert a framework pre-tokenizer. The setting is
    still legitimate for ``WordTokenizer``; the unacceptable outcome
    would be dropping it in silence.
    """

    def test_a_non_default_pretokenizer_warns(
        self, trainer_records: list[logging.LogRecord]
    ) -> None:
        SentencePieceTrainerAdapter(make_config(pretokenizer={"type": "script"}))

        assert any(
            "cannot apply a pre-tokenizer" in record.getMessage() for record in trainer_records
        )

    def test_the_warning_names_the_ignored_setting(
        self, trainer_records: list[logging.LogRecord]
    ) -> None:
        SentencePieceTrainerAdapter(make_config(pretokenizer={"type": "character"}))

        assert any(
            getattr(record, "pretokenizer", None) == "character" for record in trainer_records
        )

    def test_a_bare_string_specification_is_understood(
        self, trainer_records: list[logging.LogRecord]
    ) -> None:
        """Specs may be written as ``"script"`` rather than a mapping."""

        SentencePieceTrainerAdapter(make_config(pretokenizer="script"))

        assert any(getattr(record, "pretokenizer", None) == "script" for record in trainer_records)

    def test_the_default_whitespace_pretokenizer_is_silent(
        self, trainer_records: list[logging.LogRecord]
    ) -> None:
        """Re-joining whitespace tokens on spaces is the identity here."""

        SentencePieceTrainerAdapter(make_config())

        assert trainer_records == []


class TestStaging:
    def test_embedded_newlines_do_not_split_training_examples(self, tmp_path: Path) -> None:
        corpus_path = tmp_path / "corpus.txt"

        adapter = SentencePieceTrainerAdapter(make_config())

        written = adapter._stage_corpus(["first\nsecond", "third"], corpus_path)

        assert written == 2

        assert corpus_path.read_text(encoding="utf-8").splitlines() == [
            "first second",
            "third",
        ]

    def test_blank_sentences_are_skipped(self, tmp_path: Path) -> None:
        corpus_path = tmp_path / "corpus.txt"

        adapter = SentencePieceTrainerAdapter(make_config())

        assert adapter._stage_corpus(["a", "", "  ", "b"], corpus_path) == 2

    def test_staging_directory_does_not_survive_training(
        self, tmp_path: Path, multilingual_sentences: list[str]
    ) -> None:
        SentencePieceTrainerAdapter(make_config()).train(multilingual_sentences, tmp_path)

        assert sorted(path.name for path in tmp_path.iterdir()) == [
            "tokenizer.model",
            "tokenizer.vocab",
        ]
