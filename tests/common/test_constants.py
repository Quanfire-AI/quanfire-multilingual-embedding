from multilingual_embedding.common.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHARACTER_COVERAGE,
    DEFAULT_ENCODING,
    DEFAULT_RANDOM_SEED,
    DEFAULT_VOCAB_SIZE,
)


def test_default_encoding() -> None:
    assert DEFAULT_ENCODING == "utf-8"


def test_vocab_size() -> None:
    assert DEFAULT_VOCAB_SIZE >= 1_000


def test_batch_size() -> None:
    assert DEFAULT_BATCH_SIZE > 0


def test_random_seed() -> None:
    assert DEFAULT_RANDOM_SEED == 42


def test_character_coverage() -> None:
    assert 0.0 < DEFAULT_CHARACTER_COVERAGE <= 1.0
