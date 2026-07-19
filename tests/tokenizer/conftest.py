"""Shared fixtures for the tokenizer tests."""

from __future__ import annotations

import pytest

# Deliberately multilingual: a corpus of one script would let a
# script-specific bug pass unnoticed. Latin, Devanagari, Japanese
# (kanji + kana), Arabic and Han are all represented.
MULTILINGUAL_SENTENCES: tuple[str, ...] = (
    "the quick brown fox jumps over the lazy dog",
    "hello world this is a small test sentence",
    "machine learning models need plenty of training data",
    "multilingual embeddings share a single vocabulary space",
    "नमस्ते दुनिया यह एक परीक्षण वाक्य है",
    "भाषा मॉडल को बहुत सारे डेटा की आवश्यकता होती है",
    "हिंदी भारत की एक प्रमुख भाषा है",
    "これは日本語のテキストです",
    "機械学習のモデルは大量のデータを必要とします",
    "東京は日本の首都です",
    "مرحبا بالعالم هذه جملة اختبار",
    "تحتاج نماذج اللغة إلى الكثير من البيانات",
    "这是一个中文测试句子",
    "机器学习模型需要大量的数据",
    "北京是中国的首都",
)

# SentencePiece needs real evidence per piece, so the sentences are
# repeated. A few hundred lines is enough for a 150-piece model and
# keeps the test fast.
CORPUS_REPETITIONS = 120


@pytest.fixture(scope="session")
def multilingual_sentences() -> list[str]:
    """A small synthetic corpus spanning five scripts."""

    return [sentence for _ in range(CORPUS_REPETITIONS) for sentence in MULTILINGUAL_SENTENCES]


@pytest.fixture(scope="session")
def distinct_sentences() -> list[str]:
    """The distinct sentences underlying the corpus, in order."""

    return list(MULTILINGUAL_SENTENCES)
