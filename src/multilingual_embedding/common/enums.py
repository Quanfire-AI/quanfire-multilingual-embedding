"""
Shared enumerations used throughout the framework.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = [
    "SpecialToken",
    "TokenizerModel",
]


class TokenizerModel(StrEnum):
    """Supported tokenizer algorithms."""

    UNIGRAM = "unigram"
    BPE = "bpe"
    WORD = "word"
    CHAR = "char"


class SpecialToken(StrEnum):
    """Reserved special tokens."""

    PAD = "<pad>"
    UNK = "<unk>"
    BOS = "<bos>"
    EOS = "<eos>"
