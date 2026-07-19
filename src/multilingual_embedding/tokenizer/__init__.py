"""
Tokenizer layer: text in, model input ids out.

The pipeline is four stages, each independently configurable and each
registered by name so a YAML file can select it::

    text
      -> NormalizerPipeline      unify surface forms
      -> PreTokenizer            propose boundaries, with spans
      -> Tokenizer               map to vocabulary ids
      -> Encoding                ids, pieces, spans, attention mask

:class:`SentencePieceTrainerAdapter` produces the subword model that
:class:`SentencePieceTokenizer` consumes; :class:`WordTokenizer` is the
dependency-free alternative that trains its own vocabulary in a pass.
"""

from __future__ import annotations

from .encoding import Encoding
from .normalizer import (
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
from .pretokenizer import (
    PRETOKENIZERS,
    CharacterPreTokenizer,
    PreTokenizer,
    PunctuationPreTokenizer,
    ScriptAwarePreTokenizer,
    WhitespacePreTokenizer,
)
from .tokenizer import (
    TOKENIZERS,
    SentencePieceTokenizer,
    Tokenizer,
    WordTokenizer,
)
from .trainer import SentencePieceTrainerAdapter

__all__ = [
    "NORMALIZERS",
    "PRETOKENIZERS",
    "TOKENIZERS",
    "CharacterPreTokenizer",
    "DigitNormalizer",
    "Encoding",
    "LowercaseNormalizer",
    "NFCNormalizer",
    "NFDNormalizer",
    "NFKCNormalizer",
    "NFKDNormalizer",
    "Normalizer",
    "NormalizerPipeline",
    "PreTokenizer",
    "PunctuationPreTokenizer",
    "ScriptAwarePreTokenizer",
    "SentencePieceTokenizer",
    "SentencePieceTrainerAdapter",
    "StripAccentsNormalizer",
    "Tokenizer",
    "WhitespaceNormalizer",
    "WhitespacePreTokenizer",
    "WordTokenizer",
]
