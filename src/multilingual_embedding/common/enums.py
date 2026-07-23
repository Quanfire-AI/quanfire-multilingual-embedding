"""
Shared enumerations used throughout the framework.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = [
    "DataProvenance",
    "SpecialToken",
    "TokenizerModel",
]


class DataProvenance(StrEnum):
    """
    Where an adapter's *training* data came from.

    Recorded on every saved adapter and required before one can be
    written, because the source of the training text is a legal fact
    about the model, not a note about it. ``notes.trained_on`` already
    records a *path*, and a path says nothing about provenance — the same
    ``data/pairs/hi.jsonl.gz`` could be public judgments, synthetic text,
    or customer data that must never have been there. This is the field
    that cannot be left blank.

    The three values are exhaustive by intent, and they are the same
    vocabulary the data layer is organised around:

    * ``PUBLIC`` — public-domain or openly-licensed corpora whose licence
      permits commercial use (court judgments under CC BY, legislation by
      statute). The default route for this project.
    * ``SYNTHETIC`` — text generated to match the domain's shape, owned
      outright, carrying no third-party licence or privacy obligation.
    * ``LICENSED`` — data used under a specific contractual basis, such as
      a tenant's opt-in. The value that says a human checked there was a
      right to train on this.

    Customer data with no such basis has no value here on purpose: there
    is no vocabulary entry for it because it must not reach training.
    """

    PUBLIC = "public"
    SYNTHETIC = "synthetic"
    LICENSED = "licensed"


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
