"""
Shared type aliases used across the framework.
"""

from __future__ import annotations

from typing import NewType

__all__ = [
    "CorpusText",
    "DocumentText",
    "ParagraphText",
    "SentenceText",
    "TokenId",
    "TokenIds",
]

# -------------------------
# Token Types
# -------------------------

TokenId = NewType("TokenId", int)

type TokenIds = list[TokenId]

# -------------------------
# Text Hierarchy
# -------------------------

type SentenceText = str

type ParagraphText = list[SentenceText]

type DocumentText = list[ParagraphText]

type CorpusText = list[DocumentText]
