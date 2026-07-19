"""
Framework-wide immutable constants.

Configuration values belong in the config package.
Only true constants belong here.
"""

from __future__ import annotations

__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_CHARACTER_COVERAGE",
    "DEFAULT_ENCODING",
    "DEFAULT_RANDOM_SEED",
    "DEFAULT_VOCAB_SIZE",
]

DEFAULT_ENCODING: str = "utf-8"

DEFAULT_RANDOM_SEED: int = 42

DEFAULT_BATCH_SIZE: int = 32

DEFAULT_VOCAB_SIZE: int = 32_000

DEFAULT_CHARACTER_COVERAGE: float = 0.9995
