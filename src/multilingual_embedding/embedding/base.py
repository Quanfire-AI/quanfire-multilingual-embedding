"""
Contract shared by every embedding model.

Word2Vec is the first implementation; other static models and
contextual encoders are expected to follow. Pinning the four operations that a
training pipeline actually needs — train, expose a matrix, save, load —
means the pipeline can swap models without knowing which one it holds.

The ABC is deliberately thin. Anything a single model needs but the
others do not (negative sampling tables, subword buckets) belongs on that
model, not here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .matrix import EmbeddingMatrix

__all__ = ["EmbeddingModel"]


class EmbeddingModel(ABC):
    """
    Abstract base for trainable word embedding models.

    Implementations must raise
    :class:`~multilingual_embedding.core.exceptions.NotFittedError` from
    :attr:`matrix` when accessed before :meth:`train`.
    """

    __slots__ = ()

    @abstractmethod
    def train(
        self,
        sentences: Iterable[str],
        tokenize: Callable[[str], list[str]] | None = None,
    ) -> EmbeddingMatrix:
        """Fit the model and return the learned matrix."""

    @property
    @abstractmethod
    def matrix(self) -> EmbeddingMatrix:
        """The trained matrix."""

    @abstractmethod
    def save(self, directory: str | Path) -> Path:
        """Persist the trained model to a directory."""

    @classmethod
    @abstractmethod
    def load(cls, directory: str | Path) -> EmbeddingModel:
        """Restore a model written by :meth:`save`."""
