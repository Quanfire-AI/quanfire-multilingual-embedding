"""
Sentence embeddings composed from word vectors.

Word vectors are the hard part; turning a bag of them into one sentence
vector is mostly a question of how you weight them. Two schemes are
provided.

:class:`MeanPoolingEncoder` averages the token vectors. It is the obvious
baseline and it is genuinely hard to beat with anything cheap.

:class:`SifEncoder` implements Smooth Inverse Frequency (Arora et al.,
2017), which down-weights frequent tokens and then subtracts the leading
principal direction of the batch. It is a strong classical baseline that
consistently outperforms plain mean pooling on sentence similarity, and
it costs one SVD.

The principal-component step is what makes SIF a *batch* method: the
direction being removed is estimated from the set of sentences, and there
is no such direction for a single sentence in isolation. An unfitted
encoder therefore returns the weighted average and skips the removal
rather than pretending to a component it cannot estimate.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence

import numpy as np

from multilingual_embedding.core.exceptions import ValidationError
from multilingual_embedding.core.logging import get_logger
from multilingual_embedding.core.registry import Registry

from .matrix import EmbeddingMatrix

__all__ = [
    "SENTENCE_ENCODERS",
    "MeanPoolingEncoder",
    "SentenceEncoder",
    "SifEncoder",
]

_logger = get_logger(__name__)

_DEFAULT_SIF_ALPHA = 1e-3


def _default_tokenize(sentence: str) -> list[str]:
    """Split on whitespace; the default when no tokenizer is supplied."""

    return sentence.split()


class SentenceEncoder(ABC):
    """
    Maps text to a fixed length vector using a word embedding matrix.

    Parameters
    ----------
    matrix:
        Trained word vectors.

    tokenize:
        Sentence to token list. Defaults to ``str.split``.
    """

    __slots__ = ("_matrix", "_tokenize")

    def __init__(
        self,
        matrix: EmbeddingMatrix,
        tokenize: Callable[[str], list[str]] | None = None,
    ) -> None:
        self._matrix = matrix

        self._tokenize = tokenize if tokenize is not None else _default_tokenize

    @property
    def matrix(self) -> EmbeddingMatrix:
        """The word vectors backing this encoder."""

        return self._matrix

    @property
    def dimension(self) -> int:
        """Length of an encoded sentence vector."""

        return self._matrix.dimension

    @abstractmethod
    def encode(self, text: str) -> np.ndarray:
        """Encode one string."""

    @abstractmethod
    def encode_batch(self, texts: Sequence[str]) -> np.ndarray:
        """Encode a sequence of strings into an ``(n, dimension)`` array."""

    def _known_ids(self, text: str) -> list[int]:
        """
        Return in-vocabulary token ids for a string.

        Out-of-vocabulary tokens are dropped rather than mapped to the
        unknown row, whose vector is an artefact of training and would
        drag every sentence containing a rare word toward the same point.
        """

        vocabulary = self._matrix.vocabulary

        return [
            vocabulary.id_of(token) for token in self._tokenize(text) if vocabulary.contains(token)
        ]


class MeanPoolingEncoder(SentenceEncoder):
    """
    Average of the in-vocabulary token vectors.

    Returns a zero vector when nothing in the input is known, which keeps
    the output finite and lets a caller detect the empty case with a norm
    check instead of a NaN test.

    Example
    -------
    ::

        encoder = MeanPoolingEncoder(matrix)

        vector = encoder.encode("the river is wide")
    """

    __slots__ = ()

    def encode(self, text: str) -> np.ndarray:
        """
        Encode one string.

        Returns
        -------
        numpy.ndarray
            Vector of length ``dimension``, all zeros for empty or
            fully out-of-vocabulary input.
        """

        ids = self._known_ids(text)

        if not ids:
            return np.zeros(self.dimension, dtype=np.float32)

        return self._matrix.vectors[ids].mean(axis=0).astype(np.float32)

    def encode_batch(self, texts: Sequence[str]) -> np.ndarray:
        """
        Encode a sequence of strings.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(len(texts), dimension)``.
        """

        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        return np.vstack([self.encode(text) for text in texts]).astype(np.float32)


class SifEncoder(SentenceEncoder):
    """
    Smooth Inverse Frequency weighted average with common-component removal.

    Each token is weighted by ``alpha / (alpha + p(token))``, where ``p``
    is the token's corpus probability. Frequent tokens therefore
    contribute little without a hand-maintained stopword list. The
    weighted averages then have their first principal component removed,
    which strips the direction shared by all sentences — largely syntax
    and frequency effects rather than content.

    Parameters
    ----------
    matrix:
        Trained word vectors.

    tokenize:
        Sentence to token list. Defaults to ``str.split``.

    alpha:
        Smoothing constant. The 1e-3 default is the value from the paper
        and is insensitive over roughly 1e-4 to 1e-3.

    Raises
    ------
    ValidationError
        If ``alpha`` is not positive.

    Example
    -------
    ::

        encoder = SifEncoder(matrix)

        vectors = encoder.encode_batch(sentences)   # fits the component

        query = encoder.encode("a new sentence")    # reuses it
    """

    __slots__ = ("_alpha", "_component", "_weights")

    def __init__(
        self,
        matrix: EmbeddingMatrix,
        tokenize: Callable[[str], list[str]] | None = None,
        alpha: float = _DEFAULT_SIF_ALPHA,
    ) -> None:
        super().__init__(matrix, tokenize)

        if alpha <= 0.0:
            raise ValidationError("alpha must be > 0", name="alpha", value=alpha)

        self._alpha = alpha

        self._weights = _sif_weights(matrix, alpha)

        self._component: np.ndarray | None = None

    @property
    def alpha(self) -> float:
        """The smoothing constant."""

        return self._alpha

    @property
    def is_fitted(self) -> bool:
        """True once a common component has been estimated from a batch."""

        return self._component is not None

    def encode(self, text: str) -> np.ndarray:
        """
        Encode one string.

        The common component is removed only if the encoder has already
        seen a batch; see the module docstring for why a single sentence
        cannot supply one.

        Returns
        -------
        numpy.ndarray
            Vector of length ``dimension``, all zeros for empty or fully
            out-of-vocabulary input.
        """

        encoded = self._remove_component(self._weighted_average(text).reshape(1, -1))

        row: np.ndarray = encoded[0]

        return row

    def encode_batch(self, texts: Sequence[str]) -> np.ndarray:
        """
        Encode a sequence of strings, fitting the common component from it.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(len(texts), dimension)``.
        """

        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        averages = np.vstack([self._weighted_average(text) for text in texts]).astype(np.float32)

        self._fit_component(averages)

        return self._remove_component(averages)

    def _weighted_average(self, text: str) -> np.ndarray:
        """SIF-weighted mean of the in-vocabulary token vectors."""

        ids = self._known_ids(text)

        if not ids:
            return np.zeros(self.dimension, dtype=np.float32)

        weights = self._weights[ids]

        total = float(weights.sum())

        if total <= 0.0:
            return np.zeros(self.dimension, dtype=np.float32)

        selected: np.ndarray = self._matrix.vectors[ids]

        averaged: np.ndarray = (weights @ selected) / total

        return averaged.astype(np.float32)

    def _fit_component(self, averages: np.ndarray) -> None:
        """Estimate the leading singular direction of a batch of averages."""

        if averages.shape[0] < 2 or not np.any(averages):
            # One sentence is its own principal component; removing it
            # would zero the vector outright.
            return

        _, _, right = np.linalg.svd(averages, full_matrices=False)

        self._component = right[0].astype(np.float32)

        _logger.debug("Fitted SIF common component", extra={"batch": int(averages.shape[0])})

    def _remove_component(self, vectors: np.ndarray) -> np.ndarray:
        """Project out the fitted common component, if there is one."""

        if self._component is None:
            return vectors.astype(np.float32)

        projection = vectors @ self._component

        return (vectors - np.outer(projection, self._component)).astype(np.float32)


def _sif_weights(matrix: EmbeddingMatrix, alpha: float) -> np.ndarray:
    """Per-id SIF weight ``alpha / (alpha + p(token))``."""

    frequencies = np.asarray(matrix.vocabulary.frequencies(), dtype=np.float64)

    total = float(frequencies.sum())

    # Without frequency information every token weighs the same, which
    # degrades SIF to mean pooling rather than to nonsense.
    probabilities = frequencies / total if total > 0.0 else np.zeros_like(frequencies)

    return (alpha / (alpha + probabilities)).astype(np.float32)


SENTENCE_ENCODERS: Registry[SentenceEncoder] = Registry("sentence_encoder")

SENTENCE_ENCODERS.register("mean", MeanPoolingEncoder)

SENTENCE_ENCODERS.register("sif", SifEncoder)
