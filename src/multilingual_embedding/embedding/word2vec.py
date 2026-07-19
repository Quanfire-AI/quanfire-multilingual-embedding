"""
Skip-gram word2vec with negative sampling, in pure numpy.

Predicting each context word from the centre word is the skip-gram
objective. Doing that with a softmax over the whole vocabulary costs one
matmul against every row per token, which is why Mikolov et al. replaced
it with negative sampling: score the true context word against a handful
of randomly drawn impostors and the update touches ``1 + negatives`` rows
instead of all of them.

There is no autograd framework here, deliberately. The gradient of a
sigmoid on a dot product is ``(label - prediction) * other_vector``, and
writing it out is both faster than a graph-building library for a model
this small and far easier to reason about when the vectors come out
wrong.

Three details separate an implementation that learns from one that only
appears to:

- the noise distribution is unigram frequency to the power 0.75,
- frequent tokens are subsampled away,
- the window is redrawn per centre token.

Each is documented at its call site below.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np

from multilingual_embedding.config.base import EmbeddingConfig
from multilingual_embedding.core.exceptions import NotFittedError, ValidationError
from multilingual_embedding.core.logging import get_logger
from multilingual_embedding.utils.filesystem import ensure_directory, require_directory
from multilingual_embedding.utils.io import read_json, write_json
from multilingual_embedding.utils.serialization import from_primitive, to_primitive
from multilingual_embedding.vocabulary.builder import VocabularyBuilder
from multilingual_embedding.vocabulary.vocabulary import Vocabulary

from .base import EmbeddingModel
from .matrix import EmbeddingMatrix

__all__ = ["Word2Vec"]

_logger = get_logger(__name__)

_CONFIG_FILENAME = "word2vec.json"

# Beyond this the sigmoid is 1.0 or 0.0 to float precision, so clamping
# costs no accuracy but avoids overflow warnings on extreme dot products.
_SIGMOID_CLIP = 6.0

# Dampening exponent for the noise distribution. Raw unigram frequencies
# would make the negatives almost entirely function words, which teaches
# the model little; 0.75 lifts rare tokens without flattening the
# distribution to uniform, and is the value from the original paper.
_NOISE_EXPONENT = 0.75

_LOSS_EPSILON = 1e-10


class Word2Vec(EmbeddingModel):
    """
    Skip-gram negative sampling trainer.

    Parameters
    ----------
    config:
        Hyperparameters. Defaults to :class:`EmbeddingConfig` defaults.

    vocabulary:
        Prebuilt vocabulary. When omitted, one is built from the training
        sentences honouring ``config.min_count``.

    Raises
    ------
    NotFittedError
        From :attr:`matrix` and :meth:`most_similar` before training.

    Example
    -------
    ::

        model = Word2Vec(EmbeddingConfig(dimension=64, epochs=5))

        matrix = model.train(sentence_stream)

        model.most_similar("river", top_k=5)
    """

    __slots__ = (
        "_config",
        "_input_weights",
        "_keep_probabilities",
        "_matrix",
        "_noise_cumulative",
        "_output_weights",
        "_rng",
        "_vocabulary",
    )

    def __init__(
        self,
        config: EmbeddingConfig | None = None,
        vocabulary: Vocabulary | None = None,
    ) -> None:
        self._config = config if config is not None else EmbeddingConfig()

        self._vocabulary = vocabulary

        self._matrix: EmbeddingMatrix | None = None

        # `resolved_seed`, not `seed`: the latter may still be the
        # "inherit from the experiment" marker, and default_rng(None)
        # would quietly produce an unreproducible run.
        self._rng = np.random.default_rng(self._config.resolved_seed)

        self._input_weights: np.ndarray | None = None

        self._output_weights: np.ndarray | None = None

        self._noise_cumulative: np.ndarray | None = None

        self._keep_probabilities: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def config(self) -> EmbeddingConfig:
        """The hyperparameters this model was constructed with."""

        return self._config

    @property
    def is_trained(self) -> bool:
        """True once :meth:`train` has produced a matrix."""

        return self._matrix is not None

    @property
    def matrix(self) -> EmbeddingMatrix:
        """
        The trained embedding matrix.

        Raises
        ------
        NotFittedError
            If accessed before :meth:`train`.
        """

        if self._matrix is None:
            raise NotFittedError("Word2Vec has not been trained", model="Word2Vec")

        return self._matrix

    @property
    def vocabulary(self) -> Vocabulary:
        """
        The vocabulary the model trains against.

        Raises
        ------
        NotFittedError
            If no vocabulary was supplied and training has not run.
        """

        if self._vocabulary is None:
            raise NotFittedError("Word2Vec has no vocabulary yet", model="Word2Vec")

        return self._vocabulary

    def most_similar(
        self,
        token_or_vector: str | np.ndarray,
        top_k: int = 10,
        exclude_special: bool = True,
    ) -> list[tuple[str, float]]:
        """
        Nearest neighbours of a token, delegating to the trained matrix.

        Raises
        ------
        NotFittedError
            If accessed before :meth:`train`.
        """

        return self.matrix.most_similar(
            token_or_vector,
            top_k=top_k,
            exclude_special=exclude_special,
        )

    def __repr__(self) -> str:
        return f"Word2Vec(dimension={self._config.dimension}, trained={self.is_trained})"

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        sentences: Iterable[str],
        tokenize: Callable[[str], list[str]] | None = None,
    ) -> EmbeddingMatrix:
        """
        Fit word vectors on a re-iterable sentence stream.

        Parameters
        ----------
        sentences:
            Any re-iterable of sentence strings, typically a
            :class:`~multilingual_embedding.corpus.iterator.SentenceStream`.
            It is traversed once per epoch, plus once more up front when
            the vocabulary has to be built, so a one-shot generator is
            not sufficient.

        tokenize:
            Sentence to token list. Defaults to ``str.split``, which
            assumes the caller has already normalised and pre-tokenised
            the text.

        Returns
        -------
        EmbeddingMatrix
            The input matrix, paired with the vocabulary.

        Raises
        ------
        ValidationError
            If the vocabulary contains no trainable tokens.
        """

        splitter = tokenize if tokenize is not None else _default_tokenize

        if self._vocabulary is None:
            self._vocabulary = self._build_vocabulary(sentences, splitter)

        vocabulary = self._vocabulary

        self._prepare_tables(vocabulary)

        self._initialise_weights(len(vocabulary))

        # Only in-vocabulary occurrences produce updates, so the decay
        # schedule is scaled by those rather than by raw corpus size.
        expected_updates = max(1, vocabulary.total_frequency * self._config.epochs)

        completed = 0

        for epoch in range(self._config.epochs):
            total_loss = 0.0

            pair_count = 0

            for sentence in _progress(sentences, epoch=epoch, epochs=self._config.epochs):
                token_ids = self._select_ids(splitter(sentence), vocabulary)

                if len(token_ids) < 2:
                    completed += len(token_ids)

                    continue

                loss, pairs, completed = self._train_sentence(
                    token_ids,
                    completed=completed,
                    expected_updates=expected_updates,
                )

                total_loss += loss

                pair_count += pairs

            _logger.info(
                "Completed embedding epoch",
                extra={
                    "epoch": epoch + 1,
                    "epochs": self._config.epochs,
                    "loss": total_loss / pair_count if pair_count else 0.0,
                    "pairs": pair_count,
                },
            )

        assert self._input_weights is not None

        # W_out modelled "is this a real context word", not word meaning,
        # and is thrown away; only W_in leaves the trainer.
        self._output_weights = None

        self._matrix = EmbeddingMatrix(vocabulary.freeze(), self._input_weights)

        return self._matrix

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str | Path) -> Path:
        """
        Write the trained matrix and the hyperparameters to a directory.

        Raises
        ------
        NotFittedError
            If the model has not been trained.
        """

        target = ensure_directory(directory)

        self.matrix.save(target)

        write_json(target / _CONFIG_FILENAME, to_primitive(self._config))

        return target

    @classmethod
    def load(cls, directory: str | Path) -> Word2Vec:
        """
        Restore a model written by :meth:`save`.

        The output matrix is not persisted, so a loaded model can be used
        for lookup but not resumed for further training.
        """

        source = require_directory(directory)

        matrix = EmbeddingMatrix.load(source)

        config_path = source / _CONFIG_FILENAME

        config = EmbeddingConfig()

        if config_path.is_file():
            payload: dict[str, Any] = read_json(config_path)

            config = from_primitive(EmbeddingConfig, payload)

        model = cls(config=config, vocabulary=matrix.vocabulary)

        model._matrix = matrix

        return model

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_vocabulary(
        self,
        sentences: Iterable[str],
        splitter: Callable[[str], list[str]],
    ) -> Vocabulary:
        """Count one full pass over the corpus and build the vocabulary."""

        builder = VocabularyBuilder(min_count=self._config.min_count)

        for sentence in sentences:
            builder.add_tokens(splitter(sentence))

        vocabulary = builder.build()

        _logger.info(
            "Built embedding vocabulary",
            extra={"size": len(vocabulary), "min_count": self._config.min_count},
        )

        return vocabulary

    def _prepare_tables(self, vocabulary: Vocabulary) -> None:
        """Precompute the noise distribution and subsampling keep probabilities."""

        frequencies = np.asarray(vocabulary.frequencies(), dtype=np.float64)

        special_count = vocabulary.special_tokens.count

        # Special tokens carry zero frequency and never appear as real
        # context, so they must not be drawn as negatives either.
        weights = np.power(np.maximum(frequencies, 0.0), _NOISE_EXPONENT)

        weights[:special_count] = 0.0

        total_weight = float(weights.sum())

        if total_weight <= 0.0:
            raise ValidationError(
                "Vocabulary contains no trainable tokens",
                size=len(vocabulary),
                min_count=self._config.min_count,
            )

        # A cumulative table plus searchsorted draws a negative in
        # O(log V) from a plain uniform. np.random.choice with p= rebuilds
        # the alias structure on every call and is orders of magnitude
        # slower at the rates this loop needs.
        self._noise_cumulative = np.cumsum(weights / total_weight)

        self._noise_cumulative[-1] = 1.0

        self._keep_probabilities = self._subsample_probabilities(frequencies, vocabulary)

    def _subsample_probabilities(
        self,
        frequencies: np.ndarray,
        vocabulary: Vocabulary,
    ) -> np.ndarray | None:
        """
        Per-id probability of keeping an occurrence.

        Returns ``None`` when subsampling is disabled, which lets the
        training loop skip the draw entirely.
        """

        threshold = self._config.subsample_threshold

        if threshold <= 0.0:
            return None

        total = float(vocabulary.total_frequency)

        if total <= 0.0:
            return None

        relative = frequencies / total

        with np.errstate(divide="ignore", invalid="ignore"):
            # The classic word2vec formula: tokens below the threshold are
            # always kept, and the discard rate grows with frequency, so
            # the very common tokens stop drowning out the informative ones.
            keep = (np.sqrt(relative / threshold) + 1.0) * (threshold / relative)

        keep[~np.isfinite(keep)] = 1.0

        return np.clip(keep, 0.0, 1.0)

    def _select_ids(self, tokens: list[str], vocabulary: Vocabulary) -> list[int]:
        """
        Map tokens to ids, dropping out-of-vocabulary and subsampled ones.

        Tokens below ``min_count`` are discarded rather than folded into
        ``<unk>``: training a single vector on every rare word produces a
        centroid of unrelated meanings that then pollutes its neighbours.
        """

        keep = self._keep_probabilities

        selected: list[int] = []

        for token in tokens:
            token_id = vocabulary.id_of(token)

            if token_id < vocabulary.special_tokens.count:
                continue

            if keep is not None and self._rng.random() > keep[token_id]:
                continue

            selected.append(token_id)

        return selected

    def _initialise_weights(self, size: int) -> None:
        """
        Allocate both weight matrices.

        ``W_in`` is uniform in ``[-0.5/d, 0.5/d]`` and ``W_out`` is zero.
        The small input scale keeps early dot products near zero, where
        the sigmoid gradient is largest; a zeroed output matrix means the
        first updates are driven entirely by the labels rather than by
        noise. Both are the original word2vec choices and convergence is
        noticeably worse without them.
        """

        dimension = self._config.dimension

        span = 0.5 / dimension

        self._input_weights = self._rng.uniform(
            -span,
            span,
            size=(size, dimension),
        ).astype(np.float32)

        self._output_weights = np.zeros((size, dimension), dtype=np.float32)

    def _train_sentence(
        self,
        token_ids: list[int],
        *,
        completed: int,
        expected_updates: int,
    ) -> tuple[float, int, int]:
        """Run every centre/context pair of one sentence; return loss, pairs, progress."""

        assert self._input_weights is not None
        assert self._output_weights is not None
        assert self._noise_cumulative is not None

        window = self._config.window

        negatives = self._config.negative_samples

        length = len(token_ids)

        total_loss = 0.0

        pair_count = 0

        for position, centre in enumerate(token_ids):
            completed += 1

            learning_rate = self._learning_rate(completed, expected_updates)

            # Redrawing the window per centre token means a context word
            # at distance 3 is only sampled when the draw is >= 3, so
            # nearer words are effectively weighted more heavily without
            # any explicit distance weighting term in the gradient.
            effective = int(self._rng.integers(1, window + 1))

            start = max(0, position - effective)

            stop = min(length, position + effective + 1)

            for context_position in range(start, stop):
                if context_position == position:
                    continue

                total_loss += self._update_pair(
                    centre,
                    token_ids[context_position],
                    negatives=negatives,
                    learning_rate=learning_rate,
                )

                pair_count += 1

        return total_loss, pair_count, completed

    def _update_pair(
        self,
        centre: int,
        context: int,
        *,
        negatives: int,
        learning_rate: float,
    ) -> float:
        """Apply one SGNS update and return its loss contribution."""

        input_weights = self._input_weights

        output_weights = self._output_weights

        assert input_weights is not None
        assert output_weights is not None

        if negatives > 0:
            targets = np.empty(negatives + 1, dtype=np.int64)

            targets[0] = context

            targets[1:] = self._sample_negatives(negatives)
        else:
            targets = np.array([context], dtype=np.int64)

        labels = np.zeros(targets.shape[0], dtype=np.float32)

        labels[0] = 1.0

        hidden = input_weights[centre]

        scores = np.clip(output_weights[targets] @ hidden, -_SIGMOID_CLIP, _SIGMOID_CLIP)

        predictions = 1.0 / (1.0 + np.exp(-scores))

        gradient = (labels - predictions) * learning_rate

        # Accumulate the centre-word gradient across the positive and all
        # negatives and apply it once. Updating W_in inside the loop over
        # samples would make later samples see a moved centre vector and
        # silently changes the objective.
        centre_gradient = gradient @ output_weights[targets]

        np.add.at(output_weights, targets, gradient[:, None] * hidden[None, :])

        input_weights[centre] += centre_gradient

        return float(
            -(
                np.log(np.maximum(predictions[0], _LOSS_EPSILON))
                + np.log(np.maximum(1.0 - predictions[1:], _LOSS_EPSILON)).sum()
            )
        )

    def _sample_negatives(self, count: int) -> np.ndarray:
        """Draw ids from the dampened unigram distribution."""

        assert self._noise_cumulative is not None

        return np.searchsorted(self._noise_cumulative, self._rng.random(count))

    def _learning_rate(self, completed: int, expected_updates: int) -> float:
        """
        Linearly decay the rate from ``learning_rate`` to ``min_learning_rate``.

        A corpus longer than the estimate would drive the fraction past
        one, so the result is floored rather than allowed to go negative.
        """

        progress = min(1.0, completed / expected_updates)

        initial = self._config.learning_rate

        floor = self._config.min_learning_rate

        return max(floor, initial - (initial - floor) * progress)


def _default_tokenize(sentence: str) -> list[str]:
    """Split on whitespace; the default when no tokenizer is supplied."""

    return sentence.split()


def _progress(sentences: Iterable[str], *, epoch: int, epochs: int) -> Iterable[str]:
    """
    Wrap the stream in a progress bar when attached to a terminal.

    Disabled otherwise so that log files and CI output do not fill with
    carriage returns, and so a missing tqdm never breaks training.
    """

    try:
        from tqdm import tqdm  # type: ignore[import-untyped]
    except ImportError:  # pragma: no cover - tqdm is a declared dependency
        return sentences

    interactive = bool(getattr(sys.stderr, "isatty", lambda: False)())

    # tqdm ships no stubs, so the wrapper is pinned to the contract this
    # function promises rather than leaking Any into the training loop.
    wrapped: Iterable[str] = tqdm(
        sentences,
        desc=f"epoch {epoch + 1}/{epochs}",
        disable=not interactive,
        leave=False,
    )

    return wrapped
