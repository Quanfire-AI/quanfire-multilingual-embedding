"""
Tokenizers: text to ids and back.

Two implementations cover the two situations that actually arise.

:class:`SentencePieceTokenizer` is the production path. It wraps a
trained SentencePiece model, which handles unseen words by decomposing
them into subwords and needs no language-specific rules — the property
that makes one shared model viable across scripts.

:class:`WordTokenizer` is a dependency-free fallback built from the
framework's own normalizer, pre-tokenizer and vocabulary. It is what
tests run against, and it is the right choice for the languages and
downstream tasks where subword units are unwanted.

Both round-trip: ``decode(encode(text).ids)`` reconstructs the text up
to normalization, which is the strongest guarantee available once
casing or whitespace has been folded away.
"""

from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import sentencepiece

from multilingual_embedding.common.span import Span
from multilingual_embedding.core.exceptions import NotFittedError, ResourceNotFoundError
from multilingual_embedding.core.factory import build_from_config
from multilingual_embedding.core.logging import get_logger
from multilingual_embedding.core.registry import Registry
from multilingual_embedding.utils.filesystem import (
    ensure_directory,
    require_directory,
    require_file,
)
from multilingual_embedding.utils.io import read_json, write_json
from multilingual_embedding.vocabulary.builder import VocabularyBuilder
from multilingual_embedding.vocabulary.special_tokens import SpecialTokenSet
from multilingual_embedding.vocabulary.vocabulary import Vocabulary

from .encoding import Encoding
from .normalizer import NormalizerPipeline
from .pretokenizer import PRETOKENIZERS, PreTokenizer

__all__ = [
    "TOKENIZERS",
    "SentencePieceTokenizer",
    "Tokenizer",
    "WordTokenizer",
]

_logger = get_logger(__name__)

SENTENCEPIECE_MODEL_FILENAME = "tokenizer.model"

# Deliberately not "tokenizer.json": a WordTokenizer saved into the same
# directory writes that name, and the two payloads are not interchangeable.
SENTENCEPIECE_CONFIG_FILENAME = "sentencepiece.json"

WORD_VOCABULARY_FILENAME = "vocabulary.json"

WORD_CONFIG_FILENAME = "tokenizer.json"

_FORMAT_VERSION = 1


class Tokenizer(ABC):
    """
    Base class for tokenizers.

    A tokenizer owns the whole path from a string to model input ids and
    back. Persisting one must capture everything needed to reproduce
    that mapping exactly, since ids that shift between training and
    inference silently index the wrong embedding rows.
    """

    __slots__ = ()

    @abstractmethod
    def encode(self, text: str) -> Encoding:
        """
        Tokenize ``text`` into ids and surface pieces.

        Parameters
        ----------
        text:
            Input string.

        Returns
        -------
        An :class:`~multilingual_embedding.tokenizer.encoding.Encoding`.
        """

    @abstractmethod
    def decode(self, ids: Sequence[int]) -> str:
        """
        Reconstruct text from ids.

        Parameters
        ----------
        ids:
            Vocabulary ids, as produced by :meth:`encode`.

        Returns
        -------
        The reconstructed string, up to normalization.
        """

    @abstractmethod
    def tokenize(self, text: str) -> list[str]:
        """Return only the surface pieces for ``text``."""

    @property
    @abstractmethod
    def vocabulary_size(self) -> int:
        """Number of ids this tokenizer can emit."""

    @abstractmethod
    def save(self, directory: str | Path) -> Path:
        """
        Write everything needed to reload this tokenizer.

        Parameters
        ----------
        directory:
            Destination directory, created if absent.

        Returns
        -------
        The directory written to.
        """

    @classmethod
    @abstractmethod
    def load(cls, directory: str | Path) -> Tokenizer:
        """
        Rebuild a tokenizer from a directory written by :meth:`save`.

        Raises
        ------
        ResourceNotFoundError
            If the directory does not hold the expected artefacts.
        """

    def encode_all(self, texts: Iterable[str]) -> list[Encoding]:
        """Encode an iterable of texts."""

        return [self.encode(text) for text in texts]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


TOKENIZERS: Registry[Tokenizer] = Registry("tokenizer")


@TOKENIZERS.register("sentencepiece")
class SentencePieceTokenizer(Tokenizer):
    """
    Subword tokenizer backed by a trained SentencePiece model.

    Parameters
    ----------
    model_path:
        Path to a ``.model`` file. May be omitted so the tokenizer can
        be constructed from configuration before training has produced
        a model; call :meth:`load_model` afterwards. Every operation
        raises :class:`NotFittedError` until a model is present.

    normalizers:
        Ordered normalizer specifications applied to text before it
        reaches the model. These **must** be the chain the model was
        trained with — see
        :class:`~multilingual_embedding.tokenizer.trainer.SentencePieceTrainerAdapter`,
        which applies them to the training corpus — or encode-time text
        will not match what the pieces were learned from. Defaults to no
        normalization, matching a model trained without any.

    Example
    -------
    ::

        tokenizer = SentencePieceTokenizer("artifacts/tokenizer/tokenizer.model")

        tokenizer.decode(tokenizer.encode("नमस्ते दुनिया").ids)
        -> "नमस्ते दुनिया"
    """

    __slots__ = ("_model_path", "_normalizer", "_normalizer_specs", "_processor")

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        normalizers: Sequence[Mapping[str, Any] | str] | None = None,
    ) -> None:
        self._processor: sentencepiece.SentencePieceProcessor | None = None

        self._model_path: Path | None = None

        self._normalizer_specs: list[Mapping[str, Any] | str] = list(normalizers or ())

        self._normalizer = NormalizerPipeline.from_config(self._normalizer_specs)

        if model_path is not None:
            self.load_model(model_path)

    def load_model(self, model_path: str | Path) -> SentencePieceTokenizer:
        """
        Load a trained ``.model`` file into this tokenizer.

        Parameters
        ----------
        model_path:
            Path to the SentencePiece model file.

        Returns
        -------
        ``self``, so the call can be chained.

        Raises
        ------
        ResourceNotFoundError
            If the file does not exist.
        """

        resolved = require_file(model_path)

        processor = sentencepiece.SentencePieceProcessor()

        processor.Load(str(resolved))

        self._processor = processor

        self._model_path = resolved

        _logger.info(
            "Loaded SentencePiece model",
            extra={"path": str(resolved), "vocab_size": processor.GetPieceSize()},
        )

        return self

    @property
    def is_fitted(self) -> bool:
        """True once a model has been loaded."""

        return self._processor is not None

    @property
    def model_path(self) -> Path | None:
        """Path of the loaded model, if any."""

        return self._model_path

    @property
    def normalizer(self) -> NormalizerPipeline:
        """The normalizer chain applied before the model sees text."""

        return self._normalizer

    @property
    def vocabulary_size(self) -> int:
        """Number of pieces in the loaded model."""

        return self._require_processor().GetPieceSize()

    def encode(self, text: str) -> Encoding:
        """
        Encode ``text`` into subword ids and pieces.

        The configured normalizer chain is applied first, so that the
        text the model sees here is in the same form as the corpus it
        was trained on.

        Spans are not returned: SentencePiece pieces carry a synthetic
        word-boundary marker and do not map onto the input characters
        one for one, so reporting spans would mean fabricating them.

        Raises
        ------
        NotFittedError
            If no model has been loaded.
        """

        processor = self._require_processor()

        normalized = self._normalizer.normalize(text)

        # Segment ONCE and derive the ids from that single segmentation.
        #
        # This used to call EncodeAsIds and EncodeAsPieces separately. Two
        # independent calls are not guaranteed to segment identically, and on
        # rare inputs they returned lists differing by one element, which
        # tripped Encoding's length invariant and killed the whole run. The
        # failure was non-deterministic — it depended on the encode sequence
        # rather than the text, so the offending record almost never reproduced
        # in isolation and a retry on the same string always succeeded. That
        # made it look like flaky data instead of a defect here.
        #
        # Deriving ids from the pieces makes the lengths equal BY CONSTRUCTION
        # rather than by assumption, so the invariant cannot be violated no
        # matter how the underlying model segments. Every piece came from the
        # processor's own vocabulary, so PieceToId is total over them —
        # verified identical to EncodeAsIds across Latin, Indic scripts,
        # zero-width joiners, emoji and long inputs.
        pieces: list[str] = processor.EncodeAsPieces(normalized)

        ids: list[int] = [processor.PieceToId(piece) for piece in pieces]

        return Encoding(ids=ids, tokens=pieces, attention_mask=[1] * len(ids))

    def decode(self, ids: Sequence[int]) -> str:
        """
        Reconstruct text from subword ids.

        Raises
        ------
        NotFittedError
            If no model has been loaded.
        """

        decoded: str = self._require_processor().DecodeIds(list(ids))

        return decoded

    def tokenize(self, text: str) -> list[str]:
        """Normalize ``text`` and return its subword pieces."""

        processor = self._require_processor()

        pieces: list[str] = processor.EncodeAsPieces(self._normalizer.normalize(text))

        return pieces

    def to_vocabulary(self) -> Vocabulary:
        """
        Build a :class:`Vocabulary` mirroring the model's piece table.

        The embedding layer indexes rows by id, so the tokenizer and the
        vocabulary must agree on every id, not merely on the token set.
        Because the model was trained with the framework's special token
        ids pinned (see
        :class:`~multilingual_embedding.tokenizer.trainer.SentencePieceTrainerAdapter`),
        pieces 0-3 already are pad/unk/bos/eos, and the remaining pieces
        can be appended in model order to reproduce the id space exactly.

        Returns
        -------
        A vocabulary whose ids match this model's piece ids.

        Raises
        ------
        NotFittedError
            If no model has been loaded.

        Example
        -------
        ::

            vocabulary = tokenizer.to_vocabulary()

            vocabulary.id_of("▁hello") == processor.PieceToId("▁hello")
        """

        processor = self._require_processor()

        pieces = [processor.IdToPiece(index) for index in range(processor.GetPieceSize())]

        vocabulary = Vocabulary(
            special_tokens=SpecialTokenSet(
                pad=pieces[0],
                unk=pieces[1],
                bos=pieces[2],
                eos=pieces[3],
            )
        )

        for piece in pieces[vocabulary.special_tokens.count :]:
            vocabulary.add(piece, frequency=0)

        return vocabulary.freeze()

    def save(self, directory: str | Path) -> Path:
        """
        Copy the loaded model file into ``directory``.

        The normalizer chain is written alongside it. Without that, a
        reloaded tokenizer would normalize differently from the one that
        produced the embeddings, and the mismatch would surface only as
        degraded results.
        """

        processor_path = self._model_path

        if processor_path is None:
            raise NotFittedError(
                "SentencePieceTokenizer has no model to save",
                directory=str(directory),
            )

        target = ensure_directory(directory)

        destination = target / SENTENCEPIECE_MODEL_FILENAME

        if destination.resolve() != processor_path.resolve():
            shutil.copyfile(processor_path, destination)

        write_json(
            target / SENTENCEPIECE_CONFIG_FILENAME,
            {
                "format_version": _FORMAT_VERSION,
                "type": "sentencepiece",
                "normalizers": [
                    dict(spec) if isinstance(spec, Mapping) else spec
                    for spec in self._normalizer_specs
                ],
            },
        )

        return target

    @classmethod
    def load(cls, directory: str | Path) -> SentencePieceTokenizer:
        """
        Load the tokenizer from a directory written by :meth:`save`.

        Falls back to the single ``.model`` file present when the
        canonical filename is absent, which makes a model trained under
        a custom ``model_prefix`` loadable without renaming it.

        A directory holding only a model file — one produced by the
        trainer before :meth:`save` ran — loads with no normalization,
        which is exactly how such a model was trained.
        """

        root = require_directory(directory)

        candidate = root / SENTENCEPIECE_MODEL_FILENAME

        if not candidate.exists():
            models = sorted(root.glob("*.model"))

            if not models:
                raise ResourceNotFoundError(
                    "No SentencePiece model found in directory",
                    directory=str(root),
                )

            candidate = models[0]

        config_path = root / SENTENCEPIECE_CONFIG_FILENAME

        normalizers = read_json(config_path).get("normalizers") if config_path.exists() else None

        return cls(candidate, normalizers=normalizers)

    def _require_processor(self) -> sentencepiece.SentencePieceProcessor:
        """Return the processor or explain that training has not happened."""

        if self._processor is None:
            raise NotFittedError(
                "SentencePieceTokenizer has no model loaded; "
                "train one with SentencePieceTrainerAdapter or call load_model",
            )

        return self._processor

    def __repr__(self) -> str:
        return f"SentencePieceTokenizer(model_path={str(self._model_path)!r})"


@TOKENIZERS.register("word")
class WordTokenizer(Tokenizer):
    """
    Whole-word tokenizer over the framework's own components.

    Composes a :class:`NormalizerPipeline`, a
    :class:`~multilingual_embedding.tokenizer.pretokenizer.PreTokenizer`
    and a :class:`Vocabulary`. It has no compiled dependency, trains in
    a single pass, and produces tokens that are directly readable, which
    makes it the right tool for tests and for downstream tasks that need
    word identity rather than subword units.

    Parameters
    ----------
    normalizers:
        Ordered normalizer specifications, e.g.
        ``[{"type": "nfkc"}, "whitespace"]``. Specifications rather than
        instances, so the exact configuration can be persisted and
        reloaded.

    pretokenizer:
        Pre-tokenizer specification, e.g. ``{"type": "script"}``.

    min_count:
        Minimum corpus frequency for a token to enter the vocabulary.

    max_size:
        Cap on vocabulary size, including special tokens.

    Example
    -------
    ::

        tokenizer = WordTokenizer(pretokenizer="whitespace")

        tokenizer.train(["hello world", "hello there"])

        tokenizer.decode(tokenizer.encode("hello world").ids)
        -> "hello world"
    """

    __slots__ = (
        "_max_size",
        "_min_count",
        "_normalizer",
        "_normalizer_specs",
        "_pretokenizer",
        "_pretokenizer_spec",
        "_vocabulary",
    )

    def __init__(
        self,
        normalizers: Sequence[Mapping[str, Any] | str] | None = None,
        pretokenizer: Mapping[str, Any] | str | None = None,
        *,
        min_count: int = 1,
        max_size: int | None = None,
    ) -> None:
        self._normalizer_specs: list[Mapping[str, Any] | str] = list(
            normalizers if normalizers is not None else [{"type": "nfkc"}, {"type": "whitespace"}]
        )

        self._pretokenizer_spec: Mapping[str, Any] | str = (
            pretokenizer if pretokenizer is not None else {"type": "whitespace"}
        )

        self._normalizer = NormalizerPipeline.from_config(self._normalizer_specs)

        self._pretokenizer: PreTokenizer = build_from_config(PRETOKENIZERS, self._pretokenizer_spec)

        self._min_count = min_count

        self._max_size = max_size

        self._vocabulary: Vocabulary | None = None

    @property
    def is_fitted(self) -> bool:
        """True once a vocabulary has been built or loaded."""

        return self._vocabulary is not None

    @property
    def vocabulary(self) -> Vocabulary:
        """
        The fitted vocabulary.

        Raises
        ------
        NotFittedError
            If the tokenizer has not been trained or loaded.
        """

        if self._vocabulary is None:
            raise NotFittedError(
                "WordTokenizer has no vocabulary; call train or load first",
            )

        return self._vocabulary

    @property
    def vocabulary_size(self) -> int:
        """Size of the fitted vocabulary."""

        return len(self.vocabulary)

    @property
    def normalizer(self) -> NormalizerPipeline:
        """The normalizer chain applied before pre-tokenization."""

        return self._normalizer

    @property
    def pretokenizer(self) -> PreTokenizer:
        """The pre-tokenizer used to split normalized text."""

        return self._pretokenizer

    def train(self, sentences: Iterable[str]) -> WordTokenizer:
        """
        Build the vocabulary from a corpus.

        Parameters
        ----------
        sentences:
            Training text, consumed once.

        Returns
        -------
        ``self``, so the call can be chained.

        Example
        -------
        ::

            WordTokenizer().train(["hello world"]).vocabulary_size
        """

        builder = VocabularyBuilder(min_count=self._min_count, max_size=self._max_size)

        sentence_count = 0

        for sentence in sentences:
            builder.add_tokens(self.tokenize(sentence))

            sentence_count += 1

        self._vocabulary = builder.build()

        _logger.info(
            "Trained word tokenizer",
            extra={
                "sentences": sentence_count,
                "vocab_size": len(self._vocabulary),
                "min_count": self._min_count,
            },
        )

        return self

    def tokenize(self, text: str) -> list[str]:
        """Normalize ``text`` and return its surface tokens."""

        normalized = self._normalizer.normalize(text)

        return [token.text for token in self._pretokenizer.pre_tokenize(normalized)]

    def encode(self, text: str) -> Encoding:
        """
        Encode ``text`` into vocabulary ids, tokens and spans.

        Spans index the *normalized* text rather than the input, since
        normalization can change character counts and only the
        normalized form is what the pre-tokenizer saw.

        Raises
        ------
        NotFittedError
            If the tokenizer has not been trained or loaded.
        """

        vocabulary = self.vocabulary

        normalized = self._normalizer.normalize(text)

        tokens = self._pretokenizer.pre_tokenize(normalized)

        surfaces = [token.text for token in tokens]

        spans: list[Span] = [token.span for token in tokens]

        ids = vocabulary.encode(surfaces)

        return Encoding(
            ids=ids,
            tokens=surfaces,
            spans=spans,
            attention_mask=[1] * len(ids),
        )

    def decode(self, ids: Sequence[int]) -> str:
        """
        Join the tokens for ``ids`` with single spaces.

        Special tokens are dropped. Whitespace is the only separator
        available at this level, so text in scripts written without
        spaces comes back space-separated; that is inherent to
        word-level tokenization rather than a defect of this method.

        Raises
        ------
        NotFittedError
            If the tokenizer has not been trained or loaded.
        """

        return " ".join(self.vocabulary.decode(list(ids), skip_special=True))

    def save(self, directory: str | Path) -> Path:
        """Write the vocabulary and the component configuration."""

        # Resolve the vocabulary before touching the filesystem, so an
        # unfitted tokenizer does not leave an empty directory behind.
        vocabulary = self.vocabulary

        target = ensure_directory(directory)

        vocabulary.save(target / WORD_VOCABULARY_FILENAME)

        write_json(
            target / WORD_CONFIG_FILENAME,
            {
                "format_version": _FORMAT_VERSION,
                "type": "word",
                "normalizers": [
                    dict(spec) if isinstance(spec, Mapping) else spec
                    for spec in self._normalizer_specs
                ],
                "pretokenizer": dict(self._pretokenizer_spec)
                if isinstance(self._pretokenizer_spec, Mapping)
                else self._pretokenizer_spec,
                "min_count": self._min_count,
                "max_size": self._max_size,
            },
        )

        return target

    @classmethod
    def load(cls, directory: str | Path) -> WordTokenizer:
        """
        Rebuild a tokenizer from a directory written by :meth:`save`.

        Raises
        ------
        ResourceNotFoundError
            If the configuration or vocabulary file is missing.
        """

        root = require_directory(directory)

        config_path = root / WORD_CONFIG_FILENAME

        vocabulary_path = root / WORD_VOCABULARY_FILENAME

        for path in (config_path, vocabulary_path):
            if not path.exists():
                raise ResourceNotFoundError(
                    "Word tokenizer artefact is missing",
                    directory=str(root),
                    expected=str(path),
                )

        payload = read_json(config_path)

        tokenizer = cls(
            normalizers=payload.get("normalizers"),
            pretokenizer=payload.get("pretokenizer"),
            min_count=payload.get("min_count", 1),
            max_size=payload.get("max_size"),
        )

        tokenizer._vocabulary = Vocabulary.load(vocabulary_path)

        return tokenizer

    def __repr__(self) -> str:
        size = len(self._vocabulary) if self._vocabulary is not None else None

        return f"WordTokenizer(vocabulary_size={size})"
