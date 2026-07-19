"""
SentencePiece model training.

SentencePiece is a C++ trainer that reads a file and writes a file; it
does not accept an in-memory iterator for the corpus sizes this
framework targets, and its failure modes surface as bare ``RuntimeError``
with a message. This adapter contains all of that: it stages sentences
to a temporary file, pins the special token ids to the framework's, and
translates the one error every user hits into a
:class:`~multilingual_embedding.core.exceptions.ConfigurationError` that
says what to change.

Artefacts are only moved into their final location once training has
succeeded, so an interrupted run never leaves a half-written model where
the next run will load it.

The configured normalizer chain is applied here, as the corpus is
staged. SentencePiece learns its pieces from whatever bytes reach that
file, so normalizing at training time only would be worse than not
normalizing at all: encode-time text would then differ from what the
model saw. :class:`~multilingual_embedding.tokenizer.tokenizer.SentencePieceTokenizer`
therefore carries the same chain, and
:class:`~multilingual_embedding.pipelines.training.TrainingPipeline`
hands it the one this adapter trained with.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Iterable
from pathlib import Path

import sentencepiece

from multilingual_embedding.common.constants import DEFAULT_ENCODING
from multilingual_embedding.config.base import TokenizerConfig
from multilingual_embedding.core.exceptions import (
    ConfigurationError,
    ResourceNotFoundError,
    ValidationError,
)
from multilingual_embedding.core.logging import get_logger
from multilingual_embedding.utils.filesystem import atomic_write_path, ensure_directory
from multilingual_embedding.vocabulary.special_tokens import (
    BOS_ID,
    EOS_ID,
    PAD_ID,
    UNK_ID,
    SpecialTokenSet,
)

from .normalizer import NormalizerPipeline

__all__ = ["SentencePieceTrainerAdapter"]

_logger = get_logger(__name__)

# Substrings identifying the two vocab_size failures, matched
# case-insensitively. They are opposite problems and must not be
# conflated: one wants a smaller vocab_size, the other a larger one.
_VOCAB_TOO_HIGH_MARKERS = (
    "vocabulary size too high",
    "please set it to a value",
)

_VOCAB_TOO_LOW_MARKERS = (
    "smaller than required_chars",
    "increase vocab_size",
)

# Applying the whitespace pre-tokenizer to SentencePiece input and
# re-joining on spaces is the identity, because staging already collapses
# whitespace runs. Any other pre-tokenizer would genuinely change the
# input, and SentencePiece has no hook to honour it — hence the warning.
_INERT_PRETOKENIZER_TYPES = frozenset({"whitespace"})


class SentencePieceTrainerAdapter:
    """
    Trains a SentencePiece model from an iterable of sentences.

    Parameters
    ----------
    config:
        Tokenizer settings. Defaults to :class:`TokenizerConfig`, whose
        defaults target a 32k unigram model.

    special_tokens:
        Surface forms for the reserved pieces. Must match the vocabulary
        the trained model will be paired with.

    Example
    -------
    ::

        adapter = SentencePieceTrainerAdapter(TokenizerConfig(vocab_size=200))

        model_path = adapter.train(sentences, Path("artifacts/tokenizer"))
    """

    __slots__ = ("_config", "_normalizer", "_special")

    def __init__(
        self,
        config: TokenizerConfig | None = None,
        *,
        special_tokens: SpecialTokenSet | None = None,
    ) -> None:
        self._config = config if config is not None else TokenizerConfig()

        self._special = special_tokens if special_tokens is not None else SpecialTokenSet()

        self._normalizer = NormalizerPipeline.from_config(self._config.normalizers)

        # Warn at construction rather than at train time: the caller
        # learns the setting will not take effect as soon as they express
        # it, and the message is not buried under SentencePiece's own
        # training output.
        self._warn_if_pretokenizer_is_inapplicable()

    @property
    def config(self) -> TokenizerConfig:
        """The tokenizer settings this adapter trains with."""

        return self._config

    @property
    def normalizer(self) -> NormalizerPipeline:
        """
        The normalizer chain applied to the training corpus.

        A tokenizer built over the resulting model must apply this same
        chain, or encode-time text will not match what was trained on.
        """

        return self._normalizer

    def train(self, sentences: Iterable[str], output_directory: str | Path) -> Path:
        """
        Train a model and write its artefacts to ``output_directory``.

        Parameters
        ----------
        sentences:
            Training text, one sentence per element. Consumed once;
            blank entries are skipped. Embedded newlines are replaced
            with spaces, since the staging format is line-delimited and
            a stray newline would silently split a training example.

        output_directory:
            Directory to receive ``<model_prefix>.model`` and
            ``<model_prefix>.vocab``. Created if absent.

        Returns
        -------
        Path to the trained ``.model`` file.

        Raises
        ------
        ValidationError
            If no non-blank sentences were supplied.

        ConfigurationError
            If SentencePiece rejects the settings — most often because
            ``vocab_size`` exceeds the number of distinct pieces the
            corpus can support.

        Example
        -------
        ::

            adapter.train(["hello world", "नमस्ते दुनिया"], tmp_path)
        """

        target_directory = ensure_directory(output_directory)

        with tempfile.TemporaryDirectory(prefix="sentencepiece-") as staging:
            staging_path = Path(staging)

            corpus_path = staging_path / "corpus.txt"

            sentence_count = self._stage_corpus(sentences, corpus_path)

            if sentence_count == 0:
                raise ValidationError(
                    "Cannot train a tokenizer on an empty corpus",
                    output_directory=str(target_directory),
                )

            model_prefix = staging_path / self._config.model_prefix

            self._run_trainer(corpus_path, model_prefix, sentence_count)

            model_path = self._publish(model_prefix, target_directory)

        _logger.info(
            "Trained SentencePiece model",
            extra={
                "path": str(model_path),
                "sentences": sentence_count,
                "vocab_size": self._config.vocab_size,
                "model_type": self._config.model_type.value,
            },
        )

        return model_path

    def _warn_if_pretokenizer_is_inapplicable(self) -> None:
        """
        Say so when ``config.pretokenizer`` cannot be honoured here.

        SentencePiece consumes a raw character stream by design, which is
        what lets one model serve scripts with no whitespace word
        boundaries; there is no point at which a framework pre-tokenizer
        could be inserted without destroying that property. The setting
        is still accepted — it configures
        :class:`~multilingual_embedding.tokenizer.tokenizer.WordTokenizer`
        — so the only wrong response is to drop it silently.
        """

        spec = self._config.pretokenizer

        name = spec.get("type") if isinstance(spec, dict) else spec

        if name in _INERT_PRETOKENIZER_TYPES:
            return

        _logger.warning(
            "SentencePiece cannot apply a pre-tokenizer; the tokenizer.pretokenizer "
            "setting affects WordTokenizer only and is not in effect for this model",
            extra={"pretokenizer": name, "model_type": self._config.model_type.value},
        )

    def _stage_corpus(self, sentences: Iterable[str], corpus_path: Path) -> int:
        """
        Write sentences one per line and return how many were written.

        Each sentence passes through the configured normalizer chain
        first, so the pieces SentencePiece learns are pieces of
        normalized text. Streamed rather than buffered so that memory
        stays flat for corpora far larger than RAM.
        """

        written = 0

        with corpus_path.open("w", encoding=DEFAULT_ENCODING) as handle:
            for sentence in sentences:
                # Normalize first, flatten second: a normalizer may emit
                # a newline, and the staging format is line-delimited.
                cleaned = " ".join(self._normalizer.normalize(sentence).split())

                if not cleaned:
                    continue

                handle.write(cleaned)
                handle.write("\n")

                written += 1

        return written

    def _run_trainer(self, corpus_path: Path, model_prefix: Path, sentence_count: int) -> None:
        """Invoke the SentencePiece trainer, translating its failures."""

        try:
            sentencepiece.SentencePieceTrainer.train(
                input=str(corpus_path),
                model_prefix=str(model_prefix),
                vocab_size=self._config.vocab_size,
                character_coverage=self._config.character_coverage,
                model_type=self._config.model_type.value,
                max_sentence_length=self._config.max_sentence_length,
                # These ids MUST match vocabulary/special_tokens.py. The
                # SentencePiece defaults differ (no pad, unk=0, bos=1,
                # eos=2), and a mismatch is silent: encoding would return
                # ids that index the wrong rows of an embedding matrix
                # built against our Vocabulary, producing a model that
                # trains without error and is quietly wrong.
                pad_id=PAD_ID,
                unk_id=UNK_ID,
                bos_id=BOS_ID,
                eos_id=EOS_ID,
                pad_piece=self._special.pad,
                unk_piece=self._special.unk,
                bos_piece=self._special.bos,
                eos_piece=self._special.eos,
            )
        except RuntimeError as error:
            raise self._translate_failure(error, sentence_count) from error

    def _translate_failure(self, error: RuntimeError, sentence_count: int) -> ConfigurationError:
        """
        Turn a SentencePiece ``RuntimeError`` into a framework error.

        The corpus-too-small case is by far the most common way training
        fails, and SentencePiece reports it as an opaque runtime error,
        so it is given a message naming both the requested size and the
        knob to turn. Its mirror image — a vocab_size too small to hold
        the corpus's character inventory, which multilingual corpora hit
        easily — is reported separately, because telling a user to
        shrink vocab_size there would send them the wrong way.
        """

        message = str(error)

        lowered = message.lower()

        if any(marker in lowered for marker in _VOCAB_TOO_HIGH_MARKERS):
            return ConfigurationError(
                "vocab_size exceeds what the training corpus can support; "
                "reduce vocab_size or supply more text",
                requested_vocab_size=self._config.vocab_size,
                sentences=sentence_count,
                model_type=self._config.model_type.value,
                reason=message,
            )

        if any(marker in lowered for marker in _VOCAB_TOO_LOW_MARKERS):
            return ConfigurationError(
                "vocab_size is too small to cover the characters in the corpus; "
                "raise vocab_size or lower character_coverage",
                requested_vocab_size=self._config.vocab_size,
                character_coverage=self._config.character_coverage,
                sentences=sentence_count,
                reason=message,
            )

        return ConfigurationError(
            "SentencePiece training failed",
            requested_vocab_size=self._config.vocab_size,
            sentences=sentence_count,
            model_type=self._config.model_type.value,
            reason=message,
        )

    def _publish(self, model_prefix: Path, target_directory: Path) -> Path:
        """
        Move the trained artefacts out of staging into their home.

        Each file is placed atomically, so a reader either sees the
        previous model or the new one and never a partial file.
        """

        model_path = target_directory / f"{self._config.model_prefix}.model"

        for suffix, destination in (
            (".model", model_path),
            (".vocab", target_directory / f"{self._config.model_prefix}.vocab"),
        ):
            # String concatenation rather than Path.with_suffix, which
            # would replace an existing dotted component of a model
            # prefix such as "tokenizer.v2".
            source = Path(f"{model_prefix}{suffix}")

            if not source.exists():
                if suffix == ".model":
                    raise ResourceNotFoundError(
                        "SentencePiece reported success but produced no model file",
                        expected=str(source),
                    )

                continue

            with atomic_write_path(destination) as temporary:
                shutil.copyfile(source, temporary)

        return model_path
