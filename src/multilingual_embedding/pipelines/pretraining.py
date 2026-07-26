"""
From-scratch masked-language pretraining pipeline.

The contextual counterpart to :mod:`multilingual_embedding.pipelines.training`.
Where that pipeline runs corpus → tokenizer → Word2Vec → a co-occurrence
table, this one runs:

    corpus -> tokenizer -> fresh encoder -> masked-language pretraining

and produces a transformer that has learned what its tokens mean by
predicting ones that were hidden from it. It is stage one of the
from-scratch neural path; contrastive fine-tuning on pairs is stage two,
and both share the one :class:`NeuralTextEncoder` this pipeline builds and
saves.

Two properties are deliberate:

*It materialises the corpus.* Unlike the streaming static pipeline, the
masked-language objective shuffles documents by index each epoch, so the
corpus is held as a list. Corpus size is bounded by memory here rather than
by disk — the price of a reproducible per-epoch ordering that a resumed run
can reproduce exactly.

*It is interruptible.* A from-scratch run is long and a shared GPU box is
not always yours for its whole length. Pass ``checkpoint_dir`` to write a
rolling checkpoint each epoch and ``resume_from`` to pick one back up; a
resumed run reproduces bit-for-bit the run an uninterrupted one would have
produced. This is the pipeline built to be run on the training box.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from multilingual_embedding.common.enums import DataProvenance
from multilingual_embedding.config.base import ExperimentConfig
from multilingual_embedding.config.loader import save_config
from multilingual_embedding.core.exceptions import ConfigurationError
from multilingual_embedding.core.logging import get_logger
from multilingual_embedding.corpus.iterator import SentenceStream
from multilingual_embedding.corpus.loader import stream_documents, stream_sentences
from multilingual_embedding.corpus.statistics import CorpusStatistics, StatisticsAccumulator
from multilingual_embedding.tokenizer.tokenizer import SentencePieceTokenizer
from multilingual_embedding.tokenizer.trainer import SentencePieceTrainerAdapter
from multilingual_embedding.utils.filesystem import ensure_directory

__all__ = ["PretrainingPipeline", "PretrainingResult", "pretrain"]

_logger = get_logger(__name__)

# SentencePiece is trained with the framework's special ids pinned, so
# pieces 0-3 are always pad/unk/bos/eos (see
# ``SentencePieceTokenizer.to_vocabulary``). BOS and EOS bracket every
# sequence and carry no content to predict, so they are held out of masking
# the way BERT holds out ``[CLS]``/``[SEP]``. PAD is excluded automatically
# by the attention mask, and UNK is a legitimate prediction target.
_BOS_ID = 2
_EOS_ID = 3


@dataclass(slots=True)
class PretrainingResult:
    """
    Everything one pretraining run produced.

    Attributes
    ----------
    config:
        The resolved configuration used.

    corpus_statistics:
        Statistics over the filtered corpus actually trained on.

    tokenizer:
        The subword model trained alongside the encoder.

    report:
        What the masked-language run did — losses, masked accuracy, and
        whether it improved.

    parameters:
        Trainable parameter count of the pretrained encoder.

    experiment_directory, tokenizer_directory, encoder_directory:
        Where the artefacts were written.
    """

    config: ExperimentConfig

    corpus_statistics: CorpusStatistics

    tokenizer: SentencePieceTokenizer

    report: Any

    parameters: int

    experiment_directory: Path

    tokenizer_directory: Path

    encoder_directory: Path

    def summary(self) -> dict[str, Any]:
        """Return a compact summary suitable for logging or printing."""

        return {
            "name": self.config.name,
            "documents": self.corpus_statistics.document_count,
            "sentences": self.corpus_statistics.sentence_count,
            "vocabulary_size": self.tokenizer.vocabulary_size,
            "dimension": self.config.pretraining.dimension,
            "parameters": self.parameters,
            "epochs": self.report.epochs,
            "steps": self.report.steps,
            "initial_loss": round(self.report.initial_loss, 6),
            "final_loss": round(self.report.final_loss, 6),
            "initial_accuracy": round(self.report.initial_accuracy, 6),
            "final_accuracy": round(self.report.final_accuracy, 6),
            "improved": self.report.improved,
            "measurable": self.report.measurable,
            "encoder_directory": str(self.encoder_directory),
        }


class PretrainingPipeline:
    """
    Orchestrates a from-scratch masked-language pretraining run.

    Parameters
    ----------
    config:
        The experiment configuration. Its ``corpus.source`` must be set.
        The ``pretraining`` and ``compute`` sections govern the run; the
        ``embedding`` section (Word2Vec) is ignored.

    Example
    -------
    ::

        config = load_config("experiments/hindi.yaml")

        result = PretrainingPipeline(config).run(checkpoint_dir="ckpts")

        assert result.report.final_accuracy > result.report.initial_accuracy
    """

    __slots__ = ("_config",)

    def __init__(self, config: ExperimentConfig) -> None:
        if config.corpus.source is None:
            raise ConfigurationError(
                "Cannot pretrain without a corpus source",
                experiment=config.name,
            )

        self._config = config

    @property
    def config(self) -> ExperimentConfig:
        """The configuration this pipeline runs."""

        return self._config

    def run(
        self,
        *,
        checkpoint_dir: str | Path | None = None,
        resume_from: str | Path | None = None,
        stop_after_epoch: int | None = None,
    ) -> PretrainingResult:
        """
        Train a tokenizer and pretrain an encoder over it.

        Parameters
        ----------
        checkpoint_dir:
            Where to write ``checkpoint.pt`` after each epoch. ``None``
            pretrains without checkpointing.

        resume_from:
            A checkpoint file, or a directory holding one, to restore
            before training. ``None`` starts fresh. The corpus and schedule
            must match the run that wrote it, or the resume is refused.

        stop_after_epoch:
            Stop after finishing this epoch index (0-based). Models an
            interruption; the learning-rate schedule still spans the full
            configured epoch count, so a later resume continues it unbroken.
        """

        config = self._config

        # A pretrained encoder is a model trained on the corpus, so it
        # carries the same provenance obligation a fine-tuned one does.
        # Unlike finetune there is no measurement-only mode here — pretrain
        # always writes an encoder — so the declaration is unconditional.
        # Checked before anything lands on disk, so a missing label costs a
        # second rather than an epoch on the GPU. Customer data must never
        # reach a saved model.
        if not config.pretraining.data_provenance:
            raise ConfigurationError(
                "Pretraining saves an encoder trained on the corpus, so it "
                "requires declaring pretraining.data_provenance (where the "
                "corpus came from). Pass --data-provenance, or set it in the "
                "config. Customer data must never reach a saved model.",
                supported=sorted(DataProvenance),
                experiment=config.name,
            )

        experiment_directory = ensure_directory(config.experiment_directory)

        _logger.info(
            "Starting pretraining run",
            extra={"experiment": config.name, "directory": str(experiment_directory)},
        )

        # The resolved configuration is persisted first, so an interrupted
        # run still leaves a record of what was attempted.
        save_config(config, experiment_directory / "config.yaml")

        statistics = self._collect_statistics()

        sentences = stream_sentences(config.corpus)

        tokenizer = self._train_tokenizer(sentences)

        # The masked-language objective shuffles by index each epoch, so
        # the corpus is materialised rather than streamed. This is the one
        # stage that holds the corpus in memory.
        documents = list(sentences)

        report, parameters = self._pretrain(
            documents,
            tokenizer,
            checkpoint_dir=checkpoint_dir,
            resume_from=resume_from,
            stop_after_epoch=stop_after_epoch,
        )

        _logger.info(
            "Pretraining run complete",
            extra={
                "experiment": config.name,
                "final_loss": report.final_loss,
                "final_accuracy": report.final_accuracy,
            },
        )

        return PretrainingResult(
            config=config,
            corpus_statistics=statistics,
            tokenizer=tokenizer,
            report=report,
            parameters=parameters,
            experiment_directory=experiment_directory,
            tokenizer_directory=config.tokenizer_directory,
            encoder_directory=config.encoder_directory,
        )

    # ------------------------------------------------------------------
    # Stages
    # ------------------------------------------------------------------

    def _collect_statistics(self) -> CorpusStatistics:
        """Describe the corpus that will actually be trained on."""

        accumulator = StatisticsAccumulator()

        accumulator.extend(stream_documents(self._config.corpus))

        statistics = accumulator.result()

        if statistics.sentence_count == 0:
            raise ConfigurationError(
                "Corpus produced no sentences after filtering",
                source=str(self._config.corpus.source),
                minimum_characters=self._config.corpus.min_sentence_characters,
            )

        _logger.info("Corpus prepared", extra=statistics.to_dict()["languages"])

        return statistics

    def _train_tokenizer(self, sentences: SentenceStream) -> SentencePieceTokenizer:
        """Train the subword model and return a tokenizer over it."""

        trainer = SentencePieceTrainerAdapter(self._config.tokenizer)

        model_path = trainer.train(sentences, self._config.tokenizer_directory)

        tokenizer = SentencePieceTokenizer(
            model_path,
            normalizers=self._config.tokenizer.normalizers,
        )

        tokenizer.save(self._config.tokenizer_directory)

        _logger.info("Tokenizer trained", extra={"model": str(model_path)})

        return tokenizer

    def _pretrain(
        self,
        documents: list[str],
        tokenizer: SentencePieceTokenizer,
        *,
        checkpoint_dir: str | Path | None,
        resume_from: str | Path | None,
        stop_after_epoch: int | None,
    ) -> tuple[Any, int]:
        """
        Build a fresh encoder and pretrain it with a masked-language head.

        The neural stack is imported here, not at module load, so the
        pipeline module — and the config that drives it — stay importable
        without torch. The heavy dependency is pulled only when a run
        actually starts.
        """

        from multilingual_embedding.embedding.neural import (
            EncoderConfig,
            MaskedLanguageConfig,
            MaskedLanguageModelTrainer,
            NeuralTextEncoder,
            TransformerEncoderModel,
            resolve_device,
        )

        settings = self._config.pretraining

        compute = self._config.compute

        # One row past the real vocabulary is reserved as the mask id. The
        # tokenizer never emits it, so it can never collide with a real
        # token, and the trainer holds it out of the prediction targets.
        mask_token_id = tokenizer.vocabulary_size

        encoder_config = EncoderConfig(
            vocabulary_size=tokenizer.vocabulary_size + 1,
            dimension=settings.dimension,
            layers=settings.layers,
            heads=settings.heads,
            feedforward_dimension=settings.feedforward_dimension,
            max_length=settings.max_length,
            dropout=settings.dropout,
        )

        device = resolve_device(compute.device)

        encoder = NeuralTextEncoder(
            TransformerEncoderModel(encoder_config),
            tokenizer,
            device=device,
        )

        # Machine-shaped settings (batch size, precision) come from the
        # compute section; the science (schedule, masking) from pretraining.
        mlm_config = MaskedLanguageConfig(
            epochs=settings.epochs,
            batch_size=compute.batch_size,
            learning_rate=settings.learning_rate,
            warmup_ratio=settings.warmup_ratio,
            weight_decay=settings.weight_decay,
            max_gradient_norm=settings.max_gradient_norm,
            mask_probability=settings.mask_probability,
            tie_weights=settings.tie_weights,
            seed=settings.resolved_seed,
            precision=compute.precision,
        )

        trainer = MaskedLanguageModelTrainer(
            encoder,
            mask_token_id=mask_token_id,
            protected_ids=(_BOS_ID, _EOS_ID),
            config=mlm_config,
        )

        report = trainer.train(
            documents,
            checkpoint_dir=checkpoint_dir,
            resume_from=resume_from,
            stop_after_epoch=stop_after_epoch,
        )

        encoder.save(self._config.encoder_directory)

        _logger.info(
            "Encoder pretrained and saved",
            extra={
                "directory": str(self._config.encoder_directory),
                "parameters": encoder.model.parameter_count(),
            },
        )

        return report, encoder.model.parameter_count()


def pretrain(
    config: ExperimentConfig,
    *,
    checkpoint_dir: str | Path | None = None,
    resume_from: str | Path | None = None,
    stop_after_epoch: int | None = None,
) -> PretrainingResult:
    """Convenience wrapper around :class:`PretrainingPipeline`."""

    return PretrainingPipeline(config).run(
        checkpoint_dir=checkpoint_dir,
        resume_from=resume_from,
        stop_after_epoch=stop_after_epoch,
    )
