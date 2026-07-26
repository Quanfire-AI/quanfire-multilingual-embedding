"""
Fine-tuning a from-scratch encoder, and measuring whether it helped.

This is stage two of the from-scratch neural path. ``qfme pretrain``
produces an encoder that has only ever seen a masked-language objective:
it can fill in a blanked-out token, but it was never taught that two
paraphrases should land near each other, so as a sentence embedder it is
weak. This module fixes that, and the shape of the run is deliberately
the same three steps as adaptation:

1. Score the pretrained encoder on held-out pairs. This is the number to
   beat, and it is honest because it is the same encoder measured against
   the same pairs it will be scored on afterwards.
2. Fine-tune the whole encoder on mined pairs with an in-batch
   contrastive loss.
3. Score it again on the same held-out pairs.

Where :mod:`~multilingual_embedding.pipelines.adaptation` freezes a
published checkpoint and trains a LoRA adapter over it, this owns the
weights and moves all of them. There is no base model to preserve, so
there is no adapter and no vary/hold ablation machinery — this is not an
experiment in what transfers, it is the step that turns a pretrained
artefact into a usable model. What it writes is therefore a complete,
servable experiment directory (``tokenizer/`` and ``encoder/``), the same
shape ``qfme pretrain`` wrote, so
:meth:`~multilingual_embedding.pipelines.search.SemanticSearchPipeline.from_directory`
serves the fine-tuned model the moment it lands.

The held-out set is drawn and the training set has its positives excluded
from it, so the score is never measured on a pair the model just trained
on. And a probe of anchor vectors is re-encoded either side of training,
so a run can tell "the fine-tune did not help" apart from "the weights
did not move" — the two have opposite remedies.

Like stage one, the contrastive step is interruptible: ``run`` forwards a
``checkpoint_dir``/``resume_from`` pair through to
:meth:`ContrastiveTrainer.train`, so a long fine-tune on a shared GPU box
can be checkpointed each epoch and resumed bit-for-bit. The baseline is
re-scored from the untouched pretrained source on every resume, so the
"number to beat" is reproduced rather than reloaded.

Torch is imported inside :meth:`FinetunePipeline.run` rather than at
module scope, because the neural stack is an optional extra and the
config and pipeline layers must stay importable without it.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from multilingual_embedding.config.base import ComputeConfig, ExperimentConfig, FinetuneConfig
from multilingual_embedding.config.loader import save_config
from multilingual_embedding.core.exceptions import ConfigurationError, ResourceNotFoundError
from multilingual_embedding.core.logging import get_logger
from multilingual_embedding.corpus.pairs import MinedPair, sample_pairs
from multilingual_embedding.evaluation.retrieval import RetrievalReport, evaluate_retrieval
from multilingual_embedding.pipelines.adaptation import only, values
from multilingual_embedding.tokenizer.tokenizer import SentencePieceTokenizer
from multilingual_embedding.utils.filesystem import ensure_directory

__all__ = ["FinetunePipeline", "FinetuneResult", "finetune"]

_logger = get_logger(__name__)

# How many anchors to re-encode after training to prove the weights
# actually moved. Sixteen is enough to make a zero maximum difference mean
# something and small enough to be free — the same probe adaptation uses.
_PROBE = 16


@dataclass(slots=True)
class FinetuneResult:
    """
    What one fine-tuning run produced.

    Attributes
    ----------
    source:
        The pretrained encoder directory the run started from, unchanged
        on disk — the fine-tuned weights are written elsewhere.

    before, after:
        Retrieval reports on the *same* held-out pairs, either side of
        training.

    train_examples, eval_examples:
        The counts actually used after filtering, not the ones requested.

    moved:
        Largest absolute change in a probe vector, before against after.
        Kept so an unchanged score can be read as "did not help" rather
        than "did not train" — the two have opposite remedies.

    encoder_directory:
        Where the servable fine-tuned encoder was written, or ``None`` for
        a measurement-only run that shipped nothing.
    """

    source: Path

    before: RetrievalReport

    after: RetrievalReport

    trained_on: Path

    scored_against: Path

    train_examples: int

    eval_examples: int

    sampled_from: int

    train_kinds: list[str] = field(default_factory=list)

    eval_kinds: list[str] = field(default_factory=list)

    train_languages: list[str] = field(default_factory=list)

    eval_languages: list[str] = field(default_factory=list)

    initial_loss: float = 0.0

    final_loss: float = 0.0

    measurable: bool = False

    moved: float = 0.0

    seconds: float = 0.0

    data_provenance: str = ""

    encoder_directory: Path | None = None

    @property
    def delta(self) -> float:
        """Absolute change in recall@1."""

        return self.after.overall.recall_at_1 - self.before.overall.recall_at_1

    @property
    def relative(self) -> float:
        """Change in recall@1 as a percentage of the baseline."""

        baseline = self.before.overall.recall_at_1

        return (self.delta / baseline * 100.0) if baseline else 0.0

    @property
    def helped(self) -> bool:
        """Whether the fine-tuned encoder beat the pretrained one."""

        return self.delta > 0

    @property
    def trained(self) -> bool:
        """
        Whether the weights moved at all.

        A run where this is ``False`` has a training problem rather than a
        result, and the distinction decides what to try next.
        """

        return self.moved >= 1e-4

    def summary(self) -> str:
        """The comparison, as the lines worth reading off a terminal."""

        verdict = "BETTER" if self.delta > 0 else ("NO CHANGE" if self.delta == 0 else "WORSE")

        lines = [
            f"{'pretrained encoder':24} recall@1 {self.before.overall.recall_at_1:.4f}"
            f"   MRR {self.before.overall.mrr:.4f}",
            f"{'after contrastive':24} recall@1 {self.after.overall.recall_at_1:.4f}"
            f"   MRR {self.after.overall.mrr:.4f}",
            "",
            f"recall@1 {self.delta:+.4f}  ({self.relative:+.1f}%)   -> {verdict}",
            f"the model itself moved by {self.moved:.6f} (max change in a probe vector)",
        ]

        if self.delta == 0 and not self.trained:
            lines += [
                "",
                "The score did not change because the MODEL barely changed.",
                "That is a training problem rather than a result: try a larger",
                "learning rate or more epochs before concluding anything about",
                "the pairs.",
            ]
        elif self.delta <= 0:
            lines += [
                "",
                "The model changed but the score did not improve. Fine-tuning",
                "on pairs that are largely solvable by string matching, or too",
                "few of them, is a likely reason.",
            ]

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Reduce to primitives for persistence."""

        return {
            "source": str(self.source),
            "trained_on": str(self.trained_on),
            "scored_against": str(self.scored_against),
            "train_kinds": self.train_kinds,
            "eval_kinds": self.eval_kinds,
            "train_languages": self.train_languages,
            "eval_languages": self.eval_languages,
            "train_examples": self.train_examples,
            "eval_examples": self.eval_examples,
            "sampled_from": self.sampled_from,
            "loss": [self.initial_loss, self.final_loss],
            "measurable": self.measurable,
            "moved": self.moved,
            "data_provenance": self.data_provenance,
            "recall_at_1_before": round(self.before.overall.recall_at_1, 4),
            "recall_at_1_after": round(self.after.overall.recall_at_1, 4),
            "delta": round(self.delta, 4),
            "relative_percent": round(self.relative, 2),
            "seconds": round(self.seconds, 1),
            "encoder_directory": str(self.encoder_directory) if self.encoder_directory else None,
            "before": self.before.to_dict(),
            "after": self.after.to_dict(),
        }


class FinetunePipeline:
    """
    Runs the three-step fine-tuning experiment from a resolved config.

    Parameters
    ----------
    config:
        The experiment. Its ``finetune`` section carries the science and
        its ``compute`` section carries what the machine dictates, which
        is the split that lets one file describe a run on a laptop and on
        a GPU box.

    echo:
        Where progress goes. Defaults to :func:`print`, because the
        primary caller is a CLI and a training run that says nothing for
        forty minutes is indistinguishable from a hung one. Pass a no-op
        to silence it.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        *,
        echo: Callable[[str], None] = print,
    ) -> None:
        self.config = config

        self.echo = echo

    @property
    def finetune(self) -> FinetuneConfig:
        """The science half of the run."""

        return self.config.finetune

    @property
    def compute(self) -> ComputeConfig:
        """The machine half of the run — the section a profile swaps."""

        return self.config.compute

    def run(
        self,
        *,
        checkpoint_dir: str | Path | None = None,
        resume_from: str | Path | None = None,
        stop_after_epoch: int | None = None,
    ) -> FinetuneResult:
        """
        Score, fine-tune, score again, and write a servable encoder.

        Stage two is a long run on a shared GPU box, so — like ``qfme
        pretrain`` — the contrastive step is interruptible. Pass
        ``checkpoint_dir`` to write a rolling checkpoint each epoch and
        ``resume_from`` to pick one back up; the baseline is re-scored from
        the untouched pretrained source each time, so a resumed run
        reproduces the number it would have had uninterrupted.

        Parameters
        ----------
        checkpoint_dir:
            Where the contrastive trainer writes ``checkpoint.pt`` after
            each epoch. ``None`` fine-tunes without checkpointing.

        resume_from:
            A checkpoint file, or a directory holding one, to restore
            before training. The pair set and schedule must match the run
            that wrote it, or the resume is refused.

        stop_after_epoch:
            Stop after finishing this 0-based epoch index, modelling an
            interruption. The learning-rate schedule still spans the full
            configured epoch count, so a later resume continues it unbroken.

        Raises
        ------
        ConfigurationError
            If the source or pair file is unset, or a facet filter selects
            nothing. Checked before the encoder is loaded, so a mistake
            costs seconds rather than an hour on a GPU.

        ResourceNotFoundError
            If the source directory is missing its ``tokenizer/`` or
            ``encoder/``.
        """

        settings = self.finetune

        if settings.source is None:
            raise ConfigurationError(
                "finetune.source is required — point it at a `qfme pretrain` "
                "experiment directory holding tokenizer/ and encoder/"
            )

        if settings.pairs is None:
            raise ConfigurationError(
                "finetune.pairs is required — point it at a file written by qfme mine-pairs"
            )

        tokenizer_directory = settings.source / "tokenizer"

        encoder_directory = settings.source / "encoder"

        for directory, label in (
            (tokenizer_directory, "tokenizer"),
            (encoder_directory, "encoder"),
        ):
            if not directory.is_dir():
                raise ResourceNotFoundError(
                    "Pretrained source directory is missing a required component",
                    component=label,
                    expected=str(directory),
                )

        from multilingual_embedding.embedding.neural import (
            ContrastiveConfig,
            ContrastiveTrainer,
            NeuralTextEncoder,
            TextPair,
        )

        started = time.time()

        train, held, facts = self._select()

        tokenizer = SentencePieceTokenizer.load(tokenizer_directory)

        encoder = NeuralTextEncoder.load(
            encoder_directory,
            tokenizer,
            device=self._device,
            batch_size=self.compute.batch_size,
        )

        self.echo(f"\ndevice {encoder.device}   dimension {encoder.dimension}")

        self.echo("\n[1] scoring the pretrained encoder — this is the number to beat")

        probe = [pair.anchor for pair in held[:_PROBE]]

        probe_before = encoder.encode_batch(probe)

        before = evaluate_retrieval(encoder, held, limit=None)

        self.echo(before.summary())

        self.echo(f"\n[2] contrastive fine-tune: {len(train):,} pairs, {settings.epochs} epoch(s)")

        report = ContrastiveTrainer(
            encoder,
            ContrastiveConfig.for_compute(
                self.compute,
                epochs=settings.epochs,
                learning_rate=settings.learning_rate,
                temperature=settings.temperature,
                warmup_ratio=settings.warmup_ratio,
                weight_decay=settings.weight_decay,
                seed=settings.seed or 0,
            ),
        ).train(
            [TextPair(pair.anchor, pair.positive, pair.negatives) for pair in train],
            checkpoint_dir=checkpoint_dir,
            resume_from=resume_from,
            stop_after_epoch=stop_after_epoch,
        )

        if report.measurable:
            self.echo(f"    loss {report.initial_loss:.4f} -> {report.final_loss:.4f}")
        else:
            self.echo(
                f"    loss {report.final_loss:.4f} (one epoch, so there is nothing to "
                f"compare it against — the retrieval scores below are the evidence)"
            )

        self.echo("\n[3] scoring the fine-tuned encoder on the same held-out pairs")

        after = evaluate_retrieval(encoder, held, limit=None)

        self.echo(after.summary())

        moved = float(abs(encoder.encode_batch(probe) - probe_before).max())

        result = FinetuneResult(
            source=settings.source,
            before=before,
            after=after,
            trained_on=settings.pairs,
            scored_against=facts["evaluation_source"],
            train_examples=len(train),
            eval_examples=len(held),
            sampled_from=settings.sampled,
            train_kinds=facts["train_kinds"],
            eval_kinds=facts["eval_kinds"],
            train_languages=facts["train_languages"],
            eval_languages=facts["eval_languages"],
            initial_loss=report.initial_loss,
            final_loss=report.final_loss,
            measurable=report.measurable,
            moved=moved,
            data_provenance=settings.data_provenance,
            seconds=time.time() - started,
        )

        if settings.save:
            # The tokenizer is copied in beside the fine-tuned weights so
            # the output is a complete, self-contained experiment directory
            # that from_directory can serve without reaching back to the
            # source. Written after scoring so a run that fails to improve
            # has still measured itself before anything lands on disk.
            ensure_directory(self.config.experiment_directory)

            tokenizer.save(self.config.tokenizer_directory)

            encoder.save(self.config.encoder_directory)

            save_config(self.config, self.config.experiment_directory / "config.yaml")

            result.encoder_directory = self.config.encoder_directory

            self.echo(f"\nfine-tuned encoder written to {self.config.experiment_directory}")

        if settings.report:
            from multilingual_embedding.utils.io import write_json

            write_json(settings.report, result.to_dict())

            self.echo(f"report written to {settings.report}")

        self.echo(f"\nTOTAL {result.seconds:.0f}s")

        return result

    @property
    def _device(self) -> str | None:
        """
        The device to load onto, or ``None`` to pick the best available.

        ``"auto"`` is the config's way of deferring to the machine, which
        is exactly what passing ``None`` to the encoder loader means.
        """

        return None if self.compute.device == "auto" else self.compute.device

    def _select(self) -> tuple[list[MinedPair], list[MinedPair], dict[str, Any]]:
        """
        Draw the training and held-out sets, keeping them disjoint.

        The held-out set is drawn from a fixed-size sample of its own, so
        changing the training volume does not change what the model is
        scored on. Training positives that appear in the held-out set are
        removed, so the score is never measured on a pair the model just
        trained on.
        """

        settings = self.finetune

        assert settings.pairs is not None  # checked by run() before this is reached

        seed = settings.seed or 0

        pool = sample_pairs(settings.pairs, settings.sampled, seed=seed)

        if len(pool) < settings.sampled:
            self.echo(f"\nonly {len(pool)} pairs available; wanted {settings.sampled}")

        evaluation_source = settings.eval_pairs_file or settings.pairs

        evaluation = sample_pairs(evaluation_source, settings.eval_pairs, seed=seed + 10_000)

        held_texts = {pair.positive for pair in evaluation}

        candidates = only(
            only([p for p in pool if p.positive not in held_texts], settings.train_kinds, "kind"),
            settings.train_languages,
            "language",
        )

        train = candidates[: settings.train_pairs]

        held = only(
            only(evaluation, settings.eval_kinds, "kind"),
            settings.eval_languages,
            "language",
        )

        facts: dict[str, Any] = {
            "evaluation_source": evaluation_source,
            "train_kinds": values(train, "kind"),
            "eval_kinds": values(held, "kind"),
            "train_languages": values(train, "language"),
            "eval_languages": values(held, "language"),
        }

        self.echo(f"\ntrain {len(train):,} pairs   held out {len(held):,} pairs")

        if settings.eval_pairs_file:
            self.echo(f"scored against {evaluation_source} (held fixed)")

        return train, held, facts


def finetune(
    config: ExperimentConfig,
    *,
    echo: Callable[[str], None] = print,
    checkpoint_dir: str | Path | None = None,
    resume_from: str | Path | None = None,
    stop_after_epoch: int | None = None,
) -> FinetuneResult:
    """Run one fine-tuning experiment. Convenience wrapper over the pipeline."""

    return FinetunePipeline(config, echo=echo).run(
        checkpoint_dir=checkpoint_dir,
        resume_from=resume_from,
        stop_after_epoch=stop_after_epoch,
    )
