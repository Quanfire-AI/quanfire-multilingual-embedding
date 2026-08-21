"""
Masked-language pretraining — stage one of the from-scratch path.

A model trained from scratch begins with a random embedding matrix: every
token is an arbitrary point, and cosine similarity between any two is
noise. Contrastive fitting in :mod:`.training` can sharpen a
representation that already carries meaning, but it cannot conjure one
from noise — with a few thousand pairs it will overfit the pairs long
before the embedding table learns what the tokens *are*.

Masked-language modelling is how the table learns that, from raw text and
nothing else. Hide a slice of the tokens, ask the model to name them from
their neighbours, and the only way to succeed is to make each token's
vector predict, and be predicted by, the company it keeps. That is the
distributional hypothesis turned into a loss, and it is what the second
stage then refines. So the honest from-scratch recipe is two stages: MLM
here to give the embedding meaning, contrastive there to shape it for
retrieval. LoRA adaptation skips this entirely, because it starts from a
checkpoint whose table is already meaningful.

Three pieces make it up, and they separate the way the rest of the
subpackage does — tensors on one side, text on the other:

``MaskedLanguageModelHead`` / ``MaskedLanguageModel``
    Tensors in, logits out. The head sits on the encoder's per-token
    states (:meth:`TransformerEncoderModel.hidden_states`) and predicts a
    distribution over the vocabulary at every position. It is scaffolding:
    trained here, discarded afterwards, never saved. Only the encoder
    underneath it survives.

``mask_tokens``
    The corruption itself — a pure function over id tensors, so it is
    testable without a model. It follows the recipe BERT established:
    select 15% of real tokens, replace most with ``[MASK]``, a few with a
    random token, leave the rest intact, and supervise only the selected
    positions.

``MaskedLanguageModelTrainer``
    The loop, mirroring :class:`ContrastiveTrainer` step for step —
    AdamW with the same decay split, linear warmup then decay, gradient
    clipping, the same precision handling — differing only in objective.

Note the mask id is a *parameter* here, not a constant. The framework
vocabulary reserves ``PAD``/``UNK``/``BOS``/``EOS`` and no ``[MASK]``;
rather than migrate every trained model to widen that set, the trainer
takes the id to mask with and the ids to never corrupt as arguments. The
same structural-typing instinct that lets :class:`ContrastiveTrainer`
serve any tokenizer keeps this stage from hard-coding one vocabulary's
reserved ids.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from multilingual_embedding.core.exceptions import ValidationError
from multilingual_embedding.core.logging import get_logger
from multilingual_embedding.utils.filesystem import atomic_write_path, ensure_directory
from multilingual_embedding.utils.validation import require_positive

from .architecture import EncoderConfig, TransformerEncoderModel
from .encoder import NeuralTextEncoder, autocast_for
from .training import (
    decay_parameter_groups,
    differing_schedule_fields,
    schedule_signature,
)

__all__ = [
    "MaskedLanguageConfig",
    "MaskedLanguageModel",
    "MaskedLanguageModelHead",
    "MaskedLanguageModelTrainer",
    "PretrainingReport",
    "mask_tokens",
]

_logger = get_logger(__name__)

# The label a loss ignores. torch's cross-entropy skips positions carrying
# this target, and it is how supervision is confined to the masked tokens:
# every unselected position is labelled with it and contributes nothing.
IGNORE_INDEX = -100

_CHECKPOINT_FILENAME = "checkpoint.pt"

# Bumped when the checkpoint layout changes in a way that makes an older
# file unreadable. A run resumed from an incompatible checkpoint would
# restore the wrong state silently, so the version is checked, not trusted.
# v2 records the full schedule signature (see schedule_signature) rather
# than only documents and epochs.
_CHECKPOINT_VERSION = 2


class MaskedLanguageModelHead(nn.Module):
    """
    Projects per-token states onto a distribution over the vocabulary.

    The shape is the one BERT settled on: a dense transform with a GELU
    non-linearity and a layer norm, then a decoder to vocabulary width.
    The intermediate transform matters — without it the decoder is a bare
    linear read-out of the encoder's final states, which couples the
    representation the encoder wants to the prediction the head wants and
    hurts both.

    This is a training-time appendage. It is never persisted: once
    pretraining is done the encoder underneath is what gets saved, and the
    head is thrown away.
    """

    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()

        self.transform = nn.Linear(config.dimension, config.dimension)

        self.norm = nn.LayerNorm(config.dimension)

        self.decoder = nn.Linear(config.dimension, config.vocabulary_size)

        # Match the encoder's initialisation so the head starts on the same
        # scale as the states it reads. The encoder is already initialised
        # in its own constructor; only the head's fresh parameters are
        # touched here.
        self.apply(TransformerEncoderModel._initialise)

    def forward(self, hidden: Tensor) -> Tensor:
        """``(batch, length, dimension)`` states to ``(batch, length, vocab)`` logits."""

        transformed = self.norm(F.gelu(self.transform(hidden)))

        logits: Tensor = self.decoder(transformed)

        return logits


class MaskedLanguageModel(nn.Module):
    """
    An encoder with a masked-language head bolted on, for pretraining only.

    Composes rather than subclasses: the encoder is the object that
    survives, and wrapping it keeps every reference to its weights — the
    ones :class:`NeuralTextEncoder` and the optimizer already hold — valid
    throughout. :meth:`forward` returns vocabulary logits per position, not
    a pooled vector; the pooled contract is the encoder's own ``forward``.

    Parameters
    ----------
    encoder:
        The transformer to pretrain. Its ``hidden_states`` feed the head,
        and its ``token_embedding`` is what learns.

    tie_weights:
        Share the head's decoder matrix with the token-embedding table.
        Standard for MLM and on by default: the matrix mapping a vector
        *to* a token and the one mapping a token *to* a vector are
        transposes of the same relationship, so tying them halves the
        output parameters and ties the gradient that shapes the embedding
        directly to the prediction objective. Padding keeps its own zeroed
        row either way.
    """

    def __init__(
        self,
        encoder: TransformerEncoderModel,
        *,
        tie_weights: bool = True,
    ) -> None:
        super().__init__()

        self.encoder = encoder

        self.head = MaskedLanguageModelHead(encoder.config)

        if tie_weights:
            self.head.decoder.weight = self.encoder.token_embedding.weight

    @property
    def config(self) -> EncoderConfig:
        """The encoder's architecture."""

        return self.encoder.config

    def forward(self, token_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """``(batch, length)`` ids to ``(batch, length, vocab)`` logits."""

        hidden = self.encoder.hidden_states(token_ids, attention_mask)

        logits: Tensor = self.head(hidden)

        return logits


def mask_tokens(
    token_ids: Tensor,
    attention_mask: Tensor,
    *,
    mask_token_id: int,
    vocabulary_size: int,
    protected_ids: Collection[int],
    probability: float,
    generator: torch.Generator,
    replace_with_mask: float = 0.8,
    replace_with_random: float = 0.1,
) -> tuple[Tensor, Tensor]:
    """
    Corrupt a batch of ids for masked-language modelling.

    Selects ``probability`` of the real, non-protected tokens and returns
    the corrupted ids alongside labels that supervise only those
    positions. Of the selected tokens, ``replace_with_mask`` become the
    mask id, ``replace_with_random`` become a random token, and the rest
    are left unchanged.

    That last split is not decoration. If every selected token were
    replaced by ``[MASK]``, the model would only ever see the mask id at
    positions it must predict, and learn nothing about the tokens it sees
    at inference, where no mask exists. Leaving some intact and corrupting
    others to random tokens forces it to build a usable representation at
    *every* position, not only the masked ones.

    Parameters
    ----------
    token_ids, attention_mask:
        ``(batch, length)``. Padding — where the mask is 0 — is never
        selected, so it needs no entry in ``protected_ids``.

    mask_token_id:
        The id standing in for a hidden token.

    vocabulary_size:
        Upper bound for random replacements.

    protected_ids:
        Ids that are real tokens but must never be masked or become a
        label — ``BOS``/``EOS``/the mask id itself. Membership is checked
        against these, so a caller passes whatever its vocabulary reserves.

    probability:
        Share of eligible tokens to select. 0.15 is the usual value.

    generator:
        Seeds every random choice, so a run reproduces. Must live on the
        same device as ``token_ids``.

    Returns
    -------
    corrupted_ids, labels:
        Both ``(batch, length)``. ``labels`` carries the original id at
        selected positions and :data:`IGNORE_INDEX` everywhere else.
    """

    if not 0.0 < probability < 1.0:
        raise ValidationError(
            "mask probability must lie in (0, 1)",
            probability=probability,
        )

    device = token_ids.device

    eligible = attention_mask.bool()

    for token in protected_ids:
        eligible &= token_ids != token

    selection = torch.rand(token_ids.shape, generator=generator, device=device)

    selected = (selection < probability) & eligible

    labels = torch.where(
        selected,
        token_ids,
        torch.full_like(token_ids, IGNORE_INDEX),
    )

    # One draw decides each selected token's fate. Below `replace_with_mask`
    # it becomes the mask id; in the next `replace_with_random` band it
    # becomes a random token; above both it is left as it is — the model
    # still has to predict it, but sees the true token as input.
    fate = torch.rand(token_ids.shape, generator=generator, device=device)

    to_mask = selected & (fate < replace_with_mask)

    to_randomise = (
        selected & (fate >= replace_with_mask) & (fate < replace_with_mask + replace_with_random)
    )

    corrupted = token_ids.clone()

    corrupted[to_mask] = mask_token_id

    random_tokens = torch.randint(
        vocabulary_size,
        token_ids.shape,
        generator=generator,
        device=device,
        dtype=token_ids.dtype,
    )

    corrupted[to_randomise] = random_tokens[to_randomise]

    return corrupted, labels


@dataclass(slots=True)
class MaskedLanguageConfig:
    """
    Masked-language pretraining settings.

    Attributes
    ----------
    epochs:
        Passes over the corpus.

    batch_size:
        Documents per step. Unlike the contrastive batch this is a pure
        throughput knob — MLM has no in-batch negatives, so a larger batch
        trains faster but does not change the task.

    learning_rate:
        Peak rate, reached at the end of warmup and decayed after it.
        Higher than the contrastive default because a from-scratch table
        has further to travel; 1e-4 is a common starting point.

    warmup_ratio:
        Share of steps spent ramping the rate from zero. A random model's
        opening gradients are large, and warmup keeps the first steps from
        destabilising the table before it means anything.

    weight_decay:
        Applied to weight matrices only, via
        :func:`decay_parameter_groups`. Biases and norms are excluded.

    max_gradient_norm:
        Gradient clipping threshold.

    mask_probability:
        Share of eligible tokens hidden per sequence. 0.15 is standard —
        low enough to leave usable context, high enough to supervise
        densely.

    tie_weights:
        Share the output projection with the embedding table. See
        :class:`MaskedLanguageModel`.

    seed:
        Seeds shuffling, masking and dropout, so a run reproduces.

    precision:
        ``"fp32"`` or ``"bf16"``. Supplied by the machine, like the
        contrastive setting of the same name.
    """

    epochs: int = 3

    batch_size: int = 16

    learning_rate: float = 1e-4

    warmup_ratio: float = 0.1

    weight_decay: float = 0.01

    max_gradient_norm: float = 1.0

    mask_probability: float = 0.15

    tie_weights: bool = True

    seed: int = 42

    precision: str = "fp32"

    def __post_init__(self) -> None:
        require_positive(self.epochs, name="epochs")

        require_positive(self.batch_size, name="batch_size")

        require_positive(self.learning_rate, name="learning_rate")

        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ValidationError(
                "warmup_ratio must lie in [0, 1)",
                warmup_ratio=self.warmup_ratio,
            )

        if not 0.0 < self.mask_probability < 1.0:
            raise ValidationError(
                "mask_probability must lie in (0, 1)",
                mask_probability=self.mask_probability,
            )

        if self.precision not in {"fp32", "bf16"}:
            raise ValidationError(
                "Unsupported precision",
                precision=self.precision,
                supported=["bf16", "fp32"],
            )


@dataclass(slots=True)
class PretrainingReport:
    """
    What a pretraining run did.

    Attributes
    ----------
    epochs, steps, documents:
        Volume processed.

    initial_loss, final_loss:
        Mean cross-entropy of the first and last epoch.

    initial_accuracy, final_accuracy:
        Share of masked tokens predicted correctly, first and last epoch.
        This is the number that says whether the embedding learned
        anything: cross-entropy falling is necessary but easy to fake by
        predicting the frequent tokens, whereas masked accuracy climbing
        above the unigram-frequency baseline means the model is using
        context, which is the whole point.

    losses, accuracies:
        Per-epoch means.
    """

    epochs: int = 0

    steps: int = 0

    documents: int = 0

    initial_loss: float = 0.0

    final_loss: float = 0.0

    initial_accuracy: float = 0.0

    final_accuracy: float = 0.0

    losses: list[float] = field(default_factory=list)

    accuracies: list[float] = field(default_factory=list)

    @property
    def measurable(self) -> bool:
        """
        Whether :attr:`improved` means anything.

        As for contrastive training, a single epoch compares the first and
        last epoch — the same epoch — so the comparison is a value against
        itself and says nothing.
        """

        return len(self.losses) >= 2

    @property
    def improved(self) -> bool:
        """True when the final epoch's loss beat the first."""

        return self.measurable and self.final_loss < self.initial_loss

    def to_dict(self) -> dict[str, object]:
        """Reduce to primitives for reporting."""

        return {
            "epochs": self.epochs,
            "steps": self.steps,
            "documents": self.documents,
            "initial_loss": round(self.initial_loss, 6),
            "final_loss": round(self.final_loss, 6),
            "initial_accuracy": round(self.initial_accuracy, 6),
            "final_accuracy": round(self.final_accuracy, 6),
            "improved": self.improved,
            "measurable": self.measurable,
            "losses": [round(value, 6) for value in self.losses],
            "accuracies": [round(value, 6) for value in self.accuracies],
        }


class MaskedLanguageModelTrainer:
    """
    Pretrains a from-scratch encoder with a masked-language objective.

    Parameters
    ----------
    encoder:
        The encoder to pretrain. Its transformer is modified in place, so
        the same object is ready to hand to :class:`ContrastiveTrainer`
        afterwards — the two stages share one encoder.

    mask_token_id:
        The id to hide tokens with. The framework vocabulary has no
        ``[MASK]`` of its own, so the caller supplies one — typically an id
        reserved for the purpose, or ``UNK`` if none is.

    protected_ids:
        Real-token ids that must never be masked — ``BOS``/``EOS`` and the
        like. Padding is excluded automatically by the attention mask and
        needs no entry. The mask id is added to this set regardless.

    config:
        Training settings. Defaults are sensible for a toy run.

    Example
    -------
    ::

        trainer = MaskedLanguageModelTrainer(encoder, mask_token_id=1)

        report = trainer.train(documents)

        assert report.final_accuracy > report.initial_accuracy
    """

    __slots__ = ("_config", "_encoder", "_mask_token_id", "_model", "_protected")

    def __init__(
        self,
        encoder: NeuralTextEncoder,
        *,
        mask_token_id: int,
        protected_ids: Collection[int] = (),
        config: MaskedLanguageConfig | None = None,
    ) -> None:
        self._encoder = encoder

        self._config = config if config is not None else MaskedLanguageConfig()

        self._mask_token_id = mask_token_id

        self._protected = frozenset(protected_ids) | {mask_token_id}

        # Wrapping puts the head on the encoder's device; the encoder's own
        # weights are already there, and `.to` leaves them untouched.
        self._model = MaskedLanguageModel(
            encoder.model,
            tie_weights=self._config.tie_weights,
        ).to(encoder.device)

    @property
    def config(self) -> MaskedLanguageConfig:
        """The settings this trainer runs with."""

        return self._config

    def train(
        self,
        documents: Sequence[str],
        *,
        checkpoint_dir: str | Path | None = None,
        resume_from: str | Path | None = None,
        stop_after_epoch: int | None = None,
    ) -> PretrainingReport:
        """
        Pretrain the encoder and report what happened.

        A from-scratch run is long and a shared GPU box is not always
        yours for its whole length, so the loop is built to be
        interruptible. Pass ``checkpoint_dir`` to write a rolling
        checkpoint at every epoch boundary; pass ``resume_from`` to pick a
        run back up from one. A run resumed from its own checkpoints
        follows the same loss trajectory an uninterrupted run would have —
        the model, the optimizer, and every random stream are all
        restored, and the corpus ordering is a pure function of
        ``(seed, epoch)`` rather than of how far the run had got. The match
        is bit-for-bit on CPU; on CUDA it is as close as the backend's
        non-deterministic kernels allow, well within the noise of a
        different seed but not exact.

        Parameters
        ----------
        documents:
            The corpus. Must be non-empty and, when resuming, the same
            length as the run that wrote the checkpoint — the schedule is
            computed from it.

        checkpoint_dir:
            Where to write ``checkpoint.pt`` after each epoch. ``None``
            trains without checkpointing.

        resume_from:
            A checkpoint file, or a directory holding ``checkpoint.pt``, to
            restore before training. ``None`` starts fresh.

        stop_after_epoch:
            Stop after finishing this epoch index (0-based), modelling an
            interruption. The learning-rate schedule still spans the full
            ``config.epochs``, so a later resume continues it unbroken.

        Raises
        ------
        ValidationError
            If there are no documents, or a resumed checkpoint was written
            for a different schedule or an incompatible format.
        """

        if not documents:
            raise ValidationError(
                "masked-language pretraining needs at least one document",
                documents=len(documents),
            )

        config = self._config

        model = self._model.train()

        device = self._encoder.device

        vocabulary_size = self._model.config.vocabulary_size

        optimizer = torch.optim.AdamW(
            decay_parameter_groups(model, weight_decay=config.weight_decay),
            lr=config.learning_rate,
        )

        steps_per_epoch = max(1, math.ceil(len(documents) / config.batch_size))

        total_steps = steps_per_epoch * config.epochs

        warmup_steps = int(total_steps * config.warmup_ratio)

        precision = autocast_for(device, config.precision)

        if resume_from is not None:
            report, start_epoch, step, mask_generator = self._load_checkpoint(
                resume_from,
                model=model,
                optimizer=optimizer,
                documents=len(documents),
                device=device,
            )
        else:
            # A fresh run seeds every stream from the one configured seed:
            # the global RNG that dropout draws from, and a dedicated
            # generator that only masking draws from. Keeping masking on its
            # own stream is what lets a resumed run reproduce the corruption
            # exactly without having to also replay dropout.
            torch.manual_seed(config.seed)

            mask_generator = torch.Generator(device=device).manual_seed(config.seed)

            report = PretrainingReport(epochs=config.epochs, documents=len(documents))

            start_epoch = 0

            step = 0

        for epoch in range(start_epoch, config.epochs):
            mean_loss, accuracy, step = self._run_epoch(
                model=model,
                documents=documents,
                optimizer=optimizer,
                mask_generator=mask_generator,
                precision=precision,
                epoch=epoch,
                step=step,
                warmup_steps=warmup_steps,
                total_steps=total_steps,
                vocabulary_size=vocabulary_size,
            )

            report.losses.append(mean_loss)

            report.accuracies.append(accuracy)

            _logger.info(
                "Completed pretraining epoch",
                extra={
                    "epoch": epoch + 1,
                    "loss": round(mean_loss, 6),
                    "masked_accuracy": round(accuracy, 6),
                },
            )

            if checkpoint_dir is not None:
                self._save_checkpoint(
                    checkpoint_dir,
                    model=model,
                    optimizer=optimizer,
                    mask_generator=mask_generator,
                    report=report,
                    next_epoch=epoch + 1,
                    step=step,
                    documents=len(documents),
                )

            if stop_after_epoch is not None and epoch >= stop_after_epoch:
                break

        report.steps = step

        report.initial_loss = report.losses[0]

        report.final_loss = report.losses[-1]

        report.initial_accuracy = report.accuracies[0]

        report.final_accuracy = report.accuracies[-1]

        model.eval()

        return report

    def _run_epoch(
        self,
        *,
        model: MaskedLanguageModel,
        documents: Sequence[str],
        optimizer: torch.optim.Optimizer,
        mask_generator: torch.Generator,
        precision: Any,
        epoch: int,
        step: int,
        warmup_steps: int,
        total_steps: int,
        vocabulary_size: int,
    ) -> tuple[float, float, int]:
        """Run one epoch and return ``(mean_loss, masked_accuracy, next_step)``."""

        config = self._config

        order = self._epoch_order(epoch, len(documents))

        batch_size = config.batch_size

        epoch_losses: list[float] = []

        correct = 0

        counted = 0

        for start in range(0, len(order), batch_size):
            batch = [documents[index] for index in order[start : start + batch_size]]

            # internal to the encoder; the trainer is its only other caller.
            ids, mask = self._encoder._prepare(batch)

            corrupted, labels = mask_tokens(
                ids,
                mask,
                mask_token_id=self._mask_token_id,
                vocabulary_size=vocabulary_size,
                protected_ids=self._protected,
                probability=config.mask_probability,
                generator=mask_generator,
            )

            supervised = int((labels != IGNORE_INDEX).sum())

            # A batch that happened to select nothing — possible when a
            # tiny batch is nearly all padding — has no gradient. Skip
            # it rather than divide a zero loss by a zero count.
            if supervised == 0:
                continue

            optimizer.zero_grad(set_to_none=True)

            with precision:
                logits = model(corrupted, mask)

                loss = F.cross_entropy(
                    logits.reshape(-1, vocabulary_size),
                    labels.reshape(-1),
                    ignore_index=IGNORE_INDEX,
                )

            # torch's Tensor.backward carries no annotations, so a
            # strict checker sees an untyped call in typed context.
            loss.backward()  # type: ignore[no-untyped-call]

            nn.utils.clip_grad_norm_(model.parameters(), config.max_gradient_norm)

            for group in optimizer.param_groups:
                group["lr"] = self._learning_rate(step, warmup_steps, total_steps)

            optimizer.step()

            epoch_losses.append(float(loss.detach()))

            correct += self._masked_correct(logits.detach(), labels)

            counted += supervised

            step += 1

        mean_loss = sum(epoch_losses) / max(1, len(epoch_losses))

        accuracy = correct / max(1, counted)

        return mean_loss, accuracy, step

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _masked_correct(logits: Tensor, labels: Tensor) -> int:
        """Count masked positions whose top prediction is the true token."""

        predictions = logits.argmax(dim=-1)

        supervised = labels != IGNORE_INDEX

        return int(((predictions == labels) & supervised).sum())

    def _epoch_order(self, epoch: int, count: int) -> list[int]:
        """
        The corpus order for one epoch — a pure function of seed and epoch.

        Deliberately independent of how far the run has progressed: an
        uninterrupted run and one resumed mid-way must both visit epoch *k*
        in the identical order, so the order cannot be drawn from a running
        stream whose position depends on the number of prior steps. Seeding
        a fresh generator per epoch makes it reconstructible from the
        checkpoint's ``epoch`` alone, so the ordering never has to be saved.

        There is no de-duplication and no minimum batch size: MLM has no
        in-batch negatives, so a repeated or lone document is a perfectly
        good example — every batch teaches the same per-token task.
        """

        # randperm runs on the CPU, so the generator is a CPU one regardless
        # of where the model lives. The multiplier is a large prime, so
        # neighbouring epochs get unrelated permutations rather than seeds
        # one apart.
        generator = torch.Generator().manual_seed(self._config.seed * 1_000_003 + epoch)

        return torch.randperm(count, generator=generator).tolist()

    def _save_checkpoint(
        self,
        checkpoint_dir: str | Path,
        *,
        model: MaskedLanguageModel,
        optimizer: torch.optim.Optimizer,
        mask_generator: torch.Generator,
        report: PretrainingReport,
        next_epoch: int,
        step: int,
        documents: int,
    ) -> None:
        """
        Write a rolling checkpoint holding everything a resume needs.

        One file, replaced atomically each epoch: the newest checkpoint is
        the only one worth keeping, and a half-written one would be loaded
        as gospel by the next run, so it is renamed into place rather than
        streamed over the live file. The saved state is the full run — the
        model and optimizer, every random stream, the step counter that
        drives the learning-rate schedule, and the metrics so far — plus
        the schedule the checkpoint belongs to, so a mismatched resume is
        refused rather than silently misaligned.
        """

        target = ensure_directory(checkpoint_dir) / _CHECKPOINT_FILENAME

        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

        payload = {
            "format_version": _CHECKPOINT_VERSION,
            "epoch": next_epoch,
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": cuda_rng,
            "mask_rng": mask_generator.get_state(),
            "losses": list(report.losses),
            "accuracies": list(report.accuracies),
            "schedule": schedule_signature(
                count=documents,
                epochs=self._config.epochs,
                batch_size=self._config.batch_size,
                warmup_ratio=self._config.warmup_ratio,
                learning_rate=self._config.learning_rate,
            ),
        }

        with atomic_write_path(target) as temporary:
            torch.save(payload, temporary)

    def _load_checkpoint(
        self,
        resume_from: str | Path,
        *,
        model: MaskedLanguageModel,
        optimizer: torch.optim.Optimizer,
        documents: int,
        device: torch.device,
    ) -> tuple[PretrainingReport, int, int, torch.Generator]:
        """
        Restore a run from a checkpoint and return where to carry on from.

        Returns the report seeded with the metrics so far, the epoch to
        resume at, the step the learning-rate schedule had reached, and the
        masking generator restored to its saved position.

        Raises
        ------
        ValidationError
            If the checkpoint's format is unrecognised, or it was written
            under a different schedule — a changed corpus size, epoch count,
            batch size, warmup ratio or learning rate — any of which would
            resume onto a curve the saved step no longer matches. See
            :func:`schedule_signature`.
        """

        path = Path(resume_from).expanduser()

        if path.is_dir():
            path = path / _CHECKPOINT_FILENAME

        # A local file this trainer wrote itself, so full unpickling is
        # safe and necessary — the payload holds optimizer and RNG state,
        # not just weights.
        checkpoint: dict[str, Any] = torch.load(path, weights_only=False)

        version = checkpoint.get("format_version")

        if version != _CHECKPOINT_VERSION:
            raise ValidationError(
                "checkpoint format is not readable by this version",
                found=version,
                expected=_CHECKPOINT_VERSION,
            )

        current_schedule = schedule_signature(
            count=documents,
            epochs=self._config.epochs,
            batch_size=self._config.batch_size,
            warmup_ratio=self._config.warmup_ratio,
            learning_rate=self._config.learning_rate,
        )

        drift = differing_schedule_fields(checkpoint.get("schedule", {}), current_schedule)

        if drift:
            raise ValidationError(
                "checkpoint was written for a different training schedule",
                differing=", ".join(drift),
                checkpoint_schedule=checkpoint.get("schedule"),
                schedule=current_schedule,
            )

        model.load_state_dict(checkpoint["model"])

        optimizer.load_state_dict(checkpoint["optimizer"])

        # Restore both random streams to exactly where the saved epoch left
        # them, so the continuation draws the same masks and the same
        # dropout an uninterrupted run would have.
        torch.set_rng_state(checkpoint["torch_rng"])

        if checkpoint["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])

        mask_generator = torch.Generator(device=device)

        mask_generator.set_state(checkpoint["mask_rng"])

        report = PretrainingReport(epochs=self._config.epochs, documents=documents)

        report.losses = list(checkpoint["losses"])

        report.accuracies = list(checkpoint["accuracies"])

        return report, int(checkpoint["epoch"]), int(checkpoint["step"]), mask_generator

    def _learning_rate(self, step: int, warmup_steps: int, total_steps: int) -> float:
        """Linear warmup followed by linear decay, as in contrastive training."""

        peak = self._config.learning_rate

        if warmup_steps and step < warmup_steps:
            return peak * (step + 1) / warmup_steps

        remaining = max(0, total_steps - step)

        span = max(1, total_steps - warmup_steps)

        return peak * remaining / span
