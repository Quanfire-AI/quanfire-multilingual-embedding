"""
Contrastive training.

The objective is InfoNCE over in-batch negatives. Each example is a pair
of texts that should encode close together — a question and the passage
answering it, a heading and its section, a sentence and its translation.
Every *other* passage in the batch serves as a negative for that query.

Two consequences follow, and both shape the code below.

**Batch size is a quality parameter, not just a speed one.** A batch of
16 asks the model to pick the right passage from 16 candidates; a batch
of 1024 makes it pick from 1024, which is a far harder and more useful
task. This is why contrastive training is memory-hungry in a way that
supervised training is not, and why gradient caching exists.

**Duplicate texts within a batch are poison.** If the same passage
appears twice, the loss punishes the model for matching a correct answer,
because that copy is labelled a negative. The sampler here de-duplicates
for exactly that reason, and :func:`_candidates` applies the same rule
where mined negatives meet the batch's positives.

A pair may also carry **mined hard negatives** — passages some model
ranks near the right answer and that are nonetheless wrong, produced by
:func:`multilingual_embedding.embedding.negatives.mine_negatives`. They
join the candidate columns to the right of the positives, shared across
every anchor in the batch, and the target labels stay the diagonal. A
pair set without them trains exactly as it did before: the in-batch
objective is the case where the extra columns number zero.

The temperature scales the logits before the softmax. Lower values
sharpen the distribution and push harder on the hardest negatives; 0.05
is the usual starting point for sentence encoders and the default here.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from multilingual_embedding.config.base import ComputeConfig
from multilingual_embedding.core.exceptions import ValidationError
from multilingual_embedding.core.logging import get_logger
from multilingual_embedding.utils.filesystem import atomic_write_path, ensure_directory
from multilingual_embedding.utils.validation import require_non_negative, require_positive

from .encoder import autocast_for
from .gradcache import cached_contrastive_backward

__all__ = [
    "ContrastiveConfig",
    "ContrastiveTrainer",
    "TextPair",
    "Trainable",
    "TrainingReport",
    "decay_parameter_groups",
    "differing_schedule_fields",
    "schedule_signature",
]

_logger = get_logger(__name__)

_CHECKPOINT_FILENAME = "checkpoint.pt"

# Bumped when the checkpoint layout changes in a way that makes an older
# file unreadable. A run resumed from an incompatible checkpoint would
# restore the wrong state silently, so the version is checked, not trusted.
# Kept independent of pretraining's version: the two stages checkpoint
# different state and one's format moving should not invalidate the other's.
# v2 records the full schedule signature (see schedule_signature) rather
# than only pairs and epochs.
_CHECKPOINT_VERSION = 2


def decay_parameter_groups(
    model: nn.Module,
    *,
    weight_decay: float,
) -> list[dict[str, object]]:
    """
    Split parameters into decayed and undecayed groups.

    Biases and normalisation parameters are excluded from weight decay.
    Decaying a LayerNorm gain drives it toward zero, which scales down the
    layer's entire output. This is shared by every trainer in the package,
    because the rule is a property of the architecture, not of an objective.
    """

    decayed: list[nn.Parameter] = []

    undecayed: list[nn.Parameter] = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        if parameter.ndim < 2 or name.endswith(".bias"):
            undecayed.append(parameter)
        else:
            decayed.append(parameter)

    return [
        {"params": decayed, "weight_decay": weight_decay},
        {"params": undecayed, "weight_decay": 0.0},
    ]


def schedule_signature(
    *,
    count: int,
    epochs: int,
    batch_size: int,
    warmup_ratio: float,
    learning_rate: float,
) -> dict[str, Any]:
    """
    The knobs that decide *where in the schedule* a resumed step lands.

    A checkpoint records the optimizer step it reached, not the shape of
    the schedule that produced it. Resume the same step under a different
    schedule and the learning rate at that step silently changes: the total
    step count is ``ceil(count / batch_size) * epochs``, the warmup boundary
    is ``warmup_ratio`` of it, and the peak is ``learning_rate`` — so a
    smaller batch, an extra epoch, a different warmup or peak all move the
    curve out from under the step the checkpoint restored. None of it
    raises; the run just anneals along a curve it was never training on.

    This is the signature the two ends compare. It is deliberately the set
    that changes the curve and nothing cosmetic, so a resume is refused
    only when it would actually diverge, not whenever the config was
    touched. Shared by both trainers because the schedule maths is.
    """

    return {
        "count": count,
        "epochs": epochs,
        "batch_size": batch_size,
        "warmup_ratio": warmup_ratio,
        "learning_rate": learning_rate,
    }


def differing_schedule_fields(
    saved: Mapping[str, Any],
    current: Mapping[str, Any],
) -> list[str]:
    """
    Which schedule knobs drift between a checkpoint and the run resuming it.

    Integer knobs (``count``, ``epochs``, ``batch_size``) are compared
    exactly. The float knobs (``warmup_ratio``, ``learning_rate``) go
    through :func:`math.isclose`, so a value that round-tripped through the
    checkpoint's serialisation and came back a bit shifted does not read as
    a changed schedule and refuse a resume that is in fact identical.

    Returns the drifting field names in a stable order, empty when the two
    schedules agree. A missing key on either side counts as a difference,
    which is what makes a v1 checkpoint (no signature at all) fail the
    comparison rather than slip through it.
    """

    exact = ("count", "epochs", "batch_size")

    approximate = ("warmup_ratio", "learning_rate")

    differing: list[str] = []

    for name in exact:
        if saved.get(name) != current.get(name):
            differing.append(name)

    for name in approximate:
        left = saved.get(name)
        right = current.get(name)

        # A missing value on either side counts as drift (a v1 checkpoint
        # carries no signature at all); short-circuiting also keeps the
        # None cases away from the float() and isclose() below.
        if (
            left is None
            or right is None
            or not math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-12)
        ):
            differing.append(name)

    return differing


class Trainable(Protocol):
    """
    What this trainer needs from an encoder, and nothing more.

    Two unrelated classes satisfy it: :class:`NeuralTextEncoder`, wrapping
    a model this project trained from scratch, and
    :class:`PretrainedTextEncoder`, wrapping a published checkpoint from
    ``transformers``. Neither inherits from the other and neither should —
    a pre-norm architecture and a post-norm one have matching tensor
    shapes, so a shared base class would make cross-loading their weights
    *succeed* and be silently wrong.

    Naming the requirement structurally instead is what lets one training
    loop serve both without either knowing about the other. The
    underscored member is here because it is genuinely part of the
    contract between an encoder and its trainer, and pretending otherwise
    by widening the annotation to a union would be less honest, not more.
    """

    @property
    def device(self) -> torch.device:
        """Where the model's tensors live."""

    def train_mode(self) -> nn.Module:
        """The underlying module, switched into training mode."""

    def _prepare(self, texts: Sequence[str]) -> tuple[Tensor, Tensor]:
        """Tokenise to input ids and an attention mask, on the encoder's device."""


@dataclass(slots=True, frozen=True)
class TextPair:
    """
    Two texts that should encode close together.

    Attributes
    ----------
    anchor:
        The query side — a question, a title, a short form.

    positive:
        The passage side — the answer, the body, the long form.

    negatives:
        Passages this anchor must *not* match, mined against a model by
        :func:`multilingual_embedding.embedding.negatives.mine_negatives`.
        Empty by default, in which case the step is exactly the in-batch
        objective it always was.

        Each distinct negative in a batch becomes one more candidate
        column, shared by every anchor in that batch — the negative
        mined for one query is a perfectly good negative for the other
        fifteen, and it is already encoded. So ``k`` negatives per pair
        widen the candidate set by up to ``k x batch_size`` while
        costing one forward pass over those texts, and the memory a step
        needs grows with them.
    """

    anchor: str

    positive: str

    negatives: tuple[str, ...] = ()


def _candidates(batch: Sequence[TextPair]) -> list[str]:
    """
    The candidate column texts for a batch: positives first, then mined
    negatives that are not already among them.

    Order is the contract. Positive ``i`` must land at column ``i`` so
    that the cross-entropy target stays the diagonal; the mined negatives
    extend the columns to the right, where they are negatives for every
    anchor at once.

    The de-duplication is the same rule the sampler applies to positives
    and it is here for the same reason. A mined negative that is
    textually a positive already in the batch — its own pair's, or
    another anchor's — would appear as two columns, one labelled correct
    and one labelled wrong. The loss would then punish the model for
    matching a correct answer, which is the single mistake hard-negative
    mining is most likely to introduce. Collapsing it to one column keeps
    the text in the candidate set exactly once: right for its own anchor,
    wrong for everyone else, which is what it is.
    """

    seen = {pair.positive for pair in batch}

    extra: list[str] = []

    for pair in batch:
        for negative in pair.negatives:
            if negative in seen:
                continue

            seen.add(negative)

            extra.append(negative)

    return [pair.positive for pair in batch] + extra


@dataclass(slots=True)
class ContrastiveConfig:
    """
    Contrastive training settings.

    Attributes
    ----------
    epochs:
        Passes over the pair set.

    batch_size:
        Pairs per step. Doubles as the number of negatives each query is
        contrasted against, so larger is materially better until memory
        runs out.

    learning_rate:
        Peak rate, reached at the end of warmup and decayed after it.

    warmup_ratio:
        Share of total steps spent linearly increasing the rate from zero.
        Contrastive training is unstable in its opening steps without it,
        because the encoder's outputs are near-random and the loss
        gradient is correspondingly large.

    temperature:
        Divides the similarity logits. Lower sharpens the distribution.

    weight_decay:
        Applied to weight matrices only. Biases and normalisation
        parameters are excluded, which is standard and matters more than
        it sounds: decaying a LayerNorm gain pulls it toward zero and
        suppresses the layer.

    max_gradient_norm:
        Gradient clipping threshold.

    seed:
        Seeds shuffling and dropout, so a run reproduces.

    precision:
        ``"fp32"`` or ``"bf16"``. Machine-shaped rather than
        experiment-shaped: it belongs to whichever box the run lands on,
        which is why it is normally supplied by a compute profile rather
        than written into an experiment by hand.

    gradient_checkpoint_chunk:
        Examples encoded at once under gradient caching. ``0`` disables
        it and encodes each batch in one pass. When set, peak memory
        follows the chunk rather than the batch, which is what makes a
        batch size the card could not otherwise hold reachable. The
        gradients are identical either way.
    """

    epochs: int = 3

    batch_size: int = 16

    learning_rate: float = 2e-5

    warmup_ratio: float = 0.1

    temperature: float = 0.05

    weight_decay: float = 0.01

    max_gradient_norm: float = 1.0

    seed: int = 42

    precision: str = "fp32"

    gradient_checkpoint_chunk: int = 0

    def __post_init__(self) -> None:
        require_positive(self.epochs, name="epochs")

        require_positive(self.batch_size, name="batch_size")

        require_positive(self.learning_rate, name="learning_rate")

        require_positive(self.temperature, name="temperature")

        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ValidationError(
                "warmup_ratio must lie in [0, 1)",
                warmup_ratio=self.warmup_ratio,
            )

        require_non_negative(
            self.gradient_checkpoint_chunk,
            name="gradient_checkpoint_chunk",
        )

        # Checked here as well as in `autocast_for`, so a typo in a
        # profile fails when the config is built rather than after the
        # model has been placed on the device.
        if self.precision not in {"fp32", "bf16"}:
            raise ValidationError(
                "Unsupported precision",
                precision=self.precision,
                supported=["bf16", "fp32"],
            )

    @classmethod
    def for_compute(
        cls,
        compute: ComputeConfig,
        **settings: Any,
    ) -> ContrastiveConfig:
        """
        Build a training config from an experiment plus a machine.

        This is the join between the two halves of a configuration. The
        keyword arguments carry the experiment — epochs, learning rate,
        temperature, the things a result depends on — while ``compute``
        supplies what the machine dictates. Running the same experiment
        on another box means changing only the second argument.

        ``batch_size`` is the one setting that appears in both and is
        taken from ``compute``, because memory decides it. It does change
        the result, since it sets how many negatives each query is
        contrasted against, which is why a report records it.

        Example
        -------
        ::

            config = ContrastiveConfig.for_compute(
                experiment.compute, epochs=3, learning_rate=2e-5
            )
        """

        return cls(
            batch_size=compute.batch_size,
            precision=compute.precision,
            gradient_checkpoint_chunk=compute.gradient_checkpoint_chunk,
            **settings,
        )


@dataclass(slots=True)
class TrainingReport:
    """
    What a training run did.

    Attributes
    ----------
    epochs, steps, pairs:
        Volume processed.

    initial_loss, final_loss:
        Mean loss of the first and last epoch. The pair of them is the
        evidence that training did something.

    losses:
        Mean loss per epoch.
    """

    epochs: int = 0

    steps: int = 0

    pairs: int = 0

    initial_loss: float = 0.0

    final_loss: float = 0.0

    losses: list[float] = field(default_factory=list)

    @property
    def measurable(self) -> bool:
        """
        Whether ``improved`` means anything for this run.

        It does not for a single epoch: the first and last epoch are the
        same epoch, so the comparison is a value against itself. A real
        adaptation run reported ``loss: [1.17487, 1.17487]`` — identical
        to sixteen digits — and ``improved: False``, while its retrieval
        score rose 20%. The loss was not evidence of anything; it was the
        same number printed twice.
        """

        return len(self.losses) >= 2

    @property
    def improved(self) -> bool:
        """
        True when the final epoch's loss beat the first.

        Always ``False`` for a single-epoch run, whatever happened. Check
        :attr:`measurable` before reading this, and prefer a retrieval
        score to a loss for deciding whether training helped — a falling
        loss is compatible with having learned nothing useful.
        """

        return self.measurable and self.final_loss < self.initial_loss

    def to_dict(self) -> dict[str, object]:
        """Reduce to primitives for reporting."""

        return {
            "epochs": self.epochs,
            "steps": self.steps,
            "pairs": self.pairs,
            "initial_loss": round(self.initial_loss, 6),
            "final_loss": round(self.final_loss, 6),
            "improved": self.improved,
            "measurable": self.measurable,
            "losses": [round(value, 6) for value in self.losses],
        }


class ContrastiveTrainer:
    """
    Trains an encoder on text pairs with an InfoNCE objective.

    Parameters
    ----------
    encoder:
        The encoder to fit. Modified in place.

    config:
        Training settings.

    Example
    -------
    ::

        trainer = ContrastiveTrainer(encoder, ContrastiveConfig(epochs=5))

        report = trainer.train(pairs)

        assert report.improved
    """

    __slots__ = ("_config", "_encoder")

    def __init__(
        self,
        encoder: Trainable,
        config: ContrastiveConfig | None = None,
    ) -> None:
        self._encoder = encoder

        self._config = config if config is not None else ContrastiveConfig()

    @property
    def config(self) -> ContrastiveConfig:
        """The settings this trainer runs with."""

        return self._config

    def train(
        self,
        pairs: Sequence[TextPair],
        *,
        checkpoint_dir: str | Path | None = None,
        resume_from: str | Path | None = None,
        stop_after_epoch: int | None = None,
    ) -> TrainingReport:
        """
        Fit the encoder and report what happened.

        Stage two of the from-scratch path is long and a shared GPU box is
        not always yours for its whole length, so the loop is built to be
        interruptible, exactly as masked-language pretraining is. Pass
        ``checkpoint_dir`` to write a rolling checkpoint at every epoch
        boundary; pass ``resume_from`` to pick a run back up from one. A run
        resumed from its own checkpoints follows the same loss trajectory an
        uninterrupted run would have — the model, the optimizer, the dropout
        stream, and the shuffle stream are all restored, and the
        learning-rate schedule continues from the saved step. The match is
        bit-for-bit on CPU; on CUDA it is as close as the backend's
        non-deterministic kernels allow, which is well within the noise of a
        different seed but is not exact reproduction. Enabling
        ``torch.use_deterministic_algorithms`` would close that gap at a
        throughput cost and is left as a future opt-in rather than forced on
        every run.

        Parameters
        ----------
        pairs:
            The training pairs. When resuming, must be the same length as
            the run that wrote the checkpoint — the schedule is computed
            from it, so a different count would misalign the saved step.

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
            If there are too few pairs to form a batch with at least one
            negative in it. A batch of one whose pair carries no mined
            negatives has nothing to contrast against, so its loss is
            identically zero — which looks like success and teaches
            nothing. A single pair *with* mined negatives is a real
            training example and is allowed. Also raised if a resumed
            checkpoint was written for a different schedule or an
            incompatible format.
        """

        if len(pairs) < 2 and not any(pair.negatives for pair in pairs):
            raise ValidationError(
                "contrastive training needs at least two pairs, or one pair "
                "carrying mined negatives, since a batch of one otherwise "
                "has nothing to contrast against",
                pairs=len(pairs),
            )

        config = self._config

        model = self._encoder.train_mode()

        device = self._encoder.device

        optimizer = torch.optim.AdamW(
            self._parameter_groups(model),
            lr=config.learning_rate,
        )

        steps_per_epoch = max(1, math.ceil(len(pairs) / config.batch_size))

        total_steps = steps_per_epoch * config.epochs

        warmup_steps = int(total_steps * config.warmup_ratio)

        # Resolved once rather than per step. It also emits a warning when
        # the request cannot be honoured, which should be said once.
        precision = autocast_for(device, config.precision)

        if resume_from is not None:
            report, start_epoch, step, generator = self._load_checkpoint(
                resume_from,
                model=model,
                optimizer=optimizer,
                pairs=len(pairs),
                device=device,
            )
        else:
            # A fresh run seeds both streams from the one configured seed:
            # the global RNG dropout draws from, and a dedicated generator
            # the per-epoch shuffle draws from. Keeping the shuffle on its
            # own generator is what lets a resumed run restore the ordering
            # to exactly where it left off without disturbing dropout.
            torch.manual_seed(config.seed)

            generator = torch.Generator().manual_seed(config.seed)

            report = TrainingReport(epochs=config.epochs, pairs=len(pairs))

            start_epoch = 0

            step = 0

        for epoch in range(start_epoch, config.epochs):
            epoch_losses: list[float] = []

            for batch in self._batches(pairs, generator):
                # Zeroed before the step rather than after the forward,
                # because gradient caching backpropagates internally and
                # accumulates into whatever is already there.
                optimizer.zero_grad(set_to_none=True)

                if config.gradient_checkpoint_chunk:
                    loss = self._cached_step(
                        model,
                        batch,
                        device,
                        precision,
                        config.gradient_checkpoint_chunk,
                    )
                else:
                    # The forward pass runs under autocast; the backward
                    # pass deliberately does not. Autograd replays each
                    # operation in the dtype autocast chose for it, so
                    # wrapping the backward would be redundant at best,
                    # and torch documents it as unsupported.
                    with precision:
                        loss = self._step(model, batch, device)

                    # torch's Tensor.backward carries no annotations, so a
                    # strict checker sees an untyped call in typed context.
                    loss.backward()  # type: ignore[no-untyped-call]

                nn.utils.clip_grad_norm_(model.parameters(), config.max_gradient_norm)

                for group in optimizer.param_groups:
                    group["lr"] = self._learning_rate(step, warmup_steps, total_steps)

                optimizer.step()

                epoch_losses.append(float(loss.detach()))

                step += 1

            mean_loss = sum(epoch_losses) / max(1, len(epoch_losses))

            report.losses.append(mean_loss)

            _logger.info(
                "Completed contrastive epoch",
                extra={"epoch": epoch + 1, "loss": round(mean_loss, 6)},
            )

            if checkpoint_dir is not None:
                self._save_checkpoint(
                    checkpoint_dir,
                    model=model,
                    optimizer=optimizer,
                    generator=generator,
                    report=report,
                    next_epoch=epoch + 1,
                    step=step,
                    pairs=len(pairs),
                )

            if stop_after_epoch is not None and epoch >= stop_after_epoch:
                break

        report.steps = step

        report.initial_loss = report.losses[0]

        report.final_loss = report.losses[-1]

        model.eval()

        return report

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _save_checkpoint(
        self,
        checkpoint_dir: str | Path,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        generator: torch.Generator,
        report: TrainingReport,
        next_epoch: int,
        step: int,
        pairs: int,
    ) -> None:
        """
        Write a rolling checkpoint holding everything a resume needs.

        One file, replaced atomically each epoch: the newest checkpoint is
        the only one worth keeping, and a half-written one would be loaded
        as gospel by the next run, so it is renamed into place rather than
        streamed over the live file. The saved state is the full run — the
        model and optimizer, the dropout and shuffle streams, the step
        counter that drives the learning-rate schedule, and the losses so
        far — plus the schedule the checkpoint belongs to, so a mismatched
        resume is refused rather than silently misaligned.
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
            "shuffle_rng": generator.get_state(),
            "losses": list(report.losses),
            "schedule": schedule_signature(
                count=pairs,
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
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        pairs: int,
        device: torch.device,
    ) -> tuple[TrainingReport, int, int, torch.Generator]:
        """
        Restore a run from a checkpoint and return where to carry on from.

        Returns the report seeded with the losses so far, the epoch to
        resume at, the step the learning-rate schedule had reached, and the
        shuffle generator restored to its saved position.

        Raises
        ------
        ValidationError
            If the checkpoint's format is unrecognised, or it was written
            under a different schedule — a changed pair count, epoch count,
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
            count=pairs,
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

        # Restore both streams to exactly where the saved epoch left them,
        # so the continuation draws the same dropout and shuffles the pairs
        # into the same order an uninterrupted run would have.
        torch.set_rng_state(checkpoint["torch_rng"])

        if checkpoint["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])

        generator = torch.Generator()

        generator.set_state(checkpoint["shuffle_rng"])

        report = TrainingReport(epochs=self._config.epochs, pairs=pairs)

        report.losses = list(checkpoint["losses"])

        return report, int(checkpoint["epoch"]), int(checkpoint["step"]), generator

    def _step(
        self,
        model: nn.Module,
        batch: Sequence[TextPair],
        device: torch.device,
    ) -> Tensor:
        """
        One InfoNCE step over a batch.

        Anchors and candidates are encoded together in a single forward
        pass rather than two. Beyond halving the launch overhead, it
        guarantees both sides see identical dropout conditions, which they
        would not if encoded separately.
        """

        anchors = [pair.anchor for pair in batch]

        candidates = _candidates(batch)

        # internal to the encoder; the trainer is its only other caller.
        ids, mask = self._encoder._prepare([*anchors, *candidates])

        vectors = model(ids, mask)

        vectors = F.normalize(vectors, dim=-1)

        anchor_vectors = vectors[: len(batch)]

        candidate_vectors = vectors[len(batch) :]

        logits = anchor_vectors @ candidate_vectors.T / self._config.temperature

        # The correct passage for anchor i is candidate i — _candidates
        # puts the positives first, in batch order, for exactly this — so
        # the target labels are the diagonal whether or not mined
        # negatives extended the columns to the right of it.
        targets = torch.arange(len(batch), device=device)

        return F.cross_entropy(logits, targets)

    def _cached_step(
        self,
        model: nn.Module,
        batch: Sequence[TextPair],
        device: torch.device,
        precision: AbstractContextManager[None],
        chunk_size: int,
    ) -> Tensor:
        """
        One InfoNCE step whose batch is larger than memory allows.

        Identical in result to :meth:`_step`, and it backpropagates
        rather than returning a tensor to backpropagate — gradient
        caching has to own the backward pass to do its job.

        Anchors and candidates are laid out as one list, anchors first,
        so that chunk boundaries are ordinary slices and the stacked
        vectors split back apart at ``len(batch)``.

        Normalisation happens inside the loss rather than inside the
        encode, because gradient caching caches raw model outputs and
        takes the loss gradient with respect to those. Normalising early
        would put the operation on the wrong side of the cache.
        """

        texts = [pair.anchor for pair in batch] + _candidates(batch)

        def encode(chunk: Sequence[int]) -> Tensor:
            # internal to the encoder; the trainer is its only other caller.
            ids, mask = self._encoder._prepare([texts[index] for index in chunk])

            with precision:
                # nn.Module.__call__ is untyped, so the result widens to Any.
                vectors: Tensor = model(ids, mask)

            return vectors

        def loss_fn(vectors: Tensor) -> Tensor:
            normalised = F.normalize(vectors, dim=-1)

            anchor_vectors = normalised[: len(batch)]

            candidate_vectors = normalised[len(batch) :]

            logits = anchor_vectors @ candidate_vectors.T / self._config.temperature

            return F.cross_entropy(logits, torch.arange(len(batch), device=device))

        return cached_contrastive_backward(
            model,
            range(len(texts)),
            encode,
            loss_fn,
            chunk_size=chunk_size,
        )

    def _batches(
        self,
        pairs: Sequence[TextPair],
        generator: torch.Generator,
    ) -> Iterator[list[TextPair]]:
        """
        Yield shuffled batches, dropping duplicate positives.

        A duplicated positive within a batch is labelled a negative for
        the other anchor that shares it, so the loss actively penalises a
        correct match. Dropping the duplicate costs one example and avoids
        training against a contradiction.

        A trailing batch of one is dropped *unless* its pair carries
        mined negatives: without them it has nothing to contrast
        against, so its loss is zero and its gradient is nothing; with
        them it is an ordinary training example that happens to be
        alone.
        """

        order = torch.randperm(len(pairs), generator=generator).tolist()

        batch: list[TextPair] = []

        seen: set[str] = set()

        for index in order:
            pair = pairs[index]

            if pair.positive in seen:
                continue

            batch.append(pair)

            seen.add(pair.positive)

            if len(batch) == self._config.batch_size:
                yield batch

                batch = []

                seen = set()

        if len(batch) > 1 or any(pair.negatives for pair in batch):
            yield batch

    def _parameter_groups(self, model: nn.Module) -> list[dict[str, object]]:
        """Decayed and undecayed groups; see :func:`decay_parameter_groups`.

        The decay is the one the config carries, not a constant: a
        ``ContrastiveConfig.weight_decay`` a caller set would otherwise be
        silently ignored here while the pretraining trainer honoured its
        own, so the same knob would mean different things on the two halves
        of the from-scratch path.
        """

        return decay_parameter_groups(model, weight_decay=self._config.weight_decay)

    def _learning_rate(self, step: int, warmup_steps: int, total_steps: int) -> float:
        """Linear warmup followed by linear decay."""

        peak = self._config.learning_rate

        if warmup_steps and step < warmup_steps:
            return peak * (step + 1) / warmup_steps

        remaining = max(0, total_steps - step)

        span = max(1, total_steps - warmup_steps)

        return peak * remaining / span
