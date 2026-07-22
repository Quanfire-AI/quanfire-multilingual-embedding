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
for exactly that reason.

The temperature scales the logits before the softmax. Lower values
sharpen the distribution and push harder on the hardest negatives; 0.05
is the usual starting point for sentence encoders and the default here.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any, Protocol

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from multilingual_embedding.config.base import ComputeConfig
from multilingual_embedding.core.exceptions import ValidationError
from multilingual_embedding.core.logging import get_logger
from multilingual_embedding.utils.validation import require_non_negative, require_positive

from .encoder import autocast_for
from .gradcache import cached_contrastive_backward

__all__ = [
    "ContrastiveConfig",
    "ContrastiveTrainer",
    "TextPair",
    "Trainable",
    "TrainingReport",
]

_logger = get_logger(__name__)


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
    """

    anchor: str

    positive: str


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

    def train(self, pairs: Sequence[TextPair]) -> TrainingReport:
        """
        Fit the encoder and report what happened.

        Raises
        ------
        ValidationError
            If there are too few pairs to form a batch with at least one
            negative in it. A batch of one has nothing to contrast
            against and the loss is identically zero, which looks like
            success and teaches nothing.
        """

        if len(pairs) < 2:
            raise ValidationError(
                "contrastive training needs at least two pairs, "
                "since a batch of one has no negatives",
                pairs=len(pairs),
            )

        config = self._config

        generator = torch.Generator().manual_seed(config.seed)

        torch.manual_seed(config.seed)

        model = self._encoder.train_mode()

        device = self._encoder.device

        optimizer = torch.optim.AdamW(
            self._parameter_groups(model),
            lr=config.learning_rate,
        )

        steps_per_epoch = max(1, math.ceil(len(pairs) / config.batch_size))

        total_steps = steps_per_epoch * config.epochs

        warmup_steps = int(total_steps * config.warmup_ratio)

        report = TrainingReport(epochs=config.epochs, pairs=len(pairs))

        step = 0

        # Resolved once rather than per step. It also emits a warning when
        # the request cannot be honoured, which should be said once.
        precision = autocast_for(device, config.precision)

        for epoch in range(config.epochs):
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

        report.steps = step

        report.initial_loss = report.losses[0]

        report.final_loss = report.losses[-1]

        model.eval()

        return report

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _step(
        self,
        model: nn.Module,
        batch: Sequence[TextPair],
        device: torch.device,
    ) -> Tensor:
        """
        One InfoNCE step over a batch.

        Anchors and positives are encoded together in a single forward
        pass rather than two. Beyond halving the launch overhead, it
        guarantees both sides see identical dropout conditions, which they
        would not if encoded separately.
        """

        anchors = [pair.anchor for pair in batch]

        positives = [pair.positive for pair in batch]

        # internal to the encoder; the trainer is its only other caller.
        ids, mask = self._encoder._prepare([*anchors, *positives])

        vectors = model(ids, mask)

        vectors = F.normalize(vectors, dim=-1)

        anchor_vectors, positive_vectors = vectors.chunk(2, dim=0)

        logits = anchor_vectors @ positive_vectors.T / self._config.temperature

        # The correct passage for anchor i is positive i, so the target
        # labels are simply the diagonal.
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

        Anchors and positives are laid out as one list, anchors first, so
        that chunk boundaries are ordinary slices and the stacked vectors
        split back into halves by position.

        Normalisation happens inside the loss rather than inside the
        encode, because gradient caching caches raw model outputs and
        takes the loss gradient with respect to those. Normalising early
        would put the operation on the wrong side of the cache.
        """

        texts = [pair.anchor for pair in batch] + [pair.positive for pair in batch]

        def encode(chunk: Sequence[int]) -> Tensor:
            # internal to the encoder; the trainer is its only other caller.
            ids, mask = self._encoder._prepare([texts[index] for index in chunk])

            with precision:
                # nn.Module.__call__ is untyped, so the result widens to Any.
                vectors: Tensor = model(ids, mask)

            return vectors

        def loss_fn(vectors: Tensor) -> Tensor:
            normalised = F.normalize(vectors, dim=-1)

            anchor_vectors, positive_vectors = normalised.chunk(2, dim=0)

            logits = anchor_vectors @ positive_vectors.T / self._config.temperature

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

        A trailing batch of one is dropped: it has no negatives, so its
        loss is zero and its gradient is nothing.
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

        if len(batch) > 1:
            yield batch

    @staticmethod
    def _parameter_groups(model: nn.Module) -> list[dict[str, object]]:
        """
        Split parameters into decayed and undecayed groups.

        Biases and normalisation parameters are excluded from weight
        decay. Decaying a LayerNorm gain drives it toward zero, which
        scales down the layer's entire output.
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
            {"params": decayed, "weight_decay": 0.01},
            {"params": undecayed, "weight_decay": 0.0},
        ]

    def _learning_rate(self, step: int, warmup_steps: int, total_steps: int) -> float:
        """Linear warmup followed by linear decay."""

        peak = self._config.learning_rate

        if warmup_steps and step < warmup_steps:
            return peak * (step + 1) / warmup_steps

        remaining = max(0, total_steps - step)

        span = max(1, total_steps - warmup_steps)

        return peak * remaining / span
