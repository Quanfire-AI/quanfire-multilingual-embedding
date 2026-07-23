"""
Adapting a published encoder, rather than reimplementing one.

The transformer in :mod:`.architecture` is written out here and trains
from scratch. That was the right way to build a training loop you can
trust — a borrowed checkpoint would have masked a broken loop — but it is
the wrong way to get a model worth serving. What a published encoder has
is pretraining scale, and that is precisely the thing a single consumer
GPU cannot reproduce.

So this loads someone else's weights and puts them behind this project's
``TextEncoder`` contract. The architecture is theirs; the corpus, the
mined pairs, the adaptation and the evaluation are yours, and those are
what a domain-specific model is actually made of.

**Why not load their weights into our own transformer.** Ours is
pre-norm — the layer norm sits inside the residual branch — while most
published encoders are post-norm. The parameter *shapes* match, so such a
load succeeds and produces a model that is structurally valid and
numerically wrong, degrading quietly rather than failing. Going through
the original library avoids inventing that failure.

**What this costs.** The base install downloads nothing at runtime and
this breaks that: weights are fetched on first use and cached. It is
deliberate, it is opt-in behind an extra, and ``local_files_only`` makes
a run refuse to reach the network at all — which is what a reproducible
experiment wants.

The wrapper is a plain ``nn.Module`` taking ``(ids, mask)`` and returning
pooled vectors, which is the shape :class:`ContrastiveTrainer` already
expects. LoRA, gradient caching and the retrieval evaluation therefore
work against a pretrained model without changes.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from multilingual_embedding.core.exceptions import MultilingualEmbeddingError
from multilingual_embedding.core.logging import get_logger

from .encoder import resolve_device

__all__ = [
    "POOLING_STRATEGIES",
    "PretrainedEncoderError",
    "PretrainedTextEncoder",
]

_logger = get_logger(__name__)

# How a sequence of token vectors becomes one vector.
#
# It is not a detail. A model trained with mean pooling and served with
# CLS pooling produces plausible vectors that encode the wrong thing, and
# nothing raises — the failure appears as mediocre retrieval, which is
# indistinguishable from a model that is merely mediocre.
POOLING_STRATEGIES: tuple[str, ...] = ("mean", "cls")


class PretrainedEncoderError(MultilingualEmbeddingError):
    """Raised when a checkpoint cannot be loaded or used."""


class _PooledTransformer(nn.Module):
    """
    A published encoder, wearing this project's forward signature.

    Takes ``(ids, mask)`` and returns one pooled, normalised vector per
    sequence — the same shape :class:`TransformerEncoderModel` produces,
    so the existing trainer, LoRA and gradient caching apply unchanged.
    """

    def __init__(self, model: nn.Module, *, pooling: str, dimension: int) -> None:
        super().__init__()

        self.model = model

        self.pooling = pooling

        self._dimension = dimension

    @property
    def dimension(self) -> int:
        """Width of a vector this produces."""

        return self._dimension

    def parameter_count(self) -> int:
        """Total parameters, matching the native model's interface."""

        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, ids: Tensor, mask: Tensor) -> Tensor:
        """Encode a batch of token ids into pooled vectors."""

        output = self.model(input_ids=ids, attention_mask=mask)

        hidden = output.last_hidden_state

        if self.pooling == "cls":
            first: Tensor = hidden[:, 0]

            return first

        # Mean over real tokens only. Including padding would make a
        # sequence's vector depend on what happened to be batched
        # alongside it, which survives every shape test and quietly
        # degrades retrieval.
        weights = mask.unsqueeze(-1).to(hidden.dtype)

        summed = (hidden * weights).sum(dim=1)

        counts = weights.sum(dim=1).clamp(min=1e-9)

        pooled: Tensor = summed / counts

        return pooled


class PretrainedTextEncoder:
    """
    A published encoder behind the framework's text-to-vector contract.

    Satisfies ``TextEncoder``, so the similarity index, the search
    pipeline and :func:`evaluate_retrieval` accept it without knowing it
    is not a native model.

    Parameters
    ----------
    model, tokenizer:
        As returned by the upstream library.

    pooling:
        ``"mean"`` or ``"cls"``. **Match what the checkpoint was trained
        with.** Sentence-transformers models are usually mean-pooled;
        getting this wrong produces plausible vectors encoding the wrong
        thing, and it does not raise.

    Example
    -------
    ::

        encoder = PretrainedTextEncoder.load("intfloat/multilingual-e5-small")

        report = evaluate_retrieval(encoder, held_out_pairs)
    """

    __slots__ = (
        "_batch_size",
        "_device",
        "_max_length",
        "_model",
        "_name",
        "_normalize",
        "_tokenizer",
    )

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        *,
        pooling: str = "mean",
        device: str | torch.device | None = None,
        batch_size: int = 32,
        max_length: int = 512,
        normalize: bool = True,
        name: str = "<in-memory>",
    ) -> None:
        if pooling not in POOLING_STRATEGIES:
            raise PretrainedEncoderError(
                "Unsupported pooling strategy",
                pooling=pooling,
                supported=list(POOLING_STRATEGIES),
            )

        dimension = _hidden_size(model)

        self._device = resolve_device(str(device) if device is not None else None)

        self._model = _PooledTransformer(model, pooling=pooling, dimension=dimension)

        self._model = self._model.to(self._device).eval()

        self._tokenizer = tokenizer

        self._batch_size = batch_size

        self._max_length = max_length

        self._normalize = normalize

        self._name = name

        # `checkpoint` rather than `name`: `name` is a reserved LogRecord
        # attribute, and passing one through `extra` raises KeyError when
        # a handler is attached. It does not raise when logging is
        # unconfigured, so this failed only once another test enabled INFO
        # — a landmine that waits for a formatter.
        _logger.info(
            "Loaded pretrained encoder",
            extra={
                "checkpoint": name,
                "dimension": dimension,
                "pooling": pooling,
                "device": str(self._device),
                "parameters": self._model.parameter_count(),
            },
        )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        name: str,
        *,
        pooling: str = "mean",
        device: str | torch.device | None = None,
        batch_size: int = 32,
        max_length: int = 512,
        normalize: bool = True,
        local_files_only: bool = False,
        **options: Any,
    ) -> PretrainedTextEncoder:
        """
        Load a checkpoint by name.

        Parameters
        ----------
        name:
            Repository identifier or a local directory.

        local_files_only:
            Refuse to reach the network. Worth setting for anything whose
            result is meant to be reproducible — otherwise an upstream
            repository can change what a run trains from, silently.

        Raises
        ------
        PretrainedEncoderError
            If the extra is not installed, or the checkpoint cannot be
            loaded.
        """

        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as error:
            raise PretrainedEncoderError(
                "Adapting a published checkpoint needs the 'pretrained' extra "
                "(uv sync --extra pretrained)",
                missing="transformers",
            ) from error

        try:
            # transformers ships no annotations for these, so a strict
            # checker sees untyped calls in typed context.
            tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
                name,
                local_files_only=local_files_only,
            )

            model = AutoModel.from_pretrained(
                name,
                local_files_only=local_files_only,
                **options,
            )
        except Exception as error:
            raise PretrainedEncoderError(
                "Could not load the checkpoint",
                name=name,
                local_files_only=local_files_only,
                detail=str(error)[:300],
            ) from error

        return cls(
            model,
            tokenizer,
            pooling=pooling,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
            normalize=normalize,
            name=name,
        )

    # ------------------------------------------------------------------
    # The TextEncoder contract
    # ------------------------------------------------------------------

    @property
    def dimension(self) -> int:
        """Width of a vector produced by this encoder."""

        return self._model.dimension

    @property
    def device(self) -> torch.device:
        """Where this encoder runs."""

        return self._device

    @property
    def name(self) -> str:
        """What was loaded, for recording alongside a result."""

        return self._name

    def encode(self, text: str) -> NDArray[np.float32]:
        """Encode one string."""

        vector: NDArray[np.float32] = self.encode_batch([text])[0]

        return vector

    def encode_batch(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """
        Encode several strings into a ``(len(texts), dimension)`` array.

        Order is preserved, and batches run under ``no_grad`` since this
        is the inference path.
        """

        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        was_training = self._model.training

        self._model.eval()

        outputs: list[NDArray[np.float32]] = []

        try:
            with torch.no_grad():
                for start in range(0, len(texts), self._batch_size):
                    chunk = list(texts[start : start + self._batch_size])

                    ids, mask = self._prepare(chunk)

                    vectors = self._model(ids, mask)

                    if self._normalize:
                        vectors = torch.nn.functional.normalize(vectors, dim=-1)

                    outputs.append(vectors.detach().to("cpu").numpy().astype(np.float32))
        finally:
            if was_training:
                self._model.train()

        return np.concatenate(outputs, axis=0)

    # ------------------------------------------------------------------
    # What the trainer needs
    # ------------------------------------------------------------------

    def train_mode(self) -> nn.Module:
        """Put the model in training mode and return it."""

        self._model.train()

        return self._model

    def _prepare(self, texts: Sequence[str]) -> tuple[Tensor, Tensor]:
        """
        Tokenize and pad, returning ids and an attention mask.

        Named to match :class:`NeuralTextEncoder`, because
        :class:`ContrastiveTrainer` calls it. A pretrained encoder is
        therefore trainable by the existing loop, with the existing LoRA
        and gradient caching, and no branch anywhere for which kind of
        model is in hand.
        """

        encoded = self._tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt",
        )

        return (
            encoded["input_ids"].to(self._device),
            encoded["attention_mask"].to(self._device),
        )

    def __repr__(self) -> str:
        return (
            f"PretrainedTextEncoder(name={self._name!r}, "
            f"dimension={self.dimension}, device={self._device})"
        )


def _hidden_size(model: nn.Module) -> int:
    """
    Width of the model's output, from its own configuration.

    Read rather than inferred by running a forward pass, which would need
    a tokenizer and would fail on a model that is not yet on a device.
    """

    config = getattr(model, "config", None)

    for attribute in ("hidden_size", "d_model", "dim", "hidden_dim"):
        size = getattr(config, attribute, None)

        if isinstance(size, int) and size > 0:
            return size

    raise PretrainedEncoderError(
        "Could not determine the encoder's output width from its config",
        looked_for=["hidden_size", "d_model", "dim", "hidden_dim"],
    )
