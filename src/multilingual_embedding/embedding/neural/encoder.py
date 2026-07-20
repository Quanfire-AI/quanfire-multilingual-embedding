"""
The contextual encoder, behind the framework's text-to-vector contract.

Separates two concerns that are easy to tangle. The model in
:mod:`.architecture` deals only in tensors; this module deals in text —
tokenising, padding, batching, moving to a device, and handing back numpy
arrays. Keeping them apart means the model can be trained, exported or
swapped without touching the text handling, and the text handling can be
tested without a model.

Vectors are L2-normalised on the way out. Cosine similarity then reduces
to a dot product, which is what the similarity index assumes, and it
makes vectors from differently-sized models directly comparable.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from multilingual_embedding.core.exceptions import ValidationError
from multilingual_embedding.core.logging import get_logger
from multilingual_embedding.utils.filesystem import atomic_write_path, ensure_directory
from multilingual_embedding.utils.io import read_json, write_json
from multilingual_embedding.utils.serialization import from_primitive, to_primitive

from .architecture import EncoderConfig, TransformerEncoderModel

__all__ = ["NeuralTextEncoder", "Tokenizes", "autocast_for", "resolve_device"]

_logger = get_logger(__name__)

_WEIGHTS_FILENAME = "encoder.pt"

_CONFIG_FILENAME = "encoder.json"

_FORMAT_VERSION = 1


class Tokenizes(Protocol):
    """
    The slice of a tokenizer this encoder needs.

    Deliberately narrower than the framework's ``Tokenizer`` class, so any
    object that can turn text into ids qualifies — including a stub in a
    test, which is what lets the encoder be tested without training a
    subword model first.
    """

    def encode(self, text: str) -> Any:
        """Return an object exposing ``ids``."""
        ...


def resolve_device(preference: str | None = None) -> torch.device:
    """
    Choose a compute device.

    Order of preference is CUDA, then Apple Metal, then CPU. An explicit
    preference is honoured without checking, so that a caller can force
    CPU for a reproducibility run — GPU reductions are not
    bit-deterministic across runs.

    ``"auto"`` is spelled out because it is what a configuration file
    carries; it means the same as no preference at all.
    """

    if preference and preference != "auto":
        return torch.device(preference)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def autocast_for(device: torch.device, precision: str) -> AbstractContextManager[None]:
    """
    Return the mixed-precision context a run should train under.

    ``"fp32"`` yields a null context, so the ordinary path carries no
    autocast machinery at all.

    Only bfloat16 is offered. It shares fp32's exponent range, so it
    needs no loss scaling — the ``GradScaler`` dance that fp16 requires
    exists to stop small gradients flushing to zero, and bf16 does not
    have that failure. It trades mantissa bits instead, which training
    tolerates well.

    Apple Metal is deliberately excluded. MPS autocast support has been
    incomplete across torch versions, and silently training in a
    precision the caller did not ask for is worse than ignoring the
    request loudly, so this warns and falls back to fp32.

    Raises
    ------
    ValidationError
        If ``precision`` names something unsupported. Configuration
        catches this earlier; the check is here so the function is safe
        to call directly.
    """

    if precision == "fp32":
        return nullcontext()

    if precision != "bf16":
        raise ValidationError(
            "Unsupported precision",
            precision=precision,
            supported=["fp32", "bf16"],
        )

    if device.type == "mps":
        _logger.warning(
            "Ignoring bf16 request: autocast is unreliable on Apple Metal",
            extra={"device": str(device), "precision": precision},
        )

        return nullcontext()

    # torch ships no annotations for this one, so a strict checker sees an
    # untyped call in typed context.
    if device.type == "cuda" and not torch.cuda.is_bf16_supported():  # type: ignore[no-untyped-call]
        _logger.warning(
            "Ignoring bf16 request: this GPU does not support bfloat16",
            extra={"device": str(device)},
        )

        return nullcontext()

    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


class NeuralTextEncoder:
    """
    Encodes text with a transformer, satisfying ``TextEncoder``.

    Parameters
    ----------
    model:
        The transformer. Placed in evaluation mode on construction;
        :meth:`train_mode` restores training mode.

    tokenizer:
        Anything satisfying :class:`Tokenizes`.

    device:
        Where to run. Resolved automatically when omitted.

    batch_size:
        Sequences per forward pass in :meth:`encode_batch`.

    normalize:
        L2-normalise outputs. On by default; turning it off is only
        sensible when the caller intends to pool vectors further.

    Example
    -------
    ::

        encoder = NeuralTextEncoder(model, tokenizer)

        vector = encoder.encode("some text")        # (dimension,)

        batch = encoder.encode_batch(["a", "b"])    # (2, dimension)
    """

    __slots__ = ("_batch_size", "_device", "_model", "_normalize", "_tokenizer")

    def __init__(
        self,
        model: TransformerEncoderModel,
        tokenizer: Tokenizes,
        *,
        device: str | torch.device | None = None,
        batch_size: int = 32,
        normalize: bool = True,
    ) -> None:
        self._device = torch.device(device) if device is not None else resolve_device()

        self._model = model.to(self._device).eval()

        self._tokenizer = tokenizer

        self._batch_size = batch_size

        self._normalize = normalize

    # ------------------------------------------------------------------
    # The TextEncoder contract
    # ------------------------------------------------------------------

    @property
    def dimension(self) -> int:
        """Width of a vector produced by this encoder."""

        return self._model.dimension

    def encode(self, text: str) -> NDArray[np.float32]:
        """
        Encode one string.

        Returns a zero vector for input that tokenises to nothing, rather
        than NaN, as the contract requires.
        """

        vector: NDArray[np.float32] = self.encode_batch([text])[0]

        return vector

    def encode_batch(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """
        Encode several strings into a ``(len(texts), dimension)`` array.

        Order is preserved. Batches are processed under ``no_grad`` — this
        is the inference path, and retaining the graph would consume
        memory proportional to the whole input.
        """

        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        outputs: list[NDArray[np.float32]] = []

        was_training = self._model.training

        self._model.eval()

        try:
            with torch.no_grad():
                for start in range(0, len(texts), self._batch_size):
                    chunk = texts[start : start + self._batch_size]

                    ids, mask = self._prepare(chunk)

                    vectors = self._model(ids, mask)

                    if self._normalize:
                        vectors = self._l2_normalize(vectors)

                    outputs.append(vectors.detach().to("cpu").numpy().astype(np.float32))
        finally:
            if was_training:
                self._model.train()

        stacked: NDArray[np.float32] = np.vstack(outputs)

        return stacked

    # ------------------------------------------------------------------
    # Tensor preparation
    # ------------------------------------------------------------------

    def _prepare(self, texts: Sequence[str]) -> tuple[Tensor, Tensor]:
        """
        Tokenise and pad a chunk into id and mask tensors.

        Sequences are truncated to the model's position table rather than
        raising, because a caller encoding a long document should get a
        vector for its opening rather than an error.
        """

        limit = self._model.config.max_length

        batches = [list(self._tokenizer.encode(text).ids)[:limit] for text in texts]

        # A batch of entirely empty sequences would give a zero-width
        # tensor, which the attention kernel rejects. One padding column
        # keeps the shape valid and pooling returns zeros, which is the
        # documented answer for unencodable input.
        width = max((len(ids) for ids in batches), default=0) or 1

        pad_id = self._model.config.pad_id

        ids = torch.full((len(batches), width), pad_id, dtype=torch.long)

        mask = torch.zeros((len(batches), width), dtype=torch.long)

        for row, sequence in enumerate(batches):
            if not sequence:
                continue

            ids[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)

            mask[row, : len(sequence)] = 1

        return ids.to(self._device), mask.to(self._device)

    @staticmethod
    def _l2_normalize(vectors: Tensor) -> Tensor:
        """
        Scale rows to unit length, leaving zero rows at zero.

        The clamp is what stops an all-padding row becoming NaN, which
        would then poison every similarity computed against it.
        """

        norms = vectors.norm(dim=-1, keepdim=True).clamp(min=1e-12)

        scaled: Tensor = vectors / norms

        return scaled

    # ------------------------------------------------------------------
    # Access and persistence
    # ------------------------------------------------------------------

    @property
    def model(self) -> TransformerEncoderModel:
        """The underlying transformer."""

        return self._model

    @property
    def device(self) -> torch.device:
        """Where this encoder runs."""

        return self._device

    def train_mode(self) -> TransformerEncoderModel:
        """Put the model into training mode and return it."""

        return self._model.train()

    def save(self, directory: str | Path) -> Path:
        """
        Persist weights and configuration.

        The architecture is written alongside the weights: a state
        dictionary alone cannot be loaded without knowing the shape that
        produced it.
        """

        target = ensure_directory(directory)

        with atomic_write_path(target / _WEIGHTS_FILENAME) as temporary:
            torch.save(self._model.state_dict(), temporary)

        write_json(
            target / _CONFIG_FILENAME,
            {
                "format_version": _FORMAT_VERSION,
                "architecture": to_primitive(self._model.config),
                "normalize": self._normalize,
            },
        )

        _logger.info(
            "Saved encoder",
            extra={
                "directory": str(target),
                "parameters": self._model.parameter_count(),
                "dimension": self.dimension,
            },
        )

        return target

    @classmethod
    def load(
        cls,
        directory: str | Path,
        tokenizer: Tokenizes,
        *,
        device: str | torch.device | None = None,
        batch_size: int = 32,
    ) -> NeuralTextEncoder:
        """
        Restore an encoder written by :meth:`save`.

        Raises
        ------
        ValidationError
            If the payload was written by an incompatible format version.
        """

        source = Path(directory).expanduser()

        payload = read_json(source / _CONFIG_FILENAME)

        version = payload.get("format_version")

        if version != _FORMAT_VERSION:
            raise ValidationError(
                "Unsupported encoder format version",
                found=version,
                expected=_FORMAT_VERSION,
            )

        config = from_primitive(EncoderConfig, payload["architecture"])

        model = TransformerEncoderModel(config)

        resolved = torch.device(device) if device is not None else resolve_device()

        state = torch.load(
            source / _WEIGHTS_FILENAME,
            map_location=resolved,
            weights_only=True,
        )

        model.load_state_dict(state)

        return cls(
            model,
            tokenizer,
            device=resolved,
            batch_size=batch_size,
            normalize=bool(payload.get("normalize", True)),
        )

    def __repr__(self) -> str:
        return (
            f"NeuralTextEncoder(dimension={self.dimension}, "
            f"parameters={self._model.parameter_count()}, "
            f"device={self._device})"
        )
