"""
Contextual encoders and their training.

This subpackage is the framework's neural half. It is **not** imported by
``multilingual_embedding.embedding``, and that is deliberate: it requires
torch, and the corpus, tokenizer, vocabulary and evaluation layers must
stay installable without a training stack. Import it explicitly::

    from multilingual_embedding.embedding.neural import NeuralTextEncoder

Install the dependency with ``uv sync --extra neural``.

The pieces separate along one line — tensors on one side, text on the
other:

``architecture``
    The transformer itself. Tensors in, tensors out. Knows nothing about
    tokenizers, text or numpy.

``encoder``
    Text in, numpy vectors out. Tokenises, pads, batches, moves to a
    device, normalises. Satisfies the framework's ``TextEncoder``
    contract, so a contextual model serves through the same pipeline as
    the static one.

``training``
    Contrastive fitting on text pairs, with InfoNCE over in-batch
    negatives.
"""

from __future__ import annotations

try:  # pragma: no cover - exercised by the absent-dependency path
    import torch as _torch
except ImportError as error:  # pragma: no cover
    raise ImportError(
        "multilingual_embedding.embedding.neural requires torch. "
        "Install it with: uv sync --extra neural"
    ) from error
else:
    del _torch

from .architecture import EncoderConfig, TransformerEncoderModel
from .encoder import NeuralTextEncoder, Tokenizes, resolve_device
from .training import (
    ContrastiveConfig,
    ContrastiveTrainer,
    TextPair,
    TrainingReport,
)

__all__ = [
    "ContrastiveConfig",
    "ContrastiveTrainer",
    "EncoderConfig",
    "NeuralTextEncoder",
    "TextPair",
    "Tokenizes",
    "TrainingReport",
    "TransformerEncoderModel",
    "resolve_device",
]
