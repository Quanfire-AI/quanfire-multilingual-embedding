"""
Gradient caching, for contrastive batches larger than memory.

Contrastive quality depends on batch size in a way supervised training
does not. Every other example in a batch acts as a negative, so a batch
of 16 asks the model to pick the right passage from 16 candidates while a
batch of 1024 makes it pick from 1024 — a far harder and more useful
task. Published sentence encoders train at 1024 and above.

A 16 GB card fits perhaps 8 to 16 sequences of 512 tokens with
activations retained for the backward pass. That is an order of magnitude
short, and no amount of patience closes it: the limit is memory, not
time.

Gradient caching removes the coupling between batch size and memory, at
the cost of a second forward pass:

1. Encode every chunk under ``no_grad``, keeping only the vectors.
   Activations are discarded as each chunk finishes, so peak memory is
   set by the chunk rather than the batch.
2. Compute the loss over all vectors at once and take its gradient with
   respect to those vectors. This is a small tensor — batch by dimension
   — and it holds everything the chunks need to know about each other.
3. Re-encode each chunk *with* gradients and backpropagate using the
   cached vector gradient as the incoming signal.

The gradients this produces are mathematically identical to those from a
single large batch. It is not an approximation.

That identity has a precondition which is easy to miss and silent when
violated: **the two passes must encode each chunk identically.** Any
randomness inside the model — dropout above all — would otherwise draw
different masks the second time, and the cached gradient would then be
applied to activations it was never computed for. Nothing raises. The
gradients are simply wrong, and wrong by a wide margin rather than a
rounding margin.

So the random state is captured per chunk during the first pass and
restored before that chunk is re-encoded in the second. This module is
correct with dropout enabled; without that handling it would be correct
only at ``dropout=0``, which is exactly the configuration a test is most
likely to use.

Measured on an RTX 4070 Ti SUPER, batch 256, a 5.3M-parameter encoder
over 4,000 pairs — all four cells of the same experiment:

===========  ================  ==============
precision    no caching        chunk size 32
===========  ================  ==============
fp32         4.89 GB / 4.3 s   0.40 GB / 4.7 s
bf16         2.99 GB / 2.7 s   0.29 GB / 4.7 s
===========  ================  ==============

So **caching is what buys the memory** — 12.2x on its own, against 1.6x
for bf16 — and the two together give 16.9x. Final losses across all four
cells spanned 0.51%, which is the exactness claim holding on real
hardware rather than in a unit test.

The wall-clock cost is smaller than the 1.5 to 1.7 times previously
estimated here when measured against fp32 (1.09x), and close to it
against bf16 (1.74x), because bf16 makes the forward pass that caching
duplicates cheaper.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import NamedTuple

import torch
from torch import Tensor, nn

from multilingual_embedding.core.exceptions import ValidationError
from multilingual_embedding.core.logging import get_logger
from multilingual_embedding.utils.validation import require_positive

__all__ = ["EncodeChunk", "LossFromVectors", "cached_contrastive_backward"]

_logger = get_logger(__name__)

# Given a slice of the batch, produce its vectors with the graph attached.
EncodeChunk = Callable[[Sequence[int]], Tensor]

# Given the full stacked vectors, produce a scalar loss.
LossFromVectors = Callable[[Tensor], Tensor]


class _RandomState(NamedTuple):
    """
    Enough random state to replay a forward pass exactly.

    CPU state is always captured. The CUDA generator is captured only
    when the model is on a CUDA device, because reading it forces
    initialisation of a context that may not exist.
    """

    cpu: Tensor

    cuda: Tensor | None


def _device_of(model: nn.Module) -> torch.device:
    """
    Where the model's parameters live.

    Falls back to CPU for a model with no parameters, which is only
    really a test construction but should not raise here.
    """

    for parameter in model.parameters():
        return parameter.device

    return torch.device("cpu")


def _capture_random_state(device: torch.device) -> _RandomState:
    """Record the generator state driving dropout."""

    cuda = torch.cuda.get_rng_state(device) if device.type == "cuda" else None

    return _RandomState(cpu=torch.get_rng_state(), cuda=cuda)


def _restore_random_state(state: _RandomState, device: torch.device) -> None:
    """Rewind the generators to a captured point."""

    torch.set_rng_state(state.cpu)

    if state.cuda is not None:
        torch.cuda.set_rng_state(state.cuda, device)


def cached_contrastive_backward(
    model: nn.Module,
    indices: Sequence[int],
    encode: EncodeChunk,
    loss_fn: LossFromVectors,
    *,
    chunk_size: int,
) -> Tensor:
    """
    Run a contrastive step whose batch exceeds what memory allows.

    Populates ``.grad`` on the model's parameters exactly as a single
    large backward pass would, and returns the loss.

    Parameters
    ----------
    model:
        The encoder. Its gradients are accumulated into, so the caller is
        responsible for zeroing them beforehand.

    indices:
        Positions of the examples forming this batch.

    encode:
        Maps a subset of ``indices`` to their vectors, with the graph
        attached. Called twice per step — once without gradients and once
        with.

    loss_fn:
        Maps the full stacked vectors to a scalar. This is where the
        contrastive objective lives; gradient caching is agnostic to it.

    chunk_size:
        Examples encoded at once. Sets peak memory. The largest value
        that fits is the right one.

    Returns
    -------
    The loss, detached.

    Raises
    ------
    ValidationError
        If the batch is empty or the chunk size is not positive.

    Example
    -------
    ::

        loss = cached_contrastive_backward(
            model, range(1024), encode_slice, info_nce, chunk_size=16
        )
    """

    require_positive(chunk_size, name="chunk_size")

    positions = list(indices)

    if not positions:
        raise ValidationError("cached_contrastive_backward needs a non-empty batch")

    chunks = [
        positions[start : start + chunk_size] for start in range(0, len(positions), chunk_size)
    ]

    device = _device_of(model)

    # Pass one: vectors only. The graph for each chunk is discarded as
    # soon as the chunk is done, which is the whole point.
    #
    # The random state is recorded per chunk, not once for the batch,
    # because the second pass re-encodes chunks one at a time and each
    # must resume from exactly the state its first encoding began with.
    states: list[_RandomState] = []

    with torch.no_grad():
        cached = []

        for chunk in chunks:
            states.append(_capture_random_state(device))

            cached.append(encode(chunk))

    stacked = torch.cat(cached, dim=0).detach().requires_grad_(True)

    # The loss sees the entire batch, so every example contrasts against
    # every other exactly as it would without chunking.
    loss = loss_fn(stacked)

    loss.backward()  # type: ignore[no-untyped-call]

    if stacked.grad is None:  # pragma: no cover - guards a silent no-op
        raise ValidationError(
            "loss produced no gradient with respect to the vectors; "
            "the loss function must depend on its argument"
        )

    vector_gradients = stacked.grad.detach()

    # Pass two: re-encode with the graph and inject the cached gradient.
    # Each chunk's activations exist only while that chunk is processed.
    offset = 0

    for chunk, state in zip(chunks, states, strict=True):
        # Rewind to the state this chunk was first encoded under, so
        # dropout draws the same mask and the cached gradient lands on
        # the activations it was computed for.
        _restore_random_state(state, device)

        vectors = encode(chunk)

        incoming = vector_gradients[offset : offset + len(chunk)]

        vectors.backward(gradient=incoming)  # type: ignore[no-untyped-call]

        offset += len(chunk)

    detached: Tensor = loss.detach()

    return detached


def suggest_chunk_size(
    batch_size: int,
    available_gb: float,
    bytes_per_example: float,
) -> int:
    """
    Estimate a chunk size that fits the available memory.

    A starting point rather than an answer: real usage depends on
    sequence length, model width and the attention kernel in use. Measure
    with the largest value that does not fail, then keep it.

    Parameters
    ----------
    batch_size:
        The effective batch wanted.

    available_gb:
        Memory left for activations, after weights and optimizer state.

    bytes_per_example:
        Activation memory for one example, measured rather than guessed.
    """

    require_positive(batch_size, name="batch_size")

    require_positive(bytes_per_example, name="bytes_per_example")

    if available_gb <= 0:
        raise ValidationError("available_gb must be > 0", available_gb=available_gb)

    fits = int((available_gb * 1024**3) / bytes_per_example)

    return max(1, min(batch_size, fits))
