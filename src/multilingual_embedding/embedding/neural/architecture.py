"""
Transformer encoder architecture.

Written out rather than imported from a model library, because this is
the part of the stack the project needs to own. A borrowed checkpoint can
hide a broken training loop; a model defined here cannot.

The design follows current encoder practice rather than the 2017 paper:

*Pre-norm residuals.* Layer normalisation is applied before each
sub-layer rather than after. Post-norm needs a learning-rate warmup to
train stably at depth; pre-norm does not, which matters when the training
budget allows few attempts.

*Fused scaled dot-product attention.* ``F.scaled_dot_product_attention``
dispatches to a fused kernel where one is available, which is both faster
and more memory-frugal than materialising the full attention matrix. That
memory saving is what makes a usable sequence length fit on a 16 GB card.

*GELU activations*, as in every modern encoder.

Positions are learned rather than sinusoidal. For the sequence lengths an
embedding model sees — a few hundred tokens — learned positions cost
little and train slightly better; rotary embeddings would be the choice
for a long-context decoder, not here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from multilingual_embedding.core.exceptions import ValidationError
from multilingual_embedding.utils.validation import require_positive

__all__ = ["EncoderConfig", "TransformerEncoderModel"]


@dataclass(slots=True)
class EncoderConfig:
    """
    Shape of a transformer encoder.

    Attributes
    ----------
    vocabulary_size:
        Number of token ids the embedding table must cover.

    dimension:
        Model width. Must divide evenly by ``heads``.

    layers:
        Number of transformer blocks.

    heads:
        Attention heads per block.

    feedforward_dimension:
        Width of the position-wise feed-forward hidden layer. Four times
        ``dimension`` is the conventional ratio and the default here.

    max_length:
        Longest sequence the learned position table covers. Inputs are
        truncated to this.

    dropout:
        Applied to attention weights and feed-forward activations.

    pad_id:
        Token id that marks padding. Excluded from attention and from
        pooling, so a short sequence in a padded batch encodes identically
        to the same sequence alone.
    """

    vocabulary_size: int

    dimension: int = 256

    layers: int = 4

    heads: int = 4

    feedforward_dimension: int | None = None

    max_length: int = 256

    dropout: float = 0.1

    pad_id: int = 0

    def __post_init__(self) -> None:
        require_positive(self.vocabulary_size, name="vocabulary_size")

        require_positive(self.dimension, name="dimension")

        require_positive(self.layers, name="layers")

        require_positive(self.heads, name="heads")

        require_positive(self.max_length, name="max_length")

        if self.dimension % self.heads:
            raise ValidationError(
                "dimension must divide evenly by heads",
                dimension=self.dimension,
                heads=self.heads,
            )

        if self.feedforward_dimension is None:
            self.feedforward_dimension = 4 * self.dimension

        require_positive(self.feedforward_dimension, name="feedforward_dimension")

    @property
    def head_dimension(self) -> int:
        """Width of each attention head."""

        return self.dimension // self.heads


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention over a padded batch.

    Query, key and value projections are fused into one matrix
    multiplication, which is a meaningful saving at small model sizes
    where kernel launch overhead dominates.
    """

    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()

        self.heads = config.heads

        self.head_dimension = config.head_dimension

        self.qkv = nn.Linear(config.dimension, 3 * config.dimension, bias=False)

        self.projection = nn.Linear(config.dimension, config.dimension, bias=False)

        self.dropout = config.dropout

    def forward(self, hidden: Tensor, padding_mask: Tensor) -> Tensor:
        """
        Parameters
        ----------
        hidden:
            ``(batch, length, dimension)``.

        padding_mask:
            ``(batch, length)``, True where the token is real.

        Returns
        -------
        ``(batch, length, dimension)``.
        """

        batch, length, _ = hidden.shape

        qkv = self.qkv(hidden)

        query, key, value = qkv.chunk(3, dim=-1)

        # (batch, heads, length, head_dimension)
        def split(tensor: Tensor) -> Tensor:
            return tensor.view(batch, length, self.heads, self.head_dimension).transpose(1, 2)

        # Broadcast the key-padding mask over heads and query positions so
        # that padding is never attended to. Without this, padded slots
        # contribute to every real token's representation and a sequence
        # encodes differently depending on what it was batched with.
        attention_mask = padding_mask[:, None, None, :]

        attended = F.scaled_dot_product_attention(
            split(query),
            split(key),
            split(value),
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )

        merged = attended.transpose(1, 2).reshape(batch, length, -1)

        output: Tensor = self.projection(merged)

        return output


class FeedForward(nn.Module):
    """Position-wise feed-forward network with a GELU non-linearity."""

    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()

        assert config.feedforward_dimension is not None

        self.up = nn.Linear(config.dimension, config.feedforward_dimension)

        self.down = nn.Linear(config.feedforward_dimension, config.dimension)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden: Tensor) -> Tensor:
        output: Tensor = self.down(self.dropout(F.gelu(self.up(hidden))))

        return output


class EncoderBlock(nn.Module):
    """
    One pre-norm transformer block.

    Normalisation precedes each sub-layer and the residual is added to the
    un-normalised input, so gradients reach early layers through an
    unobstructed path.
    """

    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()

        self.attention_norm = nn.LayerNorm(config.dimension)

        self.attention = MultiHeadSelfAttention(config)

        self.feedforward_norm = nn.LayerNorm(config.dimension)

        self.feedforward = FeedForward(config)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden: Tensor, padding_mask: Tensor) -> Tensor:
        hidden = hidden + self.dropout(self.attention(self.attention_norm(hidden), padding_mask))

        hidden = hidden + self.dropout(self.feedforward(self.feedforward_norm(hidden)))

        return hidden


class TransformerEncoderModel(nn.Module):
    """
    Transformer encoder producing one pooled vector per sequence.

    Example
    -------
    ::

        config = EncoderConfig(vocabulary_size=1000, dimension=64, layers=2)

        model = TransformerEncoderModel(config)

        vectors = model(token_ids, attention_mask)   # (batch, dimension)
    """

    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()

        self.config = config

        self.token_embedding = nn.Embedding(
            config.vocabulary_size,
            config.dimension,
            padding_idx=config.pad_id,
        )

        self.position_embedding = nn.Embedding(config.max_length, config.dimension)

        self.dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(EncoderBlock(config) for _ in range(config.layers))

        self.final_norm = nn.LayerNorm(config.dimension)

        self.apply(self._initialise)

    @staticmethod
    def _initialise(module: nn.Module) -> None:
        """
        Initialise weights.

        Linear and embedding weights are drawn from a normal with standard
        deviation 0.02, the convention for transformer encoders. Padding
        rows are zeroed so an all-padding position contributes nothing
        before training has begun.
        """

        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0.0)

    @property
    def dimension(self) -> int:
        """Width of a pooled vector."""

        return self.config.dimension

    def parameter_count(self) -> int:
        """Number of trainable parameters."""

        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, token_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """
        Encode a padded batch into one vector per sequence.

        Parameters
        ----------
        token_ids:
            ``(batch, length)`` integer ids.

        attention_mask:
            ``(batch, length)``, 1 for real tokens and 0 for padding.

        Returns
        -------
        ``(batch, dimension)`` — mean-pooled over real tokens only.

        Raises
        ------
        ValidationError
            If the sequence exceeds the position table.
        """

        _, length = token_ids.shape

        if length > self.config.max_length:
            raise ValidationError(
                "sequence is longer than the position table",
                length=length,
                max_length=self.config.max_length,
            )

        positions = torch.arange(length, device=token_ids.device)

        hidden = self.token_embedding(token_ids) + self.position_embedding(positions)

        hidden = self.dropout(hidden)

        # A row of pure padding — an empty string — would be masked at
        # every position, and a softmax over an all -inf row is NaN, which
        # then contaminates the whole batch through the shared weights.
        # Such rows are allowed to attend to position 0 purely to keep the
        # kernel finite; pooling still uses the true mask below, so they
        # come out as zeros, which is what the encoder contract promises.
        attending = attention_mask.bool()

        empty = ~attending.any(dim=1)

        if bool(empty.any()):
            attending = attending.clone()

            attending[empty, 0] = True

        for block in self.blocks:
            hidden = block(hidden, attending)

        hidden = self.final_norm(hidden)

        return self._mean_pool(hidden, attention_mask)

    @staticmethod
    def _mean_pool(hidden: Tensor, attention_mask: Tensor) -> Tensor:
        """
        Average over real tokens.

        Padding is excluded from both numerator and denominator, so a
        sequence encodes identically whatever it was padded alongside. The
        denominator is floored at one to keep an all-padding row finite —
        it yields zeros rather than NaN, which is what the ``TextEncoder``
        contract promises for unencodable input.
        """

        weights = attention_mask.unsqueeze(-1).to(hidden.dtype)

        summed = (hidden * weights).sum(dim=1)

        counts = weights.sum(dim=1).clamp(min=1.0)

        return summed / counts


def sinusoidal_positions(length: int, dimension: int) -> Tensor:
    """
    Classic sinusoidal position encodings.

    Not used by default — positions are learned — but kept because it is
    the drop-in alternative when a model must generalise beyond the
    lengths it was trained on, which learned positions cannot do.
    """

    position = torch.arange(length).unsqueeze(1).float()

    scale = torch.exp(torch.arange(0, dimension, 2).float() * (-math.log(10_000.0) / dimension))

    encoding = torch.zeros(length, dimension)

    encoding[:, 0::2] = torch.sin(position * scale)

    encoding[:, 1::2] = torch.cos(position * scale)

    return encoding
