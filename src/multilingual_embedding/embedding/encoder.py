"""
The text-to-vector contract.

This is the boundary the search pipeline depends on, and the reason it
exists is worth stating plainly.

:class:`~multilingual_embedding.embedding.base.EmbeddingModel` returns an
:class:`~multilingual_embedding.embedding.matrix.EmbeddingMatrix` — a
``vocabulary × dimension`` lookup table. That shape is intrinsic to a
*static* model: every token has one vector, decided at training time and
fixed thereafter.

A *contextual* model has no such table. A transformer computes a vector
for the whole input at call time, so ``"river bank"`` and ``"savings
bank"`` produce different vectors for the same token. There is nothing to
look up and nothing to store per token.

A pipeline written against ``EmbeddingMatrix`` therefore cannot accept a
contextual model at all — not as a matter of quality, but of shape. This
module defines the narrower contract both kinds *can* satisfy: text in,
vectors out.

:class:`TextEncoder` is a :class:`~typing.Protocol` rather than an
abstract base class. That is deliberate: it lets the existing
:class:`~multilingual_embedding.embedding.sentence.SentenceEncoder`
satisfy it without changing its inheritance, and lets a future
transformer encoder satisfy it without importing anything from here. The
contract is the shape, not the ancestry.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np

__all__ = ["TextEncoder", "encoder_dimension"]


@runtime_checkable
class TextEncoder(Protocol):
    """
    Maps text to fixed-length vectors.

    Implementations must satisfy three guarantees, because callers rely
    on them and cannot check them cheaply:

    1. :meth:`encode` returns a one-dimensional array of length
       :attr:`dimension`.
    2. :meth:`encode_batch` returns a two-dimensional array of shape
       ``(len(texts), dimension)``, in the order given.
    3. Input that cannot be encoded — an empty string, or text with no
       recognisable content — yields a **zero vector rather than NaN**.
       A caller detects the degenerate case with a norm check; NaN would
       silently poison every similarity computed against it.

    Example
    -------
    Any object with this shape qualifies, including one backed by no
    model at all::

        class ConstantEncoder:
            dimension = 4

            def encode(self, text): return np.ones(4, dtype=np.float32)

            def encode_batch(self, texts):
                return np.ones((len(texts), 4), dtype=np.float32)

        isinstance(ConstantEncoder(), TextEncoder)  -> True
    """

    @property
    def dimension(self) -> int:
        """Length of a vector produced by this encoder."""
        ...

    def encode(self, text: str) -> np.ndarray:
        """Encode one string into a vector of length :attr:`dimension`."""
        ...

    def encode_batch(self, texts: Sequence[str]) -> np.ndarray:
        """Encode several strings into a ``(n, dimension)`` array."""
        ...


def encoder_dimension(encoder: TextEncoder) -> int:
    """
    Return an encoder's dimension, verifying it against real output.

    ``dimension`` is a declaration; this checks it is true. A mismatch
    between the declared and actual width is the kind of defect that
    otherwise surfaces much later as a corrupt index or a shape error
    deep inside a matrix multiplication.

    Raises
    ------
    ValueError
        If the encoder's output width disagrees with its declaration.
    """

    declared = encoder.dimension

    actual = int(encoder.encode("dimension probe").shape[0])

    if actual != declared:
        raise ValueError(f"encoder declares dimension {declared} but produced {actual}")

    return declared
