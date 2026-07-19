"""
Special token definitions.

Special tokens occupy the lowest ids, and their order is fixed. Two
reasons this matters in practice:

- ``<pad>`` is id 0, so a zero-filled array is a valid padded batch and
  padding never needs to be written explicitly.
- ``<unk>`` is id 1, so an unknown lookup has a defined answer even
  against a vocabulary that was built without ever seeing one.

Changing the order would silently invalidate every previously trained
model, since ids are baked into the embedding matrix. It is therefore
fixed here rather than made configurable.
"""

from __future__ import annotations

from dataclasses import dataclass

from multilingual_embedding.common.enums import SpecialToken

__all__ = [
    "BOS_ID",
    "EOS_ID",
    "PAD_ID",
    "SPECIAL_TOKEN_ORDER",
    "UNK_ID",
    "SpecialTokenSet",
]

# Fixed order. Index in this tuple is the token's id.
SPECIAL_TOKEN_ORDER: tuple[SpecialToken, ...] = (
    SpecialToken.PAD,
    SpecialToken.UNK,
    SpecialToken.BOS,
    SpecialToken.EOS,
)

PAD_ID = 0

UNK_ID = 1

BOS_ID = 2

EOS_ID = 3


@dataclass(slots=True, frozen=True)
class SpecialTokenSet:
    """
    The reserved tokens of a vocabulary.

    Attributes
    ----------
    pad, unk, bos, eos:
        Surface forms of the reserved tokens.
    """

    pad: str = SpecialToken.PAD.value

    unk: str = SpecialToken.UNK.value

    bos: str = SpecialToken.BOS.value

    eos: str = SpecialToken.EOS.value

    def as_tuple(self) -> tuple[str, ...]:
        """Return the surface forms in id order."""

        return (self.pad, self.unk, self.bos, self.eos)

    def as_mapping(self) -> dict[str, int]:
        """Return a surface form to id mapping."""

        return {token: index for index, token in enumerate(self.as_tuple())}

    @property
    def count(self) -> int:
        """Number of reserved ids."""

        return len(self.as_tuple())

    def __contains__(self, token: object) -> bool:
        return isinstance(token, str) and token in self.as_tuple()
