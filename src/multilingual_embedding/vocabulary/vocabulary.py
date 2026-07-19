"""
Vocabulary: the token to id mapping.

The vocabulary is the contract between the tokenizer and the embedding
model. The tokenizer produces ids against it; the embedding matrix is
indexed by them. Because those ids are baked into a trained model, a
vocabulary is treated as immutable once frozen — see :meth:`freeze`.

Ordering is deterministic: special tokens first, then remaining tokens by
descending frequency with ties broken by the token string. Two runs over
the same corpus therefore produce byte-identical vocabularies, which is
what makes a training run reproducible.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from multilingual_embedding.core.exceptions import ValidationError
from multilingual_embedding.core.logging import get_logger
from multilingual_embedding.utils.io import read_json, write_json

from .special_tokens import (
    BOS_ID,
    EOS_ID,
    PAD_ID,
    UNK_ID,
    SpecialTokenSet,
)

__all__ = ["Vocabulary"]

_logger = get_logger(__name__)

_FORMAT_VERSION = 1


class Vocabulary:
    """
    Bidirectional token/id mapping with frequency information.

    Parameters
    ----------
    special_tokens:
        Reserved tokens occupying the lowest ids.

    Notes
    -----
    Construct via :meth:`from_counter` or
    :class:`~multilingual_embedding.vocabulary.builder.VocabularyBuilder`
    rather than by adding tokens one at a time, so that the frequency
    ordering is applied.
    """

    __slots__ = ("_frequencies", "_frozen", "_ids", "_special", "_tokens")

    def __init__(self, special_tokens: SpecialTokenSet | None = None) -> None:
        self._special = special_tokens if special_tokens is not None else SpecialTokenSet()

        self._tokens: list[str] = []

        self._ids: dict[str, int] = {}

        self._frequencies: list[int] = []

        self._frozen = False

        for token in self._special.as_tuple():
            self._append(token, frequency=0)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_counter(
        cls,
        counts: Mapping[str, int] | Counter[str],
        *,
        min_count: int = 1,
        max_size: int | None = None,
        special_tokens: SpecialTokenSet | None = None,
    ) -> Vocabulary:
        """
        Build a vocabulary from token frequencies.

        Parameters
        ----------
        counts:
            Token to occurrence count mapping.

        min_count:
            Tokens occurring fewer times than this are excluded. Rare
            tokens receive too few gradient updates to acquire a useful
            vector, and each one costs a full embedding row.

        max_size:
            Cap on total size including special tokens. The most
            frequent tokens are kept.

        special_tokens:
            Reserved tokens. Defaults to pad/unk/bos/eos.
        """

        if min_count < 1:
            raise ValidationError("min_count must be >= 1", min_count=min_count)

        vocabulary = cls(special_tokens=special_tokens)

        reserved = vocabulary._special.as_mapping()

        candidates = [
            (token, count)
            for token, count in counts.items()
            if count >= min_count and token not in reserved
        ]

        # Descending frequency, then token string, so the result does not
        # depend on the iteration order of the input mapping.
        candidates.sort(key=lambda entry: (-entry[1], entry[0]))

        if max_size is not None:
            available = max_size - vocabulary._special.count

            if available < 0:
                raise ValidationError(
                    "max_size is smaller than the special token count",
                    max_size=max_size,
                    special_tokens=vocabulary._special.count,
                )

            candidates = candidates[:available]

        for token, count in candidates:
            vocabulary._append(token, frequency=count)

        _logger.info(
            "Built vocabulary",
            extra={
                "size": len(vocabulary),
                "min_count": min_count,
                "max_size": max_size,
            },
        )

        return vocabulary

    @classmethod
    def from_tokens(
        cls,
        tokens: Iterable[str],
        *,
        min_count: int = 1,
        max_size: int | None = None,
        special_tokens: SpecialTokenSet | None = None,
    ) -> Vocabulary:
        """Build a vocabulary by counting an iterable of tokens."""

        return cls.from_counter(
            Counter(tokens),
            min_count=min_count,
            max_size=max_size,
            special_tokens=special_tokens,
        )

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def id_of(self, token: str) -> int:
        """
        Return the id of a token, or the unknown id if absent.

        Never raises: an out-of-vocabulary token is an expected
        condition at inference time, not an error.
        """

        return self._ids.get(token, UNK_ID)

    def token_of(self, token_id: int) -> str:
        """
        Return the token for an id.

        Raises
        ------
        ValidationError
            If the id is outside the vocabulary. Unlike an unknown
            token, an out-of-range id means the caller's model and
            vocabulary disagree, which is a real defect.
        """

        if not 0 <= token_id < len(self._tokens):
            raise ValidationError(
                "Token id is out of range",
                token_id=token_id,
                size=len(self._tokens),
            )

        return self._tokens[token_id]

    def frequency_of(self, token: str) -> int:
        """Return the corpus frequency of a token, or zero if absent."""

        token_id = self._ids.get(token)

        return self._frequencies[token_id] if token_id is not None else 0

    def encode(self, tokens: Sequence[str]) -> list[int]:
        """Map a token sequence to ids, substituting unknown where needed."""

        return [self._ids.get(token, UNK_ID) for token in tokens]

    def decode(self, token_ids: Sequence[int], *, skip_special: bool = True) -> list[str]:
        """
        Map ids back to tokens.

        Parameters
        ----------
        skip_special:
            Omit reserved tokens from the output, which is normally what
            a caller reconstructing text wants.
        """

        tokens: list[str] = []

        for token_id in token_ids:
            token = self.token_of(token_id)

            if skip_special and token_id < self._special.count:
                continue

            tokens.append(token)

        return tokens

    def contains(self, token: str) -> bool:
        """Return True if the token has an id."""

        return token in self._ids

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def special_tokens(self) -> SpecialTokenSet:
        """The reserved tokens of this vocabulary."""

        return self._special

    @property
    def pad_id(self) -> int:
        """Id of the padding token."""

        return PAD_ID

    @property
    def unk_id(self) -> int:
        """Id of the unknown token."""

        return UNK_ID

    @property
    def bos_id(self) -> int:
        """Id of the beginning-of-sequence token."""

        return BOS_ID

    @property
    def eos_id(self) -> int:
        """Id of the end-of-sequence token."""

        return EOS_ID

    @property
    def is_frozen(self) -> bool:
        """True once the vocabulary has been sealed against mutation."""

        return self._frozen

    @property
    def total_frequency(self) -> int:
        """Sum of all token frequencies."""

        return sum(self._frequencies)

    def tokens(self) -> list[str]:
        """Return all tokens in id order."""

        return list(self._tokens)

    def frequencies(self) -> list[int]:
        """Return frequencies in id order."""

        return list(self._frequencies)

    def most_common(self, count: int = 10) -> list[tuple[str, int]]:
        """
        Return the most frequent non-special tokens.

        Already sorted by construction, so this is a slice rather than a
        sort.
        """

        start = self._special.count

        return [
            (self._tokens[index], self._frequencies[index])
            for index in range(start, min(start + count, len(self._tokens)))
        ]

    def coverage(self, tokens: Iterable[str]) -> float:
        """
        Fraction of a token stream that this vocabulary can represent.

        The complement is the unknown rate, the single most useful
        number for judging whether a vocabulary fits a corpus.
        """

        total = 0

        known = 0

        for token in tokens:
            total += 1

            if token in self._ids:
                known += 1

        return known / total if total else 0.0

    def __len__(self) -> int:
        return len(self._tokens)

    def __contains__(self, token: object) -> bool:
        return isinstance(token, str) and token in self._ids

    def __iter__(self) -> Iterator[str]:
        return iter(self._tokens)

    def __getitem__(self, token: str) -> int:
        return self.id_of(token)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vocabulary):
            return NotImplemented

        return self._tokens == other._tokens and self._frequencies == other._frequencies

    def __repr__(self) -> str:
        return f"Vocabulary(size={len(self)}, frozen={self._frozen})"

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, token: str, *, frequency: int = 1) -> int:
        """
        Add a token, or increment its frequency if already present.

        Returns the token's id.

        Raises
        ------
        ValidationError
            If the vocabulary is frozen, or the token is empty.
        """

        if self._frozen:
            raise ValidationError("Cannot modify a frozen vocabulary", token=token)

        if not token:
            raise ValidationError("Cannot add an empty token")

        existing = self._ids.get(token)

        if existing is not None:
            self._frequencies[existing] += frequency

            return existing

        return self._append(token, frequency=frequency)

    def freeze(self) -> Vocabulary:
        """
        Seal the vocabulary against further modification.

        Called once a model has been trained against it. Adding a token
        afterwards would create an id with no corresponding embedding
        row, so this turns a subtle numerical bug into a clear error.
        """

        self._frozen = True

        return self

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Reduce to primitives."""

        return {
            "format_version": _FORMAT_VERSION,
            "special_tokens": list(self._special.as_tuple()),
            "tokens": list(self._tokens),
            "frequencies": list(self._frequencies),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Vocabulary:
        """
        Rebuild from :meth:`to_dict` output.

        Raises
        ------
        ValidationError
            If the payload is malformed or was written by an
            incompatible format version.
        """

        version = data.get("format_version")

        if version != _FORMAT_VERSION:
            raise ValidationError(
                "Unsupported vocabulary format version",
                found=version,
                expected=_FORMAT_VERSION,
            )

        special_values = list(data.get("special_tokens", []))

        tokens = list(data.get("tokens", []))

        frequencies = list(data.get("frequencies", []))

        if len(tokens) != len(frequencies):
            raise ValidationError(
                "Vocabulary tokens and frequencies differ in length",
                tokens=len(tokens),
                frequencies=len(frequencies),
            )

        if len(special_values) != 4:
            raise ValidationError(
                "Vocabulary must define exactly four special tokens",
                found=len(special_values),
            )

        vocabulary = cls(
            special_tokens=SpecialTokenSet(
                pad=special_values[0],
                unk=special_values[1],
                bos=special_values[2],
                eos=special_values[3],
            )
        )

        if tokens[: len(special_values)] != special_values:
            raise ValidationError(
                "Vocabulary does not begin with its special tokens",
                head=tokens[: len(special_values)],
                expected=special_values,
            )

        for index in range(len(special_values), len(tokens)):
            vocabulary._append(tokens[index], frequency=frequencies[index])

        return vocabulary

    def save(self, path: str | Path) -> Path:
        """Write the vocabulary to a JSON file."""

        target = write_json(path, self.to_dict())

        _logger.info("Saved vocabulary", extra={"path": str(target), "size": len(self)})

        return target

    @classmethod
    def load(cls, path: str | Path) -> Vocabulary:
        """Read a vocabulary written by :meth:`save`."""

        return cls.from_dict(read_json(path))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _append(self, token: str, *, frequency: int) -> int:
        """Append a token without checking for duplicates."""

        token_id = len(self._tokens)

        self._tokens.append(token)

        self._ids[token] = token_id

        self._frequencies.append(frequency)

        return token_id
