"""
Text normalization.

Normalization runs before pre-tokenization and decides which surface
forms collapse into a single vocabulary entry. Getting it wrong is
expensive in a multilingual setting: without it the same word acquires
several embedding rows, each trained on a fraction of the evidence.

The registry holds small, single-purpose normalizers that are composed
into a :class:`NormalizerPipeline` by configuration, rather than one
monolithic normalizer with a dozen boolean flags. Order is significant
and therefore explicit — case folding before accent stripping is not the
same operation as the reverse.
"""

from __future__ import annotations

import unicodedata
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

from multilingual_embedding.core.factory import build_all_from_config
from multilingual_embedding.core.logging import get_logger
from multilingual_embedding.core.registry import Registry
from multilingual_embedding.utils.validation import require_one_of

__all__ = [
    "NORMALIZERS",
    "DigitNormalizer",
    "LowercaseNormalizer",
    "NFCNormalizer",
    "NFDNormalizer",
    "NFKCNormalizer",
    "NFKDNormalizer",
    "Normalizer",
    "NormalizerPipeline",
    "StripAccentsNormalizer",
    "WhitespaceNormalizer",
]

_logger = get_logger(__name__)


class Normalizer(ABC):
    """
    Base class for a single normalization step.

    Implementations must be pure functions of their input: the same text
    in gives the same text out, with no dependence on corpus state. That
    is what allows a normalizer chain to be described entirely by a
    config file and reapplied identically at inference time.
    """

    __slots__ = ()

    @abstractmethod
    def normalize(self, text: str) -> str:
        """
        Return the normalized form of ``text``.

        Parameters
        ----------
        text:
            Input string.

        Returns
        -------
        The normalized string.
        """

    def __call__(self, text: str) -> str:
        return self.normalize(text)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


NORMALIZERS: Registry[Normalizer] = Registry("normalizer")


class _UnicodeNormalizer(Normalizer):
    """
    Shared implementation for the four Unicode normalization forms.

    Parameters
    ----------
    form:
        One of ``"NFC"``, ``"NFD"``, ``"NFKC"`` or ``"NFKD"``.
    """

    __slots__ = ("_form",)

    _FORMS = ("NFC", "NFD", "NFKC", "NFKD")

    def __init__(self, form: str) -> None:
        self._form = require_one_of(form.upper(), name="form", allowed=self._FORMS)

    @property
    def form(self) -> str:
        """The Unicode normalization form applied."""

        return self._form

    def normalize(self, text: str) -> str:
        """Apply the configured Unicode normalization form."""

        return unicodedata.normalize(self._form, text)  # type: ignore[arg-type]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(form={self._form!r})"


@NORMALIZERS.register("nfkc")
class NFKCNormalizer(_UnicodeNormalizer):
    """
    Unicode NFKC normalization: compatibility decomposition, then compose.

    This is the default first step for every multilingual pipeline
    because Unicode offers several encodings of what readers treat as
    one character, and a tokenizer that does not unify them trains
    separate vectors for each:

    - Fullwidth Latin (``Ａ``, U+FF21) is pervasive in Japanese and
      Chinese text and maps to plain ``A``.
    - Arabic presentation forms (the U+FB50 and U+FE70 blocks) encode
      contextual ligatures and isolated/initial/medial/final glyph
      variants of letters that NFKC folds back to their base letters.
    - Devanagari and other Indic scripts encode some consonants both as
      a precomposed nukta character and as base plus combining nukta;
      composition makes those identical.

    NFKC is lossy — superscripts, ligatures and fullwidth forms do not
    survive it — which is acceptable for embedding training, where the
    goal is to maximise evidence per vocabulary entry.

    Example
    -------
    ::

        NFKCNormalizer().normalize("ＡＢＣ")  -> "ABC"
    """

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__("NFKC")


@NORMALIZERS.register("nfc")
class NFCNormalizer(_UnicodeNormalizer):
    """
    Unicode NFC normalization: canonical decomposition, then compose.

    Preferred over NFKC when compatibility folding would destroy
    meaningful distinctions, for example in corpora where fullwidth and
    halfwidth forms are themselves the object of study. NFC still
    unifies the two ways of writing an accented letter, which is the
    single most common source of duplicate vocabulary entries in
    European text.
    """

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__("NFC")


@NORMALIZERS.register("nfd")
class NFDNormalizer(_UnicodeNormalizer):
    """
    Unicode NFD normalization: canonical decomposition, no recomposition.

    Rarely useful on its own; it exists because accent handling and some
    diacritic-aware analyses are far simpler once base characters and
    combining marks are separate codepoints.
    """

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__("NFD")


@NORMALIZERS.register("nfkd")
class NFKDNormalizer(_UnicodeNormalizer):
    """
    Unicode NFKD normalization: compatibility decomposition, no recompose.

    The decomposed counterpart of NFKC, for pipelines that strip
    combining marks immediately afterwards.
    """

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__("NFKD")


@NORMALIZERS.register("lowercase")
class LowercaseNormalizer(Normalizer):
    """
    Case folding.

    Halves the vocabulary for cased scripts and is a no-op for the
    uncased ones (Devanagari, Arabic, Han, Thai), so it is safe to leave
    enabled in a multilingual pipeline.

    Example
    -------
    ::

        LowercaseNormalizer().normalize("STRASSE")  -> "strasse"
    """

    __slots__ = ()

    def normalize(self, text: str) -> str:
        """Return the case-folded form of ``text``."""

        # casefold, not lower: `lower` leaves German ß unchanged while
        # `STRASSE` lowercases to `strasse`, so the two spellings of the
        # same word would not unify. casefold also maps Greek final
        # sigma (ς) onto σ, which `lower` likewise leaves distinct.
        return text.casefold()


@NORMALIZERS.register("whitespace")
class WhitespaceNormalizer(Normalizer):
    """
    Collapse whitespace runs to a single space and trim the ends.

    Also folds the exotic separators that survive real-world extraction:
    U+00A0 no-break space (ubiquitous in HTML), U+3000 ideographic space
    (used in CJK typesetting) and the rest of the U+2000 block all become
    ordinary spaces.

    Zero-width characters are handled selectively rather than stripped
    wholesale. U+200B zero-width space and U+FEFF byte order mark are
    invisible formatting artefacts and are removed. U+200C zero-width
    non-joiner and U+200D zero-width joiner are **preserved**: in
    Devanagari they control whether a consonant cluster renders as a
    conjunct, and in Arabic and Indic scripts they distinguish genuinely
    different words. Stripping them silently merges distinct types.

    Example
    -------
    ::

        WhitespaceNormalizer().normalize("  a\\u3000 b  ")  -> "a b"
    """

    __slots__ = ()

    # Invisible characters carrying no linguistic content. Deliberately
    # excludes U+200C/U+200D, which are meaningful in Indic and Arabic.
    _REMOVED = ("​", "﻿")

    def normalize(self, text: str) -> str:
        """Return ``text`` with whitespace collapsed and trimmed."""

        cleaned = text

        for character in self._REMOVED:
            cleaned = cleaned.replace(character, "")

        # str.split() with no argument splits on any Unicode whitespace,
        # which already covers U+00A0 and U+3000, and discards empty
        # fields, so this both collapses runs and strips the ends.
        return " ".join(cleaned.split())


@NORMALIZERS.register("strip_accents")
class StripAccentsNormalizer(Normalizer):
    """
    Remove combining diacritical marks.

    Decomposes to NFD so that accents become standalone combining marks,
    drops every ``Mn`` (mark, non-spacing) codepoint, then recomposes to
    NFC.

    Use with care outside European text: the same ``Mn`` category holds
    Arabic harakat, where stripping is usually desirable, but also Indic
    vowel signs and the virama, where it is destructive. Enable it per
    language rather than globally.

    Example
    -------
    ::

        StripAccentsNormalizer().normalize("café")  -> "cafe"
    """

    __slots__ = ()

    def normalize(self, text: str) -> str:
        """Return ``text`` with non-spacing marks removed."""

        decomposed = unicodedata.normalize("NFD", text)

        stripped = "".join(
            character for character in decomposed if unicodedata.category(character) != "Mn"
        )

        return unicodedata.normalize("NFC", stripped)


@NORMALIZERS.register("digits")
class DigitNormalizer(Normalizer):
    """
    Map every Unicode decimal digit onto its ASCII equivalent.

    Numerals are written with different codepoints per script — ``२०२४``
    in Devanagari, ``٢٠٢٤`` in Arabic-Indic, ``２０２４`` in fullwidth —
    all denoting the same number. Unifying them stops the vocabulary
    filling with script-specific spellings of the same quantity, which
    carry no distinct meaning for an embedding model.

    Example
    -------
    ::

        DigitNormalizer().normalize("२०२४")  -> "2024"
    """

    __slots__ = ()

    def normalize(self, text: str) -> str:
        """Return ``text`` with all decimal digits mapped to ASCII."""

        if not any(character.isdigit() for character in text):
            return text

        characters: list[str] = []

        for character in text:
            # Restrict to Nd (decimal digit): other numeric categories
            # cover fractions, Roman numerals and circled forms, which
            # have no single-digit ASCII equivalent.
            if unicodedata.category(character) == "Nd":
                characters.append(str(unicodedata.digit(character)))
            else:
                characters.append(character)

        return "".join(characters)


class NormalizerPipeline(Normalizer):
    """
    An ordered chain of normalizers applied in sequence.

    Parameters
    ----------
    normalizers:
        Steps to apply, in order. An empty list makes the pipeline an
        identity function, which is a legitimate configuration.

    Example
    -------
    ::

        pipeline = NormalizerPipeline([NFKCNormalizer(), LowercaseNormalizer()])

        pipeline.normalize("ＨＥＬＬＯ")  -> "hello"
    """

    __slots__ = ("_normalizers",)

    def __init__(self, normalizers: Sequence[Normalizer] | None = None) -> None:
        self._normalizers: list[Normalizer] = list(normalizers or ())

    @property
    def normalizers(self) -> list[Normalizer]:
        """The configured steps, in application order."""

        return list(self._normalizers)

    @classmethod
    def from_config(
        cls,
        specs: Sequence[Mapping[str, Any] | str] | None,
    ) -> NormalizerPipeline:
        """
        Build a pipeline from a sequence of component specifications.

        Parameters
        ----------
        specs:
            Ordered specifications such as
            ``[{"type": "nfkc"}, "lowercase"]``. ``None`` or an empty
            sequence yields an identity pipeline.

        Returns
        -------
        The constructed pipeline.

        Raises
        ------
        ConfigurationError
            If a specification is malformed or names an unregistered
            normalizer.

        Example
        -------
        ::

            NormalizerPipeline.from_config([{"type": "nfkc"}, "whitespace"])
        """

        pipeline = cls(build_all_from_config(NORMALIZERS, specs))

        _logger.debug(
            "Built normalizer pipeline",
            extra={"steps": [type(step).__name__ for step in pipeline._normalizers]},
        )

        return pipeline

    def normalize(self, text: str) -> str:
        """Apply every step in order and return the result."""

        result = text

        for normalizer in self._normalizers:
            result = normalizer.normalize(result)

        return result

    def __len__(self) -> int:
        return len(self._normalizers)

    def __repr__(self) -> str:
        names = [type(step).__name__ for step in self._normalizers]

        return f"NormalizerPipeline(steps={names!r})"
