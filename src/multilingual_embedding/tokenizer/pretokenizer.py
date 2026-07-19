"""
Pre-tokenization: splitting text into candidate units.

A pre-tokenizer decides where a subword model is *allowed* to place a
boundary. It runs before the subword model and its output constrains
everything downstream, so an inappropriate choice here cannot be
recovered from later.

Every pre-tokenizer emits :class:`~multilingual_embedding.corpus.token.Token`
objects carrying spans into the *input* string, which is what allows a
prediction over tokens to be mapped back onto the original characters.
The invariant every implementation must uphold is::

    token.text == text[token.span.start:token.span.end]
"""

from __future__ import annotations

import re
import unicodedata
from abc import ABC, abstractmethod

from multilingual_embedding.core.registry import Registry
from multilingual_embedding.corpus.script import (
    Script,
    is_whitespace_delimited,
    script_of_character,
)
from multilingual_embedding.corpus.token import Token

__all__ = [
    "PRETOKENIZERS",
    "CharacterPreTokenizer",
    "PreTokenizer",
    "PunctuationPreTokenizer",
    "ScriptAwarePreTokenizer",
    "WhitespacePreTokenizer",
]

_NON_WHITESPACE = re.compile(r"\S+")


def _is_punctuation(character: str) -> bool:
    """
    Return True for punctuation and symbol characters.

    Both the ``P`` and ``S`` general categories are treated as
    punctuation: currency signs, mathematical operators and emoji are
    boundary-forming in exactly the same way as a comma.
    """

    return unicodedata.category(character)[0] in {"P", "S"}


class PreTokenizer(ABC):
    """
    Base class for pre-tokenizers.

    Implementations are stateless and reusable across threads; all
    per-call state lives in locals.
    """

    __slots__ = ()

    @abstractmethod
    def pre_tokenize(self, text: str) -> list[Token]:
        """
        Split ``text`` into tokens with spans into ``text``.

        Parameters
        ----------
        text:
            Input string.

        Returns
        -------
        Tokens in left-to-right order, each satisfying
        ``token.text == text[token.span.start:token.span.end]``.
        """

    def __call__(self, text: str) -> list[Token]:
        return self.pre_tokenize(text)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


PRETOKENIZERS: Registry[PreTokenizer] = Registry("pretokenizer")


@PRETOKENIZERS.register("whitespace")
class WhitespacePreTokenizer(PreTokenizer):
    """
    Split on runs of whitespace.

    The cheapest useful pre-tokenizer and the right default for
    whitespace-delimited scripts. It attaches punctuation to the
    neighbouring word, so ``"world!"`` stays one unit; use
    :class:`PunctuationPreTokenizer` when that is not wanted.

    Example
    -------
    ::

        WhitespacePreTokenizer().pre_tokenize("hello world")
        -> [Token("hello"), Token("world")]
    """

    __slots__ = ()

    def pre_tokenize(self, text: str) -> list[Token]:
        """Return whitespace-delimited tokens with exact spans."""

        return [
            Token.create(match.group(), start=match.start())
            for match in _NON_WHITESPACE.finditer(text)
        ]


@PRETOKENIZERS.register("char")
class CharacterPreTokenizer(PreTokenizer):
    """
    Emit one token per non-whitespace character.

    A fallback for scripts with no reliable word boundaries at all, and
    the basis of character-level embedding baselines. Whitespace is
    dropped rather than emitted, since a space carries no meaning as a
    unit of its own.

    Example
    -------
    ::

        CharacterPreTokenizer().pre_tokenize("ab c")
        -> [Token("a"), Token("b"), Token("c")]
    """

    __slots__ = ()

    def pre_tokenize(self, text: str) -> list[Token]:
        """Return one token per non-whitespace character."""

        return [
            Token.create(character, start=index)
            for index, character in enumerate(text)
            if not character.isspace()
        ]


def _split_words_and_punctuation(text: str, offset: int = 0) -> list[Token]:
    """
    Split a whitespace-delimited stretch into words and punctuation.

    Maximal runs of characters that are neither whitespace nor
    punctuation become one token each; every punctuation character
    becomes its own token. ``offset`` is added to all spans so the
    result can be embedded in a larger string.
    """

    tokens: list[Token] = []

    start: int | None = None

    for index, character in enumerate(text):
        if character.isspace() or _is_punctuation(character):
            if start is not None:
                tokens.append(Token.create(text[start:index], start=offset + start))

                start = None

            if not character.isspace():
                tokens.append(Token.create(character, start=offset + index))
        elif start is None:
            start = index

    if start is not None:
        tokens.append(Token.create(text[start:], start=offset + start))

    return tokens


@PRETOKENIZERS.register("punctuation")
class PunctuationPreTokenizer(PreTokenizer):
    """
    Split on whitespace, then peel punctuation into separate tokens.

    Keeps ``"world"`` and ``"!"`` distinct so that the vocabulary is not
    padded with a punctuated variant of every word. This is the usual
    choice for word-level models over Latin, Cyrillic and Indic text.

    Example
    -------
    ::

        PunctuationPreTokenizer().pre_tokenize("hello, world!")
        -> [Token("hello"), Token(","), Token("world"), Token("!")]
    """

    __slots__ = ()

    def pre_tokenize(self, text: str) -> list[Token]:
        """Return word and punctuation tokens with exact spans."""

        return _split_words_and_punctuation(text)


@PRETOKENIZERS.register("script")
class ScriptAwarePreTokenizer(PreTokenizer):
    """
    Segment into same-script runs, then split each run by its own rules.

    This is the pre-tokenizer that makes the pipeline multilingual
    rather than Latin-with-exceptions. Whitespace splitting is not a
    universal rule, it is a property of particular writing systems:
    Japanese, Chinese and Thai are written without spaces between words,
    so a whitespace pre-tokenizer applied to Japanese returns whole
    sentences as single tokens. Those tokens are almost all hapaxes, the
    vocabulary explodes, and every subsequent stage inherits the damage.

    The text is therefore first cut into runs of a single script, and
    each run is handled according to
    :func:`~multilingual_embedding.corpus.script.is_whitespace_delimited`:

    - whitespace-delimited runs (Latin, Devanagari, Arabic, Cyrillic,
      Hangul, ...) are split on whitespace and punctuation;
    - non-delimited runs (Han, Hiragana, Katakana, Thai) emit one token
      per character, which gives a subword model a sane starting point
      without requiring a language-specific word segmenter.

    Punctuation, digits and whitespace carry no script evidence and so
    never start a new run; they are absorbed into the run in progress
    and split by that run's rules.

    Example
    -------
    ::

        ScriptAwarePreTokenizer().pre_tokenize("Hello 世界")
        -> [Token("Hello"), Token("世"), Token("界")]
    """

    __slots__ = ()

    def pre_tokenize(self, text: str) -> list[Token]:
        """Return script-appropriate tokens with exact spans."""

        tokens: list[Token] = []

        for start, end, script in self._script_runs(text):
            if is_whitespace_delimited(script):
                tokens.extend(_split_words_and_punctuation(text[start:end], offset=start))
            else:
                tokens.extend(
                    Token.create(character, start=start + index)
                    for index, character in enumerate(text[start:end])
                    if not character.isspace()
                )

        return tokens

    @staticmethod
    def _script_runs(text: str) -> list[tuple[int, int, Script]]:
        """
        Cut ``text`` into ``(start, end, script)`` runs.

        A run ends only where one script-bearing character is followed
        by a character of a different script. Script-neutral characters
        extend the current run, so a trailing space or comma stays with
        the words it punctuates rather than forming a run of its own.
        """

        if not text:
            return []

        runs: list[tuple[int, int, Script]] = []

        run_start = 0

        current = Script.COMMON

        for index, character in enumerate(text):
            script = script_of_character(character)

            if script in {Script.COMMON, Script.UNKNOWN}:
                continue

            if current is Script.COMMON:
                current = script
            elif script is not current:
                runs.append((run_start, index, current))

                run_start = index

                current = script

        runs.append((run_start, len(text), current))

        return runs
