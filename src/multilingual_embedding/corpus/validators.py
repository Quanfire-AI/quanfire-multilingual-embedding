"""
Corpus cleaning and validation rules.

Text arriving from the wild carries boilerplate, duplicates and encoding
damage. Training on it wastes capacity and skews the vocabulary toward
noise, so filtering happens before the tokenizer ever sees the text.

Filters are conservative by default: a rule fires only when the evidence
is unambiguous. Over-aggressive cleaning silently discards valid
non-Latin text, which is a far worse failure for this framework than
letting some noise through.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

from multilingual_embedding.core.logging import get_logger
from multilingual_embedding.utils.hashing import hash_text

from .document import Document
from .script import Script, detect_script
from .sentence import Sentence

__all__ = [
    "DocumentDeduplicator",
    "FilterReport",
    "SentenceFilter",
    "validate_document",
]

_logger = get_logger(__name__)

# Codepoints that indicate decoding went wrong upstream rather than
# being legitimate content.
_REPLACEMENT_CHARACTER = "�"


@dataclass(slots=True)
class FilterReport:
    """
    Counts of what a filter accepted and why it rejected the rest.

    Kept alongside the filtered corpus so that an unexpectedly small
    training set can be traced to the rule responsible.
    """

    accepted: int = 0

    rejected_blank: int = 0

    rejected_too_short: int = 0

    rejected_too_long: int = 0

    rejected_no_letters: int = 0

    rejected_encoding_damage: int = 0

    rejected_script_mismatch: int = 0

    @property
    def rejected(self) -> int:
        """Total rejected across all rules."""

        return (
            self.rejected_blank
            + self.rejected_too_short
            + self.rejected_too_long
            + self.rejected_no_letters
            + self.rejected_encoding_damage
            + self.rejected_script_mismatch
        )

    @property
    def total(self) -> int:
        """Total sentences examined."""

        return self.accepted + self.rejected

    @property
    def acceptance_rate(self) -> float:
        """Share of sentences accepted, in [0, 1]."""

        return self.accepted / self.total if self.total else 0.0

    def to_dict(self) -> dict[str, int | float]:
        """Reduce to primitives for reporting."""

        return {
            "total": self.total,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "rejected_blank": self.rejected_blank,
            "rejected_too_short": self.rejected_too_short,
            "rejected_too_long": self.rejected_too_long,
            "rejected_no_letters": self.rejected_no_letters,
            "rejected_encoding_damage": self.rejected_encoding_damage,
            "rejected_script_mismatch": self.rejected_script_mismatch,
            "acceptance_rate": round(self.acceptance_rate, 5),
        }


@dataclass(slots=True)
class SentenceFilter:
    """
    Rule based sentence filter.

    Attributes
    ----------
    min_characters:
        Shortest acceptable sentence. Very short fragments are usually
        segmentation debris rather than sentences.

    max_characters:
        Longest acceptable sentence. An extreme length nearly always
        means segmentation failed — unsegmented markup, a table, or a
        file with no sentence terminators at all.

    require_letters:
        Reject sentences with no letter characters, such as rows of
        digits or punctuation.

    max_replacement_character_ratio:
        Reject sentences where U+FFFD exceeds this share of characters,
        which indicates the source was decoded with the wrong encoding.

    expected_script:
        When set, reject sentences whose dominant script differs. Left
        unset by default, since mixed-script text is normal in genuinely
        multilingual corpora.

    report:
        Running tally of decisions.
    """

    min_characters: int = 2

    max_characters: int = 10_000

    require_letters: bool = True

    max_replacement_character_ratio: float = 0.10

    expected_script: Script | None = None

    report: FilterReport = field(default_factory=FilterReport)

    def accepts(self, text: str) -> bool:
        """
        Decide whether a sentence passes, updating :attr:`report`.
        """

        stripped = text.strip()

        if not stripped:
            self.report.rejected_blank += 1

            return False

        if len(stripped) < self.min_characters:
            self.report.rejected_too_short += 1

            return False

        if len(stripped) > self.max_characters:
            self.report.rejected_too_long += 1

            return False

        if self.require_letters and not _contains_letter(stripped):
            self.report.rejected_no_letters += 1

            return False

        damage = stripped.count(_REPLACEMENT_CHARACTER) / len(stripped)

        if damage > self.max_replacement_character_ratio:
            self.report.rejected_encoding_damage += 1

            return False

        if (
            self.expected_script is not None
            and detect_script(stripped).dominant is not self.expected_script
        ):
            self.report.rejected_script_mismatch += 1

            return False

        self.report.accepted += 1

        return True

    def filter_sentences(self, sentences: Iterable[Sentence]) -> Iterator[Sentence]:
        """Yield only the sentences that pass."""

        for sentence in sentences:
            if self.accepts(sentence.text):
                yield sentence

    def filter_texts(self, texts: Iterable[str]) -> Iterator[str]:
        """Yield only the strings that pass."""

        for text in texts:
            if self.accepts(text):
                yield text

    def apply(self, document: Document) -> Document:
        """
        Remove failing sentences from a document, in place.

        Paragraphs left with no sentences are dropped. The document's
        own ``text`` is deliberately left unchanged so that the original
        source remains recoverable; only the segmentation is pruned.
        """

        for paragraph in document.paragraphs:
            paragraph.children = [
                sentence for sentence in paragraph.sentences if self.accepts(sentence.text)
            ]

        document.children = [paragraph for paragraph in document.paragraphs if paragraph.sentences]

        return document


class DocumentDeduplicator:
    """
    Exact duplicate detection over document text.

    Duplicated documents are common in scraped corpora and inflate the
    apparent frequency of whatever they contain, biasing both the
    tokenizer vocabulary and the embedding distribution.

    Detection is exact, on NFC-normalised, whitespace-collapsed text.
    Near-duplicate detection would need MinHash or SimHash; that is a
    deliberate omission rather than an oversight, since it carries a
    false-positive risk that exact matching does not.

    Only content hashes are retained, not the text, so memory grows by a
    fixed number of bytes per document.
    """

    __slots__ = ("_duplicates", "_seen")

    def __init__(self) -> None:
        self._seen: set[str] = set()

        self._duplicates = 0

    @property
    def duplicate_count(self) -> int:
        """Number of documents rejected as duplicates."""

        return self._duplicates

    @property
    def unique_count(self) -> int:
        """Number of distinct documents seen."""

        return len(self._seen)

    def is_duplicate(self, document: Document) -> bool:
        """Return True if this document's text was already seen."""

        digest = hash_text(_canonical(document.text))

        if digest in self._seen:
            self._duplicates += 1

            return True

        self._seen.add(digest)

        return False

    def filter(self, documents: Iterable[Document]) -> Iterator[Document]:
        """Yield only the first occurrence of each distinct document."""

        for document in documents:
            if not self.is_duplicate(document):
                yield document


def validate_document(document: Document) -> list[str]:
    """
    Return a list of human readable problems with a document.

    An empty list means the document is sound. Problems are reported
    rather than raised so that a caller can log them and continue over a
    large corpus.
    """

    problems: list[str] = []

    if not document.text.strip():
        problems.append("document text is empty")

    if not document.paragraphs:
        problems.append("document has no paragraphs")

    if document.paragraphs and document.sentence_count == 0:
        problems.append("document has paragraphs but no sentences")

    try:
        document.verify()
    except Exception as error:
        problems.append(f"span inconsistency: {error}")

    return problems


def _contains_letter(text: str) -> bool:
    """Return True if any character is a Unicode letter."""

    return any(unicodedata.category(character).startswith("L") for character in text)


def _canonical(text: str) -> str:
    """Normalise text for duplicate comparison."""

    return " ".join(unicodedata.normalize("NFC", text).split())
