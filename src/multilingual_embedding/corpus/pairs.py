"""
Manufacturing training pairs from unlabelled text.

Contrastive training needs an anchor and a positive — a query and the
passage answering it. Labelled pairs do not exist for most domains and
never will, so they have to be built out of whatever structure the
documents already carry.

Wikipedia carries a lot of it. A title is a query and its lead paragraph
is the answer; a section heading is a query and its body is the answer;
consecutive paragraphs are about the same thing. None of that is
labelling — it is reading structure the author already imposed.

**The failure this module exists to avoid is lexical leakage.** If the
anchor's words are simply present in the positive, a model can score the
pair correctly by matching strings, learn nothing about meaning, and
still show a falling loss. Wikipedia leads almost always open by
restating the title, so the most obvious pair source is also the most
contaminated. Overlap is therefore measured for every pair, reported by
kind, and filterable — see :class:`PairConfig.maximum_overlap`.

A second, quieter problem is the false negative. Contrastive training
treats every other passage in the batch as a negative, so two pairs mined
from the *same article* punish the model for noticing they are related.
Pairs carry the identifier they came from, which is what lets a sampler
keep them apart.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

from multilingual_embedding.core.exceptions import ValidationError
from multilingual_embedding.core.logging import get_logger
from multilingual_embedding.utils.hashing import hash_text
from multilingual_embedding.utils.validation import require_positive

from .document import Document
from .script import detect_script, is_whitespace_delimited

__all__ = [
    "MinedPair",
    "PairConfig",
    "PairKind",
    "PairStatistics",
    "mine_pairs",
    "token_overlap",
]

_logger = get_logger(__name__)

# Splits on anything that is not a letter, digit or combining mark, so
# it works for scripts with no whitespace word boundaries as well as for
# Latin. Deliberately crude: this measures overlap, it does not tokenize
# for training.
_WORD = re.compile(r"[^\W_]+", re.UNICODE)

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")

# Above this, a pair kind is solvable by string matching often enough that
# a falling loss stops being evidence the model learned anything. Warned
# about rather than filtered, because the right response depends on what
# else is available: with few pairs, leaky ones may still beat none.
_LEAKY_OVERLAP = 0.75


class PairKind:
    """
    Where a pair came from.

    Recorded on every pair because the kinds differ in quality and a
    caller should be able to weight, filter or measure them separately.
    """

    TITLE_LEAD = "title_lead"

    HEADING_SECTION = "heading_section"

    ADJACENT = "adjacent"


@dataclass(slots=True, frozen=True)
class MinedPair:
    """
    One anchor and one positive, with its provenance.

    Attributes
    ----------
    anchor:
        The query side — a title, a heading, a preceding paragraph.

    positive:
        The passage side.

    kind:
        Which structure produced it, one of :class:`PairKind`.

    document:
        Identifier of the document it came from. Two pairs sharing this
        are about the same subject, so a sampler that puts them in one
        batch is teaching the model that a correct match is wrong.

    language:
        Carried through so a mixed-language pair set stays separable.

    overlap:
        Share of the anchor's words that also appear in the positive, in
        ``[0, 1]``. High values mean the pair is solvable by string
        matching.
    """

    anchor: str

    positive: str

    kind: str

    document: str

    language: str | None

    overlap: float

    def to_record(self) -> dict[str, Any]:
        """Reduce to a JSON Lines record."""

        return {
            "anchor": self.anchor,
            "positive": self.positive,
            "kind": self.kind,
            "document": self.document,
            "language": self.language,
            "overlap": round(self.overlap, 4),
        }


@dataclass(slots=True)
class PairConfig:
    """
    What counts as a usable pair.

    Attributes
    ----------
    minimum_anchor_characters:
        Below this an anchor carries no query signal. One-word section
        headings are common and mostly useless.

    minimum_positive_characters:
        Below this a passage is a fragment rather than an answer.

    maximum_positive_characters:
        Passages are truncated to this. A whole section can be thousands
        of characters, most of which the encoder will never see past its
        maximum sequence length, and the opening carries the topic.

    maximum_overlap:
        Reject a pair when this share of the anchor's words already
        appear in the positive. ``1.0`` accepts everything.

        The default is deliberately permissive rather than principled.
        Title/lead pairs overlap heavily by nature — a Wikipedia lead
        restates its title — and dropping them would discard the largest
        and most reliable source of pairs. Tightening this trades volume
        for pairs that cannot be solved lexically; the right value is an
        experiment, not a constant, which is why it is exposed.

    kinds:
        Which pair sources to mine.
    """

    minimum_anchor_characters: int = 8

    minimum_positive_characters: int = 100

    maximum_positive_characters: int = 2000

    maximum_overlap: float = 1.0

    kinds: tuple[str, ...] = (
        PairKind.TITLE_LEAD,
        PairKind.HEADING_SECTION,
        PairKind.ADJACENT,
    )

    def __post_init__(self) -> None:
        require_positive(self.minimum_anchor_characters, name="minimum_anchor_characters")

        require_positive(self.minimum_positive_characters, name="minimum_positive_characters")

        require_positive(self.maximum_positive_characters, name="maximum_positive_characters")

        if self.minimum_positive_characters > self.maximum_positive_characters:
            raise ValidationError(
                "minimum_positive_characters must not exceed maximum_positive_characters",
                minimum=self.minimum_positive_characters,
                maximum=self.maximum_positive_characters,
            )

        if not 0.0 <= self.maximum_overlap <= 1.0:
            raise ValidationError(
                "maximum_overlap must lie in [0, 1]",
                maximum_overlap=self.maximum_overlap,
            )

        unknown = set(self.kinds) - {
            PairKind.TITLE_LEAD,
            PairKind.HEADING_SECTION,
            PairKind.ADJACENT,
        }

        if unknown:
            raise ValidationError("Unknown pair kind", unknown=sorted(unknown))


@dataclass(slots=True)
class PairStatistics:
    """
    What mining produced, and what it threw away.

    Reported rather than logged quietly, because the yield and the mean
    overlap are the two numbers that decide whether a pair set is worth
    training on.
    """

    documents: int = 0

    produced: int = 0

    by_kind: dict[str, int] = field(default_factory=dict)

    mean_overlap_by_kind: dict[str, float] = field(default_factory=dict)

    rejected_short_anchor: int = 0

    rejected_short_positive: int = 0

    rejected_overlap: int = 0

    rejected_duplicate: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Reduce to primitives for reporting."""

        return {
            "documents": self.documents,
            "produced": self.produced,
            "by_kind": dict(sorted(self.by_kind.items())),
            "mean_overlap_by_kind": {
                kind: round(value, 4) for kind, value in sorted(self.mean_overlap_by_kind.items())
            },
            "rejected": {
                "short_anchor": self.rejected_short_anchor,
                "short_positive": self.rejected_short_positive,
                "overlap": self.rejected_overlap,
                "duplicate": self.rejected_duplicate,
            },
        }


def _units(text: str) -> set[str]:
    """
    The comparable units of a string, chosen by script.

    Whitespace-delimited scripts give words. Han, Kana, Thai and the rest
    do not put spaces between words, so splitting on whitespace returns
    whole sentences as single tokens and two texts sharing every
    character would intersect in nothing.

    That is not a cosmetic difference. It silently reported an overlap of
    ``0.0`` for Japanese, which meant every Japanese pair passed the
    leakage filter however contaminated it was — the safety check was
    disabled for exactly the scripts that need it most. Character bigrams
    are used there instead: crude, but they compare what is actually
    shared.
    """

    if is_whitespace_delimited(detect_script(text).dominant):
        return {word.casefold() for word in _WORD.findall(text)}

    characters = "".join(_WORD.findall(text)).casefold()

    if len(characters) < 2:
        return set(characters)

    return {characters[index : index + 2] for index in range(len(characters) - 1)}


def token_overlap(anchor: str, positive: str) -> float:
    """
    Share of the anchor's units that also occur in the positive.

    Units are words for whitespace-delimited scripts and character
    bigrams otherwise, so the measure means the same thing in Hindi,
    English and Japanese. Case-folded, so a capitalised title matches its
    restatement.

    Returns ``0.0`` for an anchor with no units, rather than dividing by
    zero.

    Example
    -------
    ::

        token_overlap("Mumbai", "Mumbai is a city")       # 1.0
        token_overlap("climate", "Mumbai is a city")      # 0.0
        token_overlap("機械学習", "機械学習は面白い")        # 1.0
    """

    anchor_units = _units(anchor)

    if not anchor_units:
        return 0.0

    return len(anchor_units & _units(positive)) / len(anchor_units)


def _paragraphs(text: str) -> list[str]:
    """Split on blank lines, dropping empties."""

    return [block.strip() for block in _PARAGRAPH_BREAK.split(text) if block.strip()]


def _candidates(document: Document, config: PairConfig) -> Iterator[tuple[str, str, str]]:
    """
    Yield ``(anchor, positive, kind)`` before any quality filtering.

    Kept separate from filtering so that each rejection can be counted
    against its reason rather than disappearing into a single number.
    """

    metadata = document.metadata

    title = (metadata.title or "").strip()

    attributes = metadata.base.attributes or {}

    sections = attributes.get("sections") or []

    paragraphs = _paragraphs(document.text)

    if PairKind.TITLE_LEAD in config.kinds and title and paragraphs:
        yield title, paragraphs[0], PairKind.TITLE_LEAD

    if PairKind.HEADING_SECTION in config.kinds:
        for section in sections:
            if not isinstance(section, dict):
                continue

            heading = str(section.get("heading") or "").strip()

            body = str(section.get("text") or "").strip()

            if heading and body:
                yield heading, body, PairKind.HEADING_SECTION

    if PairKind.ADJACENT in config.kinds:
        for first, second in itertools.pairwise(paragraphs):
            yield first, second, PairKind.ADJACENT


def mine_pairs(
    documents: Iterable[Document],
    config: PairConfig | None = None,
) -> tuple[list[MinedPair], PairStatistics]:
    """
    Build training pairs from documents, and report what happened.

    Consumes the iterable once. Holds the pairs and a set of content
    hashes, so memory grows with the pair set rather than with the
    corpus — a pair set is the thing you are building, and it has to fit.

    Parameters
    ----------
    documents:
        Any iterable of documents. Structure that is present is used;
        structure that is absent is skipped, so a corpus with no
        ``sections`` still yields adjacency pairs.

    config:
        Quality rules. Defaults are permissive; see
        :attr:`PairConfig.maximum_overlap` for the one that most affects
        what a model learns.

    Returns
    -------
    The pairs, and the statistics describing what was rejected.

    Example
    -------
    ::

        pairs, stats = mine_pairs(reader.iter_documents())

        print(stats.to_dict()["mean_overlap_by_kind"])
    """

    settings = config if config is not None else PairConfig()

    statistics = PairStatistics()

    pairs: list[MinedPair] = []

    seen: set[str] = set()

    overlap_totals: dict[str, float] = {}

    for document in documents:
        statistics.documents += 1

        for anchor, positive, kind in _candidates(document, settings):
            anchor = " ".join(anchor.split())

            positive = " ".join(positive.split())

            if len(anchor) < settings.minimum_anchor_characters:
                statistics.rejected_short_anchor += 1

                continue

            if len(positive) < settings.minimum_positive_characters:
                statistics.rejected_short_positive += 1

                continue

            positive = positive[: settings.maximum_positive_characters]

            overlap = token_overlap(anchor, positive)

            if overlap > settings.maximum_overlap:
                statistics.rejected_overlap += 1

                continue

            digest = hash_text(f"{anchor}\x00{positive}")

            if digest in seen:
                statistics.rejected_duplicate += 1

                continue

            seen.add(digest)

            pairs.append(
                MinedPair(
                    anchor=anchor,
                    positive=positive,
                    kind=kind,
                    document=document.identifier or "",
                    language=document.metadata.base.language,
                    overlap=overlap,
                )
            )

            statistics.produced += 1

            statistics.by_kind[kind] = statistics.by_kind.get(kind, 0) + 1

            overlap_totals[kind] = overlap_totals.get(kind, 0.0) + overlap

    statistics.mean_overlap_by_kind = {
        kind: overlap_totals[kind] / count for kind, count in statistics.by_kind.items()
    }

    _logger.info(
        "Mined training pairs",
        extra={
            "documents": statistics.documents,
            "pairs": statistics.produced,
            "by_kind": statistics.by_kind,
        },
    )

    for kind, mean in statistics.mean_overlap_by_kind.items():
        if mean > _LEAKY_OVERLAP:
            _logger.warning(
                "Pair kind is largely solvable by string matching; a falling "
                "loss on it is weak evidence the model learned meaning",
                extra={
                    "kind": kind,
                    "mean_overlap": round(mean, 4),
                    "pairs": statistics.by_kind[kind],
                    "remedy": "lower PairConfig.maximum_overlap, or drop this kind",
                },
            )

    return pairs, statistics
