"""
Reading EU legislation (EUR-Lex Formex) into cross-lingual training pairs.

This is the **training front door** for the EU-law domain — the first
*international*, demand-led source in the factory (the pivot away from
source-led India-government text). It is the multilingual sibling of
:mod:`.annotated_acts` (statutes) and :mod:`.trade` (customs), and it exists for
one structural reason the Indic sources cannot give as cleanly: EUR-Lex
publishes the **same act — one CELEX id — as parallel expressions in up to 24
official languages**, professionally aligned, under a single clean licence. That
turns cross-lingual gold pairs from a mining problem into a *join*.

**The join key is in the data, not inferred.** Formex marks every provision with
a stable ``IDENTIFIER`` attribute (``<ARTICLE IDENTIFIER="001">``, and
``<PARAG IDENTIFIER="001.001">`` beneath it) that is **identical across
languages** — Article 1 of the GDPR is ``IDENTIFIER="001"`` in the English,
German and French expressions alike. So :func:`align_provisions` is an inner
join on that identifier, and :func:`iter_cross_lingual_pairs` emits both
directions of each aligned pair (either language may be the query, so neither is
privileged). Like the trade Act pairs these are **symmetric cross-lingual**
pairs and train with **empty** E5 prefixes — see :func:`prefix_regime`.

**Provenance (the reason this source was chosen): CC-BY 4.0.** EUR-Lex content is
reusable under Creative Commons Attribution 4.0, governed by **Commission
Decision 2011/833/EU** on the reuse of Commission documents — commercial reuse is
named explicitly; the obligation is attribution plus indication of changes.
Metadata is CC0. This is the cleanest footing of any factory source: unlike the
§52 statutory bases (train-safe but redistribute-carefully) CC-BY *permits*
redistribution with attribution. The house rule nonetheless keeps the default at
**train-only** (ship weights + card, not the corpus) until a dataset-release
decision is taken explicitly — a CC-BY dataset release is a separate call, not a
silent consequence of the licence. The carve-outs the OJ notice records (only the
Official Journal is "authentic"; third-party inserts, IP-protected material and
IAS standards need separate clearance) do not touch the legislative-text body
this module reads.

**The one thing the filter must get right: non-substantive provisions.** A
consolidated act carries articles that are no longer text — ``deleted`` /
``repealed`` stubs (``gestrichen``, ``supprimé``, …) whose whole body is the
marker word, and headings with no operative paragraph. A naive "article in, pair
out" reader would train on those. Per the oracle-diff house rule the filter's
effect is *measured*, not asserted: :func:`iter_cross_lingual_pairs` records why
each dropped provision left into an :class:`EuLawStatistics`, so a build can diff
the naive population (:meth:`EuLawFilterConfig.naive`) against the filtered one
and read the removals as a per-reason ledger.

**This module is offline.** It reads Formex a caller has already pulled from
Cellar (the pull is the outward, box-only, gated step); nothing here fetches, so
it is tested against in-memory fixtures, never a live endpoint.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any
from xml.etree import ElementTree as ET

from multilingual_embedding.core.exceptions import MultilingualEmbeddingError
from multilingual_embedding.core.logging import get_logger

from .language import normalize_language_code
from .pairs import MinedPair, token_overlap

__all__ = [
    "EULAW_ATTRIBUTION",
    "EULAW_LEGAL_BASIS",
    "EULAW_LICENSE",
    "EULAW_PROVISION_XLING_KIND",
    "EULAW_SOURCE",
    "EuLawFilterConfig",
    "EuLawReadError",
    "EuLawStatistics",
    "FormexProvision",
    "align_provisions",
    "iter_cross_lingual_pairs",
    "prefix_regime",
    "read_formex_articles",
]

_logger = get_logger(__name__)

# Carried on every record (via the pair's ``document`` id) and quoted on the
# model card. EUR-Lex is CC-BY, so — unlike the §52 statutory sources — there is
# a concrete licence to name, and the attribution obligation travels with any
# derived artefact.
EULAW_SOURCE = "eur-lex"

EULAW_LICENSE = "CC-BY-4.0"

# The standing attribution owed; the concrete CELEX/resource of a specific pull
# is appended by the caller that knows which slice it pulled.
EULAW_ATTRIBUTION = (
    "European Union legislation via EUR-Lex (eur-lex.europa.eu), "
    "© European Union, 1998-present. Reused under the Creative Commons "
    "Attribution 4.0 International licence pursuant to Commission Decision "
    "2011/833/EU. Only the Official Journal is authentic; changes were made "
    "(structured extraction into retrieval pairs)."
)

# The reuse basis, kept verbatim so the card quotes the right instrument.
EULAW_LEGAL_BASIS = (
    "Commission Decision 2011/833/EU of 12 December 2011 on the reuse of "
    "Commission documents; EUR-Lex content licensed CC-BY 4.0."
)

# Cross-lingual provision pairs are symmetric (either language is a valid query),
# so both directions train with EMPTY E5 prefixes — the same regime the trade
# Act pairs and gov-indic bilingual pairs use, and deliberately NOT the
# ``query:``/``passage:`` asymmetric regime. Kept as a table so train and serve
# read the identical prefixes off one place (the silent-bug trap every pillar
# hit).
EULAW_PROVISION_XLING_KIND = "eulaw-provision-xling"

_PREFIXES: dict[str, tuple[str, str]] = {
    EULAW_PROVISION_XLING_KIND: ("", ""),
}

_WHITESPACE = re.compile(r"\s+")

# A provision whose entire body is a "no longer text" marker. Matched against the
# WHOLE normalised body (not merely "contains"), so a substantive article that
# happens to mention the word "repealed" is not dropped — only a stub that IS the
# marker. Multilingual because the corpus is: en/de/fr/es/it cover the v1 set,
# extend as languages are added.
_REPEALED_ONLY = re.compile(
    r"^\W*(?:"
    r"deleted|repealed|"  # en
    r"gestrichen|aufgehoben|"  # de
    r"supprimée?|abrogée?|"  # fr
    r"suprimido|derogado|"  # es
    r"soppresso|abrogato"  # it
    r")\W*$",
    re.IGNORECASE,
)


class EuLawReadError(MultilingualEmbeddingError):
    """Raised when a Formex document cannot be parsed into provisions."""


@dataclass(slots=True)
class FormexProvision:
    """
    One article of an EU act, language-tagged, keyed for cross-lingual join.

    Attributes
    ----------
    number:
        The Formex ``IDENTIFIER`` of the article (``"001"``). This is the join
        key — identical across every language expression of the act — so it is
        the one field alignment reads.

    heading:
        The article title, and its subtitle when present, joined
        (``"Article 1 - Subject-matter and objectives"``). The query side of a
        pair can be a heading, so it is kept whole.

    text:
        The operative body — every paragraph, normalised to single-spaced
        prose, with the title/subtitle removed (they live in ``heading``).
    """

    number: str

    heading: str

    text: str


@dataclass(slots=True)
class EuLawFilterConfig:
    """
    What counts as a usable provision pair.

    Attributes
    ----------
    minimum_heading_characters:
        Floor on the heading. An article with no title carries no query signal.

    minimum_body_characters:
        Floor on the body (passage). Drops heading-only stubs and the residue
        of a deleted article that is not a clean marker.

    drop_repealed:
        Drop a provision whose body is only a ``deleted``/``repealed`` marker
        (in any covered language) — a no-longer-operative article a retriever
        must not learn as a live answer.

    deduplicate:
        Drop repeat provisions with the same normalised (heading, text) within
        one language. Formex annex duplication is real.
    """

    minimum_heading_characters: int = 6

    minimum_body_characters: int = 60

    drop_repealed: bool = True

    deduplicate: bool = True

    @classmethod
    def naive(cls) -> EuLawFilterConfig:
        """The oracle reference: every filter relaxed, so every provision is kept."""

        return cls(
            minimum_heading_characters=0,
            minimum_body_characters=0,
            drop_repealed=False,
            deduplicate=False,
        )


@dataclass(slots=True)
class EuLawStatistics:
    """
    Counts of what was produced and what was dropped, and why.

    The ``rejected`` map is the point: each key is the reason a provision left
    the corpus, so a naive-vs-filtered diff reads as a per-reason ledger rather
    than one opaque "fewer pairs" number.
    """

    produced: int = 0

    rejected: Counter[str] = field(default_factory=Counter)

    mean_overlap: float = 0.0

    _overlap_total: float = 0.0

    def _accept(self, overlap: float) -> None:
        self.produced += 1
        self._overlap_total += overlap
        self.mean_overlap = self._overlap_total / self.produced

    def _reject(self, reason: str) -> None:
        self.rejected[reason] += 1

    def to_dict(self) -> dict[str, Any]:
        """A JSON-friendly view for a build report."""

        return {
            "produced": self.produced,
            "mean_overlap": round(self.mean_overlap, 4),
            "rejected": dict(self.rejected),
        }


def prefix_regime(kind: str) -> tuple[str, str]:
    """
    The (anchor, positive) E5 prefixes a pair kind trains and serves with.

    Cross-lingual provision pairs are symmetric, so they take **empty** prefixes
    — the same table train and serve both read, so the two can never drift.
    """

    try:
        return _PREFIXES[kind]
    except KeyError as exc:
        raise EuLawReadError(f"no prefix regime for pair kind {kind!r}") from exc


def _text_of(element: ET.Element | None) -> str:
    """Normalised concatenation of all text under an element, or ``""``."""

    if element is None:
        return ""

    # Join with a space, not "": Formex wraps whole elements, so adjacent
    # segments are separate tokens (``<NO.PARAG>1.</NO.PARAG><ALINEA>This`` must
    # read "1. This", not "1.This"). Formex never splits a word across a tag, so
    # a space between segments is always a token boundary, and _normalize
    # collapses any doubling.
    return _normalize(" ".join(element.itertext()))


def _normalize(text: str) -> str:
    """Strip, collapse internal whitespace. The one normalisation all text gets."""

    return _WHITESPACE.sub(" ", str(text or "")).strip()


def read_formex_articles(
    xml_source: str | bytes,
    *,
    celex: str,
    language: str,
) -> list[FormexProvision]:
    """
    Parse a Formex ``<ACT>`` document into its articles.

    ``xml_source`` is the Formex XML (the ``<ACT ...>`` payload extracted from a
    Cellar ``fmx4`` manifestation), as text or bytes. Every ``<ARTICLE>`` with an
    ``IDENTIFIER`` becomes one :class:`FormexProvision`: ``TI.ART`` (and
    ``STI.ART`` when present) form the heading, and every other child's text is
    the body. Articles with no ``IDENTIFIER`` are skipped — without the join key
    they cannot be aligned, so they carry no cross-lingual signal.

    ``celex`` and ``language`` are not read from the document (the language is a
    property of the *expression* the caller fetched, not always stamped in the
    body); they are carried by the caller for provenance and the pair language
    tag.

    Raises :class:`EuLawReadError` on malformed XML.
    """

    raw = xml_source if isinstance(xml_source, bytes) else xml_source.encode("utf-8")
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        raise EuLawReadError(f"malformed Formex XML for {celex} [{language}]: {exc}") from exc

    provisions: list[FormexProvision] = []

    for article in root.iter("ARTICLE"):
        number = _normalize(article.get("IDENTIFIER", ""))

        if not number:
            continue

        title = _text_of(article.find("TI.ART"))

        subtitle = _text_of(article.find("STI.ART"))

        heading = f"{title} - {subtitle}" if title and subtitle else (title or subtitle)

        body_parts = [
            _text_of(child) for child in article if child.tag not in ("TI.ART", "STI.ART")
        ]

        text = _normalize(" ".join(part for part in body_parts if part))

        provisions.append(FormexProvision(number=number, heading=heading, text=text))

    if not provisions:
        raise EuLawReadError(
            f"no ARTICLE with an IDENTIFIER found in Formex for {celex} [{language}]"
        )

    return provisions


def align_provisions(
    left: Iterable[FormexProvision],
    right: Iterable[FormexProvision],
) -> list[tuple[FormexProvision, FormexProvision]]:
    """
    Inner-join two language expressions of one act on the Formex identifier.

    The identifier is stable across languages, so the join is exact — no text
    similarity, no heuristic. A provision present in only one expression (a
    language-specific correction, say) has no partner and is dropped: a
    cross-lingual pair needs both sides. The result preserves the left order so
    a build is reproducible.
    """

    right_by_number = {_normalize(p.number): p for p in right}

    aligned: list[tuple[FormexProvision, FormexProvision]] = []

    for provision in left:
        partner = right_by_number.get(_normalize(provision.number))

        if partner is not None:
            aligned.append((provision, partner))

    return aligned


def _reject_reason(provision: FormexProvision, config: EuLawFilterConfig) -> str | None:
    """Name the way a provision is unusable, or ``None`` if it is a live pair side."""

    if config.drop_repealed and provision.text and _REPEALED_ONLY.match(provision.text):
        return "repealed"

    if len(provision.heading) < config.minimum_heading_characters:
        return "heading-too-short"

    if len(provision.text) < config.minimum_body_characters:
        return "body-too-short"

    return None


def iter_cross_lingual_pairs(
    left: Iterable[FormexProvision],
    right: Iterable[FormexProvision],
    *,
    left_language: str,
    right_language: str,
    celex: str,
    config: EuLawFilterConfig | None = None,
    source: str = EULAW_SOURCE,
    kind: str = EULAW_PROVISION_XLING_KIND,
    statistics: EuLawStatistics | None = None,
) -> Iterator[MinedPair]:
    """
    Emit cross-lingual :class:`MinedPair`s from two language expressions of an act.

    The two provision lists are the same act in two languages. They are aligned
    on the Formex identifier (:func:`align_provisions`), each aligned pair is
    filtered on **both** sides (a pair is only as usable as its weaker side), and
    each survivor yields **both directions** — left→right and right→left — with
    empty prefixes, sharing one ``document`` id so a sampler never batches the two
    halves as each other's negatives.

    ``celex`` seeds the document id, so every pair from one act is one facet.
    When ``statistics`` is given, each drop is recorded by reason for the
    oracle-diff ledger; pass :meth:`EuLawFilterConfig.naive` as ``config`` for
    the reference population.
    """

    active = config or EuLawFilterConfig()

    stats = statistics if statistics is not None else EuLawStatistics()

    left_code = normalize_language_code(left_language)

    right_code = normalize_language_code(right_language)

    seen: set[tuple[str, str]] = set()

    for left_provision, right_provision in align_provisions(left, right):
        reason = _reject_reason(left_provision, active) or _reject_reason(right_provision, active)

        if reason is not None:
            stats._reject(reason)
            continue

        left_text = f"{left_provision.heading}. {left_provision.text}"

        right_text = f"{right_provision.heading}. {right_provision.text}"

        if active.deduplicate:
            key = (left_text, right_text)

            if key in seen:
                stats._reject("duplicate")
                continue

            seen.add(key)

        # The join key, not the left side's raw number. `align_provisions`
        # matches on `_normalize(number)`, so that is the identifier both
        # expressions actually agree on; the raw number is whatever one
        # language's Formex happened to write. Building the id from the raw
        # left number makes it depend on *which* language is on the left, and
        # a de/fr pairing whose number carries different whitespace from the
        # en/es one would land under a different document id — which would
        # split cross-lingual twins of the same provision across the train and
        # held-out sides, the one leak this corpus is structurally safe from.
        # Language-independence here is load-bearing, so it is derived rather
        # than assumed.
        position = _normalize(left_provision.number)

        document = f"{source}:{celex}:{position}"

        forward_overlap = token_overlap(left_text, right_text)

        stats._accept(forward_overlap)

        yield MinedPair(
            anchor=left_text,
            positive=right_text,
            kind=kind,
            document=document,
            language=left_code,
            positive_language=right_code,
            overlap=forward_overlap,
        )

        yield MinedPair(
            anchor=right_text,
            positive=left_text,
            kind=kind,
            document=document,
            language=right_code,
            positive_language=left_code,
            overlap=token_overlap(right_text, left_text),
        )
