"""
Reading Indian import/export (customs & foreign-trade) material into the corpus.

This is a **training front door** for the trade-compliance domain — the
import/export sibling of :mod:`.annotated_acts` (statutes) and :mod:`.kcc`
(agriculture advisory). Trade sits on a **three-layer** clean footing, and the
two axes of that footing are the two things this module builds:

- **Layer A — bare Acts (Customs Act 1962, Customs Tariff Act 1975).** These are
  section-structured and *bilingual* (authoritative Hindi + English), so they
  feed the existing recursive statute reader for the monolingual heading/section
  pairs, and the **cross-lingual** Hindi↔English pairing they also afford is
  :func:`iter_bilingual_section_pairs` here. That helper is deliberately general:
  gov-indic will reuse it for the same authoritative-bilingual showcase.
- **Layers B/C — notifications, circulars and the Foreign Trade Policy.** These
  are *not* section-structured like an Act; they arrive as flat rows shaped
  ``<notification-number, date, subject, body>``. :func:`iter_notification_pairs`
  is the one genuinely new parser — the flat-row shape of :mod:`.kcc`, not the
  recursive shape of :mod:`.annotated_acts` — mining an asymmetric
  short-subject → long-body retrieval pair from each row.

**The §52 footing differs per layer, and both strings are recorded.** Indian
government works are Crown-copyright-like (never auto-public-domain), so the
operative grant is **§52 of the Copyright Act 1957**. For a *bare Act* the
exception is §52(1)(q)(**ii**) — reproduction is free only "together with any
commentary", so bare-Act text is *train-safe* for a non-reconstructive embedding
(it emits vectors, never text) but the raw corpus is **not to be redistributed**.
For a *notification* the exception is §52(1)(q)(**i**) — Gazette matter (any
notification, order, rule or bye-law) is **freely reproducible** with no
commentary condition, the cleanest of the three layers. Both bases ride here as
constants so the model card can quote the right one for each layer.

**The one thing the notification parser must get right: the noise.** A raw
notification feed carries rows a naive "row in, pair out" reader would train on
and should not: **withdrawn / superseded** notifications (no longer operative),
**empty-subject** rows, **annexure-only** rows whose body is just a pointer to an
enclosed table, exact duplicates, and **subject-quoted-in-body** leakage where
the subject line is repeated verbatim at the top of the body — a pair a
string-matcher wins without learning meaning. Per the oracle-diff house rule the
filter's effect is *measured*, not asserted: :func:`iter_notification_pairs`
records why each dropped row left into a :class:`TradeStatistics`, so a build can
diff the naive population (:meth:`TradeFilterConfig.naive`) against the filtered
one and read the removals as a per-reason ledger rather than an opaque "fewer
pairs".

**Redistribution guardrail (carries from statute).** Keep the raw trade corpus
OFF the public repo — train on the box, publish weights + card only. Layer A
especially (the q(ii) redistribute-carefully basis). This module reads rows a
caller has already pulled; nothing here fetches, so it is offline and is tested
against in-memory fixtures, never a live portal.
"""

from __future__ import annotations

import csv
import gzip
import json
import re
import sys
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from multilingual_embedding.core.exceptions import MultilingualEmbeddingError
from multilingual_embedding.core.logging import get_logger

from .language import normalize_language_code
from .pairs import MinedPair, token_overlap

__all__ = [
    "TRADE_ATTRIBUTION",
    "TRADE_LEGAL_BASIS_ACT",
    "TRADE_LEGAL_BASIS_GAZETTE",
    "TRADE_LICENSE",
    "TRADE_NOTIFICATION_KIND",
    "TRADE_SECTION_KIND",
    "TRADE_SECTION_XLING_KIND",
    "TRADE_SOURCE",
    "BilingualAlignmentStatistics",
    "BilingualSection",
    "TradeFilterConfig",
    "TradeNotification",
    "TradeReadError",
    "TradeStatistics",
    "align_bilingual_sections",
    "extract_notification_pairs",
    "iter_bilingual_section_pairs",
    "iter_notification_pairs",
    "prefix_regime",
    "read_notification_rows",
]

_logger = get_logger(__name__)

# Recorded on every record (via the pair's ``document`` id) and carried on the
# model card. Trade material is pulled from the authoritative government portals
# directly, so — unlike KCC's GODL grant or the statute dataset's CC-BY — there
# is no third-party licence: the right to reproduce rests on the Copyright Act
# 1957 §52(1)(q) statutory basis, differently per layer (see the two legal-basis
# strings below). The constant is concise; the detail lives in the basis strings.
TRADE_SOURCE = "india-customs-trade"

TRADE_LICENSE = "GoI-statutory"

# The attribution owed to the sources, kept verbatim so the model card can quote
# it without re-deriving the citation. The concrete resource URL of a specific
# notification pull is appended by the caller that knows which slice it pulled;
# this is the standing portion.
TRADE_ATTRIBUTION = (
    "Indian customs and foreign-trade material: Central Board of Indirect Taxes "
    "and Customs (CBIC) and Directorate General of Foreign Trade (DGFT), "
    "Government of India; the Customs Act 1962 and Customs Tariff Act 1975 via "
    "India Code (indiacode.nic.in), Legislative Department, Government of India."
)

# Layer A — the bare Acts. §52(1)(q)(ii) frees reproduction of an Act only
# "together with any commentary thereon"; bare statute is not squarely inside
# that, so it is train-safe for a non-reconstructive embedding but the raw
# corpus is not redistributed. This is the record's honest self-description for
# Customs Act / Customs Tariff Act text and rides on cross-lingual section pairs.
TRADE_LEGAL_BASIS_ACT = (
    "Copyright Act 1957 s.52(1)(q)(ii); bare-Act text (Customs Act 1962, "
    "Customs Tariff Act 1975) is train-safe for a non-reconstructive embedding "
    "(emits vectors, not text) but the raw corpus is not to be redistributed"
)

# Layers B/C — notifications, circulars, orders and the Foreign Trade Policy.
# §52(1)(q)(i) makes Gazette matter freely reproducible with no commentary
# condition, so the reproduction trigger simply does not fire. This is the
# cleaner basis, and it rides on the notification pairs.
TRADE_LEGAL_BASIS_GAZETTE = (
    "Copyright Act 1957 s.52(1)(q)(i); Gazette matter (CBIC/DGFT notifications, "
    "circulars, orders and the Foreign Trade Policy) is freely reproducible — "
    "the reproduction condition does not fire"
)

# Pair kinds. The notification kind is asymmetric (a short subject queries a long
# body), so it trains with the E5 ``query:``/``passage:`` prefixes; the section
# kinds are symmetric register (a section against a section, or its cross-lingual
# twin), so they train with empty prefixes — see :func:`prefix_regime`.
TRADE_NOTIFICATION_KIND = "trade_notification"

TRADE_SECTION_KIND = "trade_section"

TRADE_SECTION_XLING_KIND = "trade_section_xling"

# The E5 prefix regime per pair kind, kept as data so the symmetric-vs-asymmetric
# decision from the pair design is one loud, testable table rather than a rule
# re-derived at train time. Asymmetric notification pairs take the non-empty
# ``query:``/``passage:`` prefixes; symmetric section pairs (mono or cross-lingual)
# take empty prefixes, exactly as statute/legal do.
_E5_PREFIXES: dict[str, tuple[str, str]] = {
    TRADE_NOTIFICATION_KIND: ("query: ", "passage: "),
    TRADE_SECTION_KIND: ("", ""),
    TRADE_SECTION_XLING_KIND: ("", ""),
}

# Notifications are English-dominant; Layer A is genuinely Hindi + English. Both
# are recorded, never guessed.
_DEFAULT_NOTIFICATION_LANGUAGE = "en"

_DEFAULT_ENGLISH_LANGUAGE = "en"

_DEFAULT_HINDI_LANGUAGE = "hi"

# Column names as a notification CSV export ships them. Kept as a mapping from
# our field to the source header so a renamed column is a one-line fix, not a
# reader rewrite. Values are matched case-insensitively.
_COLUMNS = {
    "notification_number": "NotificationNumber",
    "date": "Date",
    "subject": "Subject",
    "body": "Body",
    "instrument": "Instrument",
    "layer": "Layer",
    "year": "Year",
}

# A notification whose subject or body marks it no longer operative. Split into
# two families so the ledger names which — a withdrawal and a supersession are
# both "not a live rule" but a reviewer wants to see the shape of the feed.
_WITHDRAWN = re.compile(r"\b(withdrawn|withdrawal|rescind(?:ed)?)\b", re.IGNORECASE)

_SUPERSEDED = re.compile(
    r"\b(supersed(?:e|ed|es)|in\s+supersession|substituted\s+by)\b", re.IGNORECASE
)

# A body that is only a pointer to an enclosed annexure / schedule / table, with
# no substantive prose of its own — nothing a retriever can match a query to.
_ANNEXURE_ONLY = re.compile(
    r"^(?:see\s+|refer\s+to\s+|as\s+per\s+)?(?:the\s+)?"
    r"(?:annexure|annex|schedule|appendix|enclosure|table)\b[\s\S]{0,80}$",
    re.IGNORECASE,
)

_WHITESPACE = re.compile(r"\s+", re.UNICODE)


class TradeReadError(MultilingualEmbeddingError):
    """Raised when a notification file cannot be read or lacks its columns."""


def prefix_regime(kind: str) -> tuple[str, str]:
    """
    The ``(query_prefix, passage_prefix)`` a pair kind trains with.

    Asymmetric notification pairs take the non-empty E5 ``query:``/``passage:``
    prefixes; symmetric section pairs (monolingual or cross-lingual) take empty
    prefixes, the same regime statute and legal use. An unknown kind is treated
    as symmetric — empty prefixes never mislabel a side.
    """

    return _E5_PREFIXES.get(kind, ("", ""))


@dataclass(slots=True)
class TradeNotification:
    """
    One raw notification row, before filtering.

    Attributes
    ----------
    subject:
        The notification's subject line — the short query side of the pair.

    body:
        The operative text — the long passage side of the pair.

    notification_number:
        The citation (``"12/2024-Customs"``), carried into the pair's document
        id for provenance, and a natural query form in its own right.

    date:
        The issue date string, carried for provenance only.

    instrument, layer, year:
        Facet fields for facet-sliced evaluation (retrieval quality per
        instrument, per layer, per year). Not part of the trained text.
    """

    subject: str

    body: str

    notification_number: str = ""

    date: str = ""

    instrument: str = ""

    layer: str = ""

    year: str = ""


@dataclass(slots=True)
class BilingualSection:
    """
    One Act section in its aligned Hindi and English forms.

    The authoritative bilingual Acts (Customs Act 1962, Customs Tariff Act 1975)
    carry the *same* section in both languages, which is a natural cross-lingual
    training pair. This is the input shape :func:`iter_bilingual_section_pairs`
    reads; a caller aligns the two language versions by section id upstream.

    Attributes
    ----------
    identifier:
        The section id shared by both language versions (e.g.
        ``"customs-act-1962:s.12"``), used as the pair's document id so both
        directions trace back to one section and a sampler keeps them apart.

    english, hindi:
        The section text in each language. Either being blank means the section
        is not aligned and yields no cross-lingual pair.

    heading:
        The section's marginal heading, carried for provenance only.
    """

    identifier: str

    english: str

    hindi: str

    heading: str = ""


@dataclass(slots=True)
class TradeFilterConfig:
    """
    Knobs for the notification noise filter.

    Defaults are the disciplined pass. A *naive* reference — the oracle for the
    diff — is this config with every filter relaxed (:meth:`naive`), so a build
    can measure exactly what discipline removes.

    Attributes
    ----------
    minimum_subject_characters:
        Floor on the subject (query) length. Trade subjects are terse but
        contentful ("Revised tariff value of gold"); the floor drops fragments,
        not real subjects.

    minimum_body_characters:
        Floor on the body (passage) length. Drops one-line stubs that are not
        retrieval targets.

    maximum_overlap:
        Reject a pair whose BODY-to-subject token overlap exceeds this — i.e. a
        stub whose body merely restates the subject, a string-solvable pair with
        nothing to retrieve. Measured body->subject on purpose: subject->body
        containment is ~1.0 for every genuine notification (the subject is a
        headline drawn from the body), so the reverse direction is the one that
        isolates trivial pairs. ``1.0`` disables the check.

    drop_superseded:
        Drop notifications whose subject or body marks them withdrawn,
        rescinded or superseded — no longer operative rules a retriever should
        not learn as live answers.

    drop_empty_subject:
        Drop rows whose subject normalises to empty (a body with no query side).

    drop_annexure_only:
        Drop rows whose body is only a pointer to an enclosed annexure/table,
        with no substantive prose to retrieve.

    deduplicate:
        Drop repeat rows with the same normalised (subject, body).
    """

    minimum_subject_characters: int = 12

    minimum_body_characters: int = 40

    maximum_overlap: float = 0.7

    drop_superseded: bool = True

    drop_empty_subject: bool = True

    drop_annexure_only: bool = True

    deduplicate: bool = True

    @classmethod
    def naive(cls) -> TradeFilterConfig:
        """The oracle reference: every filter relaxed, so every row is kept."""

        return cls(
            minimum_subject_characters=0,
            minimum_body_characters=0,
            maximum_overlap=1.0,
            drop_superseded=False,
            drop_empty_subject=False,
            drop_annexure_only=False,
            deduplicate=False,
        )


@dataclass(slots=True)
class TradeStatistics:
    """
    Counts of what was produced and what was dropped, and why.

    The ``rejected`` map is the point: each key is a reason a row left the
    corpus, so a naive-vs-filtered diff reads as a per-reason ledger rather than
    a single opaque "fewer pairs" number.
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


def _normalize(text: str) -> str:
    """Strip, collapse internal whitespace. The one normalisation all text gets."""

    return _WHITESPACE.sub(" ", str(text or "")).strip()


def _superseded_reason(subject: str, body: str) -> str | None:
    """
    Name the way a notification is no longer operative, or ``None`` if it is live.

    Withdrawal and supersession are reported as distinct reasons so the ledger
    shows the shape of the feed rather than one merged "stale" bucket.
    """

    blob = f"{subject}\n{body}"

    if _WITHDRAWN.search(blob):
        return "withdrawn"

    if _SUPERSEDED.search(blob):
        return "superseded"

    return None


def _resolve_columns(fieldnames: Iterable[str]) -> dict[str, str]:
    """
    Map our field names to the file's actual headers, case-insensitively.

    Raises if the two columns we cannot do without — subject and body — are
    absent, so a renamed or wrong file fails loudly at the top rather than
    yielding an empty corpus.
    """

    present = {name.strip().casefold(): name for name in fieldnames if name}

    resolved: dict[str, str] = {}

    for field_name, header in _COLUMNS.items():
        actual = present.get(header.casefold())

        if actual is not None:
            resolved[field_name] = actual

    for required in ("subject", "body"):
        if required not in resolved:
            raise TradeReadError(
                "Notification file is missing a required column",
                required=_COLUMNS[required],
                found=sorted(present.values()),
            )

    return resolved


def read_notification_rows(source: str | Path) -> Iterator[TradeNotification]:
    """
    Stream raw :class:`TradeNotification` rows from a CSV (optionally gzipped).

    Reads ``.csv`` or ``.csv.gz`` by extension. Memory is bounded by one row.
    A row with an empty *body* is structurally absent — there is no passage to
    retrieve — and is dropped before it reaches the stats so even the naive
    population is over real rows. An empty *subject* is kept: its emptiness is a
    filter decision (counted), not missing data, so the naive reference can
    still see it. Discipline is :func:`iter_notification_pairs`'s job.
    """

    path = Path(source).expanduser()

    if not path.exists():
        raise TradeReadError("Notification file not found", path=str(path))

    # Government notifications embed tariff schedules and annexures whose single
    # Body cell can exceed Python's default 128 KiB CSV field cap (a real customs
    # 2026 row hit 130 KB). Raise the limit — halving down from sys.maxsize since
    # some platforms reject the raw value — so a large-but-legitimate body is read,
    # not crashed. Bounded by one row regardless; the cap only guards the parser.
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            break
        except OverflowError:
            limit //= 2

    opener = gzip.open if path.suffix == ".gz" else open

    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)

        if not reader.fieldnames:
            raise TradeReadError("Notification file has no header row", path=str(path))

        columns = _resolve_columns(reader.fieldnames)

        for record in reader:
            body = _normalize(record.get(columns["body"], ""))

            # No body means no passage — absent data, not a row to filter.
            if not body:
                continue

            yield TradeNotification(
                subject=_normalize(record.get(columns["subject"], "")),
                body=body,
                notification_number=_normalize(
                    record.get(columns.get("notification_number", ""), "")
                ),
                date=_normalize(record.get(columns.get("date", ""), "")),
                instrument=_normalize(record.get(columns.get("instrument", ""), "")),
                layer=_normalize(record.get(columns.get("layer", ""), "")),
                year=_normalize(record.get(columns.get("year", ""), "")),
            )


def iter_notification_pairs(
    rows: Iterable[TradeNotification],
    config: TradeFilterConfig | None = None,
    stats: TradeStatistics | None = None,
    *,
    language: str = _DEFAULT_NOTIFICATION_LANGUAGE,
) -> Iterator[MinedPair]:
    """
    Filter raw notification rows into clean subject→body :class:`MinedPair`s.

    Each surviving row yields one asymmetric pair — the short subject is the
    anchor, the long body the positive — carrying ``kind=TRADE_NOTIFICATION_KIND``
    so it trains with the E5 ``query:``/``passage:`` prefixes (see
    :func:`prefix_regime`). The pair's document id is the notification number
    (or the row index when absent), so it traces back to the source and a
    sampler can keep same-notification pairs apart.

    Every dropped row is counted by reason in ``stats`` (created if not passed
    and discarded), which is what makes the naive-vs-filtered oracle-diff a
    per-reason ledger. Each row takes exactly one path — a single reason, or
    accepted — so the counts reconcile: ``naive.produced == filtered.produced +
    sum(filtered.rejected)``.
    """

    config = config or TradeFilterConfig()

    stats = stats if stats is not None else TradeStatistics()

    language_code = normalize_language_code(language)

    seen: set[tuple[str, str]] = set()

    for index, row in enumerate(rows):
        subject = _normalize(row.subject)

        body = _normalize(row.body)

        # Superseded is checked first: a withdrawn rule is stale however long or
        # well-formed its subject, and the ledger is more useful naming it so.
        if config.drop_superseded:
            reason = _superseded_reason(subject, body)

            if reason is not None:
                stats._reject(reason)
                continue

        if config.drop_annexure_only and _ANNEXURE_ONLY.match(body):
            stats._reject("annexure_only")
            continue

        # Empty subject is named before the length floor so a body with no query
        # side reads as "empty_subject", not merely "short_subject".
        if config.drop_empty_subject and not subject:
            stats._reject("empty_subject")
            continue

        if len(subject) < config.minimum_subject_characters:
            stats._reject("short_subject")
            continue

        if len(body) < config.minimum_body_characters:
            stats._reject("short_body")
            continue

        if config.deduplicate:
            key = (subject.casefold(), body.casefold())
            if key in seen:
                stats._reject("duplicate")
                continue
            seen.add(key)

        # Asymmetric subject->body pairs: the subject is a HEADLINE drawn from the
        # body's own words, so subject->body containment is ~1.0 for every genuine
        # notification and is the SIGNAL WE WANT, not leakage. Real-data catch
        # (2026-08-15): a live DGFT trade notice scored 0.84 subject->body and was
        # wrongly cut. The trivial-pair signal is the REVERSE — a stub whose body is
        # merely the subject restated scores high on body->subject containment, while
        # a content-rich body scores near zero (the DGFT notice: 0.054). Measure that
        # direction, so the cap removes stubs, not real query/passage pairs.
        redundancy = token_overlap(body, subject)

        if redundancy > config.maximum_overlap:
            stats._reject("overlap")
            continue

        stats._accept(redundancy)

        document = row.notification_number or str(index)

        yield MinedPair(
            anchor=subject,
            positive=body,
            kind=TRADE_NOTIFICATION_KIND,
            document=f"{TRADE_SOURCE}:{document}",
            language=language_code,
            overlap=redundancy,
        )


def iter_bilingual_section_pairs(
    sections: Iterable[BilingualSection],
    *,
    english_language: str = _DEFAULT_ENGLISH_LANGUAGE,
    hindi_language: str = _DEFAULT_HINDI_LANGUAGE,
    source: str = TRADE_SOURCE,
    kind: str = TRADE_SECTION_XLING_KIND,
) -> Iterator[MinedPair]:
    """
    Emit cross-lingual :class:`MinedPair`s from aligned bilingual sections.

    Each aligned section yields **both directions** — English→Hindi and
    Hindi→English — because the register is symmetric: either language can be
    the query, so neither side is privileged. Both directions therefore train
    with **empty** E5 prefixes (``prefix_regime(TRADE_SECTION_XLING_KIND)`` is
    ``("", "")``), unlike the asymmetric notification pairs. Both directions
    share the section's document id, so a sampler never puts the two halves of
    one section in a batch as each other's negatives.

    A section with either side blank is not aligned and is skipped. The language
    arguments and ``source``/``kind`` are exposed so this helper is reused
    unchanged by gov-indic for its authoritative-bilingual showcase, not only by
    the trade Acts. The Layer A footing (§52(1)(q)(ii),
    :data:`TRADE_LEGAL_BASIS_ACT`) rides with these pairs on the model card.
    """

    english_code = normalize_language_code(english_language)

    hindi_code = normalize_language_code(hindi_language)

    for section in sections:
        english = _normalize(section.english)

        hindi = _normalize(section.hindi)

        if not english or not hindi:
            continue

        document = f"{source}:{section.identifier}"

        yield MinedPair(
            anchor=english,
            positive=hindi,
            kind=kind,
            document=document,
            language=english_code,
            overlap=token_overlap(english, hindi),
        )

        yield MinedPair(
            anchor=hindi,
            positive=english,
            kind=kind,
            document=document,
            language=hindi_code,
            overlap=token_overlap(hindi, english),
        )


class _SectionLike(Protocol):
    """
    The structural shape :func:`align_bilingual_sections` reads.

    Deliberately structural, not a concrete import: a
    :class:`.annotated_acts.StatuteSection` satisfies it, and so does anything a
    future bilingual reader (gov-indic's) yields carrying the same three
    attributes — the aligner stays decoupled from any one section reader.
    """

    number: str

    heading: str

    text: str


def _normalize_section_number(number: str) -> str:
    """
    The alignment key for a section number: whitespace-free and case-folded.

    Section numbers carry no internal whitespace, so all of it is stripped (not
    merely collapsed), and case is folded so an English ``"12A"`` keys the same
    as a Hindi reading's ``"12a"``. Empty in, empty out — a section with no
    number cannot be aligned.
    """

    return _WHITESPACE.sub("", str(number or "")).casefold()


@dataclass(slots=True)
class BilingualAlignmentStatistics:
    """
    Counts of the Hindi↔English section alignment, and why sections dropped.

    Alignment quality is *measured*, not assumed: an authoritative bilingual Act
    should align section-for-section, so a large ``english_only`` / ``hindi_only``
    tail is the signal that the two language readings are mis-keyed — a numbering
    drift, a dropped chapter — caught here as a ledger rather than surfacing later
    as thin cross-lingual coverage. The counts reconcile over the sections that
    carry a number: the distinct section numbers across both readings equal
    ``aligned`` plus the ``english_only`` / ``hindi_only`` / ``empty_side``
    entries in ``unmatched``. ``duplicate`` and ``no_number`` are parse-health
    counters *outside* that union — a section counted there never had a place in
    it.

    Attributes
    ----------
    aligned:
        Section numbers present in both readings with non-empty text on both
        sides — the ones that became a :class:`BilingualSection`.

    unmatched:
        Why every other section did not: ``english_only`` / ``hindi_only`` (the
        number is in one reading only), ``empty_side`` (in both, but one side is
        blank), ``duplicate`` (a section number a reading repeated — first wins),
        ``no_number`` (a section with no number to key on).
    """

    aligned: int = 0

    unmatched: Counter[str] = field(default_factory=Counter)

    def _align(self) -> None:
        self.aligned += 1

    def _drop(self, reason: str) -> None:
        self.unmatched[reason] += 1

    def to_dict(self) -> dict[str, Any]:
        """A JSON-friendly view for a build report."""

        return {"aligned": self.aligned, "unmatched": dict(self.unmatched)}


def _index_sections(
    sections: Iterable[_SectionLike],
    stats: BilingualAlignmentStatistics,
    side: str,
) -> dict[str, _SectionLike]:
    """
    Key one language's sections by normalised number, first occurrence winning.

    A section with no number cannot align and is counted ``no_number``; a
    repeated number is counted ``duplicate`` and ignored, so a mis-parse that
    doubles a section never silently doubles the corpus. ``side`` is accepted for
    symmetry with the caller and is not itself recorded — the reasons are shared
    across both readings so the ledger stays small.
    """

    indexed: dict[str, _SectionLike] = {}

    for section in sections:
        number = _normalize_section_number(section.number)

        if not number:
            stats._drop("no_number")
            continue

        if number in indexed:
            stats._drop("duplicate")
            continue

        indexed[number] = section

    return indexed


def align_bilingual_sections(
    english: Iterable[_SectionLike],
    hindi: Iterable[_SectionLike],
    *,
    act_identifier: str,
    stats: BilingualAlignmentStatistics | None = None,
) -> Iterator[BilingualSection]:
    """
    Align an Act's English and Hindi section readings into :class:`BilingualSection`s.

    The authoritative bilingual Acts carry the *same* section under the *same*
    number in both languages, so the section number — not the text — is the
    alignment key, stable across scripts. Each number present in both readings
    with non-empty text on both sides yields one :class:`BilingualSection` (ready
    for :func:`iter_bilingual_section_pairs`); every other section is counted in
    ``stats`` by reason (see :class:`BilingualAlignmentStatistics`) rather than
    silently dropped, so a build reads its cross-lingual coverage as a ledger.

    The section id is ``f"{act_identifier}:s.{number}"`` — the shape
    :class:`BilingualSection` documents — carrying the English reading's own
    number spelling (``"12A"``, not the folded key) for a readable provenance
    trail. Aligned sections are yielded in the English reading's order; the
    ``hindi_only`` tail is tallied after.

    Nothing here fetches or parses raw text — it aligns two already-parsed
    section streams — so it is offline and tested against in-memory fixtures, and
    the raw bilingual Act pull stays a box concern.
    """

    stats = stats if stats is not None else BilingualAlignmentStatistics()

    english_by_number = _index_sections(english, stats, "english")

    hindi_by_number = _index_sections(hindi, stats, "hindi")

    for number, english_section in english_by_number.items():
        hindi_section = hindi_by_number.get(number)

        if hindi_section is None:
            stats._drop("english_only")
            continue

        english_text = _normalize(english_section.text)

        hindi_text = _normalize(hindi_section.text)

        if not english_text or not hindi_text:
            stats._drop("empty_side")
            continue

        stats._align()

        heading = _normalize(english_section.heading) or _normalize(
            hindi_section.heading
        )

        yield BilingualSection(
            identifier=f"{act_identifier}:s.{_normalize(english_section.number)}",
            english=english_text,
            hindi=hindi_text,
            heading=heading,
        )

    for number in hindi_by_number:
        if number not in english_by_number:
            stats._drop("hindi_only")


def extract_notification_pairs(
    source: str | Path,
    output_path: str | Path,
    *,
    config: TradeFilterConfig | None = None,
    language: str = _DEFAULT_NOTIFICATION_LANGUAGE,
) -> TradeStatistics:
    """
    Read a notification CSV, filter it, and write clean pair records as JSON Lines.

    Writes gzip when ``output_path`` ends in ``.gz``. Returns the
    :class:`TradeStatistics` so the caller can report or oracle-diff it. The
    output is a pair file the trainer consumes directly — the same shape the
    KCC and statute mines write — so the next step is ``qfme adapt``.
    """

    output = Path(output_path).expanduser()

    output.parent.mkdir(parents=True, exist_ok=True)

    opener = gzip.open if output.suffix == ".gz" else open

    stats = TradeStatistics()

    with opener(output, "wt", encoding="utf-8") as handle:
        for pair in iter_notification_pairs(
            read_notification_rows(source), config, stats, language=language
        ):
            handle.write(json.dumps(pair.to_record(), ensure_ascii=False))

            handle.write("\n")

    _logger.info(
        "Wrote trade notification pair corpus",
        extra={
            "pairs": stats.produced,
            "rejected": dict(stats.rejected),
            "path": str(output),
            "source": TRADE_SOURCE,
        },
    )

    return stats
