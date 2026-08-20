"""
Reading Kisan Call Centre (KCC) transcripts into the training corpus.

This is a **training front door** for the agriculture-advisory domain, the
counterpart to :mod:`.annotated_acts` (statutes) and :mod:`.judgments` (case
law). Where those read documents that must be *mined* into pairs by their
structure, KCC arrives **already pair-shaped**: every row of the Kisan Call
Centre dataset is a farmer's *query* and the answer a call-centre agent gave.
The retrieval pair is the row itself — anchor = query, positive = answer — so
this module does not mine structure, it **reads, normalises, and filters**.
The emitted records are :class:`~multilingual_embedding.corpus.pairs.MinedPair`
records written directly (``kind="kcc_qa"``), the "hand-written pair file"
path that :meth:`MinedPair.from_record` was built to accept for a domain
nobody has mined.

**The source and its licence.** The KCC transcripts are published on
data.gov.in (Department of Agriculture & Farmers Welfare) under the
**Government Open Data License - India (GODL-India)**: a worldwide,
royalty-free grant to *use, adapt, publish, translate, and create derivative
works for all lawful commercial and non-commercial purposes*, attribution-only,
**with no ShareAlike clause** (verified against the licence text, 2026-08-14).
For a non-reconstructive embedding this clears all four intake gates — free,
clean, commercially-safe, fit — with attribution the only obligation, carried
on every record so a mechanical pipeline can never strip it.

**The one thing this module must get right: the noise.** KCC is call-centre
exhaust, not an edited corpus. A naive "row in, pair out" reader trains on:
- **test / junk queries** — ``"test"``, ``"asdf"``, bare digits, phone
  numbers, a lone ``"ok"``;
- **boilerplate answers** — ``"contact your nearest KVK"``,
  ``"information provided to farmer"``, ``"solved"`` — the same handful of
  template replies repeated across tens of thousands of rows. A pair whose
  answer is a top-frequency template teaches the model that *every* query maps
  to that one string, actively collapsing the space it is supposed to spread;
- **exact duplicates** — the same query/answer logged many times;
- **query-in-answer leakage** — the answer restating the query, a pair a
  string-matcher wins without learning meaning.

So the value here is the **filter**, and — per the oracle-diff house rule — its
effect is *measured*, not asserted: :func:`iter_kcc_pairs` records why each
dropped row was dropped into a :class:`KccStatistics`, so a build can diff the
naive population against the filtered one and see the junk leave. What the
filter must **not** do is over-clean: KCC queries are terse, romanised, and
code-mixed (``"leaf yellow paddy control"``), and terse is not junk — the same
lesson the statute corpus learned. The character floors sit low on purpose.

**Scope honesty (rides to the model card).** KCC text is predominantly
romanised/transliterated English and code-mixed, **not** native-script Indic
paragraphs. This trains *agriculture-advisory intent retrieval*; it is not, on
its own, a native-script Indic model. That caveat is a carding matter, flagged
here so it is not quietly forgotten downstream.

Nothing here fetches. It reads CSV rows a caller has already downloaded, so the
module is offline and is tested against fixtures, never a live portal.
"""

from __future__ import annotations

import csv
import gzip
import json
import re
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from multilingual_embedding.core.exceptions import MultilingualEmbeddingError
from multilingual_embedding.core.logging import get_logger

from .language import normalize_language_code
from .pairs import MinedPair, token_overlap

__all__ = [
    "KCC_ATTRIBUTION",
    "KCC_CSV_FIELDNAMES",
    "KCC_LEGAL_BASIS",
    "KCC_LICENSE",
    "KCC_SOURCE",
    "KccFilterConfig",
    "KccReadError",
    "KccRow",
    "KccStatistics",
    "extract_kcc_pairs",
    "iter_kcc_pairs",
    "read_kcc_rows",
]

_logger = get_logger(__name__)

# Recorded on every record. GODL-India is an attribution grant with no
# ShareAlike, so — like CC-BY statutes and unlike bare §52 public domain — a
# credit line is owed and must survive into the model card; it rides on every
# record so no downstream stage can drop it.
KCC_SOURCE = "kisan-call-centre"

KCC_LICENSE = "GODL-India"

# The attribution the GODL grant obliges us to carry, kept verbatim so the
# model card can quote it without re-deriving the citation. The concrete
# resource URL/DOI of the specific KCC release is appended by the caller that
# knows which slice it pulled; this is the standing portion.
KCC_ATTRIBUTION = (
    "Kisan Call Centre transcripts, Department of Agriculture & Farmers "
    "Welfare, Government of India, via data.gov.in, under the Government Open "
    "Data License - India (GODL-India)."
)

# GODL-India is a clean commercial+derivative grant, so unlike the statute
# corpus there is no reproduction caveat to flag — the honest self-description
# is simply the licence and its attribution-only condition.
KCC_LEGAL_BASIS = (
    "GODL-India: commercial use, adaptation and translation permitted, "
    "attribution required, no ShareAlike"
)

# KCC queries are overwhelmingly romanised/code-mixed English. Recorded, not
# guessed; a native-script slice would carry its own code.
_DEFAULT_LANGUAGE = "en"

# Column names as the data.gov.in KCC resource ships them. Kept as a mapping
# from our field to the source header so a release that renames a column is a
# one-line fix, not a reader rewrite. Values are matched case-insensitively.
_COLUMNS = {
    "query": "QueryText",
    "answer": "KccAns",
    "state": "StateName",
    "district": "DistrictName",
    "crop": "Crop",
    "category": "Category",
    "query_type": "QueryType",
    "created_on": "CreatedOn",
}

# The source header set, in order — the CSV the box crawl writes and
# :func:`read_kcc_rows` reads. Exposed so the crawler (:mod:`.kcc_crawl`) and
# the reader cannot drift on column names: both name the schema from one place.
KCC_CSV_FIELDNAMES = tuple(_COLUMNS.values())

# Queries that are pure noise regardless of length: known test tokens and
# content-free acknowledgements. Matched against the whitespace-collapsed,
# case-folded query. Terse *agricultural* queries are deliberately NOT here —
# "leaf yellow" is short but real; the junk set is only content-free strings.
_JUNK_QUERIES = frozenset(
    {
        "test",
        "testing",
        "asdf",
        "ok",
        "okay",
        "hello",
        "hi",
        "yes",
        "no",
        "na",
        "nil",
        "none",
        "abc",
        "xyz",
        "query",
        "question",
    }
)

# A query that is only digits / punctuation / whitespace carries no retrievable
# text (phone numbers, "123", "...."). Matched after normalisation.
_NON_TEXT_QUERY = re.compile(r"^[\W\d_]+$", re.UNICODE)

_WHITESPACE = re.compile(r"\s+", re.UNICODE)


class KccReadError(MultilingualEmbeddingError):
    """Raised when a KCC file cannot be read or lacks the expected columns."""


@dataclass(slots=True)
class KccRow:
    """
    One raw KCC transcript row, before filtering.

    Attributes
    ----------
    query:
        The farmer's question as logged — the anchor of the pair.

    answer:
        The agent's reply — the positive of the pair.

    state, district, crop, category, query_type:
        Provenance/facet fields, carried for traceability and for facet-sliced
        evaluation (retrieval quality per crop, per state). Not part of the
        trained text.

    created_on:
        The log timestamp string, carried for provenance only.
    """

    query: str

    answer: str

    state: str = ""

    district: str = ""

    crop: str = ""

    category: str = ""

    query_type: str = ""

    created_on: str = ""


@dataclass(slots=True)
class KccFilterConfig:
    """
    Knobs for the KCC noise filter.

    Defaults are the disciplined pass. A *naive* reference — the oracle for the
    diff — is this config with every filter relaxed
    (:meth:`naive`), so a build can measure exactly what discipline removes.

    Attributes
    ----------
    minimum_query_characters:
        Floor on the query length. Low on purpose: KCC queries are terse and
        code-mixed, and terse is not junk. This drops ``"ok"``, not
        ``"leaf yellow"``.

    minimum_answer_characters:
        Floor on the answer length. Drops ``"solved"`` / ``"yes"`` answers that
        are not retrieval targets.

    maximum_overlap:
        Reject a pair whose query-to-answer token overlap exceeds this — the
        answer merely restates the query, a string-solvable pair. ``1.0``
        disables the check.

    boilerplate_answer_count:
        An answer that occurs at least this many times across the input is a
        template reply (``"contact your nearest KVK"``), not a query-specific
        target, and every row carrying it is dropped. ``0`` disables the check.
        This is the KCC analogue of the statute overlap-cap: the discipline
        that stops the model gaming a frequent constant.

    drop_junk_queries:
        Drop content-free queries (the ``_JUNK_QUERIES`` set and
        digit/punctuation-only strings).

    deduplicate:
        Drop repeat rows with the same normalised (query, answer).
    """

    minimum_query_characters: int = 8

    minimum_answer_characters: int = 20

    maximum_overlap: float = 0.7

    boilerplate_answer_count: int = 50

    drop_junk_queries: bool = True

    deduplicate: bool = True

    @classmethod
    def naive(cls) -> KccFilterConfig:
        """The oracle reference: every filter relaxed, only empties removed."""

        return cls(
            minimum_query_characters=1,
            minimum_answer_characters=1,
            maximum_overlap=1.0,
            boilerplate_answer_count=0,
            drop_junk_queries=False,
            deduplicate=False,
        )


@dataclass(slots=True)
class KccStatistics:
    """
    Counts of what was produced and what was dropped, and why.

    The ``rejected`` map is the point: each key is a reason a row left the
    corpus, so a naive-vs-filtered diff reads as a per-reason ledger rather
    than a single opaque "fewer pairs" number.
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


def _is_junk_query(query_normalized: str) -> bool:
    """True for content-free queries: known test tokens, digit/punct-only."""

    folded = query_normalized.casefold()

    if folded in _JUNK_QUERIES:
        return True

    return bool(_NON_TEXT_QUERY.match(query_normalized))


def _resolve_columns(fieldnames: Iterable[str]) -> dict[str, str]:
    """
    Map our field names to the file's actual headers, case-insensitively.

    Raises if the two columns we cannot do without — query and answer — are
    absent, so a renamed or wrong file fails loudly at the top rather than
    yielding an empty corpus.
    """

    present = {name.strip().casefold(): name for name in fieldnames if name}

    resolved: dict[str, str] = {}

    for field_name, header in _COLUMNS.items():
        actual = present.get(header.casefold())

        if actual is not None:
            resolved[field_name] = actual

    for required in ("query", "answer"):
        if required not in resolved:
            raise KccReadError(
                "KCC file is missing a required column",
                required=_COLUMNS[required],
                found=sorted(present.values()),
            )

    return resolved


def read_kcc_rows(source: str | Path) -> Iterator[KccRow]:
    """
    Stream raw :class:`KccRow` objects from a KCC CSV (optionally gzipped).

    Reads ``.csv`` or ``.csv.gz`` by extension. Memory is bounded by one row,
    so this streams an arbitrarily large release. No filtering happens here —
    every non-empty-query/answer row is yielded; discipline is
    :func:`iter_kcc_pairs`'s job so the naive population stays measurable.
    """

    path = Path(source).expanduser()

    if not path.exists():
        raise KccReadError("KCC file not found", path=str(path))

    opener = gzip.open if path.suffix == ".gz" else open

    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)

        if not reader.fieldnames:
            raise KccReadError("KCC file has no header row", path=str(path))

        columns = _resolve_columns(reader.fieldnames)

        for record in reader:
            query = _normalize(record.get(columns["query"], ""))

            answer = _normalize(record.get(columns.get("answer", ""), ""))

            # An empty query or answer is not a row we can filter or train —
            # it is simply absent data, dropped before it reaches the stats so
            # even the naive population is over real rows.
            if not query or not answer:
                continue

            yield KccRow(
                query=query,
                answer=answer,
                state=_normalize(record.get(columns.get("state", ""), "")),
                district=_normalize(record.get(columns.get("district", ""), "")),
                crop=_normalize(record.get(columns.get("crop", ""), "")),
                category=_normalize(record.get(columns.get("category", ""), "")),
                query_type=_normalize(record.get(columns.get("query_type", ""), "")),
                created_on=_normalize(record.get(columns.get("created_on", ""), "")),
            )


def iter_kcc_pairs(
    rows: Iterable[KccRow],
    config: KccFilterConfig | None = None,
    stats: KccStatistics | None = None,
    *,
    language: str = _DEFAULT_LANGUAGE,
) -> Iterator[MinedPair]:
    """
    Filter raw rows into clean query→answer :class:`MinedPair` records.

    The boilerplate check needs the whole answer-frequency distribution, so the
    input is materialised once. KCC releases are per-state/per-period slices of
    a few hundred thousand rows — comfortably in memory on a laptop — and a
    caller wanting a bounded footprint slices before calling.

    ``language`` is the ISO code stamped on every emitted pair (``MinedPair`` is
    frozen, so it is set at construction, never after). Every dropped row is
    counted by reason in ``stats`` (created if not passed and discarded), which
    is what makes the naive-vs-filtered oracle-diff a per-reason ledger.
    """

    config = config or KccFilterConfig()

    stats = stats if stats is not None else KccStatistics()

    language_code = normalize_language_code(language)

    materialised = list(rows)

    # Answer frequency across the whole input, for the boilerplate cut. Keyed
    # on the case-folded answer so "Solved." and "solved" count as one template.
    if config.boilerplate_answer_count > 0:
        answer_counts = Counter(row.answer.casefold() for row in materialised)
    else:
        answer_counts = Counter()

    seen: set[tuple[str, str]] = set()

    for index, row in enumerate(materialised):
        # Junk is checked before the length floors so a content-free query is
        # named "junk_query", not merely "short_query": "test" and a bare phone
        # number are noise regardless of how many characters they run to, and
        # the ledger is more useful when it says so.
        if config.drop_junk_queries and _is_junk_query(row.query):
            stats._reject("junk_query")
            continue

        if len(row.query) < config.minimum_query_characters:
            stats._reject("short_query")
            continue

        if len(row.answer) < config.minimum_answer_characters:
            stats._reject("short_answer")
            continue

        if (
            config.boilerplate_answer_count > 0
            and answer_counts[row.answer.casefold()] >= config.boilerplate_answer_count
        ):
            stats._reject("boilerplate_answer")
            continue

        if config.deduplicate:
            key = (row.query.casefold(), row.answer.casefold())
            if key in seen:
                stats._reject("duplicate")
                continue
            seen.add(key)

        overlap = token_overlap(row.query, row.answer)

        if overlap > config.maximum_overlap:
            stats._reject("overlap")
            continue

        stats._accept(overlap)

        yield MinedPair(
            anchor=row.query,
            positive=row.answer,
            kind="kcc_qa",
            document=f"{KCC_SOURCE}:{index}",
            language=language_code,
            overlap=overlap,
        )


def extract_kcc_pairs(
    source: str | Path,
    output_path: str | Path,
    *,
    config: KccFilterConfig | None = None,
    language: str = _DEFAULT_LANGUAGE,
) -> KccStatistics:
    """
    Read a KCC CSV, filter it, and write clean pair records as JSON Lines.

    Writes gzip when ``output_path`` ends in ``.gz``. Returns the
    :class:`KccStatistics` so the caller can report or oracle-diff it. The
    output is a pair file the trainer consumes directly — the same shape the
    statute mine writes — so the next step is ``qfme adapt``.
    """

    output = Path(output_path).expanduser()

    output.parent.mkdir(parents=True, exist_ok=True)

    opener = gzip.open if output.suffix == ".gz" else open

    stats = KccStatistics()

    with opener(output, "wt", encoding="utf-8") as handle:
        for pair in iter_kcc_pairs(read_kcc_rows(source), config, stats, language=language):
            handle.write(json.dumps(pair.to_record(), ensure_ascii=False))

            handle.write("\n")

    _logger.info(
        "Wrote KCC pair corpus",
        extra={
            "pairs": stats.produced,
            "rejected": dict(stats.rejected),
            "path": str(output),
            "source": KCC_SOURCE,
        },
    )

    return stats
