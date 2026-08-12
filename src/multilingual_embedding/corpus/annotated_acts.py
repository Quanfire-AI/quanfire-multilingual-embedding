"""
Reading Indian Central Acts into the training corpus.

This is a **training front door** for the statutory-law domain, the
counterpart to :mod:`.judgments` (case law) and :mod:`.wikipedia` (general
text). Where :mod:`.judgments` reads court *judgments* from PDF, this reads
the *statutes themselves* — the bare Acts enacted by Parliament — from the
one clean, structured source that already exists: the **annotated Central
Acts dataset** (Zenodo record 5088102), 858 Central Acts extracted from
India Code into section-level JSON and released under **CC-BY-4.0**.

**Why this source and not a scrape of India Code.** India Code serves
statute text as HTML behind a bot-blocking front end, one section at a time,
with the section text wrapped in navigational chrome that poisons a corpus
if mistaken for the Act. The Zenodo dataset is that same India Code text,
already de-chromed and structured into ``(heading, paragraphs)`` per
section — so v1 needs no scraper, no OCR, and no chrome-stripping guesswork.
Its licence is CC-BY-4.0, an attribution grant, so the credit line rides on
every record and is carried through to the model card; nothing else is owed.

**The statutory footing, stated honestly.** The underlying Acts are
admitted under the Copyright Act 1957 **§52(1)(q)(ii)** — an Act of a
legislature is reproducible "together with any commentary thereon or any
other original matter", and bare statute stripped of commentary is not
squarely inside that exception. For a **non-reconstructive embedding** (the
model emits vectors, never the text) this is very likely moot: no
reproduction or publication happens at train or inference time, so the
condition that attaches to *reproduction* never fires. The flag rides on
every record rather than being waved away, and the discipline is: **train on
this, do not redistribute the bare-Act corpus itself.** We ship weights, not
the corpus.

**The one thing this module must get right: finding every section.** The
dataset does not store sections uniformly. An Act divides its sections under
one of three containers — a top-level ``Sections`` map (433 Acts), a
``Chapters`` map whose chapters each hold ``Sections`` (366 Acts), or a
``Parts`` map likewise (59 Acts) — and some Acts nest a little further. A
reader that only looks under ``Parts`` finds sections in 56 Acts and reports
the other 802 as empty, silently training on ~8% of the corpus with no error
ever raised. So section discovery here is **recursive**: it collects every
``Sections`` map at any depth, and the count it produces is pinned by an
oracle-diff test against a deliberately naive one-container reference, so the
gap between "sections a flat reader sees" and "sections that exist" is loud,
not silent. See :mod:`.oracle_diff` discipline — a spec-shaped reader whose
structural assumption is unchecked is exactly the failure that rule targets.

Nothing here fetches. It reads a directory of Act JSON (or the dataset zip
unpacked) that a caller has already downloaded, so the module is offline and
is tested against JSON fixtures, never a live portal.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from multilingual_embedding.core.exceptions import MultilingualEmbeddingError
from multilingual_embedding.core.logging import get_logger

from .language import normalize_language_code

__all__ = [
    "ANNOTATED_ACTS_ATTRIBUTION",
    "ANNOTATED_ACTS_LEGAL_BASIS",
    "ANNOTATED_ACTS_LICENSE",
    "ANNOTATED_ACTS_SOURCE",
    "StatuteDocument",
    "StatuteReadError",
    "StatuteSection",
    "extract_annotated_acts",
    "iter_sections",
    "read_annotated_acts",
]

_logger = get_logger(__name__)

# Recorded on every record. Constants rather than literals because each
# carries weight. The source is what the audit groups by. The licence is a
# CC-BY-4.0 attribution grant, so — unlike judgments' bare §52 public domain
# — a credit line is owed and must survive into the model card; it is carried
# on every record so a mechanical pipeline can never drop it. The legal basis
# is the caveated §52(1)(q)(ii) footing, flagged rather than assumed clean.
ANNOTATED_ACTS_SOURCE = "india-code-central-acts"

ANNOTATED_ACTS_LICENSE = "CC-BY-4.0"

# The attribution the CC-BY grant obliges us to carry. Kept verbatim so the
# model card can quote it without re-deriving the citation.
ANNOTATED_ACTS_ATTRIBUTION = (
    "Central Acts text from the annotated Central Acts dataset "
    "(Zenodo record 5088102, CC-BY-4.0), derived from India Code "
    "(indiacode.nic.in), Legislative Department, Government of India."
)

# The statutory footing, flagged on every record. Bare-Act reproduction sits
# under §52(1)(q)(ii)'s commentary condition; a non-reconstructive embedding
# does not reproduce, so this is train-safe, and the corpus itself is not to
# be redistributed. This string is the record's honest self-description.
ANNOTATED_ACTS_LEGAL_BASIS = (
    "Copyright Act 1957 s.52(1)(q)(ii); train-safe for a non-reconstructive "
    "embedding (emits vectors, not text); do not redistribute the bare-Act corpus"
)

# Central Acts are English; the authoritative Hindi versions are a separate
# fast-follow corpus, not this file. Recorded, not guessed.
_DEFAULT_LANGUAGE = "en"

# The keys an Act JSON uses for its three section containers. Sections live
# under exactly one of these per Act (and sometimes nested), which is why
# discovery is recursive rather than keyed to one of them.
_SECTIONS_KEY = "Sections"

_HEADING_KEY = "heading"

_PARAGRAPHS_KEY = "paragraphs"


class StatuteReadError(MultilingualEmbeddingError):
    """Raised when an Act JSON cannot be read or is not shaped like one."""


@dataclass(slots=True)
class StatuteSection:
    """
    One section of an Act: its marginal heading and its text.

    Attributes
    ----------
    number:
        The section label as the dataset gives it, e.g. ``"Section 5A."``.
        Carried for traceability; not part of a mined pair.

    heading:
        The section's marginal note — "Short title", "Definitions",
        "Punishment for criminal conspiracy". This is the query side of a
        heading/section training pair.

    text:
        The section's body, its numbered paragraphs joined in order. The
        passage side of the pair.
    """

    number: str

    heading: str

    text: str


@dataclass(slots=True)
class StatuteDocument:
    """
    One Central Act, flattened to its sections.

    The document's ``text`` is every section body joined, which is what
    title/lead and adjacency mining read; its ``sections`` drive the
    heading/section pairs that are the point of a statutory retriever.

    Attributes
    ----------
    identifier:
        The Act's identifier from the dataset (its ``Act ID``, e.g.
        ``"ACT NO. 31 OF 1952"``), so a record traces back to the Act.

    title:
        The Act's title, taken from the dataset, never read off the prose.

    sections:
        The Act's sections in document order.

    language:
        The ISO code recorded; English for Central Acts unless a caller
        says otherwise.

    enactment_date:
        The Act's enactment date string, carried for provenance.
    """

    identifier: str

    title: str

    sections: list[StatuteSection]

    language: str = _DEFAULT_LANGUAGE

    enactment_date: str | None = None

    @property
    def text(self) -> str:
        """Every section body, joined — the prose title/lead and adjacency read."""

        return "\n\n".join(section.text for section in self.sections if section.text)

    def to_record(self) -> dict[str, Any]:
        """
        Reduce to the JSON Lines record the corpus reader expects.

        The ``sections`` list is the one the reader lifts into document
        attributes and :func:`.pairs.iter_pairs` reads for heading/section
        pairs. Licence, attribution and legal basis ride on every record so
        no downstream stage can strip the obligations off the text.
        """

        return {
            "id": self.identifier,
            "language": self.language,
            "title": self.title,
            "text": self.text,
            "source": ANNOTATED_ACTS_SOURCE,
            "license": ANNOTATED_ACTS_LICENSE,
            "attribution": ANNOTATED_ACTS_ATTRIBUTION,
            "legal_basis": ANNOTATED_ACTS_LEGAL_BASIS,
            "enactment_date": self.enactment_date,
            "sections": [
                {"heading": section.heading, "text": section.text}
                for section in self.sections
                if section.heading and section.text
            ],
        }


def _flatten_paragraphs(paragraphs: Any) -> str:
    """
    Join a section's paragraphs into one body string, in order.

    The dataset stores paragraphs as a string-keyed map (``{"0": "...",
    "1": "..."}``) whose values are usually strings but occasionally nest a
    further map of sub-clauses. Keys are ordered numerically where they are
    numeric, so ``"10"`` follows ``"9"`` rather than sorting as text, and any
    nested map is flattened recursively so a sub-clause is not dropped.
    """

    if isinstance(paragraphs, str):
        return paragraphs.strip()

    if not isinstance(paragraphs, dict):
        return ""

    def sort_key(item: tuple[str, Any]) -> tuple[int, str]:
        key = item[0]
        return (int(key), "") if key.lstrip("-").isdigit() else (10**9, key)

    parts: list[str] = []

    for _, value in sorted(paragraphs.items(), key=sort_key):
        flattened = _flatten_paragraphs(value)

        if flattened:
            parts.append(flattened)

    return "\n".join(parts).strip()


def iter_sections(node: Any) -> Iterator[StatuteSection]:
    """
    Yield every section under ``node``, at any depth.

    An Act divides its sections under a top-level ``Sections`` map, or under
    ``Chapters``/``Parts`` whose members hold ``Sections`` maps, or a nesting
    of those. This walks the whole structure and collects **every**
    ``Sections`` map it finds, so no Act's sections are missed because it
    happened to use a container the reader did not special-case. That
    completeness is the property the oracle-diff test pins.
    """

    if isinstance(node, dict):
        for key, value in node.items():
            if key == _SECTIONS_KEY and isinstance(value, dict):
                for number, section in value.items():
                    if not isinstance(section, dict):
                        continue

                    heading = str(section.get(_HEADING_KEY) or "").strip()

                    text = _flatten_paragraphs(section.get(_PARAGRAPHS_KEY))

                    yield StatuteSection(number=str(number), heading=heading, text=text)
            else:
                yield from iter_sections(value)

    elif isinstance(node, list):
        for item in node:
            yield from iter_sections(item)


def _read_act(data: dict[str, Any]) -> StatuteDocument:
    """Turn one parsed Act JSON into a :class:`StatuteDocument`."""

    title = str(data.get("Act Title") or "").strip()

    identifier = str(data.get("Act ID") or "").strip() or title

    enactment = str(data.get("Enactment Date") or "").strip() or None

    sections = list(iter_sections(data))

    return StatuteDocument(
        identifier=identifier,
        title=title,
        sections=sections,
        language=_DEFAULT_LANGUAGE,
        enactment_date=enactment,
    )


def _act_paths(path: Path) -> list[Path]:
    """Resolve a path to the Act JSON files under it, sorted for reproducibility."""

    if path.is_dir():
        found = sorted(path.glob("*.json"))

        if not found:
            raise StatuteReadError("No .json Act files found in directory", path=str(path))

        return found

    if not path.exists():
        raise StatuteReadError("Act JSON not found", path=str(path))

    return [path]


def read_annotated_acts(
    source: str | Path,
    *,
    language: str = _DEFAULT_LANGUAGE,
    minimum_sections: int = 1,
    limit: int | None = None,
) -> Iterator[StatuteDocument]:
    """
    Stream Central Acts from a directory of Act JSON (or a single file).

    Memory is bounded by one Act, so this runs over the whole 858-Act
    dataset on a laptop.

    Parameters
    ----------
    source:
        A ``.json`` Act file, or a directory of them (the unpacked dataset).

    language:
        The ISO code recorded on every Act. Central Acts are English; a
        Hindi authoritative-text corpus is a separate source, so this
        defaults to English and is asserted, not inferred.

    minimum_sections:
        An Act yielding fewer sections than this is dropped and counted. The
        floor is there to catch a *parse* failure — an Act whose sections a
        reader could not find is not a short Act, it is a failed one — not to
        filter genuinely small Acts, so it sits at one.

    limit:
        Stop after this many Acts. For trying the dataset before committing.

    Raises
    ------
    StatuteReadError
        If the path holds no Act JSON, or a file is not valid JSON.
    """

    language_code = normalize_language_code(language)

    produced = 0

    skipped_empty = 0

    for act_path in _act_paths(Path(source).expanduser()):
        try:
            data = json.loads(act_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise StatuteReadError(
                "Act file is not valid JSON",
                path=str(act_path),
                detail=str(error),
            ) from error

        if not isinstance(data, dict):
            raise StatuteReadError("Act JSON is not an object", path=str(act_path))

        document = _read_act(data)

        document.language = language_code

        # Too few sections means discovery failed, not that the Act was
        # small. Dropped and counted rather than emitted: an Act with no
        # sections reads downstream as text with no training pairs and
        # silently shrinks the corpus.
        usable = [section for section in document.sections if section.heading and section.text]

        if len(usable) < minimum_sections:
            skipped_empty += 1

            continue

        produced += 1

        yield document

        if limit is not None and produced >= limit:
            break

    _logger.info(
        "Read Central Acts",
        extra={
            "acts": produced,
            "skipped_no_sections": skipped_empty,
            "source": ANNOTATED_ACTS_SOURCE,
        },
    )


def extract_annotated_acts(
    source: str | Path,
    output_path: str | Path,
    *,
    language: str = _DEFAULT_LANGUAGE,
    minimum_sections: int = 1,
    limit: int | None = None,
) -> int:
    """
    Read Central Acts and write them as a JSON Lines corpus file.

    Writes gzip when ``output_path`` ends in ``.gz``. Returns the number of
    Acts written. The output is the corpus record
    :class:`~multilingual_embedding.corpus.reader.CorpusReader` consumes, so
    the very next step is mining pairs from it — the ``sections`` on each
    record become heading/section pairs.
    """

    output = Path(output_path).expanduser()

    output.parent.mkdir(parents=True, exist_ok=True)

    opener = gzip.open if output.suffix == ".gz" else open

    written = 0

    with opener(output, "wt", encoding="utf-8") as handle:
        for document in read_annotated_acts(
            source, language=language, minimum_sections=minimum_sections, limit=limit
        ):
            handle.write(json.dumps(document.to_record(), ensure_ascii=False))

            handle.write("\n")

            written += 1

    _logger.info(
        "Wrote Central Acts corpus",
        extra={"acts": written, "path": str(output)},
    )

    return written
