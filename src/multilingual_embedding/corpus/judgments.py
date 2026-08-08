"""
Reading public court judgments into the training corpus.

This is a **training front door** for the legal domain, the counterpart
to :mod:`.wikipedia` for general text and the opposite end of the wall
from :mod:`.milpac`. Indian court judgments are **public domain by
statute** — the Copyright Act 1957 **§52(1)(q)** places the text of any
court judgment or order outside copyright — so no attribution obligation
and no rights-holder attaches to the prose itself, which is exactly why
judgments train and MILPaC (non-commercial) only scores. That statutory
footing is stronger than a Creative Commons grant: there is no licensor
who could later be found not to have held the right. It binds one thing
in return — the PDFs must come from an **official court portal** (the
Supreme Court / High Court sites, the eCourts services), never a
commercial reporter such as SCC, whose headnotes, arrangement and
editorial matter are a separately copyrightable layer over the
public-domain judgment.

The format is the difficulty. Judgments arrive as **PDF**, one file per
case, with the full text in a text layer and nothing else reliable: no
language field, no clean title, no structure a parser can trust. So this
module does three honest things and no more:

* **It names the language rather than guessing it.** Indian judgments are
  predominantly English; a caller who has Hindi judgments passes
  ``language="hi"``. The file does not say, so this does not pretend to
  know.
* **It takes the title from the file, not the prose.** A judgment's first
  line is a court name or a cause title in a layout that varies by court
  and decade; reading it off the text would be the guesswork the
  Wikipedia audit exists to catch. The PDF's metadata title is used when
  present, and the filename stem otherwise — both traceable, neither
  invented.
* **It drops a document whose extraction failed rather than emit it.** A
  scanned judgment has an *image* where its text should be, and its text
  layer is empty or nonsense. That is not a short document; it is a
  failed one, and training on the handful of characters an image PDF
  yields teaches nothing. The ``minimum_characters`` floor catches it,
  and the drop is counted and logged out loud, because a silent empty
  record reads as "extracted" when it was not.

**What this does not do, and says so:** it does not OCR. A scanned
judgment with no text layer cannot be recovered here, and this module
will drop it, not read it. Whether a given collection extracts to clean
prose or to image-only failures is a property of that collection that
**cannot be verified until it is downloaded and run through**
:mod:`.audit` — the same standard :mod:`.wikipedia` holds itself to.

``pypdf`` lives behind the optional ``judgments`` extra, imported inside
:func:`_pypdf_reader` so a core install still imports
:mod:`multilingual_embedding.corpus`. The PDF-reading step is a seam: it
is the one part whose fidelity depends on a real corpus, so it is kept
small, isolated, and overridable, and everything downstream of it —
filtering, tidying, the record shape — is exercised by fixtures that
never touch a PDF.
"""

from __future__ import annotations

import gzip
import json
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from multilingual_embedding.core.exceptions import MultilingualEmbeddingError
from multilingual_embedding.core.logging import get_logger

from .language import normalize_language_code

__all__ = [
    "JUDGMENTS_LICENSE",
    "JUDGMENTS_SOURCE",
    "JudgmentDocument",
    "JudgmentExtractionError",
    "PdfReader",
    "extract_judgments",
    "iter_judgments",
]

_logger = get_logger(__name__)

# Recorded on every record. Constants rather than literals because both
# carry weight: the source is what the audit groups by, and the statutory
# public-domain basis (§52(1)(q)) — no attribution obligation, no licensor
# who could later be found not to hold the right — is the reason this text
# may be trained on where MILPaC's non-commercial licence means it may only
# be scored. Its binding is the source: an official court portal, never a
# commercial reporter whose editorial layer is separately copyrightable.
JUDGMENTS_SOURCE = "judgments"

JUDGMENTS_LICENSE = "Public Domain (India, Copyright Act §52(1)(q))"

# The language every judgment is recorded as unless the caller says
# otherwise. English is the working default for Indian superior-court
# judgments; a collection in another language is read by passing its code,
# because the PDF does not state one worth trusting.
_DEFAULT_LANGUAGE = "en"

# A real judgment runs to thousands of characters. A document that
# extracts to fewer than this has almost certainly failed — a scanned page
# whose text layer is an image, or a PDF pypdf could not read — and is
# dropped and counted rather than trained on. Set well above a genuine
# short order so the floor separates failures from real brevity.
_DEFAULT_MINIMUM_CHARACTERS = 500

_BLANK_RUN = re.compile(r"\n{3,}")

_SPACE_RUN = re.compile(r"[ \t]{2,}")


# The PDF-reading seam. Given a path, return the document's metadata title
# (or ``None``) and its pages as a list of strings. This is the one piece
# whose behaviour depends on a real PDF, so it is a callable that can be
# swapped for a fixture in tests, keeping the extraction fidelity — the
# only unverifiable part — quarantined behind a boundary the rest of the
# module does not see through.
PdfReader = Callable[[Path], "tuple[str | None, list[str]]"]


class JudgmentExtractionError(MultilingualEmbeddingError):
    """Raised when a judgment source cannot be read or is not what it claims."""


@dataclass(slots=True)
class JudgmentDocument:
    """
    One judgment, extracted to prose.

    Attributes
    ----------
    identifier:
        The source file's stem, so a record traces back to the exact PDF
        it came from and a regenerated corpus lines up with itself.

    title:
        The PDF's metadata title when it carries one, else the filename
        stem. Taken from the file rather than read off the first line of
        prose, whose layout no parser can trust across courts.

    text:
        The extracted prose — every page's text, joined and tidied.

    language:
        The ISO code the caller supplied. The PDF does not state one, so
        this is asserted by the caller, not inferred.
    """

    identifier: str

    title: str

    text: str

    language: str

    def to_record(self) -> dict[str, Any]:
        """Reduce to the JSON Lines record the corpus readers expect."""

        return {
            "id": self.identifier,
            "language": self.language,
            "title": self.title,
            "text": self.text,
            "source": JUDGMENTS_SOURCE,
            "license": JUDGMENTS_LICENSE,
        }


def _pypdf_reader(path: Path) -> tuple[str | None, list[str]]:
    """
    The default PDF-reading seam, backed by ``pypdf``.

    Returns the metadata title, if any, and the text of each page. A page
    pypdf cannot extract contributes an empty string rather than aborting
    the whole document, because one damaged page in a long judgment should
    not lose the rest.
    """

    try:
        from pypdf import PdfReader as _PdfReader
    except ImportError as error:
        raise JudgmentExtractionError(
            "Reading judgments needs the 'judgments' extra (uv sync --extra judgments)",
            missing="pypdf",
        ) from error

    try:
        reader = _PdfReader(str(path))
    except Exception as error:  # pypdf raises many types on a bad file
        raise JudgmentExtractionError(
            "Could not open judgment PDF",
            path=str(path),
            detail=str(error),
        ) from error

    title = None

    try:
        metadata = reader.metadata

        if metadata is not None and metadata.title:
            title = str(metadata.title).strip() or None
    except Exception:  # metadata is optional and frequently malformed
        title = None

    pages: list[str] = []

    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")

    return title, pages


def _tidy(text: str) -> str:
    """Collapse the whitespace damage PDF extraction leaves behind."""

    text = _SPACE_RUN.sub(" ", text)

    text = _BLANK_RUN.sub("\n\n", text)

    return "\n".join(line.strip() for line in text.splitlines()).strip()


def _pdf_paths(path: Path) -> list[Path]:
    """Resolve a path to the PDFs under it, sorted for reproducibility."""

    if path.is_dir():
        found = sorted(path.glob("*.pdf"))

        if not found:
            raise JudgmentExtractionError("No .pdf files found in directory", path=str(path))

        return found

    if not path.exists():
        raise JudgmentExtractionError("Judgment PDF not found", path=str(path))

    return [path]


def iter_judgments(
    source: str | Path,
    *,
    language: str = _DEFAULT_LANGUAGE,
    minimum_characters: int = _DEFAULT_MINIMUM_CHARACTERS,
    limit: int | None = None,
    reader: PdfReader | None = None,
) -> Iterator[JudgmentDocument]:
    """
    Stream extracted judgments from one PDF or a directory of them.

    Memory is bounded by one document, not by the collection, so this runs
    over thousands of judgments on a laptop.

    Parameters
    ----------
    source:
        A ``.pdf`` file, or a directory holding several. A directory reads
        every ``*.pdf`` under it, sorted.

    language:
        The ISO code recorded on every judgment. The PDF does not state
        one, so the caller supplies it; defaults to English.

    minimum_characters:
        Documents shorter than this are dropped and counted. The floor is
        there to catch failed extractions — a scanned judgment whose text
        layer is an image yields almost nothing — not to filter genuinely
        short orders, so it sits well above a real short judgment.

    limit:
        Stop after this many documents. For trying a collection before
        committing to it.

    reader:
        The PDF-reading seam, a callable from a path to
        ``(metadata_title, page_texts)``. Defaults to the ``pypdf``-backed
        reader; overridable so the filtering and record logic can be
        tested without a real PDF, and so a collection needing OCR could
        supply its own extractor without this module pretending to do it.

    Raises
    ------
    JudgmentExtractionError
        If ``pypdf`` is not installed, the path holds no PDFs, or a file
        cannot be opened.

    Example
    -------
    ::

        for judgment in iter_judgments("data/corpora/judgments/", limit=10):
            print(judgment.identifier, len(judgment.text))
    """

    read = reader or _pypdf_reader

    language_code = normalize_language_code(language)

    produced = 0

    skipped_failed = 0

    for pdf_path in _pdf_paths(Path(source).expanduser()):
        title, pages = read(pdf_path)

        text = _tidy("\n\n".join(page for page in pages if page))

        # Too little text means the extraction failed, not that the
        # judgment was short. Dropped and counted rather than emitted: an
        # empty record reads as a real document downstream and is not one.
        if len(text) < minimum_characters:
            skipped_failed += 1

            continue

        stem = pdf_path.stem

        produced += 1

        yield JudgmentDocument(
            identifier=stem,
            title=(title or stem),
            text=text,
            language=language_code,
        )

        if limit is not None and produced >= limit:
            break

    _logger.info(
        "Extracted judgments",
        extra={
            "judgments": produced,
            "skipped_failed_extraction": skipped_failed,
        },
    )

    if skipped_failed:
        # Said out loud rather than left in a log nobody reads. A silent
        # drop reads as "everything extracted" when a scanned collection
        # may have yielded almost nothing.
        _logger.warning(
            "Dropped judgments whose text could not be extracted (likely scanned images)",
            extra={
                "dropped": skipped_failed,
                "kept": produced,
                "share": round(skipped_failed / max(1, skipped_failed + produced), 5),
            },
        )


def extract_judgments(
    source: str | Path,
    output: str | Path,
    *,
    language: str = _DEFAULT_LANGUAGE,
    minimum_characters: int = _DEFAULT_MINIMUM_CHARACTERS,
    limit: int | None = None,
    reader: PdfReader | None = None,
) -> int:
    """
    Extract judgments to a JSON Lines corpus file and return the count.

    The output is gzipped when its name ends ``.gz``, the sensible default
    for a corpus of this size, and every downstream reader decompresses
    transparently. The result is a training corpus: run it through
    :mod:`.audit` before using it, because whether a given collection
    extracts cleanly cannot be known until it is read.

    Example
    -------
    ::

        extract_judgments("data/corpora/judgments/", "data/corpora/judgments.jsonl.gz")
    """

    destination = Path(output).expanduser()

    destination.parent.mkdir(parents=True, exist_ok=True)

    opener = gzip.open if destination.suffix == ".gz" else open

    count = 0

    with opener(destination, "wt", encoding="utf-8") as handle:
        for judgment in iter_judgments(
            source,
            language=language,
            minimum_characters=minimum_characters,
            limit=limit,
            reader=reader,
        ):
            handle.write(json.dumps(judgment.to_record(), ensure_ascii=False) + "\n")

            count += 1

    _logger.info(
        "Wrote judgment corpus",
        extra={"path": str(destination), "judgments": count},
    )

    return count
