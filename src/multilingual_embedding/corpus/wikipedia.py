"""
Extracting a Wikipedia dump into the corpus format.

Wikipedia is the only source of substantial text in most of the scheduled
Indian languages, so this is the front door for real training data. It is
also a hostile format: MediaWiki markup nests templates inside tables
inside references, and a naive extraction leaves enough of it behind to
train the tokenizer on syntax rather than on language.

That failure is quiet — it produces a corpus that loads, trains, and
yields a worse model — which is why :mod:`.audit` exists and why the
output of this module should always be run through it.

The markup itself is parsed by ``mwparserfromhell`` rather than by
regular expressions here. Hand-rolled wiki parsers are the canonical
example of a job that looks easy and is not, and shipping one would mean
shipping exactly the defect the audit is designed to catch.

Three things ``strip_code`` does not handle on its own, all corrected
below:

* **Tables** survive as their cell contents, so a statistics table
  arrives as a run of bare numbers indistinguishable from prose.
* **Headings** collapse to their text, so ``== History ==`` becomes a
  floating word in the middle of the article.
* **Leftover artefacts** — file captions, category names, stray entities
  — that no parser removes because they are legitimate wikitext.

Sections are preserved rather than flattened. A heading and the body
beneath it are a query and a passage, which is what Phase C needs to
manufacture training pairs, and the structure is free here but expensive
to recover later.
"""

from __future__ import annotations

import bz2
import gzip
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from multilingual_embedding.core.exceptions import MultilingualEmbeddingError
from multilingual_embedding.core.logging import get_logger
from multilingual_embedding.utils.hashing import hash_text

__all__ = [
    "WikipediaArticle",
    "WikipediaExtractionError",
    "extract_dump",
    "iter_articles",
]

_logger = get_logger(__name__)

# Namespace 0 is article space. Everything else — talk pages, templates,
# user pages, categories — is project machinery rather than prose.
_ARTICLE_NAMESPACE = "0"

# Sections that carry no prose worth training on. Matched case-insensitively
# against the heading, in English and in transliteration-friendly forms;
# a language whose headings differ simply keeps its sections, which is the
# safe direction to fail in.
_BOILERPLATE_HEADINGS = frozenset(
    {
        "references",
        "external links",
        "see also",
        "further reading",
        "bibliography",
        "notes",
        "sources",
        "citations",
        "gallery",
        "footnotes",
    }
)

# Tables, and the block-level HTML that survives template expansion.
_TABLE = re.compile(r"\{\|.*?\|\}", re.DOTALL)

_HTML_BLOCK = re.compile(
    r"<(table|div|gallery|ref|references|timeline)[^>]*>.*?</\1>", re.DOTALL | re.I
)

_SELF_CLOSING = re.compile(r"<[^>]+/>")

_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)

# `[[File:…]]` and `[[Category:…]]` in every language's own namespace name.
# The prefix varies, so this matches the shape rather than the word: a
# bracketed link whose target contains a colon and which carries caption
# pipes is a media embed, not a wikilink.
_MEDIA_LINK = re.compile(r"\[\[[^\[\]]*:[^\[\]]*(\|[^\[\]]*)*\]\]")

_HEADING = re.compile(r"^\s*(={2,6})\s*(.+?)\s*\1\s*$", re.MULTILINE)

_BLANK_RUN = re.compile(r"\n{3,}")

_SPACE_RUN = re.compile(r"[ \t]{2,}")

# Residue left after parsing, all of it from wikitext that is malformed in
# the source. Wikipedia contains genuinely broken markup — an unclosed
# `<div>`, a link written `[Foo]]`, a template someone never terminated —
# and no parser can resolve what the author did not write. About 1% of
# articles are affected.
_RESIDUAL_TAG = re.compile(r"</?[a-zA-Z][^>]{0,200}>")

_RESIDUAL_ENTITY = re.compile(r"&[a-zA-Z]{2,10};|&#\d{2,6};")

_RESIDUAL_BRACE = re.compile(r"\{\{[^{}]*\}\}")

# Markers whose presence in finished text means the extraction did not
# complete. Deliberately the same set the corpus audit looks for, so this
# module holds itself to the standard it will be judged by.
_MARKUP_MARKERS: tuple[str, ...] = (
    "[[",
    "]]",
    "{{",
    "}}",
    "<ref",
    "</ref>",
    "<div",
    "&nbsp;",
    "==",
)


def _remove_residue(text: str) -> str:
    """
    Strip what the parser could not, because the source was malformed.

    Templates are removed innermost-first and repeatedly: a nested
    `{{a|{{b}}}}` only becomes removable once its inner brace pair is
    gone, so a single pass would leave the outer one behind.
    """

    previous = None

    while previous != text:
        previous = text

        text = _RESIDUAL_BRACE.sub("", text)

    text = _RESIDUAL_TAG.sub("", text)

    return _RESIDUAL_ENTITY.sub(" ", text)


def _is_clean(text: str) -> bool:
    """True when no markup marker survives in the finished text."""

    return not any(marker in text for marker in _MARKUP_MARKERS)


class WikipediaExtractionError(MultilingualEmbeddingError):
    """Raised when a dump cannot be read or parsed."""


@dataclass(slots=True)
class WikipediaArticle:
    """
    One article, cleaned.

    Attributes
    ----------
    identifier:
        The wiki's own page id, stable across dumps, so a corpus can be
        regenerated and still line up with itself.

    title:
        Article title. Also the query side of a title/lead training pair.

    text:
        Clean prose — the lead, then each retained section's body.

    sections:
        ``(heading, body)`` pairs, retained because a heading and its
        section are a natural query and passage. Empty for a stub.

    language:
        The wiki's language code, supplied by the caller rather than
        inferred, since the dump does not reliably state it.
    """

    identifier: str

    title: str

    text: str

    language: str

    sections: list[tuple[str, str]] = field(default_factory=list)

    def to_record(self) -> dict[str, Any]:
        """Reduce to the JSON Lines record the corpus readers expect."""

        record: dict[str, Any] = {
            "id": self.identifier,
            "language": self.language,
            "title": self.title,
            "text": self.text,
            "source": "wikipedia",
            "license": "CC BY-SA 4.0",
        }

        if self.sections:
            record["sections"] = [
                {"heading": heading, "text": body} for heading, body in self.sections
            ]

        return record


def _open_dump(path: Path) -> Any:
    """Open a dump, transparently for bz2, gzip or plain XML."""

    suffix = path.suffix.lower()

    if suffix == ".bz2":
        return bz2.open(path, "rb")

    if suffix == ".gz":
        return gzip.open(path, "rb")

    return path.open("rb")


def _strip_markup(wikitext: str) -> str:
    """
    Reduce wikitext to prose.

    Order matters. Tables and block HTML are removed *before* the parser
    runs, because ``strip_code`` keeps their contents — a table's cells
    survive as a run of bare values that reads like prose to everything
    downstream and is not.
    """

    import mwparserfromhell

    text = _HTML_COMMENT.sub("", wikitext)

    text = _HTML_BLOCK.sub("", text)

    text = _TABLE.sub("", text)

    text = _MEDIA_LINK.sub("", text)

    text = _SELF_CLOSING.sub("", text)

    stripped = str(mwparserfromhell.parse(text).strip_code())

    return _remove_residue(stripped)


def _split_sections(wikitext: str) -> tuple[str, list[tuple[str, str]]]:
    """
    Separate the lead from its sections.

    Returns the lead and a list of retained ``(heading, body)`` pairs.
    Boilerplate sections are dropped: a references list is a citation
    dump, and training on it teaches the model about citation formatting.
    """

    matches = list(_HEADING.finditer(wikitext))

    if not matches:
        return wikitext, []

    lead = wikitext[: matches[0].start()]

    sections: list[tuple[str, str]] = []

    for index, match in enumerate(matches):
        heading = match.group(2).strip()

        end = matches[index + 1].start() if index + 1 < len(matches) else len(wikitext)

        body = wikitext[match.end() : end]

        if heading.casefold() in _BOILERPLATE_HEADINGS:
            continue

        sections.append((heading, body))

    return lead, sections


def _tidy(text: str) -> str:
    """Collapse the whitespace damage that stripping leaves behind."""

    text = _SPACE_RUN.sub(" ", text)

    text = _BLANK_RUN.sub("\n\n", text)

    return "\n".join(line.strip() for line in text.splitlines()).strip()


def iter_articles(
    dump: str | Path,
    *,
    language: str,
    minimum_characters: int = 200,
    limit: int | None = None,
    deduplicate: bool = True,
) -> Iterator[WikipediaArticle]:
    """
    Stream cleaned articles from a ``pages-articles`` dump.

    Memory is bounded by one article, not by the dump, so this runs over
    a multi-gigabyte file on a laptop. Elements are cleared as they are
    consumed — without that, ``iterparse`` retains the whole tree and the
    process grows until it is killed.

    Parameters
    ----------
    dump:
        Path to ``*-pages-articles.xml.bz2``, or the same content
        gzipped or plain.

    language:
        ISO code recorded on every article. The dump does not state it
        in a form worth trusting, so the caller supplies it.

    minimum_characters:
        Articles shorter than this are skipped. Wikipedia is full of
        one-line stubs that add vocabulary noise without teaching
        anything about the language.

    limit:
        Stop after this many articles. For trying a dump before
        committing to it.

    deduplicate:
        Drop articles whose text has already been seen. On by default,
        because template-generated stubs are common and numerous: the
        Meetei Mayek wiki has 118 country articles sharing one sentence
        of boilerplate, which would inflate the apparent frequency of
        every token in it by two orders of magnitude. Uses the same
        whitespace-collapsed hash as the audit, so what the audit would
        report as a duplicate is what this removes.

    Raises
    ------
    WikipediaExtractionError
        If the dump cannot be parsed, or if ``mwparserfromhell`` is not
        installed.

    Example
    -------
    ::

        for article in iter_articles("hiwiki.xml.bz2", language="hi", limit=10):
            print(article.title, len(article.text))
    """

    try:
        import mwparserfromhell  # noqa: F401
    except ImportError as error:
        raise WikipediaExtractionError(
            "Wikipedia extraction needs the 'wikipedia' extra (uv sync --extra wikipedia)",
            missing="mwparserfromhell",
        ) from error

    path = Path(dump).expanduser()

    if not path.exists():
        raise WikipediaExtractionError("Dump not found", path=str(path))

    produced = 0

    skipped_redirect = 0

    skipped_short = 0

    skipped_damaged = 0

    skipped_duplicate = 0

    seen: int = 0

    digests: set[str] = set()

    handle = _open_dump(path)

    try:
        # The dump declares a default namespace, so every tag arrives
        # brace-prefixed. Matching on the local name keeps this working
        # across export schema versions rather than pinning to one.
        for _, element in ElementTree.iterparse(handle, events=("end",)):
            tag = element.tag.rsplit("}", 1)[-1]

            if tag != "page":
                continue

            seen += 1

            article = _article_from_page(element, language=language)

            element.clear()

            if article is None:
                skipped_redirect += 1

                continue

            if len(article.text) < minimum_characters:
                skipped_short += 1

                continue

            # Dropped rather than emitted with a warning. One article in a
            # hundred is a cheap price for a corpus the audit passes, and
            # markup left in body text trains the tokenizer on syntax.
            if not _is_clean(article.text):
                skipped_damaged += 1

                continue

            if deduplicate:
                digest = hash_text(" ".join(article.text.split()))

                if digest in digests:
                    skipped_duplicate += 1

                    continue

                digests.add(digest)

            produced += 1

            yield article

            if limit is not None and produced >= limit:
                break
    except ElementTree.ParseError as error:
        raise WikipediaExtractionError(
            "Dump is not well-formed XML; it may be a partial download",
            path=str(path),
            detail=str(error),
        ) from error
    finally:
        handle.close()

    _logger.info(
        "Extracted Wikipedia dump",
        extra={
            "pages_seen": seen,
            "articles": produced,
            "skipped_non_article_or_redirect": skipped_redirect,
            "skipped_short": skipped_short,
            "skipped_damaged_markup": skipped_damaged,
            "skipped_duplicate": skipped_duplicate,
        },
    )

    if skipped_duplicate:
        _logger.warning(
            "Dropped duplicate articles, usually template-generated stubs",
            extra={"dropped": skipped_duplicate, "kept": produced},
        )

    if skipped_damaged:
        # Said out loud rather than left in a log nobody reads. A silent
        # drop reads as "everything was extracted" when it was not.
        _logger.warning(
            "Dropped articles whose source markup was malformed beyond repair",
            extra={
                "dropped": skipped_damaged,
                "kept": produced,
                "share": round(skipped_damaged / max(1, skipped_damaged + produced), 5),
            },
        )


def _article_from_page(element: Any, *, language: str) -> WikipediaArticle | None:
    """
    Build an article from one ``<page>`` element.

    Returns ``None`` for anything that is not article prose: a redirect,
    or a page outside namespace 0.
    """

    def find(name: str) -> Any:
        for child in element.iter():
            if child.tag.rsplit("}", 1)[-1] == name:
                return child

        return None

    namespace = find("ns")

    if namespace is None or (namespace.text or "").strip() != _ARTICLE_NAMESPACE:
        return None

    if find("redirect") is not None:
        return None

    title_element = find("title")

    text_element = find("text")

    if title_element is None or text_element is None or not text_element.text:
        return None

    identifier_element = find("id")

    title = (title_element.text or "").strip()

    lead_source, section_sources = _split_sections(text_element.text)

    lead = _tidy(_strip_markup(lead_source))

    sections = []

    for heading, body in section_sources:
        cleaned = _tidy(_strip_markup(body))

        if cleaned:
            sections.append((_tidy(_strip_markup(heading)), cleaned))

    parts = [lead] + [body for _, body in sections]

    # `is not None` rather than a truth test. ElementTree defines an
    # element's truthiness by its number of children, so a leaf such as
    # `<id>4461</id>` is falsy and a plain `if` silently falls through to
    # the title — which is what happened, giving every article a title
    # for an id until a dump was actually read.
    identifier = title

    if identifier_element is not None and identifier_element.text:
        identifier = identifier_element.text.strip()

    return WikipediaArticle(
        identifier=identifier,
        title=title,
        text=_tidy("\n\n".join(part for part in parts if part)),
        language=language,
        sections=sections,
    )


def extract_dump(
    dump: str | Path,
    output: str | Path,
    *,
    language: str,
    minimum_characters: int = 200,
    limit: int | None = None,
    deduplicate: bool = True,
) -> int:
    """
    Extract a dump to a JSON Lines file and return the article count.

    The output is gzipped when its name ends ``.gz``. Writing gzipped is
    the sensible default: these files are large, and every corpus reader
    decompresses transparently.

    Example
    -------
    ::

        extract_dump("hiwiki.xml.bz2", "hi.jsonl.gz", language="hi")
    """

    import json

    destination = Path(output).expanduser()

    destination.parent.mkdir(parents=True, exist_ok=True)

    opener = gzip.open if destination.suffix == ".gz" else open

    count = 0

    with opener(destination, "wt", encoding="utf-8") as handle:
        for article in iter_articles(
            dump,
            language=language,
            minimum_characters=minimum_characters,
            limit=limit,
            deduplicate=deduplicate,
        ):
            handle.write(json.dumps(article.to_record(), ensure_ascii=False) + "\n")

            count += 1

    _logger.info(
        "Wrote extracted corpus",
        extra={"path": str(destination), "articles": count},
    )

    return count
