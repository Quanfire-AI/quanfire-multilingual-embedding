"""
Reading a Wikipedia interlanguage-link dump.

Cross-lingual pairs need a key that says "this Hindi article and that
Tamil article are the same subject". Wikipedia has one — the
interlanguage links an editor placed at the bottom of a page — but the
``pages-articles`` dump this project already extracts does **not** carry
them. They live in a separate SQL dump, ``{wiki}-latest-langlinks.sql``,
one file per wiki.

That dump is a stream of MySQL ``INSERT INTO `langlinks` VALUES (...)``
statements over three columns:

``ll_from``
    The page id in *this* wiki. It matches the ``id`` this project's
    extractor already writes for every article, which is what lets a row
    be joined back to a mined corpus.

``ll_lang``
    The language code of the linked article — the target wiki.

``ll_title``
    The linked article's title *in the target wiki*, in database form
    (spaces written as underscores).

So the Hindi wiki's langlinks dump, filtered to ``ll_lang == "ta"``, is
exactly ``hindi page id -> the Tamil title of the same subject``. Follow
that title into the Tamil corpus and the two documents are aligned.

This module parses the dump rather than trusting an unknown tool, for the
same reason the Wikipedia extractor is in-tree: a silent mis-parse would
drop alignments and quietly shrink the evaluation set, and there would be
nothing to point at. The tuple scanner is quote- and escape-aware because
article titles contain commas, apostrophes and parentheses, and a regex
over ``(\\d+,'..','..')`` would split a title like ``Massachusetts_(band)``
down the middle.
"""

from __future__ import annotations

import gzip
from collections.abc import Iterator, Mapping
from pathlib import Path

from multilingual_embedding.core.logging import get_logger

__all__ = [
    "build_target_title_map",
    "iter_langlinks",
    "normalize_title",
]

_logger = get_logger(__name__)

# MySQL dump string escapes. Anything not listed maps to itself (``\x``
# -> ``x``), which is what mysqldump's own unescaping does.
_ESCAPES = {
    "n": "\n",
    "t": "\t",
    "r": "\r",
    "0": "\0",
    "b": "\b",
    "Z": "\x1a",
    "\\": "\\",
    "'": "'",
    '"': '"',
}


def normalize_title(title: str) -> str:
    """
    Fold a title to a form that joins across the dump and the corpus.

    The langlinks dump writes titles in database form — spaces as
    underscores — while the extracted corpus keeps the human title with
    spaces. Both are folded here to spaces, collapsed and case-folded, so
    ``"New_Delhi"`` from the dump matches ``"New Delhi"`` from the corpus.

    Case-folding is deliberate. MediaWiki capitalises the first letter of
    a title on both sides, so folding case cannot merge two genuinely
    distinct articles in practice, and it recovers the occasional join
    that a stray capitalisation would otherwise lose. An unmatched title
    is a silently dropped alignment, so the bias is toward matching.
    """

    return " ".join(title.replace("_", " ").split()).casefold()


def _unescape(value: str) -> str:
    """Undo MySQL backslash escaping in a dumped string literal."""

    if "\\" not in value:
        return value

    out: list[str] = []

    index = 0

    length = len(value)

    while index < length:
        char = value[index]

        if char == "\\" and index + 1 < length:
            nxt = value[index + 1]

            out.append(_ESCAPES.get(nxt, nxt))

            index += 2

            continue

        out.append(char)

        index += 1

    return "".join(out)


def _iter_rows(payload: str) -> Iterator[tuple[str, str, str]]:
    """
    Yield ``(ll_from, ll_lang, ll_title)`` from one ``VALUES`` payload.

    A hand-written scanner rather than a regex, because a title is an
    arbitrary string that may contain the comma, quote and parenthesis
    that a naive pattern would treat as structure. Strings are
    single-quoted with backslash escapes; numbers are bare. Fields are
    unescaped as they are closed.

    A tuple whose column count is not three is skipped rather than raised
    on: a dump is large and a single malformed row should not abort a run
    that has already parsed millions of good ones.
    """

    index = 0

    length = len(payload)

    while index < length:
        if payload[index] != "(":
            index += 1

            continue

        # Inside a tuple until the matching ``)``.
        index += 1

        fields: list[str] = []

        chars: list[str] = []

        in_string = False

        while index < length:
            char = payload[index]

            if in_string:
                if char == "\\" and index + 1 < length:
                    # Keep the escape intact; _unescape resolves it once
                    # the whole field is known.
                    chars.append(char)

                    chars.append(payload[index + 1])

                    index += 2

                    continue

                if char == "'":
                    in_string = False

                    index += 1

                    continue

                chars.append(char)

                index += 1

                continue

            if char == "'":
                in_string = True

                index += 1

                continue

            if char == ",":
                fields.append(_unescape("".join(chars)))

                chars = []

                index += 1

                continue

            if char == ")":
                fields.append(_unescape("".join(chars)))

                index += 1

                if len(fields) == 3:
                    yield fields[0], fields[1], fields[2]

                break

            chars.append(char)

            index += 1


def iter_langlinks(path: str | Path) -> Iterator[tuple[int, str, str]]:
    """
    Stream ``(ll_from, ll_lang, ll_title)`` from a langlinks SQL dump.

    Gzip is handled by extension. The file is read line by line — a
    mysqldump wraps its rows into ``INSERT`` statements of roughly a
    megabyte each, so a line is bounded and the whole dump never has to
    fit in memory, matching the streaming discipline the rest of the
    corpus layer keeps.

    ``ll_from`` is returned as an ``int``; a row whose first column is
    not an integer is skipped, since it cannot be a page id.
    """

    resolved = Path(path).expanduser()

    opener = gzip.open if resolved.suffix == ".gz" else open

    with opener(resolved, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith("INSERT INTO"):
                continue

            marker = line.find("VALUES")

            if marker == -1:
                continue

            for raw_from, lang, title in _iter_rows(line[marker + len("VALUES") :]):
                try:
                    ll_from = int(raw_from)
                except ValueError:
                    continue

                yield ll_from, lang, title


def build_target_title_map(
    path: str | Path,
    *,
    target_language: str,
) -> dict[str, str]:
    """
    Build ``source page id -> target title`` for one target language.

    Reads the *source* wiki's langlinks dump and keeps only the rows
    pointing at ``target_language``. The page id is returned as a string
    so it joins directly against the corpus, whose ``id`` field this
    project always writes as a string.

    Parameters
    ----------
    path:
        The source wiki's ``langlinks`` SQL dump, optionally gzipped —
        e.g. ``hiwiki-latest-langlinks.sql.gz`` when the source corpus is
        Hindi.

    target_language:
        The ``ll_lang`` code to keep, e.g. ``"ta"``. Everything else is
        discarded, so the returned map speaks only about the one language
        pairing being mined.

    Returns
    -------
    A mapping from source page id to the raw (un-normalised) target
    title. Callers normalise with :func:`normalize_title` at lookup time
    so the corpus side is folded the same way.
    """

    mapping: dict[str, str] = {}

    for ll_from, lang, title in iter_langlinks(path):
        if lang == target_language:
            mapping[str(ll_from)] = title

    _logger.info(
        "Read interlanguage links",
        extra={"target_language": target_language, "aligned_titles": len(mapping)},
    )

    return mapping


def resolve_target_title(
    source_id: str | None,
    title_map: Mapping[str, str],
) -> str | None:
    """
    Look up the target title for a source document id.

    Returns ``None`` when the source has no id or no interlanguage link
    to the target language — an honest absence the caller counts, rather
    than a guess that would fabricate an alignment.
    """

    if not source_id:
        return None

    return title_map.get(source_id)
