"""
Wikipedia extraction.

The tests are written against the shapes a real dump actually contains,
because the failures that matter here are not exceptions. An extraction
that leaves markup behind, or silently gives every article the same
identifier, produces a corpus that loads and trains and yields a worse
model — which is exactly the class of defect the corpus audit exists to
name.

So the decisive assertion in most of these is not "it ran" but "the
output would pass ``audit_corpus``".
"""

from __future__ import annotations

import bz2
import gzip
import json
from pathlib import Path

import pytest

pytest.importorskip("mwparserfromhell", reason="requires the wikipedia extra")

from multilingual_embedding.corpus.audit import (
    Severity,
    audit_corpus,
)
from multilingual_embedding.corpus.document import Document
from multilingual_embedding.corpus.wikipedia import (
    WikipediaExtractionError,
    extract_dump,
    iter_articles,
)

PROSE = "ꯃꯄꯪ ꯑꯁꯤ ꯃꯁꯥꯒꯤ ꯃꯑꯣꯡ ꯂꯩꯖꯕ ꯄꯣꯠꯁꯛ ꯑꯃꯅꯤ ꯫ " * 8


def page(
    identifier: str,
    title: str,
    text: str,
    *,
    namespace: str = "0",
    redirect: bool = False,
) -> str:
    """One ``<page>`` element, shaped as the export schema writes it."""

    marker = '      <redirect title="Elsewhere" />\n' if redirect else ""

    escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    return (
        "  <page>\n"
        f"    <title>{title}</title>\n"
        f"    <ns>{namespace}</ns>\n"
        f"    <id>{identifier}</id>\n"
        f"{marker}"
        "    <revision>\n"
        "      <id>999</id>\n"
        f'      <text bytes="1" xml:space="preserve">{escaped}</text>\n'
        "    </revision>\n"
        "  </page>\n"
    )


def dump(tmp_path: Path, *pages: str, compress: bool = True) -> Path:
    """Write a minimal but schema-shaped dump."""

    body = (
        '<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/">\n'
        "  <siteinfo><sitename>Test</sitename></siteinfo>\n" + "".join(pages) + "</mediawiki>\n"
    )

    path = tmp_path / ("dump.xml.bz2" if compress else "dump.xml")

    if compress:
        path.write_bytes(bz2.compress(body.encode("utf-8")))
    else:
        path.write_text(body, encoding="utf-8")

    return path


class TestWhatIsExtracted:
    def test_an_article_becomes_one_record(self, tmp_path: Path) -> None:
        source = dump(tmp_path, page("1642", "ꯃꯄꯪ", PROSE))

        articles = list(iter_articles(source, language="mni"))

        assert len(articles) == 1

        assert articles[0].title == "ꯃꯄꯪ"

        assert articles[0].language == "mni"

    def test_the_identifier_is_the_page_id_not_the_title(self, tmp_path: Path) -> None:
        """
        This failed on a real dump and nothing complained.

        ``if element`` is falsy for a childless element in ElementTree —
        truthiness is child count — so a plain truth test on ``<id>``
        silently fell through to the title. Every article got a title for
        an identifier, which still loads, still trains, and quietly
        breaks any join against another dump keyed on page id.
        """

        source = dump(tmp_path, page("1642", "ꯃꯄꯪ", PROSE))

        assert next(iter(iter_articles(source, language="mni"))).identifier == "1642"

    def test_redirects_are_skipped(self, tmp_path: Path) -> None:
        source = dump(
            tmp_path,
            page("1", "Redirect", PROSE, redirect=True),
            page("2", "Real", PROSE),
        )

        assert [a.identifier for a in iter_articles(source, language="mni")] == ["2"]

    def test_non_article_namespaces_are_skipped(self, tmp_path: Path) -> None:
        """Talk pages and templates are project machinery, not prose."""

        source = dump(
            tmp_path,
            page("1", "Talk:Something", PROSE, namespace="1"),
            page("2", "Template:Infobox", PROSE, namespace="10"),
            page("3", "Real", PROSE),
        )

        assert [a.identifier for a in iter_articles(source, language="mni")] == ["3"]

    def test_stubs_are_skipped(self, tmp_path: Path) -> None:
        source = dump(tmp_path, page("1", "Stub", "Too short."), page("2", "Real", PROSE))

        assert [a.identifier for a in iter_articles(source, language="mni")] == ["2"]

    def test_sections_are_kept_for_pair_mining(self, tmp_path: Path) -> None:
        """
        A heading and its body are a query and a passage. Recovering that
        structure later means re-parsing the dump, so it is kept now.
        """

        wikitext = f"{PROSE}\n\n== History ==\n{PROSE}\n\n== Geography ==\n{PROSE}"

        source = dump(tmp_path, page("1", "Place", wikitext))

        article = next(iter(iter_articles(source, language="mni")))

        assert [heading for heading, _ in article.sections] == ["History", "Geography"]

    def test_boilerplate_sections_are_dropped(self, tmp_path: Path) -> None:
        """A references list teaches citation formatting, not language."""

        wikitext = f"{PROSE}\n\n== References ==\n{PROSE}\n\n== History ==\n{PROSE}"

        source = dump(tmp_path, page("1", "Place", wikitext))

        article = next(iter(iter_articles(source, language="mni")))

        assert [heading for heading, _ in article.sections] == ["History"]


class TestMarkupIsActuallyRemoved:
    """
    The failure the corpus audit was built to catch, checked at source.

    Each of these leaves text that loads and trains perfectly well while
    teaching the tokenizer about wiki syntax.
    """

    @pytest.mark.parametrize(
        ("name", "markup"),
        [
            ("template", "{{Infobox city|name=X|pop=12M}}\n" + PROSE),
            ("nested template", "{{a|{{b|c}}|d}}\n" + PROSE),
            ("reference", PROSE + "<ref>{{cite web|url=http://x}}</ref>"),
            ("table", '{| class="wikitable"\n! Y !! P\n|-\n| 1901 || 800k\n|}\n' + PROSE),
            ("html block", "<div class='x'>noise</div>\n" + PROSE),
            ("comment", "<!-- hidden note -->\n" + PROSE),
            ("entity", PROSE + "&nbsp;&amp;"),
            ("bold and links", f"'''X''' is in [[Place|the place]]. {PROSE}"),
        ],
    )
    def test_no_marker_survives(self, tmp_path: Path, name: str, markup: str) -> None:
        source = dump(tmp_path, page("1", "X", markup))

        articles = list(iter_articles(source, language="mni"))

        assert articles, f"{name}: article was dropped entirely"

        text = articles[0].text

        for marker in ("[[", "]]", "{{", "}}", "<ref", "<div", "&nbsp;", "=="):
            assert marker not in text, f"{name}: {marker!r} survived in {text[:120]!r}"

    def test_table_contents_do_not_leak_into_prose(self, tmp_path: Path) -> None:
        """
        ``strip_code`` keeps a table's cells, so a statistics table
        arrives as a run of bare values that reads like prose to
        everything downstream and is not.
        """

        wikitext = '{| class="wikitable"\n! Year !! Population\n|-\n| 1901 || 800000\n|}\n' + PROSE

        source = dump(tmp_path, page("1", "X", wikitext))

        text = next(iter(iter_articles(source, language="mni"))).text

        assert "800000" not in text

    def test_irreparable_markup_causes_the_article_to_be_dropped(self, tmp_path: Path) -> None:
        """
        Real dumps contain markup that is malformed in the source — a
        link written ``[Foo]]``, an unterminated template. No parser can
        resolve what the author never wrote, so the article is dropped
        rather than emitted dirty.
        """

        source = dump(
            tmp_path,
            page("1", "Broken", PROSE + " {{unterminated|template " + PROSE),
            page("2", "Fine", PROSE),
        )

        kept = [a.identifier for a in iter_articles(source, language="mni")]

        assert kept == ["2"]


class TestDeduplication:
    def test_repeated_boilerplate_is_dropped_by_default(self, tmp_path: Path) -> None:
        """
        Template-generated stubs are numerous and identical. The Meetei
        Mayek wiki has 118 country articles sharing one sentence, which
        would inflate every token in it by two orders of magnitude.
        """

        source = dump(
            tmp_path,
            page("1", "Aland", PROSE),
            page("2", "Bland", PROSE),
            page("3", "Cland", PROSE),
        )

        assert len(list(iter_articles(source, language="mni"))) == 1

    def test_duplicates_can_be_kept(self, tmp_path: Path) -> None:
        source = dump(tmp_path, page("1", "A", PROSE), page("2", "B", PROSE))

        kept = list(iter_articles(source, language="mni", deduplicate=False))

        assert len(kept) == 2

    def test_whitespace_variants_count_as_duplicates(self, tmp_path: Path) -> None:
        source = dump(
            tmp_path,
            page("1", "A", PROSE),
            page("2", "B", PROSE.replace(" ", "  ")),
        )

        assert len(list(iter_articles(source, language="mni"))) == 1


class TestTheOutputPassesOurOwnAudit:
    """
    The closing of the loop, and the assertion that matters most.

    The extractor and the audit were written against the same list of
    markers, so this is the test that keeps them honest about each other.
    """

    def test_extraction_produces_a_corpus_with_no_errors(self, tmp_path: Path) -> None:
        source = dump(
            tmp_path,
            page("1", "A", "{{Infobox|x=1}}\n'''A''' is [[here]].<ref>c</ref> " + PROSE),
            page("2", "B", "<div>noise</div>\n== History ==\n" + PROSE + "x"),
            page("3", "C", PROSE + "y"),
        )

        articles = list(iter_articles(source, language="mni"))

        audit = audit_corpus(
            Document.from_text(a.text, identifier=a.identifier, language=a.language)
            for a in articles
        )

        errors = [f"{f.code}: {f.message}" for f in audit.findings if f.severity is Severity.ERROR]

        assert not errors, f"extraction produced a corpus the audit rejects: {errors}"


class TestFileHandling:
    def test_plain_xml_is_accepted(self, tmp_path: Path) -> None:
        source = dump(tmp_path, page("1", "A", PROSE), compress=False)

        assert len(list(iter_articles(source, language="mni"))) == 1

    def test_limit_stops_early(self, tmp_path: Path) -> None:
        pages = [page(str(i), f"T{i}", PROSE + str(i)) for i in range(10)]

        assert len(list(iter_articles(dump(tmp_path, *pages), language="mni", limit=3))) == 3

    def test_a_missing_dump_is_reported_clearly(self, tmp_path: Path) -> None:
        with pytest.raises(WikipediaExtractionError, match="not found"):
            list(iter_articles(tmp_path / "absent.xml.bz2", language="mni"))

    def test_a_truncated_dump_is_reported_as_such(self, tmp_path: Path) -> None:
        """A partial download is the likely cause, and worth saying."""

        path = tmp_path / "partial.xml"

        path.write_text("<mediawiki><page><title>A</title>", encoding="utf-8")

        with pytest.raises(WikipediaExtractionError, match="well-formed"):
            list(iter_articles(path, language="mni"))

    def test_extract_dump_writes_gzipped_json_lines(self, tmp_path: Path) -> None:
        source = dump(tmp_path, page("1", "A", PROSE), page("2", "B", PROSE + "x"))

        output = tmp_path / "out.jsonl.gz"

        assert extract_dump(source, output, language="mni") == 2

        with gzip.open(output, "rt", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle]

        assert [r["id"] for r in rows] == ["1", "2"]

        assert all(r["language"] == "mni" for r in rows)

        assert all(r["license"] == "CC BY-SA 4.0" for r in rows)

    def test_the_written_records_are_readable_by_the_corpus_reader(self, tmp_path: Path) -> None:
        """The format the extractor emits must be the format we consume."""

        from multilingual_embedding.corpus.reader import JsonlReader

        source = dump(tmp_path, page("1", "A", PROSE))

        output = tmp_path / "out.jsonl"

        extract_dump(source, output, language="mni")

        documents = list(JsonlReader(output).iter_documents())

        assert len(documents) == 1

        assert documents[0].identifier == "1"

        assert documents[0].metadata.base.language == "mni"


class TestExtractionCannotBeStalledByOneArticle:
    """
    Real dumps contain malformed markup, and a regex that backtracks on
    it turns one bad article into a stalled extraction.

    The media-link pattern originally carried a nested quantifier —
    ``(\\|[^\\[\\]]*)*`` after ``[^\\[\\]]*`` — which is the classic
    catastrophic shape. On an unclosed media link the engine tries every
    way of splitting the pipes between the two quantifiers: 22 pipes took
    8.5 seconds, and each further pipe roughly quadrupled it.

    Nothing raised, no test failed, and Tamil extracted 4.6 times slower
    per article than Hindi.
    """

    def test_an_unclosed_media_link_with_many_pipes_is_fast(self) -> None:
        import time

        from multilingual_embedding.corpus.wikipedia import _strip_markup

        # Under the old pattern this input alone took several minutes.
        wikitext = "[[File:x.jpg" + "|thumb" * 40 + " and it never closes " + PROSE

        started = time.perf_counter()

        _strip_markup(wikitext)

        elapsed = time.perf_counter() - started

        assert elapsed < 1.0, (
            f"stripping one malformed media link took {elapsed:.1f}s; "
            f"the media-link pattern is backtracking again"
        )

    def test_media_links_are_still_removed(self) -> None:
        """The fix must not have been to stop matching."""

        from multilingual_embedding.corpus.wikipedia import _strip_markup

        for markup in (
            "[[File:photo.jpg|thumb|300px|A caption]]",
            "[[Category:Cities of India]]",
            "[[Image:map.svg|left]]",
        ):
            assert _strip_markup(markup).strip() == "", f"{markup} survived"

    def test_ordinary_wikilinks_keep_their_text(self) -> None:
        """A link without a colon is prose, not a media embed."""

        from multilingual_embedding.corpus.wikipedia import _strip_markup

        assert "the state" in _strip_markup("[[Maharashtra|the state]]")
