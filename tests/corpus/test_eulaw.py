"""
Reading EU legislation (EUR-Lex Formex) into cross-lingual training pairs.

EU law's trainable property is not "find every article" — Formex hands the
articles over with an explicit ``IDENTIFIER`` — it is that the identifier is the
**same across languages**, so a cross-lingual pair is an exact join, not a
similarity heuristic. These tests pin that join and the register the pair design
turns on: two language expressions of one act align on the identifier, each
survivor emits **both directions** with **empty** E5 prefixes, and a pair is only
as usable as its weaker language side.

The filter is pinned oracle-diff style: the disciplined pass is diffed against a
deliberately naive reference (:meth:`EuLawFilterConfig.naive`) over a fixture
carrying one provision of every non-substantive category (repealed stub,
heading-only, duplicate), and the two are required to *disagree* by exactly those
provisions — the removals reconcile to the naive population, so the junk removal
is loud and measured instead of an opaque "fewer pairs". The CC-BY attribution
and the Decision 2011/833/EU basis are asserted present so a mechanical pipeline
cannot strip the provenance off the text.
"""

from __future__ import annotations

import pytest

from multilingual_embedding.corpus.eulaw import (
    EULAW_ATTRIBUTION,
    EULAW_LEGAL_BASIS,
    EULAW_LICENSE,
    EULAW_PROVISION_XLING_KIND,
    EULAW_SOURCE,
    EuLawFilterConfig,
    EuLawReadError,
    EuLawStatistics,
    FormexProvision,
    align_provisions,
    iter_cross_lingual_pairs,
    prefix_regime,
    read_formex_articles,
)


def _act(*articles: str) -> str:
    """Wrap article fragments in a minimal Formex ``<ACT>`` envelope."""

    return '<?xml version="1.0" encoding="UTF-8"?><ACT>' + "".join(articles) + "</ACT>"


def _article(identifier: str, title: str, subtitle: str = "", *parags: str) -> str:
    sti = f"<STI.ART>{subtitle}</STI.ART>" if subtitle else ""
    body = "".join(
        f'<PARAG IDENTIFIER="{identifier}.{i:03d}"><NO.PARAG>{i}.</NO.PARAG>'
        f"<ALINEA>{p}</ALINEA></PARAG>"
        for i, p in enumerate(parags, start=1)
    )
    return f'<ARTICLE IDENTIFIER="{identifier}"><TI.ART>{title}</TI.ART>{sti}{body}</ARTICLE>'


_EN = _act(
    _article(
        "001",
        "Article 1",
        "Subject-matter",
        "This Regulation lays down rules relating to the protection of natural persons.",
    ),
    _article(
        "002",
        "Article 2",
        "Scope",
        "This Regulation applies to the processing of personal data"
        " wholly or partly by automated means.",
    ),
)

_DE = _act(
    _article(
        "001",
        "Artikel 1",
        "Gegenstand",
        "Diese Verordnung enthaelt Vorschriften zum Schutz natuerlicher Personen.",
    ),
    _article(
        "002",
        "Artikel 2",
        "Anwendungsbereich",
        "Diese Verordnung gilt fuer die Verarbeitung personenbezogener Daten.",
    ),
)


# --- parsing -----------------------------------------------------------------


def test_read_formex_extracts_identifier_heading_and_body() -> None:
    provisions = read_formex_articles(_EN, celex="TESTREG", language="en")

    assert [p.number for p in provisions] == ["001", "002"]

    first = provisions[0]
    assert first.heading == "Article 1 - Subject-matter"
    assert first.text.startswith("1. This Regulation lays down rules")
    # the paragraph number is separated from its text, not glued
    assert "1.This" not in first.text


def test_read_formex_heading_without_subtitle_is_just_the_title() -> None:
    xml = _act(_article("001", "Article 1", "", "A body long enough to be a real provision here."))

    provision = read_formex_articles(xml, celex="X", language="en")[0]

    assert provision.heading == "Article 1"


def test_read_formex_accepts_bytes() -> None:
    provisions = read_formex_articles(_EN.encode("utf-8"), celex="TESTREG", language="en")

    assert len(provisions) == 2


def test_read_formex_skips_article_without_identifier() -> None:
    xml = _act(
        "<ARTICLE><TI.ART>Untagged</TI.ART></ARTICLE>",
        _article("001", "Article 1", "", "A body long enough to be a real provision here."),
    )

    provisions = read_formex_articles(xml, celex="X", language="en")

    assert [p.number for p in provisions] == ["001"]


def test_read_formex_malformed_xml_raises() -> None:
    with pytest.raises(EuLawReadError):
        read_formex_articles("<ACT><ARTICLE>unclosed", celex="X", language="en")


def test_read_formex_no_article_raises() -> None:
    with pytest.raises(EuLawReadError):
        read_formex_articles(_act(), celex="X", language="en")


# --- alignment ---------------------------------------------------------------


def test_align_is_inner_join_on_identifier() -> None:
    en = read_formex_articles(_EN, celex="R", language="en")
    de = read_formex_articles(_DE, celex="R", language="de")

    aligned = align_provisions(en, de)

    assert [(a.number, b.number) for a, b in aligned] == [("001", "001"), ("002", "002")]


def test_align_drops_provisions_present_in_only_one_language() -> None:
    en = read_formex_articles(_EN, celex="R", language="en")  # 001, 002
    de = read_formex_articles(
        _act(
            _article(
                "001", "Artikel 1", "", "Diese Verordnung enthaelt Vorschriften zum Schutz."
            )
        ),
        celex="R",
        language="de",
    )  # 001 only

    aligned = align_provisions(en, de)

    assert [a.number for a, _ in aligned] == ["001"]


# --- cross-lingual pairs -----------------------------------------------------


def test_pairs_emit_both_directions_with_empty_prefixes() -> None:
    en = read_formex_articles(_EN, celex="R", language="en")
    de = read_formex_articles(_DE, celex="R", language="de")

    pairs = list(
        iter_cross_lingual_pairs(en, de, left_language="en", right_language="de", celex="R")
    )

    # two aligned articles x two directions
    assert len(pairs) == 4

    forward, backward = pairs[0], pairs[1]
    assert forward.language == "en" and backward.language == "de"
    assert forward.anchor == backward.positive and forward.positive == backward.anchor

    # symmetric register: empty prefixes, both sides
    assert prefix_regime(EULAW_PROVISION_XLING_KIND) == ("", "")


def test_pairs_share_one_document_id_per_article() -> None:
    en = read_formex_articles(_EN, celex="32016R0679", language="en")
    de = read_formex_articles(_DE, celex="32016R0679", language="de")

    pairs = list(
        iter_cross_lingual_pairs(
            en, de, left_language="en", right_language="de", celex="32016R0679"
        )
    )

    # both directions of article 001 share its document id
    assert pairs[0].document == pairs[1].document == "eur-lex:32016R0679:001"
    assert pairs[2].document == "eur-lex:32016R0679:002"


def test_cross_lingual_overlap_is_low_not_string_solvable() -> None:
    en = read_formex_articles(_EN, celex="R", language="en")
    de = read_formex_articles(_DE, celex="R", language="de")

    pairs = list(
        iter_cross_lingual_pairs(en, de, left_language="en", right_language="de", celex="R")
    )

    # different languages: the anchor's tokens are largely absent from the
    # positive, so a string matcher cannot win the pair
    assert all(p.overlap < 0.5 for p in pairs)


# --- filter, pinned oracle-diff ----------------------------------------------


def _noisy_pair_lists() -> tuple[list[FormexProvision], list[FormexProvision]]:
    """One aligned provision of every non-substantive category, both languages."""

    live_en = (
        "A genuine operative provision with real substantive legal content,"
        " long enough to be a real passage."
    )
    live_de = (
        "Eine echte operative Vorschrift mit echtem materiellen Inhalt,"
        " lang genug fuer eine echte Passage hier."
    )
    left = [
        # live, substantive
        FormexProvision("001", "Article 1 - Scope", live_en),
        # repealed stub (body is only the marker)
        FormexProvision("002", "Article 2 - Gone", "deleted"),
        # heading-only (no body)
        FormexProvision("003", "Article 3 - Empty", ""),
        # duplicate of 001
        FormexProvision("004", "Article 1 - Scope", live_en),
    ]
    right = [
        FormexProvision("001", "Artikel 1 - Anwendung", live_de),
        FormexProvision("002", "Artikel 2 - Weg", "gestrichen"),
        FormexProvision("003", "Artikel 3 - Leer", ""),
        FormexProvision("004", "Artikel 1 - Anwendung", live_de),
    ]
    return left, right


def test_filter_keeps_only_the_substantive_provision() -> None:
    left, right = _noisy_pair_lists()
    stats = EuLawStatistics()

    pairs = list(
        iter_cross_lingual_pairs(
            left, right, left_language="en", right_language="de", celex="R", statistics=stats
        )
    )

    # only article 001 survives -> both directions
    assert len(pairs) == 2
    assert stats.produced == 1
    assert stats.rejected == {"repealed": 1, "body-too-short": 1, "duplicate": 1}


def test_oracle_diff_naive_keeps_everything_tuned_removes_named_reasons() -> None:
    left, right = _noisy_pair_lists()

    naive_stats = EuLawStatistics()
    naive = list(
        iter_cross_lingual_pairs(
            left,
            right,
            left_language="en",
            right_language="de",
            celex="R",
            config=EuLawFilterConfig.naive(),
            statistics=naive_stats,
        )
    )

    tuned_stats = EuLawStatistics()
    tuned = list(
        iter_cross_lingual_pairs(
            left,
            right,
            left_language="en",
            right_language="de",
            celex="R",
            config=EuLawFilterConfig(),
            statistics=tuned_stats,
        )
    )

    # naive keeps all four articles (both directions); tuned keeps one
    assert len(naive) == 8
    assert naive_stats.rejected == {}
    assert len(tuned) == 2

    # the removals reconcile to the naive population exactly
    assert tuned_stats.produced + sum(tuned_stats.rejected.values()) == naive_stats.produced


def test_repealed_marker_only_fires_on_a_whole_body_marker() -> None:
    # a substantive article that merely mentions "repealed" is NOT dropped
    left = [
        FormexProvision(
            "001",
            "Article 1 - Transitional",
            "This Article is repealed as of 2020 by Article 9 of the amending act,"
            " which sets out the transitional regime.",
        )
    ]
    right = [
        FormexProvision(
            "001",
            "Artikel 1 - Uebergang",
            "Dieser Artikel wird ab 2020 durch Artikel 9 des aendernden Rechtsakts"
            " aufgehoben, der die Uebergangsregelung festlegt.",
        )
    ]
    stats = EuLawStatistics()

    pairs = list(
        iter_cross_lingual_pairs(
            left, right, left_language="en", right_language="de", celex="R", statistics=stats
        )
    )

    assert stats.rejected == {}
    assert len(pairs) == 2


# --- prefix regime & provenance ----------------------------------------------


def test_prefix_regime_unknown_kind_raises() -> None:
    with pytest.raises(EuLawReadError):
        prefix_regime("not-a-kind")


def test_provenance_constants_present() -> None:
    assert EULAW_SOURCE == "eur-lex"
    assert EULAW_LICENSE == "CC-BY-4.0"
    assert "2011/833/EU" in EULAW_ATTRIBUTION
    assert "2011/833/EU" in EULAW_LEGAL_BASIS
    assert "CC-BY" in EULAW_LEGAL_BASIS or "Creative Commons" in EULAW_ATTRIBUTION
