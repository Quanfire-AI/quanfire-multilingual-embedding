"""
Reading PIB press releases into cross-lingual training pairs.

The properties pinned here are the ones that make PIB usable as the
`embed-gov-indic` intake: fields are read from their stable element ids (not
guessed from prose), the reciprocal ``ReleaseLang`` links label a sibling set,
and a parallel release yields cross-lingual pairs in *both* directions whose
across-script overlap is ~0 — so a gain on them is evidence of meaning, not
string matching. The reproduction-licence attribution rides on every record.
"""

from __future__ import annotations

from multilingual_embedding.corpus.pib import (
    PIB_LICENSE,
    CrossLingualConfig,
    crosslingual_pairs,
    parse_release,
    parse_siblings,
)

# A minimal page in PIB's real shape: ids MinistryName / Titleh2 / ltrSubtitle /
# PrDateTime, a body paragraph, then the ReleaseLang sibling block.
_EN_HTML = """
<div class="innner-page-main-about-us-content-right-part">
  <div class="MinistryNameSubhead" id="MinistryName">Ministry of Fisheries</div>
  <h2 id="Titleh2">Union Minister Chairs Seafood Exporters Meet 2026</h2>
  <h3 id="Subtitleh3"><span id="ltrSubtitle">India to scale up value-added exports</span></h3>
  <div class="ReleaseDateSubHeaddateTime" id="PrDateTime">Posted On: 11 APR 2026 by PIB Delhi</div>
  <p>The Department of Fisheries organised the Seafood Exporters Meet in New Delhi to expand exports.</p>
  <div class="ReleaseLang">Read this release in these languages:
    <a href='https://pib.gov.in/PressReleasePage.aspx?PRID=2251082' target="_blank"> हिन्दी </a> ,
    <a href='https://pib.gov.in/PressReleasePage.aspx?PRID=2251084' target="_blank"> Marathi </a>
  </div>
</div>
"""

_HI_HTML = """
<div class="innner-page-main-about-us-content-right-part">
  <div class="MinistryNameSubhead" id="MinistryName">मत्स्य पालन मंत्रालय</div>
  <h2 id="Titleh2">केंद्रीय मंत्री ने समुद्री खाद्य निर्यातक बैठक 2026 की अध्यक्षता की</h2>
  <h3 id="Subtitleh3"><span id="ltrSubtitle">भारत मूल्यवर्धित निर्यात बढ़ाएगा</span></h3>
  <div class="ReleaseDateSubHeaddateTime" id="PrDateTime">प्रविष्टि तिथि: 11 APR 2026 by PIB Delhi</div>
  <p>मत्स्य पालन विभाग ने निर्यात बढ़ाने के लिए नई दिल्ली में समुद्री खाद्य निर्यातक बैठक आयोजित की।</p>
  <div class="ReleaseLang">इन भाषाओं में पढ़ें:
    <a href='https://pib.gov.in/PressReleasePage.aspx?PRID=2251052' target="_blank"> English </a> ,
    <a href='https://pib.gov.in/PressReleasePage.aspx?PRID=2251084' target="_blank"> Marathi </a>
  </div>
</div>
"""


def test_parse_release_reads_fields_by_id() -> None:
    rel = parse_release(_EN_HTML, prid="2251052", language="en")

    assert rel.ministry == "Ministry of Fisheries"

    assert rel.title == "Union Minister Chairs Seafood Exporters Meet 2026"

    assert rel.subtitle == "India to scale up value-added exports"

    # The body is the prose after the date, not the "Posted On" line or the nav.
    assert rel.body.startswith("The Department of Fisheries organised")

    assert "Posted On" not in rel.body

    assert "Read this release" not in rel.body


def test_parse_siblings_maps_prid_to_language() -> None:
    # The English page lists its non-English siblings, labelled.
    assert parse_siblings(_EN_HTML) == {"2251082": "hi", "2251084": "mr"}

    # The Hindi page reciprocally links back to English.
    assert parse_siblings(_HI_HTML)["2251052"] == "en"


def test_crosslingual_pairs_both_directions_zero_overlap() -> None:
    group = {
        "group": "2251052",
        "langs": {
            "en": {"title": parse_release(_EN_HTML).title, "body": parse_release(_EN_HTML).body},
            "hi": {"title": parse_release(_HI_HTML).title, "body": parse_release(_HI_HTML).body},
        },
    }

    pairs = list(crosslingual_pairs([group], CrossLingualConfig(minimum_positive_characters=10)))

    directions = {(p["anchor_language"], p["positive_language"]) for p in pairs}

    # Both en->hi and hi->en are emitted from the one parallel release.
    assert ("en", "hi") in directions

    assert ("hi", "en") in directions

    # Across Latin vs Devanagari the pairs share no word-units -> not string-solvable.
    title_body = [p for p in pairs if p["kind"] == "title_body"]

    assert title_body

    assert all(p["overlap"] == 0.0 for p in title_body)

    # The positive's language is recorded for the retrieval evaluator to group by.
    assert all(p["language"] == p["positive_language"] for p in pairs)


def test_licence_constant_present() -> None:
    assert "no NC/SA" in PIB_LICENSE
