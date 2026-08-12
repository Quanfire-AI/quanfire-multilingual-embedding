"""
Reading PIB press releases into cross-lingual training pairs.

This is the intake for `embed-gov-indic`, the multilingual government-domain
adapter. Its counterpart to :mod:`.judgments`/:mod:`.annotated_acts` (which
read one-language domain text) is that PIB's value is **parallel**: the Press
Information Bureau issues the *same* release in up to a dozen Indian languages,
and those versions are the free cross-lingual anchors a multilingual retriever
learns from — a Hindi query finding the Tamil passage about the same event.

**Where the parallelism comes from — for free.** Every PIB release page carries
a ``ReleaseLang`` block linking to the same release in the other languages, each
a separate ``PRID``. The links are reciprocal (the Hindi page links back to the
English one), so a crawler expands any release to its whole sibling set and the
document correspondence is *known* — unlike Wikipedia's :mod:`.aligned` path,
this needs no langlinks join and no title-matching heuristic.

**Native Unicode, no OCR.** PIB serves release text as native Unicode in static
HTML (Devanagari/Tamil/… in the DOM), and the fields are cleanly id-addressable
— ministry (``#MinistryName``), headline (``#Titleh2``), subtitle
(``#ltrSubtitle``), date (``#PrDateTime``), then the body paragraphs. This is
the opposite of the statutory-Hindi PDFs, which are legacy-font or scanned.

**Provenance.** PIB's copyright policy permits reproduction free of charge, with
no NonCommercial and no ShareAlike clause, provided the material is used
accurately, non-derogatorily, and the source is acknowledged — all trivially met
by a non-reconstructive embedding that emits vectors and a provenance card. The
one carve-out (exclude embedded third-party material) is the same discipline the
GODL/gazette sources carry.

This module is offline: :func:`parse_release`/:func:`parse_siblings` read HTML a
caller already fetched, and :func:`crosslingual_pairs` builds pairs from parsed
release groups. Fetching (the crawl) lives outside, so the parsing and pairing
logic is tested against HTML fixtures, never a live portal.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "PIB_ATTRIBUTION",
    "PIB_LICENSE",
    "PIB_SOURCE",
    "CrossLingualConfig",
    "PibRelease",
    "crosslingual_pairs",
    "parse_release",
    "parse_siblings",
]

PIB_SOURCE = "pib-press-releases"

# A reproduction permission, not a Creative Commons grant: PIB's copyright
# policy allows free reproduction with attribution, no NC and no SA. Recorded on
# every pair so the acknowledgement obligation reaches the model card.
PIB_LICENSE = "PIB reproduction policy (free reproduction, attribution, no NC/SA)"

PIB_ATTRIBUTION = "Source: Press Information Bureau (pib.gov.in), Government of India."

# ReleaseLang labels appear in English or the native script; mapped to ISO codes.
_LANG = {
    "english": "en", "hindi": "hi", "हिन्दी": "hi", "हिंदी": "hi", "urdu": "ur",
    "اردو": "ur", "marathi": "mr", "मराठी": "mr", "telugu": "te", "తెలుగు": "te",
    "tamil": "ta", "தமிழ்": "ta", "bengali": "bn", "বাংলা": "bn", "gujarati": "gu",
    "ગુજરાતી": "gu", "kannada": "kn", "ಕನ್ನಡ": "kn", "malayalam": "ml", "മലയാളം": "ml",
    "punjabi": "pa", "ਪੰਜਾਬੀ": "pa", "odia": "or", "oriya": "or", "ଓଡ଼ିଆ": "or",
    "assamese": "as", "অসমীয়া": "as", "manipuri": "mni", "nepali": "ne",
}

_WORD = re.compile(r"[^\W_]+", re.UNICODE)


@dataclass(slots=True)
class PibRelease:
    """One PIB release in one language."""

    prid: str

    language: str

    title: str

    body: str

    ministry: str = ""

    subtitle: str = ""

    date: str = ""


def _strip_tags(html: str) -> str:
    html = re.sub(r"<script.*?</script>|<style.*?</style>", " ", html, flags=re.S | re.I)
    html = re.sub(r"<[^>]+>", " ", html)
    html = (
        html.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&#39;", "'")
        .replace("&quot;", '"')
    )
    return re.sub(r"[ \t]+", " ", re.sub(r"\s*\n\s*", "\n", html)).strip()


def _by_id(html: str, element_id: str, closing: str) -> str:
    match = re.search(rf'id="{element_id}"[^>]*>(.*?)</{closing}>', html, re.S)
    return _strip_tags(match.group(1)) if match else ""


def parse_siblings(html: str) -> dict[str, str]:
    """
    Map ``{sibling_prid: language}`` from a release page's ``ReleaseLang`` block.

    A page lists every language *other* than its own, so the reciprocal links
    across a sibling set label the whole group.
    """

    block = re.search(r'class="ReleaseLang">(.*?)</div>', html, re.S)

    siblings: dict[str, str] = {}

    if block:
        for prid, label in re.findall(r"PRID=(\d+)'[^>]*>\s*([^<]+?)\s*</a>", block.group(1)):
            key = label.strip().lower()

            siblings[prid] = _LANG.get(key, key)

    return siblings


def _parse_body(html: str) -> str:
    start = html.find('id="PrDateTime"')

    if start < 0:
        start = html.find("innner-page-main-about-us-content-right-part")

    region = html[start:] if start >= 0 else html

    cut = region.find('class="ReleaseLang"')

    if cut > 0:
        region = region[:cut]

    # Drop the date div itself so the body is prose, not "Posted On: …".
    region = re.sub(r'id="PrDateTime".*?</div>', " ", region, flags=re.S)

    return _strip_tags(region)


def parse_release(html: str, *, prid: str = "", language: str = "") -> PibRelease:
    """
    Parse one fetched release page into a :class:`PibRelease`.

    Fields are read by their stable element ids; the body is the content region
    after the date and before the ``ReleaseLang`` block.
    """

    return PibRelease(
        prid=prid,
        language=language,
        title=_by_id(html, "Titleh2", "h2"),
        body=_parse_body(html),
        ministry=_by_id(html, "MinistryName", "div"),
        subtitle=_by_id(html, "ltrSubtitle", "span"),
        date=_by_id(html, "PrDateTime", "div").replace("प्रविष्टि तिथि:", "").strip(),
    )


def _overlap(anchor: str, positive: str) -> float:
    """Share of the anchor's word-units also in the positive; ~0 across scripts."""

    a = set(_WORD.findall(anchor.casefold()))

    if not a:
        return 0.0

    b = set(_WORD.findall(positive.casefold()))

    return len(a & b) / len(a)


@dataclass(slots=True)
class CrossLingualConfig:
    """
    What to mine and the fragment floors.

    ``kinds`` selects which cross-lingual anchor→positive shapes to emit:
    ``title_body`` (a headline in one language against the release body in
    another — the retrieval pair that matters), ``title_title`` and
    ``body_body``. Both directions of every language pair are always emitted.
    """

    kinds: tuple[str, ...] = ("title_body", "title_title")

    minimum_anchor_characters: int = 10

    minimum_positive_characters: int = 40

    maximum_positive_characters: int = 2000

    # A cross-lingual pair should be ~0 overlap; a high value means shared
    # tokens — English proper nouns/acronyms leaking into an Indic body, or a
    # same-script pairing (hi/mr) — i.e. the pair is partly solvable by string
    # matching. Records above this are dropped. 1.0 keeps everything.
    maximum_overlap: float = 1.0


def _record(anchor: str, positive: str, kind: str, group: str, a_lang: str, p_lang: str) -> dict:
    return {
        "anchor": anchor,
        "positive": positive,
        "kind": kind,
        "document": group,
        "language": p_lang,
        "anchor_language": a_lang,
        "positive_language": p_lang,
        "overlap": round(_overlap(anchor, positive), 4),
    }


def crosslingual_pairs(
    groups: Iterable[dict[str, Any]],
    config: CrossLingualConfig | None = None,
) -> Iterator[dict]:
    """
    Emit cross-lingual pair records from parallel release groups.

    Each group is ``{"group": id, "langs": {lang: {"title":…, "body":…}}}``.
    For every ordered language pair ``(A, B)`` in a group, a headline in ``A`` is
    an anchor against the body/headline in ``B`` — so the same release yields
    both an A→B and a B→A pair, and a query in any language exercises retrieval
    into any other. Records match the aligned-pair schema (``anchor_language`` /
    ``positive_language`` alongside ``language``), so they load as ordinary
    :class:`~multilingual_embedding.corpus.pairs.MinedPair` for training.
    """

    cfg = config or CrossLingualConfig()

    for group in groups:
        gid = str(group.get("group", ""))

        langs = group.get("langs", {})

        for a_lang, a in langs.items():
            title = (a.get("title") or "").strip()

            if len(title) < cfg.minimum_anchor_characters:
                continue

            for p_lang, p in langs.items():
                if p_lang == a_lang:
                    continue

                def emit(anchor: str, positive: str, kind: str):
                    record = _record(anchor, positive, kind, gid, a_lang, p_lang)

                    return record if record["overlap"] <= cfg.maximum_overlap else None

                if "title_body" in cfg.kinds:
                    body = (p.get("body") or "").strip()

                    if cfg.minimum_positive_characters <= len(body) <= cfg.maximum_positive_characters:
                        record = emit(title, body, "title_body")

                        if record is not None:
                            yield record

                if "title_title" in cfg.kinds:
                    other = (p.get("title") or "").strip()

                    if len(other) >= cfg.minimum_anchor_characters:
                        record = emit(title, other, "title_title")

                        if record is not None:
                            yield record

                if "body_body" in cfg.kinds:
                    a_body = (a.get("body") or "").strip()

                    b_body = (p.get("body") or "").strip()

                    if (
                        cfg.minimum_positive_characters <= len(a_body)
                        and cfg.minimum_positive_characters <= len(b_body) <= cfg.maximum_positive_characters
                    ):
                        record = emit(a_body[: cfg.maximum_positive_characters], b_body, "body_body")

                        if record is not None:
                            yield record
