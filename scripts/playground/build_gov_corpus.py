#!/usr/bin/env python3
"""Build the Playground's fixed government-demo corpus from PIB press releases.

The flagship demo (:file:`build_corpus.py`) is *cross-lingual* over FLORES-200:
one concept in fifteen languages, and the visitor watches retrieval cross a
language boundary. ``embed-gov-indic`` is *also* cross-lingual — but over Indian
**government press releases**, published by the Press Information Bureau as the
*same* release in many Indian languages. So its demo is the FLORES demo's twin,
in a different language family and a different domain: a query in one Indian
language finds the matching government release written in **another**.

Shape (identical to the FLORES corpus so the gateway can reuse the same loader).
A *concept* is one PIB release; its *passages* are that release's **title** in
each language it was published in. Title (not body) is used deliberately:

* a title is a self-contained, single-line statement — readable as a search
  result card without truncation, exactly what a demo needs;
* the same release's title in Hindi and in Tamil is a genuine parallel, so
  cross-lingual retrieval has a real ground-truth match (the other-language
  titles of the *same* group).

Honesty note. The published headline for ``embed-gov-indic`` (+27.9% Recall@1,
disjoint CIs) is measured on the fuller cross-lingual **title↔body** held-out
eval, not on this title-only demo set; the demo shows the *kind* of retrieval the
model does, and the audited number lives on the model card and the eval receipt.
No claim on this page exceeds what the card measures.

Selection favours releases that are **richly multilingual** (published in many
languages, so cross-lingual retrieval is non-trivial) and **topically varied** (a
title-overlap guard skips near-duplicate releases), spread across the corpus.

Pure and deterministic — no model, no GPU. Runs on the box where the corpus
lives::

    python scripts/playground/build_gov_corpus.py \\
        --source /home/arnab/quanfire-ai-data/embedding/gov-indic/corpus/gov-corpus.jsonl \\
        --out    scripts/playground/gov-corpus.json

Writes ``gov-corpus.json``: a ``concepts`` list and a ``passages`` haystack in the
same schema as :file:`corpus.json`. :file:`precompute_vectors.py` turns the
passages into the base+adapter vectors the gateway ranks a live query against.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUT = Path(__file__).resolve().parent / "gov-corpus.json"

# The Indian languages PIB publishes in, in a display order that leads with the
# highest-reach languages, then the rest. The scorer is language-agnostic; this
# only fixes display order and the human-readable names.
LANGUAGES: dict[str, str] = {
    "en": "English",
    "hi": "Hindi",
    "ur": "Urdu",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "pa": "Punjabi",
    "or": "Odia",
    "as": "Assamese",
    "ne": "Nepali",
    "kha": "Khasi",
    "mni": "Manipuri",
}

# How many parallel releases to keep. Each becomes up to len(LANGUAGES) passages
# (one title per language it was published in), so ~30 concepts is roughly a
# 300-450 passage haystack — big enough that a top-k search is non-trivial, small
# enough that the vectors stay a few MB and the UI can list every concept.
TARGET_CONCEPTS = 30

# A release must appear in at least this many of the display languages to be a
# useful cross-lingual concept (few-language releases give the demo little to
# cross).
MIN_LANGS = 8

# Display bounds for a title used as a passage. PIB titles are single lines;
# these drop the rare fragment or over-long run-on so cards read cleanly.
MIN_TITLE_CHARS, MAX_TITLE_CHARS = 20, 240

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def _clean(text: str) -> str:
    """Collapse whitespace/newlines in a title so it renders as one line."""
    return re.sub(r"\s+", " ", text or "").strip()


def _title_of(entry: dict) -> str:
    """The title string of one language's entry, cleaned."""
    if not isinstance(entry, dict):
        return ""
    return _clean(entry.get("title", ""))


def _label_words(english_title: str) -> set[str]:
    """Lowercased word set of an English title, for the topical-dedupe guard."""
    return {w.lower() for w in _WORD.findall(english_title) if len(w) > 3}


def load_groups(source: Path) -> list[dict]:
    """Read the PIB corpus: one record per release, ``{group, langs}`` where
    ``langs`` maps a language code to ``{title, body}``."""
    groups: list[dict] = []
    with source.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            groups.append(json.loads(line))
    if not groups:
        raise SystemExit(f"{source} held no releases")
    return groups


def select_concepts(groups: list[dict]) -> list[dict]:
    """Pick richly-multilingual, topically-varied releases, deterministically.

    Ordering is (most languages first, then group id) so the haystack leads with
    the releases that cross the most language boundaries; a title-overlap guard
    skips a release whose English title is a near-duplicate of one already taken,
    so the corpus is topically varied rather than many versions of one story.
    """
    scored: list[tuple[int, str, dict]] = []
    for g in groups:
        langs = g.get("langs") or {}
        # Only languages we display, and only where a usable title exists.
        titles = {
            code: _title_of(langs.get(code, {}))
            for code in LANGUAGES
            if MIN_TITLE_CHARS <= len(_title_of(langs.get(code, {}))) <= MAX_TITLE_CHARS
        }
        if "en" not in titles:  # need an English label + a stable dedupe key
            continue
        if len(titles) < MIN_LANGS:
            continue
        scored.append((len(titles), str(g.get("group", "")), {"titles": titles}))

    # Deterministic: richest coverage first, ties broken by group id.
    scored.sort(key=lambda t: (-t[0], t[1]))

    chosen: list[dict] = []
    seen_words: list[set[str]] = []
    for _n_langs, group_id, payload in scored:
        english = payload["titles"]["en"]
        words = _label_words(english)
        # Topical-dedupe: skip if >60% of this title's content words already
        # appeared in a chosen title (near-duplicate release/story).
        if words and any(
            len(words & prev) / max(1, len(words)) > 0.6 for prev in seen_words
        ):
            continue
        chosen.append({"group_id": group_id, "titles": payload["titles"]})
        seen_words.append(words)
        if len(chosen) >= TARGET_CONCEPTS:
            break
    return chosen


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Path to gov-corpus.jsonl (one {group, langs} record per release).",
    )
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output gov-corpus.json.")
    args = ap.parse_args()

    groups = load_groups(args.source)
    chosen = select_concepts(groups)
    if len(chosen) < TARGET_CONCEPTS:
        print(
            f"warning: only {len(chosen)} qualifying releases found "
            f"(wanted {TARGET_CONCEPTS}; MIN_LANGS={MIN_LANGS})"
        )

    concepts: list[dict] = []
    passages: list[dict] = []
    lang_counts: dict[str, int] = {code: 0 for code in LANGUAGES}
    for n, rel in enumerate(chosen):
        cid = f"g{n:02d}"
        concepts.append(
            {
                "id": cid,
                "group_id": rel["group_id"],
                "label": rel["titles"]["en"],  # the English title, as the concept label
            }
        )
        for code in LANGUAGES:
            title = rel["titles"].get(code)
            if not title:
                continue
            passages.append(
                {
                    "pid": f"{cid}:{code}",
                    "concept": cid,
                    "lang": code,
                    "text": title,
                }
            )
            lang_counts[code] += 1

    doc = {
        "meta": {
            "source": "Press Information Bureau (pib.gov.in) press releases",
            "attribution": "Source — Press Information Bureau (pib.gov.in), Government of India.",
            "note": (
                "Each concept is one PIB release; its passages are that release's "
                "title in every language it was published in, so cross-lingual "
                "retrieval has a real ground-truth match (the same release in "
                "another language). The audited +27.9% headline is measured on the "
                "fuller title-body held-out eval on the model card, not this "
                "title-only demo set."
            ),
            "model": "quanfire-ai/embed-gov-indic",
            "languages": list(LANGUAGES),
            "n_concepts": len(concepts),
            "n_passages": len(passages),
            "passages_per_language": lang_counts,
        },
        "language_names": LANGUAGES,
        "concepts": concepts,
        "passages": passages,
    }
    args.out.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    try:
        shown = args.out.relative_to(REPO_ROOT)
    except ValueError:
        shown = args.out
    print(
        f"wrote {shown}: {len(concepts)} releases across "
        f"{len(LANGUAGES)} languages = {len(passages)} passages"
    )


if __name__ == "__main__":
    main()
