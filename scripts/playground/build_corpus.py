#!/usr/bin/env python3
"""Build the Playground's fixed demo corpus from FLORES-200.

The public playground (playground.quanfire.ai, Phase 2) searches over a
*fixed* multilingual corpus so the base-vs-adapter comparison is instant and
puts no arbitrary-length work on the GPU: only the visitor's short query is
embedded live. This picks that corpus.

Source is ``flores-global-devtest.jsonl`` — the same professionally-translated,
line-aligned FLORES-200 slice the published global baseline
(``reports/global-baseline-verdict.json``) was scored on. Using it keeps the
demo honest: the sentences a visitor searches are the very sentences behind the
numbers on the eval receipt, and every concept is a genuine parallel across all
fifteen languages, so cross-lingual retrieval has a real right answer.

Selection favours sentences that stand on their own (a visitor reads them out of
any article context) and concepts drawn from *different* source articles, so the
haystack is topically varied rather than fifteen near-duplicate sentences about
one subject.

Pure and deterministic — no model, no GPU. Runs anywhere:

    python scripts/playground/build_corpus.py

Writes ``scripts/playground/corpus.json``. That artefact is vendored into the
gateway (quanfire-ai-backend); :file:`precompute_vectors.py` turns it into the
base+adapter vectors the gateway ranks against.
"""
from __future__ import annotations

import collections
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FLORES = REPO_ROOT / "flores-global-devtest.jsonl"
OUT = Path(__file__).resolve().parent / "corpus.json"

# The fifteen languages the global baseline covers, in a display order that
# leads with the languages a visitor is most likely to read, then groups the
# rest by region. The scorer is language-agnostic; this only fixes display.
LANGUAGES: dict[str, str] = {
    "en": "English",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "ar": "Arabic",
    "tr": "Turkish",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "th": "Thai",
    "vi": "Vietnamese",
    "id": "Indonesian",
}

# How many parallel concepts to keep. Each becomes fifteen passages (one per
# language), so 40 concepts is a 600-passage haystack — big enough that a top-k
# search is non-trivial, small enough that the vectors stay a couple of MB and
# the UI can list every concept.
TARGET_CONCEPTS = 40

# A concept's English sentence must read as a standalone statement. FLORES
# sentences are pulled from running articles, so many open on a pronoun or a
# connective whose antecedent is the previous sentence; those make confusing
# search results out of context. Keep declarative, self-contained ones.
MIN_CHARS, MAX_CHARS = 70, 190
_LEADING_STOPWORDS = {
    "he", "she", "it", "they", "them", "this", "that", "these", "those",
    "but", "and", "however", "also", "then", "there", "his", "her", "their",
    "such", "so", "meanwhile", "instead", "yet", "still", "thus", "hence",
}
_ANTECEDENT_PRONOUNS = re.compile(r"\b(he|she|it|they|them|his|her|their|this|these)\b", re.IGNORECASE)


def _standalone(text: str) -> bool:
    """Whether an English FLORES sentence reads on its own out of context."""
    if not (MIN_CHARS <= len(text) <= MAX_CHARS):
        return False
    if not text[0].isupper():  # opens lowercase or on punctuation/quote
        return False
    if text[0] in "\"'“‘(":
        return False
    if not text.rstrip().endswith("."):  # a full declarative sentence
        return False
    first = re.sub(r"[^\w].*$", "", text.split()[0]).lower()
    if first in _LEADING_STOPWORDS:
        return False
    # A leading dependent clause pronoun in the first four words almost always
    # needs an antecedent from the previous sentence.
    head = " ".join(text.split()[:4])
    if _ANTECEDENT_PRONOUNS.search(head) and first not in {"the", "a", "an"}:
        return False
    return True


def load_aligned() -> tuple[list[str], dict[str, dict[str, str]]]:
    """Return the shared id order and per-language {id: text}."""
    by_lang: dict[str, dict[str, str]] = collections.defaultdict(dict)
    for line in FLORES.open(encoding="utf-8"):
        r = json.loads(line)
        by_lang[r["lang"]][r["id"]] = r["text"]

    missing = [code for code in LANGUAGES if code not in by_lang]
    if missing:
        raise SystemExit(f"{FLORES.name} is missing languages: {missing}")

    ids = sorted(by_lang["en"])
    for code in LANGUAGES:
        if sorted(by_lang[code]) != ids:
            raise SystemExit(f"{FLORES.name} is not line-aligned for {code!r}")
    return ids, by_lang


def select_concepts(ids: list[str], by_lang: dict[str, dict[str, str]]) -> list[str]:
    """Pick standalone, well-spaced concept ids from different articles.

    FLORES ids are sequential within a source article, so consecutive ids are
    topically near-duplicate. Greedily take standalone sentences while forcing a
    gap between chosen ids, which spreads the selection across articles and
    keeps the corpus varied.
    """
    numeric = sorted(ids, key=int)
    candidates = [i for i in numeric if _standalone(by_lang["en"][i])]

    # Spread across the full id range: aim for a gap that would land ~TARGET
    # concepts if every slot were standalone, then relax if too few qualify.
    for min_gap in (12, 9, 6, 4, 2, 1):
        chosen: list[str] = []
        last = -(10**9)
        for i in candidates:
            if int(i) - last >= min_gap:
                chosen.append(i)
                last = int(i)
            if len(chosen) >= TARGET_CONCEPTS:
                break
        if len(chosen) >= TARGET_CONCEPTS:
            return chosen[:TARGET_CONCEPTS]
    return chosen  # fewer than target: return what we found


def main() -> None:
    ids, by_lang = load_aligned()
    concept_ids = select_concepts(ids, by_lang)
    if len(concept_ids) < TARGET_CONCEPTS:
        print(f"warning: only {len(concept_ids)} standalone concepts found "
              f"(wanted {TARGET_CONCEPTS})")

    concepts = []
    passages = []
    for n, flores_id in enumerate(concept_ids):
        cid = f"c{n:02d}"
        concepts.append({
            "id": cid,
            "flores_id": flores_id,
            "label": by_lang["en"][flores_id],  # the English sentence, as its title
        })
        for code in LANGUAGES:
            passages.append({
                "pid": f"{cid}:{code}",
                "concept": cid,
                "lang": code,
                "text": by_lang[code][flores_id],
            })

    doc = {
        "meta": {
            "source": "FLORES-200 devtest (global 15-language slice)",
            "note": ("The exact parallel sentences behind "
                     "reports/global-baseline-verdict.json. Every concept is "
                     "aligned across all listed languages, so cross-lingual "
                     "retrieval has a real ground-truth match."),
            "languages": list(LANGUAGES),
            "n_concepts": len(concepts),
            "n_passages": len(passages),
        },
        "language_names": LANGUAGES,
        "concepts": concepts,
        "passages": passages,
    }
    OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO_ROOT)}: "
          f"{len(concepts)} concepts x {len(LANGUAGES)} languages = {len(passages)} passages")


if __name__ == "__main__":
    main()
