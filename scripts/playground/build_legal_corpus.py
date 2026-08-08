#!/usr/bin/env python3
"""Build the Playground's fixed legal-demo corpus from held-out judgment pairs.

The flagship demo (:file:`build_corpus.py`) is *cross-lingual*: one concept in
fifteen languages, and the visitor watches retrieval cross a language boundary.
``embed-legal-en`` is English-only, so its demo is a different shape and shows a
different thing — the adapter ranking the correct Supreme Court passage *above*
the untrained base on the **exact task it was measured on**.

What that task actually is. ``embed-legal-en`` was trained and scored on
*adjacency* pairs: an anchor passage and the passage that immediately follows it
in the same judgment. The model's job is: given one ~2,000-character passage,
retrieve its true continuation out of a haystack of other passages. That is a
real retrieval task (it is how a paragraph-level "find the related passage"
feature works), and it is the task behind the published +76%. The demo
reproduces it honestly rather than inventing a prettier short-query task the
number would not back.

Honesty rule, inherited from the flagship. The flagship corpus is drawn from the
very FLORES sentences the published number was scored on. This mirrors that: the
legal haystack and the query passages are drawn from the **held-out** evaluation
set — the same 2,000 pairs ``qfme adapt`` scored ``embed-legal-en`` on, and
which training explicitly excludes (``train`` keeps only pairs whose ``positive``
is *not* among these). Reproduced here bit-for-bit with the adapter's split::

    held = sample_pairs(pairs.jsonl.gz, 2000, seed=seed + 10_000)   # seed=0

so no passage here was a training target: an adapter win is generalisation, not
memorisation.

Two presentation steps, documented because they touch what the visitor sees:

* **Running-header furniture is stripped.** SCR page scans interleave a running
  header ("SUPREME COURT REPORTS [2023] 8 S.C.R.") into the text at page breaks.
  It is scanning furniture, not judgment content, so it is removed from both the
  embedded and the displayed text — a truer semantic demo, and the header is
  exactly the kind of token a good model should ignore anyway.
* **Long passages are truncated at a sentence boundary** near ``PREVIEW_CHARS``
  so a card is readable. The demo is a demo; the untruncated numbers live on the
  model card and the eval receipt.

Query selection is biased toward **LOW word-overlap** pairs: a pair whose anchor
shares few words with its continuation can only be solved by meaning, not string
matching (the handbook's "read the low-overlap row first"), so the adapter's
semantic lift is the visible effect. Passages are spread across judgments
(``PER_DOC`` cap) so no single case dominates.

Pure and deterministic — no model, no GPU. Runs on the box::

    python scripts/playground/build_legal_corpus.py \\
        --pairs /home/arnab/quanfire-ai-data/embedding/legal-indic/corpus/pairs.jsonl.gz \\
        --out   scripts/playground/legal-corpus.json

Writes ``legal-corpus.json``: a ``passages`` haystack and a ``queries`` list,
each query carrying the ``pid`` of its ground-truth continuation so the gateway
can mark whether a model ranked it first. :file:`precompute_vectors.py` turns the
passages into the base+adapter vectors the gateway ranks a live query against.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from multilingual_embedding.corpus.pairs import sample_pairs

# The adapter's own split constants (configs/experiments/legal-indic-e1.yaml):
# eval is a fixed 2,000-pair sample at seed+10_000, seed=0.
SEED = 0
EVAL_PAIRS = 2000
EVAL_SEED = SEED + 10_000

# A passage is truncated to a readable card at a sentence boundary near here.
PREVIEW_CHARS = 600
# After cleaning + truncation a passage must fall in this window: long enough to
# carry meaning, short enough to read; a query anchor need only be non-trivial.
MIN_PASSAGE, MAX_PASSAGE = 120, 900
MIN_ANCHOR = 80

# Haystack and query-menu sizes, and the per-judgment cap that keeps both varied.
TARGET_PASSAGES = 220
TARGET_QUERIES = 40
PER_DOC = 2
# The short label shown in the query menu (the full text is still embedded).
LABEL_CHARS = 140

# SCR running-header furniture, wherever it lands (page breaks push it
# mid-passage). Three forms seen in the SCR scans, matched in one pass:
#   "SUPREME COURT REPORTS [2023] 8 S.C.R."   (labelled header)
#   "780 [2024] 7 S.C.R. Digital Supreme Court Reports"  (page no + citation + tag)
#   "Digital Supreme Court Reports"           (bare tag)
_RUNNING_HEADER = re.compile(
    r"\s*(?:SUPREME COURT REPORTS\s*)?"
    r"(?:\b\d{1,4}\s+)?"                       # optional leading page number
    r"\[?\s*\d{4}\s*\]?\s*\d+\s*S\.?\s*C\.?\s*R\.?\s*\d*"   # [YEAR] N S.C.R. [page]
    r"(?:\s*Digital Supreme Court Reports)?"
    r"|Digital Supreme Court Reports",
    re.IGNORECASE)
_WS = re.compile(r"\s+")
# PDF ligatures that survive extraction, mapped back to plain letters so cards
# read cleanly (e.g. "certiﬁed" not "certiﬁ ed").
_LIGATURES = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "ft", "ﬆ": "st"}
_DOTTED = re.compile(r"\.{4,}")            # table-of-contents leader dots
# Front-matter that is not judgment reasoning: cause-title, jurisdiction line,
# counsel-appearance blocks. A demo of "given this list of advocates, find the
# next list" is unimpressive and confusing, so these are excluded.
_FURNITURE = (
    "appellate jurisdiction", "original jurisdiction", "from the judgment and order",
    "advs.", "adv.", "sr. adv", "for the appellant", "for the respondent",
    "for the appearing parties", "for the petitioner", "counsel for",
)


def _clean(text: str) -> str:
    """Strip running-header furniture, fix ligatures, normalise whitespace."""
    for lig, plain in _LIGATURES.items():
        text = text.replace(lig, plain)
    # A ligature can extract as a spurious mid-word space ("Offi cers",
    # "certifi ed"). Rejoin ONLY the letter-preceded case, which is
    # unambiguous ("...i fi ed" is never two words); word-initial "fi le" is
    # left as-is rather than risk a bad join. Note: bare "ff" is deliberately
    # excluded — it is the one ligature that ALSO ends real English words
    # ("off the", "staff and", "tariff of"), so joining it would corrupt a
    # genuine word boundary. "fi"/"fl" have no such common word-enders.
    text = re.sub(r"(?<=[A-Za-z])(ffi|ffl|fi|fl) (?=[a-z])", r"\1", text)
    text = _WS.sub(" ", _RUNNING_HEADER.sub(" ", text)).strip()
    # A chunk can begin mid-sentence on a stray period/comma left by the split.
    return re.sub(r"^[\s.,;:]+", "", text)


def _is_prose(text: str) -> bool:
    """Whether a passage is substantive judgment prose, not furniture.

    The low-overlap bias (below) otherwise surfaces exactly the non-prose
    passages — table-of-contents dotted lines and comma-separated advocate
    name lists — because consecutive furniture chunks genuinely share few
    words. These read as noise in a public demo, so they are filtered out
    here by three cheap, robust signals.
    """
    if _DOTTED.search(text):                       # TOC leader dots
        return False
    if text.count("*") > 1:                        # index / footnote markers
        return False
    if sum(c.isdigit() for c in text) / len(text) > 0.12:
        return False                               # batch case-number lists
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    # Prose is mostly lowercase; cause-titles, HEADERS and Title-Case name
    # lists are caps-heavy. This one signal removes most furniture.
    if sum(c.islower() for c in letters) / len(letters) < 0.64:
        return False
    if text.count(". ") < 2 and not text.rstrip().endswith("."):
        return False                               # needs a few real sentences
    low = text.lower()
    return not any(marker in low for marker in _FURNITURE)


def _truncate(text: str, limit: int = PREVIEW_CHARS) -> str:
    """Trim to a sentence boundary near ``limit`` so a card reads cleanly."""
    if len(text) <= limit:
        return text
    window = text[: limit + 120]
    cut = max(window.rfind(". "), window.rfind("? "), window.rfind("! "))
    if cut >= limit - 200:          # a sentence end sits within reach
        return window[: cut + 1].strip()
    return text[:limit].rstrip() + "…"     # fall back to a hard cut + ellipsis


def _passage_text(raw: str) -> str | None:
    """Clean, truncate, and accept a passage — or None if it will not read."""
    t = _truncate(_clean(raw))
    if not t or not t[0].isalnum():
        return None
    if not (MIN_PASSAGE <= len(t) <= MAX_PASSAGE):
        return None
    return t


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True, type=Path,
                        help="the mined legal pair file (pairs.jsonl.gz)")
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parent / "legal-corpus.json")
    args = parser.parse_args()

    # 1. Reproduce the held-out evaluation set exactly (never training targets).
    held = sample_pairs(args.pairs, EVAL_PAIRS, seed=EVAL_SEED)

    # 2. A pair is usable if BOTH its cleaned continuation (a haystack passage)
    #    and its cleaned anchor (a query) survive cleaning, and the continuation
    #    is not a duplicate of one already kept.
    usable = []
    seen_positive: set[str] = set()
    for pair in held:
        positive = _passage_text(pair.positive)
        if positive is None or not _is_prose(positive):
            continue
        anchor = _clean(pair.anchor)
        if len(anchor) < MIN_ANCHOR or not _is_prose(_truncate(anchor)):
            continue
        if positive in seen_positive:
            continue
        seen_positive.add(positive)
        usable.append((pair, anchor, positive))

    # 3. Queries: low overlap first (pure-semantic), spread across judgments.
    def _ov(pair):
        return float(getattr(pair, "overlap", 0.0) or 0.0)
    ordered = sorted(usable, key=lambda t: (_ov(t[0]), t[0].document, t[1]))
    query_rows, seen_doc = [], {}
    for pair, anchor, positive in ordered:
        if seen_doc.get(pair.document, 0) >= PER_DOC:
            continue
        seen_doc[pair.document] = seen_doc.get(pair.document, 0) + 1
        query_rows.append((pair, anchor, positive))
        if len(query_rows) >= TARGET_QUERIES:
            break

    # 4. Haystack: every query's continuation MUST be present (ground truth),
    #    then fill with distractor continuations from other judgments, spread the
    #    same way, in a stable order so the file is reproducible.
    passages, pid_of_text = [], {}

    def _add(text: str, document: str, kind: str) -> str:
        pid = pid_of_text.get(text)
        if pid is not None:
            return pid
        pid = f"p{len(passages):03d}"
        pid_of_text[text] = pid
        passages.append({"pid": pid, "text": text, "document": document, "kind": kind})
        return pid

    answer_pid = {}
    for pair, _anchor, positive in query_rows:
        answer_pid[id(pair)] = _add(positive, pair.document, pair.kind)

    query_pair_ids = {id(pair) for pair, _, _ in query_rows}
    doc_count = {}
    for p in passages:
        doc_count[p["document"]] = doc_count.get(p["document"], 0) + 1
    distractors = sorted((t for t in usable if id(t[0]) not in query_pair_ids),
                         key=lambda t: (t[0].document, t[1]))
    for pair, _anchor, positive in distractors:
        if len(passages) >= TARGET_PASSAGES:
            break
        if positive in pid_of_text:
            continue
        if doc_count.get(pair.document, 0) >= PER_DOC:
            continue
        doc_count[pair.document] = doc_count.get(pair.document, 0) + 1
        _add(positive, pair.document, pair.kind)

    # 5. Emit queries with ground-truth pid, a short menu label, and the band.
    def _band(o: float) -> str:
        return "low" if o < 0.3 else ("mid" if o < 0.7 else "high")

    queries = []
    for n, (pair, anchor, _positive) in enumerate(query_rows):
        o = _ov(pair)
        label = anchor if len(anchor) <= LABEL_CHARS else anchor[:LABEL_CHARS].rstrip() + "…"
        queries.append({
            "id": f"q{n:02d}",
            "text": anchor,             # the full cleaned anchor is what gets embedded
            "label": label,             # the short form shown in the query menu
            "answer": answer_pid[id(pair)],
            "kind": pair.kind,
            "overlap": round(o, 4),
            "overlap_band": _band(o),
            "document": pair.document,
        })

    doc = {
        "meta": {
            "source": "embed-legal-en held-out evaluation pairs (Indian Supreme "
                      "Court judgments, statutory public domain, Copyright Act "
                      "1957 §52(1)(q))",
            "task": "adjacency retrieval: given a judgment passage, find its true "
                    "continuation among the haystack",
            "note": ("Passages come from the exact held-out set embed-legal-en was "
                     "scored on (sample_pairs seed=10000, 2000 pairs), never seen in "
                     "training. Running-header furniture is stripped and long passages "
                     "are truncated to a sentence boundary for display; each query "
                     "carries the pid of its ground-truth continuation. Queries are "
                     "biased toward low word-overlap so an adapter win is semantic."),
            "model": "quanfire-ai/embed-legal-en",
            "n_queries": len(queries),
            "n_passages": len(passages),
            "n_documents": len({p["document"] for p in passages}),
        },
        "queries": queries,
        "passages": passages,
    }
    args.out.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    bands = {}
    for q in queries:
        bands[q["overlap_band"]] = bands.get(q["overlap_band"], 0) + 1
    print(f"wrote {args.out}: {len(queries)} queries, {len(passages)} passages "
          f"from {doc['meta']['n_documents']} judgments; query overlap bands {bands}")


if __name__ == "__main__":
    main()
