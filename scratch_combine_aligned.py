"""
Combine per-language aligned pair files into one multilingual train set
plus a leak-free, document-disjoint evaluation.

Every language pivots through the same Hindi article, so a pair's Hindi
page id -- parts[0] of `document` when the anchor is Hindi, parts[1]
otherwise -- is the true unit of leakage. Holding out a Hindi id removes
that article's leads from *every* language at once, so no Hindi (or
aligned target) text can sit in train and eval together.

The eval holdout is stratified: Hindi clusters are added until every
language has at least PER_LANG_EVAL pairs, so the small wikis (sa, gu)
are represented without the big ones (ta, ur, bn) swamping the set.
"""
from __future__ import annotations

import gzip
import json
import random
from pathlib import Path

PAIRS = Path("data/pairs")
# (language code, pair file). ta already lives in the hi-ta file.
SOURCES = [
    ("ta", PAIRS / "hi-ta-aligned.jsonl.gz"),
    ("gu", PAIRS / "hi-gu-aligned.jsonl.gz"),
    ("sa", PAIRS / "hi-sa-aligned.jsonl.gz"),
    ("mr", PAIRS / "hi-mr-aligned.jsonl.gz"),
    ("kn", PAIRS / "hi-kn-aligned.jsonl.gz"),
    ("ml", PAIRS / "hi-ml-aligned.jsonl.gz"),
    ("te", PAIRS / "hi-te-aligned.jsonl.gz"),
    ("ur", PAIRS / "hi-ur-aligned.jsonl.gz"),
    ("bn", PAIRS / "hi-bn-aligned.jsonl.gz"),
]
TRAIN = PAIRS / "indic-aligned-train.jsonl.gz"
EVAL = PAIRS / "indic-aligned-eval.jsonl.gz"
PER_LANG_EVAL = 400
SEED = 1234


def hi_id(record: dict) -> str:
    parts = record["document"].split("|")
    return parts[0] if record["anchor_language"] == "hi" else parts[1]


def main() -> int:
    # Group every pair by its Hindi pivot id. A cluster carries pairs in
    # both directions and across languages.
    clusters: dict[str, list[dict]] = {}
    per_lang_total: dict[str, int] = {}
    for lang, path in SOURCES:
        if not path.exists():
            print(f"SKIP {lang}: {path} missing")
            continue
        n = 0
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                # tag the target language for stratification; the pair's
                # own `language` is the positive language (target or hi).
                r["_target"] = lang
                clusters.setdefault(hi_id(r), []).append(r)
                n += 1
        per_lang_total[lang] = n
        print(f"loaded {lang}: {n:,} pairs")

    keys = sorted(clusters)
    random.Random(SEED).shuffle(keys)

    # Stratified holdout: keep taking clusters while some language is
    # still short of its eval quota.
    eval_keys: set[str] = set()
    eval_by_lang: dict[str, int] = {}
    for k in keys:
        langs = {r["_target"] for r in clusters[k]}
        if any(eval_by_lang.get(L, 0) < PER_LANG_EVAL for L in langs):
            eval_keys.add(k)
            for r in clusters[k]:
                eval_by_lang[r["_target"]] = eval_by_lang.get(r["_target"], 0) + 1

    def write(path: Path, want_eval: bool) -> dict[str, int]:
        by_lang: dict[str, int] = {}
        with gzip.open(path, "wt", encoding="utf-8") as out:
            for k in keys:
                if (k in eval_keys) != want_eval:
                    continue
                for r in clusters[k]:
                    r.pop("_target", None)
                    out.write(json.dumps(r, ensure_ascii=False) + "\n")
                    by_lang[r["language"]] = by_lang.get(r["language"], 0) + 1
        return by_lang

    # Write eval first (needs _target), then train.
    eval_langs = write(EVAL, want_eval=True)
    train_langs = write(TRAIN, want_eval=False)

    print(f"\nclusters {len(keys):,}  eval-clusters {len(eval_keys):,}")
    print(f"eval  by positive-language: {dict(sorted(eval_langs.items()))} -> {EVAL}")
    print(f"train by positive-language: {dict(sorted(train_langs.items()))} -> {TRAIN}")
    print(f"eval target coverage (by target lang): {dict(sorted(eval_by_lang.items()))}")
    print("leak-free by construction: whole Hindi clusters held out across all languages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
