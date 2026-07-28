"""
Document-disjoint split of an aligned pair file.

Every article emits up to four pairs (title/lead x two directions) whose
`document` id is directional -- "54|4290" one way, "4290|54" the other.
Canonicalising to a sorted key groups all four, so holding out a key
holds out the whole article and no lead text leaks from train into eval.
"""
from __future__ import annotations

import gzip
import json
import random
import sys
from pathlib import Path

SRC = Path("data/pairs/hi-ta-aligned.jsonl.gz")
TRAIN = Path("data/pairs/hi-ta-aligned-train.jsonl.gz")
EVAL = Path("data/pairs/hi-ta-aligned-eval.jsonl.gz")
EVAL_TARGET_PAIRS = 2500
SEED = 1234


def canon(document: str) -> str:
    parts = document.split("|")
    return "|".join(sorted(parts))


def main() -> int:
    by_article: dict[str, list[dict]] = {}
    with gzip.open(SRC, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            by_article.setdefault(canon(r["document"]), []).append(r)

    keys = sorted(by_article)
    rng = random.Random(SEED)
    rng.shuffle(keys)

    eval_keys: set[str] = set()
    count = 0
    for k in keys:
        if count >= EVAL_TARGET_PAIRS:
            break
        eval_keys.add(k)
        count += len(by_article[k])

    def write(path: Path, wanted: set[str] | None, exclude: set[str] | None) -> tuple[int, dict]:
        n = 0
        langs: dict[str, int] = {}
        with gzip.open(path, "wt", encoding="utf-8") as out:
            for k in keys:
                if wanted is not None and k not in wanted:
                    continue
                if exclude is not None and k in exclude:
                    continue
                for r in by_article[k]:
                    out.write(json.dumps(r, ensure_ascii=False) + "\n")
                    n += 1
                    langs[r["language"]] = langs.get(r["language"], 0) + 1
        return n, langs

    en, elang = write(EVAL, eval_keys, None)
    tn, tlang = write(TRAIN, None, eval_keys)

    print(f"articles {len(keys):,}  eval-articles {len(eval_keys):,}")
    print(f"eval  {en:,} pairs {elang} -> {EVAL}")
    print(f"train {tn:,} pairs {tlang} -> {TRAIN}")
    inter = (set().union(*[set() for _ in [0]]))  # noop; disjointness is by construction
    print("document-disjoint by construction (whole articles held out)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
