#!/usr/bin/env python3
"""Derive genuinely non-pivot cross-lingual pairs from the pivot set.

Every mined pair so far has Hindi on one side — the embedding space is a
star around Hindi. But each pair is tagged with the Hindi *cluster* it came
from (the shared concept), and a cluster names the same article in many
languages. Joining the cluster's non-Hindi passages against each other gives
direct X<->Y pairs (Tamil<->Bengali, Malayalam<->Telugu, ...) with no Hindi
text in them. Hindi was only the index that discovered the concept.

Leak-free: we build only from *training* clusters. The held-out eval is
Hindi-pivot pairs on disjoint clusters, so it is untouched.
"""
import gzip, json, collections

TRAIN = "data/pairs/indic-aligned-train.jsonl.gz"
EVAL = "data/pairs/indic-aligned-eval.jsonl.gz"
DST = "data/pairs/indic-aligned-nonpivot.jsonl.gz"


def hi_id(r):
    p = r["document"].split("|")
    return p[0] if r["anchor_language"] == "hi" else p[1]


def non_hi(r):
    if r["anchor_language"] == "hi":
        return r["positive_language"], r["positive"]
    return r["anchor_language"], r["anchor"]


def overlap(a, b):
    sa, sb = set(a.split()), set(b.split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def main():
    eval_ids = {hi_id(json.loads(l)) for l in gzip.open(EVAL, "rt")}
    # cluster -> {language: longest passage seen}
    cl = collections.defaultdict(dict)
    for line in gzip.open(TRAIN, "rt"):
        r = json.loads(line)
        hid = hi_id(r)
        if hid in eval_ids:            # guard, though split is already disjoint
            continue
        lang, text = non_hi(r)
        if len(text) > len(cl[hid].get(lang, "")):
            cl[hid][lang] = text

    written = 0
    pairmix = collections.Counter()
    with gzip.open(DST, "wt") as out:
        for hid, langs in cl.items():
            if len(langs) < 2:
                continue
            items = sorted(langs.items())        # deterministic order
            for i, (lx, tx) in enumerate(items):
                for j, (ly, ty) in enumerate(items):
                    if i == j:
                        continue
                    rec = {
                        "anchor": tx,
                        "positive": ty,
                        "kind": "aligned_nonpivot",
                        "document": f"hi:{hid}",
                        "language": ly,
                        "anchor_language": lx,
                        "positive_language": ly,
                        "overlap": round(overlap(tx, ty), 4),
                    }
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    written += 1
                    pairmix[f"{lx}->{ly}"] += 1
    print(f"wrote {written} non-pivot pairs -> {DST}")
    print(f"from {sum(1 for v in cl.values() if len(v) >= 2)} multi-language clusters")
    top = pairmix.most_common(10)
    print("busiest directed language pairs:", dict(top))
    # per-language coverage as a target
    astarget = collections.Counter()
    for k, v in pairmix.items():
        astarget[k.split("->")[1]] += v
    print("pairs by positive-language:", dict(sorted(astarget.items())))


if __name__ == "__main__":
    main()
