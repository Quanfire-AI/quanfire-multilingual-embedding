#!/usr/bin/env python3
"""Rebalance the aligned train set toward same-script Devanagari.

The rank bump was a wash: mr/sa top-1 stayed below published e5 because the
loss is dominated by cross-script pairs. Every pair here is Hindi-pivoted, so
the non-hi side is the "target". We oversample the Devanagari targets so the
optimizer preserves their neighbourhoods, without touching cross-script pairs.

sa is the worst regression and the rarest (1.8%), so it gets the most weight.
Duplicated pairs = a weighted sampler over one epoch; nothing else changes.
"""
import gzip, json, collections

SRC = "data/pairs/indic-aligned-train.jsonl.gz"
DST = "data/pairs/indic-aligned-train-devwt.jsonl.gz"
WEIGHT = {"sa": 5, "mr": 2}  # target-language -> copies; everything else 1


def target_lang(r):
    return r["positive_language"] if r["anchor_language"] == "hi" else r["anchor_language"]


def main():
    before = collections.Counter()
    after = collections.Counter()
    written = 0
    with gzip.open(SRC, "rt") as fin, gzip.open(DST, "wt") as fout:
        for line in fin:
            r = json.loads(line)
            t = target_lang(r)
            before[t] += 1
            n = WEIGHT.get(t, 1)
            for _ in range(n):
                fout.write(line)
                after[t] += 1
                written += 1
    print(f"wrote {written} pairs -> {DST}")
    print("target-language mix before:", dict(sorted(before.items())))
    print("target-language mix after :", dict(sorted(after.items())))
    tot_b, tot_a = sum(before.values()), sum(after.values())
    for t in sorted(after):
        print(f"  {t}: {before[t]/tot_b:6.2%} -> {after[t]/tot_a:6.2%}")


if __name__ == "__main__":
    main()
