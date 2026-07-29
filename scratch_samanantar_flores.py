#!/usr/bin/env python3
"""Does sentence-scale public bitext close the FLORES gap?

Scores three encoders on two instruments, identical methodology to the
committed scorers (query side "query: ", candidate "passage: ", unit-norm):

  A. FLORES-200 devtest, non-Hindi X<->Y recall@1  -- the held-out target
     metric. base E5 0.985, v2 0.961; can sentence bitext close it?
  B. In-domain non-Hindi X<->Y recall@1 (article-scale Wikipedia eval) --
     the cost side: does training on sentences hurt article retrieval?

The samanantar-proof adapter trained ONLY on Samanantar en<->indic sentence
pairs (240k, 8 langs). FLORES is never trained on. No leakage.
"""
import json, gzip, collections
import numpy as np
from multilingual_embedding.pipelines.search import SemanticSearchPipeline
from multilingual_embedding.embedding.neural import PretrainedTextEncoder

CKPT = "intfloat/multilingual-e5-small"
LANGS = ["hi", "bn", "gu", "kn", "ml", "mr", "sa", "ta", "te", "ur"]
FLORES = "flores-indic-devtest.jsonl"
XLING = "data/pairs/indic-aligned-nonpivot-eval.jsonl.gz"
MODELS = [
    ("published-e5", lambda: PretrainedTextEncoder.load(CKPT, pooling="mean", max_length=256)),
    ("v2", lambda: SemanticSearchPipeline.from_adapter("models/indic-aligned-multi-np").encoder),
    ("samanantar-proof", lambda: SemanticSearchPipeline.from_adapter("models/samanantar-proof").encoder),
    ("mix", lambda: SemanticSearchPipeline.from_adapter("models/indic-aligned-mix").encoder),
]


def encode(enc, texts):
    Q = np.asarray(enc.encode_batch(["query: " + t for t in texts]), dtype=np.float32)
    C = np.asarray(enc.encode_batch(["passage: " + t for t in texts]), dtype=np.float32)
    Q /= np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9
    C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-9
    return Q, C


def flores_score(enc):
    by_lang = collections.defaultdict(dict)
    for line in open(FLORES, encoding="utf-8"):
        r = json.loads(line)
        by_lang[r["lang"]][r["id"]] = r["text"]
    ids = sorted(next(iter(by_lang.values())))
    texts = {L: [by_lang[L][i] for i in ids] for L in LANGS}
    n = len(ids)
    Q, C = {}, {}
    for L in LANGS:
        q, c = encode(enc, texts[L])
        Q[L], C[L] = q, c
    gold = np.arange(n)
    pair = {}
    for X in LANGS:
        for Y in LANGS:
            if X == Y:
                continue
            pair[(X, Y)] = float((Q[X].dot(C[Y].T).argmax(1) == gold).mean())
    nonhi = [(X, Y) for X in LANGS for Y in LANGS if X != Y and X != "hi" and Y != "hi"]
    allp = [(X, Y) for X in LANGS for Y in LANGS if X != Y]
    return sum(pair[p] for p in allp) / len(allp), sum(pair[p] for p in nonhi) / len(nonhi)


def xling_score(enc):
    passage, pairs = {}, []
    for line in gzip.open(XLING, "rt"):
        r = json.loads(line)
        c = r["document"]
        passage[(c, r["anchor_language"])] = r["anchor"]
        passage[(c, r["positive_language"])] = r["positive"]
        pairs.append((c, r["anchor_language"], r["positive_language"]))
    keys = list(passage)
    idx = {k: i for i, k in enumerate(keys)}
    texts = [passage[k] for k in keys]
    Q, C = encode(enc, texts)
    pools = collections.defaultdict(list)
    for (c, lang) in keys:
        pools[lang].append((c, idx[(c, lang)]))
    pool_rows = {lang: np.array([i for _, i in items]) for lang, items in pools.items()}
    pool_pos = {lang: {c: j for j, (c, _) in enumerate(items)} for lang, items in pools.items()}
    hit = tot = 0
    hit_nonhi = tot_nonhi = 0
    for (c, xl, yl) in pairs:
        q = Q[idx[(c, xl)]]
        sims = C[pool_rows[yl]].dot(q)
        own = sims[pool_pos[yl][c]]
        rank = int((sims >= own).sum() - 1)
        ok = rank < 1
        hit += ok; tot += 1
        if xl != "hi" and yl != "hi":
            hit_nonhi += ok; tot_nonhi += 1
    return hit / tot, hit_nonhi / max(tot_nonhi, 1)


def main():
    rows = []
    for name, build in MODELS:
        enc = build()
        fl_all, fl_nonhi = flores_score(enc)
        xl_all, xl_nonhi = xling_score(enc)
        del enc
        rows.append((name, fl_nonhi, fl_all, xl_nonhi, xl_all))
        print(f"{name:18s} FLORES non-hi {fl_nonhi:.4f} (all {fl_all:.4f})   "
              f"in-domain non-hi {xl_nonhi:.4f} (all {xl_all:.4f})", flush=True)

    print("\n===================== verdict =====================", flush=True)
    R = {r[0]: r for r in rows}
    base, v2 = R["published-e5"], R["v2"]
    print(f"{'model':18s} {'FLORES non-hi':>13s} {'in-domain non-hi':>17s}", flush=True)
    for name in ["published-e5", "v2", "samanantar-proof", "mix"]:
        if name in R:
            print(f"{name:18s} {R[name][1]:>13.4f} {R[name][3]:>17.4f}", flush=True)
    if "mix" in R:
        mix = R["mix"]
        print(f"\nMix goal: keep v2 in-domain AND close FLORES.", flush=True)
        print(f"  FLORES:    v2 {v2[1]:.4f} -> mix {mix[1]:.4f} ({mix[1]-v2[1]:+.4f})   "
              f"vs base {base[1]:.4f} (gap {mix[1]-base[1]:+.4f})", flush=True)
        print(f"  in-domain: v2 {v2[3]:.4f} -> mix {mix[3]:.4f} ({mix[3]-v2[3]:+.4f})", flush=True)
        held_indomain = mix[3] >= v2[3] - 0.01
        closed_flores = mix[1] >= base[1] - 0.005
        print(f"\n  Held in-domain (within 0.01 of v2)? {'YES' if held_indomain else 'NO'}", flush=True)
        print(f"  Closed FLORES (within 0.005 of base)? {'YES' if closed_flores else 'NO'}", flush=True)
        print(f"  BOTH-AT-ONCE: {'WIN' if held_indomain and closed_flores else 'partial/no'}", flush=True)
    json.dump({r[0]: {"flores_nonhi": r[1], "flores_all": r[2],
                      "indomain_nonhi": r[3], "indomain_all": r[4]} for r in rows},
              open("reports/samanantar-proof-verdict.json", "w"), indent=2)
    print("\nWrote reports/samanantar-proof-verdict.json", flush=True)


if __name__ == "__main__":
    main()
