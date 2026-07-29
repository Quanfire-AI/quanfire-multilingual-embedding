#!/usr/bin/env python3
"""Production FLORES sweep verdict: does the both-at-once win survive scale,
and does adding sa/ur sentence data lift sa/ur on FLORES?

Scores six encoders on two instruments, identical methodology to the committed
scorers (query side "query: ", candidate "passage: ", unit-norm):

  A. FLORES-200 devtest, non-Hindi X<->Y recall@1  -- held-out target metric.
  B. In-domain non-Hindi X<->Y recall@1 (article-scale Wikipedia eval) -- cost.

Also reports per-language FLORES recall for sa and ur (each as query or
candidate, non-hi context). The proof mix had NO sa/ur sentence data; the three
prod ratios do (itihasa sa + opus-100 ur), so this isolates that addition.

FLORES is never trained on. No leakage.
"""
import json, gzip, collections, os
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
    ("prod-a30s70", lambda: SemanticSearchPipeline.from_adapter("models/prod-a30s70").encoder),
    ("prod-a50s50", lambda: SemanticSearchPipeline.from_adapter("models/prod-a50s50").encoder),
    ("prod-a70s30", lambda: SemanticSearchPipeline.from_adapter("models/prod-a70s30").encoder),
]


def encode(enc, texts):
    Q = np.asarray(enc.encode_batch(["query: " + t for t in texts]), dtype=np.float32)
    C = np.asarray(enc.encode_batch(["passage: " + t for t in texts]), dtype=np.float32)
    Q /= np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9
    C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-9
    return Q, C


def flores_pairs(enc):
    """Return {(X,Y): recall@1} for every ordered non-equal language pair."""
    by_lang = collections.defaultdict(dict)
    for line in open(FLORES, encoding="utf-8"):
        r = json.loads(line)
        by_lang[r["lang"]][r["id"]] = r["text"]
    ids = sorted(next(iter(by_lang.values())))
    texts = {L: [by_lang[L][i] for i in ids] for L in LANGS}
    n = len(ids)
    Q, C = {}, {}
    for L in LANGS:
        Q[L], C[L] = encode(enc, texts[L])
    gold = np.arange(n)
    pair = {}
    for X in LANGS:
        for Y in LANGS:
            if X != Y:
                pair[(X, Y)] = float((Q[X].dot(C[Y].T).argmax(1) == gold).mean())
    return pair


def summarise(pair):
    allp = [(X, Y) for X in LANGS for Y in LANGS if X != Y]
    nonhi = [p for p in allp if "hi" not in p]
    fl_all = sum(pair[p] for p in allp) / len(allp)
    fl_nonhi = sum(pair[p] for p in nonhi) / len(nonhi)
    perlang = {}
    for L in ["sa", "ur"]:
        rel = [p for p in nonhi if L in p]           # L as query or candidate, non-hi
        perlang[L] = sum(pair[p] for p in rel) / len(rel)
    return fl_all, fl_nonhi, perlang


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
    hit = tot = hit_nonhi = tot_nonhi = 0
    for (c, xl, yl) in pairs:
        q = Q[idx[(c, xl)]]
        sims = C[pool_rows[yl]].dot(q)
        own = sims[pool_pos[yl][c]]
        ok = int((sims >= own).sum() - 1) < 1
        hit += ok; tot += 1
        if xl != "hi" and yl != "hi":
            hit_nonhi += ok; tot_nonhi += 1
    return hit / tot, hit_nonhi / max(tot_nonhi, 1)


def adapter_missing(name):
    return name.startswith("prod-") and not os.path.isdir(f"models/{name}")


def main():
    R = {}
    for name, build in MODELS:
        if adapter_missing(name):
            print(f"{name:18s} SKIPPED (adapter not present yet)", flush=True)
            continue
        enc = build()
        fl_all, fl_nonhi, perlang = summarise(flores_pairs(enc))
        xl_all, xl_nonhi = xling_score(enc)
        del enc
        R[name] = dict(flores_nonhi=fl_nonhi, flores_all=fl_all,
                       flores_sa=perlang["sa"], flores_ur=perlang["ur"],
                       indomain_nonhi=xl_nonhi, indomain_all=xl_all)
        print(f"{name:18s} FLORES non-hi {fl_nonhi:.4f} (sa {perlang['sa']:.4f} ur {perlang['ur']:.4f})"
              f"   in-domain non-hi {xl_nonhi:.4f}", flush=True)

    base, v2, proof = R["published-e5"], R["v2"], R["samanantar-proof"]
    print("\n===================== verdict =====================", flush=True)
    print(f"{'model':18s} {'FLORES nonhi':>12s} {'in-domain':>10s} {'sa':>7s} {'ur':>7s}", flush=True)
    for name in R:
        r = R[name]
        print(f"{name:18s} {r['flores_nonhi']:>12.4f} {r['indomain_nonhi']:>10.4f} "
              f"{r['flores_sa']:>7.4f} {r['flores_ur']:>7.4f}", flush=True)

    # promotion candidate: both-at-once (FLORES within 0.005 of base AND
    # in-domain within 0.01 of v2), then maximise their sum.
    print("\n-- both-at-once test (FLORES within 0.005 of base, in-domain within 0.01 of v2) --", flush=True)
    winners = []
    for name in ["prod-a30s70", "prod-a50s50", "prod-a70s30"]:
        if name not in R:
            continue
        r = R[name]
        closed = r["flores_nonhi"] >= base["flores_nonhi"] - 0.005
        held = r["indomain_nonhi"] >= v2["indomain_nonhi"] - 0.01
        both = closed and held
        print(f"  {name:14s} FLORES {r['flores_nonhi']:.4f} (closed {closed})  "
              f"in-domain {r['indomain_nonhi']:.4f} (held {held})  -> {'WIN' if both else 'partial'}", flush=True)
        if both:
            winners.append((r["flores_nonhi"] + r["indomain_nonhi"], name))
    if winners:
        best = max(winners)[1]
        print(f"\n  PROMOTION CANDIDATE: {best}", flush=True)
    else:
        print("\n  No ratio hit both-at-once; best trade-off must be chosen by hand.", flush=True)

    print("\n-- sa/ur: did sentence data help? (proof had none; prod has itihasa+opus) --", flush=True)
    prod_ref = next((n for n in ["prod-a50s50", "prod-a30s70", "prod-a70s30"] if n in R), None)
    for L in ["sa", "ur"]:
        line = (f"  {L}: base {base['flores_'+L]:.4f}  v2 {v2['flores_'+L]:.4f}  "
                f"proof {proof['flores_'+L]:.4f}")
        if prod_ref:
            line += f"  {prod_ref} {R[prod_ref]['flores_'+L]:.4f}"
        print(line, flush=True)

    json.dump(R, open("reports/prod-flores-verdict.json", "w"), indent=2)
    print("\nWrote reports/prod-flores-verdict.json", flush=True)


if __name__ == "__main__":
    main()
