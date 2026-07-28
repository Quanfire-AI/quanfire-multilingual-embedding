#!/usr/bin/env python3
"""Public benchmark: FLORES-200 devtest cross-lingual retrieval.

FLORES devtest is 1,012 sentences, fully parallel across all ten languages,
from a held-out domain (wikinews/web) we never trained on. For each directed
language pair (X -> Y): query = X sentence i, candidate pool = all 1,012 Y
sentences, correct = i. recall@1 is the standard xsim retrieval accuracy.

Same encoding methodology as our internal evals: query side "query: ",
candidate side "passage: ", unit-normalised. Scores published-e5 and v2 on
the identical pool so the two are directly comparable. No leakage possible:
FLORES is a different corpus from the Wikipedia article pairs we mined.
"""
import json, sys, collections
import numpy as np
from multilingual_embedding.pipelines.search import SemanticSearchPipeline
from multilingual_embedding.embedding.neural import PretrainedTextEncoder

EVAL = "flores-indic-devtest.jsonl"
CKPT = "intfloat/multilingual-e5-small"
LANGS = ["hi", "bn", "gu", "kn", "ml", "mr", "sa", "ta", "te", "ur"]
MODELS = [
    ("published-e5", lambda: PretrainedTextEncoder.load(CKPT, pooling="mean", max_length=256)),
    ("v2", lambda: SemanticSearchPipeline.from_adapter("models/indic-aligned-multi-np").encoder),
]


def main():
    by_lang = collections.defaultdict(dict)  # lang -> {id: text}
    for line in open(EVAL, encoding="utf-8"):
        r = json.loads(line)
        by_lang[r["lang"]][r["id"]] = r["text"]
    ids = sorted(next(iter(by_lang.values())))
    texts = {L: [by_lang[L][i] for i in ids] for L in LANGS}
    n = len(ids)
    print(f"{len(LANGS)} languages x {n} parallel sentences", flush=True)

    results = {}
    for name, build in MODELS:
        enc = build()
        Q, C = {}, {}
        for L in LANGS:
            Q[L] = np.asarray(enc.encode_batch(["query: " + t for t in texts[L]]), dtype=np.float32)
            C[L] = np.asarray(enc.encode_batch(["passage: " + t for t in texts[L]]), dtype=np.float32)
            Q[L] /= np.linalg.norm(Q[L], axis=1, keepdims=True) + 1e-9
            C[L] /= np.linalg.norm(C[L], axis=1, keepdims=True) + 1e-9
        del enc

        pair_r1 = {}          # (X,Y) -> recall@1
        gold = np.arange(n)
        for X in LANGS:
            for Y in LANGS:
                if X == Y:
                    continue
                sims = Q[X] @ C[Y].T           # n x n
                pred = sims.argmax(1)
                pair_r1[(X, Y)] = float((pred == gold).mean())
        results[name] = pair_r1

    def mean(pairs, d):
        return sum(d[p] for p in pairs) / len(pairs)

    all_pairs = [(X, Y) for X in LANGS for Y in LANGS if X != Y]
    nonhi_pairs = [(X, Y) for (X, Y) in all_pairs if X != "hi" and Y != "hi"]

    print("\n================= FLORES-200 cross-lingual recall@1 =================", flush=True)
    for name in results:
        print(f"{name:14s}  all {len(all_pairs)} pairs: {mean(all_pairs, results[name]):.4f}"
              f"   non-Hindi {len(nonhi_pairs)} pairs: {mean(nonhi_pairs, results[name]):.4f}", flush=True)

    e5, v2 = results["published-e5"], results["v2"]
    print("\nPer source language, mean recall@1 over its 9 targets (e5 -> v2):", flush=True)
    for X in LANGS:
        tgt = [(X, Y) for Y in LANGS if Y != X]
        a, b = mean(tgt, e5), mean(tgt, v2)
        print(f"  {X}:  {a:.3f} -> {b:.3f}   ({b-a:+.3f})", flush=True)

    out = {
        "benchmark": "FLORES-200 devtest, cross-lingual recall@1",
        "sentences": n, "languages": LANGS,
        "overall": {name: {"all_pairs": mean(all_pairs, results[name]),
                           "non_hindi_pairs": mean(nonhi_pairs, results[name])}
                    for name in results},
        "per_pair": {name: {f"{X}-{Y}": results[name][(X, Y)] for (X, Y) in all_pairs}
                     for name in results},
    }
    json.dump(out, open("flores-bench-result.json", "w"), indent=2, ensure_ascii=False)
    print("\nWrote flores-bench-result.json", flush=True)


if __name__ == "__main__":
    main()
