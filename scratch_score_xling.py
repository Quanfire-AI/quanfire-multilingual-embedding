#!/usr/bin/env python3
"""Well-posed non-Hindi cross-lingual retrieval (bitext style).

The mixed-pool scorer is ill-posed on a symmetric X<->Y eval: a query's own
cluster contributes a passage in every language, and same-concept passages tie
the true positive. The fix is to retrieve *within one target language*: query
X-passage of cluster C, candidates = one Y-passage per cluster, correct = C.
Different clusters are different concepts, so rank is meaningful.

Scores published-e5, v1 (pivot-only), and np (pivot+non-pivot) on the SAME
held-out clusters, so the three are directly comparable.
"""
import gzip, json, collections
import numpy as np
from multilingual_embedding.pipelines.search import SemanticSearchPipeline
from multilingual_embedding.embedding.neural import PretrainedTextEncoder

EVAL = "data/pairs/indic-aligned-nonpivot-eval.jsonl.gz"
CKPT = "intfloat/multilingual-e5-small"
MODELS = [
    ("published-e5", lambda: PretrainedTextEncoder.load(CKPT, pooling="mean", max_length=256)),
    ("v1-pivot", lambda: SemanticSearchPipeline.from_adapter("models/indic-aligned-v1").encoder),
    ("np-pivot+nonpivot", lambda: SemanticSearchPipeline.from_adapter("models/indic-aligned-multi-np").encoder),
]


def main():
    # passage[(cluster, lang)] = text  (one per cluster+lang)
    passage, pairs = {}, []
    for line in gzip.open(EVAL, "rt"):
        r = json.loads(line)
        c = r["document"]
        passage[(c, r["anchor_language"])] = r["anchor"]
        passage[(c, r["positive_language"])] = r["positive"]
        pairs.append((c, r["anchor_language"], r["positive_language"]))
    keys = list(passage)
    idx = {k: i for i, k in enumerate(keys)}
    texts = [passage[k] for k in keys]
    print(f"{len(pairs)} directed pairs, {len(keys)} unique passages", flush=True)

    # candidate pools: per target language, one passage per cluster (built once)
    pools = collections.defaultdict(list)  # lang -> list of (cluster, key_index)
    for (c, lang) in keys:
        pools[lang].append((c, idx[(c, lang)]))
    pool_rows = {lang: np.array([i for _, i in items]) for lang, items in pools.items()}
    pool_cluster = {lang: [c for c, _ in items] for lang, items in pools.items()}
    pool_pos = {lang: {c: j for j, c in enumerate(cs)} for lang, cs in pool_cluster.items()}

    for name, build in MODELS:
        enc = build()
        # same methodology as the pivot scorer: query side "query: ", candidate "passage: "
        Q = np.asarray(enc.encode_batch(["query: " + t for t in texts]), dtype=np.float32)
        C = np.asarray(enc.encode_batch(["passage: " + t for t in texts]), dtype=np.float32)
        Q /= np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9
        C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-9
        del enc

        per = collections.defaultdict(lambda: [0, 0, 0.0, 0])  # lang -> r1,r10,mrr_sum,n
        for (c, xl, yl) in pairs:
            q = Q[idx[(c, xl)]]
            cand = C[pool_rows[yl]]
            sims = cand @ q
            correct = pool_pos[yl][c]
            own = sims[correct]
            rank = int((sims >= own).sum() - 1)  # pessimistic
            b = per[yl]
            b[0] += rank < 1
            b[1] += rank < 10
            b[2] += 1.0 / (rank + 1)
            b[3] += 1
        tot = [0, 0, 0.0, 0]
        for yl in per:
            for i in range(4):
                tot[i] += per[yl][i]
        r1, r10, mrr = tot[0] / tot[3], tot[1] / tot[3], tot[2] / tot[3]
        print(f"\n{name:20s} recall@1 {r1:.4f}  recall@10 {r10:.4f}  MRR {mrr:.4f}  (n={tot[3]})", flush=True)
        line = "  ".join(f"{yl}:{per[yl][0]/per[yl][3]:.3f}" for yl in sorted(per))
        print("   per target-lang recall@1:", line, flush=True)


if __name__ == "__main__":
    main()
