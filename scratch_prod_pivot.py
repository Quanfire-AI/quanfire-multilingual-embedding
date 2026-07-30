#!/usr/bin/env python3
"""Re-baseline the hi-pivot mixed-pool instrument on prod-a70s30.

The production sweep already re-baselined two of v2's three published
instruments (X<->Y in-domain r@1 and FLORES non-Hindi r@1, both in
reports/prod-flores-verdict.json). The third — hi-pivot mixed-pool
recall@10 — was still measured on v2. This scores it on prod-a70s30
using the byte-identical protocol of scratch_hn_verdict.instrument_pivot,
so the number is comparable to the published 0.8852.

Writes reports/prod-pivot-verdict.json.
"""
from __future__ import annotations
import gzip, json
from pathlib import Path
import numpy as np

from multilingual_embedding.corpus.pairs import MinedPair
from multilingual_embedding.pipelines.adaptation import prefixed
from multilingual_embedding.pipelines.search import SemanticSearchPipeline
from multilingual_embedding.embedding.neural import PretrainedTextEncoder

CKPT = "intfloat/multilingual-e5-small"
MODELS = [
    ("e5",          lambda: PretrainedTextEncoder.load(CKPT, pooling="mean", max_length=256)),
    ("v2",          lambda: SemanticSearchPipeline.from_adapter("models/indic-aligned-multi-np").encoder),
    ("prod-a70s30", lambda: SemanticSearchPipeline.from_adapter("models/prod-a70s30").encoder),
]
NAMES = [n for n, _ in MODELS]

PIVOT_EVAL = Path("data/pairs/indic-aligned-eval.jsonl.gz")
OUT = Path("reports/prod-pivot-verdict.json")


def load_pairs(path: Path) -> list[MinedPair]:
    out = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(MinedPair.from_record(json.loads(line)))
    return out


def valid(v: np.ndarray) -> np.ndarray:
    return np.isfinite(v).all(1) & (np.linalg.norm(v, axis=1) > 1e-8)


def instrument_pivot() -> dict:
    print("\n########## hi-pivot mixed-pool ##########", flush=True)
    raw = load_pairs(PIVOT_EVAL)
    seen, deduped = set(), []
    for p in raw:
        if p.positive in seen:
            continue
        seen.add(p.positive)
        deduped.append(p)
    held = prefixed(deduped, "query: ", "passage: ")
    lang = np.array([p.language for p in held])
    anchors = [p.anchor for p in held]
    positives = [p.positive for p in held]
    print(f"loaded {len(raw)} pairs -> {len(held)} unique-positive", flush=True)

    vecs, good = {}, np.ones(len(held), bool)
    for name, build in MODELS:
        enc = build()
        a = np.asarray(enc.encode_batch(anchors), dtype=np.float64)
        p = np.asarray(enc.encode_batch(positives), dtype=np.float64)
        vecs[name] = (a, p)
        good &= valid(a) & valid(p)
        del enc
    print(f"scoring {int(good.sum())} pairs valid under all models", flush=True)

    anc_kept = [t for t, g in zip(anchors, good) if g]
    pos_kept = [t for t, g in zip(positives, good) if g]
    ids = {t: i for i, t in enumerate(dict.fromkeys(anc_kept + pos_kept))}
    anc_id = np.array([ids[t] for t in anc_kept])
    pos_id = np.array([ids[t] for t in pos_kept])
    self_mask = anc_id[:, None] == pos_id[None, :]
    np.fill_diagonal(self_mask, False)
    lang_kept = lang[good]

    def ranks_for(a, p):
        a = a / np.linalg.norm(a, axis=1, keepdims=True)
        p = p / np.linalg.norm(p, axis=1, keepdims=True)
        sim = a @ p.T
        sim = np.where(self_mask, -np.inf, sim)
        own = sim[np.arange(len(a)), np.arange(len(a))]
        return (sim >= own[:, None]).sum(1) - 1

    res = {}
    for name in NAMES:
        a, p = vecs[name]
        r = ranks_for(a[good], p[good])
        res[name] = {
            "recall_at_1": round(float((r < 1).mean()), 4),
            "recall_at_10": round(float((r < 10).mean()), 4),
            "mrr": round(float((1.0 / (r + 1)).mean()), 4),
            "by_lang_r1": {L: round(float((r[lang_kept == L] < 1).mean()), 4)
                           for L in sorted(set(lang_kept.tolist()))},
        }
        o = res[name]
        print(f"{name:12s} r@1 {o['recall_at_1']:.4f}  r@10 {o['recall_at_10']:.4f}  MRR {o['mrr']:.4f}", flush=True)
    res["_scored"] = int(good.sum())
    return res


def main() -> int:
    out = {"models": NAMES, "instrument": "hi-pivot mixed-pool", "pivot": instrument_pivot()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nWrote {OUT}", flush=True)

    d = out["pivot"]
    print("\n============= hi-pivot mixed-pool recall@10 =============", flush=True)
    for name in NAMES:
        print(f"  {name:12s} r@10 {d[name]['recall_at_10']:.4f}", flush=True)
    delta = d["prod-a70s30"]["recall_at_10"] - d["v2"]["recall_at_10"]
    print(f"\n  prod-a70s30 vs v2 (published 0.8852): {delta:+.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
