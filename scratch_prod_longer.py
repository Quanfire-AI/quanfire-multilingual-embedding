#!/usr/bin/env python3
"""Epoch-curve verdict for prod-a70s30: does a longer run beat one epoch?

Scores five models — base e5, v2 (incumbent), and prod-a70s30 at 1 / 2 / 3
epochs — on all three published instruments, each with the byte-identical
protocol of scratch_hn_verdict:
  A. hi-pivot mixed-pool   (recall@10)
  B. non-Hindi X<->Y       (recall@1, in-domain)
  C. FLORES-200 public     (non-Hindi recall@1, held out)

The one-epoch prod-a70s30 already beats v2 on all three. This decides the
promotion epoch count on the measured curve rather than assuming more is
better — more epochs on in-domain-heavy data could lift in-domain while
regressing the held-out FLORES benchmark.

Writes reports/prod-longer-verdict.json.
"""
from __future__ import annotations
import gzip, json, collections
from pathlib import Path
import numpy as np

from multilingual_embedding.corpus.pairs import MinedPair
from multilingual_embedding.pipelines.adaptation import prefixed
from multilingual_embedding.pipelines.search import SemanticSearchPipeline
from multilingual_embedding.embedding.neural import PretrainedTextEncoder

CKPT = "intfloat/multilingual-e5-small"
MODELS = [
    ("e5",             lambda: PretrainedTextEncoder.load(CKPT, pooling="mean", max_length=256)),
    ("v2",             lambda: SemanticSearchPipeline.from_adapter("models/indic-aligned-multi-np").encoder),
    ("prod-a70s30-e1", lambda: SemanticSearchPipeline.from_adapter("models/prod-a70s30").encoder),
    ("prod-a70s30-e2", lambda: SemanticSearchPipeline.from_adapter("models/prod-a70s30-e2").encoder),
    ("prod-a70s30-e3", lambda: SemanticSearchPipeline.from_adapter("models/prod-a70s30-e3").encoder),
]
NAMES = [n for n, _ in MODELS]

PIVOT_EVAL  = Path("data/pairs/indic-aligned-eval.jsonl.gz")
XLING_EVAL  = Path("data/pairs/indic-aligned-nonpivot-eval.jsonl.gz")
FLORES_EVAL = Path("flores-indic-devtest.jsonl")
FLORES_LANGS = ["hi", "bn", "gu", "kn", "ml", "mr", "sa", "ta", "te", "ur"]
OUT = Path("reports/prod-longer-verdict.json")


def load_pairs(path: Path) -> list[MinedPair]:
    out = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(MinedPair.from_record(json.loads(line)))
    return out


def valid(v: np.ndarray) -> np.ndarray:
    return np.isfinite(v).all(1) & (np.linalg.norm(v, axis=1) > 1e-8)


# =============================== A: hi-pivot ===============================
def instrument_pivot() -> dict:
    print("\n########## A. hi-pivot mixed-pool ##########", flush=True)
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
        }
        o = res[name]
        print(f"{name:15s} r@1 {o['recall_at_1']:.4f}  r@10 {o['recall_at_10']:.4f}  MRR {o['mrr']:.4f}", flush=True)
    res["_scored"] = int(good.sum())
    return res


# =============================== B: X<->Y ================================
def instrument_xling() -> dict:
    print("\n########## B. non-Hindi X<->Y (in-domain) ##########", flush=True)
    passage, pairs = {}, []
    for line in gzip.open(XLING_EVAL, "rt", encoding="utf-8"):
        r = json.loads(line)
        c = r["document"]
        passage[(c, r["anchor_language"])] = r["anchor"]
        passage[(c, r["positive_language"])] = r["positive"]
        pairs.append((c, r["anchor_language"], r["positive_language"]))
    keys = list(passage)
    idx = {k: i for i, k in enumerate(keys)}
    texts = [passage[k] for k in keys]
    pools = collections.defaultdict(list)
    for (c, lang) in keys:
        pools[lang].append((c, idx[(c, lang)]))
    pool_rows = {lang: np.array([i for _, i in items]) for lang, items in pools.items()}
    pool_cluster = {lang: [c for c, _ in items] for lang, items in pools.items()}
    pool_pos = {lang: {c: j for j, c in enumerate(cs)} for lang, cs in pool_cluster.items()}
    print(f"{len(pairs)} directed pairs, {len(keys)} unique passages", flush=True)

    res = {}
    for name, build in MODELS:
        enc = build()
        Q = np.asarray(enc.encode_batch(["query: " + t for t in texts]), dtype=np.float32)
        C = np.asarray(enc.encode_batch(["passage: " + t for t in texts]), dtype=np.float32)
        Q /= np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9
        C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-9
        del enc
        per = collections.defaultdict(lambda: [0, 0, 0.0, 0])
        for (c, xl, yl) in pairs:
            q = Q[idx[(c, xl)]]
            sims = C[pool_rows[yl]] @ q
            own = sims[pool_pos[yl][c]]
            rank = int((sims >= own).sum() - 1)
            b = per[yl]
            b[0] += rank < 1; b[1] += rank < 10; b[2] += 1.0 / (rank + 1); b[3] += 1
        tot = [sum(per[yl][i] for yl in per) for i in range(4)]
        res[name] = {
            "recall_at_1": round(tot[0] / tot[3], 4),
            "recall_at_10": round(tot[1] / tot[3], 4),
            "mrr": round(tot[2] / tot[3], 4),
            "n": tot[3],
        }
        o = res[name]
        print(f"{name:15s} r@1 {o['recall_at_1']:.4f}  r@10 {o['recall_at_10']:.4f}  MRR {o['mrr']:.4f}  (n={o['n']})", flush=True)
    return res


# =============================== C: FLORES ==============================
def instrument_flores() -> dict:
    print("\n########## C. FLORES-200 public (held out) ##########", flush=True)
    by_lang = collections.defaultdict(dict)
    for line in open(FLORES_EVAL, encoding="utf-8"):
        r = json.loads(line)
        by_lang[r["lang"]][r["id"]] = r["text"]
    ids = sorted(next(iter(by_lang.values())))
    texts = {L: [by_lang[L][i] for i in ids] for L in FLORES_LANGS}
    n = len(ids)
    gold = np.arange(n)
    all_pairs = [(X, Y) for X in FLORES_LANGS for Y in FLORES_LANGS if X != Y]
    nonhi = [(X, Y) for (X, Y) in all_pairs if X != "hi" and Y != "hi"]
    print(f"{len(FLORES_LANGS)} langs x {n} sentences", flush=True)

    res = {}
    for name, build in MODELS:
        enc = build()
        Q, C = {}, {}
        for L in FLORES_LANGS:
            Q[L] = np.asarray(enc.encode_batch(["query: " + t for t in texts[L]]), dtype=np.float32)
            C[L] = np.asarray(enc.encode_batch(["passage: " + t for t in texts[L]]), dtype=np.float32)
            Q[L] /= np.linalg.norm(Q[L], axis=1, keepdims=True) + 1e-9
            C[L] /= np.linalg.norm(C[L], axis=1, keepdims=True) + 1e-9
        del enc
        pr = {}
        for X in FLORES_LANGS:
            for Y in FLORES_LANGS:
                if X == Y:
                    continue
                pr[(X, Y)] = float((Q[X] @ C[Y].T).argmax(1).__eq__(gold).mean())
        res[name] = {
            "all_pairs_r1": round(sum(pr[p] for p in all_pairs) / len(all_pairs), 4),
            "non_hindi_r1": round(sum(pr[p] for p in nonhi) / len(nonhi), 4),
        }
        o = res[name]
        print(f"{name:15s}  all {len(all_pairs)}: {o['all_pairs_r1']:.4f}   non-Hindi {len(nonhi)}: {o['non_hindi_r1']:.4f}", flush=True)
    return res


def main() -> int:
    out = {
        "models": NAMES,
        "note": "prod-a70s30 epoch curve (1/2/3) vs e5 and v2, three published instruments",
        "pivot": instrument_pivot(),
        "xling": instrument_xling(),
        "flores": instrument_flores(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nWrote {OUT}", flush=True)

    print("\n============= EPOCH CURVE (all three published instruments) =============", flush=True)
    print(f"{'model':15s} {'in-domain r@1':>14s} {'hi-pivot r@10':>14s} {'FLORES nonHi':>14s}", flush=True)
    for name in NAMES:
        xy = out["xling"][name]["recall_at_1"]
        pv = out["pivot"][name]["recall_at_10"]
        fl = out["flores"][name]["non_hindi_r1"]
        print(f"{name:15s} {xy:>14.4f} {pv:>14.4f} {fl:>14.4f}", flush=True)

    # promotion: which epoch count beats v2 on all three AND maximises the min-margin
    v2 = out
    def beats_all(name):
        return (out["xling"][name]["recall_at_1"]  > out["xling"]["v2"]["recall_at_1"] and
                out["pivot"][name]["recall_at_10"] > out["pivot"]["v2"]["recall_at_10"] and
                out["flores"][name]["non_hindi_r1"] > out["flores"]["v2"]["non_hindi_r1"])
    print("\n---- beats v2 on ALL THREE? ----", flush=True)
    for name in ["prod-a70s30-e1", "prod-a70s30-e2", "prod-a70s30-e3"]:
        print(f"  {name:15s} {beats_all(name)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
