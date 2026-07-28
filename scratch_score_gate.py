"""
Gate score: does training on aligned pairs beat published e5 cross-lingual?

Scored on the document-disjoint held-out eval (whole articles never seen
in training), all pairs, both directions. Three models on one identical
pool, unit vectors, degenerate encodings dropped as a union so the pool
stays the same for every model. recall@1 is also broken out by the
positive's language, since a gain in one direction can hide a loss in
the other.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

from multilingual_embedding.corpus.pairs import MinedPair
from multilingual_embedding.pipelines.adaptation import prefixed
from multilingual_embedding.pipelines.search import SemanticSearchPipeline
from multilingual_embedding.embedding.neural import PretrainedTextEncoder

EVAL = Path("data/pairs/hi-ta-aligned-eval.jsonl.gz")
CHECKPOINT = "intfloat/multilingual-e5-small"
OUT = Path("reports/optionb/aligned-gate.json")

MODELS = [
    ("published-e5", lambda: PretrainedTextEncoder.load(CHECKPOINT, pooling="mean", max_length=256)),
    ("indic-b-baseline", lambda: SemanticSearchPipeline.from_adapter("models/indic-b-baseline").encoder),
    ("indic-aligned-hita", lambda: SemanticSearchPipeline.from_adapter("models/indic-aligned-hita").encoder),
]


def load_pairs(path: Path) -> list[MinedPair]:
    out = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(MinedPair.from_record(json.loads(line)))
    return out


def valid(v: np.ndarray) -> np.ndarray:
    return np.isfinite(v).all(1) & (np.linalg.norm(v, axis=1) > 1e-8)


def metrics(a: np.ndarray, p: np.ndarray, lang: list[str]) -> dict:
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    p = p / np.linalg.norm(p, axis=1, keepdims=True)
    sim = a @ p.T
    n = len(a)
    own = sim[np.arange(n), np.arange(n)]
    ranks = (sim >= own[:, None]).sum(1) - 1
    recip = 1.0 / (ranks + 1)
    lang = np.array(lang)
    out = {
        "queries": int(n),
        "recall_at_1": round(float((ranks < 1).mean()), 4),
        "recall_at_5": round(float((ranks < 5).mean()), 4),
        "recall_at_10": round(float((ranks < 10).mean()), 4),
        "mrr": round(float(recip.mean()), 4),
        "by_positive_language": {},
    }
    for L in sorted(set(lang)):
        m = lang == L
        out["by_positive_language"][L] = {
            "queries": int(m.sum()),
            "recall_at_1": round(float((ranks[m] < 1).mean()), 4),
        }
    return out


def dedup_positives(pairs: list[MinedPair]) -> list[MinedPair]:
    """Drop pairs whose positive text already appeared.

    An identical positive present twice in the candidate pool ties with
    the answer under the pessimistic rank, so recall@1 can never be 0 for
    any query -- exactly what evaluate_retrieval guards against. Dedup on
    the raw positive (before prefixing) so the kept set is identical for
    every model.
    """
    seen: set[str] = set()
    kept = []
    for p in pairs:
        if p.positive in seen:
            continue
        seen.add(p.positive)
        kept.append(p)
    return kept


def main() -> int:
    raw = load_pairs(EVAL)
    deduped = dedup_positives(raw)
    print(f"loaded {len(raw)} pairs; {len(raw) - len(deduped)} dropped as "
          f"duplicate positives -> {len(deduped)} unique-positive pairs", flush=True)
    held = prefixed(deduped, "query: ", "passage: ")
    lang = [p.language for p in held]
    print(f"eval {len(held)} pairs; positive-language mix "
          f"{ {L: lang.count(L) for L in sorted(set(lang))} }", flush=True)

    anchors = [p.anchor for p in held]
    positives = [p.positive for p in held]

    enc_vecs = {}
    good = np.ones(len(held), dtype=bool)
    for name, build in MODELS:
        enc = build()
        a = np.asarray(enc.encode_batch(anchors), dtype=np.float64)
        p = np.asarray(enc.encode_batch(positives), dtype=np.float64)
        enc_vecs[name] = (a, p)
        m = valid(a) & valid(p)
        print(f"  {name}: {int((~m).sum())} degenerate", flush=True)
        good &= m
        del enc

    keep = int(good.sum())
    lang_kept = [lang[i] for i in range(len(held)) if good[i]]
    print(f"\nscoring {keep} pairs valid under all models (dropped {len(held)-keep})\n", flush=True)

    results = {}
    for name, (a, p) in enc_vecs.items():
        r = metrics(a[good], p[good], lang_kept)
        results[name] = r
        bl = r["by_positive_language"]
        print(f"{name:20s} recall@1 {r['recall_at_1']:.4f}  recall@10 {r['recall_at_10']:.4f}  "
              f"MRR {r['mrr']:.4f}  | hi->ta {bl.get('ta',{}).get('recall_at_1')}  "
              f"ta->hi {bl.get('hi',{}).get('recall_at_1')}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"eval_file": str(EVAL), "scored": keep, "models": results},
                              indent=2, ensure_ascii=False))
    print(f"\nWrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
