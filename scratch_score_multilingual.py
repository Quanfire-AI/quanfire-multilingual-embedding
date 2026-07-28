"""
Score the multilingual aligned adapter against published e5, per language.

Same honest protocol as the hi<->ta gate: the document-disjoint eval,
positives deduplicated so an identical passage can't tie with the answer,
one identical candidate pool per model (degenerate encodings dropped as a
union), unit vectors. recall@1 / @10 / MRR are broken out by the
positive's language, so a gain in one language can't hide a loss in
another. The pool mixes every language, so each query competes against
same-language distractors -- the hard, realistic cross-lingual test.
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np

from multilingual_embedding.corpus.pairs import MinedPair
from multilingual_embedding.pipelines.adaptation import prefixed
from multilingual_embedding.pipelines.search import SemanticSearchPipeline
from multilingual_embedding.embedding.neural import PretrainedTextEncoder

EVAL = Path("data/pairs/indic-aligned-eval.jsonl.gz")
CHECKPOINT = "intfloat/multilingual-e5-small"
ADAPTER = sys.argv[1] if len(sys.argv) > 1 else "models/indic-aligned-multi"
OUT = Path("reports/optionb/indic-multilingual-score.json")

MODELS = [
    ("published-e5", lambda: PretrainedTextEncoder.load(CHECKPOINT, pooling="mean", max_length=256)),
    ("indic-aligned-multi", lambda: SemanticSearchPipeline.from_adapter(ADAPTER).encoder),
]


def load_pairs(path: Path) -> list[MinedPair]:
    out = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(MinedPair.from_record(json.loads(line)))
    return out


def dedup_positives(pairs: list[MinedPair]) -> list[MinedPair]:
    seen: set[str] = set()
    kept = []
    for p in pairs:
        if p.positive in seen:
            continue
        seen.add(p.positive)
        kept.append(p)
    return kept


def valid(v: np.ndarray) -> np.ndarray:
    return np.isfinite(v).all(1) & (np.linalg.norm(v, axis=1) > 1e-8)


def scores_for(a: np.ndarray, p: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    p = p / np.linalg.norm(p, axis=1, keepdims=True)
    sim = a @ p.T
    n = len(a)
    own = sim[np.arange(n), np.arange(n)]
    return (sim >= own[:, None]).sum(1) - 1  # pessimistic ranks


def summarize(ranks: np.ndarray, lang: np.ndarray) -> dict:
    def block(mask):
        r = ranks[mask]
        return {
            "queries": int(mask.sum()),
            "recall_at_1": round(float((r < 1).mean()), 4),
            "recall_at_10": round(float((r < 10).mean()), 4),
            "mrr": round(float((1.0 / (r + 1)).mean()), 4),
        }
    out = {"overall": block(np.ones(len(ranks), bool)), "by_positive_language": {}}
    for L in sorted(set(lang.tolist())):
        out["by_positive_language"][L] = block(lang == L)
    return out


def main() -> int:
    raw = load_pairs(EVAL)
    deduped = dedup_positives(raw)
    held = prefixed(deduped, "query: ", "passage: ")
    lang = np.array([p.language for p in held])
    print(f"loaded {len(raw)} pairs; {len(raw)-len(deduped)} duplicate positives dropped "
          f"-> {len(held)} unique-positive pairs", flush=True)
    print(f"positive-language mix: { {L:int((lang==L).sum()) for L in sorted(set(lang.tolist()))} }",
          flush=True)

    anchors = [p.anchor for p in held]
    positives = [p.positive for p in held]

    vecs, good = {}, np.ones(len(held), bool)
    for name, build in MODELS:
        enc = build()
        a = np.asarray(enc.encode_batch(anchors), dtype=np.float64)
        p = np.asarray(enc.encode_batch(positives), dtype=np.float64)
        vecs[name] = (a, p)
        m = valid(a) & valid(p)
        print(f"  {name}: {int((~m).sum())} degenerate", flush=True)
        good &= m
        del enc

    lang_kept = lang[good]
    print(f"\nscoring {int(good.sum())} pairs valid under all models "
          f"(dropped {len(held)-int(good.sum())})\n", flush=True)

    results = {}
    for name, (a, p) in vecs.items():
        ranks = scores_for(a[good], p[good])
        results[name] = summarize(ranks, lang_kept)
        o = results[name]["overall"]
        print(f"{name:20s} recall@1 {o['recall_at_1']:.4f}  recall@10 {o['recall_at_10']:.4f}  "
              f"MRR {o['mrr']:.4f}", flush=True)

    # per-language recall@1 table, both models side by side
    langs = sorted(set(lang_kept.tolist()))
    print(f"\n{'lang':6s} " + "  ".join(f"{n[:14]:>14s}" for n, _ in MODELS))
    for L in langs:
        row = "  ".join(f"{results[n]['by_positive_language'][L]['recall_at_1']:14.4f}" for n, _ in MODELS)
        print(f"{L:6s} {row}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"eval_file": str(EVAL), "adapter": ADAPTER,
                               "scored": int(good.sum()), "models": results},
                              indent=2, ensure_ascii=False))
    print(f"\nWrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
