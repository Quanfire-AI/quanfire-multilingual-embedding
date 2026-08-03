#!/usr/bin/env python3
"""Global-language baseline: what do we ALREADY do on non-Indian languages?

Before adapting for global languages we measure the current models on a
held-out FLORES-200 slice covering languages outside the Indian set. Same
discipline as the sa/ur null result: do not assume a gap exists, and do not
assume our Indic adaptation left the rest of the space untouched.

Scores three models with the byte-identical FLORES protocol of
scratch_prod_longer.py (manual query:/passage: prefixing, cosine recall@1):
  - e5            base multilingual-e5-small (claims ~100 languages)
  - v2            the previous canonical Indic adapter
  - prod-a70s30   the SHIPPED Indic adapter

For every ordered language pair X->Y it computes cross-lingual recall@1, then
reports three things that decide the next move:
  1. overall all-pairs recall@1 per model
  2. PER LANGUAGE, that language averaged over every partner (as query and as
     target) -- so a single weak language cannot hide inside the average
  3. prod-a70s30 minus e5 per language -- the regression check: a negative
     number means our Indic adaptation HURT that global language.

Reads flores-global-devtest.jsonl ({"lang","id","text"} per line, same schema
as flores-indic-devtest.jsonl). Writes reports/global-baseline-verdict.json.

Run on the GPU box with the guarded venv:
  ~/projects/quanfire-multilingual-embedding/.venv/bin/python scratch_global_baseline.py
"""
from __future__ import annotations
import json, collections
from pathlib import Path
import numpy as np

from multilingual_embedding.pipelines.search import SemanticSearchPipeline
from multilingual_embedding.embedding.neural import PretrainedTextEncoder

CKPT = "intfloat/multilingual-e5-small"
# (name, builder, adapter_dir_or_None). A model whose adapter_dir is missing is
# skipped with a warning, so this stays runnable before prod-a70s30-fr is trained
# (French-recovery run) and after it lands, without editing the list again.
MODELS = [
    ("e5",             lambda: PretrainedTextEncoder.load(CKPT, pooling="mean", max_length=256), None),
    ("v2",             lambda: SemanticSearchPipeline.from_adapter("models/indic-aligned-multi-np").encoder, "models/indic-aligned-multi-np"),
    ("prod-a70s30",    lambda: SemanticSearchPipeline.from_adapter("models/prod-a70s30").encoder, "models/prod-a70s30"),
    ("prod-a70s30-fr", lambda: SemanticSearchPipeline.from_adapter("models/prod-a70s30-fr").encoder, "models/prod-a70s30-fr"),
]
MODELS = [(n, b) for (n, b, p) in MODELS if p is None or Path(p).is_dir()]
NAMES = [n for n, _ in MODELS]

# The global set the baseline is measured on. Generous on purpose: FLORES-200
# scoring is inference on a small model, so measuring 15 languages costs about
# the same as measuring 7. Adaptation (the costly part) is narrowed later to
# only the languages this baseline shows a real gap for. The scorer is
# language-agnostic; extend the list (and the devtest file) to widen coverage.
GLOBAL_LANGS = ["en", "fr", "de", "es", "it", "pt", "ru", "ar", "tr", "zh", "ja", "ko", "th", "vi", "id"]

FLORES_EVAL = Path("flores-global-devtest.jsonl")
OUT = Path("reports/global-baseline-verdict.json")

# Batch above the default 32 to cut MPS round-trips, but not so high it
# exceeds the Mac's ~6.8 GB MPS cap at max_length 256. 96 is ~3x fewer
# device calls than the default and fits comfortably.
ENCODE_BATCH = 96


def load_texts() -> tuple[list[str], dict[str, list[str]]]:
    """Return the shared id order and per-language aligned sentence lists."""
    by_lang: dict[str, dict[str, str]] = collections.defaultdict(dict)
    for line in open(FLORES_EVAL, encoding="utf-8"):
        r = json.loads(line)
        by_lang[r["lang"]][r["id"]] = r["text"]

    missing = [L for L in GLOBAL_LANGS if L not in by_lang]
    if missing:
        raise SystemExit(f"{FLORES_EVAL} is missing languages: {missing}")

    # FLORES is line-aligned: every language must expose the same id set.
    ids = sorted(by_lang[GLOBAL_LANGS[0]])
    for L in GLOBAL_LANGS:
        if sorted(by_lang[L]) != ids:
            raise SystemExit(f"id set for {L} does not match {GLOBAL_LANGS[0]}")
    texts = {L: [by_lang[L][i] for i in ids] for L in GLOBAL_LANGS}
    return ids, texts


def _guard_device(device: object) -> None:
    """Refuse to score on Apple MPS (checks the encoder's *resolved* device).

    The MPS backend on this Mac is non-deterministic on this exact workload: two
    identical runs produced wildly different per-language numbers (base e5
    all-pairs 0.814 vs 0.927; Indonesian 0.295 vs 0.927) and once manufactured a
    phantom French regression that nearly drove a wrong data-recipe decision. On
    CUDA the same script is byte-for-byte reproducible. So this baseline is
    CUDA-only; if CUDA is absent, force CPU (slow but deterministic), never MPS.
    Set QFME_ALLOW_MPS=1 only to deliberately reproduce the bug.
    """
    import os
    if "mps" not in str(device).lower():
        return
    if os.environ.get("QFME_ALLOW_MPS") == "1":
        print("!! QFME_ALLOW_MPS=1: scoring on MPS is UNTRUSTWORTHY (non-reproducible). Results are NOT evidence.", flush=True)
        return
    raise SystemExit(
        f"REFUSING TO RUN: encoder resolved to device '{device}'. Apple MPS is "
        "non-deterministic on this scorer and silently corrupts embeddings (it "
        "invented a phantom French regression once). Run this on the GPU box "
        "(CUDA), or force the encoder to device='cpu'. Override (to reproduce the "
        "bug) with QFME_ALLOW_MPS=1."
    )


def main() -> int:
    ids, texts = load_texts()
    n = len(ids)
    gold = np.arange(n)
    pairs = [(X, Y) for X in GLOBAL_LANGS for Y in GLOBAL_LANGS if X != Y]
    print(f"{len(GLOBAL_LANGS)} langs x {n} sentences -> {len(pairs)} ordered pairs", flush=True)

    per_model: dict[str, dict] = {}
    for name, build in MODELS:
        enc = build()
        enc._batch_size = ENCODE_BATCH  # override the default 32 for MPS/CUDA throughput
        _guard_device(enc.device)
        print(f"[{name}] device {enc.device}  batch {enc._batch_size}", flush=True)
        Q, C = {}, {}
        for L in GLOBAL_LANGS:
            Q[L] = np.asarray(enc.encode_batch(["query: " + t for t in texts[L]]), dtype=np.float32)
            C[L] = np.asarray(enc.encode_batch(["passage: " + t for t in texts[L]]), dtype=np.float32)
            Q[L] /= np.linalg.norm(Q[L], axis=1, keepdims=True) + 1e-9
            C[L] /= np.linalg.norm(C[L], axis=1, keepdims=True) + 1e-9
        del enc

        pr = {}
        for (X, Y) in pairs:
            pr[(X, Y)] = float((Q[X] @ C[Y].T).argmax(1).__eq__(gold).mean())

        # Per language: average over every pair that language takes part in,
        # as query (X) and as target (Y). A weak language shows up here even
        # when the all-pairs mean looks healthy.
        per_lang = {}
        for L in GLOBAL_LANGS:
            involving = [pr[p] for p in pairs if L in p]
            per_lang[L] = round(sum(involving) / len(involving), 4)

        per_model[name] = {
            "all_pairs_r1": round(sum(pr.values()) / len(pr), 4),
            "per_language_r1": per_lang,
        }
        o = per_model[name]
        print(f"{name:12s}  all-pairs {o['all_pairs_r1']:.4f}   "
              + "  ".join(f"{L}:{per_lang[L]:.3f}" for L in GLOBAL_LANGS), flush=True)

    # Regression check: shipped Indic adapter vs base, per language.
    base = per_model["e5"]["per_language_r1"]
    prod = per_model["prod-a70s30"]["per_language_r1"]
    delta = {L: round(prod[L] - base[L], 4) for L in GLOBAL_LANGS}
    regressions = {L: d for L, d in delta.items() if d < 0}

    out = {
        "note": "Global-language held-out baseline on FLORES-200. Measure-first, "
                "before any global adaptation. Delta = prod-a70s30 minus base e5.",
        "languages": GLOBAL_LANGS,
        "sentences": n,
        "models": NAMES,
        "results": per_model,
        "prod_minus_base_per_language": delta,
        "regressions_vs_base": regressions,
    }

    # French-recovery check: only present once prod-a70s30-fr has been trained.
    # Reports the fix vs the shipped adapter per language -- did French recover,
    # and did folding in en<->fr cost anything elsewhere?
    if "prod-a70s30-fr" in per_model:
        fr_model = per_model["prod-a70s30-fr"]["per_language_r1"]
        fr_vs_prod = {L: round(fr_model[L] - prod[L], 4) for L in GLOBAL_LANGS}
        fr_vs_base = {L: round(fr_model[L] - base[L], 4) for L in GLOBAL_LANGS}
        out["fr_recovery"] = {
            "note": "prod-a70s30-fr = prod-a70s30 blend + ~30k en<->fr OPUS-100 pairs.",
            "fr_minus_prod_per_language": fr_vs_prod,
            "fr_minus_base_per_language": fr_vs_base,
            "still_regressed_vs_base": {L: d for L, d in fr_vs_base.items() if d < 0},
            "new_regressions_vs_prod": {L: d for L, d in fr_vs_prod.items() if d < 0},
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nWrote {OUT}", flush=True)

    print("\n===== prod-a70s30 vs base e5, per language (negative = regressed) =====", flush=True)
    for L in GLOBAL_LANGS:
        flag = "  <-- REGRESSED" if delta[L] < 0 else ""
        print(f"  {L:3s} base {base[L]:.4f}  prod {prod[L]:.4f}  delta {delta[L]:+.4f}{flag}", flush=True)
    if regressions:
        print(f"\n{len(regressions)} language(s) regressed vs base: {regressions}", flush=True)
    else:
        print("\nNo global language regressed vs base.", flush=True)

    if "prod-a70s30-fr" in per_model:
        fr_model = per_model["prod-a70s30-fr"]["per_language_r1"]
        print("\n===== prod-a70s30-fr (French-recovery) vs shipped prod, per language =====", flush=True)
        for L in GLOBAL_LANGS:
            d = round(fr_model[L] - prod[L], 4)
            mark = "  <-- FRENCH" if L == "fr" else ("  <-- new drop" if d < -0.005 else "")
            print(f"  {L:3s} prod {prod[L]:.4f}  fr-fix {fr_model[L]:.4f}  delta {d:+.4f}{mark}", flush=True)
        fr_base_delta = round(fr_model["fr"] - base["fr"], 4)
        print(f"\nFrench: base {base['fr']:.4f} -> prod {prod['fr']:.4f} -> fr-fix {fr_model['fr']:.4f} "
              f"(fr-fix vs base {fr_base_delta:+.4f})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
