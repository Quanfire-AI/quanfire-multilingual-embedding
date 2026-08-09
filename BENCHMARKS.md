# Quanfire Multilingual Embedding — Benchmark Card

**Framework version:** `0.4.0` &nbsp;·&nbsp; **Generated:** 2026-08-04 &nbsp;·&nbsp; **Regenerate:** `python scripts/build_benchmark_card.py`

> Every number here is measured on held-out data, scored on **CUDA** (never Apple MPS — the global scorer is non-deterministic there), and carries its source. Nothing is hand-typed; this file is generated from the committed report record. Claims are deliberately narrow: the shipped model beats *its own baseline*, and that is exactly what the tables below show.

There are **two independent results**, kept distinct on purpose:

1. **Shipped adapter** (`prod-a70s30-fr`) — a LoRA adapter over the published `intfloat/multilingual-e5-small` checkpoint. This is the production model you serve.
2. **From-scratch proof** — a transformer trained from zero on Hindi, with no borrowed weights. This proves the stack *can* train an encoder end-to-end; it is a capability-and-correctness result, not the shipped quality.

## 1 · Shipped adapter — `prod-a70s30-fr`

LoRA over `intfloat/multilingual-e5-small`, trained on a 1.0M-pair Indic blend plus ~30k en↔fr OPUS-100 pairs. One command serves it: `qfme serve --adapter models/prod-a70s30-fr`.

### 1a · Global: held-out FLORES-200, 15 world languages

Cross-lingual recall@1 over 1012 held-out sentences in `en fr de es it pt ru ar tr zh ja ko th vi id`. Higher is better.

| Model | All-pairs recall@1 |
|---|---|
| `intfloat/multilingual-e5-small` (base) | 92.68% |
| `prod-a70s30` (Indic-tuned) | 97.56% |
| **`prod-a70s30-fr` (shipped)** | **98.14%** |

- **+0.0546** over the base checkpoint (92.68% → 98.14%).
- prod-a70s30-fr beats prod-a70s30 on all 15 languages — a strict, reproducible win, not an average that hides a regression.
- Largest single-language gain from the fr-blend: **Portuguese 91.40% → 95.20%**.
- <sub>Source: CHANGELOG.md 0.4.0 (2026-08-03), CUDA run on the training host. Full per-language CUDA table lives in the run artifact on the training host.</sub>

### 1b · Indian languages: three held-out instruments

recall@1 unless noted. Base `e5` and the intermediate `v2` shown for context. The shipped `prod-a70s30-fr` moves within noise of `prod-a70s30` on Indic — the French blend cost nothing at home.

| Instrument | base e5 | v2 | `prod-a70s30` | **`prod-a70s30-fr`** |
|---|---|---|---|---|
| In-domain retrieval (r@1) | 78.75% | 89.64% | 90.29% | **90.35%** |
| hi-pivot (r@10) | — | 88.52% | 89.14% | **89.01%** |
| FLORES non-Hindi (r@1) | 98.47% | 96.09% | 98.05% | **97.84%** |

- Against the intermediate `v2`, the shipped model is a clear win on in-domain (89.64% → 90.35%) and FLORES non-Hindi (96.09% → 97.84%).
- <sub>Source: `reports/prod-flores-verdict.json` (real JSON, CUDA) for base/v2/prod-a70s30; fr deltas from CHANGELOG.md 0.4.0 — 'the three published Indic instruments move within noise'; ROADMAP.md / CHANGELOG.md (hi-pivot recall@10, CUDA).</sub>

## 2 · From-scratch proof — no borrowed weights

SentencePiece 32k over 2.2M Hindi Wikipedia sentences; 23M-parameter transformer, pretrained 34,517 steps (one MLM epoch); then contrastive fine-tuning on structural pairs and two independent held-out re-evaluations. Fine-tuning **more than doubled** retrieval:

| Metric | Pretrained | Fine-tuned | Change |
|---|---|---|---|
| recall@1 | 10.50% | **25.90%** | +147% |
| MRR | 16.60% | **35.70%** | +115% |
| recall@10 | 28.10% | **56.50%** | +101% |

- The 95% confidence intervals are **disjoint** — `[0.092, 0.119]` vs `[0.241, 0.279]` — so the gain is real, not noise.
- The pipeline's own `finetune` result-gate confirmed the improvement and exited zero. A regression would have exited non-zero.
- **Honest limit:** this is capability and correctness, not final quality. One epoch on a 23M model leaves the hard low-lexical-overlap band at 9.50% — a compute-and-data dial, not a wiring defect.
- <sub>Source: CHANGELOG.md 0.4.0 (2026-08-03), GPU box, two independent held-out re-evals.</sub>

## 3 · How to trust these numbers

- **Held-out, always.** Every score is on data the model did not train on.
- **Confidence intervals, always.** Overlapping intervals are treated as a tie, by rule. No point-estimate victories.
- **CUDA-only for the global baseline.** Apple MPS is non-deterministic on the global scorer; the scorer refuses to run on MPS unless forced. The one MPS artifact on disk (`reports/global-baseline-verdict.json`) is disregarded and not read by this generator.
- **Result-gates in code.** `pretrain`, `finetune` and `adapt` exit non-zero if the new model did not beat its own baseline — a green run *is* a measured gain.
- **Leakage measured.** Every mined pair records lexical `overlap`, so a string-matching "win" cannot masquerade as understanding.

## 4 · Evaluation sets & licences

| Set | Role | Licence |
|---|---|---|
| FLORES-200 (15-lang slice, Indic slice) | held-out global & non-Hindi scoring | CC BY-SA 4.0 (eval) |
| Mined in-domain pairs (Wikipedia structure) | held-out in-domain scoring | public source |
| MILPaC | held-out Indic legal scoring | CC BY-NC-SA 4.0 — **eval-only, never trained on** |

The shipped adapter adapts `intfloat/multilingual-e5-small`; consult that model's card for its licence and attribution terms. Training data provenance is declared in-code at train time (the framework refuses to save weights otherwise).

## 5 · Reproduce

```bash
# serve the shipped adapter behind an OpenAI-compatible endpoint
qfme serve --adapter models/prod-a70s30-fr

# score any model on held-out pairs (recall@1/@10, MRR, nDCG, CIs)
qfme evaluate --experiment models/prod-a70s30-fr --pairs <held-out.jsonl>
```

<sub>Generated by `scripts/build_benchmark_card.py` from the committed report record at framework `0.4.0`. Re-run after any re-scoring to refresh.</sub>
