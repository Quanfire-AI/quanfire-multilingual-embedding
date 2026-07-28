# A multilingual aligned adapter for ten Indian languages

**Model:** `models/indic-aligned-v1` — a rank-32 LoRA over `intfloat/multilingual-e5-small`.
**Trained:** 2026-07-28 on an RTX 4070 Ti SUPER, one epoch, ~13 minutes.
**Measured:** leak-free, document-disjoint eval; 2,283 unique-positive pairs across all ten languages.

## What it is

One low-rank adapter (589,824 parameters, 0.50% of the encoder) that teaches
`multilingual-e5-small` to place a passage and its cross-lingual twin near each
other in the same 384-dim space, across Hindi, Bengali, Gujarati, Kannada,
Malayalam, Marathi, Sanskrit, Tamil, Telugu, and Urdu.

The training signal is **mined, not synthesised**: Wikipedia langlinks pair an
article with its human-written counterpart in another language, Hindi as the
pivot. The pairs share meaning, not strings — median lexical overlap is ~0, so
the model cannot cheat by matching substrings; it has to learn the alignment.

The split is **document-disjoint by Hindi pivot cluster**: holding out a Hindi
article removes its text from *every* language's training set at once, so no
eval passage — in any language — was seen during training.

## The headline

On the held-out eval, against the published checkpoint:

| metric | published e5 | **indic-aligned-v1** | change |
|---|---|---|---|
| recall@1 | 0.0648 | **0.0850** | +31% |
| recall@10 | 0.7495 | **0.8879** | **+13.8 pts** |
| MRR | 0.2884 | **0.3443** | +19% |

recall@10 and MRR are the robust story: the right passage lands in the top ten
far more often, in every framing we tried. The largest top-1 gains are exactly
where e5 was weakest — the cross-script cases the adapter exists to fix.

## Per-language recall@1, and the honest part

| lang | e5 | v1 | |
|---|---|---|---|
| ta | 0.004 | **0.038** | 10× — the point of the whole exercise |
| ml | 0.020 | **0.074** | 3.7× |
| bn | 0.024 | **0.059** | 2.4× |
| gu | 0.024 | **0.048** | ↑ |
| hi | 0.076 | **0.114** | ↑ |
| kn | 0.023 | **0.035** | ↑ |
| te | 0.021 | **0.034** | ↑ |
| ur | 0.045 | 0.040 | ≈ (noise) |
| **mr** | **0.290** | 0.274 | **below e5** |
| **sa** | **0.245** | 0.174 | **below e5** |

Seven of ten languages improve at top-1. **Two regress: Marathi and Sanskrit** —
both Devanagari, the same script as the Hindi pivot, and both already strong in
base e5. Adapting the shared space toward *cross-script* alignment moves the
*same-script* Devanagari neighbourhoods that e5 already had right. This is a
genuine tension in a single shared adapter, not a bug.

Two experiments confirmed that reading rather than guessed at it:

- **Doubling the rank (32 → 64) was a wash.** Overall and per-language numbers
  moved within noise; Sanskrit did not recover. Capacity was not the bottleneck.
- **Reweighting the data toward Devanagari** (Sanskrit ×5, Marathi ×2, lifting
  Sanskrit from 1.8% to 7.8% of the mix) is the recipe shipped here. It nudged
  Marathi (0.263 → 0.274) and Sanskrit (0.163 → 0.174) back up, gave the best
  overall recall@1 of any variant, and cost nothing cross-script — but it did
  **not** reach e5 parity on mr/sa. The residual gap is structural.

A caution on magnitude: Sanskrit has only 98 eval queries and Marathi 186, so
per-language recall@1 there swings on one or two documents. The honest claim is
that reweighting moved mr/sa the right way at no cost — not that it fixed them.

## How to use it, given all that

The adapter's entire value is *cross-script* retrieval, which base e5 was bad
at. Base e5 is already good at *same-script* Devanagari. So the mr/sa gap costs
nothing in practice: if you ever want same-script Devanagari retrieval, base e5
was your tool anyway. If mr/sa parity is ever a hard requirement, route by
script at query time (Devanagari → base e5, else → adapter) — a serving rule,
not another training run. Per-script adapters would also work, at the price of
serving two models.

## Reproducing it

Every step is committed and rebuildable — which is why `models/indic-aligned-v1`
stays gitignored (a reproducible successor, per [`models/README.md`](../../models/README.md)):

```bash
# 1. mine + combine -> data/pairs/indic-aligned-train.jsonl.gz   (654k pairs)
# 2. reweight toward Devanagari:
python scratch_reweight_devanagari.py         # -> indic-aligned-train-devwt.jsonl.gz
# 3. adapt on the GPU box (~13 min):
qfme adapt --checkpoint intfloat/multilingual-e5-small \
  --pairs data/pairs/indic-aligned-train-devwt.jsonl.gz --adaptation in-distribution \
  --train-pairs 250000 --eval-pairs 2000 --sample-pairs 252000 \
  --rank 32 --targets query,value --max-length 256 --pooling mean \
  --save-adapter models/indic-aligned-v1 --data-provenance public --profile configs/gpu.yaml
# 4. score, leak-free, on the Mac:
python scratch_score_multilingual.py models/indic-aligned-v1
```

Raw numbers: [`indic-multilingual-score.json`](indic-multilingual-score.json)
(per-language, all variants) and the per-run adapter reports
`indic-aligned-multi{,-r64,-devwt}-adapt.json` in this directory.
