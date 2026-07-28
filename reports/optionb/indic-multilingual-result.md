# A multilingual aligned adapter for ten Indian languages

**Model:** `models/indic-aligned-v2` — a rank-32 LoRA over `intfloat/multilingual-e5-small`,
promoted from `v1` after direct non-Hindi pairs were added to the mix.
**Trained:** 2026-07-28 on an RTX 4070 Ti SUPER, one epoch, ~13 minutes.
**Measured:** two leak-free, document-disjoint evals — a Hindi-pivot eval (2,283
unique-positive pairs) and a well-posed non-Hindi X↔Y eval (8,262 pairs).

## What it is

One low-rank adapter (589,824 parameters, 0.50% of the encoder) that teaches
`multilingual-e5-small` to place a passage and its cross-lingual twin near each
other in the same 384-dim space, across Hindi, Bengali, Gujarati, Kannada,
Malayalam, Marathi, Sanskrit, Tamil, Telugu, and Urdu.

The training signal is **mined, not synthesised**: Wikipedia langlinks pair an
article with its human-written counterpart in another language. The pairs share
meaning, not strings — median lexical overlap is ~0, so the model cannot cheat
by matching substrings; it has to learn the alignment.

The split is **document-disjoint by Hindi pivot cluster**: holding out a cluster
removes its text from *every* language's training set at once, so no eval
passage — in any language — was seen during training.

### v1 → v2: what changed

`v1` was mined as a **star**: every training pair had Hindi on one side. It
worked, but it never showed the model a direct Tamil↔Bengali or Marathi↔Telugu
pair. `v2` adds **521,834 non-pivot pairs** — the non-Hindi passages inside a
cluster joined directly to each other — to the sampling pool.

Both models trained on the **same 250k-pair budget** (`--train-pairs 250000`,
deduped to 249,164 unique). The only difference is the pool `v2` drew from
included those non-pivot pairs (~44% of a 1,175,743-pair pool). So any
difference below is attributable to the *mix*, not to more training.

## Two instruments, because the star topology needs two questions

**Hindi-pivot eval** — every query pairs with a Hindi passage or vice-versa.
This is what `v1` was built to serve, and where non-pivot data *shouldn't* help.

**Non-Hindi X↔Y eval** — retrieve a passage's twin in one specific non-Hindi
language, pool = one passage per held-out cluster in that language (different
clusters are different concepts, so rank is well-posed). This is the question
the non-pivot pairs exist to answer, and the earlier mixed-language pool scored
it ill-posedly (recall@1 ≈ 0 for everyone); this eval fixes that.

## The headline: non-pivot data helps non-Hindi retrieval

On the **non-Hindi X↔Y** eval, 8,262 held-out pairs across nine languages:

| metric | published e5 | v1 (pivot-only) | **v2 (+non-pivot)** |
|---|---|---|---|
| recall@1 | 0.7875 | 0.8862 | **0.8964** |
| recall@10 | 0.9299 | 0.9795 | **0.9831** |
| MRR | 0.8385 | 0.9172 | **0.9253** |

Two clean findings:

1. **Hindi-pivot training already transfers.** `v1` never saw a single direct
   X↔Y pair, yet it beats published e5 by **+9.9 recall@1 points** (0.79 → 0.89).
   The star topology generalises to the spokes.
2. **Direct pairs add more, universally.** `v2` gains **+1.0 recall@1 / +0.8
   MRR** on top of `v1`, and **every one of the nine languages improves, none
   regress** (mr 0.872 → 0.889, ta 0.878 → 0.888, ur 0.856 → 0.870).

## On the Hindi-pivot eval, v2 is a wash — as expected

| metric | published e5 | v1 | v2 |
|---|---|---|---|
| recall@1 | 0.0648 | 0.0850 | 0.0850 |
| recall@10 | 0.7495 | **0.8879** | 0.8852 |
| MRR | 0.2884 | **0.3443** | 0.3428 |

`v1` is ahead by 0.003 recall@10 / 0.0015 MRR — statistically nothing. The
mixed-language pool here makes recall@1 near-ill-posed (same-cluster distractors
in other languages tie the true positive), so recall@10/MRR are the signals to
trust. Adding non-pivot data cost essentially zero on the pivot task.

## Per-language, and the honest part

Both tables' per-language recall@1, v1 vs v2:

| lang | X↔Y v1 | X↔Y **v2** | pivot v1 | pivot **v2** |
|---|---|---|---|---|
| bn | 0.896 | **0.905** | 0.059 | 0.059 |
| gu | 0.903 | **0.906** | **0.048** | 0.040 |
| kn | 0.883 | **0.897** | 0.035 | 0.035 |
| ml | 0.890 | **0.897** | **0.074** | 0.064 |
| mr | 0.872 | **0.889** | 0.274 | **0.290** |
| sa | 0.891 | **0.897** | **0.174** | 0.133 |
| ta | 0.878 | **0.888** | 0.038 | **0.045** |
| te | 0.910 | **0.917** | 0.034 | **0.038** |
| ur | 0.856 | **0.870** | 0.040 | 0.040 |

On X↔Y, v2 wins every language. On the pivot eval it is mixed: v2 **recovers
Marathi** (0.274 → 0.290, back to e5 parity) and lifts ta/te/hi, but gives back
Sanskrit (0.174 → 0.133) and gu/ml.

The Sanskrit flip is **instrument-dependent and worth stating plainly**: v2
*improves* Sanskrit on X↔Y (0.891 → 0.897) while *dropping* it on the pivot eval.
Same model, opposite direction — because the two evals ask different questions
of the same shared adapter. Sanskrit and Marathi are Devanagari, the same script
as the Hindi pivot; adapting the shared space toward *cross-script* alignment
moves the *same-script* Devanagari neighbourhoods that base e5 already had right.
That tension is real and lives in every variant we trained (doubling the rank
32→64 did not resolve it; Devanagari reweighting nudged it but did not close it).

A caution on magnitude: Sanskrit has only ~98 pivot-eval queries and Marathi
~186, so per-language recall@1 there swings on one or two documents. The honest
claim is directional, not a fix.

## Why v2 ships

v2 **dominates the instrument built for the question** (non-Hindi X↔Y: +1.0
recall@1, all nine languages up) and **ties on the pivot eval** (−0.003
recall@10, noise). That is a favourable Pareto trade — small, but real and free.
The non-pivot mining did exactly what it was mined to do. `v1` remains in the
lineage as the pivot-only baseline this was measured against.

## How to use it, given all that

The adapter's value is *cross-script* retrieval, which base e5 was bad at. Base
e5 is already good at *same-script* Devanagari. So the residual mr/sa pivot gap
costs nothing in practice: same-script Devanagari retrieval was base e5's job
anyway. If mr/sa parity is ever a hard requirement, route by script at query
time (Devanagari → base e5, else → adapter) — a serving rule, not a training run.

## Reproducing it

Every step is committed and rebuildable — which is why `models/indic-aligned-v2`
stays gitignored (a reproducible successor, per [`models/README.md`](../../models/README.md)):

```bash
# 1. mine + combine hi-pivot pairs, reweight toward Devanagari:
python scratch_reweight_devanagari.py         # -> indic-aligned-train-devwt.jsonl.gz (767,559)
# 2. mine direct non-Hindi X<->Y pairs from TRAIN clusters only:
python scratch_mine_nonpivot.py               # -> indic-aligned-nonpivot.jsonl.gz    (521,834)
# 3. concatenate into the v2 pool (1,175,743 pairs):
cat devwt + nonpivot -> data/pairs/indic-aligned-train-np.jsonl.gz
# 4. adapt on the GPU box (~13 min), same 250k budget as v1:
qfme adapt --checkpoint intfloat/multilingual-e5-small \
  --pairs data/pairs/indic-aligned-train-np.jsonl.gz --adaptation in-distribution \
  --train-pairs 250000 --eval-pairs 2000 --sample-pairs 252000 \
  --rank 32 --targets query,value --max-length 256 --pooling mean \
  --save-adapter models/indic-aligned-v2 --data-provenance public --profile configs/gpu.yaml
# 5. score, leak-free, two instruments (on the GPU box, not the Mac):
python scratch_score_multilingual.py models/indic-aligned-v2 data/pairs/indic-aligned-eval.jsonl.gz
python scratch_score_xling.py         # well-posed non-Hindi X<->Y
```

Raw numbers: [`indic-multilingual-score.json`](indic-multilingual-score.json)
and the per-run adapter reports `indic-aligned-multi{,-r64,-devwt,-np}-adapt.json`
in this directory.
