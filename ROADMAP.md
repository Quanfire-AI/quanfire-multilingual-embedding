# Roadmap

> An embedding model factory: corpus in, trained and evaluated model out — generic or
> domain-specific.

**Status:** Phases 0, A and B complete; Phase C mining in place and measured. A published
checkpoint adapted on real Indic text beats itself by **28.6% (Hindi)** and **40.9% (Tamil)**
on held-out retrieval, training 0.50% of its parameters. A single ten-language adapter
trained on cross-lingual aligned pairs raises cross-lingual retrieval from **0.7875 → 0.8964
recall@1** in-domain, and a production article+sentence blend (`prod-a70s30`) beats it on the
public benchmark too — **0.9805** FLORES non-Hindi (v2 0.9609, near base E5), **0.9029**
in-domain, and **0.8914** hi-pivot recall@10, exceeding v2 on all three published instruments.
Phases D–E planned.

---

## Objective

This project must **completely support building QuanFire's own embedding models**. Not a
single model: a pipeline that takes a corpus and produces a trained, evaluated, servable
encoder, either general-purpose or adapted to a specific domain.

Concretely, it must be able to:

1. Prepare a corpus from raw text, in any supported script.
2. Train a tokenizer and vocabulary over it, or reuse an existing one.
3. Produce an encoder by any of three routes — from scratch, adapted from a pretrained
   checkpoint, or a fast static baseline.
4. Mine training pairs from unlabelled domain text, since labelled pairs will not exist.
5. Evaluate the result per language and per domain, against a named baseline.
6. Persist a versioned, reproducible artefact.
7. Serve it behind an API.

The same pipeline, given a different corpus and configuration, yields a different model.
That is the product: **the factory, not any one model it makes.**

## Where the value actually sits

A general-purpose model competes with well-funded open models given away free. A model
tuned to *QuanFire's own document domains* does not, because nobody else has that corpus.

DocPro, BillAI and MindMap each handle a distinct kind of text — contracts and filings,
invoices and time entries, notes and relationships. A general model treats all three the
same. A model adapted per domain does not, and the corpus that makes it better is
proprietary by construction.

**Domain adaptation is therefore the primary capability**, and generic training is the
fallback rather than the goal.

## Honest constraints, carried forward

These are recorded so the plan stays grounded, not to reopen the decision.

- **The hardware caps model size at roughly 568M parameters.** The largest open models
  are 8B and cannot be trained on a 16 GB card. Out-scaling is unavailable; out-
  specialising is the whole strategy.
- **Data is the binding constraint, not compute.** Contrastive training needs pairs.
  Phase C exists because those pairs must be manufactured from unlabelled text.
- **A from-scratch model will be worse than a fine-tuned open checkpoint** for a long
  time, on any budget available here. Phase E exists for capability and independence, not
  because it produces the best model.

---

## Phases

### Phase 0 — The encoder contract ✅ **done**

`embedding/encoder.py` defines `TextEncoder`: text in, vectors out. The search pipeline
depends on that rather than on an embedding matrix, so a contextual model can be served
without rewriting anything downstream.

Verified by a `HashingEncoder` backed by no model, no vocabulary and no matrix, indexing
and searching end to end.

### Phase A — A contextual encoder we own ✅ **done**

A transformer encoder, trainable, exposed through `TextEncoder`.

This came first because everything downstream — evaluation, pair mining, serving — needs
a real encoder to work against.

The architecture is written out rather than borrowed. Loading a pretrained checkpoint
first would have meant a good checkpoint masking a broken training loop; a model defined
here cannot hide that. Adapting external checkpoints is a smaller, later step that now
has a verified loop to land on.

- **Delivered:** `embedding/neural/` — a pre-norm transformer encoder with fused
  attention and mean pooling; `NeuralTextEncoder` satisfying the `TextEncoder` contract;
  an InfoNCE contrastive trainer with warmup, decay, gradient clipping and decay-group
  splitting; save and load; 25 tests.
- **Exit criterion — met.** On a two-topic synthetic corpus, a 28k-parameter model over
  38 steps moved the separation between within-topic and cross-topic similarity from
  **0.175 to 1.326**, with cross-topic similarity going negative. It serves through the
  existing search pipeline unchanged.
- **Introduced:** PyTorch, as the optional `neural` extra. Verified that the base
  install still works without it.

### Phase B — Training that fits the hardware 🔧 **partly done**

The two techniques that make a 568M model trainable on a 16 GB card.

**Delivered.** `lora.py` — adapters over frozen weights, with merging, adapter-only
checkpoints, and a refusal when the target names match nothing. At BERT-base shape the
trainable share **at rank 16** is **0.81%**, the adapter checkpoint is **3.4 MB against a
419 MB model**, and Adam's optimizer state falls from **0.82 GB to 6.8 MB**. The rank is
load-bearing and was missing from this claim until it was measured again: at rank 8 the
same model gives 0.40% and a 1.7 MB adapter.

`gradcache.py` — chunked encoding with a cached vector gradient, so the contrastive batch
is bounded by disk rather than VRAM. Verified gradient-for-gradient identical to a single
large backward pass, and wired into `ContrastiveTrainer` through
`compute.gradient_checkpoint_chunk` rather than left as a library nobody calls.

One correction to an earlier claim here, because it was wrong in a way worth recording.
"Invariant to chunk size" holds **only at `dropout=0`**. Chunked encoding draws different
dropout masks than unchunked — eight rows in one call is not eight calls of one row — so
chunk sizes cannot agree with each other once dropout is on, however correct the
implementation. What must hold, and now does, is that the cached path matches the uncached
path *at the same chunk size*. Getting there meant fixing a real defect: the two encoding
passes were not sharing a random state, so the cached gradient was being applied to
activations it was never computed for, diverging from the truth by 11.3 absolute. It went
unnoticed because every test used `dropout=0.0`, which is exactly the setting that hides
it.

Mixed precision — `fp32` or `bf16`, the latter chosen over `fp16` because it shares fp32's
exponent range and so needs no loss scaling. Honoured by the trainer through autocast on
the forward pass only.

Compute profiles — a `compute` config section and `--profile`, so one branch and one
experiment file run on both a development machine and a GPU box. Devices validate by shape
rather than availability, which is what lets a GPU profile be authored and CI-tested
without a GPU.

**Still to do.** Adapting an external pretrained checkpoint, Matryoshka truncation,
checkpoint resumption. Hard-negative mining landed as Phase C work; see below.

**Verified on hardware, 20 July 2026.** An RTX 4070 Ti SUPER, batch 256, a 5.3M-parameter
encoder over 4,000 mined Hindi pairs:

| | no caching | chunk 32 |
|---|---:|---:|
| fp32 | 4.89 GB / 4.3s | 0.40 GB / 4.7s |
| bf16 | 2.99 GB / 2.7s | 0.29 GB / 4.7s |

Gradient caching carries the memory saving — **12.2x alone**, against 1.6x for bf16, 16.9x
together. Final losses spanned **0.51%** across all four cells, so the exactness claim
holds off the test bench. bf16 also turned out to be **1.6x faster**, which was not why it
was chosen.

Initial loss matched `ln(batch_size)` to within 4-6% at both batch 16 and 256 — what an
untrained contrastive model must show, and independent evidence the objective is right.

**Still unverified:** everything at realistic model size. 5.3M parameters is a toy, and
0.29 GB of a 16 GB card says nothing about where the ceiling sits for a 100M+ encoder.

- **Exit criterion — met, 21 July 2026.** `intfloat/multilingual-e5-small` adapted with
  LoRA on 20,000 mined Wikipedia pairs per language, scored against ~2,000 held-out pairs
  it never saw. Rank 32, two epochs, **0.50% of parameters trained**, on a 4070 Ti SUPER.

  | | Hindi base | Hindi adapted | Tamil base | Tamil adapted |
  |---|---:|---:|---:|---:|
  | recall@1 | 0.4238 | **0.5451** (+28.6%) | 0.3219 | **0.4535** (+40.9%) |
  | recall@10 | 0.6690 | 0.7929 (+18.5%) | 0.5269 | 0.6966 (+32.2%) |
  | MRR | 0.5136 | 0.6364 (+23.9%) | 0.3931 | 0.5397 (+37.3%) |

  **The control replicates across language families.** Gains run inversely to lexical
  overlap in both, and Tamil is Dravidian while Hindi is Indo-Aryan:

  | overlap band | Hindi | Tamil |
  |---|---:|---:|
  | low `<0.3` | +145.5% | +126.7% |
  | mid `0.3–0.7` | +39.6% | +56.9% |
  | high `>0.7` | *not significant* | +21.6% |

  Hindi's high-overlap band moved 0.5828 to 0.6290, which reads as +7.9% and is 29 extra
  correct answers out of 628 with overlapping confidence intervals. It was reported as a
  gain here before the counts were checked; it is not one. The correction sharpens the
  finding rather than weakening it — for Hindi the gain is confined to the bands where
  string matching does not help.

  A model memorising surface form improves most where strings already match. Neither does.
  One language could have been an accident; two unrelated ones make it a property of the
  method. The lexical-leakage concern that shaped pair mining is now a control the
  adaptation passes rather than a caveat on it.

  **The weaker language gained more, which is the argument for doing this at all.** E5
  serves Tamil worse than Hindi — baseline 0.3219 against 0.4238, or 76% as well. After
  adaptation Tamil reaches 83% of Hindi's score. The corpus helps most exactly where the
  published model is thinnest, and that is where a proprietary corpus earns its keep.

  More capacity has not cost anything yet: rank 16 gave Hindi +20.3%, rank 32 gave +28.6%.
  The ceiling is above rank 32.

  Stated with its limits. Two languages, both Wikipedia. A ~2,000-candidate pool rather
  than a production index. Held-out pairs from the same distribution as training, so this
  measures in-domain adaptation and not generalisation to another task. And the absolute
  numbers on the hardest slice remain low — Tamil low-overlap retrieval more than doubled
  and is still only 0.1868.

#### One adapter or one per language — answered

Three runs, identical held-out set, identical 40,000 training pairs, only the *source*
differing. Baselines matched exactly across all three, which is what makes the comparison
readable.

| trained on | overall | Hindi | Tamil |
|---|---:|---:|---:|
| *(published checkpoint)* | 1018/1985 | 480/779 | 538/1206 |
| Hindi + Tamil | **1290** | **573** | **717** |
| Hindi only | 1239 | 558 | 681 |
| Tamil only | 1267 | 561 | 706 |

**Build one adapter.** Joint training is numerically best on both languages and never
worse than either specialist. The differences are inside the confidence intervals, so it
is not *provably* better — but "at least as good, and one artefact instead of two" settles
the architecture on its own.

**The finding underneath it is larger.** Training on Tamil alone captures **87%** of the
joint gain on Hindi, and training on Hindi alone captures **80%** of the joint gain on
Tamil — on languages those runs never saw.

Most of what the adaptation learns is therefore not language-specific. It is being carried
across an Indo-Aryan and a Dravidian language, which leaves the task, the pair structure
and the encyclopedic register as the likely content. That is worth knowing before planning
corpora for the other twenty scheduled languages: the first language may buy most of the
benefit, and each additional one may add less than its collection cost implies.

It is also a caution on the framing. "Domain adaptation" has been the story throughout;
this suggests a substantial part is *task* adaptation, and the two would need separating
before the distinction is claimed in either direction.

#### Task or language — separated, 22 July 2026

The caution above was right. Four runs settle it, each varying exactly one facet with
everything else held fixed, matched training volumes, and an evaluation set pinned by
`--eval-pairs-file` so both arms of each pair scored an identical baseline.

| varied | held fixed | achievable gain captured |
|---|---|---:|
| **task shape** — `adjacent` → `heading_section` | language, corpus | **−17%** |
| **language** — Hindi → Tamil | task shape, corpus | **+95%** |

*Task axis*, scored on Tamil and Hindi `heading_section`, 1,927 queries, baseline 489:

| trained on | recall@1 | |
|---|---:|---|
| `heading_section` (20k) | 684/1927 | +39.9%, intervals disjoint |
| `adjacent` (20k) | 455/1927 | −7.0%, no gain |

*Language axis*, scored on Tamil `heading_section`, 1,272 queries, baseline 253:

| trained on | recall@1 | |
|---|---:|---|
| Tamil (15k) | 388/1272 | +53.4% |
| Hindi (15k) | 381/1272 | +50.6% |

Seven queries apart out of 1,272, with intervals almost entirely overlapping. The mid
overlap band is identical at 35/290.

**The adaptation is language-general and task-specific**, which is the reverse of the
intuitive assumption and the reverse of how this work was framed for months. Two
consequences:

- **Pairs transfer across languages.** Mine wherever the text is cleanest and most
  abundant; the other languages get almost all of the benefit. The twenty scheduled
  languages do not each need a corpus.
- **Pairs do not transfer across query shapes.** Every shape to be served must be present
  in the training mixture. This is where the collection cost actually sits.

The second point is narrower than the −17% suggests. Training on all three kinds at once
(`indic-v1`, 40,000 pairs) reached 178/458 on `heading_section` against a 129/458 baseline
— +38.0%, recovering essentially the whole dedicated-training gain, while also delivering
+40.8% on `adjacent` from the same adapter. **A mixture containing the shape works; a
single different shape does not.** So the requirement is to mine several shapes, not to
predict which one users will type.

One qualification worth carrying. The languages tie at recall@1 but not deeper: in-language
training puts the answer in the top ten 797 times against 733, and leads on nDCG@10 (0.4473
against 0.4222). Where a reranker consumes the top ten rather than the top hit, in-language
data still buys something real.

This supersedes the 80%/87% figures above, which compared specialists against joint
training and so varied training composition as well as language. They pointed the same way;
these are the controlled version.

Still Wikipedia on both sides of every comparison, so the *corpus* axis remains untested —
`--adaptation domain` exists for it and needs a non-Wikipedia pair file to run.

### Phase C — Pair mining from unlabelled text

**The phase that makes domain-specific models possible**, and the one most likely to be
underestimated.

Labelled query-passage pairs will not exist for QuanFire's domains. They must be
manufactured from document structure and content:

| Source | Pair |
|---|---|
| Document structure | title ↔ body, heading ↔ section, summary ↔ document |
| Adjacency | consecutive paragraphs, co-occurring sections |
| Metadata | invoice line ↔ description, matter ↔ time entry narrative |
| Synthetic | generated questions answered by a passage |
| Cross-lingual | translation pairs, where parallel text exists |

- **Deliverables:** miners for each strategy, including the cross-lingual row via
  `qfme mine-aligned` (**done**); hard-negative mining against a base encoder (**done**;
  verdict below — a wash in-domain, a regularizer out-of-domain); pair quality filtering
  and deduplication; a `qfme mine` command (**done**, as `mine-pairs` and `mine-negatives`).
- **Exit criterion:** a model trained purely on mined pairs beats the untrained base on
  its own distribution. **Met** for cross-lingual retrieval on mined aligned pairs
  (X↔Y recall@1 0.7875 → 0.8964, ten languages); the public-bitext transfer limit is now
  **met too** by an article+sentence blend (FLORES-200 non-Hindi 0.9826, level with base;
  the mix run below).

#### The miner is not a Wikipedia miner — 22 July 2026

Every mining figure published so far came from a MediaWiki dump, which left the phase
resting on an unchecked assumption. `data/sample/domain-corpus.jsonl` — ten synthetic
professional-services documents, English and Hindi — mines **56 pairs across all three
kinds** with no code change, and `tests/corpus/test_domain_pairs.py` (16 tests) pins the
record contract an exporter has to satisfy.

The trap it fixes has no error message: `JsonlReader` flattens unrecognised *top-level*
fields into the attributes mapping, so `sections` nested under an `attributes` key is
silently invisible and mining produces zero `heading_section` pairs while reporting
success. That is now a test rather than a paragraph.

So a domain corpus is a **format conversion**, not a missing capability. What remains
genuinely unknown is whether real client documents behave like the fixture — its
`title_lead` overlap of 0.234 against Hindi Wikipedia's 0.977 is a hypothesis written by
someone who knew overlap would be measured, and settling it needs a real export.

#### Cross-lingual, one layer down — 22 July 2026

The `Cross-lingual` row above needs parallel text nothing mines, and the exit criterion
for it stays open. But the *prerequisite* is measurable without aligned pairs:
`RetrievalReport.language_separation` reports whether the near misses of each query are
dominated by the query's own language, against the baseline a language-blind ranker would
produce from the pool's own composition. It declines to answer on a monolingual pair set
rather than returning a meaningless 1.0.

It is a diagnostic, not a score. Every pair mined here has both sides in one language, so
a separated space is the expected outcome; a value near 1.0 clears the precondition for
cross-lingual retrieval and demonstrates nothing beyond it.

#### The aligned miner closes the row, and the adapter fails the test it opened — 27 July 2026

`qfme mine-aligned` builds the parallel text the `Cross-lingual` row needed: it joins two
language corpora through a Wikipedia `langlinks` dump, so an anchor in one language is
paired with the human-written article on the same subject in another. The join is the whole
risk — a mis-parsed SQL tuple or a title that normalises differently on the two sides
silently shrinks the set with nothing to point at — so the langlinks parser and the
normalise-both-sides lookup are tested against the punctuation and escapes that break the
naïve approach, and the statistics count *why* each source document failed to align.

Run over the Hindi and Tamil dumps it aligned **32,166 of 118,571 Hindi articles (27.1%)**
into **121,383 cross-script pairs**, mean token overlap **0.014** — genuinely zero shared
units, which is the point. This is the first parallel set the project has produced, and the
first honest cross-lingual retrieval number it can report.

That number is a **caution, not a win.** Scored on a seeded 1,980-pair hi↔ta pool (both
directions, identical pool for all three models, unit vectors, degenerate encodings
dropped):

| model | recall@1 | recall@10 | MRR |
| --- | --- | --- | --- |
| published `intfloat/multilingual-e5-small` | **0.5187** | 0.800 | 0.621 |
| `indic-b-baseline` (LoRA, in-batch negatives) | 0.4141 | 0.722 | 0.520 |
| `indic-b-hardneg` (LoRA + mined negatives) | 0.3768 | 0.697 | 0.492 |

The adaptation that *raised* same-language Indic retrieval by ~30% (0.4952 → 0.6439)
**lowers** cross-lingual retrieval by ~10 points against the base it started from, and the
hard negatives lower it again. Two shifts confound the result and both point the same way:
the eval is cross-**script** (the adapter trained only within a language) and out-of-**domain**
(the adapter trained on MILPaC legal/QA, scored here on Wikipedia leads). The adapter also
emits a zero vector on 20 of 2,000 cross-lingual inputs — a 1% collapse the base never has.

The lesson is owned rather than buried: a within-language LoRA is not a free upgrade for a
multilingual model. It sharpens the language it was trained on and dulls the cross-lingual
matching that is the model's reason to exist. An adapter meant to serve cross-lingual
traffic has to be *trained* on aligned pairs — which now exist — not adapted on monolingual
ones and hoped to transfer. `reports/hi-ta-aligned.json` holds the join statistics;
`reports/optionb/aligned-transfer.json` holds the scores.

#### Trained on aligned pairs, ten languages, and the row closes in-domain — 28 July 2026

The previous entry ends on an instruction: an adapter meant to serve cross-lingual traffic
has to be *trained* on aligned pairs, not adapted on monolingual ones and hoped to transfer.
That adapter now exists. `qfme mine-aligned` was run across **ten** language editions
(hi, ta, bn, gu, kn, ml, mr, sa, te, ur) against Hindi through langlinks, and a single LoRA
adapter (`models/indic-aligned-v1`, refined to `-v2` — v2 canonical) was trained on the
resulting cross-lingual pairs rather than on within-language ones.

It reverses the caution. Two leak-free instruments, eval pairs held out, pool built so
recall@1 is well-posed:

| instrument | E5 base | adapted v2 |
| --- | ---: | ---: |
| non-Hindi X↔Y, within target language (recall@1, n=8,262) | 0.7875 | **0.8964** |
| hi-pivot mixed-pool (recall@10) | 0.7495 | **0.8852** |

The within-language LoRA *lowered* cross-lingual retrieval by ~10 points (previous entry);
the aligned-trained adapter *raises* it by ~11. Same architecture, same base, same rank —
the only change is that the training pairs are the ones the eval measures. That is the whole
lesson of the caution paid back as a result.

**The honest limit was a scale mismatch, and it is now closed — 29 July 2026.** On
**FLORES-200**, a public sentence-aligned benchmark neither model trained on, base E5 scored
**0.985** non-Hindi recall@1 and v2 moved to **0.961** — a small regression recorded, not
beaten, and for months not claimed beyond the Wikipedia distribution. The diagnosis turned
out to be simple: v2 is trained on *article-scale* aligned pairs and FLORES is a *sentence*
benchmark. It was a domain gap, not a ceiling.

The fix used the `qfme ingest-parallel` path built for exactly this: turn a held-out
sentence-aligned corpus into pairs and train on it, FLORES held strictly out. A proof slice
of **AI4Bharat Samanantar** (30k en↔indic sentence pairs × 8 languages = 240k; sa and ur are
absent upstream, so the sentence side omits them) was ingested and trained with v2's exact
hyperparameters — the only variable changed was the corpus. Three runs, one held-out
instrument (FLORES non-Hindi X↔Y r@1) and one in-domain instrument (article-scale non-Hindi
X↔Y r@1):

| Adapter | Trained on | FLORES non-Hi | in-domain non-Hi |
|---|---|---|---|
| base E5 | — | 0.9847 | 0.7875 |
| `indic-aligned-v2` | article pairs | 0.9609 | 0.8964 |
| `samanantar-proof` | sentence pairs only | **0.9896** | 0.8195 |
| `indic-aligned-mix` | ~50/50 article + sentence | 0.9826 | **0.8918** |

Sentence-only training (`samanantar-proof`) **beat base E5 on FLORES** (0.9896 vs 0.9847) and
generalised to sa/ur, which it never trained on — but mirrored v2's tradeoff, giving back the
article-scale in-domain gain (0.8964 → 0.8195). A **50/50 blend** (`indic-aligned-mix`, 250k
article + 240k sentence, shuffled) collapses the tradeoff: FLORES **0.9826** (level with base,
gap −0.002; +0.022 over v2) *and* in-domain **0.8918** (within 0.5% of v2, +10.4 points over
base). One adapter, near-top on both. The exit criterion the public-bitext row held open is
met. Recipe: `configs/experiments/samanantar-proof.yaml` and `mix.yaml`,
`scratch_samanantar_prep.py` (download + ingest), `scratch_samanantar_flores.py` (scorer);
`reports/samanantar-proof-verdict.json` holds the consolidated scores.

The scope is stated, not overclaimed: this is a proof-scale slice (240k sentence pairs, 8
languages, one epoch), the sentence side lacks sa/ur, and the blend ratio is untuned. A
production run would use more of Samanantar (and a sa/ur sentence source such as BPCC/OPUS)
and sweep the ratio. `indic-aligned-mix` is the promotion candidate over v2, pending that
scale-up.

**The production run closed that scale-up — and it beats v2 outright — 30 July 2026.** The
proof's three caveats are each answered. A ten-language sentence pool (1.42M pairs) was built:
Samanantar scaled to 150k pairs × 8 languages, plus the two languages Samanantar lacks —
**sa** from `rahular/itihasa` (75k classical Sanskrit↔English) and **ur** from OPUS-100 (148k).
Both sources are non-gated, so the recipe reproduces without an access grant (BPCC carries
sa/ur but is gated, and was deliberately avoided). That pool was blended against v2's article
corpus at a fixed 1.0M total, three ratios swept, every hyperparameter held at v2's values:

| Adapter | article : sentence | FLORES non-Hi | in-domain non-Hi |
|---|---|---|---|
| base E5 | — | 0.9847 | 0.7875 |
| `indic-aligned-v2` | article only | 0.9609 | 0.8964 |
| `prod-a30s70` | 30 : 70 | **0.9820** | 0.8937 |
| `prod-a50s50` | 50 : 50 | 0.9792 | 0.8991 |
| `prod-a70s30` | 70 : 30 | 0.9805 | **0.9029** |

`prod-a70s30` is the winner and the promotion choice: it **strictly beats v2 on both**
sweep **instruments** — FLORES **0.9805** (+0.0196 over v2, within 0.004 of base) *and* in-domain
**0.9029** (+0.0065 over v2, +0.115 over base). Not "holds v2 while closing FLORES" as the
proof mix did — it *exceeds* v2 in-domain and nearly reaches base on FLORES, all ten languages
at sentence scale. It replaces `indic-aligned-mix` as the promotion candidate over v2. Recipe:
`configs/experiments/prod-a{30s70,50s50,70s30}.yaml`, `scratch_production_prep.py` (pool),
`scratch_blend_ratio.py` (ratios), `scratch_prod_flores.py` (scorer);
`reports/prod-flores-verdict.json` holds the scores.

**Third instrument re-baselined — prod-a70s30 wins on all three — 30 July 2026.** The two
sweep instruments above are two of v2's three published numbers; the third, the **hi-pivot
mixed-pool recall@10**, was still measured on v2. Re-scored on `prod-a70s30` with the
byte-identical `scratch_hn_verdict.instrument_pivot` protocol — which reproduced the published
base (0.7495) and v2 (0.8852) exactly — it reads **0.8914 (+0.0062 over v2)**. So the
promotion candidate now **beats v2 on every published instrument at once**: X↔Y in-domain r@1
0.9029, hi-pivot r@10 0.8914, FLORES non-Hindi r@1 0.9805. Scorer `scratch_prod_pivot.py`;
`reports/prod-pivot-verdict.json` holds the scores.

One honest null result, recorded not buried: the dedicated **sa/ur sentence sources did not
lift sa/ur on FLORES.** The proof — which had *no* sa/ur training data — scores higher on both
(sa 0.9589, ur 0.9922) than any production ratio (sa ≈0.94, ur ≈0.98). Cross-lingual transfer
from the other eight languages already covered sa/ur, and classical-epic Sanskrit (itihasa) is
a domain mismatch for FLORES's modern prose. The sa/ur sources were the right thing to try;
the data says they were not the bottleneck. The re-baseline of the other published numbers on
`prod-a70s30` is now done (the hi-pivot entry just above), and the one remaining question —
whether more than one epoch improves it — has now been answered (next entry).

**The epoch sweep confirms one epoch is the stopping point — 30 July 2026.** The last open
question before stamping `prod-a70s30` canonical was whether a longer-than-one-epoch schedule
beats it. It does not. Two more runs, byte-identical to `prod-a70s30` except `epochs` (2 and 3;
`configs/experiments/prod-a70s30-e{2,3}.yaml`, single-variable), were scored against e5, v2 and
one-epoch `prod-a70s30` on all three published instruments:

| epochs | in-domain r@1 | hi-pivot r@10 | FLORES non-Hi (held out) |
|---|---|---|---|
| v2 | 0.8964 | 0.8852 | 0.9609 |
| **1 (`prod-a70s30`)** | 0.9029 | 0.8914 | **0.9805** |
| 2 | 0.9062 | 0.8971 | 0.9701 |
| 3 | 0.9084 | 0.8984 | 0.9673 |

All three epoch counts beat v2 on all three instruments, so the promotion holds either way. But
the sweep exposes a clean trade-off: more epochs lift the **in-domain** fit (in-domain and
hi-pivot both climb monotonically) while **regressing the held-out FLORES benchmark**
monotonically — 0.9805 → 0.9701 → 0.9673. Base e5 has the highest FLORES of all (0.9847);
adaptation always costs some, and one epoch preserves the most. Trading ~1.3 points of the
held-out public number for ~0.5 point of in-domain fit is a bad trade, so **one-epoch
`prod-a70s30` stays canonical** — the sweep validates the promotion rather than replacing it.
Scorer `scratch_prod_longer.py`; `reports/prod-longer-verdict.json` holds the scores; the e2/e3
adapters are mirrored under `models/prod-a70s30-e{2,3}/`.

The earlier hard-negative work is still valid on its own instrument:
`reports/optionb/hn-verdict.json` holds those consolidated four-model, three-instrument
scores.

#### Hard negatives are mined, and the rate they cost is not claimed — 23 July 2026

`qfme mine-negatives` ranks a pair set's own positives against each anchor with any
`TextEncoder` and keeps the hardest survivors; `TextPair.negatives` carries them into
training as extra candidate columns, so the similarity matrix becomes
`batch × (batch + extras)` and the targets stay the diagonal. A pair set without negatives
produces zero extra columns and trains byte-identically to before — the in-batch objective
is the zero case of the new one, not a separate path.

No second corpus is fetched. The candidate pool is the pair set's own positives, which are
already passages somebody wrote and a miner kept, and are already resident.

The reason this took a module rather than a function is the false negative. A mined
"negative" that is really a correct answer trains the model to push the right passage away,
with the largest gradient in the batch, and the loss curve *improves* while retrieval
degrades — a model taught to reject correct answers is being taught something and learns it.
Three guards stand against it: the pair's own positive by identity, any candidate from the
same source document by provenance, and anything above 0.95 similarity as a likely
paraphrase. Unencodable text is rejected by vector norm rather than by score, because the
similarity floor that happens to catch it is a difficulty setting and a run against a weak
checkpoint lowers it.

**What is not claimed.** The statistics report `outranking_the_positive` — accepted
negatives the model scored above the pair's own answer — and deliberately contain no field
named for a false-negative *rate*. That number requires labelling, so `--audit` writes the
hardest sample to JSONL with an `is_actually_correct: null` field and stops there. A test
asserts no key in the statistics contains the string `false_negative`, because a field
named for the rate would be read as the rate and would be wrong by an unknown factor in an
unknown direction.

**Measured — 27 July 2026, and it is a wash.** The comparison has now been run on the
4070 Ti. Two LoRA adapters, identical in every setting but one, trained on the same 40,000
hi/ta pairs with the same seed and scored on the same held-out 2,000: in-batch negatives
only (`indic-b-baseline`) reached recall@1 **0.6439** [CI 0.6226–0.6646], MRR 0.7101;
in-batch **plus 168,000 mined hard negatives** (`indic-b-hardneg`) reached **0.6494**
[CI 0.6282–0.6701], MRR 0.7082 — a +0.0055 shift on recall@1 (11 of 1,991 queries), a
*negative* shift on MRR, and confidence intervals that overlap almost entirely. Hard
negatives, on this pair distribution, moved nothing.

The mining report says why. **47.2%** of the kept negatives scored above their own
positive — the population false negatives are drawn from — and the audit sample confirms
the character on inspection: an anchor "political career" whose mined negative is a second,
equally-correct passage about the same person's political career. These are genuine false
negatives, and re-mining with a tighter similarity ceiling does not remove them: dropping
the ceiling from 0.95 to **0.80** rejected **12** additional negatives out of 168,000 and
moved the suspicion rate by 0.08 points, because the false negatives sit at similarity
~0.68, far below any sane ceiling. The ceiling is the wrong instrument. These positives are
short Wikipedia leads with a low self-similarity (~0.57), so any on-topic passage outranks
them; the fix is stronger positives, not filtered negatives, and that is a different pair
set, not a different flag.

So hard-negative mining is a working capability that this data does not reward. It is left
in place — the objective is the strict generalisation of in-batch and costs nothing when a
pair set carries no negatives — and it is not claimed as an improvement. Scores in
`reports/optionb/`.

**Re-measured at scale, 250k aligned pairs — 28 July 2026, and it is a trade-off.** The
ablation was rerun on the ten-language aligned data: `models/indic-aligned-hn` mines four
negatives per pair over 250,000 pairs and trains identically to the canonical `-v2`.
Against a control adapter trained on the same pairs with no negatives, the mined negatives
cost **~−1.5 recall@1 in-domain** (X↔Y 0.8955 control → 0.8801 hn) and **add ~+1.5** on the
out-of-domain FLORES-200 benchmark (0.9606 control → 0.9757 hn). The negatives act as a
regularizer: they blunt the in-distribution fit slightly and generalise slightly better off
it. v2 stays canonical; the HN adapter is not promoted.

The mining report explains why the in-domain cost is unavoidable here: **49.8%** of the kept
negatives outrank their own positive (`suspicion_rate` 0.4981) — half the negatives are
false. The absolute-ceiling guards can't catch them because they sit near the positive, not
near a paraphrase ceiling, so a new **relative** guard was implemented: `--positive-margin`
rejects any candidate scoring within a margin of its own positive, targeting the
false-negative population directly.

**Gated re-mine and retrain — 29 July 2026, and the trade-off is gone.** Re-mining the same
250k pairs with `--positive-margin` took `suspicion_rate` **0.4981 → 0.0000** (≈1.9M suspect
candidates dropped) while keeping 84% of the negative volume. The adapter trained on those
cleaned negatives, `models/indic-aligned-hn-gated`, was scored on the same three instruments:

| instrument | control | hn (plain) | **hn-gated** |
|---|---:|---:|---:|
| X↔Y in-domain recall@1 | 0.8955 | 0.8801 | **0.8940** |
| FLORES-200 non-Hindi recall@1 | 0.9606 | 0.9757 | **0.9768** |
| hi-pivot mixed-pool recall@10 | 0.8839 | 0.8695 | **0.8870** |

Gating **recovers the in-domain loss** (back to 0.8940, level with the 0.8955 control, where
plain negatives sat at 0.8801) **and keeps the FLORES gain** (0.9768, above both control and
plain hn). So the in-domain damage was the false negatives, not hard negatives as such:
remove them with the relative guard and the regulariser benefit survives at no in-domain
cost. Gated hard negatives are therefore a genuine improvement, not a wash — a candidate to
reconsider for promotion over `-v2`, held back only because the margin over it is a fraction
of a point in-domain against a larger out-of-domain lift, a call worth making deliberately.
Consolidated scores in `reports/optionb/hn-verdict.json`.

### Phase D — Serving

A web service over the artefact-loading pattern already in place, using the de facto
industry-standard request and response schema so existing clients migrate by changing a
base URL.

- **Deliverables:** embeddings endpoint; batching; model versioning; dimension
  truncation; ONNX export and quantisation; container image; auth and rate limiting.
- **Exit criterion:** p95 under 100 ms for a short input, and a client switches by
  changing only the base URL — amended below, because the second half of that turned out
  to be the wrong thing to want.

#### The experiment became a command — 22 July 2026

Every result above was produced by `scripts/adapt_pretrained.py` with flags set by hand.
That is reproducible only if the shell history survives, and a run described by twenty flags
cannot be committed, diffed or reviewed.

The experiment now lives in `pipelines/adaptation.py` as `AdaptationPipeline`, with
`AdaptationConfig` as its schema, and runs as `qfme adapt --config … --profile …`. The
script is a thin front end over the same pipeline and keeps every flag it had, so the
command lines that produced the figures above still work verbatim.

Two things fell out of the move rather than being the point of it.

`ContrastiveTrainer` was annotated as taking a `NeuralTextEncoder` and had been handed a
`PretrainedTextEncoder` for months. The script escaped it because scripts are not
type-checked; the pipeline did not. The fix is a structural `Trainable` protocol naming the
three things the trainer actually touches — `device`, `train_mode()`, `_prepare()` — which
is the same reasoning that keeps the encoder families apart rather than under a base class.

The sampler was reading the leading `count * 4` lines and shuffling those. On a Hindi and
Tamil pair file concatenated together the window covered the first 168,000 of 642,536 Hindi
lines and never reached a Tamil pair, so a run set up as joint reported
`by_language: {"hi": …}` and was read as one. It is now a reservoir over the whole file, and
`tests/corpus/test_pair_io.py` asserts the tail is reachable.

**What this does not close.** `TrainingPipeline` still has no neural stage, so a transformer
trained from scratch remains Python-API only. That is the Phase D CLI item that is still
open.

#### The local path first — done, 22 July 2026

Before an endpoint there has to be something to serve. `SemanticSearchPipeline.from_adapter`
loads a saved adapter, and `models/indic-v1` now answers queries rather than sitting on
disk as 3.4 MB of proven numbers.

The part worth recording is what the factory exists to prevent. `SemanticSearchPipeline`
already accepted any `TextEncoder`, so `cls(load_adapter(directory)[0])` would have
"worked" — loaded the right weights and then used them wrongly. An E5-family model is
trained with `query:` on one side and `passage:` on the other; served without them it
returns vectors of the right shape and norm, free of NaN, that encode the wrong thing.
Nothing raises. The score is simply lower, which is indistinguishable from the model not
being very good — and after the last two months of work, a quietly wrong retrieval number
is the most expensive defect this repository could ship.

So the prefixes are now the pipeline's, applied by `index` and `search` on their
respective sides, read out of the artefact by `from_adapter`, and readable back off a
`prefixes` property. `save_adapter` had been recording them since the start; nothing was
using them.

Two things came out of it that were not the goal:

- `index` now encodes the corpus in one `encode_batch` call rather than one `encode` per
  sentence. For a transformer that is the difference between indexing a corpus and waiting
  for it.
- That change fixed a silent bug in `SifEncoder`, whose common component is estimated from
  a batch and reused by `encode`. Indexing one sentence at a time never supplied a batch,
  so the component was never fitted and SIF had been degrading to a plain weighted average
  — with the right shapes and plausible results throughout. There is a regression test on
  it now.

#### The endpoint — done, 23 July 2026

`qfme serve --adapter models/indic-v1` puts a saved adapter behind the de facto
industry-standard embeddings schema. Three routes — `/health`, `/v1/models`,
`/v1/embeddings` — in a new top layer above `pipelines`, behind a `serve` extra so that
FastAPI stays optional for every other use. A test asserts in a fresh interpreter that
building the CLI parser leaves `fastapi` out of `sys.modules`, so an import moved to module
scope by a later edit fails the build rather than making the extra mandatory for
`qfme stats`.

The design question was not how to write the routes. It was that **the standard schema has
no field for the one thing this model cannot be served without.** There is nowhere to say
whether a string is a query or a passage. Three ways out, and only one whose failure mode is
visible:

| | Consequence |
|---|---|
| Default to `query` | Every passage-indexing job is silently wrong, and the symptom is a slightly worse index nobody attributes to this. |
| Default to `passage` | The same, on the other side. |
| Refuse, name both values, offer an operator default | One line of client code, once, per deployment. |

So an asymmetric model with no configured default answers `400` naming both valid values,
and `--default-input-type` exists for the deployment that genuinely is single-sided. That
gives up strict one-line client compatibility, which was the phase's stated exit criterion,
and it is the right trade: a client that changes only its base URL and gets quietly worse
retrieval has been served badly. The exit criterion should have said so.

The decisive test compares the vectors returned for the same text on each side and requires
them to differ. A server that stored `query: `, reported it in `prefix_applied` and never
prepended it would answer 200 with the right dimension, unit norm and no NaN, and every
other assertion would still pass.

Two defects surfaced while writing the model card, both of the same kind — a confident
number that was wrong:

- `max_length` came from `getattr(encoder, "max_length", 0)` and published `0` for a model
  whose real limit is 256. A client sizing its chunks from that field had no way to know.
  It now reads `adapter.json`, with defaults matching `load_adapter`'s exactly.
- Chasing that found `load_adapter` recording `normalize` in the manifest since format 1
  and never reading it back. An encoder saved with normalization off reloaded with it on:
  cosine scores unaffected, dot-product scores not, nothing raised. Fixed, with a test that
  asserts the property rather than the flag.

Not done, and the remainder of the phase: model versioning beyond a single served adapter,
cross-request batching, dimension truncation, ONNX export and quantisation, the container
image, auth and rate limiting. The endpoint binds `127.0.0.1` by default because of that
last one.

#### A consumer found a hole in the public surface — 23 July 2026

`quanfire-llm` now installs this repo as a pinned dependency rather than
copying from it, and the first thing that produced was a defect nobody here would have
found by reading their own code.

`SentencePieceTrainerAdapter` is a published name. The only type its constructor accepted,
`TokenizerConfig`, lives in `config`, which `CHANGELOG.md` puts explicitly outside the
guarantee. Both statements were fine on their own. Together they meant the published class
was reachable only by importing a type this repo may change in a patch release — so a
change this repo would correctly call internal could break their build, without being
breaking on this repo's own terms. A promise with a hole in it is worse than no promise,
because it is planned around.

`corpus.reader_for` had already solved this shape on the corpus side; the fix is
`tokenizer.trainer_for`, same shape, and the consumer asked for exactly that over the
alternative of re-exporting `TokenizerConfig` and declaring it public — which would have
frozen an internal type in place to close a hole in a different one. `vocab_size` and
`model_type` properties close the read-back path too. A test asserts no annotation in
`trainer_for`'s signature names `TokenizerConfig` or `config`, because the guarantee is the
entire point and review does not catch a regression in it.

The general lesson, which applies to every remaining repo split: **a name is public only if
every type in its signature is.** `tests/test_public_api.py` now names individual functions
rather than only packages, and the same rule is written into `CHANGELOG.md` beside the
surface it governs.

The corpus loader had the identical hole in five places — `load_corpus`,
`stream_documents`, `stream_sentences`, `build_reader` and `build_filter`, all exported,
all taking only a `CorpusConfig` — which the consumer had already worked around. Closed in
the same release by `corpus_from`, `documents_from` and `sentences_from`. The config forms
stay, because inside this repository a `CorpusConfig` already exists (a YAML file produced
it) and threading ten keyword arguments through `pipelines` to avoid a type it already
holds would be worse.

Second, cheaper report from the same session: wrapping reader *construction* in
`try`/`except` translates nothing, because `iter_documents` is a generator whose body does
not run until iteration, by which time the handler has exited. That is now a note beside
`CorpusReader` rather than something each consumer rediscovers.

### Phase E — From-scratch pretraining *(capability, not default)*

Masked-language pretraining followed by contrastive training, producing a model owned end
to end with no upstream licence.

Worth building for independence and for languages no pretrained checkpoint serves. Not
worth using where a fine-tuned open checkpoint is available and better.

- **Exit criterion:** a from-scratch model trained on the same corpus is within a
  defined margin of the fine-tuned one.
- **Compute:** roughly 30 days continuous locally for a 568M model over ~20B tokens, or
  ~3 days on a rented 4× A100 node.

---

## Compute profile

Training runs on a dedicated workstation:

| | |
|---|---|
| CPU | Intel i7-14700K, 20 cores / 28 threads |
| RAM | 32 GB |
| GPU | RTX 4070 Ti SUPER, **16 GB VRAM** |
| Storage | ~720 GB free |
| OS | Windows (use WSL2 — the training tooling is Linux-first) |

### Capability envelope

| Model size | Serve (bf16) | LoRA train | Full fine-tune |
|---|---|---|---|
| 118M encoder | yes | yes | yes |
| 278M encoder | yes | yes | yes |
| **568M encoder — the target** | yes | **yes** | yes, with an 8-bit optimizer |
| 1.5B | yes | yes | no |
| 4B | yes | yes | no |
| 8B | tight | **no** | no |

For a 568M base: full fine-tune with fp32 Adam leaves 7.5 GB for activations, an 8-bit
optimizer leaves 10.7 GB, and **LoRA over a frozen bf16 base leaves 14.9 GB**. Activations
are where contrastive training needs room, so LoRA is the configuration rather than an
economy measure.

**GradCache is required, not optional.** Contrastive quality depends on effective batch
size, because in-batch examples are each other's negatives. 16 GB fits roughly 8–16
sequences at 512 tokens against the 512–2048 good results need. Chunked representation
computation with recomputation on the backward pass reaches 1024+ at the cost of a second
forward pass. Without it this hardware cannot train a competitive model however long it
runs.

### Expected wall-clock

Contrastive fine-tune, 1M pairs, 3 epochs:

| Model | Wall-clock |
|---|---|
| 118M | ~0.8 days |
| **568M** | **~3.7 days** |
| 1.5B | ~9.7 days |

Validate every pipeline change on the 118M model first. Reserve 568M runs for
configurations already proven at small scale.

### Storage

~323 GB of the 720 GB free covers corpora, tokenised shards, base weights, checkpoints
and working space, provided corpora stay gzipped — which the readers handle
transparently. RAM and CPU are not constraints; the corpus layer streams, and data
preparation parallelises across 20 cores.

---

## What already exists

| Component | State |
|---|---|
| Corpus layer — segmentation, scripts, readers, dedup, statistics | Complete, 22 scheduled Indian languages plus others |
| Tokenizer — normalizers, pre-tokenizers, subword training | Complete |
| Vocabulary — deterministic ordering, pinned special ids | Complete |
| Evaluation — per-language metrics, structural geometry, reports | Complete |
| `TextEncoder` contract | Complete (Phase 0) |
| Static baseline — word2vec | Complete; kept as a measurement baseline |
| Config, artefacts, reproducibility, CLI, CI | Complete |
| Transformer encoder | **Phase A** |
| Contrastive training | **Phase B** |
| Pair mining | **Phase C** — structural/adjacency/metadata miners, `mine-aligned` (cross-lingual via langlinks) and `mine-negatives` (hard negatives, with an absolute ceiling and the new `--positive-margin` relative guard) all done and measured |
| Serving | **Phase D** — local path (`from_adapter`), the adaptation experiment as a command (`qfme adapt`) and the HTTP endpoint (`qfme serve`) all done; dimension truncation, ONNX, container, auth and a neural stage in `TrainingPipeline` outstanding |
| From-scratch pretraining | **Phase E** |

---

## Architecture policy

**Use proven architectures. Do not invent new ones.**

Transformers for text, U-Net or diffusion transformers for images, and whatever is
established for a modality when it is reached. Inventing an architecture that beats these
is a research programme with poor odds and compute requirements far beyond this hardware,
and it is explicitly not the objective.

What is written out here is *our implementation* of a standard design, which is a
different thing from a new design. Owning the implementation means the training loop can
be trusted and inspected; owning the architecture would mean owning a research risk.

The differentiation comes from data and domain, not from novel mathematics. A standard
architecture trained on a corpus nobody else holds beats a novel architecture trained on
the same public data as everyone else.

The levers worth pulling, none of which require new mathematics:

| Lever | Effect |
|---|---|
| Domain-specific tokenizer | Domain terms become single pieces rather than fragments |
| Training objective and pair selection | Where domain adaptation actually lives |
| Matryoshka dimensions | Truncatable vectors; cheaper storage, one model |
| Multi-vector late interaction | Better retrieval than single-vector, at higher index cost |
| Hybrid sparse and dense | Exact term matching alongside semantics |

**Why word2vec stays.** Not as a product — as the baseline. Its limitation is structural
rather than a matter of training: one row per token id means `river bank` and `savings
bank` receive byte-identical vectors, and no quantity of data changes that. It is kept
because every exit criterion in this roadmap is of the form "beats X", and without a
baseline that claim is unfalsifiable. It also trains in seconds on CPU with no torch,
which makes it the pipeline's smoke test.

## Principles

- **Measure before claiming.** Every phase states the baseline it must beat.
- **Report per language and per domain, never only an average.** An average is how a
  model that fails half its inputs looks acceptable.
- **Validate on the small model first.** A 118M run costs hours; a 568M run costs days.
- **Publish limitations.** The documentation states plainly where the framework stops.
- **Reproducible artefacts.** Seeded runs, configuration persisted beside every model.
