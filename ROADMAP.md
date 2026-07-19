# Roadmap

> From a word2vec baseline to a served, Indic-first text embedding model.

**Status:** Stage 0 **complete**. Stages 1–5 are planned, not built.
Everything described past Stage 0 is a proposal, not a commitment.

---

## Positioning

The objective is a general-purpose text embedding model of commercial quality, exposed
over an API.

The strategy is **not** to match the leading commercial embedding services on English.
That is not winnable: several open models already outperform them on public leaderboards,
built by organisations with compute budgets orders of magnitude larger than ours. Entering
that race means losing it slowly.

The strategy is to **own Indian languages**. The major commercial and open models are all
weak on Indic text. The best existing multilingual efforts cover 12 to 17 Indian
languages. Nine of the 22 scheduled languages — Santali, Bodo, Dogri, Maithili, Meitei,
Konkani, Kashmiri, Sindhi and Sanskrit — are served badly or not at all by anyone.

This repository already handles the text of all 22 correctly. That is the foundation of
a defensible position: *the best embedding model for Indian languages*, rather than a
worse version of a general-purpose one.

---

## What already exists, and what does not

The existing framework is roughly 40% of the work, and it is the 40% that teams usually
get wrong.

| Component | Reusable? | Notes |
|---|---|---|
| Corpus layer | **Yes, entirely** | Script-aware segmentation, combining-mark handling, readers, deduplication, statistics. Correct across 22 scheduled languages. |
| Tokenizer | **Yes** | Still required. May be replaced by the base model's own tokenizer in Stage 2. |
| Vocabulary | Partly | Needed for the baseline; a transformer brings its own. |
| Evaluation harness | **Yes, critical** | Per-language fairness reporting is exactly the evidence needed to substantiate the Indic claim. |
| Config, artifacts, reproducibility | **Yes** | Directly reusable. |
| CLI and pipelines | **Yes, with extension** | Structure holds; a new encoder path is added. |
| CI, packaging, docs | **Yes** | Directly reusable. |
| `embedding/word2vec.py` | **No** | Retained as a baseline to measure against, not as a path forward. |

### The one architectural blocker

`EmbeddingModel.train()` returns an `EmbeddingMatrix`, which is a `vocabulary × dimension`
lookup table. A transformer has no such table: it computes a vector per *input text* at
call time. `SentenceEncoder` and `SemanticSearchPipeline` are both built on
`EmbeddingMatrix` too.

So the current contract **cannot** accommodate a contextual model. An earlier version of
the handbook claimed a transformer "would slot into the existing pipeline without changes
elsewhere". That was wrong, and it is corrected. Fixing it is Stage 0.

---

## Stages

Each stage has an exit criterion. A stage is not finished when the code works; it is
finished when the criterion is met.

### Stage 0 — Make the architecture admit contextual models ✅ **done**

**Why first:** everything downstream depends on it, and it is small.

Introduce a `TextEncoder` abstraction — `encode(texts: list[str]) -> np.ndarray` — that
both static and contextual models implement. Re-point `SemanticSearchPipeline` at
`TextEncoder` rather than `EmbeddingMatrix`. The word2vec path becomes one implementation;
nothing about its behaviour changes.

- **Delivered:** `embedding/encoder.py` defining the `TextEncoder` protocol;
  `SemanticSearchPipeline` re-pointed at it, with `matrix` and `tokenizer` now optional
  extras rather than requirements; `from_static()` preserving the static path;
  17 new tests, two of them architectural.
- **Exit criterion — met.** All 963 prior tests still pass, and a `HashingEncoder`
  backed by no model, no vocabulary and no matrix indexes and searches end to end.
- **Actual effort:** well under the estimated week. No GPU, no new dependencies, no
  behaviour change to any existing encoder — both shipped encoders already satisfied the
  contract, which is why it was defined to match them.

The contract is a `Protocol` rather than an ABC, deliberately: a future contextual
encoder satisfies it by shape, without importing anything from this package.

### Stage 1 — Build the evaluation benchmark, before any model

**Why before the model:** you cannot claim "best for Indic" without evidence, and there is
no adequate public benchmark for the nine low-resource languages. Building it is both the
proof and a moat — a credible public Indic retrieval benchmark is itself a contribution
that attracts users.

Assemble held-out retrieval and similarity sets per language, drawing on the public
multilingual retrieval benchmarks, the large Indic parallel corpora, and the standard
Indic language-understanding suites. Several cover only the well-resourced languages; for
the rest, bootstrap from parallel text and human-verify a small gold set.

Score three systems on it: our word2vec baseline, a strong open multilingual model, and a
leading commercial embedding API. **Publish the numbers even where we lose** — a benchmark
that only flatters its author convinces nobody.

- **Deliverables:** per-language eval sets; a `qfme benchmark` command; a public
  leaderboard table; baseline numbers for three models.
- **Exit criterion:** a reproducible number for every one of the 22 languages, and a
  documented, defensible gap where the incumbents are weak.
- **Effort:** 3–4 weeks. Mostly data work. No GPU.
- **Risk:** for Santali, Bodo, Dogri and Meitei there may be too little text to build a
  meaningful set. If so, say so publicly rather than fabricating one.

### Stage 2 — Fine-tune an open base model

Contrastively fine-tune an open multilingual base model on Indic pairs. Select it against
these criteria rather than by reputation:

| Criterion | Target |
|---|---|
| Parameters | 500M–700M — trainable on our hardware, large enough to be competitive |
| Licence | Permissive, allowing commercial use and redistribution |
| Language coverage | 100+, with the Indic scripts already in its tokenizer |
| Context length | 512 minimum; longer is useful but not required for embeddings |
| Architecture | Encoder, or a decoder adapted with bidirectional attention |

Technique: InfoNCE with in-batch negatives plus mined hard negatives. Large effective
batch matters more than almost anything else for contrastive training — use GradCache if
GPU memory is the constraint. Add Matryoshka representation learning so callers can
truncate dimensions, which is the interface commercial APIs expose.

- **Deliverables:** training pipeline; `TransformerEncoder` implementing `TextEncoder`;
  a trained checkpoint; Stage 1 benchmark numbers for it.
- **Exit criterion:** **beats the leading commercial embedding API on the Indic
  benchmark** for at least the 12 well-resourced languages, without regressing badly on
  English.
- **Effort:** 4–6 weeks.
- **Compute:** runs locally on the 16 GB RTX 4070 Ti SUPER using LoRA plus GradCache —
  see the compute profile below. Expect 1.5–3 days per full run over ~1M pairs, so
  validate the pipeline on a smaller base model first. **No rental cost for this stage.**

### Stage 3 — Serve it

A web service over the artifact-loading pattern that already exists. Adopt the **de facto
industry-standard request and response schema** so the service is a drop-in replacement
for existing embedding APIs — that removes the main adoption barrier at essentially zero
design cost.

- **Deliverables:** `POST /v1/embeddings`; batching; model versioning; dimension
  truncation; ONNX export and quantization; container image; latency and throughput
  benchmarks; auth and rate limiting.
- **Exit criterion:** p95 latency under 100 ms for a single short input on CPU, and an
  existing client can switch to us by changing only the base URL.
- **Effort:** 3–4 weeks.
- **Note:** serving cost, not training cost, will dominate the budget from here on.

### Stage 4 — Deeper training *(conditional)*

Only if Stage 2 demonstrates real demand. Continued pretraining on the large Indic
monolingual corpora — roughly 20B tokens across 24 languages — followed by contrastive
training, rather than fine-tuning alone.

- **Exit criterion:** a material gain over the Stage 2 checkpoint on the same benchmark.
- **Effort:** 3–6 months.
- **Compute:** feasible locally, at roughly **30 days of continuous training** for a
  568M model over ~20B tokens. That is a month of electricity rather than a rental
  invoice. Renting a 4× A100 node compresses it to ~3 days if time matters more than
  money. Either way, do not begin without Stage 2 numbers justifying it.
- **Caveats for the local route:** a consumer card at full load for a month needs real
  cooling and stable power, checkpoint frequently and assume at least one crash, and the
  workstation is unavailable for anything else throughout.

### Stage 5 — Multimodal *(parked)*

Text-to-image, image-to-text and speech are each separate multi-year projects with
different data, architectures and evaluation. They share almost nothing with this codebase
beyond config and serving scaffolding.

**Deliberately not planned in detail.** Revisit only once the text model is serving real
traffic. Sequencing these earlier would starve the one thing with a defensible position.

---

## Compute profile

Training happens on a dedicated workstation, separate from the development machine:

| | |
|---|---|
| CPU | Intel i7-14700K, 20 cores / 28 threads |
| RAM | 32 GB |
| GPU | RTX 4070 Ti SUPER, **16 GB VRAM** |
| Storage | ~720 GB free |
| OS | Windows (x64) |

**This is sufficient for Stages 0 through 3.** Only Stage 4 requires rented hardware.

### What fits in 16 GB

For a 568M-parameter base model, against the usual training-memory formula:

| Configuration | Model + optimizer | Left for activations |
|---|---|---|
| Full fine-tune, bf16 + fp32 Adam | 8.5 GB | 7.5 GB |
| Full fine-tune, bf16 + 8-bit Adam | 5.3 GB | 10.7 GB |
| **LoRA (r=16), bf16 frozen base** | **1.1 GB** | **14.9 GB** |

**Use LoRA.** Activations are where contrastive training actually needs room, and LoRA
leaves nearly the whole card for them. A full fine-tune fits only with 8-bit Adam and a
batch size too small to be useful.

### The technique that makes this work

Contrastive training quality depends heavily on **effective batch size**, because
in-batch examples serve as each other's negatives. A 16 GB card fits perhaps 8–16
sequences at 512 tokens — far below the 512–2048 that good results need.

**GradCache** solves this: representations are computed in chunks without gradients, then
recomputed chunk-by-chunk during the backward pass. It decouples effective batch size from
VRAM at the cost of roughly a second forward pass. With it, effective batches of 1024+ are
reachable on this card. Without it, this hardware cannot train a competitive contrastive
model, regardless of how long it runs.

### Expected wall-clock

The 4070 Ti SUPER is roughly a third of an A100 for this workload (~44 vs ~156 dense
bf16 TFLOPS, 672 vs 1555 GB/s bandwidth). A LoRA run over ~1M pairs that takes 12–24h on
an A100 should be expected to take **1.5–3 days** here.

That makes full runs a weekend activity, not an interactive one — so **iterate on a
smaller base model first**. A ~120M-parameter multilingual encoder trains several times
faster and validates the entire pipeline; move to the 568M model only once the data, loss
and evaluation are known-good. Reserve the long runs for configurations already proven at
small scale.

### Capability envelope

What the workstation can and cannot train, computed from the memory formula above and a
FLOP estimate at ~30% achieved utilisation.

| Model size | Serve (bf16) | LoRA train | Full fine-tune |
|---|---|---|---|
| 118M encoder | yes | yes | yes |
| 278M encoder | yes | yes | yes |
| **568M encoder — the target** | yes | **yes** | yes, with an 8-bit optimizer |
| 1.5B | yes | yes | no |
| 4B | yes | yes | no |
| 8B | tight | **no** | no |

Training time, contrastive fine-tune over 1M pairs for 3 epochs:

| Model | Wall-clock |
|---|---|
| 118M | ~0.8 days |
| **568M** | **~3.7 days** |
| 1.5B | ~9.7 days |

**The ceiling is the 568M class**, and that is the strategic point rather than a
disappointment. The models topping public leaderboards are 8B and cannot be trained here
at any speed. Out-scaling them is not available; out-specialising them is. A focused
568M model trained on data nobody else has beats a general 8B model on that data, and
this hardware trains it over a weekend.

The binding constraint remains **data, not compute**.

### Storage budget

The full pipeline fits with room to spare, provided corpora stay compressed — which the
readers handle transparently.

| Item | Size |
|---|---|
| Indic monolingual corpora, 24 languages, gzipped | ~120 GB |
| Parallel corpora, ~50M pairs, gzipped | ~18 GB |
| Tokenised training shards | ~60 GB |
| Base model weights and variants | ~15 GB |
| Checkpoints | ~25 GB |
| Evaluation artefacts | ~5 GB |
| Working space and headroom | ~80 GB |
| **Total** | **~323 GB of 720 GB free** |

RAM and CPU are not constraints. The corpus layer streams rather than materialising, and
the data preparation that dominates Stage 1 parallelises well across 20 cores.

### Practical notes

- **Use WSL2, not native Windows.** CUDA support is full, performance is near-native, and
  the ML tooling — `bitsandbytes`, GradCache implementations, most training scripts — is
  Linux-first. Native Windows will cost time on dependency problems that have nothing to
  do with the model.
- **Storage needs planning.** The full Indic monolingual corpora run to ~20B tokens and
  will not sit comfortably alongside checkpoints in 720 GB. Keep corpora gzipped — the
  framework's readers already handle `.gz` transparently — and subset by language tier.
- **32 GB system RAM is adequate** because the corpus layer streams rather than
  materialising. That design choice, made for a different reason, pays off here.
- **PyTorch becomes a dependency at Stage 2.** This is the first real change to the
  project's dependency posture and should be a deliberate, recorded decision. Keep it
  confined to the new encoder package so the existing NumPy pipeline stays installable
  without it.

---

## What will actually be hard

Ordered by how likely each is to sink the project.

**Data, not compute.** Contrastive training needs *pairs*, not raw text. For the nine
low-resource languages, retrieval pairs essentially do not exist. Bootstrapping via
translation and transliteration is possible but introduces bias that the evaluation must
be honest about. **Start sourcing data now** — it gates Stage 2 and it is the long pole.

**Benchmark credibility.** A self-published benchmark by the model's author is easy to
dismiss. Mitigate by publishing the construction method, releasing the eval sets, scoring
competitors fairly, and reporting losses.

**Evaluation for languages with no ground truth.** For Santali or Bodo there may be no
one to verify a gold set cheaply. Budget for native-speaker review, or scope those
languages as best-effort and label them as such.

**Licence compatibility.** Verify any base model's terms before building on it, and
record the training data's licence — a model inherits the constraints of its corpus. A
permissive base licence is a hard selection criterion, not a preference.

**Serving economics.** A 568M model on CPU is viable; on GPU it is not free. Model the
per-request cost before committing to a pricing structure.

---

## Immediate next steps

1. **Stage 0** — the encoder abstraction. Small, unblocks everything, no external
   dependency. Can start immediately.
2. **Data sourcing in parallel** — audit what exists per language, starting with
   the public Indic parallel corpora, the monolingual corpora, and the multilingual
   retrieval benchmarks. This is the long pole; begin it before it is needed.
3. **Decide the language tier** — all 22 at best-effort, or 6–8 with genuine depth. This
   determines the shape of Stages 1 and 2.

## Open decisions

| Decision | Why it matters | Status |
|---|---|---|
| GPU access — owned or rented? | Determines Stage 2 iteration speed. | **Resolved** — local RTX 4070 Ti SUPER (16 GB) covers Stages 0–3; only Stage 4 needs rental |
| Language tier — all 22, or a focused subset? | Drives benchmark and data effort. | Open |
| Base model | Licence and size trade-off; see the selection criteria in Stage 2. | Open |
| Open weights or closed? | Affects adoption, benchmark credibility, and commercial model. | Open |
| Target dimensions | Matching the common 1536/3072 sizes eases migration from existing APIs. | Leaning Matryoshka, truncatable |

---

## Principles carried forward

The existing framework's discipline should survive this expansion:

- **Measure before claiming.** The benchmark precedes the model deliberately.
- **Report per-language, never only an average.** An average is how a model that fails
  half its languages looks acceptable.
- **Publish limitations.** The current documentation states plainly where the framework
  stops; that should remain true as capability grows.
- **Reproducible artefacts.** Seeded runs and configuration persisted alongside every
  model.
