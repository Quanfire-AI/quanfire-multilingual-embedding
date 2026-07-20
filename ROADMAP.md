# Roadmap

> An embedding model factory: corpus in, trained and evaluated model out — generic or
> domain-specific.

**Status:** Phases 0 and A complete. Phases B–E planned.

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
415 MB model**, and Adam's optimizer state falls from **0.81 GB to 6.8 MB**. The rank is
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

**Still to do.** Adapting an external pretrained checkpoint, hard-negative mining,
Matryoshka truncation, checkpoint resumption.

**Unverified.** Development happens without an NVIDIA GPU, so the CUDA paths are exercised
by nothing local. bf16 autocast is genuinely tested on CPU, including that the loss still
falls, but the memory and throughput claims that justify it are open until a run on real
hardware.

- **Exit criterion:** a fine-tuned model beats its own base checkpoint on a held-out set
  from the same domain. Beating the base is the honest bar — beating a commercial API is
  a separate claim requiring a separate benchmark.

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

- **Deliverables:** miners for each strategy; hard-negative mining against a base
  encoder; pair quality filtering and deduplication; a `qfme mine` command.
- **Exit criterion:** a model trained purely on mined pairs from a domain corpus beats
  the untrained base on that domain.

### Phase D — Serving

A web service over the artefact-loading pattern already in place, using the de facto
industry-standard request and response schema so existing clients migrate by changing a
base URL.

- **Deliverables:** embeddings endpoint; batching; model versioning; dimension
  truncation; ONNX export and quantisation; container image; auth and rate limiting.
- **Exit criterion:** p95 under 100 ms for a short input, and a client switches by
  changing only the base URL.

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
| Pair mining | **Phase C** |
| Serving | **Phase D** |
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
