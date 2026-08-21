# Quanfire embedding models — catalogue & naming

Models are trained on the GPU box and published under the
[`quanfire-ai`](https://huggingface.co/quanfire-ai) org on Hugging Face. This file
is the single source of truth for how they are named and what exists.

## Public model id

```
quanfire-ai/<pillar>-<domain>-<scope>
```

The version is **not** in the name — it lives in the HF git tag / revision, so the
id stays stable and consumers pin a revision (as the adapters already pin their
base checkpoint's revision).

| Field | Meaning | Values |
|---|---|---|
| `pillar` | which capability | `embed` (bi-encoder retriever, this project) · `rerank` (cross-encoder that reorders a retriever's top-k) · `gen` (reserved: generative pillar) |
| `domain` | what it is tuned for | `general` · `legal` · `statute` · `finance` · `medical`* · `gov` |
| `scope` | language / region reach | `multi` (broadly multilingual) · `indic` · `in` (India, multi-script) · `en` · an ISO code |

`*` reserved — not yet trained. Add a domain/scope code here **before** publishing a
model that uses it, so the registry stays closed.

"CA firms" maps to `domain = finance` (accounting, tax, audit, filings).

### Version tags

`vMAJOR.MINOR.PATCH` on the HF repo:
- **MAJOR** — retrain that changes behaviour incompatibly (new base, new objective).
- **MINOR** — better weights, same contract (more data, more epochs).
- **PATCH** — card/metadata only.

Every adapter also pins the exact upstream revision of its frozen base checkpoint.

## Internal run id (box only — never published)

Training runs on the box carry the hyper-parameters the public id hides:

```
<domain>-<scope>-a<AA>s<SS>[-e<EPOCHS>][-<note>]
```

e.g. `legal-indic-a70s30-e3` — the `a70s30` blend and epoch count that produced a
release. At publish time the run id is recorded in the model card; the public repo
id stays clean.

## Catalogue

| Public id | Base | Status | Internal run | Notes |
|---|---|---|---|---|
| `quanfire-ai/multilingual-embedding` | e5-small | **published** | `prod-a70s30-fr` | Flagship general-multilingual. Predates this scheme; kept as-is (renaming a published repo breaks every inbound link). It **is** the `embed-general-multi` slot. |
| `quanfire-ai/embed-legal-en` | e5-small | **published** | `legal-indic-e1` | English Supreme Court **judgment** retriever. In-distribution Recall@1 0.309→0.545 (+76%, CIs disjoint). Trained only on statutory public-domain judgment text (Copyright Act §52(1)(q), headnotes excised, official court portals). Out-of-origin transfer to statutory text tested and **flat** — judgment-specific by design; scope is English judgments, not general legal. |
| `quanfire-ai/embed-statute-en` | e5-small | **published** | `statute-en-e2` | English **central-statutory** (bare-Act) retriever — the register `embed-legal-en` measured itself *flat* on. In-distribution Recall@1 0.182→**0.269** (+48%, CIs disjoint [.165,.199]→[.250,.289]); on the un-gameable low-lexical-overlap slice (`<0.3`) 0.077→**0.177** (+131%). Trained on 858 Central Acts (Zenodo 5088102, CC-BY-4.0) mined into section-level pairs + adapter-mined hard negatives (positive-margin 0.05); §52(1)(q)(ii) makes bare-Act text train-safe for a **non-reconstructive** embedder → Apache-2.0 weights, corpus **not** redistributed. Statute-specific: judgments are `embed-legal-en`; state legislation, rules and notifications are unmeasured. |
| `quanfire-ai/embed-gov-indic` | e5-small | **published** | `gov-indic-e4-ep6` (v1.1) | **Cross-lingual** Indian **government press-release** retriever (16 Indian languages). In-distribution cross-lingual Recall@1 0.184→0.281 (**+53%**, CIs disjoint, also significant over v1.0's 0.235); right passage in the top-10 ~84% of the time. Trained on Press Information Bureau (PIB) releases, reused under PIB's reproduction policy (attribution, **no NC/SA**); non-reconstructive → Apache-2.0 weights. Specialist showcase of clean-provenance multilingual retrieval — other domains not validated; lowest-resource langs (Khasi/Nepali/Manipuri) thin. **v1.1** trains the same balanced corpus to 6 epochs (v1.0 was undertrained at 1 epoch); 13/16 langs improve but Gujarati & Odia regress vs v1.0 — both kept by revision tag. Runs: e1 (+8.7%, ns) → e2 (+16.2%, ns) → e3/v1.0 (+27.9%, significant) → e4/v1.1 (+53%, significant over base & v1.0). **CLAIM CORRECTED 2026-08-21 — the +53% was measured on a leaking split.** `pib.py` emits both directions of every language pair, and the old split filter excluded a training pair only when its *positive* was a held-out positive, so the reverse of a held-out pair trained freely. Re-run on a clean split (`gov-indic-e4c`, same seed, corpus, hyperparameters and 6 epochs): **0.1836 → 0.2518, +37.1%**, CIs [0.1651, 0.2038] vs [0.2307, 0.2741], still disjoint. The direction and the significance survive; the magnitude does not. **Not volume-matched** — the fix cuts training from 7,270 pairs to 2,115, because one press release yields many same-document pairs across 16 languages, so most of the pool shares a document with the held-out set. The shipped weights are unchanged; this corrects what we claim about them, and the honest reading is that this corpus is too document-poor to support a clean high-volume train. |
| `quanfire-ai/rerank-statute-en` | e5-small (cross-encoder head) | **published** | `statute-reranker-v0` (v1.0.0) | **First reranker.** Cross-encoder (full e5-small + scalar relevance head, not a LoRA adapter) that reorders a bi-encoder's top-k on English central-statutory text. Rides `embed-statute-en`'s exact corpus (858 Central Acts, Zenodo 5088102 CC-BY-4.0) + form-matched negatives → non-reconstructive, Apache-2.0 weights, corpus not redistributed. **Run 2 (box/CUDA, 1200 held-out queries): retrieve-then-rerank Recall@1 0.205→0.336 (+63.8%), paired bootstrap 95% CI [+0.106,+0.157], excludes 0.** Recall@100 ceiling 0.718 (headroom unspent → a v1.1 spends it). Run 1 had collapsed to random (−96.3%) on form-separable mined negatives; the fix draws negatives from other records' positives (form-matched) so train distribution == eval distribution. Published to HF v1.0.0, public, full-model repo (470 MB). Different objective from every published bi-encoder. |
| `quanfire-ai/rerank-gov-indic` | e5-small (cross-encoder head) | **published** | `gov-reranker-v2-semihard` (v1.0.0) | **Cross-lingual government reranker** — reorders a retriever's top-k over Indian **government press-release** text in **16 Indian languages**. Full fine-tune of multilingual-e5-small with a single-logit ranking head (`max_length` 256, 2 epochs, lr 2e-5, seed 0, bf16/CUDA). Rides `embed-gov-indic`'s PIB corpus (reproduction policy, attribution, no NC/SA) → non-reconstructive, Apache-2.0 weights, corpus not redistributed. **Retrieve-then-rerank Recall@1 0.3083 → 0.4067 (+31.9%), paired bootstrap 95% CI [+0.0692, +0.1275], excludes 0** (1,200 cross-lingual queries, 1,096-passage pool, B=2,000 seeded). Lifts 42.9% of the recoverable queries (top-100 ceiling 0.9475) to rank 1. **The finding is the negative-hardness curve:** random negatives +10.5%, the retriever's *hardest* top candidates **−45.1% (a regression** — near-duplicate releases about the same event, so training on them teaches the model to demote correct answers), semi-hard negatives from ranks [20,100) **+31.9%** (shipped). The shipped CI's lower bound sits above the random-negative version's upper bound, so it strictly dominates. Aggregate across the mixed-language query set — **not** a per-language guarantee; other domains unvalidated. |
| `quanfire-ai/embed-legal-indic` | e5-small | planned | — | **Cross-lingual** Indian legal (en↔hi/ta). Separate from `embed-legal-en`: needs Indic legal parallel data before language becomes an adapted axis. Not the English judgment model. |
| `quanfire-ai/embed-finance-indic` | e5-small | planned | — | CA firms: accounting / tax / audit / filings. |

## Two growth axes

- **Language breadth** deepens the *one* flagship (`multilingual-embedding`) — a
  single strong multilingual model beats a scatter of per-language ones. Breadth =
  more parallel data into the flagship, not more repos.
- **Domain depth** ships *separate* specialists, one per `domain`×`scope`. This is
  the defensible work; generic models are free.
