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
| `pillar` | which capability | `embed` (this project) · `gen` (reserved: generative pillar) |
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
| `quanfire-ai/embed-gov-indic` | e5-small | **published** | `gov-indic-e3` | **Cross-lingual** Indian **government press-release** retriever (16 Indian languages). In-distribution cross-lingual Recall@1 0.184→0.235 (+27.9%, CIs disjoint); right passage in the top-10 ~74% of the time. Trained on Press Information Bureau (PIB) releases, reused under PIB's reproduction policy (attribution, **no NC/SA**); non-reconstructive → Apache-2.0 weights. Specialist showcase of clean-provenance multilingual retrieval — other domains not validated; lowest-resource langs (Khasi/Nepali/Manipuri) thin. Shipped in three runs: e1 (+8.7%, ns) → e2 (+16.2%, ns) → e3 (significant). |
| `quanfire-ai/embed-legal-indic` | e5-small | planned | — | **Cross-lingual** Indian legal (en↔hi/ta). Separate from `embed-legal-en`: needs Indic legal parallel data before language becomes an adapted axis. Not the English judgment model. |
| `quanfire-ai/embed-finance-indic` | e5-small | planned | — | CA firms: accounting / tax / audit / filings. |

## Two growth axes

- **Language breadth** deepens the *one* flagship (`multilingual-embedding`) — a
  single strong multilingual model beats a scatter of per-language ones. Breadth =
  more parallel data into the flagship, not more repos.
- **Domain depth** ships *separate* specialists, one per `domain`×`scope`. This is
  the defensible work; generic models are free.
