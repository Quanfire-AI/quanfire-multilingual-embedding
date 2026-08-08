# QuanFire embedding models — catalogue & naming

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
| `domain` | what it is tuned for | `general` · `legal` · `finance` · `medical`* · `gov`* |
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
| `quanfire-ai/embed-legal-indic` | e5-small | in training | `legal-indic-*` | Indian legal (Supreme Court judgments, statutory public domain — Copyright Act §52(1)(q), official court portals). Eval origin-walled on MILPaC. |
| `quanfire-ai/embed-finance-indic` | e5-small | planned | — | CA firms: accounting / tax / audit / filings. |

## Two growth axes

- **Language breadth** deepens the *one* flagship (`multilingual-embedding`) — a
  single strong multilingual model beats a scatter of per-language ones. Breadth =
  more parallel data into the flagship, not more repos.
- **Domain depth** ships *separate* specialists, one per `domain`×`scope`. This is
  the defensible work; generic models are free.
