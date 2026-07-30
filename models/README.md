# models/

> Where a trained adapter lives once it has been brought back from the training machine.

Training happens where the GPU is. Serving, inspection and every example in this
repository happen wherever you are reading this. What crosses between the two is a
directory of a few megabytes, and this is where it goes.

## Where to put it

```
models/indic-v1/
├── adapter.json     the manifest — base checkpoint, pooling, prefixes, LoRA shape
└── adapter.pt       the low-rank weights, about 3.4 MB
```

Copy the whole directory, both files. That exact path is not a suggestion: it is the one
already written into `SemanticSearchPipeline.from_adapter`'s docstring, into
[`data/README.md`](../data/README.md), and into
[`tests/integration/test_saved_adapter.py`](../tests/integration/test_saved_adapter.py).
A copy somewhere else works, but nothing will find it for you.

## The adapters that exist

`indic-v1` above is the original two-language (hi/ta) adapter, tracked because it predates
`qfme adapt` and carries the first published numbers. Everything since is produced by
`qfme adapt` from a committed configuration, so it is reproducible and stays git-ignored —
but the names recur throughout [ROADMAP.md](../ROADMAP.md) and the reports, so they are
worth stating once:

| Directory | What it is | Status |
|---|---|---|
| `indic-v1` | Two-language (hi/ta) LoRA, the original | Tracked; historical baseline |
| `indic-aligned-v1` | Ten-language adapter trained on cross-lingual aligned pairs | Superseded by v2 |
| `indic-aligned-v2` (a.k.a. `indic-aligned-multi-np`) | The refined ten-language adapter | **Canonical** — the cross-lingual numbers are measured on this |
| `indic-aligned-c250k` | Control: same 250k pairs, no hard negatives | Ablation reference |
| `indic-aligned-hn` | Same pairs plus mined hard negatives | Ablation; a trade-off, not promoted |
| `indic-aligned-hn-gated` | Same pairs plus hard negatives mined with the `--positive-margin` guard (false negatives removed) | Ablation; the gated win — recovers in-domain *and* keeps FLORES |
| `samanantar-proof` | Sentence-scale Samanantar en↔indic bitext only (240k, 8 langs) | Proof; closes FLORES (0.9896, beats base) but gives back in-domain |
| `indic-aligned-mix` | ~50/50 blend of article pairs and Samanantar sentence pairs (490k) | Proof-scale both-at-once — closes FLORES *and* holds in-domain; superseded by the production sweep |
| `prod-a30s70` / `prod-a50s50` / `prod-a70s30` | Production ratio sweep: article : sentence at 30:70 / 50:50 / 70:30, 1.0M pairs, all 10 langs (Samanantar + itihasa sa + opus-100 ur) | Sweep to tune the blend ratio at scale |
| `prod-a70s30` | The 70:30 winner of that sweep | **Promotion candidate over v2** — beats v2 on *both* instruments (FLORES 0.9805 vs 0.9609, in-domain 0.9029 vs 0.8964) |

The ten-language adapters raise cross-lingual retrieval from 0.7875 → 0.8964 recall@1
in-domain. The plain hard-negative variant (`indic-aligned-hn`) trades ~1.5 points in-domain
(0.8955 → 0.8801 X↔Y) for ~1.5 on FLORES-200 (0.9606 → 0.9757). The **gated** variant
(`indic-aligned-hn-gated`), mined with `--positive-margin` so its suspicion rate falls from
0.498 to 0.000, removes that trade-off: it recovers in-domain to 0.8940 (level with control)
*and* keeps the FLORES gain at 0.9768 — so gated hard negatives are a genuine improvement, a
candidate to reconsider for promotion over v2.

The **public-bitext limit** — v2 regressing to 0.961 non-Hindi on FLORES-200 where base E5
scores 0.985 — turned out to be a *scale* gap, not a ceiling: v2 trained on article-scale
pairs, FLORES is a sentence benchmark. `samanantar-proof`, trained only on sentence-scale
Samanantar en↔indic bitext, **beats base E5** on FLORES (0.9896) but mirrors the trade-off
(in-domain 0.8195). The proof-scale ~50/50 blend `indic-aligned-mix` collapsed the trade-off,
and the **production sweep settled it**: a 1.0M-pair, ten-language blend (Samanantar + itihasa
sa + opus-100 ur) swept over three article:sentence ratios. The 70:30 winner, `prod-a70s30`,
**beats v2 on both instruments at once** — FLORES **0.9805** (v2 0.9609, near base) *and*
in-domain **0.9029** (v2 0.8964) — so it is the promotion candidate over v2. The dedicated
sa/ur sources did *not* lift sa/ur on FLORES (transfer already covered them); that null result
is recorded, not hidden. Full working is in the ROADMAP entries dated 27–30 July 2026. Before
`prod-a70s30` is stamped canonical everywhere, the *other* published numbers (mono-lingual,
hi-pivot) still need a re-baseline on it. When one of these is the canonical serving artefact,
copy it into place exactly as below.

## Then verify it

```bash
pytest tests/integration/test_saved_adapter.py -q -rs
```

Fifteen tests. They skip when nothing is here, so a fresh clone is quiet; once the
directory exists they assert that the copy is intact, that it reloads into a working
encoder, and that it serves through the search pipeline with the right prefixes.

They also skip if the base checkpoint named in `adapter.json` is not in the local Hugging
Face cache, because nothing in the test suite reaches the network. Warm it once:

```bash
python -c "from transformers import AutoModel, AutoTokenizer; \
n='intfloat/multilingual-e5-small'; AutoTokenizer.from_pretrained(n); AutoModel.from_pretrained(n)"
```

`-rs` prints the reason for each skip, which is how you tell "not set up yet" from
"quietly not testing anything".

## Why the copy needs verifying at all

Every other artefact in this repository is reproducible: a config file plus a corpus
regenerates it. An adapter is not, for two reasons. Its base is an external name whose
contents can change upstream, and `indic-v1` predates `qfme adapt` — it was produced by a
script, so no committed configuration rebuilds it. It is also the artefact every
published number was measured on.

That makes the hand-copy the weakest link in the whole pipeline: the one step with no
config, no checksum and no run behind it. And its failure mode is silent. A truncated
`adapter.pt` or a manifest from a different run still reloads into an encoder that
returns vectors of the right shape and the right norm, free of NaN, encoding something
other than what was measured. Nothing raises.

## Why this directory is tracked, mostly

`.gitignore` ignores `models/*` — trained artefacts are build outputs and do not belong
in git. `models/indic-v1/` and this file are the documented exceptions: 3.4 MB is trivial
for git, and an artefact that carries published claims and cannot be rebuilt has to be
kept somewhere durable. Retrained successors are reproducible from a config and stay
ignored by default.
