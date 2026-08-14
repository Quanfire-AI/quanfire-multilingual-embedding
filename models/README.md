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

Two adapters are git-tracked: `indic-v1` above — the original two-language (hi/ta) adapter,
kept because it predates `qfme adapt`, carries the first published numbers and is the
integration-test fixture — and `prod-a70s30-fr`, the current production adapter, shipped so it
can be served without a GPU rebuild and trained only on commercially-clean data. `prod-a70s30`
(the Indic-measured reference, below) is deliberately **not** tracked: its sentence side was
Samanantar (CC BY-NC) + opus-100 (unknown licence), so shipping those weights would carry a
NonCommercial restriction; its numbers are documented here but the binary is not distributed.
Everything else is produced by `qfme adapt` from a committed configuration, so it is
reproducible and stays git-ignored — but the names recur throughout
[ROADMAP.md](../ROADMAP.md) and the reports, so they are worth stating once:

| Directory | What it is | Status |
|---|---|---|
| `indic-v1` | Two-language (hi/ta) LoRA, the original | Tracked; historical baseline & test fixture |
| `indic-aligned-v1` | Ten-language adapter trained on cross-lingual aligned pairs | Superseded by v2 |
| `indic-aligned-v2` (a.k.a. `indic-aligned-multi-np`) | The refined ten-language adapter | Previous canonical — superseded by `prod-a70s30`, which beats it on all three published instruments |
| `indic-aligned-c250k` | Control: same 250k pairs, no hard negatives | Ablation reference |
| `indic-aligned-hn` | Same pairs plus mined hard negatives | Ablation; a trade-off, not promoted |
| `indic-aligned-hn-gated` | Same pairs plus hard negatives mined with the `--positive-margin` guard (false negatives removed) | Ablation; the gated win — recovers in-domain *and* keeps FLORES |
| `samanantar-proof` | Sentence-scale Samanantar en↔indic bitext only (240k, 8 langs) | Proof; closes FLORES (0.9896, beats base) but gives back in-domain |
| `indic-aligned-mix` | ~50/50 blend of article pairs and Samanantar sentence pairs (490k) | Proof-scale both-at-once — closes FLORES *and* holds in-domain; superseded by the production sweep |
| `prod-a30s70` / `prod-a50s50` / `prod-a70s30` | Production ratio sweep: article : sentence at 30:70 / 50:50 / 70:30, 1.0M pairs, all 10 langs (Samanantar + itihasa sa + opus-100 ur) | Sweep to tune the blend ratio at scale |
| `prod-a70s30` | The 70:30 winner of that sweep | **Indic-measured reference** — beats v2 on *all three* published Indic instruments (FLORES 0.9805 vs 0.9609, in-domain 0.9029 vs 0.8964, hi-pivot r@10 0.8914 vs 0.8852); **not distributed** (its sentence side was CC BY-NC + unknown-licence data); superseded as the serving artefact by the clean-provenance `prod-a70s30-fr` |
| `prod-a70s30-e2` / `prod-a70s30-e3` | `prod-a70s30` retrained at 2 / 3 epochs (single-variable epoch sweep) | Confirms one epoch is the stopping point — more epochs lift in-domain but regress held-out FLORES (0.9805 → 0.9701 → 0.9673) |
| `prod-a70s30-fr` | The 70:30 blend rebuilt on **commercially-clean** data (BPCC-Mined CC0 sentence side + Tatoeba CC BY en↔fr, over the CC BY-SA Wikipedia article side) | **Canonical / shipped production adapter** (git-tracked) — on the held-out FLORES-200 global baseline (15 world languages, scored on CUDA) all-pairs recall **0.9762** tops both `prod-a70s30` (0.9756) and the 0.9268 base, and **recovers French to 0.990** with no language regressed vs base (it trades within noise per-language rather than strictly dominating); Indic neutral within sampling noise and still clears v2 on all three; serve with `qfme serve --adapter models/prod-a70s30-fr` |
| `embed-legal-en` | Legal **domain specialist** — English Supreme Court judgment retriever, LoRA over e5-small, internal run `legal-indic-e1` | **Published** (git-tracked; `quanfire-ai/embed-legal-en`) — trained **only on statutory public-domain judgment text** (Copyright Act §52(1)(q), reporter headnotes excised), so the weights are **Apache-2.0** with no share-alike floor. In-distribution Recall@1 0.309→**0.545** (+76%, CIs disjoint). A domain-transfer test to a different English legal register (statutory adjacency) came back **flat** (0.036→0.036) — the gain is judgment-specific, and the model ships scoped to English judgments, with that flat row on its card. Full working in Appendix K of the handbook. |
| `embed-statute-en` | Statute **domain specialist** — English central-statutory (bare-Act) retriever, LoRA over e5-small, internal run `statute-en-e2`. The sibling that covers the register `embed-legal-en` measured itself *flat* on | **Published** (git-tracked; `quanfire-ai/embed-statute-en`) — trained on 858 Central Acts (Zenodo 5088102, **CC-BY-4.0**) mined into section-level pairs + adapter-mined hard negatives; §52(1)(q)(ii) bare-Act text is train-safe for a **non-reconstructive** embedder, so weights are **Apache-2.0** and the corpus is not redistributed. In-distribution Recall@1 0.182→**0.269** (+48%, CIs disjoint); on the un-gameable low-overlap slice 0.077→**0.177** (+131%). Statute-specific — judgments are `embed-legal-en`; state law/rules/notifications unmeasured. |

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
**beats v2 on all three published instruments at once** — FLORES **0.9805** (v2 0.9609, near
base), in-domain **0.9029** (v2 0.8964), and hi-pivot mixed-pool r@10 **0.8914** (v2 0.8852) —
so it is promoted over v2 as the canonical, git-tracked production adapter. (The hi-pivot number was re-baselined last, with the
byte-identical protocol that reproduces the published base 0.7495 and v2 0.8852;
`reports/prod-pivot-verdict.json`.) The dedicated sa/ur sources did *not* lift sa/ur on FLORES
(transfer already covered them); that null result is recorded, not hidden. Full working is in
the ROADMAP entries dated 27–30 July 2026. The other published numbers are now re-baselined on
`prod-a70s30`, and the last open question — whether more than one epoch helps — is answered: a
2- and 3-epoch sweep (`prod-a70s30-e{2,3}`, mirrored here) lifts the in-domain fit but
regresses the held-out FLORES benchmark monotonically (0.9805 → 0.9701 → 0.9673), so **one
epoch stays canonical** (`reports/prod-longer-verdict.json`). When one of these is the canonical
serving artefact, copy it into place exactly as below.

**Reaching past the ten languages — and the measurement lesson that came with it.** Scored on a
held-out FLORES-200 slice of fifteen major world languages, the Indic-only `prod-a70s30` shows
*positive transfer*, not forgetting: all-pairs cross-lingual recall@1 **0.9268** (base E5) →
**0.9756**, up on all fifteen. An early run of that baseline reported a lone French collapse to
0.788 — but that was a **measurement artifact of this Mac's Apple MPS backend**, which is
non-deterministic on this scorer (identical runs gave base E5 anywhere from 0.814 to 0.927).
Re-scored on the box's **CUDA** the same script is byte-for-byte reproducible across runs, and
there is no French hole (0.991, in line with every language). Rule, now enforced by a guard in
`scratch_global_baseline.py`: score the global baseline on CUDA, never MPS. Folding ~30k en↔fr
pairs into the blend (`prod-a70s30-fr`) then lifts the global all-pairs recall above both the
base and `prod-a70s30` and recovers French, at no Indic cost — the three published Indic
instruments move within sampling noise and still clear v2. So `prod-a70s30-fr` is promoted to
the canonical serving artefact.

**The clean-provenance retrain (Door A, August 2026).** The blend above was first assembled
from Samanantar (CC BY-NC) on the sentence side and opus-100 (unknown licence) for the French
fold — fine for measurement, but not weights Quanfire could redistribute for commercial use.
So `prod-a70s30-fr` was **re-sourced, not re-engineered**: the identical recipe was rebuilt on
**BPCC-Mined (CC0)** — which redistributes the same Samanantar sentences under a clean licence —
and **Tatoeba (CC BY)** for en↔fr, over the unchanged CC BY-SA Wikipedia article side. Re-scored
on CUDA the clean weights hold the win: global all-pairs **0.9762** (> `prod-a70s30` 0.9756,
> base 0.9268), French **0.990** recovered with no language regressed vs base, and Indic neutral
within noise (in-domain −0.0035 ≈ one standard error over n≈8.3k, hi-pivot flat, FLORES
non-Hindi −0.0020) while still beating v2 on all three. The shipped weights therefore train on
**CC0 + Apache-2.0 + CC BY + CC BY-SA only** and are released under CC BY-SA 4.0 — commercially
usable and redistributable. `prod-a70s30` itself was *not* re-sourced; it is kept only as the
Indic-measurement reference and its (NC-trained) weights are no longer distributed. Full working
is in the ROADMAP entries dated 31 July – 5 August 2026.

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
in git. `models/indic-v1/`, `models/prod-a70s30-fr/`, `models/embed-legal-en/` and this
file are the documented exceptions: a few MB is trivial for git, and an artefact that
carries published claims (or ships as the serving binary) and cannot be cheaply rebuilt
has to be kept somewhere durable. `embed-legal-en` qualifies on both counts and on
provenance — its statutory-public-domain training data makes the weights freely
redistributable (Apache-2.0).
`prod-a70s30-fr` is additionally the point where the shipped weights are held to clean
provenance. Retrained successors and the reference-only `prod-a70s30` stay ignored by default.
