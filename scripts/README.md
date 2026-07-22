# scripts

> Three programs that are not part of the library: the adaptation experiment, an end-to-end verifier for hardware the author does not have, and a one-off diagnostic.

Everything here is run with `python scripts/<name>.py`. None of it is importable API, none
of it is installed by the wheel, and none of it is covered by the test suite — which is why
each one prints what it measured rather than asserting it.

| Script | Purpose | Runs where |
|---|---|---|
| `adapt_pretrained.py` | **The experiment this project was built for.** Adapt a published encoder to a domain and measure whether it helped | GPU box |
| `verify_e2e.py` | Run the whole pipeline against real dumps and print one compact report | GPU box |
| `diagnose_audit.py` | Reproduce one WSL-specific `audit` crash with the full import chain | wherever it fails |

---

## `adapt_pretrained.py`

### What it does

Three steps, and the middle one is the least interesting:

1. **Score a published checkpoint on held-out pairs.** This is the number to beat, and the
   only honest baseline. Beating chance proves nothing; beating an untrained model proves
   nothing.
2. Fine-tune it on the domain corpus with LoRA.
3. **Score it again on the same held-out pairs.**

If step 3 does not beat step 1, the adaptation did not work — a result worth having rather
than a failure to hide. Fine-tuning a well-pretrained model on a narrow corpus can easily
make it worse.

Nothing here trains the base weights. LoRA leaves them frozen, so the comparison is between
one model and *itself plus a small adapter*, rather than between two models differing in
unknown ways.

### Input and output

| | |
|---|---|
| **In** | a checkpoint name or path; a mined pair file (`qfme mine-pairs` output); optionally a second pair file for the evaluation set |
| **Out** | a JSON report (`--output`), a saved 3.4 MB adapter (`--save-adapter`), and a printed comparison |

The JSON report carries `before`/`after` blocks with `recall_at_1`, `recall_at_10`, `mrr`,
`ndcg_at_10`, each with hit counts and Wilson 95% intervals, broken down `by_kind`, by
language, and `by_overlap` band. Plus `candidates`, `random_recall_at_1`,
`dropped_duplicate_positives`, the loss curve, and the actual `train_examples` /
`eval_examples` used. [`docs/reading-results.md`](../docs/reading-results.md) is how to read
it without fooling yourself.

### Why the experiment is *declared*

The same three steps answer several different questions depending on what is held fixed
between training and evaluation, and the questions are not interchangeable:

| `--adaptation` | held fixed | varied | the question |
|---|---|---|---|
| `in-distribution` | everything | — | how much adaptation helps where it trained |
| `task` | corpus, language | pair kind | did it learn retrieval, or the mining scheme |
| `language` | corpus, pair kind | language | does it cross scripts |
| `domain` | pair kind, language | the pair file | does it survive contact with your own text |
| `task+language`, `task+domain` | | two facets | explicitly two changes at once |

**The declaration is checked against what the filters actually do, and a run whose label and
data disagree is refused before it starts.** A report labelled `task` that quietly held the
kinds identical is worse than no report, because the label is what gets quoted six months
later. The check is two-sided: a `task` run whose languages *also* differ is not a task
result, it is two changes at once with one name.

The script prints a facet table before it trains — `kind`, `language`, `corpus`, each marked
`VARIES` or `fixed` — so the design of the run is visible rather than inferred from two
prose sentences.

Only `domain` licenses "this will help on our contracts". `in-distribution` is the weakest
claim, the easiest to overstate, and therefore the default: an unlabelled run must not be
able to sound like a transfer result.

### `--sample-pairs`, and why it exists

A facet filter runs *after* reservoir sampling. With the reservoir sized `--train-pairs +
--eval-pairs`, naming a kind that holds a sixth of the file yields a sixth of the sample, so
`--train-pairs` stops binding. Two runs naming different kinds then differ in **how much
data they saw** as well as in shape, and the comparison measures training-set size.

This actually happened: one run trained on ~25,000 `adjacent` pairs while its comparator
trained on ~7,000 `title_lead` pairs from the same 42,000 sample. Set `--sample-pairs`
several times `--train-pairs` so the cap binds for every kind. The script warns when the
filters leave fewer pairs than requested, and the report records the counts actually used
rather than the ones asked for.

### Usage

```bash
# in-distribution — the default, nothing varies
python scripts/adapt_pretrained.py \
    --checkpoint intfloat/multilingual-e5-small \
    --pairs verify-output/hi-pairs.jsonl.gz \
    --query-prefix "query: " --passage-prefix "passage: " \
    --output reports/hi-v1.json --save-adapter models/hi-v1

# task adaptation — train on one pair shape, score on another
python scripts/adapt_pretrained.py \
    --checkpoint intfloat/multilingual-e5-small \
    --pairs verify-output/hi-pairs.jsonl.gz \
    --query-prefix "query: " --passage-prefix "passage: " \
    --adaptation task \
    --train-kinds adjacent --eval-kinds heading_section \
    --sample-pairs 120000 --train-pairs 15000 \
    --eval-pairs-file verify-output/hi-eval.jsonl.gz \
    --output reports/shape-cross.json

# domain adaptation — train on Wikipedia, score on your own corpus
python scripts/adapt_pretrained.py \
    --checkpoint intfloat/multilingual-e5-small \
    --pairs verify-output/hi-pairs.jsonl.gz \
    --eval-pairs-file data/pairs/contracts.jsonl.gz \
    --adaptation domain \
    --output reports/domain-v1.json
```

### Flags worth knowing

| Flag | Default | Note |
|---|---|---|
| `--checkpoint` | required | HuggingFace name or local directory |
| `--pairs` | required | the training pair file |
| `--eval-pairs-file` | — | **pin the evaluation set** when comparing runs; without it the held-out split moves with the training filter |
| `--sample-pairs` | `train+eval` | raise this whenever a facet filter is set — see above |
| `--train-pairs` / `--eval-pairs` | 20000 / 2000 | |
| `--query-prefix` / `--passage-prefix` | `""` | `"query: "` / `"passage: "` for E5. **Omitting these on an E5 model silently degrades everything** |
| `--rank` / `--targets` | 16 / `query,value` | LoRA. Rank 32 is ~0.50% trainable, ~3.4 MB |
| `--precision` | `bf16` | fp16 is not offered; see `embedding/neural/README.md` |
| `--epochs` / `--batch-size` / `--learning-rate` | 1 / 64 / 1e-4 | one epoch makes the loss comparison meaningless — `measurable` records that |
| `--pooling` | `mean` | must match how the checkpoint was trained |
| `--save-adapter` | — | without it, a run produces a number and no model |
| `--adaptation` | `in-distribution` | checked against the filters |

### What it has produced

`models/indic-v1` — 40,000 mixed-kind Hindi and Tamil pairs, +38.0% recall@1 on
`heading_section` and +40.8% on `adjacent` from one 3.4 MB adapter. Plus the four controlled
runs that established the adaptation is **language-general and task-specific**. Both written
up in [`ROADMAP.md`](../ROADMAP.md).

---

## `verify_e2e.py`

```bash
python scripts/verify_e2e.py --dumps data/dumps
```

Runs extract → validate → mine-pairs → train → search against whatever dumps are present and
prints one compact report for pasting back.

It exists because of a real gap: **no CUDA path is exercised by any test**, since development
happens on a machine with no NVIDIA GPU. This is what gets run on the training box.

Every stage is timed and every claim is measured rather than asserted, so a stage that
silently does nothing shows up as a zero rather than as a tick. Failures are caught and
reported instead of aborting, because a report that stops at the first problem hides the
rest of them.

---

## `diagnose_audit.py`

```bash
python scripts/diagnose_audit.py --source data/corpora/hi.jsonl.gz
```

Written for one specific report: `qfme validate` raising `SystemError: attempting to create
PyCMethod with a METH_METHOD flag but no class` on WSL while passing on macOS. That comes
from a compiled extension rather than from this project's code, so the useful output is the
import chain and the installed versions rather than the line that happened to raise.

Reads a few hundred documents and writes nothing. Keep it until the WSL environment is
known-good; delete it after.

---

## Why these are scripts and not `qfme` subcommands

`qfme` now covers both trained paths: `stats`, `validate`, `extract`, `mine-pairs`, `train`,
`adapt`, `search`, `evaluate`.

`adapt_pretrained.py` used to be the exception, because it was where the experiment design
lived and that design was still changing weekly while these results were being produced.
Freezing it into a subcommand before the facet model settled would have meant a CLI contract
that had to break.

It settled, and the experiment moved into `multilingual_embedding.pipelines.adaptation`.
This script is now a thin front end over that pipeline — it builds an `ExperimentConfig`
from its flags and calls `AdaptationPipeline`, so there is one implementation rather than
two that drift. Every original flag still works, because every command line in `ROADMAP.md`
was written in this shape and those results should stay reproducible verbatim.

Prefer `qfme adapt` for anything new. It takes a `--config` and a `--profile`, which means
a GPU run is described by a file that can be committed and diffed rather than by a flag list
in someone's shell history:

```bash
qfme adapt --config examples/adaptation.yaml --profile configs/gpu.yaml
```

Only the from-scratch contextual path is still API-only; it is tracked in `ROADMAP.md`.

`verify_e2e.py` and `diagnose_audit.py` are diagnostics and will never be subcommands.
