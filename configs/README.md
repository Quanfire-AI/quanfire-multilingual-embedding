# configs

> Compute profiles — the half of a configuration that the machine dictates rather than
> the experiment.

## Purpose

A run is described by two kinds of setting, and they behave differently. One kind
determines the **result**: which corpus, how the tokenizer is trained, the embedding
geometry, how the outcome is measured. The other kind is whatever the **machine**
imposes: which device, what numeric precision, how much fits in memory at once, how many
loader processes are worth starting.

Only the first kind belongs to the experiment. The second is a property of the box the
job happens to land on, and this directory holds it — one file per machine shape, merged
over an experiment with `--profile`:

```bash
qfme train --config experiments/indic.yaml --profile configs/cpu.yaml
qfme train --config experiments/indic.yaml --profile configs/gpu.yaml
```

The experiment file is unchanged between those two commands, which is the entire point:
the two runs remain comparable, and the artefacts they produce can be attributed to the
experiment rather than to the hardware.

Nothing else lives here. Experiment configurations are yours to keep wherever suits;
`configs/` carries profiles only.

## Contents

| File | Encodes |
|---|---|
| `cpu.yaml` | Development machine — no CUDA-class GPU, small memory budget |
| `gpu.yaml` | A single 16GB NVIDIA card, sized for a real training run |

Both files set the same four keys under a `compute:` section and nothing else. A profile
that reached into `corpus` or `embedding` would defeat its own purpose, since the result
would then depend on which machine ran it.

What they encode, side by side:

| | `cpu.yaml` | `gpu.yaml` |
|---|---|---|
| `device` | `auto` | `auto` |
| `precision` | `fp32` | `bf16` |
| `batch_size` | 16 | 256 |
| `gradient_checkpoint_chunk` | 0 (off) | 32 |

## How the merge works

`--profile` is deep-merged over the loaded configuration, so a profile replaces the keys
it names and leaves every other key intact. The full precedence chain, lowest to highest,
is: dataclass defaults, the config file, the profile, `QFME_` environment variables, then
`--set`. An out-of-memory failure is therefore answered without editing a tracked file:

```bash
qfme train --config experiments/indic.yaml --profile configs/gpu.yaml \
    --set compute.batch_size=128
```

`ComputeConfig` in `../src/multilingual_embedding/config/base.py` is the source of truth
for the field names, defaults and validation rules.

## The settings

| Setting | Meaning |
|---|---|
| `device` | `auto` resolves CUDA, then Apple Metal, then CPU. Name one to pin it. |
| `precision` | `fp32`, or `bf16` for mixed precision. `fp16` is rejected at config load. |
| `batch_size` | Pairs per step — and, in contrastive training, the negative count. |
| `gradient_checkpoint_chunk` | Gradient-caching chunk size. `0` disables it. Peak memory follows the chunk rather than the batch, which is what lets a batch fit that otherwise would not. The gradients are identical either way. |

### What `gradient_checkpoint_chunk` is worth, measured

On an RTX 4070 Ti SUPER at batch 256 with a 5.3M-parameter model:

| Configuration | Peak VRAM | Reduction |
|---|---:|---:|
| fp32, no caching | 4.89 GB | — |
| bf16, no caching | 3.06 GB | 1.6× |
| fp32, caching at chunk 32 | 0.40 GB | **12.2×** |
| bf16 + caching | 0.29 GB | **16.9×** |

Final loss across all four spans 0.51%, which is the claim that matters: caching is exact,
not an approximation. Gradient caching is the setting that decides whether a batch fits;
bf16 is a smaller, additional win. That model was a toy and 0.29 GB of 16 is not a ceiling
— measure at your real model size before treating batch 256 as a maximum.

## Why this exists, rather than a branch per machine

The obvious alternative is one branch for the development machine and another for the
training box. It fails in a specific and expensive way.

Branches diverge. Every fix lands twice, forever, and the divergence is silent. What is
lost is the one guarantee worth having — that *the code tested on the development machine
is the code that trained on the GPU*. Once that is gone, a training run that misbehaves
cannot be diagnosed, because a model problem and branch skew look identical from the
outside. The conflicts also concentrate in the worst possible files: the ones that differ
per machine are the training and encoder modules, which is where the real work is.

The difference between the two machines is a runtime difference. It belongs in
configuration, not in source control.

## The one wrinkle: `batch_size` is not result-neutral

Every other setting here changes only how much fits and how fast it runs. `batch_size`
does not. It is machine-shaped — memory decides what is reachable — but in contrastive
training it also sets how many negatives each query is contrasted against. A batch of 16
asks the model to pick the right passage out of 16 candidates; a batch of 256 makes it
pick out of 256, which is a materially harder and more useful task.

So the two profiles here do not train the same model. `cpu.yaml` trains a worse one, on
purpose, because 16 is what the machine can hold. Use it to confirm a pipeline is wired
up correctly, never to judge model quality. This is also why `batch_size` is recorded
alongside the artefacts while the rest of the profile is not.

## Why `cpu.yaml` says `device: auto`

Despite the filename, `cpu.yaml` does not pin the device. On Apple Silicon `auto`
resolves Metal, which is a real speedup and worth taking; on any other machine without a
CUDA-class GPU it lands on CPU anyway. The file is named for the constraint it encodes —
no CUDA-class GPU, small memory budget — and not for the device that ends up being
chosen. That constraint is as true of a Windows or Linux box without a discrete card as
it is of a Mac.

Pin `cpu` explicitly for a reproducibility run, since GPU reductions are not
bit-deterministic.

The values in `cpu.yaml` happen to match the `ComputeConfig` defaults exactly. They are
written out regardless, so that reading the file tells you what the run did without
having to hold the defaults in your head.

## Writing a profile for a machine you are not on

`device: cuda` is accepted on a machine with no CUDA, deliberately. Devices are validated
by shape rather than against what is present, because a GPU profile has to be writable,
diffable, testable in CI and committable from a laptop that cannot run it. An unavailable
device fails later, at the point of use, where the error can be specific about what was
missing.

The files in this directory are loaded by the test suite rather than assumed correct. A
typo in `gpu.yaml` would otherwise surface only on the training box, which is the machine
furthest from a debugger.

## What profiles do *not* cover yet

`--profile` is wired into the config-driven subcommands — `train`, `evaluate` and the
Python `TrainingPipeline`. `scripts/adapt_pretrained.py`, which is where published
checkpoints are adapted, takes `--precision`, `--batch-size` and the rest as flags
directly, because it has no config object to merge a profile into. The defaults there
(`bf16`, batch 64) are GPU-shaped, so a run on a machine without one needs those flags set
by hand rather than a profile named. Closing that gap arrives with `qfme adapt`, tracked in
[`ROADMAP.md`](../ROADMAP.md).

## Further reading

[`docs/compute-profiles.md`](../docs/compute-profiles.md) is the full treatment: the
reasoning behind bf16 over fp16, what remains unverified without CUDA hardware, and the
WSL2 filesystem trap that dominates a training run if the corpus sits under `/mnt/c`.

There is deliberately no `workers` setting. Training is single-process, so one would
read as a tuning knob and do nothing.
