# Compute profiles

Developing on a laptop and training on a GPU box, from one branch.

---

## Why not a branch per machine

The tempting arrangement is one branch for the development machine and another for the
training box. It does not survive contact with use.

Branches diverge. Every fix lands twice, forever, and the divergence is silent — you lose
the one guarantee worth having, which is that *the code tested on the laptop is the code
that trained on the GPU*. When a training run then misbehaves, there is no way to tell a
model problem from branch skew. The conflicts also concentrate in the worst place: the
files that differ per machine are the training and encoder modules, which are the files
carrying the real work.

The difference between the two machines is a **runtime** difference. It belongs in
configuration, not in source control.

## The split

A configuration has two halves that behave differently:

| Half | Sections | Changes the result? | Differs per machine? |
|---|---|---|---|
| The experiment | `corpus`, `tokenizer`, `embedding`, `evaluation` | yes | no |
| The machine | `compute` | no, with one exception | yes |

A profile supplies only the second. The same experiment file then runs unchanged on both
boxes, and two runs stay comparable.

```bash
qfme train --config experiments/indic.yaml --profile configs/cpu.yaml
qfme train --config experiments/indic.yaml --profile configs/gpu.yaml
```

Precedence, lowest to highest: defaults, the config file, the profile, `QFME_`
environment variables, then `--set`. So an out-of-memory failure is answered without
editing anything:

```bash
qfme train --config experiments/indic.yaml --profile configs/gpu.yaml \
    --set compute.batch_size=128
```

**The exception.** `batch_size` is machine-shaped — memory decides it — but it also
changes the result, because in contrastive training it sets how many negatives each query
is contrasted against. A batch of 16 asks the model to pick the right passage from 16
candidates; a batch of 256 makes it pick from 256, which is a materially harder and more
useful task. It is therefore the one profile setting that is *not* result-neutral, and it
is recorded alongside the artefacts for that reason. The `cpu` profile trains a worse
model on purpose.

## The settings

| Setting | Meaning |
|---|---|
| `device` | `auto` resolves CUDA, then Apple Metal, then CPU. Name one to pin it — most usefully `cpu` on a GPU machine, to tell a real bug from a device-specific one. |
| `precision` | `fp32`, or `bf16` for mixed precision. |
| `batch_size` | Pairs per step, and the negative count. See above. |
| `gradient_checkpoint_chunk` | Gradient-caching chunk. `0` disables. Peak memory follows the chunk rather than the batch, which is what makes a large batch fit. Mathematically exact either way. |

**Only bf16 is offered, never fp16.** bf16 has the same exponent range as fp32, so it
needs no loss scaling — the `GradScaler` machinery fp16 requires exists to stop small
gradients flushing to zero, and bf16 does not have that failure mode. It spends mantissa
bits instead, which training tolerates well. Accepting `fp16` here would train quietly and
badly, so it is rejected at config load.

**bf16 is ignored on Apple Metal**, with a warning. MPS autocast support has been
incomplete across torch versions, and silently training in a precision nobody asked for is
worse than declining the request loudly.

## The profiles

`configs/cpu.yaml` encodes "no CUDA-class GPU, small memory budget" — as true of a Windows
or Linux box without one as of a Mac. Its `device` is `auto` rather than `cpu` despite the
name, so that Apple Silicon still picks up Metal; the file is named for the constraint, not
for the device that gets chosen. Pin `cpu` explicitly for a reproducibility run, since GPU
reductions are not bit-deterministic.

`configs/gpu.yaml` is sized for a single 16GB NVIDIA card. The numbers are a starting
point, not a measurement — VRAM use depends on model width, depth and sequence length,
none of which the file knows.

## Validating a profile on the wrong machine

`device: cuda` is accepted on a machine with no CUDA. This is deliberate and it is what
makes the whole approach work — a GPU profile has to be writable, diffable, testable in CI
and committable from the laptop. Devices are checked by shape, not against what happens to
be present; an unavailable device fails later, at the point of use, where the error can be
specific.

The profiles in `configs/` are loaded by the test suite rather than assumed correct. A typo
in `gpu.yaml` would otherwise surface only on the training box, which is the machine
furthest from a debugger.

## What the settings actually buy

Measured on an RTX 4070 Ti SUPER under WSL2 — batch 256, a 5.3M-parameter encoder, 4,000
mined Hindi pairs, all four combinations of the same experiment:

| | no caching | `gradient_checkpoint_chunk: 32` |
|---|---:|---:|
| `fp32` | 4.89 GB / 4.3s | 0.40 GB / 4.7s |
| `bf16` | 2.99 GB / 2.7s | **0.29 GB** / 4.7s |

**Gradient caching is what buys the memory** — 12.2× on its own, against 1.6× for bf16;
16.9× together. If you can set only one thing on a card that is running out of room, set
the chunk.

**bf16 turned out to buy speed**, 1.6× faster than fp32, which is not why it was chosen.
Caching costs 1.09× wall clock on fp32 and 1.74× on bf16 — it duplicates the forward pass,
and bf16 makes that pass cheaper, so the relative cost is higher where the baseline is
faster.

Final losses spanned **0.51%** across all four cells. That is the "mathematically exact"
claim holding on real hardware rather than in a unit test.

## What is still unverified

**Everything at realistic model size.** Those numbers come from a 5.3M-parameter encoder
using 0.29 GB of a 16 GB card. Activation memory scales with width, depth and sequence
length, so they say nothing about where the ceiling sits for a 100M+ model — which is the
size that matters. Re-measure before trusting `batch_size: 256` as a maximum; at this
model size it is far below one.

**Development still happens without CUDA**, so device-specific bugs will surface first on
the training box. That is a property of owning one GPU, not of this design.

## Notes on WSL2

The Windows filesystem is reachable at `/mnt/c`, and crossing that boundary is slow enough
to dominate a training run. Keep the corpus on the Linux side, under `~`.

There is deliberately no `workers` setting. Training is single-process, so one would read
as a tuning knob and do nothing.
