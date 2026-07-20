# Compute profiles

Developing on a laptop and training on a GPU box, from one branch.

---

## Why not a branch per machine

The tempting arrangement is a `cpu` branch for the laptop and a `gpu` branch for the
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
qfme train --config experiments/indic.yaml --profile configs/mac.yaml
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
is recorded alongside the artefacts for that reason. A laptop profile trains a worse model
on purpose.

## The settings

| Setting | Meaning |
|---|---|
| `device` | `auto` resolves CUDA, then Apple Metal, then CPU. Name one to pin it — most usefully `cpu` on a GPU machine, to tell a real bug from a device-specific one. |
| `precision` | `fp32`, or `bf16` for mixed precision. |
| `batch_size` | Pairs per step, and the negative count. See above. |
| `gradient_checkpoint_chunk` | Gradient-caching chunk. `0` disables. Peak memory follows the chunk rather than the batch, which is what makes a large batch fit. Mathematically exact either way. |
| `workers` | Data loading processes. `0` is right on a laptop and usually wrong on a training box. |

**Only bf16 is offered, never fp16.** bf16 has the same exponent range as fp32, so it
needs no loss scaling — the `GradScaler` machinery fp16 requires exists to stop small
gradients flushing to zero, and bf16 does not have that failure mode. It spends mantissa
bits instead, which training tolerates well. Accepting `fp16` here would train quietly and
badly, so it is rejected at config load.

**bf16 is ignored on Apple Metal**, with a warning. MPS autocast support has been
incomplete across torch versions, and silently training in a precision nobody asked for is
worse than declining the request loudly.

## Validating a profile on the wrong machine

`device: cuda` is accepted on a machine with no CUDA. This is deliberate and it is what
makes the whole approach work — a GPU profile has to be writable, diffable, testable in CI
and committable from the laptop. Devices are checked by shape, not against what happens to
be present; an unavailable device fails later, at the point of use, where the error can be
specific.

The profiles in `configs/` are loaded by the test suite rather than assumed correct. A typo
in `gpu.yaml` would otherwise surface only on the training box, which is the machine
furthest from a debugger.

## What remains unverified here

This laptop has no CUDA, so **the CUDA paths are never executed by local testing**. bf16
autocast is exercised genuinely on CPU — the operations really do run in bfloat16, and the
loss really does still fall — but CUDA kernel selection, and the speed and memory claims
that motivate bf16 in the first place, are not reachable from here. They stay unverified
until a run happens on the GPU box.

Device-specific bugs will therefore surface first on Windows. That is a property of owning
one GPU, not of this design; a branch would not have helped.

## Notes on WSL2

The Windows filesystem is reachable at `/mnt/c`, and crossing that boundary is slow enough
to dominate a training run. Keep the corpus on the Linux side, under `~`. Raising `workers`
will not save a pipeline that is reading across the mount.
