# Running the from-scratch neural path on the GPU box

The full loop — `extract → mine-pairs → pretrain → finetune → evaluate` —
on the WSL2 RTX 4070 Ti SUPER (16 GB), driven over SSH from the Mac.
Authored and unit-tested on the Mac; **run** here because pretraining and
contrastive fine-tuning want the GPU.

Two runs are described:

- **A — Manipuri (`mni`), the fast full-chain proof.** A small real
  Wikipedia. The entire chain finishes in well under an hour and proves
  the plumbing *and* that retrieval moves on real structured data.
- **B — Hindi (`hi`), the real-scale run.** Hours per pretrain epoch;
  checkpointed each epoch so it survives a dropped connection or a turn on
  the shared box.

Do **A** first. Only start **B** once A's `evaluate` shows a positive
delta.

---

## ⚠️ Box guardrails — read before touching anything

1. **Never `uv sync` and never `uv run` on this box.** Both re-resolve the
   environment against the lockfile and will *uninstall* the hand-built
   CUDA (cu130) PyTorch, replacing it with whatever the lock pins. Call the
   venv binary directly — `.venv/bin/qfme` — or activate the venv and use
   `qfme`. This runbook never uses `uv`.
2. **The code change here adds no dependencies.** Updating the box is a
   pure `git pull`. If a future change *does* add a dependency, install
   just that one wheel by hand; do not sync.
3. **Never kill a GPU process you did not start.** Check `nvidia-smi`
   first; if the card is busy with someone else's work, wait.
4. **Keep data on the Linux side (`~/...`), never `/mnt/c`.** The WSL2 →
   Windows filesystem boundary is the real throughput tax here.
5. **Long runs go in `tmux`** so an SSH drop does not take the job with it.
   Checkpointing (below) is the second safety net, not the first.

Connect:

```bash
ssh arnab@192.168.1.2
tmux new -s qfme        # or: tmux attach -t qfme
cd ~/quanfire-multilingual-embedding
```

---

## 0. Update the box and confirm the environment (read-only)

```bash
# Pure git — NO uv sync / uv run.
git fetch origin
git checkout feat/hard-negatives-and-data-policy
git pull --ff-only

# Confirm torch still sees the GPU and the extras are present.
# If this prints False or errors, STOP — do not "fix" it with uv.
.venv/bin/python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
import sentencepiece, multilingual_embedding  # neural + core import cleanly
print("imports OK")
PY

.venv/bin/qfme --version
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
```

Everything below calls `.venv/bin/qfme`. If you'd rather type `qfme`, run
`source .venv/bin/activate` once and drop the prefix — just never `uv run`.

---

## A. Manipuri — the fast full-chain proof

### A1. Get the dump and extract a corpus

```bash
mkdir -p data/dumps data/wikipedia data/pairs reports ckpts

# ~small; public data (dumps.wikimedia.org).
wget -O data/dumps/mniwiki.xml.bz2 \
  https://dumps.wikimedia.org/mniwiki/latest/mniwiki-latest-pages-articles.xml.bz2

.venv/bin/qfme extract \
  --dump data/dumps/mniwiki.xml.bz2 \
  --output data/wikipedia/mni.jsonl.gz \
  --language mni

.venv/bin/qfme validate --source data/wikipedia/mni.jsonl.gz
```

### A2. Mine training pairs from article structure

```bash
.venv/bin/qfme mine-pairs \
  --source data/wikipedia/mni.jsonl.gz \
  --output data/pairs/mni.jsonl.gz \
  --max-overlap 0.9 \
  --report reports/mni-pairs.json
```

`--max-overlap 0.9` drops `title_lead` pairs a string matcher solves for
free. The report prints how many of each kind were mined and their mean
overlap.

### A3. Pretrain (stage one) — checkpointed each epoch

```bash
.venv/bin/qfme pretrain \
  --config examples/from-scratch/experiment.yaml \
  --profile configs/gpu.yaml \
  --name mni-scratch \
  --set corpus.source=data/wikipedia/mni.jsonl.gz \
  --set tokenizer.vocab_size=8000 \
  --checkpoint-dir ckpts/mni-pretrain
```

Watch VRAM on the first epoch (`nvidia-smi` in another pane). If it's
comfortable, nothing to do; if tight, lower `--set compute.batch_size` or
raise `--set compute.gradient_checkpoint_chunk` via the profile.

**Resume** (after any interruption — same command plus):

```bash
  --resume-from ckpts/mni-pretrain
```

A resumed run reproduces the uninterrupted one bit for bit.

### A4. Fine-tune (stage two) — checkpointed each epoch

```bash
.venv/bin/qfme finetune \
  --config examples/from-scratch/experiment.yaml \
  --profile configs/gpu.yaml \
  --name mni-finetuned \
  --source artifacts/mni-scratch \
  --pairs data/pairs/mni.jsonl.gz \
  --data-provenance public \
  --checkpoint-dir ckpts/mni-finetune
```

`--name mni-finetuned` writes to `artifacts/mni-finetuned`, leaving the
pretrained `artifacts/mni-scratch` untouched. The command prints
**DID FINE-TUNING HELP?** with recall@1 before vs after. Exit code is
non-zero if the fine-tune did *not* beat the pretrained baseline — a real
outcome, not an error.

**Resume:** add `--resume-from ckpts/mni-finetune`.

### A5. Evaluate the fine-tuned encoder by retrieval

```bash
.venv/bin/qfme evaluate \
  --experiment artifacts/mni-finetuned \
  --pairs data/pairs/mni.jsonl.gz \
  --eval-pairs 2000 \
  --output reports/mni-finetuned-eval.json
```

`evaluate` routes on directory shape: it sees `encoder/` and scores by
retrieval. For an independent baseline number, point it at the pretrained
directory too:

```bash
.venv/bin/qfme evaluate \
  --experiment artifacts/mni-scratch \
  --pairs data/pairs/mni.jsonl.gz \
  --eval-pairs 2000 \
  --output reports/mni-scratch-eval.json
```

**Success looks like:** `mni-finetuned-eval.json` recall@1 and MRR higher
than `mni-scratch-eval.json`. That is the loop, proven on real data.

---

## B. Hindi — the real-scale run

Same shape, larger corpus, longer clock. The experiment.yaml defaults are
already Hindi, so no `--set` overrides are needed for corpus/vocab.

### B1–B2. Extract and mine (slow; the walkthrough measured Hindi mining at
~25 min on a laptop CPU — faster here, still not instant)

```bash
wget -O data/dumps/hiwiki.xml.bz2 \
  https://dumps.wikimedia.org/hiwiki/latest/hiwiki-latest-pages-articles.xml.bz2

.venv/bin/qfme extract \
  --dump data/dumps/hiwiki.xml.bz2 \
  --output data/wikipedia/hi.jsonl.gz \
  --language hi

.venv/bin/qfme validate --source data/wikipedia/hi.jsonl.gz

.venv/bin/qfme mine-pairs \
  --source data/wikipedia/hi.jsonl.gz \
  --output data/pairs/hi.jsonl.gz \
  --max-overlap 0.9 \
  --report reports/hi-pairs.json
```

### B3. Pretrain — run it detached, checkpointing each epoch

Inside `tmux` a plain run is already disconnect-safe. Either way:

```bash
.venv/bin/qfme pretrain \
  --config examples/from-scratch/experiment.yaml \
  --profile configs/gpu.yaml \
  --checkpoint-dir ckpts/hi-pretrain \
  2>&1 | tee reports/hi-pretrain.log
```

If an epoch gets cut short, resume with the identical command plus
`--resume-from ckpts/hi-pretrain`. To deliberately stop after an epoch
(e.g. to yield the card), add `--stop-after-epoch N`; it exits zero
because a partial run is not a failure, and the next `--resume-from`
continues the schedule unbroken.

### B4. Fine-tune

```bash
.venv/bin/qfme finetune \
  --config examples/from-scratch/experiment.yaml \
  --profile configs/gpu.yaml \
  --name hi-finetuned \
  --checkpoint-dir ckpts/hi-finetune \
  2>&1 | tee reports/hi-finetune.log
```

(`--source`, `--pairs`, `--data-provenance` all come from experiment.yaml
here, so they need not be repeated. Add `--resume-from ckpts/hi-finetune`
to continue.)

### B5. Evaluate

```bash
.venv/bin/qfme evaluate --experiment artifacts/hi-finetuned \
  --pairs data/pairs/hi.jsonl.gz --eval-pairs 2000 \
  --output reports/hi-finetuned-eval.json

.venv/bin/qfme evaluate --experiment artifacts/hi-scratch \
  --pairs data/pairs/hi.jsonl.gz --eval-pairs 2000 \
  --output reports/hi-scratch-eval.json
```

---

## Bringing results back to the Mac

From the Mac (not the box):

```bash
scp 'arnab@192.168.1.2:~/quanfire-multilingual-embedding/reports/*eval*.json' /tmp/
```

The fine-tuned encoder itself lives under `artifacts/<name>/` on the box
(`encoder/`, `tokenizer/`, `config.yaml`) and is served in place with
`qfme search --experiment artifacts/hi-finetuned --query "..."` — no
`--source`, no matrix; `from_directory` detects the contextual encoder.

## Optional — hard negatives

`finetune` already contrasts each query against every other passage in the
batch (in-batch negatives), which is enough for a first result. To add
*mined* hard negatives, run `qfme mine-negatives` against a saved model to
attach them to the pair file before B4. Skip it for the proof run.
