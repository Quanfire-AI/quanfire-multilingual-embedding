# Installing `qfme` as a normal command

Two different jobs, often confused, and the right answer differs:

| You want to | Use |
|---|---|
| **Use** the tool — run `qfme` anywhere, on your own data | `uv tool install` |
| **Develop** the project — edit code, run the tests | `uv sync` in a clone |

Everything below was executed and its output pasted verbatim.

---

## Use it: a system-wide `qfme`

One line, if your SSH key has access to the repository — no clone needed:

```bash
uv tool install 'git+ssh://git@github.com/quanfire-ai/quanfire-multilingual-embedding[neural,wikipedia]'
```

```
Installed 1 executable: qfme
```

From a clone you already have, equivalently:

```bash
cd quanfire-multilingual-embedding
uv tool install '.[neural,wikipedia]'
```

The difference matters later, not now: the git install can be updated with
`uv tool upgrade`, the local one cannot. See
[giving it to someone else](#giving-it-to-someone-else).

That is the whole install. No virtual environment to activate, no `PATH` to edit, nothing
to remember. From any directory:

```bash
$ cd /tmp
$ which qfme
/Users/you/.local/bin/qfme
$ qfme --version
qfme 0.5.0
```

### What it actually did

`uv tool install` gives the tool its own private environment and links one shim onto your
`PATH`:

```
~/.local/bin/qfme  ->  ~/.local/share/uv/tools/quanfire-multilingual-embedding/bin/qfme
```

That isolation is the point. The tool's dependencies — PyTorch among them — cannot collide
with anything else you have installed, and `qfme` still resolves from any directory.

!!! note "If `qfme: command not found` survives the install"
    `~/.local/bin` is not on your `PATH`. `uv tool update-shell` adds it; open a new
    terminal afterwards.

### Managing it

```bash
uv tool list                                    # what is installed
uv tool upgrade quanfire-multilingual-embedding # pull a newer version
uv tool uninstall quanfire-multilingual-embedding
```

### Name the extras

`'.[neural,wikipedia]'` — the quotes matter, since shells treat brackets as glob
characters.

| Extra | Adds | Omitting it means |
|---|---|---|
| `neural` | PyTorch | No contextual encoder. Everything else works. |
| `pretrained` | transformers | No adapting or loading a published checkpoint. |
| `wikipedia` | mwparserfromhell | `qfme extract` refuses to run, saying so clearly. |
| `serve` | FastAPI, uvicorn | `qfme serve` reports what to install and exits 1. |

Omit them all for a small install — corpus handling, tokenizer, vocabulary, word2vec,
search and evaluation need none of them.

---

## A complete session, from nothing

No repository, no clone, no virtual environment — a directory with a corpus in it.

```bash
mkdir ~/my-embeddings && cd ~/my-embeddings
```

**1. Check the corpus before trusting it.**

```bash
$ qfme validate --source corpus.jsonl
documents  200
sentences  1000
languages  en
scripts    Latn

No problems found.
```

**2. Describe the experiment.**

```yaml
# experiment.yaml
name: myproject
corpus:
  source: corpus.jsonl
tokenizer:
  vocab_size: 50
embedding:
  dimension: 64
  epochs: 15
```

**3. Train.**

```bash
$ qfme train --config experiment.yaml
{
  "name": "myproject",
  "documents": 200,
  "sentences": 1000,
  "vocabulary_size": 35,
  "dimension": 64,
  "characters_per_token": 3.194,
  "unknown_rate": 0.0,
  "experiment_directory": "artifacts/myproject"
}
```

**4. Search.**

```bash
$ qfme search --experiment artifacts/myproject --source corpus.jsonl \
      --query "machine learning" --top-k 2
Query: machine learning
Indexed 1000 sentences

 1. [0.9721] The teacher studies machine learning.
 2. [0.9721] The teacher studies machine learning.
```

Everything lands under the current directory:

```
artifacts/myproject/config.yaml          the resolved settings, for reproducing this run
artifacts/myproject/tokenizer/…          trained subword model
artifacts/myproject/embedding/…          vectors and vocabulary
reports/myproject/report.{json,md}       evaluation
```

**Exit codes are meant for scripting.** `qfme train` exits `1` when the corpus cannot
support the requested `vocab_size`, and `qfme validate` exits `1` on a corpus with errors,
so both belong in a pipeline with `set -e`.

---

## Develop it: a clone

Different job, different install. `uv tool install` gives you a *frozen copy*; editing the
source afterwards changes nothing.

```bash
git clone <url> && cd quanfire-multilingual-embedding
uv sync --extra neural --extra pretrained --extra wikipedia --extra serve
source .venv/bin/activate
pytest -q
```

Here `qfme` comes from `.venv/bin/` and only exists inside the activated environment, which
is why an unactivated terminal reports `command not found`. That is correct behaviour for a
development install — see [getting started](getting-started.md).

Both can coexist. If both are on your `PATH`, the activated project environment wins, which
is what you want while developing.

---

## Depend on it: from another project

Everything above installs a command. This installs a library — the case for a sibling
QuanFire repository that wants the corpus, vocabulary, tokenizer or evaluation layers
without reimplementing them.

```bash
uv add "quanfire-multilingual-embedding @ git+ssh://git@github.com/quanfire-ai/quanfire-multilingual-embedding.git@v0.5.0"
```

**Pin a tag, never a branch.** `uv tool upgrade` re-fetching the default branch is fine
for a person running a command; it is not fine for a repository whose test suite depends
on this one, because the dependency then changes under it without any commit on its side
saying so. `uv` records the resolved commit in `uv.lock`, so a tag gives a reproducible
install and a deliberate upgrade.

The base install is `numpy`, `pandas`, `pyyaml`, `sentencepiece` and `tqdm` — **no
torch**. That is deliberate: a consumer preparing text keeps full control of its own
training stack, including its own CUDA build. Add `[neural]` only if you want this
project's models too.

What you may depend on is listed in [`CHANGELOG.md`](../CHANGELOG.md) under **Public
API**, and asserted by `tests/test_public_api.py`. In short: `corpus`, `vocabulary`,
`tokenizer`, `evaluation` and `core.exceptions`, imported from the package rather than
from the module that defines them. Everything else may move without notice.

**`config` is not public, so use the entry points that do not need it.** Several classes
and functions here are configured by a type from `config`, which is free to change in a
patch release; each has a twin taking a source and plain settings, and those are the ones
to call:

| Instead of | Call |
|---|---|
| `SentencePieceTrainerAdapter(TokenizerConfig(...))` | `trainer_for(vocab_size=..., model_type=...)` |
| `build_reader(config)` | `reader_for(source, format=...)` |
| `stream_sentences(config)` | `sentences_from(source, ...)` |
| `stream_documents(config)` | `documents_from(source, ...)` |
| `load_corpus(config)` | `corpus_from(source, ...)` |

Every argument on the right is optional and omitting one keeps this project's default.
The versions on the left still work; they are for callers inside this repository, which
already hold a config because a YAML file produced one.

The SSH prerequisites in the next section apply here too.

## Giving it to someone else

The repository is private, and staying that way for now. That is not a blocker: **anyone
GitHub grants access to can install it directly**, with the same one-line command you used.

### For a collaborator with repo access

```bash
uv tool install 'git+ssh://git@github.com/quanfire-ai/quanfire-multilingual-embedding[neural,wikipedia]'
```

```
Installed 1 executable: qfme
```

Nothing else. No clone, no virtual environment, no build step. `uv` fetches over SSH,
builds the wheel, and links the shim — verified end to end, including that `qfme extract`
works afterwards, so the extras survive the git route.

They need two things first:

1. **An SSH key on their GitHub account**, working. `ssh -T git@github.com` should answer
   `Hi <name>! You've successfully authenticated`.
2. **Access to the repository** — collaborator, or membership of a team with read access.

Missing either produces:

```
git@github.com: Permission denied (publickey).
fatal: Could not read from remote repository.
```

Which is honest, unlike the anonymous HTTPS route: GitHub answers an unauthenticated
request for a private repository with `404`, so `git+https://` reports **"Repository not
found"** and sends the reader hunting for a typo in a URL that is perfectly correct.
Prefer `git+ssh://` while the repository is private, for that reason alone.

### Updating

```bash
uv tool upgrade quanfire-multilingual-embedding
```

```
Updating ssh://git@github.com/quanfire-ai/quanfire-multilingual-embedding (HEAD)
 Updated ssh://git@github.com/quanfire-ai/quanfire-multilingual-embedding (c1445b8)
```

It re-fetches the default branch, so a collaborator picks up new commits without touching
git themselves.

### When SSH is not available

A wheel needs nothing from GitHub at all — useful for an air-gapped machine, a CI runner
without a deploy key, or someone you would rather not add to the repository:

```bash
uv build --wheel                    # writes dist/*.whl
# send the file, then on their machine:
uv tool install ./quanfire_multilingual_embedding-0.5.0-py3-none-any.whl
```

The trade is that they get a frozen copy with no upgrade path — `uv tool upgrade` has
nowhere to look.

### If it ever goes public

| Route | Command | Needs |
|---|---|---|
| Public git | `uv tool install 'git+https://github.com/quanfire-ai/quanfire-multilingual-embedding[neural,wikipedia]'` | Making the repository public |
| PyPI | `uv tool install 'quanfire-multilingual-embedding[neural,wikipedia]'` | Publishing a release |

Two things want answers before either, rather than assumptions: the licence, still recorded
as undecided, and whether a model trained on CC BY-SA text carries that licence's terms —
see [data format](data-format.md).

---

## Requirements

| | |
|---|---|
| Python | 3.12. `uv` installs it if missing — you need not manage it. |
| Disk | ~200 MB base, ~1 GB with `neural` on CPU-only macOS, up to ~3 GB with a CUDA wheel |
| GPU | Not required for anything. It changes what is practical, not what runs. |
| Network | Only at install time. Nothing is downloaded at runtime — no model weights, no API keys. |
