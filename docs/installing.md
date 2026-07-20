# Installing `qfme` as a normal command

Two different jobs, often confused, and the right answer differs:

| You want to | Use |
|---|---|
| **Use** the tool — run `qfme` anywhere, on your own data | `uv tool install` |
| **Develop** the project — edit code, run the tests | `uv sync` in a clone |

Everything below was executed and its output pasted verbatim.

---

## Use it: a system-wide `qfme`

```bash
uv tool install 'git+https://github.com/<owner>/quanfire-multilingual-embedding[neural,wikipedia]'
```

Or from a clone, which is what to do while the repository is private:

```bash
cd quanfire-multilingual-embedding
uv tool install '.[neural,wikipedia]'
```

```
Installed 1 executable: qfme
```

That is the whole install. No virtual environment to activate, no `PATH` to edit, nothing
to remember. From any directory:

```bash
$ cd /tmp
$ which qfme
/Users/you/.local/bin/qfme
$ qfme --version
qfme 0.1.0
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
| `wikipedia` | mwparserfromhell | `qfme extract` refuses to run, saying so clearly. |

Omit both for a small install — corpus handling, tokenizer, vocabulary, word2vec, search
and evaluation do not need either.

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
uv sync --extra neural --extra wikipedia
source .venv/bin/activate
pytest -q
```

Here `qfme` comes from `.venv/bin/` and only exists inside the activated environment, which
is why an unactivated terminal reports `command not found`. That is correct behaviour for a
development install — see [getting started](getting-started.md).

Both can coexist. If both are on your `PATH`, the activated project environment wins, which
is what you want while developing.

---

## Giving it to someone else

**Today the repository is private**, so `uv tool install git+https://…` fails for anyone
without access — GitHub answers an anonymous request with `404`, so the error reads
"Repository not found" rather than "permission denied". Verified:

```
$ git clone https://github.com/<owner>/quanfire-multilingual-embedding
remote: Repository not found.
```

Three ways to distribute, in increasing order of effort:

| Route | Command for the recipient | Needs |
|---|---|---|
| **Wheel** | `uv tool install ./quanfire_multilingual_embedding-0.1.0-py3-none-any.whl` | You send a file from `uv build` |
| **Private git** | `uv tool install 'git+ssh://git@github.com/<owner>/<repo>[neural,wikipedia]'` | Their SSH key on the repo |
| **Public or PyPI** | `uv tool install 'quanfire-multilingual-embedding[neural,wikipedia]'` | Making the repo public, or publishing |

The wheel route needs nothing from GitHub:

```bash
uv build --wheel        # writes dist/*.whl
```

Before publishing publicly, two things need answers rather than assumptions: the licence,
which is still recorded as undecided, and whether a model trained on CC BY-SA text carries
that licence's terms — see [data format](data-format.md).

---

## Requirements

| | |
|---|---|
| Python | 3.12. `uv` installs it if missing — you need not manage it. |
| Disk | ~200 MB base, ~1 GB with `neural` on CPU-only macOS, up to ~3 GB with a CUDA wheel |
| GPU | Not required for anything. It changes what is practical, not what runs. |
| Network | Only at install time. Nothing is downloaded at runtime — no model weights, no API keys. |
