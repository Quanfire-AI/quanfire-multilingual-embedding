#!/usr/bin/env python3
"""Reproduce the Samanantar sentence-bitext proof corpus.

Downloads a slice of AI4Bharat Samanantar (en<->indic sentence pairs) from the
Hugging Face hub, extracts N line-aligned rows per language, runs each language
through `qfme ingest-parallel` (the same tool the pipeline ships), and combines
the result into one pair file the trainer consumes.

Samanantar covers 8 of our 10 languages -- sa (Sanskrit) and ur (Urdu) are
absent upstream, so the sentence side omits them; the article corpus still
carries all ten. FLORES-200 is held strictly out: it is never downloaded here.

    uv run python scratch_samanantar_prep.py            # Mac: prep only
    # then scp data/pairs/samanantar-proof-240k.jsonl.gz to the GPU box

Needs pyarrow + huggingface_hub (uv pip install pyarrow huggingface_hub).
"""
import os, subprocess, gzip, shutil, sys
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

# First shard per language; each holds ~1.5M rows, we read the first N.
SHARDS = {
    "hi": "hi/train-00000-of-00008.parquet", "ta": "ta/train-00000-of-00004.parquet",
    "bn": "bn/train-00000-of-00005.parquet", "gu": "gu/train-00000-of-00002.parquet",
    "kn": "kn/train-00000-of-00002.parquet", "ml": "ml/train-00000-of-00004.parquet",
    "mr": "mr/train-00000-of-00002.parquet", "te": "te/train-00000-of-00003.parquet",
}
N = 30_000                       # rows kept per language
MIN_CHARS = 10                   # matches ingest-parallel defaults
WORK = "scratch_samanantar"
TXT = f"{WORK}/txt"
PAIRS = f"{WORK}/pairs"
COMBINED = "data/pairs/samanantar-proof-240k.jsonl.gz"


def extract():
    os.makedirs(TXT, exist_ok=True)
    for lang, path in SHARDS.items():
        p = hf_hub_download("ai4bharat/samanantar", path, repo_type="dataset")
        en, xx = [], []
        for batch in pq.ParquetFile(p).iter_batches(batch_size=10_000, columns=["src", "tgt"]):
            d = batch.to_pydict()
            for s, t in zip(d["src"], d["tgt"]):
                if not s or not t:
                    continue
                s, t = s.replace("\n", " ").strip(), t.replace("\n", " ").strip()
                if len(s) < MIN_CHARS or len(t) < MIN_CHARS:
                    continue
                en.append(s); xx.append(t)
                if len(en) >= N:
                    break
            if len(en) >= N:
                break
        open(f"{TXT}/en-{lang}.txt", "w").write("\n".join(en) + "\n")
        open(f"{TXT}/{lang}.txt", "w").write("\n".join(xx) + "\n")
        print(f"{lang}: {len(en)} rows", flush=True)


def ingest():
    os.makedirs(PAIRS, exist_ok=True)
    for lang in SHARDS:
        subprocess.run([
            "qfme", "ingest-parallel",
            "--anchor-file", f"{TXT}/en-{lang}.txt",
            "--positive-file", f"{TXT}/{lang}.txt",
            "--anchor-language", "en", "--positive-language", lang,
            "--kind", "samanantar", "--document-prefix", f"samanantar-{lang}",
            "--output", f"{PAIRS}/sam-{lang}.jsonl.gz",
        ], check=True)


def combine():
    os.makedirs("data/pairs", exist_ok=True)
    with open(COMBINED, "wb") as out:
        for lang in SHARDS:
            with open(f"{PAIRS}/sam-{lang}.jsonl.gz", "rb") as f:
                shutil.copyfileobj(f, out)   # concatenated gzip streams read as one
    n = sum(1 for _ in gzip.open(COMBINED, "rt"))
    print(f"combined -> {COMBINED}: {n} records", flush=True)


if __name__ == "__main__":
    extract()
    ingest()
    combine()
