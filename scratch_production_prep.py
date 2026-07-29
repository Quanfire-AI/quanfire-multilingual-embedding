#!/usr/bin/env python3
"""Build the production sentence-scale pool: all ten languages, not eight.

The proof (scratch_samanantar_prep.py) used 30k rows/lang of Samanantar for the
8 languages Samanantar covers; sa (Sanskrit) and ur (Urdu) were absent, so the
sentence side omitted them. This scales the Samanantar side up and closes that
gap with two non-gated sentence sources:

  sa  itihasa (rahular/itihasa on GitHub) -- line-aligned en/sn, classical-epic
      Sanskrit. Domain differs from Wikipedia/FLORES prose; it is the best
      readily-available, non-gated en<->sa sentence bitext. ~93k pairs.
  ur  OPUS-100 en-ur (Helsinki-NLP/opus-100) -- modern sentence bitext, 753k
      rows upstream; we take N_URDU.

BPCC (Samanantar's successor, which does carry sa/ur) is a gated HF repo and is
deliberately avoided so this stays reproducible without an access grant.

Everything runs on the Mac. Only the finished pool is scp'd to the GPU box.
FLORES-200 is held strictly out -- never downloaded here.

    uv run python scratch_production_prep.py

Needs pyarrow + huggingface_hub (uv pip install pyarrow huggingface_hub).
"""
import os, subprocess, gzip, shutil, urllib.request
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

SAMANANTAR = {
    "hi": "hi/train-00000-of-00008.parquet", "ta": "ta/train-00000-of-00004.parquet",
    "bn": "bn/train-00000-of-00005.parquet", "gu": "gu/train-00000-of-00002.parquet",
    "kn": "kn/train-00000-of-00002.parquet", "ml": "ml/train-00000-of-00004.parquet",
    "mr": "mr/train-00000-of-00002.parquet", "te": "te/train-00000-of-00003.parquet",
}
N_SAMANANTAR = 150_000           # rows kept per Samanantar language
N_URDU = 150_000                 # rows kept from opus-100 en-ur
MIN_CHARS = 10
WORK = "scratch_samanantar"      # reuse the same scratch tree (gitignored)
TXT = f"{WORK}/txt"
PAIRS = f"{WORK}/pairs"
POOL = "data/pairs/sentence-pool-10lang.jsonl.gz"


def write_pair(lang, en, xx):
    """Write line-aligned txt files and ingest them into a per-lang pair file."""
    assert len(en) == len(xx), f"{lang}: {len(en)} != {len(xx)}"
    open(f"{TXT}/en-{lang}.txt", "w").write("\n".join(en) + "\n")
    open(f"{TXT}/{lang}.txt", "w").write("\n".join(xx) + "\n")
    subprocess.run([
        "qfme", "ingest-parallel",
        "--anchor-file", f"{TXT}/en-{lang}.txt",
        "--positive-file", f"{TXT}/{lang}.txt",
        "--anchor-language", "en", "--positive-language", lang,
        "--kind", "sentence", "--document-prefix", f"sentence-{lang}",
        "--output", f"{PAIRS}/sent-{lang}.jsonl.gz",
    ], check=True)
    print(f"{lang}: {len(en)} rows ingested", flush=True)


def clean(s):
    return s.replace("\n", " ").replace("\t", " ").strip()


def samanantar():
    for lang, path in SAMANANTAR.items():
        p = hf_hub_download("ai4bharat/samanantar", path, repo_type="dataset")
        en, xx = [], []
        for batch in pq.ParquetFile(p).iter_batches(batch_size=10_000, columns=["src", "tgt"]):
            d = batch.to_pydict()
            for s, t in zip(d["src"], d["tgt"]):
                if not s or not t:
                    continue
                s, t = clean(s), clean(t)
                if len(s) < MIN_CHARS or len(t) < MIN_CHARS:
                    continue
                en.append(s); xx.append(t)
                if len(en) >= N_SAMANANTAR:
                    break
            if len(en) >= N_SAMANANTAR:
                break
        write_pair(lang, en, xx)


def sanskrit():
    base = "https://raw.githubusercontent.com/rahular/itihasa/master/data/"
    en_raw = urllib.request.urlopen(base + "train.en").read().decode("utf-8").splitlines()
    sn_raw = urllib.request.urlopen(base + "train.sn").read().decode("utf-8").splitlines()
    assert len(en_raw) == len(sn_raw), f"itihasa mismatch {len(en_raw)} != {len(sn_raw)}"
    en, xx = [], []
    for s, t in zip(en_raw, sn_raw):
        s, t = clean(s), clean(t)
        if len(s) < MIN_CHARS or len(t) < MIN_CHARS:
            continue
        en.append(s); xx.append(t)
    write_pair("sa", en, xx)


def urdu():
    p = hf_hub_download("Helsinki-NLP/opus-100", "en-ur/train-00000-of-00001.parquet",
                        repo_type="dataset")
    en, xx = [], []
    for batch in pq.ParquetFile(p).iter_batches(batch_size=10_000, columns=["translation"]):
        for row in batch.to_pydict()["translation"]:
            s, t = clean(row.get("en", "")), clean(row.get("ur", ""))
            if len(s) < MIN_CHARS or len(t) < MIN_CHARS:
                continue
            en.append(s); xx.append(t)
            if len(en) >= N_URDU:
                break
        if len(en) >= N_URDU:
            break
    write_pair("ur", en, xx)


def combine():
    langs = list(SAMANANTAR) + ["sa", "ur"]
    with open(POOL, "wb") as out:
        for lang in langs:
            with open(f"{PAIRS}/sent-{lang}.jsonl.gz", "rb") as f:
                shutil.copyfileobj(f, out)   # concatenated gzip streams read as one
    n = sum(1 for _ in gzip.open(POOL, "rt"))
    print(f"combined -> {POOL}: {n} records across {len(langs)} languages", flush=True)


if __name__ == "__main__":
    os.makedirs(TXT, exist_ok=True)
    os.makedirs(PAIRS, exist_ok=True)
    os.makedirs("data/pairs", exist_ok=True)
    samanantar()
    sanskrit()
    urdu()
    combine()
