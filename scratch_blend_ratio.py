#!/usr/bin/env python3
"""Blend the article pool and the sentence pool at a fixed total, three ratios.

Pure stdlib -- safe to run on the GPU box with any python3 without touching the
CUDA venv. Deterministic: same seed -> same file, so the sweep is reproducible.

Each pool is shuffled first (both are concatenated per-language, so a naive head
would over-represent whichever language comes first), then the requested counts
are drawn, combined, and shuffled again so article and sentence pairs interleave.

    python3 scratch_blend_ratio.py ARTICLE SENTENCE N_ART N_SENT OUT [SEED]
"""
import sys, gzip, random


def load(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [ln for ln in f if ln.strip()]


def main():
    article, sentence, n_art, n_sent, out = sys.argv[1:6]
    n_art, n_sent = int(n_art), int(n_sent)
    seed = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    rng = random.Random(seed)

    art = load(article)
    sen = load(sentence)
    assert n_art <= len(art), f"want {n_art} article, have {len(art)}"
    assert n_sent <= len(sen), f"want {n_sent} sentence, have {len(sen)}"
    rng.shuffle(art)
    rng.shuffle(sen)
    combined = art[:n_art] + sen[:n_sent]
    rng.shuffle(combined)

    with gzip.open(out, "wt", encoding="utf-8") as f:
        f.writelines(ln if ln.endswith("\n") else ln + "\n" for ln in combined)
    print(f"{out}: {len(combined)} records ({n_art} article + {n_sent} sentence), seed {seed}",
          flush=True)


if __name__ == "__main__":
    main()
