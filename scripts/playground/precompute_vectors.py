#!/usr/bin/env python3
"""Embed the Playground corpus with both served models — the deploy-time step.

Turns ``corpus.json`` (built by :file:`build_corpus.py`) into the fixed vectors
the public gateway ranks a live query against. It embeds every passage twice:
once through the **adapter** sidecar and once through the **base** sidecar.

It talks to the *running sidecars* over their OpenAI-compatible
``/v1/embeddings`` endpoint rather than loading the models itself. That is
deliberate: the whole point of the fixed corpus is that a live query and the
corpus are embedded by the *same* served model, so their dot products are
comparable. Sourcing the corpus vectors from the identical endpoint the gateway
will call at request time removes any chance of a checkpoint, prefix, or pooling
mismatch between "the model that indexed the corpus" and "the model that
embeds your query". As a bonus, a clean run is proof both sidecars are up.

Every passage is sent with ``input_type=passage`` — the corpus is the passage
side of the asymmetric E5 models. The gateway sends visitor queries with
``input_type=query``. Vectors are L2-normalised here so the gateway's ranking is
a plain dot product.

Run on the GPU box, after both sidecars are serving (see the deploy scripts):

    python scripts/playground/precompute_vectors.py \\
        --adapter-url http://127.0.0.1:8009 \\
        --base-url    http://127.0.0.1:8010 \\
        --out         scripts/playground/corpus-vectors.bin

Writes ``corpus-vectors.bin`` (a raw little-endian float32 blob: the adapter
matrix then the base matrix, row-major) and ``corpus-vectors.json`` (pid order +
provenance) beside it. The format is deliberately dependency-free so the gateway
loads it with the stdlib ``array`` module — no numpy on the serving side. Both
are deployed where the gateway reads them (quanfire-ai-backend). This script
uses only the standard library plus numpy (for the embed-side normalisation).
"""
from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CORPUS = HERE / "corpus.json"

# Chunk well under the server's MAX_BATCH (512) so one request is quick and a
# transient failure re-sends little work.
CHUNK = 128
TIMEOUT = 120.0


def _embed_batch(url: str, texts: list[str], input_type: str) -> list[list[float]]:
    """POST one batch to a sidecar's /v1/embeddings, in input order."""
    payload = json.dumps({"input": texts, "input_type": input_type}).encode("utf-8")
    request = urllib.request.Request(
        f"{url.rstrip('/')}/v1/embeddings",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")
        raise SystemExit(f"{url} returned HTTP {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise SystemExit(f"could not reach {url}: {error.reason}. Is the sidecar up?") from error
    # The server preserves input order and stamps each object's index; sort by it
    # to be certain rather than trusting list position.
    rows = sorted(body["data"], key=lambda item: item["index"])
    return [row["embedding"] for row in rows]


def _model_id(url: str) -> str:
    """The served model's id, for the provenance record."""
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/v1/models", timeout=TIMEOUT) as response:
            return json.loads(response.read().decode("utf-8"))["data"][0]["id"]
    except Exception:
        return "unknown"


def embed_all(url: str, texts: list[str], input_type: str) -> np.ndarray:
    """Embed every text through one sidecar, L2-normalised, as float32."""
    vectors: list[list[float]] = []
    for start in range(0, len(texts), CHUNK):
        chunk = texts[start:start + CHUNK]
        vectors.extend(_embed_batch(url, chunk, input_type))
        print(f"  {url}: {min(start + CHUNK, len(texts))}/{len(texts)}")
    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-url", default="http://127.0.0.1:8009",
                        help="sidecar serving the shipped adapter (default :8009)")
    parser.add_argument("--base-url", default="http://127.0.0.1:8010",
                        help="sidecar serving base multilingual-e5-small (default :8010)")
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--out", type=Path, default=HERE / "corpus-vectors.bin",
                        help="raw float32 blob; a .json index is written beside it")
    args = parser.parse_args()

    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    passages = corpus["passages"]
    pids = [p["pid"] for p in passages]
    texts = [p["text"] for p in passages]

    print(f"embedding {len(texts)} passages through the adapter sidecar…")
    adapter = embed_all(args.adapter_url, texts, "passage")
    print(f"embedding {len(texts)} passages through the base sidecar…")
    base = embed_all(args.base_url, texts, "passage")

    dim = int(adapter.shape[1])
    if base.shape[1] != dim:
        raise SystemExit(f"dimension mismatch: adapter {dim} vs base {base.shape[1]}")

    # Portable artefact: a raw little-endian float32 blob (adapter block then
    # base block, row-major) plus a JSON index. The gateway reads it with the
    # stdlib `array` module and ranks in pure Python, so it needs no numpy and
    # no lockfile change to load a couple of MB of vectors.
    blob = args.out
    blob.write_bytes(
        adapter.astype("<f4").tobytes(order="C") + base.astype("<f4").tobytes(order="C")
    )

    index = {
        "pids": pids,
        "dim": dim,
        "n_passages": len(pids),
        "order": ["adapter", "base"],  # block order inside the blob
        "dtype": "float32",
        "byte_order": "little",
        "normalized": True,
        "input_type": "passage",
        "adapter_model": _model_id(args.adapter_url),
        "base_model": _model_id(args.base_url),
        "adapter_url": args.adapter_url,
        "base_url": args.base_url,
        "corpus_source": corpus["meta"]["source"],
    }
    index_path = blob.with_suffix(".json")
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"\nwrote {blob} ({len(pids)} x {dim} x 2 float32 = {blob.stat().st_size} bytes)")
    print(f"wrote {index_path}")
    print(f"  adapter model: {index['adapter_model']}   base model: {index['base_model']}")


if __name__ == "__main__":
    main()
