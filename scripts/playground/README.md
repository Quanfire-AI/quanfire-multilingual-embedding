# Playground demo corpus

The public playground (`playground.quanfire.ai`, Phase 2) searches over a
**fixed** multilingual corpus, so the base-vs-adapter comparison is instant and
only the visitor's short query is embedded live. These two scripts build that
corpus and its vectors. Everything is derived from FLORES-200 — the same
professionally-translated, line-aligned slice behind
`reports/global-baseline-verdict.json` — so the sentences a visitor searches are
exactly the ones behind the numbers on the eval receipt.

## 1. Select the sentences (here, no GPU)

```bash
python scripts/playground/build_corpus.py        # -> scripts/playground/corpus.json
```

Deterministic: picks 40 standalone, well-spaced concepts, each a genuine
parallel across all 15 languages (600 passages). Commit `corpus.json`; it is
vendored into the gateway (`quanfire-ai-backend`).

## 2. Embed with both models (on the GPU box, after the sidecars are up)

```bash
python scripts/playground/precompute_vectors.py \
    --adapter-url http://127.0.0.1:8009 \
    --base-url    http://127.0.0.1:8010 \
    --out         scripts/playground/corpus-vectors.bin
```

Sends every passage (as `input_type=passage`) to both sidecars' OpenAI-compatible
`/v1/embeddings`, L2-normalises, and writes `corpus-vectors.bin` (a raw float32
blob) + `corpus-vectors.json` (pids + provenance). The format is dependency-free
so the gateway loads it with the stdlib `array` module — no numpy on the serving
side. Sourcing the vectors from the *same served models* the gateway calls at
request time guarantees a live query and the corpus are comparable. Deploy both
artefacts where the gateway reads them.
