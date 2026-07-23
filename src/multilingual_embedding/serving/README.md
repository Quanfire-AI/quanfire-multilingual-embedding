# serving

> Puts an adapter behind an HTTP endpoint using the de facto industry-standard embeddings schema, and refuses to guess the one thing that schema cannot express.

## Purpose

An adaptation run that measures a 39% improvement and writes 3.4 MB to disk has produced
an artefact nobody outside this repository can call. `SemanticSearchPipeline.from_adapter`
closed half that gap — it turned the artefact back into a working model. This layer
closes the other half: it turns the model into something a service in another language,
on another machine, can use without importing Python.

The wire format is the standard one, so a client migrates by changing a base URL. That
choice is worth stating plainly: nothing here is novel, and that is the point. The
interesting decision is what happens where the standard format does not fit.

## Modules

| Module | Responsibility |
|---|---|
| `schemas.py` | The request and response shapes, reproduced rather than imported, plus the validation that rejects inputs which would otherwise produce a plausible wrong answer. |
| `app.py` | `ServingConfig`, `create_app`, and the resolution of query/passage asymmetry against a schema that has no field for it. |

## Key design decisions

**The server will not guess which side of the model a text belongs to.** This is the
decision the layer exists to get right. An E5-family checkpoint is trained with `query: `
on one side and `passage: ` on the other. Served without them it returns vectors of the
right shape and the right norm, free of NaN, that encode the wrong thing. Nothing raises.
Retrieval is simply worse, which is indistinguishable from the model not being very good —
and every measurement in this repository exists to tell those two apart.

The standard schema has no field for this. Three ways out:

| | Consequence |
|---|---|
| Default to `query` | Every passage-indexing job is silently wrong, and the symptom is a slightly worse index nobody attributes to this. |
| Default to `passage` | The same, on the other side. |
| Refuse, name both values, offer an operator default | One line of client code, once, per deployment. |

The third is the only one whose failure mode is visible. An asymmetric model with no
configured default answers `400` naming both valid values; a symmetric model has nothing
to choose and accepts anything. `--default-input-type` exists for the deployment that
genuinely is single-sided — a query-only search box, a passage-only indexing job — where
the operator can state once what every caller would otherwise repeat.

**`prefix_applied` is echoed in every response.** A caller who wants to reproduce a vector
elsewhere needs to know exactly what string was encoded. It is `null` — not `""` — for a
symmetric model, so "this model has no sides" stays distinguishable from "an asymmetric
model was served without a prefix", which would be a bug rather than a property.

**The model is loaded in `create_app`, before the server accepts traffic.** A missing
checkpoint or a corrupt adapter fails at startup, where a deployment notices, rather than
on some caller's first request. That is also why `cli.py` builds the app itself instead of
handing uvicorn a factory string: the failure arrives with this process's error handling,
before the port is claimed.

**The model card reads the artefact, not the encoder.** `max_length` and `normalize` are
recorded in `adapter.json` but are not part of the `TextEncoder` protocol, and a `getattr`
fallback published `max_length: 0` for a model whose real limit is 256. A card that states
a wrong number is worse than one that omits the field, because a client sizing its chunks
from it has no way to know. The defaults used when a key is absent match `load_adapter`'s
exactly, so the card describes the encoder that is actually running.

**Validation rejects what would otherwise succeed.** Each rule in `schemas.py` guards an
input that produces a well-formed response rather than an error: an empty or
whitespace-only string is what an unfilled template field looks like by the time it
reaches a wire format, and it embeds to a point in the space and takes a rank in a result
list. A batch is embedded in a single pass, so `MAX_BATCH` is what stops one caller from
holding the event loop while everyone else waits.

**The schema is reproduced, not imported.** Depending on a vendor's SDK to describe a wire
format this project does not control would be a dependency taken for a data class.

**Token counts are approximate and say so.** The served adapter wraps a Hugging Face
tokenizer rather than this project's, and reaching into it is not part of any contract, so
`_approximate_tokens` falls back to a whitespace count. The number is reported because
clients written against the standard schema read it and fail on a missing key; it is not
billed against anything here.

## What this layer does not do

No authentication, no rate limiting, no batching across requests, no dimension truncation,
no ONNX or quantised path. `qfme serve` binds `127.0.0.1` by default for exactly that
reason — the endpoint is not safe to expose directly, and a default of `0.0.0.0` would put
an unauthenticated GPU-backed service on whatever network the host happens to be on the
first time anyone runs it. Exposing it is `--host 0.0.0.0` behind something that
terminates TLS and checks credentials.

## Usage

```bash
uv sync --extra serve --extra pretrained --extra neural

qfme serve --adapter models/indic-v1 --port 8000
```

```bash
curl localhost:8000/v1/embeddings \
  -H 'content-type: application/json' \
  -d '{"input": "संविधान में मौलिक अधिकार", "input_type": "query"}'
```

```python
from multilingual_embedding.serving import ServingConfig, create_app

app = create_app(ServingConfig(adapter_directory=Path("models/indic-v1")))
```

| Route | Returns |
|---|---|
| `GET /health` | Liveness, plus which model and framework version answered — enough to tell two deployments apart. |
| `GET /v1/models` | The model card: dimension, context limit, normalization, both prefixes. |
| `POST /v1/embeddings` | Vectors, in input order, with `prefix_applied` and a usage count. |

## Dependencies

The top layer, above `pipelines`. It imports `common`, `core` and `pipelines`, and nothing
inside the framework may import it.

FastAPI, uvicorn and pydantic all sit behind the optional `serve` extra. Importing
`multilingual_embedding.serving` does *not* require them: `__init__.py` defers
`ServingConfig` and `create_app` through `__getattr__`, and `app.py` imports FastAPI inside
`create_app`. `tests/serving/test_cli_serve.py` asserts in a fresh interpreter that
building the CLI parser leaves `fastapi` out of `sys.modules`, so a later edit that moves
an import to module scope makes the extra mandatory for `qfme stats` and gets caught.

## Tests

`tests/serving/` holds **46 tests**. The decisive one compares the vectors returned for
the same text on each side and requires them to differ: a server that stores `query: `,
reports it in `prefix_applied` and never prepends it would pass every shallow check — 200,
right dimension, unit norm, no NaN — and only that comparison catches it.
