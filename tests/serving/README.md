# tests/serving

> Tests for [`multilingual_embedding.serving`](../../src/multilingual_embedding/serving/README.md) — the query/passage asymmetry across an HTTP boundary, the validation that runs before a model is touched, and the CLI wiring that must not make FastAPI mandatory.

**46 tests.** Run with `pytest tests/serving -q`.

## Files

| File | Covers |
|---|---|
| `test_app.py` | The endpoint against a real adapter: prefixes reaching the encoder, the refusal to guess a side, the operator default, symmetric models, response shape, the model card, errors, startup |
| `test_schemas.py` | What is refused before any model is loaded, and why each rule exists |
| `test_cli_serve.py` | `qfme serve` flags and defaults, and the import-graph guard |

`test_app.py` needs torch, transformers and the `serve` extra; `test_schemas.py` needs
pydantic. Both call `pytest.importorskip` at module level, so a core-only checkout skips
them instead of erroring. `test_cli_serve.py` needs nothing optional — that is the point
of its last test.

## What matters here

**The decisive test compares vectors, not fields.** A server that stores `query: `,
reports it in `prefix_applied` and never prepends it would answer 200 with the right
dimension, unit norm and no NaN. Every other assertion in this directory would still pass.
So `test_the_two_sides_return_different_vectors` sends the same text twice, declaring a
different side each time, and requires the returned vectors to differ. If they did not,
the prefix would be decoration and an E5 model would be served as a symmetric one, with
nothing to show for it but a worse result list.

**The refusal is tested for its message, not only its status.** A `400` that does not say
what to send makes the caller guess, which is exactly the thing the endpoint declines to do
on their behalf. `test_the_refusal_names_both_valid_values` asserts both `query` and
`passage` appear in the message and that `param` names the field.

**Null and empty string are held apart deliberately.** `prefix_applied` is `null` for a
symmetric model. Both values are falsy and neither raises, but they say different things,
and a test that accepted either would let the distinction rot.

**Batch order is asserted against single-item calls.** Indices are the only thing tying a
vector back to its input, so a reordering would silently mislabel every result. The test
embeds each text alone and requires the batched vectors to match position for position.

**The model card test names a real number.** `max_length == 32` for the fixture model. A
`getattr` fallback published `0` for a model whose real limit was 256, which is the defect
that test was written against — a client sizing its chunks from that field had no way to
know it was wrong.

**Startup failure is asserted as ordering.** `create_app` on a directory that does not
exist must raise there, not on some caller's first request. A deployment notices a process
that will not start; it does not notice one that starts and then fails whatever it is
first asked.

**One test runs in a fresh interpreter.** `test_building_the_parser_does_not_import_fastapi`
subprocesses out because a test session that has already imported FastAPI cannot observe
the import graph. Serving is one deployment of this project, and registering its subcommand
must not make its dependencies mandatory for `qfme stats`.

**The checkpoint is built, not downloaded.** A 1,003-token vocabulary and a two-layer,
64-dimension BERT are written to a module-scoped `tmp_path`, adapted with a non-zero LoRA
adapter and saved the way an adaptation run saves one. Tests reach no network, and the
whole directory runs in about fifteen seconds. The adapter is deliberately given non-zero
weights: a zeroed one would make the base and adapted models identical, and several
assertions here would hold whether or not it had been loaded.
