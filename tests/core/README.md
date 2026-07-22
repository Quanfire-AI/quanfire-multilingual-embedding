# tests/core

> Tests for [`multilingual_embedding.core`](../../src/multilingual_embedding/core/README.md) — registry, factory, logging, exceptions.

**29 tests.** Run with `pytest tests/core -q`.

## Files

| File | Covers |
|---|---|
| `test_registry.py` | Registration, decorator form, case-insensitive keys, duplicate rejection, `override`, unknown-key errors, `create`, iteration |
| `test_factory.py` | Building from a mapping or a bare string, override precedence, unknown-key rejection, `**kwargs` components, ordered `build_all_from_config` |
| `test_logging.py` | Logger namespacing, JSON formatting with `extra` fields, handler replacement on reconfigure, exception context rendering |

## What matters here

**Duplicate registration must fail loudly.** Two components claiming the same key would
otherwise silently shadow one another depending on import order, and the survivor would
vary between runs. The test asserts a `RegistryError` rather than last-write-wins.

**Unknown configuration keys must be rejected with the accepted set.** A typo in a YAML
file is the most common configuration error there is, and the difference between a
usable framework and a frustrating one is whether it answers "`typo` is not valid;
accepted keys are `x`" or raises an opaque `TypeError` from inside a constructor.

**Reconfiguring logging must not accumulate handlers.** A second `configure_logging`
call replacing rather than appending is what stops every log line being emitted twice
in an application that reconfigures at runtime.

**Exception context must survive as data.** `MultilingualEmbeddingError` carries
key/value context that renders into the message *and* stays available on `.context`,
so tests can assert on individual values and log aggregation can index them as fields.
