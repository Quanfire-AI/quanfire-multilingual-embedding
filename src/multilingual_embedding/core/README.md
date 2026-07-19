# core

> Framework plumbing: the exception hierarchy, logging setup, the component registry and the configuration-driven factory that builds components from it.

## Purpose

Four concerns appear in every package of this framework and belong to none of them: how failures are reported, how diagnostics are emitted, how a pluggable implementation is found by name, and how such an implementation is constructed from a configuration file. Putting them here gives each exactly one implementation, and lets the layers above stay focused on their domain. Like `common`, this package imports nothing from inside the framework, which is what keeps the import graph acyclic.

## Modules

| Module | Responsibility |
|---|---|
| `exceptions.py` | The error hierarchy rooted at `MultilingualEmbeddingError`, plus `ConfigurationError`, `ValidationError`, `RegistryError`, `SerializationError`, `ResourceNotFoundError` and `NotFittedError`. |
| `logging.py` | `get_logger` for library code, `configure_logging` for applications, `JsonFormatter` for structured output, and the `ROOT_LOGGER_NAME` namespace. |
| `registry.py` | `Registry[T]`, a name-to-implementation mapping for one component family. |
| `factory.py` | `build_from_config` and `build_all_from_config`, which turn a `{"type": ...}` specification into instances via a registry. `TYPE_KEY` names the discriminator. |

## Key design decisions

### Errors carry structured context, not formatted strings

Every deliberate framework failure derives from `MultilingualEmbeddingError`, so a caller can write `except MultilingualEmbeddingError` and catch framework problems without also swallowing unrelated builtins.

The base constructor takes a message plus arbitrary keyword arguments, and stores them on `self.context` rather than baking them into the message. `__str__` renders them — sorted, with `repr` values — only when the string form is actually needed:

```
Unknown configuration keys for component (accepted=['remove_control_characters'], registry='normalizer', type='nfkc', unknown=['remove_ctrl_chars'])
```

The alternative — an f-string built at the raise site — throws the values away. Keeping them structured means a log handler can emit them as queryable fields, and a test can assert on `error.context["unknown"]` rather than regex-matching prose that a later wording change will break.

### Logging is never configured as an import side effect

Importing this framework attaches no handlers and sets no levels. Library code calls `get_logger(__name__)` and logs; the application decides where records go by calling `configure_logging` once at startup. A library that configures logging on import steals a decision that belongs to the program embedding it, and the effect is invisible until output turns up somewhere unexpected.

`configure_logging` is safe to call more than once: it removes and closes existing handlers before adding the new one, so reconfiguring at runtime replaces the destination rather than doubling the output.

It also sets `logger.propagate = False`. Framework records are fully handled by the handler attached here, so letting them bubble to the root logger would print each one twice in any application that has configured its own root handler — the classic duplicated-log-line bug.

`get_logger` namespaces under `ROOT_LOGGER_NAME` (`"multilingual_embedding"`) and will not duplicate the prefix if the name already carries it, so passing `__name__` from inside the package is safe. The default stream is `sys.stderr`, keeping stdout clean for program output.

### `Registry` decouples selection from import

A registry maps a name to an implementation class, so a YAML file can say `type: nfkc` and get an `NFKCNormalizer` without the config layer importing the tokenizer package — which the layer order forbids anyway. Normalizers, pre-tokenizers, tokenizers and embedding models each own an instance.

Duplicate keys are rejected loudly. `register` raises `RegistryError` when a key is already taken unless `override=True` is passed explicitly. Registration happens at import time, so a silent overwrite would mean the implementation that wins depends on module import order — a bug that appears only on the machine where imports happen to resolve differently. Failing at import time makes it deterministic and immediate.

Keys are normalised to lowercase and stripped, so config files may use whichever casing reads best, and `get` reports the available keys on a miss, turning a typo into an obvious fix rather than a bare `KeyError`.

### `build_from_config` validates constructor arguments up front

Before instantiating anything, `_reject_unknown_arguments` inspects the target's signature and compares it against the supplied keys. A misspelt YAML key produces a `ConfigurationError` naming both the unknown key and the accepted ones. Without this, the failure surfaces as an opaque `TypeError` raised from inside a constructor several frames down, which says what Python could not do but not which line of which config file caused it.

Implementations accepting `**kwargs` are skipped, since anything is legal for them. A specification may also be a bare string when no arguments are needed, and `overrides` take precedence over the file — that is how runtime values such as a resolved path or a shared vocabulary get injected into a component whose other settings are static.

## Usage

```python
import io, logging
from multilingual_embedding.core import (
    Registry, build_from_config, configure_logging, get_logger,
    ConfigurationError, RegistryError,
)

class Normalizer: ...

normalizers: Registry[Normalizer] = Registry("normalizer")

@normalizers.register("nfkc")
class NFKCNormalizer(Normalizer):
    def __init__(self, remove_control_characters: bool = True) -> None:
        self.remove_control_characters = remove_control_characters

print("keys:", normalizers.keys(), "| 'NFKC' in registry:", "NFKC" in normalizers)

component = build_from_config(normalizers, {"type": "nfkc", "remove_control_characters": False})
print("built:", type(component).__name__, component.remove_control_characters)

try:
    build_from_config(normalizers, {"type": "nfkc", "remove_ctrl_chars": True})
except ConfigurationError as error:
    print("error:", error)
    print("context keys:", sorted(error.context))

try:
    normalizers.register("nfkc", NFKCNormalizer)
except RegistryError as error:
    print("duplicate:", error)

stream = io.StringIO()
configure_logging(level=logging.INFO, log_format="json", stream=stream)
get_logger("tokenizer.trainer").info("tokenizer trained", extra={"vocab_size": 32000})
print("log:", stream.getvalue().strip())
```

Output (timestamp will differ):

```
keys: ['nfkc'] | 'NFKC' in registry: True
built: NFKCNormalizer False
error: Unknown configuration keys for component (accepted=['remove_control_characters'], registry='normalizer', type='nfkc', unknown=['remove_ctrl_chars'])
context keys: ['accepted', 'registry', 'type', 'unknown']
duplicate: Duplicate registry key (existing='NFKCNormalizer', key='nfkc', registry='normalizer')
log: {"timestamp": "2026-07-19 20:43:49,832", "level": "INFO", "logger": "multilingual_embedding.tokenizer.trainer", "message": "tokenizer trained", "vocab_size": 32000}
```

The `vocab_size` passed through `extra=` appears as a top-level JSON field, not buried in the message text — that is the point of `JsonFormatter`, which merges any non-reserved `LogRecord` attribute into the payload.

## Dependencies

`core` sits second from the bottom and **imports nothing from inside the framework** — not even `common`. Only the standard library. Every package above it may import it, and in practice nearly all do.

`tests/test_architecture.py::test_foundation_layers_have_no_internal_dependencies` asserts the empty-import-set property for both `common` and `core` by parsing the source.

## Tests

Tests live in `tests/core/`, 27 in total:

| File | Tests | Coverage |
|---|---|---|
| `tests/core/test_factory.py` | 10 | Building from a mapping and from a bare string, override precedence, missing and non-string `type`, unknown-key rejection, `**kwargs` implementations, and `build_all_from_config` including the empty and wrong-shape cases. |
| `tests/core/test_registry.py` | 10 | Both registration forms, case-insensitive lookup, duplicate rejection and `override=True`, unknown-key errors, `create`, `unregister`, and the container protocol methods. |
| `tests/core/test_logging.py` | 7 | Logger namespacing, prefix de-duplication, the unnamed-logger case, JSON records being parseable, handler replacement on repeated configuration, and exception rendering with and without context. |

There is no dedicated `test_exceptions.py`. Context rendering is covered by `test_logging.py::test_exception_context_is_rendered` and `test_exception_without_context_renders_message_only`, and the individual error types are exercised wherever they are raised — the factory and registry tests assert on both the message and `error.context`.

Run them with `.venv/bin/python -m pytest tests/core -q`.
