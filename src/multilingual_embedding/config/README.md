# config

> Typed, self-validating configuration objects for every stage of an experiment, plus the loader that resolves them from files, the environment and explicit overrides.

## Purpose

A training run has dozens of settings spread across corpus reading, tokenizer training, embedding hyperparameters and evaluation. Expressed as nested dictionaries, a misspelt key or an out-of-range value stays invisible until the code that reads it runs — which for an embedding hyperparameter can be an hour into a job. This package makes configuration a set of dataclasses that validate themselves at construction, so the failure lands at load time, next to the file that caused it, before any expensive work starts.

## Modules

| Module | Responsibility |
|---|---|
| `base.py` | The config dataclasses — `CorpusConfig`, `TokenizerConfig`, `EmbeddingConfig`, `EvaluationConfig` and the root `ExperimentConfig` — each validating in `__post_init__`. `ExperimentConfig` also owns the derived artefact directories and `merged`. |
| `loader.py` | Resolution from all sources: `load_config`, `save_config`, `config_from_env`, `parse_override`, and the `ENV_PREFIX` constant. |

## Key design decisions

### Dataclasses that validate in `__post_init__`

Each config checks its own invariants at construction, using the `require_*` helpers from `utils.validation`. `EmbeddingConfig` requires a positive `dimension`, `window`, `min_count` and `epochs`; `TokenizerConfig` requires `character_coverage` strictly inside `(0.0, 1.0)`. Some checks are cross-field and could not live on an individual value: `min_learning_rate` must not exceed `learning_rate`, and `min_sentence_characters` must not exceed `max_sentence_characters`.

`__post_init__` also coerces. `Path` fields arrive from YAML as strings and are re-wrapped unconditionally — re-wrapping an existing `Path` is a no-op, and testing the runtime type would look like dead code to a type checker given the declared annotation. `TokenizerConfig.model_type` accepts a string and converts it to `TokenizerModel`, raising `ConfigurationError` listing the supported values if it does not match.

### Treat instances as immutable after construction

The configs are `@dataclass(slots=True)` but **not** frozen, and this is a sharp edge worth stating plainly: dataclasses do not re-run `__post_init__` on attribute assignment. Setting `config.embedding.dimension = 0` after construction succeeds and leaves an invalid config that no check will ever catch.

Derive variants with `ExperimentConfig.merged` instead, which rebuilds through `from_dict` and therefore revalidates every nested section.

### The precedence chain

Resolution runs lowest to highest, in `load_config`:

1. **dataclass defaults** — the field defaults in `base.py`, several of which come from `common.constants`
2. **the file** — YAML or JSON, chosen by suffix; anything else raises `ConfigurationError`
3. **environment variables** — any name prefixed `QFME_`, unless `use_environment=False`
4. **explicit overrides** — the `overrides` mapping passed by the caller, typically assembled from command-line `key.path=value` arguments via `parse_override`

Each step is a chance to adjust one value without disturbing the others: a deployment changes a setting through the environment without editing a file, a developer changes one on the command line without touching the environment.

Environment names use double underscores for nesting and lowercase the remainder, so `QFME_EMBEDDING__DIMENSION=256` becomes `{"embedding": {"dimension": 256}}`. Values are parsed as YAML scalars rather than left as strings, which is why `256` arrives as an `int` and `true` as a `bool` — a string would fail the type-directed coercion in `from_primitive` or, worse, pass it and misbehave later.

### `merged` is a recursive merge, not a replacement

`_deep_merge` walks both mappings and recurses wherever both sides hold a dictionary. Overriding `embedding.dimension` therefore leaves `embedding.window`, `embedding.epochs` and the rest of the section at their existing values. A shallow `dict.update` would replace the whole `embedding` block with a one-key dictionary, silently resetting ten siblings to their defaults — a failure mode that produces a plausible-looking run with the wrong hyperparameters rather than an error.

Non-dictionary values replace outright, including lists. Overriding `tokenizer.normalizers` swaps the entire chain, which is the right behaviour for an ordered pipeline where a positional merge would be meaningless.

### `save_config` persists the resolved config next to the artefacts

`load_config` resolves four sources into one object; only that resolved object describes what actually ran. `save_config` writes it as YAML, and every pipeline run calls it alongside the artefacts it produces. Without that record a model file cannot be traced back to the settings that made it, since the original file, the environment and the command line have all moved on.

The derived properties `experiment_directory`, `tokenizer_directory` and `embedding_directory` compute artefact locations from `output_directory` and `name`, so the layout is defined once rather than reassembled by each pipeline.

## Usage

```python
from pathlib import Path
from multilingual_embedding.config import (
    ExperimentConfig, load_config, save_config, config_from_env, parse_override,
)
from multilingual_embedding.core.exceptions import MultilingualEmbeddingError

work = Path("cfgdemo"); work.mkdir(exist_ok=True)
(work / "experiment.yaml").write_text(
    "name: hindi-trial\nembedding:\n  dimension: 256\n", encoding="utf-8"
)

config = load_config(work / "experiment.yaml", use_environment=False)
print("name:", config.name)
print("dimension:", config.embedding.dimension, "| window kept at default:", config.embedding.window)
print("tokenizer model:", repr(config.tokenizer.model_type))
print("embedding directory:", config.embedding_directory)

print("from env:", config_from_env({"QFME_EMBEDDING__DIMENSION": "512", "QFME_NAME": "trial-a"}))
print("from CLI:", parse_override("embedding.epochs=10"))

variant = config.merged({"embedding": {"epochs": 10}})
print("merged epochs:", variant.embedding.epochs, "| dimension preserved:", variant.embedding.dimension)
print("saved:", save_config(variant, work / "resolved.yaml").name)

for label, data in [
    ("invalid value", {"embedding": {"dimension": 0}}),
    ("misspelt key", {"embedding": {"dimensions": 256}}),
]:
    try:
        ExperimentConfig.from_dict(data)
    except MultilingualEmbeddingError as error:
        print(f"{label}: {type(error).__name__}: {error}")
```

Output:

```
name: hindi-trial
dimension: 256 | window kept at default: 5
tokenizer model: <TokenizerModel.UNIGRAM: 'unigram'>
embedding directory: artifacts/hindi-trial/embedding
from env: {'embedding': {'dimension': 512}, 'name': 'trial-a'}
from CLI: {'embedding': {'epochs': 10}}
merged epochs: 10 | dimension preserved: 256
saved: resolved.yaml
invalid value: ValidationError: dimension must be > 0 (name='dimension', value=0)
misspelt key: SerializationError: Unknown fields for target type (known=['batch_size', 'dimension', 'epochs', 'learning_rate', 'min_count', 'min_learning_rate', 'negative_samples', 'seed', 'subsample_threshold', 'window', 'workers'], target='EmbeddingConfig', unknown=['dimensions'])
```

Two things to note in the last two lines. Setting `dimension` fails as `ValidationError`, not `ConfigurationError` — the check comes from `utils.validation`, and although `load_config`'s docstring says `ConfigurationError`, what propagates is whatever the failing helper raised. Catch `MultilingualEmbeddingError` if you want both. And a misspelt key produces `SerializationError` naming every accepted field, because `from_primitive` refuses unknown keys rather than dropping them.

## Dependencies

`config` is the fourth layer. It may import from `common`, `core` and `utils`, and does: `common.constants` and `common.enums` for defaults and the tokenizer enum, `core.exceptions` and `core.logging`, `utils.serialization` for the primitive round trip, `utils.validation` for the checks, and `utils.io` for reading and writing files.

It **must not** import from `corpus`, `vocabulary`, `tokenizer`, `embedding`, `evaluation` or `pipelines` — every one of which imports it. This is what lets `TokenizerConfig.normalizers` hold `{"type": "nfkc"}` mappings rather than normalizer instances: the specification is resolved later, by the tokenizer layer, through `core.factory` and a registry. Naming an implementation without importing it is precisely the problem the registry exists to solve.

`tests/test_architecture.py` enforces the rule by parsing every module and rejecting imports of a layer at or above the importing one.

## Tests

Tests live in one file, `tests/config/test_config.py`, holding 30 tests across five classes:

| Class | Tests | Coverage |
|---|---|---|
| `TestCorpusConfig` | 3 | String-to-`Path` coercion, inverted length bounds, unsupported format. |
| `TestTokenizerConfig` | 4 | String-to-enum coercion, unknown model type, `character_coverage` bounds, positive `vocab_size`. |
| `TestEmbeddingConfig` | 8 | Defaults validating, six parametrised invalid-value cases, and the `min_learning_rate` / `learning_rate` cross-field rule. |
| `TestExperimentConfig` | 5 | Derived directories, `to_dict` / `from_dict` round trip, `merged` being deep, `merged` revalidating, empty name rejected. |
| `TestLoader` | 10 | YAML loading, defaults when no path is given, unsupported extension, non-mapping file, override precedence over the file, environment nesting, environment values arriving typed rather than as strings, `parse_override` and its `=` requirement, and a `save_config` / `load_config` round trip preserving a nested value. |

Run them with `.venv/bin/python -m pytest tests/config -q`.
