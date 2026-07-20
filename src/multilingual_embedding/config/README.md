# config

> Typed, self-validating configuration objects for every stage of an experiment, plus the loader that resolves them from files, a compute profile, the environment and explicit overrides.

## Purpose

A training run has dozens of settings spread across corpus reading, tokenizer training, embedding hyperparameters and evaluation. Expressed as nested dictionaries, a misspelt key or an out-of-range value stays invisible until the code that reads it runs — which for an embedding hyperparameter can be an hour into a job. This package makes configuration a set of dataclasses that validate themselves at construction, so the failure lands at load time, next to the file that caused it, before any expensive work starts.

## Modules

| Module | Responsibility |
|---|---|
| `base.py` | The config dataclasses — `CorpusConfig`, `TokenizerConfig`, `EmbeddingConfig`, `EvaluationConfig`, `ComputeConfig` and the root `ExperimentConfig` — each validating in `__post_init__`. `ExperimentConfig` also owns the derived artefact directories and `merged`. |
| `loader.py` | Resolution from all sources: `load_config`, `save_config`, `config_from_env`, `parse_override`, and the `ENV_PREFIX` constant. |

## Key design decisions

### Dataclasses that validate in `__post_init__`

Each config checks its own invariants at construction, using the `require_*` helpers from `utils.validation`. `EmbeddingConfig` requires a positive `dimension`, `window`, `min_count` and `epochs`; `TokenizerConfig` requires `character_coverage` strictly inside `(0.0, 1.0)`. Some checks are cross-field and could not live on an individual value: `min_learning_rate` must not exceed `learning_rate`, and `min_sentence_characters` must not exceed `max_sentence_characters`.

`__post_init__` also coerces. `Path` fields arrive from YAML as strings and are re-wrapped unconditionally — re-wrapping an existing `Path` is a no-op, and testing the runtime type would look like dead code to a type checker given the declared annotation. `TokenizerConfig.model_type` accepts a string and converts it to `TokenizerModel`, raising `ConfigurationError` listing the supported values if it does not match.

### Treat instances as immutable after construction

The configs are `@dataclass(slots=True)` but **not** frozen, and this is a sharp edge worth stating plainly: dataclasses do not re-run `__post_init__` on attribute assignment. Setting `config.embedding.dimension = 0` after construction succeeds and leaves an invalid config that no check will ever catch.

Derive variants with `ExperimentConfig.merged` instead, which rebuilds through `from_dict` and therefore revalidates every nested section.

### `ComputeConfig` separates the machine from the experiment

The sections above describe an experiment: which corpus, which vocabulary size, which hyperparameters. `ComputeConfig` describes the machine it happens to run on — `device`, `precision`, `batch_size` and `gradient_checkpoint_chunk`. The split exists because those two things change for entirely different reasons. A hyperparameter changes when the science changes; a device or a batch size changes when you move the same run from a development machine to a GPU box.

Without the split the two are entangled in one file, and moving a run to different hardware means editing the experiment. That edit is indistinguishable in a diff from a change of intent, and the branch that ran on the laptop is no longer the branch that ran on the GPU — so a result cannot be attributed to either. With the split, the experiment file is committed once and the machine is supplied separately:

```
qfme train --config experiments/indic.yaml --profile configs/gpu.yaml
```

`load_config`'s `profile` argument reads that second file and deep-merges it over the first, so a profile naming only `compute.batch_size` leaves every other compute setting, and every other section, untouched. Nothing in the loader restricts a profile to the `compute` section, but keeping it to that section is what makes the arrangement mean anything: the moment a profile carries a hyperparameter, the two files are both describing the experiment again. Two profiles ship in `configs/` — `cpu.yaml` and `gpu.yaml`.

Nothing in this section changes what is computed, only how much of it fits and how fast it runs, with one exception worth stating: `batch_size` sets how many in-batch negatives each query is contrasted against in contrastive training, so it does affect the result. That is why the resolved config, `compute` section included, is persisted next to the artefacts.

There is deliberately no `workers` field. Training is single-process, so nothing would read one, and a config field that silently does nothing is worse than a missing one — it reads as a tuning knob and invites someone to spend an afternoon turning it. It belongs here the day a data loader does.

### Devices are validated by shape, not by availability

`ComputeConfig.__post_init__` checks that `device` is one of `auto`, `cpu`, `cuda` or `mps` — splitting on `:` first, so `cuda:1` is accepted — and it deliberately does **not** ask whether this machine has the device. `device: cuda` is therefore valid on a laptop with no CUDA at all.

This is what makes the profile split usable. A GPU profile has to be written, read, diffed and validated on the machine of whoever is editing it, which is rarely the GPU box; CI has no GPU either, and `tests/config/test_compute_profiles.py` loads the shipped profiles as part of the ordinary suite. Validating against availability would make every one of those operations fail on hardware that was never going to run the job — rejecting a file that is entirely correct, for a reason that has nothing to do with the file.

An unavailable device still fails, just later, at the point where the runtime tries to use it and can say precisely what was missing. `precision` is checked the same way against `fp32` and `bf16`, since whether the hardware has native bfloat16 is likewise not a property of the file.

### The precedence chain

Resolution runs lowest to highest, in `load_config`:

1. **dataclass defaults** — the field defaults in `base.py`, several of which come from `common.constants`
2. **the file** — YAML or JSON, chosen by suffix; anything else raises `ConfigurationError`
3. **the compute profile** — the file named by `profile`, when one is given, deep-merged over the experiment
4. **environment variables** — any name prefixed `QFME_`, unless `use_environment=False`
5. **explicit overrides** — the `overrides` mapping passed by the caller, typically assembled from command-line `key.path=value` arguments via `parse_override`

Each step is a chance to adjust one value without disturbing the others: a profile changes the hardware settings without editing the experiment, a deployment changes a setting through the environment without editing a file, a developer changes one on the command line without touching the environment.

The profile sits below the environment and the command line on purpose. A profile is a committed description of a class of machine; an override is a one-off. Putting the profile above them would mean `--set compute.batch_size=8`, typed to test a hunch, was silently discarded.

### A `ConfigurationError` records the stage that produced it

`load_config` is the boundary between a user's file and the framework's internals, so each source is applied inside `_configuration_errors`, which presents `ValidationError` and `SerializationError` from the layers below as `ConfigurationError` — one exception type for "the configuration is wrong", whatever caught it. The original is preserved as `__cause__` and its structured context is carried through flattened, so it stays queryable.

Errors raised inside that block also gain `config_path` and `config_stage`, the latter being one of `file`, `profile`, `environment` or `overrides`. Provenance matters most when a profile is in play, because then two files could have supplied the offending value and the message alone does not say which to open:

```
Configuration is invalid: batch_size must be > 0 (config_path='configs/gpu.yaml', config_stage='profile', name='batch_size', value=0)
```

The annotation uses `setdefault`, so an error that already names a stage — raised by an inner load that knew better — keeps it. Note the scope: it covers building and merging, not reading. A file with an unsupported suffix fails in `_read_config_file`, before any stage begins, and carries `path` rather than `config_path`.

Environment names use double underscores for nesting and lowercase the remainder, so `QFME_EMBEDDING__DIMENSION=256` becomes `{"embedding": {"dimension": 256}}`. Values are parsed as YAML scalars rather than left as strings, which is why `256` arrives as an `int` and `true` as a `bool` — a string would fail the type-directed coercion in `from_primitive` or, worse, pass it and misbehave later.

### `merged` is a recursive merge, not a replacement

`_deep_merge` walks both mappings and recurses wherever both sides hold a dictionary. Overriding `embedding.dimension` therefore leaves `embedding.window`, `embedding.epochs` and the rest of the section at their existing values. A shallow `dict.update` would replace the whole `embedding` block with a one-key dictionary, silently resetting ten siblings to their defaults — a failure mode that produces a plausible-looking run with the wrong hyperparameters rather than an error.

Non-dictionary values replace outright, including lists. Overriding `tokenizer.normalizers` swaps the entire chain, which is the right behaviour for an ordered pipeline where a positional merge would be meaningless.

### `save_config` persists the resolved config next to the artefacts

`load_config` resolves five sources into one object; only that resolved object describes what actually ran. This is what makes the profile split safe to use: the artefacts record the `compute` section that was actually in force, so a run is never left ambiguous about which machine settings produced it. `save_config` writes it as YAML, and every pipeline run calls it alongside the artefacts it produces. Without that record a model file cannot be traced back to the settings that made it, since the original file, the environment and the command line have all moved on.

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
print("compute:", config.compute)

print("from env:", config_from_env({"QFME_EMBEDDING__DIMENSION": "512", "QFME_NAME": "trial-a"}))
print("from CLI:", parse_override("embedding.epochs=10"))

variant = config.merged({"embedding": {"epochs": 10}})
print("merged epochs:", variant.embedding.epochs, "| dimension preserved:", variant.embedding.dimension)
print("saved:", save_config(variant, work / "resolved.yaml").name)

# A profile overlays the machine settings and leaves the experiment alone.
profiled = load_config(work / "experiment.yaml", profile="configs/gpu.yaml", use_environment=False)
print("profiled compute:", profiled.compute)
print("experiment untouched:", profiled.name, profiled.embedding.dimension)

for label, data in [
    ("invalid value", {"embedding": {"dimension": 0}}),
    ("misspelt key", {"embedding": {"dimensions": 256}}),
]:
    try:
        ExperimentConfig.from_dict(data)
    except MultilingualEmbeddingError as error:
        print(f"{label}: {type(error).__name__}: {error}")

try:
    load_config(work / "experiment.yaml", overrides={"embedding": {"dimension": 0}}, use_environment=False)
except MultilingualEmbeddingError as error:
    print(f"via load_config: {type(error).__name__}: {error}")
```

Run from the repository root, so that `configs/gpu.yaml` resolves. Output:

```
name: hindi-trial
dimension: 256 | window kept at default: 5
tokenizer model: <TokenizerModel.UNIGRAM: 'unigram'>
embedding directory: artifacts/hindi-trial/embedding
compute: ComputeConfig(device='auto', precision='fp32', batch_size=16, gradient_checkpoint_chunk=0)
from env: {'embedding': {'dimension': 512}, 'name': 'trial-a'}
from CLI: {'embedding': {'epochs': 10}}
merged epochs: 10 | dimension preserved: 256
saved: resolved.yaml
profiled compute: ComputeConfig(device='auto', precision='bf16', batch_size=256, gradient_checkpoint_chunk=32)
experiment untouched: hindi-trial 256
invalid value: ValidationError: dimension must be > 0 (name='dimension', value=0)
misspelt key: SerializationError: Unknown fields for target type (known=['dimension', 'epochs', 'learning_rate', 'min_count', 'min_learning_rate', 'negative_samples', 'seed', 'subsample_threshold', 'window'], target='EmbeddingConfig', unknown=['dimensions'])
via load_config: ConfigurationError: Configuration is invalid: dimension must be > 0 (config_path='cfgdemo/experiment.yaml', config_stage='overrides', name='dimension', value=0)
```

The profile lines are the whole point of the split: `compute` changes completely, `name` and `embedding.dimension` do not.

The last three lines show where the exception boundary sits. Constructing a config directly through `ExperimentConfig.from_dict` raises whatever the failing internal raised — `ValidationError` from a `utils.validation` helper, or `SerializationError` from `from_primitive`, which refuses unknown keys rather than dropping them and names every accepted field when it does. Going through `load_config` normalises both to `ConfigurationError`, annotated with the path and the stage. Catch `MultilingualEmbeddingError` to cover either route.

## Dependencies

`config` is the fourth layer. It may import from `common`, `core` and `utils`, and does: `common.constants` and `common.enums` for defaults and the tokenizer enum, `core.exceptions` and `core.logging`, `utils.serialization` for the primitive round trip, `utils.validation` for the checks, and `utils.io` for reading and writing files.

It **must not** import from `corpus`, `vocabulary`, `tokenizer`, `embedding`, `evaluation` or `pipelines` — every one of which imports it. This is what lets `TokenizerConfig.normalizers` hold `{"type": "nfkc"}` mappings rather than normalizer instances: the specification is resolved later, by the tokenizer layer, through `core.factory` and a registry. Naming an implementation without importing it is precisely the problem the registry exists to solve.

`tests/test_architecture.py` enforces the rule by parsing every module and rejecting imports of a layer at or above the importing one.

## Tests

Tests live in `tests/config/`, 88 in total across two files:

| Class | Tests | Coverage |
|---|---|---|
| `TestCorpusConfig` | 3 | String-to-`Path` coercion, inverted length bounds, unsupported format. |
| `TestTokenizerConfig` | 9 | String-to-enum coercion, unknown model type, `character_coverage` bounds, positive `vocab_size`. |
| `TestEmbeddingConfig` | 14 | Defaults validating, parametrised invalid-value cases, and the `min_learning_rate` / `learning_rate` cross-field rule. |
| `TestExperimentConfig` | 5 | Derived directories, `to_dict` / `from_dict` round trip, `merged` being deep, `merged` revalidating, empty name rejected. |
| `TestSeedPropagation` | 9 | Global seed inherited, an explicit embedding seed winning, `0` treated as an override rather than an absent value, the inherited seed persisted concretely, and inheritance surviving a merge in both directions. |
| `TestNestedSectionCoercion` | 15 | Mappings coerced to the section type, untouched defaults kept, invalid and unknown nested values rejected, every section covered parametrically, real instances passing through unchanged, and a non-mapping section rejected. |
| `TestLoader` | 10 | YAML loading, defaults when no path is given, unsupported extension, non-mapping file, override precedence over the file, environment nesting, environment values arriving typed rather than as strings, `parse_override` and its `=` requirement, and a `save_config` / `load_config` round trip preserving a nested value. |
| `TestLoaderErrorContract` | 9 | The wrapping guarantee: invalid values and unknown fields surfacing as `ConfigurationError`, removed settings rejected rather than ignored, the original preserved as `__cause__`, structured context surviving the wrap, and `config_stage` correctly attributing a failure to the override or environment stage without double-wrapping. |
| `TestComputeConfig` | 7 | Laptop-shaped defaults, `cuda` accepted on a machine without CUDA, unknown device and precision rejected, nested-dict coercion, a round trip, and a test asserting every setting is actually read by something. |
| `TestProfileOverlay` | 4 | A profile overriding only what it names, the merge being deep, explicit overrides still beating the profile, and a broken profile naming itself in the error. |
| `TestShippedProfiles` | 3 | Both files in `configs/` loading, and the GPU profile actually differing from the CPU one. |

`TestShippedProfiles` is the reason `device` is validated by shape: it loads `configs/gpu.yaml` on whatever machine runs the suite, none of which need a GPU.

Run them with `.venv/bin/python -m pytest tests/config -q`.
