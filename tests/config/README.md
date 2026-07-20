# tests/config

> Tests for [`multilingual_embedding.config`](../../src/multilingual_embedding/config/README.md) — typed configuration, loading and compute profiles.

**88 tests.** Run with `pytest tests/config -q`.

## Files

| File | Covers |
|---|---|
| `test_config.py` | Per-section validation, derived directories, dict round trip, deep merge, file loading, environment overrides, `--set` parsing, persistence, seed propagation, nested-section coercion, the loader's error contract |
| `test_compute_profiles.py` | `ComputeConfig` defaults and validation, `--profile` overlay semantics, error attribution, the profiles committed under `configs/` |

## What matters here

**Validation is the point of this layer.** Each config section validates itself in
`__post_init__` so that a bad setting fails at load rather than an hour into a training
run. The tests assert the specific rules: inverted length bounds, an out-of-range
character coverage, a `min_learning_rate` above `learning_rate`, an unknown tokenizer
model, an empty experiment name.

**Merging must be deep.** `test_merge_is_deep` asserts that overriding
`embedding.dimension` leaves `embedding.window` alone. A shallow merge would silently
reset every sibling field to its default — the kind of bug that produces a model that
trains fine and is quietly wrong.

**Merging must revalidate.** An override is just as capable of being invalid as a file,
so `merged()` runs the same checks rather than trusting its caller.

**Environment values must arrive typed.** `QFME_EMBEDDING__DIMENSION=64` has to become
the integer `64`, not the string `"64"`, or it fails validation for the wrong reason.
The double-underscore nesting convention is tested alongside it.

**Round tripping must be lossless.** `to_dict` → `from_dict` → `to_dict` compares equal.
This is what makes the resolved config persisted next to a model trustworthy as a record
of what produced it; a lossy round trip would make that record a lie.

**Every route into an `ExperimentConfig` must validate.** `TestNestedSectionCoercion`
exists because direct construction once stored a raw mapping verbatim: passing
`embedding={"dimension": 0}` produced a config whose `.embedding` was a plain `dict`
carrying a value validation would have rejected. Loading from YAML caught it; writing
the same thing in a notebook did not. The tests now assert coercion to the section type
on construction, with the untouched defaults preserved.

**The seed must be resolved at construction, not at use.** `EmbeddingConfig.seed` left
at `None` means "inherit the global seed"; an explicit value overrides it. Resolving
this eagerly is what lets the persisted config record the seed the run really used
rather than a `None` that has to be re-derived to interpret. `0` is tested separately,
because a falsy-value check would read a legitimate seed as absent.

**The loader raises the type it documents.** Whether a bad value was caught by a
precondition helper (`ValidationError`) or by the deserialiser is an internal detail;
from outside, the config is simply wrong. `load_config` unifies both into
`ConfigurationError` at its boundary, and `TestLoaderErrorContract` pins that — code
written against the documented contract used to catch neither. The same tests assert
each failure is attributed to the stage that caused it, file, override or environment,
and that section errors are not wrapped twice into an unreadable message.

## Compute profiles

`test_compute_profiles.py` covers the split between the experiment and the machine it
runs on. A laptop and a training box run the same code and the same experiment,
differing only in a `compute` section supplied by `--profile`.

**A GPU profile must validate on a machine with no GPU.** This is the property the whole
approach rests on, and `test_cuda_is_accepted_on_a_machine_without_cuda` is the test that
holds it. Profiles are written, diffed, reviewed and validated in CI from machines that
have no GPU at all. Checking `device` against the hardware actually present would make
that impossible and defer every typo to the training box — the one place nobody wants to
be debugging YAML.

**Defaults are laptop-shaped.** `device: auto`, `precision: fp32`, no gradient
checkpointing. A default of bf16 or a large batch would fail first on the machine least
able to explain why.

**`fp16` is rejected on purpose.** It is the plausible thing to write and it is wrong
here: it needs loss scaling, which this trainer does not do. Accepting it would train
quietly and badly, which is worse than refusing at load.

**A profile overrides only the section it names, and merges deeply.** Two tests separate
these. The first asserts a profile changes `compute` and leaves `name`, `seed` and
`embedding.dimension` alone — otherwise results stop being comparable across the two
boxes. The second asserts a profile naming one compute key does not blank the others; a
shallow replacement of the section would silently reset `batch_size` to its default,
which is a quiet way to lose a result. Explicit `--set` overrides still beat the
profile, so a batch size can be halved after an out-of-memory failure without editing a
file.

**A broken profile must name itself.** `test_a_broken_profile_names_itself` asserts the
error carries `config_stage == "profile"`. The experiment and the profile are separate
files, and pointing at the wrong one wastes real time.

**The committed profiles are loaded, not assumed correct.** `TestShippedProfiles` loads
`configs/cpu.yaml` and `configs/gpu.yaml` and asserts the pair actually differs — the GPU
profile raises the batch, which is the contrastive quality knob, uses bf16, and enables
gradient caching to afford the larger batch. A profile pair that resolved to the same
thing would be pure ceremony, and `gpu.yaml` is otherwise only ever exercised on the
machine furthest from a debugger.
