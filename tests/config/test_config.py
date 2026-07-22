from __future__ import annotations

from pathlib import Path

import pytest

from multilingual_embedding.common.constants import DEFAULT_RANDOM_SEED
from multilingual_embedding.common.enums import TokenizerModel
from multilingual_embedding.config.base import (
    ADAPTATION_DESCRIPTIONS,
    ADAPTATIONS,
    AdaptationConfig,
    CorpusConfig,
    EmbeddingConfig,
    EvaluationConfig,
    ExperimentConfig,
    TokenizerConfig,
)
from multilingual_embedding.config.loader import (
    config_from_env,
    load_config,
    parse_override,
    save_config,
)
from multilingual_embedding.core.exceptions import (
    ConfigurationError,
    SerializationError,
    ValidationError,
)


class TestCorpusConfig:
    def test_string_source_becomes_path(self) -> None:
        assert isinstance(CorpusConfig(source="data/corpus").source, Path)

    def test_inverted_length_bounds_rejected(self) -> None:
        with pytest.raises(ConfigurationError):
            CorpusConfig(min_sentence_characters=100, max_sentence_characters=10)

    def test_unsupported_format_rejected(self) -> None:
        with pytest.raises(ConfigurationError):
            CorpusConfig(format="parquet")


class TestTokenizerConfig:
    def test_string_model_type_becomes_enum(self) -> None:
        assert TokenizerConfig(model_type="bpe").model_type is TokenizerModel.BPE

    def test_unknown_model_type_rejected(self) -> None:
        with pytest.raises(ConfigurationError):
            TokenizerConfig(model_type="nonsense")

    def test_character_coverage_bounds(self) -> None:
        with pytest.raises(ValidationError):
            TokenizerConfig(character_coverage=1.5)

        with pytest.raises(ValidationError):
            TokenizerConfig(character_coverage=0.0)

    def test_full_character_coverage_is_accepted(self) -> None:
        """1.0 means "cover every character" and is legal SentencePiece."""

        assert TokenizerConfig(character_coverage=1.0).character_coverage == 1.0

    @pytest.mark.parametrize("coverage", [0.0, -0.1, 1.0001, 2.0])
    def test_coverage_outside_the_half_open_interval_rejected(self, coverage: float) -> None:
        with pytest.raises(ValidationError):
            TokenizerConfig(character_coverage=coverage)

    def test_vocab_size_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            TokenizerConfig(vocab_size=0)


class TestEmbeddingConfig:
    def test_defaults_are_valid(self) -> None:
        assert EmbeddingConfig().dimension > 0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"dimension": 0},
            {"window": -1},
            {"min_count": 0},
            {"epochs": 0},
            {"learning_rate": 0},
            {"negative_samples": -1},
        ],
    )
    def test_invalid_values_rejected(self, kwargs: dict[str, int]) -> None:
        with pytest.raises(ValidationError):
            EmbeddingConfig(**kwargs)

    def test_min_learning_rate_cannot_exceed_learning_rate(self) -> None:
        with pytest.raises(ConfigurationError):
            EmbeddingConfig(learning_rate=0.01, min_learning_rate=0.5)

    def test_negative_seed_rejected(self) -> None:
        """Consistent with ExperimentConfig.seed, which always rejected them."""

        with pytest.raises(ValidationError):
            EmbeddingConfig(seed=-1)

    def test_seed_defaults_to_the_inherit_marker(self) -> None:
        assert EmbeddingConfig().seed is None

    def test_resolved_seed_falls_back_for_a_standalone_config(self) -> None:
        """A config used outside an experiment must still train seeded."""

        assert EmbeddingConfig().resolved_seed == DEFAULT_RANDOM_SEED

    def test_resolved_seed_uses_an_explicit_value(self) -> None:
        assert EmbeddingConfig(seed=0).resolved_seed == 0

    @pytest.mark.parametrize("removed", ["batch_size", "workers"])
    def test_removed_fields_are_not_silently_accepted(self, removed: str) -> None:
        """
        The dead settings are gone, not merely ignored.

        Construction raises TypeError rather than storing a value that
        would never be read.
        """

        with pytest.raises(TypeError):
            EmbeddingConfig(**{removed: 4})  # type: ignore[arg-type]


class TestExperimentConfig:
    def test_derived_directories(self) -> None:
        config = ExperimentConfig(name="trial", output_directory=Path("out"))

        assert config.experiment_directory == Path("out/trial")

        assert config.tokenizer_directory == Path("out/trial/tokenizer")

        assert config.embedding_directory == Path("out/trial/embedding")

    def test_dict_round_trip(self) -> None:
        config = ExperimentConfig(name="trial")

        assert ExperimentConfig.from_dict(config.to_dict()).to_dict() == config.to_dict()

    def test_merge_is_deep(self) -> None:
        """Overriding one nested key must not reset its siblings."""

        config = ExperimentConfig()

        merged = config.merged({"embedding": {"dimension": 256}})

        assert merged.embedding.dimension == 256

        assert merged.embedding.window == config.embedding.window

    def test_merge_revalidates(self) -> None:
        with pytest.raises(ValidationError):
            ExperimentConfig().merged({"embedding": {"dimension": -1}})

    def test_empty_name_rejected(self) -> None:
        with pytest.raises(ConfigurationError):
            ExperimentConfig(name="  ")


class TestSeedPropagation:
    """
    The global seed must actually reach the embedding stage.

    ``EmbeddingConfig.seed`` left at ``None`` means "inherit"; an
    explicit value means "override". Both are resolved at construction
    so the persisted config records the seed the run really used.
    """

    def test_global_seed_is_inherited(self) -> None:
        assert ExperimentConfig(seed=7).embedding.seed == 7

    def test_explicit_embedding_seed_wins(self) -> None:
        config = ExperimentConfig(seed=7, embedding=EmbeddingConfig(seed=99))

        assert config.embedding.seed == 99

        assert config.seed == 7

    def test_zero_is_an_override_not_an_absent_value(self) -> None:
        """0 is a legitimate seed and must not be read as "unset"."""

        assert ExperimentConfig(seed=7, embedding=EmbeddingConfig(seed=0)).embedding.seed == 0

    def test_default_experiment_uses_the_framework_seed(self) -> None:
        assert ExperimentConfig().embedding.seed == DEFAULT_RANDOM_SEED

    def test_inherited_seed_is_persisted_concretely(self) -> None:
        """A record saying `seed: null` would not be a record of anything."""

        assert ExperimentConfig(seed=7).to_dict()["embedding"]["seed"] == 7

    def test_inheritance_survives_a_merge(self) -> None:
        merged = ExperimentConfig().merged({"seed": 11, "embedding": {"dimension": 16}})

        assert merged.embedding.seed == 11

    def test_an_explicit_embedding_seed_survives_a_global_seed_merge(self) -> None:
        base = ExperimentConfig(seed=7, embedding=EmbeddingConfig(seed=99))

        assert base.merged({"seed": 11}).embedding.seed == 99

    def test_a_merged_embedding_seed_beats_the_merged_global_seed(self) -> None:
        merged = ExperimentConfig().merged({"seed": 11, "embedding": {"seed": 99}})

        assert merged.embedding.seed == 99

    def test_the_trainer_receives_the_inherited_seed(self) -> None:
        """The propagation is worthless if the model does not use it."""

        from multilingual_embedding.embedding.word2vec import Word2Vec

        config = ExperimentConfig(seed=7)

        assert Word2Vec(config.embedding).config.resolved_seed == 7


class TestNestedSectionCoercion:
    """
    Every route to an ExperimentConfig must validate its sections.

    Direct construction used to store a raw mapping verbatim, producing
    a config whose ``.embedding`` was a ``dict`` holding values that
    validation would have rejected.
    """

    def test_mapping_is_coerced_to_the_section_type(self) -> None:
        config = ExperimentConfig(embedding={"dimension": 64})  # type: ignore[arg-type]

        assert isinstance(config.embedding, EmbeddingConfig)

        assert config.embedding.dimension == 64

    def test_coerced_section_keeps_its_untouched_defaults(self) -> None:
        config = ExperimentConfig(embedding={"dimension": 64})  # type: ignore[arg-type]

        assert config.embedding.window == EmbeddingConfig().window

    def test_invalid_nested_value_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ExperimentConfig(embedding={"dimension": 0})  # type: ignore[arg-type]

    def test_unknown_nested_field_is_rejected(self) -> None:
        with pytest.raises(SerializationError):
            ExperimentConfig(embedding={"dimensions": 64})  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ("section", "payload", "expected"),
        [
            ("corpus", {"format": "parquet"}, ConfigurationError),
            ("tokenizer", {"vocab_size": 0}, ValidationError),
            ("embedding", {"epochs": 0}, ValidationError),
            ("evaluation", {"top_k": 0}, ValidationError),
        ],
    )
    def test_every_section_is_validated(
        self,
        section: str,
        payload: dict[str, object],
        expected: type[Exception],
    ) -> None:
        with pytest.raises(expected):
            ExperimentConfig(**{section: payload})  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ("section", "expected_type"),
        [
            ("corpus", CorpusConfig),
            ("tokenizer", TokenizerConfig),
            ("embedding", EmbeddingConfig),
            ("evaluation", EvaluationConfig),
        ],
    )
    def test_every_section_accepts_a_mapping(
        self, section: str, expected_type: type[object]
    ) -> None:
        config = ExperimentConfig(**{section: {}})  # type: ignore[arg-type]

        assert isinstance(getattr(config, section), expected_type)

    def test_real_instances_pass_through_unchanged(self) -> None:
        embedding = EmbeddingConfig(dimension=64)

        assert ExperimentConfig(embedding=embedding).embedding is embedding

    def test_a_non_mapping_section_is_rejected(self) -> None:
        with pytest.raises(ConfigurationError):
            ExperimentConfig(embedding=64)  # type: ignore[arg-type]

    def test_coerced_section_participates_in_seed_inheritance(self) -> None:
        """Coercion must happen before the seed is propagated into it."""

        config = ExperimentConfig(seed=7, embedding={"dimension": 64})  # type: ignore[arg-type]

        assert config.embedding.seed == 7


class TestLoader:
    def test_load_from_yaml(self, tmp_path: Path) -> None:
        path = tmp_path / "config.yaml"

        path.write_text(
            "name: trial\nembedding:\n  dimension: 64\n",
            encoding="utf-8",
        )

        config = load_config(path, use_environment=False)

        assert config.name == "trial"

        assert config.embedding.dimension == 64

    def test_load_missing_path_uses_defaults(self) -> None:
        assert load_config(None, use_environment=False).name == "default"

    def test_unsupported_extension_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "config.toml"

        path.write_text("name = 'x'", encoding="utf-8")

        with pytest.raises(ConfigurationError):
            load_config(path, use_environment=False)

    def test_non_mapping_file_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "config.yaml"

        path.write_text("- a\n- b\n", encoding="utf-8")

        with pytest.raises(ConfigurationError):
            load_config(path, use_environment=False)

    def test_overrides_beat_file(self, tmp_path: Path) -> None:
        path = tmp_path / "config.yaml"

        path.write_text("name: from-file\n", encoding="utf-8")

        config = load_config(
            path,
            overrides={"name": "from-override"},
            use_environment=False,
        )

        assert config.name == "from-override"

    def test_env_parsing_and_nesting(self) -> None:
        overrides = config_from_env(
            {
                "QFME_NAME": "trial",
                "QFME_EMBEDDING__DIMENSION": "64",
                "UNRELATED": "ignored",
            }
        )

        assert overrides == {"name": "trial", "embedding": {"dimension": 64}}

    def test_env_values_are_typed_not_strings(self) -> None:
        overrides = config_from_env({"QFME_EMBEDDING__DIMENSION": "64"})

        assert isinstance(overrides["embedding"]["dimension"], int)

    def test_parse_override(self) -> None:
        assert parse_override("embedding.dimension=256") == {"embedding": {"dimension": 256}}

    def test_parse_override_requires_equals(self) -> None:
        with pytest.raises(ConfigurationError):
            parse_override("embedding.dimension")

    def test_save_and_reload(self, tmp_path: Path) -> None:
        config = ExperimentConfig(name="trial", embedding=EmbeddingConfig(dimension=32))

        path = tmp_path / "saved.yaml"

        save_config(config, path)

        assert load_config(path, use_environment=False).embedding.dimension == 32


class TestLoaderErrorContract:
    """
    ``load_config`` raises the type its docstring promises.

    Whether a bad value was caught by a precondition helper or by the
    deserialiser is internal; from outside the config is simply wrong.
    Code written against the documented contract used to miss both.
    """

    @staticmethod
    def _write(tmp_path: Path, body: str) -> Path:
        path = tmp_path / "config.yaml"

        path.write_text(body, encoding="utf-8")

        return path

    def test_invalid_value_raises_configuration_error(self, tmp_path: Path) -> None:
        path = self._write(tmp_path, "embedding:\n  dimension: 0\n")

        with pytest.raises(ConfigurationError):
            load_config(path, use_environment=False)

    def test_unknown_field_raises_configuration_error(self, tmp_path: Path) -> None:
        path = self._write(tmp_path, "embedding:\n  dimensions: 64\n")

        with pytest.raises(ConfigurationError):
            load_config(path, use_environment=False)

    def test_removed_settings_are_rejected_not_ignored(self, tmp_path: Path) -> None:
        """A config naming a deleted field must fail, never load silently."""

        path = self._write(tmp_path, "embedding:\n  batch_size: 32\n  workers: 4\n")

        with pytest.raises(ConfigurationError):
            load_config(path, use_environment=False)

    def test_the_original_error_is_preserved_as_cause(self, tmp_path: Path) -> None:
        path = self._write(tmp_path, "embedding:\n  dimension: 0\n")

        with pytest.raises(ConfigurationError) as caught:
            load_config(path, use_environment=False)

        assert isinstance(caught.value.__cause__, ValidationError)

    def test_structured_context_survives_wrapping(self, tmp_path: Path) -> None:
        path = self._write(tmp_path, "embedding:\n  dimension: 0\n")

        with pytest.raises(ConfigurationError) as caught:
            load_config(path, use_environment=False)

        assert caught.value.context["name"] == "dimension"

        assert caught.value.context["value"] == 0

        assert caught.value.context["config_path"] == str(path)

        assert caught.value.context["config_stage"] == "file"

    def test_unknown_field_context_names_the_offender(self, tmp_path: Path) -> None:
        path = self._write(tmp_path, "embedding:\n  dimensions: 64\n")

        with pytest.raises(ConfigurationError) as caught:
            load_config(path, use_environment=False)

        assert isinstance(caught.value.__cause__, SerializationError)

        assert caught.value.context["unknown"] == ["dimensions"]

    def test_a_bad_override_is_attributed_to_the_override_stage(self, tmp_path: Path) -> None:
        path = self._write(tmp_path, "name: trial\n")

        with pytest.raises(ConfigurationError) as caught:
            load_config(
                path,
                overrides={"embedding": {"dimension": 0}},
                use_environment=False,
            )

        assert caught.value.context["config_stage"] == "overrides"

    def test_a_bad_environment_value_is_attributed_to_the_environment(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("QFME_EMBEDDING__DIMENSION", "0")

        with pytest.raises(ConfigurationError) as caught:
            load_config(self._write(tmp_path, "name: trial\n"))

        assert caught.value.context["config_stage"] == "environment"

    def test_section_errors_are_not_double_wrapped(self, tmp_path: Path) -> None:
        """A ConfigurationError from a section's own checks passes through."""

        path = self._write(tmp_path, "corpus:\n  format: parquet\n")

        with pytest.raises(ConfigurationError) as caught:
            load_config(path, use_environment=False)

        assert caught.value.context["format"] == "parquet"


class TestAdaptationConfig:
    """
    The science half of an adaptation run.

    Everything the *machine* dictates lives in ``ComputeConfig`` and is
    swapped by ``--profile``. What is here decides what the run measures,
    so the validation is about making an uninterpretable run impossible
    to express rather than about catching type errors.
    """

    def test_a_comma_separated_string_becomes_a_tuple(self) -> None:
        """YAML has no tuple and the CLI hands over a string, so both
        arrive here and everything downstream must see one shape."""

        config = AdaptationConfig(train_kinds="adjacent,title_lead")

        assert config.train_kinds == ("adjacent", "title_lead")

    def test_spacing_around_the_commas_is_not_part_of_the_name(self) -> None:
        assert AdaptationConfig(eval_kinds=" adjacent , title_lead ").eval_kinds == (
            "adjacent",
            "title_lead",
        )

    def test_a_list_from_yaml_becomes_a_tuple(self) -> None:
        assert AdaptationConfig(train_languages=["hi", "ta"]).train_languages == ("hi", "ta")

    def test_an_empty_filter_stays_empty(self) -> None:
        """Empty means "everything", and a tuple holding one empty string
        would instead mean "a kind called nothing", which matches no pair."""

        assert AdaptationConfig(train_kinds="").train_kinds == ()

        assert AdaptationConfig(train_kinds=",,").train_kinds == ()

    def test_string_paths_become_paths(self) -> None:
        config = AdaptationConfig(pairs="pairs.jsonl.gz", save_adapter="out/adapter")

        assert config.pairs == Path("pairs.jsonl.gz")

        assert config.save_adapter == Path("out/adapter")

    def test_the_sample_defaults_to_covering_both_sets(self) -> None:
        assert AdaptationConfig(train_pairs=100, eval_pairs=20).sampled == 120

    def test_an_explicit_sample_size_wins(self) -> None:
        """
        Raising it is how a kind filter is made to stop starving the
        training set: a kind holding a sixth of the file yields a sixth
        of the sample, so ``train_pairs`` quietly stops binding and two
        runs naming different kinds differ in volume as well as in shape.
        """

        assert AdaptationConfig(train_pairs=100, eval_pairs=20, sample_pairs=900).sampled == 900

    def test_alpha_defaults_to_twice_the_rank(self) -> None:
        assert AdaptationConfig(rank=32).resolved_alpha == 64

    def test_an_explicit_alpha_wins(self) -> None:
        assert AdaptationConfig(rank=32, alpha=32).resolved_alpha == 32

    def test_every_mode_can_say_what_it_measures(self) -> None:
        """The description is printed at the top of a run and stored in
        the report, so a mode without one would ship a blank label."""

        assert set(ADAPTATION_DESCRIPTIONS) == set(ADAPTATIONS)

        for mode in ADAPTATIONS:
            assert AdaptationConfig(adaptation=mode).measures

    def test_an_unknown_mode_is_refused_with_the_list(self) -> None:
        with pytest.raises(ConfigurationError) as caught:
            AdaptationConfig(adaptation="cross-lingual")

        assert caught.value.context["supported"] == sorted(ADAPTATIONS)

    def test_unsupported_pooling_is_refused(self) -> None:
        with pytest.raises(ConfigurationError):
            AdaptationConfig(pooling="max")

    def test_lora_must_attach_somewhere(self) -> None:
        """Zero targets trains nothing and reports a delta of zero, which
        reads as a result about the corpus."""

        with pytest.raises(ConfigurationError):
            AdaptationConfig(targets="")

    @pytest.mark.parametrize(
        "settings",
        [
            {"train_pairs": 0},
            {"eval_pairs": 0},
            {"epochs": 0},
            {"learning_rate": 0.0},
            {"rank": 0},
            {"max_length": 0},
            {"sample_pairs": 0},
            {"alpha": 0},
            {"dropout": 1.5},
            {"seed": -1},
        ],
    )
    def test_impossible_settings_are_refused(self, settings: dict[str, object]) -> None:
        with pytest.raises((ConfigurationError, ValidationError)):
            AdaptationConfig(**settings)  # type: ignore[arg-type]

    def test_the_global_seed_is_inherited(self) -> None:
        assert ExperimentConfig(seed=7).adaptation.seed == 7

    def test_an_explicit_adaptation_seed_wins(self) -> None:
        config = ExperimentConfig(seed=7, adaptation=AdaptationConfig(seed=99))

        assert config.adaptation.seed == 99

    def test_a_merged_global_seed_reaches_the_section(self) -> None:
        merged = ExperimentConfig().merged({"seed": 11, "adaptation": {"rank": 8}})

        assert merged.adaptation.seed == 11

        assert merged.adaptation.rank == 8

    def test_an_explicit_adaptation_seed_survives_a_global_seed_merge(self) -> None:
        base = ExperimentConfig(seed=7, adaptation=AdaptationConfig(seed=99))

        assert base.merged({"seed": 11}).adaptation.seed == 99

    def test_a_mapping_section_is_coerced_and_validated(self) -> None:
        """Direct construction must not store a raw dict whose values
        validation would have rejected."""

        config = ExperimentConfig(adaptation={"rank": 8, "train_kinds": "adjacent"})  # type: ignore[arg-type]

        assert isinstance(config.adaptation, AdaptationConfig)

        assert config.adaptation.train_kinds == ("adjacent",)

        with pytest.raises(ConfigurationError):
            ExperimentConfig(adaptation={"pooling": "max"})  # type: ignore[arg-type]

    def test_it_round_trips_through_a_dict(self) -> None:
        config = ExperimentConfig(
            adaptation=AdaptationConfig(
                checkpoint="intfloat/multilingual-e5-small",
                pairs=Path("pairs.jsonl.gz"),
                train_kinds="adjacent",
                adaptation="task",
                eval_kinds="title_lead",
            )
        )

        restored = ExperimentConfig.from_dict(config.to_dict())

        assert restored.adaptation == config.adaptation
