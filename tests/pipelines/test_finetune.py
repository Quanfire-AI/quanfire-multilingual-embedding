"""
Contrastive fine-tuning of a from-scratch encoder, end to end.

Two things are tested, and only one needs torch.

The first is the guards: a fine-tune with no source, no pair file, or a
source directory missing its tokenizer or encoder must be refused before
anything is loaded, and a run that would save a model without declaring
where its training data came from must be refused when the config is
built. Those are pure checks and run without a training stack.

The second is the run itself — pretrain a tiny encoder, fine-tune it on a
handful of pairs, and prove the weights moved and the result is a
servable experiment directory. The corpus and pairs are toy-sized so the
whole path runs on a laptop in seconds. What it proves is wiring: that
the pretrained encoder is loaded, the contrastive schedule reaches it,
the held-out set is kept off the training set, and the fine-tuned model
lands where a serving path looks and answers queries there. It does not
assert that retrieval improved — a few pairs over a tiny transformer are
too little to promise that, and the honest verdict of such a run is a
legitimate outcome the CLI exit code already carries.
"""

from __future__ import annotations

import json
from importlib.util import find_spec
from pathlib import Path

import pytest

from multilingual_embedding.config.base import (
    ComputeConfig,
    CorpusConfig,
    ExperimentConfig,
    FinetuneConfig,
    PretrainingConfig,
    TokenizerConfig,
)
from multilingual_embedding.core.exceptions import ConfigurationError, ResourceNotFoundError
from multilingual_embedding.corpus.pairs import MinedPair
from multilingual_embedding.pipelines.finetune import FinetunePipeline, FinetuneResult

needs_neural = pytest.mark.skipif(
    find_spec("torch") is None,
    reason="requires the neural extra",
)


_TEMPLATES: dict[str, tuple[list[str], list[str], list[str], str]] = {
    "en": (
        ["The researcher", "A student", "The engineer"],
        ["studies", "explains", "teaches"],
        ["machine learning", "natural language", "the new model"],
        ".",
    ),
    "hi": (
        ["शोधकर्ता", "एक छात्र", "अभियंता"],
        ["पढ़ता है", "समझाता है", "सिखाता है"],
        ["मशीन लर्निंग", "प्राकृतिक भाषा", "नया मॉडल"],
        "।",
    ),
    "ja": (
        ["研究者は", "学生は", "技術者は"],
        ["研究します", "説明します", "教えます"],
        ["機械学習を", "自然言語を", "新しいモデルを"],
        "。",
    ),
}


def _build_corpus(path: Path) -> int:
    """A small synthetic multilingual corpus, enough to train a tokenizer."""

    records = []

    index = 0

    for language, (subjects, verbs, objects, end) in _TEMPLATES.items():
        joiner = "" if language == "ja" else " "

        for subject in subjects:
            for verb in verbs:
                for obj in objects:
                    index += 1

                    sentence = joiner.join([subject, verb, obj]) + end

                    text = " ".join([sentence] * 4)

                    records.append({"id": f"doc-{index}", "language": language, "text": text})

    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )

    return len(records)


def _build_pairs(path: Path) -> int:
    """
    Pairs whose two sides share their object but differ in subject and
    verb, so the task is solvable without being trivially lexical — the
    same shape the adaptation tests use, over the pretraining vocabulary.
    """

    records = []

    index = 0

    for language, (subjects, verbs, objects, end) in _TEMPLATES.items():
        joiner = "" if language == "ja" else " "

        for obj in objects:
            for verb in verbs:
                index += 1

                anchor = joiner.join([subjects[0], verb, obj]) + end

                positive = joiner.join([subjects[1], verbs[(verbs.index(verb) + 1) % 3], obj]) + end

                records.append(
                    MinedPair(
                        anchor=anchor,
                        positive=positive,
                        kind="adjacent",
                        document=f"doc-{index}",
                        language=language,
                        overlap=0.33,
                    ).to_record()
                )

    with path.open("wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    return len(records)


@pytest.fixture(scope="module")
def pairs_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("pairs") / "pairs.jsonl"

    _build_pairs(path)

    return path


def _pretraining_experiment(corpus: Path, output: Path) -> ExperimentConfig:
    return ExperimentConfig(
        name="source",
        seed=0,
        output_directory=output,
        corpus=CorpusConfig(source=corpus, format="jsonl"),
        tokenizer=TokenizerConfig(vocab_size=110, character_coverage=0.9995),
        pretraining=PretrainingConfig(
            dimension=32, layers=2, heads=4, max_length=32, epochs=2, learning_rate=1e-3, seed=0
        ),
        compute=ComputeConfig(device="cpu", precision="fp32", batch_size=8),
    )


@pytest.fixture(scope="module")
def source_directory(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A ``qfme pretrain`` experiment directory, produced once for the module."""

    from multilingual_embedding.pipelines.pretraining import PretrainingPipeline

    corpus = tmp_path_factory.mktemp("corpus") / "corpus.jsonl"

    _build_corpus(corpus)

    output = tmp_path_factory.mktemp("pretrain")

    result = PretrainingPipeline(_pretraining_experiment(corpus, output)).run()

    return result.experiment_directory


def _finetune_experiment(
    source: Path, pairs: Path, output: Path, **finetune: object
) -> ExperimentConfig:
    settings: dict[str, object] = {
        "source": source,
        "pairs": pairs,
        "train_pairs": 18,
        "eval_pairs": 9,
        "sample_pairs": 27,
        "epochs": 2,
        # Large on purpose: the end-to-end test needs the weights to move
        # far enough for the probe to see it, over two layers and a few
        # pairs.
        "learning_rate": 5e-3,
        "data_provenance": "synthetic",
    }

    settings.update(finetune)

    return ExperimentConfig(
        name="finetuned",
        seed=0,
        output_directory=output,
        finetune=FinetuneConfig(**settings),  # type: ignore[arg-type]
        compute=ComputeConfig(device="cpu", precision="fp32", batch_size=8),
    )


class TestGuards:
    """What is refused, and how early — none of this needs torch."""

    def test_a_missing_source_is_refused(self) -> None:
        config = ExperimentConfig(finetune=FinetuneConfig(pairs=Path("pairs.jsonl"), save=False))

        with pytest.raises(ConfigurationError, match="source"):
            FinetunePipeline(config, echo=lambda _: None).run()

    def test_a_missing_pair_file_is_refused(self) -> None:
        config = ExperimentConfig(finetune=FinetuneConfig(source=Path("model"), save=False))

        with pytest.raises(ConfigurationError, match="pairs"):
            FinetunePipeline(config, echo=lambda _: None).run()

    def test_a_source_without_a_tokenizer_is_named(self, tmp_path: Path) -> None:
        (tmp_path / "encoder").mkdir()

        config = ExperimentConfig(
            finetune=FinetuneConfig(
                source=tmp_path, pairs=Path("pairs.jsonl"), save=False
            )
        )

        with pytest.raises(ResourceNotFoundError) as caught:
            FinetunePipeline(config, echo=lambda _: None).run()

        assert caught.value.context["component"] == "tokenizer"

    def test_a_source_without_an_encoder_is_named(self, tmp_path: Path) -> None:
        (tmp_path / "tokenizer").mkdir()

        config = ExperimentConfig(
            finetune=FinetuneConfig(
                source=tmp_path, pairs=Path("pairs.jsonl"), save=False
            )
        )

        with pytest.raises(ResourceNotFoundError) as caught:
            FinetunePipeline(config, echo=lambda _: None).run()

        assert caught.value.context["component"] == "encoder"

    def test_saving_without_provenance_is_refused_when_the_config_is_built(self) -> None:
        """
        Fail closed, and fail early: a run that would write a model must
        say where its training data came from, and the refusal lands when
        the config is built rather than after the fine-tune is spent.
        """

        with pytest.raises(ConfigurationError, match="data_provenance"):
            FinetuneConfig(source=Path("model"), pairs=Path("pairs.jsonl"))

    def test_an_unconfigured_section_is_valid(self) -> None:
        """
        The default fine-tune section ships nothing, so it must not demand
        a provenance — otherwise every experiment that never fine-tunes
        would fail to construct.
        """

        assert ExperimentConfig(name="x").finetune.data_provenance == ""


@needs_neural
class TestEndToEnd:
    @pytest.fixture(scope="class")
    def result(
        self, source_directory: Path, pairs_path: Path, tmp_path_factory: pytest.TempPathFactory
    ) -> FinetuneResult:
        output = tmp_path_factory.mktemp("finetune")

        return FinetunePipeline(
            _finetune_experiment(source_directory, pairs_path, output),
            echo=lambda _: None,
        ).run()

    def test_it_scores_the_same_held_out_set_both_times(self, result: FinetuneResult) -> None:
        assert result.before.overall.queries == result.after.overall.queries

        assert result.train_examples > 0

        assert result.eval_examples > 0

    def test_the_weights_actually_move(self, result: FinetuneResult) -> None:
        """
        A full fine-tune must change the encoder; an unchanged model would
        make every score below unchanged for a reason that has nothing to
        do with the pairs.
        """

        assert result.trained

    def test_the_fine_tuned_encoder_is_written_where_serving_looks(
        self, result: FinetuneResult
    ) -> None:
        assert result.encoder_directory is not None

        assert (result.encoder_directory / "encoder.json").is_file()

        assert (result.encoder_directory.parent / "tokenizer").is_dir()

        assert (result.encoder_directory.parent / "config.yaml").is_file()

    def test_the_saved_model_serves_queries(self, result: FinetuneResult) -> None:
        """
        The loop closed twice over: a fine-tuned encoder is served straight
        from its experiment directory, the same way a pretrained one is.
        """

        from multilingual_embedding.pipelines.search import SemanticSearchPipeline

        assert result.encoder_directory is not None

        pipeline = SemanticSearchPipeline.from_directory(result.encoder_directory.parent)

        assert pipeline.matrix is None

        corpus = [
            "The researcher studies machine learning.",
            "A student explains natural language.",
            "शोधकर्ता मशीन लर्निंग पढ़ता है।",
        ]

        indexed = pipeline.index(corpus)

        assert indexed == len(corpus)

        hits = pipeline.search("The engineer teaches the new model.", top_k=3)

        assert [hit.rank for hit in hits] == [1, 2, 3]

        assert all(-1.0 <= hit.score <= 1.0 for hit in hits)

    def test_the_summary_is_serialisable(self, result: FinetuneResult) -> None:
        json.dumps(result.to_dict())

        assert isinstance(result.summary(), str)


@needs_neural
class TestMeasurementOnly:
    def test_no_save_writes_no_model(
        self, source_directory: Path, pairs_path: Path, tmp_path: Path
    ) -> None:
        """
        A measurement-only run answers 'would this help?' and ships
        nothing: it needs no provenance and leaves no encoder on disk.
        """

        output = tmp_path / "artifacts"

        result = FinetunePipeline(
            _finetune_experiment(
                source_directory,
                pairs_path,
                output,
                save=False,
                data_provenance="",
            ),
            echo=lambda _: None,
        ).run()

        assert result.encoder_directory is None

        assert not (output / "finetuned" / "encoder").exists()

        # The measurement itself still happened.
        assert result.before.overall.queries > 0


@needs_neural
class TestHeldOut:
    def test_the_held_out_set_is_not_trained_on(
        self, source_directory: Path, pairs_path: Path, tmp_path: Path
    ) -> None:
        """
        Training and evaluation are drawn from the same file, so without
        the explicit exclusion the model would be scored on pairs it just
        trained on, and every reported score would be optimistic.
        """

        pipeline = FinetunePipeline(
            _finetune_experiment(source_directory, pairs_path, tmp_path / "run", save=False),
            echo=lambda _: None,
        )

        train, held, _ = pipeline._select()

        trained_positives = {pair.positive for pair in train}

        held_positives = {pair.positive for pair in held}

        assert not (trained_positives & held_positives)


@needs_neural
class TestCommandLine:
    def test_finetune_writes_a_servable_encoder(
        self, source_directory: Path, pairs_path: Path, tmp_path: Path
    ) -> None:
        from multilingual_embedding.cli import EXIT_ERROR, EXIT_SUCCESS, main

        output = tmp_path / "artifacts"

        code = main(
            [
                "finetune",
                "--source",
                str(source_directory),
                "--pairs",
                str(pairs_path),
                "--name",
                "cli-finetune",
                "--data-provenance",
                "synthetic",
                "--train-pairs",
                "18",
                "--eval-pairs",
                "9",
                "--sample-pairs",
                "27",
                "--epochs",
                "2",
                "--learning-rate",
                "5e-3",
                "--set",
                "compute.device=cpu",
                "--set",
                "compute.batch_size=8",
                "--set",
                f"output_directory={output}",
            ]
        )

        # The verdict (helped or not) is a legitimate outcome on a toy run
        # and is not asserted; that the model was written proves the run
        # completed through saving.
        assert code in (EXIT_SUCCESS, EXIT_ERROR)

        assert (output / "cli-finetune" / "encoder" / "encoder.json").is_file()
