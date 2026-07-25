"""
From-scratch masked-language pretraining, end to end.

Two things are tested here, and only one needs torch.

The first is the guard: a pipeline asked to pretrain without a corpus
source must refuse before it builds anything, the same way the training
and adaptation pipelines do. That is a pure check and runs without a
training stack.

The second is the run itself — corpus to tokenizer to a fresh encoder to
a masked-language objective — and that it produces an encoder on disk
whose loss actually fell. The corpus is a handful of templated sentences
repeated enough for a tiny transformer to memorise, so the whole path
runs on a laptop in seconds. What it proves is wiring, not a good model:
that the config's knobs reach the trainer, that the reserved mask id does
not collide with real vocabulary, and that the artefact is written where
a serving path would look for it.
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
    PretrainingConfig,
    TokenizerConfig,
)
from multilingual_embedding.core.exceptions import ConfigurationError
from multilingual_embedding.pipelines.pretraining import PretrainingPipeline, PretrainingResult

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
    """Write a small synthetic multilingual JSON Lines corpus."""

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


@pytest.fixture(scope="module")
def corpus_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("corpus") / "corpus.jsonl"

    _build_corpus(path)

    return path


def _experiment(corpus: Path, output: Path, **pretraining: object) -> ExperimentConfig:
    """A small but genuine pretraining configuration."""

    settings: dict[str, object] = {
        "dimension": 32,
        "layers": 2,
        "heads": 4,
        "max_length": 32,
        "epochs": 3,
        "learning_rate": 1e-3,
        "mask_probability": 0.15,
        "seed": 0,
    }

    settings.update(pretraining)

    return ExperimentConfig(
        name="pretrain-test",
        seed=0,
        output_directory=output,
        corpus=CorpusConfig(source=corpus, format="jsonl"),
        tokenizer=TokenizerConfig(vocab_size=110, character_coverage=0.9995),
        pretraining=PretrainingConfig(**settings),  # type: ignore[arg-type]
        compute=ComputeConfig(device="cpu", precision="fp32", batch_size=8),
    )


class TestGuards:
    """What is refused, and how early."""

    def test_a_missing_corpus_source_is_refused(self) -> None:
        with pytest.raises(ConfigurationError, match="corpus source"):
            PretrainingPipeline(ExperimentConfig(name="x"))


@needs_neural
class TestEndToEnd:
    @pytest.fixture(scope="class")
    def result(
        self, corpus_path: Path, tmp_path_factory: pytest.TempPathFactory
    ) -> PretrainingResult:
        output = tmp_path_factory.mktemp("pretrain")

        return PretrainingPipeline(_experiment(corpus_path, output)).run()

    def test_the_corpus_was_read(self, result: PretrainingResult) -> None:
        assert result.corpus_statistics.sentence_count > 0

    def test_the_tokenizer_was_trained(self, result: PretrainingResult) -> None:
        assert result.tokenizer.vocabulary_size > 0

    def test_the_encoder_learned(self, result: PretrainingResult) -> None:
        """
        The decisive check. A tiny transformer over a few templated
        sentences should drive its masked-language loss down; a run whose
        loss did not fall means the knobs never reached the objective or
        the mask id collided with real vocabulary.
        """

        assert result.report.measurable

        assert result.report.final_loss < result.report.initial_loss

    def test_the_encoder_was_saved_where_serving_looks(
        self, result: PretrainingResult
    ) -> None:
        assert result.encoder_directory == result.config.encoder_directory

        assert (result.encoder_directory / "encoder.json").is_file()

    def test_the_mask_row_is_past_the_real_vocabulary(
        self, result: PretrainingResult
    ) -> None:
        """
        The encoder's table has exactly one row more than the tokenizer's
        vocabulary — the reserved mask id — so a real token can never
        index into it and the tokenizer can never emit it.
        """

        payload = json.loads(
            (result.encoder_directory / "encoder.json").read_text(encoding="utf-8")
        )

        assert payload["architecture"]["vocabulary_size"] == result.tokenizer.vocabulary_size + 1

    def test_the_resolved_config_is_persisted(self, result: PretrainingResult) -> None:
        assert (result.experiment_directory / "config.yaml").is_file()

    def test_the_summary_is_serialisable(self, result: PretrainingResult) -> None:
        json.dumps(result.summary())

    def test_the_pretrained_encoder_answers_queries(
        self, result: PretrainingResult
    ) -> None:
        """
        The loop closed: a run's saved encoder is served straight from its
        experiment directory. ``from_directory`` must auto-detect the
        contextual encoder — no matrix, no separable tokenizer — and index
        and rank real sentences over it. This is what makes a pretrained
        encoder a usable model rather than a file on disk.
        """

        from multilingual_embedding.pipelines.search import SemanticSearchPipeline

        pipeline = SemanticSearchPipeline.from_directory(result.experiment_directory)

        # Auto-detected as contextual: a from-scratch encoder holds no
        # per-token matrix and its tokenizer is internal to it.
        assert pipeline.matrix is None

        corpus = [
            "The researcher studies machine learning.",
            "A student explains natural language.",
            "शोधकर्ता मशीन लर्निंग पढ़ता है।",
        ]

        indexed = pipeline.index(corpus)

        assert indexed == len(corpus)

        hits = pipeline.search("The engineer teaches the new model.", top_k=3)

        # A ranked, in-range result set. Retrieval quality is not asserted:
        # an MLM-pretrained encoder without contrastive fine-tuning is a
        # weak sentence embedder, and this test proves the serving path,
        # not the model.
        assert [hit.rank for hit in hits] == [1, 2, 3]

        assert all(-1.0 <= hit.score <= 1.0 for hit in hits)


@needs_neural
class TestInterruptible:
    def test_a_checkpoint_is_written_and_resumed(
        self, corpus_path: Path, tmp_path: Path
    ) -> None:
        """
        Stop after the first epoch, then resume: the schedule spans the
        full epoch count, so the resumed run continues it rather than
        restarting, and the two halves together cover every epoch.
        """

        checkpoints = tmp_path / "ckpts"

        partial = PretrainingPipeline(
            _experiment(corpus_path, tmp_path / "run", epochs=3)
        ).run(checkpoint_dir=checkpoints, stop_after_epoch=0)

        assert (checkpoints / "checkpoint.pt").is_file()

        # One epoch done — not yet measurable against itself, but recorded.
        assert len(partial.report.losses) == 1

        resumed = PretrainingPipeline(
            _experiment(corpus_path, tmp_path / "run", epochs=3)
        ).run(checkpoint_dir=checkpoints, resume_from=checkpoints)

        # The resume carries the first epoch's metric forward and finishes
        # the remaining two, so the two halves together cover all three
        # epochs rather than restarting the schedule.
        assert len(resumed.report.losses) == 3

        assert resumed.report.final_loss < resumed.report.initial_loss


@needs_neural
class TestCommandLine:
    def test_pretrain_writes_an_encoder(self, corpus_path: Path, tmp_path: Path) -> None:
        from multilingual_embedding.cli import EXIT_SUCCESS, main

        output = tmp_path / "artifacts"

        code = main(
            [
                "pretrain",
                "--source",
                str(corpus_path),
                "--name",
                "cli-pretrain",
                "--set",
                "corpus.format=jsonl",
                "--set",
                "tokenizer.vocab_size=110",
                "--set",
                "pretraining.dimension=32",
                "--set",
                "pretraining.layers=2",
                "--set",
                "pretraining.heads=4",
                "--set",
                "pretraining.epochs=3",
                "--set",
                "pretraining.max_length=32",
                "--set",
                "compute.device=cpu",
                "--set",
                "compute.batch_size=8",
                "--set",
                f"output_directory={output}",
            ]
        )

        assert code == EXIT_SUCCESS

        assert (output / "cli-pretrain" / "encoder" / "encoder.json").is_file()
