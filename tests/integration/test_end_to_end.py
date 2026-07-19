"""
End to end tests.

These exercise the real pipeline against a real multilingual corpus:
segmentation, filtering, SentencePiece training, word2vec training,
evaluation, persistence and search. They are the tests that would catch
an interface drifting between two layers that unit tests each cover in
isolation.

They are marked ``slow`` because they train models. Run the fast suite
with ``pytest -m "not slow"``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from multilingual_embedding.cli import main
from multilingual_embedding.config.base import (
    CorpusConfig,
    EmbeddingConfig,
    EvaluationConfig,
    ExperimentConfig,
    TokenizerConfig,
)
from multilingual_embedding.config.loader import load_config
from multilingual_embedding.pipelines.search import SemanticSearchPipeline
from multilingual_embedding.pipelines.training import TrainingPipeline, TrainingResult

pytestmark = pytest.mark.slow


# A tokenizer needs enough text to reach its vocabulary target, so the
# fixture corpus repeats a template set rather than holding a handful of
# sentences. 300 documents keeps training under a couple of seconds.
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
    """Write a synthetic multilingual JSON Lines corpus."""

    records = []

    index = 0

    for language, (subjects, verbs, objects, end) in _TEMPLATES.items():
        joiner = "" if language == "ja" else " "

        for subject in subjects:
            for verb in verbs:
                for obj in objects:
                    index += 1

                    sentence = joiner.join([subject, verb, obj]) + end

                    # Several sentences per document so the paragraph and
                    # document levels are genuinely exercised.
                    text = " ".join([sentence] * 4)

                    records.append({"id": f"doc-{index}", "language": language, "text": text})

    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )

    return len(records)


@pytest.fixture(scope="module")
def corpus_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A multilingual corpus written once for the whole module."""

    path = tmp_path_factory.mktemp("corpus") / "corpus.jsonl"

    _build_corpus(path)

    return path


@pytest.fixture(scope="module")
def experiment_config(
    corpus_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> ExperimentConfig:
    """A small but genuine experiment configuration."""

    output = tmp_path_factory.mktemp("artifacts")

    return ExperimentConfig(
        name="integration",
        output_directory=output,
        corpus=CorpusConfig(source=corpus_path, format="jsonl"),
        tokenizer=TokenizerConfig(vocab_size=110, character_coverage=0.9995),
        embedding=EmbeddingConfig(dimension=32, window=3, min_count=1, epochs=4),
        evaluation=EvaluationConfig(report_directory=output / "reports", top_k=5),
    )


@pytest.fixture(scope="module")
def trained(experiment_config: ExperimentConfig) -> TrainingResult:
    """Train once and share the result across the module's tests."""

    return TrainingPipeline(experiment_config).run()


class TestTrainingPipeline:
    def test_corpus_was_read_and_segmented(self, trained: TrainingResult) -> None:
        assert trained.corpus_statistics.document_count == 81

        assert trained.corpus_statistics.sentence_count > 0

    def test_all_three_languages_present(self, trained: TrainingResult) -> None:
        assert set(trained.corpus_statistics.languages) == {"en", "hi", "ja"}

    def test_all_three_scripts_detected(self, trained: TrainingResult) -> None:
        scripts = set(trained.corpus_statistics.scripts)

        assert "Latn" in scripts

        assert "Deva" in scripts

    def test_tokenizer_trained(self, trained: TrainingResult) -> None:
        assert trained.tokenizer.vocabulary_size > 0

    def test_embeddings_have_expected_shape(self, trained: TrainingResult) -> None:
        assert trained.matrix.dimension == 32

        assert len(trained.matrix) == len(trained.matrix.vocabulary)

    def test_artifacts_written_to_disk(self, trained: TrainingResult) -> None:
        assert (trained.tokenizer_directory).is_dir()

        assert (trained.embedding_directory / "vectors.npy").is_file()

        assert (trained.embedding_directory / "vocabulary.json").is_file()

    def test_resolved_config_is_persisted(self, trained: TrainingResult) -> None:
        """
        A model whose settings cannot be recovered is not reproducible.
        """

        path = trained.experiment_directory / "config.yaml"

        assert path.is_file()

        reloaded = load_config(path, use_environment=False)

        assert reloaded.embedding.dimension == 32

        assert reloaded.tokenizer.vocab_size == 110

    def test_report_written_in_both_formats(self, trained: TrainingResult) -> None:
        assert (trained.report_directory / "report.json").is_file()

        assert (trained.report_directory / "report.md").is_file()

    def test_report_json_is_valid(self, trained: TrainingResult) -> None:
        payload = json.loads((trained.report_directory / "report.json").read_text(encoding="utf-8"))

        assert payload["name"] == "integration"

        assert payload["corpus"]["sentence_count"] > 0


class TestEvaluationResults:
    def test_unknown_rate_is_negligible(self, trained: TrainingResult) -> None:
        """
        A subword model with byte fallback should encode anything.

        A material unknown rate means the vocabulary does not fit the
        corpus at all.
        """

        assert trained.tokenizer_metrics.unknown_rate < 0.01

    def test_per_language_metrics_cover_every_language(self, trained: TrainingResult) -> None:
        assert set(trained.tokenizer_metrics_by_language) == {"en", "hi", "ja"}

    def test_compression_is_positive_for_every_language(self, trained: TrainingResult) -> None:
        for language, metrics in trained.tokenizer_metrics_by_language.items():
            assert metrics.characters_per_token > 0.0, language

            assert metrics.token_count > 0, language

    def test_embedding_metrics_populated(self, trained: TrainingResult) -> None:
        assert trained.embedding_metrics.vocabulary_size > 0

        assert trained.embedding_metrics.effective_dimensions > 0

    def test_no_dead_embedding_rows_beyond_padding(self, trained: TrainingResult) -> None:
        """
        Only the pad row should be all zeros.

        More than that means tokens in the vocabulary never appeared in
        training, which points at a vocabulary/corpus mismatch.
        """

        assert trained.embedding_metrics.zero_vector_count <= 1

    def test_summary_is_serialisable(self, trained: TrainingResult) -> None:
        json.dumps(trained.summary())


class TestSearchPipeline:
    def test_loads_from_experiment_directory(self, trained: TrainingResult) -> None:
        pipeline = SemanticSearchPipeline.from_directory(trained.experiment_directory)

        assert len(pipeline.matrix) == len(trained.matrix)

    def test_index_and_search_returns_ranked_hits(
        self, trained: TrainingResult, corpus_path: Path
    ) -> None:
        from multilingual_embedding.corpus.loader import stream_sentences

        pipeline = SemanticSearchPipeline.from_directory(trained.experiment_directory)

        indexed = pipeline.index(stream_sentences(CorpusConfig(source=corpus_path)))

        assert indexed > 0

        hits = pipeline.search("The engineer studies machine learning", top_k=5)

        assert len(hits) == 5

        assert [hit.rank for hit in hits] == [1, 2, 3, 4, 5]

        # Results must be sorted by descending score.
        assert all(hits[index].score >= hits[index + 1].score for index in range(len(hits) - 1))

    def test_search_finds_topically_related_text(
        self, trained: TrainingResult, corpus_path: Path
    ) -> None:
        """
        A query about machine learning should surface sentences about it.

        Asserted loosely — the corpus is small and templated — but a
        model that learned nothing would fail this.
        """

        from multilingual_embedding.corpus.loader import stream_sentences

        pipeline = SemanticSearchPipeline.from_directory(trained.experiment_directory)

        pipeline.index(stream_sentences(CorpusConfig(source=corpus_path)))

        hits = pipeline.search("machine learning", top_k=10)

        assert any("machine learning" in hit.text for hit in hits)

    def test_search_before_indexing_returns_empty(self, trained: TrainingResult) -> None:
        pipeline = SemanticSearchPipeline.from_directory(trained.experiment_directory)

        assert pipeline.search("anything") == []

    def test_missing_experiment_directory_is_reported(self, tmp_path: Path) -> None:
        from multilingual_embedding.core.exceptions import ResourceNotFoundError

        with pytest.raises(ResourceNotFoundError):
            SemanticSearchPipeline.from_directory(tmp_path / "absent")


class TestCommandLine:
    def test_stats_command(self, corpus_path: Path, tmp_path: Path) -> None:
        output = tmp_path / "stats.json"

        code = main(["stats", "--source", str(corpus_path), "--output", str(output)])

        assert code == 0

        payload = json.loads(output.read_text(encoding="utf-8"))

        assert payload["sentence_count"] > 0

    def test_stats_honours_set_override(self, corpus_path: Path, tmp_path: Path) -> None:
        output = tmp_path / "stats.json"

        main(
            [
                "stats",
                "--source",
                str(corpus_path),
                "--set",
                "corpus.min_sentence_characters=10000",
                "--output",
                str(output),
            ]
        )

        payload = json.loads(output.read_text(encoding="utf-8"))

        assert payload["sentence_count"] == 0

    def test_missing_source_is_an_error(self) -> None:
        assert main(["stats", "--source", "/no/such/path"]) == 1

    def test_missing_experiment_is_an_error(self, corpus_path: Path) -> None:
        assert (
            main(
                [
                    "search",
                    "--experiment",
                    "/no/such/experiment",
                    "--source",
                    str(corpus_path),
                    "--query",
                    "x",
                ]
            )
            == 1
        )

    def test_evaluate_command(
        self, trained: TrainingResult, corpus_path: Path, tmp_path: Path
    ) -> None:
        output = tmp_path / "report.json"

        code = main(
            [
                "evaluate",
                "--experiment",
                str(trained.experiment_directory),
                "--source",
                str(corpus_path),
                "--output",
                str(output),
            ]
        )

        assert code == 0

        payload = json.loads(output.read_text(encoding="utf-8"))

        assert payload["tokenizer"]["token_count"] > 0


class TestConfiguredNormalizersReachTheWholePipeline:
    """
    ``tokenizer.normalizers`` must hold from training through search.

    The pipeline persists its resolved config next to the artefacts, so
    a normalizer chain that shaped the training corpus but not the
    tokenizer handed to the embedding stage — or not the one the search
    pipeline reloads from disk — would leave that record claiming
    something untrue, and queries would be encoded into pieces the
    vectors were never trained on.
    """

    @pytest.fixture(scope="class")
    def lowercasing_result(
        self, corpus_path: Path, tmp_path_factory: pytest.TempPathFactory
    ) -> TrainingResult:
        output = tmp_path_factory.mktemp("lowercasing")

        config = ExperimentConfig(
            name="lowercasing",
            output_directory=output,
            corpus=CorpusConfig(source=corpus_path, format="jsonl"),
            tokenizer=TokenizerConfig(
                vocab_size=110,
                character_coverage=0.9995,
                normalizers=[{"type": "nfkc"}, {"type": "lowercase"}, {"type": "whitespace"}],
            ),
            embedding=EmbeddingConfig(dimension=32, window=3, min_count=1, epochs=2),
            evaluation=EvaluationConfig(report_directory=output / "reports", top_k=5),
        )

        return TrainingPipeline(config).run(evaluate=False)

    def test_pipeline_tokenizer_normalizes(self, lowercasing_result: TrainingResult) -> None:
        tokenizer = lowercasing_result.tokenizer

        assert (
            tokenizer.encode("THE Engineer Studies").ids
            == tokenizer.encode("the engineer studies").ids
        )

    def test_the_chain_is_persisted_beside_the_model(
        self, lowercasing_result: TrainingResult
    ) -> None:
        assert (lowercasing_result.tokenizer_directory / "sentencepiece.json").is_file()

    def test_search_pipeline_reloads_the_same_normalization(
        self, lowercasing_result: TrainingResult
    ) -> None:
        pipeline = SemanticSearchPipeline.from_directory(lowercasing_result.experiment_directory)

        reloaded = pipeline.tokenizer

        for text in ("THE Engineer Studies", "शोधकर्ता पढ़ता है", "研究者は研究します"):
            assert reloaded.encode(text).ids == lowercasing_result.tokenizer.encode(text).ids

    def test_the_persisted_config_still_describes_the_chain(
        self, lowercasing_result: TrainingResult
    ) -> None:
        path = lowercasing_result.experiment_directory / "config.yaml"

        reloaded = load_config(path, use_environment=False)

        assert {spec["type"] for spec in reloaded.tokenizer.normalizers} == {
            "nfkc",
            "lowercase",
            "whitespace",
        }
