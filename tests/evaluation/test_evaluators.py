from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from multilingual_embedding.embedding.matrix import EmbeddingMatrix
from multilingual_embedding.evaluation.embedding_eval import (
    AnalogyQuestion,
    EmbeddingEvaluator,
    SimilarityPair,
    load_analogy_dataset,
    load_similarity_dataset,
    sample_neighbour_probes,
)
from multilingual_embedding.evaluation.report import EvaluationReport
from multilingual_embedding.evaluation.tokenizer_eval import (
    TokenizerEvaluator,
    evaluate_tokenizer,
    language_fairness,
)
from multilingual_embedding.vocabulary.vocabulary import Vocabulary


class TestTokenizerEvaluator:
    def test_basic_counts(self) -> None:
        metrics = evaluate_tokenizer(str.split, ["a b c", "d e"])

        assert metrics.sentence_count == 2

        assert metrics.token_count == 5

        assert metrics.tokens_per_sentence == pytest.approx(2.5)

    def test_compression_ratio(self) -> None:
        metrics = evaluate_tokenizer(str.split, ["abcd efgh"])

        # 9 characters over 2 tokens.
        assert metrics.characters_per_token == pytest.approx(4.5)

    def test_unknown_rate(self) -> None:
        def tokenize(text: str) -> list[str]:
            return ["<unk>" if word == "x" else word for word in text.split()]

        metrics = evaluate_tokenizer(tokenize, ["a x b x"])

        assert metrics.unknown_rate == pytest.approx(0.5)

    def test_fertility_of_one_for_whitespace_split(self) -> None:
        """Whitespace splitting produces exactly one token per word."""

        assert evaluate_tokenizer(str.split, ["a b c"]).fertility == pytest.approx(1.0)

    def test_fertility_above_one_for_subword_split(self) -> None:
        metrics = evaluate_tokenizer(lambda text: list(text.replace(" ", "")), ["ab cd"])

        assert metrics.fertility == pytest.approx(2.0)

    def test_vocabulary_utilisation(self) -> None:
        metrics = evaluate_tokenizer(str.split, ["a b"], vocabulary_size=10)

        assert metrics.vocabulary_utilisation == pytest.approx(0.2)

    def test_utilisation_is_none_without_vocabulary_size(self) -> None:
        assert evaluate_tokenizer(str.split, ["a"]).vocabulary_utilisation is None

    def test_empty_input_is_all_zeros(self) -> None:
        metrics = evaluate_tokenizer(str.split, [])

        assert metrics.sentence_count == 0

        assert metrics.characters_per_token == 0.0

    def test_by_language(self) -> None:
        evaluator = TokenizerEvaluator(tokenize=str.split)

        results = evaluator.evaluate_by_language({"en": ["a b"], "hi": ["नमस्ते दुनिया"]})

        assert set(results) == {"en", "hi"}

        assert results["en"].token_count == 2

    def test_by_script_groups_correctly(self) -> None:
        evaluator = TokenizerEvaluator(tokenize=str.split)

        results = evaluator.evaluate_by_script(["hello world", "नमस्ते दुनिया"])

        assert "Latn" in results

        assert "Deva" in results

    def test_most_common_tokens(self) -> None:
        evaluator = TokenizerEvaluator(tokenize=str.split)

        evaluator.evaluate(["a a a b"])

        assert evaluator.most_common_tokens(1) == [("a", 3)]

    def test_metrics_are_serialisable(self) -> None:
        json.dumps(evaluate_tokenizer(str.split, ["a b"]).to_dict())


class TestLanguageFairness:
    def test_ratio_reports_spread(self) -> None:
        """
        The ratio is the headline multilingual fairness number.

        A value of 2 means one language needs twice as many tokens for
        the same amount of text.
        """

        metrics = {
            "en": evaluate_tokenizer(str.split, ["abcdefgh ijklmnop"]),
            "ja": evaluate_tokenizer(lambda text: list(text), ["abcdefgh ijklmnop"]),
        }

        fairness = language_fairness(metrics)

        assert fairness["maximum"] > fairness["minimum"]

        assert fairness["ratio"] > 1.0

    def test_identical_languages_have_ratio_one(self) -> None:
        metrics = {
            "a": evaluate_tokenizer(str.split, ["ab cd"]),
            "b": evaluate_tokenizer(str.split, ["ef gh"]),
        }

        assert language_fairness(metrics)["ratio"] == pytest.approx(1.0)

    def test_empty_input(self) -> None:
        assert language_fairness({})["ratio"] == 0.0


@pytest.fixture
def toy_matrix() -> EmbeddingMatrix:
    """
    A small matrix with deliberate structure.

    ``cat``/``dog`` point one way and ``car``/``bus`` another, so
    similarity assertions have a known correct answer.
    """

    vocabulary = Vocabulary.from_counter(
        {"cat": 10, "dog": 9, "car": 8, "bus": 7},
        min_count=1,
    )

    vectors = np.zeros((len(vocabulary), 4), dtype=np.float32)

    vectors[vocabulary.id_of("cat")] = [1.0, 0.1, 0.0, 0.0]

    vectors[vocabulary.id_of("dog")] = [0.9, 0.2, 0.0, 0.0]

    vectors[vocabulary.id_of("car")] = [0.0, 0.0, 1.0, 0.1]

    vectors[vocabulary.id_of("bus")] = [0.0, 0.0, 0.9, 0.2]

    return EmbeddingMatrix(vocabulary, vectors)


class TestEmbeddingEvaluator:
    def test_reports_shape(self, toy_matrix: EmbeddingMatrix) -> None:
        metrics = EmbeddingEvaluator(toy_matrix).evaluate()

        assert metrics.vocabulary_size == len(toy_matrix)

        assert metrics.dimension == 4

    def test_missing_datasets_stay_none(self, toy_matrix: EmbeddingMatrix) -> None:
        """
        An absent benchmark must not look like a failing score.
        """

        metrics = EmbeddingEvaluator(toy_matrix).evaluate()

        assert metrics.similarity_correlation is None

        assert metrics.analogy_accuracy is None

    def test_similarity_correlation_is_positive_for_aligned_judgements(
        self, toy_matrix: EmbeddingMatrix
    ) -> None:
        pairs = [
            SimilarityPair("cat", "dog", 9.0),
            SimilarityPair("cat", "car", 1.0),
            SimilarityPair("dog", "bus", 1.0),
            SimilarityPair("car", "bus", 9.0),
        ]

        correlation, coverage = EmbeddingEvaluator(toy_matrix).similarity_correlation(pairs)

        assert correlation > 0.5

        assert coverage == pytest.approx(1.0)

    def test_similarity_skips_out_of_vocabulary_pairs(self, toy_matrix: EmbeddingMatrix) -> None:
        """
        Skipping is right: scoring OOV pairs as zero would penalise a
        model merely for having a smaller vocabulary.
        """

        pairs = [
            SimilarityPair("cat", "dog", 9.0),
            SimilarityPair("absent", "missing", 5.0),
        ]

        _, coverage = EmbeddingEvaluator(toy_matrix).similarity_correlation(pairs)

        assert coverage == pytest.approx(0.5)

    def test_analogy_coverage_zero_when_terms_missing(self, toy_matrix: EmbeddingMatrix) -> None:
        questions = [AnalogyQuestion("absent", "b", "c", "d")]

        accuracy, coverage = EmbeddingEvaluator(toy_matrix).analogy_accuracy(questions)

        assert coverage == 0.0

        assert accuracy == 0.0

    def test_zero_vector_count_detects_untrained_rows(self, toy_matrix: EmbeddingMatrix) -> None:
        # The four special-token rows were never populated.
        assert EmbeddingEvaluator(toy_matrix).zero_vector_count() == 4

    def test_spectrum_returns_sane_values(self, toy_matrix: EmbeddingMatrix) -> None:
        isotropy, effective = EmbeddingEvaluator(toy_matrix).spectrum()

        assert 0.0 <= isotropy <= 1.0

        assert 1 <= effective <= 4

    def test_mean_pairwise_similarity_is_bounded(self, toy_matrix: EmbeddingMatrix) -> None:
        value = EmbeddingEvaluator(toy_matrix).mean_pairwise_similarity(sample_size=50)

        assert -1.0 <= value <= 1.0

    def test_evaluation_is_reproducible(self, toy_matrix: EmbeddingMatrix) -> None:
        first = EmbeddingEvaluator(toy_matrix, seed=7).evaluate()

        second = EmbeddingEvaluator(toy_matrix, seed=7).evaluate()

        assert first.mean_pairwise_similarity == second.mean_pairwise_similarity

    def test_neighbour_probes_recorded(self, toy_matrix: EmbeddingMatrix) -> None:
        metrics = EmbeddingEvaluator(toy_matrix).evaluate(neighbour_probes=["cat"])

        assert "cat" in metrics.neighbour_examples

    def test_unknown_probe_is_skipped(self, toy_matrix: EmbeddingMatrix) -> None:
        metrics = EmbeddingEvaluator(toy_matrix).evaluate(neighbour_probes=["absent"])

        assert metrics.neighbour_examples == {}

    def test_sample_neighbour_probes(self, toy_matrix: EmbeddingMatrix) -> None:
        probes = sample_neighbour_probes(toy_matrix, count=2)

        assert probes == ["cat", "dog"]

    def test_metrics_are_serialisable(self, toy_matrix: EmbeddingMatrix) -> None:
        json.dumps(EmbeddingEvaluator(toy_matrix).evaluate().to_dict())


class TestDatasetLoading:
    def test_similarity_dataset(self, tmp_path: Path) -> None:
        path = tmp_path / "similarity.jsonl"

        path.write_text(
            '{"word_a": "cat", "word_b": "dog", "score": 8.5}\n',
            encoding="utf-8",
        )

        pairs = load_similarity_dataset(path)

        assert pairs == [SimilarityPair("cat", "dog", 8.5)]

    def test_analogy_dataset(self, tmp_path: Path) -> None:
        path = tmp_path / "analogy.jsonl"

        path.write_text(
            '{"a": "man", "b": "king", "c": "woman", "expected": "queen"}\n',
            encoding="utf-8",
        )

        assert load_analogy_dataset(path) == [AnalogyQuestion("man", "king", "woman", "queen")]


class TestEvaluationReport:
    def test_writes_both_formats(self, tmp_path: Path) -> None:
        report = EvaluationReport(name="demo")

        report.corpus = {"document_count": 3}

        directory = report.save(tmp_path / "report")

        assert (directory / "report.json").is_file()

        assert (directory / "report.md").is_file()

    def test_markdown_contains_sections(self) -> None:
        report = EvaluationReport(name="demo")

        report.corpus = {"document_count": 3}

        report.tokenizer = {"token_count": 12}

        report.embedding = {"dimension": 32}

        markdown = report.to_markdown()

        assert "# Evaluation report: demo" in markdown

        assert "## Corpus" in markdown

        assert "| document_count | 3 |" in markdown

        assert "## Embeddings" in markdown

    def test_language_table_rendered(self) -> None:
        report = EvaluationReport(name="demo")

        report.tokenizer_by_language = {
            "en": {"sentence_count": 5, "token_count": 20, "characters_per_token": 4.0},
            "ja": {"sentence_count": 5, "token_count": 60, "characters_per_token": 1.1},
        }

        markdown = report.to_markdown()

        assert "Tokenizer efficiency by language" in markdown

        assert "| en " in markdown

        assert "| ja " in markdown

    def test_notes_rendered(self) -> None:
        report = EvaluationReport(name="demo")

        report.notes.append("Evaluation skipped.")

        assert "Evaluation skipped." in report.to_markdown()

    def test_nested_values_skipped_in_flat_section(self) -> None:
        """Nested structures belong in their own section, not the table."""

        report = EvaluationReport(name="demo")

        report.corpus = {"count": 1, "nested": {"a": 1}}

        markdown = report.to_markdown()

        assert "| count | 1 |" in markdown

        assert "nested" not in markdown

    def test_report_is_serialisable(self) -> None:
        json.dumps(EvaluationReport(name="demo").to_dict())

    def test_records_framework_version(self) -> None:
        from multilingual_embedding.common.version import __version__

        assert EvaluationReport().framework_version == __version__
