"""
End to end training pipeline.

Runs the full lifecycle from raw text to a searchable embedding model:

    corpus -> tokenizer -> vocabulary -> embeddings -> evaluation -> report

The pipeline owns orchestration only. Every stage is implemented in its
own package and is usable independently; this module decides the order,
threads artefacts between stages and records what happened.

Two properties matter for a training run to be trustworthy:

*Streaming.* The corpus is never fully materialised. Tokenizer training,
vocabulary construction and every embedding epoch pull from a
re-iterable :class:`SentenceStream`, so corpus size is bounded by disk
rather than memory.

*Reproducibility.* The resolved configuration is written next to the
artefacts it produced. A model file whose settings cannot be recovered
is not reproducible, however good its metrics are.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from multilingual_embedding.config.base import ExperimentConfig
from multilingual_embedding.config.loader import save_config
from multilingual_embedding.core.exceptions import ConfigurationError
from multilingual_embedding.core.logging import get_logger
from multilingual_embedding.corpus.iterator import SentenceStream
from multilingual_embedding.corpus.loader import stream_documents, stream_sentences
from multilingual_embedding.corpus.statistics import CorpusStatistics, StatisticsAccumulator
from multilingual_embedding.embedding.matrix import EmbeddingMatrix
from multilingual_embedding.embedding.word2vec import Word2Vec
from multilingual_embedding.evaluation.embedding_eval import (
    EmbeddingEvaluator,
    EmbeddingMetrics,
    SimilarityPair,
    load_similarity_dataset,
    sample_neighbour_probes,
)
from multilingual_embedding.evaluation.report import EvaluationReport
from multilingual_embedding.evaluation.tokenizer_eval import (
    TokenizerEvaluator,
    TokenizerMetrics,
    language_fairness,
)
from multilingual_embedding.tokenizer.tokenizer import SentencePieceTokenizer
from multilingual_embedding.tokenizer.trainer import SentencePieceTrainerAdapter
from multilingual_embedding.utils.filesystem import ensure_directory

__all__ = ["TrainingPipeline", "TrainingResult", "train"]

_logger = get_logger(__name__)


@dataclass(slots=True)
class TrainingResult:
    """
    Everything one training run produced.

    Attributes
    ----------
    config:
        The resolved configuration used.

    corpus_statistics:
        Statistics over the filtered corpus actually trained on, not the
        raw source.

    tokenizer, matrix:
        The trained components, held in memory.

    tokenizer_metrics, tokenizer_metrics_by_language, embedding_metrics:
        Evaluation results.

    report:
        The assembled report.

    experiment_directory, tokenizer_directory, embedding_directory,
    report_directory:
        Where the artefacts were written.
    """

    config: ExperimentConfig

    corpus_statistics: CorpusStatistics

    tokenizer: SentencePieceTokenizer

    matrix: EmbeddingMatrix

    tokenizer_metrics: TokenizerMetrics

    embedding_metrics: EmbeddingMetrics

    report: EvaluationReport

    experiment_directory: Path

    tokenizer_directory: Path

    embedding_directory: Path

    report_directory: Path

    tokenizer_metrics_by_language: dict[str, TokenizerMetrics] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """Return a compact summary suitable for logging or printing."""

        return {
            "name": self.config.name,
            "documents": self.corpus_statistics.document_count,
            "sentences": self.corpus_statistics.sentence_count,
            "vocabulary_size": len(self.matrix),
            "dimension": self.matrix.dimension,
            "characters_per_token": round(self.tokenizer_metrics.characters_per_token, 3),
            "unknown_rate": round(self.tokenizer_metrics.unknown_rate, 6),
            "experiment_directory": str(self.experiment_directory),
        }


class TrainingPipeline:
    """
    Orchestrates a full training run.

    Parameters
    ----------
    config:
        The experiment configuration. Its ``corpus.source`` must be set.

    Example
    -------
    ::

        config = load_config("experiments/hindi.yaml")

        result = TrainingPipeline(config).run()

        print(result.summary())
    """

    __slots__ = ("_config",)

    def __init__(self, config: ExperimentConfig) -> None:
        if config.corpus.source is None:
            raise ConfigurationError(
                "Cannot train without a corpus source",
                experiment=config.name,
            )

        self._config = config

    @property
    def config(self) -> ExperimentConfig:
        """The configuration this pipeline runs."""

        return self._config

    def run(self, *, evaluate: bool = True) -> TrainingResult:
        """
        Execute every stage and write the artefacts.

        Parameters
        ----------
        evaluate:
            Whether to run the evaluation stage. Disabling it skips the
            extra corpus passes that scoring requires, which is useful
            when only the trained model is wanted.
        """

        config = self._config

        experiment_directory = ensure_directory(config.experiment_directory)

        _logger.info(
            "Starting training run",
            extra={"experiment": config.name, "directory": str(experiment_directory)},
        )

        # The resolved configuration is persisted first, so that an
        # interrupted run still leaves a record of what was attempted.
        save_config(config, experiment_directory / "config.yaml")

        statistics = self._collect_statistics()

        sentences = stream_sentences(config.corpus)

        tokenizer = self._train_tokenizer(sentences)

        matrix = self._train_embeddings(sentences, tokenizer)

        report = EvaluationReport(name=config.name, config=config.to_dict())

        report.corpus = statistics.to_dict()

        tokenizer_metrics = TokenizerMetrics()

        tokenizer_by_language: dict[str, TokenizerMetrics] = {}

        embedding_metrics = EmbeddingMetrics()

        if evaluate:
            tokenizer_metrics, tokenizer_by_language = self._evaluate_tokenizer(
                sentences, tokenizer
            )

            embedding_metrics = self._evaluate_embeddings(matrix)

            report.tokenizer = tokenizer_metrics.to_dict()

            report.tokenizer_by_language = {
                language: metrics.to_dict() for language, metrics in tokenizer_by_language.items()
            }

            if tokenizer_by_language:
                report.tokenizer["language_fairness"] = language_fairness(tokenizer_by_language)

            report.embedding = embedding_metrics.to_dict()
        else:
            report.notes.append("Evaluation was skipped at the caller's request.")

        report_directory = report.save(config.evaluation.report_directory / config.name)

        _logger.info("Training run complete", extra={"experiment": config.name})

        return TrainingResult(
            config=config,
            corpus_statistics=statistics,
            tokenizer=tokenizer,
            matrix=matrix,
            tokenizer_metrics=tokenizer_metrics,
            tokenizer_metrics_by_language=tokenizer_by_language,
            embedding_metrics=embedding_metrics,
            report=report,
            experiment_directory=experiment_directory,
            tokenizer_directory=config.tokenizer_directory,
            embedding_directory=config.embedding_directory,
            report_directory=report_directory,
        )

    # ------------------------------------------------------------------
    # Stages
    # ------------------------------------------------------------------

    def _collect_statistics(self) -> CorpusStatistics:
        """
        Describe the corpus that will actually be trained on.

        Statistics are gathered after filtering and deduplication, so
        the figures describe the training input rather than the raw
        source.
        """

        accumulator = StatisticsAccumulator()

        accumulator.extend(stream_documents(self._config.corpus))

        statistics = accumulator.result()

        if statistics.sentence_count == 0:
            raise ConfigurationError(
                "Corpus produced no sentences after filtering",
                source=str(self._config.corpus.source),
                minimum_characters=self._config.corpus.min_sentence_characters,
            )

        _logger.info("Corpus prepared", extra=statistics.to_dict()["languages"])

        return statistics

    def _train_tokenizer(self, sentences: SentenceStream) -> SentencePieceTokenizer:
        """
        Train the subword model and return a tokenizer over it.

        The tokenizer is given the same normalizer chain the trainer
        applied to the corpus, and is then saved so that the chain is
        recorded next to the model. A tokenizer reloaded by the search
        pipeline must normalize identically or it will encode queries
        into pieces the embeddings were never trained on.
        """

        trainer = SentencePieceTrainerAdapter(self._config.tokenizer)

        model_path = trainer.train(sentences, self._config.tokenizer_directory)

        tokenizer = SentencePieceTokenizer(
            model_path,
            normalizers=self._config.tokenizer.normalizers,
        )

        tokenizer.save(self._config.tokenizer_directory)

        _logger.info("Tokenizer trained", extra={"model": str(model_path)})

        return tokenizer

    def _train_embeddings(
        self,
        sentences: SentenceStream,
        tokenizer: SentencePieceTokenizer,
    ) -> EmbeddingMatrix:
        """
        Train word vectors over the tokenizer's own output.

        The embedding vocabulary is built from subword pieces rather
        than whitespace words. That is what makes the model work for
        scripts without whitespace word boundaries, where splitting on
        spaces would produce one token per sentence.
        """

        model = Word2Vec(self._config.embedding)

        matrix = model.train(sentences, tokenize=tokenizer.tokenize)

        model.save(self._config.embedding_directory)

        _logger.info(
            "Embeddings trained",
            extra={"vocabulary_size": len(matrix), "dimension": matrix.dimension},
        )

        return matrix

    def _evaluate_tokenizer(
        self,
        sentences: SentenceStream,
        tokenizer: SentencePieceTokenizer,
    ) -> tuple[TokenizerMetrics, dict[str, TokenizerMetrics]]:
        """Score the tokenizer overall and per language."""

        evaluator = TokenizerEvaluator(
            tokenize=tokenizer.tokenize,
            vocabulary_size=tokenizer.vocabulary_size,
        )

        sample = self._sampled(sentences)

        overall = evaluator.evaluate(sample)

        by_language = evaluator.evaluate_by_language(self._sentences_by_language())

        return overall, by_language

    def _evaluate_embeddings(self, matrix: EmbeddingMatrix) -> EmbeddingMetrics:
        """Score the embedding geometry."""

        evaluator = EmbeddingEvaluator(matrix, seed=self._config.seed)

        return evaluator.evaluate(
            similarity_pairs=self._similarity_pairs(),
            neighbour_probes=sample_neighbour_probes(matrix, count=8),
            top_k=self._config.evaluation.top_k,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sampled(self, sentences: SentenceStream) -> SentenceStream:
        """Apply the configured evaluation sample cap, if any."""

        sample_size = self._config.evaluation.sample_size

        return sentences.limited(sample_size) if sample_size else sentences

    def _sentences_by_language(self) -> dict[str, list[str]]:
        """
        Group sentences by their document's declared language.

        Documents with no declared language are grouped under
        ``"unknown"`` rather than dropped, so the totals still add up to
        the corpus.
        """

        grouped: dict[str, list[str]] = {}

        limit = self._config.evaluation.sample_size or None

        seen = 0

        for document in stream_documents(self._config.corpus):
            language = document.metadata.base.language or "unknown"

            for sentence in document.sentences():
                grouped.setdefault(language, []).append(sentence.text)

                seen += 1

            if limit is not None and seen >= limit:
                break

        return grouped

    def _similarity_pairs(self) -> list[SimilarityPair] | None:
        """Load the configured word similarity dataset, if one is set."""

        path = self._config.evaluation.similarity_dataset

        return load_similarity_dataset(path) if path is not None else None


def train(config: ExperimentConfig, *, evaluate: bool = True) -> TrainingResult:
    """Convenience wrapper around :class:`TrainingPipeline`."""

    return TrainingPipeline(config).run(evaluate=evaluate)
