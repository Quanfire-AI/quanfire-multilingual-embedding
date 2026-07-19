"""
End to end example: train a multilingual model and search it.

Runs the whole framework against the bundled six language sample corpus,
then issues the same query in several languages.

Usage::

    uv run python examples/train_and_search.py

Artefacts are written under ``artifacts/example/`` and the evaluation
report under ``reports/example/``. Both are gitignored.

The settings here are sized for a sample corpus that takes seconds to
train, not for a real one. See ``docs/configuration.md`` before reusing
them.
"""

from __future__ import annotations

import logging
from pathlib import Path

from multilingual_embedding import (
    CorpusConfig,
    EmbeddingConfig,
    EvaluationConfig,
    ExperimentConfig,
    SemanticSearchPipeline,
    TokenizerConfig,
    TrainingPipeline,
    TrainingResult,
    configure_logging,
)
from multilingual_embedding.corpus import stream_sentences

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent

CORPUS_PATH = REPOSITORY_ROOT / "data" / "sample" / "corpus.jsonl"

QUERIES: tuple[tuple[str, str], ...] = (
    ("en", "The engineer studies machine learning"),
    ("hi", "अभियंता मशीन लर्निंग पढ़ता है"),
    ("fr", "L'ingénieur étudie l'apprentissage automatique"),
    ("ta", "பொறியாளர் இயந்திர கற்றல் படிக்கிறார்"),
)


def build_config() -> ExperimentConfig:
    """
    Assemble the experiment configuration.

    ``vocab_size`` is deliberately small. SentencePiece fails outright
    when the target exceeds what the corpus can support, and the sample
    corpus is only 750 sentences.
    """

    return ExperimentConfig(
        name="example",
        seed=42,
        output_directory=REPOSITORY_ROOT / "artifacts",
        corpus=CorpusConfig(source=CORPUS_PATH, format="jsonl"),
        tokenizer=TokenizerConfig(vocab_size=300, character_coverage=0.9995),
        embedding=EmbeddingConfig(
            dimension=64,
            window=4,
            min_count=2,
            negative_samples=5,
            epochs=8,
        ),
        evaluation=EvaluationConfig(
            report_directory=REPOSITORY_ROOT / "reports",
            top_k=5,
        ),
    )


def report_corpus(result: TrainingResult) -> None:
    """Print what the corpus looked like after filtering."""

    statistics = result.corpus_statistics

    print("\n--- Corpus ---")

    print(f"documents      {statistics.document_count}")

    print(f"sentences      {statistics.sentence_count}")

    print(f"characters     {statistics.character_count}")

    print(f"languages      {', '.join(sorted(statistics.languages))}")

    print(f"scripts        {', '.join(sorted(statistics.scripts))}")


def report_tokenizer_fairness(result: TrainingResult) -> None:
    """
    Print per language tokenizer efficiency.

    This is the number that matters most for a multilingual model. A
    wide spread in characters per token means the vocabulary serves some
    languages far better than others, and the poorly served ones get
    less effective context for the same amount of text.
    """

    by_language = result.tokenizer_metrics_by_language

    if not by_language:
        return

    print("\n--- Tokenizer efficiency by language ---")

    print(f"{'lang':<6}{'chars/token':>13}{'fertility':>12}{'unknown':>10}")

    for language, metrics in sorted(by_language.items()):
        print(
            f"{language:<6}"
            f"{metrics.characters_per_token:>13.3f}"
            f"{metrics.fertility:>12.3f}"
            f"{metrics.unknown_rate:>10.4f}"
        )


def report_search(pipeline: SemanticSearchPipeline) -> None:
    """Run each query and print the ranked results."""

    for language, query in QUERIES:
        print(f"\n--- Query [{language}]: {query} ---")

        hits = pipeline.search(query, top_k=3)

        if not hits:
            print("  no results (the query has no in-vocabulary tokens)")

            continue

        for hit in hits:
            print(f"  {hit.rank}. [{hit.score:.4f}] {hit.text}")


def main() -> int:
    """Train, then search. Returns a process exit code."""

    configure_logging(level=logging.WARNING)

    if not CORPUS_PATH.is_file():
        print(f"Sample corpus not found at {CORPUS_PATH}")

        return 1

    config = build_config()

    print(f"Training '{config.name}' on {CORPUS_PATH.name} ...")

    result = TrainingPipeline(config).run()

    report_corpus(result)

    print("\n--- Model ---")

    print(f"vocabulary     {len(result.matrix)}")

    print(f"dimension      {result.matrix.dimension}")

    print(f"artefacts      {result.experiment_directory}")

    print(f"report         {result.report_directory}")

    report_tokenizer_fairness(result)

    # Reload from disk rather than reusing the in-memory objects, so the
    # example exercises the same path a deployed service would take.
    pipeline = SemanticSearchPipeline.from_directory(result.experiment_directory)

    indexed = pipeline.index(stream_sentences(config.corpus))

    print(f"\nIndexed {indexed} sentences for search.")

    report_search(pipeline)

    print("\n--- Nearest tokens to a shared term ---")

    for token, score in pipeline.similar_tokens("▁machine", top_k=5):
        print(f"  {score:.4f}  {token}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
