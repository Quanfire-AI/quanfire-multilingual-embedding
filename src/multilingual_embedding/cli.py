"""
Command line interface.

Installed as ``qfme``. Four subcommands cover the lifecycle::

    qfme stats    --source data/corpus.jsonl
    qfme train    --config experiments/demo.yaml
    qfme search   --experiment artifacts/demo --query "machine learning"
    qfme evaluate --experiment artifacts/demo --source data/corpus.jsonl

Every subcommand accepts ``--set key.path=value`` for ad hoc
configuration overrides, so an experiment can be varied without editing
a file:

    qfme train --config demo.yaml --set embedding.dimension=256

Exit codes: ``0`` success, ``1`` a framework error the user can act on,
``130`` interrupted. Framework errors are reported as a single readable
line rather than a traceback, since a stack trace is noise to someone
who mistyped a path. ``--verbose`` restores the traceback.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from multilingual_embedding.common.version import __version__
from multilingual_embedding.config.base import CorpusConfig, ExperimentConfig
from multilingual_embedding.config.loader import load_config, parse_override
from multilingual_embedding.core.exceptions import MultilingualEmbeddingError
from multilingual_embedding.core.logging import configure_logging, get_logger

__all__ = ["build_parser", "main"]

_logger = get_logger(__name__)

EXIT_SUCCESS = 0

EXIT_ERROR = 1

EXIT_INTERRUPTED = 130


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""

    parser = argparse.ArgumentParser(
        prog="qfme",
        description="QuanFire multilingual embedding framework",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Emit debug logs and full tracebacks",
    )

    parser.add_argument(
        "--log-format",
        choices=["text", "json"],
        default="text",
        help="Log output format (default: text)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    _add_stats_parser(subparsers)

    _add_validate_parser(subparsers)

    _add_train_parser(subparsers)

    _add_search_parser(subparsers)

    _add_evaluate_parser(subparsers)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """
    Entry point.

    Returns a process exit code rather than calling ``sys.exit``, so the
    CLI is callable from tests.
    """

    parser = build_parser()

    args = parser.parse_args(argv)

    configure_logging(
        level=logging.DEBUG if args.verbose else logging.INFO,
        log_format=args.log_format,
    )

    handlers = {
        "stats": _run_stats,
        "validate": _run_validate,
        "train": _run_train,
        "search": _run_search,
        "evaluate": _run_evaluate,
    }

    try:
        return handlers[args.command](args)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)

        return EXIT_INTERRUPTED
    except MultilingualEmbeddingError as error:
        if args.verbose:
            raise

        print(f"error: {error}", file=sys.stderr)

        return EXIT_ERROR
    except FileNotFoundError as error:
        if args.verbose:
            raise

        print(f"error: file not found: {error.filename}", file=sys.stderr)

        return EXIT_ERROR


# ----------------------------------------------------------------------
# Subcommand wiring
# ----------------------------------------------------------------------


def _add_common_config_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the arguments every config-driven subcommand shares."""

    parser.add_argument(
        "--config",
        type=Path,
        help="YAML or JSON experiment configuration",
    )

    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a configuration value, e.g. embedding.dimension=256",
    )


def _add_stats_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "stats",
        help="Report corpus statistics without training anything",
    )

    _add_common_config_arguments(parser)

    parser.add_argument("--source", type=Path, help="Corpus file or directory")

    parser.add_argument("--format", default="auto", choices=["auto", "text", "lines", "jsonl"])

    parser.add_argument("--language", help="Default language code for the corpus")

    parser.add_argument(
        "--output",
        type=Path,
        help="Write the statistics as JSON to this path",
    )


def _add_validate_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "validate",
        help="Audit a corpus for problems before training on it",
    )

    _add_common_config_arguments(parser)

    parser.add_argument("--source", type=Path, help="Corpus file or directory")

    parser.add_argument("--format", default="auto", choices=["auto", "text", "lines", "jsonl"])

    parser.add_argument("--language", help="Default language code for the corpus")

    parser.add_argument("--output", type=Path, help="Write the audit as JSON to this path")

    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero on warnings as well as errors",
    )


def _add_train_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "train",
        help="Train tokenizer and embeddings, then evaluate",
    )

    _add_common_config_arguments(parser)

    parser.add_argument("--source", type=Path, help="Corpus file or directory")

    parser.add_argument("--name", help="Experiment name")

    parser.add_argument(
        "--no-evaluate",
        action="store_true",
        help="Skip evaluation and produce only the trained artefacts",
    )


def _add_search_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "search",
        help="Search a trained experiment",
    )

    parser.add_argument(
        "--experiment",
        type=Path,
        required=True,
        help="Experiment directory produced by `qfme train`",
    )

    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Corpus to index and search over",
    )

    parser.add_argument("--query", required=True, help="Search query")

    parser.add_argument("--top-k", type=int, default=5)

    parser.add_argument("--format", default="auto", choices=["auto", "text", "lines", "jsonl"])


def _add_evaluate_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate a trained experiment against a corpus",
    )

    parser.add_argument(
        "--experiment",
        type=Path,
        required=True,
        help="Experiment directory produced by `qfme train`",
    )

    parser.add_argument("--source", type=Path, required=True, help="Corpus to evaluate on")

    parser.add_argument("--format", default="auto", choices=["auto", "text", "lines", "jsonl"])

    parser.add_argument("--output", type=Path, help="Write the report JSON here")


# ----------------------------------------------------------------------
# Handlers
# ----------------------------------------------------------------------


def _run_stats(args: argparse.Namespace) -> int:
    """Report corpus statistics."""

    from multilingual_embedding.corpus.loader import stream_documents
    from multilingual_embedding.corpus.statistics import StatisticsAccumulator

    config = _resolve_config(args)

    accumulator = StatisticsAccumulator()

    accumulator.extend(stream_documents(config.corpus))

    statistics = accumulator.result()

    payload = statistics.to_dict()

    if args.output:
        from multilingual_embedding.utils.io import write_json

        write_json(args.output, payload)

        print(f"Wrote statistics to {args.output}")
    else:
        print(json.dumps(payload, indent=2, ensure_ascii=False))

    return EXIT_SUCCESS


def _run_validate(args: argparse.Namespace) -> int:
    """
    Audit a corpus and report what is wrong with it.

    Exits non-zero when the corpus is unusable, so that a data pipeline
    can gate on it rather than discovering the problem during training.
    """

    from multilingual_embedding.corpus.audit import Severity, audit_corpus
    from multilingual_embedding.corpus.loader import stream_documents

    config = _resolve_config(args)

    audit = audit_corpus(stream_documents(config.corpus, deduplicate=False))

    if args.output:
        from multilingual_embedding.utils.io import write_json

        write_json(args.output, audit.to_dict())

        print(f"Wrote audit to {args.output}")
    else:
        print(f"documents  {audit.documents}")

        print(f"sentences  {audit.sentences}")

        print(f"languages  {', '.join(sorted(audit.languages)) or '(none declared)'}")

        print(f"scripts    {', '.join(sorted(audit.scripts)) or '(none)'}")

        if not audit.findings:
            print("\nNo problems found.")
        else:
            print()

            for finding in audit.findings:
                marker = {"error": "ERROR  ", "warning": "WARNING", "info": "INFO   "}[
                    str(finding.severity)
                ]

                print(f"{marker} {finding.message}")

                if finding.examples:
                    print(f"        e.g. {', '.join(finding.examples)}")

                if finding.remedy:
                    print(f"        -> {finding.remedy}")

    if not audit.usable:
        return EXIT_ERROR

    if args.strict and any(f.severity is Severity.WARNING for f in audit.findings):
        return EXIT_ERROR

    return EXIT_SUCCESS


def _run_train(args: argparse.Namespace) -> int:
    """Run the training pipeline."""

    from multilingual_embedding.pipelines.training import TrainingPipeline

    config = _resolve_config(args)

    if args.name:
        config = config.merged({"name": args.name})

    result = TrainingPipeline(config).run(evaluate=not args.no_evaluate)

    print(json.dumps(result.summary(), indent=2, ensure_ascii=False))

    return EXIT_SUCCESS


def _run_search(args: argparse.Namespace) -> int:
    """Index a corpus with a trained model and answer one query."""

    from multilingual_embedding.corpus.loader import stream_sentences
    from multilingual_embedding.pipelines.search import SemanticSearchPipeline

    pipeline = SemanticSearchPipeline.from_directory(args.experiment)

    corpus_config = CorpusConfig(source=args.source, format=args.format)

    indexed = pipeline.index(stream_sentences(corpus_config))

    if indexed == 0:
        print("No sentences could be indexed.", file=sys.stderr)

        return EXIT_ERROR

    hits = pipeline.search(args.query, top_k=args.top_k)

    if not hits:
        print("No results. The query may contain no in-vocabulary tokens.")

        return EXIT_SUCCESS

    print(f"Query: {args.query}")

    print(f"Indexed {indexed} sentences\n")

    for hit in hits:
        print(f"{hit.rank:>2}. [{hit.score:.4f}] {hit.text}")

    return EXIT_SUCCESS


def _run_evaluate(args: argparse.Namespace) -> int:
    """Evaluate a trained experiment against a corpus."""

    from multilingual_embedding.corpus.loader import stream_sentences
    from multilingual_embedding.embedding.matrix import EmbeddingMatrix
    from multilingual_embedding.evaluation.embedding_eval import (
        EmbeddingEvaluator,
        sample_neighbour_probes,
    )
    from multilingual_embedding.evaluation.report import EvaluationReport
    from multilingual_embedding.evaluation.tokenizer_eval import TokenizerEvaluator
    from multilingual_embedding.tokenizer.tokenizer import SentencePieceTokenizer

    experiment = Path(args.experiment)

    tokenizer = SentencePieceTokenizer.load(experiment / "tokenizer")

    matrix = EmbeddingMatrix.load(experiment / "embedding")

    corpus_config = CorpusConfig(source=args.source, format=args.format)

    tokenizer_metrics = TokenizerEvaluator(
        tokenize=tokenizer.tokenize,
        vocabulary_size=tokenizer.vocabulary_size,
    ).evaluate(stream_sentences(corpus_config))

    embedding_metrics = EmbeddingEvaluator(matrix).evaluate(
        neighbour_probes=sample_neighbour_probes(matrix, count=8),
    )

    report = EvaluationReport(name=experiment.name)

    report.tokenizer = tokenizer_metrics.to_dict()

    report.embedding = embedding_metrics.to_dict()

    if args.output:
        from multilingual_embedding.utils.io import write_json

        write_json(args.output, report.to_dict())

        print(f"Wrote report to {args.output}")
    else:
        print(report.to_markdown())

    return EXIT_SUCCESS


# ----------------------------------------------------------------------
# Configuration resolution
# ----------------------------------------------------------------------


def _resolve_config(args: argparse.Namespace) -> ExperimentConfig:
    """
    Build the experiment configuration from file, flags and overrides.

    Explicit flags such as ``--source`` are folded in as overrides so
    that precedence stays in one place: file, then environment, then
    ``--set``, then named flags.
    """

    overrides: dict[str, Any] = {}

    for assignment in args.overrides:
        _merge(overrides, parse_override(assignment))

    corpus_overrides: dict[str, Any] = {}

    if getattr(args, "source", None):
        corpus_overrides["source"] = str(args.source)

    if getattr(args, "format", None) and args.format != "auto":
        corpus_overrides["format"] = args.format

    if getattr(args, "language", None):
        corpus_overrides["language"] = args.language

    if corpus_overrides:
        _merge(overrides, {"corpus": corpus_overrides})

    config = load_config(args.config, overrides=overrides)

    if config.corpus.source is None:
        raise MultilingualEmbeddingError(
            "No corpus source configured; pass --source or set corpus.source"
        )

    return config


def _merge(target: dict[str, Any], addition: dict[str, Any]) -> None:
    """Recursively merge ``addition`` into ``target`` in place."""

    for key, value in addition.items():
        existing = target.get(key)

        if isinstance(existing, dict) and isinstance(value, dict):
            _merge(existing, value)
        else:
            target[key] = value


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
