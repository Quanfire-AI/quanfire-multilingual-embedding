"""
Command line interface.

Installed as ``qfme``. Thirteen subcommands cover the lifecycle::

    qfme stats    --source data/corpus.jsonl
    qfme validate --source data/corpus.jsonl
    qfme extract  --dump hiwiki.xml.bz2 --output hi.jsonl.gz --language hi
    qfme extract-judgments --source data/corpora/judgments --output judgments.jsonl.gz
    qfme prepare-eval --source data/corpora/milpac --output eval/milpac.jsonl.gz
    qfme mine-pairs --source hi.jsonl.gz --output pairs/hi.jsonl.gz
    qfme mine-negatives --pairs pairs/hi.jsonl.gz --adapter models/indic-v1 \
        --output pairs/hi-hard.jsonl.gz
    qfme train    --config experiments/demo.yaml
    qfme pretrain --config experiments/demo.yaml --profile configs/gpu.yaml
    qfme finetune --config experiments/demo.yaml --profile configs/gpu.yaml
    qfme adapt    --config experiments/indic.yaml --profile configs/gpu.yaml
    qfme search   --experiment artifacts/demo --query "machine learning"
    qfme evaluate --experiment artifacts/demo --source data/corpus.jsonl
    qfme serve    --adapter models/indic-v1 --port 8000

``train`` builds the static path — tokenizer, word vectors, a matrix.
``pretrain`` builds the from-scratch neural path instead: it pretrains a
fresh contextual encoder with a masked-language objective, and is
interruptible, so a long run on a shared GPU box can be checkpointed each
epoch and resumed bit for bit. ``finetune`` is stage two of that path: it
contrastively fine-tunes a pretrained encoder into a usable sentence
embedder and writes a servable model, and it is interruptible the same
way — ``--checkpoint-dir`` and ``--resume-from`` carry a long contrastive
run across a shared box exactly as they carry a pretraining run.

Four of them exit non-zero on a result rather than on an error, so a
data pipeline can gate on them. ``validate`` audits a corpus before
anything is trained on it and fails when the corpus is unusable.
``adapt`` fails when the adapted model did not beat the checkpoint it
started from, and ``finetune`` fails when the fine-tune did not beat the
pretrained encoder it started from. ``pretrain`` fails when a multi-epoch
run's loss did not fall — the objective learned nothing, and nothing
should be fine-tuned on it. Each is a legitimate outcome and one nothing
should deploy.

Every config-driven subcommand accepts ``--set key.path=value`` for ad
hoc overrides, and ``--profile`` for the settings a machine dictates, so
one experiment runs unchanged on a laptop and a GPU box:

    qfme train --config demo.yaml --set embedding.dimension=256
    qfme train --config demo.yaml --profile configs/gpu.yaml

``adapt`` is the subcommand that most needed this. The batch size,
precision and gradient-checkpoint chunk that make a run fit on a 16 GB
card belong to the machine, not to the experiment, and keeping them in a
profile is what lets the same file describe the run in both places.

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

from multilingual_embedding.common.enums import DataProvenance
from multilingual_embedding.common.version import __version__
from multilingual_embedding.config.base import ADAPTATIONS, CorpusConfig, ExperimentConfig
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

    _add_extract_parser(subparsers)

    _add_extract_judgments_parser(subparsers)

    _add_prepare_eval_parser(subparsers)

    _add_mine_pairs_parser(subparsers)

    _add_mine_negatives_parser(subparsers)

    _add_train_parser(subparsers)

    _add_pretrain_parser(subparsers)

    _add_finetune_parser(subparsers)

    _add_adapt_parser(subparsers)

    _add_search_parser(subparsers)

    _add_evaluate_parser(subparsers)

    _add_serve_parser(subparsers)

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
        "extract": _run_extract,
        "extract-judgments": _run_extract_judgments,
        "prepare-eval": _run_prepare_eval,
        "mine-pairs": _run_mine_pairs,
        "mine-negatives": _run_mine_negatives,
        "train": _run_train,
        "pretrain": _run_pretrain,
        "finetune": _run_finetune,
        "adapt": _run_adapt,
        "search": _run_search,
        "evaluate": _run_evaluate,
        "serve": _run_serve,
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
        "--profile",
        type=Path,
        help=(
            "Compute profile merged over the configuration, e.g. "
            "configs/gpu.yaml. Carries the settings the machine dictates, "
            "so one experiment runs unchanged on a laptop and a GPU box"
        ),
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


def _add_extract_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "extract",
        help="Extract a Wikipedia dump into the corpus format",
    )

    parser.add_argument(
        "--dump",
        type=Path,
        required=True,
        help="Path to a *-pages-articles.xml.bz2 dump",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination JSON Lines file; gzipped when it ends .gz",
    )

    parser.add_argument(
        "--language",
        required=True,
        help="ISO code recorded on every article, e.g. hi",
    )

    parser.add_argument(
        "--minimum-characters",
        type=int,
        default=200,
        help="Skip articles shorter than this (default: 200)",
    )

    parser.add_argument(
        "--limit",
        type=int,
        help="Stop after this many articles, for trying a dump",
    )

    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Keep articles whose text repeats an earlier one",
    )


def _add_extract_judgments_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "extract-judgments",
        help="Extract public court-judgment PDFs into the training corpus",
        description=(
            "Read a directory of court-judgment PDFs into the corpus format "
            "for training. The judgments this reads are licensed CC BY, whose "
            "attribution-only terms permit training a shipped adapter on them "
            "— the opposite end of the wall from MILPaC, which may only be "
            "scored against. This does not OCR: a scanned judgment with no "
            "text layer is dropped and counted, not read, and whether a "
            "collection extracts cleanly is only known once it is run through "
            "`qfme validate`. Requires the `judgments` extra: "
            "uv sync --extra judgments."
        ),
    )

    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="A judgment .pdf, or a directory of them",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination JSON Lines file; gzipped when it ends .gz",
    )

    parser.add_argument(
        "--language",
        default="en",
        help=(
            "ISO code recorded on every judgment (default: en). The PDF does "
            "not state one, so it is asserted here rather than inferred"
        ),
    )

    parser.add_argument(
        "--minimum-characters",
        type=int,
        default=500,
        help=(
            "Drop documents shorter than this (default: 500). The floor "
            "catches failed extractions — a scanned page yields almost "
            "nothing — not genuinely short orders"
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        help="Stop after this many judgments, for trying a collection",
    )


def _add_prepare_eval_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "prepare-eval",
        help="Turn the MILPaC benchmark into a held-out evaluation pair file",
        description=(
            "Read the MILPaC .xlsx workbooks into a pair file for "
            "`qfme adapt --eval-pairs-file`. MILPaC is the evaluation set, "
            "held out by origin from the training corpus, and it is licensed "
            "CC BY-NC-SA — score against it, never train a shipped adapter on "
            "it. Requires the `milpac` extra: uv sync --extra milpac."
        ),
    )

    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="A MILPaC .xlsx workbook, or a directory of them",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination JSON Lines pair file; gzipped when it ends .gz",
    )

    parser.add_argument(
        "--languages",
        default="hi,ta",
        help=(
            "Comma-separated Indic target languages to keep (default: hi,ta). "
            "Widening this scores languages the model may not be adapted for"
        ),
    )

    parser.add_argument(
        "--datasets",
        help=(
            "Comma-separated MILPaC datasets to keep, e.g. Acts,IP,CCI-FAQ. "
            "Omit to keep all three"
        ),
    )


def _add_mine_pairs_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "mine-pairs",
        help="Build contrastive training pairs from a corpus",
    )

    parser.add_argument("--source", type=Path, required=True, help="Corpus file or directory")

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination JSON Lines file; gzipped when it ends .gz",
    )

    parser.add_argument("--format", default="auto", choices=["auto", "text", "lines", "jsonl"])

    parser.add_argument(
        "--max-overlap",
        type=float,
        default=1.0,
        help=(
            "Reject a pair when this share of the anchor's words already appear "
            "in the positive. 1.0 keeps everything; lower trades volume for pairs "
            "that cannot be solved by string matching (default: 1.0)"
        ),
    )

    parser.add_argument(
        "--kinds",
        default="title_lead,heading_section,adjacent",
        help="Comma-separated pair sources to mine",
    )

    parser.add_argument(
        "--report",
        type=Path,
        help="Write the mining statistics as JSON to this path",
    )


def _add_mine_negatives_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "mine-negatives",
        help="Attach hard negatives to a pair file, mined against a saved model",
    )

    parser.add_argument(
        "--pairs",
        type=Path,
        required=True,
        help="Mined pair file to read; gzip handled by extension",
    )

    parser.add_argument(
        "--adapter",
        type=Path,
        required=True,
        help=(
            "Model whose confusions are mined: a LoRA adapter directory, or a "
            "from-scratch experiment directory (with encoder/ + tokenizer/). "
            "Hard negatives are relative to a model, so this should be the model "
            "about to be retrained"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination JSON Lines file; gzipped when it ends .gz",
    )

    parser.add_argument(
        "--count",
        type=int,
        default=4,
        help="Negatives per pair (default: 4). Each one widens every training batch",
    )

    parser.add_argument(
        "--pool",
        type=int,
        default=32,
        help="Candidates examined per anchor before filtering (default: 32)",
    )

    parser.add_argument(
        "--max-similarity",
        type=float,
        default=0.95,
        help=(
            "Reject a candidate at or above this cosine score as a probable "
            "paraphrase of the positive rather than a negative (default: 0.95)"
        ),
    )

    parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.0,
        help=(
            "Reject a candidate at or below this score as no harder than the "
            "in-batch negatives training already has (default: 0.0)"
        ),
    )

    parser.add_argument(
        "--allow-same-document",
        action="store_true",
        help=(
            "Keep candidates from the anchor's own document. Off by default: two "
            "passages from one article are usually about one subject, and training "
            "against that teaches the model to reject a correct answer"
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        help="Mine only a uniform sample of this many pairs, for a trial run",
    )

    parser.add_argument(
        "--audit",
        type=Path,
        help=(
            "Write the most suspicious mined negatives here as JSON Lines, each "
            "with an unfilled is_actually_correct field. Labelling this sample by "
            "hand is the only route to a real false-negative rate; every number "
            "the run prints is a count of what a heuristic rejected"
        ),
    )

    parser.add_argument(
        "--audit-sample",
        type=int,
        default=50,
        help="How many negatives to put in the audit file (default: 50)",
    )

    parser.add_argument(
        "--report",
        type=Path,
        help="Write the mining statistics as JSON to this path",
    )

    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Refuse the network; require the base checkpoint to be cached already",
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


def _add_pretrain_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "pretrain",
        help="Pretrain a from-scratch contextual encoder with a masked-language objective",
        description=(
            "Stage one of the from-scratch neural path: train a tokenizer and "
            "pretrain a fresh transformer over it by predicting masked tokens. "
            "The `pretraining` and `compute` config sections govern the run; the "
            "static `embedding` (Word2Vec) section is ignored. The run is "
            "interruptible — pass --checkpoint-dir to write a rolling checkpoint "
            "each epoch and --resume-from to pick one back up, reproducing an "
            "uninterrupted run bit for bit. This is the run built for the GPU box."
        ),
    )

    _add_common_config_arguments(parser)

    parser.add_argument("--source", type=Path, help="Corpus file or directory")

    parser.add_argument("--name", help="Experiment name")

    parser.add_argument(
        "--data-provenance",
        choices=[str(value) for value in DataProvenance],
        help=(
            "Where the pretraining corpus came from: public, synthetic or "
            "licensed. Required — a pretrained encoder is a model trained on "
            "that text, so it carries this as a legal fact about itself. May "
            "also be set as pretraining.data_provenance in the config"
        ),
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="Write checkpoint.pt here after each epoch (omit to run without checkpointing)",
    )

    parser.add_argument(
        "--resume-from",
        type=Path,
        help=(
            "A checkpoint file, or a directory holding one, to restore before "
            "training. The corpus and schedule must match the run that wrote it"
        ),
    )

    parser.add_argument(
        "--stop-after-epoch",
        type=int,
        help=(
            "Stop after finishing this 0-based epoch index. Models an "
            "interruption; the schedule still spans the full configured epoch "
            "count, so a later --resume-from continues it unbroken"
        ),
    )


def _add_finetune_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "finetune",
        help="Contrastively fine-tune a from-scratch encoder, and score whether it helped",
        description=(
            "Stage two of the from-scratch neural path: take an encoder from "
            "`qfme pretrain` — which has only ever seen a masked-language "
            "objective and is a weak sentence embedder — and fine-tune the whole "
            "of it on mined pairs with an in-batch contrastive loss. It scores "
            "the encoder on held-out pairs before and after, so the report says "
            "whether the fine-tune helped rather than merely that it ran, and it "
            "writes a complete, servable experiment directory that `qfme search` "
            "and `qfme serve` load directly. Every flag below overrides the "
            "`finetune` section of the configuration."
        ),
    )

    _add_common_config_arguments(parser)

    parser.add_argument("--name", help="Experiment name for the fine-tuned model")

    parser.add_argument(
        "--source",
        type=Path,
        help="A `qfme pretrain` experiment directory, holding tokenizer/ and encoder/",
    )

    parser.add_argument("--pairs", type=Path, help="Mined pair file to train on")

    parser.add_argument(
        "--eval-pairs-file",
        type=Path,
        help=(
            "Score against this file instead of --pairs. Hold it fixed to "
            "compare fine-tunes: without it, a run that changes what it trains "
            "on also changes what it is judged by"
        ),
    )

    parser.add_argument("--train-pairs", type=int, help="Pairs to train on")

    parser.add_argument("--eval-pairs", type=int, help="Pairs to score against")

    parser.add_argument(
        "--sample-pairs",
        type=int,
        help=(
            "Pairs to draw from the file before filtering by kind. Defaults to "
            "--train-pairs plus --eval-pairs, which is wrong when a kind filter "
            "is set: a kind holding a sixth of the file yields a sixth of the "
            "sample, and --train-pairs stops binding"
        ),
    )

    parser.add_argument("--train-kinds", help="Comma-separated pair kinds to train on")

    parser.add_argument("--eval-kinds", help="Comma-separated pair kinds to score on")

    parser.add_argument("--train-languages", help="Comma-separated languages to train on")

    parser.add_argument("--eval-languages", help="Comma-separated languages to score on")

    parser.add_argument("--epochs", type=int)

    parser.add_argument("--learning-rate", type=float)

    parser.add_argument("--temperature", type=float, help="Contrastive softmax temperature")

    parser.add_argument(
        "--data-provenance",
        choices=[str(value) for value in DataProvenance],
        help=(
            "Where the training data came from: public, synthetic or licensed. "
            "Required unless --no-save — a saved model carries this as a legal "
            "fact about itself, and it has no default so it cannot be answered "
            "by omission"
        ),
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
        help=(
            "Measure only: score before and after without writing the "
            "fine-tuned encoder. A run that answers 'would this help?' and "
            "ships nothing, so it needs no data-provenance declaration"
        ),
    )

    parser.add_argument("--output", type=Path, help="Write the full comparison as JSON")

    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help=(
            "Write checkpoint.pt here after each contrastive epoch, so a long "
            "run on a shared GPU box can be resumed (omit to run without "
            "checkpointing)"
        ),
    )

    parser.add_argument(
        "--resume-from",
        type=Path,
        help=(
            "A checkpoint file, or a directory holding one, to restore before "
            "fine-tuning. The pair set and schedule must match the run that wrote it"
        ),
    )

    parser.add_argument(
        "--stop-after-epoch",
        type=int,
        help=(
            "Stop after finishing this 0-based epoch index. Models an "
            "interruption; the schedule still spans the full configured epoch "
            "count, so a later --resume-from continues it unbroken"
        ),
    )


def _add_adapt_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "adapt",
        help="Adapt a published checkpoint with LoRA, and score whether it helped",
        description=(
            "Score a published checkpoint on held-out pairs, fine-tune it with "
            "LoRA, and score it again on the same pairs. The published model is "
            "the baseline, because it is the only honest one. Every flag below "
            "overrides the `adaptation` section of the configuration, so a run "
            "assembled on the command line can be pinned to a file afterwards "
            "and reproduced."
        ),
    )

    _add_common_config_arguments(parser)

    parser.add_argument("--name", help="Experiment name, recorded with the adapter")

    parser.add_argument("--checkpoint", help="Model name or local directory to adapt")

    parser.add_argument("--pairs", type=Path, help="Mined pair file to train on")

    parser.add_argument(
        "--eval-pairs-file",
        type=Path,
        help=(
            "Score against this file instead of --pairs. Hold it fixed to "
            "compare training sets: without it, a run that changes what it "
            "trains on also changes what it is judged by"
        ),
    )

    parser.add_argument("--train-pairs", type=int, help="Pairs to train on")

    parser.add_argument("--eval-pairs", type=int, help="Pairs to score against")

    parser.add_argument(
        "--sample-pairs",
        type=int,
        help=(
            "Pairs to draw from the file before filtering by kind. Defaults to "
            "--train-pairs plus --eval-pairs, which is wrong when a kind filter "
            "is set: a kind holding a sixth of the file yields a sixth of the "
            "sample, and --train-pairs stops binding"
        ),
    )

    parser.add_argument("--train-kinds", help="Comma-separated pair kinds to train on")

    parser.add_argument("--eval-kinds", help="Comma-separated pair kinds to score on")

    parser.add_argument("--train-languages", help="Comma-separated languages to train on")

    parser.add_argument("--eval-languages", help="Comma-separated languages to score on")

    parser.add_argument(
        "--adaptation",
        choices=sorted(ADAPTATIONS),
        help=(
            "What this run claims to measure. Checked against what the filters "
            "actually do, and the run is refused if they disagree — the label "
            "outlives the command line, so it must not be able to be wrong"
        ),
    )

    parser.add_argument("--epochs", type=int)

    parser.add_argument("--learning-rate", type=float)

    parser.add_argument("--rank", type=int, help="LoRA rank")

    parser.add_argument("--targets", help="Comma-separated module names LoRA attaches to")

    parser.add_argument("--pooling", choices=["mean", "cls"])

    parser.add_argument("--max-length", type=int)

    parser.add_argument("--query-prefix", help='e.g. "query: " for E5')

    parser.add_argument("--passage-prefix", help='e.g. "passage: " for E5')

    parser.add_argument(
        "--save-adapter",
        type=Path,
        help=(
            "Directory to write the trained adapter to. Without it the run "
            "produces a measurement rather than a model"
        ),
    )

    parser.add_argument(
        "--data-provenance",
        choices=[str(value) for value in DataProvenance],
        help=(
            "Where the training data came from: public, synthetic or "
            "licensed. Required to save an adapter — a saved model carries "
            "this as a legal fact about itself, and it has no default so it "
            "cannot be answered by omission"
        ),
    )

    parser.add_argument("--output", type=Path, help="Write the full comparison as JSON")


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
        help="Evaluate a trained experiment",
        description=(
            "Score a trained experiment, choosing the instrument by what the "
            "experiment is. A static model (`qfme train`) is scored intrinsically "
            "against a corpus: tokenizer coverage and the neighbourhoods of its "
            "word vectors, so it wants --source. A contextual encoder (`qfme "
            "pretrain` / `qfme finetune`) has no per-token matrix to probe — its "
            "quality is whether it retrieves the right passage — so it is scored "
            "by retrieval over a mined pair file, and wants --pairs. The right "
            "path is picked automatically from the experiment directory."
        ),
    )

    parser.add_argument(
        "--experiment",
        type=Path,
        required=True,
        help="Experiment directory produced by `qfme train`, `pretrain` or `finetune`",
    )

    parser.add_argument(
        "--source",
        type=Path,
        help="Corpus to evaluate a static model against (intrinsic metrics)",
    )

    parser.add_argument(
        "--pairs",
        type=Path,
        help=(
            "Mined pair file to score a contextual encoder against by retrieval. "
            "Required for a `pretrain`/`finetune` experiment, which has no matrix "
            "to probe intrinsically. Must be HELD OUT from training — evaluate "
            "scores exactly these pairs and cannot tell which the model has seen, "
            "so pointing it at the training file inflates the score silently"
        ),
    )

    parser.add_argument(
        "--eval-pairs",
        type=int,
        default=2_000,
        help="Pairs to draw from --pairs for the retrieval score (default: 2000)",
    )

    parser.add_argument("--format", default="auto", choices=["auto", "text", "lines", "jsonl"])

    parser.add_argument("--output", type=Path, help="Write the report JSON here")


def _add_serve_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "serve",
        help="Serve a model over HTTP behind an embeddings endpoint",
        description=(
            "Serve a model behind an embeddings endpoint. The model can be a "
            "LoRA adapter from `qfme adapt --save-adapter`, or a from-scratch / "
            "fine-tuned encoder directory from `qfme pretrain` / `qfme finetune`; "
            "the server routes on shape. Requires the `serve` extra: "
            "uv sync --extra serve --extra pretrained"
        ),
    )

    parser.add_argument(
        "--adapter",
        type=Path,
        required=True,
        help=(
            "Model to serve: a LoRA adapter directory from `qfme adapt "
            "--save-adapter`, or a from-scratch / fine-tuned experiment "
            "directory (with encoder/ + tokenizer/) from `qfme pretrain` / "
            "`qfme finetune`"
        ),
    )

    parser.add_argument(
        "--model-id",
        default="",
        help="Name clients use (default: the adapter directory's own name)",
    )

    parser.add_argument(
        "--default-input-type",
        choices=["query", "passage"],
        help=(
            "Side to assume when a request omits input_type. Without it an "
            "asymmetric model answers 400 rather than guessing, because "
            "embedding text on the wrong side returns a plausible vector "
            "that encodes the wrong thing. Set it only when every caller of "
            "this deployment is on one side."
        ),
    )

    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")

    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")

    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Refuse to fetch the base checkpoint from the network",
    )


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


def _run_extract(args: argparse.Namespace) -> int:
    """Extract a Wikipedia dump, then say what to do next."""

    from multilingual_embedding.corpus.wikipedia import extract_dump

    count = extract_dump(
        args.dump,
        args.output,
        language=args.language,
        minimum_characters=args.minimum_characters,
        limit=args.limit,
        deduplicate=not args.keep_duplicates,
    )

    print(f"Wrote {count} articles to {args.output}")

    # The extraction is not finished until it has been judged. Saying so
    # here is the difference between a corpus that was checked and one
    # that merely exists.
    print(f"Next: qfme validate --source {args.output}")

    return EXIT_SUCCESS


def _run_extract_judgments(args: argparse.Namespace) -> int:
    """Extract judgment PDFs into the training corpus, then say what to do next."""

    from multilingual_embedding.corpus.judgments import extract_judgments

    count = extract_judgments(
        args.source,
        args.output,
        language=args.language,
        minimum_characters=args.minimum_characters,
        limit=args.limit,
    )

    print(f"Wrote {count} judgments to {args.output}")

    # PDF extraction is exactly the step whose fidelity cannot be trusted
    # until it is checked, so the corpus is not done until the audit has
    # seen it — the same next step as a Wikipedia extraction.
    print(f"Next: qfme validate --source {args.output}")

    return EXIT_SUCCESS


def _run_prepare_eval(args: argparse.Namespace) -> int:
    """Build the MILPaC held-out evaluation set, then say what to do with it."""

    from multilingual_embedding.corpus.milpac import extract_milpac

    languages = tuple(code.strip() for code in args.languages.split(",") if code.strip())

    datasets = (
        tuple(name.strip() for name in args.datasets.split(",") if name.strip())
        if args.datasets
        else None
    )

    count = extract_milpac(
        args.source,
        args.output,
        languages=languages,
        datasets=datasets,
    )

    print(f"Wrote {count} evaluation pairs to {args.output}")

    # Named --eval-pairs-file, never --pairs. The origin wall and the
    # non-commercial licence both depend on this file staying on the
    # scoring side of the run.
    print(f"Next: qfme adapt --config <experiment> --eval-pairs-file {args.output}")

    return EXIT_SUCCESS


def _run_mine_pairs(args: argparse.Namespace) -> int:
    """Mine contrastive pairs, and report how leaky they are."""

    import gzip
    import json as _json

    from multilingual_embedding.corpus.pairs import (
        PairConfig,
        PairStatistics,
        iter_pairs,
    )
    from multilingual_embedding.corpus.reader import reader_for

    reader = reader_for(args.source, format=args.format)

    config = PairConfig(
        maximum_overlap=args.max_overlap,
        kinds=tuple(kind.strip() for kind in args.kinds.split(",") if kind.strip()),
    )

    destination = Path(args.output).expanduser()

    destination.parent.mkdir(parents=True, exist_ok=True)

    opener = gzip.open if destination.suffix == ".gz" else open

    statistics = PairStatistics()

    # Streamed rather than collected. A pair set from a mid-sized
    # Wikipedia is millions of pairs holding two strings each, and
    # holding them all before writing is gigabytes for no reason.
    with opener(destination, "wt", encoding="utf-8") as handle:
        for pair in iter_pairs(reader.iter_documents(), config, statistics):
            handle.write(_json.dumps(pair.to_record(), ensure_ascii=False) + "\n")

    summary = statistics.to_dict()

    print(f"Wrote {statistics.produced} pairs to {destination}")

    print()

    print(f"{'kind':<18}{'pairs':>9}{'mean overlap':>15}")

    for kind, count in sorted(summary["by_kind"].items()):
        overlap = summary["mean_overlap_by_kind"][kind]

        # Flagged inline rather than only in a log, because a reader
        # scanning this table is exactly who needs to know that a kind
        # can be solved without understanding anything.
        note = "  <- solvable by string match" if overlap > 0.75 else ""

        print(f"{kind:<18}{count:>9}{overlap:>15.2f}{note}")

    if args.report:
        from multilingual_embedding.utils.io import write_json

        write_json(args.report, summary)

        print(f"\nStatistics written to {args.report}")

    return EXIT_SUCCESS


def _run_mine_negatives(args: argparse.Namespace) -> int:
    """
    Mine hard negatives against a model, and report what was thrown away.

    The model can be a LoRA adapter or a from-scratch experiment directory;
    ``--adapter`` routes on shape the same way ``evaluate`` does, so the
    same command mines against either kind.

    Unlike ``mine-pairs`` this cannot stream. The candidate pool is the
    pair set's own positives, so every pair has to be resident and
    encoded before the first negative can be found. ``--limit`` takes a
    uniform sample for a trial run, which is also the sane way to see
    what the filters are doing before spending an hour on the full set.
    """

    try:
        from multilingual_embedding.embedding.negatives import (
            NegativeConfig,
            mine_negatives,
        )
        from multilingual_embedding.pipelines.search import SemanticSearchPipeline
    except ImportError as error:  # pragma: no cover - depends on the install
        print(
            f"error: mining hard negatives needs a loadable model ({error.name} is "
            "missing). Install it with: uv sync --extra neural (add --extra "
            "pretrained to mine against a LoRA adapter)",
            file=sys.stderr,
        )

        return EXIT_ERROR

    from multilingual_embedding.corpus.pairs import MinedPair, sample_pairs
    from multilingual_embedding.utils.io import read_jsonl, write_jsonl

    source = Path(args.pairs).expanduser()

    if args.limit:
        pairs = sample_pairs(source, args.limit)
    else:
        pairs = [MinedPair.from_record(record) for record in read_jsonl(source)]

    print(f"Read {len(pairs):,} pairs from {source}")

    target = Path(args.adapter).expanduser()

    # Hard negatives are relative to a model, and that model can be either
    # a fine-tuned adapter or a from-scratch experiment directory. Route on
    # shape, exactly like `evaluate`: an experiment writes `encoder/` +
    # `tokenizer/`, so if both are present serve the contextual encoder;
    # otherwise treat the path as a LoRA adapter. `from_directory` exposes
    # the same `.encoder` and `.prefixes` (symmetric `("", "")`) the miner
    # reads, so nothing downstream has to know which kind it got.
    if (target / "encoder").is_dir() and (target / "tokenizer").is_dir():
        pipeline = SemanticSearchPipeline.from_directory(target)
    else:
        pipeline = SemanticSearchPipeline.from_adapter(
            args.adapter,
            local_files_only=args.local_files_only,
        )

    query_prefix, passage_prefix = pipeline.prefixes

    print(f"Mining against {args.adapter} (prefixes {query_prefix!r} / {passage_prefix!r})")

    mined, statistics = mine_negatives(
        pairs,
        pipeline.encoder,
        NegativeConfig(
            count=args.count,
            pool=args.pool,
            maximum_similarity=args.max_similarity,
            minimum_similarity=args.min_similarity,
            allow_same_document=args.allow_same_document,
            # Read off the adapter rather than accepted as flags. An E5
            # model mined without its prefixes ranks by the wrong
            # geometry and still returns a full set of negatives.
            query_prefix=query_prefix,
            passage_prefix=passage_prefix,
            audit_sample=args.audit_sample,
        ),
    )

    written = write_jsonl(args.output, (pair.to_record() for pair in mined))

    summary = statistics.to_dict()

    print(f"\nWrote {written:,} pairs to {args.output}")

    print()

    print(f"{'candidates examined':<28}{summary['candidates']:>10,}")

    for reason, label in (
        ("rejected_as_positive", "  the pair's own positive"),
        ("rejected_same_document", "  same document"),
        ("rejected_too_similar", "  above the ceiling"),
        ("rejected_too_easy", "  below the floor"),
        ("rejected_unencodable", "  unencodable"),
    ):
        print(f"{label:<28}{summary[reason]:>10,}")

    print(f"{'negatives kept':<28}{summary['accepted']:>10,}")

    print(f"{'pairs with none':<28}{summary['pairs_without_negatives']:>10,}")

    # Only when it happens. A row of zeros on every healthy run trains
    # the reader to skip the block this number lives in, and a corpus
    # the encoder cannot read is the one case where that costs something.
    if summary["anchors_unencodable"]:
        print(f"{'  anchor unencodable':<28}{summary['anchors_unencodable']:>10,}")

    print(f"{'mean similarity':<28}{summary['mean_similarity']:>10.3f}")

    print()

    # Printed as prose rather than as a row, because a reader who takes
    # this for a measured false-negative rate will train on poison and
    # watch the loss fall while doing it.
    print(
        f"{summary['outranking_the_positive']:,} of the kept negatives "
        f"({summary['suspicion_rate']:.1%}) score above their own positive."
    )

    print(
        "That is the population false negatives are drawn from, not a "
        "false-negative rate. Label the audit sample to get one."
    )

    if args.audit:
        count = write_jsonl(args.audit, (record.to_record() for record in statistics.audit))

        print(f"\n{count} negatives written to {args.audit} for labelling")

    if args.report:
        from multilingual_embedding.utils.io import write_json

        write_json(args.report, summary)

        print(f"\nStatistics written to {args.report}")

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


def _run_pretrain(args: argparse.Namespace) -> int:
    """Pretrain a from-scratch contextual encoder with a masked-language head."""

    from multilingual_embedding.pipelines.pretraining import PretrainingPipeline

    config = _resolve_config(args)

    if args.name:
        config = config.merged({"name": args.name})

    # A flag beats the config, but either satisfies the pipeline's wall.
    if args.data_provenance:
        config = config.merged({"pretraining": {"data_provenance": args.data_provenance}})

    result = PretrainingPipeline(config).run(
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume_from,
        stop_after_epoch=args.stop_after_epoch,
    )

    print(json.dumps(result.summary(), indent=2, ensure_ascii=False))

    # Non-zero only when a run that *could* be judged failed the judgement:
    # a multi-epoch run whose loss did not fall learned nothing, and a
    # chained pipeline should stop rather than fine-tune on it. A single
    # epoch — including a deliberate --stop-after-epoch 0 in the
    # interruptible workflow — is not measurable against itself, and a
    # partial run that will be resumed is not a failure, so it exits zero.
    if result.report.measurable and not result.report.improved:
        return EXIT_ERROR

    return EXIT_SUCCESS


# Flags that override the `finetune` config section, mapped to the field
# they set. Same data-driven wiring as _ADAPT_OVERRIDES below. --no-save
# is handled apart from this map, because it is a boolean whose absence
# and whose false value are the same thing to argparse.
_FINETUNE_OVERRIDES = {
    "source": "source",
    "pairs": "pairs",
    "eval_pairs_file": "eval_pairs_file",
    "train_pairs": "train_pairs",
    "eval_pairs": "eval_pairs",
    "sample_pairs": "sample_pairs",
    "train_kinds": "train_kinds",
    "eval_kinds": "eval_kinds",
    "train_languages": "train_languages",
    "eval_languages": "eval_languages",
    "epochs": "epochs",
    "learning_rate": "learning_rate",
    "temperature": "temperature",
    "data_provenance": "data_provenance",
    "output": "report",
}


def _run_finetune(args: argparse.Namespace) -> int:
    """Contrastively fine-tune a from-scratch encoder and report whether it helped."""

    from multilingual_embedding.pipelines.finetune import FinetunePipeline

    overrides: dict[str, Any] = {}

    for assignment in args.overrides:
        _merge(overrides, parse_override(assignment))

    section: dict[str, Any] = {}

    for flag, field_name in _FINETUNE_OVERRIDES.items():
        value = getattr(args, flag, None)

        if value is not None:
            section[field_name] = str(value) if isinstance(value, Path) else value

    # Only set `save` when --no-save is given: its default lives in the
    # config, and writing save=True here on every run would silently
    # override a `save: false` set in the config file.
    if args.no_save:
        section["save"] = False

    if section:
        _merge(overrides, {"finetune": section})

    if args.name:
        _merge(overrides, {"name": args.name})

    config = load_config(args.config, profile=args.profile, overrides=overrides)

    result = FinetunePipeline(config).run(
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume_from,
        stop_after_epoch=args.stop_after_epoch,
    )

    print("\n" + "=" * 68)

    print("DID FINE-TUNING HELP?")

    print("=" * 68)

    print(result.summary())

    # A run deliberately cut short with --stop-after-epoch has not finished
    # training, so its before/after comparison is not the verdict yet — it
    # exits zero the way an interrupted pretrain does, because a partial run
    # that will be resumed is not a failure.
    interrupted = (
        args.stop_after_epoch is not None
        and args.stop_after_epoch < config.finetune.epochs - 1
    )

    if interrupted:
        return EXIT_SUCCESS

    # Otherwise non-zero when the fine-tune did not beat the pretrained
    # encoder it started from, so a pipeline that chains pretrain ->
    # finetune -> deploy stops rather than shipping a model that made
    # retrieval worse.
    return EXIT_SUCCESS if result.helped else EXIT_ERROR


# Flags that override the `adaptation` config section, mapped to the
# field they set. Kept as data rather than as thirty lines of `if`, so
# that adding a setting means adding a field and a flag, and the two
# cannot drift apart in the wiring between them.
_ADAPT_OVERRIDES = {
    "checkpoint": "checkpoint",
    "pairs": "pairs",
    "eval_pairs_file": "eval_pairs_file",
    "train_pairs": "train_pairs",
    "eval_pairs": "eval_pairs",
    "sample_pairs": "sample_pairs",
    "train_kinds": "train_kinds",
    "eval_kinds": "eval_kinds",
    "train_languages": "train_languages",
    "eval_languages": "eval_languages",
    "adaptation": "adaptation",
    "epochs": "epochs",
    "learning_rate": "learning_rate",
    "rank": "rank",
    "targets": "targets",
    "pooling": "pooling",
    "max_length": "max_length",
    "query_prefix": "query_prefix",
    "passage_prefix": "passage_prefix",
    "save_adapter": "save_adapter",
    "data_provenance": "data_provenance",
    "output": "report",
}


def _run_adapt(args: argparse.Namespace) -> int:
    """Adapt a published checkpoint and report whether it helped."""

    from multilingual_embedding.pipelines.adaptation import AdaptationPipeline

    overrides: dict[str, Any] = {}

    for assignment in args.overrides:
        _merge(overrides, parse_override(assignment))

    section: dict[str, Any] = {}

    for flag, field_name in _ADAPT_OVERRIDES.items():
        value = getattr(args, flag, None)

        # `is not None` rather than truthiness, so `--query-prefix ""`
        # can deliberately clear a prefix the config file set. An empty
        # prefix on an E5 model is a real choice, and a wrong one made
        # silently is exactly what save_adapter records prefixes to
        # prevent.
        if value is not None:
            section[field_name] = str(value) if isinstance(value, Path) else value

    if section:
        _merge(overrides, {"adaptation": section})

    if args.name:
        _merge(overrides, {"name": args.name})

    config = load_config(args.config, profile=args.profile, overrides=overrides)

    result = AdaptationPipeline(config).run()

    print("\n" + "=" * 68)

    print("DID ADAPTATION HELP?")

    print("=" * 68)

    print(result.summary())

    # Non-zero when the adapter did not beat the checkpoint it started
    # from. A pipeline that chains adaptation into a deployment step
    # should stop there rather than ship a model that made things worse,
    # and reading that off stdout is not something a shell script can do.
    return EXIT_SUCCESS if result.helped else EXIT_ERROR


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
    """
    Evaluate a trained experiment, routing by what the experiment is.

    A contextual encoder has no per-token matrix, so the intrinsic
    neighbour-probe evaluation cannot describe it; its quality is
    retrieval, and it takes a different path. Which one is decided by the
    directory shape rather than by a flag, so the caller does not have to
    know which kind of model a directory holds.
    """

    experiment = Path(args.experiment)

    if (experiment / "encoder").is_dir():
        return _evaluate_contextual(args, experiment)

    return _evaluate_static(args, experiment)


def _evaluate_static(args: argparse.Namespace, experiment: Path) -> int:
    """Score a static model intrinsically against a corpus."""

    from multilingual_embedding.corpus.loader import stream_sentences
    from multilingual_embedding.embedding.matrix import EmbeddingMatrix
    from multilingual_embedding.evaluation.embedding_eval import (
        EmbeddingEvaluator,
        sample_neighbour_probes,
    )
    from multilingual_embedding.evaluation.report import EvaluationReport
    from multilingual_embedding.evaluation.tokenizer_eval import TokenizerEvaluator
    from multilingual_embedding.tokenizer.tokenizer import SentencePieceTokenizer

    if not args.source:
        print(
            "error: a static model is scored against a corpus; pass --source",
            file=sys.stderr,
        )

        return EXIT_ERROR

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


def _evaluate_contextual(args: argparse.Namespace, experiment: Path) -> int:
    """
    Score a contextual encoder by retrieval over a mined pair file.

    A from-scratch or fine-tuned encoder has no per-token matrix to probe,
    so the honest question is whether it retrieves the right passage for a
    query. This reuses the same retrieval evaluator a fine-tune runs on its
    held-out set.

    One caution the tool cannot enforce: ``evaluate`` scores *exactly* the
    pairs it is given and has no knowledge of what the encoder trained on,
    so it cannot carve out a held-out split the way ``finetune`` does. Point
    it at the training pairs and the score is measured on data the model has
    seen — inflated, silently. Pass a file disjoint from training. Because
    ``finetune`` samples its held-out set with a different seed and its own
    facet filters, this number is comparable to, but will not exactly equal,
    the ``after`` a fine-tune reported unless it is handed that same set.
    """

    from multilingual_embedding.corpus.pairs import sample_pairs
    from multilingual_embedding.evaluation.retrieval import evaluate_retrieval
    from multilingual_embedding.tokenizer.tokenizer import SentencePieceTokenizer

    if not args.pairs:
        print(
            "error: a contextual encoder is scored by retrieval, not intrinsically; "
            "pass --pairs (a mined pair file)",
            file=sys.stderr,
        )

        return EXIT_ERROR

    # Imported lazily for the same reason as the training paths: torch is
    # an optional extra, and evaluate must stay runnable for a static
    # model on a machine that never installed it.
    from multilingual_embedding.embedding.neural import NeuralTextEncoder

    tokenizer = SentencePieceTokenizer.load(experiment / "tokenizer")

    encoder = NeuralTextEncoder.load(experiment / "encoder", tokenizer)

    pairs = sample_pairs(args.pairs, args.eval_pairs)

    report = evaluate_retrieval(encoder, pairs, limit=None)

    payload = {
        "name": experiment.name,
        "eval_pairs": len(pairs),
        "retrieval": report.to_dict(),
    }

    if args.output:
        from multilingual_embedding.utils.io import write_json

        write_json(args.output, payload)

        print(f"Wrote report to {args.output}")
    else:
        print(f"retrieval over {len(pairs):,} pairs from {args.pairs}")

        print("(assumes these pairs are held out from training)\n")

        print(report.summary())

    return EXIT_SUCCESS


def _run_serve(args: argparse.Namespace) -> int:
    """Serve a model — a LoRA adapter or a from-scratch encoder — over HTTP.

    Binds the loopback address by default. The endpoint has no
    authentication and no rate limiting, so a default of ``0.0.0.0``
    would put an unauthenticated GPU-backed service on whatever network
    the host happens to be on the first time someone runs this. Reaching
    it from elsewhere is an explicit ``--host 0.0.0.0`` behind something
    that terminates TLS and checks credentials.
    """

    try:
        import uvicorn

        from multilingual_embedding.serving.app import ServingConfig, create_app
    except ImportError as error:  # pragma: no cover - depends on the install
        print(
            f"error: serving needs the `serve` extra ({error.name} is missing). "
            "Install it with: uv sync --extra serve --extra pretrained",
            file=sys.stderr,
        )

        return EXIT_ERROR

    config = ServingConfig(
        adapter_directory=Path(args.adapter),
        model_id=args.model_id,
        default_input_type=args.default_input_type,
        local_files_only=args.local_files_only,
    )

    # Built here rather than passed as a factory string so that a broken
    # adapter fails now, with this process's error handling, instead of
    # inside uvicorn's startup after it has already claimed the port.
    app = create_app(config)

    print(f"Serving {config.resolved_model_id()} on http://{args.host}:{args.port}")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

    return EXIT_SUCCESS


# ----------------------------------------------------------------------
# Configuration resolution
# ----------------------------------------------------------------------


def _resolve_config(args: argparse.Namespace) -> ExperimentConfig:
    """
    Build the experiment configuration from file, flags and overrides.

    Explicit flags such as ``--source`` are folded in as overrides so
    that precedence stays in one place: file, then profile, then
    environment, then ``--set``, then named flags.
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

    config = load_config(args.config, profile=args.profile, overrides=overrides)

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
