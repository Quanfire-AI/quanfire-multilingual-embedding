"""
End-to-end verification, for pasting back.

Runs the whole pipeline against whatever dumps are present and prints one
compact report. The point is to be run on a machine the author does not
have — a GPU box above all, since no CUDA path is exercised by any test
on a laptop without an NVIDIA card.

Usage::

    python scripts/verify_e2e.py --dumps data/dumps

Every stage is timed and every claim is measured rather than asserted, so
a stage that silently does nothing shows up as a zero rather than as a
tick. Failures are caught and reported instead of aborting, because a
report that stops at the first problem hides the rest of them.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import pathlib
import platform
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Documents the smoke test trains on. Chosen so the whole verification
# finishes in a couple of minutes; `--full` uses everything.
SMOKE_DOCUMENTS = 150

REPORT: list[str] = []


@contextlib.contextmanager
def quiet_stderr() -> Any:
    """
    Silence a C library that writes to fd 2 directly.

    SentencePiece prints its full trainer configuration and a progress
    log to standard error. Redirecting Python's ``sys.stderr`` does not
    reach it, because the writes happen below Python — the file
    descriptor itself has to be pointed elsewhere.
    """

    saved = os.dup(2)

    with open(os.devnull, "w") as null:
        os.dup2(null.fileno(), 2)

        try:
            yield
        finally:
            os.dup2(saved, 2)

            os.close(saved)


def clock() -> str:
    """Wall-clock time, for a log a human reads while waiting."""

    return time.strftime("%H:%M:%S")


def duration(seconds: float) -> str:
    """A span in the units a reader wants, not always seconds."""

    if seconds < 60:
        return f"{seconds:.1f}s"

    minutes, remainder = divmod(int(seconds), 60)

    if minutes < 60:
        return f"{minutes}m {remainder:02d}s"

    hours, minutes = divmod(minutes, 60)

    return f"{hours}h {minutes:02d}m {remainder:02d}s"


def say(line: str = "") -> None:
    """Record a line for the report and show it as it happens."""

    REPORT.append(line)

    print(line, flush=True)


@dataclass
class Stage:
    """One verification step and what it measured."""

    name: str

    ok: bool = False

    seconds: float = 0.0

    detail: dict[str, Any] = field(default_factory=dict)

    error: str = ""

    peak_memory_mb: float | None = None


STAGES: list[Stage] = []


def run(name: str, function: Any, *args: Any, **kwargs: Any) -> Any:
    """Time a stage, record it, and keep going if it fails."""

    stage = Stage(name=name)

    STAGES.append(stage)

    # Announced before it runs, not after. Without this a stage that
    # takes an hour is indistinguishable from a stage that has hung, and
    # neither the operator nor the author can say which — which is
    # exactly what happened on the first real dump.
    say(f"  [{clock()}] {name} ...")

    started = time.perf_counter()

    try:
        result, detail = function(*args, **kwargs)

        stage.ok, stage.detail = True, detail

        say(f"  [{clock()}] {name} done in {duration(time.perf_counter() - started)}")

        return result
    except Exception as error:
        stage.error = f"{type(error).__name__}: {error}"

        # The whole traceback, not a tail. A three-line excerpt names the
        # line that raised and hides the import chain that caused it,
        # which is precisely what matters for an error arriving from a
        # compiled dependency rather than from this code.
        stage.detail = {"traceback": traceback.format_exc().splitlines()}

        say(f"  [{clock()}] {name} FAILED: {stage.error}")

        return None
    finally:
        stage.seconds = time.perf_counter() - started

        stage.peak_memory_mb = resident_mb()


# ----------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------


def resident_mb() -> float | None:
    """
    Peak resident memory in megabytes.

    The units differ by platform and getting them wrong is easy: macOS
    reports ``ru_maxrss`` in bytes, Linux in kilobytes. Reported in MB
    rather than GB because a mislabelled GB figure claimed this process
    had used 176 GB, which is the kind of number that gets ignored rather
    than questioned.
    """

    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        divisor = 1024**2 if sys.platform == "darwin" else 1024
    except Exception:
        return None

    return round(peak / divisor, 2)


def describe_environment() -> dict[str, Any]:
    """What machine is this, and what can it actually do."""

    info: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    }

    try:
        import multilingual_embedding

        info["qfme"] = getattr(multilingual_embedding, "__version__", "unknown")
    except ImportError:
        info["qfme"] = "NOT INSTALLED"

    try:
        import torch

        info["torch"] = torch.__version__

        info["cuda_available"] = torch.cuda.is_available()

        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)

            info["cuda_capability"] = ".".join(
                str(part) for part in torch.cuda.get_device_capability(0)
            )

            info["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)

            # The whole reason bf16 is offered rather than fp16.
            info["bf16_supported"] = torch.cuda.is_bf16_supported()

        info["mps_available"] = torch.backends.mps.is_available()
    except ImportError:
        info["torch"] = "NOT INSTALLED (neural extra missing)"

    try:
        import mwparserfromhell

        info["mwparserfromhell"] = mwparserfromhell.__version__
    except ImportError:
        info["mwparserfromhell"] = "NOT INSTALLED (wikipedia extra missing)"

    return info


# ----------------------------------------------------------------------
# Stages
# ----------------------------------------------------------------------


def is_complete_gzip(path: Path) -> bool:
    """
    True when a gzip file decompresses to its end.

    A file left behind by an interrupted run is a valid gzip *prefix*,
    so its mere existence proves nothing — reusing one would feed a
    truncated corpus into everything downstream and the damage would show
    up as a quality problem rather than an error. Reading it through is
    cheap next to re-extracting a dump.
    """

    import gzip

    if not path.exists():
        return False

    try:
        with gzip.open(path, "rb") as handle:
            while handle.read(1024 * 1024):
                pass
    except (OSError, EOFError):
        return False

    return True


def stage_extract(
    dump: Path, language: str, output: Path, reuse: bool
) -> tuple[Path, dict[str, Any]]:
    """Extract a dump, and report how much of it was not article prose."""

    from multilingual_embedding.corpus.wikipedia import extract_dump

    if reuse and is_complete_gzip(output):
        with __import__("gzip").open(output, "rt", encoding="utf-8") as handle:
            count = sum(1 for _ in handle)

        return output, {
            "reused": True,
            "dump_mb": round(dump.stat().st_size / 1024**2, 1),
            "articles_written": count,
            "output_mb": round(output.stat().st_size / 1024**2, 1),
        }

    count = extract_dump(dump, output, language=language)

    return output, {
        "reused": False,
        "dump_mb": round(dump.stat().st_size / 1024**2, 1),
        "articles_written": count,
        "output_mb": round(output.stat().st_size / 1024**2, 1),
    }


def stage_audit(corpus: Path) -> tuple[Any, dict[str, Any]]:
    """The extractor must produce a corpus its own audit accepts."""

    from multilingual_embedding.corpus.audit import Severity, audit_corpus
    from multilingual_embedding.corpus.reader import reader_for

    audit = audit_corpus(reader_for(corpus).iter_documents())

    # A stage that ran without raising is not the same as a stage whose
    # answer was good. The audit reported usable=False on Hindi and the
    # report printed PASS beside it, because "did not raise" was the only
    # thing being checked. The verdict is the result here.
    if not audit.usable:
        blocking = ", ".join(f.code for f in audit.findings if f.severity is Severity.ERROR)

        raise RuntimeError(f"audit judged the corpus unusable: {blocking}")

    return audit, {
        "documents": audit.documents,
        "sentences": audit.sentences,
        "scripts": sorted(audit.scripts),
        "usable": audit.usable,
        "errors": [f.code for f in audit.findings if f.severity is Severity.ERROR],
        "warnings": [f.code for f in audit.findings if f.severity is Severity.WARNING],
    }


def stage_mine(corpus: Path, output: Path) -> tuple[Path, dict[str, Any]]:
    """Mine pairs, and report how lexically leaky each kind is."""

    import gzip

    from multilingual_embedding.corpus.pairs import PairStatistics, iter_pairs
    from multilingual_embedding.corpus.reader import reader_for

    statistics = PairStatistics()

    opener = gzip.open if output.suffix == ".gz" else open

    with opener(output, "wt", encoding="utf-8") as handle:
        for pair in iter_pairs(reader_for(corpus).iter_documents(), None, statistics):
            handle.write(json.dumps(pair.to_record(), ensure_ascii=False) + "\n")

    summary = statistics.to_dict()

    return output, {
        "pairs": summary["produced"],
        "by_kind": summary["by_kind"],
        "mean_overlap": summary["mean_overlap_by_kind"],
    }


def stage_train_static(
    corpus: Path, name: str, full: bool, work: Path
) -> tuple[Path, dict[str, Any]]:
    """
    Train the static model through the ordinary pipeline.

    Deliberately undersized unless ``--full`` is given, and capped to a
    sample of the corpus.

    Measured end to end — tokenizer training, tokenization and word2vec
    together — this path runs at about **70 sentences per second** on a
    laptop CPU: 7,203 sentences took 102 seconds. word2vec alone manages
    roughly 300/s and degrades as the vocabulary grows.

    So a mid-sized wiki is minutes and a large one is hours. That is a
    real limit of the static path, pure numpy and single-process, rather
    than a property of this script.

    So the smoke test trains on a sample and says so. A verification run
    that takes an hour is a verification run nobody performs.
    """

    import gzip

    from multilingual_embedding.config.base import ExperimentConfig
    from multilingual_embedding.pipelines.training import TrainingPipeline

    source = corpus

    sampled = 0

    if not full:
        # A truncated copy rather than a flag, because the pipeline takes
        # a corpus and should not learn about sampling for a script's
        # convenience.
        source = corpus.parent / f"{name}-sample.jsonl.gz"

        with (
            gzip.open(corpus, "rt", encoding="utf-8") as reader,
            gzip.open(source, "wt", encoding="utf-8") as writer,
        ):
            for _, line in zip(range(SMOKE_DOCUMENTS), reader, strict=False):
                writer.write(line)

                sampled += 1

    config = ExperimentConfig(
        name=name,
        # Under --work, not the default "artifacts" relative to the
        # current directory. Left at the default this writes into
        # whatever directory it was launched from — which for a checkout
        # means the repository itself, and which collided with the test
        # suite when both ran at once.
        output_directory=str(work / "artifacts"),
        evaluation={"report_directory": str(work / "reports")},
        corpus={"source": str(source)},
        tokenizer={"vocab_size": 8000 if full else 2000},
        embedding=({"dimension": 128, "epochs": 3} if full else {"dimension": 32, "epochs": 1}),
    )

    with quiet_stderr():
        result = TrainingPipeline(config).run()

    metrics = result.tokenizer_metrics

    return result.experiment_directory, {
        "documents_trained_on": "all" if full else sampled,
        "vocabulary_size": len(result.matrix.vocabulary),
        "dimension": result.matrix.dimension,
        "characters_per_token": round(metrics.characters_per_token, 4),
        "unknown_rate": round(metrics.unknown_rate, 6),
        "languages_measured": sorted(result.tokenizer_metrics_by_language),
    }


def stage_train_contextual(
    pairs_path: Path,
    device: str,
    precision: str,
    batch_size: int,
    chunk: int,
    pair_limit: int,
    layers: int,
    dimension: int,
) -> tuple[Any, dict[str, Any]]:
    """
    Train the transformer on mined pairs.

    This is the stage that matters on a GPU box, because it is the only
    one that touches CUDA, bf16 and gradient caching at once.
    """

    import gzip
    import hashlib

    import torch

    from multilingual_embedding.embedding.neural import (
        ContrastiveConfig,
        ContrastiveTrainer,
        EncoderConfig,
        NeuralTextEncoder,
        TextPair,
        TransformerEncoderModel,
    )

    class Encoding:
        def __init__(self, ids: list[int]) -> None:
            self.ids = ids

    class Hashing:
        """Deterministic, so the numbers reproduce across machines."""

        def encode(self, text: str) -> Encoding:
            return Encoding(
                [
                    1
                    + int.from_bytes(
                        hashlib.blake2b(word.encode("utf-8"), digest_size=4).digest(),
                        "big",
                    )
                    % 8190
                    for word in text.split()
                ][:128]
            )

    with gzip.open(pairs_path, "rt", encoding="utf-8") as handle:
        records = [json.loads(line) for _, line in zip(range(pair_limit), handle, strict=False)]

    pairs = [TextPair(r["anchor"], r["positive"]) for r in records]

    torch.manual_seed(0)

    model = TransformerEncoderModel(
        EncoderConfig(
            vocabulary_size=8192,
            dimension=dimension,
            layers=layers,
            heads=8,
            max_length=128,
            dropout=0.1,
        )
    )

    encoder = NeuralTextEncoder(model, Hashing(), device=device)

    report = ContrastiveTrainer(
        encoder,
        ContrastiveConfig(
            # Two epochs, not one. `improved` compares the first epoch's
            # mean loss with the last, so a single-epoch run compares an
            # epoch with itself and can never show improvement — the
            # check would report False however well training went.
            epochs=2,
            batch_size=batch_size,
            learning_rate=3e-4,
            seed=1,
            precision=precision,
            gradient_checkpoint_chunk=chunk,
        ),
    ).train(pairs)

    peak = None

    if device.startswith("cuda"):
        peak = round(torch.cuda.max_memory_allocated() / 1024**3, 2)

    return report, {
        "device": str(encoder.device),
        "precision": precision,
        "batch_size": batch_size,
        "gradient_checkpoint_chunk": chunk,
        "pairs": len(pairs),
        "parameters": model.parameter_count(),
        "initial_loss": round(report.initial_loss, 4),
        "final_loss": round(report.final_loss, 4),
        "improved": report.improved,
        "peak_vram_gb": peak,
    }


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------


def run_contextual(pairs_path: Path) -> None:
    """
    Train the transformer, and on CUDA measure what only CUDA can answer.

    The bf16 and gradient-caching claims are the ones no test on a
    machine without an NVIDIA card can check, so on CUDA this runs the
    configured shape and then the same shape without either, and reports
    peak VRAM for both. Without the comparison, a number is just a
    number.
    """

    import contextlib

    with contextlib.suppress(ImportError):
        import torch

        from multilingual_embedding.embedding.neural import resolve_device

        device = str(resolve_device())

        say()

        say("--- contextual training ---")

        run(
            "train-contextual[smoke]",
            stage_train_contextual,
            pairs_path,
            device,
            "fp32",
            16,
            0,
            512,
            2,
            64,
        )

        if not torch.cuda.is_available():
            return

        # A full 2x2, not two opposite corners.
        #
        # The first version compared bf16+gradcache against
        # fp32+nogradcache and measured a 16.9x reduction in peak VRAM on
        # a 4070 Ti SUPER. The number was real and unattributable: two
        # variables moved at once, so it could have been almost all
        # gradient caching, almost all bf16, or any split between them.
        # Measuring the two middle cells costs ten seconds and is the
        # difference between a number and a claim.
        for precision in ("fp32", "bf16"):
            for chunk in (0, 32):
                label = f"{precision}+{'gradcache' if chunk else 'nocache'}"

                torch.cuda.empty_cache()

                torch.cuda.reset_peak_memory_stats()

                run(
                    f"ablation[{label}]",
                    stage_train_contextual,
                    pairs_path,
                    device,
                    precision,
                    256,
                    chunk,
                    4000,
                    4,
                    256,
                )

        summarise_ablation()


def summarise_ablation() -> None:
    """
    Lay the 2x2 out, so attribution is read rather than inferred.

    Printing four numbers in a list leaves the reader doing the division
    that decides which feature earned its place.
    """

    cells = {
        stage.name[len("ablation[") : -1]: stage.detail
        for stage in STAGES
        if stage.name.startswith("ablation[") and stage.ok
    }

    if len(cells) < 4:
        return

    say()

    say("  ABLATION — peak VRAM in GB, batch 256, identical model and data")

    say(f"  {'':10}{'no gradcache':>16}{'gradcache':>14}")

    for precision in ("fp32", "bf16"):
        plain = cells.get(f"{precision}+nocache", {}).get("peak_vram_gb")

        cached = cells.get(f"{precision}+gradcache", {}).get("peak_vram_gb")

        say(f"  {precision:10}{plain!s:>16}{cached!s:>14}")

    baseline = cells.get("fp32+nocache", {}).get("peak_vram_gb")

    if baseline:
        say()

        for label in ("bf16+nocache", "fp32+gradcache", "bf16+gradcache"):
            value = cells.get(label, {}).get("peak_vram_gb")

            if value:
                say(f"  {label:16} {baseline / value:5.1f}x less than fp32+nocache")

    say()

    say("  Final losses, which should barely differ if the maths holds:")

    for label, detail in sorted(cells.items()):
        say(f"    {label:16} {detail.get('final_loss')}   {detail.get('seconds', '')}")


def finish(started_at: float, work: pathlib.Path) -> int:
    """Print the results table, the closing banner, and write the report."""

    say()

    say("=" * 68)

    say("RESULTS")

    say("=" * 68)

    for stage in STAGES:
        mark = "PASS" if stage.ok else "FAIL"

        say(f"[{mark}] {stage.name:34} {duration(stage.seconds):>12}")

        if stage.error:
            say(f"       {stage.error}")

        for key, value in stage.detail.items():
            say(f"       {key}: {value}")

        if stage.peak_memory_mb is not None:
            say(f"       peak_resident_mb: {stage.peak_memory_mb}")

    failures = [stage for stage in STAGES if not stage.ok]

    say()

    say("=" * 68)

    say(f"{len(STAGES) - len(failures)}/{len(STAGES)} stages passed")

    say(f"COMPLETED {time.strftime('%Y-%m-%d %H:%M:%S')}")

    say(f"TOTAL     {duration(time.time() - started_at)}")

    if failures:
        say(f"FAILED    {', '.join(stage.name for stage in failures)}")

    say("=" * 68)

    report_path = work / "verification-report.txt"

    report_path.write_text("\n".join(REPORT) + "\n", encoding="utf-8")

    print(f"\nFull report written to {report_path}")

    print("Paste that file back.")

    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--dumps",
        type=Path,
        default=Path("data/dumps"),
        help="Directory holding *-pages-articles.xml.bz2 files",
    )

    parser.add_argument(
        "--work",
        type=Path,
        default=Path("verify-output"),
        help="Where intermediates are written",
    )

    parser.add_argument(
        "--pairs",
        type=Path,
        help=(
            "Run only the contextual training stages against this pair "
            "file. For getting the GPU answer out of a machine without "
            "waiting for the corpus stages to run again"
        ),
    )

    parser.add_argument(
        "--reuse",
        action="store_true",
        help=(
            "Skip extraction when a complete output already exists. "
            "Checked by decompressing it, so a file left behind by an "
            "interrupted run is re-extracted rather than trusted"
        ),
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help=(
            "Train at realistic size rather than smoke-test size. Minutes "
            "per epoch on a mid-sized wiki; the default proves the pipeline "
            "runs, not that the model is good"
        ),
    )

    parser.add_argument(
        "--skip-contextual",
        action="store_true",
        help="Skip transformer training (no torch, or no time)",
    )

    arguments = parser.parse_args()

    arguments.work.mkdir(parents=True, exist_ok=True)

    started_at = time.time()

    say("=" * 68)

    say("QuanFire embedding — end-to-end verification")

    say(f"STARTED  {time.strftime('%Y-%m-%d %H:%M:%S')}")

    say("=" * 68)

    environment = describe_environment()

    say()

    say("ENVIRONMENT")

    for key, value in environment.items():
        say(f"  {key:22} {value}")

    if arguments.pairs is not None:
        say()

        say(f"CONTEXTUAL ONLY, against {arguments.pairs}")

        run_contextual(arguments.pairs)

        return finish(started_at, arguments.work)

    dumps = sorted(arguments.dumps.glob("*pages-articles*.xml.bz2"))

    if not dumps:
        say()

        say(f"No dumps found in {arguments.dumps}. Download at least one:")

        say(
            "  curl -O https://dumps.wikimedia.org/hiwiki/latest/"
            "hiwiki-latest-pages-articles.xml.bz2"
        )

        return 1

    say()

    say(f"DUMPS FOUND: {len(dumps)}")

    corpora: list[Path] = []

    for dump in dumps:
        # `hiwiki-latest-pages-articles.xml.bz2` -> `hi`
        language = dump.name.split("wiki")[0]

        say()

        say(f"--- {dump.name}  (language={language}) ---")

        corpus = run(
            f"extract[{language}]",
            stage_extract,
            dump,
            language,
            arguments.work / f"{language}.jsonl.gz",
            arguments.reuse,
        )

        if corpus is None:
            continue

        corpora.append(corpus)

        run(f"audit[{language}]", stage_audit, corpus)

        run(
            f"mine-pairs[{language}]",
            stage_mine,
            corpus,
            arguments.work / f"{language}-pairs.jsonl.gz",
        )

        run(
            f"train-static[{language}]",
            stage_train_static,
            corpus,
            f"verify-{language}",
            arguments.full,
            arguments.work,
        )

    if not arguments.skip_contextual and corpora:
        language = corpora[0].name.split(".")[0]

        pairs_path = arguments.work / f"{language}-pairs.jsonl.gz"

        if pairs_path.exists():
            run_contextual(pairs_path)

    return finish(started_at, arguments.work)


if __name__ == "__main__":
    sys.exit(main())
