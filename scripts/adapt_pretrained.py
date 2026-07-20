"""
Adapt a published encoder to a domain, and measure whether it helped.

The experiment this project has been building toward. Three steps:

1. Score a published checkpoint on held-out pairs. **This is the number
   to beat**, and it is the only honest baseline — beating chance, or
   beating an untrained model, proves nothing about whether adaptation
   was worth doing.
2. Fine-tune it on the domain corpus with LoRA.
3. Score it again on the same held-out pairs.

If step 3 does not beat step 1, the adaptation did not work, and that is
a result worth having rather than a failure to hide. Fine-tuning a
well-pretrained model on a narrow corpus can easily make it worse.

Nothing here trains the base weights. LoRA leaves them frozen, so the
comparison is between one model and itself plus a small adapter, rather
than between two models that differ in unknown ways.

Usage::

    python scripts/adapt_pretrained.py \\
        --checkpoint intfloat/multilingual-e5-small \\
        --pairs verify-output/hi-pairs.jsonl.gz \\
        --query-prefix "query: " --passage-prefix "passage: "
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Example:
    """One held-out or training pair, with its provenance."""

    anchor: str

    positive: str

    language: str | None = None

    kind: str | None = None

    overlap: float = 0.0


def load_pairs(path: Path, count: int, seed: int) -> list[Example]:
    """
    Read pairs and shuffle them.

    Shuffled before splitting because a mined pair file is in corpus
    order: the first N pairs come from the first few hundred articles,
    which is a topic sample rather than a random one.
    """

    opener = gzip.open if path.suffix == ".gz" else open

    rows: list[dict[str, Any]] = []

    with opener(path, "rt", encoding="utf-8") as handle:  # type: ignore[operator]
        for _, line in zip(range(count * 4), handle, strict=False):
            rows.append(json.loads(line))

    random.Random(seed).shuffle(rows)

    return [
        Example(
            anchor=row["anchor"],
            positive=row["positive"],
            language=row.get("language"),
            kind=row.get("kind"),
            overlap=float(row.get("overlap", 0.0)),
        )
        for row in rows[:count]
    ]


def prefixed(examples: list[Example], query: str, passage: str) -> list[Example]:
    """
    Apply the asymmetric prefixes a retrieval checkpoint expects.

    E5 and several other families are trained with distinct markers on
    the query and passage sides, and omitting them measurably degrades
    retrieval. The framework's ``TextEncoder`` contract has one
    ``encode_batch`` and no notion of which side a text is, so this is
    applied to the text before it reaches the encoder rather than by
    complicating the contract for one family's convention.
    """

    if not query and not passage:
        return examples

    return [
        Example(
            anchor=query + example.anchor,
            positive=passage + example.positive,
            language=example.language,
            kind=example.kind,
            overlap=example.overlap,
        )
        for example in examples
    ]


def report(label: str, scores: Any) -> None:
    print(f"  {label:24} recall@1 {scores.recall_at_1:.4f}   MRR {scores.mrr:.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--checkpoint", required=True, help="Model name or local directory")

    parser.add_argument("--pairs", type=Path, required=True, help="Mined pair file")

    parser.add_argument("--train-pairs", type=int, default=20000)

    parser.add_argument("--eval-pairs", type=int, default=2000)

    parser.add_argument("--epochs", type=int, default=1)

    parser.add_argument("--batch-size", type=int, default=64)

    parser.add_argument("--learning-rate", type=float, default=1e-4)

    parser.add_argument("--rank", type=int, default=16, help="LoRA rank")

    parser.add_argument(
        "--targets",
        default="query,value",
        help="Comma-separated module names LoRA attaches to",
    )

    parser.add_argument("--precision", default="bf16", choices=["fp32", "bf16"])

    parser.add_argument("--gradient-checkpoint-chunk", type=int, default=32)

    parser.add_argument("--pooling", default="mean", choices=["mean", "cls"])

    parser.add_argument("--query-prefix", default="", help='e.g. "query: " for E5')

    parser.add_argument("--passage-prefix", default="", help='e.g. "passage: " for E5')

    parser.add_argument("--max-length", type=int, default=256)

    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--output", type=Path, help="Write the comparison as JSON")

    arguments = parser.parse_args()

    from multilingual_embedding.embedding.neural import (
        ContrastiveConfig,
        ContrastiveTrainer,
        PretrainedTextEncoder,
        TextPair,
    )
    from multilingual_embedding.embedding.neural.lora import (
        LoRAConfig,
        apply_lora,
        parameter_summary,
    )
    from multilingual_embedding.evaluation.retrieval import evaluate_retrieval

    started = time.time()

    print("=" * 68)

    print(f"ADAPTING  {arguments.checkpoint}")

    print(f"STARTED   {time.strftime('%Y-%m-%d %H:%M:%S')}")

    print("=" * 68)

    total = arguments.train_pairs + arguments.eval_pairs

    examples = load_pairs(arguments.pairs, total, arguments.seed)

    if len(examples) < total:
        print(f"\nonly {len(examples)} pairs available; wanted {total}")

    train = prefixed(
        examples[arguments.eval_pairs :], arguments.query_prefix, arguments.passage_prefix
    )

    held = prefixed(
        examples[: arguments.eval_pairs], arguments.query_prefix, arguments.passage_prefix
    )

    print(f"\ntrain {len(train):,} pairs   held out {len(held):,} pairs")

    encoder = PretrainedTextEncoder.load(
        arguments.checkpoint,
        pooling=arguments.pooling,
        max_length=arguments.max_length,
    )

    print(f"device {encoder.device}   dimension {encoder.dimension}")

    # --- 1. the baseline -------------------------------------------------
    print("\n[1] scoring the checkpoint as published — this is the number to beat")

    # Kept so the run can tell "adaptation did not help" apart from
    # "adaptation did not happen". Without it, an unchanged score is
    # ambiguous, and the two have opposite remedies: a bigger learning
    # rate, or better pairs.
    probe = [example.anchor for example in held[:16]]

    probe_before = encoder.encode_batch(probe)

    before = evaluate_retrieval(encoder, held, limit=None)

    print(before.summary())

    # --- 2. adapt --------------------------------------------------------
    targets = tuple(name.strip() for name in arguments.targets.split(",") if name.strip())

    apply_lora(
        encoder._model, LoRAConfig(rank=arguments.rank, alpha=2 * arguments.rank, targets=targets)
    )

    summary = parameter_summary(encoder._model)

    share = summary["trainable"] / summary["total"] * 100

    print(
        f"\n[2] LoRA rank {arguments.rank} on {targets}: "
        f"{summary['trainable']:,} of {summary['total']:,} trainable ({share:.2f}%)"
    )

    training = ContrastiveTrainer(
        encoder,
        ContrastiveConfig(
            epochs=arguments.epochs,
            batch_size=arguments.batch_size,
            learning_rate=arguments.learning_rate,
            seed=arguments.seed,
            precision=arguments.precision,
            gradient_checkpoint_chunk=arguments.gradient_checkpoint_chunk,
        ),
    ).train([TextPair(example.anchor, example.positive) for example in train])

    print(f"    loss {training.initial_loss:.4f} -> {training.final_loss:.4f}")

    # --- 3. measure again ------------------------------------------------
    print("\n[3] scoring the adapted model on the same held-out pairs")

    after = evaluate_retrieval(encoder, held, limit=None)

    print(after.summary())

    # --- the comparison --------------------------------------------------
    print("\n" + "=" * 68)

    print("DID ADAPTATION HELP?")

    print("=" * 68)

    report("published checkpoint", before.overall)

    report("after LoRA adaptation", after.overall)

    delta = after.overall.recall_at_1 - before.overall.recall_at_1

    relative = (delta / before.overall.recall_at_1 * 100) if before.overall.recall_at_1 else 0.0

    verdict = "BETTER" if delta > 0 else ("NO CHANGE" if delta == 0 else "WORSE")

    print(f"\n  recall@1 {delta:+.4f}  ({relative:+.1f}%)   -> {verdict}")

    moved = float(abs(encoder.encode_batch(probe) - probe_before).max())

    print(f"  the model itself moved by {moved:.6f} (max change in a probe vector)")

    if delta == 0 and moved < 1e-4:
        print(
            "\n  The score did not change because the MODEL barely changed.\n"
            "  That is a training problem rather than a result: try a larger\n"
            "  --learning-rate, more --epochs, or a higher --rank before\n"
            "  concluding anything about the pairs."
        )
    elif delta <= 0:
        print(
            "\n  The model changed but the score did not improve. That is a\n"
            "  result, not a failure — fine-tuning a well-pretrained model on\n"
            "  a narrow corpus can make it worse, and pairs that are largely\n"
            "  solvable by string matching are a likely reason."
        )

    print("\n  by lexical overlap, before -> after:")

    for band in sorted(set(before.by_overlap) | set(after.by_overlap)):
        was = before.by_overlap.get(band)

        now = after.by_overlap.get(band)

        if was and now:
            print(
                f"    {band:14} {was.recall_at_1:.4f} -> {now.recall_at_1:.4f}  "
                f"({now.queries:,} queries)"
            )

    print(f"\nTOTAL {time.time() - started:.0f}s")

    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)

        arguments.output.write_text(
            json.dumps(
                {
                    "checkpoint": arguments.checkpoint,
                    "settings": vars(arguments) | {"pairs": str(arguments.pairs)},
                    "before": before.to_dict(),
                    "after": after.to_dict(),
                    "loss": [training.initial_loss, training.final_loss],
                    "trainable_share": share,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

        print(f"written to {arguments.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
