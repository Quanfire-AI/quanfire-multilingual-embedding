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

**Which experiment is being run is declared, not inferred.** The same
three steps answer several different questions depending on what is held
fixed between training and evaluation, and the questions are not
interchangeable:

===================  ==========================================
``--adaptation``     what it measures
===================  ==========================================
``in-distribution``  how much adaptation helps where it trained
``task``             whether it learned retrieval or the mining
                     scheme — different pair kinds, same corpus
``language``         whether it crosses scripts — same kinds,
                     different languages
``domain``           whether it survives your own text — same
                     kinds, a different pair file
===================  ==========================================

The declaration is checked against what the filters actually do, and a
run whose label and data disagree is refused before it starts. A report
labelled `task` that quietly held the kinds identical is worse than no
report, because the label is what gets quoted later.

Usage::

    # in-distribution: the default, nothing varies
    python scripts/adapt_pretrained.py \\
        --checkpoint intfloat/multilingual-e5-small \\
        --pairs verify-output/hi-pairs.jsonl.gz \\
        --query-prefix "query: " --passage-prefix "passage: "

    # task adaptation: train on one pair shape, score on another
    python scripts/adapt_pretrained.py ... \\
        --adaptation task \\
        --train-kinds adjacent --eval-kinds heading_section

    # domain adaptation: train on Wikipedia, score on your own corpus
    python scripts/adapt_pretrained.py ... \\
        --adaptation domain \\
        --pairs wiki-pairs.jsonl.gz --eval-pairs-file contracts.jsonl.gz
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
    Draw a uniform sample from the whole file, by reservoir sampling.

    Reading the first ``count * 4`` lines and shuffling those was wrong in
    a way that produced a plausible result. A mined pair file is in corpus
    order, and two concatenated files are in language order: for a Hindi
    and Tamil pair set joined together, the window covered the first
    168,000 of 642,536 Hindi lines and never reached a Tamil pair. The
    run reported `by_language: {"hi": ...}` and was read as a joint
    experiment.

    Reservoir sampling makes every line equally likely to be chosen in one
    pass, holding only the sample. It costs a full read of the file, which
    is seconds against a training run, and it cannot silently exclude
    whatever happens to sit past an arbitrary cut-off.
    """

    opener = gzip.open if path.suffix == ".gz" else open

    rng = random.Random(seed)

    reservoir: list[dict[str, Any]] = []

    seen = 0

    with opener(path, "rt", encoding="utf-8") as handle:  # type: ignore[operator]
        for line in handle:
            seen += 1

            if len(reservoir) < count:
                reservoir.append(json.loads(line))

                continue

            # Each later line replaces a random held one with probability
            # count/seen, which leaves every line equally represented.
            index = rng.randrange(seen)

            if index < count:
                reservoir[index] = json.loads(line)

    rng.shuffle(reservoir)

    return [
        Example(
            anchor=row["anchor"],
            positive=row["positive"],
            language=row.get("language"),
            kind=row.get("kind"),
            overlap=float(row.get("overlap", 0.0)),
        )
        for row in reservoir
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


def only(examples: list[Example], wanted: str, facet: str) -> list[Example]:
    """
    Keep examples whose ``facet`` is named, or everything when none are.

    Two facets are filterable, and which one is varied decides what the
    run measures:

    ``kind``
        The shape of the retrieval task. `adjacent` retrieves the
        paragraph following a paragraph; `title_lead` retrieves a lead
        from a title. Same text, different question asked of it.

    ``language``
        The corpus the text comes from. Same question, different text.

    Vary one and hold the other fixed, and the result attributes itself.
    Vary both and it cannot.
    """

    names = {name.strip() for name in wanted.split(",") if name.strip()}

    if not names:
        return examples

    kept = [example for example in examples if getattr(example, facet) in names]

    if not kept:
        available = sorted({getattr(e, facet) for e in examples if getattr(e, facet)})

        raise SystemExit(f"no pairs with {facet} in {sorted(names)}; the file holds {available}")

    return kept


def values(examples: list[Example], facet: str) -> list[str]:
    """The distinct values of a facet actually present, for reporting."""

    return sorted({getattr(e, facet) for e in examples if getattr(e, facet)})


def varying(train: list[str], held: list[str]) -> bool:
    """
    Whether a facet is varied between training and evaluation.

    Varied means disjoint. Partial overlap is neither held fixed nor
    varied, and a run built on it answers no clean question, so it is
    reported as not varying and the declaration check rejects it.
    """

    return bool(train and held and not set(train) & set(held))


# What each declared experiment requires to vary. Everything not listed
# for a mode must be held fixed, and the check is two-sided: a `task` run
# whose languages also differ is not a task result, it is two changes at
# once with one name.
ADAPTATIONS: dict[str, tuple[str, ...]] = {
    "in-distribution": (),
    "task": ("kind",),
    "language": ("language",),
    "domain": ("corpus",),
    "task+language": ("kind", "language"),
    "task+domain": ("kind", "corpus"),
}

_DESCRIPTIONS = {
    "in-distribution": "same task, same corpus — how much adaptation helps where it was trained",
    "task": "same corpus, different task shape — did it learn retrieval or the mining scheme",
    "language": "same task, different language — does the adaptation cross scripts",
    "domain": "same task, different corpus — does it survive contact with your own text",
    "task+language": "task shape and language both change",
    "task+domain": "task shape and corpus both change",
}


def check_adaptation(declared: str, observed: set[str]) -> None:
    """
    Refuse a run whose filters do not implement the experiment it claims.

    The declaration is the point. Naming a run `--adaptation task` and
    then leaving the kinds identical produces a report labelled as a
    transfer result that measured nothing of the sort, and the label is
    what gets quoted six months later. So the label is checked against
    what the data actually does, and disagreement stops the run before it
    spends an hour on a GPU.
    """

    required = set(ADAPTATIONS[declared])

    if observed == required:
        return

    missing = sorted(required - observed)

    extra = sorted(observed - required)

    problems = []

    if missing:
        problems.append(
            f"declared --adaptation {declared}, but {', '.join(missing)} is held fixed "
            f"rather than varied"
        )

    if extra:
        problems.append(
            f"declared --adaptation {declared}, but {', '.join(extra)} varies too, so the "
            f"result cannot be attributed to one of them"
        )

    raise SystemExit(
        "\n".join(problems) + "\n\nvary a facet by giving --train-* and --eval-* disjoint values "
        "(kind, language) or different files (corpus, via --eval-pairs-file).\n"
        f"declared modes: {', '.join(sorted(ADAPTATIONS))}"
    )


def report(label: str, scores: Any) -> None:
    print(f"  {label:24} recall@1 {scores.recall_at_1:.4f}   MRR {scores.mrr:.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--checkpoint", required=True, help="Model name or local directory")

    parser.add_argument("--pairs", type=Path, required=True, help="Mined pair file to train on")

    parser.add_argument(
        "--eval-pairs-file",
        type=Path,
        help=(
            "Score against this file instead of --pairs. Hold it fixed to "
            "compare training sets: without it, a run that changes what it "
            "trains on also changes what it is judged by, and the two "
            "cannot be separated"
        ),
    )

    parser.add_argument("--train-pairs", type=int, default=20000)

    parser.add_argument("--eval-pairs", type=int, default=2000)

    parser.add_argument(
        "--sample-pairs",
        type=int,
        help=(
            "How many pairs to draw from the file before filtering by kind. "
            "Defaults to --train-pairs plus --eval-pairs, which is right when "
            "no kind filter is set and wrong when one is: a kind holding a "
            "sixth of the file yields a sixth of the sample, so --train-pairs "
            "stops binding and two runs that name different kinds train on "
            "different amounts of data. Set this several times --train-pairs "
            "to make the cap bind for every kind"
        ),
    )

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

    parser.add_argument(
        "--train-kinds",
        default="",
        help=(
            "Comma-separated pair kinds to train on, e.g. 'adjacent'. "
            "Empty uses all. Set this and --eval-kinds to different values "
            "to test whether the adaptation generalises across task shapes "
            "or only learned the one it was shown"
        ),
    )

    parser.add_argument(
        "--eval-kinds",
        default="",
        help="Comma-separated pair kinds to score on. Empty uses all",
    )

    parser.add_argument(
        "--train-languages",
        default="",
        help=(
            "Comma-separated languages to train on, e.g. 'hi'. Empty uses "
            "all. The domain axis: set this and --eval-languages to "
            "different values to measure whether the adaptation crosses "
            "corpora rather than only the task shape"
        ),
    )

    parser.add_argument(
        "--eval-languages",
        default="",
        help="Comma-separated languages to score on. Empty uses all",
    )

    parser.add_argument(
        "--adaptation",
        default="in-distribution",
        choices=sorted(ADAPTATIONS),
        help=(
            "What this run claims to measure. Checked against what the "
            "filters actually do, and the run is refused if they disagree — "
            "the label outlives the command line, so it must not be able to "
            "be wrong"
        ),
    )

    parser.add_argument(
        "--save-adapter",
        type=Path,
        help=(
            "Directory to write the trained adapter to. Without this the "
            "adapter is discarded when the process exits and the run has "
            "produced a measurement rather than a model"
        ),
    )

    arguments = parser.parse_args()

    from multilingual_embedding.embedding.neural import (
        ContrastiveConfig,
        ContrastiveTrainer,
        PretrainedTextEncoder,
        TextPair,
        save_adapter,
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

    total = arguments.sample_pairs or (arguments.train_pairs + arguments.eval_pairs)

    examples = load_pairs(arguments.pairs, total, arguments.seed)

    if len(examples) < total:
        print(f"\nonly {len(examples)} pairs available; wanted {total}")

    # The held-out set is drawn from a fixed-size sample of its own, so
    # that changing --train-pairs does not change what the model is
    # scored on. It moved before, and two Hindi runs reported baselines
    # of 0.4238 and 0.4764 for the same checkpoint — a difference in the
    # measuring stick that looked like a difference in the model.
    evaluation_source = arguments.eval_pairs_file or arguments.pairs

    evaluation = load_pairs(evaluation_source, arguments.eval_pairs, arguments.seed + 10_000)

    held_texts = {example.positive for example in evaluation}

    train_pool = [example for example in examples if example.positive not in held_texts]

    selected = only(
        only(train_pool, arguments.train_kinds, "kind"),
        arguments.train_languages,
        "language",
    )

    train = prefixed(
        selected[: arguments.train_pairs],
        arguments.query_prefix,
        arguments.passage_prefix,
    )

    held = prefixed(
        only(
            only(evaluation, arguments.eval_kinds, "kind"),
            arguments.eval_languages,
            "language",
        ),
        arguments.query_prefix,
        arguments.passage_prefix,
    )

    train_kinds, eval_kinds = values(train, "kind"), values(held, "kind")

    train_languages, languages = values(train, "language"), values(held, "language")

    print(f"\ntrain {len(train):,} pairs   held out {len(held):,} pairs")

    # Laid out as a table because the whole design of the run is which row
    # differs. Reading it off two prose sentences is how a run ends up
    # varying two things at once without anyone noticing.
    print(f"\n  {'facet':10} {'trained on':32} {'scored on':32}")

    observed: set[str] = set()

    axes = (
        ("kind", train_kinds, eval_kinds),
        ("language", train_languages, languages),
        # File names rather than full paths, so a deep --pairs path cannot
        # push the VARIES column off the terminal — which is the one thing
        # in this table anybody needs to read.
        ("corpus", [Path(arguments.pairs).name], [Path(evaluation_source).name]),
    )

    for facet, left, right in axes:
        differs = varying(left, right)

        if differs:
            observed.add(facet)

        marker = "VARIES" if differs else "fixed"

        print(f"  {facet:10} {','.join(left) or '-':30.30} {','.join(right) or '-':30.30} {marker}")

    check_adaptation(arguments.adaptation, observed)

    print(f"\n  --adaptation {arguments.adaptation}: {_DESCRIPTIONS[arguments.adaptation]}")

    # A facet filter that starves the training set turns a comparison of
    # task shapes into a comparison of training-set sizes. Said out loud,
    # because it is invisible in the scores and it silently decided a run:
    # `--train-pairs 40000 --train-kinds title_lead` drew about 7,000
    # pairs from a 42,000 sample while `--train-kinds adjacent` drew
    # 25,000 from the same sample.
    filtered = arguments.train_kinds or arguments.train_languages

    if filtered and len(train) < arguments.train_pairs:
        print(
            f"\n  WARNING: asked for {arguments.train_pairs:,} training pairs and the "
            f"filters left {len(train):,}.\n"
            f"  Raise --sample-pairs (currently {total:,}) until this stops, or runs "
            f"naming different facets will differ in data volume as well as in shape."
        )

    if arguments.eval_pairs_file:
        print(f"scored against {evaluation_source} (held fixed)")

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

    lora = LoRAConfig(rank=arguments.rank, alpha=2 * arguments.rank, targets=targets)

    apply_lora(encoder._model, lora)

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

    if training.measurable:
        print(f"    loss {training.initial_loss:.4f} -> {training.final_loss:.4f}")
    else:
        print(
            f"    loss {training.final_loss:.4f} (one epoch, so there is nothing "
            f"to compare it against — the retrieval scores below are the evidence)"
        )

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

    if arguments.save_adapter:
        # Saved after scoring, with the scores in it, so the artefact
        # carries the evidence for itself rather than pointing at a
        # report that may not travel with it.
        save_adapter(
            encoder,
            arguments.save_adapter,
            lora=lora,
            query_prefix=arguments.query_prefix,
            passage_prefix=arguments.passage_prefix,
            notes={
                "trained_on": str(arguments.pairs),
                "scored_against": str(evaluation_source),
                "train_pairs": len(train),
                "adaptation": arguments.adaptation,
                "train_kinds": train_kinds,
                "train_languages": train_languages,
                "held_out_languages": languages,
                "recall_at_1_before": round(before.overall.recall_at_1, 4),
                "recall_at_1_after": round(after.overall.recall_at_1, 4),
                "epochs": arguments.epochs,
                "learning_rate": arguments.learning_rate,
            },
        )

        print(f"\nadapter written to {arguments.save_adapter}")

        print(f"  reload with: load_adapter({str(arguments.save_adapter)!r})")

    print(f"\nTOTAL {time.time() - started:.0f}s")

    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)

        arguments.output.write_text(
            json.dumps(
                {
                    "checkpoint": arguments.checkpoint,
                    "trained_on": str(arguments.pairs),
                    "scored_against": str(evaluation_source),
                    "held_out_languages": languages,
                    "adaptation": arguments.adaptation,
                    "measures": _DESCRIPTIONS[arguments.adaptation],
                    "varying_facets": sorted(observed),
                    "train_kinds": train_kinds,
                    "eval_kinds": eval_kinds,
                    "train_languages": train_languages,
                    "eval_languages": languages,
                    # The counts actually used, not the ones requested. A
                    # kind filter can cut either far below its flag, and a
                    # report that records only the request describes a run
                    # that did not happen.
                    "train_examples": len(train),
                    "eval_examples": len(held),
                    "sampled_from": total,
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
