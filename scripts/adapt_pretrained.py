"""
Adapt a published encoder to a domain, and measure whether it helped.

**This script is now a thin front end.** The experiment lives in
``multilingual_embedding.pipelines.adaptation`` and is reachable as
``qfme adapt``, which takes a ``--config`` and a ``--profile`` and is
therefore reproducible from a file that can be committed and diffed.
Prefer it. This is kept because every result in ``ROADMAP.md`` was
produced by a command line of this shape, and those commands should keep
working exactly as they did.

The three steps it runs:

1. Score a published checkpoint on held-out pairs. **This is the number
   to beat**, and it is the only honest baseline — beating chance, or
   beating an untrained model, proves nothing about whether adaptation
   was worth doing.
2. Fine-tune it on the domain corpus with LoRA.
3. Score it again on the same held-out pairs.

If step 3 does not beat step 1, the adaptation did not work, and that is
a result worth having rather than a failure to hide.

**Which experiment is being run is declared, not inferred.** The same
three steps answer several different questions depending on what is held
fixed between training and evaluation:

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
run whose label and data disagree is refused before it starts.

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

The same three runs as ``qfme adapt``, which additionally accepts
``--config`` and ``--profile``::

    qfme adapt --config experiments/indic.yaml --profile configs/gpu.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from multilingual_embedding.common.enums import DataProvenance
from multilingual_embedding.config.base import (
    ADAPTATIONS,
    AdaptationConfig,
    ComputeConfig,
    ExperimentConfig,
)
from multilingual_embedding.core.exceptions import MultilingualEmbeddingError
from multilingual_embedding.pipelines.adaptation import AdaptationPipeline


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

    parser.add_argument(
        "--data-provenance",
        default="",
        choices=[str(value) for value in DataProvenance],
        help=(
            "Where the training data came from: public, synthetic or "
            "licensed. Required the moment --save-adapter is set, because a "
            "saved model carries this as a legal fact about itself and there "
            "is no default that could answer it by omission"
        ),
    )

    arguments = parser.parse_args()

    config = ExperimentConfig(
        name=Path(arguments.checkpoint).name or "adaptation",
        seed=arguments.seed,
        adaptation=AdaptationConfig(
            checkpoint=arguments.checkpoint,
            pairs=arguments.pairs,
            eval_pairs_file=arguments.eval_pairs_file,
            train_pairs=arguments.train_pairs,
            eval_pairs=arguments.eval_pairs,
            sample_pairs=arguments.sample_pairs,
            train_kinds=arguments.train_kinds,
            eval_kinds=arguments.eval_kinds,
            train_languages=arguments.train_languages,
            eval_languages=arguments.eval_languages,
            adaptation=arguments.adaptation,
            epochs=arguments.epochs,
            learning_rate=arguments.learning_rate,
            rank=arguments.rank,
            targets=arguments.targets,
            pooling=arguments.pooling,
            max_length=arguments.max_length,
            query_prefix=arguments.query_prefix,
            passage_prefix=arguments.passage_prefix,
            save_adapter=arguments.save_adapter,
            data_provenance=arguments.data_provenance,
            report=arguments.output,
            seed=arguments.seed,
        ),
        # The three settings a machine dictates. `qfme adapt` reads these
        # from a --profile instead, which is the whole reason it exists.
        compute=ComputeConfig(
            precision=arguments.precision,
            batch_size=arguments.batch_size,
            gradient_checkpoint_chunk=arguments.gradient_checkpoint_chunk,
        ),
    )

    print("=" * 68)

    print(f"ADAPTING  {arguments.checkpoint}")

    print(f"STARTED   {time.strftime('%Y-%m-%d %H:%M:%S')}")

    print("=" * 68)

    try:
        result = AdaptationPipeline(config).run()
    except MultilingualEmbeddingError as error:
        # A declaration that disagrees with the filters, or a facet filter
        # that selects nothing. Both are the user's mistake and neither
        # deserves a traceback.
        raise SystemExit(str(error)) from error

    print("\n" + "=" * 68)

    print("DID ADAPTATION HELP?")

    print("=" * 68)

    print(result.summary())

    return 0


if __name__ == "__main__":
    sys.exit(main())
