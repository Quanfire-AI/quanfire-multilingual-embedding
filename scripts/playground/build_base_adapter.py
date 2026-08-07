#!/usr/bin/env python3
"""Build the *base* baseline the playground compares the shipped adapter against.

The playground's whole claim is "the LoRA weights, and nothing else, did this".
For that comparison to be honest the base must be served **exactly** the way the
adapter is — same checkpoint, same pooling, same max_length, same prefixes (the
shipped adapter is symmetric: it records empty prefixes), same normalisation. The
*only* thing that may differ is the trained low-rank weights.

The clean way to get that is an **untrained** LoRA adapter. ``LoRALinear`` zero-
initialises its up-projection (``nn.init.zeros_(self.lora_up.weight)``), so before
any training the adapter's contribution is ``scaling * up(down(x)) == 0`` and the
wrapped model is byte-for-byte the base checkpoint. Saving that untrained adapter
and serving it with ``qfme serve`` therefore reproduces the raw base — through the
identical serving stack the trained adapter uses — which is precisely the "before"
baseline behind reports/global-baseline-verdict.json.

Rather than hard-code the serving convention, this reads it from the adapter the
playground actually ships (``--like``): pooling, max_length, normalize, prefixes
and the LoRA shape are copied across, so the base sidecar can never drift from the
adapter sidecar. Only the weights are fresh (and zero-effect).

Run on the GPU box, once, before starting the base sidecar (needs the 'pretrained'
extra and the base checkpoint reachable — in the primed HF cache with
--local-files-only, or over the network):

    python scripts/playground/build_base_adapter.py \\
        --like /opt/quanfire/models/qme-model \\
        --out  /opt/quanfire/models/qme-base \\
        --revision 614241f622f53c4eeff9890bdc4f31cfecc418b3 \\
        --local-files-only

Then ``qfme serve --adapter /opt/quanfire/models/qme-base --port 8010`` serves the
base, and precompute_vectors.py embeds the corpus through both.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from multilingual_embedding.embedding.neural.adapter import save_adapter
from multilingual_embedding.embedding.neural.lora import LoRAConfig, apply_lora
from multilingual_embedding.embedding.neural.pretrained import PretrainedTextEncoder

# The base the shipped adapter is a LoRA over. Overridable, but this is the only
# base the published numbers were measured against.
DEFAULT_CHECKPOINT = "intfloat/multilingual-e5-small"
# The upstream revision the deploy pins the base to (see deploy/config.sh). Kept
# in sync so the base sidecar resolves the same weights everywhere.
DEFAULT_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"


def _serving_spec(like: Path) -> dict:
    """Read the serving convention from the deployed adapter's adapter.json.

    Falls back to the shipped prod-a70s30 values if a field is absent, so an
    older manifest still yields a faithful base.
    """
    manifest = json.loads((like / "adapter.json").read_text(encoding="utf-8"))
    lora = manifest.get("lora", {})
    return {
        "checkpoint": manifest.get("checkpoint", DEFAULT_CHECKPOINT),
        "pooling": manifest.get("pooling", "mean"),
        "max_length": int(manifest.get("max_length", 256)),
        "normalize": bool(manifest.get("normalize", True)),
        # The shipped adapter is symmetric — both prefixes empty. Copy whatever it
        # records so the base is served on identical terms.
        "query_prefix": manifest.get("query_prefix", ""),
        "passage_prefix": manifest.get("passage_prefix", ""),
        "rank": int(lora.get("rank", 32)),
        "alpha": int(lora.get("alpha", 64)),
        "dropout": float(lora.get("dropout", 0.0)),
        "targets": tuple(lora.get("targets", ("query", "value"))),
        # Prefer the revision the adapter itself pinned; else the CLI default.
        "checkpoint_revision": manifest.get("checkpoint_revision"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--like", type=Path, required=True,
                        help="the deployed adapter dir to mirror serving conventions from "
                             "(e.g. /opt/quanfire/models/qme-model)")
    parser.add_argument("--out", type=Path, required=True,
                        help="where to write the untrained base adapter (e.g. /opt/quanfire/models/qme-base)")
    parser.add_argument("--revision", default=None,
                        help=f"base checkpoint revision to pin (default: the adapter's own pin, "
                             f"else {DEFAULT_REVISION})")
    parser.add_argument("--local-files-only", action="store_true",
                        help="resolve the base only from the local HF cache")
    args = parser.parse_args()

    spec = _serving_spec(args.like)
    revision = args.revision or spec["checkpoint_revision"] or DEFAULT_REVISION

    print(f"mirroring {args.like}:")
    print(f"  checkpoint  {spec['checkpoint']} @ {revision}")
    print(f"  pooling     {spec['pooling']}   max_length {spec['max_length']}   "
          f"normalize {spec['normalize']}")
    print(f"  prefixes    query={spec['query_prefix']!r}  passage={spec['passage_prefix']!r}")
    print(f"  lora        rank {spec['rank']}  alpha {spec['alpha']}  "
          f"targets {list(spec['targets'])}")

    encoder = PretrainedTextEncoder.load(
        spec["checkpoint"],
        pooling=spec["pooling"],
        max_length=spec["max_length"],
        normalize=spec["normalize"],
        revision=revision,
        local_files_only=args.local_files_only,
    )

    lora = LoRAConfig(
        rank=spec["rank"],
        alpha=spec["alpha"],
        dropout=spec["dropout"],
        targets=spec["targets"],
    )
    adapted = apply_lora(encoder._model, lora)
    print(f"attached an untrained (zero-effect) adapter to {adapted} layers")

    save_adapter(
        encoder,
        args.out,
        lora=lora,
        data_provenance="public",
        query_prefix=spec["query_prefix"],
        passage_prefix=spec["passage_prefix"],
        checkpoint_revision=revision,
        notes={
            "role": "playground base baseline",
            "untrained": True,
            "explanation": (
                "Untrained LoRA (up-projection zero-initialised) over the base "
                "checkpoint: its contribution is exactly zero, so served vectors "
                "are the raw base. Exists so the playground serves the base through "
                "the same stack as the adapter, isolating the trained weights."
            ),
            "mirrors": str(args.like),
        },
    )
    print(f"\nwrote {args.out}")
    print(f"serve it:  qfme serve --adapter {args.out} --port 8010"
          + (" --local-files-only" if args.local_files_only else ""))


if __name__ == "__main__":
    main()
