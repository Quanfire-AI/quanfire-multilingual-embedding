"""
Using an adapted checkpoint: the four lines, and the trap in them.

Runs against ``models/indic-v1/`` — a LoRA adapter over
``intfloat/multilingual-e5-small``, trained on mined Hindi and Tamil
Wikipedia pairs.

Usage::

    uv run python examples/use_adapter.py

Needs ``--extra neural --extra pretrained``, an adapter at
``models/indic-v1/`` (see :file:`models/README.md`), and the base
checkpoint — fetched on first run, then cached.

The script shows two ways to use the model and the reason to prefer the
second. Nothing here is an evaluation: a handful of sentences measures
nothing. See :file:`docs/reading-results.md` for what a real number
requires.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from multilingual_embedding.embedding.neural import load_adapter
from multilingual_embedding.pipelines.search import SemanticSearchPipeline

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent

ADAPTER = REPOSITORY_ROOT / "models" / "indic-v1"

QUERY = "मशीन लर्निंग क्या है"

# One of these answers the query; the rest are near misses on topic, so
# a model that only matched surface words would not obviously win.
PASSAGES: tuple[str, ...] = (
    "मशीन लर्निंग कृत्रिम बुद्धिमत्ता की एक शाखा है जिसमें सिस्टम डेटा से सीखते हैं।",
    "भारत का संविधान 26 जनवरी 1950 को लागू हुआ था।",
    "இயந்திர கற்றல் என்பது தரவிலிருந்து கற்றுக்கொள்ளும் அமைப்புகளைப் பற்றியது.",
    "कंप्यूटर हार्डवेयर में प्रोसेसर और मेमोरी शामिल होते हैं।",
)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Similarity between two vectors that may or may not be normalised."""

    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> None:
    if not ADAPTER.is_dir():
        raise SystemExit(f"No adapter at {ADAPTER}. See models/README.md for where to put one.")

    # ------------------------------------------------------------------
    # 1. Load. The metadata is returned alongside the encoder, not as an
    #    afterthought — the prefixes are needed to use it correctly, and
    #    returning the encoder alone would invite forgetting them.
    # ------------------------------------------------------------------
    encoder, meta = load_adapter(ADAPTER)

    print(f"base checkpoint   {meta.checkpoint}")

    print(f"dimension         {encoder.dimension}")

    print(f"prefixes          {meta.query_prefix!r} / {meta.passage_prefix!r}")

    print(f"adapter size      {meta['adapter_parameters']:,} parameters")

    notes = meta.get("notes", {})

    if "recall_at_1_before" in notes:
        print(
            f"recorded run      recall@1 {notes['recall_at_1_before']} "
            f"-> {notes['recall_at_1_after']} on {notes.get('held_out_languages')}"
        )

    # ------------------------------------------------------------------
    # 2. The raw path. Correct, and it is the caller's job to keep it
    #    correct — the prefixes are applied by hand, and asymmetrically:
    #    the query gets one marker and the passages get the other.
    # ------------------------------------------------------------------
    print("\n--- encoder directly, prefixes applied by hand ---")

    query_vector = encoder.encode(meta.query_prefix + QUERY)

    passage_vectors = encoder.encode_batch([meta.passage_prefix + passage for passage in PASSAGES])

    for passage, vector in zip(PASSAGES, passage_vectors, strict=True):
        print(f"  {cosine(query_vector, vector):+.4f}  {passage[:52]}")

    # ------------------------------------------------------------------
    # 3. What dropping the prefixes costs. On four sentences with one
    #    obvious answer, usually nothing visible — the right passage
    #    still wins. That is the point. The failure does not raise, the
    #    vectors stay the right shape and finite, and the top hit stays
    #    correct on easy queries; what degrades is the *margin*, which
    #    only shows up as lost recall at scale, on the hard queries,
    #    long after the code shipped. A demonstration that broke loudly
    #    here would be a less honest picture of the risk.
    # ------------------------------------------------------------------
    print("\n--- the same, with the prefixes forgotten ---")

    bare_query = encoder.encode(QUERY)

    bare_passages = encoder.encode_batch(list(PASSAGES))

    for passage, vector in zip(PASSAGES, bare_passages, strict=True):
        print(f"  {cosine(bare_query, vector):+.4f}  {passage[:52]}")

    with_prefix = [cosine(query_vector, v) for v in passage_vectors]

    without_prefix = [cosine(bare_query, v) for v in bare_passages]

    print(f"\n  best match with prefixes:    {PASSAGES[int(np.argmax(with_prefix))][:52]}")

    print(f"  best match without them:     {PASSAGES[int(np.argmax(without_prefix))][:52]}")

    kept = np.diff(sorted(with_prefix)[-2:])[0]

    lost = np.diff(sorted(without_prefix)[-2:])[0]

    print(f"  margin over runner-up:       {kept:+.4f} with, {lost:+.4f} without")

    print(
        "  the winner is the same either way — what shrinks is the gap, "
        "which is why this goes unnoticed"
        if int(np.argmax(with_prefix)) == int(np.argmax(without_prefix))
        else "  the winner changed — the prefixes were doing visible work here"
    )

    # ------------------------------------------------------------------
    # 4. The path to prefer. from_adapter reads the same metadata and
    #    applies both prefixes on the right sides, so there is nothing
    #    left to get wrong.
    # ------------------------------------------------------------------
    print("\n--- through the search pipeline ---")

    pipeline = SemanticSearchPipeline.from_adapter(ADAPTER)

    pipeline.index(PASSAGES)

    for hit in pipeline.search(QUERY, top_k=3):
        print(f"  {hit.rank}. {hit.score:+.4f}  {hit.text[:52]}")

    print(f"\n  pipeline prefixes: {pipeline.prefixes}")


if __name__ == "__main__":
    main()
