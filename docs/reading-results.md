# Reading a retrieval report

How to tell a real result from a flattering one, using the reports
`scripts/adapt_pretrained.py` produces.

The order below is deliberate. Each step can disqualify the ones after it, so working
top-down stops you interpreting a number that was never valid.

---

## Step 1 — Is the measurement valid at all?

Before reading any score, three fields decide whether it means anything.

**`candidates` — the pool.** Recall@1 against 100 candidates and against 100,000 are
different tasks. A retrieval number without its pool size is uninterpretable. The runs so
far used ~2,000, which is small; a production index is millions, and scores will be lower
there. That is not a flaw in the measurement, but it is a limit on what it licenses you to
say.

**`random_recall_at_1` and `lift_over_chance`.** Chance is `1 / candidates`. At a pool of
1,994 that is 0.0005, so a recall@1 of 0.42 is 845× chance. This matters most when
comparing across runs with different pool sizes — raw recall is not comparable, lift is
closer.

**`dropped_duplicate_positives`.** Two queries sharing a passage cannot both be scored
correctly. They are removed; a large number here means the pair set needs deduplicating
before anything else is believed. Nine out of 2,000 is noise.

## Step 2 — Did anything actually change?

**Compare `before` and `after`, not `after` and zero.** The baseline is the published
checkpoint. Beating chance proves nothing; beating an untrained model proves nothing.
Beating the checkpoint you started from is the only comparison that says adaptation was
worth doing.

**Check `loss` has two different numbers.** With one epoch the first and last epoch are the
same epoch, so the report prints one value twice and `improved` is `false` regardless of
what happened. The `measurable` flag says whether the comparison exists at all. A real run
of this kind reported `[1.17487, 1.17487]` and `improved: false` while retrieval rose 20%.

**The loss is the weakest evidence in the report.** It falls whenever the model fits the
training objective, including when the objective is solvable by matching strings. Prefer
the retrieval scores.

## Step 3 — Convert percentages back to counts

This is the step most often skipped, and it changes conclusions.

A band's percentage hides how few queries produced it. Multiply `recall_at_1` by `queries`:

| band | queries | before | after | |
|---|---:|---|---|---|
| Hindi low `<0.3` | 155 | 22/155 | 54/155 | real |
| Hindi high `>0.7` | 628 | 366/628 | 395/628 | **within noise** |
| Hindi `title_lead` | 154 | 147/154 | 151/154 | **within noise** |
| Tamil low `<0.3` | 182 | 15/182 | 34/182 | real |
| Tamil high `>0.7` | 677 | 320/677 | 389/677 | real |

Hindi's high-overlap band moved 0.5828 → 0.6290, which reads as a confident +7.9%. It is
29 additional correct answers out of 628, and the 95% intervals overlap. It should not be
reported as a gain.

The report now carries `recall_at_1_hits` and `recall_at_1_ci95` on every group so this
check does not have to be done by hand. **Two results differ only if their intervals do
not overlap.**

## Step 4 — Is the change real, or is it string matching?

**Read `by_overlap` before believing anything in `overall`.**

Overlap is the share of the anchor's words that already appear in the passage. A pair at
0.9 can be answered by matching substrings, with no understanding of either text. A model
that learned only that will still show a falling loss and a rising overall recall.

So the shape of the gains is the test:

| pattern | reading |
|---|---|
| gains largest in **low** overlap | learned meaning — the hard cases improved |
| gains largest in **high** overlap | learned surface form — treat the headline as hollow |
| gains flat across bands | ambiguous; check the counts before concluding |

Both adapted models here gain most in the low band, and the pattern holds in an Indo-Aryan
and a Dravidian language. That is why the headline numbers are believable — not because
they are large.

**`by_kind` is a secondary version of the same check.** `title_lead` pairs are nearly
solved before any adaptation, because a Wikipedia lead restates its title; a model
improving mostly there has learned little.

## Step 5 — Which metric to weight

| metric | what it rewards | when to lead with it |
|---|---|---|
| `recall_at_1` | exactly right, first | the strictest, and the honest headline |
| `recall_at_10` | right answer somewhere in ten | when a reranker will see the top ten |
| `mrr` | position, harmonically | comparing runs whose recall@1 is close |
| `ndcg_at_10` | position, discounted | closest to a ranking-quality summary |

They usually move together. When they do not — recall@10 up while recall@1 is flat — the
model got better at *including* the answer without getting better at *ranking* it, which
matters if nothing reranks downstream.

## Step 6 — State the limits with the number

Every result so far carries these, and they belong in any claim made from it:

- **Two languages**, both Wikipedia.
- **~2,000 candidates**, not a production index.
- **Held-out pairs from the training distribution** — this measures in-domain adaptation,
  not generalisation to another task or corpus.
- **Absolute levels on the hardest slice are low.** Tamil low-overlap retrieval more than
  doubled and is still 0.1868.

A relative gain on a weak absolute base is still a real gain. It is not the same as a
solved problem, and the difference is worth stating before someone else does.

---

## A worked example

From `adapt-ta-v2.json`:

```
recall@1  0.3219 -> 0.4535   (+40.9%)
```

1. **Valid?** 1,991 candidates, chance 0.0005, 9 duplicates dropped. Yes.
2. **Changed?** Baseline is the published checkpoint, loss 1.0960 → 0.7749 over two
   epochs, so `measurable` is true. Yes.
3. **Counts?** Overall is 641 → 903 of 1,991. Comfortably outside noise.
4. **Real or lexical?** low `<0.3` +126.7%, mid +56.9%, high +21.6%. Gains run inversely
   to overlap. Real.
5. **Metrics agree?** recall@10 +32.2%, MRR +37.3%. Yes.
6. **Limits?** One language, Wikipedia, 1,991-candidate pool, in-domain.

**Conclusion:** adaptation improved Tamil retrieval by 40.9% on held-out in-domain pairs,
with gains concentrated where lexical matching cannot help. Not: "our model is 40% better
than E5."
