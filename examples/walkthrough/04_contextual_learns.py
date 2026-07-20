"""The contextual encoder: prove it learns, don't assume it."""

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
    """The one attribute the encoder needs from a tokenizer."""

    def __init__(self, ids: list[int]) -> None:
        self.ids = ids


class Tok:
    """
    A hashing tokenizer, so this demo needs no trained subword model.

    Uses blake2b rather than the built-in ``hash``. Python randomises
    string hashing per process, so ``hash`` would assign different ids on
    every run and the numbers printed below would not reproduce — which
    is precisely what a demonstration must not do.
    """

    def encode(self, text: str) -> Encoding:
        return Encoding([1 + self._id(word) % 62 for word in text.split()])

    @staticmethod
    def _id(word: str) -> int:
        digest = hashlib.blake2b(word.encode("utf-8"), digest_size=4).digest()

        return int.from_bytes(digest, "big")


torch.manual_seed(0)
model = TransformerEncoderModel(
    EncoderConfig(vocabulary_size=64, dimension=32, layers=2, heads=4, max_length=32, dropout=0.1)
)
enc = NeuralTextEncoder(model, Tok(), device="cpu")
print(f"parameters: {model.parameter_count():,}   device: {enc.device}")

weather = ["rain", "storm", "cloud", "wind", "thunder", "drizzle"]
finance = ["bank", "loan", "credit", "market", "invoice", "ledger"]
pairs = []
for i in range(24):
    a, b = weather[i % 6], weather[(i + 3) % 6]
    pairs.append(TextPair(f"{a} {b}", f"{b} {a} today"))
    c, d = finance[i % 6], finance[(i + 3) % 6]
    pairs.append(TextPair(f"{c} {d}", f"{d} {c} today"))


def margin():
    w = enc.encode_batch([f"{x} {y}" for x in weather for y in weather if x != y])
    f = enc.encode_batch([f"{x} {y}" for x in finance for y in finance if x != y])
    within = (w @ w.T).mean() / 2 + (f @ f.T).mean() / 2
    cross = (w @ f.T).mean()
    return within, cross


w0, c0 = margin()
print(f"\nBEFORE training: within-topic {w0:+.3f}  cross-topic {c0:+.3f}  margin {w0 - c0:.3f}")

report = ContrastiveTrainer(
    enc, ContrastiveConfig(epochs=24, batch_size=8, learning_rate=3e-3, seed=1)
).train(pairs)

w1, c1 = margin()
print(f"AFTER  training: within-topic {w1:+.3f}  cross-topic {c1:+.3f}  margin {w1 - c1:.3f}")
print(f"\nloss {report.initial_loss:.3f} -> {report.final_loss:.3f}  over {report.steps} steps")
print(f"improved: {report.improved}")
print("\nA model that learned nothing would keep margin near zero.")
