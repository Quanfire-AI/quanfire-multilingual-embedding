"""Domain adaptation without retraining the model."""

from multilingual_embedding.embedding.neural import EncoderConfig, TransformerEncoderModel
from multilingual_embedding.embedding.neural.lora import (
    LoRAConfig,
    apply_lora,
    lora_state_dict,
    parameter_summary,
)

# BERT-base shape, so the numbers mean something outside a toy.
cfg = EncoderConfig(vocabulary_size=30522, dimension=768, layers=12, heads=12, max_length=512)
model = TransformerEncoderModel(cfg)
before = sum(p.numel() for p in model.parameters())

RANK = 16  # every number below scales with this; rank 8 halves them
apply_lora(model, LoRAConfig(rank=RANK, alpha=2 * RANK))
s = parameter_summary(model)

print(f"rank                  {RANK:>14}")
print(f"total parameters      {s['total']:>14,}")
print(f"trainable parameters  {s['trainable']:>14,}")
print(f"trainable share       {s['trainable'] / s['total'] * 100:>13.2f}%")
print()


def mb(n):
    return n * 4 / 1024 / 1024


adapter = sum(t.numel() for t in lora_state_dict(model).values())
print(f"full model on disk    {mb(before):>10.1f} MB")
print(f"adapter on disk       {mb(adapter):>10.1f} MB")
print(f"Adam state, full      {mb(before * 2) / 1024:>10.2f} GB")
print(f"Adam state, LoRA      {mb(adapter * 2):>10.1f} MB")
print()
print("One base model + several small adapters = several domain models,")
print("each a few MB, all served from one set of frozen weights.")
