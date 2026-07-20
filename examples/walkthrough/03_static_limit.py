"""Why the contextual encoder exists: word2vec gives 'bank' ONE vector."""

from multilingual_embedding.config.base import EmbeddingConfig
from multilingual_embedding.embedding.word2vec import Word2Vec

river = [
    "the river bank was muddy and wet",
    "we sat on the river bank at dawn",
    "the river bank flooded after rain",
    "grass grew along the river bank",
]
finance = [
    "the savings bank approved the loan",
    "she went to the savings bank today",
    "the savings bank raised its rates",
    "a savings bank holds your money",
]
sentences = (river + finance) * 40

model = Word2Vec(EmbeddingConfig(dimension=32, epochs=30, min_count=1, window=3, seed=0))
model.train(sentences)
m = model.matrix

print("Nearest neighbours of 'bank':")
for tok, score in m.most_similar("bank", top_k=4):
    print(f"   {tok:10} {score:.4f}")
print()
print(f"sim(river, savings)   = {m.similarity('river', 'savings'):.4f}")
print(f"occurrences of 'bank' = {sum(s.split().count('bank') for s in sentences)}")
print(f"rows in the matrix    = {len(m.vocabulary)}")
print()
print("'bank' occurs in both senses above, yet the model stores ONE row for it.")
print("There is no second vector to retrieve — the sense is not representable.")
