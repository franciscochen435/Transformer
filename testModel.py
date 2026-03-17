import torch
from transformer.Embedding import Embedding

vocab_size = 1000
d_model = 64
max_seq_len = 10

embed = Embedding(vocab_size, d_model, max_seq_len)

x = torch.randint(0, vocab_size, (2, 5))  # (B=2, T=5)

out = embed(x)

print("input shape:", x.shape)
print("output shape:", out.shape)

x = torch.tensor([[5, 5, 5, 5]])

out = embed(x)

print(out[0,0])
print(out[0,1])
print(out[0,2])