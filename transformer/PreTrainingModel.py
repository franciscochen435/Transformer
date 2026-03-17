import torch
import torch.nn as nn
from .Embedding import Embedding
from .TransformerBlock import TransformerBlock

# GPT-style Transformer (decoder-only architecture)
# Perform next-token prediction (language modeling)
#     - Output probability distribution over vocabulary for each position
class PreTrainingModel(nn.Module):
    def __init__(self, vocab_size, max_seq_len, d_model, n_heads, n_layers, d_ff, dropout=0.1):
        super().__init__()
        self.embed = Embedding(vocab_size, d_model, max_seq_len, dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)   # (B, T, vocab_size)
        return logits