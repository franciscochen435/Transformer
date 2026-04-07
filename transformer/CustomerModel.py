import torch.nn as nn
from .Embedding import Embedding
from .TransformerBlock import TransformerBlock

class CustomerModel(nn.Module):
    def __init__(self, vocab_size, max_seq_len, d_model, n_heads, n_layers, d_ff, dropout=0.15):
        super().__init__()
        self.embed = Embedding(vocab_size, d_model, max_seq_len, dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, max_seq_len, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids):
        if input_ids.dim() != 2:
            raise ValueError(
                f"input_ids must have shape (B, T), got shape {tuple(input_ids.shape)}."
            )
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        return logits
