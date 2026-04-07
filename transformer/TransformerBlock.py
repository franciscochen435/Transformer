import torch.nn as nn
from .SelfAttention import SelfAttention
from .FeedForward import FeedForward

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, max_seq_len, dropout=0.15):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = SelfAttention(d_model, n_heads, max_seq_len, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)

    def forward(self, x):
        attn_out = self.attn(self.ln1(x))
        x = x + attn_out
        ff_out = self.ff(self.ln2(x))
        return x + ff_out
