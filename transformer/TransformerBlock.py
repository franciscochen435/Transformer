import torch
import torch.nn as nn
from .SelfAttention import SelfAttention
from .FeedForward import FeedForward

# A single Transformer decoder block
# Key Design:
#     - Pre-LayerNorm (better training stability)
#     - Residual connections (help gradient flow)
#     - combine self-attention and feedforward
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.15):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = SelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x
