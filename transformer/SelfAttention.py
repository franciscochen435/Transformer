import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *

# For each token, calculate the dependency with other tokens. 
class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.15):
        super().__init__()
        # it is multi-heap self-attention, making sure each head has the same dimension
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # q: query, k: key, v: value
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, x):
        B, T, C = x.shape

        Q = self.q_proj(x)  # (B, T, C)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # (B, H, T, D)
        # Batch, #head, #token, #dim in each head
        Q = Q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, T, T)

        # Casual mask, making sure the model can only view the previous value
        mask = self.mask[:, :, :T, :T]
        scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = attn @ V  # (B, H, T, D)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)

        return out
