import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_len, dropout=0.15):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids):
        # input_ids: (B, T)
        _, T = input_ids.shape
        if T > self.max_seq_len:
            raise ValueError(
                f"Input sequence length {T} exceeds max_seq_len {self.max_seq_len}."
            )
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)  # (1, T)
        x = self.token_embed(input_ids) + self.pos_embed(positions)
        return self.dropout(x)
