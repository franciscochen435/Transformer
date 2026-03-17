import torch
import torch.nn as nn

# Map/embed tokens to specific parameters which can contain the information 
# about position and similarity
# dropout: avoid overfitting
class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_len, dropout=0.1):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids):
        # input_ids: (B, T) (Batch size, Sequence length)
        B, T = input_ids.shape
        positions = torch.arange(0, T, device=input_ids.device).unsqueeze(0)  # (1, T)
        # the key formula is x = token_embedding + position_embedding
        x = self.token_embed(input_ids) + self.pos_embed(positions)
        # output_ids: (B, T, d_model)
        return self.dropout(x) 