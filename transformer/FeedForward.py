import torch
import torch.nn as nn

# Apply a non-linear transformation to each token independently
# Increase model capacity (adds depth beyond attention)
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            # Expand feature dimension
            nn.Linear(d_model, d_ff),
            # Non-linear activation
            nn.GELU(),
            # Project back to original dimension
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)