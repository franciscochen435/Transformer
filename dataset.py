import torch
from torch.utils.data import Dataset

class LMDataset(Dataset):
    def __init__(self, token_ids, seq_len, stride=64):
        if seq_len <= 0:
            raise ValueError(f"seq_len must be > 0, got {seq_len}.")
        if stride <= 0:
            raise ValueError(f"stride must be > 0, got {stride}.")
        self.token_ids = token_ids
        self.seq_len = seq_len
        self.stride = stride

    def __len__(self):
        if len(self.token_ids) < self.seq_len + 1:
            return 0
        return (len(self.token_ids) - self.seq_len - 1) // self.stride + 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}.")

        start = idx * self.stride
        chunk = self.token_ids[start : start + self.seq_len + 1]
        if len(chunk) != self.seq_len + 1:
            raise ValueError("Unexpected chunk length; check __len__ and slicing logic.")

        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y
