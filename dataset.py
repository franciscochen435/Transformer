import torch
from torch.utils.data import Dataset

class LMDataset(Dataset):
    def __init__(self, token_ids, seq_len, stride=64):
        self.token_ids = token_ids
        self.seq_len = seq_len
        self.stride = stride

    def __len__(self):
        if len(self.token_ids) < self.seq_len + 1:
            return 0
        return (len(self.token_ids) - self.seq_len - 1) // self.stride + 1

    def __getitem__(self, idx):
        start = idx * self.stride
        chunk = self.token_ids[start : start + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y
