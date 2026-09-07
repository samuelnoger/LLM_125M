from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass
class CorpusEncoding:
    tokens: torch.Tensor
    vocab_size: int

class TextDataset(Dataset):
    def __init__(self, token_ids: torch.Tensor, block_size: int) -> None:
        if token_ids.ndim != 1:
            raise ValueError("token_ids must be a 1D tensor")
        if len(token_ids) <= block_size:
            raise ValueError("token_ids must be longer than block_size")
        self.token_ids = token_ids.long()
        self.block_size = block_size

    def __len__(self) -> int:
        # Non-overlapping chunk count
        return (len(self.token_ids) - 1) // self.block_size

    def __getitem__(self, index: int):
        # Jump ahead by block_size for every step
        start_idx = index * self.block_size
        chunk = self.token_ids[start_idx : start_idx + self.block_size + 1]
        return chunk[:-1], chunk[1:]
