"""
Dataset and DataLoader utilities for autoregressive language modeling.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple


class TextDataset(Dataset):
    """
    PyTorch Dataset for autoregressive language modeling.
    Returns (x, y) pairs where y is x shifted by one token.
    """

    def __init__(self, token_ids: List[int], block_size: int):
        """
        Args:
            token_ids: Full list of encoded token IDs.
            block_size: Context length (number of tokens per sample).
        """
        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.block_size = block_size

    def __len__(self) -> int:
        # Number of valid starting positions
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx: idx + self.block_size]
        y = self.data[idx + 1: idx + 1 + self.block_size]
        return x, y


def train_val_split(
    token_ids: List[int],
    val_fraction: float = 0.1,
) -> Tuple[List[int], List[int]]:
    """
    Split token IDs into training and validation sets.

    Args:
        token_ids: Full list of encoded token IDs.
        val_fraction: Fraction of data to use for validation (default 10%).

    Returns:
        Tuple of (train_ids, val_ids).
    """
    n = len(token_ids)
    split = int(n * (1 - val_fraction))
    return token_ids[:split], token_ids[split:]


def get_dataloader(
    token_ids: List[int],
    block_size: int,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    """
    Create a PyTorch DataLoader from token IDs.

    Args:
        token_ids: List of encoded token IDs.
        block_size: Context length.
        batch_size: Batch size.
        shuffle: Whether to shuffle the data.

    Returns:
        A PyTorch DataLoader yielding (x, y) batches.
    """
    dataset = TextDataset(token_ids, block_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        drop_last=True,
    )
