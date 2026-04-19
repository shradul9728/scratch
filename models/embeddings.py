"""
Token and Positional Embeddings for the GPT model.
"""
import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """
    Standard token embedding layer.
    Maps integer token IDs to dense vectors of dimension d_model.
    """

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len) integer token IDs.

        Returns:
            (batch, seq_len, d_model) token embeddings.
        """
        return self.embedding(x)


class PositionalEmbedding(nn.Module):
    """
    Learnable positional embedding.
    Maps position indices [0, block_size) to dense vectors of dimension d_model.
    """

    def __init__(self, block_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(block_size, d_model)

    def forward(self, seq_len: int, device: torch.device = None) -> torch.Tensor:
        """
        Args:
            seq_len: Length of the sequence to generate positions for.
            device: Device to create position indices on.

        Returns:
            (seq_len, d_model) positional embeddings (broadcast-ready over batch dim).
        """
        positions = torch.arange(seq_len, device=device)
        return self.embedding(positions)
