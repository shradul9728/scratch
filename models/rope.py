"""
Rotary Positional Embeddings (RoPE).
Applies rotary transforms to Q and K tensors for relative position encoding.
"""
import torch
import torch.nn as nn


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE).
    Precomputes sin/cos frequency tables and applies rotary transforms per-head.
    """

    def __init__(self, d_k: int, max_seq_len: int = 4096, base: float = 10000.0):
        """
        Args:
            d_k: Dimension per attention head (must be even).
            max_seq_len: Maximum supported sequence length.
            base: Base for the frequency computation.
        """
        super().__init__()
        assert d_k % 2 == 0, "d_k must be even for RoPE"

        # Compute frequency bands: theta_i = 1 / (base^(2i/d_k)) for i in [0, d_k/2)
        freqs = 1.0 / (base ** (torch.arange(0, d_k, 2).float() / d_k))
        self.register_buffer('freqs', freqs)  # (d_k/2,)

        # Precompute positions
        positions = torch.arange(max_seq_len).float()
        # Outer product: (max_seq_len, d_k/2)
        angles = torch.outer(positions, freqs)
        self.register_buffer('cos_cache', angles.cos())  # (max_seq_len, d_k/2)
        self.register_buffer('sin_cache', angles.sin())  # (max_seq_len, d_k/2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary embedding to input tensor.

        Args:
            x: (..., seq_len, d_k) — typically Q or K after head splitting.

        Returns:
            Rotated tensor of the same shape.
        """
        seq_len = x.size(-2)
        cos = self.cos_cache[:seq_len]  # (T, d_k/2)
        sin = self.sin_cache[:seq_len]  # (T, d_k/2)

        # Split x into pairs: (x1, x2) where x1 = x[..., ::2], x2 = x[..., 1::2]
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]

        # Apply rotation: [x1*cos - x2*sin, x1*sin + x2*cos]
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos

        # Interleave back
        out = torch.stack([out1, out2], dim=-1).flatten(-2)
        return out
