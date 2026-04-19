"""
Position-wise Feed-Forward Network for the Transformer.
"""
import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """
    Two-layer feed-forward network with GELU activation and dropout.
    FFN(x) = Dropout(Linear(GELU(Linear(x))))
    Inner dimension is 4 * d_model (standard Transformer scaling).
    """

    def __init__(self, d_model: int, dropout: float = 0.0, bias: bool = False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model, bias=bias),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model, bias=bias),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            (batch, seq_len, d_model)
        """
        return self.net(x)
