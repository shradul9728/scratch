"""
Transformer Block: the fundamental repeating unit of the GPT model.
Pre-Norm architecture: LayerNorm → Attention → Residual → LayerNorm → FFN → Residual
"""
import torch
import torch.nn as nn

from models.attention import MultiHeadAttention, GroupedQueryAttention
from models.ffn import FeedForward
from models.moe import MoELayer


class TransformerBlock(nn.Module):
    """
    Single Transformer decoder block with Pre-Norm architecture.
    Supports standard MHA or GQA, and dense FFN or MoE.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        block_size: int = 1024,
        bias: bool = False,
        use_gqa: bool = False,
        n_kv_heads: int = None,
        use_moe: bool = False,
        n_experts: int = 8,
        top_k_experts: int = 2,
    ):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model, elementwise_affine=True)
        self.ln2 = nn.LayerNorm(d_model, elementwise_affine=True)

        # Attention: MHA or GQA
        if use_gqa and n_kv_heads is not None:
            self.attn = GroupedQueryAttention(
                d_model=d_model, n_heads=n_heads, n_kv_heads=n_kv_heads,
                dropout=dropout, block_size=block_size, bias=bias,
            )
        else:
            self.attn = MultiHeadAttention(
                d_model=d_model, n_heads=n_heads,
                dropout=dropout, block_size=block_size, bias=bias,
            )

        # Feed-Forward: dense or MoE
        if use_moe:
            self.ffn = MoELayer(
                d_model=d_model, n_experts=n_experts, top_k=top_k_experts,
                bias=bias, dropout=dropout,
            )
        else:
            self.ffn = FeedForward(d_model=d_model, dropout=dropout, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            (batch, seq_len, d_model)
        """
        # Pre-Norm + Attention + Residual
        x = x + self.attn(self.ln1(x))
        # Pre-Norm + FFN + Residual
        x = x + self.ffn(self.ln2(x))
        return x
