"""
Attention mechanisms for the GPT model.
  - Scaled Dot-Product Attention
  - Multi-Head Attention (MHA)
  - Grouped Query Attention (GQA)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor = None,
    dropout: nn.Dropout = None,
) -> torch.Tensor:
    """
    Compute Scaled Dot-Product Attention.

    Args:
        Q: Query tensor (..., seq_len, d_k)
        K: Key tensor (..., seq_len, d_k)
        V: Value tensor (..., seq_len, d_v)
        mask: Optional boolean mask where True means IGNORE (set to -inf).
        dropout: Optional dropout module applied to attention weights.

    Returns:
        Attention output (..., seq_len, d_v)
    """
    d_k = Q.size(-1)
    # (... , seq_len_q, seq_len_k)
    attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        attn_scores = attn_scores.masked_fill(mask, float('-inf'))

    attn_weights = F.softmax(attn_scores, dim=-1)

    if dropout is not None:
        attn_weights = dropout(attn_weights)

    return torch.matmul(attn_weights, V)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention with causal masking.
    Splits d_model into n_heads parallel attention heads.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0,
                 block_size: int = 1024, bias: bool = False):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Combined projection for Q, K, V
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask — upper triangular = True (positions to mask out)
        mask = torch.triu(torch.ones(block_size, block_size, dtype=torch.bool), diagonal=1)
        self.register_buffer('mask', mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            (batch, seq_len, d_model)
        """
        B, T, C = x.shape

        # Project and split into Q, K, V
        qkv = self.qkv_proj(x)  # (B, T, 3*C)
        Q, K, V = qkv.split(C, dim=-1)

        # Reshape to (B, n_heads, T, d_k)
        Q = Q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # Causal mask for current sequence length
        causal_mask = self.mask[:T, :T]  # (T, T)

        # Attention
        out = scaled_dot_product_attention(Q, K, V, mask=causal_mask, dropout=self.attn_dropout)

        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.out_proj(out))


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA).
    Uses fewer K/V heads than Q heads to reduce KV-cache memory.
    Each KV head is shared across a group of Q heads.
    """

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int,
                 dropout: float = 0.0, block_size: int = 1024, bias: bool = False):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_groups = n_heads // n_kv_heads
        self.d_k = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, self.n_kv_heads * self.d_k, bias=bias)
        self.v_proj = nn.Linear(d_model, self.n_kv_heads * self.d_k, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        mask = torch.triu(torch.ones(block_size, block_size, dtype=torch.bool), diagonal=1)
        self.register_buffer('mask', mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            (batch, seq_len, d_model)
        """
        B, T, C = x.shape

        Q = self.q_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.n_kv_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.n_kv_heads, self.d_k).transpose(1, 2)

        # Expand K/V to match Q heads by repeating across groups
        # (B, n_kv_heads, T, d_k) -> (B, n_heads, T, d_k)
        K = K.repeat_interleave(self.n_groups, dim=1)
        V = V.repeat_interleave(self.n_groups, dim=1)

        causal_mask = self.mask[:T, :T]
        out = scaled_dot_product_attention(Q, K, V, mask=causal_mask, dropout=self.attn_dropout)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.out_proj(out))
