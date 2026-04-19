"""Tests for models/block.py"""
import torch
from models.block import TransformerBlock


def test_output_shape():
    B, T, C = 2, 16, 64
    block = TransformerBlock(d_model=C, n_heads=4, block_size=32)
    x = torch.randn(B, T, C)
    out = block(x)
    assert out.shape == (B, T, C), f"Expected {(B, T, C)}, got {out.shape}"


def test_residual_connections():
    C = 64
    block = TransformerBlock(d_model=C, n_heads=4, dropout=0.0, block_size=32)
    x = torch.randn(1, 4, C)
    out = block(x)
    # Output should not be all zeros (residual connection adds input)
    assert not torch.allclose(out, torch.zeros_like(out))
    # Output should differ from input (the attention/ffn should modify it)
    assert not torch.equal(out, x)


def test_moe_block():
    B, T, C = 2, 8, 64
    block = TransformerBlock(
        d_model=C, n_heads=4, block_size=32,
        use_moe=True, n_experts=4, top_k_experts=2,
    )
    x = torch.randn(B, T, C)
    out = block(x)
    assert out.shape == (B, T, C)
