"""Tests for models/rope.py"""
import torch
from models.rope import RotaryPositionalEmbedding


def test_output_shape():
    d_k = 16
    rope = RotaryPositionalEmbedding(d_k=d_k, max_seq_len=128)
    x = torch.randn(2, 4, 32, d_k)  # (B, n_heads, T, d_k)
    out = rope(x)
    assert out.shape == x.shape


def test_rotation_applied():
    d_k = 16
    rope = RotaryPositionalEmbedding(d_k=d_k, max_seq_len=128)
    x = torch.randn(1, 1, 8, d_k)
    out = rope(x)
    # Output should differ from input (rotation was applied)
    assert not torch.equal(out, x)


def test_different_positions_give_different_rotations():
    d_k = 16
    rope = RotaryPositionalEmbedding(d_k=d_k, max_seq_len=128)
    # Same vector at two different positions should get different rotations
    x = torch.ones(1, 1, 4, d_k)
    out = rope(x)
    # Position 0 and position 1 should have different outputs
    assert not torch.allclose(out[0, 0, 0], out[0, 0, 1])
