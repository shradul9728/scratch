"""Tests for models/ffn.py"""
import torch
from models.ffn import FeedForward


def test_output_shape():
    B, T, C = 2, 16, 64
    ffn = FeedForward(d_model=C)
    x = torch.randn(B, T, C)
    out = ffn(x)
    assert out.shape == (B, T, C)


def test_no_in_place_modification():
    C = 64
    ffn = FeedForward(d_model=C)
    x = torch.randn(1, 4, C)
    x_clone = x.clone()
    _ = ffn(x)
    # Input should not be modified in-place
    assert torch.equal(x, x_clone)
