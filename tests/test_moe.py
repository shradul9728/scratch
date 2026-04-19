"""Tests for models/moe.py"""
import torch
from models.moe import MoELayer


def test_output_shape():
    B, T, C = 2, 8, 64
    moe = MoELayer(d_model=C, n_experts=4, top_k=2)
    x = torch.randn(B, T, C)
    out = moe(x)
    assert out.shape == (B, T, C)


def test_load_balancing_loss_nonzero():
    B, T, C = 4, 16, 64
    moe = MoELayer(d_model=C, n_experts=4, top_k=2)
    x = torch.randn(B, T, C)
    _ = moe(x)
    assert moe.aux_loss.item() > 0, "Auxiliary load-balancing loss should be non-zero"


def test_different_tokens_different_experts():
    """With enough tokens, different tokens should route to different experts."""
    C = 64
    moe = MoELayer(d_model=C, n_experts=8, top_k=2)

    # Large batch of diverse tokens
    x = torch.randn(1, 128, C) * 10  # Scale up for diversity
    _ = moe(x)

    # Check that the gating assigns to more than just 1-2 experts
    router_logits = moe.gate(x.view(-1, C))
    _, topk_indices = torch.topk(router_logits, 2, dim=-1)
    unique_experts = topk_indices.unique()
    assert len(unique_experts) > 2, "Tokens should route to more than 2 experts"
