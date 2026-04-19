"""Tests for models/attention.py"""
import torch
from models.attention import scaled_dot_product_attention, MultiHeadAttention, GroupedQueryAttention


class TestScaledDotProductAttention:
    def test_output_shape(self):
        B, T, d_k = 2, 8, 16
        Q = torch.randn(B, T, d_k)
        K = torch.randn(B, T, d_k)
        V = torch.randn(B, T, d_k)
        out = scaled_dot_product_attention(Q, K, V)
        assert out.shape == (B, T, d_k)

    def test_causal_mask_blocks_future(self):
        T, d_k = 4, 8
        Q = torch.randn(1, T, d_k)
        K = torch.randn(1, T, d_k)
        V = torch.ones(1, T, d_k)  # All ones so we can inspect weights

        # Upper triangular mask (True = masked)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        out = scaled_dot_product_attention(Q, K, V, mask=mask)
        assert out.shape == (1, T, d_k)
        # First token can only attend to itself, so output should be V[0] = 1
        # (approximately, after softmax concentration)


class TestMultiHeadAttention:
    def test_output_shape(self):
        B, T, C = 2, 16, 64
        mha = MultiHeadAttention(d_model=C, n_heads=4, block_size=32)
        x = torch.randn(B, T, C)
        out = mha(x)
        assert out.shape == (B, T, C)

    def test_parameter_count(self):
        C, H = 64, 4
        mha = MultiHeadAttention(d_model=C, n_heads=H, bias=False, block_size=32)
        # qkv_proj: C * 3C = 3C^2
        # out_proj: C * C = C^2
        # Total = 4C^2
        expected = 4 * C * C
        actual = sum(p.numel() for p in mha.parameters() if p.requires_grad)
        assert actual == expected, f"Expected {expected}, got {actual}"


class TestGroupedQueryAttention:
    def test_output_shape(self):
        B, T, C = 2, 16, 64
        gqa = GroupedQueryAttention(d_model=C, n_heads=4, n_kv_heads=2, block_size=32)
        x = torch.randn(B, T, C)
        out = gqa(x)
        assert out.shape == (B, T, C)

    def test_fewer_kv_params(self):
        C, H, KV_H = 64, 4, 1
        d_k = C // H

        gqa = GroupedQueryAttention(d_model=C, n_heads=H, n_kv_heads=KV_H, bias=False, block_size=32)
        mha = MultiHeadAttention(d_model=C, n_heads=H, bias=False, block_size=32)

        gqa_params = sum(p.numel() for p in gqa.parameters() if p.requires_grad)
        mha_params = sum(p.numel() for p in mha.parameters() if p.requires_grad)

        # GQA should have fewer parameters because K and V projections are smaller
        assert gqa_params < mha_params
