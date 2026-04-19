"""Tests for models/embeddings.py"""
import torch
from models.embeddings import TokenEmbedding, PositionalEmbedding


def test_token_embedding_shape():
    emb = TokenEmbedding(vocab_size=100, d_model=64)
    x = torch.randint(0, 100, (2, 16))
    out = emb(x)
    assert out.shape == (2, 16, 64)


def test_positional_embedding_shape():
    emb = PositionalEmbedding(block_size=128, d_model=64)
    out = emb(seq_len=32)
    assert out.shape == (32, 64)


def test_positional_embedding_on_device():
    emb = PositionalEmbedding(block_size=128, d_model=64)
    out = emb(seq_len=16, device=torch.device('cpu'))
    assert out.device.type == 'cpu'
