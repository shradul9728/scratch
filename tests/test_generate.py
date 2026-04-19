"""Tests for text generation (models/gpt.py generate method)."""
import torch
from models.gpt import GPT


def make_model():
    return GPT(
        vocab_size=128,
        block_size=32,
        d_model=64,
        n_heads=4,
        n_layers=2,
        dropout=0.0,
    )


def test_generate_valid_tokens():
    model = make_model()
    idx = torch.randint(0, 128, (1, 5))
    out = model.generate(idx, max_new_tokens=20, temperature=1.0)
    # All tokens should be in valid range
    assert out.min() >= 0
    assert out.max() < 128
    assert out.shape == (1, 25)


def test_generate_with_top_k():
    model = make_model()
    idx = torch.randint(0, 128, (1, 5))
    out = model.generate(idx, max_new_tokens=10, temperature=0.8, top_k=10)
    assert out.shape == (1, 15)
    assert out.min() >= 0
    assert out.max() < 128


def test_temperature_zero_is_greedy():
    model = make_model()
    idx = torch.randint(0, 128, (1, 3))
    out1 = model.generate(idx.clone(), max_new_tokens=10, temperature=1e-10)
    out2 = model.generate(idx.clone(), max_new_tokens=10, temperature=1e-10)
    assert torch.equal(out1, out2)
