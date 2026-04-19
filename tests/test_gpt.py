"""Tests for models/gpt.py"""
import torch
from models.gpt import GPT


class TestGPTModel:
    def setup_method(self):
        self.model = GPT(
            vocab_size=256,
            block_size=32,
            d_model=64,
            n_heads=4,
            n_layers=2,
            dropout=0.0,
            bias=False,
        )

    def test_logits_shape(self):
        B, T = 2, 16
        idx = torch.randint(0, 256, (B, T))
        logits, loss = self.model(idx)
        assert logits.shape == (B, T, 256)
        assert loss is None

    def test_loss_is_scalar_with_targets(self):
        B, T = 2, 16
        idx = torch.randint(0, 256, (B, T))
        targets = torch.randint(0, 256, (B, T))
        logits, loss = self.model(idx, targets)
        assert logits.shape == (B, T, 256)
        assert loss is not None
        assert loss.dim() == 0  # scalar

    def test_count_parameters(self):
        n = self.model.count_parameters()
        assert n > 0
        assert isinstance(n, int)

    def test_generate_output_length(self):
        idx = torch.randint(0, 256, (1, 5))
        out = self.model.generate(idx, max_new_tokens=10, temperature=1.0)
        assert out.shape == (1, 15)  # 5 prompt + 10 generated

    def test_greedy_deterministic(self):
        idx = torch.randint(0, 256, (1, 5))
        torch.manual_seed(42)
        out1 = self.model.generate(idx.clone(), max_new_tokens=10, temperature=1e-10)
        torch.manual_seed(42)
        out2 = self.model.generate(idx.clone(), max_new_tokens=10, temperature=1e-10)
        assert torch.equal(out1, out2), "Greedy decoding should be deterministic"
