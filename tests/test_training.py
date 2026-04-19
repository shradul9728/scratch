"""Tests for utils/training.py — optimizer, LR schedule, checkpointing."""
import os
import torch
import torch.nn as nn
import tempfile

from utils.training import configure_optimizer, get_lr, save_checkpoint, load_checkpoint


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
        self.ln = nn.LayerNorm(10)

    def forward(self, x):
        return self.ln(self.linear(x))


class TestConfigureOptimizer:
    def test_returns_optimizer(self):
        model = SimpleModel()
        opt = configure_optimizer(model, lr=1e-3)
        assert isinstance(opt, torch.optim.AdamW)

    def test_two_param_groups(self):
        model = SimpleModel()
        opt = configure_optimizer(model, lr=1e-3, weight_decay=0.1)
        assert len(opt.param_groups) == 2
        # Decay group and no-decay group
        assert opt.param_groups[0]['weight_decay'] == 0.1
        assert opt.param_groups[1]['weight_decay'] == 0.0


class TestGetLR:
    def test_warmup_increases(self):
        lr1 = get_lr(0, warmup_steps=100, max_steps=1000, max_lr=1e-3, min_lr=1e-5)
        lr2 = get_lr(50, warmup_steps=100, max_steps=1000, max_lr=1e-3, min_lr=1e-5)
        lr3 = get_lr(99, warmup_steps=100, max_steps=1000, max_lr=1e-3, min_lr=1e-5)
        assert lr1 < lr2 < lr3

    def test_warmup_reaches_max(self):
        lr = get_lr(100, warmup_steps=100, max_steps=1000, max_lr=1e-3, min_lr=1e-5)
        # At step=warmup_steps, cosine starts; progress=0 → cos(0)=1 → lr=max_lr
        assert abs(lr - 1e-3) < 1e-8

    def test_cosine_decay(self):
        lr_start = get_lr(100, warmup_steps=100, max_steps=1000, max_lr=1e-3, min_lr=1e-5)
        lr_mid = get_lr(550, warmup_steps=100, max_steps=1000, max_lr=1e-3, min_lr=1e-5)
        lr_end = get_lr(999, warmup_steps=100, max_steps=1000, max_lr=1e-3, min_lr=1e-5)
        assert lr_start > lr_mid > lr_end

    def test_min_lr_at_end(self):
        lr = get_lr(1000, warmup_steps=100, max_steps=1000, max_lr=1e-3, min_lr=1e-5)
        assert lr == 1e-5


class TestCheckpointing:
    def test_save_and_load(self, tmp_path):
        model = SimpleModel()
        opt = configure_optimizer(model, lr=1e-3)

        # Modify model state
        with torch.no_grad():
            model.linear.weight.fill_(42.0)

        path = str(tmp_path / "test_ckpt.pt")
        save_checkpoint(model, opt, step=100, loss=0.5, path=path)
        assert os.path.isfile(path)

        # Create fresh model and load
        model2 = SimpleModel()
        step = load_checkpoint(path, model2)
        assert step == 100
        assert torch.allclose(model2.linear.weight, torch.full_like(model2.linear.weight, 42.0))
