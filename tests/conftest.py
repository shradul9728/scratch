"""
Pytest configuration and shared fixtures for the GPT project test suite.
"""
import sys
import os
import pytest
import torch

# Ensure the project root is on the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@pytest.fixture
def device():
    """Return the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


@pytest.fixture
def small_config():
    """A small model config dict for fast testing."""
    return {
        'vocab_size': 256,
        'block_size': 32,
        'd_model': 64,
        'n_heads': 4,
        'n_layers': 2,
        'dropout': 0.0,
        'bias': False,
    }
