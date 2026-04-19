"""
Smoke test: verify Python environment, PyTorch installation, and GPU availability.
"""
import torch
import sys


def test_python_version():
    """Python 3.10+ is required."""
    assert sys.version_info >= (3, 10), (
        f"Python 3.10+ required, got {sys.version_info.major}.{sys.version_info.minor}"
    )


def test_pytorch_installed():
    """PyTorch must be importable and functional."""
    x = torch.randn(2, 3)
    assert x.shape == (2, 3)


def test_device_available():
    """Report the available compute device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        name = torch.cuda.get_device_name(0)
        print(f"CUDA device: {name}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("MPS device available (Apple Silicon)")
    else:
        device = torch.device('cpu')
        print("No GPU detected — using CPU")

    # Simple tensor-on-device smoke test
    t = torch.tensor([1.0, 2.0, 3.0], device=device)
    assert t.device.type == device.type
    assert t.sum().item() == 6.0
