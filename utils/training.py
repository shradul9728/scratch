"""
Training utilities: optimizer configuration, learning rate scheduling, checkpointing.
"""
import os
import math
import torch
import torch.nn as nn
from typing import Optional, Tuple


def configure_optimizer(
    model: nn.Module,
    lr: float = 3e-4,
    weight_decay: float = 0.1,
    betas: Tuple[float, float] = (0.9, 0.95),
) -> torch.optim.Optimizer:
    """
    Create an AdamW optimizer with weight decay applied only to 2D+ parameters
    (i.e., weight matrices), not biases or LayerNorm parameters.

    Args:
        model: The model to optimize.
        lr: Learning rate.
        weight_decay: Weight decay coefficient.
        betas: Adam beta hyperparameters.

    Returns:
        Configured AdamW optimizer.
    """
    # Separate parameters into decay and no-decay groups
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Biases, LayerNorm weights, and embedding weights don't get weight decay
        if param.dim() < 2 or 'ln' in name or 'bias' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ]

    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=betas)
    return optimizer


def get_lr(
    step: int,
    warmup_steps: int,
    max_steps: int,
    max_lr: float,
    min_lr: float,
) -> float:
    """
    Cosine annealing learning rate with linear warmup.

    Args:
        step: Current training step.
        warmup_steps: Number of warmup steps (linear increase).
        max_steps: Total number of training steps.
        max_lr: Peak learning rate (reached at end of warmup).
        min_lr: Minimum learning rate (floor of cosine decay).

    Returns:
        Learning rate for the current step.
    """
    # Linear warmup
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps

    # After max_steps, return min_lr
    if step >= max_steps:
        return min_lr

    # Cosine decay from max_lr to min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (max_lr - min_lr) * cosine_decay


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    loss: float,
    path: str,
) -> None:
    """
    Save a training checkpoint.

    Args:
        model: The model to save.
        optimizer: The optimizer to save.
        step: Current training step.
        loss: Current loss value.
        path: File path to save the checkpoint.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
        'loss': loss,
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> int:
    """
    Load a training checkpoint.

    Args:
        path: Path to the checkpoint file.
        model: The model to load weights into.
        optimizer: Optional optimizer to restore state.

    Returns:
        The training step at which the checkpoint was saved.
    """
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint.get('step', 0)
