"""
Main training loop for GPT From Scratch.

Usage:
    python train.py                     # Use default config.yaml
    python train.py --config my.yaml    # Use custom config
"""
import os
import sys
import argparse
import time

import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.config import load_config
from utils.data_loader import load_text
from utils.tokenizer import CharTokenizer, BPETokenizer
from utils.dataset import train_val_split, get_dataloader
from utils.training import configure_optimizer, get_lr, save_checkpoint
from models.gpt import GPT


def parse_args():
    parser = argparse.ArgumentParser(description="Train GPT From Scratch")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config YAML')
    return parser.parse_args()


@torch.no_grad()
def estimate_loss(model, train_dl, val_dl, eval_iters, device, use_amp=False):
    """Estimate average loss on train and val sets."""
    model.eval()
    losses = {}
    for split_name, dl in [('train', train_dl), ('val', val_dl)]:
        total_loss = 0.0
        count = 0
        dl_iter = iter(dl)
        for _ in range(eval_iters):
            try:
                x, y = next(dl_iter)
            except StopIteration:
                dl_iter = iter(dl)
                x, y = next(dl_iter)
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast('cuda', enabled=use_amp):
                _, loss = model(x, y)
            total_loss += loss.item()
            count += 1
        losses[split_name] = total_loss / max(count, 1)
    model.train()
    return losses


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # ── Device ──
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # ── Data ──
    print("Loading data...")
    text = load_text(cfg.data.dataset_path)
    print(f"Dataset size: {len(text):,} characters")

    # ── Tokenizer ──
    if cfg.data.tokenizer == 'bpe':
        tokenizer = BPETokenizer()
    else:
        tokenizer = CharTokenizer(text)
    
    token_ids = tokenizer.encode(text)
    vocab_size = tokenizer.vocab_size
    print(f"Tokenizer: {cfg.data.tokenizer} | Vocab size: {vocab_size:,} | Total tokens: {len(token_ids):,}")

    # ── Dataset ──
    train_ids, val_ids = train_val_split(token_ids)
    block_size = cfg.data.block_size
    batch_size = cfg.training.batch_size

    train_dl = get_dataloader(train_ids, block_size, batch_size, shuffle=True)
    val_dl = get_dataloader(val_ids, block_size, batch_size, shuffle=False)

    # ── Model ──
    model = GPT(
        vocab_size=vocab_size,
        block_size=block_size,
        d_model=cfg.model.d_model,
        n_heads=cfg.model.n_heads,
        n_layers=cfg.model.n_layers,
        dropout=cfg.model.dropout,
        bias=cfg.model.bias,
        use_gqa=cfg.model.get('use_gqa', False),
        n_kv_heads=cfg.model.get('n_kv_heads', None),
        use_moe=cfg.model.get('use_moe', False),
        n_experts=cfg.model.get('n_experts', 8),
        top_k_experts=cfg.model.get('top_k_experts', 2),
    ).to(device)
    print(f"Model parameters: {model.count_parameters():,}")

    # ── Optimizer ──
    optimizer = configure_optimizer(
        model,
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        betas=(cfg.training.beta1, cfg.training.beta2),
    )

    # ── Mixed Precision ──
    use_amp = cfg.training.get('use_amp', False) and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    if use_amp:
        print("Mixed precision (FP16) training ENABLED")

    # ── Gradient Accumulation ──
    grad_accum_steps = cfg.training.get('grad_accum_steps', 1)
    if grad_accum_steps > 1:
        print(f"Gradient accumulation: {grad_accum_steps} steps (effective batch = {batch_size * grad_accum_steps})")

    # ── WandB (optional) ──
    use_wandb = cfg.training.get('log_wandb', False)
    if use_wandb:
        try:
            import wandb
            wandb.init(project=cfg.training.wandb_project, config=cfg.to_dict())
        except ImportError:
            print("wandb not installed, disabling logging.")
            use_wandb = False

    # ── Training Loop ──
    max_iters = cfg.training.max_iters
    warmup_iters = cfg.training.warmup_iters
    eval_interval = cfg.training.eval_interval
    eval_iters = cfg.training.eval_iters
    save_interval = cfg.training.save_interval
    checkpoint_dir = cfg.training.checkpoint_dir
    grad_clip = cfg.training.grad_clip

    train_losses = []
    val_losses = []
    steps_log = []

    train_iter = iter(train_dl)
    model.train()
    start_time = time.time()

    pbar = tqdm(range(max_iters), desc="Training", ncols=100)
    for step in pbar:
        # Learning rate schedule
        lr = get_lr(step, warmup_iters, max_iters, cfg.training.learning_rate, cfg.training.min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Gradient accumulation loop
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for micro_step in range(grad_accum_steps):
            # Get batch
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dl)
                x, y = next(train_iter)
            x, y = x.to(device), y.to(device)

            # Forward with AMP
            with torch.amp.autocast('cuda', enabled=use_amp):
                logits, loss = model(x, y)
                loss = loss / grad_accum_steps  # scale loss for accumulation

            # Backward with scaler
            scaler.scale(loss).backward()
            accum_loss += loss.item()

        # Gradient clipping and optimizer step
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        # Progress bar
        pbar.set_postfix({'loss': f'{accum_loss:.4f}', 'lr': f'{lr:.2e}'})

        # Evaluation
        if step % eval_interval == 0 or step == max_iters - 1:
            losses = estimate_loss(model, train_dl, val_dl, eval_iters, device, use_amp=use_amp)
            train_losses.append(losses['train'])
            val_losses.append(losses['val'])
            steps_log.append(step)

            elapsed = time.time() - start_time
            print(f"\nStep {step:>5d} | train loss: {losses['train']:.4f} | val loss: {losses['val']:.4f} | lr: {lr:.2e} | time: {elapsed:.1f}s")

            if use_wandb:
                import wandb
                wandb.log({
                    'train_loss': losses['train'],
                    'val_loss': losses['val'],
                    'lr': lr,
                    'step': step,
                })

        # Checkpoint
        if step > 0 and step % save_interval == 0:
            ckpt_path = os.path.join(checkpoint_dir, f'checkpoint_step{step}.pt')
            save_checkpoint(model, optimizer, step, accum_loss, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    # ── Final checkpoint ──
    final_path = os.path.join(checkpoint_dir, 'checkpoint_final.pt')
    save_checkpoint(model, optimizer, max_iters, accum_loss, final_path)
    print(f"Saved final checkpoint: {final_path}")

    # ── Loss curve plot ──
    os.makedirs('results', exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(steps_log, train_losses, label='Train Loss', linewidth=2)
    plt.plot(steps_log, val_losses, label='Val Loss', linewidth=2)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/loss_curve.png', dpi=150)
    print("Loss curve saved to results/loss_curve.png")

    if use_wandb:
        import wandb
        wandb.finish()

    print("Training complete!")


if __name__ == '__main__':
    main()
