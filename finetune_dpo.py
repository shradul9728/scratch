"""
Direct Preference Optimization (DPO) Fine-tuning for GPT.

Trains the model to prefer chosen responses over rejected ones using the DPO loss,
which avoids the need for a separate reward model.

Usage:
    python finetune_dpo.py --checkpoint checkpoints/sft/sft_final.pt --dpo_data data/dpo_data.jsonl
"""
import os
import json
import copy
import argparse

import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.config import load_config
from utils.data_loader import load_text
from utils.tokenizer import CharTokenizer, BPETokenizer
from utils.training import configure_optimizer, save_checkpoint, load_checkpoint
from models.gpt import GPT


def parse_args():
    parser = argparse.ArgumentParser(description="DPO Fine-tuning")
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--checkpoint', type=str, required=True, help='SFT checkpoint')
    parser.add_argument('--dpo_data', type=str, required=True,
                        help='Path to DPO data (JSONL: {prompt, chosen, rejected})')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--beta', type=float, default=0.1, help='DPO temperature parameter')
    parser.add_argument('--output_dir', type=str, default='checkpoints/dpo')
    return parser.parse_args()


def load_dpo_data(path: str):
    """Load DPO preference data. Each line: {"prompt": "...", "chosen": "...", "rejected": "..."}"""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def compute_log_probs(model, input_ids: torch.Tensor, response_start: int) -> torch.Tensor:
    """
    Compute the sum of log probabilities for the response tokens.

    Args:
        model: The GPT model.
        input_ids: (1, seq_len) full sequence (prompt + response).
        response_start: Index where the response begins.

    Returns:
        Scalar: sum of log probs over response tokens.
    """
    logits, _ = model(input_ids[:, :-1])
    # Log probabilities
    log_probs = F.log_softmax(logits, dim=-1)
    # Gather log probs for actual next tokens
    targets = input_ids[:, 1:]
    token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    # Sum only over response tokens
    response_log_probs = token_log_probs[:, max(response_start - 1, 0):]
    return response_log_probs.sum()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    # Tokenizer
    if cfg.data.tokenizer == 'bpe':
        tokenizer = BPETokenizer()
    else:
        text = load_text(cfg.data.dataset_path)
        tokenizer = CharTokenizer(text)

    vocab_size = tokenizer.vocab_size

    # Policy model (to be optimized)
    policy_model = GPT(
        vocab_size=vocab_size,
        block_size=cfg.data.block_size,
        d_model=cfg.model.d_model,
        n_heads=cfg.model.n_heads,
        n_layers=cfg.model.n_layers,
        dropout=0.0,
        bias=cfg.model.bias,
    ).to(device)

    load_checkpoint(args.checkpoint, policy_model)
    print(f"Loaded SFT checkpoint: {args.checkpoint}")

    # Reference model (frozen copy of the SFT model)
    ref_model = copy.deepcopy(policy_model)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    print("Created frozen reference model")

    # DPO data
    dpo_data = load_dpo_data(args.dpo_data)
    print(f"Loaded {len(dpo_data)} DPO preference pairs")

    # Optimizer
    optimizer = configure_optimizer(policy_model, lr=args.lr, weight_decay=0.0)

    # Training
    beta = args.beta
    policy_model.train()

    for epoch in range(args.epochs):
        total_loss = 0.0
        pbar = tqdm(dpo_data, desc=f"DPO Epoch {epoch+1}/{args.epochs}")
        for item in pbar:
            prompt_ids = tokenizer.encode(item['prompt'])
            chosen_ids = tokenizer.encode(item['chosen'])
            rejected_ids = tokenizer.encode(item['rejected'])

            # Build full sequences
            chosen_full = prompt_ids + chosen_ids
            rejected_full = prompt_ids + rejected_ids

            # Truncate to block_size
            block_size = cfg.data.block_size
            chosen_full = chosen_full[:block_size]
            rejected_full = rejected_full[:block_size]

            if len(chosen_full) < 2 or len(rejected_full) < 2:
                continue

            resp_start = len(prompt_ids)

            chosen_tensor = torch.tensor([chosen_full], dtype=torch.long, device=device)
            rejected_tensor = torch.tensor([rejected_full], dtype=torch.long, device=device)

            # Policy log probs
            policy_chosen_lp = compute_log_probs(policy_model, chosen_tensor, resp_start)
            policy_rejected_lp = compute_log_probs(policy_model, rejected_tensor, resp_start)

            # Reference log probs (no grad)
            with torch.no_grad():
                ref_chosen_lp = compute_log_probs(ref_model, chosen_tensor, resp_start)
                ref_rejected_lp = compute_log_probs(ref_model, rejected_tensor, resp_start)

            # DPO loss: -log(sigmoid(beta * (delta_chosen - delta_rejected)))
            # where delta = policy_lp - ref_lp
            chosen_reward = policy_chosen_lp - ref_chosen_lp
            rejected_reward = policy_rejected_lp - ref_rejected_lp
            loss = -F.logsigmoid(beta * (chosen_reward - rejected_reward))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / max(len(dpo_data), 1)
        print(f"Epoch {epoch+1} average DPO loss: {avg_loss:.4f}")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, 'dpo_final.pt')
    save_checkpoint(policy_model, optimizer, 0, avg_loss, save_path)
    print(f"DPO model saved to {save_path}")


if __name__ == '__main__':
    main()
