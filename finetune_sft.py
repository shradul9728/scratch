"""
Supervised Fine-Tuning (SFT) for GPT.

Fine-tunes a pre-trained GPT checkpoint on instruction-response pairs.
Loss is computed only on the response portion.

Usage:
    python finetune_sft.py --checkpoint checkpoints/checkpoint_final.pt --sft_data data/sft_data.jsonl
"""
import os
import json
import argparse

import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.config import load_config
from utils.data_loader import load_text
from utils.tokenizer import CharTokenizer, BPETokenizer
from utils.training import configure_optimizer, get_lr, save_checkpoint, load_checkpoint
from models.gpt import GPT


def parse_args():
    parser = argparse.ArgumentParser(description="SFT Fine-tuning")
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--checkpoint', type=str, required=True, help='Pre-trained checkpoint')
    parser.add_argument('--sft_data', type=str, required=True, help='Path to SFT data (JSONL: {instruction, response})')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--output_dir', type=str, default='checkpoints/sft')
    return parser.parse_args()


def load_sft_data(path: str):
    """Load SFT data from a JSONL file. Each line: {"instruction": "...", "response": "..."}"""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


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

    # Model
    model = GPT(
        vocab_size=vocab_size,
        block_size=cfg.data.block_size,
        d_model=cfg.model.d_model,
        n_heads=cfg.model.n_heads,
        n_layers=cfg.model.n_layers,
        dropout=0.1,
        bias=cfg.model.bias,
    ).to(device)

    load_checkpoint(args.checkpoint, model)
    print(f"Loaded pre-trained checkpoint: {args.checkpoint}")
    print(f"Model parameters: {model.count_parameters():,}")

    # SFT Data
    sft_data = load_sft_data(args.sft_data)
    print(f"Loaded {len(sft_data)} SFT examples")

    # Prepare tokenized pairs
    examples = []
    for item in sft_data:
        prompt = item['instruction']
        response = item['response']

        prompt_ids = tokenizer.encode(prompt)
        response_ids = tokenizer.encode(response)

        # Concatenate: [prompt | response]
        full_ids = prompt_ids + response_ids
        if len(full_ids) > cfg.data.block_size:
            full_ids = full_ids[:cfg.data.block_size]
            response_start = min(len(prompt_ids), cfg.data.block_size)
        else:
            response_start = len(prompt_ids)

        examples.append({
            'input_ids': full_ids,
            'response_start': response_start,
        })

    # Optimizer
    optimizer = configure_optimizer(model, lr=args.lr, weight_decay=0.01)

    # Training
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        pbar = tqdm(examples, desc=f"Epoch {epoch+1}/{args.epochs}")
        for ex in pbar:
            ids = ex['input_ids']
            resp_start = ex['response_start']

            if len(ids) < 2:
                continue

            x = torch.tensor(ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
            y = torch.tensor(ids[1:], dtype=torch.long, device=device).unsqueeze(0)

            logits, _ = model(x)

            # Mask loss to only compute on response tokens
            # Create loss mask: 0 for prompt tokens, 1 for response tokens
            loss_mask = torch.zeros_like(y, dtype=torch.float)
            loss_mask[:, max(resp_start - 1, 0):] = 1.0

            # Compute per-token loss
            per_token_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
                reduction='none',
            ).view_as(y)

            # Apply mask and average
            masked_loss = (per_token_loss * loss_mask).sum() / loss_mask.sum().clamp(min=1)

            optimizer.zero_grad(set_to_none=True)
            masked_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += masked_loss.item()
            pbar.set_postfix({'loss': f'{masked_loss.item():.4f}'})

        avg_loss = total_loss / max(len(examples), 1)
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, 'sft_final.pt')
    save_checkpoint(model, optimizer, 0, avg_loss, save_path)
    print(f"SFT model saved to {save_path}")


if __name__ == '__main__':
    main()
