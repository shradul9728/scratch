"""
Inference CLI for GPT From Scratch.

Usage:
    python generate.py --checkpoint checkpoints/checkpoint_final.pt --prompt "To be or not to be"
    python generate.py --checkpoint checkpoints/checkpoint_final.pt --prompt "O Romeo" --temperature 0.5 --top_k 50
"""
import argparse
import torch

from utils.config import load_config
from utils.data_loader import load_text
from utils.tokenizer import CharTokenizer, BPETokenizer
from models.gpt import GPT
from utils.training import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text with a trained GPT model")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config YAML')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--prompt', type=str, default='\n', help='Text prompt to start generation')
    parser.add_argument('--max_tokens', type=int, default=None, help='Max new tokens to generate')
    parser.add_argument('--temperature', type=float, default=None, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=None, help='Top-k filtering')
    return parser.parse_args()


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
    print(f"Using device: {device}")

    # Tokenizer — need to rebuild to get vocab_size
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
        dropout=0.0,  # No dropout at inference
        bias=cfg.model.bias,
        use_gqa=cfg.model.get('use_gqa', False),
        n_kv_heads=cfg.model.get('n_kv_heads', None),
        use_moe=cfg.model.get('use_moe', False),
        n_experts=cfg.model.get('n_experts', 8),
        top_k_experts=cfg.model.get('top_k_experts', 2),
    ).to(device)

    # Load checkpoint
    load_checkpoint(args.checkpoint, model)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Model parameters: {model.count_parameters():,}")

    # Generation settings (CLI overrides config)
    max_tokens = args.max_tokens or cfg.generation.max_new_tokens
    temperature = args.temperature if args.temperature is not None else cfg.generation.temperature
    top_k = args.top_k if args.top_k is not None else cfg.generation.top_k

    # Encode prompt
    prompt_ids = tokenizer.encode(args.prompt)
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    print(f"\n{'='*60}")
    print(f"Prompt: {args.prompt}")
    print(f"Temperature: {temperature} | Top-k: {top_k} | Max tokens: {max_tokens}")
    print(f"{'='*60}\n")

    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            idx,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
        )

    generated_text = tokenizer.decode(output_ids[0].tolist())
    print(generated_text)
    print(f"\n{'='*60}")
    print(f"Generated {len(output_ids[0]) - len(prompt_ids)} new tokens.")


if __name__ == '__main__':
    main()
