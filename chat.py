"""
Interactive GPT Chat — type prompts and see the model complete them in real-time.

Usage:
    python chat.py
    python chat.py --config config_cpu.yaml --checkpoint checkpoints/checkpoint_final.pt
"""
import argparse
import torch

from utils.config import load_config
from utils.data_loader import load_text
from utils.tokenizer import CharTokenizer, BPETokenizer
from models.gpt import GPT
from utils.training import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive GPT Chat")
    parser.add_argument('--config', type=str, default='config_cpu.yaml')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/checkpoint_final.pt')
    parser.add_argument('--max_tokens', type=int, default=200)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top_k', type=int, default=40)
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

    # Tokenizer
    if cfg.data.tokenizer == 'bpe':
        tokenizer = BPETokenizer()
    else:
        text = load_text(cfg.data.dataset_path)
        tokenizer = CharTokenizer(text)

    # Model
    model = GPT(
        vocab_size=tokenizer.vocab_size,
        block_size=cfg.data.block_size,
        d_model=cfg.model.d_model,
        n_heads=cfg.model.n_heads,
        n_layers=cfg.model.n_layers,
        dropout=0.0,
        bias=cfg.model.bias,
    ).to(device)

    load_checkpoint(args.checkpoint, model)
    model.eval()

    param_count = model.count_parameters()

    print()
    print("=" * 60)
    print("  GPT FROM SCRATCH — Interactive Mode")
    print("=" * 60)
    print(f"  Model      : {cfg.model.n_layers}L / {cfg.model.n_heads}H / {cfg.model.d_model}D")
    print(f"  Parameters : {param_count:,}")
    print(f"  Device     : {device}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Top-k      : {args.top_k}")
    print(f"  Max tokens : {args.max_tokens}")
    print("=" * 60)
    print("  Type a prompt and press Enter to generate.")
    print("  Commands:  /temp 0.5  /topk 20  /tokens 300  /quit")
    print("=" * 60)
    print()

    temperature = args.temperature
    top_k = args.top_k
    max_tokens = args.max_tokens

    while True:
        try:
            prompt = input("\033[96mYou > \033[0m")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not prompt.strip():
            continue

        # Handle commands
        if prompt.strip().startswith("/"):
            parts = prompt.strip().split()
            cmd = parts[0].lower()

            if cmd == "/quit" or cmd == "/exit":
                print("Goodbye!")
                break
            elif cmd == "/temp" and len(parts) == 2:
                temperature = float(parts[1])
                print(f"  Temperature set to {temperature}")
                continue
            elif cmd == "/topk" and len(parts) == 2:
                top_k = int(parts[1])
                print(f"  Top-k set to {top_k}")
                continue
            elif cmd == "/tokens" and len(parts) == 2:
                max_tokens = int(parts[1])
                print(f"  Max tokens set to {max_tokens}")
                continue
            elif cmd == "/help":
                print("  /temp <float>   — set sampling temperature")
                print("  /topk <int>     — set top-k filtering")
                print("  /tokens <int>   — set max generation length")
                print("  /quit           — exit")
                continue
            else:
                print("  Unknown command. Type /help for options.")
                continue

        # Encode and generate
        prompt_ids = tokenizer.encode(prompt)
        idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            output_ids = model.generate(
                idx,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
            )

        generated_text = tokenizer.decode(output_ids[0].tolist())

        print(f"\033[93mGPT > \033[0m{generated_text}")
        print()


if __name__ == '__main__':
    main()
