"""
FastAPI Inference Server for Scratch GPT.

Serves the trained model for the web playground.

Usage:
    pip install fastapi uvicorn
    python api/serve.py
    
    # Or with custom options:
    python api/serve.py --checkpoint checkpoints/code/checkpoint_final.pt --port 8000
"""
import os
import sys
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from utils.config import load_config
from utils.data_loader import load_text
from utils.tokenizer import CharTokenizer, BPETokenizer
from utils.training import load_checkpoint
from models.gpt import GPT


app = FastAPI(title="Scratch GPT API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model state
model = None
tokenizer = None
device = None
config = None


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 200
    temperature: float = 0.7
    top_k: int = 50


class GenerateResponse(BaseModel):
    generated_text: str
    prompt: str
    tokens_generated: int


class HealthResponse(BaseModel):
    status: str
    model_params: int
    device: str
    tokenizer: str


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        model_params=model.count_parameters() if model else 0,
        device=str(device),
        tokenizer=config.data.tokenizer if config else "unknown",
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    if model is None:
        return GenerateResponse(
            generated_text="Model not loaded",
            prompt=req.prompt,
            tokens_generated=0,
        )

    # Encode prompt
    prompt_ids = tokenizer.encode(req.prompt)
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            idx,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
        )

    generated_text = tokenizer.decode(output_ids[0].tolist())

    return GenerateResponse(
        generated_text=generated_text,
        prompt=req.prompt,
        tokens_generated=req.max_tokens,
    )


def load_model(config_path: str, checkpoint_path: str):
    global model, tokenizer, device, config

    config = load_config(config_path)

    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"Using device: {device}")

    # Tokenizer
    if config.data.tokenizer == 'bpe':
        tokenizer = BPETokenizer()
    else:
        text = load_text(config.data.dataset_path)
        tokenizer = CharTokenizer(text)

    # Model
    model = GPT(
        vocab_size=tokenizer.vocab_size,
        block_size=config.data.block_size,
        d_model=config.model.d_model,
        n_heads=config.model.n_heads,
        n_layers=config.model.n_layers,
        dropout=0.0,
        bias=config.model.bias,
    ).to(device)

    load_checkpoint(checkpoint_path, model)
    model.eval()

    print(f"Model loaded: {model.count_parameters():,} parameters")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"API ready at http://localhost:8000")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Scratch GPT Inference Server")
    parser.add_argument('--config', type=str, default='config_gpu_code.yaml')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/code/checkpoint_final.pt')
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()

    load_model(args.config, args.checkpoint)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)
