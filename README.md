<div align="center">

# ⚡ Scratch

### GPT From Scratch — A Complete Transformer Implementation

[![Live Demo](https://img.shields.io/badge/🌐_Live_Demo-web--bice--tau--53.vercel.app-00d4ff?style=for-the-badge)](https://web-bice-tau-53.vercel.app)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Next.js](https://img.shields.io/badge/Next.js-15-000000?style=for-the-badge&logo=next.js&logoColor=white)](https://nextjs.org)

*A 49.2M parameter GPT model built entirely from scratch using PyTorch. No HuggingFace, no shortcuts — pure Transformer architecture trained on Python code.*

</div>

---

## 🎯 What is Scratch?

Scratch is an educational + functional implementation of a GPT (Generative Pre-trained Transformer) model, built from absolute zero.

### ✨ Features

| Feature | Description |
|---------|-------------|
| 🧠 **Multi-Head Attention** | Scaled dot-product attention with causal masking |
| 🔄 **Grouped Query Attention** | Fewer KV heads for memory-efficient inference |
| ⚡ **Mixture of Experts** | Sparse MoE with top-k routing and load balancing |
| 🌀 **Rotary Embeddings (RoPE)** | Relative position encoding via rotation matrices |
| 📊 **Full Training Pipeline** | AdamW, cosine LR schedule, gradient clipping, AMP |
| 🎯 **DPO Alignment** | Direct Preference Optimization for post-training |
| 🌐 **Interactive Website** | Next.js playground deployed on Vercel |
| 🔌 **Inference API** | FastAPI server for live model interaction |

---

## 📐 Architecture

```
Input Tokens
    ↓
Token Embedding + Positional Embedding (384-dim)
    ↓
┌─────────────────────────────────────┐
│  Transformer Block (×6)            │
│  ├─ LayerNorm → Multi-Head Attn    │
│  ├─ Residual Connection            │
│  ├─ LayerNorm → Feed-Forward (MLP) │
│  └─ Residual Connection            │
└─────────────────────────────────────┘
    ↓
Final LayerNorm → Linear Head → Logits (100,277 vocab)
```

| Config | Value |
|--------|-------|
| Parameters | **49.2M** |
| Layers | 6 |
| Heads | 6 |
| d_model | 384 |
| FFN dim | 1,536 |
| Vocab | 100,277 (BPE) |
| Context | 256 tokens |

---

## 📊 Training Results

Trained on **122MB of Python code** (9,996 files) for **50,000 steps** on an NVIDIA RTX 4060.

| Metric | Value |
|--------|-------|
| Final Train Loss | **1.94** |
| Final Val Loss | **2.07** |
| Training Time | ~4.3 hours |
| Precision | FP16 (Mixed) |
| Effective Batch | 32 (8 × 4 accum) |

---

## 🚀 Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Train the model

```bash
# Download TinyShakespeare (or use your own data)
python train.py --config config.yaml

# Train on Python code (GPU recommended)
python train.py --config config_gpu_code.yaml
```

### Generate text

```bash
python generate.py --checkpoint checkpoints/code/checkpoint_final.pt --prompt "def fibonacci(n):" --temperature 0.7
```

### Interactive chat

```bash
python chat.py --config config_gpu_code.yaml --checkpoint checkpoints/code/checkpoint_final.pt
```

### Start the API server

```bash
python api/serve.py
```

### Run the website locally

```bash
cd web
npm install
npm run dev
```

---

## 📁 Project Structure

```
scratch/
├── models/
│   ├── attention.py      # Multi-Head & Grouped Query Attention
│   ├── embeddings.py     # Token + Positional Embeddings
│   ├── ffn.py            # Feed-Forward Network (GELU)
│   ├── block.py          # Transformer Block (Pre-Norm)
│   ├── gpt.py            # Full GPT Model + Generation
│   ├── rope.py           # Rotary Positional Embeddings
│   └── moe.py            # Mixture of Experts
├── utils/
│   ├── config.py         # YAML Config Loader
│   ├── data_loader.py    # Raw Text Reader
│   ├── tokenizer.py      # Char + BPE Tokenizers
│   ├── dataset.py        # PyTorch Dataset + DataLoader
│   └── training.py       # Optimizer, LR Schedule, Checkpointing
├── api/
│   └── serve.py          # FastAPI Inference Server
├── web/                  # Next.js Website (Vercel)
├── tests/                # 56 Unit Tests
├── train.py              # Training Loop (AMP + Grad Accum)
├── generate.py           # CLI Text Generation
├── chat.py               # Interactive Chat
├── finetune_sft.py       # Supervised Fine-Tuning
└── finetune_dpo.py       # Direct Preference Optimization
```

---

## 🧪 Tests

```bash
python -m pytest tests/ -v
# 56 passed ✅
```

---

## 🌐 Website

The interactive website is deployed at **[web-bice-tau-53.vercel.app](https://web-bice-tau-53.vercel.app)**

| Page | Description |
|------|-------------|
| **Home** | Hero landing page with project overview |
| **Playground** | Interactive chat with the model |
| **Architecture** | Visual Transformer pipeline walkthrough |
| **Training** | Loss curves and training metrics |

---

## 🛠 Tech Stack

- **Model**: PyTorch 2.x, tiktoken
- **Frontend**: Next.js 15, Tailwind CSS, Framer Motion
- **Backend**: Supabase, FastAPI
- **Deployment**: Vercel
- **Testing**: pytest (56 tests)

---

## 📜 License

MIT License — feel free to use, modify, and learn from this project.

---

<div align="center">
  <p>Built with ❤️ and PyTorch</p>
</div>
