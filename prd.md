# Product Requirements Document (PRD): GPT From Scratch

## 1. Project Overview
**Project Name:** Custom GPT (Generative Pre-trained Transformer) From Scratch
**Vision:** To build, train, and infer from a decoder-only Transformer model from scratch. This project aims to demystify Large Language Models (LLMs) by implementing the core architecture, data processing, and training pipelines without relying on high-level generation libraries like HuggingFace `transformers`.

## 2. Objectives
- **Educational & Architectural:** Deepen understanding of Attention mechanisms, Positional Encoding, and the Transformer block.
- **Modularity:** Create highly modular, well-documented components (Tokenizer, Dataset, Model, Trainer, Generator).
- **Scalability:** Ensure the architecture can scale from a tiny character-level model (e.g., for Shakespeare text) to a larger sub-word level model if given sufficient compute.

## 3. Scope & Phases

### Phase 1: Data Pipeline & Tokenization
- Implement a simple data loader capable of reading raw text corpora.
- **Tokenization:** 
  - V1: Character-level Tokenizer (mapping unique characters to integers).
  - V2: Byte-Pair Encoding (BPE) or Integration with `tiktoken` for sub-word tokenization.
- **Batching:** Create a PyTorch `DataLoader` or custom batching mechanism yielding `(x, y)` pairs where `y` is `x` shifted by one token (autoregressive target).

### Phase 2: Core Model Architecture (The Transformer)
- **Embeddings:** Token and Positional Embeddings.
- **Self-Attention:** Scaled Dot-Product Attention with causal masking (lower triangular mask to prevent looking ahead).
- **Multi-Head Attention (MHA):** Splitting embeddings into multiple heads, computing attention, and concatenating.
- **Feed-Forward Network (FFN):** Two-layer linear transformation with GELU/ReLU activation.
- **Transformer Block:** Combining MHA and FFN with residual connections and Layer Normalization.
- **Final Output:** Linear head projecting to vocabulary size.

### Phase 3: Training Pipeline
- **Loss Function:** Cross-Entropy Loss.
- **Optimizer:** AdamW with configurable learning rate, weight decay.
- **Learning Rate Scheduler:** Cosine annealing with warmup.
- **Training Loop:** Forward pass, loss computation, backward pass, gradient clipping, optimizer step.
- **Logging & Checkpointing:** Track Training/Validation loss via WandB or TensorBoard; save model weights periodically.

### Phase 4: Inference & Generation
- Autoregressive text generation function.
- Support for configurable temperature (to control randomness).
- Support for top-k filtering to sample from only the most likely next tokens.

## 4. Technology Stack
- **Language:** Python 3.10+
- **Core Framework:** PyTorch (Provides automatic differentiation and tensor ops without abstracting away the math).
- **Compute:** CUDA/MPS support for GPU acceleration.
- **Auxiliary:** `numpy`, `tqdm` (for progress bars), `matplotlib`/`wandb` (for metrics).

## 5. Directory Structure
```text
/
├── data/               # Raw and tokenized text datasets
├── models/             # Transformer architecture (attention.py, blocks.py, gpt.py)
├── utils/              # Tokenizer, data loaders, config parsing
├── train.py            # Main training loop
├── generate.py         # Inference CLI
├── config.yaml         # Hyperparameter configurations
└── requirements.txt    # Dependencies
```

## 6. Milestones & Timeline
*   **Milestone 1 (Week 1):** Data ingestion, batching, and custom Tokenizer complete.
*   **Milestone 2 (Week 2):** Core model implementation (MHA, FFN, standard Transformer blocks) passes unit tests for expected output shapes.
*   **Milestone 3 (Week 3):** Training loop complete. Model can successfully overfit on a tiny batch of text.
*   **Milestone 4 (Week 4):** Model trained on a mid-sized dataset (e.g., TinyShakespeare or OpenWebText subset). Text generation CLI functioning with sensible structural output.

## 7. Future Considerations
- KV-Caching for faster inference.
- FlashAttention integration for memory-efficient training.
- Instruction Fine-Tuning (SFT) and RLHF pipelines once the base model is operational.
