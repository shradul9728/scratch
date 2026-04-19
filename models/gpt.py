"""
GPT Model: Full decoder-only Transformer for autoregressive language modeling.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.embeddings import TokenEmbedding, PositionalEmbedding
from models.block import TransformerBlock


class GPT(nn.Module):
    """
    GPT (Generative Pre-trained Transformer) model.

    Architecture:
        Token Embedding + Positional Embedding
        → N x TransformerBlock
        → LayerNorm
        → Linear projection to vocab_size
    """

    def __init__(
        self,
        vocab_size: int,
        block_size: int = 256,
        d_model: int = 384,
        n_heads: int = 6,
        n_layers: int = 6,
        dropout: float = 0.2,
        bias: bool = False,
        use_gqa: bool = False,
        n_kv_heads: int = None,
        use_moe: bool = False,
        n_experts: int = 8,
        top_k_experts: int = 2,
    ):
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size

        # Embeddings
        self.token_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionalEmbedding(block_size, d_model)
        self.drop = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
                block_size=block_size,
                bias=bias,
                use_gqa=use_gqa,
                n_kv_heads=n_kv_heads,
                use_moe=use_moe,
                n_experts=n_experts,
                top_k_experts=top_k_experts,
            )
            for _ in range(n_layers)
        ])

        # Final layer norm and output head
        self.ln_f = nn.LayerNorm(d_model, elementwise_affine=True)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: share token embedding weights with the output head
        self.token_emb.embedding.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Custom weight initialization following GPT-2 conventions."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        """
        Forward pass.

        Args:
            idx: (batch, seq_len) integer token IDs.
            targets: (batch, seq_len) target token IDs (optional).

        Returns:
            logits: (batch, seq_len, vocab_size)
            loss: scalar Cross-Entropy loss if targets provided, else None.
        """
        B, T = idx.shape
        assert T <= self.block_size, f"Sequence length {T} exceeds block_size {self.block_size}"

        # Embeddings
        tok_emb = self.token_emb(idx)               # (B, T, d_model)
        pos_emb = self.pos_emb(T, device=idx.device) # (T, d_model)
        x = self.drop(tok_emb + pos_emb)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
            # Add MoE auxiliary losses if applicable
            for block in self.blocks:
                if hasattr(block.ffn, 'aux_loss'):
                    loss = loss + 0.01 * block.ffn.aux_loss

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = None,
    ) -> torch.Tensor:
        """
        Autoregressive text generation.

        Args:
            idx: (batch, seq_len) conditioning token IDs.
            max_new_tokens: Number of new tokens to generate.
            temperature: Sampling temperature (< 1 = more deterministic).
            top_k: If set, only sample from the top-k most likely tokens.

        Returns:
            (batch, seq_len + max_new_tokens) generated token IDs.
        """
        for _ in range(max_new_tokens):
            # Crop context to block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # (B, vocab_size) — last token

            # Temperature scaling
            if temperature != 1.0:
                logits = logits / temperature

            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)

            # Greedy decoding when temperature is very small
            if temperature < 1e-8:
                idx_next = torch.argmax(probs, dim=-1, keepdim=True)
            else:
                idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat([idx, idx_next], dim=1)

        return idx
