"""
Mixture-of-Experts (MoE) Feed-Forward layer.
Implements top-k expert routing with load-balancing loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertFFN(nn.Module):
    """A single expert: same architecture as a standard FFN."""

    def __init__(self, d_model: int, bias: bool = False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model, bias=bias),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model, bias=bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MoELayer(nn.Module):
    """
    Mixture-of-Experts layer with top-k routing and auxiliary load-balancing loss.

    Replaces a dense FFN in the Transformer block. Each token is routed to
    the top-k experts, and the output is the weighted sum of expert outputs.
    """

    def __init__(
        self,
        d_model: int,
        n_experts: int = 8,
        top_k: int = 2,
        bias: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k

        # Gating / router network
        self.gate = nn.Linear(d_model, n_experts, bias=False)

        # Expert networks
        self.experts = nn.ModuleList([
            ExpertFFN(d_model, bias=bias) for _ in range(n_experts)
        ])

        self.dropout = nn.Dropout(dropout)

        # Stores the auxiliary load-balancing loss from the last forward pass
        self.aux_loss = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            (batch, seq_len, d_model) — weighted mixture of expert outputs.
        """
        B, T, C = x.shape
        x_flat = x.view(-1, C)  # (B*T, C)
        N = x_flat.size(0)

        # Router logits and top-k selection
        router_logits = self.gate(x_flat)  # (N, n_experts)
        topk_values, topk_indices = torch.topk(router_logits, self.top_k, dim=-1)  # (N, top_k)
        topk_weights = F.softmax(topk_values, dim=-1)  # (N, top_k)

        # Compute auxiliary load-balancing loss
        # f_i = fraction of tokens routed to expert i
        # P_i = mean routing probability for expert i
        # aux_loss = n_experts * sum(f_i * P_i)
        router_probs = F.softmax(router_logits, dim=-1)  # (N, n_experts)
        # One-hot encoding of selected experts
        expert_mask = F.one_hot(topk_indices, self.n_experts).float()  # (N, top_k, n_experts)
        expert_mask = expert_mask.sum(dim=1)  # (N, n_experts)
        f = expert_mask.mean(dim=0)  # (n_experts,) fraction of tokens per expert
        P = router_probs.mean(dim=0)  # (n_experts,) mean probability per expert
        self.aux_loss = (self.n_experts * (f * P).sum())

        # Dispatch to experts and combine
        output = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            expert_idx = topk_indices[:, k]  # (N,)
            weight = topk_weights[:, k].unsqueeze(-1)  # (N, 1)

            for e in range(self.n_experts):
                mask = (expert_idx == e)
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts[e](expert_input)
                    output[mask] += weight[mask] * expert_output

        output = self.dropout(output)
        return output.view(B, T, C)
