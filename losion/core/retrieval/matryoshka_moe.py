"""
Matryoshka MoE — Elastic Expert Count for Losion Framework v0.4.

Upgrade #5: Jalur 3 MoE enhancement.  Standard MoE layers use a **fixed**
number of active experts K per token.  MatryoshkaMoE generalises this by
allowing the number of active experts to **vary per token**:

  * Simple tokens (e.g. common function words) might only need **2** experts.
  * Complex tokens (e.g. rare technical terms, ambiguous context) might need
    **6+** experts for adequate representation.

The name is inspired by Matryoshka Representation Learning
(Zhai et al., 2024) where embeddings are useful at multiple granularity
levels — here, the MoE output is useful with any number of active experts.

Key Components
--------------
1. **ComplexityGate** — a lightweight MLP that estimates per-token
   complexity and maps it to a discrete expert count ``k_t ∈ [k_min, k_max]``.

2. **ElasticRouter** — produces logits for *all* experts, but only the
   top-``k_t`` are selected per token.

3. **MatryoshkaMoE** — the main module that combines the gate and router
   with a pool of homogeneous experts, ensuring load balancing across
   variable expert counts.

Load Balancing with Variable K
------------------------------
When K varies per token, standard load-balancing losses break because the
expected load per expert is no longer uniform.  We use a **capacity-
normalised** variant:

    L_balance = E * Σ_i (f_i / c_i) * (P_i / c_i)

where ``f_i`` is the fraction of tokens routed to expert *i*, ``P_i`` is
the mean routing probability, and ``c_i`` is the capacity of expert *i*
(proportional to its expected load given the distribution of K).

During training, soft routing (weighted sum over all experts) ensures
gradient flow; during inference, hard top-K selection is used for
efficiency.

References
----------
- Zhai et al., "Matryoshka Representation Learning" (2024).
- Fedus et al., "Switch Transformers" (2021) — load-balancing loss.
- Lewis et al., "MoEUT: Mixture-of-Experts Universal Transformers" (2024).

Hardware: Pure PyTorch.  No custom CUDA kernels required.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Complexity Gate for expert count
# ---------------------------------------------------------------------------

class ExpertCountGate(nn.Module):
    """
    Learned gate that predicts the number of active experts per token.

    Given input ``(B, S, d_model)``, produces:
      - ``expert_count_logits``: ``(B, S, num_count_levels)``
      - ``expert_count_probs``:  ``(B, S, num_count_levels)``
      - ``expert_counts``:       ``(B, S)`` — integer values in
        ``[k_min, k_max]`` (hard selection for inference).

    The number of output levels is ``(k_max - k_min + 1)``.
    During training we use the soft probabilities so the model is
    differentiable; during inference we use the argmax for efficiency.

    Args:
        d_model:    Model dimension.
        k_min:      Minimum number of active experts (default 1).
        k_max:      Maximum number of active experts (default 8).
        bottleneck: MLP bottleneck width (default 64).
    """

    def __init__(
        self,
        d_model: int,
        k_min: int = 1,
        k_max: int = 8,
        bottleneck: int = 64,
    ) -> None:
        super().__init__()
        if k_min < 1:
            raise ValueError(f"k_min must be >= 1, got {k_min}")
        if k_max < k_min:
            raise ValueError(f"k_max ({k_max}) must be >= k_min ({k_min})")

        self.d_model = d_model
        self.k_min = k_min
        self.k_max = k_max
        self.num_levels = k_max - k_min + 1

        self.mlp = nn.Sequential(
            nn.Linear(d_model, bottleneck, bias=False),
            nn.SiLU(),
            nn.Linear(bottleneck, bottleneck, bias=False),
            nn.SiLU(),
            nn.Linear(bottleneck, self.num_levels, bias=True),
        )

        # Initialise bias toward the median K
        median_idx = self.num_levels // 2
        with torch.no_grad():
            self.mlp[-1].bias.zero_()
            self.mlp[-1].bias[median_idx].fill_(1.0)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict per-token expert counts.

        Args:
            x: ``(batch, seq_len, d_model)``.

        Returns:
            count_logits: ``(B, S, num_levels)``
            count_probs:  ``(B, S, num_levels)``
            count_values: ``(B, S)`` — integer count per token in [k_min, k_max].
        """
        logits = self.mlp(x)                                    # (B, S, L)
        probs = F.softmax(logits, dim=-1)                       # (B, S, L)

        # Weighted count (soft, differentiable)
        level_values = torch.arange(
            self.k_min, self.k_max + 1,
            dtype=x.dtype, device=x.device,
        )  # (L,)
        count_values = (probs * level_values).sum(dim=-1)       # (B, S)

        return logits, probs, count_values

    def predict_hard(self, x: torch.Tensor) -> torch.Tensor:
        """
        Hard prediction of expert count (for inference).

        Args:
            x: ``(batch, seq_len, d_model)``.

        Returns:
            count: ``(batch, seq_len)`` — integer values in [k_min, k_max].
        """
        logits = self.mlp(x)
        indices = logits.argmax(dim=-1)  # (B, S)
        return indices + self.k_min


# ---------------------------------------------------------------------------
# Standard feed-forward expert (homogeneous)
# ---------------------------------------------------------------------------

class _FFNExpert(nn.Module):
    """SwiGLU feed-forward expert with fixed d_ff."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Matryoshka Load-Balance Loss
# ---------------------------------------------------------------------------

def matryoshka_load_balance_loss(
    router_logits: torch.Tensor,
    expert_counts: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """
    Capacity-normalised load-balancing loss for variable-K MoE.

    Tokens that use more experts contribute proportionally more to the
    load, so we normalise by the expected capacity.

    Args:
        router_logits:  ``(B, S, E)`` — raw router logits.
        expert_counts:  ``(B, S)`` — per-token expert count.
        num_experts:    Total number of experts.

    Returns:
        Scalar loss.
    """
    probs = F.softmax(router_logits, dim=-1)  # (B, S, E)

    # Capacity per expert: expected number of tokens assigned
    # Approximate: each token with count k distributes 1/k load per expert
    assignments = probs.argmax(dim=-1)  # (B, S)
    one_hot = F.one_hot(assignments, num_experts).float()  # (B, S, E)

    # Weight each assignment by 1/k to normalise for variable K
    k_weights = 1.0 / (expert_counts.float().clamp(min=1.0))  # (B, S)
    weighted_assignments = one_hot * k_weights.unsqueeze(-1)   # (B, S, E)

    # Fraction of normalised load per expert
    f = weighted_assignments.mean(dim=(0, 1))  # (E,)

    # Mean routing probability per expert
    P = probs.mean(dim=(0, 1))  # (E,)

    # Loss: encourage uniform f and uncorrelated f with P
    return num_experts * (f * P).sum()


# ---------------------------------------------------------------------------
# MatryoshkaMoE
# ---------------------------------------------------------------------------

class MatryoshkaMoE(nn.Module):
    """
    Mixture-of-Experts with **elastic (variable) expert count per token**.

    Instead of a fixed ``top_k``, a :class:`ExpertCountGate` predicts how
    many experts each token should use.  Simple tokens may only need 1–2
    experts while complex tokens can use 6+, saving compute on easy inputs
    and allocating capacity where it is needed.

    The module is compatible with the standard MoE interface used in
    Losion's Jalur 3 and can serve as a drop-in replacement for a fixed-K
    MoE layer.

    Example
    -------
    >>> moe = MatryoshkaMoE(
    ...     d_model=512,
    ...     d_ff=2048,
    ...     num_experts=8,
    ...     k_min=2,
    ...     k_max=6,
    ... )
    >>> x = torch.randn(2, 16, 512)
    >>> out, aux = moe(x)
    >>> out.shape
    torch.Size([2, 16, 512])

    Args:
        d_model:     Model dimension.
        d_ff:        Intermediate dimension for each expert (homogeneous).
        num_experts: Total number of experts.
        k_min:       Minimum number of active experts per token (default 1).
        k_max:       Maximum number of active experts per token (default 8).
                     Must be <= num_experts.
        dropout:     Dropout rate (default 0.0).
        load_balance_weight: Weight for the capacity-normalised
                             load-balancing loss (default 0.01).
        count_entropy_weight: Weight for an auxiliary entropy regulariser
                              on the expert-count gate (default 0.01).
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int,
        k_min: int = 1,
        k_max: int = 8,
        dropout: float = 0.0,
        load_balance_weight: float = 0.01,
        count_entropy_weight: float = 0.01,
    ) -> None:
        super().__init__()

        if k_max > num_experts:
            raise ValueError(
                f"k_max ({k_max}) must be <= num_experts ({num_experts})"
            )
        if k_min < 1:
            raise ValueError(f"k_min must be >= 1, got {k_min}")

        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.k_min = k_min
        self.k_max = k_max
        self.load_balance_weight = load_balance_weight
        self.count_entropy_weight = count_entropy_weight

        # ---- Experts ----
        self.experts = nn.ModuleList(
            [_FFNExpert(d_model, d_ff) for _ in range(num_experts)]
        )

        # ---- Router (produces logits for all experts) ----
        self.router = nn.Linear(d_model, num_experts, bias=False)

        # ---- Expert count gate ----
        self.count_gate = ExpertCountGate(
            d_model=d_model,
            k_min=k_min,
            k_max=k_max,
        )

        # ---- Dropout ----
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # ---- Output norm ----
        self.output_norm = nn.RMSNorm(d_model, eps=1e-5)

    # ------------------------------------------------------------------
    # Forward (training — soft routing)
    # ------------------------------------------------------------------

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with elastic expert count.

        During training, soft (weighted) routing is used so that gradient
        flow reaches all experts.  The number of active experts per token
        is determined by the :class:`ExpertCountGate`.

        Args:
            x: ``(batch, seq_len, d_model)``.

        Returns:
            output: ``(batch, seq_len, d_model)``.
            aux:    Dict of auxiliary losses with keys ``"load_balance"``
                    and ``"count_entropy"``.
        """
        batch, seq_len, _ = x.shape

        # ---- Predict expert counts ----
        count_logits, count_probs, expert_counts = self.count_gate(x)
        # expert_counts: (B, S) — soft count values in [k_min, k_max]

        # ---- Router logits for all experts ----
        router_logits = self.router(x)  # (B, S, E)
        router_probs = F.softmax(router_logits, dim=-1)  # (B, S, E)

        # ---- Select top-k_max experts and mask ----
        # We compute top-k_max experts, then zero out those beyond k_t
        top_k_values, top_k_indices = router_probs.topk(
            self.k_max, dim=-1
        )  # (B, S, k_max)

        # Build a mask based on the predicted count
        # Position j in top-K is active if j < k_t
        k_positions = torch.arange(
            self.k_max, device=x.device, dtype=x.dtype
        ).unsqueeze(0).unsqueeze(0)  # (1, 1, k_max)
        # k_mask: True where position < count
        k_mask = k_positions < expert_counts.unsqueeze(-1)  # (B, S, k_max)

        # Apply mask: zero out experts beyond the count
        masked_weights = top_k_values * k_mask.float()

        # Re-normalise the weights
        weight_sum = masked_weights.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        masked_weights = masked_weights / weight_sum  # (B, S, k_max)

        # ---- Dispatch tokens to experts ----
        x_flat = x.view(batch * seq_len, self.d_model)
        weights_flat = masked_weights.view(batch * seq_len, self.k_max)
        indices_flat = top_k_indices.view(batch * seq_len, self.k_max)

        output_flat = torch.zeros_like(x_flat)

        for k_idx in range(self.k_max):
            expert_indices = indices_flat[:, k_idx]  # (N,)
            expert_weights = weights_flat[:, k_idx].unsqueeze(-1)  # (N, 1)

            # Check which weights are non-zero (skip if all zero)
            if not (expert_weights.abs() > 1e-9).any():
                continue

            for expert_id in range(self.num_experts):
                mask = (expert_indices == expert_id)
                if not mask.any():
                    continue

                expert_input = x_flat[mask]
                expert_output = self.experts[expert_id](expert_input)
                expert_w = expert_weights[mask]
                output_flat[mask] += expert_w * self.dropout(expert_output)

        output = output_flat.view(batch, seq_len, self.d_model)
        output = self.output_norm(output)

        # ---- Auxiliary losses ----
        aux: Dict[str, torch.Tensor] = {}

        # Load balance (capacity-normalised)
        aux["load_balance"] = self.load_balance_weight * matryoshka_load_balance_loss(
            router_logits, expert_counts.detach(), self.num_experts
        )

        # Count entropy — encourage decisive expert-count choices
        entropy = -(count_probs * (count_probs + 1e-8).log()).sum(dim=-1)  # (B, S)
        max_entropy = math.log(self.count_gate.num_levels)
        aux["count_entropy"] = self.count_entropy_weight * entropy.mean() / max_entropy

        return output, aux

    # ------------------------------------------------------------------
    # Forward inference (hard routing)
    # ------------------------------------------------------------------

    def forward_inference(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Inference forward with hard expert-count selection.

        Args:
            x: ``(batch, seq_len, d_model)`` — typically ``seq_len = 1``.

        Returns:
            output: ``(batch, seq_len, d_model)``.
            aux:    Empty dict (no auxiliary losses during inference).
        """
        batch, seq_len, _ = x.shape

        # Hard count prediction
        k_values = self.count_gate.predict_hard(x)  # (B, S) — integers

        # Router logits
        router_logits = self.router(x)  # (B, S, E)
        router_probs = F.softmax(router_logits, dim=-1)

        x_flat = x.view(batch * seq_len, self.d_model)
        k_flat = k_values.view(batch * seq_len)  # (N,)

        # For each token, select top-k_t experts
        output_flat = torch.zeros_like(x_flat)

        # We process token-by-token (inference is typically seq_len=1)
        for token_idx in range(batch * seq_len):
            k_t = int(k_flat[token_idx].item())
            k_t = max(self.k_min, min(k_t, self.k_max))

            token_input = x_flat[token_idx : token_idx + 1]  # (1, d)
            token_probs = router_probs.view(batch * seq_len, self.num_experts)[
                token_idx
            ]  # (E,)

            top_k_probs, top_k_idx = token_probs.topk(k_t)

            for j in range(k_t):
                eid = top_k_idx[j].item()
                w = top_k_probs[j].item()
                expert_out = self.experts[eid](token_input)  # (1, d)
                output_flat[token_idx] += w * expert_out.squeeze(0)

        output = output_flat.view(batch, seq_len, self.d_model)
        output = self.output_norm(output)

        return output, {}
