"""
Heterogeneous MoE — Variable Expert Sizes for Losion Framework v0.4.

Upgrade #4: Jalur 3 MoE enhancement.  In a standard MoE every expert has the
same hidden dimension (d_ff), which forces a one-size-fits-all design.
HeterogeneousMoE allows experts to differ in capacity:

  * **Large experts** (high d_ff) handle general-knowledge patterns that
    benefit from rich representation.
  * **Small experts** (low d_ff) specialise in narrow, high-frequency
    patterns — they are cheaper to compute and need fewer parameters.

The router is **size-aware**: larger experts receive a capacity bonus so
they can serve more tokens without becoming a bottleneck, while small
experts are preferred for tokens that only need specialised processing.

Architecture
------------
Each expert *i* has its own ``d_ff_i`` (supplied via ``expert_ff_dims``).
Internally the module stores per-expert up/down projections:

    expert_i.up_proj   :  (d_model, d_ff_i)     — no bias
    expert_i.down_proj :  (d_ff_i, d_model)      — no bias
    expert_i.gate_proj :  (d_model, d_ff_i)      — no bias  (SwiGLU-style)

The router produces logits of shape ``(batch, seq_len, num_experts)``
which are then **bias-corrected** by a learnable per-expert capacity scalar
before top-K selection.

Load balancing is maintained through:
  1. An auxiliary load-balancing loss (standard Switch-Transformer style).
  2. Per-expert capacity factors that scale with ``sqrt(d_ff_i)`` so
     larger experts can absorb proportionally more tokens.

References
----------
- Fedus et al., "Switch Transformers: Scaling to Trillion Parameter Models"
  (2021) — load-balancing loss.
- Zhou et al., "Mixture-of-Experts with Expert Choice Routing" (2022) —
  expert-choice routing ideas.

Hardware: Pure PyTorch.  No custom CUDA kernels required.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Single Expert with configurable d_ff
# ---------------------------------------------------------------------------

class HeterogeneousExpert(nn.Module):
    """
    A single feed-forward expert with a **configurable** intermediate
    dimension ``d_ff``.

    Uses the SwiGLU activation pattern (same as LLaMA / PaLM):

        output = down_proj( SiLU(gate_proj(x)) * up_proj(x) )

    Args:
        d_model: Input/output dimension.
        d_ff:    Intermediate (hidden) dimension for this expert.
    """

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

        # Initialise with small weights for stability
        nn.init.normal_(self.up_proj.weight, std=0.01)
        nn.init.normal_(self.gate_proj.weight, std=0.01)
        nn.init.zeros_(self.down_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: ``(batch, seq_len, d_model)`` or ``(N, d_model)``.

        Returns:
            Same shape as input.
        """
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Size-aware router
# ---------------------------------------------------------------------------

class SizeAwareRouter(nn.Module):
    """
    Router that biases its logits toward larger experts so they can
    handle proportionally more tokens.

    The raw router output is:

        logits = Linear(x)                    # (B, S, E)
        logits = logits + capacity_bias       # (B, E)  — broadcast

    where ``capacity_bias`` is a **learnable** vector initialised
    proportionally to ``sqrt(d_ff_i / d_ff_max)`` so that larger experts
    start with a natural advantage.

    Args:
        d_model:      Model dimension.
        num_experts:  Number of experts.
        expert_ff_dims: List of per-expert d_ff values, length = num_experts.
        top_k:        Number of experts to select per token (default 2).
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        expert_ff_dims: List[int],
        top_k: int = 2,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k

        self.proj = nn.Linear(d_model, num_experts, bias=False)

        # Capacity bias — initial value proportional to sqrt(d_ff / d_ff_max)
        d_ff_max = max(expert_ff_dims)
        init_bias = torch.tensor(
            [math.sqrt(d / d_ff_max) for d in expert_ff_dims],
            dtype=torch.float32,
        )
        # Scale down so the bias is a gentle nudge, not a dominant term
        init_bias = init_bias * 0.1
        self.capacity_bias = nn.Parameter(init_bias)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute routing weights.

        Args:
            x: ``(batch, seq_len, d_model)``.

        Returns:
            weights:       ``(batch, seq_len, top_k)``  — softmax-normalised
                           weights for the selected experts.
            indices:       ``(batch, seq_len, top_k)``  — expert indices.
            router_logits: ``(batch, seq_len, num_experts)`` — raw logits
                           (needed for the load-balancing loss).
        """
        # (B, S, E)
        logits = self.proj(x) + self.capacity_bias.unsqueeze(0).unsqueeze(0)

        # Top-K selection
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)

        return top_k_weights, top_k_indices, logits


# ---------------------------------------------------------------------------
# Load-balancing loss (Switch Transformer style)
# ---------------------------------------------------------------------------

def heterogeneous_load_balance_loss(
    router_logits: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """
    Auxiliary load-balancing loss for heterogeneous MoE.

    Encourages each expert to receive a roughly equal fraction of the
    total routing probability mass, which prevents routing collapse.

    .. math::
        L_{balance} = N \\cdot \\sum_{i=1}^{N} f_i \\cdot P_i

    where:
        f_i = fraction of tokens routed to expert i
        P_i = mean routing probability for expert i
        N   = num_experts

    Args:
        router_logits: ``(batch, seq_len, num_experts)``.
        num_experts:   Total number of experts.

    Returns:
        Scalar loss.
    """
    probs = F.softmax(router_logits, dim=-1)  # (B, S, E)
    # Fraction of tokens assigned to each expert (via argmax)
    assignments = probs.argmax(dim=-1)  # (B, S)
    one_hot = F.one_hot(assignments, num_experts).float()  # (B, S, E)
    f = one_hot.mean(dim=(0, 1))  # (E,)
    P = probs.mean(dim=(0, 1))    # (E,)
    return num_experts * (f * P).sum()


# ---------------------------------------------------------------------------
# HeterogeneousMoE
# ---------------------------------------------------------------------------

class HeterogeneousMoE(nn.Module):
    """
    Mixture-of-Experts with **variable expert sizes**.

    Unlike a standard MoE where every expert shares the same ``d_ff``,
    HeterogeneousMoE accepts a list ``expert_ff_dims`` that specifies the
    intermediate dimension for each expert independently.  This enables:

    * Large experts (e.g. ``d_ff = 4096``) for broad, general patterns.
    * Small experts (e.g. ``d_ff = 1024``) for narrow, specialised patterns.
    * A size-aware router that gives larger experts a capacity advantage.

    The total parameter count is lower than a homogeneous MoE where every
    expert uses the *maximum* ``d_ff``, while still providing high capacity
    where it matters.

    Example
    -------
    >>> moe = HeterogeneousMoE(
    ...     d_model=512,
    ...     expert_ff_dims=[2048, 4096, 1024, 4096, 512, 2048],
    ...     top_k=2,
    ... )
    >>> x = torch.randn(2, 16, 512)
    >>> out, aux = moe(x)
    >>> out.shape
    torch.Size([2, 16, 512])
    >>> "load_balance" in aux
    True

    Args:
        d_model:         Model dimension.
        expert_ff_dims:  List of per-expert intermediate dimensions.
                         Length determines the number of experts.
        top_k:           Number of experts to activate per token (default 2).
        dropout:         Dropout rate (default 0.0).
        load_balance_weight: Weight for the auxiliary load-balancing loss
                             (default 0.01).
    """

    def __init__(
        self,
        d_model: int,
        expert_ff_dims: List[int],
        top_k: int = 2,
        dropout: float = 0.0,
        load_balance_weight: float = 0.01,
    ) -> None:
        super().__init__()

        if len(expert_ff_dims) < top_k:
            raise ValueError(
                f"Need at least top_k={top_k} experts, "
                f"got {len(expert_ff_dims)}."
            )

        self.d_model = d_model
        self.expert_ff_dims = list(expert_ff_dims)
        self.num_experts = len(expert_ff_dims)
        self.top_k = top_k
        self.load_balance_weight = load_balance_weight

        # ---- Experts ----
        self.experts = nn.ModuleList(
            [HeterogeneousExpert(d_model, d_ff) for d_ff in expert_ff_dims]
        )

        # ---- Size-aware router ----
        self.router = SizeAwareRouter(
            d_model=d_model,
            num_experts=self.num_experts,
            expert_ff_dims=expert_ff_dims,
            top_k=top_k,
        )

        # ---- Dropout ----
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # ---- Output norm ----
        self.output_norm = nn.RMSNorm(d_model, eps=1e-5)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through heterogeneous MoE.

        Each token is routed to the top-K experts; expert outputs are
        weighted by the router softmax scores and summed.

        Args:
            x: ``(batch, seq_len, d_model)``.

        Returns:
            output: ``(batch, seq_len, d_model)``.
            aux:    Dict of auxiliary losses with key ``"load_balance"``.
        """
        batch, seq_len, _ = x.shape

        # ---- Router ----
        weights, indices, router_logits = self.router(x)
        # weights: (B, S, K),  indices: (B, S, K)

        # Flatten for expert computation
        x_flat = x.view(batch * seq_len, self.d_model)  # (N, d_model)
        weights_flat = weights.view(batch * seq_len, self.top_k)  # (N, K)
        indices_flat = indices.view(batch * seq_len, self.top_k)  # (N, K)

        # ---- Dispatch to experts ----
        # We accumulate the weighted output for each token
        output_flat = torch.zeros_like(x_flat)  # (N, d_model)

        for k_idx in range(self.top_k):
            expert_indices = indices_flat[:, k_idx]  # (N,)
            expert_weights = weights_flat[:, k_idx].unsqueeze(-1)  # (N, 1)

            # Process each expert
            for expert_id in range(self.num_experts):
                # Mask: which tokens go to this expert at this K position?
                mask = (expert_indices == expert_id)  # (N,)
                if not mask.any():
                    continue

                # Select tokens for this expert
                expert_input = x_flat[mask]  # (n, d_model)
                expert_output = self.experts[expert_id](expert_input)  # (n, d_model)

                # Weight and accumulate
                expert_w = expert_weights[mask]  # (n, 1)
                output_flat[mask] += expert_w * self.dropout(expert_output)

        # Reshape
        output = output_flat.view(batch, seq_len, self.d_model)
        output = self.output_norm(output)

        # ---- Auxiliary losses ----
        aux: Dict[str, torch.Tensor] = {
            "load_balance": self.load_balance_weight
            * heterogeneous_load_balance_loss(router_logits, self.num_experts),
        }

        return output, aux

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def parameter_count_per_expert(self) -> List[int]:
        """Return the number of trainable parameters per expert."""
        return [
            sum(p.numel() for p in expert.parameters())
            for expert in self.experts
        ]

    def total_parameter_count(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters())

    def equivalent_homogeneous_d_ff(self) -> float:
        """
        The d_ff that a homogeneous MoE with the same total parameter
        count would have.

        Returns:
            Equivalent d_ff value (float).
        """
        total_expert_params = sum(self.parameter_count_per_expert())
        # Each expert has: up_proj (d*d_ff) + gate_proj (d*d_ff) + down_proj (d_ff*d)
        # = 3 * d * d_ff params per expert
        # So: total = num_experts * 3 * d * d_ff_equiv
        d_ff_equiv = total_expert_params / (self.num_experts * 3 * self.d_model)
        return d_ff_equiv
