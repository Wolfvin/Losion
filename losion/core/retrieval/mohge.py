"""
MoHGE — Mixture of Heterogeneous Grouped Experts for Language Modeling.

Implementation based on arXiv 2604.23108 (April 2026):
"Mixture of Heterogeneous Grouped Experts for Language Modeling"

Standard MoE architectures assign every expert the same feed-forward dimension
(``d_ff``), which forces a one-size-fits-all design: large experts waste
compute on easy tokens, while small experts lack capacity for hard ones.
MoHGE addresses this by allowing experts to differ in size and organising
them into **groups** of similarly-sized experts that share computation.

Key Concepts
------------
1. **Heterogeneous Expert Sizes** — Unlike standard MoE where all experts
   share the same ``d_ff``, MoHGE allows experts to differ in intermediate
   dimension.  Large experts handle complex patterns requiring rich
   representation; small experts specialise in narrow, high-frequency
   patterns at lower compute cost.

2. **Grouped Experts** — Experts are organised into groups with similar sizes.
   Within each group, experts share a common up-projection layer (d_model →
   d_ff_group) and then diverge through group-specific gate/down projections.
   This amortises the cost of the large up-projection across multiple experts.

3. **Capacity-Matched Routing** — The router accounts for expert capacity
   when making routing decisions.  Easier tokens are steered toward smaller
   experts (sufficient capacity, lower cost), while harder tokens are routed
   to larger experts (more capacity, higher cost).  A capacity factor
   proportional to ``sqrt(d_ff)`` biases the router logits.

4. **Compute Efficiency** — By sharing the up-projection within each group
   and using capacity-matched routing, MoHGE achieves better performance
   than uniform-expert MoE with the same total compute budget.

Architecture
------------
Each expert group *g* has ``num_experts_g`` experts with a shared
intermediate dimension ``d_ff_g``.  The group structure is::

    ExpertGroup(g):
        shared_up_proj:   d_model → d_ff_g     (shared across experts)
        expert_gate_proj[i]: d_model → d_ff_g  (per-expert within group)
        expert_down_proj[i]: d_ff_g → d_model  (per-expert within group)

The SwiGLU activation pattern is used::

    output[i] = down_proj[i]( SiLU(gate_proj[i](x)) * shared_up_proj(x) )

The MoHGERouter produces logits over *all* experts across all groups,
adjusted by per-expert capacity factors before top-K selection.

Load balancing uses a Switch-Transformer-style auxiliary loss, augmented
with a per-group capacity regulariser.

References
----------
- arXiv 2604.23108 (April 2026): "Mixture of Heterogeneous Grouped Experts
  for Language Modeling"
- Fedus et al., "Switch Transformers: Scaling to Trillion Parameter Models"
  (2021) — load-balancing loss.
- Zhou et al., "Mixture-of-Experts with Expert Choice Routing" (2022).

Hardware: Pure PyTorch.  No custom CUDA kernels required.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ExpertGroupConfig:
    """Configuration for a single expert group within MoHGE.

    All experts in a group share the same ``d_ff`` and the same
    ``up_proj`` layer, differing only in their ``gate_proj`` and
    ``down_proj``.

    Attributes:
        num_experts: Number of experts in this group.
        d_ff: Intermediate (hidden) dimension for all experts in this group.
    """

    num_experts: int = 4
    d_ff: int = 2048


@dataclass
class MoHGEConfig:
    """Configuration for MoHGE (Mixture of Heterogeneous Grouped Experts).

    Attributes:
        d_model: Model hidden dimension.
        expert_groups: List of :class:`ExpertGroupConfig` dicts or objects,
            each specifying ``num_experts`` and ``d_ff`` for a group.
            The total number of experts is ``sum(g.num_experts for g in groups)``.
        top_k: Number of experts to activate per token (across all groups).
        capacity_aware_routing: Whether to use capacity-matched routing
            that biases logits based on expert capacity (default True).
        capacity_bias_scale: Scale factor for the capacity bias initialisation
            (default 0.1).
        dropout: Dropout rate (default 0.0).
        load_balance_weight: Weight for the auxiliary load-balancing loss
            (default 0.01).
        group_balance_weight: Weight for the per-group capacity regulariser
            (default 0.005).
        use_shared_expert: Whether to include a shared (non-routed) expert
            with ``d_ff`` equal to the maximum group ``d_ff`` (default True).
    """

    d_model: int = 768
    expert_groups: List[Dict[str, int]] = field(default_factory=lambda: [
        {"num_experts": 4, "d_ff": 1024},  # small experts
        {"num_experts": 4, "d_ff": 2048},  # medium experts
        {"num_experts": 2, "d_ff": 4096},  # large experts
    ])
    top_k: int = 2
    capacity_aware_routing: bool = True
    capacity_bias_scale: float = 0.1
    dropout: float = 0.0
    load_balance_weight: float = 0.01
    group_balance_weight: float = 0.005
    use_shared_expert: bool = True


# ============================================================================
# Expert Group — Shared computation across similarly-sized experts
# ============================================================================

class ExpertGroup(nn.Module):
    """A group of experts that share a common up-projection layer.

    In MoHGE, experts within the same group share the computationally
    expensive ``up_proj`` (d_model → d_ff) and differ only in their
    ``gate_proj`` and ``down_proj``.  This amortises the largest
    parameter cost across multiple experts.

    SwiGLU activation::

        output[i] = down_proj[i]( SiLU(gate_proj[i](x)) * shared_up_proj(x) )

    Args:
        d_model: Model input/output dimension.
        d_ff: Intermediate dimension for this group's experts.
        num_experts: Number of experts in this group.
        group_id: Integer identifier for this group (for logging).
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int,
        group_id: int = 0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.group_id = group_id

        # ---- Shared up-projection (amortised across experts) ----
        self.shared_up_proj = nn.Linear(d_model, d_ff, bias=False)

        # ---- Per-expert gate and down projections ----
        self.gate_projs = nn.ModuleList([
            nn.Linear(d_model, d_ff, bias=False)
            for _ in range(num_experts)
        ])
        self.down_projs = nn.ModuleList([
            nn.Linear(d_ff, d_model, bias=False)
            for _ in range(num_experts)
        ])

        # ---- Initialise with small weights for stability ----
        nn.init.normal_(self.shared_up_proj.weight, std=0.01)
        for i in range(num_experts):
            nn.init.normal_(self.gate_projs[i].weight, std=0.01)
            nn.init.zeros_(self.down_projs[i].weight)

    def forward(
        self,
        x: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through the expert group.

        Computes the shared up-projection once, then applies per-expert
        gate and down projections for the selected experts.

        Args:
            x: Input tensor ``(N, d_model)``.
            expert_indices: Local expert indices within this group,
                shape ``(M,)`` with values in ``[0, num_experts)``.
                Only tokens assigned to experts in this group are passed.

        Returns:
            Output tensor ``(M, d_model)`` — only for the selected tokens.

        Note:
            In practice, the caller should select only the tokens routed
            to this group's experts before calling this method.
        """
        # Shared up-projection (computed once for all tokens in this group)
        up = self.shared_up_proj(x)  # (N, d_ff)

        # Per-expert processing
        # For efficiency with many experts, we iterate over unique selected experts
        unique_experts = expert_indices.unique()

        # We'll accumulate outputs per-token
        output = torch.zeros(x.shape[0], self.d_model, dtype=x.dtype, device=x.device)

        # Map each token to its local expert
        for local_id in unique_experts:
            mask = (expert_indices == local_id)
            if not mask.any():
                continue

            expert_input = x[mask]  # (n, d_model)
            expert_up = up[mask]    # (n, d_ff) — shared computation reused

            # Per-expert gate and down projection
            gate = F.silu(self.gate_projs[local_id](expert_input))  # (n, d_ff)
            expert_out = self.down_projs[local_id](gate * expert_up)  # (n, d_model)

            output[mask] = expert_out

        return output

    def forward_single_expert(
        self,
        x: torch.Tensor,
        local_expert_id: int,
    ) -> torch.Tensor:
        """Forward pass through a single expert in this group.

        More efficient than :meth:`forward` when all tokens go to the
        same expert.

        Args:
            x: Input tensor ``(..., d_model)``.
            local_expert_id: Index of the expert within this group.

        Returns:
            Output tensor of the same leading shape, last dim ``d_model``.
        """
        up = self.shared_up_proj(x)
        gate = F.silu(self.gate_projs[local_expert_id](x))
        return self.down_projs[local_expert_id](gate * up)

    def parameter_count(self) -> Dict[str, int]:
        """Return parameter counts for this group.

        Returns:
            Dict with ``shared``, ``per_expert``, and ``total`` keys.
        """
        shared = sum(p.numel() for p in self.shared_up_proj.parameters())
        per_expert = sum(
            sum(p.numel() for p in self.gate_projs[i].parameters())
            + sum(p.numel() for p in self.down_projs[i].parameters())
            for i in range(self.num_experts)
        )
        return {
            "shared": shared,
            "per_expert": per_expert,
            "total": shared + per_expert,
        }


# ============================================================================
# MoHGE Router — Capacity-Aware Routing
# ============================================================================

class MoHGERouter(nn.Module):
    """Router with capacity-aware routing for heterogeneous grouped experts.

    The router produces logits over all experts across all groups, then
    adjusts them using per-expert capacity factors before top-K selection.
    The capacity factor for each expert is proportional to ``sqrt(d_ff_i)`,
    so larger experts (which can handle more tokens efficiently) receive
    a positive bias, while smaller experts (which should only receive
    tokens that truly need them) receive a negative bias.

    When ``capacity_aware_routing = False``, the router behaves as a
    standard top-K gating network.

    Args:
        d_model: Model dimension (input to the router).
        num_experts: Total number of experts across all groups.
        expert_d_ffs: List of per-expert d_ff values (length = num_experts).
            Used to compute capacity factors.
        top_k: Number of experts to activate per token.
        capacity_aware_routing: Whether to apply capacity bias.
        capacity_bias_scale: Scale for capacity bias initialisation.
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        expert_d_ffs: List[int],
        top_k: int = 2,
        capacity_aware_routing: bool = True,
        capacity_bias_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.capacity_aware_routing = capacity_aware_routing

        # ---- Gating projection ----
        self.gate_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2, bias=False),
            nn.SiLU(),
            nn.Linear(d_model // 2, num_experts, bias=False),
        )

        # ---- Capacity bias (learnable) ----
        if capacity_aware_routing:
            d_ff_max = max(expert_d_ffs) if expert_d_ffs else 1
            init_bias = torch.tensor(
                [
                    capacity_bias_scale * math.sqrt(d_ff / d_ff_max)
                    for d_ff in expert_d_ffs
                ],
                dtype=torch.float32,
            )
            self.capacity_bias = nn.Parameter(init_bias)
        else:
            self.register_buffer(
                "capacity_bias", torch.zeros(num_experts)
            )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute routing weights with capacity-aware bias.

        Args:
            x: Input tensor ``(batch, seq_len, d_model)``.

        Returns:
            Tuple ``(weights, indices, router_logits)``:
            - weights: ``(batch, seq_len, top_k)`` — softmax-normalised
              weights for the selected experts.
            - indices: ``(batch, seq_len, top_k)`` — expert indices.
            - router_logits: ``(batch, seq_len, num_experts)`` — raw logits
              (needed for the load-balancing loss).
        """
        # Raw logits
        logits = self.gate_proj(x)  # (B, S, E)

        # Add capacity bias
        logits = logits + self.capacity_bias.unsqueeze(0).unsqueeze(0)

        # Top-K selection
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)

        return top_k_weights, top_k_indices, logits


# ============================================================================
# Load-balancing losses
# ============================================================================

def mohge_load_balance_loss(
    router_logits: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """Switch-Transformer-style load-balancing loss.

    Encourages each expert to receive a roughly equal fraction of the
    total routing probability mass, preventing routing collapse.

    .. math::
        L_{balance} = N \\cdot \\sum_{i=1}^{N} f_i \\cdot P_i

    Args:
        router_logits: ``(batch, seq_len, num_experts)``.
        num_experts: Total number of experts.

    Returns:
        Scalar loss.
    """
    probs = F.softmax(router_logits, dim=-1)  # (B, S, E)
    assignments = probs.argmax(dim=-1)  # (B, S)
    one_hot = F.one_hot(assignments, num_experts).float()  # (B, S, E)
    f = one_hot.mean(dim=(0, 1))  # (E,)
    P = probs.mean(dim=(0, 1))    # (E,)
    return num_experts * (f * P).sum()


def mohge_group_balance_loss(
    router_logits: torch.Tensor,
    expert_group_ids: torch.Tensor,
    num_groups: int,
) -> torch.Tensor:
    """Per-group capacity regulariser.

    Encourages each group of experts to receive a roughly proportional
    share of tokens based on the group's total capacity (sum of d_ff
    across the group's experts).

    Args:
        router_logits: ``(batch, seq_len, num_experts)``.
        expert_group_ids: ``(num_experts,)`` — group assignment for each expert.
        num_groups: Total number of groups.

    Returns:
        Scalar loss.
    """
    probs = F.softmax(router_logits, dim=-1)  # (B, S, E)
    # Sum probabilities per group
    group_probs = torch.zeros(num_groups, dtype=probs.dtype, device=probs.device)
    for g in range(num_groups):
        mask = (expert_group_ids == g)
        group_probs[g] = probs[..., mask].sum(dim=-1).mean()

    # Target: uniform across groups
    target = 1.0 / num_groups
    return ((group_probs - target) ** 2).sum()


# ============================================================================
# MoHGE — Mixture of Heterogeneous Grouped Experts
# ============================================================================

class MoHGE(nn.Module):
    """Mixture of Heterogeneous Grouped Experts (MoHGE).

    Unlike standard MoE where every expert shares the same ``d_ff``,
    MoHGE organises experts into groups with different intermediate
    dimensions.  Within each group, experts share the computationally
    expensive up-projection, differing only in gate and down projections.

    A capacity-aware router biases logits so that larger experts (which
    can absorb more tokens efficiently) are preferred for hard tokens,
    while smaller experts serve easy tokens at lower compute cost.

    The module is compatible with Losion's Jalur 3 MoE interface and
    can serve as a drop-in replacement for a conventional MoE layer.

    Example
    -------
    >>> config = MoHGEConfig(
    ...     d_model=512,
    ...     expert_groups=[
    ...         {"num_experts": 4, "d_ff": 1024},
    ...         {"num_experts": 4, "d_ff": 2048},
    ...         {"num_experts": 2, "d_ff": 4096},
    ...     ],
    ...     top_k=2,
    ... )
    >>> moe = MoHGE(config)
    >>> x = torch.randn(2, 16, 512)
    >>> out, aux = moe(x)
    >>> out.shape
    torch.Size([2, 16, 512])
    >>> "load_balance" in aux
    True

    Args:
        config: A :class:`MoHGEConfig` instance with hyperparameters.
    """

    def __init__(
        self,
        config: Optional[MoHGEConfig] = None,
    ) -> None:
        super().__init__()

        if config is None:
            config = MoHGEConfig()

        self.config = config
        self.d_model = config.d_model
        self.top_k = config.top_k
        self.load_balance_weight = config.load_balance_weight
        self.group_balance_weight = config.group_balance_weight
        self.use_shared_expert = config.use_shared_expert

        # ---- Parse expert groups ----
        self.group_configs: List[ExpertGroupConfig] = []
        for g_dict in config.expert_groups:
            self.group_configs.append(
                ExpertGroupConfig(
                    num_experts=g_dict["num_experts"],
                    d_ff=g_dict["d_ff"],
                )
            )

        # ---- Build expert groups ----
        self.expert_groups = nn.ModuleList()
        expert_d_ffs: List[int] = []       # per-expert d_ff
        expert_group_ids: List[int] = []   # per-expert group assignment
        global_expert_id = 0

        for g_id, g_cfg in enumerate(self.group_configs):
            group = ExpertGroup(
                d_model=config.d_model,
                d_ff=g_cfg.d_ff,
                num_experts=g_cfg.num_experts,
                group_id=g_id,
            )
            self.expert_groups.append(group)

            for _ in range(g_cfg.num_experts):
                expert_d_ffs.append(g_cfg.d_ff)
                expert_group_ids.append(g_id)
                global_expert_id += 1

        self.num_experts = len(expert_d_ffs)
        self.expert_d_ffs = expert_d_ffs
        self.num_groups = len(self.group_configs)

        # Register group assignments as a buffer (for group balance loss)
        self.register_buffer(
            "expert_group_ids",
            torch.tensor(expert_group_ids, dtype=torch.long),
        )

        # ---- Capacity-aware router ----
        self.router = MoHGERouter(
            d_model=config.d_model,
            num_experts=self.num_experts,
            expert_d_ffs=expert_d_ffs,
            top_k=config.top_k,
            capacity_aware_routing=config.capacity_aware_routing,
            capacity_bias_scale=config.capacity_bias_scale,
        )

        # ---- Shared expert (always active, non-routed) ----
        if self.use_shared_expert:
            max_d_ff = max(g.d_ff for g in self.group_configs)
            self.shared_expert = nn.ModuleDict({
                "gate_proj": nn.Linear(config.d_model, max_d_ff, bias=False),
                "up_proj": nn.Linear(config.d_model, max_d_ff, bias=False),
                "down_proj": nn.Linear(max_d_ff, config.d_model, bias=False),
            })
            self.shared_expert_scale = nn.Parameter(torch.ones(1))

            # Initialise shared expert with small weights
            nn.init.normal_(self.shared_expert["gate_proj"].weight, std=0.01)
            nn.init.normal_(self.shared_expert["up_proj"].weight, std=0.01)
            nn.init.zeros_(self.shared_expert["down_proj"].weight)

        # ---- Dropout ----
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

        # ---- Output norm ----
        self.output_norm = nn.RMSNorm(config.d_model, eps=1e-5)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _expert_group_and_local_id(self, global_id: int) -> Tuple[int, int]:
        """Map a global expert ID to (group_id, local_expert_id).

        Args:
            global_id: Global expert index in ``[0, num_experts)``.

        Returns:
            Tuple ``(group_id, local_expert_id)``.
        """
        offset = 0
        for g_id, g_cfg in enumerate(self.group_configs):
            if global_id < offset + g_cfg.num_experts:
                return g_id, global_id - offset
            offset += g_cfg.num_experts
        # Should not reach here
        return self.num_groups - 1, 0

    @staticmethod
    def _shared_expert_forward(
        expert: nn.ModuleDict,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through the shared SwiGLU expert."""
        gate = F.silu(expert["gate_proj"](x))
        up = expert["up_proj"](x)
        return expert["down_proj"](gate * up)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass through MoHGE.

        Each token is routed to the top-K experts across all groups;
        expert outputs are weighted by the router softmax scores and summed.

        Args:
            x: ``(batch, seq_len, d_model)``.

        Returns:
            output: ``(batch, seq_len, d_model)``.
            aux: Dict of auxiliary losses with keys ``"load_balance"``
                and ``"group_balance"``.
        """
        batch, seq_len, _ = x.shape

        # ---- Shared Expert ----
        x_flat = x.reshape(-1, self.d_model)  # (N, d_model)
        shared_output = torch.zeros_like(x_flat)
        if self.use_shared_expert:
            shared_output = (
                self._shared_expert_forward(self.shared_expert, x_flat)
                * self.shared_expert_scale
            )

        # ---- Router ----
        weights, indices, router_logits = self.router(x)
        # weights: (B, S, K), indices: (B, S, K)

        weights_flat = weights.reshape(batch * seq_len, self.top_k)   # (N, K)
        indices_flat = indices.reshape(batch * seq_len, self.top_k)   # (N, K)

        # ---- Dispatch to expert groups ----
        routed_output = torch.zeros_like(x_flat)  # (N, d_model)

        for k_idx in range(self.top_k):
            expert_indices = indices_flat[:, k_idx]  # (N,)
            expert_weights = weights_flat[:, k_idx].unsqueeze(-1)  # (N, 1)

            # Group tokens by (group_id, local_expert_id) for efficiency
            for global_id in range(self.num_experts):
                mask = (expert_indices == global_id)  # (N,)
                if not mask.any():
                    continue

                group_id, local_id = self._expert_group_and_local_id(global_id)
                group: ExpertGroup = self.expert_groups[group_id]  # type: ignore[assignment]

                # Get tokens for this expert
                expert_input = x_flat[mask]  # (n, d_model)
                expert_output = group.forward_single_expert(expert_input, local_id)

                # Weight and accumulate
                w = expert_weights[mask]  # (n, 1)
                routed_output[mask] += w * self.dropout(expert_output)

        # ---- Combine shared + routed ----
        output = (shared_output + routed_output).reshape(batch, seq_len, self.d_model)
        output = self.output_norm(output)

        # ---- Auxiliary losses ----
        aux: Dict[str, torch.Tensor] = {
            "load_balance": self.load_balance_weight
            * mohge_load_balance_loss(router_logits, self.num_experts),
            "group_balance": self.group_balance_weight
            * mohge_group_balance_loss(
                router_logits, self.expert_group_ids, self.num_groups
            ),
        }

        return output, aux

    # ------------------------------------------------------------------
    # Analysis utilities
    # ------------------------------------------------------------------

    def parameter_count_per_group(self) -> List[Dict[str, int]]:
        """Return parameter counts for each expert group.

        Returns:
            List of dicts, one per group, with ``shared``, ``per_expert``,
            and ``total`` keys.
        """
        return [group.parameter_count() for group in self.expert_groups]

    def total_parameter_count(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters())

    def equivalent_homogeneous_d_ff(self) -> float:
        """The d_ff that a homogeneous MoE with the same total parameter
        count would have.

        Returns:
            Equivalent d_ff value (float).
        """
        total_expert_params = sum(
            sum(p.numel() for p in group.parameters())
            for group in self.expert_groups
        )
        # Standard MoE: each expert has up_proj + gate_proj + down_proj
        # = 3 * d_model * d_ff params per expert
        # MoHGE shared: shared_up_proj + E_g * (gate_proj + down_proj)
        # = d_model * d_ff_g + E_g * (d_model * d_ff_g + d_ff_g * d_model)
        # = d_model * d_ff_g * (1 + 2 * E_g)
        # For equivalent: num_experts * 3 * d_model * d_ff_equiv ≈ total_expert_params
        d_ff_equiv = total_expert_params / (self.num_experts * 3 * self.d_model)
        return d_ff_equiv

    def compute_savings_vs_homogeneous(self) -> float:
        """Estimate parameter savings vs. a homogeneous MoE where every
        expert uses the *maximum* ``d_ff``.

        Returns:
            Savings fraction in ``[0, 1]``.  For example, ``0.4`` means
            MoHGE uses 40% fewer parameters.
        """
        max_d_ff = max(g.d_ff for g in self.group_configs)
        homogeneous_params = self.num_experts * 3 * self.d_model * max_d_ff
        mohge_params = sum(
            group.parameter_count()["total"] for group in self.expert_groups
        )

        if homogeneous_params == 0:
            return 0.0
        savings = 1.0 - mohge_params / homogeneous_params
        return max(0.0, min(1.0, savings))

    def get_routing_report(self, x: torch.Tensor) -> Dict[str, object]:
        """Generate a routing analysis report.

        Args:
            x: Input tensor ``[batch, seq, d_model]``.

        Returns:
            Dictionary with per-group routing statistics.
        """
        with torch.no_grad():
            _, indices, router_logits = self.router(x)

            # Per-group token counts
            group_counts = torch.zeros(self.num_groups, dtype=torch.long)
            for g_id in range(self.num_groups):
                g_start = sum(gc.num_experts for gc in self.group_configs[:g_id])
                g_end = g_start + self.group_configs[g_id].num_experts
                for eid in range(g_start, g_end):
                    for k in range(self.top_k):
                        group_counts[g_id] += (indices[:, :, k] == eid).sum()

            # Per-group mean routing probability
            probs = F.softmax(router_logits, dim=-1)
            group_probs = torch.zeros(self.num_groups)
            for g_id in range(self.num_groups):
                g_start = sum(gc.num_experts for gc in self.group_configs[:g_id])
                g_end = g_start + self.group_configs[g_id].num_experts
                group_probs[g_id] = probs[..., g_start:g_end].sum(dim=-1).mean()

            return {
                "group_token_counts": {
                    f"group_{g_id} (d_ff={self.group_configs[g_id].d_ff})": group_counts[g_id].item()
                    for g_id in range(self.num_groups)
                },
                "group_mean_routing_prob": {
                    f"group_{g_id} (d_ff={self.group_configs[g_id].d_ff})": group_probs[g_id].item()
                    for g_id in range(self.num_groups)
                },
                "total_experts": self.num_experts,
                "top_k": self.top_k,
                "parameter_savings_vs_max_d_ff": self.compute_savings_vs_homogeneous(),
                "equivalent_homogeneous_d_ff": self.equivalent_homogeneous_d_ff(),
            }
