"""
S'MoRE — Sub-tree MoE with Residual Experts (Meta, NeurIPS 2025).

Adapted from "S'MoRE: Sub-tree MoE with Residual Experts" (Meta AI, NeurIPS 2025):
instead of maintaining N independent expert FFNs, S'MoRE composes each expert
from a shared pool of residual sub-trees.  Multiple experts can reuse the same
sub-trees with different composition weights, yielding **parameter-efficient
expert diversity** — the key insight is that experts can differ not by owning
unique parameters but by *how they blend shared sub-trees*.

Architecture
------------
1. **ResidualSubTree** — A shared FFN sub-tree with SwiGLU layers and residual
   connections.  Multiple ``ComposedExpert`` instances reference the *same*
   sub-tree objects, so parameters are shared across experts.

2. **ComposedExpert** — An expert that *softly* combines the outputs of several
   shared sub-trees via learned routing weights, plus a small expert-specific
   residual branch for unique capacity.

3. **SmoreMoE** — The full MoE layer.  Tokens are routed to the top-K
   ``ComposedExpert`` instances via a standard gating network.  An optional
   shared expert (always active) provides baseline capacity.  Load balancing
   is enforced via a Switch-Transformer-style auxiliary loss.

Parameter Savings
-----------------
In a standard MoE with *E* experts, each with dimension *d_ff*, the total
expert parameter count is roughly ``E × 3 × d_model × d_ff`` (SwiGLU).

With S'MoRE using *S* sub-trees of depth *D* and residual dimension *d_r*:
  - Sub-tree params:  ``S × D × 3 × d_model × d_r``
  - Per-expert composition weights:  ``S`` scalars (negligible)
  - Per-expert residual branch:  ``3 × d_model × d_r``

For typical configs (E=8, S=4, D=2, d_r=256, d_ff=3072), savings exceed
60% compared to independent experts while maintaining competitive quality.

References
----------
- Meta AI, "S'MoRE: Sub-tree MoE with Residual Experts" (NeurIPS 2025).
- DeepSeek-AI, "DeepSeekMoE: Towards Ultimate Expert Specialization in
  Mixture-of-Experts Language Models" (arXiv:2401.06066, 2024).
- Jiang et al., "Mixtral of Experts" (2024).
- Fedus et al., "Switch Transformers: Scaling to Trillion Parameter Models"
  (2022) — load-balancing loss.

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
class SmoreConfig:
    """Configuration for S'MoRE (Sub-tree MoE with Residual Experts).

    Attributes:
        num_experts: Total number of composed experts.
        num_active_experts: Number of experts to activate per token (top-K).
        d_model: Model hidden dimension.
        d_ff: Feed-forward intermediate dimension (reference size).
        num_sub_trees: Number of shared residual sub-trees.
        sub_tree_depth: Depth (number of SwiGLU layers) per sub-tree.
        residual_dim: Intermediate dimension within each sub-tree layer
            and the expert-specific residual branch.
    """

    num_experts: int = 8
    num_active_experts: int = 2
    d_model: int = 768
    d_ff: int = 3072
    num_sub_trees: int = 4
    sub_tree_depth: int = 2
    residual_dim: int = 256


# ============================================================================
# Routing Info
# ============================================================================

@dataclass
class SmoreRoutingInfo:
    """Routing information returned by SmoreMoE for monitoring.

    Attributes:
        routing_weights: [batch, seq, num_experts] — full softmax weights.
        top_k_indices: [batch, seq, num_active_experts] — selected expert ids.
        top_k_weights: [batch, seq, num_active_experts] — renormalised weights.
        expert_loads: [num_experts] — token count per expert.
        sub_tree_usage: [num_sub_trees] — mean usage weight across all experts.
        parameter_savings: Estimated parameter savings fraction vs. standard MoE.
    """

    routing_weights: torch.Tensor
    top_k_indices: torch.Tensor
    top_k_weights: torch.Tensor
    expert_loads: torch.Tensor
    sub_tree_usage: torch.Tensor
    parameter_savings: float


# ============================================================================
# ResidualSubTree — Shared sub-tree component
# ============================================================================

class ResidualSubTree(nn.Module):
    """A shared residual sub-tree that multiple experts can reference.

    Each sub-tree is a stack of SwiGLU FFN layers with residual connections.
    The intermediate dimension is ``residual_dim`` (typically much smaller
    than ``d_ff``), making each sub-tree lightweight.  Because sub-trees are
    *shared* across experts, the total parameter count grows with the number
    of sub-trees — not with the number of experts.

    Architecture per layer::

        h = x + SwiGLU(LayerNorm(x))
          = x + down_proj(SiLU(gate_proj(norm(x))) * up_proj(norm(x)))

    Args:
        d_model: Input/output dimension.
        residual_dim: Intermediate dimension for each SwiGLU layer.
        depth: Number of SwiGLU layers in this sub-tree.
    """

    def __init__(
        self,
        d_model: int,
        residual_dim: int,
        depth: int = 2,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.residual_dim = residual_dim
        self.depth = depth

        # Stack of SwiGLU layers with their own layer norms
        self.layers = nn.ModuleList([
            self._make_swiglu_layer(d_model, residual_dim)
            for _ in range(depth)
        ])

        # Output projection to match d_model (if residual_dim != d_model)
        self.output_norm = nn.RMSNorm(d_model, eps=1e-5)

    @staticmethod
    def _make_swiglu_layer(d_model: int, residual_dim: int) -> nn.ModuleDict:
        """Create a single SwiGLU layer with LayerNorm."""
        return nn.ModuleDict({
            "norm": nn.RMSNorm(d_model, eps=1e-5),
            "gate_proj": nn.Linear(d_model, residual_dim, bias=False),
            "up_proj": nn.Linear(d_model, residual_dim, bias=False),
            "down_proj": nn.Linear(residual_dim, d_model, bias=False),
        })

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the sub-tree.

        Args:
            x: Input tensor ``(..., d_model)``.

        Returns:
            Output tensor of the same shape as input.
        """
        h = x
        for layer in self.layers:
            # Pre-norm SwiGLU with residual
            normed = layer["norm"](h)
            gate = F.silu(layer["gate_proj"](normed))
            up = layer["up_proj"](normed)
            h = h + layer["down_proj"](gate * up)
        return self.output_norm(h)


# ============================================================================
# ComposedExpert — Expert built from shared sub-trees
# ============================================================================

class ComposedExpert(nn.Module):
    """An expert composed from multiple shared sub-trees.

    Rather than owning unique FFN parameters, a ``ComposedExpert`` holds:
      1. **Composition weights** — learned scalars that softly blend the
         outputs of the shared sub-trees.  This is how experts differ:
         each expert weighs sub-trees differently.
      2. **Expert-specific residual branch** — a small SwiGLU FFN
         (``d_model → residual_dim → d_model``) that adds unique capacity
         beyond what the shared sub-trees provide.

    The forward pass is::

        sub_tree_outputs = [tree_i(x) for tree_i in referenced_trees]
        composed = Σ_i  softmax(comp_weights)_i * sub_tree_outputs_i
        residual = swiglu_residual(x)
        output = composed + residual_scale * residual

    Args:
        d_model: Input/output dimension.
        residual_dim: Intermediate dimension for the expert-specific branch.
        sub_trees: List of ``ResidualSubTree`` modules to compose.
            These are *shared* — the same tree object may appear in
            multiple experts.
    """

    def __init__(
        self,
        d_model: int,
        residual_dim: int,
        sub_trees: List[ResidualSubTree],
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.residual_dim = residual_dim
        self.num_sub_trees = len(sub_trees)

        # Store references to shared sub-trees (NOT owned parameters)
        self.sub_trees = nn.ModuleList(sub_trees)

        # Composition weights over sub-trees (learned, initialised uniformly)
        self.comp_weights = nn.Parameter(
            torch.zeros(self.num_sub_trees)
        )

        # Expert-specific residual branch (unique capacity)
        self.residual_gate_proj = nn.Linear(d_model, residual_dim, bias=False)
        self.residual_up_proj = nn.Linear(d_model, residual_dim, bias=False)
        self.residual_down_proj = nn.Linear(residual_dim, d_model, bias=False)
        self.residual_norm = nn.RMSNorm(d_model, eps=1e-5)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

        # Initialise residual branch with small weights
        nn.init.normal_(self.residual_gate_proj.weight, std=0.01)
        nn.init.normal_(self.residual_up_proj.weight, std=0.01)
        nn.init.zeros_(self.residual_down_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: compose sub-trees + expert-specific residual.

        Args:
            x: Input tensor ``(..., d_model)``.

        Returns:
            Output tensor of the same shape.
        """
        # Soft composition weights over sub-trees
        alpha = F.softmax(self.comp_weights, dim=0)  # (num_sub_trees,)

        # Weighted sum of sub-tree outputs
        composed = torch.zeros_like(x)
        for i, tree in enumerate(self.sub_trees):
            composed = composed + alpha[i] * tree(x)

        # Expert-specific residual branch
        normed = self.residual_norm(x)
        gate = F.silu(self.residual_gate_proj(normed))
        up = self.residual_up_proj(normed)
        residual = self.residual_down_proj(gate * up)

        return composed + self.residual_scale * residual

    def get_sub_tree_weights(self) -> torch.Tensor:
        """Return the current soft composition weights over sub-trees.

        Returns:
            Tensor of shape ``(num_sub_trees,)`` summing to 1.
        """
        return F.softmax(self.comp_weights, dim=0)


# ============================================================================
# Load-balancing auxiliary loss (Switch Transformer style)
# ============================================================================

def smore_load_balance_loss(
    router_logits: torch.Tensor,
    num_experts: int,
    top_k: int,
) -> torch.Tensor:
    """Switch-Transformer-style load-balancing loss.

    Encourages each expert to receive a roughly equal fraction of the
    total routing probability mass, preventing routing collapse.

    .. math::
        L_{balance} = N \\cdot \\sum_{i=1}^{N} f_i \\cdot P_i

    Args:
        router_logits: ``(batch, seq_len, num_experts)`` — raw router logits.
        num_experts: Total number of experts.
        top_k: Number of active experts per token.

    Returns:
        Scalar auxiliary loss.
    """
    probs = F.softmax(router_logits, dim=-1)  # (B, S, E)

    # Fraction of tokens assigned to each expert (via argmax)
    assignments = probs.argmax(dim=-1)  # (B, S)
    one_hot = F.one_hot(assignments, num_experts).float()  # (B, S, E)
    f = one_hot.mean(dim=(0, 1))  # (E,)

    # Mean routing probability per expert
    P = probs.mean(dim=(0, 1))  # (E,)

    return num_experts * (f * P).sum()


# ============================================================================
# SmoreMoE — Full MoE layer with S'MoRE expert composition
# ============================================================================

class SmoreMoE(nn.Module):
    """Mixture-of-Experts using S'MoRE (Sub-tree MoE with Residual Experts).

    Each composed expert is built from shared sub-trees rather than
    independent FFN parameters, yielding significant parameter savings
    while maintaining expert diversity through composition weights.

    The module is compatible with the standard MoE interface used in
    Losion's Jalur 3 and can serve as a drop-in replacement for a
    conventional MoE layer.

    Example
    -------
    >>> config = SmoreConfig(num_experts=8, num_active_experts=2)
    >>> moe = SmoreMoE(config)
    >>> x = torch.randn(2, 16, 768)
    >>> output, aux_loss, info = moe(x)
    >>> output.shape
    torch.Size([2, 16, 768])

    Args:
        config: A :class:`SmoreConfig` instance controlling architecture
            hyperparameters.  All fields have sensible defaults.
        use_shared_expert: If ``True``, add a shared (always-active) expert
            similar to DeepSeek-V3 (default ``True``).
        load_balance_weight: Weight for the auxiliary load-balancing loss
            (default ``0.01``).
    """

    def __init__(
        self,
        config: Optional[SmoreConfig] = None,
        use_shared_expert: bool = True,
        load_balance_weight: float = 0.01,
    ) -> None:
        super().__init__()

        if config is None:
            config = SmoreConfig()

        self.config = config
        self.num_experts = config.num_experts
        self.num_active_experts = min(config.num_active_experts, config.num_experts)
        self.d_model = config.d_model
        self.d_ff = config.d_ff
        self.num_sub_trees = config.num_sub_trees
        self.use_shared_expert = use_shared_expert
        self.load_balance_weight = load_balance_weight

        # ---- Shared Sub-Trees ----
        self.sub_trees = nn.ModuleList([
            ResidualSubTree(
                d_model=config.d_model,
                residual_dim=config.residual_dim,
                depth=config.sub_tree_depth,
            )
            for _ in range(config.num_sub_trees)
        ])

        # ---- Composed Experts ----
        self.experts = nn.ModuleList([
            self._build_composed_expert(config)
            for _ in range(config.num_experts)
        ])

        # ---- Shared Expert (always active) ----
        if use_shared_expert:
            self.shared_expert = self._make_swiglu_expert(config.d_model, config.d_ff)
            self.shared_expert_scale = nn.Parameter(torch.ones(1))

        # ---- Gating Network (token → expert) ----
        self.gate_proj = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2, bias=False),
            nn.SiLU(),
            nn.Linear(config.d_model // 2, config.num_experts, bias=False),
        )

        # ---- Output Norm ----
        self.output_norm = nn.RMSNorm(config.d_model, eps=1e-5)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_composed_expert(self, config: SmoreConfig) -> ComposedExpert:
        """Build a ComposedExpert that references all shared sub-trees.

        Each expert references *all* sub-trees but with different
        composition weights, ensuring diversity.
        """
        return ComposedExpert(
            d_model=config.d_model,
            residual_dim=config.residual_dim,
            sub_trees=list(self.sub_trees),  # reference all sub-trees
        )

    @staticmethod
    def _make_swiglu_expert(d_model: int, d_ff: int) -> nn.ModuleDict:
        """Create a standard SwiGLU FFN expert."""
        return nn.ModuleDict({
            "gate_proj": nn.Linear(d_model, d_ff, bias=False),
            "up_proj": nn.Linear(d_model, d_ff, bias=False),
            "down_proj": nn.Linear(d_ff, d_model, bias=False),
        })

    @staticmethod
    def _expert_forward(expert: nn.ModuleDict, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through a SwiGLU expert."""
        gate = F.silu(expert["gate_proj"](x))
        up = expert["up_proj"](x)
        return expert["down_proj"](gate * up)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, SmoreRoutingInfo]:
        """Forward pass through S'MoRE.

        Args:
            x: Input tensor ``[batch, seq_len, d_model]``.

        Returns:
            output: ``[batch, seq_len, d_model]``.
            aux_loss: Scalar auxiliary load-balancing loss.
            routing_info: :class:`SmoreRoutingInfo` for monitoring.
        """
        if x.dim() != 3:
            raise ValueError(
                f"Input must be 3D [batch, seq, d_model], got {x.dim()}D"
            )

        batch_size, seq_len, _ = x.shape
        orig_shape = x.shape

        # ---- Shared Expert ----
        x_flat = x.reshape(-1, self.d_model)  # (N, d_model)
        shared_output = torch.zeros_like(x_flat)
        if self.use_shared_expert:
            shared_output = (
                self._expert_forward(self.shared_expert, x_flat)
                * self.shared_expert_scale
            )

        # ---- Routing ----
        router_logits = self.gate_proj(x)  # (B, S, E)
        routing_weights = F.softmax(router_logits, dim=-1)  # (B, S, E)

        top_k_weights, top_k_indices = torch.topk(
            routing_weights, self.num_active_experts, dim=-1
        )  # (B, S, K)
        # Renormalise top-K weights
        top_k_weights = top_k_weights / (
            top_k_weights.sum(dim=-1, keepdim=True) + 1e-8
        )

        # Flatten for expert dispatch
        top_k_weights_flat = top_k_weights.reshape(-1, self.num_active_experts)
        top_k_indices_flat = top_k_indices.reshape(-1, self.num_active_experts)

        # ---- Dispatch tokens to composed experts ----
        routed_output = torch.zeros_like(x_flat)

        for k_idx in range(self.num_active_experts):
            expert_indices = top_k_indices_flat[:, k_idx]  # (N,)
            weights = top_k_weights_flat[:, k_idx]  # (N,)

            for expert_id in range(self.num_experts):
                mask = (expert_indices == expert_id)
                if not mask.any():
                    continue

                expert_input = x_flat[mask]  # (n, d_model)
                expert_output = self.experts[expert_id](expert_input)  # (n, d_model)
                expert_weights = weights[mask].unsqueeze(-1)  # (n, 1)
                routed_output[mask] += expert_weights * expert_output

        # ---- Combine ----
        output = (shared_output + routed_output).reshape(orig_shape)
        output = self.output_norm(output)

        # ---- Auxiliary Load-Balance Loss ----
        aux_loss = self.load_balance_weight * smore_load_balance_loss(
            router_logits, self.num_experts, self.num_active_experts
        )

        # ---- Expert loads ----
        expert_loads = torch.zeros(
            self.num_experts, dtype=torch.long, device=x.device
        )
        for k_idx in range(self.num_active_experts):
            for e in range(self.num_experts):
                expert_loads[e] += (
                    top_k_indices[:, :, k_idx] == e
                ).sum()

        # ---- Sub-tree usage ----
        sub_tree_usage = torch.zeros(self.num_sub_trees, device=x.device)
        with torch.no_grad():
            for expert in self.experts:
                w = expert.get_sub_tree_weights()  # (S,)
                # Weight by how often this expert is selected
                expert_frac = expert_loads[expert.get_sub_tree_weights().argmax()].float()
                sub_tree_usage += w * expert_frac
            # Normalise
            total = sub_tree_usage.sum().clamp(min=1e-8)
            sub_tree_usage = sub_tree_usage / total

        # ---- Routing info ----
        routing_info = SmoreRoutingInfo(
            routing_weights=routing_weights,
            top_k_indices=top_k_indices,
            top_k_weights=top_k_weights,
            expert_loads=expert_loads,
            sub_tree_usage=sub_tree_usage,
            parameter_savings=self.estimate_parameter_savings(),
        )

        return output, aux_loss, routing_info

    # ------------------------------------------------------------------
    # Parameter savings estimation
    # ------------------------------------------------------------------

    def estimate_parameter_savings(self) -> float:
        """Estimate parameter savings vs. a standard MoE with independent experts.

        Compares the total parameter count of the S'MoRE architecture
        (shared sub-trees + composition weights + expert residuals) against
        a standard MoE where each expert has its own SwiGLU FFN with
        dimension ``d_ff``.

        Returns:
            Savings fraction in ``[0, 1]``.  For example, ``0.65`` means
            S'MoRE uses 65% fewer parameters than a standard MoE.
        """
        cfg = self.config

        # Standard MoE: E experts × SwiGLU (3 × d_model × d_ff)
        standard_params = cfg.num_experts * 3 * cfg.d_model * cfg.d_ff

        # S'MoRE params:
        # 1. Sub-trees: S × depth × 3 × d_model × residual_dim
        sub_tree_params = (
            cfg.num_sub_trees
            * cfg.sub_tree_depth
            * 3
            * cfg.d_model
            * cfg.residual_dim
        )
        # 2. Per-expert composition weights: E × S (negligible)
        comp_params = cfg.num_experts * cfg.num_sub_trees
        # 3. Per-expert residual branch: E × 3 × d_model × residual_dim
        residual_params = cfg.num_experts * 3 * cfg.d_model * cfg.residual_dim

        smore_params = sub_tree_params + comp_params + residual_params

        if standard_params == 0:
            return 0.0
        savings = 1.0 - smore_params / standard_params
        return max(0.0, min(1.0, savings))

    def get_expert_specialization(
        self, x: torch.Tensor, top_k: int = 5
    ) -> Dict[int, List[int]]:
        """Analyse expert specialisation.

        Returns the token indices most frequently routed to each expert.

        Args:
            x: Input tensor ``[batch, seq, d_model]``.
            top_k: Number of top tokens to report per expert.

        Returns:
            Dictionary mapping expert id → list of top token indices.
        """
        with torch.no_grad():
            router_logits = self.gate_proj(x)
            routing_weights = F.softmax(router_logits, dim=-1)
            _, top_k_indices = torch.topk(
                routing_weights, self.num_active_experts, dim=-1
            )

            specialization: Dict[int, List[int]] = {}
            for e in range(self.num_experts):
                token_indices: List[int] = []
                for k in range(self.num_active_experts):
                    mask = (top_k_indices[:, :, k] == e)
                    indices = mask.nonzero(as_tuple=True)
                    for b, s in zip(*indices):
                        token_indices.append(b.item() * x.shape[1] + s.item())

                from collections import Counter
                counts = Counter(token_indices)
                specialization[e] = [
                    idx for idx, _ in counts.most_common(top_k)
                ]

            return specialization

    def get_sub_tree_report(self) -> Dict[str, object]:
        """Report on sub-tree usage across experts.

        Returns:
            Dictionary with per-expert composition weights and
            aggregate sub-tree usage statistics.
        """
        with torch.no_grad():
            expert_weights = {}
            for i, expert in enumerate(self.experts):
                w = expert.get_sub_tree_weights()
                expert_weights[f"expert_{i}"] = {
                    f"sub_tree_{j}": w[j].item() for j in range(self.num_sub_trees)
                }

            # Aggregate: mean weight per sub-tree across experts
            all_weights = torch.stack([
                expert.get_sub_tree_weights() for expert in self.experts
            ])  # (E, S)
            mean_weights = all_weights.mean(dim=0)  # (S,)
            aggregate = {
                f"sub_tree_{j}": mean_weights[j].item()
                for j in range(self.num_sub_trees)
            }

            return {
                "per_expert_composition": expert_weights,
                "aggregate_sub_tree_usage": aggregate,
                "parameter_savings": self.estimate_parameter_savings(),
                "num_shared_sub_trees": self.num_sub_trees,
                "num_composed_experts": self.num_experts,
            }
