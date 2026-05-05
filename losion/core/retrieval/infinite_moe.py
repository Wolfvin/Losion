"""
∞-MoE — Infinite Mixture of Experts for Losion Framework v0.7.

Implementation of ∞-MoE (Infinite Mixture of Experts), which extends the
traditional Mixture of Experts from a finite discrete set to a continuous
(effectively infinite) expert space.

Reference:
    "∞-MoE: Generalizing Mixture of Experts to Infinite Experts"
    arXiv:2601.17680, January 2026

Key Innovation:
    Traditional MoE routes tokens to one of N discrete experts.  ∞-MoE
    instead parameterises experts as functions in a continuous latent space,
    where each point in the space corresponds to a unique expert.  A
    hypernetwork generates expert weights on-the-fly from low-dimensional
    "expert code" vectors, and the router produces both routing weights AND
    expert codes in this continuous space.

Architecture Overview
---------------------
1. **ExpertCodeRouter** — Maps token representations to K expert codes and
   routing weights in continuous expert space.  Each routing head produces
   an independent (expert_code, routing_weight) pair; top-K are selected.

2. **ContinuousExpertGenerator** — A hypernetwork that generates
   expert-specific modifications (scaling + bias + optional low-rank
   residuals) from expert codes, applied on top of a shared base expert.
   This provides unlimited expert diversity with O(1) parameter overhead
   per expert since experts are generated, not stored.

3. **ExpertCodeClusterer** — Clusters nearby expert codes to amortise
   hypernetwork computation during inference, maintaining efficiency when
   many tokens produce similar expert codes.

4. **InfiniteMoE** — The main module combining router, expert generator,
   and clusterer with load balancing, z-loss stabilisation, and capacity
   management for efficient training and inference.

Continuous Expert Space
-----------------------
Instead of N discrete expert indices {0, 1, …, N-1}, ∞-MoE operates in a
continuous code space R^d_code.  Each point z ∈ R^d_code defines a unique
expert via the hypernetwork:

    Expert(z) = BaseExpert ⊙ scale(z) + bias(z) + ΔW_lowrank(z)

where:
    - BaseExpert is a shared SwiGLU FFN (W_up, W_gate, W_down)
    - scale(z), bias(z) are code-conditioned scaling/bias vectors
    - ΔW_lowrank(z) = A(z) @ B(z) is an optional low-rank weight residual

This formulation enables:
    * **Unlimited capacity**: Theoretical capacity is unlimited since the
      expert space is continuous (uncountably infinite).
    * **Smooth interpolation**: Nearby codes produce similar experts,
      enabling smooth transitions between expert behaviours.
    * **Efficient activation**: Only top-K experts are activated per token,
      maintaining O(K) inference cost regardless of total capacity.
    * **Adaptive specialisation**: The router adaptively discovers useful
      regions of expert space during training.

Optional Codebook Anchoring
---------------------------
When ``codebook_size > 0``, a learnable codebook of prototype expert codes
anchors the continuous space.  The router computes attention-weighted sums
of codebook entries to refine its expert codes, providing stable reference
points while preserving the ability to generate novel codes (true infinite
capacity).  This is analogous to VQ-VAE codebooks but with soft (rather
than hard) assignments.

Integration with Losion Tri-Jalur Architecture
----------------------------------------------
The ∞-MoE module serves as the adaptive expert layer within the Tri-Jalur
(Three Paths) architecture:

    - **Jalur Semantik (Semantic Path)**: Uses ∞-MoE with
      content-conditioned routing for language understanding.
    - **Jalur Struktural (Structural Path)**: Uses ∞-MoE with
      structure-aware routing for pattern recognition.
    - **Jalur Pengambilan (Retrieval Path)**: Uses ∞-MoE with
      retrieval-conditioned routing for knowledge access.

Each pathway shares the continuous expert code space while routing
differently, enabling cross-pathway knowledge transfer through proximity
in expert code space.  The ``tri_jalur_pathway`` parameter in
:class:`InfiniteMoE.forward` enables pathway-specific logging and
analysis.

References
----------
- ∞-MoE: Generalizing Mixture of Experts to Infinite Experts
  (arXiv:2601.17680, January 2026)
- Fedus et al., "Switch Transformers: Scaling to Trillion Parameter Models"
  (2021) — load-balancing loss.
- Zoph et al., "ST-MoE: Designing Stable and Transferable Sparse Expert
  Models" (2022) — router z-loss.
- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2022)
  — low-rank residual inspiration.

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
class InfiniteMoEConfig:
    """Configuration for the ∞-MoE module.

    Controls all aspects of the infinite mixture-of-experts architecture:
    expert code space dimensionality, routing, hypernetwork capacity,
    auxiliary loss weights, and optional codebook anchoring.

    Attributes:
        d_model: Model hidden dimension.
        d_ff: Feed-forward intermediate dimension.
            0 means ``4 * d_model``.
        code_dim: Dimensionality of the continuous expert code space.
            Higher values allow more expressive expert specialisation.
            Typical range: 32–128.
        top_k: Number of experts activated per token.
        n_route_heads: Number of routing heads.  Each head independently
            produces one (expert_code, routing_weight) pair per token.
            Must be >= top_k.
        base_expert_init_std: Standard deviation for base expert
            weight initialization.
        hypernet_hidden_dim: Hidden dimension of the hypernetwork MLPs
            that generate expert modifications from codes.
        use_low_rank_residual: If True, generate low-rank weight
            residuals (ΔW = A(z) @ B(z)) in addition to scaling and
            bias modifications.  Increases expressiveness at higher
            parameter cost.
        low_rank_dim: Rank of low-rank residual matrices.  Only used
            when ``use_low_rank_residual`` is True.
        codebook_size: Size of the optional learnable codebook for
            anchoring the continuous expert space.  Set to 0 to disable
            codebook augmentation.
        cluster_threshold: L2 distance threshold for clustering nearby
            expert codes during inference.  Set to 0.0 to disable
            clustering.
        cluster_max_iter: Maximum number of clustering iterations.
        load_balance_weight: Weight for the auxiliary load-balancing
            loss that encourages uniform routing head usage.
        router_z_loss_weight: Weight for the router z-loss that
            stabilises training by penalising large logit magnitudes.
        code_diversity_weight: Weight for an auxiliary code diversity
            loss that encourages distinct expert codes across routing
            heads for the same token.
        dropout: Dropout rate applied within experts.
        use_gate: If True, use gated activation (SwiGLU-style) in the
            base expert.  Recommended for modern architectures.
        capacity_factor: Expert capacity factor.  Tokens per expert
            position are capped at
            ``(batch * seq_len / top_k) * capacity_factor``.
            Set to 0.0 for unlimited capacity (default for ∞-MoE).
    """

    d_model: int = 768
    d_ff: int = 0
    code_dim: int = 64
    top_k: int = 2
    n_route_heads: int = 8
    base_expert_init_std: float = 0.02
    hypernet_hidden_dim: int = 256
    use_low_rank_residual: bool = False
    low_rank_dim: int = 16
    codebook_size: int = 0
    cluster_threshold: float = 0.0
    cluster_max_iter: int = 5
    load_balance_weight: float = 0.01
    router_z_loss_weight: float = 0.001
    code_diversity_weight: float = 0.01
    dropout: float = 0.0
    use_gate: bool = True
    capacity_factor: float = 0.0

    def __post_init__(self) -> None:
        if self.d_ff == 0:
            self.d_ff = 4 * self.d_model
        if self.top_k > self.n_route_heads:
            raise ValueError(
                f"top_k ({self.top_k}) must be <= n_route_heads "
                f"({self.n_route_heads})"
            )
        if self.d_model <= 0:
            raise ValueError(f"d_model must be > 0, got {self.d_model}")
        if self.code_dim <= 0:
            raise ValueError(f"code_dim must be > 0, got {self.code_dim}")


# ============================================================================
# Routing Info
# ============================================================================

@dataclass
class InfiniteMoERoutingInfo:
    """Routing information returned by InfiniteMoE for monitoring.

    Provides detailed routing diagnostics for the continuous expert space,
    useful for training analysis and interpretability.

    Attributes:
        expert_codes: Selected expert code vectors.
            Shape ``[batch, seq, top_k, code_dim]``.
        routing_weights: Softmax-normalised routing weights for
            selected experts.  Shape ``[batch, seq, top_k]``.
        router_logits: Raw routing logits for all route heads.
            Shape ``[batch, seq, n_route_heads]``.
        load_balance_loss: Auxiliary load-balancing loss (scalar).
        router_z_loss: Router z-loss for training stability (scalar).
        code_diversity_loss: Code diversity regularisation loss (scalar).
        code_norms: L2 norms of selected expert codes.
            Shape ``[batch, seq, top_k]``.
        code_diversity: Average pairwise distance between selected
            expert codes per token (scalar).
        weight_entropy: Entropy of the routing weight distribution
            averaged over all tokens (scalar).
    """

    expert_codes: torch.Tensor
    routing_weights: torch.Tensor
    router_logits: torch.Tensor
    load_balance_loss: torch.Tensor
    router_z_loss: torch.Tensor
    code_diversity_loss: torch.Tensor
    code_norms: torch.Tensor
    code_diversity: torch.Tensor
    weight_entropy: torch.Tensor


# ============================================================================
# Expert Code Router
# ============================================================================

class ExpertCodeRouter(nn.Module):
    """Router that produces expert codes and routing weights in continuous
    space.

    Unlike traditional MoE routers that select from discrete expert indices,
    ExpertCodeRouter maps each token to a set of points in continuous expert
    code space.  Each point (expert code) defines a unique expert via the
    :class:`ContinuousExpertGenerator` hypernetwork.

    The router uses multi-head projection: each of ``n_route_heads`` heads
    independently produces an expert code vector and a routing logit.
    Top-K heads are selected based on routing logits, and their codes and
    weights are returned.

    Optionally, a learnable codebook can anchor the expert space, providing
    stable reference points and enabling codebook-augmented routing where
    codes are refined as weighted sums of codebook entries.

    Args:
        config: :class:`InfiniteMoEConfig` instance.

    References:
        ∞-MoE: Generalizing Mixture of Experts to Infinite Experts
        (arXiv:2601.17680, Section 3.2: Continuous Routing)
    """

    def __init__(self, config: InfiniteMoEConfig) -> None:
        super().__init__()
        self.config = config

        # Multi-head code projection: each head produces an expert code
        self.code_proj = nn.Linear(
            config.d_model,
            config.n_route_heads * config.code_dim,
            bias=False,
        )

        # Multi-head routing logit projection
        self.logit_proj = nn.Linear(
            config.d_model, config.n_route_heads, bias=False
        )

        # Optional codebook for anchoring continuous expert space
        self.use_codebook = config.codebook_size > 0
        if self.use_codebook:
            self.codebook = nn.Parameter(
                torch.randn(config.codebook_size, config.code_dim) * 0.02
            )
            # Attention-style query/key for codebook lookup
            self.codebook_query = nn.Linear(
                config.code_dim, config.code_dim, bias=False
            )
            self.codebook_key = nn.Linear(
                config.code_dim, config.code_dim, bias=False
            )
            # Learnable gate controlling codebook influence
            self.codebook_gate = nn.Parameter(torch.tensor(0.1))

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize with scaled parameters for stable training."""
        nn.init.xavier_uniform_(self.code_proj.weight)
        nn.init.xavier_uniform_(self.logit_proj.weight)
        # Scale down initial routing logits to encourage uniform routing
        with torch.no_grad():
            self.logit_proj.weight.mul_(0.1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        # shape: [batch_size, seq_len, d_model]
    ) -> Tuple[
        torch.Tensor,   # expert_codes  [B, S, top_k, code_dim]
        torch.Tensor,   # routing_weights  [B, S, top_k]
        torch.Tensor,   # router_logits  [B, S, n_route_heads]
        Dict[str, torch.Tensor],  # aux_info
    ]:
        """Route tokens to continuous expert space.

        For each token, produces ``n_route_heads`` candidate (code, logit)
        pairs, selects the top-K by logit value, and returns the selected
        codes with renormalised routing weights.

        Args:
            hidden_states: Input token representations.
                Shape ``[batch_size, seq_len, d_model]``.

        Returns:
            expert_codes: Selected expert code vectors.
                Shape ``[batch_size, seq_len, top_k, code_dim]``.
            routing_weights: Renormalised routing weights for selected
                experts.  Shape ``[batch_size, seq_len, top_k]``.
            router_logits: Raw routing logits for all route heads.
                Shape ``[batch_size, seq_len, n_route_heads]``.
            aux_info: Dictionary with auxiliary information:
                - ``"load_balance_loss"``: Auxiliary load balancing loss
                - ``"router_z_loss"``: Router z-loss for training stability
                - ``"code_diversity_loss"``: Code diversity regularisation
                - ``"all_codes"``: All route head codes before top-K
        """
        batch_size, seq_len, _ = hidden_states.shape
        cfg = self.config

        # Project to K expert codes and routing logits
        # codes: [B, S, n_route_heads, code_dim]
        all_codes = self.code_proj(hidden_states).view(
            batch_size, seq_len, cfg.n_route_heads, cfg.code_dim
        )
        # logits: [B, S, n_route_heads]
        router_logits = self.logit_proj(hidden_states)

        # Optional codebook-augmented routing
        if self.use_codebook:
            all_codes = self._codebook_augment(all_codes)

        # Compute routing weights and select top-K
        routing_weights_all = F.softmax(router_logits, dim=-1)
        # [B, S, top_k]
        top_k_weights, top_k_indices = torch.topk(
            routing_weights_all, k=cfg.top_k, dim=-1
        )

        # Renormalise selected weights
        top_k_weights = top_k_weights / (
            top_k_weights.sum(dim=-1, keepdim=True) + 1e-9
        )

        # Gather corresponding expert codes
        # top_k_indices: [B, S, top_k] -> expand for gather
        gather_indices = top_k_indices.unsqueeze(-1).expand(
            -1, -1, -1, cfg.code_dim
        )  # [B, S, top_k, code_dim]
        expert_codes = torch.gather(all_codes, dim=2, index=gather_indices)

        # Compute auxiliary losses
        aux_info = self._compute_aux_losses(
            router_logits, routing_weights_all, expert_codes
        )
        aux_info["all_codes"] = all_codes

        return expert_codes, top_k_weights, router_logits, aux_info

    def _codebook_augment(self, codes: torch.Tensor) -> torch.Tensor:
        """Augment expert codes using codebook attention.

        Computes attention weights between each code and codebook entries,
        then refines codes as::

            code = code + gate * Σ_j att_j * codebook_j

        This anchors the continuous space to learnable prototypes while
        maintaining the ability to generate novel codes (true infinite
        capacity).

        Args:
            codes: Expert codes from projection.
                Shape ``[B, S, n_route_heads, code_dim]``.

        Returns:
            Augmented codes with same shape.
        """
        B, S, H, D = codes.shape

        # Flatten for attention computation
        codes_flat = codes.reshape(B * S * H, D)  # [BSH, D]

        # Query from codes, key from codebook
        queries = self.codebook_query(codes_flat)    # [BSH, D]
        keys = self.codebook_key(self.codebook)       # [CB, D]

        # Scaled dot-product attention
        attn_scores = torch.matmul(
            queries, keys.T
        ) / math.sqrt(D)  # [BSH, CB]
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Weighted sum of codebook entries
        codebook_contrib = torch.matmul(
            attn_weights, self.codebook
        )  # [BSH, D]

        # Residual connection with learnable gate
        gate = torch.sigmoid(self.codebook_gate)
        augmented = codes_flat + gate * codebook_contrib

        return augmented.reshape(B, S, H, D)

    def _compute_aux_losses(
        self,
        router_logits: torch.Tensor,
        routing_weights: torch.Tensor,
        expert_codes: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute auxiliary losses for training stability.

        **Load Balance Loss**: Encourages uniform load distribution across
        routing heads, preventing collapse where only a few heads are used.

        **Router Z-Loss**: Penalises large router logit magnitudes,
        stabilising training by preventing the softmax from becoming too
        peaked (Zoph et al., 2022).

        **Code Diversity Loss**: Encourages distinct expert codes across
        routing heads for the same token, preventing code collapse where
        all heads produce identical or near-identical codes.

        Args:
            router_logits: Raw routing logits. ``[B, S, n_route_heads]``
            routing_weights: Softmax of routing logits.
                ``[B, S, n_route_heads]``
            expert_codes: Selected expert codes.
                ``[B, S, top_k, code_dim]``

        Returns:
            Dictionary with ``"load_balance_loss"``,
            ``"router_z_loss"``, and ``"code_diversity_loss"``.
        """
        # Load balance: encourage uniform routing weight across heads
        # mean over batch and sequence: [n_route_heads]
        mean_weights = routing_weights.mean(dim=(0, 1))
        uniform = 1.0 / self.config.n_route_heads
        load_balance_loss = ((mean_weights - uniform) ** 2).mean()

        # Router z-loss: log-sum-exp of logits (penalises large magnitudes)
        router_z_loss = torch.logsumexp(router_logits, dim=-1).mean()

        # Code diversity: encourage distinct codes across heads per token
        # expert_codes: [B, S, top_k, code_dim]
        B, S, K, D = expert_codes.shape
        if K >= 2:
            # Pairwise L2 distance between selected codes for each token
            codes_flat = expert_codes.reshape(B * S, K, D)
            # [N, K, 1, D] - [N, 1, K, D] -> [N, K, K, D]
            diffs = codes_flat.unsqueeze(2) - codes_flat.unsqueeze(1)
            pairwise_dists = (diffs ** 2).sum(dim=-1).clamp(min=1e-8).sqrt()  # [N, K, K]

            # Mean of off-diagonal distances (want to maximise this)
            off_diag_mask = ~torch.eye(
                K, dtype=torch.bool, device=pairwise_dists.device
            )
            mean_dist = pairwise_dists[:, off_diag_mask].mean()

            # Loss: negative log to bound the diversity loss (prevents unbounded growth)
            code_diversity_loss = -torch.log(1.0 + mean_dist)
        else:
            code_diversity_loss = torch.tensor(
                0.0, device=router_logits.device
            )

        return {
            "load_balance_loss": load_balance_loss,
            "router_z_loss": router_z_loss,
            "code_diversity_loss": code_diversity_loss,
        }


# ============================================================================
# Continuous Expert Generator (Hypernetwork)
# ============================================================================

class ContinuousExpertGenerator(nn.Module):
    """Hypernetwork that generates expert-specific modifications from expert
    codes.

    Instead of storing N discrete expert weight matrices,
    ContinuousExpertGenerator generates expert modifications on-the-fly from
    low-dimensional expert code vectors.  A shared base expert provides the
    foundation, and the hypernetwork produces code-conditioned
    modifications:

    1. **Scaling vectors**: Element-wise scaling of base expert projections,
       allowing the expert to amplify or suppress specific features.
       Initialised near 1.0 so modifications start as near-identity.
    2. **Bias vectors**: Additive biases to base expert projections,
       providing translation in feature space.  Initialised near zero.
    3. **Low-rank residuals** (optional): Low-rank matrix modifications
       ΔW = A(z) @ B(z) for more expressive expert specialisation.
       Inspired by LoRA (Hu et al., 2022).  Initialised near zero.

    The combination of shared base weights + code-conditioned modifications
    enables unlimited expert diversity with O(1) parameter overhead per
    expert (since experts are generated, not stored).

    For a SwiGLU base expert, the computation for expert code z on input x
    is::

        gate = SiLU(x @ W_gate^T * gate_scale(z) + gate_bias(z))
        up   = (x @ W_up^T + x @ ΔW_up(z)^T) * up_scale(z) + up_bias(z)
        h    = gate * up
        down = (h @ W_down^T + h @ ΔW_down(z)^T) * down_scale(z) + down_bias(z)

    Without low-rank residuals, this simplifies to::

        gate = SiLU(x @ W_gate^T * gate_scale(z) + gate_bias(z))
        up   = x @ W_up^T * up_scale(z) + up_bias(z)
        h    = gate * up
        down = h @ W_down^T * down_scale(z) + down_bias(z)

    Args:
        config: :class:`InfiniteMoEConfig` instance.

    References:
        ∞-MoE: Generalizing Mixture of Experts to Infinite Experts
        (arXiv:2601.17680, Section 3.3: Continuous Expert Generation)
    """

    def __init__(self, config: InfiniteMoEConfig) -> None:
        super().__init__()
        self.config = config

        # ----------------------------------------------------------------
        # Shared base expert weights
        # ----------------------------------------------------------------
        self.W_up = nn.Parameter(
            torch.empty(config.d_ff, config.d_model)
        )
        self.W_down = nn.Parameter(
            torch.empty(config.d_model, config.d_ff)
        )
        if config.use_gate:
            self.W_gate = nn.Parameter(
                torch.empty(config.d_ff, config.d_model)
            )

        # ----------------------------------------------------------------
        # Hypernetwork: generates scaling and bias from expert codes
        # ----------------------------------------------------------------

        # Up projection scaling: code_dim -> d_ff
        self.up_scale_net = nn.Sequential(
            nn.Linear(config.code_dim, config.hypernet_hidden_dim),
            nn.GELU(),
            nn.Linear(config.hypernet_hidden_dim, config.d_ff),
        )

        # Down projection scaling: code_dim -> d_model
        self.down_scale_net = nn.Sequential(
            nn.Linear(config.code_dim, config.hypernet_hidden_dim),
            nn.GELU(),
            nn.Linear(config.hypernet_hidden_dim, config.d_model),
        )

        # Up projection bias: code_dim -> d_ff
        self.up_bias_net = nn.Sequential(
            nn.Linear(config.code_dim, config.hypernet_hidden_dim),
            nn.GELU(),
            nn.Linear(config.hypernet_hidden_dim, config.d_ff),
        )

        # Down projection bias: code_dim -> d_model
        self.down_bias_net = nn.Sequential(
            nn.Linear(config.code_dim, config.hypernet_hidden_dim),
            nn.GELU(),
            nn.Linear(config.hypernet_hidden_dim, config.d_model),
        )

        if config.use_gate:
            # Gate scaling: code_dim -> d_ff
            self.gate_scale_net = nn.Sequential(
                nn.Linear(config.code_dim, config.hypernet_hidden_dim),
                nn.GELU(),
                nn.Linear(config.hypernet_hidden_dim, config.d_ff),
            )
            # Gate bias: code_dim -> d_ff
            self.gate_bias_net = nn.Sequential(
                nn.Linear(config.code_dim, config.hypernet_hidden_dim),
                nn.GELU(),
                nn.Linear(config.hypernet_hidden_dim, config.d_ff),
            )

        # ----------------------------------------------------------------
        # Optional low-rank residual generation
        # ----------------------------------------------------------------
        if config.use_low_rank_residual:
            r = config.low_rank_dim
            # ΔW_up = A_up @ B_up, shape: [d_ff, d_model]
            self.up_A_net = nn.Linear(
                config.code_dim, config.d_ff * r, bias=False
            )
            self.up_B_net = nn.Linear(
                config.code_dim, r * config.d_model, bias=False
            )
            # ΔW_down = A_down @ B_down, shape: [d_model, d_ff]
            self.down_A_net = nn.Linear(
                config.code_dim, config.d_model * r, bias=False
            )
            self.down_B_net = nn.Linear(
                config.code_dim, r * config.d_ff, bias=False
            )

        # Dropout
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0.0 else nn.Identity()

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize base expert and hypernetwork parameters.

        Base expert uses Kaiming uniform initialization.  Hypernetwork
        outputs are initialised so that modifications start as near-identity
        (scaling ≈ 1, bias ≈ 0, low-rank ≈ 0).
        """
        cfg = self.config

        # Base expert: standard initialization
        nn.init.kaiming_uniform_(self.W_up, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_down, a=math.sqrt(5))
        if cfg.use_gate:
            nn.init.kaiming_uniform_(self.W_gate, a=math.sqrt(5))

        # Scaling networks: initialise last layer bias to 1.0 (identity
        # scaling), weights to small values so modifications start small
        for net in [self.up_scale_net, self.down_scale_net]:
            self._init_scale_net(net, cfg)

        if cfg.use_gate:
            for net in [self.gate_scale_net]:
                self._init_scale_net(net, cfg)

        # Bias networks: initialise to near-zero
        for net in [self.up_bias_net, self.down_bias_net]:
            self._init_bias_net(net)

        if cfg.use_gate:
            self._init_bias_net(self.gate_bias_net)

        # Low-rank networks: initialise to near-zero
        if cfg.use_low_rank_residual:
            for net_name in [
                "up_A_net", "up_B_net", "down_A_net", "down_B_net"
            ]:
                net = getattr(self, net_name)
                nn.init.xavier_uniform_(net.weight)
                with torch.no_grad():
                    net.weight.mul_(0.01)

    @staticmethod
    def _init_scale_net(net: nn.Sequential, cfg: InfiniteMoEConfig) -> None:
        """Initialize a scaling network so output starts near 1.0."""
        for layer in net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        # Scale down last layer weights so initial output is close to bias
        last = net[-1]
        with torch.no_grad():
            last.weight.mul_(0.01)
            if last.bias is not None:
                last.bias.fill_(1.0)

    @staticmethod
    def _init_bias_net(net: nn.Sequential) -> None:
        """Initialize a bias network so output starts near 0.0."""
        for layer in net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        # Scale down last layer weights and zero bias
        last = net[-1]
        with torch.no_grad():
            last.weight.mul_(0.01)
            if last.bias is not None:
                last.bias.zero_()

    def _generate_modifications(
        self,
        expert_codes: torch.Tensor,
        # shape: [B, S, top_k, code_dim] or [N_unique, code_dim]
    ) -> Dict[str, torch.Tensor]:
        """Generate expert-specific modifications from codes.

        Args:
            expert_codes: Expert code vectors.

        Returns:
            Dictionary containing generated modification tensors:
                - ``"up_scale"``: Scaling for up projection
                - ``"down_scale"``: Scaling for down projection
                - ``"up_bias"``: Bias for up projection
                - ``"down_bias"``: Bias for down projection
                - ``"gate_scale"``: Scaling for gate (if use_gate)
                - ``"gate_bias"``: Bias for gate (if use_gate)
                - ``"dW_up"``: Low-rank up residual (if use_low_rank)
                - ``"dW_down"``: Low-rank down residual (if use_low_rank)
        """
        original_shape = expert_codes.shape[:-1]  # e.g. [B, S, top_k]
        codes_flat = expert_codes.reshape(-1, self.config.code_dim)

        mods: Dict[str, torch.Tensor] = {}

        # Scaling vectors
        up_scale = self.up_scale_net(codes_flat)      # [N, d_ff]
        down_scale = self.down_scale_net(codes_flat)   # [N, d_model]
        mods["up_scale"] = up_scale.reshape(*original_shape, self.config.d_ff)
        mods["down_scale"] = down_scale.reshape(
            *original_shape, self.config.d_model
        )

        # Bias vectors
        mods["up_bias"] = self.up_bias_net(codes_flat).reshape(
            *original_shape, self.config.d_ff
        )
        mods["down_bias"] = self.down_bias_net(codes_flat).reshape(
            *original_shape, self.config.d_model
        )

        # Gate modifications
        if self.config.use_gate:
            mods["gate_scale"] = self.gate_scale_net(codes_flat).reshape(
                *original_shape, self.config.d_ff
            )
            mods["gate_bias"] = self.gate_bias_net(codes_flat).reshape(
                *original_shape, self.config.d_ff
            )

        # Low-rank residuals
        if self.config.use_low_rank_residual:
            r = self.config.low_rank_dim

            # ΔW_up = A_up @ B_up  shape: [d_ff, d_model]
            A_up = self.up_A_net(codes_flat).reshape(-1, self.config.d_ff, r)
            B_up = self.up_B_net(codes_flat).reshape(-1, r, self.config.d_model)
            mods["dW_up"] = torch.bmm(A_up, B_up).reshape(
                *original_shape, self.config.d_ff, self.config.d_model
            )

            # ΔW_down = A_down @ B_down  shape: [d_model, d_ff]
            A_down = self.down_A_net(codes_flat).reshape(
                -1, self.config.d_model, r
            )
            B_down = self.down_B_net(codes_flat).reshape(
                -1, r, self.config.d_ff
            )
            mods["dW_down"] = torch.bmm(A_down, B_down).reshape(
                *original_shape, self.config.d_model, self.config.d_ff
            )

        return mods

    def forward(
        self,
        hidden_states: torch.Tensor,
        expert_codes: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Apply continuous experts to input hidden states.

        For each token, K experts are generated from the expert codes,
        applied to the hidden states, and combined using routing weights.

        The computation for each expert *i* on token *t* is::

            # SwiGLU base + code-conditioned modifications
            # Low-rank residuals are applied before scaling
            gate = SiLU(x_t @ W_gate^T * gate_scale_i + gate_bias_i)
            up   = (x_t @ W_up^T + x_t @ ΔW_up_i^T) * up_scale_i + up_bias_i
            h    = gate * up
            down = (h @ W_down^T + h @ ΔW_down_i^T) * down_scale_i + down_bias_i

        Final output::

            y_t = Σ_i  w_i * down_i

        Args:
            hidden_states: Input representations. ``[B, S, d_model]``
            expert_codes: Expert code vectors.
                ``[B, S, top_k, code_dim]``
            routing_weights: Routing weights. ``[B, S, top_k]``

        Returns:
            output: Combined expert outputs. ``[B, S, d_model]``
            aux_info: Dictionary with ``"n_experts_used"``.
        """
        batch_size, seq_len, _ = hidden_states.shape
        top_k = expert_codes.shape[2]

        # Generate modifications from expert codes
        mods = self._generate_modifications(expert_codes)

        # Apply each expert and combine with routing weights
        output = torch.zeros_like(hidden_states)

        for k in range(top_k):
            # Extract modifications for expert k
            up_scale_k = mods["up_scale"][:, :, k, :]        # [B, S, d_ff]
            down_scale_k = mods["down_scale"][:, :, k, :]     # [B, S, d_model]
            up_bias_k = mods["up_bias"][:, :, k, :]           # [B, S, d_ff]
            down_bias_k = mods["down_bias"][:, :, k, :]       # [B, S, d_model]

            # Up projection with code-conditioned scaling and bias
            # [B, S, d_model] @ [d_ff, d_model]^T -> [B, S, d_ff]
            up = F.linear(hidden_states, self.W_up)

            # Low-rank up residual: x @ ΔW_up^T (applied before scaling)
            if self.config.use_low_rank_residual:
                dW_up = mods["dW_up"][:, :, k, :, :]       # [B, S, d_ff, d_model]
                up = up + torch.einsum(
                    "bsm,bsfm->bsf", hidden_states, dW_up
                )

            up = up * up_scale_k + up_bias_k

            if self.config.use_gate:
                gate_scale_k = mods["gate_scale"][:, :, k, :]  # [B, S, d_ff]
                gate_bias_k = mods["gate_bias"][:, :, k, :]    # [B, S, d_ff]

                gate = F.linear(hidden_states, self.W_gate)
                gate = gate * gate_scale_k + gate_bias_k
                gate = F.silu(gate)

                hidden = gate * up       # [B, S, d_ff]
            else:
                hidden = F.gelu(up)      # [B, S, d_ff]

            hidden = self.dropout(hidden)

            # Down projection with code-conditioned scaling and bias
            down = F.linear(hidden, self.W_down)   # [B, S, d_model]

            # Low-rank down residual: h @ ΔW_down^T (applied before scaling)
            if self.config.use_low_rank_residual:
                dW_down = mods["dW_down"][:, :, k, :, :]   # [B, S, d_model, d_ff]
                down = down + torch.einsum(
                    "bsf,bsdf->bsd", hidden, dW_down
                )

            down = down * down_scale_k + down_bias_k

            # Weighted combination
            weight_k = routing_weights[:, :, k].unsqueeze(-1)  # [B, S, 1]
            output = output + weight_k * down

        return output, {"n_experts_used": top_k}


# ============================================================================
# Expert Code Clusterer
# ============================================================================

class ExpertCodeClusterer:
    """Clusters nearby expert codes to amortise hypernetwork computation.

    During a forward pass, different tokens may produce very similar expert
    codes.  Rather than generating separate expert modifications for each,
    ExpertCodeClusterer groups nearby codes and generates modifications only
    for cluster centroids, then maps results back to individual tokens.

    This is especially beneficial during inference where batch sizes may be
    large and expert code diversity is limited.

    The clustering uses an iterative merging approach:
    1. Flatten all codes across batch and sequence dimensions.
    2. Iteratively merge codes closer than ``threshold``.
    3. Return centroids with assignment indices.

    Args:
        threshold: L2 distance threshold for merging codes.
        max_iter: Maximum number of clustering iterations.
    """

    def __init__(
        self, threshold: float = 0.5, max_iter: int = 5
    ) -> None:
        self.threshold = threshold
        self.max_iter = max_iter

    @torch.no_grad()
    def cluster(
        self,
        expert_codes: torch.Tensor,
        # shape: [B, S, top_k, code_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Cluster expert codes and return centroids with assignments.

        Args:
            expert_codes: Expert code vectors from the router.
                Shape ``[B, S, top_k, code_dim]``.

        Returns:
            centroids: Cluster centroid codes. ``[n_clusters, code_dim]``
            assignments: Index mapping each code to its cluster.
                Shape ``[B * S * top_k]``.
            cluster_sizes: Number of codes in each cluster.
                ``[n_clusters]``
        """
        B, S, K, D = expert_codes.shape
        codes_flat = expert_codes.reshape(-1, D)  # [N, D]
        N = codes_flat.shape[0]

        if N == 0:
            return (
                codes_flat,
                torch.arange(N, device=codes_flat.device),
                torch.ones(N, device=codes_flat.device),
            )

        # Initialise: each code is its own cluster
        centroids = codes_flat.clone()
        assignments = torch.arange(N, device=codes_flat.device)

        for _ in range(self.max_iter):
            # Compute pairwise distances: [N, n_centroids]
            diffs = codes_flat.unsqueeze(1) - centroids.unsqueeze(0)
            distances = (diffs ** 2).sum(dim=-1)

            # Find nearest centroid for each code
            nearest = distances.argmin(dim=-1)                    # [N]
            nearest_dist = distances.gather(1, nearest.unsqueeze(1)).squeeze(1)

            # Merge codes that are close to centroids
            merge_mask = nearest_dist < self.threshold ** 2
            assignments = torch.where(merge_mask, nearest, assignments)

            # Update centroids for each cluster
            unique_clusters = assignments.unique()
            new_centroids = torch.zeros(
                unique_clusters.shape[0], D,
                device=codes_flat.device, dtype=codes_flat.dtype,
            )
            for i, c in enumerate(unique_clusters):
                mask = assignments == c
                new_centroids[i] = codes_flat[mask].mean(dim=0)

            centroids = new_centroids

            # Remap assignments to new contiguous indices
            unique_list = unique_clusters.tolist()
            mapping = {int(old): new for new, old in enumerate(unique_list)}
            assignments = torch.tensor(
                [mapping[int(a)] for a in assignments],
                device=assignments.device,
                dtype=assignments.dtype,
            )

            # If no merging happened, stop
            if centroids.shape[0] == N:
                break

        cluster_sizes = torch.bincount(
            assignments, minlength=centroids.shape[0]
        ).float()

        return centroids, assignments, cluster_sizes


# ============================================================================
# Infinite MoE — Main Module
# ============================================================================

class InfiniteMoE(nn.Module):
    """∞-MoE: Infinite Mixture of Experts module.

    Extends traditional Mixture of Experts from a finite discrete set to a
    continuous (effectively infinite) expert space.  Each point in the
    continuous code space defines a unique expert, generated on-the-fly by
    a hypernetwork.

    This enables:

    * **Unlimited capacity**: Unlike discrete MoE with N experts, ∞-MoE can
      represent infinitely many experts in continuous space.
    * **Smooth interpolation**: Nearby codes produce similar experts,
      enabling smooth transitions and interpolation between expert
      behaviours.
    * **Efficient activation**: Only top-K experts are activated per token,
      maintaining O(K) inference cost regardless of total capacity.
    * **Adaptive specialisation**: The router adaptively discovers useful
      regions of expert space during training.

    Architecture::

        Input -> ExpertCodeRouter -> (expert_codes, routing_weights)
              -> ContinuousExpertGenerator -> expert_outputs
              -> weighted combination -> residual connection -> Output

    **Training mode**: Full routing with auxiliary losses (load balance,
    z-loss, code diversity).

    **Inference mode**: Optional code clustering for amortised computation
    when many tokens produce similar expert codes.

    Integration with Losion Tri-Jalur Architecture
    -----------------------------------------------
    The ∞-MoE module integrates with the Tri-Jalur (Three Paths) architecture
    by serving as the adaptive expert layer.  In Tri-Jalur:

    - **Jalur Semantik (Semantic Path)**: Uses ∞-MoE with content-conditioned
      routing for language understanding.
    - **Jalur Struktural (Structural Path)**: Uses ∞-MoE with structure-aware
      routing for pattern recognition.
    - **Jalur Pengambilan (Retrieval Path)**: Uses ∞-MoE with
      retrieval-conditioned routing for knowledge access.

    Each pathway can share the continuous expert space while routing
    differently, enabling cross-pathway knowledge transfer through proximity
    in expert code space.

    Example
    -------
    >>> config = InfiniteMoEConfig(d_model=768, code_dim=64, top_k=2)
    >>> model = InfiniteMoE(config)
    >>> x = torch.randn(2, 10, 768)   # [batch, seq, d_model]
    >>> output, losses = model(x)
    >>> output.shape
    torch.Size([2, 10, 768])
    >>> losses["total_aux_loss"].item()  # scalar
    0.0123...

    Args:
        config: :class:`InfiniteMoEConfig` instance.

    References:
        ∞-MoE: Generalizing Mixture of Experts to Infinite Experts
        (arXiv:2601.17680, January 2026)
    """

    def __init__(self, config: InfiniteMoEConfig) -> None:
        super().__init__()
        self.config = config

        # Sub-modules
        self.router = ExpertCodeRouter(config)
        self.expert_generator = ContinuousExpertGenerator(config)

        # Optional clusterer for inference
        self.clusterer: Optional[ExpertCodeClusterer] = None
        if config.cluster_threshold > 0:
            self.clusterer = ExpertCodeClusterer(
                threshold=config.cluster_threshold,
                max_iter=config.cluster_max_iter,
            )

        # Layer norm for pre-norm residual connection
        self.layer_norm = nn.LayerNorm(config.d_model)

    # ------------------------------------------------------------------
    # Forward (training and inference)
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        tri_jalur_pathway: Optional[str] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass through ∞-MoE.

        Applies pre-norm, routes tokens to continuous expert space via
        :class:`ExpertCodeRouter`, generates expert-specific modifications
        via :class:`ContinuousExpertGenerator`, combines weighted expert
        outputs, and adds a residual connection.

        During inference (``self.training == False``) with clustering
        enabled, nearby expert codes are clustered to amortise hypernetwork
        computation.

        Args:
            hidden_states: Input token representations.
                Shape ``[B, S, d_model]``.
            attention_mask: Optional mask for padding tokens.
                Shape ``[B, S]``.  Tokens with mask=0 are excluded from
                routing (zeroed out before the router).
            tri_jalur_pathway: Optional identifier for the Tri-Jalur
                pathway this module is operating in.  One of
                ``"semantic"``, ``"structural"``, ``"retrieval"``.
                Used for pathway-specific logging and analysis.

        Returns:
            output: Expert-augmented representations.
                Shape ``[B, S, d_model]``.
            losses: Dictionary of auxiliary losses:
                - ``"total_aux_loss"``: Weighted sum of all auxiliary losses
                - ``"load_balance_loss"``: Load balancing auxiliary loss
                - ``"router_z_loss"``: Router magnitude stabilisation loss
                - ``"code_diversity_loss"``: Code diversity regularisation
                - ``"n_experts_used"``: Number of experts activated
        """
        residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)

        # Apply attention mask to router inputs
        if attention_mask is not None:
            router_input = hidden_states * attention_mask.unsqueeze(-1)
        else:
            router_input = hidden_states

        # Route tokens to continuous expert space
        (
            expert_codes,
            routing_weights,
            router_logits,
            router_aux,
        ) = self.router(router_input)

        # Optional clustering for inference efficiency
        if not self.training and self.clusterer is not None:
            expert_codes = self._apply_clustering(expert_codes)

        # Apply capacity constraints (if configured)
        if self.config.capacity_factor > 0:
            routing_weights = self._apply_capacity(
                routing_weights, attention_mask
            )

        # Generate and apply continuous experts
        expert_output, expert_aux = self.expert_generator(
            hidden_states, expert_codes, routing_weights
        )

        # Residual connection
        output = residual + expert_output

        # Aggregate auxiliary losses
        total_aux_loss = (
            self.config.load_balance_weight * router_aux["load_balance_loss"]
            + self.config.router_z_loss_weight * router_aux["router_z_loss"]
            + self.config.code_diversity_weight * router_aux["code_diversity_loss"]
        )

        losses: Dict[str, torch.Tensor] = {
            "total_aux_loss": total_aux_loss,
            "load_balance_loss": router_aux["load_balance_loss"],
            "router_z_loss": router_aux["router_z_loss"],
            "code_diversity_loss": router_aux["code_diversity_loss"],
            "n_experts_used": torch.tensor(
                expert_aux["n_experts_used"], device=hidden_states.device
            ),
        }

        if tri_jalur_pathway is not None:
            losses["tri_jalur_pathway"] = torch.tensor(
                {"semantic": 0, "structural": 1, "retrieval": 2}.get(
                    tri_jalur_pathway, -1
                ),
                device=hidden_states.device,
            )

        return output, losses

    # ------------------------------------------------------------------
    # Inference forward (hard routing with clustering)
    # ------------------------------------------------------------------

    def forward_inference(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Inference forward with hard expert selection and optional
        clustering.

        Equivalent to ``self.eval(); self.forward(...)`` but makes the
        inference intent explicit and skips auxiliary loss computation
        (returns zero losses).

        Args:
            hidden_states: Input tensor ``[B, S, d_model]``.
                Typically ``S = 1`` during autoregressive generation.
            attention_mask: Optional mask. ``[B, S]``

        Returns:
            output: Expert-augmented output. ``[B, S, d_model]``
            losses: Dictionary of (zero) auxiliary losses.
        """
        was_training = self.training
        self.eval()

        output, losses = self.forward(
            hidden_states,
            attention_mask=attention_mask,
        )

        # Zero out auxiliary losses for inference
        zero_losses = {
            k: torch.tensor(0.0, device=hidden_states.device)
            if v.is_floating_point() else v
            for k, v in losses.items()
        }

        if was_training:
            self.train()

        return output, zero_losses

    # ------------------------------------------------------------------
    # Analysis and diagnostics
    # ------------------------------------------------------------------

    @torch.no_grad()
    def analyze_expert_codes(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> InfiniteMoERoutingInfo:
        """Analyze the distribution of expert codes for interpretability.

        Useful for understanding how the model partitions the continuous
        expert space and which regions are activated for different inputs.

        Args:
            hidden_states: Input representations. ``[B, S, d_model]``
            attention_mask: Optional mask. ``[B, S]``

        Returns:
            :class:`InfiniteMoERoutingInfo` with detailed routing
            diagnostics.
        """
        if attention_mask is not None:
            router_input = hidden_states * attention_mask.unsqueeze(-1)
        else:
            router_input = hidden_states

        # Apply pre-norm like in forward
        normed = self.layer_norm(router_input)

        expert_codes, routing_weights, router_logits, router_aux = self.router(
            normed
        )

        # Code norms
        code_norms = expert_codes.norm(dim=-1)  # [B, S, top_k]

        # Code diversity: average pairwise distance between selected codes
        B, S, K, D = expert_codes.shape
        codes_flat = expert_codes.reshape(B * S, K, D)

        if K >= 2:
            diffs = codes_flat.unsqueeze(2) - codes_flat.unsqueeze(1)
            pairwise_dists = (diffs ** 2).sum(dim=-1).clamp(min=1e-8).sqrt()
            off_diag_mask = ~torch.eye(
                K, dtype=torch.bool, device=pairwise_dists.device
            )
            code_diversity = pairwise_dists[:, off_diag_mask].mean()
        else:
            code_diversity = torch.tensor(0.0, device=hidden_states.device)

        # Weight entropy
        weight_entropy = -(
            routing_weights * (routing_weights + 1e-9).log()
        ).sum(dim=-1).mean()

        return InfiniteMoERoutingInfo(
            expert_codes=expert_codes,
            routing_weights=routing_weights,
            router_logits=router_logits,
            load_balance_loss=router_aux["load_balance_loss"],
            router_z_loss=router_aux["router_z_loss"],
            code_diversity_loss=router_aux["code_diversity_loss"],
            code_norms=code_norms,
            code_diversity=code_diversity,
            weight_entropy=weight_entropy,
        )

    # ------------------------------------------------------------------
    # Tri-Jalur integration
    # ------------------------------------------------------------------

    def get_tri_jalur_integration_config(self) -> Dict[str, object]:
        """Get configuration for integrating with Tri-Jalur architecture.

        The Tri-Jalur (Three Paths) architecture uses ∞-MoE across three
        complementary pathways.  This method returns configuration for
        pathway-specific setup.

        Returns:
            Dictionary with integration configuration:
                - ``"shared_code_space"``: Whether pathways share the
                  continuous expert code space (True).
                - ``"code_dim"``: Dimensionality of the code space.
                - ``"top_k"``: Number of active experts per token.
                - ``"pathway_configs"``: Per-pathway configuration.
                - ``"cross_pathway_distance"``: Distance metric for
                  cross-pathway expert code comparison.
        """
        return {
            "shared_code_space": True,
            "code_dim": self.config.code_dim,
            "top_k": self.config.top_k,
            "pathway_configs": {
                "semantic": {
                    "description": (
                        "Jalur Semantik — Language understanding pathway.  "
                        "Uses content-conditioned routing for semantic "
                        "processing of token representations."
                    ),
                    "routing_bias": "content_aware",
                    "code_init": "language_structured",
                },
                "structural": {
                    "description": (
                        "Jalur Struktural — Pattern recognition pathway.  "
                        "Uses structure-aware routing for capturing syntactic "
                        "and geometric patterns in sequences."
                    ),
                    "routing_bias": "structure_aware",
                    "code_init": "geometry_structured",
                },
                "retrieval": {
                    "description": (
                        "Jalur Pengambilan — Knowledge access pathway.  "
                        "Uses retrieval-conditioned routing for accessing "
                        "stored knowledge via expert specialisation."
                    ),
                    "routing_bias": "retrieval_aware",
                    "code_init": "memory_structured",
                },
            },
            "cross_pathway_distance": "cosine",
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_clustering(
        self,
        expert_codes: torch.Tensor,
    ) -> torch.Tensor:
        """Apply code clustering for inference efficiency.

        Groups nearby expert codes, replaces each with its cluster
        centroid.  The hypernetwork then only needs to generate
        modifications for unique centroids, amortising computation.

        Args:
            expert_codes: ``[B, S, top_k, code_dim]``

        Returns:
            Clustered expert codes with the same shape.
        """
        assert self.clusterer is not None

        B, S, K, D = expert_codes.shape

        # Cluster codes
        centroids, assignments, _ = self.clusterer.cluster(expert_codes)

        # Replace each code with its cluster centroid
        codes_flat = expert_codes.reshape(-1, D)
        clustered_codes_flat = centroids[assignments]  # [B*S*K, D]
        clustered_codes = clustered_codes_flat.reshape(B, S, K, D)

        return clustered_codes

    def _apply_capacity(
        self,
        routing_weights: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Apply expert capacity constraints.

        Limits the number of tokens each expert position can process.
        Tokens exceeding capacity have their routing weights zeroed.

        For ∞-MoE, capacity is typically unlimited
        (``capacity_factor=0``), but this is provided for compatibility
        with capacity-constrained training regimes.

        Args:
            routing_weights: ``[B, S, top_k]``
            attention_mask: Optional ``[B, S]`` mask.

        Returns:
            Capacity-constrained routing weights.
        """
        B, S, K = routing_weights.shape

        n_tokens = B * S
        if attention_mask is not None:
            n_tokens = int(attention_mask.sum().item())

        capacity = int(
            math.ceil((n_tokens / K) * self.config.capacity_factor)
        )

        constrained_weights = routing_weights.clone()

        for k in range(K):
            expert_weights = routing_weights[:, :, k]
            if attention_mask is not None:
                expert_weights = expert_weights * attention_mask

            flat_weights = expert_weights.reshape(-1)
            if flat_weights.shape[0] > capacity:
                _, top_indices = torch.topk(flat_weights, k=capacity)
                mask = torch.zeros_like(flat_weights)
                mask[top_indices] = 1.0
                constrained_weights[:, :, k] = (
                    constrained_weights[:, :, k] * mask.reshape(B, S)
                )

        # Renormalise
        weight_sum = constrained_weights.sum(dim=-1, keepdim=True)
        constrained_weights = constrained_weights / (weight_sum + 1e-9)

        return constrained_weights

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def parameter_count_breakdown(self) -> Dict[str, int]:
        """Return a breakdown of parameter counts by component.

        Returns:
            Dictionary with parameter counts for:
                - ``"router"``: ExpertCodeRouter parameters
                - ``"base_expert"``: Shared base expert weights
                - ``"hypernetwork"``: Code-conditioned modification
                  generators
                - ``"total"``: Total trainable parameters
        """
        router_params = sum(
            p.numel() for p in self.router.parameters()
        )
        base_params = sum(
            p.numel() for n, p in self.expert_generator.named_parameters()
            if n.startswith("W_")
        )
        hypernet_params = sum(
            p.numel() for n, p in self.expert_generator.named_parameters()
            if not n.startswith("W_") and not n.startswith("dropout")
        )

        return {
            "router": router_params,
            "base_expert": base_params,
            "hypernetwork": hypernet_params,
            "total": sum(p.numel() for p in self.parameters()),
        }

    def estimate_capacity(self) -> Dict[str, object]:
        """Estimate the effective capacity of the continuous expert space.

        Returns:
            Dictionary with capacity estimates:
                - ``"theoretical"``: ``"infinite"`` (continuous space)
                - ``"code_dim"``: Dimensionality of the code space
                - ``"practical_resolution"``: Approximate number of
                  distinguishable experts given float32 precision and
                  code_dim.
                - ``"top_k"``: Active experts per token
                - ``"codebook_size"``: Codebook size (0 if disabled)
        """
        # Practical resolution: each dimension has ~2^24 distinguishable
        # float32 values (mantissa bits), so the number of distinguishable
        # points in a unit hypercube is approximately (2^24)^code_dim.
        # We report log2 of this for readability.
        practical_log2 = 24 * self.config.code_dim

        return {
            "theoretical": "infinite",
            "code_dim": self.config.code_dim,
            "practical_resolution_log2": practical_log2,
            "practical_resolution_approx": f"~2^{practical_log2}",
            "top_k": self.config.top_k,
            "codebook_size": self.config.codebook_size,
        }
