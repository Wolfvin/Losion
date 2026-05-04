"""
Cross-Jalur Attention-MoE Routing — Attention-Informed Expert Selection
========================================================================

Bridges Jalur 2 (Attention) and Jalur 3 (MoE/Retrieval) by using the
attention matrix from the Attention pathway to guide MoE expert routing
in the Retrieval pathway.

Based on "Improving Routing in Sparse MoE with Graph of Tokens"
(arXiv 2505.00792, May 2025), which shows that tokens that attend to
each other should route to similar experts.  By building a token
affinity graph from attention weights and smoothing routing decisions
through it, the cross-jalur router achieves:

1. **Attention-Informed Routing** — Tokens with high mutual attention
   receive similar expert assignments, improving coherence of the
   MoE output for semantically related tokens.

2. **Cross-Jalur Information Flow** — A learnable bridge between
   Jalur 2 and Jalur 3 that allows the attention pathway's structural
   knowledge to inform the retrieval pathway's capacity allocation.

3. **Graph-Based Token Affinity** — Constructs a sparse token affinity
   graph from the attention matrix, then propagates routing logits
   across graph edges so that neighbouring (high-attention) tokens
   influence each other's expert selection.

4. **Reduced Routing Fluctuations** — Attention-guided routing is more
   stable across training steps than token-independent routing, because
   the attention structure changes slowly relative to per-token logits.

Architecture
------------
The pipeline has three stages:

    attention_weights  →  AttentionGraphBuilder  →  affinity_graph
    token_hidden + affinity_graph  →  CrossJalurRouter  →  modified_logits
    original_logits + modified_logits  →  RoutingSmoother  →  final_logits

**AttentionGraphBuilder** extracts a sparse token affinity graph from
the raw attention matrix.  Sparsification is controlled by
``top_k_neighbors`` (keep only the K strongest attention edges per
token) and ``affinity_threshold`` (discard edges below a score
threshold).

**CrossJalurRouter** takes the affinity graph and produces
attention-informed routing logits by propagating the original routing
logits across graph edges (one round of graph convolution).  A
learnable gate controls how much cross-token influence is applied.

**RoutingSmoother** blends the original MoE routing logits with the
attention-informed logits using an exponential moving average during
training and a simple convex combination at inference time.  The
blending weight ``alpha`` can be fixed or learned.

Compatibility
-------------
CrossJalurRouting is **drop-in compatible** with any MoE implementation
in Losion.  It only requires that the caller provides:

- ``attention_weights``: ``(batch, num_heads, seq_len, seq_len)`` —
  the raw attention matrix from the Attention pathway.
- ``hidden_states``: ``(batch, seq_len, d_model)`` — the token
  representations (used by the router projection).

The module outputs modified routing logits of shape
``(batch, seq_len, num_experts)`` that can be fed directly into any
standard top-K selection.

References
----------
- "Improving Routing in Sparse MoE with Graph of Tokens",
  arXiv 2505.00792 (May 2025) — core idea of attention-informed routing.
- Zhou et al., "Mixture-of-Experts with Expert Choice Routing" (2022) —
  baseline expert-choice routing.
- Fedus et al., "Switch Transformers" (2021) — load-balancing loss.

Hardware: Pure PyTorch.  No custom CUDA kernels required.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ============================================================================
# Data classes
# ============================================================================


@dataclass
class CrossJalurRoutingInfo:
    """Diagnostic information produced by CrossJalurRouting.

    Attributes:
        original_logits:
            Raw MoE router logits before attention modification
            ``(batch, seq_len, num_experts)``.
        modified_logits:
            Logits after attention-informed modification
            ``(batch, seq_len, num_experts)``.
        smoothed_logits:
            Final logits after blending original and modified
            ``(batch, seq_len, num_experts)``.
        affinity_graph:
            Sparse token affinity adjacency matrix
            ``(batch, seq_len, seq_len)``.
        graph_sparsity:
            Fraction of edges retained after sparsification (scalar).
        blend_alpha:
            Blending weight used (scalar or per-layer).
        graph_conv_norm:
            Mean norm of the graph-convolved logits before gating.
    """

    original_logits: torch.Tensor
    modified_logits: torch.Tensor
    smoothed_logits: torch.Tensor
    affinity_graph: torch.Tensor
    graph_sparsity: float
    blend_alpha: float
    graph_conv_norm: float


# ============================================================================
# AttentionGraphBuilder
# ============================================================================


class AttentionGraphBuilder(nn.Module):
    """Constructs a sparse token affinity graph from attention weights.

    The raw attention matrix ``(B, H, S, S)`` is first reduced across
    heads (mean or max), then sparsified by keeping only the top-K
    strongest attention edges per query token.  Optionally, edges with
    attention weight below ``affinity_threshold`` are also removed.

    The resulting graph is symmetric (if ``symmetrize=True``) and
    row-normalised so that each row sums to 1 (forming a Markov-style
    transition matrix for graph convolution).

    Args:
        num_heads:
            Number of attention heads (used for head reduction).
        top_k_neighbors:
            Number of strongest attention edges to retain per query
            token.  Lower values yield a sparser graph.  Default 8.
        affinity_threshold:
            Minimum attention weight for an edge to be retained.
            Set to 0.0 to disable threshold-based filtering.  Default
            0.01.
        head_reduction:
            How to combine attention weights across heads.  One of
            ``"mean"`` or ``"max"``.  Default ``"mean"``.
        symmetrize:
            Whether to symmetrize the adjacency matrix after
            sparsification by taking ``A = (A + A^T) / 2``.  Default
            True.
        self_loop_weight:
            Weight added to the diagonal of the affinity graph before
            normalisation.  A non-zero value ensures each token retains
            some influence on its own routing.  Default 1.0.
    """

    def __init__(
        self,
        num_heads: int = 8,
        top_k_neighbors: int = 8,
        affinity_threshold: float = 0.01,
        head_reduction: str = "mean",
        symmetrize: bool = True,
        self_loop_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.top_k_neighbors = top_k_neighbors
        self.affinity_threshold = affinity_threshold
        self.head_reduction = head_reduction
        self.symmetrize = symmetrize
        self.self_loop_weight = self_loop_weight

        if head_reduction not in ("mean", "max"):
            raise ValueError(
                f"head_reduction must be 'mean' or 'max', got '{head_reduction}'"
            )

    def forward(
        self,
        attention_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        """Build a sparse token affinity graph from attention weights.

        Args:
            attention_weights:
                Raw attention matrix ``(batch, num_heads, seq_len, seq_len)``.

        Returns:
            Tuple ``(affinity_graph, sparsity)`` where:
            - ``affinity_graph`` is ``(batch, seq_len, seq_len)`` with
              row-normalised values.
            - ``sparsity`` is the fraction of edges retained (float).
        """
        if attention_weights.dim() != 4:
            raise ValueError(
                f"attention_weights must be 4D (batch, heads, seq, seq), "
                f"got {attention_weights.dim()}D"
            )

        batch_size, num_heads, seq_len, _ = attention_weights.shape

        # ---- Step 1: Reduce across heads ----
        if self.head_reduction == "mean":
            affinity = attention_weights.mean(dim=1)  # (B, S, S)
        else:
            affinity = attention_weights.max(dim=1).values  # (B, S, S)

        # ---- Step 2: Sparsify by top-K per query token ----
        k = min(self.top_k_neighbors, seq_len)
        if k < seq_len:
            # Find the k-th largest value per row
            top_k_values, _ = torch.topk(affinity, k, dim=-1)  # (B, S, K)
            threshold_per_row = top_k_values[:, :, -1:]  # (B, S, 1) — k-th value
            mask = affinity >= threshold_per_row  # (B, S, S)
            affinity = affinity * mask.float()

        # ---- Step 3: Threshold-based filtering ----
        if self.affinity_threshold > 0.0:
            threshold_mask = affinity >= self.affinity_threshold
            affinity = affinity * threshold_mask.float()

        # ---- Step 4: Symmetrise ----
        if self.symmetrize:
            affinity = (affinity + affinity.transpose(-2, -1)) / 2.0

        # ---- Step 5: Add self-loops ----
        if self.self_loop_weight > 0.0:
            diag = torch.eye(seq_len, device=affinity.device, dtype=affinity.dtype)
            affinity = affinity + self.self_loop_weight * diag.unsqueeze(0)

        # ---- Step 6: Row-normalise ----
        row_sums = affinity.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        affinity = affinity / row_sums

        # ---- Sparsity metric ----
        total_possible = batch_size * seq_len * seq_len
        num_nonzero = (affinity > 0).sum().item()
        # Subtract self-loop entries from sparsity calculation
        num_self_loops = batch_size * seq_len
        sparsity = (num_nonzero - num_self_loops) / max(
            total_possible - num_self_loops, 1
        )

        return affinity, sparsity


# ============================================================================
# CrossJalurRouter
# ============================================================================


class CrossJalurRouter(nn.Module):
    """Attention-informed MoE router that uses the token affinity graph
    to modify routing logits via one round of graph convolution.

    The router performs the following computation:

        1. Compute base routing logits:   L = W_router · x
        2. Graph-convolve over affinity:  L' = α · A · L  +  (1-α) · L
        3. Apply learnable gate:          L'' = σ(gate) · L' + (1-σ(gate)) · L

    where ``A`` is the affinity graph from ``AttentionGraphBuilder``,
    ``α`` is the graph convolution strength, and ``σ(gate)`` is a
    learnable sigmoid gate that controls the influence of attention
    information.

    The key insight from arXiv 2505.00792 is that step 2 propagates
    routing information across tokens that attend to each other.  If
    token *i* attends strongly to token *j*, and token *j* has a high
    routing logit for expert *e*, then token *i*'s logit for expert *e*
    is also boosted.  This encourages semantically related tokens to
    route to the same experts, improving expert specialisation.

    Args:
        d_model:
            Model hidden dimension.
        num_experts:
            Number of MoE experts.
        graph_conv_strength:
            Strength of the graph convolution (α in step 2).  Default
            0.5.
        use_learnable_gate:
            Whether to use a learnable sigmoid gate (step 3).  If
            False, the gate is fixed at 1.0 (full attention influence).
            Default True.
        gate_init_value:
            Initial value for the learnable gate before sigmoid.
            A value of 0.0 corresponds to σ(0) = 0.5.  Default 0.0.
        top_k:
            Number of experts to select per token (for top-K routing).
            Default 2.
        dropout:
            Dropout rate applied to routing logits.  Default 0.0.
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        graph_conv_strength: float = 0.5,
        use_learnable_gate: bool = True,
        gate_init_value: float = 0.0,
        top_k: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.graph_conv_strength = graph_conv_strength
        self.use_learnable_gate = use_learnable_gate
        self.top_k = top_k

        # Base router projection
        self.router_proj = nn.Linear(d_model, num_experts, bias=False)

        # Learnable gate: controls how much attention information is used
        if use_learnable_gate:
            self.gate = nn.Parameter(torch.tensor(gate_init_value))
        else:
            self.register_buffer(
                "gate", torch.tensor(float("inf"))
            )  # σ(inf) = 1.0

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Initialise router weights
        nn.init.normal_(self.router_proj.weight, std=0.01)

    def forward(
        self,
        hidden_states: torch.Tensor,
        affinity_graph: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Compute attention-informed routing logits.

        Args:
            hidden_states:
                Token representations ``(batch, seq_len, d_model)``.
            affinity_graph:
                Sparse token affinity graph from AttentionGraphBuilder
                ``(batch, seq_len, seq_len)``.

        Returns:
            Tuple ``(original_logits, modified_logits, combined_logits, conv_norm)``:
            - ``original_logits``: ``(batch, seq_len, num_experts)``
            - ``modified_logits``: ``(batch, seq_len, num_experts)``
            - ``combined_logits``: ``(batch, seq_len, num_experts)``
            - ``conv_norm``: mean norm of the graph-convolved logits
        """
        batch_size, seq_len, _ = hidden_states.shape

        # ---- Step 1: Base routing logits ----
        original_logits = self.router_proj(hidden_states)  # (B, S, E)
        original_logits = self.dropout(original_logits)

        # ---- Step 2: Graph convolution ----
        # A · L  where A is (B, S, S) and L is (B, S, E)
        # Result: each token's logit is a weighted average of its
        # neighbours' logits, with weights from the affinity graph.
        convolved_logits = torch.bmm(
            affinity_graph, original_logits
        )  # (B, S, E)

        # Blend convolved with original using graph_conv_strength
        alpha = self.graph_conv_strength
        modified_logits = alpha * convolved_logits + (1.0 - alpha) * original_logits

        # Track norm of convolved logits for diagnostics
        with torch.no_grad():
            conv_norm = convolved_logits.norm(dim=-1).mean().item()

        # ---- Step 3: Learnable gate ----
        gate_value = torch.sigmoid(self.gate)
        combined_logits = (
            gate_value * modified_logits + (1.0 - gate_value) * original_logits
        )

        return original_logits, modified_logits, combined_logits, conv_norm


# ============================================================================
# RoutingSmoother
# ============================================================================


class RoutingSmoother(nn.Module):
    """Blends original routing logits with attention-informed logits.

    The smoother maintains an exponential moving average (EMA) of the
    attention-informed routing logits during training, which stabilises
    the routing signal across steps.  At inference time it uses a simple
    convex combination.

    The blending formula is:

        L_final = (1 - α) · L_original + α · L_attention_informed

    where ``α`` is controlled by:

    - ``blend_mode="fixed"``: ``α = blend_alpha`` (constant).
    - ``blend_mode="learned"``: ``α = σ(w)`` where ``w`` is a learnable
      scalar.
    - ``blend_mode="ema"``: ``α = blend_alpha`` but ``L_attention_informed``
      is replaced with its EMA.

    Args:
        blend_alpha:
            Blending weight for the attention-informed logits.
            Range [0, 1].  Default 0.3.
        blend_mode:
            One of ``"fixed"``, ``"learned"``, or ``"ema"``.
            Default ``"learned"``.
        ema_decay:
            Decay rate for the EMA of attention-informed logits.
            Only used when ``blend_mode="ema"``.  Default 0.99.
        num_experts:
            Number of experts (needed to initialise the EMA buffer
            when ``blend_mode="ema"``).  Default 16.
    """

    def __init__(
        self,
        blend_alpha: float = 0.3,
        blend_mode: str = "learned",
        ema_decay: float = 0.99,
        num_experts: int = 16,
    ) -> None:
        super().__init__()
        self.blend_alpha = blend_alpha
        self.blend_mode = blend_mode
        self.ema_decay = ema_decay
        self.num_experts = num_experts

        if blend_mode not in ("fixed", "learned", "ema"):
            raise ValueError(
                f"blend_mode must be 'fixed', 'learned', or 'ema', "
                f"got '{blend_mode}'"
            )

        # Learnable blending weight
        if blend_mode == "learned":
            # Initialise so that σ(0) ≈ 0.5, then scale by blend_alpha
            self.blend_weight = nn.Parameter(
                torch.tensor(math.log(blend_alpha / max(1.0 - blend_alpha, 1e-8)))
            )

        # EMA buffer for attention-informed logits
        if blend_mode == "ema":
            self.register_buffer(
                "ema_logits", torch.zeros(1, 1, num_experts)
            )
            self.register_buffer("ema_initialized", torch.tensor(False))

    def forward(
        self,
        original_logits: torch.Tensor,
        modified_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        """Blend original and attention-informed routing logits.

        Args:
            original_logits:
                Base MoE router logits ``(batch, seq_len, num_experts)``.
            modified_logits:
                Attention-informed logits from CrossJalurRouter
                ``(batch, seq_len, num_experts)``.

        Returns:
            Tuple ``(smoothed_logits, alpha)`` where:
            - ``smoothed_logits``: ``(batch, seq_len, num_experts)``
            - ``alpha``: the blending weight used (float)
        """
        # ---- Determine blending weight ----
        if self.blend_mode == "fixed":
            alpha = self.blend_alpha
        elif self.blend_mode == "learned":
            alpha = torch.sigmoid(self.blend_weight).item()
        elif self.blend_mode == "ema":
            alpha = self.blend_alpha
            # Update EMA
            if self.training:
                mean_modified = modified_logits.mean(dim=(0, 1), keepdim=True)
                if not self.ema_initialized.item():
                    self.ema_logits.copy_(mean_modified.detach())
                    self.ema_initialized.fill_(True)
                else:
                    self.ema_logits.mul_(self.ema_decay).add_(
                        mean_modified.detach(), alpha=1.0 - self.ema_decay
                    )
                # Replace modified_logits with EMA version
                modified_logits = (
                    self.blend_alpha * self.ema_logits
                    + (1.0 - self.blend_alpha) * modified_logits
                )
        else:
            alpha = self.blend_alpha

        # ---- Convex combination ----
        smoothed = (1.0 - alpha) * original_logits + alpha * modified_logits

        return smoothed, alpha


# ============================================================================
# CrossJalurRouting — Main Module
# ============================================================================


class CrossJalurRouting(nn.Module):
    """Cross-Jalur Attention-MoE Routing — main module.

    Combines ``AttentionGraphBuilder``, ``CrossJalurRouter``, and
    ``RoutingSmoother`` into a single drop-in module that takes
    attention weights and hidden states as input and produces modified
    routing logits for any MoE layer in the Losion framework.

    The module implements the core idea from arXiv 2505.00792: tokens
    that attend to each other should route to similar experts.  By
    propagating routing information through the attention-derived
    affinity graph, the router produces more coherent and stable expert
    assignments.

    Example
    -------
    >>> router = CrossJalurRouting(
    ...     d_model=512,
    ...     num_experts=16,
    ...     num_heads=8,
    ...     top_k=2,
    ... )
    >>> hidden = torch.randn(2, 32, 512)
    >>> attn = torch.softmax(torch.randn(2, 8, 32, 32), dim=-1)
    >>> weights, indices, info = router(hidden, attn)
    >>> weights.shape
    torch.Size([2, 32, 2])
    >>> indices.shape
    torch.Size([2, 32, 2])

    Args:
        d_model:
            Model hidden dimension.
        num_experts:
            Number of MoE experts.
        num_heads:
            Number of attention heads (for graph building).
        top_k:
            Number of experts to activate per token.  Default 2.
        graph_top_k_neighbors:
            Number of top attention edges to keep per token when
            building the affinity graph.  Default 8.
        graph_affinity_threshold:
            Minimum attention weight for an edge.  Default 0.01.
        graph_conv_strength:
            Strength of the graph convolution.  Default 0.5.
        use_learnable_gate:
            Whether to use a learnable gate for attention influence.
            Default True.
        blend_alpha:
            Blending weight for the smoother.  Default 0.3.
        blend_mode:
            Smoother blending mode (``"fixed"``, ``"learned"``, or
            ``"ema"``).  Default ``"learned"``.
        load_balance_weight:
            Weight for the auxiliary load-balancing loss.  Default 0.01.
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        num_heads: int = 8,
        top_k: int = 2,
        graph_top_k_neighbors: int = 8,
        graph_affinity_threshold: float = 0.01,
        graph_conv_strength: float = 0.5,
        use_learnable_gate: bool = True,
        blend_alpha: float = 0.3,
        blend_mode: str = "learned",
        load_balance_weight: float = 0.01,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.load_balance_weight = load_balance_weight

        # ---- AttentionGraphBuilder ----
        self.graph_builder = AttentionGraphBuilder(
            num_heads=num_heads,
            top_k_neighbors=graph_top_k_neighbors,
            affinity_threshold=graph_affinity_threshold,
        )

        # ---- CrossJalurRouter ----
        self.cross_jalur_router = CrossJalurRouter(
            d_model=d_model,
            num_experts=num_experts,
            graph_conv_strength=graph_conv_strength,
            use_learnable_gate=use_learnable_gate,
            top_k=top_k,
        )

        # ---- RoutingSmoother ----
        self.smoother = RoutingSmoother(
            blend_alpha=blend_alpha,
            blend_mode=blend_mode,
            num_experts=num_experts,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, CrossJalurRoutingInfo]:
        """Compute attention-informed MoE routing.

        Args:
            hidden_states:
                Token representations ``(batch, seq_len, d_model)``.
            attention_weights:
                Raw attention matrix from the Attention pathway
                ``(batch, num_heads, seq_len, seq_len)``.

        Returns:
            Tuple ``(weights, indices, routing_info)`` where:
            - ``weights``: ``(batch, seq_len, top_k)`` — softmax-normalised
              weights for the selected experts.
            - ``indices``: ``(batch, seq_len, top_k)`` — expert indices.
            - ``routing_info``: ``CrossJalurRoutingInfo`` with diagnostics.
        """
        # ---- Step 1: Build affinity graph ----
        affinity_graph, graph_sparsity = self.graph_builder(attention_weights)

        # ---- Step 2: Compute attention-informed routing logits ----
        original_logits, modified_logits, combined_logits, conv_norm = (
            self.cross_jalur_router(hidden_states, affinity_graph)
        )

        # ---- Step 3: Smooth / blend logits ----
        smoothed_logits, blend_alpha = self.smoother(
            original_logits, combined_logits
        )

        # ---- Step 4: Top-K selection ----
        top_k_logits, top_k_indices = torch.topk(
            smoothed_logits, self.top_k, dim=-1
        )
        top_k_weights = F.softmax(top_k_logits, dim=-1)

        # ---- Step 5: Build routing info ----
        routing_info = CrossJalurRoutingInfo(
            original_logits=original_logits,
            modified_logits=modified_logits,
            smoothed_logits=smoothed_logits,
            affinity_graph=affinity_graph,
            graph_sparsity=graph_sparsity,
            blend_alpha=blend_alpha,
            graph_conv_norm=conv_norm,
        )

        return top_k_weights, top_k_indices, routing_info

    def compute_aux_loss(
        self,
        routing_info: CrossJalurRoutingInfo,
    ) -> torch.Tensor:
        """Compute auxiliary load-balancing loss.

        Uses the Switch Transformer style load-balancing loss on the
        smoothed routing logits to encourage even expert utilisation.

        Args:
            routing_info: ``CrossJalurRoutingInfo`` from ``forward()``.

        Returns:
            Scalar auxiliary loss.
        """
        logits = routing_info.smoothed_logits  # (B, S, E)
        probs = F.softmax(logits, dim=-1)
        assignments = probs.argmax(dim=-1)  # (B, S)
        one_hot = F.one_hot(assignments, self.num_experts).float()
        f = one_hot.mean(dim=(0, 1))  # (E,)
        P = probs.mean(dim=(0, 1))  # (E,)
        balance_loss = self.num_experts * (f * P).sum()
        return self.load_balance_weight * balance_loss

    def compute_routing_stability_loss(
        self,
        routing_info: CrossJalurRoutingInfo,
        prev_routing_info: Optional[CrossJalurRoutingInfo] = None,
    ) -> torch.Tensor:
        """Compute routing stability loss.

        Penalises large changes in routing decisions between consecutive
        steps.  This is the loss that implements "reduced routing
        fluctuations" from arXiv 2505.00792.

        Args:
            routing_info: Current step's routing info.
            prev_routing_info: Previous step's routing info (optional).
                If None, returns zero loss.

        Returns:
            Scalar stability loss.
        """
        if prev_routing_info is None:
            return torch.tensor(0.0, device=routing_info.smoothed_logits.device)

        # KL divergence between current and previous routing distributions
        current_probs = F.log_softmax(routing_info.smoothed_logits, dim=-1)
        prev_probs = F.softmax(prev_routing_info.smoothed_logits, dim=-1)

        # KL(prev || current)
        kl = (prev_probs * (prev_probs.log() - current_probs)).sum(dim=-1)
        return kl.mean()


# ============================================================================
# Convenience: load-balancing loss (standalone)
# ============================================================================


def cross_jalur_load_balance_loss(
    router_logits: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """Auxiliary load-balancing loss for cross-jalur routing.

    This is the same formulation as the Switch Transformer
    load-balancing loss, provided as a standalone function for
    compatibility with other MoE implementations.

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
    assignments = probs.argmax(dim=-1)  # (B, S)
    one_hot = F.one_hot(assignments, num_experts).float()  # (B, S, E)
    f = one_hot.mean(dim=(0, 1))  # (E,)
    P = probs.mean(dim=(0, 1))  # (E,)
    return num_experts * (f * P).sum()
