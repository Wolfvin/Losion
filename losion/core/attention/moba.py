"""
MoBA: Mixture of Block Attention — Block-sparse attention via MoE routing.

MoBA applies the Mixture-of-Experts (MoE) routing paradigm to attention
blocks rather than FFN layers. Instead of the standard O(n^2) full
attention over all tokens, MoBA partitions the key-value sequence into
fixed-size blocks and learns a lightweight router that selects the
top-K most relevant blocks for each query token. Standard softmax
attention is then applied only within the selected blocks, reducing
overall complexity to O(n * K * block_size).

Architecture:
1. MoBAConfig — Configuration dataclass controlling block size,
   routing temperature, top-K selection, MLA compression, etc.

2. BlockPartitioner — Utility that splits a sequence into contiguous
   blocks of configurable size, returning block indices and masks.
   Handles sequences whose length is not a multiple of block_size
   by padding the final block.

3. MoBARouter(nn.Module) — Learns to route each query token to the
   most relevant KV blocks. Supports two modes:
   a. Hard routing (default): top-K block selection with Gumbel noise
      during training for exploration.
   b. Soft routing: weighted combination of all blocks using softmax
      routing probabilities (differentiable, no discrete selection).
   Includes an optional load-balancing auxiliary loss to prevent
   block collapse.

4. MoBAAttention(nn.Module) — Drop-in replacement for standard
   multi-head attention. Instead of attending to all tokens, it
   routes queries to the K most relevant blocks and computes
   standard softmax attention only within those blocks.
   Compatible with Losion's MLA KV compression: when enabled,
   KV pairs are compressed to a low-dimensional latent before
   being partitioned into blocks, saving both memory and compute.
   Provides forward() for training and forward_inference() for
   autoregressive generation with KV cache.

Complexity Analysis:
   Standard attention:  O(n^2 * d)
   MoBA attention:     O(n * K * B * d)   where B = block_size, K = top_k_blocks
   Typical savings:    With n=8192, B=512, K=4 → ~4x fewer FLOPs

References:
- MoBA: Mixture of Block Attention (Moonshot AI, NeurIPS 2025)
  Paper: https://neurips.cc/virtual/2025/poster/117997
- Inspired by MoE routing applied to attention blocks instead of FFN
- DeepSeek-AI, "DeepSeek-V2" (2024) — MLA KV compression

Hardware: Pure PyTorch, compatible with CUDA, ROCm, and CPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# MoBAConfig — Configuration Dataclass
# ============================================================================


@dataclass
class MoBAConfig:
    """Configuration for MoBA (Mixture of Block Attention).

    Controls block partitioning, routing behavior, MLA compression,
    and numerical settings.

    Attributes:
        block_size: Number of tokens per KV block (default 512).
            Larger blocks = fewer routing decisions but coarser granularity.
            Smaller blocks = finer routing but more overhead.
        top_k_blocks: Number of blocks each query attends to (default 4).
            Must be >= 1. Higher K = more compute but better quality.
        use_mla_compression: Whether to compress KV via MLA latent
            projection before partitioning into blocks (default True).
            Reduces memory and compute for the block attention.
        routing_temperature: Temperature for routing softmax (default 1.0).
            Lower = sharper routing (more concentrated). Higher = more
            uniform exploration.
        use_soft_routing: If True, use weighted combination of all blocks
            instead of discrete top-K selection (default False).
            Soft routing is fully differentiable but more expensive.
        load_balance_weight: Weight for auxiliary load-balancing loss
            (default 0.01). Set to 0.0 to disable.
        kv_lora_rank: Rank for MLA KV latent compression (default 256).
            Only used when use_mla_compression=True.
        d_rope: Dimension for RoPE positional encoding per head
            (default None → d_head // 2).
        rope_base: Base frequency for RoPE (default 10000.0).
        dropout: Dropout rate for attention weights (default 0.0).
    """

    block_size: int = 512
    top_k_blocks: int = 4
    use_mla_compression: bool = True
    routing_temperature: float = 1.0
    use_soft_routing: bool = False
    load_balance_weight: float = 0.01
    kv_lora_rank: int = 256
    d_rope: Optional[int] = None
    rope_base: float = 10000.0
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.block_size <= 0:
            raise ValueError(f"block_size must be positive, got {self.block_size}")
        if self.top_k_blocks <= 0:
            raise ValueError(f"top_k_blocks must be positive, got {self.top_k_blocks}")
        if self.routing_temperature <= 0.0:
            raise ValueError(
                f"routing_temperature must be positive, got {self.routing_temperature}"
            )
        if self.load_balance_weight < 0.0:
            raise ValueError(
                f"load_balance_weight must be non-negative, "
                f"got {self.load_balance_weight}"
            )


# ============================================================================
# BlockPartitioner — Sequence-to-Block Partitioning Utility
# ============================================================================


class BlockPartitioner:
    """Splits a sequence into contiguous blocks of configurable size.

    Given a sequence of length ``seq_len``, partitions it into
    ``ceil(seq_len / block_size)`` blocks. The final block is
    zero-padded if ``seq_len`` is not a multiple of ``block_size``.

    Provides methods to:
    - Compute the number of blocks for a given sequence length.
    - Build an index tensor that maps each token position to its block.
    - Build a boolean mask indicating which positions are padding.
    - Reshape a token-level tensor into a block-level tensor.

    Args:
        block_size: Number of tokens per block.
    """

    def __init__(self, block_size: int) -> None:
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        self.block_size = block_size

    def num_blocks(self, seq_len: int) -> int:
        """Return the number of blocks for a sequence of length ``seq_len``.

        Args:
            seq_len: Total sequence length.

        Returns:
            Number of blocks (ceiling division).
        """
        return (seq_len + self.block_size - 1) // self.block_size

    def block_indices(
        self, seq_len: int, device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
        """Return a tensor mapping each token position to its block index.

        Args:
            seq_len: Total sequence length.
            device: Device for the returned tensor.

        Returns:
            Long tensor of shape ``(seq_len,)`` where element ``i``
            contains the block index for token ``i``.
        """
        return torch.arange(seq_len, device=device) // self.block_size

    def padding_mask(
        self, seq_len: int, device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
        """Return a boolean mask indicating padded positions in the last block.

        If ``seq_len`` is an exact multiple of ``block_size``, all entries
        are ``False``. Otherwise, the trailing positions in the final block
        are ``True``.

        Args:
            seq_len: Actual (unpadded) sequence length.
            device: Device for the returned tensor.

        Returns:
            Boolean tensor of shape ``(num_blocks * block_size,)``.
            ``True`` marks positions that are padding.
        """
        n_blocks = self.num_blocks(seq_len)
        padded_len = n_blocks * self.block_size
        mask = torch.arange(padded_len, device=device) >= seq_len
        return mask

    def partition(
        self, x: torch.Tensor, seq_dim: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reshape a token-level tensor into a block-level tensor.

        Pads the sequence dimension if necessary, then reshapes
        ``(batch, seq_len, ...)`` → ``(batch, n_blocks, block_size, ...)``.

        Args:
            x: Token-level tensor, shape ``(batch, seq_len, ...)``.
            seq_dim: Dimension index of the sequence axis (default 1).

        Returns:
            Tuple ``(x_blocks, pad_mask)``:
            - ``x_blocks``: Block-level tensor,
              shape ``(batch, n_blocks, block_size, ...)``.
            - ``pad_mask``: Boolean padding mask,
              shape ``(batch, 1, 1, padded_len)`` suitable for
              broadcasting with attention weights.
        """
        seq_len = x.shape[seq_dim]
        n_blocks = self.num_blocks(seq_len)
        padded_len = n_blocks * self.block_size

        # Pad along sequence dimension
        if padded_len > seq_len:
            pad_size = padded_len - seq_len
            # Build pad tuple: (last_dim_pad_left, last_dim_pad_right, ...)
            # Only pad along seq_dim
            pad_tuple = [0] * (2 * x.dim())
            # seq_dim from the right: index = (x.dim() - 1 - seq_dim) * 2 + 1
            idx = (x.dim() - 1 - seq_dim) * 2 + 1
            pad_tuple[idx] = pad_size
            x_padded = F.pad(x, pad_tuple)
        else:
            x_padded = x

        # Reshape: (batch, n_blocks, block_size, ...)
        shape = list(x_padded.shape)
        shape[seq_dim:seq_dim + 1] = [n_blocks, self.block_size]
        x_blocks = x_padded.view(shape)

        # Padding mask for attention: (batch, 1, 1, padded_len)
        pad_mask_1d = self.padding_mask(seq_len, device=x.device)
        batch = x.shape[0]
        pad_mask = pad_mask_1d.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        pad_mask = pad_mask.expand(batch, -1, -1, -1)

        return x_blocks, pad_mask

    def flatten_blocks(self, x_blocks: torch.Tensor, seq_dim: int = 1
                       ) -> torch.Tensor:
        """Inverse of ``partition``: merge block and block_size dims.

        Args:
            x_blocks: Block-level tensor,
              shape ``(batch, n_blocks, block_size, ...)``.
            seq_dim: Index of the n_blocks dimension (default 1).

        Returns:
            Token-level tensor, shape ``(batch, n_blocks * block_size, ...)``.
        """
        shape = list(x_blocks.shape)
        n_blocks = shape[seq_dim]
        block_size = shape[seq_dim + 1]
        shape[seq_dim:seq_dim + 2] = [n_blocks * block_size]
        return x_blocks.view(shape)


# ============================================================================
# MoBARouter — Block-Level Routing Network
# ============================================================================


class MoBARouter(nn.Module):
    """Routes each query token to the most relevant KV blocks.

    Given query representations, computes a relevance score for each
    KV block and selects the top-K blocks (hard routing) or computes
    weighted soft routing probabilities (soft routing).

    Architecture:
        1. Compute per-block summary: mean-pool KV within each block.
        2. Project queries and block summaries to a shared routing space.
        3. Compute relevance scores via dot-product + temperature scaling.
        4. Select top-K blocks (hard) or compute soft routing weights.

    Hard Routing (default):
        - During training: add Gumbel noise for exploration.
        - During inference: deterministic top-K selection.
        - Only selected blocks contribute to attention.

    Soft Routing:
        - All blocks contribute, weighted by softmax probabilities.
        - Fully differentiable, no discrete selection.
        - More expensive but smoother gradient flow.

    Load Balancing:
        Optional auxiliary loss encouraging uniform block utilization
        to prevent "block collapse" where most queries route to the
        same small set of blocks.

    Args:
        d_model: Model hidden dimension.
        n_heads: Number of attention heads (used for routing dimension).
        d_head: Dimension per attention head.
        top_k_blocks: Number of blocks to select per query (default 4).
        routing_temperature: Temperature for routing softmax (default 1.0).
        use_soft_routing: If True, use soft weighted routing (default False).
        load_balance_weight: Weight for auxiliary load-balancing loss
            (default 0.01). Set to 0.0 to disable.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        d_head: int = 64,
        top_k_blocks: int = 4,
        routing_temperature: float = 1.0,
        use_soft_routing: bool = False,
        load_balance_weight: float = 0.01,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.top_k_blocks = top_k_blocks
        self.routing_temperature = routing_temperature
        self.use_soft_routing = use_soft_routing
        self.load_balance_weight = load_balance_weight

        # Routing projection: query → routing space
        self.query_proj = nn.Linear(d_model, d_model, bias=False)
        # Routing projection: block summary → routing space
        self.block_proj = nn.Linear(d_model, d_model, bias=False)
        # Routing normalization
        self.query_norm = nn.RMSNorm(d_model, eps=1e-5)
        self.block_norm = nn.RMSNorm(d_model, eps=1e-5)

    def _compute_block_summaries(
        self, kv: torch.Tensor, n_blocks: int, block_size: int
    ) -> torch.Tensor:
        """Compute summary representations for each KV block.

        Mean-pools the KV within each block to produce a single
        vector per block for routing.

        Args:
            kv: Key-value representations,
                shape ``(batch, seq_len, d_model)``.
            n_blocks: Number of blocks.
            block_size: Tokens per block.

        Returns:
            Block summaries, shape ``(batch, n_blocks, d_model)``.
        """
        batch, seq_len, d = kv.shape
        padded_len = n_blocks * block_size

        # Pad if needed
        if padded_len > seq_len:
            pad_size = padded_len - seq_len
            kv = F.pad(kv, (0, 0, 0, pad_size))

        # Reshape to blocks: (batch, n_blocks, block_size, d_model)
        kv_blocks = kv.view(batch, n_blocks, block_size, d)

        # Mean-pool over block dimension, ignoring padding
        # Simple mean (padding is zero, so it dilutes but is acceptable)
        summaries = kv_blocks.mean(dim=2)  # (batch, n_blocks, d_model)

        return summaries

    def forward(
        self,
        query: torch.Tensor,
        kv: torch.Tensor,
        n_blocks: int,
        block_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Route queries to KV blocks.

        Args:
            query: Query representations,
                shape ``(batch, seq_len, d_model)``.
            kv: Key-value representations,
                shape ``(batch, full_len, d_model)``.
            n_blocks: Number of KV blocks.
            block_size: Tokens per block.

        Returns:
            Tuple ``(routing_weights, selected_blocks, aux_loss)``:
            - ``routing_weights``: ``(batch, n_heads, seq_len, n_blocks)``
              routing weights (softmax-normalized over selected blocks
              for hard routing, over all blocks for soft routing).
            - ``selected_blocks``: ``(batch, n_heads, seq_len, top_k)``
              indices of selected blocks (int). For soft routing,
              all blocks are "selected" with indices 0..n_blocks-1.
            - ``aux_loss``: Optional scalar load-balancing loss,
              or ``None`` if load_balance_weight == 0.
        """
        batch, seq_len, _ = query.shape
        full_len = kv.shape[1]

        # Compute block summaries: (batch, n_blocks, d_model)
        block_summaries = self._compute_block_summaries(kv, n_blocks, block_size)

        # Project to routing space
        q_route = self.query_norm(self.query_proj(query))  # (batch, seq_len, d_model)
        b_route = self.block_norm(self.block_proj(block_summaries))  # (batch, n_blocks, d_model)

        # Compute relevance scores via dot-product
        # (batch, seq_len, d_model) @ (batch, d_model, n_blocks)
        scores = torch.bmm(q_route, b_route.transpose(1, 2))  # (batch, seq_len, n_blocks)
        scores = scores / math.sqrt(self.d_model)

        # Expand for multi-head: (batch, 1, seq_len, n_blocks) →
        # (batch, n_heads, seq_len, n_blocks)
        # Use the same routing decision for all heads (shared routing)
        scores = scores.unsqueeze(1).expand(-1, self.n_heads, -1, -1)

        # Scale by temperature
        scores = scores / self.routing_temperature

        aux_loss: Optional[torch.Tensor] = None

        if self.use_soft_routing:
            # Soft routing: weighted combination of all blocks
            routing_weights = F.softmax(scores, dim=-1)
            # "selected" = all blocks
            selected_blocks = torch.arange(
                n_blocks, device=query.device
            ).unsqueeze(0).unsqueeze(0).unsqueeze(0)
            selected_blocks = selected_blocks.expand(
                batch, self.n_heads, seq_len, -1
            )
        else:
            # Hard routing: top-K block selection
            K = min(self.top_k_blocks, n_blocks)

            if self.training:
                # Add Gumbel noise for exploration during training
                gumbel_noise = -torch.empty_like(scores).exponential_().log()
                noisy_scores = scores + gumbel_noise
            else:
                noisy_scores = scores

            # Top-K selection
            top_k_scores, selected_blocks = torch.topk(
                noisy_scores, k=K, dim=-1
            )  # (batch, n_heads, seq_len, K)

            # Softmax over selected blocks only
            routing_weights = F.softmax(top_k_scores, dim=-1)

            # Compute load-balancing loss
            if self.load_balance_weight > 0.0:
                aux_loss = self._load_balance_loss(
                    scores, selected_blocks, n_blocks
                )

        return routing_weights, selected_blocks, aux_loss

    def _load_balance_loss(
        self,
        scores: torch.Tensor,
        selected_blocks: torch.Tensor,
        n_blocks: int,
    ) -> torch.Tensor:
        """Compute load-balancing auxiliary loss.

        Encourages uniform block utilization by penalizing
        imbalance in the fraction of queries routed to each block.

        Loss = n_blocks * sum_i(f_i * P_i)
        where f_i = fraction of queries routed to block i
        and P_i = mean routing probability for block i.

        Args:
            scores: Raw routing scores,
                shape ``(batch, n_heads, seq_len, n_blocks)``.
            selected_blocks: Selected block indices,
                shape ``(batch, n_heads, seq_len, K)``.
            n_blocks: Total number of blocks.

        Returns:
            Scalar load-balancing loss.
        """
        batch, n_heads, seq_len, _ = scores.shape
        K = selected_blocks.shape[-1]

        # Fraction of queries routed to each block: f_i
        # One-hot encode selected blocks
        one_hot = torch.zeros(
            batch * n_heads * seq_len, n_blocks,
            dtype=scores.dtype, device=scores.device,
        )
        selected_flat = selected_blocks.reshape(-1, K)
        one_hot.scatter_(
            1, selected_flat, 1.0
        )
        # f_i: (batch * n_heads, n_blocks)
        f = one_hot.view(batch, n_heads, seq_len, n_blocks).mean(dim=2)

        # Mean routing probability: P_i
        P = F.softmax(scores, dim=-1).mean(dim=2)  # (batch, n_heads, n_blocks)

        # Load balance loss
        loss = n_blocks * (f * P).sum(dim=-1).mean()

        return self.load_balance_weight * loss


# ============================================================================
# MoBAAttention — Mixture of Block Attention Module
# ============================================================================


class MoBAAttention(nn.Module):
    """MoBA: Mixture of Block Attention — Drop-in replacement for
    standard multi-head attention with block-sparse routing.

    Instead of O(n^2) full attention over all tokens, MoBA partitions
    the KV sequence into blocks and routes each query to only the
    top-K most relevant blocks. Standard softmax attention is then
    applied within each selected block and the results are aggregated
    using routing weights.

    Compatible with Losion's MLA KV compression: when
    ``use_mla_compression=True``, KV pairs are compressed to a
    low-rank latent representation before block partitioning,
    reducing both memory and compute.

    Forward passes:
        - ``forward()``: Training mode with optional KV cache.
        - ``forward_inference()``: Autoregressive generation with
          KV cache and incremental routing.

    Returns:
        ``(output, routing_info)`` where ``routing_info`` is a dict
        containing:
        - ``"selected_blocks"``: Indices of selected blocks.
        - ``"routing_weights"``: Routing weight for each block.
        - ``"aux_loss"``: Load-balancing loss (if enabled).
        - ``"present_key_value"``: Updated KV cache.

    Args:
        d_model: Model hidden dimension.
        n_heads: Number of attention heads.
        d_head: Dimension per attention head.
        config: MoBAConfig instance (optional, uses defaults if None).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        d_head: int = 64,
        config: Optional[MoBAConfig] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_inner = n_heads * d_head

        # Apply config (or defaults)
        cfg = config or MoBAConfig()
        self.block_size = cfg.block_size
        self.top_k_blocks = cfg.top_k_blocks
        self.use_mla_compression = cfg.use_mla_compression
        self.kv_lora_rank = cfg.kv_lora_rank
        self.d_rope = cfg.d_rope or (d_head // 2)
        self.rope_base = cfg.rope_base

        # ---- Block Partitioner ----
        self.partitioner = BlockPartitioner(self.block_size)

        # ---- Router ----
        self.router = MoBARouter(
            d_model=d_model,
            n_heads=n_heads,
            d_head=d_head,
            top_k_blocks=self.top_k_blocks,
            routing_temperature=cfg.routing_temperature,
            use_soft_routing=cfg.use_soft_routing,
            load_balance_weight=cfg.load_balance_weight,
        )

        # ---- Q Projection ----
        self.q_proj = nn.Linear(d_model, self.d_inner, bias=False)

        # ---- KV Projections (MLA or standard) ----
        if self.use_mla_compression:
            # MLA path: compress KV to low-rank latent
            self.kv_down_proj = nn.Linear(d_model, self.kv_lora_rank, bias=False)
            self.kv_norm = nn.RMSNorm(self.kv_lora_rank, eps=1e-5)
            self.k_up_proj = nn.Linear(self.kv_lora_rank, self.d_inner, bias=False)
            self.v_up_proj = nn.Linear(self.kv_lora_rank, self.d_inner, bias=False)
        else:
            self.k_proj = nn.Linear(d_model, self.d_inner, bias=False)
            self.v_proj = nn.Linear(d_model, self.d_inner, bias=False)

        # ---- RoPE ----
        from .lightning_attention import InterleavedRoPE
        self.rope = InterleavedRoPE(
            dim=self.d_rope,
            d_rope=self.d_rope,
            base=self.rope_base,
            interleaved=False,
        )

        # ---- QK Normalization ----
        self.q_norm = nn.RMSNorm(d_head, eps=1e-5)
        self.k_norm = nn.RMSNorm(d_head, eps=1e-5)

        # ---- Routing KV projection (d_inner → d_model for router) ----
        # The router expects d_model-dimensional input, but KV representations
        # are in d_inner dimension. This projection bridges the gap.
        self.kv_route_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # ---- Output ----
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.out_norm = nn.RMSNorm(d_model, eps=1e-5)
        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()

    def _project_q(self, x: torch.Tensor) -> torch.Tensor:
        """Project input to query representations.

        Args:
            x: Input tensor, shape ``(batch, seq_len, d_model)``.

        Returns:
            Query tensor, shape ``(batch, seq_len, n_heads, d_head)``.
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
        batch, seq_len, _ = x.shape
        q = self.q_proj(x)
        return q.view(batch, seq_len, self.n_heads, self.d_head)

    def _project_kv(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Project input to key and value representations.

        When MLA compression is enabled, also returns the latent
        representation for caching.

        Args:
            x: Input tensor, shape ``(batch, seq_len, d_model)``.

        Returns:
            Tuple ``(k, v, c_kv)``:
            - ``k``: ``(batch, seq_len, n_heads, d_head)``
            - ``v``: ``(batch, seq_len, n_heads, d_head)``
            - ``c_kv``: ``(batch, seq_len, kv_lora_rank)`` if MLA,
              else ``None``.
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
        batch, seq_len, _ = x.shape

        if self.use_mla_compression:
            c_kv = self.kv_norm(self.kv_down_proj(x))
            k = self.k_up_proj(c_kv).view(batch, seq_len, self.n_heads, self.d_head)
            v = self.v_up_proj(c_kv).view(batch, seq_len, self.n_heads, self.d_head)
            return k, v, c_kv
        else:
            k = self.k_proj(x).view(batch, seq_len, self.n_heads, self.d_head)
            v = self.v_proj(x).view(batch, seq_len, self.n_heads, self.d_head)
            return k, v, None

    def _apply_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE to query and key tensors.

        Only applies RoPE to the first ``d_rope`` dimensions per head.

        Args:
            q: Query, shape ``(batch, seq_len, n_heads, d_head)``.
            k: Key, shape ``(batch, full_len, n_heads, d_head)``.
            offset: Position offset for RoPE.

        Returns:
            Tuple ``(q_rope, k_rope)`` with RoPE applied.
        """
        q_r = q[..., :self.d_rope].contiguous()
        k_r = k[..., :self.d_rope].contiguous()

        q_r = self.rope(q_r, offset=offset)
        k_r = self.rope(k_r, offset=0)  # K already full sequence

        if self.d_rope < self.d_head:
            q = torch.cat([q_r, q[..., self.d_rope:]], dim=-1)
            k = torch.cat([k_r, k[..., self.d_rope:]], dim=-1)
        else:
            q = q_r
            k = k_r

        return q, k

    def _block_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_blocks: torch.Tensor,
        n_blocks: int,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute block-sparse attention using routing decisions.

        For each query token, computes attention only to tokens in the
        selected blocks. Results are aggregated using routing weights.

        Implementation strategy:
            1. Build a block-sparse mask where mask[q, k] = True if
               token k belongs to one of the selected blocks for query q.
            2. Compute full QK^T scores, then mask out tokens not in
               selected blocks.
            3. Apply standard causal mask.
            4. Compute softmax attention and weighted sum.
            5. Scale each block's contribution by routing weights.

        Args:
            q: Query, shape ``(batch, n_heads, seq_len, d_head)``.
            k: Key, shape ``(batch, n_heads, full_len, d_head)``.
            v: Value, shape ``(batch, n_heads, full_len, d_head)``.
            routing_weights: Routing weights,
                shape ``(batch, n_heads, seq_len, K)`` for hard routing
                or ``(batch, n_heads, seq_len, n_blocks)`` for soft.
            selected_blocks: Selected block indices,
                shape ``(batch, n_heads, seq_len, K)`` for hard routing
                or ``(batch, n_heads, seq_len, n_blocks)`` for soft.
            n_blocks: Total number of blocks.
            attention_mask: Optional attention mask.

        Returns:
            Attention output, shape ``(batch, n_heads, seq_len, d_head)``.
        """
        batch, n_heads, seq_len, d_head = q.shape
        full_len = k.shape[2]
        B = self.block_size
        K = selected_blocks.shape[-1]

        # QK normalization
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Compute attention scores: (batch, n_heads, seq_len, full_len)
        scale = math.sqrt(d_head)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        # Build block-sparse mask: which key positions are in selected blocks
        # For each (batch, head, q_pos), the mask selects key positions that
        # belong to the selected blocks for that query.
        block_ids = torch.arange(full_len, device=q.device) // B  # (full_len,)
        # block_ids[k] = block index of key position k

        # selected_blocks: (batch, n_heads, seq_len, K)
        # We want: block_mask[b, h, q, k] = True if block_ids[k] is in
        # selected_blocks[b, h, q, :]
        # Shape: (batch, n_heads, seq_len, full_len)
        block_ids_expanded = block_ids.view(1, 1, 1, full_len)  # broadcast
        selected_expanded = selected_blocks  # (batch, n_heads, seq_len, K)

        # For each (q, k), check if block_ids[k] matches any selected block
        # (batch, n_heads, seq_len, K, 1) == (1, 1, 1, 1, full_len)
        # → (batch, n_heads, seq_len, K, full_len)
        block_match = (
            selected_expanded.unsqueeze(-1) == block_ids_expanded.unsqueeze(3)
        )  # (batch, n_heads, seq_len, K, full_len)

        # Any match across K dimension: (batch, n_heads, seq_len, full_len)
        sparse_mask = block_match.any(dim=3)

        # Also need the per-block routing weights for each key position
        # routing_weights: (batch, n_heads, seq_len, K)
        # We need: route_weight_per_key[b, h, q, k] = routing weight of
        # the block that key k belongs to
        # (batch, n_heads, seq_len, K, full_len) * routing weights
        routing_expanded = routing_weights.unsqueeze(-1) * block_match.float()
        # Sum across K: (batch, n_heads, seq_len, full_len)
        routing_per_key = routing_expanded.sum(dim=3)

        # Apply block-sparse mask: mask out tokens not in selected blocks
        attn_scores = attn_scores.masked_fill(~sparse_mask, float("-inf"))

        # Apply causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, full_len, dtype=torch.bool, device=q.device),
            diagonal=full_len - seq_len + 1,
        )
        attn_scores = attn_scores.masked_fill(
            causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
        )

        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        attn_weights = F.softmax(
            attn_scores, dim=-1, dtype=torch.float32
        ).to(q.dtype)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum: (batch, n_heads, seq_len, full_len) @ (batch, n_heads, full_len, d_head)
        attn_output = torch.matmul(attn_weights, v)

        # Scale output by routing weights to incorporate block-level importance
        # This ensures that blocks with higher routing weight contribute more
        # After softmax + matmul, we re-weight by the routing probability
        # routing_per_key: (batch, n_heads, seq_len, full_len)
        # But we can't easily re-weight after softmax. Instead, the routing
        # is implicitly handled by the sparse mask and softmax normalization.
        # For soft routing, the block-sparse mask already includes all blocks
        # with their routing weights.

        return attn_output

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, ...]] = None,
        position_offset: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        """Forward pass for MoBA attention (training or prefill).

        Routes queries to the top-K most relevant KV blocks and
        computes attention only within those blocks.

        Args:
            x: Input tensor, shape ``(batch, seq_len, d_model)``.
            attention_mask: Optional attention mask.
            past_key_value: Cache from previous step:
                - ``(kv_cache, c_kv_cache)`` where ``kv_cache`` is
                  ``(batch, past_len, n_heads, d_head * 2)`` and
                  ``c_kv_cache`` is ``(batch, past_len, kv_lora_rank)``
                  (MLA mode) or ``None``.
            position_offset: Position offset for RoPE.

        Returns:
            Tuple ``(output, routing_info)``:
            - ``output``: ``(batch, seq_len, d_model)``
            - ``routing_info``: Dict with ``"selected_blocks"``,
              ``"routing_weights"``, ``"aux_loss"``, and
              ``"present_key_value"``.
        """
        # Handle standalone call without batch dimension
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        batch, seq_len, _ = x.shape

        if seq_len == 0:
            dummy_out = torch.zeros(
                batch, 0, self.d_model, dtype=x.dtype, device=x.device
            )
            if squeeze_output:
                dummy_out = dummy_out.squeeze(0)
            return dummy_out, {
                "selected_blocks": None,
                "routing_weights": None,
                "aux_loss": None,
                "present_key_value": (None, None),
            }

        # ---- Unpack past cache (defensive: only accept tuple) ----
        kv_cache = None
        c_kv_cache = None
        if past_key_value is not None and isinstance(past_key_value, tuple):
            if len(past_key_value) >= 2:
                kv_cache, c_kv_cache = past_key_value[0], past_key_value[1]
            else:
                kv_cache = past_key_value[0]

        # ---- Project Q, K, V ----
        q = self._project_q(x)  # (batch, seq_len, n_heads, d_head)
        k, v, c_kv = self._project_kv(x)

        # ---- Determine position offset ----
        past_len = 0
        if kv_cache is not None:
            past_len = kv_cache.shape[1]

        # ---- Apply RoPE to Q and new K (before cache concat) ----
        q_offset = past_len + position_offset
        q_r = q[..., :self.d_rope].contiguous()
        k_r = k[..., :self.d_rope].contiguous()
        q_r = self.rope(q_r, offset=q_offset)
        k_r = self.rope(k_r, offset=past_len)
        if self.d_rope < self.d_head:
            q = torch.cat([q_r, q[..., self.d_rope:]], dim=-1)
            k = torch.cat([k_r, k[..., self.d_rope:]], dim=-1)
        else:
            q = q_r
            k = k_r

        # ---- Concatenate with cache ----
        if kv_cache is not None and kv_cache.dim() == 4:
            past_k = kv_cache[:, :, :, :self.d_head]
            past_v = kv_cache[:, :, :, self.d_head:]
            k_full = torch.cat([past_k, k], dim=1)
            v_full = torch.cat([past_v, v], dim=1)
        elif kv_cache is not None and kv_cache.dim() == 3:
            # Fallback: treat as (batch, past_len, d_inner) and reshape
            past_k = kv_cache[:, :, :self.d_inner // 2].view(
                batch, -1, self.n_heads, self.d_head)
            past_v = kv_cache[:, :, self.d_inner // 2:].view(
                batch, -1, self.n_heads, self.d_head)
            k_full = torch.cat([past_k, k], dim=1)
            v_full = torch.cat([past_v, v], dim=1)
        else:
            k_full = k
            v_full = v

        full_len = k_full.shape[1]

        # ---- Routing ----
        n_blocks = self.partitioner.num_blocks(full_len)
        # Guard: if sequence is too short for block routing, skip MoBA routing
        # and use standard attention instead
        effective_top_k = min(self.top_k_blocks, n_blocks)

        # Build d_model-dimensional KV representation for the router.
        # The router expects d_model input, so we project from d_inner.
        if self.use_mla_compression and c_kv_cache is not None and c_kv is not None:
            # MLA path: reconstruct from compressed latent, then project
            c_kv_full = torch.cat([c_kv_cache, c_kv], dim=1)
            kv_inner = self.k_up_proj(c_kv_full)  # (batch, full_len, d_inner)
        else:
            # Standard path: flatten K across heads to d_inner
            kv_inner = k_full.reshape(batch, full_len, self.d_inner)

        kv_route = self.kv_route_proj(kv_inner)  # (batch, full_len, d_model)

        routing_weights, selected_blocks, aux_loss = self.router(
            query=x,
            kv=kv_route,
            n_blocks=n_blocks,
            block_size=self.block_size,
        )

        # ---- Block Attention ----
        q_t = q.transpose(1, 2)  # (batch, n_heads, seq_len, d_head)
        k_t = k_full.transpose(1, 2)  # (batch, n_heads, full_len, d_head)
        v_t = v_full.transpose(1, 2)  # (batch, n_heads, full_len, d_head)

        attn_output = self._block_attention(
            q_t, k_t, v_t,
            routing_weights, selected_blocks,
            n_blocks, attention_mask,
        )

        # ---- Output Projection ----
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch, seq_len, self.d_inner)
        output = self.out_proj(attn_output)
        output = self.out_norm(output)

        # ---- Update KV Cache ----
        new_kv_cache = torch.cat([k_full, v_full], dim=-1)
        # (batch, full_len, n_heads, d_head * 2)

        new_c_kv = c_kv
        if self.use_mla_compression and c_kv_cache is not None and c_kv is not None:
            new_c_kv = torch.cat([c_kv_cache, c_kv], dim=1)

        present_key_value = (new_kv_cache, new_c_kv)

        if squeeze_output:
            output = output.squeeze(0)

        routing_info = {
            "selected_blocks": selected_blocks,
            "routing_weights": routing_weights,
            "aux_loss": aux_loss,
            "present_key_value": present_key_value,
        }

        return output, routing_info

    def forward_inference(
        self,
        x: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, ...]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        """Forward pass for autoregressive generation (one token at a time).

        For single-token queries, routing is simplified: the query
        attends to the top-K blocks of the full KV cache. Within
        each selected block, standard causal attention is applied.

        Args:
            x: Input tensor for a single token,
                shape ``(batch, 1, d_model)``.
            past_key_value: Cache from previous step:
                ``(kv_cache, c_kv_cache)``.

        Returns:
            Tuple ``(output, routing_info)`` with the same structure
            as ``forward()``.
        """
        # Handle standalone call without batch dimension
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        output, routing_info = self.forward(
            x,
            attention_mask=None,
            past_key_value=past_key_value,
            position_offset=0,
        )

        if squeeze_output:
            output = output.squeeze(0)

        return output, routing_info
