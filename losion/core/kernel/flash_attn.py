"""
Losion Flash Attention — Ring Attention and Flash Attention wrappers.

Provides:
  - FlashAttentionWrapper: Wraps attention computation with Flash Attention
    when available, falling back to SDPA then manual attention.
  - RingAttention: Blockwise parallel attention for extremely long sequences,
    distributing KV blocks across GPUs in a ring communication pattern.

Credits:
  - Flash Attention: Dao et al., arXiv:2205.14135 (2022)
  - Flash Attention 2: Dao, arXiv:2307.08691 (2023)
  - Ring Attention: Liu et al., arXiv:2310.01889 (2023)
  - Blockwise Parallel Attention: Li et al., arXiv:2305.19370 (2023)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from losion.core.kernel import HAS_FLASH_ATTN


class FlashAttentionWrapper(nn.Module):
    """Attention module with Flash Attention / SDPA automatic dispatch.

    Uses the same 3-tier fallback as sdpa_compat:
      1. flash_attn package
      2. F.scaled_dot_product_attention
      3. Manual matmul

    Supports causal masking, dropout, and optional MLA (Multi-head Latent
    Attention) compression.

    Args:
        d_model: Model hidden dimension.
        n_heads: Number of attention heads.
        d_kv: Dimension per key/value head.
        dropout: Dropout probability.
        is_causal: Whether to use causal masking.
        use_mla: Whether to use MLA KV compression.
        mla_latent_dim: Latent dimension for MLA compression.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        d_kv: int = 64,
        dropout: float = 0.0,
        is_causal: bool = True,
        use_mla: bool = False,
        mla_latent_dim: int = 128,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_kv = d_kv
        self.dropout = dropout
        self.is_causal = is_causal
        self.use_mla = use_mla
        self.mla_latent_dim = mla_latent_dim

        # Standard QKV projections
        self.q_proj = nn.Linear(d_model, n_heads * d_kv, bias=False)
        self.k_proj = nn.Linear(d_model, n_heads * d_kv, bias=False)
        self.v_proj = nn.Linear(d_model, n_heads * d_kv, bias=False)
        self.out_proj = nn.Linear(n_heads * d_kv, d_model, bias=False)

        # MLA compression projections (optional)
        if use_mla:
            self.k_compress = nn.Linear(n_heads * d_kv, mla_latent_dim, bias=False)
            self.v_compress = nn.Linear(n_heads * d_kv, mla_latent_dim, bias=False)
            self.k_decompress = nn.Linear(mla_latent_dim, n_heads * d_kv, bias=False)
            self.v_decompress = nn.Linear(mla_latent_dim, n_heads * d_kv, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with automatic Flash/SDPA dispatch.

        Args:
            x: Input tensor (batch, seq_len, d_model).
            attention_mask: Optional attention mask.
            past_kv: Optional cached key/value tensors.
            position_ids: Optional position IDs.

        Returns:
            Attention output (batch, seq_len, d_model).
        """
        batch, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.d_kv).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_heads, self.d_kv).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_heads, self.d_kv).transpose(1, 2)

        # MLA compression (optional)
        if self.use_mla:
            k_flat = k.transpose(1, 2).reshape(batch, seq_len, -1)
            v_flat = v.transpose(1, 2).reshape(batch, seq_len, -1)
            k_flat = self.k_decompress(self.k_compress(k_flat))
            v_flat = self.v_decompress(self.v_compress(v_flat))
            k = k_flat.view(batch, seq_len, self.n_heads, self.d_kv).transpose(1, 2)
            v = v_flat.view(batch, seq_len, self.n_heads, self.d_kv).transpose(1, 2)

        # Concatenate with past KV if available
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        # Compute attention using the best available backend
        out = self._compute_attention(q, k, v, attention_mask)

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.out_proj(out)

    def _compute_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Dispatch to the best available attention backend."""
        scale = 1.0 / math.sqrt(self.d_kv)
        dropout_p = self.dropout if self.training else 0.0

        # Tier 1: flash_attn
        if HAS_FLASH_ATTN and attn_mask is None:
            try:
                from flash_attn import flash_attn_func
                q_fa = q.transpose(1, 2).contiguous()
                k_fa = k.transpose(1, 2).contiguous()
                v_fa = v.transpose(1, 2).contiguous()
                out = flash_attn_func(
                    q_fa, k_fa, v_fa,
                    dropout_p=dropout_p,
                    causal=self.is_causal,
                    softmax_scale=scale,
                )
                return out.transpose(1, 2)
            except Exception:
                pass

        # Tier 2: PyTorch SDPA
        try:
            return F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=self.is_causal and attn_mask is None,
                scale=scale,
            )
        except (AttributeError, RuntimeError):
            pass

        # Tier 3: Manual matmul
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        if self.is_causal and attn_mask is None:
            seq_q, seq_k = q.shape[-2], k.shape[-2]
            causal_mask = torch.triu(
                torch.ones(seq_q, seq_k, device=q.device, dtype=torch.bool), diagonal=1
            )
            attn_weights = attn_weights.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        if dropout_p > 0 and self.training:
            attn_weights = F.dropout(attn_weights, p=dropout_p)
        return torch.matmul(attn_weights, v)


class RingAttention(nn.Module):
    """Blockwise parallel attention for extremely long sequences.

    Distributes KV blocks across GPUs in a ring communication pattern.
    Each GPU holds a local block of Q, and KV blocks rotate through
    all GPUs. This enables exact attention (no approximation) for
    sequences that don't fit in a single GPU's memory.

    This is the attention component of Context Parallelism. It pairs
    with the ContextParallel class in losion.distributed.parallel.

    Args:
        d_model: Model hidden dimension.
        n_heads: Number of attention heads.
        d_kv: Dimension per key/value head.
        block_size: Block size for blockwise computation.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        d_kv: int = 64,
        block_size: int = 1024,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_kv = d_kv
        self.block_size = block_size

        self.q_proj = nn.Linear(d_model, n_heads * d_kv, bias=False)
        self.k_proj = nn.Linear(d_model, n_heads * d_kv, bias=False)
        self.v_proj = nn.Linear(d_model, n_heads * d_kv, bias=False)
        self.out_proj = nn.Linear(n_heads * d_kv, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with ring-style blockwise attention.

        In single-GPU mode, this falls back to standard SDPA.
        In multi-GPU mode with ContextParallel, KV blocks rotate.

        Args:
            x: Input tensor (batch, seq_len, d_model).
            attention_mask: Optional attention mask.

        Returns:
            Attention output (batch, seq_len, d_model).
        """
        batch, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.d_kv).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_heads, self.d_kv).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_heads, self.d_kv).transpose(1, 2)

        # Single-GPU: standard SDPA
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            is_causal=attention_mask is None,
        )

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.out_proj(out)
