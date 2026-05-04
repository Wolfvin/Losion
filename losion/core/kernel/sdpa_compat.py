"""
Losion SDPA Compatibility Layer — Unified Attention with 3-tier Fallback.

Provides a single interface for attention that automatically selects the best
available backend:

  1. flash_attn package (fastest, requires flash-attn pip package)
  2. F.scaled_dot_product_attention (PyTorch 2.0+, auto-dispatches Flash/MemEff/Math)
  3. Manual matmul fallback (universal, always works)

All attention layers in Losion should use sdpa_attention() instead of
hand-rolled attention. This ensures optimal performance on every hardware
without code changes.

Credits:
  - Flash Attention: Dao et al., arXiv:2205.14135 (2022)
  - PyTorch SDPA: torch.nn.functional.scaled_dot_product_attention (2023)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from losion.core.kernel import HAS_FLASH_ATTN


# ============================================================================
# Core SDPA Attention Function
# ============================================================================


def sdpa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = True,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Compute scaled dot-product attention with automatic best backend.

    Fallback hierarchy:
      1. flash_attn package → fastest, supports variable length + causal
      2. F.scaled_dot_product_attention → PyTorch native, auto Flash/MemEff/Math
      3. Manual QK^T matmul → always works, slower

    Args:
        q: Query tensor (batch, n_heads, seq_len, d_kv).
        k: Key tensor (batch, n_heads, seq_len, d_kv).
        v: Value tensor (batch, n_heads, seq_len, d_kv).
        attn_mask: Optional attention mask.
        dropout_p: Dropout probability.
        is_causal: Whether to use causal masking.
        scale: Scale factor (default: 1/sqrt(d_kv)).

    Returns:
        Attention output tensor (batch, n_heads, seq_len, d_kv).
    """
    d_kv = q.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(d_kv)

    # ---- Tier 1: flash_attn package ----
    if HAS_FLASH_ATTN and attn_mask is None:
        try:
            from flash_attn import flash_attn_func

            # flash_attn expects (batch, seq_len, n_heads, d_kv)
            q_fa = q.transpose(1, 2).contiguous()
            k_fa = k.transpose(1, 2).contiguous()
            v_fa = v.transpose(1, 2).contiguous()

            out = flash_attn_func(
                q_fa, k_fa, v_fa,
                dropout_p=dropout_p if torch.is_grad_enabled() else 0.0,
                causal=is_causal,
                softmax_scale=scale,
            )
            return out.transpose(1, 2)  # back to (batch, n_heads, seq, d_kv)
        except Exception:
            pass  # Fall through to Tier 2

    # ---- Tier 2: F.scaled_dot_product_attention ----
    try:
        return F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=dropout_p if torch.is_grad_enabled() else 0.0,
            is_causal=is_causal and attn_mask is None,
            scale=scale,
        )
    except (AttributeError, RuntimeError):
        pass  # Fall through to Tier 3

    # ---- Tier 3: Manual matmul attention ----
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

    if is_causal and attn_mask is None:
        seq_len = q.shape[-2]
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool), diagonal=1
        )
        attn_weights = attn_weights.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    if attn_mask is not None:
        attn_weights = attn_weights + attn_mask

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)

    if dropout_p > 0 and torch.is_grad_enabled():
        attn_weights = F.dropout(attn_weights, p=dropout_p)

    return torch.matmul(attn_weights, v)


# ============================================================================
# SDPACompat — Drop-in replacement for manual attention modules
# ============================================================================


class SDPACompat(nn.Module):
    """SDPA-compatible attention wrapper.

    Replaces hand-rolled attention implementations with the unified
    sdpa_attention function. Supports all standard attention configurations
    including causal masking, dropout, and custom attention masks.

    This module handles the QKV projection → SDPA → output projection pipeline.

    Args:
        d_model: Model hidden dimension.
        n_heads: Number of attention heads.
        d_kv: Dimension per key/value head (default: d_model // n_heads).
        dropout: Dropout probability.
        is_causal: Whether to use causal masking by default.
        bias: Whether to use bias in projections.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        d_kv: Optional[int] = None,
        dropout: float = 0.0,
        is_causal: bool = True,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_kv = d_kv or (d_model // n_heads)
        self.dropout = dropout
        self.is_causal = is_causal

        self.q_proj = nn.Linear(d_model, n_heads * self.d_kv, bias=bias)
        self.k_proj = nn.Linear(d_model, n_heads * self.d_kv, bias=bias)
        self.v_proj = nn.Linear(d_model, n_heads * self.d_kv, bias=bias)
        self.out_proj = nn.Linear(n_heads * self.d_kv, d_model, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass using SDPA.

        Args:
            x: Input tensor (batch, seq_len, d_model).
            attention_mask: Optional attention mask.
            past_kv: Optional past key/value for KV cache (not yet supported).
            position_ids: Optional position IDs (unused, RoPE handled externally).

        Returns:
            Attention output tensor (batch, seq_len, d_model).
        """
        batch, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.d_kv).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_heads, self.d_kv).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_heads, self.d_kv).transpose(1, 2)

        out = sdpa_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=self.is_causal,
        )

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.out_proj(out)
