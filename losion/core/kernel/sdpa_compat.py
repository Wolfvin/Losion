"""
SDPA Compatibility Layer — Unified Attention Interface for Losion.

Provides a single `sdpa_attention()` function that automatically selects
the best attention backend:
  1. Flash Attention 3 (warp-specialized, Hopper H100+) — fastest
  2. Flash Attention 2 (flash_attn package) — fast, O(n) memory
  3. PyTorch SDPA — auto-selects Flash/Memory-Efficient/Math backend
  4. Math fallback — manual QK^T softmax, O(n^2) memory

All Losion attention modules should call `sdpa_attention()` instead of
manual `torch.matmul + F.softmax` patterns.

Benefits over manual attention:
  - O(n) memory with Flash Attention (vs O(n^2))
  - 2-4x training speedup on CUDA/ROCm
  - Automatic kernel selection per hardware
  - Supports custom masks, dropout, GQA
  - Works with torch.compile

References:
  - FlashAttention-2: Dao (arXiv:2307.08691)
  - FlashAttention-3: Dao et al. (arXiv:2407.08608)
  - PyTorch SDPA: pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention
  - Losion Framework: Wolfvin (github.com/Wolfvin/Losion)
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from losion.core.kernel import (
    HAS_FLASH_ATTENTION,
    FLASH_ATTN_VERSION,
    _DISABLE_FLASH,
    _FORCE_SDPA,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Backend Selection
# ============================================================================

def _select_backend(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = True,
) -> str:
    """Select the best attention backend for the given inputs.

    Priority: flash_attn3 > flash_attn2 > sdpa > math

    Args:
        q, k, v: Query, Key, Value tensors.
        attn_mask: Optional attention mask.
        dropout_p: Dropout probability.
        is_causal: Whether causal masking is needed.

    Returns:
        One of "flash3", "flash2", "sdpa", "math".
    """
    # Force SDPA if environment variable set
    if _FORCE_SDPA:
        return "sdpa"

    # Flash Attention doesn't support custom additive masks
    # or dropout > 0 in some versions
    if attn_mask is not None and dropout_p == 0.0:
        # Custom mask — use SDPA which handles it
        return "sdpa"

    # Flash Attention 3 (Hopper warp-specialized)
    if FLASH_ATTN_VERSION == "fa3" and not _DISABLE_FLASH:
        if dropout_p == 0.0 and q.dtype in (torch.float16, torch.bfloat16):
            return "flash3"

    # Flash Attention 2
    if FLASH_ATTN_VERSION in ("fa2", "fa3") and not _DISABLE_FLASH:
        if dropout_p == 0.0 and q.dtype in (torch.float16, torch.bfloat16):
            return "flash2"

    # PyTorch SDPA (handles Flash/Memory-Efficient/Math automatically)
    if hasattr(F, 'scaled_dot_product_attention'):
        return "sdpa"

    return "math"


# ============================================================================
# Unified Attention Function
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
    """Unified scaled dot-product attention with automatic backend selection.

    This is the PRIMARY attention function that all Losion modules should
    use. It replaces manual `torch.matmul + F.softmax` patterns and
    automatically selects the fastest available backend.

    Args:
        q: Query tensor (batch, n_heads, seq_len, d_kv) or (batch, seq_len, n_heads, d_kv).
        k: Key tensor (batch, n_heads, full_len, d_kv) or (batch, full_len, n_heads, d_kv).
        v: Value tensor (batch, n_heads, full_len, d_kv) or (batch, full_len, n_heads, d_kv).
        attn_mask: Optional additive attention mask.
        dropout_p: Dropout probability.
        is_causal: Whether to apply causal masking.
        scale: Optional scale factor. If None, uses 1/sqrt(d_kv).

    Returns:
        Attention output tensor with same shape as q.
    """
    backend = _select_backend(q, k, v, attn_mask, dropout_p, is_causal)

    if backend == "flash3":
        return _flash3_attention(q, k, v, is_causal, scale)
    elif backend == "flash2":
        return _flash2_attention(q, k, v, is_causal, scale)
    elif backend == "sdpa":
        return _sdpa_attention(q, k, v, attn_mask, dropout_p, is_causal, scale)
    else:
        return _math_attention(q, k, v, attn_mask, dropout_p, is_causal, scale)


def _flash3_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = True,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """FlashAttention-3 with warp-specialization (Hopper H100+).

    References:
        - FlashAttention-3: Dao et al. (arXiv:2407.08608)
        - Warp-specialized async execution on Hopper
    """
    try:
        from flash_attn import flash_attn_func

        # flash_attn_func expects (batch, seq_len, n_heads, d_kv)
        if q.dim() == 4 and q.shape[1] < q.shape[2]:
            # (batch, n_heads, seq_len, d_kv) → (batch, seq_len, n_heads, d_kv)
            q_fa = q.transpose(1, 2)
            k_fa = k.transpose(1, 2)
            v_fa = v.transpose(1, 2)
        else:
            q_fa, k_fa, v_fa = q, k, v

        softmax_scale = scale or (1.0 / math.sqrt(q_fa.shape[-1]))

        output = flash_attn_func(
            q_fa, k_fa, v_fa,
            causal=is_causal,
            softmax_scale=softmax_scale,
        )

        # Transpose back if needed
        if q.shape[1] < q.shape[2]:
            output = output.transpose(1, 2)

        return output
    except Exception as e:
        logger.debug(f"Flash3 failed, falling back to SDPA: {e}")
        return _sdpa_attention(q, k, v, None, 0.0, is_causal, scale)


def _flash2_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = True,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """FlashAttention-2 via flash_attn package.

    References:
        - FlashAttention-2: Dao (arXiv:2307.08691)
    """
    try:
        from flash_attn import flash_attn_func

        # flash_attn_func expects (batch, seq_len, n_heads, d_kv)
        if q.dim() == 4 and q.shape[1] < q.shape[2]:
            q_fa = q.transpose(1, 2)
            k_fa = k.transpose(1, 2)
            v_fa = v.transpose(1, 2)
        else:
            q_fa, k_fa, v_fa = q, k, v

        softmax_scale = scale or (1.0 / math.sqrt(q_fa.shape[-1]))

        output = flash_attn_func(
            q_fa, k_fa, v_fa,
            causal=is_causal,
            softmax_scale=softmax_scale,
        )

        if q.shape[1] < q.shape[2]:
            output = output.transpose(1, 2)

        return output
    except Exception as e:
        logger.debug(f"Flash2 failed, falling back to SDPA: {e}")
        return _sdpa_attention(q, k, v, None, 0.0, is_causal, scale)


def _sdpa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = True,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """PyTorch SDPA — auto-selects Flash/Memory-Efficient/Math backend.

    References:
        - PyTorch F.scaled_dot_product_attention
    """
    # SDPA expects (batch, n_heads, seq_len, d_kv)
    # If input is (batch, seq_len, n_heads, d_kv), transpose
    transposed = False
    if q.dim() == 4 and q.shape[2] > q.shape[1]:
        # Likely (batch, seq_len, n_heads, d_kv) → transpose
        # Heuristic: seq_len > n_heads typically
        if q.shape[1] > q.shape[2]:
            pass  # Already (batch, n_heads, seq_len, d_kv)
        else:
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            transposed = True

    output = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )

    if transposed:
        output = output.transpose(1, 2)

    return output


def _math_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = True,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Math fallback — manual QK^T softmax attention.

    Only used when no optimized backend is available (e.g., CPU-only).
    O(n^2) memory — avoid for long sequences.
    """
    # Ensure (batch, n_heads, seq_len, d_kv) layout
    if q.dim() == 4 and q.shape[1] < q.shape[2]:
        pass  # Already correct layout
    else:
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

    d_kv = q.shape[-1]
    scale_factor = scale or (1.0 / math.sqrt(d_kv))

    attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale_factor

    if is_causal and attn_mask is None:
        seq_len = q.shape[2]
        full_len = k.shape[2]
        causal_mask = torch.triu(
            torch.ones(seq_len, full_len, dtype=torch.bool, device=q.device),
            diagonal=full_len - seq_len + 1,
        )
        attn_weights = attn_weights.masked_fill(
            causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
        )
    elif attn_mask is not None:
        attn_weights = attn_weights + attn_mask

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)

    if dropout_p > 0.0 and torch.is_grad_enabled():
        attn_weights = F.dropout(attn_weights, p=dropout_p)

    output = torch.matmul(attn_weights, v)
    return output


# ============================================================================
# Block-Sparse Attention (for MoBA)
# ============================================================================

def sdpa_block_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_mask: torch.Tensor,
    routing_weights: Optional[torch.Tensor] = None,
    is_causal: bool = True,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Block-sparse attention using SDPA with block masking.

    Used by MoBA (Mixture of Block Attention) for efficient
    block-sparse attention patterns.

    Args:
        q: Query (batch, n_heads, seq_len, d_kv).
        k: Key (batch, n_heads, full_len, d_kv).
        v: Value (batch, n_heads, full_len, d_kv).
        block_mask: Boolean mask (batch, n_heads, seq_len, full_len).
            True = masked (do NOT attend), False = attend.
        routing_weights: Optional routing weights for scaling.
        is_causal: Whether to apply causal masking.
        scale: Optional scale factor.

    Returns:
        Attention output (batch, n_heads, seq_len, d_kv).
    """
    d_kv = q.shape[-1]
    scale_factor = scale or (1.0 / math.sqrt(d_kv))

    # Compute attention scores
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale_factor

    # Apply block-sparse mask
    attn_scores = attn_scores.masked_fill(block_mask, float("-inf"))

    # Apply causal mask
    if is_causal:
        seq_len = q.shape[2]
        full_len = k.shape[2]
        causal_mask = torch.triu(
            torch.ones(seq_len, full_len, dtype=torch.bool, device=q.device),
            diagonal=full_len - seq_len + 1,
        )
        attn_scores = attn_scores.masked_fill(
            causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
        )

    attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(q.dtype)

    output = torch.matmul(attn_weights, v)
    return output


# ============================================================================
# Sliding Window Attention
# ============================================================================

def sdpa_sliding_window_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: int,
    is_causal: bool = True,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Sliding window attention using SDPA.

    Each query only attends to keys within a window of `window_size` tokens.

    Args:
        q: Query (batch, n_heads, seq_len, d_kv).
        k: Key (batch, n_heads, full_len, d_kv).
        v: Value (batch, n_heads, full_len, d_kv).
        window_size: Size of the sliding window.
        is_causal: Whether to apply causal masking.
        scale: Optional scale factor.

    Returns:
        Attention output (batch, n_heads, seq_len, d_kv).
    """
    seq_len = q.shape[2]
    full_len = k.shape[2]
    d_kv = q.shape[-1]
    scale_factor = scale or (1.0 / math.sqrt(d_kv))

    # Create sliding window mask
    # For each query position i, attend to keys in [max(0, i - window_size + 1), i]
    query_pos = torch.arange(seq_len, device=q.device).unsqueeze(1)  # (seq_len, 1)
    key_pos = torch.arange(full_len, device=q.device).unsqueeze(0)   # (1, full_len)

    # Window mask: key_pos >= query_pos - window_size + 1
    window_mask = key_pos >= (query_pos - window_size + 1)

    # Causal mask: key_pos <= query_pos
    if is_causal:
        causal_mask = key_pos <= query_pos
        mask = window_mask & causal_mask
    else:
        mask = window_mask

    # Invert for masked_fill (True = masked)
    inv_mask = ~mask

    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale_factor
    attn_scores = attn_scores.masked_fill(
        inv_mask.unsqueeze(0).unsqueeze(0), float("-inf")
    )

    attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(q.dtype)
    output = torch.matmul(attn_weights, v)
    return output


# ============================================================================
# Linear Attention (for DeltaNet / Lightning Attention)
# ============================================================================

def linear_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    feature_map: str = "elu",
    is_causal: bool = True,
) -> torch.Tensor:
    """Linear attention with feature mapping for O(n) complexity.

    Used by DeltaNet and Lightning Attention for sub-quadratic attention.

    Args:
        q: Query (batch, n_heads, seq_len, d_kv).
        k: Key (batch, n_heads, seq_len, d_kv).
        v: Value (batch, n_heads, seq_len, d_kv).
        feature_map: Feature map type ("elu", "relu", "identity").
        is_causal: Whether to apply causal cumsum.

    Returns:
        Attention output (batch, n_heads, seq_len, d_kv).
    """
    if feature_map == "elu":
        q_feat = F.elu(q) + 1
        k_feat = F.elu(k) + 1
    elif feature_map == "relu":
        q_feat = F.relu(q)
        k_feat = F.relu(k)
    else:
        q_feat = q
        k_feat = k

    if is_causal:
        # Cumulative sum approach for causal linear attention
        # kv_cum[t] = sum_{s<=t} k[s] @ v[s]^T
        # output[t] = q[t] @ kv_cum[t]

        # Compute outer products: (batch, n_heads, seq_len, d_kv, d_kv)
        kv = torch.einsum("bhsd,bhse->bhsde", k_feat, v)

        # Cumulative sum along sequence dimension
        kv_cum = kv.cumsum(dim=2)

        # Compute output: q @ kv_cum
        output = torch.einsum("bhsd,bhsde->bhse", q_feat, kv_cum)
    else:
        # Non-causal: single matrix multiply
        # output = q @ (k^T @ v)
        kv = torch.einsum("bhsd,bhse->bhde", k_feat, v)  # (batch, n_heads, d_kv, d_kv)
        output = torch.einsum("bhsd,bhde->bhse", q_feat, kv)

    return output


__all__ = [
    "sdpa_attention",
    "sdpa_block_sparse_attention",
    "sdpa_sliding_window_attention",
    "linear_attention_forward",
    "_select_backend",
]
