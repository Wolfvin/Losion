"""
Flash Attention Integration Layer for Losion.

Provides unified access to Flash Attention 2/3/4 with automatic
hardware detection and fallback to PyTorch SDPA.

Flash Attention versions:
  - FA2: O(n) memory, supports CUDA/ROCm, stable (arXiv:2307.08691)
  - FA3: Warp-specialized, async, 1.5-2x over FA2 on H100 (arXiv:2407.08608)
  - FA4: Blackwell B200 kernel pipelining, 2.7x over Triton (arXiv:2603.05451)

The `flash_attention_forward()` function selects the best available
version automatically and handles all format conversions.

References:
  - FlashAttention-2: Dao (arXiv:2307.08691)
  - FlashAttention-3: Dao et al. (arXiv:2407.08608)
  - FlashAttention-4: (arXiv:2603.05451)
  - Losion Framework: Wolfvin (github.com/Wolfvin/Losion)
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from losion.core.kernel import (
    FLASH_ATTN_VERSION,
    HAS_FLASH_ATTENTION,
    _DISABLE_FLASH,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Flash Attention Forward
# ============================================================================

def flash_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = True,
    scale: Optional[float] = None,
    kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
    """Flash Attention forward with automatic version selection.

    Selects the best available Flash Attention implementation:
    1. flash_attn package (FA2 or FA3 depending on what's installed)
    2. PyTorch SDPA with Flash backend
    3. SDPA with memory-efficient backend
    4. Math fallback

    Handles all format conversions automatically:
    - Input: (batch, n_heads, seq_len, d_kv) or (batch, seq_len, n_heads, d_kv)
    - flash_attn expects: (batch, seq_len, n_heads, d_kv)
    - SDPA expects: (batch, n_heads, seq_len, d_kv)

    Args:
        q: Query tensor.
        k: Key tensor.
        v: Value tensor.
        attn_mask: Optional attention mask (additive).
        dropout_p: Dropout probability.
        is_causal: Whether to use causal masking.
        scale: Optional scale factor.
        kv_cache: Optional KV cache for inference.

    Returns:
        Tuple of (output, updated_kv_cache).
    """
    if not HAS_FLASH_ATTENTION or _DISABLE_FLASH:
        return _sdpa_forward(q, k, v, attn_mask, dropout_p, is_causal, scale, kv_cache)

    # Handle KV cache
    if kv_cache is not None:
        cached_k, cached_v = kv_cache
        k = torch.cat([cached_k, k], dim=-2)  # Concat along seq dim
        v = torch.cat([cached_v, v], dim=-2)

    new_kv_cache = (k, v) if kv_cache is not None else None

    # Flash Attention doesn't support custom masks or dropout > 0 in some cases
    if attn_mask is not None or dropout_p > 0.0:
        output = _sdpa_forward(q, k, v, attn_mask, dropout_p, is_causal, scale)[0]
        return output, new_kv_cache

    # Try flash_attn package
    if FLASH_ATTN_VERSION in ("fa2", "fa3"):
        output = _flash_attn_package(q, k, v, is_causal, scale)
        if output is not None:
            return output, new_kv_cache

    # Fallback to SDPA
    output = _sdpa_forward(q, k, v, None, 0.0, is_causal, scale)[0]
    return output, new_kv_cache


def _flash_attn_package(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = True,
    scale: Optional[float] = None,
) -> Optional[torch.Tensor]:
    """Use flash_attn package (FA2 or FA3)."""
    try:
        from flash_attn import flash_attn_func

        # Detect input layout and transpose if needed
        # flash_attn expects (batch, seq_len, n_heads, d_kv)
        if q.dim() == 4:
            if q.shape[1] <= q.shape[2]:
                # Likely (batch, n_heads, seq_len, d_kv) → transpose
                q_fa = q.transpose(1, 2).contiguous()
                k_fa = k.transpose(1, 2).contiguous()
                v_fa = v.transpose(1, 2).contiguous()
                need_transpose_back = True
            else:
                q_fa, k_fa, v_fa = q, k, v
                need_transpose_back = False
        else:
            return None

        softmax_scale = scale or (1.0 / math.sqrt(q_fa.shape[-1]))

        output = flash_attn_func(
            q_fa, k_fa, v_fa,
            causal=is_causal,
            softmax_scale=softmax_scale,
        )

        if need_transpose_back:
            output = output.transpose(1, 2).contiguous()

        return output

    except Exception as e:
        logger.debug(f"flash_attn package failed: {e}")
        return None


def _sdpa_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = True,
    scale: Optional[float] = None,
    kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
    """PyTorch SDPA fallback."""
    if kv_cache is not None:
        cached_k, cached_v = kv_cache
        k = torch.cat([cached_k, k], dim=-2)
        v = torch.cat([cached_v, v], dim=-2)

    new_kv_cache = (k, v) if kv_cache is not None else None

    # Ensure (batch, n_heads, seq_len, d_kv) layout for SDPA
    if q.dim() == 4 and q.shape[1] < q.shape[2]:
        pass  # Already correct
    elif q.dim() == 4:
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

    output = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )

    return output, new_kv_cache


__all__ = [
    "flash_attention_forward",
]
