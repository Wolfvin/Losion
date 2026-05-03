"""
Losion — Jalur 2: Attention + Compression.

Base modules:
  InterleavedRoPE — Rotary Position Embedding with interleaved pattern
  MLA             — Multi-head Latent Attention (DeepSeek-V2 style)

v0.4 additions:
  LightningAttention  — O(1) inference, 4M token context via linear attention
  SharedAttentionPool — Zamba2-style shared attention parameter pool
  SharedAttentionLayer — Attention layer referencing shared pool
  SharedAttentionConfig — Configuration for sharing pattern

v0.5 additions (Priority 1):
  KDAProjection       — Key-Direction Attention projection (arXiv:2510.26692)
  KDAMLA              — KDA+MLA Hybrid Attention (~75% KV cache reduction)

v0.6 additions (NeurIPS 2025 Best Paper):
  GatedAttentionConfig  — Configuration for Gated Attention
  GatedAttentionHead    — Single head with sigmoid gate after softmax
  GatedMultiHeadAttention — Multi-head with per-head gating + MLA + RoPE

v0.7 additions (NeurIPS 2025):
  MoBAConfig       — Configuration for Mixture of Block Attention
  BlockPartitioner — Sequence-to-block partitioning utility
  MoBARouter       — Block-level MoE routing network
  MoBAAttention    — Mixture of Block Attention (block-sparse attention via MoE routing)

Context extension additions:
  ContextExtensionConfig — Configuration for RoPE/SSM context extension
  RoPEExtension          — Extends RoPE context window (YaRN, NTK, linear, dynamic NTK)
  SSMStateExtension      — Extends SSM context by scaling state dimensions

v0.9 additions (Architecture Document Implementations):
  AttnResConfig        — Configuration for Attention Residuals
  AttnResMode          — Full/Block/Hybrid mode enum
  FullAttnRes          — Full Attention Residuals (all previous layers)
  BlockAttnRes         — Block Attention Residuals (efficient approximation)
  AttnResManager       — Coordinates Full/Block/Hybrid modes
  TokenAttnResCompression — Token-dimension AttnRes + Compression
  Child3WConfig        — Configuration for Child-3W routing
  Child3WSet           — Single Child-3W attention parameter set
  Child3WRouter        — Router for Child-3W sets
  Child3WAttention      — Full Child-3W attention with routing
"""

from losion.core.attention.lightning_attention import (
    InterleavedRoPE,
    MLA,
    LightningAttention,
)
from losion.core.attention.shared_attention import (
    SharedAttentionPool,
    SharedAttentionLayer,
    SharedAttentionConfig,
)
from losion.core.attention.kda_mla import (
    KDAProjection,
    KDAMLA,
)
from losion.core.attention.gated_attention import (
    GatedAttentionConfig,
    GatedAttentionHead,
    GatedMultiHeadAttention,
)
from losion.core.attention.moba import (
    MoBAConfig,
    BlockPartitioner,
    MoBARouter,
    MoBAAttention,
    MoBAAttention as MoBA,  # Alias for backward compatibility
)
from losion.core.attention.context_extension import (
    ContextExtensionConfig,
    RoPEExtension,
    SSMStateExtension,
)
from losion.core.attention.attn_res import (
    AttnResConfig,
    AttnResMode,
    FullAttnRes,
    BlockAttnRes,
    AttnResManager,
    TokenAttnResCompression,
)
from losion.core.attention.child_3w import (
    Child3WConfig as Child3WAttentionConfig,
    Child3WSet,
    Child3WRouter,
    Child3WAttention,
)
from losion.core.attention.sliding_window import (
    SlidingWindowConfig,
    SlidingWindowAttention,
)
from losion.core.attention.mosa import (
    MoSAConfig,
    MoSAAttention,
    SparseAttentionExpert,
)

# ============================================================================
# Unified Attention Interface — SDPA with Flash Attention auto-detection
# ============================================================================


def _detect_flash_attention() -> bool:
    """Detect if Flash Attention is available.
    
    Checks for:
    1. flash_attn package (NVIDIA)
    2. flash_attn_rocm package (AMD)
    3. PyTorch SDPA with Flash Attention backend
    
    Returns:
        True if Flash Attention is available.
    """
    # Check flash_attn package
    try:
        from flash_attn import flash_attn_func
        return True
    except ImportError:
        pass
    
    # Check flash_attn_rocm (AMD)
    try:
        from flash_attn_rocm import flash_attn_func
        return True
    except ImportError:
        pass
    
    # Check PyTorch SDPA Flash backend
    try:
        import torch
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            # Check if Flash backend is available
            with torch.device('cuda' if torch.cuda.is_available() else 'cpu'):
                return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    except Exception:
        pass
    
    return False


# Module-level flag (computed once at import time)
HAS_FLASH_ATTENTION: bool = _detect_flash_attention()


def attention_forward(
    q: "torch.Tensor",
    k: "torch.Tensor",
    v: "torch.Tensor",
    attn_mask: "Optional[torch.Tensor]" = None,
    dropout_p: float = 0.0,
    is_causal: bool = True,
) -> "torch.Tensor":
    """Unified attention forward with automatic Flash Attention selection.
    
    Automatically selects the best attention implementation:
    1. Flash Attention 2 (if flash_attn installed) — fastest, O(n) memory
    2. PyTorch SDPA — automatic kernel selection, O(n) memory with Flash backend
    3. Math fallback — standard attention, O(n^2) memory
    
    Args:
        q: Query tensor (batch, n_heads, seq_len, d_kv).
        k: Key tensor (batch, n_heads, full_len, d_kv).
        v: Value tensor (batch, n_heads, full_len, d_kv).
        attn_mask: Optional attention mask (additive).
        dropout_p: Dropout probability.
        is_causal: Whether to use causal masking.
    
    Returns:
        Attention output tensor (batch, n_heads, seq_len, d_kv).
    """
    import torch
    import torch.nn.functional as F
    
    # Try flash_attn package first (most optimized)
    if HAS_FLASH_ATTENTION and attn_mask is None and dropout_p == 0.0:
        try:
            from flash_attn import flash_attn_func
            # flash_attn_func expects (batch, seq_len, n_heads, d_kv)
            q_fa = q.transpose(1, 2)
            k_fa = k.transpose(1, 2)
            v_fa = v.transpose(1, 2)
            output = flash_attn_func(q_fa, k_fa, v_fa, causal=is_causal)
            return output.transpose(1, 2)
        except Exception:
            pass  # Fall through to SDPA
    
    # Use PyTorch SDPA (handles Flash, memory-efficient, and math backends)
    return F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
    )


__all__ = [
    "InterleavedRoPE",
    "MLA",
    "LightningAttention",
    "SharedAttentionPool",
    "SharedAttentionLayer",
    "SharedAttentionConfig",
    "KDAProjection",
    "KDAMLA",
    "GatedAttentionConfig",
    "GatedAttentionHead",
    "GatedMultiHeadAttention",
    "MoBAConfig",
    "BlockPartitioner",
    "MoBARouter",
    "MoBAAttention",
    "MoBA",  # Alias for MoBAAttention
    # Context extension
    "ContextExtensionConfig",
    "RoPEExtension",
    "SSMStateExtension",
    # v0.9 AttnRes
    "AttnResConfig",
    "AttnResMode",
    "FullAttnRes",
    "BlockAttnRes",
    "AttnResManager",
    "TokenAttnResCompression",
    # v0.9 Child-3W
    "Child3WAttentionConfig",
    "Child3WSet",
    "Child3WRouter",
    "Child3WAttention",
    # v0.10 Memory Efficiency
    "SlidingWindowConfig",
    "SlidingWindowAttention",
    "MoSAConfig",
    "MoSAAttention",
    "SparseAttentionExpert",
    # Flash Attention Detection
    "HAS_FLASH_ATTENTION",
    "_detect_flash_attention",
    "attention_forward",
]
