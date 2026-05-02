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

__all__ = [
    "InterleavedRoPE",
    "MLA",
    "LightningAttention",
    "SharedAttentionPool",
    "SharedAttentionLayer",
    "SharedAttentionConfig",
    "KDAProjection",
    "KDAMLA",
]
