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
)
from losion.core.attention.context_extension import (
    ContextExtensionConfig,
    RoPEExtension,
    SSMStateExtension,
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
    # Context extension
    "ContextExtensionConfig",
    "RoPEExtension",
    "SSMStateExtension",
]
