"""
Gated Attention — Qwen-style Sigmoid-Gated Softmax Attention.

NeurIPS 2025 Best Paper contribution from the Qwen Team. The key insight is
that inserting a learned sigmoid gate *after* softmax attention eliminates
attention sinks and introduces beneficial sparsity, which synergises with
MoE routing in hybrid architectures.

Architecture:
1. GatedAttentionConfig — Dataclass holding hyper-parameters.
   Controls head count, KV dimension, MLA compression, gate initialisation,
   and optional QK normalisation.

2. GatedAttentionHead — Single attention head with sigmoid gate.
   attention_output = softmax(QK^T / sqrt(d)) * V
   gated_output     = sigmoid(W_g * attention_output) * attention_output
   The gate is a per-head learned linear projection followed by sigmoid.
   Supports MLA KV compression and RoPE position encoding.
   forward(x, kv_cache=None, position_ids=None) -> (output, updated_kv_cache)

3. GatedMultiHeadAttention — Multi-head version with per-head gating.
   Gate projection: nn.Linear(d_model, n_heads) -> sigmoid per head.
   Supports optional QK normalisation for training stability.
   MLA-compatible KV cache for memory-efficient inference.
   forward() and forward_inference() methods.

Key benefits:
- Eliminates attention sinks: the sigmoid gate can suppress the "sink" token
  that would otherwise absorb disproportionate attention mass.
- Beneficial sparsity: gates naturally learn to zero-out irrelevant heads,
  yielding a soft form of head-level sparsity without hard pruning.
- MoE synergy: in hybrid MoE + attention architectures, the gate's sparsity
  pattern aligns naturally with expert routing, reducing interference between
  active and inactive experts.

References:
- Qwen Team, "Gated Attention" (NeurIPS 2025 Best Paper)
- DeepSeek-AI, "DeepSeek-V2" (2024) — MLA
- Su, J. et al., "RoFormer: Enhanced Transformer with Rotary Position
  Embedding" (2021) — RoPE

Hardware: Pure PyTorch, compatible with CUDA, ROCm, and CPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# GatedAttentionConfig
# ============================================================================

@dataclass
class GatedAttentionConfig:
    """Configuration for Gated Attention modules.

    Attributes:
        n_heads: Number of attention heads.
        d_kv: Dimension per head (also called d_head elsewhere).
        use_mla: Whether to use Multi-head Latent Attention KV compression.
        mla_latent_dim: Latent dimension for MLA KV compression.
            Only used when ``use_mla`` is True.
        gate_init: Initialisation strategy for the gate projection.
            "ones"  — initialise so sigmoid output is ~1 (near-identity start).
            "zeros" — initialise so sigmoid output is ~0.5.
        use_qk_norm: Whether to apply RMSNorm to Q and K before the
            attention score computation (improves training stability).
        d_model: Model hidden dimension (required for module construction).
        rope_base: Base frequency for RoPE sinusoidal embeddings.
        dropout: Dropout rate applied to attention weights.
    """

    n_heads: int = 8
    d_kv: int = 64
    use_mla: bool = True
    mla_latent_dim: int = 128
    gate_init: str = "ones"
    use_qk_norm: bool = True
    d_model: int = 512
    rope_base: float = 10000.0
    dropout: float = 0.0


# ============================================================================
# GatedAttentionHead — Single Head with Sigmoid Gate
# ============================================================================

class GatedAttentionHead(nn.Module):
    """Single attention head with a sigmoid gate applied after softmax.

    The gating mechanism follows the Qwen Gated Attention formulation:

        attention_weights = softmax(Q @ K^T / sqrt(d_kv))
        attention_output  = attention_weights @ V
        gate              = sigmoid(W_g @ attention_output)
        gated_output      = gate * attention_output

    The gate is a *learned* per-head linear projection (``W_g``) whose output
    is passed through sigmoid.  At initialisation the weights are set so that
    the gate value is close to 1.0, making the module start as near-identity
    and gradually learn to suppress unnecessary attention pathways.

    Supports:
    - MLA KV compression (optional): compresses KV cache to a low-rank latent.
    - RoPE position encoding via :class:`InterleavedRoPE`.

    Args:
        config: A :class:`GatedAttentionConfig` instance.
    """

    def __init__(self, config: GatedAttentionConfig) -> None:
        super().__init__()

        self.config = config
        self.d_kv: int = config.d_kv
        self.d_model: int = config.d_model
        self.use_mla: bool = config.use_mla
        self.mla_latent_dim: int = config.mla_latent_dim
        self.use_qk_norm: bool = config.use_qk_norm

        # ---- Q projection ----
        self.q_proj = nn.Linear(self.d_model, self.d_kv, bias=False)

        # ---- K, V projections (MLA or standard) ----
        if self.use_mla:
            self.kv_down_proj = nn.Linear(self.d_model, self.mla_latent_dim, bias=False)
            self.kv_norm = nn.RMSNorm(self.mla_latent_dim, eps=1e-5)
            self.k_up_proj = nn.Linear(self.mla_latent_dim, self.d_kv, bias=False)
            self.v_up_proj = nn.Linear(self.mla_latent_dim, self.d_kv, bias=False)
        else:
            self.k_proj = nn.Linear(self.d_model, self.d_kv, bias=False)
            self.v_proj = nn.Linear(self.d_model, self.d_kv, bias=False)

        # ---- Sigmoid gate: W_g projects d_kv -> d_kv ----
        self.W_g = nn.Linear(self.d_kv, self.d_kv, bias=False)
        self._init_gate()

        # ---- RoPE ----
        from .lightning_attention import InterleavedRoPE
        self.rope = InterleavedRoPE(
            dim=self.d_kv,
            d_rope=self.d_kv // 2,
            base=config.rope_base,
            interleaved=False,
        )

        # ---- QK normalisation (optional) ----
        if self.use_qk_norm:
            self.q_norm = nn.RMSNorm(self.d_kv, eps=1e-5)
            self.k_norm = nn.RMSNorm(self.d_kv, eps=1e-5)

        # ---- Output projection ----
        self.out_proj = nn.Linear(self.d_kv, self.d_model, bias=False)

    # ------------------------------------------------------------------
    # Gate initialisation
    # ------------------------------------------------------------------

    def _init_gate(self) -> None:
        """Initialise gate projection weights.

        "ones"  — large positive weights so sigmoid ≈ 1 (near-identity).
        "zeros" — zero weights so sigmoid ≈ 0.5.
        """
        strategy = self.config.gate_init
        with torch.no_grad():
            if strategy == "ones":
                # Initialise to a moderate positive value so sigmoid ≈ 1
                nn.init.ones_(self.W_g.weight)
                self.W_g.weight.data.mul_(2.0)  # sigmoid(2) ≈ 0.88
            elif strategy == "zeros":
                nn.init.zeros_(self.W_g.weight)
            else:
                # Default Xavier uniform
                nn.init.xavier_uniform_(self.W_g.weight)

    # ------------------------------------------------------------------
    # KV projection helpers
    # ------------------------------------------------------------------

    def _project_kv(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project input to K and V, optionally via MLA latent compression.

        Args:
            x: Input tensor ``(batch, seq_len, d_model)``.

        Returns:
            Tuple ``(k, v)`` each of shape ``(batch, seq_len, d_kv)``.
        """
        if self.use_mla:
            c_kv = self.kv_norm(self.kv_down_proj(x))
            k = self.k_up_proj(c_kv)
            v = self.v_up_proj(c_kv)
        else:
            k = self.k_proj(x)
            v = self.v_proj(x)
        return k, v

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass for a single gated attention head.

        Args:
            x: Input tensor ``(batch, seq_len, d_model)``.
            kv_cache: Optional tuple ``(cached_k, cached_v)`` from a
                previous step.  ``cached_k`` has shape
                ``(batch, past_len, d_kv)`` (or
                ``(batch, past_len, mla_latent_dim)`` when MLA is used).
            position_ids: Optional position indices ``(batch, seq_len)``.
                If ``None``, positions are inferred from cache length.

        Returns:
            Tuple ``(output, updated_kv_cache)`` where ``output`` has shape
            ``(batch, seq_len, d_model)`` and ``updated_kv_cache`` is a
            tuple ``(k_full, v_full)``.
        """
        batch, seq_len, _ = x.shape

        # ---- Project Q, K, V ----
        q = self.q_proj(x)  # (batch, seq_len, d_kv)
        k, v = self._project_kv(x)

        # ---- MLA KV cache: store latent, up-project on the fly ----
        if self.use_mla and kv_cache is not None:
            c_kv_new = self.kv_norm(self.kv_down_proj(x))
            c_kv_cached = kv_cache[0]  # (batch, past_len, mla_latent_dim)
            c_kv_full = torch.cat([c_kv_cached, c_kv_new], dim=1)
            k = self.k_up_proj(c_kv_full)
            v = self.v_up_proj(c_kv_full)
            present_kv = (c_kv_full,)
        elif self.use_mla:
            c_kv_new = self.kv_norm(self.kv_down_proj(x))
            present_kv = (c_kv_new,)
        else:
            # Standard KV cache
            if kv_cache is not None:
                k_cached, v_cached = kv_cache
                k = torch.cat([k_cached, k], dim=1)
                v = torch.cat([v_cached, v], dim=1)
            present_kv = (k, v)

        full_len = k.shape[1]

        # ---- RoPE ----
        # Reshape for InterleavedRoPE: (batch, seq_len, 1, d_kv)
        q_4d = q.unsqueeze(2)
        k_4d = k.unsqueeze(2)

        offset = 0
        if position_ids is not None:
            offset = position_ids[0, 0].item() if position_ids.numel() > 0 else 0
        elif kv_cache is not None and not self.use_mla:
            offset = kv_cache[0].shape[1]
        elif kv_cache is not None and self.use_mla:
            offset = kv_cache[0].shape[1]

        q_4d = self.rope(q_4d, offset=offset)
        k_4d = self.rope(k_4d, offset=0)
        q = q_4d.squeeze(2)
        k = k_4d.squeeze(2)

        # ---- Optional QK normalisation ----
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # ---- Scaled dot-product attention ----
        scale = math.sqrt(self.d_kv)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale  # (batch, seq_len, full_len)

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, full_len, dtype=torch.bool, device=x.device),
            diagonal=full_len - seq_len + 1,
        )
        attn_weights = attn_weights.masked_fill(causal_mask.unsqueeze(0), float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)

        attn_output = torch.matmul(attn_weights, v)  # (batch, seq_len, d_kv)

        # ---- Sigmoid gate ----
        gate = torch.sigmoid(self.W_g(attn_output))  # (batch, seq_len, d_kv)
        gated_output = gate * attn_output

        # ---- Output projection ----
        output = self.out_proj(gated_output)  # (batch, seq_len, d_model)

        return output, present_kv


# ============================================================================
# GatedMultiHeadAttention — Multi-Head with Per-Head Gating
# ============================================================================

class GatedMultiHeadAttention(nn.Module):
    """Multi-head attention with per-head sigmoid gating.

    This is the full multi-head variant of :class:`GatedAttentionHead`.
    Each head receives an independent scalar gate computed from the input:

        gate_logits = W_gate @ x   # (batch, seq_len, n_heads)
        gate        = sigmoid(gate_logits)
        # gate is broadcast over the d_kv dimension of each head's output

    The per-head gate allows the model to learn *which heads* to activate
    for a given input — a form of soft head-level sparsity that naturally
    complements MoE expert routing.

    Features:
    - **Per-head gating**: ``nn.Linear(d_model, n_heads)`` -> sigmoid per head.
    - **QK normalisation** (optional): RMSNorm on Q and K for training stability.
    - **MLA-compatible KV cache**: when ``use_mla=True``, KV is compressed to
      a low-rank latent, and the cache stores only the latent vectors.
    - **RoPE**: Rotary position encoding applied to Q and K.

    Args:
        config: A :class:`GatedAttentionConfig` instance.
    """

    def __init__(self, config: GatedAttentionConfig) -> None:
        super().__init__()

        self.config = config
        self.n_heads: int = config.n_heads
        self.d_kv: int = config.d_kv
        self.d_model: int = config.d_model
        self.d_inner: int = self.n_heads * self.d_kv
        self.use_mla: bool = config.use_mla
        self.mla_latent_dim: int = config.mla_latent_dim
        self.use_qk_norm: bool = config.use_qk_norm

        # ---- Q projection ----
        self.q_proj = nn.Linear(self.d_model, self.d_inner, bias=False)

        # ---- K, V projections (MLA or standard) ----
        if self.use_mla:
            self.kv_down_proj = nn.Linear(self.d_model, self.mla_latent_dim, bias=False)
            self.kv_norm = nn.RMSNorm(self.mla_latent_dim, eps=1e-5)
            self.k_up_proj = nn.Linear(self.mla_latent_dim, self.d_inner, bias=False)
            self.v_up_proj = nn.Linear(self.mla_latent_dim, self.d_inner, bias=False)
        else:
            self.k_proj = nn.Linear(self.d_model, self.d_inner, bias=False)
            self.v_proj = nn.Linear(self.d_model, self.d_inner, bias=False)

        # ---- Per-head gate projection ----
        self.gate_proj = nn.Linear(self.d_model, self.n_heads, bias=False)
        self._init_gate()

        # ---- QK normalisation (optional, per head) ----
        if self.use_qk_norm:
            self.q_norm = nn.RMSNorm(self.d_kv, eps=1e-5)
            self.k_norm = nn.RMSNorm(self.d_kv, eps=1e-5)

        # ---- RoPE ----
        from .lightning_attention import InterleavedRoPE
        self.rope = InterleavedRoPE(
            dim=self.d_kv,
            d_rope=self.d_kv // 2,
            base=config.rope_base,
            interleaved=False,
        )

        # ---- Output ----
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)
        self.out_norm = nn.RMSNorm(self.d_model, eps=1e-5)
        self.attn_dropout = nn.Dropout(config.dropout)

    # ------------------------------------------------------------------
    # Gate initialisation
    # ------------------------------------------------------------------

    def _init_gate(self) -> None:
        """Initialise per-head gate projection so gates start near 1."""
        strategy = self.config.gate_init
        with torch.no_grad():
            if strategy == "ones":
                nn.init.ones_(self.gate_proj.weight)
                self.gate_proj.weight.data.mul_(2.0)  # sigmoid(2) ≈ 0.88
            elif strategy == "zeros":
                nn.init.zeros_(self.gate_proj.weight)
            else:
                nn.init.xavier_uniform_(self.gate_proj.weight)

    # ------------------------------------------------------------------
    # KV helpers
    # ------------------------------------------------------------------

    def _project_kv(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project input to K and V with optional MLA compression.

        Args:
            x: ``(batch, seq_len, d_model)``.

        Returns:
            Tuple ``(k, v)`` each of shape ``(batch, seq_len, n_heads, d_kv)``.
        """
        batch, seq_len, _ = x.shape

        if self.use_mla:
            c_kv = self.kv_norm(self.kv_down_proj(x))  # (batch, seq_len, mla_latent_dim)
            k = self.k_up_proj(c_kv)
            v = self.v_up_proj(c_kv)
        else:
            k = self.k_proj(x)
            v = self.v_proj(x)

        k = k.view(batch, seq_len, self.n_heads, self.d_kv)
        v = v.view(batch, seq_len, self.n_heads, self.d_kv)
        return k, v

    # ------------------------------------------------------------------
    # Core attention
    # ------------------------------------------------------------------

    def _compute_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute multi-head softmax attention with SDPA dispatch.

        Uses F.scaled_dot_product_attention when available (PyTorch 2.0+),
        which auto-dispatches to Flash Attention, memory-efficient attention,
        or math attention depending on hardware. Falls back to manual
        matmul attention for older PyTorch versions.

        Args:
            q: ``(batch, n_heads, seq_len, d_kv)``.
            k: ``(batch, n_heads, full_len, d_kv)``.
            v: ``(batch, n_heads, full_len, d_kv)``.
            attention_mask: Optional additive mask.

        Returns:
            Attention output ``(batch, n_heads, seq_len, d_kv)``.
        """
        batch, n_heads, seq_len, d_kv = q.shape
        full_len = k.shape[2]

        # Optional QK normalisation
        if self.use_qk_norm:
            q = self.q_norm(q.transpose(1, 2)).transpose(1, 2)
            k = self.k_norm(k.transpose(1, 2)).transpose(1, 2)

        # Try SDPA (auto Flash/MemEff/Math dispatch)
        try:
            is_causal = attention_mask is None and seq_len > 1
            dropout_p = self.config.dropout if self.training else 0.0
            return F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
            )
        except (AttributeError, RuntimeError):
            pass

        # Fallback: manual matmul attention
        scale = math.sqrt(d_kv)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale

        # Causal mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        else:
            causal_mask = torch.triu(
                torch.ones(seq_len, full_len, dtype=torch.bool, device=q.device),
                diagonal=full_len - seq_len + 1,
            )
            attn_weights = attn_weights.masked_fill(
                causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
            )

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        return torch.matmul(attn_weights, v)  # (batch, n_heads, seq_len, d_kv)

    # ------------------------------------------------------------------
    # Forward (training / prefill)
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, ...]] = None,
        position_offset: int = 0,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Forward pass — Gated Multi-Head Attention.

        Steps:
            1. Project Q, K, V (optionally through MLA latent).
            2. Apply RoPE to Q and K.
            3. Concatenate with cached K, V if available.
            4. Compute softmax attention.
            5. Apply per-head sigmoid gate: ``gate = sigmoid(W_g @ x)``.
            6. Gated output = ``gate * attention_output``.
            7. Output projection + RMSNorm.

        Args:
            x: Input tensor ``(batch, seq_len, d_model)``.
            attention_mask: Optional additive mask for padding.
            past_key_value: Tuple from previous step.
                When ``use_mla``: ``(cached_c_kv, None)`` where
                ``cached_c_kv`` has shape ``(batch, past_len, mla_latent_dim)``.
                Otherwise: ``(cached_k, cached_v)`` where each has shape
                ``(batch, past_len, n_heads, d_kv)``.
            position_offset: Position offset for RoPE (default 0).

        Returns:
            Tuple ``(output, present_key_value)`` where ``output`` has shape
            ``(batch, seq_len, d_model)``.
        """
        batch, seq_len, _ = x.shape

        if seq_len == 0:
            dummy = torch.zeros(batch, 0, self.d_model, dtype=x.dtype, device=x.device)
            return dummy, (None, None)

        # ---- Unpack past state ----
        cached_kv = None
        if past_key_value is not None:
            cached_kv = past_key_value[0]

        # ---- Q projection ----
        q = self.q_proj(x)  # (batch, seq_len, d_inner)
        q = q.view(batch, seq_len, self.n_heads, self.d_kv)

        # ---- K, V projection ----
        k, v = self._project_kv(x)  # (batch, seq_len, n_heads, d_kv)

        # ---- RoPE on Q and K ----
        offset = position_offset
        if self.use_mla and cached_kv is not None:
            offset = cached_kv.shape[1]
        elif not self.use_mla and cached_kv is not None:
            offset = cached_kv.shape[1]

        q_rope = q[..., : self.d_kv // 2].contiguous()
        k_rope = k[..., : self.d_kv // 2].contiguous()

        q_rope = self.rope(q_rope, offset=offset)
        k_rope = self.rope(k_rope, offset=0)

        half = self.d_kv // 2
        q = torch.cat([q_rope, q[..., half:]], dim=-1)
        k = torch.cat([k_rope, k[..., half:]], dim=-1)

        # ---- KV cache handling ----
        if self.use_mla:
            c_kv_new = self.kv_norm(self.kv_down_proj(x))  # (batch, seq_len, mla_latent_dim)
            if cached_kv is not None:
                c_kv_full = torch.cat([cached_kv, c_kv_new], dim=1)
            else:
                c_kv_full = c_kv_new

            # Up-project from full latent for attention
            k_full = self.k_up_proj(c_kv_full).view(batch, -1, self.n_heads, self.d_kv)
            v_full = self.v_up_proj(c_kv_full).view(batch, -1, self.n_heads, self.d_kv)

            # Apply RoPE to reconstructed K
            k_full_rope = k_full[..., :half].contiguous()
            k_full_rope = self.rope(k_full_rope, offset=0)
            k_full = torch.cat([k_full_rope, k_full[..., half:]], dim=-1)

            present_kv = (c_kv_full, None)
        else:
            if cached_kv is not None:
                cached_k, cached_v = cached_kv
                k_full = torch.cat([cached_k, k], dim=1)
                v_full = torch.cat([cached_v, v], dim=1)
            else:
                k_full = k
                v_full = v
            present_kv = (k_full, v_full)

        full_len = k_full.shape[1]

        # ---- Transpose to (batch, n_heads, seq/full_len, d_kv) ----
        q_t = q.transpose(1, 2)
        k_t = k_full.transpose(1, 2)
        v_t = v_full.transpose(1, 2)

        # ---- Compute attention ----
        attn_output = self._compute_attention(q_t, k_t, v_t, attention_mask)
        # (batch, n_heads, seq_len, d_kv)

        # ---- Per-head sigmoid gate ----
        gate_logits = self.gate_proj(x)  # (batch, seq_len, n_heads)
        gate = torch.sigmoid(gate_logits)  # (batch, seq_len, n_heads)
        gate = gate.permute(0, 2, 1).unsqueeze(-1)  # (batch, n_heads, seq_len, 1)

        gated_output = gate * attn_output  # (batch, n_heads, seq_len, d_kv)

        # ---- Reshape and project output ----
        gated_output = gated_output.transpose(1, 2).contiguous()
        gated_output = gated_output.view(batch, seq_len, self.d_inner)

        output = self.out_proj(gated_output)
        output = self.out_norm(output)

        return output, present_kv

    # ------------------------------------------------------------------
    # Forward (inference — token-by-token)
    # ------------------------------------------------------------------

    def forward_inference(
        self,
        x: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, ...]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Forward pass optimised for single-token inference.

        Delegates to :meth:`forward` with ``attention_mask=None`` and
        ``position_offset=0`` (position is inferred from the KV cache).

        Args:
            x: Single-token input ``(batch, 1, d_model)``.
            past_key_value: Cached state from the previous step.

        Returns:
            Tuple ``(output, present_key_value)``.
        """
        return self.forward(
            x,
            attention_mask=None,
            past_key_value=past_key_value,
            position_offset=0,
        )
