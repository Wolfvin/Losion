"""
Parallel Hybrid Head — Hymba-inspired parallel SSM + Attention head.

Instead of sequential SSM → Attention → MoE, the parallel head processes
the input through SSM and Attention SIMULTANEOUSLY, then combines outputs.
This is inspired by NVIDIA Hymba (ICLR 2025) which showed parallel heads
achieve better performance for small-to-medium models.

Key advantages:
  - Both SSM and Attention see the original input (not post-SSM/Attn)
  - SSM provides context summarization, Attention provides high-resolution recall
  - Meta tokens for cross-layer information sharing
  - Fewer total FLOPs than sequential processing

Inspired by:
  - NVIDIA Hymba (ICLR 2025, arXiv 2411.13676)
  - Jamba (AI21, ICLR 2025): Sequential hybrid at 1:7 Attention:SSM ratio

Credits:
  - Hymba: NVIDIA Research, ICLR 2025
  - Jamba: AI21 Labs, ICLR 2025

Hardware: Pure PyTorch, compatible with CUDA / ROCm / CPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x = x / rms
        return (self.weight * x).to(dtype)


@dataclass
class ParallelHeadConfig:
    """Configuration for Parallel Hybrid Head.

    Attributes:
        d_model: Model hidden dimension.
        n_heads: Number of attention heads.
        d_kv: Dimension per key/value head.
        ssm_d_state: SSM state dimension.
        ssm_d_conv: SSM local convolution width.
        ssm_expand: SSM expansion factor.
        use_mla: Whether to use MLA compression for attention.
        mla_latent_dim: MLA latent dimension.
        num_meta_tokens: Number of meta tokens for cross-layer info.
        ssm_weight_init: Initial weight for SSM pathway (0.5 = equal).
        attn_weight_init: Initial weight for Attention pathway (0.5 = equal).
        use_gated_fusion: Whether to use learned gating for fusion.
    """
    d_model: int = 192
    n_heads: int = 4
    d_kv: int = 48
    ssm_d_state: int = 16
    ssm_d_conv: int = 4
    ssm_expand: int = 2
    use_mla: bool = True
    mla_latent_dim: int = 48
    num_meta_tokens: int = 4
    ssm_weight_init: float = 0.5
    attn_weight_init: float = 0.5
    use_gated_fusion: bool = True


class SSMHead(nn.Module):
    """SSM pathway for parallel head — state-based context summarization.

    Uses simplified Mamba-style SSM with input-dependent gating.

    Args:
        d_model: Model dimension.
        d_state: SSM state dimension.
        expand: Expansion factor.
        d_conv: Convolution width.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        expand: int = 2,
        d_conv: int = 4,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand

        self.proj_in = nn.Linear(d_model, self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner, bias=True,
        )
        self.proj_out = nn.Linear(self.d_inner, d_model, bias=False)
        self.gate = nn.Linear(d_model, self.d_inner, bias=False)
        self.state_proj = nn.Linear(d_model, self.d_inner, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through SSM head.

        Args:
            x: Input tensor (batch, seq_len, d_model).

        Returns:
            Output tensor (batch, seq_len, d_model).
        """
        z = self.proj_in(x)
        gate = torch.sigmoid(self.gate(x))
        z = z * gate

        z_conv = z.transpose(1, 2)
        z_conv = self.conv1d(z_conv)[:, :, :x.shape[1]]
        z = z_conv.transpose(1, 2)

        s = torch.sigmoid(self.state_proj(x))
        z = z * s

        return self.proj_out(z)


class AttentionHead(nn.Module):
    """Attention pathway for parallel head — high-resolution recall.

    Uses MLA-compressed attention for memory efficiency.

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        d_kv: Dimension per head.
        use_mla: Whether to use MLA compression.
        mla_latent_dim: MLA latent dimension.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        d_kv: int = 48,
        use_mla: bool = True,
        mla_latent_dim: int = 48,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_kv = d_kv
        self.d_inner = n_heads * d_kv
        self.use_mla = use_mla

        self.q_proj = nn.Linear(d_model, self.d_inner, bias=False)

        if use_mla:
            self.kv_down = nn.Linear(d_model, mla_latent_dim, bias=False)
            self.kv_norm = RMSNorm(mla_latent_dim)
            self.k_up = nn.Linear(mla_latent_dim, self.d_inner, bias=False)
            self.v_up = nn.Linear(mla_latent_dim, self.d_inner, bias=False)
        else:
            self.k_proj = nn.Linear(d_model, self.d_inner, bias=False)
            self.v_proj = nn.Linear(d_model, self.d_inner, bias=False)

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.q_norm = RMSNorm(d_kv)
        self.k_norm = RMSNorm(d_kv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through attention head.

        Args:
            x: Input tensor (batch, seq_len, d_model).

        Returns:
            Output tensor (batch, seq_len, d_model).
        """
        batch, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.d_kv)

        if self.use_mla:
            c_kv = self.kv_norm(self.kv_down(x))
            k = self.k_up(c_kv).view(batch, seq_len, self.n_heads, self.d_kv)
            v = self.v_up(c_kv).view(batch, seq_len, self.n_heads, self.d_kv)
        else:
            k = self.k_proj(x).view(batch, seq_len, self.n_heads, self.d_kv)
            v = self.v_proj(x).view(batch, seq_len, self.n_heads, self.d_kv)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scale = math.sqrt(self.d_kv)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device),
            diagonal=1,
        )
        attn_weights = attn_weights.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(x.dtype)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, self.d_inner)
        return self.out_proj(attn_output)


class ParallelHybridHead(nn.Module):
    """Parallel SSM + Attention hybrid head (Hymba-inspired).

    Processes input through SSM and Attention pathways simultaneously,
    then combines outputs with learned gating or fixed weighting.

    The parallel design means:
    - SSM sees the original input (not post-attention)
    - Attention sees the original input (not post-SSM)
    - Both pathways contribute independently to the output
    - Meta tokens can be used for cross-layer information sharing

    Memory savings vs sequential:
    - SSM has no KV cache (constant state)
    - If Attention uses sliding window: O(W) instead of O(N)
    - Combined: SSM handles long-range, Attention handles local detail
    - Result: Can reduce attention layers needed, reducing total KV cache

    Args:
        config: ParallelHeadConfig with head parameters.
    """

    def __init__(self, config: ParallelHeadConfig) -> None:
        super().__init__()
        self.config = config
        self.d_model = config.d_model

        # Pre-norms
        self.ssm_norm = RMSNorm(config.d_model)
        self.attn_norm = RMSNorm(config.d_model)

        # Parallel pathways
        self.ssm_head = SSMHead(
            d_model=config.d_model,
            d_state=config.ssm_d_state,
            expand=config.ssm_expand,
            d_conv=config.ssm_d_conv,
        )
        self.attn_head = AttentionHead(
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_kv=config.d_kv,
            use_mla=config.use_mla,
            mla_latent_dim=config.mla_latent_dim,
        )

        # Fusion mechanism
        if config.use_gated_fusion:
            # Learned gating: input-dependent fusion weights
            self.fusion_gate = nn.Linear(config.d_model, 2, bias=False)
        else:
            # Fixed fusion weights
            self.register_buffer(
                'ssm_weight',
                torch.tensor(config.ssm_weight_init),
            )
            self.register_buffer(
                'attn_weight',
                torch.tensor(config.attn_weight_init),
            )

        # Meta tokens for cross-layer info (Hymba innovation)
        if config.num_meta_tokens > 0:
            self.meta_tokens = nn.Parameter(
                torch.randn(1, config.num_meta_tokens, config.d_model) * 0.02
            )
            self.meta_proj = nn.Linear(config.d_model * 2, config.d_model, bias=False)

        # Output norm
        self.output_norm = RMSNorm(config.d_model)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through parallel hybrid head.

        Args:
            x: Input tensor (batch, seq_len, d_model).
            attention_mask: Optional attention mask.

        Returns:
            Output tensor (batch, seq_len, d_model).
        """
        batch, seq_len, _ = x.shape

        # Parallel processing: both heads see the same input
        ssm_input = self.ssm_norm(x)
        attn_input = self.attn_norm(x)

        ssm_out = self.ssm_head(ssm_input)       # (batch, seq_len, d_model)
        attn_out = self.attn_head(attn_input)      # (batch, seq_len, d_model)

        # Meta token integration (prepend and blend)
        if hasattr(self, 'meta_tokens') and self.config.num_meta_tokens > 0:
            meta = self.meta_tokens.expand(batch, -1, -1)

            # Process meta tokens through both pathways
            meta_ssm = self.ssm_head(self.ssm_norm(meta))
            meta_attn = self.attn_head(self.attn_norm(meta))

            # Blend meta info into sequence tokens
            # Simple: use meta as additional context via projection
            meta_context = torch.cat([
                meta_ssm.mean(dim=1, keepdim=True).expand(-1, seq_len, -1),
                meta_attn.mean(dim=1, keepdim=True).expand(-1, seq_len, -1),
            ], dim=-1)
            meta_blend = self.meta_proj(meta_context)

            ssm_out = ssm_out + meta_blend * 0.1
            attn_out = attn_out + meta_blend * 0.1

        # Fusion
        if hasattr(self, 'fusion_gate'):
            # Learned gating
            gate_logits = self.fusion_gate(x)  # (batch, seq_len, 2)
            gate_weights = F.softmax(gate_logits, dim=-1)
            w_ssm = gate_weights[:, :, 0:1]
            w_attn = gate_weights[:, :, 1:2]
        else:
            w_ssm = self.ssm_weight
            w_attn = self.attn_weight

        combined = w_ssm * ssm_out + w_attn * attn_out

        # Residual + norm
        output = x + combined
        output = self.output_norm(output)

        return output
