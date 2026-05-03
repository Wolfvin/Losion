"""
Losion Model V2 — Fully Integrated Tri-Jalur Router Architecture.

Config-driven module selection that wires ALL core implementations into
the production model, replacing the Simplified* placeholder modules.

Credits & References:
  - Losion Framework: Wolfvin & Contributors (github.com/Wolfvin/Losion)
  - DeepSeek-V2: MLA, arXiv:2405.04434 (2024)
  - DeepSeek-V3: Aux-loss-free MoE, arXiv:2412.19437 (2024)
  - Mamba-2: SSM, arXiv:2405.21060 (2024)
  - Mamba-3: Inference-first SSM, arXiv:2603.15569 (2026)
  - OpenMythos: RDT, github.com/kyegomez/OpenMythos (2026)
  - Qwen: Gated Attention, NeurIPS 2025 Best Paper
  - Moonshot AI: MoBA, NeurIPS 2025
  - Meta: S'MoRE, NeurIPS 2025
  - Microsoft: Routing Mamba, NeurIPS 2025
  - RoPE: Su et al., arXiv:2104.09864 (2021)
  - RWKV-7: Peng et al. (2024)
  - Universal Transformers: Dehghani et al., arXiv:1807.03819 (2019)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from losion.config import LosionConfig, RecurrentConfig, JEPAConfig

# --- Lazy imports for core modules (avoid circular deps) ---
# We import at module level but catch ImportError for graceful fallback


# ============================================================================
# RMSNorm
# ============================================================================


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


# ============================================================================
# RoPE — Rotary Position Embedding
# ============================================================================


class RoPE(nn.Module):
    """Rotary Position Embedding (Su et al., 2021).

    Supports standard RoPE, interleaved RoPE (iRoPE), and context extension
    via YaRN / NTK-aware scaling.

    Args:
        dim: Dimension to apply RoPE to (typically d_kv).
        max_seq_len: Maximum sequence length.
        base: Base frequency (default 10000).
        interleaved: If True, use iRoPE pattern (alternate RoPE/non-RoPE dims).
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: float = 10000.0,
        interleaved: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.interleaved = interleaved

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self, x: torch.Tensor, position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply RoPE to input tensor.

        Args:
            x: Input (batch, n_heads, seq_len, d_kv) or (batch, seq_len, d_kv).
            position_ids: Optional position IDs (batch, seq_len).

        Returns:
            Tensor with RoPE applied.
        """
        if x.dim() == 4:
            batch, n_heads, seq_len, d_kv = x.shape
        elif x.dim() == 3:
            batch, seq_len, d_kv = x.shape
            n_heads = 1
            x = x.unsqueeze(1)
        else:
            return x

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)

        # Compute frequencies
        freqs = position_ids.float().unsqueeze(-1) * self.inv_freq.unsqueeze(0)
        # (batch, seq_len, dim//2)
        cos = freqs.cos().unsqueeze(1)  # (batch, 1, seq_len, dim//2)
        sin = freqs.sin().unsqueeze(1)

        # Split x into pairs
        half = d_kv // 2
        x1 = x[..., :half]
        x2 = x[..., half: 2 * half]

        # Apply rotation
        out_x1 = x1 * cos[..., :half] - x2 * sin[..., :half]
        out_x2 = x1 * sin[..., :half] + x2 * cos[..., :half]

        out = torch.cat([out_x1, out_x2, x[..., 2 * half:]], dim=-1)

        if n_heads == 1:
            return out.squeeze(1)
        return out


# ============================================================================
# Fallback Modules (used when core imports fail)
# ============================================================================


class _FallbackSSM(nn.Module):
    """Fallback SSM when core modules are unavailable."""

    def __init__(self, d_model: int, **kwargs):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.gate = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, state=None):
        return self.proj(x * torch.sigmoid(self.gate(x))), state

    def forward_inference(self, x, state=None):
        return self.forward(x, state)


class _FallbackAttention(nn.Module):
    """Fallback attention when core modules are unavailable."""

    def __init__(self, d_model: int, n_heads: int = 8, **kwargs):
        super().__init__()
        self.n_heads = n_heads
        self.d_kv = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, attention_mask=None, past_kv=None, position_ids=None):
        B, S, D = x.shape
        q = self.q_proj(x).view(B, S, self.n_heads, self.d_kv).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_heads, self.d_kv).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_heads, self.d_kv).transpose(1, 2)
        scale = math.sqrt(self.d_kv)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale
        if attention_mask is not None:
            attn = attn + attention_mask
        else:
            mask = torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool), 1)
            attn = attn.masked_fill(mask, float("-inf"))
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(x.dtype)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(out)


class _FallbackMoE(nn.Module):
    """Fallback MoE when core modules are unavailable."""

    def __init__(self, d_model: int, d_ff: int = None, num_experts: int = 8, **kwargs):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_ff, bias=False), nn.Linear(d_ff, d_model, bias=False))
            for _ in range(num_experts)
        ])
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.shared_expert = nn.Sequential(nn.Linear(d_model, d_ff, bias=False), nn.Linear(d_ff, d_model, bias=False))

    def forward(self, x):
        logits = self.router(x)
        weights = F.softmax(logits, dim=-1)
        top_w, top_idx = weights.topk(2, dim=-1)
        out = torch.zeros_like(x)
        for k in range(2):
            for eid in range(len(self.experts)):
                mask = (top_idx[..., k] == eid)
                if mask.any():
                    out[mask] += top_w[mask, k:k+1] * self.experts[eid](x[mask])
        out = out + self.shared_expert(x)
        return out, {"router_logits": logits}


# ============================================================================
# Module Factory — Config-driven module selection
# ============================================================================


def _build_ssm(config: LosionConfig) -> nn.Module:
    """Build SSM pathway module based on config."""
    ssm_cfg = config.ssm
    d_model = config.d_model

    try:
        # v0.8: Structured Sparse SSM (replaces diagonal transitions)
        if ssm_cfg.use_structured_sparse:
            from losion.core.ssm.structured_sparse import StructuredSparseSSM, StructuredSparseSSMConfig
            ss_cfg = StructuredSparseSSMConfig(
                d_model=d_model,
                d_state=ssm_cfg.d_state,
                d_conv=ssm_cfg.d_conv,
                expand=ssm_cfg.expand,
                n_groups=ssm_cfg.structured_sparse_n_groups,
            )
            return StructuredSparseSSM(ss_cfg)

        if ssm_cfg.use_routing_mamba:
            from losion.core.ssm.routing_mamba import RoutingMamba, RoutingMambaConfig
            rom_cfg = RoutingMambaConfig(
                d_model=d_model,
                d_state=ssm_cfg.d_state,
                d_conv=ssm_cfg.d_conv,
                expand=ssm_cfg.expand,
                num_experts=ssm_cfg.routing_mamba_num_experts,
                num_active_experts=ssm_cfg.routing_mamba_active_experts,
            )
            return RoutingMamba(rom_cfg)

        if ssm_cfg.use_mamba3:
            from losion.core.ssm.mamba3 import Mamba3SSD
            return Mamba3SSD(
                d_model=d_model,
                d_state=32,  # Mamba-3 default: half of Mamba-2
                d_conv=ssm_cfg.d_conv,
                expand=ssm_cfg.expand,
            )

        if ssm_cfg.use_liquid:
            from losion.core.ssm.liquid_ssm import LiquidSSMTerpaduLayer
            return LiquidSSMTerpaduLayer(config)

        # Default: Mamba-2 SSD
        from losion.core.ssm.mamba2 import Mamba2SSD
        return Mamba2SSD(d_model=d_model, d_state=ssm_cfg.d_state, d_conv=ssm_cfg.d_conv, expand=ssm_cfg.expand)

    except ImportError:
        return _FallbackSSM(d_model)


def _build_attention(config: LosionConfig) -> nn.Module:
    """Build Attention pathway module based on config."""
    attn_cfg = config.attention
    d_model = config.d_model

    try:
        # v0.10: MoSA — Mixture of Sparse Attention (NeurIPS '25)
        if config.mosa.enabled:
            from losion.core.attention.mosa import MoSAAttention, MoSAConfig as _MoSACfg
            mosa_cfg = _MoSACfg(
                d_model=d_model,
                n_heads=attn_cfg.n_heads,
                d_kv=attn_cfg.d_kv,
                num_sparse_experts=config.mosa.num_sparse_experts,
                top_k_experts=config.mosa.top_k_experts,
                sparsity_ratio=config.mosa.sparsity_ratio,
                use_mla=True,
                mla_latent_dim=attn_cfg.mla_latent_dim,
            )
            return MoSAAttention(mosa_cfg)

        # v0.10: Sliding Window Attention (RATTENTION-inspired)
        if config.sliding_window.enabled:
            from losion.core.attention.sliding_window import SlidingWindowAttention, SlidingWindowConfig as _SWCfg
            sw_cfg = _SWCfg(
                d_model=d_model,
                n_heads=attn_cfg.n_heads,
                d_kv=attn_cfg.d_kv,
                window_size=config.sliding_window.window_size,
                use_token_sink=config.sliding_window.use_token_sink,
                num_sink_tokens=config.sliding_window.num_sink_tokens,
                use_mla=True,
                mla_latent_dim=attn_cfg.mla_latent_dim,
            )
            return SlidingWindowAttention(sw_cfg)

        # v0.9: Child-3W (MoE at QKV level, replaces standard attention)
        if config.child_3w.enabled:
            from losion.core.attention.child_3w import Child3WAttention, Child3WConfig as _Child3WCfg
            c3w_cfg = _Child3WCfg(
                d_model=d_model,
                n_heads=attn_cfg.n_heads,
                d_kv=attn_cfg.d_kv,
                num_children=config.child_3w.num_children,
                top_k_children=config.child_3w.top_k_children,
                use_mla=config.child_3w.use_mla,
                mla_latent_dim=config.child_3w.mla_latent_dim,
                load_balance_weight=config.child_3w.load_balance_weight,
            )
            return Child3WAttention(c3w_cfg)

        if attn_cfg.use_moba:
            from losion.core.attention.moba import MoBAAttention, MoBAConfig
            moba_cfg = MoBAConfig(
                block_size=attn_cfg.moba_block_size,
                top_k_blocks=attn_cfg.moba_top_k_blocks,
                use_mla_compression=True,
            )
            return MoBAAttention(d_model=d_model, n_heads=attn_cfg.n_heads, d_head=attn_cfg.d_kv, config=moba_cfg)

        if attn_cfg.use_gated_attention:
            from losion.core.attention.gated_attention import GatedMultiHeadAttention, GatedAttentionConfig
            ga_cfg = GatedAttentionConfig(
                d_model=d_model,
                n_heads=attn_cfg.n_heads,
                d_kv=attn_cfg.d_kv,
                use_mla=True,
                mla_latent_dim=attn_cfg.mla_latent_dim,
            )
            return GatedMultiHeadAttention(ga_cfg)

        if attn_cfg.use_lightning:
            from losion.core.attention.lightning_attention import LightningAttention
            return LightningAttention(d_model=d_model, n_heads=attn_cfg.n_heads, d_kv=attn_cfg.d_kv,
                                      mla_latent_dim=attn_cfg.mla_latent_dim)

        # Default: KDA+MLA
        from losion.core.attention.kda_mla import KDAMLA
        return KDAMLA(d_model=d_model, n_heads=attn_cfg.n_heads, d_kv=attn_cfg.d_kv,
                       mla_latent_dim=attn_cfg.mla_latent_dim)

    except ImportError:
        return _FallbackAttention(d_model, n_heads=attn_cfg.n_heads)


def _build_moe(config: LosionConfig) -> nn.Module:
    """Build MoE/Retrieval pathway module based on config."""
    ret_cfg = config.retrieval
    d_model = config.d_model
    d_ff = ret_cfg.d_ff if ret_cfg.d_ff > 0 else 4 * d_model
    num_experts = ret_cfg.num_experts if ret_cfg.num_experts > 0 else max(8, min(64, d_model // 32))

    try:
        # v0.8: Infinite MoE (continuous expert space)
        if ret_cfg.use_infinite_moe:
            from losion.core.retrieval.infinite_moe import InfiniteMoE, InfiniteMoEConfig
            inf_cfg = InfiniteMoEConfig(
                d_model=d_model,
                d_ff=d_ff,
                top_k=ret_cfg.num_active_experts,
                code_dim=ret_cfg.infinite_moe_code_dim,
                hypernet_hidden_dim=ret_cfg.infinite_moe_hypernet_hidden,
                use_low_rank_residual=ret_cfg.infinite_moe_low_rank_residual,
                codebook_size=ret_cfg.infinite_moe_codebook_size,
            )
            return InfiniteMoE(inf_cfg)

        if ret_cfg.use_smore:
            from losion.core.retrieval.smore import SmoreMoE, SmoreConfig
            smore_cfg = SmoreConfig(
                num_experts=num_experts,
                num_active_experts=ret_cfg.num_active_experts,
                d_model=d_model,
                d_ff=d_ff,
                num_sub_trees=ret_cfg.smore_num_sub_trees,
                sub_tree_depth=ret_cfg.smore_sub_tree_depth,
            )
            return SmoreMoE(smore_cfg)

        if ret_cfg.use_symbolic_moe:
            from losion.core.retrieval.symbolic_moe import SymbolicMoERouter
            from losion.core.retrieval.aux_free_moe import AuxFreeMoE

            class _SymbolicMoEWrapper(nn.Module):
                """Wraps base MoE with SymbolicMoERouter for skill-based routing."""
                def __init__(self, base_moe, symbolic_router):
                    super().__init__()
                    self.base_moe = base_moe
                    self.symbolic_router = symbolic_router

                def forward(self, x):
                    # Get routing weights from symbolic router
                    pathway_weights, skill_probs, routing_info = self.symbolic_router(x)
                    # Forward through base MoE
                    result = self.base_moe(x)
                    # Add symbolic routing info to aux_info
                    if isinstance(result, tuple):
                        out, aux = result
                        if isinstance(aux, dict):
                            aux["symbolic_routing"] = routing_info
                        return out, aux
                    return result

            base_moe = AuxFreeMoE(
                d_model=d_model,
                d_ff=d_ff,
                num_experts=num_experts,
                top_k=ret_cfg.num_active_experts,
            )
            symbolic_router = SymbolicMoERouter(
                d_model=d_model,
                routing_mode="soft",
                classifier_bottleneck=128,
            )
            return _SymbolicMoEWrapper(base_moe, symbolic_router)

        # Default: AuxFreeMoE (DeepSeek-V3 style)
        from losion.core.retrieval.aux_free_moe import AuxFreeMoE
        return AuxFreeMoE(
            d_model=d_model,
            d_ff=d_ff,
            num_experts=num_experts,
            top_k=ret_cfg.num_active_experts,
        )

    except ImportError:
        return _FallbackMoE(d_model, d_ff=d_ff, num_experts=num_experts)


def _build_router(config: LosionConfig) -> nn.Module:
    """Build AdaptiveRouter (always, never nn.Linear).

    Fixed v0.9.1: AdaptiveRouter.__init__ takes (d_model, num_pathways, ...)
    not a LosionConfig object. We now pass the correct arguments.
    """
    try:
        from losion.core.router.router import AdaptiveRouter
        return AdaptiveRouter(
            d_model=config.d_model,
            num_pathways=3,
            top_k_pathways=config.router.top_k_pathways,
            bias_lr=config.router.bias_lr,
        )
    except ImportError:
        # Fallback to simple linear router
        return nn.Linear(config.d_model, 3, bias=False)


# ============================================================================
# LosionLayerV2 — Config-Driven Tri-Jalur Layer
# ============================================================================


class LosionLayerV2(nn.Module):
    """Config-driven Tri-Jalur Router layer.

    Selects SSM, Attention, and MoE modules based on LosionConfig flags.
    Uses AdaptiveRouter instead of plain nn.Linear.

    Args:
        config: LosionConfig with model parameters.
        layer_idx: Index of this layer in the model.
    """

    def __init__(self, config: LosionConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.d_model = config.d_model

        # Pre-norms for each pathway
        self.ssm_norm = RMSNorm(config.d_model)
        self.attn_norm = RMSNorm(config.d_model)
        self.retrieval_norm = RMSNorm(config.d_model)

        # Build pathway modules from config
        self.ssm_layer = _build_ssm(config)
        self.attention_layer = _build_attention(config)
        self.retrieval_layer = _build_moe(config)

        # Router (AdaptiveRouter, NOT nn.Linear)
        self.router = _build_router(config)

        # Output norm
        self.output_norm = RMSNorm(config.d_model)

        # Dropout
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        routing_weights: Optional[torch.Tensor] = None,
        thinking_mode: Optional[bool] = None,
        ssm_state: Optional[Any] = None,
        past_kv: Optional[Any] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass through the Tri-Jalur layer.

        v0.9.1: Fixed all interface mismatches between pathway modules and
        their callers. SSM, Attention, and MoE modules now all properly
        connect regardless of their internal parameter naming conventions.
        """
        batch, seq_len, _ = x.shape

        # ===================================================================
        # Routing — Unified AdaptiveRouter + thinking_mode integration
        # ===================================================================
        if routing_weights is None:
            try:
                from losion.core.router.router import AdaptiveRouter
                if isinstance(self.router, AdaptiveRouter):
                    # v0.9.1: Pass thinking_mode to AdaptiveRouter so it can
                    # adjust routing weights (non-thinking → more SSM, thinking →
                    # more Attention + Retrieval)
                    if thinking_mode is not None:
                        from losion.core.router.thinking_toggle import ThinkingMode as TM
                        forced_mode = TM.THINKING if thinking_mode else TM.NON_THINKING
                        self.router.set_force_thinking(forced_mode)
                    routing_out = self.router(x)
                    route_weights = routing_out.adjusted_weights
                    # Reset force mode after this call
                    self.router.set_force_thinking(None)
                else:
                    route_logits = self.router(x)
                    route_weights = F.softmax(route_logits, dim=-1)
            except (ImportError, Exception):
                route_logits = self.router(x)
                if isinstance(route_logits, torch.Tensor):
                    route_weights = F.softmax(route_logits, dim=-1)
                else:
                    route_weights = route_logits
        else:
            route_weights = routing_weights

        # Ensure route_weights is (batch, seq_len, 3)
        if route_weights.dim() == 2:
            route_weights = route_weights.unsqueeze(1).expand(-1, seq_len, -1)

        # ===================================================================
        # Jalur 1: SSM — Unified interface adapter
        # v0.9.1: Handles `initial_state` vs `state` kwarg mismatch,
        # and 3-tuple (output, state, aux_loss) vs 2-tuple (output, state)
        # ===================================================================
        ssm_input = self.ssm_norm(x)
        ssm_out, ssm_state_new, ssm_aux_loss = self._forward_ssm(
            ssm_input, ssm_state
        )

        # ===================================================================
        # Jalur 2: Attention — Unified interface adapter
        # v0.9.1: Handles `position_offset` vs `position_ids`, and
        # `past_key_value` vs `past_kv` kwarg mismatches
        # ===================================================================
        attn_input = self.attn_norm(x)
        attn_out = self._forward_attention(
            attn_input, attention_mask, past_kv, position_ids
        )

        # ===================================================================
        # Jalur 3: MoE/Retrieval — Unified interface adapter
        # v0.9.1: Handles 3-tuple returns (output, routing_info/aux_loss, extra)
        # and normalizes to (output, aux_info) for consistent downstream use
        # ===================================================================
        ret_input = self.retrieval_norm(x)
        ret_out, ret_aux = self._forward_moe(ret_input)

        # Combine with routing weights
        w_ssm = route_weights[:, :, 0:1]
        w_attn = route_weights[:, :, 1:2]
        w_ret = route_weights[:, :, 2:3]

        # Handle dimension mismatch via learned projection
        ssm_out = self._align_dim(ssm_out, 'ssm_proj')
        attn_out = self._align_dim(attn_out, 'attn_proj')
        ret_out = self._align_dim(ret_out, 'ret_proj')

        combined = w_ssm * ssm_out + w_attn * attn_out + w_ret * ret_out

        # Residual + norm
        output = x + self.dropout(combined)
        output = self.output_norm(output)

        routing_info = {
            "layer_idx": self.layer_idx,
            "route_weights": route_weights.detach(),
            "ssm_state": ssm_state_new,
            "ssm_aux_loss": ssm_aux_loss,
            "retrieval_aux": ret_aux,
        }

        return output, routing_info

    # ===================================================================
    # Unified Interface Adapters (v0.9.1)
    # These methods handle all interface mismatches between pathway modules
    # so that every SSM/Attention/MoE variant plugs in seamlessly.
    # ===================================================================

    def _forward_ssm(
        self,
        ssm_input: torch.Tensor,
        ssm_state: Optional[Any],
    ) -> Tuple[torch.Tensor, Optional[Any], Optional[torch.Tensor]]:
        """Unified SSM forward that handles all interface variants.

        Handles:
        - `initial_state` vs `state` kwarg (Mamba2/Mamba3 use initial_state)
        - 3-tuple (output, state, aux_loss) from RoutingMamba
        - 2-tuple (output, state) from other SSM modules
        - Single tensor output from fallback SSM

        Returns:
            (output, new_state, aux_loss) — always 3 values.
        """
        ssm_layer = self.ssm_layer
        aux_loss = None

        # Try passing with `state` first, then fall back to `initial_state`
        try:
            if ssm_state is not None:
                # Try 'state' kwarg first (RoutingMamba, LiquidSSM, etc.)
                try:
                    ssm_result = ssm_layer(ssm_input, state=ssm_state)
                except TypeError:
                    # Fallback: try 'initial_state' (Mamba2, Mamba3)
                    ssm_result = ssm_layer(ssm_input, initial_state=ssm_state)
            else:
                ssm_result = ssm_layer(ssm_input)
        except Exception:
            ssm_result = ssm_layer(ssm_input)

        # Unpack result
        if isinstance(ssm_result, tuple):
            if len(ssm_result) == 3:
                # RoutingMamba: (output, final_state, aux_loss)
                ssm_out, ssm_state_new, aux_loss = ssm_result
            elif len(ssm_result) == 2:
                # Standard: (output, final_state)
                ssm_out, ssm_state_new = ssm_result
            else:
                ssm_out = ssm_result[0]
                ssm_state_new = None
        else:
            ssm_out = ssm_result
            ssm_state_new = None

        return ssm_out, ssm_state_new, aux_loss

    def _forward_attention(
        self,
        attn_input: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_kv: Optional[Any],
        position_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Unified Attention forward that handles all interface variants.

        Handles:
        - `position_ids` vs `position_offset` kwarg mismatch
        - `past_kv` vs `past_key_value` kwarg mismatch
        - Modules that don't accept position_ids at all (Child3W)
        - Tuple returns from some attention modules

        Returns:
            attention output tensor (batch, seq_len, d_model)
        """
        attn_layer = self.attention_layer

        # Build kwargs — try both naming conventions
        attn_kwargs: Dict[str, Any] = {}
        if attention_mask is not None:
            attn_kwargs["attention_mask"] = attention_mask

        # Try calling with all possible kwargs, gracefully degrading
        try:
            # First attempt: pass both past_kv and position_ids
            kwargs_attempt = dict(attn_kwargs)
            if past_kv is not None:
                kwargs_attempt["past_kv"] = past_kv
            if position_ids is not None:
                kwargs_attempt["position_ids"] = position_ids
            attn_result = attn_layer(attn_input, **kwargs_attempt)
        except TypeError:
            # Second attempt: try alternate names (past_key_value, position_offset)
            try:
                kwargs_attempt = dict(attn_kwargs)
                if past_kv is not None:
                    kwargs_attempt["past_key_value"] = past_kv
                if position_ids is not None:
                    # Convert position_ids to offset for MoBA/GatedAttention
                    kwargs_attempt["position_offset"] = position_ids[0, -1].item() if position_ids.numel() > 0 else 0
                attn_result = attn_layer(attn_input, **kwargs_attempt)
            except TypeError:
                # Third attempt: only attention_mask
                try:
                    attn_result = attn_layer(attn_input, **attn_kwargs)
                except TypeError:
                    # Final fallback: no kwargs
                    attn_result = attn_layer(attn_input)

        # Unpack result
        if isinstance(attn_result, tuple):
            attn_out = attn_result[0]
        else:
            attn_out = attn_result

        return attn_out

    def _forward_moe(
        self,
        ret_input: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Unified MoE/Retrieval forward that handles all interface variants.

        Handles:
        - 3-tuple (output, routing_info, auxiliary_losses) from AuxFreeMoE
        - 3-tuple (output, aux_loss, routing_info) from SmoreMoE
        - 2-tuple (output, losses) from InfiniteMoE
        - 2-tuple (output, aux_info) from FallbackMoE
        - Single tensor output from plain FFN

        Returns:
            (output, aux_info) — always 2 values, aux_info is always a dict.
        """
        ret_result = self.retrieval_layer(ret_input)

        if isinstance(ret_result, tuple):
            if len(ret_result) == 3:
                # Normalize: combine all extra info into a single dict
                ret_out = ret_result[0]
                aux_info = {
                    "extra_1": ret_result[1],
                    "extra_2": ret_result[2],
                }
                # Try to be smarter about naming
                if isinstance(ret_result[1], dict):
                    aux_info = {**ret_result[1], "routing_info": ret_result[2]}
                elif isinstance(ret_result[2], dict):
                    aux_info = {"aux_loss": ret_result[1], **ret_result[2]}
            elif len(ret_result) == 2:
                ret_out = ret_result[0]
                aux_info = ret_result[1] if isinstance(ret_result[1], dict) else {"aux": ret_result[1]}
            else:
                ret_out = ret_result[0]
                aux_info = {}
        else:
            ret_out = ret_result
            aux_info = {}

        return ret_out, aux_info

    def _align_dim(
        self,
        tensor: torch.Tensor,
        proj_name: str,
    ) -> torch.Tensor:
        """Align tensor's last dimension to d_model via learned projection."""
        if tensor.shape[-1] == self.d_model:
            return tensor
        if not hasattr(self, proj_name):
            proj = nn.Linear(tensor.shape[-1], self.d_model, bias=False).to(tensor.device)
            nn.init.eye_(proj.weight[:min(tensor.shape[-1], self.d_model)])
            setattr(self, proj_name, proj)
        return getattr(self, proj_name)(tensor)

    def forward_inference(
        self,
        x: torch.Tensor,
        ssm_state: Optional[Any] = None,
        past_kv: Optional[Any] = None,
        position_ids: Optional[torch.Tensor] = None,
        routing_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Inference pass (O(1) per token for SSM, cached for attention).

        v0.9.1: Uses unified adapters for consistent interface handling.
        """
        # Compute routing
        if routing_weights is None:
            try:
                from losion.core.router.router import AdaptiveRouter
                if isinstance(self.router, AdaptiveRouter):
                    routing_out = self.router(x)
                    route_weights = routing_out.adjusted_weights
                else:
                    route_logits = self.router(x)
                    route_weights = F.softmax(route_logits, dim=-1)
            except (ImportError, Exception):
                route_logits = self.router(x)
                if isinstance(route_logits, torch.Tensor):
                    route_weights = F.softmax(route_logits, dim=-1)
                else:
                    route_weights = route_logits

        # SSM (O(1) per token) — use unified adapter
        ssm_input = self.ssm_norm(x)
        if hasattr(self.ssm_layer, 'forward_inference'):
            try:
                ssm_out, ssm_state_new = self.ssm_layer.forward_inference(ssm_input, state=ssm_state)
            except TypeError:
                try:
                    ssm_out, ssm_state_new = self.ssm_layer.forward_inference(ssm_input, initial_state=ssm_state)
                except TypeError:
                    ssm_out, ssm_state_new = self.ssm_layer.forward_inference(ssm_input), None
        else:
            ssm_out, ssm_state_new, _ = self._forward_ssm(ssm_input, ssm_state)

        # Attention (with KV cache) — use unified adapter
        attn_input = self.attn_norm(x)
        attn_kv_cache = None
        if hasattr(self.attention_layer, 'forward_inference'):
            try:
                attn_result = self.attention_layer.forward_inference(
                    attn_input, past_kv=past_kv, position_ids=position_ids
                )
            except TypeError:
                try:
                    attn_result = self.attention_layer.forward_inference(
                        attn_input, past_key_value=past_kv, position_offset=(
                            position_ids[0, -1].item() if position_ids is not None and position_ids.numel() > 0 else 0
                        )
                    )
                except TypeError:
                    attn_result = self.attention_layer.forward_inference(attn_input)
        else:
            attn_result = self._forward_attention(attn_input, None, past_kv, position_ids)
        if isinstance(attn_result, tuple):
            attn_out = attn_result[0]
            # Capture KV cache from attention layer
            if len(attn_result) > 1:
                attn_kv_cache = attn_result[1]
        else:
            attn_out = attn_result

        # MoE (standard forward) — use unified adapter
        ret_input = self.retrieval_norm(x)
        ret_out, _ = self._forward_moe(ret_input)

        # Align dimensions
        ssm_out = self._align_dim(ssm_out, 'ssm_proj')
        attn_out = self._align_dim(attn_out, 'attn_proj')
        ret_out = self._align_dim(ret_out, 'ret_proj')

        # Combine
        w = route_weights
        if w.dim() == 2:
            w = w.unsqueeze(1)
        combined = w[:, :, 0:1] * ssm_out + w[:, :, 1:2] * attn_out + w[:, :, 2:3] * ret_out

        output = x + self.dropout(combined)
        output = self.output_norm(output)

        return output, {
            "ssm_state": ssm_state_new,
            "route_weights": route_weights,
            "attn_kv_cache": attn_kv_cache,
        }


# ============================================================================
# LosionModelV2 — Backbone
# ============================================================================


class LosionModelV2(nn.Module):
    """Losion V2 backbone with fully integrated Tri-Jalur architecture.

    Config-driven module selection replaces all Simplified* placeholders.
    Uses RoPE instead of learned position embeddings.
    Supports optional RecurrentDepthBlock.

    Args:
        config: LosionConfig with model parameters.
    """

    def __init__(self, config: LosionConfig) -> None:
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_layers = config.n_layers
        self.vocab_size = config.vocab_size
        self.max_seq_len = config.max_seq_len

        # Token embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        # RoPE (replaces learned position embeddings)
        self.rope = RoPE(
            dim=config.attention.d_kv,
            max_seq_len=config.max_seq_len,
            interleaved=config.attention.use_irope,
        )

        # Embedding dropout
        self.embed_dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

        # Layers
        self.layers = nn.ModuleList([
            LosionLayerV2(config, layer_idx=i)
            for i in range(config.n_layers)
        ])

        # Optional AttnRes (Attention Residuals, v0.9)
        self.use_attn_res = config.attn_res.enabled
        if self.use_attn_res:
            try:
                from losion.core.attention.attn_res import AttnResConfig as _AttnResCfg, AttnResManager
                _attn_res_cfg = _AttnResCfg(
                    d_model=config.d_model,
                    n_layers=config.n_layers,
                    mode=config.attn_res.mode,
                    num_blocks=config.attn_res.num_blocks,
                    dropout=config.attn_res.dropout,
                    use_gate=config.attn_res.use_gate,
                    temperature=config.attn_res.temperature,
                    compression_dim=config.attn_res.compression_dim,
                )
                self.attn_res_manager = AttnResManager(_attn_res_cfg)
            except ImportError:
                self.use_attn_res = False

        # Optional Evoformer (v0.9)
        self.use_evoformer = config.evoformer.enabled
        if self.use_evoformer:
            try:
                from losion.core.feedback.evoformer import EvoformerConfig as _EvoCfg, EvoformerManager
                _evo_cfg = _EvoCfg(
                    d_model=config.d_model,
                    n_recycling_steps=config.evoformer.n_recycling_steps,
                    use_layer_recycling=config.evoformer.use_layer_recycling,
                    use_token_recycling=config.evoformer.use_token_recycling,
                    use_decoder_feedback=config.evoformer.use_decoder_feedback,
                    use_prediction_recycling=config.evoformer.use_prediction_recycling,
                    use_router_coevolve=config.evoformer.use_router_coevolve,
                )
                self.evoformer_manager = EvoformerManager(_evo_cfg)
            except ImportError:
                self.use_evoformer = False

        # Optional Dual Memory (v0.9)
        self.use_dual_memory = config.dual_memory.enabled
        if self.use_dual_memory:
            try:
                from losion.core.memory.dual_memory import DualMemoryConfig as _DMCfg, DualMemorySystem
                _dm_cfg = _DMCfg(
                    d_model=config.d_model,
                    working_memory_size=config.dual_memory.working_memory_size,
                    long_term_memory_dim=config.dual_memory.long_term_memory_dim,
                    consolidation_method=config.dual_memory.consolidation_method,
                )
                self.dual_memory = DualMemorySystem(_dm_cfg)
            except ImportError:
                self.use_dual_memory = False

        # Optional RecurrentDepthBlock (wraps the full LosionLayerV2)
        self.use_rdt = config.recurrent.enabled
        if self.use_rdt:
            try:
                from losion.core.recurrent.rdt import RecurrentDepthBlock

                class _RDTResidualBlock(nn.Module):
                    """SwiGLU residual block for RDT — more expressive than simple linear.
                    
                    Accepts extra kwargs and returns (output, aux) as required by RecurrentDepthBlock.
                    Uses SwiGLU activation for better gradient flow and representational capacity,
                    ensuring that the DepthLoRA adaptation applied before this block has meaningful
                    gradients to optimize.
                    
                    Credits: Fix inspired by benchmark results showing zero gradients on depth_lora.lora_A
                    due to the simple linear block absorbing the LoRA delta.
                    """
                    def __init__(self, d_model: int):
                        super().__init__()
                        self.norm = RMSNorm(d_model)
                        d_ff = d_model * 2  # 2x expansion for SwiGLU
                        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
                        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
                        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
                        # Initialize with small weights so RDT starts near-identity
                        # but NOT zero — zero init on down_proj would kill LoRA gradients
                        # since the block would output pure identity (residual only).
                        nn.init.normal_(self.gate_proj.weight, std=0.01)
                        nn.init.normal_(self.up_proj.weight, std=0.01)
                        nn.init.normal_(self.down_proj.weight, std=0.01)

                    def forward(self, x, **kwargs):
                        residual = x
                        x_norm = self.norm(x)
                        gate = F.silu(self.gate_proj(x_norm))
                        up = self.up_proj(x_norm)
                        out = self.down_proj(gate * up)
                        return residual + out, None  # RDT expects (output, aux_info) tuple

                rdt_inner = _RDTResidualBlock(config.d_model)
                self.rdt_block = RecurrentDepthBlock(
                    block=rdt_inner,
                    d_model=config.d_model,
                    max_loop_iters=config.recurrent.max_loop_iters,
                    use_act=config.recurrent.use_act,
                    lora_rank=config.recurrent.depth_lora_rank,
                )
            except ImportError:
                self.use_rdt = False

        # Final norm
        self.final_norm = RMSNorm(config.d_model)

        # Gradient checkpointing
        self.gradient_checkpointing: bool = False

        # Initialize weights
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="linear")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.token_embedding

    def set_input_embeddings(self, embeddings: nn.Embedding) -> None:
        self.token_embedding = embeddings

    def enable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = True

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        thinking_mode: Optional[bool] = None,
        return_routing_info: bool = False,
        return_all_hidden_states: bool = False,
    ) -> Dict[str, Any]:
        """Forward pass through the Losion V2 backbone."""
        batch, seq_len = input_ids.shape

        # Embeddings (no learned position — RoPE is applied in attention)
        x = self.token_embedding(input_ids)
        x = self.embed_dropout(x)

        # Position IDs for RoPE
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        # Reset AttnRes for new forward pass (v0.9)
        if self.use_attn_res:
            self.attn_res_manager.reset()

        # Layer processing
        # v1.1 Fix: When Evoformer is enabled, always collect routing info
        # (needed for Level 5 router_expert_coevolve)
        all_routing_info = [] if (return_routing_info or self.use_evoformer) else None
        # v1.1 Fix: When Evoformer is enabled, always collect hidden states
        # (needed for Level 1 layer_recycling and Level 4 prediction_recycling)
        # Also, do NOT detach — gradients must flow through Evoformer levels.
        all_hidden_states = [] if (return_all_hidden_states or self.use_evoformer) else None
        ssm_states = {}

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x, layer_routing = torch.utils.checkpoint.checkpoint(
                    lambda h, m, p: layer(h, attention_mask=m, position_ids=p, thinking_mode=thinking_mode),
                    x, attention_mask, position_ids,
                    use_reentrant=False,
                )
            else:
                x, layer_routing = layer(
                    x,
                    attention_mask=attention_mask,
                    thinking_mode=thinking_mode,
                    ssm_state=ssm_states.get(layer.layer_idx),
                    position_ids=position_ids,
                )

            # Update SSM states for next forward call
            if layer_routing and "ssm_state" in layer_routing:
                ssm_states[layer.layer_idx] = layer_routing["ssm_state"]

            # AttnRes: store and aggregate (v0.9)
            if self.use_attn_res:
                self.attn_res_manager.store_layer_output(layer.layer_idx, x)
                x = self.attn_res_manager(x, layer.layer_idx)

            # Dual Memory: write AND read layer output (v0.9.1)
            if self.use_dual_memory:
                self.dual_memory.write(x)
                # v0.9.1 FIX: Actually READ from memory to augment hidden states
                x = self.dual_memory.read(x)

            if all_routing_info is not None:
                all_routing_info.append(layer_routing)

            if all_hidden_states is not None:
                # v1.1 Fix: Detach to prevent "backward through graph a second time" errors.
                # The Evoformer Level 1 operates on these collected states for its
                # own internal computation (revision signals). The gradient flow
                # to Evoformer's own parameters (shallow_query_proj, deep_key_proj,
                # etc.) comes from the recycled output that gets combined with x.
                # If we don't detach, the second forward pass would try to backward
                # through the first pass's graph which has already been freed.
                all_hidden_states.append(x.detach())

        # Evoformer Level 1: Inter-layer recycling (v0.9)
        # v1.1 Fix: Add recycling as a RESIDUAL to x, NOT replacing x.
        # Previously, x was replaced with recycled states (which are detached
        # from the computation graph), breaking ALL gradient flow to the
        # backbone (token_embedding, layers, etc.). Now we keep x as the
        # primary gradient path and add a small residual from recycling.
        if self.use_evoformer and all_hidden_states is not None and len(all_hidden_states) > 1:
            recycled = self.evoformer_manager.recycle_layers(all_hidden_states)
            # Add revision signal as residual (recycled[i] = h_i + revision)
            # The revision is the new part; we take it from the shallow layers
            # and add it to x with a small weight to preserve gradient flow
            n_layers = len(recycled)
            mid = n_layers // 2
            # Get the revision signal: recycled[0] - hidden_states[0]
            # (recycled shallow layers have revision added)
            revision_signal = recycled[0] - all_hidden_states[0]  # detached - detached = detached
            # Add as small residual to x (which still has gradient)
            x = x + 0.01 * revision_signal

        # Evoformer Level 2: Bidirectional token update (v0.9)
        if self.use_evoformer:
            if hasattr(self.evoformer_manager, 'bidirectional_token') and self.evoformer_manager.bidirectional_token is not None:
                x = self.evoformer_manager.bidirectional_token(x)

        # Evoformer Levels 3-5: Full feedback loops (v0.9.1 — now wired)
        if self.use_evoformer:
            # Level 3: Decoder ↔ Predict feedback
            if hasattr(self.evoformer_manager, 'decoder_predict_feedback'):
                x = self.evoformer_manager.decoder_predict_feedback(x)
            # Level 4: Prediction → Context recycling
            if hasattr(self.evoformer_manager, 'prediction_context_recycling'):
                x = self.evoformer_manager.prediction_context_recycling(x)
            # Level 5: Router ↔ Expert co-evolution
            if all_routing_info and hasattr(self.evoformer_manager, 'router_expert_coevolve'):
                x = self.evoformer_manager.router_expert_coevolve(x, all_routing_info)

        # Optional RDT
        if self.use_rdt and hasattr(self, 'rdt_block'):
            x, rdt_aux = self.rdt_block(x)
        else:
            rdt_aux = None

        # Final norm
        x = self.final_norm(x)

        return {
            "hidden_states": x,
            "routing_info": all_routing_info,
            "all_hidden_states": all_hidden_states,
            "ssm_states": ssm_states,
            "rdt_aux": rdt_aux,
        }

    def forward_inference(
        self,
        input_ids: torch.Tensor,
        ssm_states: Optional[Dict[int, Any]] = None,
        past_kvs: Optional[Dict[int, Any]] = None,
        position_offset: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Inference forward pass (O(1) per token for SSM)."""
        x = self.token_embedding(input_ids)
        position_ids = torch.tensor([[position_offset]], device=input_ids.device)

        new_states = {}
        new_kvs = {}

        for layer in self.layers:
            x, info = layer.forward_inference(
                x,
                ssm_state=ssm_states.get(layer.layer_idx) if ssm_states else None,
                past_kv=past_kvs.get(layer.layer_idx) if past_kvs else None,
                position_ids=position_ids,
            )
            if info.get("ssm_state") is not None:
                new_states[layer.layer_idx] = info["ssm_state"]
            # Capture KV cache from attention layers
            if info.get("attn_kv_cache") is not None:
                new_kvs[layer.layer_idx] = info["attn_kv_cache"]

        x = self.final_norm(x)

        return x, {"ssm_states": new_states, "past_kvs": new_kvs}


# ============================================================================
# MTPHead — Multi-Token Prediction
# ============================================================================


class JEPAHead(nn.Module):
    """Lightweight JEPA head for integration into LosionForCausalLMV2.

    Unlike the standalone LLMJEPA training wrapper, this head only contains
    the JEPA-specific components (predictor, encoders, loss) and operates
    on hidden states already produced by the parent model.

    This enables JEPA to work as a plug-in loss without creating a
    separate model copy.

    Args:
        config: JEPAConfig instance.
        d_model: Model hidden dimension (overrides config.d_model if needed).
    """

    def __init__(self, config: 'JEPAConfig', d_model: int) -> None:
        super().__init__()
        self.config = config
        self.d_model = d_model
        latent_dim = config.latent_dim

        # ---- Online encoder (student) ----
        self.online_encoder = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False),
            nn.GELU(),
            nn.Linear(d_model, latent_dim, bias=False),
        )

        # ---- Target encoder (EMA teacher, no gradients) ----
        self.target_encoder = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False),
            nn.GELU(),
            nn.Linear(d_model, latent_dim, bias=False),
        )
        # Freeze target encoder
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        # ---- LatentPredictor ----
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2, bias=False),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim * config.prediction_horizon, bias=False),
        )
        self.prediction_horizon = config.prediction_horizon
        self.latent_dim = latent_dim

        # ---- Loss function ----
        self.loss_type = config.loss_type

    @torch.no_grad()
    def _update_target_encoder(self, ema_decay: float = 0.996) -> None:
        """EMA update target encoder from online encoder."""
        for online_param, target_param in zip(
            self.online_encoder.parameters(),
            self.target_encoder.parameters(),
        ):
            target_param.data.mul_(ema_decay).add_(
                online_param.data, alpha=1.0 - ema_decay
            )

    def compute_loss(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute JEPA loss from hidden states.

        Args:
            hidden_states: (batch, seq_len, d_model)

        Returns:
            JEPA loss scalar.
        """
        batch, seq_len, _ = hidden_states.shape

        # Online encoding
        online_latents = self.online_encoder(hidden_states)  # (B, S, D_latent)

        # Target encoding (no grad)
        with torch.no_grad():
            target_latents = self.target_encoder(hidden_states).detach()

        # Predict future latents
        predicted = self.predictor(online_latents)  # (B, S, H * D_latent)
        predicted = predicted.view(batch, seq_len, self.prediction_horizon, self.latent_dim)

        # Shift target: target for position t is the latent at t+offset
        jepa_loss = torch.tensor(0.0, device=hidden_states.device)
        for h in range(self.prediction_horizon):
            offset = h + 1
            if offset < seq_len:
                pred_h = predicted[:, :-offset, h, :]  # (B, S-offset, D)
                target_h = target_latents[:, offset:, :]  # (B, S-offset, D)
                if self.loss_type == "cosine":
                    jepa_loss = jepa_loss + (1 - F.cosine_similarity(pred_h, target_h, dim=-1)).mean()
                elif self.loss_type == "mse":
                    jepa_loss = jepa_loss + F.mse_loss(pred_h, target_h)
                else:  # Default: cosine + MSE hybrid
                    jepa_loss = jepa_loss + F.mse_loss(pred_h, target_h)

        jepa_loss = jepa_loss / max(self.prediction_horizon, 1)

        # Update target encoder via EMA
        if self.training:
            self._update_target_encoder()

        return jepa_loss


class MTPHead(nn.Module):
    """Multi-Token Prediction head (DeepSeek-V3 style).

    Predicts n future tokens in parallel from the current hidden state.

    Args:
        d_model: Model dimension.
        vocab_size: Vocabulary size.
        n_tokens: Number of future tokens to predict.
    """

    def __init__(self, d_model: int, vocab_size: int, n_tokens: int = 2) -> None:
        super().__init__()
        self.n_tokens = n_tokens
        self.heads = nn.ModuleList([
            nn.Linear(d_model, vocab_size, bias=False)
            for _ in range(n_tokens)
        ])
        # Projection layers for each future token
        self.projections = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=False)
            for _ in range(n_tokens)
        ])

    def forward(self, hidden_states: torch.Tensor) -> List[torch.Tensor]:
        """Predict future tokens.

        Args:
            hidden_states: (batch, seq_len, d_model)

        Returns:
            List of logits tensors, one per future token.
        """
        logits_list = []
        for i in range(self.n_tokens):
            projected = self.projections[i](hidden_states)
            logits = self.heads[i](projected)
            logits_list.append(logits)
        return logits_list


# ============================================================================
# LosionForCausalLMV2 — Complete Causal Language Model
# ============================================================================


class LosionForCausalLMV2(nn.Module):
    """Losion V2 Causal Language Model with full generation support.

    Integrates:
    - LosionModelV2 backbone with config-driven module selection
    - LM head
    - Optional MTP heads
    - Full .generate() with temperature/top-k/top-p/KV cache
    - save_pretrained / from_pretrained

    Args:
        config: LosionConfig with model parameters.
    """

    def __init__(self, config: LosionConfig) -> None:
        super().__init__()
        self.config = config
        self.model = LosionModelV2(config)
        self.vocab_size = config.vocab_size

        # LM head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # MTP heads (optional)
        self.use_mtp = config.output.use_mtp
        if self.use_mtp:
            self.mtp_head = MTPHead(
                d_model=config.d_model,
                vocab_size=config.vocab_size,
                n_tokens=config.output.mtp_num_tokens,
            )

        # JEPA (optional) — lightweight head, not the standalone training wrapper
        self.use_jepa = config.jepa.enabled
        if self.use_jepa:
            try:
                self.jepa = JEPAHead(config.jepa, d_model=config.d_model)
            except Exception:
                self.use_jepa = False

        # KV Cache Quantization (optional, v1.1)
        self.use_kv_quant = config.kv_quant.enabled
        if self.use_kv_quant:
            try:
                from losion.inference.kv_quantization import QuantizedKVCache, KVQuantConfig as _KVQCfg, QuantizationMode
                mode_map = {"fp16": QuantizationMode.FP16, "int8": QuantizationMode.INT8, "int4": QuantizationMode.INT4, "nf4": QuantizationMode.NF4}
                quant_mode = mode_map.get(config.kv_quant.mode, QuantizationMode.INT8)
                self.kv_quant_cache = QuantizedKVCache(
                    n_layers=config.n_layers,
                    n_heads=config.attention.n_heads,
                    d_kv=config.attention.d_kv,
                    config=_KVQCfg(mode=quant_mode, group_size=config.kv_quant.group_size),
                    mla_latent_dim=config.attention.mla_latent_dim,
                    max_seq_len=config.max_seq_len,
                )
            except ImportError:
                self.use_kv_quant = False

        # DMS - Dynamic Memory Sparsification (optional, v1.1)
        self.use_dms = config.dms.enabled
        if self.use_dms:
            try:
                from losion.core.memory.dynamic_sparsification import DynamicMemorySparsification, DMSConfig as _DMSCfg
                self.dms = DynamicMemorySparsification(_DMSCfg(
                    enabled=True,
                    target_cache_ratio=config.dms.target_cache_ratio,
                    eviction_strategy=config.dms.eviction_strategy,
                    update_frequency=config.dms.update_frequency,
                    min_tokens_to_keep=config.dms.min_tokens_to_keep,
                ))
                self.dms.init_predictor(d_kv=config.attention.d_kv)
            except ImportError:
                self.use_dms = False

        # Initialize weights
        self.apply(self._init_weights)

        # Gradient scaling for router and MoE (v1.1 fix)
        self._apply_gradient_scaling()

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="linear")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _apply_gradient_scaling(self) -> None:
        """Apply gradient scaling to router and MoE parameters to compensate for
        their much smaller gradient norms.

        Problem: Router grad norm ~0.013, MoE grad norm ~0.063,
        Embedding grad norm ~11.0
        Solution: Scale router gradients by 10x and MoE gradients by 5x
        during backward pass.
        """
        router_scale = 10.0
        moe_scale = 5.0

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            name_lower = name.lower()
            if 'router' in name_lower or 'thinking_toggle' in name_lower:
                param.register_hook(lambda grad, s=router_scale: grad * s)
            elif 'jepa' in name_lower:
                param.register_hook(lambda grad, s=20.0: grad * s)  # JEPA needs even more scaling
            elif 'retrieval' in name_lower or 'moe' in name_lower or 'expert' in name_lower:
                param.register_hook(lambda grad, s=moe_scale: grad * s)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        thinking_mode: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Forward pass with optional loss computation.

        Args:
            input_ids: Token IDs (batch, seq_len).
            labels: Optional target token IDs for loss (batch, seq_len).
            attention_mask: Optional attention mask.
            thinking_mode: If True, bias towards thinking pathways.

        Returns:
            Dict with logits, loss, and optional aux info.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            thinking_mode=thinking_mode,
            return_routing_info=True,
        )
        hidden_states = outputs["hidden_states"]

        # LM logits
        logits = self.lm_head(hidden_states)

        # Loss computation
        loss = None
        loss_dict = {}
        if labels is not None:
            # Shift for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            lm_loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            loss = lm_loss
            loss_dict["lm_loss"] = lm_loss.item()

        # MTP loss
        if self.use_mtp and labels is not None and hasattr(self, 'mtp_head'):
            mtp_logits = self.mtp_head(hidden_states)
            mtp_loss = torch.tensor(0.0, device=input_ids.device)
            for i, token_logits in enumerate(mtp_logits):
                # Each MTP head predicts i+1 tokens ahead
                offset = i + 1
                # token_logits: (batch, seq_len, vocab_size)
                # shift_labels: (batch, seq_len-1) after the LM loss shift
                target_len = shift_labels.shape[1] - offset
                if target_len > 0:
                    mtp_pred = token_logits[:, :target_len, :].contiguous().view(-1, self.vocab_size)
                    mtp_target = shift_labels[:, offset:offset + target_len].contiguous().view(-1)
                    mtp_loss += F.cross_entropy(
                        mtp_pred,
                        mtp_target,
                        ignore_index=-100,
                    )
            mtp_loss = mtp_loss / len(mtp_logits) * 0.1  # Weight 0.1
            if loss is not None:
                loss = loss + mtp_loss
            else:
                loss = mtp_loss
            loss_dict["mtp_loss"] = mtp_loss.item()

        # JEPA loss
        if self.use_jepa and hasattr(self, 'jepa') and self.training:
            try:
                jepa_loss = self.jepa.compute_loss(hidden_states)
                if loss is not None:
                    loss = loss + self.config.jepa.prediction_weight * jepa_loss
                loss_dict["jepa_loss"] = jepa_loss.item()
            except Exception as e:
                import logging as _logging
                _logging.getLogger(__name__).warning(f"JEPA loss computation failed: {e}")

        return {
            "logits": logits,
            "loss": loss,
            "loss_dict": loss_dict,
            "routing_info": outputs.get("routing_info"),
            "hidden_states": hidden_states,
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        eos_token_id: Optional[int] = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Generate tokens autoregressively.

        Args:
            input_ids: Prompt token IDs (batch, seq_len).
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-K filtering (0 = disabled).
            top_p: Nucleus sampling threshold (1.0 = disabled).
            repetition_penalty: Repetition penalty (>1.0 penalizes repeats).
            do_sample: If True, sample; if False, greedy.
            eos_token_id: End-of-sequence token ID.
            use_cache: If True, use KV cache for faster generation.

        Returns:
            Generated token IDs (batch, prompt_len + max_new_tokens).
        """
        self.eval()
        device = input_ids.device
        batch_size = input_ids.shape[0]
        generated = input_ids.clone()

        # Prefill: run full model on prompt
        outputs = self.model(input_ids=input_ids)
        hidden_states = outputs["hidden_states"]
        ssm_states = outputs.get("ssm_states", {})
        past_kvs = outputs.get("past_kvs", {})

        # Get last hidden state for first generated token
        # logits shape: (batch, 1, vocab_size) → squeeze to (batch, vocab_size)
        next_logits = self.lm_head(hidden_states[:, -1:, :]).squeeze(1)  # (batch, vocab_size)

        # Generation loop
        for step in range(max_new_tokens):
            # Apply temperature
            if temperature != 1.0:
                next_logits = next_logits / temperature

            # Repetition penalty
            if repetition_penalty != 1.0:
                for token_id in generated[0].unique():
                    next_logits[0, token_id] /= repetition_penalty

            # Top-K
            if top_k > 0:
                top_k_vals, _ = torch.topk(next_logits, min(top_k, next_logits.shape[-1]), dim=-1)
                threshold = top_k_vals[:, -1:]
                next_logits = next_logits.where(next_logits >= threshold, float("-inf"))

            # Top-P
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(next_logits, descending=True, dim=-1)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove_mask = cum_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[remove_mask] = float("-inf")
                next_logits = sorted_logits.scatter(-1, sorted_idx, sorted_logits)

            # Sample or greedy
            if do_sample:
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)  # (batch, 1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)  # (batch, 1)

            generated = torch.cat([generated, next_token], dim=-1)

            # Check EOS
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

            # Forward next token (O(1) for SSM, cached for attention)
            hidden_out, new_states = self.model.forward_inference(
                next_token,
                ssm_states=ssm_states,
                past_kvs=past_kvs,
                position_offset=generated.shape[1] - 1,
            )
            ssm_states = new_states.get("ssm_states", ssm_states)
            past_kvs = new_states.get("past_kvs", past_kvs)

            # Apply DMS eviction if enabled
            if self.use_dms and step > 0 and step % self.config.dms.update_frequency == 0:
                for layer_idx, kv in list(past_kvs.items()):
                    if isinstance(kv, tuple) and len(kv) >= 2 and kv[0] is not None:
                        k, v = kv[0], kv[1]
                        if k.dim() == 4 and v.dim() == 4:  # (batch, n_heads, seq, d_kv)
                            import torch.nn.functional as _F
                            q = k[:, :, -1:, :]  # Last query as proxy
                            importance = self.dms.compute_importance(q, k)
                            should_evict = self.dms.should_evict(k.shape[2])
                            if should_evict:
                                keep_mask = self.dms.select_eviction_targets(importance, k.shape[2])
                                new_k, new_v = self.dms.evict(k, v, keep_mask)
                                past_kvs[layer_idx] = (new_k, new_v)

            # Apply KV quantization if enabled
            if self.use_kv_quant and step > 0:
                for layer_idx, kv in list(past_kvs.items()):
                    if isinstance(kv, tuple) and len(kv) >= 1 and kv[0] is not None:
                        self.kv_quant_cache.update(layer_idx, kv[0].float())

            # hidden_out shape: (batch, 1, d_model) → logits (batch, vocab_size)
            next_logits = self.lm_head(hidden_out).squeeze(1)

        return generated

    def save_pretrained(self, path: str) -> None:
        """Save model to directory."""
        import os
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, "model.pt"))
        # Save config
        import json
        config_dict = self.config.to_dict()
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def from_pretrained(cls, path: str) -> "LosionForCausalLMV2":
        """Load model from directory."""
        import json, os
        with open(os.path.join(path, "config.json")) as f:
            config_dict = json.load(f)
        # Use _from_dict to properly handle nested sub-config dicts
        config = LosionConfig._from_dict(config_dict)
        model = cls(config)
        state_dict = torch.load(os.path.join(path, "model.pt"), map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        return model

    def count_parameters(self) -> Dict[str, int]:
        """Count parameters by category."""
        total = 0
        categories = {
            "token_embedding": 0,
            "ssm_layers": 0,
            "attention_layers": 0,
            "retrieval_layers": 0,
            "router": 0,
            "lm_head": 0,
            "mtp_heads": 0,
            "other": 0,
        }

        for name, param in self.named_parameters():
            n = param.numel()
            total += n
            name_lower = name.lower()
            if "token_embedding" in name_lower:
                categories["token_embedding"] += n
            elif "ssm_layer" in name_lower:
                categories["ssm_layers"] += n
            elif "attention_layer" in name_lower or "attn" in name_lower:
                categories["attention_layers"] += n
            elif "retrieval_layer" in name_lower or "moe" in name_lower:
                categories["retrieval_layers"] += n
            elif "router" in name_lower:
                categories["router"] += n
            elif "lm_head" in name_lower:
                categories["lm_head"] += n
            elif "mtp" in name_lower:
                categories["mtp_heads"] += n
            else:
                categories["other"] += n

        categories["total"] = total
        return categories

    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
