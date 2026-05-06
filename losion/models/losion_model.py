"""
Losion Model — Backbone implementation of the Tri-Jalur Router architecture.

Implements LosionModel (backbone), LosionLayer (single Tri-Jalur layer),
LosionLayerOutput, and RMSNorm.

The Tri-Jalur architecture routes tokens through three complementary pathways:
  - Jalur 1 (SSM): Sequential/state-space processing (Mamba-2 style)
  - Jalur 2 (Attention): Long-range dependency modeling (MLA style)
  - Jalur 3 (Retrieval): Diverse knowledge via MoE

Hardware: Pure PyTorch, compatible with CUDA / ROCm / CPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from losion.config import LosionConfig


# ============================================================================
# RMSNorm
# ============================================================================


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Simpler and faster than LayerNorm — normalizes by the RMS of the
    input without subtracting the mean.

    Args:
        dim: Normalization dimension.
        eps: Epsilon for numerical stability (default 1e-6).
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization.

        Args:
            x: Input tensor of any shape with last dimension == dim.

        Returns:
            Normalized tensor with same shape.
        """
        dtype = x.dtype
        x = x.float()
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x = x / rms
        return (self.weight * x).to(dtype)


# ============================================================================
# Simplified SSM Layer (Jalur 1)
# ============================================================================


class SimplifiedSSM(nn.Module):
    """Simplified SSM pathway for the Tri-Jalur layer.

    Uses input-dependent gating with a causal convolution and simple
    state-based processing. For production, this would be replaced
    with Mamba-2 SSD / RWKV-7 / DeltaNet.

    Args:
        d_model: Model dimension.
        d_state: SSM state dimension.
        expand: Expansion factor.
        d_conv: Local convolution width.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        expand: int = 2,
        d_conv: int = 4,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = d_model * expand
        self.d_conv = d_conv

        self.proj_in = nn.Linear(d_model, self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )
        self.proj_out = nn.Linear(self.d_inner, d_model, bias=False)
        self.gate = nn.Linear(d_model, self.d_inner, bias=False)

        # SSM projection for state-dependent computation
        self.state_proj = nn.Linear(d_model, self.d_inner, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through simplified SSM.

        Args:
            x: Input tensor (batch, seq_len, d_model).

        Returns:
            Output tensor (batch, seq_len, d_model).
        """
        batch, seq_len, _ = x.shape

        # Input projection + gating
        z = self.proj_in(x)
        gate = torch.sigmoid(self.gate(x))
        z = z * gate

        # Causal convolution
        z_conv = z.transpose(1, 2)  # (batch, d_inner, seq_len)
        z_conv = self.conv1d(z_conv)[:, :, :seq_len]  # Trim padding
        z = z_conv.transpose(1, 2)  # (batch, seq_len, d_inner)

        # State-dependent modulation (simplified SSM-like behavior)
        s = self.state_proj(x)
        s = torch.sigmoid(s)
        z = z * s

        # Output projection
        output = self.proj_out(z)
        return output


# ============================================================================
# Simplified Attention Layer (Jalur 2)
# ============================================================================


class SimplifiedAttention(nn.Module):
    """Simplified MLA-style attention for the Tri-Jalur layer.

    Uses multi-head attention with optional KV compression.
    For production, this would use full MLA or Lightning Attention.

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        d_kv: Dimension per key/value head.
        mla_latent_dim: MLA latent compression dimension.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        d_kv: int = 64,
        mla_latent_dim: int = 128,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_kv = d_kv
        self.mla_latent_dim = mla_latent_dim
        self.d_inner = n_heads * d_kv

        # Q projection
        self.q_proj = nn.Linear(d_model, self.d_inner, bias=False)

        # KV compression (MLA-style)
        self.kv_down = nn.Linear(d_model, mla_latent_dim, bias=False)
        self.kv_norm = RMSNorm(mla_latent_dim)
        self.k_up = nn.Linear(mla_latent_dim, self.d_inner, bias=False)
        self.v_up = nn.Linear(mla_latent_dim, self.d_inner, bias=False)

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # QK normalization
        self.q_norm = RMSNorm(d_kv)
        self.k_norm = RMSNorm(d_kv)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through simplified attention.

        Args:
            x: Input tensor (batch, seq_len, d_model).
            attention_mask: Optional attention mask (unused, causal by default).

        Returns:
            Output tensor (batch, seq_len, d_model).
        """
        batch, seq_len, _ = x.shape

        # Q projection
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.d_kv)

        # KV compression
        c_kv = self.kv_norm(self.kv_down(x))  # (batch, seq_len, mla_latent_dim)
        k = self.k_up(c_kv).view(batch, seq_len, self.n_heads, self.d_kv)
        v = self.v_up(c_kv).view(batch, seq_len, self.n_heads, self.d_kv)

        # QK normalization
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Transpose to (batch, n_heads, seq_len, d_kv)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Attention scores
        scale = math.sqrt(self.d_kv)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale

        # Causal mask
        if attention_mask is not None:
            # Validate attention_mask format — must be additive bias
            # (0.0 = attend, -inf = ignore), NOT boolean or HuggingFace-style.
            if attention_mask.dtype == torch.bool:
                raise TypeError(
                    "attention_mask must be an additive bias tensor (float), "
                    "not a boolean mask. Convert with: "
                    "mask = torch.where(bool_mask, 0.0, float('-inf'))"
                )
            attn_weights = attn_weights + attention_mask
        else:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device),
                diagonal=1,
            )
            attn_weights = attn_weights.masked_fill(
                causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
            )

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(x.dtype)
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, self.d_inner)
        output = self.out_proj(attn_output)
        return output


# ============================================================================
# Simplified MoE / Retrieval Layer (Jalur 3)
# ============================================================================


class ExpertFFN(nn.Module):
    """Single expert feed-forward network (SwiGLU style).

    Args:
        d_model: Model dimension.
        d_ff: Feed-forward intermediate dimension.
    """

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class SimplifiedMoE(nn.Module):
    """Simplified Top-K MoE with bias-based routing.

    For production, this would use the full MoERetrieval with Engram,
    heterogeneous experts, etc.

    Args:
        d_model: Model dimension.
        d_ff: Feed-forward intermediate dimension.
        num_experts: Number of experts.
        num_active_experts: Number of active experts per token.
        top_k_routing: Top-K routing (clamped to num_experts).
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int = 16,
        num_active_experts: int = 2,
        top_k_routing: int = 2,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.num_active_experts = num_active_experts
        self.top_k_routing = min(top_k_routing, num_experts)

        # Router
        self.router = nn.Linear(d_model, num_experts, bias=False)

        # Experts
        self.experts = nn.ModuleList([
            ExpertFFN(d_model, d_ff) for _ in range(num_experts)
        ])

        # Shared expert (optional)
        self.shared_expert = ExpertFFN(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass through Top-K MoE.

        Uses vectorized batched expert computation instead of O(K×E) Python loop.
        Each token is processed by its top-K selected experts in a single
        vectorized pass per K-slot.

        Args:
            x: Input tensor (batch, seq_len, d_model).

        Returns:
            Tuple (output, aux_info) where aux_info contains routing details.
        """
        batch, seq_len, _ = x.shape
        x_flat = x.view(batch * seq_len, self.d_model)

        # Router logits
        router_logits = self.router(x_flat)  # (B*S, num_experts)
        weights, indices = torch.topk(router_logits, self.top_k_routing, dim=-1)
        weights = F.softmax(weights, dim=-1)  # (B*S, top_k)

        # Vectorized MoE computation: iterate only over top_k slots (K),
        # not over all experts (E). For each slot, batch tokens by their
        # assigned expert for efficient computation.
        output_flat = torch.zeros_like(x_flat)
        for k_idx in range(self.top_k_routing):
            expert_idx = indices[:, k_idx]  # (B*S,) — expert assignment per token
            w = weights[:, k_idx:k_idx + 1]  # (B*S, 1) — weight for this slot

            # Process each expert that has at least one token assigned
            unique_experts = expert_idx.unique()
            for eid in unique_experts:
                eid_val = eid.item()
                mask = (expert_idx == eid)  # (B*S,) — which tokens go to this expert
                if not mask.any():
                    continue
                expert_out = self.experts[eid_val](x_flat[mask])  # (n_tokens, d_model)
                output_flat[mask] += w[mask] * expert_out

        # Shared expert
        shared_out = self.shared_expert(x_flat)
        output_flat = output_flat + shared_out

        output = output_flat.view(batch, seq_len, self.d_model)

        # Aux info
        # Convention note: softmax is applied ONLY to top-k logits (not all-E
        # logits then top-k). This means expert_weights sum to 1 within the
        # top-k set but do NOT reflect probability relative to the full expert
        # pool. If downstream code (e.g., load balancing loss) needs the full
        # distribution, use router_logits instead. The field is named
        # normalized_topk_weights (not expert_weights) to make this explicit.
        aux = {
            "router_logits": router_logits.view(batch, seq_len, self.num_experts),
            "expert_indices": indices.view(batch, seq_len, self.top_k_routing),
            "normalized_topk_weights": weights.view(batch, seq_len, self.top_k_routing),
        }

        return output, aux


class SimplifiedEngram(nn.Module):
    """Simplified Engram Memory for the Retrieval pathway.

    Uses a small embedding table for fact retrieval simulation.
    For production, this would use the full EngramMemory with hash tables.

    Args:
        d_model: Model dimension.
        num_buckets: Number of hash buckets.
        embedding_dim: Embedding dimension.
    """

    def __init__(
        self,
        d_model: int,
        num_buckets: int = 1_000_000,
        embedding_dim: int = 256,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_buckets = num_buckets
        self.embedding_dim = embedding_dim

        # Simplified: use a small learnable embedding instead of full hash table
        self.query_proj = nn.Linear(d_model, embedding_dim, bias=False)
        self.output_proj = nn.Linear(embedding_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass (identity-like with projection for compatibility)."""
        q = self.query_proj(x)
        return self.output_proj(q)


class SimplifiedRetrieval(nn.Module):
    """Simplified Retrieval layer combining MoE and Engram.

    Args:
        d_model: Model dimension.
        d_ff: Feed-forward intermediate dimension.
        num_experts: Number of MoE experts.
        num_active_experts: Active experts per token.
        top_k_routing: Top-K routing.
        use_engram: Whether to use engram memory.
        engram_dim: Engram embedding dimension.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int = 16,
        num_active_experts: int = 2,
        top_k_routing: int = 2,
        use_engram: bool = True,
        engram_dim: int = 128,
    ) -> None:
        super().__init__()
        self.moe = SimplifiedMoE(
            d_model=d_model,
            d_ff=d_ff,
            num_experts=num_experts,
            num_active_experts=num_active_experts,
            top_k_routing=top_k_routing,
        )
        self.use_engram = use_engram
        if use_engram:
            num_buckets = min(1_000_000, d_model * 1000)
            self.engram = SimplifiedEngram(
                d_model=d_model,
                num_buckets=num_buckets,
                embedding_dim=engram_dim,
            )
        else:
            self.engram = None  # type: ignore

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass through retrieval layer.

        Args:
            x: Input tensor (batch, seq_len, d_model).

        Returns:
            Tuple (output, aux_info).
        """
        moe_out, moe_aux = self.moe(x)

        if self.use_engram and self.engram is not None:
            engram_out = self.engram(x)
            output = moe_out + 0.1 * engram_out  # Small engram contribution
        else:
            output = moe_out

        return output, moe_aux


# ============================================================================
# LosionLayerOutput
# ============================================================================


@dataclass
class LosionLayerOutput:
    """Output from a single LosionLayer.

    Attributes:
        hidden_states: Hidden state tensor (batch, seq_len, d_model).
        routing_info: Optional routing information for this layer.
        all_hidden_states: Optional list of all hidden states (when requested).
    """
    hidden_states: torch.Tensor
    routing_info: Optional[Any] = None
    all_hidden_states: Optional[List[torch.Tensor]] = None


# ============================================================================
# LosionLayer — Single Tri-Jalur Layer
# ============================================================================


class LosionLayer(nn.Module):
    """Single Tri-Jalur Router layer.

    Routes tokens through three pathways:
    - Jalur 1 (SSM): Sequential/state-space processing
    - Jalur 2 (Attention): Long-range dependency modeling
    - Jalur 3 (Retrieval): Diverse knowledge via MoE

    The router determines pathway weights for each token.

    Args:
        config: LosionConfig with model parameters.
        layer_idx: Index of this layer in the model.
    """

    def __init__(self, config: LosionConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.d_model = config.d_model

        # Auto-scale number of experts if num_experts == 0
        num_experts = config.retrieval.num_experts
        if num_experts == 0:
            num_experts = max(8, min(64, config.d_model // 32))

        # Auto-scale top_k_routing
        top_k_routing = config.retrieval.top_k_routing
        top_k_routing = min(top_k_routing, num_experts)
        num_active_experts = min(config.retrieval.num_active_experts, num_experts)

        # Auto-scale d_ff
        d_ff = config.retrieval.d_ff if config.retrieval.d_ff > 0 else 4 * config.d_model

        # Auto-scale engram buckets
        engram_buckets = min(1_000_000, config.d_model * 1000)

        # ---- Pre-norms for each pathway ----
        self.ssm_norm = RMSNorm(config.d_model)
        self.attn_norm = RMSNorm(config.d_model)
        self.retrieval_norm = RMSNorm(config.d_model)

        # ---- Jalur 1: SSM ----
        self.ssm_layer = SimplifiedSSM(
            d_model=config.d_model,
            d_state=config.ssm.d_state,
            expand=config.ssm.expand,
            d_conv=config.ssm.d_conv,
        )

        # ---- Jalur 2: Attention ----
        self.attention_layer = SimplifiedAttention(
            d_model=config.d_model,
            n_heads=config.attention.n_heads,
            d_kv=config.attention.d_kv,
            mla_latent_dim=config.attention.mla_latent_dim,
        )

        # ---- Jalur 3: Retrieval (MoE + Engram) ----
        self.retrieval_layer = SimplifiedRetrieval(
            d_model=config.d_model,
            d_ff=d_ff,
            num_experts=num_experts,
            num_active_experts=num_active_experts,
            top_k_routing=top_k_routing,
            use_engram=config.retrieval.use_engram,
            engram_dim=config.retrieval.engram_dim,
        )

        # ---- Router (simple linear router) ----
        self.router = nn.Linear(config.d_model, 3, bias=False)

        # ---- Output norm ----
        self.output_norm = RMSNorm(config.d_model)

        # ---- Dropout ----
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        routing_weights: Optional[torch.Tensor] = None,
        thinking_mode: Optional[bool] = None,
        inference_sparse: bool = False,
        sparse_threshold: float = 0.05,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """Forward pass through the Tri-Jalur layer.

        Args:
            x: Input tensor (batch, seq_len, d_model).
            attention_mask: Optional attention mask. Must be an additive bias
                tensor of shape [batch, heads, seq_len, seq_len] where 0.0
                means "attend" and -inf means "ignore". Boolean masks and
                HuggingFace-style (1=attend, 0=ignore) masks are NOT supported.
            routing_weights: Optional pre-computed routing weights.
            thinking_mode: If True, bias towards attention + retrieval pathways.
            inference_sparse: If True, skip computation for pathways whose
                routing weight is below ``sparse_threshold``. Only effective
                during inference (not training), where gradients don't need
                to flow through all pathways. This can reduce compute by up
                to 3x when a pathway is completely deactivated.
            sparse_threshold: Minimum routing weight to compute a pathway.
                Default 0.05 (5%). Only used when inference_sparse=True.

        Returns:
            Tuple (output, routing_info).
        """
        batch, seq_len, _ = x.shape

        # ---- Compute routing weights ----
        if routing_weights is None:
            route_logits = self.router(x)  # (batch, seq_len, 3)

            # Adjust for thinking mode — apply boost to LOGITS before softmax,
            # NOT to probabilities after softmax (double softmax melemahkan boost).
            # See AdaptiveRouter._adjust_for_thinking() for the same fix.
            if thinking_mode is True:
                # Boost attention + retrieval, reduce SSM
                boost = torch.tensor([-0.2, 0.1, 0.1], device=x.device, dtype=x.dtype)
                route_logits = route_logits + boost.unsqueeze(0).unsqueeze(0)

            route_weights = F.softmax(route_logits, dim=-1)
        else:
            route_weights = routing_weights

        # ---- Determine which pathways to compute ----
        # During training, always compute all 3 pathways (gradient flows to all).
        # During inference with inference_sparse=True, skip pathways only if
        # ALL tokens in the batch have weight below threshold. We use max()
        # instead of mean() because mean can be misleading: if 5% of tokens
        # have w_ssm=0.8 but 95% have w_ssm≈0, mean≈0.04 < threshold but
        # those 5% tokens still NEED the SSM pathway. Using max ensures we
        # never silently skip a pathway that any token depends on.
        w_ssm_max = route_weights[:, :, 0].max().item()
        w_attn_max = route_weights[:, :, 1].max().item()
        w_ret_max = route_weights[:, :, 2].max().item()

        compute_ssm = not (inference_sparse and not self.training and w_ssm_max < sparse_threshold)
        compute_attn = not (inference_sparse and not self.training and w_attn_max < sparse_threshold)
        compute_ret = not (inference_sparse and not self.training and w_ret_max < sparse_threshold)

        # ---- Jalur 1: SSM ----
        if compute_ssm:
            ssm_input = self.ssm_norm(x)
            ssm_out = self.ssm_layer(ssm_input)
        else:
            ssm_out = torch.zeros_like(x)

        # ---- Jalur 2: Attention ----
        if compute_attn:
            attn_input = self.attn_norm(x)
            attn_out = self.attention_layer(attn_input, attention_mask=attention_mask)
        else:
            attn_out = torch.zeros_like(x)

        # ---- Jalur 3: Retrieval ----
        if compute_ret:
            ret_input = self.retrieval_norm(x)
            ret_out, ret_aux = self.retrieval_layer(ret_input)
        else:
            ret_out = torch.zeros_like(x)
            ret_aux = None

        # ---- Combine with routing weights ----
        w_ssm = route_weights[:, :, 0:1]    # (batch, seq_len, 1)
        w_attn = route_weights[:, :, 1:2]
        w_ret = route_weights[:, :, 2:3]

        combined = w_ssm * ssm_out + w_attn * attn_out + w_ret * ret_out

        # ---- Residual + norm ----
        output = x + self.dropout(combined)
        output = self.output_norm(output)

        # ---- Routing info ----
        routing_info = {
            "layer_idx": self.layer_idx,
            "route_weights": route_weights.detach(),
            "retrieval_aux": ret_aux,
        }

        return output, routing_info


# ============================================================================
# LosionModel — Backbone
# ============================================================================


class LosionModel(nn.Module):
    """Losion backbone model with Tri-Jalur Router architecture.

    Consists of:
    - Token embedding
    - N × LosionLayer (Tri-Jalur routed layers)
    - Final RMS normalization

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

        # ---- Token embedding ----
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        # ---- Position embedding (learned, no weight tying) ----
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)

        # ---- Embedding dropout ----
        self.embed_dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

        # ---- Layers ----
        self.layers = nn.ModuleList([
            LosionLayer(config, layer_idx=i)
            for i in range(config.n_layers)
        ])

        # ---- Final norm ----
        self.final_norm = RMSNorm(config.d_model)

        # ---- Gradient checkpointing ----
        self.gradient_checkpointing: bool = False

        # ---- Initialize weights ----
        self.apply(lambda m: self._init_weights(m))

    def _init_weights(self, module: nn.Module) -> None:
        """LLM-standard weight initialization.

        Uses GPT-2 / GPT-NeoX style initialization:
        - Embeddings: normal(0, 0.02)
        - Linear layers: normal(0, 0.02 / sqrt(2 * n_layers))
          The sqrt(2 * n_layers) scaling prevents hidden state explosion
          in deep residual networks (GPT-2 paper Section 2.3).
        - Conv1d: normal(0, 0.02 / sqrt(2 * n_layers))
          Consistent with Linear init (not PyTorch default kaiming).
        - Biases: zeros
        """
        if isinstance(module, nn.Linear):
            # Scaled init for residual stream stability
            std = 0.02 / math.sqrt(2 * self.n_layers)
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            # Same scaled init as Linear — Conv1d is used in SimplifiedSSM
            # for causal convolution, and should follow the same residual
            # stream scaling as the rest of the model, not PyTorch's default
            # kaiming_uniform_.
            std = 0.02 / math.sqrt(2 * self.n_layers)
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_input_embeddings(self) -> nn.Embedding:
        """Get the input token embedding layer."""
        return self.token_embedding

    def set_input_embeddings(self, embeddings: nn.Embedding) -> None:
        """Set the input token embedding layer."""
        self.token_embedding = embeddings

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing for memory efficiency."""
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self) -> None:
        """Disable gradient checkpointing."""
        self.gradient_checkpointing = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        thinking_mode: Optional[bool] = None,
        return_routing_info: bool = False,
        return_all_hidden_states: bool = False,
        inference_sparse: bool = False,
        sparse_threshold: float = 0.05,
    ) -> LosionLayerOutput:
        """Forward pass through the Losion backbone.

        Args:
            input_ids: Token IDs (batch, seq_len).
            attention_mask: Optional attention mask.
            thinking_mode: If True, bias towards thinking pathways.
            return_routing_info: If True, return routing info per layer.
            return_all_hidden_states: If True, return all intermediate hidden states.
            inference_sparse: If True, skip computation for pathways whose
                routing weight is below ``sparse_threshold`` during inference.
                Only effective when not training. Can reduce compute by up
                to 3x when a pathway is completely deactivated.
            sparse_threshold: Minimum routing weight to compute a pathway.
                Default 0.05 (5%). Only used when inference_sparse=True.

        Returns:
            LosionLayerOutput with hidden_states and optional routing/states info.
        """
        batch, seq_len = input_ids.shape

        # ---- Validate sequence length ----
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"seq_len={seq_len} melebihi max_seq_len={self.max_seq_len}. "
                f"Gunakan context extension atau truncate input."
            )

        # ---- Embeddings ----
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.embed_dropout(x)

        # ---- Layer processing ----
        all_routing_info: Optional[List[Any]] = [] if return_routing_info else None
        all_hidden_states: Optional[List[torch.Tensor]] = [] if return_all_hidden_states else None

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                # Gradient checkpointing: recompute activations in backward pass.
                # Fix (I-07): Compute routing info OUTSIDE the checkpoint so it
                # is not lost. torch.utils.checkpoint.checkpoint can only return
                # tensors — dicts/tuples with non-tensor values are dropped.
                # Solution: checkpoint only the heavy computation (pathway forward
                # + combine), and compute routing info separately with no_grad.

                # Step 1: Compute routing weights (cheap — single linear + softmax)
                with torch.no_grad():
                    route_logits = layer.router(x)
                    if thinking_mode is True:
                        boost = torch.tensor(
                            [-0.2, 0.1, 0.1], device=x.device, dtype=x.dtype
                        )
                        route_logits = route_logits + boost.unsqueeze(0).unsqueeze(0)
                    route_weights = F.softmax(route_logits, dim=-1)

                # Step 2: Checkpoint the heavy computation only
                def _checkpoint_compute(layer_module, attn_mask, r_weights):
                    def _forward(*args):
                        hidden = args[0]
                        # Pass pre-computed routing weights to avoid re-computation
                        out, _ = layer_module(
                            hidden,
                            attention_mask=attn_mask,
                            routing_weights=r_weights,
                        )
                        return out
                    return _forward

                x = torch.utils.checkpoint.checkpoint(
                    _checkpoint_compute(layer, attention_mask, route_weights),
                    x,
                    use_reentrant=False,
                )

                # Step 3: Build routing info from pre-computed weights
                layer_routing = {
                    "layer_idx": layer.layer_idx,
                    "route_weights": route_weights.detach(),
                    "retrieval_aux": None,  # Not available under checkpointing
                }
            else:
                x, layer_routing = layer(
                    x,
                    attention_mask=attention_mask,
                    thinking_mode=thinking_mode,
                    inference_sparse=inference_sparse,
                    sparse_threshold=sparse_threshold,
                )

            if all_routing_info is not None:
                all_routing_info.append(layer_routing)

            if all_hidden_states is not None:
                all_hidden_states.append(x.detach())

        # ---- Final norm ----
        x = self.final_norm(x)

        return LosionLayerOutput(
            hidden_states=x,
            routing_info=all_routing_info,
            all_hidden_states=all_hidden_states,
        )

    def count_parameters(self) -> Dict[str, int]:
        """Count parameters by category.

        Returns:
            Dictionary with parameter counts by category.
        """
        total = 0
        token_embedding = 0
        ssm_layers = 0
        attention_layers = 0
        retrieval_layers = 0

        for name, param in self.named_parameters():
            n = param.numel()
            total += n

            name_lower = name.lower()
            if "token_embedding" in name_lower or "position_embedding" in name_lower:
                token_embedding += n
            elif "ssm_layer" in name_lower:
                ssm_layers += n
            elif "attention_layer" in name_lower or "attn" in name_lower:
                attention_layers += n
            elif "retrieval_layer" in name_lower or "moe" in name_lower or "engram" in name_lower:
                retrieval_layers += n

        return {
            "total": total,
            "token_embedding": token_embedding,
            "ssm_layers": ssm_layers,
            "attention_layers": attention_layers,
            "retrieval_layers": retrieval_layers,
        }
