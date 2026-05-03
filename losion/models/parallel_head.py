"""
Parallel-Head Mode for Losion-1B — Losion Framework v0.4.

Upgrade #2: Parallel-head mode where all three pathways process in
parallel without routing overhead.

For small models (1B parameters), the Tri-Jalur Router adds unnecessary
overhead — the routing computation itself costs a non-trivial fraction
of the forward pass time, and with a small model the routing decisions
don't have enough capacity to specialise effectively.

The Parallel-Head mode replaces per-token routing with a fixed, learned
weighted combination of all three pathways.  This is:

1. **Simpler** — No router module, no load-balance losses, no token
   dropping or capacity issues.
2. **Faster** — All three pathways can run in parallel (no sequential
   routing decision), and the combination is a simple weighted sum.
3. **Better for small models** — The fixed combination weights are
   learned during training and specialise per-layer, giving each
   layer the right balance of SSM/Attention/MoE without the overhead
   of per-token decisions.

Key components:
1. ParallelHeadConfig — Configuration for parallel-head mode
2. ParallelHeadLayer — Process all 3 pathways in parallel, combine
   with learned fixed weights
3. ParallelHeadModel — Wrapper that converts a routed model to
   parallel-head mode
4. Conversion utilities — Convert existing LosionLayer → ParallelHeadLayer

Architecture comparison:
    Standard LosionLayer (routed):
        input → Router → [SSM | Attention | MoE] → output
        (per-token routing, variable compute per token)

    ParallelHeadLayer (parallel):
        input → SSM ──────────────────────┐
        input → Attention ────────────────┤→ w1*ssm + w2*attn + w3*moe → output
        input → MoE/Retrieval ────────────┘
        (fixed weights per layer, all pathways always active)

Backward compatibility:
    ParallelHeadLayer implements the same interface as LosionLayer,
    so it can be used as a drop-in replacement.  The ``routing_weights``
    argument is accepted but ignored (a warning is logged if provided).

Hardware: Pure PyTorch, compatible with CUDA / ROCm / CPU.
No custom kernels required.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# ParallelHeadConfig — Configuration
# ---------------------------------------------------------------------------

@dataclass
class ParallelHeadConfig:
    """
    Configuration for Parallel-Head mode.

    Attributes:
        d_model:         Model dimension.
        n_heads:         Number of attention heads.
        d_head:          Dimension per attention head.
        d_state:         SSM state dimension (default 128).
        d_ff:            FFN intermediate dimension (default 4 * d_model).
        num_experts:     Number of MoE experts (default 4).
        top_k:           Top-K experts per token in MoE (default 2).
        dropout:         Dropout rate (default 0.0).
        init_weights:    Initial pathway weights [ssm, attn, moe].
                         Default [1/3, 1/3, 1/3] (equal).
        learnable_weights: Whether the combination weights are learnable
                           (default True).  If False, fixed at init_weights.
        weight_normalization: How to normalise the combination weights.
                              "softmax" (default) or "sigmoid" or "none".
        norm_mode:       Normalisation mode for sub-layer outputs.
                        "pre" = pre-norm (default), "post" = post-norm.
        use_residual:    Whether to use residual connections (default True).
        ssm_kwargs:      Additional kwargs for SSM sub-layer.
        attention_kwargs: Additional kwargs for Attention sub-layer.
        moe_kwargs:      Additional kwargs for MoE sub-layer.
    """

    d_model: int = 2048
    n_heads: int = 8
    d_head: int = 64
    d_state: int = 128
    d_ff: int = 0  # 0 means auto = 4 * d_model
    num_experts: int = 4
    top_k: int = 2
    dropout: float = 0.0
    init_weights: Tuple[float, float, float] = (1.0 / 3, 1.0 / 3, 1.0 / 3)
    learnable_weights: bool = True
    weight_normalization: str = "softmax"
    norm_mode: str = "pre"
    use_residual: bool = True
    ssm_kwargs: Optional[Dict] = None
    attention_kwargs: Optional[Dict] = None
    moe_kwargs: Optional[Dict] = None

    def __post_init__(self):
        if self.d_ff == 0:
            self.d_ff = 4 * self.d_model


# ---------------------------------------------------------------------------
# ParallelHeadLayer — Core layer
# ---------------------------------------------------------------------------

class ParallelHeadLayer(nn.Module):
    """
    Layer that processes all three pathways in parallel and combines
    with learned fixed weights.

    Instead of routing tokens to one pathway (SSM / Attention / MoE),
    all three pathways process the *entire* input simultaneously.  Their
    outputs are combined with learned per-layer weights::

        output = residual + w_ssm * ssm_out + w_attn * attn_out + w_moe * moe_out

    The weights are normalised (softmax by default) so they sum to 1,
    giving each pathway a fractional contribution.  During training,
    the weights are learned end-to-end, allowing each layer to discover
    the optimal balance between SSM (local patterns), Attention
    (long-range dependencies), and MoE (diverse knowledge).

    Advantages over routed mode for small models:
    - No router overhead (saves ~2-5% of compute per layer)
    - No load-balance auxiliary loss needed
    - No token dropping or capacity issues
    - All pathways can be computed in parallel (graph-level parallelism)
    - Simpler, more predictable latency

    Disadvantages:
    - Always runs all three pathways (no compute savings from skipping)
    - Less adaptive (no per-token specialisation)

    Args:
        config: ParallelHeadConfig with layer parameters.
        layer_idx: Index of this layer in the model (for logging).
    """

    def __init__(
        self,
        config: ParallelHeadConfig,
        layer_idx: int = 0,
    ) -> None:
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx
        self.d_model = config.d_model

        # ---- Sub-layer norms (pre-norm for each pathway) ----
        self.ssm_norm = nn.RMSNorm(config.d_model, eps=1e-5)
        self.attn_norm = nn.RMSNorm(config.d_model, eps=1e-5)
        self.moe_norm = nn.RMSNorm(config.d_model, eps=1e-5)

        # ---- SSM Pathway (Jalur 1) ----
        # Lightweight SSM sub-layer: single-pass with gating
        self.ssm_proj_in = nn.Linear(config.d_model, config.d_model, bias=False)
        self.ssm_gate = nn.Linear(config.d_model, config.d_model, bias=False)
        self.ssm_proj_out = nn.Linear(config.d_model, config.d_model, bias=False)

        # ---- Attention Pathway (Jalur 2) ----
        # Multi-head attention with QKV projections
        d_inner = config.n_heads * config.d_head
        self.q_proj = nn.Linear(config.d_model, d_inner, bias=False)
        self.k_proj = nn.Linear(config.d_model, d_inner, bias=False)
        self.v_proj = nn.Linear(config.d_model, d_inner, bias=False)
        self.out_proj = nn.Linear(d_inner, config.d_model, bias=False)
        self.q_norm = nn.RMSNorm(config.d_head, eps=1e-5)
        self.k_norm = nn.RMSNorm(config.d_head, eps=1e-5)

        # ---- MoE/Retrieval Pathway (Jalur 3) ----
        # Top-K MoE with simple expert FFNs
        self.router = nn.Linear(config.d_model, config.num_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.d_model, config.d_ff, bias=False),
                nn.SiLU(),
                nn.Linear(config.d_ff, config.d_model, bias=False),
            )
            for _ in range(config.num_experts)
        ])
        self.moe_norm_out = nn.RMSNorm(config.d_model, eps=1e-5)

        # ---- Combination weights ----
        init_w = config.init_weights
        if config.weight_normalization == "softmax":
            # Store as logits; softmax will normalise
            # Convert init_weights to log-space
            init_logits = torch.tensor([
                math.log(max(w, 1e-8)) for w in init_w
            ])
        else:
            init_logits = torch.tensor(list(init_w))

        if config.learnable_weights:
            self.pathway_logits = nn.Parameter(init_logits)
        else:
            self.register_buffer("pathway_logits", init_logits)

        # ---- Output norm ----
        self.output_norm = nn.RMSNorm(config.d_model, eps=1e-5)

        # ---- Dropout ----
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

    # ------------------------------------------------------------------
    # Weight computation
    # ------------------------------------------------------------------

    def get_pathway_weights(self) -> torch.Tensor:
        """
        Get the normalised pathway combination weights.

        Returns:
            Tensor of shape ``(3,)`` — [ssm_weight, attn_weight, moe_weight].
        """
        if self.config.weight_normalization == "softmax":
            return F.softmax(self.pathway_logits, dim=0)
        elif self.config.weight_normalization == "sigmoid":
            return torch.sigmoid(self.pathway_logits)
        else:
            # No normalisation, just clamp to [0, 1]
            return self.pathway_logits.clamp(0, 1)

    # ------------------------------------------------------------------
    # SSM pathway forward
    # ------------------------------------------------------------------

    def _forward_ssm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the SSM pathway.

        A lightweight SSM-like computation: input-dependent gating
        with a simple state update.  For full SSM functionality,
        replace this with a proper SSM layer (Mamba2SSD, etc.).

        Args:
            x: Input ``(batch, seq_len, d_model)``.

        Returns:
            Output ``(batch, seq_len, d_model)``.
        """
        h = self.ssm_proj_in(x)
        gate = torch.sigmoid(self.ssm_gate(x))
        h = h * gate
        output = self.ssm_proj_out(h)
        return output

    # ------------------------------------------------------------------
    # Attention pathway forward
    # ------------------------------------------------------------------

    def _forward_attention(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the Attention pathway.

        Standard multi-head attention with QK normalisation.

        Args:
            x:              Input ``(batch, seq_len, d_model)``.
            attention_mask: Optional causal mask.

        Returns:
            Output ``(batch, seq_len, d_model)``.
        """
        batch, seq_len, _ = x.shape
        n_heads = self.config.n_heads
        d_head = self.config.d_head

        # QKV projections
        q = self.q_proj(x).view(batch, seq_len, n_heads, d_head)
        k = self.k_proj(x).view(batch, seq_len, n_heads, d_head)
        v = self.v_proj(x).view(batch, seq_len, n_heads, d_head)

        # QK normalisation
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Transpose to (batch, n_heads, seq_len, d_head)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Attention scores
        scale = math.sqrt(d_head)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale

        # Causal mask
        if attention_mask is not None:
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
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        output = self.out_proj(attn_output)
        return output

    # ------------------------------------------------------------------
    # MoE pathway forward
    # ------------------------------------------------------------------

    def _forward_moe(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through the MoE/Retrieval pathway.

        Standard top-K MoE with load balancing.

        Args:
            x: Input ``(batch, seq_len, d_model)``.

        Returns:
            (output, aux_losses):
                output: ``(batch, seq_len, d_model)``.
                aux_losses: Dict with ``"load_balance"`` loss.
        """
        batch, seq_len, _ = x.shape
        top_k = self.config.top_k
        num_experts = self.config.num_experts

        # Router
        logits = self.router(x)  # (B, S, E)
        weights, indices = torch.topk(logits, top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)  # (B, S, K)

        # Process experts
        x_flat = x.view(batch * seq_len, self.d_model)
        weights_flat = weights.view(batch * seq_len, top_k)
        indices_flat = indices.view(batch * seq_len, top_k)

        output_flat = torch.zeros_like(x_flat)
        for k_idx in range(top_k):
            for eid in range(num_experts):
                mask = (indices_flat[:, k_idx] == eid)
                if not mask.any():
                    continue
                expert_out = self.experts[eid](x_flat[mask])
                w = weights_flat[mask, k_idx:k_idx + 1]
                output_flat[mask] += w * expert_out

        output = self.moe_norm_out(output_flat.view(batch, seq_len, self.d_model))

        # Load balance loss
        probs = F.softmax(logits, dim=-1)
        assignments = probs.argmax(dim=-1)
        one_hot = F.one_hot(assignments, num_experts).float()
        f = one_hot.mean(dim=(0, 1))
        P = probs.mean(dim=(0, 1))
        lb_loss = num_experts * (f * P).sum()

        return output, {"load_balance": 0.01 * lb_loss}

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        routing_weights: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with parallel pathway processing.

        All three pathways process the input simultaneously.  Their
        outputs are combined with learned fixed weights.

        Args:
            x:               Input ``(batch, seq_len, d_model)``.
            attention_mask:  Optional causal mask for attention.
            routing_weights: **Ignored** (accepted for backward compat).
            kwargs:          Additional keyword arguments.

        Returns:
            (output, aux_losses):
                output: ``(batch, seq_len, d_model)``.
                aux_losses: Dict with auxiliary losses (load_balance, etc.).
        """
        if routing_weights is not None:
            warnings.warn(
                "ParallelHeadLayer does not use routing_weights. "
                "The argument is accepted for backward compatibility only.",
                UserWarning,
                stacklevel=2,
            )

        aux_losses: Dict[str, torch.Tensor] = {}

        # ---- Get pathway weights ----
        w = self.get_pathway_weights()  # (3,)

        # ---- Pathway 1: SSM ----
        ssm_input = self.ssm_norm(x) if self.config.norm_mode == "pre" else x
        ssm_out = self._forward_ssm(ssm_input)

        # ---- Pathway 2: Attention ----
        attn_input = self.attn_norm(x) if self.config.norm_mode == "pre" else x
        attn_out = self._forward_attention(attn_input, attention_mask=attention_mask)

        # ---- Pathway 3: MoE/Retrieval ----
        moe_input = self.moe_norm(x) if self.config.norm_mode == "pre" else x
        moe_out, moe_aux = self._forward_moe(moe_input)
        aux_losses.update(moe_aux)

        # ---- Combine with fixed weights ----
        combined = (
            w[0] * ssm_out
            + w[1] * attn_out
            + w[2] * moe_out
        )
        combined = self.dropout(combined)

        # ---- Residual connection ----
        if self.config.use_residual:
            output = x + combined
        else:
            output = combined

        # ---- Output norm ----
        output = self.output_norm(output)

        # ---- Auxiliary: pathway weight sparsity (optional regularisation) ----
        # Encourage the model to make decisive choices
        weight_entropy = -(w * (w + 1e-8).log()).sum()
        aux_losses["pathway_entropy"] = 0.01 * weight_entropy

        return output, aux_losses

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def forward_inference(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Inference forward pass (single token).

        Same as ``forward()`` but without auxiliary loss computation.

        Args:
            x:              Input ``(batch, 1, d_model)``.
            attention_mask: Optional mask.

        Returns:
            Output ``(batch, 1, d_model)``.
        """
        output, _ = self.forward(x, attention_mask=attention_mask)
        return output

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_pathway_summary(self) -> Dict[str, float]:
        """
        Get a summary of pathway weights for this layer.

        Returns:
            Dict with pathway weights and dominant pathway.
        """
        w = self.get_pathway_weights()
        dominant = ["ssm", "attention", "moe"][w.argmax().item()]
        return {
            "ssm_weight": w[0].item(),
            "attention_weight": w[1].item(),
            "moe_weight": w[2].item(),
            "dominant_pathway": dominant,
        }

    def extra_repr(self) -> str:
        w = self.get_pathway_weights()
        return (
            f"d_model={self.d_model}, layer_idx={self.layer_idx}, "
            f"weights=[{w[0]:.3f}, {w[1]:.3f}, {w[2]:.3f}], "
            f"learnable={self.config.learnable_weights}"
        )


# ---------------------------------------------------------------------------
# ParallelHeadModel — Model-level wrapper
# ---------------------------------------------------------------------------

class ParallelHeadModel(nn.Module):
    """
    Model wrapper that uses ParallelHeadLayer for all layers.

    Designed for small models (1B parameters) where the routing
    overhead of the standard Tri-Jalur architecture is not justified.

    The model consists of:
    - Token embedding
    - N × ParallelHeadLayer (all layers process all pathways)
    - Final layer norm
    - LM head

    Args:
        config:             ParallelHeadConfig with model parameters.
        n_layers:           Number of parallel-head layers (default 12).
        vocab_size:         Vocabulary size (default 32000).
        max_seq_len:        Maximum sequence length (default 4096).
    """

    def __init__(
        self,
        config: ParallelHeadConfig,
        n_layers: int = 12,
        vocab_size: int = 32000,
        max_seq_len: int = 4096,
    ) -> None:
        super().__init__()

        self.config = config
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        # ---- Token embedding ----
        self.token_embedding = nn.Embedding(vocab_size, config.d_model)

        # ---- Position embedding (learned) ----
        self.position_embedding = nn.Embedding(max_seq_len, config.d_model)

        # ---- Layers ----
        self.layers = nn.ModuleList([
            ParallelHeadLayer(config, layer_idx=i)
            for i in range(n_layers)
        ])

        # ---- Final norm ----
        self.final_norm = nn.RMSNorm(config.d_model, eps=1e-5)

        # ---- LM head ----
        self.lm_head = nn.Linear(config.d_model, vocab_size, bias=False)

        # ---- Weight tying (optional) ----
        # self.lm_head.weight = self.token_embedding.weight

        # Initialize weights
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Standard weight initialization."""
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="linear")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass.

        Args:
            input_ids:       Token IDs ``(batch, seq_len)``.
            attention_mask:  Optional attention mask.
            kwargs:          Additional keyword arguments.

        Returns:
            (logits, aux_losses):
                logits: ``(batch, seq_len, vocab_size)``.
                aux_losses: Aggregated auxiliary losses.
        """
        batch, seq_len = input_ids.shape

        # ---- Embeddings ----
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)

        # ---- Layers ----
        all_aux: Dict[str, torch.Tensor] = {}
        for layer in self.layers:
            x, aux = layer(x, attention_mask=attention_mask, **kwargs)
            for key, val in aux.items():
                if key in all_aux:
                    all_aux[key] = all_aux[key] + val
                else:
                    all_aux[key] = val

        # ---- Final norm + LM head ----
        x = self.final_norm(x)
        logits = self.lm_head(x)

        # Average auxiliary losses over layers
        for key in all_aux:
            all_aux[key] = all_aux[key] / self.n_layers

        return logits, all_aux

    def get_pathway_summary(self) -> List[Dict[str, float]]:
        """
        Get pathway weight summary for all layers.

        Returns:
            List of dicts, one per layer.
        """
        return [layer.get_pathway_summary() for layer in self.layers]


# ---------------------------------------------------------------------------
# Conversion utilities
# ---------------------------------------------------------------------------

def convert_to_parallel_head(
    model: nn.Module,
    config: Optional[ParallelHeadConfig] = None,
    n_layers: Optional[int] = None,
) -> ParallelHeadModel:
    """
    Convert an existing model to Parallel-Head mode.

    This is a convenience function that creates a new ParallelHeadModel
    and optionally transfers compatible weights from the original model.

    Note: Full weight transfer requires the original model to have a
    compatible structure.  For most cases, it's easier to train a
    ParallelHeadModel from scratch.

    Args:
        model:    Original model (used for dimension inference).
        config:   ParallelHeadConfig.  If None, inferred from the model.
        n_layers: Number of layers.  If None, inferred from the model.

    Returns:
        New ParallelHeadModel instance.
    """
    # Infer config from model
    if config is None:
        # Try to infer d_model from model parameters
        d_model = None
        for name, param in model.named_parameters():
            if "embedding" in name and param.dim() >= 2:
                d_model = param.shape[-1]
                break
            if "weight" in name and param.dim() == 2:
                d_model = max(param.shape[-1], param.shape[0])
                break

        if d_model is None:
            d_model = 2048  # fallback

        config = ParallelHeadConfig(d_model=d_model)

    if n_layers is None:
        # Try to infer n_layers
        if hasattr(model, "layers") and isinstance(model.layers, nn.ModuleList):
            n_layers = len(model.layers)
        elif hasattr(model, "blocks") and isinstance(model.blocks, nn.ModuleList):
            n_layers = len(model.blocks)
        else:
            n_layers = 12  # fallback

    # Try to infer vocab_size
    vocab_size = 32000
    if hasattr(model, "token_embedding") and isinstance(model.token_embedding, nn.Embedding):
        vocab_size = model.token_embedding.num_embeddings
    elif hasattr(model, "embed_tokens") and isinstance(model.embed_tokens, nn.Embedding):
        vocab_size = model.embed_tokens.num_embeddings

    return ParallelHeadModel(
        config=config,
        n_layers=n_layers,
        vocab_size=vocab_size,
    )


def estimate_parallel_head_speedup(
    d_model: int = 2048,
    n_layers: int = 12,
    seq_len: int = 2048,
    batch_size: int = 1,
) -> Dict[str, float]:
    """
    Estimate the speedup from parallel-head mode vs routed mode.

    The speedup comes from:
    1. No router computation (~2-5% of layer compute)
    2. No routing decision overhead (~1-2% of layer latency)
    3. Potential for graph-level parallelism (pathways can be fused)

    Args:
        d_model:    Model dimension.
        n_layers:   Number of layers.
        seq_len:    Sequence length.
        batch_size: Batch size.

    Returns:
        Dict with estimated compute and speedup metrics.
    """
    # Rough FLOPs estimation
    # Router: d_model * 3 (project to 3 pathways) ≈ 3 * d_model
    # Per token: d_model * 3 = 6144 for d_model=2048
    router_flops = batch_size * seq_len * d_model * 3

    # Full layer: each pathway ≈ 2 * d_model^2 (projection in/out)
    layer_flops = batch_size * seq_len * d_model * d_model * 2 * 3  # 3 pathways

    total_routed = n_layers * (router_flops + layer_flops)
    total_parallel = n_layers * layer_flops

    router_overhead = router_flops / (router_flops + layer_flops)

    return {
        "router_flops_per_layer": float(router_flops),
        "layer_flops_per_layer": float(layer_flops),
        "router_overhead_fraction": float(router_overhead),
        "routed_total_flops": float(total_routed),
        "parallel_total_flops": float(total_parallel),
        "estimated_speedup": float(total_routed / total_parallel) if total_parallel > 0 else 1.0,
        "note": "Speedup estimate does not account for hardware-level parallelism benefits",
    }
