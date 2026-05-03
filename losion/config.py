"""
Losion Configuration — Unified configuration for the Losion Framework.

Provides LosionConfig and all sub-configurations for the Tri-Jalur Router
architecture (SSM + Attention + MoE).

Usage:
    >>> config = LosionConfig(d_model=768, n_layers=12, vocab_size=32000)
    >>> config = LosionConfig.from_yaml("configs/losion-1b.yaml")
    >>> est = config.estimated_parameters()
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False


# ============================================================================
# Enums
# ============================================================================


class RoutingType(enum.Enum):
    """Routing strategy for the Tri-Jalur Router."""
    ADAPTIVE = "adaptive"
    FIXED = "fixed"
    RANDOM = "random"
    LEARNED = "learned"


class ThinkingMode(enum.Enum):
    """Thinking mode for the model.

    Used in config-level settings (AttentionConfig.thinking_mode).
    Different from losion.core.router.ThinkingMode which controls
    the router's internal thinking toggle.
    """
    TRIGGERED = "triggered"
    ALWAYS = "always"
    NEVER = "never"
    AUTO = "auto"


class PrecisionType(enum.Enum):
    """Precision type for hardware configuration."""
    FP32 = "fp32"
    BF16 = "bf16"
    FP16 = "fp16"
    FP8 = "fp8"


# ============================================================================
# Sub-Configurations
# ============================================================================


@dataclass
class SSMConfig:
    """Configuration for SSM pathway (Jalur 1).

    Attributes:
        d_state: SSM state dimension.
        d_conv: Local convolution width.
        expand: Expansion factor for SSM inner dimension.
        ssd_chunk_size: Chunk size for SSD (State Space Duality) parallel scan.
        use_wkv: Whether to use RWKV-7 WKV kernel.
        use_delta_net: Whether to use Gated DeltaNet.
        interleaving_ratios: Interleaving ratios for SSM variants [Mamba2, RWKV, DeltaNet].
        use_liquid: Whether to use Liquid SSM (adaptive compute depth, v0.4).
        complexity_bottleneck: Bottleneck dimension for Liquid SSM complexity estimation.
        depth_entropy_weight: Weight for depth entropy regularization in Liquid SSM.
        use_mamba3: Whether to use Mamba-3 SSD (half state, inference-first, v0.6).
        use_routing_mamba: Whether to use Routing Mamba MoE over SSM projections (v0.6).
        routing_mamba_num_experts: Number of SSM projection experts for Routing Mamba.
        routing_mamba_active_experts: Number of active SSM experts per token.
    """
    d_state: int = 64
    d_conv: int = 4
    expand: int = 2
    ssd_chunk_size: int = 256
    use_wkv: bool = False
    use_delta_net: bool = False
    interleaving_ratios: List[int] = field(default_factory=lambda: [4, 1, 1])
    use_liquid: bool = False
    complexity_bottleneck: int = 64
    depth_entropy_weight: float = 0.01
    # v0.6 additions
    use_mamba3: bool = False
    use_routing_mamba: bool = False
    routing_mamba_num_experts: int = 4
    routing_mamba_active_experts: int = 2
    # v0.8 additions — Structured Sparse Transition (NeurIPS '25)
    use_structured_sparse: bool = False
    structured_sparse_n_groups: int = 4


@dataclass
class AttentionConfig:
    """Configuration for Attention pathway (Jalur 2).

    Attributes:
        n_heads: Number of attention heads.
        d_kv: Dimension per key/value head.
        mla_latent_dim: Latent dimension for MLA KV compression.
        use_irope: Whether to use Interleaved RoPE.
        irope_ratio: Ratio of RoPE dimensions.
        base_interleaving_ratio: Base interleaving ratio for attention.
        thinking_interleaving_ratio: Interleaving ratio when in thinking mode.
        thinking_mode: Thinking mode configuration.
        use_lightning: Whether to use Lightning Attention (v0.4).
        lightning_window_size: Window size for Lightning Attention local window.
        lightning_chunk_size: Chunk size for Lightning Attention parallel training.
        lightning_feature_map: Feature map for linear attention ("elu", "relu", "cos").
        use_shared_attention: Whether to use Shared Attention (Zamba2-style, v0.4).
        shared_n_groups: Number of shared attention groups.
        shared_pattern: Pattern for shared attention ("all_shared", "interleaved").
        shared_unique_ratio: Ratio of unique (non-shared) parameters per layer.
        use_gated_attention: Whether to use Gated Attention (Qwen, NeurIPS '25 Best Paper, v0.6).
        use_moba: Whether to use MoBA Mixture of Block Attention (Moonshot AI, NeurIPS '25, v0.6).
        moba_block_size: Block size for MoBA.
        moba_top_k_blocks: Number of top-K blocks for MoBA routing.
    """
    n_heads: int = 8
    d_kv: int = 64
    mla_latent_dim: int = 128
    use_irope: bool = True
    irope_ratio: float = 3.0
    base_interleaving_ratio: int = 5
    thinking_interleaving_ratio: int = 2
    thinking_mode: ThinkingMode = ThinkingMode.AUTO
    use_lightning: bool = False
    lightning_window_size: int = 2048
    lightning_chunk_size: int = 4096
    lightning_feature_map: str = "elu"
    use_shared_attention: bool = False
    shared_n_groups: int = 1
    shared_pattern: str = "all_shared"
    shared_unique_ratio: float = 0.25
    # v0.6 additions
    use_gated_attention: bool = False
    use_moba: bool = False
    moba_block_size: int = 512
    moba_top_k_blocks: int = 4
    # v0.8 additions
    use_cross_jalur_routing: bool = False
    cross_jalur_blend_alpha: float = 0.3
    cross_jalur_graph_top_k: int = 8


@dataclass
class RetrievalConfig:
    """Configuration for Retrieval/MoE pathway (Jalur 3).

    Attributes:
        num_experts: Number of MoE experts (0 = auto-scale).
        num_active_experts: Number of active experts per token.
        d_ff: Feed-forward intermediate dimension.
        use_engram: Whether to use Engram Memory.
        engram_dim: Dimension of engram embeddings.
        use_shared_expert: Whether to use a shared expert.
        top_k_routing: Top-K experts per token for routing.
        use_heterogeneous: Whether to use Heterogeneous MoE (v0.4).
        heterogeneous_min_dim: Minimum expert dimension for heterogeneous MoE.
        heterogeneous_max_dim: Maximum expert dimension for heterogeneous MoE.
        use_matryoshka: Whether to use Matryoshka MoE (v0.4).
        matryoshka_min_experts: Minimum experts for Matryoshka MoE.
        matryoshka_max_experts: Maximum experts for Matryoshka MoE.
        use_gradient_routed: Whether to use Gradient-routed MoE (v0.4).
        gradient_routed_lr: Learning rate for gradient-routed MoE.
        use_asymmetric: Whether to use Asymmetric MoE placement (v0.4).
        asymmetric_moe_layers: Layer indices with MoE (for asymmetric placement).
        use_smore: Whether to use S'MoRE (Sub-tree MoE with Residual Experts, Meta NeurIPS '25, v0.6).
        smore_num_sub_trees: Number of shared sub-trees for S'MoRE.
        smore_sub_tree_depth: Depth of each sub-tree for S'MoRE.
        use_symbolic_moe: Whether to use Symbolic-MoE (skill-based discrete routing, v0.6).
    """
    num_experts: int = 16
    num_active_experts: int = 2
    d_ff: int = 0  # 0 means auto = 4 * d_model
    use_engram: bool = True
    engram_dim: int = 128
    use_shared_expert: bool = True
    top_k_routing: int = 2
    use_heterogeneous: bool = False
    heterogeneous_min_dim: int = 1024
    heterogeneous_max_dim: int = 8192
    use_matryoshka: bool = False
    matryoshka_min_experts: int = 2
    matryoshka_max_experts: int = 4
    use_gradient_routed: bool = False
    gradient_routed_lr: float = 0.01
    use_asymmetric: bool = False
    asymmetric_moe_layers: List[int] = field(default_factory=list)
    # v0.6 additions
    use_smore: bool = False
    smore_num_sub_trees: int = 4
    smore_sub_tree_depth: int = 2
    use_symbolic_moe: bool = False
    # v0.8 additions — Infinite MoE & Cross-Jalur Routing
    use_infinite_moe: bool = False
    infinite_moe_code_dim: int = 32
    infinite_moe_hypernet_hidden: int = 256
    infinite_moe_low_rank_residual: bool = True
    infinite_moe_codebook_size: int = 256

    def __post_init__(self) -> None:
        if self.num_experts > 0 and self.top_k_routing < self.num_active_experts:
            raise ValueError(
                f"top_k_routing ({self.top_k_routing}) must be >= "
                f"num_active_experts ({self.num_active_experts})"
            )


@dataclass
class RouterConfig:
    """Configuration for the Tri-Jalur Router.

    Attributes:
        routing_type: Type of routing strategy.
        use_thinking_toggle: Whether to use ThinkingToggle (Qwen3-style).
        bias_lr: Learning rate for bias update in BiasRouter.
        aux_loss_weight: Weight for auxiliary load-balancing loss (0.0 = aux-loss-free).
        top_k_pathways: Number of active pathways per token.
    """
    routing_type: RoutingType = RoutingType.ADAPTIVE
    use_thinking_toggle: bool = True
    bias_lr: float = 0.01
    aux_loss_weight: float = 0.0
    top_k_pathways: int = 2


@dataclass
class AttnResConfig:
    """Configuration for Attention Residuals (MoonshotAI 2026, v0.9).

    AttnRes replaces fixed-weight residual connections with learned
    attention-based aggregation across layers.

    Attributes:
        enabled: Whether to enable AttnRes.
        mode: "full", "block", or "hybrid".
        num_blocks: Number of blocks for Block AttnRes mode.
        dropout: Dropout rate for attention weights.
        use_gate: Whether to use gating after aggregation.
        temperature: Temperature for attention softmax.
        compression_dim: Compress layer representations (0 = no compression).
        use_token_compression: Enable token-dimension AttnRes + Compression.
        token_compression_type: "linear", "gated", or "ssm".
        token_compression_d_state: Dimension of compressed token state.
    """
    enabled: bool = False
    mode: str = "block"
    num_blocks: int = 8
    dropout: float = 0.0
    use_gate: bool = True
    temperature: float = 1.0
    compression_dim: int = 0
    use_token_compression: bool = False
    token_compression_type: str = "gated"
    token_compression_d_state: int = 256


@dataclass
class EvoformerConfig:
    """Configuration for Evoformer feedback loops (v0.9).

    5 levels of bidirectional feedback inspired by AlphaFold's Evoformer.

    Attributes:
        enabled: Whether to enable Evoformer feedback.
        n_recycling_steps: Number of recycling iterations.
        use_layer_recycling: Enable Level 1 — inter-layer feedback.
        use_token_recycling: Enable Level 2 — bidirectional token update.
        use_decoder_feedback: Enable Level 3 — decoder ↔ predict feedback.
        use_prediction_recycling: Enable Level 4 — prediction → context.
        use_router_coevolve: Enable Level 5 — router ↔ expert co-evolution.
    """
    enabled: bool = False
    n_recycling_steps: int = 3
    use_layer_recycling: bool = True
    use_token_recycling: bool = True
    use_decoder_feedback: bool = True
    use_prediction_recycling: bool = True
    use_router_coevolve: bool = True


@dataclass
class Child3WConfig:
    """Configuration for Child-3W attention routing (v0.9).

    MoE at the QKV level: multiple child attention parameter sets
    with routing between them.

    Attributes:
        enabled: Whether to enable Child-3W routing.
        num_children: Number of Child-3W sets.
        top_k_children: Active children per token.
        use_mla: Whether to use MLA compression.
        mla_latent_dim: MLA latent dimension.
        load_balance_weight: Auxiliary load balancing loss weight.
    """
    enabled: bool = False
    num_children: int = 4
    top_k_children: int = 2
    use_mla: bool = False
    mla_latent_dim: int = 0
    load_balance_weight: float = 0.01


@dataclass
class AnchoredDecoderConfig:
    """Configuration for Anchored Diffusion Decoder (v0.9).

    Continuous vector prediction + lightweight anchored diffusion refinement.
    Replaces softmax → token ID with predict → continuous vector → 2-3 step decode.

    Attributes:
        enabled: Whether to enable anchored decoder.
        n_refine_steps: Number of refinement steps.
        d_refine: Internal dimension for refinement.
        use_evoformer_feedback: Whether to use Evoformer feedback loop.
        n_feedback_iterations: Number of feedback iterations.
        disambiguation_heads: Heads for disambiguation attention.
    """
    enabled: bool = False
    n_refine_steps: int = 3
    d_refine: int = 512
    use_evoformer_feedback: bool = True
    n_feedback_iterations: int = 2
    disambiguation_heads: int = 8


@dataclass
class SlidingWindowConfig:
    """Configuration for Sliding Window Attention (v0.10, RATTENTION-inspired).

    Limits KV cache to a fixed window size, reducing memory from O(N) to O(W).
    RATTENTION (Apple, Sep 2025) shows window size as small as 512 matches
    full-attention quality when combined with global token sinks.

    Attributes:
        enabled: Whether to enable sliding window attention.
        window_size: Sliding window size (number of tokens to cache).
        use_token_sink: Whether to add global "sink" tokens for distant context.
        num_sink_tokens: Number of sink tokens (default 1, StreamingLLM-style).
    """
    enabled: bool = False
    window_size: int = 512
    use_token_sink: bool = True
    num_sink_tokens: int = 1


@dataclass
class MoSAConfig:
    """Configuration for MoSA — Mixture of Sparse Attention (v0.10, NeurIPS '25).

    MoE-inspired content-based learnable sparse attention. Dynamically selects
    tokens for each attention head using expert-choice routing.

    Attributes:
        enabled: Whether to enable MoSA sparse attention.
        num_sparse_experts: Number of sparse attention pattern experts.
        top_k_experts: Active experts per token.
        sparsity_ratio: Target sparsity ratio (0.5 = keep 50% of tokens).
    """
    enabled: bool = False
    num_sparse_experts: int = 4
    top_k_experts: int = 2
    sparsity_ratio: float = 0.5


@dataclass
class KVQuantConfig:
    """Configuration for KV Cache Quantization (v0.10, TurboQuant-inspired).

    Stores KV pairs in reduced precision (int8/int4) instead of fp16.
    TurboQuant (Google, 2025/2026) approaches information-theoretic limit.

    Attributes:
        enabled: Whether to enable KV cache quantization.
        mode: Quantization mode ("fp16", "int8", "int4", "nf4").
        group_size: Group size for group-wise quantization (int4/nf4).
    """
    enabled: bool = False
    mode: str = "int8"
    group_size: int = 64


@dataclass
class DMSConfig:
    """Configuration for Dynamic Memory Sparsification (v0.10, NeurIPS '25).

    Inference-time KV cache sparsification for hyper-scaling.
    Enables longer generation within the same memory budget.

    Attributes:
        enabled: Whether to enable DMS.
        target_cache_ratio: Target KV cache ratio (0.5 = keep 50%).
        eviction_strategy: How to select tokens for eviction.
        update_frequency: How often to run eviction (every N tokens).
        min_tokens_to_keep: Minimum tokens to keep in cache.
    """
    enabled: bool = False
    target_cache_ratio: float = 0.5
    eviction_strategy: str = "importance"
    update_frequency: int = 64
    min_tokens_to_keep: int = 32


@dataclass
class ParallelHeadConfig:
    """Configuration for Parallel Hybrid Head (v0.10, Hymba-inspired).

    Processes input through SSM and Attention simultaneously (parallel),
    then combines outputs. NVIDIA Hymba (ICLR 2025) shows this is superior
    for small-to-medium models.

    Attributes:
        enabled: Whether to use parallel head instead of sequential.
        num_meta_tokens: Number of meta tokens for cross-layer info.
        use_gated_fusion: Whether to use learned gating for fusion.
    """
    enabled: bool = False
    num_meta_tokens: int = 4
    use_gated_fusion: bool = True


@dataclass
class DualMemoryConfig:
    """Configuration for Two-Level Memory System (v0.9).

    Working memory (recent, detailed) + Long-term memory (compressed, persistent).

    Attributes:
        enabled: Whether to enable dual memory system.
        working_memory_size: Number of entries in working memory.
        long_term_memory_dim: Dimension of compressed long-term state.
        consolidation_method: "attention", "gated", or "mean".
    """
    enabled: bool = False
    working_memory_size: int = 512
    long_term_memory_dim: int = 256
    consolidation_method: str = "attention"


@dataclass
class OutputConfig:
    """Configuration for output head.

    Attributes:
        use_mtp: Whether to use Multi-Token Prediction head.
        mtp_num_tokens: Number of future tokens to predict with MTP.
        use_flow_matching: Whether to use flow matching for output refinement.
        use_speculative: Whether to use MTP speculative decoding (v0.4).
        speculative_draft_tokens: Number of draft tokens for speculative decoding.
        use_leap_mtp: Whether to use L-MTP Leap Multi-Token Prediction (v0.8, NeurIPS '25).
        leap_mtp_schedule: Leap schedule type ("geometric", "arithmetic", "adjacent").
        leap_mtp_num_leaps: Number of leap heads.
        leap_mtp_max_leap: Maximum leap distance.
    """
    use_mtp: bool = False
    mtp_num_tokens: int = 2
    use_flow_matching: bool = False
    use_speculative: bool = False
    speculative_draft_tokens: int = 2
    # v0.8 additions — L-MTP (Leap Multi-Token Prediction)
    use_leap_mtp: bool = False
    leap_mtp_schedule: str = "geometric"
    leap_mtp_num_leaps: int = 4
    leap_mtp_max_leap: int = 8
    # v0.9 — Anchored Diffusion Decoder
    use_anchored_decoder: bool = False
    anchored_n_refine_steps: int = 3
    anchored_d_refine: int = 512


@dataclass
class RecurrentConfig:
    """Configuration for Recurrent-Depth Transformer (RDT, v0.6).

    Based on OpenMythos reconstruction of Claude Mythos architecture.
    Enables looped transformer blocks with shared weights for parameter efficiency.

    Attributes:
        enabled: Whether to enable RDT looped blocks.
        max_loop_iters: Maximum number of loop iterations (default ~16).
        use_lti_stable: Whether to use LTI-stable injection for training stability.
        use_act: Whether to use Adaptive Computation Time (variable depth halting).
        act_halting_threshold: Threshold for ACT halting decision.
        use_depth_lora: Whether to use per-iteration LoRA adaptation.
        depth_lora_rank: Rank for depth LoRA modules.
        use_loop_index_embedding: Whether to use loop-index positional embeddings.
    """
    enabled: bool = False
    max_loop_iters: int = 16
    use_lti_stable: bool = True
    use_act: bool = True
    act_halting_threshold: float = 0.99
    use_depth_lora: bool = True
    depth_lora_rank: int = 8
    use_loop_index_embedding: bool = True


@dataclass
class JEPAConfig:
    """Configuration for LLM-JEPA training (v0.6).

    Predicts future latent states instead of next tokens for principled training.

    Attributes:
        enabled: Whether to enable JEPA training.
        prediction_horizon: How many future latent states to predict.
        latent_dim: Dimension of predicted latent states.
        predictor_depth: Depth of the latent predictor network.
        loss_type: JEPA loss type ("vicreg", "cosine", or "mse").
        teacher_ema_decay: EMA decay rate for the target encoder.
        prediction_weight: Weight for JEPA loss vs standard LM loss.
    """
    enabled: bool = False
    prediction_horizon: int = 4
    latent_dim: int = 256
    predictor_depth: int = 3
    loss_type: str = "vicreg"
    teacher_ema_decay: float = 0.996
    prediction_weight: float = 0.1


@dataclass
class DAPOConfig:
    """Configuration for DAPO training (v0.8).

    DAPO: Decoupled Clip & Dynamic Sampling Policy Optimization.
    Improves over GRPO with 4 key innovations.

    Ref: Yu et al., "DAPO: An Open-Source LLM RL System at Scale", arXiv 2503.14476 (2025)

    Attributes:
        enabled: Whether to enable DAPO (replaces GRPO in Phase 3).
        clip_ratio_low: Lower bound clip ratio (prevents policy collapse).
        clip_ratio_high: Upper bound clip ratio (prevents reward hacking).
        dynamic_sampling: Whether to filter prompts with uniform rewards.
        token_level_loss: Whether to use token-level policy gradient loss.
        overlong_filter: Whether to filter overlong responses.
        num_responses_per_prompt: Number of responses sampled per prompt.
        kl_coefficient: KL penalty coefficient against reference policy.
    """
    enabled: bool = False
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.28
    dynamic_sampling: bool = True
    token_level_loss: bool = True
    overlong_filter: bool = True
    num_responses_per_prompt: int = 8
    kl_coefficient: float = 0.1


@dataclass
class RLVRConfig:
    """Configuration for RLVR training (v0.8).

    RLVR: Reinforcement Learning with Verifiable Rewards.
    Uses objective, programmable reward functions instead of learned reward models.

    Ref: NeurIPS 2025 (posters 119944, 116633), arXiv 2601.05607, 2603.22117

    Attributes:
        enabled: Whether to enable RLVR verifiable rewards.
        difficulty_schedule: Verification difficulty ("easy", "medium", "hard", "curriculum").
        math_tolerance: Numeric tolerance for math verification.
        code_timeout: Timeout for code execution verification.
        use_math_verifier: Whether to use math answer verification.
        use_code_verifier: Whether to use code execution verification.
        use_format_verifier: Whether to use format checking.
        curriculum_warmup_steps: Steps for curriculum difficulty warmup.
    """
    enabled: bool = False
    difficulty_schedule: str = "curriculum"
    math_tolerance: float = 1e-4
    code_timeout: int = 5
    use_math_verifier: bool = True
    use_code_verifier: bool = True
    use_format_verifier: bool = True
    curriculum_warmup_steps: int = 1000


@dataclass
class PrefetchConfig:
    """Configuration for Expert Prefetching (v0.8).

    Speculating Experts: Uses computed representations to predict which MoE experts
    will be needed in subsequent layers, enabling prefetching.

    Ref: arXiv 2603.19289 (March 2026)

    Attributes:
        enabled: Whether to enable expert prefetching.
        predictor_hidden_dim: Hidden dimension for the lightweight predictor.
        prefetch_budget: Maximum number of experts to prefetch per layer.
        adaptive_temperature: Whether to adaptively adjust prediction temperature.
    """
    enabled: bool = False
    predictor_hidden_dim: int = 128
    prefetch_budget: int = 4
    adaptive_temperature: bool = True


@dataclass
class BitNetConfig:
    """Configuration for BitNet 1.58-bit quantization.

    Attributes:
        enabled: Whether BitNet quantization is enabled.
        warmup_steps: Number of warmup steps before quantization starts.
        initial_quant_ratio: Initial quantization ratio.
        threshold: Quantization threshold.
        ste_mode: Straight-Through Estimator mode ("identity" or "atan").
    """
    enabled: bool = False
    warmup_steps: int = 2000
    initial_quant_ratio: float = 0.0
    threshold: float = 0.0
    ste_mode: str = "identity"


@dataclass
class FP8Config:
    """Configuration for FP8 training.

    Attributes:
        enabled: Whether FP8 training is enabled.
        fp8_scheme: FP8 quantization scheme ("dynamic" or "static").
    """
    enabled: bool = False
    fp8_scheme: str = "dynamic"


@dataclass
class QuantizationConfig:
    """Configuration for quantization methods.

    Attributes:
        bitnet: BitNet 1.58-bit quantization configuration.
        fp8: FP8 training configuration.
    """
    bitnet: BitNetConfig = field(default_factory=BitNetConfig)
    fp8: FP8Config = field(default_factory=FP8Config)


@dataclass
class NASConfig:
    """Configuration for Neural Architecture Search (post-training).

    Attributes:
        enabled: Whether NAS is enabled.
        search_epochs: Number of search epochs.
        darts_lr: Learning rate for DARTS architecture parameters.
    """
    enabled: bool = False
    search_epochs: int = 10
    darts_lr: float = 0.001


@dataclass
class TrainingConfig:
    """Configuration for training.

    Attributes:
        batch_size: Training batch size.
        learning_rate: Peak learning rate.
        max_steps: Maximum number of training steps.
        weight_decay: Weight decay coefficient.
        warmup_steps: Number of warmup steps.
        grad_clip: Maximum gradient norm for clipping.
        fp8_enabled: Whether FP8 training is enabled.
        precision: Training precision string.
        use_amp: Whether to use automatic mixed precision.
        amp_dtype: AMP data type ("bf16" or "fp16").
    """
    batch_size: int = 32
    learning_rate: float = 3e-4
    max_steps: int = 100000
    weight_decay: float = 0.1
    warmup_steps: int = 2000
    grad_clip: float = 1.0
    fp8_enabled: bool = False
    precision: str = "bf16"
    use_amp: bool = False
    amp_dtype: str = "bf16"

    def __post_init__(self) -> None:
        if self.amp_dtype not in ("bf16", "fp16"):
            raise ValueError(
                f"amp_dtype must be 'bf16' or 'fp16', got '{self.amp_dtype}'"
            )


@dataclass
class HardwareConfig:
    """Configuration for hardware.

    Attributes:
        device: Target device ("auto", "cuda", "cpu").
        backend: Compute backend ("auto", "cuda", "rocm").
        compile_model: Whether to use torch.compile.
        precision: Precision type for inference/compute.
    """
    device: str = "auto"
    backend: str = "auto"
    compile_model: bool = True
    precision: PrecisionType = PrecisionType.BF16


# ============================================================================
# LosionConfig — Main Configuration
# ============================================================================


@dataclass
class LosionConfig:
    """Unified configuration for the Losion model.

    The Losion model uses a Tri-Jalur (Three-Pathway) Router architecture
    combining SSM, Attention, and MoE pathways.

    Attributes:
        model_name: Name of the model configuration.
        d_model: Model hidden dimension.
        n_layers: Number of transformer layers.
        vocab_size: Vocabulary size.
        max_seq_len: Maximum sequence length.
        dropout: Dropout rate.
        ssm: SSM pathway configuration.
        attention: Attention pathway configuration.
        retrieval: Retrieval/MoE pathway configuration.
        router: Router configuration.
        output: Output head configuration.
        training: Training configuration.
        hardware: Hardware configuration.
        quantization: Quantization configuration.
        nas: Neural Architecture Search configuration.
    """
    model_name: str = "losion-base"
    d_model: int = 768
    n_layers: int = 12
    vocab_size: int = 32000
    max_seq_len: int = 4096
    dropout: float = 0.0

    # Sub-configurations
    ssm: SSMConfig = field(default_factory=SSMConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    recurrent: RecurrentConfig = field(default_factory=RecurrentConfig)
    jepa: JEPAConfig = field(default_factory=JEPAConfig)
    dapo: DAPOConfig = field(default_factory=DAPOConfig)
    rlvr: RLVRConfig = field(default_factory=RLVRConfig)
    prefetch: PrefetchConfig = field(default_factory=PrefetchConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    nas: NASConfig = field(default_factory=NASConfig)
    # v0.9 additions
    attn_res: AttnResConfig = field(default_factory=AttnResConfig)
    evoformer: EvoformerConfig = field(default_factory=EvoformerConfig)
    child_3w: Child3WConfig = field(default_factory=Child3WConfig)
    anchored_decoder: AnchoredDecoderConfig = field(default_factory=AnchoredDecoderConfig)
    dual_memory: DualMemoryConfig = field(default_factory=DualMemoryConfig)
    # v0.10 additions — Memory Efficiency (RATTENTION, TurboQuant, MoSA, DMS, Hymba)
    sliding_window: SlidingWindowConfig = field(default_factory=SlidingWindowConfig)
    mosa: MoSAConfig = field(default_factory=MoSAConfig)
    kv_quant: KVQuantConfig = field(default_factory=KVQuantConfig)
    dms: DMSConfig = field(default_factory=DMSConfig)
    parallel_head: ParallelHeadConfig = field(default_factory=ParallelHeadConfig)

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {self.n_layers}")
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if self.max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {self.max_seq_len}")

        # Auto-set d_ff if zero
        if self.retrieval.d_ff == 0:
            self.retrieval.d_ff = 4 * self.d_model

    @classmethod
    def from_yaml(cls, path: str) -> "LosionConfig":
        """Load configuration from a YAML file.

        Supports both flat and nested YAML structures as seen in
        the Losion config files (e.g., losion-1b.yaml).

        Args:
            path: Path to the YAML configuration file.

        Returns:
            LosionConfig instance.

        Raises:
            ImportError: If PyYAML is not installed.
            FileNotFoundError: If the YAML file does not exist.
        """
        if not _YAML_AVAILABLE:
            raise ImportError(
                "PyYAML is required to load YAML configs. "
                "Install it with: pip install pyyaml"
            )

        with open(path, "r") as f:
            raw = yaml.safe_load(f)

        if raw is None:
            raw = {}

        return cls._from_dict(raw)

    @classmethod
    def _from_dict(cls, raw: Dict[str, Any]) -> "LosionConfig":
        """Create LosionConfig from a dictionary (parsed YAML).

        Handles both nested (model.ssm.d_state) and flat (d_model) formats.
        """
        # If there's a top-level "model" key, extract it
        model_raw = raw.get("model", raw)

        # Top-level model parameters
        kwargs: Dict[str, Any] = {}
        kwargs["model_name"] = raw.get("model_name", model_raw.get("model_name", "losion-base"))
        kwargs["d_model"] = model_raw.get("d_model", 768)
        kwargs["n_layers"] = model_raw.get("n_layers", 12)
        kwargs["vocab_size"] = model_raw.get("vocab_size", 32000)
        kwargs["max_seq_len"] = model_raw.get("max_seq_len", 4096)
        kwargs["dropout"] = model_raw.get("dropout", 0.0)

        # SSM config
        ssm_raw = model_raw.get("ssm", {})
        ssm = SSMConfig(
            d_state=ssm_raw.get("d_state", 64),
            d_conv=ssm_raw.get("d_conv", 4),
            expand=ssm_raw.get("expand", 2),
            ssd_chunk_size=ssm_raw.get("chunk_size", ssm_raw.get("ssd_chunk_size", 256)),
            use_wkv=ssm_raw.get("use_wkv", False),
            use_delta_net=ssm_raw.get("use_delta_net", False),
            interleaving_ratios=ssm_raw.get("interleaving_ratios", [4, 1, 1]),
            use_liquid=ssm_raw.get("use_liquid", False),
            complexity_bottleneck=ssm_raw.get("complexity_bottleneck", 64),
            depth_entropy_weight=ssm_raw.get("depth_entropy_weight", 0.01),
            # v0.6 additions
            use_mamba3=ssm_raw.get("use_mamba3", False),
            use_routing_mamba=ssm_raw.get("use_routing_mamba", False),
            routing_mamba_num_experts=ssm_raw.get("routing_mamba_num_experts", 4),
            routing_mamba_active_experts=ssm_raw.get("routing_mamba_active_experts", 2),
            # v0.8 additions
            use_structured_sparse=ssm_raw.get("use_structured_sparse", False),
            structured_sparse_n_groups=ssm_raw.get("structured_sparse_n_groups", 4),
        )
        kwargs["ssm"] = ssm

        # Attention config
        attn_raw = model_raw.get("attention", {})
        thinking_mode_val = attn_raw.get("thinking_mode", "auto")
        if isinstance(thinking_mode_val, str):
            thinking_mode_val = ThinkingMode(thinking_mode_val)
        attn = AttentionConfig(
            n_heads=attn_raw.get("n_heads", 8),
            d_kv=attn_raw.get("d_kv", 64),
            mla_latent_dim=attn_raw.get("mla_latent_dim", 128),
            use_irope=attn_raw.get("use_irope", True),
            irope_ratio=attn_raw.get("irope_ratio", 3.0),
            base_interleaving_ratio=attn_raw.get("base_interleaving_ratio", 5),
            thinking_interleaving_ratio=attn_raw.get("thinking_interleaving_ratio", 2),
            thinking_mode=thinking_mode_val,
            use_lightning=attn_raw.get("use_lightning", False),
            lightning_window_size=attn_raw.get("lightning_window_size", 2048),
            lightning_chunk_size=attn_raw.get("lightning_chunk_size", 4096),
            lightning_feature_map=attn_raw.get("lightning_feature_map", "elu"),
            use_shared_attention=attn_raw.get("use_shared_attention", False),
            shared_n_groups=attn_raw.get("shared_n_groups", 1),
            shared_pattern=attn_raw.get("shared_pattern", "all_shared"),
            shared_unique_ratio=attn_raw.get("shared_unique_ratio", 0.25),
            # v0.6 additions
            use_gated_attention=attn_raw.get("use_gated_attention", False),
            use_moba=attn_raw.get("use_moba", False),
            moba_block_size=attn_raw.get("moba_block_size", 512),
            moba_top_k_blocks=attn_raw.get("moba_top_k_blocks", 4),
            # v0.8 additions
            use_cross_jalur_routing=attn_raw.get("use_cross_jalur_routing", False),
            cross_jalur_blend_alpha=attn_raw.get("cross_jalur_blend_alpha", 0.3),
            cross_jalur_graph_top_k=attn_raw.get("cross_jalur_graph_top_k", 8),
        )
        kwargs["attention"] = attn

        # Retrieval config
        ret_raw = model_raw.get("retrieval", {})
        ret = RetrievalConfig(
            num_experts=ret_raw.get("num_experts", 16),
            num_active_experts=ret_raw.get("num_active_experts", 2),
            d_ff=ret_raw.get("d_ff", 0),
            use_engram=ret_raw.get("use_engram", True),
            engram_dim=ret_raw.get("engram_dim", 128),
            use_shared_expert=ret_raw.get("use_shared_expert", True),
            top_k_routing=ret_raw.get("top_k_routing", 2),
            use_heterogeneous=ret_raw.get("use_heterogeneous", False),
            heterogeneous_min_dim=ret_raw.get("heterogeneous_min_dim", 1024),
            heterogeneous_max_dim=ret_raw.get("heterogeneous_max_dim", 8192),
            use_matryoshka=ret_raw.get("use_matryoshka", False),
            matryoshka_min_experts=ret_raw.get("matryoshka_min_experts", 2),
            matryoshka_max_experts=ret_raw.get("matryoshka_max_experts", 4),
            use_gradient_routed=ret_raw.get("use_gradient_routed", False),
            gradient_routed_lr=ret_raw.get("gradient_routed_lr", 0.01),
            use_asymmetric=ret_raw.get("use_asymmetric", False),
            asymmetric_moe_layers=ret_raw.get("asymmetric_moe_layers", []),
            # v0.6 additions
            use_smore=ret_raw.get("use_smore", False),
            smore_num_sub_trees=ret_raw.get("smore_num_sub_trees", 4),
            smore_sub_tree_depth=ret_raw.get("smore_sub_tree_depth", 2),
            use_symbolic_moe=ret_raw.get("use_symbolic_moe", False),
            # v0.8 additions
            use_infinite_moe=ret_raw.get("use_infinite_moe", False),
            infinite_moe_code_dim=ret_raw.get("infinite_moe_code_dim", 32),
            infinite_moe_hypernet_hidden=ret_raw.get("infinite_moe_hypernet_hidden", 256),
            infinite_moe_low_rank_residual=ret_raw.get("infinite_moe_low_rank_residual", True),
            infinite_moe_codebook_size=ret_raw.get("infinite_moe_codebook_size", 256),
        )
        kwargs["retrieval"] = ret

        # Router config
        router_raw = model_raw.get("router", {})
        routing_type_val = router_raw.get("routing_type", "adaptive")
        if isinstance(routing_type_val, str):
            routing_type_val = RoutingType(routing_type_val)
        router = RouterConfig(
            routing_type=routing_type_val,
            use_thinking_toggle=router_raw.get("use_thinking_toggle", True),
            bias_lr=router_raw.get("bias_lr", 0.01),
            aux_loss_weight=router_raw.get("aux_loss_weight", 0.0),
            top_k_pathways=router_raw.get("top_k_pathways", 2),
        )
        kwargs["router"] = router

        # Output config
        out_raw = model_raw.get("output", {})
        output = OutputConfig(
            use_mtp=out_raw.get("use_mtp", False),
            mtp_num_tokens=out_raw.get("mtp_num_tokens", 2),
            use_flow_matching=out_raw.get("use_flow_matching", False),
            use_speculative=out_raw.get("use_speculative", False),
            speculative_draft_tokens=out_raw.get("speculative_draft_tokens", 2),
            # v0.8 additions — L-MTP
            use_leap_mtp=out_raw.get("use_leap_mtp", False),
            leap_mtp_schedule=out_raw.get("leap_mtp_schedule", "geometric"),
            leap_mtp_num_leaps=out_raw.get("leap_mtp_num_leaps", 4),
            leap_mtp_max_leap=out_raw.get("leap_mtp_max_leap", 8),
            # v0.9 — Anchored Diffusion Decoder
            use_anchored_decoder=out_raw.get("use_anchored_decoder", False),
            anchored_n_refine_steps=out_raw.get("anchored_n_refine_steps", 3),
            anchored_d_refine=out_raw.get("anchored_d_refine", 512),
        )
        kwargs["output"] = output

        # Training config
        train_raw = raw.get("training", {})
        training = TrainingConfig(
            batch_size=train_raw.get("batch_size", 32),
            learning_rate=train_raw.get("learning_rate", 3e-4),
            max_steps=train_raw.get("max_steps", 100000),
            weight_decay=train_raw.get("weight_decay", 0.1),
            warmup_steps=train_raw.get("warmup_steps", 2000),
            grad_clip=train_raw.get("grad_clip", 1.0),
            fp8_enabled=train_raw.get("fp8_enabled", False),
            precision=train_raw.get("precision", "bf16"),
        )
        kwargs["training"] = training

        # Hardware config
        hw_raw = raw.get("hardware", {})
        precision_val = hw_raw.get("precision", "bf16")
        if isinstance(precision_val, str):
            precision_val = PrecisionType(precision_val)
        hardware = HardwareConfig(
            device=hw_raw.get("device", "auto"),
            backend=hw_raw.get("backend", "auto"),
            compile_model=hw_raw.get("compile_model", True),
            precision=precision_val,
        )
        kwargs["hardware"] = hardware

        # Quantization config
        quant_raw = model_raw.get("quantization", {})
        bitnet_raw = quant_raw.get("bitnet", {})
        fp8_raw = quant_raw.get("fp8", {})
        quantization = QuantizationConfig(
            bitnet=BitNetConfig(
                enabled=bitnet_raw.get("enabled", False),
                warmup_steps=bitnet_raw.get("warmup_steps", 2000),
                initial_quant_ratio=bitnet_raw.get("initial_quant_ratio", 0.0),
                threshold=bitnet_raw.get("threshold", 0.0),
                ste_mode=bitnet_raw.get("ste_mode", "identity"),
            ),
            fp8=FP8Config(
                enabled=fp8_raw.get("enabled", False),
                fp8_scheme=fp8_raw.get("fp8_scheme", "dynamic"),
            ),
        )
        kwargs["quantization"] = quantization

        # NAS config
        nas_raw = model_raw.get("nas", {})
        nas = NASConfig(
            enabled=nas_raw.get("enabled", False),
            search_epochs=nas_raw.get("search_epochs", 10),
            darts_lr=nas_raw.get("darts_lr", 0.001),
        )
        kwargs["nas"] = nas

        # v0.6: Recurrent config
        rec_raw = model_raw.get("recurrent", {})
        recurrent = RecurrentConfig(
            enabled=rec_raw.get("enabled", False),
            max_loop_iters=rec_raw.get("max_loop_iters", 16),
            use_lti_stable=rec_raw.get("use_lti_stable", True),
            use_act=rec_raw.get("use_act", True),
            act_halting_threshold=rec_raw.get("act_halting_threshold", 0.99),
            use_depth_lora=rec_raw.get("use_depth_lora", True),
            depth_lora_rank=rec_raw.get("depth_lora_rank", 8),
            use_loop_index_embedding=rec_raw.get("use_loop_index_embedding", True),
        )
        kwargs["recurrent"] = recurrent

        # v0.6: JEPA config
        jepa_raw = model_raw.get("jepa", {})
        jepa = JEPAConfig(
            enabled=jepa_raw.get("enabled", False),
            prediction_horizon=jepa_raw.get("prediction_horizon", 4),
            latent_dim=jepa_raw.get("latent_dim", 256),
            predictor_depth=jepa_raw.get("predictor_depth", 3),
            loss_type=jepa_raw.get("loss_type", "vicreg"),
            teacher_ema_decay=jepa_raw.get("teacher_ema_decay", 0.996),
            prediction_weight=jepa_raw.get("prediction_weight", 0.1),
        )
        kwargs["jepa"] = jepa

        # v0.8: DAPO config
        dapo_raw = model_raw.get("dapo", {})
        dapo = DAPOConfig(
            enabled=dapo_raw.get("enabled", False),
            clip_ratio_low=dapo_raw.get("clip_ratio_low", 0.2),
            clip_ratio_high=dapo_raw.get("clip_ratio_high", 0.28),
            dynamic_sampling=dapo_raw.get("dynamic_sampling", True),
            token_level_loss=dapo_raw.get("token_level_loss", True),
            overlong_filter=dapo_raw.get("overlong_filter", True),
            num_responses_per_prompt=dapo_raw.get("num_responses_per_prompt", 8),
            kl_coefficient=dapo_raw.get("kl_coefficient", 0.1),
        )
        kwargs["dapo"] = dapo

        # v0.8: RLVR config
        rlvr_raw = model_raw.get("rlvr", {})
        rlvr = RLVRConfig(
            enabled=rlvr_raw.get("enabled", False),
            difficulty_schedule=rlvr_raw.get("difficulty_schedule", "curriculum"),
            math_tolerance=rlvr_raw.get("math_tolerance", 1e-4),
            code_timeout=rlvr_raw.get("code_timeout", 5),
            use_math_verifier=rlvr_raw.get("use_math_verifier", True),
            use_code_verifier=rlvr_raw.get("use_code_verifier", True),
            use_format_verifier=rlvr_raw.get("use_format_verifier", True),
            curriculum_warmup_steps=rlvr_raw.get("curriculum_warmup_steps", 1000),
        )
        kwargs["rlvr"] = rlvr

        # v0.8: Prefetch config
        pf_raw = model_raw.get("prefetch", {})
        prefetch = PrefetchConfig(
            enabled=pf_raw.get("enabled", False),
            predictor_hidden_dim=pf_raw.get("predictor_hidden_dim", 128),
            prefetch_budget=pf_raw.get("prefetch_budget", 4),
            adaptive_temperature=pf_raw.get("adaptive_temperature", True),
        )
        kwargs["prefetch"] = prefetch

        # v0.9: AttnRes config
        ar_raw = model_raw.get("attn_res", {})
        attn_res = AttnResConfig(
            enabled=ar_raw.get("enabled", False),
            mode=ar_raw.get("mode", "block"),
            num_blocks=ar_raw.get("num_blocks", 8),
            dropout=ar_raw.get("dropout", 0.0),
            use_gate=ar_raw.get("use_gate", True),
            temperature=ar_raw.get("temperature", 1.0),
            compression_dim=ar_raw.get("compression_dim", 0),
            use_token_compression=ar_raw.get("use_token_compression", False),
            token_compression_type=ar_raw.get("token_compression_type", "gated"),
            token_compression_d_state=ar_raw.get("token_compression_d_state", 256),
        )
        kwargs["attn_res"] = attn_res

        # v0.9: Evoformer config
        evo_raw = model_raw.get("evoformer", {})
        evoformer = EvoformerConfig(
            enabled=evo_raw.get("enabled", False),
            n_recycling_steps=evo_raw.get("n_recycling_steps", 3),
            use_layer_recycling=evo_raw.get("use_layer_recycling", True),
            use_token_recycling=evo_raw.get("use_token_recycling", True),
            use_decoder_feedback=evo_raw.get("use_decoder_feedback", True),
            use_prediction_recycling=evo_raw.get("use_prediction_recycling", True),
            use_router_coevolve=evo_raw.get("use_router_coevolve", True),
        )
        kwargs["evoformer"] = evoformer

        # v0.9: Child-3W config
        c3w_raw = model_raw.get("child_3w", {})
        child_3w = Child3WConfig(
            enabled=c3w_raw.get("enabled", False),
            num_children=c3w_raw.get("num_children", 4),
            top_k_children=c3w_raw.get("top_k_children", 2),
            use_mla=c3w_raw.get("use_mla", False),
            mla_latent_dim=c3w_raw.get("mla_latent_dim", 0),
            load_balance_weight=c3w_raw.get("load_balance_weight", 0.01),
        )
        kwargs["child_3w"] = child_3w

        # v0.9: Dual Memory config
        dm_raw = model_raw.get("dual_memory", {})
        dual_memory = DualMemoryConfig(
            enabled=dm_raw.get("enabled", False),
            working_memory_size=dm_raw.get("working_memory_size", 512),
            long_term_memory_dim=dm_raw.get("long_term_memory_dim", 256),
            consolidation_method=dm_raw.get("consolidation_method", "attention"),
        )
        kwargs["dual_memory"] = dual_memory

        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to a dictionary.

        Handles enum values by converting them to their string values.
        """
        import dataclasses

        def _convert(obj: Any) -> Any:
            if isinstance(obj, enum.Enum):
                return obj.value
            if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
                return {k: _convert(v) for k, v in dataclasses.asdict(obj).items()}
            if isinstance(obj, list):
                return [_convert(item) for item in obj]
            if isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            return obj

        return _convert(self)

    def estimated_parameters(self) -> int:
        """Estimate the total number of parameters in the model.

        Provides a rough estimate based on model dimensions. Useful for
        comparing different configurations without building the model.

        Returns:
            Estimated parameter count.
        """
        d = self.d_model
        n = self.n_layers
        v = self.vocab_size

        # Token embedding
        emb_params = v * d

        # Per-layer estimate
        # SSM pathway: roughly 4 * d^2 (projections + gating)
        ssm_params = 4 * d * d * self.ssm.expand

        # Attention pathway: Q, K, V, O projections + MLA compression
        attn_params = (
            4 * d * self.attention.n_heads * self.attention.d_kv  # Q, K, V, O
            + d * self.attention.mla_latent_dim  # KV down-projection
            + self.attention.mla_latent_dim * self.attention.n_heads * self.attention.d_kv * 2  # K, V up
        )

        # Retrieval/MoE pathway
        num_experts = self.retrieval.num_experts if self.retrieval.num_experts > 0 else max(8, min(64, d // 32))
        d_ff = self.retrieval.d_ff if self.retrieval.d_ff > 0 else 4 * d
        active_experts = self.retrieval.num_active_experts
        # Each expert: SwiGLU FFN (gate + up + down) = 3 * d * d_ff
        expert_params = num_experts * 3 * d * d_ff
        # Shared expert (if enabled)
        shared_params = 3 * d * d_ff if self.retrieval.use_shared_expert else 0
        # Router
        router_params = d * num_experts
        # Engram memory projection
        engram_params = d * self.retrieval.engram_dim + self.retrieval.engram_dim * d if self.retrieval.use_engram else 0

        retrieval_params = expert_params + shared_params + router_params + engram_params

        # Layer norm (2 per layer: pre + post)
        norm_params = 2 * d

        # Per-layer total
        layer_params = ssm_params + attn_params + retrieval_params + norm_params

        # Total across layers
        total_layer_params = n * layer_params

        # Final layer norm
        final_norm_params = d

        # LM head (if not tied)
        lm_head_params = v * d

        # MTP heads (if enabled)
        mtp_params = 0
        if self.output.use_mtp:
            mtp_params = self.output.mtp_num_tokens * (d * d + d * v)

        total = emb_params + total_layer_params + final_norm_params + lm_head_params + mtp_params

        return total

    def __repr__(self) -> str:
        parts = [
            f"LosionConfig(",
            f"  model_name={self.model_name!r},",
            f"  d_model={self.d_model},",
            f"  n_layers={self.n_layers},",
            f"  vocab_size={self.vocab_size},",
            f"  max_seq_len={self.max_seq_len},",
            f"  dropout={self.dropout},",
            f"  ssm=SSMConfig(d_state={self.ssm.d_state}, d_conv={self.ssm.d_conv}, expand={self.ssm.expand}),",
            f"  attention=AttentionConfig(n_heads={self.attention.n_heads}, d_kv={self.attention.d_kv}, mla_latent_dim={self.attention.mla_latent_dim}),",
            f"  retrieval=RetrievalConfig(num_experts={self.retrieval.num_experts}, num_active_experts={self.retrieval.num_active_experts}),",
            f"  router=RouterConfig(routing_type={self.router.routing_type!r}, use_thinking_toggle={self.router.use_thinking_toggle}),",
            f"  output=OutputConfig(use_mtp={self.output.use_mtp}, mtp_num_tokens={self.output.mtp_num_tokens}),",
            f"  training=TrainingConfig(batch_size={self.training.batch_size}, learning_rate={self.training.learning_rate}),",
            f"  hardware=HardwareConfig(device={self.hardware.device!r}, precision={self.hardware.precision!r}),",
            f")",
        ]
        return "\n".join(parts)
