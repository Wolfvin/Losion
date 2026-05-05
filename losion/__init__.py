"""
Losion — Hybrid AI Framework with Tri-Jalur Router Architecture.

Version 2.2.0 — "Deep Audit & Bug Purge"

v2.2.0 Deep Audit & Bug Purge:
  - CRITICAL: Double-softplus dt bias FIXED in Mamba2/3/Liquid/PoST/StructuredSparse
    — dt_bias was pre-softplused then softplus applied again, making dt 11x too large
  - CRITICAL: Per-channel dt/A selectivity RESTORED in Mamba3/RoutingMamba/Liquid/PoST
    — Averaging over d_inner destroyed Mamba's core per-channel selectivity innovation
  - CRITICAL: WKV state shape mismatch FIXED — 3D init_state now correctly 2D
  - CRITICAL: DeltaNet/GDN state output uses per-position state, not final state
  - CRITICAL: MLA Key reconstruction no longer uses Value dims as Key dimensions
  - CRITICAL: Double RoPE application in GatedMultiHeadAttention MLA path fixed
  - CRITICAL: MLA KV cache returns proper 2-tuple (not 1-tuple) for interface compat
  - CRITICAL: GRPO policy update now uses fresh forward pass (was stale, ratio always 1.0)
  - CRITICAL: NAS compute_nas_loss returns total_loss (was returning task_loss, discarding arch reg)
  - CRITICAL: Missing `import math` in generation.py fixed
  - HIGH: Butterfly materialization composes (multiplies) instead of overwriting
  - HIGH: Mamba3 dual token shift now functional during inference (prev_token cache)
  - HIGH: Mamba3 uses correct dt*B discretization instead of softplus(dt)*B
  - HIGH: FG2-GDN temperature parameters are now nn.Parameter (were buffers, non-learnable)
  - HIGH: RoutingMamba inference adds exp clamp to prevent overflow
  - HIGH: SmoreMoE uses expert index (not sub-tree index) for load tracking
  - HIGH: BidirectionalTokenUpdate uses no causal mask (was using causal = not bidirectional)
  - HIGH: AttnRes stores non-detached outputs for gradient flow to original layers
  - HIGH: Child3W avoids double-counting when same child selected twice
  - HIGH: ExpertChoiceMoE renormalizes by token count for multi-expert accumulation
  - HIGH: GRPO reward uses actual reward_fn (was random noise placeholder)
  - HIGH: DAPO decoupled clip uses correct single-clamp formulation
  - HIGH: Beam search applies length_penalty normalization
  - HIGH: In-place logits modification replaced with clone() for gradient safety
  - MEDIUM: LiquidSSM A-modulation consistent between training and inference
  - MEDIUM: C slicing uses explicit end index in inference paths
  - MEDIUM: StructuredSparse creates tensors on correct device/dtype
  - MEDIUM: SymbolicMoE MATHEMATICAL enum typo fixed
  - MEDIUM: context_extension.py missing logger import added
  - MEDIUM: ACT halting_threshold default changed from 0.99 to 0.01
  - MEDIUM: PagedKVCache free_page detects double-free
  - MEDIUM: Generation add_request squeezes 2D input_ids
  - MEDIUM: LeapMTP total_loss uses requires_grad=True initialization
  - MEDIUM: InfiniteMoE diversity loss bounded with log(1+dist)
  - MEDIUM: Evoformer RouterCoevolve uses 1e-4 coefficient (was *0 = dead gradient)
  - MEDIUM: DualMemory write() no longer detaches (gradient flows to LTM)
  - MEDIUM: PostDecay gamma padding enforced with d_inner%nhads==0 assertion
  - MEDIUM: LiquidSSM depth_entropy normalized by max_entropy
  - MEDIUM: LiquidSSM complexity_scale expanded per-channel instead of averaged to scalar
  - MEDIUM: KDA+MLA cumulative_state initialized from initial_state
  - MEDIUM: SharedAttentionPool QK norm applied after RoPE (consistent with other modules)
  - LOW: DeltaNet/GDN torch.stack→torch.cat for correct state output shape

v2.1.0 Honest Code & Real Kernels:
  - CRITICAL: _triton_associative_scan now has a REAL Triton GPU kernel
    (previously it just called _pytorch_associative_scan — fake Triton claim)
  - CRITICAL: use_cache in generate() is now FUNCTIONAL — KV pairs are
    cached during prefill and reused during decode, reducing attention
    from O(n²) to O(n) per token
  - CRITICAL: Evoformer gradient flow fixed — hidden_states are NO LONGER
    detached when Evoformer is active, so gradients flow naturally through
    the recycling pathway to all layers
  - HIGH: iRoPE actually implemented — self.interleaved now controls real
    interleaved rotation pattern in RoPE.forward() (previously stored but unused)
  - HIGH: _align_dim lazy module creation FIXED — projections now created
    eagerly in __init__ via _infer_output_dim(), compatible with
    torch.compile and deterministic DDP initialization
  - HIGH: Inter-chunk propagation VECTORIZED — eliminated Python for-loop
    over chunks, replaced with vectorized prefix-scan (same log-space cumsum
    trick as intra-chunk scan)
  - MEDIUM: Gradient checkpointing lambda closure bug FIXED — replaced with
    module-level _checkpoint_layer_fn() to avoid reference capture issues
  - MEDIUM: MTP loss requires_grad guard FIXED — uses self.training instead
    of mtp_l.requires_grad, which was False under torch.no_grad() context
  - MEDIUM: Dead code modules documented — losion/agent/, losion/safety/,
    losion/core/reasoning/ marked as "experimental" (not in model forward path)
  - LOW: Audit documentation now honest — no more inflated scores

v1.9.0 Complete Gradient Flow & Vectorized Attention:
  - Evoformer LayerRecycling: revision now applied to deep layers so recycled[-1] carries gradient
  - Evoformer RouterExpertCoevolve: update_state returns differentiable update, no silent no_grad
  - DualMemory: write() no longer detaches, consolidate() returns differentiable new_state
  - DualMemory: retrieve() consolidates on every read for gradient flow to LTM params
  - LosionModelV2: all_hidden_states no longer detached for evoformer gradient flow
  - AuxFreeMoE: vocab_size default changed to None (explicit config required)
  - LightningAttention: vectorized pair_mask construction replaces nested batch+seq loop
  - losion_orchestrator.py: all bare except Exception:pass replaced with proper logging
  - mamba2.py: removed unused einops import
  - CI/CD: proper GitHub Actions workflow with version sync, lint, test
  - 16 previously no-grad parameters now receive gradients (evoformer + dual_memory)

v1.8.0 Per-Channel Selectivity & Deep Gradient Flow:
  - Mamba2SSD dt_avg/A_avg FIXED: per-channel dt dan A sekarang terjaga sepenuhnya
  - ThinkingToggle depth_multiplier FULLY differentiable: sigmoid soft-blending menggantikan hard if/else
  - Entropy regularization dari SEMUA layer (sebelumnya hanya layer 0)
  - MTP target alignment FIXED: menggunakan labels (bukan shift_labels) sesuai DeepSeek-V3
  - set_force_thinking race condition FIXED: thinking_mode sekarang passed as kwarg
  - Bare except Exception:pass FIXED: proper error logging dengan logging.warning()
  - SSM forward fallback sekarang log error, bukan silent pass

v1.7.0 Full Differentiable Gradient Flow & Loop-Free SSM:
  - ThinkingToggle: depth_multiplier & confidence sekarang torch.Tensor (bukan Python float)
  - Entropy regularization FIXED: compute_routing_entropy TANPA torch.no_grad(), gradien mengalir
  - Entropy key lookup FIXED: routing_info sekarang punya "adjusted_weights" key
  - Route weights NON-DETACHED: gradien mengalir ke router melalui entropy loss
  - ssd_chunk_scan TANPA Python token loop: cumsum-based parallel scan (log-space trick)
  - MyPy || true REMOVED dari CI: type-checking sekarang required

v1.6.1 Critical Bug Fixes & Gradient Flow Repair:
  - ThinkingToggle dead grad FIXED: thinking_score tensor now flows to router differentiably
  - MTP vocab_size hardcoded 32000 FIXED: config.vocab_size now properly forwarded
  - symbolic_moe num_active_experts TypeError FIXED
  - Mamba2 SSD connected to chunk_parallel_scan (no Python loop for seq>chunk_size)
  - GatedAttentionHead now uses SDPA with manual fallback
  - RoPE offset for K FIXED in both GatedAttentionHead and GatedMultiHeadAttention
  - CI || true REMOVED from pytest/train_test.py — tests can actually fail now
  - Entropy regularization ACTIVATED in LosionForCausalLMV2 loss
  - aux_free_moe load counting vectorized via bincount
  - aux_free_moe device mismatch fixed
  - Router double softmax replaced with single renormalization
  - Top-P sampling scatter FIXED in generate()
  - _align_dim now uses add_module for proper state_dict tracking
  - MTP labels now forwarded to MoE layers for gradient flow
  - chunk_parallel_scan output shape fixed — preserves per-channel info

v1.6.0 Training & Pretraining Fully Optimized:
  - RWKV-7 FULLY parallel WKV scan (cumsum-based, ZERO Python token loop)
  - Mamba-2/3 cumsum-based parallel scan (no Python token loop)
  - SDPA / Flash Attention with 3-tier fallback (flash_attn → SDPA → manual)
  - Activation Offloading: CPU/GPU offload with async prefetching
  - Memory-Aware Batch Scheduler: dynamic batch size based on VRAM
  - 8-bit Optimizer: bitsandbytes/torchao Adam with 75% memory reduction
  - LoRA / QLoRA: parameter-efficient fine-tuning (0.1-1% parameters)
  - Progressive Sequence Length: gradual seq_len increase for faster training
  - Communication-Computation Overlap: async AllReduce with gradient bucketing
  - Selective Gradient Checkpointing: per-op checkpointing (cheap vs expensive ops)
  - Dynamic Loss Scaler: adaptive scaling for mixed-precision training
  - FSDP2 / fully_shard API with LosionLayerV2 per-layer sharding
  - Per-jalur gradient checkpointing with selective recomputation
  - CPU offload for optimizer states (ZeRO-Offload style)
  - FP8 training via torchao (optional, simulated fallback)
  - PathwayEarlyExit with adaptive thresholds
  - PagedKVCacheManager with INT4 quantization
  - torch.compile(mode="reduce-overhead") integration
  - Gradient compression for distributed training (PowerSGD)
  - Context parallelism (ring attention + SSM state propagation)
  - Expert parallelism with AllToAll dispatcher
  - CI/CD with version sync check
  - LosionForCausalLMV2 as primary export

v1.5.0 Training & Kernel Optimization:
  - SDPA / Flash Attention with 3-tier fallback
  - RWKV-7 parallel WKV scan
  - Per-jalur gradient checkpointing
  - CPU offload, FSDP2, FP8 training
  - PagedKVCacheManager, EarlyExit
  - CI/CD with version sync

v1.0.0 End-to-End Verified:
  All 40+ components tested with forward+backward passes.
  17M-parameter model trained for 10 steps, all pathways verified.

Losion combines three complementary computational pathways into a single
adaptive architecture:

  Jalur 1 (SSM):           Mamba-2 SSD + RWKV-7 WKV + Gated DeltaNet
                            + Mamba-3 + Routing Mamba + Liquid SSM
                            + PoST Decay + FG2-GDN + Structured Sparse SSM
  Jalur 2 (Attention):     MLA + iRoPE + Lightning Attention + RoPE
                            + Gated Attention + MoBA + KDA+MLA
                            + Cross-Jalur Attention-MoE Routing
                            + AttnRes (Attention Residuals, MoonshotAI 2026)
                            + Child-3W (QKV-level MoE routing)
  Jalur 3 (Retrieval):     MoE + Engram Memory + Expert Choice
                            + S'MoRE + Symbolic-MoE + AuxFreeMoE
                            + ∞-MoE + MoHGE (Heterogeneous Grouped Experts)

Router:  Adaptive (BiasRouter + ThinkingToggle + Symbolic-MoE), GRPO/DAPO-trained.
         + Router ↔ Expert Co-Evolution (Evoformer Level 5)
"""

__version__ = "2.2.0"
__author__ = "Losion Contributors"
__license__ = "MIT"

# ============================================================================
# Public API — Primary Exports
# ============================================================================

# --- Configuration ---
from losion.config import (
    LosionConfig,
    SSMConfig,
    AttentionConfig,
    RetrievalConfig,
    RouterConfig,
    TrainingConfig,
    HardwareConfig,
    RecurrentConfig,
    JEPAConfig,
    DAPOConfig,
    RLVRConfig,
    OutputConfig,
    AttnResConfig,
    EvoformerConfig,
    Child3WConfig,
    DualMemoryConfig,
    AnchoredDecoderConfig,
    PrefetchConfig,
    QuantizationConfig,
    BitNetConfig,
    FP8Config,
    NASConfig,
    PrecisionType,
    RoutingType,
    ThinkingMode,
)

# --- Models (V2 as primary) ---
from losion.models.losion_model_v2 import (
    LosionModelV2,
    LosionForCausalLMV2,
    LosionLayerV2,
    MTPHead,
    RoPE,
    RMSNorm,
)

# --- Legacy V1 models (backward compatible) ---
from losion.models.losion_model import LosionModel, LosionLayer
from losion.models.losion_decoder import LosionForCausalLM

# --- Kernel Optimizations ---
try:
    from losion.core.kernel.sdpa_compat import sdpa_attention, SDPACompat
except ImportError:
    pass

try:
    from losion.core.kernel.early_exit import PathwayEarlyExit
except ImportError:
    pass

try:
    from losion.core.kernel.flash_attn import FlashAttentionWrapper, RingAttention
except ImportError:
    pass

try:
    from losion.core.kernel.ssm_kernels import (
        associative_scan,
        chunk_parallel_scan,
        rwkv7_parallel_wkv,
        HAS_TRITON,
    )
except ImportError:
    pass

try:
    from losion.core.kernel.kv_cache import PagedKVCacheManager, INT4KVCacheQuantizer
except ImportError:
    pass

try:
    from losion.core.kernel.training_optim import (
        MemoryEfficientTrainer,
        CUDAGraphOptimizer,
        GradientCompressor,
        FusedAdamW,
        CPUOffloadOptimizer,
    )
except ImportError:
    pass

try:
    from losion.core.kernel.fp8_utils import FP8TrainingWrapper, has_fp8_support
except ImportError:
    pass

# --- Advanced Training Optimizations (v1.6.0) ---
try:
    from losion.core.kernel.advanced_training import (
        ActivationOffloader,
        MemoryAwareBatchScheduler,
        EightBitOptimizer,
        LoRAAdapter,
        LoRALayer,
        ProgressiveSequenceScheduler,
        CommComputeOverlap,
        SelectiveGradientCheckpointing,
        DynamicLossScaler,
    )
except ImportError:
    pass

# --- Distributed ---
try:
    from losion.distributed.parallel import (
        ParallelismConfig,
        LosionFSDPWrapper,
        ContextParallel,
        LosionDistributedTrainer,
    )
except ImportError:
    pass

# --- Training ---
try:
    from losion.training.trainer import LosionTrainer, TrainerConfig
except ImportError:
    pass

# --- Inference ---
try:
    from losion.inference.kv_cache import KVCache
except ImportError:
    pass

__all__ = [
    # Version
    "__version__",
    "__author__",
    "__license__",
    # Config
    "LosionConfig",
    "SSMConfig",
    "AttentionConfig",
    "RetrievalConfig",
    "RouterConfig",
    "TrainingConfig",
    "HardwareConfig",
    "RecurrentConfig",
    "JEPAConfig",
    "DAPOConfig",
    "RLVRConfig",
    "OutputConfig",
    # Models (V2 primary) — CORE, IN PRODUCTION FORWARD PATH
    "LosionModelV2",
    "LosionForCausalLMV2",
    "LosionLayerV2",
    "MTPHead",
    "RoPE",
    "RMSNorm",
    # Legacy V1
    "LosionModel",
    "LosionLayer",
    "LosionForCausalLM",
    # Kernel — CORE (used in SSM forward path)
    "associative_scan",
    "chunk_parallel_scan",
    "rwkv7_parallel_wkv",
    "HAS_TRITON",
    # Kernel — EXPERIMENTAL (exported but NOT in model forward path)
    # These are utility classes for advanced usage. They are not wired into
    # LosionModelV2 by default but can be used standalone.
    "PathwayEarlyExit",
    "FlashAttentionWrapper",
    "RingAttention",
    "sdpa_attention",
    "SDPACompat",
    "PagedKVCacheManager",
    "INT4KVCacheQuantizer",
    "MemoryEfficientTrainer",
    "CUDAGraphOptimizer",
    "GradientCompressor",
    "FusedAdamW",
    "CPUOffloadOptimizer",
    "FP8TrainingWrapper",
    "has_fp8_support",
    # Advanced Training — EXPERIMENTAL
    "ActivationOffloader",
    "MemoryAwareBatchScheduler",
    "EightBitOptimizer",
    "LoRAAdapter",
    "LoRALayer",
    "ProgressiveSequenceScheduler",
    "CommComputeOverlap",
    "SelectiveGradientCheckpointing",
    "DynamicLossScaler",
    # Distributed — EXPERIMENTAL
    "ParallelismConfig",
    "LosionFSDPWrapper",
    "ContextParallel",
    "LosionDistributedTrainer",
    # Training
    "LosionTrainer",
    "TrainerConfig",
    # Inference
    "KVCache",
]
