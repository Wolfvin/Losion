"""
Losion — Hybrid AI Framework with Tri-Jalur Router Architecture.

Version 1.6.1 — "Critical Bug Fixes & Gradient Flow Repair"

v1.6.1 Critical Bug Fixes & Gradient Flow Repair:
  - ThinkingToggle dead grad FIXED: thinking_score tensor now flows to router differentiably
  - MTP vocab_size hardcoded 32000 FIXED: config.vocab_size now properly forwarded
  - symbolic_moe num_active_experts TypeError FIXED
  - Mamba2 SSD connected to chunk_parallel_scan (no Python loop for seq>chunk_size)
  - GatedAttentionHead now uses SDPA with manual fallback
  - RoPE offset for K FIXED in both GatedAttentionHead and GatedMultiHeadAttention
  - CI || true REMOVED — tests can actually fail now
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

__version__ = "1.6.1"
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
    # Models (V2 primary)
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
    # Kernel
    "PathwayEarlyExit",
    "FlashAttentionWrapper",
    "RingAttention",
    "sdpa_attention",
    "SDPACompat",
    "associative_scan",
    "chunk_parallel_scan",
    "rwkv7_parallel_wkv",
    "HAS_TRITON",
    "PagedKVCacheManager",
    "INT4KVCacheQuantizer",
    "MemoryEfficientTrainer",
    "CUDAGraphOptimizer",
    "GradientCompressor",
    "FusedAdamW",
    "CPUOffloadOptimizer",
    "FP8TrainingWrapper",
    "has_fp8_support",
    # Advanced Training (v1.6.0)
    "ActivationOffloader",
    "MemoryAwareBatchScheduler",
    "EightBitOptimizer",
    "LoRAAdapter",
    "LoRALayer",
    "ProgressiveSequenceScheduler",
    "CommComputeOverlap",
    "SelectiveGradientCheckpointing",
    "DynamicLossScaler",
    # Distributed
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
