"""
Losion — Hybrid AI Framework with Tri-Jalur Router Architecture.

Version 2.0.0 — "Alive Gradients & Production Ready"

v2.0.0 Alive Gradients & Production Ready:
  - CRITICAL FIX: AuxFreeMoE MTP loss now propagated to model total loss
    Previously 32.2% of model params (MTPMoEHead pred_heads) were dead weight —
    mtp_loss was computed but never added to loss, so 32 tensors had zero gradient.
    Now loss += avg_moe_mtp_loss from all layers' retrieval_aux["mtp_loss"].
  - All gradient paths verified: every parameter receives non-zero gradient during training
  - Repo polished for open-source publication

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

__version__ = "2.0.0"
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
