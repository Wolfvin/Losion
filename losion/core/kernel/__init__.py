"""
Losion Kernel Optimizations — Low-level kernels for training and inference.

Provides optimized implementations for:
  - SDPA / Flash Attention with multi-tier fallback (sdpa_compat module)
  - SSM parallel scans: associative, chunk, RWKV-7 WKV (ssm_kernels module)
  - Early exit with adaptive thresholds (early_exit module)
  - Paged KV Cache with INT4 quantization (kv_cache module)
  - Training memory optimizations: CPU offload, gradient compression,
    CUDA Graphs, fused optimizer (training_optim module)
  - FP8 training wrappers (fp8_utils module)
  - Ring Attention for context parallelism (flash_attn module)

Module quick-reference:
  from losion.core.kernel import PagedKVCacheManager
  from losion.core.kernel import PathwayEarlyExit
  from losion.core.kernel import FlashAttentionWrapper, RingAttention
  from losion.core.kernel import sdpa_attention, SDPACompat
  from losion.core.kernel import associative_scan, chunk_parallel_scan, rwkv7_parallel_wkv
  from losion.core.kernel import MemoryEfficientTrainer, GradientCompressor, FusedAdamW, CPUOffloadOptimizer
  from losion.core.kernel import FP8TrainingWrapper, has_fp8_support

Credits:
  - Flash Attention: Dao et al., arXiv:2205.14135 (2022)
  - SDPA: PyTorch native F.scaled_dot_product_attention (2023)
  - Triton: OpenAI Triton language (2023)
  - ZeRO-Offload: Rajbhandari et al., SC 2021
  - torchao: PyTorch Architecture Optimization (2024)
  - Ring Attention: Liu et al., arXiv:2310.01889 (2023)
  - PagedAttention: Kwon et al., SOSP 2023
  - PowerSGD: Vogt et al., NeurIPS 2019
"""

# Detect available backends
import torch

HAS_FLASH_ATTN = False
HAS_TRITON = False
HAS_CUDA = torch.cuda.is_available()

# Check flash_attn package
try:
    import flash_attn
    HAS_FLASH_ATTN = True
except ImportError:
    pass

# Check Triton
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    pass

# ============================================================================
# Re-export all public API from submodules
# This allows: from losion.core.kernel import PagedKVCacheManager
# ============================================================================

# --- sdpa_compat ---
try:
    from losion.core.kernel.sdpa_compat import sdpa_attention, SDPACompat
except ImportError:
    pass

# --- flash_attn ---
try:
    from losion.core.kernel.flash_attn import FlashAttentionWrapper, RingAttention
except ImportError:
    pass

# --- ssm_kernels ---
try:
    from losion.core.kernel.ssm_kernels import (
        associative_scan,
        chunk_parallel_scan,
        rwkv7_parallel_wkv,
        multi_mode_ssm_scan,
    )
except ImportError:
    pass

# --- early_exit ---
try:
    from losion.core.kernel.early_exit import PathwayEarlyExit
except ImportError:
    pass

# --- kv_cache ---
try:
    from losion.core.kernel.kv_cache import PagedKVCacheManager, INT4KVCacheQuantizer, KVEvictionManager
except ImportError:
    pass

# --- training_optim ---
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

# --- fp8_utils ---
try:
    from losion.core.kernel.fp8_utils import FP8TrainingWrapper, has_fp8_support
except ImportError:
    pass

# --- advanced_training (v1.6.0) ---
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

__all__ = [
    # Backend flags
    "HAS_FLASH_ATTN",
    "HAS_TRITON",
    "HAS_CUDA",
    # sdpa_compat
    "sdpa_attention",
    "SDPACompat",
    # flash_attn
    "FlashAttentionWrapper",
    "RingAttention",
    # ssm_kernels
    "associative_scan",
    "chunk_parallel_scan",
    "rwkv7_parallel_wkv",
    "multi_mode_ssm_scan",
    # early_exit
    "PathwayEarlyExit",
    # kv_cache
    "PagedKVCacheManager",
    "INT4KVCacheQuantizer",
    "KVEvictionManager",
    # training_optim
    "MemoryEfficientTrainer",
    "CUDAGraphOptimizer",
    "GradientCompressor",
    "FusedAdamW",
    "CPUOffloadOptimizer",
    # fp8_utils
    "FP8TrainingWrapper",
    "has_fp8_support",
    # advanced_training (v1.6.0)
    "ActivationOffloader",
    "MemoryAwareBatchScheduler",
    "EightBitOptimizer",
    "LoRAAdapter",
    "LoRALayer",
    "ProgressiveSequenceScheduler",
    "CommComputeOverlap",
    "SelectiveGradientCheckpointing",
    "DynamicLossScaler",
]
