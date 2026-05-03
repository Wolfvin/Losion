"""
Losion Kernel Optimization Module — v1.4.0

This module provides high-performance kernel implementations for all three
Losion pathways (SSM, Attention, MoE) plus cross-cutting optimizations.

Architecture:
  sdpa_compat       — Unified SDPA/Flash Attention interface with auto-detection
  ssm_kernels       — Parallel associative scan + Triton SSM kernels
  flash_attn        — FlashAttention 2/3/4 integration layer
  triton_kernels    — Custom Triton kernels for blend/route/attention ops
  compile_utils     — torch.compile custom FX graph optimization passes
  fp8_native        — Native FP8 training (torchao + Transformer Engine)
  speculative       — Speculative decoding with SSM-as-drafter
  early_exit        — Early exit / dynamic depth via router confidence
  paged_kv          — PagedAttention + E8 lattice VQ KV cache compression
  parallel_pathway  — Parallel pathway execution (Nemotron-3 style)
  expert_prefetch   — Expert prefetch via routing prediction
  fsdp_utils        — FSDP2 + FP8 combined training pipeline

Credits:
  - FlashAttention-2/3: Dao et al. (arXiv:2205.14135, arXiv:2407.08608)
  - FlashAttention-4: (arXiv:2603.05451)
  - Mamba-2 SSD: Gu & Dao (arXiv:2405.21075)
  - PyTorch Mamba2 Kernel Fusion: pytorch.org/blog/accelerating-mamba2-with-kernel-fusion
  - Mamba-3: (arXiv:2603.15569)
  - FlashMoE: (arXiv:2506.04667)
  - DeepSeek-V3 FP8: (arXiv:2412.19437)
  - NVIDIA Transformer Engine: github.com/NVIDIA/TransformerEngine
  - PagedAttention: vLLM (Kwon et al., SOSP 2023)
  - PagedEviction: (arXiv:2509.04377)
  - E8 Lattice VQ: vLLM Issue #39241
  - Triton Anatomy: (arXiv:2511.11581)
  - Warp Specialization in Triton: PyTorch Blog 2025
  - torch.compile FX passes: (blog.ezyang.com/2024/11/ways-to-use-torch-compile)
  - Ring Attention / Striped Attention: (DistFlashAttn, LoongTrain)
  - SpecForge: (arXiv:2603.18567)
  - SwiftSpec: (dl.acm.org/doi/10.1145/3779212.3790246)
  - KTransformers: (dl.acm.org/doi/10.1145/3731569.3764843, SOSP 2025)
  - AutoKernel: (arXiv:2603.21331)
  - Nemotron 3: (arXiv:2604.12374)
  - Routing Mamba: NeurIPS 2025
  - Occult: ICML 2025
  - INT4 Decoding GQA: PyTorch Blog
  - FSDP2+FP8: pytorch.org/blog/training-using-float8-fsdp2
  - SimpleFSDP: (ResearchGate 385510534)
  - Early Exit Survey: (arXiv:2501.07670)
  - Losion Framework: Wolfvin (github.com/Wolfvin/Losion)
"""

from __future__ import annotations

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ============================================================================
# Auto-Detection of Available Accelerators
# ============================================================================

def _detect_cuda() -> bool:
    """Detect CUDA availability."""
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

def _detect_rocm() -> bool:
    """Detect AMD ROCm availability."""
    try:
        import torch
        if torch.cuda.is_available():
            return hasattr(torch.version, 'hip') and torch.version.hip is not None
    except Exception:
        pass
    return False

def _detect_triton() -> bool:
    """Detect Triton availability."""
    try:
        import triton
        return True
    except ImportError:
        return False

def _detect_flash_attn() -> str:
    """Detect Flash Attention version available.

    Returns:
        "fa3" if FlashAttention-3 (Hopper) available,
        "fa2" if flash_attn package available,
        "sdpa" if PyTorch SDPA Flash backend available,
        "none" otherwise.
    """
    try:
        from flash_attn import flash_attn_func
        # Check for FA3 features (warp-specialization on Hopper)
        try:
            from flash_attn import flash_attn_with_kvcache
            return "fa3"
        except ImportError:
            return "fa2"
    except ImportError:
        pass

    try:
        from flash_attn_rocm import flash_attn_func
        return "fa2"
    except ImportError:
        pass

    try:
        import torch
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            if torch.cuda.is_available():
                cap = torch.cuda.get_device_capability()
                if cap[0] >= 8:
                    return "sdpa"
    except Exception:
        pass

    return "none"

def _detect_transformer_engine() -> bool:
    """Detect NVIDIA Transformer Engine availability."""
    try:
        import transformer_engine
        return True
    except ImportError:
        return False

def _detect_torchao() -> bool:
    """Detect torchao availability."""
    try:
        import torchao
        return True
    except ImportError:
        return False

def _detect_fp8_hardware() -> bool:
    """Detect FP8-capable hardware (H100+, MI300X+)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        cap = torch.cuda.get_device_capability()
        # H100 = sm_90, B200 = sm_100
        if cap[0] >= 9:
            return True
    except Exception:
        pass
    return False


# Module-level detection flags (computed once at import time)
HAS_CUDA: bool = _detect_cuda()
HAS_ROCM: bool = _detect_rocm()
HAS_TRITON: bool = _detect_triton()
FLASH_ATTN_VERSION: str = _detect_flash_attn()
HAS_FLASH_ATTENTION: bool = FLASH_ATTN_VERSION != "none"
HAS_TRANSFORMER_ENGINE: bool = _detect_transformer_engine()
HAS_TORCHAO: bool = _detect_torchao()
HAS_FP8_HARDWARE: bool = _detect_fp8_hardware()

# Environment variable overrides
_FORCE_SDPA: bool = os.environ.get("LOSION_FORCE_SDPA", "0") == "1"
_DISABLE_FLASH: bool = os.environ.get("LOSION_DISABLE_FLASH", "0") == "1"
_DISABLE_COMPILE: bool = os.environ.get("LOSION_DISABLE_COMPILE", "0") == "1"
_DISABLE_TRITON: bool = os.environ.get("LOSION_DISABLE_TRITON", "0") == "1"


def get_device_info() -> dict:
    """Get comprehensive device and kernel capability info.

    Returns:
        Dict with device capabilities and available optimizations.
    """
    info = {
        "cuda": HAS_CUDA,
        "rocm": HAS_ROCM,
        "triton": HAS_TRITON and not _DISABLE_TRITON,
        "flash_attn": FLASH_ATTN_VERSION if not _DISABLE_FLASH else "disabled",
        "transformer_engine": HAS_TRANSFORMER_ENGINE,
        "torchao": HAS_TORCHAO,
        "fp8_hardware": HAS_FP8_HARDWARE,
        "sdpa_available": False,
        "gpu_name": None,
        "gpu_memory_gb": 0,
    }

    try:
        import torch
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_mem / 1e9
            info["sdpa_available"] = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
            info["compute_capability"] = torch.cuda.get_device_capability()
    except Exception:
        pass

    return info


def recommend_kernel_config(seq_len: int = 4096, batch_size: int = 1,
                            d_model: int = 512, n_layers: int = 12) -> dict:
    """Recommend optimal kernel configuration based on hardware and model size.

    Args:
        seq_len: Sequence length.
        batch_size: Batch size.
        d_model: Model hidden dimension.
        n_layers: Number of layers.

    Returns:
        Dict with recommended kernel settings.
    """
    config = {
        "attention_backend": "sdpa",
        "ssm_backend": "chunk_parallel",
        "use_flash_attention": False,
        "use_triton_ssm": False,
        "use_fp8": False,
        "use_torch_compile": not _DISABLE_COMPILE,
        "use_early_exit": True,
        "early_exit_threshold": 0.05,
        "use_speculative_decoding": True,
        "speculative_draft_tokens": 4,
        "use_paged_kv": False,
        "kv_cache_dtype": "bf16",
        "use_parallel_pathway": False,
        "chunk_size": 256,
    }

    # Flash Attention recommendation
    if HAS_FLASH_ATTENTION and not _DISABLE_FLASH:
        config["use_flash_attention"] = True
        config["attention_backend"] = FLASH_ATTN_VERSION

    # SDPA for all cases (fallback or primary)
    if FLASH_ATTN_VERSION == "none" or _DISABLE_FLASH:
        config["attention_backend"] = "sdpa"
        try:
            import torch
            if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                config["attention_backend"] = "sdpa"
        except Exception:
            config["attention_backend"] = "math"

    # Triton SSM kernels
    if HAS_TRITON and not _DISABLE_TRITON:
        config["use_triton_ssm"] = True
        config["ssm_backend"] = "triton_fused"

    # FP8 training
    if HAS_FP8_HARDWARE and (HAS_TORCHAO or HAS_TRANSFORMER_ENGINE):
        config["use_fp8"] = True

    # Paged KV cache for inference with long sequences
    if seq_len > 2048:
        config["use_paged_kv"] = True

    # INT4 KV cache for very long sequences
    if seq_len > 8192:
        config["kv_cache_dtype"] = "int4"

    # Parallel pathway execution for large models on high-end GPUs
    try:
        import torch
        if torch.cuda.is_available():
            mem = torch.cuda.get_device_properties(0).total_mem
            if mem > 40e9:  # > 40GB VRAM
                config["use_parallel_pathway"] = True
    except Exception:
        pass

    return config


# ============================================================================
# Lazy Imports — only import sub-modules when actually used
# ============================================================================

def get_sdpa_compat():
    """Get SDPA compatibility module."""
    from losion.core.kernel import sdpa_compat
    return sdpa_compat

def get_ssm_kernels():
    """Get SSM kernel module."""
    from losion.core.kernel import ssm_kernels
    return ssm_kernels

def get_flash_attn():
    """Get Flash Attention integration module."""
    from losion.core.kernel import flash_attn as fa_module
    return fa_module

def get_triton_kernels():
    """Get Triton kernels module."""
    from losion.core.kernel import triton_kernels
    return triton_kernels

def get_compile_utils():
    """Get torch.compile utilities module."""
    from losion.core.kernel import compile_utils
    return compile_utils

def get_fp8_native():
    """Get native FP8 training module."""
    from losion.core.kernel import fp8_native
    return fp8_native

def get_speculative():
    """Get speculative decoding module."""
    from losion.core.kernel import speculative
    return speculative

def get_early_exit():
    """Get early exit module."""
    from losion.core.kernel import early_exit
    return early_exit

def get_paged_kv():
    """Get PagedAttention + KV cache compression module."""
    from losion.core.kernel import paged_kv
    return paged_kv

def get_parallel_pathway():
    """Get parallel pathway execution module."""
    from losion.core.kernel import parallel_pathway
    return parallel_pathway

def get_expert_prefetch():
    """Get expert prefetch module."""
    from losion.core.kernel import expert_prefetch
    return expert_prefetch

def get_fsdp_utils():
    """Get FSDP utilities module."""
    from losion.core.kernel import fsdp_utils
    return fsdp_utils


__all__ = [
    # Detection flags
    "HAS_CUDA",
    "HAS_ROCM",
    "HAS_TRITON",
    "FLASH_ATTN_VERSION",
    "HAS_FLASH_ATTENTION",
    "HAS_TRANSFORMER_ENGINE",
    "HAS_TORCHAO",
    "HAS_FP8_HARDWARE",
    # Environment overrides
    "_FORCE_SDPA",
    "_DISABLE_FLASH",
    "_DISABLE_COMPILE",
    "_DISABLE_TRITON",
    # Utility functions
    "get_device_info",
    "recommend_kernel_config",
    # Lazy importers
    "get_sdpa_compat",
    "get_ssm_kernels",
    "get_flash_attn",
    "get_triton_kernels",
    "get_compile_utils",
    "get_fp8_native",
    "get_speculative",
    "get_early_exit",
    "get_paged_kv",
    "get_parallel_pathway",
    "get_expert_prefetch",
    "get_fsdp_utils",
]
