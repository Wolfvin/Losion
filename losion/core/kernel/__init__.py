"""
Losion Kernel Optimizations — Low-level kernels for training and inference.

Provides optimized implementations for:
  - SDPA / Flash Attention with multi-tier fallback
  - SSM parallel scans (associative, chunk, RWKV-7 WKV)
  - Early exit with adaptive thresholds
  - Paged KV Cache with INT4 quantization
  - Training memory optimizations (CPU offload, gradient compression)
  - FP8 training wrappers
  - CUDA Graph optimization
  - Fused optimizer kernels

Credits:
  - Flash Attention: Dao et al., arXiv:2205.14135 (2022)
  - SDPA: PyTorch native F.scaled_dot_product_attention (2023)
  - Triton: OpenAI Triton language (2023)
  - ZeRO-Offload: Rajbhandari et al., SC 2021
  - torchao: PyTorch Architecture Optimization (2024)
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

__all__ = [
    "HAS_FLASH_ATTN",
    "HAS_TRITON",
    "HAS_CUDA",
    # Submodules exported via losion.__init__
]
