"""
Losion FP8 Training — Native FP8 support via torchao.

Provides FP8 training conversion for Losion models using the torchao
library from the PyTorch team. Supports both NVIDIA (H100+) and
AMD (MI300X) hardware with proper fallback to BF16.

Credits:
  - torchao: PyTorch Architecture Optimization team
  - FP8 training: Micikevicius et al., "FP8 Formats for Deep Learning" (2022)
  - AMD FP8: CDNA3 native FP8 on MI300X

Hardware: CUDA (H100+), ROCm (MI300X+), fallback BF16 elsewhere.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def convert_to_fp8_training(
    model: nn.Module,
    enabled: bool = True,
) -> nn.Module:
    """Convert a Losion model to FP8 training mode.
    
    Uses torchao's convert_to_float8_training which handles:
    - FP8 GEMM operations (where hardware supports it)
    - Dynamic scaling for stability
    - Proper gradient flow through FP8 layers
    
    Falls back gracefully to BF16 if torchao is unavailable or
    hardware doesn't support FP8.
    
    Args:
        model: Losion model to convert.
        enabled: Whether to enable FP8 conversion. If False, returns
            the model unchanged.
    
    Returns:
        Model with FP8 training enabled (or unchanged if unavailable).
    """
    if not enabled:
        return model
    
    try:
        from torchao.float8 import convert_to_float8_training
        
        model = convert_to_float8_training(model)
        logger.info("Model converted to FP8 training mode via torchao")
        return model
        
    except ImportError:
        logger.warning(
            "torchao not available. FP8 training requires: pip install torchao. "
            "Falling back to BF16."
        )
        return model
    except Exception as e:
        logger.warning(f"FP8 conversion failed: {e}. Falling back to BF16.")
        return model


def check_fp8_support() -> dict:
    """Check if the current hardware and software support FP8.
    
    Returns:
        Dictionary with support information:
        - fp8_supported: Whether FP8 is available
        - torchao_available: Whether torchao is installed
        - hardware_support: Whether GPU supports FP8
        - device_name: GPU name (if available)
        - recommendation: Recommended precision setting
    """
    result = {
        "fp8_supported": False,
        "torchao_available": False,
        "hardware_support": False,
        "device_name": None,
        "recommendation": "bf16",
    }
    
    # Check torchao
    try:
        import torchao
        result["torchao_available"] = True
    except ImportError:
        result["recommendation"] = "bf16 (install torchao for FP8)"
        return result
    
    # Check hardware
    if not torch.cuda.is_available():
        result["recommendation"] = "fp32 (no GPU)"
        return result
    
    device_name = torch.cuda.get_device_name(0)
    result["device_name"] = device_name
    
    # Check for FP8-capable hardware
    # NVIDIA: H100 (compute capability 9.0+)
    # AMD: MI300X (CDNA3)
    try:
        props = torch.cuda.get_device_properties(0)
        compute_cap = f"{props.major}.{props.minor}"
        
        # H100+ (compute capability >= 9.0)
        if props.major >= 9:
            result["hardware_support"] = True
            result["fp8_supported"] = True
            result["recommendation"] = "fp8"
        # Check for AMD MI300X
        elif any(kw in device_name.lower() for kw in ["mi300", "instinct"]):
            result["hardware_support"] = True
            result["fp8_supported"] = True
            result["recommendation"] = "fp8"
        else:
            result["recommendation"] = "bf16 (GPU doesn't support FP8)"
    except Exception:
        result["recommendation"] = "bf16"
    
    return result


def get_optimal_precision() -> str:
    """Get the optimal precision setting for the current hardware.
    
    Returns:
        "fp8", "bf16", or "fp32" based on hardware capabilities.
    """
    support = check_fp8_support()
    return support["recommendation"].split()[0]  # Just the precision part
