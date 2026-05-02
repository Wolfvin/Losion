"""
Losion Utils — Modul utilitas untuk framework Losion.

Berisi fungsi utilitas untuk:
- Deteksi hardware dan kompatibilitas (CUDA, ROCm, CPU)
- Setup logging untuk training
- Estimasi memori dan resource

Penggunaan:
    >>> from losion.utils import detect_device, get_gpu_info, setup_logging
    >>> device = detect_device()
    >>> gpu_info = get_gpu_info()
    >>> logger = setup_logging("losion-training")
"""

from __future__ import annotations

from losion.utils.hardware import (
    check_cuda_available,
    check_rocm_available,
    detect_device,
    estimate_vram_needed,
    get_gpu_info,
    recommend_settings,
)
from losion.utils.logging import setup_logging

__all__ = [
    # Hardware
    "detect_device",
    "get_gpu_info",
    "check_cuda_available",
    "check_rocm_available",
    "recommend_settings",
    "estimate_vram_needed",
    # Logging
    "setup_logging",
]
