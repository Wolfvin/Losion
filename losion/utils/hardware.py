"""
Losion Hardware — Deteksi Hardware dan Kompatibilitas
======================================================

Modul ini menyediakan fungsi untuk mendeteksi dan mengelola
perangkat keras yang tersedia (CUDA, ROCm, CPU) dan
merekomendasikan konfigurasi yang optimal.

Fungsi utama:
- detect_device(): Deteksi perangkat terbaik yang tersedia
- get_gpu_info(): Informasi detail GPU
- check_cuda_available(): Periksa ketersediaan CUDA
- check_rocm_available(): Periksa ketersediaan ROCm
- recommend_settings(): Rekomendasi konfigurasi berdasarkan hardware
- estimate_vram_needed(): Estimasi VRAM yang diperlukan

Hardware: Pure PyTorch, kompatibel dengan CUDA, ROCm, dan CPU.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from losion.config import LosionConfig

logger = logging.getLogger(__name__)


# ============================================================================
# Device Detection
# ============================================================================


def detect_device() -> str:
    """
    Deteksi perangkat komputasi terbaik yang tersedia.

    Prioritas:
    1. CUDA (NVIDIA GPU)
    2. ROCm (AMD GPU via HIP)
    3. CPU (fallback)

    Returns:
        String perangkat: 'cuda', 'rocm', atau 'cpu'
    """
    try:
        import torch

        if not torch.cuda.is_available():
            logger.info("Tidak ada GPU terdeteksi, menggunakan CPU")
            return "cpu"

        # Periksa apakah ini ROCm (AMD GPU)
        if check_rocm_available():
            logger.info("ROCm (AMD GPU) terdeteksi")
            return "rocm"

        # Default ke CUDA (NVIDIA GPU)
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"CUDA (NVIDIA GPU) terdeteksi: {device_name}")
        return "cuda"

    except ImportError:
        logger.warning("PyTorch tidak terinstal, menggunakan CPU")
        return "cpu"


def get_gpu_info() -> Dict[str, Any]:
    """
    Ambil informasi detail tentang GPU yang tersedia.

    Returns:
        Dictionary berisi informasi GPU:
        - available: Apakah GPU tersedia
        - device_count: Jumlah GPU
        - device_name: Nama GPU pertama
        - memory_total_gb: Total memori GPU (GB)
        - memory_free_gb: Memori GPU yang tersedia (GB)
        - compute_capability: Compute capability (CUDA)
        - cuda_version: Versi CUDA
        - is_rocm: Apakah ini ROCm
        - multi_gpu: Apakah ada lebih dari 1 GPU
    """
    info: Dict[str, Any] = {
        "available": False,
        "device_count": 0,
        "device_name": None,
        "memory_total_gb": None,
        "memory_free_gb": None,
        "compute_capability": None,
        "cuda_version": None,
        "is_rocm": False,
        "multi_gpu": False,
    }

    try:
        import torch

        info["available"] = torch.cuda.is_available()

        if not info["available"]:
            return info

        info["device_count"] = torch.cuda.device_count()
        info["multi_gpu"] = info["device_count"] > 1

        # Info GPU pertama
        info["device_name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        info["memory_total_gb"] = round(props.total_mem / (1024 ** 3), 2)

        # Memori yang tersedia
        if info["device_count"] > 0:
            free_mem = torch.cuda.mem_get_info(0)[0]  # (free, total)
            info["memory_free_gb"] = round(free_mem / (1024 ** 3), 2)

        # Compute capability
        if hasattr(props, "major") and hasattr(props, "minor"):
            info["compute_capability"] = f"{props.major}.{props.minor}"

        # CUDA version
        info["cuda_version"] = torch.version.cuda

        # ROCm detection
        info["is_rocm"] = check_rocm_available()

    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Error saat mendapatkan info GPU: {e}")

    return info


def check_cuda_available() -> bool:
    """
    Periksa apakah CUDA (NVIDIA) tersedia di sistem ini.

    Returns:
        True jika CUDA tersedia
    """
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def check_rocm_available() -> bool:
    """
    Periksa apakah ROCm (AMD) tersedia di sistem ini.

    Deteksi dilakukan dengan:
    1. Memeriksa nama device (AMD/Radeon/Instinct)
    2. Memeriksa environment variable HIP_PATH
    3. Memeriksa hiprtc availability

    Returns:
        True jika ROCm tersedia
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return False

        # Cek dari nama device
        try:
            device_name = torch.cuda.get_device_name(0).lower()
            if any(keyword in device_name for keyword in ["amd", "radeon", "instinct", "gfx"]):
                return True
        except Exception:
            pass

        # Cek environment variable
        if os.environ.get("HIP_PATH") or os.environ.get("ROCM_HOME"):
            return True

        # Cek dari PyTorch build info
        if hasattr(torch.version, "hip") and torch.version.hip is not None:
            return True

        return False

    except ImportError:
        return False


# ============================================================================
# Settings Recommendation
# ============================================================================


def recommend_settings(device: str, model_size: str = "base") -> Dict[str, Any]:
    """
    Rekomendasikan pengaturan training berdasarkan hardware dan ukuran model.

    Model sizes yang didukung:
    - "tiny": ~100M parameters (d_model=512, n_layers=8)
    - "small": ~500M parameters (d_model=1024, n_layers=16)
    - "base": ~2B parameters (d_model=2048, n_layers=32)
    - "large": ~7B parameters (d_model=4096, n_layers=48)
    - "xl": ~15B parameters (d_model=6144, n_layers=64)

    Args:
        device: Perangkat target ('cuda', 'rocm', 'cpu')
        model_size: Ukuran model ('tiny', 'small', 'base', 'large', 'xl')

    Returns:
        Dictionary berisi rekomendasi pengaturan:
        - batch_size: Ukuran batch yang direkomendasikan
        - gradient_accumulation_steps: Langkah akumulasi gradien
        - precision: Presisi yang direkomendasikan
        - compile_model: Apakah torch.compile direkomendasikan
        - num_workers: Jumlah dataloader workers
        - mixed_precision: Apakah mixed precision direkomendasikan
        - vram_needed_gb: Estimasi VRAM yang diperlukan
    """
    # Pengaturan default
    settings: Dict[str, Any] = {
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "precision": "fp32",
        "compile_model": False,
        "num_workers": 0,
        "mixed_precision": False,
        "vram_needed_gb": 0.0,
    }

    # GPU info (jika tersedia)
    gpu_info = get_gpu_info()
    gpu_memory_gb = gpu_info.get("memory_total_gb", 0.0) or 0.0
    compute_cap = gpu_info.get("compute_capability", "0.0")

    if device == "cpu":
        # CPU training — sangat lambat, hanya untuk testing
        settings.update({
            "batch_size": 1,
            "gradient_accumulation_steps": 32,
            "precision": "fp32",
            "compile_model": False,
            "num_workers": 4,
            "mixed_precision": False,
        })

        # Ukuran model untuk CPU
        if model_size in ("tiny", "small"):
            settings["batch_size"] = 2
            settings["gradient_accumulation_steps"] = 16
        else:
            logger.warning(
                f"Model '{model_size}' terlalu besar untuk CPU training. "
                "Gunakan GPU untuk training model yang lebih besar."
            )

    elif device in ("cuda", "rocm"):
        # GPU training
        # Estimasi VRAM berdasarkan ukuran model
        vram_estimates = {
            "tiny": 2.0,
            "small": 8.0,
            "base": 24.0,
            "large": 80.0,
            "xl": 160.0,
        }
        settings["vram_needed_gb"] = vram_estimates.get(model_size, 24.0)

        # Presisi berdasarkan compute capability
        try:
            major_version = int(compute_cap.split(".")[0])
        except (ValueError, IndexError, AttributeError):
            major_version = 0

        # BF16 tersedia di Ampere+ (compute capability >= 8.0)
        supports_bf16 = major_version >= 8

        # FP8 tersedia di Hopper+ (compute capability >= 9.0)
        supports_fp8 = major_version >= 9

        if supports_fp8 and model_size in ("large", "xl"):
            settings["precision"] = "fp8"
            settings["mixed_precision"] = True
        elif supports_bf16:
            settings["precision"] = "bf16"
            settings["mixed_precision"] = True
        else:
            settings["precision"] = "fp16"
            settings["mixed_precision"] = True

        # Batch size berdasarkan VRAM dan model size
        if gpu_memory_gb > 0:
            # Estimasi batch size: (VRAM - model_size) / per_sample_memory
            model_memory_gb = vram_estimates.get(model_size, 24.0)
            available_for_batch = gpu_memory_gb - model_memory_gb * 0.7  # 70% untuk model

            if available_for_batch > 0:
                # Perkiraan ~0.5 GB per sample untuk base model
                per_sample_gb = {
                    "tiny": 0.05,
                    "small": 0.1,
                    "base": 0.5,
                    "large": 2.0,
                    "xl": 4.0,
                }.get(model_size, 0.5)

                estimated_batch = max(1, int(available_for_batch / per_sample_gb))
                settings["batch_size"] = min(estimated_batch, 64)
            else:
                settings["batch_size"] = 1
                settings["gradient_accumulation_steps"] = max(1, int(8 / max(model_memory_gb / gpu_memory_gb, 0.1)))
        else:
            # Tidak ada info VRAM, gunakan default konservatif
            default_batch = {"tiny": 32, "small": 16, "base": 4, "large": 1, "xl": 1}
            settings["batch_size"] = default_batch.get(model_size, 4)

        # Gradient accumulation untuk effective batch size yang lebih besar
        effective_batch_target = {
            "tiny": 128,
            "small": 256,
            "base": 256,
            "large": 128,
            "xl": 64,
        }.get(model_size, 256)

        settings["gradient_accumulation_steps"] = max(
            1, effective_batch_target // settings["batch_size"]
        )

        # Compile model untuk GPU (PyTorch 2.0+)
        settings["compile_model"] = True

        # Dataloader workers
        settings["num_workers"] = min(8, os.cpu_count() or 4)

    return settings


def estimate_vram_needed(config: LosionConfig) -> int:
    """
    Estimasi VRAM (dalam bytes) yang diperlukan untuk training.

    Menghitung berdasarkan konfigurasi model:
    - Parameter model
    - Gradien
    - Optimizer state (AdamW)
    - Activations

    Args:
        config: LosionConfig

    Returns:
        Estimasi VRAM dalam bytes
    """
    # Estimasi jumlah parameter
    num_params = config.estimated_parameters()

    # Bytes per parameter berdasarkan presisi
    precision_bytes = {
        "fp32": 4,
        "bf16": 2,
        "fp8": 1,
    }
    bytes_per_param = precision_bytes.get(config.hardware.precision.value, 4)

    # Model parameters
    model_memory = num_params * bytes_per_param

    # Gradients (same as model)
    gradient_memory = num_params * bytes_per_param

    # Optimizer state (AdamW: 2 * FP32 per parameter)
    optimizer_memory = num_params * 8

    # Activations (rough estimate)
    batch_size = config.training.batch_size
    seq_len = config.max_seq_len
    d_model = config.d_model
    n_layers = config.n_layers

    # Per-layer activation memory
    activation_per_layer = batch_size * seq_len * d_model * 4 * 4  # 4x intermediate, 4 bytes
    total_activations = n_layers * activation_per_layer

    # Attention memory (worst case)
    attn_memory = n_layers * batch_size * config.attention.n_heads * seq_len * seq_len * 4

    # Total with 20% safety margin
    total = int(1.2 * (model_memory + gradient_memory + optimizer_memory + total_activations + attn_memory))

    return total


def get_device_capability(device_id: int = 0) -> Optional[str]:
    """
    Ambil compute capability GPU.

    Args:
        device_id: ID GPU (default 0)

    Returns:
        String compute capability (misalnya "8.0") atau None
    """
    try:
        import torch
        if torch.cuda.is_available() and device_id < torch.cuda.device_count():
            props = torch.cuda.get_device_properties(device_id)
            return f"{props.major}.{props.minor}"
    except Exception:
        pass
    return None


def print_hardware_summary() -> None:
    """
    Cetak ringkasan hardware yang tersedia.

    Berguna untuk debugging dan logging di awal training.
    """
    device = detect_device()
    gpu_info = get_gpu_info()

    print("=" * 60)
    print("Losion — Hardware Summary")
    print("=" * 60)
    print(f"  Device: {device}")
    print(f"  GPU Available: {gpu_info['available']}")

    if gpu_info["available"]:
        print(f"  GPU Name: {gpu_info['device_name']}")
        print(f"  GPU Count: {gpu_info['device_count']}")
        print(f"  GPU Memory: {gpu_info['memory_total_gb']} GB")
        if gpu_info["memory_free_gb"] is not None:
            print(f"  GPU Free Memory: {gpu_info['memory_free_gb']} GB")
        print(f"  Compute Capability: {gpu_info['compute_capability']}")
        print(f"  CUDA Version: {gpu_info['cuda_version']}")
        print(f"  Is ROCm: {gpu_info['is_rocm']}")
        print(f"  Multi-GPU: {gpu_info['multi_gpu']}")

    print("=" * 60)
