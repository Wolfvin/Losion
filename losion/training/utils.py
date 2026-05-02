"""
Training Utilities untuk Losion Framework
==========================================

Fungsi utilitas untuk training pipeline:
- Learning rate schedulers (cosine, linear)
- Parameter counting dan memory estimation
- Distributed training setup
- Checkpoint save/load

Hardware: Pure PyTorch, kompatibel dengan CUDA, ROCm, dan CPU.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from losion.config import LosionConfig

logger = logging.getLogger(__name__)


# ============================================================================
# Learning Rate Schedulers
# ============================================================================


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Cosine schedule dengan linear warmup.

    Schedule:
    1. Warmup: Linear increase dari 0 ke max_lr selama num_warmup_steps
    2. Cosine decay: Decay dari max_lr ke min_lr selama sisa langkah

    Formula (setelah warmup):
        lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(π * progress))

    Args:
        optimizer: Optimizer yang akan di-schedule
        num_warmup_steps: Jumlah langkah warmup
        num_training_steps: Total langkah training
        min_lr_ratio: Rasio LR minimum terhadap LR awal (default 0.1)

    Returns:
        LambdaLR scheduler
    """

    def lr_lambda(current_step: int) -> float:
        # Warmup phase
        if current_step < num_warmup_steps:
            if num_warmup_steps > 0:
                return float(current_step) / float(max(1, num_warmup_steps))
            return 1.0

        # Cosine decay phase
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_linear_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Linear schedule dengan linear warmup.

    Schedule:
    1. Warmup: Linear increase dari 0 ke max_lr selama num_warmup_steps
    2. Linear decay: Decay dari max_lr ke min_lr selama sisa langkah

    Args:
        optimizer: Optimizer yang akan di-schedule
        num_warmup_steps: Jumlah langkah warmup
        num_training_steps: Total langkah training
        min_lr_ratio: Rasio LR minimum terhadap LR awal (default 0.0)

    Returns:
        LambdaLR scheduler
    """

    def lr_lambda(current_step: int) -> float:
        # Warmup phase
        if current_step < num_warmup_steps:
            if num_warmup_steps > 0:
                return float(current_step) / float(max(1, num_warmup_steps))
            return 1.0

        # Linear decay phase
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(min_lr_ratio, 1.0 - progress * (1.0 - min_lr_ratio))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================================================================
# Parameter Counting & Memory Estimation
# ============================================================================


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Hitung jumlah parameter model, dipisah per kategori.

    Args:
        model: Model PyTorch

    Returns:
        Dictionary berisi jumlah parameter per kategori:
        - total: Total parameter
        - trainable: Parameter yang bisa dilatih
        - frozen: Parameter yang di-freeze
        - embeddings: Parameter embedding
        - attention: Parameter attention
        - ffn: Parameter feed-forward
        - other: Parameter lainnya
    """
    total = 0
    trainable = 0
    frozen = 0
    embeddings = 0
    attention = 0
    ffn = 0
    other = 0

    for name, param in model.named_parameters():
        num_params = param.numel()
        total += num_params

        if param.requires_grad:
            trainable += num_params
        else:
            frozen += num_params

        # Kategorikan berdasarkan nama
        name_lower = name.lower()
        if "embed" in name_lower:
            embeddings += num_params
        elif "attn" in name_lower or "attention" in name_lower or "mla" in name_lower:
            attention += num_params
        elif "ffn" in name_lower or "feed_forward" in name_lower or "swiglu" in name_lower:
            ffn += num_params
        else:
            other += num_params

    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "embeddings": embeddings,
        "attention": attention,
        "ffn": ffn,
        "other": other,
    }


def estimate_training_memory(config: LosionConfig) -> int:
    """
    Estimasi memori (dalam bytes) yang diperlukan untuk training.

    Menghitung:
    - Parameter model (4 bytes per parameter untuk FP32, 2 untuk BF16)
    - Gradien (sama dengan parameter)
    - Optimizer state (8 bytes per parameter untuk AdamW: momentum + variance)
    - Activations (perkiraan kasar berdasarkan batch size dan seq_len)

    Args:
        config: LosionConfig

    Returns:
        Estimasi memori dalam bytes
    """
    # Estimasi jumlah parameter
    num_params = config.estimated_parameters()

    # Bytes per parameter
    if config.hardware.precision.value == "fp8":
        bytes_per_param = 1
    elif config.hardware.precision.value == "bf16":
        bytes_per_param = 2
    else:
        bytes_per_param = 4  # FP32

    # Model parameters
    model_memory = num_params * bytes_per_param

    # Gradients (same size as parameters)
    gradient_memory = num_params * bytes_per_param

    # Optimizer state (AdamW: momentum + variance, each 4 bytes)
    optimizer_memory = num_params * 8  # 2 * 4 bytes per param

    # Activations (rough estimate)
    # Per layer: batch * seq_len * d_model * 4 (forward activations)
    # Plus attention: batch * n_heads * seq_len * seq_len
    batch_size = config.training.batch_size
    seq_len = config.max_seq_len
    d_model = config.d_model
    n_layers = config.n_layers

    activation_per_layer = batch_size * seq_len * d_model * 4  # 4x for intermediate
    attention_per_layer = batch_size * config.attention.n_heads * seq_len * seq_len
    total_activations = n_layers * (activation_per_layer + attention_per_layer) * 4  # FP32

    # Total (dengan safety margin 20%)
    total_memory = int(1.2 * (model_memory + gradient_memory + optimizer_memory + total_activations))

    return total_memory


def estimate_vram_needed(config: LosionConfig) -> int:
    """
    Alias untuk estimate_training_memory — estimasi VRAM yang diperlukan.

    Args:
        config: LosionConfig

    Returns:
        Estimasi VRAM dalam bytes
    """
    return estimate_training_memory(config)


# ============================================================================
# Distributed Training Setup
# ============================================================================


def setup_distributed() -> Tuple[int, int, bool]:
    """
    Setup distributed training environment.

    Mendeteksi dan menginisialisasi distributed training berdasarkan
    environment variables (SLURM, torchrun, etc).

    Returns:
        Tuple (local_rank, world_size, is_distributed):
        - local_rank: Rank lokal (0 jika tidak distributed)
        - world_size: Jumlah proses total (1 jika tidak distributed)
        - is_distributed: Apakah training distributed
    """
    # Cek environment variables untuk distributed setup
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))

    is_distributed = world_size > 1

    if is_distributed:
        try:
            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group(
                    backend="nccl" if torch.cuda.is_available() else "gloo",
                    init_method="env://",
                    world_size=world_size,
                    rank=rank,
                )

                if torch.cuda.is_available():
                    torch.cuda.set_device(local_rank)

                logger.info(
                    f"Distributed training initialized: "
                    f"rank={rank}, local_rank={local_rank}, world_size={world_size}"
                )
        except Exception as e:
            logger.warning(f"Gagal inisialisasi distributed training: {e}")
            is_distributed = False
            local_rank = 0
            world_size = 1

    return local_rank, world_size, is_distributed


def cleanup_distributed() -> None:
    """
    Cleanup distributed training resources.

    Dipanggil setelah training selesai.
    """
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


# ============================================================================
# Checkpoint Save/Load
# ============================================================================


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[torch.optim.lr_scheduler.LambdaLR],
    global_step: int,
    save_dir: str,
    max_checkpoints: int = 5,
) -> None:
    """
    Simpan checkpoint training.

    Menyimpan:
    - model_state_dict: State dict model
    - optimizer_state_dict: State dict optimizer (jika ada)
    - scheduler_state_dict: State dict scheduler (jika ada)
    - global_step: Langkah training saat ini

    Args:
        model: Model yang akan disimpan
        optimizer: Optimizer yang akan disimpan (opsional)
        scheduler: Scheduler yang akan disimpan (opsional)
        global_step: Langkah training saat ini
        save_dir: Direktori penyimpanan
        max_checkpoints: Maksimum checkpoint yang disimpan (rotasi)
    """
    os.makedirs(save_dir, exist_ok=True)

    # Ambil state dict (handle DDP/FSDP wrapper)
    if hasattr(model, "module"):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()

    checkpoint = {
        "model_state_dict": model_state_dict,
        "global_step": global_step,
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    # Simpan checkpoint
    checkpoint_path = os.path.join(save_dir, "checkpoint.pt")
    torch.save(checkpoint, checkpoint_path)

    # Rotasi checkpoint (hapus yang lama jika melebihi max)
    _rotate_checkpoints(os.path.dirname(save_dir), max_checkpoints)


def load_checkpoint(
    checkpoint_path: str,
    device: Optional[torch.device] = None,
) -> Optional[Dict]:
    """
    Load checkpoint training.

    Args:
        checkpoint_path: Path ke direktori atau file checkpoint
        device: Device untuk load checkpoint (opsional)

    Returns:
        Dictionary berisi checkpoint data, atau None jika gagal
    """
    # Jika path adalah direktori, cari checkpoint.pt
    if os.path.isdir(checkpoint_path):
        checkpoint_file = os.path.join(checkpoint_path, "checkpoint.pt")
    else:
        checkpoint_file = checkpoint_path

    if not os.path.exists(checkpoint_file):
        logger.warning(f"Checkpoint tidak ditemukan: {checkpoint_file}")
        return None

    try:
        map_location = device if device is not None else "cpu"
        checkpoint = torch.load(checkpoint_file, map_location=map_location, weights_only=True)
        logger.info(f"Checkpoint berhasil di-load: {checkpoint_file}")
        return checkpoint
    except Exception as e:
        logger.error(f"Gagal load checkpoint: {e}")
        return None


def _rotate_checkpoints(checkpoint_dir: str, max_checkpoints: int) -> None:
    """
    Rotasi checkpoint — hapus checkpoint lama jika melebihi batas.

    Args:
        checkpoint_dir: Direktori checkpoint
        max_checkpoints: Maksimum checkpoint yang disimpan
    """
    if max_checkpoints <= 0:
        return

    try:
        # Cari semua subdirektori checkpoint
        entries = os.listdir(checkpoint_dir)
        checkpoint_dirs = [
            os.path.join(checkpoint_dir, e)
            for e in entries
            if e.startswith("checkpoint-") and os.path.isdir(os.path.join(checkpoint_dir, e))
        ]

        # Sort berdasarkan modification time
        checkpoint_dirs.sort(key=lambda x: os.path.getmtime(x))

        # Hapus checkpoint tertua jika melebihi batas
        while len(checkpoint_dirs) > max_checkpoints:
            oldest = checkpoint_dirs.pop(0)
            try:
                import shutil
                shutil.rmtree(oldest)
                logger.info(f"Checkpoint lama dihapus: {oldest}")
            except Exception as e:
                logger.warning(f"Gagal hapus checkpoint {oldest}: {e}")
    except Exception as e:
        logger.warning(f"Gagal rotasi checkpoint: {e}")


# ============================================================================
# Gradient Utilities
# ============================================================================


def get_gradient_norm(model: nn.Module, norm_type: float = 2.0) -> float:
    """
    Hitung norma gradien model.

    Berguna untuk monitoring dan gradient clipping.

    Args:
        model: Model PyTorch
        norm_type: Tipe norma (default 2.0 = L2 norm)

    Returns:
        Norma gradien
    """
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type

    return total_norm ** (1.0 / norm_type)


def clip_gradients(
    model: nn.Module,
    max_norm: float = 1.0,
    norm_type: float = 2.0,
) -> float:
    """
    Clip gradien model.

    Args:
        model: Model PyTorch
        max_norm: Norma maksimum
        norm_type: Tipe norma

    Returns:
        Norma gradien sebelum clipping
    """
    grad_norm = get_gradient_norm(model, norm_type)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm, norm_type)
    return grad_norm


# ============================================================================
# Mixed Precision Utilities
# ============================================================================


def get_autocast_dtype(precision: str) -> torch.dtype:
    """
    Ambil dtype untuk autocast berdasarkan konfigurasi presisi.

    Args:
        precision: String presisi ("fp32", "bf16", "fp8")

    Returns:
        torch.dtype yang sesuai
    """
    if precision == "bf16":
        return torch.bfloat16
    elif precision == "fp8":
        return torch.float8_e4m3fn  # FP8 E4M3 (tersedia di PyTorch 2.1+)
    else:
        return torch.float32


def prepare_model_for_training(
    model: nn.Module,
    precision: str = "bf16",
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    Persiapkan model untuk training.

    Melakukan:
    - Konversi ke dtype yang sesuai
    - Pindah ke device yang sesuai
    - Set model ke mode training

    Args:
        model: Model PyTorch
        precision: Presisi target
        device: Device target

    Returns:
        Model yang sudah disiapkan
    """
    dtype = get_autocast_dtype(precision)

    if device is not None:
        model = model.to(device)

    if precision != "fp32":
        model = model.to(dtype=dtype)

    model.train()

    return model
