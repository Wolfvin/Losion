"""
Losion Logging — Setup Logging untuk Training
==============================================

Modul ini menyediakan fungsi untuk mengatur logging
pada framework Losion, termasuk:
- Console logging dengan format yang konsisten
- File logging untuk training logs
- Integrasi dengan tensorboard dan wandb (opsional)

Hardware: Pure Python, kompatibel dengan semua platform.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional


# ============================================================================
# Logging Format
# ============================================================================


# Format default untuk Losion logging
DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# ============================================================================
# Setup Functions
# ============================================================================


def setup_logging(
    name: str = "losion",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    log_dir: Optional[str] = None,
    format_string: Optional[str] = None,
    console: bool = True,
) -> logging.Logger:
    """
    Setup logging untuk Losion framework.

    Membuat logger dengan handler yang konsisten untuk
    console output dan optional file output.

    Args:
        name: Nama logger (default "losion")
        level: Level logging (default logging.INFO)
        log_file: Path ke file log (opsional, override log_dir)
        log_dir: Direktori untuk file log (opsional)
        format_string: Format string kustom (opsional)
        console: Apakah output ke console (default True)

    Returns:
        Logger yang sudah dikonfigurasi

    Contoh:
        >>> logger = setup_logging("losion-training", log_dir="./logs")
        >>> logger.info("Training dimulai")
        >>> logger.warning("VRAM rendah")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Hindari duplikasi handler
    if logger.handlers:
        return logger

    # Format
    fmt = format_string or DEFAULT_FORMAT
    formatter = logging.Formatter(fmt, datefmt=DEFAULT_DATE_FORMAT)

    # ---- Console handler ----
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # ---- File handler ----
    if log_file is not None or log_dir is not None:
        if log_file is None and log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, f"{name}.log")

        if log_file is not None:
            os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else ".", exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


def setup_training_logging(
    experiment_name: str = "losion-experiment",
    output_dir: str = "./outputs",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Setup logging khusus untuk training.

    Membuat logger dengan:
    - Console output (INFO level)
    - File output (DEBUG level, lebih detail)
    - Struktur direktori yang rapi

    Args:
        experiment_name: Nama eksperimen
        output_dir: Direktori output
        level: Level console logging

    Returns:
        Logger yang sudah dikonfigurasi
    """
    log_dir = os.path.join(output_dir, experiment_name, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Main logger
    logger = setup_logging(
        name=f"losion.{experiment_name}",
        level=level,
        log_dir=log_dir,
        console=True,
    )

    # Detailed file logger (DEBUG level)
    detailed_logger = logging.getLogger(f"losion.{experiment_name}.detailed")
    detailed_logger.setLevel(logging.DEBUG)

    # File handler untuk detailed log
    detailed_file = os.path.join(log_dir, "detailed.log")
    file_handler = logging.FileHandler(detailed_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(DEFAULT_FORMAT, datefmt=DEFAULT_DATE_FORMAT)
    )
    detailed_logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Ambil logger dengan nama tertentu.

    Jika logger belum ada, buat baru dengan konfigurasi default.

    Args:
        name: Nama logger

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)

    # Jika belum ada handler, tambahkan console handler
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        handler.setFormatter(
            logging.Formatter(DEFAULT_FORMAT, datefmt=DEFAULT_DATE_FORMAT)
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger


class TrainingLogger:
    """
    Logger khusus untuk training metrics.

    Menyediakan metode yang nyaman untuk logging
    training metrics dengan format yang konsisten.

    Args:
        name: Nama logger
        output_dir: Direktori output
        use_tensorboard: Aktifkan TensorBoard logging
        use_wandb: Aktifkan Wandb logging
    """

    def __init__(
        self,
        name: str = "losion-training",
        output_dir: str = "./outputs",
        use_tensorboard: bool = False,
        use_wandb: bool = False,
    ) -> None:
        self.name = name
        self.output_dir = output_dir
        self.logger = setup_training_logging(name, output_dir)

        # Optional integrations
        self.tb_writer = None
        self.wandb_run = None

        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_dir = os.path.join(output_dir, name, "tensorboard")
                self.tb_writer = SummaryWriter(tb_dir)
                self.logger.info(f"TensorBoard logging aktif: {tb_dir}")
            except ImportError:
                self.logger.warning(
                    "torch.utils.tensorboard tidak tersedia. "
                    "Install dengan: pip install tensorboard"
                )

        if use_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(project=name)
                self.logger.info(f"Wandb logging aktif: {wandb.run.name if wandb.run else 'unknown'}")
            except ImportError:
                self.logger.warning(
                    "wandb tidak tersedia. Install dengan: pip install wandb"
                )

    def log_metrics(
        self,
        metrics: dict,
        step: int,
        prefix: str = "",
    ) -> None:
        """
        Log metrics ke semua backend yang aktif.

        Args:
            metrics: Dictionary berisi metrics
            step: Langkah training saat ini
            prefix: Prefix untuk nama metric (opsional)
        """
        # Console logging
        metric_str = " | ".join(
            f"{prefix}{k}: {v:.4f}" if isinstance(v, float) else f"{prefix}{k}: {v}"
            for k, v in metrics.items()
        )
        self.logger.info(f"Step {step}: {metric_str}")

        # TensorBoard
        if self.tb_writer is not None:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self.tb_writer.add_scalar(f"{prefix}{k}", v, step)

        # Wandb
        if self.wandb_run is not None:
            try:
                import wandb
                wandb.log({f"{prefix}{k}": v for k, v in metrics.items()}, step=step)
            except Exception as e:
                self.logger.warning(f"Gagal log ke wandb: {e}")

    def log_config(self, config_dict: dict) -> None:
        """
        Log konfigurasi training.

        Args:
            config_dict: Dictionary berisi konfigurasi
        """
        self.logger.info("=" * 60)
        self.logger.info("Training Configuration")
        self.logger.info("=" * 60)
        for k, v in config_dict.items():
            self.logger.info(f"  {k}: {v}")
        self.logger.info("=" * 60)

        # Wandb config
        if self.wandb_run is not None:
            try:
                import wandb
                wandb.config.update(config_dict)
            except Exception:
                pass

    def log_phase_transition(
        self,
        from_phase: str,
        to_phase: str,
        step: int,
        reason: str,
    ) -> None:
        """
        Log transisi fase training.

        Args:
            from_phase: Fase sebelumnya
            to_phase: Fase selanjutnya
            step: Langkah saat transisi
            reason: Alasan transisi
        """
        self.logger.info(
            f"🔄 Fase Transition: {from_phase} → {to_phase} "
            f"(step={step}, reason={reason})"
        )

        self.log_metrics(
            {"phase_transition": 1.0},
            step=step,
            prefix="phase/",
        )

    def close(self) -> None:
        """Tutup semua handler logging."""
        if self.tb_writer is not None:
            self.tb_writer.close()

        if self.wandb_run is not None:
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass
