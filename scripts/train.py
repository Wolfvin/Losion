"""
Losion Training Script — entry point utama untuk training model.

Usage:
    python scripts/train.py --config configs/losion-1b.yaml
    python scripts/train.py --config configs/losion-7b.yaml --resume checkpoint.pt
    python scripts/train.py --config configs/losion-48b.yaml --phase 3

    # AMD ROCm:
    HIP_VISIBLE_DEVICES=0 python scripts/train.py --config configs/losion-1b.yaml

    # Multi-GPU:
    torchrun --nproc_per_node=4 scripts/train.py --config configs/losion-7b.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Tambahkan project root ke sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Losion Training Script — training model dengan arsitektur Tri-Jalur Router",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # === Konfigurasi ===
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path ke file YAML konfigurasi model (e.g., configs/losion-1b.yaml)",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Override konfigurasi (format: key=value, e.g., training.batch_size=16)",
    )

    # === Checkpoint ===
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path ke checkpoint untuk resume training",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Direktori output untuk checkpoint dan log (override config)",
    )

    # === Data ===
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Direktori dataset (format JSONL)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Path ke tokenizer file",
    )

    # === Training ===
    parser.add_argument(
        "--phase",
        type=int,
        default=None,
        choices=[1, 2, 3, 4],
        help="Mulai langsung dari fase tertentu (1-4)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Override jumlah maksimum langkah training",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Override learning rate",
    )

    # === Monitoring ===
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        default=False,
        help="Aktifkan Wandb logging",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="losion",
        help="Nama proyek Wandb",
    )

    # === Hardware ===
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="Gunakan FP16 mixed precision",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        default=True,
        help="Gunakan BF16 mixed precision (default)",
    )
    parser.add_argument(
        "--no_bf16",
        action="store_true",
        default=False,
        help="Nonaktifkan BF16 (gunakan FP32)",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        default=False,
        help="Gunakan torch.compile() untuk optimisasi",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed untuk reproducibility",
    )

    # === Debug ===
    parser.add_argument(
        "--dry_run",
        action="store_true",
        default=False,
        help="Jalankan 1 langkah saja untuk verifikasi",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Verbose logging",
    )

    return parser.parse_args()


def load_config(config_path: str) -> "LosionConfig":
    """Load konfigurasi dari file YAML.

    Args:
        config_path: Path ke file YAML konfigurasi.

    Returns:
        Instance LosionConfig.

    Raises:
        FileNotFoundError: Jika file konfigurasi tidak ditemukan.
    """
    from losion.config import LosionConfig

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"File konfigurasi tidak ditemukan: {config_path}")

    config = LosionConfig.from_yaml(config_path)
    return config


def apply_overrides(config: "LosionConfig", overrides: list[str]) -> "LosionConfig":
    """Terapkan override ke konfigurasi.

    Format: key=value, e.g., training.batch_size=16

    Args:
        config: Konfigurasi dasar.
        overrides: List override string.

    Returns:
        Konfigurasi yang sudah di-override.
    """
    for override in overrides:
        if "=" not in override:
            logging.warning(f"Override format salah (harus key=value): {override}")
            continue

        key, value = override.split("=", 1)

        # Navigasi ke attribute yang tepat
        parts = key.split(".")
        obj = config
        for part in parts[:-1]:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                logging.warning(f"Attribute tidak ditemukan: {key}")
                break
        else:
            # Set nilai
            final_key = parts[-1]
            if hasattr(obj, final_key):
                # Coba infer tipe dari nilai saat ini
                current = getattr(obj, final_key)
                if isinstance(current, bool):
                    setattr(obj, final_key, value.lower() in ("true", "1", "yes"))
                elif isinstance(current, int):
                    setattr(obj, final_key, int(value))
                elif isinstance(current, float):
                    setattr(obj, final_key, float(value))
                elif isinstance(current, str):
                    setattr(obj, final_key, value)
                else:
                    logging.warning(f"Tipe tidak dikenali untuk {key}: {type(current)}")
            else:
                logging.warning(f"Attribute tidak ditemukan: {key}")

    return config


def create_dummy_dataloader(
    vocab_size: int,
    batch_size: int,
    seq_len: int = 2048,
    num_batches: int = 100,
) -> "DataLoader":
    """Buat dummy DataLoader untuk testing.

    Menghasilkan token IDs acak yang menyerupai data training.
    Gunakan HANYA untuk testing dan development.

    Args:
        vocab_size: Ukuran vocabulary.
        batch_size: Ukuran batch.
        seq_len: Panjang sequence.
        num_batches: Jumlah batch.

    Returns:
        DataLoader dengan data dummy.
    """
    import torch
    from torch.utils.data import DataLoader, Dataset

    class DummyDataset(Dataset):
        """Dataset dummy untuk testing."""

        def __init__(self, vocab_size: int, seq_len: int, num_samples: int):
            self.data = torch.randint(0, vocab_size, (num_samples, seq_len))
            self.labels = torch.cat(
                [self.data[:, 1:], torch.zeros(num_samples, 1, dtype=torch.long)],
                dim=1,
            )

        def __len__(self) -> int:
            return len(self.data)

        def __getitem__(self, idx: int) -> dict:
            return {
                "input_ids": self.data[idx],
                "labels": self.labels[idx],
            }

    dataset = DummyDataset(vocab_size, seq_len, num_batches * batch_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )
    return dataloader


def main() -> None:
    """Entry point utama training script."""
    args = parse_args()

    # === Setup logging ===
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("losion.train")

    # === Load konfigurasi ===
    logger.info(f"Loading konfigurasi dari: {args.config}")
    config = load_config(args.config)

    # === Terapkan command-line overrides ===
    if args.override:
        config = apply_overrides(config, args.override)

    if args.output_dir:
        config.training.output_dir = args.output_dir  # type: ignore
    if args.max_steps:
        config.training.max_steps = args.max_steps
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if args.no_bf16:
        config.training.fp8_enabled = False

    # === Print konfigurasi ===
    logger.info(f"Konfigurasi: {config}")

    # === Import komponen training ===
    from losion.config import LosionConfig
    from losion.models.losion_decoder import LosionForCausalLM
    from losion.training.curriculum import TrainingPhase
    from losion.training.trainer import LosionTrainer, TrainerConfig
    from losion.training.utils import count_parameters

    # === Buat trainer config ===
    trainer_config = TrainerConfig(
        output_dir=args.output_dir or "./checkpoints",
        max_train_steps=config.training.max_steps,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        bf16=not args.no_bf16,
        fp16=args.fp16,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        seed=args.seed,
        resume_from_checkpoint=args.resume,
    )

    # === Buat trainer ===
    logger.info("Membuat LosionTrainer...")
    trainer = LosionTrainer(
        model_config=config,
        trainer_config=trainer_config,
    )

    # === Override fase jika ditentukan ===
    if args.phase is not None:
        phase_map = {
            1: TrainingPhase.PHASE_1_INDIVIDUAL,
            2: TrainingPhase.PHASE_2_JOINT,
            3: TrainingPhase.PHASE_3_RL,
            4: TrainingPhase.PHASE_4_ADVANCED,
        }
        target_phase = phase_map[args.phase]
        trainer.curriculum.set_phase(target_phase)
        trainer._apply_phase_to_model(target_phase)
        logger.info(f"Starting langsung di Fase {args.phase}: {target_phase.value}")

    # === Buat dataloader ===
    if args.data_dir:
        # TODO: Implementasi dataset loading dari file
        logger.warning(
            f"Custom data dir '{args.data_dir}' belum didukung penuh. "
            "Menggunakan dummy data."
        )

    train_dataloader = create_dummy_dataloader(
        vocab_size=config.vocab_size,
        batch_size=config.training.batch_size,
        seq_len=min(2048, config.max_seq_len),
    )

    eval_dataloader = create_dummy_dataloader(
        vocab_size=config.vocab_size,
        batch_size=config.training.batch_size,
        seq_len=min(512, config.max_seq_len),
        num_batches=10,
    )

    # === Dry run ===
    if args.dry_run:
        logger.info("=== DRY RUN MODE ===")
        logger.info("Menjalankan 1 langkah training untuk verifikasi...")

        batch = next(iter(train_dataloader))
        input_ids = batch["input_ids"].to(trainer.device)
        labels = batch["labels"].to(trainer.device)

        with torch.amp.autocast(
            "cuda",
            dtype=torch.bfloat16 if trainer_config.bf16 else torch.float32,
            enabled=trainer_config.bf16 and trainer.device.type == "cuda",
        ):
            output = trainer.model(input_ids=input_ids, labels=labels)

        logger.info(f"  Loss: {output.loss.item():.4f}")
        logger.info(f"  AR Loss: {output.ar_loss.item():.4f}")
        if output.mtp_loss is not None:
            logger.info(f"  MTP Loss: {output.mtp_loss.item():.4f}")
        logger.info(f"  Logits shape: {output.logits.shape}")
        logger.info("=== DRY RUN SELESAI ===")
        return

    # === Mulai training ===
    logger.info("=== Memulai Training Losion ===")
    logger.info(f"  Config: {args.config}")
    logger.info(f"  Device: {trainer.device}")
    logger.info(f"  Precision: {'bf16' if trainer_config.bf16 else 'fp16' if trainer_config.fp16 else 'fp32'}")
    logger.info(f"  Batch size: {config.training.batch_size}")
    logger.info(f"  Max steps: {config.training.max_steps}")
    logger.info(f"  Wandb: {'enabled' if args.use_wandb else 'disabled'}")

    result = trainer.train(
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
    )

    # === Print hasil ===
    logger.info("=== Training Selesai ===")
    logger.info(f"  Total steps: {result['total_steps']}")
    logger.info(f"  Total time: {result['total_time']:.1f} detik")
    logger.info(f"  Best eval loss: {result['best_eval_loss']:.4f}")
    logger.info(f"  Final phase: {result['final_phase']}")

    # === Simpan hasil ===
    output_dir = trainer_config.output_dir
    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, "training_result.json")
    with open(result_path, "w") as f:
        # Convert non-serializable values
        serializable_result = {
            k: v if isinstance(v, (int, float, str, bool, list, dict)) else str(v)
            for k, v in result.items()
        }
        json.dump(serializable_result, f, indent=2)
    logger.info(f"  Hasil disimpan: {result_path}")


if __name__ == "__main__":
    main()
