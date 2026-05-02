"""
Losion Checkpoint Conversion Utility — konversi checkpoint antar format.

Mendukung konversi:
- Losion ↔ HuggingFace format
- Full checkpoint → sharded checkpoint
- bf16 → fp16 / fp32
- Extract config dari checkpoint

Usage:
    # Konversi ke HuggingFace format
    python scripts/convert_checkpoint.py --input checkpoints/best --output converted/ --format huggingface

    # Konversi precision
    python scripts/convert_checkpoint.py --input checkpoints/best --output converted_fp16/ --precision fp16

    # Shard checkpoint besar
    python scripts/convert_checkpoint.py --input checkpoints/best --output sharded/ --shard 5GB

    # Extract konfigurasi saja
    python scripts/convert_checkpoint.py --input checkpoints/best --output config.json --extract_config
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Optional

# Tambahkan project root ke sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Losion Checkpoint Conversion Utility",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path ke checkpoint input (direktori atau file .pt)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path ke output (direktori atau file)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="losion",
        choices=["losion", "huggingface", "safetensors"],
        help="Format output checkpoint",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default=None,
        choices=["fp32", "fp16", "bf16"],
        help="Konversi precision tensor (None = tidak diubah)",
    )
    parser.add_argument(
        "--shard",
        type=str,
        default=None,
        help="Shard checkpoint dengan ukuran maksimum (e.g., 5GB, 10GB)",
    )
    parser.add_argument(
        "--extract_config",
        action="store_true",
        default=False,
        help="Hanya extract konfigurasi dari checkpoint",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        default=False,
        help="Tampilkan apa yang akan dilakukan tanpa mengeksekusi",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Verbose logging",
    )

    return parser.parse_args()


def load_losion_checkpoint(input_path: str) -> Dict:
    """Load checkpoint Losion.

    Args:
        input_path: Path ke direktori checkpoint.

    Returns:
        Dictionary berisi model state dict dan konfigurasi.
    """
    import torch

    checkpoint_data = {}

    # Load state dict
    model_path = os.path.join(input_path, "model.pt")
    if os.path.exists(model_path):
        checkpoint_data["state_dict"] = torch.load(
            model_path, map_location="cpu", weights_only=True
        )
    else:
        # Coba load langsung sebagai file .pt
        if input_path.endswith(".pt"):
            checkpoint_data["state_dict"] = torch.load(
                input_path, map_location="cpu", weights_only=True
            )
        else:
            raise FileNotFoundError(f"Checkpoint tidak ditemukan: {input_path}")

    # Load konfigurasi
    config_path = os.path.join(input_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            checkpoint_data["config"] = json.load(f)

    return checkpoint_data


def convert_precision(
    state_dict: Dict,
    target_precision: str,
) -> Dict:
    """Konversi precision tensor dalam state dict.

    Args:
        state_dict: Model state dict.
        target_precision: Target precision (fp32, fp16, bf16).

    Returns:
        State dict dengan precision yang dikonversi.
    """
    import torch

    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }

    target_dtype = dtype_map[target_precision]
    converted = {}

    for key, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor) and tensor.is_floating_point():
            converted[key] = tensor.to(target_dtype)
        else:
            converted[key] = tensor

    return converted


def shard_checkpoint(
    state_dict: Dict,
    max_shard_size: str,
    output_dir: str,
) -> list[str]:
    """Shard checkpoint besar menjadi file-file kecil.

    Args:
        state_dict: Model state dict.
        max_shard_size: Ukuran maksimum per shard (e.g., "5GB").
        output_dir: Direktori output.

    Returns:
        List path file shard yang dibuat.
    """
    import torch

    # Parse ukuran
    size_units = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}
    size_str = max_shard_size.upper()

    max_bytes = None
    for unit, multiplier in size_units.items():
        if size_str.endswith(unit):
            value = float(size_str[: -len(unit)])
            max_bytes = int(value * multiplier)
            break

    if max_bytes is None:
        raise ValueError(f"Format ukuran tidak valid: {max_shard_size}")

    # Bagi parameter ke shard
    shards = []
    current_shard = {}
    current_size = 0

    for key, tensor in state_dict.items():
        tensor_size = tensor.nelement() * tensor.element_size()

        if current_size + tensor_size > max_bytes and current_shard:
            shards.append(current_shard)
            current_shard = {}
            current_size = 0

        current_shard[key] = tensor
        current_size += tensor_size

    if current_shard:
        shards.append(current_shard)

    # Simpan shard
    os.makedirs(output_dir, exist_ok=True)
    shard_paths = []

    for i, shard in enumerate(shards):
        shard_name = f"model-{i+1:05d}-of-{len(shards):05d}.pt"
        shard_path = os.path.join(output_dir, shard_name)
        torch.save(shard, shard_path)
        shard_paths.append(shard_path)

    # Simpan index
    index = {
        "metadata": {
            "total_size": sum(
                t.nelement() * t.element_size()
                for t in state_dict.values()
            ),
            "num_shards": len(shards),
        },
        "weight_map": {},
    }

    for i, shard in enumerate(shards):
        shard_name = f"model-{i+1:05d}-of-{len(shards):05d}.pt"
        for key in shard.keys():
            index["weight_map"][key] = shard_name

    index_path = os.path.join(output_dir, "model.index.json")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    return shard_paths


def convert_to_huggingface(
    state_dict: Dict,
    config: Optional[Dict],
    output_dir: str,
) -> None:
    """Konversi checkpoint ke format HuggingFace.

    Args:
        state_dict: Model state dict.
        config: Konfigurasi model.
        output_dir: Direktori output.
    """
    import torch

    os.makedirs(output_dir, exist_ok=True)

    # Rename keys untuk format HuggingFace
    hf_state_dict = {}
    for key, value in state_dict.items():
        # Mapping sederhana: losion.model → model
        new_key = key
        if new_key.startswith("model."):
            new_key = new_key[6:]  # Remove "model." prefix
        # Tambahkan prefix "model." untuk backbone
        if not new_key.startswith("lm_head") and not new_key.startswith("model."):
            new_key = f"model.{new_key}"
        hf_state_dict[new_key] = value

    # Simpan state dict
    torch.save(hf_state_dict, os.path.join(output_dir, "pytorch_model.bin"))

    # Simpan config dalam format HuggingFace
    if config:
        hf_config = {
            "architectures": ["LosionForCausalLM"],
            "model_type": "losion",
            "torch_dtype": "bfloat16",
        }

        # Merge dengan config yang ada
        if "d_model" in config:
            hf_config["d_model"] = config["d_model"]
        if "n_layers" in config:
            hf_config["n_layers"] = config["n_layers"]
        if "vocab_size" in config:
            hf_config["vocab_size"] = config["vocab_size"]
        if "max_seq_len" in config:
            hf_config["max_seq_len"] = config["max_seq_len"]

        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(hf_config, f, indent=2)

    # Simpan generation config
    gen_config = {
        "do_sample": True,
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": 50,
        "max_new_tokens": 512,
    }
    with open(os.path.join(output_dir, "generation_config.json"), "w") as f:
        json.dump(gen_config, f, indent=2)


def main() -> None:
    """Entry point utama conversion script."""
    args = parse_args()

    # === Setup logging ===
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("losion.convert")

    # === Import ===
    import torch

    # === Extract config only ===
    if args.extract_config:
        logger.info("Extracting konfigurasi...")
        checkpoint = load_losion_checkpoint(args.input)

        if "config" in checkpoint:
            with open(args.output, "w") as f:
                json.dump(checkpoint["config"], f, indent=2)
            logger.info(f"Konfigurasi disimpan: {args.output}")
        else:
            logger.error("Konfigurasi tidak ditemukan dalam checkpoint")
        return

    # === Load checkpoint ===
    logger.info(f"Loading checkpoint dari: {args.input}")
    checkpoint = load_losion_checkpoint(args.input)
    state_dict = checkpoint["state_dict"]
    config = checkpoint.get("config")

    # Log info
    total_params = sum(v.nelement() for v in state_dict.values() if isinstance(v, torch.Tensor))
    total_size_mb = sum(
        v.nelement() * v.element_size() for v in state_dict.values() if isinstance(v, torch.Tensor)
    ) / (1024 ** 2)

    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Checkpoint size: {total_size_mb:.1f} MB")

    # === Dry run ===
    if args.dry_run:
        logger.info("=== DRY RUN ===")
        logger.info(f"  Input: {args.input}")
        logger.info(f"  Output: {args.output}")
        logger.info(f"  Format: {args.format}")
        logger.info(f"  Precision: {args.precision or 'unchanged'}")
        logger.info(f"  Shard: {args.shard or 'no'}")
        return

    # === Konversi precision ===
    if args.precision:
        logger.info(f"Konversi precision ke {args.precision}...")
        state_dict = convert_precision(state_dict, args.precision)

    # === Format output ===
    if args.format == "huggingface":
        logger.info("Konversi ke format HuggingFace...")
        convert_to_huggingface(state_dict, config, args.output)

    elif args.format == "safetensors":
        logger.info("Konversi ke SafeTensors format...")
        try:
            from safetensors.torch import save_file

            os.makedirs(args.output, exist_ok=True)
            save_file(state_dict, os.path.join(args.output, "model.safetensors"))
            if config:
                with open(os.path.join(args.output, "config.json"), "w") as f:
                    json.dump(config, f, indent=2)
        except ImportError:
            logger.error(
                "safetensors tidak terinstal. Install dengan: pip install safetensors"
            )
            return

    elif args.format == "losion":
        os.makedirs(args.output, exist_ok=True)
        torch.save(state_dict, os.path.join(args.output, "model.pt"))
        if config:
            with open(os.path.join(args.output, "config.json"), "w") as f:
                json.dump(config, f, indent=2)

    # === Sharding ===
    if args.shard:
        logger.info(f"Sharding checkpoint (max {args.shard} per shard)...")
        shard_paths = shard_checkpoint(state_dict, args.shard, args.output)
        logger.info(f"Dibuat {len(shard_paths)} shard")

    logger.info(f"Konversi selesai! Output: {args.output}")


if __name__ == "__main__":
    main()
