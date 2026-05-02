"""
Losion Evaluation Script — evaluasi model pada benchmark standar.

Usage:
    python scripts/evaluate.py --config configs/losion-1b.yaml
    python scripts/evaluate.py --config configs/losion-7b.yaml --checkpoint checkpoints/best
    python scripts/evaluate.py --config configs/losion-7b.yaml --benchmark mmlu,gsm8k
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Tambahkan project root ke sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Losion Evaluation Script — evaluasi model pada benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path ke file YAML konfigurasi model",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path ke checkpoint direktori (jika None, gunakan model baru)",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="perplexity",
        help="Benchmark untuk evaluasi (comma-separated): perplexity,mmlu,gsm8k,humaneval",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size untuk evaluasi",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=2048,
        help="Panjang sequence untuk evaluasi",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Jumlah sampel untuk evaluasi",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="File output untuk hasil evaluasi (JSON)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device untuk evaluasi (auto, cuda, cpu)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Verbose logging",
    )

    return parser.parse_args()


def evaluate_perplexity(
    model,
    vocab_size: int,
    batch_size: int,
    seq_len: int,
    num_samples: int,
    device: str,
) -> Dict[str, float]:
    """Evaluasi perplexity model pada data acak.

    NOTE: Ini menggunakan data dummy. Untuk evaluasi nyata,
    gunakan dataset yang sesuai.

    Args:
        model: Model LosionForCausalLM.
        vocab_size: Ukuran vocabulary.
        batch_size: Batch size.
        seq_len: Panjang sequence.
        num_samples: Jumlah sampel.
        device: Device evaluasi.

    Returns:
        Dictionary berisi metrik perplexity.
    """
    import torch

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    num_batches = num_samples // batch_size

    logger = logging.getLogger("losion.evaluate")

    with torch.no_grad():
        for i in range(num_batches):
            # Generate random input
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
            labels = input_ids.clone()
            labels[:, :-1] = input_ids[:, 1:]
            labels[:, -1] = -100  # Ignore last token

            input_ids = input_ids.to(device)
            labels = labels.to(device)

            # Forward pass
            output = model(input_ids=input_ids, labels=labels)

            if output.loss is not None:
                total_loss += output.loss.item() * batch_size * (seq_len - 1)
                total_tokens += batch_size * (seq_len - 1)

            if (i + 1) % 10 == 0:
                avg_loss = total_loss / max(total_tokens, 1)
                ppl = float("inf") if avg_loss > 100 else 2.718 ** avg_loss
                logger.info(f"  Batch {i+1}/{num_batches} | Loss: {avg_loss:.4f} | PPL: {ppl:.2f}")

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = float("inf") if avg_loss > 100 else 2.718 ** avg_loss

    return {
        "perplexity": perplexity,
        "avg_loss": avg_loss,
        "total_tokens": total_tokens,
    }


def evaluate_routing_distribution(
    model,
    vocab_size: int,
    batch_size: int,
    seq_len: int,
    device: str,
) -> Dict[str, object]:
    """Evaluasi distribusi routing model.

    Menganalisis bagaimana router mendistribusikan token
    ke ketiga jalur.

    Args:
        model: Model LosionForCausalLM.
        vocab_size: Ukuran vocabulary.
        batch_size: Batch size.
        seq_len: Panjang sequence.
        device: Device evaluasi.

    Returns:
        Dictionary berisi statistik routing.
    """
    import torch

    model.eval()

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)

    with torch.no_grad():
        output = model(input_ids=input_ids, return_routing_info=True)

    if not output.routing_info:
        return {"error": "Routing info tidak tersedia"}

    # Agregasi statistik routing
    ssm_weights = []
    attn_weights = []
    retr_weights = []
    thinking_modes = []

    for layer_routing in output.routing_info:
        weights = layer_routing.adjusted_weights  # [batch, seq, 3]
        ssm_weights.append(weights[:, :, 0].mean().item())
        attn_weights.append(weights[:, :, 1].mean().item())
        retr_weights.append(weights[:, :, 2].mean().item())
        thinking_modes.append(layer_routing.thinking_assessment.mode.value)

    return {
        "mean_ssm_weight": sum(ssm_weights) / len(ssm_weights),
        "mean_attn_weight": sum(attn_weights) / len(attn_weights),
        "mean_retr_weight": sum(retr_weights) / len(retr_weights),
        "ssm_weights_per_layer": ssm_weights,
        "attn_weights_per_layer": attn_weights,
        "retr_weights_per_layer": retr_weights,
        "thinking_modes": thinking_modes,
        "num_layers": len(ssm_weights),
    }


def evaluate_parameter_stats(model) -> Dict[str, object]:
    """Evaluasi statistik parameter model.

    Args:
        model: Model LosionForCausalLM.

    Returns:
        Dictionary berisi statistik parameter.
    """
    import torch

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Per-komponen
    component_params = model.count_parameters()

    # Memory estimation
    param_memory_mb = total_params * 2 / (1024 ** 2)  # bf16

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "trainable_fraction": trainable_params / total_params,
        "parameter_memory_mb": param_memory_mb,
        "components": component_params,
    }


def main() -> None:
    """Entry point utama evaluation script."""
    args = parse_args()

    # === Setup logging ===
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    global logger
    logger = logging.getLogger("losion.evaluate")

    # === Import ===
    import torch
    from losion.config import LosionConfig
    from losion.models.losion_decoder import LosionForCausalLM

    # === Load konfigurasi ===
    logger.info(f"Loading konfigurasi dari: {args.config}")
    config = LosionConfig.from_yaml(args.config)

    # === Resolve device ===
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info(f"Device: {device}")

    # === Load model ===
    if args.checkpoint:
        logger.info(f"Loading model dari checkpoint: {args.checkpoint}")
        model = LosionForCausalLM.from_pretrained(args.checkpoint, device=device)
    else:
        logger.info("Membuat model baru (random weights)")
        model = LosionForCausalLM(config)
        model.to(device)

    model.eval()

    # === Print model info ===
    param_stats = evaluate_parameter_stats(model)
    logger.info(f"Total parameters: {param_stats['total_parameters']:,}")
    logger.info(f"Trainable: {param_stats['trainable_parameters']:,} ({param_stats['trainable_fraction']:.1%})")
    logger.info(f"Memory (bf16): {param_stats['parameter_memory_mb']:.1f} MB")

    # === Jalankan benchmark ===
    benchmarks = args.benchmark.split(",")
    results = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "device": device,
        "parameter_stats": param_stats,
        "benchmarks": {},
    }

    start_time = time.time()

    for bench in benchmarks:
        bench = bench.strip().lower()
        logger.info(f"\n=== Evaluasi: {bench} ===")

        if bench == "perplexity":
            ppl_result = evaluate_perplexity(
                model=model,
                vocab_size=config.vocab_size,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                num_samples=args.num_samples,
                device=device,
            )
            results["benchmarks"]["perplexity"] = ppl_result
            logger.info(f"  Perplexity: {ppl_result['perplexity']:.2f}")
            logger.info(f"  Avg Loss: {ppl_result['avg_loss']:.4f}")

        elif bench == "routing":
            routing_result = evaluate_routing_distribution(
                model=model,
                vocab_size=config.vocab_size,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                device=device,
            )
            results["benchmarks"]["routing"] = routing_result
            logger.info(f"  SSM weight: {routing_result.get('mean_ssm_weight', 'N/A')}")
            logger.info(f"  Attention weight: {routing_result.get('mean_attn_weight', 'N/A')}")
            logger.info(f"  Retrieval weight: {routing_result.get('mean_retr_weight', 'N/A')}")

        elif bench == "params":
            # Sudah dijalankan di atas
            results["benchmarks"]["params"] = param_stats

        else:
            logger.warning(f"Benchmark tidak dikenali: {bench}")
            logger.warning("Benchmark yang tersedia: perplexity, routing, params")

    total_time = time.time() - start_time
    results["total_time_seconds"] = total_time

    # === Print ringkasan ===
    logger.info("\n=== Ringkasan Evaluasi ===")
    logger.info(f"  Waktu total: {total_time:.1f} detik")
    for bench_name, bench_result in results["benchmarks"].items():
        logger.info(f"  {bench_name}: {bench_result}")

    # === Simpan hasil ===
    output_file = args.output_file
    if output_file is None:
        output_file = os.path.join(
            args.checkpoint or ".",
            "eval_result.json",
        )

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    # Convert non-serializable values
    def make_serializable(obj):
        if isinstance(obj, (int, float, str, bool)):
            return obj
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(v) for v in obj]
        else:
            return str(obj)

    with open(output_file, "w") as f:
        json.dump(make_serializable(results), f, indent=2)
    logger.info(f"Hasil disimpan: {output_file}")


if __name__ == "__main__":
    logger = None
    main()
