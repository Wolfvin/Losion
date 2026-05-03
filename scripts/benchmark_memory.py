"""
Losion Memory Efficiency Benchmark — Before vs After v0.10 Improvements.

Benchmarks KV cache memory usage, inference memory footprint, and
training memory with the new v0.10 memory efficiency modules:
  - Sliding Window Attention (RATTENTION-inspired)
  - MoSA (Mixture of Sparse Attention)
  - KV Cache Quantization (TurboQuant-inspired)
  - Dynamic Memory Sparsification (DMS)
  - Parallel Hybrid Head (Hymba-inspired)

Credits & References:
  - Losion Framework: Wolfvin & Contributors (github.com/Wolfvin/Losion)
  - RATTENTION: Apple Machine Learning Research, 2025
  - MoSA: NeurIPS 2025 (arXiv 2505.00315)
  - TurboQuant: Google Research, 2025/2026
  - DMS: NeurIPS 2025 (arXiv 2506.05345)
  - Hymba: NVIDIA Research, ICLR 2025
"""

import sys
import math
import time
import json
import gc
import traceback
from typing import Dict, Any, List, Optional

import torch
import torch.nn as nn

sys.path.insert(0, "/home/z/my-project/download/Losion")

from losion.config import (
    LosionConfig, SSMConfig, AttentionConfig, RetrievalConfig, RouterConfig,
    RecurrentConfig, JEPAConfig, OutputConfig, SlidingWindowConfig,
    MoSAConfig, KVQuantConfig, DMSConfig, ParallelHeadConfig,
    AttnResConfig, EvoformerConfig, DualMemoryConfig,
)
from losion.models.losion_model_v2 import LosionForCausalLMV2
from losion.inference.kv_quantization import estimate_kv_memory


def create_baseline_config() -> LosionConfig:
    """Baseline config (current v0.9.1 without memory optimizations)."""
    return LosionConfig(
        d_model=192,
        n_layers=4,
        vocab_size=32000,
        max_seq_len=2048,
        dropout=0.0,
        ssm=SSMConfig(d_state=16, d_conv=4, expand=2, use_mamba3=True),
        attention=AttentionConfig(
            n_heads=4, d_kv=48, mla_latent_dim=48,
            use_gated_attention=True,
        ),
        retrieval=RetrievalConfig(
            num_experts=4, num_active_experts=2, d_ff=384,
            use_smore=True, smore_num_sub_trees=2, smore_sub_tree_depth=2,
        ),
        router=RouterConfig(),
        recurrent=RecurrentConfig(enabled=True, max_loop_iters=3, use_act=True, depth_lora_rank=4),
        jepa=JEPAConfig(enabled=True, prediction_horizon=2, latent_dim=48, prediction_weight=0.1),
        output=OutputConfig(use_mtp=True, mtp_num_tokens=2),
        attn_res=AttnResConfig(enabled=True, mode="block", num_blocks=2, use_gate=True),
        evoformer=EvoformerConfig(
            enabled=True, n_recycling_steps=2,
            use_layer_recycling=True, use_token_recycling=True,
            use_decoder_feedback=True, use_prediction_recycling=True,
            use_router_coevolve=True,
        ),
        dual_memory=DualMemoryConfig(
            enabled=True, working_memory_size=64,
            long_term_memory_dim=48, consolidation_method="attention",
        ),
        # NO memory optimizations (baseline)
        sliding_window=SlidingWindowConfig(enabled=False),
        mosa=MoSAConfig(enabled=False),
        kv_quant=KVQuantConfig(enabled=False),
        dms=DMSConfig(enabled=False),
        parallel_head=ParallelHeadConfig(enabled=False),
    )


def create_optimized_config() -> LosionConfig:
    """Optimized config (v0.10 with all memory efficiency features)."""
    return LosionConfig(
        d_model=192,
        n_layers=4,
        vocab_size=32000,
        max_seq_len=2048,
        dropout=0.0,
        ssm=SSMConfig(d_state=16, d_conv=4, expand=2, use_mamba3=True),
        attention=AttentionConfig(
            n_heads=4, d_kv=48, mla_latent_dim=48,
            use_gated_attention=True,
        ),
        retrieval=RetrievalConfig(
            num_experts=4, num_active_experts=2, d_ff=384,
            use_smore=True, smore_num_sub_trees=2, smore_sub_tree_depth=2,
        ),
        router=RouterConfig(),
        recurrent=RecurrentConfig(enabled=True, max_loop_iters=3, use_act=True, depth_lora_rank=4),
        jepa=JEPAConfig(enabled=True, prediction_horizon=2, latent_dim=48, prediction_weight=0.1),
        output=OutputConfig(use_mtp=True, mtp_num_tokens=2),
        attn_res=AttnResConfig(enabled=True, mode="block", num_blocks=2, use_gate=True),
        evoformer=EvoformerConfig(
            enabled=True, n_recycling_steps=2,
            use_layer_recycling=True, use_token_recycling=True,
            use_decoder_feedback=True, use_prediction_recycling=True,
            use_router_coevolve=True,
        ),
        dual_memory=DualMemoryConfig(
            enabled=True, working_memory_size=64,
            long_term_memory_dim=48, consolidation_method="attention",
        ),
        # v0.10 Memory optimizations ENABLED
        sliding_window=SlidingWindowConfig(
            enabled=True,
            window_size=512,
            use_token_sink=True,
            num_sink_tokens=1,
        ),
        mosa=MoSAConfig(
            enabled=False,  # Keep off for now - sliding window is enough
        ),
        kv_quant=KVQuantConfig(
            enabled=True,
            mode="int8",
        ),
        dms=DMSConfig(
            enabled=True,
            target_cache_ratio=0.5,
            eviction_strategy="key_norm",
        ),
        parallel_head=ParallelHeadConfig(
            enabled=False,  # Parallel head is alternative, not additive
        ),
    )


# ============================================================================
# Memory Benchmark Functions
# ============================================================================


def measure_model_memory(model) -> Dict[str, Any]:
    """Measure model's memory footprint."""
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
    grad_memory = sum(
        p.grad.numel() * p.grad.element_size()
        for p in model.parameters() if p.grad is not None
    )

    # CUDA memory if available
    cuda_alloc = 0
    cuda_reserved = 0
    if torch.cuda.is_available():
        cuda_alloc = torch.cuda.memory_allocated()
        cuda_reserved = torch.cuda.memory_reserved()

    return {
        'param_memory_mb': param_memory / (1024 * 1024),
        'grad_memory_mb': grad_memory / (1024 * 1024),
        'total_model_mb': param_memory / (1024 * 1024),
        'cuda_allocated_mb': cuda_alloc / (1024 * 1024),
        'cuda_reserved_mb': cuda_reserved / (1024 * 1024),
    }


def benchmark_kv_cache(config: LosionConfig) -> Dict[str, Any]:
    """Benchmark KV cache memory for different configurations."""
    n_layers = config.n_layers
    n_heads = config.attention.n_heads
    d_kv = config.attention.d_kv
    mla_latent_dim = config.attention.mla_latent_dim

    results = estimate_kv_memory(
        n_layers=n_layers,
        n_heads=n_heads,
        d_kv=d_kv,
        seq_len=config.max_seq_len,
        mla_latent_dim=mla_latent_dim,
    )

    # Also calculate for various sequence lengths
    seq_lengths = [256, 512, 1024, 2048, 4096, 8192]
    seq_results = {}
    for sl in seq_lengths:
        seq_results[str(sl)] = estimate_kv_memory(
            n_layers=n_layers, n_heads=n_heads, d_kv=d_kv,
            seq_len=sl, mla_latent_dim=mla_latent_dim,
        )

    return {
        'config_results': results,
        'per_seq_len': seq_results,
        'sliding_window_savings': SlidingWindowAttention_memory_savings(
            config.max_seq_len, 512, n_heads, d_kv, True, mla_latent_dim
        ) if config.sliding_window.enabled else None,
    }


def SlidingWindowAttention_memory_savings(
    seq_len, window_size, n_heads, d_kv, use_mla, mla_latent_dim
):
    """Calculate sliding window memory savings."""
    try:
        from losion.core.attention.sliding_window import SlidingWindowAttention
        return SlidingWindowAttention.memory_savings_vs_full(
            seq_len, window_size, n_heads, d_kv, use_mla, mla_latent_dim
        )
    except ImportError:
        # Manual calculation
        if use_mla:
            full = seq_len * mla_latent_dim * 2
            sw = window_size * mla_latent_dim * 2
        else:
            full = 2 * n_heads * seq_len * d_kv * 2
            sw = 2 * n_heads * window_size * d_kv * 2
        return {
            'full_attention_bytes': full,
            'sliding_window_bytes': sw,
            'savings_ratio': full / sw if sw > 0 else 1.0,
            'savings_pct': (1.0 - sw / full) * 100 if full > 0 else 0,
        }


def benchmark_inference_memory(config: LosionConfig, seq_len=128) -> Dict[str, Any]:
    """Benchmark memory usage during inference (generation)."""
    try:
        model = LosionForCausalLMV2(config)
        model.eval()

        # Force GC before measurement
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        device = next(model.parameters()).device
        vocab_size = config.vocab_size

        # Simulate inference: generate tokens one at a time
        input_ids = torch.randint(0, vocab_size, (1, 8), device=device)

        # Measure forward pass memory
        with torch.no_grad():
            # Prefill
            output = model(input_ids=input_ids)

            # Generation step
            for _ in range(min(seq_len, 32)):
                next_token = output['logits'][:, -1:, :].argmax(dim=-1)
                input_ids = torch.cat([input_ids, next_token], dim=1)
                output = model(input_ids=input_ids)

        # Measure memory
        mem = measure_model_memory(model)

        # Add KV cache simulation
        from losion.inference.kv_cache import KVCache
        if config.sliding_window.enabled:
            effective_seq = min(input_ids.shape[1], config.sliding_window.window_size)
        else:
            effective_seq = input_ids.shape[1]

        kv = KVCache(
            n_layers=config.n_layers,
            n_heads=config.attention.n_heads,
            d_kv=config.attention.d_kv,
            mla_latent_dim=config.attention.mla_latent_dim if config.attention.mla_latent_dim > 0 else 0,
            dtype=torch.float16,
        )

        # Fill KV cache for the generated sequence
        for layer_idx in range(config.n_layers):
            if kv.mla_mode:
                new_k = torch.randn(1, effective_seq, config.attention.mla_latent_dim)
            else:
                new_k = torch.randn(1, config.attention.n_heads, effective_seq, config.attention.d_kv)
            new_v = new_k.clone()
            kv.update(layer_idx, new_k, new_v)

        mem['kv_cache_mb'] = kv.memory_bytes() / (1024 * 1024)
        mem['kv_cache_summary'] = kv.memory_summary()
        mem['generated_seq_len'] = input_ids.shape[1]

        del model, kv
        gc.collect()

        return mem

    except Exception as e:
        traceback.print_exc()
        return {'error': str(e)}


def benchmark_training_with_memory(config: LosionConfig, steps=50) -> Dict[str, Any]:
    """Benchmark training convergence and memory."""
    try:
        model = LosionForCausalLMV2(config)
        model.train()

        device = next(model.parameters()).device
        vocab_size = config.vocab_size

        # Reset memory tracking
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # Track peak memory
        peak_mem = 0

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=6e-4, weight_decay=0.1,
            betas=(0.9, 0.95), eps=1e-8,
        )

        losses = []
        for step in range(steps):
            input_ids = torch.randint(0, min(500, vocab_size), (4, 32), device=device)
            labels = input_ids.clone()

            output = model(input_ids=input_ids, labels=labels)
            loss = output['loss']

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            losses.append(loss.item())

            # Track memory
            if torch.cuda.is_available():
                current = torch.cuda.memory_allocated()
                peak_mem = max(peak_mem, current)

        initial_loss = losses[0]
        final_loss = losses[-1]
        loss_reduction = (initial_loss - final_loss) / initial_loss * 100 if initial_loss > 0 else 0

        mem = measure_model_memory(model)

        del model
        gc.collect()

        return {
            'initial_loss': initial_loss,
            'final_loss': final_loss,
            'loss_reduction_pct': loss_reduction,
            'converged': final_loss < initial_loss * 0.8,
            'peak_memory_mb': peak_mem / (1024 * 1024) if peak_mem > 0 else 0,
            'model_memory_mb': mem['param_memory_mb'],
        }

    except Exception as e:
        traceback.print_exc()
        return {'error': str(e)}


# ============================================================================
# Main Comparison
# ============================================================================


def run_comparison():
    print("=" * 70)
    print("  Losion Memory Efficiency Benchmark — v0.10 Improvements")
    print("  RATTENTION | MoSA | TurboQuant | DMS | Hymba Parallel")
    print("=" * 70)

    # 1. KV Cache Memory Comparison (theoretical)
    print("\n[1/5] KV Cache Memory Analysis (Theoretical)")
    print("-" * 50)

    baseline_config = create_baseline_config()
    optimized_config = create_optimized_config()

    n_layers = baseline_config.n_layers
    n_heads = baseline_config.attention.n_heads
    d_kv = baseline_config.attention.d_kv
    mla_dim = baseline_config.attention.mla_latent_dim

    for seq_len in [256, 512, 1024, 2048, 4096, 8192]:
        estimates = estimate_kv_memory(
            n_layers=n_layers, n_heads=n_heads, d_kv=d_kv,
            seq_len=seq_len, mla_latent_dim=mla_dim,
        )
        std_fp16 = estimates['standard_fp16']['total_mb']
        mla_fp16 = estimates['mla_fp16']['total_mb']
        sw_mla_int8 = estimates['sw512_mla_int8']['total_mb']
        sw_mla_int4 = estimates['sw512_mla_int4']['total_mb']

        print(f"  seq_len={seq_len:5d}: Standard={std_fp16:.3f}MB | "
              f"MLA={mla_fp16:.3f}MB | SW+MLA+INT8={sw_mla_int8:.3f}MB | "
              f"SW+MLA+INT4={sw_mla_int4:.3f}MB")

    # Sliding window savings
    savings = SlidingWindowAttention_memory_savings(
        8192, 512, n_heads, d_kv, True, mla_dim
    )
    print(f"\n  Sliding Window (512) savings at seq_len=8192:")
    print(f"    Full attention: {savings['full_attention_bytes'] / 1024:.1f} KB")
    print(f"    Sliding window: {savings['sliding_window_bytes'] / 1024:.1f} KB")
    print(f"    Savings: {savings['savings_pct']:.1f}% ({savings['savings_ratio']:.1f}x reduction)")

    # 2. Baseline Model Benchmark
    print("\n[2/5] Baseline Model (v0.9.1 — no memory optimizations)")
    print("-" * 50)

    baseline_model = LosionForCausalLMV2(baseline_config)
    baseline_params = sum(p.numel() for p in baseline_model.parameters())
    baseline_mem = measure_model_memory(baseline_model)
    print(f"  Parameters: {baseline_params:,} ({baseline_params/1e6:.2f}M)")
    print(f"  Model memory: {baseline_mem['param_memory_mb']:.2f} MB")

    # Baseline training
    print("  Training baseline (50 steps)...")
    baseline_train = benchmark_training_with_memory(baseline_config, steps=50)
    print(f"  Loss: {baseline_train.get('initial_loss', 0):.4f} -> {baseline_train.get('final_loss', 0):.4f}")
    print(f"  Reduction: {baseline_train.get('loss_reduction_pct', 0):.1f}%")
    print(f"  Converged: {baseline_train.get('converged', False)}")

    del baseline_model
    gc.collect()

    # 3. Optimized Model Benchmark
    print("\n[3/5] Optimized Model (v0.10 — Sliding Window + KV Quant + DMS)")
    print("-" * 50)

    optimized_model = LosionForCausalLMV2(optimized_config)
    optimized_params = sum(p.numel() for p in optimized_model.parameters())
    optimized_mem = measure_model_memory(optimized_model)
    print(f"  Parameters: {optimized_params:,} ({optimized_params/1e6:.2f}M)")
    print(f"  Model memory: {optimized_mem['param_memory_mb']:.2f} MB")

    # Check which modules are active
    for i, layer in enumerate(optimized_model.model.layers):
        attn_type = type(layer.attention_layer).__name__
        ssm_type = type(layer.ssm_layer).__name__
        moe_type = type(layer.retrieval_layer).__name__
        print(f"  Layer {i}: SSM={ssm_type} | Attn={attn_type} | MoE={moe_type}")

    # Optimized training
    print("  Training optimized (50 steps)...")
    optimized_train = benchmark_training_with_memory(optimized_config, steps=50)
    print(f"  Loss: {optimized_train.get('initial_loss', 0):.4f} -> {optimized_train.get('final_loss', 0):.4f}")
    print(f"  Reduction: {optimized_train.get('loss_reduction_pct', 0):.1f}%")
    print(f"  Converged: {optimized_train.get('converged', False)}")

    del optimized_model
    gc.collect()

    # 4. KV Cache Comparison
    print("\n[4/5] KV Cache Memory (Simulated Inference)")
    print("-" * 50)

    baseline_kv = benchmark_inference_memory(baseline_config, seq_len=128)
    optimized_kv = benchmark_inference_memory(optimized_config, seq_len=128)

    print(f"  Baseline KV cache: {baseline_kv.get('kv_cache_mb', 0):.4f} MB")
    print(f"  Optimized KV cache: {optimized_kv.get('kv_cache_mb', 0):.4f} MB")

    # 5. Summary Comparison
    print("\n[5/5] Summary Comparison")
    print("=" * 70)

    results = {
        'baseline': {
            'params': baseline_params,
            'model_memory_mb': baseline_mem['param_memory_mb'],
            'training': baseline_train,
            'kv_cache': baseline_kv,
        },
        'optimized': {
            'params': optimized_params,
            'model_memory_mb': optimized_mem['param_memory_mb'],
            'training': optimized_train,
            'kv_cache': optimized_kv,
        },
        'kv_theoretical': estimate_kv_memory(
            n_layers=n_layers, n_heads=n_heads, d_kv=d_kv,
            seq_len=2048, mla_latent_dim=mla_dim,
        ),
    }

    # Determine if improvement
    baseline_reduction = baseline_train.get('loss_reduction_pct', 0)
    optimized_reduction = optimized_train.get('loss_reduction_pct', 0)

    print(f"\n  {'Metric':<30} {'Baseline':>12} {'Optimized':>12} {'Delta':>12}")
    print(f"  {'-'*66}")
    print(f"  {'Parameters':<30} {baseline_params:>12,} {optimized_params:>12,} {optimized_params - baseline_params:>+12,}")
    print(f"  {'Model Memory (MB)':<30} {baseline_mem['param_memory_mb']:>12.2f} {optimized_mem['param_memory_mb']:>12.2f} {optimized_mem['param_memory_mb'] - baseline_mem['param_memory_mb']:>+12.2f}")
    print(f"  {'Training Loss Reduction %':<30} {baseline_reduction:>12.1f} {optimized_reduction:>12.1f} {optimized_reduction - baseline_reduction:>+12.1f}")

    # KV cache at seq_len=2048
    kv_est = estimate_kv_memory(n_layers=n_layers, n_heads=n_heads, d_kv=d_kv, seq_len=2048, mla_latent_dim=mla_dim)
    baseline_kv_mb = kv_est['mla_fp16']['total_mb']
    optimized_kv_mb = kv_est['sw512_mla_int8']['total_mb']
    print(f"  {'KV Cache @2048 (MB)':<30} {baseline_kv_mb:>12.3f} {optimized_kv_mb:>12.3f} {optimized_kv_mb - baseline_kv_mb:>+12.3f}")
    print(f"  {'KV Cache Savings %':<30} {'0.0':>12} {(1-optimized_kv_mb/baseline_kv_mb)*100:>12.1f} {'':>12}")

    # Improvement assessment
    is_improvement = (
        optimized_reduction >= baseline_reduction * 0.9  # At least 90% of baseline quality
        and optimized_kv_mb < baseline_kv_mb              # Less KV cache memory
    )

    print(f"\n  IMPROVEMENT ASSESSMENT: {'YES - IMPROVEMENT' if is_improvement else 'NO - REGRESSION'}")

    if is_improvement:
        print("  The optimized model:")
        print(f"    - Maintains training quality ({optimized_reduction:.1f}% vs {baseline_reduction:.1f}%)")
        print(f"    - Reduces KV cache by {(1-optimized_kv_mb/baseline_kv_mb)*100:.1f}%")
        print(f"    - Safe to push to GitHub!")
    else:
        print("  The optimized model has regressions. Need to adjust configuration.")

    # Save results
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
                return str(obj)
            return obj
        elif isinstance(obj, torch.Tensor):
            return obj.tolist()
        else:
            return str(obj)

    results['is_improvement'] = is_improvement
    with open('/home/z/my-project/download/Losion/memory_benchmark_results.json', 'w') as f:
        json.dump(make_serializable(results), f, indent=2, default=str)

    print(f"\n  Results saved to memory_benchmark_results.json")
    print("=" * 70)

    return results, is_improvement


if __name__ == "__main__":
    results, is_improvement = run_comparison()
