"""
Losion Distributed — Multi-GPU/multi-node distributed training.

Provides distributed training tools for Losion models including:
  - 4D Parallelism (Data + Tensor + Pipeline + Context)
  - FSDP model wrapping with configurable sharding strategies
  - Context parallelism for long sequence training
  - Expert parallelism for MoE layers

Credits:
  - PyTorch FSDP2 (2025)
  - WLB-LLM 4D Parallelism (OSDI 2025)
  - AutoSP (arXiv:2604.27089)
  - DeepSpeed Ulysses

Usage:
    >>> from losion.distributed import ParallelismConfig, LosionDistributedTrainer
    >>> from losion.config import LosionConfig
    >>> config = ParallelismConfig(dp_size=4, tp_size=2, fsdp_sharding_strategy="full")
    >>> trainer = LosionDistributedTrainer(config, model)
    >>> metrics = trainer.train(dataset, num_epochs=3)
"""

from __future__ import annotations

from losion.distributed.parallel import (
    ContextParallel,
    LosionDistributedTrainer,
    LosionFSDPWrapper,
    ParallelismConfig,
    ParallelismMode,
    ShardingStrategy,
)

__all__ = [
    "ContextParallel",
    "LosionDistributedTrainer",
    "LosionFSDPWrapper",
    "ParallelismConfig",
    "ParallelismMode",
    "ShardingStrategy",
]
