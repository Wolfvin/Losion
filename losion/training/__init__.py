"""
Losion Training — Modul pelatihan untuk framework Losion.

Mengimplementasikan training 4-fase dengan dukungan:
- LosionTrainer: Trainer utama dengan 4 fase curriculum
- GRPOTrainer: Group Relative Policy Optimization (dari DeepSeek-R1)
- AdvancedGRPOTrainer: GRPO + Self-Play + Value Head (DeepMind)
- CurriculumScheduler: Penjadwal transisi antar fase
- Advanced RLHF: Self-Play Preference, Value Head, Self-Consistency
- Advanced Backprop: Chinchilla Scaling, Soft Capping, Scheduled Sampling
- Advanced Memory/Data: Progressive KV, Attention Sinks, Modality-Aware Loss

Penggunaan:
    >>> from losion.training import LosionTrainer, GRPOTrainer, CurriculumScheduler
    >>> from losion.training import AdvancedGRPOTrainer
    >>> from losion.config import LosionConfig
    >>> config = LosionConfig()
    >>> trainer = LosionTrainer(config)
    >>> trainer.train()
"""

from __future__ import annotations

from losion.training.trainer import LosionTrainer
from losion.training.grpo import GRPOTrainer
from losion.training.curriculum import CurriculumScheduler
from losion.training.advanced_rlhf import (
    AdvancedGRPOTrainer,
    AdvancedGRPOConfig,
    JalurValueHead,
    SelfPlayPreferenceGenerator,
    SelfConsistencyVerifier,
    DirichletNoiseInjector,
)
from losion.training.advanced_backprop import (
    ChinchillaScaler,
    ChinchillaScalingResult,
    PerJalurLRScheduler,
    LogitSoftCapper,
    ScheduledSampler,
    ConfidenceHeads,
    ParallelAttentionFFN,
    GradientOverlapScheduler,
    MemoryEfficientBackprop,
)
from losion.training.advanced_memory_data import (
    ProgressiveKVCompressor,
    AttentionSinkManager,
    DynamicExpertBufferAllocator,
    ModalityAwareLossWeighter,
    ChinchillaDataSizer,
    SampleFilterPipeline,
    TemplateConditionalRouter,
)

__all__ = [
    "LosionTrainer",
    "GRPOTrainer",
    "CurriculumScheduler",
    # Advanced RLHF
    "AdvancedGRPOTrainer",
    "AdvancedGRPOConfig",
    "JalurValueHead",
    "SelfPlayPreferenceGenerator",
    "SelfConsistencyVerifier",
    "DirichletNoiseInjector",
    # Advanced Backprop
    "ChinchillaScaler",
    "ChinchillaScalingResult",
    "PerJalurLRScheduler",
    "LogitSoftCapper",
    "ScheduledSampler",
    "ConfidenceHeads",
    "ParallelAttentionFFN",
    "GradientOverlapScheduler",
    "MemoryEfficientBackprop",
    # Advanced Memory & Data
    "ProgressiveKVCompressor",
    "AttentionSinkManager",
    "DynamicExpertBufferAllocator",
    "ModalityAwareLossWeighter",
    "ChinchillaDataSizer",
    "SampleFilterPipeline",
    "TemplateConditionalRouter",
]
