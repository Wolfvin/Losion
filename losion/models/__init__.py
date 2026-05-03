"""
Losion — Model Implementations.

Core models:
  LosionModel — Backbone with Tri-Jalur Router architecture (V1, simplified)
  LosionForCausalLM — Causal LM with loss computation and save/load (V1)

V2 models (production, config-driven):
  LosionModelV2 — Backbone with full config-driven module selection
  LosionForCausalLMV2 — Complete causal LM with generate(), MTP, JEPA

V0.4 addition:
  ParallelHeadLayer — Parallel-head mode for Losion-1B (eliminate routing overhead)
"""

from losion.models.parallel_head import ParallelHybridHead, ParallelHeadConfig as _PHCfg
from losion.models.losion_model import LosionModel, LosionLayer, LosionLayerOutput, RMSNorm
from losion.models.losion_decoder import LosionForCausalLM, LosionCausalLMOutput
from losion.models.losion_model_v2 import (
    LosionModelV2,
    LosionLayerV2,
    LosionForCausalLMV2,
    MTPHead,
    RoPE,
)

__all__ = [
    "ParallelHybridHead",
    "LosionModel",
    "LosionLayer",
    "LosionLayerOutput",
    "RMSNorm",
    "LosionForCausalLM",
    "LosionCausalLMOutput",
    # V2 exports
    "LosionModelV2",
    "LosionLayerV2",
    "LosionForCausalLMV2",
    "MTPHead",
    "RoPE",
]
