"""
Losion — Model Implementations.

Core models:
  LosionModel — Backbone with Tri-Jalur Router architecture
  LosionForCausalLM — Causal LM with loss computation and save/load
  LosionLayer — Single Tri-Jalur layer (SSM + Attention + MoE)
  RMSNorm — Root Mean Square Layer Normalization

v0.4 addition:
  ParallelHeadLayer — Parallel-head mode for Losion-1B (eliminate routing overhead)
"""

from losion.models.parallel_head import ParallelHeadLayer
from losion.models.losion_model import LosionModel, LosionLayer, LosionLayerOutput, RMSNorm
from losion.models.losion_decoder import LosionForCausalLM, LosionCausalLMOutput

__all__ = [
    "ParallelHeadLayer",
    "LosionModel",
    "LosionLayer",
    "LosionLayerOutput",
    "RMSNorm",
    "LosionForCausalLM",
    "LosionCausalLMOutput",
]
