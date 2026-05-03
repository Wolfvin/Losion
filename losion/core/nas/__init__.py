"""
Losion — Neural Architecture Search Module.

v0.4 addition:
  NASConfig — Configuration for NAS
  NASLayerChoice — Layer choice module for differentiable NAS
  NASController — DARTS-style differentiable NAS controller
  HybridLayerWrapper — Wrapper for hybrid layer configurations
"""

from losion.core.nas.layer_search import NASConfig, NASLayerChoice, NASController, HybridLayerWrapper

__all__ = ["NASConfig", "NASLayerChoice", "NASController", "HybridLayerWrapper"]
