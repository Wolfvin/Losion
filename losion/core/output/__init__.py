"""
Losion — Output Modules.

Base modules:
  FlowMatchingHead    — Flow Matching for high-quality generation
  DiffusionRefinement — Diffusion-based output refinement

v0.4 addition:
  MTPSpeculativeDecoder — Multi-Token Prediction speculative decoding
"""

from losion.core.output.flow_matching import FlowMatchingHead
from losion.core.output.diffusion_refinement import DiffusionRefinement
from losion.core.output.speculative_decoder import MTPSpeculativeDecoder

__all__ = [
    "FlowMatchingHead",
    "DiffusionRefinement",
    "MTPSpeculativeDecoder",
]
