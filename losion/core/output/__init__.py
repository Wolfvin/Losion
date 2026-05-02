"""
Losion — Output Modules.

Base modules:
  FlowMatchingDecoder — Flow Matching for high-quality generation
  DiffusionRefinement — Diffusion-based output refinement
  SpeculativeDecoder  — Speculative decoding with MTP
  MultiTokenPrediction — Multi-Token Prediction heads

v0.5 additions:
  SSMDraftModel — SSM pathway as draft model for speculative decoding
  MirrorSpeculativeDecoder — Mirror speculative decoding with SSM draft
"""

from losion.core.output.flow_matching import FlowMatchingDecoder
from losion.core.output.diffusion_refinement import DiffusionRefinement
from losion.core.output.speculative_decoder import SpeculativeDecoder, MultiTokenPrediction
from losion.core.output.mirror_speculative import SSMDraftModel, MirrorSpeculativeDecoder

__all__ = [
    "FlowMatchingDecoder",
    "DiffusionRefinement",
    "SpeculativeDecoder",
    "MultiTokenPrediction",
    # Mirror Speculative Decoding
    "SSMDraftModel",
    "MirrorSpeculativeDecoder",
]
