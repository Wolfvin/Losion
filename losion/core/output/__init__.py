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

v0.8 additions:
  LeapMTPHead — Leap Multi-Token Prediction (NeurIPS '25)

v0.9 additions (Architecture Document Implementations):
  AnchoredDecoderConfig  — Configuration for Anchored Diffusion Decoder
  DisambiguationBlock   — Resolve between similar tokens based on context
  CoherenceBlock        — Ensure parallel token consistency
  AnchoredDiffusionDecoder — Full anchored decoder pipeline
  ContinuousOutputHead  — Continuous vector output head (replaces softmax)
"""

from losion.core.output.flow_matching import FlowMatchingDecoder
from losion.core.output.diffusion_refinement import DiffusionRefinement
from losion.core.output.speculative_decoder import SpeculativeDecoder, MultiTokenPrediction
from losion.core.output.mirror_speculative import SSMDraftModel, MirrorSpeculativeDecoder
from losion.core.output.anchored_decoder import (
    AnchoredDecoderConfig,
    DisambiguationBlock,
    CoherenceBlock,
    AnchoredDiffusionDecoder,
    ContinuousOutputHead,
)

__all__ = [
    "FlowMatchingDecoder",
    "DiffusionRefinement",
    "SpeculativeDecoder",
    "MultiTokenPrediction",
    "SSMDraftModel",
    "MirrorSpeculativeDecoder",
    # v0.9 Anchored Decoder
    "AnchoredDecoderConfig",
    "DisambiguationBlock",
    "CoherenceBlock",
    "AnchoredDiffusionDecoder",
    "ContinuousOutputHead",
]
