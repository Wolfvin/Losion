"""
Losion — Feedback Modules (Evoformer Universal Principle).

v0.9 additions:
  EvoformerConfig          — Configuration for all 5 Evoformer levels
  LayerRecyclingBlock      — Level 1: Inter-layer bidirectional feedback
  BidirectionalTokenUpdate — Level 2: Token-level bidirectional update
  DecoderPredictFeedback   — Level 3: Decoder ↔ Predict feedback loop
  PredictionContextRecycling — Level 4: Prediction → Context recycling
  RouterExpertCoevolve     — Level 5: Router ↔ Expert co-evolution
  EvoformerManager         — Coordinates all 5 levels
"""

from losion.core.feedback.evoformer import (
    EvoformerConfig,
    LayerRecyclingBlock,
    BidirectionalTokenUpdate,
    DecoderPredictFeedback,
    PredictionContextRecycling,
    RouterExpertCoevolve,
    EvoformerManager,
)

__all__ = [
    "EvoformerConfig",
    "LayerRecyclingBlock",
    "BidirectionalTokenUpdate",
    "DecoderPredictFeedback",
    "PredictionContextRecycling",
    "RouterExpertCoevolve",
    "EvoformerManager",
]
