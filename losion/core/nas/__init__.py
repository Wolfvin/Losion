"""
Losion — Neural Architecture Search Module.

v0.4 addition:
  DARTSNAS — DARTS-style differentiable NAS for post-training layer optimization
"""

from losion.core.nas.layer_search import DARTSNAS

__all__ = ["DARTSNAS"]
