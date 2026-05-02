"""
Losion — Model Implementations.

v0.4 addition:
  ParallelHeadLayer — Parallel-head mode for Losion-1B (eliminate routing overhead)
"""

from losion.models.parallel_head import ParallelHeadLayer

__all__ = ["ParallelHeadLayer"]
