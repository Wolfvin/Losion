"""
Losion — Memory Modules.

v0.9 additions:
  DualMemoryConfig  — Configuration for Two-Level Memory System
  WorkingMemory     — Short-term memory (ring buffer, direct access)
  LongTermMemory    — Long-term memory (compressed, persistent state)
  DualMemorySystem  — Full two-level memory with consolidation and retrieval
"""

from losion.core.memory.dual_memory import (
    DualMemoryConfig,
    WorkingMemory,
    LongTermMemory,
    DualMemorySystem,
)

__all__ = [
    "DualMemoryConfig",
    "WorkingMemory",
    "LongTermMemory",
    "DualMemorySystem",
]
