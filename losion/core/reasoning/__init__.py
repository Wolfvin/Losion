"""
Losion Reasoning Module — DeepMind-inspired reasoning engines.

Integrates techniques from:
- AlphaZero: Monte Carlo Tree Search (MCTS) for inference-time reasoning
- Gemini Deep Think: Parallel thinking with multi-path exploration
- AlphaProof: Neuro-symbolic verification for mathematical/logical outputs

v0.5 additions (Priority 1):
- Path-Lock Expert (PLE): Architectural reasoning control (arXiv:2604.27201)
  Forces specific expert pathways for reasoning patterns, zero extra FLOPs

Modules:
    mcts: MCTS Reasoning Engine for inference-time compute scaling
    parallel_thinking: Parallel path exploration (Gemini Deep Think style)
    neuro_symbolic: Neuro-symbolic verification layer
    path_lock_expert: Path-Lock Expert for architectural reasoning control
"""

from .mcts import MCTSReasoner, MCTSConfig, MCTSNode
from .parallel_thinking import ParallelThinker, ThinkingPath, ThinkingBudget
from .neuro_symbolic import NeuroSymbolicVerifier, VerificationResult
from .path_lock_expert import (
    PathLockConfig,
    PathLockExpert,
    PathLockLayer,
    PathLockOutput,
    ReasoningType,
)

__all__ = [
    "MCTSReasoner", "MCTSConfig", "MCTSNode",
    "ParallelThinker", "ThinkingPath", "ThinkingBudget",
    "NeuroSymbolicVerifier", "VerificationResult",
    "PathLockConfig", "PathLockExpert", "PathLockLayer",
    "PathLockOutput", "ReasoningType",
]
