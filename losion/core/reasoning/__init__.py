"""
Losion Reasoning Module — DeepMind-inspired reasoning engines.

Integrates techniques from:
- AlphaZero: Monte Carlo Tree Search (MCTS) for inference-time reasoning
- Gemini Deep Think: Parallel thinking with multi-path exploration
- AlphaProof: Neuro-symbolic verification for mathematical/logical outputs

Modules:
    mcts: MCTS Reasoning Engine for inference-time compute scaling
    parallel_thinking: Parallel path exploration (Gemini Deep Think style)
    neuro_symbolic: Neuro-symbolic verification layer
"""

from .mcts import MCTSReasoner, MCTSConfig, MCTSNode
from .parallel_thinking import ParallelThinker, ThinkingPath, ThinkingBudget
from .neuro_symbolic import NeuroSymbolicVerifier, VerificationResult

__all__ = [
    "MCTSReasoner", "MCTSConfig", "MCTSNode",
    "ParallelThinker", "ThinkingPath", "ThinkingBudget",
    "NeuroSymbolicVerifier", "VerificationResult",
]
