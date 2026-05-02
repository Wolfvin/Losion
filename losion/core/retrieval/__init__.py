"""
Losion — Jalur 3: Retrieval MoE + Engram Memory.

Base modules:
  EngramMemory      — O(1) factual retrieval via hash table
  ExpertChoiceMoE   — MoE with Expert Choice routing (Google Research, 2022)

v0.4 additions:
  HeterogeneousMoE      — Variable-size experts for efficient capacity allocation
  MatryoshkaMoE         — Elastic expert count (Matryoshka-style nested MoE)
  GradientRoutedMoE     — Loss-aligned routing via gradient signals
  AsymmetricMoEPlacement — Selective MoE placement with layer-wise sparsity
"""

from losion.core.retrieval.engram import EngramMemory, EngramEntry
from losion.core.retrieval.expert_choice import ExpertChoiceMoE, ExpertChoiceRouter
from losion.core.retrieval.heterogeneous_moe import HeterogeneousMoE
from losion.core.retrieval.matryoshka_moe import MatryoshkaMoE
from losion.core.retrieval.gradient_routed_moe import GradientRoutedMoE
from losion.core.retrieval.asymmetric_placement import AsymmetricMoEPlacement

__all__ = [
    "EngramMemory",
    "EngramEntry",
    "ExpertChoiceMoE",
    "ExpertChoiceRouter",
    "HeterogeneousMoE",
    "MatryoshkaMoE",
    "GradientRoutedMoE",
    "AsymmetricMoEPlacement",
]
