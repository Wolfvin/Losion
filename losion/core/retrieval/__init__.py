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

v0.5 additions (Priority 1):
  AuxFreeMoERouter      — DeepSeek-V3 style aux-loss-free router
  MTPMoEHead            — Multi-Token Prediction head for MoE training
  AuxFreeMoE            — Complete aux-loss-free MoE with MTP training

v0.6 additions:
  SmoreMoE              — Sub-tree MoE with Residual Experts (Meta, NeurIPS 2025)
  SymbolicMoERouter     — Skill-based discrete routing (Symbolic-MoE, 2025)
"""

from losion.core.retrieval.engram import EngramMemory, EngramEntry
from losion.core.retrieval.expert_choice import ExpertChoiceMoE, ExpertChoiceRouter
from losion.core.retrieval.heterogeneous_moe import HeterogeneousMoE
from losion.core.retrieval.matryoshka_moe import MatryoshkaMoE
from losion.core.retrieval.gradient_routed_moe import GradientRoutedMoE
from losion.core.retrieval.asymmetric_placement import AsymmetricMoEPlacement
from losion.core.retrieval.aux_free_moe import (
    AuxFreeMoERouter,
    MTPMoEHead,
    AuxFreeMoE,
    AuxFreeRoutingInfo,
)
from losion.core.retrieval.smore import (
    SmoreConfig,
    SmoreMoE,
    SmoreRoutingInfo,
    ResidualSubTree,
    ComposedExpert,
)
from losion.core.retrieval.symbolic_moe import (
    SkillType,
    SkillClassifier,
    SymbolicRoutingRule,
    SymbolicMoERouter,
    SymbolicRoutingInfo,
)

__all__ = [
    "EngramMemory",
    "EngramEntry",
    "ExpertChoiceMoE",
    "ExpertChoiceRouter",
    "HeterogeneousMoE",
    "MatryoshkaMoE",
    "GradientRoutedMoE",
    "AsymmetricMoEPlacement",
    "AuxFreeMoERouter",
    "MTPMoEHead",
    "AuxFreeMoE",
    "AuxFreeRoutingInfo",
    "SmoreConfig",
    "SmoreMoE",
    "SmoreRoutingInfo",
    "ResidualSubTree",
    "ComposedExpert",
    "SkillType",
    "SkillClassifier",
    "SymbolicRoutingRule",
    "SymbolicMoERouter",
    "SymbolicRoutingInfo",
]
