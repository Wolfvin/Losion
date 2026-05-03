"""
Losion — Hybrid AI Framework with Tri-Jalur Router Architecture.

Version 0.8.0 — "Next-Gen Training & Infinite Experts"

Losion combines three complementary computational pathways into a single
adaptive architecture:

  Jalur 1 (SSM):           Mamba-2 SSD + RWKV-7 WKV + Gated DeltaNet
                            + Mamba-3 + Routing Mamba + Liquid SSM
                            + PoST Decay + FG2-GDN
  Jalur 2 (Attention):     MLA + iRoPE + Lightning Attention + RoPE
                            + Gated Attention + MoBA + KDA+MLA
                            + Cross-Jalur Attention-MoE Routing
  Jalur 3 (Retrieval):     MoE + Engram Memory + Expert Choice
                            + S'MoRE + Symbolic-MoE + AuxFreeMoE
                            + ∞-MoE (Infinite Mixture of Experts)

Router:  Adaptive (BiasRouter + ThinkingToggle + Symbolic-MoE), GRPO/DAPO-trained.

v0.8 Upgrades — "Next-Gen Training & Infinite Experts":
  TRAINING:     DAPO — Decoupled Clip & Dynamic Sampling Policy Optimization
                RLVR — Reinforcement Learning with Verifiable Rewards
                Losion Training Orchestrator (unified 4-phase pipeline)
                Integrates: WSD, JEPA, TACO, ETR, GRPO/DAPO, Gen Distill,
                BitDistill, Curriculum, Active Learning, Evolutionary Search
  MOE:          ∞-MoE — Infinite Mixture of Experts (continuous expert space)
                Cross-Jalur Attention-MoE Routing (graph-based)
  OUTPUT:       L-MTP — Leap Multi-Token Prediction (geometric leaps)
  INFERENCE:    Expert Prefetching (Speculating Experts, arXiv 2603.19289)
  MODEL:        LosionModelV2 — ∞-MoE support, fixed dimension projections
  CONFIG:       DAPOConfig, RLVRConfig, PrefetchConfig sub-configs
"""

__version__ = "0.8.0"
__author__ = "Losion Contributors"
__license__ = "MIT"
