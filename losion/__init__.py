"""
Losion — Hybrid AI Framework with Tri-Jalur Router Architecture.

Version 1.0.0 — "Verified & Alive"

v1.0.0 End-to-End Verified:
  All 40+ components have been tested with actual forward+backward passes.
  A 17M-parameter model was trained for 10 steps and all pathways verified:
  - SSM (Jalur 1): Gradient flows correctly through Mamba-3/RoutingMamba
  - Attention (Jalur 2): GatedAttention/MoBA properly connected
  - MoE (Jalur 3): SmoreMoE/AuxFreeMoE with proper load balancing
  - Router: AdaptiveRouter with ThinkingToggle dynamically routes
  - RDT: RecurrentDepthBlock with proper block wrapper
  - Evoformer: All 5 levels wired and functional
  - DualMemory: Write+Read cycle verified
  - JEPA: JEPAHead loss computed and gradients flow
  - MTP: Multi-token prediction loss correctly shaped
  - Generation: Autoregressive generation works end-to-end
  - Save/Load: Round-trip verified with zero difference

  Critical wiring fixes in v1.0.0:
  - MoBAAttention constructor: Fixed config vs positional arg mismatch
  - GatedAttention config: Added d_model field to GatedAttentionConfig
  - LLMJEPA: Replaced standalone wrapper with lightweight JEPAHead
  - RDT: Inner block now returns (output, aux) tuple + accepts **kwargs
  - MTP loss: Fixed shape mismatch in shifted label computation
  - Generation: Fixed dimension mismatch in token concatenation
  - Mamba3SSD: Fixed config object vs keyword arg constructor mismatch
  - SymbolicMoE: Fixed fall-through that didn't return a module
  - from_pretrained: Uses _from_dict for proper nested config loading

Losion combines three complementary computational pathways into a single
adaptive architecture:

  Jalur 1 (SSM):           Mamba-2 SSD + RWKV-7 WKV + Gated DeltaNet
                            + Mamba-3 + Routing Mamba + Liquid SSM
                            + PoST Decay + FG2-GDN + Structured Sparse SSM
  Jalur 2 (Attention):     MLA + iRoPE + Lightning Attention + RoPE
                            + Gated Attention + MoBA + KDA+MLA
                            + Cross-Jalur Attention-MoE Routing
                            + AttnRes (Attention Residuals, MoonshotAI 2026)
                            + Child-3W (QKV-level MoE routing)
  Jalur 3 (Retrieval):     MoE + Engram Memory + Expert Choice
                            + S'MoRE + Symbolic-MoE + AuxFreeMoE
                            + ∞-MoE + MoHGE (Heterogeneous Grouped Experts)

Router:  Adaptive (BiasRouter + ThinkingToggle + Symbolic-MoE), GRPO/DAPO-trained.
         + Router ↔ Expert Co-Evolution (Evoformer Level 5)
"""

__version__ = "1.0.0"
__author__ = "Losion Contributors"
__license__ = "MIT"
