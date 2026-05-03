"""
Losion — Hybrid AI Framework with Tri-Jalur Router Architecture.

Version 0.9.1 — "Puzzle Connected"

v0.9.1 Connectivity Fixes:
  All components now properly interconnect as a unified puzzle:
  - SSM modules: Unified `state`/`initial_state` kwarg handling
  - SSM modules: RoutingMamba 3-tuple return properly unpacked
  - Attention modules: `position_ids`/`position_offset` auto-adapted
  - Attention modules: `past_kv`/`past_key_value` auto-adapted
  - MoE modules: 3-tuple returns normalized to 2-tuple consistently
  - Router: AdaptiveRouter now receives thinking_mode via set_force_thinking()
  - Router: _build_router() now passes correct constructor args
  - Evoformer: Levels 3-5 now wired into LosionModelV2
  - DualMemory: read() method added; write+read cycle completes
  - Training: _unfreeze_pathway() uses correct attribute name
  - Exports: V2 models (LosionModelV2, LosionForCausalLMV2) exported

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

__version__ = "0.9.1"
__author__ = "Losion Contributors"
__license__ = "MIT"
