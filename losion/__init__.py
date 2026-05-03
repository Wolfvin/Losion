"""
Losion — Hybrid AI Framework with Tri-Jalur Router Architecture.

Version 1.0.0 — "Unified & Complete"

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

v0.9 Upgrades — "Architecture Document Realized":
  ATTNRES:      Attention Residuals — learned aggregation replacing fixed residuals
                Block AttnRes (efficient O(N·d) approximation)
                Token-dimension AttnRes + Compression (O(n) with intelligent forgetting)
                Credits: MoonshotAI 2026, arXiv GPQA-Diamond +7.5
  EVOFORMER:    5-Level Evoformer Universal Principle (AlphaFold-inspired)
                Level 1: Inter-Layer Recycling (deep ↔ shallow feedback)
                Level 2: Bidirectional Token Update (later tokens revise earlier)
                Level 3: Decoder ↔ Predict Feedback (refinement loops)
                Level 4: Prediction → Context Recycling (predictions revise context)
                Level 5: Router ↔ Expert Co-Evolution (mutual specialization)
                Credits: Jumper et al., Nature 2021 (Nobel Prize 2024)
  CHILD-3W:     MoE at QKV Level — Router + Child-3W routing
                Multiple independent Wq/Wk/Wv sets with routing
                More granular than standard MoE (representation-level specialization)
  ANCHORED DECODER: Continuous Vector Pipeline + Lightweight Diffusion
                Predict continuous vector (NO softmax) → 2-3 step anchored diffusion
                Disambiguation + Coherence + Evoformer feedback
                Credits: Losion Architecture Document Section 15
  DUAL MEMORY:  Two-Level Memory System (working + long-term)
                Working memory: recent, detailed, direct access (ring buffer)
                Long-term memory: compressed, selective, persistent (AttnRes state)
                Memory consolidation: working → long-term compression
                Credits: Losion Architecture Document Section 11.4
  CONFIG:       AttnResConfig, EvoformerConfig, Child3WConfig,
                AnchoredDecoderConfig, DualMemoryConfig sub-configs
"""

__version__ = "1.0.0"
__author__ = "Losion Contributors"
__license__ = "MIT"
