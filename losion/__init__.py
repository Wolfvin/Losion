"""
Losion — Hybrid AI Framework with Tri-Jalur Router Architecture.

Version 0.6.0 — "Mythos & Mamba"

Losion combines three complementary computational pathways into a single
adaptive architecture:

  Jalur 1 (SSM):           Mamba-2 SSD + RWKV-7 WKV + Gated DeltaNet
                            + Mamba-3 (v0.6) + Routing Mamba (v0.6)
  Jalur 2 (Attention):     MLA + iRoPE + Lightning Attention
                            + Gated Attention (v0.6) + MoBA (v0.6)
  Jalur 3 (Retrieval):     MoE + Engram Memory + Expert Choice
                            + S'MoRE (v0.6) + Symbolic-MoE (v0.6)

Router:  Bias-based, aux-loss-free, GRPO-trained, Thinking Toggle.

v0.6 Upgrades — "Mythos & Mamba" (8 new components):
  RECURRENT:  Recurrent-Depth Transformer (RDT) + LTI-Stable + ACT
              (from OpenMythos / Claude Mythos reconstruction)
  ATTENTION:  Gated Attention (Qwen, NeurIPS '25 Best Paper)
              MoBA - Mixture of Block Attention (Moonshot AI, NeurIPS '25)
  SSM:        Mamba-3 SSD (half state, inference-first)
              Routing Mamba (RoM) - MoE over SSM projections (Microsoft, NeurIPS '25)
  MoE:        S'MoRE - Shared sub-tree expert composition (Meta, NeurIPS '25)
              Symbolic-MoE - Skill-based discrete routing
  TRAINING:   LLM-JEPA - Future latent state prediction (JEPA for LLMs)
"""

__version__ = "0.6.0"
__author__ = "Losion Contributors"
__license__ = "MIT"

# Agent Layer — autonomous agent capabilities on top of the model
# Available as `losion.agent` — separate from the neural architecture
from losion import agent
