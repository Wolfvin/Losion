"""
Losion — Hybrid AI Framework with Tri-Jalur Router Architecture.

Version 0.4.0 — "Lightning & Liquid"

Losion combines three complementary computational pathways into a single
adaptive architecture:

  Jalur 1 (SSM):           Mamba-2 SSD + RWKV-7 WKV + Gated DeltaNet
                            (4:1:1 interleaving, Liquid SSM in v0.4)
  Jalur 2 (Attention):     MLA + iRoPE + Pairformer + Lightning Attention
                            (8x KV compression, O(1) inference in v0.4)
  Jalur 3 (Retrieval):     MoE + Engram Memory + Expert Choice
                            (Heterogeneous + Matryoshka MoE in v0.4)

Router:  Bias-based, aux-loss-free, GRPO-trained, Thinking Toggle.

v0.4 Upgrades (12 total):
  HIGH:   Lightning Attention, Parallel-head mode (1B), BitNet 1.58-bit
  MEDIUM: Heterogeneous MoE, Matryoshka MoE, Gradient-routed MoE,
          FP8 training, Post-training NAS
  LOW:    Shared attention (Zamba2-style), MTP speculative decoding,
          Asymmetric MoE placement
  LONG:   Liquid SSM variant (adaptive compute depth)
"""

__version__ = "0.5.0"
__author__ = "Losion Contributors"
__license__ = "MIT"

# Agent Layer — autonomous agent capabilities on top of the model
# Available as `losion.agent` — separate from the neural architecture
from losion import agent
