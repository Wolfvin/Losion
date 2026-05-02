"""
Losion Core — Tri-Jalur Architecture Modules.

Subpackages:
  ssm          — Jalur 1: State Space Models (Mamba-2, RWKV-7, DeltaNet, Liquid SSM)
  attention    — Jalur 2: Attention + Compression (MLA, iRoPE, Lightning Attention, Shared)
  retrieval    — Jalur 3: Retrieval MoE (Expert Choice, Heterogeneous, Matryoshka, Gradient-routed)
  router       — Adaptive Router (BiasRouter + ThinkingToggle)
  reasoning    — Reasoning modules (MCTS, Neuro-symbolic, Parallel Thinking)
  elastic      — Elastic capacity (Matryoshka dimensions)
  output       — Output modules (Flow Matching, Diffusion Refinement, Speculative Decoder)
  quantization — Quantization (BitNet 1.58-bit, FP8 training)
  nas          — Neural Architecture Search (DARTS-style layer search)
  recurrent    — Recurrent-Depth Transformer (LTI stability, ACT, Depth LoRA)
"""

from losion.core import ssm, attention, retrieval, router, reasoning, elastic, output, quantization, nas, recurrent

__all__ = [
    "ssm", "attention", "retrieval", "router",
    "reasoning", "elastic", "output",
    "quantization", "nas", "recurrent",
]
