"""
Losion Core — Recurrent-Depth Transformer (RDT) Modules.

Implements depth-recurrent computation with adaptive halting and stable
state injection, based on the OpenMythos reconstruction of Claude Mythos
architecture.

Submodules:
  rdt — Recurrent-Depth Transformer with LTI stability, ACT, Depth LoRA

References:
  - Universal Transformers (Dehghani et al., 2019, arXiv:1807.03819)
  - OpenMythos (github.com/kyegomez/OpenMythos)
  - Relaxed Recursive Transformers (Bae et al., 2024, arXiv:2410.20672)
  - Reasoning with Latent Thoughts (Saunshi et al., 2025, arXiv:2502.17416)
  - COCONUT (arXiv:2412.06769)
  - Loop, Think, & Generalize (arXiv:2604.07822)
  - Parcae Scaling Laws (arXiv:2604.12946)
"""

from losion.core.recurrent.rdt import (
    AdaptiveComputationTime,
    DepthLoRA,
    LoopIndexEmbedding,
    LTIStableInjection,
    RecurrentDepthAuxInfo,
    RecurrentDepthBlock,
)

__all__ = [
    "LTIStableInjection",
    "AdaptiveComputationTime",
    "LoopIndexEmbedding",
    "DepthLoRA",
    "RecurrentDepthBlock",
    "RecurrentDepthAuxInfo",
]
