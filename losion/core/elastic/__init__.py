"""
Losion Elastic Inference Module — Matryoshka-style nested models + Attention-preferred LoRA.

Integrates techniques from:
- Gemma 3n / MatFormer: Matryoshka Nested Transformer for elastic deployment
- Allows extracting smaller valid submodels from a single trained weight set

v0.5 additions:
  AttnLoRAConfig   — Configuration for attention-preferred LoRA
  AttnLoRALayer    — Single LoRA adapter layer
  AttnLoRAModel    — Applies attention-preferred LoRA to a Losion model

Modules:
    matryoshka: Nested Transformer for elastic inference at deployment time
    attn_lora: Attention-preferred LoRA with asymmetric rank allocation
"""

from .matryoshka import MatryoshkaLayer, MatryoshkaConfig, ElasticExtractor
from .attn_lora import AttnLoRAConfig, AttnLoRALayer, AttnLoRAModel

__all__ = [
    "MatryoshkaLayer", "MatryoshkaConfig", "ElasticExtractor",
    # Attention-preferred LoRA
    "AttnLoRAConfig", "AttnLoRALayer", "AttnLoRAModel",
]
