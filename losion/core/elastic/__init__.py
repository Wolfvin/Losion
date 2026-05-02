"""
Losion Elastic Inference Module — Matryoshka-style nested models.

Integrates techniques from:
- Gemma 3n / MatFormer: Matryoshka Nested Transformer for elastic deployment
- Allows extracting smaller valid submodels from a single trained weight set

Modules:
    matryoshka: Nested Transformer for elastic inference at deployment time
"""

from .matryoshka import MatryoshkaLayer, MatryoshkaConfig, ElasticExtractor

__all__ = [
    "MatryoshkaLayer", "MatryoshkaConfig", "ElasticExtractor",
]
