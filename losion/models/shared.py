"""
Shared components between LosionModel (V1) and LosionModelV2.

V1 and V2 share several identical components that were previously duplicated
across both model files. This module provides shared base classes and mixins
to prevent code drift (audit finding A3.4).

Shared components:
- RMSNorm: Root Mean Square Layer Normalization
- WeightInitMixin: GPT-2 / GPT-NeoX style weight initialization
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Simpler and faster than LayerNorm — normalizes by the RMS of the
    input without subtracting the mean.

    This is the canonical implementation shared by both V1 and V2 models.
    Previously, RMSNorm was defined independently in both losion_model.py
    and losion_model_v2.py, leading to code drift risk.

    Args:
        dim: Normalization dimension.
        eps: Epsilon for numerical stability (default 1e-6).
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization.

        Args:
            x: Input tensor of any shape with last dimension == dim.

        Returns:
            Normalized tensor with same shape.
        """
        dtype = x.dtype
        x = x.float()
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x = x / rms
        return (self.weight * x).to(dtype)


class WeightInitMixin:
    """Mixin providing GPT-2 / GPT-NeoX style weight initialization.

    This mixin standardizes weight initialization across V1 and V2 models.
    Both previously had identical `_init_weights` methods — this mixin
    prevents drift by providing a single source of truth.

    Usage:
        class LosionModel(nn.Module, WeightInitMixin):
            def __init__(self, config):
                ...
                self.apply(lambda m: self._init_weights(m))

    Initialization rules:
    - Embeddings: normal(0, 0.02)
    - Linear layers: normal(0, 0.02 / sqrt(2 * n_layers))
      The sqrt(2 * n_layers) scaling prevents hidden state explosion
      in deep residual networks (GPT-2 paper Section 2.3).
    - Conv1d: normal(0, 0.02 / sqrt(2 * n_layers))
      Consistent with Linear init (not PyTorch default kaiming).
    - Biases: zeros
    """

    n_layers: int  # Must be set by the class using this mixin

    def _init_weights(self, module: nn.Module) -> None:
        """LLM-standard weight initialization.

        Args:
            module: The nn.Module to initialize.
        """
        if isinstance(module, nn.Linear):
            std = 0.02 / math.sqrt(2 * self.n_layers)
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            std = 0.02 / math.sqrt(2 * self.n_layers)
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
