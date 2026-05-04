"""
Losion Early Exit — Adaptive computation via pathway-level early exit.

Provides PathwayEarlyExit, which allows the Tri-Jalur router to skip
computation in pathways that are not needed for a given token. This
dramatically reduces compute during both training and inference.

The key insight: not every token needs all three pathways (SSM, Attention,
MoE). Simple tokens (common words, punctuation) can be processed with
just SSM, while complex tokens (reasoning, rare words) need all three.

Credits:
  - Universal Transformers: Dehghani et al., arXiv:1807.03819 (2019)
  - Adaptive Computation Time: Graves, arXiv:1603.08983 (2016)
  - DeepSeek-V3: Auxiliary-loss-free load balancing, arXiv:2412.19437 (2024)
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PathwayEarlyExit(nn.Module):
    """Per-pathway early exit with adaptive thresholds.

    For each token, the router outputs routing weights for the three
    pathways (SSM, Attention, MoE). If a pathway's weight falls below
    a threshold, that pathway is skipped entirely — no forward pass,
    no gradient computation, no memory allocation.

    The threshold is adaptive: it starts conservative (all pathways active)
    and gradually increases during training to allow more skipping.

    Integration with LosionLayerV2:
        In LosionLayerV2.forward(), after computing routing weights:
            if self.early_exit is not None:
                active_mask = self.early_exit(route_weights)
                # Zero out inactive pathways before computation
                route_weights = route_weights * active_mask.float()

    Args:
        n_pathways: Number of pathways (default 3 for Tri-Jalur).
        initial_threshold: Starting threshold (conservative, 0.01).
        max_threshold: Maximum threshold (aggressive, 0.15).
        warmup_steps: Steps before threshold starts increasing.
        ramp_steps: Steps over which threshold increases from initial to max.
    """

    def __init__(
        self,
        n_pathways: int = 3,
        initial_threshold: float = 0.01,
        max_threshold: float = 0.15,
        warmup_steps: int = 1000,
        ramp_steps: int = 5000,
    ) -> None:
        super().__init__()
        self.n_pathways = n_pathways
        self.initial_threshold = initial_threshold
        self.max_threshold = max_threshold
        self.warmup_steps = warmup_steps
        self.ramp_steps = ramp_steps

        # Learnable per-pathway thresholds (initialized to initial_threshold)
        self.threshold_logits = nn.Parameter(
            torch.full((n_pathways,), self._threshold_to_logit(initial_threshold))
        )

        # Step counter for adaptive threshold schedule
        self.register_buffer("_step", torch.tensor(0, dtype=torch.long))

    def forward(
        self,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Compute active pathway mask based on routing weights.

        Args:
            routing_weights: Per-token pathway weights (batch, seq_len, n_pathways).

        Returns:
            Boolean mask (batch, seq_len, n_pathways) — True = active.
        """
        # Get current threshold (adaptive schedule)
        threshold = self.get_current_threshold()

        # Active if weight >= threshold
        active_mask = routing_weights >= threshold.unsqueeze(0).unsqueeze(0)

        # Ensure at least one pathway is active per token
        any_active = active_mask.any(dim=-1, keepdim=True)
        # If no pathway active, activate the highest-weight one
        if not any_active.all():
            max_pathway = routing_weights.argmax(dim=-1, keepdim=True)
            force_active = torch.zeros_like(active_mask).scatter_(
                -1, max_pathway, True
            )
            active_mask = torch.where(any_active, active_mask, force_active)

        return active_mask

    def get_current_threshold(self) -> torch.Tensor:
        """Get the current adaptive threshold per pathway.

        During warmup: returns initial_threshold (conservative).
        During ramp: linearly increases from initial to max.
        After ramp: uses learned threshold (constrained to [initial, max]).

        Returns:
            Threshold tensor (n_pathways,).
        """
        step = self._step.item()

        if step < self.warmup_steps:
            # Warmup: fixed conservative threshold
            return torch.full(
                (self.n_pathways,), self.initial_threshold,
                device=self.threshold_logits.device,
            )

        # Get learned threshold via sigmoid (bounded to [initial, max])
        learned = torch.sigmoid(self.threshold_logits) * self.max_threshold
        learned = torch.clamp(learned, min=self.initial_threshold, max=self.max_threshold)

        if step < self.warmup_steps + self.ramp_steps:
            # Ramp: blend between initial and learned
            progress = (step - self.warmup_steps) / self.ramp_steps
            initial = torch.full_like(learned, self.initial_threshold)
            return initial * (1 - progress) + learned * progress

        return learned

    def step(self) -> None:
        """Increment the step counter for adaptive threshold schedule."""
        self._step.add_(1)

    @staticmethod
    def _threshold_to_logit(threshold: float) -> float:
        """Convert threshold to logit for parameterization."""
        import math
        if threshold <= 0:
            return -10.0
        if threshold >= 1:
            return 10.0
        return math.log(threshold / (1 - threshold))

    def compute_exit_rate(self, routing_weights: torch.Tensor) -> torch.Tensor:
        """Compute the fraction of tokens that skip each pathway.

        Useful for monitoring and logging.

        Args:
            routing_weights: (batch, seq_len, n_pathways).

        Returns:
            Skip rate per pathway (n_pathways,).
        """
        with torch.no_grad():
            active_mask = self.forward(routing_weights)
            skip_rate = 1.0 - active_mask.float().mean(dim=(0, 1))
        return skip_rate
