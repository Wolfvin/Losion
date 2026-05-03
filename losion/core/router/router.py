"""
AdaptiveRouter — Router adaptif untuk Tri-Jalur Architecture.

Menggabungkan BiasRouter (DeepSeek-style load balancing) dengan
ThinkingToggle (Qwen3-style complexity detection) untuk menghasilkan
routing weights yang optimal untuk setiap token.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bias_router import BiasRouter, PathwayRoutingInfo
from .thinking_toggle import (
    ThinkingAssessment,
    ThinkingMode,
    ThinkingToggle,
    TaskType,
)


@dataclass
class AdaptiveRoutingOutput:
    """Output lengkap dari AdaptiveRouter."""

    routing_weights: torch.Tensor  # [batch, seq, 3] — bobot per jalur
    routing_info: PathwayRoutingInfo  # Detail routing dari BiasRouter
    thinking_assessment: ThinkingAssessment  # Assessment dari ThinkingToggle
    adjusted_weights: torch.Tensor  # [batch, seq, 3] — weights setelah thinking adjustment
    depth_multiplier: float  # Depth multiplier dari thinking toggle
    pathway_labels: list  # Label jalur: ["sequential", "reasoning", "factual"]


class AdaptiveRouter(nn.Module):
    """
    Adaptive Router — menggabungkan BiasRouter + ThinkingToggle.

    Mekanisme 2-tahap:
    1. Classification: analisis input -> klasifikasi (sequential/reasoning/factual)
    2. Allocation: distribusi komputasi ke jalur dengan soft routing

    Thinking toggle mengontrol depth:
    - Non-thinking: Jalur 1 dominan, minimal Jalur 2
    - Thinking: Jalur 2+3 diaktifkan penuh

    Output: routing_weights [batch, seq, 3] untuk setiap token

    Args:
        d_model: Model dimension
        num_pathways: Number of pathways (default 3)
        top_k_pathways: Active pathways per token (default 2)
        bias_lr: Learning rate for bias update (default 0.01)
        thinking_threshold: Complexity threshold for thinking (default 0.5)
    """

    # Label untuk 3 jalur
    PATHWAY_LABELS = ["sequential", "reasoning", "factual"]

    def __init__(
        self,
        d_model: int,
        num_pathways: int = 3,
        top_k_pathways: int = 2,
        bias_lr: float = 0.01,
        thinking_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_pathways = num_pathways
        self.top_k_pathways = min(top_k_pathways, num_pathways)

        # === Bias-Based Router ===
        self.bias_router = BiasRouter(
            d_model=d_model,
            num_pathways=num_pathways,
            top_k_pathways=top_k_pathways,
            bias_lr=bias_lr,
        )

        # === Thinking Toggle ===
        self.thinking_toggle = ThinkingToggle(
            d_model=d_model,
            threshold=thinking_threshold,
        )

        # === Thinking-Weight Adjustment Layer ===
        # Modifikasi routing weights berdasarkan thinking assessment
        # Fix: Include task_type_probs + thinking_score in adjuster input so that
        # task_classifier AND context_integrator receive gradients through the adjuster.
        # Input: routing_weights(3) + complexity_signal(1) + thinking_score(1) + task_type_probs(5)
        from .thinking_toggle import TaskType
        num_task_types = len(TaskType)
        self.thinking_adjuster = nn.Sequential(
            nn.Linear(num_pathways + 1 + 1 + num_task_types, num_pathways * 2),
            nn.SiLU(),
            nn.Linear(num_pathways * 2, num_pathways),
        )

        # === Initial pathway priors ===
        # Sebelum melihat input, jalur mana yang lebih mungkin?
        # [sequential, reasoning, factual]
        self.register_buffer(
            "pathway_priors", torch.tensor([0.4, 0.3, 0.3])
        )

    def forward(
        self, x: torch.Tensor
    ) -> AdaptiveRoutingOutput:
        """
        Forward pass AdaptiveRouter.

        Proses 2-tahap:
        1. ThinkingToggle: assess complexity → thinking/non-thinking
        2. BiasRouter: compute routing weights → adjust with thinking info

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            AdaptiveRoutingOutput dengan semua informasi routing
        """
        if x.dim() != 3:
            raise ValueError(
                f"Input harus 3D [batch, seq, d_model], mendapat {x.dim()}D"
            )

        batch_size, seq_len, _ = x.shape
        device = x.device
        dtype = x.dtype

        # === Tahap 1: Classification via ThinkingToggle ===
        thinking_assessment = self.thinking_toggle(x)

        # === Tahap 2: Allocation via BiasRouter ===
        routing_weights, routing_info = self.bias_router(x)

        # === Adjust weights berdasarkan thinking assessment ===
        adjusted_weights = self._adjust_for_thinking(
            routing_weights, thinking_assessment
        )

        # === Hitung depth multiplier ===
        depth_multiplier = thinking_assessment.depth_multiplier

        return AdaptiveRoutingOutput(
            routing_weights=routing_weights,
            routing_info=routing_info,
            thinking_assessment=thinking_assessment,
            adjusted_weights=adjusted_weights,
            depth_multiplier=depth_multiplier,
            pathway_labels=self.PATHWAY_LABELS,
        )

    def _adjust_for_thinking(
        self,
        routing_weights: torch.Tensor,
        assessment: ThinkingAssessment,
    ) -> torch.Tensor:
        """
        Sesuaikan routing weights berdasarkan thinking assessment.

        Non-thinking: tingkatkan Jalur 1 (sequential), kurangi Jalur 2+3
        Thinking: tingkatkan Jalur 2+3, kurangi Jalur 1

        Args:
            routing_weights: [batch, seq, num_pathways]
            assessment: ThinkingAssessment

        Returns:
            Adjusted routing weights [batch, seq, num_pathways]
        """
        batch_size, seq_len, _ = routing_weights.shape
        device = routing_weights.device
        dtype = routing_weights.dtype

        # Complexity score per token
        complexity = assessment.complexity_score  # [batch, seq]
        thinking_signal = complexity.unsqueeze(-1)  # [batch, seq, 1]

        # v1.1 Fix: Include thinking_score (from context_integrator) so it
        # receives gradients through the adjuster. Broadcast per-batch score
        # to per-token: [batch] → [batch, 1, 1] → [batch, seq, 1]
        thinking_score_signal = assessment.thinking_score  # [batch]
        if thinking_score_signal is not None:
            thinking_score_signal = thinking_score_signal.unsqueeze(-1).unsqueeze(-1)  # [batch, 1, 1]
            thinking_score_signal = thinking_score_signal.expand(batch_size, seq_len, 1)  # [batch, seq, 1]
        else:
            thinking_score_signal = torch.zeros(batch_size, seq_len, 1, device=device, dtype=dtype)

        # Concat routing weights + thinking signal + thinking score + task type probs
        # Fix: Include task_type_probs AND thinking_score so task_classifier
        # AND context_integrator receive gradients through the thinking_adjuster.
        adjuster_input = torch.cat(
            [routing_weights, thinking_signal, thinking_score_signal, assessment.task_type_probs], dim=-1
        )  # [batch, seq, num_pathways + 1 + 1 + num_task_types]

        # Hitung adjustment
        adjustment = self.thinking_adjuster(adjuster_input)  # [batch, seq, num_pathways]

        # Combine: base weights + adjustment (residual-style)
        adjusted = routing_weights + 0.1 * adjustment  # Scale factor kecil

        # Re-normalize agar sum = 1
        adjusted = F.softmax(adjusted, dim=-1)

        # Apply thinking mode constraints
        if assessment.mode == ThinkingMode.NON_THINKING:
            # Non-thinking: tingkatkan Jalur 1 (sequential)
            # Jalur 1 = index 0
            jalur1_boost = torch.tensor(
                [0.3, -0.15, -0.15], device=device, dtype=dtype
            )
            adjusted = adjusted + jalur1_boost.unsqueeze(0).unsqueeze(0)
            adjusted = F.softmax(adjusted, dim=-1)
        else:
            # Thinking: tingkatkan Jalur 2 (reasoning) dan 3 (factual)
            jalur23_boost = torch.tensor(
                [-0.15, 0.15, 0.15], device=device, dtype=dtype
            )
            adjusted = adjusted + jalur23_boost.unsqueeze(0).unsqueeze(0)
            adjusted = F.softmax(adjusted, dim=-1)

        return adjusted

    def update_bias(self) -> None:
        """
        Update routing bias berdasarkan running statistics.

        Dipanggil secara periodik selama training.
        """
        self.bias_router.update_bias()

    def set_force_thinking(
        self, mode: Optional[ThinkingMode]
    ) -> None:
        """
        Force thinking mode untuk semua input.

        Set None untuk kembali ke automatic detection.

        Args:
            mode: ThinkingMode untuk force, atau None untuk auto
        """
        self.thinking_toggle.set_force_mode(mode)

    def set_thinking_threshold(self, threshold: float) -> None:
        """
        Update thinking threshold.

        Args:
            threshold: Nilai threshold (0.0 - 1.0)
        """
        self.thinking_toggle.set_threshold(threshold)

    def get_pathway_summary(
        self, output: AdaptiveRoutingOutput
    ) -> dict:
        """
        Ringkasan distribusi routing untuk monitoring.

        Args:
            output: Output dari forward

        Returns:
            Dictionary berisi ringkasan routing
        """
        with torch.no_grad():
            adjusted = output.adjusted_weights  # [batch, seq, 3]
            mean_weights = adjusted.mean(dim=(0, 1))  # [3]

            return {
                "pathway_labels": self.PATHWAY_LABELS,
                "mean_weights": {
                    label: mean_weights[i].item()
                    for i, label in enumerate(self.PATHWAY_LABELS)
                },
                "thinking_mode": output.thinking_assessment.mode.value,
                "dominant_task": output.thinking_assessment.dominant_task.value,
                "depth_multiplier": output.depth_multiplier,
                "thinking_confidence": output.thinking_assessment.confidence,
                "complexity_mean": output.thinking_assessment.complexity_score.mean().item(),
            }

    def compute_routing_entropy(
        self, routing_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Hitung entropy dari distribusi routing.

        Entropy rendah = routing yang focused (1-2 jalur dominan)
        Entropy tinggi = routing yang merata (semua jalur aktif)

        Args:
            routing_weights: [batch, seq, num_pathways]

        Returns:
            Mean entropy scalar
        """
        with torch.no_grad():
            # Hindari log(0)
            clamped = routing_weights.clamp(min=1e-8)
            entropy = -(clamped * clamped.log()).sum(dim=-1)  # [batch, seq]
            max_entropy = torch.log(
                torch.tensor(self.num_pathways, dtype=routing_weights.dtype)
            )
            # Normalize ke [0, 1]
            normalized_entropy = entropy / max_entropy
            return normalized_entropy.mean()

    def router_params(self) -> list:
        """Return router parameters for separate learning rate scheduling.

        Router gradients are typically much smaller than pathway gradients
        (router grad norm ~0.013 vs SSM ~0.628), so the router needs a
        higher learning rate to learn effectively.

        Returns:
            List of router parameters.
        """
        return list(self.parameters())

    def compute_entropy_regularization(
        self, routing_weights: torch.Tensor, weight: float = 0.01
    ) -> torch.Tensor:
        """Compute entropy regularization to prevent routing collapse.

        Encourages the router to distribute tokens across all pathways
        rather than always selecting one pathway. This prevents the
        tri-jalur architecture from degenerating into a fixed ensemble.

        Args:
            routing_weights: [batch, seq, num_pathways] routing weights.
            weight: Regularization weight (default 0.01).

        Returns:
            Scalar entropy regularization loss (negative entropy, to be minimized).
        """
        clamped = routing_weights.clamp(min=1e-8)
        entropy = -(clamped * clamped.log()).sum(dim=-1)  # [batch, seq]
        max_entropy = torch.log(
            torch.tensor(self.num_pathways, dtype=routing_weights.dtype, device=routing_weights.device)
        )
        normalized_entropy = entropy / max_entropy  # [0, 1]
        # We want to MAXIMIZE entropy, so minimize negative entropy
        return -weight * normalized_entropy.mean()

    def get_param_groups(self, router_lr: float = 1e-3, default_lr: float = 1e-4) -> list:
        """Create parameter groups with separate learning rates for router.

        Router parameters need 5-10x higher learning rate because their
        gradients are typically much smaller than pathway gradients.

        Args:
            router_lr: Learning rate for router parameters.
            default_lr: Default learning rate for non-router parameters.

        Returns:
            List of parameter group dicts for optimizer.
        """
        router_param_ids = {id(p) for p in self.parameters()}
        return [
            {"params": list(self.parameters()), "lr": router_lr},
        ]
