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
    """Output lengkap dari AdaptiveRouter.

    v1.7.0: depth_multiplier sekarang torch.Tensor [batch] (bukan float).
    """

    routing_weights: torch.Tensor  # [batch, seq, 3] — bobot per jalur
    routing_info: PathwayRoutingInfo  # Detail routing dari BiasRouter
    thinking_assessment: ThinkingAssessment  # Assessment dari ThinkingToggle
    adjusted_weights: torch.Tensor  # [batch, seq, 3] — weights setelah thinking adjustment
    depth_multiplier: torch.Tensor  # [batch] — differentiable! Depth multiplier dari thinking toggle
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
        self.thinking_adjuster = nn.Sequential(
            nn.Linear(num_pathways + 1, num_pathways * 2),
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

        # === Hitung depth multiplier (tensor sekarang, bukan float) ===
        depth_multiplier = thinking_assessment.depth_multiplier
        # v1.7.0: depth_multiplier adalah tensor [batch]. Untuk AdaptiveRoutingOutput
        # yang menyimpan scalar, ambil mean. Untuk routing langsung, gunakan tensor.

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

        v1.6.1 fix: Menggunakan thinking_score tensor secara differentiable,
        bukan Python float. Menghapus double softmax yang melemahkan boost.
        Gradien mengalir: context_integrator → thinking_score →
        thinking_signal → thinking_adjuster → adjusted_weights.

        Args:
            routing_weights: [batch, seq, num_pathways]
            assessment: ThinkingAssessment

        Returns:
            Adjusted routing weights [batch, seq, num_pathways]
        """
        batch_size, seq_len, _ = routing_weights.shape
        device = routing_weights.device
        dtype = routing_weights.dtype

        # Use thinking_score tensor (differentiable) instead of Python float
        # thinking_score: [batch] — dari context_integrator
        if assessment.thinking_score is not None:
            # Expand ke [batch, seq, 1] untuk broadcasting
            thinking_signal = assessment.thinking_score.unsqueeze(1).unsqueeze(2)
            thinking_signal = thinking_signal.expand(-1, seq_len, -1)
        else:
            # Fallback ke complexity_score jika thinking_score tidak tersedia
            complexity = assessment.complexity_score  # [batch, seq]
            thinking_signal = complexity.unsqueeze(-1)  # [batch, seq, 1]

        # Concat routing weights + thinking signal
        adjuster_input = torch.cat(
            [routing_weights, thinking_signal], dim=-1
        )  # [batch, seq, num_pathways + 1]

        # Hitung adjustment via learned MLP (differentiable)
        adjustment = self.thinking_adjuster(adjuster_input)  # [batch, seq, num_pathways]

        # Combine: base weights + adjustment (residual-style)
        # Scale factor kecil agar adjustment tidak mendominasi
        adjusted = routing_weights + 0.1 * adjustment

        # Re-normalize agar sum = 1 (single softmax, bukan double)
        adjusted = F.softmax(adjusted, dim=-1)

        # Apply thinking mode bias — menggunakan additive bias pada logits
        # sebelum softmax akhir, bukan setelah softmax (double softmax melemahkan efek)
        #
        # Catatan: mode adalah Python enum (control flow), ini OK karena
        # tidak memutus gradient. Gradien mengalir melalui thinking_signal
        # yang digunakan oleh thinking_adjuster MLP.
        if assessment.mode == ThinkingMode.NON_THINKING:
            # Non-thinking: bias ke Jalur 1 (sequential)
            boost = torch.tensor(
                [0.3, -0.15, -0.15], device=device, dtype=dtype
            )
            adjusted = adjusted + boost.unsqueeze(0).unsqueeze(0)
            # Single renormalization
            adjusted = adjusted / (adjusted.sum(dim=-1, keepdim=True) + 1e-8)
        else:
            # Thinking: bias ke Jalur 2 (reasoning) dan 3 (factual)
            boost = torch.tensor(
                [-0.15, 0.15, 0.15], device=device, dtype=dtype
            )
            adjusted = adjusted + boost.unsqueeze(0).unsqueeze(0)
            # Single renormalization
            adjusted = adjusted / (adjusted.sum(dim=-1, keepdim=True) + 1e-8)

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

        v1.7.0: Handles tensor depth_multiplier and confidence.

        Args:
            output: Output dari forward

        Returns:
            Dictionary berisi ringkasan routing
        """
        with torch.no_grad():
            adjusted = output.adjusted_weights  # [batch, seq, 3]
            mean_weights = adjusted.mean(dim=(0, 1))  # [3]

            # Handle tensor depth_multiplier and confidence
            dm = output.depth_multiplier
            conf = output.thinking_assessment.confidence
            dm_val = dm.mean().item() if isinstance(dm, torch.Tensor) else dm
            conf_val = conf.mean().item() if isinstance(conf, torch.Tensor) else conf

            return {
                "pathway_labels": self.PATHWAY_LABELS,
                "mean_weights": {
                    label: mean_weights[i].item()
                    for i, label in enumerate(self.PATHWAY_LABELS)
                },
                "thinking_mode": output.thinking_assessment.mode.value,
                "dominant_task": output.thinking_assessment.dominant_task.value,
                "depth_multiplier": dm_val,
                "thinking_confidence": conf_val,
                "complexity_mean": output.thinking_assessment.complexity_score.mean().item(),
            }

    def compute_routing_entropy(
        self, routing_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Hitung entropy dari distribusi routing.

        Entropy rendah = routing yang focused (1-2 jalur dominan)
        Entropy tinggi = routing yang merata (semua jalur aktif)

        v1.7.0: TANPA torch.no_grad() agar gradien mengalir kembali
        ke router weights melalui entropy loss. Ini memungkinkan
        entropy regularization benar-benar melatih router.

        Args:
            routing_weights: [batch, seq, num_pathways]

        Returns:
            Mean entropy scalar (differentiable!)
        """
        # Hindari log(0) dengan soft clamp
        clamped = routing_weights.clamp(min=1e-8)
        entropy = -(clamped * clamped.log()).sum(dim=-1)  # [batch, seq]
        max_entropy = torch.log(
            torch.tensor(self.num_pathways, dtype=routing_weights.dtype, device=routing_weights.device)
        )
        # Normalize ke [0, 1]
        normalized_entropy = entropy / max_entropy
        return normalized_entropy.mean()
