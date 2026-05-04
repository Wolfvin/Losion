"""
ThinkingToggle — Deteksi Thinking vs Non-Thinking Mode.

Diadaptasi dari Qwen3: mendeteksi apakah input memerlukan
reasoning mendalam (thinking) atau cukup respons cepat (non-thinking).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ThinkingMode(Enum):
    """Mode operasi model."""

    THINKING = "thinking"
    NON_THINKING = "non_thinking"


class TaskType(Enum):
    """Klasifikasi tipe task."""

    SEQUENTIAL = "sequential"  # Teks sekuensial standar
    REASONING = "reasoning"  # Reasoning mendalam
    FACTUAL = "factual"  # Lookup fakta
    CREATIVE = "creative"  # Generasi kreatif
    CODING = "coding"  # Kode/programming


@dataclass
class ThinkingAssessment:
    """Hasil assessment thinking/non-thinking."""

    mode: ThinkingMode
    complexity_score: torch.Tensor  # [batch, seq] — skor kompleksitas
    task_type_probs: torch.Tensor  # [batch, seq, num_task_types]
    dominant_task: TaskType
    depth_multiplier: float  # Berapa kali lipat depth untuk Jalur 2+3
    confidence: float  # Confidence dalam assessment
    thinking_score: torch.Tensor = None  # [batch] — differentiable thinking score dari context_integrator


class ThinkingToggle(nn.Module):
    """
    Thinking/Non-Thinking Toggle (dari Qwen3).

    Mendeteksi apakah input memerlukan reasoning mendalam (thinking)
    atau cukup dengan respons cepat (non-thinking).

    Thinking mode mengaktifkan:
    - Lebih banyak global attention layers di Jalur 2
    - Lebih banyak MoE experts di Jalur 3
    - Reasoning chain-of-thought internal

    Deteksi berdasarkan:
    - Input complexity score (learned)
    - Task type classification
    - User-specified mode override

    Args:
        d_model: Model dimension
        threshold: Complexity threshold for thinking mode (default 0.5)
    """

    # Jumlah task types yang dikenali
    NUM_TASK_TYPES = len(TaskType)

    def __init__(
        self,
        d_model: int,
        threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.threshold = threshold

        # === Complexity Scorer ===
        # Mengukur seberapa kompleks input token
        self.complexity_scorer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, d_model // 4),
            nn.SiLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),  # Output: [0, 1]
        )

        # === Task Type Classifier ===
        # Klasifikasi jenis task
        self.task_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, self.NUM_TASK_TYPES),
        )

        # === Context Integration ===
        # Menggabungkan complexity + task type untuk keputusan final
        self.context_integrator = nn.Sequential(
            nn.Linear(1 + self.NUM_TASK_TYPES, d_model // 4),
            nn.SiLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )

        # === Depth Calculator ===
        # Menghitung depth multiplier berdasarkan thinking score
        # Non-thinking: depth ~0.3-0.5 (minimal Jalur 2)
        # Thinking: depth ~1.0-2.0 (full Jalur 2+3)
        self.depth_min = 0.3
        self.depth_max = 2.0

        # User-specified override — stored as a buffer so it is
        # included in state_dict and survives save/load.
        # Encoding: -1 = auto (None), 0 = NON_THINKING, 1 = THINKING
        self.register_buffer(
            "_force_mode_code",
            torch.tensor(-1, dtype=torch.long),
            persistent=True,
        )

    def forward(
        self, x: torch.Tensor
    ) -> ThinkingAssessment:
        """
        Assess apakah input memerlukan thinking mode.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            ThinkingAssessment dengan detail keputusan
        """
        if x.dim() != 3:
            raise ValueError(
                f"Input harus 3D [batch, seq, d_model], mendapat {x.dim()}D"
            )

        # 1. Complexity Score
        complexity = self.complexity_scorer(x).squeeze(-1)  # [batch, seq]

        # 2. Task Type Classification
        task_logits = self.task_classifier(x)  # [batch, seq, num_task_types]
        task_probs = F.softmax(task_logits, dim=-1)  # [batch, seq, num_task_types]

        # 3. Aggregate per-sample (mean over sequence)
        mean_complexity = complexity.mean(dim=-1, keepdim=True)  # [batch, 1]
        mean_task_probs = task_probs.mean(dim=1)  # [batch, num_task_types]

        # 4. Context Integration
        context_input = torch.cat(
            [mean_complexity, mean_task_probs], dim=-1
        )  # [batch, 1 + num_task_types]
        thinking_score = self.context_integrator(
            context_input
        ).squeeze(-1)  # [batch]

        # 5. Determine mode
        force_mode = self._get_force_mode()
        if force_mode is not None:
            mode = force_mode
        else:
            # Gunakan rata-rata thinking score
            # NOTE: .item() memutus computational graph, tapi mode
            # hanya digunakan untuk control flow (branching), bukan
            # untuk komputasi tensor. Gradien mengalir melalui
            # thinking_score yang disimpan di ThinkingAssessment.
            avg_score = thinking_score.mean().item()
            mode = (
                ThinkingMode.THINKING
                if avg_score > self.threshold
                else ThinkingMode.NON_THINKING
            )

        # 6. Determine dominant task type
        overall_task_probs = task_probs.mean(dim=(0, 1))  # [num_task_types]
        dominant_task_idx = overall_task_probs.argmax().item()
        dominant_task = list(TaskType)[dominant_task_idx]

        # 7. Calculate depth multiplier
        # NOTE: depth_multiplier adalah Python float untuk logging/monitoring.
        # Gradien mengalir melalui thinking_score tensor yang disimpan
        # di ThinkingAssessment dan digunakan secara differentiable
        # oleh AdaptiveRouter._adjust_for_thinking().
        avg_thinking_score = thinking_score.mean().item()
        if mode == ThinkingMode.THINKING:
            depth_multiplier = (
                self.depth_min
                + (self.depth_max - self.depth_min) * avg_thinking_score
            )
        else:
            depth_multiplier = self.depth_min + 0.2 * avg_thinking_score

        # 8. Confidence
        confidence = 1.0 - abs(avg_thinking_score - self.threshold) / max(
            self.threshold, 1.0 - self.threshold
        )
        confidence = min(max(confidence, 0.0), 1.0)

        return ThinkingAssessment(
            mode=mode,
            complexity_score=complexity,
            task_type_probs=task_probs,
            dominant_task=dominant_task,
            depth_multiplier=depth_multiplier,
            confidence=confidence,
            thinking_score=thinking_score,  # [batch] — differentiable!
        )

    def _get_force_mode(self) -> Optional[ThinkingMode]:
        """Decode the persistent buffer back to a ThinkingMode (or None)."""
        code = int(self._force_mode_code.item())
        if code == -1:
            return None
        elif code == 0:
            return ThinkingMode.NON_THINKING
        elif code == 1:
            return ThinkingMode.THINKING
        else:
            # Corrupted value — reset to auto
            self._force_mode_code.fill_(-1)
            return None

    def set_force_mode(self, mode: Optional[ThinkingMode]) -> None:
        """
        Force mode tertentu, bypass detection.

        Set None untuk kembali ke automatic detection.

        The mode is persisted via a registered buffer so it survives
        ``model.state_dict()`` / ``model.load_state_dict()`` round-trips.

        Args:
            mode: ThinkingMode untuk force, atau None untuk auto
        """
        if mode is None:
            self._force_mode_code.fill_(-1)
        elif mode == ThinkingMode.NON_THINKING:
            self._force_mode_code.fill_(0)
        elif mode == ThinkingMode.THINKING:
            self._force_mode_code.fill_(1)
        else:
            raise ValueError(f"Unknown ThinkingMode: {mode!r}")

    def set_threshold(self, threshold: float) -> None:
        """
        Update complexity threshold.

        Args:
            threshold: Nilai threshold baru (0.0 - 1.0)
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                f"Threshold harus antara 0.0 dan 1.0, mendapat {threshold}"
            )
        self.threshold = threshold

    def get_thinking_mask(
        self, assessment: ThinkingAssessment, seq_len: int
    ) -> torch.Tensor:
        """
        Buat binary mask menandai token mana yang perlu thinking.

        Berguna untuk selective activation di Jalur 2.

        Args:
            assessment: Hasil assessment dari forward
            seq_len: Panjang sequence

        Returns:
            Mask [batch, seq] — 1.0 untuk thinking, 0.0 untuk non-thinking
        """
        mask = (assessment.complexity_score > self.threshold).float()
        return mask

    def get_depth_schedule(
        self, assessment: ThinkingAssessment
    ) -> torch.Tensor:
        """
        Hitung depth schedule per token.

        Token dengan complexity tinggi mendapat depth lebih besar.

        Args:
            assessment: Hasil assessment dari forward

        Returns:
            Depth weights [batch, seq] — nilai antara depth_min dan depth_max
        """
        complexity = assessment.complexity_score  # [batch, seq]

        # Linear interpolation berdasarkan complexity
        depth = (
            self.depth_min
            + (self.depth_max - self.depth_min) * complexity
        )

        # Jika non-thinking mode, clamp ke minimum
        if assessment.mode == ThinkingMode.NON_THINKING:
            depth = depth.clamp(max=self.depth_min + 0.3)

        return depth
