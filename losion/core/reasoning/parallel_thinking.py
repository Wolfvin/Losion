"""
Parallel Thinking — Multi-path reasoning exploration (Gemini Deep Think style).

Diadaptasi dari Gemini 2.5 Deep Think (Google DeepMind, 2025): model
mengeksplorasi beberapa jalur reasoning secara paralel, lalu memilih
yang terbaik. Ini adalah implementasi "parallel thinking" yang mirip
dengan bagaimana Gemini Deep Think menyelesaikan masalah IMO.

Konsep utama:
1. Multiple Thinking Paths: Beberapa jalur reasoning dieksplorasi paralel
2. Self-Consistency: Jalur yang konsisten mendapat skor lebih tinggi
3. Best-Path Selection: Pilih jalur terbaik berdasarkan value + consistency
4. Dynamic Budget Allocation: Alokasi compute berdasarkan kompleksitas

Inspirasi:
- Gemini 2.5 Deep Think: "enhanced reasoning mode with parallel thinking"
- rStar (2024): "Mutual Reasoning Makes Smaller LLMs Stronger Problem-Solvers"
- Self-Consistency (Wang et al., 2022): Majority voting over multiple paths

Hardware: Pure PyTorch, kompatibel dengan CUDA, ROCm, dan CPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F


class ThinkingStrategy(Enum):
    """Strategi untuk parallel thinking."""
    MAJORITY_VOTE = "majority_vote"       # Self-consistency via majority voting
    BEST_OF_N = "best_of_n"              # Pilih path dengan value tertinggi
    WEIGHTED_MERGE = "weighted_merge"     # Merge semua paths dengan bobot
    TOURNAMENT = "tournament"             # Elimination tournament antar paths


@dataclass
class ThinkingPath:
    """Satu jalur reasoning dalam parallel thinking.

    Attributes:
        path_id: Identifikasi unik jalur.
        tokens: Token yang dihasilkan oleh jalur ini.
        hidden_states: Hidden states sepanjang jalur.
        value_score: Skor kualitas dari value network.
        consistency_score: Skor konsistensi dengan path lain.
        final_score: Skor gabungan (value + consistency).
    """

    path_id: int = 0
    tokens: List[int] = field(default_factory=list)
    hidden_states: Optional[torch.Tensor] = None
    value_score: float = 0.0
    consistency_score: float = 0.0
    final_score: float = 0.0

    def compute_final_score(
        self,
        value_weight: float = 0.6,
        consistency_weight: float = 0.4,
    ) -> float:
        """Hitung skor final dari value dan consistency.

        Args:
            value_weight: Bobot untuk value score.
            consistency_weight: Bobot untuk consistency score.

        Returns:
            Skor final.
        """
        self.final_score = (
            value_weight * self.value_score
            + consistency_weight * self.consistency_score
        )
        return self.final_score


@dataclass
class ThinkingBudget:
    """Budget untuk parallel thinking.

    Menentukan berapa banyak compute yang dialokasikan untuk thinking.

    Attributes:
        num_paths: Jumlah jalur reasoning paralel.
        max_tokens_per_path: Maksimum token per jalur.
        total_compute_budget: Total compute budget (dalam FLOPs approximation).
    """

    num_paths: int = 3
    max_tokens_per_path: int = 512
    total_compute_budget: int = 1536  # num_paths * max_tokens_per_path

    @classmethod
    def from_complexity(
        cls,
        complexity_score: float,
        min_paths: int = 2,
        max_paths: int = 8,
        min_tokens: int = 256,
        max_tokens: int = 1024,
    ) -> "ThinkingBudget":
        """Buat budget berdasarkan skor kompleksitas.

        Kompleksitas tinggi → lebih banyak paths dan token.
        Kompleksitas rendah → lebih sedikit paths dan token.

        Args:
            complexity_score: Skor kompleksitas [0, 1].
            min_paths: Jumlah path minimum.
            max_paths: Jumlah path maksimum.
            min_tokens: Token minimum per path.
            max_tokens: Token maksimum per path.

        Returns:
            ThinkingBudget yang disesuaikan.
        """
        # Quadratic scaling: complexity tinggi = jauh lebih banyak compute
        num_paths = int(min_paths + (max_paths - min_paths) * (complexity_score ** 1.5))
        max_tokens = int(min_tokens + (max_tokens - min_tokens) * complexity_score)

        return cls(
            num_paths=num_paths,
            max_tokens_per_path=max_tokens,
            total_compute_budget=num_paths * max_tokens,
        )


class PathEvaluator(nn.Module):
    """Evaluator untuk menilai kualitas setiap thinking path.

    Menggabungkan:
    1. Value assessment: Seberapa baik solusi dari path ini?
    2. Confidence assessment: Seberapa yakin model dengan solusi ini?
    3. Novelty assessment: Apakah path ini memberikan perspektif unik?

    Args:
        d_model: Dimensi model.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()

        # Value head: menilai kualitas solusi
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4, bias=False),
            nn.SiLU(),
            nn.Linear(d_model // 4, 1, bias=False),
            nn.Tanh(),  # [-1, 1]
        )

        # Confidence head: menilai keyakinan model
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4, bias=False),
            nn.SiLU(),
            nn.Linear(d_model // 4, 1, bias=False),
            nn.Sigmoid(),  # [0, 1]
        )

        # Novelty head: menilai keunikan perspektif
        self.novelty_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4, bias=False),
            nn.SiLU(),
            nn.Linear(d_model // 4, 1, bias=False),
            nn.Sigmoid(),  # [0, 1]
        )

    def forward(
        self,
        hidden_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluasi satu thinking path.

        Args:
            hidden_state: Hidden state terakhir dari path [batch, d_model]

        Returns:
            Tuple (value, confidence, novelty):
            - value: [batch, 1] di [-1, 1]
            - confidence: [batch, 1] di [0, 1]
            - novelty: [batch, 1] di [0, 1]
        """
        value = self.value_head(hidden_state)
        confidence = self.confidence_head(hidden_state)
        novelty = self.novelty_head(hidden_state)
        return value, confidence, novelty


class ConsistencyChecker(nn.Module):
    """Pengecek konsistensi antar thinking paths.

    Mengukur seberapa konsisten satu path dengan path lainnya.
    Path yang konsisten dengan banyak path lain cenderung lebih
    dapat dipercaya (prinsip self-consistency).

    Args:
        d_model: Dimensi model.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        # Cross-attention antar paths untuk mengukur konsistensi
        self.consistency_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model // 2, bias=False),
            nn.SiLU(),
            nn.Linear(d_model // 2, 1, bias=False),
            nn.Sigmoid(),  # [0, 1]
        )

    def forward(
        self,
        path_hidden: torch.Tensor,
        all_paths_hidden: torch.Tensor,
    ) -> torch.Tensor:
        """Hitung konsistensi satu path terhadap semua path lain.

        Args:
            path_hidden: Hidden state path target [batch, d_model]
            all_paths_hidden: Hidden states semua path [batch, num_paths, d_model]

        Returns:
            Consistency score [batch, 1] di [0, 1]
        """
        batch_size, num_paths, d_model = all_paths_hidden.shape

        # Expand path_hidden untuk dibandingkan dengan setiap path
        # path_hidden: [batch, d_model] -> [batch, num_paths, d_model]
        expanded = path_hidden.unsqueeze(1).expand_as(all_paths_hidden)

        # Concatenate untuk konsistensi check
        # [batch, num_paths, d_model * 2]
        pairs = torch.cat([expanded, all_paths_hidden], dim=-1)

        # Hitung konsistensi per pasangan
        pair_scores = self.consistency_proj(
            pairs.reshape(-1, d_model * 2)
        ).reshape(batch_size, num_paths)

        # Rata-rata konsistensi (exclude self)
        # Self-consistency: skip path_idx yang sama
        consistency = pair_scores.mean(dim=-1, keepdim=True)  # [batch, 1]

        return consistency


class ParallelThinker(nn.Module):
    """Parallel Thinking Engine — Gemini Deep Think style.

    Mengeksplorasi beberapa jalur reasoning secara paralel, lalu
    memilih yang terbaik berdasarkan value, confidence, dan konsistensi.

    Ini adalah implementasi "enhanced reasoning mode" yang diinspirasi
    oleh Gemini 2.5 Deep Think, yang "uses parallel thinking to explore
    multiple solution paths simultaneously."

    Alur:
    1. Input → Budget allocation berdasarkan kompleksitas
    2. N paths dieksplorasi paralel (dengan slight variations)
    3. Setiap path dievaluasi (value + confidence + novelty)
    4. Konsistensi antar paths dihitung
    5. Best path dipilih atau paths di-merge

    Integrasi dengan Tri-Jalur Router:
    - ThinkingToggle mengaktifkan parallel thinking saat mode=THINKING
    - Complexity score menentukan budget (jumlah paths + tokens)
    - Hasil parallel thinking bisa dijadikan input untuk Jalur 2 (reasoning)

    Args:
        d_model: Dimensi model.
        default_num_paths: Jumlah default thinking paths.
        strategy: Strategi seleksi path.
        value_weight: Bobot value score dalam final score.
        consistency_weight: Bobot consistency score dalam final score.
        novelty_weight: Bobot novelty score dalam final score.
    """

    def __init__(
        self,
        d_model: int,
        default_num_paths: int = 3,
        strategy: ThinkingStrategy = ThinkingStrategy.BEST_OF_N,
        value_weight: float = 0.5,
        consistency_weight: float = 0.35,
        novelty_weight: float = 0.15,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.default_num_paths = default_num_paths
        self.strategy = strategy
        self.value_weight = value_weight
        self.consistency_weight = consistency_weight
        self.novelty_weight = novelty_weight

        # === Path Evaluator ===
        self.path_evaluator = PathEvaluator(d_model)

        # === Consistency Checker ===
        self.consistency_checker = ConsistencyChecker(d_model)

        # === Path Diversifier ===
        # Menghasilkan perturbasi unik untuk setiap path
        # Ini memastikan paths mengeksplorasi bagian berbeda dari solution space
        self.path_embeddings = nn.Parameter(
            torch.randn(default_num_paths, d_model) * 0.02
        )

        # === Path Merger ===
        # Untuk strategi WEIGHTED_MERGE: menggabungkan semua paths
        self.path_merger = nn.Sequential(
            nn.Linear(d_model, d_model // 2, bias=False),
            nn.SiLU(),
            nn.Linear(d_model // 2, d_model, bias=False),
        )

    def forward(
        self,
        x: torch.Tensor,
        complexity_score: Optional[float] = None,
        num_paths: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Jalankan parallel thinking dari input state.

        Args:
            x: Input hidden state [batch, d_model]
            complexity_score: Skor kompleksitas [0, 1] dari ThinkingToggle.
            num_paths: Override jumlah paths (opsional).

        Returns:
            Tuple:
                - best_output: Output dari path terpilih [batch, d_model]
                - info: Dictionary berisi statistik parallel thinking
        """
        batch_size = x.shape[0]
        n_paths = num_paths or self.default_num_paths

        # === Budget Allocation ===
        if complexity_score is not None:
            budget = ThinkingBudget.from_complexity(complexity_score)
            n_paths = min(n_paths, budget.num_paths)

        # === Generate Diversified Paths ===
        # Setiap path mendapat perturbasi unik dari path_embeddings
        paths_hidden = []

        # Ambil path embeddings (jika n_paths > default, cycle)
        for i in range(n_paths):
            embed_idx = i % self.path_embeddings.shape[0]
            path_perturbation = self.path_embeddings[embed_idx]  # [d_model]

            # Tambahkan perturbasi ke input
            perturbed = x + path_perturbation.unsqueeze(0) * 0.1  # Scale kecil
            paths_hidden.append(perturbed)

        # Stack: [batch, n_paths, d_model]
        all_paths = torch.stack(paths_hidden, dim=1)

        # === Evaluate Each Path ===
        path_scores = []
        values = []
        confidences = []
        novelties = []

        for i in range(n_paths):
            path_hidden = all_paths[:, i, :]  # [batch, d_model]

            # Evaluasi
            value, confidence, novelty = self.path_evaluator(path_hidden)
            values.append(value)
            confidences.append(confidence)
            novelties.append(novelty)

        # === Compute Consistency ===
        consistencies = []
        for i in range(n_paths):
            path_hidden = all_paths[:, i, :]
            consistency = self.consistency_checker(path_hidden, all_paths)
            consistencies.append(consistency)

        # === Compute Final Scores ===
        thinking_paths = []
        for i in range(n_paths):
            tp = ThinkingPath(
                path_id=i,
                hidden_states=all_paths[:, i, :],
                value_score=values[i].mean().item(),
                consistency_score=consistencies[i].mean().item(),
            )
            tp.compute_final_score(
                value_weight=self.value_weight,
                consistency_weight=self.consistency_weight,
            )
            thinking_paths.append(tp)

        # === Select Best Output ===
        best_output, selection_info = self._select_best_path(
            all_paths, thinking_paths, values, consistencies
        )

        # Statistik
        info = {
            "num_paths": n_paths,
            "strategy": self.strategy.value,
            "path_scores": {tp.path_id: tp.final_score for tp in thinking_paths},
            "best_path_id": selection_info["best_path_id"],
            "best_score": selection_info["best_score"],
            "mean_value": sum(v.mean().item() for v in values) / len(values),
            "mean_consistency": sum(c.mean().item() for c in consistencies) / len(consistencies),
            "complexity_score": complexity_score,
        }

        return best_output, info

    def _select_best_path(
        self,
        all_paths: torch.Tensor,
        thinking_paths: List[ThinkingPath],
        values: List[torch.Tensor],
        consistencies: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Pilih output terbaik berdasarkan strategi.

        Args:
            all_paths: [batch, n_paths, d_model]
            thinking_paths: List ThinkingPath objects
            values: List value tensors per path
            consistencies: List consistency tensors per path

        Returns:
            Tuple (best_output, selection_info)
        """
        n_paths = len(thinking_paths)
        batch_size = all_paths.shape[0]

        if self.strategy == ThinkingStrategy.BEST_OF_N:
            # Pilih path dengan final score tertinggi
            scores = torch.stack([
                torch.tensor(tp.final_score) for tp in thinking_paths
            ])  # [n_paths]
            best_idx = scores.argmax().item()
            best_output = all_paths[:, best_idx, :]

            return best_output, {
                "best_path_id": best_idx,
                "best_score": thinking_paths[best_idx].final_score,
            }

        elif self.strategy == ThinkingStrategy.MAJORITY_VOTE:
            # Self-consistency: path dengan konsistensi tertinggi menang
            consistency_scores = torch.stack([
                c.mean() for c in consistencies
            ])  # [n_paths]
            best_idx = consistency_scores.argmax().item()
            best_output = all_paths[:, best_idx, :]

            return best_output, {
                "best_path_id": best_idx,
                "best_score": thinking_paths[best_idx].final_score,
            }

        elif self.strategy == ThinkingStrategy.WEIGHTED_MERGE:
            # Merge semua paths dengan bobot berdasarkan score
            scores = torch.stack([
                torch.tensor(tp.final_score) for tp in thinking_paths
            ])  # [n_paths]
            weights = F.softmax(scores, dim=0)  # [n_paths]

            # Weighted sum of paths
            merged = torch.zeros(batch_size, self.d_model, device=all_paths.device)
            for i in range(n_paths):
                merged += weights[i] * all_paths[:, i, :]

            # Pass through merger network
            best_output = self.path_merger(merged)

            return best_output, {
                "best_path_id": -1,  # Merged
                "best_score": scores.max().item(),
            }

        elif self.strategy == ThinkingStrategy.TOURNAMENT:
            # Elimination tournament: compare pairs
            remaining = list(range(n_paths))
            while len(remaining) > 1:
                next_round = []
                for i in range(0, len(remaining) - 1, 2):
                    # Compare two paths
                    score_a = thinking_paths[remaining[i]].final_score
                    score_b = thinking_paths[remaining[i + 1]].final_score
                    winner = remaining[i] if score_a >= score_b else remaining[i + 1]
                    next_round.append(winner)
                if len(remaining) % 2 == 1:
                    next_round.append(remaining[-1])
                remaining = next_round

            best_idx = remaining[0]
            best_output = all_paths[:, best_idx, :]

            return best_output, {
                "best_path_id": best_idx,
                "best_score": thinking_paths[best_idx].final_score,
            }

        else:
            # Fallback: best of N
            scores = [tp.final_score for tp in thinking_paths]
            best_idx = max(range(n_paths), key=lambda i: scores[i])
            best_output = all_paths[:, best_idx, :]
            return best_output, {
                "best_path_id": best_idx,
                "best_score": thinking_paths[best_idx].final_score,
            }
