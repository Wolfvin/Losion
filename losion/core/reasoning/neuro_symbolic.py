"""
Neuro-Symbolic Verification — Formal verification for reasoning outputs.

Diadaptasi dari AlphaProof & AlphaGeometry 2 (Google DeepMind, 2024-2025):
sistem neuro-symbolic yang menggabungkan neural language model dengan
symbolic reasoning engine untuk memverifikasi dan memperbaiki output
reasoning.

AlphaProof menggunakan pendekatan:
1. Neural model menghasilkan kandidat solusi (intuisi)
2. Symbolic engine memverifikasi kebenaran formal (rigor)
3. Jika verifikasi gagal, neural model diberi feedback untuk koreksi
4. Loop berlanjut hingga solusi terverifikasi atau budget habis

AlphaGeometry 2 menggunakan:
1. Neural model mengusulkan auxiliary constructions (kreativitas)
2. Symbolic engine (DD+AR) melakukan deductive reasoning (ketelitian)
3. Kombinasi menghasilkan solusi yang benar DAN kreatif

Implementasi ini mengadaptasi konsep tersebut untuk LLM:
1. Neural pathway: Losion's Tri-Jalur menghasilkan kandidat output
2. Symbolic verification: Rule-based checker memvalidasi output
3. Feedback loop: Jika gagal, informasi error di-feed kembali
4. Adaptive: Hanya diaktifkan untuk task yang membutuhkan (math, code, logic)

Hardware: Pure PyTorch, kompatibel dengan CUDA, ROCm, dan CPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable, Tuple
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F


class VerificationStatus(Enum):
    """Status verifikasi symbolic."""
    VERIFIED = "verified"           # Output terverifikasi benar
    FAILED = "failed"              # Output gagal verifikasi
    PARTIAL = "partial"            # Sebagian terverifikasi
    UNSURE = "unsure"              # Tidak bisa diverifikasi (out of scope)
    NEEDS_REVISION = "needs_revision"  # Gagal tapi bisa diperbaiki


@dataclass
class VerificationResult:
    """Hasil verifikasi neuro-symbolic.

    Attributes:
        status: Status verifikasi.
        confidence: Tingkat keyakinan [0, 1].
        error_type: Jenis error jika gagal (None jika berhasil).
        error_location: Lokasi error (token index atau span).
        feedback: Feedback untuk koreksi (embedding atau text).
        num_iterations: Jumlah iterasi verifikasi yang dijalankan.
    """

    status: VerificationStatus = VerificationStatus.UNSURE
    confidence: float = 0.0
    error_type: Optional[str] = None
    error_location: Optional[Tuple[int, int]] = None
    feedback: Optional[torch.Tensor] = None
    num_iterations: int = 0


class SymbolicRuleEngine(nn.Module):
    """Engine untuk aturan symbolic verification.

    Mengimplementasikan aturan verifikasi yang bisa dikonfigurasi
    untuk berbagai domain (matematika, kode, logika).

    Aturan diimplementasikan sebagai differentiable functions agar
    bisa di-integrasikan dengan gradient-based training.

    Args:
        d_model: Dimensi model.
        num_rules: Jumlah aturan symbolic yang tersedia.
        rule_dim: Dimensi representasi aturan.
    """

    def __init__(
        self,
        d_model: int,
        num_rules: int = 16,
        rule_dim: int = 128,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_rules = num_rules
        self.rule_dim = rule_dim

        # === Rule Embeddings ===
        # Setiap aturan direpresentasikan sebagai vektor yang bisa di-learn
        self.rule_embeddings = nn.Parameter(
            torch.randn(num_rules, rule_dim) * 0.02
        )

        # === Rule Applicability Network ===
        # Menentukan aturan mana yang applicable untuk input tertentu
        self.rule_selector = nn.Sequential(
            nn.Linear(d_model, rule_dim, bias=False),
            nn.SiLU(),
            nn.Linear(rule_dim, num_rules, bias=False),
        )

        # === Verification Network ===
        # Menilai apakah output memenuhi aturan tertentu
        self.verifier = nn.Sequential(
            nn.Linear(d_model + rule_dim, d_model // 2, bias=False),
            nn.SiLU(),
            nn.Linear(d_model // 2, 1, bias=False),
            nn.Sigmoid(),  # [0, 1] — 1 = verified, 0 = failed
        )

        # === Error Localization ===
        # Mengidentifikasi bagian mana dari output yang bermasalah
        self.error_localizer = nn.Sequential(
            nn.Linear(d_model + rule_dim, d_model // 2, bias=False),
            nn.SiLU(),
            nn.Linear(d_model // 2, 1, bias=False),
            nn.Sigmoid(),  # [0, 1] per token — tinggi = kemungkinan error
        )

        # === Feedback Generator ===
        # Menghasilkan feedback embedding untuk koreksi
        self.feedback_generator = nn.Sequential(
            nn.Linear(d_model + rule_dim, d_model, bias=False),
            nn.SiLU(),
            nn.Linear(d_model, d_model, bias=False),
        )

    def forward(
        self,
        output_hidden: torch.Tensor,
        output_sequence: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Jalankan symbolic verification.

        Args:
            output_hidden: Hidden state output [batch, d_model]
                           (pooled dari sequence).
            output_sequence: Hidden states per token [batch, seq, d_model]
                             (opsional, untuk error localization).

        Returns:
            Tuple (verification_scores, error_locations, feedback):
            - verification_scores: [batch, num_rules] — skor verifikasi per aturan
            - error_locations: [batch, seq] — probabilitas error per token
                               (zeros jika output_sequence tidak diberikan)
            - feedback: [batch, d_model] — feedback embedding untuk koreksi
        """
        batch_size = output_hidden.shape[0]

        # 1. Pilih aturan yang applicable
        rule_logits = self.rule_selector(output_hidden)  # [batch, num_rules]
        rule_probs = F.softmax(rule_logits, dim=-1)  # [batch, num_rules]

        # 2. Hitung verification score per aturan
        # Gabungkan output hidden dengan setiap rule embedding
        verification_scores = torch.zeros(
            batch_size, self.num_rules, device=output_hidden.device
        )

        for r in range(self.num_rules):
            rule_embed = self.rule_embeddings[r]  # [rule_dim]
            # Expand untuk batch
            rule_input = torch.cat(
                [
                    output_hidden,
                    rule_embed.unsqueeze(0).expand(batch_size, -1),
                ],
                dim=-1,
            )  # [batch, d_model + rule_dim]
            verification_scores[:, r] = self.verifier(rule_input).squeeze(-1)

        # 3. Error localization (jika sequence tersedia)
        if output_sequence is not None:
            seq_len = output_sequence.shape[1]
            # Gunakan rule dengan skor terendah sebagai indikator error
            worst_rule_idx = verification_scores.argmin(dim=-1)  # [batch]
            worst_rule_embed = self.rule_embeddings[worst_rule_idx]  # [batch, rule_dim]

            error_input = torch.cat(
                [
                    output_sequence,
                    worst_rule_embed.unsqueeze(1).expand(-1, seq_len, -1),
                ],
                dim=-1,
            )  # [batch, seq, d_model + rule_dim]

            error_locations = self.error_localizer(
                error_input.reshape(-1, self.d_model + self.rule_dim)
            ).reshape(batch_size, seq_len)  # [batch, seq]
        else:
            error_locations = torch.zeros(
                batch_size, 1, device=output_hidden.device
            )

        # 4. Generate feedback
        worst_rule_embed_avg = torch.stack([
            self.rule_embeddings[verification_scores[b].argmin()]
            for b in range(batch_size)
        ])  # [batch, rule_dim]

        feedback_input = torch.cat(
            [output_hidden, worst_rule_embed_avg], dim=-1
        )
        feedback = self.feedback_generator(feedback_input)  # [batch, d_model]

        return verification_scores, error_locations, feedback


class NeuroSymbolicVerifier(nn.Module):
    """Neuro-Symbolic Verifier — AlphaProof-style verification layer.

    Menggabungkan:
    1. Neural pathway: menghasilkan kandidat output
    2. Symbolic verification: memvalidasi output
    3. Feedback loop: memberikan koreksi jika gagal

    Hanya diaktifkan untuk task yang membutuhkan verifikasi:
    - Matematika (persamaan, bukti)
    - Kode (syntax, logika)
    - Penalaran logis (syllogisme, kontradiksi)

    Integrasi dengan Tri-Jalur:
    - Berada di akhir pipeline, setelah output layer
    - Bisa dimatikan untuk task kreatif (generation, translation)
    - Feedback bisa di-feed kembali ke Jalur 2 untuk koreksi

    Args:
        d_model: Dimensi model.
        num_rules: Jumlah aturan symbolic.
        max_revision_iterations: Maksimum iterasi revisi.
        verification_threshold: Threshold untuk status VERIFIED.
    """

    def __init__(
        self,
        d_model: int,
        num_rules: int = 16,
        max_revision_iterations: int = 3,
        verification_threshold: float = 0.8,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_revision_iterations = max_revision_iterations
        self.verification_threshold = verification_threshold

        # === Symbolic Rule Engine ===
        self.rule_engine = SymbolicRuleEngine(d_model, num_rules)

        # === Revision Network ===
        # Menggabungkan original output + feedback untuk koreksi
        self.revision_network = nn.Sequential(
            nn.Linear(d_model * 2, d_model, bias=False),
            nn.SiLU(),
            nn.Linear(d_model, d_model, bias=False),
        )

        # === Confidence Estimator ===
        # Mengestimasi confidence berdasarkan verification scores
        self.confidence_estimator = nn.Sequential(
            nn.Linear(num_rules, num_rules // 2, bias=False),
            nn.SiLU(),
            nn.Linear(num_rules // 2, 1, bias=False),
            nn.Sigmoid(),  # [0, 1]
        )

        # === Task Type Gate ===
        # Menentukan apakah verifikasi diperlukan untuk task ini
        self.task_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 4, bias=False),
            nn.SiLU(),
            nn.Linear(d_model // 4, 1, bias=False),
            nn.Sigmoid(),  # [0, 1] — 1 = needs verification
        )

    def forward(
        self,
        output_hidden: torch.Tensor,
        output_sequence: Optional[torch.Tensor] = None,
        task_type: Optional[str] = None,
        force_verify: bool = False,
    ) -> Tuple[torch.Tensor, VerificationResult]:
        """Verifikasi output menggunakan neuro-symbolic approach.

        Args:
            output_hidden: Hidden state output [batch, d_model]
            output_sequence: Hidden states per token [batch, seq, d_model]
            task_type: Tipe task (opsional, untuk gating).
            force_verify: Force verifikasi meskipun task gate menolak.

        Returns:
            Tuple (revised_output, verification_result):
            - revised_output: [batch, d_model] — output yang mungkin direvisi
            - verification_result: Hasil verifikasi
        """
        batch_size = output_hidden.shape[0]

        # === Check apakah verifikasi diperlukan ===
        gate_value = self.task_gate(output_hidden)  # [batch, 1]
        needs_verification = force_verify or (gate_value > 0.5).any()

        if not needs_verification:
            # Task ini tidak membutuhkan verifikasi
            return output_hidden, VerificationResult(
                status=VerificationStatus.UNSURE,
                confidence=gate_value.mean().item(),
                num_iterations=0,
            )

        # === Iterative Verification ===
        current_output = output_hidden
        best_result = VerificationResult()

        for iteration in range(self.max_revision_iterations):
            # 1. Jalankan symbolic verification
            ver_scores, error_locs, feedback = self.rule_engine(
                current_output, output_sequence
            )
            # ver_scores: [batch, num_rules]
            # error_locs: [batch, seq]
            # feedback: [batch, d_model]

            # 2. Hitung confidence
            confidence = self.confidence_estimator(ver_scores)  # [batch, 1]

            # 3. Tentukan status
            mean_confidence = confidence.mean().item()
            min_rule_score = ver_scores.min(dim=-1).values.mean().item()

            if min_rule_score >= self.verification_threshold:
                status = VerificationStatus.VERIFIED
            elif min_rule_score >= 0.5:
                status = VerificationStatus.PARTIAL
            elif min_rule_score >= 0.3:
                status = VerificationStatus.NEEDS_REVISION
            else:
                status = VerificationStatus.FAILED

            # Update best result
            if mean_confidence > best_result.confidence:
                best_result = VerificationResult(
                    status=status,
                    confidence=mean_confidence,
                    error_type="rule_violation" if status != VerificationStatus.VERIFIED else None,
                    error_location=None,
                    feedback=feedback,
                    num_iterations=iteration + 1,
                )

            # 4. Jika sudah verified, berhenti
            if status == VerificationStatus.VERIFIED:
                break

            # 5. Jika perlu revisi, gabungkan output + feedback
            if status in (VerificationStatus.NEEDS_REVISION, VerificationStatus.FAILED):
                revision_input = torch.cat(
                    [current_output, feedback], dim=-1
                )
                current_output = self.revision_network(revision_input)
            else:
                # PARTIAL: tetap coba revisi tapi dengan residual connection
                revision_input = torch.cat(
                    [current_output, feedback], dim=-1
                )
                revision = self.revision_network(revision_input)
                current_output = current_output + 0.3 * revision  # Soft revision

        return current_output, best_result

    def should_verify(self, x: torch.Tensor) -> torch.Tensor:
        """Tentukan apakah input memerlukan verifikasi.

        Berguna untuk routing di Tri-Jalur: jika memerlukan verifikasi,
        output melewati neuro-symbolic layer sebelum final output.

        Args:
            x: Input hidden state [batch, d_model]

        Returns:
            Gate value [batch, 1] — tinggi = perlu verifikasi
        """
        return self.task_gate(x)
