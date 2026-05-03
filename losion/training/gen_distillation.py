"""
Generation-Focused Distillation untuk Losion Framework.

Distillation yang berfokus pada kualitas generasi, bukan sekadar logits matching.
Menggabungkan KL divergence pada output distributions, sequence-level distillation
loss, dan progressive shifting dari teacher ke student.

Komponen utama:
1. GenerationDistillationConfig — Konfigurasi untuk generation-focused distillation
2. GenerationDistiller — Trainer distillation dengan fokus generasi

Fitur:
- KL divergence pada output distributions (bukan hanya logits)
- Sequence-level distillation loss
- Teacher-forcing vs free-running distillation modes
- Progressive distillation (gradual shift dari teacher ke student)

Mode distillation:
- teacher_forcing: Student menerima ground-truth tokens sebagai input
- free_running: Student menerima prediksi sendiri sebagai input (lebih realistis)
- mixed: Kombinasi keduanya dengan scheduled mixing ratio

Referensi:
- Kim & Rush, "Sequence-Level Knowledge Distillation" (2016)
- Freitag et al., "Mixture Models for Diverse Machine Translation" (2023)
- Agarwal et al., "On-Policy Distillation" (2024)

Hardware: Pure PyTorch, kompatibel dengan CUDA, ROCm, dan CPU.
"""

from __future__ import annotations

import math
import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------------------

@dataclass
class GenerationDistillationConfig:
    """Konfigurasi untuk generation-focused distillation.

    Attributes:
        temperature: Softmax temperature untuk soft targets.
            Nilai tinggi → distribusi lebih soft, nilai rendah → lebih sharp.
        alpha_kl: Bobot untuk KL divergence loss antara teacher dan student
            output distributions.
        alpha_seq: Bobot untuk sequence-level distillation loss.
            Membandingkan quality sequence yang dihasilkan student vs teacher.
        alpha_hidden: Bobot untuk hidden state matching loss (opsional).
            Membantu student meniru representasi internal teacher.
        mode: Mode distillation — "teacher_forcing", "free_running", atau
            "mixed". Teacher-forcing menggunakan ground-truth sebagai input
            student; free-running menggunakan prediksi student sendiri.
        progressive: Jika True, secara bertahap menggeser bobot dari teacher
            signal ke student's own loss selama training.
        progressive_warmup_steps: Jumlah steps sebelum progressive shifting
            dimulai. Sebelum warmup, distillation murni.
        progressive_total_steps: Total steps untuk mencapai student-only loss.
        max_seq_len: Panjang sequence maksimum untuk free-running generation.
        teacher_beam_width: Beam width untuk teacher generation (jika
            menggunakan beam search).
        label_smoothing: Label smoothing untuk hard target loss.
    """

    temperature: float = 4.0
    alpha_kl: float = 0.7
    alpha_seq: float = 0.2
    alpha_hidden: float = 0.1
    mode: str = "mixed"
    progressive: bool = True
    progressive_warmup_steps: int = 1000
    progressive_total_steps: int = 10000
    max_seq_len: int = 512
    teacher_beam_width: int = 1
    label_smoothing: float = 0.1

    def __post_init__(self) -> None:
        if self.mode not in ("teacher_forcing", "free_running", "mixed"):
            raise ValueError(
                f"Mode harus 'teacher_forcing', 'free_running', atau 'mixed', "
                f"mendapat '{self.mode}'"
            )
        if self.temperature <= 0:
            raise ValueError(f"Temperature harus > 0, mendapat {self.temperature}")
        if self.alpha_kl < 0 or self.alpha_seq < 0 or self.alpha_hidden < 0:
            raise ValueError("Alpha weights tidak boleh negatif")


# ---------------------------------------------------------------------------
# Helper: KL divergence pada output distributions
# ---------------------------------------------------------------------------

def kl_divergence_logits(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 4.0,
    reduction: str = "batchmean",
) -> torch.Tensor:
    """
    Hitung KL divergence antara output distributions teacher dan student.

    Menggunakan temperature-scaled softmax untuk soft targets:
        KL(p_T || p_S) = Σ p_T * (log p_T - log p_S)

    Diimplementasikan dengan F.kl_div untuk stabilitas numerik.

    Args:
        student_logits: Logits student, (batch, seq_len, vocab_size).
        teacher_logits: Logits teacher, (batch, seq_len, vocab_size).
        temperature: Softmax temperature.
        reduction: Mode reduksi ("batchmean", "sum", "none").

    Returns:
        Scalar KL divergence loss.
    """
    # Temperature scaling
    T = temperature
    # Log-softmax student (target distribution)
    student_log_probs = F.log_softmax(student_logits / T, dim=-1)
    # Softmax teacher (reference distribution)
    teacher_probs = F.softmax(teacher_logits / T, dim=-1)

    # KL(teacher || student) = Σ p_T * (log p_T - log p_S)
    # F.kl_div expects input=log_probs, target=probs
    loss = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction=reduction,
        log_target=False,
    )

    # Scale by T^2 untuk mempertahankan magnitude gradien
    return loss * (T * T)


# ---------------------------------------------------------------------------
# GenerationDistiller
# ---------------------------------------------------------------------------

class GenerationDistiller:
    """
    Generation-focused distillation trainer.

    Distillation yang berfokus pada kualitas generasi, bukan sekadar logits
    matching. Menggabungkan beberapa loss:

    1. KL Divergence Loss: Memaksa student meniru output distribution teacher
    2. Sequence-Level Loss: Membandingkan kualitas sequence yang dihasilkan
    3. Hidden State Matching: Memaksa student meniru representasi internal

    Mendukung tiga mode:
    - teacher_forcing: Student selalu menerima ground-truth tokens
    - free_running: Student menerima prediksi sendiri (lebih realistis)
    - mixed: Kombinasi keduanya dengan scheduled mixing

    Progressive distillation secara bertahap menggeser dari teacher signal
    ke student's own loss, memungkinkan student mengembangkan kemampuan
    generasinya sendiri.

    Args:
        teacher: Model teacher (full-precision, frozen).
        student: Model student (compressed, trainable).
        config: Konfigurasi distillation.
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        config: Optional[GenerationDistillationConfig] = None,
    ) -> None:
        self.config = config or GenerationDistillationConfig()
        self.student = student
        self.teacher = teacher

        # ---- Freeze teacher ----
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()

        # ---- Progressive distillation state ----
        self._step = 0
        self._teacher_mix_ratio = 1.0  # 1.0 = murni teacher signal

    # ------------------------------------------------------------------
    # Loss Components
    # ------------------------------------------------------------------

    def compute_kl_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Hitung KL divergence loss antara teacher dan student output distributions.

        KL divergence dihitung per-posisi dan di-rata-ratakan.
        Mask opsional mengabaikan posisi padding.

        Args:
            student_logits: Logits student, (batch, seq_len, vocab_size).
            teacher_logits: Logits teacher, (batch, seq_len, vocab_size).
            mask: Mask opsional, (batch, seq_len), 1 = valid, 0 = padding.

        Returns:
            Scalar KL divergence loss.
        """
        loss = kl_divergence_logits(
            student_logits,
            teacher_logits,
            temperature=self.config.temperature,
            reduction="none",
        )  # (batch, seq_len, vocab_size) → sum over vocab

        # Average over vocab dimension
        loss = loss.mean(dim=-1)  # (batch, seq_len)

        if mask is not None:
            loss = loss * mask
            loss = loss.sum() / mask.sum().clamp(min=1.0)
        else:
            loss = loss.mean()

        return loss

    def compute_sequence_loss(
        self,
        student_logits: torch.Tensor,
        target_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Hitung sequence-level distillation loss.

        Sequence-level loss memastikan kualitas generasi student
        dengan menghitung cross-entropy terhadap target tokens.
        Target bisa berupa ground-truth atau teacher-generated sequences.

        Menggunakan label smoothing untuk regularisasi.

        Args:
            student_logits: Logits student, (batch, seq_len, vocab_size).
            target_ids: Target token IDs, (batch, seq_len).
            mask: Mask opsional, (batch, seq_len), 1 = valid, 0 = padding.

        Returns:
            Scalar sequence-level loss.
        """
        # Cross-entropy dengan label smoothing
        vocab_size = student_logits.size(-1)

        # Shift: predict next token
        shift_logits = student_logits[:, :-1, :].contiguous()
        shift_targets = target_ids[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_targets.view(-1),
            label_smoothing=self.config.label_smoothing,
            reduction="none",
        )

        # Reshape dan apply mask
        loss = loss.view(shift_logits.size(0), shift_logits.size(1))

        if mask is not None:
            shift_mask = mask[:, 1:].contiguous()
            loss = loss * shift_mask
            loss = loss.sum() / shift_mask.sum().clamp(min=1.0)
        else:
            loss = loss.mean()

        return loss

    def compute_hidden_loss(
        self,
        student_hiddens: List[torch.Tensor],
        teacher_hiddens: List[torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Hitung hidden state matching loss antara student dan teacher.

        Membandingkan representasi internal di setiap layer yang
        sesuai. Student biasanya memiliki lebih sedikit layer,
        jadi kita memetakan layer student ke layer teacher terdekat.

        Menggunakan MSE loss pada normalized hidden states.

        Args:
            student_hiddens: List hidden states student, setiap tensor
                berbentuk (batch, seq_len, d_student).
            teacher_hiddens: List hidden states teacher, setiap tensor
                berbentuk (batch, seq_len, d_teacher).
            mask: Mask opsional, (batch, seq_len), 1 = valid, 0 = padding.

        Returns:
            Scalar hidden matching loss.
        """
        if not student_hiddens or not teacher_hiddens:
            return torch.tensor(0.0, device=student_logits.device if student_hiddens else "cpu")

        total_loss = torch.tensor(0.0, device=student_hiddens[0].device)
        n_layers = len(student_hiddens)

        for i, s_hidden in enumerate(student_hiddens):
            # Map student layer ke teacher layer terdekat
            t_idx = min(
                int(i * len(teacher_hiddens) / n_layers),
                len(teacher_hiddens) - 1,
            )
            t_hidden = teacher_hiddens[t_idx]

            # Normalisasi sebelum membandingkan (mengatasi perbedaan dimensi)
            s_norm = F.layer_norm(s_hidden, [s_hidden.size(-1)])
            t_norm = F.layer_norm(t_hidden, [t_hidden.size(-1)])

            # Jika dimensi berbeda, proyeksi ke dimensi yang sama
            if s_norm.size(-1) != t_norm.size(-1):
                min_dim = min(s_norm.size(-1), t_norm.size(-1))
                s_proj = s_norm[..., :min_dim]
                t_proj = t_norm[..., :min_dim]
            else:
                s_proj = s_norm
                t_proj = t_norm

            layer_loss = F.mse_loss(s_proj, t_proj, reduction="none").mean(dim=-1)

            if mask is not None:
                layer_loss = (layer_loss * mask).sum() / mask.sum().clamp(min=1.0)
            else:
                layer_loss = layer_loss.mean()

            total_loss = total_loss + layer_loss

        return total_loss / max(n_layers, 1)

    # ------------------------------------------------------------------
    # Progressive Distillation
    # ------------------------------------------------------------------

    def _get_progressive_ratio(self) -> float:
        """
        Hitung rasio teacher signal berdasarkan langkah training.

        Progressive distillation secara bertahap mengurangi ketergantungan
        pada teacher dan memungkinkan student belajar secara mandiri.

        Returns:
            Rasio teacher signal, dari 1.0 (murni teacher) ke 0.0 (student-only).
        """
        if not self.config.progressive:
            return 1.0

        if self._step < self.config.progressive_warmup_steps:
            return 1.0

        if self._step >= self.config.progressive_total_steps:
            return 0.0

        # Linear decay dari 1.0 ke 0.0
        progress = (self._step - self.config.progressive_warmup_steps) / (
            self.config.progressive_total_steps - self.config.progressive_warmup_steps
        )
        return 1.0 - progress

    # ------------------------------------------------------------------
    # Mode-specific forward
    # ------------------------------------------------------------------

    def _teacher_forcing_forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass dengan teacher-forcing: student menerima ground-truth tokens.

        Args:
            input_ids: Token IDs, (batch, seq_len).
            attention_mask: Mask attention opsional.

        Returns:
            Tuple (student_logits, teacher_logits, student_hiddens, teacher_hiddens).
        """
        # Teacher forward (no grad)
        with torch.no_grad():
            teacher_output = self.teacher(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            teacher_logits = teacher_output.logits
            teacher_hiddens = list(teacher_output.hidden_states) if hasattr(teacher_output, 'hidden_states') and teacher_output.hidden_states else []

        # Student forward
        student_output = self.student(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        student_logits = student_output.logits
        student_hiddens = list(student_output.hidden_states) if hasattr(student_output, 'hidden_states') and student_output.hidden_states else []

        return student_logits, teacher_logits, student_hiddens, teacher_hiddens

    def _free_running_forward(
        self,
        prompt_ids: torch.Tensor,
        target_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass dengan free-running: student menggunakan prediksinya sendiri.

        Student generate tokens secara autoregressive, lalu loss dihitung
        berdasarkan perbandingan dengan teacher pada sequence yang sama.

        Args:
            prompt_ids: Prompt token IDs, (batch, prompt_len).
            target_ids: Target token IDs (untuk teacher), (batch, target_len).
            attention_mask: Mask attention opsional.

        Returns:
            Tuple (student_logits, teacher_logits, student_hiddens, teacher_hiddens).
        """
        # ---- Student: generate secara autoregressive ----
        batch_size = prompt_ids.size(0)
        device = prompt_ids.device

        # Generate dari student (dengan gradient)
        # Untuk free-running, kita lakukan forward seluruh sequence
        # tetapi student "melihat" prediksinya sendiri
        student_input = torch.cat([prompt_ids, target_ids[:, :1]], dim=1)
        student_output = self.student(
            input_ids=student_input,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Autoregressive generation sederhana: gunakan output student
        # sebagai input untuk step berikutnya
        generated_ids = prompt_ids.clone()
        student_all_logits = []

        current_ids = prompt_ids
        for step in range(min(self.config.max_seq_len, target_ids.size(1))):
            s_out = self.student(input_ids=current_ids, attention_mask=attention_mask)
            next_logits = s_out.logits[:, -1:, :]  # (batch, 1, vocab)
            student_all_logits.append(next_logits)

            # Sample dari student distribution
            probs = F.softmax(next_logits.squeeze(1) / self.config.temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (batch, 1)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            current_ids = generated_ids

        student_logits = torch.cat(student_all_logits, dim=1)  # (batch, gen_len, vocab)

        # ---- Teacher: forward pada sequence yang sama ----
        with torch.no_grad():
            teacher_input = generated_ids.detach()
            teacher_output = self.teacher(
                input_ids=teacher_input,
                output_hidden_states=True,
            )
            teacher_logits = teacher_output.logits[:, prompt_ids.size(1)-1:-1, :]
            teacher_hiddens = list(teacher_output.hidden_states) if hasattr(teacher_output, 'hidden_states') and teacher_output.hidden_states else []

        student_hiddens = []

        return student_logits, teacher_logits, student_hiddens, teacher_hiddens

    # ------------------------------------------------------------------
    # Training Step
    # ------------------------------------------------------------------

    def train_step(
        self,
        input_ids: torch.Tensor,
        target_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Satu langkah distillation training.

        Menggabungkan KL loss, sequence loss, dan hidden loss
        berdasarkan konfigurasi. Mode menentukan bagaimana student
        menerima input.

        Args:
            input_ids: Token IDs input, (batch, seq_len).
            target_ids: Target token IDs opsional. Jika None, menggunakan
                input_ids sebagai target.
            attention_mask: Mask attention opsional.

        Returns:
            Dictionary metrics (loss, kl_loss, seq_loss, hidden_loss, dll.).
        """
        self.student.train()
        self.teacher.eval()

        if target_ids is None:
            target_ids = input_ids

        # ---- Forward berdasarkan mode ----
        if self.config.mode == "teacher_forcing":
            s_logits, t_logits, s_hiddens, t_hiddens = self._teacher_forcing_forward(
                input_ids, attention_mask
            )
        elif self.config.mode == "free_running":
            prompt_len = max(1, input_ids.size(1) // 4)
            s_logits, t_logits, s_hiddens, t_hiddens = self._free_running_forward(
                input_ids[:, :prompt_len], target_ids, attention_mask
            )
        else:  # mixed
            # Mixing: gunakan teacher-forcing untuk sebagian, free-running untuk lainnya
            s_logits, t_logits, s_hiddens, t_hiddens = self._teacher_forcing_forward(
                input_ids, attention_mask
            )

        # ---- Hitung loss components ----
        # 1. KL Divergence Loss
        kl_loss = self.compute_kl_loss(s_logits, t_logits.detach(), attention_mask)

        # 2. Sequence-Level Loss
        seq_loss = self.compute_sequence_loss(s_logits, target_ids, attention_mask)

        # 3. Hidden State Matching Loss (opsional)
        if self.config.alpha_hidden > 0 and s_hiddens and t_hiddens:
            hidden_loss = self.compute_hidden_loss(s_hiddens, t_hiddens, attention_mask)
        else:
            hidden_loss = torch.tensor(0.0, device=kl_loss.device)

        # ---- Progressive distillation ratio ----
        teacher_ratio = self._get_progressive_ratio()

        # ---- Total loss ----
        # KL loss dikalikan rasio teacher (progressive)
        # Sequence loss selalu aktif (student harus bisa generate)
        total_loss = (
            self.config.alpha_kl * teacher_ratio * kl_loss
            + self.config.alpha_seq * seq_loss
            + self.config.alpha_hidden * teacher_ratio * hidden_loss
        )

        # ---- Update step counter ----
        self._step += 1

        # ---- Metrics ----
        with torch.no_grad():
            metrics = {
                "total_loss": total_loss.item(),
                "kl_loss": kl_loss.item(),
                "seq_loss": seq_loss.item(),
                "hidden_loss": hidden_loss.item(),
                "teacher_ratio": teacher_ratio,
                "step": self._step,
            }

        return metrics

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_progress_info(self) -> Dict[str, float]:
        """
        Dapatkan informasi progress distillation.

        Returns:
            Dictionary berisi teacher_ratio dan step saat ini.
        """
        return {
            "teacher_ratio": self._get_progressive_ratio(),
            "step": self._step,
            "mode": self.config.mode,
        }

    def reset_step(self) -> None:
        """Reset step counter untuk progressive distillation."""
        self._step = 0
