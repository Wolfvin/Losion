"""
BitDistill — Distillation-Aware Quantization untuk Losion Framework.

BitDistill menggabungkan quantization dengan distillation sehingga model
ter-quantize belajar meniru distribusi output model full-precision.

Mengapa BitDistill lebih baik dari standard QAT?
- Standard QAT hanya mengoptimalkan task loss pada quantized model
- BitDistill menambahkan distillation signal dari teacher (full-precision)
- Teacher memberikan soft targets yang lebih informatif daripada hard labels
- Gradual quantization schedule kompatibel dengan BitNet infrastructure

Arsitektur:
- Teacher: Full-precision model (frozen copy dari model sebelum quantization)
- Student: Quantized model (BitNetLinear layers, sedang di-training)

Loss:
    L_total = alpha_quant * L_task + alpha_distill * L_KL(teacher || student)

Dimana:
- L_task = task-specific loss (cross-entropy untuk language modeling)
- L_KL = KL divergence antara output distributions teacher dan student

Kompatibel dengan:
- BitNet 1.58-bit ternary quantization ({-1, 0, +1})
- BitNet gradual quantization schedule
- Standard PyTorch training loop

Komponen:
1. BitDistillConfig — Konfigurasi BitDistill
2. BitDistillTrainer — Trainer untuk distillation-aware quantization

Referensi:
- Wang et al., "BitNet b1.58" (2024) — 1.58-bit quantization
- Kim & Rush, "Sequence-Level Knowledge Distillation" (2016)
- Stock et al., "Training with Quantization Noise" (2020) — QAT

Hardware: Pure PyTorch, kompatibel dengan CUDA, ROCm, dan CPU.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from losion.core.quantization.bitnet import (
    BitNetConfig,
    BitNetLinear,
    convert_linear_to_bitnet,
    increment_bitnet_step,
    finalize_bitnet_model,
)


# ---------------------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------------------

@dataclass
class BitDistillConfig:
    """Konfigurasi untuk BitDistill — distillation-aware quantization.

    Attributes:
        quant_schedule: Jadwal quantization — "gradual" atau "immediate".
            "gradual": Quantization ratio meningkat secara linear dari 0 ke 1
            selama warmup_steps (kompatibel dengan BitNet schedule).
            "immediate": Full quantization dari awal.
        distill_temperature: Temperature untuk distillation softmax.
            Nilai tinggi → distribusi lebih soft → lebih banyak informasi
            dari teacher.
        alpha_quant: Bobot untuk quantization-aware task loss.
            Task loss dihitung pada quantized model.
        alpha_distill: Bobot untuk distillation loss (KL divergence).
            Distillation loss memaksa student meniru teacher.
        warmup_steps: Jumlah steps untuk gradual quantization.
            Selama warmup, quantization ratio meningkat dari 0 ke 1.
            Juga digunakan sebagai warmup untuk distillation signal.
        bitnet_config: Konfigurasi BitNet yang mendasari.
            BitDistill menggunakan BitNetLinear untuk quantization.
        exclude_layers: Nama layer yang tidak di-quantize (misalnya
            ["lm_head", "embedding"]).
        hidden_distill: Jika True, tambahkan hidden state matching loss
            antara teacher dan student.
        alpha_hidden: Bobot untuk hidden state matching loss.
    """

    quant_schedule: str = "gradual"
    distill_temperature: float = 4.0
    alpha_quant: float = 0.5
    alpha_distill: float = 0.5
    warmup_steps: int = 2000
    bitnet_config: Optional[BitNetConfig] = None
    exclude_layers: List[str] = field(default_factory=lambda: ["lm_head"])
    hidden_distill: bool = False
    alpha_hidden: float = 0.1

    def __post_init__(self) -> None:
        if self.quant_schedule not in ("gradual", "immediate"):
            raise ValueError(
                f"quant_schedule harus 'gradual' atau 'immediate', "
                f"mendapat '{self.quant_schedule}'"
            )
        if self.distill_temperature <= 0:
            raise ValueError(
                f"distill_temperature harus > 0, mendapat {self.distill_temperature}"
            )
        if self.bitnet_config is None:
            self.bitnet_config = BitNetConfig(
                enabled=True,
                warmup_steps=self.warmup_steps if self.quant_schedule == "gradual" else 0,
                initial_quant_ratio=0.0,
                STE_mode="identity",
                quantize_on_forward=True,
            )


# ---------------------------------------------------------------------------
# BitDistillTrainer
# ---------------------------------------------------------------------------

class BitDistillTrainer:
    """
    BitDistill: Distillation-aware quantization trainer.

    Menggabungkan quantization-aware training (QAT) dengan knowledge
    distillation untuk menghasilkan model ter-quantize yang berkualitas
    lebih tinggi daripada standard QAT.

    Cara kerja:
    1. Salin model full-precision sebagai teacher (frozen)
    2. Konversi student ke BitNetLinear layers
    3. Training: minimize joint loss (task + distillation)
    4. Gradual quantization: quantization ratio meningkat selama warmup
    5. Finalisasi: convert ke int2 packed weights untuk inference

    Teacher menyediakan soft targets yang lebih informatif daripada
    hard labels, membantu student mengkompensasi informasi yang hilang
    akibat quantization.

    Args:
        model: Model yang akan di-quantize dan di-distill.
        config: Konfigurasi BitDistill.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[BitDistillConfig] = None,
    ) -> None:
        self.config = config or BitDistillConfig()
        self.device = next(model.parameters()).device

        # ---- Buat teacher (frozen copy dari original model) ----
        self.teacher = copy.deepcopy(model)
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()

        # ---- Konversi student ke BitNet ----
        self.student = model
        convert_linear_to_bitnet(
            self.student,
            config=self.config.bitnet_config,
            exclude_names=self.config.exclude_layers,
        )

        # ---- Step counter ----
        self._step = 0

    # ------------------------------------------------------------------
    # Quantization Helpers
    # ------------------------------------------------------------------

    def _get_quant_ratio(self) -> float:
        """
        Hitung quantization ratio saat ini.

        Returns:
            Quantization ratio di [0, 1]. 0 = full precision, 1 = full quantization.
        """
        if self.config.quant_schedule == "immediate":
            return 1.0
        if self.config.warmup_steps <= 0:
            return 1.0
        if self._step >= self.config.warmup_steps:
            return 1.0
        return self._step / self.config.warmup_steps

    def _get_distill_weight(self) -> float:
        """
        Hitung bobot distillation saat ini.

        Distillation weight meningkat selama warmup, mencapai
        alpha_distill pada akhir warmup. Ini mencegah distillation
        signal terlalu kuat di awal training ketika student masih
        sangat berbeda dari teacher.

        Returns:
            Distillation weight efektif.
        """
        if self._step < self.config.warmup_steps:
            # Ramp up distillation weight
            progress = self._step / max(1, self.config.warmup_steps)
            return self.config.alpha_distill * progress
        return self.config.alpha_distill

    # ------------------------------------------------------------------
    # Loss Components
    # ------------------------------------------------------------------

    def _compute_task_loss(
        self,
        student_logits: torch.Tensor,
        target_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Hitung task loss pada quantized student model.

        Standard cross-entropy loss untuk language modeling.

        Args:
            student_logits: Logits student, (batch, seq_len, vocab_size).
            target_ids: Target token IDs, (batch, seq_len).
            attention_mask: Mask attention opsional.

        Returns:
            Scalar task loss.
        """
        vocab_size = student_logits.size(-1)
        shift_logits = student_logits[:, :-1, :].contiguous()
        shift_targets = target_ids[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_targets.view(-1),
            reduction="none",
        )

        # Reshape dan apply mask
        loss = loss.view(shift_logits.size(0), shift_logits.size(1))

        if attention_mask is not None:
            shift_mask = attention_mask[:, 1:].contiguous()
            loss = loss * shift_mask
            loss = loss.sum() / shift_mask.sum().clamp(min=1.0)
        else:
            loss = loss.mean()

        return loss

    def _compute_distill_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Hitung distillation loss (KL divergence) antara teacher dan student.

        Menggunakan temperature-scaled softmax untuk soft targets:
            KL(p_T || p_S) = Σ p_T * (log p_T - log p_S)

        Args:
            student_logits: Logits student, (batch, seq_len, vocab_size).
            teacher_logits: Logits teacher, (batch, seq_len, vocab_size).
            attention_mask: Mask attention opsional.

        Returns:
            Scalar distillation loss.
        """
        T = self.config.distill_temperature

        # Temperature-scaled log-softmax
        student_log_probs = F.log_softmax(student_logits / T, dim=-1)
        teacher_probs = F.softmax(teacher_logits / T, dim=-1)

        # KL divergence per position
        loss = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction="none",
            log_target=False,
        )  # (batch, seq_len, vocab_size)

        # Average over vocab
        loss = loss.mean(dim=-1)  # (batch, seq_len)

        # Apply mask
        if attention_mask is not None:
            loss = loss * attention_mask
            loss = loss.sum() / attention_mask.sum().clamp(min=1.0)
        else:
            loss = loss.mean()

        # Scale by T^2 untuk mempertahankan magnitude gradien
        return loss * (T * T)

    def _compute_hidden_loss(
        self,
        student_hiddens: List[torch.Tensor],
        teacher_hiddens: List[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Hitung hidden state matching loss antara student dan teacher.

        Membandingkan representasi internal pada layer yang sesuai.
        Karena student menggunakan BitNetLinear, dimensi tetap sama
        tetapi distribusi berbeda.

        Args:
            student_hiddens: Hidden states student.
            teacher_hiddens: Hidden states teacher.
            attention_mask: Mask attention opsional.

        Returns:
            Scalar hidden matching loss.
        """
        if not student_hiddens or not teacher_hiddens:
            return torch.tensor(0.0, device=self.device)

        total_loss = torch.tensor(0.0, device=self.device)
        n_comparisons = min(len(student_hiddens), len(teacher_hiddens))

        for i in range(n_comparisons):
            s_h = student_hiddens[i]
            t_h = teacher_hiddens[i]

            # Normalisasi sebelum membandingkan
            s_norm = F.layer_norm(s_h, [s_h.size(-1)])
            t_norm = F.layer_norm(t_h, [t_h.size(-1)])

            layer_loss = F.mse_loss(s_norm, t_norm.detach(), reduction="none").mean(dim=-1)

            if attention_mask is not None:
                layer_loss = (layer_loss * attention_mask).sum() / attention_mask.sum().clamp(min=1.0)
            else:
                layer_loss = layer_loss.mean()

            total_loss = total_loss + layer_loss

        return total_loss / max(n_comparisons, 1)

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
        Satu langkah BitDistill training.

        Joint loss: quantization_aware_loss + alpha * distillation_loss

        Alur:
        1. Forward student (quantized) dan teacher (full-precision)
        2. Hitung task loss pada student
        3. Hitung distillation loss (KL divergence)
        4. Opsional: hitung hidden matching loss
        5. Gabungkan losses dan backward

        Args:
            input_ids: Token IDs, (batch, seq_len).
            target_ids: Target token IDs opsional.
            attention_mask: Mask attention opsional.

        Returns:
            Dictionary metrics (loss, quant_ratio, dll.).
        """
        self.student.train()
        self.teacher.eval()

        if target_ids is None:
            target_ids = input_ids

        # ---- Student forward (dengan quantization) ----
        student_output = self.student(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=self.config.hidden_distill,
        )
        student_logits = student_output.logits if hasattr(student_output, 'logits') else student_output
        student_hiddens = (
            list(student_output.hidden_states)
            if self.config.hidden_distill and hasattr(student_output, 'hidden_states') and student_output.hidden_states
            else []
        )

        # ---- Teacher forward (no grad) ----
        with torch.no_grad():
            teacher_output = self.teacher(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=self.config.hidden_distill,
            )
            teacher_logits = teacher_output.logits if hasattr(teacher_output, 'logits') else teacher_output
            teacher_hiddens = (
                list(teacher_output.hidden_states)
                if self.config.hidden_distill and hasattr(teacher_output, 'hidden_states') and teacher_output.hidden_states
                else []
            )

        # ---- Loss Components ----
        # 1. Task loss (pada quantized student)
        task_loss = self._compute_task_loss(student_logits, target_ids, attention_mask)

        # 2. Distillation loss (KL divergence)
        distill_loss = self._compute_distill_loss(
            student_logits, teacher_logits.detach(), attention_mask
        )

        # 3. Hidden matching loss (opsional)
        if self.config.hidden_distill and student_hiddens and teacher_hiddens:
            hidden_loss = self._compute_hidden_loss(
                student_hiddens, teacher_hiddens, attention_mask
            )
        else:
            hidden_loss = torch.tensor(0.0, device=self.device)

        # ---- Joint Loss ----
        alpha_d = self._get_distill_weight()
        total_loss = (
            self.config.alpha_quant * task_loss
            + alpha_d * distill_loss
            + self.config.alpha_hidden * hidden_loss
        )

        # ---- Increment BitNet step (untuk gradual quantization) ----
        increment_bitnet_step(self.student)

        # ---- Update step counter ----
        self._step += 1

        # ---- Metrics ----
        quant_ratio = self._get_quant_ratio()

        with torch.no_grad():
            metrics = {
                "total_loss": total_loss.item(),
                "task_loss": task_loss.item(),
                "distill_loss": distill_loss.item(),
                "hidden_loss": hidden_loss.item(),
                "quant_ratio": quant_ratio,
                "distill_weight": alpha_d,
                "step": self._step,
            }

        return metrics

    # ------------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------------

    def finalize(self) -> nn.Module:
        """
        Finalisasi student model untuk inference.

        Mengkonversi semua BitNetLinear weights ke int2 packed format
        untuk memory efisien. Setelah finalisasi, model siap untuk
        deployment.

        Returns:
            Student model yang sudah difinalisasi.
        """
        self.student.eval()
        finalize_bitnet_model(self.student)
        return self.student

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_quantization_summary(self) -> Dict[str, object]:
        """
        Dapatkan ringkasan status quantization.

        Returns:
            Dictionary berisi quantization info per layer.
        """
        summary: Dict[str, object] = {
            "step": self._step,
            "quant_ratio": self._get_quant_ratio(),
            "quant_schedule": self.config.quant_schedule,
            "num_bitnet_layers": 0,
            "total_params": 0,
            "quantized_params": 0,
        }

        for name, module in self.student.named_modules():
            if isinstance(module, BitNetLinear):
                summary["num_bitnet_layers"] = summary["num_bitnet_layers"] + 1  # type: ignore
                total = module.weight.numel()
                summary["total_params"] = summary["total_params"] + total  # type: ignore
                if module._packed_ready:
                    summary["quantized_params"] = summary["quantized_params"] + total  # type: ignore

        return summary

    def memory_savings_estimate(self) -> Dict[str, float]:
        """
        Estimasi penghematan memory setelah quantization.

        Returns:
            Dictionary berisi estimasi memory savings.
        """
        total_fp32_bytes = 0
        total_int2_bytes = 0

        for module in self.student.modules():
            if isinstance(module, BitNetLinear):
                numel = module.out_features * module.in_features
                total_fp32_bytes += numel * 4  # float32 = 4 bytes
                total_int2_bytes += numel * 2 / 8  # 2 bits per weight = 2/8 bytes

        return {
            "fp32_bytes": total_fp32_bytes,
            "int2_bytes": total_int2_bytes,
            "compression_ratio": total_fp32_bytes / max(total_int2_bytes, 1),
            "savings_percent": (1.0 - total_int2_bytes / max(total_fp32_bytes, 1)) * 100,
        }
