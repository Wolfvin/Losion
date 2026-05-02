"""
Advanced Training & Backprop — DeepMind/Google AI Techniques
=============================================================

Menggabungkan 7 teknik untuk training dan backpropagation yang lebih efisien:

1. Chinchilla Per-Jalur Scaling — optimal parameter allocation per pathway
2. Per-Jalur Learning Rate Schedules — mencegah cheap-branch oscillation
3. Logit Soft Capping (Gemma 2) — stabilisasi training tanpa hard clipping
4. Scheduled Sampling (GraphCast) — bridge teacher-forcing/autoregressive gap
5. Confidence Heads (AlphaFold 3) — dense auxiliary training signals
6. Parallel Attention+FFN (PaLM) — doubled effective depth per layer
7. Gradient Communication Overlapping (PaLM 2) — hide 40-60% latency

Referensi:
- Hoffmann et al., "Training Compute-Optimal Large Language Models" (Chinchilla, 2022)
- Gemma Team, "Gemma 2: Open Language Model" (2024)
- Lam et al., "GraphCast: Learning skillful medium-range global weather forecasting" (2023)
- Abramson et al., "Accurate structure prediction of biomolecular interactions with AlphaFold 3" (2024)
- Chowdhery et al., "PaLM: Scaling Language Modeling with Pathways" (2022)

Hardware: Pure PyTorch, kompatibel dengan CUDA, ROCm, dan CPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 1. Chinchilla Per-Jalur Scaling
# ============================================================================


@dataclass
class ChinchillaScalingResult:
    """Hasil analisis Chinchilla scaling per jalur.

    Attributes:
        total_flops: Total FLOPs budget.
        jalur_flops: FLOPs per jalur [jalur1, jalur2, jalur3].
        jalur_params: Optimal parameters per jalur.
        jalur_data: Optimal data tokens per jalur.
        token_to_param_ratio: Rasio token/parameter per jalur.
    """

    total_flops: int
    jalur_flops: List[int]
    jalur_params: List[int]
    jalur_data: List[int]
    token_to_param_ratio: List[float]


class ChinchillaScaler:
    """
    Chinchilla-optimal scaling untuk Tri-Jalur architecture.

    Chinchilla (Hoffmann et al., 2022) menunjukkan bahwa untuk budget
    komputasi C, parameter optimal N dan data D mengikuti C ≈ 6ND.
    Rasio optimal: ~20 token per parameter.

    Losion mengadaptasi ini PER JALUR:
    - Setiap jalur punya FLOP budget berbeda (SSM murah, Attention mahal)
    - Parameter dialokasikan proporsional ke FLOP budget
    - MoE: hanya hitung active parameters (bukan total experts)

    Args:
        total_flops_budget: Total FLOPs budget (contoh: 1e22).
        jalur_flop_ratios: Rasio FLOPs per jalur (default [0.2, 0.5, 0.3]).
    """

    # Chinchilla constants: C = 6 * N * D
    # Optimal ratio: D/N ≈ 20
    CHINCHILLA_CONSTANT = 6.0
    OPTIMAL_TOKEN_RATIO = 20.0

    def __init__(
        self,
        total_flops_budget: float = 1e22,
        jalur_flop_ratios: Tuple[float, float, float] = (0.2, 0.5, 0.3),
    ) -> None:
        self.total_flops = total_flops_budget
        self.jalur_ratios = list(jalur_flop_ratios)
        # Normalize ratios
        total_ratio = sum(self.jalur_ratios)
        self.jalur_ratios = [r / total_ratio for r in self.jalur_ratios]

    def compute_optimal_scaling(
        self,
        moe_active_ratio: float = 0.1,
    ) -> ChinchillaScalingResult:
        """
        Hitung alokasi parameter dan data yang optimal per jalur.

        Args:
            moe_active_ratio: Rasio active/total parameters di MoE (Jalur 3).
                Contoh: 6/64 experts = 0.094 → hanya 9.4% params aktif.

        Returns:
            ChinchillaScalingResult dengan alokasi optimal.
        """
        results_flops = []
        results_params = []
        results_data = []
        results_ratio = []

        for ratio in self.jalur_ratios:
            # FLOP budget untuk jalur ini
            jalur_flops = int(self.total_flops * ratio)

            # Chinchilla: C = 6 * N * D, D = 20 * N
            # C = 6 * N * 20 * N = 120 * N^2
            # N = sqrt(C / 120)
            active_params = math.sqrt(jalur_flops / (self.CHINCHILLA_CONSTANT * self.OPTIMAL_TOKEN_RATIO))

            # Data tokens
            data_tokens = int(self.OPTIMAL_TOKEN_RATIO * active_params)

            results_flops.append(jalur_flops)
            results_params.append(int(active_params))
            results_data.append(data_tokens)
            results_ratio.append(self.OPTIMAL_TOKEN_RATIO)

        return ChinchillaScalingResult(
            total_flops=int(self.total_flops),
            jalur_flops=results_flops,
            jalur_params=results_params,
            jalur_data=results_data,
            token_to_param_ratio=results_ratio,
        )

    def validate_config(
        self,
        config: Any,
    ) -> Dict[str, Any]:
        """
        Validasi apakah konfigurasi Losion sesuai Chinchilla scaling.

        Args:
            config: LosionConfig.

        Returns:
            Dictionary berisi analisis dan rekomendasi.
        """
        # Estimate FLOP ratios berdasarkan konfigurasi
        d = config.d_model

        # Jalur 1 (SSM): ~2 * d * expand * d per layer
        ssm_flops_per_layer = 2 * d * config.ssm.expand * d
        # Jalur 2 (Attention): ~4 * d * d (QKV+O projections)
        attn_flops_per_layer = 4 * d * d + 2 * d * d * 4  # + FFN
        # Jalur 3 (MoE): num_active * 2 * d * d_ff
        d_ff = min(d * 4, 4096)
        moe_flops_per_layer = config.retrieval.num_active_experts * 2 * d * d_ff

        total_per_layer = ssm_flops_per_layer + attn_flops_per_layer + moe_flops_per_layer

        actual_ratios = [
            ssm_flops_per_layer / total_per_layer,
            attn_flops_per_layer / total_per_layer,
            moe_flops_per_layer / total_per_layer,
        ]

        # MoE active ratio
        moe_active = config.retrieval.num_active_experts / max(config.retrieval.num_experts, 1)

        return {
            "actual_flop_ratios": {
                "jalur_1_ssm": actual_ratios[0],
                "jalur_2_attention": actual_ratios[1],
                "jalur_3_moe": actual_ratios[2],
            },
            "moe_active_ratio": moe_active,
            "optimal_ratios": self.jalur_ratios,
            "recommendation": (
                f"Jalur 2 mendapat {actual_ratios[1]:.1%} FLOPs. "
                f"Chinchilla optimal: {self.jalur_ratios[1]:.1%}. "
                f"{'OK' if abs(actual_ratios[1] - self.jalur_ratios[1]) < 0.1 else 'Perlu penyesuaian parameter'}"
            ),
        }


# ============================================================================
# 2. Per-Jalur Learning Rate Schedules
# ============================================================================


class PerJalurLRScheduler:
    """
    Learning rate schedule yang berbeda per jalur (Chinchilla adaptation).

    Setiap jalur punya karakteristik training yang berbeda:
    - Jalur 1 (SSM): Cheap per step → LR cepat peak, cepat decay
    - Jalur 2 (Attention): Expensive per step → LR lambat peak, lambat decay
    - Jalur 3 (MoE): Medium → LR medium

    Mencegah masalah: cheap branch oscillasi saat expensive branch under-train.

    Args:
        base_lr: Base learning rate.
        total_steps: Total training steps.
        warmup_ratios: Warmup ratio per jalur [ssm, attn, retrieval].
        decay_rates: Decay rate per jalur (cosine sharpness).
    """

    def __init__(
        self,
        base_lr: float = 3e-4,
        total_steps: int = 100000,
        warmup_ratios: Tuple[float, float, float] = (0.03, 0.06, 0.04),
        decay_rates: Tuple[float, float, float] = (0.8, 0.5, 0.6),
    ) -> None:
        self.base_lr = base_lr
        self.total_steps = total_steps
        self.warmup_ratios = list(warmup_ratios)
        self.decay_rates = list(decay_rates)

    def get_lr(
        self,
        step: int,
        jalur_idx: int,
    ) -> float:
        """
        Hitung learning rate untuk jalur tertentu pada step tertentu.

        Menggunakan cosine schedule dengan warmup yang berbeda per jalur.

        Args:
            step: Training step saat ini.
            jalur_idx: Indeks jalur (0=SSM, 1=Attention, 2=MoE).

        Returns:
            Learning rate untuk jalur ini pada step ini.
        """
        if jalur_idx < 0 or jalur_idx >= 3:
            return self.base_lr

        warmup_steps = int(self.total_steps * self.warmup_ratios[jalur_idx])
        decay_rate = self.decay_rates[jalur_idx]

        if step < warmup_steps:
            # Linear warmup
            return self.base_lr * step / max(warmup_steps, 1)
        else:
            # Cosine decay
            progress = (step - warmup_steps) / max(self.total_steps - warmup_steps, 1)
            return self.base_lr * (0.5 * (1.0 + math.cos(math.pi * progress ** decay_rate)))

    def get_all_lrs(self, step: int) -> List[float]:
        """Ambil LR semua jalur pada step tertentu."""
        return [self.get_lr(step, i) for i in range(3)]


# ============================================================================
# 3. Logit Soft Capping (Gemma 2)
# ============================================================================


class LogitSoftCapper(nn.Module):
    """
    Logit Soft Capping — mencegah logit divergence (Gemma 2).

    Gemma 2 menggunakan soft capping sebelum dan sesudah softmax:
        capped = soft_cap * tanh(x / soft_cap)

    Ini mencegah logit divergence selama training tanpa hard clipping,
    yang bisa menyebabkan mode collapse.

    Diterapkan pada:
    - AR output logits
    - Flow Matching velocity predictions
    - MTP auxiliary head logits

    Args:
        cap_value: Nilai soft cap (default 50.0, dari Gemma 2).
    """

    def __init__(self, cap_value: float = 50.0) -> None:
        super().__init__()
        self.cap_value = cap_value
        self.register_buffer("cap", torch.tensor(cap_value))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply soft capping ke logits.

        Formula: capped = cap * tanh(logits / cap)

        Args:
            logits: Input logits [batch, seq, vocab_size] atau shape apapun.

        Returns:
            Capped logits dengan shape yang sama.
        """
        return self.cap * torch.tanh(logits / self.cap)


# ============================================================================
# 4. Scheduled Sampling (GraphCast)
# ============================================================================


class ScheduledSampler:
    """
    Scheduled Sampling — bridge teacher-forcing/autoregressive gap (GraphCast).

    Selama training awal, gunakan teacher forcing (ground truth sebagai input).
    Secara bertahap, ganti dengan model's own predictions.

    Ini mencegah "exposure bias" di mana model tidak pernah melihat
    kesalahannya sendiri selama training.

    Probability of using model's own prediction:
        p = min(1, step / total_steps) * max_scheduled_ratio

    Args:
        total_steps: Total training steps.
        max_scheduled_ratio: Rasio maksimum penggunaan model predictions (default 0.5).
        warmup_steps: Steps sebelum scheduled sampling dimulai (default 1000).
        schedule_type: Tipe schedule ("linear", "exponential", "inverse_sigmoid").
    """

    def __init__(
        self,
        total_steps: int = 100000,
        max_scheduled_ratio: float = 0.5,
        warmup_steps: int = 1000,
        schedule_type: str = "linear",
    ) -> None:
        self.total_steps = total_steps
        self.max_ratio = max_scheduled_ratio
        self.warmup_steps = warmup_steps
        self.schedule_type = schedule_type

    def get_sampling_probability(self, step: int) -> float:
        """
        Hitung probability menggunakan model prediction vs ground truth.

        Args:
            step: Training step saat ini.

        Returns:
            Probability [0, max_ratio] of using model's own prediction.
        """
        if step < self.warmup_steps:
            return 0.0  # Pure teacher forcing

        effective_step = step - self.warmup_steps
        effective_total = self.total_steps - self.warmup_steps

        if effective_total <= 0:
            return self.max_ratio

        progress = min(effective_step / effective_total, 1.0)

        if self.schedule_type == "linear":
            return progress * self.max_ratio
        elif self.schedule_type == "exponential":
            # Exponential: lebih lambat di awal, lebih cepat di akhir
            k = 5.0
            return (1 - math.exp(-k * progress)) / (1 - math.exp(-k)) * self.max_ratio
        elif self.schedule_type == "inverse_sigmoid":
            # Inverse sigmoid: dari Bengio et al. (2015)
            k = 1000
            return min(k / (k + math.exp(effective_step / k)), self.max_ratio)
        else:
            return progress * self.max_ratio

    def sample_input(
        self,
        ground_truth: torch.Tensor,
        model_prediction: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        """
        Pilih input untuk step training berikutnya.

        Dengan probability p, gunakan model prediction.
        Dengan probability 1-p, gunakan ground truth.

        Args:
            ground_truth: Ground truth tokens [batch, seq].
            model_prediction: Model predicted tokens [batch, seq].
            step: Training step.

        Returns:
            Selected input tokens [batch, seq].
        """
        p = self.get_sampling_probability(step)

        if p <= 0:
            return ground_truth

        # Bernoulli mask
        mask = torch.rand(ground_truth.shape, device=ground_truth.device) < p
        mask = mask.long()

        return mask * model_prediction + (1 - mask) * ground_truth


# ============================================================================
# 5. Confidence Heads (AlphaFold 3)
# ============================================================================


class ConfidenceHeads(nn.Module):
    """
    Confidence Heads — predict output quality metrics (AlphaFold 3).

    AlphaFold 3 menggunakan multiple auxiliary heads (pLDDT, pAE, pTM)
    yang memprediksi kualitas output, memberikan rich supervisory signals.

    Losion mengadaptasi ini dengan 3 confidence heads:
    1. Routing Confidence: Apakah routing decision benar?
    2. Prediction Difficulty: Seberapa sulit token berikutnya?
    3. Diffusion Quality: Apakah flow matching akan menghasilkan output bagus?

    Heads ini memberikan dense auxiliary training signals TANPA
    mempengaruhi inference (bisa di-distill away).

    Args:
        d_model: Dimensi model.
        num_confidence_types: Jumlah confidence heads (default 3).
    """

    def __init__(
        self,
        d_model: int,
        num_confidence_types: int = 3,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_types = num_confidence_types

        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(d_model, d_model // 4, bias=False),
            nn.SiLU(),
            nn.Linear(d_model // 4, d_model // 8, bias=False),
            nn.SiLU(),
        )

        # Per-type confidence heads
        self.heads = nn.ModuleList([
            nn.Linear(d_model // 8, 1, bias=True)
            for _ in range(num_confidence_types)
        ])

        # Initialize to predict near-0.5 (uncertain)
        for head in self.heads:
            nn.init.zeros_(head.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict confidence scores.

        Args:
            hidden_states: [batch, seq, d_model]

        Returns:
            Dictionary berisi:
            - routing_confidence: [batch, seq] — seberapa yakin routing benar
            - prediction_difficulty: [batch, seq] — seberapa sulit token berikutnya
            - diffusion_quality: [batch, seq] — seberapa bagus flow matching output
        """
        h = self.trunk(hidden_states)  # [batch, seq, d_model//8]

        # Sigmoid untuk output [0, 1]
        routing_conf = torch.sigmoid(self.heads[0](h).squeeze(-1))
        pred_diff = torch.sigmoid(self.heads[1](h).squeeze(-1))
        diff_quality = torch.sigmoid(self.heads[2](h).squeeze(-1))

        return {
            "routing_confidence": routing_conf,
            "prediction_difficulty": pred_diff,
            "diffusion_quality": diff_quality,
        }

    def compute_auxiliary_loss(
        self,
        hidden_states: torch.Tensor,
        ar_loss_per_token: torch.Tensor,
        routing_entropy: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute auxiliary loss dari confidence heads.

        Supervisory signals:
        - Routing confidence: target = routing_entropy (high entropy = low confidence)
        - Prediction difficulty: target = ar_loss_per_token (high loss = difficult)
        - Diffusion quality: target = 1 - ar_loss_per_token (low loss = high quality)

        Args:
            hidden_states: [batch, seq, d_model]
            ar_loss_per_token: [batch, seq] — AR loss per token
            routing_entropy: [batch, seq] — routing entropy per token

        Returns:
            Scalar auxiliary loss.
        """
        predictions = self.forward(hidden_states)

        # Normalize targets ke [0, 1]
        with torch.no_grad():
            # Prediction difficulty: normalize AR loss
            ar_normalized = ar_loss_per_token / (ar_loss_per_token.max() + 1e-8)
            # Routing confidence: 1 - normalized entropy
            ent_normalized = routing_entropy / (routing_entropy.max() + 1e-8)
            # Diffusion quality: 1 - AR loss
            quality_target = 1.0 - ar_normalized

        # MSE losses
        loss_routing = F.mse_loss(predictions["routing_confidence"], ent_normalized)
        loss_difficulty = F.mse_loss(predictions["prediction_difficulty"], ar_normalized)
        loss_quality = F.mse_loss(predictions["diffusion_quality"], quality_target)

        return loss_routing + loss_difficulty + loss_quality


# ============================================================================
# 6. Parallel Attention + FFN (PaLM)
# ============================================================================


class ParallelAttentionFFN(nn.Module):
    """
    Parallel Attention + FFN — PaLM-style parallel formulation.

    PaLM menghitung attention dan FFN secara PARALEL (bukan sekuensial):
        output = x + Attention(LayerNorm(x)) + FFN(LayerNorm(x))

    Ini mengurangi depth ~50% dengan kualitas yang hampir sama.
    Efektif menggandakan "effective depth" dalam latency budget yang sama.

    Untuk Jalur 2: MLA attention dan FFN/compression berjalan paralel.

    Args:
        d_model: Dimensi model.
        n_heads: Jumlah attention heads.
        d_kv: Dimensi per-head KV.
        mla_latent_dim: Dimensi latent MLA.
        ffn_dim_multiplier: Pengali dimensi FFN.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 16,
        d_kv: int = 128,
        mla_latent_dim: int = 512,
        ffn_dim_multiplier: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        # Shared pre-norm (PaLM: satu LayerNorm untuk kedua branch)
        self.pre_norm = nn.RMSNorm(d_model, eps=1e-5)

        # Attention branch (MLA simplified)
        self.q_proj = nn.Linear(d_model, n_heads * d_kv, bias=False)
        self.k_proj = nn.Linear(d_model, n_heads * d_kv, bias=False)
        self.v_proj = nn.Linear(d_model, n_heads * d_kv, bias=False)
        self.o_proj = nn.Linear(n_heads * d_kv, d_model, bias=False)
        self.n_heads = n_heads
        self.d_kv = d_kv

        # MLA compression
        self.kv_compress = nn.Linear(2 * n_heads * d_kv, mla_latent_dim, bias=False)
        self.kv_decompress = nn.Linear(mla_latent_dim, 2 * n_heads * d_kv, bias=False)

        # FFN branch (SwiGLU)
        ffn_dim = d_model * ffn_dim_multiplier
        self.ffn_gate = nn.Linear(d_model, ffn_dim, bias=False)
        self.ffn_up = nn.Linear(d_model, ffn_dim, bias=False)
        self.ffn_down = nn.Linear(ffn_dim, d_model, bias=False)

        # Dropout
        self.attn_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.ffn_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parallel Attention + FFN forward.

        output = x + Attention(LN(x)) + FFN(LN(x))

        Args:
            x: Input [batch, seq, d_model]
            attention_mask: Optional mask.

        Returns:
            Output [batch, seq, d_model]
        """
        # Shared pre-norm
        x_normed = self.pre_norm(x)

        # === Attention branch ===
        batch, seq_len, _ = x_normed.shape
        q = self.q_proj(x_normed).view(batch, seq_len, self.n_heads, self.d_kv)
        k = self.k_proj(x_normed).view(batch, seq_len, self.n_heads, self.d_kv)
        v = self.v_proj(x_normed).view(batch, seq_len, self.n_heads, self.d_kv)

        # MLA: compress KV
        kv = torch.cat([k, v], dim=-1)  # [batch, seq, n_heads, 2*d_kv]
        kv_flat = kv.reshape(batch, seq_len, -1)  # [batch, seq, n_heads*2*d_kv]
        kv_compressed = self.kv_compress(kv_flat)  # [batch, seq, mla_latent_dim]
        kv_reconstructed = self.kv_decompress(kv_compressed)
        kv_reconstructed = kv_reconstructed.view(batch, seq_len, self.n_heads, 2 * self.d_kv)
        k = kv_reconstructed[:, :, :, :self.d_kv]
        v = kv_reconstructed[:, :, :, self.d_kv:]

        # Scaled dot-product attention
        q = q.transpose(1, 2)  # [batch, n_heads, seq, d_kv]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scale = math.sqrt(self.d_kv)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).reshape(batch, seq_len, -1)
        attn_output = self.o_proj(attn_output)
        attn_output = self.attn_dropout(attn_output)

        # === FFN branch (parallel, not sequential) ===
        gate = F.silu(self.ffn_gate(x_normed))
        up = self.ffn_up(x_normed)
        ffn_output = self.ffn_down(gate * up)
        ffn_output = self.ffn_dropout(ffn_output)

        # === Combine: residual + both branches ===
        return x + attn_output + ffn_output


# ============================================================================
# 7. Gradient Communication Overlapping (PaLM 2)
# ============================================================================


class GradientOverlapScheduler:
    """
    Gradient Communication Overlapping — hide communication latency (PaLM 2).

    PaLM 2 mendemonstrasikan bahwa gradient synchronization bisa
    di-overlap dengan backward pass computation, mengurangi
    communication overhead 40-60%.

    Untuk Losion: saat menghitung gradient Jalur 1, simultan
    synchronize gradient Jalur 2; saat Jalur 3, synchronize Jalur 1.

    Ini membutuhkan dua communication stream (PyTorch CUDA streams).

    Note: Ini adalah scheduling logic. Actual overlapping membutuhkan
    distributed training setup.

    Args:
        num_jalurs: Jumlah jalur (default 3).
        overlap_strategy: Strategi overlapping ("round_robin", "flops_weighted").
    """

    def __init__(
        self,
        num_jalurs: int = 3,
        overlap_strategy: str = "round_robin",
    ) -> None:
        self.num_jalurs = num_jalurs
        self.strategy = overlap_strategy

    def get_communication_schedule(
        self,
        current_jalur: int,
    ) -> Optional[int]:
        """
        Tentukan jalur mana yang gradient-nya harus di-synchronize
        saat backward pass jalur tertentu sedang berjalan.

        Args:
            current_jalur: Jalur yang sedang di-backward (0, 1, 2).

        Returns:
            Jalur yang gradient-nya harus di-synchronize, atau None.
        """
        if self.strategy == "round_robin":
            # Simple: synchronize jalur sebelumnya
            return (current_jalur - 1) % self.num_jalurs
        elif self.strategy == "flops_weighted":
            # Synchronize jalur termurah saat backward jalur termahal
            # Jalur 2 (attention) = termahal → saat backward Jalur 2, sync Jalur 1
            if current_jalur == 2:
                return 0  # Sync Jalur 1 saat backward Jalur 2
            elif current_jalur == 0:
                return 2  # Sync Jalur 3 saat backward Jalur 1
            else:
                return 0  # Sync Jalur 1 saat backward Jalur 3
        return None

    def create_overlap_plan(self) -> List[Dict[str, Any]]:
        """
        Buat rencana overlapping untuk satu backward pass lengkap.

        Returns:
            List of scheduling steps, setiap step berisi:
            - compute_jalur: Jalur yang sedang di-compute backward
            - sync_jalur: Jalur yang gradient-nya di-synchronize
        """
        plan = []
        for i in range(self.num_jalurs):
            sync = self.get_communication_schedule(i)
            plan.append({
                "step": i,
                "compute_jalur": i,
                "sync_jalur": sync,
                "description": (
                    f"Backward Jalur {i} + Sync gradient Jalur {sync}"
                    if sync is not None
                    else f"Backward Jalur {i}"
                ),
            })
        return plan


# ============================================================================
# Memory-Efficient Backpropagation
# ============================================================================


class MemoryEfficientBackprop:
    """
    Memory-efficient backpropagation utilities.

    Menggabungkan beberapa teknik untuk mengurangi memory footprint
    selama backward pass:

    1. Gradient Checkpointing (aktif di LosionModel)
    2. Gradient Accumulation across Experts (GShard)
    3. Selective Gradient Computation (hanya compute gradient untuk
       active experts, bukan semua)

    Args:
        gradient_accumulation_steps: Langkah akumulasi gradient.
        expert_grad_accumulation: Akumulasi gradient per expert sebelum sync.
        selective_grad: Hanya compute gradient untuk active experts.
    """

    def __init__(
        self,
        gradient_accumulation_steps: int = 1,
        expert_grad_accumulation: int = 4,
        selective_grad: bool = True,
    ) -> None:
        self.grad_accum_steps = gradient_accumulation_steps
        self.expert_grad_accum = expert_grad_accumulation
        self.selective_grad = selective_grad

    def get_expert_sync_frequency(self, num_experts: int, num_devices: int) -> int:
        """
        Hitung frekuensi gradient synchronization per expert.

        GShard-style: accumulate gradients locally untuk K micro-batches
        sebelum all-reduce. K = num_experts / num_devices.

        Args:
            num_experts: Jumlah total experts.
            num_devices: Jumlah devices.

        Returns:
            Frekuensi synchronization (dalam micro-batches).
        """
        if num_devices <= 1:
            return 1  # Tidak perlu sync untuk single device
        return max(1, min(self.expert_grad_accum, num_experts // num_devices))

    def should_compute_grad(
        self,
        expert_idx: int,
        active_expert_indices: torch.Tensor,
    ) -> bool:
        """
        Apakah gradient perlu dihitung untuk expert ini?

        Selective gradient computation: hanya compute gradient untuk
        experts yang menerima token (active experts). Experts yang
        tidak menerima token tidak perlu gradient computation.

        Args:
            expert_idx: Indeks expert.
            active_expert_indices: Indeks active experts [batch*seq, top_k].

        Returns:
            True jika gradient perlu dihitung.
        """
        if not self.selective_grad:
            return True

        return (active_expert_indices == expert_idx).any()
